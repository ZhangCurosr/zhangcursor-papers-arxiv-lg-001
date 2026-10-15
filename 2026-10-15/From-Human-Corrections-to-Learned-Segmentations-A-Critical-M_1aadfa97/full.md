# From Human Corrections to Learned Segmentations: A Critical Mathematical Review of Information, Learning, and Human Efort

Jon Sporring   
Department of Computer Science,   
University of Copenhagen,   
DK-2100 Copenhagen,   
Denmark   
e-mail: sporring@di.ku.dk

October 15, 2026

## Abstract

When does a human correction become useful learning, and what establishes better segmentation at matched human efort? This critical narrative review follows feedback through its selection, learner update, and evaluation. It compares active learning, machine teaching, interactive segmentation, and sequential predictive evaluation using a common mathematical account of target information, predictive error, response predictability, and human work. An exact Bayesian update converts evaluation-label information into expected log-loss improvement; an excess-risk decomposition explains what an approximate learner can lose. Conditional information decompositions separate the roles of selected locations, labels, and displayed context. Empirical comparisons show why response protocols, update mechanisms, and efort denominators matter, distinguishing current-mask refinement from reusable learning and simulated interaction from human evidence. We identify what logs can reveal, what requires independent evaluation, and what remains model-dependent. The resulting research priorities are to test what selection reveals, whether using it improves assimilation, and whether the resulting interaction policy benefits segmentation at matched efort. These are distinct claims: predicting a teacher’s responses need not identify human intentions or establish better learning. No new acquisition algorithm or empirical superiority is claimed. keywords: Interactive segmentation, Machine teaching, Active learning, Human feedback, Information gain, Minimum description length

Reading guide The review follows a correction from selection to learner update and evaluation. A target specifies the desired label distributions; a response records what the human chooses to mark; a predictor determines the resulting segmentation error. These are distinct objects. Table 1 provides a quick reference, and boxed figures illustrate the distinctions on four pixels, beginning with Figure 1. The literature selection is described in Section 2.1. Standard derivations are in A; the qualified parametric MDL benchmark is in B. All logarithms are base two unless stated otherwise.

## 1 When do corrections become useful learning?

Corrective segmentation combines a prediction, a human judgment of that prediction, and an update. The same mark may refine the current mask, train a model for later views, or do both. RootPainter exemplifies the latter setting: corrections accumulate while a task-specific segmentation model is trained [Smith et al., 2022a, Smith et al., 2022b]. The central question is not merely how many labels have been collected, but what follows from observing them, updating a learner, and comparing the resulting outcomes:

When is corrective feedback informative, when does it become useful learning, and what does a claimed improvement per unit of human efort establish?

Existing approaches answer diferent versions of this question. Query selection, pedagogical example selection, mask refinement, and sequential compression difer both in the decisions they control and in what counts as success. A common mathematical description lets us compare these assumptions without treating every method as an approximation to a single objective.

We use progressive teaching for the coupled corrective process, not for a competing acquisition algorithm or a claim that humans teach optimally. The review is organised around three questions:

1. Which participant selects each part of the supervision, and what observation model makes that selection informative?

2. Does a method seek target information, predictive improvement, or economical use of human work, and when do these objectives agree?

3. Which assumptions connect acquired corrections to lasting learning, and what evidence is needed when those assumptions fail?

The contribution is a critical mathematical comparison of the assumptions connecting these three questions. We distinguish the evidence in a selected response from the efect of a learner update and from the resource comparison needed to call that efect eficient. These distinctions concern both current- image refinement and reusable predictors; success at one does not establish success at the other. RootPainter motivates the reusable-learning case, but does not define the review’s conclusion. A teacher-response model is useful only if it helps explain or improve a specified outcome; predicting familiar strokes is not itself evidence of better learning.

## Notation at a glance

Random objects and their realized values are distinguished explicitly where needed, for example $R _ { i }$ and $r _ { i } ,$ or $M _ { i }$ and $\mathcal { M } _ { i } .$ . All information quantities use base-two logarithms. The table previews notation defined in the main text; the boxed figures develop one four-pixel example.

Table 1: Core symbols, meanings, and relations.
<table><tr><td>Symbol</td><td>Meaning and principal relation</td></tr><tr><td colspan="2">Images, targets, and predictions</td></tr><tr><td> $J , \mathcal { X } , \mathcal { V }$ </td><td>Fixed image collection, address domain, and label alphabet;  $J : \mathcal { X }  \mathbb { R } ^ { d }$ </td></tr><tr><td> $X \sim \nu$ </td><td>Evaluation address and weights; ν is not the view-selection distribution. Random interpretation across sessions and its fixed session realization.</td></tr><tr><td> $F \sim \mu , f$ </td><td>Random target label distribution and its realization;  $P _ { f } : \mathcal { X } \to \mathcal { P } ( \mathcal { Y } )$ </td></tr><tr><td> $P _ { F } , P _ { f }$   $T _ { F } , T _ { f }$ </td><td>Random and realized evaluation-equivalence classes; members of  $T _ { f }$  agree</td></tr><tr><td></td><td>with  $P _ { f }$  ν-almost everywhere.</td></tr><tr><td> $S _ { i - 1 } , Q _ { i } , \mathcal { Q }$ </td><td>Pre-response state, display predictor  $Q _ { i } = Q ( \cdot \mid \cdot , S _ { i - 1 } )$  , and learner family.</td></tr><tr><td colspan="2">Interaction and memory</td></tr><tr><td> $\mathcal { X } _ { i } , \mathcal { V } _ { i }$ </td><td>Realized view and display; marked addresses need not fill the view.</td></tr><tr><td> $Z _ { i } , \mathcal { Z } _ { i }$ </td><td>Random pre-response memory and its realization  $\mathcal { Z } _ { i } = ( \mathcal { M } _ { i - 1 } , \mathcal { X } _ { i } , \mathcal { V } _ { i } )$ </td></tr><tr><td> $R _ { i } = ( \Xi _ { i } , H _ { i } )$ </td><td>Random response: marked support and label assignment.</td></tr><tr><td> $r _ { i } = ( \xi _ { i } , h _ { i } )$ </td><td>Realized response;  $\xi _ { i } \subseteq \mathcal { X } _ { i } , h _ { i } : \xi _ { i } \to \mathcal { Y } .$ </td></tr><tr><td> $M _ { i } , M _ { i }$ </td><td>Random post-response memory  $M _ { i } = ( Z _ { i } , R _ { i } )$  and realization  $\mathcal { M } _ { i } = ( \mathcal { Z } _ { i } , r _ { i } )$ </td></tr><tr><td> $\Lambda , \mathrm { v i s } , \Phi$ </td><td>View selector, visualization function, and learner state update.</td></tr><tr><td> $\rho _ { f }$ </td><td>Teacher-response distribution at fixed interpretation:  $R _ { i } \sim \rho _ { f } ( \cdot \mid Z _ { i } = \mathcal { Z } _ { i } )$  Source-marginal response distribution given context, and shared causal coding</td></tr><tr><td> $\rho _ { i } , \widehat { \rho } _ { i }$ </td><td>model.</td></tr><tr><td colspan="2">Beliefs, information, and effort</td></tr><tr><td> $\mathcal { C } _ { i } , w ( P \mid \mathcal { M } )$ </td><td>Toy candidate set and learner model weights; neither is the target class or</td></tr><tr><td> $\widetilde { Q } , \widehat { Q }$ </td><td>necessarily the source posterior. Uniform toy candidate-average predictor and posterior mixture used in BALD.</td></tr><tr><td> $\mathrm { C E } ( P _ { f } , Q ) , \tau ( T _ { f } , Q )$ </td><td>Cross-entropy risk and evaluation-weighted KL distortion;  ${ \mathrm { C E } } = H _ { f } ( Y \mid$   $X ) + \tau .$ </td></tr><tr><td>Gaini</td><td>Full-round target-entropy reduction, including any new context information; not necessarily predictive improvement</td></tr><tr><td> $W _ { n }$ </td><td>Random marked-location work  $\begin{array} { r } { W _ { n } = \sum _ { i = 1 } ^ { n } | \Xi _ { i } | ; } \end{array}$  repeats count, elapsed time</td></tr><tr><td> $C _ { n }$ </td><td>is not implied. Measured human cost  $\textstyle \sum _ { i = 1 } ^ { n } c _ { i }$  in a declared unit;  $C _ { n } = W _ { n }$  is the mark-count</td></tr><tr><td> $\ell _ { i } , \ell _ { \mathrm { P T } } ( n )$ </td><td>specialization. Ideal response length  $\ell _ { i } = - \log _ { 2 } \widehat { \rho _ { i } } ( R _ { i } \mid Z _ { i } )$  and cumulative expectation  $\begin{array} { r } { \ell _ { \mathrm { P T } } ( n ) = \mathbb { E } \sum _ { i = 1 } ^ { n } \ell _ { i } . } \end{array}$ </td></tr></table>

## 2 Related perspectives: who, how, and what makes it eficient?

## 2.1 Review questions and literature selection

This is a critical narrative review, not a systematic survey or meta-analysis. We select work by its observation model, learner update, or evaluation design, not by whether it favours teacher modelling. Foundational methods explain the criteria; primary segmentation studies supply empirical evidence. Architecture variants without a distinct interaction or evaluation mechanism are excluded.

The detailed studies in Table 2 span reusable learning (Crayons, RootPainter, RootPainter3D, Cellpose, XpertLoop), image-specific adaptation (BIFSeg), prompting and reuse (microscopy SAM), workflow integration (MONAI Label), response modelling (RClicks), and cost-aware acquisition (CEREALS). This contrast-based selection is not a representative statistical sample. Adjacent studies test other feedback units, contextual reuse, and cross-image adaptation; they inform scope rather than the table’s estimates.

Prior reviews provide candidate references. A bounded primary-source search through 14 October 2026 and targeted citation tracing supplement them for corrective segmentation, response modelling, and reusable learning. C records queries, inspected versions, and selection decisions. Coverage is not exhaustive or independently double-screened. Preprints are identified explicitly.

Comparisons illustrate mechanisms, not maximal efects. D records population and horizon, intervention and controls, endpoint and efort, uncertainty unit, selection rationale, and qualifying alternative results. Missing reporting is distinguished from design limitations. Heterogeneous outcomes are not pooled. Equations give qualified specializations; examples establish possibilities, not rankings.

## 2.2 Relation to existing reviews

Dudley and Kristensson organise interactive machine learning around the user, model, data, and interface, including incomplete feedback and the dificulty of translating user intent into input [Dudley and Kristensson, 2018]. Wu et al. review human involvement in data processing, model training, and system design [Wu et al., 2022]; Ren et al. survey deep active-learning strategies and their practical challenges [Ren et al., 2021]. For medical image segmentation, Marinov et al. distinguish interactive learning, interactive inference, and interactive online learning, and examine simulated-user realism, missing baselines, and inadequate user-centred evaluation [Marinov et al., 2024]. These reviews establish both the human–learner loop and the need to scrutinize its evaluation.

Our narrower comparison asks which components of a spatial response are selected by whom, what observation model makes that selection informative, and whether an improvement concerns the current mask or a reusable learner. It connects these distinctions to target information, predictive distortion, response predictability, and human work. The additional contribution is a formal account of the assumptions connecting observation, update, and efort: two methods with similar interfaces can optimize diferent quantities, while methods from diferent applications can share an observation or cost assumption. The common notation is useful only insofar as it exposes these diferences and their consequences for evaluation.

Four transfer tests organise the synthesis. Does a requested-label model describe a human-selected support? Does better current-mask quality show learning for later images? Does a more predictable response indicate more useful supervision? Does a smaller mark count establish less human efort? The answer to each depends on an explicit observation, update, or measurement assumption, rather than on the method’s name. Sections 5, 6, and 8 make those assumptions mathematical; Section 9 turns them into criteria for interpreting evidence.

## 2.3 Who selects supervision, and how do the participants interact?

In active learning, the learner selects a query. In machine teaching, the teacher selects supervision with a learner in mind. Corrective segmentation can combine both: the system proposes a view, the teacher chooses where to mark, and either participant may initiate navigation. This allocation of choices is more informative than classifying the entire application as only active learning or only teaching [Settles, 2009, Zhu et al., 2018].

Iterative machine teaching explicitly adapts examples to learner state [Liu et al., 2017]; cross-space work relaxes shared-representation assumptions [Liu et al., 2018]. Interactive machine learning studies the human– learner loop more generally [Amershi et al., 2014], while coactive learning studies feedback that improves a structured prediction without requiring an optimal correction [Shivaswamy and Joachims, 2015]. Its regret analysis assumes informative feedback, with variants allowing slack or informativeness in expectation; an arbitrary correction need not satisfy these assumptions. Communicative learning places mutual modelling within a broader account of learning as information exchange [Yuan and Zhu, 2023].

Pedagogical reasoning makes the selection mechanism explicit: Shafto et al. model teachers choosing examples for a learner and learners interpreting that choice [Shafto et al., 2014]. Their concept-learning examples include both point locations and labels. Iterative teacher-aware learning transfers this idea to parameter learning through a model of why a teacher selected an example [Yuan et al., 2021]. Its update compares the chosen example with labelled alternatives under a heuristic teacher-selection model. Such alternatives are generally unlabelled in sparse corrective segmentation. Moreover, a mismatch between assumed and actual teaching behaviour can remove the benefit; teacher-awareness is not automatically better than treating examples as ordinary samples.

Consequently, neither having a history nor inferring information from a teacher’s choice is a novelty claim by itself. The application-specific question is what is gained by jointly representing displayed predictions, selected spatial supports, label values, learner updates, and human work.

## 2.4 What changes when a segmentation is corrected?

Classical interactive segmentation uses sparse input to constrain an image segmentation objective [Boykov and Jolly, 2001, Rother et al., 2004]. Modern refinement systems often evaluate quality after a given number of clicks, or clicks required to reach a threshold [Benenson et al., 2019, Chen et al., 2022, Liu et al., 2023]. This difers from training a reusable pixel classifier: already in Crayons, users painted examples, inspected

classifications, and corrected errors to train a model [Fails and Olsen, 2003]. Trainable Weka Segmentation and ilastik likewise learn classifiers from interactive annotations for application to further data [Arganda-Carreras et al., 2017, Berg et al., 2019]. RootPainter belongs to this established corrective-learning lineage, with a deep learner and sparse corrective supervision [Smith et al., 2022a, Smith et al., 2022b].

There are intermediate and hybrid regimes. BIFSeg adapts network parameters to the particular image being segmented, using a bounding box and scribbles; updating parameters therefore need not imply learning for later images [Wang et al., 2018]. Cellpose 2.0 alternates correcting predicted masks, retraining, and processing subsequent images [Pachitariu and Stringer, 2022]. Its corrected masks provide diferent supervision from sparse labels whose unmarked surroundings remain unknown. MONAI Label integrates interactive annotation, model training, and active-learning services [Diaz-Pinto et al., 2024]; supporting these components is distinct from establishing the benefit of a particular acquisition policy.

Promptable models add another distinction. Segment Anything conditions a pretrained model on spatial prompts to produce a mask [Kirillov et al., 2023]. Accumulating prompts can improve that mask without updating the model parameters. Segment Anything for Microscopy combines interactive annotation with microscopy-specific models and fine-tuning, and evaluates annotation time and quality with users [Archit et al., 2025]. Prompt state, image-specific adaptation, and reusable-model training should therefore be recorded separately rather than identified by the presence of an interactive interface.

Reuse need not change weights: MultiverSeg retains completed image–mask pairs as context, whereas You Point, I Learn updates parameters during and after interaction [Wong et al., 2025, Xu et al., 2026]. The HiTTA preprint also carries corrected-mask updates forward [Hu et al., 2024]. All three evaluate simulated interactions; HiTTA uses existing annotators’ masks to simulate preferences and evaluates the better of two prediction heads.

Live studies address diferent endpoints. MedSAM times initialization, inference, and refinement for two radiologists completing ten adrenal-tumour cases, with fixed task order [Ma et al., 2024]; it does not test reusable learning. SPA adapts a latent distribution from preferences among candidate masks and tests cross-image transfer of that state; its five-participant study reports shorter interactions than MedSAM [Zhu et al., 2025a]. Preference-state reuse difers from modelling natural spatial-support selection.

Corrections and reuse coexist in two preprints: Dendukuri et al.’s timed surgical-video study, where SAM2 assistance outpaces iterative training [Dendukuri et al., 2026], and Shaeri et al.’s critic interface with fine-tuning and cross-image propagation [Shaeri et al., 2025]. ExiL combines simulated refinement trajectories, expert-approved updates, and single-expert timing [Tavangar et al., 2026]. These precedents establish history-based correction without estimating natural human selection probabilities.

Human responses versus simulated clicks The response protocol specifies which errors are noticed, corrected, or ignored. RClicks collects real click placements and learns a clickability model for more realistic segmentation benchmarking [Antonov et al., 2024]. Its data include later-click tasks constructed from selected model-error masks; these are not unrestricted longitudinal teaching sessions. It establishes a direct precedent for modelling spatial response support, but does not establish that using such a model improves a reusable learner. This separates three claims: realistic response prediction, robust mask refinement, and more efective corrective training. SegAgent provides a complementary caution: its human-like annotation trajectories are generated from reference-mask errors, not recorded human teaching sessions [Zhu et al., 2025b]. Imitation of a click generator does not by itself validate the generator as a human-response model.

## 2.5 What does the empirical evidence establish?

Table 2 compares findings within their protocols, not scores across tasks. SD denotes standard deviation and SE standard error; neither is a confidence interval. Counts and figure/table locators refer to the cited experiments. Dice, downstream measurement correlations, and mean segmentation accuracy (mSA) remain separate endpoints: none is converted to KL distortion.

Table 2: Study-level evidence: selection, assimilation, and human efort. Prompt refinement, current-image adaptation, and reusable-model training are distinct update scopes. “Time” and counts are not interchangeable efort measures.
<table><tr><td>Study</td><td>Selection, feedback, and update</td><td>Finding, uncertainty, and scope</td></tr><tr><td>Crayons [Fails and Olsen, 2003]</td><td>Human-painted examples and corrections train a reusable pixel classifier; the comparison changes the fitting algorithm, not a query policy.</td><td>Ten users, counterbalanced order: skin-classification time to 97.5% reference agreement averaged 3:06/3:51 minutes for the stronger-generalizing algorithm versus 10:35/10:26 for the faster-fitting alternative in the two order groups (Tables 1–2). No intervals or tests are reported. Slower individual updates</td></tr><tr><td>RootPainter [Smith et al., 2022a]</td><td>Humans select foreground/background marks after viewing predictions; sparse corrections accumulate in reusable-model training. Initial clear examples precede corrective annotation.</td><td>completion. Two author-annotators, three datasets: time-matched corrective versus dense annotation gave held-out measurement  $R ^ { 2 }$  of 0.80 (SE 0.04) versus 0.75 (SE 0.07), with no significant strategy effect at the 0.05 threshold (Table 3). Corrective annotation came first. The comparison establishes neither superiority nor equivalence.</td></tr><tr><td>RootPainter3D [Smith et al., 2022b]</td><td>A physician selects spatial corrections; a model is trained throughout annotation of successive CT volumes. Both completed contours and the reusable predictor change.</td><td>One physician, 933 hearts: the last ten averaged 122 seconds with assistance versus 421 manually (Fig. 5); no timing interval is reported. In 20 paired contour comparisons, corrected-output Dice was 0.952 (SD 0.007) against manual contours (Table 1). Timings include inspection but exclude inactivity of at least 20 seconds; the design does not isolate selection modelling.</td></tr><tr><td>BIFSeg [Wang et al., 2018]</td><td>Human bounding boxes and scribbles guide segmentation and image-specific network adaptation. Parameters change, but the intervention is targeted to the current image.</td><td>Two clinical users. With identical initial predictions and one-round scribbles, tumour-core Dice was 87.49% versus 85.93% for CRF, and whole-tumour Dice 88.11% versus 85.18% (p &lt; 0.05, paired tests; Table VI). This is evidence of better use of the same corrections for current-case refinement, not accumulated</td></tr><tr><td>Cellpose 2.0 [Pachitariu and Stringer, øahce masks; accumulated masks</td><td>A human corrects and completes retrain a reusable model before the next image. The design compares online/offline annotation and pretrained/from-scratch initialization.</td><td>learning across cases. One annotator, two categories: online/pretrained annotation used 167 versus 663 and 158 versus 897 manually drawn objects compared with offline annotation (Fig. 5). Test performance approached within-annotator agreement with 100–200 objects; cost-ratio intervals are not reported. Pretraining and repeated fitting</td></tr><tr><td>MONAI Label [Diaz-Pinto et al., 2024] selection and human clicks/scribbles,</td><td>The framework supports system query with mask refinement and reusable-model retraining. The reported workflow is not one fixed acquisition method.</td><td>time. DeepGrow validation Dice rose from 0.891 to 0.967 (2D) and 0.730 to 0.959 (3D) across annotation stages (Table 2), without uncertainty intervals. These are workflow demonstrations, not controlled acquisition-policy comparisons. Inconsistent validation counts and timing totals in the source prevent a precise cost-effect</td></tr><tr><td>RClicks [Antonov et al., 2024]</td><td>Humans select click locations for specified targets/error displays. A learned response-location model supplies simulated clicks for evaluating current-mask refinement, not within-session reusable-model training.</td><td>475,544 real clicks support the response model. Its multi-click simulations require approximately 5–29% more clicks than conventional simulated clicking to reach 90% IoU, capped at 20 clicks (Table 4). Reported SD measures click-group variability, not uncertainty in measured human time. This tests benchmarking sensitivity, not</td></tr><tr><td>Microscopy SAM [Archit et al., 2025]</td><td>Human prompts refine masks; dataset-specific fine-tuning is evaluated separately from interactive prompting. Automatic predictions can then serve further annotation.</td><td>reusable teaching. Five organoid annotators: fine-tuned µSAM and CellPose both averaged 0.7 seconds/object. Annotation mSA was 0.72 versus 0.76, but test mSA was  $0 . 5 1 \pm 0 . 0 2$  versus 0.42 ± 0.05 (SD across annotators; Fig. 6a). Point estimates for completion and later generalization rank differently; rates do not measure complete</td></tr><tr><td>CEREALS</td><td>The learner chooses image regions using [Mackowiak et al., 2018] uncertainty and predicted annotation cost; an oracle reveals existing labels. Segmentation and cost models are refitted after acquisition.</td><td>training-plus-human cost. Five simulated Cityscapes runs: cost-aware selection reached 95% of full-data mIoU at 17.07% of recorded polygon-click cost, versus 33.76% for entropy-only selection of 128 × 128 regions (Figs. 4–5). Threshold-cost intervals are not reported. This supports cost-aware acquisition under the robot oracle, not a</td></tr><tr><td>XpertLoop [Jaiswal et al., 2026]</td><td>Experts correct presegmentations for iterative retraining; random and active case selection are compared. Timing compares manual annotation with correction of optimized-model predictions.</td><td>measured human-time reduction. One annotator per timed task: maximum time savings were 90.3% for kidneys and 48.2% for tumours  $( p < 0 . 0 0 1$  and  $p = 0 . 0 0 3 ;$  Fig. 5). Estimated tumour cost savings were $5.63/examination (95% CI -7.26 to 26.09), using wage/time modelling rather than user-population uncertainty. This is not a matched end-to-end training-budget comparison.</td></tr></table>

Three links emerge. Assimilation: BIFSeg holds corrections fixed while changing their use; Crayons trades fitting speed against generalization. These change the update Φ, not merely the information received. Equation (24) provides a log-loss account of this distinction, not an information-theoretic reinterpretation of their Dice or completion-time efects.

Selection: RClicks tests a human-data-based response simulator, whereas CEREALS changes acquisition Λ under a specified cost proxy. Neither tests the complete chain from predicting natural support choices to better reusable learning. Transferring simulated gains requires a representative response model and a validated efort denominator.

Eficiency: RootPainter’s matched-time estimate is inconclusive; RootPainter3D and XpertLoop show completion savings under specific training histories. Cellpose’s pretraining comparison and microscopy SAM’s diferent completion/test rankings show why initialization, update horizon, and outcome must be held explicit. Late or optimized-model savings are not whole-session savings. These distinctions transfer across losses; the KL identities do not convert the studies’ endpoints into distortion.

## 2.6 Which labels should the learner request?

For exposition, let a learner assign weights $w ( P \mid { \mathcal { M } } )$ to a finite collection C of candidate label distributions after history M. Its posterior predictive distribution is

$$
{ \widehat { Q } } ( y \mid x , { \mathcal { M } } ) = \sum _ { P \in { \mathcal { C } } } w ( P \mid { \mathcal { M } } ) P ( y \mid x ) , \qquad w \geq 0 , \quad \sum _ { P } w ( P \mid { \mathcal { M } } ) = 1 .\tag{1}
$$

Sums become integrals for continuous model posteriors. This is an explicit modelling choice, not an enumeration of every possible segmentation function. Approximate posteriors or ensembles may make the calculation tractable but can miss plausible targets.

Predictive entropy and BALD Predictive-entropy acquisition ranks x by $H ( \widehat { Q } ( \cdot \mid x , \mathcal { M } ) )$ . BALD instead uses the label’s mutual information with the candidate model [Houlsby et al., 2011, Houlsby et al., 2012]:

$$
\mathrm { B A L D } ( x ) = H ( \widehat { Q } ( \cdot \mid x , \mathcal { M } ) ) - \sum _ { P } w ( P \mid \mathcal { M } ) H ( P ( \cdot \mid x ) ) .\tag{2}
$$

It subtracts label variability shared by individual candidate models. If every candidate predicts $( 1 / 2 , 1 / 2 )$ predictive entropy is one bit but BALD is zero. If all candidates are deterministic, their entropies vanish and the criteria coincide. BALD uses a posterior predictive distribution for acquisition; it need not prescribe the application’s deployed predictor. The distinction between model uncertainty and within-model labe variability is also important in segmentation uncertainty modelling [Kendall and Gal, 2017].

BatchBALD and diversity For a batch $B ,$ , write $P _ { B }$ for a candidate’s joint label distribution and $\begin{array} { r } { \widehat { Q } _ { B } = \sum _ { P } w ( P \mid \mathcal { M } ) P _ { B } } \end{array}$ . BatchBALD uses

$$
\mathrm { B a t c h B A L D } ( B ) = H ( \widehat { Q } _ { B } ) - \sum _ { P } w ( P \mid \mathcal { M } ) H ( P _ { B } ) .\tag{3}
$$

Joint scoring accounts for redundant information that a sum of individual scores misses [Kirsch et al., 2019]. The published construction uses conditional independence of labels given model parameters and greedy batch selection. Marginal label distributions alone do not determine $P _ { B }$ without such an assumption or an explicit joint model. Nor does scoring all pixels in a view model which subset a teacher will actually annotate. Core-set acquisition instead seeks geometric batch coverage in a learned representation [Sener and Savarese, 2018]. Its image-classification evidence makes it a useful diversity baseline, not a guarantee of coverage of rare segmentation errors.

Expected improvement and variance reduction Cohn et al. analyse squared prediction error, use a negligible-bias assumption, and acquire labels to reduce expected prediction variance [Cohn et al., 1996]. In a discrete evaluation domain, their idea can be written schematically as

$$
x _ { \mathrm { n e x t } } \in \arg \operatorname* { m i n } _ { x } \mathbb { E } _ { Y \sim \widehat { Q } ( \cdot | x , \mathcal { M } ) } \left[ \sum _ { z } \nu ( z ) v _ { x , Y } ( z ) \right] ,\tag{4}
$$

where $v _ { x , Y } ( z )$ is the estimated variance after fitting the additional example. This is a precedent for selecting supervision by its efect on the learner, but is not a general guarantee of reduced KL distortion. Related segmentation methods select for expected change or spatial uncertainty [Vezhnevets et al., 2012, Top et al., 2011].

## 2.7 Who should annotate, and at what cost?

Yan et al. jointly consider uncertain instances and input-dependent annotator reliability [Yan et al., 2011]. Their selection objective can be expressed, with renamed coeficients, as

$$
\operatorname* { m i n } _ { x , \lambda } \left\{ \eta \left[ \log \mathrm { i t } \widehat { Q } ( 1 \mid x ) \right] ^ { 2 } + \sum _ { a } \lambda _ { a } e _ { a } ( x ) \right\} , \qquad \lambda _ { a } \geq 0 , \quad \sum _ { a } \lambda _ { a } = 1 .\tag{5}
$$

Here $x$ is their feature-space query, $e _ { a }$ is an annotator noise score, and $\eta$ balances the terms; the continuous proposal is mapped to the available pool. The natural-log logit convention can be absorbed into the coeficient if another base is used. This is an annotator-selection model, not a model of human-selected spatial correction support.

CEREALS combines informative regions with estimated annotation cost in semantic segmentation [Mackowiak et al., 2018]. Haertel et al. study return-on-investment acquisition and conditions relating local benefit/cost selection to performance across a budget, with experiments in part-of-speech annotation [Haertel et al., 2015]. Its relevance is the cost–benefit structure, not a claim about pixelwise KL per mark. A ratio criterion is not in general an optimal multi-step policy.

XpertLoop combines expert correction timing with reusable-model training and acquisition comparisons [Jaiswal et al., 2026]. In its reported organ tasks, size–entropy selection reaches strong performance in fewer loops than random selection. Plain entropy can underperform random selection, and suitability varies by dataset. Size–entropy adds inference over unlabelled cases, while stopping uses test-set performance; loop counts therefore establish neither equal total costs nor independently assessed gains.

Benenson et al.’s human annotation study makes work-proxy limitations concrete [Benenson et al., 2019, Sections 4–5]. Under a three-round protocol allowing up to four clicks per round, zero-click judgments took about four seconds, and four-click rounds were faster than three-click rounds. Later rounds took longer, consistent with smaller errors becoming harder to find; a ranker using interaction-history features predicted completed-mask quality. These findings motivate accounting for inspection and changing correction behaviour (Sections 6 and 9), without establishing universal teaching stages or improved reusable learning. The headline polygon comparison uses an estimated historical timing denominator, not a matched-user trial. Alongside Cellpose’s correction counts and microscopy SAM’s user timings [Pachitariu and Stringer, 2022, Archit et al., 2025], this shows why efort units and available actions must be specified.

## 2.8 What else must a correction model distinguish?

Probabilistic segmentation can represent multiple masks and spatially correlated uncertainty [Kohl et al., 2018, Monteiro et al., 2020]. These models represent distributions over complete label maps, not just their pixel marginals. Zhang et al. jointly estimate a latent consensus-label distribution and image-/pixeldependent annotator confusion matrices under explicit conditional-independence and regularization assumptions [Zhang et al., 2023]. This is not an assumption-free separation of valid alternative interpretations from annotation errors, nor a model of which regions annotators choose to mark. Error-tolerant segmentation separately addresses incorrect annotations [Wang et al., 2020]. These distinctions matter even when the eventual annotation is one hard label. Human feedback may also depend on the learner’s behaviour [MacGlashan et al., 2017].

Selective-labels research cautions that the cases on which labels are observed need not represent the evaluation population [Lakkaraju et al., 2017]. For correction logs, exposure, attention, error detection, and marking are distinct filters. Rubin’s missing-data analysis gives conditions under which an observation mechanism can be ignored for inference [Rubin, 1976]. Positive–unlabelled learning distinguishes selection assumptions and the information needed to identify class probabilities [Bekker and Davis, 2020]. The closest corrective analogy is a detected-and-marked error versus an unmarked location, not foreground versus background. Confirmatory or erroneous marks require a richer model.

In active learning, Farquhar et al. distinguish acquisition-induced estimation bias from overfitting and show that removing the former need not improve deep-model training [Farquhar et al., 2021]. Their estimators use the acquisition probabilities; displaying an image does not reveal the human’s attention or marking probabilities. These connections explain why non-correction is not automatically acceptance and why an adaptive log does not by itself supply an unbiased evaluation sample.

Finally, MDL and prequential coding connect prediction with description length [Rissanen, 1978, Grünwald, 2007, Grünwald and Roos, 2020]. Neural prequential coding already evaluates how a learning procedure predicts later labels from earlier ones [Blier and Ollivier, 2018]. Dawid’s prequential perspective evaluates probability forecasts against subsequently observed outcomes [Dawid, 1984]; proper scoring rules formalize the quality of such forecasts [Gneiting and Raftery, 2007]. Sections 7 and 8 ask what changes when the observations are adaptive spatial responses rather than a predetermined sequence of labels.

## 2.9 Teaching order and adaptation on both sides

Curriculum learning studies how the selection and order of training examples afect learning [Bengio et al., 2009]. Torpey et al. show that image order changes Dice and simulated correction counts when completed predictions become context for later images [Torpey et al., 2026]. Across starting images, uncertainty-based ordering does not consistently beat random ordering; its best-start advantage uses retrospective selection.

Order sensitivity therefore does not by itself identify an efective selection policy or establish human-time savings. Continual-learning research organises mechanisms for retaining previous learning, including replay [De Lange et al., 2022]. Its changing-task classification benchmarks are not the same as a fixed-target correc tive session, but they identify learner-side mechanisms that can make identical supervision produce diferent outcomes when presented in diferent orders.

Human teaching is not necessarily a curriculum optimized for the learner. Cakmak and Thomaz study natural teaching and learner-specific guidance [Cakmak and Thomaz, 2014]; Fiebrink et al. observe users modifying training strategies and sometimes task goals after evaluating model behaviour [Fiebrink et al., 2011]. The latter study concerns interactive gesture/music systems rather than segmentation. Together these works motivate modelling strategy and adaptation, but do not establish a universal sequence of teaching stages or validate a particular RootPainter protocol.

## 2.10 Synthesis: which part of the corrective loop is modelled?

Table 3 compares mechanisms, not benchmark rankings. A method can address several rows; for example, a cost model can be combined with a disagreement-based query rule. The key distinction is which choices are treated as controlled actions, which as observations, and which are left outside the model.

Table 3: Representative approaches and the assumptions needed to transfer their interpretation to corrective learning.
<table><tr><td>Approach</td><td>Decision and observation</td><td>Issue in the corrective loop</td></tr><tr><td>BALD [Houlsby et al., 2011]</td><td>Learner requests a label; beliefs determine its expected informativeness.</td><td>A teacher may choose a different support within the displayed view.</td></tr><tr><td>BatchBALD [Kirsch et al., 2019]</td><td>Learner requests a batch; joint label information accounts for</td><td>A jointly informative batch is not necessarily the subset the human</td></tr><tr><td>Variance reduction [Cohn et al., 1996]</td><td>redundancy. Query scored through its expected effect on a fitted predictor.</td><td>will label. Improvement depends on the loss, model assumptions, and actual</td></tr><tr><td>Annotator-aware AL [Yan et al., 2011]</td><td>Learner selects an instance and an annotator using reliability estimates.</td><td>update. Reliability and the human&#x27;s choice of what to correct are distinct</td></tr><tr><td>Iterative teaching and teacher-aware learning [Liu et al., 2017,</td><td>Teacher selects supervision; learner may interpret its selection mechanism.</td><td>mechanisms. Spatial support, displayed prediction, and human effort require an explicit specialization.</td></tr><tr><td>Yuan et al., 2021] Interactive segmentation [Chen et al., 2022, Smith et al., 2022a]</td><td>Human or simulated teacher supplies spatial feedback to a mask or reusable model. A learner predicts successive labels</td><td>Per-image refinement and learning for subsequent views require different evaluation.</td></tr></table>

This comparison motivates three distinctions developed next: a target is not an individual response; information in a response is not the improvement produced by a learner; and predictable supervision is not necessarily eficient teaching. The formalism below supplies a shared language for these distinctions, rather than replacing the surveyed algorithms with one new method. Section 9 returns to the literature through an objective comparison and the open problems exposed by these assumptions.

## 3 Targets, predictions, and the corrective interaction

## 3.1 A fixed collection and a distribution-valued target

Fix an image collection $J : \mathcal { X }  \mathbb { R } ^ { d }$ , a finite address domain $x ,$ and a finite label alphabet Y. An address can be an image–pixel pair $x = ( j , p )$ . Evaluation weights $\nu ( x ) \geq 0$ satisfy $\textstyle \sum _ { x } \nu ( x ) = 1$ . Writing $X \sim \nu$ represents evaluation averaging; it does not imply that the coordinates themselves move or that views are sampled from $\nu .$

A teacher interpretation $f$ specifies the label distribution

$$
P _ { f } : \mathcal { X } \longrightarrow \mathcal { P } ( \mathcal { Y } ) , \qquad \ d \ b { x } \longmapsto P _ { f } ( \cdot \mid \ b { x } ) .\tag{6}
$$

This is what the teacher wants the learner to approximate. A deterministic segmentation is the special case of a point mass at every address. More generally, $P _ { f }$ may retain ambiguity even when f is fixed.

To discuss information, consider an ensemble of sessions with $F \sim \mu .$ . Since the image collection is fixed, $\mu$ is already conditional on that collection; random image collections instead require a conditional source distribution $\mu ( \cdot \mid J )$ . The random distribution $P _ { F }$ is the image of $F$ under $f \mapsto P _ { f }$ , not another independently chosen object. For a hypothetical evaluation draw, whether $F$ is discrete or continuous,

$$
P ( X = x , Y = y \mid F = f ) = \nu ( x ) P _ { f } ( y \mid x ) .\tag{7}
$$

Here $X$ is chosen independently for evaluation. Actual correction addresses are chosen by the interaction and generally do not have distribution $\nu .$

The source distribution $\mu ,$ the target $P _ { f }$ , the learner’s predictive distribution $Q _ { i }$ , and its beliefs over candidate models are diferent objects. In particular, the learner need not know $\mu$ or $P _ { f }$

Scope of the mathematical comparison The formal core concerns pointwise prediction on a fixed collection; it is not a common probabilistic model for every application reviewed. Table 4 makes the required extensions explicit. Random view selection within the collection is already represented and does not require an unseen-image model.

Table 4: Scope of the formal core and limits of transfer.
<table><tr><td>Setting or claim</td><td>Representation or additional requirement</td></tr><tr><td>Fixed-pool additive log loss</td><td>Directly represented by  $P _ { f } , \nu ,$  and the learner&#x27;s predictions</td></tr><tr><td>Joint-mask uncertainty;</td><td>Pixel marginals do not determine the joint mask distribution; use a joint</td></tr><tr><td>topology or expected Dice/IoU</td><td>target and a suitable loss.</td></tr><tr><td>Generalization to unseen images</td><td>Specify an image population and an independent evaluation protocol; random views of the fixed pool are insufficient.</td></tr><tr><td>Continuous target families</td><td>Mutual information can be defined through joint/product KL; finite discrete entropy differences require additional restrictions.</td></tr><tr><td>Changing human interpretation</td><td>Introduce target dynamics and the evaluation reference; asynchronous training alone does not make the target change.</td></tr></table>

## 3.2 Pre-response and post-response memory

Let $S _ { i - 1 }$ be the learner state after the first i − 1 responses, and let

$$
Q _ { i } ( \cdot \mid x ) = Q ( \cdot \mid x , S _ { i - 1 } ) .\tag{8}
$$

Thus $Q _ { i }$ generates the round-i display; $Q _ { i + 1 }$ is the readout after response i. This convention avoids confusing a round index with an optimiser iteration. A general state is not necessarily a set of models; it may include weights, optimiser variables, pending updates, and a displayed checkpoint. In the finite example of Figure 1, it is a candidate set.

Use $M _ { i }$ for random post-response memory and $\mathcal { M } _ { i }$ for its realized value. Before the response, the system selects a view $\mathcal { X } _ { i }$ and displays $\nu _ { i }$ . Its selector Λ may depend on $Q _ { i }$ , past memory, and state; random selection uses protocol randomness. The visualization function is deterministic conditional on its inputs, $\mathrm { e . g . }$ $\mathcal { V } _ { i } = \operatorname { v i s } ( Q _ { i } , \mathcal { X } _ { i } )$ . The view record identifies the actual checkpoint and any display settings relevant to the teacher. Define nested records

$$
\mathcal { M } _ { 0 } = \emptyset , \qquad \mathcal { Z } _ { i } = ( \mathcal { M } _ { i - 1 } , \mathcal { X } _ { i } , \mathcal { V } _ { i } ) , \qquad \mathcal { M } _ { i } = ( \mathcal { Z } _ { i } , r _ { i } ) .\tag{9}
$$

Their random counterparts satisfy $M _ { i } = ( Z _ { i } , R _ { i } )$ . Nesting preserves order, repetitions, and the distinction between what was displayed and what was corrected. It is not an unordered union of annotated pixels.

Let $R _ { i } = ( \Xi _ { i } , H _ { i } )$ denote a random spatial response, with realization

$$
r _ { i } = ( \xi _ { i } , h _ { i } ) , \qquad \xi _ { i } \subseteq \mathscr { X } _ { i } , \qquad h _ { i } : \xi _ { i } \to \mathscr { V } .\tag{10}
$$

An empty response is permitted and must be recorded if it is to be analysed. In one concrete logging convention, an explicit submit commits a batch of label assignments under an unchanged view/checkpoint; “skip” commits the empty batch. Reassigning an address already in the current batch first commits that batch and starts another. Record timestamps separately; erasures and navigation need their own action symbols, not background labels. The baseline in Figure 7 uses only this paint/skip alphabet. The teacher-response kernel is

$$
R _ { i } \sim \rho _ { f } ( \cdot \mid Z _ { i } = \mathcal { Z } _ { i } ) .\tag{11}
$$

Dependence on round number can be recovered from memory. Fixed teacher characteristics can be included in the kernel specification; unobserved behavioural variation can be marginalized. Unlike $P _ { f } ,$ this distribution describes where and how the teacher responds to a display, not just a label at a fixed address. Its labels need not be independent draws from $P _ { f } \colon$ strategic selection and annotation noise must be specified separately. One contradictory label does not eliminate a distributional target whose response model assigns it positive probability.

The learner then updates its state using Φ, possibly with independent training randomness. In an asynchronous implementation, $Q _ { i + 1 }$ must refer to a declared next checkpoint or a declared terminal readout, not necessarily an immediately converged fit of every received correction. This afects assimilation and delay, without by itself changing the target $f .$

Mixed initiative and what must be logged The system-initiative model chooses the view before the teacher’s response. If the teacher navigates or overrides the view, that action may itself convey target information. It must either be logged and accounted for as an earlier teacher message in the causal sequence, or remain as additional information in $Z _ { i } .$ . Section 8 keeps that term explicit. A response contains its marked addresses, not necessarily the whole displayed view.

Figure 1 instantiates the interaction on two images with two pixels each, making the first correction and learner update explicit.

## 4 Evaluation-equivalence and predictive distortion

Define

$$
T _ { f } = \{ P _ { g } : P _ { g } ( \cdot \mid x ) = P _ { f } ( \cdot \mid x ) \mathrm { { ~ f o r ~ } } \nu \mathrm { { - a l m o s t ~ e v e r y } ~ } x \} .\tag{13}
$$

$T _ { f }$ is a single equivalence class; $T _ { F }$ is a random class induced by F. Neither is the learner’s current candidate set. Changing ν can change the target class; changing the learner family does not. Diferent parameters yielding the same conditional distributions do not represent diferent evaluation targets.

Log loss is a strictly proper predictive loss: in expectation it is minimized by the data-generating label distribution, where the relevant scores are well-defined [Gneiting and Raftery, 2007]. For log loss, the

## Setup and first correction

Take two images with two addresses each, $\mathcal { V } = \{ 0 , 1 \}$ , and $\mathcal { X } = \{ ( a , 1 ) , ( a , 2 ) , ( b , 1 ) , ( b , 2 ) \}$ . The shared intensities and target labels are

<table><tr><td>Address</td><td> $( a , 1 )$   $( a , 2 )$ </td><td>(b,1)</td><td>(b,2)</td></tr><tr><td> $J ( x )$ </td><td>0.2 0.7</td><td>0.4</td><td>0.9</td></tr><tr><td> $P _ { f } ( 1 \mid x )$ </td><td>0 0</td><td>0</td><td>1</td></tr></table>

Let $P _ { 0 } , \ldots , P _ { 1 5 }$ denote all deterministic binary label vectors, numbered in the displayed address order. Thus $P _ { f } = P _ { 1 }$ . For information calculations, assume the source is uniform over these 16 targets. Set $S _ { 0 } = \mathcal { C } _ { 0 } = \{ P _ { 0 } , . . . , P _ { 1 5 } \}$ , and update by removing candidates inconsistent with faithful noiseless labels. Address choices in this example do not encode extra target information: they are independent of the unknown labels conditional on past history and the view. This assumption is what makes uniform beliefs over surviving candidates legitimate.

The system first displays image a. A predictor can give both labels probability $1 / 2$ , displayed as foreground under a specified threshold tie rule. The teacher marks $r _ { 1 } = \{ ( a , 1 ) \mapsto 0 \}$ . The record becomes $\mathcal { M } _ { 1 } = \big ( ( \emptyset , \mathcal { X } _ { 1 } , \mathcal { V } _ { 1 } \big ) , r _ { 1 } \big )$ and $\mathcal { C } _ { 1 } = \{ P _ { 0 } , \ldots , P _ { 7 } \}$ . One possible learner is the average

$$
\widetilde Q _ { i + 1 } ( y \mid x ) = \frac { 1 } { | { \mathcal C } _ { i } | } \sum _ { P \in { \mathcal C } _ { i } } P ( y \mid x ) .\tag{12}
$$

It changes the foreground probabilities from $( 1 / 2 , 1 / 2 , 1 / 2 , 1 / 2 ) \mathrm { ~ t o ~ } ( 0 , 1 / 2 , 1 / 2 , 1 / 2 )$ . The candidate set is an explicit state only in this small example; the framework does not require enumerating all segmentations.

Figure 1: A four-pixel interaction with a uniform source over 16 deterministic targets and one faithfu correction.

fixed-target cross-entropy risk decomposes as

$$
\mathrm { C E } ( P _ { f } , Q ) = - \sum _ { x } \nu ( x ) \sum _ { y } P _ { f } ( y \mid x ) \log _ { 2 } Q ( y \mid x )  \\  = \underbrace { - \sum _ { x } \nu ( x ) \sum _ { y } P _ { f } ( y \mid x ) \log _ { 2 } P _ { f } ( y \mid x ) } _ { H _ { f } ( Y \mid X ) } + \underbrace { \sum _ { x } \nu ( x ) D _ { \mathrm { K L } } ( P _ { f } ( \cdot \mid x ) \| Q ( \cdot \mid x ) ) } _ { \tau ( T _ { f } , Q ) } .\tag{14}
$$

The algebra is given in A.1. The first term is the label-variability floor for that fixed target. The second is predictive distortion, measured in bits per evaluation draw. We use 0 log 0 = 0 and assign infinite loss when $P _ { f } ( y \mid x ) > 0$ but $Q ( y \mid x ) = 0$ at a positively weighted address. Finite-risk calculations assume positivity on the target’s support. Distortion is well-defined on $T _ { f }$ because its members agree wherever ν assigns weight. For a learner family Q, the approximation floor is

$$
\tau _ { \mathcal { Q } } ^ { \mathrm { i n f } } ( T _ { f } ) = \operatorname* { i n f } _ { { Q \in \mathcal { Q } } } \tau ( T _ { f } , Q ) .\tag{15}
$$

It is distinct from $H _ { f } ( Y \mid X )$ and from optimisation or finite-data error. If $P _ { f } \notin \mathcal { Q }$ , the infimum need not be attained and can be positive (or infinite). More labels do not automatically make an arbitrary update reduce distortion.

Figure 2 applies these definitions to the first correction in Figure 1. Changing only the evaluation weights changes both the equivalence class and the measured improvement.

## Evaluation-equivalence and distortion

In the four-pixel example, the label-variability floor is zero. With uniform $\nu , T _ { f } = \{ P _ { 1 } \}$ and distortion falls from 1 to $3 / 4$ after $r _ { 1 }$ . If each address in image a has weight $1 / 2$ and image b has weight zero, $T _ { f } = \{ P _ { 0 } , P _ { 1 } , P _ { 2 } , P _ { 3 } \}$ and distortion instead falls from 1 to $1 / 2$ . The larger equivalence class is a less demanding evaluation target, not a larger remaining candidate set. These values sum over addresses and labels, not over candidate models. A deterministic target contributes one nonzero label term at each address, including background labels.

Figure 2: Evaluation weights determine which targets are equivalent and how much the same correction reduces distortion.

## 5 Information gain, expected improvement, and their diferences

## 5.1 What is uncertain, and under which distribution?

For the entropy-diference arguments in this review, assume $T _ { F }$ is discrete with $H ( T _ { F } ) < \infty$ . Initially,

$$
p _ { T _ { F } , 0 } ( t ) = P ( T _ { F } = t ) , \qquad H ( T _ { F } ) = - \sum _ { t } p _ { T _ { F } , 0 } ( t ) \log _ { 2 } p _ { T _ { F } , 0 } ( t ) .\tag{16}
$$

For discrete $\begin{array} { r } { F , p _ { T _ { F } , 0 } ( t ) = \sum _ { f : T _ { f } = t } \mu ( f ) } \end{array}$ ; in general it is $P ( F \in \{ f : T _ { f } = t \} )$ . The entropy sum counts distinct classes, not interpretations. Only $T _ { F }$ , not necessarily $F ,$ is assumed discrete.

After a realized memory,

$$
p _ { T _ { F } , i } ( t ) = P ( T _ { F } = t \mid M _ { i } = \mathcal { M } _ { i } ) , \qquad H ( p _ { T _ { F } , i } ) = H ( T _ { F } \mid M _ { i } = \mathcal { M } _ { i } ) .\tag{17}
$$

These probabilities are induced by the source and interaction model. The prior $\mu$ does not change with the round; conditioning on the history changes the posterior. They are not automatically the learner’s internal beliefs w. The notation is a theoretical description even when computing these probabilities is impractical.

For a possible response r in the support of the response distribution, let $\mathcal { M } _ { i } ^ { r } = ( \mathcal { Z } _ { i } , r )$ and define the full-round gain

$$
\mathrm { G a i n } _ { i } ( r ; \mathcal { Z } _ { i } ) = H ( T _ { F } \mid M _ { i - 1 } = \mathcal { M } _ { i - 1 } ) - H ( T _ { F } \mid M _ { i } = \mathcal { M } _ { i } ^ { r } ) .\tag{18}
$$

We suppress the context argument when fixed. A realized gain can be negative: an unexpected response can broaden the posterior. Entropy reduction is not a distance from the true target or a guarantee of correct belief. The expression also includes any information supplied by the view and display before the response. The response-only counterpart replaces its first entropy by $H ( T _ { F } \mid Z _ { i } = \mathcal { Z } _ { i } )$

Beyond discrete targets Unrestricted distribution-valued targets need not have a finite Shannon entropy. One may specify a finite-resolution target representation for the entropy calculations, or define mutual information directly through KL divergence between joint and product measures, with finiteness conditions. Replacing the displayed entropies by diferential entropies is not a harmless notation change: the latter depend on parameterization and do not provide the same discrete uncertainty interpretation. We make no finite-gain claim for arbitrary continuous targets here.

## 5.2 Fixed-target expectations versus source averages

Fix $f ,$ a pre-response record $\mathcal { Z } _ { i }$ , and a current predictor $Q _ { i }$ . For a deterministic update, let $Q _ { i + 1 } ( r )$ be the next predictor. Then

$$
\mathbb { E } [ \mathrm { G a i n } _ { i } ( R _ { i } ) \mid { \cal F } = f , Z _ { i } = \mathcal { Z } _ { i } ] = \sum _ { r } \rho _ { f } ( r \mid \mathcal { Z } _ { i } ) \mathrm { G a i n } _ { i } ( r ; \mathcal { Z } _ { i } ) ,\tag{19}
$$

$$
\mathbb { E } [ \tau ( T _ { f } , Q _ { i + 1 } ) \mid F = f , Z _ { i } = \mathcal { Z } _ { i } ] = \sum _ { r } \rho _ { f } ( r \mid \mathcal { Z } _ { i } ) \tau ( T _ { f } , Q _ { i + 1 } ( r ) ) .\tag{20}
$$

For random updates the second summand includes the conditional expectation over update randomness. If $Z _ { i }$ does not fix the current predictor, also condition on its realized state. Averaging before a random view is selected requires an outer expectation over the context distribution. This context distribution is part of the acquisition protocol, not the evaluation weights ν.

The fixed-target expected gain in (19) is not in general a mutual information, and need not be nonnegative. Mutual information appears when targets and responses are averaged under the same joint source model, as derived in Section 8. This distinction prevents treating a teacher’s known-f response distribution as the learner’s predictive mixture.

For a fixed target and finite expected log losses, the entropy floor cancels:

$$
\operatorname { C E } ( P _ { f } , Q _ { i } ) - \operatorname { \mathbb { E } } [ \operatorname { C E } ( P _ { f } , Q _ { i + 1 } ) ] = \tau ( T _ { f } , Q _ { i } ) - \operatorname { \mathbb { E } } [ \tau ( T _ { f } , Q _ { i + 1 } ) ] ,\tag{21}
$$

where both expectations use the same fixed-target context. There is no general equality between either side and target information gain.

## 5.3 When information improves prediction

Fix $Z _ { i } = z ;$ if this record does not determine the current learner state, condition on that state throughout as well. Draw $X \sim \nu$ independently of $( F , R _ { i } )$ , then a fresh $Y \sim P _ { F } ( \cdot \mid X )$ independently of $R _ { i }$ conditional on $( F , X )$ . Assume the expected log losses below are finite. Under the correct joint source model, the Bayes predictors are

$$
\begin{array} { c } { { Q _ { i } ^ { \mathrm { B } } ( y \mid x ) = P ( Y = y \mid X = x , Z _ { i } = z ) , } } \\ { { Q _ { i + 1 } ^ { \mathrm { B } } ( y \mid x , R _ { i } ) = P ( Y = y \mid X = x , Z _ { i } = z , R _ { i } ) . } } \end{array}\tag{22}
$$

The expected improvement from the Bayes update is

$$
{ \mathbb E } [ \tau ( T _ { F } , Q _ { i } ^ { \mathrm { B } } ) - \tau ( T _ { F } , Q _ { i + 1 } ^ { \mathrm { B } } ) \mid Z _ { i } = z ] = I ( Y ; R _ { i } \mid X , Z _ { i } = z ) \ge 0 .\tag{23}
$$

For an actual learner based on the same records, define its nonnegative excess log-loss risks relative to these Bayes predictors:

$$
\begin{array} { r l } & { \ a _ { i } = \mathbb { E } \big [ D _ { \mathrm { K L } } \big ( Q _ { i } ^ { \mathrm { B } } ( \cdot \mid X ) \| Q _ { i } ( \cdot \mid X ) \big ) \mid Z _ { i } = z \big ] , } \\ & { \ a _ { i + 1 } = \mathbb { E } \big [ D _ { \mathrm { K L } } \big ( Q _ { i + 1 } ^ { \mathrm { B } } ( \cdot \mid X , R _ { i } ) \| Q _ { i + 1 } ( \cdot \mid X ) \big ) \mid Z _ { i } = z \big ] . } \end{array}
$$

The expectations also average fresh independent update randomness, if used. Adding these excess risks to the Bayes losses gives

$$
\mathbb { E } [ \tau ( T _ { F } , Q _ { i } ) - \tau ( T _ { F } , Q _ { i + 1 } ) \mid Z _ { i } = z ] = I ( Y ; R _ { i } \mid X , Z _ { i } = z ) + a _ { i } - a _ { i + 1 } .\tag{24}
$$

Prediction therefore improves on average precisely when the predictive information exceeds the increase $a _ { i + 1 } - a _ { i }$ in excess risk. An update can forgo this benefit by fitting corrections poorly or damaging predictions elsewhere; reducing existing excess risk can instead add to the information benefit. A.2 gives the derivation. Both identities average over targets and responses under the same conditional source model; they do not guarantee improvement for each target or response. The information term is response-only predictive information, bounded above by $I ( T _ { F } ; R _ { i } \mid Z _ { i } = z ) ;$ ; neither is the full-round realized gain.

BIFSeg’s comparison from identical initial predictions and scribbles shows that diferent uses of the same corrections can yield diferent current-mask Dice scores [Wang et al., 2018]. The fixed-stream replay proposed in Section 9 similarly isolates assimilation from changed responses. These outcome comparisons do not estimate the information term in (24).

For other losses, an optimal decision can ignore the response, so its expected risk cannot increase if the available decisions are unchanged. The equality with mutual information is specific to log loss. Dice or topology requires the corresponding joint-mask decision model, not merely pixel marginals.

Figure 3 evaluates the next round under a history-insensitive response policy. It uses the source and learner of Figure 1 with uniform evaluation weights, and averages over the next view and correction.

Continue from $r _ { 1 } = \{ ( a , 1 ) \mapsto 0 \}$ with the uniform candidate-average predictor, uniform evaluation weights, and the fixed target $P _ { 1 }$ . The system selects image a or b with probability $1 / 2 .$ In the selected image, a history-insensitive teacher selects its first pixel with probability $1 - \alpha .$ , or selects either pixel uniformly with probability α:

$$
\rho _ { f } ^ { ( \alpha ) } = ( 1 - \alpha ) \rho _ { f } ^ { \mathrm { f i r s t } } + \alpha \rho _ { f } ^ { \mathrm { u n i f o r m } } , \qquad 0 \leq \alpha \leq 1 .\tag{25}
$$

Labels are faithful and repeatable. The following probabilities marginalize over the random view; they are not conditional on one selected image.

<table><tr><td>Response r</td><td> $P ( R _ { 2 } = r \mid F = f , M _ { 1 } = \mathcal { M } _ { 1 } )$ </td><td> $\mathrm { G a i n _ { 2 } }$ </td><td> $\tau ( T _ { f } , \widetilde { Q } _ { 3 } )$ </td></tr><tr><td> $( a , 1 ) \mapsto 0$ </td><td> $1 / 2 - \alpha / 4$ </td><td>0</td><td> $3 / 4$ </td></tr><tr><td> $( a , 2 ) \mapsto 0$ </td><td> $\alpha / 4$ </td><td>1</td><td> $1 / 2$ </td></tr><tr><td> $( b , 1 ) \mapsto 0$ </td><td> $1 / 2 - \alpha / 4$ </td><td>1</td><td> $1 / 2$ </td></tr><tr><td> $( b , 2 ) \mapsto 1$ </td><td> $\alpha / 4$ </td><td>1</td><td> $1 / 2$ </td></tr></table>

$$
\mathbb { E } [ \mathrm { G a i n } _ { 2 } ] = \frac { 1 } { 2 } + \frac { \alpha } { 4 } , \qquad \mathbb { E } [ \tau ( T _ { f } , \widetilde { Q } _ { 3 } ) ] = \frac { 5 } { 8 } - \frac { \alpha } { 1 6 } .\tag{26}
$$

Here every newly labelled pixel contributes one target bit and removes $1 / 4$ bit of distortion. This proportionality follows from the deliberately independent, uniform, noiseless example, not from a general theorem. α changes the response distribution and hence the distribution of future predictors; it does not change ν or $P _ { f }$

Figure 3: Expected gain and distortion after one more correction depend on the teacher’s response policy, even with faithful labels.

The proportionality in Figure 3 is special to its assumptions. Figure 4 changes evaluation relevance while keeping the learner calibrated; Figure 5 instead changes the learner’s predictions and specifies its update. Both comparisons separate target information from predictive benefit.

<table><tr><td colspan="4">Equal information, unequal evaluation relevance</td></tr><tr><td colspan="4">Keep the uniform source and candidate-average learner after , but set  $\nu ( a , 1 ) = 1 / 4 , \nu ( a , 2 ) = 1 / 2 ,$   $r _ { 1 }$   $\nu ( b , 1 ) = \nu ( b , 2 ) = 1 / 8 .$  Every weight is positive, so the evaluation-equivalence classes are unchanged. The predictor is calibrated to the source posterior and its distortion is 50  $3 / 4 .$  For an unknown independent fair label, observing it gives one target bit while removing  $\nu ( x )$  bits of evaluation-weighted distortion:</td></tr><tr><td></td><td>Correction Gain Post-update distortion Distortion reduction</td><td></td></tr><tr><td> $( a , 2 ) \mapsto 0$  1</td><td> $1 / 4$ </td><td> $1 / 2$ </td></tr><tr><td> $( b , 2 ) \mapsto 1$  1</td><td> $5 / 8$ </td><td> $1 / 8$ </td></tr></table>

Both responses cost one mark. Diferent predictive benefits here arise from evaluation relevance, not from miscalibrated beliefs or unsuccessful fitting. Figure 5 restores uniform ν and varies the learner instead.

Figure 4: Equal information gain can give diferent distortion reductions when evaluation weights difer.

## Alternative learner updates

Keep the uniform source posterior on $\mathcal { C } _ { 1 }$ but choose a diferent learner predictor. To make the update fully specified, let the learner maintain independent Bernoulli beliefs at the three unknown pixels. A correction sets its corresponding probability to the supplied hard label and leaves all other probabilities unchanged. These learner beliefs difer from the uniform source posterior used to calculate gain. This is a local illustrative variant, not the uniform update of (12).   
First take $Q _ { 2 } ( 1 \mid x ) = ( 0 , 1 / 2 , 1 / 2 , 1 / 4 )$ , giving initial distortion 1.

<table><tr><td></td><td></td><td>Correction Gain Post-update distortion Distortion reduction</td><td></td></tr><tr><td> $( a , 2 ) \mapsto 0$ </td><td>1</td><td> $3 / 4$ </td><td> $1 / 4$ </td></tr><tr><td> $( b , 2 ) \mapsto 1$ </td><td>1</td><td> $1 / 2$ </td><td> $1 / 2$ </td></tr></table>

The two labels reveal equal target information but repair predictions of diferent quality. Now take $Q _ { 2 } ( 1 \mid x ) = ( 0 , 7 / 8 , 1 / 2 , 1 / 2 )$ , giving initial distortion $5 / 4 .$ . Compare two-mark responses:

<table><tr><td>Correction</td><td></td><td>Gain Post-update distortion Distortion reduction</td><td></td></tr><tr><td> $( b , 1 ) \mapsto 0 , ( b , 2 ) \mapsto 1$ </td><td>2</td><td> $3 / 4$ </td><td> $1 / 2$ </td></tr><tr><td> $( a , 1 ) \mapsto 0 , ( a , 2 ) \mapsto 0$ </td><td>1</td><td> $1 / 2$ </td><td> $3 / 4$ </td></tr></table>

The second response repeats one known label but fixes the more serious remaining prediction error. Both have equal marked-location work. For example, the probability of the true background label at $( a , 2 )$ was $1 / 8 { \mathrm { ; } }$ , contributing $- { \textstyle { \frac { 1 } { 4 } } } \log _ { 2 } ( 1 / 8 ) = 3 / 4$ to initial distortion. Address choices are fixed comparisons, not a separate channel encoding $f .$ These examples demonstrate the distinction between acquisition and assimilation; they do not demonstrate superiority of any real acquisition method.

Figure 5: More target information need not produce a larger distortion reduction when learner beliefs difer from the source posterior.

## 5.4 When does this perspective recover BALD?

The standard expected-KL form of (2) is

$$
\mathrm { B A L D } ( x ) = \sum _ { P } w ( P \mid { \mathcal M } ) D _ { \mathrm { K L } } \big ( P ( \cdot \mid x ) \| \widehat { Q } ( \cdot \mid x , { \mathcal M } ) \big ) .\tag{27}
$$

A.3 gives the expansion. BALD averages model disagreement at a proposed address. Distortion instead averages discrepancy from the fixed teacher target over evaluation addresses. The two KL expressions have diferent averaging distributions and meanings. Equation (23) identifies the distinction: BALD values information about the candidate model; predictive improvement values information about labels at evaluation addresses.

For exact agreement with response information about $T _ { F }$ , the learner’s beliefs must match the sourceconditioned beliefs, the response must be the requested label under the same observation model, and label probabilities must depend on the candidate only through its evaluation class. To see the last condition, let P be the random candidate model and $T = T ( { \mathsf { P } } )$ its class. Then

$$
I ( \mathsf { P } ; Y \mid x , \mathcal { M } ) = I ( T ; Y \mid x , \mathcal { M } ) + I ( \mathsf { P } ; Y \mid T , x , \mathcal { M } ) .\tag{28}
$$

The last term vanishes at an evaluation-relevant address when members of each class specify the same label probabilities. It need not vanish for queries outside the support of $\nu .$ Unknown candidate parameters are not always evaluation-relevant unknowns.

Figure 6 specializes the comparison to the four-pixel setting. Its diference in expected gain also reflects diferent available actions, not only a diferent acquisition criterion.

## Comparison with BALD

In the uniform four-pixel example all surviving candidates are deterministic, so BALD is zero at $( a , 1 )$ and one bit at every other pixel. A requested new pixel gives gain one and post-update distortion $1 / 2 .$ The history-insensitive teacher of (25) gives (26) instead. This comparison changes the action space: unrestricted pixel requests are compared with random views and a teacher who may repeat a mark. It is not evidence that BALD is intrinsically better than teacher modelling. A fair experiment must match views, response opportunities, work, and learner updates.

Figure 6: The four-pixel BALD comparison changes both the selection rule and the response opportunities; it is not a matched-policy experiment.

## 6 Human work and teaching objectives

For responses with random supports $\Xi _ { i }$ , define marked-location work

$$
W _ { n } = \sum _ { i = 1 } ^ { n } | \Xi _ { i } | .\tag{29}
$$

Repeated marks in diferent rounds count again, but support sets do not measure retracing, inspection, navigation, or time. For human efort, specify $\textstyle C _ { n } = \sum _ { i = 1 } ^ { n } c _ { i }$ in a declared unit: $c _ { i } = | \Xi _ { i } |$ recovers $W _ { n } ;$ active or elapsed seconds measure diferent resources. Record repeats, edits, empty responses, and waiting without double-counting overlapping activities. An empty response can cost time. Navigation and inspection must be held fixed or charged if a policy changes them; otherwise target information can enter through an uncharged action.

Fix a learner, a horizon n, an evaluation distribution, and a class Π of allowed interaction policies. A policy specifies which choices are controlled by the system and which by the teacher. Two ideal objectives are

$$
\begin{array} { r l } { \underset { \pi \in \Pi } { \operatorname* { s u p } } } & { \mathbb { E } _ { \pi } \left[ \displaystyle \sum _ { i = 1 } ^ { n } \mathrm { G a i n } _ { i } ( R _ { i } ; Z _ { i } ) \right] } \\ { \underset { \pi \in \Pi } { \operatorname* { i n f } } } & { \mathbb { E } _ { \pi } \big [ \tau ( T _ { F } , Q _ { n + 1 } ) \big ] } \end{array}
$$

$$
\mathrm { s u b j e c t ~ t o ~ } \mathbb { E } _ { \pi } [ C _ { n } ] \leq b ,\tag{30}
$$

$$
\mathrm { s u b j e c t ~ t o ~ } \mathbb { E } _ { \pi } [ C _ { n } ] \leq b .\tag{31}
$$

Expectations average over targets and interaction and training randomness under the declared protocol. For a single fixed target, condition throughout on $F = f$ instead. Gain and its posterior are defined under the joint model induced by each policy, not by applying one policy’s posterior to another policy’s responses. Suprema and infima need not be attained. These are definitions of objectives, not claims that their values or optimizers can be computed from logs. An expected-cost constraint does not bound the efort in every individual session; a hard budget needs an explicit termination rule. The predictor at the horizon must have a declared readout and computational budget. These are fixed-horizon objectives; adaptive stopping requires a stopping-time formulation and accounting for a teacher-controlled stop signal.

Comparisons at a common budget additionally require a common action set, initialization, evaluation distribution, and declared training resource. Alternatively, changes in these components must be explicit experimental factors. A more powerful annotation interface or extra fitting iterations can explain a benefit attributed to a query score. Neither equal mark counts nor a ratio labelled “information per click” removes those alternatives.

Gain and log-loss distortion are measured in bits, but are not the same quantity: one concerns target uncertainty, the other error per evaluation draw. Work is measured in marks, or separately in seconds. Constraints such as (30) avoid adding incompatible units. Local ratios such as expected distortion reduction divided by expected work are useful diagnostics, but are not generally optimal multi-round policies. A ratio of expectations is also diferent from an expectation of a ratio, particularly when responses can be empty.

## 7 Prequential evaluation: what response predictability establishes

For a discrete alphabet and integer lengths satisfying Kraft’s inequality, the shortest expected prefix-code length lies between entropy and entropy plus one bit. Ideal lengths $- \log _ { 2 } { q ( a ) }$ instead correspond to a chosen probability model; arithmetic coding can implement a sequence code with small additional overhead. These ideal lengths need not be integers.

Two-part MDL describes a model and then data conditional on that model. Prequential coding predicts and encodes observations using only already decoded observations [Rissanen, 1978, Grünwald, 2007, Grünwald and Roos, 2020]. This instantiates the sequential forecast evaluation advocated by [Dawid, 1984], with the logarithmic scoring rule [Gneiting and Raftery, 2007]. For $D _ { N } = \left( d _ { 1 } , \ldots , d _ { N } \right)$

$$
\ell _ { \mathrm { t w o } } ( \theta , D _ { N } ) = \ell ( \theta ) + \ell ( D _ { N } \mid \theta ) ,\tag{32}
$$

$$
\ell _ { \mathrm { p r e q } } ( D _ { N } ) = - \sum _ { t = 1 } ^ { N } \log _ { 2 } q _ { t } ( d _ { t } \mid d _ { 1 } , \dots , d _ { t - 1 } ) .\tag{33}
$$

For continuous parameters, a density at an exact θ is not a probability mass and $- \log _ { 2 } p ( \theta )$ is not by itself a parameter code. A two-part code must specify a finite precision and a decodable parameter description.

The comparative use here is forecasting a fixed response stream before seeing each response, not minimizing how much useful supervision a teacher supplies. For well-specified i.i.d. data in a fixed regular k-parameter family, suitable two-part and Bayesian-mixture codes have leading redundancy $\left( k / 2 \right) \log _ { 2 } N$ relative to fitted likelihood. Regularity requires an interior parameter, nonsingular Fisher information, and concentration; an expected expansion also requires control of the remainder (B). This is not an arbitrary neural-training guarantee or a bound on human efort. Adaptive responses do not justify replacing N by rounds or marks.

## 8 Information in adaptive response streams

Fix the shared initial image collection, evaluation weights, protocol, and initial state; all probabilities in this section condition on them. The initial memory is empty. Let $\widehat { \rho } _ { i } ( r \mid z )$ be a normalized causal response model, shared by encoder and decoder before the response is encoded. It must not depend on the current unknown response or the hidden target except through available information. Define

$$
\ell _ { i } = - \log _ { 2 } { \widehat { \rho } } _ { i } ( R _ { i } \mid Z _ { i } ) , \qquad \ell _ { \mathrm { P T } } ( n ) = \mathbb { E } \left[ \sum _ { i = 1 } ^ { n } \ell _ { i } \right] .\tag{34}
$$

This codes a correction process, including support selection, rather than only the target label at a requested pixel. The coder and decoder must agree on context and coding probabilities; they need not independently train identical segmentation networks. If obtaining a context requires additional communication, its cost is outside (34) unless included explicitly. In particular, the response code is not a code for the entire interactive system’s trafic.

The true context-conditional response mass is $\rho _ { i } ( r \mid z ) = P ( R _ { i } = r \mid Z _ { i } = z )$ , obtained by marginalizing the target-conditioned kernels under the source posterior. The estimated coder $\widehat { \rho } _ { i }$ is neither $P _ { f }$ nor $Q _ { i }$ and need not be statistically consistent. Assume it is positive wherever the true response mass is positive when a finite length is desired.

Proposition 1 (Response coding and information). For discrete responses, shared decoding context, a normalized causal $\widehat { \rho } _ { i }$ , and finite terms,

$$
\begin{array} { r } { \mathbb { E } [ \ell _ { i } ] = H ( R _ { i } \mid Z _ { i } ) + \mathbb { E } _ { Z _ { i } } D _ { \mathrm { K L } } ( \rho _ { i } ( \cdot \mid Z _ { i } ) \| \widehat { \rho } _ { i } ( \cdot \mid Z _ { i } ) ) , } \end{array}\tag{35}
$$

$$
\ell _ { \mathrm { P T } } ( n ) \geq \sum _ { i = 1 } ^ { n } H ( R _ { i } \mid Z _ { i } ) \geq \sum _ { i = 1 } ^ { n } I ( T _ { F } ; R _ { i } \mid Z _ { i } ) .\tag{36}
$$

These are the standard cross-entropy decomposition and entropy bound on mutual information, applied to context-conditional responses; the algebra is in A.4. Their relevance is that they separate response information from coding-model error.

The residual $H ( R _ { i } \mid T _ { F } , Z _ { i } )$ can include support-choice variability, annotation noise, and behaviour that distinguishes interpretations within one target class. It is not generally the label-variability floor $H _ { f } ( Y \mid X )$ Combining the identities gives a useful diagnostic decomposition:

$$
\begin{array} { r } { \mathbb { E } [ \ell _ { i } ] = I ( T _ { F } ; R _ { i } \mid Z _ { i } ) + H ( R _ { i } \mid T _ { F } , Z _ { i } ) + \mathbb { E } _ { Z _ { i } } D _ { \mathrm { K L } } \big ( \rho _ { i } \| \widehat { \rho _ { i } } \big ) . } \end{array}\tag{37}
$$

Long code can mean useful information, response variabili $\mathrm { t y , }$ or a poor coder. Short code can mean good prediction or repeated, uninformative corrections. Neither minimizing nor maximizing code length alone optimizes teaching.

## 8.1 A measurable comparison on the same interaction stream

Compare two normalized causal models, $\widehat { \rho } _ { i } ^ { ( 0 ) }$ and $\widehat { \rho } _ { i } ^ { ( 1 ) }$ , on the same recorded contexts and responses. Their forecasts must be fixed before the scored response arrives, with fitting and model selection restricted to prior data or separate development sessions. Condition throughout this comparison on the shared model and update specifications, development information, and fitting randomness fixed before scoring begins. Their description costs are excluded: this is a conditional predictive-code comparison, not a comparison of complete MDL descriptions. Assume both response models have finite expected log loss, so their expected contrast does not subtract infinite quantities. Use identical response alphabets and batch boundaries. The observed length advantage of model 1 is

$$
\Delta \ell _ { n } = \sum _ { i = 1 } ^ { n } \log _ { 2 } \frac { \widehat \rho _ { i } ^ { ( 1 ) } ( r _ { i } \mid \mathcal { Z } _ { i } ) } { \widehat \rho _ { i } ^ { ( 0 ) } ( r _ { i } \mid \mathcal { Z } _ { i } ) } .\tag{38}
$$

This score is computable without $\mu .$ . Its source expectation satisfies

$$
\mathbb { E } [ \Delta \ell _ { n } ] = \sum _ { i = 1 } ^ { n } \mathbb { E } _ { Z _ { i } } \left[ D _ { \mathrm { K L } } ( \rho _ { i } \Vert \widehat { \rho } _ { i } ^ { ( 0 ) } ) - D _ { \mathrm { K L } } ( \rho _ { i } \Vert \widehat { \rho } _ { i } ^ { ( 1 ) } ) \right] ,\tag{39}
$$

with all response distributions conditional on the same $Z _ { i }$ . This follows by cancelling the common entropy in (35); its individual KL terms need not be observable. History, overlay, or support-geometry features can be compared this way, holding the conditional label model fixed when testing support prediction. A positive score establishes better prediction of that stream, not identification of true gain, improved segmentation, or the value of a diferent acquisition policy that would change the stream. Figure 7 supplies a normalized baseline for this test.

## A complete response-code baseline

For a displayed set of m addresses, draw the batch size uniformly from $0 , \ldots , m$ , then a uniform subset of that size and independent uniform labels. For every admissible paint/skip response,

$$
{ \widehat \rho } _ { i } ^ { ( 0 ) } ( \xi , h \mid z ) = \frac { 1 } { ( m + 1 ) { \binom { m } { | \xi | } } | \mathcal { V } | ^ { | \xi | } } .\tag{40}
$$

There are ${ \binom { m } { k } } | { \mathcal { V } } | ^ { k }$ responses of size $k ,$ so their total mass is $1 / ( m + 1 )$ . This includes the unique empty response and sums to one.

In image a of the four-pixel example, $m = 2$ and $| \mathscr { y } | = 2$ . Skipping has probability $1 / 3 ;$ each specified one- or two-pixel response has probability $1 / 1 2$ . Thus $\{ ( a , 1 ) \mapsto 0 \}$ costs $\log _ { 2 }$ 12 bits even if its label is already known. A causal history-aware forecast can shorten that code without making the repeated label informative.

This intentionally simple reference models neither stroke geometry nor teacher optimality. Learned size, support, and label factors can replace its uniform factors; score each forecast before updating it. Independence is a baseline choice, not an assumption about real corrections.

Figure 7: A normalized response model makes same-stream scoring operational without estimating the teacher’s target or information gain.

## 8.2 Which information belongs to the response?

For nested records with empty initial memory, discrete $T _ { F }$ of finite entropy, and the joint source–interaction distribution, the full-round gain in (18) satisfies

$$
\mathbb { E } \left[ \sum _ { i = 1 } ^ { n } \mathrm { G a i n } _ { i } ( R _ { i } ; Z _ { i } ) \right] = I ( T _ { F } ; M _ { n } ) .\tag{41}
$$

The conditional-entropy averaging and telescoping steps are given in A.5. This expectation is not the fixed-f average of (19). Nor is the resulting information necessarily all conveyed by the responses: the displayed context may contain teacher choices or other target-dependent information.

Because $Z _ { i }$ contains $M _ { i - 1 }$ and $M _ { i } = ( Z _ { i } , R _ { i } )$ , the chain rule yields

$$
H ( T _ { F } \mid M _ { i - 1 } ) - H ( T _ { F } \mid M _ { i } ) = I ( T _ { F } ; Z _ { i } \mid M _ { i - 1 } ) + I ( T _ { F } ; R _ { i } \mid Z _ { i } ) ,\tag{42}
$$

$$
I ( T _ { F } ; M _ { n } ) = \sum _ { i = 1 } ^ { n } I ( T _ { F } ; Z _ { i } \mid M _ { i - 1 } ) + \sum _ { i = 1 } ^ { n } I ( T _ { F } ; R _ { i } \mid Z _ { i } ) .\tag{43}
$$

One can verify the first line by adding and subtracting $H ( T _ { F } \mid Z _ { i } ) $ ; the second telescopes.

Proposition 2 (When response length bounds full-memory gain). Under the preceding coding and finiteentropy conditions, suppose the pre-response context supplies no fresh target information:

$$
I ( T _ { F } ; Z _ { i } \mid M _ { i - 1 } ) = 0 \quad f o r \ e v e r y \ i .\tag{44}
$$

Then

$$
\ell _ { \mathrm { P T } } ( n ) \geq I ( T _ { F } ; M _ { n } ) = \mathbb { E } \left[ \sum _ { i = 1 } ^ { n } \mathrm { G a i n } _ { i } ( R _ { i } ; Z _ { i } ) \right] .\tag{45}
$$

This follows by setting the context terms in (43) to zero and applying (36); see also A.5.

A suficient protocol generates views and displays from the logged past, fixed initial setup, and fresh target-independent randomization, with no additional hidden target-dependent input. Such views can depend on a learned $Q _ { i } ;$ conditioning on the relevant past is what matters. Teacher navigation can violate (44). Either code that choice in causal order as a teacher message, or keep its context-information term. Neither proposition identifies $M _ { n }$ with $\mathbf { R } _ { n } = ( R _ { 1 } , \ldots , R _ { n } )$ , and the argument does not require reconstructing a random view from response addresses alone. This is the role of the nested pre-/post-response memory.

## 8.3 Support, labels, and work

The response-code chain rule is

$$
\widehat { \rho } _ { i } ( \xi , h \mid z ) = \widehat { \rho } _ { i } ( \xi \mid z ) \widehat { \rho } _ { i } ( h \mid \xi , z ) .\tag{46}
$$

It assumes no independence. Consequently,

$$
\ell _ { \mathrm { P T } } ( n ) = \mathbb { E } \left[ - \sum _ { i = 1 } ^ { n } \log _ { 2 } \widehat { \rho } _ { i } ( \Xi _ { i } \mid Z _ { i } ) \right] + \mathbb { E } \left[ - \sum _ { i = 1 } ^ { n } \log _ { 2 } \widehat { \rho } _ { i } ( H _ { i } \mid \Xi _ { i } , Z _ { i } ) \right] .\tag{47}
$$

In a shared ordering of marked addresses, the conditional label code factors into $W _ { n }$ sequential label terms (A.6). Labels may be dependent; no retraining after each within-response label is required. Support size and addresses have their own cost. Thus work counts marked locations while coding weights events by predictability. Pixel counts are not independent sample counts.

There is a parallel information decomposition,

$$
I ( T _ { F } ; R _ { i } \mid Z _ { i } ) = I ( T _ { F } ; \Xi _ { i } \mid Z _ { i } ) + I ( T _ { F } ; H _ { i } \mid \Xi _ { i } , Z _ { i } ) .\tag{48}
$$

This support-first decomposition is order-dependent, not a causal attribution: redundant information enters the first term. Selected addresses can be informative without their information being estimable or useful to the learner. The unmarked complement is determined by the view and support, not an independent channel to count again.

Figure 8 isolates support information using a known selection convention. It changes the four-pixel source and teacher policy explicitly, showing a possibility rather than assuming typical human behaviour.

## The address can distinguish targets when labels do not

Use the four addresses with positive evaluation weights, but restrict the source to equal probabilities on $P _ { 0 } = ( 0 , 0 , 0 , 0 )$ and $P _ { 1 } = ( 0 , 0 , 0 , 1 )$ . Display image a under the same initial predictor. Specify a known teacher policy: under $P _ { 0 }$ , mark $( a , 1 ) \mapsto 0 { : }$ ; under $P _ { 1 }$ , mark $( a , 2 ) \mapsto 0$ . Both targets agree on both displayed labels. Nevertheless,

$$
H ( T _ { F } \mid Z ) = 1 , \qquad I ( T _ { F } ; \Xi \mid Z ) = 1 , \qquad I ( T _ { F } ; H \mid \Xi , Z ) = 0 .
$$

The address identifies the target; its label adds no information once the address is known. A learner using only compatibility with the supplied label retains both candidates. This is a deliberately explicit signalling convention, not a model of typical human behaviour: without a known or independently validated selection mechanism, the same inference is unjustified. It isolates the possibility expressed by (48) from the empirical question of whether real support choices can be exploited.

Figure 8: Under a known selection convention, the chosen address identifies the target even though both targets agree on the supplied labels.

Information acquired is not necessarily retained If the terminal predictor is a function of $M _ { n }$ (or a stochastic readout conditionally independent of $T _ { F }$ given $M _ { n } )$ , data processing gives $I ( T _ { F } ; Q _ { n + 1 } ) \le I ( T _ { F } ; M _ { n } )$ This is a statement about information, not an upper or lower bound on predictive distortion without further assumptions. If the learner has additional target-dependent side information, it must be included in the conditioning or the state. The same attention to reconstruction and side information is needed in rate–distortion formulations [Nokleby et al., 2015]. Improving with every prefix does not establish optimal successive refinement [Equitz and Cover, 1991].

## 9 What does an improvement per human efort establish?

## 9.1 A map of objectives rather than a single eficiency score

Table 5 summarizes the distinctions developed above. Methods are related by their objective structure, not asserted to optimize identical quantities. The cited applications extend beyond segmentation; the specialization here is the corrective spatial interaction.

For comparing coding models, hold the response stream fixed and predict each response before fitting it. A short code on a diferent stream does not establish a better teaching policy. Conversely, comparing policies requires evaluating the resulting predictions and work, not just their code lengths. These comparisons separate response predictability, information acquisition, and learning benefit.

## 9.2 Observed, estimated, and latent quantities

Table 6 separates what the mathematical definitions permit from what an interaction log identifies. Fitting a statistical response predictor does not necessarily identify the human’s attention or selection mechanism. Likewise, exact target gain need not be an empirical endpoint in order to test improved learning.

For fixed checkpoints $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ , representative audit labels give a particularly useful bridge. With independent draws $X _ { j } \sim \nu , Y _ { j } \sim P _ { f } ( \cdot \mid X _ { j } )$ , independent of fitting and checkpoint selection, define

$$
{ \widehat { \Delta } } _ { m } = { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } \log _ { 2 } { \frac { Q ^ { ( 1 ) } ( Y _ { j } \mid X _ { j } ) } { Q ^ { ( 0 ) } ( Y _ { j } \mid X _ { j } ) } } ,\tag{49}
$$

$$
\mathbb { E } [ \widehat { \Delta } _ { m } ] = \tau ( T _ { f } , Q ^ { ( 0 ) } ) - \tau ( T _ { f } , Q ^ { ( 1 ) } ) .\tag{50}
$$

Assume the relevant log losses are finite. The unknown entropy floor cancels, so estimating predictive improvement does not require recovering µ or absolute distortion. The audit’s reference-label mechanism must represent the declared target; annotator noise or consensus labels can define a diferent estimand. Spatial,

Table 5: Quantities and their relation to established approaches.
<table><tr><td>Quantity</td><td>Interpretation</td><td>Related approach and qualification</td></tr><tr><td>Expected gain</td><td>Target information acquired</td><td>BALD/BatchBALD under matching beliefs, observation models, and target relevance [Houlsby et al., 2011,</td></tr><tr><td>Expected distortion</td><td>Predictive error</td><td>Kirsch et al., 2019]. Variance reduction for squared error under negligible bias, not general KL</td></tr><tr><td>Gain/work</td><td>Information per declared cost unit</td><td>[Cohn et al., 1996]. Cost-aware acquisition; CEREALS uses informativeness and predicted annotation cost, not exact target gain</td></tr><tr><td>Distortion reduction/work</td><td>Predictive improvement per declared cost unit</td><td>[Mackowiak et al., 2018]. Related to ROI acquisition; Haertel et al. consider accuracy benefit and annotation cost [Haertel et al., 2015].</td></tr><tr><td>Prequential length</td><td>Predictability of a response stream</td><td>Applies sequential predictive evaluation to support and labels</td></tr><tr><td>Length/work</td><td>Code bits per declared cost unit</td><td>[Dawid, 1984, Blier and Ollivier, 2018]. A response-model diagnostic, not a measure of target gain.</td></tr></table>

Table 6: Evidence required to operationalize the framework.
<table><tr><td>Quantity</td><td>Available evidence</td><td>What does not follow</td></tr><tr><td>View, checkpoint, support, labels, times</td><td>Direct logs, if the interface records them.</td><td>Display is not inspection; non-correction is not acceptance.</td></tr><tr><td> $W _ { n }$  and measured  $C _ { n }$ </td><td>Support counts or a declared action/timing protocol.</td><td>Equal marks need not mean equal human effort.</td></tr><tr><td>Response log loss</td><td>Score causal forecasts on the same recorded stream.</td><td>A better response predictor need not identify gain or improve learning.</td></tr><tr><td>Predictive risk difference</td><td>Representative, independently assessed labels for declared checkpoints.</td><td>Training corrections alone need not represent the evaluation distribution.</td></tr><tr><td>Source posterior, exact target gain, entropy floor</td><td>A specified source and observation model or additional identifying evidence.</td><td>A single correction log generally does not identify these quantities.</td></tr><tr><td>Responses to undisplayed views</td><td>Interventions, or justified overlap and Replaying recorded corrections does response-model assumptions.</td><td>not reveal counterfactual human choices.</td></tr></table>

image, and participant dependence must be reflected in uncertainty estimates, and choosing a checkpoint on the same audit requires separate validation or an appropriate sequential procedure.

## 9.3 Evidence for learning and eficiency

A benefit claim must specify the outcome, reference population, cost unit, allowed actions, and training resources. Independent paired risk comparisons apply beyond log loss; the KL and mutual-information identities do not. Dice and topology require evaluating whole masks rather than only marginal label distributions. Table 2 distinguishes current-mask completion, later-image prediction, and response prediction, as well as human time from cost proxies.

Three questions organize the research priorities below: what selection reveals, whether using it improves assimilation, and whether the resulting interaction policy improves outcomes at matched efort. They require diferent evidence. A useful response predictor need not identify human intentions, but interpreting non-correction as acceptance requires more than predictive accuracy.

## 9.4 What does selection reveal?

Pedagogical and teacher-aware learning interpret why an example was chosen [Shafto et al., 2014, Yuan et al., 2021]; RClicks models click locations [Antonov et al., 2024]. For spatial corrections, first test whether history and display improve response forecasts on a common recorded stream (38). This tests prediction of behaviour, not the correctness of unmarked locations

In particular, displayed, inspected, and accepted are diferent events. An unmarked region can be correct, overlooked, ambiguous, or deferred. A simple identification problem illustrates the limitation. At a randomly audited displayed location, let A indicate an actual prediction error and C indicate a corrective mark. Under the restrictive assumption that only errors are marked,

$$
P ( C = 1 \mid z ) = P ( A = 1 \mid z ) P ( C = 1 \mid A = 1 , z ) .\tag{51}
$$

Observed marking frequency alone cannot identify both error prevalence and the probability of detecting and marking an error. Allowing erroneous or confirmatory marks adds a term for A = 0 and does not resolve the ambiguity. This is a basic identification issue, not a new independence assumption. It parallels the prevalence–labelling ambiguity in positive–unlabelled learning [Bekker and Davis, 2020]; missing-data theory similarly makes ignoring the observation mechanism conditional on explicit assumptions [Rubin, 1976]. Interactive-learning interface research already warns against interpreting unassigned examples as negative evidence [Dudley and Kristensson, 2018].

An identification study can audit a randomly chosen subset of displayed but unmarked regions using an independent reference or a separate inspection protocol. Compare ignoring non-correction, treating it as acceptance, and using calibrated soft evidence of acceptance. Exposure probabilities and audit sampling must be recorded. Stated assumptions and direct validation are preferable to silently turning every blank region into a pseudo-label.

## 9.5 Does using selection improve assimilation?

Compare ordinary supervised learning from addressed labels with a learner that additionally models their selection, using the same correction stream and fitting resources. Both receive the pixel addresses and labels; the distinction is whether the selection mechanism informs the update. Independent audit risk (49) measures the outcome without recovering the source posterior. The excess-risk decomposition (24) explains why more informative input alone is insuficient; BIFSeg illustrates the matched-correction comparison with a current-image Dice endpoint [Wang et al., 2018].

To test sequencing, replay the same multiset of corrections in diferent orders at matched computational budgets, comparing delayed distortion and errors in previously corrected regions. Curriculum and continual learning motivate order, weighting, and replay interventions [Bengio et al., 2009, De Lange et al., 2022]. Unlike image ordering that also changes subsequent corrections [Torpey et al., 2026], fixed-stream replay separates these learner-side efects from acquisition.

Recorded corrections respond to particular displays: replay cannot reveal what a human would supply under a changed policy. Repeats likewise do not identify forgetting; delay, stale display, ambiguity, or annotation error can also explain them.

## 9.6 Does the interaction policy improve outcomes at matched efort?

Test the closed-loop policy with actual interaction: changing views or updates also changes what the teacher sees and supplies. Compare with strong uncertainty-plus-diversity, BatchBALD where feasible, and core-setstyle baselines under common action, initialization, and fitting budgets. Record view, checkpoint, support, labels, edits, empty responses, navigation, time, and when corrections enter training. Participant and case allocation, protocol order, and independent evaluation determine the causal interpretation. Report paired or clustered uncertainty, the average performance trajectory, and prespecified rare-structure failures. A favourable response score or replay result motivates this test but does not establish its outcome.

## 9.7 Extensions: teaching stages, exposure, and stopping

Changes in teaching strategies [Cakmak and Thomaz, 2014, Fiebrink et al., 2011] and the greater search efort for small residual errors [Benenson et al., 2019] motivate stage-dependent models. RootPainter starts with clear examples [Smith et al., 2022a]; progression through imbalance correction to boundary refinement remains a hypothesis. Test stage features beyond round, current error, class balance, expertise, fatigue, and display delay, then evaluate any stage-dependent sampling or replay intervention.

Exploratory views and independent audits address blind spots outside a learner’s beliefs. Formal guarantees impose additional conditions.

Equivalence-class active learning groups hypotheses by a shared decision, not necessarily by the evaluationequivalence relation used here [Golovin et al., 2010]. Greedy cardinality-budget guarantees require adaptive monotonicity and submodularity; cost-to-cover results need stronger conditions, including the corrected cover analysis of [Golovin and Krause, 2011]. These are not guarantees for arbitrary neural updates or human corrections. Overlapping noisy response distributions require accumulated evidence, not elimination after one contradictory mark.

A low predicted gain or an empty correction does not certify small remaining distortion. Confidence sequences support repeated monitoring under specified sampling and estimand assumptions [Howard et al., 2021]; applying them requires representative observations and a declared risk for a changing predictor. Neither an acquisition model’s confidence nor a fixed training budget supplies that evidence.

## 10 Discussion and conclusion

Three conclusions organize the evidence. First, response protocols afect measured performance: RClicks changes benchmark dificulty, while Benenson et al. link correction behaviour to search efort and completedmask quality [Antonov et al., 2024, Benenson et al., 2019]. Second, assimilation matters: BIFSeg and Crayons compare ways of using supervision, rather than information content alone [Wang et al., 2018, Fails and Olsen, 2003]. Third, fewer annotations do not establish less human efort, and completion quality does not establish reusable-model performance. The Cellpose and microscopy SAM comparisons illustrate why the cost unit and endpoint matter [Pachitariu and Stringer, 2022, Archit et al., 2025]. Their heterogeneous outcomes support these distinctions, not a ranking of systems.

The mathematical account explains when information produces expected log-loss improvement and how an approximate update can lose it. Across losses, independent evaluation can test predictive benefit without estimating exact gain or identifying human intentions. The principal open question is whether modelling selection and sequencing improves assimilation and, in actual interaction, segmentation at matched efort beyond what addressed labels and fitting resources explain.

The formal core holds the target fixed within a session. A changing teacher interpretation, or one influenced by model exposure, would require an additional state model; performative prediction is an adjacent literature [Perdomo et al., 2020]. That extension is not needed merely because RootPainter trains asynchronously.

## A Standard identities used in the comparison

This appendix records the routine algebra underlying the main text. These are established probability and information identities, not new results. The application-specific issues are the choice of random objects, available conditioning information, and learner update. All sums below use the support of the corresponding distribution, with the zero-probability conventions specified in Section 4. Finite-entropy or finite-loss conditions are imposed wherever diferences of quantities are taken.

## A.1 Log-loss risk, entropy, and distortion

For a fixed target, insert $P _ { f } ( y \mid x )$ into the logarithm:

$$
\begin{array} { l } { \displaystyle \mathrm { C E } ( P _ { f } , Q ) = - \sum _ { x , y } \nu ( x ) P _ { f } ( y \mid x ) \log _ { 2 } Q ( y \mid x ) } \\ { \displaystyle = - \sum _ { x , y } \nu ( x ) P _ { f } ( y \mid x ) \log _ { 2 } P _ { f } ( y \mid x ) + \sum _ { x , y } \nu ( x ) P _ { f } ( y \mid x ) \log _ { 2 } \frac { P _ { f } ( y \mid x ) } { Q ( y \mid x ) } } \\ { \displaystyle = H _ { f } ( Y \mid X ) + \sum _ { x } \nu ( x ) D _ { \mathrm { K L } } ( P _ { f } ( \cdot \mid x ) \| Q ( \cdot \mid x ) ) } \\ { \displaystyle = H _ { f } ( Y \mid X ) + \tau ( T _ { f } , Q ) . } \end{array}
$$

The entropy term is independent of Q. For two predictors, possibly with a random post-response predictor, it therefore cancels from the expected risk diference in (21). That cancellation holds for a fixed $P _ { f }$ and common evaluation weights, not for a changed target or evaluation population. It introduces no relation to target entropy $H ( T _ { F } )$

## A.2 Predictive information and assimilation

Condition throughout on $Z _ { i } = z$ and use the fresh evaluation draw of Section 5. The Bayes predictors have expected log losses $H ( Y \mid X , z )$ and $H ( Y \mid X , R _ { i } , z )$ . The expected target entropy floor is unchanged by iterated expectation, so subtraction proves (23).

For the actual predictors and excess risks defined in the main text, adding and subtracting the respective Bayes log probabilities gives

$$
\begin{array} { r l } & { \mathbb { E } [ \mathrm { C E } ( P _ { F } , Q _ { i } ) \mid z ] = H ( Y \mid X , z ) + a _ { i } , } \\ & { \mathbb { E } [ \mathrm { C E } ( P _ { F } , Q _ { i + 1 } ) \mid z ] = H ( Y \mid X , R _ { i } , z ) + a _ { i + 1 } . } \end{array}
$$

Subtracting and cancelling the same expected target entropy floor proves (24), under the stated finiteloss and conditioning assumptions. Conditional data processing gives $I ( Y ; R _ { i } \mid X , z ) \leq I ( T _ { F } ; R _ { i } \mid z )$ : at evaluation-relevant addresses, the fresh label depends on the response only through the target class.

## A.3 BALD as expected KL disagreement

Fix an address x and observed memory M. Abbreviate w $( P \mid M )$ by $w _ { P }$ and $\widehat { Q } ( y \mid x , { \mathcal { M } } )$ by $\widehat { Q } _ { y }$ . Using $\begin{array} { r } { \widehat { Q } _ { y } = \sum _ { P } w _ { P } P ( y \mid x ) } \end{array}$

$$
\begin{array} { l } { \displaystyle \sum _ { P } w _ { P } D _ { \mathrm { K L } } ( P ( \cdot \mid x ) \| \widehat { Q } ( \cdot \mid x , \mathcal { M } ) ) = \displaystyle \sum _ { P , y } w _ { P } P ( y \mid x ) \log _ { 2 } P ( y \mid x ) - \displaystyle \sum _ { y } \widehat { Q } _ { y } \log _ { 2 } \widehat { Q } _ { y } } \\ { = H ( \widehat { Q } ) - \displaystyle \sum _ { P } w _ { P } H ( P ( \cdot \mid x ) ) . } \end{array}
$$

Under the joint model $P ( \mathsf { P } = P , Y = y \mid x , M ) = w _ { P } P ( y \mid x )$ , this is $I ( \mathsf { P } ; Y \mid x , { \mathcal { M } } )$ . Identifying it with information about the source target additionally requires the belief and target-relevance conditions of Section 5.4.

## A.4 Entropy, code length, and coding-model mismatch

For completeness, if integer prefix lengths satisfy $\begin{array} { r } { C = \sum _ { a } 2 ^ { - \ell \left( a \right) } \le 1 } \end{array}$ , define $q ( a ) = 2 ^ { - \ell ( a ) } / C$ on the code alphabet. Then

$$
\sum _ { a } p ( a ) \ell ( a ) = H ( p ) + D _ { \mathrm { K L } } ( p \| q ) - \log _ { 2 } C \geq H ( p ) .
$$

The choice $\ell ( a ) = \lceil - \log _ { 2 } p ( a ) \rceil$ on positive-probability symbols satisfies Kraft’s inequality and gives an expected length less than $H ( p ) + 1$ . The main text instead uses ideal probability-based lengths, for which implementation overhead is separate.

At a fixed context z, write $\rho _ { i } ( r \mid z )$ for the true response mass. Adding and subtracting its log probability gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \ell _ { i } \mid Z _ { i } = z ] = - \sum _ { r } \rho _ { i } ( r \mid z ) \log _ { 2 } \widehat { \rho } _ { i } ( r \mid z ) } \\ { \displaystyle = - \sum _ { r } \rho _ { i } ( r \mid z ) \log _ { 2 } \rho _ { i } ( r \mid z ) + \sum _ { r } \rho _ { i } ( r \mid z ) \log _ { 2 } \frac { \rho _ { i } ( r \mid z ) } { \widehat { \rho } _ { i } ( r \mid z ) } } \\ { \displaystyle = H ( R _ { i } \mid Z _ { i } = z ) + D _ { \mathrm { K L } } ( \rho _ { i } ( \cdot \mid z ) \| \widehat { \rho } _ { i } ( \cdot \mid z ) ) . } \end{array}
$$

Averaging over $Z _ { i }$ proves (35). The other step in Proposition 1 is

$$
I ( T _ { F } ; R _ { i } \mid Z _ { i } ) = H ( R _ { i } \mid Z _ { i } ) - H ( R _ { i } \mid T _ { F } , Z _ { i } ) \leq H ( R _ { i } \mid Z _ { i } ) .
$$

Summing over rounds proves (36). Nonnegativity of the last conditional entropy uses discrete responses; it is not a general property of diferential entropy.

## A.5 Posterior-entropy averaging and telescoping gains

For a discrete memory value $m ,$ let $p _ { m } ( t ) = P ( T _ { F } = t \mid M _ { i } = m )$ . Then

$$
\begin{array} { l } { \mathbb { E } _ { M _ { i } } [ H ( p _ { M _ { i } } ) ] = - \displaystyle \sum _ { m } P ( M _ { i } = m ) \sum _ { t } P ( T _ { F } = t \mid M _ { i } = m ) \log _ { 2 } P ( T _ { F } = t \mid M _ { i } = m ) } \\ { \displaystyle = - \sum _ { m , t } P ( T _ { F } = t , M _ { i } = m ) \log _ { 2 } P ( T _ { F } = t \mid M _ { i } = m ) } \\ { \displaystyle = H ( T _ { F } \mid M _ { i } ) . } \end{array}
$$

For non-discrete memory the outer sum is an integral with respect to its distribution; the target entropy remains discrete. No independence between target and memory is assumed.

Write $h _ { i } ( m ) = H ( T _ { F } \mid M _ { i } = m )$ . Along an actual history, the hypothetical record with response $r = R _ { i }$ is the actual $M _ { i } ,$ , so

$$
\begin{array} { c l l } { { } } & { { } } & { { \displaystyle \sum _ { i = 1 } ^ { n } \mathrm { G a i n } _ { i } ( R _ { i } ; Z _ { i } ) = \sum _ { i = 1 } ^ { n } \bigl [ h _ { i - 1 } ( M _ { i - 1 } ) - h _ { i } ( M _ { i } ) \bigr ] } } \\ { { } } & { { } } & { { = H ( T _ { F } ) - h _ { n } ( M _ { n } ) . } } \end{array}
$$

Taking the joint source/history expectation and using the preceding identity gives (41). A fixed-target expectation instead averages with a diferent distribution and does not in general yield this mutual information.

The memory decomposition (43) remains in the main text because it determines which protocol choices must be accounted for. Under (44), its context terms vanish and

$$
\mathbb { E } \sum _ { i = 1 } ^ { n } \mathrm { G a i n } _ { i } ( R _ { i } ; Z _ { i } ) = \sum _ { i = 1 } ^ { n } I ( T _ { F } ; R _ { i } \mid Z _ { i } ) \leq \ell _ { \mathrm { P T } } ( n ) ,
$$

which proves Proposition 2. This uses nested memory and the stated protocol condition, not interchangeability of $M _ { n }$ and ${ \mathbf { R } } _ { n }$

## A.6 Support and label chain rules

At each fixed context, the probability product rule gives ${ \widehat { \rho } } _ { i } ( \xi , h \mid z ) = { \widehat { \rho } } _ { i } ( \xi \mid z ) { \widehat { \rho } } _ { i } ( h \mid \xi , z )$ . Taking negative logarithms, summing over rounds, and taking expectations gives (47). For a shared ordering $\xi = \{ x _ { 1 } , \dots , x _ { m } \}$ the second factor is

$$
\widehat { \rho } _ { i } ( h \mid \xi , z ) = \prod _ { j = 1 } ^ { m } \widehat { \rho } _ { i } \big ( h ( x _ { j } ) \mid \xi , z , h ( x _ { 1 } ) , \ldots , h ( x _ { j - 1 } ) \big ) .
$$

The factors denote the corresponding conditional label probabilities derived from the same response model. This gives one label term per marked location, not independent observations or one optimiser update per term.

The information identity follows by adding and subtracting the entropy after support is known:

$$
\begin{array} { r l } & { I ( T _ { F } ; \Xi _ { i } , H _ { i } \mid Z _ { i } ) = H ( T _ { F } \mid Z _ { i } ) - H ( T _ { F } \mid Z _ { i } , \Xi _ { i } , H _ { i } ) } \\ & { \qquad = \left[ H ( T _ { F } \mid Z _ { i } ) - H ( T _ { F } \mid Z _ { i } , \Xi _ { i } ) \right] } \\ & { \qquad + \left[ H ( T _ { F } \mid Z _ { i } , \Xi _ { i } ) - H ( T _ { F } \mid Z _ { i } , \Xi _ { i } , H _ { i } ) \right] } \\ & { \qquad = I ( T _ { F } ; \Xi _ { i } \mid Z _ { i } ) + I ( T _ { F } ; H _ { i } \mid \Xi _ { i } , Z _ { i } ) . } \end{array}
$$

Neither the coding factorization nor this information decomposition assumes independence of support and labels.

## B The parametric MDL benchmark and its qualifications

Unlike the algebra in A, this asymptotic benchmark requires substantive statistical assumptions. We retain the construction to clarify its interpretation, not as a theorem about adaptive human correction streams. For broader treatments see [Grünwald, 2007, Grünwald and Roos, 2020].

Hold the discrete observation alphabet or finite-precision recording scheme fixed as N varies. All likelihoods below are probability masses of the recorded data, not density values; parameters may still be continuous. The regularity assumptions apply to this recorded-data model.

Consider a fixed regular k-parameter model for well-specified i.i.d. data, an interior true parameter with nonsingular Fisher information, and the usual smoothness and concentration conditions. Write ${ \widehat { \theta } } _ { N }$ for its maximum-likelihood fit. A suitable parameter grid at scale $N ^ { - 1 / 2 }$ in each coordinate uses approximately $\left( k / 2 \right) \log _ { 2 } N$ bits on a fixed bounded parameter region. Under regularity, rounding the fit to that grid changes the data log loss by a bounded-order amount. A suitable two-part code consequently has the benchmark expansion

$$
\ell _ { \mathrm { t w o } } ( D _ { N } ) = - \log _ { 2 } p _ { \widehat { \theta } _ { N } } ( D _ { N } ) + \frac { k } { 2 } \log _ { 2 } N + O _ { P } ( 1 ) .\tag{52}
$$

Here $O _ { P } ( 1 )$ is bounded in probability, not a uniform bound for every possible dataset. The precision, model family, and code construction are essential.

For a smooth positive prior density π near the true parameter, the Bayesian mixture code has an exact prequential factorization:

$$
q ( D _ { N } ) = \int p _ { \theta } ( D _ { N } ) \pi ( \theta ) d \theta = \prod _ { t = 1 } ^ { N } q ( d _ { t } \mid D _ { t - 1 } ) .\tag{53}
$$

Under regular Laplace-approximation conditions, its negative log probability has the same leading likelihood and $\left( k / 2 \right) \log _ { 2 } N$ terms as (52). If the remainder also has uniformly bounded absolute expectation, taking expectations yields

$$
\mathbb { E } \big [ \ell _ { \mathrm { p r e q } } ( D _ { N } ) \big ] = \mathbb { E } [ - \log _ { 2 } { p _ { \widehat { \theta } _ { N } } ( D _ { N } ) } ] + \frac { k } { 2 } \log _ { 2 } { N } + O ( 1 ) .\tag{54}
$$

Boundedness in probability alone would not justify that expected expansion.

This gives a concrete suficient coding construction; it is not a guarantee for an arbitrary plug-in training sequence. For example, unsmoothed Bernoulli maximum likelihood after the first observation assigns probability zero to the opposite outcome and can produce infinite subsequent code length.

The term describes redundancy relative to a retrospectively fitted likelihood within the stated family. It is neither all the communication required nor a universal lower bound on teacher efort. If model dimension is selected from varying $\Theta _ { k } .$ , describing the family index has a further cost. For a response model, k would be its identifiable statistical dimension, not automatically the number of segmentation-network parameters or target bits. Spatial dependence, adaptive views, misspecification, and changing teacher behaviour all require separate analysis. In particular, neither the number of rounds n nor the number of marks $W _ { n }$ can automatically replace the i.i.d. sample size N in (54).

## C Literature-selection and search record

## C.1 Question, design, and scope

The review asks when corrective spatial feedback is informative, when it becomes useful learning, and what an improvement per human efort establishes. It is a critical narrative review with purposive source selection, not a systematic review or meta-analysis. The anchor corpus combines established methods, prior reviews, representative corrective-segmentation systems, and statistical foundations; it was not prospectively sampled under a registered search. A targeted search on 14 October 2026 checked recent counterexamples and missing mechanisms. It does not establish comprehensive coverage of 2024–2026.

Discovery used a general web-search service, year terms, and targeted publisher/proceedings searches, followed by primary-source retrieval. Ranked results change over time; no total-hit or PRISMA-style count is claimed. News, aggregators, and social media supplied discovery leads only. Year terms are not verified publication-date filters: dates and publication status were checked against primary records, with preprints identified as such.

Sources were included when they clarified who chooses views, response locations, labels, or correction extent; actual versus simulated responses and their observation models; prompt-state updates, image-specific adaptation, or reusable learning; or the endpoint, denominator, and design underlying an eficiency claim. Theory and nonsegmentation studies supply mechanisms or assumptions, not direct evidence of human segmentation eficacy. General architectural advances without a relevant interaction, selection, assimilation, or evaluation mechanism were excluded; exhaustive coverage of SAM variants was outside scope. Empirical claims were deferred when the primary protocol could not be inspected. Reviews positioned the synthesis but did not replace primary evidence of individual systems’ performance.

## C.2 Exact queries

The following strings preserve the discovery and retrieval queries, including quotation marks, year terms, site restrictions, and exclusions. The targeted checks were not an exhaustive forward/backward citation search.

## Discovery

Q1 2024 2025 2026 interactive segmentation human corrective learning annotation selection retraining

Q2 2024 2025 2026 interactive segmentation human click model user study correction behavior

Q3 2024 2025 2026 medical segmentation annotation effort human in the loop correction active learning

Q4 "interactive segmentation" "human" "click" "2025" site:openaccess.thecvf.com

Q5 "corrective" "segmentation" "2025" "learning" site:arxiv.org

Q6 "segmentation" "human" "annotation time" "2026" "learning" site:pubs.rsna.org

Candidate metadata and retrieval

F1 "XpertLoop" human annotation learning 2026

F2 "SPA" "Efficient User-Preference Alignment" arxiv

F3 "SegAgent" "Imitating Human" arxiv

F4 "Segment anything in medical images" "annotation study" "10"

F5 "XpertLoop" "doi"   
F6 "Towards Clinician-Preferred Segmentation" "10.1007"   
F7 "XpertLoop" "Jaiswal" "Radiology" doi   
Mechanism and publication checks   
T1 "RClicks" "teacher" "learning"   
T2 "Iterative Teacher-Aware Learning" segmentation human   
T3 "RootPainter" "click" "model" human correction selection   
T4 "Cellpose" "human" "correction" "selection" learning   
T5 "RClicks" "learning" "human" "training" 2025 2026   
T6 "Iterative Teacher-Aware Learning" "segmentation" -site:researchgate.net   
T7 "RClicks" "retraining" segmentation   
T8 "You Point, I Learn" DOI 2025   
T9 "You Point, I Learn" "ICLR 2026" Xu Liang   
T10 "MultiVerSeg" "time" "human" ICCV 2025   
T11 MultiverSeg Scalable Interactive Segmentation Biomedical Imaging Datasets In Context Guidance   
DOI authors   
T12 "MultiverSeg" site:arxiv.org   
T13 "Active Learning for Efficient Annotation of Surgical Videos with Weak Supervision" journal DO   
T14 "Explainable Human-in-the-Loop Segmentation via Critic Feedback Signals" publication DOI   
T15 "Learning What To Ask For When" Torpey   
T16 "Learning What To Ask For When" arxiv   
T17 "Torpey" "Image Ordering" segmentation pdf   
T18 "Learning What To Ask For When" "simulat"   
T19 "Learning What To Ask For When" "context" site:openaccess.thecvf.com/content/CVPR2026W   
XpertLoop retrieval checks   
X1 XpertLoop Jaiswal 2026 annotation human expert segmentation   
X2 "XpertLoop"   
X3 "XpertLoop" "radiologist"   
X4 "XpertLoop" "Jaiswal"   
X5 "Jaiswal" "human-guided" annotation radiology 2026   
X6 site:pubs.rsna.org "Jaiswal" "annotation"   
X7 "expert-guided annotation" "doi"   
X8 "XpertLoop" segmentation -site:linkedin.com -site:xpertloop.ai   
X9 "10.1148/ryai.250200"   
X10 "Impact on Cost and Expert Time"

## C.3 Retrieval paths and screening decisions

Mechanism-focused discovery retrieved ExiL, whose references led to You Point, I Learn and MultiverSeg; their primary texts and proceedings were checked. MultiverSeg-related discovery led to Torpey et al., verified against the indexed CVF record and proceedings PDF. XpertLoop title/DOI checks led to the RSNA article and PubMed record. Dendukuri’s title search led to the preprint, author announcement, and oficial IPCAI long-abstract programme; Shaeri’s led to the preprint and an author publication-status page. These paths establish relevant contrasts, not citation-network saturation.

Included candidates with primary text inspected MedSAM [Ma et al., 2024] was included for measured annotation-pipeline time; SPA [Zhu et al., 2025a] for preference observations and tested cross-image preference-state transfer; and HiTTA [Hu et al., 2024] for cross-image assimilation under simulated preferences. SegAgent [Zhu et al., 2025b] was retained briefly as a boundary case separating constructed annotation trajectories from observed human trajectories. Dendukuri et al. [Dendukuri et al., 2026], Shaeri et al. [Shaeri et al., 2025], and ExiL [Tavangar et al., 2026] provide adjacent evidence of timed corrective learning under distinct protocols. XpertLoop [Jaiswal et al., 2026] enters Table 2 as a direct correction/retraining and expert-time study. MultiverSeg [Wong et al., 2025] supplies an in-context reuse precedent; You Point, I Learn [Xu et al., 2026] an online parameter-assimilation precedent; and Torpey et al. [Torpey et al., 2026] simulated sequencing evidence. Their inspected versions, locators, observation units, and inferential limits are recorded in D.

## Other screened hits

PRISM A Promptable and Robust Interactive Segmentation Model with Visual Prompts (PMCID: PMC12128912). The primary indexed description concerns simulated corrective prompts and iterative mask refinement. Not selected for detailed appraisal because representative fixed-model refinement methods were already included; cataloguing further prompt architectures would broaden the scope.

CASC-AI Consensus-aware Self-corrective AI Agents for Noise Cell Segmentation (arXiv:2502.07302). The primary abstract concerns ofline noisy-label consensus/weighting, not live selection or sequencing of human corrections. Not selected: annotator noise was already addressed by sources matched to the relevant assumptions.

Predictive-corrective brain MRI Rethinking Convergence in Deep Learning: The Predictive-Corrective Paradigm for Anatomy-Informed Brain MRI Segmentation (arXiv:2510.15439). The primary abstract uses “corrective” for a residual network, not human feedback. Excluded on interaction-scope grounds.

RAIL Region-Aware Instructive Learning for Semi-Supervised Tooth Segmentation in CBCT (arXiv:2505.03538). In the primary abstract, teacher/student denotes neural training components, not human example selection. Excluded from the human-interaction synthesis.

Segment Anything for Microscopy The 2025 study [Archit et al., 2025] was already included in the detailed evidence matrix; the discovery hit does not constitute a separate study. Its user study is relevant to efort.

McSweeney et al. Transfer learning for data-eficient abdominal muscle segmentation with convolutional neural networks (doi:10.1002/mp.15533, 2022). The primary paper, Sections 2.1–2.4, uses upfront expert annotations, ofline transfer learning on random training subsets, and a ten-radiographer interobserver benchmark. It evaluates no interactive correction, adaptive selection, or annotation-time intervention and was not selected for corrective feedback comparisons.

Coverage limits The included sources cover corrective retraining, in-context reuse, response simulation, expert-guided adaptation, and ordering. They support neither exhaustive coverage nor a claim that selection/history has no precedents. Primary full texts support the sequencing and adaptation comparisons; XpertLoop was appraised from the publisher’s full text. No exhaustive database export or complete citationnetwork sweep was performed. Independent appraisal of consequential evidence classifications remains desirable. These are limits on coverage and confidence, not evidence against individual studies.

## D Study selection and qualitative appraisal

## D.1 Comparison and extraction rule

The studies in Table 2 were selected to illustrate distinct links between selection, assimilation, and efort: reusable classification, sparse corrective learning, longitudinal clinical correction, current-image adaptation, completed-mask retraining, workflow integration, human-response simulation, prompting plus reuse, costaware acquisition, and clinically timed iterative retraining. This is purposive coverage of mechanisms, not a representative sample of studies or applications.

Comparisons were selected for those links, not to maximize efects or rank systems. The appraisal records the unit, participants, and horizon; intervention, comparator, and controls; endpoint, cost, and uncertainty unit; and supported inference. Alternative conditions qualify the selected comparisons but do not constitute an exhaustive sensitivity audit. NR means not reported for the selected comparison, not an observed null result. Uncontrolled order and bundled interventions are design limitations distinct from missing reporting. SD, SE, and confidence intervals are not interchangeable, and none automatically quantifies variability across human users. Numerical outcomes for the main comparisons remain in Table 2; the records below supply source locators, selection rationale, and additional qualifications. Page numbers refer to printed pagination unless explicitly identified as PDF pages.

## D.2 Appraisal of the main comparisons

Crayons [Fails and Olsen, 2003] Source: p. 44, Tables 1–2 and Evaluation. Ten participants in two counterbalanced order groups performed four visual-classification tasks. The endpoint was elapsed time to a reusable classifier exceeding 97.5% agreement with a reference classifier. The skin task contrasts stronger generalization (CWSS) with faster individual fits (MSSS), illustrating that fitting speed and interactive eficiency difer. Intervals/tests are NR. Diferences are smaller for simpler tasks; paper cards slightly favour MSSS in one order. Reference-classifier agreement is not a population-risk test.

RootPainter [Smith et al., 2022a] Source: pp. 780–782, Tables 2–3 and Fig. 4. Two author-users trained on three plant-image datasets, with corrective annotation always preceding duration-matched dense annotation. Held-out downstream measurements assess the final reusable predictors. The selected comparison directly matches human time; it does not establish superiority. The reported means and SEs aggregate six user–dataset runs, and the mixed-efects strategy test is nonsignificant at the stated 0.05 level. Dense annotation wins for one nodule user; root results are close. SE is not an efect confidence interval, and order was uncontrolled.

RootPainter3D [Smith et al., 2022b] Source: pp. 463, 466–467, Fig. 5 and Table 1. One physician contoured 933 CT hearts with cumulative training; twenty were also contoured in Eclipse. The final-ten comparison demonstrates late-session completion savings, including inspection/correction but excluding inactivity of at least twenty seconds. Timing confidence intervals are NR. Early assisted cases are slower, and final-case times omit accumulated initialization efort. Case order, user experience, and training evolve together. Agreement between corrected outputs and manual contours does not establish independent accuracy of every uncorrected prediction.

BIFSeg [Wang et al., 2018] Source: final TMI 2018 article, pp. 1567, 1571–1572, Tables V–VI. Two clinical users evaluated fetal and brain-tumour MRI refinement. The selected brain comparison fixes initial predictions and one round of scribbles, comparing BIFSeg with CRF. Shared supervision isolates diferences in assimilation more directly than changing a query policy. The tables also compare unsupervised refinement and unweighted fine-tuning: assistance works without extra scribbles, but supervised refinement improves further. Parameters adapt to the current image, not subsequent cases. The table’s ± dispersion is not explicitly identified as SD or SE.

Cellpose 2.0 [Pachitariu and Stringer, 2022] Source: pp. 1639–1640, Fig. 5. One annotator supplied five training and three test images in each of two categories. The online/pretrained versus ofline comparison connects completed-mask retraining to later-image prediction; cost-ratio intervals are NR. The factorial comparison also includes online/from-scratch models, which require 370 and 528 drawn objects, showing a material contribution from initialization. Repeated test annotation measures within-annotator consistency, not generalization across annotators. Drawn-object counts do not establish equal session duration.

MONAI Label [Diaz-Pinto et al., 2024] Source: PDF pp. 7–8, Section 5.1, Tables 1–2. The spleen-CT demonstration adds corrected volumes across annotation stages. It was selected as workflow evidence, not an acquisition-policy experiment. Manual paintbrush, contour, and scribble baselines change the assistance comparison; training time is reported separately. Uncertainty intervals and user-population variability are NR. Six validation volumes in the prose versus nine in Table 2, together with inconsistent timing totals, prevent precise reconstruction of a cost-efect estimate. These are reporting ambiguities, not measured evidence against eficiency.

RClicks [Antonov et al., 2024] Source: PDF pp. 7–9, Section 5 and Table 4; collection in Section 3. Real clicks train a response-location model; simulations compare clicking strategies across segmentation models using clicks to 90% IoU, capped at twenty. This was selected to show that response assumptions alter benchmark dificulty. SD concerns sampled click-group variability; timing uncertainty is inapplicable because human time was not measured. Efects vary by model and dataset. Selected model-error displays supply later-click tasks, rather than unconstrained longitudinal human sessions. This comparison does not improve a reusable learner.

Microscopy SAM [Archit et al., 2025] Source: p. 586, User study 1; p. 588, Fig. 6a. Five organoid annotators compared fine-tuned µSAM and CellPose. The comparison distinguishes annotation completion from subsequent reusable-model outcomes; it does not prove a significant rank reversal. Test mSA dispersion is SD across annotators. The alternative manual condition averages 6.7 ± 0.9 seconds/object and mSA $0 . 4 9 \pm 0 . 0 7$ . Initialization and downstream training belong to the pipeline, whereas per-object annotation rates omit complete training-plus-human cost.

CEREALS [Mackowiak et al., 2018] Source: PDF pp. 8–10, Section 4, Figs. 4–5. Five Cityscapes simulations use a label oracle and recorded polygon-click cost. The selected 128 × 128-region comparison isolates learned cost-aware versus entropy-only acquisition under that simulation, not human time. Thresholdcost intervals are NR. Oracle-cost acquisition reaches the same target at 14.68% of annotation cost, exposing the gap introduced by cost prediction. The target is 95% of full-data mIoU 0.605, not 95% absolute mIoU. Region size and the recorded-click cost model condition the comparison.

XpertLoop [Jaiswal et al., 2026] Source: published RSNA primary text, pp. 2, 5, 8–11 and Fig. 5; annotation-time and cost analyses. The study uses 1,941 CT/MRI scans from ten datasets and 57 nnU-Net models. It retrains nnU-Net from scratch on accumulated corrected masks, comparing random and size–entropy selection. It was selected because live expert timing and reusable-model training coexist. One radiologist per timed task annotated diferent randomly selected case subsets manually or corrected optimized-model predictions. Maximum kidney/tumour savings must not be generalized across tasks. Modelled tumour cost uncertainty includes no saving and derives from assumed wage/time distributions, not multiple clinicians. Infrastructure and development costs are excluded. This is neither a matched end-to-end training-budget comparison nor an isolated benefit of modelling human selection. The acquisition analysis (pp. 3, 7, 10–11, Tables 2–3 and Discussion) reports fewer loops to strong performance for size–entropy than random selection in the reported organ tasks; plain entropy can underperform in the ablations, and suitability depends on the dataset. Size–entropy adds inference over unlabelled cases. Mean test-set Dice improvement below 2% determines stopping, so that set informs the procedure rather than providing an untouched evaluation. Loop counts alone do not compare total human-plus- computational cost at independent evaluation.

## D.3 Adjacent cases and source checks

These sources clarify nearby mechanisms without entering a pooled numerical comparison. Publication status and available evidence are stated separately; absence of measured human time is not treated as a null benefit.

Large-scale interactive segmentation [Benenson et al., 2019] Source: CVPR 2019 work, inspected arXiv PDF pp. 5–8, Sections 4.1–4.2 and 5.1–5.4, Figs. 7–9. A pool of 100 full-time annotators supplies up to four clicks per round in three rounds separated by days. Zero-click judgments take about four seconds; four-click rounds are faster than three-click rounds. Mean round times increase from 10.8 to 11.7 to 11.9 seconds, consistent with smaller errors taking longer to locate. The proposed explanation of the three- versusfour-click pattern is a hypothesis, not an identified causal efect; timing-efect confidence intervals are NR. A ranker uses click counts, round number, mask change, and click-to-mask distances to predict completed-mask quality (Sections 4.2 and 5.4, Fig. 8). These results support accounting for inspection, changing correction behaviour, and history-dependent quality signals within the protocol, not universal teaching stages. A separate downstream experiment trains on completed masks (Section 5.3) without testing whether using the ranker improves reusable learning. The headline speed comparison uses 34 measured seconds per corrected mask against 108 estimated seconds for historical COCO polygons, scaling historical speed linearly by vertex count (Section 5.1); it is not a matched-user timing trial.

MedSAM [Ma et al., 2024] Source: final 2024 Nature Communications article; Results, “MedSAM can improve the annotation eficiency”; Methods, “Human annotation study”; Fig. 5b. Two radiologists annotated ten adrenal-tumour cases manually and, after a one-week interval, with model assistance. Assisted time includes initial markers, inference, and manual refinement. This is an actual human-time comparison of annotation pipelines; the fixed order and small participant sample limit causal and population claims. It does not demonstrate reusable learning from those corrections.

SPA [Zhu et al., 2025a] Source: final ICCV 2025 paper, Sections 3.1–3.5 and 4.3.2, pp. 23735, 23737– 23738; oficial supplement for protocol details. Multiple-choice mask selection updates a latent preference distribution; training simulates feedback using multi-rater masks. Section 3.4 initializes preference state per image, while Section 4.3.2 also transfers user-adapted GMM parameters across images (kidney Dice 91.68% to 92.43%). A separate study with five medical participants and 100 fundus images reports shorter interactions than MedSAM. That user study does not isolate transfer benefits. Mask choice is a diferent response unit from spatial correction: reusable preference state is not a learned likelihood of natural support choice. The latent-Gaussian/i.i.d. convergence claim is not a guarantee for general spatial correction behaviour.

HiTTA [Hu et al., 2024] Source: arXiv:2405.08270v1, Sections 2.1, 2.3, 3.1–3.4 and Table 2; no peerreviewed record was verified in the search. Corrected masks update a model used on subsequent images. Existing ophthalmologists’ masks assigned to diferent domains simulate preferences; these are not live timed corrections. Section 3.2 (p. 6) evaluates the better prediction from the original and preference heads. Domain-preference DSC is 91.14% versus 87.21% without human feedback, with no uncertainty intervals. The result supports adaptation under this simulated protocol, not demonstrated human-efort eficiency or an independently validated, deployable head-selection rule.

SegAgent [Zhu et al., 2025b] Source: final CVPR 2025 paper, Sections 3 and 4.1 (p. 3689), and the trajectory-generation algorithm. An automated click simulator generates training trajectories from referencemask errors; these are not recorded human trajectories. Successful imitation establishes agent performance under that generator, not realism of human-response probabilities or improvement in reusable segmentation learning at matched human efort. This is the boundary case for which the study was included.

Surgical-video annotation [Dendukuri et al., 2026] Source: arXiv:2607.13237v1, Sections 2.2, 3.2– 3.2.2, Tables 1–3. The inspected full paper remains a preprint; a related IPCAI 2026 long-abstract acceptance was verified separately against the author announcement and oficial programme, not treated as full-paper publication. Two trained annotators supply corrections for later training; acquisition is random with classbalancing weights. AL-10 averages 3.56 minutes/video versus 6.34 manually, but SAM2 assistance takes 1.60. CorLoc gains over upfront supervision depend on budget and loss weighting; AL-2 can underperform. Video counts difer between prose and Table 1. Diferent annotation aids enter the comparison, precluding attribution to selection modelling alone or a general 50% efort-saving claim.

Critic feedback [Shaeri et al., 2025] Source: arXiv:2510.09945v1, Sections 3.2.4, 3.3, 4.4 and 5.1–5.4. The author page lists WACV submission, not verified acceptance. Twelve participants corrected twenty images each with randomized task order. Corrected-label fitting and similarity-based propagation accompany the interface. Reported times are 24 seconds/image versus 54 for click refinement and 95 for pixel annotation;

timing intervals are NR. Editing, feedback, propagation, and fitting change together. Section 5 describes variability as confidence intervals, whereas Table 1 labels it seed SD; seed variability does not quantify a human-time efect. This is a qualified cross-image feedback precedent, not causal identification or an isolated natural-selection model.

ExiL [Tavangar et al., 2026] Source: accepted MICCAI 2026 full text, LNCS 16884, Sections 2.2–2.5 and 3, especially Section 2.5 (pp. 6–7) and Table 2. The oficial proceedings page had not supplied a DOI or final pagination at the check. Synthetic brush trajectories teach refinement; expert-approved masks and intermediate correction states update the deployed model. One expert timed 360 frames from three volunteers with randomized frame order and common stopping/acceptance guidance. This is live within-annotator feasibility evidence, with limited transfer across users. The synthetic autofill comparison restricts SAM to one refinement; the timed SAM-assisted workflow instead reports 2.9 autofill cycles/frame. A completion trajectory is modelled, not a likelihood of natural human support choices.

MultiverSeg [Wong et al., 2025] Source: arXiv:2412.15058v2 of the ICCV 2025 work, Sections 3 and 5–6; publication metadata checked against the published record. Completed image–mask pairs enter the context for subsequent images without within-session weight retraining. Simulated clicks and scribbles assess how earlier completed cases aid later ones. This is evidence of reusable state, not measured human-time savings: cross-image learning need not entail parameter updating.

You Point, I Learn [Xu et al., 2026] Source: arXiv:2503.06717v4 (20 April 2026), marked as published at ICLR 2026 and checked against the oficial proceedings; Sections 2.1, 2.4, 3.2 and Appendices A.2, A.4. Midand post-interaction parameter updates reuse corrections across images under distribution shift. Inference clicks are simulated in the largest reference-mask error component; live human time is not measured and natural correction preferences are not estimated. Appendix A.4 shows that post-interaction gains can diminish by the last click, so benefits depend on evaluation horizon and click budget.

Image ordering [Torpey et al., 2026] Source: final CVPR Workshops (PHAROS-AIF-MIH) 2026 paper, Sections 4.2–4.5, 5.2 and 6.4, pp. 6420–6425, Tables 1–2. Ten-image subsets are evaluated under 100 random orders with fixed MultiverSeg weights. Completed predictions or reference masks enter later context; deterministic clicks follow reference-mask errors. Across ten datasets with predicted-mask context, the sampled best–worst Dice range is 6.07 percentage points larger at five than twenty clicks (95% CI 5.62–6.55). IQR comparisons also show order sensitivity. Confidence intervals bootstrap image subsets with hierarchical aggregation, not users. Uncertainty ordering averaged across starts has mixed efects versus random, with most intervals crossing zero; selecting the best of ten starts retrospectively is more favourable. This separates order sensitivity from a reliable acquisition rule. Click ceilings condition cost comparisons; neither count reductions nor oracle-start efects establish human-time savings. Order changes subsequent corrections, so this does not replay a fixed correction multiset.

Prior-review positioning [Marinov et al., 2024] Source: final TPAMI 2024 review, pp. 11010–11013, Sections V-A3, V-D and VI-B1–B3. Beyond training/application categories, Marinov et al. appraise absent or weak baselines, fragmented comparisons, narrow accuracy metrics, and scarce user-centred measures. Section VI-B2 discusses the dificulty of simulating expert annotation styles and recommends diverse simulated users; VI-B3 distinguishes annotation eficiency from interaction eficacy. These are established evaluation concerns. This review’s additional role is to formalize assumptions connecting selected responses, learner updates, and declared efort. This entry positions a review; it is not primary efect evidence.

Synthesis boundary These contrasts support protocol-specific conclusions about fitting, completion, response simulation, and cost-aware acquisition. No aggregate efect is calculated across endpoints, uncertainty units, or cost measures. The absence of one complete selection-model-to-live-eficiency demonstration within this set is not evidence that none exists elsewhere.

## References

[Amershi et al., 2014] Amershi, S., Cakmak, M., Knox, W. B., and Kulesza, T. (2014). Power to the people: The role of humans in interactive machine learning. AI Magazine, 35(4):105–120.

[Antonov et al., 2024] Antonov, A., Moskalenko, A., Shepelev, D., Krapukhin, A., Soshin, K., Konushin, A., and Shakhuro, V. (2024). RClicks: Realistic click simulation for benchmarking interactive segmentation. In Advances in Neural Information Processing Systems, volume 37.

[Archit et al., 2025] Archit, A., Freckmann, L., Nair, S., Khalid, N., Hilt, P., Rajashekar, V., Freitag, M., Teuber, C., Spitzner, M., Tapia Contreras, C., Buckley, G., von Haaren, S., Gupta, S., Grade, M., Wirth, M., Schneider, G., Dengel, A., Ahmed, S., and Pape, C. (2025). Segment anything for microscopy. Nature Methods, 22(3):579–591.

[Arganda-Carreras et al., 2017] Arganda-Carreras, I., Kaynig, V., Rueden, C., Eliceiri, K. W., Schindelin, J., Cardona, A., and Seung, H. S. (2017). Trainable Weka segmentation: A machine learning tool for microscopy pixel classification. Bioinformatics, 33(15):2424–2426.

[Bekker and Davis, 2020] Bekker, J. and Davis, J. (2020). Learning from positive and unlabeled data: A survey. Machine Learning, 109:719–760.

[Benenson et al., 2019] Benenson, R., Popov, S., and Ferrari, V. (2019). Large-scale interactive object segmentation with human annotators. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11700–11709.

[Bengio et al., 2009] Bengio, Y., Louradour, J., Collobert, R., and Weston, J. (2009). Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 41–48. ACM.

[Berg et al., 2019] Berg, S., Kutra, D., Kroeger, T., Straehle, C. N., Kausler, B. X., Haubold, C., Schiegg, M., Ales, J., Beier, T., Rudy, M., Eren, K., Cervantes, J. I., Xu, B., Beuttenmueller, F., Wolny, A., Zhang, C., Koethe, U., Hamprecht, F. A., and Kreshuk, A. (2019). ilastik: Interactive machine learning for (bio)image analysis. Nature Methods, 16:1226–1232.

[Blier and Ollivier, 2018] Blier, L. and Ollivier, Y. (2018). The description length of deep learning models. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

[Boykov and Jolly, 2001] Boykov, Y. Y. and Jolly, M.-P. (2001). Interactive graph cuts for optimal boundary & region segmentation of objects in N-D images. In Proceedings of the Eighth IEEE International Conference on Computer Vision, volume 1, pages 105–112.

[Cakmak and Thomaz, 2014] Cakmak, M. and Thomaz, A. L. (2014). Eliciting good teaching from humans for machine learners. Artificial Intelligence, 217:198–215.

[Chen et al., 2022] Chen, X., Zhao, Z., Zhang, Y., Duan, M., Qi, D., and Zhao, H. (2022). FocalClick: Towards practical interactive image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1300–1309.

[Cohn et al., 1996] Cohn, D. A., Ghahramani, Z., and Jordan, M. I. (1996). Active learning with statistical models. Journal of Artificial Intelligence Research, 4:129–145.

[Dawid, 1984] Dawid, A. P. (1984). Present position and potential developments: Some personal views. statistical theory: The prequential approach. Journal of the Royal Statistical Society. Series A (General), 147(2):278–292.

[De Lange et al., 2022] De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., and Tuytelaars, T. (2022). A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7):3366–3385.

[Dendukuri et al., 2026] Dendukuri, M., Jogan, M., Hashimoto, D. A., and Liao, G. (2026). Active learning for eficient annotation of surgical videos with weak supervision. arXiv preprint arXiv:2607.13237.

[Diaz-Pinto et al., 2024] Diaz-Pinto, A., Alle, S., Nath, V., Tang, Y., Ihsani, A., Asad, M., Pérez-García, F., Mehta, P., Li, W., Flores, M., Roth, H. R., Vercauteren, T., Xu, D., Dogra, P., Ourselin, S., Feng, A., and Cardoso, M. J. (2024). MONAI Label: A framework for AI-assisted interactive labeling of 3D medical images. Medical Image Analysis, 95:103207.

[Dudley and Kristensson, 2018] Dudley, J. J. and Kristensson, P. O. (2018). A review of user interface design for interactive machine learning. ACM Transactions on Interactive Intelligent Systems, 8(2):8:1–8:37.

[Equitz and Cover, 1991] Equitz, W. H. R. and Cover, T. M. (1991). Successive refinement of information. IEEE Transactions on Information Theory, 37(2):269–275.

[Fails and Olsen, 2003] Fails, J. A. and Olsen, Jr., D. R. (2003). Interactive machine learning. In Proceedings of the 8th International Conference on Intelligent User Interfaces, pages 39–45.

[Farquhar et al., 2021] Farquhar, S., Gal, Y., and Rainforth, T. (2021). On statistical bias in active learning: How and when to fix it. In International Conference on Learning Representations.

[Fiebrink et al., 2011] Fiebrink, R., Cook, P. R., and Trueman, D. (2011). Human model evaluation in interactive supervised learning. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 147–156. ACM.

[Gneiting and Raftery, 2007] Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378.

[Golovin and Krause, 2011] Golovin, D. and Krause, A. (2011). Adaptive submodularity: Theory and applications in active learning and stochastic optimization. Journal of Artificial Intelligence Research, 42:427–486. Consulted with the corrected arXiv revision of 6 December 2017.

[Golovin et al., 2010] Golovin, D., Krause, A., and Ray, D. (2010). Near-optimal Bayesian active learning with noisy observations. In Advances in Neural Information Processing Systems, volume 23, pages 766–774. Curran Associates, Inc.

[Grünwald and Roos, 2020] Grünwald, P. and Roos, T. (2020). Minimum description length revisited. International Journal of Mathematics for Industry, 11(1):1930001.

[Grünwald, 2007] Grünwald, P. D. (2007). The Minimum Description Length Principle. MIT Press.

[Haertel et al., 2015] Haertel, R., Ringger, E. K., Felt, P., and Seppi, K. (2015). An analytic and empirical evaluation of return-on-investment-based active learning. In Proceedings of the 9th Linguistic Annotation Workshop, pages 11–20. Association for Computational Linguistics.

[Houlsby et al., 2012] Houlsby, N., Hernández-Lobato, J. M., Huszár, F., and Ghahramani, Z. (2012). Collaborative Gaussian processes for preference learning. In Advances in Neural Information Processing Systems, volume 25. Curran Associates, Inc.

[Houlsby et al., 2011] Houlsby, N., Huszár, F., Ghahramani, Z., and Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745.

[Howard et al., 2021] Howard, S. R., Ramdas, A., McAulife, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080.

[Hu et al., 2024] Hu, S., Liao, Z., Liu, Z., and Xia, Y. (2024). Towards clinician-preferred segmentation: Leveraging human-in-the-loop for test time adaptation in medical image segmentation. arXiv preprint arXiv:2405.08270.

[Jaiswal et al., 2026] Jaiswal, A., Rinneburger, M., Meyer, F., Arjune, S., Oberlinkels, L., Wawer Matos Reimer, P. A., Lotter-Becker, L., Akünal, Ü., Bujotzek, M., Denner, S., Maier-Hein, K., Stepansky, L., May, M. S., Habert, M., Köhn, A., Schöneck, M., Wawer Matos Reimer, R. P., Müller, R.-U., Lennartz, S., Große Hokamp, N., Bucher, A. M., Persigehl, T., and Caldeira, L. L. (2026). Impact on cost and expert time of data-eficient deep learning for medical image segmentation. Radiology: Artificial Intelligence, 8(4):e250200.

[Kendall and Gal, 2017] Kendall, A. and Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? In Advances in Neural Information Processing Systems, volume 30.

[Kirillov et al., 2023] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., and Girshick, R. (2023). Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

[Kirsch et al., 2019] Kirsch, A., van Amersfoort, J., and Gal, Y. (2019). BatchBALD: Eficient and diverse batch acquisition for deep Bayesian active learning. In Advances in Neural Information Processing Systems, volume 32, pages 7026–7037. Curran Associates, Inc.

[Kohl et al., 2018] Kohl, S., Romera-Paredes, B., Meyer, C., De Fauw, J., Ledsam, J. R., Maier-Hein, K., Eslami, S. M. A., Jimenez Rezende, D., and Ronneberger, O. (2018). A probabilistic U-Net for segmentation of ambiguous images. In Advances in Neural Information Processing Systems, volume 31, pages 6965–6975. Curran Associates, Inc.

[Lakkaraju et al., 2017] Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J., and Mullainathan, S. (2017). The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 275–284.

[Liu et al., 2023] Liu, Q., Xu, Z., Bertasius, G., and Niethammer, M. (2023). SimpleClick: Interactive image segmentation with simple vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22290–22300.

[Liu et al., 2017] Liu, W., Dai, B., Humayun, A., Tay, C., Yu, C., Smith, L. B., Rehg, J. M., and Song, L. (2017). Iterative machine teaching. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2149–2158.

[Liu et al., 2018] Liu, W., Dai, B., Li, X., Liu, Z., Rehg, J. M., and Song, L. (2018). Towards black-box iterative machine teaching. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 3141–3149.

[Ma et al., 2024] Ma, J., He, Y., Li, F., Han, L., You, C., and Wang, B. (2024). Segment anything in medical images. Nature Communications, 15:654.

[MacGlashan et al., 2017] MacGlashan, J., Ho, M. K., Loftin, R., Peng, B., Wang, G., Roberts, D. L., Taylor, M. E., and Littman, M. L. (2017). Interactive learning from policy-dependent human feedback. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2285–2294.

[Mackowiak et al., 2018] Mackowiak, R., Lenz, P., Ghori, O., Diego, F., Lange, O., and Rother, C. (2018). CEREALS: Cost-efective region-based active learning for semantic segmentation. In British Machine Vision Conference.

[Marinov et al., 2024] Marinov, Z., Jäger, P. F., Egger, J., Kleesiek, J., and Stiefelhagen, R. (2024). Deep interactive segmentation of medical images: A systematic review and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10998–11018.

[Monteiro et al., 2020] Monteiro, M., Le Folgoc, L., Coelho de Castro, D., Pawlowski, N., Marques, B., Kamnitsas, K., van der Wilk, M., and Glocker, B. (2020). Stochastic segmentation networks: Modelling spatially correlated aleatoric uncertainty. In Advances in Neural Information Processing Systems, volume 33, pages 12756–12767. Curran Associates, Inc.

[Nokleby et al., 2015] Nokleby, M., Beirami, A., and Calderbank, R. (2015). A rate-distortion framework for supervised learning. In IEEE 25th International Workshop on Machine Learning for Signal Processing, pages 1–6.

[Pachitariu and Stringer, 2022] Pachitariu, M. and Stringer, C. (2022). Cellpose 2.0: How to train your own model. Nature Methods, 19:1634–1641.

[Perdomo et al., 2020] Perdomo, J., Zrnic, T., Mendler-Dünner, C., and Hardt, M. (2020). Performative prediction. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7599–7609.

[Ren et al., 2021] Ren, P., Xiao, Y., Chang, X., Huang, P.-Y., Li, Z., Gupta, B. B., Chen, X., and Wang, X. (2021). A survey of deep active learning. ACM Computing Surveys, 54(9):180:1–180:40.

[Rissanen, 1978] Rissanen, J. (1978). Modeling by shortest data description. Automatica, 14(5):465–471.

[Rother et al., 2004] Rother, C., Kolmogorov, V., and Blake, A. (2004). GrabCut: Interactive foreground extraction using iterated graph cuts. ACM Transactions on Graphics, 23(3):309–314.

[Rubin, 1976] Rubin, D. B. (1976). Inference and missing data. Biometrika, 63(3):581–592.

[Sener and Savarese, 2018] Sener, O. and Savarese, S. (2018). Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations.

[Settles, 2009] Settles, B. (2009). Active learning literature survey. Technical Report 1648, University of Wisconsin–Madison.

[Shaeri et al., 2025] Shaeri, P., Woo, R. T., Mohammadpour, Y., and Middel, A. (2025). Explainable human-in-the-loop segmentation via critic feedback signals. arXiv preprint arXiv:2510.09945.

[Shafto et al., 2014] Shafto, P., Goodman, N. D., and Grifiths, T. L. (2014). A rational account of pedagogical reasoning: Teaching by, and learning from, examples. Cognitive Psychology, 71:55–89.

[Shivaswamy and Joachims, 2015] Shivaswamy, P. and Joachims, T. (2015). Coactive learning. Journal of Artificial Intelligence Research, 53:1–40.

[Smith et al., 2022a] Smith, A. G., Han, E., Petersen, J., Olsen, N. A. F., Giese, C., Athmann, M., Dresbøll, D. B., and Thorup-Kristensen, K. (2022a). RootPainter: Deep learning segmentation of biological images with corrective annotation. New Phytologist, 236(2):774–791.

[Smith et al., 2022b] Smith, A. G., Petersen, J., Terrones-Campos, C., Berthelsen, A. K., Forbes, N. J., Darkner, S., Specht, L., and Vogelius, I. R. (2022b). RootPainter3D: Interactive-machine-learning enables rapid and accurate contouring for radiotherapy. Medical Physics, 49(1):461–473.

[Tavangar et al., 2026] Tavangar, A., Chiu, L. K., Khodashenas, H., Berry, G. K., and Hooshiar, A. (2026). Expert-like bone ultrasound segmentation through expert-in-the-loop mask-conditioned progressive learning. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2026, volume 16884 of Lecture Notes in Computer Science. Springer Nature Switzerland.

[Top et al., 2011] Top, A., Hamarneh, G., and Abugharbieh, R. (2011). Active learning for interactive 3D image segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2011, volume 6893 of Lecture Notes in Computer Science, pages 603–610.

[Torpey et al., 2026] Torpey, G., Guttag, J., and Wong, H. E. (2026). Learning what to ask for when: Image ordering for in-context interactive medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 6418–6426.

[Vezhnevets et al., 2012] Vezhnevets, A., Buhmann, J. M., and Ferrari, V. (2012). Active learning for semantic segmentation with expected change. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3162–3169.

[Wang et al., 2018] Wang, G., Li, W., Zuluaga, M. A., Pratt, R., Patel, P. A., Aertsen, M., Doel, T., David, A. L., Deprest, J., Ourselin, S., and Vercauteren, T. (2018). Interactive medical image segmentation using deep learning with image-specific fine tuning. IEEE Transactions on Medical Imaging, 37(7):1562–1573.

[Wang et al., 2020] Wang, T., Qi, S., Jia, Z., Sun, Q., Fu, P., and Ge, Q. (2020). Error-tolerant label prior for interactive image segmentation. Information Sciences, 538:384–395.

[Wong et al., 2025] Wong, H. E., Gonzalez Ortiz, J. J., Guttag, J., and Dalca, A. V. (2025). MultiverSeg: Scalable interactive segmentation of biomedical imaging datasets with in-context guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20966–20980.

[Wu et al., 2022] Wu, X., Xiao, L., Sun, Y., Zhang, J., Ma, T., and He, L. (2022). A survey of human-in-theloop for machine learning. Future Generation Computer Systems, 135:364–381.

[Xu et al., 2026] Xu, W., Liang, Z., Anthony, H., Ibrahim, Y., Cohen, F., Yang, G., and Kamnitsas, K. (2026). You point, I learn: Online adaptation of interactive segmentation models for handling distribution shifts in medical imaging. In Proceedings of the International Conference on Learning Representations.

[Yan et al., 2011] Yan, Y., Rosales, R., Fung, G., and Dy, J. G. (2011). Active learning from crowds. In Proceedings of the 28th International Conference on Machine Learning.

[Yuan et al., 2021] Yuan, L., Zhou, D., Shen, J., Gao, J., Chen, J. L., Gu, Q., Wu, Y. N., and Zhu, S.-C. (2021). Iterative teacher-aware learning. In Advances in Neural Information Processing Systems, volume 34.

[Yuan and Zhu, 2023] Yuan, L. and Zhu, S.-C. (2023). Communicative learning: A unified learning formalism. Engineering, 25(6):77–100.

[Zhang et al., 2023] Zhang, L., Tanno, R., Xu, M., Huang, Y., Bronik, K., Jin, C., Jacob, J., Zheng, Y., Shao, L., Ciccarelli, O., Barkhof, F., and Alexander, D. C. (2023). Learning from multiple annotators for medical image segmentation. Pattern Recognition, 138:109400.

[Zhu et al., 2025a] Zhu, J., Wu, J., Ouyang, C., Kamnitsas, K., and Noble, J. A. (2025a). SPA: Eficient userpreference alignment against uncertainty in medical image segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23731–23740.

[Zhu et al., 2025b] Zhu, M., Tian, Y., Chen, H., Zhou, C., Guo, Q., Liu, Y., Yang, M., and Shen, C. (2025b). SegAgent: Exploring pixel understanding capabilities in MLLMs by imitating human annotator trajectories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3686–3696.

[Zhu et al., 2018] Zhu, X., Singla, A., Zilles, S., and Raferty, A. N. (2018). An overview of machine teaching. arXiv preprint arXiv:1801.05927.