# CANDLE: CORTICAL NULL-SPACE DECOMPOSITION FOR NONINVASIVE BRAIN SOURCE IMAGING

Shuntaro Suzuki, Yuiga Wada, Komei Sugiura

Keio University

{shuntaro20021227, yuiga, komei.sugiura}@keio.jp

## ABSTRACT

Electrophysiological source imaging (ESI) aims to estimate cortical source activity from noninvasive electrophysiological measurements such as electroencephalogram (EEG). However, ESI is fundamentally ill-posed because source activity is substantially higher-dimensional than sensor observations, resulting in non-unique solutions. Recent learning-based approaches address this ambiguity by learning data-driven source priors, yet they often struggle to generalize across subject-specific cortical geometries. To address this, we propose CANDLE, a learning-based ESI model that estimates source activity on subject-specific cortical geometries. CANDLE learns a prior over the null space induced by the source-to-sensor mapping derived from T1-weighted MRI, restricting learning to unobservable source components while preserving geometric constraints. To train CANDLE, we develop a whole-brain simulator spanning over 1,100 subjectspecific cortical geometries with source configurations derived from over 26,000 statistical brain maps. Trained exclusively on simulated data, CANDLE outperformed prior ESI methods on simulated source activity estimation and generalized to two empirical tasks: (i) intracranial stimulation localization from simultaneously recorded scalp EEG and (ii) epileptogenic zone estimation from presurgical interictal EEG. Our project page is available at https://candle-esi.pages.dev.

## 1 INTRODUCTION

Monitoring whole-brain neural activity at both high temporal and spatial resolution is central to a broad range of applications, from the diagnosis of neurological disorders (Cao et al., 2022) to the development of brain-computer interfaces (BCIs) (Suzuki et al., 2026; Jayalath & Jones, 2026). This motivates the development of neuroimaging techniques capable of faithfully capturing such neural dynamics. Functional magnetic resonance imaging (fMRI) provides high spatial but limited temporal resolution and measures neural activity indirectly, whereas invasive electrophysiological techniques such as electrocorticography (ECoG) and stereoelectroencephalography (SEEG) provide high-fidelity recordings at fine temporal scales but require neurosurgical implantation and offer spatially restricted coverage (He & Liu, 2008; Momi et al., 2025). Noninvasive electrophysiological modalities such as electroencephalography (EEG) and magnetoencephalography (MEG) therefore constitute a promising avenue for whole-brain monitoring at high temporal resolution (Feng et al., 2025). Nevertheless, EEG/MEG exhibit limited spatial resolution at the sensor level, owing to relatively sparse sensor sampling and substantial spatial mixing of neural signals (i.e., volume conduction and field spread) (Wens, 2023).

Electrophysiological source imaging (ESI), which seeks to recover cortical source activity from EEG/MEG signals, has been extensively studied as a means of overcoming this limitation (Sun et al., 2022; Wang et al., 2026). By reconstructing neural activity in cortical source space, ESI holds the promise of whole-brain monitoring with both high temporal and spatial resolution. The utility of ESI has further been suggested for applications such as presurgical localization of epileptogenic zones (Corona et al., 2023) and improved decoding in BCIs (Gideoni et al., 2024).

ESI, however, is a fundamentally ill-posed problem: the source activity to be estimated is substantially higher-dimensional than the observed EEG/MEG, and the solution is consequently nonunique. Conventional ESI methods (Van Veen et al., 1997; Dale et al., 2000) therefore typically require careful tuning of regularization schemes and source-space constraints for individual applications. This reliance on manual constraints not only limits the broad applicability of ESI but also undermines reproducibility. To alleviate these limitations, learning-based ESI models (Hecker et al., 2021; Sun et al., 2022) have recently been proposed to learn data-driven priors over source activity.

![](images/ce89eeff4d5c9968d7676f8dead7a6c0df0d0f5539aa8287d4d9780f8034a6e5.jpg)  
Figure 1: Overview of CANDLE. Given EEG X and the corresponding T1-weighted MRI G, a subject-specific lead-field matrix L is derived as an anatomical constraint. The denoising model f<sub>θ</sub> outputs denoised X, from which the range-space component is analytically recovered, while g<sub>γ</sub> estimates the unobservable null-space component. Their combination yields the source estimate Y<sup>ˆ</sup> .

Despite this progress, most learning-based ESI models are trained in a subject-aggregated common cortical space, such as fsaverage (Fischl et al., 1999), and therefore cannot explicitly account for inter-individual differences in cortical geometry. Indeed, Sun et al. (2023) reported that applying learning-based ESI models to a particular subject requires subject-specific post-training.

To address this limitation, we propose CANDLE<sup>1</sup>, a learning-based ESI model that directly estimates source activity on subject-specific cortical geometries from EEG. CANDLE learns the null-space component (Schwab et al., 2019) of source activity induced by a T1-derived source-to-sensor mapping. This allows the model to capture a prior over source activity while preserving subject-specific geometric constraints. Furthermore, we introduce a whole-brain simulator tailored to training CAN-DLE. Because ground-truth source activity is unobservable in real recordings, learning-based ESI models are typically trained on data synthesized by whole-brain simulators (Schirner et al., 2022; Ziaeemehr et al., 2025), and evaluated through sim-to-real validation on downstream tasks. Existing simulators, however, are not designed to jointly capture the anatomical variability and realistic source configurations required for training CANDLE. Our simulator addresses this gap by generating neurobiologically grounded source activity across diverse subject-specific cortical geometries.

## Our contributions are summarized as follows:

• We propose CANDLE, a learning-based ESI model that incorporates subject-specific cortical geometry derived from each subject’s T1-weighted MRI. CANDLE learns the nullspace component of cortical source activity, thereby imposing a data-driven prior on the unobservable source subspace while preserving subject-specific geometric constraints.

• We develop a whole-brain simulator for training CANDLE that synthesizes realistic source activity across subject-specific cortical geometries from over 1,100 individuals and defines neurobiologically grounded source regions by clustering over 26,000 statistical brain maps.

• Without subject-specific tuning, CANDLE outperformed prior ESI methods on simulated source activity estimation and generalized to two relevant empirical tasks: (i) intracranial stimulation localization from simultaneously recorded scalp EEG and (ii) epileptogenic zone estimation from presurgical interictal EEG.

## 2 METHODOLOGY

We propose CANDLE, a learning-based electrophysiological source imaging (ESI) model that explicitly accounts for subject-specific cortical geometry. Figure 1 provides an overview of CANDLE.

Given EEG $\pmb { X } \in \mathbb { R } ^ { C \times T }$ and the corresponding T1-weighted MRI ${ \mathcal { G } } .$ , the model estimates the underlying cortical source activity $\hat { \pmb Y } \in \mathbb R ^ { { N } \times { T } }$ . Here, $C , N$ , and T denote the number of electrodes, the number of cortical source regions, and the sequence length, respectively.

Furthermore, to train CANDLE, we construct a whole-brain simulator illustrated in Figure 2. The simulator integrates subject-specific cortical geometries derived from more than 1,100 individuals with source configurations via clustering more than 26,000 statistical brain maps. In the following, Subsection 2.1 describes CANDLE, and Subsection 2.2 describes the construction of the simulator.

## 2.1 PROPOSED METHOD: CANDLE

Subject-specific cortical geometry constraint. In CANDLE, we derive a lead-field matrix from G that maps ground-truth cortical source activity $\pmb { Y } \in \mathbb { R } ^ { N \times T }$ to X and serves as a subject-specific anatomical constraint. EEG arises from synchronized postsynaptic potentials of cortical pyramidal neuron populations and propagate through intervening tissues $( \mathrm { e . g . }$ ., cerebrospinal fluid) before being measured at the scalp (Hallez et al., 2007). This generating process is formulated as $\pmb { X } = \pmb { L } \pmb { Y } + \pmb { \mathrm { \epsilon } } .$ where $\pmb { L } \in \mathbb { R } ^ { C \times N }$ denotes the lead-field matrix and ϵ represents measurement noise.

Ideally, ESI could recover Y from X by inverting the operator L; however, this inverse problem is inherently ill-posed because the cortical source space is typically much higher-dimensional than the EEG sensor space $( \mathrm { i } . \mathrm { e } . , N \gg C )$ . Numerous source configurations can result in the same EEG observation. As described below, we address this by learning a data-driven prior over the distribution of $\mathbf { Y }$ within the source space decomposed according to L.

Guided by prior neurophysiological insights into EEG generation (Tao et al., 2005; Birot et al., 2014), we derive L from G based on cortical regions defined by the Hagmann/Lausanne parcellation (Hagmann et al., 2008) and a three-layer boundary element model (Hallez et al., 2007) comprising the inner-skull, outer-skull, and scalp surfaces. Further details are provided in Appendix B.1.

Cortical source activity estimation. CANDLE estimates $\hat { Y }$ while preserving the subject-specific geometric constraint by learning only the subspace of the source space that is unobservable from $\mathbf { \bar { X } } .$ . We begin by introducing the range-null space decomposition (Schwab et al., 2019) induced by the linear operator L. Let $L ^ { \dagger }$ denote the pseudoinverse of $\scriptstyle L ,$ satisfying $L L ^ { \dagger } L = L$ . We define two projection operators, $L _ { \mathrm { r } } \triangleq L ^ { \dagger } L$ and $L _ { \mathrm { n } } \triangleq I - L ^ { \dagger } L ,$ which project a source-space sample Y onto the range space $\mathcal { R } ( L ^ { \dagger } )$ and the null space $\mathcal { N } ( \pmb { L } )$ , respectively. Since $\mathbb { R } ^ { N } = \mathcal { R } \dot { ( } L ^ { \dagger } ) \oplus \dot { ^ { \perp } } \dot { \mathcal { N } } ( L )$ , any source activity Y can be uniquely decomposed as

$$
Y = L _ { \mathrm { r } } Y + L _ { \mathrm { n } } Y .\tag{1}
$$

Notably, since $L Y = L L _ { \mathrm { r } } Y + L L _ { \mathrm { n } } Y = L Y + { \bf 0 }$ , these two components have fundamentally different observability: $\pmb { L } _ { \mathrm { { r } } } \pmb { Y }$ can be interpreted as the component of $\dot { \mathbf { Y } }$ observable in X through $\scriptstyle L ,$ whereas $\ b { L _ { \mathrm { n } } Y }$ is completely unobservable from X. In the noiseless case, where $X = L Y$ the range-space component can therefore be exactly recovered as $L _ { \mathrm { r } } Y = L ^ { \dagger } X$ , while only the unobservable null-space component $\ b { L _ { \mathrm { n } } Y }$ is learned. By restricting learning to degrees of freedom not determined by the measurements, this decomposition preserves source configurations consistent with X under L while enabling a plausible estimate of $\hat { \mathbf { Y } }$ . This approach is closely related to prior methods that exploit range-null space decomposition for linear ill-posed problems in computer vision, such as image inpainting and super-resolution (Wang et al., 2023b; Jacome et al., 2025).

In practice, however, X often has a low signal-to-noise ratio due to instrumentation noise and motion artifacts. Consequently, directly recovering $\pmb { L } _ { \mathrm { { r } } } \pmb { Y }$ as $L ^ { \dagger } X$ can propagate measurement noise into the source space, potentially yielding an unstable or overly diffuse estimate of $\hat { Y }$ .

To this end, we first denoise X before estimating $\hat { Y }$ via range–null space decomposition as

$$
{ \pmb X } _ { \mathrm { d e n o i s e } } = f _ { \theta } ( { \pmb X } ) , \qquad { \hat { \pmb Y } } = \underbrace { { \pmb L } ^ { \dagger } { \pmb X } _ { \mathrm { d e n o i s e } } } _ { \mathrm { R a n g e - s p a c e c o m p o n e n t } } + \underbrace { { \pmb L } _ { \mathrm { n } } g _ { \gamma } ( { \pmb L } ^ { \dagger } { \pmb X } _ { \mathrm { d e n o i s e } } ) } _ { \mathrm { N u l l - s p a c e c o m p o n e n t } } .\tag{2}
$$

Here, $f _ { \theta }$ and $g _ { \gamma }$ denote models for denoising X and estimating the null-space component of $\mathbf { Y } ,$ respectively. Under the above formulation, we define the training objective as

$$
\begin{array} { r } { \mathcal { L } = \ell _ { 2 } ( \boldsymbol { Y } , \hat { \boldsymbol { Y } } ) + \underbrace { \ell _ { 2 } ( L \boldsymbol { Y } , \boldsymbol { X } _ { \mathrm { d e n o i s e } } ) } _ { \mathrm { D e n o i s i n g l o s s } } + \underbrace { \ell _ { 2 } ( L _ { \mathrm { n } } \boldsymbol { Y } , L _ { \mathrm { n } } \hat { \boldsymbol { Y } } ) } _ { \mathrm { N u l l - s p a c e l o s s } } , } \end{array}\tag{3}
$$

![](images/91d2ed7e335334a2200c75e0ddd13e49c35f2a2df8495aaf9ad1d8b99ac3eeb0.jpg)  
Figure 2: Whole-brain simulator for training CANDLE. (a) Subject-specific source activity simulation. Neural mass models assigned to N cortical regions are coupled via structural connectivity to generate source activity Y , which is projected to EEG X through a subject-specific lead-field matrix $\bar { \boldsymbol { L } }$ derived from the T1-weighted MRI of each of 1,113 subjects. (b) Neurobiologically grounded source definition. A total of 26,273 statistical maps are characterized by their correlations with $^ { 1 , 3 1 5 }$ cognitive terms and clustered into representative source patterns. For each simulation, one pattern is sampled to modulate the excitatory gain of the corresponding cortical regions.

where $\ell _ { 2 } ( \cdot , \cdot )$ denotes the $\ell _ { 2 }$ loss. The first term directly supervises the final source estimate $\hat { Y }$ whereas the second and third terms encourage $f _ { \theta }$ and $g _ { \gamma }$ to be optimized for their respective roles of denoising and null-space component estimation. Further details of the model architecture and training configuration are provided in Appendix B.2.

## 2.2 SIMULATOR CONSTRUCTION

Subject-specific source activity simulation. We synthesize cortical source activity and corresponding EEG by simulating a biophysical computational model on diverse subject-specific cortical geometries derived from a large-scale T1-weighted MRI database, as shown in Figure 2(a). Computational models of neural dynamics span multiple levels of granularity, from microscopic to mesoscopic and macroscopic scales (Breakspear, 2017), with the appropriate level depending on the phenomenon of interest. Because scalp EEG primarily reflects synchronized postsynaptic activity of cortical pyramidal neuron populations over spatially extended cortical patches (Tao et al., 2005), we employ a neural mass model (NMM) as a mesoscopic description of cortical dynamics. Specifically, we adopt the Jansen–Rit model (Jansen & Rit, 1995), a canonical NMM capable of reproducing characteristic EEG phenomena, including alpha oscillations and event-related potentials (Luck, 2014). Details of the Jansen–Rit model and its parameter settings are provided in Appendix B.3.

A single NMM, however, describes only the local dynamics of one cortical region and does not by itself simulate source activity distributed over the entire cortex of a given subject. We therefore construct a whole-brain simulator by assigning an NMM to each node of a cortical-region graph and coupling the nodes according to the structural connectivity defined by Hagmann et al. (2008), yielding Y. Furthermore, to account for diverse cortical geometries, we instantiate the simulator across 1,113 subjects from the HCP Young Adult 2025 dataset (Van Essen et al., 2013), one of the largest publicly available MRI datasets. For each subject, L is derived from the corresponding G using the procedure described in Subsection 2.1 and Appendix B.1. The corresponding X is then synthesized as $X = L Y + \epsilon$ , where ϵ is modeled as additive Gaussian noise. This protocol extends existing whole-brain simulators (Wang et al., 2025; Ziaeemehr et al., 2025) to ESI by incorporating subject-specific cortical geometries and their corresponding source-to-sensor mappings.

Neurobiologically grounded source definition. To synthesize realistic source activity, we define source regions from statistical brain maps derived from meta-analyses of fMRI studies and amplify the NMM activity of the corresponding cortical regions, as shown in Figure 2(b). To construct the source regions, we follow Suzuki & Yamashita (2021) and Wang et al. (2026) and first collect 26,273 statistical images from NeuroVault (Gorgolewski et al., 2015), a large-scale repository of statistical brain maps. Using Neurosynth (Yarkoni et al., 2011), a large-scale meta-analytic platform for fMRI studies, we then associate each image with its correlations to 1,315 cognitive terms (e.g., “motor”, “auditory”), yielding a cognitive term vector for each image. These vectors are grouped into 200 clusters via PCA-kNN, and the images within each cluster are averaged to obtain a representative map per cluster, covering a broad range of cognitive terms. To further diversify the spatial distribution of the source regions, we randomly deactivate a subset of the cortical regions in each representative map according to a Bernoulli distribution $( p = 0 . 3 )$ and smooth the resulting map. Applying this perturbation four times to each representative map yields 800 distinct source regions. Y is then modulated by amplifying the excitatory postsynaptic potential gain of the NMM in each cortical region belonging to a given source configuration (see Appendix B.3), and the resulting activity is propagated to X. For each simulation, a source configuration is sampled and instantiated on a subject-specific cortical geometry, yielding a single sample represented by the triplet (G, X, Y).

![](images/c968b33d140d2eae4c8a9883dfd43aefdf5a01a2d54b4e1e84f949c344011cf9.jpg)

(a)  
![](images/690cab30c5c8daa91705af62d90a6cfcff953ac429adf4deb95b99754b1a15b6.jpg)  
(b)

![](images/5f875598d08247bca0c597bd6ad8bcb179796a84316996eb4b3fc97ddccb83da.jpg)

![](images/d122062921e90697bfa4905fd873a4269b65f838b068fd56e1705312f5ffba13.jpg)  
Figure 3: Simulated source activity estimation. (a) Quantitative comparison of ESI methods using HD95, Dice score, and F1 score across IoU thresholds. Error bars indicate the standard error of the mean (SEM). Statistical significance relative to CANDLE was assessed using two-sided Wilcoxon signed-rank tests (Holm corrected, \*\*\* $p < 0 . 0 0 1 )$ . (b) Qualitative comparison for a representative sample, showing the EEG signals, estimated source activities, and their spatial distributions.

## 3 EXPERIMENTS

We evaluated CANDLE across three complementary settings of increasing realism. We first assessed source activity estimation on simulated data, where ground-truth cortical activity is available, and subsequently evaluated its zero-shot sim-to-real generalization on two empirical tasks: (i) intracranial stimulation localization and (ii) epileptogenic zone estimation. We further conducted ablation studies on both empirical tasks to assess the contributions of CANDLE’s major components.

Across all experiments, we compared CANDLE with representative ESI methods, including LCMV (Van Veen et al., 1997), sLORETA (Pascual-Marqui, 2002), wMNE (Lin et al., 2006), eLORETA (Pascual-Marqui et al., 2011), ConvDip (Hecker et al., 2021), DeepSIF (Sun et al., 2022), and GBF (Wang et al., 2026). Among these, ConvDip and DeepSIF are learning-based ESI models and constitute the most direct baselines for CANDLE. For both empirical tasks, ConvDip, Deep-SIF, and CANDLE were trained exclusively on simulated data and applied without further tuning. Unless otherwise stated, statistical significance relative to CANDLE was assessed using two-sided Wilcoxon signed-rank tests with Holm correction for multiple comparisons.

## 3.1 SIMULATED SOURCE ACTIVITY ESTIMATION

Experimental setup. To evaluate ESI generalization to unseen cortical geometries, we synthesized 12,762 triplets (G, X, Y) across 1,113 subject-specific cortical geometries following the procedure in Section 2.2. We then performed a random subject-wise split into training, validation, and test sets at a ratio of 0.70:0.15:0.15. For evaluation, we focused on spatial localization, a primary objective of ESI. Following prior ESI studies (Sun et al., 2022), we temporally pooled $\hat { Y }$ to obtain a spatial source map and applied Otsu’s thresholding (Otsu, 1979) to identify the estimated active region. We quantified localization performance using the 95th-percentile Hausdorff distance (HD95) and Dice score, which capture boundary discrepancy and spatial overlap between the estimated and ground-truth source regions, respectively. We further evaluated the F1 score at IoU thresholds from 0.1 to 0.9 in 0.1 increments. Definitions of evaluation metrics are provided in Appendix C.

(a)  
![](images/240637395c77ec1497e4aca2dffc96ab5f0d708aaab66447b299075ad5865a8b.jpg)

(b)  
![](images/d728bf4c40307f726f4e84c1b75690435883cf079ea9ba4ecea7437a07997c10.jpg)

![](images/6601333c0c0051b83ca2169a6a334df37834ea5e4e8650c8fbff13f53a8d5a17.jpg)

![](images/0cc18dc68648d992963501a132071bf4f79bb589a88ac0c71af6e37cb82ae580.jpg)

![](images/208889e9cf653227049a324b226a2a29a4867fe08d6753d9997c60156aa94a2e.jpg)  
Figure 4: Intracranial stimulation localization. (a) Schematic of intracranial single-pulse electrical stimulation (SPES) site localization from simultaneous EEG. (b) Quantitative comparison using spatial dispersion and peak distance, with peak distance stratified by SPES site depth. Error bars indicate SEM. Statistical significance relative to CANDLE was assessed using two-sided Wilcoxon signed-rank tests (Holm corrected, \*\* $p < 0 . 0 1$ \*\*\* $p < 0 . 0 0 1 $ ). (c) Qualitative comparison for two representative runs, showing EEG responses and estimated SPES locations on glass-brain maps.

Results. Figure 3 summarizes the results on simulated source activity estimation. As shown in Figure 3(a), CANDLE outperformed all baselines in HD95 and Dice score and consistently achieved the highest F1 scores across IoU thresholds. In particular, CANDLE achieved an HD95 of 37.3 mm and a Dice score of 0.48, improving upon the second-best baseline by 2.5 mm and 0.15 points, respectively. The improvements over all baseline methods were statistically significant (p < 0.001).

Furthermore, Figure 3(b) presents qualitative results for a representative sample, comparing CAN DLE with the three best-performing baselines. The ground-truth source activity is spatially localized around the posterior temporal cortex and exhibits its maximum global field power (GFP) at approxi mately 0.2 s, while the corresponding EEG exhibits prominent GFP peaks at approximately 0.2 and 0.6 s. Given the EEG observation, the baseline methods estimate relatively diffuse source regions extending from the temporal to the occipital cortex, whereas CANDLE recovers a more localized source region around the temporal cortex. CANDLE also more faithfully reconstructs the temporal dynamics of the underlying source activity: ConvDip largely suppresses the dominant source-level GFP peak, while DeepSIF shifts its timing, whereas CANDLE recovers the peak near 0.2 s. Together, these results suggest that explicitly incorporating subject-specific cortical geometry enables CANDLE to generalize to unseen anatomies while preserving both the spatial localization and temporal dynamics of cortical source activity. Full numerical results and further qualitative results are reported in Table 2 and Figure 8 in Appendix D, respectively.

## 3.2 INTRACRANIAL STIMULATION LOCALIZATION

Experimental setup. To evaluate ESI under controlled perturbations with anatomically known source locations, we used the dataset of Parmigiani et al. (2022), which contains simultaneous scalp

![](images/31efa409cb183606b53e10deef9ff0813d11933f07379ba364064b8824d8434f.jpg)  
Figure 5: Epileptogenic zone (EZ) estimation. (a) Schematic of presurgical EZ localization from interictal EEG. (b) Quantitative comparison of ESI methods using HD95 and centroid distance. Error bars indicate SEM. Statistical significance relative to CANDLE was assessed using two-sided Wilcoxon signed-rank tests (Holm corrected, $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 )$ . (c) Qualitative comparison for a representative patient, showing the interictal EEG and estimated source distributions.

EEG and stereoelectroencephalography (SEEG) recordings during single-pulse electrical stimulation (SPES). Specifically, we evaluated whether the intracranial stimulation site could be localized from the corresponding scalp EEG, as illustrated in Figure 4(a). We included data from 32 patients with drug-resistant focal epilepsy, corresponding to 291 stimulation runs in total (9.1 per subject on average). Each run comprised an average of 31.5 SPES trials delivered at the same stimulation site. Following Wang et al. (2026), we averaged the EEG responses across trials within each run and used the resulting evoked response as input to each ESI method.

For evaluation, we considered the estimated source activity within a 10-ms window centered on SPES onset. This window excludes subsequent evoked responses (e.g., the N1 and N2 components (Hajnal et al., 2024)), which reflect propagation to remote regions rather than activity at the stimulation site. Localization performance was quantified using spatial dispersion and peak distance, which measure the spatial spread of the estimated source activity and the distance between its peak and the known SPES site, respectively. Both metrics were averaged across runs within each subject. Detailed definitions are provided in Appendix C.

Results. Figure 4(b) summarizes the quantitative results for intracranial stimulation localization. We evaluated spatial dispersion and peak distance across all stimulation runs and further stratified peak distance by SPES site depth. Across all runs, CANDLE achieved a spatial dispersion of 54.1 mm and a peak distance of 32.9 mm, the lowest errors among all methods and improvements of 2.5 mm and 3.6 mm over the second-best baseline, respectively. These improvements were statistically significant over five of the seven baselines $( p < 0 . 0 1 ) $ ; the differences from sLORETA and GBF were not significant. When stratified by SPES-site depth, CANDLE achieved a peak distance of 32.48 mm for shallow stimulation sites at depths of 0–10 mm, achieving the second-best performance behind GBF (29.20 mm). Notably, the localization errors of the baseline methods generally increased as the stimulation site became deeper, whereas CANDLE remained comparatively robust to source depth and achieved the best performance in all depth ranges beyond 10 mm.

Figure 4(c) presents qualitative results for two representative stimulation runs, showing the EEG signals and the estimated SPES locations on glass-brain maps. In both cases, the ground-truth SPES sites (black stars) were located in the right temporal lobe, and the CANDLE estimates (red circles) were closest to the stimulation sites among all methods. Together, these results suggest that the source activity prior learned by CANDLE exclusively from simulated data transfers to empirical EEG recordings. In particular, its robustness to SPES site depth suggests that the learned null-space prior can complement limited scalp information for deeper sources, enabling plausible localization under challenging source configurations. Complete quantitative results are listed in Table 3 in Appendix D, together with further representative runs in Figure 9.

![](images/ab57c794fcd68eef38c1873e1674ed54e34aadc1411123cb7e28474268289f00.jpg)  
Figure 6: Ablation studies of CANDLE. Performance of the full model (a) and variants ablating the architecture (b-i, b-ii), training objective (c-i, c-ii), and simulator design (d-i, d-ii), evaluated on intracranial stimulation localization (spatial dispersion and peak distance) and epileptogenic zone estimation (HD95 and centroid distance). Error bars indicate SEM.

## 3.3 EPILEPTOGENIC ZONE ESTIMATION

Experimental setup. To evaluate ESI in a clinically relevant setting, we considered epileptogenic zone (EZ) estimation from presurgical interictal EEG signals. The EZ is defined as the minimum cortical tissue whose resection or disconnection is necessary to achieve seizure freedom (Zijlmans et al., 2019). Although the EZ cannot be directly observed preoperatively, ESI of interictal epileptiform discharges (IEDs) observed in interictal EEG provides complementary information for estimating epileptogenic regions and is widely used in presurgical planning (Sun et al., 2023; 2022).

Accordingly, we evaluated CANDLE using the dataset of Vorderwulbecke et al. (2025), which pro-¨ vides presurgical EEG containing IEDs together with postsurgical T1-weighted MRI and surgical outcomes. Specifically, we evaluated whether the epileptogenic region could be localized from presurgical EEG together with the presurgical MRI for methods requiring subject-specific anatomy, as illustrated in Figure 5(a). We included 22 patients with favorable 12-month postsurgical outcomes, defined as International League Against Epilepsy (ILAE) (Wieser et al., 2001) classes 1 or 2. Because a favorable surgical outcome suggests that the clinically relevant EZ was encompassed by the resection, we used the postsurgical resection cavity as a reference for the EZ. The resection cavity was segmented from each postsurgical T1-weighted MRI using MELD-PostOp (Seo et al., 2026), a U-Net-based model for postoperative resection-cavity segmentation.

For evaluation, we obtained the source region by temporal pooling and Otsu’s thresholding (Otsu, 1979), following Subsection 3.1. We evaluated localization using HD95 and centroid distance rather than overlap-based metrics (e.g., IoU), because surgical resections may extend beyond the true EZ to ensure complete removal of epileptogenic tissue. Detailed definitions are provided in Appendix C.

Results. Figure 5(b) presents the quantitative results for EZ estimation. CANDLE achieved the lowest errors among all methods for both HD95 and centroid distance. Specifically, CANDLE obtained an HD95 of 61.7 mm and a centroid distance of 35.4 mm, improving the second-best baseline by 0.9 mm and 3.4 mm, respectively. Compared with LCMV, CANDLE achieved significant improvements in both metrics $( p < 0 . 0 1 )$ , and significantly improved HD95 over GBF $( p < 0 . 0 5 )$ .

Figure 5(c) further presents qualitative results for a representative patient whose resection involved the anterior temporal lobe. Given the corresponding presurgical interictal EEG, sLORETA, wMNE, and eLORETA produced spatially diffuse estimates extending beyond the resected region, while DeepSIF localized activity predominantly outside the resection. In contrast, although the CANDLE estimate showed some posterior extension along the temporal lobe, its dominant source region was largely concentrated around the resected anterior temporal region. Together, these results extend the zero-shot transfer observed under controlled stimulation in Section 3.2 to a more clinically realis tic setting. Appendix D reports the corresponding full numerical results (Table 4) and qualitative examples for additional patients (Figure 10).

## 3.4 ABLATION STUDIES

Figure 6 summarizes three ablation studies evaluated on intracranial stimulation localization (Parmigiani et al., 2022) and epileptogenic zone estimation (Vorderwulbecke et al., 2025). We use spatial¨ dispersion and peak distance for the former, and HD95 and centroid distance for the latter.

Architecture ablation. To validate the architecture of CANDLE, we compared (a) the full model with variants removing (b-i) the denoising model $f _ { \theta }$ or (b-ii) the null-space estimator $g _ { \gamma }$ . Both ablations degraded performance across four metrics, with peak distance increasing by 1.4 and 4.8 mm for Models (b-i) and (b-ii), respectively. The degradation was consistently larger for Model (b-ii), with statistically significant increases relative to the full model in spatial dispersion $( p < 0 . 0 0 1 )$ , peak distance $( p < 0 . 0 5 )$ , and HD95 $( p < 0 . 0 5 )$ . These results suggest that both $f _ { \theta }$ and $g _ { \gamma }$ contribute to CANDLE’s performance, while the larger impact of removing $g _ { \gamma }$ highlights the importance of estimating source activity in the null space that is unobservable from scalp measurements.

Loss ablation. We next examined the training objective by comparing (a) the full model with variants excluding either (c-i) the denoising loss or (c-ii) the null-space loss in Equation (3). Model (c-i) degraded all four metrics, including a 13.1-mm increase in peak distance. The degradation in spatial dispersion and peak distance was significant $( p < 0 . 0 0 1 )$ ). By contrast, Model (c-ii) degraded three of the four metrics, except centroid distance, with peak distance increasing by 1.3 mm. These results suggest that both losses contribute to CANDLE’s optimization, with the denoising loss playing a more critical role. This is likely because $f _ { \theta }$ precedes the range–null space decomposition, such that inadequate denoising undermines the effectiveness of the entire subsequent source reconstruction.

Simulator ablation. Finally, we assessed the simulator design using two variants of the full simulator in Section 2.2: (d-i) a fixed fsaverage head model (Fischl et al., 1999) and (d-ii) random source regions of interest (ROIs). Model (d-i) degraded performance across all four metrics, whereas Model (d-ii) degraded three of the four metrics, with HD95 as the exception. In particular, peak distance increased by 0.7 and 0.5 mm for Models (d-i) and (d-ii), respectively. These results suggest that both anatomical diversity across subjects and neurobiologically grounded source regions contribute to effective training of CANDLE.

## 4 CONCLUSION AND LIMITATIONS

Conclusion. In this study, we addressed electrophysiological source imaging (ESI). We proposed CANDLE, a learning-based ESI model that accounts for subject-specific cortical geometry by learning a prior over the unobservable null-space component of source activity. To train CANDLE, we further developed a whole-brain simulator that generates neurobiologically grounded source activity across more than 1,100 subject-specific cortical geometries. Without further tuning, CANDLE outperformed prior ESI methods on simulated source estimation and two empirical tasks: intracranial stimulation localization and epileptogenic zone estimation.

Limitations. CANDLE has several limitations that motivate future work. (i) The current implementation considers EEG recorded with a fixed montage (Oostenveld et al., 2001) and represents cortical source activity using the Hagmann/Lausanne parcellation (Hagmann et al., 2008). Extending the framework to accommodate diverse sensor configurations and source-space parcellations would broaden its applicability to recording settings and potentially other electrophysiological modalities, such as MEG. (ii) Our whole-brain simulator models cortical dynamics using a single canonical neural mass model, the Jansen–Rit model, although different neural dynamics models may better capture different physiological regimes. Future work could incorporate diverse neural dynamics models to learn more generalizable source priors and potentially improve sim-to-real transfer.

## AI USE STATEMENT

In this work, we used generative AI tools to help develop theoretical models or conceptual frameworks, design or provide feedback on research methodology or experiments, and implement methods. Additionally, we used generative AI tools for sourcing and searching for information, identifying relevant literature, and editing the paper to improve readability. We did not use generative AI tools to generate synthetic datasets, formulate mathematical claims, provide critical ingredients for proving mathematical claims, assist in writing proofs, propose or refine hypotheses, assist with translation, clean or reformat datasets, support qualitative or thematic data analysis, or interpret results. AI-assisted implementations were reviewed and tested by the authors, and AI-assisted literature retrieval was independently verified against the original sources. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This study involved only secondary analyses of previously collected human neuroimaging and electrophysiological datasets. We used data from the Human Connectome Project (Van Essen et al., 2013) and the datasets of Parmigiani et al. (2022) and Vorderwulbecke et al. (2025). No new data ¨ from human participants were collected as part of this work. All data were used in accordance with the applicable data-use terms and the ethical approvals reported in the original studies.

## REPRODUCIBILITY STATEMENT

We provide the methodological details required to reproduce CANDLE and the whole-brain simulator in Section 2.1 and Appendix B.2, including the model architecture, training configuration, data preprocessing, head-model construction, and neural mass model parameters. The experimenta protocols and evaluation metrics are described in Section 3 and Appendix C, respectively. The supplementary materials additionally include the implementation code and trained model checkpoints used in our experiments.

## REFERENCES

Shiva Asadzadeh, Tohid Yousefi Rezaii, Soosan Beheshti, Azra Delpak, and Saeed Meshgini. A systematic review of EEG source localization techniques and their applications on diagnosis of brain abnormalities. J. Neurosci. Methods, 339:108740, 2020.

Gwenael Birot et al. Head model and electrical source imaging: A study of 38 epileptic patients.´ NeuroImage: Clinical, 5:77–83, 2014.

Michael Breakspear. Dynamic models of large-scale brain activity. Nat. Neurosci., 20(3):340–352, 2017.

Miao Cao, Daniel Galvis, Simon J Vogrin, William P Woods, Sara Vogrin, Fan Wang, Wessel Woldman, John R Terry, Andre Peterson, Chris Plummer, et al. Virtual intracranial EEG signals reconstructed from MEG with potential for epilepsy surgery. Nat. Commun., 13(1):994, 2022.

Ludovica Corona, Eleonora Tamilia, M Scott Perry, Joseph R Madsen, Jeffrey Bolton, Scellig SD Stone, Steve M Stufflebeam, Phillip L Pearl, and Christos Papadelis. Non-invasive mapping of epileptogenic networks predicts surgical outcome. Brain, 146(5):1916–1931, 2023.

Carlos Coronel-Oliveros, Fernando Lehue, Ruben Herzog, Iv´ an Mindlin, Marilyn Gatica, Natalia´ Kowalczyk-Grkbska, Vicente Medel, Josephine Cruzat, Raul Gonzalez-Gomez, Hernan Hernan-´ dez, et al. A multi-frequency whole-brain neural mass model with homeostatic feedback inhibition. PLoS Comput. Biol., 22(5):e1013463, 2026.

Anders M Dale, Arthur K Liu, Bruce R Fischl, Randy L Buckner, John W Belliveau, Jeffrey D Lewine, and Eric Halgren. Dynamic Statistical Parametric Mapping: Combining fMRI and MEG for High-Resolution Imaging of Cortical Activity. Neuron, 26(1):55–67, 2000.

Zhao Feng, Ioannis Kakkos, George K. Matsopoulos, Cuntai Guan, and Yu Sun. Explaining E/MEG Source Imaging and Beyond: An Updated Review. IEEE JBHI, 29(12):9271–9286, 2025.

Bruce Fischl. FreeSurfer. NeuroImage, 62(2):774–781, 2012.

Bruce Fischl, Martin I Sereno, Roger BH Tootell, and Anders M Dale. High-resolution intersubject averaging and a coordinate system for the cortical surface. Hum. Brain Mapp., 8(4):272–284, 1999.

Yonatan Gideoni, Ryan Charles Timms, and Oiwi Parker Jones. Non-invasive neural decoding in source reconstructed brain space. ArXiv, 2410.19838, 2024.

Krzysztof J Gorgolewski, Gael Varoquaux, Gabriel Rivera, Yannick Schwarz, Satrajit S Ghosh, Camille Maumet, Vanessa V Sochat, Thomas E Nichols, Russell A Poldrack, Jean-Baptiste Poline, et al. NeuroVault.org: a web-based repository for collecting and sharing unthresholded statistical maps of the human brain. Front. Neuroinform., 9:8, 2015.

Patric Hagmann, Leila Cammoun, Xavier Gigandet, Reto Meuli, Christopher J Honey, Van J Wedeen, and Olaf Sporns. Mapping the structural core of human cerebral cortex. PLoS biology, 6(7):e159, 2008.

Boglarka Hajnal, Johanna Petra Szab´ o, Em´ ´ılia Toth, Corey J Keller, Lucia Wittner, Ashesh D Mehta,´ Lorand Er´ oss, Istv˝ an Ulbert, D´ aniel Fab´ o, and L´ aszl´ o Entz. Intracortical mechanisms of single´ pulse electrical stimulation (SPES) evoked excitations and inhibitions in humans. Sci. Rep., 14 (1):13784, 2024.

Hans Hallez, Bart Vanrumste, Roberta Grech, Joseph Muscat, Wim De Clercq, Anneleen Vergult, Yves D’Asseler, Kenneth P Camilleri, Simon G Fabri, Sabine Van Huffel, et al. Review on solving the forward problem in eeg source analysis. J. NeuroEng. Rehabil., 4(1):46, 2007.

Bin He and Zhongming Liu. Multimodal Functional Neuroimaging: Integrating Functional MRI and EEG/MEG. IEEE Rev. Biomed. Eng., 1:23–40, 2008.

Lukas Hecker, Rebekka Rupprecht, Ludger Tebartz Van Elst, and Jurgen Kornmeier. ConvDip: A¨ convolutional neural network for better EEG source imaging. Front. Neurosci., 15:569918, 2021.

Alan L Hodgkin and Andrew F Huxley. A quantitative description of membrane current and its application to conduction and excitation in nerve. J. Physiol., 117(4):500–544, 1952.

Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc Le. Transformer Quality in Linear Time. In ICML, pp. 9099–9117, 2022.

Roman Jacome, Romario Gualdron-Hurtado, Le´ on Su´ arez-Rodr´ ´ıguez, et al. NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems. ArXiv, 2510.01608, 2025.

Ben H Jansen and Vincent G Rit. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biol. Cybern., 73(4):357–366, 1995.

Dulhan Jayalath and Oiwi Parker Jones. MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training. In ICML, 2026.

Meng Jiao, Xiaochen Xian, Boyu Wang, Yu Zhang, Shihao Yang, Spencer Chen, Hai Sun, and Feng Liu. XDL-ESI: Electrophysiological sources imaging via explainable deep learning framework with validation on simultaneous EEG and IEEG. NeuroImage, 299:120802, 2024a.

Meng Jiao, Shihao Yang, Xiaochen Xian, Neel Fotedar, and Feng Liu. Multi-Modal Electrophysiological Source Imaging With Attention Neural Networks Based on Deep Fusion of EEG and MEG. IEEE TNSRE, 32:2492–2502, 2024b.

Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, et al. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift. In ICLR, 2022.

Kimi Team. Kimi linear: An expressive, efficient attention architecture. ArXiv, 2510.26692, 2025.

Fa-Hsuan Lin, Thomas Witzel, Seppo P Ahlfors, Steven M Stufflebeam, John W Belliveau, and Matti S Ham¨ al¨ ainen. Assessing and improving the spatial accuracy in MEG source localization¨ by depth-weighted minimum-norm estimates. NeuroImage, 31(1):160–171, 2006.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. In ICLR, 2024.

Steven J Luck. An introduction to the event-related potential technique. The MIT Press, 2 edition, 2014.

Christopher W Lynn and Danielle S Bassett. The physics of brain network structure, function and control. Nat. Rev. Phys., 1(5):318–332, 2019.

Davide Momi, Zheng Wang, Sara Parmigiani, Ezequiel Mikulan, Sorenza P Bastiaens, Mohammad P Oveisi, Kevin Kadak, Gianluca Gaglioti, Allison C Waters, Sean Hill, et al. Stimulation mapping and whole-brain modeling reveal gradients of excitability and recurrence in cortical networks. Nat. Commun., 16(1):3222, 2025.

J.C. Mosher, P.S. Lewis, and R.M. Leahy. Multiple dipole modeling and localization from spatiotemporal MEG data. IEEE Trans. Biomed. Eng., 39(6):541–557, 1992.

Robert Oostenveld et al. The five percent electrode system for high-resolution EEG and ERP measurements. Clin. Neurophysiol., 112(4):713–719, 2001.

Nobuyuki Otsu. A Threshold Selection Method from Gray-Level Histograms. IEEE Trans. Syst. Man Cybern., 9(1):62–66, 1979.

S Parmigiani, E Mikulan, S Russo, S Sarasso, FM Zauli, A Rubino, A Cattani, M Fecchio, D Gi ampiccolo, J Lanzone, et al. Simultaneous stereo-EEG and high-density scalp EEG recordings to study the effects of intracerebral stimulation parameters. Brain Stimul., 15(3):664–675, 2022.

R D Pascual-Marqui. Standardized low-resolution brain electromagnetic tomography (sLORETA): technical details. Methods Find. Exp. Clin. Pharmacol, 24(Suppl D):5–12, 2002.

Roberto D Pascual-Marqui, Dietrich Lehmann, Martha Koukkou, Kieko Kochi, Peter Anderer, Bernd Saletu, Hideaki Tanaka, Koichi Hirata, E Roy John, Leslie Prichep, et al. Assessing interactions in the brain with exact low-resolution electromagnetic tomography. Philos. Trans. R. Soc. A, 369(1952):3768–3784, 2011.

Michael Schirner, Lia Domide, Dionysios Perdikis, Paul Triebkorn, Leon Stefanovski, Roopa Pai, Paula Prodan, Bogdan Valean, Jessica Palmer, Chloe Langford, et al. Brain simulation as a cloudˆ service: The Virtual Brain on EBRAINS. NeuroImage, 251:118973, 2022.

Johannes Schwab, Stephan Antholzer, and Markus Haltmeier. Deep null space learning for inverse problems: convergence analysis and rates. Inverse Problems, 35(2):025008, 2019.

Jieun Seo, Mathilde Ripart, Helene Kaas, Cornelius Kronlage, Ben Sinclair, Lucy Vivash, Merran R Courtney, Terence J O’Brien, Siby Gopinath, Harilal Parasuram, et al. Automated segmentation of postsurgical resection cavities on magnetic resonance imaging in focal epilepsy: A Multicentre Epilepsy Lesion Detection study. Epilepsia, 2026.

Abbas Sohrabpour, Zhengxiang Cai, Shuai Ye, Benjamin Brinkmann, Gregory Worrell, and Bin He. Noninvasive electromagnetic source imaging of spatiotemporally distributed epileptogenic brain sources. Nat. Commun., 11(1):1946, 2020.

Rui Sun, Abbas Sohrabpour, Gregory A Worrell, and Bin He. Deep neural networks constrained by neural mass models improve electrophysiological source imaging of spatiotemporal brain dynamics. PNAS, 119(31):e2201128119, 2022.

Rui Sun, Wenbo Zhang, Anto Bagic, and Bin He. Deep learning based source imaging provides´ strong sublobar localization of epileptogenic zone from MEG interictal spikes. NeuroImage, 281: 120366, 2023.

Keita Suzuki and Okito Yamashita. MEG current source reconstruction using a meta-analysis fMRI prior. NeuroImage, 236:118034, 2021.

Shuntaro Suzuki, Shunya Nagashima, and Komei Sugiura. Cortical-SSM: a deep state space model for motor imagery decoding from EEG signals. J. Neural Eng., 23(4):046032, 2026.

James X Tao, Amit Ray, Susan Hawes-Ebersole, and John S Ebersole. Intracranial EEG substrates of scalp EEG interictal spikes. Epilepsia, 46(5):669–676, 2005.

David C. Van Essen, Stephen M. Smith, Deanna M. Barch, Timothy E.J. Behrens, Essa Yacoub, and Kamil Ugurbil. The WU-Minn Human Connectome Project: An overview. NeuroImage, 80: 62–79, 2013.

B.D. Van Veen, W. Van Drongelen, M. Yuchtman, and A. Suzuki. Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. IEEE Trans. Biomed. Eng., 44(9):867–880, 1997.

Bernd J Vorderwulbecke, Margherita Carboni, Sebastien Tourbier, Laurent Spinelli, Denis Brunet,¨ Martin Seeber, Christian M Korff, Shahan Momjian, Maria Vargas, Margitta Seeck, et al. High-Density EEG Source Localisation of averaged interictal epileptic Discharges validated by surgical Outcome. Sci. Data, 12(1):1441, 2025.

Chaoming Wang, Tianqiu Zhang, Xiaoyu Chen, Sichao He, Shangyang Li, and Si Wu. BrainPy, a flexible, integrative, efficient, and extensible framework for general-purpose brain dynamics programming. eLife, 12:e86365, 2023a.

Huifang E Wang, Borana Dollomaja, Paul Triebkorn, Gian Marco Duma, Adam Williamson, Julia Makhalova, Jean-Didier Lemarechal, Fabrice Bartolomei, and Viktor Jirsa. Virtual brain twins for stimulation in epilepsy. Nat. Comput. Sci., 5(9):754–768, 2025.

Song Wang, Kexin Lou, Chen Wei, Zhiyuan Sheng, Jiahao Tang, Kaining Peng, Xinke Shen, Shuhao Mei, Liang Chen, Dongfeng Gu, et al. A geometry aware framework enhances noninvasive mapping of whole human brain dynamics. Nat. Biomed. Eng., 2026.

Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model. In ICLR, 2023b.

Fabrice Wendling, Fabrice Bartolomei, Jean J Bellanger, and Patrick Chauvel. Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition. Eur. J. Neurosci., 15 (9):1499–1508, 2002.

Vincent Wens. Exploring the limits of MEG spatial resolution with multipolar expansions. NeuroImage, 270:119953, 2023.

HG Wieser, WT Blume, D Fish, E Goldensohn, A Hufnagel, D King, MR Sperling, and H Luders. Proposal for a new classification of outcome with respect to epileptic seizures following epilepsy surgery. Epilepsia, 42(2):282–286, 2001.

Hugh R Wilson and Jack D Cowan. Excitatory and inhibitory interactions in localized populations of model neurons. Biophys. J., 12(1):1–24, 1972.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. In NeurIPS, volume 37, pp. 115491–115522, 2024.

Tal Yarkoni, Russell A Poldrack, Thomas E Nichols, David C Van Essen, et al. Large-scale automated synthesis of human functional neuroimaging data. Nat. Methods, 8(8):665–670, 2011.

Abolfazl Ziaeemehr, Marmaduke Woodman, Lia Domide, Spase Petkoski, Viktor Jirsa, and Meysam Hashemi. Virtual Brain Inference (VBI), a flexible and integrative toolkit for efficient probabilistic inference on whole-brain models. eLife, 14:RP106194, 2025.

Maeike Zijlmans, Willemiek Zweiphenning, and Nicole Van Klink. Changing concepts in presurgical assessment for epilepsy surgery. Nat. Rev. Neurol., 15(10):594–606, 2019.

## A RELATED WORKS

Electrophysiological source imaging. Electrophysiological source imaging (ESI) methodologies and their applications have been systematically reviewed by Asadzadeh et al. (2020) and Feng et al. (2025). Because ESI is severely ill-posed, conventional approaches address this non-uniqueness by imposing explicit spatial constraints on the cortical source space. Early methods such as MU-SIC (Mosher et al., 1992) restrict source activity to one or a few equivalent current dipoles, whereas minimum-norm–based estimators such as sLORETA (Pascual-Marqui, 2002) and dSPM (Dale et al., 2000) impose depth-dependent weighting on distributed source estimates. More recent methods introduce richer structured priors over the source space. FAST-IRES (Sohrabpour et al., 2020) promotes source sparsity under a temporal basis derived from the observed EEG signals, yielding spatially localized source estimates. GBF (Wang et al., 2026) regresses the weights of subject-specific eigenmodes derived from cortical geometry to obtain spatially smooth source estimates.

In contrast to these explicitly constrained formulations, learning-based ESI models seek to acquire data-driven priors over source activity, thereby reducing reliance on manually specified spatial constraints. ConvDip (Hecker et al., 2021), an early representative, learns a CNN-based mapping from EEG topographies to source activity. DeepSIF (Sun et al., 2022) further demonstrates that a model trained exclusively on synthetic data generated using biophysical computational models can transfer to real recordings. Subsequent work has extended this paradigm in several directions, with XDL-ESI (Jiao et al., 2024a) introducing explainability into learning-based ESI, while MMDF-ANN (Jiao et al., 2024b) improves source-estimation accuracy by jointly exploiting multimodal EEG/MEG signals. Most existing learning-based methods, however, are trained in a common template source space such as fsaverage (Fischl et al., 1999), and their application to individual subjects may there fore require post-training adaptation to subject-specific anatomy (Sun et al., 2023).

Brain activity simulators. Mathematical models of brain activity span multiple spatial scales, from biophysically detailed single-neuron models such as Hodgkin–Huxley (Hodgkin & Huxley, 1952) to macroscopic neural field models of large-scale cortical dynamics (Breakspear, 2017). Between these scales, neural mass models (NMMs) describe the aggregate dynamics of interacting neuronal populations and are well suited to electrophysiological activity at the spatial scale relevant to ESI. Representative NMMs include the Wilson–Cowan (Wilson & Cowan, 1972) and Jansen–Rit (Jansen & Rit, 1995) models, with subsequent variants developed for specific regimes such as epileptiform activity (Wendling et al., 2002) and sleep-related dynamics (Coronel-Oliveros et al., 2026).

More recently, brain activity simulations have been extended to the whole-brain scale by embedding NMMs at the nodes of structural connectomes derived from diffusion MRI (Lynn & Bassett, 2019). Frameworks such as BrainPy (Wang et al., 2023a), The Virtual Brain (Schirner et al., 2022), and Virtual Brain Inference (Ziaeemehr et al., 2025) facilitate such simulations. However, these simulators are primarily designed to model brain dynamics and often abstract away anatomical variability and the spatial organization of cortical activity, limiting their suitability for applications requiring anatomically realistic simulations across subjects, such as training learning-based ESI models.

## B DETAILED PROCEDURE OF CANDLE

## B.1 LEAD-FIELD MATRIX CALCULATION

Among various head-modeling approaches for constructing the lead-field matrix L, CANDLE derives L from the subject-specific anatomical image G by modeling volume conduction through the head tissues using a boundary element model (BEM) (Hallez et al., 2007). Specifically, we first segment the tissue boundaries relevant to EEG generation and volume conduction from G using a watershed-based segmentation procedure. These surfaces include the white-matter surface, which defines the cortical source space, as well as the inner-skull, outer-skull, and scalp surfaces through which the electrical potentials propagate before reaching the EEG electrodes. Each tissue boundary is subsequently tessellated to construct a surface-based head model, from which volume conduction is modeled using a three-layer BEM. We assign conductivities of 0.3, 0.006, and 0.3 S/m to the brain, skull, and scalp compartments, respectively.

BEM formulations can incorporate different numbers and configurations of tissue compartments. In this work, we adopt a three-layer model. Birot et al. (2014) showed that increasing the complexity of volume-conduction models has only a negligible effect on source localization performance for ESI with empirical EEG recordings. Based on this finding, we employ a three-layer BEM to reduce computational cost while preserving the anatomical detail relevant to EEG propagation.

We further discretize the cortical source space according to the Hagmann/Lausanne parcellation (Hagmann et al., 2008). Previous physiological studies have suggested that cortical activity must be synchronized over an area of approximately $\mathrm { 6 { - } 1 0 \ c m ^ { 2 } }$ to generate scalp potentials of sufficient magnitude to be detectable by EEG (Tao et al., 2005). Motivated by this spatial scale, we model the cortical surface using the $N = 9 9 4$ regions of the Hagmann/Lausanne parcellation, each covering approximately $\mathrm { 1 . 5 ~ c m ^ { 2 } }$ of cortical surface. The resulting subject-specific model defines $L ,$ which maps cortical source activity to EEG measurements.

## B.2 ADDITIONAL METHODOLOGICAL DETAILS

Model configuration. The architectures adopted for the denoising model $f _ { \theta }$ and null-space estimation model $g _ { \gamma }$ are illustrated in Figure 7. As the backbone, we employ Kimi Delta Attention (Kimi Team, 2025), a cuttingedge architecture from the family of Delta Rule-based models (Yang et al., 2024) that have demonstrated strong sequence-modeling performance across diverse modalities. We extend this backbone with a gated attention unitbased module (Hua et al., 2022), which has also shown effectiveness in sequence modeling. We also adopt inverted normalization along the temporal dimension, following prior works (Kim et al., 2022; Liu et al., 2024) showing that this strategy improves robustness to non-stationarity in time-series modeling. We then stack M such blocks to construct the overall architecture.

![](images/a0aa77a7c679e2cf428813c7d8aa8b01159f91a197ff05915a0222a9f0965095.jpg)

We trained CANDLE for 20 epochs with a batch size of 64 using AdamW $( \beta _ { 1 } = 0 . 9 , ^ { - } \beta _ { 2 } = 0 . 9 9 9 )$ , a learning rate of $1 . 0 \times 1 0 ^ { - 4 }$ , and a weight decay of $1 . 0 \times 1 0 ^ { - 2 }$ We set the sequence length to $\bar { T } = 5 0 0$ , corresponding to a one-second EEG segment sampled at 500 Hz, and used $M \ : = \ : 2$ stacked blocks. The number of EEG channels

Figure 7: Architectures adopted for the denoising model $f _ { \theta }$ and null-space estimation model $g _ { \gamma }$ in CANDLE.

was fixed at $C = 6 4$ , following the standard BioSemi 64-channel montage (Oostenveld et al., 2001), while the cortical source space comprised $N = 9 9 4$ regions based on the Hagmann–Lausanne parcellation (Hagmann et al., 2008), as detailed in Appendix B.1. Under this configuration, CANDLE contained approximately 11.6 M trainable parameters and required approximately 15.30 G FLOPs.

We split the simulated dataset into training, validation, and test sets. The training set was used for parameter optimization, the validation set for model selection, and the test set exclusively for final evaluation. After each training epoch, we evaluated the checkpoint on the validation set using the 95th-percentile Hausdorff distance (HD95), following the procedure described in Subsection 3.1 and Appendix C. Among the resulting checkpoints, we selected the model achieving the lowest HD95 on the validation set and used this checkpoint for evaluation on the test set.

All experiments were performed on an NVIDIA GeForce RTX 4090 with 24 GB of VRAM and an Intel Core i9-13900KF processor with 64 GB of RAM. Training CANDLE under the configuration described above required approximately 1.8 hours.

Data preprocessing. EEG signals were downsampled to 500 Hz and band-pass filtered between 0.5 and 45 Hz using a Butterworth filter. We subsequently applied common average referencing and z-score normalization. For subject-specific anatomical processing, T1-weighted MRIs were processed using FreeSurfer (Fischl, 2012) to obtain the cortical geometry required for constructing the subject-specific lead-field matrix.

Table 1: Parameters of the Jansen–Rit model.
<table><tr><td>Symbol</td><td>Value</td><td>Description</td></tr><tr><td> $A _ { i }$ </td><td>3.25 mV</td><td>Average amplitude of the excitatory postsynaptic potential</td></tr><tr><td> $B$ </td><td> $2 2 \mathrm { m V }$ </td><td>Average amplitude of the inhibitory postsynaptic potential</td></tr><tr><td>a</td><td> $1 0 0 \mathrm { s } ^ { - 1 }$ </td><td>Time constant of the excitatory synaptic response</td></tr><tr><td> $b$ </td><td> $5 0 \mathrm { s } ^ { - 1 }$ </td><td>Time constant of the inhibitory synaptic response</td></tr><tr><td> $C _ { 1 } , C _ { 2 }$ </td><td>135, 108</td><td>Average pyramidal cell-excitatory interneuron synaptic connectivity</td></tr><tr><td> $C _ { 3 } , C _ { 4 }$ </td><td>33.75, 33.75</td><td>Average pyramidal cell-inhibitory interneuron synaptic connectivity</td></tr><tr><td> $e _ { \mathrm { m a x } }$ </td><td>5 Hz</td><td>Maximum firing rate</td></tr><tr><td> $v _ { 0 }$ </td><td>6mV</td><td>Membrane potential at half-maximum firing rate</td></tr><tr><td> $r$ </td><td> $0 . 5 6 \mathrm { m V } ^ { - 1 }$ </td><td>Slope of the sigmoid function at  $v _ { 0 }$ </td></tr></table>

## B.3 JANSEN–RIT MODEL

We employ the Jansen–Rit neural mass model (Jansen & Rit, 1995) to generate mesoscopic cortical activity for each cortical region in the whole-brain simulator. The model describes the interactions among pyramidal neurons and excitatory and inhibitory interneurons through second-order synaptic dynamics. For cortical region i, the observable source activity $y _ { i } ( t )$ is defined as the difference between the excitatory and inhibitory postsynaptic potentials,

$$
y _ { i } ( t ) = y _ { 1 , i } ( t ) - y _ { 2 , i } ( t ) ,\tag{4}
$$

where $y _ { 1 , i }$ and $y _ { 2 , i }$ denote the excitatory and inhibitory postsynaptic potentials received by the pyramidal population, respectively. The dynamics are given by

$$
\ddot { y } _ { 0 , i } ( t ) = A _ { i } a \sigma ( y _ { i } ( t ) ) - 2 a \dot { y } _ { 0 , i } ( t ) - a ^ { 2 } y _ { 0 , i } ( t ) ,\tag{5}
$$

$$
\ddot { y } _ { 1 , i } ( t ) = A _ { i } a \left[ C _ { 2 } \sigma ( C _ { 1 } y _ { 0 , i } ( t ) ) + H _ { i } ( t ) + \xi _ { i } ( t ) \right] - 2 a \dot { y } _ { 1 , i } ( t ) - a ^ { 2 } y _ { 1 , i } ( t ) ,\tag{6}
$$

$$
\ddot { y } _ { 2 , i } ( t ) = B b \left[ C _ { 4 } \sigma ( C _ { 3 } y _ { 0 , i } ( t ) ) \right] - 2 b \dot { y } _ { 2 , i } ( t ) - b ^ { 2 } y _ { 2 , i } ( t ) ,\tag{7}
$$

where $y _ { 0 , i }$ denotes the excitatory postsynaptic potential generated by the pyramidal population, $H _ { i } ( t )$ denotes the interregional excitatory input from other cortical regions, $\xi _ { i } ( t )$ denotes the stochastic input, and $\sigma ( \cdot )$ converts the average membrane potential into the population firing rate:

$$
\sigma ( v ) = \frac { e _ { \operatorname* { m a x } } } { 1 + \exp \{ r ( v _ { 0 } - v ) \} } .\tag{8}
$$

The remaining model parameters are summarized in Table 1.

To extend the local Jansen–Rit dynamics to the whole cortex, we instantiate one neural mass model at each of the N cortical regions and couple their pyramidal populations according to the structural connectivity of the Hagmann–Lausanne parcellation (Hagmann et al., 2008). The interregional input $H _ { i } ( t )$ is computed as

$$
H _ { i } ( t ) = \sum _ { j = 1 } ^ { N } W _ { i j } \sigma \left( y _ { j } ( t ) \right) ,\tag{9}
$$

where $W _ { i j }$ denotes the normalized structural connectivity from region j to region i, with $W _ { i i } = 0$ The resulting regional activities are concatenated to form the whole-brain cortical source activity $\pmb { Y } ( t ) = [ y _ { 1 } ( \breve { t } ) , \dots , y _ { N } ( t ) ] ^ { \top }$

Following Sun et al. (2022), the source configurations defined in Subsection 2.2 are implemented by locally modulating the excitatory synaptic gain $A _ { i } .$ For a sampled source configuration S, we set

$$
A _ { i } = \left\{ \begin{array} { l l } { 3 . 5 0 \mathrm { m V } , } & { i \in \mathcal { S } , } \\ { 3 . 2 5 \mathrm { m V } , } & { i \notin \mathcal { S } , } \end{array} \right.\tag{10}
$$

while keeping all other Jansen–Rit parameters unchanged. Thus, source activity is generated by selectively increasing the excitatory postsynaptic response of neurobiologically defined cortical regions rather than directly injecting an artificial waveform into the source space.

The stochastic differential equations were numerically integrated using the stochastic Heun method at a sampling rate of 2,000 Hz. Each simulation began with a 1-s burn-in period to reduce dependence on the initial state. This period was discarded, and the subsequent 10 s of simulated activity were used for analysis. We generated three independent trials for each source configuration under each of three noise conditions, resulting in 30 s of simulated activity per noise condition. The simulated cortical activity was subsequently downsampled from 2,000 to 500 Hz.

## C EVALUATION METRICS

We provide detailed definitions of the evaluation metrics used in Section 3. All spatial distances are computed in the subject-specific anatomical space.

95th-percentile Hausdorff distance (HD95). Let S and $\widehat { s }$ denote the ground-truth and estimated source regions, respectively. For two sets of spatial points A and B, the directed nearest-neighbor distances are defined as

$$
D ( A , B ) = \left\{ \operatorname* { m i n } _ { b \in B } \| a - b \| _ { 2 } \left| a \in A \right. \right\} .\tag{11}
$$

The HD95 is then defined as follows, where $Q _ { 0 . 9 5 } ( \cdot )$ denotes the 95th percentile:

$$
\mathrm { H D 9 5 } ( \mathcal { S } , \widehat { \mathcal { S } } ) = \operatorname* { m a x } \left\{ Q _ { 0 . 9 5 } \Big ( D ( \mathcal { S } , \widehat { \mathcal { S } } ) \Big ) , Q _ { 0 . 9 5 } \Big ( D ( \widehat { \mathcal { S } } , \mathcal { S } ) \Big ) \right\} .\tag{12}
$$

Dice score. The Dice score quantifies the spatial overlap between the estimated and ground-truth source regions and is defined as

$$
\operatorname { D i c e } ( S , { \widehat { S } } ) = { \frac { 2 | S \cap { \widehat { S } } | } { | S | + | { \widehat { S } } | } } .\tag{13}
$$

Spatial dispersion. For intracranial stimulation localization, we quantify the spatial spread of the estimated source activity relative to the stimulation site using spatial dispersion (SD). Let $r _ { \mathrm { s t i m } }$ denote the stimulation location, $\mathbfit { r } _ { k }$ the location of the k-th cortical source, and $d _ { k } = \| r _ { k } - r _ { \mathrm { s t i m } } \| _ { 2 }$ Following the definition of Vorderwulbecke et al. (2025), spatial dispersion is calculated as follows,¨ where $R _ { k }$ denotes the estimated activity of the k-th source at the time point of maximum source activity:

$$
\mathrm { S D } = \sqrt { \frac { \sum _ { k = 1 } ^ { N } d _ { k } ^ { 2 } R _ { k } ^ { 2 } } { \sum _ { k = 1 } ^ { N } R _ { k } ^ { 2 } } } .\tag{14}
$$

F1 score across IoU thresholds. IoU measures the overlap between the estimated and reference source regions relative to their union. For each IoU threshold from 0.1 to 0.9 in increments of 0.1, an estimate was considered correct if its IoU with the ground-truth region exceeded the threshold, and the corresponding F1 score was computed.

Peak distance. Peak distance is the Euclidean distance between the known stimulation site and the cortical source with the maximum estimated activity.

Centroid distance. Centroid distance is the Euclidean distance between the spatial centroids of the estimated and ground-truth source regions.

## D ADDITIONAL EXPERIMENTAL RESULTS

Additional qualitative results. Figures 8, 9, and 10 present additional qualitative results for simulated source activity estimation, intracranial stimulation localization, and epileptogenic zone estimation, respectively. Figure 8 shows two additional test samples, comparing all eight ESI methods. As in Figure 3(b), the conventional methods and the learning-based baselines produced spatially dif fuse or displaced estimates, whereas CANDLE recovered a source region whose location and extent closely matched the ground truth. Figure 9 shows four additional stimulation runs from different patients, with the estimated SPES locations displayed in left lateral, superior, and right lateral views. Figure 10 shows two additional patients with right- and left-hemisphere resections. In both cases, the dominant source region estimated by CANDLE lay within or adjacent to the resection cavity.

![](images/8cddac4eb18e7664404bdd0ed19e61f6631d824ece5d4ed57e356157096e7186.jpg)  
Figure 8: Additional qualitative results for simulated source activity estimation. Two representative test samples, (a) and (b). In each panel, the leftmost column shows the EEG input with its global field power (GFP), the ground-truth source activity, and its temporally pooled spatial map. The remaining columns show, for each of the eight ESI methods, the estimated source activity with its GFP and the corresponding temporally pooled spatial map.

Full quantitative results. Tables 2, 3, and 4 report the complete numerical results underlying Figures 3(a), 4(b), and 5(b), respectively. Across all three tables, CANDLE achieved the best or second-best value in every metric, and the only case in which a baseline outperformed CANDLE was the shallowest SPES sites (0–10 mm), where GBF yield a lower peak distance.

![](images/3c863eec10b7f9d750e010d3311d9437feef14b1075d7a308997be3a222e6d0a.jpg)

Figure 9: Additional qualitative results for intracranial stimulation localization. Four representative stimulation runs from different patients. In each row, the left panel shows the input EEG with its GFP, where the dashed line marks SPES onset. The remaining panels show the estimated SPES locations on glass-brain maps in left lateral, superior, and right lateral views.  
![](images/ec7877faeb96f1a6f06aef74e17126d3d047305f6407b47d5e4a334a752577da.jpg)  
Figure 10: Additional qualitative results for epileptogenic zone estimation. Two representative patients with (a) a right-hemisphere (RH) and (b) a left-hemisphere (LH) resection. For each patient, the interictal EEG with its GFP is shown together with the temporally pooled source estimates of all eight ESI methods and the postsurgical resection cavity (green) used as the reference EZ.

Table 2: Full quantitative results for simulated source activity estimation. Bold and underlined values indicate the best and second-best performances, respectively. Statistical significance relative to CANDLE was assessed using two-sided Wilcoxon signed-rank tests (Holm corrected, † : $p \textless$ 0.05, $\ddagger : p < 0 . 0 1$ $\dag \dag : p < 0 . 0 0 \dag \dag$ ). Corresponds to Figure 3(a).
<table><tr><td>Metric</td><td>LCMV</td><td>sLORETA</td><td>wMNE</td><td>eLORETA</td><td>ConvDip</td><td>DeepSIF</td><td>GBF</td><td>CANDLE</td></tr><tr><td>HD95 [mm] ↓</td><td> $5 7 . 9 4 ^ { \dag \dag }$  ±9.07</td><td> $6 6 . 0 8 ^ { \dag \dag }$  ±8.86</td><td> $7 0 . 7 4 ^ { \dag \dag }$  ±8.58</td><td> $6 9 . 7 1 ^ { \dag \dag }$  ±8.00</td><td> $\underline { { 4 1 . 0 5 } } ^ { \dagger \dagger }$  ±7.35</td><td> $4 5 . 1 2 ^ { \dagger \dagger }$  ±7.93</td><td> $6 5 . 3 1 ^ { \dagger \dagger }$  ±9.27</td><td>37.26 ±9.60</td></tr><tr><td>Dice score↑</td><td> $0 . 2 2 ^ { \dag \dag }$  ±0.06</td><td> $0 . 2 4 ^ { \dag \dag }$  ±0.05</td><td> $0 . 2 1 ^ { \dag \dag }$  ±0.05</td><td> $0 . 1 9 ^ { \dagger \dagger }$  ±0.05</td><td> $0 . 3 3 ^ { \dagger \dagger }$  ±0.07</td><td> $\underline { { 0 . 3 3 } } ^ { \dagger \dagger }$  ±0.08</td><td> $0 . 2 6 ^ { \dag \dag }$  ±0.05</td><td>0.48 ±0.07</td></tr><tr><td>F1@0.1↑</td><td> $0 . 4 8 ^ { \dagger \dagger }$  ±0.15</td><td> $0 . 5 6 ^ { \dag \dag }$   $\pm 0 . 1 6$ </td><td> $0 . 5 2 ^ { \dag \dag }$  ±0.15</td><td> $0 . 4 7 ^ { \dagger \dagger }$  ±0.15</td><td> $\underline { { 0 . 6 7 } } ^ { \dagger \dagger }$   $\pm 0 . 1 4$ </td><td> $0 . 6 2 ^ { \dag \dag }$   $\pm 0 . 1 4$ </td><td> $0 . 6 2 ^ { \dag \dag }$  ±0.16</td><td>0.83 ±0.11</td></tr><tr><td>F1@0.2↑</td><td> $0 . 3 1 ^ { \dagger \dagger }$  ±0.13</td><td> $0 . 3 4 ^ { \dagger \dagger }$   $\pm 0 . 1 4$ </td><td> $0 . 2 6 ^ { \dag \dag }$  ±0.13</td><td> $0 . 2 1 ^ { \dag \dag }$  ±0.13</td><td> $\underline { { 0 . 4 8 } } ^ { \dagger \dagger }$  ±0.15</td><td> $0 . 4 7 ^ { \dagger \dagger }$   $\pm 0 . 1 6$ </td><td> $0 . 3 7 ^ { \dagger \dagger }$  ±0.15</td><td>0.71 ±0.12</td></tr><tr><td>F1@0.3↑</td><td> $0 . 1 4 ^ { \dag \dag }$  ±0.11</td><td> $0 . 1 2 ^ { \dag \dag }$  ±0.11</td><td> $0 . 0 7 ^ { \dagger \dagger }$  ±0.08</td><td> $0 . 0 5 ^ { \dagger \dagger }$  ±0.07</td><td> $0 . 3 2 ^ { \dag \dag }$  ±0.13</td><td> $\underline { { 0 . 3 3 } } ^ { \dagger \dagger }$   $\pm 0 . 1 4$ </td><td> $0 . 1 4 ^ { \dag \dag }$  ±0.10</td><td>0.58 ±0.15</td></tr><tr><td>F1@0.4↑</td><td> $0 . 0 3 ^ { \dagger \dagger }$  ±0.05</td><td> $0 . 0 2 ^ { \dag \dag }$  ±0.04</td><td> $0 . 0 1 ^ { \dag \dag }$  ±0.03</td><td> $^ { 0 . 0 1 ^ { \dag \dag } } _ { \pm 0 . 0 2 }$ </td><td> $0 . 1 9 ^ { \dagger \dagger }$  ±0.10</td><td> $0 . 2 2 ^ { \dagger \dagger }$   $\pm 0 . 1 3$ </td><td> $0 . 0 3 ^ { \dag \dag }$  ±0.05</td><td>0.43 ±0.15</td></tr><tr><td>F1@0.5↑</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.01</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.01</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 9 ^ { \dag \dag }$  ±0.08</td><td> $0 . 1 2 ^ { \dagger \dagger }$   $\pm 0 . 0 9$ </td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.02</td><td>0.27 ±0.13</td></tr><tr><td>F1@0.6↑</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 4 ^ { \dag \dag }$  ±0.05</td><td> $\underline { { 0 . 0 6 } } ^ { \dagger \dagger }$  ±0.07</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.01</td><td>0.15 ±0.10</td></tr><tr><td>F1@0.7↑</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 1 ^ { \dag \dag }$  ±0.03</td><td> $\underline { { 0 . 0 2 } } ^ { \dagger \dagger }$  ±0.04</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td>0.06 ±0.07</td></tr><tr><td>F1@0.8↑</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$  ±0.00</td><td> $0 . 0 0 ^ { \dag \dag }$ </td><td> $0 . 0 0 ^ { \dag \dag }$ </td><td> $0 . 0 0 ^ { \dag \dag }$ </td><td> $0 . 0 0 ^ { \dag \dag }$ </td><td> $0 . 0 0 ^ { \dag \dag }$ </td><td>0.02</td></tr><tr><td>F1@0.9↑</td><td> $0 . 0 0 ^ { \ddagger }$  ±0.00</td><td> $0 . 0 0 ^ { \ddagger }$  ±0.00</td><td>±0.00  $0 . 0 0 ^ { \ddagger }$  ±0.00</td><td>±0.00 0.00‡ ±0.00</td><td>±0.01  $0 . 0 0 ^ { \ddagger }$  ±0.00</td><td>±0.02 0.00‡ ±0.00</td><td>±0.00  $0 . 0 0 ^ { \ddagger }$  ±0.00</td><td>±0.04 0.01 ±0.02</td></tr></table>

Table 3: Full quantitative results for intracranial stimulation localization on the dataset of Parmigiani et al. (2022). Boldface, underlining, and significance markers follow the conventions in Table 2. Corresponds to Figure 4(b).
<table><tr><td>Metric</td><td>LCMV</td><td>sLORETA</td><td>wMNE</td><td>eLORETA</td><td>ConvDip</td><td>DeepSIF</td><td>GBF</td><td>CANDLE</td></tr><tr><td>Spatial dispersion [mm] ↓</td><td> $6 9 . 7 1 ^ { \dag \dag }$  ±6.59</td><td>57.83 ±6.52</td><td> $6 1 . 3 8 ^ { \dagger \dagger }$  ±7.04</td><td> $6 2 . 6 2 ^ { \dag \dag }$  ±6.72</td><td> $5 9 . 0 5 ^ { \dag \dag }$  ±7.11</td><td> $6 4 . 9 5 ^ { \dag \dag }$  ±6.43</td><td>56.60 ±7.07</td><td>54.11 ±7.09</td></tr><tr><td>Peak distance [mm] ↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>36.47</td><td></td></tr><tr><td>Overall</td><td> $5 8 . 1 8 ^ { \dagger \dagger }$  ±12.64</td><td>37.47 ±13.66</td><td> $4 8 . 1 1 ^ { \dagger \dagger }$  ±17.90</td><td> $5 0 . 0 9 ^ { \dag \dag }$  ±13.44</td><td> $4 1 . 1 2 ^ { \ddagger }$  ±12.28</td><td> $4 6 . 0 2 ^ { \dag \dag }$  ±14.39</td><td>±10.98</td><td>32.86 ±10.83</td></tr><tr><td>0-10mm</td><td> $5 5 . 6 5 ^ { \dagger \dagger }$  ±16.96</td><td>32.62 ±19.49</td><td>40.84 ±21.89</td><td> $4 9 . 4 0 ^ { \dagger }$  ±22.32</td><td>38.82 ±16.89</td><td> $4 6 . 9 5 ^ { \dag \dag }$  ±19.47</td><td>29.20 ±16.83</td><td>32.48 ±12.32</td></tr><tr><td>10–20 mm</td><td> $5 5 . 5 2 ^ { \dagger \dagger }$  ±19.21</td><td>34.83 ±15.19</td><td> $4 9 . 4 4 ^ { \dagger }$  ±25.81</td><td> $5 6 . 2 7 ^ { \dag \dag }$  ±21.66</td><td>45.59† ±20.76</td><td> $4 6 . 1 0 ^ { \dagger }$  ±19.09</td><td>36.39 ±13.20</td><td>31.35 ±15.36</td></tr><tr><td>20–30 mm</td><td> $6 1 . 2 5 ^ { \dagger \dagger }$  ±19.67</td><td>41.53 ±16.05 46.48</td><td>54.68† ±26.18</td><td>44.66 ±18.45</td><td>40.82 ±18.16</td><td>44.73 ±20.10</td><td>46.88 ±17.40</td><td>36.04 ±11.80</td></tr><tr><td>30–45 mm</td><td>57.01 ±9.64</td><td>±16.33</td><td>54.48 ±20.64</td><td> $6 2 . 4 4 ^ { \dagger }$  ±19.92</td><td>49.92 ±12.18</td><td>54.45 ±12.76</td><td> $5 1 . 0 5 ^ { \ddagger }$  ±16.28</td><td>38.04 ±14.56</td></tr></table>

Table 4: Full quantitative results for epileptogenic zone estimation on the dataset of (Vorderwulbecke¨ et al., 2025). Boldface, underlining, and significance markers follow the conventions in Table 2. Corresponds to Figure 5(b).
<table><tr><td>Metric</td><td>LCMV</td><td>sLORETA</td><td>wMNE</td><td>eLORETA</td><td>ConvDip</td><td>DeepSIF</td><td>GBF</td><td>CANDLE</td></tr><tr><td rowspan="3">HD95 [mm] ↓</td><td>72.82‡</td><td>66.79</td><td>62.57</td><td>63.93</td><td>67.80</td><td>65.37</td><td>71.08†</td><td>61.72</td></tr><tr><td>±10.43</td><td>±16.12</td><td>±18.20</td><td>±16.13</td><td>±11.08</td><td>±12.33</td><td>±15.10</td><td>±12.88</td></tr><tr><td> $4 8 . 2 5 ^ { \ddagger }$ </td><td>38.82</td><td>40.56</td><td>39.89</td><td>40.14</td><td>39.07</td><td>40.29</td><td>35.38</td></tr><tr><td>Centroid distance [mm] ↓</td><td>±6.69</td><td>±12.81</td><td>±15.27</td><td>±15.61</td><td>±13.99</td><td>±12.36</td><td>±13.67</td><td>±13.98</td></tr></table>