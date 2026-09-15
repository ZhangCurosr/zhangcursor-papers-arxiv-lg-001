# Sensory Precision Inference for Multimodal Arbitration under Uncertainty

Tin Mišić<sup>1</sup> and Takato Horii<sup>1,2</sup>

<sup>1</sup> The University of Osaka, Japan <sup>2</sup> IRCN, The University of Tokyo, Japan Correspondence: misic.tin.rtf@ecs.osaka-u.ac.jp

Abstract. Autonomous agents operating on multisensory data cannot assume that all sensory modalities remain consistently informative. In real environments, sensory streams are frequently corrupted by noise, missing data, or inter-modal incongruence, requiring adaptive arbitration between competing sensory hypotheses. While active inference provides a principled framework for uncertainty-guided inference, the role of dynamically inferred sensory precision in generative multimodal arbitration under sensory conflict remains comparatively underexplored. We propose a multimodal active inference model in which latent beliefs and modality-specific sensory precisions are jointly updated through iterative free-energy minimization. In our proposed model, sensory precision dynamics not only reflect sensory uncertainty but actively shape the evolution of latent beliefs during multimodal conflict. In addition, we introduce a learned prior over sensory precisions that induces structured, class-dependent precision patterns and influences cross-modal inference dynamics. We evaluate the model using a synthetic multimodal MNIST dataset combining visual, auditory, and tactile representations of digit classes under controlled sensory noise, modality dropout, and intermodal incongruence. Results show that dynamic precision inference improves reconstruction robustness under corrupted sensory evidence, enables coherent multisensory belief formation from partial observations, and produces stable arbitration between conflicting modalities. Furthermore, learned precision priors generate interpretable precision structures that shape inference dynamics and cross-modal latent structure. These findings support sensory precision inference as a mechanistic control process for adaptive multimodal belief formation under uncertainty, highlighting precision dynamics as a computational mechanism for robust and interpretable multisensory integration.

Keywords: Cross-modal arbitration · Sensory precision · Active inference

## 1 Introduction

Autonomous agents operating in multisensory environments cannot assume that all sensory modalities remain consistently reliable. Real-world sensory streams are frequently degraded by noise, partial sensory loss, or inter-modal incongruence, requiring adaptive arbitration between competing sources of sensory evidence. Biological systems dynamically redistribute inferential reliance across modalities, selectively prioritizing sensory signals that are expected to provide reliable information. Phenomena such as the McGurk efect [11] and the Stroop efect [22] further demonstrate the ability of the brain to resolve conflicting sensory evidence through selective modulation of perceptual influence.

![](images/8606c367ed8412f5b2077169049d94159e7e95b8af1d416e80322e69adf4b585.jpg)  
Fig. 1: Proposed multimodal precision inference framework. The outer loop performs parameter learning for the encoder, decoder, fusion, and precision-prior networks, while the inner loop performs iterative free-energy minimization over latent beliefs and modality precisions.

Within active inference, precision corresponds to the expected reliability of sensory observations when inferring latent causes [18, 2]. Computationally, precision modulates the gain of prediction errors during inference [3]: highly precise sensory signals exert stronger influence on belief updates, whereas low-precision signals are attenuated. In predictive processing accounts, such precision modulation has frequently been associated with attentional allocation [1], allowing inferential influence to adapt dynamically according to contextual uncertainty.

Previous work has investigated precision dynamics in spatial attention, active sensing, and policy selection [1, 13, 12, 18, 19]. These approaches demonstrate how precision can guide selective sampling and attentional shifts according to the agent’s beliefs and environmental context. Priors over sensory precision further shape how inferential influence is distributed before sensory observations are fully resolved, arising from prior preferences or learned expectations regarding environmental structure [21, 20, 1, 13]. In multimodal settings, latent causes are often associated with distinct modality-specific reliability structures, suggesting that object representations may encode structured expectations regarding the relative informativeness of diferent sensory modalities [9, 24, 14]. Under uncertain conditions, such learned precision priors may guide the redistribution of inferential influence across modalities.

Related challenges have also been explored in robotics and multimodal machine learning. Variational Bayesian approaches to human-robot interaction and multimodal concept learning have incorporated modality-specific weighting and interdependencies between sensory streams [7, 6, 23, 17, 16]. More broadly, multimodal neural architectures have investigated robustness to noisy or degraded sensory inputs through weighted attention mechanisms and uncertainty-aware encodings [4, 8, 10, 5, 25, 15]. However, in many existing approaches modality weighting remains fixed, learned statically during training, or optimized primarily for discriminative objectives. Comparatively little work has examined dynamically inferred modality precision within generative multimodal inference, particularly under sensory conflict, dropout, or inter-modal incongruence.

To address these limitations, we propose a multimodal active inference framework focused on precision-based attentional control during perceptual inference<sup>3</sup>, with the following contributions:

– A multimodal active inference architecture that performs joint inference over latent representations and modality-specific sensory precisions, enabling adaptive arbitration under noise, modality dropout, and sensory conflict.

– A learned class-dependent precision-prior mechanism that encodes structured expectations regarding modality reliability and provides top-down influences on precision allocation during inference.

– A computational analysis of precision dynamics demonstrating how precision weighted prediction errors influence latent belief trajectories, suppress unreliable sensory evidence, and support multimodal conflict resolution.

The framework is evaluated using controlled experiments on a synthetic multimodal MNIST dataset comprising visual, auditory, and tactile representations. A graphical overview of the proposed framework is shown in Fig. 1.

## 2 Proposed Method

## 2.1 Generative Model

Consider a set of multimodal observations $\mathbf { o } = \{ o ^ { ( 1 ) } , . . . , o ^ { ( M ) } \}$ , where each modality $m \in \{ 1 , . . . , M \}$ corresponds to a sensory observation of dimensionality $d _ { m }$ The model assumes a shared latent representation $\textbf { z } \in \mathbb { R } ^ { D }$ , which captures the underlying latent causes of the sensory observations. In addition, modalityspecific precision variables $\mathbf { w } \in \mathbb { R } ^ { M }$ parameterize the expected reliability of each sensory modality during inference. The full generative model factorizes as

$$
p _ { \theta } ( \mathbf { o } ^ { ( 1 : M ) } , \mathbf { z } , \mathbf { w } ) = p _ { \theta } ( \mathbf { z } ) p _ { \theta } ( \mathbf { w } | \mathbf { z } ) \prod _ { m = 1 } ^ { M } p _ { \theta } ( \mathbf { o } ^ { ( m ) } | \mathbf { z } , \mathbf { w } ) .\tag{1}
$$

The latent prior over z is defined as a standard Gaussian,

$$
p _ { \theta } ( \mathbf { z } ) = \mathcal { N } ( \mathbf { z } ; \mathbf { 0 } , \mathbf { I } ) .\tag{2}
$$

To model structured expectations regarding modality reliability, we introduce a latent-dependent prior over sensory precision variables:

$$
p _ { \theta } ( \mathbf { w } | \mathbf { z } ) = \mathcal { N } ( \mathbf { w } ; f _ { \theta } ( \mathbf { z } ) , \mathbf { I } ) ,\tag{3}
$$

where $f _ { \boldsymbol { \theta } } ( \mathbf { z } )$ is a learned mapping from latent representations to expected modality precision configurations. Each modality likelihood is modeled as

$$
p _ { \theta } ( \mathbf { o } ^ { ( m ) } | \mathbf { z } , \mathbf { w } ) = \mathcal { N } ( \mathbf { o } ^ { ( m ) } ; g _ { m } ( \mathbf { z } ) , \varPi _ { m } ( \mathbf { w } ) ^ { - 1 } ) ,\tag{4}
$$

where $g _ { m } ( \mathbf { z } )$ denotes the modality-specific decoder network.

The precision variables are transformed into positive precision scalars through the exponential function. The resulting modality precision matrices for each modality are given by

$$
\quad \varPi _ { m } ( \mathbf { w } ) = \varSigma _ { m } ( \mathbf { w } ) ^ { - 1 } = \frac { \exp ( w _ { m } ) } { \sigma _ { m , o } ^ { 2 } } \mathbf { I } _ { d _ { m } } ,\tag{5}
$$

with $\Sigma _ { m }$ being the modality covariance matrices and $\sigma _ { m , o } ^ { 2 }$ being the modalityspecific base-scale observation covariances, which are estimated from trainingset statistics and fixed during inference. Under this formulation, increasing $w _ { m }$ increases the precision assigned to modality $m _ { \colon }$ , thereby increasing the influence of modality-specific prediction errors.

## 2.2 Variational Free Energy

Inference is performed through minimization of variational free energy under a mean-field variational approximation,

$$
q ( \mathbf { z } , \mathbf { w } ) = q ( \mathbf { z } ) q ( \mathbf { w } ) ,\tag{6}
$$

with Gaussian approximate posteriors

$$
\begin{array} { r l } & { q ( \mathbf { z } ) = \mathcal { N } ( \mathbf { z } ; \mu _ { z } , \varSigma _ { z } ) , } \\ & { q ( \mathbf { w } ) = \mathcal { N } ( \mathbf { w } ; \mu _ { w } , \varSigma _ { w } ) , } \end{array}\tag{7}
$$

with covariances $\Sigma _ { z }$ and $\Sigma _ { w }$ . The variational free energy objective is defined as

$$
\begin{array} { r l } & { \mathcal { F } [ q ] = \mathbb { E } _ { q ( z , w ) } \left[ l o g \ q ( \mathbf { z } , \mathbf { w } ) - l o g \ p _ { \theta } ( \mathbf { o } , \mathbf { z } , \mathbf { w } ) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { q ( z , w ) } \left[ - l o g \ p _ { \theta } ( \mathbf { o } | \mathbf { z } , \mathbf { w } ) \right] + D _ { K L } ( q ( \mathbf { z } ) | | p ( \mathbf { z } ) ) } \\ & { \quad \quad \quad + \mathbb { E } _ { q ( z ) } \left[ D _ { K L } ( q ( \mathbf { w } ) | | p ( \mathbf { w } | \mathbf { z } ) ) \right] . } \end{array}\tag{8}
$$

Inference is implemented through iterative optimization of posterior means, yielding a point-estimate approximation of the variational posterior. Under a

point-estimate approximation, posterior covariance terms are treated as constants and omitted from the optimization objective. The resulting free energy depends only on the posterior means:

$$
\begin{array} { l } { \displaystyle \mathcal { F } ( \mu _ { z } , \mu _ { w } ) = \frac { 1 } { 2 } \sum _ { m } \varepsilon _ { m } ^ { T } \varPi _ { m } ( \mu _ { w } ) \varepsilon _ { m } + \frac { 1 } { 2 } \log \operatorname* { d e t } ( 2 \pi \varSigma _ { m } ( \mu _ { w } ) ) } \\ { \displaystyle ~ + \frac { 1 } { 2 } \mu _ { z } ^ { T } \mu _ { z } + \frac { 1 } { 2 } \varepsilon _ { w } ^ { T } \varepsilon _ { w } , } \end{array}\tag{9}
$$

where modality prediction errors and precision prediction errors are defined as

$$
\begin{array} { l } { { \varepsilon _ { m } = o ^ { ( m ) } - g _ { m } ( \mu _ { z } ) , } } \\ { { \varepsilon _ { w } = \mu _ { w } - f ( \mu _ { z } ) . } } \end{array}\tag{10}
$$

The free energy objective therefore jointly penalizes sensory prediction errors, latent complexity, and deviations from expected precision configurations.

## 2.3 Dynamic Precision Inference

Beliefs over latent states and sensory precisions are jointly updated through gradient descent on variational free energy:

$$
\begin{array} { r l } & { \mu _ { z }  \mu _ { z } - \eta _ { z } \cfrac { \partial \mathcal { F } } { \partial \mu _ { z } } , } \\ & { } \\ & { \mu _ { w }  \mu _ { w } - \eta _ { w } \cfrac { \partial \mathcal { F } } { \partial \mu _ { w } } . } \end{array}\tag{11}
$$

The latent-state gradient is given by

$$
\frac { \partial \mathcal { F } } { \partial \mu _ { z } } = \mu _ { z } - \sum _ { m = 1 } ^ { M } \left( \frac { \partial g _ { m } ( \mu _ { z } ) } { \partial \mu _ { z } } \right) ^ { T } \varPi _ { m } ( \mu _ { w } ) \varepsilon _ { m } - \left( \frac { \partial f ( \mu _ { z } ) } { \partial \mu _ { z } } \right) ^ { T } \varepsilon _ { w } .\tag{12}
$$

This update consists of three distinct contributions. The first term corresponds to the latent prior, which regularizes latent beliefs toward the origin. The second term corresponds to precision-weighted sensory prediction errors arising from each modality. The third term introduces a top-down contribution arising from the learned precision prior $f ( \mathbf { z } )$ . Through this term, latent representations encode structured expectations regarding modality reliability, allowing precision prediction errors to influence latent-state dynamics. Consequently, latent beliefs are shaped not only by bottom-up sensory prediction errors, but also by learned expectations regarding modality-specific uncertainty.

The precision-state gradient is given by

$$
\frac { \partial \mathcal { F } } { \partial \mu _ { w } } = \varepsilon _ { w } + \frac { 1 } { 2 } \exp ( \mu _ { w } ) \odot \left( \varepsilon _ { m } ^ { T } \pi _ { m } ( \mu _ { w } ) \varepsilon _ { m } - \frac { d _ { m } } { \exp ( \mu _ { w } ) } \right) .\tag{13}
$$

Under this formulation, precisions dynamically adapt according to both sensory prediction error magnitude and latent-dependent precision expectations. Modalities producing large prediction errors tend to receive reduced inferential influence, while modalities consistent with the inferred latent state are preferentially weighted during multimodal arbitration.

![](images/9aa29157c3f11a3a6ef3f50d91a3f759c8560fe06145ad9f1a748bbba9473305.jpg)  
(a)

![](images/9c125e98c0b06ddc78f5621a9d81df674e92040787dc1396b01b96b137baef0b.jpg)  
(b)

![](images/79b02cd4f23ce29ee96628e222f0479c303f44ab8b2705942c46e5378b0808cc.jpg)  
(c)  
Fig. 2: (a) Audio and tactile class groupings used to introduce modality-specific ambiguity during dataset generation. (b) Learned precision-prior mapping f(z). (c) Pairwise distances between class centroids in w-space; marked pairs correspond to classes sharing the same audio or tactile group from (a).

## 3 Model Implementation and Experimental Design

## 3.1 Dataset and Multimodal Setup

The experiments were conducted using a synthetic multimodal dataset constructed from the MNIST handwritten digit dataset consisting of 10 digit classes. The original 28 × 28 grayscale images were used as the visual modality, while additional auditory and tactile modalities were synthetically generated as 64- dimensional and 16-dimensional feature vectors, respectively.

To induce modality-specific ambiguity and varying reliability structures, the synthetic auditory and tactile modalities were generated from Gaussian distributions centered around randomly initialized representative modality vectors. For the tactile modality, eight digit classes were assigned to three shared modality groups containing three, three, and two digit classes respectively, while the remaining two classes sampled observations stochastically from these existing group distributions. Similarly, the auditory modality contained three shared modality groups of two digit classes each, one group of three classes, and the remaining class sampled randomly from the existing auditory groups. A table with the modality groupings can be seen in Fig.2a. This construction introduces partial overlap and stochastic ambiguity within individual modalities, preventing perfect unimodal discrimination and encouraging reliance on multimodal integration and adaptive precision inference during belief formation.

All modality features were normalized to zero mean and unit variance across the dataset. The dataset followed the standard MNIST split consisting of 60,000 training samples and 10,000 test samples, with the training set further divided into 90% training and 10% validation subsets.

## 3.2 Model Implementation, Training and Inference

The model consists of modality-specific encoders and decoders together with an iterative free-energy inference process over latent beliefs and modality precisions. The visual modality encoder consists of a convolutional neural network operating on $2 8 \times 2 8$ grayscale images, while the auditory and tactile modalities use multilayer perceptrons (MLPs). Individual modality embeddings are concatenated and fused through an MLP fusion network to produce the shared latent representation $\mathbf { z } \in \mathbb { R } ^ { 3 2 }$ . The latent representation is then passed to modalityspecific decoder networks which reconstruct the sensory observations for each modality. In addition, the latent state z is provided as input to the learned precision-prior network $f ( \mathbf { z } )$ , which predicts the expected modality precision configuration $\mathbf { w } \in \mathbb { R } ^ { 3 }$ . The latent precision variables w determine the weighting of modality-specific prediction errors during iterative inference. Prediction errors were normalized according to modality dimensionality to prevent highdimensional modalities from dominating inference dynamics.

Training was performed using a dual-loop optimization procedure consisting of an outer learning loop and an inner inference loop. During the outer loop, the parameters of the encoders, fusion network, decoders, and precision-prior network $f ( \mathbf { z } )$ were optimized through gradient descent on variational free energy. The inner loop performed iterative inference over latent beliefs z and modality precisions w while network parameters remained fixed. For each training sample, modality observations were first encoded through amortized inference using the modality-specific encoders and fusion network to obtain an initial latent state estimate $\mathbf { z } _ { 0 }$ . The initial latent precision estimate $\mathbf { w } _ { 0 }$ was then obtained through the precision-prior network $f ( \mathbf { z } _ { 0 } )$ . These initial latent states were subsequently refined through iterative free-energy minimization in the inner loop.

The inner loop consisted of five inference iterations during training, while substantially larger numbers of iterations (typically 50–100) were used during test-time inference to analyze the convergence properties and arbitration dynamics of the model.

To preserve stable encoder learning while preventing gradients from propagating through the iterative inference process, latent states refined during the inner loop were reattached to the original amortized latent states using a stopgradient formulation:

$$
\begin{array} { r l } & { \mu _ { z } = \mu _ { z , 0 } + ( \mu _ { z , \mathrm { r e f i n e d } } - \mu _ { z , 0 } ) _ { \mathrm { d e t a c h } } , } \\ & { \mu _ { w } = \mu _ { w , 0 } + ( \mu _ { w , \mathrm { r e f i n e d } } - \mu _ { w , 0 } ) _ { \mathrm { d e t a c h } } . } \end{array}\tag{14}
$$

This hybrid formulation combines eficient amortized initialization with iterative free-energy minimization, enabling latent states and modality precisions to adapt dynamically to sensory conflict, uncertainty, and modality dropout while avoiding unstable gradient propagation through multiple inference steps.

For training only, auxiliary regularization terms were added to encourage zero-centered precision latents and similar variance across precision dimensions, improving the interpretability of the learned precision-prior structure.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { F } + \lambda _ { \mathrm { c e n t e r } } L _ { \mathrm { c e n t e r } } + \lambda _ { \mathrm { v a r } } L _ { \mathrm { v a r } } , } \end{array}\tag{15}
$$

## 3.3 Experimental Conditions

To evaluate the contribution of dynamic precision inference and learned precision priors, we compared three model variants corresponding to progressively more expressive forms of multimodal inference:

1. Fixed-precision baseline: Modality precisions were fixed and identical across all modalities with $w _ { m } = 0 . 0$ . Inference was performed only over the latent state z, while the learned precision-prior network $f ( \mathbf { z } )$ was disabled. This condition corresponds to standard multimodal latent inference without adaptive precision weighting.

2. Dynamic-precision model: Modality precisions were inferred dynamically during iterative free-energy minimization, while the learned precision-prior network $f ( \mathbf { z } )$ remained disabled. All modality precisions were initialized with $w _ { m } = 0 . 0$ , but were free to adapt independently during inference. This condition isolates the contribution of dynamic precision inference alone.

3. Full model: Both dynamic precision inference and the learned precisionprior network $f ( \mathbf { z } )$ were enabled. In this setting, latent representations influenced the prior distribution over modality precisions, allowing learned classdependent precision structures to shape multimodal arbitration dynamics.

The proposed model variants were evaluated across two experimental settings designed to examine robustness under sensory corruption, multimodal inference from partial observations, and arbitration under inter-modal incongruence.

Noise Robustness This experiment evaluated whether dynamic precision inference suppresses increasingly unreliable sensory modalities. For each test sample, one modality was corrupted using additive Gaussian noise:

$$
o _ { m } ^ { \mathrm { n o i s y } } = o _ { m } + \alpha \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I ) ,
$$

where $\alpha \in [ 0 . 0 , 1 . 0 ]$ controlled the noise magnitude. Five noise levels were evaluated ranging from no corruption $( \alpha = 0 . 0 )$ to strong corruption $( \alpha = 1 . 0 )$

For each experimental condition, 100 test samples were evaluated using 100 inner-loop inference iterations.

Modality precisions w and modality-specific gradient contributions to latentstate updates of z were analyzed to examine how precision inference modulates the influence of noisy sensory inputs.

Additionally, we examined how sensory corruption afects latent inference dynamics by comparing multimodal and unimodal inference across model variants. Latent robustness was quantified using latent trajectory deviation (LTD), defined as the average Euclidean distance between a perturbed inference trajectory and the corresponding clean inference trajectory across all inference steps.

Incongruency Arbitration This experiment evaluated whether modalityspecific precisions actively guide latent belief formation under conflicting sensory evidence. Incongruent samples were constructed by combining modalities from two randomly selected samples A and B, while the remaining modality was masked. For each trial, one modality provided observations from sample A and another from sample B, producing conflicting multimodal evidence.

To avoid encoder bias toward higher-dimensional modalities, the latent state z was initialized at the mean of the encoder-initialized latent states of samples A and B. Initial modality precisions were then biased by a parameter ∆w ∈ [−1, 1], assigning opposite precision latents to the modalities originating from the two samples (+∆w for A, −∆w for B).

Experiments were conducted on 100 randomly generated incongruent samples using 50 inner-loop inference iterations. Arbitration performance was evaluated using three measures: (1) arbitration accuracy, defined by the Euclidean distance of the final latent state to the latent encodings of samples A and B, (2) latent trajectory length during inference, measuring convergence stability and directness in latent space, and (3) combined multimodal reconstruction error relative to both original samples, indicating overall reconstruction error across samples.

## 4 Results

## 4.1 Learned Prior Structure

The learned precision-prior network f(z) successfully acquired structured classdependent precision representations. As shown in Fig. 2b, distinct latent categories occupy diferent regions of the precision space, reflecting modality-specific reliability expectations. Figure 2c shows pairwise distances between class centroids in w-space. Classes sharing audio or tactile groupings remain separated despite their modality-level ambiguities, indicating that the learned prior encodes class-dependent reliability structure rather than modality-specific group assignments.

## 4.2 Noise Robustness

Modality Suppression Under Noise This experiment examined the behavior of inferred modality precisions and modality-specific gradient contributions to latent-state updates under increasing sensory corruption. Results are shown for visual noise; the behavior was equivalent when corrupting the other modalities.

As shown in Fig. 3a, the inferred precision of the corrupted modality decreases with increasing noise in both the dynamic-precision and full-model conditions. In the full model, the average precision of the uncorrupted modalities simultaneously increases, indicating adaptive cross-modal redistribution of inferential influence through the learned precision prior. In contrast, the fixedprecision baseline cannot adjust modality weighting during inference.

Figure 3b shows the modality-specific gradient contributions to latent-state updates. As visual noise increases, prediction-error gradients from the corrupted modality grow substantially in the baseline conditions. The full model suppresses these gradients through precision modulation, reducing the influence of noisy sensory evidence while increasing reliance on uncorrupted modalities.

![](images/c32a139abe54e60ea1536f6ca5ffabd2f838f2d808bc043bcf58107b05a0c805.jpg)  
(a) Analysis of latent w dynamics.

![](images/eac6cf90bf0e4be2fdbe061cb99ed82ade639c66e6d8c5e4961c2bc82f1580cb.jpg)  
(b) Analysis of gradient contributions.  
Fig. 3: Noise robustness results for diferent levels of noise.

Latent Robustness To examine how sensory corruption influences latent belief dynamics, we compared inference trajectories obtained from noisy observations to those obtained from the corresponding clean observations. Robustness was quantified using latent trajectory deviation (LTD),

$$
\mathrm { L T D } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left. \mathbf { z } _ { t } ^ { \mathrm { p e r t u r b e d } } - \mathbf { z } _ { t } ^ { \mathrm { c l e a n } } \right. ,\tag{16}
$$

which measures the average distance between perturbed and clean latent trajectories throughout inference. Lower LTD values indicate that latent belief dynamics remain closer to the clean inference process despite sensory corruption.

Figure 4 shows that unimodal inference produces substantially larger trajectory deviations than multimodal inference across all model variants, demonstrating the stabilizing efect of additional sensory cues under noisy conditions. Furthermore, the full model consistently exhibits the lowest LTD values, indicating that dynamic precision inference and learned precision priors improve robustness of latent belief dynamics to sensory corruption. This efect is consistent with the suppression of gradients from corrupted modalities and the increased inferential influence of uncorrupted sensory channels observed in the previous analysis.

## 4.3 Incongruency Arbitration

Incongruent sensory inputs require arbitration between competing latent hypotheses. Figure 5a shows that the fixed-precision baseline remains near chance level, whereas both dynamic-precision models achieve higher accuracy that increases with precision bias toward the target modality. The full model consistently outperforms the zero-prior variant and retains a modest advantage even at $\varDelta w = 0$ , indicating that learned precision priors contribute to arbitration beyond externally imposed precision biases. This improvement is accompanied by lower reconstruction error (Fig. 5b) and shorter latent trajectories (Fig. 5c)

![](images/cf57780c19a5c38d119cab8e64244dbcd5147f0fa6f6e43c04e1fd16a1473912.jpg)  
Fig. 4: Analysis of multimodal robustness to noise.

compared to the baseline conditions, indicating more accurate and direct convergence toward a consistent latent state.

Figure 5d illustrates a representative arbitration trial for the full model under diferent precision biases $( \varDelta w )$ . Increasing precision for one modality causes the latent state z to converge toward the corresponding sample representation while moving away from the competing alternative. At $\varDelta w = 0$ , trajectory crossings are consistent with an influence of latent-dependent precision expectations encoded by $f ( \mathbf { z } )$ . These results show that precision shapes both sensory weighting and latent belief dynamics during multimodal conflict.

## 5 Discussion

The results demonstrate that precision inference successfully suppresses corrupted sensory channels and does not merely track uncertainty, but actively shapes latent belief dynamics. During multimodal conflict, changes in modalityspecific precision enable successful arbitration between competing sensory hypotheses, supporting active inference accounts that interpret precision as a computational mechanism for attentional allocation. These findings provide evidence that precision-weighted prediction errors can support adaptive multimodal attention and conflict resolution.

Furthermore, learned precision priors induce class-dependent expectations regarding modality reliability. Diferent object categories acquire distinct modality importance structures that influence precision inference under noisy and incongruent sensory conditions. In this sense, the learned prior provides a latentdependent precision mechanism in the form of top-down expectations about sensory reliability, analogous to attentional biases observed in biological systems.

Several limitations should be acknowledged. The experiments were conducted on a synthetic multimodal dataset, and the learned precision structures are therefore tied to artificially generated modality statistics. In addition, the proposed framework considers static observations rather than temporally evolving sensory streams, limiting its ability to model longer-term attentional dynamics.

![](images/cccbdb3b176908f77cb795a45cbd3aaf9bc7217756784c35a7fa74dfec2235ed.jpg)  
(a) Arbitration accuracy.

![](images/b27fda5fdff0fc462f0e61b4727779051081b13959fa669630002ec86fcc86a3.jpg)  
(b) Reconstruction error sum.

![](images/32c3dd36f1a2a502b5d23927a2e2f46f9252b162902c288d5ffb8f0fed62f4c0.jpg)  
(c) Latent trajectory length.

![](images/f59962da7932ea6e7270d65cf9673fb1de5650eda563b15e2360fa8afeca336b.jpg)  
(d) Single trial arbitration analysis.  
Fig. 5: Incongruency arbitration analysis.

Future work will investigate more realistic multimodal environments, hierarchical precision representations, active sensory modality sampling, and precision formulations that more tightly couple latent beliefs and uncertainty estimation.

## 6 Conclusion

This paper introduced a multimodal active inference framework that jointly infers latent beliefs and modality-specific sensory precisions during free-energy minimization. In addition, we proposed learned class-dependent precision priors that encode structured expectations regarding modality reliability. Through experiments involving sensory noise and multimodal incongruence, we demonstrated that dynamic precision inference improves robustness to sensory corruption, suppresses unreliable sensory evidence, and enables efective arbitration between competing sensory inputs.

The results suggest that sensory precision serves not only as an estimate of uncertainty, but also as a mechanism that shapes the evolution of latent beliefs during inference. Furthermore, learned precision priors provide top-down biases that influence multimodal arbitration and belief formation. Together, these findings highlight the potential of precision inference as a computational mechanism for adaptive and interpretable multimodal perception within active inference.

Acknowledgments. This research has been supported by JSPS grant JP23H04834.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Feldman, H., Friston, K.J.: Attention, uncertainty, and free-energy. Front. Hum. Neurosci. 4, 215 (Dec 2010)

2. Friston, K.: The free-energy principle: a rough guide to the brain? Trends Cogn. Sci. 13(7), 293–301 (Jul 2009)

3. Friston, K.: The free-energy principle: a unified brain theory? Nat. Rev. Neurosci. 11(2), 127–138 (Feb 2010)

4. Fronzaglia, J., Spirkin, A., Marcelino, F., Chang, Y.: The efects of noise on multimodal spiking neural networks. In: 2025 IEEE International Conference on AI and Data Analytics (ICAD). pp. 1–7 (2025). https://doi.org/10.1109/ICAD65464.2025.11114039

5. Gao, Z., Jiang, X., Xu, X., Shen, F., Li, Y., Shen, H.T.: Embracing unimodal aleatoric uncertainty for robust multimodal fusion. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 26866–26875 (2024). https://doi.org/10.1109/CVPR52733.2024.02538

6. Horii, T., Nagai, Y.: Active inference through energy minimization in multimodal afective human–robot interaction. Frontiers in Robotics and AI Volume 8 - 2021 (2021). https://doi.org/10.3389/frobt.2021.684401, https://www.frontiersin. org/journals/robotics-and-ai/articles/10.3389/frobt.2021.684401

7. Horii, T., Nagai, Y., Asada, M.: Modeling development of multimodal emotion perception guided by tactile dominance and perceptual improvement. IEEE Transactions on Cognitive and Developmental Systems 10(3), 762–775 (2018). https://doi.org/10.1109/TCDS.2018.2809434

8. Liu, W., Qiu, J.L., Zheng, W.L., Lu, B.L.: Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition. IEEE Transactions on Cognitive and Developmental Systems 14(2), 715–729 (2022). https://doi.org/10.1109/TCDS.2021.3071170

9. Lynott, D., Connell, L.: Modality exclusivity norms for 400 nouns: the relationship between perceptual experience and surface word form. Behav. Res. Methods 45(2), 516–526 (Jun 2013)

10. Mai, S., Sun, Y., Xiong, A., Zeng, Y., Hu, H.: Multimodal boosting: Addressing noisy modalities and identifying modality contribution. IEEE Transactions on Multimedia 26, 3018–3033 (2024). https://doi.org/10.1109/TMM.2023.3306489

11. McGurk, H., MacDonald, J.: Hearing lips and seeing voices. Nature 264(5588), 746–748 (1976)

12. Mirza, M.B., Adams, R.A., Friston, K., Parr, T.: Introducing a bayesian model of selective attention based on active inference. Sci. Rep. 9(1), 13915 (Sep 2019)

13. Mišić, T., Koledić, K., Bonsignorio, F., Petrović, I., Marković, I.: An active inference model of covert and overt visual attention. In: Communications in Computer and Information Science, pp. 167–181. Communications in Computer and Information Science, Springer Nature Switzerland, Cham (2026)

14. Molholm, S., Martinez, A., Shpaner, M., Foxe, J.J.: Object-based attention is multisensory: co-activation of an object’s representations in ignored sensory modalities. Eur. J. Neurosci. 26(2), 499–509 (Jul 2007)

15. Nakamura, H., Okada, M., Taniguchi, T.: Representation uncertainty in selfsupervised learning as variational inference (2023), https://arxiv.org/abs/2203. 11437

16. Nakamura, T., Nagai, T., Iwahashi, N.: Multimodal categorization by hierarchical dirichlet process. In: 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 1520–1525 (2011). https://doi.org/10.1109/IROS.2011.6094763

17. Nakamura, T., Nagai, T., Iwahashi, N.: Bag of multimodal hierarchical dirichlet processes: Model of complex conceptual structure for intelligent robots. In: 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 3818– 3823 (2012). https://doi.org/10.1109/IROS.2012.6385502

18. Parr, T., Benrimoh, D.A., Vincent, P., Friston, K.J.: Precision and false perceptual inference. Front. Integr. Neurosci. 12, 39 (Sep 2018)

19. Parr, T., Friston, K.J.: Uncertainty, epistemics and active inference. J. R. Soc. Interface 14(136), 20170376 (Nov 2017)

20. Parvizi-Wayne, D.: How preferences enslave attention: calling into question the endogenous/exogenous dichotomy from an active inference perspective. Phenomenol. Cogn. Sci. (Sep 2024)

21. Shomstein, S., Zhang, X., Dubbelde, D.: Attention and platypuses. Wiley Interdiscip. Rev. Cogn. Sci. 14(1), e1600 (Jan 2023)

22. Stroop, J.R.: Studies of interference in serial verbal reactions. J. Exp. Psychol. 18(6), 643–662 (Dec 1935)

23. Taniguchi, T., Yoshino, R., Takano, T.: Multimodal hierarchical dirichlet processbased active perception by a robot. Frontiers in Neurorobotics Volume 12 - 2018 (2018). https://doi.org/10.3389/fnbot.2018.00022, https://www.frontiersin. org/journals/neurorobotics/articles/10.3389/fnbot.2018.00022

24. van de Weijer, J., Bianchi, I., Paradis, C.: Sensory modality profiles of antonyms. Lang. Cogn. 16(1), 93–107 (Mar 2024)

25. Zhang, Z., Tang, H., Sheng, J., Zhang, Z., Ren, Y., Li, Z., Yin, D., Ma, D., Liu, T.: Debiasing multimodal large language models via noise-aware preference optimization (2025), https://arxiv.org/abs/2503.17928