# Breakdown of Local Denoising as Semantic Speciation

Guangkuo Liu<sup>♠</sup>   
JILA and Department of Physics   
University of Colorado Boulder Boulder, CO 80309, USA   
guangkuo.liu@colorado.edu Yifan F. Zhang   
Department of Electrical   
and Computer Engineering Princeton University   
Princeton, NJ 08544, USA   
yz4281@princeton.edu

Rahul Nandkishore CTQM and Department of Physics University of Colorado Boulder Boulder, CO 80309, USA rahul.nandkishore@colorado.edu

Mert Okyay<sup>♠</sup> CTQM and Department of Physics University of Colorado Boulder Boulder, CO 80309, USA mert.okyay@colorado.edu

Fangjun Hu QuEra Computing Inc.   
1284 Soldiers Field Road,   
Boston, MA 02135, USA fhu@quera.com   
Xun Gao   
JILA and Department of Physics   
University of Colorado Boulder   
Boulder, CO 80309, USA   
xun.gao@colorado.edu

## Abstract

The dynamics of generative models exhibit two apparently distinct temporal windows: a speciation window, in which a sample commits to a semantic class, and a nonlocality window, in which local context windows become insufficient for generation. Motivated by evidence of their near-concurrence in a variety of frontier models, we investigate their relationship through the spatial distribution of semantic information. Under a “common cause” hypothesis, we prove that the nonlocality window must lie in the speciation window. This hypothesis postulates that semantic labels explain a fraction of the correlations between distant tokens, a condition that is natural for many real datasets. We further give conditions under which both windows shrink to a single limiting time as system size grows, defining a “phase transition”, and verify this behavior analytically in Gaussian mixtures. Together, these results identify conditions under which semantic information explains the concurrence of speciation and nonlocality, connecting two complementary perspectives on the emergence of semantic structure in generative modeling.

## 1 Introduction

Real-world data often contains distinct semantic classes. An image dataset may contain both cats and dogs; a dataset of English text may include sonnets from Shakespeare as well as essays on history. A good generative model must be able to sample from the underlying distribution and thus reproduce this structure Pham et al. [2024], Shah et al. [2025]. Such features must therefore emerge during inference. When during generation is this identity determined, and when can a small perturbation still change it? We call the period during which the sample commits to a semantic class the speciation window Biroli et al. [2024], Li and Chen [2024]. Early empirical studies show that speciation window persists for a very short time Meng et al. [2022], Choi et al. [2022], a phenomenon that has been connected to the theory of phase transition in physics Raya and Ambrogioni [2023], Biroli et al. [2024], Sclocchi et al. [2025], Takahashi et al. [2026].

A separate question is how much context a model needs when generating one part of a sample and how this context varies during generation. For example, generating a patch of fur may require only nearby texture, while generating an animal’s eye might require information from farther away to make sure it only has two, and is at the right location. At some stages, a local neighborhood may be sufficient; at others, restricting the model to that neighborhood may prevent accurate generation. We call the period during which generation requires information of a neighboorhood size that reaches the size of the image nonlocality window. Recent work by Hu et al. [2025] has identified this window in simple datasets. Other recent studies analyse the emergence of locality structure through data Lukoianov et al. [2025] and how this structure facilitates generalization Kamb and Ganguli [2024], Niedoba et al. [2024], Hunt et al. [2026], hinting that locality might be a crucial knob to understanding generative modeling.

Although speciation and nonlocality are a priori distinct, recent experiments suggest that their windows closely align. A very recent study Zhang et al. [2026] observed this alignment in two open-source diffusion models, DiT-XL Peebles and Xie [2023] and Stable Diffusion 3 Esser et al. [2024], under an analysis of neural circuitry and controlled experiments. A complementary analysis of patch-based scores connects architectural locality to collective spatial instabilities and the formation of coherent patterns, with growing spatial correlations observed in trained convolutional diffusion models Ambrogioni [2026]. Related observations in autoregressive models show that semantic commitment also occurs in short windows beyond diffusion Li et al. [2025]. These observations motivate our central question: why should the period when a sample acquires its semantic identity also be the period when generation needs distant context?

In this work, we formalize this connection theoretically under a simple hypothesis—that semantic information is nonlocally encoded. For example, in a dataset containing zebras and leopards, observing stripes in one region can help predict stripes in a distant region because both reflect the animal’s species. Unconditioned models develop these coherent features across distant domains, necessitating nonlocal computation to coordinate the generation process. As such, the window of semantic speciation must also include a window of nonlocality, temporally aligning the two phenomena. To make this intuition precise, we identify two information quantities that characterize semantic speciation and nonlocal computation, and establish a quantitative relation between them. Mutual information (MI) between local parts of samples and the class label quantifies the amount of the semantic information exposed in local regions, revealing the speciation window Handke et al. [2025, 2026]; Conditional mutual information (CMI) between parts of the sample characterizes the error incurred when compute is restricted to a local region Hu et al. [2025], revealing the nonlocality window. We derive upper and lower bounds on CMI using MI’s, under a common-cause hypothesis: the semantic labels explain a fraction of the dependence across regions. This formalizes the nonlocal encoding of semantic information as contributions to CMI. These bounds result in the containment of the nonlocality window in the speciation window. Since our theory is fundamentally informationtheoretic, the theorem applies to the generation dynamics of both autoregressive and diffusion models, independent of the noising process. The concrete examples and experiments in this paper focus on diffusion models; direct tests in autoregressive models are left to future work, although recent work Zhu et al. [2026] observes token-entropy spikes at transitions into erroneous reasoning, suggesting a possible link.

We anchor these statements in exact analytical calculations in Gaussian mixture models, where score functions and the aforementioned information-theoretic functions are obtained. We also derive the time scales of the semantic and nonlocality windows in this setting. A ‘thermodynamic limit’ (where the data dimension is taken to infinity) closes the semantic and nonlocal windows and yields a sharp transition. Beyond Gaussian mixtures, we also construct a more general condition for window closure, requiring semantic classes to separate faster than within-class fluctuations as system size grows. Our results unify semantic and architectural perspectives imposed by dynamics along generation trajectories.

![](images/d293c36e32ab93d81fca48e9fcb287a79aa79724e927640786421ca3c0ab8d90.jpg)  
Figure 1: Semantic speciation locates the window in which local denoising breaks down. Denoising trajectory is from $t = 1$ (right) to $t = 0$ (left). The upper panel depicts a semantic speciation from white noise to cat or dog for local region (blue) and global region (magenta); the corresponding thresholds define a speciation window (black vertical lines). The noisy images in the window depict generation times in which the semantic identity is visible in the whole image, but invisible in a local patch (blue box). In the lower row, A is the patch to denoise (green), B its minimal surrounding context (blue) for below-threshold denoising, and C the rest of the image (magenta). The green curve depicts $I _ { t } ( A ; C \mid B )$ whose peak characterizes a breakdown of exact denoising using local context alone. Under the common-cause hypothesis, Theorem 1 places the threshold-defined nonlocality window inside the speciation window.

## 2 Background and Related Work

## 2.1 Local scores by decaying conditional mutual information

A diffusion model is completely determined by obtaining the score function of the data distribution over diffusion time. If the underlying data distribution is $p _ { \mathrm { d a t a } }$ the score function takes the form $s ( \boldsymbol x , t ) = \nabla _ { \boldsymbol x } \log p _ { \mathrm { d a t a } } ( \boldsymbol x , t )$ where $p _ { \mathrm { d a t a } } ( x , t )$ is the data distribution convolved with the Gaussian noise along the diffusion path. We take $X _ { 0 } \sim p _ { \mathrm { d a t a } }$ as a sample from the data distribution, and use the interpolation convention with $0 \leq t \leq 1$ , where

$$
\begin{array} { r } { X _ { t } = ( 1 - t ) X _ { 0 } + t Z , \quad Z \sim { \mathcal { N } } ( 0 , \mathbb { I } ) . } \end{array}\tag{1}
$$

A brief review of diffusion models is provided in Appendix A.

Our interest is in local diffusion models, in which the generation of a pixel is determined only by its neighbourhood and not need the whole image Kamb and Ganguli [2024], Niedoba et al. [2024], Hu et al. [2025], Hunt et al. [2026]. If such a task is possible, operationally, we expect the score function to take a local form. We define locality via a tripartition of the image, shown at the left/right of the bottom row of $\mathrm { F i g . 1 } .$ , where the pixel (or small patch) to denoise is denoted $A ,$ a square annulus around A of radius r is denoted $\bar { B , }$ , and the rest of the image is denoted C. If A is close to the edges of the image, the parts of the regions outside the image domain is ignored. Images are assumed to live in $\bar { \mathbb { R } ^ { d } }$ , where $d = N \times N$ and N is the linear dimension; we work with square images for simplicity and details of the results do not depend on the aspect ratio. Each image draw $x \in \mathbb { R } ^ { d }$ is then partitioned as $x = ( x _ { A } , x _ { B } , x _ { C } )$ and as a shorthand, we denote $x _ { A B } = ( x _ { A } , x _ { B } )$ . Thus, implied by the operational constraint, a local score acts on $A$ and depends only on A and B (we use a region R and the pixels $x _ { R }$ in R interchangeably). If the ideal data distribution is known, this would be the score function of the AB marginal, given by $\nabla _ { A } \log p ( x _ { A B } )$ (see equation 34). In the absence of the full data distribution, how can such a score be constructed, and when is it useful to do so?

Inspired from work by Sang and Hsieh [2025] on mixed-state phases in quantum systems, Hu et al. [2025] bounds the recovery error from using a local denoiser instead of the global one, using the decay length scale of the CMI of the ABC tripartition. The CMI is defined as the remaining mutual information between A and C after revealing the annulus B, i.e.,

$$
I ( A ; C \mid B ) = I ( A ; B C ) - I ( A ; B ) .\tag{2}
$$

The CMI, by definition, is the expectation of the Kullback-Leibler (KL) divergence between the joint conditional $p ( x _ { A } , x _ { C } \mid x _ { B } )$ and its factorized conditional $p ( \boldsymbol { x } _ { A } \mid \boldsymbol { x } _ { B } ) p ( \boldsymbol { x } _ { C } \mid \boldsymbol { x } _ { B } ) ;$ ; if the CMI is zero, the joint distribution on ABC factorizes and the regions form a Markov chain $A - B - C$ Hu et al. [2025], Zhang et al. [2026]. As a result, differentiating the score function on A kills the C dependence completely, where $\partial _ { x _ { A } }$ ln $p ( x ) = \partial _ { x _ { \iota } }$ <sub>x</sub> ln $p _ { A B } ( x _ { A } , x _ { B } )$ for $p = p _ { A B } p _ { C | B } { \mathrm { - g i v i n g } }$ exact locality of the score function. If the CMI is not zero, it still controls the error in denoising a noisy distribution by bounding the total variation. We relegate the careful definition of the quantities in this statement, as well as proving the statement itself (Theorem 3) to Appendix A.

While the CMI is an information-theoretic quantity that satisfies data-processing inequality in the first two inputs, A and $C ,$ , and thus would decay if only one was noised, it does not satisfy such an inequality on the conditioned variable B. Adding noise to B may degrade correlations between A or $C$ and $B ,$ strengthening the ones between A and $C ,$ leading to an increase in CMI Zhang and Gopalakrishnan [2025]. Along the noise trajectory, CMI may grow too long-ranged while the context window B remains small, such that local denoising incurs larger and larger errors. We call the window in which a local denoiser fails and the global denoiser succeeds the ”nonlocality window”.

A problem with CMI as an empirical probe of phenomenology is that information theoretic quantities are notoriously hard to sample in high-dimensional data Poole et al. [2019]. As a consequence, score functions (for diffusion models) are used as operational probes in diagnosing properties of the underlying data structure Premkumar [2026], Zhang et al. [2026]. While intuitive, is it justified to study the error in the denoiser itself instead of the errors in its outputs? In Appendix A.4, we prove the following bound

$$
\Delta _ { \mathrm { l o c } } ( t ) ^ { 4 } \leq \frac { 4 d _ { A } \left( d _ { A } + 3 \right) } { t ^ { 4 } } I _ { t } \left( A : C \mid B \right) , \quad \Delta _ { \mathrm { l o c } } ^ { 2 } ( t ) = \mathbb { E } _ { p _ { t } ( x ) } \left\| \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \right\| ^ { 2 } ,\tag{3}
$$

where $d _ { A } = \dim A$ , which implies that the “locality $\mathrm { g a p } ^ { \prime \prime } \Delta _ { \mathrm { l o c } }$ can be used to probe the CMI. This result complements the results of Hu et al. [2025]: their theorem shows that CMI bounds the error incurred by a local denoiser (see Appendix A for the careful statement) whereas our result bounds the score matching error in the denoiser itself. This establishes that the locality gap itself can serve as a probe of the nonlocality transition, which already was empirically demonstrated by Zhang et al. [2026].

## 2.2 Semantic speciation

Another type of transition is studied through semantic speciation: as forward noise increases, the noisy observation carries less information about the original image’s class. This is commonly probed by forward–backward (FB) experiments, which denoise a corrupted image and assess whether the reconstruction retains its semantic identity Biroli et al. [2024], Sclocchi et al. [2025], Zhang et al. [2026].

An ideal FB experiment draws a clean sample $X _ { 0 }$ which intrinsically comes with a label $S _ { 0 } .$ , adds noise to a region R of the sample to obtain $X _ { R , t }$ . Given that observation, it draws a fresh clean image and label from the posterior $\grave { p ( } X _ { R , 0 } , S \mid X _ { R , t } )$ . Call the returned label Sb. Finally, it checks whether ${ \dot { S } } = S$

Conditioned on $X _ { R , t }$ , the original label S and the returned label $\widehat { S }$ are independently drawn from the same distribution

$$
q _ { s } \equiv p _ { t } ( s \mid X _ { R , t } ) .
$$

Therefore, the label agreement probability for $X _ { R , t }$ is

$$
\operatorname* { P r } ( S = \widehat { S } \mid X _ { R , t } ) = \sum _ { s } \operatorname* { P r } ( S = s \mid X _ { R , t } ) \operatorname* { P r } ( \widehat { S } = s \mid X _ { R , t } ) = \sum _ { s } q _ { s } ^ { 2 } .\tag{4}
$$

Averaging over the distribution of $X _ { R , t }$ gives the success rate of FB experiment for region R at time $t ,$

$$
\mathit { P } _ { \mathrm { F B } } ( R , t ) = \mathbb { E } \sum _ { s } q _ { s } ^ { 2 } .\tag{5}
$$

Our analysis considers exact posterior sampling and true semantic labels. If the clean observation determines its label, then $P _ { \mathrm { F B } } ( R , 0 ) = 1$ ; at complete noise it approaches $\textstyle \sum _ { s } p ( s ) ^ { 2 }$ , equal to $1 / L$ for L balanced classes. Empirical forward–backward experiments, however, need not reach ideal endpoints because the reverse sampler and classifier are imperfect. Sclocchi et al. [2025] report a drop in the peak of the source–reconstruction classifier-logit cosine distribution, while Zhang et al. [2026] report a rise in reconstruction classification error. We also study observations restricted to a spatial region R. Using the same type of classifier-logit cosine probe, our ImageNet [Deng et al., 2009, Russakovsky et al., 2015] crop experiment shows a rapid switch of the empirical peak that occurs earlier for smaller regions (Appendix B). We call the interval of rapid semantic change the “speciation window”, whose starting and ending points are marked by the local and global speciation, respectively, see Section 3.2.2.

## 3 Connecting Speciation and Nonlocality

We now connect the nonlocality window, in which local context is insufficient for accurate denoising, to the speciation window, in which semantic labels become uncertain under forward noising. After giving an intuitive argument using score functions, we use an information-theoretic formulation to establish conditions under which the nonlocality window is contained in the speciation window.

## 3.1 A score-based argument of window containment

Here we provide a heuristic argument using the score function that builds operational intuition for why the nonlocality window is contained in the speciation window. A natural way to connect the locality gap to semantic information is to subtract the localized version of Bayes’ theorem from its global version:

$$
\begin{array} { r } { \nabla _ { A } \log p _ { t } ( x _ { t } ) - \nabla _ { A } \log p _ { t } ( x _ { A B , t } ) = \mathbf { \Psi } - \left[ \nabla _ { A } \log p _ { t } ( s \mid x _ { t } ) - \nabla _ { A } \log p _ { t } ( s \mid x _ { A B , t } ) \right] } \\ { \mathbf { \Psi } + \left[ \nabla _ { A } \log p _ { t } ( x _ { t } \mid s ) - \nabla _ { A } \log p _ { t } ( x _ { A B , t } \mid s ) \right] . } \end{array}\tag{6}
$$

The second bracket is the vector inside the locality gap, conditioned on the semantic label s. In consistency with the common-cause hypothesis 1, we expect an informative label to reduce this gap by specifying shared aspects of the image’s global organization, leaving less dependence on distant context:

$$
\begin{array} { r } { \mathbb { E } \Big [ \| \nabla _ { A } \log p _ { t } ( x _ { t } \mid s ) - \nabla _ { A } \log p _ { t } ( x _ { A B , t } \mid s ) \| _ { 2 } ^ { 2 } \Big ] < \mathbb { E } \Big [ \| \nabla _ { A } \log p _ { t } ( x _ { t } ) - \nabla _ { A } \log p _ { t } ( x _ { A B , t } ) \| _ { 2 } ^ { 2 } \Big ] , } \end{array}
$$

with expectations taken over noisy images and their semantic labels.

Under this score-gap suppression assumption, a small unconditional gap implies that both brackets on the right-hand side of equation 6 are small in squared expectation. Conversely, when the unconditional gap is nonzero, the conditional gap cannot account for it entirely, so the first bracket must also be nonzero in squared expectation.

To interpret the first bracket, let s be the label of the clean image. The quantities $p _ { t } ( s \mid x _ { t } )$ and $p _ { t } ( s \mid x _ { A B , t } )$ are the posterior probabilities assigned to that label using global and local observations, respectively. Their log-gradients measure how these probabilities respond to perturbations in A. At low noise, both observations can identify the label reliably, and we expect their semantic posteriors to be relatively insensitive to small perturbations. At high noise, both observations become uninformative about the label, and their posteriors approach the prior. The difference in posterior responses is therefore expected to be most pronounced between the loss of reliable local label information and the loss of global label information. This motivates the conjecture that the nonlocality window lie within the speciation window. Section 3.2 makes this connection precise using mutual information and an explicit common-cause assumption.

## 3.2 The information-theoretic definition of windows

## 3.2.1 Nonlocality window

As shown by Hu et al. [2025] (and reproduced in our notation in Appendix A), the conditional mutual information $I _ { t } ( A ; C \mid B )$ bounds the error of local recovery. Therefore we define the breakdown window of local denoising as the window where CMI is above a threshold.

Definition 1 (Nonlocality window). For an annulus tripartition $A B C ,$ , and an information tolerance $\delta > 0$ , define the start and end ofthe nonlocality window as

$$
t _ { \mathrm { n o n l o c } } ^ { \mathrm { s t a r t } , \delta } : = \operatorname* { i n f } \{ t \in [ 0 , 1 ] : I _ { t } ( A ; C \mid B ) > \delta \} ,\tag{7}
$$

$$
t _ { \mathrm { n o n l o c } } ^ { \mathrm { e n d } , \delta } : = \operatorname* { s u p } \{ t \in [ 0 , 1 ] : I _ { t } ( A ; C \mid B ) > \delta \} .\tag{8}
$$

## 3.2.2 Speciation window

We show that the mutual information $I _ { t } ( S ; R )$ has the same window where it sharply decreases from the maximal $H ( S )$ to the minimal value 0, using the following lemma proved in Appendix C.

Lemma 1 (Two-sided sandwich bounds). Let $p ( s )$ be the prior distribution oflabels $s \in S , | S | = L$ and write the posterior as $q _ { s } \equiv p _ { t } ( s \mid x _ { R , t } )$ . Let $P _ { \mathrm { F B } } ( R , t )$ denote the success probability of a forward-backward experiment, as described in equation 5. On one hand, we have

$$
\frac 1 2 \mathbb { E } \sum _ { s } ( q _ { s } - p ( s ) ) ^ { 2 } \leq I _ { t } ( S ; R ) \leq \sum _ { s } \mathbb { E } \frac { ( q _ { s } - p ( s ) ) ^ { 2 } } { p ( s ) } .\tag{9}
$$

On the other hand,

$$
\begin{array} { r } { \log P _ { \mathrm { F B } } ( R , t ) \geq I _ { t } ( S ; R ) - H ( S ) \geq - h _ { 2 } \left( P _ { \mathrm { F B } } ( R , t ) \right) - ( 1 - P _ { \mathrm { F B } } ( R , t ) ) \log ( L - 1 ) , } \end{array}\tag{10}
$$

where $h _ { 2 } ( p ) = - p \log p - ( 1 - p ) \log ( 1 - p )$ is the binary entropy.

The first inequality 9 applies to the zero–information endpoint: $I _ { t } ( S ; R ) = 0$ if and only $\mathrm { i f } q _ { s } - p ( s ) =$ 0 for all s and $x _ { R , t } ,$ , reducing the posterior-sampling FB experiment to prior sampling. The second inequality 10 applies to the full-information endpoint, i.e., $I _ { t } ( S ; R ) \stackrel { \textstyle = } { = } H ( S )$ if and only if we succeed at $P _ { \mathrm { F B } } \mathbf { \bar { ( } } R , t ) = 1$ . Therefore, $P _ { \mathrm { F B } } ( R , t )$ and $I _ { t } ( S ; R )$ share the same window where they both drop from their respective maxima to minima.

The mutual information for a global region $I _ { t } ( S ; A B C )$ must be greater than that for a local region $I _ { t } ( S ; B )$ by data processing inequality, indicating that $\tilde { I _ { t } } ( S ; B )$ has an earlier drop than $I _ { t } ( S ; A \bar { B } C )$ Therefore, we choose our definition for speciation window taking this locality nuance into consideration. As the score function arguments in Section 3.1 suggest, we define the speciation window starting with nonzero local response to A and ending with global response to A.

Definition 2 (Speciation window). For an annulus tripartition $A B C$ , and an information tolerance $\delta > 0$ , define the start and end ofthe speciation window as

$$
t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \delta } : = \operatorname* { i n f } \{ t \in [ 0 , 1 ] : I _ { 0 } ( S ; A B ) - I _ { t } ( S ; B ) \geq \delta \} ,\tag{11}
$$

$$
t _ { \mathrm { s p e c } } ^ { \mathrm { e n d } , \delta } : = \operatorname* { s u p } \{ t \in [ 0 , 1 ] : I _ { t } ( S ; A B C ) \geq \delta \} .\tag{12}
$$

The start time indicates the earliest time when local label recognition ability drops by a certain amount, and the end time indicates the latest time when global recognition ability retains that amount.

## 3.2.3 Common-cause hypothesis

Conditioning on semantic labels can act in two ways. One is synergistic, where it increases conditional mutual information,

$$
I ( A ; C \mid B , S ) \geq I ( A ; C \mid B ) .
$$

This occurs when label S contains information that can only be determined by A and C together, but not from any of them alone. For example, count of total objects or parity of a set of bits. The other is redundant, where it decreases conditional mutual information,

$$
I ( A ; C \mid B , S ) \leq I ( A ; C \mid B ) .
$$

This occurs when label S acts as a common cause for A and C, such as the label of a cat image, which explains the correlation between the cat’s head and tail. We assume that the latter is the case for natural datasets, and we postulate the following common-cause hypothesis.

![](images/6d2b3cd67567c88f4e456b44cc6ea6b3e7acfd7e5afd88e55f8d35a88d106d4d.jpg)  
Figure 2: Semantic conditioning reduces the average local–global prediction gap in SD3. A smaller conditional gap means that restricting distant image-token interactions changes the model prediction less when semantic information is supplied. This is consistent with the common-cause interpretation: the description explains part of the shared image structure that would otherwise require distant context.

Hypothesis 1 (Common-cause hypothesis). For an annulus tripartition ABC, semantic labels S, and a positive constant $0 < \alpha \leq 1$ , we have

$$
I _ { t } ( A ; C \mid B , S ) \leq ( 1 - \alpha ) I _ { t } ( A ; C \mid B )\tag{13}
$$

for all time $t \in [ 0 , 1 ]$

Such labels always exist: consider the extreme case where the label S specifies the entire clean image, then $I ( A ; { \dot { C } } \mid B , S ) = 0$ , in which case we have $\alpha = 1$ . In general, we expect that the more informative the label is, the larger α is.

Gaussian mixtures with identity covariance satisfy this hypothesis for the labels matching the mixture labels at $\alpha = 1$ as shown in Appendix E. For natural datasets, Figure 2 provides empirical evidence consistent with the common-cause interpretation: semantic conditioning reduces the average local– global prediction gap in SD3, suggesting that shared semantic information reduces reliance on distant image context. We average over 64 scenes described using 5, 15–17, and 35–39 words, respectively, giving 192 prompts in total and each prompt uses three random seeds. We observe a time-weighted gap reduction of 20.7%, 23.4%, and 24.4% for short, medium, and long descriptions (see Appendix D for the experimental setup, score parameterization).

## 3.2.4 Nonlocality window is contained in speciation window

The two windows are defined using different information quantities. The common-cause hypothesis connects them.

Theorem 1 (Containment of the nonlocality window). Fix an annulus tripartition ABC. Suppose that, for all $t \in [ 0 , 1 ] , S \to X _ { A B , 0 } \to X _ { A B , 1 }$ is a Markov chain and Hypothesis 1 holds with the same $\alpha > 0 .$ . Choose $\begin{array} { r } { 0 < \delta < \operatorname* { s u p } _ { t \in [ 0 , 1 ] } I _ { t } ( A ; C \mid B ) } \end{array}$ . Then

$$
t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \alpha \delta } \leq t _ { \mathrm { n o n l o c } } ^ { \mathrm { s t a r t } , \delta } \leq t _ { \mathrm { n o n l o c } } ^ { \mathrm { e n d } , \delta } \leq t _ { \mathrm { s p e c } } ^ { \mathrm { e n d } , \alpha \delta } .\tag{14}
$$

Proof. The common-cause hypothesis gives, at each time,

$$
\alpha I _ { t } ( A ; C \mid B ) \le I _ { t } ( A ; C \mid B ) - I _ { t } ( A ; C \mid B , S ) .\tag{15}
$$

By the chain rule, the difference on the right has two useful forms. The first is

$$
\begin{array} { r l } & { I _ { t } ( A ; C \mid B ) - I _ { t } ( A ; C \mid B , S ) = I _ { t } ( S ; A B ) - I _ { t } ( S ; B ) - I _ { t } ( S ; A B C ) + I _ { t } ( S ; B C ) } \\ & { \qquad \le I _ { 0 } ( S ; A B ) - I _ { t } ( S ; B ) . } \end{array}\tag{16}
$$

The last step uses $I _ { t } ( S ; A B ) \le I _ { 0 } ( S ; A B )$ , by data processing along the forward noising channel, and $I _ { t } ( S ; \bar { A } B C ) \ge \dot { I } _ { t } ( S ; \bar { B } C )$ , by spatial data processing. Rearranging the same four mutual

informations gives the second form,

$$
\begin{array} { r l } & { I _ { t } ( A ; C \mid B ) - I _ { t } ( A ; C \mid B , S ) = I _ { t } ( S ; B C ) - I _ { t } ( S ; B ) - I _ { t } ( S ; A B C ) + I _ { t } ( S ; A B ) } \\ & { \qquad \leq I _ { t } ( S ; A B C ) . } \end{array}\tag{17}
$$

Here $I _ { t } ( S ; B ) \ge 0$ and $I _ { t } ( S ; A B C ) - I _ { t } ( S ; A B ) \ge 0$ , while $I _ { t } ( S ; B C ) \le I _ { t } ( S ; A B C )$

Before $t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \alpha \delta }$ , the definition of the speciation start gives $I _ { 0 } ( S ; A B ) - I _ { t } ( S ; B ) < \alpha \delta$ . Equations 15 and equation 16 then give $I _ { t } ( A ; C \mid \bar { B } ) < \delta$ . After $t _ { \mathrm { s p e c } } ^ { \mathrm { - e n d } , \alpha \delta }$ , the definition of the speciation end gives $I _ { t } ( S ; { \bar { A } } B C ) < \alpha \delta ;$ ; equations 15 and 17 again give ${ \mathit { I } } _ { t } ( A ; C \mid B ) < \delta$ . Thus every time at which CMI exceeds δ lies between the speciation endpoints. Since δ is below the CMI supremum, this set is nonempty. Taking its infimum and supremum proves the claim. □

## 3.3 System size scaling of two windows and phase transitions

The containment theorem 1 relates the two windows, but does not determine how their widths scale with system size. This scaling describes whether a finite-size crossover sharpens into a phase transition. To get a sense of how the windows scale with system size in simple distributions, we explicitly study the windows for a two-component Gaussian mixture. Relegating the calculational details to Appendix $\mathrm { E , }$ we only quote the analytical results. We study the distribution $p ( x , s )$ where $p ( x \mid s ) \propto \mathrm { { e x p } } ( - ( x - \mu _ { s } ) ^ { 2 } / 2 \bar { \sigma ^ { 2 } } )$ and $p ( s = \pm 1 ) = 1 / 2$ , taking $\dot { \mu _ { s } } = s \vec { 1 } \in \mathbb { R } ^ { d }$ . The scalings for the two windows for this distribution are

$$
1 - t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \alpha \delta } \asymp \frac { \sqrt { \log N } } { N } , 1 - t _ { \mathrm { n o n l o c } } ^ { \mathrm { s t a r t } , \delta } \asymp \frac { 1 } { N } , 1 - t _ { \mathrm { n o n l o c } } ^ { \mathrm { e n d } , \delta } \asymp \frac { 1 } { N } , 1 - t _ { \mathrm { s p e c } } ^ { \mathrm { e n d } , \alpha \delta } \asymp \frac { 1 } { N ^ { 2 } } ,\tag{18}
$$

which all converge to $t = 1$ in the large-N limit. The intuition behind these scalings follow from the simple fact that decoding success is determined by the signal-to-noise ratio $\| \mu _ { s } ( t ) \| / \sigma ( t )$ . For $\mu _ { s }$ we chose, the signal in B or $B C$ is extensive $\lVert \mu _ { s } \rVert \dot { \propto } N ( 1 - t )$ , and thus decoding becomes ambiguous at $1 - t = \bar { O ( 1 / N ) }$ . Going left to right (choosing $\delta \stackrel { \cdot } { \sim } 1 / N ^ { 2 }$ to get a nontrivial CMI) the start of the speciation window is marked by when the classifier using B loses δ amount of information, whose error is given by a Gaussian tail; setting this equal to the loss yields a $\sqrt { \log { 1 / \delta } }$ in addition to the signal to noise factor $1 / N$ . The locality windows are about a similar but a different question: when is decoding hard with $B ^ { \acute { } }$ only but easy with BC? Since the signal is extensive in each, both locality windows directly inherit $\dot { 1 } - t = \dot { O } ( 1 / N )$ . Lastly, in the end of the speciation window, we are in the weak signal regime, and even the global classifier is weak. From a quadratic approximation to the mutual information $I ( S : A B C )$ integral as $N ^ { 2 } ( 1 - t ) ^ { 2 }$ , which needs to be equal to $\delta ,$ , we get $1 - t = O ( 1 / N ^ { 2 } )$ ). In summary, the mechanism behind this sharpening can be understood through the decoding transition of the Gaussian mixture model, which behaves like a soft repetition code, where each local patch carries partial information about the semantic label, and a growing buffer combines these clues.

Inspired by the Gaussian mixture results, we give sufficient conditions for both windows to sharpen to a common critical point at pure noise, $t = 1$ , as system size grows. The theorem below makes this condition for local semantic identification precise and generalizes the calculation restricted to the Gaussian mixture model.

Theorem 2. (informal) Let $m _ { s } = \mathbb { E } [ X _ { B , 0 } \mid S = s ]$ be the mean image on the buffer for label s, and write

$$
D _ { N } = \operatorname* { m i n } _ { s \neq s ^ { \prime } } \Vert m _ { s } - m _ { s ^ { \prime } } \Vert ^ { 2 } .
$$

Suppose within-class fluctuations have Gaussian-type tails with variance scale at most $C _ { N }$ along every unit direction. Thus $D _ { N }$ measures the separation ofthe class means, while $C _ { N }$ measures the fluctuation that can obscure this separation. For afixedfinite label set, assume the common-cause hypothesis holds with size-independent α $\varepsilon > 0$ and use $X _ { t } \dot { } = ( 1 - t ) X _ { 0 } + t Z . \ I f { \cal D } _ { N } / ( C _ { N } + 1 ) \to \infty ,$ then both windows shrink toward $t = 1 ,$ , provided they exist and their defining information thresholds do not decrease too rapidly with system size relative to this growing separation-to-noise ratio. The precise regularity condition on the thresholds and thefull assumptions are given in Appendix F.

When $D _ { N } / ( C _ { N } + 1 )$ grows, the semantic signal in the buffer increasingly dominates the within-class fluctuations and the added diffusion noise. As in a repetition code, the label can then be read from the buffer with vanishing error at any fixed $t < 1 ;$ only at pure noise is the signal completely lost.

Reliable decoding makes the remaining label uncertainty $H ( S \mid X _ { B , t } )$ small. The common-cause hypothesis and the chain rule then give

$$
\alpha I _ { t } ( A ; C \mid B ) \le I _ { t } ( S ; A \mid B ) - I _ { t } ( S ; A \mid B , C ) \le H ( S \mid X _ { B , t } ) .
$$

Thus, when the buffer already identifies the label, the CMI is small. Moreover, $I _ { 0 } ( S ; A B ) \ -$ $I _ { t } ( S ; B ) \le H ( S \mid X _ { B , t } )$ , so the speciation window cannot start until the buffer begins to lose the label. The quantitative decoding bound pushes this start toward $t = 1$ at the allowed thresholds, and Theorem 1 places the other three endpoints between that start and 1.

## 4 Conclusion

We connect semantic speciation and nonlocality through the information shared between distant regions of a sample through our common-cause hypothesis. Under this hypothesis, we prove that the nonlocality window lies within the speciation window. This connection interprets semantics as shared global information and speciation as its decoding during generation. After demonstrating that both windows close for Gaussian mixture data distributions as system size grows, inspired by its properties, we further give a sufficient condition for both windows to sharpen to a common critical point at pure noise as system size grows.

These results open new questions about how semantic information shapes nonlocality and phase transitions in generative models. The common-cause and semantic-decoding assumptions can be tested in graphical models and real datasets. These studies can reveal how spatial structure affects when the transitions occur and how sharply they develop. The theory can also guide when denoisers use distant context and how their receptive fields change during generation. We leave these directions to future work.

## Acknowledgments

We acknowledge assistance from generative AI tools for writing, coding and verifying mathematical claims and proofs. G. L. would like to thank Jialiang Zhang and Ruohua Li for discussion on FlexAttention implementation. G. L. and X. G. acknowledge support from NSF PFC grant No. PHYS 2317149. F. H. acknowledges support from the QuEra Quantum Innovation Postdoctoral Fellowship.

## References

L. Ambrogioni. How out-of-equilibrium phase transitions can seed pattern formation in trained diffusion models, 2026. URL https://arxiv.org/abs/2603.20092. 2

G. Biroli, T. Bonnaire, V. de Bortoli, and M. Mezard. Dynamical regimes of diffusion models.´ Nature Communications, 15(1):9957, Nov. 2024. ISSN 2041-1723. doi: 10.1038/s41467-024-54281-3. URL https://doi.org/10.1038/s41467-024-54281-3. 1, 2, 4

S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004. URL https://web.stanford.edu/ boyd/cvxbook/. 16

J. Choi, J. Lee, C. Shin, S. Kim, H. Kim, and S. Yoon. Perception prioritized training of diffusion models. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11462–11471, 2022. doi: 10.1109/CVPR52688.2022.01118. 2

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5

P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis, 2021. URL https: //arxiv.org/abs/2105.05233. 17

R. Durrett. Probability: Theory and Examples. Cambridge University Press, 5 edition, 2019. URL https://sites.math.duke.edu/<sub>\~</sub>rtd/PTE/PTE5\_011119.pdf. Linked author’s draft dated January 11, 2019. 15, 16

A. Dytso, H. V. Poor, and S. Shamai (Shitz). A general derivative identity for the conditional mean estimator in Gaussian noise and some applications. arXiv preprint arXiv:2104.01883, 2021. URL https://arxiv.org/abs/2104.01883. 15

P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller, H. Saini, Y. Levi, D. Lorenz, A. Sauer,¨ F. Boesel, D. Podell, T. Dockhorn, Z. English, K. Lacey, A. Goodwin, Y. Marek, and R. Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL https: //arxiv.org/abs/2403.03206. 2

O. Fawzi and R. Renner. Quantum conditional mutual information and approximate markov chains. Communications in Mathematical Physics, 340(2):575–611, Sept. 2015. ISSN 1432-0916. doi: 10.1007/s00220-015-2466-x. URL http://dx.doi.org/10.1007/s00220-015-2466-x. 12

I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, 2016. http://www. deeplearningbook.org. 14

F. Handke, F. Koulischer, G. Raya, and L. Ambrogioni. Measuring semantic information production in generative diffusion models, 2025. URL https://arxiv.org/abs/2506.10433. 2

F. Handke, D. Stanceviˇ c, F. Koulischer, T. Demeester, and L. Ambrogioni. The entropic signature of´ class speciation in diffusion models, 2026. URL https://arxiv.org/abs/2602.09651. 2

J. Ho and T. Salimans. Classifier-free diffusion guidance, 2022. URL https://arxiv.org/abs/ 2207.12598. 17

J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. 13

F. Hu, G. Liu, Y. F. Zhang, and X. Gao. Local diffusion models and phases of data distributions, 2025. URL https://arxiv.org/abs/2508.06614. 2, 3, 4, 6, 12, 13

H. Hunt, M. Kamb, and S. Ganguli. An exact information theory of generalization phase transitions in bayesian diffusion models. arXiv preprint arXiv:2607.08041, 2026. 2, 3

M. Kamb and S. Ganguli. An analytic theory of creativity in convolutional diffusion models. arXiv preprint arXiv:2412.20292, 2024. 2, 3

M. Li and S. Chen. Critical windows: non-asymptotic theory for feature emergence in diffusion models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024. 1

M. Li, A. Karan, and S. Chen. Blink of an eye: a simple theory for feature localization in generative models. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=QvqnPVGWAN. 2

A. Lukoianov, C. Yuan, J. Solomon, and V. Sitzmann. Locality in image diffusion models emerges from data statistics. arXiv preprint arXiv:2509.09672, 2025. 2

D. J. C. MacKay. Information Theory, Inference and Learning Algorithms. Cambridge University Press, Cambridge, UK, 2003. ISBN 9780521642989. 13

C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, and S. Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=aBsCjcPu\_tE. 2

M. Niedoba, B. Zwartsenberg, K. Murphy, and F. Wood. Towards a mechanistic explanation of diffusion model generalization. arXiv preprint arXiv:2411.19339, 2024. 2, 3

W. Peebles and S. Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 2

B. Pham, G. Raya, M. Negri, M. J. Zaki, L. Ambrogioni, and D. Krotov. Memorization to generalization: The emergence of diffusion models from associative memory. In NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning, 2024. URL https: //openreview.net/forum?id=zVMMaVy2BY. 1

B. Poole, S. Ozair, A. van den Oord, A. A. Alemi, and G. Tucker. On variational bounds of mutual information, 2019. URL https://arxiv.org/abs/1905.06922. 4

A. Premkumar. On the separability of information in diffusion models, 2026. URL https://arxiv. org/abs/2509.23937. 4

G. Raya and L. Ambrogioni. Spontaneous symmetry breaking in generative diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=lxGFGMMSVl. 2

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet large scale visual recognition challenge. Inter national Journal ofComputer Vision, 115(3):211–252, 2015. doi: 10.1007/s11263-015-0816-y. 5

S. Sang and T. H. Hsieh. Stability of mixed-state quantum phases via finite markov length. Phys. Rev. Lett., 134:070403, Feb 2025. doi: 10.1103/PhysRevLett.134.070403. URL https://link.aps. org/doi/10.1103/PhysRevLett.134.070403. 4

A. Sclocchi, A. Favero, and M. Wyart. A phase transition in diffusion models reveals the hierarchical nature of data. Proceedings of the National Academy of Sciences, 122(1):e2408799121, 2025. doi: 10.1073/pnas.2408799121. URL https://www.pnas.org/doi/abs/10.1073/ pnas.2408799121. 2, 4, 5, 17

K. Shah, A. Kalavasis, A. R. Klivans, and G. Daras. Does generation require memorization? creative diffusion models using ambient diffusion. arXiv preprint arXiv:2502.21278, 2025. 1

J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In F. Bach and D. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR. URL https: //proceedings.mlr.press/v37/sohl-dickstein15.html. 13

Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=PxTIG12RRHS. 13

T. Takahashi, T. Takahashi, and Y. Kabashima. Dynamical regimes of discrete diffusion models, 2026. URL https://arxiv.org/abs/2604.10961. 2

A. Wibisono and V. Jog. Convexity of mutual information along the heat flow. arXiv preprint arXiv:1801.06968, 2018. URL https://arxiv.org/abs/1801.06968. 16

S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16133–16142, 2023. 17

L. Yu, X. Shi, X. Kong, T. Jia, and G. V. Steeg. Mmg: Mutual information estimation via the mmse gap in diffusion, 2025. URL https://arxiv.org/abs/2509.20609. 12

Y. Zhang and S. Gopalakrishnan. Conditional mutual information and information-theoretic phases of decohered gibbs states, 2025. URL https://arxiv.org/abs/2502.13210. 4

Y. F. Zhang, F. Hu, G. Liu, M. Okyay, and X. Gao. Concurrence of symmetry breaking and nonlocality phase transitions in diffusion models, 2026. URL https://arxiv.org/abs/2605.04830. 2, 4, 5, 12

W. Zhu, J. Zhang, L. Yu, K. Yue, and Z. Tang. Dissecting failure dynamics in large language model reasoning. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8893–8914. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.401. URL https://aclanthology.org/2026.acl-long.401/. 2

## A Local Denoising Models and Error Bounds

In this appendix, we review and advance various aspects of local diffusion models. First, we review a sufficient condition for locality of a denoiser in terms of the conditional mutual information, proven already in Hu et al. [2025], for the reader’s convenience. Second, specialising to Gaussian noise relevant for diffusion models, we prove that the same quantity also bounds the locality gap—a quantity introduced in Zhang et al. [2026] to probe the locality transition. This directly connects the observations therein to the CMI, and supports the notion that locality gap is sensitive to the locality transition. Noting that mutual informations are very hard to sample, training a diffusion model and measuring the locality gap provides a way of controlling CMI, which follows the same spirit as Yu et al. [2025].

## A.1 Denoising generative models

Denoising generative models learn to recover clean data from corrupted observations. Let $p , q : \Omega \to$ R be probability densities where Ω is the space of events. Let $\mathcal { N } ( y \mid x )$ be a noise channel. It induces a noisy version of a data density p as

$$
\mathcal { N } ( p ) ( y ) = \int _ { \Omega } \mathrm { d } x \mathcal { N } ( y \mid x ) p ( x ) .\tag{19}
$$

$\mathcal { B } _ { \mathcal { N } , q } ( y \mid x )$ is its Bayes recovery channel with prior $q ,$ defined as

$$
\boldsymbol { B } _ { N , q } ( y \mid \boldsymbol { x } ) = \frac { \mathcal { N } ( y \mid \boldsymbol { x } ) q ( \boldsymbol { x } ) } { \mathcal { N } ( q ) ( \boldsymbol { y } ) } .\tag{20}
$$

Generation begins from a tractable, highly corrupted distribution and successively applies learned approximations to such reverse channels. This framework includes autoregressive models, for which the forward channel progressively masks a suffix of the variables and the reverse process reveals $X _ { i }$ according to $p ( X _ { i } \mid \bar { X } _ { < i } )$ , as well as diffusion models, for which the corruption is gradual, typically Gaussian, and the reverse dynamics are parameterized through a denoiser or score function. The results below are first stated for a generic noise channel and then specialized to diffusion models.

## A.2 Bounding local recovery error with CMI (generic denoising model)

In this section, we prove that CMI bounds the squared total variance (TV) between a distribution and its noised and then locally denoised version. We will not try be rigorous in the measure-theoretic sense, but the statements can be formalised as such if desired.

The nontrivial ingredient in proving the desired result is the classical Fawzi-Renner inequality from Fawzi and Renner [2015], which we now state.

Proposition 1 (Classical Fawzi-Renner Inequality). Let $\hat { p } : \Omega \to { \mathbb { R } }$ be the distribution denoised with prior q after applying the noise channel, $i . e .$

$$
\hat { p } ( x ) : = [ \mathcal { B } _ { \mathcal { N } , q } ( \mathcal { N } ( p ) ) ] ( x ) .
$$

then,

$$
D _ { \mathrm { K L } } ( p \Vert q ) - D _ { \mathrm { K L } } ( \mathcal { N } ( p ) \Vert \mathcal { N } ( q ) ) \geq D _ { \mathrm { K L } } ( p \Vert \hat { p } ) ,\tag{21}
$$

where $D _ { \mathrm { K L } }$ is the Kullback-Leibler divergence/relative entropy.

Now suppose our samples x can be partitioned into three sets, A, B, C with respective r.v.s $x _ { A } , x _ { B } , x _ { C }$ independent of any spatial geometry. Further suppose the noise channel acts only on $\begin{array} { r } { A , \mathrm { i . e . , } \mathcal { N } ( p ) \bar { ( \boldsymbol { y } ) } = \sum _ { \boldsymbol { x } _ { A } \in \Omega _ { A } } \bar { \mathcal { N } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \bar { p } ( \boldsymbol { x } ) } \end{array}$ where, in particular, $y _ { B } = x _ { B }$ and $y _ { C } = x _ { C }$ . We also denote the clean variables as X and the noisy variables as Y . Then, taking $p = p ( x )$ and $q ( x ) = p ( x _ { A } , x _ { B } ) p ( x _ { C } )$ , by definition of the mutual information $I ( X ; Y )$ between two r.v.s X and Y as the relative entropy between the joint and the product distribution, we have

$$
D _ { \mathrm { K L } } ( p \| q ) = I ( A B ; C ) , \quad D _ { \mathrm { K L } } ( \mathcal { N } ( p ) \| \mathcal { N } ( q ) ) = I ( \tilde { A } B ; C ) .\tag{22}
$$

The CMI is defined in terms of MI’s, for our application, we write (for Z a generic r.v.)

$$
I ( Z B ; C ) = I ( Z ; C \mid B ) + I ( B ; C ) .\tag{23}
$$

The term $I ( B ; C )$ cancels between the two MI’s, giving

$$
D _ { \mathrm { K L } } ( p | | q ) - D _ { \mathrm { K L } } ( { \cal N } ( p ) | | { \cal N } ( q ) ) = I ( A ; C \mid B ) - I ( \tilde { A } ; C \mid B ) \le I ( A ; C \mid B ) ,
$$

by positivity of the CMI $I ( \tilde { A } ; C \mid B ) \ge 0$ . Thus, we have obtained

$$
I ( A ; C \mid B ) \ge D _ { \mathrm { K L } } ( p \| \hat { p } ) .\tag{24}
$$

Lastly, by Pinsker’s inequality MacKay [2003], we have $D _ { \mathrm { K L } } ( p \| \hat { p } ) ~ \geq ~ 2 \mathrm { T V } ( p \| \hat { p } ) ^ { 2 }$ where $\begin{array} { r } { \mathrm { T V } ( \dot { p } \| q ) \dot { = } \int _ { \Omega } | p ( x ) - \hat { p } ( \dot { x } ) | / 2 . } \end{array}$ Chaining the inequalities together, we obtain the following theorem. Theorem 3 (CMI bounds local reconstruction error Hu et al. [2025].). For a distribution p, a noise channel N acting only on A, with an ABC tripartition and with $\begin{array} { r l } { \hat { p } ( x ) } & { { } = } \end{array}$ $[ B _ { N , p ( x _ { A } , x _ { B } ) p ( x _ { C } ) } ( { N ( p ) } ) ] ( x )$ we have

$$
\begin{array} { r } { 2 \mathrm { T V } ( p \| \hat { p } ) ^ { 2 } \leq I ( A ; C \mid B ) , } \end{array}\tag{25}
$$

i.e., CMI ofthe distribution before the noise is added controls the locality ofthe denoiser.

The last thing to clear up is to show that the denoiser in Theorem 3 is truly local. This follows by definition, where $q = p ( x _ { A } , x _ { B } ) p ( x _ { C } )$ and

$$
\mathcal { B } _ { N , q } ( \mathcal { N } ( p ) ) ( x ) = \frac { \mathcal { N } ( y _ { A } \mid x _ { A } ) p ( x _ { A } x _ { B } ) p ( x _ { C } ) } { \int \mathrm { d } x _ { A } \mathcal { N } ( y _ { A } \mid x _ { A } ) p ( x _ { A } x _ { B } ) p ( x _ { C } ) } = \frac { \mathcal { N } ( y _ { A } \mid x _ { A } ) p ( x _ { A } x _ { B } ) } { \int \mathrm { d } x _ { A } \mathcal { N } ( y _ { A } \mid x _ { A } ) p ( x _ { A } x _ { B } ) } .\tag{26}
$$

Suppose we break down a single round of diffusion noise pixel-by-pixel, and only consider a single step of this round. Taking A to be the pixel to whom the noise is added, the CMI bounds the total variance (by Pinsker’s inequality and Fawzi-Renner, as applied in Hu et al., 2025, Section III.2 and Supplementary Material S2.A

$$
\mathrm { T V } ( p \| \hat { p } ) ^ { 2 } \leq I ( A ; C \mid B ) ,\tag{27}
$$

where $\mathrm { T V } ( p \| q )$ is the total variation $\textstyle \sum _ { x \in \Omega } | p ( x ) - q ( x ) | / 2$ and $\hat { p } ( x ) : = ( \mathcal { B } _ { \mathcal { N } , p ( x _ { A } , x _ { B } ) } ( \mathcal { N } _ { A } ( p ) ) ( x )$ is the denoised distribution after adding noise to P only on A via the noise channel $\mathcal { N } ( y _ { A } \mid x _ { A } )$ . Lastly, $B _ { \mathcal { N } , \mathcal { N } _ { A } }$ is the “Bayes recovery channel” given by Bayes’ rule $\mathcal { B } _ { N , q } ( x \mid y ) = \mathcal { N } ( y \mid x ) q ( x ) / \mathcal { N } ( q ) ( \dot { y } )$ for any probability distribution $q ( x )$ . If the CMI is zero, the total variance between the initial distribution p and the noised–locally-denoised pˆ is zero—the local denoiser is exact. If instead, the CMI is not small but decays exponentially in the radius of B, i.e. if we have

$$
I ( A ; C \mid B ) \leq \gamma \exp ( - r / \xi ) ,\tag{28}
$$

where $\xi$ is known as the Markov length [Hu et al., 2025, Section III.1], we can pick a r small enough to guarantee a total variation error of ϵ, i.e., demanding $\epsilon ^ { 2 } \geq \gamma \exp ( - r / \xi )$ , we get

$$
r \gtrsim \xi \log \frac { \sqrt { \gamma } } { \epsilon } .\tag{29}
$$

Then, stitching together many single-step diffusions, we may derive a similar bound on the full diffusion path [Hu et al., 2025, Section III.3] where each individual step size must at least be $r \geq \xi \log ( \bar { N } K \sqrt { \gamma } / \epsilon )$ , where K is the window size and N is the number of discretization steps. As such, a decaying CMI guarantees a local denoiser.

## A.3 Diffusion models

Diffusion models, as a special type of denoising generative models, propose a parametrisation of a data distribution based on the Langevin SDE Sohl-Dickstein et al. [2015], Ho et al. [2020], Song et al. [2021]

$$
\mathrm { d } X _ { t } = \mu ( X _ { t } , t ) \mathrm { d } t + \sigma ( t ) \mathrm { d } \eta ,\tag{30}
$$

where $\eta$ is a Gaussian random variable, drawn independently at each time step. Under suitable assumptions, there exists a corresponding Fokker-Planck equation (FPE)

$$
\partial _ { t } P = - \partial _ { x } ( \mu ( x , t ) P ) + \partial _ { x } ( \sigma ^ { 2 } ( t ) \partial _ { x } P ) / 2 .\tag{31}
$$

The FPE admits an exact time-reversed form Song et al. [2021]. Let $Q ( x , t ) = P ( x , T - t )$ with T the reversal time

$$
\partial _ { t } Q = - \partial _ { x } ( \tilde { \mu } ( x , T - t ) Q ) + \partial _ { x } ( \sigma ^ { 2 } ( T - t ) \partial _ { x } Q ) / 2 ,\tag{32}
$$

with $\tilde { \mu } ( x , t ) = - \mu ( x , t ) + \sigma ^ { 2 } ( t ) \partial _ { x }$ log $Q$ . This implies the reversed SDE has a modified drift shifted $y - \sigma ^ { 2 } ( t ) \partial _ { x } \log Q$ , which we need to know to be able to reverse each trajectory. The hard part is the so-called score function $\partial _ { x } \log Q$ , which requires the knowledge of the full distribution $P ( x , t )$ . The application to generative modeling takes $P ( x , 0 ) = p _ { \mathrm { d a t a } } .$ , such that $P ( x , t  \infty ) =$ $\mathcal { N } ( \mu ( \dot { x } ) , \sigma ^ { 2 } ( \infty ) )$ . Since sampling the late-time Gaussian is easy, the difficulty of sampling p<sub>data</sub> is purely relegated to the reversal process. Interestingly, this reveals that we do not need all of $p _ { \mathrm { d a t a } } .$ we only need the gradient of its log, the score function. Therefore, we do not care about multiplicative constants in $p _ { \mathrm { d a t a } }$ , which are intractable to compute in many cases. To see this, take an energy based model $p _ { \theta } ( x ) \bar { } = \exp ( - E _ { \theta } ( x ) ) / Z _ { \theta }$ [Goodfellow et al., 2016, Section 18.4]. The score is

$$
s _ { \theta } ( x ) = - \partial _ { x } \log Z _ { \theta } - \partial _ { x } E _ { \theta } ( x ) ,\tag{33}
$$

and the first term is zero log $Z _ { \theta }$ does not depend on x, and as such, is zero.

## A.4 Bounding local denoiser error with CMI (diffusion model)

While we have provided an information-theoretic guarantee on when a local score may be constructed, we have not operationalized what it means to have a local score function and how to interpret deviations from locality in terms of optimal denoising.

Define the expectation of a function $f ( x )$ over the noisy data distribution be $p _ { t } ( x )$ . We now will argue, in expectation, that the score function of the marginal distribution $p ( x _ { A B } )$ (obtained by marginalizing over $x _ { C } )$ is the smallest over all t possible local approximations. Suppose we approximate the score $\nabla _ { A } \log { p _ { t } ( x ) }$ by a function $f ( x _ { A B } )$ . The error in the approximation satisfies

$$
\begin{array} { r l } & { \mathbb { E } _ { p t } \Vert \nabla _ { A } \log p _ { t } ( x ) - f ( x _ { A B } ) \Vert ^ { 2 } = \mathbb { E } _ { p t } \Vert \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \Vert ^ { 2 } } \\ & { \qquad + \mathbb { E } _ { p t } \Vert \nabla _ { A } \log p _ { t } ( x _ { A B } ) - f ( x _ { A B } ) \Vert ^ { 2 } , } \end{array}\tag{34}
$$

and as such, the best local approximation is the score of the $A B$ marginal. Any other approximation incurs an extra cost $\| \nabla _ { A } \log \dot { p } _ { t } ( x _ { A B } ) - f ( x _ { A B } ) \| ^ { 2 }$ in expectation.

The proof follows directly, by adding and subtracting the local score marginal $\nabla _ { A } \log p _ { t } ( x _ { A B } )$ which yields

$$
\begin{array} { r l } & { \mathbb { E } _ { p _ { t } ( x ) } \| \nabla _ { A } \log p _ { t } ( x ) - f ( x _ { A B } ) \| ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { p _ { t } ( x ) } \| \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \| ^ { 2 } + \mathbb { E } _ { p _ { t } ( x ) } \| \nabla _ { A } \log p _ { t } ( x _ { A B } ) - f ( x _ { A B } ) \| ^ { 2 } } \\ & { \qquad + 2 \mathbb { E } _ { p _ { t } ( x ) } \bigl ( \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \bigr ) \cdot \bigl ( \nabla _ { A } \log p _ { t } ( x _ { A B } ) - f ( x _ { A B } ) \bigr ) . } \end{array}
$$

Noting the expectation is over $p _ { t } ( x )$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { p _ { t } ( x ) } \big ( \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \big ) \cdot \big ( \nabla _ { A } \log p _ { t } ( x _ { A B } ) - f ( x _ { A B } ) \big ) } \\ & { \quad = \mathbb { E } _ { p _ { t } ( x _ { A B } ) } \big [ \big ( \nabla _ { A } \log p _ { t } ( x _ { A B } ) - f ( x _ { A B } ) \big ) \cdot \mathbb { E } _ { p _ { t } ( x _ { C } | x _ { A B } ) } \big ( \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \big ) \big ] . } \end{array}\tag{35}
$$

By definition, $\begin{array} { r } { \int \mathrm { d } \boldsymbol { x } _ { C } p _ { t } ( \boldsymbol { x } _ { C } \mid \boldsymbol { x } _ { A B } ) \nabla _ { A } \log p _ { t } ( \boldsymbol { x } ) = \nabla _ { A } \log p _ { t } ( \boldsymbol { x } _ { A B } ) } \end{array}$ , and the cross term is zero.

Thus, the error in any local approximation is the locality gap

$$
\Delta _ { \mathrm { l o c } } ( t ) ^ { 2 } = \mathbb { E } _ { p _ { t } ( x ) } \Vert \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) \Vert ^ { 2 } ,\tag{36}
$$

which is a fundamental probe of the locality of the true score function: it is zero, if and only if, the true score is local. While the locality gap is a so-called instantenous probe—it probes the score at a single time instance and does not determine directly the output ‘quality’—it nevertheless describes the downsteam error in the generated image: using Tweedie’s identity applied to the $A B$ marginal, one can show

$$
\mathbb { E } [ x _ { A } \mid x ( t ) ] - \mathbb { E } [ x _ { A } \mid x _ { A B } ( t ) ] = \frac { t ^ { 2 } } { 1 - t } ( \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x _ { A B } ) ) ,\tag{37}
$$

i.e., the vector difference in the locality gap is the change in the optimal prediction $x _ { A }$ due to revealing $C ,$ and also

$$
\mathbb { E } [ \vert x _ { A } - \mathbb { E } [ x _ { A } \vert x _ { A B } ( t ) ] \vert ^ { 2 } ] - \mathbb { E } [ \vert x _ { A } - \mathbb { E } [ x _ { A } \vert x ( t ) ] \vert ^ { 2 } ] = \frac { t ^ { 4 } } { ( 1 - t ) ^ { 2 } } \Delta _ { \mathrm { l o c } } ^ { 2 } ,\tag{38}
$$

i.e., the locality gap is a direct probe of the extra denoising error on A incurred by hiding C in the optimal denoiser. In this sense, the norms of the instantenous probes control pixelwise differences/errors in the denoised output.

Furthermore, just like the total variation between the data distribution P and its noised-then-locallydenoised version ${ \hat { P } } ,$ , we can also show that CMI controls the size of the locality gap. In particular, it is possible to show

Proposition 2 (Controlling the locality gap by CMI). Assume

$$
X ( t ) = \alpha ( t ) X ( 0 ) + \sigma ( t ) Z , \qquad \sigma ( t ) > 0 ,
$$

where $Z$ is an independent standard Gaussian and X(0) has finite second moment. Write $m =$ dim $X _ { A }$ , and define the locality gap by

$$
\begin{array} { r } { \Delta _ { \mathrm { l o c } } ( t ) : = \left( \mathbb { E } _ { p _ { t } } \| \nabla _ { A } \log p _ { t } ( \boldsymbol { x } ( t ) ) - \nabla _ { A } \log p _ { t } ( \boldsymbol { x } _ { A B } ( t ) ) \| ^ { 2 } \right) ^ { 1 / 2 } , } \end{array}
$$

where $\| \cdot \|$ is the Euclidean norm on the A coordinates. Then, with natural logarithms,

$$
\Delta _ { \mathrm { l o c } } ( t ) ^ { 2 } \leq \frac { 2 \sqrt { m ( m + 3 ) } } { \sigma ( t ) ^ { 2 } } \sqrt { I _ { t } ( A ; C \mid B ) } .
$$

Proof. Fix t. Tweedie’s formula and the Hatsell–Nolte identity [Dytso et al., 2021, Eq. (3) and Proposition 1] give

$$
\nabla _ { A } ^ { 2 } \log p _ { t } ( x ( t ) ) = - { \frac { I _ { m } } { \sigma ( t ) ^ { 2 } } } + { \frac { \operatorname { C o v } ( \alpha ( t ) X _ { A } ( 0 ) \mid X ( t ) ) } { \sigma ( t ) ^ { 4 } } } = { \frac { \operatorname { C o v } ( Z _ { A } \mid X ( t ) ) - I _ { m } } { \sigma ( t ) ^ { 2 } } } .\tag{39}
$$

The second equality uses $\alpha ( t ) X _ { A } ( 0 ) = X _ { A } ( t ) - \sigma ( t ) Z _ { A }$ at fixed $X ( t )$

We use two standard moment identities. Here $\Vert \cdot \Vert _ { \mathrm { F } }$ denotes the Frobenius norm, whose square is the sum of squared matrix entrie. First, the $L ^ { 2 } .$ -contraction ofconditional expectation states that, for any square-integrable scalar random variable $V$ and side information $S ,$

$$
\mathbb { E } \Big [ ( \mathbb { E } [ V \mid S ] ) ^ { 2 } \Big ] \leq \mathbb { E } [ V ^ { 2 } ] .
$$

This follows from conditional Jensen’s inequality and the tower property [Durrett, 2019, Theorem 4.1.11].

By Wick’s theorem for Gaussian moments we have,

$$
\mathbb { E } \Vert Z _ { A } \Vert ^ { 4 } = \sum _ { i , j = 1 } ^ { m } \mathbb { E } [ Z _ { A , i } ^ { 2 } Z _ { A , j } ^ { 2 } ] = \sum _ { i , j = 1 } ^ { m } ( 1 + 2 \delta _ { i j } ) = m ( m + 2 ) .
$$

Since a covariance matrix is positive semidefinite,

$$
\| \operatorname { C o v } ( Z _ { A } \mid X ( t ) ) \| _ { \mathrm { F } } \leq \operatorname { t r } \operatorname { C o v } ( Z _ { A } \mid X ( t ) ) \leq \mathbb { E } [ \| Z _ { A } \| ^ { 2 } \mid X ( t ) ] .
$$

Thus, expanding equation 39, dropping the nonpositive trace term, and applying the two moment identities above gives

$$
\begin{array} { r l } { \mathbb { E } _ { p _ { t } } \Vert \nabla _ { A } ^ { 2 } \log p _ { t } ( x ( t ) ) \Vert _ { \mathrm { F } } ^ { 2 } = \frac { m - 2 \mathbb { E } \mathrm { t r } \operatorname { C o v } ( Z _ { A } \mid X ( t ) ) + \mathbb { E } \Vert \operatorname { C o v } ( Z _ { A } \mid X ( t ) ) \Vert _ { \mathrm { F } } ^ { 2 } } { \sigma ( t ) ^ { 4 } } } & { } \\ { \leq \frac { m + \mathbb { E } \left[ \left( \mathbb { E } [ \Vert Z _ { A } \Vert ^ { 2 } \mid X ( t ) ] \right) ^ { 2 } \right] } { \sigma ( t ) ^ { 4 } } } & { } \\ { \leq \frac { m + \mathbb { E } \Vert Z _ { A } \Vert ^ { 4 } } { \sigma ( t ) ^ { 4 } } = \frac { m ( m + 3 ) } { \sigma ( t ) ^ { 4 } } . } \end{array}\tag{40}
$$

Now add a fictitious Gaussian noise of variance $u \geq 0$ to A alone. The resulting A variable has the same distribution as

$$
\alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } ,
$$

while $X _ { B } ( t )$ and $X _ { C } ( t )$ remain unchanged. The Gaussian noise $Z _ { A }$ is still independent of $( X ( 0 ) , X _ { B } \dot { ( t ) } , X _ { C } ( t ) )$ . Therefore equation 40 remains valid with $\sigma ( t ) ^ { 2 }$ replaced by $\sigma ( t ) ^ { \widehat { } 2 } + u$

The de Bruijn identity states that, for a Gaussian-smoothed density $\rho _ { u } ( y )$ evolving according to

$$
\partial _ { u } \rho _ { u } = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } \partial _ { y _ { i } } ^ { 2 } \rho _ { u } ,
$$

the entropy satisfies

$$
\frac { d } { d u } \left[ - \int _ { \mathbb { R } ^ { m } } \rho _ { u } ( y ) \log \rho _ { u } ( y ) d y \right] = \frac { 1 } { 2 } \int _ { \mathbb { R } ^ { m } } \rho _ { u } ( y ) \| \nabla _ { y } \log \rho _ { u } ( y ) \| ^ { 2 } d y
$$

[Wibisono and Jog, 2018, Lemma 1]. It also holds for conditional entropy when the conditioning variables are unchanged by the added noise: apply the identity to each conditional density and average.

Apply this identity to $I ( A ; C \mid B ) = H ( A \mid B ) - h ( A \mid B , C )$ where $H ( A )$ is the differential entropy. The marginal score is the conditional expectation of the full score,

$$
\begin{array} { r } { \mathbb { E } [ \nabla _ { A } \log p _ { t } ( \boldsymbol { X } ( t ) ) \vert \boldsymbol { X } _ { A B } ( t ) ] = \nabla _ { A } \log p _ { t } ( \boldsymbol { x } _ { A B } ( t ) ) . } \end{array}
$$

The orthogonal-projection property of conditional expectation [Durrett, 2019, Theorem 4.1.15] therefore gives

$$
\begin{array} { r l } {  { \frac { d } { d u } I \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } : X _ { C } ( t ) \mid X _ { B } ( t ) \Big ) \bigg | _ { u = 0 } } } \\ & { \mathrm { ~ \ ~ \ } = \frac { 1 } { 2 } \mathbb { E } _ { p _ { t } } \big [ \| \nabla _ { A } \log p _ { t } ( x _ { A B } ( t ) ) \| ^ { 2 } - \| \nabla _ { A } \log p _ { t } ( x ( t ) ) \| ^ { 2 } \big ] } \\ & { \mathrm { \ ~ \ } = - \frac { 1 } { 2 } \Delta _ { \mathrm { l o c } } ( t ) ^ { 2 } . } \end{array}\tag{41}
$$

The Fisher-information dissipation identity states, for the same Gaussian heat flow, that

$$
\frac { d } { d u } \int _ { \mathbb { R } ^ { m } } \rho _ { u } ( y ) \| \nabla _ { y } \log \rho _ { u } ( y ) \| ^ { 2 } d y = - \int _ { \mathbb { R } ^ { m } } \rho _ { u } ( y ) \| \nabla _ { y } ^ { 2 } \log \rho _ { u } ( y ) \| _ { \mathrm { F } } ^ { 2 } d y
$$

[Wibisono and Jog, 2018, Lemma 1]. Together with de Bruijn’s identity, this says that the second entropy derivative is minus one half of the mean squared log-density Hessian.

Apply this to the two conditional entropies in the CMI, we obtain

$$
\begin{array} { r l } & { \frac { d ^ { 2 } } { d u ^ { 2 } } \frac { J } { J } \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } : X _ { C } ( t ) \ | \ X _ { B } ( t ) \Big ) } \\ & { \quad = \frac { 1 } { 2 } \mathbb { E } \Big \| \nabla _ { A } ^ { 2 } \log p \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } \Big \vert \ X _ { B } ( t ) , X _ { C } ( t ) \Big ) \Big \| _ { \mathrm { F } } ^ { 2 } } \\ & { \quad \quad - \frac { 1 } { 2 } \underline { { \mathbb { E } } } \Big \| \nabla _ { A } ^ { 2 } \log p \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } \Big \vert \ X _ { B } ( t ) \Big ) \Big \| _ { \mathrm { F } } ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \geq 0 } \\ & { \leq \frac { 1 } { 2 } \mathbb { E } \Big \| \nabla _ { A } ^ { 2 } \log p \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } \Big \vert \ X _ { B } ( t ) , X _ { C } ( t ) \Big ) \Big \| _ { \mathrm { F } } ^ { 2 } } \\ & { \quad = \frac { 1 } { 2 } \mathbb { E } \Big \| \nabla _ { A } ^ { 2 } \log p \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } , X _ { B } ( t ) , X _ { C } ( t ) \Big ) \Big \| _ { \mathrm { F } } ^ { 2 } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \leq \frac { m ( m + 3 ) } { 2 ( \sigma ( t ) ^ { 2 } + u ) ^ { 2 } } \leq \frac  m ( m  \end{array}\tag{42}
$$

where we used equation 40 for the second last inequality. Integrating equation 42 twice gives Taylor’s quadratic upper bound Boyd and Vandenberghe [2004]: if $f ^ { \prime \prime } ( u ) \leq L$ for $u \geq 0$ , then

$$
f ( u ) \leq f ( 0 ) + u f ^ { \prime } ( 0 ) + \frac { L } { 2 } u ^ { 2 } .
$$

Using equation 41 and nonnegativity of CMI, we obtain

$$
\begin{array} { r l } & { 0 \le I \Big ( \alpha ( t ) X _ { A } ( 0 ) + \sqrt { \sigma ( t ) ^ { 2 } + u } Z _ { A } : X _ { C } ( t ) \mid X _ { B } ( t ) \Big ) } \\ & { \quad \le I _ { t } ( A ; C \mid B ) - \cfrac { u } { 2 } \Delta _ { \mathrm { l o c } } ( t ) ^ { 2 } + \cfrac { m ( m + 3 ) } { 4 \sigma ( t ) ^ { 4 } } u ^ { 2 } . } \end{array}
$$

Choose the nonnegative minimizer of this quadratic,

$$
u = \frac { \sigma ( t ) ^ { 4 } \Delta _ { \mathrm { l o c } } ( t ) ^ { 2 } } { m ( m + 3 ) } .
$$

Substitution gives

$$
\frac { \sigma ( t ) ^ { 4 } \Delta _ { \mathrm { l o c } } ( t ) ^ { 4 } } { 4 m ( m + 3 ) } \leq I _ { t } ( A ; C \mid B ) .
$$

Rearranging and taking a square root proves the proposition.

## A.5 Conditional scores and Classifier-Free Guidance

An important aspect of diffusion models is that they produce images that are faithful to classes of semantic information; a good model trained on cat and dog images produces one animal at a time, it does not produce an amalgamation of the two. How does the model steer towards a specific semantic class? To model this behaviour, we assume the data distribution is a joint distribution $p ( x , s )$ between images x and labels s where the diffusion noise only acts on the conditional image distribution, i.e., $p _ { t } ( x , s ) = p ( s ) p _ { t } ( x \mid s )$ . Since the model only has access to the image marginal score, we can write the score as using Bayes’ rule

$$
\nabla _ { A } \log { p _ { t } ( x ) } = \nabla _ { A } \log { p _ { t } ( x \mid s ) } - \nabla _ { A } \log { p _ { t } ( s \mid x ) } .\tag{43}
$$

Rearranging this equation, we obtain the conditioning gap

$$
\Delta _ { \mathrm { c o n d } } ( t ) ^ { 2 } : = \mathbb { E } \Vert \nabla _ { A } \log p _ { t } ( s \mid x ) \Vert ^ { 2 } = \mathbb { E } \Vert \nabla _ { A } \log p _ { t } ( x ) - \nabla _ { A } \log p _ { t } ( x \mid s ) \Vert ^ { 2 } ,\tag{44}
$$

whose norm answers two questions: (i) how sensitive is a global classifier to a change in A or (ii) how much does conditioning change the global score function? Just like the locality gap, it also is the fundamental score error in approximating the conditional score by any unconditional function of the global image. Furthermore, by using Tweedie on the label-conditional distribution on the full image $p _ { t } ( x \mid s )$ , we can show that the conditioning gap controls both the pixelwise difference in the denoised image, as well as the extra squared error in the downstream sample due to hiding/revealing the semantic label s. Operationally, the score difference $\nabla _ { A }$ log $p _ { t } ( s \mid x )$ is precisely the term added in diffusion models by classifier guidance Dhariwal and Nichol [2021] and the application of Bayes rule to expand it in terms of the conditional and unconditional scores is the basis of classifier-gree guidance Ho and Salimans [2022] which drops the need to train an explicit classifier to obtain the gradient $\nabla _ { A } \log p _ { t } ( s \mid x )$ . Thus, the conditioning gap probes how strongly guidance can change the score on A at each time: a small gap means that even revealing the label provides little additional direction for denoising.

## B Semantic Speciation for Local Region

We adapt the forward–backward ImageNet protocol of Sclocchi et al. Sclocchi et al. [2025], using the same unconditional 256-pixel diffusion checkpoint and 250-step respaced reverse sampler. Here $t \in [ 0 , 1 ]$ denotes normalized diffusion time. We select one validation image from each of 100 classes in the ImageNet-1K (ILSVRC2012) validation set. A separate classifier chooses a class-bearing $6 4 \times 6 4$ crop from 25 candidate locations; a 128 × 128 crop uses the same center, and the global observation is the full image. Local crops are enlarged to 256 pixels before noising, so the reverse process receives no pixels outside the selected region. We draw one reverse sample per image and time.

Following Sclocchi et al., we measure cosine similarity between classifier logits of each source and reconstruction, see Figure 3. We standardize ConvNeXt V2 Large [Woo et al., 2023] logits using 1,000 clean reference images and plot the most populated bin of the 100 pairwise similarities. The figure uses fixed 0.10-wide bins throughout.

This binned cosine peak is distinct from $P _ { \mathrm { F B } } ( R , t ) ;$ ; enlarged crops also differ from the full images used to train the denoiser.

![](images/7d0c0804df93a938d2dcfa921372e9b7a835910d04d90000218518975318604f.jpg)  
Figure 3: Label–reconstruction similarity remains high at early diffusion times and then falls toward zero. The sharp drop occurs first for the $6 4 \times 6 4$ crop, then for the $1 2 8 \times 1 2 8$ crop, and last for the full image.

## C Proof of Lemma 1

Proof. The mutual information is the average divergence between the posterior and the prior:

$$
I _ { t } ( S ; R ) = \mathbb { E } \sum _ { s } q _ { s } \log { \frac { q _ { s } } { p ( s ) } } .
$$

For each observation, Pinsker’s inequality bounds this divergence below by $\begin{array} { r } { \frac { 1 } { 2 } \mathbb { E } ( \sum _ { s } | q _ { s } - p ( s ) | ) ^ { 2 } } \end{array}$ which is at least $\begin{array} { r } { \frac { 1 } { 2 } \mathbb { E } \sum _ { s } ( q _ { s } - p ( s ) ) ^ { 2 } } \end{array}$ . The inequality log $u \leq u - 1$ bounds it above by $\sum  _ { s } ( q _ { s } -$ $p ( s ) ) ^ { 2 } / p ( s )$ . Averaging proves equation 9.

For equation 10, Jensen’s inequality gives $- \sum _ { s } q _ { s }$ log $q _ { s } \geq - \log \sum _ { s } q _ { s } ^ { 2 }$ . Average this inequality and use Jensen’s inequality together with $\begin{array} { r } { P _ { \mathrm { F B } } ( R , \bar { t } ) = \mathbb { E } \sum _ { s } q _ { s } ^ { 2 } } \end{array}$ to get ${ \bar { H } } ( S \mid X _ { R , t } ) \geq - \log P _ { \mathrm { F B } } ( R , t )$ This is the first bound because $I _ { t } ( S ; R ) - \dot { H } ( \dot { S } ) = - \overline { { H } } \overset { \circ } { ( } \tilde { S } ^ { \prime } \mid \check { X _ { R , t } } )$

For the other bound, define

$$
E = \left\{ 0 , \ S = \widehat { S } , \qquad e : = \operatorname* { P r } ( E = 1 ) = 1 - P _ { \mathrm { F B } } ( R , t ) . \right.
$$

The original and returned labels are independent given $X _ { R , t } ,$ , so knowing the returned label does not reduce $\mathbf { \bar { \cal H } } ( S \mid X _ { R , t } )$ . The entropy chain rule therefore gives

$$
\begin{array} { r l } & { H ( S \mid X _ { R , t } ) = H ( S \mid X _ { R , t } , \widehat { S } ) } \\ & { \qquad = H ( S , E \mid X _ { R , t } , \widehat { S } ) } \\ & { \qquad = \underbrace { H ( E \mid X _ { R , t } , \widehat { S } ) } _ { \mathrm { ~ W a s ~ t h e ~ g u e s s ~ w r o n g ? ~ } } + \underbrace { H ( S \mid E , X _ { R , t } , \widehat { S } ) } _ { \mathrm { ~ I f ~ w r o n g , ~ w h i c h ~ l a b e l ? ~ } } } \end{array}\tag{45}
$$

The uncertainty in this yes-or-no answer, without any additional information, is $H ( E ) = h _ { 2 } ( e )$ Knowing $X _ { R , t }$ and $\widehat { S }$ can only reduce that uncertainty. Thus

$$
H ( E \mid X _ { R , t } , { \widehat { S } } ) \leq h _ { 2 } ( e ) .
$$

Now suppose we have been told the value of $E .$ When $E = 0 .$ , the original label is exactly ${ \widehat { S } } .$ There is no remaining uncertainty:

$$
H ( S \mid E = 0 , X _ { R , t } , \widehat { S } ) = 0 .
$$

When $E = 1$ , the original label cannot equal ${ \widehat { S } } ,$ so there are at most $L - 1$ possibilities. A distribution over $L - 1$ possibilities has entropy at most $\log ( L - 1 )$

$$
H ( S \mid E = 1 , X _ { R , t } , \widehat { S } ) \leq \log ( L - 1 ) .
$$

The second situation occurs with probability $e .$ Averaging the two cases gives

$$
H ( S \mid E , X _ { R , t } , { \widehat { S } } ) \leq ( 1 - e ) \cdot 0 + e \log ( L - 1 ) .
$$

Substitute into equation 45 we obtain the second bound in equation 10.

## D Empirical probe of the common-cause hypothesis

We probe the common-cause interpretation in Stable Diffusion 3 Medium (Figure 2). Each scene has three nested descriptions: longer versions retain the shorter description and append semantic details. We generate trajectories at 1024×1024 resolution using 30 FlowMatch Euler steps (scheduler shift 3), global attention throughout, and classifier-free guidance of scale 4. At every pre-update latent state, we evaluate the same pretrained weights with global attention and with local attention implemented using FlexAttention. The local variant restricts image–image attention to a clipped $1 5 \times 1 5$ token neighborhood (Chebyshev radius 7) in all 24 transformer blocks, while retaining all text connections. For each description-length group, we measure

$$
\Delta _ { b } ( t ) = \mathbb { E } _ { i , r } \left[ \frac { 1 } { d } \left\| v _ { \theta } ^ { \mathrm { l o c } } ( x _ { t } ^ { i , r } , t , c _ { i , b } ) - v _ { \theta } ^ { \mathrm { g l o b } } ( x _ { t } ^ { i , r } , t , c _ { i , b } ) \right\| _ { 2 } ^ { 2 } \right] , \qquad b \in \{ \mathrm { C } , \mathrm { U } \} ,\tag{46}
$$

where $d$ is the number of latent coordinates, $c _ { i , \mathrm { C } }$ is the scene description, $c _ { i , \mathrm { U } } = \emptyset$ is the empty prompt, and the empirical expectation averages scenes i and seeds r. Both branches are evaluated on the same globally guided trajectory generated for the corresponding description; the predictions v<sub>θ</sub> are measured before applying guidance. Because SD3 predicts flow velocity, $\Delta _ { b }$ is a score-gap proxy.

In Figure 2, lines average over seeds and then scenes; shading denotes pointwise 95% confidence intervals from 20,000 paired scene-cluster bootstrap resamples, stratified by subject category. The time-weighted reduction $\begin{array} { r } { 1 - \sum _ { k } w _ { k } \Delta _ { \mathrm { c o n d } } ( t _ { k } ) / \sum _ { k } \mathrm { \hat { w } } _ { k } \Delta _ { \mathrm { u n c o n d } } ( t _ { k } ) } \end{array}$ , with $w _ { k } = t _ { k } - t _ { k + 1 }$ , is $2 0 . 7 \%$ 23.4%, and 24.4% for short, medium, and long descriptions. Diffusion time increases from clean $( t = 0 )$ to noise $( t = 1 )$ ; generation proceeds right to left.

These results provide noise-dependent evidence consistent with semantic common causes. Attention truncation remains an operational probe rather than an exact marginal-score construction, so the experiment does not directly establish the mutual-information inequality in Hypothesis 1.

## E Gaussian Mixture Calculations

In this Appendix we provide details of the calculations of the score gaps and the CMI for Gaussian mixtures of various kinds.

## E.1 Scores for Gaussian Mixtures

In this subsection we compute various scores and score gaps for Gaussian mixtures, for which almost all results are analytic.

We consider $N \times N$ images as a random variable $X _ { 0 }$ , whose instances are flattened into vectors $\vec { x } _ { 0 } \in \mathbb { R } ^ { N ^ { 2 } }$ and use an interpolation $X _ { t } = ( 1 - t ) X _ { 0 } + t Z$ where $Z \sim \mathcal { N } ( 0 , \mathbb { I } _ { N ^ { 2 } } )$ . Take the joint distribution of labels $S = \pm 1$ and images X to be an equal-weight ferromagnetic Gaussian mixture where $p ( S = s ) = 1 / 2 , p ( x _ { 0 } \mid S = s ) \sim { \mathcal { N } } ( s \mu { \vec { 1 } } , \sigma ^ { 2 } \mathbb { I } _ { N ^ { 2 } } )$ , where $\vec { 1 } \in \mathbb { R } ^ { \breve { N } ^ { 2 } }$ is the vector of ones, $\mu \in \mathbb { R }$ is a parameter setting the $O ( 1 )$ separation scale of the means, and $\sigma ^ { 2 }$ is the intrinsic variance of the distribution. The observed distribution is the image marginal

$$
p ( \vec { x } _ { 0 } ) = \frac { 1 } { Z _ { \mu , \sigma ^ { 2 } } } \sum _ { s = \pm 1 } \left[ \exp ( - \frac { 1 } { 2 \sigma ^ { 2 } } ( \vec { x } - s \mu \vec { 1 } ) ^ { 2 } ) \right] .\tag{47}
$$

Since sum of Gaussian random variables remains a Gaussian, the noised distribution itself is a two-component mixture with time-dependent parameters $\mu _ { t } = \mu ( 1 - t )$ and $\sigma _ { t } ^ { 2 } = ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 }$ We thus perform calculations suppressing the time dependence and restore as needed.

We first rewrite the s-conditional Gaussian of the marginal image distribution on some region R as

$$
p ( x _ { R } \mid s ) = { \frac { \alpha ( x _ { R } ) } { Z _ { \mu , \sigma ^ { 2 } } } } \exp ( s B ( x _ { R } ) ) ,\tag{48}
$$

where $\begin{array} { r } { B ( x _ { R } ) = ( \mu / \sigma ^ { 2 } ) \sum _ { i \in R } x _ { i } } \end{array}$ is a soft majority vote of all pixels in region $R ,$ named $B$ as it is effectively a ‘magnetic field’ bias on region $\bar { R }$ given label s. s-independent terms are grouped into

the prefactor $\alpha ( x _ { R } )$ which will cancel out of relevant scores. Since the label distribution is uniform, this is all we need: the reverse conditional distributions are given by the Bayes’ formula corollary

$$
p ( s \mid x _ { R } ) = { \frac { p ( x _ { R } \mid s ) } { \sum _ { s ^ { \prime } = \pm 1 } p ( x _ { R } \mid s ^ { \prime } ) } } = { \frac { \exp ( s B ( x _ { R } ) ) } { 2 \cosh B ( x _ { R } ) } } .\tag{49}
$$

As a result, Bayes optimal inference of the global label is $\begin{array} { r } { \mathbb { E } [ S ~ \mid ~ X _ { R } ] \equiv \sum _ { s \in \pm 1 } s p ( s ~ \mid ~ x _ { R } ) = } \end{array}$ tanh $B ( x _ { R } )$ . Thus the task of inferring which Gaussian the observation $x _ { R }$ comes from is equivalent to studying the magnetisation of a non-interacting Ising magnet.

From the two conditional distributions, we can obtain the scores directly. For example,

$$
s _ { A } ( \boldsymbol x _ { R } \mid \boldsymbol s ) : = \nabla _ { \boldsymbol x _ { A } } \log p ( \boldsymbol x _ { R } \mid \boldsymbol s ) = - \frac 1 { \sigma ^ { 2 } } ( \boldsymbol x _ { A } - \boldsymbol s \boldsymbol \mu ) ,\tag{50}
$$

which implies

$$
s _ { A } ( \boldsymbol x _ { R } ) : = \nabla _ { \boldsymbol x _ { A } } \log p ( \boldsymbol x _ { R } ) = - \frac { \boldsymbol x _ { A } } { \sigma ^ { 2 } } + \frac { \mu } { \sigma ^ { 2 } } \operatorname { t a n h } B ( \boldsymbol x _ { R } ) ,\tag{51}
$$

via the trick $\nabla _ { x _ { \epsilon } }$ log $\begin{array} { r } { p ( x _ { R } ) = \sum _ { s } p ( s \mid x _ { R } ) \nabla _ { A } } \end{array}$ log $p ( x _ { R } \mid s )$ . In the score $s _ { A } .$ , the first term only includes $A .$ , and the rest of the image enters through the tanh. Then we have the locality gap

$$
\Delta _ { \mathrm { l o c } } ^ { U } = \frac { \mu } { \sigma ^ { 2 } } [ \operatorname { t a n h } B ( x ) - \operatorname { t a n h } B ( x _ { A B } ) ] .\tag{52}
$$

This gives an explicit characterisation of what the locality gap is comparing: it asks how important $\textstyle \sum _ { i \in C } x _ { i }$ is in inferring the label, softened through the tanh. Its peak in t is directly given by the shift induced by $C .$

The same results also let us calculate the global conditioning gap

$$
\Delta _ { \mathrm { c o n d } } ^ { G } = \frac { \mu } { \sigma ^ { 2 } } ( \operatorname { t a n h } ( B ( x ) ) - s ) .\tag{53}
$$

If we look at their difference,

$$
\Delta _ { \mathrm { l o c } } ^ { U } - \Delta _ { \mathrm { c o n d } } ^ { G } = \frac { \mu } { \sigma ^ { 2 } } ( s - \operatorname { t a n h } ( B ( x _ { A B } ) ) .\tag{54}
$$

## E.2 CMI for Gaussian mixtures

Just like the scores, the conditional mutual information for the two component Gaussian mixture is analytically reducible to a single integral.

Exact formula for CMI. To start, we recall the definition of the CMI

$$
I ( A ; C \mid B ) : = \mathbb { E } _ { x _ { B } } D _ { \mathrm { K L } } ( p ( x _ { A } , x _ { C } \mid x _ { B } ) \| p ( x _ { A } \mid x _ { B } ) p ( x _ { C } \mid x _ { B } ) ) ,\tag{55}
$$

$$
\equiv \int \mathrm { d } x p ( x ) \log { \frac { p ( x _ { A } , x _ { C } \mid x _ { B } ) } { p ( x _ { A } \mid x _ { B } ) p ( x _ { C } \mid x _ { B } ) } } ,\tag{56}
$$

and get rid of conditioning terms by restoring B marginals. The result is

$$
I ( A ; C \mid B ) \equiv \int \mathrm { d } x p ( x ) \log { \frac { p ( x _ { A } , x _ { C } , x _ { B } ) p ( x _ { B } ) } { p ( x _ { A } , x _ { B } ) p ( x _ { C } , x _ { B } ) } }\tag{57}
$$

$$
= H ( A B ) - H ( B ) + H ( B C ) - H ( A B C ) ,\tag{58}
$$

$$
= I ( A ; C | B , S ) + [ I ( S ; A B ) - I ( S ; B ) ] - [ I ( S ; A B C ) - I ( S ; B C ) ] .\tag{59}
$$

For the joint Gaussian mixture, $I ( A ; C | B , S ) = 0$ since conditional on the label, the distribution completely factorizes, and we only need to compute mutual informations/differential entropies associated to a region R where $R \in \{ A B , B C , B , { \bar { A } } B C \}$

The entropy $H ( R )$ is

$$
\begin{array} { l } { { \displaystyle H ( R ) = - \int \mathrm { d } x _ { R } p ( x _ { R } ) \log p ( x _ { R } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \int \mathrm { d } x _ { R } p ( x _ { R } ) ( x _ { R } - \mu _ { R } ) ^ { 2 } } \ ~ } \\ { { \displaystyle ~ - \int \mathrm { d } x _ { R } p ( x _ { R } ) \log ( 1 + \exp ( - \frac { 2 } { \sigma ^ { 2 } } x _ { R } \cdot \mu _ { R } ) ) } , } \end{array}\tag{60}
$$

obtained by factoring out one of the Gaussians inside the log. The first term gives two trivial gaussian integrals by expanding $p ( x _ { R } )$ as a sum of two Gaussian integrals (which we do not write out, they will cancel out between all the terms in the CMI equation 57). The second term is the nontrivial one because it includes a sum inside the log, as well as integrals over all the pixels in R. However, since the integrand only depends on the dot product $x _ { R } \cdot \mu _ { R }$ , we may write $x _ { R } = x _ { R } ^ { \parallel } + x _ { R } ^ { \perp }$ where $x _ { R } ^ { \perp } \cdot \mu _ { R } = 0$ and $\mathrm { d } x _ { R } = \mathrm { d } x _ { R } ^ { \parallel } \mathrm { d } x _ { R } ^ { \perp }$ , and $\mathrm { d } x _ { R } ^ { \perp }$ integrates out to give a constant that cancels with some of the normalisation in $p ( x _ { R } )$ . The result is a one-dimensional integral

$$
H ( R ) = W ( R ) - { \frac { 1 } { Z _ { R } ^ { \parallel } } } \int \mathrm { d } x _ { R } ^ { \parallel } \exp \left( - { \frac { ( x _ { R } ^ { \parallel } - \| \mu _ { R } \| ) ^ { 2 } } { 2 \sigma ^ { 2 } } } \right) \log \left( 1 + \exp ( - { \frac { 2 } { \sigma ^ { 2 } } } \| \mu _ { R } \| x _ { R } ^ { \parallel } ) \right)\tag{61}
$$

Lastly, we perform a change of variable by writing the parallel component as ‘mean + fluctuations’, i.e. $x _ { R } ^ { \parallel } = \mu _ { R } + \sigma z$ , which gives

$$
H ( R ) = W ( R ) - { \frac { 1 } { \sqrt { 2 \pi } } } \int \mathrm { d } z \exp \left( - z ^ { 2 } / 2 \right) \log \left( 1 + \exp ( - 2 K ^ { 2 } - 2 K z ) \right) ,\tag{62}
$$

in terms of one free parameter, the signal-to-noise ratio $K _ { R } : = \| \mu _ { R } \| / \sigma$ . Defining

$$
{ \cal F } ( K ) : = \frac { 1 } { \sqrt { 2 \pi } } \int \mathrm { d } z \exp \left( - z ^ { 2 } / 2 \right) \log \left( 1 + \exp ( - 2 K ^ { 2 } - 2 K z ) \right) ,\tag{63}
$$

we can write the CMI as

$$
I ( A ; C \mid B ) = F ( K _ { B } ) - F ( K _ { A B } ) - [ F ( K _ { B C } ) - F ( K _ { A B C } ) ] .\tag{64}
$$

Asymptotic expansion for the $F$ integral. The integral $F ( K )$ can be exactly evaluated in the large signal limit $K \gg 1$ . Performing another change of variables $y = 2 K ( K + z )$ yields

$$
F ( K ) = \frac { 1 } { 2 K } \exp ( - K ^ { 2 } / 2 ) \int \frac { \mathrm { d } y } { \sqrt { 2 \pi } } \exp \left( - y ^ { 2 } / 8 K ^ { 2 } \right) \exp ( y / 2 ) \log \left( 1 + \exp ( - y ) \right) ,\tag{65}
$$

and as $K \to \infty , \exp \left( - y ^ { 2 } / 8 K ^ { 2 } \right) \to 1$ , and

$$
F ( K ) \asymp \sqrt { \frac { \pi } { 2 K ^ { 2 } } } \exp ( - K ^ { 2 } / 2 ) ,\tag{66}
$$

where $\textstyle \int \mathrm { d } y \exp ( y / 2 )$ log $( 1 + \exp ( - y ) ) = 2 \pi$ by elementary methods.

Diverging Markov Length. Now send the image size to infinity, growing both $| B |$ and $| C |$ as $O ( N ^ { 2 } )$ while $| A | = O ( 1 )$ , a “thermodynamic limit” relevant for denoising a small patch. This yields, in the $K  \infty$ limit, a simple form for the CMI which reads

$$
I ( A ; C \mid B ) \asymp \sqrt { \frac { \pi } { 2 K _ { B } ^ { 2 } } } \exp \left[ - \frac { \mu ^ { 2 } } { 2 \sigma ^ { 2 } } | B | \right] \left( 1 - \exp \left[ - \frac { \mu ^ { 2 } } { 2 \sigma ^ { 2 } } | A | \right] \right) ,\tag{67}
$$

up to exponentially small corrections in |C|. Since this decays faster than exponential in $r \propto \sqrt { B }$ the Markov length $\xi = 0$ , and there is no locality transition whenever K is large.

If we recall the noise path $X _ { t } = ( 1 - t ) X _ { 0 } + t Z$ with Z standard Gaussian noise, we have a Gaussian mixture for every $t = 1$ with effective parameters $\mu _ { t } = \mu ( 1 - t )$ and $\sigma _ { t } ^ { 2 } = ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 }$ . As the noise is i.i.d. on every patch, the signal to noise for any R is

$$
K _ { R } ^ { 2 } ( t ) = \frac { ( 1 - t ) ^ { 2 } \mu ^ { 2 } } { ( 1 - t ) ^ { 2 } \sigma ^ { 2 } + t ^ { 2 } } | R | .\tag{68}
$$

Is the signal-to-noise large as $N \to \infty$ for every $0 \leq t \leq 1 2$ Taking $t = 1 - k / N$ yields

$$
{ \cal K } _ { R } ^ { 2 } ( t ) \asymp \mu ^ { 2 } k ^ { 2 } | R | / N ^ { 2 } = O ( 1 ) ,\tag{69}
$$

for $R = B$ and $R = B C$ , and the asymptotic expansion fails. In fact, in this limit, we are able to show Markov length $\xi  \infty$ . Noticing that $K _ { R _ { 1 } R _ { 2 } } ^ { 2 } = K _ { R _ { 1 } } ^ { 2 } + K _ { R _ { 2 } } ^ { 2 }$ for disjoint $R _ { 1 }$ and $R _ { 2 }$ , we can write

$$
F ( K _ { R } ( t ) ) - F ( K _ { A R } ( t ) ) = - K _ { A } ( t ) ^ { 2 } F ^ { \prime } ( K _ { R } ( t ) ) ,\tag{70}
$$

where we use $K _ { A }  0$ and the <sup>′</sup> denotes differentiation with respect to $K ^ { 2 }$ . The CMI takes the form

$$
I _ { t } ( A ; C \mid B ) \asymp K _ { A } ^ { 2 } ( t ) [ F ^ { \prime } ( K _ { B C } ( t ) ) - F ^ { \prime } ( K _ { B } ( t ) ) ]\tag{71}
$$

which we try to maximize over t. In particular, setting

$$
I _ { \mathrm { p e a k } } = \operatorname* { s u p } _ { * } I _ { t } ( A ; C \mid B ) , \quad t _ { * } = \arg \operatorname* { m a x } _ { * } I _ { t } ( A ; C \mid B )\tag{72}
$$

we are able to show no exponential decay exists. First, define an effective variance parameter

$$
\sigma _ { \mathrm { e f f } } ^ { 2 } ( t ) = \sigma ^ { 2 } + \frac { t ^ { 2 } } { ( 1 - t ) ^ { 2 } } ,\tag{73}
$$

such that we may interpret the noise path as purely expanding the variances. Then, plug in the definition of $K _ { R } ( t )$ , which gives

$$
\operatorname* { s u p } _ { t } K _ { A } ^ { 2 } ( t ) [ F ^ { \prime } ( K _ { B C } ( t ) ) - F ^ { \prime } ( K _ { B } ( t ) ) ] = \operatorname* { s u p } _ { t } \frac { \mu ^ { 2 } | A | } { \sigma _ { \mathrm { e f f } } ^ { 2 } ( t ) } \left[ F ^ { \prime } \left( \frac { \mu ^ { 2 } ( | B | + | C | ) } { \sigma _ { \mathrm { e f f } } ^ { 2 } ( t ) } \right) - F ^ { \prime } \left( \frac { \mu ^ { 2 } | B | } { \sigma _ { \mathrm { e f f } } ^ { 2 } ( t ) } \right) \right] .\tag{74}
$$

The limit keeps $| B | / | C | = O ( 1 )$ , so let’s isolate that parameter, setting $\lambda = 1 + | C | / | B |$ , we get

$$
\operatorname * { s u p } _ { t } K _ { A } ^ { 2 } ( t ) [ F ^ { \prime } ( K _ { B C } ( t ) ) - F ^ { \prime } ( K _ { B } ( t ) ) ] = \frac { | A | } { | B | } \operatorname * { s u p } _ { K _ { B } } K _ { B } ^ { 2 } \left[ F ^ { \prime } \left( \lambda K _ { B } ^ { 2 } \right) - F ^ { \prime } \left( K _ { B } ^ { 2 } \right) \right] ,\tag{75}
$$

where we also set $\mu ^ { 2 } | B | / \sigma _ { \mathrm { e f f } } ^ { 2 } ( t ) = K _ { B } ^ { 2 }$ . Also note that the supremum over t is equivalent to supremum over $K _ { B } ^ { 2 } .$ , which is the only t dependent parameter, which is the main point of this substitution. Crucially, the supremum over $K _ { B } ^ { 2 }$ yields just a number and does not depend on $| B |$ by itself anywhere. As a result, $\bar { I } _ { t _ { * } } ( A ; C \mid B ) \asymp \breve { 1 } / | B |$ , and the Markov length is divergent.

Lastly, we can obtain k explicitly. Noting that we need $K _ { B } ^ { 2 } ( t _ { * } ) \asymp O ( 1 )$ at the supremum, we need to obtain

$$
\frac { \mu ^ { 2 } | B | } { \sigma _ { \mathrm { e f f } } ^ { 2 } ( t _ { * } ) } = O ( 1 ) \implies 1 - t _ { * } = \frac { 1 } { \mu } \sqrt { \frac { K _ { B } ^ { 2 } ( t _ { * } ) } { | B | } } .\tag{76}
$$

Since $| B | = O ( N ^ { 2 } )$ , the peak time t scales as $1 - 1 / N$ , as expected.

The time windows. Since we can control exactly the CMI in the Gaussian mixture, let us try to give asymptotic formulas for each, demonstrating that the windows need not be exactly identical.

First, recall the nonlocality times $t _ { \mathrm { n o n l o c \_ } } ^ { \mathrm { s t a r t / e n d } , \delta }$ are defined by finding times such that the CMI $I _ { t } ( A ; C \ )$ $B ) \approx \delta$ by Defn. 1. We assume we pick δ as a fixed fraction of the peak CMI value. Since we already evaluated the CMI in the transition window, we immediately know scale identically as the peak, i.e. $t _ { \mathrm { n o n l o c } } ^ { \mathrm { s t a r t / e n d } , \delta } \sim 1 - O ( 1 / N )$

Second, recall that $t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t / e n d } , \delta }$ is defined mutual informations. For the start, we want to find the first t such that $I _ { 0 } ( S ; A B ) ^ { \prime } - I _ { t } ( S ; B ) = \delta .$ . Assuming δ is smaller than the peak value, this is equivalent to calculating $F ( \dot { K _ { B } } ( t ) ) - F ( \dot { K _ { A B } } ( 0 ) ) = \delta$ . Since B occupies a large fraction of the image, the second term is exponentially suppressed in |B| and we may drop it. For large signal to noise, we then just need to solve

$$
\begin{array} { r l r } {  { \frac { \exp ( - K _ { B } ^ { 2 } ( t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \delta } ) / 2 ) } { K _ { B } } = \delta \implies K _ { B } ^ { 2 } ( t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \delta } ) = 2 \log \frac { 1 } { \delta } + O ( \log \frac { 1 } { K _ { B } } ) } } \\ & { } & { \implies 1 - t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \delta } = \frac { 1 } { \mu } \sqrt { \frac { K _ { B } ^ { 2 } ( t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \delta } ) } { | B | } } \sim \sqrt { \frac { \log 1 / \delta } { | B | } } . } \end{array}\tag{77}
$$

Focusing near $t = 1$ , near the critical time $\delta \sim 1 / N ^ { 2 }$ , we thus get $1 - t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \delta } \sim \sqrt { \log ( N ) } / N$

Lastly, the speciation end time is defined as the largest time such that $I _ { t } ( S ; A B C ) = \delta )$ . Near $t \to 1$ when the global label is degraded, the signal to noise is very small, and instead we evaluate the the integral in equation 63 in the small K limit. This yields $I ( S ; \dot { A } B C ) = \log 2 - F ( K _ { A B C } ) = K _ { A B C } ^ { 2 } / 2$ which we set equal to δ. Using the formula for the exponent being $O ( 1 )$ , we have

$$
1 - t _ { \mathrm { s p e c } } ^ { \mathrm { e n d } , \delta } = \frac { \sqrt { \delta } } { \sqrt { | A B C | } } = O ( 1 / N ^ { 2 } ) ,\tag{78}
$$

where we again took $\delta = O ( 1 / N ^ { 2 } )$ . This proves that the locality window has a size proportional to $1 / N$ , while the semantic window has the scalings $1 - O ( { \sqrt { \log { N } } } / { N } )$ and $1 - O ( 1 / \hat { N } ^ { 2 } )$ .

![](images/0f5c4bdb9b088445220006b35c86b7b2d89d5fcf73dad92472c162feaf0d97ee.jpg)  
(a)

![](images/9b46aaaa7cfb8c30c440de6a097e920f83103e5f1287db1417fb0f5a22848495.jpg)  
(b)  
Figure 4: Exact numerical evaluation of the Gaussian-mixture information measures with specific values given in E.3. (a) Mutual information curves for the four spatial regions. The cliff diagnoses the symmetry-breaking phase transition. Inset: zoomed-in region near critical window. (b) Local and global conditional information gaps $I _ { t } ( S ; A | B ) , I _ { t } ( S ; A | \bar { B } C )$ , and their difference which gives conditional interaction information $\mathsf { \bar { I } } _ { t } \mathsf { \bar { ( } A } ; \dot { C } ; S | \dot { B } )$ . The peak diagnoses the non-locality transition.

## E.3 Numerical evaluation of the exact GMM formulas

We numerically evaluate the exact information-theoretic formulas derived above for the ultra-local two-component Gaussian mixture model. We take the homogeneous mean vector $\mu = ( 1 , \ldots , 1 )$ , unit bare variance $\sigma ^ { 2 } = 1$ , and $\mathrm { ~ a ~ 2 0 ~ } \times \mathrm { ~ 2 0 ~ }$ flattened system. The spatial partition is chosen as

$$
a = \| \mu _ { A } \| ^ { 2 } = 1 6 , \qquad b = \| \mu _ { B } \| ^ { 2 } = 8 0 , \qquad c = \| \mu _ { C } \| ^ { 2 } = 3 0 4 ,
$$

corresponding to a local patch A, a buffer B occupying 20% of the system, and the remaining context C. The forward noising process is

$$
X _ { t } = ( 1 - t ) X _ { 0 } + t Z , \qquad Z \sim { \mathcal { N } } ( 0 , I ) .
$$

We numerically evaluate the core integral $F ( W )$ by adaptive numerical quadrature and the resulting curves are interpolated on a uniform grid $t \in [ 0 , 0 . 9 9 ]$ with spacing $\Delta t \stackrel { - } { = } 0 . 0 0 5$

Fig. 4(a) tracks the four mutual informations $I _ { t } ( S ; B ) , I _ { t } ( S ; A B ) , I _ { t } ( S ; B C ) , I _ { t } ( S ; A B C )$ . These curves are monotone decreasing under the forward diffusion dynamics and develop sharp latetime cliffs. The smaller regions lose semantic information earlier, while the global regions remain informative until later times. This is the finite-size numerical signature of the forward-backward semantic transition.

Fig. 4(b) plots the two conditional gaps $I _ { t } ( S ; A | B ) = I _ { t } ( S ; A B ) - I _ { t } ( S ; B )$ , and $I _ { t } ( S ; A | B C ) =$ $I _ { t } \big ( \boldsymbol { S } ; A \dot { \boldsymbol { B C } } ) - I _ { t } ( \boldsymbol { S } ; B \boldsymbol { C } )$ . These gaps peak near the steepest parts of the corresponding MI cliffs. For the parameters used here, $G _ { \mathrm { l o c } }$ peaks near $t \simeq 0 . 8 8$ , whereas $G _ { \mathrm { g l o b } }$ is smaller and delayed, peaking near $t \simeq 0 . 9 4$ . This delay reflects the greater robustness of the global context ABC to the forward noise.

The conditional interaction information $I _ { t } ( A ; S ; C | B ) = I _ { t } ( S ; A | B ) - I _ { t } ( S ; A | B C )$ is also illustrated in Fig. 4(b). For the identity-covariance ultra-local GMM, conditioning on the semantic label removes all residual spatial dependence, so $I _ { t } ( A ; S ; C | B ) = I _ { t } ( A ; C | B )$ . Thus the same curve is also the unconditional CMI diagnosing non-locality. The CII/CMI curve has a single sharp peak, numerically located near $t \simeq 0 . 8 8$ for the present parameters, coinciding with the local semantic information cliff. This provides an exact-calculation check of the proposed concurrence between the symmetry-breaking and non-locality transitions.

## F Convergence from Separated Class Means

We state and prove the formal version of the convergence result discussed in the main text.

Theorem 4 (Convergence from separated class means). (Formal version ofTheorem 2)Let S have a fixed finite set of labels with positive prior probabilities. Suppose that the common-cause hypothesis holds with a size-independent $\alpha > 0$ . For each N, let B be annulus with radius $N ,$ , and use $X _ { t } = ( 1 - t ) X _ { 0 } + t Z$ , where $Z \sim { \mathcal { N } } ( 0 , I )$ is independent of $( X _ { 0 } , S )$ . Write $m _ { s } = \mathbb { E } [ X _ { B , 0 } \mid S = s ]$ Suppose there are constants $D _ { N } , C _ { N } > 0 ;$ , potentially scaling with N, such that,for all $\dot { s } \neq s ^ { \prime }$ and all $\bar { \boldsymbol u } \in \mathbb { R } ^ { | B }$

$$
\begin{array} { r } { \| m _ { s } - m _ { s ^ { \prime } } \| _ { 2 } ^ { 2 } \geq D _ { N } , } \end{array}\tag{79}
$$

$$
\begin{array} { r } { \mathbb { E } \Big [ e ^ { u ^ { \top } ( X _ { B , 0 } - m _ { s } ) } \mid S = s \Big ] \leq e ^ { C _ { N } \| u \| _ { 2 } ^ { 2 } / 2 } . } \end{array}\tag{80}
$$

Suppose the separation between semantic means $D _ { N }$ scalesfaster than the tail bound within class $C _ { N }$ so that they admit a choice of $\cdot _ { \delta _ { N } }$ below the CMI peak at each size, $0 < \alpha \delta _ { N } < H ( S ) / 2$ and

$$
\log \frac { H ( S ) } { \alpha \delta _ { N } } = o ( D _ { N } / ( C _ { N } + 1 ) ) ,\tag{81}
$$

then both speciation endpoints and both nonlocality endpoints converge to $t = 1$ . In particular,

$$
1 - t _ { \mathrm { s p e c } } ^ { \mathrm { s t a r t } , \alpha \delta _ { N } } = O \left( \sqrt { \frac { \log [ H ( S ) / ( \alpha \delta _ { N } ) ] } { D _ { N } / ( C _ { N } + 1 ) } } \right) .\tag{82}
$$

Proof. First, we bound how often a noisy observation of B is assigned the wrong label. In class $s ,$ the mean observation is $( 1 - t ) m _ { s }$ . Use the rule that picks the closest class mean. It can mistake s for $s ^ { \prime }$ only if the observation crosses half the gap between these means.

Let unit vector $e = ( m _ { s ^ { \prime } } - m _ { s } ) / \| m _ { s ^ { \prime } } - m _ { s } \| _ { 2 }$ <sub>2</sub> point toward that competing mean. Comparing the squared distances to $( 1 - t ) m _ { s }$ and $( 1 - t ) m _ { s ^ { \prime } }$ shows that the displacement $\bar { Y } = X _ { B , t } - \bar { ( } 1 - \bar { t } ) m _ { s }$ must satisfy

$$
e ^ { \top } Y \geq z , \qquad z = { \frac { 1 - t } { 2 } } \| m _ { s ^ { \prime } } - m _ { s } \| _ { 2 }\tag{83}
$$

for an error to occur.

Conditional on $S = s$ , we have $Y = ( 1 - t ) ( X _ { B , 0 } - m _ { s } ) + t Z _ { B }$ . To bound the chance of crossing the midpoint, we use a Chernoff bound on the conditional r.v. $e ^ { \top } Y \mid S$ . In particular, we have

$$
\begin{array} { r l } & { \operatorname* { P r } ( \mathrm { p a i r w i s e ~ c r o s s i n g ~ } | \mathrm { ~ } S = s ) = \operatorname* { P r } ( e ^ { \top } Y \geq z \mid S = s ) , } \\ & { \qquad = \operatorname* { P r } ( e ^ { \lambda e ^ { \top } Y } \geq e ^ { \lambda z } \mid S = s ) , } \\ & { \qquad \leq e ^ { - \lambda z } \mathbb { E } [ e ^ { \lambda e ^ { \top } Y } \mid S = s ] , } \end{array}\tag{84}
$$

where we take any $\lambda > 0$ , and the last step uses Markov’s inequality. Plugging $u = \lambda ( 1 - t ) \epsilon$ into 80 yields

$$
\mathbb { E } \Big [ e ^ { \lambda e ^ { \top } ( 1 - t ) ( X _ { B , 0 } - m _ { s } ) } \mid S = s \Big ] \leq e ^ { C _ { N } \lambda ^ { 2 } ( 1 - t ) ^ { 2 } / 2 } .\tag{85}
$$

Together with E $\left\lceil e ^ { \lambda e ^ { \top } t Z _ { B } } \mid S = s \right\rceil = e ^ { \lambda ^ { 2 } t ^ { 2 } / 2 }$ by Gaussianity, we obtain

Pr(pairwise crossing

$$
S = s ) \leq \exp \left[ - \lambda z + { \frac { \lambda ^ { 2 } } { 2 } } { \big ( } C _ { N } ( 1 - t ) ^ { 2 } + t ^ { 2 } { \big ) } \right] .\tag{86}
$$

For $t < 1$ , right hand side is minimized at $\lambda = z / [ C _ { N } ( 1 - t ) ^ { 2 } + t ^ { 2 } ]$ , yielding $\exp [ - z ^ { 2 } / ( 2 ( C ( 1 -$ $t ) ^ { 2 } + t ^ { 2 } ) )$ ]. By $7 9 , z \geq ( 1 - t ) \sqrt { D _ { N } } / 2 ;$ also $C _ { N } ( 1 - t ) ^ { 2 } + t ^ { 2 } \leq C _ { N } + 1$ . A union bound over the $L - 1$ competing labels gives the following estimate for t $< 1$

$$
\begin{array} { r } { \operatorname* { P r } \mathrm { ( n e a r e s t - m e a n ~ e r r o r ) } \le ( L - 1 ) e ^ { - D _ { N } / ( C _ { N } + 1 ) ( 1 - t ) ^ { 2 } / 8 } . } \end{array}\tag{87}
$$

Next, we connect classification error to the forward–backward success rate. Given $X _ { B , t } ,$ , write $q _ { s } = p _ { t } ( s \mid X _ { B , t } )$ . The best classifier chooses the label with largest $q _ { s } ,$ , so its error probability is $\mathbb { E } [ 1 - \operatorname* { m a x } _ { s } { q _ { s } } ]$ . Since $\begin{array} { r } { 1 - \sum _ { s } q _ { s } ^ { 2 } \le 2 ( 1 - \operatorname* { m a x } _ { s } q _ { s } ) } \end{array}$ , equation 5 and 87 give

$$
1 - P _ { \mathrm { F B } } ( B , t ) \leq 2 \mathbb { E } [ 1 - \operatorname* { m a x } _ { s } q _ { s } ] \leq 2 ( L - 1 ) e ^ { - D _ { N } / ( C _ { N } + 1 ) ( 1 - t ) ^ { 2 } / 8 } .\tag{88}
$$

The two-sided sandwich bounds in 10 now give $H ( S ~ \mid ~ X _ { B , t } ) ~ \le ~ h _ { 2 } ( P _ { \mathrm { F B } } ( B , t ) ) ~ + ~ ( 1 ~ -$ $P _ { \mathrm { F B } } ( B , t ) ) \log ( L - 1 )$ ). For fixed L, the right-hand side is at most a constant times $\sqrt { 1 - P _ { \mathrm { F B } } ( B , t ) }$ Combining this with the error bound above yields

$$
H ( S \mid X _ { B , t } ) \leq K e ^ { - k D _ { N } / ( C _ { N } + 1 ) ( 1 - t ) ^ { 2 } } ,\tag{89}
$$

where $K , k > 0$ do not depend on N or t.

Now set $t _ { N } = 1 - b \sqrt { \log [ H ( S ) / ( \alpha \delta _ { N } ) ] / ( D _ { N } / ( C _ { N } + 1 ) ) }$ , with b a fixed large constant. The threshold condition makes $t _ { N }  1 , s _ { 0 } t _ { N } \in [ 0 , 1 ]$ for large N. For every $t \leq t _ { N }$ , equation 89 is at most $K [ \alpha \delta _ { N } / H ( S ) ] ^ { k b ^ { 2 } }$ . Choose b so that $k b ^ { 2 } > 1$ and $K 2 ^ { - ( k b ^ { 2 } - 1 ) } < H ( S )$ . Since $\alpha \delta _ { N } / H ( S ) <$ $1 / 2$ , this bound is smaller than $\alpha \delta _ { N }$ . Finally,

$$
I _ { 0 } ( S ; A B ) - I _ { t } ( S ; B ) \le H ( S ) - I _ { t } ( S ; B ) = H ( S \mid X _ { B , t } ) .\tag{90}
$$

Thus the speciation start is no earlier than t<sub>N</sub>, which proves 82. Theorem 1 places the other three endpoints between that start and 1. All four endpoints therefore converge to 1. □