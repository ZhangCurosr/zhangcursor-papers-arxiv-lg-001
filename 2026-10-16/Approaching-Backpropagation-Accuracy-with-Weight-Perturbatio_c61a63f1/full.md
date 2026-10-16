# Approaching Backpropagation Accuracy with Weight Perturbation on MNIST

Patrick Inoue<sup>∗</sup>

## Abstract

Weight perturbation (WP) estimates learning directions from random parameter perturbations and a scalar change in loss, without backpropagating derivatives. We examine how batch size, training duration, and estimator stabilization afect its attainable accuracy in fully connected MNIST networks. In the shallow configuration, large-batch training reaches approximately 97% test accuracy, compared with an early backpropagation (BP) reference near 97.5–98%. The same comparison exposes a substantial eficiency gap: under a ten-epoch budget, shallow WP reaches 90.2%, whereas the reported BP reference reaches 97.4%. Epoch-indexed curves show diminishing returns from increasing batch size beyond 4,000. Learning-rate reduction and averaging independent perturbation-based updates also stabilize deeper networks, although their recorded results do not establish BP-level accuracy. We distinguish these empirical findings from a finite-horizon approximation argument: under stated regularity conditions, population WP can approximate a fixed gradient-descent trajectory arbitrarily closely at any fixed finite depth. Positive classification margins then sufice to preserve the reference predictions on a finite test set with arbitrarily high probability. Together, the results identify near-BP accuracy as a configuration- and budget-dependent possibility, rather than evidence of equal eficiency or an unconditional accuracy guarantee across depth.

Keywords: weight perturbation; node perturbation; backpropagation; MNIST; gradient estimation.

## 1 Introduction

Learning from global performance feedback without transmitting error derivatives through neural circuitry is a central challenge in biologically plausible credit assignment. Backpropagation (BP) solves this problem computationally, but its conventional implementation relies on coordinated forward and backward computations, precise layer-specific error signals, and corresponding feedback pathways whose biological realization remains debated [1, 27]. Biological synapses, in contrast, exhibit spontaneous fluctuations and remodeling, while synaptic plasticity can be regulated by neuromodulatory signals associated with reward and behavioral outcomes [28–31]. These properties motivate learning mechanisms that exploit local stochastic exploration and globally available evaluative signals to guide synaptic adaptation.

Weight perturbation (WP) ofers a particularly direct approach to this problem. Synaptic weights are transiently perturbed, and the resulting change in task performance determines the direction of subsequent plasticity [2, 3]. Each synapse requires only information about its own perturbation and a shared scalar feedback signal. The learning rule operates without an explicit diferentiable model of the network, backward error propagation, or symmetric feedback weights. WP therefore provides a model-free candidate for biologically motivated learning, connecting stochastic synaptic plasticity with gradient-based optimization [8, 12, 28]. Perturbation-based learning has also been combined with competitive Hebbian plasticity in sparse excitatory neural networks [25], illustrating its compatibility with additional biologically motivated learning mechanisms.

The principal obstacle is the statistical eficiency of high-dimensional credit assignment. WP infers a parameter update from scalar performance diferences, introducing substantial variance into the estimated learning direction. Consequently, convergence can be considerably slower than gradient-based training [11, 12]. On MNIST, Züge et al. reported a test accuracy of $9 2 . 3 \pm 0 . 2 \%$ for WP, compared with 97.7 ± 0.2% for stochastic gradient descent, in a network with one hidden layer after 50,000 weight updates [12]. This gap raises a fundamental question: does stochastic synaptic credit assignment impose a limit on attainable classification accuracy, or can improved estimator stability and extended training overcome it?

Here, we show that extended large-batch training brings WP to approximately 97% test accuracy on MNIST, approaching the early BP reference of 97.5–98% in a fully connected network with one hidden layer. The remaining accuracy gap is approximately 0.5–1 percentage point. Training eficiency remains substantially diferent: under a ten-epoch budget, WP achieves 90.2%, compared with 97.4% for the reported BP reference. We further show that learning-rate reduction and averaging independent perturbation estimates can stabilize WP training in networks with five hidden layers. Finally, a finitehorizon analysis establishes conditions under which population WP can approximate a gradient-descent trajectory arbitrarily closely. These findings demonstrate that high classification accuracy is attainable through model-free, perturbation-based synaptic learning, while identifying the training conditions and computational costs that govern its practical performance [4].

## 1.1 Perturbation learning: foundations and recent developments

Scalar-feedback learning has a substantial history in model-free learning, stochastic error descent, and statistical gradient-following algorithms [2, 3, 5]. Simultaneous perturbation stochastic approximation provides a related finite-diference route to optimization [6], while early neural-network work connected weight and node perturbations and their computational costs [7]. Conductance-perturbation models extended these ideas to spiking networks and birdsong learning [8, 9]. Population averaging was subsequently used to reduce variability in perturbation-based learning [10].

The relative eficiency of WP and node perturbation (NP) is not determined by parameter count alone. Analyses in linear networks identify dimension-dependent learning costs [11], whereas Züge et al. show that trial duration and efective task dimensionality can make WP comparable or superior to NP [12]. The 2023 journal article succeeds the 2021 preprint used in the thesis and is reproduced with its supplement in Züge’s 2025 dissertation [13]. For NP, Hiratani et al. identify difusion-driven instability under model mismatch and a stabilizing role for normalization [14]. Dalm et al. relate NP to directional derivatives and combine this perspective with input decorrelation, reporting substantially improved multilayer learning in their revised 2026 preprint [15].

For MNIST, Züge et al. report that their fully trained 784 → 100 → 10 network reaches 92.3 ± 0.2% with WP versus 97.7 ± 0.2% with SGD at B = 1,000 after 50,000 updates (five-run means and standard deviations; supplementary Table 2) [12]. The no-hidden-layer control has a smaller gap at a lower accuracy (Fig. S11); its teacher-target variants are diferent tasks. The dissertation retains these experiments, without adding a long-budget MNIST result that closes the hidden-layer gap [13]. Our study extends the batch-size and training-budget range; it does not establish superiority under a matched cross-study protocol.

Recent work includes the Dopamine optimizer for XOR and time-series tasks [16]. Gaussian zeroth-order optimization [17], population-based evolution strategies [18], and forward-evaluation language-model fine-tuning [19] provide broader context, but not matched from-scratch MNIST benchmarks. Forward gradients use automatic diferentiation [20] and are distinct from derivative-free loss-diference estimators.

## 2 Materials and Methods

## 2.1 Task and architectures

MNIST contains 60,000 training and 10,000 test images with ten digit classes [21]. Following the thesis [4], images are flattened to 784 components, scaled to [0.01, 0.99], and processed by fully connected hidden layers with hyperbolic-tangent activation and biases, followed by a ten-unit softmax output. The shallow architecture is 784 → 100 → 10. The deeper configuration, called Scen1 in the thesis, has five hidden layers of 100 units each. A wider five-hidden-layer configuration uses widths 2500, 2000, 1500, 1000, 500, following the architecture family of Cireşan et al. [22], but without their image-deformation protocol.

Weights use the stated Xavier-normal initialization with variance $2 / ( m _ { \mathrm { i n } } + m _ { \mathrm { o u t } } )$ [4, 23]. The loss on a batch B of size B is

$$
E _ { \mathcal { B } } ( \theta ) = - \frac { 1 } { B } \sum _ { n \in \mathcal { B } } \sum _ { k = 1 } ^ { 1 0 } \widetilde { y } _ { n k } \log p _ { \theta , k } ( x _ { n } ) ,\tag{1}
$$

where $\widetilde { y }$ denotes the label-smoothed target distribution. The batch-mean convention follows the thesis’s methods prose and appendix; its displayed loss equation omits $1 / B .$ . Reported learning rates are therefore retained without rescaling.

## 2.2 Weight perturbation and population averaging

Let $\theta \in \mathbb { R } ^ { d }$ collect the perturbed parameters. At each update, draw $\xi \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } )$ and retain this perturbation for every example in the batch. Both loss evaluations use the same inputs and targets. The baseline-subtracted estimator and update are

$$
\widehat { g } _ { \sigma , 1 } ( \theta ; \mathcal { B } ) = \frac { E _ { \mathcal { B } } ( \theta + \xi ) - E _ { \mathcal { B } } ( \theta ) } { \sigma ^ { 2 } } \xi , \qquad \theta _ { t + 1 } = \theta _ { t } - \eta _ { t } \widehat { g } _ { \sigma , 1 } .\tag{2}
$$

Perturbed parameters are temporary; the update is applied to the unperturbed vector. A lower perturbed loss reinforces the perturbation direction. This is the simultaneous Gaussian WP rule used in the thesis and its reference study [4, 12].

Population WP averages complete loss–perturbation products from P independent perturbations at the same parameter vector [4, 10]:

$$
\widehat { g } _ { \sigma , P } ( \boldsymbol { \theta } ; \mathcal { B } ) = \frac { 1 } { P } \sum _ { p = 1 } ^ { P } \frac { E _ { \mathcal { B } } ( \boldsymbol { \theta } + \boldsymbol { \xi } ^ { ( p ) } ) - E _ { \mathcal { B } } ( \boldsymbol { \theta } ) } { \sigma ^ { 2 } } \boldsymbol { \xi } ^ { ( p ) } .\tag{3}
$$

The baseline is shared, and population members are not retained as an inference ensemble. This is an average of products, not a product of separately averaged losses and perturbations. Independence gives the exact conditional identity

$$
\operatorname { C o v } _ { \xi } ( { \widehat { g } } _ { \sigma , P } \mid \theta , B ) = { \frac { 1 } { P } } \operatorname { C o v } _ { \xi } ( { \widehat { g } } _ { \sigma , 1 } \mid \theta , B ) .\tag{4}
$$

Increasing B and increasing P therefore act on diferent sources of variability: batch composition and perturbation realization. Population averaging reduces estimator variance; it does not remove disturbances before their propagation through the network.

## 2.3 Node perturbation and comparison protocol

NP perturbs neuronal inputs or activities rather than every parameter. In the temporally extended formulation motivating the experiments, node perturbations vary across examples, and their correlations with presynaptic activity are accumulated before multiplication by a scalar loss diference [12]. WP instead holds one parameter perturbation fixed over the batch. NP supplies a reference mechanism for understanding this distinction; no additional NP experiments are introduced here.

The shallow reference uses up to approximately 50,000 updates. The update-indexed extended comparisons display up to approximately 450,000 updates. The large-batch WP–BP comparison uses batch sizes from 2,000 to 10,000. The separate budget comparison also includes $B = 1 { , } 0 0 0$ and displays up to 22,500 epochs in its epoch-indexed panel. A separate ten-epoch comparison uses locally selected settings from Random Grid Path Search [4, 24]; these settings are not imputed to the long runs.

For U updates with a fixed batch size, nominal data exposure is

$$
N _ { \mathrm { e p o c h s } } = { \frac { U B } { 6 0 { , } 0 0 0 } } .\tag{5}
$$

This counts batch presentations, not all forward evaluations. Population WP requires $P + 1$ forward evaluations per update, versus two for $P = 1$ . BP additionally computes derivatives. Thus equal update counts, equal epochs, and equal runtime are distinct comparison criteria.

![](images/e986b7d6a257fb706f436c000e23f8f5ad1b41023cf944620ad09ad1ab00154b.jpg)

![](images/855550703716ff5f401de1733a116549ba02c729190bdb740858a1a5d8e5fc06.jpg)  
Figure 1: Shallow reference from the author’s thesis [4], reproducing the setting of Züge et al. [12]. Left: batch size 1; right: 1,000. Blue denotes WP and orange BP. Both horizontal axes count weight updates in thousands. The original curves and axes are unchanged.

## 3 Results

## 3.1 From the small-batch gap to near-BP accuracy

The starting gap depends strongly on configuration (Fig. 1). At approximately 50,000 updates, shallow WP reaches about 63% accuracy with B = 1, compared with about 95% for BP. With B = 1,000, WP reaches approximately 92–93%, against BP near 98%. Equal updates do not equalize data exposure: these settings correspond to roughly 0.83 and 833 nominal epochs, respectively.

The extended comparison is the principal accuracy result (Fig. 2). For batches of 4,000 and 10,000, shallow WP rises to approximately 96.8% and 97.0%, respectively, near the end of the displayed 450,000- update budget. The early BP reference is approximately 97.5–98%, leaving a visually estimated gap of about 0.5–1 percentage point. These curves directly support near-BP performance for this shallow configuration.

BP reaches its high accuracy early and subsequently declines towards roughly 94% during prolonged training. WP eventually exceeds these late BP checkpoints, but this does not establish superiority over a suitably selected BP model. The thesis itself notes that such prolonged BP training was included for comparison with WP and was not a recommended BP training protocol [4, Sec. 3.3]. The empirical conclusion is proximity to the earlier BP accuracy at a much larger training cost.

## 3.2 Batch size, data exposure, and learning speed

Figure 3 separates the two budget views reported in the thesis. Against update count, batches of 4,000–10,000 approach a similar accuracy near 97%, while batches of 1,000 and 2,000 remain lower. Against epochs, the largest batch no longer gives the strongest curve: $B = 4 { , } 0 0 0$ reaches a higher accuracy within the displayed exposure range than $B = 1 0 { , } 0 0 0$ . Beyond a useful stabilization regime, processing more examples per update yields diminishing returns.

The distinction is substantial: 450,000 updates at $B = 4 { , } 0 0 0$ represent 30,000 nominal epochs, and at $B = 1 0 { , } 0 0 0$ they represent 75,000. The ten-epoch results give a complementary short-budget comparison (Table 1). Shallow WP reaches 90.2%, 7.2 percentage points below the reported BP reference. Five-hiddenlayer WP reaches 73.1%. Near-BP long-run accuracy consequently does not imply near-BP learning speed.

![](images/35c54c1beb8041b197c931994452a23e4a07602afdc629591b1ca34faf51d150.jpg)

![](images/800f7c11c6fa782c7b77c04c74afe99727337613a0b88dd8c173f9179f4668aa.jpg)  
Figure 2: Extended training at batch sizes 4,000 (left) and 10,000 (right), selected from the original four-panel thesis figure [4]. Blue is the shallow WP model (original label “Baseline”), orange the BP reference, and green five-hidden-layer WP (“Scen1”). The green dashed segment at $B = 1 0 { , } 0 0 0$ extends the last shown level; it is not a measured continuation. Axes and solid curves are preserved. The truncated vertical range starts at 0.65. The relevant accuracy comparison is with BP’s early peak, not merely its deteriorated final checkpoint.

Table 1: Ten-epoch results transcribed from the thesis [4, Sec. 3.3]. These settings are specific to the short-budget search. The WP perturbation-strength setting is reported as $1 . 6 \times 1 0 ^ { - 3 }$
<table><tr><td>Configuration</td><td>Batch size</td><td>Learning rate Accuracy (%)</td><td></td></tr><tr><td>WP, shallow</td><td>2,000</td><td> $2 . 1 5 \times 1 0 ^ { - 3 }$ </td><td>90.2</td></tr><tr><td>WP, five hidden layers</td><td>2,000</td><td> $4 . 0 0 \times 1 0 ^ { - 3 }$ </td><td>73.1</td></tr><tr><td>BP, reported reference</td><td>4,000</td><td> $1 . 5 0 \times 1 0 ^ { - 2 }$ </td><td>97.4</td></tr></table>

$$
\begin{array} { c } { { \begin{array} { r l } { \longleftarrow B = 1 , 0 0 0 } & { { } \longleftarrow B = 2 , 0 0 0 } \\ { \longleftarrow B = 6 , 0 0 0 } & { { } \longleftarrow B = 1 0 , 0 0 0 } \end{array} } } \end{array} { } \begin{array} { r l } { { B = 4 , 0 0 0 } } & { { } } \\ { { B = 1 0 , 0 0 0 } } & { { } } \end{array}
$$

(a) Nominal training epochs  
![](images/f44276e91da10de9ce48f73cb086adbcd3eef19828bd826071c6d838328e58a2.jpg)

(b) Weight updates  
![](images/ce2cbab271c0c1819dee4c3d501c29d2471690942692dcd6bb6cfcd4c13ed9a0.jpg)  
Figure 3: Budget dependence of shallow WP, preserving the two original panels from the thesis [4]. Colors encode batch size as in the reinstated legend: 1,000 (blue), 2,000 (orange), 4,000 (green), 6,000 (red), and 10,000 (purple). Top: epochs; bottom: weight updates. Both horizontal axes carry a $1 0 ^ { 3 }$ multiplier, and both vertical ranges start at 0.90.

## 3.3 Stabilization of deeper networks

The five-hidden-layer comparison exposes a stronger stability constraint. At B = 4,000, Scen1 initially improves but subsequently collapses below the plotted range. At $B = 1 0 { , } 0 0 0$ , its solid curve reaches approximately 93% before the dashed extension begins. Larger batches can improve trainability without establishing deep-network parity.

The thesis further reports that decreasing the learning rate stabilizes narrow deep networks. Population averaging provides a complementary intervention: ten perturbations in Scen1 yield accuracy comparable to a reduced-learning-rate reference, while four perturbations stabilize the wide five-hidden-layer configuration where preceding single-perturbation settings failed [4, Sec. 3.4.3]. Indeed, the thesis’s final comparison states that the deeper reference had not achieved BP-comparable accuracy within the considered budgets [4, Sec. 3.6].

## 4 Discussion

WP can attain near-BP accuracy on MNIST in a fully connected network trained through scalar loss feedback. Extended large-batch training reaches approximately 97% test accuracy, within about 0.5–1 percentage point of the early BP reference (Fig. 2). This result demonstrates that the lower accuracies observed in the small-batch runs are not an intrinsic ceiling of the WP learning rule. A substantial part of the accuracy deficit can be recovered through stabilization and additional training. The experiments therefore separate two properties that are easily conflated when evaluating biologically motivated learning: attainable accuracy and learning eficiency.

The high-accuracy regime extends the earlier MNIST comparison of Züge et al. [12]. Their network with one hidden layer reached 92.3% with WP and 97.7% with SGD after 50,000 updates at a batch size of 1,000. Our extended experiments show that WP can move substantially closer to the BP reference when larger batches and longer training are available. The contribution is this empirical demonstration of near-BP accuracy in the hidden-layer setting. It complements the established interpretation of WP as a stochastic gradient estimator [2, 3, 11] by showing how much performance can be recovered in a nonlinear classification network.

Computational eficiency remains the principal practical limitation. In the ten-epoch comparison, WP reaches 90.2%, whereas the reported BP reference reaches 97.4% (Table 1). Approaching the latter accuracy with WP requires substantially more training. This behavior is consistent with the variance introduced when a high-dimensional update is estimated from a scalar loss diference [11, 12]. Smaller learning rates limit destabilizing parameter changes, while averaging independent perturbation estimates reduces update variability [10]. These mechanisms support continued learning, but they incur costs through additional updates or additional forward evaluations.

The batch-size comparison further shows why more averaging is not automatically more eficient. Larger batches expose each perturbation to a greater fraction of the task, while population averaging combines independent perturbation realizations. In the update-indexed curves, batches from 4,000 to 10,000 approach similar accuracies. When performance is indexed by epochs, the largest batch loses its advantage (Fig. 3). The useful operating regime therefore balances update stability against the amount of computation devoted to each step. Maximizing batch size or population size alone can yield diminishing returns; the objective is sustained learning with an afordable amount of averaging.

Network depth makes this balance more demanding. In the tested deeper architectures, settings that support shallow learning can produce unstable updates, and recovering useful learning requires smaller learning rates or stronger averaging. These interventions improve trainability, although the deeper configurations remain below the BP comparison within the investigated budgets. The results identify stabilization and slow convergence as central challenges for scaling WP. A larger representation hierarchy increases the demands on the learning process, so architectural capacity and efective training progress must be considered together. Improving the information extracted from each perturbation is consequently an important direction for extending WP to deeper networks.

Hybrid learning provides a complementary direction. Stricker et al. [25] combine competitive Hebbian feature learning with a joint Hebbian–WP classification-layer update in sparse excitatory networks. Inoue et al. [26] extend this framework to deeper nonnegative architectures using a PCA-related competitive update and inter-layer normalization. In these systems, hidden representations develop through local activity-dependent learning, while perturbation-based supervision acts at the readout. The present study addresses WP across all weight matrices. Together, these approaches motivate examining how the allocation of learning between local representation mechanisms and scalar-feedback updates afects accuracy, stability, and computational cost.

The empirical scope of the present results is fully connected MNIST classification. Extending the comparison to more complex tasks will determine how the required training budget changes with problem dificulty and network size. Relating the large exposure budgets to biologically feasible learning times is also an important question. The immediate methodological priority is to reduce the cost of reaching the high-accuracy regime demonstrated here, through more informative perturbation estimates and a better balance between learning rate, batch size, and population averaging.

## 4.1 Conclusion

Model-free weight perturbation reaches approximately 97% accuracy on MNIST and approaches the performance of backpropagation. This demonstrates that a biologically motivated learning rule can support high classification accuracy despite substantially slower learning. Larger batches, longer training, and stable updates are decisive in recovering that performance. The accompanying finite-horizon analysis establishes conditions for arbitrarily close approximation of a fixed gradient-descent trajectory. For the configurations examined here, the main practical challenge is the eficiency with which WP reaches high accuracy, especially as network depth increases.

## Acknowledgements

The author thanks Andreas Knoblauch and German Nemirovski for supervising the master’s thesis underlying this study. The original experiments used the bwUniCluster 2.0 computing infrastructure, as documented in the thesis [4].

## Prior dissemination and availability

The empirical results and original learning-curve artwork derive from the author’s 2023 master’s thesis [4], authored as Patrick Stricker and referenced through its OPUS-HSAS record (document 1713). This article selects and reframes its MNIST findings and adds the explicit finite-horizon analysis and updated literature context; the experiments are not presented as newly performed. The figures are reproduced as selected panels with restored legends and clarified captions. The underlying run logs and training code are not distributed with this article.

## Generative AI Statement

The author declares that generative AI (OpenAI ChatGPT) was used in the preparation of this manuscript for initial drafting, language editing, proofreading, and assistance with the formulation of the finite-horizon analysis. The original master’s thesis (Stricker, 2023), including all underlying experiments and results, was conducted and written entirely without generative AI. The author assumes full responsibility for the scientific content of this manuscript.

## A Finite-horizon approximation and classification agreement

## A.1 Estimator bias and variance

For a fixed batch, write $F ( \theta ) = E _ { B } ( \theta )$ and $u \sim \mathcal { N } ( 0 , I _ { d } )$ . Under conditions permitting Gaussian integration by parts,

$$
\begin{array} { r } { \mathbb E \widehat { g } _ { \sigma , P } ( \theta ) = \nabla F _ { \sigma } ( \theta ) , \qquad F _ { \sigma } ( \theta ) = \mathbb E _ { u } F ( \theta + \sigma u ) . } \end{array}\tag{6}
$$

This Gaussian-smoothing identity underlies randomized gradient-free methods [17, 18]; the smallperturbation gradient interpretation of WP is already established [12]. The baseline term vanishes in expectation, while $\sigma ^ { - 1 } \mathbb { E } [ F ( \theta + \sigma u ) u ] = \mathbb { E } \nabla F ( \theta + \sigma u )$ . Thus finite-σ WP is unbiased for the smoothed gradient, not generally the original gradient.

Suficient smoothness gives bias $O ( \sigma ^ { 2 } )$ , and independence gives

$$
\mathbb { E } \| \widehat { g } _ { \sigma , P } - \nabla F \| ^ { 2 } = \| \nabla F _ { \sigma } - \nabla F \| ^ { 2 } + \frac { \mathrm { t r } { \mathrm { C o v } ( \widehat { g } _ { \sigma , 1 } ) } } { P } .\tag{7}
$$

Increasing P removes variance, not smoothing bias. Decreasing σ alone does not remove the directional variability of a single estimate. In the linearized limit, writing $g = \nabla F ( \theta )$ , Gaussian fourth moments give

$$
\operatorname { C o v } \big ( ( g ^ { \top } u ) u \big ) = \| g \| ^ { 2 } I _ { d } + g g ^ { \top } , \qquad \operatorname { \mathbb { E } } \left\| \frac { 1 } { P } \sum _ { p = 1 } ^ { P } ( g ^ { \top } u _ { p } ) u _ { p } - g \right\| ^ { 2 } = \frac { d + 1 } { P } \| g \| ^ { 2 } .\tag{8}
$$

Consistency is therefore compatible with substantial cost in high dimension.

## A.2 Finite-horizon trajectory approximation

The following statement formalizes the theoretical approximation claim; it is not an additional empirical result.

Proposition 1 (Tracking a fixed gradient-descent reference). Fix a finite-dimensional network, an initial vector, a finite sequence of batches, and $T < \infty$ learning rates $\eta _ { t } \geq 0$ . Let

$$
\theta _ { t + 1 } ^ { * } = \theta _ { t } ^ { * } - \eta _ { t } \nabla F _ { t } ( \theta _ { t } ^ { * } ) , \qquad t = 0 , \textrm { \tiny . . . } , T - 1 ,\tag{9}
$$

where $F _ { t }$ is the prescribed batch loss. On a compact neighborhood of the reference trajectory, assume Lipschitz gradients and the uniform bounds

$$
\begin{array} { r } { \| \mathbb { E } \widehat { g } _ { \sigma , P , t } ( \theta ) - \nabla F _ { t } ( \theta ) \| \le C _ { b } \sigma ^ { 2 } , \qquad \mathrm { t r } \operatorname { C o v } ( \widehat { g } _ { \sigma , 1 , t } ( \theta ) ) \le V } \end{array}\tag{10}
$$

for suficiently small σ, with finite $C _ { b } , V$ . Assume deterministic loss evaluation, independent Gaussian perturbations, and exact arithmetic. For any $\varepsilon , \delta > 0$ , there exist $\sigma > 0$ and finite $P$ such that $W P ,$ using the same initialization, batches, and learning rates, satisfies

$$
\operatorname* { P r } \biggr ( \operatorname* { m a x } _ { 0 \leq t \leq T } \| \theta _ { t } - \theta _ { t } ^ { * } \| < \varepsilon \biggr ) > 1 - \delta .
$$

If the reference classifier at T has a unique largest logit on every example of a finite test set and logits are continuous in the parameters, σ and P can also be chosen to yield identical test predictions with probability greater than $1 - \delta$

Proof. Let L be a common Lipschitz constant. Conditional on the past and remaining in the neighborhood, Eq. (7) bounds mean-square estimation error by $C _ { b } ^ { 2 } \sigma ^ { 4 } + V / P$ . Markov’s inequality bounds the probability of an error exceeding $a > 0$ at an update by $( C _ { b } ^ { 2 } \sigma ^ { 4 } + V \dot { / } P ) / a ^ { 2 }$ . Whenever all estimation errors are at most $^ { a , }$ the parameter discrepancy $e _ { t } = \| \theta _ { t } - \theta _ { t } ^ { * } \|$ obeys

$$
e _ { t + 1 } \leq ( 1 + \eta _ { t } L ) e _ { t } + \eta _ { t } a , \quad e _ { t } \leq a \sum _ { s = 0 } ^ { t - 1 } \eta _ { s } \prod _ { r = s + 1 } ^ { t - 1 } ( 1 + \eta _ { r } L ) .\tag{11}
$$

Choose a so this finite-horizon bound is below both ε and the neighborhood radius. Choose σ small and P large enough that $T ( C _ { b } ^ { 2 } \sigma ^ { 4 } + V / P ) / a ^ { 2 } < \delta$ . A conditional union bound up to the first possible exit, together with Eq. (11), establishes the trajectory statement.

For classification, let $c _ { n }$ be each reference prediction. Its minimum logit margin is

$$
m = \operatorname* { m i n } _ { n } \left[ z _ { c _ { n } } ( x _ { n } ; \theta _ { T } ^ { * } ) - \operatorname* { m a x } _ { k \neq c _ { n } } z _ { k } ( x _ { n } ; \theta _ { T } ^ { * } ) \right] > 0 .
$$

Finiteness and continuity give a parameter neighborhood in which every logit changes by less than $m / 3$ . All predictions are preserved there. Choose ε within this neighborhood and apply the trajectory result. □

The reference includes mini-batch gradient descent with gradients computed by BP, but not arbitrary optimizers unless their state is also tracked. Constants may grow with depth, dimension, and horizon. The statement asserts neither global optimality nor equivalence at an afordable budget. Finite-precision cancellation and noisy loss evaluation can prevent arbitrarily small practical perturbations.

## References

[1] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323:533–536, 1986. doi:10.1038/323533a0.

[2] A. Dembo and T. Kailath. Model-free distributed learning. IEEE Transactions on Neural Networks, 1(1):58–70, 1990. doi:10.1109/72.80205.

[3] G. Cauwenberghs. A fast stochastic error-descent algorithm for supervised learning and optimization. In Advances in Neural Information Processing Systems 5, pp. 244–251, 1992.

[4] P. Stricker. Biologically Plausible Weight Perturbation Learning for Solving the Vanishing Gradients Problem. Master’s thesis, M.Sc. Data Science, Albstadt-Sigmaringen University, March 2023. OPUS-HSAS, document 1713.

[5] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992. doi:10.1007/BF00992696.

[6] J. C. Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. IEEE Transactions on Automatic Control, 37(3):332–341, 1992. doi:10.1109/9.119632.

[7] B. Flower and M. Jabri. Summed weight neuron perturbation: An O(N) improvement over weight perturbation. In Advances in Neural Information Processing Systems 5, pp. 212–219, 1992.

[8] I. R. Fiete and H. S. Seung. Gradient learning in spiking neural networks by dynamic perturbation of conductances. Physical Review Letters, 97:048104, 2006. doi:10.1103/PhysRevLett.97.048104.

[9] I. R. Fiete, M. S. Fee, and H. S. Seung. Model of birdsong learning based on gradient estimation by dynamic perturbation of neural conductances. Journal of Neurophysiology, 98:2038–2057, 2007.

[10] P. Suszyński and P. Wawrzyński. Learning population of spiking neural networks with perturbation of conductances. In International Joint Conference on Neural Networks, 2013. doi:10.1109/IJCNN.2013.6706756.

[11] J. Werfel, X. Xie, and H. S. Seung. Learning curves for stochastic gradient descent in linear feedforward networks. Neural Computation, 17(12):2699–2718, 2005.

[12] P. Züge, C. Klos, and R.-M. Memmesheimer. Weight versus node perturbation learning in temporally extended tasks: Weight perturbation often performs similarly or better. Physical Review X, 13:021006, 2023. doi:10.1103/PhysRevX.13.021006. Earlier version: bioRxiv 2021.10.04.463055, 2021.

[13] P. Züge. Biologically plausible learning and dynamics in neural networks. Doctoral dissertation, University of Bonn, 2025. doi:10.48565/bonndoc-715.

[14] N. Hiratani, Y. Mehta, T. Lillicrap, and P. E. Latham. On the stability and scalability of node perturbation learning. In Advances in Neural Information Processing Systems 35, 2022. doi:10.52202/068431- 2314.

[15] S. Dalm, M. van Gerven, and N. Ahmad. Node perturbation can efectively train multi-layer neural networks. arXiv:2310.00965v8, revised 2 June 2026; first posted 2023.

[16] S. Nambusubramaniyan, S. Safavi, R. Guru, and A. Knoblauch. Dopamine-driven synaptic credit assignment in neural networks. arXiv:2510.22178, 2025.

[17] Y. Nesterov and V. Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17:527–566, 2017. doi:10.1007/s10208-015-9296-2.

[18] T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864, 2017.

[19] S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, and S. Arora. Fine-tuning language models with just forward passes. In Advances in Neural Information Processing Systems 36, 2023.

[20] A. G. Baydin, B. A. Pearlmutter, D. Syme, F. Wood, and P. Torr. Gradients without backpropagation. arXiv:2202.08587, 2022.

[21] Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. doi:10.1109/5.726791.

[22] D. C. Cireşan, U. Meier, L. M. Gambardella, and J. Schmidhuber. Deep big simple neural nets excel on handwritten digit recognition. arXiv:1003.0358, 2010.

[23] X. Glorot and Y. Bengio. Understanding the dificulty of training deep feedforward neural networks. In Proceedings of AISTATS, PMLR 9:249–256, 2010.

[24] A. Knoblauch. Adapting loss functions to learning progress improves accuracy of classification in neural networks. In Foundations of Intelligent Systems, ISMIS 2022, pp. 272–282, 2022. doi:10.1007/978-3- 031-16564-1\_26.

[25] P. Stricker, F. Röhrbein, and A. Knoblauch. Weight perturbation and competitive Hebbian plasticity for training sparse excitatory neural networks. In International Joint Conference on Neural Networks, pp. 1–8, 2024. doi:10.1109/IJCNN60899.2024.10650478.

[26] P. Inoue, F. Röhrbein, and A. Knoblauch. Guiding sparse neural networks with neurobiological principles to elicit biologically plausible representations. arXiv:2603.03234v1, 2026.

[27] T. P. Lillicrap, A. Santoro, L. Marris, C. J. Akerman, and G. Hinton. Backpropagation and the brain. Nature Reviews Neuroscience, 21:335–346, 2020. doi:10.1038/s41583-020-0277-3.

[28] H. S. Seung. Learning in spiking neural networks by reinforcement of stochastic synaptic transmission. Neuron, 40(6):1063–1073, 2003. doi:10.1016/S0896-6273(03)00761-X.

[29] N. E. Ziv and N. Brenner. Synaptic tenacity or lack thereof: Spontaneous remodeling of synapses. Trends in Neurosciences, 41(2):89–99, 2018. doi:10.1016/j.tins.2017.12.003.

[30] J. N. J. Reynolds and J. R. Wickens. Dopamine-dependent plasticity of corticostriatal synapses. Neural Networks, 15(4–6):507–521, 2002. doi:10.1016/S0893-6080(02)00045-X.

[31] L. Kuśmierz, T. Isomura, and T. Toyoizumi. Learning with three factors: Modulating Hebbian plasticity with errors. Current Opinion in Neurobiology, 46:170–177, 2017. doi:10.1016/j.conb.2017.08.020.