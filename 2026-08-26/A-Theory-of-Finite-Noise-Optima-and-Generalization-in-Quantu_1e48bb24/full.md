# A Theory of Finite-Noise Optima and Generalization in Quantum Machine Learning

Ziyu Zhang,<sup>1,</sup> <sup>∗</sup> Zikang Jia,<sup>2,</sup> <sup>∗</sup> Xiaosong Li,<sup>1</sup> and Yulong Dong<sup>3,</sup> <sup>†</sup>

<sup>1</sup>Department of Chemistry, University of Washington, Seattle, Washington 98195, USA   
<sup>2</sup>Department of Mathematics, University of Michigan, Ann Arbor, Michigan 48109, USA <sup>3</sup>Department of Electrical Engineering and Computer Science, University of Michigan, Ann Arbor, Michigan 48109, USA (Dated: August 26, 2026)

Quantum noise is expected to degrade quantum machine learning by driving circuits away from their noiseless implementations. Yet recent studies show moderate noise can reduce testing error, a behavior unexplained by weak-noise perturbative error accumulation or strong-noise trainability collapse. Here we develop a statistical learning theory connecting microscopic noise processes to macroscopic learning performance. At its heart is a noise-order purity parameter, derived from a surrogate model analysis, that predicts the noise-induced reduction in model complexity and the consequent reduction in the generalization gap. Noise simultaneously increases prediction bias. Their competition explains the intermediate-noise regime left open between these limits. It produces a finite-noise optimum whose location depends on the learning setup and can disappear in the largesample limit. Numerical experiments validate these predictions. Noise programming can move a model towards this optimum. These results make the non-monotonic efect of noise predictable and provide a route to harness it.

## I. INTRODUCTION

Quantum noise is conventionally treated as an error because it drives a physical implementation away from its prescribed computation [1–3]. This view is natural for quantum algorithms whose accuracy relies on a carefully synthesized structure, and it is reinforced in quantum machine learning by two well-studied limits. Near the noiseless limit, analysis and mitigation are organized around deviations from the ideal circuit [4–10]. At suficiently large noise and circuit depth, parameter influence is suppressed, training collapses and a deep circuit becomes efectively shallow [11–13]. Both limits are destructive, inviting the conclusion that noise must degrade a quantum learning model throughout the range between them. Yet neither limit determines what happens before the onset of the high-noise collapse.

A learned predictor is not required to reproduce a prescribed noiseless computation [14, 15]. It adapts jointly to its trainable components, finite data and the physical noise acting on the circuit, and its objective is performance on unseen data rather than circuit fidelity [16–19]. Noise can therefore remove useful signal, but it can also suppress excess freedom used to fit the training sample [20–22]. The use of stochastic perturbations in training as a source of regularization has been studied in classical machine learning [23–25]. Recent quantum studies suggest similar possibilities. Moderate noise can improve variational quantum algorithms and quantum neural networks [26–32]. Quantum models can also remain robust to noise [33, 34], while noise redistributes locally important parameter directions [35]. These observations challenge a monotonic picture of noise, but do not yet explain when the improvement occurs, what controls the optimal noise level or why the benefit eventually disappears.

The obstacle is not a lack of bounds, but a mismatch between their mathematical objects and the learning question [20, 36–39]. Norm-based perturbative bounds control worst-case deviation; the triangle inequalities that make them general also discard cancellations and interactions among stochastic components, leaving them valid but uninformative in the intermediate regime [4–6]. Strong-noise analyses instead characterize the loss of representation and trainability [7, 11–13, 40]. Quantum Fisher information describes local state-space geometry and data-dependent learnable directions, but does not directly determine target-dependent testing loss [16, 35, 41, 42]. What is missing is a task-aligned analysis that retains the structure of the noisy predictor and follows how its changing statistical complexity afects generalization.

In this work, we fill this gap by developing a statistical learning theory for noisy quantum learning models. Using a noise-order surrogate, we organize the noisy response by the number of noise occurrences and introduce the noise-order purity, which measures how broadly the response is mixed across noise orders. We show that this single quantity controls the efective model complexity and therefore the generalization gap. Noise simultaneously attenuates the learnable signal. The competition between these efects produces a finite-noise optimum whose location is determined by the complete learning setup. Numerical results verify the predicted links from noise-order purity to model complexity and from model complexity to the generalization gap. They further show that the improvement occurs while the model remains trainable, before the strong-noise collapse. This theory complements the established weak- and strong-noise limits by resolving the learning behavior between them. Making this regime predictable also allows noise to be harnessed: controlled perturbations can move a model towards its finite-noise optimum. The same idea may inform how stochasticity is introduced during training and how protection is allocated in partially fault-tolerant architectures.

## II. RESULTS

## II.1. Finite noise induces non-monotonic generalization

We first study the finite-noise performance of a noisy QML model on the diabetes regression dataset, which contains baseline clinical measurements from 442 patients and a quantitative measure of disease progression one year later [43, 44]. We use a four-qubit parametrized circuit to predict this outcome from two baseline clinical measurements. We fix the dataset split, circuit architecture, and training protocol throughout the experiment, while varying the physical noise rate $\beta .$ For each value of $\beta _ { ; }$ the model is trained and evaluated under the same noisy circuit dynamics. We consider depolarizing, bit-flip, phase-damping, and amplitude-damping noise. The first three channels are unital, whereas amplitude damping is non-unital. Further details are provided in Methods.

Fig. 1 reveals a counterintuitive separation between training and test behavior. Across all four noise channels, the training loss $\mathcal { L } _ { \mathrm { t r a i n } } ( \beta )$ generally increases as the physical noise rate becomes larger. The noisy model therefore fits the training data less accurately. However, the test loss $\mathcal { L } _ { \mathrm { t e s t } } ( \beta )$ does not follow the same trend. Instead, it first decreases below the nearly noiseless value and only increases again once the noise becomes suficiently strong. In this finite-noise window, the noisier model performs worse on the data used for training, but better on testing data.

Simple explanations such as optimization quality are not consistent with this behavior. The improvement in test performance is not caused by a lower training loss or by a more successful fit to the training samples. It is instead a change in generalization. We quantify this efect using the

![](images/7b670cfaf84df649be6bdde0a69a0d771c6ddcb4bf591e3f1b8005b338cbc533.jpg)

![](images/f7ce1cc9de2955e7800e1a424fd375cf342982f192d5bd43005fa0f79d44746b.jpg)  
Figure 1. Non-monotonic response of the four-qubit regression model to physical noise. (a–d) Final training and testing mean-squared errors for four quantum noise models. Curves show the mean over ten independently trained runs and shaded bands show one standard deviation. (e) Generalization gap normalized by its value at zero noise. The dashed line shows the surrogate prediction introduced below.

generalization gap [45, 46]

$$
\Delta _ { \mathrm { g e n } } ( \beta ) = \mathcal { L } _ { \mathrm { t e s t } } ( \beta ) - \mathcal { L } _ { \mathrm { t r a i n } } ( \beta ) .\tag{1}
$$

As shown in Fig. 1(e), the gap decreases rapidly in the same intermediate-noise regime where the test loss improves. Thus, finite quantum noise can reduce the discrepancy between training and test performance before the large-noise degradation sets in.

This finite-noise transition is not captured by the two limiting regimes most commonly discussed in QML. In the nearly noiseless regime, the noise-induced efect is absent by construction. In the very noisy regime, noise can strongly suppress useful gradients and degrade trainability. The behavior observed in Fig. 1 lies between these limits: noise is already strong enough to change generalization, but not yet strong enough to destroy the learnable signal. This raises a central question: what changes in the QML model when noise is strong enough to improve generalization, but not yet strong enough to destroy the learnable signal? We address this question by introducing a stochastic surrogate model for noisy QML predictions.

## II.2. A noise-order surrogate explains noise-induced regularization

To understand the performance improvement observed in Fig. 1, we introduce a surrogate model for noisy QML analysis. Consider a circuit with g noisy locations and physical noise rate $\beta .$ . Each noise realization yields a noise-injected circuit configuration whose response depends on where noise occurs, the noise channel, and the circuit structure. The useful structure is that these many responses can be organized by noise order. Rather than tracking each microscopic configuration separately, we group the noisy circuit responses by the total number of noise occurrences, and represent the combined response at order m by an efective noise-order predictor $\psi _ { m } ( \rho _ { 0 } ; \pmb { \theta } )$ . The overall noisy predictor at physical noise rate $\beta$ is then written as

$$
\phi _ { \beta } ( \rho _ { 0 } ; \pmb { \theta } ) = \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) \psi _ { m } ( \rho _ { 0 } ; \pmb { \theta } ) , \qquad p _ { m } ( \beta , g ) = { \binom { g } { m } } \beta ^ { m } ( 1 - \beta ) ^ { g - m } .\tag{2}
$$

This representation has the form of an ensemble average: the predictors $\psi _ { m }$ contain the circuit- and noise-dependent response at each noise order, while the binomial weights $p _ { m } ( \beta , g )$ describe how the physical noise rate distributes probability mass across those orders. When $\beta$ is close to zero, most weight is concentrated on the nearly noiseless branch. As $\beta$ increases, the weight spreads over a broader range of finite-noise branches, so the learned predictor is not merely controlled by a single circuit structure.

This surrogate also shows how noise changes the training objective. In the surrogate model, the microscopic noise path is coarse-grained to its occurrence count: an order m is sampled with probability $p _ { m } ( \beta , g )$ , and the model evaluates the corresponding noise-order predictor $\psi _ { m }$ . Let $\mathbb { E } _ { S }$ denote the empirical average over the training set. As derived in Section A.2, the surrogate loss can be decomposed as

$$
\widetilde { \mathcal { L } } ( \pmb { \theta } , \beta ) = \mathbb { E } _ { \mathcal { S } } \left. \phi _ { \beta } ( \rho _ { 0 } ; \pmb { \theta } ) - y \right. ^ { 2 } + \mathbb { E } _ { \mathcal { S } } \sum _ { m _ { 1 } < m _ { 2 } } p _ { m _ { 1 } } ( \beta , g ) p _ { m _ { 2 } } ( \beta , g ) \left. \psi _ { m _ { 1 } } ( \rho _ { 0 } ; \pmb { \theta } ) - \psi _ { m _ { 2 } } ( \rho _ { 0 } ; \pmb { \theta } ) \right. ^ { 2 } .\tag{3}
$$

These two terms expose a tradeof between fitting with the averaged noisy predictor and suppressing variations across noise orders. The first term is the prediction error of the averaged noisy model and reflects the distortion of the model hypothesis space under quantum noise. The second term comes from fluctuations across noisy circuit realizations and penalizes disagreement between predictors at diferent noise orders. Thus, finite stochastic noise does not only perturb the circuit output; it also favors models whose predictions are more consistent across the noisy orders.

This decomposition gives a mechanism for the separation between training and test behavior in Fig. 1. Noise can worsen the training fit because the averaged noisy predictor is distorted relative to the nearly noiseless circuit. At the same time, the stochastic loss discourages strong sensitivity to the realized noise order, which can improve generalization in an intermediate-noise regime. The next question is how to quantify the strength of this efect as $\beta$ changes. We answer this by introducing the noise-order purity, which measures the concentration of the noise-order distribution and predicts the efective complexity of the noisy model.

## II.3. Noise-order purity predicts efective complexity

As introduced above, the surrogate model represents the noisy QML outcome as a weighted ensemble of noise-order predictors. This representation separates the circuit-dependent responses $\psi _ { m }$ from the statistical weights $p _ { m } ( \beta , g )$ . The weights are binomial probabilities determined only by the physical noise rate $\beta$ and the number of noisy locations $g .$ As $\beta$ increases from zero, probability mass spreads from the nearly noiseless branch to a range of finite-noise branches. When $g \beta \ll 1$ this distribution is well described by a Poisson approximation concentrated near low noise orders, where many-noise occurrences are rare. At larger finite noise, the distribution becomes broader and approaches a Gaussian shape in the central-limit regime. Thus, increasing $\beta$ changes not only the amount of noise, but also the structure of the ensemble being averaged over.

We quantify this spreading by the noise-order purity

$$
s _ { 2 } ( \beta , g ) = \sum _ { m = 0 } ^ { g } p _ { m } ^ { 2 } ( \beta , g ) .\tag{4}
$$

This quantity is close to one when the noisy predictor is dominated by a single noise order, such as the nearly noiseless branch at small $\beta$ . It decreases when the probability mass spreads over many finite-noise orders. Thus, lower noise-order purity corresponds to a stronger averaging efect across noisy circuit responses.

The noise-order purity also gives a quantitative scale for this transition. It is well approximated by

$$
s _ { 2 } ( \beta , g ) \approx e ^ { - 2 \beta g } I _ { 0 } ( 2 \beta g ) ,\tag{5}
$$

where $I _ { 0 }$ is the modified Bessel function of the first kind. This expression interpolates between the weak-noise expansion $s _ { 2 } ( \beta , g ) = 1 - 2 g \beta + \mathcal { O } ( g ^ { 2 } \beta ^ { 2 } )$ for $g \beta \ll 1$ and the broader-distribution scaling $s _ { 2 } ( \beta , g ) \approx ( 4 \pi g \beta ) ^ { - 1 / 2 }$ when $g \beta \gg 1$ and $\beta \leq 1 / 2$ . Rather than changing at a single rate, $s _ { 2 }$ shows a regime-dependent decay as the noise-order distribution first leaves the nearly noiseless branch and then spreads over many finite-noise orders. This regime-dependent shape is shown clearly in Fig. 1(e). This crossover provides a quantitative way to describe the finite-noise transition observed in the model performance.

The noise-order purity is also directly connected to the learning-theoretic analysis of the surrogate noisy QML model. As shown by the surrogate-loss decomposition in Eq. (3), stochastic noise induces a regularization term that penalizes disagreement among noise-order predictors. The analysis in Section A.3 shows that, when diferent noise-order predictors are weakly correlated, the efective model complexity satisfies

$$
d _ { \mathrm { e f f } } ( \beta ) \propto s _ { 2 } ( \beta , g ) ,\tag{6}
$$

Although this proportionality is derived under idealized assumptions, it is validated by the numerical results across the QML models studied in this work.

This prediction connects the noise physics to the generalization behavior in Fig. 1. As finite noise spreads probability mass across noise orders, $s _ { 2 }$ decreases and the efective number of fitted degrees of freedom is reduced. Following the local linearization analysis [47], the expected generalization gap obeys

$$
\Delta _ { \mathrm { g e n } } ( \beta ) \approx \frac { 2 \sigma _ { \mathrm { d a t a } } ^ { 2 } } { n _ { \mathrm { t r a i n } } } d _ { \mathrm { e f f } } ( \beta ) \propto s _ { 2 } ( \beta , g ) .\tag{7}
$$

Here, $\sigma _ { \mathrm { d a t a } } ^ { 2 }$ is the data-noise variance and $n _ { \mathrm { t r a i n } }$ is the number of training samples. Since these quantities are fixed across the noise sweep, the $\beta \mathrm { - }$ -dependence of the generalization gap is controlled by the efective complexity, and hence by the noise-order purity. This relation links an experimentally measurable quantity, the drop of the generalization gap, to a structural property of the noisy circuit ensemble. Its agreement with the observed decay of $\Delta _ { \mathrm { g e n } } ( \beta )$ is shown in Fig. 1(e).

## II.4. Efective-complexity scaling in trained models

The decreasing generalization gap in Fig. 1(e) provides indirect evidence for the predicted reduction in efective complexity. We test this prediction directly by estimating $d _ { \mathrm { e f f } }$ from the Jacobian of the trained noisy predictor and comparing it with the noise-order purity in Eq. (6).

![](images/208096cf7b6700a04287bed2e87880a5114455e51c5db744bae951d9eeadf315.jpg)  
Figure 2. Efective model complexity under physical noise. (a) Efective model complexity as a function of noise rate for the four noise models. Lines show the mean over ten runs and shaded regions show one standard deviation. (b) Efective model complexity normalized by its zero-noise value and plotted against $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ . Error bars show one standard deviation; fitted values of $g _ { \mathrm { e f f } }$ are reported in Section B.2.

Fig. 2(a) shows that all four noise models reduce the efective model complexity, but at diferent rates. We account for this noise-model dependence by fitting $g _ { \mathrm { e f f } }$ separately for each model. After normalization, all four curves follow the predicted dependence on $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ in Fig. 2(b). The ordering of $g _ { \mathrm { e f f } }$ can be understood from how directly each noise model changes the populations probed by the final measurement, which is $Z ^ { \otimes 4 }$ in this case. For example, phase damping suppresses only coherences and therefore afects the measured signal less directly. Depolarizing noise instead changes both populations and coherences, leading to the largest $g _ { \mathrm { e f f } }$ . The fitted values and further details are reported in Section B.2. This agreement directly validates the predicted scaling $d _ { \mathrm { e f f } } \propto s _ { 2 }$ and explains the concurrent reduction of the generalization gap in Fig. 1(e).

## II.5. Finite-noise improvement precedes trainability collapse

The finite-noise improvement identified above occurs in an intermediate regime, and should be distinguished from the very noisy regime where quantum models become dificult to train. Previous studies of noisy QML often emphasize the large-noise limit, where noise suppresses useful gradients and can efectively reduce the trainable circuit depth [12, 13, 40, 48–53]. This limit is important, but it does not explain the performance improvement observed in Fig. 1, because the improvement occurs before the model enters the gradient-collapse regime.

We evaluate this distinction by computing the parameter-wise output-gradient norm on the training inputs:

$$
\begin{array} { r } { \gamma _ { i } ( \beta ) = \sqrt { \mathbb { E } _ { \rho _ { 0 } \sim  { \mathcal { S } } _ { \mathrm { t r a i n } } } \left( \partial _ { \theta _ { i } } \phi _ { \beta } \left( \rho _ { 0 } ; \pmb { \theta } ^ { ( \beta ) } \right) \right) ^ { 2 } } , } \end{array}\tag{8}
$$

Here, $\pmb \theta ^ { ( \beta ) }$ denotes the parameters obtained at noise rate $\beta .$ . Unlike the loss gradient, which also depends on the residual, $\gamma _ { i }$ measures how strongly the model output changes with parameter i and serves as a proxy for trainability.

![](images/0884c187aeeb792fc08a35830245f69f2cc21b218c8bd45538f9415c0bd8c426.jpg)

![](images/749afe26849f6f52d1508dc47384fdf88cfd9e8deffaefd600f668757fad592c.jpg)  
Figure 3. Parameter-wise output-gradient norms under physical noise. (a–d) Root-mean-square outputgradient norm $\gamma _ { i } ( \beta )$ versus parameter index i for representative noise rates. Each point is averaged over ten independent runs. In (b), the $\beta = 0 . 5$ bit-flip channel symmetrically averages the identity and bit-flip branches in the final noisy layer, giving $\gamma _ { i } ( 0 . 5 ) = 0$ exactly; the corresponding curve is therefore absent on the logarithmic scale. (e) Output-gradient norm averaged over parameters and normalized by its zero-noise value. Roman numerals indicate three noise regimes.

Fig. 3(a–d) shows this quantity for every parameter. For the three unital noise models, the output-gradient norms decrease relatively uniformly across parameters as the noise rate increases. Amplitude damping instead develops a clear dependence on distance from the final measurement: parameters farther from the measurement have smaller output-gradient norms. This trend is consistent with recent results on quantum machine learning under non-unital noise [13].

Fig. 3(e) averages $\gamma _ { i }$ over parameters and normalizes it by its zero-noise value. This normalized quantity measures how strongly noise suppresses the output-gradient norm. All four noise models exhibit three regimes. The output-gradient norm changes little in regime I, decreases in regime II, and is strongly suppressed in regime III. Importantly, regime II coincides with the test-error improvement in Fig. 1, while the output-gradient norm remains appreciable.

Together, these results distinguish noise-induced regularization from trainability collapse. In regime II, noise-order averaging reduces efective complexity while the output-gradient norm remains appreciable. Only stronger noise drives the system into regime III, where both training and test errors increase as trainability collapses.

## II.6. Noise programming through controlled parameter perturbations

In practice, the native noise level of a quantum device is largely fixed. To exploit the nonmonotonic response under this constraint, we introduce a noise-programming strategy that injects controlled perturbations into the parameter updates during training. After each optimization step, an independent Gaussian-distributed perturbation with standard deviation σ is added to the parameters. This models additional uncertainty in the variational gate parameters. Increasing the programming strength therefore increases the efective noise level experienced by the model. To compare devices at diferent native noise levels, Fig. 4 maps the testing error across $\beta$ and $\sigma .$

![](images/0e3ef1650eb5696fa25fce105822cf48e9d8cd52b00aae6eb5bcca23123ea72c.jpg)  
Figure 4. Testing error across physical noise rates and noise-programming levels. Mean testing MSE after training at physical noise rate $\beta$ and programming level $k ,$ corresponding to a parameter-perturbation strength $\sigma = 2 ^ { - k }$ . The row labelled “none” is the unprogrammed baseline. Each cell is averaged over ten runs.

The horizontal direction recovers the non-monotonic response to native physical noise when the programming perturbation does not dominate. At stronger programming, this structure gradually disappears due to the high efective noise rate. On the other hand, moving along a vertical slice at fixed $\beta$ shows how increasing the programming strength mimics an increase in the efective noise rate for a given device. This intervention is directional: programming can increase the efective noise level but cannot undo the native physical noise. When the unprogrammed model lies on the lownoise side of the optimum, increasing the programming strength can move it towards a higher-noise but lower-error part of the landscape. Once the physical noise has passed the optimum, the same intervention cannot return the model to the lower-noise regime. Fig. 4 shows this asymmetry for all four physical noise models. This directional accessibility turns the non-monotonic response into a control strategy. Noise programming can therefore move an under-regularized QML model towards the intermediate-noise optimum, where noise-induced regularization improves its predictive performance. Practically, mini-batch stochastic gradient descent generates fluctuations in parameter updates and may therefore provide a natural algorithmic route to noise programming [54, 55].

## II.7. Noise-induced regularization persists in molecular QML

The four-qubit model allows the finite-noise mechanism to be examined in a setting where the noise response and local model complexity can both be evaluated in detail. We next test whether the same mechanism remains relevant when the predictor contains a larger quantum representation and trainable classical post-processing. Quantum and hybrid quantum–classical models have recently been explored for learning molecular properties and graph representations [56–61]. We study eight- and nine-qubit quantum graph neural networks that predict the HOMO–LUMO gap for molecules drawn from QM9-HA8 and PCQM4Mv2-HA9 [62–64] under gate-level quantum noise. Fig. 5 compares the performance across variable factors including training-set sizes, noise models and molecular datasets.

Across these comparisons in Fig. 5(a–c), the testing MSE consistently exhibits a finite-noise optimum. The training MSE instead increases with the noise rate (Fig. 5(d)), indicating the increasing bias introduced by noise. Fig. 5(e) confirms the linear relation between the generalization gap and the efective model complexity, while Fig. 5(f) shows that the efective complexity remains proportional to the noise-order purity. These results extend the noise-induced generalization mechanism beyond the four-qubit model: finite noise reduces efective complexity, and the resulting reduction in the generalization gap can improve testing performance before the training error becomes dominant.

![](images/6807600062a4765437b80e81f41fca2e74944a8974ac9a28e0e5c83a65764774.jpg)

![](images/68f1bb02772e770ed607480a061a813b09a6be5d7af149efc3aae96862b0e7d7.jpg)

![](images/f0c6acea26a0b392c46ee8b42379092648221184a7aebd410efd203fa6281af9.jpg)

![](images/f55676741be05fdcb9f8a915c5335e995513415f5beebffec523b1f02b2e59dd.jpg)

![](images/09d87b7166ccee86199fab0c844fc77e5beb79385ab87adce0ca85fa1b58d22f.jpg)

![](images/1f51a111954b845a4d4dc151583ebfc8049c470cb72464dd904379f437474ee1.jpg)  
QM9 BF, = QM9 BF, = QM9 DP, = PCQM BF, =  
Figure 5. Noise-induced regularization in molecular QML. Testing MSE is compared (a) between $n _ { \mathrm { t r a i n } } = 8 0$ and 160 on QM9-HA8 under bit-flip noise, (b) between bit-flip and depolarizing noise on $\mathrm { Q M 9 }$ -HA8 with $n _ { \mathrm { t r a i n } } ~ = ~ 1 6 0$ , and (c) between QM9-HA8 and PCQM4Mv2-HA9 under bit-flip noise with $n _ { \mathrm { t r a i n } } = 8 0$ (d) Training MSE for the four distinct settings in (a–c). (e) Generalization gap versus efective model complexity, with both quantities normalized by their values at $\beta = 0$ within each run. (f) Efective model complexity normalized by its value at $\beta = 0$ versus $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ , where $g _ { \mathrm { e f f } }$ is fitted separately for each setting. The QM9-HA8 and PCQM4Mv2-HA9 models are evaluated after 300 and 100 epochs, respectively. Lines and shaded regions in (a–d) show the mean and one standard deviation over five runs; points and error bars in $\mathrm { ( e , f ) }$ show the corresponding mean and standard deviation.

## II.8. Finite-noise optima disappear in the large-sample limit

The finite-noise optimum is not unconditional. Let SNR denote the strength of the learnable signal relative to the data-noise variance. The analysis in Section A gives

$$
s ^ { \star } = \frac { \mathrm { S N R } } { \mathrm { S N R } + d _ { \mathrm { e f f } } ^ { ( 0 ) } / n _ { \mathrm { t r a i n } } } , \qquad s _ { 2 } ( \beta ^ { \star } , g ) = s ^ { \star } .\tag{9}
$$

Here, $d _ { \mathrm { e f f } } ^ { ( 0 ) }$ is the efective model complexity in the absence of physical noise. When the data set is given and the SNR is fixed, the optimum is therefore controlled by the ratio $d _ { \mathrm { e f f } } ^ { ( 0 ) } / n _ { \mathrm { t r a i n } }$ . Decreasing this ratio moves $s ^ { \star }$ towards one and hence moves $\beta ^ { \star }$ towards the low-noise limit.

![](images/9cc515d9007a1e54059ff9a430ccdd6d6e7bcb350e3292df204714045c281080.jpg)  
Figure 6. Training-set-size dependence of the finite-noise response. (a) Training MSE, (b) testing MSE and (c) generalization gap under bit-flip noise for nested QM9-HA8 training sets with $n _ { \mathrm { t r a i n } } = 8 0$ , 160, 320 and 1000. The first three models are evaluated after 300 epochs and the $n _ { \mathrm { t r a i n } } = 1 0 0 0$ model after 100 epochs. Lines and shaded regions show the mean and one standard deviation over five fixed runs.

We test this shift using nested QM9-HA8 training sets with $n _ { \mathrm { t r a i n } } = 8 0 , 1 6 0$ , 320 and 1000 $( \mathrm { F i g . ~ 6 } )$ . The three smaller training sets retain a clear finite-noise reduction in testing error. For $n _ { \mathrm { t r a i n } } = 1 0 0 0$ , the low-noise generalization gap is much smaller and the nonmonotonic response almost disappears. Consistent with Eq. (9), the optimum generally shifts towards the low-noise limit as $n _ { \mathrm { t r a i n } }$ increases. The exception is $n _ { \mathrm { t r a i n } } = 8 0$ and 160, for which the optimum occurs at the same sampled noise rate. The model has $q = 3 1 0$ trainable parameters, so both cases are overparameterized [19, 65–67]. In this regime, the efective model complexity is limited by the lack of training data. Therefore, increasing $n _ { \mathrm { t r a i n } }$ also allows the efective model complexity to grow by exposing additional model directions. Consequently, $d _ { \mathrm { e f f } } ^ { ( 0 ) } / n _ { \mathrm { t r a i n } }$ remains nearly unchanged. Once $n _ { \mathrm { t r a i n } }$ exceeds $q ,$ this data limitation is removed and the optimum moves towards zero. This is also consistent with the statistical intuition. When the training-set size approaches infinity, the model sees the full data distribution, and therefore the generalization gap vanishes because the testing set contains no additional statistical information. The increasing training error with $n _ { \mathrm { t r a i n } }$ in $\mathrm { F i g . 6 ( a ) }$ together with the shrinking generalization gap in Fig. $6 ( \mathrm { c ) }$ , reflects the mitigation of overfitting towards the large-sample limit.

## III. DISCUSSION

In this work, we develop a statistical learning theory for noisy quantum machine learning models. It connects microscopic noise processes to macroscopic learning performance and explains the nonmonotonic efect of noise on testing performance. The key quantity is the noise-order purity, which measures how broadly the model response is mixed across noise orders. The intermediate noise regime is very diferent from the weak- and strong-noise limits. It is not well described by perturbative error accumulation, which controls worst-case deviation, or by trainability collapse, which characterizes the loss of representation. Noise-order purity and the learning-theoretic analysis bridge this missing layer, while the numerical results confirm the separation of the three regimes.

The non-monotonic efects reported in recent studies are therefore not occasional artifacts, but a general phenomenon that can be understood and predicted.

Noise is not intrinsically beneficial, and even when it is beneficial, the improvement is not unconditional. Our theoretical analysis shows that the finite-noise optimum behaves diferently in over- and underparameterized models and can eventually disappear in the large-sample limit. This analysis combines a noise-order surrogate with a local linearization around the trained optimum. An understanding of noise-induced regularization focusing on optimization dynamics remains open. It is also an open question whether the physical noisy model can be analyzed directly to capture the non-monotonicity without introducing the surrogate.

One intriguing result is that noise programming can move an under-regularized noisy model towards its finite-noise optimum. This opens a new direction for harnessing noise in quantum machine learning based on the learning-theoretic analysis. Diferent circuit components need not contribute equally to the information used for prediction, and noise acting on them need not have the same efect. Understanding this dependence may help allocate protection towards the components that matter most for prediction in partially fault-tolerant architectures.

The present simulations assume independent gate-level noise. The central hardware question is whether noise-order purity continues to predict model complexity and generalization when errors are correlated or time dependent and circuit outputs are estimated from finite shots. Addressing these efects will extend the prediction and programming of finite-noise optima to more complex and realistic quantum application settings.

## IV. METHODS

## IV.1. QML models and datasets

## IV.1.1. Four-qubit regression model

We use the diabetes regression dataset distributed with scikit-learn [43, 44]. From the ten clinical covariates, we retain body mass index and the logarithm of the serum triglyceride level (features 2 and 8 in the dataset). From the 442 samples, we randomly select disjoint training and testing sets of 40 and 400 samples, respectively. The two input features are independently rescaled to $[ 0 , \pi ]$ and the regression target is rescaled to [−1, 1]. Both transformations are fitted on the training set and subsequently applied to the testing set.

We use a four-qubit parametrized quantum circuit adapted from Refs. [31, 35]. The circuit comprises nine layers with the same gate layout and independently trainable parameters. In each layer, the two input features are re-uploaded through $R _ { X }$ rotations on qubits 0 and 2. The trainable block then applies IsingXX rotations to four nearest-neighbour pairs on a periodic ring, followed by one $R _ { Y }$ rotation on each qubit. Each layer therefore contains eight trainable parameters, giving $q = 7 2$ parameters in total. The scalar model output $\phi _ { \beta } ( \rho _ { 0 } ; \theta )$ is the expectation value of $Z ^ { \otimes { \bar { 4 } } }$ under the noisy circuit dynamics.

## IV.1.2. Molecular QGNN models

We study molecular regression on fixed-size subsets of QM9 and PCQM4Mv2 [62–64]. QM9-HA8 contains molecules with eight heavy atoms, while PCQM4Mv2-HA9 contains molecules with nine heavy atoms. In both cases, the target is the HOMO–LUMO gap in electronvolts. We construct fixed ordered training pools of 2,000 and 320 molecules for QM9-HA8 and PCQM4Mv2-HA9, respectively, together with disjoint testing sets of 1,500 molecules. All reported training sets are nested prefixes of these pools, and the testing sets are kept fixed across training sizes, noise rates and runs.

Each heavy atom is assigned to one qubit, giving eight- and nine-qubit quantum graph neural networks for QM9-HA8 and PCQM4Mv2-HA9, respectively. Atom features are encoded by singlequbit rotations, and chemical bonds determine bond-type-dependent two-qubit interactions in one EDU-QGC-inspired graph-circuit layer [32, 60, 68, 69]. We extract the complete computationalbasis probability vector from the final quantum state and use a trainable linear readout,

$$
\phi _ { \beta } ( x ; \pmb { \theta } ) = b + \sum _ { z } w _ { z } \langle z | \rho _ { \beta } ( x ; \pmb { \theta } _ { \mathrm { q c } } ) | z \rangle , \qquad \pmb { \theta } = ( \pmb { \theta } _ { \mathrm { q c } } , \pmb { w } , b ) .\tag{10}
$$

Here x denotes the complete molecular input and $\theta _ { \mathrm { q c } }$ contains the trainable quantum-circuit parameters. The QM9-HA8 model contains 53 quantum-circuit and 257 classical linear-readout parameters, while the PCQM4Mv2-HA9 model contains 56 and 513, respectively, giving 310 and 569 trainable parameters in total. Further circuit and parameter-counting details are given in Section C.1. In Section C.3, we use a more complex nonlinear classical readout to show that the finite-noise response is not an artifact of the linear readout.

## IV.2. Quantum noise models

We consider four single-qubit noise channels: depolarizing, bit-flip, phase-damping and amplitudedamping noise. At each noisy location, the channel acts on the density matrix through the Kraus map [70]

$$
\mathcal { E } _ { \beta } ( \rho ) = \sum _ { \ell } K _ { \ell } ( \beta ) \rho K _ { \ell } ^ { \dagger } ( \beta ) ,\tag{11}
$$

where $\beta \in [ 0 , 1 ]$ is the physical noise rate. In the single-qubit case, the associated Kraus operators are:

$$
\mathrm { d e p o l a r i z i n g : \quad } K _ { 0 } = \sqrt { 1 - \beta } I , \qquad ( K _ { 1 } , K _ { 2 } , K _ { 3 } ) = \sqrt { \frac { \beta } { 3 } } ( X , Y , Z ) ,
$$

$$
{ \mathrm { b i t ~ f l i p : } } \quad K _ { 0 } = { \sqrt { 1 - \beta } } I , \qquad K _ { 1 } = { \sqrt { \beta } } X ,
$$

$$
\mathrm { p h a s e \ d a m p i n g } { \cdot } \quad K _ { 0 } = \left( \frac { 1 } { 0 } \begin{array} { c } { { 0 } } \\ { { \sqrt { 1 - \beta } } } \end{array} \right) , \qquad K _ { 1 } = \left( \begin{array} { c c } { { 0 } } & { { 0 } } \\ { { 0 } } & { { \sqrt { \beta } } } \end{array} \right) ,
$$

$$
K _ { 0 } = { \binom { 1 } { 0 } } \ { \sqrt { 1 - \beta } } ) , \qquad K _ { 1 } = { \binom { 0 } { 0 } } \ { \sqrt { \beta } } ) .
$$

amplitude damping:

(12)

In all models, the selected channel is applied after every elementary gate in the data encoding and trainable circuit. After a two-qubit gate, the single-qubit channel is applied independently to both participating qubits. No noise is applied to the classical post-processing including the regression. Noiseless circuits are evaluated with state-vector simulation, while noisy circuits are evaluated with the density-matrix simulator in PennyLane [71]. The four-qubit circuit has a fixed number of noisy locations, whereas the molecular circuit follows the input graph and therefore has a molecule-dependent count. The corresponding gate accounting is given in Section C.1.

## IV.3. Training and evaluation

The four-qubit model is trained by minimizing the mean-squared error between its scalar output and the regression target. We use full-batch Adam optimization [72] with learning rate 0.01 for 800 epochs and do not use early stopping. The physical noise channel at the selected value of $\beta$ is included during both training and evaluation. Testing samples are not used to update the parameters.

For each physical noise model and noise rate, we train ten independently initialized models. The 72 initial parameters in each run are sampled from a standard normal distribution using a fixed random seed. The same set of ten initial parameter vectors is reused across all noise rates and noise models, so changes across the noise sweep are not caused by diferent initializations. Training and testing metrics are evaluated after the final update. Reported curves show the mean over the ten runs, and uncertainty bands or error bars show one sample standard deviation across runs.

The molecular models are trained end to end by minimizing the mean-squared error in electronvolts squared. We use Adam with learning rate 0.03, a mini-batch size of 16 and gradient clipping with maximum norm 1. Each epoch visits every training molecule once in a deterministically shuffled order. The readout bias is initialized to the mean target in the training set. QM9-HA8 and PCQM4Mv2-HA9 models are trained for 300 and 100 epochs, respectively, and all metrics are evaluated at the final checkpoint.

For each molecular setting, five fixed initializations are used across the complete noise sweep. The physical noise channel is included during both training and evaluation, and every final checkpoint is evaluated on all 1,500 testing molecules. Molecular curves show the mean over the five runs, with uncertainty given by one sample standard deviation.

## IV.4. Local model complexity and efective noise count

We estimate model complexity by linearizing the physical noisy predictor around the trained parameters. For run r,

$$
\phi _ { \beta } ( \rho _ { a } ; \pmb { \theta } ^ { ( \beta , r ) } + \delta \pmb { \theta } ) \approx \phi _ { \beta } ( \rho _ { a } ; \pmb { \theta } ^ { ( \beta , r ) } ) + \sum _ { i = 1 } ^ { q } J _ { a i } ^ { ( r ) } ( \beta ) \delta \theta _ { i } , \qquad J _ { a i } ^ { ( r ) } ( \beta ) = \frac { \partial \phi _ { \beta } ( \rho _ { a } ; \pmb { \theta } ) } { \partial \theta _ { i } } \bigg \rvert _ { \pmb { \theta = \theta } ^ { ( \beta , r ) } } .\tag{13}
$$

The nonlinear model therefore reduces locally to a linear regression problem with the predictor Jacobian $J ^ { ( r ) } ( \beta )$ as its design matrix. Let $\lambda _ { i } ^ { ( r ) } ( \beta )$ denote the eigenvalues of $J ^ { ( r ) } ( \beta ) ^ { \mathsf { T } } J ^ { ( r ) } ( \beta ) / n _ { \mathrm { t r a i n } } .$ These eigenvalues quantify how strongly the model output responds along diferent parameter directions. A hard count of nonzero eigenvalues is sensitive to numerical precision and treats directions of very diferent strengths equally. We instead use the soft count [47, 73]

$$
d _ { \mathrm { e f f } } ^ { ( r ) } ( \beta ) = \sum _ { j = 1 } ^ { q } \frac { \lambda _ { j } ^ { ( r ) } ( \beta ) } { \lambda _ { j } ^ { ( r ) } ( \beta ) + 1 0 ^ { - 4 } } \approx \sum _ { j = 1 } ^ { q } \mathbb { I } \Big ( \lambda _ { j } ^ { ( r ) } ( \beta ) \neq 0 \Big ) .\tag{14}
$$

Directions with eigenvalues well above $1 0 ^ { - 4 }$ contribute approximately one, whereas weaker directions contribute proportionally less. The same counting scale is used for every noise model, noise rate and run.

For the molecular models, $J ^ { ( r ) } ( \beta )$ is taken with respect to the complete parameter vector θ, including both the quantum-circuit and linear-readout parameters.

For comparisons across noise models, each run is normalized by its own zero-noise value, $\widetilde { d } _ { \mathrm { e f f } } ^ { ( r ) } ( \beta ) =$ $d _ { \mathrm { e f f } } ^ { ( r ) } ( \beta ) / d _ { \mathrm { e f f } } ^ { ( r ) } ( 0 )$ , before averaging across runs. We obtain the efective noise count by fitting this averaged curve to the exact binomial noise-order purity,

$$
g _ { \mathrm { e f f } } = \operatorname * { a r g m i n } _ { g \in \{ 1 , \dots , 1 0 0 0 \} } \frac { 1 } { N _ { \beta } } \sum _ { \beta } \left( s _ { 2 } ( \beta , g ) - \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \widetilde { d } _ { \mathrm { e f f } } ^ { ( r ) } ( \beta ) \right) ^ { 2 } ,\tag{15}
$$

where $s _ { 2 } ( \beta , g )$ is the noise-order purity defined in $\mathrm { E q . \ ( 4 ) }$ , and $R = 1 0$ and 5 for the four-qubit and molecular experiments, respectively. The dashed reference curve in Fig. 1(e) uses the exact four-qubit noisy-location count $g \ : = \ : 1 2 6$ , rather than a fitted count. The fitted g summarizes how strongly a noise model changes the measured predictor across these locations. Its channel dependence and order-resolved numerical validation are discussed in Section B.

The parameter-wise output-gradient norm used in Fig. 3 is computed from the same predictor Jacobian:

$$
\gamma _ { i } ^ { ( r ) } ( \beta ) = \sqrt { \frac { 1 } { n _ { \mathrm { t r a i n } } } \sum _ { a = 1 } ^ { n _ { \mathrm { t r a i n } } } \left( J _ { a i } ^ { ( r ) } ( \beta ) \right) ^ { 2 } } .\tag{16}
$$

The profiles in Fig. $\mathrm { 3 ( a - d ) }$ show $\textstyle \sum _ { r = 1 } ^ { 1 0 } \gamma _ { i } ^ { ( r ) } ( \beta ) / 1 0$ . For the scalar response in Fig. $3 ( \mathrm { e } )$ , we average $\gamma _ { i } ^ { ( r ) }$ over both parameters and runs and divide by the corresponding average at $\beta = 0$

## IV.5. Noise programming

Noise programming is implemented by adding an independent Gaussian perturbation after every Adam update. If $\Delta _ { t } ^ { \mathrm { A d a m } }$ denotes the optimizer update at step t, the programmed parameters obey

$$
\theta _ { t + 1 } = \theta _ { t } + \Delta _ { t } ^ { \mathrm { A d a m } } + \sigma \pmb { \xi } _ { t } , \qquad \xi _ { t } \sim \mathcal { N } ( 0 , I _ { q } ) .\tag{17}
$$

A new perturbation is sampled independently at every update and for every run. We use programming strengths $\sigma = 2 ^ { - k }$ for $k = 1 , \ldots , 1 0$ , together with an unprogrammed baseline $\sigma = 0$ . All other training settings, including the initial parameters and the physical noise channel, are held fixed.

For every physical noise rate $\beta$ and programming level k, the model is retrained for 800 full-batch epochs in each of the ten runs. Each cell in the programming landscape is the final testing MSE averaged across these runs.

## DATA AVAILABILITY

The diabetes dataset is distributed with scikit-learn [43, 44]. QM9 [62] and PCQM4Mv2 [63, 64] are publicly available from their original sources. The fixed molecular subsets and data splits used in this study, the processed numerical data underlying the figures and the trained checkpoints are available at https: $/ / { \tt g } \dot { \tt 1 }$ thub.com/dongsnaq/Finite-Noise-Generalization-QML.

## CODE AVAILABILITY

The code developed in this work is available via GitHub at https://github.com/dongsnaq/ Finite-Noise-Generalization-QML.

## ACKNOWLEDGEMENTS

The machine learning research was supported by the Ofice of Naval Research under MURI Award No. N00014-23-1-2001. The quantum computing research was supported by the U.S. National Science Foundation through the Chemical Theory, Models, and Computational Methods (CTMC) program under Grant No. CHE-2515209 to X.L. This research used resources of the National Energy Research Scientific Computing Center, a DOE Ofice of Science User Facility supported by the Ofice of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231, using NERSC awards DDR-ERCAP0038972 and DDR-ERCAP0038957 (Y.D.).

## AUTHOR CONTRIBUTIONS

Y.D. and X.L. conceived the project. Y.D. developed the theory, designed and performed the numerical study, analyzed the results and prepared the original manuscript. Z.J. developed the initial four-qubit simulation, which Z.Z. subsequently extended. Z.Z. also developed the molecular QML simulation framework. Both Z.Z. and Z.J. contributed to the numerical study and initial explorations. All authors contributed to the discussion and writing of the manuscript.

## COMPETING INTERESTS

The authors declare no competing interests.

## CONTENTS

I. Introduction 1   
II. Results 2   
II.1. Finite noise induces non-monotonic generalization 2   
II.2. A noise-order surrogate explains noise-induced regularization 3   
II.3. Noise-order purity predicts efective complexity 4   
II.4. Efective-complexity scaling in trained models 5   
II.5. Finite-noise improvement precedes trainability collapse 6   
II.6. Noise programming through controlled parameter perturbations 7   
II.7. Noise-induced regularization persists in molecular QML 8   
II.8. Finite-noise optima disappear in the large-sample limit 9   
III. Discussion 10   
IV. Methods 11   
IV.1. QML models and datasets 11   
IV.1.1. Four-qubit regression model 11   
IV.1.2. Molecular QGNN models 11   
IV.2. Quantum noise models 12   
IV.3. Training and evaluation 13   
IV.4. Local model complexity and efective noise count 13   
IV.5. Noise programming 14   
A. Theory of noise-induced regularization 17   
A.1. Problem setup and stochastic noise-occurrence expansion 17   
A.2. Noise-induced regularization in the surrogate loss 19   
A.3. Noise-induced generalization efect and reduced efective model complexity 20   
A.4. Scaling regimes of the noise-order purity 22   
A.5. Mean-squared prediction error in the presence of quantum noise 23   
A.6. Extensions beyond the Bernoulli noise model 24   
B. Numerical validation of the noise-order surrogate 25   
B.1. Surrogate efective complexity in trained models 25   
B.2. Interpretation of the efective noise count 26   
B.3. Eficient evaluation of noisy order-resolved quantities by dynamic programming 27   
C. Molecular QML simulation details and ablation studies 28   
C.1. Circuit structure and other simulation details 28   
C.2. Ablation of optimization epochs 29   
C.3. Ablation of more complex classical post-processing 29   
C.4. Ablation of noise models 30   
References 31

## Appendix A: Theory of noise-induced regularization

## A.1. Problem setup and stochastic noise-occurrence expansion

In this section, we introduce the setup of the problems we consider in this work. By introducing a stochastic noise-occurrence expansion, we discuss a representation of QML models in the presence of noise and use it as a surrogate model to understand the mechanism behind non-monotonic noise response.

The $\mathrm { Q M L }$ model is implemented in terms of a sequence of parametrized gates $\{ U _ { 1 } ( \theta _ { 1 } ) , \cdot \cdot \cdot , U _ { g } ( \theta _ { g } ) \}$ The input density matrix $\rho _ { 0 }$ carries the information of the data point, such as encodings of chemical structures and molecular features. The idealized output of such a parametrized circuit is

$$
\rho _ { \mathrm { o u t } } ^ { ( 0 ) } = \mathcal { U } _ { g } \circ \cdots \circ \mathcal { U } _ { 1 } ( \rho _ { 0 } ) ,\tag{A1}
$$

where $\mathcal { U } _ { j } ( \boldsymbol { \rho } ) = U _ { j } ( \boldsymbol { \theta } _ { j } ) { \rho } U _ { j } ^ { \dagger } ( \boldsymbol { \theta } _ { j } )$ is the corresponding conjugate action. For simplicity, we assume that the data enter through the initial state $\rho _ { 0 }$ . More general data-dependent gate maps leave the analysis unchanged and are reflected in the model-dependent quantities appearing in the final results. The final quantum state is measured with some physical observable, for example an observable O, so that the information is extracted as

$$
\phi _ { 0 } ( \rho _ { 0 } ; \pmb { \theta } ) : = \mathrm { T r } ( O \rho _ { \mathrm { o u t } } ^ { ( 0 ) } ) .\tag{A2}
$$

In the presence of noise, we assume that after each ideal gate action there is an additional quantum channel distorting the idealized dynamics. Suppose the noise channels are $\mathcal { E } _ { 1 } , \cdots , \mathcal { E } _ { g }$ . The noisy output is

$$
\rho _ { \mathrm { o u t } } ^ { \mathcal { E } } = \mathcal { E } _ { g } \circ \mathcal { U } _ { g } \circ \cdot \cdot \cdot \circ \mathcal { E } _ { 1 } \circ \mathcal { U } _ { 1 } ( \rho _ { 0 } ) ,\tag{A3}
$$

and the corresponding noisy predictor is

$$
\begin{array} { r } { \phi _ { \mathcal { E } } ( \rho _ { 0 } ; \pmb { \theta } ) : = \mathrm { T r } ( O \rho _ { \mathrm { o u t } } ^ { \mathcal { E } } ) . } \end{array}\tag{A4}
$$

We consider a training set $\pmb { S } = \{ ( \rho _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ and the corresponding mean square loss:

$$
\begin{array} { r } { \mathcal { L } ( \pmb { \theta } ) = \mathbb { E } _ { S } \left\| \phi _ { \mathcal { E } } ( \rho _ { 0 } ; \pmb { \theta } ) - y \right\| ^ { 2 } . } \end{array}\tag{A5}
$$

To understand the non-monotonic noise response, we introduce a stochastic noise-occurrence model to study the mechanism. We first discuss the simplest model to show the intuition. Nevertheless, the mechanism does not only apply to this simple model. We discuss the general form of the model and the corresponding extension at the end.

Consider the simplest case where the noise channel at each location either does nothing or applies another noisy operation. This captures, for example, dephasing or bit-flip noise, where an additional Pauli operator is applied at random. In this case, each noise realization generates an efective circuit that deviates from the idealized circuit by one or more noise occurrences. Therefore, the noise realization can be described by a stochastic process, and the corresponding predictor is associated with a particular noise path. This motivates a stochastic surrogate model in which disagreement among noise-induced predictors appears as an explicit variance term. Such a representation provides a mechanism by which moderate noise can reduce efective model complexity and improve generalization, while excessive noise can destroy learnable signal.

To quantify this intuition, we consider a stochastic noise model where the noise at the j-th location is associated with a Bernoulli random variable $z _ { j } \sim \mathrm { B e r } ( \beta )$ . Conditioned on $z _ { j }$ , the inserted channel is

$$
\begin{array} { r } { \mathscr { E } _ { j , z _ { j } } = ( 1 - z _ { j } ) \mathrm { I d } + z _ { j } \mathscr { N } _ { j } = \left\{ \begin{array} { l l } { \mathrm { I d } , } & { z _ { j } = 0 , } \\ { \mathscr { N } _ { j } , } & { z _ { j } = 1 , } \end{array} \right. } \end{array}\tag{A6}
$$

where ${ \mathcal { N } } _ { j }$ is another noise channel, such as a conjugate action of an additional gate. The averaged single-location channel is therefore

$$
\mathbb { E } _ { z _ { j } } \mathscr { E } _ { j , z _ { j } } = ( 1 - \beta ) \mathrm { I d } + \beta { \cal N } _ { j } .\tag{A7}
$$

For a noise realization $\boldsymbol { z } = ( z _ { 1 } , \cdots , z _ { g } )$ , the conditional noisy predictor is

$$
\begin{array} { r } { \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) : = \operatorname { T r } \left( O \mathscr { E } _ { g , z _ { g } } \circ \mathcal { U } _ { g } \circ \dotsb \circ \mathscr { E } _ { 1 , z _ { 1 } } \circ \mathcal { U } _ { 1 } ( \rho _ { 0 } ) \right) . } \end{array}\tag{A8}
$$

Let

$$
c ( z ) = \sum _ { j = 1 } ^ { g } z _ { j }\tag{A9}
$$

denote the number of noise occurrences. We define the occurrence-m predictor as the average over all noise realizations with exactly m noise occurrences:

$$
\begin{array} { r } { \psi _ { m } ( \rho _ { 0 } ; \pmb { \theta } ) : = \mathbb { E } _ { z } \left( \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) \ | \ c ( z ) = m \right) . } \end{array}\tag{A10}
$$

The noisy mean predictor induced by this stochastic noise-occurrence model can then be expanded with respect to the occurrence count:

$$
\begin{array} { r l } { \displaystyle \phi _ { \beta } ( \rho _ { 0 } ; \pmb { \theta } ) : = \mathbb { E } _ { z } \left( \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) \right) } & { } \\ { \displaystyle } & { = \sum _ { m = 0 } ^ { g } \mathbb { P } ( c ( z ) = m ) \psi _ { m } ( \rho _ { 0 } ; \pmb { \theta } ) } \\ { \displaystyle } & { = \sum _ { m = 0 } ^ { g } \binom { g } { m } \beta ^ { m } ( 1 - \beta ) ^ { g - m } \psi _ { m } ( \rho _ { 0 } ; \pmb { \theta } ) . } \end{array}\tag{A11}
$$

Here, the formalism can be viewed as a noise-occurrence expansion: the basis elements are efective predictors generated by diferent noise occurrence counts, and the expansion weights are determined by the binomial distribution of the occurrence count.

We consider the following stochastic loss as a surrogate model for analysis:

$$
\widetilde { \mathcal { L } } ( \pmb { \theta } , \beta ) = \mathbb { E } _ { S , z } \left. \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) - y \right. ^ { 2 } .\tag{A12}
$$

This surrogate loss is useful because it exposes an explicit variance term across noise realizations. By the bias-variance identity, we have

$$
\begin{array} { r l } & { \widetilde { \mathcal { L } } ( \pmb { \theta } , \beta ) = \mathbb { E } _ { S , z } \left\| \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) - y \right\| ^ { 2 } } \\ & { \quad \quad \quad = \mathbb { E } _ { S } \left\| \phi _ { \beta } ( \rho _ { 0 } ; \pmb { \theta } ) - y \right\| ^ { 2 } + \mathbb { E } _ { S , z } \mathrm { V a r } _ { z } \left( \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) \right) . } \end{array}\tag{A13}
$$

The first term is the mean-channel loss associated with the averaged noisy predictor $\phi _ { \beta }$ , while the second term measures the disagreement among predictors generated by diferent noise realizations. Thus, in the stochastic surrogate model, disagreement among noise realizations appears as an explicit variance penalty. In the noisy mean predictor itself, the same noise-occurrence structure acts through the ensemble average $\begin{array} { r } { \phi _ { \beta } = \sum _ { m } p _ { m } \psi _ { m } , } \end{array}$ , which filters the accessible function class and provides a mechanism for noise-induced regularization.

## A.2. Noise-induced regularization in the surrogate loss

In this subsection, we analyze the regularization term in Eq. (A13). Recall that

$$
p _ { m } ( \beta , g ) : = \mathbb { P } ( c ( z ) = m ) = { \binom { g } { m } } \beta ^ { m } ( 1 - \beta ) ^ { g - m } ,\tag{A14}
$$

where $\begin{array} { r } { c ( z ) = \sum _ { j = 1 } ^ { g } z _ { j } } \end{array}$ is the number of noise occurrences. The variance term in the stochastic surrogate loss is

$$
\begin{array} { r l } & { \mathcal { R } ( \pmb { \theta } ; \beta , g ) : = \mathbb { E } _ { \mathcal { S } } \mathrm { V a r } _ { z } \left( \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) \right) = \mathbb { E } _ { \mathcal { S } , z } \left\| \phi _ { z } ( \rho _ { 0 } ; \pmb { \theta } ) - \phi _ { \beta } ( \rho _ { 0 } ; \pmb { \theta } ) \right\| ^ { 2 } } \\ & { \quad \quad \quad = \mathbb { E } _ { \mathcal { S } , z } \left\| \displaystyle \sum _ { m = 0 } ^ { g } \left( \mathbb { I } _ { c ( z ) = m } - p _ { m } ( \beta , g ) \right) \psi _ { m } ( \rho _ { 0 } ; \pmb { \theta } ) \right\| ^ { 2 } } \\ & { \quad \quad \quad = \mathbb { E } _ { \mathcal { S } } \displaystyle \sum _ { m _ { 1 } , m _ { 2 } = 0 } ^ { g } \psi _ { m _ { 1 } } ( \rho _ { 0 } ; \pmb { \theta } ) \psi _ { m _ { 2 } } ( \rho _ { 0 } ; \pmb { \theta } ) \Lambda _ { m _ { 1 } , m _ { 2 } } ( \beta , g ) . } \end{array}\tag{A15}
$$

Here, the covariance matrix $\Lambda ( \beta , g )$ is defined by

$$
\begin{array} { r } { \Lambda _ { m _ { 1 } , m _ { 2 } } ( \beta , g ) : = \mathbb { E } _ { g } \left( \mathbb { I } _ { c ( z ) = m _ { 1 } } \mathbb { I } _ { c ( z ) = m _ { 2 } } \right) - p _ { m _ { 1 } } ( \beta , g ) p _ { m _ { 2 } } ( \beta , g ) = \delta _ { m _ { 1 } , m _ { 2 } } p _ { m _ { 1 } } ( \beta , g ) - p _ { m _ { 1 } } ( \beta , g ) p _ { m _ { 2 } } ( \beta , g ) , } \end{array}\tag{A16}
$$

where $\delta _ { m _ { 1 } , m _ { 2 } }$ is the Kronecker delta symbol. It can be shown that the covariance matrix $\Lambda ( \beta , g )$ is positive semidefinite. The regularization term can also be written as a pairwise disagreement penalty:

$$
\mathcal { R } ( \pmb { \theta } ; \beta , g ) = \frac { 1 } { 2 } \mathbb { E } _ { S } \sum _ { m _ { 1 } , m _ { 2 } = 0 } ^ { g } p _ { m _ { 1 } } ( \beta , g ) p _ { m _ { 2 } } ( \beta , g ) \left\| \psi _ { m _ { 1 } } ( \rho _ { 0 } ; \pmb { \theta } ) - \psi _ { m _ { 2 } } ( \rho _ { 0 } ; \pmb { \theta } ) \right\| ^ { 2 } .\tag{A17}
$$

This form shows that the stochastic surrogate loss penalizes disagreement among predictors associated with diferent noise occurrence counts.

The trace of the covariance matrix is

$$
\mathrm { T r } ( \Lambda ( \beta , g ) ) = 1 - \sum _ { m = 0 } ^ { g } p _ { m } ^ { 2 } ( \beta , g ) = : 1 - s _ { 2 } ( \beta , g ) ,\tag{A18}
$$

where the noise-order purity

$$
s _ { 2 } ( \beta , g ) : = \sum _ { m = 0 } ^ { g } p _ { m } ^ { 2 } ( \beta , g )\tag{A19}
$$

measures the concentration of the noise occurrence-count distribution. Therefore, $1 - s _ { 2 } ( \beta , g )$ measures the total variance of this distribution. When the occurrence-count distribution spreads over more values of m, $s _ { 2 } ( \beta , g )$ decreases and the total variance captured by $\Lambda ( \beta , g )$ increases. In the next subsection, we show that the same concentration quantity $s _ { 2 } ( \beta , g )$ also controls the local sensitivity of the averaged noisy predictor.

## A.3. Noise-induced generalization efect and reduced efective model complexity

In this subsection, we analyze how the noise-induced regularization term afects the generalization of the stochastic surrogate model. We work with the occurrence-count coarsened surrogate, where a noise realization with $c ( z ) = m$ is represented by the predictor $\psi _ { m }$ . This is the surrogate model used to expose the efect of noise-occurrence averaging on model complexity.

In this subsection, we slightly overload notation and write $\psi _ { m } ( \pmb { \theta } ) \in \mathbb { R } ^ { n }$ for the vector of predictions $( \psi _ { m } ( \rho _ { i } ; \pmb { \theta } ) ) _ { i = 1 } ^ { n }$ , and $y \in \mathbb { R } ^ { n }$ for the vector of labels. The averaged noisy predictor on the training set is

$$
\phi ( \pmb \theta , \beta ) = \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) \psi _ { m } ( \pmb \theta ) .\tag{A20}
$$

Using the decomposition in $\mathrm { E q . \ ( A 1 3 ) }$ , the stochastic surrogate loss can be written as

$$
\widetilde { \mathcal { L } } ( \pmb { \theta } , \beta ) = \frac { 1 } { n } \left\| \phi ( \pmb { \theta } , \beta ) - y \right\| ^ { 2 } + \frac { 1 } { n } \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) \left\| \psi _ { m } ( \pmb { \theta } ) \right\| ^ { 2 } - \frac { 1 } { n } \left\| \phi ( \pmb { \theta } , \beta ) \right\| ^ { 2 } .\tag{A21}
$$

Here, division by n makes the squared vector norms equivalent to empirical averages over S. Suppose there are q trainable parameters. The parameter count q and the number of noisy locations $g$ need not coincide. For example, in the four-qubit model studied here, $q = 7 2$ and $g = 1 2 6$ . Let

$$
J _ { m } = \nabla _ { \pmb { \theta } } \psi _ { m } ( \pmb { \theta } ) \in \mathbb { R } ^ { n \times q } , \qquad \boldsymbol { J } = \nabla _ { \pmb { \theta } } \phi ( \pmb { \theta } , \beta ) = \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } .\tag{A22}
$$

The first-order condition (FOC) of the surrogate loss is

$$
\nabla _ { \pmb { \theta } } \widetilde { \mathcal { L } } ( \pmb { \theta } , \beta ) = \frac { 2 } { n } \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } ^ { \top } ( \psi _ { m } - y ) = 0 .\tag{A23}
$$

This is equivalent to diferentiating the decomposed form, since

$$
J ^ { \top } ( \phi - y ) + \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } ^ { \top } \psi _ { m } - J ^ { \top } \phi = \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } ^ { \top } ( \psi _ { m } - y ) .\tag{A24}
$$

Now we diferentiate the FOC with respect to the label vector y. Under the local linear approximation, we assume that the Jacobians $J _ { m }$ are stable in a neighborhood of the optimum and ignore the second-order residual terms. Then

$$
\sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } ^ { \top } ( J _ { m } \mathrm { d } \theta - \mathrm { d } y ) = 0 .\tag{A25}
$$

Therefore,

$$
\mathrm { d } \pmb { \theta } = \left( \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } ^ { \top } J _ { m } \right) ^ { - 1 } J ^ { \top } \mathrm { d } y .\tag{A26}
$$

Since $\mathrm { d } \phi = J \mathrm { d } \theta$ , the local response of the fitted predictor to label perturbations is

$$
\mathrm { d } \phi = J \left( \sum _ { m = 0 } ^ { g } p _ { m } ( { \boldsymbol { \beta } } , g ) J _ { m } ^ { \top } J _ { m } \right) ^ { - 1 } J ^ { \top } \mathrm { d } y = : H ( \theta ; { \boldsymbol { \beta } } , g ) \mathrm { d } y .\tag{A27}
$$

The matrix $H ( \pmb \theta ; \beta , g )$ is the local hat matrix of the stochastic surrogate model. Unlike the ordinary least-squares hat matrix, it is generally not an exact projector; its trace plays the role of an efective number of fitted degrees of freedom.

Under the standard regression model with homoscedastic data noise $y = f + \epsilon , \mathbb { E } ( \epsilon ) = 0$ , and $\mathrm { C o v } ( \epsilon ) = \sigma ^ { 2 } I _ { n }$ , the expected generalization gap of a local linear smoother $[ 4 7 ]$ is approximated by

$$
\Delta _ { \mathrm { g e n } } : = \mathbb { E } ( \mathcal { L } _ { \mathrm { t e s t } } - \mathcal { L } _ { \mathrm { t r a i n } } ) \approx \frac { 2 \sigma ^ { 2 } } { n } \operatorname { T r } ( H ) = : \frac { 2 \sigma ^ { 2 } } { n } d _ { \mathrm { e f f } } .\tag{A28}
$$

Here, $d _ { \mathrm { e f f } } : = \operatorname { T r } ( H )$ is the efective model complexity of the stochastic surrogate model.

To connect this efective complexity to the noise occurrence distribution, we consider a simplified overlap model for the Jacobians. Suppose that the occurrence-count predictors have comparable local sensitivities and approximately uniform cross-overlap:

$$
J _ { m } ^ { \top } J _ { m } \approx c I _ { q } , \qquad J _ { m _ { 1 } } ^ { \top } J _ { m _ { 2 } } \approx c \kappa I _ { q } \quad ( m _ { 1 } \neq m _ { 2 } ) ,\tag{A29}
$$

where $c > 0$ controls the overall local sensitivity and $\kappa \in [ 0 , 1 ]$ measures the average alignment between diferent occurrence-count predictors. Then

$$
\sum _ { m = 0 } ^ { g } p _ { m } J _ { m } ^ { \top } J _ { m } \approx c I _ { q } .\tag{A30}
$$

Moreover,

$$
J ^ { \top } J = \sum _ { m _ { 1 } , m _ { 2 } = 0 } ^ { g } p _ { m _ { 1 } } p _ { m _ { 2 } } J _ { m _ { 1 } } ^ { \top } J _ { m _ { 2 } } \approx c \left( s _ { 2 } ( \beta , g ) + \kappa ( 1 - s _ { 2 } ( \beta , g ) ) \right) I _ { q } .\tag{A31}
$$

Therefore,

$$
d _ { \mathrm { e f f } } = \operatorname { T r } ( H ) \approx q \left( s _ { 2 } ( \beta , g ) + \kappa ( 1 - s _ { 2 } ( \beta , g ) ) \right) = q \left( \kappa + ( 1 - \kappa ) s _ { 2 } ( \beta , g ) \right) .\tag{A32}
$$

When diferent occurrence-count predictors are weakly aligned, $\kappa \approx 0 .$ , this reduces to

$$
d _ { \mathrm { e f f } } \approx q s _ { 2 } ( \beta , g ) .\tag{A33}
$$

Thus, when the noise occurrence-count distribution spreads over more values of $m , s _ { 2 } ( \beta , g )$ decreases and the stochastic surrogate model has smaller efective complexity. For a QML model with q trainable parameters, the above analysis shows that the efective model complexity of the stochastic surrogate model is reduced by a factor $s _ { 2 } ( \beta , g )$ . This provides a mechanism by which moderate noise can reduce the generalization gap. At the same time, stronger noise generally increases the training loss by attenuating the learnable signal and increasing the approximation bias. The competition between the decreasing generalization gap and the increasing training loss naturally leads to a non-monotonic testing loss. We will provide a more quantitative analysis in the next subsection.

## A.4. Scaling regimes of the noise-order purity

According to Eq. (A33), the efective model complexity is approximately reduced by the factor $s _ { 2 } ( \beta , g )$ . We refer to $s _ { 2 } ( \beta , g )$ as the noise-order purity:

$$
s _ { 2 } ( \beta , g ) : = \sum _ { m = 0 } ^ { g } p _ { m } ^ { 2 } ( \beta , g ) .\tag{A34}
$$

When $\beta = 0$ , the only possible noise order is the noiseless one, and hence $s _ { 2 } ( \beta , g ) = 1$ . As the noise strength increases, the probability distribution spreads over more noise orders, and $s _ { 2 } ( \beta , g )$ decreases. In our simplified model, the noise order follows a binomial distribution, which allows a quantitative analysis of $s _ { 2 } ( \beta , g )$

We first analyze two limiting regimes. In the weak-noise regime $g \beta \ll 1$ , the noise-order distribution is concentrated near zero. Therefore,

$$
s _ { 2 } ( \beta , g ) = \sum _ { m = 0 } ^ { g } p _ { m } ^ { 2 } ( \beta , g ) = p _ { 0 } ^ { 2 } ( \beta , g ) + \mathcal O ( g ^ { 2 } \beta ^ { 2 } ) = 1 - 2 g \beta + \mathcal O ( g ^ { 2 } \beta ^ { 2 } ) .\tag{A35}
$$

This shows that very weak noise only induces a perturbative reduction in the efective model complexity.

In the moderate-noise regime, where $g \beta \gg 1$ , the binomial distribution can be approximated by a Gaussian distribution with mean $\mu = g \beta$ and variance $\sigma ^ { 2 } = g \beta ( 1 - \beta )$ . Hence,

$$
s _ { 2 } ( \beta , g ) \approx \frac { 1 } { 2 \pi \sigma ^ { 2 } } \int _ { - \infty } ^ { \infty } e ^ { - ( x - \mu ) ^ { 2 } / \sigma ^ { 2 } } \mathrm { d } x = \frac { 1 } { 2 \sqrt { \pi } \sigma } = \frac { 1 } { 2 \sqrt { \pi g \beta ( 1 - \beta ) } } .\tag{A36}
$$

Thus, when the noise-order distribution is suficiently spread out, the noise-order purity decreases as $1 / \sqrt { g \beta ( 1 - \beta ) }$ .

The transition between these regimes can also be captured by a Poisson approximation with rate $g \beta .$ . Therefore,

$$
s _ { 2 } ( \beta , g ) \approx e ^ { - 2 g \beta } \sum _ { m = 0 } ^ { \infty } \frac { ( g \beta ) ^ { 2 m } } { ( m ! ) ^ { 2 } } = e ^ { - 2 g \beta } I _ { 0 } ( 2 g \beta ) ,\tag{A37}
$$

where $I _ { 0 }$ is the modified Bessel function of the first kind. This expression interpolates between the weak-noise expansion $s _ { 2 } ( \beta , g ) = 1 - 2 g \beta + \mathcal { O } ( g ^ { 2 } \beta ^ { 2 } )$ when $g \beta \ll 1$ , and the Gaussian scaling $s _ { 2 } ( \beta , g ) \approx ( 4 \pi g \beta ) \bar { - } 1 / 2$ when $g \beta \gg 1$ and $\beta \ll 1$

These scaling regimes have a direct implication for the efective complexity of the stochastic surrogate model. For simplicity, assume that the number of trainable parameters is $q = g .$ In the weak-noise regime, $s _ { 2 } ( \beta , g ) \approx 1$ , so the efective complexity remains $d _ { \mathrm { e f f } } \approx g$ . In contrast, in the moderate-noise regime,

$$
d _ { \mathrm { e f f } } \approx g s _ { 2 } ( \beta , g ) \approx \frac { \sqrt { g } } { 2 \sqrt { \pi \beta ( 1 - \beta ) } } .\tag{A38}
$$

Thus, noise can reduce the efective model complexity from $\mathcal O ( g )$ to $\mathcal { O } ( \sqrt { g } )$ when the noise-order distribution is suficiently spread out. Since the generalization gap of the stochastic surrogate model scales as $2 \sigma ^ { 2 } d _ { \mathrm { e f f } } / n$ , the corresponding reduction in the generalization gap is of order $\mathcal { O } ( 1 \bar { / } \sqrt { g } )$ . This reduction in efective complexity and generalization gap explains why moderate noise can improve testing performance.

## A.5. Mean-squared prediction error in the presence of quantum noise

In this subsection, we derive the approximate training and testing MSE in the presence of quantum noise. For simplicity, we write

$$
s : = s _ { 2 } ( \beta , g )\tag{A39}
$$

for the noise-order purity. We also use $\lVert \cdot \rVert _ { 2 }$ as the normalized empirical norm over the training inputs in this subsection.

Based on the local regression approximation in Eq. (A27), the surrogate model acts as a local smoother. Under the weak-overlap approximation of noise-order predictors, we have

$$
H ( \pmb \theta ; \beta , g ) \cong s P _ { T } ,\tag{A40}
$$

where $P _ { T }$ is the orthogonal projection onto the local learnable space $\tau _ { \ast }$ , and rank $( P \tau ) = q$ . We decompose the true signal as

$$
f = f _ { \parallel } + f _ { \perp } , \qquad f _ { \parallel } = P _ { T } f , \quad f _ { \perp } = ( I - P _ { T } ) f .\tag{A41}
$$

Here, $f _ { \parallel }$ is the learnable component in the local model space, and $f _ { \perp }$ is the component outside the local model space.

We assume homoscedastic data noise

$$
y = f + \epsilon , \qquad \mathbb { E } ( \epsilon ) = 0 , \qquad \mathbb { E } ( \epsilon \epsilon ^ { \top } ) = \sigma ^ { 2 } I _ { n } .\tag{A42}
$$

The training MSE is then approximated by

$$
\begin{array} { r l } & { \mathbb { E } \mathrm { M S E } _ { \mathrm { t r a i n } } ( \theta ; \beta , g ) = \mathbb { E } \left\| \phi ( \theta , \beta ) - y \right\| _ { 2 } ^ { 2 } \approx \mathbb { E } \left\| \big ( H ( \theta ; \beta , g ) - I \big ) y \right\| _ { 2 } ^ { 2 } } \\ & { \quad \quad \quad \approx \left\| f _ { \perp } \right\| _ { 2 } ^ { 2 } + ( 1 - s ) ^ { 2 } \left\| f _ { \parallel } \right\| _ { 2 } ^ { 2 } + \displaystyle \frac { \sigma ^ { 2 } } { n } ( n - 2 q s + q s ^ { 2 } ) . } \end{array}\tag{A43}
$$

The first term is the unlearnable component outside the local model space. The second term is the bias induced by noise. It increases when the noise-order purity s decreases. The third term is the contribution from data noise. This expression shows that the training MSE increases as the noise rate increases.

Using Eqs. (A28) and (A33), the generalization gap is

$$
\Delta _ { \mathrm { g e n } } \approx \frac { 2 \sigma ^ { 2 } q } { n } s .\tag{A44}
$$

The testing MSE is the sum of the training MSE and the generalization gap. Adding this expression to the training MSE gives

$$
\mathbb { E } \mathrm { M S E } _ { \mathrm { t e s t } } ( \pmb { \theta } ; \beta , g ) \approx \left\| f _ { \perp } \right\| _ { 2 } ^ { 2 } + ( 1 - s ) ^ { 2 } \left\| f _ { \parallel } \right\| _ { 2 } ^ { 2 } + \sigma ^ { 2 } + \frac { \sigma ^ { 2 } q } { n } s ^ { 2 } .\tag{A45}
$$

The generalization gap cancels the negative term $- 2 \sigma ^ { 2 } q s / n$ in the training MSE, which arises from fitting data noise in the training set. As the noise rate increases, $s = s _ { 2 } ( \beta , g )$ decreases. This reduces the generalization gap, but it also increases the bias term $( 1 - s ) ^ { 2 } \left\| f _ { \parallel } \right\| _ { 2 } ^ { 2 } .$ The competition between these two efects leads to a non-monotonic testing MSE.

The testing MSE is minimized at

$$
s ^ { \star } = \frac { \left\| f _ { \parallel } \right\| _ { 2 } ^ { 2 } } { \left\| f _ { \parallel } \right\| _ { 2 } ^ { 2 } + \sigma ^ { 2 } q / n } = \frac { \mathrm { S N R } } { { \mathrm { S N R } } + q / n } ,\tag{A46}
$$

where SNR $\vdots = \left\| f _ { \parallel } \right\| _ { 2 } ^ { 2 } / \sigma ^ { 2 }$ is the signal-to-noise ratio. The corresponding optimal noise rate $\beta ^ { \star }$ is determined by

$$
s _ { 2 } ( \beta ^ { \star } , g ) = s ^ { \star } ,\tag{A47}
$$

when such a solution is in the achievable range of $s _ { 2 } ( \beta , g )$ . This shows that noise is more likely to improve the testing MSE when the learnable signal is weak, the data noise is large, the locally learnable model dimension is large, or the training set is small.

As a remark, in the derivation above, we assume for simplicity that the model parameter count is smaller than the training data size. In this case, the rank of the projector $P _ { T }$ is simplified to $q .$ When the model is overparameterized, however, the rank of the projector is no longer simply given by the model parameter count, but is jointly upper bounded by the model parameter count and the training data size. In this case, this rank should be replaced by $d _ { \mathrm { e f f } } ^ { ( 0 ) }$ , the zero-noise efective model complexity. The corresponding form is

$$
s ^ { \star } \approx \frac { \mathrm { S N R } } { \mathrm { S N R } + d _ { \mathrm { e f f } } ^ { ( 0 ) } / n } .\tag{A48}
$$

## A.6. Extensions beyond the Bernoulli noise model

The analysis above uses the simple stochastic noise model in $\mathrm { E q . ~ ( A 6 ) }$ for clarity. In this model, each noisy location either applies the identity channel or applies a noisy operation. This gives a binomial expansion where the zero-noise branch corresponds to the noiseless implementation. However, this identity branch is not essential to the mechanism.

A direct extension is to replace the identity branch by a general principal channel. For example, suppose the channel at the $j \mathrm { - t h }$ noisy location is given by

$$
\mathcal { E } _ { j , z _ { j } } = ( 1 - z _ { j } ) A _ { j } + z _ { j } B _ { j } , \quad \mathbb { E } _ { z _ { j } } ( \mathcal { E } _ { j , z _ { j } } ) = ( 1 - \beta ) A _ { j } + \beta \mathcal { B } _ { j } ,\tag{A49}
$$

where $z _ { j } \sim \mathrm { B e r } ( \beta )$ , and $A _ { j }$ and $B _ { j }$ are two quantum channels. The same noise-order expansion still applies, but the principal branch is no longer the noiseless implementation. Instead, the principal branch corresponds to the gate sequence distorted by the channel sequence $A _ { j }$ . This difers from the simple identity-branch case, where the noiseless model hypothesis space is preserved. In this more general setting, the hypothesis space itself is distorted by the principal noise branch. Nevertheless, the same binomial weights $p _ { m } ( \beta , g )$ and the same noise-order purity $s _ { 2 } ( \beta , g )$ appear in the analysis, although the detailed predictors $\psi _ { m }$ are changed.

More general noise models may contain more than two branches. In that case, the noisy channel can be viewed as a mixture of several efective operations. A full noise realization is specified by a path of branch choices across the circuit. The binomial weights are then replaced by the corresponding path probabilities. We can similarly define a generalized noise-order purity by summing the squared probabilities of these paths. The same intuition remains: when the path probabilities spread over more efective noise realizations, the surrogate model has a stronger averaging efect; when the noise becomes too strong, the learnable signal can also be suppressed.

![](images/760b59eb68f55de38d7a3cfe2cc3fa98b259e8d5c0b49dce172b051728daaa57.jpg)

![](images/e58d9e7e6d1705be5bfb4025b2b1b7900393ee7a60feb19dfc4d5ca74518e76b.jpg)  
Depolarizing Bit flip Amplitude damping Phase damping � = �  
Figure 7. Efective model complexity of the noise-order surrogate. (a) Surrogate efective model complexity as a function of noise rate. Lines show the mean over ten runs and shaded regions show one standard deviation. (b) Efective model complexity normalized by its zero-noise value and plotted against $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ Error bars show one standard deviation; fitted values of $g _ { \mathrm { e f f } }$ are reported in the text.

## Appendix B: Numerical validation of the noise-order surrogate

## B.1. Surrogate efective complexity in trained models

Section A predicts that averaging over noise orders reduces the local efective complexity of the stochastic surrogate model. The main text tests this prediction using the Jacobian of the physical noisy predictor in Fig. 2. Here we evaluate the surrogate complexity directly from the order-resolved Jacobians $J _ { m }$ defined in Eq. (A22), providing a separate numerical test of the approximation leading to Eq. (A33). These Jacobians are evaluated eficiently using the dynamic-programming procedure in Section B.3.

For a fixed trained checkpoint and noise rate, let $\begin{array} { r } { J = \sum _ { m } p _ { m } ( \beta , g ) J _ { m } } \end{array}$ denote the Jacobian of the averaged noisy predictor and define

$$
A ( \beta ) = \frac 1 n \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) J _ { m } ^ { \top } J _ { m } , \qquad B ( \beta ) = \frac 1 n J ^ { \top } J .\tag{B1}
$$

The local hat matrix in Eq. (A27) then gives the numerical surrogate complexity

$$
d _ { \mathrm { e f f } } ^ { \mathrm { s u r } } ( \beta ) = \mathrm { T r } \left( ( A ( \beta ) + 1 0 ^ { - 4 } I _ { q } ) ^ { - 1 } B ( \beta ) \right) .\tag{B2}
$$

The matrix $A ( \beta )$ can be singular, so we use the same $1 0 ^ { - 4 }$ soft-counting threshold as in Eq. (14) to set the spectral resolution of its inverse. This threshold is applied only to the post-training model complexity evaluation and does not modify the training objective. For each run, we normalize $d _ { \mathrm { e f f } } ^ { \mathrm { s u r } } ( \beta )$ by its value at $\beta = 0$ before averaging over the ten trained runs.

The surrogate complexity decreases with noise for all four noise models in Fig. 7(a). After normalization, the noise-model-dependent curves are described by the same noise-order purity $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ in Fig. 7(b). The fitted efective counts are summarized in Table I. The surrogate loss explicitly penalizes disagreement among noise-order predictors, whereas the actual loss captures this regularization only implicitly through the structure of their averaged prediction. The same physical noise therefore induces stronger efective regularization in the surrogate, as reflected by its larger fitted $g _ { \mathrm { e f f } }$ . Thus, the surrogate calculation reproduces the same $s _ { 2 }$ scaling predicted by Eq. (A33) and observed in Fig. 2 for the physical-model complexity computed from the actual loss.

The surrogate makes the noise-order regularization explicit by retaining the variation among order-resolved Jacobians in its local curvature, whereas the physical-model complexity captures its consequence through the Jacobian of the averaged noisy predictor. Indeed,

$$
A ( \beta ) - B ( \beta ) = \frac { 1 } { n } \sum _ { m = 0 } ^ { g } p _ { m } ( \beta , g ) ( J _ { m } - J ) ^ { \top } ( J _ { m } - J ) \succeq 0 .\tag{B3}
$$

This matrix order inequality implies that the surrogate model complexity is no greater than the physical-model complexity. This confirms that the surrogate model makes the noise-order regularization explicit.

## B.2. Interpretation of the efective noise count

As shown in Table I, both calculations give the ordering

$$
g _ { \mathrm { e f f } } ^ { \mathrm { d e p o l a r i z i n g } } > g _ { \mathrm { e f f } } ^ { \mathrm { b i t ~ f i i p } } > g _ { \mathrm { e f f } } ^ { \mathrm { a m p l i t u d e } } > g _ { \mathrm { e f f } } ^ { \mathrm { p h a s e } } .\tag{B4}
$$

Table I. Fitted efective noise counts obtained from the ten-run mean normalized complexity curves in Figs. 2 and 7. Both calculations use the soft-counting threshold $1 0 ^ { - 4 }$
<table><tr><td></td><td>Depolarizing</td><td>Bit flip</td><td>Amplitude damping</td><td>Phase damping</td></tr><tr><td>Physical model</td><td>32</td><td>24</td><td>11</td><td>6</td></tr><tr><td>Surrogate</td><td>91</td><td>59</td><td>20</td><td>8</td></tr></table>

This ordering can be understood from how each noise model changes components of the state that can reach the final $Z ^ { \otimes 4 }$ measurement. Phase damping suppresses of-diagonal coherences but leaves computational-basis populations unchanged at the location where it acts. It afects the measured output only when subsequent gates convert the lost coherence into population diferences, so many physical phase-damping locations have a weak efect on the predictor Jacobian. This gives the smallest $g _ { \mathrm { e f f } }$

Amplitude damping also suppresses coherence, but it additionally transfers population from |1⟩ to |0⟩. It can therefore change a later $Z$ measurement more directly than phase damping. Bitflip noise exchanges the two computational-basis populations and reverses the sign of the local Z component, producing a still stronger response in the measured parity. Depolarizing noise contracts every traceless single-qubit Pauli component and therefore perturbs the broadest set of population and coherence directions that can contribute to the final prediction. It consequently gives the largest efective count.

Earlier errors can be rotated between population and coherence components by the remaining circuit, so their efects depend on the trained parameters, input states, and measured observable. The fitted $g _ { \mathrm { e f f } }$ summarizes the accumulated efect of each noise model on the measured predictor Jacobian.

Algorithm 1 Order-resolved state and Jacobian propagation   
Require: Input state $\rho _ { 0 } .$ , parameters $\theta ,$ and event maps $\mathcal { N } _ { \beta , \ell }$   
1: $\rho [ 0 ]  \rho _ { 0 }$ and $T [ 0 , j ] \gets 0$ for $j = 1 , \dotsc , q$   
2: Treat entries outside the current order range as zero

## B.3. Eficient evaluation of noisy order-resolved quantities by dynamic programming

At each of the g noisy locations, the physical channel at a fixed $\beta$ is written exactly as

$$
\mathcal { E } _ { \beta , \ell } = ( 1 - \beta ) \mathrm { I d } + \beta N _ { \beta , \ell } , \qquad N _ { \beta , \ell } ( \boldsymbol { \rho } ) = \frac { \mathcal { E } _ { \beta , \ell } ( \boldsymbol { \rho } ) - ( 1 - \beta ) \boldsymbol { \rho } } { \beta } .\tag{B5}
$$

When evaluating the order-resolved Jacobians in Section B.1, expanding this decomposition across $g$ noisy locations produces $2 ^ { g }$ event placements, with $\textstyle { \binom { g } { m } }$ placements at noise order $m .$ . Direct evaluation is therefore exponential in $g .$ We instead use dynamic programming to accumulate all placements of the same order in a single averaged state.

To derive the recurrence, let $S \subseteq \{ 1 , \dots , \ell \}$ denote the event locations among the first ℓ noisy locations, and let $\rho _ { S }$ be the state generated by that placement. We define the order-m state as the conditional average over all placements of that order,

$$
\rho _ { m } ^ { ( \ell ) } = \frac { 1 } { \binom { \ell } { m } } \sum _ { S \subseteq \{ 1 , \ldots , \ell \} } \rho _ { S } .\tag{B6}
$$

When location ℓ is added, an order-m placement either retains order $m$ from the first $\ell - 1$ locations or adds an event to an order- $\cdot ( m - 1 )$ placement. Their multiplicities are $\binom { \ell - 1 } { m }$ and $\binom { \ell - 1 } { m - 1 }$ respectively. A one-step analysis therefore gives the recurrence

$$
\rho _ { m } ^ { ( \ell ) } = \frac { \ell - m } { \ell } \rho _ { m } ^ { ( \ell - 1 ) } + \frac { m } { \ell } \mathcal { N } _ { \beta , \ell } \left( \rho _ { m - 1 } ^ { ( \ell - 1 ) } \right) .\tag{B7}
$$

Before this update, every retained sector is propagated through the circuit gates between locations $\ell - 1$ and ℓ. Since all unitary, channel, and residual maps are linear, averaging placements before subsequent propagation gives exactly the same result as propagating every placement separately and averaging at the end. Eq. (B7) therefore includes all $\hat { \binom { \ell } { m } }$ possible locations at a given order without assuming that diferent circuit locations are equivalent.

By linearity, after all g locations, the physical state and prediction are reconstructed as

$$
\rho ( { \boldsymbol { \beta } } ) = \sum _ { m = 0 } ^ { g } p _ { m } ( { \boldsymbol { \beta } } , g ) \rho _ { m } ^ { ( g ) } , \qquad \phi _ { \boldsymbol { \beta } } = \sum _ { m = 0 } ^ { g } p _ { m } ( { \boldsymbol { \beta } } , g ) \psi _ { m } , \qquad \psi _ { m } = \mathrm { T r } \left( O \rho _ { m } ^ { ( g ) } \right) .\tag{B8}
$$

No additional combinatorial factor appears in this expression: $\rho _ { m } ^ { ( g ) }$ is already the conditional average within the sector, while $p _ { m } ( \beta , g )$ is the total probability mass of that sector.

The order-resolved Jacobians are obtained in the same forward pass by propagating the tangents $T _ { m , j } ^ { ( \ell ) } = \partial \rho _ { m } ^ { ( \ell ) } / \partial \theta _ { j }$ alongside each state sector. Because the channel maps are linear, the tangents obey the same order recurrence in Eq. (B7); parametrized gates contribute through their analytic derivatives. Measuring the final tangents gives $J _ { m }$ , whose weighted sum recovers the full Jacobian J in Eq. (A22).

3: $\ell \gets 0$   
4: for each circuit gate $U$ in execution order do   
5: Propagate every ρ[m] and $T [ m , j ]$ through $U$   
6: if $U = U _ { j } ( \theta _ { j } )$ is parametrized then   
7: Add the analytic gate-derivative contribution to $T [ m , j ]$ for every m   
8: end if   
9: for each noisy location following U do   
10: $\ell \gets \ell + 1$ and copy the current arrays to $\rho _ { \mathrm { o l d } }$ and $T _ { \mathrm { o l d } }$   
11: for $m = 0 , \ldots , \ell$ do   
12: Update $\rho [ m ]$ from $\rho _ { \mathrm { o l d } } [ m ]$ and $\mathcal { N } _ { \beta , \ell } ( \rho _ { \mathrm { o l d } } [ m - 1 ] )$ using Eq. (B7)   
13: Apply the same recurrence to $T [ m , j ]$ for all j   
14: end for   
15: end for   
16: end for   
17: Measure $\rho [ m ]$ and $T [ m , j ]$ to obtain $\psi _ { m }$ and $J _ { m }$   
18: return $\{ \psi _ { m } , J _ { m } \} _ { m = 0 } ^ { g }$

Let n be the number of inputs, q the number of trainable parameters, and $d = 2 ^ { N _ { \mathrm { q } } }$ the Hilbertspace dimension. Retaining all noise orders requires $\mathcal { O } ( n q g ^ { 2 } \bar { d } ^ { 3 } )$ time because the algorithm propagates at most $g + 1$ sectors and their $q$ tangents through $g$ noisy locations. This is polynomial in $^ { g , }$ in contrast to the $\mathcal { O } ( 2 ^ { g } )$ cost of enumerating all noise placements. In numerical calculations, sectors with negligible binomial probability can be omitted to further reduce the computational cost.

## Appendix C: Molecular QML simulation details and ablation studies

## C.1. Circuit structure and other simulation details

The molecular models use one qubit for each heavy atom and one EDU-QGC-inspired graphcircuit layer [32, 60, 69]. The atom encoder applies trainable element-dependent rotations modulated by the atom degree, hydrogen count and aromaticity, together with a shared rotation that encodes hybridization. A trainable three-angle rotation is then applied to each node. For every molecular bond, the circuit applies a bond-type-dependent local basis change, a $\mathrm { C N O T } { - } R _ { Z } { \mathrm { - } } \mathrm { C N O T }$ interaction and a phase rotation, followed by the inverse basis change. Parameters in the bond block are shared across bonds of the same type.

The eight-qubit circuit contains 53 trainable quantum parameters, and the nine-qubit circuit contains 56. The diagonal of the final density matrix gives 256 and 512 computational-basis probabilities, respectively. A linear map from these probabilities to the molecular target adds 257 parameters for QM9-HA8 and 513 for PCQM4Mv2-HA9, giving 310 and 569 trainable parameters in the complete predictors.

A noisy location is assigned to each qubit immediately after an elementary gate acts on it. A single-qubit gate therefore contributes one location, while a two-qubit gate contributes two. In the four-qubit circuit, each layer contains two data-encoding locations, eight locations from the four two-qubit gates and four locations from the trainable single-qubit gates. The nine layers therefore give

$$
g = 9 ( 2 + 8 + 4 ) = 1 2 6 .\tag{C1}
$$

![](images/16348db1b5f7c53ae75cc3d130a36a92c4c5a943bc54b348842acd8bfd1a5c2f.jpg)  
Figure 8. Ablation of optimization epochs. Training MSE, testing MSE and generalization gap are evaluated after 100, 200 and 300 epochs for QM9-HA8 with (a–c) n = 80 and (d– ${ \mathrm { ~  ~ f ~ } } ) ~ n = 1 6 0$ training molecules. Lines show the mean over five fixed runs, and shaded regions show one sample standard deviation.

For the molecular models, the atom encoders and node rotations contribute locations on every atom, while the bond blocks contribute locations along the chemical edges. The resulting count $g ( G )$ therefore depends on the molecular graph G. The physical noisy simulation applies the channel at every executed location, while the fitted $g _ { \mathrm { e f f } }$ summarizes how rapidly the efective model complexity decreases with noise across the molecular dataset.

## C.2. Ablation of optimization epochs

We evaluate the same five QM9-HA8 training trajectories after 100, 200 and 300 epochs to determine whether the finite-noise response depends on the optimization endpoint. With longer training, the training error decreases while the testing error and generalization gap at low noise increase, indicating stronger overfitting $\left( \mathrm { F i g . 8 } \right)$ . The overall non-monotonic response remains stable across the three optimization endpoints.

## C.3. Ablation of more complex classical post-processing

The molecular QML model uses a trainable classical map from the computational-basis probabilities to the molecular target. To test whether the finite-noise improvement depends on the linear readout, we replace it with a fully connected hidden layer containing 128 tanh units while keeping the quantum circuit, training set and noise implementation fixed. This increases the total parameter count from 310 to 33,078. Both readouts fit the low-noise training data closely and retain a lower testing error at finite noise (Fig. 9(a–c)). The nonlinear readout produces a more variable response across the noise sweep but does not improve the minimum testing error. The finite-noise improvement is therefore not specific to the choice of classical post-processing, and additional classical capacity alone does not improve prediction.

![](images/667abc26d5a8d6fadea70258966bfd34d800262ca63626d2e2f9a0cf0c75b7cc.jpg)

![](images/8ee56104167d43c65989c9afdf14c01f4613fa525d3bc37418a49b5733e068a7.jpg)

![](images/54d6cff3ebb86f843b767c0aadb069194e1ac19f6da7681da2cb4957aef27821.jpg)

![](images/3dddb30ca6a30afb6a1c20f352fd7bd9ab5236a4d1c1726e53151c170a7b40a1.jpg)

![](images/0b471dd9c17c5f0ede3079251af84929a5d74887dbdbe4ad09f0ca578a5b72f7.jpg)  
Figure 9. Ablation of more complex classical post-processing. (a–c) Training MSE, testing MSE and generalization gap for a linear readout and a nonlinear readout with 128 tanh hidden units. (d,e) Absolute efective model complexities computed using the full parameter vector and the quantum-circuit parameters only, respectively. Both models use the same 53-parameter quantum circuit, n = 80 QM9-HA8 training molecules, bit-flip noise and 300-epoch endpoint. Lines and shaded regions show the mean and one standard deviation over five fixed runs.

This behavior is also reflected in the efective model complexity (Fig. 9(d,e)). Despite their very diferent classical parameter counts, the two models have comparable full-model and quantumonly complexities across the noise sweep. The reduction in model complexity is therefore governed primarily by the noisy quantum representation rather than the added classical capacity.

## C.4. Ablation of noise models

The main molecular comparison uses bit-flip and depolarizing noise, while the full comparison across noise models is reported here without crowding the main figure. We compare all four noise models for the same QM9-HA8 predictor, training set and optimization settings. Bit flip, depolarizing and amplitude damping produce a clear intermediate-noise reduction in testing error, whereas phase damping does not produce the same non-monotonic response (Fig. 10(a)). The corresponding training errors show that these minima occur before strong noise removes the ability to fit the training data (Fig. 10(b)).

All four noise models reduce the efective model complexity, but at diferent rates (Fig. 10(d)). This diference explains why their minima occur at diferent physical noise rates: the same value of $\beta$ does not produce the same reduction in model complexity across noise models. The fitted $g _ { \mathrm { e f f } }$ accounts for this mapping in Fig. 5(f), where the normalized complexity curves are compared against the noise-order purity $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ . Thus, the channel dependence changes how quickly the model moves through the finite-noise regime, while its efect on prediction also depends on the measured representation.

Under phase damping, the training error increases with noise, but the generalization gap does not decrease monotonically. The testing error therefore does not show the non-monotonic response observed for the other noise models. To test whether this behavior originates from the measurement basis, we repeat the phase-damping sweep using the same full-probability readout in either the Z or X basis (Fig. 11). Switching to the X basis recovers both the high-noise collapse and the linear relation between the generalization gap and efective model complexity, whereas the $Z _ { - }$ basis predictor retains a nonzero complexity and deviates from this relation. The complexity itself remains well described by the noise-order purity in both bases. Phase damping removes coherence in the Z basis, which directly afects an X-basis measurement but has a weaker efect on a $Z -$ basis measurement. The benefit of noise-induced regularization therefore depends not only on the noise model, but also on the measured representation.

[1] John Preskill. Quantum Computing in the NISQ Era and Beyond. Quantum, 2:79, 2018.

[2] M. Cerezo, Andrew Arrasmith, Ryan Babbush, Simon C. Benjamin, Suguru Endo, Keisuke Fujii, Jarrod R. McClean, Kosuke Mitarai, Xiao Yuan, Lukasz Cincio, and Patrick J. Coles. Variational Quantum Algorithms. Nat. Rev. Phys., 3(9):625–644, 2021.

[3] Kishor Bharti, Alba Cervera-Lierta, Thi Ha Kyaw, Tobias Haug, Sumner Alperin-Lea, Abhinav Anand, Matthias Degroote, Hermanni Heimonen, Jakob S. Kottmann, Tim Menke, Wai-Keong Mok, Sukin Sim, Leong-Chuan Kwek, and Al´an Aspuru-Guzik. Noisy Intermediate-Scale Quantum Algorithms. Rev. Mod. Phys., 94(1):015004, 2022.

[4] Kristan Temme, Sergey Bravyi, and Jay M. Gambetta. Error Mitigation for Short-Depth Quantum Circuits. Phys. Rev. Lett., 119(18):180509, 2017.

[5] Suguru Endo, Zhenyu Cai, Simon C. Benjamin, and Xiao Yuan. Hybrid Quantum-Classical Algorithms and Quantum Error Mitigation. Journal of the Physical Society of Japan, 90(3):032001, 2021.

[6] Zhenyu Cai, Ryan Babbush, Simon C. Benjamin, Suguru Endo, William J. Huggins, Ying Li, Jarrod R. McClean, and Thomas E. O’Brien. Quantum Error Mitigation. Rev. Mod. Phys., 95(4):045005, 2023.

[7] Daniel Stilck Fran¸ca and Raul Garc´ıa-Patr´on. Limitations of Optimization Algorithms on Noisy Quantum Devices. Nat. $P h y s .$ , 17(11):1221–1227, 2021.

[8] Kento Tsubouchi, Takahiro Sagawa, and Nobuyuki Yoshioka. Universal Cost Bound of Quantum Error Mitigation Based on Quantum Estimation Theory. Phys. Rev. Lett., 131(21):210601, 2023.

[9] Ryuji Takagi, Hiroyasu Tajima, and Mile Gu. Universal Sampling Lower Bounds for Quantum Error Mitigation. Phys. Rev. Lett., 131(21):210602, 2023.

[10] Yihui Quek, Daniel Stilck Fran¸ca, Sumeet Khatri, Johannes Jakob Meyer, and Jens Eisert. Exponentially Tighter Bounds on Limitations of Quantum Error Mitigation. Nat. Phys., 20(10):1648–1658, 2024.

[11] Yuxuan Du, Min-Hsiu Hsieh, Tongliang Liu, Shan You, and Dacheng Tao. Learnability of Quantum

![](images/fdad6aa252def6908f0616e329d9bd4ad78ba105aeabe88d3873147c2a85a4b6.jpg)  
Figure 10. Ablation of noise models in the molecular QML model. (a) Testing MSE, (b) training MSE, (c) generalization gap and (d) efective model complexity normalized by its zero-noise value for bit-flip, depolarizing, amplitude-damping and phase-damping noise. The damping channels are shown over $0 \leq \beta \leq$ 1, while bit-flip and depolarizing noise are shown over $0 \leq \beta \leq 0 . 5 .$ . All models use the same QM9-HA8 training set with $n = 1 6 0$ and are evaluated after 300 epochs. Lines and shaded regions show the mean and one standard deviation over five fixed runs.

Neural Networks. PRX Quantum, 2(4):040337, 2021.

[12] Samson Wang, Enrico Fontana, M. Cerezo, Kunal Sharma, Akira Sone, Lukasz Cincio, and Patrick J. Coles. Noise-Induced Barren Plateaus in Variational Quantum Algorithms. Nat. Commun., 12(1):6961, 2021.

[13] Antonio Anna Mele, Armando Angrisani, Soumik Ghosh, Sumeet Khatri, Jens Eisert, Daniel Stilck Fran¸ca, and Yihui Quek. Noise-Induced Shallow Circuits and the Absence of Barren Plateaus. Nat. Phys., 22(5):751–756, 2026.

[14] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost, Nathan Wiebe, and Seth Lloyd. Quantum Machine Learning. Nature, 549(7671):195–202, 2017.

[15] M. Cerezo, Guillaume Verdon, Hsin-Yuan Huang, Lukasz Cincio, and Patrick J. Coles. Challenges and Opportunities in Quantum Machine Learning. Nature Computational Science, 2(9):567–576, 2022.

[16] Amira Abbas, David Sutter, Christa Zoufal, Aurelien Lucchi, Alessio Figalli, and Stefan Woerner. The

![](images/f82f3b82e14d9dfb84b9da9a2d0c4a0c60af4bfa513df689b3122bd365f5612a.jpg)

![](images/87a2ff45d997a9c7ee994a01b8d46b5c94fa0eac833972aa8dc58b851037b394.jpg)

![](images/478769939b4e540f1bab6ede4d3643dee1c748870e1421486c6addac3e9d035a.jpg)

![](images/c8f500338d11c2a007a0ec7c9c3c8e2e51b9c212d9dca52f7860e39972368703.jpg)

$$
\Delta _ { \mathsf { g e n } }
$$

![](images/10b76a032750583ba7638fb1cccd713ee41df283407edd46b0ca731bf2b73d6d.jpg)

![](images/c0aa04cb27778176c11c17175b2116d3fa983713f5120ed5bd759f5570d17524.jpg)  
Figure 11. Measurement dependence under phase damping. (a) Training MSE, (b) testing MSE, (c) generalization gap and (d) efective model complexity normalized by its zero-noise value for full-probability measurements in the $Z$ and X bases. (e) Normalized generalization gap versus normalized efective model complexity. (f) Normalized efective model complexity versus $s _ { 2 } ( \beta , g _ { \mathrm { e f f } } )$ , with $g _ { \mathrm { e f f } }$ fitted separately for the two bases. Green circles and orange squares denote the Z- and X-basis measurements, respectively. Both models use the same QM9-HA8 training set with $n = 1 6 0$ and are evaluated after 300 epochs. Lines and shaded regions in (a–d), and points and error bars in (e,f), show the mean and one standard deviation over five fixed runs.

Power of Quantum Neural Networks. Nature Computational Science, 1(6):403–409, 2021.

[17] Matthias C. Caro, Hsin-Yuan Huang, M. Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, and Patrick J. Coles. Generalization in Quantum Machine Learning from Few Training Data. Nat. Commun., 13(1):4919, 2022.

[18] Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R. McClean. Power of Data in Quantum Machine Learning. Nat. Commun., 12(1):2631, 2021.

[19] Evan Peters and Maria Schuld. Generalization Despite Overfitting in Quantum Machine Learning Models. Quantum, 7:1210, 2023.

[20] Kaifeng Bu, Dax Enshan Koh, Lu Li, Qingxian Luo, and Yaobo Zhang. Statistical Complexity of Quantum Circuits. Phys. Rev. A, 105(6):062431, 2022.

[21] Kaifeng Bu, Dax Enshan Koh, Lu Li, Qingxian Luo, and Yaobo Zhang. Efects of Quantum Resources and Noise on the Statistical Complexity of Quantum Circuits. Quantum Sci. Technol., 8(2):025013, 2023.

[22] Valentin Heyraud, Zejian Li, Zakari Denis, Alexandre Le Boit´e, and Cristiano Ciuti. Noisy Quantum Kernel Machines. Phys. Rev. A, 106(5):052421, 2022.

[23] Chris M. Bishop. Training with Noise Is Equivalent to Tikhonov Regularization. Neural Computation,

7(1):108–116, 1995.

[24] Guozhong An. The Efects of Adding Noise During Backpropagation Training on a Generalization Performance. Neural Computation, 8(3):643–674, 1996.

[25] Nitish Srivastava, Geofrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(56):1929–1958, 2014.

[26] Yuxuan Du, Min-Hsiu Hsieh, Tongliang Liu, Dacheng Tao, and Nana Liu. Quantum Noise Protects Quantum Classifiers Against Adversaries. Phys. Rev. Res., 3(2):023153, 2021.

[27] Laia Domingo, G. Carlo, and Florentino Borondo. Taking Advantage of Noise in Quantum Reservoir Computing. Sci. Rep., 13(1):8790, 2023.

[28] Joris Kattem¨olle and Guido Burkard. Ability of Error Correlations to Improve the Performance of Variational Quantum Algorithms. Phys. Rev. A, 107(4):042426, 2023.

[29] Junyu Liu, Frederik Wilde, Antonio Anna Mele, Xin Jin, Liang Jiang, and Jens Eisert. Stochastic Noise Can Be Helpful for Variational Quantum Algorithms. Phys. Rev. A, 111(5):052441, 2025.

[30] Antonio Sannia, Francesco Tacchino, Ivano Tavernelli, Gian Luca Giorgi, and Roberta Zambrini. Engineered Dissipation to Mitigate Barren Plateaus. npj Quantum Information, 10(1):81, 2024.

[31] Viacheslav Kuzmin, Wilfrid Somogyi, Ekaterina Pankovets, and Alexey Melnikov. Method for Noise-Induced Regularization in Quantum Neural Networks. Adv. Quantum Technol., 8(12):e00603, 2025.

[32] Linghua Zhu, Ziyu Zhang, Yulong Dong, and Xiaosong Li. Rethinking Quantum Noise in Quantum Machine Learning: When Noise Improves Learning. arXiv preprint arXiv:2601.13275, 2026.

[33] Nam H. Nguyen, Elizabeth C. Behrman, and James E. Steck. Quantum Learning with Noise and Decoherence: A Robust Quantum Neural Network. Quantum Machine Intelligence, 2(1):1, 2020.

[34] Kerstin Beer, Dmytro Bondarenko, Terry Farrelly, Tobias J. Osborne, Robert Salzmann, Daniel Scheiermann, and Ramona Wolf. Training Deep Quantum Neural Networks. Nat. Commun., 11(1):808, 2020.

[35] Francesco Scala, Giacomo Guarnieri, and Aurelien Lucchi. Noise-Induced Equalization in Quantum Learning Models. Quantum Science and Technology, 11(4):045015, 2026.

[36] Matthias C. Caro and Ishaun Datta. Pseudo-Dimension of Quantum Circuits. Quantum Machine Intelligence, 2(2):14, 2020.

[37] Casper Gyurik, Dyon van Vreumingen, and Vedran Dunjko. Structural Risk Minimization for Quantum Linear Classifiers. Quantum, 7:893, 2023.

[38] Matthias C. Caro, Hsin-Yuan Huang, Nicholas Ezzell, Joe Gibbs, Andrew T. Sornborger, Lukasz Cincio, Patrick J. Coles, and Zo¨e Holmes. Out-of-Distribution Generalization for Learning Quantum Dynamics. Nat. Commun., 14(1):3751, 2023.

[39] Elies Gil-Fuster, Jens Eisert, and Carlos Bravo-Prieto. Understanding Quantum Machine Learning Also Requires Rethinking Generalization. Nat. Commun., 15(1):2277, 2024.

[40] Mart´ın Larocca, Supanut Thanasilp, Samson Wang, Kunal Sharma, Jacob Biamonte, Patrick J. Coles, Lukasz Cincio, Jarrod R. McClean, Zo¨e Holmes, and M. Cerezo. Barren Plateaus in Variational Quantum Computing. Nat. Rev. Phys., 7(4):174–189, 2025.

[41] Johannes Jakob Meyer. Fisher Information in Noisy Intermediate-Scale Quantum Applications. Quantum, 5:539, 2021.

[42] Tobias Haug and M. S. Kim. Generalization of Quantum Machine Learning Models Using Quantum Fisher Information Metric. Phys. Rev. Lett., 133(5):050603, 2024.

[43] Bradley Efron, Trevor Hastie, Iain Johnstone, and Robert Tibshirani. Least Angle Regression. The Annals of Statistics, 32(2):407–499, 2004.

[44] scikit-learn developers. scikit-learn Diabetes Dataset Documentation. https://scikit-learn.org/ stable/datasets/toy\_dataset.html#diabetes-dataset, 2026.

[45] Leonardo Banchi, Jason Pereira, and Stefano Pirandola. Generalization in Quantum Machine Learning: A Quantum Information Standpoint. PRX Quantum, 2(4):040321, 2021.

[46] Matthias C. Caro, Elies Gil-Fuster, Johannes Jakob Meyer, Jens Eisert, and Ryan Sweke. Encoding-Dependent Generalization Bounds for Parametrized Quantum Circuits. Quantum, 5:582, 2021.

[47] Bradley Efron. The Estimation of Prediction Error: Covariance Penalties and Cross-Validation. Journal of the American Statistical Association, 99(467):619–632, 2004.

[48] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven. Barren

Plateaus in Quantum Neural Network Training Landscapes. Nat. Commun., 9(1):4812, 2018.

[49] M. Cerezo, Akira Sone, Tyler Volkof, Lukasz Cincio, and Patrick J. Coles. Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits. Nat. Commun., 12(1):1791, 2021.

[50] Kunal Sharma, M. Cerezo, Lukasz Cincio, and Patrick J. Coles. Trainability of Dissipative Perceptron-Based Quantum Neural Networks. Phys. Rev. Lett., 128(18):180505, 2022.

[51] Supanut Thanasilp, Samson Wang, Nhat Anh Nghiem, Patrick Coles, and Marco Cerezo. Subtleties in the Trainability of Quantum Machine Learning Models. Quantum Machine Intelligence, 5(1):21, 2023.

[52] Michael Ragone, Bojko N. Bakalov, Fr´ed´eric Sauvage, Alexander F. Kemper, Carlos Ortiz Marrero, Mart´ın Larocca, and M. Cerezo. A Lie Algebraic Theory of Barren Plateaus for Deep Parameterized Quantum Circuits. Nat. Commun., 15(1):7172, 2024.

[53] Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, and Marco Pistoia. Characterizing Barren Plateaus in Quantum Ans¨atze with the Adjoint Representation. Nat. Commun., 15(1):7171, 2024.

[54] Arvind Neelakantan, Luke Vilnis, Quoc V. Le, Ilya Sutskever, Lukasz Kaiser, Karol Kurach, and James Martens. Adding Gradient Noise Improves Learning for Very Deep Networks. arXiv preprint arXiv:1511.06807, 2015.

[55] Stephan Mandt, Matthew D. Hofman, and David M. Blei. Stochastic Gradient Descent as Approximate Bayesian Inference. Journal of Machine Learning Research, 18(134):1–35, 2017.

[56] Maria Schuld and Nathan Killoran. Quantum Machine Learning in Feature Hilbert Spaces. Phys. Rev. Lett., 122(4):040504, 2019.

[57] Vojtˇech Havl´ıˇcek, Antonio D. C´orcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. Supervised Learning with Quantum-Enhanced Feature Spaces. Nature, 567(7747):209–212, 2019.

[58] Sofiene Jerbi, Lukas J. Fiderer, Hendrik Poulsen Nautrup, Jonas M. K¨ubler, Hans J. Briegel, and Vedran Dunjko. Quantum Machine Learning Beyond Kernel Methods. Nat. Commun., 14(1):517, 2023.

[59] Manas Sajjan, Junxu Li, Raja Selvarajan, Shree Hari Sureshbabu, Sumit Suresh Kale, Rishabh Gupta, Vinit Singh, and Sabre Kais. Quantum Machine Learning for Chemistry and Physics. Chemical Society Reviews, 51(15):6475–6573, 2022.

[60] Ju-Young Ryu, Eyuel Elala, and June-Koo Kevin Rhee. Quantum Graph Neural Network Models for Materials Search. Materials, 16(12):4300, 2023.

[61] Min Lu, Lei Du, Ziwei Cui, Yiming Zhao, Qipeng Yan, Jianyu Zhao, Ye Li, Menghan Dou, Qingchun Wang, Yu-Chun Wu, and Guo-Ping Guo. Quantum-Embedded Graph Neural Network Architecture for Molecular Property Prediction. J. Chem. Inf. Model., 65(15):8057–8065, 2025.

[62] Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Quantum Chemistry Structures and Properties of 134 Kilo Molecules. Sci. Data, 1(1):140022, 2014.

[63] Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs. In Joaquin Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

[64] Maho Nakata and Tomomi Shimazaki. PubChemQC Project: A Large-Scale First-Principles Electronic Structure Database for Data-Driven Chemistry. J. Chem. Inf. Model., 57(6):1300–1308, 2017.

[65] Mart´ın Larocca, Nathan Ju, Diego Garc´ıa-Mart´ın, Patrick J. Coles, and Marco Cerezo. Theory of Overparametrization in Quantum Neural Networks. Nature Computational Science, 3(6):542–551, 2023.

[66] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding Deep Learning Requires Rethinking Generalization. In International Conference on Learning Representations, 2017.

[67] Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Of. Proc. Natl. Acad. Sci. U.S.A., 116(32):15849– 15854, 2019.

[68] Guillaume Verdon, Trevor McCourt, Enxhell Luzhnica, Vikash Singh, Stefan Leichenauer, and Jack Hidary. Quantum Graph Neural Networks. arXiv preprint arXiv:1909.12264, 2019.

[69] P´eter Mernyei, Konstantinos Meichanetzidis, and <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan. Equivariant Quantum Graph

Circuits: Constructions for Universal Approximation over Graphs. Quantum Machine Intelligence, 5:6, 2023.

[70] Michael A. Nielsen and Isaac L. Chuang. Quantum Computation and Quantum Information. Cambridge University Press, 2002.

[71] Ville Bergholm et al. PennyLane: Automatic Diferentiation of Hybrid Quantum-Classical Computations. arXiv preprint arXiv:1811.04968, 2018.

[72] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980, 2014.

[73] Jianming Ye. On Measuring and Correcting the Efects of Data Mining and Model Selection. Journal of the American Statistical Association, 93(441):120–131, 1998.