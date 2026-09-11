# Predicting Privacy Leakage from Weight Spectral Density

Richard J. Preen Department of Computer Science and Creative Technologies University of the West of England Bristol, UK BS16 1QY richard2.preen@uwe.ac.uk

Jim Smith Department of Computer Science and Creative Technologies University of the West of England Bristol, UK BS16 1QY james.smith@uwe.ac.uk

## Abstract

Membership inference attacks (MIAs) are widely used to audit the privacy disclosure risk of machine learning models, however current state-of-the-art attacks require training computationally expensive shadow models, making large-scale privacy evaluation impractical. In this work, we investigate whether inexpensive spectral metrics derived from the heavy-tailed self-regularisation framework can serve as proxies for MIA vulnerability. We evaluate several WeightWatcher spectral metrics on image and tabular classification tasks and compare their relationship with MIA privacy leakage against conventional measures of generalisation. Across datasets, stable rank exhibits a strong positive correlation with overall MIA success, while Log α-Norm shows a consistent negative correlation with MIA vulnerability at the low false-positive regime. These associations are observed to be stronger than those obtained using the generalisation gap. The results indicate that neural network spectra may contain information about privacy leakage that is not fully captured by conventional measures of overfitting, motivating spectral analysis as a promising direction for scalable privacy auditing.

## 1 Introduction

Machine learning models are increasingly trained on sensitive data held within trusted research environments (TREs)<sup>1</sup>. Before such a model can be released from a TRE, its owners are required to demonstrate that it does not disclose sensitive information about the individuals in its training set [Jefferson et al., 2022]. Membership inference attacks (MIAs) [Shokri et al., 2017] have become one of the standard methods used to quantify the overall risk as part of the auditing process. An MIA attempts to determine whether a specific record was used to train a target model, and tools such as SACRO-ML [Smith et al., 2025] now package them for use directly within TRE disclosure-control workflows.

The most powerful of the current state-of-the-art MIAs, such as the likelihood ratio attack (LiRA) [Carlini et al., 2022] and the robust membership inference attack (RMIA) [Zarifzadeh et al., 2024], typically rely on training reference (shadow) models to approximate the target model’s behaviour.

This makes them computationally expensive, and in many settings prohibitively so. The problem is compounded in TRE auditing contexts where many candidate models may need to be screened before release.

A parallel line of research has studied the phenomenon of grokking [Power et al., 2022], wherein models undergo a delayed transition from memorisation to generalisation long after training loss has converged. This observation raises a broader question: what internal properties of a trained neural network’s weight structure distinguish models that memorise from those that generalise, and can such properties be measured cheaply, without access to the data used to fit the model?

WeightWatcher (WW) [Martin and Mahoney, 2021, Martin et al., 2021, Prakash and Martin, 2025] offers a promising lens through which to examine this question. Grounded in the theory of heavytailed self-regularisation (HT-SR), WW characterises the empirical spectral density (ESD) of a model’s weight matrices from the trained weights alone, making them cheap-to-compute. These metrics have been linked empirically to a model’s degree of implicit self-regularisation and, in turn, to its generalisation performance.

This raises the question: if WW metrics reflect the degree to which a model has generalised versus memorised its training data, might they also serve as proxies for privacy leakage? Intuitively, a model whose weight matrices remain close to random has not organised its parameters around training examples, whereas a model exhibiting strong power-law structure may have done so in ways that are recoverable by an attacker. Yet the relationship between spectral properties and susceptibility to MIA is far from obvious, and no prior work has investigated it empirically.

In this paper, we investigate whether WW metrics correlate with generalisation gap and vulnerability to MIAs such as LiRA. If such correlations exist, WW could provide a computationally tractable, data-free method for estimating privacy risk, becoming a powerful tool for practitioners who cannot afford the overhead of shadow model pipelines.

In particular, this paper makes the following contributions:

• We conduct the first empirical study of the relationship between neural network weight spectral metrics and vulnerability to MIA.

• We show that WW stable rank is strongly associated with overall MIA success, and that this association is consistently stronger than that of the generalisation gap.

• We show that WW Log α-Norm is consistently associated with MIA success in the low false-positive-rate regime, where the generalisation gap is a weak and inconsistent predictor.

• We show that spectral metrics and the generalisation gap capture complementary information about privacy leakage, and that combining them improves prediction of MIA vulnerability over either signal alone.

• We demonstrate that spectral metrics can be used to identify high-risk models as a binary classification task, indicating practical utility for scalable, data-free privacy auditing.

## 2 Background

## 2.1 Heavy-Tailed Self-Regularisation and WeightWatcher

WW is an open source<sup>2</sup> diagnostic tool for analysing trained deep neural networks without requiring access to training or test data [Martin and Mahoney, 2021, Martin et al., 2021]. It is based on the theory of HT-SR, which analyses the ESD of the correlation matrix associated with each layer’s weight matrix. The central finding of HT-SR theory is that well-trained, generalisable networks are not random: rather than resembling the spectra of random matrices, their layer-wise ESDs display structured, self-organised correlations that emerge during optimisation, in many cases converging to heavy-tailed, power-law form [Martin and Mahoney, 2021]. This behaviour is interpreted as a signature of implicit self-regularisation, analogous to the strong correlations observed in selforganising physical systems, and has been proposed as an explanation for why overparameterised networks generalise well in practice.

Martin and Mahoney [2021] characterise this process in terms of six qualitative spectral phases through which a layer can progress as training proceeds: (i) random-like, in which the weights remain statistically indistinguishable from a random matrix; (ii) bleeding-out, in which small deviations from randomness first appear; (iii) bulk+spikes, in which discrete signal components emerge above an otherwise random bulk; (iv) bulk-decay, in which the bulk itself begins to decay as correlations strengthen; (v) heavy-tailed, in which scale-free correlations dominate the spectrum and self-regularisation is strong; and (vi) rank-collapse, in which over-regularisation causes most of the layer’s effective information to be lost. Hyperparameters that govern the implicit regularisation strength of stochastic gradient descent (SGD), most notably batch size, have been shown to move models through these phases, with smaller batch sizes typically producing stronger implicit self-regularisation.

To characterise where a layer sits along this progression, WW fits the tail of each layer’s ESD to a (truncated) power law and reports a small set of summary statistics. The power-law exponent α describes the shape of this tail; under HT-SR theory, models with α in the range [2, 4] are considered well regularised, whereas $\alpha < 2$ is associated with over-fitting and rank collapse and $\alpha > 4$ with an under-trained, near-random layer. Norm-based metrics such as the log α-norm and the log spectral norm combine this shape information with the scale of the spectrum, making them more directly comparable across layers and architectures of different width and depth. A further metric, stable rank, measures the effective dimensionality of a weight matrix and is a noise-tolerant alternative to the rank of the weight matrix. Because all of these quantities are computed directly from trained weights, WW metrics have been proposed as scalable proxies for generalisation performance that require no forward passes over data [Martin et al., 2021, Prakash and Martin, 2025], motivating our investigation of whether they might similarly serve as proxies for a model’s susceptibility to membership inference.

## 2.2 Membership Inference Attacks

MIAs aim to determine whether a specific record was used to train a target model, and have become a standard tool for empirically auditing privacy disclosure [Shokri et al., 2017]. Broadly, existing attacks trade off attack strength against computational cost: the strongest attacks require training many reference (shadow) models of similar architecture and under the same conditions as the target, while a growing body of work seeks to approximate this signal more cheaply.

## 2.2.1 Reference-model attacks

LiRA [Carlini et al., 2022] is widely regarded as the closest practical approximation to a record’s ground-truth vulnerability. LiRA trains multiple shadow models both with and without the target record, fits Gaussian models to the resulting IN and OUT loss distributions, and performs a likelihood ratio test to infer membership. Because it directly approximates the leave-one-out counterfactual behaviour of the target model, LiRA achieves strong, well-calibrated performance across architectures and datasets, particularly on rare or outlier records. Its principal drawback is cost: obtaining reliable estimates typically requires dozens of shadow models trained under matched conditions, which is prohibitive for large modern architectures or datasets.

Subsequent work has sought to reduce this overhead while retaining LiRA’s discriminative power. RMIA [Zarifzadeh et al., 2024] reformulates membership inference as a pairwise test in which a target sample is compared against many population samples, using a small number of reference models together with a large pool of unlabelled population data to normalise the resulting likelihood ratio; this substantially reduces the number of reference models required relative to LiRA while achieving comparable power. Galichin et al. [2025] instead reduce the cost of training the shadow models themselves by incorporating knowledge distillation from the target model, aligning shadow model behaviour with the target more closely and reporting improvements over LiRA and related loss-trajectory attacks on image classification benchmarks, though training remains the dominant cost. Across these variants, the requirement to retrain one or more models under conditions approximating the target remains the primary obstacle to using reference-model attacks for routine, large-scale privacy auditing.

## 2.2.2 Reference-free and one-shot estimators

A parallel line of work avoids reference-model training entirely, instead exploiting signals available from the target model alone. White-box attacks that inspect gradients or intermediate activations have shown that even well-generalised models can leak substantial membership information: Nasr et al. [2019] show that gradients, particularly from later layers, are more informative than either activations or outputs, and that this leakage is not well predicted by a model’s generalisation gap. Li et al. [2024] similarly use statistical neuron-selection methods together with a single shadow model to identify which neurons carry the most membership-relevant signal, while related work on interpretability-based attacks shows that feature-importance information can itself be exploited to distinguish members from non-members [Liu et al., 2024].

Fully black-box, shadow-free alternatives have also been proposed: Liu et al. [2023] infer membership from the Jacobian norm of the target model’s predictions, clustering samples by sensitivity, and can operate from as little as a single query record. One-shot, label-only attacks exploit the relative robustness of member samples to adversarial perturbation [Peng et al., 2024], while the GLiR attack of Leemann et al. [2023] performs one-shot auditing from training-time gradient statistics without any shadow models, and Suri et al. [2024] show that white-box access to inverse-Hessian information can outperform black-box loss-based attacks under SGD, albeit at considerable computational cost and with access to the full training set. Steinke et al. [2023] and Andrew et al. [2024] take a complementary approach, deriving privacy-auditing estimates from a single training run by treating a random subset of the training data as an implicit control group, avoiding the need to train separate reference models altogether.

Other estimators dispense with reference models entirely, avoiding the cost of retraining, though most still rely on some form of auxiliary data or held-out computation. QMIA [Bertran et al., 2023] trains a quantile regression model on held-out data to estimate a per-record confidence threshold as a function of the input itself, conditioning thresholds on the local difficulty of a record rather than using a single global threshold as in earlier loss-based attacks [Yeom et al., 2018]; a related approach has been used to detect training-set membership for documents used to train large language models [Zhang et al., 2024]. QMIA requires no reference models and scales well, but may perform poorly in sparse regions of the input distribution where few comparable records are available for quantile estimation, precisely where disclosure risk is often greatest.

Training-dynamics estimators such as LT-IQR [Pollock et al., 2025] instead exploit the trajectory of a record’s loss over training, treating records whose loss falls unusually far as more likely to have been memorised; such methods can provide useful relative rankings when full training trajectories are retained, but typically stop short of calibrated disclosure probabilities. Most closely related to our approach, Dodd et al. [2025] propose a model-level, reference-free estimator based on the observation that memorisation suppresses the heavy-loss tail of the training-loss distribution, and estimate aggregate disclosure from the separation between training and test-loss tails without any additional training. Like WW metrics, this approach requires no reference models and is computationally cheap, but provides only an aggregate, model-level signal rather than per-record vulnerability estimates.

## 2.2.3 Cheap-to-compute privacy proxy metrics

Because MIAs succeed by exploiting differences in a model’s behaviour on training versus held-out data, a model’s generalisation gap has long been used as an inexpensive, if coarse, proxy for its privacy risk [Yeom et al., 2018, Shokri et al., 2017]. Larger gaps between training and test performance are generally associated with greater vulnerability to MIAs, and the generalisation gap requires no additional model training beyond the target model itself. However, this relationship is not always reliable: Nasr et al. [2019] find that generalisation error is a poor predictor of privacy risk in white-box settings, since large, expressive models may memorise individual training records in ways that are not reflected in aggregate held-out performance. This gap between an easily-measured, coarse-grained statistic and a model’s true susceptibility to attack is a central motivation for identifying alternative, equally inexpensive metrics that more directly track privacy leakage.

Drawing on traditional statistical disclosure control concepts, including degrees of freedom and k-anonymity, Preen and Smith [2026] investigate the use of cheap-to-compute model-level metrics to reduce the computational cost of MIA assessment. They show that for tree-based classification models, structural measures derived from trained models and their predictions provide high-precision but low-recall indicators of MIA vulnerability. This enables high risk models to be identified without needing to perform expensive reference model-based MIA assessment. These results demonstrate the potential for inexpensive metrics to complement more intensive MIA assessment, reducing the overall computational cost of privacy auditing in secure data facilities.

Table 1: Summary of evaluation metrics. WW metrics are computed directly from model weights without access to data; target model and MIA metrics require data and/or shadow models.
<table><tr><td>Category</td><td>Metric</td><td>Description</td></tr><tr><td rowspan="3">Generalisation</td><td>Train accuracy</td><td>Fit to training data</td></tr><tr><td>Test accuracy</td><td>Fit to held-out data</td></tr><tr><td>Generalisation gap</td><td> $\epsilon _ { \mathrm { g a p } } = \epsilon _ { \mathrm { t e s t } } - \epsilon _ { \mathrm { t r a i n } }$ </td></tr><tr><td rowspan="4">WW (spectral)</td><td>α</td><td>Power-law tail exponent of ESD; shape metric</td></tr><tr><td>Log α-norm</td><td>Norm-based shape and scale metric</td></tr><tr><td>Log spectral norm</td><td>Log max singular value</td></tr><tr><td>Stable rank</td><td>Effective dimensionality of weight matrix</td></tr><tr><td rowspan="2">Privacy (LiRA)</td><td>AUC</td><td>Overall MIA discriminative power</td></tr><tr><td>TPR@0.001</td><td>Attack success at low false-positive rate</td></tr></table>

## 2.3 Summary

Taken together, this literature reveals a persistent trade-off: reference-model attacks such as LiRA and RMIA offer the most reliable estimates of privacy leakage but are computationally prohibitive to apply broadly, while existing reference-free alternatives either require additional held-out data and model training (QMIA), depend on retaining full training trajectories (LT-IQR), or provide only coarse, aggregate disclosure estimates (loss-tail methods). The generalisation gap remains the most widely used training-free proxy, yet its relationship with MIA vulnerability is inconsistent, and it can be a noisy predictor when models diverge primarily in how much they memorise their training data rather than in their held-out performance. WW metrics are, by construction, even cheaper to obtain than the generalisation gap, requiring no evaluation data at all, and are grounded in a theory that explicitly links a model’s weight-space structure to the degree of implicit regularisation, and hence to overfitting and memorisation. However, no prior work has directly examined whether these spectral metrics track vulnerability to state-of-the-art MIAs such as LiRA, nor whether they carry information about privacy leakage beyond that captured by the generalisation gap.

## 3 Methodology

To study the relationship between WW measures of generalisation and MIA risk, we train a range of deep neural networks with varying depths and widths and with various hyperparameter configurations. For each of these models, we measure the MIA risk using the open source<sup>3</sup> SACRO-ML [Smith et al., 2025] implementation of LiRA [Carlini et al., 2022]. In addition, we report WW and target model metrics, including generalisation gap, and model accuracy.

## 3.1 Metrics

We evaluate models along three axes: generalisation, spectral complexity via WW, and privacy leakage via LiRA. These metrics are summarised in Table 1 and further detailed below.

Generalisation metrics. We record train accuracy, test accuracy, and generalisation gap, defined as $\epsilon _ { \mathrm { g a p } } = \epsilon _ { \mathrm { t e s t } } - \epsilon _ { \mathrm { t r a i n } } .$ . A large generalisation gap indicates that the model has memorised training examples rather than learned a broadly applicable function, and is therefore expected to be more susceptible to MIA [Shokri et al., 2017].

Spectral metrics. WW analyses the ESD of each layer’s correlation matrix X = W<sup>⊤</sup>W, where W is the layer weight matrix, without requiring access to any training or test data. For each layer, it fits the tail of the ESD to a (truncated) power-law distribution and extracts summary statistics that characterise the shape and scale of the spectrum. We use the following layer-aggregated metrics:

• α (Power-law exponent). The (negative) slope of the tail of the ESD on a log-log scale. Under HT-SR theory, smaller α indicates stronger implicit self-regularisation. The range α $\in [ 2 , 4 ]$ is considered healthy; $\alpha < 2$ suggests overfitting and rank collapse, while $\alpha > 4$ indicates an under-trained or near-random layer.

• Log α-norm. A norm-based metric that accounts for both shape and scale and is suitable for comparing networks with differing hyperparameters and depths simultaneously.

• Log spectral norm. The logarithm of the largest singular value of W, which is useful to compare models of different depths at a coarse grain level.

• Stable rank. Defined as $\mathcal { R } ( \mathbf { W } ) = { \| \mathbf { W } \| _ { F } ^ { 2 } } / { \| \mathbf { W } \| _ { 2 } ^ { 2 } }$ , the stable rank measures the effective dimensionality of the weight matrix. It is a robust, noise-tolerant alternative to the matrix rank. Higher stable rank indicates that the layer remains more random-like with weaker implicit self-regularisation. In HT-SR theory, this lack of regularisation provides the model with excess capacity, suggesting a greater tendency for memorisation.

Membership inference metrics. We evaluate privacy leakage using the online variant of LiRA [Carlini et al., 2022], which frames membership inference as a likelihood ratio test over the outputs of shadow models; here we use 64 shadow models. We report two standard metrics: (i) AUC, the area under the ROC curve, where 0.5 corresponds to a random-guess attacker and 1.0 to a perfect attacker; and (ii) TPR@0.001, the true positive rate at a fixed false positive rate of 0.001, which measures attack success in the low-false-positive regime most relevant to realistic adversarial settings [Carlini et al., 2022].

## 3.2 Datasets and Target Models

Datasets. For initial exploration we use CIFAR-10, a 10-class image classification dataset with 60,000 $3 2 \times 3 2 \times 3$ RGB images; and the OpenML<sup>4</sup> Volkert dataset (ID: 41166), a 10-class tabular dataset with 58,310 instances and 180 numeric features. For the Volkert dataset, features are normalised with zero mean and unit variance.

Architectures. All models are feedforward multi-layer perceptrons (MLPs) with ReLU activations, no dropout or normalisation layers are used. A variety of depths D (number of hidden layers) and widths W (units per hidden layer), denoted $D \times W$ are used to include a range of target models. All architectures contain a 10-class linear output layer. Cross-entropy loss with implicit softmax is used for training.

Training configurations. For each architecture, we train models across a grid of hyperparameters: learning rates $\bar { \in } \{ 0 . 0 0 1 , 0 . 0 1 \}$ and weight decay $\in \{ 0 , 0 . 0 0 0 5 \}$ . Stochastic gradient descent with a momentum of 0.9 and batch size of 32 is used for optimisation. All models are initialised using Kaiming uniform initialisation [He et al., 2015]. To capture varying levels of memorisation, we evaluate models after 100 epochs. This yields 11 architectures×2 learning rates×2 weight decay values = 44 distinct target models with diverse training trajectories and generalisation properties.

While it has become standard practice to use a 50-50% train/test split to compare different MIAs [Carlini et al., 2022], here we use the standard 50,000/10,000 split for CIFAR-10 and a random stratified 80-20% split for the Volkert dataset to reflect a more realistic data availability when auditing realworld models.

Table 2 summarises the architectures (network type) and hyperparameters (learning rate, epochs, and weight decay). Other components (such as activations) remain fixed across all experiments to isolate the effects of the varied hyperparameters.

## 4 Results

Figures 1 and 2 examine the relationship between spectral metrics and LiRA privacy leakage on the CIFAR-10 and Volkert datasets. Across both datasets, stable rank exhibits the strongest association with LiRA AUC, with Spearman’s rank correlations of $\rho = 0 . 8 7$ on CIFAR-10 and $\rho = 0 . 6 0$ on Volkert $( p \leq 0 . 0 1 )$ . In addition, Log α-Norm is most strongly associated with LiRA TPR@0.001, with moderate negative correlations of $\rho = - 0 . 5 5$ and $\rho = - 0 . 4 0 \left( p \leq 0 . 0 1 \right)$ . These results suggest that different spectral characteristics capture complementary aspects of privacy leakage, with stable rank reflecting overall attack success and Log α-Norm better tracking leakage in the low FPR regime.

Table 2: Target model architectures and hyperparameters; 44 total models.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Values</td><td>Rationale / Notes</td></tr><tr><td rowspan="2">Architecture</td><td>Network type</td><td> $\mathbf { M L P - } D { \times } W$  for  $( D , W ) \in \{ ( 1 , 4 0 9 6 ) .$  (2, 1024), (2, 2048), (3, 512), (3, 1024), (3, 2048), (4, 1024), (4, 2048), (4, 4096),</td><td>Varied depth D and width W.</td></tr><tr><td>Activation Dropout / normalisation None</td><td>(8, 512), (8, 1024)} ReLU</td><td>Fixed across all runs. Avoids confound with weight decay.</td></tr><tr><td>Optimisation Learning rate</td><td>Optimiser Weight decay</td><td>SGD (momentum = 0.9) {0.001, 0.01}  $\{ 0 , 5 \times 1 0 ^ { - 4 } \}$ </td><td>Fixed across all runs. Low vs. high, implicit regularisation. Explicit  $L _ { 2 }$  regularisation strength.</td></tr><tr><td>Data</td><td>Dataset fraction Data augmentation Label noise</td><td>100% None None (0%)</td><td>Fixed across all runs. Fixed across all runs. Fixed across all runs.</td></tr><tr><td>Training</td><td>Batch size Epochs</td><td>32 100</td><td>Fixed across all runs. Range of under and overfitting.</td></tr></table>

Relationship between spectral metrics and privacy leakage. Within the HT-SR framework, the power-law exponent α provides a measure of implicit regularisation, with $\alpha \approx 2$ corresponding to well-regularised models and $\alpha \ll 2$ indicating rank collapse. Here the relationship between α and LiRA AUC is seen to be inconsistent across datasets, with almost no correlation on CIFAR-10 $( \rho = - 0 . 0 8 )$ but a moderate positive correlation on Volkert $( \rho = 0 . 6 1 )$ ). However, α is consistently negatively correlated with LiRA TPR@0.001 across datasets $( \rho = - 0 . 3 6 $ and $\rho = - 0 . 3 7 )$ and this relationship strengthens for Log α-Norm $( \rho = - 0 . 5 5$ and $\rho = - 0 . 4 0 )$ .

Stable rank is seen to be positively associated with LiRA AUC. In HT-SR theory, stronger selfregularisation (characterised by lower stable rank and smaller α) drives better generalisation. In this experimental setup, stable rank appears to function primarily as a proxy for model capacity (larger matrices naturally have higher stable rank). No correlation between stable rank and LiRA TPR@0.001 is observed on $\mathrm { C I F A \bar { R } - 1 0 } ( \rho = 0 . 0 2 3 )$ , however a moderate negative correlation is seen on Volkert $( \rho = - 0 . 4 4 4 )$

Comparison with generalisation gap. Figure 3 compares conventional measures of generalisation with LiRA privacy leakage. As expected, models with larger $\epsilon _ { \mathrm { g a p } }$ are more vulnerable to MIAs, with positive correlations between $\epsilon _ { \mathrm { g a p } }$ and LiRA AUC of $\rho = 0 . 6 7$ on CIFAR-10 and $\rho = 0 . 5 4$ on Volkert $( p ~ \le ~ 0 . 0 1 )$ , consistent with previous work [Yeom et al., 2018, Carlini et al., 2022]. However, these relationships are consistently weaker than those observed for stable rank. Similarly, the correlation between $\epsilon _ { \mathrm { g a p } }$ and LiRA TPR@0.001 is weaker $( \rho = 0 . 0 4$ on CIFAR-10 and $\rho = - 0 . 3 4$ on Volkert) than the corresponding relationship with Log α-Norm $( \rho = - 0 . 5 5$ and $\rho = - 0 . 4 0 )$ . Since TPR@0.001 reflects attacker performance at an operationally relevant FPR, these results suggest that spectral metrics may provide stronger indicators of practically significant privacy leakage than the conventional $\epsilon _ { \mathrm { g a p } }$

This weaker relationship may partly reflect how $\epsilon _ { \mathrm { g a p } }$ is constructed: within our hyperparameter grid, the gap appears to be driven predominantly by train accuracy rather than test accuracy. On $\mathrm { C I F A R - } 1 0 , \epsilon _ { \mathrm { g a p } }$ correlates strongly with train accuracy $( \rho = 0 . 6 2 , \dot { p } < 0 . 0 0 1 )$ but only weakly and non-significantly with test accuracy $( \rho = 0 . 2 3 , p = 0 . 1 3 )$ ; the same asymmetry holds on Volkert (train accuracy: $\rho = 0 . 4 7 , p = 0 . 0 0 1$ ; test accuracy: $\rho = - 0 . 0 9 , p = 0 . 5 8 )$ ). This suggests that models in our grid diverge primarily through their degree of training-set memorisation rather than through differences in held-out generalisation performance, which may explain why $\epsilon _ { \mathrm { g a p } }$ is a noisier predictor of LiRA vulnerability than stable rank: the gap conflates two components whose relationship to privacy leakage differ, whereas stable rank appears to track the memorisation-relevant component more directly.

![](images/97e79b54955fbb8c99cedc3694e8bc7754c286c4a63ed19c468998fd838355bc.jpg)  
(a)

![](images/2ec6c81dbdb7a8c5fd5fb314f2daf82a6c499bf1013585354c8d63a2a70abb43.jpg)  
(b)

![](images/9d43004aa5bfecb86e358024e72a502236bcefa3213c83c620caf6d92853d2d8.jpg)

![](images/4719dbcc61b7691accb28a1570d70e8ab3403b2d106b4db76f3d72cb467b64ca.jpg)

![](images/d386a7bbe7598e4de302cae3c814536fd1b1aa2878b25dceadb9516331a3bac9.jpg)  
(c)  
(d)

(e)  
![](images/40ac66f914e6594ce034d7040bb055d2f6fddc5c816ff234fbca7d6d0f13b334.jpg)  
(f)

![](images/2b96b261a56a749bddae4ecde74d51389964cd2cef3c70c46dee05c796946c95.jpg)  
(g)

![](images/096d3e00203bbcae3e6c284e8d928716a666711e9c77aae26d05bc68680a431e.jpg)  
(h)  
Figure 1: CIFAR-10: Relationship between privacy risk (LiRA AUC and TPR@0.001) and spectral metrics (α, log α-norm, log spectral norm, and stable rank). LiRA TPR is shown on log-scale.

![](images/95793935dbb9345f8c7351eb061a062d0b6c46570f16b35ccf11c12fe519df6e.jpg)  
(a)

![](images/4883f603288cefd163aeba56434a176c39b632109ec4d2f32428508da479cf3e.jpg)  
(b)

![](images/7600eeccdbe82e217ee8fc7a13df86d084150aa7022cd8f44db94168ad8873da.jpg)

![](images/d9f0f35d618d3a1f5795fc96f260c982276e7065f90faa976d0d2bf7f48414df.jpg)

![](images/d5b0e7563372d68fb12f46b964a652254315699e9e227688d33b6e565d423e20.jpg)  
(e)

(d)  
(c)  
![](images/3a0e03a216dfeaa5515b622d95e95eca0f6c9a611f64aaa69e6424632e959145.jpg)  
(f)

![](images/1073c137d80df3b2fe52516b8869bcb3609f8ef4dec981e9b0ecba8da334b12f.jpg)  
(g)

![](images/0e168a559ce8769163b0476b6dbdca33a71931dff6e15943697b3fbfc2742909.jpg)  
(h)  
Figure 2: Volkert: Relationship between privacy risk (LiRA AUC and TPR@0.001) and spectral metrics (α, log α-norm, log spectral norm, and stable rank). LiRA TPR is shown on log-scale.

Relationship between spectral metrics and generalisation. Figure 4 investigates how spectral metrics relate to conventional measures of generalisation. Stable rank is positively correlated with test accuracy on both datasets $( \rho = 0 . 7 3$ on CIFAR-10 and $\rho = 0 . 5 7$ on Volkert; $p \leq 0 . 0 1 )$ , indicating that the same spectral characteristics associated with improved predictive performance are also associated with increased privacy leakage. In contrast, Log α-Norm exhibits only weak correlations with both test accuracy and $\epsilon _ { \mathrm { g a p } } ,$ with the direction of these relationships differing between datasets. Together, these findings suggest that spectral metrics capture information beyond conventional generalisation measures, helping to explain their stronger association with LiRA privacy leakage.

A summary of correlations between spectral, generalisation, and privacy metrics is shown in Table 3.

Combined predictors of privacy leakage. To assess whether stable rank and $\epsilon _ { \mathrm { g a p } }$ capture complementary information about privacy leakage, we fit a multivariate ordinary least squares regression predicting LiRA AUC from both standardised predictors jointly. On CIFAR-10, the combined model explains substantially more variance $( R ^ { 2 } = 0 . { \dot { 7 } } 1 )$ than either predictor alone $( \epsilon _ { \mathrm { g a p } } \colon R ^ { 2 } = 0 .$ .43; stable rank: $R ^ { 2 } = 0 . 4 7 )$ , with both standardised coefficients remaining highly significant when included together (gap: $\beta = 0 . 0 3 2 , p < 0 . 0 0 1 ;$ ; stable rank: $\beta = 0 . 0 3 5 , p < 0 . 0 0 1 )$ . A similar pattern holds on Volkert, where the combined model again outperforms either individual predictor $( R ^ { 2 } = 0 . 5 7$ vs. 0.38 and 0.26 respectively; gap: $\beta = 0 . 0 2 6 , p < 0 . 0 0 1$ ; stable rank: $\beta = 0 . 0 2 0 , p < 0 . 0 0 1 )$

![](images/8e87d8dd7e849b3a2b96934c4ad33fab5fec2b9e711215bb4d1d336e96845c08.jpg)  
(a) CIFAR-10

![](images/5891df98a0a0a086907859b97cebcb526a502c143675e54bdd7d946742fcb4b4.jpg)  
(b) CIFAR-10

![](images/6ef0958867766d5b9a839a3257a415c35826eca57158e6ee643cac25099005fb.jpg)

![](images/3a7c1030587e9f80390466a0efc95ceeb7353870c0992240ac90ededa1080f61.jpg)

![](images/36ec1dfbcff827a9020f0fcc2af04e809c2fcca613414e842bd3ec0627a5a06a.jpg)  
(e) Volkert

(c) CIFAR-10  
(d) CIFAR-10  
![](images/d595d395ba3120e01a514e41af280c1ea30337a6c9db033a1bcf453145eacc6e.jpg)  
(f) Volkert

![](images/2a5ed2529ae7bcd9c010bdbe1db797fc2da11e4507f057b97beb54a4d049aafc.jpg)  
(g) Volkert

![](images/3e6726e2bc842e43ae0ab70fe1989f9d040872eeaf1d2cd6caa6136f475c49d7.jpg)  
(h) Volkert  
Figure 3: Relationship between the generalisation metrics and privacy disclosure risk (LiRA AUC and TPR@0.001) on CIFAR-10 and Volkert datasets. LiRA TPR is shown on log-scale.

![](images/aa96f76c65f28f0dbfd1395042572999ba87be14cabdbac674ea2396fd895278.jpg)  
(a) CIFAR-10

![](images/3e6bc68ebfb84ea84e6cf20d204b35a39011d75621d248c6a3f4715b545f3f22.jpg)  
(b) CIFAR-10

![](images/c427d057a499fabbde164a50fc0948709660afcabcf3588f49d8c4169275b61c.jpg)  
(c) CIFAR-10

![](images/1d24227906326190a6332511b28f8e6c6878d603fe09625406baa0d6d8c48e28.jpg)  
(d) CIFAR-10

![](images/a8a3bd293ded2b40b6e4084123babc1f5f47fc9fe48cfa2ea03ffb0f0e585434.jpg)  
(e) Volkert

![](images/dcf866b99bcf93771b06b095212759c42a3811d302afd669f6e07141e2cf272d.jpg)  
(f) Volkert

![](images/7d6e15cccca56bd4a0c86000f04964bd46101fe81e45613c3c8f702f0c74aa89.jpg)  
(g) Volkert

![](images/ddf858de711173fd45d52968261d3459208660189573d29ff9a9d832f6abe21e.jpg)  
(h) Volkert  
Figure 4: Relationship between generalisation and spectral metrics (Log α-Norm, and stable rank) on CIFAR-10 and Volkert datasets.

although here $\epsilon _ { \mathrm { g a p } }$ contributes a larger standardised effect than stable rank, reversing the ordering observed on CIFAR-10.

Using log stable rank in place of raw stable rank further increases performance on both datasets (combined $R ^ { 2 } = 0 . 8 2$ on CIFAR-10 and $R ^ { 2 } = 0 . 6 2$ on Volkert), and additionally resolves the coefficient-ordering reversal, with log stable rank contributing at least as much predictive weight as the generalisation gap on both datasets.

Together, these results indicate that stable rank and $\epsilon _ { \mathrm { g a p } }$ provide complementary, non-redundant information about privacy leakage across both datasets.

We repeated this analysis for LiRA TPR@0.001, log-transforming the target to address strong right-skew in its raw distribution (consistent with the log-scale presentation in Figures 1 and 2). On CIFAR-10, the combined model explains modestly more variance than either predictor alone $( R ^ { 2 } = 0 . 1 9 { \mathrm { ~ v s } } .$ 0.16 for Log α-Norm and 0.001 for $\epsilon _ { \mathrm { g a p } }$ individually), with Log α-Norm remaining the dominant, significant predictor $( \beta = - 0 . 8 0 , p = \check { 0 } . 0 0 4 )$ while $\epsilon _ { \mathrm { g a p } }$ does not reach significance $( \beta = - 0 . 3 3 , p = 0 . 2 1 )$ . On Volkert, the combined model again modestly outperforms either predictor alone $( R ^ { 2 } = \bar { 0 } . 1 6 \mathrm { v s } . 0 . 0 \Omega$ and 0.07 respectively), but here the pattern reverses: $\epsilon _ { \mathrm { g a p } }$ is the (marginally) significant predictor $( \beta = - 0 . 5 0 , p = 0 . 0 4 5 )$ , while Log α-Norm falls just short of significance $( \bar { \beta } = - 0 . 4 \bar { 4 } , p = 0 . 0 7 5 )$

Table 3: Summary of Spearman correlations (ρ) between spectral, generalisation, and privacy metrics.
<table><tr><td rowspan="2"></td><td colspan="2">CIFAR-10</td><td colspan="2">Volkert</td></tr><tr><td>ρ</td><td>p-value</td><td>ρ</td><td>p-value</td></tr><tr><td>WW metrics vs. LiRA AUC</td><td></td><td></td><td></td><td></td></tr><tr><td>α vs. AUC</td><td>-0.08</td><td>0.61</td><td>0.61</td><td>0.00</td></tr><tr><td>Log α-Norm vs. AUC</td><td>-0.50</td><td>0.00</td><td>0.10</td><td>0.52</td></tr><tr><td>Log Spectral Norm vs. AUC</td><td>-0.55</td><td>0.00</td><td>-0.27</td><td>0.08</td></tr><tr><td>Stable Rank vs. AUC</td><td>0.87</td><td>0.00</td><td>0.60</td><td>0.00</td></tr><tr><td>WW metrics vs. LiRA TPR@0.001</td><td></td><td></td><td></td><td></td></tr><tr><td>α vs. TPR@0.001</td><td>-0.36</td><td>0.02</td><td>-0.37</td><td>0.01</td></tr><tr><td>Log α-Norm vs. TPR@0.001</td><td>-0.55</td><td>0.00</td><td>-0.41</td><td>0.01</td></tr><tr><td>Log Spectral Norm vs. TPR @0.001</td><td>-0.45</td><td>0.00</td><td>-0.18</td><td>0.25</td></tr><tr><td>Stable Rank vs. TPR@0.001</td><td>0.02</td><td>0.88</td><td>-0.44</td><td>0.00</td></tr><tr><td>Generalisation metrics vs. Privacy</td><td></td><td></td><td></td><td></td></tr><tr><td>€gap Vs. AUC</td><td>0.68</td><td>0.00</td><td>0.54</td><td>0.00</td></tr><tr><td>egap Vs. TPR@0.001</td><td>0.04</td><td>0.78</td><td>-0.34</td><td>0.03</td></tr><tr><td>Test Acc. vs. AUC</td><td>0.77</td><td>0.00</td><td>0.40</td><td>0.01</td></tr><tr><td>Test Acc. vs. TPR@0.001</td><td>-0.13</td><td>0.41</td><td>-0.37</td><td>0.01</td></tr><tr><td>Generalisation metrics vs. WW metrics</td><td></td><td></td><td></td><td></td></tr><tr><td>€gap vs. Log α-Norm</td><td>-0.39</td><td>0.01</td><td>-0.08</td><td>0.62</td></tr><tr><td>€gap vs. Stable Rank</td><td>0.49</td><td>0.00</td><td>0.16</td><td>0.29</td></tr><tr><td>Test Acc. vs. Log α-Norm</td><td>-0.22</td><td>0.15</td><td>0.44</td><td>0.00</td></tr><tr><td>Test Acc. vs. Stable Rank</td><td>0.73</td><td>0.00</td><td>0.57</td><td>0.00</td></tr><tr><td>Generalisation gap vs. Accuracy</td><td></td><td></td><td></td><td></td></tr><tr><td>€gap vs. Train Acc.</td><td>0.62</td><td>0.00</td><td>0.47</td><td>0.00</td></tr><tr><td>€gap vs. Test Acc.</td><td>0.23</td><td>0.13</td><td>-0.09</td><td>0.58</td></tr></table>

Combined explanatory power for TPR@0.001 is substantially lower than for AUC on both datasets, indicating that while spectral and generalisation-based metrics jointly explain most of the variance in overall attack success, predicting privacy leakage in the low-FPR regime remains considerably harder and the relative contribution of each predictor is less consistent across domains. Results are summarised in Table 4.

Predicting high-risk models. To assess the practical utility for flagging high-risk models, we treat vulnerability as a binary outcome for each target model (LiRA TPR@0.001 > 0.015, corresponding to a 15× improvement over random guessing) and compute ROC-AUC for each metric as a standalone classifier. Log α-Norm and α achieve consistently strong separability on both CIFAR-10 (0.79, 0.75) and Volkert (0.70, 0.74). In contrast, $\epsilon _ { \mathrm { g a p } }$ is only informative on Volkert (0.79) but performs near chance on CIFAR-10 (0.47), while stable rank shows the opposite pattern, performing reasonably on CIFAR-10 (0.67) but worse than random on Volkert (0.29).

This pattern is robust to the choice of threshold: repeating the analysis at 10× and 20× random guessing yields the same qualitative ordering, with α remaining the most consistently strong standalone predictor across both datasets (0.70–0.77) and $\epsilon _ { \mathrm { g a p } }$ remaining weak on CIFAR-10 (0.40–0.50) but strong on Volkert (0.79–0.84) at every threshold tested.

Combining $\epsilon _ { \mathrm { g a p } }$ with Log α-Norm via logistic regression produces more consistent separability across datasets at every threshold tested (Table 5), leaving CIFAR-10 performance largely unchanged while substantially improving Volkert, most notably at 20× where standalone Log α-Norm performs much worse on Volkert than CIFAR-10 (0.61 vs. 0.83), but performance converges on the two datasets once combined (0.81 vs. 0.82). This suggests that combining spectral and generalisation-based signals offers a more dependable classifier than relying on any single metric alone, particularly as standalone metrics diverge across domains, consistent with the low-FPR regime being harder to predict (as observed above) and further motivating combined approaches for use in practice.

Table 4: Ordinary least squares regression predicting LiRA AUC from standardised generalisation gap and log stable rank, and log LiRA TPR@0.001 from generalisation gap and Log α-Norm (n = 44 per dataset). $\beta$ denotes the standardised regression coefficient, reflecting the change in AUC (or log TPR) per one standard deviation change in the predictor.
<table><tr><td rowspan="2"></td><td colspan="2">CIFAR-10</td><td colspan="2">Volkert</td></tr><tr><td>Stat</td><td>p-value</td><td>Stat</td><td>p-value</td></tr><tr><td>LiRA AUC</td><td></td><td></td><td></td><td></td></tr><tr><td> $\epsilon _ { \mathrm { g a p } } \mathrm { o n l y } ( R ^ { 2 } )$ </td><td>0.43</td><td></td><td>0.38</td><td></td></tr><tr><td> $\mathrm { L o g ~ S t a b l e ~ R a n k ~ o n l y ~ } ( R ^ { 2 } )$ </td><td>0.68</td><td></td><td>0.32</td><td></td></tr><tr><td>Combined  $R ^ { 2 }$ </td><td>0.82</td><td></td><td>0.62</td><td></td></tr><tr><td> $\beta _ { \mathrm { g a p } } ( \mathrm { c o m b i n e d } )$ </td><td>0.03</td><td>0.00</td><td>0.03</td><td>0.00</td></tr><tr><td> $\beta _ { \mathrm { l o g ~ s t a b l e ~ r a n k } } ( \mathrm { c o m b i n e d } )$ </td><td>0.04</td><td>0.00</td><td>0.02</td><td>0.00</td></tr><tr><td>(log) LiRA TPR@0.001</td><td></td><td></td><td></td><td></td></tr><tr><td> $\epsilon _ { \mathrm { g a p } } \mathrm { o n l y } ( R ^ { 2 } )$ </td><td>0.00</td><td></td><td>0.09</td><td></td></tr><tr><td> $\mathsf { L i g \alpha \alpha \mathrm { - N o r m \ o n l y \ } } ( R ^ { 2 } )$ </td><td>0.16</td><td></td><td>0.07</td><td></td></tr><tr><td> $\mathrm { C o m b i n e d } R ^ { 2 }$ </td><td>0.19</td><td></td><td>0.16</td><td></td></tr><tr><td> $\beta _ { \mathrm { g a p } } ( \mathrm { c o m b i n e d } )$ </td><td>-0.33</td><td></td><td>0.21-0.50</td><td>0.05</td></tr><tr><td> $\beta _ { \mathrm { L o g } \alpha \mathrm { - N o r m } } ^ { \mathrm { - } \mathrm { ~ } } ( \mathrm { c o m b i n e d } )$ </td><td>-0.80</td><td></td><td>0.00-0.44</td><td>0.08</td></tr></table>

Table 5: ROC-AUC for classifying models as vulnerable versus safe (LiRA TPR@0.001 above vs. below 10×, 15×, or 20× random guessing) using Log α-Norm and $\epsilon _ { \mathrm { g a p } }$ individually, and combined via logistic regression, n = 44 per dataset.
<table><tr><td></td><td>10× 15× 20×</td></tr><tr><td>CIFAR-10</td><td></td></tr><tr><td> $\epsilon _ { \mathrm { g a p } }$ </td><td>Log α-Norm 0.74 0.79 0.83 0.500.470.40</td></tr><tr><td>Combined</td><td>0.73 0.77 0.82</td></tr><tr><td>Volkert</td><td>Log α-Norm 0.70 0.70 0.61</td></tr><tr><td> $\epsilon _ { \mathrm { g a p } }$  Combined</td><td>0.82 0.79 0.84 0.81 0.77 0.81</td></tr></table>

## 5 Conclusions and Limitations

Summary This paper presents a first investigation into whether inexpensive WW spectral metrics can be used to predict MIA privacy disclosure risk. Across image and tabular datasets, we find that stable rank is strongly associated with LiRA AUC, while Log α-Norm exhibits a consistent negative association with LiRA TPR@0.001.

Importantly, these relationships are stronger than those observed using the conventional $\epsilon _ { \mathrm { g a p } }$ . Moreover, combining spectral and generalisation-based metrics improves prediction over either alone, indicating that spectral metrics and $\epsilon _ { \mathrm { g a p } }$ capture different complementary aspects of privacy vulnerability.

To test whether spectral and generalisation-based metrics provide complementary rather than redundant information, we fit multivariate regressions combining $\epsilon _ { \mathrm { g a p } }$ with the strongest associated WW metric for each privacy outcome. Combining $\epsilon _ { \mathrm { g a p } }$ and (log) stable rank substantially improved prediction of LiRA AUC over either predictor alone on both datasets, with both coefficients remaining significant jointly, indicating genuinely complementary signal. Gains for LiRA TPR@0.001 were more modest, and which predictor dominated reversed between datasets, suggesting that the low FPR regime is inherently harder to predict consistently across domains.

Beyond continuous prediction, we further show that treating vulnerability as a binary outcome enables high-risk models to be flagged: α and Log α-Norm reliably separate high-risk from low-risk models on both datasets, whereas $\epsilon _ { \mathrm { g a p } }$ and stable rank do not generalise consistently across domains for this task. Combining $\epsilon _ { \mathrm { g a p } }$ with Log α-Norm offers a modest, more consistent improvement over either metric alone, echoing the difficulty of predicting leakage in the low-FPR regime observed in the continuous analysis.

These findings indicate that spectral properties of neural network weight matrices provide a promising and computationally efficient signal for assessing privacy risk without requiring expensive attack simulations. Since spectral metrics can be computed directly from trained model weights, they offer the potential for scalable privacy auditing and model selection based on privacy characteristics.

While the results demonstrate consistent correlations across two datasets, they do not establish a causal relationship between spectral properties and privacy leakage. Our working hypothesis assumes that WW metrics predict MIA vulnerability indirectly by acting as proxies for model generalisation and overfitting, with generalisation itself serving as a proxy for privacy risk. However, the stronger associations observed between WW metrics and LiRA performance than between $\epsilon _ { \mathrm { g a p } }$ and LiRA suggest that this hypothesis may be incomplete. Future work should therefore investigate whether spectral metrics directly characterise properties of the loss distributions exploited by likelihood-based attacks. In addition, this study considers only pairwise combinations and monotonic relationships; richer multivariate or non-linear combinations of spectral features, potentially incorporating all WW metrics jointly, may provide substantially stronger predictors of privacy disclosure risk, particularly in the harder to predict low FPR regime.

Limitations. This study is limited to a single image dataset (CIFAR-10) and a single tabular dataset (Volkert) with MLPs trained for 100 epochs. Although these datasets provide complementary domains and CIFAR-10 is one of the most widely used benchmarks in MIA research [Hu et al., 2022], it remains unclear whether the observed relationships extend to larger, more complex architectures (e.g., ResNets and Transformers) or across more diverse data modalities (e.g., text). Additionally, we evaluate only one MIA method (LiRA). While LiRA is widely regarded as a strong MIA, alternative attack formulations may exhibit different relationships with spectral properties. Our multivariate analysis is similarly limited in scope: with only 44 models per dataset, the regressions should be interpreted as suggestive rather than definitive, and we did not attempt cross-dataset transfer or held-out validation of the fitted models, which would provide a more stringent test of generalisability. Consequently, the results should be interpreted as evidence that WW metrics are promising indicators of LiRA privacy risk, rather than as establishing their general applicability across architectures, datasets, or attack methodologies.

## Acknowledgments and Disclosure of Funding

This work was funded by UK Research and Innovation (Grant Number MC\_PC\_24038) as part of the Data and Analytics Research Environments UK (DARE UK) programme, delivered in partnership with Health Data Research UK (HDR UK) and Administrative Data Research UK (ADR UK). The authors wish to thank Charles H. Martin for discussions of WeightWatcher.

## References

Galen Andrew et al. One-shot empirical privacy estimation for federated learning. In The Twelfth International Conference on Learning Representations, Vienna, Austria, 2024. OpenReview.net. URL https://openreview.net/forum?id=0BqyZSWfzo.

Martin Bertran et al. Scalable membership inference attacks via quantile regression. In A. Oh et al., editors, Advances in Neural Information Processing Systems, volume 36, pages 314–330, Red Hook, NY, USA, 2023. Curran Associates Inc.

Nicholas Carlini et al. Membership inference attacks from first principles. In Rakesh Bobba, editor, IEEE Symposium on Security and Privacy, pages 1897–1914, Piscataway, NJ, USA, 2022. IEEE Press. doi: 10.1109/SP46214.2022.9833649.

Euodia Dodd, Nataša Krco, Igor Shilov, and Yves-Alexandre de Montjoye. The tail tells all:ˇ Estimating model-level membership inference vulnerability without reference models. arXiv, 2510.19773, 2025.

Andrey V. Galichin, Mikhail Pautov, Alexey Zhavoronkin, Oleg Y. Rogov, and Ivan Oseledets. GLiRA: Closed-box membership inference attack via knowledge distillation. IEEE Transactions on Information Forensics and Security, 20:3893–3906, March 2025. doi: 10.1109/TIFS.2025. 3550068.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Ruzena Bajcsy, Greg Hager, and Yi Ma, editors, IEEE International Conference on Computer Vision, pages 1026–1034, Los Alamitos, CA, USA, 2015. IEEE Computer Society. doi: 10.1109/ICCV.2015.123.

Hongsheng Hu et al. Membership inference attacks on machine learning: A survey. ACM Computing Surveys, 54(11s):1–37, September 2022. doi: 10.1145/3523273.

Emily Jefferson et al. GRAIMATTER Green Paper: Recommendations for disclosure control of trained Machine Learning (ML) models from Trusted Research Environments (TREs). arXiv, 2211.01656, November 2022. doi: 10.48550/arXiv.2211.01656.

Tobias Leemann, Martin Pawelczyk, and Gjergji Kasneci. Gaussian membership inference privacy. In A. Oh et al., editors, Advances in Neural Information Processing Systems, volume 36, pages 73866–73878, Red Hook, NY, USA, 2023. Curran Associates, Inc.

Chenxi Li, Abhinav Kumar, Zhen Guo, Jie Hou, and Reza Tourani. Unveiling the unseen: Exploring whitebox membership inference through the lens of explainability. arXiv, 2407.01306, July 2024. doi: 10.48550/arXiv.2407.01306.

Han Liu, Yuhao Wu, Zhiyuan Yu, and Ning Zhang. Please tell me more: Privacy impact of explainability through the lens of membership inference attack. In IEEE Symposium on Security and Privacy, pages 4791–4809, Piscataway, NJ, USA, 2024. IEEE Press. doi: 10.1109/SP54263. 2024.00120.

Lan Liu, Yi Wang, Gaoyang Liu, Kai Peng, and Chen Wang. Membership inference attacks against machine learning models via prediction sensitivity. IEEE Transactions on Dependable and Secure Computing, 20(3):2341–2347, May 2023. doi: 10.1109/TDSC.2022.3180828.

Charles H. Martin and Michael W. Mahoney. Implicit self-regularization in deep neural networks: Evidence from random matrix theory and implications for learning. Journal of Machine Learning Research, 22(165):1–73, 2021.

Charles H. Martin, Tongsu (Serena) Peng, and Michael W. Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications, 12(4122):1–13, July 2021. doi: 10.1038/s41467-021-24025-8.

Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In IEEE Symposium on Security and Privacy, volume 1, pages 739–753, Piscataway, NJ, USA, 2019. IEEE Press. doi: 10.1109/SP.2019.00065.

Yuefeng Peng, Jaechul Roh, Subhransu Maji, and Amir Houmansadr. OSLO: One-shot label-only membership inference attacks. In A. Globerson et al., editors, Advances in Neural Information Processing Systems, volume 37, pages 62310–62333, Red Hook, NY, USA, 2024. Curran Associates, Inc.

Joseph Pollock, Igor Shilov, Euodia Dodd, and Yves-Alexandre de Montjoye. Free Record-Level privacy risk evaluation through Artifact-Based methods. In 34th USENIX Security Symposium (USENIX Security 25), pages 5525–5544. USENIX Association, 2025.

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: General ization beyond overfitting on small algorithmic datasets. arXiv, 2201.02177, January 2022. doi: 10.48550/arXiv.2201.02177.

Hari K. Prakash and Charles H. Martin. Grokking and generalization collapse: Insights from HTSR theory. arXiv, 2506.04434, June 2025. doi: 10.48550/arXiv.2506.04434.

Richard J. Preen and Jim Smith. A hierarchical approach for assessing the vulnerability of tree-based classification models to membership inference attack. Transactions on Data Privacy, 19(1):29–55, 2026.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In Kevin R. B. Butler, editor, IEEE Symposium on Security and Privacy, pages 3–18, Piscataway, NJ, USA, 2017. IEEE Press. doi: 10.1109/SP.2017.41.

Jim Smith et al. Safe machine learning model release from trusted research environments: The SACRO-ML package. arXiv, 2212.01233, August 2025. doi: 10.48550/arXiv.2212.01233.

Thomas Steinke, Milad Nasr, and Matthew Jagielski. Privacy auditing with one (1) training run. In A. Oh et al., editors, Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 49268–49280, Red Hook, NY, USA, 2023. Curran Associates, Inc.

Anshuman Suri, Xiao Zhang, and David Evans. Do parameters reveal more than loss for membership inference? Transactions on Machine Learning Research, 2024(12), 2024. URL https:// openreview.net/forum?id=fmKJfbGKFC.

Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In Cas Cremers, editor, IEEE Computer Security Foundations Symposium, pages 268–282, Piscataway, NJ, USA, 2018. IEEE Press. doi: 10.1109/CSF.2018.00027.

Sajjad Zarifzadeh, Philippe Liu, and Reza Shokri. Low-cost high-power membership inference attacks. In International Conference on Machine Learning, pages 58244–58282, Vienna, Austria, 2024. JMLR.org.

Rongting Zhang, Martin Bertran, and Aaron Roth. Order of magnitude speedups for LLM membership inference. arXiv, 2409.14513, September 2024. doi: 10.48550/arXiv.2409.14513.