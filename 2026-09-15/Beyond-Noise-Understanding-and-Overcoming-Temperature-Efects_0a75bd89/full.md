# Beyond Noise: Understanding and Overcoming Temperature Efects in Analog DNN Inference

Niklas Summ , Xiao Wang , Hendrik Borras , Bernhard Klein , and Holger Fröning

Hardware and Artificial Intelligence Lab, Institute of Computer Engineering,

Heidelberg University, Germany

niklas.summ@stud.uni-heidelberg.de,

{xiao.wang,hendrik.borras,bernhard.klein,holger.froening}@ziti.uni-heidelberg.de

Abstract. The energy eficiency of analog computing makes it one of the most promising candidates for deploying resource-intensive machine learning workloads on constrained platforms such as mobile and embedded devices. However, analog accelerators are inherently susceptible to noise and non-idealities arising from physical component variations, whose behavior is further sensitive to environmental factors. These efects can significantly degrade inference accuracy. In this work, we conduct a comprehensive experimental study on a representative example of analog hardware to investigate the impact of temperature. We first characterize the behavior of stochastic and systematic non-idealities across a range of operating temperatures. Following this, we compare a set of simulationbased and hardware-based mitigation strategies aimed at improving robustness against temperature-induced performance degradation. Our results suggest that temperature-induced degradation is driven primarily by systematic non-idealities rather than stochastic noise alone. Noiseaware training improves robustness, while hardware-in-the-loop training and temperature-aware calibration provide the strongest accuracy retention across varying thermal conditions.

Keywords: Analog Computations · Temperature-Induced Noise · Noisy Training · Hardware-aware Training · Calibration.

## 1 Introduction

The computational demands of deep neural networks continue to grow rapidly, creating increasing pressure on conventional digital hardware. While modern GPUs and neural processing units provide high throughput, their energy consumption and data movement overhead remain major obstacles. Analog computing has therefore re-emerged as a promising alternative for neural network inference [8,13]. It exploits physical quantities without digitization to perform highly-parallel energy-eficient multiply-accumulate operations.

However, the benefits of analog computing come at the cost of reduced computational reliability. In contrast to digital arithmetic, analog computation is inherently afected by non-idealities such as device-to-device variations, drift, thermal fluctuations, nonlinearities, and noise. In the following, we will use the term “noise” to refer to stochastic efects, while the term “systematic nonidealities” will refer to contributions that ofset the signal amplitude in a predictable manner. While there are recent works which use noise as a computational resource [2,3], we note that even in such cases, noise can still degrade performance. Thus, robust deployment of neural networks on analog hardware requires characterization and training methods that explicitly account for imperfect and changing computation conditions. Of particular interest for edge deployments, such as personal computing, are temperature variations, as these cannot reasonably be kept constant.

A widely used approach to improve robustness against analog computation errors is noisy training [18,4], where noise is injected during training to expose the model to perturbations similar to those expected during inference. Prior work has shown that noisy training can substantially improve robustness compared to naïve training, quantization-aware training, or general robustness methods such as Sharpness-Aware Minimization [14].

From a simulation perspective, recent work identifies noisy training as the strongest baseline for robustness under noisy analog computations, but also as fundamentally limited when the noise level during deployment difers from the noise level assumed during training. Variance-Aware Noisy Training (VANT) [15] addresses this limitation by exposing the model to a range of possible inferencetime noise conditions rather than a single fixed noise configuration.

However, these studies generally rely on abstract or simulated noise models. While these models are useful for isolating algorithmic efects, real analog hardware often exhibits more complex behavior. In particular, environmental conditions such as temperature can influence not only the magnitude of noise, but also the operating characteristics of the analog circuitry. Thus, it remains unclear whether temperature-induced degradation in analog neural network inference can be modeled simply as a change in noise variance, or whether additional efects must be considered. This distinction is important. If temperature only changes the variance of an otherwise stable noise process, then methods such as VANT provide a natural and direct solution. However, if temperature also changes signal gain, layer-wise behavior, or the efective scaling of intermediate activations, then robustness methods based only on noise variance may be incomplete. Initial experiments applying VANT to BrainScaleS-2 (BSS-2), a mixed-signal analog neuromorphic processor, pointed in this direction.

Thus, the objective of this work is to characterize how temperature afects analog neural network inference on BSS-2 and to analyze the implications of these efects for robust training methods such as noisy training and VANT. Specifically, this paper makes the following contributions:

1. We characterize temperature-dependent analog neural network inference on BSS-2 with a focus on disentangling noise and systematic non-idealities.

2. We show that temperature-induced degradation is not fully captured by noise variance alone, as temperature afects both stochastic noise and systematic non-idealities.

3. We evaluate the efectiveness of hardware-in-the-loop training, noisy training, VANT, and diferent calibration strategies for mitigating temperatureinduced degradation.

## 2 Background

While analog computing can be realized with a variety of technologies, including optical [5], photonic [12], and phase-change memory approaches [9], the most widely adopted and the focus of this work are electronic CMOS-based implementations [8]. In these systems, operations such as multiplication and accumulation are directly mapped to the underlying circuit dynamics, enabling highly parallel and energy-eficient computation. Despite these advantages, analog CMOS systems remain susceptible to environmental variations, particularly temperature fluctuations. Although temperature compensation techniques for analog DNN accelerators have been explored [7,6], existing approaches cannot be directly applied to BSS-2 due to its distinct operating principles.

## 2.1 BrainScaleS-2

BSS-2 is a mixed-signal analog neuromorphic system based on a custom ASIC fabricated using a 65 nm CMOS process [11]. Although originally developed for the emulation of spiking neural networks (SNNs), the platform can also accelerate artificial neural networks (ANNs) in a non-spiking configuration by performing analog matrix–vector multiplications [17].

![](images/6ff9c1583005c52c8982eace865cc7c8355d4e1b7e0dc0c5cd34b6358299c24e.jpg)

The BSS-2 ASIC comprises four blocks of 128 256 synapses and 512 neuron circuits, each equipped with a dedicated ADC channel for activation readout (Fig. 1). A single execution supports matrix–vector multiplications with matrices of up to 128 512 elements in signed mode or 256  512 elements in unsigned mode. Larger computations must be partitioned across multiple hardware executions.

Fig. 2 illustrates the principle of analog computation on BSS-2. Matrix–vector multiplication is realized through charge accumulation,

Fig. 1: Internal structure of the BSS-2 ASIC, with input drivers (triangles on the left), neurons (triangles at the bottom), and synapses (green dots) [13].

where the product of an input value and a synaptic weight is represented as an electrical charge $Q = I \cdot \varDelta t .$ . The current I is determined by the locally stored 6-bit unsigned synaptic weight. In signed mode, the weight is extended by a sign bit, yielding an efective 7-bit signed representation. The pulse duration ∆t encodes the unsigned 5-bit input value. Each neuron accumulates the charge contributions generated by all synapses in its column. The accumulated charge is converted into a current by an operational transconductance amplifier (OTA), integrated on the neuron’s membrane capacitance, and subsequently digitized by an 8-bit ADC.

![](images/e36106b00afc2b189be7d06915a63118910d91f397b59ec07e26d01277f4c306.jpg)  
Fig. 2: Principle of analog matrix–vector multiplication on BSS-2 [13].

The analog realization in BSS-2 introduces both systematic and stochastic non-idealities. Manufacturing-induced mismatches lead to fixed-pattern variations in the electrical properties of individual circuit components and, consequently, systematic deviations between computational units. To compensate for these variations, BSS-2 provides a set of digitally configurable parameters that are optimized during a calibration procedure. Most importantly, the calibration adjusts the synaptic current strengths to achieve consistent responses across neurons. Furthermore, it calibrates the pulse-generation circuits to compensate for random ofsets in current-pulse encoding [17]. To reduce the impact of stochastic noise, BSS-2 supports resending the same input vector multiple times within a single integration phase. Nevertheless, residual systematic and stochastic nonidealities remain after calibration. In later sections, we revisit calibration in the context of temperature-induced variations and its impact on inference accuracy.

## 2.2 Hardening Neural Networks Against Analog Non-Idealities

To mitigate hardware-induced non-idealities, prior work has shown that hardwarein-the-loop training is highly efective [17,4]. By incorporating the target hardware into the forward path during training, the model can directly adapt to hardware-specific imperfections, similar to quantization-aware training [10]. However, its computational cost and dependence on a specific hardware state limit its applicability in dynamic environments.

An alternative is noisy training, which improves robustness by injecting additive Gaussian noise during training [18]. Most noisy training approaches assume a fixed noise level and therefore do not account for variations caused by changing operating conditions, such as temperature fluctuations, voltage instability, or device aging. Variance-Aware Noisy Training (VANT) [15] addresses this limitation by sampling the injected noise from a distribution rather than using a fixed value. Specifically, the noise level is drawn from $\mathcal { N } ( \alpha \cdot \sigma _ { \mathrm { t r a i n } } , \theta )$ , where $\sigma _ { \mathrm { t r a i n } }$ denotes the nominal noise level of the target hardware and α and θ control the mean and variance of the sampled noise. This enables the model to become robust to a range of noise conditions rather than a single operating point, making VANT a promising approach for deployment under varying operating conditions.

## 3 Noise Characterization of BrainScaleS-2 (BSS-2)

In this section, we present a detailed noise characterization using the BSS-2 system. We first examine the per-neuron behavior at room temperature and then investigate the efects of temperature on neuron-wise and global system characteristics. All experiments presented in this and the following sections were performed on the same device, operating in the signed weight mode.

## 3.1 Metrics

To evaluate hardware performance in matrix–vector multiplication tasks, we conduct a series of experiments to characterize the associated non-idealities. For this purpose, 100 matrix–vector multiplications are executed on the hardware platform and compared against numerically exact reference computations.

Input vectors consist of 128 elements, while the matrices have dimensions of $1 2 8 \times 2 5 6$ . Both sizes are chosen such that they map eficiently to the hardware. All matrix and vector elements were independently sampled from uniform distributions spanning the full value ranges supported by BSS-2. This preserves realistic activity patterns by assigning diferent computations across neurons rather than broadcasting an identical computation, which could induce atypical crosstalk. For each of the 100 executions, the matrix and input vector were resampled, reducing bias from any particular neuron-computation assignment and enabling a fair statistical comparison across neurons.

For the output vector, the hardware introduces a scaling factor relative to the exact result, referred to as the gain $G _ { \mathrm { o p } } .$ . It is defined as the median of elementwise ratios between the hardware output $\mathbf { y } _ { \mathrm { h w } }$ and the exact output vector y<sub>exact</sub>, where $i \in \{ 1 , \ldots , K \}$ indexes vector components:

$$
G _ { \mathrm { o p } } = \mathrm { m e d i a n } \left( \frac { y _ { \mathrm { h w } , i } } { y _ { \mathrm { e x a c t } , i } } \right) .
$$

The hardware error $\varDelta _ { i }$ is then defined as the diference between the hardware output and the gain-scaled exact output:

$$
\varDelta _ { i } = y _ { \mathrm { h w } , i } - G _ { \mathrm { o p } } \cdot y _ { \mathrm { e x a c t } , i } .
$$

After $N$ matrix–vector multiplication runs, we compute the neuron-wise mean error $\mu _ { i }$ and standard deviation $\sigma _ { i }$ of the error:

$$
\mu _ { i } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \varDelta _ { i , j } , \qquad \sigma _ { i } = \sqrt { \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \left( \varDelta _ { i , j } - \mu _ { i } \right) ^ { 2 } } .
$$

Fig. 3a shows $\mu _ { i }$ and $\sigma _ { i }$ for 32 representative neurons. The mean error $\mu _ { i }$ (blue bars) varies substantially across neurons and reflects deterministic, devicedependent efects associated with calibration and hardware-specific imperfections. We refer to these contributions as systematic non-idealities.

![](images/9edf7c46423dcdcaf91a6b0b513da5ff3147f8a2c6bc25446454f2051b552e5e.jpg)  
(a) Mean and standard deviation of per-neuron error for a subset of 32 neurons $( T _ { \mathrm { o p e r a t i n g } } = 4 0 ^ { \circ } \mathrm { C } )$

![](images/e5fa22c62f1008d8e7ff682978dcb02ebd9b7e4a0cdaf8f06a43f7144d1c8a55.jpg)  
(b) Per-neuron mean error over temperature for a subset of 8 neurons (each color represents one neuron).

![](images/d93bd83ca79f44dfae488de28fdfca8403e007629552bab1605503791fea0389.jpg)  
(c) Error-to-signal ratio over temperature (described in section 3.2).

Fig. 3: Error analysis over 100 random matrix–vector multiplications on BSS-2, reflecting the combined behavior of noise and systematic non-idealities.

In contrast, $\sigma _ { i }$ (red error bars) is comparatively uniform across neurons, indicating a largely consistent stochastic contribution, which we refer to as noise.

Across many neurons, $\sigma _ { i }$ is typically dominant, indicating that noise is the primary error source. However, for a subset of neurons, we find that $| \mu _ { i } | \gg$ $\sigma _ { i } .$ , indicating that systematic non-idealities can locally dominate and lead to significant yet structured deviations. This motivates the following investigation of the respective impacts of noise and systematic non-idealities on computations.

## 3.2 Temperature-Dependent Noise Behavior

In addition to the gain and error measurements described in Section 3.1, we investigate how these quantities evolve under varying temperature conditions. Temperature effects on lower-level BSS-2 circuit characteristics have previously been studied [1]; however, their impact on the presented gain and error metrics remains unclear.

![](images/ff2febc1475116c29d926e47e43bb7ee330f5ff67fbb2fc3229e74febf021f53.jpg)

To vary the operating temperature without hardware modifications, a heat gun was used (Fig. 4), while the device temperature was monitored using the on-board sensor. Due

Fig. 4: Experiment setup for temperature measurements with BSS-2.

to ambient conditions and self-heating, the baseline operating temperature was approximately $4 0 ^ { \circ } \mathrm { C }$ . The temperature was increased to about $9 0 ^ { \circ } \mathrm { C }$ over 6 min and then allowed to cool back to baseline over 10 min, covering an efective range of $5 0 ^ { \circ } \mathrm { C }$ in a total of 16 min. Measurements were recorded every 10 s, yielding nearly 100 measurement points.

As a first step, the impact of temperature changes on both noise and systematic non-idealities is analyzed. Fig. 3b illustrates the temperature-dependent evolution of the non-idealities for a subset of 8 neurons. At baseline temperature, most neurons exhibit near-optimal behavior, with mean errors close to zero. In contrast, the green dots correspond to an outlier neuron exhibiting a substantial mean error. With increasing temperature, systematic non-idealities shift with varying magnitude and direction across neurons, revealing heterogeneous temperature efects. While most neurons shift further from zero, indicating degraded behavior, the green outlier partially recovers.

![](images/017983770643f366a7d5f4de191a564e778bb43bd33480b6a329f82a4a405276.jpg)  
(a) Noise

![](images/7ca29434f9cf5334628afcc0282336bad37f020c11282c13b9377a0a234953b1.jpg)  
(b) Gain

![](images/306afef71e58659cbbf23736ee569606878ef61353234089b2a86aed62e4380a.jpg)  
(c) Noise-to-signal ratio  
Fig. 5: Temperature-dependent behavior of noise (left) and gain (middle) and noise-to-signal ratio (right).

To evaluate the impact of these deviations on signal quality, we computed the error-to-signal ratio (ESR), defined as the ratio of the absolute error to the absolute signal amplitude, where the error comprises both stochastic noise and systematic non-idealities. As shown in Fig. 3c, the mean ESR averaged across all neurons increases with temperature, indicating that elevated temperatures degrade signal fidelity.

We assume that both stochastic noise and systematic non-idealities contribute additively to the mean ESR, which allows us to disentangle their contributions. To this end, we first investigate the evolution of the noise and gain with increasing temperature. Fig. 5a shows the mean standard deviation σ, averaged across all neurons, while Fig. 5b depicts the corresponding gain $G _ { \mathrm { o p } }$ . Both quantities decrease with increasing temperature.

Despite this reduction, the efective noise-to-signal ratio (NSR), shown in Fig. 5c, increases with temperature. Importantly, the NSR captures only stochastic noise and excludes any changes in systematic non-idealities. When comparing the ESR results (Fig. 3c) with the NSR results (Fig. 5c), it becomes apparent that while both measurements trend in the same direction, the stochastic noise contribution to degraded signal quality at elevated temperatures is small and systematic non-idealities dominate.

## 4 Error Countermeasures

To further investigate how temperature-induced variations uncovered in Section 3 afect neural network performance, we train and deploy neural networks directly on the real hardware platform. This setup allows us to capture the full impact of hardware-in-the-loop dynamics under realistic operating conditions. To evaluate the efectiveness of diferent countermeasures, we conduct experiments across a range of operating temperatures, systematically assessing their ability to mitigate performance degradation under varying thermal noise conditions.

Table 1: Layer-wise network architecture and BSS-2 execution parameters.
<table><tr><td>Layer</td><td>Input Features</td><td>Output Features</td><td>Bias</td><td>No. of Resends</td><td>Gain</td><td>Noise Std.</td></tr><tr><td>1</td><td>2560</td><td>128</td><td>No</td><td>2</td><td>0.003</td><td>5.4</td></tr><tr><td>2</td><td>128</td><td>128</td><td>No</td><td>4</td><td>0.006</td><td>1.5</td></tr><tr><td>3</td><td>128</td><td>10</td><td>No</td><td>4</td><td>0.006</td><td>1.7</td></tr></table>

## 4.1 Dataset and Neural Network Architecture

Experiments were conducted on a ten-class classification task derived from the Google Speech Commands (GSC) dataset [16]. The audio samples are preprocessed to a compact yet informative mel-spectrogram representation in the form of a 64  40 feature matrix, making it well suited for deployment on a small-scale MLP architecture.

The neural network consists of three fully-connected layers. Since layers exceeding the hardware’s native computational capacity require multiple sequential executions, noise accumulates beyond single-pass estimates in these layers. Moreover, the observed noise depends not only on hardware properties but also on the concrete distribution of activations and weights, necessitating a layer-wise noise estimation. Table 1 summarizes the network architecture together with the expected noise standard deviation during execution on BSS-2 hardware, using samples from the target dataset together with weights from a quantization-aware pretrained model. The table further reports the resend counts and resulting gain factors. Bias terms are disabled, as they are unsupported on BSS-2.

## 4.2 Accuracy and Robustness Evaluation

We investigate the impact of diferent countermeasures on model accuracy and robustness. The evaluated methods can be grouped into two categories: simulationbased and hardware-based approaches. All models were trained using the Adam optimizer with a batch size of 128. The learning rate was initialized at $1 \times 1 0 ^ { - 3 }$ and gradually reduced to $1 \times 1 0 ^ { - 6 }$ using a cosine annealing schedule.

Simulation-Based Training For the simulation-based methods, we aimed to reproduce hardware efects entirely in software. To this end, a custom PyTorch layer was implemented to emulate execution on the BSS-2 system. The layer incorporates the hardware-specific quantization described in Section 2.1 with signed weights, injects noise, and scales outputs according to gain factors obtained from hardware characterization measurements. During backpropagation, quantization and noise injection are bypassed using a straight-through estimator (STE) [10], while only the scaling factors are retained and reapplied in the backward pass. All simulation-based methods were trained in full precision for the first ten epochs. Quantization-aware training (QAT) [10] and scaling efects were enabled from epoch 11 onward. The evaluated training strategies difer only in how noise is incorporated during training:

– Noise-Free $Q A T \colon$ No noise was injected during training. Quantization and scaling efects were simulated from epoch 11 onward, and training continued for a total of 300 epochs.

– Standard noisy training: Noise injection was enabled from epoch 21 onward, with the noise level gradually increasing until it reaches the target strength listed in Table 1 at epoch 121. Training then continued under full-noise conditions until epoch 300.

– Variance-Aware Noisy Training (VANT): The noise schedule was identical to that of standard noisy training. However, the target noise strength was sampled individually for each training sample. Following [15], we set $\sigma _ { \mathrm { t r a i n } }$ equal to the target noise strength and express θ as a scaling factor of $\sigma _ { \mathrm { t r a i n } }$ to account for layer-dependent efective noise levels.

Hardware-in-the-Loop Training In addition to simulation-based approaches, we performed hardware-in-the-loop training using the BSS-2 hardware during the forward pass. Since training entirely with hardware-in-the-loop execution resulted in prohibitively slow convergence, models were initialized from a network pretrained with standard noisy training and subsequently fine-tuned for 80 epochs using hardware execution in the forward path.

Hardware Calibration To assess the influence of calibration temperature, calibration settings were generated at $T _ { \mathrm { c a l } } \in \{ 4 0 ^ { \circ } \mathrm { C } , 6 0 ^ { \circ } \mathrm { C } , 8 0 ^ { \circ } \mathrm { C } \}$ , yielding three corresponding hardware-in-the-loop trained models. Hardware-in-the-loop training itself was performed at the baseline operating temperature of $4 0 ^ { \circ } \mathrm { C }$ , since training at elevated temperatures is impractical due to the associated time cost.

Temperature Robustness Evaluation Robustness to temperature variations was evaluated by performing inference at operating temperatures $\left( T _ { \mathrm { o p e r a t i n g } } \right)$ ranging from $4 0 ^ { \circ } \mathrm { C }$ to $9 0 ^ { \circ } \mathrm { C }$ . Fig. 7 shows the corresponding validation accuracy curves.

![](images/9d9586e70ca5479bf0b5f52618a656c42cce548ffef37999fd9007d4347e2140.jpg)

For all simulation-based training models, peak accuracy is observed near ${ \cal T } _ { \mathrm { o p e r a t i n g } } =$ $T _ { \mathrm { c a l } }$ , indicating that calibration is most efective close to the temperature at which it was generated. This trend is consistent with the ESR measurements in Fig. 6, where the min-

Fig. 6: Error-to-signal ratio over operating temperature $T _ { \mathrm { o p e r a t i n g } }$ for each calibration.

imum ESR occurs near the corresponding calibration temperature. Compared with noise-free QAT, both standard noisy training and VANT substantially improve robustness, with VANT providing an additional, albeit modest, benefit when the operating temperature deviates from the calibration point.

In contrast, the hardware-in-the-loop trained models exhibit markedly different behavior. Unlike the simulation-based approaches, the highest accuracies are not necessarily observed when ${ T _ { \mathrm { o p e r a t i n g } } = T _ { \mathrm { c a l } } }$ . Instead, the observed behavior depends on the calibration used during training, which may difer from the calibration applied during inference.

![](images/2c864f36b8b683fb876925c0373ef1fa4a3211f130845e22aaaf23ab563ef104.jpg)  
(a) Inference $T _ { \mathrm { c a l } } = 4 0 ^ { \circ } \mathrm { C }$

![](images/7dfa460a6ee2da8f1c4eaf748884313c9f9c5e249350b59aca5e83774a0a5211.jpg)  
(b) Inference $T _ { \mathrm { c a l } } = 6 0 ^ { \circ } \mathrm { C }$

![](images/ada1c13dd1be59d882e72ad3e63ed83e83c4997acc4aa26a9ad18cbe88811d19.jpg)  
(c) Inference $T _ { \mathrm { c a l } } = 8 0 ^ { \circ } \mathrm { C }$  
Fig. 7: Validation accuracy versus operating temperature for diferent training approaches under three hardware calibration conditions. Each subplot represents inference using one calibration created at $T _ { \mathrm { c a l } }$

Models trained with calibrations generated at elevated temperatures $\mathrm { ( > 4 0 ^ { \circ } C ) }$ tend to perform better below their calibration point, since hardware-in-the-loop training occurs at baseline temperature regardless of calibration, which enables the model to compensate for non-idealities expected in that regime. Above the calibration temperature, however, accuracy degrades rapidly as the training setting no longer reflects actual operating behavior.

As a result, using an inference calibration generated at the higher end of the temperature range, as shown in Fig. 7c, can provide high robustness across the entire investigated temperature range, as most operating temperature settings fall below the calibration point. In particular, the model trained using the $6 0 ^ { \circ } \mathrm { C }$ calibration demonstrates remarkable robustness.

Overall, we observe that in many cases the use of VANT is recommended as a first mitigation strategy for temperature-induced efects, as it delivers reasonable performance at no hardware training overhead. For further improvements, HW-in-the-loop training on a well-selected calibration can achieve better performance.

## 5 Summary and Outlook

This work investigates the impact of temperature on analog neural-network inference using BrainScaleS-2. The observed temperature-dependent efects manifest as gain drift, neuron-specific deviations, and changes in the signal-to-noise ratio. Together, these efects lead to measurable accuracy degradation that cannot be attributed to noise variations alone. We analyze the error-to-signal and noise-tosignal ratios to separate stochastic and systematic contributions. The analysis suggests that systematic non-idealities are the dominant source of degradation.

In exploring mitigation strategies for temperature-induced accuracy losses, we obtain three main findings. First, hardware-in-the-loop training generally produces the best-performing models, highlighting the importance of calibration awareness during both training and execution. Second, through the first application of VANT to real neuromorphic hardware, we observe that VANT generally outperforms standard noisy training, with the clearest benefits occurring when the operating temperature deviates from the calibration temperature. Moreover, under certain conditions, simulation-based training with VANT can achieve performance comparable to hardware-in-the-loop training. Third, recalibrating the chip around its expected operating temperature can shift the optimum operating point and significantly improve robustness. This trend appears to hold across all investigated training methods.

Looking ahead, one promising direction is the use of more expressive models, which have been shown to exhibit improved robustness in the context of VANT. Furthermore, systematic non-idealities exhibit a highly predictable drift as a function of temperature. We therefore propose that explicitly modeling the temperature dependence of these non-idealities could enable a substantially improved training procedure. Although such an approach requires per-chip calibration, it may provide a path toward deep neural networks that are better adapted to the temperature variations encountered in real-world deployment scenarios.

## Acknowledgments

We gratefully acknowledge the members of the Electronic Visions group for the design and provision of the chip used in this work, as well as for their technical support and access to the experimental infrastructure. We especially thank Johannes Schemmel for supporting this collaboration, and Yannik Stradmann for serving as our primary point of contact.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Billaudelle, S.: From transistors to learning systems: circuits and algorithms for brain-inspired computing. Ph.D. thesis, Universität Heidelberg (2022)

2. Brückerhof-Plückelmann, F., Borras, H., Klein, B., Varri, A., Becker, M., Dijkstra, J., Brückerhof, M., Wright, C.D., Salinga, M., Bhaskaran, H., Risse, B., Fröning, H., Pernice, W.: Probabilistic photonic computing with chaotic light. Nature Communications 15(1), 10445 (Dec 2024), https://doi.org/10.1038/ s41467-024-54931-6

3. Brückerhof-Plückelmann, F., Borras, H., Hulyal, S.U., Meyer, L., Ji, X., Hu, J., Sun, J., Klein, B., Ebert, F., Dijkstra, J., McRae, L., Schmidt, P., Kippenberg, T.J., Fröning, H., Pernice, W.: Uncertainty reasoning with photonic Bayesian machines. CoRR abs/2512.02217 (2025), https://arxiv.org/abs/2512.02217

4. Klein, B., Kuhn, L., Weis, J., Emmel, A., Stradmann, Y., Schemmel, J., Fröning, H.: Towards addressing noise and static variations of analog computations using eficient retraining. In: ECML-PKDD ITEM Workshop. Springer International Publishing (2021). https://doi.org/10.1007/978-3-030-93736-2\_32

5. Lin, X., Rivenson, Y., Yardimci, N.T., Veli, M., Luo, Y., Jarrahi, M., Ozcan, A.: All-optical machine learning using difractive deep neural networks. Science 361(6406), 1004–1008 (2018). https://doi.org/10.1126/science.aat8084

6. Meng, J., Shim, W., Yang, L., Yeo, I., Fan, D., Yu, S., Seo, J.s.: Temperatureresilient RRAM-based in-memory computing for DNN inference. IEEE Micro 42(1), 89–98 (2022). https://doi.org/10.1109/MM.2021.3131114

7. Monga, D.C., Singh, G., Numan, O., Adam, K., Andraud, M., Halonen, K.A.I.: TRIM: Thermal auto-compensation for resistive in-memory computing. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 45(2), 943–954 (2026). https://doi.org/10.1109/TCAD.2025.3586889

8. Murmann, B.: Mixed-signal computing for deep neural network inference. IEEE Transactions on Very Large Scale Integration (VLSI) Systems 29(1), 3–13 (2021). https://doi.org/10.1109/TVLSI.2020.3020286

9. Ortner, T., Petschenig, H., Vasilopoulos, A., Renner, R., Brglez, S., Limbacher, T., Pinero, E., Linares-Barranco, A., Pantazi, A., Legenstein, R.: Rapid learning with phase-change memory-based in-memory computing through learning-tolearn. Nature Communications 16(1), 1243 (2025). https://doi.org/10.1038/ s41467-025-56345-4

10. Roth, W., Schindler, G., Klein, B., Peharz, R., Tschiatschek, S., Fröning, H., Pernkopf, F., Ghahramani, Z.: Resource-eficient neural networks for embedded systems. Journal of Machine Learning Research 25(50), 1–51 (2024), http: //jmlr.org/papers/v25/18-566.html

11. Schemmel, J., Billaudelle, S., Dauer, P., Weis, J.: Accelerated analog neuromorphic computing. Advances in Analog Circuit Design pp. 83–102 (2021). https://doi. org/10.1007/978-3-030-91741-8\_6

12. Shen, Y., Harris, N.C., Skirlo, S., Prabhu, M., Baehr-Jones, T., Hochberg, M., Sun, X., Zhao, S., Larochelle, H., Englund, D., Soljačić, M.: Deep learning with coherent nanophotonic circuits. Nature Photonics 11(7), 441–446 (2017). https: //doi.org/10.1038/nphoton.2017.93

13. Stradmann, Y., Billaudelle, S., Breitwieser, O., Ebert, F.L., Emmel, A., Husmann, D., Ilmberger, J., Müller, E., Spilger, P., Weis, J., Schemmel, J.: Demonstrating analog inference on the BrainScaleS-2 mobile system. IEEE Open Journal of Circuits and Systems 3, 252–262 (2022). https://doi.org/10.1109/OJCAS.2022. 3208413

14. Wang, X., Borras, H., Klein, B., Fröning, H.: On hardening DNNs against noisy computations. In: AccML Workshop, collocated with HiPEAC Conference (2025), https://accml.dcs.gla.ac.uk/papers/2025/7th\_AccML\_paper\_1.pdf

15. Wang, X., Borras, H., Klein, B., Fröning, H.: Variance-aware noisy training: Hardening DNNs against unstable analog computations. In: European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. ECML-PKDD (2025), https://doi.org/10.1007/ 978-3-032-06109-6\_9

16. Warden, P.: Speech commands: A dataset for limited-vocabulary speech recognition. CoRR abs/1804.03209 (2018), http://arxiv.org/abs/1804.03209

17. Weis, J., Spilger, P., Billaudelle, S., Stradmann, Y., Emmel, A., Müller, E., Breitwieser, O., Grübl, A., Ilmberger, J., Karasenko, V., Kleider, M., Mauch, C., Schreiber, K., Schemmel, J.: Inference with artificial neural networks on analog neuromorphic hardware. In: ECML-PKDD ITEM Workshop. vol. 1325, pp. 201– 212. Springer International Publishing (2020)

18. Zhou, C., Kadambi, P., Mattina, M., Whatmough, P.N.: Noisy machines: Understanding noisy neural networks and enhancing robustness to analog hardware errors using distillation. CoRR abs/2001.04974 (2020), https://arxiv.org/abs/2001. 04974