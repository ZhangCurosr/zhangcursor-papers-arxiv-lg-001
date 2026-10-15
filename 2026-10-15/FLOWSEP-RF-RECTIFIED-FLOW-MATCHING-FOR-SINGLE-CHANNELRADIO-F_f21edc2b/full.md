# FLOWSEP-RF: RECTIFIED FLOW MATCHING FOR SINGLE-CHANNELRADIO-FREQUENCY SIGNAL SEPARATION

Mohamed Kamel, Tamer Nadeem

Department of Computer Science, Virginia Commonwealth University, Richmond, VA, USA {kamelms,tnadeem}@vcu.edu

## ABSTRACT

Single-channel radio-frequency (RF) signal separation is important for interference mitigation when a signal of interest overlaps with a co-channel interferer in time and frequency. Classical matched filtering and LMMSE receivers are limited under structured non-Gaussian interference, while diffusion and score-based methods often require iterative sampling or optimization. We propose FlowSep-RF, a rectified flow matching framework that learns a clean-endpoint transport from the received mixture to the signal of interest. The model combines time-domain IQ samples with frequencydomain features and recovers the separated waveform using one endpoint velocity evaluation. We evaluate FlowSep-RF on QPSK mixtures with OFDM-BPSK, OFDM-QPSK, and RFChallenge CommSignal2 interference against classical, discriminative, and generative baselines. FlowSep-RF achieves the lowest average BER and MSE among evaluated baselines across all three settings with one-step inference.

Index Terms— RF signal separation, rectified flow matching, co-channel interference, wireless communications, generative signal processing

## 1. INTRODUCTION

Co-channel interference is a central challenge in modern wireless systems. In single-channel RF receivers, the signal of interest (SOI) and an interfering waveform may overlap in time and frequency, while spatial filtering is unavailable. The receiver must therefore recover the desired digital signal directly from a mixed baseband waveform.

Classical receivers such as matched filtering and linear minimum mean-square error (LMMSE) filtering are effective when the impairment is well modeled as additive Gaussian noise. However, interference from another communication system is structured and non-Gaussian, so linear filtering may fail to exploit waveform-specific structure. Recent RF source separation benchmarks, including the RF Challenge, have therefore motivated data-driven methods that operate on raw complex IQ samples and evaluate separation using waveform fidelity and downstream demodulation accuracy [1, 2].

Deep neural separators have shown strong performance for RF interference rejection. Prior work exploits temporal structure in cyclostationary signals, while recent U-Net-based and Transformer-based models learn direct mappings from mixtures to separated signals [3, 4, 5]. These models provide fast inference, but direct regression does not explicitly model the transport from a corrupted mixture to a clean SOI.

Generative models provide another direction for signal separation. Diffusion and score-based models have been used as learned priors for inverse problems and source separation [6, 7, 8]. In RF separation, score-based methods such as randomized Gaussian smoothing exploit the discrete structure of communication signals [9]. Related work has also used score-based or diffusion priors for radar interference mitigation and Bayesian signal separation [10, 11]. However, these approaches commonly require iterative sampling, unfolding, or optimization, which can limit their use in low-latency receivers.

Flow matching and rectified flow offer a more efficient generative formulation by learning a velocity field along a path between source and target distributions, avoiding long reverse diffusion chains [12, 13, 14]. Recent signalprocessing work has used flow-based models for ECG generation, speech enhancement, and real-time speech restoration [15, 16, 17, 18].

In this paper, we propose FlowSep-RF, a rectified flow matching framework for single-channel RF signal separation. FlowSep-RF learns a clean-endpoint transport from the received mixture to the SOI, rather than predicting and subtracting the interference. The model combines time-domain IQ modeling with frequency-domain features and recovers the separated waveform using one neural function evaluation.

The main contributions of this work are as follows:

• We formulate single-channel RF signal separation as a clean-endpoint rectified flow matching problem from the received mixture to the SOI.

• We introduce a dual-domain FlowSep-RF architecture that combines time-domain IQ features with frequencydomain structure for one-step RF separation.

• We evaluate FlowSep-RF on QPSK SOI mixtures with OFDM-BPSK, OFDM-QPSK, and RFChallenge CommSignal2 interference, showing improved BER and MSE over classical, discriminative, and generative baselines.

## 2. PROBLEM FORMULATION

We consider single-channel complex baseband RF signal separation. Let $s ~ \in ~ \mathbb { C } ^ { N }$ denote the signal of interest (SOI), $b \in \mathbb { C } ^ { N }$ a co-channel interferer, and $\boldsymbol { y } \in \mathbb { C } ^ { N }$ the received mixture. The mixture is modeled as

$$
y = s + \kappa b ,\tag{1}
$$

where $\kappa \in \mathbb { R } _ { + }$ controls the interference scale. The corresponding signal-to-interference ratio (SIR) is

$$
\gamma _ { \mathrm { S I R } } = 1 0 \log _ { 1 0 } \frac { \| s \| _ { 2 } ^ { 2 } } { \| \kappa b \| _ { 2 } ^ { 2 } } .\tag{2}
$$

For unit-power sources, this reduces to $\gamma _ { \mathrm { S I R } } = 1 0 \log _ { 1 0 } ( 1 / \kappa ^ { 2 } )$

Given y, the goal is to recover an estimate sˆ of the SOI. Since the true interference scale is unavailable in blind receivers, FlowSep-RF uses a power-based estimate $\hat { \kappa } ( y )$ , with $\begin{array} { r } { \hat { P } _ { y } = N ^ { - 1 } \sum _ { n = 1 } ^ { N } | y [ n ] | ^ { 2 } } \end{array}$ and $\hat { \kappa } = \sqrt { \operatorname* { m a x } ( 0 , ( \hat { P } _ { y } - \hat { P } _ { s } ) / \hat { P } _ { b } ) }$ where $\hat { P } _ { s }$ and $\hat { P } _ { b }$ are nominal source and interference powers. The corresponding estimated SIR is $\hat { \gamma } _ { \mathrm { S I R } } = 1 0 \log _ { 1 0 } ( 1 / \hat { \kappa } ^ { 2 } )$ During training, the conditioning inputs are jittered as $\tilde { \kappa } =$ $\kappa \xi .$ , with $\xi \sim \mathcal { U } ( 0 . 9 , 1 . 1 )$ , to improve robustness to estimation error. Each complex waveform is represented by its real and imaginary components in $\mathbb { R } ^ { 2 \times N }$

The SOI carries a transmitted bit sequence $\mathbf { u } \in \{ 0 , 1 \} ^ { L }$ After separation, the recovered waveform is demodulated as $\hat { \mathbf { u } } = \mathcal { Q } ( \hat { s } )$ . We evaluate waveform fidelity using MSE and communication reliability using BER:

$$
\begin{array} { r l } & { \mathrm { M S E } = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \vert \hat { s } [ n ] - s [ n ] \vert ^ { 2 } , } \\ & { \mathrm { B E R } = \displaystyle \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \mathbf { 1 } \big \lbrace \hat { u } _ { \ell } \neq u _ { \ell } \big \rbrace . } \end{array}\tag{3}
$$

BER is the primary metric because RF separation is ultimately judged by recovery of the transmitted information, while MSE measures waveform reconstruction accuracy.

## 3. FLOWSEP-RF

FlowSep-RF separates the SOI by learning a direct transport from the received mixture to the clean waveform, as shown in Fig. 1. Unlike interference-prediction methods that estimate κb and subtract it from y, FlowSep-RF treats the mixture as the initial endpoint and the SOI as the target endpoint. This gives a clean-endpoint rectified flow formulation aligned with the RF separation objective.

## 3.1. Clean-endpoint rectified flow

For each training example, we set the endpoints to the received mixture and clean SOI and sample $t \sim \mathcal { U } ( 0 , 1 )$

$$
\begin{array} { l l } { { x _ { 0 } = y , \qquad x _ { 1 } = s , \qquad x _ { t } = ( 1 - t ) y + t s , } } \\ { { \qquad v ^ { \star } = x _ { 1 } - x _ { 0 } = s - y . } } \end{array}\tag{4}
$$

FlowSep-RF learns a conditional velocity field

$$
v _ { \theta } = v _ { \theta } ( x _ { t } , y , t , \tilde { \gamma } _ { \mathrm { S I R } } , \tilde { \kappa } ) ,\tag{5}
$$

where $\tilde { \kappa }$ denotes the jittered scale parameter during training and the power-based estimate $\hat { \kappa } ( y )$ during inference. The mixture y is provided as conditioning information at all times. At inference, the learned field is evaluated at the mixture endpoint, $t = 0$ , and the separated signal is obtained with one neural function evaluation:

$$
\hat { s } = y + v _ { \theta } ( y , y , 0 , \hat { \gamma } _ { \mathrm { S I R } } , \hat { \kappa } ) .\tag{6}
$$

This one-step endpoint update distinguishes FlowSep-RF from iterative diffusion or score-based separators.

## 3.2. Dual-domain architecture

RF mixtures contain useful structure in both time-domain IQ samples and the frequency domain. The time branch receives the real-valued representation of the interpolated sample and mixture,

$$
z _ { \mathrm { t i m e } } = [ x _ { t } , y ] \in \mathbb { R } ^ { 4 \times N } ,\tag{7}
$$

where the four channels correspond to the real and imaginary components of $x _ { t }$ and $y .$ . In parallel, the frequency branch applies the discrete Fourier transform to $x _ { t }$ and y and forms

$$
z _ { \mathrm { f r e q } } = [ \Re X _ { t } , \Im X _ { t } , \Re Y , \Im Y , \log ( | X _ { t } | + \epsilon ) , \log ( | Y | + \epsilon ) ] ,\tag{8}
$$

where $X _ { t } = \mathcal { F } ( x _ { t } ) , Y = \mathcal { F } ( y )$ , and ϵ is a small constant for numerical stability. Both branches are processed by convolutional residual blocks. The time embedding and scalar RF conditions are encoded as $c = \phi ( t , \tilde { \gamma } _ { \mathrm { S I R } } , \tilde { \kappa } )$ and injected into the residual blocks through feature-wise modulation. The network predicts a time-domain velocity $v _ { \mathrm { t i m e } }$ and a frequencydomain velocity $v _ { \mathrm { f r e q } } ,$ where the latter is mapped back to the time domain using the inverse Fourier transform. The final velocity is

$$
v _ { \theta } = v _ { \mathrm { t i m e } } + \alpha v _ { \mathrm { f r e q } } ,\tag{9}
$$

where $\alpha$ is a learned scalar gate initialized to give a moderate frequency-domain contribution.

## 3.3. Training objective

The main flow-matching loss trains the model to predict the clean-endpoint velocity along the rectified path:

$$
\mathcal { L } _ { \mathrm { r f } } = \mathbb { E } _ { t } \big [ w ( \gamma _ { \mathrm { S I R } } ) \big | \big | v _ { \theta } ( x _ { t } , y , t , \tilde { \gamma } _ { \mathrm { S I R } } , \tilde { \kappa } ) - ( s - y ) \big | \big | _ { 1 } \big ] .\tag{10}
$$

![](images/4673f76076ff71561946d104101ee4042147cf47337f87bae87dab2263d01221.jpg)  
Fig. 1. FlowSep-RF overview. Training forms a clean-endpoint path $x _ { t } = ( 1 - t ) y + t s$ from the mixture y to the clean SOI s and learns $v ^ { \star } = s - y$ using time- and frequency-domain features conditioned on RF variables. Inference uses estimated RF conditions and one endpoint update, $\hat { s } = y + v _ { \theta } ( y , y , 0 , \hat { \gamma } _ { \mathrm { S I R } } , \hat { \kappa } )$ , followed by demodulation for BER evaluation.

We also apply an endpoint loss at $t = 0$

$$
\mathcal { L } _ { \mathrm { e n d } } = w ( \gamma _ { \mathrm { S I R } } ) \left| \left| v _ { \theta } ( y , y , 0 , \tilde { \gamma } _ { \mathrm { S I R } } , \tilde { \kappa } ) - ( s - y ) \right| \right| _ { 1 } ,\tag{11}
$$

with endpoint reconstruction $\hat { s } _ { 0 } = y + v _ { \theta } ( y , y , 0 , \tilde { \gamma } _ { \mathrm { S I R } } , \tilde { \kappa } )$ . To encourage waveform and receiver-level fidelity, we use clean waveform, symbol-proxy, and spectral magnitude losses:

$$
\mathcal { L } _ { \mathrm { w f } } = w ( \gamma _ { \mathrm { S I R } } ) \Vert \hat { s } _ { 0 } - s \Vert _ { 1 } ,\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { s y m } } = w ( \gamma _ { \mathrm { S I R } } ) \Vert \mathcal { P } ( \hat { s } _ { 0 } ) - \mathcal { P } ( s ) \Vert _ { 1 } ,\tag{13}
$$

$$
\mathcal { L } _ { \mathrm { s p e c } } = w ( \gamma _ { \mathrm { S I R } } ) \Vert \Psi ( \hat { s } _ { 0 } ) - \Psi ( s ) \Vert _ { 1 } .\tag{14}
$$

Here, $\mathcal { P } ( \cdot )$ denotes the receiver-compatible symbol projection used as a differentiable proxy for demodulation quality, and $\Psi ( \cdot )$ denotes the log-magnitude Fourier representation. The final training loss is

$$
\mathcal { L } = \lambda _ { \mathrm { r f } } \mathcal { L } _ { \mathrm { r f } } + \lambda _ { \mathrm { e n d } } \mathcal { L } _ { \mathrm { e n d } } + \lambda _ { \mathrm { w f } } \mathcal { L } _ { \mathrm { w f } } + \lambda _ { \mathrm { s y m } } \mathcal { L } _ { \mathrm { s y m } } + \lambda _ { \mathrm { s p e c } } \mathcal { L } _ { \mathrm { s p e c } } .\tag{15}
$$

The SIR weight $w ( \gamma _ { \mathrm { S I R } } )$ up-weights difficult low-SIR and mid-SIR mixtures relative to easier high-SIR examples.

## 4. EXPERIMENTAL SETUP

## 4.1. Datasets

We evaluate FlowSep-RF on three single-channel RF separation settings. In all experiments, the SOI is an RRC-filtered QPSK waveform and the received mixture is generated using Eq. (1). The first two settings use synthetic OFDM interference with BPSK and QPSK modulation, denoted OFDM-BPSK and OFDM-QPSK. The third setting uses real RFChallenge CommSignal2 interference, covering recorded RF interference in addition to synthetic structured interferers.

Each example contains N = 2560 complex IQ samples, with 160 QPSK symbols and 320 transmitted bits. Mixtures are generated over SIR values of $- 2 4 , - 2 1 , - 1 8 , - 1 5 , - 1 2 ,$ $- 9 , - 6 .$ and −3 dB. For each interference type, we use 200k training mixtures, 20k validation mixtures, and a held-out test grid with 400 examples per SIR level. Test examples are not used during training or model selection.

## 4.2. Baselines

We compare FlowSep-RF with classical, discriminative, and generative baselines. The classical baselines are matched filtering (MF) and LMMSE followed by matched filtering. The discriminative baselines are U-Net-RF and DirectComm-RF, which use one-pass neural inference. The generative baselines are DDPM, reverse diffusion, and α-RGS, a score-based RF source separation method [9]. These baselines cover linear receivers, fast supervised separators, and iterative generative separation methods.

## 4.3. Implementation details

FlowSep-RF uses the dual-domain architecture described in Section 3, with $d _ { \mathrm { m o d e l } } ~ = ~ 1 9 2$ hidden channels and eight residual blocks. We train with AdamW using a learning rate of $2 \times 1 0 ^ { - 4 }$ , batch size 96, and loss weights $\lambda _ { \mathrm { r f } } = 1 . 0 , \lambda _ { \mathrm { e n d } } =$ 1. $. 0 , \lambda _ { \mathrm { w f } } = 0 . 1 0 , \lambda _ { \mathrm { s y m } } = 0 . 2 5$ , and $\lambda _ { \mathrm { s p e c } } = 0 . 0 5$ . Low-SIR and mid-SIR examples are up-weighted during training. All models are trained for 20 epochs, and the checkpoint with the best validation performance is used for testing.

At inference, FlowSep-RF applies Eq. (6) with one endpoint velocity evaluation. The conditioning variables use the power-based estimates described in Section 2, while multiplicative jitter during training improves robustness to estimation error. The U-Net-RF and DirectComm-RF baselines use comparable one-pass neural inference, while DDPM, reverse diffusion, and α-RGS require iterative sampling or optimization.

![](images/db8aad17a0be16dae2a0b982c6d38e12c1d3208fbab243b1410316994b099b18.jpg)  
(a) BER, OFDM-BPSK

![](images/edcac0c3681e530332f881cfe69c774cd8214a4d107dccdc07d9d73362b70106.jpg)  
(b) BER, OFDM-QPSK

![](images/6f452d6e7b8ccb013d9a9dcb9a55fd474f7346c3de3c64b2e51575b706b5a9f8.jpg)  
(c) BER, CommSignal2

![](images/61f3c716750029810cf685a15d1e7d7353ec0fdeff0a883036b327ebe12e3e36.jpg)  
(d) MSE, OFDM-BPSK

![](images/e986ad125c174a161f48539056bda7628371c3796bd64644c09e5d999c30398e.jpg)  
(e) MSE, OFDM-QPSK

![](images/7b29c3c7b4872505a43d29ac60f052ee5b25a78365bebb208aa17a7f5e698965.jpg)  
(f) MSE, CommSignal2  
Fig. 2. BER and MSE versus SIR for FlowSep-RF and baseline methods on the three interference settings. The top row reports BER after demodulation, and the bottom row reports waveform MSE. Lower is better.

Table 1. Main comparison on QPSK SOI separation. BER/MSE are averaged over the test SIR grid. Lower is better; dash indicates unavailable or non-applicable.
<table><tr><td rowspan="2">Method</td><td colspan="2">OFDM-BPSK</td><td colspan="2">OFDM-QPSK</td><td colspan="2">CommSignal2</td></tr><tr><td>BER</td><td>MSE</td><td>BER</td><td>MSE</td><td>BER</td><td>MSE</td></tr><tr><td>MF</td><td>0.186716</td><td></td><td>0.189842</td><td></td><td>0.196826</td><td></td></tr><tr><td>LMMSE + MF</td><td>0.186716</td><td>0.904915</td><td>0.189842</td><td>0.904157</td><td>0.196826</td><td>0.904333</td></tr><tr><td>Reverse Diffusion</td><td>0.015707</td><td>0.066848</td><td>0.037840</td><td>0.166426</td><td>0.387684</td><td>0.511738</td></tr><tr><td>α-RGS</td><td>0.001711</td><td>0.007473</td><td>0.008855</td><td>0.021780</td><td>0.066684</td><td>0.139131</td></tr><tr><td>DirectComm-RF</td><td>0.013484</td><td>28.138059</td><td>0.006269</td><td>28.037159</td><td>0.027385</td><td>7.222556</td></tr><tr><td>DDPM</td><td>0.000908</td><td>0.003512</td><td>0.004946</td><td>0.019559</td><td>0.026872</td><td>0.107607</td></tr><tr><td>U-Net-RF</td><td>0.000851</td><td>0.002324</td><td>0.005512</td><td>0.012761</td><td>0.078125</td><td>0.077119</td></tr><tr><td>FlowSep-RF</td><td>0.000645</td><td>0.001470</td><td>0.003776</td><td>0.006231</td><td>0.023066</td><td>0.034845</td></tr></table>

We report BER, MSE, and inference efficiency; BER and MSE are defined in Eq. (3), while efficiency is measured by test-time neural function evaluations or optimization steps.

## 5. RESULTS

Table 1 reports average BER and MSE over the test SIR grid. FlowSep-RF achieves the lowest average BER and MSE in all three settings using the same configuration. On OFDM-BPSK, it achieves a BER of $6 . 4 5 \times 1 0 ^ { - 4 }$ and an MSE of $1 . 4 7 \times 1 0 ^ { - 3 }$ . On OFDM-QPSK and CommSignal2, it also outperforms DDPM, U-Net-RF, DirectComm-RF, Reverse Diffusion, and α-RGS. Fig. 2 shows that this trend holds across the SIR grid.

Inference efficiency. FlowSep-RF requires one neural function evaluation at the mixture endpoint, matching the one-pass inference cost of U-Net-RF and DirectComm-RF. In contrast, DDPM uses 25 denoising steps, reverse diffusion is iterative, and α-RGS uses 2000 test-time optimization steps.

Table 2. Ablation study. BER and MSE are averaged over the test SIR grid. Lower is better.
<table><tr><td rowspan="2">Variant</td><td colspan="2">OFDM-BPSK</td><td colspan="2">OFDM-QPSK</td><td colspan="2">CommSignal2</td></tr><tr><td>BER</td><td>MSE</td><td>BER</td><td>MSE</td><td>BER</td><td>MSE</td></tr><tr><td>Full FlowSep-RF</td><td>0.000645</td><td>0.001470</td><td>0.003776 0.006231</td><td></td><td></td><td>0.0230660.034845</td></tr><tr><td>w/o rectified flow (direct reg.)</td><td>0.000920</td><td>0.002150</td><td>0.005120</td><td>0.009410</td><td>0.031200</td><td>0.045100</td></tr><tr><td>w/o symbol-proxy loss</td><td>0.000782</td><td>0.001615</td><td>0.004120</td><td>0.006892</td><td>0.025140</td><td>0.038102</td></tr><tr><td>w/o spectral loss</td><td>0.0007150.001850</td><td></td><td>0.004050 0.007420 0.024800 0.040150</td><td></td><td></td><td></td></tr><tr><td>w/o SIR weighting</td><td>0.000810 0.001740 0.004310 0.007100 0.026100 0.039800</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Ablation study. Table 2 reports the ablation study. The direct-regression variant uses the same backbone but replaces velocity matching with direct mapping from y to s. Removing the rectified-flow objective or any auxiliary component degrades BER and MSE across all three settings. The symbol-proxy, spectral, and SIR-weighting terms improve receiver accuracy, frequency agreement, and low-SIR robustness, respectively.

## 6. CONCLUSION

We presented FlowSep-RF, a rectified flow matching framework for single-channel RF signal separation. FlowSep-RF learns a clean-endpoint transport from the received mixture to the SOI, combines time- and frequency-domain features, and performs one-step inference. Experiments on QPSK mixtures with synthetic OFDM and RFChallenge CommSignal2 interference show the lowest average BER and MSE among evaluated baselines, and ablations confirm the value of the rectified-flow objective and receiver-aware, spectral, and SIRaware training terms. These results support clean-endpoint rectified flow for structured co-channel interference.

## Compliance with Ethical Standards

This is a numerical simulation study for which no ethical approval was required. No funding was received for this study. The authors have no relevant financial or nonfinancial interests to disclose.

## 7. REFERENCES

[1] Alejandro Lancho, Amir Weiss, Gary C. F. Lee, Tejas Jayashankar, Binoy G. Kurien, Yury Polyanskiy, and Gregory W. Wornell, “RF challenge: The data-driven radio frequency signal separation challenge,” IEEE Open Journal of the Communications Society, vol. 6, pp. 4083–4100, 2025.

[2] David Tse and Pramod Viswanath, Fundamentals of Wireless Communication, Cambridge University Press, 2005.

[3] Gary C. F. Lee, Amir Weiss, Alejandro Lancho, Jennifer Tang, Yuheng Bu, Yury Polyanskiy, and Gregory W. Wornell, “Exploiting temporal structures of cyclostationary signals for data-driven single-channel source separation,” in 2022 IEEE International Workshop on Machine Learningfor Signal Processing (MLSP), 2022, pp. 1–6.

[4] Olaf Ronneberger, Philipp Fischer, and Thomas Brox, “U-Net: Convolutional networks for biomedical image segmentation,” 2015.

[5] Egor Lifar, Semyon Savkin, Rachana Madhukara, Tejas Jayashankar, Yury Polyanskiy, and Gregory W. Wornell, “The radio-frequency transformer for signal separation,” 2026.

[6] Jonathan Ho, Ajay Jain, and Pieter Abbeel, “Denoising diffusion probabilistic models,” in Proceedings of the 34th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2020, NIPS ’20, Curran Associates Inc.

[7] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole, “Score-based generative modeling through stochastic differential equations,” 2021.

[8] Vivek Jayaram and John Thickstun, “Source separation with deep generative priors,” in Proceedings ofthe 37th International Conference on Machine Learning. 2020, ICML’20, JMLR.org.

[9] Tejas Jayashankar, Gary C. F. Lee, Alejandro Lancho, Amir Weiss, Yury Polyanskiy, and Gregory W. Wornell, “Score-based source separation with applications to digital communication signals,” 2024.

[10] Xinyi Wei, Jihwan Youn, Jeroen Overdevest, Jun Li, Satish Ravindran, and Ruud J.G. van Sloun, “Deep unfolding using score-based generative networks for automotive radar interference mitigation,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1– 5.

[11] Yi Zhang, Rui Guo, and Yonina C. Eldar, “Bayesian signal separation via plug-and-play diffusion-within-gibbs sampling,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 16–20.

[12] Xingchao Liu, Chengyue Gong, and Qiang Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in International Conference on Learning Representations, 2023.

[13] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le, “Flow matching for generative modeling,” 2023.

[14] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio, “Conditional flow matching: Simulation-free dynamic optimal transport,” 2023.

[15] Yuta Nambu, Masahiro Kohjima, and Ryuji Yamamoto, “Cardioflow: Learning to generate ecg from ppg with rectified flow,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[16] Naisong Zhou, Saisamarth Rajesh Phaye, Milos Cernak, Tijana Stojkovic, Andy Pearce, Andrea Cavallaro, and ´ Andy Harper, “Shortcut flow matching for speech enhancement: Step-invariant flows via single stage training,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 15842–15846.

[17] Tsun-An Hsieh and Sebastian Braun, “Towards realtime generative speech restoration with flow-matching,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 15847–15851.

[18] Tsun-An Hsieh and Minje Kim, “Adaptive deterministic flow matching for target speaker extraction,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 15412–15416.