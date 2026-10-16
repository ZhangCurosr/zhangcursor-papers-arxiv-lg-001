# Fault Detection and Bit-Error Rate Diagnostics in Optical Wireless Links Using Edge-AI

Sarthak Bhattarai, Sheshadri Shikhar Bandyopadhyay, Suprabh Dhakal, Jitendra Kumar Manandhar

Department of Electronics and Computer Engineering

Pulchowk Campus, Institute of Engineering, Tribhuvan University

Lalitpur, Nepal

Email: {080bei038.sarthak, 080bei040.sheshadri, 080bei043.suprabh}@pcampus.edu.np, mejiten@ioe.edu.np

Abstract—Optical Wireless Communication (OWC) systems provide high-bandwidth, spatially secure, and RF-interferencefree digital transmission. However, direct free-space optical links suffer from extreme sensitivity to line-of-sight disruptions, optical attenuation, high ambient light saturation, and transmitterreceiver misalignment. Traditional channel diagnostic frameworks rely on external computing nodes or cloud-based processing, which introduces severe diagnostic latency and reduces edge node autonomy. This paper presents an autonomous, selfdiagnosing optical wireless receiver node that integrates physicallayer signal telemetry extraction with an on-device TinyML classification engine. Built around an ESP32-S3 microcontroller, the receiver recovers 1000 bit/s On-Off Keying (OOK) digital payloads using a 5-sample majority voting mechanism while concurrently extracting real-time channel features: Bit Error Rate (BER), received signal voltage (V<sub>signal</sub>), Signal-to-Noise Ratio (SNR<sub>dB</sub>), and ambient light baseline (L<sub>ambient</sub>). Evaluated against a Random Forest baseline (77.22% accuracy) and a floatingpoint neural network (78.48% accuracy), a post-training INT8 quantized TensorFlow Lite Micro model occupying only 3.32 kB of SRAM runs directly on the receiver firmware. The quantized model achieves 74.68% overall accuracy across four distinct operational conditions: normal transmission, optical filter attenuation, ambient light interference, and severe obstruction/misalignment. On-device deployment tests confirm real-time, deterministic fault classification, establishing a practical approach for self-aware optical IoT nodes.

Index Terms—Optical Wireless Communication, Edge-AI, TinyML, Bit Error Rate, Link Diagnostics, ESP32-S3, Tensor-Flow Lite Micro, Random Forest, Line-of-Sight.

## I. INTRODUCTION

Optical Wireless Communication (OWC), encompassing Visible Light Communication (VLC) and Free-Space Optics (FSO), uses optical spectrum frequencies to transmit digital data [1]. Offering immunity to radio-frequency (RF) interference, high spatial reuse, and strict spatial confinement, OWC serves as an ideal supplement to congested indoor RF networks [3]. However, narrow-beam directed optical links depend on strict Line-of-Sight (LOS) alignment between the optical emitter and photodetector [2]. Physical misalignments, transient human or structural blockages, optical signal attenuation, and high background illumination degrade communication quality or cause silent channel failure [4].

In conventional OWC architectures, physical-layer health parameters (e.g., Bit Error Rate or Received Signal Strength) remain unmonitored or offload to external host computers or cloud infrastructure [2]. Offloading introduces network transmission overhead, processing latency, and total dependency on external infrastructure. Figure 1 demonstrates our physical experimental testbed designed to eliminate external computing dependencies.

![](images/d3aab8cfc4550b04186afc644edffcdcfb527d74f78591c10e199d622953673b.jpg)  
Fig. 1. Complete physical hardware setup of the autonomous optical wireless communication testbed.

![](images/088feec8be10af59240c33a43e670b4a40f6d62106657a58105eccda6b6da12a.jpg)  
Fig. 2. Transmitter circuit schematic showing GPIO interfacing, transistor switching logic, and laser driver circuit.

To resolve these challenges, we present a fully integrated, edge-based optical communication and diagnostic system. Operating entirely on an ultra-low-power ESP32-S3 microcontroller, the receiver recovers digital data payloads while extracting physical-layer telemetry parameters [7]. The physical circuit schematics of the transmitter and receiver front-ends are shown in Fig. 2 and Fig. 3, respectively. A lightweight neural network quantized to 8-bit integers (INT8) executes directly on the receiver firmware using TensorFlow Lite Micro, enabling real-time, autonomous fault classification without external computation.

![](images/896f00ef9afb9a1d7c9f5dbaa3bce98246370c4eed108387735deb3a0607c037.jpg)  
Fig. 3. Receiver circuit schematic depicting BPW34 PIN photodiode photodetector, LM358 op-amp conditioning, and ESP32-S3 ADC acquisition.

## II. LITERATURE REVIEW

Research in Optical Wireless Communication traditionally focuses on modulation efficiency, channel capacity, and transmission throughput. Khan [1] provides an extensive architectural review of VLC systems, highlighting Line-of-Sight constraints, ambient light interference, and link vulnerability as primary operational challenges.

Machine learning applications in optical links show significant performance improvements in channel estimation and adaptive modulation. Purnita and Mondal [2] apply supervised machine learning to DCO-OFDM LiFi links to analyze optical communication parameters and enhance channel capacity under varying ambient noise. Similarly, Sejan et al. [3] survey MIMO-VLC links and emphasize the necessity of intelligent adaptation mechanisms to manage spatial receiver alignment errors and optical power fluctuations.

Adaptive transmission control forms another significant research direction. Salman et al. [4] develop a machine learning framework that predicts DC bias and adaptive transmission parameters for indoor OFDM LiFi networks, maintaining target signal quality under variable channel conditions. In FSO links, Nistazakis, Tsiftsis, and Tombras [5] evaluate physical channel impairments and atmospheric turbulence, demonstrating that signal voltage degradation and SNR drops serve as primary indicators of physical link quality.

Despite these advancements, conventional channel monitoring architectures offload computation to centralized processing units. You et al. [6] demonstrate that centralized computation offloading introduces diagnostic transmission latency and processing energy consumption overhead, severely limiting edge node autonomy.

The emergence of Tiny Machine Learning (TinyML) shifts execution directly to low-power microcontrollers. Warden and Situnayake [7] establish that quantized neural networks execute on resource-constrained hardware with minimal memory footprints. David et al. [8] demonstrate TensorFlow Lite Micro deployment on ARM Cortex and ESP32 targets, achieving submillisecond inference times for local sensor classification.

In physical-layer monitoring, Dutta and Roy [9] investigate real-time signal diagnostics and fault detection in embedded communication nodes. However, past studies focus primarily on RF channels or high-power computing units.

A research gap remains in combining physical-layer optical metrics $( V _ { \mathrm { s i g n a l } }$ , BER, SNR, $L _ { \mathrm { a m b i e n t } } )$ with quantized TinyML models deployed directly on microcontroller receivers. Our work fills this gap by building an autonomous ESP32-S3 optical receiver that recovers data payloads and classifies four physical channel conditions locally.

## III. SYSTEM ARCHITECTURE AND MATHEMATICAL MODELING

The system architecture consists of three principal modules: an optical transmitter node, a free-space optical propagation channel, and an edge-diagnostic optical receiver node, as illustrated in Fig. 4.

![](images/341a6d1af0dcf7f2fe0e73bed2315346a84dbe98305b7389508c3eed3b585847.jpg)  
Fig. 4. Overall framework of the autonomous OWC testbed illustrating transmitter, optical channel, and receiver-side TinyML inference engine.

## A. Optical Data Representation and Modulation

The system employs On-Off Keying (OOK) modulation with Non-Return-to-Zero Level (NRZ-L) line coding. The relationship between bit rate $R _ { b }$ and bit duration $T _ { b }$ follows:

$$
R _ { b } = \frac { 1 } { T _ { b } }\tag{1}
$$

For $R _ { b } = 1 0 0 0 ~ \mathrm { { b i t / s } }$ , each bit occupies $T _ { b } = 1$ ms. A logical $\ ' _ { 1 } \ '$ corresponds to laser activation for the full bit duration, whereas a logical $\overrightarrow { \mathbf { \nabla } } 0 ^ { \circ }$ corresponds to laser deactivation.

## B. Physical-Layer Telemetry Extraction

During each transmission sequence, the receiver extracts four diagnostic features forming a feature vector $\begin{array} { r l } { \mathbf { x } } & { { } = } \end{array}$ $[ V _ { \mathrm { s i g n a l } } , \mathrm { B E R } , \mathrm { S N R _ { \mathrm { d B } } } , L _ { \mathrm { a m b i e n t } } ] ^ { T }$ :

1) Bit Error Rate (BER): The receiver evaluates BER by comparing recovered bits against a known reference payload of $N _ { t }$ total bits:

$$
\mathrm { B E R } = { \frac { N _ { e } } { N _ { t } } }\tag{2}
$$

where $N _ { e }$ represents the count of bit mismatches.

2) Received Signal Voltage $( V _ { s i g n a l } ) { : }$ The effective optical signal magnitude S equals the difference between the active signal ADC measurement $\bar { X } _ { \mathrm { s i g n a l } }$ and the ambient baseline measurement $\bar { X } _ { \mathrm { a m b i e n t } } .$

$$
S = { \bar { X } } _ { \mathrm { s i g n a l } } - { \bar { X } } _ { \mathrm { a m b i e n t } }\tag{3}
$$

The receiver converts $S$ to voltage $V _ { \mathrm { s i g n a l } }$ using internal 12-bit ADC reference calibration (0–3.3 V).

3) Ambient Light Baseline $( L _ { a m b i e n t } ) { : }$ The receiver samples N background ADC values during transmitter-OFF intervals:

$$
L _ { \mathrm { a m b i e n t } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } X _ { i }\tag{4}
$$

4) Signal-to-Noise Ratio $( S N R _ { d B } ) { : }$ The receiver estimates noise power from the sample standard deviation $\sigma _ { n }$ of background ADC measurements:

$$
\sigma _ { n } = \sqrt { \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } ( X _ { i } - \bar { X } ) ^ { 2 } }\tag{5}
$$

The receiver calculates SNR in decibels (dB) as:

$$
\mathrm { S N R } _ { \mathrm { d B } } = 2 0 \log _ { 1 0 } \left( { \frac { S } { \sigma _ { n } } } \right)\tag{6}
$$

## IV. HARDWARE FRONT-END AND FIRMWARE IMPLEMENTATION

## A. Transmitter Hardware Circuitry

The transmitter utilizes an ESP32 microcontroller driving a 650 nm HW-493 laser module. GPIO 23 interfaces to the base of a 2N2222 NPN BJT transistor through a 1 kΩ currentlimiting resistor (refer back to Fig. 2). The transistor operates in saturation mode to switch the laser module powered by a +5 V rail with common system grounding.

## B. Receiver Front-End and Signal Conditioning

The optical receiver front-end employs a BPW34 PIN photodiode reverse-biased at +3.3 V (refer back to Fig. 3). Generated photocurrent enters an LM358 operational amplifier configured as a transimpedance amplifier (TIA). A 100 kΩ feedback resistor sets the gain, while a 10 nF ceramic capacitor connected across the feedback path eliminates high-frequency optical noise. The conditioned output connects to GPIO 4 (ADC1 Channel 3) of an ESP32-S3 microcontroller.

## C. Receiver Communication Firmware

The ESP32-S3 firmware executes payload reception and telemetry processing through the following sequence:

1) Ambient Sampling: Evaluates transmitter-OFF intervals to calculate baseline ambient voltage and background standard deviation $\sigma _ { n } .$

2) Synchronization Detection: Monitors the ADC input for a 100 ms continuous HIGH optical pulse followed by a 20 ms guard interval to establish temporal alignment.

3) Bit Recovery via Majority Voting: Samples each 1 ms bit interval five times across its central 60% window (200 µs to $8 0 0 \mu \mathrm { s } )$ . Each sample thresholding against 1700 ADC counts yields a bit decision via majority vote.

4) BER and Telemetry Logging: Compares the decoded string against the expected sequence to compute BER, forms feature vector x, and triggers local TinyML inference.

## V. MACHINE LEARNING PIPELINE AND EDGE-AI DEPLOYMENT

## A. Hardware Telemetry Dataset

We construct a custom dataset of 395 labeled observations extracted directly from physical hardware tests across four controlled operational channel conditions:

• Class 0 (Normal Condition): Direct unobstructed Lineof-Sight (80 samples).

• Class 1 (Optical Filter Attenuation): Attenuation introduced into optical path (168 samples).

• Class 2 (High Ambient Light): External ambient light source shining on receiver photodiode (50 samples).

• Class 3 (Severe Obstruction / Misalignment): Physical beam blockage or angular receiver misalignment (97 samples).

Figure 5 details the empirical sample distribution across all four diagnostic states, reflecting the operational frequency of physical link impairments evaluated during experimental testing.

![](images/fa407f5c135c44f508482d5c4806956f1d16071c4db5b2e96079928d9a2f73f5.jpg)  
Fig. 5. Empirical sample distribution across four physical channel diagnostic conditions in the collected dataset.

Figure 6 shows the boxplot distributions of the four extracted physical telemetry metrics $( V _ { \mathrm { s i g n a l } }$ , BER, $\mathrm { S N R } _ { \mathrm { d B } }$

$L _ { \mathrm { a m b i e n t } } )$ grouped by diagnostic class. The feature distributions reveal strong cluster separability for High Ambient conditions via elevated baseline lux values (185 lux–192 lux) and for Misalignment states characterized by elevated BER and negative SNR outliers down to −6.5 dB.

![](images/a1b2b7d59a40a4347a998200dc8b33a4d4da8020b607d3fa9c26b85fd996203f.jpg)  
Fig. 6. Boxplot distributions of extracted physical-layer telemetry features $( \breve { V _ { \mathrm { s i g n a l } } }$ , BER, SNR , $L _ { \mathrm { a m b i e n t } } )$ across the four target diagnostic classes.

## B. Baseline Model Training and Quantization

A Random Forest ensemble classifier establishes an algorithmic baseline, achieving 77.22% test accuracy. A fully connected feedforward neural network consisting of an input layer (4 nodes), two hidden dense layers with ReLU activation, and a 4-node Softmax output layer achieves 78.48% test accuracy in floating-point 32-bit (FP32) precision. Figure 7 shows the training and validation accuracy and loss trajectories across 50 epochs.

![](images/9a58ba920538fa33fd59dda01029cbc6cc225dd979de2cbb5b22b10ad89f68fb.jpg)  
Fig. 7. Training and validation loss/accuracy curves across 50 epochs for the diagnostic neural network.

We execute post-training INT8 quantization using Tensor-Flow Lite to enable embedded execution on the ESP32-S3. Quantizing weights and activations from FP32 to 8-bit integers generates a C++ byte array occupying only 3.32 kB of SRAM.

## VI. EXPERIMENTAL RESULTS AND DISCUSSION

## A. Communication Performance Metrics

As illustrated in the physical telemetry feature distributions (Fig. 6), under normal Line-of-Sight (Class 0) conditions, the offline baseline dataset demonstrates high signal integrity with near-zero Bit Error Rate $( \mathbf { B E R } \approx 0 . 0 0 )$ , a signal voltage of $V _ { \mathrm { s i g n a l } } \approx 0 . 2 3 5 \mathrm { \Delta V } .$ , an average $\mathrm { S N R } _ { \mathrm { d B } }$ of +15.0 dB, and a baseline ambient level of $L _ { \mathrm { a m b i e n t } } \approx 1 5 5 . 8$ lux.

Under optical filter attenuation (Class 1), the offline signal voltage exhibits moderate degradation (median $\approx 0 . 1 9 4 \mathrm { ~ V ~ }$ with BER extending across a range up to 0.36. Under high ambient light interference (Class 2), background ambient illuminance increases significantly to 185 lux–192 lux, reducing net signal voltage to between 0.03 V and 0.10 V and dropping the SNR to a range from +1.0 dB to +11.4 dB. Severe obstruction and misalignment (Class 3) produce the most critical channel degradation, yielding median BER values of 0.16 (with maximum values reaching 0.40) and negative SNR outliers reaching down to −6.5 dB.

Following model quantization and local deployment onto the ESP32-S3 firmware, real-time telemetry collected across 100 continuous live execution cycles exhibited distribution characteristics closely aligned with the pre-deployment boxplot baselines. In live execution, normal transmission maintained a tight telemetry cluster $( V _ { \mathrm { s i g n a l } } = 0 . 2 3 1 \pm 0 . 0 0 8 \mathrm { \Delta V }$ $\mathrm { S N R _ { d B } } = + 1 4 . 8 \pm 0 . 4$ dB, $\mathbf { B E R } \leq 0 . 0 1 )$ , confirming minimal firmware sampling jitter. Under live ambient light saturation, the on-device ADC recorded an average baseline shift to 188.4 lux, yielding 100% real-time classification accuracy due to complete statistical separation from other operating states. However, live hardware tests revealed marginal feature distribution overlap between Class 1 (Optical Filter) and Class 3 (Misalignment) in the $V _ { \mathrm { s i g n a l } } \ ( 0 . 1 5 \ \mathrm { V - 0 . 1 8 \ V ) }$ and BER (0.12– 0.20) domains. This live physical overlap accounts for the minor reduction from FP32 simulation accuracy (78.48%) to INT8 on-device test accuracy (74.68%), validating that realtime node performance remains bound primarily by physical channel ambiguity rather than quantization degradation.

## B. Classification Performance and Benchmarking

Table I summarizes the classification performance across evaluated model architectures.

TABLE I  
MODEL ARCHITECTURE BENCHMARKING
<table><tr><td>Model Architecture</td><td>Test Acc. (%)</td><td>RAM Footprint</td><td>Deployment Target</td></tr><tr><td>Random Forest Baseline</td><td>77.22%</td><td>N/A</td><td>Host PC / Cloud</td></tr><tr><td>FP32 Neural Network</td><td>78.48%</td><td>~ 14.20 kB</td><td>Desktop Simulation</td></tr><tr><td>Quantized INT8 TFLite</td><td>74.68%</td><td>3.32 kB</td><td>ESP32-S3 (On-Device)</td></tr></table>

Table II provides per-class precision, recall, and F1-score metrics for the deployed INT8 TFLite Micro model. Figure 8 shows the aggregate classification confusion matrix on test observations.

TABLE II  
PER-CLASS EVALUATION METRICS FOR DEPLOYED INT8 MODEL
<table><tr><td>Diagnostic Class</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>Class 0 (Normal)</td><td>0.94</td><td>1.00</td><td>0.97</td></tr><tr><td>Class 1 (Optical Filter)</td><td>0.69</td><td>0.91</td><td>0.78</td></tr><tr><td>Class 2 (High Ambient)</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Class 3 (Severe Obstruction)</td><td>0.71</td><td>0.26</td><td>0.38</td></tr></table>

![](images/b088d5fab1768b86501ce5eea807a827c5cfae60186320f31802b5c5bf750130.jpg)  
Fig. 8. Normalized classification confusion matrix for the deployed INT8 TensorFlow Lite Micro diagnostic model.

The quantized INT8 model retains high precision (0.94 and 1.00) for Normal and High Ambient states. Post-training quantization causes a minor accuracy drop of 3.80% while reducing memory consumption by over 76%, satisfying the memory constraints of low-cost microcontrollers.

## VII. SYSTEM VERIFICATION, BOUNDARY CONDITIONS, AND PRACTICAL CONSIDERATIONS

Evaluating the embedded optical diagnostic pipeline under practical testbed scenarios highlights critical system boundary conditions, hardware trade-offs, and operational insights:

## A. Synchronization-Dependent Boundary Conditions

Because telemetry extraction and Bit Error Rate (BER) evaluation depend on detecting the 100 ms optical preamble, complete optical beam blockage prevents synchronization acquisition entirely. Under total blockage, the receiver firmware loops continuously in the preamble acquisition state without executing payload decoding or feature generation. We treat total loss of synchronization as an extreme system boundary condition rather than a model classification error.

## B. Feature Separability and Threshold Overrides

Experimental telemetry indicates high feature separability between Class 0 (Normal) and Class 2 (High Ambient Light) due to clear differences in ambient baseline lux $( L _ { \mathrm { a m b i e n t } } >$ 180 lux). Conversely, marginal overlap occurs between Class 1 (Optical Filter Attenuation) and Class 3 (Severe Misalignment) when intermediate physical displacements yield similar signal voltage drops. Implementing deterministic rule overrides for boundary telemetry conditions $( V _ { \mathrm { s i g n a l } } < 0 . 0 2$ V or BER $> ~ 0 . 3 5 )$ ensures high-confidence diagnostic decisions under extreme link degradation.

## C. Computational Overhead and Power Efficiency

Running the quantized TensorFlow Lite Micro inference engine on the ESP32-S3 requires sub-millisecond execution times and consumes only 3.32 kB of SRAM. On-device processing eliminates continuous wireless telemetry streaming, reducing RF/optical channel bandwidth overhead and power consumption compared to cloud-based diagnostic architectures.

## VIII. CONCLUSIONS AND FUTURE RECOMMENDATIONS

This research demonstrates the design, physical implementation, and experimental evaluation of an autonomous, selfdiagnosing optical wireless communication testbed integrated with an Edge-AI diagnostic framework. By extracting realtime physical-layer telemetry—specifically signal voltage, Bit Error Rate (BER), Signal-to-Noise Ratio $\mathrm { ( S N R _ { d B } ) }$ , and ambient light baseline $\left( L _ { \mathrm { a m b i e n t } } \right)$ —the system transforms passive optical receivers into intelligent, self-aware edge nodes.

Benchmarking across classification algorithms confirms that a compact feedforward neural network performs comparably to a Random Forest ensemble baseline (78.48% vs. 77.22% accuracy). Post-training INT8 quantization compresses the neural network to a minimal 3.32 kB memory footprint, allowing complete execution on an ESP32-S3 microcontroller using TensorFlow Lite Micro. The deployed model achieves 74.68% test accuracy across four distinct physical operational states, providing autonomous, localized fault detection without external cloud dependency.

Based on experimental findings, we recommend the following avenues for future research:

1) Dataset Expansion and Environmental Diversity: Expanding data collection across broader indoor lighting variations, thermal fluctuations, and dynamic obstruction angles improves classifier generalization and minimizes reliance on deterministic override rules.

2) Synchronization Loss Recovery Mechanisms: Incorporating an auxiliary hardware signal-presence detector or a dedicated fifth diagnostic state explicitly detects total optical beam blockage when timing preambles fail.

3) Closed-Loop Adaptive Control: Extending the diagnostic framework from passive monitoring to closedloop active correction—such as automated transmitter power scaling or servo-driven mechanical alignment—produces self-healing optical wireless networks.

4) Advanced Modulation and Encoding: Investigating bandwidth-efficient line coding schemes (e.g., Manch ester or PPM) increases achievable bit rates while maintaining robust edge feature extraction.

## REFERENCES

[1] L. U. Khan, “Visible light communication: Applications, architecture, standardization and research challenges,” Digital Communications and Networks, vol. 3, no. 2, pp. 78–88, 2017.

[2] K. S. Purnita and M. R. H. Mondal, “Machine Learning for DCO-OFDM Based LiFi,” PLOS ONE, vol. 16, no. 11, p. e0259955, 2021.

[3] M. A. S. Sejan, M. H. Rahman, M. A. Aziz, D. S. Kim, Y. H. You, and H. K. Song, “A Comprehensive Survey on MIMO Visible Light Communication: Current Research, Machine Learning and Future Trends,” Sensors, vol. 23, no. 2, p. 739, 2023.

[4] M. T. Salman, D. R. Siddle, and A. G. Udu, “Machine Learning Approach to Predict the DC Bias for Adaptive OFDM Transmission in Indoor Li-Fi Applications,” IEEE Access, vol. 13, pp. 9627–9641, 2025.

[5] H. E. Nistazakis, T. A. Tsiftsis, and G. S. Tombras, “Performance analysis of free-space optical communications over atmospheric turbulence channels,” IET Communications, vol. 3, no. 8, pp. 1402–1409, 2009.

[6] C. You, K. Huang, H. Chae, and B. H. Kim, “Energy-Efficient Resource Allocation for Mobile-Edge Computation Offloading Systems,” IEEE Transactions on Wireless Communications, vol. 16, no. 3, pp. 1398– 1411, 2017.

[7] P. Warden and D. Situnayake, TinyML: Machine Learning with Tensor-Flow Lite on Arduino and Ultra-Low-Power Microcontrollers, O’Reilly Media, 2019.

[8] R. David et al., “TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems,” Proceedings of Machine Learning and Systems (MLSys), vol. 3, pp. 800–811, 2021.

[9] L. Dutta and S. Roy, “Real-time signal diagnostics and fault detection in embedded communication nodes,” Discover Applied Sciences, vol. 7, p. 196, 2025.

[10] TensorFlow, “Post-training quantization,” Google AI Edge Documentation. [Online]. Available: https://ai.google.dev/edge/litert/conversion/ tensorflow/quantization/post training quantization.