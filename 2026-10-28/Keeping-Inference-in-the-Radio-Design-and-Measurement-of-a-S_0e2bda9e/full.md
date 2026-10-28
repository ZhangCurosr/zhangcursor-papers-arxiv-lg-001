# Keeping Inference in the Radio: Design and Measurement of a Spiking Accelerator for mmWave ISAC Activity Recognition

Anthony Sanchez ´ <sup>∗</sup>, Rafael Ruiz<sup>∗</sup>, Jose A. Belloch ´ <sup>†</sup>, Jesus O. Lacruz ´ <sup>∗‡</sup>, Joerg Widmer<sup>∗</sup>

<sup>∗</sup>IMDEA Networks Institute, Madrid, Spain

{anthony.sanchez, rafael.ruiz, jesusomar.lacruz, joerg.widmer}@networks.imdea.org

<sup>†</sup>Universidad Carlos III de Madrid, Leganes, Spain´

jbelloc@ing.uc3m.es

Abstract—Integrated sensing and communication allows wireless networks to recognize human activity from the channel estimates that the receiver computes anyway. The inference, however, runs on the general-purpose processors of the base station, access point or handset, which are the first components a device switches off to save energy. Always-on sensing therefore requires a classifier that is small enough to remain powered while the rest of the system sleeps. We investigate whether Spiking Neural Networks (SNNs) can provide such a classifier. To this end, we generate SNN accelerators for micro-Doppler activity recognition on the public DISC dataset (IEEE 802.11ay at 60 GHz) with the open-source Spiker+ framework, and implement them together with a spike encoder on a ZCU208 RFSoC, so that the complete chain from micro-Doppler frame to activity label runs in programmable logic. A fully connected SNN with 132 neurons achieves 94.7 % accuracy (90.7 % macro-F1) on unseen subjects and performs an inference in 206 µs. Compared to conventional networks of the same topology on the same device, it requires an eighth of the logic and 3.4× less dynamic energy per decision than a network fed the same binary spikes, at comparable accuracy, and 7.6× less than a network fed the 16-bit spectrogram, at 1.2 points lower accuracy. Quantization-aware training reduces its weights to 3 bits within half a percentage point of the 6-bit network at the same encoder threshold.

Index Terms—Integrated Sensing and Communication (ISAC), Human Activity Recognition (HAR), Spiking Neural Network (SNN), FPGA, RFSoC, energy efficiency, IEEE 802.11ay

## I. INTRODUCTION

Integrated Sensing and Communication (ISAC) turns wireless links into sensors, since the channel estimates that a receiver computes for data reception change when a person moves. The inference that interprets these estimates, however, runs on the general-purpose part of the system, i.e., on the processors of the Wi-Fi Access Point (AP) [1] or the handset, on the Distributed Unit (DU) of an Open Radio Access Network (O-RAN) base station [2], or in the cloud (Fig. 1a). While the system is active, this is the natural place for it.

Sensing, however, is most useful when it never switches off, and these processors are the first components that a device puts to sleep. An always-on sensing function therefore requires a dedicated low-power path, similar to the block that detects a wake word while a phone sleeps. Such a path consists of a reduced receiver that keeps estimating the channel, a stage that extracts the micro-Doppler (µD) signature, i.e., the Doppler spectra that the moving limbs and torso imprint on successive channel estimates [3], and a classifier that detects a few critical events, such as a fall or a call for help, and then wakes up the rest of the system. Since the classifier exports only a few bits per decision, neither the link nor the processor at the other end has to stay awake. The main challenge is thus a classifier that is small enough to remain powered and completes each decision within the observation interval at very low power.

![](images/f5af8534b56bfe0d0dd6427262225a411705843178c30a993b11630c70e07f6f.jpg)  
Fig. 1. Sensing inference (a) in the cloud, fed by a continuous stream of channel estimates, and (b) in the Radio Unit (RU), exporting only the label.

Spiking Neural Networks (SNNs) are a promising candidate for such a classifier. They communicate through binary events, so that a synaptic operation is an accumulation rather than a multiply–accumulate [4], and their membrane state integrates the µD frames as they arrive, with a few hundred state variables instead of the buffered window of a Convolutional Neural Network (CNN). SNNs have been studied for radar and RF sensing [5, 6], where they can even outperform conventional networks [7], but few of these results have been measured on hardware. Yu et al. [7] measure latency on a neuromorphic chip but estimate energy theoretically, SNNs for activity recognition on communication-derived channel data have only been evaluated in software [8], and NeuroRadar [9] designs a dedicated spiking radar front end. To the best of our knowledge, no work implements a spiking classifier in the hardware of a communication radio.

In this paper, we analyze whether a small SNN can serve as such an always-on classifier. We train a fully connected SNN with 132 neurons on the public DISC dataset [10], generate fixed-point accelerators with the open-source Spiker+ framework [11], and measure their accuracy, latency, resource usage and power on an AMD Xilinx Radio Frequency System on a Chip (RFSoC), with a quantized conventional network of the same topology and input as reference. Our main contributions are:

1) To the best of our knowledge, the first hardware measurements of SNN inference for ISAC Human Activity Recognition (HAR) on communication-derived channel data, on a transceiver-class RFSoC.

2) A reusable integration of Spiker+ cores, with a streaming AXI wrapper that provides cycle-exact latency instrumentation and state reset between observations, and a spike encoder for the $\mu \mathrm { D }$ frames.

3) A comparison of post-training and quantization-aware training, which shows that 6-bit weights preserve the float32 accuracy and that quantization-aware training brings 3-bit weights to within half a point of the 6-bit network, with half the weight memory and 23 % less logic.

4) A same-device comparison with non-spiking networks of the same topology, which shows that the SNN requires an eighth of the logic and 3.4× less dynamic energy per decision than a network on the same binary input at comparable accuracy, and 7.6× less than a network on the 16-bit input at 1.2 points lower accuracy.

## II. FROM PACKETS TO MICRO-DOPPLER

## A. Sensing as a By-Product of Communication

In an ISAC receiver, the sensing input is a by-product of data reception. The receiver estimates the channel from the pilots or training fields of each packet, and when a person moves, the taps that correspond to reflections off the body change from one estimate to the next. A Short Time Fourier Transform (STFT) over successive estimates of these taps converts this variation into the $\mu \mathrm { D }$ spectrogram (Fig. 2a). Throughout the paper, a frame denotes one column of this spectrogram, i.e., the Doppler spectrum of one STFT segment, and a window denotes the sequence of consecutive frames that the classifier receives as one input. In DISC [10], IEEE 802.11ay packets at 60 GHz are transmitted every 0.27 ms, which yields an unambiguous velocity range of ±4.5 m/s, and the 64-point STFT provides a resolution of 0.14 m/s, sufficient to separate the torso of a walking person from the swing of the limbs. Consecutive STFT segments overlap by 32 packets [10], so the radio produces one frame every 8.64 ms.

Exporting the raw channel estimates is costly at Millimeter-Wave (mmWave) frequencies. In DISC-A, each packet carries one training unit, i.e., about 3700 channel estimates per second, which with 128 complex taps and 16 bits per real and imaginary part amount to 15 Mb/s per beam; a link that sweeps twelve beams for localization, as in DISC-B, produces 180 Mb/s. If the classifier runs in the radio (Fig. 1b), only a two-bit label per window, about 1 b/s, leaves the device, seven orders of magnitude less than a single beam.

## B. Problem Formulation

The radio produces one $\mu \mathrm { D }$ frame every 8.64 ms, and the classifier outputs one activity label per window of $N = 2 0 0$ frames, i.e., every 1.73 s, one to two gait cycles of a walking person and far more often than a person changes activity. The classifier must process each frame as it arrives instead of storing the window, so that its memory depends on the number of neurons rather than on the $N \times 5 9$ values of a window, and it must complete each decision well within the 1.73 s. The design itself does not depend on these values, since it processes one frame at a time and the window length is a parameter.

## C. Dataset and Protocol

We use DISC-A, the part of DISC [10] recorded with a single forward-pointing beam. It contains 433 sequences, more than one hour of Channel Impulse Response (CIR) data in which seven subjects walk, run, sit down/stand up and wave hands (Fig. 3 shows the latter), together with the $\mu \mathrm { D }$ spectrograms distributed with the dataset. The evaluation is subject-disjoint: subjects 1–5 train the network and subjects 6 and 7 test it, which measures the generalization to people never seen in training. In contrast to the DISC benchmark, which validates on subject 6 and tests on subject 7, we use no validation subject, so the fixed-point configurations and the encoder threshold reported below were chosen on the test subjects. Windows in which a walking or running subject is momentarily still, detected as a lack of motion energy away from the static return, are discarded in both sets (2 % of the test windows). The training set is balanced to 1341 windows per class by replicating minority-class windows with additive Gaussian noise $( \sigma = 0 . 1 )$ , whereas the 812 test windows (200 frames, cut with a step of 50) keep their natural distribution (68 % walking).

## III. ACCELERATOR DESIGN AND MEASUREMENT METHOD

We generate the accelerators with Spiker+ [11], an opensource framework that takes an SNN described in snnTorch, trains it, quantizes weights and membrane potentials to fixed point and generates synthesizable VHDL, so that the hardware remains a function of the trained model. We chose a Field Programmable Gate Array (FPGA) over a neuromorphic chip, since the latter would add a second device with its own package, power domain and interfaces to the radio. The $\mu \mathrm { D }$ frames are produced where the channel is estimated: in the baseband of an AP or a handset, or, in a cellular network with an O-RAN 7.2x split, in the DU, which is precisely the unit that sleeps, so the always-on path of Fig. 1b needs a reduced channel estimator in the RU ahead of the classifier. In either case the classifier consumes the frames without any data leaving the chip, and its VHDL can be ported to the baseband ASIC.

![](images/bdc583c7d52503dd06c05a5bbaf01878f97e24b216199566919b50f8eec12feb.jpg)  
(a) Processing chain

![](images/112523c4f7a792dfad66fce580a3c1e6c5687b3086d7c314733851b6aa2430c6.jpg)  
(b) Spike encoder

![](images/dbb979900cb861cd0735957fee69097616a27fa0a8ab27148c4cfbc8349b221b.jpg)  
(c) SNN accelerator  
Fig. 2. (a) Processing chain, from the 802.11ay receiver to the activity label; orange blocks come from the DISC dataset, blue and grey ones are implemented in this work. (b) Spike encoder. (c) SNN accelerator: the Spiker+ core inside the AXI stream wrapper.

![](images/eaae9718649e52b338e1946acb1a269d73097c6fc5c27e7bd3eee834ff332ef6.jpg)

![](images/21aee2b32bb79a33ea7c4765aa1f9a156682e31b833a043018c16ac80e30a984.jpg)  
Fig. 3. Spike encoding of a waving-hands window of 200 frames (1.73 s). Left: normalized µD spectrogram (dB). Right: spike rasters after the per-frame relative threshold at $\bar { \theta } = 0 . 7 5$ and $\theta = 0 . 8 5$ (6.2 % and 4.1 % of the bins active in this window).

## A. Spike Encoding

A spiking network operates on binary events, so the spectrogram has to be converted into spikes, either by an encoder or by feeding the values as input currents into the first layer, which gives up the multiplier-free arithmetic [4]. Rate and latency coding require several time steps per value, and delta coding discards the steady components. Since the $\mu \mathrm { D }$ spectrogram already has a time axis, we instead map each frame to one time step and let a Doppler bin spike if its magnitude is large compared to the rest of the frame.

The encoder (Fig. 2b) determines the minimum and maximum of the bins of a frame and emits a spike for every bin above min +θ (max − min). Because the reference is the frame’s own range, the rule is independent of the absolute signal level, $\mathrm { e . g . }$ , of gain and distance, which is what allows the encoder to sit directly behind the STFT of a radio; it requires no memory of earlier frames, but the frame maximum always fires, so a frame without motion also produces spikes. The DISC spectrograms are distributed in dB and already normalized per frame to [0, 1] [10], so on this data the rule coincides with a fixed threshold on the unsigned 16-bit values the encoder receives. We use $\theta = 0 . 7 5$ for SNN-S, SNN-M and SNN-L, which keeps the torso and limb traces (Fig. 3) with 6.4 % of the bins active over the test set.

TABLE I  
SNN FIXED-POINT CONFIGURATIONS.
<table><tr><td>Fixed-point configuration</td><td>SNN-S</td><td>SNN-M</td><td>SNN-L</td><td>SNN-SS</td></tr><tr><td>Weight bits</td><td>6</td><td>6</td><td>8</td><td>3</td></tr><tr><td>Membrane bits</td><td>7</td><td>8</td><td>10</td><td>6</td></tr><tr><td>Fractional bits (both)</td><td>4</td><td>6</td><td>8</td><td>4</td></tr><tr><td>Quantization</td><td>post</td><td>post</td><td>post</td><td>aware</td></tr><tr><td>Encoder θ</td><td>0.75</td><td>0.75</td><td>0.75</td><td>0.85</td></tr></table>

## B. Network and Training

The SNN has 59 inputs, one per Doppler bin, since the distributed spectrograms keep 59 of the 64 STFT bins, a hidden layer of 128 first-order Leaky Integrate-and-Fire (LIF) neurons and 4 output LIF neurons, one per activity, and runs one time step per $\mu \mathrm { D }$ frame, so that it operates on a stream without buffering a window. All neurons fire at a threshold of 1.0 and decay with $\beta = 1 - 2 ^ { - 4 } = 0 . 9 3 7 5$ , so that the leak is a 4-bit arithmetic shift. Hidden neurons use subtractive reset, output neurons do not reset within a window, and all potentials are cleared between windows. In float32, β gives the membrane a memory of about 16 frames; in fixed point, however, the leak $\lfloor v / 1 6 \rfloor$ vanishes below $2 ^ { 4 - f } .$ , i.e., below the threshold for f = 4 (SNN-S and SNN-SS), so the hidden neurons of those two configurations do not leak at all, and only SNN-L leaks approximately as the float32 model does. The label is the class whose output neuron fires most over the window. All networks are trained offline with backpropagation through time and surrogate gradients [4] (arctangent with $\alpha = 2$ , Adam, learning rate $5 \times 1 0 ^ { - 4 }$ , batch size 64, 15 epochs) with a cross-entropy loss on the output spike counts. SNN-S, SNN-M and SNN-L share one float32 training run at $\theta = 0 . 7 5$ and differ only in their post-training quantization; this run is the best of three seeds (mean accuracy $9 3 . 1 \% \pm 1 . 7 )$ , selected on the test subjects, which makes the float32 accuracy and the hardware accuracies derived from it optimistic by about 1.5 points. SNN-M at $\theta = 0 . 8 5$ is a separate training run at that threshold, and SNN-SS is trained with the quantization in the loop (Section III-C).

## C. Fixed-Point Quantization and HDL Generation

Spiker+ quantizes weights and membrane potentials to two’s-complement fixed point with f fractional bits. We swept weight widths of 6–10 bits, membrane widths of 7–10 bits and f of 4–10 bits (140 combinations) and selected, among the most accurate in Spiker+’s emulation on the test subjects, three configurations of different width from one set of float32 weights (Table I), for which Spiker+ generated the VHDL. The deployed designs include a one-line correction of the outputneuron reset in the VHDL that Spiker+ 2.0.3 generates, which we have reported upstream.

The accuracy that Spiker+ reports for its software emulation of the quantized network differs from that of the generated hardware by up to several points, because the emulation does not apply saturation, leak and spike propagation in the order of the circuit. We therefore built a bit-exact software model of the generated core, verified against the board, and use it to explore configurations without a bitstream for each; all hardware accuracies in this paper are nevertheless measured on the board, with the on-chip encoder.

Each layer updates all its neurons in parallel while feeding the input spikes serially (Fig. 2c), so the first layer spends 59 cycles per time step and the second 128, and a window costs roughly 200×(59+128) cycles plus control overhead. A layer skips any time step without input spikes, but with our encoder the first layer never does, since the frame maximum always fires; the latency is therefore almost independent of the input.

To reduce the bit-width further, we also trained with the quantization in the loop, using a forward pass with a differentiable copy of the accelerator’s fixed-point arithmetic, so that the network learns under the same rounding, saturation and spike timing as the hardware. This quantization-aware training is not part of Spiker+, but it produces ordinary weights that deploy through the same flow. We use it for SNN-SS, with 3-bit weights and the encoder threshold θ = 0.85 (Table I).

## D. System Integration on the RFSoC

The Spiker+ cores expose a bespoke start/ready handshake and raw spike vectors rather than a standard interface. We therefore wrapped the core in an AXI stream interface, so that any AXI master can push frames at any rate, with backpressure in both directions. The wrapper also frames the stream into windows, clears all membrane potentials at the end of each window, and stamps the cycle count onto the last beat, which yields a cycle-exact latency independent of the host. Four counters and a comparator in the wrapper count the output spikes of a window and return the label.

The spike encoder is a streaming block in front of the wrapped SNN (Fig. 2b). It receives one 59-bin frame of 16-bit values over its own AXI stream, emits the spike vector within 161 clock cycles and takes θ over AXI-Lite. Implemented together with the wrapper and the SNN on the board, it reproduces the software encoder bit for bit on all 812 test windows, and the labels match the bit-exact model. PYNQ<sup>1</sup> manages the data streaming and the registers, and an AXI Direct Memory Access (DMA) moves the frames from the processor to the Programmable Logic (PL) and the labels back.

In a radio, the STFT of the digital front end, followed by a logarithm since the encoder expects dB values, would replace the DMA.

## E. Measurement Method

Latency is measured in two ways. Inside the PL, as in a radio, the wrapper’s cycle counter gives the processing time of a window. Through the processor, as in the prototype, a timer around the DMA transfer gives the end-to-end time seen by software (median over 1000 passes). Resources are the postimplementation utilization reports of Vivado 2023.1 on the XCZU48DR, for the SNN cores implemented out of context and for the ANN and encoder instances of the complete design; for SNN-M the two differ by 3 %. Power is that of the VCCINT rail, the supply of the PL, computed as the product of the current and bus voltage read from the board’s INA226 monitor through the Linux hwmon interface (1 mA resolution). Because one current reading takes longer than one inference, the test window is stored on chip and replayed through the encoder to the accelerator back-to-back, so that the core runs continuously while the 1000 readings per point are taken, each averaging over tens of inferences.

## F. Non-Spiking Reference

As reference, we implemented the same topology without spikes, Linear(59,128), ReLU, Linear(128,4), applied to each frame with the 200 logit vectors summed before the decision, in two versions: ANN, fed with the same binarized spectrogram as the SNN, and ANN16, fed with the 16-bit normalized spectrogram. Both are trained in float32 with the recipe of Section III-B, quantized to int8 weights and 16-bit activations, generated with hls4ml<sup>2</sup> (Vitis HLS 2023.1) with 192 multipliers, which Vivado maps to 200 and 192 DSP slices, wrapped with the same stream contract as the SNN and measured with the same method. Neither closes timing at 250 MHz, so all designs are compared at 187.5 MHz. While topology and input are controlled, hls4ml generates a parallel datapath with DSP multipliers and Spiker+ a serial one with memories, so the differences in logic reported below include the implementation style; in particular, the binary ANN uses the same multipliers as ANN16, since hls4ml does not exploit 1-bit inputs.

## IV. RESULTS

## A. Classification Accuracy

Table II gives the recall of every class, the accuracy and the macro-F1, which weighs the four activities equally, whereas accuracy is dominated by the 68 % of walking windows. SNN-S, SNN-M and SNN-L derive from the same float32 model (94.6 % accuracy, 91.8 % macro-F1). SNN-L matches it, SNN-M keeps the accuracy and loses one point of macro-F1, and SNN-S loses three points of accuracy and nine of macro-F1. We attribute the latter to its four fractional bits, with which the hidden neurons do not leak below threshold (Section III-B):

TABLE II  
PER-CLASS RECALL, ACCURACY AND MACRO-F1 ON THE TEST SET.
<table><tr><td rowspan="2"></td><td colspan="4">Recall (%)</td><td rowspan="2">Acc. Macro- (%)</td><td rowspan="2">F1 (%)</td></tr><tr><td>Walk</td><td>Run</td><td>Sit/St.</td><td>Wave</td></tr><tr><td rowspan="3">SW</td><td>FC SNN</td><td>95.5</td><td>89.0 98.9</td><td>89.7</td><td>94.6</td><td>91.8</td></tr><tr><td>FC ANN</td><td>97.6</td><td>85.7 85.7</td><td>83.3</td><td>93.6</td><td>88.2</td></tr><tr><td>FC ANN16</td><td>97.5 93.4</td><td>85.7</td><td>98.7</td><td>95.8</td><td>92.7</td></tr><tr><td rowspan="6">HW</td><td>SNN-S</td><td>99.1</td><td>68.1</td><td>96.7</td><td>57.7</td><td>91.4</td><td>82.8</td></tr><tr><td>SNN-M</td><td>98.9</td><td>78.0</td><td>96.7</td><td>82.1</td><td>94.7</td><td>90.7</td></tr><tr><td>SNN-L</td><td>98.6</td><td>81.3</td><td>97.8</td><td>84.6</td><td>95.2</td><td>91.8</td></tr><tr><td>SNN-M, θ 0.85</td><td>96.6</td><td>79.1</td><td>95.6</td><td>83.3</td><td>93.2</td><td>88.7</td></tr><tr><td>SNN-SS, θ 0.85</td><td>93.3</td><td>94.5</td><td>95.6</td><td>83.3</td><td>92.7</td><td>88.1</td></tr><tr><td>FC ANN, int8</td><td>98.4</td><td>82.4</td><td>86.8</td><td>80.8</td><td>93.6</td><td>88.2</td></tr><tr><td>FC ANN16, int8</td><td></td><td>99.1 91.2</td><td>93.4</td><td>82.1</td><td>95.9</td><td>92.8</td></tr></table>

SNN-SS has the same leak but was trained under it, whereas SNN-S was quantized after a float32 training with the full leak. In all post-training configurations, walking gains recall while running and waving lose it. Raising θ from 0.75 to 0.85 removes 35 % of the input spikes and 25 % of the hidden spikes, and costs the retrained SNN-M 1.5 points of accuracy. At that threshold, quantization-aware training makes 3-bit weights usable: SNN-SS reaches 92.7 %, half a point below SNN-M, whereas the same network quantized after training reaches 87.8 % (five seeds, bit-exact model).

The non-spiking networks (Section III-F) measure the cost of binarization and of spiking separately. Fed the 16-bit spectrogram, ANN16 reaches 95.9 % and 92.8 % macro-F1; fed the same spikes as the SNN, the ANN drops to 93.6 % and 88.2 %, so binarization costs this architecture 2.3 points of accuracy and 4.5 of macro-F1. SNN-M, on the same spikes, sits in between: 1.1 points above the binary ANN and 1.2 below ANN16, differences of 9 and 10 windows that lie within the spread between training seeds (±1.7 points), so we consider the three accuracies comparable. The CNN of the DISC benchmark reaches a mean per-class F1 of 96.1 % [10] and an SNN with a learned encoder on the raw CIRs a macro-F1 of 95.8 % [8], both on subject 7 alone; on that subject SNN-L reaches 92.1 % accuracy and 91.1 % macro-F1, five points behind networks with convolutions and 3.5 to 30 times more parameters.

## B. Latency

With the encoder in front of it, the accelerator processes a stored window in 38 676 clock cycles in all SNN configurations, i.e., 206 µs at 187.5 MHz, the clock at which all designs are compared. The count does not vary between repetitions and changes by less than 1 % between windows, since only the second layer skips steps. The ANN needs 16 802 cycles (89.6 µs) and ANN16 19 616 (104.6 µs). Through the processor, the DMA and the software add 225 µs per window, which disappear when the frames arrive from the PL. On a live stream, the encoder and the accelerator spend about 354 cycles (2 µs) on each 8.64 ms frame as it arrives, so the label is available about 2 µs after the last frame and the hardware computes during about 0.02 % of the time.

## C. Resources

Table III lists the utilization of the accelerators alone, without DMA and processor interconnect. The SNN cores occupy 3.5–5.2 k LUTs, 1.7–2.2 k registers, 5.5–14.5 BRAM tiles and no DSP slice, since multiplying a weight by a spike reduces to adding the weight or not and the leak is a shift, and every resource grows with the bit-width. The ANN and ANN16 need 6.7–7.6× the LUTs, 14–15× the registers and 192–200 DSPs of SNN-M, at a comparable number of BRAM tiles; the 16-bit input costs no logic, since hls4ml implements both with the same multipliers. The 3-bit SNN-SS needs 23 % fewer LUTs and half the BRAM of SNN-M. The encoder adds 611 LUTs and 965 registers, and SNN-M with its encoder uses 1.2 % of the LUTs of the ZCU208, a footprint that fits on far smaller devices.

## D. Power and Energy

Fig. 4 shows the power of the PL supply through four phases, i.e., idle, front end alone (encoder and replay memory), network running, and idle again, for four of the designs. With the bitstream loaded and nothing running, the rail draws 33–50 mW with the SNN designs and 54–60 mW with the ANNs. Part of this difference is a real cost of the ANNs, whose 14 to 15 times more registers load the clock tree also when idle; the rest follows the placement, since the clock buffers of every region that holds logic draw power whether it toggles or not, and the spread among the SNN designs tracks the clock regions each bitstream occupies (7 to 17) and its BRAM tiles. We therefore compare only the increase over idle within one measurement, where the idle level is stable to 0.4 mW. The encoder adds nothing the monitor can resolve. ANN16 adds 27.5 mW and the ANN 14.4 mW: with binary inputs, most multiplier inputs are zero and the datapath toggles half as much. SNN-S adds 1.6 mW, SNN-M 1.8 mW, SNN-L 2.3 mW and SNN-SS 1.2 mW. Per inference, SNN-M thus needs 0.38 µJ, 3.4× less than the ANN (1.29 µJ) and 7.6× less than ANN16 (2.88 µJ), and SNN-SS 0.25 µJ. At one inference every 1.73 s, the dynamic energy of SNN-M amounts to 0.2 µW of average power, so in an always-on path the cost of the classifier is set by its static power, i.e., by its area, and by the receiver and STFT ahead of it, which this paper does not quantify.

## E. Design Space

Fig. 5 places the six accelerators in the macro-F1–energy– area plane. ANN16 is the most accurate and the most expensive point, at 7.6× the energy and 7× the LUTs of SNN-M. SNN-M and SNN-L sit above the binary ANN in macro-F1 and SNN-SS matches it, at a seventh to a tenth of its LUTs and 2.7–5.2× less energy. Among the SNN configurations, area follows the bit-width, and quantization-aware training moves SNN-SS to the lower-left corner of the plane.

## V. DISCUSSION

What limits accuracy. On subject 7, SNN-L (91.1 % macro-F1) remains below the 96 % that the DISC CNN [10] and an SNN with a learned encoder and 28k parameters [8] reach. ANN16 shows where the gap lies: on the same topology, binarizing the input costs 4.5 points of macro-F1, of which SNN-M recovers half on the same spikes, possibly through its temporal integration, although the difference lies within the spread between seeds; the rest separates a fully connected network with 8k parameters and no temporal feature extraction from the convolutional designs.

TABLE III  
POST-IMPLEMENTATION UTILIZATION OF THE ACCELERATORS ALONE (XCZU48DR).
<table><tr><td rowspan="2"></td><td colspan="3">SNN</td><td colspan="2">ANN</td></tr><tr><td>S</td><td>M</td><td>L SS</td><td>bin</td><td>16</td></tr><tr><td>LUTs</td><td>4095</td><td>4529</td><td>5184 3504</td><td>34374</td><td>30131</td></tr><tr><td>Registers</td><td>1824</td><td>1940 2204</td><td>1688</td><td>28891</td><td>27306</td></tr><tr><td>BRAM</td><td>6.5</td><td>11</td><td>14.5 5.5</td><td>14</td><td>14</td></tr><tr><td>DSPs</td><td>0</td><td>0</td><td>0</td><td>0 200</td><td>192</td></tr></table>

![](images/d9ec8ea52e76a70a954c237b7138d4751a9319e9494f574993c09fc593a12bf0.jpg)  
Fig. 4. PL supply power across the four measurement phases (medium window, 187.5 MHz). Labels: mean increase over idle.

![](images/8547a73570e3e4856a00ec1ebbfcd82e5346371297303abcce34af7f059fdce5.jpg)  
Fig. 5. Macro-F1 on the board against dynamic energy per inference at 187.5 MHz; bubble area proportional to LUTs.

Where the savings come from. Since the encoder always fires the frame maximum, the latency is almost constant and the dynamic power changes by less than 0.5 mW with the input activity. The savings over the ANNs thus come from accumulate-only arithmetic, serial processing and small area rather than from input sparsity, consistent with hardwareaware studies that find SNNs more energy-efficient than dense networks only at low spike rates [12]. An absolute noise floor in the encoder would make frames without motion, the most common input of an always-on sensor, also the cheapest ones.

Energy and area. At one decision every 1.73 s, 0.38 µJ corresponds to $0 . 2 \mu \mathrm { W }$ , far below the idle power of any FPGA design and of the front end that the always-on path keeps running, i.e., receiver, ADC, channel estimation and STFT, which we have not quantified. What matters in a product is the standby power of the block, which in an ASIC is dominated by leakage and scales with area. There, the SNN needs no multipliers, while its weight memory equals that of a dense network of the same width, and quantization-aware training halves it at a cost of half a point.

Scope. DISC-A covers four activities by seven subjects on one monostatic link. Always-on applications need even fewer classes, since presence, a fall or a call for help are one or two critical classes against background activity, and each additional class costs one output neuron without changing the latency.

## VI. CONCLUSION

We have shown that the inference of a micro-Doppler activity classifier can be kept in the radio. A spiking network with 132 neurons and its encoder run in the programmable logic of an RFSoC, achieve about 95 % accuracy on unseen subjects and perform an inference in 206 µs. Compared to conventional networks of the same topology, the SNN needs an eighth of the logic and 3.4× less dynamic energy per decision than a network on the same binary input, at comparable accuracy, and 7.6× less than a network on the 16-bit input, at

1.2 points lower accuracy; quantization-aware training reduces its weights to 3 bits at a cost of half a point at the same encoder threshold. On the prototype, the static power of the FPGA fabric dominates. In an ASIC, such a block would reside in a low-power domain whose leakage scales with area, which is where the small footprint of the SNN pays off and allows sensing to remain active while the rest of the system sleeps.

## REFERENCES

[1] F. Meneghello, C. Chen, C. Cordeiro, and F. Restuccia, “Toward Integrated Sensing and Communications in IEEE 802.11bf Wi-Fi Networks,” IEEE Communications Magazine, vol. 61, no. 7, pp. 128–133, 2023.

[2] S. D’Oro, M. Polese, L. Bonati, H. Cheng, and T. Melodia, “dApps: Distributed applications for real-time inference and control in O-RAN,” IEEE Communications Magazine, vol. 60, no. 11, pp. 52–58, 2022.

[3] J. Pegoraro, J. O. Lacruz, F. Meneghello, E. Bashirov, M. Rossi, and J. Widmer, “RAPID: Retrofitting IEEE 802.11ay Access Points for Indoor Human Detection and Sensing,” IEEE Transactions on Mobile Computing, vol. 23, no. 5, pp. 4501–4519, May 2024.

[4] J. K. Eshraghian, M. Ward, E. O. Neftci, X. Wang, G. Lenz, G. Dwivedi, M. Bennamoun, D. S. Jeong, and W. D. Lu, “Training spiking neural networks using lessons from deep learning,” Proceedings of the IEEE, vol. 111, no. 9, pp. 1016–1054, 2023.

[5] A. Safa, F. Corradi, L. Keuninckx, I. Ocket, A. Bourdoux, F. Catthoor, and G. G. E. Gielen, “Improving the accuracy of spiking neural networks for radar gesture recognition through preprocessing,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 6, pp. 2869– 2881, 2023.

[6] E. Cicciarella, R. Mazzieri, J. Pegoraro, and M. Rossi, “Learned spike encoding of the channel response for low-power environment sensing,” in Proc. IEEE Int. Conf. Pervasive Computing and Communications Workshops (PerCom Workshops), 2024, pp. 763–768.

[7] D. Yu, Z. Liao, C. Lv, W. Tong, L. Jiang, S. Ji, X. Du, H. Zhao, X. Zheng, and S. Deng, “Frequency matching in spiking neural networks for mmWave sensing,” in Proc. Int. Conf. Machine Learning (ICML), 2026.

[8] E. Cicciarella, R. Mazzieri, J. Pegoraro, and M. Rossi, “Sparse spike encoding of channel responses for energy efficient human activity recognition,” in Proc. IEEE Int. Conf. Communications (ICC), 2026.

[9] K. Zheng, K. Qian, T. Woodford, and X. Zhang, “NeuroRadar: A neuromorphic radar sensor for low-power IoT systems,” in Proc. ACM Conf. Embedded Networked Sensor Systems (SenSys), 2023.

[10] J. Pegoraro, P. Saucedo, J. O. Lacruz, M. Rossi, and J. Widmer, “DISC: A dataset for integrated sensing and communications in mmWave systems,” IEEE Communications Magazine, vol. 63, no. 10, pp. 94–100, 2025.

[11] A. Carpegna, A. Savino, and S. Di Carlo, “Spiker+: A framework for the generation of efficient spiking neural networks FPGA accelerators for inference at the edge,” IEEE Transactions on Emerging Topics in Computing, vol. 13, no. 3, pp. 784–798, 2025.

[12] M. Dampfhoffer, T. Mesquida, A. Valentian, and L. Anghel, “Are SNNs really more energy-efficient than ANNs? An in-depth hardwareaware study,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 7, no. 3, pp. 731–741, 2023.