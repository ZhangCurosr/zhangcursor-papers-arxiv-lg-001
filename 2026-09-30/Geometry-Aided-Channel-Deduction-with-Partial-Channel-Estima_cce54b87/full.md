# Geometry-Aided Channel Deduction with Partial Channel Estimates and Uncalibrated Digital Twin

Hongning Ruan, Zhaoyang Zhang, Zirui Chen, Ziqing Xing, Zhaohui Yang, and Merouane Debbah´

Abstract—The acquisition of high-dimensional channel state information (CSI) in wireless multi-input multi-output (MIMO) orthogonal frequency division multiplexing (OFDM) communications usually requires high pilot overhead, or relies on accurate and complete positional or environmental information. In this paper, we propose a geometry-aided channel deduction (GCD) approach, which utilizes an uncalibrated digital twin (DT) with only approximate environmental geometry and positions to assist the channel acquisition. The key rationale behind is that, even imprecise geometric information, which can be easily obtained in advance through radio sensing technologies or existing geographic databases, provides certain structural features about the current channel; meanwhile, the coarse instantaneous channel estimates using only a small amount of pilots provide dedicated information that aligns with the channel structure and further compensates for the geometry inaccuracy and other channel unknowns. To this end, we first extract geometric features from the DT, which contain only simple structural information of the channel. Then we propose random prompt augmentation, a novel method to generate an appropriate prompt that converts geometric multi-path structure into a CSI-like representation while suppressing the disturbance of other unknown channel parameters. The prompt is then fused with the pilot-based instantaneous channel estimate via a channel deduction network. To further enhance the network’s versatility, we incorporate pilot configurations into the existing learning architecture to support variable pilot patterns. Comprehensive experiments validate the superiority of the proposed method, which demonstrates high channel acquisition quality, low pilot overhead, and strong robustness. Furthermore, the structural prompt also serves as scenario-related context, enabling our approach to generalize well in new scenarios.

Index Terms—Channel acquisition, channel deduction, digital twin, ray tracing, cross-scenario generalization

## I. INTRODUCTION

## A. Background

Future wireless networks are envisioned to evolve toward unprecedented levels of connectivity, data rate, spectral efficiency, and intelligence, enabled by a diverse range of emerging technologies, including extremely large-scale multipleinput multiple-output (MIMO) [2], terahertz communications, reconfigurable intelligent surfaces (RIS), stacked intelligent metasurfaces (SIM) [3], and integrated sensing and communication (ISAC) [4]. Along this evolution, the increasing scale of antenna arrays and wider transmission bandwidths are driving wireless channels toward higher dimensionality. Acquiring such high-dimensional channel state information (CSI) with low pilot overhead poses significant challenges.

Since the physical environments experienced by channels under different subcarriers and antennas are similar, a subset of the CSI can be mapped to the entire CSI, thereby reducing pilot overhead. However, the complexity of electromagnetic (EM) propagation characteristics makes it difficult for traditional interpolation methods to achieve optimal performance. Some methods [5], [6] estimate path parameters using pilot measurements to obtain complete CSI, but they typically involve high computational complexity for parameter estimation. Deep learning excels at uncovering latent features and representing high-dimensional data, and has been applied to the field of channel estimation [7]. However, due to the dynamic and variable nature of wireless communication systems, such data-driven methods face challenges in terms of generalization. A key issue is whether deep neural networks can maintain their performance when channel propagation conditions change due to dynamic environmental variations or user mobility across different scenarios [8]. Additionally, the neural network should be able to adapt to diverse system configurations and variable pilot patterns. These factors directly influence the feasibility of learning methods to be deployed in real-world systems.

## B. Related Works

1) Pilot-Based Channel Estimation: Deep neural networks have been widely used for channel estimation. For example, [9], [10] employ convolutional neural network (CNN) to perform frequency domain interpolation, where pilot symbols are inserted into certain subcarriers. [11] employs multi-layer perceptron (MLP) to perform spatial domain extrapolation, using CSI of a subset of antennas to infer that of other antennas. [12] proposed complex-domain MLP-Mixer (CMixer), which employs physics-inspired design to map partial estimates from a subset of antennas and subcarriers to complete spatialfrequency domain CSI, outperforming MLP- and CNN-based approaches. These neural networks are typically tailored to specific pilot configurations. Recent studies [13]–[15] have explored the potential of diffusion models in channel estimation, using posterior sampling to support arbitrary pilot settings. However, such methods still require significant pilot resources to ensure channel estimation quality. In fact, when the multi-path characteristics of the channel are extremely complex, relying solely on limited pilot resources typically fails to provide sufficient channel features.

TABLE I: Comparison of relevant works on channel acquisition.
<table><tr><td>Category</td><td>Auxiliary information</td><td>Low pilot overhead</td><td>Robustness to inaccurate information</td><td>No error propagation</td><td>Cross-scenario generalization capability</td><td>Low data collection cost</td><td>Related works</td></tr><tr><td>Pilot-based channel estimation (pilot only)</td><td>None</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>[9]-[15]</td></tr><tr><td rowspan="4">Pilot-free channel prediction (auxiliary information only)</td><td>Past channels</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td>[16], [17]</td></tr><tr><td>Position</td><td>√</td><td></td><td>√</td><td></td><td>√</td><td>[18], [19]</td></tr><tr><td>Real-time sensing</td><td>√</td><td></td><td>√</td><td>√</td><td></td><td>[20]</td></tr><tr><td>Radio environment</td><td>√</td><td></td><td>√</td><td></td><td></td><td>[21]-[26]</td></tr><tr><td rowspan="4">Hybrid channel acquisition (pilot &amp; auxiliary information)</td><td>Past channels</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>[27]</td></tr><tr><td>Position</td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>[28]</td></tr><tr><td>Real-time sensing</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>[29]</td></tr><tr><td>Channel dataset Geometry</td><td>√ √</td><td>√ √</td><td>√ √</td><td>√ √</td><td>√</td><td>[28], [30] Ours</td></tr></table>

2) Pilot-Free Channel Prediction: Some studies have focused on achieving pilot-free channel prediction, based on the premise that some available information can be deterministically mapped to complete CSI. For example, the current channel can be predicted using channels from previous time slots [16], [17]. However, the small-scale mobility of users and subtle changes in the environment are often difficult to predict. Additionally, the auto-regressive inference accumulates prediction errors over time, leading to significant degradation in performance. Some studies [18], [19] utilize user positions to predict channel information, which require extremely high positioning accuracy, as any positioning errors comparable to the wavelength can cause drastic changes in prediction results. Furthermore, since the wireless channel is jointly influenced by the user position and the surrounding environment, channel prediction models solely relying on positions lack generalization capabilities in new scenarios. [20] utilizes multi-modal sensing data to achieve adaptive CSI generation in dynamic environment, which ignores the impact of unknown environmental EM characteristics. Other studies focus on reconstructing the physical environment to create a digital twin (DT) that reflect the characteristics of wireless signal propagation. For example, [21]–[23] utilize known environmental geometry and collect measured channel data to calibrate unknown environmental materials or learn EM interactions. The calibrated DT can be provided to a ray tracer to predict channel data at arbitrary positions. In contrast, [24]– [26] focus on reconstructing a wireless radiation field based on measured channel data, without requiring environmental geometry. Despite their effectiveness, these methods still rely on high-precision user positioning, and the calibration or reconstruction process incurs data collection and computational costs for each individual scenario. Furthermore, all these methods overlook the diversity of antenna radiation characteristics, which is influenced by numerous factors such as hardware conditions and user device orientations. As a result, when no pilot resources are available, it is practically only possible to obtain large-scale information – such as received signal strength and channel gain [31], [32] – rather than complete CSI. While such information is useful in certain applications, it cannot be used for more complex wireless tasks, such as symbol detection and precoding in multi-path scenarios.

3) Hybrid Channel Acquisition: Recent studies have enhanced channel estimation by combining pilot resources with auxiliary information. On one hand, the supplementary channel features provided by the auxiliary information signifi cantly reduce pilot overhead; on the other hand, by introducing a small amount of pilot resources, these methods relax the accuracy requirements for the auxiliary information, as some small-scale or unpredictable features can be calibrated through instantaneous estimates from pilots. For example, [27] proposed the channel deduction (CD) framework that combines estimation with prediction, leveraging past channels to provide large-scale features to reduce pilot overhead. [28] proposed position-aided CMixer (PCMixer) that fuses position and channel information to obtain the complete channel, thereby enabling tolerance for positioning errors. To obtain more accurate and scenario-aware features, recent studies have incorporated scenario information to enhance performance and generalization capability. For example, [29] extracts environmental features from multi-view images and combines it with partial CSI to obtain complete CSI. This method requires the mobile terminals to be equipped with sensing facilities and to continuously collect image data. [28] also proposed spatial channel deduction (SCD), which utilizes historical channel datasets from the current scenario to provide largescale features. Similarly, [30] extracts channel knowledge from a full DT to enhance channel estimation. While they achieve outstanding performance, the acquisition of real-world channel data is expensive, whereas the full DT requires accurate and comprehensive environmental information, leading to high costs for deployment in new scenarios.

For clarity, we summarize relevant studies in Table I.

## C. Motivations and Contributions

Recent advances in radio sensing technologies and existing geographic databases enable us to easily obtain static environmental maps, thereby supporting the construction of an approximate DT of the physical world. By utilizing the environmental geometry and transceiver positions, the multipath structure of the channel can be obtained using a ray tracer [33], [34]. Although these geometric details may contain inaccuracies, and many factors, such as environmental materials and antenna radiation characteristics, remain unobservable, the obtained structural features serve as effective prior information to enhance channel estimation, thereby significantly reducing pilot overhead. Meanwhile, geometric errors and other channel unknowns can be effectively compensated through instantaneous channel estimates using a small amount of pilots. Therefore, an uncalibrated DT without EM material properties is sufficient to effectively assist in channel acquisition. Furthermore, from the perspective of neural network training, geometric prior also introduces scenario-related context, enabling the neural network to extract common knowledge across different scenarios through multi-scenario collaborative learning, thereby demonstrating strong generalization capabilities in new scenarios.

Although our main idea shares the same spirit as recent studies that use DT [35] or channel knowledge map (CKM) [36] to enhance pilot-based channel estimation, it also differs significantly from these work in the following aspects. First, an ideal DT not only requires extremely high geometric accuracy with errors much smaller than the wavelength, but also entails substantial costs for data collection and computation to calibrate the unknown EM material properties [21]. In contrast, we use an approximate geometric map without specific material properties to provide only the channel structures and then generate appropriate prompts with them using a novel random prompt augmentation mechanism, which suppresses the disturbance of other channel unknowns, such as the amplitude attenuation and phase shift of each path. Second, real-time DTs require online ray tracing to continuously update channel features, which incurs significant computational overhead and inference latency [34]. In contrast, we employ only offline ray tracing to pre-extract channel features, and during online inference, we simply query them via neighborhood sampling based on user position, thereby avoiding the high costs associated with real-time ray tracing. Third, the CKM-based approaches typically maintain the CKM at the BS and require user positions for online CKM query [37], thus they are in general more suitable for uplink channel acquisition and may incur user privacy concerns. In contrast, our method stores and samples the pre-extracted channel features at the user end, which not only provides a solution for downlink CSI acquisition that usually involves more challenging spatial domain interpolation, but also completely circumvents the privacy issues.

The main contributions of this paper are as follows:

• We conduct a detailed theoretical analysis of the channel model, treat geometric path parameters as easily obtainable channel features, and propose the geometryaided channel deduction (GCD) framework based on our analysis. This framework employs a channel deduction network to fuse geometric features extracted from the DT with pilot-based partial estimates, thereby deriving the complete channel.

• We introduce random prompt augmentation, an innovative method to construct CSI prompts from simple structural information. This method not only converts geometric structural features into a complete CSI-like representation, helping the network fuse channel information from different modalities, but also suppresses other unknown features, enabling the network to fully leverage dominant structural information.

• We present two neural network implementations for channel information fusion. One is GCDNet, which builds upon existing advanced learning architecture; the other is mGCDNet, which integrates pilot positional information to support variable pilot configurations. By enriching the training data diversity, mGCDNet improves versatility and enables more comprehensive learning.

• We conduct extensive experiments to evaluate the proposed schemes. The results demonstrate their excellent channel acquisition accuracy and strong robustness against non-ideal geometric information. Furthermore, by incorporating multiple scenarios into the training process, we further enhance their overall learning performance and generalization capability.

The remainder of this paper is organized as follows. Section II introduces the channel model and analyzes the availability of various channel features. Based on this analysis, Section III proposes the GCD framework and details its implementation. Section IV provides performance evaluation of proposed schemes. Finally, Section V concludes this paper.

## II. SYSTEM MODEL

## A. Electromagnetic Wave Propagation

This subsection formulates the channel response of a singlefrequency wave propagating from a transmitting antenna to a receiving antenna [38, Chapter 3].

The direction-dependent radiation pattern of an antenna is defined as $g ( \widehat { k } ) ~ = ~ g ( \widehat { k } ) \widehat { \psi } ( \widehat { k } )$ , where $\widehat { k }$ is a 3D unit vector representing the radiation direction, $g ( \widehat { \pmb k } )$ is the complex amplitude gain, and $\widehat { \psi } ( \widehat { k } )$ represents the antenna polarization, $\mathrm { i . e . }$ , the oscillation orientation of the transmitted field. $g ( \hat { k } )$ is normalized such that $g ( \widehat { \pmb { k } } ) = 1$ corresponds to a lossless isotropic antenna<sup>1</sup>. Under this definition, the electric far field of an antenna is proportional to $( 1 / d ) e ^ { - j k d } g ( \widehat { \pmb { k } } )$ ), where d is the propagation distance and k is the wavenumber.

Generally, the propagation channel consists of $N _ { \mathrm { p } }$ paths, where the $p { \cdot } \mathrm { t h }$ path has the following properties: path length $d _ { p } ,$ , departure direction $\widehat { k } _ { \mathrm { T } , p }$ at the transmitter, arrival direction $k _ { \mathrm { R } , p }$ at the receiver, and scattering matrix $\Xi _ { p } = \xi _ { p } \Psi _ { p } \in$ $\mathbb { C } ^ { 3 \times 3 }$ , where $\xi _ { p } \in \mathbb { C }$ represents the attenuation and phase shift due to EM interactions between the wave and the scattering environment, and $\begin{array} { l l l } { \Psi _ { p } } & { \in } & { \mathbb { C } ^ { 3 \times 3 } } \end{array}$ accounts for the relative attenuations and phase shifts between different polarization components of the field. For the line-of-sight (LoS) path, $\Xi _ { p }$ is the identity matrix, i.e., $\Xi _ { p } = \mathbf { I }$

<sup>1</sup>Formally, $g ( \widehat { \pmb k } )$ must satisfy $\int _ { 0 } ^ { 2 \pi } \int _ { 0 } ^ { \pi } | g ( \widehat { \pmb { k } } ) | ^ { 2 }$ sin $\theta \mathrm { d } \theta \mathrm { d } \varphi = 4 \pi \eta .$ , where $\underset {  } { \theta } , \varphi$ are the zenith and azimuth angles defining the radiation direction $\begin{array} { r } { \widehat { \pmb { k } } = [ \sin \theta \cos \varphi , } \end{array}$ sin θ sin $\varphi ,$ cos $\theta ] ^ { \mathsf { T } } ,$ and $\eta$ is the antenna efficiency, i.e., the proportion of the input power that is converted into radiation.

The channel frequency response can be expressed as

$$
h = \sum _ { p = 1 } ^ { N _ { \mathrm { p } } } \alpha _ { p } e ^ { - j 2 \pi f \tau _ { p } } ,\tag{1}
$$

where f is the frequency, $\tau _ { p } = d _ { p } / c$ is the propagation delay of the p-th path, and c denotes the speed of light. Let the radiation patterns of the transmitting and receiving antennas be ${ \pmb g } _ { \mathrm { T } } ( \widehat { \pmb k } ) = g _ { \mathrm { T } } ( \widehat { \pmb k } ) \widehat { \psi } _ { \mathrm { T } } ( \widehat { \pmb k } )$ and $g _ { \mathrm { R } } ( \widehat { \boldsymbol { k } } ) = g _ { \mathrm { R } } ( \widehat { \boldsymbol { k } } ) \widehat { \psi } _ { \mathrm { R } } ( \widehat { \boldsymbol { k } } )$ , respectively. Then the path coefficient $\alpha _ { p } \in \mathbb { C }$ is given by

$$
\alpha _ { p } = \frac { \lambda } { 4 \pi d _ { p } } \pmb { g } _ { \mathrm { { R } } } ( \widehat { \pmb { k } } _ { \mathrm { { R } } , p } ) ^ { \sf H } \Xi _ { p } \pmb { g } _ { \mathrm { { T } } } ( \widehat { \pmb { k } } _ { \mathrm { { T } } , p } ) ,\tag{2}
$$

which accounts for the propagation distance, antenna patterns, EM interactions between the wave and the environment, as well as the polarization mismatch between the incident wave and the receiving antenna.

## B. Channel Model

We consider a MIMO system employing orthogonal frequency division multiplexing (OFDM), where a base station (BS) equipped with $N _ { \mathrm { t } }$ antennas serves single-antenna users via $N _ { \mathrm { c } }$ subcarriers. The downlink channel from the BS to a single user is represented by the spatial-frequency domain channel matrix $\mathbf { H } \in \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$

The frequency offset of the $n _ { \mathrm { c } } { \mathrm { - } } \mathrm { t h }$ subcarrier is $\begin{array} { r l } { \Delta f _ { n _ { \mathrm { c } } } } & { { } = } \end{array}$ $n _ { \mathrm { c } } \Delta _ { \mathrm { f } } .$ , where $\Delta _ { \mathrm { f } }$ is the subcarrier spacing. Let $d _ { n _ { \mathrm { t } } }$ denote the relative position of the $n _ { \mathrm { t } } { \mathrm { - } } \mathrm { t h }$ BS antenna with respect to the local origin. Assuming that the wavefront at the BS is locally planar, the antenna’s relative position results in the delay difference of $\Delta \tau _ { n _ { \mathrm { t } } , p } ~ = ~ - d _ { n _ { \mathrm { t } } } ^ { \top } \widehat { k } _ { \mathrm { T } , p } / c$ . By introducing frequency offsets and delay differences into (1), the frequency response for the $n _ { \mathrm { t } }$ -th antenna and $n _ { \mathrm { c } } { \mathrm { - } } \mathrm { t h }$ subcarrier is

$$
\mathbf { H } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ] = \sum _ { p = 1 } ^ { N _ { \mathrm { p } } } \alpha _ { p } e ^ { - j 2 \pi ( f + \Delta f _ { n _ { \mathrm { c } } } ) ( \tau _ { p } + \Delta \tau _ { n _ { \mathrm { t } } , p } ) }\tag{3}
$$

$$
\approx \sum _ { p = 1 } ^ { N _ { \mathrm { p } } } \alpha _ { p } e ^ { - j 2 \pi f \tau _ { p } } e ^ { - j 2 \pi n _ { \mathrm { c } } \Delta _ { \mathrm { f } } \tau _ { p } } e ^ { j k d _ { n _ { \mathrm { t } } } ^ { \intercal } \widehat { k } _ { \mathrm { T } , p } } .\tag{4}
$$

To estimate the real-time channel, $N _ { \mathrm { t } , \pi }$ antennas and $N _ { \mathrm { c } , \pi }$ subcarriers are used to transmit non-precoded pilots, enabling the acquisition of the partial channel ${ \bf H } _ { \pi } = { \bf H } [ \Omega ] \in$ $\mathbb { C } ^ { { N _ { \mathrm { t } , \pi } } \times { N _ { \mathrm { c } , \pi } } }$ , where $\Omega ~ = ~ \Omega _ { \mathrm { t } } \times \Omega _ { \mathrm { c } }$ is a subset of antennasubcarrier indices, and × denotes the Cartesian product. We assume that the pilots are evenly spaced in the $N _ { \mathrm { t } } \times N _ { \mathrm { c } }$ spatialfrequency grid. Therefore, the pilot pattern can be expressed in the following form:

$$
\begin{array} { r l r } & { } & { \Omega _ { \mathrm { t } } = S _ { \mathrm { t } } + \{ 0 , R _ { \mathrm { t } } , \cdots , ( N _ { \mathrm { t } , \pi } - 1 ) \times R _ { \mathrm { t } } \} , } \\ & { } & { S _ { \mathrm { t } } \in \{ 1 , 2 , \cdots , R _ { \mathrm { t } } \} , \quad R _ { \mathrm { t } } = N _ { \mathrm { t } } / N _ { \mathrm { t } , \pi } , } \\ & { } & { \Omega _ { \mathrm { c } } = S _ { \mathrm { c } } + \{ 0 , R _ { \mathrm { c } } , \cdots , ( N _ { \mathrm { c } , \pi } - 1 ) \times R _ { \mathrm { c } } \} , } \\ & { } & { S _ { \mathrm { c } } \in \{ 1 , 2 , \cdots , R _ { \mathrm { c } } \} , \quad R _ { \mathrm { c } } = N _ { \mathrm { c } } / N _ { \mathrm { c } , \pi } , } \end{array}\tag{5a}
$$

(5b)

where $R _ { \mathrm { t } }$ and $S _ { \mathrm { t } }$ denote the interval and the start of antenna indices in the spatial domain, while $R _ { \mathrm { c } }$ and $S _ { \mathrm { c } }$ denote those of subcarrier indices in the frequency domain.

Specifically, within each coherence time block, during which the channel state is approximately constant, the BS transmits pilot symbols on each subcarrier in $\Omega _ { \mathrm { c } }$ through $N _ { \mathrm { q } }$ transmissions $( N _ { \mathrm { q } } \mathrm { ~ \ge ~ } N _ { \mathrm { t } , \pi } )$ . In the $q \mathrm { - }$ th transmission, the received signal at the user on the $n _ { \mathrm { c } } ^ { }$ -th subcarrier $( n _ { \mathrm { c } } \in \Omega _ { \mathrm { c } } )$ is given by $y _ { q , n _ { \mathrm { c } } } = h _ { n _ { \mathrm { c } } } ^ { \mathsf T } x _ { q , n _ { \mathrm { c } } } + w _ { q , n _ { \mathrm { c } } }$ , where $\begin{array} { r } { h _ { n _ { \mathrm { c } } } \ = \ \mathbf { H } [ \cdot } \end{array}$ $, n _ { \mathrm { c } } ] \in \mathbb { C } ^ { N _ { \mathrm { t } } }$ is the channel response on the $n _ { \mathrm { c } } { \cdot }$ -th subcarrier, $\mathbf { x } _ { q , n _ { \mathrm { c } } } \in \mathbb { C } ^ { N _ { \mathrm { t } } }$ is the pilot vector for the q-th transmission, which satisfies $\| \pmb { x } _ { q , n _ { \mathrm { c } } } \| _ { 2 } ^ { 2 } = P$ , with P being the transmit power, and $w _ { q , n _ { \mathrm { c } } } \sim \mathcal { C N } ( 0 , \sigma _ { \mathrm { w } } ^ { 2 } )$ is the thermal noise, with $\sigma _ { \mathrm { w } } ^ { 2 }$ being the thermal noise power. Based on a pilot transmission scheme similar to that in [39], we assume $N _ { \mathrm { q } } = N _ { \mathrm { t } , \pi }$ , and on each subcarrier in $\Omega _ { \mathrm { c } }$ , the antennas in $\Omega _ { \mathrm { t } }$ transmit pilot symbols successively through the $N _ { \mathrm { q } }$ transmissions, with the $n _ { \mathrm { t } } { \mathrm { - } } \mathrm { t h }$ antenna $( n _ { \mathrm { t } } \in \Omega _ { \mathrm { t } } )$ corresponding to the $q ( n _ { \mathrm { t } } )$ -th transmission. In this case, each pilot vector ${ \pmb x } _ { q , n _ { \mathrm { c } } }$ has only one non-zero element $x _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } }$ corresponding to the $n _ { \mathrm { t ^ { \prime } } }$ -th antenna, and the channel $h _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } } = \mathbf { H } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ]$ at pilot positions can be simply estimated as $h _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } } ^ { \prime } = y _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } } / x _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } } = h _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } } + \widetilde { w } _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } } ,$ where $\smash { \widetilde { w } _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } } = w _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } } / x _ { q ( n _ { \mathrm { t } } ) , n _ { \mathrm { c } } } \sim \mathcal { C N } ( 0 , \widetilde { \sigma } _ { \mathrm { w } } ^ { 2 } ) }$ is the estimation noise, $\tilde { \sigma } _ { \mathrm { w } } ^ { 2 } = \sigma _ { \mathrm { w } } ^ { 2 } / P$ is the estimation noise power. In this paper, by default, we consider the ideal noise-free case where the clean partial CSI $\mathbf { H } _ { \pi }$ can be obtained.

## C. Obtainable Channel Features

We rewrite the MIMO-OFDM channel (4) as follows:

$$
\mathbf { H } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ] = \boldsymbol { \phi } _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } } ^ { \top } \widetilde { \boldsymbol { \alpha } } ,\tag{6}
$$

where $\phi _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } } , \widetilde { \alpha } \in \mathbb { C } ^ { N _ { \mathrm { p } } }$ are $N _ { \mathrm { p } }$ -dimensional vectors, whose p-th elements are respectively defined as

$$
\phi _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } , p } = e ^ { - j 2 \pi n _ { \mathrm { c } } \Delta _ { \mathrm { f } } \tau _ { p } } e ^ { j k { \pmb d } _ { n _ { \mathrm { t } } } ^ { \top } \widehat { \pmb k } _ { \mathrm { T } , p } } ,
$$

$$
\widetilde { \alpha } _ { p } = \alpha _ { p } e ^ { - j 2 \pi f \tau _ { p } } .\tag{7}
$$

(8)

The first component of the channel model, $\phi _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } }$ , contains the phase offset caused by the $n _ { \mathrm { t } } { \mathrm { - } } \mathrm { t h }$ antenna and $n _ { \mathrm { c } } { \mathrm { - } } \mathrm { t h }$ subcarrier, which is determined by $\{ ( d _ { p } , \widehat { \pmb { k } } _ { \mathrm { T } , p } ) \} _ { p = 1 } ^ { N _ { \mathrm { p } } }$ and $( f , \Delta _ { \mathrm { f } } , d _ { n _ { \mathrm { t } } } )$ The path parameters $\{ ( d _ { p } , \widehat { \pmb { k } } _ { \mathrm { T } , p } ) \} _ { p = 1 } ^ { N _ { \mathrm { p } } }$ , which serve as the geometric features describing the channel structure, can be obtained using ray tracing as long as the environmental map and the positions of the BS and the user are available. In practical applications, the BS can obtain the environmental map through radio sensing technologies or geographic information system (GIS), while the BS and the user can obtain their respective positions via global navigation satellite system (GNSS). However, for privacy reasons, the BS and users may be reluctant to disclose their positions externally. It should be noted that the acquired geometric information may exhibit errors that render the high-frequency phase term $e ^ { - j 2 \pi f \tau _ { p } }$ unpredictable [40]. Hence, we exclude $e ^ { - j 2 \pi f \tau _ { p } }$ from the expression of $\phi _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } }$ and incorporate it into the second component $\widetilde { \alpha } .$ As for $( f , \Delta _ { \mathrm { f } } , d _ { n _ { \mathrm { t } } } )$ , these are fixed system configurations determined by frequency band allocation and antenna array layout, and we assume they can be obtained with precision. Consequently, $\phi _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } }$ can be computed using the available information.

In contrast, the second component, ${ \widetilde { \alpha } } ,$ is typically unobtainable. To gain deeper insights into αe, we substitute $\alpha _ { p }$ in (8) with (2), yielding the following complete expression:

$$
\widetilde { \alpha } _ { p } = \frac { \lambda } { 4 \pi d _ { p } } g _ { \mathrm { R } } ( \widehat { k } _ { \mathrm { R } , p } ) ^ { \sf H } \Xi _ { p } g _ { \mathrm { T } } ( \widehat { k } _ { \mathrm { T } , p } ) e ^ { - j 2 \pi f \tau _ { p } } ,\tag{9}
$$

![](images/6e23c62711254e50d28d3539715d0af24c1062959d5a49c26f1e881601fd8ef0.jpg)  
Fig. 1: Overview of the proposed channel acquisition framework.

which is the product of several factors. The only factor that can be explicitly obtained is $\lambda / ( 4 \pi d _ { p } )$ , which describes the large-scale attenuation determined by the propagation distance. The scattering matrix $\Xi _ { p }$ depends on the EM properties of environmental materials, while the antenna patterns $g _ { \mathrm { T } } ( \widehat { k } _ { \mathrm { T } , p } )$ and $g _ { \mathrm { R } } ( \widehat { k } _ { \mathrm { R } , p } )$ depend on hardware conditions and antenna orientations. These factors are typically difficult to obtain. Furthermore, as mentioned earlier, the high-frequency term $e ^ { - j 2 \pi f \tau _ { p } }$ is unpredictable due to geometric errors.

From the above analysis, it can be seen that $\phi _ { n _ { \mathrm { t } } , n _ { \mathrm { c } } }$ represents the known structural features projected onto the $n _ { \mathrm { t } }$ -th antenna and $n _ { \mathrm { c } }$ -th subcarrier, whereas $\widetilde { \alpha }$ represents unknown information shared across different antennas and subcarriers. Therefore, although the channel data has a high dimension of $O ( N _ { \mathrm { t } } N _ { \mathrm { c } } )$ , the dimension of the underlying unknown features is quite low. Thus, by fully leveraging available geometric prior information, the unknown features can be compensated with minimal pilot overhead, thereby achieving complete channel acquisition.

## III. PROPOSED FRAMEWORK

To leverage available structural features and improve channel acquisition efficiency, we propose a novel framework named geometry-aided channel deduction (GCD). As shown in Fig. 1, the BS maintains an uncalibrated DT as the scenario representation, which consists of the BS position and an approximate environmental map. The user retrieves geometric features from the scenario representation based on its own position and converts them into the scenario prompt using known system configurations, thereby providing contextual information about the scenario. Meanwhile, the instantaneous channel is partially estimated by the user based on the received signal and known pilots. Finally, a neural network deployed at the user side fuses the partial channel estimate with the scenario prompt to generate the complete channel.

The following subsections detail the proposed framework, which is illustrated in Fig. 2(a).

## A. Geometric Feature Extraction

Although the multi-path structure can be obtained via ray tracing based on the environmental map and transceiver positions, there are several drawbacks to directly applying this method. First, ray tracing requires positional information from both the BS and the user, which raises privacy concerns because regardless of which one performs the ray tracing, the other must disclose its position. Furthermore, the user position is constantly changing, which means ray tracing must be repeatedly performed based on real-time user positions. This consumes substantial computational resources and results in significant inference latency [34].

Considering these issues, we enable the BS to pre-extract multi-path features offline without requiring actual user positions. We first sample N virtual user positions $\pmb { x } _ { 1 } , \cdots , \pmb { x } _ { N }$ in the environmental map, ensuring these positions roughly cover the activity area of potential users. We then perform ray tracing using these positions to obtain the geometric feature set ${ \mathcal { F } } =$ $\{ ( \mathcal { P } _ { 1 } , { \pmb x } _ { 1 } ) , \cdot \cdot \cdot , ( \mathcal { P } _ { N } , { \pmb x } _ { N } ) \}$ , where $\begin{array} { r c l } { \mathcal { P } _ { i } } & { = } & { \{ ( d _ { i , p } , \widehat { \pmb { k } } _ { i , p } ) \} _ { p = 1 } ^ { N _ { \mathrm { p } , i } } } \end{array}$ represents the channel structure at $\mathbf { \Delta } _ { \mathbf { x } _ { i } , \mathbf { \xi } } N _ { \mathrm { p } , i }$ denotes the number of paths from the BS to ${ \mathbf { x } } _ { i } ,$ and $d _ { i , p } , \hat { k } _ { i , p }$ represent the length and departure direction of each path, respectively. Since $\mathcal { F }$ contains only path parameters at a finite number of discrete positions, $\mathcal { F }$ serves as a compact scenario representation with a small data volume and can be provisioned to the user during initial access and maintained at the user side. For multiple users within the same scenario, the same $\mathcal { F }$ can also be shared via broadcast or multicast, thereby reducing redundant transmission overhead.

Once the user has obtained the complete information of ${ \mathcal F } ,$ the user is able to search for the nearest neighbors among the virtual user positions $\pmb { x } _ { 1 } , \cdots , \pmb { x } _ { N }$ within $\mathcal { F }$ based on its own position $^ { \mathbf { \delta x } , }$ with its positional information used only locally. We denote the n nearest virtual user positions as $\pmb { x } _ { i _ { 1 } } , \cdots , \pmb { x } _ { i _ { n } }$ and their corresponding channel structure as $\mathcal { P } _ { i _ { 1 } } , \cdots , \mathcal { P } _ { i _ { n } } .$ Given the high similarity of scattering environment within the spatial neighborhood, the structural features at position x can be well approximated by those at its neighbors.

## B. Random Prompt Augmentation

Our proposed framework needs to fuse the geometric features and the partial CSI, which belong to different modalities and exhibit significant differences in data format. To ensure more learnable intra-modal fusion, we convert the geometric features $\mathcal { P } _ { i } ~ ( i \ = ~ i _ { 1 } , \cdot \cdot \cdot , i _ { n } )$ into the complete channel representation $ { \widetilde { \mathbf { H } } } _ { i } ~ \in ~ \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ , which serves as the contextual prompt that can be directly fused with the partial CSI $\mathbf { H } _ { \pi }$ . This modality alignment process appears similar to the process in vision-language models that converts images into the text representation (i.e., tokens). However, unlike the semantic relationship between text and images that is primarily determined by human cognition, the relationship between path parameters and channel responses is strictly constrained by the physical laws [41]. Simply applying a learnable module to transform these two modalities into the same feature space cannot guarantee compliance with the physical constraints, thereby reducing learning efficiency. Instead, we propose random prompt augmentation, which manually computes a pseudo channel $\widetilde { \mathbf { H } } _ { i }$ based on the channel model using the path parameters $\mathcal { P } _ { i }$ , thus strictly adhering to the physical constraints and perfectly transforming the path parameters into a channel representation. The subsequent neural network can process the pseudo channel the same as real channels, thereby achieving seamless modality alignment.

![](images/32bb05bf05695a4d716777b4b6b845cf836157f81db915564190c72fe76177d0.jpg)  
Fig. 2: (a) Detailed diagram of the proposed framework. (b) Comparison of the original pre-mapping module in [27] and the proposed pre-mapping module

We assume that $\dot { \bf H } _ { i } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ]$ has a form similar to that in (6), i.e., $\phi _ { i , n _ { \mathrm { t } } , n _ { \mathrm { c } } } ^ { \mathsf { T } } \widetilde { \pmb { \alpha } } _ { i }$ . Here, the first component $\phi _ { i , n _ { \mathrm { t } } , n _ { \mathrm { c } } } \in \mathbb { C } ^ { N _ { \mathrm { p } , i } }$ is obtainable given the multi-path structure and system configurations, and its p-th element can be calculated as in (7):

$$
\phi _ { i , n _ { \mathrm { t } } , n _ { \mathrm { c } } , p } = e ^ { - j 2 \pi n _ { \mathrm { c } } \Delta _ { \mathrm { f } } \tau _ { i , p } } e ^ { j k { \pmb d } _ { n _ { \mathrm { t } } } ^ { \mathsf { T } } \widehat { \pmb k } _ { i , p } } ,\tag{10}
$$

and $\tau _ { i , p } ~ = ~ d _ { i , p } / c$ . Based on our previous analysis of (9), each element $\widetilde { \alpha } _ { i , p }$ of the second component $\widetilde { \alpha } _ { i }$ is the product of a large-scale attenuation factor $\lambda / ( 4 \pi d _ { i , p } )$ and a series of unknown factors. To handle the unknown channel parameters in $\widetilde { \alpha } _ { i }$ , we introduce a placeholder vector $ { \widetilde { z } } _ { i } \in \mathbb { C } ^ { N _ { \mathrm { p } , i } }$ to replace $\widetilde { \alpha } _ { i } ,$ , and compute $\widetilde { \mathbf { H } } _ { i }$ as follows:

$$
\widetilde { \mathbf { H } } _ { i } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ] = \boldsymbol { \phi } _ { i , n _ { \mathrm { t } } , n _ { \mathrm { c } } } ^ { \intercal } \widetilde { z } _ { i } ,\tag{11}
$$

where each element of $\widetilde { z } _ { i }$ is modeled as a random variable obeying the complex Gaussian distribution:

$$
\widetilde { z } _ { i , p } = \frac { \lambda } { 4 \pi d _ { i , p } } z _ { i , p } , \quad z _ { i , p } \sim \mathcal { C N } ( 0 , \sigma _ { \mathrm { z } } ^ { 2 } ) ,\tag{12}
$$

and $\sigma _ { z }$ is a hyperparameter that controls the magnitude scale of the unknown factors. Our method is robust to the selection of $\sigma _ { \mathrm { z } } .$ In our experiments, $\sigma _ { z }$ is set to 0.5.

The random placeholders introduced here serve multiple purposes. First, they supplement unknown information within the channels, converting known multi-path structure into the complete channel representation. This enables the subsequent neural network to fuse different channel information within the same feature space. Second, by introducing different realizations of the random placeholder for each pseudo channel, the neural network suppresses the influence of these unknown channel parameters, thereby focusing on exploiting useful known structural information shared by all pseudo channels. This is because the neural network is unlikely to learn regular patterns from random data, thus avoiding the learning of spurious correlations between the placeholders and the real channel. This strategy is analogous to data augmentation in neural network training, which prevents the neural network from overfitting to unimportant features within data samples. However, in our method, data augmentation is not applied to training samples but rather to the input prompt, enabling the network to identify important features within the contextual information.

## C. Channel Deduction Network

1) GCDNet: Following [27], we fuse the pseudo channels with the partial channel using a channel deduction network and refer to it as GCDNet. It first pre-maps the partial CSI $\mathbf { H } _ { \pi }$ to the form of a complete CSI $\mathbf { H } _ { \mathrm { P M } } \in \bar { \mathbb { C } } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ . Then, the network fuses features from $\widetilde { { \mathbf H } } _ { i _ { 1 } } , \dots , \widetilde { { \mathbf H } } _ { i _ { n } }$ and H<sub>PM</sub> to obtain a more informative representation H<sub>FM</sub> $\mathbf { \Psi } \in \mathbb { C } ^ { N _ { \mathrm { { t } } } \times N _ { \mathrm { { c } } } }$ . Finally, ${ \bf { H } } _ { \mathrm { { F M } } }$ is further refined to yield the final output Hb .

The channel pre-mapping module and the refinement module process channel data in spatial-frequency domain. Both modules are implemented using CMixer [12], whose network structure is illustrated in Fig. 3(a). CMixer treats the input $\mathbf { H } _ { \mathrm { i n } } \in \mathbb { C } ^ { N _ { \mathrm { t , i n } } \times N _ { \mathrm { c , i n } } }$ and the output $\mathbf { H } _ { \mathrm { o u t } } \in \mathbb { C } ^ { N _ { \mathrm { t , o u t } } \times N _ { \mathrm { c } } }$ <sup>,out</sup> as real-valued tensors of shapes $N _ { \mathrm { t , i n } } \times N _ { \mathrm { c , i n } } \times 2$ and $N _ { \mathrm { t , o u t } } \times$ $N _ { \mathrm { c , o u t } } \times 2$ , respectively. It comprises K stacked interleaved layers for channel representation learning, as well as two interleaved linear projections for dimension transformation. The detailed structure of the interleaved layer is shown in Fig. 3(c). Its unique interleaved learning design closely aligns with the intrinsic structures of channel data, enabling CMixer to outperform existing learning methods in channel-related tasks with high parameter efficiency. In the channel deduction network, a K-layer CMixer is used to implement the pre-mapping module PM : $\mathbb { C } ^ { N _ { \mathrm { t } , \pi } \times N _ { \mathrm { c } , \pi } } \to \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ , converting the partial CSI $\mathbf { H } _ { \pi }$ to ${ \bf H } _ { \mathrm { P M } } = \mathrm { P M } ( { \bf H } _ { \pi } )$ . Meanwhile, another K-layer CMixer serves as the refinement module RM : $\mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } } \xrightarrow [ ] { } \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ generating the output $\widehat { \mathbf { H } } = \mathrm { R M } ( \mathbf { H } _ { \mathrm { F M } } )$

The channel information fusion module is implemented using attention mechanism, which enables mutual interaction of channel features from $\widetilde { \mathbf { H } } _ { i _ { 1 } } , \dots , \widetilde { \mathbf { H } } _ { i _ { n } }$ and $\mathbf { H } _ { \mathrm { P M } }$ . The detailed network structure is illustrated in Fig. 3(d)-(e). The input channels are first reshaped and compressed via linear projection, transforming them into D-dimensional real-valued vectors. Learnable embeddings are added to these vectors to distinguish the features from instantaneous estimate or pseudo channels. Subsequently, these vectors are fed into an L-layer Transformer encoder for information interaction. Finally, the last output vector corresponding to $\mathbf { H } _ { \mathrm { P M } }$ is transformed to the output channel via linear projection and reshaping. The entire process is expressed as $\mathbf { H } _ { \mathrm { F M } } = \mathrm { F M } ( \tilde { \mathbf { H } } _ { i _ { 1 } } , \cdot \cdot \cdot , \tilde { \mathbf { H } } _ { i _ { n } } , \mathbf { H } _ { \mathrm { P M } } )$ where FM : $\dot { \mathbb { C } } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } } \times \dot { \dots } \times \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } } \xrightarrow { \mathrm { ~ } } \dot { \mathbb { C } } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ denotes the fusion module.

![](images/107ae7279a9b1342ad20b4d4f4a61c0c6592967768031bbf5bb1eaa8637722bb.jpg)  
Fig. 3: Specific structures of the modules used in the channel deduction network. The plus sign “+” denotes residual connection, and “C” denotes concatenation

2) Mask-Enhanced GCDNet: The above implementation of the pre-mapping module takes only partial CSI $\mathbf { H } _ { \pi }$ as input, ignoring information about the pilot settings. Consequently, the entire network can only be applied to a fixed pilot pattern Ω, i.e., fixed intervals $R _ { \mathrm { t } } , R _ { \mathrm { c } }$ and fixed starts $S _ { \mathrm { t } } , S _ { \mathrm { c } }$

To enhance the network’s universality, we incorporate pilot positional information into the pre-mapping module. We define the pilot mask $\mathbf { M } \in \{ 0 , 1 \} ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ as follows, which indicates the pilot positions in the spatial-frequency grid.

$$
\mathbf { M } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ] = \left\{ 1 , \quad ( n _ { \mathrm { t } } , n _ { \mathrm { c } } ) \in \Omega , \right.\tag{13}
$$

According to the pilot positions, the partial CSI $\mathbf { H } _ { \pi }$ can be padded with zeros to obtain a sparse channel matrix $\widetilde { \textbf { H } } =$ H ⊙ $\mathbf { M } \in \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ , where ⊙ denotes the Hadamard product.

We extend the network structure for pre-mapping, as shown in Fig. 3(b). The padded CSI He is first converted to a realvalued tensor of shape $N _ { \mathrm { t } } { \times } N _ { \mathrm { c } } { \times } 2$ , and then concatenated with the mask M to obtain a composite tensor of shape $N _ { \mathrm { t } } \times N _ { \mathrm { c } } \times 3$ Then it passes through K interleaved learning layers, which are quite similar to the original CMixer. Finally, we employ $1 \times 1$ convolution with 3 input channels and 2 output channels to transform the data shape from $N _ { \mathrm { t } } \times N _ { \mathrm { c } } \times 3$ into $N _ { \mathrm { t } } \times N _ { \mathrm { c } } \times 2$ We refer to this extended structure as mask-enhanced CMixer (mCMixer). The output is expressed as ${ \bf H } _ { \mathrm { P M } } = \mathrm { m P M } ( \tilde { \bf H } , { \bf M } )$ where mPM : $\mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } } \times \{ 0 , 1 \} ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }  \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ denotes the extended pre-mapping module.

By replacing the original pre-mapping module with a $K \mathfrak { - }$ layer mCMixer, the extended GCDNet is capable of processing partial channels with various pilot patterns. A comparison between the original pre-mapping module and the extended pre-mapping module is illustrated in Fig. 2(b). We refer to this extended network as mask-enhanced GCDNet (mGCDNet).

## D. Training and Inference

The training of (m)GCDNet requires a dataset D collected in the communication scenario, where each data sample (H, x) contains the complete CSI and user position. Additionally, this scenario should provide a geometric feature set $\mathcal { F }$ as the scenario representation. Thanks to the scenario-related context introduced by the geometric prior, (m)GCDNet supports strong adaptation and generalization across various scenarios. Therefore, we adopt multi-scenario collaborative training to leverage the rich data from different scenarios for more comprehensive and thorough learning. Assume M different scenarios are involved in training, each with its own training dataset and geometric feature set. For the special case when $M \ = \ 1$ collaborative learning reduces to single-scenario learning, a quite common setting in existing studies. Furthermore, after introducing the mask-enhanced pre-mapping module, mGCDNet is capable of handling different pilot patterns. We introduce multiple pilot configurations $\{ \Omega _ { \ell } \} _ { }$ <sub>ℓ</sub> during training to enhance the network’s universality and enable more thorough learning.

At each training step, we uniformly sample a scenario index m from $\{ 1 , \cdots , M \}$ , and then sample a CSI-position pair $( \mathbf { H } , \pmb { x } )$ from the m-th scenario’s dataset. Meanwhile, we sample Ω from all pre-defined pilot patterns $\{ \Omega _ { \ell } \} _ { \ell }$ . Using the complete CSI H and the pilot pattern $\Omega ,$ we can derive the partial CSI $\mathbf { H } _ { \pi }$ , the padded CSI He , and the pilot mask M, which serve as inputs to the channel deduction network. Subsequently, we follow the aforementioned procedure to acquire the complete CSI Hb . To mitigate the impact of the absolute magnitude of channel responses, we normalize the CSI data before feeding it into the network. Specifically, we first use known partial CSI to estimate the average power, denoted as $P _ { \bf H } = \| \widetilde { \bf H } \| _ { \mathrm { F } } ^ { 2 } / \| { \bf M } \| _ { 1 }$ , and then normalize each input channel matrix H via $\mathbf { H }  \mathbf { H } / \sqrt { P _ { \mathbf { H } } }$ . This ensures that the network inputs have a unit average power, which helps stabilize the training process and reduces the learning burden. The network is trained using the Adam optimizer, and the loss function is defined as the mean squared error (MSE) between the normalized H, Hb , i.e., $\mathcal { L } = \Vert \mathbf { H } - \widehat { \mathbf { H } } \Vert _ { \mathrm { F } } ^ { 2 } / ( N _ { \mathrm { t } } N _ { \mathrm { c } } )$

![](images/623ade2c02a357ed957393e789ff3aa4a68cb301ef8033884b932088383f90ea.jpg)  
Fig. 4: Satellite images of the four cities.

TABLE II: Different settings of pilot patterns.
<table><tr><td>Setting</td><td>Interval</td><td>Start</td></tr><tr><td>FRFS</td><td> $R _ { \mathrm { t } } = 4$   $R _ { \mathrm { c } } = 1 6$ </td><td> $S _ { \mathrm { t } } = 1$   $S _ { \mathrm { c } } = 1$ </td></tr><tr><td></td><td> $R _ { \mathrm { t } } = 4$ </td><td> $S _ { \mathrm { t } } \in \{ 1 , 2 , \cdots , R _ { \mathrm { t } } \}$ </td></tr><tr><td>FRVS VRVS</td><td> $R _ { \mathrm { c } } = 1 6$   $R _ { \mathrm { t } } \in \{ 2 , 4 , 8 \}$ </td><td> $S _ { \mathrm { c } } \in \{ 1 , 2 , \cdots , R _ { \mathrm { c } } \}$   $S _ { \mathrm { t } } \in \{ 1 , 2 , \cdots , R _ { \mathrm { t } } \}$ </td></tr></table>

During inference, we normalize the network inputs using the estimated magnitude $\sqrt { P _ { \mathbf { H } } }$ , similar to the training phase. To recover the acquired channel with absolute magnitude scale, the output CSI can be denormalized via Hb $ \sqrt { P _ { \mathrm { H } } } \hat { \mathbf { H } }$

Since ray tracing and feature extraction are performed offline in our approach, online inference only requires neighborhood search, pseudo CSI construction, and the neural network forward pass. Assuming that neighborhood search is implemented by identifying n minimum elements among the N distances between x and each of $\pmb { x } _ { 1 } , \cdots , \pmb { x } _ { N }$ , the overall computational complexity for neighborhood search is $O ( N + N \log n ) = O ( N \log n )$ . The complexity for pseudo CSI construction is $O ( N _ { \mathrm { t } } N _ { \mathrm { c } } N _ { \mathrm { p } } n )$ , where $N _ { \mathrm { p } }$ is the maximum number of paths, while the forward pass of (m)GCDNet has a computational complexity of $O ( K ( N _ { \mathrm { c } } N _ { \mathrm { t } } ^ { 2 } + N _ { \mathrm { t } } N _ { \mathrm { c } } ^ { 2 } ) +$ $N _ { \mathrm { t } } N _ { \mathrm { c } } D n + L ( D ^ { 2 } n + D n ^ { 2 } ) )$ [27].

## IV. SIMULATION RESULTS

## A. Experimental Settings

1) Scenario Setup: We collect the environmental maps of four global cities from OpenStreetMap, including San Francisco, Shanghai, Singapore, and London. For each city, we manually set three BS positions, constructing 12 scenarios. The satellite images of these four cities and the 12 constructed scenarios are shown in Fig. 4 and Fig. 5, respectively. These environmental maps are imported into Sionna RT [42] to generate channel data and geometric features. The materials of the ground, building walls, and building roofs are set as concrete, marble, and metal, respectively. The propagation paths with no more than 5 reflections are enabled. In each scenario, the users are randomly distributed within a 400 m × 400 m square area centered on the BS. User heights range from 1 m to 2 m. Virtual user positions are sampled on a regular cell grid with a cell size of 1 m × 1 m and a height of 1.5 m. The BS is equipped with a uniform linear array (ULA) with $N _ { \mathrm { t } } = 1 6$ array elements. All users are equipped with randomly oriented dipole antennas to simulate diverse antenna radiation characteristics. The system center frequency is $f = 5 \mathrm { G H z } ,$ and the bandwidth is set to 40 MHz and divided into $N _ { \mathrm { c } } = 2 5 6$ subcarriers.

![](images/71dac91d2aafdc85c1c179f1d794da2f68ee0cbf7ed255c572261a4e55118565.jpg)  
Fig. 5: Environmental maps of the 12 scenarios.

TABLE III: Settings of network structure parameters.
<table><tr><td>Scheme</td><td>Network Module</td><td>Paremeter settings</td></tr><tr><td>(m)GCDNet</td><td>(m)CMixer Transformer Encoder</td><td> $K = 3 , N _ { \mathrm { t } } ^ { \prime } = N _ { \mathrm { t } } , N _ { \mathrm { c } } ^ { \prime } = N _ { \mathrm { c } }$   $L = 6 , D \stackrel { \cdot } { = } 5 1 2$ </td></tr><tr><td>(m)PCMixer</td><td>CMixer ResMLP</td><td> $\widetilde { K } = 8 , N _ { \mathrm { t } } ^ { \prime } = N _ { \mathrm { t } } , N _ { \mathrm { c } } ^ { \prime } = N _ { \mathrm { c } }$   $\widetilde { L } = 6 , D = 5 1 2$ </td></tr><tr><td>(m)CMixer</td><td></td><td> $\widetilde { K } = 8 , N _ { \mathrm { t } } ^ { \prime } = N _ { \mathrm { t } } , N _ { \mathrm { c } } ^ { \prime } = N _ { \mathrm { c } }$ </td></tr></table>

We consider the following pilot pattern settings, with details shown in Table II. Each setting defines a collection of pilot patterns $\{ \Omega _ { \ell } \} _ { \ell }$ used in training.

• Fixed R & fixed S (FRFS): In each domain (space or frequency), both the pilot interval and start are fixed. In this case, there is only one possible pilot pattern.

• Fixed R & variable S (FRVS): In each domain, the pilot interval is fixed, while the start is variable. In this case, the number of pilots is fixed, but these pilots are inserted at variable positions in the spatial-frequency grid.

• Variable R & variable S (VRVS): In each domain, both the pilot interval and start are variable.

TABLE IV: Number of parameters and FLOPs of networks in different schemes, evaluated under $\dot { R } _ { \mathrm { t } } = 4 ,$ $R _ { \mathrm { c } } = 1 6$
<table><tr><td rowspan="2">Scheme</td><td colspan="2">Mask-free</td><td colspan="2">Mask-enhanced</td></tr><tr><td>Parameters</td><td>FLOPs</td><td>Parameters</td><td>FLOPs</td></tr><tr><td>(m)GCDNet</td><td>21.85 M</td><td>610.76 M</td><td>24.73 M</td><td>708.48 M</td></tr><tr><td>(m)PCMixer</td><td>15.33 M</td><td>182.38 M</td><td>21.56 M</td><td>194.83 M</td></tr><tr><td>(m)CMixer</td><td>4.51 M</td><td>152.84 M</td><td>10.69 M</td><td>362.21 M</td></tr><tr><td>(m)PCMixer-L</td><td>43.27 M</td><td>1.25 G</td><td>55.72 M</td><td>1.27 G</td></tr><tr><td>(m)CMixer-L</td><td>17.44 M</td><td>1.16 G</td><td>40.32 M</td><td>2.69 G</td></tr></table>

2) Baselines and Parameter Settings: We adopt two advanced learning schemes for channel acquisition as baselines. One is CMixer-based channel mapping [12], which utilizes a Ke -layer CMixer to generate the complete channel using only pilot-based estimates. The other is PCMixer [28], which first processes the user position and partial channel separately using two Le-layer ResMLPs with hidden size D, followed by $1 \times 1$ convolutional fusion, and finally generates the full channel via a Ke -layer CMixer for refinement. The original implementations of CMixer and PCMixer only support fixed pilot pattern. To better align with our work, we extend these two baselines to their mask-enhanced versions. For CMixerbased channel mapping, we introduce mCMixer proposed in the previous section to support channel mapping with variable pilot patterns. For PCMixer, we replace its input $\mathbf { H } _ { \pi }$ with the concatenation of He and M, yielding mask-enhanced PCMixer (mPCMixer). The detailed parameter settings of these network structures are shown in Table III. Besides, we double the hidden dimensions of the network modules in (m)PCMixer and (m)CMixer, resulting in additional baselines named (m)PCMixer-L and (m)CMixer-L, which possess a significantly higher level of complexity than (m)GCDNet.

We employ different numbers of training epochs for the three pilot pattern settings in Table II due to their difference in training data diversity. Specifically, we train each network for 1000 epochs under the FRFS setting, for 4000 epochs under the FRVS setting, and for 10000 epochs under the VRVS setting. The batch size is set to 500. The learning rate is initially set to $1 0 ^ { - 4 }$ and decayed by a factor of 0.8 every 1/10 of the training duration. During training, the number of pseudo channels n is randomly selected from 0 to 16. During testing, n is set to 16 by default. Table IV shows the number of parameters and the floating-point operations (FLOPs) of networks in the above schemes<sup>2</sup>.

In addition, we introduce generative channel estimation as another baseline, which employs diffusion model based posterior sampling (DMPS) [43]. We implement this scheme based on the implementation of [13], except that we replace the denoising network, originally a CNN, with more advanced diffusion Transformer (DiT) [44]. In our experiment, the DiT consists of 8 DiT blocks with a hidden size of 512 and is trained for 10000 epochs.

To demonstrate the advantages of deep learning methods over traditional signal processing algorithms, we also compare with the classical linear minimum mean squared error (LMMSE) channel estimation, where the covariance matrix is computed based on training data. Furthermore, we incorporate the idea of basis projection proposed in [35] into LMMSE, thereby forming another baseline, LMMSE-BP, which similarly utilizes environmental geometry as prior knowledge.

3) Evaluation Metrics: We evaluate the channel acquisition quality for each data sample using normalized MSE (NMSE) and cosine correlation $\rho ,$ which are defined as

$$
\mathrm { N M S E } = \Vert \mathbf { H } - \widehat { \mathbf { H } } \Vert _ { \mathrm { F } } ^ { 2 } / \Vert \mathbf { H } \Vert _ { \mathrm { F } } ^ { 2 } ,\tag{14}
$$

$$
\rho = \frac { 1 } { N _ { \mathrm { c } } } \sum _ { n _ { \mathrm { c } } = 1 } ^ { N _ { \mathrm { c } } } \frac { | \widehat { \pmb { h } } _ { n _ { \mathrm { c } } } ^ { \ H } { \pmb { h } } _ { n _ { \mathrm { c } } } | } { \| \widehat { \pmb { h } } _ { n _ { \mathrm { c } } } \| _ { 2 } \| { \pmb { h } } _ { n _ { \mathrm { c } } } \| _ { 2 } } ,\tag{15}
$$

where $h _ { n _ { \mathrm { c } } } , \widehat { h } _ { n _ { \mathrm { c } } }$ are the $n _ { \mathrm { c } }$ -th columns of $\mathbf { H } , { \widehat { \mathbf { H } } } ,$ respectively. To translate the channel acquisition accuracy into meaningful communication gains, we use the average achievable rate (AAR) to evaluate the performance of precoding based on the acquired CSI, which is defined as

$$
\mathrm { A A R } = \frac { 1 } { N _ { \mathrm { c } } } \sum _ { n _ { \mathrm { c } } = 1 } ^ { N _ { \mathrm { c } } } \log _ { 2 } \left( 1 + \frac { P } { \sigma _ { \mathrm { w } } ^ { 2 } } | h _ { n _ { \mathrm { c } } } ^ { \top } v _ { n _ { \mathrm { c } } } | ^ { 2 } \right) ( \mathrm { b p s / H z } ) ,\tag{16}
$$

where ${ \pmb v } _ { n _ { \mathrm { c } } } \in \mathbb { C } ^ { N _ { \mathrm { t } } }$ denotes the precoding vector for the $n _ { \mathrm { c } ^ { - } }$ th subcarrier, which is assumed to have a unit power, i.e., $\| \pmb { v } _ { n _ { \mathrm { c } } } \| _ { 2 } ^ { 2 } = 1$ . For simplicity, we employ maximum ratio transmission (MRT), whose precoding vector is given by $v _ { n _ { \mathrm { c } } } = \widehat { h } _ { n _ { \mathrm { c } } } ^ { * } / \| \widehat { h } _ { n _ { \mathrm { c } } } \| _ { 2 }$ , where the acquired channel $\widehat { h } _ { n _ { \mathrm { c } } }$ accounts for estimation noise (with a power of $\widetilde \sigma _ { \mathrm { w } } ^ { 2 } = \sigma _ { \mathrm { w } } ^ { 2 } / P )$ . As a reminder, $P$ and $\sigma _ { \mathrm { w } } ^ { 2 }$ represent the BS transmit power and thermal noise power, respectively. The thermal noise power can be calculated as $\sigma _ { \mathrm { w } } ^ { 2 } = k T B$ , where $k = 1 . 3 8 \times 1 0 ^ { - \bar { 2 } 3 }$ J/K is the Boltzmann constant, $T = 2 9 0 \mathrm { K }$ is the temperature, and B is the system bandwidth.

## B. Single-Scenario Learning

In this subsection, we train and evaluate all schemes in a single scenario – San Francisco BS 1. The training, validation, and testing datasets comprise 40 k, 10 k, and 10 k data samples, respectively.

1) Training under Different Pilot Pattern Settings: To verify the effectiveness of the mask enhancement, we evaluate the channel acquisition performance with networks trained under different pilot pattern settings, i.e., FRFS, FRVS, and VRVS. For fair comparison, all schemes are evaluated under $R _ { \mathrm { t } } = 4 .$ $R _ { \mathrm { c } } = 1 6$ . The NMSE performance of mask-free schemes are shown in Fig. 6(a). When trained and evaluated with fixed pilot pattern (FRFS), CMixer struggles to perform effectively due to the complex multi-path characteristics and limited pilot resources. In contrast, both PCMixer and GCDNet utilize additional information, thereby achieving superior accuracy. However, under the FRVS setting, all these schemes fail because they do not utilize the necessary information about pilot patterns. Worse still, these schemes are not applicable to the VRVS setting, as their network structures strictly determine the partial channel size, thus only supporting fixed pilot interval.

![](images/f639f36a7aaa2e292a87c85e21b3d7d3e4c7ea19d814a7ac9d459e490dc7736b.jpg)  
(a)

![](images/06a59be365674746365ffd007ec63345dd693813dc05f36d4bb2bb55b69792fe.jpg)  
(b)

![](images/1c263514b5892e0e8ead5f345f02405bb8f5e62292b504e7ed61786b9fca0959.jpg)  
(c)  
Fig. 6: NMSE performance with networks trained under different pilot pattern settings. In the box plots, the boxes extend from the first quartile (Q1) to the third quartile (Q3) with a line at the median, and the whiskers extend from the 10th percentile to the 90th percentile. (a) NMSE box plot of mask-free schemes. (b) NMSE box plot of mask-enhanced schemes. (c) Median NMSE of mask-enhanced schemes under different noise conditions.

TABLE V: NMSE and $\rho$ performance with networks trained under different pilot pattern settings.
<table><tr><td rowspan="3">Scheme</td><td colspan="4">FRFS</td><td colspan="4">FRVS</td><td colspan="4">VRVS</td></tr><tr><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>mGCDNet</td><td>-8.60</td><td>-14.45</td><td>0.9412</td><td>0.9854</td><td>-9.95</td><td>-17.46</td><td>0.9614</td><td>0.9940</td><td>-9.69</td><td>-16.37</td><td>0.9602</td><td>0.9924</td></tr><tr><td>mPCMixer</td><td>-11.98</td><td>-16.65</td><td>0.9712</td><td>0.9903</td><td>-9.29</td><td>-14.68</td><td>0.9550</td><td>0.9876</td><td>-7.53</td><td>-12.01</td><td>0.9359</td><td>0.9801</td></tr><tr><td>mCMixer</td><td>-1.17</td><td>-1.47</td><td>0.6001</td><td>0.7424</td><td>-6.18</td><td>-12.24</td><td>0.9034</td><td>0.9805</td><td>-6.77</td><td>-12.45</td><td>0.9061</td><td>0.9825</td></tr><tr><td>mPCMixer-L</td><td>-12.97</td><td>-18.47</td><td>0.9782</td><td>0.9944</td><td>-8.85</td><td>-13.67</td><td>0.9518</td><td>0.9857</td><td>-6.44</td><td>-9.67</td><td>0.9264</td><td>0.9700</td></tr><tr><td>mCMixer-L</td><td>-1.58</td><td>-2.55</td><td>0.6440</td><td>0.8134</td><td>-8.61</td><td>-17.12</td><td>0.9410</td><td>0.9937</td><td>-7.02</td><td>-12.65</td><td>0.9122</td><td>0.9852</td></tr></table>

The NMSE performance of mask-enhanced schemes are shown in Fig. 6(b). All schemes perform well under the FRVS and VRVS settings. Notably, the performance of mGCDNet and mCMixer under the FRVS and VRVS settings is even better than under the FRFS setting, indicating that variable pilot patterns enrich the training data and facilitate more comprehensive learning. Furthermore, it can be observed that under the FRVS and VRVS settings, the proposed mGCDNet achieves the best performance among all schemes, even though it has much fewer parameters and FLOPs than mPCMixer-L or mCMixer-L. Table V presents the quantitative results under different settings, further validating the superiority of the proposed mGCDNet under variable pilot configurations.

In Fig. 6(c), we evaluate mask-enhanced schemes with variable pilot patterns under different noise conditions. The noisy estimate can be expressed as $\mathbf { H } _ { \pi } ^ { \prime } = \mathbf { H } _ { \pi } + \mathbf { W }$ , where $\mathbf { W } \in \mathbb { C } ^ { N _ { \mathrm { t } , \pi } \times N _ { \mathrm { c } , \pi } }$ is the additive noise matrix whose elements are independently sampled from ${ \mathcal C } \mathcal { N } ( 0 , \widetilde { \sigma } _ { \mathrm { w } } ^ { 2 } )$ , with $\widetilde { \sigma } _ { \mathrm { w } } ^ { 2 }$ being the estimation noise power. We define the signal-to-noise ratio (SNR) as $P _ { \mathbf { H } } / \widetilde { \sigma } _ { \mathrm { w } } ^ { 2 }$ . As SNR decreases, the performance of all schemes degrades. Notably, mGCDNet exhibits stronger robustness to input noise under VRVS compared to FRVS, which once again validates the advantages of enriching training pilot configurations.

In the remainder of this section, we evaluate the maskenhanced schemes that are trained under the VRVS setting, as it improves performance, robustness, and universality.

2) Performance under Different Pilot Intervals: Fig. 7(a) illustrates the performance of different schemes under various pilot intervals $( R _ { \mathrm { t } } , R _ { \mathrm { c } } )$ . To provide a more comprehensive demonstration, we select three groups of pilot intervals $( R _ { \mathrm { t } } , R _ { \mathrm { c } } )$ , plotting the cumulative distribution function (CDF) of NMSE under the ideal noise-free condition in Fig. 7(b)- (d) and the NMSE performance under different SNR in Fig. 7(e)-(g). Table VI shows the detailed quantitative results under the noise-free condition. Clearly, DMPS and LMMSE can work effectively only when pilot resources are abundant (e.g., $R _ { \mathrm { t } } ~ = ~ 2 , ~ R _ { \mathrm { c } } ~ = ~ 4 )$ , because when only a small number of pilots are observed, neither the diffusion model nor the covariance matrix provides sufficient prior information to determine the complete CSI. Although LMMSE-BP improves LMMSE’s performance by utilizing basis projection based on geometric prior, it relies heavily on the initial performance of LMMSE channel estimation. Therefore, poor LMMSE performance hinders the performance improvement of LMMSE-BP. mCMixer also exhibits severe performance degradation under conditions with very few pilots (e.g., $R _ { \mathrm { t } } = 8 , R _ { \mathrm { c } } = 6 4 )$ . In contrast, mPCMixer and mGCDNet operate properly across various pilot intervals by utilizing auxiliary information, and mGCDNet consistently outperforms the other schemes.

In the remainder of this section, all schemes are evaluated under $R _ { \mathrm { t } } = 4 , R _ { \mathrm { c } } = 1 6$ to strike a balance between channel acquisition quality and pilot overhead.

3) Impact of Prompt Length and Position Error: Fig. 8(a) illustrates mGCDNet’s performance under various prompt lengths, i.e., the number of neighboring samples n. As n increases, the performance of mGCDNet first improves and then tends to plateau. Even with only a single neighboring sample, mGCDNet still outperforms other schemes by a wide margin. Recall that the pseudo channels He <sub>i</sub> $( i = i _ { 1 } , \cdots , i _ { n } )$ are derived from the geometric features of n neighboring samples in ${ \mathcal F } ,$ with each sample assigned a random placeholder $\widetilde { z } _ { i } .$ Here, we try a different strategy: instead of searching for n neighboring samples, we use only the nearest one and repeat its geometric features for n times, assigning a different placeholder each time. The results are shown by the dashed lines in Fig. 8(a). We observe that the performance difference between reusing a single neighbor n times and using n neighbors is negligible. This suggests that the performance gain from increasing n primarily stems from the multiple realizations of the random placeholder.

![](images/2d6ca1619ac96d27795e09be67ca9c2e441a4e1b6b25cfb4b7f38cc7c785aaf4.jpg)

Fig. 7: NMSE performance under different pilot intervals. (a) Median NMSE under various groups of $( R _ { \mathrm { t } } , R _ { \mathrm { c } } )$ . (b)-(d) Cumulative probability distribution of NMSE with $\mathbf { \bar { \rho } } ( R _ { \mathrm { t } } , R _ { \mathrm { c } } ) \in \{ ( 8 , 6 4 )$ , (4, 16), (2, 4)}. (e)-(g) Median NMSE under different noise conditions with $( R _ { \mathrm { t } } , R _ { \mathrm { c } } ) \in \{ ( 8 , 6 \bar { 4 } ) , ( 4 , 1 6 ) , ( 2 , 4 ) \}$ TABLE VI: NMSE and ρ performance under various pilot intervals
<table><tr><td rowspan="3">Scheme</td><td colspan="4"> $R _ { \mathrm { t } } = 8 , R _ { \mathrm { c } } = 6 4$ </td><td colspan="4"> $R _ { \mathrm { t } } = 4 , R _ { \mathrm { c } } = 1 6$ </td><td colspan="4"> $R _ { \mathrm { t } } = 2 , R _ { \mathrm { c } } = 4$ </td></tr><tr><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td>mGCDNet</td><td>-7.70</td><td>-12.90</td><td>0.9437</td><td>0.9857</td><td>-9.69</td><td>-16.37</td><td>0.9602</td><td>0.9924</td><td>-10.43</td><td>-17.36</td><td>0.9643</td><td>0.9938</td></tr><tr><td>mPCMixer</td><td>-5.59</td><td>-9.18</td><td>0.9143</td><td>0.9694</td><td>-7.53</td><td>-12.01</td><td>0.9359</td><td>0.9801</td><td>-8.48</td><td>-12.79</td><td>0.9424</td><td>0.9819</td></tr><tr><td>mCMixer</td><td>-1.91</td><td>-4.28</td><td>0.6987</td><td>0.8982</td><td>-6.77</td><td>-12.45</td><td>0.9061</td><td>0.9825</td><td>-9.47</td><td>-14.82</td><td>0.9492</td><td>0.9890</td></tr><tr><td>DMPS</td><td>-0.01</td><td>-0.01</td><td>0.2418</td><td>0.1584</td><td>-0.12</td><td>-0.12</td><td>0.3619</td><td>0.3499</td><td>-1.32</td><td>-7.87</td><td>0.6611</td><td>0.9518</td></tr><tr><td>LMMSE</td><td>0.07</td><td>0.18</td><td>0.3106</td><td>0.3176</td><td>0.05</td><td>0.75</td><td>0.4773</td><td>0.4814</td><td>-4.59</td><td>-8.32</td><td>0.8187</td><td>0.9253</td></tr><tr><td>LMMSE-BP</td><td>-0.23</td><td>-0.10</td><td>0.8626</td><td>0.9259</td><td>-1.52</td><td>-0.68</td><td>0.9163</td><td>0.9664</td><td>-7.65</td><td>-12.33</td><td>0.9593</td><td>0.9947</td></tr></table>

![](images/20479b57e25a9e14318c0a89de5c5deec396ad0aa9eb539220911cb469a3d2b7.jpg)  
(a)

![](images/82d949de266e9f0f12b147fa5c4e861a13da59454206627d0a241cfd37888067.jpg)  
(b)  
Fig. 8: Impact of prompt length (i.e., the number of neighboring samples) and position error. (a) Median NMSE under different numbers of neighboring samples. (b) Median NMSE under different scales of user position errors.

Next, we introduce errors to user positions. The approximate position is represented as ${ \widehat { \pmb { x } } } = { \pmb x } + \Delta { \pmb x }$ , where ∆x is the position error following a 2D uniform distribution on $[ - l , l ] \times$ $[ - l , l ]$ , with l being the error $\mathrm { s c a l e } ^ { 3 }$ . The results are shown in Fig. 8(b). As the position error increases, the performance of both mGCDNet and mPCMixer degrades. When $l < 4  { \mathrm { m } }$ mGCDNet consistently demonstrates a performance advantage over the other schemes, performing well within the positioning accuracy achievable by existing systems. We also observe that using n distinct neighboring samples exhibits slightly stronger robustness against position errors than reusing the nearest neighbor n times. This is because, even if the position used for neighborhood sampling is significantly biased, multiple neighboring samples may still form a region that covers the true position and provides approximate structural features [8].

TABLE VII: NMSE and ρ performance in two example scenarios.
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Scheme</td><td colspan="2">NMSE (dB)</td><td colspan="2">ρ</td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td rowspan="3">San Francisco BS1</td><td>mGCDNet</td><td>-11.48</td><td>-19.08</td><td>0.9678</td><td>0.9955</td></tr><tr><td>mPCMixer</td><td>-9.59</td><td>-15.61</td><td>0.9515</td><td>0.9908</td></tr><tr><td>mCMixer</td><td>-5.29</td><td>-12.84</td><td>0.8505</td><td>0.9820</td></tr><tr><td rowspan="3">Singapore BS1</td><td>mGCDNet</td><td>-8.28</td><td>-13.64</td><td>0.9298</td><td>0.9859</td></tr><tr><td>mPCMixer</td><td>0.31</td><td>0.09</td><td>0.1828</td><td>0.0944</td></tr><tr><td>mCMixer</td><td>0.24</td><td>0.11</td><td>0.3519</td><td>0.2238</td></tr></table>

## C. Multi-Scenario Learning

In this subsection, we train each scheme using training data from the first 6 scenarios jointly and evaluate it across all 12 scenarios. The training, validation, and testing datasets for each scenario contain 20 k, 10 k, and 10 k data samples, respectively.

1) Performance in Original and New Scenarios: Fig. 9(a) shows the NMSE performance of different schemes across the first 6 scenarios involved in training. Both mGCDNet and mPCMixer utilize auxiliary information to enhance channel acquisition accuracy and achieve multi-scenario joint learning, with mGCDNet consistently outperforming mPCMixer. Fig. 9(b) shows the NMSE performance across the remaining 6 scenarios, where all schemes are evaluated directly without additional training or finetuning. mPCMixer and mCMixer completely fail in these unseen scenarios. In contrast, mGCD-Net demonstrates excellent generalization ability, achieving satisfactory NMSE performance on most testing samples in the new scenarios. Table VII presents the quantitative results of different schemes in San Francisco BS 1 and Singapore BS 1, which serve as examples of the learned scenarios and the unseen scenarios, respectively. These results further validate the superiority of the proposed mGCDNet.

![](images/b97c82673e77ed75b432f8ef6e10167c348728ae32b95df176e2f4fbe2f71c29.jpg)  
(a)

![](images/f276b857e3728d71dec02eaaff1efa3d686ec57fae22c2614e6f430e9f37e008.jpg)  
(b)

Fig. 9: NMSE performance in multi-scenario learning. (a) NMSE box plots in learned scenarios. (b) NMSE box plots in unseen scenarios without additional training.  
![](images/052ee468a6058181004ac8dae17e54fa1c284866d80b8b38761112f84c6d711d.jpg)  
(a)

![](images/8a5a9cea615fafc17cad530c4a9766f47b84146cbce5f92aa34888663fdfa4a1.jpg)  
(b)  
Fig. 10: AAR performance under different BS transmit power values. (a) Mean AAR in a learned scenario. (b) Mean AAR in an unseen scenario.

Besides, we introduce AAR as an additional evaluation metric. Fig. 10 illustrates the precoding performance under different BS transmit power values P in the two example scenarios. mGCDNet exhibits particular advantage in unseen scenarios, with AAR values comparable to those in learned scenarios and significantly outperforming other schemes. This result demonstrates the feasibility of mGCDNet for precoding in cross-scenario applications.

2) Performance under Non-Ideal Geometric Information: To verify mGCDNet’s robustness against non-ideal geometric information, we directly test mGCDNet in one of the unseen scenarios – Singapore BS 1. First, we consider the following cases of inaccurate environmental maps:

• Real-world traffic flow: There are moving vehicles that affect the channel, but the environmental map captures only static buildings. In our experiment, we manually place several vehicles in the environmental map, as shown in Fig. 11(a) (left). We then regenerate the channel data while keeping the geometric features unchanged.

• Missing buildings: Some buildings are missing from the environmental map. We manually remove some buildings near the base station to ensure a significant change in the propagation paths, as shown in Fig. 11(a) (middle), and then regenerate the geometric features.

• Incorrect building heights: Some buildings in the environmental map lack accurate height information. In this case, we manually adjust the heights of some buildings, as shown in Fig. 11(a) (right), and then regenerate the geometric features.

• Building position errors: The building positions in the environmental map are biased. Similar to inaccurate user positions, we independently introduce biased positions for each building in the environmental map. We then regenerate the geometric features.

The results are shown in Fig. 11(b). Under traffic conditions, mGCDNet exhibits little performance change. This is because, compared to static background buildings, moving vehicles are relatively small in size and thus have a relatively minor impact on the channel. In contrast, non-ideal information regarding the buildings has a greater impact on mGCDNet’s performance, as these cases may alter the number or visibility of propagation paths. Nevertheless, mGCDNet maintains satisfactory performance on most testing samples.

In Fig. 12(a)-(b), we introduce errors to the BS position and user positions. It can be observed that the performance exhibits a similar trend as the BS/user position error increases.

Next, we consider the imperfect ray tracing results. In Fig. 12(c), we evaluate the impact of the number of paths by limiting the maximum number and removing exceeding paths. As the number of retained paths decreases, the NMSE performance of mGCDNet gradually declines. Meanwhile, the NMSE performance also depends on the lengths of the retained paths, as shorter paths usually exhibit less attenuation and have a greater impact on the channel. Hence, retaining shorter paths results in a smaller performance decline compared to retaining the same number of longer paths. In Fig. 12(d), we evaluate the impact of path depth. i.e., the number of interactions with the environment. As the maximum depth decreases, the NMSE performance of mGCDNet also degrades. Overall, mGCDNet maintains stable performance when the number of retained paths exceeds 8 to 16 and the maximum depth is at least 3.

3) Finetuning in New Scenarios: When deployed in unseen scenarios, performance degradation on hard testing samples is inevitable due to the significant difference in data distribution compared to the previous scenarios involved in training. Nevertheless, by leveraging the common knowledge gained from pre-training in previous scenarios, mGCDNet can efficiently adapt to these new scenarios with only a small amount of training data and computational resources. To illustrate this point, we perform a few steps of finetuning using data from the new scenario Singapore BS 1. As shown in Fig. 13(a), even after hundreds of finetuning steps, the performance of mCMixer and mPCMixer in the new scenario remains inferior to the initial performance of mGCDNet without any finetuning. Furthermore, as shown in Fig. 13(b), both mCMixer and mPCMixer exhibit catastrophic forgetting in previous scenarios, resulting in severe performance degradation. In contrast, mGCDNet benefits from multi-scenario pre-training, enabling it to quickly adapt to new scenarios while maintaining outstanding performance in previous ones. More specifically, with only a small number of finetuning steps, mGCDNet can rapidly generalize well to those hard samples in new scenarios, thereby achieving more concentrated NMSE distribution.

![](images/973f267891d5e3da795f495a4c63fe77a575828ae619e071e29b1bfd9bcd1d20.jpg)

![](images/1bb9a776c1ef62dbf5ad536678d379ab62fa664d476639e7a3c9e285e883fb0f.jpg)  
(a)

![](images/23e135abceb1dfe5fcf407d2c19f55c5afe01ba1b41ee5592e9b74b561e5a243.jpg)

![](images/4de84f3b5dee922d695002f99bb42450cf5ed1d24dca857aada95994ab971977.jpg)  
(b)  
Fig. 11: Experiments on non-ideal environmental geometry. (a) Example cases of the environmental map. (b) NMSE box plot of mGCDNet in various cases

![](images/bec2330e7fdf5f21625614c6dd1b0019032bd8307f55f9c3ebee847778257d98.jpg)  
(a)

![](images/f96a4bad8ecc21a8a749c107d24b4111c5fcd72b9af64f9dde78ed2d83cb57be.jpg)

![](images/3c41611f0a725762fce85a9959c652ed8afa7dddbdb7cfea2f3260118ef2d640.jpg)  
(c)

(b)  
![](images/e82b6685830d23935843c63fec74881e5a087dba25750794ac537e8113d57338.jpg)  
(d)

Fig. 12: NMSE performance of mGCDNet under BS/user position errors and imperfect ray tracing results, where the median NMSE and quartiles are presented. (a)-(b) NMSE under different scales of user/BS position errors. (c) NMSE under different limits of path number. (d) NMSE under different limits of path depths.  
![](images/467bd8405776a2179bc0cde31fc711493b3484032654be33f4b3227c3058b9f8.jpg)  
(a)

![](images/52ba72bf9c8aab94eb998212bb76cb715bd1524533987ec482f97c5131c0f90e.jpg)  
(b)  
Fig. 13: NMSE performance after finetuning in the new scenario Singapore BS 1. (a) NMSE in the new scenario. (b) NMSE in a previous scenario.

TABLE VIII: System configurations of different scenarios.
<table><tr><td colspan="2">Scenario</td><td>Frequency</td><td>Bandwidth</td><td colspan="2">BS antenna</td></tr><tr><td rowspan="3">San Francisco</td><td>BS1</td><td>5GHz</td><td>40 MHz</td><td> $1 \times 1 6$ </td><td>Short dipole</td></tr><tr><td>BS 2</td><td>3.5 GHz</td><td>20 MHz</td><td> $1 \times 1 6$ </td><td>Short dipole</td></tr><tr><td>BS 3</td><td>5 GHz</td><td>20 MHz</td><td> $4 \times 4$ </td><td>λ/2 dipole</td></tr><tr><td rowspan="3">Shanghai</td><td>BS1</td><td>6.7 GHz</td><td>50 MHz</td><td> $\overline { { 2 \times 8 } }$ </td><td>Isotropic</td></tr><tr><td>BS 2</td><td>2.4 GHz</td><td>20 MHz</td><td> $2 \times 8$ </td><td>TR 38.901</td></tr><tr><td>BS3</td><td>5.9 GHz</td><td>46MHz</td><td> $4 \times 4$ </td><td>TR 38.901</td></tr><tr><td rowspan="3">Singapore</td><td>BS1</td><td>3.5 GHz</td><td>20 MHz</td><td> $2 \times 8$ </td><td>λ/2 dipole</td></tr><tr><td>BS 2</td><td>5 GHz</td><td>34MHz</td><td> $4 \times 4$ </td><td>λ/2 dipole</td></tr><tr><td>BS3</td><td>6.7 GHz</td><td>40 MHz</td><td> $1 \times 1 6$ </td><td>TR 38.901</td></tr><tr><td rowspan="3">London</td><td>BS1</td><td>4.6 GHz</td><td>20 MHz</td><td> $4 \times 4$ </td><td>TR 38.901</td></tr><tr><td>BS 2</td><td>5 GHz</td><td>40 MHz</td><td> $4 \times 4$ </td><td>Isotropic</td></tr><tr><td>BS3</td><td>2.4 GHz</td><td>20 MHz</td><td> $1 \times 1 6$ </td><td>Short dipole</td></tr><tr><td colspan="2">ASU Campus</td><td>3.5 GHz</td><td>20 MHz</td><td> $1 \times 1 6$ </td><td>Isotropic</td></tr></table>

![](images/30dfaa2b3085ac9efdcec512dd0742553f932583a7727e7720244547bf55f47e.jpg)  
(a)

![](images/3c115588cf5f6bb9a3fc19dee7d7a61ef13cee4303e37e8020a923b6acfd3a8c.jpg)  
(b)  
Fig. 14: NMSE performance in multi-scenario learning. (a) NMSE box plots in learned scenarios. (b) NMSE box plots in unseen scenarios without additional training.

4) Generalization across System Configurations: To evaluate the generalization capabilities across different system configurations, we generate another group of channel datasets, where each scenario has its own carrier frequency, bandwidth, BS antenna array geometry, and antenna pattern. We first employ Sionna RT to regenerate channel data based on the 12 scenarios with new system configurations. We enable the propagation paths with no more than 5 reflections. Additionally, to involve different ray tracing platforms and settings, we introduce the ASU Campus scenario from DeepMIMO [45], an outdoor scenario with channel data simulated using Wireless Insite. The satellite image of this scenario is shown in Fig. 15(a). The simulation considers paths containing up to 6 reflections, 1 diffraction, or 1 scattering event. The detailed system configurations of these scenarios are shown in Table VIII. We train each scheme in the first 6 scenarios and evaluate directly across all 13 scenarios, including the newly added ASU Campus.

TABLE IX: NMSE and ρ performance in three example scenarios.
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Scheme</td><td colspan="2">NMSE (dB)</td><td colspan="2"> $\rho$ </td></tr><tr><td>Mean</td><td>Median</td><td>Mean</td><td>Median</td></tr><tr><td rowspan="3">San Francisco BS1</td><td>mGCDNet</td><td>-9.75</td><td>-16.30</td><td>0.9568</td><td>0.9924</td></tr><tr><td>mPCMixer</td><td>-7.70</td><td>-14.05</td><td>0.9291</td><td>0.9876</td></tr><tr><td>mCMixer</td><td>-5.44</td><td>-11.49</td><td>0.8576</td><td>0.9772</td></tr><tr><td rowspan="3">Singapore BS1</td><td>mGCDNet</td><td>-8.14</td><td>-11.74</td><td>0.9399</td><td>0.9815</td></tr><tr><td>mPCMixer</td><td>0.50</td><td>0.12</td><td>0.1579</td><td>0.1030</td></tr><tr><td>mCMixer</td><td>-0.09</td><td>0.04</td><td>0.4783</td><td>0.4613</td></tr><tr><td rowspan="3">ASU Campus</td><td>mGCDNet</td><td>-4.85</td><td>-7.37</td><td>0.8540</td><td>0.9555</td></tr><tr><td>mPCMixer</td><td>0.58</td><td>0.08</td><td>0.1882</td><td>0.1172</td></tr><tr><td>mCMixer</td><td>0.75</td><td>0.35</td><td>0.2418</td><td>0.1336</td></tr></table>

![](images/25f6250faf672f286459bee7c2e9e354e440ecbf33cf0032cc721c94dffd0382.jpg)  
(a)

![](images/545a2afc80688d2aaf6d5e489e43a9d6b001d94233a5a3490850a6a8d4869b4b.jpg)  
(b)  
Fig. 15: Experiments on ASU Campus. (a) Satellite image of the scenario. (b) Mean AAR under different BS transmit power values.

Fig. 14(a) shows the NMSE performance across the 6 scenarios involved in training. mGCDNet consistently outperforms the other schemes. Fig. 14(b) shows the NMSE performance across 6 unseen scenarios. The baseline schemes completely fail in these new scenarios, while mGCDNet exhibits generalization capabilities on part of the testing samples, especially in Singapore BS 1. These testing samples may be covered by the data distribution from previous training scenarios, which is why the proposed method works. Thus, by enriching the training data and increasing scenario diversity to achieve broader coverage of the data distribution, the proposed method is expected to generalize to more testing samples in new scenarios and improve the overall performance. Alternatively, computation-efficient finetuning can be performed using data with new system configurations.

Fig. 15(b) shows the AAR performance of different schemes in ASU Campus. Similar to the results in Sionna RT scenarios, mGCDNet demonstrates significant advantages over baseline schemes. Table IX presents the quantitative results of different schemes in three example scenarios, further validating the superiority of the proposed mGCDNet.

![](images/58d1d2e98e97424362ec579d038be38c88a53e5070841bf47fad83638d2ac636.jpg)  
(a)

![](images/b248eadd7bdee1ca50b49a71fbf49650c83cfaf50033dc8d1fdf8dc3ce1c25c5.jpg)  
(b)

Fig. 16: Single-scenario learning performance of schemes with different prompt designs. (a) Cumulative probability distribution of NMSE. (b) Median NMSE under different prompt lengths.  
![](images/0f44c9becd3f862a979a2dc7fce04d21936822f2cc73ec1d993ff805cdac13d3.jpg)  
(a)

![](images/d250ba9ccb97f69c9548a013f8cc00ad1b87595c972f69ec647ebf633fd4d1df.jpg)  
(b)  
Fig. 17: NMSE performance of schemes with different prompt designs in multi-scenario learning. (a) NMSE box plots in learned scenarios. (b) NMSE box plots in unseen scenarios without additional training.

## D. Ablation Studies on Prompt Design

In this subsection, we design several alternative approaches to incorporate the geometric information and compare them with the proposed random prompt augmentation method. All these methods follow the same framework – fusing partial CSI with geometric features to acquire full CSI. The only difference among them lies in how the geometric features are handled. Through this comparison, we demonstrate that the proposed prompt design not only outperforms traditional channel estimation methods but also excels at leveraging geometric information compared with other geometry-aided methods. Specifically, we devise the following methods:

• Geometry-channel fusion (GCF): This scheme allows the neural network to learn on its own how to convert the geometric path parameters into an appropriate prompt that can be fused with the partial CSI. Since the number of paths is uncertain and there is no specific order among the different paths, we employ a Transformer without positional embedding to process the path parameters $\mathcal { P } _ { i } ,$ followed by average pooling to obtain a latent feature vector $\pmb { p } _ { i } \in \mathbb { R } ^ { D }$ . We use a shared Transformer to process the geometric features of all n neighbors, thereby obtaining n feature vectors $p _ { i _ { 1 } } , \cdots , p _ { i _ { n } }$ . These feature vectors serve as the contextual prompt and are fed into the channel information fusion module. We refer to this scheme as mGCFNet.

![](images/eb6430c94e3943b07b0237965726cd3ea31f66a9919cb27084464ef54082a637.jpg)  
(a)

![](images/cb11ce50d79399170357df5c1246b5d1f9a8e7b0184ba8d7a584a4c5eb99425b.jpg)  
(b)  
Fig. 18: Impact of prompt length on NMSE performance of schemes with different prompt designs. (a) Median NMSE in a learned scenario. (b) Median NMSE in an unseen scenario.

• Basis-channel fusion (BCF): This scheme draws inspiration from several existing studies on DT-aided channel acquisition [30], [35]. We select the path parameters $\mathcal { P } _ { i }$ of the nearest neighbor and compute a set of matrices $\{ \Phi _ { i , p } \} _ { p = 1 } ^ { { N _ { \mathrm { p } , i } } }$ , with each matrix $\bar { \Phi _ { i , p } } ~ \in ~ \mathbb { C } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ defined as $\Phi _ { i , p } \dot { [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ] } = \phi _ { i , n _ { \mathrm { t } } , n _ { \mathrm { c } } , p }$ (see (10)). These matrices form a basis (which may be non-orthogonal) of the channel subspace. We feed these $N _ { \mathrm { p } , i }$ basis matrices as the contextual prompt into the channel information fusion module. We refer to this scheme as mBCFNet.

• Deterministic pseudo channel construction: This method is similar to the proposed prompt design, with the difference being that the placeholder is not random. Specifically, we construct a deterministic pseudo channel $\hat { \mathbf { H } } _ { i } ^ { \mathrm { ( D ) } } \in \bar { \mathbb { C } } ^ { N _ { \mathrm { t } } \times N _ { \mathrm { c } } }$ by replacing the unknown $\widetilde { \alpha } _ { i }$ with a placeholder $\mathcal { \widetilde { z } } _ { i } ^ { \mathrm { ( D ) } }$

$$
\widetilde { \mathbf { H } } _ { i } ^ { \mathrm { ( D ) } } [ n _ { \mathrm { t } } , n _ { \mathrm { c } } ] = \boldsymbol { \phi } _ { i , n _ { \mathrm { t } } , n _ { \mathrm { c } } } ^ { \intercal } \widetilde { z } _ { i } ^ { \mathrm { ( D ) } } ,\tag{17}
$$

where each element of $\mathcal { \widetilde { z } } _ { i } ^ { \mathrm { ( D ) } }$ is

$$
\widetilde { z } _ { i , p } ^ { \mathrm { ( D ) } } = \frac { \lambda } { 4 \pi d _ { i , p } } z _ { i , p } ^ { \mathrm { ( D ) } } , \quad z _ { i , p } ^ { \mathrm { ( D ) } } = \frac { \sqrt { \pi } } { 2 } \sigma _ { \mathrm { z } } .\tag{18}
$$

Compared with (12), the variable $z _ { i , p } ^ { \mathrm { ( D ) } }$ here is a constant value instead of a random one. We set this value as the expected magnitude of the original random variable, i.e., $z _ { i , p } ^ { \mathrm { ( D ) } } = \mathbb { E } [ | z _ { i , p } | ]$ , with $z _ { i , p } \sim \mathcal { C N } ( 0 , \sigma _ { \mathrm { z } } ^ { 2 } )$ as defined in (12). We refer to mGCDNet with this prompt design as mGCDNet (D).

All these schemes adopt the same training setting as the proposed mGCDNet. We first train each scheme in a single scenario – San Francisco BS 1. Fig. 16(a) shows the CDF of NMSE. We can observe that mGCDNet performs best while mGCDNet (D) ranks second. Both mBCFNet and mGCFNet perform poorly. Fig. 16(b) investigates the impact of prompt length on performance. For mBCFNet, a prompt length of n means retaining the basis matrices corresponding to at most n shortest paths; for the other three schemes, the prompt length represents the number of neighboring samples. We observe that as the prompt length increases, the performance of mBCFNet and mGCFNet actually declines. This suggests that these two schemes fail to effectively utilize the auxiliary geometric information within the prompt.

Next, we train each scheme in the first 6 scenarios and evaluate across all 12 scenarios. The results are shown in Fig. 17. Compared with the single-scenario case, the performance of mBCFNet and mGCFNet improves due to multi-scenario learning, and mBCFNet even outperforms mGCDNet (D). Fig. 18(a) and Fig. 18(b) illustrate the impact of prompt length on performance in learned and unseen scenarios, respectively. In the learned scenario, the performance of all schemes improves as the prompt length increases, indicating that all schemes can utilize the contextual information within the prompt. This contrasts with the results in single-scenario learning in Fig. 16(b), likely because these schemes have to learn to exploit the contextual information within the prompt during training to accommodate multiple scenarios. In the unseen scenario, mGCDNet, mGCDNet (D), and mBCFNet perform well with the assistance of contextual prompts, whereas mGCFNet completely fails. This may be because mGCFNet does not leverage the channel model and can only learn to convert the geometric information to CSI prompt based on training data, thus lacking generalization capabilities in new scenarios where the data distribution changes drastically.

Overall, mGCDNet, mGCDNet (D), and mBCFNet incorporate the channel model into prompt design, successfully achieving multi-scenario collaborative learning and crossscenario generalization. Among these schemes, mGCDNet achieves the best performance in both single-scenario and multi-scenario cases, validating the effectiveness of the proposed random prompt augmentation method.

## V. CONCLUSION

In this paper, we propose a channel acquisition framework named GCD, which extracts geometric prior information from the DT and fuses it with pilot-based coarse estimates. We carefully design the method for scenario prompt generation and augmentation by considering the physical structure of channel data. For channel information fusion, we draw upon advanced learning architecture while enhancing it to support variable pilot configurations. Comprehensive experimental results validate the superiority of the proposed schemes, which achieve high-quality channel acquisition with low pilot overhead, strong robustness to biased geometric information, and significant cross-scenario generalization capability.

Our work highlights the importance of prompting and pretraining for wireless models. By providing necessary and accessible prompts – such as environmental geometry, pilot patterns, and system configurations – the model can undergo thorough pre-training using rich data from multiple scenarios with variable pilot patterns and diverse system configurations, thereby learning universal knowledge and in turn enhancing the model’s intelligence. Furthermore, modality transformation and network architecture design that adhere to the physical structure of wireless channels also contribute to improving the model’s learning efficiency. These insights may serve as references for future wireless model design.

[1] H. Ruan, Z. Zhang, Z. Chen et al., “Geometry-aided channel deduction: A robust channel acquisition framework utilizing coarse scenario prompt,” in IEEE Int. Symp. Person. Indoor Mobile Radio Commun. (PIMRC), 2026 (accepted).

[2] L. Lu, G. Y. Li, A. L. Swindlehurst et al., “An overview of massive MIMO: Benefits and challenges,” IEEE J. Sel. Top. Signal Process., vol. 8, no. 5, pp. 742–758, 2014.

[3] J. Zhang, Q. Li, K. Fang et al., “Spherical stacked intelligent metasurfaces: A paradigm for full-space wave-domain processing,” IEEE Commun. Mag., 2026.

[4] F. Liu, Y. Cui, C. Masouros et al., “Integrated sensing and communications: Toward dual-functional wireless networks for 6G and beyond,” IEEE J. Sel. Areas Commun., vol. 40, no. 6, pp. 1728–1767, 2022.

[5] P. Schniter and A. Sayeed, “Channel estimation and precoder design for millimeter-wave communications: The sparse way,” in Asilomar Conf. Signals Syst. Comput. (ACSSC), 2014, pp. 273–277.

[6] P. Schniter and S. Rangan, “Compressive phase retrieval via generalized approximate message passing,” IEEE Trans. Signal Process., vol. 63, no. 4, pp. 1043–1055, 2014.

[7] Z. Zhang, J. Zhang, Y. Zhang et al., “AI-based time-, frequency-, and space-domain channel extrapolation for 6G: Opportunities and challenges,” IEEE Veh. Technol. Mag., vol. 18, no. 1, pp. 29–39, 2023.

[8] Z. Chen, H. Ruan, Z. Zhang et al., “Analogical learning for crossscenario generalization: Framework and application to intelligent localization,” arXiv preprint arXiv:2504.08811, 2025.

[9] P. Dong, H. Zhang, G. Y. Li et al., “Deep CNN-based channel estimation for mmWave massive MIMO systems,” IEEE J. Sel. Top. Signal Process., vol. 13, no. 5, pp. 989–1000, 2019.

[10] L. Li, H. Chen, H.-H. Chang et al., “Deep residual learning meets OFDM channel estimation,” IEEE Wireless Commun. Lett., vol. 9, no. 5, pp. 615–618, 2019.

[11] B. Lin, F. Gao, S. Zhang et al., “Deep learning-based antenna selection and CSI extrapolation in massive MIMO systems,” IEEE Trans. Wirel. Commun., vol. 20, no. 11, pp. 7669–7681, 2021.

[12] Z. Chen, Z. Zhang, Z. Yang et al., “Channel mapping based on interleaved learning with complex-domain MLP-mixer,” IEEE Wireless Commun. Lett., vol. 13, no. 5, pp. 1369–1373, 2024.

[13] X. Zhou, L. Liang, J. Zhang et al., “Generative diffusion models for high dimensional channel estimation,” IEEE Trans. Wirel. Commun., vol. 24, no. 7, pp. 5840–5854, 2025.

[14] M. Arvinte and J. I. Tamir, “MIMO channel estimation using scorebased generative models,” IEEE Trans. Wirel. Commun., vol. 22, no. 6, pp. 3698–3713, 2022.

[15] X. Fan, X. Zhou, L. Liang et al., “Low-complexity MIMO channel estimation with latent diffusion models,” arXiv preprint arXiv:2510.21386, 2025.

[16] H. Jiang, M. Cui, D. W. K. Ng et al., “Accurate channel prediction based on transformer: Making mobility negligible,” IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2717–2732, 2022.

[17] Z. Xiao, Z. Zhang, Z. Chen et al., “Mobile MIMO channel prediction with ODE-RNN: A physics-inspired adaptive approach,” in IEEE Int. Symp. Person. Indoor Mobile Radio Commun. (PIMRC), 2022, pp. 1301–1307.

[18] Z. Xiao, Z. Zhang, C. Huang et al., “C-GRBFnet: A physics-inspired generative deep neural network for channel representation and prediction,” IEEE J. Sel. Areas Commun., vol. 40, no. 8, pp. 2282–2299, 2022.

[19] B. Chatelier, V. Corlay, M. Crussiere et al., “Model-based learning for multi-antenna multi-frequency location-to-channel mapping,” IEEE J. Sel. Top. Signal Process., vol. 19, no. 3, pp. 520–535, 2025.

[20] G. Liang, M. Yang, D. Liu et al., “Environment-aware channel inference via cross-modal flow: From multimodal sensing to wireless channels,” IEEE Trans. Mob. Comput., 2026.

[21] J. Hoydis, F. A. Aoudia, S. Cammerer et al., “Learning radio environments by differentiable ray tracing,” IEEE Trans. Mach. Learn. Commun. Netw., vol. 2, pp. 1527–1539, 2024.

[22] S. Jiang, Q. Qu, X. Pan et al., “Learnable wireless digital twins: Reconstructing electromagnetic field with neural representations,” IEEE Open J. Commun. Soc., vol. 6, pp. 1568–1590, 2025.

[23] Z. An, L. Shangguan, J. Kaewell et al., “RadioTwin: A digital building material twin for wideband, cross-link, cross-band wireless channel prediction,” in IEEE Int. Symp. Dyn. Spectr. Access Networks (DySPAN), 2025, pp. 1–10.

[24] X. Zhao, Z. An, Q. Pan et al., “NeRF2: Neural radio-frequency radiance fields,” in ACM Int. Conf. Mob. Comput. Netw. (MobiCom), 2023, pp. 1–15.

[25] H. Lu, C. Vattheuer, B. Mirzasoleiman et al., “NeWRF: A deep learning framework for wireless radiation field reconstruction and channel prediction,” in Int. Conf. Mach. Learn. (ICML), 2024, pp. 33 147–33 159.

[26] C. Wen, J. Tong, Y. Hu et al., “WRF-GS: Wireless radiation field reconstruction with 3D Gaussian splatting,” in IEEE Conf. Comput. Commun. (INFOCOM), 2025, pp. 1–10.

[27] Z. Chen, Z. Zhang, Z. Yang et al., “Channel deduction: A new learning framework to acquire channel from outdated samples and coarse estimate,” IEEE J. Sel. Areas Commun., vol. 43, no. 3, pp. 944–958, 2025.

[28] Z. Chen, Z. Zhang, Z. Xing et al., “Spatial channel deduction: Acquiring channel from approximate position and coarse estimate,” in IEEE Int. Symp. Person. Indoor Mobile Radio Commun. (PIMRC), 2025, pp. 1–6.

[29] L. Shi, J. Zhang, L. Yu et al., “Can wireless environment information decrease pilot overhead: A channel prediction example,” IEEE Wireless Commun. Lett., vol. 14, no. 3, pp. 861–865, 2025.

[30] Y. Cai, J. Zhang, L. Yu et al., “Digital twin channel-based CSI prediction: An environment-based subspace extraction approach for achieving low overhead and high robustness,” IEEE Trans. Veh. Technol., 2026.

[31] H. Sun, L. Zhu, and R. Zhang, “Channel gain map estimation for wireless networks based on scatterer model,” IEEE Trans. Wirel. Commun., vol. 24, no. 8, pp. 7012–7028, 2025.

[32] H. Sun, L. Zhu, J. Xu et al., “Channel gain map estimation based on 3-D virtual scatterer model,” IEEE Trans. Wirel. Commun., vol. 25, pp. 15 741–15 757, 2026.

[33] A. Alkhateeb, S. Jiang, and G. Charan, “Real-time digital twins: Vision and research directions for 6G and beyond,” IEEE Commun. Mag., vol. 61, no. 11, pp. 128–134, 2023.

[34] M. Zhu, L. Cazzella, F. Linsalata et al., “Toward real-time digital twins of EM environments: Computational benchmark for ray launching software,” IEEE Open J. Commun. Soc., vol. 5, pp. 6291–6302, 2024.

[35] L. Del Moro, F. Linsalata, M. Mizmizi et al., “Bayesian EM digital twins channel estimation,” IEEE Wireless Commun. Lett., vol. 14, no. 5, pp. 1326–1330, 2025.

[36] J. Wang, J. Zhang, Y. Zhang et al., “Radio environment knowledge pool for 6G digital twin channel,” IEEE Commun. Mag., vol. 63, no. 5, pp. 158–164, 2025.

[37] Y. Zeng, J. Chen, J. Xu et al., “A tutorial on environment-aware communications via channel knowledge map for 6G,” IEEE Commun. Surv. Tutor., vol. 26, no. 3, pp. 1478–1519, 2024.

[38] H. Asplund, J. Karlsson, F. Kronestedt et al., Advanced Antenna Systems for 5G Network Deployments: Bridging the Gap Between Theory and Practice. Academic Press, 2020.

[39] H. Lee, H. Choi, H. Kim et al., “Downlink channel reconstruction for spatial multiplexing in massive MIMO systems,” IEEE Trans. Wirel. Commun., vol. 20, no. 9, pp. 6154–6166, 2021.

[40] C. Ruah, O. Simeone, J. Hoydis et al., “Calibrating wireless ray tracing for digital twinning using local phase error estimates,” IEEE Trans. Mach. Learn. Commun. Netw., vol. 2, pp. 1193–1215, 2024.

[41] Z. Chen, Z. Zhang, C. Liu et al., “Towards wireless native big AI model: the mission and approach differ from large language model,” Sci. China Inf. Sci., vol. 68, no. 7, p. 170303, 2025.

[42] J. Hoydis, F. A¨ıt Aoudia, S. Cammerer et al., “Sionna RT: Differentiable ray tracing for radio propagation modeling,” in IEEE Globecom Workshops (GC Wkshps), 2023, pp. 317–321.

[43] X. Meng and Y. Kabashima, “Diffusion model based posterior sampling for noisy linear inverse problems,” arXiv preprint arXiv:2211.12343, 2022.

[44] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in IEEE/CVF Int. Conf. Comput. Vision (ICCV), 2023, pp. 4172–4182.

[45] A. Alkhateeb, “DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications,” arXiv preprint arXiv:1902.06435, 2019.