# A Hierarchical Method for Environmental Sounds Recognition in Complex Acoustic Scenes

Zhouwen Li1, Haijun Lin1\*, Bowen Lv1, Yupeng Chen1 1 Hunan Normal University School of Engineering and Design Changsha, China

Abstract：Environmental noise identification is essential for noise control, yet conventional single-stage classifiers suffer from class confusion under imbalanced multiclass conditions. This paper proposes a lightweight hierarchical model integrating a Frequency-Time Adaptive Network (FTANet) with a two-stage classification mechanism. FTANet employs parallel asymmetric one-dimensional convolutions to capture temporal dynamics and frequencyband distributions, while adaptive fusion attention (AFA) enhances discriminative acoustic features. In the first stage, FTANet classifies sounds into five coarse-grained categories covering 25 fine-grained classes. In the second stage, samples are routed to the corresponding category-specific classifiers for fine-grained recognition. On a selfconstructed dataset, FTANet achieves 95.45% coarse-grained accuracy with only 0.2004M parameters. On UrbanSound8K, it achieves 79.24% mean accuracy and 80.37% mean F1-score in ten-fold evaluation, outperforming compared convolutional models. Field tests on the Tianjin Online Acoustic Environment Monitoring Platform yield 87.94% coarse-grained accuracy and 97.21% conditional fine-grained accuracy for natural sounds correctly routed to C2. The complete hierarchical model achieves 87.18% overall accuracy and 87.40% weighted-average F1-score, demonstrating its feasibility for urban environmental noise monitoring.

Keywords: Environmental sound classification; hierarchical classification; frequency-time adaptive network, lightweight network;

## 1. Introduction

With the development of urban transportation, industry, and social activities, the urban sounds are becoming increasingly complex. Different sounds overlap in time, space, and frequency, which has higher requirements for urban sound environment monitoring, public noise control, ecological environment assessment, etc. Conventional acoustic monitoring mainly relies on sound pressure level indicators to characterize environmental noise. Carratù et al. [1] developed a sound level meter capable of automatically estimating measurement uncertainty, while Zagubień and Wolniewicz [2] investigated environmental noise measurement under different microphone positioning conditions. Although these studies have contributed to the reliability of acoustic measurements, sound pressure level indicators alone cannot identify the specific sources of environmental noise. Automatic sound-source identification can therefore complement conventional acoustic measurements by providing information for noise-source analysis, targeted monitoring, and urban noise management.

Environmental sound classification (ESC) has become an important field in acoustic research [3]. and has been widely applied in urban acoustic environment monitoring [4-6], public safety early warning [7], smart-city operations [8], ecological and environmental assessment [9], and surveillance systems [10-12]. Milivojčević et al. [13] investigated vehicle fuel-type classification using psychoacoustic features, while Lv et al. [14] developed a sound recognition method based on spectrogram features and a fusion classification model for animal acoustic monitoring. Wang and Gao [15] employed acoustic-pressure information fusion and a one-dimensional convolutional neural network for pipeline leak detection. These studies demonstrate the value of acoustic feature extraction and intelligent classification in measurement applications. However, environmental sounds are typically nonstationary [16] and frequently affected by background noise and overlapping sources, making reliable multiclass recognition challenging.

Many existing ESC methods employ two-dimensional convolutional neural networks (2D CNNs) to extract features from audio spectrograms [17–20]. However, the time and frequency axes of spectrograms represent different acoustic characteristics: the time axis describes the temporal evolution of sound energy, whereas the frequency axis reflects its spectral distribution [21]. Conventional square convolution kernels jointly process these two dimensions without explicitly distinguishing their characteristics, potentially introducing redundant features. Recent studies in Measurement have explored more effective acoustic feature modeling. Li et al. [22] employed time-frequency analysis and an attention mechanism for train noise prediction, while Li et al. [23] combined time-frequency convolution and self-attention for nonstationary mechanical sound-source separation. Liu et al. [24] integrated spectral subtraction with a CNN-TCN model for acoustic fault diagnosis under noisy conditions. Although these approaches demonstrate the importance of time-frequency feature extraction, developing compact models that efficiently capture discriminative temporal and spectral information remains an important issue for environmental sound recognition. Moreover, deeper convolutional architectures may increase computational costs and restrict practical deployment [25].

Another challenge arises from the multiclass structure of environmental sounds. Existing ESC systems commonly employ single-stage classifiers that directly distinguish all fine-grained sound categories. When the number of categories increases, acoustic similarity and class imbalance can lead to classification confusion, particularly among underrepresented classes. Although hierarchical classification has previously been investigated for environmental noise recognition [26], the integration of lightweight time-frequency feature modeling with hierarchical multiclass recognition remains insufficiently explored, particularly for practical urban acoustic monitoring applications.

To solve these problems of time-frequency feature modeling and multiclass classification, this paper proposes a lightweight hierarchical environmental sound recognition method with complex acoustic scenes. The main contributions are written as follows:

1) A lightweight Frequency-Time Adaptive Network (FTANet) is designed. The asymmetric one-dimensional convolutions are employed to extract temporal dynamics and frequency-band distribution features separately, and the adaptive fusion attention mechanism is used to recalibrate the features, while reducing the number of model parameters and enhancing the feature expression ability under complex background noise.

2) A hierarchical classification method is constructed. On the basis of the first level classification results, the overall 25-class classification task is decomposed into several smaller within-category classification tasks, thereby alleviating inter-category confusion under imbalanced data conditions.

## 2. Environmental Sound Recognition Method Based on FTANet and Hierarchical Strategy

Environmental sounds are diverse in types , and it is difficult to require the sounds samples. In addition to, the recognition accuracy of the conventional flat and single-stage classifiers is generally low. This paper proposes an environmental sound recognition model based on FTANet and a two-stage hierarchical classification architecture, which significantly improves the recognition accuracy.

## 2.1. Network Architecture

The architecture of this proposed model for complex environmental sound recognition is shown in Fig. 1. The model framework consists primarily of three components: the input preprocessor, the FTANet based feature model, and the two-stage hierarchical classification. After adaptive framing and Mel/FBank feature extraction of sound signals, a three channel time-frequency representation is formed and input into the FTANet backbone network. Through timefrequency feature extraction, adaptive fusion attention, residual learning, and multi-stage feature modeling, a shared acoustic feature representation is obtained. Based on the first-stage classification result, the corresponding secondstage sub-classifier is then selected to perform within-category fine-grained sound recognition.

FTANet is the core network. Its time-frequency convolution module separately extracts dynamic variations along the time dimension and energy distributions along the frequency dimension. The adaptive fusion attention module further performs channel-wise recalibration of the fused features, thereby enhancing effective features associated with the target sound. On this basis, the two-stage hierarchical classification method decomposes the global multiclass recognition task into coarse-grained category discrimination and within-category fine-grained recognition, which reduces the decision space of the classifiers and minimize the feature interferences between different acoustic categories.

To illustrate the characteristic differences of environmental sounds in time and frequency dimensions, six representative environmental sounds are selected from five coarse-grained categories for analysis, including the bird sound and the frog sound from natural sounds, the blower noise from industrial sounds, the broadcast sound from social activity noises, the automobile engine sound from transportation noises, and the explosion sound from construction noises. By comparing their waveforms, short-time energies, time-frequency spectrograms, and variations in spectral features, it is provided a basis for the architectural design of the Frequency-Time Adaptive Network.

![](images/6523b1ccee01a19f4a879c08f01f94541e3b6a78e8a6eac9d0c3ab3cb11021f4.jpg)  
Time (s)

![](images/17b41dd70850218eb1318bf7aa13a70246db8398db6b6839d40e4f29abb18ecb.jpg)  
Fig. 1. Architecture of this proposed method for feature-modeling and hierarchical classification

## 2.2. Frequency-Time Adaptive Network (FTANet)

Environmental sounds have different time-frequency characteristics.The time-domain and frequency-domain acoustic characteristics of bird calls, vechicle engine noise, blasting noise, blower noises, broadcast sounds, and frog calls are shown in Fig.2-3, respectively. As shown in Fig. 2, The "blasting noise" is manifested as a typical and localized discontinuous pulse peak in the short-term energy curve, with its short-term energy rapidly increasing and decreasing within a local time range, but the blower noise presents a relatively stable energy envelope in short-term energ

![](images/5d076122656756fa99e756b97cc64fad1888c373bd0d6b4edc69520fc573a31b.jpg)

![](images/70b3a708294ce3589fd4007634be5ac71bc80f755b91bf8d3d8eae009b1bafba.jpg)

![](images/e4872dd4d8cb2c54f85a409571ea030915e23271c2f9d736986089e02b37b9fd.jpg)

![](images/2fba1e1f4a4c0a07ccdf5e73eee0b7c59ca314788e0e7de3b61472093398ec35.jpg)

![](images/15fdede8e5d4f4dc4b05f90b7ffeae288f1c961e1cea38618a05a958079b41c8.jpg)  
(a) Time-domain plots of the original waveforms  
(b) Short-time energy curves  
Fig. 2. Time-domain analysis of the six representative sounds

The differences in frequency dimension between different sounds are shown in Fig.3. The time-frequency spectra of "bird calls" and "frog calls" both show obvious transverse narrowband harmonic spectral lines, with their spectral centroids undergoing high-frequency jumps over time, while the spectral roll off points of " vechicle engine noise" and "blower noise" remain in the low-frequency range for a long time, exhibiting wideband steady-state focusing.

![](images/197390774347013a80aa7d6ad8ba42a90ebe860d4bcbafb832cd712590f87a9c.jpg)  
(a) Time-frequency spectrograms

![](images/1edae51ed3d13d2078109876ea3bd311978d85fce2e9d04792bc1d1748d57b31.jpg)  
(b) Temporal trajectories of spectral features  
Fig. 3. Frequency-domain analysis of the six representative sounds.

The above analysis indicates that there are significant directional differences in the characteristics of environmental sound on the time and frequency axes: the time dimension mainly describes the dynamic evolution process of acoustic events, while the frequency dimension characterizes the energy distribution on different frequency bands. The conventional square two-dimensional convolutions use the same local receptive field to jointly model both dimensions. Although they can extract the local time-frequency features, they have difficulty independently modeling temporal dynamics and frequency-band structures according to their different physical meanings. To solve this limitation, a time-frequency convolution module (FTConv) in the front-end of FTANet is designed, which uses two parallel asymmetric one-dimensional convolution branches to extract features in the frequency and time directions, respectively. Then, an adaptive fusion attention (AFA) module is introduced to recalibrate the fused time-frequency features. The FTANet architecture is shown in Fig. 1, and the FTConv and AFA modules are the key components of FTANet.

## 2.2.1. Frequency-Time Convolution Module (FTConv)

The FTConv module is used to extract the time-domain and the frequency-domain features from environmental acoustic signals. Its structure is shown in Fig. 4.

![](images/2882843a8e01914af3dec9932927cae405d469536f6fdc6902ade62fde06812d.jpg)  
Fig. 4. Structure of the FTConv module.

As shown in Fig. 4, the forward outputs of the FTConv module are given by:

$$
\begin{array} { r } { \{ \mathbf { F } _ { c , f , t } ^ { l } = \mathrm { S i L u } ( \displaystyle \sum _ { k = 1 } ^ { C _ { i n } } \sum _ { i = - 1 } ^ { 1 } \mathbf { W } _ { f r e q , c , k , i } ^ { 3 \times 1 } \cdot \mathbf { X } _ { k , f + i , t } ^ { l } ) } \\ { [ \mathbf { T } _ { c , f , t } ^ { l } = \mathrm { S i L u } ( \displaystyle \sum _ { k = 1 } ^ { C _ { i n } } \displaystyle \sum _ { j = - 1 } ^ { 1 } \mathbf { W } _ { t i m e , c , k , j } ^ { 1 \times 3 } \cdot \mathbf { X } _ { k , f , t + j } ^ { l } )  } \end{array}\tag{1}
$$

where $\mathbf { X } ^ { l }$ denotes the input feature of the l-th layer. The frequency-branch kernel $\mathbf { W } _ { f r e q } { \in } \mathbb { R } ^ { C o u t \times C i n \times 3 \times 1 }$ and the timebranch convolution kernel $\mathbf { W } _ { t i m e } { \in } \mathbb { R } ^ { C o u t \times C i n \times 1 \times 3 }$ are used in the frequency and time branches, respectively. The intermediate joint representation obtained through concatenation is given by $\mathbf { U } ^ { l } = [ \mathbf { F } ^ { l } ; \mathbf { T } ^ { l } ] \in \mathbb { R } ^ { 2 C o u t \times F \times T } ,$ . It is passed through ${ \mathrm { ~ a ~ l ~ } } \times 1$ convolution to perform cross-channel linear interactions, followed by batch normalization and SiLU activation to achieve feature fusion and scale alignment:

$$
{ \bf O } ^ { l } = S i L U \left( \frac { \hat { \bf U } ^ { l } - \mathrm { E } [ \hat { \bf U } ^ { l } ] } { \sqrt { V a r [ \hat { \bf U } ^ { l } ] + \epsilon } } \cdot \gamma + \beta \right)\tag{2}
$$

where $\hat { \mathbf { U } } ^ { l } { = } \mathbf { W } _ { \mathrm { f u s i o n } } ^ { 1 \times 1 } { \ast } \mathbf { U } ^ { l }$ denotes the feature map obtained by applying the cross-channel interaction convolution kernel W $\begin{array} { l } { 1 \times 1 } \\ { \mathrm { f u s i o n } } \end{array}$ to $\mathrm { E } [ \hat { \mathbf { U } } ^ { l } ]$ and $\mathrm { V a r } [ \hat { \mathbf { U } } ^ { l } ]$ denote the mean and variance, respectively, of this feature map over the current mini-batch. The parameters $\gamma$ and $\beta$ are the learnable scale and shift parameters of the batch normalization layer, respectively. By using this method above, the independently extracted time features and frequency features are mapped into an unified feature space, as the input for the subsequent adaptive fusion attention module.

## 2.2.2. Adaptive Fusion Attention Mechanism

After dual-branch feature extraction by FTConv and feature fusion through the $1 \times 1$ convolution, the time and frequency features directions undergo preliminary cross-channel interaction. However, different channels have different contributions to the discrimination of the sound categories. If the same weight is assigned to all channels, it may result in more redundant responses. Therefore, an AFA module is added after feature fusion. Adaptively learn the importance of each channel based on input features, and recalibrate the fused time-frequency features. The structure of AFA is shown in Fig. 5.

![](images/e0038d85f3b67468c8072a52fb550d5fcd393f22411878bbd244620a1d19f705.jpg)  
Fig. 5. Structure of the AFA module.

Let $\mathbf { O } ^ { l } { \in } \mathbb { R } ^ { C \times F \times T }$ denote the fused feature obtained from FTConv. In AFA, the global average pooling is first performed in both frequency and time dimensions, and then the two-dimensional time-frequency response is compressed into a one-dimensional vector Z∈ℝ<sup>c</sup>. The output of the c-th channel is given by:

$$
\mathbf { Z } _ { c } = \frac { 1 } { F \times T } { \sum _ { f = 1 } ^ { F } } { \sum _ { t = 1 } ^ { T } } \mathbf { O } _ { c , f , t } ^ { l }\tag{3}
$$

where $F$ and T denote the frequency and time dimensions of the feature map, respectively. By using this method, the global time-frequency response of each channel is aggregated into a scalar to obtain channel information that contains the global signal.

Subsequently, AFA employs a two-layer fully connected structure to model nonlinear dependencies among channels. Firstly, the reduction layer $\mathbf { W } _ { 1 }$ compresses the channel dimension according to the reduction ratio $r ,$ producing a lowdimensional bottleneck representation. After SiLU nonlinear activation, the restoration layer $\mathbf { W } _ { 2 }$ maps the representation back to the original channel dimension. Finally, a normalized weight for each channel is generated by using the Sigmoid function :

$$
\mathbf { S } = \mathcal { \delta } \big ( \mathbf { W } _ { 2 } \cdot \mathrm { S i L U } ( \mathbf { W } _ { 1 } \cdot \mathbf { z } ) \big )\tag{4}
$$

where $\mathbf { W } _ { 1 }$ and $\mathbf { W } _ { 2 }$ denote the learnable parameters used for channel reduction and restoration, respectively, and $\delta ( \cdot )$ is the Sigmoid activation function. By using the bottleneck structure, the nonlinear relationships between different channels while reducing parameter overhead, and constrains the weights of each channel within the range of [0,1].

After obtaining the weight vector $\mathbf { s } ,$ it is applied back to the original fused feature $\hat { \mathbf { U } } ^ { l }$ to obtain the recalibrated feature $\mathbf { P } _ { } ^ { l } ,$ i.e.,

![](images/c59461136c11e94731791ba770f2c983b20f12c6a62428ad0939a322256c2e3f.jpg)

$$
\mathbf { P } _ { c , f , t } ^ { l } = \mathbf { S } _ { c } \cdot \hat { \mathbf { U } } _ { c , f , t } ^ { l } , \quad \forall c \in \{ 1 , 2 , . . . , 2 C _ { o u t } \}\tag{5}
$$

Dynamically adjust the response intensity of different channels based on input samples, so that the network pays more attention to channels with higher discriminative power for the current sound category, while reducing relatively redundant feature responses.

## 2.2.3. Multi-Stage Feature Modeling and Classification Output

After recalibrating the channel characteristics by using AFA, FTANet models the features through residual learning and max pooling. A progressive three-stage structure is employed to gradually extract time-frequency representations at different scales. Each stage consists of FTConv, AFA, a residual module, and max pooling. The output of the preceding stage serves as the input to the subsequent stage, allowing the features to acquire higher-level acoustic representations while their time-frequency resolution is progressively reduced. After three stages of feature modeling, the final features are sequentially passed through global average pooling, Dropout, and a fully connected layer to obtain the class scores. The multi-stage feature modeling and classification output of FTANet is shown in Fig. 6.

![](images/1f2585473c40ada499d06c995c5903db0a03d9b111549a2d87af7c70b6c2766c.jpg)  
Fig. 6. Structure of multi-stage feature modeling and classification output in FTANet

Taking the AFA output $\mathbf { P } ^ { l }$ as its input, the two $3 \ \times \ 3$ convolutions are firstly used to learn an incremental representation of the input features in the residual module. This representation is then added to the original input through an identity mapping. The output can be expressed as:

$$
\mathbf { R } ^ { l } = S i L U ( \mathbf { P } ^ { l } + D _ { r e s } ^ { l } ( \mathbf { P } ^ { l } ) )\tag{6}
$$

where $D _ { r e s } ^ { l }$ denotes the residual mapping in the l-th stage, which is composed of two $3 \times 3$ convolutional layers together with batch normalization and nonlinear activation, and R<sup>l</sup> denotes the output of the residual module. While further extracting local time-frequency patterns, this structure preserves the original information in the AFA-recalibrated features through the identity connection. This helps reduce information loss during multi-stage feature transformations and improves network optimization.

A 2 × 2 max-pooling is subsequently applied to down sample the residual features along both the frequency and time dimensions:

$$
\mathbf { H } ^ { l } = \mathbf { M a x P o o l } _ { 2 \times 2 } ( \mathbf { R } ^ { l } )\tag{7}
$$

In Eq. (7), the local responses are preserved while reducing the feature-map resolution and computational complexity in subsequent stages. It also allows subsequent convolutions to cover a larger effective time-frequency range.

Based on the above structure, FTANet employs three stages of feature modeling. The three stages use the same “FTConv-AFA-Residual-MaxPool” processing pattern but with different resolutions and numbers of channels. In the stage 1, a three-channel time-frequency feature map is taken as inputs. The time and frequency branches of FTConv each generates 16 channels, which are concatenated to form a 32-channel feature map. After AFA, residual learning, and $2 \times 2$ max pooling, the first-stage output is obtained as H<sup>1</sup>∈ℝ<sup>32×F/2×T/2</sup>. The stage 2 takes H<sup>1</sup> as input and increases the number of output channels of each time-frequency branch to 32. The outputs are concatenated to form a 64-channel feature map. The same feature recalibration, residual learning, and down sampling are then applied to obtain $\mathbf { H } ^ { 2 } { \in } \mathbb { R } ^ { 6 4 \times F / 4 \times T / 4 }$ , thereby enhancing feature representation capability. In the stage 3, H<sup>2</sup> is further used as input and timefrequency feature modeling is continued at a lower time-frequency resolution. Meanwhile, the number of output channels is maintained at 64 to avoid the additional parameter overhead caused by a further increase in network width. The final backbone output is obtained as H<sup>3</sup>∈ℝ<sup>64×F/8×T/8</sup>.

To convert the two-dimensional time-frequency features into a compact feature representation suitable for classification, the global average pooling (GAP) is first applied to aggregate $\mathbf { H } ^ { 3 }$ along the frequency and time dimensions. The pooling result for the c-th channel is expressed as:

$$
\mathbf { q } _ { c } = \frac { 1 } { F _ { 3 } \times T _ { 3 } } \sum _ { f = 1 } ^ { F _ { 3 } } \sum _ { t = 1 } ^ { T _ { 3 } } \mathbf { H } _ { c , f , t } ^ { 3 } , c = 1 , 2 , . . . , C _ { 3 }\tag{8}
$$

This produces a global channel feature map of size ${ \mathrm { C } } _ { 3 } \times 1 \times 1$ . After flattening, it forms a one-dimensional acoustic feature vector $\scriptstyle \mathbf { q } = [ \mathbf { q } _ { 1 } , \mathbf { q } _ { 2 } , \dots \mathbf { q } _ { C 3 } ] ^ { \mathbf { T } } \in \mathbb { R } ^ { C 3 }$ . During training, Dropout regularization with a drop probability of 0.5 is applied to q to reduce the risk of model overfitting. The feature vector obtained after Dropout is denoted by $\mathbf { q } ^ { l } .$ Finally, the regularized feature vector ${ \bf q } ^ { l }$ is fed into the fully connected classification layer and mapped to the target class space, producing the class-score vector $\mathbf { A } \mathbf { : }$

$$
\mathbf { A } = \mathbf { W } _ { c l s } \mathbf { q } ^ { l } + \mathbf { b } _ { c l s }\tag{9}
$$

where $\mathbf { W } _ { c l s } { \in } \mathbb { R } ^ { K \times C 3 }$ and $\mathbf { b } _ { c l s } \in \mathbb { R } ^ { K }$ denote the weight matrix and bias vector of the fully connected classification layer, respectively. K denotes the number of classes in the current classification task, and $\mathbf { A } { = } [ \mathbf { a } _ { 1 } , \mathbf { a } _ { 2 } , { \ldots } { } , \mathbf { a } _ { \mathrm { k } } ] ^ { \mathrm { T } }$ denotes the classscore vector output by the network.

## 2.3. Two-Stage Hierarchical Classification Architecture

To alleviate cross category confusion caused by directly using a single-stage classifier, , a two-stage hierarchical classification method is constructed in this paper. In this method, the FTANet's internal network architecture is not changed. Instead, it organizes multiple independent FTANet classification models based on the hierarchical relationships between environmental sound labels, and divides the classification process into two stages: the coarsegrained category discrimination and the within-category fine-grained classification. The two-stage hierarchical classification architecture is shown in Fig. 1(the right component of Fig.1)

Assuming the set of coarse-grained categories is $\mathbf { M } { = } \{ m _ { 1 } , m _ { 2 , . . . , m _ { 5 } } \}$ , i.e., the industrial sounds, the construction sounds, the transportation sounds, the social activity sounds, and the natural sounds. Each coarse-grained category m<sub>k</sub> has a fine-grained label subset $Y _ { k } { \subset } Y _ { a l l } .$ . These subsets satisfy the mutual-exclusivity and completeness constraints, i.e.,

$$
\left\{ \bigcup _ { k = 1 \atop Y _ { i } \cap Y _ { j } } ^ { 5 } Y _ { k } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { 2 5 } \} \right.\tag{10}
$$

In the first stage, the FTANet classification model is trained, which consists of five coarse-grained categories. Let this model be denoted by $V ^ { ( 1 ) } { ( : \pmb \Theta ^ { ( 1 ) } ) }$ , where $\Theta ^ { ( 1 ) }$ denotes all learnable parameters of the first-stage FTANet. Assuming the input feature is $\mathbf { X } ^ { l } ,$ the first-stage model outputs a five-dimensional class-score vector, i.e.,

$$
\mathbf { A } ^ { ( 1 ) } = V ^ { ( 1 ) } ( \mathbf { X } ; \pmb { \Theta } ^ { ( 1 ) } ) = [ a _ { 1 } ^ { ( 1 ) } , a _ { 2 } ^ { ( 2 ) } , . . . , a _ { 5 } ^ { ( 5 ) } ] ^ { T } \in \mathbb { R } ^ { 5 }\tag{11}
$$

Where $\mathbf { A } ^ { ( 1 ) }$ is the class-score vector produced by the FTANet model after three stages of feature modeling, global average pooling, Dropout, and the fully connected classification layer. The softmax function is then used to obtain the probability that the input sample belongs to the five coarse-grained categories, i.e.,

$$
p _ { k } ^ { ( 1 ) } = P ( m _ { k } | \mathrm { \bf ~ X } ) = \frac { \exp ( a _ { k } ^ { ( 1 ) } ) } { \sum _ { i = 1 } ^ { 5 } \exp ( a _ { i } ^ { ( 1 ) } ) } , k = 1 , 2 , . . . , 5 .\tag{12}
$$

The coarse-grained category with the highest first-stage classification probability is then selected. Its category index is expressed as $k ^ { * } { = } _ { \mathrm { k } \in \{ 1 , 2 , \ldots , 5 \} } p _ { k } ^ { ( 1 ) }$ , where $k ^ { * }$ determines the fine-grained subclassifier invoked in the second stage. During inference, only the second-stage FTANet corresponding to the predicted coarse-grained category $m _ { k ^ { * } }$ is invoked. Consequently, the subsequent classification is restricted to the corresponding within-category fine-grained label space.

Assuming the five second-stage sub-classifiers are denoted by $V _ { 1 } ^ { ( 2 ) } , V _ { 2 } ^ { ( 2 ) } { , . . . , V _ { 5 } ^ { ( 2 ) } }$ . The k-th second-stage classifier is trained only by the samples belonging to the coarse-grained category $m _ { k } ,$ and its classification label space is $Y _ { k } .$ . Once $k ^ { * }$ has been predicted by the first stage, the original preprocessed input feature X is transmitted to the corresponding second-stage FTANet, i.e.,

$$
\mathbf { A } _ { k ^ { * } } ^ { ( 2 ) } = V _ { k ^ { * } } ^ { ( 2 ) } ( \mathbf { X } ; \mathbf { \Theta } _ { k ^ { * } } ^ { ( 2 ) } ) , \mathbf { A } _ { k ^ { * } } ^ { ( 2 ) } \in \mathbb { R } ^ { K _ { K ^ { * } } }\tag{13}
$$

where $\{ \pmb { \Theta } _ { \mathbf { k } ^ { * } } ^ { ( 2 ) }$ denotes the learnable parameters of the $k ^ { * } { \mathrel { - } } \mathrm { t h }$ second-stage FTANet, and $K _ { k ^ { * } } { = } | Y _ { k ^ { * } } |$ is the number of finegrained classes contained in the selected coarse-grained category. For any $y \in Y _ { k ^ { * } }$ , the second-stage classification probability is expressed as:

$$
p _ { y } ^ { ( 2 ) } = P ( y | \mathbf { X } , k ^ { * } ) = \frac { \exp ( a _ { k ^ { * } , y } ^ { ( 2 ) } ) } { \displaystyle \sum _ { j = { Y _ { k ^ { * } } } } ^ { 5 } \exp ( a _ { k ^ { * } , j } ^ { ( 2 ) } ) } .\tag{14}
$$

The final fine-grained classification result is determined by the class with the highest probability from the selected second-stage classifier: $\hat { y } { = } _ { \mathrm { y } } ^ { \mathrm { a r g m a x } } \hat { p } _ { y } ^ { ( 2 ) }$

Through the above hierarchical classification, the original global fine-grained classification task is decomposed into "first level coarse-grained category discrimination+second level intra class fine-grained recognition". The first-stage FTANet establishes decision boundaries among sound-source categories, whereas the second-stage FTANet further performs fine-grained discrimination among sounds that are easily confused within the same coarse-grained category, which reduces the category search space. Meanwhile, during inference, only the first-stage model and one secondstage model corresponding to the first-stage prediction are invoked, rather than all second-stage sub-classifiers working simultaneously.

## 3. Experimental Studies

## 3.1. Data Preprocessing and Training Strategy

The experiments are tested by using a self-constructed dataset of complex environmental sounds and a public dataset. The coarse-grained categories of the self-constructed dataset are defined with reference to the classification of noise sources in the Law of the People’s Republic of China on Noise Pollution Prevention and Control, including the industrial noises, the construction noises, the transportation noises, and the social activity noises. Considering that real urban acoustic environments also contain a large number of natural background sounds, a natural sound category is added. A hierarchical classification system is constructed, whichconsists of five coarse-grained categories and their corresponding 25 fine-grained sound classes, and they are shown in Table 1.

Table 1. Label taxonomy of the self-constructed dataset.
<table><tr><td>Category ID</td><td>Noise Category</td><td>Subclass ID</td><td>Sound Class</td></tr><tr><td>C1</td><td>Industrial Noise</td><td>S01</td><td>Machining Noise</td></tr><tr><td>C1</td><td>Industrial Noise</td><td>S02</td><td>Rotating Machinery Noise</td></tr><tr><td>C1</td><td>Industrial Noise</td><td>S03</td><td>Air Compressor Noise</td></tr><tr><td>C1</td><td>Industrial Noise</td><td>S04</td><td>Ventilation Fan Noise</td></tr><tr><td>C1</td><td>Industrial Noise</td><td>S05</td><td>Blower Noise</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S06</td><td>Insect Sounds</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S07</td><td>Frog Calls</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S08</td><td>Cicada Calls</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S09</td><td>Rain</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S10</td><td>Thunder</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S11</td><td>Wind</td></tr><tr><td>C2</td><td>Natural Sounds</td><td>S12</td><td>Bird Calls</td></tr><tr><td>C3</td><td>Social Noise</td><td>S13</td><td>Conversation</td></tr><tr><td>C3</td><td>Social Noise</td><td>S14</td><td>Broadcast Sound</td></tr><tr><td>C3</td><td>Social Noise</td><td>S15</td><td>Amplified Music/Sound</td></tr><tr><td>C4</td><td>Construction Noise</td><td>S16</td><td>Material-Handling Machinery Noise</td></tr><tr><td>C4</td><td>Construction Noise</td><td>S17</td><td>Mechanical Impact Noise</td></tr><tr><td>C4</td><td>Construction Noise</td><td>S18</td><td>Blasting Noise</td></tr><tr><td>C4</td><td>Construction Noise</td><td>S19</td><td>Electric Hammer/Pile Driver Noise</td></tr><tr><td>C4</td><td>Construction Noise</td><td>S20</td><td>Crane Operation Noise</td></tr><tr><td>C5 C5</td><td>Traffic Noise</td><td>S21</td><td>Vehicle Braking Noise</td></tr><tr><td>C5</td><td>Traffic Noise</td><td>S22</td><td>Vehicle Engine Noise</td></tr><tr><td>C5</td><td>Traffic Noise</td><td>S23</td><td>Vehicle Horn</td></tr><tr><td>C5</td><td>Traffic Noise</td><td>S24</td><td>Train Rumble</td></tr><tr><td></td><td>Traffic Noise</td><td>S25</td><td>Aircraft Roar</td></tr></table>

The distribution of 25 categories sounds in Fig. 7.

![](images/b5ecd51e7b322aa48bc80b2300d49e5871ef3c76964f4a6d2b0af9e872fead00.jpg)  
Fig. 7. Distribution of 25 categories sounds.

As shown in Fig. 7, the self-constructed dataset is relatively imbalance, i.e., some high-frequency environmental sounds have a larger number of samples, while some occasional or easily affected sound samples by background noise have a relatively smaller number. . The self-constructed dataset contains a total of 9,029 sounds samples. In the experiments, the samples in each fine-grained class are randomly divided into training sets, validation sets, and testing sets at a ratio of 6:2:2.

Environmental sound signal preprocessing primarily consists of signal resampling, Fbank feature extraction, and feature construction. Firstly, all audio recordings are converted to mono and resampled to f<sub>s</sub>=16000 Hz to unify the sampling rate differences generated by different acquisition devices. Audio recordings with sampling rates different from 16 kHz were resampled using the sinc-interpolation resampling method implemented in Torchaudio, which incorporates anti-aliasing filtering. Mel filter-bank (Fbank) features are then extracted. Compared with Mel-frequency cepstral coefficients (MFCCs), the Fbank features preserve local correlations between frequency bands. The number of Mel filters is set to M=80, and the analysis window length is set to 25 ms, corresponding to $\mathrm { N _ { L } } = [ 0 . 0 2 5 \times \mathrm { f _ { s } } ] = 4 0 0$ samples per frame.

To ensure that sound signals of different lengths have a unified time dimension before being input into the network, the frame shift $N _ { S }$ is adaptively determined according to the target number of frames T and the fixed window length $N _ { L } , \mathrm { i . e . }$

$$
N _ { s } = \Phi ( L , N _ { L } , T ) = \operatorname* { m a x } \left( \left\lfloor \frac { L - N _ { L } } { T - 1 } \right\rfloor , 1 \right)\tag{15}
$$

where L is the number of sound samples. Based on $N _ { L }$ and $N _ { S } ,$ the two-dimensional Fbank time-frequency characteristics of the sound signals are obtained by using the Short Time Fourier Transform (STFT) and Mel filter bank. On this basis, a three-channel input feature is further constructed to enhance the representation of the dynamic variations of sounds. Assuming the two-dimensional Fbank feature is denoted by $\mathbf { S } { \in } \mathbb { R } ^ { \mathrm { M } \times \mathrm { T } } ,$ , the first-order differences and the second-order differences are calculated, and the two-dimensional time-frequency representation is extended to a three-channel tensor by using the mapping $\mathcal { I } { : \mathbb { R } ^ { \mathrm { M } \times \mathrm { T } } } {  } \mathbb { R } ^ { 3 \times \mathrm { M } \times \mathrm { T } }$ . The three-channel tensor is defined as:

$$
\left\{ \begin{array} { l l } { \mathbf { X } _ { 1 , f , t } = \mathbf { S } _ { f , t } } \\ { \mathbf { X } _ { 2 , f , t } = \nabla _ { t } \mathbf { S } _ { f , t } = \mathbf { S } _ { f , t } - \mathbf { S } _ { f , t - 1 } } \\ { \mathbf { X } _ { 3 , f , t } = \nabla _ { f } \mathbf { S } _ { f , t } = \mathbf { S } _ { f , t } - \mathbf { S } _ { f - 1 , t } } \end{array} \right.\tag{16}
$$

The final input feature tensor is denoted by $\mathbf { X } ^ { l } \in \mathbb { R } ^ { 3 \times \mathrm { M } \times \mathrm { T } } ,$ where the three channels contain the static Fbank feature, the time-direction first-order difference, and the frequency-direction first-order difference, respectively. The tensor serves as the standard input to FTANet.

A stage-wise independent training strategy is used, and the first-stage coarse classifier and the second-stage subclassifiers are separately trained. The first level classifier is used for discriminating 5 coarse-grained categories, while the second level sub classifier completes fine-grained sound classification. The Adam optimizer is used for all models, the initial learning rate is $5 \times 1 0 ^ { - 4 }$ , the weight decay is $1 \times 1 0 ^ { - 4 }$ , and the batch size is 32. The each model is trained for 15 epochs. All experiments are conducted by using the PyTorch framework on a single NVIDIA GeForce RTX 4090 GPU.

## 3.2. Performance Comparison with Conventional Models

To evaluate FTANet's recognition performance and model complexity for environmental sound classification, several typical networks are selected as comparison models: the conventional heavyweight convolutional networks ESResNet [27] and VGG16; the Transformer-based Audio Spectrogram Transformer (AST) [28]; and the lightweight networks 3 × 3 CNN, SSRP-CNN [29], and ESC-NAS [30]. The experiments are conducted on the self-constructed environmental sound dataset and the public UrbanSound8K dataset [31]. The results are written in Tables 2 and 3.

To evaluate the classification performances, the accuracy, the precision, the recall, and F1-score are adopted, i.e.,

$$
\left\{ \begin{array} { l l } { P r e c i s i o n _ { i } { = } \displaystyle { \frac { T P _ { i } } { T P _ { i } + F P _ { i } } } } \\ { R e c a l l _ { i } { = } \displaystyle { \frac { T P _ { i } } { T P _ { i } + F N _ { i } } } } \\ { F 1 _ { i } { = } \displaystyle { \frac { 2 P r e c i s i o n _ { i } \times R e c a l l _ { i } } { P r e c i s i o n _ { i } + R e c a l l _ { i } } } } \end{array} \right.\tag{17}
$$

where $T P _ { i }$ , $F P _ { i } ,$ , and $F N _ { i }$ denote the numbers of the true-positive samples, the false-positive samples, and the falsenegative samples for the �-th category, respectively. The overall accuracy is calculated as

$$
A c c u r a c y = \frac { \sum _ { i = 1 } ^ { C } T P _ { i } } { N }\tag{18}
$$

where C is the number of categories and �is the total number of test samples. To evaluate the performance across all categories, the macro-averaged F1-score is calculated as

$$
M a c r o - F 1 = \frac { 1 } { C } \sum _ { i = 1 } ^ { C } F 1 _ { i }\tag{19}
$$

Table 2. Performance comparison on the self-constructed dataset with five coarse-grained categories.
<table><tr><td>Method</td><td>Category ID</td><td>Precision</td><td>Recall</td><td>F1-score</td><td>Overall Accuracy</td><td>Macro-F1</td><td>#Params(M)</td><td>Flops(G)</td></tr><tr><td rowspan="5">FTANet</td><td>C1 C2</td><td>0.9604 0.9539</td><td>0.9209 1.0000</td><td>0.9402 0.9764</td><td rowspan="5">0.9545</td><td rowspan="5">0.9523</td><td rowspan="5">0.2004</td><td rowspan="5">0.3011</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>C3</td><td>0.9498</td><td>0.9342</td><td>0.9419</td></tr><tr><td>C4</td><td>0.9507</td><td>0.9578</td><td>0.9543</td></tr><tr><td>C5</td><td>0.9574</td><td>0.9397</td><td>0.9485</td></tr><tr><td rowspan="5">CNN</td><td>C1</td><td>0.8988</td><td>0.9343</td><td>0.9162</td><td rowspan="5">0.9300</td><td rowspan="5">0.9369</td><td rowspan="5">0.2232</td><td rowspan="5">0.3179</td></tr><tr><td>C2</td><td>0.9809</td><td>0.9879</td><td>0.9844</td></tr><tr><td>C3</td><td>0.8840</td><td>0.9805</td><td>0.9298</td></tr><tr><td>C4</td><td>0.9452</td><td>0.8772</td><td>0.9099</td></tr><tr><td>C5</td><td>0.9593</td><td>0.9298</td><td>0.9443</td></tr><tr><td rowspan="5">VGG16</td><td>C1</td><td>0.8891</td><td>0.9264</td><td>0.9074</td><td rowspan="5">0.9076</td><td rowspan="5">0.9173</td><td rowspan="5">134.2810</td><td rowspan="5">2.0771</td></tr><tr><td>C2</td><td>0.9894</td><td>0.9253</td><td>0.9563</td></tr><tr><td>C3</td><td>0.8689</td><td>0.9342</td><td>0.9003</td></tr><tr><td>C4</td><td>0.9294</td><td>0.8741</td><td>0.9009</td></tr><tr><td>C5</td><td>0.8996</td><td>0.9455</td><td>0.9219</td></tr><tr><td rowspan="5">SSRP-CNN</td><td>C1</td><td>0.8818</td><td>0.9209</td><td>0.9009</td><td rowspan="5">0.9470</td><td rowspan="5">0.9436</td><td rowspan="5">0.3899</td><td rowspan="5">0.1421</td></tr><tr><td>C2 C3</td><td>0.9912 0.8996</td><td>0.9956</td><td>0.9934</td></tr><tr><td>C4</td><td></td><td>0.9959</td><td>0.9453</td></tr><tr><td></td><td>0.9381</td><td>0.9032</td><td>0.9204</td></tr><tr><td>C5</td><td>0.9901</td><td>0.9281</td><td>0.9581</td></tr><tr><td rowspan="5">ESResNet</td><td>C1</td><td>0.8237</td><td>0.9462</td><td>0.8807</td><td rowspan="5">0.9334</td><td rowspan="5">0.9306</td><td rowspan="5">11.1791</td><td rowspan="5">1.8235</td></tr><tr><td>C2 C3</td><td>0.9782 0.9192</td><td>0.9846 0.9835</td><td>0.9814</td></tr><tr><td>C4</td><td>0.9535</td><td>0.8139</td><td>0.9503</td></tr><tr><td></td><td></td><td></td><td>0.8782</td></tr><tr><td>C5</td><td>0.9716</td><td>0.9536</td><td>0.9625</td></tr><tr><td></td><td>C1</td><td>0.8854</td><td>0.9051</td><td>0.8951</td><td></td><td></td><td></td><td></td></tr><tr><td>ESC-NAS</td><td>C2 C3</td><td>0.9145 0.9426</td><td>0.8703 0.8107</td><td>0.8919 0.8717</td><td>0.8853</td><td>0.8855</td><td>0.0482</td><td>0.0139</td></tr><tr><td></td><td>C4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.9335</td><td>0.9057</td><td>0.9194</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>C5 C1</td><td>0.7967</td><td>0.9095</td><td>0.8494</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>C2</td><td>0.9614 0.9891</td><td>0.9462</td><td>0.9537</td><td></td><td></td><td></td><td></td></tr><tr><td>AST</td><td></td><td></td><td>1.0000</td><td>0.9945</td><td></td><td></td><td></td><td>51.2949</td></tr><tr><td></td><td>C3</td><td>0.9275</td><td>1.0000</td><td>0.9624</td><td>0.9648</td><td>0.9634</td><td>85.2588</td><td></td></tr><tr><td></td><td>C4</td><td>0.9569</td><td>0.9355</td><td>0.9460</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>C5</td><td>0.9715</td><td>0.9490</td><td>0.9601</td><td></td><td></td><td></td><td></td></tr></table>

As shown in Table 2, based on the self-constructed environmental sound dataset with five coarse-grained categories, FTANet the accuracy of FTANet is 95.45% and the Macro-F1 score is 95.23%, which is second only to AST. In AST, the accuracy is 96.48% and the Macro-F1 is 96.34%, which is exceeding FTANet by 1.03% and 1.11%, respectively. However, AST has 85.2588 M parameters and 51.2949 G FLOPs, whereas FTANet has only 0.2004 M parameters and 0.3011 G FLOPs, which greatly reduces the computational complexity.

The advantages of FTANet are also evident in comparisons with the convolutional and lightweight models. Compared with the 3 × 3 CNN, FTANet increases accuracy from 93.00% to 95.45%, while reducing the parameter and FLOPs by approximately 10.22% and 5.28%, respectively. FTANet also outperforms SSRP-CNN by 0.75% in accuracy while using approximately 48.60% fewer parameters, although SSRP-CNN has fewer FLOPs. Compared with the heavyweight VGG16 and ESResNet models, the FTANet improves the accuracy by 4.69% and 2.11%, respectively, while requiring substantially fewer parameters and FLOPs. Although ESC-NAS has the fewest parameters and FLOPs, its accuracy is 88.53%, which is lower than that of FTANet. In a word, the FTANet provides a favorable balance between recognition performance and model complexity, which demonstrates its advantages for deployment in resource-constrained environmental sound classification systems.

To further validate the generalization performance of FTANet on a public dataset, the experiments are tested by following the 10-fold data partition of UrbanSound8K. The experimental results are shown in Table 3.

Table 3. Experimental results by using the UrbanSound8K dataset
<table><tr><td rowspan="2"></td><td colspan="2">FTANet</td><td colspan="2">CNN</td><td colspan="2">VGG16</td><td colspan="2">SSRP-CNN</td><td colspan="2">ESResNet</td><td colspan="2">ESC-NAS</td><td colspan="2">AST</td></tr><tr><td></td><td>Accuracy F1-score Accuracy F1-score Accuracy F1-score Accuracy F1-score Accuracy F1-score Accuracy F1-score Accuracy F1-score</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fold1</td><td>0.7446</td><td>0.7686</td><td>0.7446</td><td>0.7781</td><td>0.6861</td><td>0.7269</td><td>0.7560</td><td>0.7684</td><td>0.7331</td><td>0.7574</td><td>0.6231</td><td>0.6370</td><td>0.8797</td><td>0.8974</td></tr><tr><td>Fold2</td><td>0.8176</td><td>0.8206</td><td>0.8041</td><td>0.8108</td><td>0.6993</td><td>0.7070</td><td>0.7354</td><td>0.7562</td><td>0.7162</td><td>0.7436</td><td>0.6374</td><td>0.6637</td><td>0.8986</td><td>0.9099</td></tr><tr><td>Fold3</td><td>0.7459</td><td>0.7733</td><td>0.7211</td><td>0.7329</td><td>0.6443</td><td>0.6818</td><td>0.6649</td><td>0.6985</td><td>0.6886</td><td>0.7023</td><td>0.5405</td><td>0.5608</td><td>0.7741</td><td>0.8012</td></tr><tr><td>Fold4</td><td>0.7859</td><td>0.7772</td><td>0.8192</td><td>0.8158</td><td>0.6909</td><td>0.6789</td><td>0.7798</td><td>0.7896</td><td>0.7515</td><td>0.7461</td><td>0.6606</td><td>0.6636</td><td>0.8889</td><td>0.8857</td></tr><tr><td>Fold5</td><td>0.8291</td><td>i0.8311</td><td>0.7788</td><td>0.7703</td><td>0.8024</td><td>0.8128</td><td>0.8451</td><td>0.8462</td><td>0.8462</td><td>0.8524</td><td>0.6688</td><td>0.6700</td><td>0.9327</td><td>0.9383</td></tr><tr><td>Fold6</td><td>0.8214</td><td>0.8305</td><td>0.8688</td><td>0.8802</td><td>0.7789</td><td>0.7921</td><td>0.7825</td><td>0.8006</td><td>0.8117</td><td>0.8271</td><td>0.6877</td><td>0.7050</td><td>0.8979</td><td>0.9076</td></tr><tr><td>Fold7</td><td>0.7780</td><td>0.7863</td><td>0.8103</td><td>0.8186</td><td>0.7136</td><td>0.7085</td><td>0.7613</td><td>0.7745</td><td>0.7494</td><td>0.7590</td><td>0.6038</td><td>0.6191</td><td>0.8687</td><td>0.8513</td></tr><tr><td>Fold8</td><td>0.7618</td><td>0.7891</td><td>0.6960</td><td>0.7266</td><td>0.6675</td><td>0.6756</td><td>0.6824</td><td>0.6964</td><td>0.7246</td><td>0.7378</td><td>0.6811</td><td>0.6875</td><td>0.8697</td><td>0.8817</td></tr><tr><td>Fold9</td><td>0.8174</td><td>0.8307</td><td>0.8591</td><td>0.8704</td><td>0.7733</td><td>0.7840</td><td>0.8125</td><td>0.8132</td><td>0.7904</td><td>0.8094</td><td>0.7316</td><td>0.7483</td><td>0.9020</td><td>0.9148</td></tr><tr><td>Fold10</td><td>0.8220</td><td>0.8298</td><td>0.7885</td><td>0.7963</td><td>0.7694</td><td>0.7819</td><td>0.8124</td><td>0.8252</td><td>0.8041</td><td>0.8169</td><td>0.7013</td><td>0.7014</td><td>0.9140</td><td>0.9225</td></tr><tr><td>Mean</td><td>0.7924</td><td>0.8037</td><td>0.7891</td><td>0.8000</td><td>0.7226</td><td>0.7349</td><td>0.7632</td><td>0.7769</td><td>0.7616</td><td>0.7752</td><td>0.6536</td><td>0.6656</td><td>0.8826</td><td>0.8910</td></tr><tr><td>∆Mean</td><td></td><td></td><td>0.0033</td><td>0.0037</td><td>0.0698</td><td>0.0688</td><td>0.0292</td><td>0.0268</td><td>0.0308</td><td>0.0285</td><td>0.1388</td><td>0.1381</td><td>-0.0899</td><td>-0.0888</td></tr></table>

As shown in Table 3, FTANet achieves an average accuracy of 79.24% and an average F1-score of 80.37% in the ten-fold evaluation on the UrbanSound8K dataset. As indicated by the ΔMeanFTANet results(i.e., the error between the mean of FTANet and the mean of other methods), FTANet outperforms the 3 × 3 CNN by 0.33% and 0.37% in the mean accuracy and the mean F1-score, respectively. Compared with VGG16, SSRP-CNN, ESResNet, and ESC-NAS, FTANet improves the mean accuracy by 6.98%, 2.92%, 3.08%, and 13.88%, respectively, while improving the mean F1-score by 6.88%, 2.68%, 2.85%, and 13.81%. Therefore, FTANet achieves the highest mean accuracy and F1-score among the compared convolution-based models on UrbanSound8K.

In AST, the mean accuracy and the F1-score are higher than those of FTANet by 8.99% and 8.88%, respectively. However, as shown in Table 2, AST has 85.2588 M parameters and 51.2949 G FLOPs, whereas FTANet has only 0.2004 M parameters and 0.3011 G FLOPs. Thus, FTANet provides competitive classification performance at a substantially lower model complexity. In summary, the FTANet achieves a favorable balance between classification performance, generalization capability, and computational complexity.

## 3.3. Comparison of Single-Stage and Two-Stage Classification Architectures

To verify the effectiveness of the two-stage hierarchical classification strategy for fine-grained environmental sound recognition, the FTANet is used as a common feature-extraction backbone to construct a single-stage classification model and a two-stage hierarchical classification model for comparison. The single-stage model directly maps the extracted acoustic features to all 25 fine-grained classes. By contrast, in the two-stage hierarchical classification model, a first-stage coarse classifier is firstly used for determining the broad category to which the sample belongs, and then the corresponding second-stage sub-classifier is selected to perform within-category fine-grained recognition. The experimental results are shown in Table 4.

Table 4. Experimental results of single-stage and two-stage classification on the 25-class dataset.
<table><tr><td rowspan="2">Coarse class</td><td rowspan="2">Fine class</td><td colspan="3">Single-stage</td><td colspan="3">Two-stage</td><td rowspan="2">ΔF1↑</td></tr><tr><td>Precision</td><td>Recall</td><td>F1-score</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td rowspan="5">C1</td><td>S01</td><td>0.7500</td><td>0.4762</td><td>0.5825</td><td>0.8542</td><td>0.6508</td><td>0.7387</td><td>0.1562</td></tr><tr><td>S02</td><td>0.9333</td><td>0.9722</td><td>0.9524</td><td>0.9863</td><td>1.0000</td><td>0.9931</td><td>0.0407</td></tr><tr><td>S03</td><td>0.6286</td><td>0.8627</td><td>0.7273</td><td>0.9608</td><td>0.9608</td><td>0.9608</td><td>0.2335</td></tr><tr><td>S04</td><td>0.7667</td><td>1.0000</td><td>0.8679</td><td>0.7841</td><td>1.0000</td><td>0.8790</td><td>0.0111</td></tr><tr><td>S05</td><td>0.5962</td><td>0.5082</td><td>0.5487</td><td>0.6613</td><td>0.6721</td><td>0.6667</td><td>0.1180</td></tr><tr><td rowspan="7">C2</td><td>S06</td><td>0.6667</td><td>0.8857</td><td>0.7607</td><td>0.6897</td><td>0.8571</td><td>0.7643</td><td>0.0036</td></tr><tr><td>S07</td><td>0.7292</td><td>0.5385</td><td>0.6195</td><td>0.8533</td><td>0.9846</td><td>0.9143</td><td>0.2948</td></tr><tr><td>S08</td><td>0.7586</td><td>0.7586</td><td>0.7586</td><td>0.8627</td><td>0.7586</td><td>0.8073</td><td>0.0487</td></tr><tr><td>S09</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9592</td><td>1.0000</td><td>0.9792</td><td>-0.0208</td></tr><tr><td>S10</td><td>0.9859</td><td>0.9722</td><td>0.9790</td><td>1.0000</td><td>0.9722</td><td>0.9859</td><td>0.0069</td></tr><tr><td>S11</td><td>0.9531</td><td>0.8356</td><td>0.8905</td><td>0.9467</td><td>0.9726</td><td>0.9595</td><td>0.069</td></tr><tr><td>S12</td><td>0.5641</td><td>0.3143</td><td>0.4037</td><td>0.6735</td><td>0.4714</td><td>0.5546</td><td>0.1509</td></tr><tr><td rowspan="3">C3</td><td>S13</td><td>0.9706</td><td>0.8919</td><td>0.9296</td><td>0.8795</td><td>0.9865</td><td>0.9299</td><td>0.0003</td></tr><tr><td>S14</td><td>0.8966</td><td>0.9811</td><td>0.9369</td><td>0.8455</td><td>0.9811</td><td>0.9083</td><td>-0.0286</td></tr><tr><td>S15</td><td>0.9130</td><td>0.6667</td><td>0.7706</td><td>0.9344</td><td>0.9048</td><td>0.9194</td><td>0.1488</td></tr><tr><td rowspan="5">C4</td><td>S16</td><td>0.3333</td><td>0.0238</td><td>0.0444</td><td>0.5000</td><td>0.3333</td><td>0.4000</td><td>0.3556</td></tr><tr><td>S17</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9930</td><td>1.0000</td><td>0.9965</td><td>-0.0035</td></tr><tr><td>S18</td><td>0.7234</td><td>0.4533</td><td>0.5574</td><td>0.7647</td><td>0.3467</td><td>0.4771</td><td>-0.0803</td></tr><tr><td>S19</td><td>0.6667</td><td>0.7018</td><td>0.6838</td><td>0.6049</td><td>0.8596</td><td>0.7101</td><td>0.0263</td></tr><tr><td>S20</td><td>0.6148</td><td>0.9540</td><td>0.7477</td><td>0.8058</td><td>0.9540</td><td>0.8737</td><td>0.1260</td></tr><tr><td rowspan="5">C5</td><td>S21</td><td>0.6043</td><td>0.8317</td><td>0.7000</td><td>0.6977</td><td>0.8911</td><td>0.7826</td><td>0.0826</td></tr><tr><td>S22</td><td>0.2024</td><td>0.1977</td><td>0.2000</td><td>0.5940</td><td>0.9186</td><td>0.7215</td><td>0.5215</td></tr><tr><td>S23</td><td>0.8679</td><td>0.7419</td><td>0.8000</td><td>0.9149</td><td>0.6935</td><td>0.7890</td><td>-0.0110</td></tr><tr><td>S24</td><td>1.0000</td><td>0.1410</td><td>0.2472</td><td>0.8333</td><td>0.7051</td><td>0.7639</td><td>0.5167</td></tr><tr><td>S25</td><td>0.3198</td><td>0.6058</td><td>0.4186</td><td>0.7692</td><td>0.2885</td><td>0.4196</td><td>0.0010</td></tr><tr><td colspan="2">Overall Accuracy↑</td><td></td><td>0.7132</td><td></td><td></td><td>0.8149</td><td></td><td></td></tr><tr><td colspan="2">Macro-F1↑</td><td></td><td>0.6851</td><td></td><td></td><td>0.7958</td><td></td><td></td></tr><tr><td colspan="2">Time(s)</td><td></td><td>0.2170</td><td></td><td></td><td>1.2390</td><td></td><td></td></tr></table>

As shown in Table 4, in the single-stage classification model, the overall accuracy and the Macro-F1 score is respectively 71.32% and 68.51% across the 25 fine-grained environmental sound classes. After the two-stage hierarchical classification is adopted, the overall accuracy increases to 81.49% and the Macro-F1 score increases to 79.58%, which improves 10.17% and 11.07%, respectively. At the class level, the two-stage method improves the F1- score for 20 of the 25 fine-grained classes, indicating that the performance improvement is not limited to only a few categories. These results demonstrate the effectiveness of the two-stage hierarchical classification.

In terms of inference efficiency, the single-stage model requires only one forward classification pass, resulting in an inference time of 0.2170 s. The two-stage hierarchical classification model requires the first-stage coarse classification and second-stage fine-grained classification to be performed sequentially, increasing the inference time to 1.2390 s. Thus, although two-stage hierarchical classification improves recognition accuracy, it also requires additional inference overhead, which reflects a trade-off between classification performance and computational efficiency. To further analyze the differences in discriminative performance between the two classification approaches across different finegrained classes, the confusion matrices are shown in Fig.8.

![](images/ddd65bf21cc71d7c60401ea89f6484d4079bd6726553da0a4f2fab4062678a4e.jpg)  
Predicted Label  
Fig. 8. Confusion matrices of the two methods across all fine-grained classes.

As shown in Fig. 8, the single-stage classification model exhibits relatively clear misclassification among several semantically similar classes. In the two-stage hierarchical classification model, the diagonal responses are more concentrated for most classes, and some intercategory confusion is alleviated. To further examine the feature distributions by using the two classification methods, t-distributed stochastic neighbor embedding (t-SNE) was used to perform dimensionality-reduction visualization of the high-dimensional acoustic features. The results are shown in Fig. 9.

![](images/6f9dbeb1969bbf839397405be839b57dbca0a1b7ed43e3b2c69ba5b5bf1b0215.jpg)  
Fig. 9. t-SNE comparison of test-set feature distributions.

As shown in Fig. 9, the feature distributions of some fine-grained classes in the single-stage classification model exhibit relatively clear overlap. After two-stage hierarchical classification is adopted, the feature clusters of some different classes exhibit clearer clustering patterns and interclass boundaries. These visualization results are broadly consistent with the changes observed in the confusion matrices, which indicates that the hierarchical classification strategy can alleviate feature confusion among different classes.

## 3.4. Ablation Study of FTANet

An ablation study evaluates the contribution of each component of FTANet. Method A uses FTConv without $1 \times 1$ feature fusion or AFA; Method B uses FTConv and 1 × 1 feature fusion without AFA; Method C uses FTConv and AFA without 1 × 1 feature fusion; Method D uses standard 3 × 3 convolutions and AFA without 1 × 1 feature fusion; and Method E uses standard 3 × 3 convolutions without 1 × 1 feature fusion or AFA. The complete FTANet uses all three components. All other network structures and training settings remain identical across the configurations. Table 5 presents their classification accuracy, parameter counts, and FLOPs.

Table 5. Ablation study on the five coarse-grained classes.
<table><tr><td rowspan="2">Method</td><td colspan="2">Conv</td><td rowspan="2">1×1Fusion</td><td rowspan="2">AFA</td><td rowspan="2">Accuracy↑</td><td rowspan="2">#Params(M)</td><td rowspan="2">Flops(G)</td></tr><tr><td>3×3</td><td>FTConv</td></tr><tr><td>FTANet</td><td></td><td>√</td><td>√</td><td>√</td><td>0.9545</td><td>0.2004</td><td>0.3011</td></tr><tr><td>A</td><td></td><td>√</td><td></td><td></td><td>0.9359</td><td>0.1861</td><td>0.2847</td></tr><tr><td>B</td><td></td><td>√</td><td>√</td><td></td><td>0.9294</td><td>0.1958</td><td>0.3008</td></tr><tr><td>C</td><td></td><td>√</td><td></td><td>√</td><td>0.9394</td><td>0.1907</td><td>0.2850</td></tr><tr><td>D</td><td>√</td><td></td><td></td><td>√</td><td>0.9310</td><td>0.2281</td><td>0.3182</td></tr><tr><td>E</td><td>√</td><td></td><td></td><td></td><td>0.9300</td><td>0.2232</td><td>0.3179</td></tr></table>

A comparison between Methods E and A shows that it improves the model’s accuracy from 93.00% to 93.59% by using the FTConv instead of the standard 3×3 convolution. Meanwhile, the parameter complexity decreases from 0.2232 M to 0.1861 M, which approximately reduces16.62%, and the FLOPs decrease from 0.3179 G to 0.2847 G, with a reduction of approximately 10.44%. These results demonstrate that it improves classification performance while reducing model complexity by using two asymmetric convolutional branches to separately model features in the time and frequency, thereby verifying the effectiveness of FTConv in modeling the time-frequency features of environmental sounds.

The AFA module improves the performances with different convolutional structures. Adding the AFA to the standard 3×3 convolutional baseline (Method D), the accuracy increases from 93.00% to 93.10%. When the AFA is Added to the FTConv (Method C), the accuracy increases from 93.59% to 93.94%. In both experiments, the AFA only added approximately 0.005 M additional parameters and 0.0003 G additional FLOPs. This indicates that the AFA can improve the channel feature representation capability of the network with relatively less additional computational overhead.

Further analysis of the 1×1 feature-fusion module shows that Method B ( i.e., adding feature fusion alone to the FTConv) reduces the accuracy from 93.59% to 92.94%, which indicates that the simple cross-channel linear fusion does not directly improve performance. By contrast, when the feature fusion method and the AFA are used together, the complete FTANet achieves an accuracy of 95.45%, which is higher than that of Method B by 2.51% and is higher than that of Method C by 1.51%. These results indicate that after the 1×1 convolution performs cross-channel interaction between the time and frequency branches, the AFA can further adjust the response strength of each channel according to the input features, thereby a more discriminative joint time-frequency feature representation can be obtained.

In summary, the complete FTANet has the highest accuracy of 95.45%, with a parameter count of 0.2004 M and FLOPs of 0.3011 G. Comparing with the standard 3×3 convolutional baseline, the Method E, the FTANet improves the accuracy by 2.45% while reducing the parameter count by approximately 10.22%. These results demonstrate that the combination of the FTConv, the feature fusion, and the AFA, can improve the performance of the environmental sound classification while reducing the computational complexity of the model.

## 4. Application

To verify the feasibility of the proposed hierarchical recognition method to the urban acoustic environment monitoring, the method is integrated into the Tianjin Online Acoustic Environment Monitoring Platform ( in Tianjin city, China) and tested in field. The application workflow of this method on the platform is shown in Fig. 10.

![](images/1a55e67d3ed1248e1313a49d7713c23e21cee3c59a25327e1667d9cd6b8954fb.jpg)  
Fig. 10. Application workflow of the hierarchical classification method on the monitoring platform.

The front-end acoustic sensors are deployed in the urban street environments to continuously collect environmental sounds. The collected acoustic data are transmitted to a cloud server via HTTP. The server performs sound signal analysis, online recognition, and result generation, and then the recognition results are returned to the platform’s display and early-warning modules.

Based on the existing platform classification system, the proposed hierarchical classification procedure is integrated into the online recognition pipeline. In the first-stage classification, the system discriminates the five coarse-grained categories, i.e., the industrial noise, the construction noise, the transportation noise, the social activity noise, and the natural sounds. The subsequent classification process is determined according to the first-stage prediction results. In this deployment, the system initially incorporated the second-stage sub-classifier for natural sounds to perform withincategory fine-grained recognition. For the other four coarse-grained categories, the first-stage classification results are output directly. The platform’s online recognition interface and the deployment of the field monitoring equipment are shown in Fig. 11.

![](images/0cea6432c30f04c9854d7503ebbfe6568192040251601bfc714cadc45c22f45b.jpg)  
(a) Online classification platform

![](images/be3e5bc778ee8ef10d5de6168fd257205c2ee23bf8a023c4ed890df1f953d303.jpg)  
(b) Field scene and acoustic acquisition equipment  
Fig. 11. Deployment of the online platform and field monitoring equipment.

In Fig.11(a), the platform’s online recognition interface is shown and the system can record and display recognition results such as the sound category, duration, and early-warning information. In Fig.11(b), the actual monitoring environment and the deployment of the front-end acoustic acquisition equipment are shown.

To further evaluate the practical performance of the system under complex acoustic-scene conditions, the 1000 fieldmonitoring acoustic samples are randomly selected from data collected nearly five months at the Hangu Education

Center in Binhai New Area, Tianjin, China. After removing the 212 invalid recordings that containing unclassifiable noises, the 788 valid samples are retained for quantitative evaluation.

Unlike acoustic samples in a manually processing dataset, the recording sounds in field may simultaneously contain multiple acoustic sources and therefore may be associated with more than one valid label. Accordingly, a prediction is considered correct if the predicted label matches any one of the assigned truth labels. Because the class distribution in the field-monitoring dataset is highly imbalanced, weighted-average accuracy and weighted-average F1-score are used to evaluate the overall classification performance. The weight that assigned to each class is determined by its proportion of valid samples, i.e.,

$$
w _ { i } { = } \frac { n _ { i } } { \sum _ { k = 1 } ^ { K } n _ { k } }\tag{20}
$$

where $n _ { i }$ is the number of valid samples in the i-th class and K is the number of classes that containing valid samples. The classes that marked as N/A are excluded from the calculation. The weighted-average accuracy and the F1-score are calculated as

$$
A c c u r a c y _ { w e i g h t e d } { = } \sum _ { i = 1 } ^ { K } w _ { i } \times A c c u r c y _ { i }\tag{21}
$$

$$
F 1 - s c o r e _ { w e i g h t e d } = \sum _ { i = 1 } ^ { K } w _ { i } \times F 1 - s c o r e _ { i }\tag{22}
$$

The highly imbalanced sample distribution in Table 6 reflects the actual acoustic characteristics of the deployment site. The Hangu Education Center is not located in an industrial zone; therefore, no valid C1 samples are recorded. The construction activities are also limited in the surrounding area, which results in only 18 C4 samples. During the monitoring period, no strong thunderstorm events occurred, and the site was not characterized by concentrated frog or insect activities. Wind sounds were also generally too weak to be captured as valid target events by the monitoring system. Consequently, no valid samples were obtained for S06, S07, S10, or S11. The uneven distribution therefore represents the actual composition and occurrence frequency of sound sources at the deployment site rather than an artificial imbalance introduced during dataset construction.

Table 6. Classification results on the field-monitoring dataset.
<table><tr><td rowspan="2">Coarse Class</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Fine class Samples Correct Samples Incorrect Samples</td><td rowspan="2"></td><td colspan="2">Coarse</td><td colspan="2">Fine</td><td colspan="2">Overall Weighted</td></tr><tr><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td><td>Accuracy</td><td>F1-score</td></tr><tr><td>C1</td><td>--</td><td>N/A N/A</td><td>N/A N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A N/A</td><td>N/A N/A</td><td rowspan="10"></td><td rowspan="10"></td></tr><tr><td></td><td>S06 S07</td><td>N/A</td><td>N/A</td><td>N/A N/A</td><td></td><td>N/A</td><td></td><td>N/A</td></tr><tr><td>C2</td><td>S08</td><td>263</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>209</td><td>54</td><td>0.7963</td><td>0.9713</td><td>0.9807</td><td></td></tr><tr><td></td><td>S09 S10</td><td>2 N/A</td><td>1</td><td>1</td><td></td><td>1.0000</td><td>1.0000</td><td></td></tr><tr><td></td><td>S11</td><td>N/A</td><td>N/A N/A</td><td>N/A</td><td></td><td>N/A N/A</td><td>N/A</td><td>0.8718</td></tr><tr><td></td><td></td><td>5</td><td>5</td><td>N/A</td><td></td><td></td><td></td><td>N/A</td></tr><tr><td></td><td>S12</td><td>127</td><td>109</td><td>0 18</td><td>0.8583</td><td></td><td>1.0000</td><td>1.0000 N/A</td></tr><tr><td>C3 C4</td><td>-- --</td><td>18</td><td>12</td><td>6</td><td>0.6667</td><td>0.8289 0.7742</td><td>N/A N/A</td><td>N/A</td></tr><tr><td>C5</td><td>--</td><td>373</td><td>357</td><td>16</td><td>0.9571</td><td>0.8981</td><td>N/A</td><td>N/A</td></tr><tr><td></td><td>--</td><td>788</td><td>693</td><td>95</td><td>0.8794</td><td>0.8789</td><td></td><td>0.9812</td></tr><tr><td>Weighted Average C2 end-to-end fine-grained classification</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.9721 0.7741</td><td>0.8685</td></tr></table>

At the coarse-grained classification stage, 693 of the 788 valid samples are classified correctly under the one-match criterion, resulting in an overall accuracy of 87.94% and a weighted-average F1-score of 87.89%. The classification accuracies for C2, C3, C4, and C5 are 79.63%, 85.83%, 66.67%, and 95.71%, respectively. These results demonstrate that this proposed method maintains good overall classification performance under the highly imbalanced class distribution encountered in the real-world monitoring environment.

Within C2, the 215 samples are correctly identified at the coarse-grained stage, and subsequently passed to the natural-sound fine-grained classifier. The fine-grained classifier achieved a weighted-average accuracy of 97.21% and a weighted-average F1-score of 98.12%. For S08, the accuracy and the F1-score are 97.13% and 98.07%, respectively. All routed samples belonging to S09 and S12 are correctly classified, with both classes achieving the accuracy and the F1-score of 100.00%. When both coarse-grained routing and subsequent fine-grained recognition are considered, the end-to-end accuracy and the F1-score for C2 are 77.41% and 86.85%, respectively. Across all 788 field-monitoring samples, the complete deployed classification architecture achieved an overall weighted accuracy of 87.18% and a weighted-average F1-score of 87.40%. These results verify the effectiveness of the proposed hierarchical procedure in performing both coarse-grained recognition and within-category fine-grained recognition on real-world fieldmonitoring data.

![](images/c33d192f4d31ca9f0ad4b30b9a5f4980ebc098b0ca1a7bc415c7ca1bafc74149.jpg)

![](images/2c8fafacca8d875c218700938e8cdeec0389ac94f0574e75e0aea05c06e98e90.jpg)  
Fig. 12. Row-normalized confusion matrices for field-test classification.

As shown in Fig. 12, the predictions ofboth classification stages are generally concentrated along the main diagonal, which indicates that most field samples are correctly recognized. The remaining errors are confined to a small number of coarse-grained categories and natural-sound subclasses, which demonstrates the effectiveness and practical feasibility of the proposed method for urban acoustic environment monitoring.

## 5. Conclusions

This paper focuses on the problems of time-frequency feature modeling and imbalanced multiclass classification of environmental sounds in complex acoustic scenes, and proposes an environmental noise recognition method based on the FTANet architecture and a hierarchical strategy. A lightweight Frequency-Time Adaptive Network (FTANet) was designed to model time-domain dynamics and frequency-band distributions by separately using asymmetric onedimensional convolutions. Adaptive fusion attention was further employed to perform channel-wise recalibration of the fused features, thereby enhancing the representation of discriminative acoustic information.

This two-stage hierarchical classification method is also constructed, which consists of coarse-grained category discrimination and within-category fine-grained recognition. On the self-constructed environmental sound dataset with five coarse-grained categories, the FTANet achieves an accuracy of 95.45% and a Macro-F1 score of 95.23%, with only 0.2004 M parameters and 0.3011 G FLOPs. The FTANet achieves the highest performance among the convolution-based models and ranks second only to AST. Compared with the AST, the FTANet reduces the parameter count and FLOPs by approximately 99.76% and 99.41%, respectively, which demonstrates a favorable balance between classification performance and model complexity. On UrbanSound8K, the FTANet achieves a mean accuracy of 79.24% and a mean F1-score of 80.37% across the ten folds, which outperforms all the convolution-based comparison models and validates its generalization performance across different datasets. For the 25-class environmental sound classification task, the two-stage hierarchical method improves the overall accuracy from 71.32% to 81.49% and the Macro-F1 score from 68.51% to 79.58%, with improvements of 10.17% and 11.07%, respectively. These results demonstrate that it can effectively reduce intercategory feature interferences and improve fine-grained recognition performances by decomposing the global classification task into coarse-grained category discrimination and within-category fine-grained recognition.

The proposed method is further integrated into the Tianjin Online Acoustic Environment Monitoring Platform, where the hierarchical classification procedure and the second-stage sub-classifier for natural sounds are deployed. Field testing is conducted on 788 valid acoustic samples that obtained from nearly five months of monitoring data. Despite the highly imbalanced samples, the site-specific class distribution and the multiple sound sources, the overall weighted accuracy of this method is 87.18% and its weighted-average F1-score is 87.40%. These results verify the effectiveness and practical feasibility of the proposed hierarchical environmental noise recognition method for the urban acoustic environment monitoring.

## References

[1] CARRATÙ M, LIGUORI C, PACIELLO V, et al. A Sound Level Meter featured with automatic estimation of the measurement uncertainty[J]. Measurement, 2022, 188: 110409.

[2] ZAGUBIEŃ A, WOLNIEWICZ K. Development and validation of a portable test stand for sound measurement near the building façade[J]. Measurement, 2023, 214: 112856.

[3] BANSAL A, GARG N K. Environmental sound classification: A descriptive review of the literature[J]. Intelligent Systems with Applications, 2022, 16: 200115.

[4] TAN J K A, HASEGAWA Y, LAU S K. A comprehensive environmental sound categorization scheme of an urban city[J]. Applied Acoustics, 2022, 199: 109018.

[5] AUMOND P, LAVANDIER C, RIBEIRO C, et al. A study of the accuracy of mobile technology for measuring urban noise pollution in large scale participatory sensing campaigns[J]. Applied Acoustics, 2017, 117: 219-226.

[6] CAO J, CAO M, WANG J, et al. Urban noise recognition with convolutional neural network[J]. Multimedia Tools and Applications, 2019, 78(20): 29021-29041.

[7] İŞLER B. Urban sound recognition in smart cities using an IoT-fog computing framework and deep learning models: A performance comparison[J]. Applied Sciences, 2025, 15(3): 1201.

[8] HUANG Z, LIU C, FEI H, et al. Urban sound classification based on 2-order dense convolutional network using dual features[J]. Applied Acoustics, 2020, 164: 107243.

[9] XIE J, ZHONG Y, ZHANG J, et al. A review of automatic recognition technology for bird vocalizations in the deep learning era[J]. Ecological Informatics, 2023, 73: 101927.

[10] RADHAKRISHNAN R, DIVAKARAN A, SMARAGDIS P. Audio analysis for surveillance applications[C]//Proceedings of the 2005 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics. Piscataway: IEEE, 2005: 158-161.

[11] CROCCO M, CRISTANI M, TRUCCO A, et al. Audio surveillance: A systematic review[J]. ACM Computing Surveys, 2016, 48(4): 52:1-52:46.

[12] LAFFITTE P, WANG Y, SODOYER D, et al. Assessing the performances of different neural network architectures for the detection of screams and shouts in public transportation[J]. Expert Systems with Applications, 2019, 117: 29-41.

[13] MILIVOJČEVIĆ M, ĆIRIĆ D, PREZELJ J, et al. Analysis of unsupervised learning approach for classification of vehicle fuel type using psychoacoustic features[J]. Measurement, 2024, 227: 114318.

[14] LV M, SUN Z, ZHANG M, et al. Sound recognition method for white feather broilers based on spectrogram features and the fusion classification model[J]. Measurement, 2023, 222: 113696.

[15] WANG W, GAO Y. Pipeline leak detection method based on acoustic-pressure information fusion[J]. Measurement, 2023, 212: 112691.

[16] DONG X, YIN B, CONG Y, et al. Environment sound event classification with a two-stream convolutional neural network[J]. IEEE Access, 2020, 8: 125714-125721.

[17] HERSHEY S, CHAUDHURI S, ELLIS D P W, et al. CNN architectures for large-scale audio classification[C]//Proceedings of the 2017 IEEE International Conference on Acoustics, Speech and Signal Processing. Piscataway: IEEE, 2017: 131-135.

[18] İNİK Ö. CNN hyper-parameter optimization for environmental sound classification[J]. Applied Acoustics, 2023, 202: 109168.

[19] CINAR I. DeepSoundVisionNet: A new approach to urban sound classification using visual representations of audio signals[J]. Engineering Applications of Artificial Intelligence, 2026, 170: 114193.

[20] LI J, HAN L, LI X, et al. An evaluation of deep neural network models for music classification using spectrograms[J]. Multimedia Tools and Applications, 2022, 81(4): 4621-4647.

[21] WANG H, ZOU Y, CHONG D, et al. Environmental sound classification with parallel temporal-spectral attention[C]//Proceedings of Interspeech 2020. [S.l.]: ISCA, 2020: 821-825.

[22] LI D, LI C, YANG J, et al. Bayesian optimization-attention-feedforward neural network based train traction motor-gearbox coupled noise prediction[J]. Measurement, 2024, 238: 115323.

[23] LI Z, OUYANG B, XU X, et al. Non-stationary mechanical sound source separation: An all-neural beamforming network driven by time–frequency convolution and self-attention[J]. Measurement, 2025, 242: 115933.

[24] LIU Y, XU Z, HE Y, et al. Acoustic fault diagnosis method for rotating machinery based on improved spectral subtraction and CNN-TCN model[J]. Measurement, 2025, 256: 118482.

[25] DAI W, DAI C, QU S, et al. Very deep convolutional neural networks for raw waveforms[C]//Proceedings of the 2017 IEEE International Conference on Acoustics, Speech and Signal Processing. Piscataway: IEEE, 2017: 421-425.

[26] VALERO X, ALÍAS F. Hierarchical classification of environmental noise sources considering the acoustic signature of vehicle pass-bys[J]. Archives of Acoustics, 2012, 37(4): 423-434.

[27] GUZHOV A, RAUE F, HEES J, et al. ESResNet: Environmental sound classification based on visual domain models[C]//Proceedings of the 2020 25th International Conference on Pattern Recognition. Piscataway: IEEE, 2021: 4933- 4940.

[28] GONG Y, CHUNG Y A, GLASS J. AST: Audio spectrogram transformer[C]//Proceedings of Interspeech 2021. [S.l.]: ISCA, 2021: 571-575.

[29] RIAZATI SERESHT H, MOHAMMADI K. Environmental sound classification with low-complexity convolutional neural network empowered by sparse salient region pooling[J]. IEEE Access, 2023, 11: 849-862.

[30] RANMAL D, RANASINGHE P, PARANAYAPA T, et al. ESC-NAS: Environment sound classification using hardware-aware neural architecture search for the edge[J]. Sensors, 2024, 24(12): 3749.

[31] SALAMON J, JACOBY C, BELLO J P. A dataset and taxonomy for urban sound research[C]//Proceedings of the 22nd ACM International Conference on Multimedia. New York: ACM, 2014: 1041-1044.