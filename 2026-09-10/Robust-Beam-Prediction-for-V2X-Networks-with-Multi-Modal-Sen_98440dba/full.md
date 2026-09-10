# Robust Beam Prediction for V2X Networks with Multi-Modal Sensing

Chen Shang<sup>1</sup>, Dinh Thai Hoang<sup>1</sup>, Diep N. Nguyen<sup>1</sup>, and Jiadong Yu<sup>2</sup>

<sup>1</sup>School of Electrical and Data Engineering, University of Technology Sydney, Australia

<sup>2</sup>Internet of Things Thrust, The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China

Abstract—Integrated sensing and communication (ISAC) provides a promising foundation for beam prediction in future vehicle-to-everything (V2X) networks. However, existing sensingassisted beamforming methods still rely heavily on radiofrequency sensing, which may become unreliable in complex vehicular environments. Meanwhile, the growing availability of heterogeneous sensors, such as cameras and LiDAR, offers new opportunities to improve beam prediction through richer environmental perception. Motivated by this, this paper proposes a multi-modal beam prediction framework for V2X networks. Specifically, we develop BeamTransFuser, a hierarchical Transformer-based architecture that progressively fuses camera, LiDAR, radar, and GPS observations for robust beam prediction. In addition, to handle possible missing modalities in practical deployment, we introduce a generative module that reconstructs missing modality features from the available observations. Experimental results on a real-world multi-modal V2X dataset show that the proposed framework consistently outperforms representative baselines, while the generative module further improves robustness under incomplete sensing conditions.

Index Terms—Integrated sensing and communication, multimodal beam prediction, V2X networks, modality generation, 6G.

## I. INTRODUCTION

Wireless communication systems have evolved steadily over the past decades toward more intelligent and efficient operation. Among the technologies envisioned for sixthgeneration (6G) networks, integrated sensing and communication (ISAC) has emerged as a particularly important paradigm [1]. By unifying sensing and communication within the same framework, ISAC can lower signaling overhead, reduce beam alignment delay, and improve spectral efficiency [2]. These advantages are especially relevant to vehicleto-everything (V2X) networks, where rapid mobility and fastvarying link conditions require highly accurate beam alignment to sustain high-data-rate and low-latency communication services [3], [4].

Despite its promise, practical ISAC deployment in vehicular environments still faces significant challenges. On the one hand, existing ISAC design paradigms generally involve inherent trade-offs among sensing accuracy, communication efficiency, and implementation complexity [5], [6]. Specifically, sensing-centric designs usually prioritize sensing performance but may sacrifice communication throughput and standard compatibility, whereas communication-centric designs focus on transmission efficiency at the expense of sensing capability. As a result, neither of these two designs can simultaneously provide strong sensing and communication performance in a balanced manner. On the other hand, joint-design approaches attempt to balance both functions, yet they often require more sophisticated signal processing, tighter synchronization, and higher hardware overhead, which makes practical real-time implementation more challenging [5].

Furthermore, a common characteristic of these approaches is that they still rely predominantly on radio-frequency (RF)- based sensing. In dense urban roads, the quality of RF observations can deteriorate significantly because of blockage, multipath effects, and non-line-of-sight (NLoS) transmission conditions [7]. Such impairments may degrade sensing fidelity, distort environmental perception, and eventually reduce beam alignment reliability. This issue becomes even more critical in V2X networks, where high vehicle mobility and rapidly changing road conditions place stricter requirements on accurate and timely beam prediction. Therefore, relying on RF sensing alone is often insufficient for practical and reliable beam prediction in realistic V2X scenarios.

In this context, exploiting heterogeneous sensing modalities beyond RF signals becomes a natural way to improve beam prediction robustness [8]. For example, the position-based method in [9] relies mainly on GPS information for beam prediction, while TII [10] incorporates camera and GPS data through a Transformer-based design. CMDF [11] and ICMFE [12] further exploit camera-radar sensing for beamforming-related tasks. Moreover, the multimodal method in [9] considers richer sensing combinations involving camera, LiDAR, radar, and GPS. These works verify that introducing additional sensing modalities can improve beam prediction by providing complementary geometric, semantic, and environmental information. Nevertheless, most existing methods either consider only part of the available sensing modalities or lack a unified architecture that can fully exploit the complementarity among camera, LiDAR, radar, and GPS data. Moreover, they generally assume complete sensing inputs during inference. In practical deployment, however, some modalities may be unavailable due to sensor failure, environmental interference, or hardware constraints. In such cases, a model trained with fixed multi-modal inputs may encounter input mismatch and fail to operate properly during inference.

Motivated by the above observations, this paper develops a robust multi-modal beam prediction framework for practical V2X networks by jointly leveraging camera, Li-

DAR, radar, and GPS observations. Specifically, we propose BeamTransFuser, a hierarchical Transformer-based architecture that extracts modality-specific features and progressively fuses them through multi-stage cross-modal interaction, so as to better exploit the complementary information carried by these heterogeneous sensing modalities. In addition, to handle incomplete sensing conditions in practical deployment, we further introduce a generative module that reconstructs missing modality features from the available observations. In this way, the proposed framework can support effective beam prediction even when part of the sensing input is unavailable. The main contributions of this paper are summarized below:

• We propose BeamTransFuser, a hierarchical multi-modal beam prediction framework that jointly leverages camera, LiDAR, radar, and GPS observations for robust beam selection in V2X networks.

• We design a modality generation mechanism that reconstructs missing modality features from the available sensing inputs, thereby improving beam prediction robustness under incomplete sensing conditions.

• We conduct extensive experiments on a real-world multimodal V2X dataset. The results demonstrate that the proposed framework consistently surpasses representative baseline methods in beam prediction performance, while the generative module effectively reconstructs missing modality features to enable reliable beam prediction under incomplete sensing conditions.

## II. SYSTEM OVERVIEW AND PROBLEM FORMULATION

Fig. 1 presents the multi-modal sensing-assisted communication system considered in this work, where the RSU uses a uniform linear array for signal transmission and reception. Different from conventional ISAC frameworks that depend primarily on a single sensing source, the RSU integrates several sensing modalities, namely GPS, camera, LiDAR, and radar, to obtain real-time awareness of the surrounding environment. Specifically, since these modalities provide different types of information, they can complement one another under diverse propagation and mobility conditions. For example, when RF sensing is degraded by blockage or NLoS propagation, LiDAR can still provide geometric structure information, while camera observations can offer useful semantic cues for beam prediction. As a result, the RSU can exploit these heterogeneous sensing inputs to support more reliable beam prediction under practical V2X conditions.

We consider a predefined beamforming codebook ${ \mathcal { F } } =$ $\{ \mathbf { f } _ { m } \} _ { m = 1 } ^ { M }$ at the RSU, where each $\mathbf { f } _ { m }$ denotes a complexvalued beamforming vector and M is the total number of beam candidates. Let $\mathbf { f } _ { m }$ be the transmit beamformer applied to the downlink symbol s. Accordingly, the received signal at the vehicle is given by:

$$
\begin{array} { r } { \mathcal { C } = \mathbf { h } ^ { \mathrm { H } } \mathbf { f } _ { m } s + z _ { c } , } \end{array}\tag{1}
$$

where h is the downlink channel vector between the RSU and the vehicle, and $z _ { c } ~ \sim ~ \mathcal { C N } ( 0 , \sigma ^ { 2 } )$ denotes circularly symmetric complex Gaussian noise. Accordingly, the signal-

![](images/270b81e23a3057e84af37e9610e3bba759e4e523f840a3aaa6794879bf3fdd7e.jpg)  
Fig. 1: Multi-modal sensing-assisted beam prediction system. The RSU collects real-time observations, processes them with a pretrained model, and then determines the beamforming decision.

to-noise ratio (SNR) and the corresponding achievable rate can be respectively expressed as:

$$
\gamma ( { \bf f } _ { m } ) = \frac { | { \bf h } ^ { \mathrm { H } } { \bf f } _ { m } | ^ { 2 } } { \sigma ^ { 2 } } , ~ R = \log _ { 2 } \big ( 1 + \gamma ( { \bf f } _ { m } ) \big ) .\tag{2}
$$

Therefore, the optimal beamforming vector is selected as:

$$
\mathbf { f } ^ { * } = \underset { \mathbf { f } _ { m } \in \mathcal { F } } { \arg \operatorname* { m a x } } \log _ { 2 } \big ( 1 + \gamma ( \mathbf { f } _ { m } ) \big ) .\tag{3}
$$

Instead of performing exhaustive beam search during online transmission, this work considers multi-modal sensingassisted beam prediction, where heterogeneous sensing observations are directly used to infer the target beam index. Let $\mathcal { M } _ { \Theta } ( \cdot )$ denote the beam prediction model parameterized by Θ. Suppose the training dataset is given by $\boldsymbol { \mathcal { X } } = \{ \mathcal { X } _ { i } \} _ { i = 1 } ^ { N } ,$ where each sample contains multi-modal sensing inputs, and let $\mathcal { Y } = \{ \mathcal { V } _ { i } \} _ { i = 1 } ^ { N }$ denote the corresponding beam-index labels. Therefore, the learning task can be formulated as:

$$
\mathbf { P 1 } \colon \Theta ^ { * } = \underset { \Theta } { \arg \operatorname* { m i n } } ~ \sum _ { i = 1 } ^ { N } \mathcal { L } \left( \mathcal { M } _ { \Theta } ( \mathcal { X } _ { i } ) , \mathcal { Y } _ { i } \right) ,\tag{4}
$$

where $\mathcal { L } ( \cdot )$ represents the training loss. The core difficulty of P1 lies in how to effectively extract useful representations from heterogeneous sensing modalities and fuse them into a unified feature space for accurate beam prediction. To tackle this problem, we next introduce the proposed BeamTransFuser framework.

## III. THE PROPOSED MULTI-MODAL FRAMEWORK

As shown in Fig. 2, the proposed BeamTransFuser consists of four modality-specific branches for camera, LiDAR, radar, and GPS inputs, followed by a hierarchical fusion backbone built on Transformer modules. Specifically, each branch first extracts features from its corresponding sensing modality, allowing modality-aware representations to be learned before cross-modal interaction is introduced. The extracted features are then progressively fused through multi-modal fusion blocks inserted at different stages of the backbone, so that cross-modal dependencies can be modeled from low-level representations to high-level semantic features. Moreover, the backbone adopts a four-stage hierarchical design, where each modality is processed through progressive layers, while multimodal fusion blocks are interleaved between adjacent stages to enable cross-modal interaction across different representation levels. In addition, residual connections (i.e., ⊕) are employed to retain modality-specific information and improve training stability. After the final fusion stage, the resulting modality features are pooled and integrated by a learnable fusion module with softmax-normalized weights (i.e., ⊗) before being passed to the decoder and beam generator for beam index prediction. The details of the proposed framework are presented as follows.

![](images/31568fe99823d232da5502c83a552d1d58a903ec0ddd793cbb8ce7d4776f5df9.jpg)  
Fig. 2: BeamTransFuser architecture with modality-specific branches and hierarchical cross-modal fusion.

## A. Multi-Modal Branches

As shown in Fig. 2, camera, LiDAR, and radar inputs are first processed by dedicated convolutional encoders that transform raw data into spatial feature representations. In contrast, GPS data are low-dimensional and are mapped into the common feature space through a multilayer perceptron, without relying on the same spatial encoding pipeline. After this initial encoding step, each modality is further handled by a modality-aware feature extractor. Specifically, ResNet34 [13] is used in the camera stream to extract discriminative visual features from RGB images, whereas lighter ResNet16 [13] networks are adopted for the LiDAR and radar branches to reduce computational complexity while preserving essential structural cues.

To enable progressive cross-modal interaction, the encoded modality-specific representations are further processed by the hierarchical fusion backbone. At each stage, the features are first aligned to a common representation scale and then converted into token sequences for Transformer-based fusion. This design allows modality-specific characteristics to be retained while gradually integrating complementary information across different sensing sources. Consequently, low-level geometric patterns and high-level semantic cues can be fused in a progressive manner, which is particularly useful in dynamic V2X scenarios where both local structural details and global contextual information influence beam prediction.

## B. Multi-Modal Fusion Block

Effectively integrating camera, LiDAR, radar, and GPS features is nontrivial, since these modalities differ substantially in spatial structure, semantic meaning, and information density. To this end, we employ the Transformer [14] to fuse heterogeneous modality features and model cross-modal dependencies within a unified representation space. At each fusion stage, the modality-specific features are first pooled and projected into a common token representation. By concatenating the tokens from all sensing modalities, we obtain the unified token sequence $\mathbf { F } \in \mathbb { R } ^ { \mathcal { N } \times \mathcal { D } }$ , where $\mathcal { N }$ denotes the total number of tokens and D is the embedding dimension. Accordingly, the query (Q), key (K), and value (V) matrices are computed as [14]:

$$
\mathbf { Q } = \mathbf { F W } ^ { \mathrm { Q } } , \quad \mathbf { K } = \mathbf { F W } ^ { \mathrm { K } } , \quad \mathbf { V } = \mathbf { F W } ^ { \mathrm { V } } ,\tag{5}
$$

where $\mathbf { W } ^ { \mathrm { Q } } , \mathbf { W } ^ { \mathrm { K } }$ , and ${ \bf W } ^ { \mathrm { V } }$ are learnable projection matrices. The fused token representation, denoted by $\mathbf { \tilde { F } } \in \mathbb { R } ^ { \mathcal { N } \times \mathcal { D } }$ , is then obtained through scaled dot-product attention [14]:

$$
\mathrm { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } ) = \mathrm { s o f t m a x } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { { \mathcal { D } } / h } } } \right) ,
$$

$$
\tilde { \mathbf { F } } = \mathrm { A t t e n t i o n } ( \mathbf { Q } , \mathbf { K } ) \mathbf { V } ,\tag{6}
$$

(7)

where h is the number of attention heads. Through this operation, each token in F is updated by selectively aggregating informative features from other tokens in the shared multimodal sequence, producing the fused representation F<sup>˜</sup> . After fusion, F<sup>˜</sup> is reorganized according to the original modality partitions and returned to the corresponding branches for subsequent processing. For camera, LiDAR, and radar, the fused tokens are reshaped into feature maps and passed to the next stage. For GPS, since it does not have a native 2D spatial structure, the fused representation is kept in token form and directly forwarded to the next stage.

## C. Beam Generator and Model Training

After the final fusion stage, the modality features are integrated through a learnable fusion module with softmaxnormalized weights (i.e., ⊗), so that the contribution of each modality can be adaptively adjusted. Specifically, let $\mathbf { f } _ { m }$ denote the globally pooled feature of modality m, and let $\alpha _ { m }$ and $w _ { m }$ denote its normalized importance weight and learnable scalar score, respectively. The normalized weight $\alpha _ { m }$ is obtained through a softmax operation:

$$
\alpha _ { m } = \frac { \exp ( w _ { m } ) } { \sum _ { j } \exp ( w _ { j } ) } ,\tag{8}
$$

The final fused representation is then given by:

$$
\hat { \mathbf { F } } = \sum _ { m } \alpha _ { m } \mathbf { f } _ { m } .\tag{9}
$$

This fused representation is passed to a compact MLP-based beam generator to produce the beam score vector:

$$
\begin{array} { r } { \hat { \mathcal { Y } } = \operatorname { M L P } ( \hat { \mathbf { F } } ) , } \end{array}\tag{10}
$$

The predicted beam index corresponds to the beamforming vector in the codebook with the largest score. In this way, more accurate beam prediction directly translates into better beam alignment and, consequently, improved communication performance.

## IV. ENHANCING BEAMTRANSFUSER ROBUSTNESS VIA MODALITY GENERATION

To improve the robustness of BeamTransFuser under incomplete sensing conditions, we introduce a generative module for missing-modality completion. The role of this module is to infer the representation of an unavailable sensing modality from the remaining observations, so that the downstream beam prediction pipeline can still operate when the multimodal input is incomplete. Specifically, a generative model learns how to map latent variables to data samples, thereby producing outputs that follow the semantic and structural patterns observed in the training data [15].

In this work, we build the proposed generation mechanism on the variational auto-encoder (VAE) framework [16]. In particular, this choice is motivated by the deployment requirements of the considered V2X scenario. Compared with generative adversarial networks (GANs) [17], VAE-based models are generally easier to optimize and offer more stable training behavior, especially when the target data distribution is complex or highly diverse. Compared with diffusion-based models [18], VAE-based generation is also more suitable for delay-sensitive applications, since diffusion-based generation typically involves a long sequence of denoising updates at inference time, resulting in significantly higher latency. By contrast, VAE-based generation can be performed in a lightweight feed-forward manner, which is more compatible with realtime beam prediction in V2X systems [16]. Furthermore, since missing-modality completion requires the generated output to remain consistent with the currently observed sensing inputs, we further adopt the conditional variational autoencoder (CVAE), in which both the encoder and decoder are conditioned on the available modalities [19].

Let $\mathbf { x } \in \mathcal { X }$ denote the available modalities and let $\mathbf { y } \in \mathcal { X }$ denote the missing modality to be generated. Under this setting, the objective is to infer y conditioned on x, which is equivalent to modeling the conditional distribution $p ( \mathbf { y } \vert \mathbf { x } )$ For instance, if radar observations are missing while camera and LiDAR are available, then camera and LiDAR form x and the missing radar modality corresponds to y. In this way, the generation process is guided by the observed sensing context rather than by an unconditional data prior [19]. Specifically, to increase the flexibility of the conditional generation process, a latent variable z is introduced to capture hidden factors that are not explicitly represented by the observed modalities. Accordingly, the conditional distribution of $\mathbf { y }$ given x can be expressed as:

$$
p ( \mathbf { y } \vert \mathbf { x } ) = \int _ { \mathbf { z } } p _ { \pmb { \theta } } ( \mathbf { y } \vert \mathbf { x } , \mathbf { z } ) p ( \mathbf { z } \vert \mathbf { x } ) d \mathbf { z } ,\tag{11}
$$

where $p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { x } , \mathbf { z } )$ denotes the conditional likelihood of generating y given x and z, parameterized by $\theta \ \mathrm { ( i . e . }$ , the decoder), and $p ( \mathbf { z } | \mathbf { x } )$ is the prior distribution of the latent variable. The generative model is then trained to learn this conditional distribution, which is equivalent to maximizing the corresponding conditional likelihood. However, since this likelihood involves marginalizing over the latent variable ${ \mathbf { z } } ,$ it is generally intractable to optimize directly. To this end, a variational posterior $q _ { \phi } ( \mathbf { z } | \mathbf { x } , \mathbf { y } )$ , parameterized by $\phi \ ( \mathrm { i . e . }$ the encoder), is introduced to approximate the true posterior, and the model is trained by optimizing the standard CVAE

objective [16], [19]:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { C V A E } } ( \boldsymbol { \theta } , \boldsymbol { \phi } ) = - \mathbb { E } _ { \mathbf { z } \sim q _ { \boldsymbol { \phi } } ( \mathbf { z } | \mathbf { x } , \mathbf { y } ) } \left[ \log p _ { \boldsymbol { \theta } } ( \mathbf { y } | \mathbf { x } , \mathbf { z } ) \right] } \\ & { \qquad + \mathrm { K L } \left( q _ { \boldsymbol { \phi } } ( \mathbf { z } | \mathbf { x } , \mathbf { y } ) \parallel p ( \mathbf { z } | \mathbf { x } ) \right) . } \end{array}\tag{12}
$$

The first term in (12) corresponds to the reconstruction objective, which encourages the decoder to produce a modality representation aligned with the target y. The second term regularizes the variational posterior toward the latent prior, which helps improve the stability and generalization capability of the generative module. In addition, after the initial CVAE training stage, we further perform a lightweight task-aware fine-tuning step, in which the reconstruction objective is jointly optimized with a downstream beam prediction loss. This refinement improves the task relevance of the generated modality while maintaining consistency with the target feature space.

It is worth noting that explicitly modeling an inputdependent conditional prior $p ( \mathbf { z } | \mathbf { x } )$ generally requires an additional prior network, which increases optimization complexity as well as inference overhead. To keep the generative module lightweight for delay-sensitive V2X deployment, we follow [19], [20] and relax the conditional prior to a fixed isotropic Gaussian distribution, i.e., $p ( \mathbf { z } | \mathbf { x } ) = p ( \mathbf { z } ) = \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ Note that this simplification only affects the latent prior. The overall model remains conditional through the variational posterior $q _ { \phi } ( \mathbf { z } | \mathbf { x } , \mathbf { y } )$ and the decoder $p _ { \pmb { \theta } } ( \mathbf { y } | \mathbf { x } , \mathbf { z } )$ . Finally, the loss in (12) can then be optimized efficiently via stochastic gradient descent together with the reparameterization trick [19].

We illustrate the modality generation process as follows. First, the generative module is trained offline and independently of BeamTransFuser using paired incomplete and complete modality samples. After training, only the decoder is retained for inference. Given the available modalities x, we sample a latent variable $\mathbf { z } \sim p ( \mathbf { z } )$ and feed both x and z into the decoder to reconstruct the missing modality. The generated modality features are then injected into BeamTransFuser at the feature level, i.e., between the feature extraction stage and the multi-modal fusion backbone in Fig. 2. Finally, the generated features are jointly processed with the observed modality features by the hierarchical fusion backbone to perform beam prediction.

## V. PERFORMANCE EVALUATION

## A. Dataset and Settings

We evaluate the proposed BeamTransFuser on the DeepSense 6G dataset [21], a real-world multi-modal V2X dataset that is openly accessible to the research community. The dataset provides synchronized camera, LiDAR, radar, GPS, and beam-label data, where each label corresponds to the optimal beam index selected from a 64-beam codebook. Moreover, this dataset is well suited for evaluating the considered task, as it reflects realistic urban V2X environments with LoS/NLoS transitions, multipath propagation, traffic dynamics, and illumination variation. In particular, we focus on the urban street scenarios 31-34 [21], in which the RSU collects multi-modal sensing data from an RGB camera, radar,

![](images/edf52f54fe092047b739efde2453e708589df14b40ac4c15f99852c2bf80a555.jpg)  
(a)

![](images/850de5582efe26c8533c7df01fa76d843a6db60b30d0fb82f7442fe6f3ad31c7.jpg)  
(b)  
Fig. 3: Training behavior of the proposed model. (a) Training and validation loss versus epoch. (b) Training and validation loss versus training time.

LiDAR, the mmWave receiver, and GPS. After combining the development and adaptation subsets, we obtain 11,243 samples in total, among which 90% are used for training and the remaining 10% for validation.

For BeamTransFuser, the convolutional encoders are configured with a kernel size of $7 \times 7$ and stride 2, and each multi-modal fusion block uses four attention heads with a feed-forward expansion ratio of four. The model is trained for 30 epochs using focal loss with a learning rate of $1 0 ^ { - 4 }$ In the evaluation, we adopt DBA-score and Top-k accuracy as the main performance metrics [21]. Specifically, DBA-score characterizes how far the predicted beam is from the target beam, and thus serves as a fine-grained measure of beam alignment quality [21]. We further compare the proposed framework with the following baselines reported on the same dataset, including Avatar [22], the position-based and multimodal methods in [9], TII [10], CMDF [11], ICMFE [12], and QTNs [23].

## B. Experimental Results

1) Convergence Behavior and Training Time: We first examine the convergence behavior and training time in Fig. 3. As shown in Fig. 3(a), both the training and validation losses decrease steadily during training and gradually stabilize afterward, without any noticeable increase in the validation loss at later epochs. This indicates that the proposed model does not exhibit evident overfitting and maintains satisfactory generalization performance on the validation set. Fig. 3(b) further presents the loss evolution with respect to training time. All training experiments were performed on an NVIDIA GeForce RTX 4090 GPU. It can be seen that the loss curves become much flatter after approximately one hour, suggesting that the optimization has largely converged, while further training only brings marginal improvement.

2) Beam Prediction Performance: We then evaluate beam prediction performance in terms of DBA-score. As shown in Table I, the proposed method achieves the highest overall DBA-score among all compared schemes. It also exhibits consistently strong results in all four scenarios, with DBAscores of 1.0000, 0.9038, 0.8988, and 0.8945 for Scenarios 31, 32, 33, and 34, respectively. This overall stability indicates that the proposed framework can maintain reliable beam prediction under diverse urban conditions, including variations in illumination across daytime scenarios (Scenarios 31 and 32) and nighttime scenarios (Scenarios 33 and 34), as well as different propagation conditions under both LoS and NLoS settings.

TABLE I: DBA-score results of selected beam prediction schemes
<table><tr><td>Scheme</td><td>Overall</td><td>S31</td><td>S32</td><td>S33</td><td>S34</td></tr><tr><td>Avatar [22]</td><td>0.7162</td><td>0.6536</td><td>0.7074</td><td>0.8576</td><td>0.7120</td></tr><tr><td>[9]</td><td></td><td></td><td>0.8906</td><td></td><td></td></tr><tr><td>TII [10]</td><td>0.7844</td><td>0.7298</td><td>0.7852</td><td>0.8462</td><td>0.8433</td></tr><tr><td>CMDF [11]</td><td>0.8910</td><td></td><td></td><td></td><td></td></tr><tr><td>ICMFE [12]</td><td>0.8969</td><td>1.0000</td><td>0.9020</td><td>0.8874</td><td>0.9074</td></tr><tr><td>QTNs [23]</td><td></td><td>0.7605</td><td>0.8707</td><td>0.8864</td><td>0.9124</td></tr><tr><td>BeamTransFuser (Ours)</td><td>0.9129</td><td>1.0000</td><td>0.9038</td><td>0.8988</td><td>0.8945</td></tr></table>

– indicates an unreported result in the cited work.

However, some baselines follow a different trend. For instance, Avatar [22] and TII [10] both achieve higher DBAscores in Scenario 33 than in Scenario 32, with improvements of 0.15 and 0.061, respectively. This may indicate that some nighttime scenes, such as those with weaker background clutter or more homogeneous illumination, are more favorable to these methods. At the same time, the amount of improvement is not uniform across baselines, which suggests that their effectiveness is strongly influenced by how different sensing modalities are utilized and fused.

Compared with these baselines, BeamTransFuser remains much more stable from Scenarios 32 to 34, reflecting stronger robustness to environmental variation. It is also worth noting that, although QTNs [23] and the method in [12] obtain slightly higher DBA-scores in Scenario 34, the margins are small, namely 1.99% and 1.42%, respectively. From an overall perspective, BeamTransFuser still ranks first with a DBAscore of 0.9129. These results suggest that, even though some baselines perform competitively in individual scenarios, BeamTransFuser remains more stable over the full set of evaluated conditions, demonstrating stronger generalization and more reliable beam prediction for realistic V2X deployment.

The strong DBA performance of BeamTransFuser can be attributed to two main factors. First, the model jointly exploits complementary information from camera, LiDAR, radar, and GPS, allowing geometric structure, visual semantics, motion cues, and location information to be utilized in a unified manner. This is particularly beneficial in urban V2X scenarios, where a single sensing modality may become unreliable under blockage, NLoS propagation, or environmental variation. Second, the hierarchical Transformer-based fusion design enables cross-modal information to be progressively integrated across multiple representation levels, rather than being merged only once at the final stage. As a result, BeamTransFuser can better capture modality-specific structural information together with semantic context, which leads to more robust and consistent beam prediction across diverse scenarios.

3) Performance with Modality Completion: We next investigate the robustness of BeamTransFuser in the presence of missing modalities. To emulate practical cases where one sensing modality is unavailable, e.g., due to sensor failure or incomplete deployment, we replace the missing input with either zero-filled data or Gaussian noise and treat the resulting outputs as degraded baselines. We then compare these baselines with the case where the missing modality is reconstructed by the proposed generative model.

TABLE II: Top-k beam prediction accuracy under missingmodality conditions
<table><tr><td>Miss. Mod.</td><td>Repl. Cond.</td><td>Top-1 (%) Top-2 (%) Top-3 (%)</td><td></td><td></td></tr><tr><td rowspan="3">Radar</td><td>Zero-filled</td><td>6.67</td><td>9.07</td><td>15.02</td></tr><tr><td>Gaussian noise</td><td>6.67</td><td>9.42</td><td>14.22</td></tr><tr><td>Gen (Cam+LiDAR)</td><td>45.51</td><td>70.31</td><td>82.58</td></tr><tr><td rowspan="3">LiDAR</td><td>Zero-filled</td><td>6.67</td><td>9.07</td><td>13.96</td></tr><tr><td>Gaussian noise</td><td>6.67</td><td>9.07</td><td>13.78</td></tr><tr><td>Gen (Cam+Radar)</td><td>56.18</td><td>79.82</td><td>89.07</td></tr><tr><td rowspan="2">Camera</td><td>Zero-filled</td><td>2.40</td><td>4.27</td><td>6.40</td></tr><tr><td>Gaussian noise</td><td>3.02</td><td>6.31</td><td>8.27</td></tr></table>

As shown in Table II, when Radar or LiDAR is unavailable, the Top-1 accuracy drops sharply under both zero-filled and Gaussian-noise replacements, indicating that the absence of these sensing inputs severely degrades beam prediction performance. By contrast, replacing the missing modality with CVAE-generated features substantially restores the prediction accuracy, which verifies the effectiveness of the proposed modality generation mechanism under incomplete sensing conditions. Specifically, in the missing-Radar case, the Top-1 accuracy improves from 6.67% to 45.51%, while in the missing-LiDAR case, it increases from 6.67% to 56.18%. These notable gains indicate that the generative module can recover informative modality features from the remaining observations and provide meaningful compensation when one sensing source is unavailable. We do not report the generatedcamera case, because reconstructing high-dimensional RGB features is considerably more difficult and introduces much higher complexity. Even so, the strong recovery achieved in the missing-Radar and missing-LiDAR settings already demonstrates the practical value of the proposed generation mechanism for robust beam prediction under incomplete sensing.

## VI. CONCLUSION

In this work, we have proposed a multi-modal beam prediction framework for V2X networks based on hierarchical fusion and modality generation. Specifically, we have developed BeamTransFuser, which exploits complementary information from camera, LiDAR, radar, and GPS through progressive Transformer-based fusion for accurate and robust beam prediction. To further handle possible incomplete sensing conditions in practical deployment, we have incorporated a generative module that reconstructs missing modality features from the available observations, thereby improving robustness without requiring retraining. Experimental results on a realworld multi-modal V2X dataset have demonstrated the effectiveness of the proposed approach in improving beam prediction accuracy over competing schemes. They also confirm that the generative module enables stable beam prediction when part of the sensing input is unavailable. In future work, we will extend this framework to more challenging urban scenarios and multi-vehicle settings.

## REFERENCES

[1] ITU-R WP5D, “Draft New Recommendation ITU-R M. [IMT. Framework for 2030 and Beyond],” 2023.

[2] D. Zhang et al., “Integrated sensing and communications over the years: An evolution perspective,” IEEE Commun. Surv. Tutor., pp. 1–1, 2026.

[3] C. Shang, J. Yu, and D. Thai Hoang, “Energy-efficient and intelligent isac in V2X networks with spiking neural networks-driven DRL,” IEEE Trans. Wireless Commun., vol. 25, pp. 1182–1195, 2026.

[4] C. Shang, D. T. Hoang, and J. Yu, “Multi-modal beamforming with model compression and modality generation for v2x networks,” IEEE Trans. Mobile Comput., pp. 1–15, 2026.

[5] Y. Xiong et al., “On the fundamental tradeoff of integrated sensing and communications under gaussian channels,” IEEE Trans. Inf. Theory, vol. 69, no. 9, pp. 5723–5751, 2023.

[6] C. Shang et al., “Sensing-assisted swipt with hybrid learning for lowpower sensors on aerial-to-ground mobile platforms,” IEEE J. Sel. Areas Commun., vol. 44, pp. 5043–5059, 2026.

[7] Z. Du et al., “Toward ISAC-empowered vehicular networks: Framework, Advances, and Opportunities,” IEEE Wireless Commun., vol. 32, no. 2, pp. 222–229, 2025.

[8] X. Cheng et al., “Intelligent multi-modal sensing-communication integration: Synesthesia of machines,” IEEE Commun. Surv. Tutorials, vol. 26, no. 1, pp. 258–301, 2024.

[9] B. Shi et al., “Multimodal deep learning empowered millimeter-wave beam prediction,” in 2024 IEEE 99th Vehicular Technology Conference, 2024, pp. 1–6.

[10] Y. Tian et al., “Multimodal transformers for wireless communications: A case study in beam prediction,” arXiv preprint arXiv:2309.11811, 2023.

[11] Q. Zhu et al., “Advancing multi-modal beam prediction with multipathlike data augmentation and efficient fusion mechanism.” New York, NY, USA: Association for Computing Machinery, 2024.

[12] Q. Zhu, Y. Wang, W. Li, H. Huang, and G. Gui, “Advancing multi-modal beam prediction with cross-modal feature enhancement and dynamic fusion mechanism,” IEEE Trans. Commun., vol. 73, no. 9, pp. 7931– 7940, 2025.

[13] K. He et al., “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[14] A. Vaswani et al., “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[15] S. Bond-Taylor et al., “Deep generative modelling: A comparative review of vaes, gans, normalizing flows, energy-based and autoregressive models,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 7327–7347, 2022.

[16] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in International Conference on Learning Representations, Apr. 2014, pp. 1–14.

[17] I. J. Goodfellow et al., “Generative adversarial nets,” Advances in neural information processing systems, vol. 27, 2014.

[18] F.-A. Croitoru et al., “Diffusion models in vision: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 9, pp. 10 850–10 869, 2023.

[19] K. Sohn et al., “Learning structured output representation using deep conditional generative models,” in Advances in Neural Information Processing Systems, C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, Eds., vol. 28. Curran Associates, Inc., 2015.

[20] D. P. Kingma et al., “Semi-supervised learning with deep generative models,” Advances in neural information processing systems, vol. 27, 2014.

[21] A. Alkhateeb et al., “Deepsense 6G: A large-scale real-world multimodal sensing and communication dataset,” IEEE Commun. Mag., vol. 61, no. 9, pp. 122–128, 2023.

[22] “Deepsense ITU multi modal beam prediction challenge 2022 – deepsense,” Deepsense6g.net, 2022. [Online]. Available: https: //www.deepsense6g.net/challenge2022

[23] S. Tariq et al., “Deep quantum-transformer networks for multimodal beam prediction in isac systems,” IEEE Internet Things J., vol. 11, no. 18, pp. 29 387–29 401, 2024.