# Bidirectional Multimodal Fusion of Sky Images and Time-Series for Solar Forecasting with Large Language Models

Ken Chen<sup>a,∗</sup>, Maneesha Perera<sup>a</sup>, Wei Wang<sup>a</sup>, Sachith Seneviratne<sup>a</sup>, Hansani Weeratunge<sup>b</sup>, Saman Halgamuge<sup>a</sup>

<sup>a</sup>Department of Mechanical Engineering, The University of Melbourne, Melbourne, Australia

<sup>b</sup>Department of Mechanical Engineering, Sri Lanka Institute of Information Technology, Sri Lanka

## Abstract

Short-term photovoltaic (PV) power and global horizontal irradiance (GHI) forecasts are essential for efective dispatch, reserve scheduling, and grid operations with high PV penetration. At these forecasting horizons, errors are predominantly driven by cloud induced ramps: relying solely on historical numerical data may struggle to anticipate an incoming cloud, making ground-based sky images a crucial complementary physical signal. Furthermore, forecast performance is highly sensitive to location and local observing conditions, creating a strong need for site-specific data that are often scarce. Recently, large language models (LLMs) have demonstrated competitive performance and high data eficiency in time-series forecasting. Cross-modality alignment maps time-series data patches into the language embedding space, allowing pretrained representations to be leveraged for time-series forecasting with minimal task-specific data. Despite their success, existing LLM-based forecasting methods remain predominantly unimodal, relying primarily on historical numerical time-series data. Efectively incorporating sky imagery into an LLM-based forecasting framework remains under-explored and an open challenge. In this paper, we propose SolCloudLLM, an LLM-based multimodal forecasting framework. SolCloudLLM aligns sky-image patches with time-series patches and fuses their corresponding representations through

bidirectional multimodal fusion, yielding a unified representation that is subsequently mapped into the embedding space of an LLM. Extensive experiments on the SIRTA and SKIPP’D datasets demonstrate that SolCloudLLM consistently outperforms the best baseline methods in Mean Squared Error (MSE) across all forecasting horizons (H ∈ {16, 32, 64}), achieving a maximum relative MSE reduction of 25.4% on SIRTA. Stratified analysis on SKIPP’D further indicates that the benefits of multimodal fusion are concentrated primarily under cloudy conditions at H=32 and H=64. Notably, SolCloudLLM achieves the best performance in nearly all few-shot settings (using only 5% to 10% of training data), whereas other deep learning baselines experience substantial performance degradation and are frequently outperformed by the non-learning physical method.

Keywords: solar forecasting, sky images, large language models, multimodal forecasting, bidirectional multimodal fusion, photovoltaic power

## 1. Introduction

The reliable integration of solar energy into modern power grids requires precise short-term (i.e. intra-hour) forecasts of photovoltaic (PV) output and global horizontal irradiance (GHI). Such accurate predictions are the cornerstone for optimal dispatch, dynamic reserve scheduling, and stable market operations [1]. In these short-term horizons, the main source of forecasting error is caused by the rapid fluctuations in solar power caused by moving clouds. While a passing cloud can drastically reduce solar irradiance in under a minute, historical numerical data is inherently blind to such approaching meteorological shifts [2]. As a result, purely time-series forecasting methods are fundamentally reactive, typically adjusting only after a sudden drop in solar power output has been observed. Incorporating ground-based sky imagery helps address this blind spot. By continuously monitoring cloud trajectories and optical textures, these sky images supply a forward-looking visual context that purely numerical sequences lack [3, 4].

Beyond temporal dynamics, cross-site discrepancies in camera hardware, local microclimates, and PV array specifications induce significant domain shifts, causing models optimized for one facility often struggle to generalize to a new site without localized retraining [5, 6]. Compounding this issue is the high acquisition cost of paired multimodal datasets, which leaves most target sites with strictly limited historical records. Under such constrained conditions, developing highly data-eficient methods is no longer merely advantageous, but also becomes a strict operational necessity.

Recently, large language models (LLMs) have been adapted for numerical time-series forecasting, leveraging the rich sequential priors encoded within frozen pretrained transformers. This is typically achieved through cross-modality alignment, which maps patched time-series tokens into the language embedding space, enabling the backbone to generate forecasts without requiring training from scratch on new datasets [7, 8]. This paradigm not only achieves competitive accuracy on widely used long-term forecasting benchmarks (including ETT, Weather, Electricity (ECL), Trafic, and ILI [9]), but also enables highly data-eficient few-shot adaptation when labeled data are scarce, as the majority of LLM parameters remain frozen. For instance, Time-LLM—a representative reprogramming framework—aligns patched numerical tokens with a frozen language backbone via cross-attention and prefix prompting [8]. While this framework, along with broader LLM applications [10], has been introduced to solar power prediction [11, 12], existing pipelines remain strictly unimodal, relying solely on historical time-series numerical sequences without integrating synchronized sky images.

On the other hand, traditional sky-image nowcasting demonstrates that modeling cloud kinematics can significantly reduce forecast lag and preempt abrupt irradiance fluctuations [13, 14]. Similarly, satellite-based cloud-tracking pipelines enhance downstream distributed PV forecasts, particularly when visual feature extraction is optimized end-to-end against the final power prediction [15]. Concurrently, another line of research fuses ground-based sky imagery with historical PV or GHI sequences using task-specific convolutional or attention-based architectures [16, 17, 18]. While these studies definitively establish the complementary nature of visual and temporal data, they ofer no mechanisms for integrating sky images into the aforementioned LLM frameworks. Consequently, efectively injecting sky imagery into an LLM architecture, while preserving its inherent data eficiency, remains underexplored and a critical open challenge.

To address this gap, we propose SolCloudLLM, a novel multimodal reprogramming framework that seamlessly integrates sky imagery with numerical time-series sequences. Unlike recent vision-language approaches that rely on heavy late-fusion mechanisms (fusing modalities only after they have been independently encoded) [19], SolCloudLLM integrates the visual and numerical streams before aligning the fused representation with the language embedding. Specifically, sky imagery frames are encoded via a lightweight CNN, and the resulting visual features are patched and temporally aligned with time-series patches. Furthermore, rather than employing plain concatenation, which merely juxtaposes the two token streams without cross-modal interaction, we introduce a token-level Bidirectional Multimodal Fusion module inspired by feature-wise linear modulation (FiLM) [20]. This mechanism enables each modality to dynamically generate scale-and-shift parameters for the other: sky-image tokens modulate time-series patches, allowing the recent numerical history to be interpreted under the current cloud state; conversely, time-series tokens modulate image patches, emphasizing or suppressing visual cues based on recent numerical context. The modulated streams are then fused via concatenation and subsequent projection, forming a unified representation that is reprogrammed and mapped into a frozen LLM backbone. By maintaining a shared temporal index between modalities prior to mapping, SolCloudLLM attaches precise cloud context to the corresponding look-back segments. Importantly, relying on a lightweight fusion design alongside a frozen LLM backbone ensures that SolCloudLLM retains the exceptional data-eficient adaptation capabilities of LLM-based time-series forecasters, maintaining optimal performance even under strict few-shot constraints.

The main contributions are as follows.

1. We propose SolCloudLLM, a multimodal forecasting framework based on LLM - that precisely aligns sky-image tokens with time-series patches and fuses them via token-level bidirectional multimodal fusion before mapping the unified representation into a frozen LLM backbone.

2. Extensive evaluations on open access solar datasets - SIRTA [21] and SKIPP’D [22] demonstrate that SolCloudLLM consistently outperforms baselines in terms of Mean Squared Error (MSE) across forecasting horizons $H \in \{ 1 6 , 3 2 , 6 4 \}$ . Notably, it achieves peak relative MSE of 25.4% on SIRTA (H=32) and 14.5% on SKIPP’D (H=64) compared to the best baseline method.

3. Through stratified condition-aware and data-eficiency analyses, we explicitly identify the scenarios where visual modalities excel: multimodal gains are most pronounced under cloudy conditions at H=32 and H=64. Furthermore, in few-shot settings utilizing only 5% to 10% of training data, SolCloudLLM exhibits remarkable robustness, often achieving larger relative improvements over pure time-series models than in full-data regimes.

The remainder of this paper is organized as follows: Section 2 reviews the related work, followed by a detailed description of the proposed architecture in Section 3. Section 4 presents the datasets, baselines, and experimental results, while Section 5 provides concluding remarks.

## 2. Related work

## 2.1. Time-series forecasting for solar power and irradiance

Classical unimodal baselines include smart persistence, statistical methods, and predictors augmented by numerical weather prediction (NWP) or physical constraints [1, 23]. While these approaches serve as robust benchmarks, particularly in data-scarce regimes, they lack the capacity to capture forwardlooking cloud kinematics, relying entirely on the dynamics implicit within recent historical sequences.

To capture complex, nonlinear temporal dependencies from numerical histories, deep learning architectures have evolved significantly. Early approaches utilized recurrent networks, notably Long Short-Term Memory (LSTM) [24], and temporal convolutional networks [25]. The field subsequently shifted toward transformer-based forecasters, introducing architectures such as Informer [26], Autoformer [27], and PatchTST [28], which leverages channelindependent patching. Concurrently, linear baselines like DLinear [29] have proven highly competitive in scenarios where complex attention mechanisms are unwarranted. However, despite their advanced capacity for numerical representation, these unimodal series encoders intrinsically lack the synchronized visual context necessary to preempt abrupt meteorological shifts at the token level.

Building upon these deep learning architectures, large language models (LLMs) have recently been adapted for time-series forecasting. By leveraging frozen pretrained transformers, these approaches transfer self-attention mechanisms to sequential data with minimal task-specific adaptation [7]. Subsequent researchers have introduced diverse alignment strategies: TEST aligns time-series embeddings with text prototypes [30], TEMPO integrates seasonaltrend decomposition with prompt-based GPT adaptation [31], S<sup>2</sup>IP-LLM utilizes semantic-space anchors as prompts [32], and NNCL-TLLM constructs time-series-compatible text prototypes [33]. Notably, Time-LLM establishes a representative reprogramming recipe by mapping patched series into a frozen language backbone via cross-attention and prefix prompting [8].This specific reprogramming framework has already been successfully translated to distributed PV power forecasting [11, 12], reflecting a broader trend of

LLM adoption across renewable energy systems for tasks ranging from forecasting to control and fault diagnosis [10]. While several of these advanced LLM forecasters are technically multimodal, they strictly treat the auxiliary modality as text. For instance, GPT4MTS employs textual context as soft prompts alongside numerical patches [34]. A parallel line of work explores time-series foundation models, such as TimesFM [35] and Chronos [36], which are pretrained for zero-shot forecasting and have recently been extended to support multivariate targets and numerical covariates. However, these models remain series-native and do not accommodate sky images as input. Moreover, their public ecosystem of pretrained checkpoints and task-specific recipes is less mature than that of LLMs. Ultimately, both frozen-LLM pipelines and time-series foundation models consistently omit synchronized sky imagery, leaving a critical visual gap in existing forecasting architectures.

## 2.2. Sky-image and multimodal solar forecasting

Ground-based sky cameras and geostationary satellites capture critical visual context regarding cloud kinematics that purely numerical histories cannot encode [2, 3, 4]. The growing availability of public multimodal datasets now facilitates rigorous cross-site evaluation of these visual pipelines.

Purely vision-based forecasters map camera sequences directly to future irradiance or PV output. Early implementations relied on Convolutional Neural Networks (CNNs) processing single frames or short sequences [37, 38]. Subsequent advancements introduced CNN-LSTM architectures to capture extended temporal contexts [39], as well as spatiotemporal models like ECLIPSE, which explicitly track cloud motion to minimize forecast lag [13]. More recently, generative video models such as SkyGPT have been employed to synthesize future sky frames for probabilistic predictions [14].

A parallel line of research focuses on intra-network multimodal fusion, combining sky imagery with historical numerical series. For instance, SUNSET concatenates downsampled sky frames with lagged PV measurements within a specialized CNN [16], while follow-up studies have comprehensively evaluated various fusion operators for these heterogeneous inputs [17]. While these architectures successfully leverage visual motion as the primary physical driver of intra-hour variability, integrating them with an LLM-based framework is outside their design scope.

Extending this paradigm to a regional scale, satellite-based methods forecast nonlinear cloud dynamics from geostationary sequences, mapping the predicted states to PV output [40, 15], or directly nowcasting surface solar radiation [41]. Comprehensive frameworks like Omnivision further unify ground-based images, satellite observations, and numerical logs into a multi-view architecture [18].

Recently, foundation models have also been explored for this domain. For instance, PV-VLM encodes sky images using a pretrained vision-language model (VLM) and fuses these embeddings with a PatchTST-style temporal encoder via late cross-modal attention [19]. In this design, the VLM operates as a semantic feature extractor, and fusion occurs after independent unimodal representations are fully formed. In contrast, SolCloudLLM employs a lightweight CNN to encode sky frames, temporally aligns the resulting visual tokens with Time-LLM series patches, and applies bidirectional multimodal fusion prior to reprogramming the unified representation into a frozen language backbone. This architecture maintains a shared temporal index between modalities, ensuring cloud context is explicitly attached to the corresponding numerical look-back segments. Furthermore, rather than relying on plain concatenation which merely juxtaposes token streams without explicit cross-modal interaction, bidirectional multimodal fusion allows sky-image tokens to scale and shift series patches, and vice versa, before the modulated streams are concatenated, dimensionally reduced, and mapped.

## 3. Method

## 3.1. Problem formulation

Let $x _ { 1 : T } = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { T } ] \in \mathbb { R } ^ { T }$ denote a univariate historical sequence of photovoltaic (PV) power or global horizontal irradiance (GHI). Let $I _ { 1 : T } \in$ R<sup>T</sup> <sup>×C×H′×W′</sup> denote the strictly synchronized sky-image sequence, where $C { = } 3$ for RGB channels and each frame is resized to a spatial resolution of $H ^ { \prime } { \times } W ^ { \prime } = 6 4 { \times } 6 4$ . Our goal is to forecast the future PV or GHI trajectory directly from the joint historical observations $( x _ { 1 : T } , I _ { 1 : T } )$

Given a historical look-back window of length T and a target forecast horizon H, we adopt a symmetric setting where $T { = } H$ across varied horizons $H \in \{ 1 6 , 3 2 , 6 4 \}$ . The multi-step forecasting objective is formulated as:

$$
\hat { y } _ { T + 1 : T + H } = f _ { \boldsymbol { \theta } } ( x _ { 1 : T } , I _ { 1 : T } ) ,\tag{1}
$$

where $\hat { y } _ { T + 1 : T + H } \in \mathbb { R } ^ { H }$ represents the predicted numerical sequence, and $f _ { \theta }$ denotes the forecasting model parameterized by θ. Finally, to ensure stable optimization and prevent data leakage, the target variables are standardized using a scaler fitted exclusively on the training split.

![](images/bb204e1effb36750511c656d0a1ea70bb681567dd8524b01902a7ed9fb45d301.jpg)  
Figure 1: The overall architecture of SolCloudLLM. Historical solar time-series and aligned sky images are encoded into patch tokens and integrated via a bidirectional multimodal fusion module. The fused representations then undergo patch reprogramming, followed by a frozen LLM backbone and a linear projection head to generate the final forecast. The fire and snowflake icons denote trainable and frozen components, respectively.

## 3.2. Architecture: SolCloudLLM

Figure 1 illustrates the overall architecture of SolCloudLLM, which can be conceptually divided into three main stages. First, the dual modality encoders process the historical solar time-series and synchronized sky image sequences into temporally aligned patch embeddings (detailed in Section 3.4). Second, a Bidirectional Multimodal Fusion module explicitly models cross-modal interactions by mutually modulating the token streams (Section 3.5). Finally, the fused multimodal representation undergoes time-series reprogramming (Section 3.3) and is concatenated with prompt metadata, before being processed by a frozen language backbone to yield the final forecast. Importantly, the core LLM components remain strictly frozen to preserve data-eficient adaptation, as denoted by the fire and snowflake icons.

## 3.3. Preliminaries

We adopt the time-series reprogramming framework introduced in Time-LLM as the base framework for our work [8]. The normalized historical series is first segmented into overlapping patches of length P with stride S (configured as $P { = } 4$ and S=1 in our main experiments). These patches are linearly projected to yield a time-series token sequence $z ^ { \mathrm { t s } } \in \mathbb { R } ^ { N \times d }$ , where N denotes the number of patches and d is the embedding dimension.

To bridge the modality gap between the time series patches and the LLM, the reprogramming module aligns $z ^ { \mathrm { t s } }$ with the language embedding space via a multi-head cross-attention mechanism. Specifically, the time-series tokens act as queries, while a set of text prototypes derived from the frozen LLM’s vocabulary embedding matrix serve as keys and values. Following the standard Time-LLM protocol, statistical prompt tokens are constructed from the global numerical context and prepended to the aligned sequence as a Prompt-as-Prefix.

These integrated tokens are then processed by the frozen LLM backbone. Finally, to produce the multi-step prediction, the output of the hidden states of the LLM is flattened and passed through a linear projection layer to generate the final numerical forecast $\hat { y } \in \mathbb { R } ^ { H }$

## 3.4. Image encoding and temporal alignment

To extract spatial visual descriptors from the sky frames, we employ a lightweight CNN image encoder based on the architecture introduced in SUNSET [16], which has been proven efective for sky-image feature extraction. As an alternative, we further evaluate a heavier ViT-small encoder in our ablation studies (Section 4). Given the sequential image input $I _ { 1 : T }$ , the CNN encoder produces a sequence of high-dimensional feature maps.

Crucially, to synchronize the visual modality with the numerical time-series patches, these image features are temporally aggregated (e.g., via pooling over the corresponding patch windows of length P and stride S) to match the patch count N. Subsequently, a projection Multi-Layer Perceptron (MLP) maps these aggregated visual descriptors to the identical token dimension d utilized in the time-series path. This yields the visual token sequence $z ^ { \mathrm { i m g } } \in \mathbb { R } ^ { N \times d }$ , which maintains a strict temporal correspondence with the patch indices $z ^ { \mathrm { t s } }$ . Establishing this aligned, shared temporal index allows SolCloudLLM to perform precise token-level multimodal conditioning before the unified representation enters the reprogramming module.

## 3.5. Bidirectional multimodal fusion

A critical architectural decision is the integration point of visual features. If fusion occurs after reprogramming, visual cues fail to dynamically contextualize the numerical tokens prior to their mapping into the language space. SolCloudLLM therefore introduces a Bidirectional Multimodal Fusion module before reprogramming, operating directly on the time-series patch pathway rather than acting as a replacement backbone. To enable mutual crossmodal conditioning, this module employs a bidirectional afine modulation mechanism inspired by feature-wise linear modulation (FiLM) [20].

Given aligned tokens $z ^ { \mathrm { t s } }$ and $z ^ { \mathrm { i m g } }$ , two lightweight parameter networks, $\phi _ { t }$ and $\phi _ { v } ,$ produce afine modulation parameters:

$$
\begin{array} { r } { \left( g ^ { \mathrm { i m g } } , b ^ { \mathrm { i m g } } \right) = \operatorname { s p l i t } \left( \phi _ { t } ( z ^ { \mathrm { t s } } ) \right) , } \end{array}\tag{2}
$$

$$
\begin{array} { r } { \left( g ^ { \mathrm { t s } } , b ^ { \mathrm { t s } } \right) = \operatorname { s p l i t } \left( \phi _ { v } ( z ^ { \mathrm { i m g } } ) \right) . } \end{array}\tag{3}
$$

The two branches are then modulated in opposite directions:

$$
\tilde { z } ^ { \mathrm { i m g } } = g ^ { \mathrm { i m g } } \odot z ^ { \mathrm { i m g } } + b ^ { \mathrm { i m g } } ,\tag{4}
$$

$$
\tilde { z } ^ { \mathrm { t s } } = g ^ { \mathrm { t s } } \odot z ^ { \mathrm { t s } } + b ^ { \mathrm { t s } } .\tag{5}
$$

Following a non-linear processing step on each branch, a scaled residual connection is applied before fusion:

$$
\hat { z } ^ { \mathrm { i m g } } = \sigma ( \tilde { z } ^ { \mathrm { i m g } } ) + \alpha z ^ { \mathrm { i m g } } ,\tag{6}
$$

$$
\hat { z } ^ { \mathrm { t s } } = \sigma \big ( \tilde { z } ^ { \mathrm { t s } } \big ) + \alpha z ^ { \mathrm { t s } } ,\tag{7}
$$

where $\sigma ( \cdot )$ denotes a non-linear activation function, specifically the Gaussian Error Linear Unit (GELU) [42], and α is a small scaling factor (0.1 in our setup). The modulated streams are subsequently concatenated and dimensionally reduced back to d via a linear projection ψ:

$$
z ^ { \mathrm { f u s e } } = \psi \big ( [ \hat { z } ^ { \mathrm { t s } } ; \hat { z } ^ { \mathrm { i m g } } ] \big ) ,\tag{8}
$$

followed by Layer Normalization. This bidirectional pre-reprogramming design ensures that each modality provides contextual guidance to the other before the unified representation is projected into the frozen language space. We compare this against a standard concatenation baseline in Section 4.

## 3.6. Training objective

The models are optimized to minimize the mean squared error (MSE) over the forecast horizon:

$$
\mathcal { L } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \bigl ( \hat { y } _ { T + h } - y _ { T + h } \bigr ) ^ { 2 } .\tag{9}
$$

We use the Adam optimizer for up to 10 epochs, incorporating early stopping based on validation performance. All reported testing metrics are evaluated using the best validation checkpoint under this protocol.

## 4. Experiments

## 4.1. Datasets and baselines

We compare our method using two open-source solar datasets: SIRTA [21] and SKIPP’D [22].

SIRTA. SIRTA provides global horizontal irradiance (GHI) series with strictly synchronized sky images from Palaiseau, France [21]. The sampling interval is 2 min/step. The prediction horizons $H \in \{ 1 6 , 3 2 , 6 4 \}$ correspond to 32, 64, and 128 minutes, respectively. For this dataset, we employ a strict chronological split: data from 2017 to 2018 is utilized for training, January to June 2019 for validation, and July to December 2019 for testing. These three periods are contiguous and mutually exclusive.

SKIPP’D. SKIPP’D provides photovoltaic (PV) power series synchronized with ground-based sky images collected at the Stanford University campus in California [22]. The sampling interval is 1 min/step, horizons $H \in \{ 1 6 , 3 2 , 6 4 \}$ corresponding to 16, 32, and 64 minutes, respectively. We adopt the exact day-level hold-out configuration established in the original dataset paper, retaining a fixed 20-day test set. We then randomly sample 100 days for validation, with the remainder serving as training data. These subsets are strictly disjoint and uniformly mixed across 2017–2019. For this dataset, we report normalized MSE and MAE with sunny/cloudy stratification following the fixed day-level partition provided by the dataset authors.

Common protocols. Across both datasets, all input sky images are resized to a spatial resolution of $6 4 \times 6 4$ . All rolling sequences are strictly constrained within a single calendar day. Target variables are standardized using scalers fitted exclusively on the training splits. Finally, for few-shot experiments, we subsample only the training days, ensuring the validation and test sets remain completely intact for rigorous evaluation.

Baselines. To evaluate our proposed method, we benchmark it against a selection of prominent forecasting methods. Time-LLM [8] serves as the primary unimodal (time-series only) LLM reference, reprogramming patched numerical tokens into a frozen language backbone via cross-attention and prefix prompting. PatchTST [28] represents a state-of-the-art purely numerical Transformer utilizing channel-independent patching, while DLinear [29] provides a competitive linear baseline that decomposes the sequence into trend and seasonal components. Finally, the Smart Persistence Model (SPM) acts as our non-learning, physics-based benchmark. Unlike naive persistence, SPM extrapolates future values by holding the most recently observed clear-sky index constant and scaling it against the theoretical clear-sky profile [1].

## 4.2. Implementation details

The fused tokens $z ^ { \mathrm { f u s e } }$ are fed into the time-series reprogramming module, which maps them into the language embedding space via cross-attention over text prototypes derived from the LLM vocabulary. We employ a frozen GPT-2 backbone [43] truncated to $L { = } 6$ layers with a native hidden size of 768. The LLM output is subsequently flattened and passed through a linear projection head to yield the final H-step forecast.

For the pre-reprogramming architectures, the initial patch embedding token width is set to $d _ { \mathrm { m o d e l } } { = } 1 6$ , with a feed-forward width of 128. All models are trained using a batch size of 128 and a base learning rate of $1 0 ^ { - 4 }$ Specifically for SolCloudLLM, all parameters of the bidirectional multimodal fusion module utilize a 10× learning rate multiplier by default to ensure stable multimodal alignment.

## 4.3. Main results

Tables 1 and 2 present the overall normalized MSE and MAE for all evaluated methods across three prediction horizons. The best and second-best results in each column are highlighted in boldface and underlined, respectively. The bottom row quantifies the relative error reduction achieved by SolCloudLLM compared to the strongest baseline, where positive values indicate a performance improvement.

On the SIRTA dataset, SolCloudLLM consistently outperforms all baselines across every horizon. Specifically, when compared to the unimodal Time-LLM, SolCloudLLM yields substantial relative MSE reductions of 20.7%, 25.4%, and 24.8% across three horizons.

Table 1: SIRTA results (normalized MSE/MAE). Best per column in bold, second best underlined. Bottom row: relative gain of SolCloudLLM vs. the best baseline. H denotes the symmetric look-back and forecast length in time steps, where each step corresponds to 2 min.
<table><tr><td></td><td colspan="2">H=16</td><td colspan="2">H=32</td><td colspan="2">H=64</td></tr><tr><td>Method</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>SolCloudLLM(ours)</td><td>0.2200</td><td>0.2897</td><td>0.2557</td><td>0.3392</td><td>0.3688</td><td>0.4334</td></tr><tr><td>Time-LLM</td><td>0.2775</td><td>0.3500</td><td>0.3427</td><td>0.4212</td><td>0.4903</td><td>0.5231</td></tr><tr><td>PatchTST</td><td>0.3000</td><td>0.3652</td><td>0.3763</td><td>0.4448</td><td>0.5756</td><td>0.5759</td></tr><tr><td>DLinear</td><td>0.3303</td><td>0.4170</td><td>0.3969</td><td>0.4877</td><td>0.6288</td><td>0.6351</td></tr><tr><td>SPM</td><td>0.3580</td><td>0.3288</td><td>0.4014</td><td>0.3684</td><td>0.5484</td><td>0.4647</td></tr><tr><td>Gain vs. best baseline (%)</td><td>+20.7</td><td>+11.9</td><td>+25.4</td><td>+7.9</td><td>+24.8</td><td>+6.7</td></tr></table>

Similarly, on the SKIPP’D dataset, SolCloudLLM achieves the lowest MSE at all prediction intervals, demonstrating improvements over Time-LLM by 7.0% (H=16), 12.0% (H=32), and 14.5% (H=64). While absolute prediction errors naturally increase with the horizon length across both datasets, the unimodal methods degrade significantly faster than our proposed framework. This divergence confirms that the benefits of visual sky-image conditioning persist, and often amplify, when forecasts must anticipate prolonged, clouddriven ramp events.

## 4.4. Sunny/cloudy regime analysis

Table 3 stratifies the evaluation metrics from the SKIPP’D dataset into sunny and cloudy regimes according to the fixed day-level partition defined by the dataset authors, for all evaluated methods. As anticipated, forecasting errors on cloudy days dominate the overall MSE, reflecting the high residual variance driven by transient cloud dynamics.

Under cloudy conditions, SolCloudLLM consistently achieves the lowest MSE across all prediction horizons. When compared to the strongest unimodal baseline, Time-LLM, the relative performance gains of SolCloudLLM scale significantly with the horizon length: 3.5% at H=16, 10.8% at H=32, and 13.8% at H=64.

Conversely, absolute forecasting errors on sunny days are inherently minimal. Under these clear-sky conditions, SolCloudLLM remains highly competitive, with the physics-based SPM only marginally outperforming it at H=32.

Table 2: SKIPP’D results (normalized MSE/MAE). Best per column in bold, second best underlined. Bottom row: relative gain of SolCloudLLM vs. the best baseline. H denotes the symmetric look-back and forecast length in time steps, where each step corresponds to 1 min.
<table><tr><td></td><td colspan="2">H=16</td><td colspan="2">H=32</td><td colspan="2">H=64</td></tr><tr><td>Method</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>SolCloudLLM(ours)</td><td>0.1242</td><td>0.1560</td><td>0.1529</td><td>0.1954</td><td>0.2069</td><td>0.2607</td></tr><tr><td>Time-LLM</td><td>0.1335</td><td>0.1919</td><td>0.1737</td><td>0.2213</td><td>0.2421</td><td>0.2846</td></tr><tr><td>PatchTST</td><td>0.1499</td><td>0.2190</td><td>0.2143</td><td>0.2877</td><td>0.3713</td><td>0.3794</td></tr><tr><td>DLinear</td><td>0.1892</td><td>0.2946</td><td>0.2333</td><td>0.3512</td><td>0.3860</td><td>0.4947</td></tr><tr><td>SPM</td><td>0.1476</td><td>0.1572</td><td>0.1856</td><td>0.1948</td><td>0.2813</td><td>0.2661</td></tr><tr><td>Gain vs. best baseline (%)</td><td>+7.0</td><td>+0.8</td><td>+12.0</td><td>-0.3</td><td>+14.5</td><td>+2.0</td></tr></table>

Table 3: SKIPP’D sunny/cloudy MSE. Best per column in bold, second best underlined. H denotes the symmetric look-back and forecast length in time steps, where each step corresponds to 1 min.
<table><tr><td rowspan="2">Method</td><td colspan="2">H=16</td><td colspan="2">H=32</td><td colspan="2">H=64</td></tr><tr><td>Sunny</td><td>Cloudy</td><td>Sunny</td><td>Cloudy</td><td>Sunny</td><td>Cloudy</td></tr><tr><td>SolCloudLLM(ours)</td><td>0.00051</td><td>0.2594</td><td>0.00421</td><td>0.3154</td><td>0.00625</td><td>0.4182</td></tr><tr><td>Time-LLM</td><td>0.00987</td><td>0.2688</td><td>0.00984</td><td>0.3534</td><td>0.01167</td><td>0.4853</td></tr><tr><td>PatchTST</td><td>0.01684</td><td>0.2954</td><td>0.04287</td><td>0.4019</td><td>0.05916</td><td>0.7002</td></tr><tr><td>DLinear</td><td>0.06250</td><td>0.3277</td><td>0.10392</td><td>0.3749</td><td>0.23743</td><td>0.5425</td></tr><tr><td>SPM</td><td>0.00121</td><td>0.3076</td><td>0.00400</td><td>0.3842</td><td>0.00688</td><td>0.5705</td></tr></table>

Ultimately, the crucial operational takeaway is that the integration of sky imagery becomes increasingly vital when models must anticipate prolonged, cloud-driven ramp events.

## 4.5. Few-shot data eficiency

To evaluate sample eficiency, we subsample the training days to 10% and 5% while maintaining the full validation and test sets. Tables 4 and 5 report the overall MSE under these restricted budgets, with the corresponding 100% results provided in Tables 1 and 2. With the notable exception of H=16 on the SIRTA dataset, the deep-learning baselines (Time-LLM, PatchTST, and DLinear) perform worse than the physics-based SPM across all settings under limited training data, often by a substantial margin. This highlights a severe degradation in conventional deep forecasters to forecast longer horizons when trained on limited data. Notably, although Time-LLM relies on a frozen LLM backbone, its performance still declines, demonstrating that pretrained sequence knowledge alone is insuficient under severe data scarcity. Despite the broad collapse of baseline methods, SolCloudLLM retains the best performance in most of the few-shot configurations. The sole exception occurs on the SKIPP’D dataset at H=16 (5% data), where SPM is ahead (0.1476 vs. 0.1559). This pattern confirms that our model’s robustness stems directly from the multimodal fusion rather than the time-series reprogramming alone. We attribute this resilience to two coupled architectural choices: (i) retaining a frozen LLM backbone to restrict task-specific training parameter growth, and (ii) injecting visual context via our lightweight bidirectional multimodal fusion module.

Table 4: SIRTA few-shot overall MSE (day-level train subsample, 10%/5% only). Full-data (100%) results are in Table 1. Best per column in bold, second best underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">H=16</td><td colspan="2">H=32</td><td colspan="2">H=64</td></tr><tr><td>10%</td><td>5%</td><td>10%</td><td>5%</td><td>10%</td><td>5%</td></tr><tr><td>SolCloudLLM(ours)</td><td>0.2636</td><td>0.3047</td><td>0.2943</td><td>0.3283</td><td>0.4423</td><td>0.4952</td></tr><tr><td>Time-LLM</td><td>0.3061</td><td>0.3099</td><td>0.4138</td><td>0.4163</td><td>0.7502</td><td>0.7527</td></tr><tr><td>PatchTST</td><td>0.3349</td><td>0.3381</td><td>0.4573</td><td>0.4681</td><td>0.7778</td><td>0.8163</td></tr><tr><td>DLinear</td><td>0.3472</td><td>0.3485</td><td>0.4241</td><td>0.4275</td><td>0.7477</td><td>0.7616</td></tr><tr><td>SPM</td><td>0.3580</td><td>0.3580</td><td>0.4014</td><td>0.4014</td><td>0.5484</td><td>0.5484</td></tr></table>

Table 5: SKIPP’D few-shot overall MSE (day-level train subsample, 10%/5% only). Fulldata (100%) results are in Table 2. Best per column in bold, second best underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">H=16</td><td colspan="2">H=32</td><td colspan="2">H=64</td></tr><tr><td>10%</td><td>5%</td><td>10%</td><td>5%</td><td>10%</td><td>5%</td></tr><tr><td>SolCloudLLM(ours)</td><td>0.1296</td><td>0.1559</td><td>0.1639</td><td>0.1719</td><td>0.2269</td><td>0.2693</td></tr><tr><td>Time-LLM</td><td>0.1575</td><td>0.1633</td><td>0.2392</td><td>0.2414</td><td>0.4570</td><td>0.4767</td></tr><tr><td>PatchTST</td><td>0.1654</td><td>0.1672</td><td>0.2523</td><td>0.2594</td><td>0.4631</td><td>0.4920</td></tr><tr><td>DLinear</td><td>0.1992</td><td>0.2004</td><td>0.2484</td><td>0.2501</td><td>0.4627</td><td>0.4727</td></tr><tr><td>SPM</td><td>0.1476</td><td>0.1476</td><td>0.1856</td><td>0.1856</td><td>0.2813</td><td>0.2813</td></tr></table>

Table 6: Fusion ablation (overall normalized MSE). SolCloudLLM uses bidirectional multimodal fusion; Concat replaces only the fusion operator.
<table><tr><td rowspan="2">Method</td><td colspan="3">SIRTA</td><td colspan="3">SKIPP&#x27;D</td></tr><tr><td>H=16</td><td>H=32</td><td>H=64</td><td>H=16</td><td>H=32</td><td>H=64</td></tr><tr><td>SolCloudLLM</td><td>0.2200</td><td>0.2557</td><td>0.3688</td><td>0.1242</td><td>0.1529</td><td>0.2069</td></tr><tr><td>Concat</td><td>0.2339</td><td>0.2672</td><td>0.4017</td><td>0.1287</td><td>0.1855</td><td>0.1931</td></tr></table>

## 4.6. Fusion ablation

Table 6 compares the full SolCloudLLM bidirectional multimodal fusion architecture against a standard concatenation (Concat) baseline under the established setting across $H \in \{ 1 6 , 3 2 , 6 4 \}$

On the SIRTA dataset, SolCloudLLMdemonstrates superior performance over the Concat baseline at all three horizons. For SKIPP’D, our approach maintains its advantage at H=16 and H=32, while Concat achieves a marginally lower error at the longest horizon of H=64.

## 4.7. Image encoder ablation

To evaluate the impact of the visual backbone, we substitute the default CNN encoder with a ViT-small [44] architecture, while keeping the bidirectional multimodal fusion module unchanged. Table 7 reports the overall normalized MSE on both datasets across $H \in \{ 1 6 , 3 2 , 6 4 \}$ . On the SIRTA dataset, the CNN encoder consistently outperforms the ViT across all horizons, exhibiting relative MSE advantages of 7.3% (H=16), 11.6% (H=32), and 4.7% (H=64). Conversely, on the SKIPP’D dataset, the ViT yields slightly worse performance at H=16 (+4.2%) and H=32 (+8.8%), though it performs marginally better at the longest horizon, H=64 (−3.5%).

Given that the CNN encoder outperforms the ViT in the majority of scenarios and its lightweight architecture inherently benefits few-shot data eficiency, we conclude that the CNN is the superior choice for our primary experiments.

## 5. Conclusion

In this work, we introduced SolCloudLLM, a multimodal forecasting framework that combines contemporaneous sky image features into an LLMbased time-series forecaster for solar forecasting. To achieve this, we propose a lightweight bidirectional multimodal fusion module that enables explicit cross-modal interaction through token-wise afine modulation. This design efectively conditions the numerical tokens before they enter the frozen LLM backbone, allowing the visual and numerical modalities to mutually modulate one another and fostering a richer, deeply integrated cross-modal representation. Extensive experiments on two open-source solar datasets demonstrate that SolCloudLLM consistently outperforms the best baseline across all tested horizons (H ∈ {16, 32, 64}), achieving peak relative MSE reductions of 25.4% on SIRTA (H=32) and 14.5% on SKIPP’D (H=64).

Table 7: Image encoder ablation (overall normalized MSE). CNN is the default Sol-CloudLLM image encoder. ViT-small replaces only the image backbone.
<table><tr><td rowspan="2">Encoder</td><td colspan="3">SIRTA</td><td colspan="3">SKIPP&#x27;D</td></tr><tr><td>H=16</td><td>H=32</td><td>H=64</td><td>H=16</td><td>H=32</td><td>H=64</td></tr><tr><td>CNN (default)</td><td>0.2200</td><td>0.2557</td><td>0.3688</td><td>0.1242</td><td>0.1529</td><td>0.2069</td></tr><tr><td>ViT-small</td><td>0.2360</td><td>0.2853</td><td>0.3860</td><td>0.1294</td><td>0.1664</td><td>0.1996</td></tr></table>

Our proposed framework excels in severe data-scarcity scenarios and at longer forecast horizons under cloudy conditions, where unimodal approaches struggle. In few-shot scenarios using only 5% to 10% of available training days, conventional deep-learning baselines frequently underperform the physics-based SPM. In contrast, SolCloudLLM demonstrates remarkable resilience, leveraging the combination of a frozen LLM backbone and compact cross-modal modulation to maintain superior forecasting performance. Our ablation studies further validate the advantages of the proposed bidirectional multimodal fusion module in efectively integrating these disparate modalities in most cases.

As a primary limitation, we acknowledge that ground-based sky cameras inherently provide only localized and short-term cues regarding cloud dynamics. Consequently, future work will focus on developing a more comprehensive multimodal framework that simultaneously incorporates ground-based sky images, large-scale satellite imagery, and text-based weather forecasts. By synthesizing these diverse data sources, it is possible to deliver robust solar forecasting capabilities across both short and extended prediction horizons.

## References

[1] H. T. C. Pedro, C. F. M. Coimbra, Assessment of forecasting techniques for solar power production with no exogenous inputs, Solar Energy 86 (7) (2012) 2017–2028. doi:10.1016/j.solener.2012.04.004. URL https://doi.org/10.1016/j.solener.2012.04.004

[2] F. Lin, Y. Zhang, J. Wang, Recent advances in intra-hour solar forecasting: A review of ground-based sky image methods, International Journal of Forecasting 39 (1) (2023) 244–265. doi:10.1016/j.ijforecast.2021. 11.002. URL https://doi.org/10.1016/j.ijforecast.2021.11.002

[3] Q. Paletta, G. Terrén-Serrano, Y. Nie, B. Li, J. Bieker, W. Zhang, L. Dubus, S. Dev, C. Feng, Advances in solar forecasting: Computer vision with deep learning, Advances in Applied Energy 11 (2023) 100150. doi:10.1016/j.adapen.2023.100150. URL https://doi.org/10.1016/j.adapen.2023.100150

[4] Y. Nie, X. Li, Q. Paletta, M. Aragon, A. Scott, A. Brandt, Opensource sky image datasets for solar forecasting with deep learning: A comprehensive survey, Renewable and Sustainable Energy Reviews 189 (2024) 113977. doi:10.1016/j.rser.2023.113977. URL https://doi.org/10.1016/j.rser.2023.113977

[5] Y. Nie, Q. Paletta, A. Scott, L. M. Pomares, G. Arbod, S. Sgouridis, J. Lasenby, A. Brandt, Sky image-based solar forecasting using deep learning with heterogeneous multi-location data: Dataset fusion versus transfer learning, Applied Energy 369 (2024) 123467. doi:10.1016/j. apenergy.2024.123467. URL https://doi.org/10.1016/j.apenergy.2024.123467

[6] Q. Paletta, Y. Nie, Y.-M. Saint-Drenan, B. Le Saux, Improving cross-site generalisability of vision-based solar forecasting models with physicsinformed transfer learning, Energy Conversion and Management 309 (2024) 118398. doi:10.1016/j.enconman.2024.118398. URL https://doi.org/10.1016/j.enconman.2024.118398

[7] T. Zhou, P. Niu, L. Sun, R. Jin, et al., One fits all: Power general time series analysis by pretrained lm, Advances in neural information processing systems 36 (2023) 43322–43355.

[8] M. Jin, S. Wang, L. Ma, Z. Chu, J. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan, et al., Time-llm: Time series forecasting by reprogramming large language models, in: International conference on learning representations, Vol. 2024, 2024, pp. 23857–23880.

[9] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, M. Long, Timesnet: Temporal 2d-variation modeling for general time series analysis, arXiv preprint arXiv:2210.02186 (2022).

[10] N. Chandana, S. N. Rao, M. Kumaraswamy, A. Pallakonda, R. D. A. Raj, Large language models in renewable energy systems: A comprehensive review of forecasting, control, policy, and fault diagnosis, Next Energy 11 (2026) 100586. doi:10.1016/j.nxener.2026.100586. URL https://doi.org/10.1016/j.nxener.2026.100586

[11] H. Lin, M. Yu, A novel distributed pv power forecasting approach based on time-llm, in: 2025 IEEE 4th International Conference on Industrial Electronics for Sustainable Energy Systems (IESES), IEEE, 2025, pp. 13–17.

[12] M. Fan, C. Lv, H. Fan, M. Li, L. Yang, Z. Zhang, S. Wang, X. Tan, Distributed photovoltaic power prediction based on solar-llm, in: Ninth International Conference on Energy System, Electricity, and Power (ESEP 2024), Vol. 13632, SPIE, 2025, pp. 350–357.

[13] Q. Paletta, A. Hu, G. Arbod, J. Lasenby, ECLIPSE: Envisioning CLoud induced perturbations in solar energy, Applied Energy 326 (2022) 119924. doi:10.1016/j.apenergy.2022.119924. URL https://doi.org/10.1016/j.apenergy.2022.119924

[14] Y. Nie, E. Zelikman, A. Scott, Q. Paletta, A. Brandt, SkyGPT: Probabilistic ultra-short-term solar forecasting using synthetic sky images from physics-constrained VideoGPT, Advances in Applied Energy 14 (2024) 100172. doi:10.1016/j.adapen.2024.100172. URL https://doi.org/10.1016/j.adapen.2024.100172

[15] M. Perera, J. De Hoog, K. Bandara, H. Weeratunge, S. Halgamuge, Distributed solar generation forecasting using attention-based deep neural networks for cloud movement prediction, Energy 361 (2026) 141985.

doi:10.1016/j.energy.2026.141985. URL https://doi.org/10.1016/j.energy.2026.141985

[16] Y. Sun, V. Venugopal, A. R. Brandt, Short-term solar power forecast with deep learning: Exploring optimal input and output configuration, Solar Energy 188 (2019) 730–741. doi:10.1016/j.solener.2019.06.041. URL https://doi.org/10.1016/j.solener.2019.06.041

[17] V. Venugopal, Y. Sun, A. R. Brandt, Short-term solar PV forecasting using computer vision: The search for optimal CNN architectures for incorporating sky images and PV generation history, Journal of Renewable and Sustainable Energy 11 (6) (2019) 066102. doi:10.1063/1.5122796. URL https://doi.org/10.1063/1.5122796

[18] Q. Paletta, G. Arbod, J. Lasenby, Omnivision forecasting: Combining satellite and sky images for improved deterministic and probabilistic intra-hour solar energy predictions, Applied Energy 336 (2023) 120818. doi:10.1016/j.apenergy.2023.120818. URL https://doi.org/10.1016/j.apenergy.2023.120818

[19] H. Lin, M. Yu, PV-VLM: A multimodal vision-language approach incorporating sky images for intra-hour photovoltaic power forecasting, arXiv preprint arXiv:2504.13624 (2025). URL https://arxiv.org/abs/2504.13624

[20] E. Perez, F. Strub, H. De Vries, V. Dumoulin, A. Courville, FiLM: Visual reasoning with a general conditioning layer, Proceedings of the AAAI Conference on Artificial Intelligence 32 (1) (2018). doi:10.1609/aaai. v32i1.11671. URL https://doi.org/10.1609/aaai.v32i1.11671

[21] M. Haefelin, L. Barthès, O. Bock, C. Boitel, S. Bony, D. Bouniol, H. Chepfer, M. Chiriaco, J. Cuesta, J. Delanoë, P. Drobinski, J.-L. Dufresne, C. Flamant, M. Grall, A. Hodzic, F. Hourdin, F. Lapouge, Y. Lemaître, A. Mathieu, Y. Morille, C. Naud, V. Noël, W. O’Hirok, J. Pelon, C. Pietras, A. Protat, B. Romand, G. Scialom, R. Vautard, SIRTA, a ground-based atmospheric observatory for cloud and aerosol research, Annales Geophysicae 23 (2) (2005) 253–275. doi:10.5194/ angeo-23-253-2005. URL https://doi.org/10.5194/angeo-23-253-2005

[22] Y. Nie, X. Li, A. Scott, Y. Sun, V. Venugopal, A. Brandt, Skipp’d: A sky images and photovoltaic power generation dataset for short-term solar forecasting, Solar Energy 255 (2023) 171–179.

[23] R. H. Inman, H. T. Pedro, C. F. Coimbra, Solar forecasting methods for renewable energy integration, Progress in energy and combustion science 39 (6) (2013) 535–576.

[24] S. Hochreiter, J. Schmidhuber, Long short-term memory, Neural Computation 9 (8) (1997) 1735–1780. doi:10.1162/neco.1997.9.8.1735. URL https://doi.org/10.1162/neco.1997.9.8.1735

[25] S. Bai, J. Z. Kolter, V. Koltun, An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, arXiv preprint arXiv:1803.01271 (2018). URL https://arxiv.org/abs/1803.01271

[26] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, W. Zhang, Informer: Beyond eficient transformer for long sequence time-series forecasting, Proceedings of the AAAI Conference on Artificial Intelligence 35 (12) (2021) 11106–11115. doi:10.1609/aaai.v35i12.17325. URL https://doi.org/10.1609/aaai.v35i12.17325

[27] H. Wu, J. Xu, J. Wang, M. Long, Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting, Advances in neural information processing systems 34 (2021) 22419–22430.

[28] Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam, A time series is worth 64 words: Long-term forecasting with transformers, arXiv preprint arXiv:2211.14730 (2022).

[29] A. Zeng, M. Chen, L. Zhang, Q. Xu, Are transformers efective for time series forecasting?, Proceedings of the AAAI Conference on Artificial Intelligence 37 (9) (2023) 11121–11128. doi:10.1609/aaai.v37i9.26317. URL https://doi.org/10.1609/aaai.v37i9.26317

[30] C. Sun, H. Li, Y. Li, S. Hong, Test: Text prototype aligned embedding to activate llm’s ability for time series, in: International Conference on Learning Representations, Vol. 2024, 2024, pp. 37854–37881.

[31] D. Cao, F. Jia, S. Arik, T. Pfister, Y. Zheng, W. Ye, Y. Liu, Tempo: Prompt-based generative pre-trained transformer for time series forecasting, in: International Conference on Learning Representations, Vol. 2024, 2024, pp. 18546–18578.

[32] Z. Pan, Y. Jiang, S. Garg, A. Schneider, Y. Nevmyvaka, D. Song, S<sup>2</sup>IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting, in: Proceedings of the 41st International Conference on Machine Learning, Vol. 235 of Proceedings of Machine Learning Research, PMLR, 2024, pp. 39135–39153. URL https://proceedings.mlr.press/v235/pan24c.html

[33] J. Bogahawatte, S. Seneviratne, M. Perera, S. Halgamuge, Rethinking time series forecasting with llms via nearest neighbor contrastive learning, arXiv preprint arXiv:2412.04806 (2024).

[34] F. Jia, K. Wang, Y. Zheng, D. Cao, Y. Liu, GPT4MTS: Prompt-based large language model for multimodal time-series forecasting, Proceedings of the AAAI Conference on Artificial Intelligence 38 (21) (2024) 23343– 23351. doi:10.1609/aaai.v38i21.30383. URL https://doi.org/10.1609/aaai.v38i21.30383

[35] A. Das, W. Kong, R. Sen, Y. Zhou, A decoder-only foundation model for time-series forecasting, arXiv preprint arXiv:2310.10688 (2023).

[36] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor, et al., Chronos: Learning the language of time series, arXiv preprint arXiv:2403.07815 (2024).

[37] Q. Paletta, J. Lasenby, Convolutional neural networks applied to sky images for short-term solar irradiance forecasting, arXiv preprint arXiv:2005.11246Accepted for EU PVSEC 2020. (2020). URL https://arxiv.org/abs/2005.11246

[38] C. Feng, J. Zhang, W. Zhang, B.-M. Hodge, Convolutional neural networks for intra-hour solar forecasting based on sky image sequences, Applied Energy 310 (2022) 118438. doi:10.1016/j.apenergy.2021. 118438. URL https://doi.org/10.1016/j.apenergy.2021.118438

[39] T. A. Siddiqui, S. Bharadwaj, S. Kalyanaraman, A deep learning approach to solar-irradiance forecasting in sky-videos, in: 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), 2019, pp. 2166– 2174. doi:10.1109/WACV.2019.00234. URL https://doi.org/10.1109/WACV.2019.00234

[40] Z. Si, M. Yang, Y. Yu, T. Ding, Photovoltaic power forecast based on satellite images considering efects of solar position, Applied Energy 302 (2022) 117514. doi:10.1016/j.apenergy.2021.117514. URL https://doi.org/10.1016/j.apenergy.2021.117514

[41] Y. Cui, P. Wang, J. Fokke Meirink, N. Ntantis, J. S. Wijnands, Solar radiation nowcasting based on geostationary satellite images and deep learning models, Solar Energy 282 (2024) 112866. doi:10.1016/j. solener.2024.112866. URL https://doi.org/10.1016/j.solener.2024.112866

[42] D. Hendrycks, K. Gimpel, Gaussian error linear units (gelus), arXiv preprint arXiv:1606.08415 (2016).

[43] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al., Language models are unsupervised multitask learners, OpenAI blog 1 (8) (2019) 9.

[44] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations, 2021.