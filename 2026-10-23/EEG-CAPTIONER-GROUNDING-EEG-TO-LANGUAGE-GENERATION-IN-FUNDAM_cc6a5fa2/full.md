# EEG-CAPTIONER: GROUNDING EEG-TO-LANGUAGE GENERATION IN FUNDAMENTALSIGNAL PROPERTIES

Leyuan Jin<sup>1</sup>, Jiahao Mei<sup>1</sup>, Yixuan Yuan<sup>2</sup>, Shimeng Li<sup>1</sup>, Mengyue Wu<sup>1</sup>, Yue Ding<sup>2</sup>

<sup>1</sup>X-LANCE Lab, Shanghai Jiao Tong University, Shanghai, China

<sup>2</sup>Shanghai Mental Health Center, Shanghai Jiao Tong University School of Medicine, Shanghai, China

## ABSTRACT

Electroencephalography (EEG)-to-language models offer a promising approach for translating complex neural signals into humanreadable descriptions. However, generating plausible language does not necessarily demonstrate that a model has learned to interpret the underlying EEG signal. A direct way to assess such interpretation is to probe objectively measurable properties of the signal itself. We therefore construct an EEG-language dataset of 160,781 segments annotated with signal-level question answering (QA) pairs, where each question targets one of five property families and requires a quantitative comparison involving spectral, spatial, temporal, or lateral signal properties. Each segment is also paired with a feature-grounded EEG caption. We further propose a Dual-Domain EEG Encoder that jointly models temporal waveforms and spectral-spatial representations, and introduce EEG-Captioner, a unified EEG-to-language framework trained with both signal-level QA and EEG caption supervision. Experiments show that the proposed encoder achieves the highest overall macro-averaged balanced accuracy (Macro-BA) among the compared EEG encoders, while EEG-Captioner achieves a Macro-BA of 88.81% on signal-level QA and improves caption factual consistency from 82.06% to 83.46% over caption-only language adaptation, demonstrating the complementary benefits of signal-level QA and caption supervision.

Index Terms— Electroencephalography (EEG), EEG-to-language, large language models, EEG representation learning, EEG caption generation

## 1. INTRODUCTION

Electroencephalography (EEG) provides a non-invasive and temporally precise measurement of neural activity. Interpreting EEG requires reasoning jointly about multiple signal characteristics, including temporal dynamics, spectral patterns, spatial distributions, and hemispheric asymmetries. Large language models (LLMs) offer a promising way to translate such complex signals into humanreadable descriptions. However, generating plausible text does not necessarily imply that an EEG-to-language model has captured the underlying signal characteristics. Faithful EEG descriptions should therefore be grounded in measurable signal characteristics.

Recent EEG-to-language studies have made progress toward such grounding. E<sup>2</sup>-LLM [1] integrates pretrained EEG representations with an LLM for affective reasoning, but does not explicitly supervise the underlying signal properties. CerebraGloss [2] instruction-tunes a vision-language model for fine-grained clinical EEG interpretation, focusing primarily on clinically relevant EEG characteristics. NeuroNarrator [3] incorporates spectro-spatial and temporal modeling and uses LLM-based scoring to assess signal properties in generated captions. Despite these advances, existing approaches still lack explicit and comprehensive supervision and direct evaluation of fundamental EEG properties, making it difficult to determine whether generated descriptions are systematically grounded in the underlying temporal, spectral, spatial, and lateral characteristics. This need for explicit signal-level supervision also raises a question of whether the EEG representations used by language-based models preserve these signal properties. EEG representation learning has progressed from compact convolutional and Transformer architectures, such as EEGNet [4] and SleepTransformer [5], to large-scale pretrained models including BIOT [6], EEGPT [7], LaBraM [8], and CBraMod [9]. Although strong on downstream tasks, these models are not primarily designed to preserve the EEG properties needed for grounded language generation.

To address these limitations, we first construct an EEG-language dataset that provides explicit supervision and evaluation of fundamental EEG properties through signal-level question answering (QA) and feature-grounded captions. The QA annotations cover five property families across temporal, spectral, spatial, and lateral dimensions. To better preserve the signal information required by these properties, we further propose a Dual-Domain EEG Encoder that jointly models raw temporal dynamics and spectral-spatial structure. Instead of serializing precomputed EEG features as text, the encoder maps raw EEG into compact learned tokens, enabling more efficient and scalable EEG-to-language generation. Building on the dataset and encoder, we introduce EEG-Captioner, a unified framework jointly trained with signal-level QA and caption supervision for grounded EEG caption generation.

Experiments show that the proposed Dual-Domain EEG Encoder achieves higher macro-averaged balanced accuracy (Macro-BA) than the compared encoders on signal-level QA, while joint QA-caption training further improves both QA performance and grounded caption generation over the single-supervision variants. The main contributions are:

• We construct an EEG-language dataset that pairs EEG segments with signal-level QA and feature-grounded captions, enabling explicit supervision and direct evaluation of fundamental EEG properties alongside caption generation.

• We propose a Dual-Domain EEG Encoder that jointly models temporal waveforms and spectral-spatial representations, achieving stronger performance than existing EEG encoders on the proposed signal-level QA tasks.

• We introduce EEG-Captioner, a unified EEG-to-language framework that jointly learns from signal-level QA and caption supervision, improving both fundamental EEG understanding and grounded caption generation.

Table 1. Composition of the proposed EEG-language dataset. Each EEG segment generates one caption and 25 candidate relational questions across five task categories, with five questions per category. The reported relational counts are the numbers retained after numerical validity and signal-quality filtering.
<table><tr><td>Component</td><td>Task</td><td>Question</td><td>Answer Space</td><td>Valid Count</td></tr><tr><td rowspan="6">Signal-level QA</td><td>Dominant Band</td><td>Which frequency band has the highest power in a given time interval and spatial region?</td><td>Delta, Theta, Alpha, Beta, Gamma</td><td>652,345</td></tr><tr><td>Strongest Region</td><td>Which brain region has the highest power in a given frequency band and time interval?</td><td>Frontal, Central, Parietal, Occipital, Temporal</td><td>6,879</td></tr><tr><td>Peak Period</td><td>In which time interval—Early, Middle, or Last—does a given EEG feature reach its highest value?</td><td>Early, Middle, Last</td><td>311,861</td></tr><tr><td>Temporal Pattern</td><td>How does a given EEG feature change across the Early, Middle, and Last intervals?</td><td>Increasing, Decreasing, Increase-then-Decrease,</td><td>491,189</td></tr><tr><td>Lateral Dominance</td><td>Which hemisphere shows stronger activity in</td><td>Decrease-then-Increase, Stable Left, Right, Approximately</td><td>784,302</td></tr><tr><td>Free-form Caption</td><td>corresponding channels or brain regions? Please describe the EEG signal in one coherent English caption.</td><td>Symmetric In the early interval, the EEG is dominated by ,</td><td>160,781</td></tr></table>

## 2. METHOD

The EEG-language dataset construction pipeline is illustrated in Fig. 1, while the overall framework of EEG-Captioner is shown in Fig. 2. We first describe the construction of the EEG-language dataset, followed by the Dual-Domain EEG Encoder and the joint EEG-to-language training framework.

## 2.1. EEG-Language Dataset Construction

![](images/aeb5c6c4139a8d07823da60fdcba52dc15dd28dcfdc262627f63b5061cf4da58.jpg)  
Fig. 1. Illustration of the EEG-Language dataset construction pipeline.

We construct the EEG-language dataset from the Temple University EEG Corpus (TUEG) v2.0.2 [10]. EEG recordings are processed using a standardized pipeline consisting of channel harmonization, filtering and resampling, fixed-length segmentation, and multi-level quality control.

For each valid EEG segment, we compute a deterministic numerical feature summary describing its temporal, spectral, spatial, and lateral characteristics. Specifically, each segment is divided into Early, Middle, and Last intervals, from which we extract channellevel band power, region-of-interest (ROI)-level band power, sourcenetwork power, inter-hemispheric asymmetry, time-domain features, and spectral-domain features.

Based on these features, we construct signal-level QA for five EEG property families: dominant band, strongest region, peak period, temporal pattern, and lateral dominance. Rather than regressing absolute feature values, which can vary substantially across recordings and subjects, we formulate the tasks as relational comparisons that capture interpretable structure across frequency bands, spatial regions, temporal intervals, and hemispheres. Each property family contains five predefined feature conditions, yielding 25 candidate QA pairs for each segment. Ground-truth answers are computed directly from full-precision numerical features. We further apply taskspecific validity and ambiguity criteria to remove unreliable annotations, including invalid feature values, insufficient separation between competing maxima, and inconsistent temporal patterns. The strongest region task retains fewer samples because regional comparisons require all constituent channels to satisfy the validity criteria.

Along with signal-level QA pairs, we generate a featuregrounded caption for each segment. The numerical feature summary is provided to DeepSeek-V4-Flash [11], which is instructed to describe the observed temporal, spectral, spatial, and lateral characteristics while avoiding clinical or cognitive inference. Each sample therefore contains the raw EEG segment, the retained signal-level QA pairs, and an EEG caption, as summarized in Table 1.

## 2.2. Dual-Domain EEG Encoder

We propose a Dual-Domain EEG Encoder to jointly represent information from the raw temporal waveform and its spectral-spatial structure. The encoder contains two parallel branches whose representations are subsequently aligned and fused.

In the temporal branch, each channel waveform is divided into short patches and projected into temporal tokens, allowing the encoder to model local waveform dynamics directly from the raw signal. In parallel, the spectral-spatial branch divides the EEG segment into Early, Middle, and Last intervals and computes channelwise power spectral density (PSD) using Welch’s method [12]. Within each interval, channel self-attention [13] models dependencies across electrodes from the resulting spectral representations.

![](images/bdf0d438e3b496e1d9afac05ba07de39362909daf4be575418aad34e9165b542.jpg)  
Fig. 2. Overview of the proposed Dual-Domain EEG Encoder and EEG-to-LLM framework for relational QA and caption generation; representative outputs are shown on the right.

The outputs of the two branches are then aligned and fused to combine temporal dynamics with spectral-spatial context. Channelwise attention pooling followed by a Global Transformer further aggregates information across channels and temporal positions, producing the final EEG tokens used by downstream language tasks.

## 2.3. EEG-Captioner

EEG-Captioner connects the Dual-Domain EEG Encoder to an LLM through a learnable projector. Given an EEG segment, the encoder produces a sequence of EEG tokens, which are mapped by the projector into the LLM embedding space and provided as multimodal context for language generation.

We jointly train the framework using the two annotation types provided by the EEG-language dataset. For signal-level QA, the model receives a property-specific question and generates the corresponding answer. For caption generation, the model generates a natural-language description of the signal characteristics. Both tasks share the same EEG encoder, projector, and LLM, while taskspecific instructions specify the expected output format. The model is jointly optimized on both types of training examples.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

Dataset. All experiments are conducted on the proposed EEGlanguage dataset. The 160,781 EEG segments are split into 128,625 / 16,078 / 16,078 training, validation, and test samples in a filenamedisjoint manner, and the same partitions are used for all experiments.

Evaluation metrics. For fundamental signal understanding, we report balanced accuracy (BA) for each relation family and Macro-BA, computed as their unweighted average. For caption generation, we report BLEU-4 [14] and ROUGE-L [15] to measure lexical overlap with reference captions. To assess signal grounding beyond lexical similarity, we further evaluate factual consistency using GPT-5.6 Sol through the ChatGPT web interface with High reasoning effort. The evaluator compares each generated caption with a feature summary derived from the corresponding EEG segment and assesses four signal-level aspects: dominant spectral pattern, temporal trend, spectral features, and spatial distribution, producing an overall factual consistency score from 0% to 100%.

Implementation details. For the classifier stage, all encoders are trained with a single-layer classifier for 30,000 steps using AdamW, cosine decay, 1,000 warmup steps, and weight decay 0.01. CBraMod and LaBraM are initialized from pretrained weights and optimized with an encoder learning rate of $\mathrm { \hat { 3 } } \times 1 0 ^ { - 5 }$ , whereas the Dual-Domain encoder is trained from scratch with a learning rate of $3 \times 1 0 ^ { - 4 }$ ; all classifier heads use a learning rate of $3 \times \bar { 1 } 0 ^ { - 4 }$ Effective batch sizes are 16 for CBraMod/LaBraM and 48 for Dual-Domain. For the encoder-to-Qwen stage, the encoders from the classifier stage are frozen and connected to frozen Qwen2.5-Omni-7B [16] through a newly initialized projector. Low-Rank Adaptation (LoRA) [17] is applied to the Q/K/V/O projections of the language model $( r = 8 , \alpha = 3 2$ , dropout=0.05). The projector and LoRA parameters are trained for 30,000 steps using AdamW with a learning rate of $1 \times 1 0 ^ { - 4 }$ , weight decay 0.01, and an effective batch size of 4 on two GPUs. For EEG-Captioner, all variants use the same frozen Dual-Domain EEG encoder obtained from the classifier stage, thereby isolating differences in the language-adaptation strategy from differences in EEG-encoder pretraining. For the caption-only adaptation, the projector and LoRA parameters are re-initialized and trained for 15,000 steps using only caption supervision at a learning rate of $5 \times 1 0 ^ { - 5 }$ . Validation performance had converged by the end of the 15,000-step training. EEG-Captioner instead initializes the projector and LoRA parameters from the signal-level QA checkpoint and continues training for three consecutive 15,000-step stages with caption fractions of 0.25, 0.75, and 0.90, respectively. All captioning stages use AdamW with weight decay 0.01, an effective batch size of 4, and seed 42.

## 3.2. Evaluation Protocols

Encoder comparison. We compare the proposed Dual-Domain EEG Encoder with LaBraM and CBraMod, two representative stateof-the-art EEG encoders, at both representation and language levels.

Table 2. Comparison of EEG encoders on five relational tasks and EEG-input sanity checks. In the Classifier stage, each EEG encoder is trained jointly with a relational classification head. In the Qwen stage, the trained encoder is frozen, and the projector and LoRA adapters are trained for relational instruction tuning. All values are balanced accuracy (%), and Macro-BA is the unweighted mean across the five task families.
<table><tr><td>EEG Encoder</td><td>Stage</td><td>Dominant Band(%)</td><td>Strongest Region(%)</td><td>Peak Period(%)</td><td>Temporal Pattern(%)</td><td>Lateral Dominance(%)</td><td>Macro-BA (%)</td></tr><tr><td>CBraMod</td><td>Classifier</td><td>71.68</td><td>72.59</td><td>34.13</td><td>33.60</td><td>40.27</td><td>50.45</td></tr><tr><td>LaBraM</td><td>Classifier</td><td>90.56</td><td>70.60</td><td>89.87</td><td>71.68</td><td>57.60</td><td>76.06</td></tr><tr><td>Dual-Domain</td><td>Classifier</td><td>93.76</td><td>91.29</td><td>93.87</td><td>87.52</td><td>82.93</td><td>89.87</td></tr><tr><td>CBraMod</td><td>Qwen</td><td>61.76</td><td>69.53</td><td>36.27</td><td>36.00</td><td>36.27</td><td>47.96</td></tr><tr><td>LaBraM</td><td>Qwen</td><td>89.76</td><td>72.23</td><td>84.00</td><td>68.80</td><td>52.53</td><td>73.47</td></tr><tr><td>Dual-Domain</td><td>Qwen</td><td>88.64</td><td>87.01</td><td>94.67</td><td>81.60</td><td>78.40</td><td>86.06</td></tr><tr><td colspan="8">EEG input sanity check for Dual-Domain + Qwen</td></tr><tr><td>Shuffled EEG</td><td>Qwen</td><td>20.00</td><td>20.37</td><td>35.73</td><td>18.72</td><td>28.53</td><td>24.67</td></tr><tr><td>Zero EEG</td><td>Qwen</td><td>20.00</td><td>20.00</td><td>33.33</td><td>20.00</td><td>33.33</td><td>25.33</td></tr></table>

Table 3. Comparison of EEG-Captioner training strategies. Factual Consistency evaluates the accuracy of fundamental EEG signal descriptions in generated captions with respect to the corresponding numerical feature summary using GPT-5.6 Sol.
<table><tr><td>Training Supervision</td><td>Signal-level QA Macro-BA(%)↑</td><td>Caption Generation Factual Consistency(%) ↑</td><td>ROUGE-L ↑</td><td>BLEU-4↑</td></tr><tr><td></td><td>47.82</td><td>82.06</td><td>0.4722</td><td>0.3124</td></tr><tr><td>Caption-only adaptation Signal-level QA only</td><td>86.06</td><td>44.10</td><td>0.1470</td><td>0.0083</td></tr><tr><td>Signal-level QA + Caption</td><td>88.81</td><td>83.46</td><td>0.4865</td><td>0.3302</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

At the representation level, performance is evaluated over the five signal-level QA property families using balanced accuracy. At the language level, all encoders are evaluated with the same Qwen-based setup on signal-level QA. We additionally assess the dependence of the Dual-Domain + Qwen model on EEG input through perturbation experiments, replacing the original EEG with cross-sample shuffled EEG or zero signals while keeping the questions unchanged.

EEG-Captioner evaluation. We compare EEG-Captioner with two variants under the same model configuration. The caption-only adaptation variant trains only the language adaptation modules with caption supervision, while retaining the same frozen QA-pretrained Dual-Domain encoder used in the other variants for a controlled comparison. The signal-level-QA-only variant corresponds to the model after signal-level QA training, without subsequent caption training. Captioning performance is evaluated on the same held-out test set using identical metrics across all variants.

## 4. RESULTS

Comparison of EEG Encoders. Table 2 compares different EEG encoders under signal-level QA evaluation. At the classifier stage, the proposed Dual-Domain EEG Encoder achieves 89.87% Macro-BA, outperforming LaBraM (76.06%) and CBraMod (50.45%). The gains are particularly pronounced for strongest-region, temporalpattern, and lateral-dominance prediction, suggesting that the proposed representation better preserves information relevant to these signal-level EEG properties.

After transferring the frozen encoders to Qwen, the Dual-Domain encoder retains its advantage, achieving 86.06% Macro-BA compared with 73.47% for LaBraM and 47.96% for CBraMod, suggesting that the signal-level information captured by the proposed encoder can be effectively transferred to language-based QA.

EEG input perturbation further leads to substantial performance degradation, suggesting that the model’s predictions depend on the EEG input rather than solely on language-side shortcuts.

Effectiveness of EEG-Captioner. Table 3 compares the three variants using the same frozen QA-pretrained EEG encoder. On signal-level QA, caption-only adaptation achieves 47.82% Macro-BA, while signal-level-QA-only training reaches 86.06%. EEG-Captioner further improves Macro-BA to 88.81%, suggesting that caption supervision can complement signal-level QA training. For caption generation, EEG-Captioner improves factual consistency from 82.06% to 83.46% compared with caption-only adaptation, while increasing ROUGE-L from 0.4722 to 0.4865 and BLEU-4 from 0.3124 to 0.3302. Overall, the joint training strategy maintains strong signal-level QA performance while providing modest improvements in caption quality.

## 5. CONCLUSION

In this paper, we present EEG-Captioner, a unified EEG-to-language framework for grounded EEG understanding. We construct an EEGlanguage dataset that pairs EEG captions with signal-level QA pairs covering fundamental temporal, spectral, spatial, and lateral signal properties. We further introduce a Dual-Domain EEG Encoder to model temporal and spectral-spatial information. Experiments show that the proposed encoder achieves improved performance on signal-level QA tasks, while joint QA-caption training improves both signal-level QA performance and caption factual consistency. These results suggest that incorporating explicit signal-level supervision can help improve the grounding of EEG-to-language generation in measurable EEG properties.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study used the de-identified Temple University Hospital EEG Corpus (TUEG), which was collected with approval from the Temple University Institutional Review Board and in accordance with the Declaration of Helsinki [10]. No new participants were recruited or data collected in this study.

## 7. REFERENCES

[1] Fei Ma, Han Lin, Yifan Xie, Hongwei Ren, Xiaoyu Shen, Wenbo Ding, and Qi Tian, “Eˆ2-LLM: Bridging Neural Signals and Interpretable Affective Analysis,” Jan. 2026.

[2] Wei Gu, Tianming Luo, Qiran Zhang, Mohan Ye, Xiao Shen, Wenxin Chen, Yunhuan Li, Yichen Zhang, Jing Hong, Baoliang Lu, and Wei-Long Zheng, “CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation,” in The Fourteenth International Conference on Learning Representations, 2026.

[3] Guoan Wang, Shihao Yang, Jun-en Ding, Hao Zhu, and Feng Liu, “NeuroNarrator: A Generalist EEG-to-Text Foundation Model for Clinical Interpretation via Spectro-Spatial Grounding and Temporal State-Space Reasoning,” 2026.

[4] Vernon J. Lawhern, Amelia J. Solon, Nicholas R. Waytowich, Stephen M. Gordon, Chou P. Hung, and Brent J. Lance, “EEG-Net: A compact convolutional neural network for EEG-Based Brain-Computer Interfaces,” Journal of Neural Engineering, vol. 15, no. 5, pp. 056013, 2018.

[5] Huy Phan, Kaare Mikkelsen, Oliver Y. Chen, Philipp Koch,´ Alfred Mertins, and Maarten De Vos, “SleepTransformer: Automatic Sleep Staging With Interpretability and Uncertainty Quantification,” IEEE Transactions on Biomedical Engineering, vol. 69, no. 8, pp. 2456–2467, Aug. 2022.

[6] Chaoqi Yang, M. Westover, and Jimeng Sun, “BIOT: Biosignal Transformer for Cross-data Learning in the Wild,” Advances in Neural Information Processing Systems, vol. 36, pp. 78240– 78260, Dec. 2023.

[7] Guangyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, and Haifeng Li, “EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals,” in Advances in Neural Information Processing Systems. 2024, vol. 37, pp. 39249–39280, Curran Associates, Inc.

[8] Wei-Bang Jiang, Li-Ming Zhao, and Bao-Liang Lu, “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI,” May 2024.

[9] Jiquan Wang, Sha Zhao, Zhiling Luo, Yangxuan Zhou, Haiteng Jiang, Shijian Li, Tao Li, and Gang Pan, “CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding,” Nov. 2025.

[10] Iyad Obeid and Joseph Picone, “The Temple University Hospital EEG Data Corpus,” Frontiers in Neuroscience, vol. 10, May 2016.

[11] DeepSeek-AI, “DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence,” June 2026.

[12] P. Welch, “The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms,” IEEE Transactions on Audio and Electroacoustics, vol. 15, no. 2, pp. 70–73, June 1967.

[13] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin, “Attention is All you Need,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[14] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu, “Bleu: a Method for Automatic Evaluation of Machine Translation,” in Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, July 2002, pp. 311–318.

[15] Chin-Yew Lin, “ROUGE: A Package for Automatic Evaluation of Summaries,” in Text Summarization Branches Out, Barcelona, Spain, July 2004, pp. 74–81, Association for Computational Linguistics.

[16] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin, “Qwen2.5- Omni Technical Report,” Mar. 2025.

[17] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” Oct. 2021.