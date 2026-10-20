# MULTILINGUAL VOICEMAIL AND ANSWERING MACHINE DETECTION VIA PRETRAINED AUDIO EMBEDDINGS AND GRADIENT BOOSTING MACHINES

Kemal Altwlkany<sup>1,2</sup>, Muamer Paric<sup>1</sup>

<sup>1</sup>Infobip

<sup>2</sup>Faculty of Science, University of Sarajevo

## ABSTRACT

Answering machine detection (AMD) is a fundamental task in telephony that classifies whether an answered call is picked up by a human or a voicemail (machine). Despite its widespread use across cloud communication platforms, existing approaches rely on proprietary systems with undisclosed architectures, or require computationally expensive models that cannot operate in real time. We propose a lightweight AMD system that combines pretrained audio embeddings with gradient boosting machines, trained to prioritize the correct detection of human-answered calls. We evaluate seven models across two datasets: three models that we propose, a proprietary industry model, two prior academic approaches, and a large language model baseline that relies on automatic speech recognition. Furthermore, we release OpenAMD<sup>1</sup>, the first publicly available AMD dataset comprising of more than 500 recordings in 7 languages. Our best model achieves the highest precision across both datasets while maintaining low inference latency. We assess the robustness of models to additive noise and packet loss, and perform tests of statistical significance to discover per-language and per-gender model performance. Finally, we evaluate our model on an out-ofdomain task of detecting robocalls, where it achieves 94.34% recall without any additional training, which is comparable to the industry standard.

Index Terms— voicemail, answering machine, audio classification, audio embeddings, gradient boosting machine

## 1. INTRODUCTION

Answering machine detection (AMD) is a common task in telephony with the goal of classifying the first few seconds of an answered telephone call to determine who had answered the call: a human or an answering machine. The problem originates because it is not possible to distinguish answering machines from humans in telephony from signaling protocols, i.e. two common signaling protocols: Session Initiation Protocol (SIP) [1] and Signaling System No.7 (SS7) [2] do not carry information about who answered the call.

The term answering machine is a broad term encapsulating any form of automated telephone answer and includes interactive voice responses (IVRs) and voicemails, regardless of whether they are artificially synthesized or human-recorded voicemails. The problem is framed from the caller’s perspective, i.e. the caller wants to know whether the callee is a human, or an answering machine. AMD usually operates on a few seconds of the call after which the caller is provided with information about who answered the call. Based on the result, the caller can determine the further progression of the call.

## 1.1. Challenges

The task is challenging due to several reasons. First, detection must be performed within a limited time, as phone calls are terminated in real time. If detection is not performed soon enough, there is a risk that the callee will hang up. Second, the bandwidth of telephony audio is limited; the international standard codec for telephony is G.711 [3], specifically its PCMA and PCMU variants. These codecs operate at 8 kHz sampling rate and introduce non-linear quantization effects: 16-bit audio is compressed into 8-bit audio at a logarithmic scale. Furthermore, audio encoded with G.711 is bandlimited between 300 Hz and 3400 Hz, meaning a portion of the spectral content of speech is filtered out. Third, the acoustic characteristics of voicemails vary depending on the callee’s country, telecom operator, and language. Some voicemails are artificially synthesized messages, while some are prerecorded messages by real humans. Finally, it is often not possible to know the speaking language of the callee based on just their phone number, meaning that solutions that rely on transcribing the call must be multilingual.

## 1.2. Practical Importance

Despite its limited presence in academic literature, AMD is a core component in telephony and in modern communication platforms, having found its application across multiple tasks. We identify four principal use cases that motivate the development of an accurate AMD model.

## 1.2.1. Privacy protection in anonymized calls

Many communication platforms offer user anonymization such as number masking [4, 5, 6]. This is a common feature in driver, delivery, and marketplace applications. However, if a user does not answer a masked call it can reach their voicemail system and some voicemail systems might reveal the true identity to the caller, e.g.: “You’ve reached 061 23...”. AMD enables platforms to detect voicemails that would reveal the true identity of the user, thereby preserving their privacy.

## 1.2.2. Countermeasures for robocall fraud

Robocalling is the practice of making large volumes of automated calls, often for fraudulent purposes. It is a welldocumented problem in telecommunications security, with fraudsters stealing millions of dollars per year [7, 8]. Prior research has shown that robocallers use AMD to determine when humans picked up calls in order to maximize the effectiveness of their attacks [9]. A deeper understanding of AMD can help the security community in three ways: (i) analyzing the detection techniques employed by robocallers, (ii) developing adversarial countermeasures that can deceive attacker AMD systems and thereby reduce the effectiveness of robocall attacks, and (iii) adapting AMD models for the downstream task of robocall detection itself, an approach taken in this paper as well.

## 1.2.3. Public notification systems

Government agencies, educational institutions, and emergency services rely on voice calls to deliver information. If the callee does not pick up, they can use AMD to leave messages in the callee’s voicemail box such as document submission, school closers, vaccination reminders, emergency alerts, etc.

## 1.2.4. Commercial campaign and call center optimization

Voice campaigns are widely used in telemarketing, appointment reminders, and customer engagement. When a human answers the call, they can be provided with the information, but if a machine answers the call, the caller can optimize costs by not interacting with the machine, or a message can be left automatically. Cost savings can be substantial, as it is estimated that about 40-60% of calls are answered by voicemail and in the US [10], with an estimated 2.5-4% of the working population employed in call centers [11, 12].

## 1.3. Paper Contributions

1. We propose a lightweight AMD model based on pretrained audio embeddings and gradient boosting machines that prioritizes human detection rate. The model was derived through hyperparameter optimization and a model-capacity ablation where the gradient boosting component was replaced with logistic regression while keeping the remainder of the process fixed.

2. To the best of our knowledge, we release the first AMD dataset that is publicly available. This dataset contains 233 human recordings and 300 machine recordings in 7 languages. We refer to this dataset as the OpenAMD dataset and describe it in more detail later.

3. We compare our approach against 2 competitor models, 1 state-of-the-art proprietary model, and 1 baseline model that uses automatic speech recognition and a large language model to perform AMD. These comparisons are followed by a detailed study of the robustness and biases of AMD models, with respect to their performance across different languages and genders, as well as their robustness to noise and networking issues that can occur in practice. Finally, the performance of our model is evaluated on an out-of-domain task of detecting robocalls.

## 2. RELATED WORK

Many telecommunication and cloud communication platforms provide AMD, such as Telnyx [14], Twilio [15], Amazon AWS [10], Infobip [16], ElevenLabs [17], and others. The exact implementations are unknown, with the claimed accuracies, multilingual support, inference latency, and features varying depending on the provider.

Prior to the widespread adoption of machine learning, early AMD systems used hand-crafted features and heuristics. A prominent example is Asterisk [18, 19], an opensource private branch exchange (PBX). Asterisk’s AMD [20] uses energy-based measures to detect speech and applies a temporal heuristic: short responses are labeled as human, long greetings as voicemail. This approach is simple and computationally efficient but lags behind modern models. It fails when humans use long greetings or voicemail greetings are short, and its speech detector cannot distinguish background noise from silence - any sufficiently energetic signal is treated as speech.

The approach of Gomez et al. [21] uses automatic speech recognition (ASR) and relies on three features: (i) the time required to answer the phone, (ii) the percentage of silence in the first three seconds of the call, and (iii) the recognized sequence of words using the ASR module. These features are combined into an empirically derived function that estimates the probability that a call is answered by a human. Although this resulted in improved performance, the reliance on language-specific keywords limits generalization to multilingual settings. The system was evaluated on a small dataset of 100 phone calls with only 21 machines.

![](images/8399729336ca3a59131ccb2421246317fdb65e801b1234d38e681a0e42fc40ce.jpg)  
Fig. 1. High-level overview of our proposed method. We compute the log-mel spectrogram from the input audio waveform and extract 512-dimensional clip-wise embeddings using PANNs CNN6 [13]. We train a GBM model on the binary classification task of AMD by using the CNN6 embeddings as inputs.

Anisimov et al. [12] proposed the first deep learning based approach to AMD. They train a convolutional neural network (CNN) to classify calls as human or machine. The input to the CNN are mel-frequency cepstral coefficients (MFCCs) [22, 23] extracted from the first 2.5 seconds of the call. The authors reported an accuracy of 94% on a dataset of 850 samples per both classes.

In [24], the authors addressed several of the previously reported limitations by proposing a recurrent neural network (RNN) architecture. Their approach extracted embeddings from YAMNet [25], an audio tagging model based on the MobileNet architecture [26], and fed those embeddings into a gated recurrent unit (GRU) network. The system achieved 96.7% balanced accuracy on a dataset of approximately 4,200 calls.

Finally, [27] is a large AMD model that was trained by fine-tuning Meta’s XLS-R [28], a self-supervised crosslingual speech model that is based on wav2vec 2.0 [29]. The model contains 300 million parameters and was pretrained on more than 436 000 hours of speech. The AMD version was fine-tuned on English only and achieves 98% accuracy.

## 2.1. Pretrained Audio Representations

The use of pretrained audio representations for downstream tasks has become a popular paradigm in audio and specifically speech-based machine learning applications [29, 30]. Models are trained on large quantities of unlabeled data and fine-tuned on small sets of labeled data.

In [13], Kong et al. train a variety of neural networks on the task of audio tagging and achieve state-of-the-art results on several downstream audio pattern recognition tasks such as acoustic scene classification, music classification, sound event detection, etc. The authors proposed various neural architectures that rely on: conventional CNNs (architectures such as AlexNet and VGG-like), ResNets, MobileNets, onedimensional CNNs such as DaiNet and LeeNet, and a newly proposed Wavegram-Logmel-CNN architecture. The authors jointly refer to these various networks as pretrained audio neural networks (PANNs). The CNN14 model from PANNs was used as the encoder in CLAP, a contrastive languageaudio pretraining approach [30]. Another PANNs model, CNN6, which is based on AlexNet [31] achieves a mean average precision (mAP) of 0.343 on AudioSet [32], compared to their best performing Wavegram-Logmel-CNN model which achieves 0.439 mAP, but has 81 million parameters compared to the CNN6’s 4.8 million.

## 3. METHOD

We formulate AMD as a binary classification task. Given a raw audio signal of length T sampled at $f _ { s } ,$ , the goal is to predict a label y ∈ {HUMAN, MACHINE}, with humans being the negative and machines the positive class. In practice, T is bounded by a configurable detection window, typically between 1 and 5 seconds for most AMD applications [14, 15, 16, 10, 17], and $f _ { s }$ is 8 kHz for narrowband telephony [3].

Our approach decomposes the task into two stages: (i) a fixed feature extractor that maps the raw audio waveform into a compact embedding vector, and (ii) a lightweight classifier that operates on this embedding. This decoupled design allows the classifier to be replaced or updated independently of the feature extractor, and avoids the computational cost of end-to-end fine-tuning. A high level overview of our proposed method is provided by Figure 1.

## 3.1. CNN6 Audio Embeddings

For the feature extraction stage, we use CNN6 from the PANNs [13] collection of pretrained audio models. The CNN6 model takes as input a 64-channel log-mel spectrogram; its architecture consists of four convolutional blocks with $5 \times 5$ kernels, batch normalization, ReLU activations, and average pooling, followed by two fully connected layers, totaling 4.8 million parameters. We only use the first fully connected layer of CNN6 that produces a 512-dimensional latent representation (embedding) of the audio input. CNN6 acts as a frozen feature extractor: given an input audio segment x, we compute the embedding $\mathbf { e } = f _ { \mathrm { C N N 6 } } ( \mathbf { x } ) \in \mathbb { R } ^ { 5 1 2 }$ without updating any network parameters.

## 3.2. LightGBM Classifier

We train a LightGBM [33] classifier on the 512-dimensional CNN6 embeddings. LightGBM is a gradient boosting framework [34, 35] that constructs an ensemble of decision trees using histogram-based algorithms [36] for efficient split finding. Compared to neural classifiers operating on the same embedding space, gradient boosting offers two advantages for the AMD use case: (i) the models produced by Light-GBM are extremely lightweight, enabling real-time inference on CPU-only environments, (ii) tree-based ensembles are inherently regularized through mechanisms such as max depth, minimum leaf samples, and feature subsampling, thereby reducing the risk of overfitting on small datasets.

## 3.3. Training data

We train our GBM model on a proprietary dataset of about 3500 manually labeled recordings, of which 800 are human and 2700 machines. The audio recordings are real, industry examples of AMD and are already transcoded via G.711 [3], with the presence of background and static noise. Although most of the recordings are in English, the dataset does contain recordings in other languages.

To the best of our knowledge, we were unable to find any publicly available datasets for AMD. This comes to no surprise given the high sensitivity of the topic from a user privacy perspective, making it difficult to record and publish data. The lack of publicly available AMD datasets makes it difficult to reproduce and verify the models.

## 3.3.1. OpenAMD Dataset

We introduce the OpenAMD dataset, a multilingual AMD dataset that contains realistic recordings of answered phone calls, both by humans and by machines, comprising 533 recordings in total.

The human samples in the dataset were obtained by recording phone calls from users that participated in the study, as we found the recordings to be more natural when participants would answer the phone instead of speak into a recording device in a studio environment. In total we have 8 different speakers in 6 languages.

We used Azure Cognitive Services Neural TTS [37] to synthesize machine samples in 7 languages (English, Chinese (Mandarin), Arabic, German, etc.), containing a total of 44 different voices, balanced in terms of gender.

Since the voicemails are neurally synthesized, we simulate realistic telephony conditions by introducing augmentations such as adding ringing tones, silent gaps and pauses, and simulating pick-up clicks of calls being connected. Furthermore, we downsample all audio to 8 kHz and transcode it using G.711 [3], randomly altering between PCMA and PCMU. This process introduces the non-linear quantization artifacts, bandwidth limitation, and dynamic range compression characteristic of telephony-grade audio, making the synthetic samples more acoustically similar to real recordings in the training set.

We make the OpenAMD dataset publicly available and accompany the audio recordings with useful metadata: speaker ID (or neural voice for machines), speaker gender, recording text (for both humans and machines), and recording language.

## 3.4. Experimental setup

All results that we present are aggregates over 5 repeated experiment runs and are computed at a 95% confidence interval using the bootstrap approach [38]. Experiments were performed on an NVIDIA DGX B200 platform.

## 3.4.1. Baseline models

We establish a high-quality baseline AMD model that transcribes calls using ASR and uses an LLM to classify them based on the transcript as human or machine. The transcription model needs to be a large multilingual model, and the LLM needs to be sufficiently powerful to understand the task at hand. We opted for Whisper [39] (large-v3-turbo model) and for ChatGPT5.4. Whisper has close to 800 million parameters, while the exact number of parameters of ChatGPT-5.4 is undisclosed, but widely estimated to surpass 2 trillion parameters. This approach does therefore not scale well and would not be of practical use, as AMD must be a resource efficient model that can run in real time on a CPU. However, it can be beneficial to use this approach as a baseline given its large predictive capability.

Regarding existing approaches, we compare to RNN-YAMNet [24] and wav2vec-vm-finetune [27], and a proprietary state-of-the-art model AMD-Infobip [16].

## 3.4.2. GBM models

We hypothesize that, given the AMD use cases described in the introductory section, it is of paramount importance for the model to classify humans correctly. In particular, the model should prioritize precision and specificity, since falsely classifying humans as machines may lead to worse outcomes than falsely classifying machines as humans. Therefore, we provide three model versions:

Table 1. Performance of AMD models on proprietary dataset.
<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Specificity</td><td>F1</td></tr><tr><td>RNN-YAMNet [24]</td><td> $8 3 . 3 5 \pm 2 . 2 5$ </td><td> $8 7 . 7 2 \pm 3 . 0 4$ </td><td> $7 4 . 4 0 \pm 4 . 0 4$ </td><td> $9 1 . 0 4 \pm 2 . 4 3$ </td><td> $8 0 . 4 9 \pm 2 . 5 6$ </td></tr><tr><td>AMD-Infobip [16]</td><td> $8 8 . 0 9 \pm 1 . 8 6$ </td><td> $9 2 . 0 9 \pm 2 . 5 9$ </td><td> $8 1 . 2 0 \pm 3 . 5 5$ </td><td> $9 4 . 0 1 \pm 1 . 9 6$ </td><td> $8 6 . 2 9 \pm 2 . 4 4$ </td></tr><tr><td>CNN6-LogReg (our)</td><td> $8 5 . 6 1 \pm 2 . 3 8$ </td><td> $9 1 . 5 5 \pm 2 . 6 5$ </td><td> $7 5 . 8 7 \pm 4 . 4 3$ </td><td> $9 3 . 9 9 \pm 1 . 8 9$ </td><td> $8 2 . 9 6 \pm 2 . 7 3$ </td></tr><tr><td>CNN6-GBM-balanced (our)</td><td> $8 5 . 9 1 \pm 2 . 4 6$ </td><td> $9 2 . 3 3 \pm 2 . 9 5$ </td><td> $7 5 . 8 3 \pm 3 . 8 3$ </td><td> $9 4 . 5 9 \pm 2 . 2 7$ </td><td> $8 3 . 2 5 \pm 3 . 0 3$ </td></tr><tr><td>CNN6-GBM (our)</td><td> $8 5 . 2 0 \pm 2 . 1 0$ </td><td> ${ \bf 9 4 . 7 0 \pm 2 . 3 7 }$ </td><td> $7 2 . 0 2 \pm 4 . 0 8$ </td><td> ${ \bf 9 6 . 5 4 \pm 1 . 4 9 }$ </td><td> $8 1 . 8 0 \pm 2 . 8 1 $ </td></tr><tr><td>wav2vec-vm-fine [27]</td><td> $8 4 . 7 2 \pm 2 . 2 1$ </td><td> $8 7 . 7 7 \pm 2 . 9 9$ </td><td> $7 7 . 7 9 \pm 3 . 8 7$ </td><td> $9 0 . 6 8 \pm 2 . 1 3$ </td><td> $8 2 . 4 6 \pm 2 . 8 3$ </td></tr><tr><td>whisper-GPT5.4 (baseline)</td><td> ${ \bf 9 1 . 0 9 \pm 1 . 5 9 }$ </td><td> $8 6 . 5 8 \pm 3 . 4 0$ </td><td> ${ \bf 9 5 . 5 3 \pm 1 . 8 6 }$ </td><td> $8 7 . 2 8 \pm 2 . 7 7$ </td><td> ${ \bf 9 0 . 8 3 \pm 1 . 8 0 }$ </td></tr></table>

![](images/27579dd81ac703a0c790995636f3dd4fe6b6c72eafb33396a3177ae166747415.jpg)  
Fig. 2. Critical difference diagram showing the average ranks of the models across accuracy.

• CNN6-GBM, our primary model, prioritizes precision and specificity, thus minimizing the chance of wrongly classifying humans

• CNN6-GBM-balanced, our balanced model that provides a good trade-off between recall and specificity

• CNN6-LogReg, a model trained using logistic regression instead of GBM, with the goal of investigating the contribution of the non-linear gradient boosting component and whether CNN6 produces linearly separable embeddings

## 4. RESULTS

Table 1 provides the benchmarks of all 7 models on a proprietary dataset (different from the training set). The whisper-GPT-5.4 baseline provides the best balanced performance such as accuracy and F1-score, but the CNN6-GBM model has the best specificity and precision.

We compared the accuracy of the 7 classifiers using the Friedman test which showed a statistically significant difference among classifiers $( \chi ^ { 2 } ( 6 ) = 4 4 8 . 7 8 , \ p \ll 1 e \mathrm { - } 1 0 ) .$ with Kendall’s $W = . 7 4 8 .$ , indicating a large degree of consistency in classifier rankings. Post-hoc Nemenyi tests with Holm-Bonferroni correction were used to compare average

## 4.1. Proprietary dataset

ranks across classifiers and showed that models CNN6-GBM, CNN6-GBM-balanced, CNN6-LogReg, and wav2vec-vmfinetune do not differ significantly in terms of accuracy, whereas whisper-GPT-5.4 and AMD-Infobip are significantly better than other models, while RNN-YAMNet is the worst ranked model. This is illustrated by the critical difference diagram in Figure 2.

We repeated the procedure but analyzed specificity among the 7 classifiers and obtained: $( \chi ^ { 2 } ( 6 ) = 5 2 3 . 0 4 , \ p \ll 1 e \mathrm { - } 1 0 )$ with Kendall’s W = .872, indicating an even larger degree of consistency in classifier rankings among specificity compared to accuracy. Post-hoc Nemenyi tests with Holm-Bonferroni correction showed that our CNN6-GBM differs significantly from other models, while there is no statistically significant difference between the performance of AMD-Infobip and CNN6-LogReg, and between RNN-YAMNet and wav2vecvm-finetune. This is illustrated in Figure 3.

## 4.2. OpenAMD dataset

We now benchmark the models on the OpenAMD dataset and provide the results in Table 2. The ranks of the models differ compared to the proprietary dataset, with the exception of CNN6-GBM being the best performing model in terms of precision on both datasets. On the OpenAMD dataset, RNN-YAMNet obtains the best accuracy, recall, and F1 scores, with wav2vec-vm-finetune now achieving the best specificity.

## 4.2.1. Robustness to noise and coding artifacts

In Figure 4 we show the performance of the models during non-ideal conditions: when the audio has been corrupted with noise or when packet loss is present in the network (simulated using the Gilbert-Elliot model [40]). The results indicate that the whisper-GPT-5.4 model is most robust to noise distortions, whereas the GBM-based models are the worst performing models for noisy audio. The GBM-based models maintain good performance in poor network quality conditions, with RNN-YAMNet and AMD-Infobip being the most robust models for packet loss.

![](images/907dc87beeddd8e873abb49455a918029d58d42721145b6884a3c0bdc0472533.jpg)  
Fig. 3. Critical difference diagram showing the average ranks of the models across specificity.

Table 2. Performance of AMD models on OpenAMD dataset.
<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Specificity</td><td>F1</td></tr><tr><td>RNN-YAMNet [24]</td><td> ${ \pm } 2 . 5 6 \pm 2 . 0 6$ </td><td> $9 1 . 6 9 \pm 3 . 3 8$ </td><td> ${ \bf 9 9 . 3 3 \pm 1 . 0 6 }$ </td><td> $8 8 . 4 1 \pm 4 . 3 6$ </td><td> ${ \pm 1 . 8 8 }$ </td></tr><tr><td>AMD-Infobip [16]</td><td> $9 1 . 7 4 \pm 2 . 4 4$ </td><td> $8 8 . 5 5 \pm 3 . 6 4$ </td><td> $9 8 . 0 0 \pm 1 . 8 2 $ </td><td> $8 3 . 6 9 \pm 4 . 9 8$ </td><td> $9 3 . 0 4 \pm 2 . 1 9$ </td></tr><tr><td>CNN6-LogReg (our)</td><td> $8 8 . 7 4 \pm 2 . 8 1$ </td><td> $8 7 . 9 7 \pm 3 . 7 3$ </td><td> $9 2 . 6 7 \pm 3 . 1 4$ </td><td> $8 3 . 6 9 \pm 4 . 7 2$ </td><td> $9 0 . 2 6 \pm 2 . 7 0$ </td></tr><tr><td>CNN6-GBM-balanced (our)</td><td> $9 1 . 9 3 \pm 2 . 4 4$ </td><td> $9 2 . 4 1 \pm 3 . 1 2$ </td><td> $9 3 . 3 3 \pm 2 . 9 4$ </td><td> $9 0 . 1 3 \pm 4 . 0 4$ </td><td> $9 2 . 8 7 \pm 2 . 2 6$ </td></tr><tr><td>CNN6-GBM (our)</td><td> $9 2 . 5 0 \pm 2 . 4 4$ </td><td> ${ \pm } 5 . 1 4 \pm 2 . 6 4$ </td><td> $9 1 . 3 3 \pm 3 . 5 1$ </td><td> $9 3 . 9 9 \pm 3 . 1 2$ </td><td> $9 3 . 2 0 \pm 2 . 3 8$ </td></tr><tr><td>wav2vec-vm-finetune [27]</td><td> $8 4 . 4 3 \pm 3 . 0 0$ </td><td> $9 5 . 0 2 \pm 2 . 9 7$ </td><td> $7 6 . 3 3 \pm 4 . 7 2$ </td><td> ${ \pm } 4 . 8 5 \pm 2 . 9 9$ </td><td> $8 4 . 6 6 \pm 3 . 2 3$ </td></tr><tr><td>whisper-GPT-5.4 (baseline)</td><td> $9 1 . 5 6 \pm 2 . 4 4$ </td><td> $8 7 . 6 1 \pm 3 . 5 2$ </td><td> $9 9 . 0 0 \pm 1 . 3 0 $ </td><td> $8 1 . 9 7 \pm 5 . 0 5$ </td><td> $9 2 . 9 6 \pm 2 . 0 8$ </td></tr></table>

## 4.2.2. Per-language performance

We analyzed whether the performance of the models depends on the language of the callee/voicemail. A chi-squared test of independence was performed on each model’s accuracy across different languages, with Holm-Bonferroni correction for multiple comparisons. Six of the models showed no statistically significant language dependence, i.e. the accuracies of the models are independent of the language. For wav2vecvm-finetune we obtained $( \chi ^ { 2 } ( 4 ) = 2 0 . 2 8$ , p = 0.003) with Cramer’s $V = 0 . 2 1 8$ indicating a statistically significant but small association, i.e. the model’s performance depends on the language, but not strongly. We hypothesize that this might be due to the model being fine-tuned on English recordings only, thereby degrading the previously learned knowledge of the XLS-R model [28].

## 4.2.3. Per-gender performance

We ran a separate Fisher’s exact test for machine recall and for human specificity to see if male vs. female recordings of humans/machines get correctly detected at the same rate. The recall and specificity per each model are provided by Figure 5. For machine recall, only wav2vec-vm-fintune exhibited a significant gender disparity (odds ratio $O R = . 0 6 8 , p < . 0 0 1 )$ indicating that male voicemails are ≈ 15 times as likely to be misclassified as human. For human specificity, no statistically significant results were obtained, although RNN-YAMnet provides a relatively large gap between the performance in male vs. female specificity.

## 4.3. Inference latency

In Table 3 we show the inference latency of the CNN6-based models and wav2vec-vm-finetune. Note that we did not have access to the inference latency information of the RNN-YAMNet, AMD-Infobip, and whisper-GPT-5.4 models, but it is clear that the latter cannot operate in real-time given its reliance on Whisper [39] and ChatGPT-5.4. The average latency of CNN6-based models is two orders of magnitude less than that of the wav2vec-vm-finetune model, with the balanced GBM model having a more variable inference latency compared to other CNN6-based models.

Table 3. Inference latency of AMD models (in milliseconds).
<table><tr><td>Model</td><td>Mean</td><td>Median</td><td>Std</td></tr><tr><td>CNN6-GBM</td><td>1.23</td><td>1.21</td><td>0.092</td></tr><tr><td>CNN6-GBM-balanced</td><td>1.36</td><td>1.33</td><td>0.358</td></tr><tr><td>CNN6-LogReg</td><td>1.19</td><td>1.18</td><td>0.062</td></tr><tr><td>wav2vec-vm-finetune</td><td>131.99</td><td>132.85</td><td>14.179</td></tr></table>

![](images/64ce386179d8c710a842d20c4a873840197280cce561b56e78a0aead8bae29d3.jpg)

Fig. 4. Accuracy of AMD models in noisy and poor network quality conditions. Left: audio is distorted by noise (x-axis indicates SNR). Right: audio is corrupted by packet loss. Shaded areas and black bars indicate the 95% CI.  
![](images/83e5297313821b7faf9749a9b695148a9d8e543777cc0064c7b9931bc4d15f3d.jpg)

![](images/5e88d9a2128bc05cba48664cc5757841a6725021092bbcc69bba687e8752d3b8.jpg)  
Fig. 5. Machine recall and human specificity, per gender and per model. Black bars indicate the 95% CI.

## 4.4. Ablation study

We performed a grid search on the parameters of CNN6- GBM, summarized in Table 4. From the results it appears that our model has found “sweet spots”; optimal parameter configurations are located “in the middle”, with no parameter taking extreme values. Increasing the number of estimators beyond 200 yields no positive gains, nor does increasing the number of leaves beyond 61. Additionally, we argue that the non-linearity of gradient boosting results in a small but noticeable performance improvement as opposed to logistic regression, as can be inferred by comparing the performances of the CNN6-GBM and CNN6-LogReg models. CNN6- LogReg performs worse than CNN6-GBM in terms of accuracy, precision, and F1-score on the OpenAMD dataset, with the difference in specificity greater than 10% on average. Given that the AMD-LogReg inference latency is on average

Table 4. CNN6-GBM hyperparameter search space and selected values.
<table><tr><td>Parameter</td><td>Search space</td><td>Selected</td></tr><tr><td>Learning rate</td><td>{0.03, 0.1, 0.3}</td><td>0.1</td></tr><tr><td>Number of estimators</td><td>{50, 100, 200, 300, 400, 500}</td><td>200</td></tr><tr><td>Num leaves</td><td>{17, 31, 41, 51, 61, 127}</td><td>61</td></tr><tr><td>Min child samples</td><td>{5, 10, 20, 30, 50, 100, 200}</td><td>50</td></tr></table>

0.04 ms faster, we argue that the performance boost provided by GBM is worth the negligible increase in computational complexity.

## 4.5. Transfer to Out-of-Domain Task

We use the CNN6-GBM model as-is and apply it to an outof-domain task of detecting robocalls [9, 7, 8] without any additional training or fine-tuning. We use the dataset from the US Federal Trade Commission (FTC) [41] that contains 1432 robocalls (all positive samples) and manage to detect 1351 robocalls, which is a recall rate of 94.34%, close to the industry standard of 93-97% [42, 43]. We hope that the performance of CNN6-GBM could be further improved by additional training so that it could be used to detect fraudulent robocalls.

## 4.6. Discussion

We proposed three CNN6-based models, with significantly lower inference time than competitor approaches. Our best performing model CNN6-GBM, tuned to prioritize the rate of human detection provided the best precision on both datasets and is statistically the best performing model in terms of specificity on the first dataset. We argue that the high precision rate is due to our goal of minimizing false positives; we want to ensure that the model is very confident before it classifies a call as voicemail, thereby reducing the chances of falsely categorizing humans as machines. Removing the nonlinear gradient boosting component impacts performance as indicated by the CNN6-LogReg model, while yielding minimal improvements in inference latency. The hyperparameter optimization study showed that increasing the number of estimators or leaves improves the model’s predictive capabilities up to a certain point, after which no improvement gains were seen. There is room for improvement regarding noise robustness, as the CNN6-based models perform worse than competitors in this area, although they maintain decent performance during poor network conditions. The CNN6- based models showed no bias in terms of language or gender, implying that the models learned to generalize well and can be used in multilingual settings, regardless of the language or gender of the speaker.

## 5. CONCLUSION

In this paper we showed a new approach to AMD by using pretrained audio neural networks and combining them with gradient boosting machines. Our model reaches comparable accuracy to the industry standard and current state-of-the-art, it is the best performing model in terms of precision while maintaining high specificity, thereby maximizing the correct detection rate of humans. We created OpenAMD, a multilingual and diverse dataset and make it publicly available. Our analysis shows that our models generalize well and are not biased to any language or gender. The robustness to packet loss is good, although there is room for improvement regarding robustness to additive noise. Finally, our model showed to perform well on the out-of-domain task of robocall detection without any additional training or fine-tuning.

## 6. REFERENCES

[1] Eve Schooler, Jonathan Rosenberg, Henning Schulzrinne, Alan Johnston, Gonzalo Camarillo, Jon Peterson, Robert Sparks, and Mark J. Handley, “SIP: Session Initiation Protocol,” RFC 3261, July 2002.

[2] International Telecommunication Union, “Introduction to CCITT Signalling System No. 7,” ITU-T Recommendation Q.700, International Telecommunication Union, Geneva, Switzerland, Nov. 1988.

[3] International Telecommunication Union, “Pulse code modulation (PCM) of voice frequencies,” ITU-T Recommendation G.711, International Telecommunication Union, Geneva, Switzerland, Nov. 1988.

[4] Sinch, “Protect privacy with number masking,” http s://sinch.com/voice/programmable-voi ce/number-masking/, 2026, [Online] Accessed: 2026-05-11.

[5] Telnyx, “Phone number masking,” https://telnyx .com/use-cases/phone-number-masking, 2026, [Online] Accessed: 2026-05-11.

[6] Twilio, “What are Masked Phone Numbers?,” http s://www.twilio.com/docs/glossary/wha t-are-masked-phone-numbers, 2026, [Online] Accessed: 2026-05-11.

[7] Sathvik Prasad, Elijah Bouma-Sims, Athishay Kiran Mylappan, and Bradley Reaves, “Who’s calling? characterizing robocalls through audio and metadata analysis,” in 29th USENIX Security Symposium (USENIX Security 20), 2020, pp. 397–414.

[8] Sathvik Prasad, Aleksandr Nahapetyan, and Bradley Reaves, “Characterizing robocalls with multiple vantage points,” in 2025 IEEE Symposium on Security and Privacy (SP). IEEE, 2025, pp. 19–36.

[9] Huahong Tu, Adam Doupé, Ziming Zhao, and Gail-Joon Ahn, “Sok: Everyone hates robocalls: A survey of techniques against telephone spam,” in 2016 IEEE Symposium on Security and Privacy (SP). IEEE, 2016, pp. 320–338.

[10] Amazon AWS, “Best practices for answering machine detection,” https://docs.aws.amazon.com/ connect/latest/adminguide/outbound-c ampaign-best-practices.html, 2026, [Online] Accessed: 2026-05-11.

[11] Communications Workers of America, “Federal Call Center Factsheet,” https://cwa-union.org/s ites/default/files/federal\_call\_cent

er\_factsheet\_.pdf, [Online] Accessed: 2026- 05-11.

[12] Nikolay Anisimov, Mikhail Olkhovsky, and Konstantin Kishinsky, “Answering machines detection in contact centers using convolutional neural networks,” FizTech-USA, 2023.

[13] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D Plumbley, “Panns: Largescale pretrained audio neural networks for audio pattern recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880– 2894, 2020.

[14] Telnyx, “Answering Machine Detection,” https:// developers.telnyx.com/docs/voice/pro grammable-voice/answering-machine-d etection, 2026, [Online] Accessed: 2026-05-11.

[15] Twilio, “Answering Machine Detection,” https: //www.twilio.com/docs/voice/answerin g-machine-detection, 2026, [Online] Accessed: 2026-05-11.

[16] Infobip, “Automated machine detection,” https:// www.infobip.com/docs/voice-and-vid eo/calls#automated-machine-detection -understanding-calls-api, 2026, [Online] Accessed: 2026-05-11.

[17] ElevenLabs, “Voicemail detection,” https://elev enlabs.io/docs/eleven-agents/customiz ation/tools/system-tools/voicemail-d etection, 2026, [Online] Accessed: 2026-05-11.

[18] Jim Van Meggelen, Leif Madsen, and Jared Smith, Asterisk: Thefuture oftelephony, " O’Reilly Media, Inc.", 2007.

[19] Leif Madsen, Jim Van Meggelen, and Russell Bryant, Asterisk: The definitive guide, " O’Reilly Media, Inc.", 2011.

[20] Enxhi Minaj, “Optimization of answering machine detection application in asterisk,” M.S. thesis, Lappeenranta-Lahti University of Technology (LUT University), Lappeenranta, Finland, 2020.

[21] Jon A Gómez, Luis M Montes-Novella, and José Garcıa, “Answering machine detector for call centers,” in Proc. iberSPEECH 2014, 2014, pp. 198–207.

[22] Zrar Kh Abdul and Abdulbasit K Al-Talabani, “Mel frequency cepstral coefficient and its applications: A review,” Ieee Access, vol. 10, pp. 122136–122158, 2022.

[23] Ambuj Mehrish, Navonil Majumder, Rishabh Bharadwaj, Rada Mihalcea, and Soujanya Poria, “A review of deep learning techniques for speech processing,” Information Fusion, vol. 99, pp. 101869, 2023.

[24] Kemal Altwlkany, Sead Delalic, Elmedin Selmanovi´ c,´ Adis Alihodžic, and Ivica Lovri´ c, “A recurrent neural´ network approach to the answering machine detection problem,” in 2024 47th MIPRO ICT and Electronics Convention (MIPRO). IEEE, 2024, pp. 85–90.

[25] Google, “YAMNet: A deep net for audio classification,” https://github.com/tensorflow/models /tree/master/research/audioset/yamne t, 2020, [Online] Accessed: 2026-05-11.

[26] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen, “Mobilenetv2: Inverted residuals and linear bottlenecks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 4510–4520.

[27] Jake Downie, “wav2vec-vm-finetune,” https://hu ggingface.co/jakeBland/wav2vec-vm-fin etune, 2025, [Online] Accessed: 2026-05-11.

[28] Arun Babu, Changhan Wang, Andros Tjandra, Kushal Lakhotia, Qiantong Xu, Naman Goyal, Kritika Singh, Patrick von Platen, Yatharth Saraf, Juan Pino, Alexei Baevski, Alexis Conneau, and Michael Auli, “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” in Interspeech 2022, 2022, pp. 2278–2282.

[29] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12449–12460, 2020.

[30] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang, “Clap learning audio concepts from natural language supervision,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[31] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton, “Imagenet classification with deep convolutional neural networks,” Advances in neural information processing systems, vol. 25, 2012.

[32] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter, “Audio set: An ontology and human-labeled dataset for audio events,” in 2017 IEEE international conference on acoustics,

speech and signal processing (ICASSP). IEEE, 2017, pp. 776–780.

[33] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu, “LightGBM: A highly efficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems (NeurIPS), 2017, vol. 30, pp. 3146–3154.

[34] Jerome H Friedman, “Greedy function approximation: a gradient boosting machine,” Annals of statistics, pp. 1189–1232, 2001.

[35] Alexey Natekin and Alois Knoll, “Gradient boosting machines, a tutorial,” Frontiers in neurorobotics, vol. 7, pp. 63623, 2013.

[36] Ping Li, Qiang Wu, and Christopher Burges, “Mcrank: Learning to rank using multiple classification and gradient boosting,” Advances in neural information processing systems, vol. 20, 2007.

[37] Microsoft, “Azure Speech Studio,” https://spee ch.microsoft.com/, 2026, [Online] Accessed: 2026-05-11.

[38] Trevor Hastie and Robert Tibshirani and Jerome Friedman, The Elements ofStatistical Learning – Data Mining, Inference, and Prediction, Springer, New York, 2009.

[39] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28492–28518.

[40] Jaden Pieper, “Relationships between gilbert-elliot burst error model parameters and error statistics,” Tech. Rep., Institute for Telecommunication Sciences, 2023.

[41] Sathvik Prasad and Bradley Reaves, “Robocall audio from the ftc’s project point of no entry,” North Carolina State University, Tech. Rep. TR-2023-1, 2023.

[42] Benjamin Elizalde and Dimitra Emmanouilidou, “Detection of robocall and spam calls using acoustic features of incoming voicemails,” Proceedings ofMeetings on Acoustics, vol. 45, no. 1, pp. 060004, 02 2022.

[43] Sharbani Pandit, Krishanu Sarker, Roberto Perdisci, Mustaque Ahamad, and Diyi Yang, “Combating robocalls with phone virtual assistant mediated interaction,” in 32nd USENIX Security Symposium (USENIX Security 23), 2023, pp. 463–479.