# Benchmarking Fast Domain Adaptation for Unsupervised Speech Units

Robin San Roman<sup>§</sup> CoML team, ENS, PSL university, CNRS, EHESS Paris, France

Paul Michel CoML team, ENS, PSL university, CNRS, EHESS Paris, France

Manel Khentout<sup>§</sup> CoML team, ENS, PSL university, CNRS, EHESS Paris, France

Yossi Adi Hebrew University of Jerusalem Jerusalem, Israel

Tu Anh Nguyen CoML team, ENS, PSL university, CNRS, EHESS Paris, France

Emmanuel Dupoux CoML team, ENS, PSL university, CNRS, EHESS Paris, France

Abstract—Representation learning has attracted great attention and managed to reach good performances as a pretraining method for downstream tasks or as a first step towards unsupervised speech modeling. Yet, little is known about how such methods deal with out-of-domain speech and how could they be adapted in a few shot to new domains. This is important especially for accented speech where one observes a long tail of accents that diverge from the standard ones. We introduce ABX-Accent, a benchmark based on the AESRC dataset that features 10 different accents of English. It includes a small (< 10 hours) unlabelled training set in each of the accents and adaptations of the Zero Resources Challenge ABX evaluation metrics to each of the accents. We illustrate this benchmark with a baseline model that uses adaptive domain normalization to fine tune a pretrained Contrastive Predictive Coding model on the accents. This method is first developed on LibriSpeech using a male/female split. When applied to the new benchmark, the proposed method yields a relative improvement of 23.6% on across-speaker ABX scores on average compared to non adapted models. The data and metrics will be open sourced upon paper acceptance.

Index Terms: speech recognition, unsupervised representation learning, usupervised domain adaptation.

## I. INTRODUCTION

Self-supervised speech representation learning (SSL) has recently gained a lot of traction, both as a way to pretrain models on large amounts of unlabelled data, which can be fine tuned for a variety of downstream tasks [1], [2], and as a building block for textless NLP systems that learn language models directly from speech [3], [4]. The assumption in both cases is that it is relatively easier to find large amounts of unlabelled audio than it is to find labelled data. Yet, even unlabeled audio is present in skewed distributions, with a predominance of high resourced languages (e.g., English, Spanish, etc.), and within these, a predominance of a few dominant dialects or accents. It is likely that as in other areas of machine learning, selfsupervised speech representation learning may suffer when the data distribution is heavily imbalanced, creating bias in favor of the dominant mode, and even more when trained models are used out-of-domain.

Here, we focus on the problem of accented speech, which has the characteristics that it is impossible to ever consider a training set that will encompass all possible accents. Sociolinguistic studies show that language’s accent vary geographically, through time and across generations. In addition, second language learners bring in their own idiosyncratic accents. This yields a long tail of constantly renewed accents which raises problems of lack of inclusiveness and bias in the downstream applications built with such datasets. A possible way to deal with this situation is to setup systems that can adapt to each new accent with as little data as possible.

Building on the AESRC dataset that features 10 different accents of English, and present two contributions. (1), we setup a ABX-Accent, a benchmark that evaluates how SSL models can adapt to novel accents with little data and no supervision (around 10 hours per accent, with additional speaker-level adaptation data at test time). The evaluation metric is ABX [5], a metric used in the zero resource speech challenge series [6], [7] focused on the phonetic quality of the learned representations and which has been shown to predict downstream language modeling tasks based on these representations [3]. (2) we explore some simple methods for domain adaptation to provide baseline results in our new benchmark.

## II. RELATED WORK

## A. Representation learning

Inspired by successes in computer vision, and natural language processing, self supervised learning (SSL) applied to speech, aims at learning to extract features from unlabelled data that can be used for downstream tasks such as ASR, voice conversion, emotion recognition and other tasks [1], [2]. A large collection of methods have been used, including clustering and mixture models [8], but recent work tend to focus two main classes of models: masked prediction models and compressive reconstruction models. The models of the first class learn a representation for speech that can best predict a masked portion of the data based on either past samples [9], [10], or both past and future samples [11]–[14]. Models of the second class are auto-encoders that learn a discrete latent representation that can reconstruct the input data [15], [16]. Here, we use as a illustration of our benchmark a simple model from the first class (contrastive predictive coding, CPC, [9]).

![](images/1e4d06141a8ae8403ca14c112c878ad7e770f5291a1c91190cb53700f4fb11ba.jpg)  
Fig. 1: Structure of the ABX-Accent benchmkark. Each of the K accent has a training set, a dev and a test set, each containing different speakers (K=10 accents). To allow for speaker adaptation, the dev and test set provide for each speaker a 2min adaptation set.

## B. Domain adaptation

Robustness to speaker or domain changes in unsupervised speech models is a common problem. In older HMM-GMM ASR systems, common techniques included feature normalization like VTLN or fMMLR that intended to learn a transformation of the input features in order to match the distributions across domains. Since the advent of deep learning, the focus has been to increase the size of the training set in order to cover as many domains as possible, or conditioning the model with speaker representations. In [17] authors try to address the imbalanced amount of data between speakers with a resampling strategy. In [18], the authors use adaptation layers, a technique that we will use in this paper. Other domain adaptation techniques such as Teacher-Student learning [19] showed promising results but heavily rely on the availability of huge amount of unlabelled or weakly labelled data. Adversarial training [20], [21] is a common way of performing domain adaptation, this technique uses a discriminator that is trained to infer the domain from the intermediate features of a model. This discriminator creates a loss that reduces the shift between source and target domain features.

## III. THE ABX-ACCENT BENCHMARK

## A. Dataset

We build our benchmark on top of the AESRC dataset [22] , which contains ten different regional accents. From this dataset we make a split for each accent between a train, a dev and a test set, balancing for male and female speakers (See Figure 1). The Test and dev sets consist of two hours of speech for each accent, with six females and six males (approximately 10 minutes per speaker). In addition, for each of these speakers, we prepare an additional 2:00 minutes for extracting speaker-specific statistics or embeddings for speaker adaptation methods. It allows methods that aim to compensate for the high variability within non native speakers. The rest of the speakers is used in the training set (see Table I for the duration and number of speakers per accent).

TABLE I: Duration and number of speakers for the Train Set of ABX-Accent. Prefix ”e-” highlights that the data is English speech.
<table><tr><td>Accent</td><td>Acronym</td><td>Duration</td><td>Nb of speakers</td></tr><tr><td>American</td><td>e-us</td><td>9:56h</td><td>21 H, 25 F</td></tr><tr><td>British</td><td>e-uk</td><td>15:06h</td><td>38 H, 36 F</td></tr><tr><td>Canadian</td><td>e-ca</td><td>8:07h</td><td>9H, 10 F</td></tr><tr><td>Chinese</td><td>e-ch</td><td>8:23h</td><td>14 H, 12 F</td></tr><tr><td>Indian</td><td>e-in</td><td>7:34h</td><td>8H, 10 F</td></tr><tr><td>Portuguese</td><td>e-pt</td><td>9:18h</td><td>13 H, 15 F</td></tr><tr><td>Korean</td><td>e-ko</td><td>8:24h</td><td>11 H, 11 F</td></tr><tr><td>Japanese</td><td>e-ja</td><td>8:25h</td><td>11 H, 11 F</td></tr><tr><td>Russian</td><td>e-ru</td><td>7:35h</td><td>8H,9F</td></tr><tr><td>Spanish</td><td>e-es</td><td>8:06h</td><td>10 H, 10 F</td></tr></table>

We use the AESRC transcriptions of the dev and test set which we phonemize (using phonemizer) and force align (using kaldi) to obtain the timestamps of each phonemes. The train set does not have transcriptions.

## B. Metrics

We provide in our Github repository all the necessary resources to compute the ABX error that have been used in several zero resource challenges [23] [24]. ABX metric evaluate for a pair of sounds representations (A, B) from for example (”bap”, ”bop”), the probability that the representation X of another instance of the sound ”bap” is closer to A than B. ABX error rate is computed by averaging over all the minimal pairs of phone trigrams in the corpus. In this benchmark we focus on the more challenging ABX across-speaker metric, which uses X instances from a different speaker than the one of the pair (A,B).

## IV. ADAPTATION BASELINE

Here, we present a simple baseline system based on Contrastive Predictive Coding (CPC), which illustrates the challenges posed by this benchmark.

## A. Contrastive Predicting coding

CPC has been introduced in [25] as a simple SSL method for learning speech representations [26], [27]. Those models are made of 2 components :

• An encoder that generates a sequence of embeddings (z<sub>1</sub>, z , ..) from the audio data.

• An autoregressive network that generates a context $c _ { t }$ from the sequence of embeddings.

The training objective consists in

$$
\mathcal { L } _ { t } = - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \log \bigg ( \frac { \exp ( z _ { t + m } ^ { T } W _ { m } c _ { t } ) } { \sum _ { x \in X } \exp ( x ^ { T } W _ { m } c _ { t } ) } \bigg )\tag{1}
$$

Where $W _ { m }$ is a linear classifier, X is the set of negative examples.

TABLE II: ABX scores (%) across speakers for CPC models trained on the Old domain (male or female LibriSpeech) and tested on the Old and New domains (the other sex). Lower scores are better and are averaged across males and females experiments. On average the baseline models only trained on the Old domain give 9.06% and 11.86% on the Old and New test sets respectively. Bold correspond to the best scores for each line and underline, overall. FT: simple fine tuning. resamp-FT: fine tuning with resampling. DN: domain normalization.
<table><tr><td rowspan="2">Split /Domain</td><td colspan="2">FT</td><td colspan="2">resamp-FT</td><td colspan="2">DN</td><td colspan="2">DN+resamp-FT</td></tr><tr><td>Old</td><td>New</td><td>Old</td><td>New</td><td>Old</td><td>New</td><td>Old</td><td>New</td></tr><tr><td>2min 1 spk</td><td>9.16</td><td>11.82</td><td>9.24</td><td>11.79</td><td>9.06</td><td>15.22</td><td>9.10</td><td>10.8</td></tr><tr><td>20min 10 spk</td><td>9.31</td><td>11.42</td><td>9.17</td><td>11.15</td><td>9.10</td><td>13.68</td><td>9.15</td><td>9.31</td></tr><tr><td>2h 60 spk</td><td>9.41</td><td>10.62</td><td>9.05</td><td>10.02</td><td>9.22</td><td>9.76</td><td>9.05</td><td>8.59</td></tr><tr><td>8h 240 spk</td><td>9.57</td><td>9.74</td><td>8.89</td><td>9.38</td><td>9.09</td><td>8.98</td><td>8.89</td><td>8.19</td></tr><tr><td>16h 479 spk</td><td>9.56</td><td>9.96</td><td>9.00</td><td>9.33</td><td>8.93</td><td>8.93</td><td>8.34</td><td>8.04</td></tr></table>

In this work the encoder is made of 5 1-d convolutional layers with 256 channels each. Kernel sizes (10, 8, 4, 4, 4) and strides of (5, 4, 2, 2, 2). The output of every convolutional layer is normalized either using the layer normalization or the adaptive domain normalization that will be discussed in the following section. The context network consists in 2 LSTM layers with 256 hidden dimensions. Similarly to [28] the linear function $W _ { n }$ has been replaced with a single layer transformer. In the experiments we use the outputs of the context network as the speech embedding.

As described in [28] contrastive loss is particularly effective when the set X of negative examples are taken from the same speaker as the positive one. Using multiple speakers makes the contrastive task solvable only by distinguishing speakers. Using the same speaker for the negative examples forces the model to incorporate speech content into the representations.

## B. Adaptive Domain normalization

Normalizing features tend to make the embeddings more invariant to the speaker and thus, focusing more on the semantic content of the speech. This work applies an adaptive normalization [18] in various settings to see which kind is most resilient to new speakers.

Formally with a set of domains $X _ { 1 } , . . , X _ { D }$ , denote $h _ { t } ^ { l - 1 }$ as the p dimensionnal output of the model’s $l - 1 ^ { \mathrm { t h } }$ layer. In order to reduce the computational cost of the normalization, a non linear transformation is applied such that :

$$
g _ { t } ^ { l - 1 } = \operatorname { t a n h } ( W _ { g } h _ { t } ^ { l - 1 } + b _ { g } )\tag{2}
$$

Where $W _ { g }$ is a $d _ { g } \times p$ weight matrix with $d _ { g } < < p .$ . One can then use a weighted summation of the frames from a domain d. Using a softmax weighting on obtain the following set of weights :

$$
\alpha _ { d , t } = \frac { \exp ( \operatorname* { m e a n } ( g _ { s , t } ^ { l - 1 } } { \sum _ { \tau } \mathcal { k } ^ { \ell } [ \tau \in X _ { d } ] \exp ( \operatorname* { m e a n } ( g _ { s , t } ^ { l - 1 } ) ) }\tag{3}
$$

Then the context vector of the domain d can be computed with :

$$
c _ { d } = \sum _ { t } \alpha _ { d , t } \rlap { / } { \cal { k } } [ \tau \in X _ { d } ] g _ { t } ^ { l - 1 }\tag{4}
$$

One can train 2 linear layers $\mathcal { R } _ { g } ^ { d }  \mathcal { R } ^ { p }$ to output the new scales and biases $\gamma _ { s } ^ { l }$ and $\beta _ { s } ^ { l }$

$$
\begin{array} { r l } & { \gamma _ { s } ^ { l } = W _ { \gamma } ^ { l } c _ { d } + b _ { \gamma } ^ { l } } \\ & { \beta _ { s } ^ { l } = W _ { \beta } ^ { l } c _ { d } + b _ { \beta } ^ { l } , } \end{array}\tag{5}
$$

Finally the normalized output of the layer l − 1 is given by:

$$
\tilde { h } _ { d , t } ^ { l } = \hat { h } _ { d , t } ^ { l - 1 } \gamma _ { d } ^ { l } + \beta _ { d } ^ { l }\tag{6}
$$

We elaborate more on the different settings for the domain choices in the experiments part (Sec V).

## C. Domain Adaptation

Domain adaptation is a task consisting of taking a model pretrained on domains $\mathcal { D } _ { 0 } , ~ . . . \mathcal { D } _ { N - 1 }$ that usually contains a lot of data and trying to apply this model to different domains $\mathcal { D } _ { N } , . . . \mathcal { D } _ { N + N ^ { \prime } - 1 }$ that usually have a limited amount of available data. In speech processing this task is often related to new speakers who have different accent, or talk in different condition.

Even though retraining the model from scratch on the augmented dataset might be the best performing method, this doesn’t allow for fast adaptation to new speakers and is very costly. This work presents a fine tuning approach that aims to quickly adapt to out of domain data with limited computation time and limited data.

## V. EXPERIMENTS

## A. Librispeech Male/Female Experiment

The first experiment conducted to evaluate our methods was to train models on Librispeech considering two domains Male and Female speakers whose labels are available in the metadata. We trained a vanilla CPC model on all female speakers in train-clean to adapt it to male speakers (and conversely). The fine tuning data used consisted of several splits of the female speakers from train-clean, with different amounts of data and speakers. We defined 16 different splits:

As a primary baseline, the fine tuning is performed using only the data from the new domain, this experiment is referred to as the simple baseline. In all other experiments the data from the original domain is also used at fine tuning time. However, the new domain is oversampled so that half of the batches come from a male speaker and the other half from a female speaker. This helps regularize and avoid catastrophic inferences.

TABLE III: ABX score across speakers within domain, on the accented English test set for different domain adaptation method (FT: fine tuning with resampling, DN: domain normalization). We boldface the best results per columns.
<table><tr><td></td><td>e-us</td><td>e-uk</td><td>e-ca</td><td>e-ch</td><td>e-in</td><td> $\overline { { \mathrm { \ e - j a } } }$ </td><td>e-ko</td><td> $\tt e - p t$ </td><td>e-ru</td><td>e-es</td><td>Average</td></tr><tr><td>MFCC</td><td>37.9</td><td>37.3</td><td>35.5</td><td>32.7</td><td>33.5</td><td>36.2</td><td>32.6</td><td>35.0</td><td>35.2</td><td>35.6</td><td> $3 5 . 1 \pm 2$ </td></tr><tr><td>AESRC</td><td>16.1</td><td>14.9</td><td>22.4</td><td>16.0</td><td>17.1</td><td>20.1</td><td>21.9</td><td>15.5</td><td>18.6</td><td>16.3</td><td> $1 7 . 9 \pm 2$ </td></tr><tr><td>LS pretrain</td><td>13.1</td><td>13.2</td><td>20.3</td><td>14.2</td><td>15.4</td><td>19.4</td><td>20.6</td><td>13.7</td><td>16.5</td><td>14.7</td><td> $1 6 . 1 \pm 3$ </td></tr><tr><td>FT (single)</td><td>12.9</td><td>12.9</td><td>20.6</td><td>13.9</td><td>16.0</td><td>10.9</td><td>20.4</td><td>13.6</td><td>13.0</td><td>12.7</td><td> $\overline { { 1 4 . 7 \pm 3 } }$ </td></tr><tr><td>FT (joint)</td><td>12.3</td><td>12.4</td><td>20.1</td><td>13.4</td><td>11.7</td><td>10.5</td><td>19.9</td><td>13.4</td><td>12.2</td><td>12.1</td><td> $1 3 . 8 \pm 3$ </td></tr><tr><td>DN + FT (single)</td><td>13.0</td><td>13.3</td><td>20.3</td><td>12.8</td><td>11.0</td><td>9.8</td><td>20.0</td><td>18.2</td><td>9.9</td><td>11.7</td><td> $1 4 . 0 \pm 4$ </td></tr><tr><td> $\mathrm { D N } + \mathrm { F T }$  (joint)</td><td>13.2</td><td>11.8</td><td>18.2</td><td>9.0</td><td>11.7</td><td>9.7</td><td>13.4</td><td>12.1</td><td>12.1</td><td>11.6</td><td> ${ \bf 1 2 . 3 \pm 2 }$ </td></tr></table>

The average CPC accuracy is the validation metric used for the early stopping. Empirical experiments show that it is correlated with the ABX score. As can be seen in the simple baseline, the performance on the original domain tend to drop significantly while training on the new one. This leads to very early training stop if there is not enough data to make the increased performance on the new domain to compensate for the drop on the original domain. On the contrary using the original domain during fine tuning with even sampling keeps the performance on females very stable which helps the model perform more epochs while increasing the average accuracy. This experiment shows the benefits of keeping data from the original domain when adapting to a different one. From this observation, all the later experiment use this resampling strategy.The last two columns describe the effect of the domain normalization. The new domain statistics stored into the domain normalization layer are initialized with the statistics of the original domain. With this normalization, fine tuning all the weights causes the model to drop to drop significantly in performance in the first steps losing the benefits of the pretraining and resulting in very slow convergence comparable to retraining from scratch. To deal with this issue, we included a warm up fine tuning stage were the weights of the network are frozen, only speaker statistic and weights $W _ { g } , W _ { \beta } , W _ { \gamma }$ are updated. Results from both ABX tests demonstrate that more data is needed to benefit from only updating the domain normalization. The last column describes the performance of the model after continuing training of previous models with weights unfrozen. As can be observed, this fine tuning framework outperforms other methods in every case. We even report increased performance while using 8x less data for the new domain. One limitation of this approach is that it requires having the domain label as input. In the Male/Female case, this is not an issue since gender is known to be easily obtainable (e.g. with a linear classifier over MFCC features). However this issue is much more relevant in the case of accented speech when unaware of the utterance’s domain.

## B. ABX-Accent Adaptation experiments

This section presents domain adaptation experiments on ABX-Accent. We used the best methods from section V-A and apply them in our benchmark. We considered Librispeech clean to be the original domain on which models are pretrained. The models are then fine tuned with the accented speech data. During this fine tuning phase Librispeech samples are sampled according to our resampling strategy. We provide on our github page all necessary material to construct the same splits used in our experiments.

Table III sums up ABX scores of the benchmark data over some of the accents from ABX-Accent, ABX across refers to ABX calculated across speakers from the same accent. AESRC Training refers to straightforward CPC training on ABX-Accent without initial pretraining, using pretraining on Librispeech significantly boost performances even in the simplest experiments. Experiments referred as ”joint” are performed over all the accents i.e. there is a single model per line. On the contrary other line report results from different model for each accent. The resampling strategy was used in all the fine tuning experiments. In the single domain finetuning experiments, sample alternate between librispeech samples and samples from the accent. In the ”joint” experiments we sample in turn from the eleven domain (ten accents and librispeech).

Similarly to the preliminary experiments the domain normalization warm up followed by complete fine tuning tends to have the best performances. In average we found that models specialised on a specific domain achieve better results. However in some cases we report that there can be benefits to have more data even from different domains. It can be observed that using all the data available overall improve the model performances on the several accents. Even though some for some accents (e.g. Russian) the specialized model fine tuned only with one domain achieves better ABX score than the model trained on the complete training set.

## VI. CONCLUSION

We presented a new accent adaptation benchmark for selfsupervised speech representation learning which we tested on baseline systems based on an adaptive normalization method. We demonstrated with a toy male/female domain split on librispeech that this method has the potential of improving ABX scores by about 33% relative. Applied to the accent dataset, it show improvements of 23% relative, suggesting that the accent adaptation problem is more difficult and that there is room for improvement. We hope that this benchmark will trigger further work in domain adaptation.

## ACKNOWLEDGMENTS

This work was performed using HPC resources from GENCI-IDRIS (Grant 2024-AD011014739R1) and was supported in part by the Agence Nationale pour la Recherche (ANR-17-EURE-0017 Frontcog, ANR10-IDEX-0001-02 PSL\*). ED and MK in their EHESS roles were funded by an ERC grant (InfantSimulator). Views and opinions expressed are those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

## REFERENCES

[1] S. Yang, P. Chi, Y. Chuang, C. J. Lai, K. Lakhotia, Y. Y. Lin, A. T. Liu, J. Shi, X. Chang, G. Lin, T. Huang, W. Tseng, K. Lee, D. Liu, Z. Huang, S. Dong, S. Li, S. Watanabe, A. Mohamed, and H. Lee, “SUPERB: speech processing universal performance benchmark,” CoRR, vol. abs/2105.01051, 2021. [Online]. Available: https://arxiv.org/abs/2105.01051

[2] A. Mohamed, H.-y. Lee, L. Borgholt, J. D. Havtorn, J. Edin, C. Igel, K. Kirchhoff, S.-W. Li, K. Livescu, L. Maaløe et al., “Self-supervised speech representation learning: A review,” IEEE Journal of Selected Topics in Signal Processing, 2022.

[3] K. Lakhotia, E. Kharitonov, W. Hsu, Y. Adi, A. Polyak, B. Bolte, T. A. Nguyen, J. Copet, A. Baevski, A. Mohamed, and E. Dupoux, “Generative spoken language modeling from raw audio,” CoRR, vol. abs/2102.01192, 2021. [Online]. Available: https://arxiv.org/abs/2102. 01192

[4] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour, “Audiolm: a language modeling approach to audio generation,” 2022. [Online]. Available: https://arxiv.org/abs/2209.03143

[5] T. Schatz, “ABX-Discriminability Measures and Applications,” Theses, Universite Paris 6 (UPMC), Sep. 2016. [Online]. Available: https:´ //hal.science/tel-01407461

[6] E. Dunbar, J. Karadayi, M. Bernard, X.-N. Cao, R. Algayres, L. Ondel, L. Besacier, S. Sakti, and E. Dupoux, “The zero resource speech challenge 2020: Discovering discrete subword and word units,” 2020. [Online]. Available: https://arxiv.org/abs/2010.05967

[7] E. Dunbar, M. Bernard, N. Hamilakis, T. A. Nguyen, M. de Seyssel, P. Roze, M. Rivi´ ere, E. Kharitonov, and E. Dupoux, “The zero resource\` speech challenge 2021: Spoken language modelling,” 2021. [Online]. Available: https://arxiv.org/abs/2104.14700

[8] B. Varadarajan, S. Khudanpur, and E. Dupoux, “Unsupervised learning of acoustic sub-word units,” in Proceedings of ACL-08: HLT, Short Papers, 2008, pp. 165–168.

[9] A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” arXiv preprint arXiv:1807.03748, 2018.

[10] Y.-A. Chung, W.-N. Hsu, H. Tang, and J. Glass, “An unsupervised autoregressive model for speech representation learning,” arXiv preprint arXiv:1904.03240, 2019.

[11] S. Schneider, A. Baevski, R. Collobert, and M. Auli, “wav2vec: Unsupervised pre-training for speech recognition,” arXiv preprint arXiv:1904.05862, 2019.

[12] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12 449– 12 460, 2020.

[13] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.

[14] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “Wavlm: Large-scale self-supervised pretraining for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[15] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural´ audio compression,” arXiv preprint arXiv:2210.13438, 2022.

[16] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “Soundstream: An end-to-end neural audio codec,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2021.

[17] M. Riviere and E. Dupoux, “Towards unsupervised learning of speech\` features in the wild,” in 2021 IEEE Spoken Language Technology Workshop (SLT), 2021, pp. 156–163.

[18] F. Ding, W. Guo, B. Gu, Z.-H. Ling, and J. Du, “Adaptive speaker normalization for ctc-based speech recognition.” in INTERSPEECH, 2020, pp. 1266–1270.

[19] Z. Meng, J. Li, Y. Gaur, and Y. Gong, “Domain adaptation via teacher-student learning for end-to-end speech recognition,” 2020. [Online]. Available: https://arxiv.org/abs/2001.01798

[20] S. Sun, B. Zhang, L. Xie, and Y. Zhang, “An unsupervised deep domain adaptation approach for robust speech recognition,” Neurocomputing, vol. 257, pp. 79–87, 2017, machine Learning and Signal Processing for Big Multimedia Analysis. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231217301492

[21] Z. Meng, Z. Chen, V. Mazalov, J. Li, and Y. Gong, “Unsupervised adaptation with domain separation networks for robust speech recognition,” in 2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, dec 2017. [Online]. Available: https://doi.org/10.1109%2Fasru.2017.8268938

[22] X. Shi, F. Yu, Y. Lu, Y. Liang, Q. Feng, D. Wang, Y. Qian, and L. Xie, “The accented english speech recognition challenge 2020: open datasets, tracks, baselines, results and methods,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 6918–6922. [Online]. Available: https://arxiv.org/abs/2102.10233

[23] E. Dunbar, J. Karadayi, M. Bernard, X.-N. Cao, R. Algayres, L. Ondel, L. Besacier, S. Sakti, and E. Dupoux, “The zero resource speech challenge 2020: Discovering discrete subword and word units,” 2020. [Online]. Available: https://arxiv.org/abs/2010.05967

[24] E. Dunbar, M. Bernard, N. Hamilakis, T. A. Nguyen, M. de Seyssel, P. Roze, M. Rivi´ ere, E. Kharitonov, and E. Dupoux, “The zero resource\` speech challenge 2021: Spoken language modelling,” 2021. [Online]. Available: https://arxiv.org/abs/2104.14700

[25] A. van den Oord, Y. Li, and O. Vinyals, “Representation learning with contrastive predictive coding,” 2019. [Online]. Available: https: //arxiv.org/abs/1807.03748

[26] M. Riviere, A. Joulin, P.-E. Mazar\` e, and E. Dupoux, “Unsupervised´ pretraining transfers well across languages,” 2020. [Online]. Available: https://arxiv.org/abs/2002.02848

[27] E. Kharitonov, M. Riviere, G. Synnaeve, L. Wolf, P.-E. Mazar \` e,´ M. Douze, and E. Dupoux, “Data augmenting contrastive learning of speech representations in the time domain,” 2020. [Online]. Available: https://arxiv.org/abs/2007.00991

[28] M. Riviere, A. Joulin, P.-E. Mazar \` e, and E. Dupoux, “Unsupervised´ pretraining transfers well across languages,” 2020.