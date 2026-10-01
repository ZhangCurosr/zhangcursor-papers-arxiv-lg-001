# GHOST IN THE ENCODER: DECODABLE ARTIST IDENTITYREPRESENTATIONS IN LYRICS-TO-SONG GENERATION

Arhan Vohra\*

Choenden Kyirong\*

Laura Ibáñez-Martínez

Martín Rocamora

Music Technology Group, Universitat Pompeu Fabra

arhan.vohra01@estudiant.upf.edu, choendenkyirong@gmail.com

\*Equal contribution

## ABSTRACT

Text-to-song generation models can be prompted to imitate specific artists or regurgitate entire songs from their training data. Although these phenomena have been documented behaviorally on small datasets, little is known about the internal representations that may give rise to them. Prior interpretability work on generative audio has focused on locating semantic concepts such as genre or time signature within model activations. In this work, we show that a trained model can be probed for linearly decodable representations of artist identity from song lyrics alone, without any additional identifiers. Through a controlled case study of ACE-Step 1.5 spanning 2,000 songs across 100 artists, we demonstrate that the artist associated with a given set of lyrics can be identified within the model’s internal activations, and that this conditioning signal propagates from the lyric encoder to the diffusion backbone during inference. These findings indicate that lyrics constitute an artist-level conditioning channel not addressed by prompt-side replication safeguards. More broadly, our work highlights how latent-space analysis can be used to audit what generative music models have implicitly learned from their training data.

## 1. INTRODUCTION

Consider the challenge of teaching music students to appreciate the works of Bob Dylan. Perhaps we’d like for them to learn from Dylan how to craft songs imaginatively across many genres while preserving their voice as songwriters. Unfortunately, some students may just memorize every lyric from “Highway 61 Revisited”, and others might incorporate crude impressions of Dylan’s vocal delivery and harmonica solos in their own songs. The difficulty is that all of these outcomes require exposure to the same body of work, and the line between generalization, memorization, and impersonation is drawn not by what was heard, but by what was learned from it.

Recent studies have shown that AI music generation models exhibit all three of these behaviors to varying extents: song-level memorization [1, 2], artist-level impersonation [3,4], and concept-level generalization of features such as genre and mood [5–8].

While generative models can be probed mechanistically to reveal interpretable representations of musical semantics (e.g., time signature) and specific genres, their tendency to replicate artist-level characteristics has only been documented at the behavioral level. Coelho [3] presented descriptive evidence that closed-source models can be conditioned to imitate certain artists via metataginformed prompting, while Nagarajan & Dong [4] used LLM-generated descriptors and CLAP-based similarity to test artist conditioning on a limited two-artist dataset with MusicGen. Roh et al. [1] showed that full-song replication can also be elicited by prompting models with the lyrics from popular songs, and that this behavior is preserved after homophonic replacement of the song’s lexical content (“Bob’s confetti” from “mom’s spaghetti”). However, song generation is approached as a black-box process in all three of these accounts, and evaluation has been limited to fewer than 50 songs.

Our study directly addresses the need for latent-space explanations expressed in prior work [1] by proposing a mechanistic approach to understanding how music generation models develop artist-specific representations from lyrics alone, using linear probing [9] across the conditioning pipeline. We demonstrate its effectiveness with a systematic case study on ACE-Step 1.5 [10], a state-ofthe-art lyrics-to-song model with open weights and relatively modest compute requirements. Our class-balanced dataset spans 2,000 songs from 100 artists across 5 genres, namely: pop, rock, hip-hop, R&B, and country. To the best of our knowledge, this is the first interpretability study of artist representations in music generation.

We show that the lyric encoder in ACE-Step 1.5 produces highly decodable linear representations of artist identity without receiving any explicit artist information in its input. We define artist identity as the artist label associated with a song’s lyrics. This is a functional definition rather than a perceptual feature of music, allowing us to show that artist-identifying representations exist within a music model without making assumptions about their relationship to the acoustic output.

The representations remain linearly decodable through the model’s cross-attention layers into the denoising process and in the generated audio itself. We examine how decodability varies with artist genre, lexical identifiers, as well as the text-conditioning pipeline, and consider how our findings may inform future work on artist-level representations.

## 2. RELATED WORK

## 2.1 Replication and Memorization in GenAI

Memorization and replication are well documented phenomena in generative models across modalities. Language models have been shown to leak verbatim text sequences from their training data in response to prompt-based attacks, including private information about individuals [11] and copyrighted books in their entirety [12]. Image generation models are similarly capable of producing near-perfect replicas of their training data [13]. This behavior is known to be amplified by the presence of duplicate entries in the training set [14], and can be triggered via “highly specific captions” seen during training [15].

This propensity for generative models to regurgitate their training data has also been observed in music models. Some open-source model releases, such as YuE [2] and MusicGen [16], include memorization studies to quantify this phenomenon, and ACE-Step warns users of the potential for “unintentional copyright infringement” on its GitHub page [17]. Recently, Roh et al. [1] established that song replication can be triggered by prompting YuE with verbatim song lyrics, and that prompt-side guardrails to prevent this in closed-source models like Suno can be bypassed by modifying the lyrics with semantically different text sequences that maintain the original’s phonetic structure.

Other studies have explored the use of descriptive prompts to elicit artist imitation without explicitly mentioning their names, in order to circumvent prompt filtering. Coelho [3] proposed the use of tags sourced from publicly available metadata to curate list-like artist prompts, with anecdotal evidence that this reproduces their idiosyncratic stylistic choices during generation. Additionally, Nagarajan & Dong [4] described a process to generate policy-compliant artist prompts using LLMs, and used audio embeddings to compare the generated audio against a reference track.

While a broader body of work has addressed training data attribution and anti-memorization interventions in generative audio models [18–21], the mechanisms that may enable artist-level imitation and song-level replication have not yet been dissected.

## 2.2 Lyrics-to-song Models

Current lyrics-to-song systems are composed of several components with distinct responsibilities, including pretrained text encoders, audio codecs, alignment modules, and generative backbones [2,22–24]. The generative backbones can follow autoregressive [2,22,25], diffusion-based [23,26], or hybrid [24,27] paradigms. Across all three, text content enters the generative process through pretrained encoders that may be jointly fine-tuned with the rest of the model.

We study ACE-Step 1.5 [10], a hybrid model in which a Qwen-based language model generates structured Chainof-Thought metadata while a Diffusion Transformer (DiT) synthesizes audio. The DiT receives conditioning through three parallel pathways merged via cross-attention: a caption encoder (Qwen3-0.6B) [28], a timbre encoder (if reference audio is provided), and a lyric encoder based on an 8-layer transformer and Qwen3 token embeddings. Notably, ACE-Step 1.5 generates songs significantly faster than other open-weight models of comparable quality, making it a particularly feasible choice for our case study.

## 2.3 Interpretability of Music Generation Models

The linear representation hypothesis [29] posits that highlevel concepts can be captured by linear directions in a vector space, and probing the hidden layers of a model [9] can reveal which concepts it has internalized. Further, these can be used to quantify and prevent adverse behavior such as gender bias in language models [30].

Wei et al. [8] probed MusicGen for music-theoretical concepts, finding that certain layers encode these properties more strongly than others, while SMITIN [7] also leveraged probing to identify and steer attention heads responsible for traits such as instrument presence. Most recently, Singh et al. [5] discovered interpretable features from MusicGen’s residual stream with sparse autoencoders, and TADA [6] demonstrated that musical concept representations can be localized to specific layers of diffusion models.

Our study draws from this line of inquiry to investigate how text-based conditioning relates to artist identity representations in ACE-Step 1.5, and whether this relationship can functionally explain the imitation and replication behavior observed in prior work.

## 3. METHOD

## 3.1 Dataset

We construct our prompting dataset by retrieving Englishlanguage song lyrics from the top 100 artists equally distributed across 5 genres (rock, pop, hip-hop, R&B, and country) on Last.fm, as of April 2026. For each artist, we fetch their top 20 songs from the Genius.com API (ranked by the API’s internal popularity metric), excluding variations (remixes, covers, edits) and songs released after 2024. We anonymize any appearances of a given artist’s own name in their lyrics by using the phonetic replacement method described in [1]. Since ACE-Step 1.5’s 1.8M-song training corpus is undocumented, training-set membership cannot be verified directly; our selection of popular pre-2024 songs is intended to maximize its likelihood.

## 3.2 Prompting and Activation Extraction

To observe how ACE-Step encodes and propagates artist information during inference, we insert pytorch hooks into intermediate layers while generating songs conditioned on the datasets’ lyrics, storing the activations for probing.

<table><tr><td>Conditioning input</td><td>no-CoT</td><td>with-CoT</td></tr><tr><td>Lyric encoder</td><td>lyrics</td><td>lyrics</td></tr><tr><td>Caption</td><td>null</td><td>generated from lyrics</td></tr><tr><td>BPM / key</td><td>null</td><td>generated from lyrics</td></tr><tr><td>Timbre</td><td>null</td><td>null</td></tr></table>

Table 1. Conditioning inputs to the DiT under the two inference configurations. Both pass the same lyrics to the lyric encoder; $\mathtt { w i t h - C o T }$ additionally activates the chain-of-thought module, which derives a caption and infers BPM/key. All other inputs are held fixed.

The ACE-Step 1.5 generation pipeline contains encoders for several conditioning pathways (text caption, lyrics, timbre transfer) which are merged together into the DiT’s cross-attention mechanism. This model also offers a unique conditioning pathway through its chain-of-thought (CoT) module, which can be used to generate a relevant song caption, as well as tempo and key conditioning, based on a lyric prompt alone. In order to understand how the CoT process modifies artist encoding, we consider two configurations of the inference process: no-CoT (where the reasoning model is deactivated) and with-CoT. The key differences are shown in Table 1.

All experiments use acestep-v15-turbo, called with the following inputs for consistency: duration=180s, bpm=None, key=None, text\_caption=None, timbre\_reference=None. In the with-CoT configuration, the tempo, key, and text caption inputs are overridden by the CoT model.

In both configurations, we pass in song lyrics from the dataset and extract activation tensors from the following locations:

Conditioning stream outputs (R<sup>2048</sup>): token-level outputs of the lyric encoder, text/caption stream, (unused) timbre transfer stream, and the merged conditioning sequence passed to the DiT.

Lyric encoder hidden layers $\left( \mathbb { R } ^ { 2 0 4 8 } \right)$ : token-level hidden states from the 8 lyric-encoder transformer layers.

DiT cross-attention K/V projections (R<sup>1024</sup>): key and value projections of the merged conditioning sequence, extracted at 5 equally-spaced pairs of DiT layers (10 layers total). This accounts for the model’s hybrid architecture, where alternating layers use sliding-window (local) or group-query (global) attention.

DiT audio-latent states (R<sup>2048</sup>): pre-cross-attention states, cross-attention outputs, and post-cross-attention states, extracted at the same 10 DiT layers across 8 denoising timesteps.

Together, these probes trace artist-identifying information from lyric encoding, through merged conditioning and cross-attention, into the DiT denoising trajectory.

## 3.3 Internal Probing

For each hook point h, we quantify the extent to which artist identity is linearly decodable [9]. We measure this using the accuracy of a multi-class classifier trained on h.

Let $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be a dataset, where $x _ { i }$ is an input example and $y _ { i } \in \mathcal { V }$ is the corresponding label. Let $H _ { h } ( x _ { i } )$ denote the activation tensor captured at hook point h.

Since activations may have non-feature axes (e.g., time or token dimensions), we convert each activation tensor into a single feature vector by averaging over all nonfeature axes. This yields

$$
a _ { i } = \mathrm { m e a n } ( H _ { h } ( x _ { i } ) ) \in \mathbb R ^ { d _ { h } } ,
$$

where the mean is taken over all axes except the final feature dimension. For example, if $H _ { h } ( x _ { i } ) \in \bar { \mathbb { R } } ^ { T \times d _ { h } }$ , then

$$
a _ { i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } H _ { h } ( x _ { i } ) _ { t } .
$$

For each cross-validation fold k, we train a linear multiclass logistic regression classifier on the corresponding training split $T _ { k }$ :

$$
p _ { \phi ^ { ( k ) } } \left( y \mid a _ { i } \right) = \mathrm { s o f t m a x } \Big ( W ^ { ( k ) } a _ { i } + b ^ { ( k ) } \Big )
$$

where $\phi ^ { ( k ) } = \{ W ^ { ( k ) } , b ^ { ( k ) } \}$ . The training objective for each fold is the cross-entropy loss evaluated on $T _ { k }$

## 3.4 Output-side Probing

While probing can establish that intermediate activations contain linear representations of artist identity, we need a scalable metric to detect artist identity in the generated audio. We use LAION-CLAP and MuQ-MuLan embeddings [31,32] to analyze whether artist identity is decodable from the generated output. This approach is well-supported by previous studies on concept-based interpretability and similarity in audio, which use CLAP-like models as a quantitative proxy to measure semantic or musical alignment [5–7], and recently, artist similarity in music generation [4]. Mirroring the probing approach described earlier, we extract one embedding per generated song and train a linear classifier on the embeddings to test whether artist information remains detectable in the audio output.

Additionally, we compare ACE-Step’s generated audio to the corresponding original recordings with songlevel similarity scores from CLEWS-DVI [33], a versionmatching embedding model, in order to understand whether artist decodability may be related to song-level replication. This follows the use of version-matching models in systematic studies of replication [2].

## 4. EXPERIMENTS

## 4.1 Probing the Conditioning Encoders

RQ1: Does the lyric encoder develop linearly decodable representations of artist identity?

![](images/b5eab323a21eda4637fedcf06f69365453c1ac4289dd30a132af55a1210a5047.jpg)  
Figure 1. Artist probe accuracy rises through the lyric encoder layers. Error bars show the 95% confidence interval.

We prompt ACE-Step 1.5 with song lyrics over our entire 2,000 song dataset and extract activations from each layer of the lyric encoder as previously described. We report held-out accuracy from the probes after 5-fold cross validation and permutation tests $( n = 1 0 0 0 )$ . These results are compared against identically structured probing experiments on two baselines: (1) the pre-trained Qwen3 token embeddings used as the encoder’s input, and (2) bagof-words (BoW) based on TF-IDF.

A probe trained on the pooled output of the lyric encoder correctly identifies the source artist with an accuracy of 0.344, approximately 4 times the BoW baseline (0.085), 35 times higher than random chance, and well outside the null distribution $( p < 0 . 0 0 1 )$ .

Artist-discriminative representations rise through the first five layers of the encoder and plateau thereafter, peaking at layer 5 (0.361) before stabilizing between 0.34–0.36 in layers 5–7. The pre-trained Qwen3 embedding produces an artist probe accuracy of 0.153; although artist decodability at the encoder’s intake is already better-thanrandom, the learned layers amplify this substantially (Figure 1).

We also find that genre identity is decodable from the lyric encoder output with an accuracy of 0.764, raising the question of whether the artist probe is detecting meaningful artist-specific signals or is confounded by genre-level representations.

## RQ2: Is the probe detecting artist or genre identity?

To rule out genre as a confounding factor, we probe artist identity within each genre subset of our data. This tests whether the probe captures individual artists or merely the vocabularies and songwriting norms of their genres.

Figure 2 shows that mean artist decodability at the encoder varies substantially across genres, from 0.585 for hip-hop down to 0.338 for country. Hip-hop is also the genre where the TF-IDF baseline is strongest (0.403), suggesting that lexical distinctiveness between artists accounts for a larger share of the signal here than in other genres, where probing the encoder is over twice as accurate as bag-

![](images/6f5d9402e3f82f1f890798bbd4505853fc339397abe6cb6a530def2035aa8ea4.jpg)  
Figure 2. Genre-wise mean artist decodability from the lyric encoder’s output.

of-words.

## 4.2 Probing Downstream Generation

## RQ3: Does the DiT preserve artist-level decodability during generation?

To understand if the Diffusion Transformer actually retains the learned artist representations, we probe the crossattention K/V projections and audio-token hidden states during denoising. The generator does not receive any style caption or metadata (no-CoT).

Our probe reveals that artist identity remains strongly decodable, albeit slightly attenuated by the cross-attention interface, at 0.292 on average at the K/V projections (a 15% decrease from 0.344 at the lyric encoder). Individual artist decodability is highly correlated between the lyric encoder and cross-attention probes $( \rho \approx 0 . 9 0 , r \approx 0 . 9 1 )$ , suggesting that the most decodable artists are similar at both points.

In the audio-token hidden states (Figure 3), we see an emerging pattern in how the model attends to these representations during the diffusion process. The noisy latent in layer 0 has a probe accuracy of exactly 0.01 before denoising, matching random chance classification between 100 artists. In the first layers, especially layers 0 and 1, artist decodability sharply increases before and after being reweighted by cross-attention. From layers 12-13, artist decodability is generally lower (0.10-0.14 in the last layers), with gradual decay over consecutive time steps.

Prior work has suggested that concept representations can be localized to specific layers of generative music models [6], and that diffusion models may internalize or “lock in” different concepts at different steps in the denoising process [34]. The early layers may be re-encoding artist information into acoustic structure, developing non-linear representations, or discarding it entirely. While a causal evaluation of these representations would be required to positively distinguish amongst these possibilities, we examine the model’s audio outputs in our final experiments (RQ4-5) as an end-to-end behavioral check on what survived the generation process.

![](images/320ffaa4fc44a29c8a079442280c06a31fe935affe763c9e052db21fc2ad4bd3.jpg)

![](images/d115b4629bbae6a7635ca83ed459be23fea2184bca0c68f954531cd5fac3667f.jpg)

![](images/91f228a497776124079a377c7bbf8b39f0aff1083339b2f393baa93af386c8e6.jpg)  
Figure 3. Cross-attention injects decodable artist representations into the first few layers of the DiT audio-token hidden states; these become progressively less decodable in deeper layers.

RQ4: Is artist identity detectable in the generated audio?

Our internal probes establish that artist identity is linearly decodable at successive stages of ACE-Step’s conditioning pipeline, but this does not confirm that the signal reaches the generated output. We test this by extracting MuQ-MuLan and CLAP embeddings from the generated audio and train the same probing classifier on these embeddings.

Across all 100 artists, the MuQ-MuLan probe achieves a mean accuracy of 0.109 under no-CoT conditioning. The same probe applied to original song recordings reaches 0.449, indicating that the generation process may preserve some artist-discriminative information. The CLAP-based probe shows considerably lower artist decodability (0.043) overall, which may relate to MuQ’s musicspecific pretraining.

In order to control for genre, we repeat the probe for each genre class (Table 2) showing that the artist signal survives genre controls in all five genres to different degrees, with MuQ-MuLan accuracy ranging from 0.128 (pop) to 0.208 (rock) at a chance level of 0.05. The genre ranking at the audio level partially diverges from the encoder: rock artists are more decodable than hip-hop artists on average. This suggests that the features driving encoder-level decodability may differ from those that survive into the generated audio. However, since neither embedding model has access to the input lyrics, any artist information recovered by the probe must still be present in the acoustic properties of the generated audio itself due to the conditioning of the lyric encoder.

These results confirm that artist-specific conditioning persists through the full generation pipeline, from lyrics through internal representations to the acoustic properties of the generated audio, and that this signal operates at the artist level rather than reflecting song specific memorization alone.

RQ5: How does CoT conditioning modify artist rep-

## resentations in the generation pipeline?

The chain-of-thought LM is somewhat unique to ACE-Step 1.5: it is simultaneously responsible for optimizing and formatting the user’s inputs, generating missing conditions (text caption, tempo, key), and subsequently producing a text embedding based on them, which is used as latent conditioning for the DiT. We broadly assess how this module may impact artist identity representations.

In our no-CoT experiments, the text caption embeddings probed artist identity at near-chance accuracy (0.008), due to this conditioning stream being inactive. When the LM is turned on, artist identity probes at 0.069 at the same point, which is higher, but still below our baseline TF-IDF lyric classifier (0.085). We also find that a TF-IDF classifier trained on the LM-generated caption has a comparable probing accuracy to the one trained on lyrics (0.073), suggesting that any caption decodability may be lexically-grounded and not necessarily learned from training.

Since the lyric encoder is deterministic and works in parallel to the CoT module, probe accuracy here remains unchanged across both conditions. However, decodability drops from 0.344 at the lyric encoder to 0.085 at the merge point where the LM’s embeddings are concatenated with the lyric encoder output, and consequently, the K/V projections (0.292 → 0.089). Still, this may reflect the dilution of the lyric stream’s per-token signal under mean-pooling rather than a loss of artist information in the merged sequence.

We also compare the audio outputs produced under both conditions using CLAP/MuQ-MuLan classification and cover identification Retrieval@K via CLEWS. The songs produced by with-CoT are harder to probe for artist identity than those from no-CoT (Table 2), which would suggest that retained artist-level conditioning in the audio is weaker when an LM-generated caption is present. Yet, it appears that the audio generated with an LM-caption is more likely to be identified as a cover of the original recording by CLEWS, which is trained to recognize compositional similarities between song versions [33].

<table><tr><td></td><td colspan="3">MuQ-MuLan</td><td colspan="3">CLAP</td></tr><tr><td></td><td>Originals</td><td>no-CoT</td><td>with-CoT</td><td>Originals</td><td>no-CoT</td><td>with-CoT</td></tr><tr><td>Full dataset (100 artists, chance = 0.01) All genres</td><td>0.449</td><td>0.109</td><td>0.101</td><td>0.224</td><td>0.043</td><td>0.042</td></tr><tr><td>Within-genre (20 artists, chance = 0.05)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hip-hop</td><td>0.445</td><td>0.180</td><td>0.183</td><td>0.273</td><td>0.115</td><td>0.130</td></tr><tr><td>Rock</td><td>0.659</td><td>0.208</td><td>0.130</td><td>0.394</td><td>0.108</td><td>0.073</td></tr><tr><td>R&amp;B</td><td>0.510</td><td>0.165</td><td>0.140</td><td>0.250</td><td>0.083</td><td>0.100</td></tr><tr><td>Country</td><td>0.558</td><td>0.145</td><td>0.128</td><td>0.368</td><td>0.088</td><td>0.053</td></tr><tr><td>Pop</td><td>0.413</td><td>0.128</td><td>0.100</td><td>0.300</td><td>0.073</td><td>0.068</td></tr></table>

Table 2. Artist probe accuracy on generated audio embeddings. All results $p < 0 . 0 0 1$

<table><tr><td>K</td><td> $\mathtt { w i t h - C o T }$ </td><td> $\scriptstyle \mathrm { n o - C o T }$ </td><td>Chance</td></tr><tr><td>10</td><td>0.055</td><td>0.022</td><td>0.005</td></tr><tr><td>25</td><td>0.088</td><td>0.045</td><td>0.013</td></tr><tr><td>50</td><td>0.137</td><td>0.073</td><td>0.025</td></tr><tr><td>100</td><td>0.210</td><td>0.131</td><td>0.050</td></tr><tr><td>200</td><td>0.304</td><td>0.204</td><td>0.100</td></tr><tr><td>400</td><td>0.449</td><td>0.333</td><td>0.200</td></tr></table>

Table 3. CoverID Retrieval@K via CLEWS. Random $\mathrm { b a s e l i n e } = K / N$ where $N = 2 { , } 0 0 0$

We provide listening examples <sup>1</sup> of songs from each genre where CLEWS correctly retrieved the original recording from its with-CoT generated version (R@1). In informal listening, it appears that songs generated with-CoT bear noticeably more musical and structural resemblance to the original tracks than no-CoT, but seemingly deviate from the expected vocal timbre, gender, or instrumentation, as a result of the auto-generated caption. Perhaps ACE-Step is reproducing covers of popular songs from its training data, in which case, a version-matching embedding like CLEWS would be invariant to these differences. As expected, the no-CoT songs are less varied in their style/timbre overall.

The dissociation between CoT conditions in artistprobe accuracy, cover-retrieval rate, and our exploratory perceptual observations, suggests the LM caption may function as a song-recognition trigger, surfacing trainingdata reproductions that lyric conditioning alone does not. We leave systematic investigation of this pathway to future work.

## 5. CONCLUSION

We demonstrate that the lyric encoder in ACE-Step 1.5 contains linear representations of artist identity that can be retrieved without any explicit artist information. These representations persist through the Diffusion Transformer’s cross-attention interface and are injected into audio-token hidden states at the early layers. Using lexical classification and transformer-based embeddings as baselines, we show that this signal is acquired during pretraining rather than inherited from the input encoder, and that it is modulated by ACE-Step’s CoT caption pathway in ways that warrant further study.

Our contribution highlights the potential for mechanistic investigations of learned artist representations in music models. While our probing study suggests possible explanations for replication patterns, the potential for intervention via causal mechanisms remains to be explored.

Previous studies exposed conditioning exploits in generative models that could bypass replication safeguards by triggering memorization [1, 3, 4]. Our work locates a conditioning channel that can identify which artist’s work the model is attempting to imitate during the generative process, which may be an interpretable form of leakage that explains these exploits.

Future interpretability work in this domain may include studying the role of style captions in exposing artist representations [4], or extending our probing approach to other models and architectures, alongside perceptual evaluation. More broadly, we encourage others to explore latent-space auditing as a means of understanding behavioral patterns in music models at a representational level.

## 6. AI USAGE STATEMENT

Claude and Claude Code were used to help edit and grammar check the paper, clean up code, and create the supplementary website.

## 7. ACKNOWLEDGEMENTS

This work is supported by the "Cátedra IA y Música" project (TSI-100929-2023-1), funded by the Secretaría de Estado de Digitalización e Inteligencia Artificial, the European Union-Next Generation EU funds and BMAT Music Innovators. And by the "IMPA" project (PID2023-152250OB-I00) funded by MCIU/AEI/10.13039/501100011033/FEDER, UE.

## 8. REFERENCES

[1] J. Roh, Z. Novack, Y. Peng, N. Mireshghallah, T. Berg-Kirkpatrick, and A. Houmansadr, “Bob’s confetti: Phonetic memorization attacks in music and video generation,” arXiv preprint arXiv:2507.17937, 2025.

[2] R. Yuan, H. Lin, S. Guo, G. Zhang, J. Pan, Y. Zang, H. Liu, Y. Liang, W. Ma, X. Du et al., “YuE: Scaling open foundation models for long-form music generation,” arXiv preprint arXiv:2503.08638, 2025.

[3] G. Coelho, “The artist is present: Traces of artists residing and spawning in text-to-audio AI,” in Proceedings of the International Conference on AI and Musical Creativity (AIMC), Brussels, Belgium, 2025.

[4] A. Nagarajan and H.-W. Dong, “The name-free gap: Policy-aware stylistic control in music generation,” in Conference on Neural Information Processing Systems: AI for Music Workshop, 2025. [Online]. Available: https://openreview.net/forum?id=ylcGhAt75p

[5] N. Singh, M. Cherep, and P. Maes, “Discovering and steering interpretable concepts in large generative music models,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=mGtEoLYr9j

[6] Ł. Staniszewski, K. Zaleska, M. Modrzejewski, and K. Deja, “TADA! tuning audio diffusion models through activation steering,” in ICLR 2026 Workshop on Representational Alignment (Re^4-Align), 2026. [Online]. Available: https://openreview.net/forum?id= OUux4upENk

[7] J. Koo, G. Wichern, F. G. Germain, S. Khurana, and J. L. Roux, “SMITIN: Self-monitored inference-time intervention for generative music transformers,” IEEE Open Journal of Signal Processing, vol. 6, pp. 266– 275, 2025.

[8] M. Wei, M. Freeman, C. Donahue, and C. Sun, “Do music generation models encode music theory?” in Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024, pp. 680–687. [Online]. Available: https://doi.org/10.5281/ zenodo.14877132

[9] G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” in Proceedings of the International Conference on Learning Representations (ICLR), Workshop Track, 2017.

[10] J. Gong, Y. Song, W. Zhao, S. Wang, S. Xu, and J. Guo, “ACE-Step 1.5: Pushing the boundaries of open-source music generation,” arXiv preprint arXiv:2602.00744, 2026.

[11] N. Carlini, F. Tramèr, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown,

D. Song, U. Erlingsson, A. Oprea, and C. Raffel, “Extracting training data from large language models,” in Proceedings of the 30th USENIX Security Symposium, 2021, pp. 2633–2650.

[12] M. Nasr, N. Carlini, J. Hayase, M. Jagielski, A. F. Cooper, D. Ippolito, C. A. Choquette-Choo, E. Wallace, F. Tramèr, and K. Lee, “Scalable extraction of training data from (production) language models,” in Proceedings ofthe International Conference on Learning Representations (ICLR), 2025.

[13] N. Carlini, J. Hayes, M. Nasr, M. Jagielski, V. Sehwag, F. Tramèr, B. Balle, D. Ippolito, and E. Wallace, “Extracting training data from diffusion models,” in Proceedings of the 32nd USENIX Security Symposium, 2023.

[14] G. Somepalli, V. Singla, M. Goldblum, J. Geiping, and T. Goldstein, “Understanding and mitigating copying in diffusion models,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id= HtMXRGbUMt

[15] ——, “Diffusion art or digital forgery? investigating data replication in diffusion models,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 6048–6058.

[16] J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Défossez, “Simple and controllable music generation,” in Advances in Neural Information Processing Systems, vol. 36, 2023.

[17] ACE-Step Contributors, “ACE-Step 1.5 repository README, version 0.1.7,” GitHub repository documentation, 2026, commit d5d958eebeaf4de20f29bcf122d161dc0397b146, accessed Apr. 25, 2026. [Online]. Available: https://github.com/ace-step/ACE-Step-1.5/blob/ d5d958eebeaf4de20f29bcf122d161dc0397b146/ README.md

[18] J. Barnett, H. F. García, and B. Pardo, “Exploring musical roots: Applying audio embeddings to empower influence attribution for a generative music model,” in Proceedings of the 25th International Society for Music Information Retrieval Conference, 2024, pp. 360–368. [Online]. Available: https: //doi.org/10.5281/zenodo.14877350

[19] W. Choi, J. Koo, K. W. Cheuk, J. Serrà, M. A. Martínez-Ramírez, Y. Ikemiya, N. Murata, Y. Takida, W.-H. Liao, and Y. Mitsufuji, “Large-scale training data attribution for music generative models via unlearning,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems Creative AI Track: Humanity, 2025. [Online]. Available: https://openreview.net/forum?id=qj3ps8lNIf

[20] F. Messina, F. Ronchini, L. Comanducci, P. Bestagini, and F. Antonacci, “Mitigating data replication in textto-audio generative diffusion models through antimemorization guidance,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 15 742–15 746.

[21] D. Bralios, G. Wichern, F. G. Germain, Z. Pan, S. Khurana, C. Hori, and J. L. Roux, “Generation or replication: Auscultating audio latent diffusion models,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024, pp. 1156–1160. [Online]. Available: https://doi.org/10.1109/ICASSP48485.2024.10447705

[22] Z. Liu, S. Ding, Z. Zhang, X. Dong, P. Zhang, Y. Zang, Y. Cao, D. Lin, and J. Wang, “SongGen: A single stage auto-regressive transformer for text-to-song generation,” arXiv preprint arXiv:2502.13128, 2025.

[23] Z. Ning, H. Chen, Y. Jiang, C. Hao, G. Ma, S. Wang, J. Yao, and L. Xie, “DiffRhythm: Blazingly fast and embarrassingly simple end-to-end full-length song generation with latent diffusion,” arXiv preprint arXiv:2503.01183, 2025.

[24] J. Gong, Y. Song, W. Zhao, S. Wang, S. Xu, and J. Guo, “ACE-Step: A step towards music generation foundation model,” arXiv preprint arXiv:2506.00045, 2025.

[25] P. Dhariwal, H. Jun, C. Payne, J. W. Kim, A. Radford, and I. Sutskever, “Jukebox: A generative model for music,” arXiv preprint arXiv:2005.00341, 2020.

[26] Z. Evans, J. Parker, C. Carr, Z. Zukowski, J. Taylor, and J. Pons, “Long-form music generation with latent diffusion,” arXiv preprint arXiv:2404.10301, 2024.

[27] C. Yang, S. Wang, H. Chen, W. Tan, J. Yu, and H. Li, “Songbloom: Coherent song generation via interleaved autoregressive sketching and diffusion refinement,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id= Fa0kehLK6s

[28] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[29] K. Park, Y. J. Choe, and V. Veitch, “The linear representation hypothesis and the geometry of large language models,” in Proceedings of the International Conference on Machine Learning (ICML), 2024.

[30] J. Vig, S. Gehrmann, Y. Belinkov, S. Qian, D. Nevo, Y. Singer, and S. Shieber, “Investigating Gender Bias in Language Models Using Causal Mediation Analysis,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 12 388–12 401. [Online]. Available: https://proceedings.neurips.cc/paper/2020/hash/ 92650b2e92217715fe312e6fa7b90d82-Abstract.html

[31] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive languageaudio pretraining with feature fusion and keywordto-caption augmentation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[32] H. Zhu, Y. Zhou, H. Chen, J. Yu, Z. Ma, R. Gu, Y. Luo, W. Tan, and X. Chen, “MuQ: Self-supervised music representation learning with mel residual vector quantization,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 3653–3664, 2025.

[33] J. Serrà, R. O. Araz, D. Bogdanov, and Y. Mitsufuji, “Supervised contrastive learning from weaklylabeled audio segments for musical version matching,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 267. PMLR, 2025, pp. 53 923–53 939. [Online]. Available: https://proceedings.mlr.press/v267/serra25a.html

[34] A. Görgün, F. Sammani, N. Deligiannis, B. Schiele, and J. Fischer, “Temporal concept dynamics in diffusion models via prompt-conditioned interventions,” in Proceedings of the International Conference on Learning Representations (ICLR), 2026. [Online]. Available: https://openreview.net/forum?id=ABjaSsrYPD