# The 2026 PNPL Competition: Word Classification and Efficient Cross-Subject Generalisation in LibriBrain100

Francesco Mantegna<sup>1,∗</sup> Gereon Elvers<sup>1,∗</sup> Dulhan Jayalath<sup>1,∗</sup> Gilad Landau<sup>1</sup>

Tasha Kim<sup>1</sup> Miran Özdogan<sup>1</sup> Luisa Kurth<sup>1</sup> Teyun Kwon<sup>1</sup> SungJun Cho<sup>1,2</sup>

Benjamin Ballyk<sup>1</sup> Alex Fung<sup>1,3</sup> Anna Greer<sup>1</sup> Pratik Somaiya<sup>1</sup> Christian Herff<sup>4</sup>

Yorguin Mantilla Ramos<sup>5</sup> Hamza Abdelhedi<sup>5</sup> Karim Jerbi<sup>5,6</sup> Greg Farquhar<sup>7</sup>

Brendan Shillingford<sup>7</sup> Mark Woolrich<sup>2</sup> Oiwi Parker Jones<sup>1</sup>

<sup>1</sup>PNPL , Department of Engineering Science, University of Oxford, UK <sup>2</sup>OHBA, Oxford Centre for Integrative Neuroimaging, University of Oxford, UK <sup>3</sup>FMRIB, Oxford Centre for Integrative Neuroimaging, University of Oxford, UK <sup>4</sup>Maastricht University, The Netherlands <sup>5</sup>UNIQUE, Université de Montréal, Canada <sup>6</sup>Mila–Quebec AI Institute, Canada <sup>7</sup>Google DeepMind, UK

<sup>∗</sup>Joint first authors

{francesco, gereon, dulhan, oiwi}@robots.ox.ac.uk

## Abstract

The ambition of the 2025 PNPL competition (Landau et al., 2025) was to launch a multi-year curriculum for non-invasive speech decoding. Designed to progress from foundational tasks toward the linguistic complexity required for a practical brain-computer interface (BCI), it set the stage with speech detection and phoneme classification tasks. Winning submissions reached F1-macro scores of 95.6% and 73.6% on the respective tasks (Elvers et al., 2026), highly significant advances. This success was built on the LibriBrain dataset (Özdogan et al., 2025), the largest within-subject MEG dataset recorded at the time with ∼50 hours of data for one subject. However, while within-subject scale drives strong decoding performance, a practical BCI must generalise to new users from minutes of data, not hours.

The 2026 PNPL competition responds to this challenge with LibriBrain100 (Mantegna et al., 2026), an extended LibriBrain dataset with 32 additional subjects (∼40 minutes each) plus even more within-subject data (∼80 hours). Advancing the curriculum of tasks to focus on word classification, two complementary tracks are presented in this competition: the Deep track targets within-subject word classification at scale, aiming at the best possible performance; the Broad track targets cross-subject generalisation, progressively reducing the amount of subject-specific fine-tuning data from ∼40 to ∼20 to ∼10 minutes, a duration that falls within a clinically feasible range and brings us a step closer to a non-invasive BCI capable of restoring communication to people living with profound paralysis.

## 1 Competition description

## 1.1 Background and impact

Invasive brain-computer interfaces (BCIs) for speech have advanced at a remarkable pace. Since the first demonstration of connected-speech decoding from surgically implanted electrodes in a paralysed individual (Moses et al., 2021), vocabulary sizes have rapidly grown from 50 words to over 125,000 words (Willett et al., 2023) while word-error rates (WERs) have shrunk to 2.5% (Card et al., 2024)—a superhuman score well below the ∼5–10% reported for human transcription on standard automatic speech recognition (ASR) benchmarks (Xiong et al., 2016). Yet invasive BCIs face fundamental barriers to widespread deployment: brain surgery carries inherent risks, per-patient data collection does not easily scale, and the populations who might benefit most are often those least able to undergo elective neurosurgery; non-invasive alternatives are therefore the real prize.

Among non-invasive neuroimaging modalities, magnetoencephalography (MEG) stands out for its combination of millisecond temporal resolution and typical spatial precision around 5– 10 mm (Hämäläinen et al., 1993)—with the capacity to go even lower to 2–4 mm (Barratt et al., 2018)—putting it in the range of invasive modalities like ECoG but without the surgical risks. Recent years have seen genuine advances in MEG-based speech decoding (Défossez et al., 2023; d’Ascoli et al., 2025; Jayalath et al., 2025b), but progress has been harder to interpret and harder to compare across studies than in the invasive setting. A key reason is the lack of shared infrastructure: common datasets, fixed evaluation splits, standard metrics, and baseline implementations that allow the community to build cumulatively on each other’s work.

The PNPL competition series is motivated by the view that closing this gap will require not only better models, but better community infrastructure. The 2025 PNPL competition (Landau et al., 2025) was designed with this goal in mind. Building on the LibriBrain dataset (Özdogan et al., 2025)—the largest within-subject MEG dataset recorded at the time, with ∼50 hours of data from a single participant—it introduced common benchmarks for speech detection and phoneme classification, together with a Python library for automatic data download and loading, standardised train/validation/test splits for replicability, reference models, a public leaderboard, tutorial code, and a dedicated competition website. These tasks were deliberately chosen to be relatively simple: speech detection reduces to binary classification over time samples, while phoneme classification is a well-defined supervised learning problem with manageable output structure. This was part of a broader curricular design— by starting with accessible tasks, we aimed to lower the barrier to entry for researchers from machine learning who might not yet have the domain knowledge to confidently navigate human brain data, while laying the groundwork for more ambitious forms of neural speech decoding. Winning submissions reached F1-macro scores of 95.6% and 73.6% on the respective tasks, driving significant advances in the field (Elvers et al., 2026). Over the summer of 2025, the 2025 PNPL competition attracted 155 registered teams from 15 countries and generated 6,041 submissions. By April 2026, the LibriBrain dataset had reached over 16,000 downloads — indicating a clear demand for the materials we provided, even after the 2025 competition ended.

The 2026 PNPL competition represents a logical next step, with significantly more data and larger ambitions. A central reason this next step is now feasible is the scale of the expanded LibriBrain100 dataset (Mantegna et al., 2026). Compared with existing public MEG speech datasets, LibriBrain100 occupies a distinctive position: it combines substantially greater within-subject depth with a larger multi-subject cohort, supporting both high-performance within-subject modelling and systematic evaluation of cross-subject generalisation (visualised in Appendix A). This combination of depth and breadth is essential for the two-track design of the competition: the Deep track exploits the unusually large amount of data available for a single subject, while the Broad track tests whether models trained across subjects can adapt efficiently to new individuals. The task of word classification is a natural progression in the curriculum: it remains sufficiently structured to support clear benchmarking and broad participation, but it is substantially closer to the longer-term goal of full brain-to-text (B2T) (Herff et al., 2015), which is analogous to speech-to-text but with brain signals as inputs. Whereas speech detection and phoneme classification primarily target lower-level acoustic structure in the signal, word classification brings lexical and semantic information into scope and begins to probe whether models can recover meaningful units of speech from MEG data. As a stepping stone toward a working non-invasive B2T, it is well-motivated: word classification was a key component of the pipeline in the first invasive BCI for a paralysed individual (Moses et al., 2021), and remains a useful intermediate target given the apparent difficulty of non-invasive B2T (Jo et al., 2024) — though we have finally begun to see the first WER scores for non-invasive speech B2T that are significantly better than chance (Jayalath et al., 2025a). Although Brain2Qwerty models report interesting results for decoding sentences from MEG acquired while subjects overtly type on a keyboard (Lévy et al., 2026; Zhang et al., 2026), we view this as distinct from speech BCIs: there is currently no clear path for typing-based decoding to be used by patients unable to move their fingers.

![](images/20308bbb7ef2696fcc6b36da5d8a91e8bf192a6497891f130712f9009c3ac5ad.jpg)

![](images/28ae53d61495e42bab3542b4d2957c2077d6a5eacd6bf4881cff309cdc73005c.jpg)  
Figure 1: Subject-wise data splits. (a) Schematic illustration of the dataset. A single extensively recorded subject is shown alongside other subjects with progressively reduced amounts of available training data. (b) Quantitative representation of the data splits. A broken y-axis is used to accommodate the large difference between sub-0 and the other subjects, while preserving the visibility of differences in training data proportions. Vertical dashed lines separate the subject groups, and labels indicate the proportion of training data available within each group. A decreasing curve illustrates the overall reduction in available training data across subjects.

Word classification has recently attracted growing attention in the non-invasive decoding literature (d’Ascoli et al., 2025; Jayalath et al., 2025a; Jayalath and Parker Jones, 2026). Yet without standard evaluation protocols, results remain difficult to compare across studies. A key source of incomparability is the choice of target vocabulary. Two studies that report the same evaluation metric on the most frequent few hundred words in their respective datasets may appear comparable (e.g., d’Ascoli et al., 2025; Jayalath et al., 2025a), but this is complicated if the target words differ or if their frequencies vary between datasets (Jayalath et al., 2026). The choice of words also affects what can be communicated. For all of these reasons, it is important to report the target vocabulary explicitly. Official competition rankings are evaluated on a 50-word competition vocabulary designed to leverage high-frequency words in the training corpora and to support expressive communication (Appendix B).

To make it easier to compare competition results with prior results in the literature, we further track performance on a second 50-word vocabulary, the Moses 50 vocabulary, which has been used many times in invasive studies (e.g., Moses et al., 2021; Willett et al., 2023). Finally, we report an information-theoretic measure designed to compare results with different target vocabularies: openvocabulary mutual information (OVMI) (Jayalath et al., 2026). OVMI accounts for both decoding performance and vocabulary coverage (Section 1.5).

## 1.2 Novelty

The 2025 PNPL competition was the first dedicated to language decoding from non-invasive brain data (Landau et al., 2025). Building on its success, the 2026 competition introduces several elements that are new both to the PNPL series and to the broader field.

Word classification as a standardised competition task. Although word classification has recently attracted attention in the non-invasive decoding literature (d’Ascoli et al., 2025; Jayalath et al., 2025a; Jayalath and Parker Jones, 2026), standardised evaluation is harder here than for tasks with fixed symbol sets such as phoneme classification. Unlike phonemes, words form an effectively open set, and the choice of vocabulary—not merely its size—affects measured performance (Jayalath et al., 2026). Custom vocabularies, such as the most-frequent K words in a corpus (d’Ascoli et al., 2025), maximise training examples and reported accuracy scores. On the other hand, results for custom vocabularies are not comparable across studies and high-frequency words are often dominated by function words of limited communicative utility. Standardised cross-study vocabularies such as the Moses 50 (Moses et al., 2021) enable comparisons (including with invasive BCIs), but may be poorly attested in a given corpus. OVMI (Jayalath et al., 2026) addresses both concerns, jointly accounting for decoding accuracy and corpus coverage. Because they achieve different goals, we use all three methods of evaluation (Section 1.5) — a practice we recommend to the field. The primary metric for both the leaderboard and final ranking is the custom, dataset-tailored top-10 balanced accuracy.

Cross-subject generalisation as a competition target. The 2025 competition and the LibriBrain dataset focused exclusively on a single participant. Notably, all recent state-of-the-art invasive systems have likewise been trained and evaluated on a single individual (Moses et al., 2021; Metzger et al., 2023; Willett et al., 2023; Card et al., 2024), meaning cross-subject generalisation has received little attention in the field as a whole. The 2026 competition is the first to explicitly target it, introducing a dedicated Broad track that evaluates how well models can adapt to new individuals with progressively limited data. This mirrors the constraints of clinical deployment and provides a novel evaluation paradigm for data-efficient generalisation. Uniquely, the Broad track also includes holdout subjects for whom no subject-specific training data is released at all, enabling evaluation of zero-shot cross-subject generalisation — the most clinically relevant regime.

Contrast with invasive brain-to-text competitions. Building on high-performance but invasive speech neuroprosthesis systems (Willett et al., 2023; Card et al., 2024), the Brain-to-Text Benchmark ’24 and its successor competition target the decoding of full transcripts from intracortical recordings (Willett et al., 2024). Our goal is to build the corresponding non-invasive benchmark for MEG. Since open-vocabulary B2T from speech-related neural activity remains out of reach for current non-invasive systems (though see Jayalath et al., 2025a), we instead take a curricular approach and focus on word classification. This gives the community a concrete intermediate task on which to measure progress in representation learning, cross-subject generalisation, and decoding under the constraints of non-invasive neural data.

LibriBrain100 as competition resource. The competition is the first to be built around a large-scale, multi-subject MEG dataset. LibriBrain100 (Mantegna et al., 2026) combines the deepest withinsubject MEG dataset recorded to date (∼80 hours) with data from 32 additional subjects, enabling both within-subject and cross-subject benchmarking within a single competition.

## 1.3 Data

For this year’s competition, we primarily use the LibriBrain100 dataset (Mantegna et al., 2026), which comprises non-invasive MEG recordings from 33 subjects listening to naturalistic speech (stimuli details in Appendix C). Subject 0—the single participant from the original LibriBrain dataset (Özdogan et al., 2025)—is extended to ∼80 hours of recordings, the deepest within-subject MEG dataset recorded to date. For an additional 32 subjects, ∼40 minutes of MEG data were collected, but for the competition we have initially released tiered amounts: the full ∼40 minutes for 12 subjects, ∼20 minutes for 10, and ∼10 minutes for a further 10—a regime within the range of what will often be clinically feasible (Figure 1). The motivation for this is to challenge competition participants to predict with decreasing amounts of subject-specific data. For a final group of 8 subjects not included in LibriBrain100 (subjects 33–40), there are no subject-specific training data, requiring zero-shot cross-subject generalisation: the most demanding but clinically useful regime. We will release the full data for subjects 1–32 after the competition.

The LibriBrain100 data come with standard train, validation, and test splits for reproducibility. Evaluation in the competition is conducted exclusively on an additional competition holdout set, which is drawn from sources not revealed to participants in advance. For the competition, we release the MEG recordings for this holdout set but withhold the labels. Submissions consist of predictions on these recordings, uploaded to our evaluation platform. The holdout is divided into two disjoint partitions: one used to update the public leaderboard throughout the competition, and one reserved for final ranking of submissions. This design mirrors the 2025 competition and is intended to prevent overfitting to the holdout distribution while still giving participants timely feedback on their progress.

MEG data and labels are saved in HDF5 and TSV formats, respectively. The data are also available on Hugging Face in both serialised (HDF5) and raw (FIF) formats, across two repositories: libribrain contains the original LibriBrain recordings, while libribrain2 contains the additional data comprising LibriBrain100. Note that raw FIF files have not been preprocessed and include head movement artefacts and other noise; they are also substantially larger than the serialised versions.

To make interacting with the dataset easy, we provide an updated Python library that automatically downloads and loads data for PyTorch. It can be installed with a single command: pip install pnpl. We recommend using the library to load serialised data, as it handles downloads automatically and retrieves only the partitions requested (Appendix D); no knowledge of underlying repo structures is required.

## 1.4 Tasks and applications

The task in this competition is word classification from non-invasive brain recordings. Given a window of MEG data $\boldsymbol { x } \in \mathbb { R } ^ { \hat { 3 } 0 6 \times T }$ (sensors × time samples), the goal is to predict the word $y \in \mathcal V$ being heard by the participant, where V is a fixed retrieval vocabulary (see Mantegna et al., 2026). We use a custom 50-word vocabulary designed for reliable evaluation on LibriBrain100 (see Appendix B). Relevant neural information for this task may include both phonetic representations (reflecting acoustic and articulatory processing in auditory and motor cortices) and lexical semantic representations which cover most of the cortex (Huth et al., 2016). Given the distributed nature of semantic representations in particular, MEG’s whole-brain coverage is a distinct advantage over surgically implanted arrays, which sample only a limited (surgically accessed) cortical region. This makes word classification a richer decoding target than either speech detection or phoneme classification alone.

Word classification occupies an important position in the curriculum of non-invasive speech decoding. It is more structured than open-vocabulary brain-to-text, which has only recently yielded word error rates better than chance from non-invasive recordings (Jayalath et al., 2025a), making it a reliable and reproducible benchmark. At the same time, it goes substantially beyond the tasks of the 2025 competition (Landau et al., 2025): speech detection and phoneme classification target sub-lexical structure, but word classification directly probes the recovery of meaningful linguistic units. It is also practically motivated. Word classification, combined with speech detection and a language model, formed the core pipeline of the first invasive speech BCI for a paralysed individual (Moses et al., 2021). Solving word classification from non-invasive data, together with the speech detection foundation laid in 2025, would therefore constitute a meaningful step toward a non-invasive analogue.

The competition hosts two complementary word classification tasks that run concurrently:

(a) Deep track: This track evaluates word classification in a single, densely-sampled subject. Participants may train on any data they wish. The goal is the best possible decoding performance on the densely-sampled individual (Subject 0), to push the limits of non-invasive word classification.

(b) Broad track: This track evaluates cross-subject generalisation. The challenge is to produce accurate word classification predictions for individuals with limited subject-specific data: for 12 subjects, ∼40 minutes of subject-specific data is available; for 10 subjects, ∼20 minutes; and for a further 10 subjects, ∼10 minutes. The ∼10 minute regime is of particular interest for clinical applications, where extended patient data collection is often infeasible. Holdout data is provided for a final set of 8 subjects with no released subject-specific training data, enabling evaluation of zero-shot generalisation. To perform well on this track, submissions must generalise across all data regimes—motivating the emphasis on data-efficiency in the competition title.

## 1.5 Metrics

To measure success in the word classification tasks we use the Top-10 Balanced Accuracy with a fixed retrieval vocabulary of $K = 5 0$

$$
\mathrm { B A c c @ 1 0 } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { R e c a l l @ 1 0 } _ { k } .
$$

This metric has been used many times for decoding words in the recent decoding literature, with retrieval vocabularies ranging from $K = 5 0$ to $\bar { K } = 2 5 0$ (d’Ascoli et al., 2025; Jayalath and Parker Jones, 2026). For the competition, we track both a custom 50-word vocabulary and the Moses 50 vocabulary (Appendix B). Recall@ $1 0 _ { k }$ denotes the Top-10 recall for class k:

$$
\mathrm { R e c a l l @ 1 0 } _ { k } = \frac { 1 } { N _ { k } } \sum _ { i : y _ { i } = k } \mathbb { I } [ y _ { i } \in \left\{ \hat { y } _ { i , 1 } , \dots , \hat { y } _ { i , 1 0 } \right\} ] ,
$$

where $y _ { i }$ is the true label, $\hat { y } _ { i , r }$ the class with the r-th highest predicted score, and $N _ { k }$ the number of evaluation examples with true label k. We use a balanced metric so that each word contributes equally, regardless of how frequently it appears in the evaluation set (see, e.g., Thölke et al., 2023). In practice, BAcc@10 scores range from 0 to 1 and can be reported as percentages. For a vocabulary of $\bar { K } = 5 0$ words, uniform random guessing without replacement yields a BAcc@10 of 20% in expectation, since the correct class has probability $1 0 / 5 \bar { 0 }$ of appearing in a random top-10 set. A score of 100% means that the correct word is always included in the model’s top-10 predictions. A limitation of this metric is that it does not distinguish between cases where the correct word is ranked first or tenth; therefore, we also compute the Top-1 Balanced Accuracy (BAcc@1) to help break ties.

Finally, as an auxiliary metric, we report Open-Vocabulary Mutual Information (OVMI) (Jayalath et al., 2026). OVMI is designed to support comparison across decoders with different target vocabularies. It is an information-theoretic quantity that measures the mutual information between a user’s intent and the output of a decoding model relative to a reference communication distribution. Concretely, $\operatorname { O V M I } ( S ) { \overset { \cdot } { = } } C ( S ) I ( X ; Y \mid ^ { \cdot } X \in S )$ , where C(S) is the lexical coverage of the retrieval vocabulary S under a reference distribution p and $I ( X ; Y \mid { \dot { X } } { \dot { \in { \cal S } } } )$ is the in-vocabulary MI between the intended word X and the decoder output $Y ,$ . Conceptually, p acts like a prior over the words a user is likely to intend. For the competition, we use SUBTLEX-UK as $p$ (van Heuven et al., 2014) and report OVMI in bits per word perceived under this distribution. OVMI is 0 under uniform random guessing; higher OVMI scores are better.

## 1.6 Baselines, code, and material provided

To provide participants with baseline scores that are representative of the state-of-the-art and straight forward to implement and improve upon, we provide two reference models—one for each track.

The Deep track uses an implementation of the supervised word decoding model of d’Ascoli et al. (2025) as reference. This model achieves strong within-subject word classification performance and provides a competitive baseline for participants targeting the best possible performance on Subject 0.

The Broad track uses MEG-XL (Jayalath and Parker Jones, 2026) as a reference model. MEG-XL is a self-supervised foundation model pre-trained on ∼300 hours of MEG from 800 subjects, intended to be fine-tuned for word classification. Crucially, it is designed to generalise well with small amounts of subject-specific fine-tuning data, where it outperforms the d’Ascoli et al. (2025) model, making it particularly well-suited to the data-efficient cross-subject generalisation setting of the Broad track. Code and checkpoints are available through GitHub<sup>1</sup>, providing participants with a competitive starting point that they can directly improve upon. The choice of reference models reflects recent advances in the field: with large amounts of within-subject data, supervised models outperform selfsupervised ones (Jayalath and Parker Jones, 2026), while with limited per-subject data, pre-trained priors of self-supervised models like MEG-XL confer a substantial advantage (Mantegna et al., 2026).

For our baseline results, we train or fine-tune each reference model following the procedure described in their respective papers and report results on the competition test splits (Table 1). For the Deep track, we use all the available training data for Subject 0. For the Broad track, across Subjects 1–32, we use half of the data from the Sherlock 1 Session 11 recording for fine-tuning and the other half for validation. We suspect joint training on Subject 0 and Subjects 1–32 will improve generalisation on the latter subjects further, but leave this and other avenues of exploration to competition participants.

Table 1: Scores for reference models and random baselines. The column in (grey) highlights the competition evaluation metric (BAcc@10 on the competition vocabulary). All scores are on each track’s test data from the checkpoint with the best validation performance on the competition metric.
<table><tr><td rowspan="2">Method</td><td colspan="3">Competition Vocabulary</td><td colspan="3">Moses Vocabulary</td></tr><tr><td>BAcc@1</td><td>BAcc@10</td><td>OVMI</td><td>BAcc@1</td><td>BAcc@10</td><td>OVMI</td></tr><tr><td>Deep track</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>d’Ascoli (reference)</td><td>25.60%</td><td>73.23%</td><td>.220</td><td>38.84%</td><td>83.60%</td><td>.157</td></tr><tr><td>Random chance</td><td>2.00%</td><td>20.00%</td><td>.000</td><td>2.00%</td><td>20.00%</td><td>.000</td></tr><tr><td>Broad track</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MEG-XL (reference)</td><td>6.20%</td><td>42.96%</td><td>.014</td><td>10.21%</td><td>52.23%</td><td>.011</td></tr><tr><td>d’Ascoli</td><td>5.14%</td><td>33.13%</td><td>.009</td><td>6.10%</td><td>49.56%</td><td>.006</td></tr><tr><td>Random chance</td><td>2.00%</td><td>20.00%</td><td>.000</td><td>2.00%</td><td>20.00%</td><td>.000</td></tr></table>

As an entry point to the library and competition, we provide three (3) tutorial notebooks in the form of interactive Google Colab notebooks, optimised for usage with a free GPU:

1. First tutorial: Data loading, introduction to LibriBrain100 and word classification task. Contains code for fine-tuning the reference model on Subject 0.

2. Second tutorial: Loading subject-specific data, fine-tuning the reference model for crosssubject generalisation, more details on the Broad track.

3. Third tutorial: Prediction generation on the holdout data, submission to the leaderboard.

## 1.7 Competition website

The competition website<sup>2</sup> serves as a central hub, aggregating the starter kit, documentation, tutorial notebooks, leaderboards, and links to the Kaggle submission platform that accepts prediction files in CSV format. To maximise ease of onboarding, all tutorials are provided as interactive Colab notebooks that run in the browser, requiring no local installation beyond: pip install pnpl.

## 2 Organisation

## 2.1 Protocol

Participating is designed to be simple and accessible. To participate, one can install the pnpl Python library (pip install pnpl), which automatically downloads the dataset and provides a standard PyTorch DataLoader. Once data is loaded, participants can develop solutions and generate predictions on the holdout data. The pnpl library can be used to write these predictions to a CSV file, which can then be uploaded to the submission platform (Appendix D). Solutions are evaluated automatically according to the metrics in Section 1.5, and rankings updated continuously on the public leaderboard.

Both the Deep and Broad tracks run concurrently for three months (15 July – 15 October). Participants can submit to either or both tracks with progress reflected on the appropriate leaderboard.

## 2.2 Rules and engagement

Goal. The competition aims to be open and accessible to a broad community of researchers. This includes participants that may have no prior experience in analysing neural data. Our rules are intended to encourage open participation, while protecting integrity of the evaluation process.

Competition rules.

1. Participation is open to everyone. Domain-specific knowledge or specialised hardware is not required for fully participating in the competition.

2. Competition tracks. The competition is organised around the LibriBrain100 dataset and consists of two distinct tracks, both targeting word classification using balanced accuracy over a fixed vocabulary: (a) Deep track: word classification within a single, deeply-sampled subject, and (b) Broad track: word classification across many held-out subjects.

Note, the two tracks are ranked separately, with (i) a leaderboard and (ii) prizes separately tracked for each. Participants may submit to either or both tracks.

3. Permitted materials. Participants may use any training data for either track. This includes (i) external, publicly available datasets, (ii) self-collected or synthetically-generated data, or (iii) pre-trained models available from public repositories on Hugging Face (https:// huggingface.co) or GitHub (https://github.com/). Participants must independently ensure their data use complies with the appropriate licensing and ethical requirements.

4. Winners. Final rankings are determined against an independent holdout dataset to prevent data leakage. We discourage any attempt to infer held-out labels outside of the official evaluation procedure. The organisers reserve the right to make final decisions on edge cases.

5. Prize distribution. The highest-ranked three (3) submissions in each track are to be considered for prizes, provided they beat the performance of reference model baselines described in Section 1.6. Each team is eligible to win at most one prize across the competition. If the same team places in the prize-winning positions on both tracks, they will be listed on all relevant leaderboards, but prizes for the additional track pass to the next eligible team. In the unlikely event of a tie, the prize will be split between the tied teams.

6. The organisers will contact all prize contenders for additional verification of their submissions. Participants may be asked to share the training code, model checkpoints, or descriptions of their approach. If neither correctness of the approach nor compliance with the competition rules can be verified, the submission may be excluded from prize eligibility.

7. The organising team encourages participants to share their code in the spirit of open science, including direct contributions (e.g., data loaders, reusable baselines, documentation) to the pnpl GitHub repository (https://github.com/neural-processing-lab/pnpl).

Communication and community. To facilitate open and real-time communication, the competition uses the same Discord server established during last year’s competition<sup>3</sup>.

## 2.3 Schedule and readiness

All essential competition materials are online. For prizes, the equivalent of \$5,000 USD has been raised.

• 15 July, 2026: Submissions to both tracks officially open. All necessary materials are released, including documentation and unlabelled evaluation data for each track.

• 15 July – 15 October, 2026: Competition progress is tracked on the public leaderboards. The organising team provides support through the competition website, GitHub, and public Discord server. Participants may submit to either track during the competition period.

• 15 October, 2026: Submissions to both tracks officially close.

• 15 October – 30 November, 2026: The organising team reviews and verifies the highestranking submissions. Contenders for all prizes will be contacted to further confirm submission details.

• 1 December, 2026: The confirmed winners are announced through the competition website. The organising team independently conducts post-competition analysis of the results. Winners will be connected with sponsors to coordinate prizes.

## 2.4 Competition promotion and incentives

We are promoting the competition through social networks and academic mailing lists (e.g., NeurIPS affinity groups). We launched a dedicated website to share information, resources, and updates regarding the competition. This includes live leaderboards. During the competition, the organising team plans to release blog posts about the tasks and data to foster discussion.

## 3 Resources

The pnpl Python library is available to download the data and integrate it seamlessly into standard deep learning frameworks. Reference model code and checkpoints are publicly available on GitHub. The tutorial code runs on Google Colab in the browser, providing some GPU access to get started.

## Organising team

Francesco Mantegna is a Postdoctoral Researcher in PNPL , Department of Engineering Science, University of Oxford. He received his PhD from NYU under the supervision of David Poeppel.

Gereon Elvers is an Encode: AI for Science Fellow in PNPL , University of Oxford. He was previously a master’s student, visiting PNPL from the Technical University of Munich (TUM).

Dulhan Jayalath is a DPhil student in Machine Learning at the University of Oxford, supervised by Oiwi Parker Jones as part of PNPL and funded by an Amazon Web Services (AWS) studentship as part of the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (AIMS).

Gilad Landau is a DPhil student in Engineering and member of PNPL , supervised by Oiwi Parker Jones at the University of Oxford.

Tasha Kim is a DPhil student in Engineering and member of PNPL , supervised by Oiwi Parker Jones at the University of Oxford.

Miran Özdogan is a DPhil student in Computer Science and member of PNPL , supervised by Oiwi Parker Jones and Michael Bronstein, at the University of Oxford.

Luisa Kurth is a DPhil student in PNPL and the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (AIMS), University of Oxford.

Teyun Kwon is a DPhil student in PNPL and the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (AIMS), University of Oxford.

SungJun Cho is a DPhil student in Neuroscience supervised by Mark Woolrich and Oiwi Parker Jones at the University of Oxford.

Benjamin Ballyk is a DPhil student in Engineering and member of PNPL , supervised by Oiwi Parker Jones at the University of Oxford.

Alex Fung is a DPhil student in Neuroscience and member of PNPL , supervised by Alex Green, Saad Jbabdi, and Oiwi Parker Jones at the University of Oxford.

Anna Greer is a DPhil student in the EPSRC Centre for Doctoral Training in Autonomous Intelligent Machines and Systems (AIMS), University of Oxford. She contributed during a rotation mini-project in PNPL .

Pratik Somaiya is a Software Engineer at the Oxford Robotics Institute.

Christian Herff is an Associate Professor and head of the Neural Interfacing Lab in the Department of Neurosurgery at Maastricht University.

Yorguin Mantilla Ramos is a master’s student at the Université de Montréal and Graduate Research Assistant at Mila.

Hamza Abdelhedi is a Biomedical Engineering PhD student at the Université de Montréal, supervised by Karim Jerbi and specialising in Neuro-AI.

Karim Jerbi is Professor in the Psychology Department of the Université de Montréal and Associate Professor at Mila. He heads UNIQUE, the Quebec-wide Neuro-AI research center, and is also Canada Research Chair in Computational Neuroscience and Cognitive Neuroimaging.

Greg Farquhar is a Staff Research Scientist at Google DeepMind.

Brendan Shillingford is a Staff Research Scientist at Google DeepMind.

Mark Woolrich is Professor of Computational Neuroscience at the University of Oxford. He is Head of Analysis and Associate Director of the Oxford Centre for Human Brain Activity (OHBA).

Oiwi Parker Jones heads the Parker Jones Neural Processing Lab (PNPL ) in the Department of Engineering Science, University of Oxford. He is also a Fellow in Computer Science at Jesus College Oxford, a Principal Investigator at the Oxford Robotics Institute, and an Honorary Fellow in the Nuffield Department of Clinical Neurosciences.

## Acknowledgments

Many thanks to Pillar VC for sponsoring this year’s competition. We gratefully acknowledge the use of the University of Oxford Advanced Research Computing (ARC) facility (http://dx.doi.org/10. 5281/zenodo.22558), Hartree Centre resources, and the NVIDIA Corporation for donating additional GPUs. PNPL is supported by the MRC (MR/X00757X/1), Royal Society (RG\R1\241267), NSF (2314493), NFRF (NFRFT-2022-00241), SSHRC (895-2023-1022), and ARIA (SCNI-SE01-P004).

## References

Armeni, K., Güçlü, U., van Gerven, M., and Schoffelen, J.-M. (2022). A 10-hour within-participant magnetoencephalography narrative dataset to test models of language comprehension. Scientific Data, 9:278.

Barratt, E. L., Francis, S. T., Morris, P. G., and Brookes, M. J. (2018). Mapping the topological organisation of beta oscillations in motor cortex using MEG. NeuroImage, 181:831–844.

Card, N. S., Wairagkar, M., Iacobacci, C., Hou, X., Singer-Clark, T., Willett, F. R., Kunz, E. M., Fan, C., Vahdati Nia, M., Deo, D. R., Srinivasan, A., Choi, E. Y., Glasser, M. F., Hochberg, L. R., Henderson, J. M., Shahlaie, K., Stavisky, S. D., and Brandman, D. M. (2024). An accurate and rapidly calibrating speech neuroprosthesis. New England Journal of Medicine, 391(7):609–618.

d’Ascoli, S., Bel, C., Rapin, J., Banville, H., Benchetrit, Y., Pallier, C., and King, J.-R. (2025). Towards decoding individual words from non-invasive brain recordings. Nature Communications, 16:10521.

Doyle, A. C. (1888). A Study in Scarlet. Ward, Lock & Co.

Doyle, A. C. (1890). The Sign of the Four. Spencer Blackett.

Doyle, A. C. (1892). The Adventures of Sherlock Holmes. George Newnes.

Doyle, A. C. (1893). The Memoirs of Sherlock Holmes. George Newnes.

Doyle, A. C. (1902). The Hound of the Baskervilles. George Newnes.

Doyle, A. C. (1905). The Return of Sherlock Holmes. George Newnes.

Doyle, A. C. (1915). The Valley of Fear. George H. Doran Company.

Doyle, A. C. (1917). His Last Bow. John Murray.

Doyle, A. C. (1927). The Case-Book of Sherlock Holmes. John Murray.

Défossez, A., Caucheteux, C., Rapin, J., Kabeli, O., and King, J.-R. (2023). Decoding speech perception from non-invasive brain recordings. Nature Machine Intelligence, 5(10):1097–1107.

Elvers, G., Landau, G., Mantegna, F., Özdogan, M., Kim, T., Kwon, T., Cho, S., Ballyk, B., Kurth, L., Jayalath, D., Somaiya, P., de Zuazo, X., Shillingford, B., Farquhar, G., Jiang, M., Jerbi, K., Abdelhedi, H., Mantilla Ramos, Y., Gulcehre, C., Woolrich, M., Voets, N., and Parker Jones, O. (2026). Benchmarking non-invasive speech BCIs: Lessons learned from the 2025 PNPL competition. Journal of Machine Learning Research. In press.

Garofolo, J. S., Lamel, L. F., Fisher, W. M., Fiscus, J. G., Pallett, D. S., Dahlgren, N. L., and Zue, V. (1993). TIMIT acoustic-phonetic continuous speech corpus. Linguistic Data Consortium. https://catalog.ldc.upenn.edu/LDC93S1.

Gwilliams, L., Flick, G., Marantz, A., Pylkkanen, L., Poeppel, D., and King, J.-R. (2023). Introducing MEG-MASC: A high-quality magneto-encephalography dataset for evaluating natural speech processing. Scientific Data, 10(1):862.

Hämäläinen, M., Hari, R., Ilmoniemi, R. J., Knuutila, J., and Lounasmaa, O. V. (1993). Magnetoencephalography—theory, instrumentation, and applications to noninvasive studies of the working human brain. Reviews of Modern Physics, 65(2):413–497.

Herff, C., Heger, D., de Pesters, A., Telaar, D., Brunner, P., Schalk, G., and Schultz, T. (2015). Brain-to-text: decoding spoken phrases from phone representations in the brain. Frontiers in Neuroscience, 9(217):1–11.

Huth, A. G., de Heer, W. A., Griffiths, T. L., Theunissen, F. E., and Gallant, J. L. (2016). Natural speech reveals the semantic maps that tile human cerebral cortex. Nature, 532:453–458.

Jayalath, D., Ballyk, B., and Parker Jones, O. (2026). A common measure of communication for speech brain–computer interfaces. Manuscript in preparation.

Jayalath, D., Landau, G., and Parker Jones, O. (2025a). Unlocking non-invasive brain-to-text. International Conference on Machine Learning (ICML), Workshop on Generative AI and Biology. arXiv preprint arXiv:2505.13446.

Jayalath, D., Landau, G., Shillingford, B., Woolrich, M. W., and Parker Jones, O. (2025b). The Brain’s Bitter Lesson: Scaling speech decoding with self-supervised learning. International Conference on Machine Learning (ICML). arXiv preprint arXiv:2406.04328.

Jayalath, D. and Parker Jones, O. (2026). MEG-XL: Data-efficient brain-to-text via longcontext pre-training. International Conference on Machine Learning (ICML). arXiv preprint arXiv:2602.02494.

Jo, H., Yang, Y., Han, J., Duan, Y., Xiong, H., and Lee, W. H. (2024). Are EEG-to-text models working? arXiv preprint. https://arxiv.org/abs/2405.06459.

Landau, G., Özdogan, M., Elvers, G., Mantegna, F., Somaiya, P., Jayalath, D., Kurth, L., Kwon, T., Shillingford, B., Farquhar, G., Jiang, M., Jerbi, K., Abdelhedi, H., Mantilla Ramos, Y., Gulcehre, C., Woolrich, M., Voets, N., and Parker Jones, O. (2025). The 2025 PNPL competition: Speech detection and phoneme classification in the LibriBrain dataset. NeurIPS, Competition Track. arXiv preprint arXiv:2506.10165.

Lévy, J., Zhang, M., Pinet, S., Rapin, J., Banville, H., d’Ascoli, S., and King, J.-R. (2026). Noninvasive decoding of typed sentences from human brain activity. Nature Neuroscience.

Mantegna, F., Jayalath, D., Elvers, G., Kim, T., Ballyk, B., Fung, A., Cho, S., Kwon, T., Kurth, L., Özdogan, M., Landau, G., Somaiya, P., Voets, N., Woolrich, M., and Parker Jones, O. (2026). LibriBrain100: One hundred hours of broad and deep MEG data for neural speech decoding at scale. arXiv preprint arXiv:2608.25204.

Metzger, S. L., Littlejohn, K. T., Silva, A. B., Moses, D. A., Seaton, M. P., Wang, R., Dougherty, M. E., Liu, J. R., Wu, P., Berger, M. A., Zhuravleva, I., Tu-Chan, A., Ganguly, K., Anumanchipalli, G. K., and Chang, E. F. (2023). A high-performance neuroprosthesis for speech decoding and avatar control. Nature, 620:1037–1046.

Moses, D. A., Metzger, S. L., Liu, J. R., Anumanchipalli, G. K., Makin, J. G., Sun, P. F., Chartier, J., Dougherty, M. E., Liu, P. M., Abrams, G. M., Tu-Chan, A., Ganguly, K., and Chang, E. F. (2021). Neuroprosthesis for decoding speech in a paralyzed person with anarthria. New England Journal of Medicine, 385(3):217–227.

Schoffelen, J.-M., Oostenveld, R., Lam, N. H. L., Uddén, J., Hultén, A., and Hagoort, P. (2019). A 204-subject multimodal neuroimaging dataset to study language processing. Scientific Data, 6(17):1–13.

Tang, J., LeBel, A., Jain, S., and Huth, A. G. (2023). Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 26(5):858–866.

Thölke, P., Mantilla-Ramos, Y.-J., Abdelhedi, H., Maschke, C., Dehgan, A., Harel, Y., Kemtur, A., Mekki Berrada, L., Sahraoui, M., Young, T., Bellemare Pépin, A., El Khantour, C., Landry, M., Pascarella, A., Hadid, V., Combrisson, E., O’Byrne, J., and Jerbi, K. (2023). Class imbalance should not throw you off balance: Choosing the right classifiers and performance metrics for brain decoding with imbalanced data. NeuroImage, 277:120253.

van Heuven, W. J. B., Mandera, P., Keuleers, E., and Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly Journal of Experimental Psychology, 67:1176 – 1190.

Willett, F. R., Kunz, E. M., Fan, C., Avansino, D. T., Wilson, G. H., Choi, E. Y., Kamdar, F., Glasser, M. F., Hochberg, L. R., Druckmann, S., Shenoy, K. V., and Henderson, J. M. (2023). A high-performance speech neuroprosthesis. Nature, 620:1031–1036.

Willett, F. R., Li, J., Le, T., Fan, C., Chen, M., Shlizerman, E., Chen, Y., Zheng, X., Okubo, T. S., Benster, T., Lee, H. D., Kounga, M., Buchanan, E. K., Zoltowski, D., Linderman, S. W., and Henderson, J. M. (2024). Brain-to-Text Benchmark ’24: Lessons learned. arXiv preprint arXiv:2412.17227.

Wrench, A. (1999). The MOCHA-TIMIT articulatory database. https://www.cstr.ed.ac.uk/research/ projects/artic/mocha.html. Created November 1999; accessed 2026-02-07.

Xiong, W., Droppo, J., Huang, X., Seide, F., Seltzer, M., Stolcke, A., Yu, D., and Zweig, G. (2016). Achieving human parity in conversational speech recognition. arXiv preprint arXiv:1610.05256.

Zhang, M., Lévy, J., Rommel, C., Rapin, J., Bel, C., Bonnaire, J., Nieto, D., Bourdillon, P., Pinet, S., d’Ascoli, S., Moreau, T., and King, J.-R. (2026). Accurate decoding of natural sentences from non-invasive brain recordings. arXiv preprint arXiv:2608.18114.

Özdogan, M., Landau, G., Elvers, G., Jayalath, D., Somaiya, P., Mantegna, F., Woolrich, M., and Parker Jones, O. (2025). LibriBrain: Over 50 hours of within-subject MEG to improve speech decoding methods at scale. NeurIPS, Datasets and Benchmarks Track.

## Appendices / Supplemental Materials

## A Dataset Comparison

Figure 2 compares LibriBrain100 against other public MEG speech datasets along three dimensions: cohort breadth, single-subject depth, and total recording time. Each point corresponds to one dataset, with the x-axis showing the number of subjects and the y-axis showing the maximum number of recording hours available for any single subject. Bubble area is proportional to the total number of recording hours in the dataset. The arrow highlights the step from LibriBrain to LibriBrain100: LibriBrain100 substantially increases the total recording time and number of subjects while preserving the unusually deep single-subject regime that made LibriBrain useful for within-subject speech decoding. The public MEG speech datasets compared to LibriBrain100 (Mantegna et al., 2026) are LibriBrain (Özdogan et al., 2025), Armeni (Armeni et al., 2022), Le Petit Prince (d’Ascoli et al., 2025), MEG-MASC (Gwilliams et al., 2023), and MOUS (Schoffelen et al., 2019).

![](images/7daaf86eda429d42804c9d14b311f7184862592b2015fb71168ee120a029e2ec.jpg)  
Figure 2: Dataset comparison. Comparison of public MEG speech datasets by number of subjects and maximum hours per subject. Both axes are logarithmically scaled. Each bubble represents one dataset, with bubble area proportional to total recording hours. LibriBrain100 occupies a distinctive position: it combines a competitive number of subjects with substantially more total recording time and substantially greater single-subject depth than most existing datasets. The arrow indicates the expansion from LibriBrain to LibriBrain100.

## B Retrieval Vocabulary

Table 2 lists the two 50-word retrieval vocabularies used for evaluating word classification in this competition. Both metrics defined in Section 1.5 (top-10 balanced accuracy and OVMI) are computed independently on each vocabulary and reported separately. Only BAcc@10 on the competition vocabulary is the primary metric used for competition leaderboards and prize decisions.

Competition vocabulary. The first vocabulary is a 50-word set selected to achieve low-variance performance estimation. Two criteria guided selection. First, words were required to occur frequently enough to provide sufficient examples for both training and held-out evaluation. Second, the set was constructed to span multiple grammatical categories—function words (determiners, pronouns, auxiliaries, prepositions, conjunctions) together with common content words (e.g. time, good, think, people, new)—so that the vocabulary supports the composition of practically meaningful utterances.

<table><tr><td>#</td><td>Word</td><td>#</td><td>Word</td><td>#</td><td>Word</td><td>#</td><td>Word</td><td>#</td><td></td><td>Word</td></tr><tr><td>1</td><td>a</td><td>11</td><td>be</td><td>21</td><td>him</td><td>31</td><td>on</td><td>41</td><td></td><td>they</td></tr><tr><td>2</td><td>all</td><td>12</td><td>but</td><td>22</td><td>i</td><td>32</td><td>our</td><td>42</td><td></td><td>think</td></tr><tr><td>3</td><td>always</td><td>13</td><td>can</td><td>23</td><td>in</td><td>33</td><td>out</td><td>43</td><td></td><td>this</td></tr><tr><td>4</td><td>am</td><td>14</td><td>do</td><td>24</td><td>is</td><td>34</td><td>people</td><td>44</td><td></td><td>time</td></tr><tr><td>5</td><td>an</td><td>15</td><td>for</td><td>25</td><td>it</td><td>35</td><td>really</td><td>45</td><td>to</td><td></td></tr><tr><td>6</td><td>and</td><td>16</td><td>good</td><td>26</td><td>it&#x27;s</td><td>36</td><td>she</td><td>46</td><td></td><td>very</td></tr><tr><td>7</td><td>any</td><td>17</td><td>had</td><td>27</td><td>my</td><td>37</td><td>SO</td><td>47</td><td></td><td>was</td></tr><tr><td>8</td><td>are</td><td>18</td><td>has</td><td>28</td><td>new</td><td>38</td><td>that</td><td>48</td><td>we</td><td></td></tr><tr><td>9</td><td>as</td><td>19</td><td>have</td><td>29</td><td>not</td><td>39</td><td>the</td><td>49</td><td></td><td>were</td></tr><tr><td>10</td><td>at</td><td>20</td><td>he</td><td>30</td><td>of</td><td>40</td><td>there</td><td>50</td><td>will</td><td></td></tr><tr><td colspan="9">(b) Moses 50 vocabulary (Moses et al., 2021)</td><td></td></tr><tr><td>#</td><td>Word</td><td>#</td><td>Word</td><td>#</td><td>Word</td><td>#</td><td></td><td>Word</td><td></td><td>Word</td></tr><tr><td>1</td><td>am</td><td>11</td><td>faith</td><td>21</td><td>here</td><td>31 32</td><td></td><td>need</td><td>41</td><td>that</td></tr><tr><td>2</td><td>are</td><td>12</td><td>family</td><td>22</td><td>hope</td><td></td><td></td><td>no</td><td>42</td><td>they</td></tr><tr><td>3</td><td>bad</td><td>13</td><td>feel</td><td>23</td><td>how</td><td></td><td></td><td>not</td><td>43</td><td>thirsty</td></tr><tr><td>4</td><td>bring</td><td>14</td><td>glasses</td><td>24</td><td></td><td>hungry 34 35</td><td></td><td>nurse</td><td>44</td><td>tired</td></tr><tr><td>5</td><td>clean</td><td>15</td><td>going</td><td>25</td><td>i</td><td></td><td></td><td>okay</td><td>45</td><td>up</td></tr><tr><td>6</td><td>closer</td><td>16</td><td>good</td><td>26</td><td>is</td><td></td><td></td><td>outside</td><td>46</td><td>very</td></tr><tr><td>7</td><td>comfortable</td><td>17</td><td>goodbye</td><td>27</td><td>it</td><td>37</td><td></td><td>please</td><td>47</td><td>what</td></tr><tr><td>8</td><td>coming</td><td>18</td><td>have</td><td>28</td><td>like</td><td>38</td><td></td><td>right</td><td>48</td><td>where</td></tr><tr><td>9</td><td>computer</td><td>19</td><td>hello</td><td>29</td><td>music</td><td>39 40</td><td></td><td>success</td><td>49</td><td>yes</td></tr><tr><td>10</td><td>do</td><td>20</td><td>help</td><td>30</td><td>my</td><td></td><td></td><td>tell</td><td>50</td><td>you</td></tr></table>

Table 2: The two 50-word retrieval vocabularies used in the word-classification task, listed alphabetically with index values for reference. (a) The competition vocabulary, selected from words that have sufficient occurrences for low-variance evaluation. (b) The Moses 50 vocabulary (Moses et al., 2021), developed for assistive communication and used in invasive decoding studies.  
(a) Competition vocabulary (50 words)

This design preserves continuity with recent non-invasive word-decoding work, where similar highfrequency vocabularies are standard (d’Ascoli et al., 2025; Jayalath and Parker Jones, 2026).

Moses 50. The second vocabulary is the 50-word list of Moses et al. (2021), developed with patient input for assistive communication and subsequently adopted in invasive decoding studies (Willett et al., 2023). In contrast to the competition vocabulary, Moses 50 emphasises content words tied to clinical and daily-living needs (hungry, thirsty, tired, glasses, nurse, music) and social or affective expressions (hello, goodbye, please, hope). Reporting results on Moses 50 enables direct comparison between non-invasive systems evaluated here and invasive systems evaluated against the same vocabulary for the first time. But even with the size of the LibriBrain100 corpus, not all of the Moses 50 words are adequately attested in it. Evaluating on the Moses 50 data alone would underrepresent what is possible and the words in it are not the only words we want to decode.

Despite their modest size, both vocabularies support a useful range of short utterances. From Moses 50: assistive expressions such as “i feel tired”, “please bring my glasses”, “i am not comfortable”, and “tell my family”. From the competition vocabulary: short conversational forms such as “i think so”, “we have time”, “i am very good”, “i am not good at all”, “can i have this”, and “we can do this”. The two vocabularies share 13 words, so their union covers 87 distinct word types.

## C Stimulus Materials

Subject 0 listened to the complete Sherlock Holmes canon (Doyle, 1888, 1890, 1892, 1893, 1902, 1905, 1915, 1917, 1927), the TIMIT acoustic-phonetic corpus (Garofolo et al., 1993), the MOCHA-TIMIT articulatory database (Wrench, 1999), and 30 podcast stories from The Moth, a subset of those used in Tang et al. (2023). Subjects 1–32 listened to the validation and test recordings from LibriBrain (Özdogan et al., 2025). Further details for stimuli, recording protocols, and preprocessing can be found in the LibriBrain100 dataset paper (Mantegna et al., 2026).

## D Code Samples

The recommended way to participate in the competition is through our custom Python library, pnpl (pip install pnpl), which integrates ready-to-use PyTorch dataloaders, including automatic dataset downloads from Hugging Face, and task configurations:

```python
1 from pnpl.datasets import LibriBrain100
2 from pnpl.tasks import WordClassification
3
4 ds = LibriBrain100(
5 data_path="./data/LibriBrain100", # data downloaded if missing from path
6 task=WordClassification(tmin=0.2, tmax=0.6),
7 partition="train",
8
9
10 x, y = ds[0] # x: (channels, time), y: integer word class id
```

In addition, it includes a small utility for submitting predictions to the Kaggle competition.

```python
1 from pnpl.competition import write_submission, submit_to_kaggle
2
3 write_submission(
4 "submission.csv",
5 indices=indices, # (N,)
6 primary_probs=primary, # (N, 50) — Competition Vocabulary
7 secondary_probs=secondary, # (N, 50) — Moses Vocabulary
8
9 submit_to_kaggle("submission.csv")
```