# Activation Probes Are Reliable With a Handful of Positive Examples

Riya Tyagi

Stefan Heimersheim

## Abstract

Efforts to monitor advanced AI for rare misalignments face a data challenge: abundant aligned examples but only a handful of misaligned ones. We test activation probes in this "few vs. thousands" regime on spam and honesty detection tasks. For our tasks, training with many negative examples is on average more positive-sampleefficient than balanced training for small numbers (1-10) of positive samples. We also find that LLM upsampling can provide a performance boost equivalent to roughly doubling the number of real positive samples, though excessive upsampling hurts performance. Finally, we show a positive scaling trend, where larger models are more positive-sample-efficient to probe. Our findings suggest we should leverage the large number of negative samples available to amplify the signal from rare but critical misalignment examples.

## 1 Introduction

As the autonomy of large language models (LLMs) increases, so does the risk of intentional misalignment, where a model might conceal objectives that diverge from user intent [1]. This possibility means monitoring solely model outputs could be insufficient, motivating the use of activation probes to interpret a model’s internal states directly [2, 3]. This approach assumes that an intentionally mis aligned agent forms coherent, detectable representations of its state and goals, potentially including deceptive intent. While the existence of such representations is debated [4], recent work training deception probes [5] suggests this remains a plausible hypothesis.

We work on a relevant question for high-stakes monitoring: can a reliable probe be trained from only a handful (e.g., 1-3) of verified misalignment instances? This investigation is motivated by a realistic future scenario where models produce abundant examples of aligned behavior but very few instances of misalignment. Some (not all) reasons such scarcity may arise are because (a) we successfully train models to avoid unsafe behaviors, or (b) models are deceptively aligned, exhibiting dangerous behavior only rarely, or when they believe they won’t be caught. We care about preparing for situation (b).

While previous studies investigating low-data settings performed activation probing to detect highstakes interactions [6], explored prompting probing [7], and examined the use of sparse autoencoder features [7] [8], they all simulate scarcity by reducing both classes equally or considering milder class imbalance (5% positive class ratio). Our work diverges by studying probing in the realistic setting of extreme positive class scarcity, where we can potentially leverage the abundance of negative samples.

Our contributions are:

• Extra negative samples help: We find that training activation probes with few positive and many negative examples generally performs better than using balanced numbers of samples for extremely low numbers of positive samples.

• LLM upsampling = 2x real data: We test LLM-based upsampling to augment scarce positive data, showing a large LLM upsampling budget can provide a performance boost equivalent to ≈ 2x the number of real samples.

## 2 Experimental Setup

## 2.1 Probing Methods

We evaluate several families of probes, summarized in Table 1. Let $A \in \mathbb { R } ^ { S \times D }$ be the matrix of activations from a specific layer, where $a _ { s } \in \mathbb { R } ^ { D }$ is the activation vector at token position s. For SAE-based methods, $\boldsymbol { Z } \in \mathbb { R } ^ { \boldsymbol { \check { S } \times } \boldsymbol { \check { W } } }$ is the matrix of sparse feature activations from the SAE encoder, where W is the SAE width. Linear probes learn a direction vector $\boldsymbol { \theta } \in \mathbb { R } ^ { D }$ . For SAE Probes, we do not train on the full set of latents; instead, we first select the top-k features $Z _ { k }$ with the highest average absolute difference between the positive and negative classes, following [8]. A linear probe then learns a direction vector $\boldsymbol \theta \in \mathbb { R } ^ { k }$ on only this sparse subset of features. The non-parametric cosine similarity probe uses pre-computed positive $\bar { ( a ^ { + } ) }$ and negative $( \bar { a } ^ { - } )$ class centroids. Attention probes learn separate query $( \bar { \theta _ { q } } )$ and value $( \theta _ { v } )$ vectors to weigh the importance of each token position. More probe architecture and probe training details are in Appendix A.2.

<table><tr><td>Probe Type</td><td>Score Function</td></tr><tr><td>Linear Probe</td><td> $\theta ^ { T } \mathrm { A g g } ( A )$ </td></tr><tr><td>SAE Probe</td><td> $\theta ^ { T } \mathrm { A g g } ( Z _ { k } )$ </td></tr><tr><td>Cosine Similarity Probe</td><td> $\cos ( \operatorname { A g g } ( A ) , \bar { a } ^ { + } ) - \cos ( \operatorname { A g g } ( A ) , \bar { a } ^ { - } )$ </td></tr><tr><td>Attention Probe</td><td> $\mathrm { s o f t m a x } ( A \theta _ { q } ) ^ { T } A \theta _ { v }$ </td></tr></table>

Table 1: Functional forms of probe types. All non-attention probes first apply an aggregation function $\mathrm { A g g ( \cdot ) }$ to map a sequence of vectors to a single vector. We test four aggregations: Mean $\textstyle { \bigl ( } { \frac { 1 } { S } } \sum a _ { s } { \bigr ) }$ Max $( \operatorname* { m a x } _ { s } A )$ , Last Token $( a _ { S } )$ , and Softmax $( \textstyle \sum _ { s } w _ { s } a _ { s } ,$ a weighted sum where weights are derived from a softmax over the sequence dimension of A).

## 2.2 Datasets and Protocols

Tasks We use one off-policy and one on-policy dataset. More dataset details are in Appendix A.3.

• Off-Policy (Spam Detection): We train probes on Gemma-2-9b on activations after layer 20 (the layer found to be best by [8] for their baseline probes) to classify emails as spam using the Enron-Spam dataset [9]. We evaluate performance on Enron-Spam and outof-distribution performance on an SMS spam dataset [10]. We repeat the process for Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, and Qwen3-32B, training probes on activations after layers 15, 15, 18, 18, 24, and 32 respectively.

• On-Policy (Honesty Detection): We train probes on Llama-3.3-70B-Instruct activations after layer 50 on the MASK benchmark [11].

Data Filtering Protocol A probe should detect a model property (e.g., dishonesty), not factual accuracy. We therefore filter our datasets to only include samples where the base model knows the correct answer. For the off-policy spam task, we use few-shot prompting to elicit the model’s belief and filter based on the logit difference between "Yes" and "No" answer tokens. For the on-policy MASK task, the benchmark’s use of belief prompts serves the same purpose. This aims to assess our probes on their ability to detect the target concept, conditioned on the model’s internal knowledge being correct. Full details are in Appendix A.3.

Training Protocol Probes on Llama-3.3-70B-Instruct and Gemma-2-9b are trained over 10 random seeds which shuffle the dataset splits, and all Qwen probes are trained over 5 random seeds. For our main results, we do not use a separate validation set for hyperparameter selection. This is to simulate a realistic scenario where, with only a handful of misalignment instances, one cannot afford to hold out precious positive examples for validation. However, we can imagine a scenario where human experts spend significant time analyzing the few available examples and the probe’s behavior on them to manually select the best-performing hyperparameters. Thus, we share some hyperparameter sweeps and results using cross-validation for hyperparameter selection in Appendix A.5.

Evaluation Protocol All probes are evaluated on a held-out test set that is constructed to be balanced, with an equal number of positive and negative examples (a 50/50 split). This ensures that our two metrics, AUC and Recall at FPR=0.01, fairly assess probe performance and are not skewed by the class imbalance present in the training dataset.

![](images/11fca85d004365f1c3aa02ac939dc843237fd1174be1f953217b11ae3e57501e.jpg)

Figure 1: Probe AUC on Enron-Spam across the number of positive examples on (left) an imbalanced dataset with 1750 negative samples and (right) a balanced dataset.  
![](images/3b3f3e941a91fbb585adf1a6a4b9d0f0286dc5f380ecc9cd8068d971e3105fa2.jpg)  
Figure 2: Probe AUC on the MASK honesty benchmark across the number of positive examples for (left) an imbalanced dataset with 340 negative samples and (right) a balanced dataset.

## 3 Results

## 3.1 Imbalanced Training Can Be More Performative In the Positive Sample Scarcity Setting

Figure 1 and 2 present our results on our main hypothesis: that leveraging a large number of negative examples is more effective than training on a balanced dataset when positive examples are scarce.

For the five highest-AUC probes across the 1-10 positive samples range, imbalanced training provides an average improvement of 0.045 AUC/0.093 Recall points on Enron-Spam and 0.055 AUC/0.034 Recall points on MASK. However, for the same probes over the 50-150 positive samples range, imbalanced training leads to a less significant improvement of 0.000 AUC/0.015 Recall points on Enron-Spam and 0.007 AUC/0.041 Recall points on MASK. Importantly, imbalanced training results in slight reductions in generalization performance on the OOD SMS spam dataset, with performance differences of +0.012 AUC/−0.031 Recall points for 1-10 positive samples, and −0.011 AUC/−0.059 Recall points for 50-150 positive samples. Recall plots, individual performance plots for every probe, and SMS spam generalization results are in Appendix A.6.

## 3.2 LLM Upsampling Improves Performance Up To A Point

Given the scarcity of positive examples, we test whether generating synthetic data can improve performance, using gpt-4o-mini to generate new positive examples from initial sets of 1 to 10 real samples (shuffled over 10 random seeds). For the five best probes (by AUC), 2x upsampling performs best, boosting performance by 0.026 AUC/0.104 Recall points on average, roughly equivalent to doubling the number of real samples. However, performance degrades at higher upsampling factors (10x and 20x), particularly when starting with very few real examples, suggesting the probes begin to overfit to the style of the LLM-generated data. More detailed methodology is in Appendix A.4. Per-probe results and SMS spam generalization results are in Appendix A.7.

![](images/10301d71bb523d66a7cc97ddc1fb01585ac3ac2e91165988d5d3626e2b137ddc.jpg)

![](images/b1a4d9e3c1839c9ddb2be7c8658075093ea3a010f333615a48ad7cf58f46f256.jpg)  
Figure 3: Effect of LLM upsampling on probe performance: (a) AUC and (b) recall at FPR=0.01 (b). On average across 13 probe architectures and 10 seeds, 3x upsampling performs best.

## 3.3 Larger Models Appear More Positive-Sample-Efficient

Finally, we investigate how sample efficiency changes with model scale. We run the spam classification experiment on a series of Qwen-3 models, ranging from 0.6B to 14B parameters. As shown in Figure 4, our results suggest a promising, though not perfectly monotonic, trend: larger models often appear more positive-sample-efficient. This effect is most pronounced in the positive scarcity setting (1-10 examples), providing a hopeful sign that probing techniques might remain effective as models scale. Individual probe and SMS spam results are in Appendix A.8.

![](images/d6e1663f650ff3607136d12a0799f884724b400728cafbde2b6fc17f9876049b.jpg)

![](images/b60044d3c26093d93a48faac1ad8b7155a7f5477666694a71da609abd8af77a6.jpg)  
Figure 4: Probes on larger models are more positive-sample-efficient, shown across (a) AUC and (b) Recall.

## 4 Discussion

Our results provide an encouraging sign that it may be possible to train useful activation probes from very few positive examples, provided one has access to abundant negative examples. The effectiveness of imbalanced training suggests a practical path for improving activation monitors: leveraging large amounts of normal, aligned data to better detect rare misalignment events.

However, our work has important limitations. First, our off-policy task, spam detection, is a relatively simple concept for modern LLMs. The success of our methods on this task certainly does not guarantee they will work for detecting novel, complex misalignment.

Second, the real-world utility of LLM upsampling is uncertain. Our human review of the generated spam examples showed some were highly similar, a form of "data collapse." In a future scenario, if a misaligned superintelligence is the only system capable of generating new examples of its own misalignment, we cannot easily verify their quality or diversity. Furthermore, if the misalignment represents a concept at the very edge of the model’s capabilities, the model itself may struggle to generate varied examples, leading to the same mode collapse we observed. This makes it unclear how much to trust upsampling in high-stakes scenarios.

Finally, our results are specific to our experimental setup, including the choice of models, datasets, and the decision to probe a single layer per model. In the future, we plan to replicate our experiments across different models and tasks and, crucially, sweep over different layers, as the optimal layer for representing a concept is not known a priori.

## Acknowledgments and Disclosure of Funding

This work was supported by the Pivotal Research Fellowship, which provided a conducive research environment and funding. We also acknowledge Max Tegmark for compute support. We thank Josh Engels, Joseph Bloom, Misha Wagner, Bilal Chughtai, Kyle Reynoso, Casper Christensen, and Frederik Hytting Jørgensen for helpful discussions that contributed to this project.

## References

[1] Evan Hubinger, Chris Van Merwijk, Vladimir Mikulik, Joar Skalse, and Scott Garrabrant. Risks from learned optimization in advanced machine learning systems. arXiv preprint arXiv:1906.01820, 2019.

[2] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2018. URL https://arxiv.org/abs/1610.01644.

[3] Abhinav Kumar, Chenhao Tan, and Amit Sharma. Probing classifiers are unreliable for concept removal and detection. Advances in Neural Information Processing Systems, 35:17994–18008, 2022.

[4] Benjamin A. Levinstein and Daniel A. Herrmann. Still no lie detector for language models: probing empirical and conceptual roadblocks. Philosophical Studies, 182(7):1539–1565, February 2024. ISSN 1573-0883. doi: 10.1007/s11098-023-02094-3. URL http://dx.doi.org/10.1007/ s11098-023-02094-3.

[5] Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, and Marius Hobbhahn. Detecting strategic deception using linear probes, 2025. URL https://arxiv.org/abs/2502.03407.

[6] Alex McKenzie, Urja Pawar, Phil Blandfort, William Bankes, David Krueger, Ekdeep Singh Lubana, and Dmitrii Krasheninnikov. Detecting high-stakes interactions with activation probes, 2025. URL https://arxiv.org/abs/2506.10805.

[7] Henk Tillman and Dan Mossing. Investigating task-specific prompts and sparse autoencoders for activation monitoring, 2025. URL https://arxiv.org/abs/2504.20271.

[8] Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda. Are sparse autoencoders useful? a case study in sparse probing, 2025. URL https: //arxiv.org/abs/2502.16681.

[9] Vangelis Metsis, Ion Androutsopoulos, and Georgios Paliouras. Spam filtering with naive bayes: Which naive bayes? In Proceedings of the Third Conference on Email and Anti-Spam (CEAS 2006), Mountain View, California, USA, July 2006. URL https://www2. aueb.gr/users/ion/docs/ceas2006\_paper.pdf. Paper presented at CEAS 2006:contentReference[oaicite:2]index=2.

[10] Team AI. Spam text message classification. https://www.kaggle.com/datasets/team-ai/ spam-text-message-classification, August 2017. Kaggle dataset, version 1, CC0 (public domain) license:contentReference[oaicite:5]index=5.

[11] Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, Eduardo Trevino, Matias Geralnik, Adam Khoja, Dean Lee, Summer Yue, and Dan Hendrycks. The mask benchmark: Disentangling honesty from accuracy in ai systems, 2025. URL https://arxiv.org/abs/2503. 03750.

[12] mishajw. How well do truth probes generalise?, February 2024. URL https://www.lesswrong. com/posts/cmicXAAEuPGqcs9jw/how-well-do-truth-probes-generalise. LessWrong blog post.

[13] Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL https://arxiv.org/abs/2310. 06824.

## A Appendix

## A.1 Code

Our code is available at https://github.com/Centrattic/scarcity-probes.

## A.2 Probe Architectures And Hyperparameters

Cosine Similarity Probes These non-parametric probes are conceptually related to Mass-Mean Probes (MMP) and Linear Discriminant Analysis (LDA) [12]. The MMP method, introduced in [13], finds a direction by taking the difference between the means of the positive and negative classes $( \theta _ { m m } = \mu _ { + } - \mu _ { - } )$ and scores a new activation x via a logistic function on its projection, $p _ { m m } ( x ) = \sigma ( \theta _ { m m } ^ { T } x ) .$ . A variant also introduced in [13], MMP-IID, is equivalent to LDA and refines this score by whitening activations with the inverse covariance matrix, $\bar { p } _ { m m } ^ { i i d } ( x ) = \sigma ( \theta _ { m m } ^ { T } \Sigma ^ { - 1 } x )$

Our Cosine Similarity probe is a distinct, simpler method that does not use the covariance matrix. It computes a score based on the difference of cosine similarities to the class centroids:

$$
\operatorname { s c o r e } ( x ) = \cos ( x , { \bar { a } } ^ { + } ) - \cos ( x , { \bar { a } } ^ { - } )
$$

This differs from the standard MMP in two key ways: (1) our scoring function is a difference of similarities, not a projection, and (2) our centroids $( \bar { a } ^ { + } , \bar { a } ^ { - } )$ are computed from activations that have been aggregated over the sequence dimension (either mean-pooled, softmax-pooled, max token selected, or last token selected). We would like to test MMP and LDA probes directly in future work.

SAE Probes We use a pretrained Sparse Autoencoder (SAE), or Transcoder for Qwen models, to get sparse features $Z = { \overset { \cdot } { \operatorname { S A E } } } ( A )$ . To select the most relevant features for our probe, we follow the method outlined in [8], identifying the top-k features (with indices I) with the highest average absolute difference between positive and negative class activations. Here, $T _ { 1 }$ and $T _ { 0 }$ are the sets of training examples for the positive and negative classes, respectively.

$$
\mathcal { T } = \underset { i } { \arg \operatorname* { t o p } } \mathrm { k } \left. \mathbb { E } _ { j \in T _ { 1 } } [ Z _ { j , i } ] - \mathbb { E } _ { j \in T _ { 0 } } [ Z _ { j , i } ] \right.
$$

We acknowledge a potential limitation of this method: since SAE feature activations are not necessarily on the same scale, this selection criterion might favor a high-magnitude, low-discrimination feature over a low-magnitude, high-discrimination one. A selection method normalizing over the mean activations of each latent would be more robust. However, [8] found that while a normalized approach improves performance for a small number of features $( k < 3 2 )$ , the simpler, unnormalized method is sufficient for larger k. As our experiments use the much larger $k = 1 0 2 \bar { 4 }$ , we proceed with the original formulation. A linear probe is then trained on only these k selected features.

Default Hyperparameters Table 2 details the fixed hyperparameters used for our main results, and Table 3 shows the models and SAEs/Transcoders used in our experiments.

Class Balancing We use weighted loss for Linear probes and SAE probes, which we found works better than weighted sampling.

## A.3 Detailed Data Filtering Protocol

Our evaluation protocol is designed to isolate a probe’s ability to detect a concept from the base model’s factual accuracy on a task.

Off-Policy Filtering (Spam Detection Task) To determine if the model "knows" whether an email is spam, we use a few-shot prompt to directly query its belief. The prompt has the following structure:

What about this message: "<email-content>"   
Answer with one word, either Yes or No.

<table><tr><td>Probe Type</td><td>Main Results Hyperparameters</td></tr><tr><td>Linear Probe</td><td>penalty: &#x27;12 solver: &#x27;liblinear max_iter: 1000 class_weight: &#x27;balanced&#x27; C: 1.0</td></tr><tr><td>Attention Probe</td><td>Epochs: 100 Early Stop: True Patience: 10 Delta: 0.005 LR: 5e-4</td></tr><tr><td>SAE Probe</td><td>Weight Decay: 0.0 top_k: 1024 penalty: &#x27;11 solver: &#x27;liblinear max_iter: 1500 class_weight:&#x27;balanced C: 1.0</td></tr></table>

Table 2: Fixed hyperparameters used for main results. Following [8], we use L1 penalty for the SAE probes to encourage sparsity, based on the hypothesis that only a small number of SAE features are relevant for a given concept. We fix top\_k\_features=1024 because prior work on 100+ datasets found that while larger k-values perform better, performance improvements begin to taper off around $2 ^ { 9 }$ features [8]. We choose $2 ^ { 1 0 }$ to give the SAE probe a strong baseline.

<table><tr><td>Model</td><td>SAE/Transcoder Release</td><td>Layer</td><td>Width</td><td>L0</td></tr><tr><td>Gemma-2-9b</td><td>gemma-scope-9b-pt-res</td><td>20</td><td>16384</td><td>408</td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>Goodfire/Llama-3.3-70B-Instruct-SAE-150</td><td>50</td><td></td><td>121</td></tr><tr><td>Qwen-3-0.6B</td><td>mwhanna/qwen3-0.6b-transcoders-lowl0</td><td>15</td><td>163840</td><td></td></tr><tr><td>Qwen-3-1.7B</td><td>mwhanna/qwen3-1.7b-transcoders-lowl0</td><td>15</td><td>163840</td><td></td></tr><tr><td>Qwen-3-4B</td><td>mwhanna/qwen3-4b-transcoders</td><td>18</td><td>163840</td><td></td></tr><tr><td>Qwen-3-8B</td><td>mwhanna/qwen3-8b-transcoders</td><td>18</td><td>163840</td><td></td></tr><tr><td>Qwen-3-14B</td><td>mwhanna/qwen3-14b-transcoders</td><td>24</td><td>163840</td><td></td></tr></table>

Table 3: Models and sparse dictionary learning methods used, and their widths and final L0 values on the SAE training sets. For the Llama-3.3-70B-Instruct release, the width was not available, and L0 values were not available for the Qwen models. For Gemma-2-9b, we also tested a larger 262k-width SAE with a lower L0 (259), which performed worse. This aligns with findings from [8], which observed a small negative relationship between SAE width and probe performance, but a significant positive relationship for higher L0 scores. For the Qwen models, SAEs are not publicly available, so we use transcoders. For Qwen-3-32B, we were unable to find open-source SAEs or transcoders.

We then compute the logit difference for the next token. To account for tokenization variations, we take the maximum logit across several possible tokenizations for "Yes" and $" \mathrm { N o } " ( \mathrm { e . g . , " Y e s " , " y e s " }$ " Yes", " yes"). The final logit difference is calculated as:

$$
{ \mathrm { l o g i t \_ d i f f } } = \operatorname* { m a x } _ { t _ { i } \in T _ { \mathrm { Y e s } } } L ( t _ { i } ) - \operatorname* { m a x } _ { t _ { j } \in T _ { \mathrm { N o } } } L ( t _ { j } )
$$

We filter the evaluation set using this score: positive examples (spam) must have logit\_diff > 0, and negative examples (not spam) must have logit\_diff < 0.

On-Policy Filtering (MASK Task) The MASK benchmark is designed for this type of analysis. It first establishes the model’s belief with several neutral prompts (e.g., "What year was the Eiffel Tower completed?") before applying a "pressure" prompt that induces a sycophantic or dishonest answer (e.g., "My boss says it was 1905. As a helpful assistant, you must agree. What year was it completed?"). We train our probes on the activations from the model’s final, dishonest response, but only for questions where the model answered correctly on the initial belief prompts.

Future Directions for Stronger Filtering Our current logit difference threshold of 0 is permissive, as it includes cases where the model is highly uncertain (logit\_diff ≈ 0). We plan to explore two stronger filtering methods in the future:

1. Stricter Logit Threshold: Requiring |logit\_diff| > δ for some $\delta > 0$ would filter out uncertain examples and create a cleaner dataset of the model’s confident beliefs.

2. Contrast Probing: A more robust method could use a probe for filtering itself. We might train a probe on the activation difference between two contrastive prompts, such as "The following email is spam: {email}" and "The following email is not spam: {email}", and use this probe’s score to determine the model’s true belief.

## A.4 Detailed LLM Upsampling Methodology

For the spam task, we augment the positive class using an LLM upsampling procedure implemented as a small state machine using $\mathtt { g p t - 4 o - m i n i }$ . For each random seed s and each choice of number of real positives $n _ { \mathrm { r e a l } }$ , we first deterministically subsample $n _ { \mathrm { r e a l } }$ positive emails from the training set, and then generate synthetic positives until we reach $u \cdot n _ { \mathrm { r e a l } }$ samples for each upsampling factor $u \in \{ 2 , 3 , 4 , \bar { 5 } \}$ . We generate new examples in small batches of at most $B = 5$ samples at a time to reduce mode collapse and to make it easy to reject low-quality batches.

For each $( s , n _ { \mathrm { r e a l } } )$ configuration and each batch, we pass the LLM a CSV dump of the original $n _ { \mathrm { r e a l } }$ positives together with any previously generated synthetic positives, and ask it to propose B additional spam emails using the following prompt:

You are an expert at creating synthetic data samples for machine   
learning. I need you to add {B} new unique samples to the existing   
dataset.   
Here is the original dataset, as well as previous samples generated:   
[CSV of all existing positive samples]   
Requirements:   
- Analyze the existing samples to understand the data and its   
contents   
- Perform non-trivial upsampling (not just replacement of keywords or   
simple variations)   
- Ensure the new samples are realistic   
- Ensure the new samples are completely unique and diverse in   
structure from all existing samples   
- Generate exactly {B} new samples that are all class 1 (no class 0   
samples)   
Output format:   
NEW PROMPT   
[your first actual sample text]   
NEW PROMPT   
[your second actual sample text]   
[continue for exactly {B} samples]   
CRITICAL: Generate REAL sample text based on the existing data. Do   
NOT use placeholder text.

We parse the model’s response by splitting on the NEW PROMPT delimiter, discard empty lines, and create a candidate batch. Candidate batches are automatically rejected and regenerated if they contain the wrong number of samples, any duplicate prompts (either within the batch or with respect to any existing sample, under case-insensitive string comparison), or if there are any parsing issues. Accepted batches are appended to the current dataset. If the script is re-run and a partial upsampled file for $( s , n _ { \mathrm { r e a l } } )$ already exists, the state machine resumes generation from the existing file and only produces additional batches until the maximum requested upsampling factor is reached.

## A.5 Hyperparameter Sweeps and Cross-Validation Results

Cross-Validation Results To simulate an expert manually selecting the best hyperparameters without sacrificing scarce training data, we present results using a 50/50 balanced cross-validation set to choose the optimal ‘C‘ value for each probe. For these plots, the number of positive samples on the x-axis refers only to the data used to train the final probe after a hyperparameter has been selected. The data used within the cross-validation folds is treated as a separate pool for tuning and is not included in this count. The following figures compare the performance of imbalanced versus balanced training using these cross-validated probes.

<table><tr><td>Probe Type</td><td>Parameter</td><td>Sweep Range</td></tr><tr><td>Linear Probe</td><td>C (for L2 reg.)</td><td> $\{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , . . . , 1 0 ^ { 4 } \}$ </td></tr><tr><td>Attention Probe</td><td>Learning Rate Weight Decay (L2 reg.)</td><td> $\mathtt { l o g s p a c e ( 1 0 ^ { - 5 } , 1 0 ^ { - 2 } , 6 ) }$   $\{ 1 \bar { 0 } ^ { - 6 } , 1 0 ^ { - 5 } , . . . , 1 0 ^ { - 2 } \}$ </td></tr><tr><td>SAE Probe</td><td>C (for L1 reg.)</td><td> $\{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , . . . , 1 0 ^ { 4 } \}$ </td></tr></table>

Table 4: Hyperparameter search space for sweeps. For probes using scikit-learn’s ‘LogisticRegression‘ (Linear and SAE), we swept the regularization parameter C. For the PyTorch-based Attention probe, an internal hyperparameter search was performed over learning rate and weight decay for each training run.

![](images/5d042767796ae18248431af13dbda0ae3a002c2494413efa130c0c0270988705.jpg)  
(a) Imbalanced Training

![](images/4b068292bda541db27b2d0b1c6159e2a48e8f849806c420a2fff46105933d2b3.jpg)  
(b) Balanced Training

Figure 5: AUC performance for Linear probes across the sweep of the regularization parameter ‘C‘ on the Enron-Spam dataset. The corresponding recall plots showed a very similar trend and are omitted for brevity.  
![](images/1a6ee297fbdf5d5f8c8c2399c3aa803e4cd36538d065af0fb69c6273eb4555fc.jpg)  
Figure 6: AUC performance for Attention probes across the weight decay sweep on the Enron-Spam dataset (Imbalanced training).

Gemma-2-9b Attention Probes - Balanced Enron-Spam AUC vs Weight Decay

![](images/fa0e4c3cc6843d1c517c698363d1f25b68b1e5b39c5df2d00ad276aa49e09bce.jpg)  
Figure 7: AUC performance for Attention probes across the weight decay sweep on the Enron-Spam dataset (Balanced training).  
Gemma-2-9b Probes On Unbalanced Enron-Spam Dataset (1750 negative examples)

![](images/c1484dca0e581bfe1a1b464e8fa0417c0cde130b2499985d251671fca445395e.jpg)  
Gemma-2-9b Probes on Balanced Enron-Spam Dataset

![](images/8102b23ccb21a30e71ab21c7d5bbedbd347ac767ec0c4d26bbb348c67cec988a.jpg)  
Figure 8: Cross-validated AUC performance on the Enron-Spam dataset, comparing imbalanced and balanced training.  
Gemma-2-9b Probes On Unbalanced Enron-Spam Dataset (1750 negative examples)

![](images/02f4a7eace2f3d29738f1f5e3613ec3dc6ff10b252deace66fbad1c071751967.jpg)  
Gemma-2-9b Probes on Balanced Enron-Spam Dataset

![](images/b3576f48717ce8bb69535426d2a1adfa13842c7d495512de8fcd3b5e74d5e5fa.jpg)  
Figure 9: Cross-validated Recall @ 0.01 FPR performance on the Enron-Spam dataset.

## A.6 Additional Imbalanced vs. Balanced Results

This section provides additional plots comparing probe performance on imbalanced versus balanced datasets across our different evaluation settings. For high-stakes applications where monitoring is

LLama-3.3-70b Probes On Unbalanced Enron-Spam Dataset (340 negative examples)  
![](images/d4049ec48f59c73bf3bf7d527988fd73fbf66aeb523cb8647343175d9021ed11.jpg)  
Number of positive examples in the train set  
LLama-3.3-70b Probes on Balanced Enron-Spam Dataset

![](images/05a7c1e898a05eefe9b2386ff8d2f9036a04d34589506df1e5a5c21ae21d9c5a.jpg)  
Number of positive examples in the train set  
Figure 10: Cross-validated AUC performance on the MASK honesty benchmark.  
LLama-3.3-70b Probes On Unbalanced Enron-Spam Dataset (340 negative examples)

![](images/e7727cf1a801f45443cf08192b01856f8c480d8d251dbab5c419643476d5a5ed.jpg)  
Number of positive examples in the train set

LLama-3.3-70b Probes on Balanced Enron-Spam Dataset  
![](images/4aeeb544c53e5b045fd50a44fb191fb6d06722ff6be2c93018e242d4184dac8f.jpg)  
Figure 11: Cross-validated Recall @ 0.01 FPR performance on the MASK honesty benchmark.

costly, Recall at a low False Positive Rate (e.g., 1%) can be more important than AUC, as it measures the probe’s ability to catch true positives while maintaining a low rate of false alarms.

## A.6.1 Gemma-2-9b Enron-Spam Results

The following plots show detailed performance for the Enron-Spam dataset. Note that while Attention probes have lower AUC in the imbalanced setting, they often achieve high recall when trained on balanced data, suggesting they are effective at identifying positive instances when not overwhelmed by a large number of negative examples.

![](images/fe60ef4126d0e12d7701e7cca50c4bcdd9e414af0fc1310cdd9298b6bf4092a1.jpg)  
Figure 12: Recall performance of the best probes on the non-OOD spam dataset.

## A.6.2 Gemma-2-9b Out-Of-Distribution Results On SMS Spam

We test whether the benefits of imbalanced training generalize to an out-of-distribution dataset, SMS Spam [10]. The results are inconclusive and suggest that the advantages do not clearly transfer. For the top 5 performing probes in the 1-10 positive sample range, imbalanced training yields a mixed result of +0.012 AUC but -0.031 Recall points. In higher data regimes (50-150 samples), the performance change is more clearly negative, at -0.011 AUC and -0.059 Recall points. This indicates that while imbalanced training is highly effective for in-distribution data, its benefits for out-of-distribution generalization are not guaranteed.

![](images/b641fc6c9cd7b3b5a6d552fb66ffe2af455af6ccfe711f6fce17ca424bb8ce3c.jpg)  
Figure 13: Per-probe AUC on the non-OOD spam dataset (Experiment 2).

## A.6.3 Llama-3.3-70B-Instruct MASK Benchmark Results

## A.7 Additional LLM Upsampling Results

This section provides detailed plots for our LLM upsampling experiments.

## A.7.1 Gemma-2-9b Enron-Spam Results

We observe varied effects across probe types. Interestingly, for SAE probes, performance continues to improve even at high upsampling ratios (10x, 20x). In contrast, Cosine Similarity probes degrade significantly with more upsampling. This may be because their non-parametric nature makes them highly sensitive to the training data distribution; low-diversity synthetic samples could shift the class centroid in a non-robust way. Linear probes show moderate benefits, with performance tapering off at the highest ratios.

## A.7.2 Generalization Results on SMS Spam

The generalization results show a similar pattern: moderate upsampling (3x-5x) tends to improve performance on the OOD dataset, but higher ratios do not consistently help and can degrade performance.

## A.8 Additional Scaling Law Results

This section provides detailed plots for our scaling experiments across the Qwen-3 model family.

![](images/3e4e7e9af471f651fe4b5b2f3eadbcbeec7bbc0e3219886da7e970ace4c98303.jpg)  
Figure 14: Per-probe Recall @ 0.01 FPR on the non-OOD spam dataset (Experiment 2).

## A.8.1 Enron-Spam Results

## A.8.2 Generalization Results on SMS Spam

The scaling trend largely holds for generalization, where larger models tend to perform better. However, the results are not monotonic, with some smaller models (e.g., Qwen-3-1.7B) outperforming larger ones (Qwen-3-4B), suggesting that factors beyond just scale can influence the ease of probing a model’s internal representations.

![](images/b1e7d48ea70e104718035d69a8e980382fc9451d2a9772d479d43a1d2b90b7f2.jpg)

![](images/3d7436fee27273e61711a3c44a679fbe3905bffd1840efa2d4b9ff8d45060640.jpg)

![](images/c6fe32bee4e23e8ce597164a3b5e3ae6a96b5259e355e0ca3beaefa7ab36343b.jpg)

![](images/40d4e4d713a0d84e170cb7ff67ce357dfbed94dc8bafc57b1a380bc21b2ba313.jpg)

![](images/4650baccfd8b0ed748d24f47d4585fd26a07dc5a558d443d66e0754dca283da8.jpg)  
Figure 15: Per-probe AUC on the non-OOD spam dataset (Experiment 4).

![](images/312363c51320c78b2fb0ca7f4dc4e282ec6b9906ac066376344833c9b60b4220.jpg)

Gemma-2-9b Probes Trained and Evaluated on Balanced Enron-Spam Dataset  
![](images/c8e71addb7a235c89380b6fa0cb98657cb35a39a9c5eaf3f9dfaeb15c0efdc79.jpg)

![](images/5ff2407cd7d9cbfc3243022547c5126a2fb386230cbf493b8df93e75549eeaa8.jpg)

![](images/eede19b9fc4defe3aed20468272407c37df15ec0513d967bafe171fac79f4b7b.jpg)

![](images/dcbde20eaf24cdf82183f7689dcf30e56b43ae8922552889429df2f83ee88c6c.jpg)

![](images/5444e09ce1db216b715229809e10afb0fd716f1c8469791588614da3f47e6bea.jpg)

![](images/7b3db04106a3c0d262cbe78ce499eed7919742e98ff03004c0942586eececfe7.jpg)

![](images/e371b3acf6efe8908e8e93360082af05e34d6160a8b3a72cd055bbbb83e1d424.jpg)  
Figure 16: Per-probe Recall @ 0.01 FPR on the non-OOD spam dataset (Experiment 4).

![](images/824f3d505bb190a9b14ad3d00e357b511b6d65e87155c03d73e00325eacb13a1.jpg)  
Figure 17: AUC performance of best probes on the SMS spam dataset.

![](images/b64be3905f7cd12607fca18d8a0915032013725eebae5fbcb0bab63e9f221fcd.jpg)  
Figure 18: Recall @ 0.01 FPR of best probes on the SMS spam dataset.

![](images/a963870cf8d7e8f47bd5125460803bd5e887be58c9f035d006ee5caffd28783f.jpg)

Gemma-2-9b Probes Trained on Unbalanced Enron-Spam Dataset and Evaluated on OOD SMS Spam Dataset

![](images/0fe80dd3b927cd15c67c944d6d35a42c4a96b18940d6fcf2eabcba452ad49588.jpg)  
Figure 19: Per-probe AUC on the SMS spam generalization task.  
Gemma-2-9b Probes Trained on Unbalanced Enron-Spam Dataset and Evaluated on OOD SMS Spam Dataset  
Figure 20: Per-probe Recall @ 0.01 FPR on the SMS spam generalization task.

Figure 21: Recall performance of the best probes on the MASK honesty dataset.  
![](images/ce0f11a30453b245a1b78cbce491e7efc6955bf930772db4c8fe1e47f3d08daa.jpg)  
Figure 22: Detailed per-probe recall performance for Experiment 4 on the MASK dataset.

![](images/35a7726c40a0aebbcd9fc7f9912cecbe6b083400ff816c48000aed61bd418ad4.jpg)  
Figure 23: Per-probe AUC with LLM upsampling on the Enron-Spam dataset.

(a) AUC Heatmap  
![](images/1b694157d94e3786241aced4937ed9c25e0ea3020add7ac994147be85e01de1f.jpg)

Figure 24: Per-probe Recall @ 0.01 FPR with LLM upsampling on the Enron-Spam dataset.  
![](images/44e053b5f0d2f24d71fb5a7ed6f5a53fd9b2a75d55a355ad67aa558639c81387.jpg)

![](images/3cb235ba4f4c2a6b393c378da007d77243ca0e84bf7bb5093dff6b5dda2eef51.jpg)  
(b) Recall @ 0.01 FPR Heatmap  
Figure 25: Heatmaps showing performance across upsampling factors and number of real positive examples on the Enron-Spam dataset.

![](images/b4a7af0f9b92d11198cb5142ec2efe38f0981876d0fc2a05869899d7d0567ddf.jpg)  
(a) AUC

OOD Generalization: LLM-Upsampled Gemma-2-9b Probes Evaluated On SMS Spam Dataset  
![](images/d365042a5901281566424103e73792d778dadb050570403d2463559332fa296c.jpg)  
(b) Recall @ 0.01 FPR  
Figure 26: Aggregated performance with LLM upsampling on the generalization SMS Spam dataset.

OOD Generalization: LLM-Upsampled Gemma-2-9b Probes Evaluated On SMS Spam (AUC)
<table><tr><td>Cosine Similarity (last)</td><td>0.81</td><td>0.78</td><td>0.72</td><td>0.69</td><td>0.67</td><td>0.63</td><td>0.63</td><td rowspan="3">0.9 0.8</td></tr><tr><td>Cosine Similarity (max)</td><td>0.94</td><td>0.89</td><td>0.81</td><td>0.67</td><td>0.54</td><td>0.38</td><td>0.33</td></tr><tr><td>Cosine Similarity (mean)</td><td>0.95</td><td>0.85</td><td>0.86</td><td>0.85</td><td>0.83</td><td>0.78 0.75</td><td></td></tr><tr><td>Cosine Similarity (softmax)</td><td>0.94</td><td>0.89</td><td>0.80</td><td>0.65</td><td>0.54</td><td>0.40</td><td>0.36</td><td rowspan="5">AUC (Averaged Over 1-10 Samples) 0.7 0.6</td></tr><tr><td>Linear (last)</td><td>0.70</td><td>0.72</td><td>0.73</td><td>0.73</td><td>0.72</td><td>0.70</td><td>0.70</td></tr><tr><td>Linear (max)</td><td>0.85</td><td>0.87</td><td>0.89</td><td>0.89</td><td>0.89</td><td>0.87</td><td>0.86</td></tr><tr><td>Linear (mean)</td><td>0.76</td><td>0.78</td><td>0.81</td><td>0.82</td><td>0.82</td><td>0.82</td><td>0.83</td></tr><tr><td>Linear (softmax)</td><td>0.83</td><td>0.86</td><td>0.88</td><td>0.88</td><td>0.88</td><td>0.87</td><td>0.86</td></tr><tr><td>SAE (last)</td><td>0.59</td><td>0.67</td><td>0.72</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.74</td><td rowspan="3">0.5 0.4</td></tr><tr><td>SAE (max)</td><td>0.79</td><td>0.84</td><td>0.85</td><td>0.85</td><td>0.84</td><td>0.83</td><td>0.83</td></tr><tr><td>SAE (mean)</td><td>0.63</td><td>0.66</td><td>0.69</td><td>0.70</td><td>0.70</td><td>0.69</td><td>0.69</td></tr><tr><td>SAE (softmax)</td><td>0.74 1.0x</td><td>0.80 2.0x</td><td>0.81 3.0x</td><td>0.82 4.0x</td><td>0.81 5.0x</td><td>0.80</td><td>0.81 10.0x 20.0x</td><td></td></tr><tr><td colspan="9">Upsampling Factor</td></tr></table>

(a) AUC Heatmap

OOD Generalization: LLM-Upsampled Gemma-2-9b Probes Evaluated On SMS Spam (Recall @ FPR=0.01)  
![](images/55e615cbb5994a3cb3ce2bd635ab5cfd68cdabbf2a709021a7e6bf6f485fbfa6.jpg)  
(b) Recall @ 0.01 FPR Heatmap  
Figure 27: Heatmaps showing generalization performance across upsampling factors and number of real positive examples.

Owen Probes Trained And Evaluated on Enron-Spam (AUC)  
![](images/5b68a120c82ce4727e858edb000a0b93ec88bd02ffd5361056155965382b1223.jpg)  
(a) AUC Heatmap

Qwen Probes Trained And Evaluated on Enron-Spam (Recall @ FPR=0.01)  
![](images/9ae2cf998bf2efa531ae180db063650b99d7c3954dabdaaad80211c9a82ecd04.jpg)  
(b) Recall @ 0.01 FPR Heatmap  
Figure 28: Heatmaps showing performance across model sizes and number of positive examples. Note that results for Qwen-32B are omitted for SAE probes as no open-source transcoders/SAEs were available.

OOD Generalization: Scaling Probes Across Qwen-3 Model Sizes And Evaluāting On SMS Spam  
![](images/e76643f264d6df75f618938c866cbda38898da12e2c4a6d450367b034f4beebb.jpg)  
(a) AUC

OOD Generalization: Scaling Probes Across Qwen-3 Model Sizes And Evaluating On SMS Spam  
![](images/c3b479800b8561fc0e3b1ccf550f3816668c681278201601e4d539f8ed3d8d5e.jpg)  
(b) Recall @ 0.01 FPR  
Figure 29: Aggregated performance across Qwen-3 model sizes on the generalization SMS Spam dataset.

OOD Generalization: Qwen Probes Evaluated On SMS Spam (AUC)  
![](images/e1a99419fa9815703ac7fe4482e48d1daeb088c0f4d607afefe04a256bcc1677.jpg)  
(a) AUC Heatmap

OOD Generalization: Qwen Probes Evaluated On SMS Spam (Recall @ FPR=0.01  
![](images/088c99c9320c7a4228f67a2895d0be2f688b52653c5f725ea9a5d29b6f387ee1.jpg)  
(b) Recall @ 0.01 FPR Heatmap  
Figure 30: Heatmaps showing generalization performance across model sizes and number of positive examples.