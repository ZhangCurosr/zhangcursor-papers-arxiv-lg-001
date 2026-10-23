# PAIR: Preference-Aligned Intent Retrieval for Premium People Recommendation at LinkedIn

Zhongkai Shangguan   
LinkedIn   
Sunnyvale, CA, USA   
zshangguan@linkedin.com   
Ruijie Jiang   
LinkedIn   
Sunnyvale, CA, USA   
rjiang@linkedin.com   
Zijian Zhou   
LinkedIn   
Sunnyvale, CA, USA   
zijzhou@linkedin.com   
Yanbin Kang   
LinkedIn   
Sunnyvale, CA, USA   
ybkang@linkedin.com   
Qiannan Yin   
LinkedIn   
Sunnyvale, CA, USA   
qyin@linkedin.com   
Girish Kathalagiri   
LinkedIn   
Sunnyvale, CA, USA   
gkathalagiri@linkedin.com

Zhengming Xing LinkedIn Sunnyvale, CA, USA zhxing@linkedin.com

Yihan Cao   
LinkedIn   
Sunnyvale, CA, USA   
yihcao@linkedin.com   
Chang Yue   
LinkedIn   
Sunnyvale, CA, USA   
cyue@linkedin.com

Changshuai Wei LinkedIn Sunnyvale, CA, USA chawei@linkedin.com

## Abstract

General-purpose embeddings trained on profile similarity often misrank candidates in professional networks, where relevance is inherently complementary rather than symmetric. For example, job seekers usually engage with recruiters rather than peers with similar credentials. Current user recommendation systems are often built on profile similarity or social signals such as shared connec tions, which are efective in friend-oriented social networks where users tend to connect with similar people or within existing social circles. However, this does not hold in professional platforms like LinkedIn, where interactions are often goal-driven and cross-role interactions, requiring matching based on complementary intents rather than similarity or social proximity. In this paper, we propose Preference-Aligned Intent Retrieval (PAIR), a retrieval framework that is designed to model user intent preference instead of profile similarity. Our approach is inspired by the preferencelearning perspective of Direct Preference Optimization, but instantiates preference learning as an exposure-conditioned multi-negative contrastive objective over embeddings. This formulation simplifies training and scales eficiently to large candidate sets without requiring an explicit reference model. Furthermore, we incorporate user interaction history by aggregating click histories with positional encoding to capture temporal order, which enables constant-cost updates of user representations over a fixed history window, allowing new interactions to be incorporated without re-encoding history. As a result, personalization can be updated eficiently using lightweight CPU operations without repeated LLM inference during serving. We evaluate PAIR on LinkedIn’s Premium retrieval task. A 0.6B-parameter model achieves a MAR@100 of 0.45. The model is deployed in LinkedIn Premium Browsemaps, where online

A/B tests show improvements in click-through rate and Premium retention without measurable latency regression.

## CCS Concepts

• Information systems → Recommender systems; Information retrieval.

## Keywords

embedding-based retrieval, preference learning, user intent modeling, real-time personalization, large language models, professional networks

## 1 Introduction

When a software engineer begins searching for a new role on LinkedIn, their browsing behavior shifts: instead of engaging with fellow engineers, they click on recruiters, hiring managers, and talent acquisition professionals. This shift illustrates a fundamental mismatch in professional network retrieval—relevance is often complementary rather than symmetric. General-purpose embedding models trained on profile similarity cluster users with similar backgrounds together, but the candidates a job-seeking engineer actually engages with look nothing like the engineer’s own profile.

This mismatch directly afects LinkedIn Premium Browsemaps, a people recommendation product for Premium subscribers. Premium Browsemaps surfaces a curated set of profiles on member profile pages, generating profile visits that feed downstream retention signals such as “Who viewed your profile” notifications. The retrieval stage defines the entire candidate space for downstream ranking: intent-aligned candidates excluded at retrieval cannot be recovered by the ranker. In our ofline evaluation, LinkedIn’s production member embedding (Holistic-171) achieves MAR@100 of only 0.288 on

Premium user retrieval, and a general-purpose of-the-shelf 0.6B LLM embedding scores even lower at 0.192. These results suggest that neither general-purpose collaborative embeddings nor generic semantic embeddings adequately model the complementary and goal-driven interactions underlying professional networking.

Deploying intent-aware retrieval at LinkedIn scale requires addressing three practical constraints. First, the training signal must capture preference rather than similarity: clicked candidates should rank above comparable non-clicked candidates, even when the non clicked candidates are more semantically similar to the query user. Second, user intent is inherently dynamic. A member actively jobsearching this week may shift toward networking or business development next month. However, repeatedly re-encoding hundreds of millions of user profiles through an LLM whenever interaction history changes is computationally infeasible in production. Third, diferent recommendation surfaces operate under diferent latency, memory, and storage constraints, requiring flexible embedding dimensionality that can adapt to multiple serving budgets within a single model.

We present Preference-Aligned Intent Retrieval (PAIR), a production-oriented retrieval framework for professional people recommendation. PAIR formulates embedding training as exposureconditioned preference learning: instead of treating negatives as arbitrary in-batch samples, it constructs preference supervision from production exposure logs, where clicked candidates are pre ferred over exposed-but-unclicked alternatives. The resulting optimization resembles multi-negative contrastive learning, but differs fundamentally in supervision semantics: the production retriever defines an implicit exposure distribution, while user clicks define preferences within that distribution. To model dynamic intent without introducing online LLM re-inference, PAIR adopts a triple-tower architecture that combines a static profile embedding with a positional-attention history embedding aggregated from recently clicked profiles. Because history embeddings are precomputed, updating user intent requires only O(Kd) vector operations over a fixed history window. Finally, Matryoshka training [11] enables a single model to produce embeddings that can be truncated to diferent dimensions for diferent latency and storage budgets across recommendation surfaces.

A 0.6B-parameter PAIR model achieves MAR@100 of 0.450 on our Premium retrieval benchmark—a 56% relative improvement over Holistic-171 and 2.3× over the of-the-shelf base encoder— demonstrating that task-specific preference supervision matters more than model scale for this retrieval task. In an online A/B test on Premium Browsemaps, PAIR improves click-through rate by 7.5% and Premium retention rate by 2.1% over the production baseline, with no measurable latency regression. Contributions:

• We formulate Premium people retrieval as exposure-conditioned preference learning, where the production retriever defines the exposure distribution and user clicks induce preference supervision. This reframes retrieval optimization from profile similarity matching toward professional intent alignment.

• We introduce PAIR, a production-oriented triple-tower retrieval architecture with additive history fusion and positional attention, enabling dynamic intent updates through �(��) vector operations without online LLM re-inference. Unlike concatenationbased or sequential approaches, history updates require only lightweight embedding-space computation.

• We provide a systematic analysis of training and serving tradeofs for LLM-based retrieval at scale, including mixed-precision stability, temperature sensitivity, history aggregation strategies, and Matryoshka dimensionality across diferent latency and storage budgets.

• We deploy PAIR in LinkedIn Premium Browsemaps and validate it through both ofline retrieval evaluation and an online A/B test, demonstrating improvements in user engagement and Premium retention without measurable serving latency regression.

The remainder of this paper is organized as follows. Section 3 describes the production system and problem formulation. Section 4 presents the PAIR framework. Section 5 reports ofline retrieval evaluation and online A/B test results.

## 2 Related Work

LLM-Based Embeddings for Retrieval: Large language models have become efective backbones for dense retrieval. Contrastive pre-training methods such as E5 [28] demonstrated that weaklysupervised text pairs can produce competitive embeddings, and subsequent work on synthetic-data-driven training [29] extended this approach to LLM-scale encoders. Recent models further improve embedding quality through architectural and training advances: LLM2Vec [1] converts decoder-only LLMs into bidirectional encoders through masked next-token prediction and contrastive learning, while NV-Embed [12] introduces latent attention pooling with two-stage contrastive instruction tuning. These advances establish that LLMs fine-tuned for embedding tasks consistently outperform smaller purpose-built encoders on retrieval benchmarks.

Within recommendation, LLMs have been used as generative recommenders [4, 15], knowledge augmentation modules [16, 23, 33], and bridges between collaborative filtering and language representations [14, 36]. At LinkedIn, LLaMA-based dual encoders have been applied to feed retrieval [22], and MixLM [13] combines embedding and text tokens for eficient reranking. A comprehensive survey by Lin et al. [17] categorizes these approaches by role.

Our work uses an LLM as the embedding backbone but targets a challenge largely unaddressed by existing methods: reshaping a general-purpose embedding space toward preference-aligned professional retrieval, where relevant candidates are often complementary rather than semantically similar to the query user.

Embedding-Based Retrieval at Production Scale: Dual-encoder retrieval architectures underpin candidate generation in most largescale recommendation systems. Dense Passage Retrieval (DPR) [9] established the dual-encoder paradigm for open-domain question answering, and Huang et al. [7] scaled embedding-based retrieval (EBR) to Facebook Search with a unified embedding framework, hard negative mining, and ANN-based serving. Yi et al. [31] addressed sampling bias in two-tower models at YouTube through streaming frequency estimation, showing that correcting for popularity bias in negative sampling substantially improves retrieval quality. Most production retrieval systems optimize semantic or collaborative similarity, which is often misaligned with professional relevance in goal-driven networking settings. In people recommendation specifically, prior work has studied retrieval architectures for employment platforms [26], while LinkedIn’s production systems highlight the importance of scalable retrieval and personalization constraints.

Our setting difers in that professional relevance is often asymmetric and goal-driven: a job seeker engages with recruiters, not with other job seekers. General-purpose member embeddings trained on broad objectives do not capture this complementary intent, motivating a task-specific preference-aligned embedding.

Preference Learning from Logged Feedback: Learning from implicit feedback is central to recommendation, but logged behavior is shaped by the production retrieval policy: clicks are only observed for candidates that were surfaced. Counterfactual learningto-rank and debiased methods address this through propensity modeling. Direct Preference Optimization (DPO) [21] optimizes language model policies from pairwise preferences without an explicit reward model, and recent work adapts preference optimization to recommendation [2]. These methods typically operate on token-level probabilities or generative recommendation policies.

The construction of informative negatives is closely related. ANCE [30] uses an ANN index to mine hard negatives during training, and Robinson et al. [24] provide theoretical analysis showing that harder negatives improve the quality of learned representations in contrastive settings. Our work is inspired by the preference-learning perspective but instantiates it as contrastive retrieval training over embeddings. Rather than mining negatives through ANN search, PAIR derives preference supervision from production exposure logs: exposed-but-unclicked candidates serve as exposure-conditioned negatives that have already passed the production retrieval filter, while random negatives provide broader coverage. This eliminates the need for a separate reference-model forward pass and naturally encodes the production retriever’s exposure distribution.

User History Modeling in Retrieval: Interaction history is a key signal for personalized retrieval. Sequential models such as HSTU [32] and its extensions [3, 6] demonstrate the value of scaling sequence-based user modeling, while multi-interest approaches [19] and large-scale user representations [34] show that users maintain multiple concurrent intents that a single static profile vector cannot capture. PinnerFormer [20] addresses the gap between batch and real-time user representations at Pinterest by training with a dense all-action loss that approximates real-time sequence inference in a batch setting. Beyond recommendation, goal-conditioned and time aware interactive systems have also demonstrated the importance of adapting model outputs to evolving user goals and temporal context [8, 25]. Such findings further motivate representations that distinguish relatively stable user characteristics from transient behavioral intent. However, many history-aware approaches require sequence-model inference or nontrivial state updates when interactions change. Such approaches become prohibitively expensive at LinkedIn scale, where user interaction histories evolve continuously. Our additive history design operates entirely in embedding space: the query representation is the sum of a static profile embedding and a dynamically aggregated history embedding, enabling constant-cost updates over a fixed history window without online LLM re-inference in the serving path. This trades of some sequence modeling expressiveness for predictable, low-cost serving behavior suited to daily or near-real-time intent refreshes.

Flexible Embedding Representations: Embedding dimensionality directly afects storage, latency, and memory bandwidth in large-scale retrieval. Matryoshka Representation Learning [11] trains embeddings that remain efective when truncated to lower dimensions. Other approaches explore sparse compression [10], analyze embedding collapse under scaling [5], or replace continuous embeddings with discrete semantic IDs [18, 35]. We adopt Matryoshka loss within PAIR so that a single model supports multiple serving configurations, from full 1024-dimensional embeddings for quality-sensitive surfaces to truncated representations for latency-constrained channels, without retraining separate models.

## 3 System Overview and Problem Formulation

## 3.1 Premium Browsemaps Product Surface

Premium Browsemaps is LinkedIn’s people recommendation prod uct tailored for Premium subscribers, designed to enhance member visibility and reinforce the value proposition of LinkedIn Premium through profile discovery and social exposure. Surfaced directly on member profile pages, the module recommends a select set of Premium profiles for users to explore. Unlike traditional social recommendation systems that primarily optimize for engagement or social proximity, Browsemaps aims to facilitate high-value professional discovery while enabling broader exposure opportunities across Premium members. From a product perspective, profile visits generated by Browsemaps create a crucial downstream feedback loop tied to Premium retention. When recommended profiles receive trafic, LinkedIn leverages visibility-related signals, such as "Who viewed your profile" notifications, to strengthen the user’s perception that Premium membership accelerates discoverability and professional reach.

At production scale, Browsemaps operates as a two-stage retrievalranking pipeline serving hundreds of millions of users. In the candidate generation stage, lightweight attribute filters (e.g.,, location, industry, seniority, and language) first narrow the Premium member pool into a manageable candidate set. An embedding-based retrieval (EBR) system then retrieves the top ∼ 100 candidates. This retrieval stage is critical, as it defines the candidate space for downstream ranking models and directly shapes both recom mendation quality and ecosystem-level exposure allocation. In the ranking stage, a downstream ranking model scores the retrieved candidates using dense features and selects the final four recommendations displayed to the user. This work focuses specifically on the retrieval stage, where we demonstrate how preference-aligned representation learning can better capture professional discovery intent beyond simple profile similarity, thereby improving both short-term engagement metrics and long-term ecosystem health.

## 3.2 Retrieval Problem in Professional Discovery

A key challenge in Premium Browsemaps is that professional discovery is often driven by complementary intent rather than symmetric profile similarity. General-purpose embedding retrieval models tend to place users with similar profile attributes close together in the embedding space. However, high-value professional interactions often occur across roles: job seekers may engage with recruiters or hiring managers, business developers may seek potential clients or partners, and talent acquisition professionals may search for candidates with complementary skills. As a result, similarity-based EBR may retrieve professionally similar members while missing candidates that better match the query user’s current discovery intent. This mismatch is particularly important in a two-stage retrievalranking system. Since downstream ranking models can only score candidates surfaced by retrieval, intent-aligned candidates excluded during candidate generation cannot be recovered later. We therefore formulate retrieval as a preference-aligned ranking problem: given a query user � and a candidate pool D, the goal is to learn a scoring function $s ( q , d )$ such that preferred candidates receive higher retrieval scores than non-preferred alternatives:

$$
s ( q , d ^ { + } ) > s ( q , d ^ { - } ) ,
$$

where $d ^ { + }$ denotes a clicked or positively engaged candidate and �<sup>−</sup> denotes an unclicked or sampled negative candidate. This preferenceoriented formulation motivates the representation learning framework introduced in Section 4.1.

## 3.3 Modeling Static Identity and Dynamic Intent

Premium subscribers difer from regular LinkedIn users in ways that directly afect retrieval design. They exhibit higher session frequency, stronger goal orientation, and more diverse engagement patterns. Premium users also fall into distinct intent clusters, including job seekers browsing recruiters and hiring managers, business developers seeking potential clients or partners, and talent acquisition professionals sourcing candidates. These intent signals are transient: a user in active job-search mode this week may shift to passive networking next month. Such behavioral dynamics are not captured by static profile attributes alone. To address this limitation, we represent each query user using two complementary sources of information: static profile identity and dynamic behavioral intent. Profile Features: Each user profile is serialized into a structured text sequence consumed by the LLM encoder. The sequence includes headline, summary, position history (title, company, duration), education (degree, institution, field), skills, certifications, patents, honors, publications, and languages. Fields are concatenated using section markers and truncated to the encoder’s maximum context length. Users without populated profile fields are excluded from the candidate pool.

Behavioral Features: User intent is modeled through click history consisting of the � most recently clicked profiles, ordered by recency. Each clicked profile is represented by a precomputed embedding obtained from the same LLM encoder applied to the clicked user’s profile text. The aggregation of these history embed dings into a single intent vector is described in Section 4.3. This design separates static identity (who the user is) from dynamic intent (what they currently need), allowing intent representations to be updated without re-encoding the user’s own profile.

## 3.4 Data Construction

We construct our training dataset using historical production logs from Premium Browsemaps. For each recommendation impression session, a member’s click on a recommended candidate is treated as a positive preference signal (�<sup>+</sup>). To robustly learn intent-aligned embeddings for retrieval, we construct a contrastive negative candidate pool (�<sup>−</sup>) for each session using a 1:3 positive-to-negative ratio. This pool consists of two components: (1) hard negatives, which are candidates that received impressions in the same session but were not clicked, and (2) easy negatives, which are randomly sampled from the active Premium member pool outside the exposed candidate set. Exposed-but-unclicked candidates are prioritized; when fewer than three are available in a session, the remaining slots are filled with random negatives. Because the logged interactions are collected from the production environment, the observed data is inherently afected by selection bias conditioned on the historically deployed retrieval policy. Rather than treating these observational clicks as absolute, static relevance labels, our framework interprets them as relative preference signals within the context of the displayed candidate set. This framing aligns naturally with our PAIR formulation (Section 4.1), where the production model’s exposure distribution serves as the implicit reference policy from which preference pairs are constructed. We use a time-based split to ensure that training interactions precede test interactions.

## 4 Method

The production pipeline described in Section 3 imposes three practical constraints on retrieval embedding learning: preference alignment beyond profile similarity, serving-eficient history updates without LLM re-inference, and flexible embedding dimensionality across products with diferent latency and storage budgets. We address these constraints with Preference-Aligned Intent Retrieval (PAIR), a triple-tower intent architecture, additive history aggregation, and Matryoshka training. Our focus is not architectural novelty alone, but retrieval learning under large-scale production serving constraints.

## 4.1 Preference-Aligned Intent Retrieval

PAIR is designed to learn retrieval embeddings from relative user preferences observed in production exposure logs. Initial fine-tuning with standard contrastive learning improved retrieval accuracy over the of-the-shelf baseline, but still produced a similarity-oriented embedding space: the model tended to cluster users with similar profiles, while many high-value professional interactions are complementary. For example, a job-seeking engineer may be matched with other engineers, even though their observed behavior indicates stronger interest in recruiters or hiring managers.

To address this mismatch, we formulate PAIR as an exposureconditioned preference contrastive learning protocol. The objective is conceptually inspired by relative preference learning, but it is not a direct embedding-space instantiation of Direct Preference Optimization (DPO). In particular, PAIR does not explicitly model a reference policy or optimize a token-level probability ratio. Instead, the historically deployed retriever defines the exposure distribution from which preference comparisons are observed.

Given a query user �, one clicked candidate $d ^ { + }$ , and � negative candidates $\{ d _ { i } ^ { - } \} _ { i = 1 } ^ { K } ,$ we define the retrieval score as:

$$
s ( q , d ) = \cos ( { \bf e } _ { q } ^ { \mathrm { f n a l } } , { \bf e } _ { d } ) / \tau ,\tag{1}
$$

where $\mathbf { e } _ { q } ^ { \mathrm { f i n a l } }$ is the final query representation, $\mathbf { e } _ { d }$ is the candidate embedding, and � is a temperature parameter. We optimize the

following multi-negative objective:

$$
\mathcal { L } _ { \mathrm { P A I R } } = - \log \frac { \exp ( s ( q , d ^ { + } ) ) } { \exp ( s ( q , d ^ { + } ) ) + \sum _ { i = 1 } ^ { K } \exp ( s ( q , d _ { i } ^ { - } ) ) } .\tag{2}
$$

Mathematically, Equation (2) has the same computational form as multi-negative contrastive learning [27]. The distinction of PAIR lies in the semantics of its supervision. Unlike conventional hardnegative mining, exposed-but-unclicked candidates are not merely close neighbors in embedding space. They are candidates selected by the deployed retrieval policy and therefore represent realistic alternatives that the production system would actually consider. PAIR therefore optimizes preference ordering within the support of the serving policy, rather than only separating arbitrary or ANNmined negatives.

Specifically, positives and negatives are constructed from production exposure logs: clicked candidates are treated as preferred, exposed-but-unclicked candidates are treated as exposure-conditioned negatives, and randomly sampled candidates provide broader coverage ofthe member space. Thus, the model learns exposure-conditioned professional preferences rather than profile similarity alone.

This design makes the training protocol compatible with productionscale embedding retrieval because it avoids an explicit referencemodel forward pass. However, because preferences are only observed within exposed candidate sets, PAIR inherits selection bias from the historical retrieval policy. We view this as a practical production tradeof: the objective improves retrieval quality under realistic exposure distributions, while future iterations can incorporate exploration-aware sampling to mitigate feedback loops.

## 4.2 Triple-Tower Intent Architecture

To separate stable profile identity from transient behavioral intent, we adopt a triple-tower architecture, as shown in Figure 1. The query profile tower encodes the query user’s serialized profile text as ${ \bf e } _ { q } = f _ { \theta } ( x _ { q } )$ , the document tower encodes candidate profiles as $\mathbf { e } _ { d } = f _ { \theta } ( x _ { d } )$ , and the history tower aggregates embeddings of recently clicked profiles into an intent vector ${ \mathbf { e } } _ { \mathcal { H } } .$ . The query profile tower and document tower share the same encoder parameters, while the history tower operates on precomputed profile embeddings.

A direct unnormalized addition of profile and history embed dings may lead to scale mismatch between towers. We therefore use normalized additive fusion:

$$
\tilde { \mathbf { e } } _ { q } = \frac { \mathbf { e } _ { q } } { \Vert \mathbf { e } _ { q } \Vert _ { 2 } } , \quad \tilde { \mathbf { e } } _ { \mathcal { H } } = \frac { \mathbf { e } _ { \mathcal { H } } } { \Vert \mathbf { e } _ { \mathcal { H } } \Vert _ { 2 } } ,\tag{3}
$$

$$
\mathbf { e } _ { q } ^ { \mathrm { f i n a l } } = \mathrm { L 2 N o r m } \left( \tilde { \mathbf { e } } _ { q } + \gamma \tilde { \mathbf { e } } _ { \mathcal { H } } \right) ,\tag{4}
$$

where $\gamma$ is a learnable scalar that controls the contribution of behavioral history. Candidate embeddings are also L2-normalized before cosine retrieval. This normalization prevents either the profile tower or the history tower from dominating the final representation due to norm diferences.

The additive design allows profile embeddings to be computed ofline, while intent updates require only lightweight vector operations when user histories change. We use Qwen3-Embedding-0.6B as the base encoder, a 0.6B-parameter model selected to balance retrieval quality and batch inference throughput. In our production setting, it can process active Premium users within the daily batch window, while larger models introduce substantially higher GPU cost without proportional retrieval-quality gains.

## 4.3 Additive History with Positional Attention

Given the � most recently clicked profiles $\mathcal { H } _ { q } = \left\{ h _ { 1 } , \ldots , h _ { K } \right\}$ , each clicked profile is represented by a precomputed embedding $\mathbf { e } _ { h _ { i } } =$ $f _ { \theta } ( x _ { h _ { i } } )$ . Because recent clicks often better reflect current user intent, we compute a position-aware attention weight:

$$
\alpha _ { i } = \frac { \exp \bigl ( \mathbf { w } ^ { T } ( \mathbf { e } _ { h _ { i } } + \mathrm { P E } ( i ) ) \bigr ) } { \sum _ { j = 1 } ^ { K } \exp \Bigl ( \mathbf { w } ^ { T } ( \mathbf { e } _ { h _ { j } } + \mathrm { P E } ( j ) ) \Bigr ) } , \quad \mathbf { e } _ { \mathcal { H } } = \sum _ { i = 1 } ^ { K } \alpha _ { i } \mathbf { e } _ { h _ { i } } ,\tag{5}
$$

where $\mathrm { P E } ( i )$ is a learned positional encoding for position �, with the most recent click assigned to $i = 1 ,$ , and w is a learnable attention vector. Both are trained end-to-end with the PAIR objective.

The attention module is intentionally query-agnostic. This design learns a global recency- and item-importance weighting over the history window, allowing the history vector to be refreshed independently of the query profile tower. Query-conditioned attention is more expressive, since it can model interactions between the current profile representation and each historical click. However, it introduces additional query-history interaction computation and complicates serving. In contrast, our query-agnostic design preserves the production property that history updates require only lightweight vector operations and no query-history joint inference.

With a fixed history window, updating e<sub>H</sub> requires only �(��) vector operations and no LLM re-inference. This is significantly cheaper than concatenating profile and history text followed by full LLM encoding. We set �=10 to cover the typical recent interaction window of active Premium users while keeping serving cost bounded. The design intentionally trades of some sequence modeling expressiveness for predictable, low-cost serving behavior, making it suitable for hourly or near-real-time intent refreshes in production. We evaluate this trade-of through pooling-based history aggregation ablations in Section 5.

## 4.4 Dimension-Flexible Embedding Training

Diferent LinkedIn surfaces operate under diferent latency and storage budgets. Premium Browsemaps can use full 1024-dimensional embeddings to maximize retrieval quality, while latency-sensitive products may require smaller representations such as 256 dimensions. Training separate models for each serving configuration would increase maintenance costs and complicate deployment.

We therefore adopt Matryoshka loss [11], optimizing PAIR jointly across multiple prefix dimensions � ∈ {200, 256, 512, 1024}:

$$
\mathcal { L } = \sum _ { d \in \mathcal { D } } \mathcal { L } _ { \mathrm { P A I R } } ^ { ( d ) } ,\tag{6}
$$

where $\mathcal { D } = \{ 2 0 0 , 2 5 6 , 5 1 2 , 1 0 2 4 \}$ , and $\mathcal { L } _ { \mathrm { P A I R } } ^ { ( d ) }$ denotes the PAIR loss computed using only the first � embedding dimensions. The 200- dimensional prefix is included to approximate the storage footprint of the production 171-dimensional embedding, while 256, 512, and 1024 dimensions represent progressively higher-quality serving configurations.

At serving time, embeddings can be truncated to the target dimension without additional computation or retraining. This enables a single model to support multiple retrieval surfaces with diferent eficiency requirements while preserving a shared training and deployment pipeline.

![](images/feff1b2d51a799c1f2096e90057a2483703981ff9d9563bb1a6761302d6c2eac.jpg)  
Figure 1: Preference-Aligned Intent Retrieval architecture. The model adopts a triple-tower design. The profile tower and document tower share parameters to encode query and candidate profiles. The history tower aggregates recent interaction embeddings using positional attention to capture short-term behavioral intent. The final query representation combines static profile identity and dynamic intent through normalized additive fusion, enabling eficient history updates without re-encoding the full profile through the LLM.

## 4.5 Production Serving Considerations

The proposed design is optimized for production retrieval rather than only ofline evaluation. Profile and candidate embeddings are computed through an ofline batch pipeline, while behavioral intent is updated through lightweight vector aggregation over precomputed embeddings. This avoids LLM re-inference when user click histories change and makes the model suitable for large-scale serving. The Matryoshka training objective further allows the same model to support multiple embedding dimensions under diferent latency and storage budgets. Premium Browsemaps uses the full embedding representation to maximize retrieval quality, while latency-sensitive surfaces can truncate embeddings at serving time without retraining a separate model.

## 5 Experiments

We evaluate PAIR along three dimensions. First, we measure ofline retrieval quality against LinkedIn’s production member embedding and an of-the-shelf 0.6B embedding encoder. Second, we analyze the contribution of history augmentation and the quality-eficiency tradeof enabled by Matryoshka dimensionality. Third, we validate whether ofline retrieval gains translate into online engagement improvements through a controlled A/B test on Premium Browsemaps.

## 5.1 Experimental Setup

Model and training: The base encoder is an internally trained 0.6B-parameter embedding model based on the Qwen architecture. We train the model with DeepSpeed in bf16 precision on 8 GPUs. In preliminary tuning, fp16 mixed-precision training caused gradient overflow under DeepSpeed, so all reported models use bf16. We set the temperature to � = 0.02 for all experiments and use cosine learning-rate scheduling with warmup and gradient accumulation. Data: We use a time-based split, with all training interactions occurring before the test window, to avoid leakage from future user behavior. The training set contains 209K query-level training instances involving 4.4M unique member profiles across queries, candidates, and history users. The test set contains 11.8K query-level instances involving 93K unique member profiles. Each instance consists of one clicked candidate as the positive and three negatives sampled from non-clicked impressions and randomly sampled members. User profiles include headline, summary, positions, education, skills, certifications, patents, honors, publications, and languages. Users with no profile information are filtered.

Table 1: Main results on Premium user retrieval. Of-theshelf models are evaluated without fine-tuning. PAIR models are fine-tuned from the 0.6B embedding encoder. ACC is computed over 4 candidates consisting of 1 positive and 3 negatives. MAR@� is evaluated on a fixed pool of 100 candidates.
<table><tr><td>Model</td><td>ACC ↑</td><td>MAR@25↑</td><td>MAR@100 ↑</td></tr><tr><td colspan="4">Baselines</td></tr><tr><td>Holistic-171</td><td>0.794</td><td>0.169</td><td>0.288</td></tr><tr><td>Off-the-shelf 0.6B embedding</td><td>0.771</td><td>0.100</td><td>0.192</td></tr><tr><td colspan="4">PAIR fine-tuned from the 0.6B encoder</td></tr><tr><td>Two-tower</td><td>0.960</td><td>0.255</td><td>0.436</td></tr><tr><td>+ Max history</td><td>0.963</td><td>0.252</td><td>0.430</td></tr><tr><td>+ Mean history</td><td>0.965</td><td>0.253</td><td>0.430</td></tr><tr><td>+ GeM history</td><td>0.966</td><td>0.264</td><td>0.447</td></tr><tr><td>+ Attention history</td><td>0.965</td><td>0.246</td><td>0.431</td></tr><tr><td>+ P-ATT history</td><td>0.967</td><td>0.265</td><td>0.450</td></tr></table>

Metrics: We report two ofline retrieval metrics. Accuracy (ACC) measures whether the model assigns the highest similarity score to the clicked candidate among four candidates consisting of one positive and three negatives. The random baseline is 0.25. Mean Average Recall at � (MAR@�) evaluates whether clicked candi dates are ranked within the top-� positions of a fixed 100-candidate evaluation pool constructed from logged candidates and sampled negatives. The same candidate pool is used for all models to ensure a fair comparison. Since ACC is computed over only four candidates and quickly reaches saturation, MAR@� provides the more informative ofline retrieval signal.

Baselines: We compare against three groups ofbaselines: (1) Holistic-171, LinkedIn’s production general-purpose member embedding with 171 dimensions, trained across multiple LinkedIn engagement objectives using both profile and behavioral signals; (2) the of-the shelf 0.6B embedding model without task-specific fine-tuning; and (3) PAIR-trained models initialized from the same 0.6B encoder, with and without history augmentation.

## 5.2 Main Ofline Results

Table 1 presents the main ofline retrieval results on the Premium user retrieval benchmark. We compare LinkedIn’s production Holistic-171 embedding, the of-the-shelf 0.6B embedding encoder without task-specific fine-tuning, and several PAIR variants trained with Premium preference supervision. The results allow us to separate the efect of preference fine-tuning from the additional contribution of history augmentation.

Preference fine-tuning is the dominant source of improvement: The PAIR model with P-ATT history achieves MAR@100 of 0.450, a 56% relative improvement over the production Holistic-171 embedding and a 2.3× improvement over the of-the-shelf 0.6B embedding model. Even without history augmentation, the two-tower PAIR model reaches MAR@100 of0.436, improving over the production baseline by 51%. This indicates that adapting the embedding space to Premium click-derived preference data is the dominant source of ofline retrieval improvement.

Table 2: History aggregation tradeof. All history methods aggregate �=10 precomputed history embeddings with 1024 dimensions. None require LLM re-inference during history updates.
<table><tr><td>Method</td><td>MAR@100 ↑</td><td>Update Cost</td><td>Operations</td></tr><tr><td>No history</td><td>0.436</td><td></td><td>Static profile only</td></tr><tr><td>Max pooling</td><td>0.430</td><td>O(Kd)</td><td>Element-wise max</td></tr><tr><td>Mean pooling</td><td>0.430</td><td>O(Kd)</td><td>Element-wise mean</td></tr><tr><td>GeM pooling</td><td>0.447</td><td>O(Kd)</td><td>Power mean</td></tr><tr><td>Attention</td><td>0.431</td><td>O(Kd)</td><td>Dot product + softmax + weighted sum</td></tr><tr><td>P-ATT</td><td>0.450</td><td>O(Kd)</td><td>Positional encoding + dot product + softmax + weighted sum</td></tr></table>

History augmentation provides a moderate additional gain: Adding P-ATT history improves MAR@100 from 0.436 to 0.450, corresponding to a 3.2% relative improvement over the history-free two-tower model. This gain is smaller than the improvement from preference-based fine-tuning, indicating that the dominant improvement comes from adapting the embedding space to Premium preference supervision. History augmentation therefore serves primarily as a deployment-eficient mechanism for incorporating dynamic intent, yielding additional gains on top of preference fine-tuning while avoiding online LLM re-inference.

ACC is close to saturation: All PAIR variants achieve ACC above 0.96, compared with the 0.25 random baseline. This suggests that the 4-candidate discrimination task is nearly saturated. We therefore treat MAR@� over the 100-candidate pool as the primary ofline metric for comparing retrieval quality.

Dimensionality alone does not explain the gain: Although PAIR uses a larger default embedding dimension than Holistic-171, the improvement is not solely due to dimensionality. As shown in Table 3, the 256-dimensional version ofPAIR still achieves MAR@100 of 0.432, substantially outperforming the 171-dimensional production baseline.

## 5.3 History Augmentation Analysis

History augmentation captures short-term intent that is not available in static profile features. We compare five aggregation methods over the � most recent clicked profiles: max pooling, mean pooling, GeM pooling, attention pooling, and positional attention. All history methods operate on precomputed profile embeddings and therefore avoid LLM re-inference during history updates.

As shown in Table 2, P-ATT achieves the highest MAR@100, followed closely by GeM pooling. The margin between P-ATT and GeM is modest, indicating that much of the benefit comes from incorporating recent behavioral context rather than from a highly expressive sequence model. Some simple aggregation methods underperform the history-free baseline, suggesting that history signals must be incorporated carefully rather than added indiscriminately.

P-ATT provides a simple way to encode recency while preserving the constant-cost update property needed for serving. The attention is intentionally query-agnostic: it learns a global recency- and item-importance weighting over the history window, allowing the history vector to be refreshed independently of the query profile tower. This design is less expressive than query-conditioned attention, but avoids query-history joint inference and keeps history updates lightweight.

Table 3: Matryoshka dimension-quality tradeof. All variants are produced by a single PAIR model trained with joint Matryoshka loss. Relative storage is normalized to the 1024-dimensional embedding. Relative latency is based on dimension-dependent retrieval cost and fixed serving overhead.
<table><tr><td>Dim</td><td>MAR@100 ↑</td><td>Rel. Storage</td><td>Rel. Latency</td><td>Intended Surface</td></tr><tr><td>200</td><td>0.421</td><td>0.20×</td><td>0.35×</td><td>Candidate pre-filtering</td></tr><tr><td>256</td><td>0.432</td><td>0.25×</td><td>0.40×</td><td>Latency-sensitive channels</td></tr><tr><td>512</td><td>0.444</td><td>0.50×</td><td>0.65×</td><td>Standard retrieval</td></tr><tr><td>1024</td><td>0.450</td><td>1.00×</td><td>1.00×</td><td>Premium Browsemaps</td></tr></table>

## 5.4 Dimension and Serving Tradeof

Embedding dimensionality directly afects storage cost, memory bandwidth, and retrieval latency in large-scale serving. We therefore evaluate the same Matryoshka-trained PAIR model under mul tiple truncated prefix dimensions. This experiment focuses on the quality-eficiency tradeof enabled by dimension-flexible embeddings, rather than comparing separately trained models for each dimension.

As shown in Table 3, full 1024-dimensional embeddings achieve the best retrieval quality, while lower-dimensional prefixes preserve much of the gain at substantially lower storage and latency cost. The 256-dimensional variant achieves MAR@100 of 0.432, outperforming the 171-dimensional production embedding while requiring only 25% of the storage of the full 1024-dimensional rep resentation. The 200-dimensional prefix is included to approximate the storage footprint of the production 171-dimensional embedding. These results indicate that a single Matryoshka-trained model can support multiple recommendation surfaces with diferent eficiency requirements without retraining separate models.

## 5.5 Online A/B Test

Ofline metrics measure retrieval quality in isolation. Online evaluation is necessary to validate whether improved candidate generation translates into user-facing engagement and business outcomes. We conducted a controlled A/B test on LinkedIn Premium Browsemaps to measure the efect of replacing the production retrieval embedding with PAIR embeddings in the recommendation pipeline.

Experiment setup: The treatment group receives recommendations retrieved using PAIR embeddings with P-ATT history augmentation at 1024 dimensions, while the control group uses the production Holistic-171 embedding. Both groups use the same downstream ranker, allowing us to isolate the efect of the retrieval embedding. Trafic is split at the user level and stratified by subscription tier and activity level to ensure balanced cohorts. Guardrail metrics, including page load latency, recommendation coverage, and serving error rate, are monitored continuously to detect potential regressions.

Online results: As shown in Table 4, replacing the production embedding with PAIR improves online engagement. CTR increases by 7.5%, indicating that the retrieval stage surfaces candidates that better match user intent before downstream ranking. Premium retention also improves by 2.1%, suggesting that better profile discovery and increased exposure are associated with improved Premium user outcomes.

Table 4: Online A/B test results on Premium Browsemaps. Treatment: PAIR with P-ATT history augmentation at 1024 dimensions. Control: production Holistic-171 embedding. Both arms use the same downstream ranker. Metrics are reported as relative changes over the control group. Statistical significance under the production experimentation framework is indicated by <sup>∗</sup>.
<table><tr><td>Metric</td><td>Relative Change</td></tr><tr><td>Click-through rate (CTR)</td><td>+7.5%*</td></tr><tr><td>Premium retention rate</td><td>+2.1%*</td></tr><tr><td>Page load latency</td><td>+0.1%</td></tr><tr><td>Recommendation coverage</td><td>-0.2%</td></tr></table>

Importantly, the treatment does not degrade serving reliability. Page load latency changes by only 0.1%, and recommendation coverage remains efectively unchanged. These guardrail results suggest that the embedding replacement improves recommendation quality without introducing measurable serving regressions.

Discussion: The online results are directionally consistent with the ofline findings in Section 5.2. Although the ofline MAR@100 improvement over the production embedding is large, the online efect size is smaller because retrieval changes are filtered through the downstream ranker and user-facing product constraints. This attenuation is expected in a multi-stage recommendation system: candidate generation improves the available candidate set, while final engagement depends on ranking, presentation, and user context.

Limitations: The A/B test evaluates PAIR within a fixed downstream ranker; joint optimization of retrieval and ranking may yield additional gains. The experiment primarily measures short-term engagement and retention outcomes within the observed experiment window, while longer-term subscription efects require extended observation. Additionally, while the mixed-negative construction used in training reflects a practical production tradeof, a more exhaustive comparison of random-only, impression-only, and mined hard negatives remains future work. Finally, the test is conducted on Premium Browsemaps only; generalization to other surfaces, such as email recommendations, job matching, or other member recommendation surfaces, remains to be validated.

## 6 Conclusion

We present PAIR, a production-oriented retrieval framework for professional people recommendation in LinkedIn Premium Browsemaps. The core idea is to learn retrieval embeddings from exposureconditioned user preferences rather than profile similarity alone. By fine-tuning a 0.6B-parameter embedding model on Premium clickderived preference signals, PAIR achieves a MAR@100 of 0.450, corresponding to a 56% relative improvement over the production Holistic-171 embedding and a 2.3× improvement over the of-theshelf base encoder. These results show that task-specific preference supervision can substantially reshape a general embedding space for professional discovery.

Beyond preference fine-tuning, we introduce a serving-eficient intent representation that combines static profile identity with dynamic click-history signals. The normalized additive fusion design captures transient user intent while allowing history updates through lightweight �(��) vector operations for a fixed history window �, avoiding LLM re-inference in the serving path. Ma tryoshka training further enables a single model to support multiple embedding dimensions, allowing retrieval surfaces with diferent latency and storage budgets to trade of eficiency and quality without retraining separate models.

## References

[1] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders. arXiv preprint arXiv:2404.05961 (2024).

[2] Yuxin Chen, Junfei Lian, Wayne Xin Huang, Wenjie Zhao, and Xing Xie. 2024. On Softmax Direct Preference Optimization for Recommendation. In Advances in Neural Information Processing Systems, Vol. 37. Curran Associates, Inc., Red Hook, NY, USA, 1–14.

[3] Yue Dong et al. 2025. Scaling Generative Recommendations with Context Paral lelism on Hierarchical Sequential Transducers. In Proceedings ofthe 19th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 1–10.

[4] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as Language Processing (P5): A Unified Pretrain, Personalized Prompt, and Predict Paradigm. In Findings ofthe Association for Computational Linguistics: ACL 2022. Association for Computational Linguistics, Dublin, Ireland, 3862–3881.

[5] Xingzhuo Guo, Junwei Pan, Ximei Wang, and Jie Jiang. 2023. On the Embedding Collapse when Scaling up Recommendation Models. In Proceedings ofthe 17th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 482–492.

[6] Lars Hertel, Neil Daftary, Fedor Borisyuk, Aman Gupta, and Rahul Mazumder. 2024. Eficient User History Modeling with Amortized Inference for Deep Learn ing Recommendation Models. arXiv preprint arXiv:2412.06924 (2024), 9 pages.

[7] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Luca Canber. 2020. Embedding Based Retrieval in Facebook Search. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, New York, NY, USA, 2553–2561.

[8] Zanming Huang, Zhongkai Shangguan, Jimuyang Zhang, Gilad Bar, Matthew Boyd, and Eshed Ohn-Bar. 2022. Assister: Assistive navigation via conditional instruction generation. In European Conference on Computer Vision. Springer, 271–289.

[9] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open Domain Question Answering. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguis tics, 6769–6781.

[10] Petr Kasalický et al. 2025. The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems. In Proceedings of the 19th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 1–11.

[11] Aditya Kusupati, Gantavya Bhatt, Aniket Rez, Matthew Wallingford, Aditya Shen, Prateek Jain, Sham Kakade, and Aravind Srinivasan. 2022. Matryoshka Representation Learning. In Advances in Neural Information Processing Systems, Vol. 35. Curran Associates, Inc., Red Hook, NY, USA, 30233–30249.

[12] Chankyu Lee, Rajarshi Roy, Menber Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. arXiv preprint arXiv:2405.17428 (2024).

[13] Guoyao Li, Ran He, Shusen Jing, Kayhan Behdin, et al. 2025. MixLM: High-Throughput and Efective LLM Ranking via Text-Embedding Mix-Interaction. arXiv preprint arXiv:2512.07846 (2025), 10 pages.

[14] Xiangyang Li, Bo Chen, Huifeng Guo, et al. 2023. CTRL: Connect Collaborative and Language Model for CTR Prediction. In Proceedings ofthe 17th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 188–198.

[15] Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, and Chunxiao Xing. 2023. E4SRec: An Elegant Efective Eficient Extensible Solution of Large Language Models for Sequential Recommendation. arXiv preprint arXiv:2312.02443 (2023), 10 pages.

[16] Yunqi Li, Hanxiong Chen, Shuyuan Xu, Yingqiang Ge, Juntao Tan, Shuchang Zhu, and Yongfeng Zhang. 2023. KAR: Enabling Knowledge-Augmented Reasoning in Large Language Models for Recommendation. arXiv preprint arXiv:2306.10933 (2023), 12 pages.

[17] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, and Weinan Zhang. 2025. How Can Recommender Systems Benefit from Large Language Models: A Survey. ACM

Transactions on Information Systems 43, 2 (2025), 1–47.

[18] M. Jefrey Mei et al. 2025. Semantic IDs for Music Recommendation. In Proceedings ofthe 19th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 1–11.

[19] Yue Meng et al. 2025. User Long-Term Multi-Interest Retrieval Model for Recommendation. In Proceedings of the 19th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 1–10.

[20] Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg. 2022. Pinner-Former: Sequence Modeling for User Representation at Pinterest. In Proceedings of the 28th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, New York, NY, USA, 3702–3712.

[21] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Advances in Neural Information Processing Systems, Vol. 36. Curran Associates, Inc., Red Hook, NY, USA, 53728–53741.

[22] Sudarshan Srinivasa Ramanujam, Antonio Alonso, Saurabh Kataria, et al. 2025. Large Scale Retrieval for the LinkedIn Feed using Causal Language Models. arXiv preprint arXiv:2510.14223 (2025), 12 pages.

[23] Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Representation Learning with Large Language Models for Recommendation. In Proceedings ofthe ACM Web Conference 2024. ACM, New York, NY, USA, 3464–3475.

[24] Joshua Robinson, Ching-Yao Chuang, Suvrit Sra, and Stefanie Jegelka. 2021. Contrastive Learning with Hard Negative Samples. In Proceedings of the 9th International Conference on Learning Representations.

[25] Zhongkai Shangguan. 2026. Advancing Assistive Systems With Multimodal Machine Learning. Ph. D. Dissertation. Boston University.

[26] Jianqiang Shen, Yuchin Juan, Shaobo Zhang, et al. 2024. Learning to Retrieve for Job Matching. arXiv preprint arXiv:2402.13435 (2024), 10 pages.

[27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. arXiv preprint arXiv:1807.03748 (2018), 13 pages.

[28] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv preprint arXiv:2212.03533 (2022).

[29] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving Text Embeddings with Large Language Models. arXiv preprint arXiv:2401.00368 (2024), 15 pages.

[30] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In Proceedings ofthe 9th International Conference on Learning Representations.

[31] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. 2019. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 269–277.

[32] Jiaqi Zhai, Lucy Liao, Xing Liu, et al. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. In Proceedings ofthe 41st International Conference on Machine Learning. PMLR, Vienna, Austria, 58555–58570.

[33] Chao Zhang, Shiwei Wu, Haoxin Zhang, et al. 2024. NoteLLM: A Retrievable Large Language Model for Note Recommendation. In Proceedings ofthe 18th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 170–180.

[34] Wei Zhang, Jiaqi Zhai, Xueming Wang, et al. 2023. Scaling User Modeling: Largescale Online User Representations for Ads Personalization at Meta. In Proceedings ofthe 17th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 453–463.

[35] Carolina Zheng, Minhui Huang, et al. 2025. Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID. In Proceedings ofthe 19th ACM Conference on Recommender Systems. ACM, New York, NY, USA, 1–10.

[36] Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, andJundong Li. 2024. CLLM4Rec: Collaborative Large Language Model for Recommender Systems. In Proceedings of the ACM Web Conference 2024. ACM, New York, NY, USA, 3400–3411.