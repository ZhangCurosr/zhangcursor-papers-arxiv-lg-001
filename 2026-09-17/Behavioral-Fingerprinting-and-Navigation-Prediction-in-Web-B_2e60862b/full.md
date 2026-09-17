# Behavioral Fingerprinting and Navigation Prediction in Web Browsing

Ralph Elsaghbini

Computer sciences and networks department (INFRES)

Institut Polytechnique de Paris, Telecom Paris,

Palaiseau, France

ralph.elsaghbini@telecom-paris.fr

Walid Fahs

Faculty of Engineering

Islamic University of Lebanon, Faculty of Engineering

Wardanieh, Lebanon

walid.fahs@iul.edu.lb

Omran Berjawi

Computer sciences and networks department (INFRES)

Institut Polytechnique de Paris, Telecom Paris,

Palaiseau, France

omran.berjawi@telecom-paris.fr

Rida Khatoun

Computer sciences and networks department (INFRES)

Institut Polytechnique de Paris, Telecom Paris,

Palaiseau, France

rida.khatoun@telecom-paris.fr

Abstract—Web browsing often appears ephemeral: users visit a few websites, complete a task, and move on. However, even short fragments of browsing activity can contain rich and structured behavioral signals. In this work, we conduct a comparative empirical study of two complementary behavioral inference tasks: session-level user identification and next-domain prediction. Both tasks are derived from the same cleaned event stream and evaluated on large-scale anonymous browsing traces, with sessionization and splitting adapted to the temporal requirements of each task. For user identification, we evaluate classical and neural models operating on session-level behavioral and domain features. For next-domain prediction, we combine graph-based modeling with Large Language Models (LLMs). Experimental results show that short browsing sessions are highly identifiable, while future navigation actions are highly predictable from long-term interaction structure combined with recent behavioral context. Furthermore, LLM-derived semantic features yield only marginal gains over purely structural and sequential models, indicating that repeated interaction patterns remain the dominant predictive signal in the evaluated webbrowsing setup. These findings highlight the extent to which interaction history substantially contributes to both user identifiability and navigation predictability in browsing traces.

Index Terms—Web browsing behavior, user identification, behavioral predictability, and privacy risk.

## I. INTRODUCTION

Web browsing is one of the most ubiquitous human behaviors in digital contexts. Users constantly engage with different sources, like news websites, search engines, social networks, and other web services, using short and disconnected browsing sessions. Despite their apparent ephemeral nature, such human behaviors emerge from consistent processes at various temporal scales. Previous research has proven the presence of significant regularities in human browsing behavior, which are formed due to human habits and repeated visits to certain websites [1], [2]. This suggests that even seemingly noisy browsing traces may encode structured behavioral signals.

Two main research perspectives have been developed to explore this structure: First, there is behavioral predictability that concerns how accurately future behaviors can be predicted from previous ones. The common approach for this line of work is sequential analysis of navigational patterns by predicting the next visited domain from the previous one [1], [3]. Second, there is behavioral identifiability that considers whether particular individuals could be uniquely identified from their navigation behavior and domain preferences. This perspective has also become known as behavioral fingerprinting or re-identification [4], [5].

Although both identifiability and predictability originate from user domain interaction patterns, they are typically studied independently across prior work. Such an approach to behavioral analysis causes certain limitations because short-term temporal dynamics are commonly analyzed for predictability and long-term dynamics for identifiability. In addition, the findings within both perspectives are highly sensitive to methodological choices, including the process of sessionization, temporal splitting and representationlearning [1], [3]. Thus, it becomes hard to identify how much of the observed structure is related to the actual behavioral regularities.

In this paper, we study two complementary behavioral inference tasks derived from web browsing activity: sessionlevel user identification and next-domain prediction. Our objective is to systematically compare how classical machine learning models, neural architectures, graph-based methods, and LLM-augmented approaches behave across these two related tasks. For the user identification task, each session is transformed into a fixed-dimensional representation composed of behavioral statistics, visited-domain information, and demographic-related features. We then evaluate multiple classification models, including linear, tree-based, neural, and representation-learning approaches, to measure how strongly user identity is encoded in short browsing activity.

For the next-domain prediction task, we propose and evaluate the Structure-Dominant Hybrid Behavioral Model (SD-HBM), which combines graph representations and LLMderived semantic features for next-domain prediction. We evaluate all models on the same underlying event stream, using a common preprocessing pipeline with task-specific sessionization and splitting. For next-domain prediction the protocol is leakage-safe by construction, with chronological per-user splits and an interaction graph built exclusively from training sessions. This allows us to systematically quantify how different modeling families exploit the same behavioral signals and to assess the relative contribution of structural, sequential, and semantic information in browsing traces. The contributions of this work are as follows:

• We conduct a comparative empirical study of two behavioral inference tasks derived from web browsing traces: session-level user identification and next-domain prediction.

• We evaluate classical, neural, graph-based, and LLMaugmented models under a consistent experimental pipeline.

• We analyze the role of structural interaction patterns in both identity inference and navigation prediction.

• We quantify the effect of LLM-based semantic augmentation across different integration strategies.

The remainder of this paper is organized as follows. Section II reviews related work. Section III introduces our behavioral inference framework and describes the modeling approaches. Section IV presents the experimental setup. Section V reports results. Section VI and Section VII discuss the results and the limitations. Section VIII concludes the paper.

## II. RELATED WORK

Web browsing behavior has been researched within multiple communities, including web science, recommendation systems, user modeling, and privacy studies.

Web browsing behavior is highly regular over time: users repeatedly revisit a small set of websites, forming stable habits. Kulshrestha et al. [1] demonstrate that browsing behavior is highly predictable but highly dependent on the methodology applied, including the way sessions are identified, pre-processing techniques, and the representation of features. Research in media psychology similarly reports stable browsing routines, with stronger habit associated with higher online predictability [2].

Significant effort has been devoted to sequence and interaction modeling for next-action predictions. While earlier solutions based on Markov models and recurrent neural networks have been proposed, later research employed attention and Transformer-based architectures to enable the capturing of more long-range dependencies among interactions [6]–[8]. However, as shown empirically, there is no guarantee that the use of complicated architectures leads to an improvement in performance [9], [10].

In addition to sequential models, some techniques consider higher-order structure in interaction datasets. Motif-based techniques capture repeated patterns of user activity [11], while representation learning approaches leverage long-term user preference modeling using multi-session clickstream data [12]. This work underscores the notion that browsing behavior is characterized not only by immediate dependencies but by multi-scale structure.

In parallel to prediction-oriented efforts, other literature explores whether browsing behavior could lead to the identification of an individual. Existing literature highlights that a user’s browsing history and visited domains could serve as a fingerprint and make it possible to perform re-identification based on partial trace evidence [4]. Privacy literature further highlights that user behaviors are a very sensitive type of attribute, which can be revealed even via aggregation or anonymization techniques. Analysis of privacy-preserving advertising mechanisms, including Google’s Topics API, shows that it is possible to launch successful re-identification attacks [13]. Measurement results indicate that browsing behaviors can still leak user information even without direct identifiers in place [14].

Our work performs controlled experiments to analyze how different modeling families, including classical machine learning models, graph-based methods, neural sequence models, and LLMs, capture behavioral regularities in web browsing traces.

## III. BEHAVIORAL MODELING FRAMEWORK

We model web browsing traces as structured behavioral signals that combine persistent user-specific patterns with transient contextual dynamics.

## A. Browsing Traces as Behavioral Signals

We consider a set of browsing events $\left( { { u } _ { i } } , { { d } _ { i } } , t _ { i } \right)$ , where each event corresponds to a user $u _ { i }$ visiting a domain $d _ { i }$ at time $t _ { i } .$ These events form temporally ordered traces that reflect repeated interactions between users and web domains as a structured behavioral process:

• Persistent structure: stable user-specific preferences that manifest as repeated visitation patterns over time.

• Transient structure: short-term contextual effects that influence immediate navigation decisions within sessions.

To capture these dynamics, we segment browsing traces into sessions using a time-gap rule. A session $\begin{array} { r l } { S } & { { } = } \end{array}$ $( d _ { 1 } , d _ { 2 } , \ldots , d _ { T } )$ represents a contiguous episode of activity that reflects both habitual behavior and immediate intent.

## B. Two Complementary Inference Tasks

We study two inference tasks that probe different aspects of the same underlying behavioral structure: one focuses on user identifiability, while the other focuses on navigation predictability.

• Session-Level User Identification: given a short browsing session S, the goal is to infer the identity of the generating user u. This task measures the extent to which persistent behavioral signatures are expressed in short activity fragments. High performance indicates that userspecific preferences are strongly encoded even in limited observations.

• Next-Domain Prediction: given a user u and a recent sequence of visited domains $( d _ { t - L + 1 } , \ldots , d _ { t } )$ , the goal is to predict the next domain $d _ { t + 1 }$ . This task captures short-term navigation dynamics and measures how predictable behavior is from recent context combined with long-term interaction history.

## C. Modeling Approaches

The modeling instantiations are used to operationalize the tasks introduced above. We distinguish between models designed to capture session-level identity signals and models designed to capture navigation dynamics over user–domain interactions.

1) Session-Based Identity Models: Each session is encoded as a fixed-dimensional feature vector combining behavioral statistics, domain-level signals, and demographic attributes, from which the models estimate a posterior $p ( u \mid S )$ over users. We evaluate four families of increasing expressive capacity:

• Linear Support Vector Machine (SVM): a linear classifier used to assess whether identity-related signals are linearly separable in the session feature space, serving as a baseline for the inherent structure of identity information.

• Random Forest (RF): a non-linear ensemble model that captures interactions between behavioral features and domain-level signals while remaining robust to sparsity and heterogeneous feature distributions.

• Multilayer Perceptron (MLP): a neural network that learns complex non-linear combinations of behavioral and domain features, capturing higher-order dependen cies that may encode user-specific behavioral signatures.

• Autoencoder-Based Model (AE): a representationlearning approach in which session features are first compressed into a latent space via reconstruction, after which a classification head is trained on the learned representation to evaluate whether identity information is preserved under dimensionality reduction.

2) Structure-Dominant Hybrid Behavioral Model (SD-HBM): We propose the Structure-Dominant Hybrid Behavioral Model (SD-HBM) for next-domain prediction, which models browsing as a structured interaction process over a heterogeneous bipartite graph $G = ( \mathcal { U } \cup \mathcal { D } , E )$ , where edges represent observed user–domain interactions. To account for repeated visits, edge weights use a log-scaled interaction frequency $w ( u , d ) = \log ( 1 + \mathrm { c o u n t } ( u , d ) )$ , which limits the influence of highly repetitive interactions while preserving relative preference strength.

a) Long-Term Structural Representation: To encode persistent behavioral preferences, SD-HBM employs a twolayer GraphSAGE encoder operating on the bipartite graph G. The encoder performs iterative neighborhood aggregation, allowing each user representation to be computed from both directly connected domains and higher-order connectivity patterns reached through shared domains. This yields structural embeddings that capture long-term interests and stable browsing tendencies, and that place users who visit overlapping sets of domains close together in the embedding space.

b) Short-Term Sequential Modeling: Short-term navigation behavior is modeled using a GRU-based sequence encoder applied to the most recent L visited domains. The GRU captures temporal dependencies and transition patterns within user sessions, summarizing the recent trajectory into a single state. To enhance temporal sensitivity, embeddings representing hour-of-day and day-of-week are incorporated into the sequential representation. In addition, the embedding of the last visited domain is explicitly concatenated to preserve immediate transition signals, which are often strongly predictive in browsing data.

c) LLM-derived semantic feature representation: To complement structural and sequential signals, SD-HBM incorporates an LLM-based feature extraction module. The learned graph-based user embedding is provided to the LLM under a constrained structured prompt, and the model returns a compact set of semantic descriptors summarizing user behavior: dominant and secondary browsing categories, behavioral diversity, routine strength, and category-level affinities such as social, news, productivity and media. These descriptors provide a higher-level semantic abstraction of the learned interaction representation.

d) Hybrid Behavioral Fusion: Finally, all representations are integrated into a single latent vector. The GraphSAGE user embedding, GRU-based sequential embedding, temporal encodings, last-domain embedding, and LLM-derived semantic features are concatenated and passed through a feed-forward prediction head that scores candidate domains. The model is trained in two stages, with the graph encoder pretrained and subsequently frozen, and the sequence head trained for next-domain prediction.

## IV. EXPERIMENTAL SETUP

## A. Dataset Overview

We conduct our experiments using a large anonymous web browsing dataset that has been made available through Zenodo<sup>1</sup>. The dataset consists of time-stamped browsing sessions that have been captured from a group of users within about a month. The data includes information about the identifier of the users, visited domain names, time stamps, and the active duration of browsing sessions.

Additionally, demographic features of the users and a domain-to-category map are provided. Auxiliary features like demographic details and the domain category map are exclusively used for generating features for the session-based identification task and are never used to build future information in the prediction task. After filtering out incomplete records, we obtain a final dataset of more than nine million browsing events from over two thousand users. The dataset is released publicly in anonymized form: users are represented by opaque panel identifiers, and no URLs, page content, or directly identifying attributes are included. Our analysis operates exclusively on domain-level records.

## B. Preprocessing and Session Construction

All experiments are based on a common preprocessing pipeline applied to raw browsing event logs. Events missing a user identifier, domain, or timestamp are removed, and the remaining records are sorted chronologically per user. Sessions are then constructed using a time-gap heuristic: a new session is initiated whenever the interval between consecutive events for the same user exceeds a threshold ∆. This procedure yields temporally coherent sequences that reflect both short-term navigation patterns and longer-term browsing behavior.

We adopt different sessionization thresholds for the two tasks to reflect their distinct temporal requirements. For session-level user identification, we evaluate short inactivity gaps ranging from 1 second to 30 minutes and select a 2-second threshold, which preserves fine-grained behavioral fragments while minimizing noise from unrelated actions. At this resolution, a session corresponds to a tightly grouped burst of requests, which may include resources loaded alongside a single deliberately visited page. For next-domain prediction, we test thresholds between 2 seconds and 120 seconds and select 30 seconds, which provides a balance between sequence coherence and sufficient length for predictive modeling.

Threshold selection was carried out during pipeline development as an exploratory design choice based on preliminary runs; sessionization parameters are fixed before model training and are not tuned against the reported test results.

## C. Task Construction

a) Task 1: Session-Level User Identification.: Each session is encoded into a fixed-dimensional feature vector. Activity features summarize the volume and intensity of the session, including the number of visits, the number of distinct domains visited, and the total and average active time. Temporal features record when the session occurred, using hour of day and day of week together with cyclical encodings. Repetition features describe how concentrated a session is on a small number of domains. Visited domains are represented in two complementary ways: explicit visit counts for the most frequently visited domains in the corpus, and a hashed representation that maps the remaining long tail into a fixed number of bins, preserving domain-level information without an unbounded feature space. Category features aggregate visits through the domain-to-category map, including per-category counts and shares, the number of distinct categories, and the entropy of the category distribution.

A small number of cross-session features relate each session to the immediately preceding session of the same user, namely the elapsed time between them and the overlap of their domain and category sets. Finally, demographic attributes are encoded as indicator variables. The dataset is thus composed of sample sessions coupled with user identities. The prediction task is modeled as a multi-class classification over users. To limit class imbalance across users with very different activity levels, we retain the first 50 sessions per user and discard users with fewer than two sessions. This yields 2,140 users and 103,314 sessions for this task, and accounts for the difference in user counts between the two tasks. Table I presents the main properties of Task 1 dataset.

TABLE I  
DATASET CHARACTERISTICS FOR TASK 1 (SESSION-LEVEL USER IDENTIFICATION).
<table><tr><td>Characteristic</td><td>Value</td></tr><tr><td>Time span</td><td>October 2018</td></tr><tr><td>Number of users (classes)</td><td>2,140</td></tr><tr><td>Number of sessions</td><td>103,314</td></tr><tr><td>Number of raw browsing events</td><td>~9 million</td></tr><tr><td>Session definition</td><td>Time-gap rule (2 seconds)</td></tr><tr><td>Train/Test split</td><td>80% / 20% (stratified by user)</td></tr></table>

b) Task 2: next-domain prediction.: In order to predict the next domain in the navigation sequence, supervised samples are generated through a sliding window over sessions. Based on previously visited L domains, the model makes predictions over the next domain. In order to make the prediction easier by limiting the size of the output space, only the top K frequently visited domains are considered in the training data. Table II reports the main statistics of the Task 2 dataset. Domains outside this vocabulary are mapped to a single UNK class, which groups the long tail of rarely visited sites. UNK is treated as an ordinary prediction target rather than being excluded from evaluation: samples whose true next domain falls outside the vocabulary remain in the test set and are counted in the accuracy denominator. Supervised samples are generated only from sessions containing more than L events, so the reported predictability characterizes sustained browsing episodes rather than the full session population.

TABLE II  
DATASET CHARACTERISTICS FOR TASK 2 ( NEXT-DOMAIN PREDICTION).
<table><tr><td>Characteristic</td><td>Value</td></tr><tr><td>Number of users</td><td>2,148</td></tr><tr><td>Number of raw browsing events</td><td>9,151,243</td></tr><tr><td>Number of sessions (gap = 30s)</td><td>2,215,002</td></tr><tr><td>Sequence length L</td><td>10</td></tr><tr><td>Number of supervised samples</td><td>356,707</td></tr><tr><td>Target vocabulary size</td><td>Top-10,000 + UNK</td></tr><tr><td>UNK fraction</td><td>0.78%</td></tr><tr><td>Split strategy</td><td>Chronological per user</td></tr></table>

## D. Implementation Details

We conduct empirical hyperparameter tuning over architectural and optimization configurations. For session-level user identification, a validation set is held out from the training partition and used for model selection and early stopping. For next-domain prediction, configurations are selected on the held-out evaluation split under a fixed training budget, as described below.

1) Task 1: Session-Level User Identification: SVM and RF use standard library implementations, for classical machine learning baselines. The best-performing model is a residual MLP with seven fully connected layers. Layers 1 to 5 each contain 1024 hidden neurons, followed by a sixth layer with 512 neurons. The final layer maps to the user classes. Each hidden layer is followed by ReLU activation, Batch Normalization (BatchNorm1d), and Dropout with a rate of 0.2 to reduce overfitting. The model is trained using the Cross-Entropy loss with label smoothing set to 0.05. Optimization is performed using Adam with a learning rate of 0.001, batch size of 1024, and a maximum of 200 training epochs. Early stopping is applied with a patience of 8 epochs based on validation Top-1 accuracy.

2) Task 2: next-domain prediction (SD-HBM): Structure-Dominant Hybrid Behavioral Model (SD-HBM) extends the Graph-Sequence Model (GSM) with LLM-based semantic feature augmentation.

A two-layer GraphSAGE encoder is first applied over the constructed interaction graph G to learn 128-dimensional user and domain embeddings. The encoder is trained using a pairwise ranking objective over observed edges with negative sampling (five negatives per positive interaction). Optimization is performed using AdamW with a learning rate of 0.001 for 5 epochs, a fixed budget selected during development; the encoder is subsequently frozen and used as a static representation.

Then, short-term navigation behavior is modeled using a GRU encoder over the last L = 10 visited domains, each represented by a 128-dimensional embedding. The GRU has one layer with hidden size 128, and its final hidden state is used as the sequence representation. We additionally include contextual features consisting of the 128-dimensional user embedding, the last-domain embedding, and temporal embeddings for hour-of-day (16D) and day-of-week (8D).

All components are concatenated into a 408-dimensional representation. When LLM-derived semantic features are included, the representation dimension increases to 418. The fused vector is projected into a 128-dimensional space using a fully connected layer with GELU and 0.2 dropout, followed by L2 normalization. The model is trained using an InfoNCE / sampled softmax objective optimized with AdamW (learning rate 0.002, batch size 4096) and gradient clipping with norm 1.0.

For LLM-based semantic feature generation, we augment the model with semantic features generated by GPT-5.4-mini, using the provider’s default decoding settings. Generation is constrained by a strict JSON schema that fixes the set of output keys and restricts numeric fields to [0, 1], which bounds the output space; we did not measure variability across repeated calls for the same input.

To contextualize the performance of the proposed SD-HBM, we evaluate three complementary baseline approaches under the same experimental setting. The Graph-Based Markov Model (GBM) defines a probabilistic random walk over the user–domain interaction graph, where transition probabilities are estimated from normalized visit counts. Predictions are obtained by a random walk with restart, where the restart distribution is concentrated on the current user and on the most recently visited domains, with greater weight assigned to more recent visits. The Graph-Sequence Model (GSM) combines GraphSAGE-based user and domain embeddings with a GRU-based sequence encoder to jointly model long-term interaction structure and short-term navigation dynamics without semantic augmentation. In addition, we consider a semantic augmentation baseline, referred to as the GSM + LLM Reranking, where the GSM first produces a ranked list of candidate next domains, and a LLM is then used to rerank these candidates based on the recent browsing sequence, user context, and optional domain category information by estimating their semantic plausibility. All nextdomain models are trained on chronologically ordered splits, and the interaction graph is constructed from training sessions only. Model selection for this task is performed on the heldout evaluation split, so the reported figures correspond to the best epoch under a fixed training budget rather than to an independently validated configuration. The same protocol is applied to every model in this task, which keeps the compar ison internally consistent. Standard regularization techniques, including early stopping, dropout, and gradient clipping, are applied consistently across neural models to ensure stable training and fair comparison. In terms of computational cost, the classical identification baselines train on CPU within minutes, whereas the neural and graph-based models require GPU acceleration; graph encoder pretraining is the most expensive stage, and inference in all cases is a single forward pass.

## E. Evaluation Metrics

Evaluation of all models across both tasks is performed based on Top-1 accuracy to ensure comparability of inference performance under a unified evaluation criterion. It measures the ratio of correctly predicted samples to the total number of evaluated samples. The accuracy is defined for a set of predicted labels $\hat { y } _ { i }$ and actual labels $y _ { i }$ by:

$$
\mathrm { T o p - 1 \ a c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ \hat { y } _ { i } = y _ { i } ] .\tag{1}
$$

where N is the total number of samples.

Top-1 accuracy is a strict criterion in both settings: the label space contains 2,140 users for identification and 10,001 classes for next-domain prediction, so a uniform random predictor would achieve well below one percent in either task. We report Top-1 only in order to keep a single comparable criterion across model families that produce scores in different ways. Richer ranking metrics, such as Recall@k or mean reciprocal rank, would give a fuller picture of how candidates are ordered below the top position, and we leave their systematic reporting to future work.

## F. Evaluation Protocol

Our experimental design aims to avoid temporal and structural leakage during training and evaluation. The following constraints are enforced:

• For next-domain prediction, sessions are ordered chronologically per user and split so that earlier sessions are used for training and later sessions are held out for evaluation. For session-level user identification, sessions are partitioned using a stratified random split over users (Table I), which preserves the class distribution across all identities. Because this split is not chronological, sessions recorded close together in time may fall on opposite sides of the partition, and the reported identification accuracy should therefore be interpreted as an upper bound relative to a strictly temporal protocol.

• Graph construction from training data only: Graphs used for navigation task are generated solely using training sessions; there are no test-time interactions in these graphs.

• No future information in features: Session-level features are computed from events within the session and from preceding sessions only; no feature depends on activity that occurs later in the same user’s trace.

• Consistent evaluation protocol: Models are all evaluated using the same test splits across all statistical, neural, graph-based, and LLM-based approaches.

## V. EXPERIMENTAL RESULTS

This section presents empirical findings for both tasks. Results are interpreted based on their relevance to understanding the underlying nature of the web browsing activity. Specifically, we analyze (i) whether short browsing sessions are identifiable, (ii) how predictable navigation behavior is, and (iii) whether semantic augmentation via LLMs provides additional behavioral signal.

## A. Are Browsing Sessions Identifiable?

First, we look at whether short browsing sessions are enough to enable identifying the generating users. Table III summarizes the performance of all models in terms of user identification at the session level, depending on model expressiveness. There is a noticeable trend that can be observed here. Specifically, linear models have poor results (30.5% accuracy), which implies that the identity signals cannot be separated using linear algorithms. Nonetheless, the performance grows with increasing complexity, with tree models obtaining impressive results (84.8%) and neural models reaching the best performance, with MLP getting 89.8% of Top-1 accuracy.

This indicates that the behavioral patterns are already sufficiently distinct even for short periods of web navigation. Most importantly, there is no single feature that holds identityrelated signals – the information about users is extracted from complicated interactions between behavior, domains, and demographics.

## B. How Predictable is Navigation Behavior?

We next analyze the predictability of short-term navigation behavior. Table IV reports results for graph-based, sequential, and LLM-enhanced models. The GBM model achieves 71.44% accuracy, showing that transition statistics over the interaction graph already capture pronounced repetition in user navigation. Because the walk restarts predominantly from recently visited domains, this figure reflects a combination of persistent user–domain structure and short-term recency rather than long-term structure alone.

Building on this, the GSM model improves performance to 79.38%, demonstrating the benefit of combining long-term structural embeddings with short-term sequential modeling.

Further gains are observed when incorporating LLM-based components. The GSM + LLM Reranking model achieves 79.84%, showing that semantic reordering of candidate domains can refine predictions produced by the structural model. Finally, the proposed SD-HBM model achieves the best performance with 80.15%, indicating that integrating semantic features directly into the representation space is more effective than post-hoc reranking.

## C. Do LLMs Add Behavioral Information?

Across all configurations, LLM-based components provide consistent but small improvements over the strongest non-LLM baseline (GSM, 79.38%): reranking yields 79.84%, and feature augmentation reaches 80.15% (SD-HBM). Semantic features therefore contribute complementary information, but the dominant predictive signal remains encoded in the interaction graph and recent behavioral sequences, and integrating semantic features into the representation is more effective than post-hoc reranking.

TABLE III  
SESSION-LEVEL USER IDENTIFICATION PERFORMANCE (TASK 1).
<table><tr><td>Model</td><td>Top-1 Accuracy</td></tr><tr><td>MLP</td><td>0.8978</td></tr><tr><td>Random Forest</td><td>0.8480</td></tr><tr><td>Autoencoder (AE)</td><td>0.8068</td></tr><tr><td>SVM</td><td>0.3050</td></tr></table>

TABLE IV  
NEXT-DOMAIN PREDICTION PERFORMANCE (TASK 2).
<table><tr><td>Model</td><td>Top-1 Accuracy</td></tr><tr><td>Graph-Based Markov Model (GBM)</td><td>0.7144</td></tr><tr><td rowspan="2">Graph-Sequence Model (GSM) GSM + LLM Reranking</td><td>0.7938</td></tr><tr><td>0.7980</td></tr><tr><td>SD-HBM</td><td>0.8015</td></tr></table>

## VI. DISCUSSION

Our results suggest that user identifiability and navigation predictability are driven by a shared underlying interaction structure rather than fundamentally distinct behavioral processes. Across both tasks, repeated user–domain interactions emerge as the dominant source of signal, indicating that behavioral traces encode stable identity-related and short-term predictive information within the same representational space.

From a modeling perspective, the strong performance of graph-based and sequence-based approaches indicates that most of the exploitable structure in browsing behavior is already captured through interaction history and temporal ordering. Sequential modeling provides additional gains by refining local dynamics, but does not fundamentally alter the predictive capacity provided by structural representations.

The limited impact of LLM-derived semantic feature augmentation further suggests that high-level semantic interpretations of behavior add only marginal information beyond what is already encoded in interaction patterns. In particular, semantic features appear to refine rather than transform the underlying representation, indicating that structural interaction patterns contribute substantially more predictive information than the evaluated LLM-derived semantic features.

From a privacy perspective, these results indicate that domain-level browsing traces carry a persistent identity signal: a single short session is often sufficient to recover the generating user among more than two thousand candidates, without access to URLs, page content, or explicit identifiers. The panel data used here is public and anonymized, yet anonymization at the identifier level evidently does not remove behavioral linkability. Plausible mitigations therefore operate on the representation rather than the identifier: aggregating visits to the category level, coarsening timestamps, or suppressing rare domains would each reduce the distinctiveness that our features exploit. We do not evaluate such defenses here, and measuring identification accuracy under adversarial or privacy-preserving transformations remains an important direction for future work.

## VII. LIMITATIONS

Despite these findings, several limitations should be acknowledged. First, the analysis is based on a single dataset collected over a limited time period, which restricts the ability to capture long-term behavioral evolution, seasonal effects, or cross-period generalization.

Second, the study operates at the domain level, without incorporating page-level content, search queries, or intra-site navigation structure. While this abstraction improves scalability and reduces noise, it necessarily omits finer-grained behavioral signals that may further improve both identity and intent modeling.

Third, the two tasks use different evaluation protocols. Next-domain prediction is evaluated under a chronological split, whereas user identification uses a stratified random partition over users. The identification result is therefore not directly comparable to a strictly temporal evaluation and should be read as an upper bound.

Fourth, session construction at a 2-second threshold groups requests occurring in rapid succession. Such groups may reflect resources co-loaded with a single deliberate visit rather than a sequence of navigation decisions, so identification at this resolution partly reflects page-loading footprints in addition to user choices.

Fifth, supervised samples for next-domain prediction are drawn only from sessions longer than L events, so the reported predictability characterizes sustained browsing episodes rather than browsing activity as a whole. The autoencoder is likewise evaluated at a single latent size, so we do not characterize how identification accuracy degrades under stronger compression.

Sixth, all reported results correspond to single training runs. We do not report variance across random seeds or statistical tests, and the sub-one-point differences separating the strongest next-domain models are within the range plausibly attributable to run-to-run variation; their ordering should therefore be treated as indicative rather than conclusive. Relatedly, we do not include trivial reference baselines such as repeating the last visited domain or predicting each user’s most frequently visited domain, which would help establish the margin achieved over simple heuristics.

Seventh, the identification features include demographic attributes, which are constant per user. The reported accuracy therefore reflects a combination of behavioral and demographic signal, and isolating the purely behavioral component is left to future work.

Finally, modeling choices such as sessionization thresholds, Top-K vocabulary restriction, and session filtering may still influence the observed performance.

## VIII. CONCLUSION

In this work, we investigated user identifiability and navigation predictability as two complementary inference tasks derived from web browsing traces. We used a common experimental framework, with task-specific preprocessing, to study how different modeling families exploit interaction patterns in user–domain behavior. Our findings show that both identity inference and next-action prediction are largely governed by repeated interaction structures between users and domains. Structural and sequential models capture most of the informative signal, while semantic augmentation using LLMs provides only limited additional benefit. Overall, the results indicate that browsing behavior is primarily shaped by interaction history and repetition patterns, with semantic interpretations playing a secondary role. This supports the view that both identity-related and predictive signals in web browsing appear to be strongly influenced by similar interaction regularities.

## REFERENCES

[1] J. Kulshrestha, M. Oliveira, O. Karac¸alik, D. Bonnay, and C. Wagner, “Web routineness and limits of predictability: Investigating demographic and behavioral differences using web tracking data,” in Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), 2021, pp. 327–338.

[2] A. Schnauber-Stockmann et al., “Routines and the predictability of day-to-day web use,” Media Psychology, 2023.

[3] D. Paulino et al., “Leveraging webtracesense for user interaction log analysis and visualization,” ACM Transactions on Interactive Intelligent Systems, 2024.

[4] M. Oliveira et al., “Browsing behavior exposes identities on the web,” Scientific Reports, 2025.

[5] C. Song et al., “Redefining website fingerprinting attacks with multiagent llms,” arXiv preprint, 2025.

[6] S. Wang, L. Cao, Y. Wang, Q. Z. Sheng, M. A. Orgun, and D. Lian, “A survey on session-based recommender systems,” ACM Computing Surveys, vol. 54, no. 7, pp. 1–38, 2021.

[7] T. F. Boka, R. B. Neupane, and Z. Niu, “A survey of sequential recommendation systems: Techniques, evaluation, and future directions,” Information Systems, vol. 125, p. 102427, 2024.

[8] L.-W. Pan, W.-K. Pan, M.-Y. Wei, H.-Z. Yin, and Z. Ming, “A survey on sequential recommendation,” Frontiers of Computer Science, vol. 20, no. 3, p. 2003606, 2026.

[9] S. Latifi, N. Mauro, and D. Jannach, “Session-aware recommendation: A surprising quest for the state-of-the-art,” Information Sciences, vol. 573, pp. 291–315, 2021.

[10] S. Latifi, D. Jannach, and A. Ferraro, “Sequential recommendation: A study on transformers, nearest neighbors and sampled metrics,” Information Sciences, vol. 609, pp. 660–678, 2022.

[11] Z. Cui, Y. Cai, S. Wu, X. Ma, and L. Wang, “Motif-aware sequential recommendation,” in Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 2021, pp. 1738–1742.

[12] W. Black, A. Manlove, J. Pennington, A. Marchini, E. Ilhan, and V. Markeviciute, “Trace: Transformer-based user representations from attributed clickstream event sequences,” CoRR, vol. abs/2409.12972, 2024. [Online]. Available: https://arxiv.org/abs/2409.12972

[13] Y. Beugin and P. McDaniel, “A public and reproducible assessment of the topics API on real data,” CoRR, vol. abs/2403.19577, 2024. [Online]. Available: https://arxiv.org/abs/2403.19577

[14] A. Verna, N. Jha, M. Trevisan, and M. Mellia, “A first view of topics API usage in the wild,” in Proceedings of the ACM International Conference on Emerging Networking Experiments and Technologies (CoNEXT), 2024.