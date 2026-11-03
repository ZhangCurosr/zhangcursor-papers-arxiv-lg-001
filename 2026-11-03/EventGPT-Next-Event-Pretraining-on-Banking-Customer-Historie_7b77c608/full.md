# EventGPT: Next-Event Pretraining on Banking Customer Histories

Denis Burakov

November 2026

## Abstract

A model pretrained to predict a customer’s next event can estimate the probability of future events without training on their labels and provide a reusable representation for supervised downstream banking tasks. We pretrain EventGPT, a hierarchical causal transformer, by next-token maximum likelihood estimation on the public Multimodal Banking Dataset (MBD-mini: 100,224 clients, 38.8 million transactions and 66.3 million geolocation pings). Keeping one state per event, EventGPT trains in about two hours on one L4 GPU at windows up to 8,192 tokens. From its next-event distribution, with no labels, EventGPT scores five targets of the form “the client makes a transaction of a given kind for the first time next month” at ROC-AUC 0.79 to 0.92. Gradient boosting on 114 engineered features needs 10,000 to 300,000 labeled client-months to match it on four targets and stays below it on the fifth. For product prediction, frozen EventGPT embeddings score below the engineered features (mean ROC-AUC 0.787 against 0.808 over five test folds), and a rank-averaged ensemble of EventGPT, profile fields and token counts with the engineered features adds 0.014 to 0.017, of which 0.007 to 0.016 is EventGPT’s own. The code and pretrained weights are released.

## 1 Introduction

A model pretrained to predict a customer’s next event can be useful in two ways. It can answer questions about future events directly from its own predictions and give one representation per customer that supervised downstream banking tasks can reuse. EventGPT<sup>1</sup> is trained by predicting each next event token of a customer’s event stream. This paper describes the architecture, training recipe, and measures both uses on a public banking benchmark.

The usual starting point of modeling bank outcomes is a feature table including counts, sums and recencies computed per customer and window that is fed into a machine learning algorithm. Building the table requires business knowledge (which field is the amount, which event types to include, which windows), and that work is repeated for every new event source. Each new question also needs labels and a model of its own. When events of many kinds with diferent fields are combined, as in a lender’s applications, repayments and account charges, there is often no obvious table to build. A pretrained event model needs neither the table nor (for questions about events) the labels.

Several financial institutions describe foundation models of this kind. Nubank’s nuFormer is a flat causal decoder over customer transaction tokens [4], Revolut’s PRAGMA is a masked model with separate encoders for profile state, single events and history [15], and Visa’s TransactionGPT encodes the aspects of each transaction before modeling across transactions [6]. Starling Bank has announced a transactional foundation model trained on 15 billion of its transactions, to be deployed in 2027 [20]. None of these releases checkpoints or evaluates on publicly available data. Even if a checkpoint were released, however, it would be of little use to another institution: its vocabulary is the company’s own event nomenclature, so each institution pretrains its own backbone (Section 8). EventGPT combines elements of existing approaches: a per-event encoder, as in PRAGMA and TransactionGPT, with the causal pretraining objective of nuFormer. It is designed for pretraining on consumer-lender product histories. Here, we train and evaluate it exclusively on publicly available data to make the results reproducible.

## Contributions

1. An open reference model: to our knowledge, the first open-weights generative model for banking event sequences, trained and evaluated on a public benchmark with reproducible baselines. The weights read MBD’s schema, so another institution reruns the code on its own events (Section 8). The tokenization of MBD-mini, the model, the training and evaluation code and the checkpoints are released (Section 10).

2. A hierarchical architecture for event streams: an event encoder pools the tokens of each event into one state, a history decoder attends causally over the event states, and a token decoder generates the next event. The last hidden state serves as the customer embedding and incorporates information about the time since the client’s last activity. Over a window of � tokens, the token layers attend within events at linear cost $O ( n )$ ， while the history layers attend over approximately $n / 8$ events at cost $O ( ( n / 8 ) ^ { 2 } )$ against $O ( n ^ { 2 } )$ for attention over tokens (Sections 3 and 8).

3. Zero-shot scoring of event targets from the token decoder’s distribution over the next event, which scores five first-occurrence targets at ROC-AUC 0.79 to 0.92 with no labels. CatBoost on the engineered features reaches the same ROC-AUC after 10,000 to 300,000 randomly sampled client-months on four targets and stays below it on the fifth at 300,000. A Markov chain over event kinds scaled by the client’s recent activity, which also needs no labels, scores 0.79 to 0.88. EventGPT scores above it on two of the five targets and on 48 of the 74 kinds with at least 50 test positives, and below it on 7 (Section 7.1).

4. An evaluation of frozen and fine-tuned embeddings on MBD against engineered features and a schema-free token-count table, with paired standard errors clustered by client. The embedding is below both tables and exceeds token counts only inside its own window. We add the profile fields and full-history token counts to the embedding. The gap then falls to 0.004, averaged over five test folds, and is not statistically significant on any fold. Late fusion of the profile fields adds 0.013 to 0.019 to the ROC-AUC of the product targets and brings target\_4 level with the engineered features (Sections 6–7).

5. The embedding as a second ranker beside a feature table: averaging its ranks with those of the engineered features adds 0.014 to 0.017 to their ROC-AUC on each of five test folds (intervals excluding zero, weights fixed in advance). The same ensemble built without EventGPT, from the profile fields and token counts alone, adds 0.001 to 0.008, so 0.007 to 0.016 of the gain is EventGPT’s own $( p < 0 . 0 1$ on each of ten fold and seed pairs). As features inside the table the embedding leaves their ROC-AUC unchanged or lower (Section 7.4).

6. Embeddings as point-in-time features: an embedding for a date uses only events before that date, so it can be stored and reused across tasks without leaking the future (Section 4).

## Terms

The as-of date is the first day of a month; a customer’s history at it is every event before it, and every score and embedding is computed at one. Product targets, also called the oficial targets, are MBD’s four labels target\_1 to target\_4 (the customer takes product � during the month), and their mean ROC-AUC is the oficial mean. Event targets are five kinds of transaction (type and subtype) made for the first time during the month. Engineered features are 114 features built by hand from the schema (counts, amounts, recencies and the profile fields), and token counts count the 200 most frequent tokens over time windows with no knowledge of the schema; CatBoost reads both. The hidden state is the history decoder’s output after an event, and the last one, at the as-of date, is the customer embedding (Section 3.4). A zero-shot score is the model’s probability that the next event is of a given kind, read with no labels and no classification head.

## 2 Related work

![](images/045ae4e6cf7f29a8b4ff9023918118e7cf2300a25ef8ac7998fb09c8f2ef3358.jpg)  
Figure 1: Event-sequence models for banking by year of publication. Color marks the pretraining objective; TabBERT and TabGPT are a masked and a causal model from one paper, and MBD is the dataset used here.

## Event-sequence models in banking

Most of the models cited here appeared in 2025 and 2026, with contrastive, masked and generative objectives (Figure 1). CoLES learns client embeddings by contrastive learning on subsequences of a client’s transactions [2], and the pytorch-lifestream library implements it with other pretraining methods for event sequences [19]. Its GPT-style objective predicts each field of the next transaction with a separate head; EventGPT instead decodes the next event token by token, which gives a joint distribution over its fields. Event Stream GPT, a library for generative transformers over continuous-time medical event streams, also models the dependencies between the fields of an event [11]. LATTE aligns event embeddings with the embeddings a frozen language model gives to short text summaries of a client’s behavior [7]. TabBERT and TabGPT treat each transaction as a row of fields, encode fields into a row embedding and model the rows with a masked or a causal transformer [16]; their code and a synthetic card-transaction dataset are public, without pretrained checkpoints. The MBD paper evaluates these methods, aggregate features and a supervised recurrent network on the same product-propensity task used here [12].

## Industrial foundation models

Beyond the designs described in Section 1, nuFormer fuses its embedding with a tabular network at fine-tuning time [4], and PRAGMA serves credit scoring, fraud and lifetime value from one backbone [15]. NVIDIA describes the same recipe, tokenizer to decoder to embedding to classifier, for card transactions [13]. A masked model of transactions with their text descriptions has been trained on data from thousands of North American institutions [17]. Their checkpoints are not publicly available.

## Positions from time

Rotary embeddings rotate queries and keys by an angle proportional to a position [21]. The rotation does not require the position to be an integer, so EventGPT can feed elapsed time in its place. Qwen2.5-Omni sets rotary positions from absolute time, one step per 40 milliseconds, to align audio and video (TMRoPE) [22]. The Transformer Hawkes Process encodes the time stamps of events with sinusoids in place of positions [23].

## Factorized attention

TimeSformer divides the attention of a video transformer into attention within each frame and attention across frames [3]. EventGPT factorizes a customer’s history the same way, with attention within each event and then across events.

## 3 Model

EventGPT has three stages (Figure 2):

1. Event encoder: turns each event into a single vector.

2. History decoder: processes these event vectors for each client.

3. Token decoder: predicts the tokens of the next event.

Next-token cross-entropy trains all three stages jointly. Each stage consists of a stack of prenorm transformer blocks; they difer in the inputs over which their attention layers operate (Figure 3).

![](images/2df1d0515c70bbe3ceadb2285618deafc68fda2875e44f5d05f0d95530886cb4.jpg)  
Figure 2: The three stages of EventGPT on three MBD events. Token layers read each event on its own and mean-pool it into one vector; causal event layers with temporal RoPE model the event vectors; token layers decode the next event given the hidden state of the previous one.

## 3.1 Event encoder

An event is a short run of tokens that starts with [EVT] (event token), followed by its name, its time gap and its fields as key=value tokens (Section 5). The fields of an event form a record, whose meaning does not depend on the order in which its fields appear. The event encoder is therefore bidirectional within an event and cannot attend across an event boundary. Two layers produce token states that are mean-pooled into one vector per event.

![](images/e01adbf39fdcfa3bb8625d2f0bb322322983fbc9ec126fb827c0b307ec14fb48.jpg)  
Every block is pre-norm with residual adds; attention has 12 heads of width 32 with query-key RMSNorm, and the MLP widens 384 to 1,536

Figure 3: The layers inside each stage. The event encoder attends within one event, without positions, and mean-pools the event’s tokens into $e _ { i }$ . The history decoder attends causally over the event vectors, with temporal RoPE on queries and keys, and gives the hidden state $h _ { i } .$ . The token decoder adds $h _ { i }$ to every token embedding of event � + 1 and attends causally within that event; the output layer shares its weights with the token embedding.

![](images/66cb3d5d0ee2dde919a73d264f2775ea1f1244e96a811fa7bc91003547364237.jpg)  
Figure 4: Attention at the two levels. Within an event every token attends to every other; across events attention is causal and confined to one customer. The tokens of an event reach earlier events only through the hidden state of the previous event.

## 3.2 History decoder and temporal RoPE

The history decoder needs each event’s vector position, which EventGPT derives from elapsed time. Six causal layers run over each client’s event vectors. The hidden state after event � reads the vectors of events $1 , \ldots , i ,$ one per event, instead of their tokens. Positions come from elapsed time. TMRoPE is linear in time [22], and our temporal RoPE takes its natural logarithm. With $\Delta t _ { i }$ the seconds from the client’s first event in the window to event �, the rotary position is

$$
p _ { i } = 6 4 \ln \biggl ( 1 + \frac { \Delta t _ { i } } { 6 0 } \biggr ) .\tag{1}
$$

Rotary attention depends on the diference between two positions, and here that diference is 64 ln of a ratio of elapsed times. With the 60 seconds added, a gap from a minute to an hour (a ratio of 30.5) is then about as long as a gap from a day to a month (30), and two events a year apart stay far apart even when no event lies between them (Figure 5). Elapsed times are derived from the dt tokens: each time-gap bucket is represented by its midpoint in seconds. The same gap spans less rotation later in the window: one day corresponds to 44 units between the first and second day, but only 0.09 units between days 699 and 700.

![](images/552cfa7c8bf686c2c1bc39fc5c9d670e72ec2f0539d603da25a489e0367a225a.jpg)

![](images/61e7b5e2d34a64d6281c892b2d916b4186dd8ec8000760b32c627c4e2c85b9ba.jpg)  
Figure 5: Ordinal positions against temporal RoPE, Eq. (1). With ordinal positions every event is one step from the next; with temporal RoPE the distance between two events is set by the ratio of their elapsed times.

## 3.3 Token decoder and the training signal

Each token of event � receives its embedding, a learned position within the event, and $h _ { i - 1 }$ ， the history decoder’s hidden state after the previous event. For the first event in a window, this state is set to zero. Two causal layers process the tokens and predict the next token through an output layer tied to the input embedding (Figure 6). The token decoder uses $h _ { i - 1 }$ rather than the pooled representation of event �. The event encoder has no separate loss: it receives gradients through the history decoder from the next event’s token-prediction loss, so its representation is shaped by how useful the event’s fields are for predicting subsequent events.

![](images/14f88621baaf5aa08d416547e2f49906744814fe9d8778dc18715be903df4701.jpg)  
Figure 6: What the token decoder reads. Each token of event 3 attends to the earlier tokens of its own event and to $h _ { 2 } .$ , the history decoder’s hidden state after event 2. It has no path to the tokens inside earlier events, which reach it only through the event vectors and $h _ { 2 }$

## 3.4 Customer embedding

The embedding of a client at an as-of date is read from the history decoder’s hidden states for events preceding that date. The last hidden state (last) is the embedding used throughout the paper. We compare it with the mean of the hidden states (mean) and a recency-weighted mean, which assigns weight $\exp ( - a / 3 2 )$ to the event � events before the last (recency), in Section I. All three representations are obtained in one forward pass.

Each history ends with a now event, [EVT] dt=<gap>, whose gap runs from the last event to the as-of date. The final hidden state therefore corresponds to the as-of date: two clients with identical event histories but diferent periods of inactivity receive diferent embeddings.

When a history exceeds the window length, the window retains the most recent events and preserves [USR] and the profile event at its beginning. This prevents the static attributes from being discarded for clients with the longest histories.

## 3.5 Static attributes

The profile event passes through the same event encoder as all other events, but its four attributes also have a direct path to the classification head. We implement this in two ways. In late fusion we append the input embeddings of the four field tokens to the hidden state and fit the classification head on both without retraining the backbone. In early fusion we add the attributes during pretraining. We concatenate the four embeddings, project them to the model width with a linear layer initialized at zero and add the result to the vector of every event from the profile event on. We add no profile vector to the [USR] event, because its hidden state predicts the profile event’s tokens. We take the field tokens from the client’s stream and use no knowledge of what the attributes mean.

## Training windows

A window drawn at a random ofset typically starts partway through a client’s history and therefore omits [USR] and the profile event that begin every evaluation history. We also train on aligned windows, which retain these two events at the beginning (Appendix D).

![](images/35eea86886721819d1bfab451cf447da8ced8105f983fd481c8958e2966f72bb.jpg)  
Figure 7: A training window drawn at a random ofset and the aligned window over the same events, on a client from fold 1, shortened to 110 tokens. The random window starts inside an event and holds no [USR] or profile event; the aligned window puts both back in front, and the gap token of its first event counts from the client’s first event.

## 4 Embeddings as point-in-time features

An embedding computed once per client and as-of date can serve multiple downstream tasks through a small classification head, with the backbone frozen. In this paper, the same set of embeddings serves ten labels (four product targets, five event targets and churn described in Section 6), each using logistic regression. A new task then requires only fitting a head on the stored vectors. Revolut and Nubank describe the same arrangement for their backbones [15, 4].

This reuse is safe only if the embedding at an as-of date depends exclusively on events before that date. An embedding that incorporates future events inflates the ofline results of every task built on it, because all task-specific heads share it. EventGPT enforces this point-in-time property through its architecture and data construction (Appendix E).

This guarantee does not, however, extend to the pretraining population. Pretraining uses events from 2021 and 2022 for the pretraining clients, including events from the test clients’ label months. An out-of-time run measures the efect of this overlap in Section 9.

## 5 Data

## 5.1 MBD-mini

MBD is a public dataset of bank clients for research on event sequences [12]. It holds each client’s transactions, geolocation pings and support dialogues, and monthly labels for the purchase of four products. MBD-mini, a 10% client sample published under CC BY 4.0 [10], holds 100,224 clients in five folds of equal size (Table 1). All fields are anonymized integer codes and amounts are rescaled. Timestamps are in UTC, and the bank’s day starts at 21:00 UTC (midnight in Moscow).

Table 1: MBD-mini modalities. Events per client are the 50th, 90th and 99th percentiles.
<table><tr><td>modality</td><td>events</td><td>clients</td><td>span</td><td>per client (p50 / p90 / p99)</td></tr><tr><td>transactions</td><td>38.8M</td><td>98,721</td><td>Jan 2021 – Dec 2022</td><td>170 / 1,007 / 2,243</td></tr><tr><td>geolocation pings</td><td>66.3M</td><td>72,573</td><td>Dec 2021 – Dec 2022</td><td>507 / 2,293 / 5,580</td></tr><tr><td>dialogues</td><td>0.22M</td><td>46,320</td><td>Feb 2021 – Dec 2022</td><td>3 / 10 / 27</td></tr></table>

## 5.2 Events and tokens

Each client’s stream starts with [USR] and a profile event, then interleaves transactions and geolocation in time order:

profile [EVT] name=profile dt=140 s21 s22 s31 s32  
transaction [EVT] name=trx\_<type> dt=<b> sub=<subtype> amt\_bin=<b> [cur] s11 s12 d11 d12  
geolocation [EVT] name=geo dt=<b> g4=<cell> g5=<cell>

The profile event states the client’s static fields once, and one geolocation event summarizes each day (Appendix F).

## Time and amounts

The gap since the client’s previous event becomes a token $\mathtt { d t } = \lfloor 8 \ln ( 1 + \Delta s / 8 ) \rfloor$ , capped at 140, so 141 buckets span seconds to decades. Amounts become one of 21 bins with quantile edges fitted on the pretraining folds. A currency token cur appears only when the currency is not the one shared by 99.9% of transactions.

The stream holds 53.4 million events in all, about 8 tokens per event. Folds 1–4 give the pretraining stream, with 2% of their clients held out for validation loss: 332 million training tokens and 6.8 million validation tokens. Fold 0 is never seen in pretraining or vocabulary fitting.

## 6 Evaluation protocol

## 6.1 As-of dates and labels

A row is a client and a month from February 2022 to January 2023, 12 rows per client. As in MBD, the history for a month is every event strictly before the first day of that month [12], here in Moscow time; that day is the month’s as-of date.

## Oficial targets

target\_1 to target\_4 indicate that the client takes product � during the month. They are rare. On the test fold, target\_1 has 1,072 positives in 243,936 rows (0.44%), target\_2 132 (0.05%), target\_3 934 (0.38%) and target\_4 634 (0.26%). Of these rows, 1,332 belong to 111 clients with no transactions or geolocation, and so no stream. The classification heads and the baselines both drop them, which leaves 242,604 scored rows. No single event type marks a purchase: the type most often seen for the first time in the month of a purchase covers at most 10% of its positives.

## Event targets

Purchases leave traces in the stream. We rank every transaction kind (type/subtype) on the four training folds by the ratio of its first-occurrence rate in positive client-months to its first-occurrence rate in negative client-months. For target\_2 the kinds 48/31, 23/49, 26/28 and 48/10 rank first, third, fifth and sixth (rate ratios 215, 94, 66 and 54), and for target\_3 50/19 ranks first (25). The kinds ranked second and fourth for target\_2, 12/20 and 48/49 (106 and 92), are not used. We chose the five kinds before this ranking and did not keep the script, so we cannot establish which data the choice used. The ranking shows that the selected kinds are strongly enriched in the training folds. It does not reproduce the choice, since two unselected kinds rank above two selected ones. Each of these kinds defines an event target: the kind appears in the client’s history for the first time during the month. A churn target marks a month with no transaction from a client who had at least one before the as-of date. Both are read from the stream.

## The last month

The transaction tables end on 31 December 2022. In January 2023 no client has a transaction, so 99.1% of that month’s rows are labeled churn, against 12.7% to 20.4% in the other months, and the month holds 20,032 of the test fold’s 55,920 churn positives. A model that can tell the month of the as-of date scores those rows without reading any behavior, and the time since a client’s first event gives the month away. With January 2023 included, EventGPT on aligned windows (Section D) reached 0.910 on churn against 0.830 on random windows; without it, both reach 0.93. We therefore leave January 2023 out of the churn label, in training and in testing, which keeps 35,888 test positives. The event targets have no positives in that month and change by at most 0.014 without it. The oficial targets are the dataset’s own labels and keep all twelve months.

The product targets are rare and drift (Figure 12): target\_1 falls from 0.56% of rows in February 2022 to 0.16% in January 2023, while target\_2 stays near 0.05%. The event targets lie between 0.01% and 0.4%, and churn rises from 12.7% to 20.1% before the last month.

## 6.2 Heads and metrics

For each label, we fit a logistic regression (� = 0.1) on the frozen, standardized embeddings from folds 1–4 and score all rows in fold 0. Positives are rare, so the training sample includes every row positive for at least one label, plus 60,000 rows sampled uniformly from the remaining rows. We reweight the sampled rows to account for their share of the full training set. We report ROC-AUC and PR-AUC (average precision). PR-AUC matters because a ranking can have a high ROC-AUC while still placing mostly negatives near the top. A random ranking has PR-AUC equal to the label prevalence.

We estimate standard errors using DeLong’s method [5], which expresses ROC-AUC as a mean of placement values, and cluster the variance by client [14]. The twelve monthly rows for each test client are not independent. We therefore sum each row’s influence on the AUC within its client and take the variance across clients:

$$
\psi _ { i j } = \frac { V _ { i j } - \widehat { \mathrm { A U C } } } { n _ { y _ { i j } } } , \qquad S _ { i } = \sum _ { j } \psi _ { i j } , \qquad \widehat { \mathrm { V a r } } ( \widehat { \mathrm { A U C } } ) = \frac { N } { N - 1 } \sum _ { i = 1 } ^ { N } S _ { i } ^ { 2 } ,\tag{2}
$$

where $V _ { i j }$ is the placement value of row � of client � (for a positive, the share of negatives it outranks; for a negative, the share of positives that outrank it), $n _ { y _ { i j } }$ is the number of rows in its class, and $N = 2 0 { , } 2 1 7$ . On simulated data with twelve correlated rows per client, this gives a standard error close to that from a bootstrap over clients (0.0129 versus 0.0132); the unclustered variance gives 0.0102. The same client-level sums give the covariance between two AUCs evaluated on the same rows, allowing us to calculate standard errors for both the oficial mean and the diference between two models. Since both models rank the same clients, the comparison is paired. Its standard error (about 0.005 on the oficial mean) is therefore smaller than that of either model’s AUC (about 0.006).

## 6.3 Zero-shot event targets

We score event targets directly with the pretrained model, without a classification head or labeled examples. A target is positive if transaction kind (�, �) appears in the client’s history for the first time during the month. After the history at an as-of date we append the start of a next event, [EVT] name=trx\_X dt=<g>, where � is the bucketed gap from the last event to the as-of date. The token decoder gives �(name=trx\_X) at [EVT] and $P ( \mathbf { s u b } \mathbf { \bar { \Gamma } } | \mathbf { \bar { \Gamma } } \mathbf { n a m e } \mathbf { \bar { \Gamma } } \mathbf { r } \mathbf { x _ { - } } \mathbf { \bar { \Gamma } } \mathbf { X } , \mathbf { d t } \mathbf { \bar { \Gamma } } \mathbf { \bar { \mathbf { g } } } )$ at the dt token, and the score is their product, the probability that the next event is of kind (�, � ). It ranks clients but is not the probability of the label, which counts any occurrence in the month; sampled months give that probability (Section 7.1). A client who already had the kind before the as-of date cannot be positive under the label’s definition, so that client’s score is set to zero. This rule uses no labels, and we apply it to the supervised scores as well when comparing the two. For EventGPT the history stages run once per row and the token decoder once per type $X ;$ the flat decoder is scored the same way and runs whole once per type.

We read a history in three ways (Figure 8): the hidden state at the as-of date for a supervised head, this one-step score, and months sampled from the model (Section 7.1).

![](images/ba13030d6fd1985d5967588863624b861957f24bf354d46eaced8124a7e226c3.jpg)  
Figure 8: Inference from one history. The hidden state ℎ at the as-of date is the embedding read by a supervised head. From ℎ the model gives the distribution of the next event, which the one-step score reads for the target type; the widths are the probabilities for the client of Figure 9, whose next event was a 50/19. Sampling events one after another continues each branch to a horizon, here the end of the label month; 16 sampled futures give the monthly probability of the target.

Zero-shot scoring is evaluated on every test row positive for at least one of the five event targets, together with 20,000 rows sampled uniformly from the test fold. For each target, ROC-AUC is calculated using its own positives and the uniform sample; positives for other targets are not treated as negatives. Uniform sampling of negatives does not bias ROC-AUC. The supervised comparison uses the logistic head on the last hidden state, scored on the same rows.

## 6.4 Baselines

## Engineered features

On the same rows we compute 114 features from events before the as-of date. They cover transaction counts and the sum, mean and maximum amount, the count and amount for each of the 15 most frequent event types, and geolocation pings and active days. Each is computed over all history, the last month and the last three months, and the days since the first and the last transaction and the four profile fields (as categorical features) complete the table. CatBoost [18] (depth 6, learning rate 0.05, at most 2,000 trees with early stopping on 10% of training clients) is fitted for each label on the same training rows.

Refitted without one group of features at a time, the oficial mean falls by 0.010 without the profile fields (0.036 on target\_4), by 0.009 without the two timing features, by 0.005 without the full-history window and by 0.001 without geolocation, and it does not change without the amounts. Building these features takes knowledge of the schema.

## Token counts

This baseline does not require knowledge of the event schema. For each of the 200 most frequent tokens of the stream (any key=value, amount bin or time bucket, together 90.9% of all non-special tokens), it counts occurrences over all history, the last 91 days and the last 30 days. It also includes the number of events in each window and the number of days since the first and last events, giving 605 features in total. We compute the features either over the full history or over the same 2,048-token window used by EventGPT, and fit CatBoost with the same settings. The baseline uses only token counts and timestamps; it does not require manually designed features for particular event types.

## Flat decoder

A GPT-style decoder over the same token stream, with width 384, ten layers and learned positions: 23.5 million parameters against EventGPT’s 22.9 million, trained the same way. Its embeddings are the hidden state of the last token, the mean of the token hidden states and a recency-weighted mean over about 256 tokens, which 32 events hold on average.

## Published results

The MBD paper reports mean ROC-AUC on full MBD, a population ten times larger with its own five-fold split (Table 2). These numbers give context only: MBD-mini is a diferent sample, and the comparison on full MBD is planned (Section 9). We therefore train CoLES [2] on our pretraining folds with the MBD paper’s transaction configuration (a GRU of width 256 over per-field embeddings, contrastive loss on random slices of a client’s transactions, 15 epochs) and read it with the same logistic head as EventGPT. With two seeds it scores 0.771 and 0.772 on the oficial mean, close to its published 0.773, against 0.786 for EventGPT on aligned windows (seed 42; paired diferences 0.015 and 0.014, standard errors 0.005 and 0.006) and 0.816 for the engineered features. CoLES is below EventGPT on target\_1 to target\_3, on every event target and on churn, and above it on target\_4 (0.784 and 0.795 against 0.754).

Table 2: Mean ROC-AUC over the four product targets on full MBD, as reported by [12] (late fusion for the two-modality column).
<table><tr><td>method</td><td>transactions</td><td>transactions + geo</td></tr><tr><td>aggregation</td><td>0.783</td><td></td></tr><tr><td>CoLES</td><td>0.773</td><td>0.775</td></tr><tr><td>TabBERT</td><td>0.762</td><td>0.764</td></tr><tr><td>TabGPT</td><td>0.802</td><td>0.800</td></tr><tr><td>supervised RNN</td><td>0.819</td><td>0.819</td></tr></table>

## 6.5 Pretraining

EventGPT has width 384, 12 heads of width 32, two event-encoder layers, six history layers and two token-decoder layers: 22.9 million parameters, of which 4.9 million are the tied token embedding. It is trained with AdamW (learning rate $6 \times 1 0 ^ { - 4 } , \beta = ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.1), with linear warmup for 200 steps and cosine decay to 10% of the peak. Each step processes 16 windows of 2,048 tokens (32,768 tokens) drawn at uniform ofsets from the packed stream, for 20,000 steps, about two passes over the data. Windows start mid-event and mid-client, and [USR] tokens keep attention within each client.

On one NVIDIA L4 in bfloat16 a run of 655 million tokens takes 1.5 hours (about 118,000 tokens per second), against 3.9 hours (about 47,000) for the flat decoder. Model results are the mean of two seeds, 42 and 43, unless stated otherwise.

The final configuration, labeled aligned in the tables, trains on aligned windows (Section D) and normalizes queries and keys in every attention head by their root mean square (query-key normalization) [8]. Without the normalization, the seed-43 run on aligned windows diverged at step 4,400, at the peak learning rate: its validation loss rose from 1.16 to 1.5 and did not recover. With it, both seeds train smoothly to a validation loss of 1.020 and 1.019, at about 90,000 tokens per second (2.0 hours). The comparisons with the flat decoder use windows at random ofsets for both models. The run with early fusion (Section 3.5) adds it to this configuration, at a cost of 0.6 million parameters.

## 7 Results

## 7.1 Event targets without labels

Scored from the next-event distribution with no labels (Section 6.3), the five event targets reach ROC-AUC 0.79 to 0.92 on both seeds (Table 4). A supervised head on the same model scores 0.045 to 0.061 higher on 26/28 and 48/10 $( p < 0 . 0 0 1$ in every run). On the other three targets the two are level after a Bonferroni correction over 20 paired tests, except 23/49 with aligned windows and seed 43, where the head is higher (Appendix A). The score scaled by recent activity, a form we chose after these scores were known (Section I), is higher than the head on 48/31 and lower on 26/28 and 48/10 (Appendix A).

The flat decoder, scored the same way, reaches 0.78 to 0.92, the same range as EventGPT, so zero-shot scoring needs only a model trained to predict the next event.

We also evaluate two simple baselines based on event statistics, without a trained model or labels (Table 3). A first-order Markov chain over event kinds, fitted on the event streams of the four pretraining folds, estimates the probability that the next event is of the target kind. Using only the last event gives scores of 0.58 to 0.67; averaging the probabilities over the last ten events raises them to 0.72 to 0.84. The number of transactions in the 30 days before the as-of date scores 0.76 to 0.85 on its own. Multiplying the averaged Markov probability by the number of events in those 30 days gives an estimate of the expected count of the target kind at the client’s recent activity rate. This combined baseline scores 0.79 to 0.88. EventGPT scores higher on 23/49 and 50/19, by 0.057 and 0.095, respectively. On the other three targets, the diferences are within two standard errors. For 48/31, 26/28 and 48/10, recent transaction activity accounts for much of the predictive performance of the zero-shot score.

We chose the five targets by hand (Section 6), so we also score every transaction kind with at least 50 first occurrences in the test rows, 74 kinds, with the same rule. EventGPT zero-shot on aligned windows has a median ROC-AUC of 0.825 (quartiles 0.772 and 0.894, seed 43; median 0.828 on seed 42), so the five targets are typical of the kinds. Against Markov × events on the same rows, EventGPT (seed 43) is higher by more than two paired standard errors on 48 kinds, lower on 7 and level on 19. The median diference is 0.042, and 0.057 weighted by positives. The five targets show the model’s advantage less than the kinds as a whole.

Figure 9 shows what the model expects for one client. Its single-step score for 50/19 ranks in the top 1.1% of test rows, and months sampled from the model put a first 50/19 in the label month with probability 0.081, against a base rate of 0.003. The client’s actual next event was a 50/19, the model’s tenth most likely type at 1.7%.

Table 3: ROC-AUC on the five event targets from event statistics without labels, on the rows of Table 4 with the rule for earlier occurrences. Markov: the transition probability to the target kind from the last event, and its mean over the last ten events, fitted on the pretraining folds. Activity: transactions in the 30 days before the as-of date. Markov × events: the mean transition probability times the number of events in those 30 days. EventGPT is zero-shot on aligned windows, seed 42. The last column is EventGPT minus Markov × events, with the paired standard error clustered by client.
<table><tr><td></td><td colspan="3">Markov</td><td colspan="3">Markov</td><td colspan="2">EventGPT —</td></tr><tr><td>target</td><td></td><td>pos. last event last ten activity</td><td></td><td></td><td>× events</td><td>EventGPT</td><td></td><td>Markov × events</td></tr><tr><td>first 48/31</td><td>59</td><td>0.671</td><td>0.835</td><td>0.848</td><td>0.882</td><td>0.906 (0.022)</td><td></td><td>+0.024 (0.025)</td></tr><tr><td>first 23/49</td><td>195</td><td>0.667</td><td>0.833</td><td>0.811</td><td>0.860</td><td>0.916 (0.011)</td><td></td><td>+0.057 (0.016)</td></tr><tr><td>first 26/28</td><td>252</td><td>0.577</td><td>0.743</td><td>0.787</td><td>0.833</td><td>0.823 (0.013)</td><td></td><td>−0.010 (0.014)</td></tr><tr><td>first 48/10</td><td>290</td><td>0.608</td><td>0.718</td><td>0.759</td><td>0.791</td><td>0.777 (0.014)</td><td></td><td>-0.013 (0.013)</td></tr><tr><td>first 50/19</td><td>770</td><td>0.667</td><td>0.802</td><td>0.773</td><td>0.811</td><td>0.906 (0.006)</td><td></td><td>+0.095 (0.007)</td></tr></table>

Table 4: ROC-AUC on the five event targets. Zero-shot scores use no labels and set the score of clients who already had the kind to zero (Section 6.3); the flat decoder column is the mean of two seeds, as is the aligned column. The supervised head is the logistic head on the seed-42 embedding (random windows), on the same rows. In brackets the standard error, clustered by client; the seed-43 and aligned columns have standard errors within 0.005 of seed 42. The diference column pairs the seed-42 zero-shot score with the head. Its � comes from the standard error of the diference, which is smaller than either score’s own because both scores rank the same clients (Section 6.2). The engineered-feature column is the full test fold, so it is comparable in expectation but not row for row. The supervised head and the engineered features are scored without the rule for earlier occurrences.
<table><tr><td>target pos.</td><td>seed 42</td><td></td><td>EventGPT zero-shot</td><td>seed 43 aligned</td><td>flat decoder zero-shot</td><td>supervised head</td><td></td><td>zero-shot − head (p)</td><td>engineered features</td></tr><tr><td>first 48/31</td><td></td><td>59 0.916</td><td>(0.019)</td><td>0.902</td><td>0.913</td><td>0.909</td><td>0.891 (0.026)</td><td>+0.025 (0.19)</td><td>0.920</td></tr><tr><td>first 23/49</td><td></td><td>195 0.918</td><td>(0.011)</td><td>0.918</td><td>0.909</td><td>0.905</td><td>0.930 (0.009)</td><td>–0.011 (0.16)</td><td>0.920</td></tr><tr><td>first 26/28</td><td></td><td>252 0.829</td><td>(0.014)</td><td>0.827</td><td>0.826</td><td>0.799</td><td>0.875 (0.010)</td><td>−0.046 (&lt; 0.001)</td><td>0.868</td></tr><tr><td>first 48/10</td><td></td><td>2900.791</td><td>(0.014)</td><td>0.784</td><td>0.780</td><td>0.784</td><td>0.836 (0.012)</td><td>−0.045 (&lt; 0.001)</td><td>0.835</td></tr><tr><td>first 50/19</td><td></td><td></td><td>770 0.906 (0.006)</td><td>0.904</td><td>0.905</td><td>0.915</td><td>0.896 (0.006)</td><td>+0.010 (0.07)</td><td>0.937</td></tr></table>

![](images/35958eccaa54c9aee2734bb57841c937e2573a279e52224cd1b613f6c8000936.jpg)  
Figure 9: One test client at the as-of date 1 December 2022, drawn with a fixed seed among the 770 test rows positive for first 50/19. Top row: the client’s last ten events and its label month. Below: three of 64 months sampled from EventGPT (aligned windows, seed 42) from the same history. Right: the model’s distribution over the type of the next event, with the type that came next, and the probability of a first 50/19 in the month, from 64 sampled months and from one step, against the share of test rows that are positive.

Without the rule for earlier occurrences the score on 50/19 falls from 0.91 to 0.72. The model predicts the kind for clients who already make it every month. On the rare kind 48/31 the rule makes no diference. The score is the probability of the kind as the next event, while the label covers the whole month.

## A month of sampled events

We also let the model write the label month. For each test row we sample 16 continuations event by event, from the last event to the end of the month, and sum over the sampled steps the probability that the event at the step is of the target kind and falls inside the month, up to the first sampled event of that kind. The mean over continuations estimates the probability that the kind occurs during the month. Averaged over seeds 42 and 43, with every positive and 10,000 uniform test rows, it scores 0.010 to 0.036 above the one-step score on all five targets, by at least three paired standard errors on four of them (Table 5). The scaled score of Appendix I, the one-step score times one plus the client’s transactions in the 30 days before the as-of date, was chosen on the test fold. The rollout scores level with it on four targets and 0.012 higher on 50/19, with no such choice. It expects 15.5 transactions per row in the month against 15.5 observed, with a Spearman correlation of 0.81 across rows.

As forecasts of the month itself, the sampled months are weaker. The probability of no transaction in the month scores 0.846 on churn, close to the days since the last transaction alone (0.842) and below the number of transactions in the 30 days before the as-of date (0.884). Matched by kind, a sampled month and the real one share 0.459 of their transactions (counted over the larger of the two in each kind), against 0.545 for the client’s previous month repeated and 0.506 for transactions drawn at the client’s rate over the last 90 days.

Table 5: Event targets scored from 16 sampled months per row against the one-step zero-shot score and the scaled score, EventGPT on aligned windows, mean of seeds 42 and 43. ROC-AUC on each target’s positives and 10,000 uniform test rows, with the rule for earlier occurrences; diferences with paired standard errors clustered by client.
<table><tr><td>target</td><td>positives</td><td>one-step</td><td>scaled</td><td>month rollout</td><td>rollout — one-step</td><td></td></tr><tr><td>first 48/31</td><td>59</td><td>0.912</td><td>0.927</td><td>0.922</td><td>0.010</td><td>0 (0.007)</td></tr><tr><td>first 23/49</td><td>195</td><td>0.908</td><td>0.928</td><td>0.926</td><td>0.018 (0.005)</td><td></td></tr><tr><td>first 26/28</td><td>252</td><td>0.826</td><td>0.859</td><td>0.861</td><td>0.036</td><td>6 (0.005)</td></tr><tr><td>first 48/10</td><td>290</td><td>0.781</td><td>0.812</td><td>0.813</td><td>0.032</td><td>2 (0.004)</td></tr><tr><td>first 50/19</td><td>770</td><td>0.907</td><td>0.907</td><td>0.919</td><td>0.012 (0.002)</td><td></td></tr></table>

## Labels needed to match the zero-shot score

We measure how many labels the engineered features need to match the zero-shot score. We fit CatBoost on the engineered features of � client-months drawn at random from the 699,380 rows of the pretraining folds and score the whole test fold (Figure 10, Table 10). A draw without a positive scores 0.5. CatBoost’s scores follow the same rule for earlier occurrences.

The engineered features pass the zero-shot score after 10,000 to 30,000 labeled rows on 50/19, the most common kind, and after 30,000 to 300,000 on three others. On 23/49 they are still below it at 300,000 rows and pass it with every positive of the pretraining folds (0.930). This is between about 20 and 400 positives, and more than 260 on 23/49. Draws of 1,000 rows hold at most two positives and draws of 10,000 rows hold 2 to 34, so these row counts are rough thresholds.

Uniform draws spend most labels on negatives. We also draw � positives of the target and 10,000 negatives from the pretraining folds, as a team that labels its known cases and a sample of the rest would (Table 6). The engineered features then pass the zero-shot score with 50 positives on 48/31, 100 on 26/28, 48/10 and 50/19, and 400 on 23/49. Uniform draws pass it at 66 to 394 positives, so either way the zero-shot score is worth about 50 to 400 labeled positives.

![](images/ac08e153e4a8cc8ce35cbfd8e8a58a96d8f875f2ecc547b80e7a1b987ac98e73.jpg)  
Figure 10: ROC-AUC on the test fold of CatBoost on the engineered features, fitted on � clientmonths drawn at random from the pretraining folds (mean over draws), against EventGPT’s zeroshot score (mean of two seeds, no labels). Both apply the rule for earlier occurrences. Values are those of Table 10.

Table 6: Engineered features fitted on � positives of the target and 10,000 negatives drawn at random from the pretraining folds, against EventGPT zero-shot (as in Table 10). ROC-AUC on the test fold with the rule for earlier occurrences, mean over five draws; in brackets the standard deviation over draws. The pretraining folds hold fewer than 200 positives of 48/31.
<table><tr><td rowspan="2">(no labels)</td><td rowspan="2">zero-shot</td><td colspan="7">engineered features, labeled positives n</td></tr><tr><td>5</td><td>10</td><td>20</td><td>50</td><td>100</td><td>200</td><td>400</td></tr><tr><td>first 48/31</td><td>0.909</td><td>0.865 (0.025)</td><td>0.833 (0.101)</td><td>0.893 (0.018)</td><td>0.911 (0.009)</td><td>0.914 (0.006)</td><td></td><td></td></tr><tr><td>first 23/49</td><td>0.918</td><td>0.839 (0.023)</td><td>0.857 (0.015)</td><td>0.854 (0.018)</td><td>0.882 (0.018)</td><td>0.897 (0.024)</td><td>0.910 (0.008)</td><td>0.927 (0.002)</td></tr><tr><td>first 26/28</td><td>0.828</td><td>0.800 (0.040)</td><td>0.763 (0.057)</td><td>0.813 (0.039)</td><td>0.826 (0.023)</td><td>0.844 (0.017)</td><td>0.860 (0.011)</td><td>0.874 (0.003)</td></tr><tr><td>first 48/10</td><td>0.787</td><td>0.678 (0.089)</td><td>0.721 (0.030)</td><td>0.746 (0.047)</td><td>0.777 (0.018)</td><td>0.797 (0.026)</td><td>0.817 (0.005)</td><td>0.836 (0.009)</td></tr><tr><td>first 50/19</td><td>0.905</td><td>0.815 (0.031)</td><td>0.847 (0.022)</td><td>0.875 (0.021)</td><td>0.897 (0.007)</td><td>0.921 (0.006)</td><td>0.927 (0.002)</td><td>0.926 (0.010)</td></tr></table>

## 7.2 Frozen embeddings with a supervised head

We fit a classification head on the frozen embedding for each label (Table 7). Within a 2,048-token window EventGPT scores level with a flat decoder of the same size and with token counts. Both feature tables over the full history score higher.

Table 7: ROC-AUC on the test fold. Model columns are the mean of two seeds, with a logistic head on the frozen embedding; EventGPT is trained on windows at random ofsets or on aligned windows (Section D). Token counts and engineered features are fitted with CatBoost. Churn leaves out January 2023 (Section 6). The engineered features are fitted on the training rows drawn once January 2023 had left the churn label, the other columns on rows drawn before, which also held the 20,000 January churn positives. Every column scores the event targets without the rule for earlier occurrences. PR-AUC for the oficial targets is in Table 11.
<table><tr><td></td><td>engineered</td><td colspan="2">token counts</td><td colspan="2">flat decoder</td><td colspan="2">EventGPT, last</td></tr><tr><td>target</td><td>features</td><td>full history</td><td>2,048 tokens</td><td>last</td><td>recency</td><td></td><td>random aligned</td></tr><tr><td>target_1</td><td>0.787</td><td>0.781</td><td>0.757</td><td>0.692</td><td>0.738</td><td>0.740</td><td>0.756</td></tr><tr><td>target_2</td><td>0.843</td><td>0.849</td><td>0.792</td><td>0.725</td><td>0.823</td><td>0.829</td><td>0.850</td></tr><tr><td>target_3</td><td>0.816</td><td>0.809</td><td>0.788</td><td>0.729</td><td>0.783</td><td>0.776</td><td>0.789</td></tr><tr><td>target_4</td><td>0.818</td><td>0.774</td><td>0.770</td><td>0.709</td><td>0.765</td><td>0.753</td><td>0.752</td></tr><tr><td>mean</td><td>0.816</td><td>0.803</td><td>0.777</td><td>0.714</td><td>0.777</td><td>0.774</td><td>0.787</td></tr><tr><td>first 48/31</td><td>0.920</td><td>0.905</td><td>0.913</td><td>0.794</td><td>0.909</td><td>0.889</td><td>0.900</td></tr><tr><td>first 23/49</td><td>0.920</td><td>0.905</td><td>0.915</td><td>0.796</td><td>0.918</td><td>0.918</td><td>0.929</td></tr><tr><td>first 26/28</td><td>0.868</td><td>0.879</td><td>0.874</td><td>0.745</td><td>0.844</td><td>0.860</td><td>0.868</td></tr><tr><td>first 48/10</td><td>0.835</td><td>0.857</td><td>0.846</td><td>0.715</td><td>0.793</td><td>0.804</td><td>0.807</td></tr><tr><td>first 50/19</td><td>0.937</td><td>0.935</td><td>0.922</td><td>0.769</td><td>0.856</td><td>0.837</td><td>0.849</td></tr><tr><td>churn</td><td>0.956</td><td>0.955</td><td>0.955</td><td>0.875</td><td>0.899</td><td>0.935</td><td>0.932</td></tr></table>

## Hierarchy against the flat decoder

We train both models on windows at random ofsets. EventGPT scores 0.774 on the oficial mean and the flat decoder 0.777. The flat decoder’s two seeds difer by 0.018, more than this gap (Table 12). The flat decoder reaches 0.777 only by pooling about 256 token states. Its last token scores 0.714 and is below EventGPT’s last hidden state on every label. EventGPT also scores higher on churn (0.935 against 0.899). EventGPT trains at 2.5 times the flat decoder’s throughput at 2,048 tokens and 4.3 times at 4,096 (Table 14) but has a higher next-token loss (1.02 against 0.94). We feed its token decoder only the hidden state after the previous event. The loss gap is largest on the geolocation cell, a field that repeats from recent events: top-1 accuracy 0.57 and 0.58 against 0.68 and 0.69. On the event name the two are level (0.57 against 0.58). A lower next-token loss therefore does not give a better embedding.

## Against feature tables

Within the same 2,048-token window EventGPT scores level with token counts on random windows (0.774 against 0.777) and above them on aligned windows (0.787). Over the full history token counts reach 0.803 and the engineered features 0.816. Token counts also score higher on churn, within the window and over the full history (0.955 against 0.935 and 0.932). On transactions alone EventGPT is level with token counts over the full transaction history (Section I).

## 7.3 Aligned training windows

Aligned windows raise the oficial mean from 0.774 to 0.787 on both seeds (Table 7). They also raise the classification head on every event target and leave the zero-shot scores unchanged (Table 4). Without query-key normalization aligned windows score 0.788 (seed 42, Table 15), so the gain comes from the windows. Churn does not change. The large churn gain that aligned windows first seemed to give came from recognizing the last month (Section 6). Without January 2023 the gain on the product targets holds (0.787 against 0.774). The mismatch between random training windows and evaluation histories costs the product targets about 0.01.

## 7.4 Reducing the gap to engineered features

We score every model on the same test rows as the engineered features, and fit the heads of both seeds on the same training rows (Table 8). The hidden state scores 0.030 and 0.028 below the engineered features. Late fusion of the profile fields brings it to 0.016 and 0.015 below. With early fusion and full-history token counts the gap falls to 0.006 and 0.009 $( p = 0 . 1 6$ and 0.057).

Appendix H breaks the gap down by static attributes, timing, history before the window, knowledge of the schema and amount resolution.

## Rank-averaging ensembles

We test the embedding inside the engineered features, as 32 principal components or as a CatBoost embedding feature. The oficial mean stays the same or falls (Appendix C). We then average the ranks of EventGPT with the profile fields and token counts and of the engineered features, with equal weights. The ensemble reaches 0.830 on both seeds, 0.014 above the engineered features, with intervals that exclude zero (Table 8). Any second model can raise a rank average, so we build the same ensemble without EventGPT: a logistic head on the profile fields, one-hot encoded, and the same token counts, rank-averaged with the engineered features. On the ten fold and seed pairs of Table 9 it scores 0.001 to 0.008 above the engineered features (mean 0.004). The ensemble with EventGPT scores 0.007 to 0.016 above it (mean 0.012, � < 0.01 on each pair, paired standard errors clustered by client). The gain is largest on target\_2 and target\_4 and appears on four of the five event targets. Churn does not change. An ensemble with full-history token counts in place of the engineered features reaches 0.816, the level of the engineered features, with no knowledge of the schema.

Table 8: The gap to the engineered features, aligned windows with query-key normalization, two pretraining seeds. Oficial mean ROC-AUC with its diference from the engineered features (0.816), a 95% interval and � from the paired standard error, clustered by client. The last two rows are rank-averaged ensembles: they average, with equal weights, the ranks given by the embedding with late-fused profile fields and token counts and by CatBoost on token counts over the full history or on the engineered features. The timing and token-count rows each add their features to the hidden state and the late-fused profile fields, not to the row above. Fitted on these training rows, the token counts reach 0.783 against 0.803 in Table 7, most of it on target\_2 (0.780 against 0.849).
<table><tr><td rowspan="2">model</td><td colspan="5">seed 42</td><td colspan="5">seed 43</td></tr><tr><td>mean</td><td></td><td>difference [95% int.]</td><td></td><td>p</td><td>mean</td><td></td><td>difference [95% int.]</td><td></td><td>p</td></tr><tr><td colspan="9">EventGPT</td><td></td></tr><tr><td>hidden state</td><td>0.786</td><td></td><td> -0.030 [−0.038, -0.022]</td><td></td><td>&lt; 0.001</td><td>0.788</td><td></td><td>-0.028 [-0.037, -0.020]</td><td></td><td>&lt; 0.001</td></tr><tr><td>+ profile fields, late fusion</td><td></td><td></td><td>0.800 -0.016 [-0.025, -0.007]</td><td></td><td>&lt; 0.001</td><td>0.801</td><td></td><td>-0.015 [−0.026, -0.005]</td><td></td><td>0.004</td></tr><tr><td colspan="9">EventGPT with early fusion of the profile fields</td><td></td></tr><tr><td>hidden state</td><td>0.784</td><td></td><td>-0.032 [−0.041, -0.022]</td><td></td><td>&lt; 0.001</td><td>0.793</td><td>-0.023</td><td>[-0.032, -0.015]</td><td></td><td>&lt; 0.001</td></tr><tr><td>+ profile fields, late fusion</td><td>0.803</td><td>-0.013</td><td>[-0.023, -0.004</td><td></td><td>0.007</td><td>0.807</td><td>-0.009</td><td>-0.018, +0.000]</td><td></td><td>0.057</td></tr><tr><td>+ profile fields + timing</td><td>0.804</td><td>-0.012</td><td>[-0.022, -0.003]</td><td></td><td>0.009</td><td>0.810</td><td>-0.006</td><td>[−0.014, +0.003]</td><td></td><td>0.19</td></tr><tr><td>+ profile fields + token counts</td><td>0.810</td><td>一 -0.006</td><td>[−0.015, +0.003]</td><td></td><td>0.16</td><td>0.807</td><td>1 -0.009</td><td></td><td>-0.018, +0.000]</td><td>0.057</td></tr><tr><td>ensemble with token counts</td><td>0.816</td><td>+0.000</td><td>[-0.006, +0.007]</td><td></td><td>0.90</td><td>0.816</td><td>+0.001</td><td>[-0.006, +0.007]</td><td></td><td>0.87</td></tr><tr><td>ensemble with engineered features</td><td>0.830</td><td></td><td>+0.014 [+0.010, +0.019]</td><td></td><td>&lt; 0.001</td><td>0.830</td><td></td><td>+0.014 [+0.009, +0.019]</td><td></td><td>&lt; 0.001</td></tr></table>

We fix the weights in advance. With any weight on EventGPT from 0.3 to 0.6 the oficial mean stays within 0.002 of its peak on both seeds. We score the ensemble and the engineered features on the same test rows with the same fit of the engineered features, so the variation between fits does not enter the 0.014. That variation moves full-history token counts from 0.803 to 0.783 between two draws of training rows. A second draw of the training rows of EventGPT’s classification head (seed 42) gives 0.808 against 0.810 and the same ensemble score, 0.830.

## Top of the ranking

On PR-AUC and recall among the top-ranked test rows the ensemble is level with or slightly above the engineered features (Appendix G).

## Five-fold cross-validation

We repeat the configuration of Table 8 with each of folds 1–4 as the test fold. We pretrain on the other four folds with two seeds and refit the engineered features and token counts on each split (Table 9). The ensemble scores above the engineered features on all ten fold–seed pairs, by 0.014 to 0.018 (� < 0.001 on each). EventGPT with the profile fields and token counts scores 0.002 to 0.009 below the engineered features, and no diference is significant $( p \ge 0 . 0 5 7 )$ . On PR-AUC the ensemble scores 0.021 against 0.019 for the engineered features, averaged over folds, and is level on fold 3. In the top 1% of test rows the two hold the same share of positives (0.150). EventGPT with the profile fields and token counts holds 0.132.

Table 9: Oficial mean ROC-AUC with each fold as the test fold, EventGPT with early fusion of the profile fields as in Table 8, mean of two pretraining seeds. Engineered features and token counts are refitted on each split. The last row gives the mean over folds and, in parentheses, the standard deviation. The hidden-state column is the early-fusion model; its fold-0 value (0.789) difers from the aligned column of Table 7 (0.787), which is trained without early fusion.
<table><tr><td rowspan="2">test fold</td><td rowspan="2">engineered features</td><td colspan="2">EventGPT</td><td rowspan="2">ensemble</td><td rowspan="2">ensemble – engineered</td></tr><tr><td>hidden state</td><td>+ profile, counts</td></tr><tr><td>0</td><td>0.816</td><td>0.789</td><td>0.808</td><td>0.830</td><td>+0.014</td></tr><tr><td>1</td><td>0.804</td><td>0.790</td><td>0.802</td><td>0.819</td><td>+0.016</td></tr><tr><td>2</td><td>0.810</td><td>0.782</td><td>0.805</td><td>0.826</td><td>+0.016</td></tr><tr><td>3</td><td>0.813</td><td>0.794</td><td>0.809</td><td>0.830</td><td>+0.017</td></tr><tr><td>4</td><td>0.798</td><td>0.780</td><td>0.796</td><td>0.814</td><td>+0.015</td></tr><tr><td>mean</td><td>0.808 (0.007)</td><td>0.787 (0.006)</td><td>0.804 (0.005)</td><td>0.824 (0.007)</td><td>+0.016 (0.001)</td></tr></table>

## 7.5 Ablations and length of history

We change one element of the model or its data at a time (Appendix I). Pretraining on transactions only raises the oficial mean from 0.774 to 0.793. Temporal RoPE scores 0.781 against 0.778 for learned ordinal positions, a diference smaller than the one between seeds. Dropping the loss on gaps, amount bins and geohash cells leaves the oficial mean unchanged (0.775 against 0.774). Windows of 4,096 and 8,192 tokens do not raise EventGPT’s oficial mean, while token counts gain 0.024 from 2,048 to 8,192 tokens (Appendix J).

## 8 Discussion

On MBD, next-event pretraining scores five event targets without labels at ROC-AUCs that CatBoost on the engineered features reaches only with labels. On four targets, CatBoost needs 10,000 to 300,000 randomly sampled client-months, and on the fifth it remains below the zero-shot score even with 300,000. Event statistics without a trained model, using a Markov chain over event kinds scaled by recent activity, come within two standard errors of EventGPT on three of the five targets but fall below it on 48 of 74 kinds, with a median gap of 0.042. Product purchases require a classification head, and the frozen embedding then scores 0.021 below the engineered features, averaged over five test folds. Adding profile fields and full-history token counts reduces the gap to 0.004. As a second model alongside the engineered features, EventGPT adds 0.014 to 0.017 on each of the five folds, with a mean gain of 0.016. Of this gain, 0.012 remains against the same ensemble built from profile fields and token counts without EventGPT. In the top 1% of test rows, the ensemble and engineered features recall the same share of positives (0.150). At the top 0.5% and 5%, the ensemble recalls slightly more on average (0.103 against 0.100 and 0.362 against 0.358), although the efect varies by target. Precision is the same at all three cutofs. The results suggest that next-event pretraining captures behavioral information useful both for direct event scoring and for ranking alongside conventional features. Static attributes and events outside the model’s window remain important, however, and are better represented by the engineered features.

## Hierarchy benefits

The zero-shot results primarily reflect next-event pretraining rather than the hierarchical architecture: the flat decoder also reaches ROC-AUCs of 0.78 to 0.92. With both models trained on random-ofset windows, their oficial means are similar (0.777 for the flat decoder and 0.774 for EventGPT), while the two flat-decoder seeds difer by 0.018. The hierarchy provides a more useful event-level representation: its last hidden state is the customer embedding, without an additional pooling choice. The flat decoder’s last-token representation scores 0.714; a recency-weighted mean over about 256 token states is needed to reach 0.777.

On churn, EventGPT scores 0.935 against 0.899 for the flat decoder, at both window lengths used to train the flat decoder (Table 15). We add four recency features to the churn classification task: the days since the last event and since the last transaction, and the numbers of events and transactions in the 30 days before the cut. These raise the flat decoder to 0.933 on both seeds and EventGPT to 0.940 and 0.941, suggesting that the EventGPT state already encodes much of the recency signal. With learned ordinal positions in place of temporal RoPE, EventGPT scores 0.927 and 0.928 on churn, compared with 0.933 with temporal RoPE at the same learning rate. This suggests that temporal RoPE contributes to the diference, although the comparison does not isolate its efect from other training diferences.

The hierarchy also reduces training cost. The flat decoder runs all ten layers over every token in the window, with attention cost growing quadratically in the window length. EventGPT runs four of its ten layers over tokens: two in the event encoder and two in the token decoder. Under block attention, a token attends only to tokens in its own event, about eight. The six history layers operate over events, about one for every eight tokens. Thus, each history layer passes one eighth as many tokens through the MLP and attends over about 1/64 as many token pairs as a flat layer. Counted in layer passes, EventGPT makes about 4.75 passes per token, against 10 for the flat decoder.

On one L4 GPU, 20,000 steps (655 million tokens) take EventGPT about 1.5 hours at every window length from 2,048 to 8,192 tokens, and the flat decoder 3.9 hours at 2,048 tokens and 6.2 hours at 4,096 (Table 14). We did not train the flat decoder at 8,192 tokens (Appendix J). At inference, a sampled event requires one history-decoder step over the event states, about 250 in a full window, and one token-decoder pass over the event’s own tokens. A flat decoder with a KV cache requires about eight steps per event, each operating over all token states in the window.

The next-event distribution also answers queries that a classification head cannot. Appending a hypothetical event to a history and reading the distribution after it gives the model’s prediction for clients with that history and event. This prediction is conditional on the history: it reflects which clients had such events in MBD and does not estimate the efect of causing one. The log-likelihood of a client’s recent events under the model could also flag unusual behavior or a change in that client’s activity. MBD has no fraud or distress labels with which to test this use.

The open question is whether the model can carry history beyond its window. We would next test a compressed summary of the earlier history, read as a prefix before the window. Another test would use a lender’s mixed applications, repayments and account events, where a feature table is harder to build.

## Pretraining per institution

Each institution pretrains an event model on its own events. The architecture, the rules that turn an event into tokens and the training procedure carry over to another institution. The weights do not transfer the way a language model’s do. A language model reads text through a vocabulary that all its users share, so one checkpoint serves any text in its languages. An event model’s vocabulary is the issuer’s event schema: its tokens name the event types, field codes and categories of one bank’s systems, and its amount bins are quantiles of that bank’s amounts.

Every field value is an integer code assigned by the bank. Thus, tokens such as name=trx\_1 and s21=18808 have no shared meaning across institutions. The 12,828 tokens and the 21 amount-bin edges were fitted on the pretraining folds of this one population, and a token id has its meaning only together with the vocabulary file it was fitted with. Even within one institution, fitting the vocabulary again assigns ids anew, so we store a fingerprint of the vocabulary with every checkpoint and refuse to score a checkpoint against a vocabulary it was not trained on. The time-gap tokens are the exception, since $\lfloor 8 \ln ( 1 + \Delta s / 8 ) \rfloor$ means the same at every bank.

A checkpoint that transfers would need tokens that mean the same at every bank. The time-gap tokens already do. Event types and fields could be named by text descriptions, as in the masked model trained on data from thousands of North American institutions [17], or mapped to a shared schema of event kinds. Amounts would need bins on a common scale, such as currency units, in place of one bank’s quantiles. Codes with no shared meaning, such as counterparty types, could be hashed into a fixed number of buckets, which fixes the vocabulary size but gives a code no meaning across banks. We have not tested any of these.

Serving such a model starts with pretraining on the institution’s own history, and the comparisons in this paper are between methods trained on the same data. Privacy and competition keep the industrial checkpoints of Section 2 private in any case, and without its issuer’s schema another institution could not use one.

## 9 Limitations

The results are on MBD-mini, for which no published numbers exist, so the comparison with the methods of the MBD paper (Table 2) awaits a run on full MBD. MBD’s events are mostly transactions with one field layout, which gives the event encoder less to do than histories that mix applications, repayments and other events with diferent fields. The geolocation stream starts a year after the transactions, so early as-of dates carry little of it, and dialogue embeddings are not used. Zero-shot scoring covers only targets that are events in the stream; MBD’s product targets are external labels and need a head. Zero-shot scoring predicts the next event, whereas the label records whether the event occurs at any point during the month. Its comparison with the supervised head is therefore evaluated on a sample of the test fold.

The reported standard errors reflect variation across test clients, not variation from trainingset sampling or model initialization. We repeat the main comparison on five test folds and every other result on fold 0 alone. On fold 0, the engineered features and the ensemble score 0.008 and 0.006 above their five-fold means. A head fitted on another draw of training rows moves target\_2, with 132 test positives, by up to 0.07 and the oficial mean by up to 0.02, so diferences of that size between models fitted on diferent draws are not established. We ran fine-tuning once, on one seed with one setting of the learning rates, and chose its step on clients from the pretraining folds. We ran widths beyond 384 with one seed, trained no model larger than 140 million parameters and did not tune the regularization of the head for the wider embeddings.

The split is by client, as in MBD. Pretraining reads the pretraining clients’ events from 2021 and 2022, which include the test clients’ label months, although each test history ends before its label month. We separate the two with an out-of-time run: we pretrain on transactions before February 2022 only, with 54% of the tokens. We compare it with a transactions-only model pretrained over the full period, which scores 0.793 (mean of two seeds). With the same 20,000 steps, the out-of-time model scores 0.789, with churn scores of 0.950 and 0.949. We use transactions alone because the geolocation stream starts in December 2021. At the same number of steps, the out-of-time run sees each token about 4.4 times instead of 2.4. We therefore also compare runs with matched token exposure: at 11,000 steps, the out-of-time run sees each token about 2.4 times. Its oficial mean is 0.787 against 0.793 for full-period pretraining, with churn scores of 0.947 and 0.949. Aligned windows also give the model each client’s tenure through the gap token of the first event, and with tenure the calendar time of the as-of date; the last-month efect on churn (Section 6) is one consequence. In production, as-of dates lie beyond the pretraining period, and what the model learned about calendar time may no longer hold.

## 10 Code and data availability

The code (tokenization, model, training, evaluation and baselines) is at https://github. com/deburky/EventGPT, and the pretrained checkpoints are on the Hugging Face Hub as deburky/EventGPT. MBD-mini is available on the Hugging Face Hub as ai-lab/MBD-mini under CC BY 4.0 [10].

## References

[1] Abnar, S., and Zuidema, W. (2020). Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 4190–4197.

[2] Babaev, D., Ovsov, N., Kireev, I., Ivanova, M., Gusev, G., Nazarov, I., and Tuzhilin, A. (2022). CoLES: Contrastive learning for event sequences with self-supervision. In Proceedings of the 2022 International Conference on Management of Data (SIGMOD). doi:10.1145/3514221.3526129.

[3] Bertasius, G., Wang, H., and Torresani, L. (2021). Is space-time attention all you need for video understanding? In Proceedings of the 38th International Conference on Machine Learning (ICML). arXiv:2102.05095.

[4] Braithwaite, D. T., Cavalcanti, M., McEver, R. A., Udagawa, H., Silva, D., Ramanath, R., Meneses, F., Yoshida, A., Wingert, E., Ramos, M., Zanfelice, B., and Gupta, A. (2025). Your spending needs attention: Modeling financial habits with transformers. arXiv:2507.23267.

[5] DeLong, E. R., DeLong, D. M., and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 44(3), 837–845.

[6] Dou, Y., Jiang, Z., Zhang, T., Hu, M., Xu, Z., Jain, S., Saini, U. S., et al. (2025). TransactionGPT. arXiv:2511.08939.

[7] Fadeev, E., Mollaev, D., Shestov, A., Zoloev, O., Sakhno, A., Korolev, D., Kireev, I., Savchenko, A., and Makarenko, M. (2025). LATTE: Learning aligned transactions and textual embeddings for bank clients. arXiv:2508.10021.

[8] Henry, A., Dachapally, P. R., Pawar, S., and Chen, Y. (2020). Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, 4246–4253.

[9] Liu, S., Li, J., Zhao, G., Zhang, Y., Meng, X., Yu, F. R., Ji, X., and Li, M. (2025). EventGPT: Event stream understanding with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). doi:10.1109/CVPR52734.2025.02713.

[10] MBD-mini. Hugging Face dataset ai-lab/MBD-mini, CC BY 4.0. https://huggingface.co/datasets/ai-lab/MBD-mini.

[11] McDermott, M., Nestor, B., Argaw, P., and Kohane, I. (2023). Event Stream GPT: A data pre-processing and modeling library for generative, pre-trained transformers over continuous-time sequences of complex events. In Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track.

[12] Mollaev, D., Kostin, A., Postnova, M., Karpukhin, I., Kireev, I., Gusev, G., and Savchenko, A. (2024). Multimodal Banking Dataset: Understanding client needs through event sequences. arXiv:2409.17587.

[13] NVIDIA. Build your own transaction foundation model for financial intelligence. https://developer.nvidia.com/blog/ build-your-own-transaction-foundation-model-for-financial-intelligence/.

[14] Obuchowski, N. A. (1997). Nonparametric analysis of clustered ROC curve data. Biometrics 53(2), 567–578.

[15] Ostroukhov, M., Mikhailov, R., Iashin, V., Sokolov, A., Akshonov, A., Protasov, V., Goncharov, A., Beloborodov, D., Mullin, V., Enzmann, R. Y., Kolovos, G., Renders, J., Nesterov, P., and Repushko, A. (2026). PRAGMA: Revolut foundation model. arXiv:2604.08649.

[16] Padhi, I., Schif, Y., Melnyk, I., Rigotti, M., Mroueh, Y., Dognin, P., Ross, J., Nair, R., and Altman, E. (2021). Tabular transformers for modeling multivariate time series. In ICASSP 2021. arXiv:2011.01843.

[17] Polleti, G., Santana, M., and Fontes, E. (2025). Open Banking foundational model: Learning language representations from few financial transactions. arXiv:2511.12154.

[18] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. In Advances in Neural Information Processing Systems 31.

[19] Sakhno, A., et al. (2025). PyTorch-Lifestream: Learning embeddings on discrete event sequences. In Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI), Demo Track, 11104–11108. doi:10.24963/ijcai.2025/1272. https://github.com/dllllb/pytorch-lifestream.

[20] Starling Bank (2026). Starling backs UK sovereign cloud. Press release, 14 October 2026. https://www.starlingbank.com/news/starling-backs-uk-sovereign-cloud/.

[21] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021). RoFormer: Enhanced transformer with rotary position embedding. arXiv:2104.09864.

[22] Xu, J., Guo, Z., He, J., Hu, H., et al. (2025). Qwen2.5-Omni technical report. arXiv:2503.20215.

[23] Zuo, S., Jiang, H., Li, Z., Zhao, T., and Zha, H. (2020). Transformer Hawkes process. In Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 119, 11692–11702.

## A Paired tests of the zero-shot scores

We test each zero-shot score against the supervised head of the same model and seed. The head scores 0.045 to 0.061 higher on 26/28 and 48/10 for both seeds of both models $( p <$ 0.001). On 23/49 the head scores 0.008 and 0.011 higher with random windows $( p = 0 . 3 3$ and 0.16) and 0.025 and 0.034 higher with aligned windows $( p = 0 . 0 0 3$ and $< 0 . 0 0 1 )$ . On $4 8 / 3 1$ the zero-shot score is 0.029 higher with aligned windows and seed 43 $( p = 0 . 0 0 8 )$ . On 50/19 it is 0.012 higher with random windows and seed 43 $( p = 0 . 0 4 )$ . The other six runs on these two targets show no significant diference $( p \ge 0 . 0 7 )$ . We run 20 paired tests: five targets for both seeds of both models. At a Bonferroni threshold of 0.0025 the head stays significantly higher on $2 6 / 2 8$ and $4 8 / 1 0$ in every run and on 23/49 with aligned windows and seed 43. Neither zero-shot win passes this threshold. Training on aligned windows leaves the zero-shot scores unchanged.

We also multiply the score by one plus the client’s transactions in the 30 days before the as-of date, a form we chose after the scores of Table 4 were known (Section I). We test the scaled score against the head for the same five targets and four runs. It scores 0.013 to 0.043 higher on $4 8 / 3 1$ and 0.012 to 0.029 lower on 26/28 and 48/10. On 23/49 and 50/19 the two difer by at most 0.014 $( p \ge 0 . 0 2 7 )$ . These 20 tests and the 20 above make 40, for a Bonferroni threshold of 0.00125. The head’s results above still pass it. Of the new tests only the head on 26/28 with aligned windows and seed 42 passes $( p < 0 . 0 0 1 )$ . The scaled score on $4 8 / 3 1$ with aligned windows and seed 43 comes closest $( p = 0 . 0 0 1 4 )$

## B Delta scoring

A client’s history ending at its as-of date lets a daily job rescore only the clients whose input changed (Figure 11). A client’s embedding depends only on the tokens of its history, so a client with no new event keeps its embedding and its scores. The now event is the exception. Its dt token measures the silence up to the as-of date and moves to the next bucket as the silence grows. A bucket spans about one eighth of the gap, so a client silent for � days changes bucket on about $8 / g$ of days once � exceeds eight. The job therefore rescores every client with a new event and every client whose now token changed. It then reproduces a full pass exactly, provided the streams are built with the vocabulary of the checkpoint (Section 8).

![](images/0993159913a34396fc5f39d6a7834aa8ddef5dcc79a4466166a82a56df29eed0.jpg)  
Figure 11: Delta scoring with a now event. Client A has a new event since yesterday’s run. Client B has none, but its silence grew from three to four days and its now token moved from dt=83 to dt=85. Client C has been silent for 45 days, and one more day leaves its now token at dt=104. The job rescores A and B and keeps C’s scores from yesterday.

We count both groups for each day of 2022 on MBD-mini. With geolocation, 57% of clients have a new event on an average day and a further 32% have a new now token, so the job rescores 90% of clients. On transactions alone the shares are 31% and 48%, and the job rescores 79%. The saving is small on MBD because most of its clients act every few days. It grows with the share of clients who stay silent for weeks.

## C Fine-tuning and gradient boosting on the embedding

We fine-tune the aligned checkpoint of seed 42, trained without query-key normalization, end to end. We add one logistic output per label and train on the training rows of the classification heads. We train only the head for the first 2,000 steps and then every weight for one pass, at a peak learning rate of $3 \times 1 0 ^ { - 5 }$ for the pretrained weights and $1 0 ^ { - 3 }$ for the head. We choose the step by the oficial mean on a tenth of the training clients. The oficial mean stays at 0.788. The five event targets rise in this run, which has one seed and one setting of the learning rates. On 48/31 and 23/49 the fine-tuned model scores above the engineered features (0.940 and 0.931 against 0.920). This run kept January 2023 in the churn label, so its churn score includes the last-month efect. The validation clients belong to the pretraining folds, and their oficial mean is 0.06 above the test mean.

We test the embedding inside CatBoost, as 384 numeric columns or as one embedding feature. With an embedding feature CatBoost computes its own features, by default linear discriminant projections and counts of nearest training rows per class. The numeric columns score 0.775 and the embedding feature 0.700, against 0.788 for the logistic head. The embedding feature scores 0.501 on target\_4. We also add the embedding feature to the engineered features. Their oficial mean falls from 0.815 to 0.808; these rows were drawn before January 2023 left the churn label, so the engineered features score 0.815 here against 0.816 in Table 7.

## D Training windows and tenure

Windows drawn at random ofsets usually start partway through a client’s history and lack [USR] and the profile event. Every evaluation history opens with both. We remove this mismatch with aligned windows (Figure 7). When a window starts partway through a client’s history, we put [USR] and the profile event back at its start, as we do when we truncate a long evaluation history. We stamp the profile event at the client’s first event. We start both the dt token of the next event and the temporal RoPE positions of Eq. (1) at the client’s first event. Apart from the profile event and this one dt token, an aligned window holds the same tokens as a random one. We build evaluation and zero-shot histories the same way for these models.

## E Point-in-time guarantees

## In the architecture

The history decoder is causal, so the hidden state after event � depends on events 1, … , � only. The event encoder reads in both directions only within one event, and the token decoder reaches earlier events only through $h _ { i - 1 }$ . No token of event � has a path to the encoded vector of event � itself. The embedding after the last event before an as-of date is the same whether or not later events exist in the stream. Evaluation also builds each input to end at its as-of date, so later events are never present in it.

## In the data

How events are built matters as well, and each step of Section 5 keeps the property.

• The history for an as-of date is every event strictly before the first day of the month, in the bank’s own time zone.

• A daily geolocation event is stamped at the day’s last ping, so a summary of the day never appears before the pings it summarizes.

• The profile event carries fields that never change within a client and is stamped at the client’s first event, so it enters a history only when some event does.

• The vocabulary and the amount-bin edges are fitted on the pretraining folds only, and the test fold is never seen in pretraining.

Because the history itself fixes its as-of date, an embedding can be stored in a feature store like any other dated feature. It is keyed by client, stamped with its as-of date instead of the time it was computed, and joined to labels by the store’s point-in-time join, which returns for each label the latest embedding before it. The join decides which embedding a label sees, and since each history ends at its as-of date, that embedding depends only on earlier events. Embeddings from diferent checkpoints are not comparable, so each version of the backbone needs its own feature set, and a head is valid only on the version it was fitted on.

## Delta scoring

The same property lets a daily job rescore only the clients whose input changed and reproduce a full pass exactly (Appendix B).

## F Tokenization

## Profile event

The four fields src\_type21, src\_type22, src\_type31 and src\_type32 never change within a client: each takes one value per client in all 98,721 clients with transactions. On fold 1, by contrast, src\_type11 and src\_type12 change on 13–15% of consecutive transactions, and the destination fields and the event type on 55–63%. The four fields are attributes of the client’s account. Repeated on every transaction they took 4 of 13 tokens, so we state them once, in a profile event at the time of the client’s first event. Transactions shrink to 9 tokens, and the pretraining stream from 454 million to 332 million tokens.

## Geolocation

Raw pings outnumber transactions 1.7 to 1 and would crowd them out of a window. We keep one geolocation event per client per Moscow day. It carries the day’s most frequent geohash\_5 cell and the geohash\_4 cell that contains it, and it is stamped at the day’s last ping, so it never runs ahead of the pings it summarizes. This leaves 14.7 million events.

## Vocabulary

Values seen in the pretraining folds become tokens. Values of the two counterparty-like fields seen fewer than 1,000 times, and geohash cells seen fewer than 100 (g4) or 300 (g5) times, fall into one other token per key. The vocabulary has 12,828 tokens, and no token of the held-out fold is unknown.

Table 10: Engineered features fitted on � labeled client-months drawn at random, against Event-GPT zero-shot (mean of the two seeds of Table 4, no labels). ROC-AUC on the test fold, mean over draws (five up to 30,000 rows, three at 100,000, two at 300,000), with the rule for earlier occurrences applied to both; in brackets the mean number of positives in a draw. The last column is the engineered-feature baseline of Table 4, fitted on every positive of the pretraining folds and 60,000 negatives, under the same rule.
<table><tr><td rowspan="2">target</td><td rowspan="2">zero-shot (no labels)</td><td colspan="6">engineered features, labeled rows n</td><td rowspan="2">all labels</td></tr><tr><td>1,000</td><td>3,000</td><td>10,000</td><td>30,000</td><td>100,000</td><td>300,000</td></tr><tr><td>first 48/31</td><td>0.909</td><td>0.573 (0)</td><td>0.650 (0)</td><td>0.718 (2)</td><td>0.871 (6)</td><td>0.894 (19)</td><td></td><td>0.921</td></tr><tr><td>first 23/49</td><td>0.918</td><td>0.784 (2)</td><td>0.809 (2)</td><td>0.866 (7)</td><td>0.856 (25)</td><td>0.896 (81)</td><td>0.912 (66) 0.902 (263)</td><td>0.930</td></tr><tr><td>first 26/28</td><td>0.828</td><td>0.735 (2)</td><td>0.739 (4)</td><td>0.776 (11)</td><td>0.817 (34)</td><td>0.858 (128)</td><td>0.849 (355)</td><td>0.880</td></tr><tr><td>first 48/10</td><td>0.787</td><td>0.739 (2)</td><td>0.668 (5)</td><td>0.695 (11)</td><td>0.763 (34)</td><td>0.771 (134)</td><td>0.826 (394)</td><td>0.841</td></tr><tr><td>first 50/19</td><td>0.905</td><td>0.746 (2)</td><td>0.821 (8)</td><td>0.854 (34)</td><td>0.921 (95)</td><td>0.929 (342)</td><td>0.932 (992)</td><td>0.937</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## G Top of the ranking

On PR-AUC the hidden state scores below the engineered features on every oficial target, by more than its ROC-AUC suggests (Table 11). With the profile fields and token counts EventGPT scores above the engineered features on target\_2 and target\_4. On fold 0 the ensemble has the highest PR-AUC and top-1% recall. Over five folds it ties with the engineered features on top-1% recall. We also measure recall in the top 0.5%, 1% and 5% of test rows over five folds and two seeds, averaged over the four product targets. The ensemble recalls 0.103, 0.150 and 0.362 against 0.100, 0.150 and 0.358 for the engineered features. It recalls more on target\_4 in 28 of the 30 fold, seed and cut combinations and less on target\_1 in the top 0.5% in all ten. Precision in the same cuts is 0.040, 0.032 and 0.018 for both.

Table 11: PR-AUC on the oficial targets, mean of two seeds, and the share of positives in the top 1% of test rows. A random ranking scores the prevalence and 0.010. Late fusion and the ensemble are those of Table 8.
<table><tr><td></td><td></td><td>engineered</td><td colspan="2">EventGPT</td><td></td></tr><tr><td>target</td><td>prevalence</td><td>features</td><td></td><td>hidden state + profile fields, token counts ensemble</td><td></td></tr><tr><td>target_1</td><td>0.0044</td><td>0.023</td><td>0.014</td><td>0.016</td><td>0.023</td></tr><tr><td>target_2</td><td>0.0005</td><td>0.014</td><td>0.008</td><td>0.017</td><td>0.021</td></tr><tr><td>target_3</td><td>0.0038</td><td>0.021</td><td>0.014</td><td>0.019</td><td>0.022</td></tr><tr><td>target_4</td><td>0.0026</td><td>0.017</td><td>0.011</td><td>0.018</td><td>0.019</td></tr><tr><td>mean</td><td>0.0028</td><td>0.019</td><td>0.012</td><td>0.017</td><td>0.021</td></tr><tr><td>recall, top 1%</td><td>0.010</td><td>0.157</td><td>0.113</td><td>0.154</td><td>0.166</td></tr></table>

## H Sources of the gap to engineered features

## Static attributes

Without the profile fields the engineered features lose most on target\_4 (0.036). We add the profile fields to the hidden state by late fusion (Section 3.5). target\_4 rises from 0.754 to 0.821 (seed 42). With early fusion as well it reaches 0.830 and 0.822 against 0.818 for the engineered features (� = 0.11 and 0.61). We also pretrain with early fusion. The hidden state then scores 0.789 against 0.787 without it.

Table 12: EventGPT and the flat decoder, both on windows at random ofsets, per pretraining seed. Throughput on one NVIDIA L4; validation loss and top-1 accuracy of the next token at the end of pretraining; the oficial mean uses the last hidden state for EventGPT and the recencyweighted embedding for the flat decoder. The flat decoder attends to every earlier token, so it can copy fields of recent events, and most of its lower loss is on the geolocation cell; on the event name the two are level. Its oficial mean is 0.003 above EventGPT’s, less than the 0.018 between its own two seeds.
<table><tr><td rowspan="2"></td><td colspan="2">EventGPT</td><td colspan="2">flat decoder</td></tr><tr><td>seed 42</td><td>seed 43</td><td>seed 42</td><td>seed 43</td></tr><tr><td>throughput (tokens per second)</td><td>117,300</td><td>117,600</td><td>47,100</td><td>55,600</td></tr><tr><td>validation loss</td><td>1.023</td><td>1.020</td><td>0.944</td><td>0.947</td></tr><tr><td>top-1 accuracy, geohash_5</td><td>0.582</td><td>0.567</td><td>0.686</td><td>0.680</td></tr><tr><td>top-1 accuracy, event name</td><td>0.573</td><td>0.574</td><td>0.579</td><td>0.580</td></tr><tr><td>official mean ROC-AUC</td><td>0.774</td><td>0.775</td><td>0.786</td><td>0.768</td></tr></table>

## Timing

We add five timing features to the hidden state: the days since the first and the last event and the event counts over 30 days, 90 days and all history. They raise the oficial mean by 0.001 and 0.003 and leave churn unchanged. The engineered features lose 0.009 without their own timing features, so the hidden state already carries this information.

## History before the window

We add the full-history token counts to the hidden state and the profile fields. They raise churn, first 50/19 (from about 0.84 to 0.92) and first 48/10 (from about 0.80 to 0.83) on both seeds. Token counts over EventGPT’s window score 0.026 below token counts over the full history. A longer window did not recover this history for EventGPT (Section J).

## Schema knowledge

The engineered features score 0.816 and full-history token counts 0.803, so knowledge of the schema adds 0.013.

## Amount resolution

EventGPT sees each amount as one of 21 quantile bins. We replace each amount in the engineered features by the mean of its bin. The oficial mean falls by 0.005 with 21 bins and does not change with 256 bins.

## I Ablations

We change one element of the model or its data at a time (Table 13). We run two seeds except where marked.

Table 13: Mean ROC-AUC over the oficial targets, logistic head. Rows with two seeds give the mean and, in brackets, the two seeds’ scores; the row marked <sup>‡</sup> uses transactions only; the row marked <sup>§</sup> trains at learning rate $3 \times 1 0 ^ { - 4 }$ . Every row but the windows and width rows uses windows at random ofsets; the width rows add QK normalization. The pretraining-period rows compare with transactions only over the full period (0.793), and every other row with the first.
<table><tr><td>element</td><td>variant</td><td>seeds</td><td>mean ROC-AUC</td></tr><tr><td rowspan="3">architecture</td><td>hierarchy (EventGPT), last state</td><td>2</td><td>0.774 (0.774, 0.775)</td></tr><tr><td>flat decoder, last token</td><td>2</td><td>0.714 (0.713, 0.714)</td></tr><tr><td>flat decoder, recency over tokens</td><td>2</td><td>0.777 (0.768, 0.786)</td></tr><tr><td rowspan="2">positions§</td><td>temporal RoPE</td><td>2</td><td>0.781 (0.777, 0.786)</td></tr><tr><td>ordinal</td><td>2</td><td>0.778 (0.775, 0.780)</td></tr><tr><td rowspan="2">modalities</td><td>transactions + geolocation</td><td>2</td><td>0.774 (0.774, 0.775)</td></tr><tr><td>transactions only</td><td>2</td><td>0.793 (0.791, 0.795)</td></tr><tr><td rowspan="2">pretraining period‡</td><td>before Feb. 2022, same steps</td><td>2</td><td>0.789 (0.788, 0.790)</td></tr><tr><td>before Feb. 2022, same passes per token</td><td>2</td><td>0.787 (0.784, 0.791)</td></tr><tr><td>training loss</td><td>without dt, amt_bin, g4, g5</td><td>1</td><td>0.775</td></tr><tr><td rowspan="2">embedding</td><td>mean</td><td>2</td><td>0.755 (0.752, 0.758)</td></tr><tr><td>recency</td><td>2</td><td>0.765 (0.761, 0.769)</td></tr><tr><td>training windows</td><td>aligned, with QK normalization</td><td>2</td><td>0.787 (0.786, 0.788)</td></tr><tr><td>attention</td><td>QK normalization</td><td>1</td><td>0.779</td></tr><tr><td>width</td><td>384 / 512 / 1024, aligned</td><td>1</td><td>0.786 / 0.791 / 0.791</td></tr><tr><td>event encoder</td><td>one feed-forward per event kind</td><td>1</td><td>0.779</td></tr></table>

We pretrain and evaluate on transactions alone. The oficial mean rises from 0.774 to 0.793 and churn from 0.934 to 0.950 on both seeds, and the classification head rises on all five event targets. Without geolocation the engineered features lose 0.001 while EventGPT gains 0.019. Without daily geolocation events the 2,048-token window holds more transactions. Token counts show the same efect: their loss to the window shrinks from 0.026 to 0.008. Within the transactions-only window EventGPT scores above token counts and level with token counts over the full transaction history. Every other result here uses geolocation, as MBD does.

Without geolocation the zero-shot scores fall to 0.72 to 0.89, from 0.79 to 0.92. The label asks whether the kind occurs within the month, which also depends on how many events the month holds. With a geolocation event on most days, a client who transacts rarely has a lower score for every transaction kind. The score then carries the client’s activity, which the monthly label rewards. We correlate the log score with the log of the client’s recent transactions: 0.39 to 0.49 with geolocation and −0.15 to 0.20 without (seed 42, as in the

next paragraph).

We multiply the score by one plus the client’s transactions in the 30 days before the as-of date. This uses no labels. The scaled score reaches 0.79 to 0.93 on transactions alone and 0.82 to 0.93 with geolocation, above the count alone. We first tried this scaling on the test fold, after the scores of Table 4 were known, beside a second form based on transactions per day over the whole history. We then compared the two on training fold 1. The 30-day count scores above the daily rate for every model and target and raises every unscaled score. Table 4 keeps the unscaled scores, which use only the model.

We remove the loss on gaps, amount bins and geohash cells, the tokens a product target is least likely to need. We keep these tokens in the input and drop only their prediction. The oficial mean does not change (0.775 against 0.774). Churn rises slightly (seed 42), and the zero-shot scores change by at most 0.023.

We replace temporal RoPE with learned ordinal positions in the history decoder. At the learning rate of the other rows, $6 \times 1 0 ^ { - 4 }$ , the ordinal model diverges after 3,000 to 5,000 steps in both seeds. At $3 \times 1 0 ^ { - 4 }$ both train stably. Temporal RoPE scores 0.781 and ordinal positions 0.778. Temporal RoPE scores higher on both seeds, but the mean diference, 0.003, is smaller than the diference between two seeds of either model (0.009 and 0.005). The ordinal model also adds learned positions within each event, which the RoPE model leaves out, so the comparison does not isolate the position scheme.

We also read the history decoder’s attention as weights over a client’s events. We average the heads and roll the attention out through the six layers [1], with the residual path weighted one half. We pool the hidden states with the last event’s rolled-out weights. On 2,000 test histories the weights spread over a median of 102 of 219 events (the exponential of their entropy) and put 2.4% on the last event, so the pooled vector is close to the mean. Averaged over two seeds it scores 0.760 (0.757 and 0.763), between mean pooling (0.757) and recency pooling (0.765) and 0.015 below the last hidden state (0.774). We also learn the pooling: an attention head over the frozen hidden states of the last 128 events, with one learned query per label, fitted on 60,000 training rows. On aligned windows it scores 0.718 and 0.727 on the two seeds, against 0.784 and 0.790 for the last hidden state on the same rows.

We test query-key normalization and one feed-forward network per event kind in the event encoder (transactions, geolocation and the rest, routed by the event name). Each adds 0.005 on seed 42. Two seeds of the same configuration difer by up to 0.018 for the flat decoder and up to 0.022 for EventGPT across the five test folds, so a diference of 0.005 on one seed is not established.

We test width 512 (16 heads of width 32, 38.4 million parameters) and width 1024 (16 heads of width 64, 139.7 million) on aligned windows with query-key normalization (seed 42). Both lower the validation loss, to 0.997 at width 1024 from 1.020, and raise the oficial mean from 0.786 to 0.791. With late fusion of the profile fields the gain disappears. At width 1024 late fusion lowers the oficial mean, while it raises it at the smaller widths. The four field embeddings add 4,096 columns to a head regularized as at width 384, which may explain this.

## J Length of history

We train EventGPT and the flat decoder on windows of 4,096 and 8,192 tokens (Table 15). MBD histories are long: the 90th percentile client has about 1,000 transactions and 300 days of geolocation, while a 2,048-token window holds about 250 events. Token counts gain 0.024 from 2,048 to 8,192 tokens. EventGPT stays level on random windows (0.774 to 0.772) and loses 0.008 on aligned windows (0.788 to 0.780). The flat decoder stays level at 4,096 tokens. Churn does not change with the window for any model. The extra history carries information for the product targets, and EventGPT’s embedding does not take it up at this size. Pooled readouts do not take it up either: at 8,192 tokens mean and recency pooling score 0.748 and 0.762 on random windows and 0.743 and 0.765 on aligned windows, below the last hidden state.

EventGPT trains at about 126,000 tokens per second at 8,192 tokens against 118,000 at 2,048. Its history decoder attends over events, and with block attention its token stages grow linearly with the window. The flat decoder falls from 47,000 tokens per second at 2,048 tokens to about 30,000 at 4,096. We did not train it at 8,192 tokens: at 4,096 a run already takes about six hours on the L4, and its attention cost grows with the square of the window.

## Block attention within events

Neither the event encoder nor the token decoder attends across an event boundary (Figure 4), but a dense attention mask over a window of � tokens still costs memory in proportion to $t ^ { 2 }$ We lay each event out instead as one row of at most � slots, with � the longest event in the window (ten tokens on MBD), and attend within rows. Both stages then cost in proportion to the number of events times $L ^ { 2 }$ . On real windows the block and the dense computations agree to within $1 . 2 \times 1 0 ^ { - 6 }$ on every event. Block attention makes windows of 8,192 tokens afordable.

Table 14: Training time on one NVIDIA L4 GPU for 20,000 steps of 32,768 tokens (655 million tokens), windows at random ofsets, one run per cell, without query-key normalization. Throughput is in thousands of tokens per second, from the training log; for runs resumed after a preemption it is measured within each uninterrupted segment. Query-key normalization slows EventGPT to about 88,000 tokens per second (2.1 hours at 2,048 tokens, aligned windows).
<table><tr><td colspan="3">EventGPT</td><td colspan="2">flat decoder</td></tr><tr><td>window</td><td>hours</td><td>tokens/ s/s</td><td>hours</td><td>tokens/s</td></tr><tr><td>2,048</td><td>1.5</td><td>118k</td><td>3.9</td><td>47k</td></tr><tr><td>4,096</td><td>1.5</td><td>125k</td><td>6.2</td><td>29k</td></tr><tr><td>8,192</td><td>1.4</td><td>126k</td><td></td><td></td></tr></table>

Table 15: Mean ROC-AUC over the oficial targets (churn without January 2023 in parentheses) against the window, seed 42 for the models. The aligned column is trained without query-key normalization (0.786 with it at 2,048 tokens). Token counts at full history reach 0.803.
<table><tr><td rowspan="2">window</td><td rowspan="2">token counts (CatBoost)</td><td colspan="2">EventGPT</td><td rowspan="2">flat decoder recency</td></tr><tr><td>random</td><td>aligned</td></tr><tr><td>2,048</td><td>0.777</td><td>0.774 (0.934)</td><td>0.788 (0.933)</td><td>0.786 (0.900)</td></tr><tr><td>4,096</td><td>0.782</td><td>0.772 (0.936)</td><td></td><td>0.785 (0.896)</td></tr><tr><td>8,192</td><td>0.801</td><td>0.772 (0.935)</td><td>0.780 (0.935)</td><td></td></tr></table>

![](images/f414deeb5436a2bf8f2b7268b41c12ad59b4bc8504770200cb2c9c7862c12d01.jpg)  
Figure 12: Share of rows positive for each label by month, on a log scale, over all five folds (the test fold in full, the training rows weighted back to their share). Product targets are solid, event targets dashed and churn blue. January 2023 follows the last transaction, so the event targets and churn end in December.