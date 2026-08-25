# RAD: Rule-Augmented Relational Anomaly Detection

Noah Dahle   
Vanderbilt University   
Nashville, Tennessee, USA   
noah.dahle@vanderbilt.edu

Anne Tumlin Vanderbilt University Nashville, Tennessee, USA anne.m.tumlin@vanderbilt.edu

Xenofon Koutsoukos Vanderbilt University Nashville, Tennessee, USA xenofon.koutsoukos@vanderbilt.edu

Ngoc Tran Vanderbilt University   
Nashville, Tennessee, USA   
ngoc.n.tran@vanderbilt.edu   
Tyler Derr   
Vanderbilt University   
Nashville, Tennessee, USA   
tyler.derr@vanderbilt.edu

## Abstract

Anomaly detection is often applied to data stored in relational databases, yet most existing methods require flattening multiple tables into a single feature matrix. This flattening can obscure entity identity, schema structure, and multi-hop dependencies, limiting the detection of anomalies that depend on relational context rather than isolated feature values. Beyond preserving relational structure, relational anomaly detection raises an additional challenge: how to incorporate symbolic behavioral evidence into learned relational representations. To address these challenges, we study relational anomaly detection, where the goal is to identify anomalous entities or events in a multi-table database. We propose RAD, a rule-augmented relational anomaly detector that combines heterogeneous graph representation learning with refined symbolic rule signals. RAD derives candidate rules from random-forest paths over flattened summaries of the entities or events being scored, refines them into compact interpretable predicates, injects the resulting rule features into the graph model, and learns anomaly scores using reconstruction-based and pairwise-ranking supervision. To evalu ate this setting, we introduce a relational anomaly detection bench mark spanning three settings: LANL cybersecurity event detection and two unexpected user-churn anomaly tasks derived from Amazon and H&M relational databases. Experiments show that RAD improves anomaly ranking over flattened tabular detectors and relational baselines under natural class imbalance, achieving the best average rank on AUROC and AUPRC across the benchmark. Ablations show that direct rule injection and ranking-based supervision are key contributors to performance, while edge reconstruction is not uniformly beneficial. Our code and data are available at: https://github.com/noahd15/RAD\_RelationalAnomalyDetection.

## Keywords

relational anomaly detection, symbolic rule mining, relational deep learning, cybersecurity event detection, unexpected user-churn ACM Reference Format:

Noah Dahle, Anne Tumlin, Ngoc Tran, Xenofon Koutsoukos, and Tyler Derr. 2026. RAD: Rule-Augmented Relational Anomaly Detection. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge

![](images/fff0735977686e93c10503f702e50c0f068acab3faf9ee00ce644e8c04452c3f.jpg)

Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3799682.3841086

## 1 Introduction

Anomaly detection is widely used to identify rare or unexpected behavior in domains such as cybersecurity, fraud detection, healthcare, and cyber-physical systems [5, 7, 19]. In many of these settings, the data is not naturally stored as a single table of independent feature vectors. Instead, the data is stored in relational databases containing multiple entity and event types connected through primaryforeign key relationships. For example, a cybersecurity database may include users, machines, and authentication events. A single authentication event may look normal in isolation but become suspicious when linked to an unusual user-machine pair [11, 20]. This creates two related challenges for anomaly detection on relational databases: preserving relational structure and incorporating symbolic behavioral evidence into learned anomaly representations.

Consider an authentication event that is unremarkable in isolation but anomalous because the user-machine pair has never co-occurred before, the machine is otherwise accessed only by service accounts, and the user authenticated to four other machines in the preceding ten minutes. Recovering these facts requires joins across authentication history and multi-hop neighbor context — information a flattened row-level summary discards.

To apply these methods to relational databases, practitioners often flatten multiple tables into a single feature matrix using joins, aggregations, and hand-crafted features [11, 20, 25]. Although this makes the data compatible with tabular anomaly detectors, it can also obscure the signals that define relational anomalies. Entity identity, schema roles, multi-hop dependencies, and temporal context may be lost during flattening [11, 20]. As a result, anomalies whose abnormality depends on cross-table structure can be dificult to detect from flattened row-level features alone.

In this work, we study relational anomaly detection: the task of ranking anomalous target entities or events in a multi-table relational database. In this setting, each target instance must be scored not only from its own attributes but also from its relational context. This context may include linked entities, typed relationships, historical activity, and schema-dependent interactions. Unlike ordinary tabular anomaly detection, relational anomaly detection requires models to preserve and reason over the structure of the database while still producing anomaly scores that are useful under severe class imbalance. Recent work on relational deep learning has shown that relational data can be modeled directly as graphs while preserving table and relationship structure, but this line of work is not itself an anomaly detection framework and does not define anomaly-specific scoring objectives [11, 20]. Moreover, preserving relational structure alone is not suficient for relational anomaly detection. Many anomalies correspond to behavioral or semantic patterns that are dificult to capture from relational message passing alone. In practice, anomaly detection systems often rely on symbolic behavioral rules, heuristics, or analyst-defined patterns that encode suspicious combinations of events, entities, or activities. A central challenge is therefore how to incorporate symbolic behavioral evidence into learned relational representations while still preserving end-to-end anomaly-oriented learning.

We propose RAD, a rule-augmented relational anomaly detector for multi-table databases. RAD represents a relational database as a heterogeneous graph, mines symbolic rules from an auxiliary flattened view using random forests [2], injects the resulting rule features into target nodes, and learns anomaly scores via a het erogeneous graph autoencoder with GraphSAGE-style message passing [12], reconstruction-based scoring, and pairwise ranking supervision. We evaluate RAD on a new relational anomaly detection benchmark spanning LANL cybersecurity event detection and two unexpected user-churn tasks derived from the Amazon and H&M relational databases in RelBench [20].

Experiments on LANL, Amazon, and H&M compare RAD against flattened tabular anomaly detectors [17, 21–23, 25], as well as relational baselines based on Relational Deep Learning [11]. More specifically, we structure our empirical evaluation around five research questions: (RQ1) how RAD compares with tabular and relational baselines across relational anomaly tasks; (RQ2) whether mined rule features improve detection; (RQ3) the efect of graph edge reconstruction; (RQ4) how much LLM-assisted rule refinement adds over random-forest rules alone; and (RQ5) how sensitive unexpected-churn results are to anomaly prevalence. Results show that relational modeling and rule augmentation improve anomaly detection, especially on ranking-oriented metrics such as AUPRC and precision@k. These metrics are important because anomaly detection is usually highly imbalanced, and practical systems often care most about whether true anomalies appear near the top of the ranked anomaly list [4]. The results also show that rule features are most useful when injected into the graph representation rather than added after graph scoring. Ranking-based supervision further improves anomaly retrieval under class imbalance. Ablations show that direct rule injection and ranking-based supervision are key contributors to performance, while edge reconstruction has dataset-dependent efects and is not uniformly beneficial.

Our contributions are summarized as follows:

• We formalize relational anomaly detection as distinct from standard tabular and graph anomaly detection, where anomalousness may depend on cross-table, multi-hop, temporal, and schema-dependent interactions.

• We propose RAD, a rule-augmented relational anomaly detector for multi-table databases. RAD mines rules from relational summaries, injects rule-derived features into target nodes, and learns anomaly scores with heterogeneous-graph representations, reconstruction, and pairwise-ranking objectives.

• We introduce a relational anomaly detection benchmark suite combining LANL cybersecurity event detection with two unexpected user-churn anomaly tasks derived from the Amazon and H&M datasets [20]. The benchmark provides unified target definitions, anomaly-label constructions, relational graph construction, and ranking-based evaluation protocols.

• We empirically compare RAD against tabular anomaly detectors, relational baselines, and targeted ablations, showing that direct rule injection improves anomaly retrieval under severe class imbalance.

## 2 Related Work

Existing anomaly detection methods difer in how they represent data, how they define anomalousness, and whether they expose interpretable evidence for a detected anomaly. Relational anomaly detection requires all three: the model must preserve multi-table structure, produce useful anomaly rankings under severe class imbalance, and integrate symbolic behavioral evidence that may be dificult to express through relational message passing alone. Prior work addresses these requirements only partially. In this section, we review these existing paradigms and close by positioning our proposed framework against the limitations of current literature.

## 2.1 From Tabular Anomaly Detection to Relational Structure

Most classical anomaly detection methods assume each instance can be represented as an independent feature vector. Isolation Forest [17], one-class SVM [23], autoencoder-based anomaly detection [22], Deep SVDD [21], and recent tabular representationlearning methods, e.g., DRL [25], define anomalousness through isolation, support estimation, reconstruction error, compact latent representations, or learned feature decompositions.

The dificulty is that relational databases are not naturally collections of independent feature vectors. Applying tabular anomaly detectors to multi-table data requires flattening the database through joins, aggregations, and hand-crafted features. Flattening makes standard detectors applicable, but it changes the object being modeled: entity identity, schema roles, multi-hop dependencies, and temporal context must be compressed into row-level summaries. This is a poor fit for anomalies whose suspiciousness is relational, such as an event that appears ordinary in isolation but unusual given the user, machine, product, transaction, or activity history to which it is connected. Thus, tabular anomaly detection provides useful scoring objectives, but its representation is mismatched to relational anomaly detection.

## 2.2 From Graph Anomaly Detection to Relational Databases

Graph-based anomaly detection addresses part of this limitation by modeling dependencies among entities. Methods such as DOMI-NANT [9] and SL-GAD [27] use graph neural networks and selfsupervised objectives to combine attribute and structural information when detecting anomalies. However, graph anomaly detection is usually formulated for graph data rather than relational databases. A relational database contains typed tables, primary-foreign key relationships, event records, entity records, and schema-dependent roles. Treating such data as a generic graph can blur distinctions among entity types and relation types that are important for interpreting anomalies. Relational Deep Learning [11] and RelBench [20] address this representation issue by preserving multi-table structure and enabling learning directly over relational databases. This makes relational learning a natural foundation for relational anomaly detection.

The remaining gap is the anomaly objective. Relational learning frameworks are typically designed for supervised prediction, and while graph imbalance has been studied across the node, edge, and graph levels [24, 26, 26], they are not for ranking rare anomalous target entities under severe class imbalance. In contrast, anomaly detection requires scoring functions and evaluation protocols that prioritize retrieval of rare positives near the top of a ranked list. Our benchmark suite builds on this distinction: LANL provides a cybersecurity relational anomaly setting, while the Amazon and H&M relational databases from RelBench are adapted into unexpected churn anomaly detection tasks. The underlying RelBench databases are existing resources, but the anomaly-oriented task construction and unified ranking protocol are specific to this work.

## 2.3 From Symbolic Rules to Learned Anomaly Scoring

Symbolic and rule-based methods provide another piece of the problem. Association-rule mining [1], tree-derived rules [2], and interpretable decision sets [16] can expose human-readable regularities in structured data. Recent work such as ChatRule [18] and KnowGraph [28] further shows how logical rules and knowledgeguided reasoning can capture semantic patterns that are dificult to express through dense embeddings alone.

The limitation is that symbolic rules and learned relational representations are often treated as separate components. Rule-based methods may expose interpretable behavioral patterns, but they are commonly used only for explanation, filtering, or post-hoc reasoning rather than as part of the representation-learning process itself. Conversely, graph and relational neural models can learn expressive embeddings from relational structure, but the resulting anomaly representations may be dificult to connect to explicit behavioral evidence or analyst-defined suspicious patterns. Thus, preserving relational structure alone is often insuficient for relational anomaly detection. A central challenge is how to incorporate symbolic behavioral evidence into learned relational representations while still supporting end-to-end anomaly-oriented learning.

## 2.4 Summary of Gaps and Our Contribution

Positioned against this landscape, RAD’s contribution is threefold. First, we formalize relational anomaly detection as a setting distinct from tabular and graph anomaly detection, and introduce a bench mark suite spanning a cybersecurity task with externally verified ground truth and two derived e-commerce tasks under a unified ranking protocol. Second, we identify the point at which symbolic evidence enters the model as a design decision: rule features are injected into target nodes before message passing, allowing them to shape learned representations rather than just adjust scores after the fact, and the ablation in Section 7 evaluates the choice directly against post-hoc fusion. Third, we report empirical findings that hold independent of RAD’s specific architecture: edge reconstruction is not uniformly beneficial, and LLM-assisted rule refinement functions well as a compaction and deduplication step rather than as just a source of new signal. With our contributions positioned against this landscape, we next formalize the mathematical foundations of our setting.

## 3 Preliminaries

We represent a relational database as a collection of typed tables, $\mathcal { D } = \{ T _ { 1 } , T _ { 2 } , . . . , T _ { m } \}$ . Each table $T _ { i }$ contains rows of a particular entity or event type, and the database schema S specifies table attributes, primary keys, foreign keys, and relationships among tables[20]. For example, rows may correspond to users, computers, authentication events, customers, reviews, transactions, or articles depending on the dataset.

A relational database can be converted to a graph, $\mathcal { G } = ( V , E , \tau , \rho )$ where each row becomes a node $v \in V ,$ each primary-foreign key relationship becomes an edge $e \in E , \tau ( o )$ gives the node type of $v ,$ and $\rho ( e )$ gives the edge type of �. This construction follows the relational database-to-graph view used in relational deep learning, where tables and schema relationships define typed nodes and typed edges for message passing [11, 20]. Unlike a flattened feature matrix, this representation preserves entity identity, table type, and schema-defined relationships.

Let $T _ { \mathrm { t a r g e t } }$ be a designated target table whose rows are the objects to be scored. The corresponding target nodes are denoted $V _ { \mathrm { t a r g e t } } \subset$ � . The target type is dataset-dependent. In LANL, target nodes are authentication events, while in the Amazon and H&M settings, target nodes are users or customers [13, 20]. Each target node $v ~ \in ~ V _ { \mathrm { t a r g e t } }$ may have a binary label $y _ { v } ~ \in ~ \{ 0 , 1 \}$ , where $y _ { v } ~ = ~ 1$ indicates an anomaly and $y _ { v } = 0$ indicates a normal target node. Labels may be available only for a subset of target nodes.

## 4 Problem Definition

We formally introduce our proposed problem setting as follows:

Definition 4.1 (Relational Anomaly Detection). Given a relational database D with schema S, a designated target table $T _ { \mathrm { t a r g e t } } ,$ a heterogeneous graph representation $\mathcal { G } = ( V , E , \tau )$ , target nodes $V _ { \mathrm { t a r g e t } } \subset V$ , and anomaly labels for a training subset $V _ { \mathrm { t r a i n } } \subset V _ { \mathrm { t a r g e t } } .$ the goal is to learn an anomaly scoring function, $s : V _ { \mathrm { t a r g e t } } \to \mathbb { R }$ such that anomalous target nodes receive higher scores than normal target nodes. For labeled target nodes $v _ { i } , v _ { j } \in V _ { \mathrm { t r a i n } }$ with $y _ { i } = 1$ and $y _ { j } = 0 ;$ the desired ranking behavior is $s ( v _ { i } ) > s ( v _ { j } )$

The learned scoring function is evaluated on a held-out test set $V _ { \mathrm { t e s t } } \subset V _ { \mathrm { t a r g e t } : }$ where $V _ { \mathrm { t r a i n } } \cap V _ { \mathrm { t e s t } } = \emptyset$ . Evaluation is performed by ranking nodes in $V _ { \mathrm { t e s t } }$ according to �(�).

When timestamps are available, we impose temporal consistency. For a target node � observed at time $t _ { v } ,$ the scoring function may only use graph structure and node attributes available at or before $t _ { v } \colon$ $s ( v ) = f ( \mathcal { G } _ { \leq t _ { v } } ,  { \boldsymbol } X _ { \leq t _ { v } } )$ . This prevents future information from leak ing into the representation of earlier target nodes. The constraint is especially important for event-level cybersecurity detection and churn-style prediction tasks.

Unlike ordinary supervised classification or tabular anomaly detection, the goal here is a ranking that depends on typed relational context in $\mathcal { G } .$ not a calibrated per-instance label.

## 5 RAD: Rule-Augmented Relational Anomaly Detection

RAD is a rule-augmented relational anomaly detector for multitable databases. The model has four stages, shown in Figure 1. First, the relational database is converted into a heterogeneous graph that preserves entity types and schema-defined relationships. Second, RAD constructs an auxiliary flattened feature view for the target entities and mines symbolic rules from this view. Third, the selected rules are evaluated on each target entity and injected as binary rule features into the corresponding target nodes. Fourth, a heterogeneous graph autoencoder learns rule-aware relational representations and produces anomaly scores using reconstructionbased and ranking-based objectives.

The design separates rule discovery from relational representation learning. The flattened view is used only to discover interpretable rule predicates. The final anomaly detector operates on the heterogeneous graph, where rules are injected into target nodes before message passing. As a result, symbolic evidence can shape node embeddings and interact with relational context, rather than being added only as a post-hoc adjustment to final anomaly scores.

## 5.1 Relational Graph Construction

Given the heterogeneous graph representation from Section 3, RAD constructs typed nodes from table rows and typed edges from primary-foreign key relationships. Each target row corresponds to a target node, and neighboring nodes correspond to related entities or events connected through the database schema. The exact target and relation types are dataset-dependent and are described in Section 6.

Each node has an initial feature vector. Since diferent node types may have diferent attributes and feature dimensions, RAD uses type-specific input projections before message passing. Graph construction respects the temporal constraint of Section 4

The goal of this step is not to flatten the database into one feature matrix, but to preserve typed relational structure so that anomaly scores can depend on both node attributes and relational context.

## 5.2 Rule Mining and Validation

RAD mines symbolic rules from an auxiliary flattened feature view of the target entities. This flattened view is used only for rule discovery, rule evaluation, and tabular baselines; it does not replace the heterogeneous graph used by RAD for relational representation learning. Rule mining’s purpose is to extract compact behavioral predicates that can later be injected into target-node features.

Given the flattened training view and the available anomaly labels, RAD trains a random forest classifier after removing identifier, timestamp, label, and unavailable future-information columns. Numeric features are coerced to numeric values, missing values are imputed, and categorical features are encoded before training. Each root-to-leaf path in the trained forest defines a candidate rule. A path is converted into a conjunctive predicate by tracing the feature-threshold decisions from the leaf back to the root. Each split contributes one condition of the form $x _ { j } \leq c \ \mathrm { o r } \ x _ { j } > c ,$ , producing a candidate predicate defined as:

$$
q ( v ) = c _ { 1 } ( v ) \wedge c _ { 2 } ( v ) \wedge \cdot \cdot \cdot \wedge c _ { m } ( v ) .
$$

![](images/d1b14fae038053f8260d15ed32e09104605a93728b45cab461ac4c0b5bf6c03f.jpg)  
Figure 1: Overview of RAD. A database is represented by a flattened view for rule mining and a heterogeneous graph for representation learning. Random-forest paths are refined into predicates, converted into binary features, and injected into nodes before heterogeneous graph encoding. The model produces anomaly scores and ranks target nodes.

Following standard rule mining notions of support, confidence, and lift [1], we adapt these quantities to candidate anomaly predicates over target nodes. Let � denote a labeled split, let $v \in S$ be a target node, let $y _ { v } \in \{ 0 , 1 \}$ be its anomaly label, and let $q ( v ) \in \{ 0 , 1 \}$ indicate whether candidate predicate � covers node �. The indicator 1[·] equals one when its argument is true and zero otherwise.

The support of predicate � on split � is

$$
{ \mathrm { s u p p o r t } } _ { S } ( q ) = \sum _ { v \in S } 1 [ q ( v ) = 1 ] ,
$$

which counts the number of nodes covered by the predicate. The confidence of � is the ratio of covered nodes that are anomalous:

$$
{ \mathrm { c o n f i d e n c e } } _ { S } ( q ) = { \frac { \sum _ { v \in S } \mathbf { 1 } [ q ( v ) = 1 \land y _ { v } = 1 ] } { \sum _ { v \in S } \mathbf { 1 } [ q ( v ) = 1 ] } } .
$$

When the denominator is zero, confidence is defined as zero.

We also compute lift by comparing predicate confidence against the anomaly prevalence in the same split:

$$
\operatorname * { l i f t } _ { S } ( q ) = \frac { \mathrm { c o n f i d e n c e } _ { S } ( q ) } { \pi _ { S } } , \qquad \pi _ { S } = \frac { 1 } { | S | } \sum _ { v \in S } 1 [ y _ { v } = 1 ] .
$$

Lift measures whether a predicate covers anomalies at a higher rate than expected from the base anomaly prevalence.

Candidate predicates are filtered before being used as features. Predicates that reference identifiers, label fields, unavailable columns, or future information are removed. Predicates that fail parsing, fail execution, or do not satisfy the minimum coverage requirements are also discarded. The remaining predicates are ranked using support, confidence, and lift computed only from the training or validation data designated for rule selection. Duplicate predicates are removed.

From the retained candidate predicates, RAD samples a smaller set of seed rules for optional LLM-assisted refinement. The LLM is used only as a constrained rule-proposal mechanism. Given the flattened feature schema, feature statistics, compact positive examples from the training data, and sampled random-forest seed rules, it proposes SQL WHERE predicates that prune, tighten, combine, or generalize the seed predicates. The LLM does not assign anomaly labels, produce anomaly scores, introduce external attributes, or access test labels.

Generated predicates are validated before use. Each generated predicate is executed against the flattened rule-selection view and rescored using the same support, confidence, and lift definitions above. Invalid predicates and predicates that fail the required cov erage checks are discarded. The highest-ranked valid predicates are retained as the final rule set.

The final rule set is frozen after training and validation. At test time, RAD only evaluates these fixed predicates on held-out target entities. No rule mining, rule selection, threshold relaxation, or predicate updating uses test labels.

Each retained predicate is converted into a binary rule feature. For rule $k ,$ the rule feature for target entity � is

$$
r _ { k } ( v ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f } } v { \mathrm { ~ s a t i s f i e s ~ r u l e } } k , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

The full rule vector is

$$
r ( \boldsymbol { v } ) = [ r _ { 1 } ( \boldsymbol { v } ) , r _ { 2 } ( \boldsymbol { v } ) , \ldots , r _ { K } ( \boldsymbol { v } ) ] .
$$

The resulting binary rule matrix is aligned with the corresponding target nodes and injected into the heterogeneous graph as described in Section 5.3.

## 5.3 Rule-Augmented Node Features

After rule mining and validation, RAD evaluates the final rule set on each target entity. This produces a binary rule vector �(�) for each target node � $\in V _ { \mathrm { t a r g e t } } .$ Each entry indicates whether the target entity satisfies one selected rule.

RAD injects this rule vector directly into the target node features before message passing. For each target node $v ,$ the augmented feature vector is

$$
\tilde { x } _ { v } = [ x _ { v } \lVert r ( v ) ] ,
$$

where $x _ { v }$ is the original target-node feature vector and ∥ denotes concatenation. Non-target nodes keep their original feature vectors.

This direct injection is a key design choice. Because rule features are added before graph encoding, rule evidence can influence message passing and representation learning. A target node’s embedding can therefore reflect not only its own attributes and relational neighborhood, but also the symbolic rules it satisfies. This difers from post-hoc rule fusion, where rule features are added after the graph model has already produced anomaly scores. Post-hoc fusion can adjust final scores, but it cannot change the representations learned by the graph encoder. We evaluate this design choice in the direct-injection versus post-hoc-fusion ablation in Section 8.

## 5.4 Heterogeneous Graph Encoder

RAD uses a heterogeneous graph encoder to learn rule-aware node representations from the augmented relational graph. Because different node types may have diferent feature spaces, each node type first uses a type-specific input projection. For target nodes, the input is the rule-augmented feature vector $\tilde { x } _ { v } ;$ for non-target nodes, the input is the original feature vector $x _ { v } .$

After projection, RAD applies relation-aware message passing over the typed edges of the relational graph. Messages are aggregated separately for each relation type and then combined to update the node representation. This preserves schema information by allowing messages arriving through diferent database relationships to contribute separately before being merged.

In our implementation, the encoder is a two-layer heterogeneous GraphSAGE model with mean aggregation [12]. After the final message-passing layer, each target node � has an embedding $z _ { v }$ that encodes its original attributes, injected rule features, and typed relational neighborhood. These embeddings are used by the reconstruction and ranking objectives described next.

Formally, for node � of type � (�), the encoder computes

$$
\begin{array} { r l } & { \boldsymbol { h } _ { \boldsymbol { v } } ^ { ( 0 ) } = W _ { \tau ( \boldsymbol { v } ) } \tilde { \boldsymbol { x } } _ { \boldsymbol { v } } + b _ { \tau ( \boldsymbol { v } ) } , } \\ & { \boldsymbol { m } _ { \boldsymbol { v } } ^ { ( l , r ) } = \frac { 1 } { | \mathcal { N } _ { r } ( \boldsymbol { v } ) | } \displaystyle \sum _ { \boldsymbol { u } \in \mathcal { N } _ { r } ( \boldsymbol { v } ) } h _ { \boldsymbol { u } } ^ { ( l - 1 ) } , } \\ & { \boldsymbol { h } _ { \boldsymbol { v } } ^ { ( l ) } = \sigma \Bigg ( W _ { s \mathrm { e f f } } ^ { ( l ) } h _ { \boldsymbol { v } } ^ { ( l - 1 ) } + \displaystyle \sum _ { r \in \mathcal { R } ( \boldsymbol { v } ) } W _ { r } ^ { ( l ) } \boldsymbol { m } _ { \boldsymbol { v } } ^ { ( l , r ) } \Bigg ) , } \\ & { \boldsymbol { z } _ { \boldsymbol { v } } = h _ { \boldsymbol { v } } ^ { ( L ) } , \qquad \boldsymbol { L } = 2 , } \end{array}
$$

where $N _ { r } ( \boldsymbol { v } )$ denotes the neighbors of � under relation $r , \mathcal { R } ( v )$ the relation types incident to $\tau ( v )$ , and $\tilde { x } _ { v }$ the rule-augmented feature vector from Section 4.3 for target nodes and $x _ { v }$ otherwise. We use GraphSAGE-style mean aggregation because it is inductive (required for scoring LANL’s sampled subgraphs at test time), stable under variable-degree sampled neighborhoods, and matches the RDL baseline encoder in Section 7.3, isolating the efect of rule injection from backbone choice.

## 5.5 Anomaly Scoring and Training Objective

RAD scores target nodes using a graph autoencoder objective with optional structure reconstruction and ranking-based anomaly supervision, following the general use of graph autoencoders to learn latent node representations and reconstruct graph structure [14].

5.5.1 Atribute Reconstruction. The attribute decoder attempts to reconstruct the rule-augmented target-node features $\tilde { x } _ { v }$ from the target-node embedding $z _ { v } ,$ which we formalize as:

$$
\hat { \tilde { x } } _ { v } = g _ { \phi } ( z _ { v } ) .
$$

The attribute reconstruction loss for target node � is

$$
\mathcal { L } _ { \mathrm { a t t r } } ( v ) = \left\| \hat { \tilde { x } } _ { v } - \tilde { x } _ { v } \right\| _ { 2 } ^ { 2 } .
$$

This term encourages the encoder to learn representations that preserve target-node attributes and rule features. The reconstruction error also provides an anomaly signal. Target nodes that do not follow common attribute, rule, or relational patterns should be harder to reconstruct.

5.5.2 Edge Reconstruction. RAD can also include an edge reconstruction term. For an edge $( u , v , r )$ of relation type �, the structure decoder predicts whether the edge exists from the node embeddings. This can be formalized as follows:

$$
\hat { a } _ { u v } ^ { \left( r \right) } = \sigma \left( z _ { u } ^ { \top } W _ { r } z _ { v } \right) ,
$$

where $W _ { r }$ is a relation-specific decoder parameter and $\hat { a } _ { u v } ^ { ( r ) }$ is the predicted probability that an edge of type � exists between nodes � and � in the relational graph.

The edge reconstruction loss is binary cross-entropy over observed edges and sampled non-edges:

$$
\mathcal { L } _ { \mathrm { s t r u c t } } = - \sum _ { \left( u , v , r \right) \in E ^ { + } \cup E ^ { - } } \left[ a _ { u v } ^ { \left( r \right) } \log \hat { a } _ { u v } ^ { \left( r \right) } + \left( 1 - a _ { u v } ^ { \left( r \right) } \right) \log \left( 1 - \hat { a } _ { u v } ^ { \left( r \right) } \right) \right] .
$$

Here, $E ^ { + }$ is the set of observed typed edges, $E ^ { - }$ is a set of sampled negative edges, and $a _ { u v } ^ { ( r ) } \in \{ 0 , 1 \}$ indicates whether the typed edge exists.

We treat edge reconstruction as optional because it may help when anomalousness is expressed through unusual links, but it may also overemphasize sampled graph topology. We therefore evaluate both BCE and no-BCE variants in Section 8. The no-BCE variant sets the edge-reconstruction weight to zero.

5.5.3 Ranking-Based Anomaly Supervision. Reconstruction alone is an indirect anomaly detection objective. A powerful autoencoder may reconstruct some anomalous nodes well, and the practical goal is to rank rare anomalies near the top rather than to produce calibrated binary predictions. For this reason, RAD uses pairwise ranking supervision when anomaly labels are available, following pairwise learning-to-rank objectives that encourage positive instances to score above negative instances [3].

Let � be the set of labeled anomalous target nodes and � be the set of presumed-normal target nodes. RAD samples pairs $( i , j ) \sim$ $P \times N$ and encourages anomalous nodes to receive higher anomaly scores than normal nodes:

$$
\mathcal { L } _ { \mathrm { r a n k } } = \mathbb { E } _ { ( i , j ) \sim P \times N } \left[ \mathrm { s o f t p l u s } ( s _ { j } - s _ { i } ) \right] .
$$

Here, $s _ { i }$ and $s _ { j }$ are anomaly scores for the anomalous and normal target nodes, respectively. In our implementation, the anomaly score is based on reconstruction error, optionally combined with the edge-reconstruction score when used.

The full training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { a t t r } } + \alpha \mathcal { L } _ { \mathrm { s t r u c t } } + \beta \mathcal { L } _ { \mathrm { r a n k } } , } \end{array}
$$

where � controls the contribution of edge reconstruction and $\beta$ controls the contribution of ranking supervision. In no-BCE ablations, $\alpha = 0 .$ This objective aligns RAD with anomaly ranking under class imbalance while still using reconstruction to learn rule-aware relational representations.

## 6 Relational Anomaly Detection Benchmark

We introduce a relational anomaly detection benchmark suite for evaluating anomaly ranking in multi-table databases. The suite contains one cybersecurity task based on the LANL Comprehensive, Multi-Source Cybersecurity Events dataset [13] and two ecommerce tasks derived from the Amazon and H&M relational databases in RelBench [20]. The underlying Amazon and H&M databases are not new; our contribution is to adapt them into unexpected-churn anomaly detection tasks and evaluate all three tasks under a unified relational anomaly detection protocol.

Table 1: Summary of the relational anomaly detection benchmark tasks. LANL is treated as a multi-table event database, while Amazon and H&M are derived from relational userbehavior databases and converted into unexpected churn anomaly detection tasks.
<table><tr><td>Task</td><td>Domain</td><td>Target</td><td></td><td></td><td></td><td>Tables Records Targets Anomaly</td></tr><tr><td>LANL</td><td>Cybersecurity Auth event</td><td></td><td>7</td><td>1.65B</td><td>1.64B</td><td>Red-team auth.</td></tr><tr><td>Amazon</td><td>E-commerce</td><td>Customer</td><td>3</td><td>15.0M</td><td>1.85M</td><td>Unexpected review churn</td></tr><tr><td>H&amp;M</td><td>Retail</td><td>Customer</td><td>3</td><td>16.7M</td><td>1.37M</td><td>Unexpected purchase churn</td></tr></table>

## 6.1 Benchmark Design Principles

A task is suitable for relational anomaly detection when anomalousness may depend on relational context rather than only on the target entity’s own attributes. We therefore select tasks that contain multiple schema-linked tables, define a target entity or event type, provide anomaly labels or a principled anomaly-label construction, contain rare positives suitable for ranking-based evaluation, and include temporal information when needed to prevent leakage. These criteria distinguish relational anomaly detection from flattened tabular anomaly detection: the target score may depend on linked entities, typed relationships, historical behavior, and multi-hop schema-dependent context.

Table 1 summarizes the benchmark tasks, target entities, database scale, and anomaly definitions. For Amazon and H&M, the default unexpected-churn anomaly rate is 1.5%, with 0.5% and 3.0% used in the prevalence-sensitivity sweep.

## 6.2 LANL Cybersecurity Event Detection

The LANL cybersecurity dataset contains enterprise authentication events, process execution logs, DNS activity, network flows, and red-team attack labels [13]. We use authentication events as target entities and red-team authentication events as anomalies. This task represents event-level anomaly detection in a relational cybersecurity database. LANL is well suited for relational anomaly detection because suspicious authentication behavior often depends on context: a login may appear normal as a row but become suspicious when linked to unusual users, machines, process activity, etc. As the full dataset is large, we evaluate sampled relational subgraphs centered on authentication events. Anchor events are sampled across the full time range, with a small number of red-team events included to ensure positives. For each anchor, we retrieve linked users, source and destination computers, and nearby process, DNS, and flows when available.

## 6.3 Unexpected User-Churn Detection

We derive two unexpected-churn tasks from relational user-behavior databases. Unlike ordinary churn prediction, these tasks treat churn as anomalous only when it occurs without a preceding activity decline that would make the churn expected. This distinction is important since predictable disengagement is not necessarily anomalous. The resulting tasks ask whether relational context helps identify users whose behavior changes in surprising ways across linked reviews, purchases, products, transactions, and item metadata.

6.3.1 Amazon Review Churn. The Amazon task scores customers whose reviewing behavior ends unexpectedly. Each customer is connected to review events and reviewed products, allowing the relational graph to represent not only how often a customer reviews, but also what kinds of products they review and how their activity relates to product-level context. The auxiliary flattened view for Amazon summarizes each customer’s own review activity, review counts, rating history, product diversity as a count, and review-history span, and does not incorporate attributes of the reviewed products themselves. Mined rules for this task are therefore restricted to one-hop customer-level aggregates; multi-hop product context reaches the model only through message passing over the customer-review-product graph, not through rule features.

6.3.2 H&M Purchase Churn. The H&M task evaluates unexpected disappearance from purchase activity. Customers are linked to transaction records and articles, with article attributes providing item-level context for purchase behavior. This task difers from Amazon because activity is expressed through purchases rather than reviews, and because product metadata can help distinguish ordinary inactivity from unusual changes in shopping behavior. The flattened view summarizes recent transactions, price history, article diversity, and transaction-history span, while the relational graph preserves customer-transaction-article links.

6.3.3 Unexpected-Churn Label Construction. For Amazon and H&M, we convert churn labels into unexpected-churn anomaly labels by removing churn cases that are explained by a strong prior activity decline.

A user is therefore anomalous only if they churn and their previous activity does not already indicate an expected disengagement pattern. The unexpected-churn label is derived from the original RelBench [20] churn label via the activity-decline filter below, not an independent construction. The filter removes only those churned users whose own prior activity history already signals disengagement; it does not introduce a new labeling criterion unrelated to the underlying data. The prevalence sweep in Section 7 varies the filtering threshold across 0.5%, 1.5%, and 3.0% and shows that RAD’s ranking advantage over the supervised RDL baseline is stable across this range, which is the evidence that the default 1.5% threshold was not selected to favor a particular result.

Let $y _ { u } ^ { \mathrm { c h u r n } } \in \{ 0 , 1 \}$ denote the original churn label for user �. We compute a recent activity-change score $g _ { u }$ and a baseline activity level $\mu _ { u }$ from the user’s activity history before the prediction time. A user is marked as having an expected activity decline when

$$
D ( u ) = 1 \left[ g _ { u } \leq - \theta ( \mu _ { u } ) \right] ,
$$

![](images/42b12e8f4b2a97670bace3640c4c101acd690262dcff86d8adffcacd8b87e1b2.jpg)  
Figure 2: Unexpected-churn filtering. Churned users with an expected prior activity decline are removed; remaining churned users are treated as unexpected-churn anomalies.

where

$$
\theta ( \mu _ { u } ) = \lambda _ { 0 } + \log _ { \lambda _ { 1 } } ( \mu _ { u } + 1 ) ( 1 + \lambda _ { 2 } \mu _ { u } ) .
$$

The threshold adapts to the user’s baseline activity level, so users with diferent activity histories are not judged using the same fixed decline cutof. In our experiments, we use $\lambda _ { 0 } = 0 . 0 5 , \lambda _ { 1 } = 1 0 . 0$ , and $\lambda _ { 2 } = 0 . 0 2$

The filtered anomaly label is

$$
\begin{array} { r } { y _ { u } ^ { \mathrm { a n o m } } = y _ { u } ^ { \mathrm { c h u r n } } ( 1 - D ( u ) ) . } \end{array}
$$

Thus, a user is labeled anomalous only if the user churned and was not already identified as having an expected activity decline. Churned users with an expected prior decline are removed from the positive anomaly class. After filtering, training positives are downsampled to a target anomaly prevalence of 1.5%, and the validation set is downsampled to match the resulting training prevalence. Evaluation is performed on held-out target users using ranking metrics.

6.3.4 Prevalence Sweep for Label Sensitivity. The unexpected-churn label depends on the strictness of the filtering threshold used to remove expected churn cases. To test whether the benchmark is robust to this design choice, we perform a prevalence sweep over the target anomaly labeling rate. After constructing candidate unexpected-churn positives, we vary the retained positive rate to produce target prevalences of 0.5%, 1.5%, and 3.0%. The 1.5% task is used as the default benchmark configuration, while the 0.5% and 3.0% tasks evaluate sensitivity under stricter and more permissive anomaly definitions.

For each prevalence level, we preserve the same train/validation/test protocol and compare RAD against the supervised RDL baseline using AUROC and AUPRC. The sweep changes only the anomalylabel filtering rate; the relational graph construction, feature views, model architecture, and evaluation protocol are otherwise held fixed. This isolates the efect of label prevalence from changes in model design.

## 6.4 Dataset-Specific Flattened Views and Rule Mining

Although RAD operates on heterogeneous relational graphs, rule mining and tabular baselines require an auxiliary flattened view for each target entity. Table 2 summarizes the dataset-specific flattened feature views along with random-forest and predicate-filtering settings that instantiate Section 5.2’s generic rule-mining procedure.

Table 2: Feature views and random-forest rule-mining settings by dataset.
<table><tr><td>Setting</td><td>LANL</td><td>Amazon</td><td>H&amp;M</td></tr><tr><td>Feature groups</td><td>Auth attrs.; user-pair novelty; recent auth., DNS, flow, process</td><td>Recent reviews; rating history; product diversity; review-history span</td><td>Recent purchases; price history; article diversity; activity trend/level</td></tr><tr><td>Training cap</td><td>500K</td><td>200K</td><td>200K</td></tr><tr><td>Forest size</td><td>1200</td><td>200</td><td>200</td></tr><tr><td>Depth / leaf</td><td>None / 1</td><td>10 / 5</td><td>10 / 5</td></tr><tr><td>Class weight</td><td>Balanced</td><td>Balanced</td><td>Balanced</td></tr><tr><td>Path extraction</td><td>First 200</td><td>First 100</td><td>First 100</td></tr><tr><td>Leaf filter</td><td>≥8, p+ ≥ .50</td><td>≥5, p+ ≥ .50</td><td>≥5, p+ ≥ .50</td></tr><tr><td>Selection split</td><td>Validation</td><td>Train sample</td><td>Train sample</td></tr><tr><td>Rule filter</td><td>Supp. ≥3, conf. ≥.02</td><td>Supp. ≥1</td><td>Supp. ≥1</td></tr><tr><td>Seed / pool</td><td>30 / 2000</td><td>30/ 2000</td><td>30/ 2000</td></tr></table>

## 7 Experiments

The experiments evaluate whether RAD improves relational anomaly detection through relational representation learning, rule aug mentation, and anomaly-oriented supervision, returning to the five research questions introduced in Section 1.

## 7.1 Experimental Protocol

We evaluate RAD on the benchmark from Section 6, using the pipeline described in Section 5. Each setting defines a target entity type (authentication events for LANL, customers for Amazon and H&M); the model produces a continuous anomaly score per target node, evaluated only on held-out entities. Flattened rows are aligned with graph target nodes by target identifier so all methods share the same target population and splits. The supervised MLP uses ground-truth training labels and is evaluated on the same held-out test entities; test labels are used only for evaluation. Results are reported over ten random seeds as mean ± standard deviation, using AUROC, AUPRC, and precision@k, with AUPRC and precision@k emphasized because anomaly detection is highly imbalanced and practical systems prioritize retrieving true anomalies near the top of a ranked list [4].

## 7.2 Implementation Details

RAD uses a two-layer relation-aware GraphSAGE encoder with mean aggregation, hidden dimension 128, Adam optimization, learning rate 0.001, MSE attribute reconstruction, optional BCE edge reconstruction, and early stopping on validation AUPRC. The struc ture weight �, ranking weight �, and model hyperparameters are selected using Optuna on the validation split, with validation AUPRC as the optimization objective. The selected configuration is then fixed for held-out test evaluation.

The supervised MLP operates on the flattened target-entity features and is trained using binary cross-entropy on ground-truth training labels. For LANL, MLP hyperparameters are selected using the validation split. For Amazon and H&M, the MLP uses the default, untuned hyperparameters. In all cases, the resulting model is evaluated on the held-out test split over ten random seeds.

For experiments with edge reconstruction, the structure reconstruction weight � controls the contribution of the binary crossentropy edge reconstruction term; no-BCE ablations set � to zero.

For LANL, rule mining uses the validation-based filtering and support thresholds described in Table 2.

## 7.3 Baselines

We compare against flattened tabular anomaly detectors, including Isolation Forest [17], Deep SVDD [21], autoencoder reconstruction [22], AE+Rank, and DRL [25]. We also compare against relational deep learning baselines that operate directly on the heterogeneous graph without rule augmentation [11]. Finally, we include a supervised MLP trained on the flattened target-entity features as a fully supervised reference baseline. This provides a reference for how the anomaly detection methods perform relative to a model trained directly using ground-truth anomaly labels. These baselines separate the efects of tabular scoring, relational representation learning, rule augmentation, ranking supervision, and full label supervision. Table 3 summarizes the baselines.

Table 3: Baselines compared in our experiments, categorized by input representation and supervision signal.
<table><tr><td>Model</td><td>Tab.</td><td>Graph</td><td>Rank</td><td>Sup.</td></tr><tr><td>Isolation Forest [17]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Deep SVDD [21]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>AE [22]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>AE + Rank</td><td>√</td><td>x</td><td>√</td><td>x</td></tr><tr><td>DRL [25]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>RDL GNN [11]</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Supervised MLP</td><td>√</td><td>x</td><td>x</td><td>√</td></tr></table>

“Tab.” indicates flattened tabular input, “Graph” indicates heterogeneous relational graph input, “Rank” indicates pairwise anomaly-ranking supervision, and “Sup.” indicates direct supervision using ground-truth anomaly labels.

## 7.4 Evaluation Metrics

All methods produce a continuous anomaly score for each held-out target entity, where larger scores indicate greater anomalousness. We report AUROC for global ranking quality, AUPRC for precisionrecall performance on the anomalous class, and precision@k for the fraction of true anomalies among the top-� ranked targets, with � set to the number of positives in the test set. We emphasize AUPRC and precision@k because precision-recall evaluation is more informative than ROC analysis under severe class imbalance, and because practical anomaly detection systems often inspect only the top-ranked alerts [4, 8].

AUROC is the probability that a uniformly drawn positive target node receives a higher score than a uniformly drawn negative one. AUPRC is the area under the precision-recall curve, summed across recall levels. Precision@k is the fraction of true positives among the top-� ranked targets, with � set to the number of positives in the evaluated split.

## 8 Results

The results evaluate whether RAD improves relational anomaly detection through relational graph modeling, rule augmentation, and anomaly-oriented supervision. We focus primarily on AUPRC and precision@k because the benchmark tasks are highly imbalanced and require retrieving rare anomalies near the top of a ranked list.

Table 4: Main results across datasets over 10 seeds. Results are reported as mean ± standard deviation. Avg. rank columns report the average rank per metric across datasets; lower is better.
<table><tr><td></td><td colspan="3">LANL</td><td colspan="3">Amazon</td><td colspan="3">H&amp;M</td><td colspan="3">Avg. Rank</td></tr><tr><td>Model</td><td>AUROC</td><td>AUPRC</td><td>P@k</td><td>AUROC</td><td>AUPRC</td><td>P@k</td><td>AUROC</td><td>AUPRC</td><td>P@k</td><td>AUROC</td><td>AUPRC</td><td>P@k</td></tr><tr><td>RAD (rules, no BCE)</td><td> $\mathbf { 0 . 9 9 6 \pm 0 . 0 0 1 }$ </td><td> $0 . 6 5 9 \pm 0 . 0 5 3$ </td><td> $0 . 6 0 1 \pm 0 . 0 1 2$ </td><td>0.693 ±0.010</td><td> $\mathbf { 0 . 0 4 3 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 5 0 \pm 0 . 0 0 7$ </td><td> $0 . 8 2 8 \pm 0 . 0 7 1$ </td><td> $\mathbf { 0 . 1 4 3 \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 1 2 9 \pm 0 . 0 2 2 }$ </td><td>2.0</td><td>1.2</td><td>2.0</td></tr><tr><td>RAD (rules, BCE)</td><td> $0 . 8 6 0 \pm 0 . 1 5 5$ </td><td>0.557 ±0.029</td><td> $0 . 5 7 2 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td>0.695 ±0.010</td><td> $\mathbf { 0 . 0 4 3 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 5 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 7 8 7 \pm 0 . 0 7 4$ </td><td> $0 . 1 1 0 \pm 0 . 0 2 0$ </td><td> $0 . 1 2 5 \pm 0 . 0 2 2$ </td><td>3.3</td><td>1.8</td><td>2.3</td></tr><tr><td>RDL weighted [11]</td><td> $0 . 5 5 3 \pm 0 . 1 5 8$ </td><td> $0 . 0 0 7 \pm 0 . 0 1 4$ </td><td> $0 . 0 2 1 \pm 0 . 0 6 4$ </td><td> $\mathbf { 0 . 7 2 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 0$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 4$ </td><td> $0 . 7 0 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 0 3 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 4 \pm 0 . 0 1 3$ </td><td>4.7</td><td>4.8</td><td>3.7</td></tr><tr><td>RDL unweighted [11]</td><td> $0 . 6 8 5 \pm 0 . 2 0 5$ </td><td> $0 . 0 1 5 \pm 0 . 0 2 8$ </td><td> $0 . 0 2 6 \pm 0 . 0 7 8$ </td><td> $0 . 7 1 3 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 5$ </td><td> $0 . 6 9 3 \pm 0 . 0 1 9$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 4$ </td><td> $0 . 0 6 4 \pm 0 . 0 1 6$ </td><td>4.3</td><td>3.8</td><td>3.3</td></tr><tr><td>AE+Rank [22]</td><td>0.943 ±0.035</td><td> $0 . 5 2 5 \pm 0 . 0 7 8$ </td><td>0.540 ±0.065</td><td>0.518 ±0.014</td><td>0.016 ±0.001</td><td> $0 . 0 1 5 \pm 0 . 0 0 3$ </td><td> $0 . 4 5 6 \pm 0 . 0 2 4$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 0$ </td><td>0.005 ±0.004</td><td>5.2</td><td>5.2</td><td>5.7</td></tr><tr><td>DRL [25]</td><td> $0 . 6 7 3 \pm 0 . 1 5 9$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 4$ </td><td> $0 . 4 6 3 \pm 0 . 0 2 2$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 0$ </td><td> $0 . 4 5 6 \pm 0 . 0 1 2$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 9$ </td><td>7.5</td><td>8.2</td><td>7.2</td></tr><tr><td> $\mathrm { A E } \left[ 2 2 \right]$ </td><td> $0 . 8 8 8 \pm 0 . 0 4 6$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 4 7 4 \pm 0 . 0 1 3$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 9$ </td><td> $0 . 4 0 8 \pm 0 . 0 2 0$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 7 \pm 0 . 0 0 5$ </td><td>6.7</td><td>7.3</td><td>7.8</td></tr><tr><td> $\mathrm { D e e p S V D D \left[ 2 1 \right] }$ </td><td> $0 . 7 3 6 \pm 0 . 0 7 3$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 3$ </td><td> $0 . 4 5 8 \pm 0 . 0 3 1$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 4 1 2 \pm 0 . 0 2 3$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 3$ </td><td>7.7</td><td>8.7</td><td>8.8</td></tr><tr><td>Isolation Forest [17]</td><td> $0 . 6 4 4 \pm 0 . 0 1 8$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 4 5 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 0$ </td><td> $0 . 4 0 2 \pm 0 . 0 1 4$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 5$ </td><td>9.7</td><td>9.3</td><td>9.5</td></tr><tr><td>Supervised MLP</td><td> $0 . 9 6 5 \pm 0 . 0 0 3$ </td><td> $0 . 0 5 0 \pm 0 . 0 1 1$ </td><td>0.093 ±0.035</td><td> $0 . 6 5 9 \pm 0 . 0 2 4$ </td><td> $0 . 0 2 7 \pm 0 . 0 0 3$ </td><td> $0 . 0 4 1 \pm 0 . 0 0 9$ </td><td> $0 . 6 4 2 \pm 0 . 0 2 8$ </td><td> $0 . 0 2 7 \pm 0 . 0 0 5$ </td><td> $0 . 0 3 9 \pm 0 . 0 2 3$ </td><td>4.0</td><td>4.7</td><td>4.7</td></tr></table>

## 8.1 RQ1: Overall Relational Anomaly Detection Performance

Table 4 reports the main results across LANL, Amazon, and H&M. RAD achieves the strongest average AUPRC rank across the benchmark, with the clearest gains on LANL. On LANL, RAD substantially outperforms flattened tabular detectors and relational baselines on AUPRC and precision@k, indicating that relational structure and rule augmentation are especially useful for retrieving rare suspicious authentication events.

On Amazon, the supervised RDL baselines achieve stronger AU-ROC and precision@k than RAD, but RAD achieves higher AUPRC. This suggests that RDL provides strong global ranking on this task, while RAD improves precision-recall behavior over the anomalous class. On H&M, RAD achieves the best performance across AUROC, AUPRC, and precision@k. Overall, these results support the cen tral motivation of relational anomaly detection: anomaly ranking benefits from modeling both relational context and rule-derived behavioral evidence, especially under severe class imbalance.

## 8.2 RQ2: Efect of Rule Augmentation

Figure 3 evaluates the contribution of rule augmentation and rule construction choices. Rule-augmented RAD variants generally improve anomaly retrieval relative to variants without rule features, especially on AUPRC and precision@k. This indicates that symbolic predicates provide useful behavioral evidence beyond the original target-node attributes and relational graph structure.

The comparison between rule-construction variants shows that compact predicate-based rules are more useful than raw decisionfeature injection. Random-forest path rules already provide meaningful anomaly signal, while LLM-assisted refinement helps filter, tighten, and compact the candidate predicates before they are injected into the graph. These results support the design choice of converting mined rules into explicit binary rule features rather than treating the random forest only as a black-box auxiliary model.

## 8.3 RQ3: Efect of Structure Reconstruction

The efect of explicit graph structure reconstruction is mixed and dataset-dependent. In Table 4, the no-BCE RAD variant is strongest overall: it achieves higher average AUROC, AUPRC, and precision@k ranks than the BCE variant. On LANL, disabling BCE gives substantially stronger AUROC, AUPRC, and precision@k, suggesting that edge reconstruction may overemphasize artifacts of the sampled cybersecurity subgraphs. On Amazon, BCE and no-BCE perform similarly, with BCE slightly higher on AUROC and precision@k but essentially tied on AUPRC. On H&M, the no-BCE variant is stronger across all three metrics.

![](images/0b4a0ab9c907af79e12298aebae3bad23da3b3b6cc54bd136e07e86c8e6f9186.jpg)  
Figure 3: Rule-construction and structure-reconstruction ablations across datasets over 10 seeds. Bars show means and error bars show standard deviations. The y-axis is broken to separate AUROC from smaller AUPRC and P@k values.

These results suggest that edge reconstruction should be treated as an optional auxiliary objective rather than a uniformly beneficial component. It may help when anomalousness is expressed through relational topology, but it can also distract the model from anomaly ranking when the sampled graph structure is noisy or only indirectly related to the anomaly label.

## 8.4 RQ4: Efect of LLM-Assisted Rule Refinement

LLM-assisted rule refinement provides a useful but not uniformly dominant improvement over random-forest-derived rules alone. The largest benefits appear on LANL, where refined rules improve anomaly retrieval compared with less compact rule representations. On Amazon and H&M, refined rules remain competitive with random-forest sampled rules, suggesting that the LLM stage should be viewed primarily as a rule-filtering and rule-compaction step rather than as a replacement for random-forest rule mining.

Importantly, the LLM does not assign labels or produce anomaly scores. Its role is limited to proposing candidate predicates that are subsequently validated, filtered, and scored using the same training/validation rule-selection procedure described in Section 5.2. This keeps the anomaly detection pipeline grounded in the observed relational data and prevents test-label leakage.

![](images/f0583a9e07df34757bb1befe6a649629ff04cb38e9f4d9a0183048807c3cf0a3.jpg)  
(a) Amazon

![](images/b7ac17264fd9bd1a4b0e184721745d90f1a698ce6459663028a9fc6296c2ff62.jpg)  
(b) H&M  
Figure 4: Label-prevalence sensitivity for unexpected-churn detection. AUROC uses the left axis and AUPRC the right axis for target prevalences of 0.5%, 1.5%, and 3.0%; counts show retained positives.

## 8.5 RQ5: Sensitivity to Unexpected-Churn Prevalence

Figure 4 evaluates whether the unexpected-churn results depend on the target anomaly prevalence used in the label filter. Across 0.5%, 1.5%, and 3.0% prevalence settings, the main performance trends remain broadly stable. On Amazon, RAD achieves its strongest AUROC at the default 1.5% setting and its strongest AUPRC at 3.0%, while supervised RDL improves more gradually as additional positives are retained. On H&M, increasing prevalence generally improves performance, and RAD remains competitive and often stronger than supervised RDL across the sweep.

These results support using 1.5% as the default benchmark configuration while showing that the conclusions are not driven by a single arbitrary prevalence threshold. The prevalence sweep also shows how performance changes as the anomaly definition becomes more permissive, while the relative comparison between RAD and the supervised relational baseline remains stable enough to support the benchmark design.

## 8.6 Limitations

RAD’s rule mining is bounded by the auxiliary flattened view and requires labeled data for both the random-forest stage and support/confidence/lift ranking, so it does not extend as-is to unlabeled settings. Amazon and H&M results are reported on a held-out partition of the validation split rather than a leaderboard test set, since RelBench withholds true test labels for these tasks. The unexpected churn label itself is a derived construction; Section 8.5’s prevalence sweep shows conclusions are stable across filter thresholds but the filter formula remains a design choice. LANL evaluation uses sampled subgraphs around anchor events rather than the full authentication log, and several tabular/relational baselines show substantial seed variance on LANL.

## 9 Conclusion

This paper introduced RAD, a rule-augmented framework for anomaly detection in relational databases. RAD is motivated by the observation that many anomalies are not isolated row-level outliers, but deviations whose meaning depends on relationships among entities, events, and historical behavior. Flattening relational data into a single feature matrix can obscure these dependencies, while relational representation learning alone does not provide explicit behavioral evidence or an anomaly-specific scoring objective.

RAD combines heterogeneous graph representation learning, mined symbolic rule features, reconstruction-based scoring, and ranking-based anomaly supervision. The relational graph preserves database structure, while the rule-mining component provides compact behavioral predicates derived from flattened summaries. By injecting rule features into target nodes before message passing, RAD allows symbolic evidence to influence learned relational rep resentations rather than only post-hoc anomaly scores.

Across LANL cybersecurity event detection, Amazon review churn, and H&M purchase churn, RAD improves anomaly retrieval under class imbalance, especially on AUPRC. The results suggest that relational anomaly detection is best treated as a joint problem of representation, scoring, and behavioral evidence.

Several promising directions remain for future work. Scaling relational anomaly detection to larger databases and longer temporal histories will require more eficient graph construction and inference, consistent with recent work identifying scalability and temporal modeling as key challenges for relational deep learning [10, 15]. RAD also could be extended toward unsupervised rule discovery and rules defined directly over complex multi-hop relational structure rather than flattened summaries. Finally, richer relational architectures [6] could provide even stronger backbones, while symbolic rules could be further leveraged to provide interpretable explanations for detected anomalies.

## Acknowledgments

This work was supported by the National Security Agency (NSA) under grant number H98230-23-C-0279, and by the National Science Foundation (NSF) under grant numbers IIS-2239881, ECCS-2325417, IIS-2524380, and DGE-2622415.

## GenAI Usage Disclosure

Generative AI tools were used to assist with code development and manuscript writing. All experimental design, implementation choices, results, analysis, and final manuscript decisions were reviewed by the authors.

## References

[1] Rakesh Agrawal and Ramakrishnan Srikant. 1994. Fast algorithms for mining as sociation rules in large databases, VLDB’94: Proceedings of the 20th Internationa Conference on Very Large Data Bases. San Francisco, CA, USA (1994), 487–499. [2] Leo Breiman. 2001. Random forests. Machine learning 45 (2001), 5–32.

[3] Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. 2005. Learning to rank using gradient descent. In Proceedings ofthe 22nd international conference on Machine learning. 89–96.

[4] Guilherme O. Campos, Arthur Zimek, Jörg Sander, Ricardo J. Campello, Barbora Micenková, Erich Schubert, Ira Assent, and Michael E. Houle. 2016. On the evaluation ofunsupervised outlier detection: measures, datasets, and an empirica study. Data Min. Knowl. Discov. 30, 4 (July 2016), 891–927. doi:10.1007/s10618- 015-0444-8

[5] Varun Chandola, Arindam Banerjee, and Vipin Kumar. 2009. Anomaly detection: A survey. ACM Comput. Surv. 41, 3, Article 15 (July 2009), 58 pages. doi:10.1145/ 1541880.1541882

[6] Tianlang Chen, Charilaos Kanatsoulis, and Jure Leskovec. 2025. Relgnn: Composite message passing for relational deep learning. arXiv preprint arXiv:2502.06784 (2025).

[7] Austin Coursey, Junyi Ji, Marcos Quinones-Grueiro, William Barbour, Yuhang Zhang, Tyler Derr, Gautam Biswas, and Daniel B Work. 2024. FT-AED: Benchmark dataset for early freeway trafic anomalous event detection. Advances in Neural Information Processing Systems 37 (2024), 15526–15549.

[8] Jesse Davis and Mark Goadrich. 2006. The relationship between Precision-Recall and ROC curves. In Proceedings of the 23rd international conference on Machine learning. 233–240.

[9] Kaize Ding, Jundong Li, Rohit Bhanushali, and Huan Liu. 2019. Deep anomaly detection on attributed networks. In Proceedings of the 2019 SIAM international conference on data mining. SIAM, 594–602.

[10] Vijay Prakash Dwivedi, Charilaos Kanatsoulis, Shenyang Huang, and Jure Leskovec. 2025. Relational deep learning: Challenges, foundations and nextgeneration architectures. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 5999–6009.

[11] Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan, Joshua Robinson, Rex Ying, Jiaxuan You, and Jure Leskovec. 2024. Position: Relational Deep Learning - Graph Representation Learning on Relational Databases. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235), Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (Eds.). PMLR, 13592–13607. https://proceedings.mlr.press/v235/fey24a.html

[12] William L. Hamilton, Rex Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In Advances in Neural Information Processing Systems.

[13] Alexander D. Kent. 2015. Comprehensive, Multi-Source Cyber-Security Events. Los Alamos National Laboratory. doi:10.17021/1179829

[14] Thomas N Kipf and Max Welling. 2016. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308 (2016).

[15] Veronica Lachi, Antonio Longa, Beatrice Bevilacqua, Bruno Lepri, Andrea Passerini, and Bruno Ribeiro. 2025. Boosting relational deep learning with pretrained tabular models. arXiv preprint arXiv:2504.04934 (2025).

[16] Himabindu Lakkaraju, Stephen H Bach, and Jure Leskovec. 2016. Interpretable decision sets: A joint framework for description and prediction. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 1675–1684.

[17] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation forest. In 2008 eighth ieee international conference on data mining. IEEE, 413–422.

[18] Linhao Luo, Jiaxin Ju, Bo Xiong, Yuan-Fang Li, Gholamreza Hafari, and Shirui Pan. 2024. ChatRule: Mining Logical Rules with Large Language Models for Knowledge Graph Reasoning. arXiv:2309.01538 [cs.AI] https://arxiv.org/abs/2309.01538

[19] Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel. 2021. Deep Learning for Anomaly Detection: A Review. ACM Comput. Surv. 54, 2, Article 38 (March 2021), 38 pages. doi:10.1145/3439950

[20] Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, Xinwei He, and Jure Leskovec. 2024. RelBench: A Benchmark for Deep Learning on Relational Databases. Advances in Neural Information Processing Systems 37 (2024). https://arxiv.org/abs/2407.20060

[21] Lukas Ruf, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. 2018. Deep One-Class Classification. In Proceedings ofthe 35th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 80), Jennifer Dy and Andreas Krause (Eds.). PMLR, 4393–4402. https://proceedings.mlr.press/ v80/ruf18a.html

[22] Mayu Sakurada and Takehisa Yairi. 2014. Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction (MLSDA’14). Association for Computing Machinery, New York, NY, USA, 4–11. doi:10.1145/2689746.2689747

[23] Bernhard Schölkopf, Robert C Williamson, Alex Smola, John Shawe-Taylor, and John Platt. 1999. Support Vector Method for Novelty Detection. In Advances in Neural Information Processing Systems, S. Solla, T. Leen, and K. Müller (Eds.), Vol. 12. MIT Press. https://proceedings.neurips.cc/paper\_files/paper/1999/file/ 8725fb777f25776fa9076e44fcfd776-Paper.pdf

[24] Yu Wang, Yuying Zhao, Neil Shah, and Tyler Derr. 2022. Imbalanced graph classification via graph-of-graph neural networks. In Proceedings ofthe 31st ACM international conference on information & knowledge management. 2067–2076.

[25] Hangting Ye, He Zhao, Wei Fan, Mingyuan Zhou, Dan dan Guo, and Yi Chang. 2025. DRL: Decomposed Representation Learning for Tabular Anomaly Detection. In The Thirteenth International Conference on Learning Representations. https: //openreview.net/forum?id=CJnceDksRd

[26] Tianxiang Zhao, Xiang Zhang, and Suhang Wang. 2021. Graphsmote: Imbalanced node classification on graphs with graph neural networks. In Proceedings of the 14th ACM international conference on web search and data mining. 833–841.

[27] Yu Zheng, Ming Jin, Yixin Liu, Lianhua Chi, Khoa T. Phan, and Yi-Ping Phoebe Chen. 2023. Generative and Contrastive Self-Supervised Learning for Graph Anomaly Detection. IEEE Transactions on Knowledge and Data Engineering 35, 12 (2023), 12220–12233. doi:10.1109/TKDE.2021.3119326

[28] Andy Zhou, Xiaojun Xu, Ramesh Raghunathan, Alok Lal, Xinze Guan, Bin Yu, and Bo Li. 2024. KnowGraph: Knowledge-Enabled Anomaly Detection via Logical Reasoning on Graph Data. arXiv:2410.08390 [cs.CR] https://arxiv.org/abs/2410. 08390