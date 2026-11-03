# A Common Language for Dynamic Graphs: Interaction Tokenization Enables Foundation Models

Fuyuan Ma<sup>∗</sup> mafy23@mails.jlu.edu.cn School of Artificial Intelligence, Jilin University Changchun, Jilin, China

Xin Zhang   
xzhang25@mails.jlu.edu.cn   
School of Artificial Intelligence, Jilin   
University   
Changchun, Jilin, China

Yili Wang<sup>∗</sup> wangyili@jlu.edu.cn School of Artificial Intelligence, Jilin University Changchun, Jilin, China

Ying Wang<sup>†</sup>   
wangying2010@jlu.edu.cn   
School of Computer Science and   
Technology, Jilin University   
Changchun, Jilin, China

Xin He hexin20@mails.jlu.edu.cn School of Artificial Intelligence, Jilin University Changchun, Jilin, China

Xin Wang   
xinwang@jlu.edu.cn   
School of Artificial Intelligence, Jilin   
University   
Changchun, Jilin, China

## Abstract

Dynamic graphs capture evolving interactions in Web systems, including online communication, user engagement, and recommendation. Graph foundation models aim to reuse predictive capabilities across networks, but extending this capability to dynamic graphs faces two challenges. First, node identities are graph-specific, and similar interaction patterns can appear at diferent temporal and activity scales, making histories dificult to represent consistently across networks. Second, the predictive roles of historical patterns vary across interaction dynamics, limiting transfer from a narrow range of training processes. In this paper, we show that a common interaction language, combined with learning across diverse interaction processes, ofers a basis for dynamic graph foundation modeling. We propose CL-DGFM, a Common Language Dynamic Graph Foundation Model for temporal link prediction. Its Dynamic Interaction Tokenizer encodes interaction memory, historical connectivity, and ordered destination transitions into calibrated tabular inputs, making dynamic histories accessible to existing predictors without an additional trainable graph encoder. Its Dynamic Interaction Prior trains a pretrained tabular foundation predictor across diverse synthetic dynamics. Labeled historical context guides prediction, with a Contextual Value Adapter modulating attention values. The model transfers to real graphs without target-graph parameter updates. Experiments on TGB and TGB-Seq demonstrate competitive performance. Further comparisons show that the com mon language supports basic machine learning and neural models, and enables direct use of frozen tabular foundation models on dynamic graphs.

## CCS Concepts

• Theory of computation → Dynamic graph algorithms.

## Keywords

Dynamic Graph Foundation Models, Temporal Link Prediction, Graph Tokenization

![](images/c3d713ede09beadaa311c4ea49dd9ce04068b343328586db51e82498de082e62.jpg)  
Figure 1: Motivation for CL-DGFM. (a) Graph-Domain Foundation Modeling couples encoding with prediction. (b) Foundation Model Adaptation in static settings loses temporal distinctions. (c) An envisioned DGFM uses a common interaction language for shared prediction across dynamic graphs.

## 1 Introduction

Dynamic graphs capture the evolving interactions underlying Web systems, including online communication, user engagement, and recommendations. Temporal link prediction uses past interactions to predict which entities will interact at a given time. Existing temporal graph models are typically trained separately for each dataset, requiring new training to serve networks with diferent entities and activity patterns. Yet these networks can share recurring interaction patterns, including repeated contacts, indirect connections, and ordered destination visits. Such regularities raise the possibility of learning predictive knowledge that can be reused across networks. This leads to a fundamental question: can we build a dynamic graph foundation model (DGFM) that predicts across networks without dataset-specific retraining?

Research on static graph foundation models has explored how predictive capabilities can be reused across graphs, ofering a starting point for extending foundation modeling to dynamic settings. Existing approaches follow two broad routes, illustrated in Fig. 1. ➀ Graph-Domain Foundation Modeling (Fig. 1a): Dedicated graph architectures learn transferable prediction rules through graph pretraining. NodePFN combines local message passing with contextual prediction and trains on synthetic graph priors [2]. Its graph encoding is integrated into the predictive architecture, and its static node-level formulation does not model how event histories shape subsequent interactions. ➁ Foundation Model Adaptation (Fig. 1b): G2T-FM, TFMLinker, and GraphPFN extend existing foundation predictors through graph-derived inputs, learned link encoders, or graph aggregation modules [5, 6, 20]. These methods demonstrate the value of reusing pretrained predictive knowledge, but their static formulations leave interaction timing and order outside the representation. Early temporal approaches such as TGPM and DyGFM explore transferable patterns in evolving networks through graph-specific pretraining [24, 40]. A central design question is how to express dynamic histories in a common language and learn predictive relationships that a shared model can apply across diferent interaction processes.

A common interaction language must describe historical behavior independently of its network. However, node identities are graph-specific, and raw interaction counts and time intervals depend on network activity. This raises Challenge ❶: How can interaction histories be made comparable across dynamic graphs? For example, repeated contacts within minutes in a highly active network and within days in a less active network may both reflect a persistent relationship, despite their diferent time intervals. Directly comparing these intervals can obscure the similarity between the two histories. Conversely, equal interaction counts can conceal diferences in when contacts occurred, how endpoints were indirectly connected, and in what order destinations were visited. Removing node identities or summarizing histories through counts alone therefore does not establish a common language. Cross-graph comparability requires accounting for diferences in temporal and activity scales while preserving distinctions in interaction timing, historical connectivity, and destination order.

A common interaction language makes histories comparable across graphs, but a shared model must still learn how to use these histories for prediction. Diferent networks may rely on diferent combinations of repeated contacts, historical connectivity, and destination transitions. Learning from a limited range of interaction processes can consequently favor particular historical signals, restricting transfer to networks where other signals matter. This raises Challenge ❷: How can predictive capabilities transfer across networks with diferent interaction dynamics? For example, repeated contacts may be informative in networks dominated by persistent relationships, whereas destination transitions may be more useful when interactions follow sequential routines. Strong performance under one process therefore does not ensure reliable prediction under another. Transfer requires learning to use multiple sources of historical evidence without becoming dependent on a particular interaction process. This calls for training tasks that cover diverse interaction mechanisms and their combinations, together with historical examples that help relate these signals to observed outcomes in the target network. These challenges motivate the envisioned DGFM in Fig. 1c: a common interaction language makes histories comparable across networks, while a shared predictor uses historical context to support transfer across diferent interaction dynamics without target-graph parameter updates.

To realize this vision, we propose CL-DGFM, a Common Language Dynamic Graph Foundation Model for temporal link prediction. For Challenge ❶, we introduce a Dynamic Interaction Tokenizer that encodes interaction memory, historical connectivity, and ordered destination transitions through shared historical descriptors. Temporal and activity calibration support comparability across networks with diferent identities and scales. These descriptors form feature vectors for conventional machine learning and neural models. Stacking them into tables further enables pretrained tabular foundation models to perform dynamic graph prediction directly, without an additional trainable graph encoder. For Challenge ❷, CL-DGFM combines dynamic-prior learning with contextual prediction. A Dynamic Interaction Prior generates synthetic sequences with varying preferences, recurrence, transitions, and structural dependencies, training an existing foundation predictor across diferent relationships between histories and future outcomes. Labeled historical context guides prediction on each target graph, with a Contextual Value Adapter adjusting attention values according to the context. Model parameters remain fixed across target graphs. Experiments on TGB and TGB-Seq validate this approach, while comparisons across predictor families demonstrate the common language’s broader use beyond the complete CL-DGFM framework.

Our contributions are summarized below:

• We introduce CL-DGFM, which reuses a pretrained tabular foundation predictor for temporal link prediction through a common interaction language and synthetic training, without target-graph parameter updates.

• We develop a Dynamic Interaction Tokenizer that makes dynamic histories accessible to general-purpose predictors. A Dynamic Interaction Prior and contextual prediction with a Contextual Value Adapter support learning and applying predictive relationships across interaction dynamics.

• Experiments on TGB and TGB-Seq show that CL-DGFM achieves the highest mean MRR on 10 of 13 datasets. Further analyses demonstrate that the common language enables direct use of frozen tabular foundation models and competitive prediction with basic machine learning and neural models, while revealing the roles of historical components, training diversity, and context.

## 2 Related Work

## 2.1 Graph Foundation Models

Graph foundation models transfer predictive knowledge across heterogeneous graphs and tasks. Graph-domain foundation modeling learns shared prediction rules through graph-oriented pretraining. One for All and GIT unify prediction tasks, while Uni-Graph and GOFA integrate language representations with graph modeling [13, 18, 21, 33]. NodePFN combines message passing and context attention through synthetic-prior training [2]. They couple transferable representations with graph-specific architectures. Foundation model adaptation supplies graph information to pretrained predictors. G2T-FM constructs neighborhood and structural features [5], GraphPFN adds graph aggregation to LimiX and trains on synthetic graphs [6], and TFMLinker uses topology-aware encoders and context prototypes for static link prediction [20]. Their static formulations omit interaction timing and order. CL-DGFM extends this route to dynamic graphs via a parameter-free interaction tokenizer and a dynamic interaction prior.

## 2.2 Temporal Link Prediction

Memory and neighborhood modeling extracts historical evidence through node states or temporal aggregation. TGN maintains event-driven memories, while TGAT, GraphMixer, and DyGFormer use temporal attention, mixing, or neighbor co-occurrence [3, 29, 34, 38]. Structural and sequential pattern modeling captures more explicit dependencies. CAWN and TPNet encode temporal walks, while CRAFT emphasizes frequency patterns and histories conditioned on the destination [22, 32, 36]. These models typically require dataset-specific training. For transfer across evolving networks, TGPM pretrains interaction-patch representations, DyGFM separates semantic and temporal representations for multi-domain pretraining, and TIDES combines a frozen time-series foundation encoder with a GNN [4, 24, 40]. Our common interaction language instead supports diferent predictor families, with CL-DGFM using historical context without target-graph parameter updates.

## 2.3 In-Context Learning and Synthetic Priors

Prior-data fitted networks learn to predict query labels from labeled context using tasks sampled from synthetic priors [25]. TabPFN and TabICL develop this approach for tabular prediction [14, 28], while Mitra explores mixtures of synthetic priors [42]. TabDPT instead pretrains on real tables [23], and LimiX studies a shared conditional prediction interface across structured-data tasks [41]. Domain transfer also requires suitable input representations [7, 8]. TabPFN-TS formulates forecasting as tabular regression using temporal features, while RDBLearn constructs parameter-free relational representations [15, 35]. For dynamic graphs, CL-DGFM couples a shared representation of memory, connectivity, and event order with synthetic priors that vary their predictive roles.

## 3 Problem Formulation

We study how a graph foundation model can perform temporal link prediction across diferent dynamic graphs. We first define the prediction task and then specify the expected behavior of a shared model on previously unseen networks.

Temporal link prediction. A dynamic graph represents interactions between entities over time. We write it as $G = ( \mathcal { V } ^ { G } , \mathcal { E } ^ { G } )$ where $\mathcal { V } ^ { G }$ is the node set and $\mathcal { E } ^ { G } = \{ ( u _ { i } , v _ { i } , t _ { i } ) \} _ { i = 1 } ^ { N _ { G } }$ is the event sequence. Each event records an interaction from source $u _ { i }$ to destination �<sub>�</sub> at time �<sub>�</sub>. Interactions may repeat, and diferent events may share a timestamp. For a prediction at time �, the available history contains only earlier events:

$$
\mathcal { H } _ { < t } ^ { G } = \{ ( u _ { i } , v _ { i } , t _ { i } ) \in \mathcal { E } ^ { G } \mid t _ { i } < t \} ,\tag{1}
$$

subject to the history-visibility rules of the evaluation protocol.

A query $q = ( u , t , C _ { q } )$ specifies a source node, a query time, and a candidate destination list $C _ { q } = ( v _ { q , 1 } , \ldots , v _ { q , K _ { q } } )$ . The task is to rank these candidates using the available history, so that the observed destination receives a higher score than alternative destinations. Candidates may be previous interaction partners or nodes that have never interacted directly with the source. Given the source � and query time �, the task focuses on predicting the interaction destination from $C _ { q }$

Foundation modeling across dynamic graphs. We seek a graph foundation model that reuses predictive knowledge across networks with diferent entities, structures, and interaction dynamics. Given the historical graph $G _ { < t } = ( \mathcal { V } ^ { G } , \mathcal { H } _ { < t } ^ { G } )$ and a query $q ,$ the model produces candidate scores:

$$
\mathbf { \boldsymbol { s } } _ { q } ^ { G } = \mathrm { G F M } _ { \theta } ( G _ { < t } , q ) \in \mathbb { R } ^ { K _ { q } } ,\tag{2}
$$

where $s _ { q , j } ^ { G }$ is the score assigned to destination $v _ { q , j } .$ . Sorting these scores gives the predicted destination ranking. The same parameters � are used across target graphs, including previously unseen networks, without target-graph parameter updates. Prediction may use the target graph’s historical interactions available under the evaluation protocol, and no correspondence between node identities across graphs is required.

## 4 Method

In this section, we present CL-DGFM (Figure 2). We first construct a common interaction language, then generate training sequences with diferent history–outcome relationships, and finally train a contextual foundation predictor that uses historical context to predict on target graphs without parameter updates.

## 4.1 Dynamic Interaction Tokenizer

Reusing a predictor across dynamic graphs requires historical evidence with consistent meanings despite diferent node identities and activity scales. Our tokenizer describes each candidate (�, �, �) through three complementary components: interaction memory captures direct contacts, historical connectivity supplies indirect evidence, and ordered destination transitions capture sequential behavior. All components use only $\mathcal { H } _ { < t } ^ { G }$ , without encoding node identities. Their outputs are integrated and calibrated into shared numerical inputs for synthetic learning and real-graph prediction.

4.1.1 Interaction Memory(IM). Equal interaction counts can conceal diferent temporal distributions, while frequent contacts may reflect either pair-specific behavior or high destination activity. We therefore encode pair and destination histories at multiple temporal scales. At query time �, let $E _ { u v }$ contain timestamps of all previous $u  v$ interactions, and $I _ { v }$ those of retained recent events involving destination � in either direction. For either timestamp multiset $D ,$ we define a temporal memory summary using four positive, graph-calibrated scales $h \in { \mathcal { B } } ;$

$$
\mathbf { m } _ { D } ( t ) = \left[ \log \left( 1 + \sum _ { \tau \in D } e ^ { - ( t - \tau ) / h } \right) \right] _ { h \in \mathcal { B } } .\tag{3}
$$

Here, � is an event timestamp. Smaller scales emphasize recent interactions, while larger scales retain older evidence.

Applying this summary to $E _ { u v }$ and $I _ { v }$ captures pair-specific and destination activity. We combine them with supplementary summaries a(�, �, �) to retain frequency, recency, history availability, and relative pair activity:

$$
\begin{array} { r } { \mathbf { z } ^ { \mathrm { m e m } } ( u , v , t ) = \mathbf { m } _ { E _ { u v } } ( t ) \oplus \mathbf { m } _ { I _ { v } } ( t ) \oplus \mathbf { a } ( u , v , t ) , } \end{array}\tag{4}
$$

where ⊕ denotes concatenation. The vector $\mathbf { z } ^ { \mathrm { m e m } } \in \mathbb { R } ^ { 1 8 }$ contains eight memory features, three log-counts and three recencies for pair, source, and destination histories, a reverse interaction logcount, observed-pair and unseen-destination indicators, and one normalized memory feature. The latter divides the pair’s decayed count by one plus the destination’s decayed count at the same selected scale, before log compression.

![](images/e1495ab6e2b6d7aafd01e4c29aefcef2f8f412866b0343b5974450f1798f786b.jpg)  
Figure 2: Overview of CL-DGFM. (1) The Dynamic Interaction Tokenizer converts real and synthetic histories into a common tabular representation. (2) The Dynamic Interaction Prior generates diverse interaction dynamics for training. (3) Contextual Dynamic Graph Foundation Modeling reuses a pretrained tabular foundation predictor, with a Contextual Value Adapter modulating attention values from labeled history. The model learns on synthetic graphs and ranks candidates on real graphs through historical context, without target-graph parameter updates or an additional trainable graph encoder.

4.1.2 Historical Connectivity(HC). Without previous contact, pair memory cannot distinguish candidates by their direct histories. We therefore summarize indirect connections through historical paths, using temporal decay to emphasize recent evidence and activity weighting to discount highly active intermediaries. Let $\mathcal { P } _ { \ell } ( u , v , t )$ contain retained length-ℓ paths between � and �, treating past events as undirected connections without requiring chronological traversal. For path $\rho ,$ let $\bar { a } ( \rho )$ be its mean event age at �, and $w ( \rho )$ a unit weight or an inverse-activity weight for intermediaries or the destination. The connectivity vector is:

$$
\mathbf { z } ^ { \mathrm { c o n n } } ( u , v , t ) = \left[ \log \left( 1 + \sum _ { \rho \in \mathcal { P } _ { \ell } ( u , v , t ) } w ( \rho ) e ^ { - \bar { a } ( \rho ) / h } \right) \right] _ { ( \ell , w , h ) \in \Omega _ { P } } .\tag{5}
$$

Here, $\Omega _ { P }$ specifies path lengths, weights, and decay scales ℎ. With $\ell \in \{ 2 , 3 \} , \mathbf { z } ^ { \mathrm { c o n n } } ( u , v , t ) \in \mathbb { R } ^ { 7 }$ records six connectivity statistics for paths of lengths two and three and one three-step count normalized by destination activity through $w ( \rho )$

4.1.3 Ordered Destination Transitions(ODT). Static path connectivity ignores the sequential order in which destinations are visited.

We complement it with collaborative destination transitions observed across other sources, capturing which destinations typically follow �’s recent visits. We consider the destinations of $\dot { u ^ { \mathrm { ~ s ~ } } }$ latest � outgoing events and look for other sources that visited these destinations and subsequently visited candidate �. Each such pair of visits forms a transition occurrence, with both visits strictly before �. Let $O _ { L } ( u , v , t )$ collect retained occurrences, counted separately for each of these outgoing events.

Transitions are more informative when they relate to �’s more recent visits or occur earlier among another source’s subsequent visits. We encode these preferences in a weight $\omega ( e )$ for each occurrence �, optionally downweighting highly active sources. To favor closely spaced visits, we decay each contribution by the interval $\Delta ( e ) > $ 0 between the two visits, using a time scale $h > 0$ For the current query $( u , v , t )$ , we aggregate this evidence into a destination-transition score:

$$
T _ { L } ( \omega , h ) = \sum _ { e \in O _ { L } ( u , v , t ) } \omega ( e ) e ^ { - \Delta ( e ) / h } .\tag{6}
$$

Using $L \in \{ 1 , 3 \}$ captures evidence from the latest visit and a broader recent history. Let $\Omega _ { T }$ collect the settings $( L , \omega , h )$ , including destination-activity normalization through �. We log-compress the scores and append an indicator of immediate repetition, where $v _ { u } ^ { \mathrm { l a s t } }$ denotes $\boldsymbol { u ^ { \prime } s }$ latest destination:

$$
\begin{array} { r } { \mathbf { z } ^ { \mathrm { s e q } } ( u , v , t ) = \left[ \log \left( 1 + T _ { L } ( \omega , h ) \right) \right] _ { ( L , \omega , h ) \in \Omega _ { T } } \oplus \left[ \mathbf { 1 } \{ v = v _ { u } ^ { \mathrm { l a s t } } \} \right] . } \end{array}\tag{7}
$$

The vector $\mathbf { z } ^ { \mathrm { s e q } } ( u , v , t ) \ \in \ \mathbb { R } ^ { 6 }$ combines four transition statistics capturing reference recency, time gaps, and source activity with a destination-activity-normalized count and the latest-destination indicator. The indicator is zero without outgoing history.

Input integration and calibration. We concatenate the three complementary representations into a shared history descriptor:

$$
\begin{array} { r } { \mathbf { z } ( u , v , t ) = \mathbf { z } ^ { \mathrm { m e m } } ( u , v , t ) \oplus \mathbf { z } ^ { \mathrm { c o n n } } ( u , v , t ) \oplus \mathbf { z } ^ { \mathrm { s e q } } ( u , v , t ) \in \mathbb { R } ^ { 3 1 } . } \end{array}\tag{8}
$$

To limit feature magnitudes while retaining their relative ordering within each candidate list, we combine bounded values with candidate-wise ranks. For query $q$ at source � and time �, write $\mathbf { z } _ { q , j } = \mathbf { z } ( u , v _ { q , j } , t )$ for its $j \mathrm { t h }$ candidate among $K _ { q }$ candidates. Let $\mathbf { r } _ { q , j }$ contain its coordinatewise ascending ranks within this list, using average ranks for ties and rescaling into $( - 1 , 1 )$ . With compression scale $c _ { x } > 0 _ { ; }$ , the final candidate input is:

$$
\begin{array} { r } { \mathbf { x } _ { q , j } = \operatorname { t a n h } ( \mathbf { z } _ { q , j } / c _ { x } ) \oplus \mathbf { r } _ { q , j } \in \mathbb { R } ^ { 6 2 } . } \end{array}\tag{9}
$$

Stacking these vectors as rows gives the query input $X _ { q } \in \mathbb { R } ^ { K _ { q } \times p }$ where $ { p } = 6 2$ . The same label-free transformations construct historical context inputs, with ranks computed within each original candidate list before context-row selection. Time-scale calibration and numerical settings are specified in Appendix B.

## 4.2 Dynamic Interaction Prior

The tokenizer establishes shared historical descriptors, but their predictive efects vary across interaction processes. Repetition may reinforce future contact under persistent preferences or suppress it under repetition inhibition. Our Dynamic Interaction Prior varies these mechanisms so that the foundation predictor learns to interpret historical evidence using labeled context. We draw generator configurations $\xi$ from a prior Π, varying source activity and destination-selection mechanisms. These configurations remain hidden from the predictor.

4.2.1 Source Activity and Event Timing. Diferent activity patterns yield diferent amounts and temporal distributions of historical evidence. We first generate event times and sources using exponential Hawkes processes [12]. For each source �, the event rate $\lambda _ { u } ( t )$ combines baseline activity with exponentially decaying excitation from its past events, producing heterogeneous activity and bursts.

For event �, the total source rate governs the next event time $t _ { i } .$ $\mathrm { A t } ~ t _ { i } ,$ , we sample source $u _ { i }$ proportionally to rates based on events strictly before $t _ { i } ,$ , then sample destination $v _ { i }$ from the conditional distribution $\pi _ { \xi } ( { \bf \cdot } | u _ { i } , t _ { i } )$ defined next.

4.2.2 History-Dependent Destination Selection. We combine preference, recurrence, transition, and structural mechanisms into a destination-sampling distribution, allowing diferent historical signals and preferences to govern the next interaction. Preference scores ${ g } _ { \mathrm { p r e f } }$ capture source–destination afinity and destination popularity, allowing interactions between pairs with no previous contact. Recurrence scores $g _ { \mathrm { r e c } }$ instead sum exponentially decayed past $u \to v$ interactions, favoring repeated direct contacts.

The other two mechanisms condition on $\boldsymbol { u ^ { \prime } s }$ latest destination $v _ { u } ^ { \mathrm { l a s t } }$ <sup>t</sup>. Transition scores $g _ { \mathrm { s e q } }$ use a sampled distribution $Q ( v \mid a )$ specifying preferences for visiting � after destination �. Structural scores $g _ { \mathrm { s t r } }$ draw on a sampled set $S _ { u } ( t )$ of other sources that previously visited $v _ { u } ^ { \mathrm { l a s t } }$ . Let $C _ { w v } ( t )$ count past $w  v$ events and $n _ { w } ^ { \mathrm { o u t } } ( t )$ count all past outgoing events of $\boldsymbol { w } .$ These scores are:

$$
\begin{array} { l } { \displaystyle g _ { \mathrm { s e q } } ( u , v , t ) = Q ( v \mid v _ { u } ^ { \mathrm { l a s t } } ) , } \\ { \displaystyle g _ { \mathrm { s t r } } ( u , v , t ) = \sum _ { w \in S _ { u } ( t ) } \frac { C _ { w v } ( t ) } { 1 + n _ { w } ^ { \mathrm { o u t } } ( t ) } . } \end{array}\tag{10}
$$

Thus, closure favors destinations visited by these other sources while discounting highly active sources.

For � ∈ {pref, rec, seq, str}, let $\bar { g } _ { m }$ denote $g _ { m }$ normalized to sum to one over destinations. We mix these scores with nonnegative weights $\alpha _ { m }$ summing to one. To allow repeated contacts to become less likely despite recurrence, we penalize previous pair interactions with strength $\gamma \geq 0 ;$

$$
\pi _ { \xi } ( v \mid u , t ) \propto e ^ { - \gamma C _ { u v } ( t ) } \sum _ { m } \alpha _ { m } \bar { g } _ { m } ( u , v , t ) .\tag{11}
$$

We normalize over valid destinations and draw $\begin{array} { r } { \boldsymbol { v } _ { i } \ \sim \ \boldsymbol { \pi } \boldsymbol { \xi } ( { \cdot } \ { \mathrm { ~ \small ~ \left| ~ { ~  ~ } ~ } } ) }\right. \end{array}$ $u _ { i } , t _ { i } )$ , completing the event $\left( { { u } _ { i } } , { { v } _ { i } } , t _ { i } \right)$ . Repeating this procedure and updating the history after each event produces a sequence $\varepsilon .$ Varying the generator configuration yields sequences with diferent history–outcome relationships, from which Section 4.3 constructs training tasks. Definitions and variants of $g _ { \mathrm { p r e f } }$ and $g _ { \mathrm { r e c } ; }$ , sampling and fallback rules for $g _ { \mathrm { s e q } }$ and $g _ { \mathrm { s t r } }$ , and numerical details of $\pi _ { \xi }$ are provided in Appendix C.

## 4.3 Contextual Dynamic Graph Foundation Modeling

A common representation alone does not determine which historical patterns predict future interactions. We use labeled historical context to interpret the shared inputs from Section 4.1. Training tasks constructed from the sequences generated in Section 4.2 expose the predictor to diferent relationships between historical evidence and future interactions.

4.3.1 History-Conditioned Prediction. We build CL-DGFM on a pretrained TabICLv2 backbone [28], whose in-context predictor uses labeled historical examples to score query candidates. For earlier events, we pair observed destinations with sampled alternatives, labeled one and zero, respectively. Applying Section 4.1 yields a context table $X _ { C } ~ \in ~ \mathbb { R } ^ { n _ { C } \bar { \times } 6 2 }$ and labels $\mathbf { y } _ { C } \in \{ 0 , 1 \} ^ { n _ { C } }$ , where $n _ { C }$ is the number of retained rows. For query $q ,$ the same tokenizer produces $X _ { q } \in \mathbb { R } ^ { K _ { q } \times 6 2 }$ for its $K _ { q }$ candidates. Each row uses only history preceding its event time.

Historical features can have diferent predictive meanings across tasks. We introduce a Contextual Value Adapter that uses labeled context to adjust the content transmitted by context rows. Its inputs are the existing feature values in $X _ { C }$ , their labels $\mathbf { y } _ { C }$ , and the backbone’s context-row representations. Statistics of positive and negative context rows guide memory-scale combinations and gate a low-rank value correction. Let $\bar { H } _ { C }$ contain the context representations after label embedding and normalization, and let $Q , K _ { C }$ , and $V _ { C }$ denote the first attention layer’s projected queries, context keys, and context values. The adapter modifies attention as follows:

$$
\begin{array} { r } { \Delta V _ { C } = \mathrm { V a l u e A d a p t e r } ( \bar { H } _ { C } ; X _ { C } , \mathbf { y } _ { C } ) , } \\ { O = \mathrm { A t t n } ( Q , K _ { C } , V _ { C } + \Delta V _ { C } ) . \qquad } \end{array}\tag{12}
$$

Here, $\Delta V _ { C }$ is the correction in projected value space. The backbone’s multi-head attention operator Atn uses queries from context and query rows, with keys and values only from context rows.

The attention output � continues through the backbone, whose scoring head maps the resulting query representations to candidate scores. Let $F _ { \theta }$ denote the complete predictor, with � collecting the trainable backbone and adapter parameters. We obtain the ranking vector defined in Eq. (2):

$$
\begin{array} { r } { { \bf s } _ { q } ^ { G } = F _ { \theta } ( X _ { C } , { \bf y } _ { C } , X _ { q } ) \in \mathbb { R } ^ { K _ { q } } . } \end{array}\tag{13}
$$

Its �th entry scores candidate $v _ { q , j } ,$ , with larger values indicating stronger predictions. The training objective below uses these scores. Context selection and adapter details are in Appendix D.

4.3.2 Dynamic-Prior Learning and Cross-Graph Inference. From a sequence E generated in Section 4.2, we construct a prediction task $\mathcal { T }$ using earlier events as labeled context and later events as queries, following the input construction above. Let $\mathcal { D } _ { \Pi }$ denote the task distribution induced by sequence generation and this construction. Variation across these sequences encourages the predictor to learn how context determines the predictive roles of historical features.

Starting from the pretrained backbone, we freeze its input encoders and jointly train the in-context predictor and Contextual Value Adapter. For a task $\mathcal { T } \sim \mathcal { D } _ { \Pi }$ , let � be its underlying graph, $\scriptstyle { \mathcal { I } } _ { Q }$ its query set, and $j _ { q } ^ { \star }$ the observed destination’s candidate index. Using the scores from Eq. (13), we minimize a listwise ranking loss:

$$
\mathcal { L } ( \theta ) = - \mathbb { E } \mathcal { T } \sim \mathcal { D } _ { \Pi } \left[ \frac { 1 } { \left| \boldsymbol { \mathcal { I } } _ { Q } \right| } \sum _ { q \in \boldsymbol { \mathcal { I } } _ { Q } } \log \frac { \exp ( s _ { q , j _ { q } ^ { \star } } ^ { G } ) } { \sum _ { j = 1 } ^ { K _ { q } } \exp ( s _ { q , j } ^ { G } ) } \right] .\tag{14}
$$

Here, $s _ { q , j } ^ { G }$ is the �th entry of $\mathbf { \Delta } \mathbf { s } _ { q } ^ { G } .$ . The objective favors the observed destination over alternative candidates, using interaction outcomes rather than generator parameters as supervision.

On a real graph $G ,$ we construct labeled context from its training interval and represent each later query using history available before its timestamp. The same predictor produces $\mathbf { \Delta } \mathbf { s } _ { q } ^ { G }$ for candidate ranking, ordering candidates by score and selecting the top-ranked destination as the predicted link target. All learned parameters remain fixed, while the graph’s labeled context conditions both value modulation and in-context prediction. Task sampling, optimization, and inference details are provided in Appendix D.

## 5 Experiments

Our experiments address three research questions: RQ1: How efectively does CL-DGFM predict temporal links across graphs without target-graph parameter updates? RQ2: How efective is the Dynamic Interaction Tokenizer as a common representation across graphs and predictors? RQ3: How do interaction diversity during training and historical context support cross-graph transfer?

## 5.1 Experimental Setup

We compare with 16 temporal link prediction baselines. Memory and neighborhood modeling covers EdgeBank [26], DyRep [30],

Table 1: Performance on TGB (100 × MRR; higher is better). Best and second-best results per column are bold red and underlined, respectively.
<table><tr><td>Method</td><td>Wiki</td><td>Review</td><td>Coin</td><td>Flight</td><td>Comment</td></tr><tr><td>CAWN</td><td> $7 1 . 1 0 _ { \pm 0 . 6 0 }$ </td><td> $1 9 . 3 0 _ { \pm 0 . 1 0 }$ </td><td></td><td></td><td></td></tr><tr><td>CTAN</td><td> $6 6 . 8 0 _ { \pm 0 . 7 0 }$ </td><td> $4 0 . 5 0 _ { \pm 0 . 4 0 }$ </td><td> $7 4 . 8 0 _ { \pm 0 . 4 0 }$ </td><td></td><td> $6 7 . 1 0 _ { \pm 6 . 7 0 }$ </td></tr><tr><td>DyGFormer</td><td> $7 9 . 8 0 _ { \pm 0 . 4 0 }$ </td><td> $2 2 . 4 0 _ { \pm 1 . 5 0 }$ </td><td> $7 5 . 2 0 _ { \pm 0 . 4 0 }$ </td><td></td><td> $6 7 . 0 0 _ { \pm 0 . 1 0 }$ </td></tr><tr><td>DyRep</td><td> $5 . 0 0 _ { \pm 1 . 7 0 }$ </td><td> $2 2 . 0 0 _ { \pm 3 . 0 0 }$ </td><td> $4 5 . 2 0 _ { \pm 4 . 6 0 }$ </td><td> $5 5 . 6 0 _ { \pm 1 . 4 0 }$ </td><td> $2 8 . 9 0 _ { \pm 3 . 3 0 }$ </td></tr><tr><td>EdgeBank</td><td> $5 7 . 1 0$ </td><td> $2 . 5 0$ </td><td>58.00</td><td>38.70</td><td>14.90</td></tr><tr><td>GraphMixer</td><td> $1 1 . 8 0 _ { \pm 0 . 2 0 }$ </td><td> $\underline { { 5 2 . 1 0 } } _ { \pm 1 . 5 0 }$ </td><td></td><td></td><td></td></tr><tr><td>HyperEvent</td><td> $8 1 . 0 0 _ { \pm 0 . 2 0 }$ </td><td></td><td> $7 7 . 3 0 _ { \pm 0 . 2 0 }$ </td><td> $8 7 . 7 0 _ { \pm 0 . 3 0 }$ </td><td> $7 6 . 0 0 _ { \pm 0 . 2 0 }$ </td></tr><tr><td>TGAT</td><td> $1 4 . 1 0 _ { \pm 0 . 7 0 }$ </td><td> $3 5 . 5 0 _ { \pm 1 . 2 0 }$ </td><td></td><td></td><td></td></tr><tr><td>TGN</td><td> $3 9 . 6 0 _ { \pm 6 . 0 0 }$ </td><td> $3 4 . 9 0 _ { \pm 2 . 0 0 }$ </td><td> $5 8 . 6 0 _ { \pm 3 . 7 0 }$ </td><td> $7 0 . 5 0 _ { \pm 2 . 0 0 }$ </td><td> $3 7 . 9 0 _ { \pm 2 . 1 0 }$ </td></tr><tr><td>TNCN</td><td> $7 1 . 8 0 _ { \pm 0 . 1 0 }$ </td><td> $3 7 . 7 0 _ { \pm 1 . 0 0 }$ </td><td> $7 6 . 2 0 _ { \pm 0 . 4 0 }$ </td><td> $8 2 . 0 0 _ { \pm 0 . 4 0 }$ </td><td> $6 9 . 7 0 _ { \pm 0 . 6 0 }$ </td></tr><tr><td>TPNet</td><td> $8 2 . 7 0 _ { \pm 0 . 1 0 }$ </td><td></td><td> $\underline { { 8 3 . 2 0 } } _ { \pm 0 . 1 0 }$ </td><td> $\underline { { 8 8 . 4 0 } } _ { \pm 0 . 1 0 }$ </td><td> $\underline { { 8 2 . 5 0 } } _ { \pm 0 . 6 0 }$ </td></tr><tr><td>CL-DGFM</td><td> $\underline { { 8 2 . 2 4 } } _ { \pm 0 . 5 1 }$ </td><td> $5 2 . 8 9 _ { \pm 0 . 2 0 }$ </td><td> $8 4 . 9 5 _ { \pm 0 . 0 6 }$ </td><td> $9 0 . 6 8 _ { \pm 0 . 0 5 }$ </td><td> $9 3 . 2 8 _ { \pm 0 . 1 6 }$ </td></tr></table>

JODIE [19], TGN [29], TGAT [34], TCL [31], GraphMixer [3], DyG-Former [38], CTAN [11], and TSNN [1], which use edge memorization, evolving node states, or temporal neighborhood aggregation for prediction. Structural and sequential pattern modeling covers CAWN [32], TNCN [43], TPNet [22], HyperEvent [9], CRAFT [36], and TAMI [39], which explicitly represent temporal connectivity and dependencies within interaction histories.

To assess predictor compatibility, we additionally compare ten predictors using the same tokenizer. Machine learning models include logistic regression, random forest, CatBoost [27], and Light-GBM [17]. Deep learning models include MLP, ResNet, and FT-Transformer [10]. Pretrained foundation models include TabI-CLv2 [28], TabPFNv2 [14], and LimiX-2M [41]. Baseline sources and settings are provided in Appendix A.

Dataset. We evaluate on 5 datasets from TGB [16] and 8 from TGB-Seq [37], covering online interactions, recommendation, financial transactions, transportation, citations, and hyperlinks.

Evaluation. We evaluate on complete oficial test sets with candidates from the benchmark and report 100 × MRR. Trained exclusively on synthetic interaction streams, CL-DGFM uses the same parameters across all target graphs, conditioning on strictly earlier interactions and fixed labeled context from each graph’s training split without using native node or edge attributes. Learned temporal baselines are trained and validated separately on each graph. We report the mean and sample standard deviation of CL-DGFM over three training seeds. Dataset, evaluation, and implementation details are provided in Appendices A and D.

## 5.2 Cross-Graph Prediction Performance (RQ1)

We evaluate whether a common interaction language enables shared prediction across networks with diferent interaction dynamics. Tables 1 and 2 report the results.

Obs. 1. A common interaction language supports shared prediction across dynamic graphs. CL-DGFM achieves the highest mean MRR on 10 of 13 datasets. It exceeds the strongest listed baselines by 10.78, 15.24, and 11.69 percentage points on Comment, Flickr, and YouTube, respectively. These results support the route proposed in the introduction: shared historical descriptors and learning across diverse synthetic dynamics enable an existing

Table 2: Performance on TGB-Seq (100 × MRR; higher is better). Baselines train per target graph; CL-DGFM uses a shared model with labeled context and fixed parameters. Each column’s best result is bold red and its second best is underlined.
<table><tr><td>Method</td><td>ML-20M</td><td>Taobao</td><td>Yelp</td><td>GoogleLocal</td><td>Flickr</td><td>YouTube</td><td>Patent</td><td>WikiLink</td></tr><tr><td colspan="9">Memory and neighborhood modeling Training per target graph</td></tr><tr><td>DyRep</td><td> $1 9 . 0 0 _ { \pm 1 . 6 9 }$ </td><td> $4 0 . 0 3 _ { \pm 2 . 4 0 }$ </td><td> $5 7 . 6 9 _ { \pm 1 . 0 5 }$ </td><td> $3 7 . 7 3 _ { \pm 1 . 3 4 }$ </td><td> $3 8 . 0 4 _ { \pm 4 . 1 9 }$ </td><td> $3 5 . 1 2 _ { \pm 4 . 1 3 }$ </td><td> $2 1 . 0 1 _ { \pm 1 . 1 4 }$ </td><td> $4 2 . 6 3 _ { \pm 1 . 3 3 }$ </td></tr><tr><td>JODIE</td><td> $2 1 . 1 6 _ { \pm 0 . 7 3 }$ </td><td> $4 8 . 3 6 _ { \pm 2 . 1 8 }$ </td><td> $6 9 . 8 8 _ { \pm 0 . 3 1 }$ </td><td> $4 1 . 8 6 _ { \pm 1 . 4 9 }$ </td><td> $4 6 . 2 1 _ { \pm 0 . 8 3 }$ </td><td> $4 1 . 6 7 _ { \pm 2 . 8 6 }$ </td><td> $2 4 . 6 0 _ { \pm 0 . 3 8 }$ </td><td> $5 7 . 9 4 _ { \pm 1 . 3 3 }$ </td></tr><tr><td>TGN</td><td> $2 3 . 9 9 _ { \pm 0 . 2 0 }$ </td><td> $6 0 . 2 8 _ { \pm 0 . 5 4 }$ </td><td> $6 9 . 7 9 _ { \pm 0 . 2 4 }$ </td><td> $5 4 . 1 3 _ { \pm 1 . 9 7 }$ </td><td> $4 6 . 0 3 _ { \pm 6 . 7 8 }$ </td><td> $5 5 . 1 6 _ { \pm 5 . 8 9 }$ </td><td> $\underline { { 2 2 . 8 3 } } _ { \pm 2 . 2 5 }$ </td><td> $6 2 . 9 4 _ { \pm 2 . 1 6 }$ </td></tr><tr><td>TGAT</td><td> $1 0 . 4 7 _ { \pm 0 . 2 0 }$ </td><td></td><td>一</td><td> $1 9 . 7 8 _ { \pm 0 . 2 4 }$ </td><td> $2 3 . 5 3 _ { \pm 3 . 3 5 }$ </td><td> $4 3 . 5 6 _ { \pm 2 . 5 3 }$ </td><td> $8 . 4 9 _ { \pm 0 . 1 8 }$ </td><td></td></tr><tr><td>TCL</td><td> $1 2 . 0 4 _ { \pm 0 . 0 2 }$ </td><td> $3 1 . 5 5 _ { \pm 0 . 0 3 }$ </td><td> $2 4 . 3 9 _ { \pm 0 . 6 7 }$ </td><td> $1 8 . 3 0 _ { \pm 0 . 0 2 }$ </td><td> $4 0 . 0 0 _ { \pm 1 . 7 6 }$ </td><td> $5 0 . 1 7 _ { \pm 1 . 9 8 }$ </td><td> $1 0 . 6 0 _ { \pm 1 . 7 5 }$ </td><td> $4 3 . 0 2 _ { \pm 2 . 1 6 }$ </td></tr><tr><td>GraphMixer</td><td> $2 1 . 9 7 _ { \pm 0 . 1 7 }$ </td><td> $3 1 . 5 4 _ { \pm 0 . 0 2 }$ </td><td> $3 3 . 9 6 _ { \pm 0 . 1 9 }$ </td><td> $2 1 . 3 1 _ { \pm 0 . 1 4 }$ </td><td> $4 5 . 0 1 _ { \pm 0 . 0 8 }$ </td><td> $5 8 . 8 7 _ { \pm 0 . 1 2 }$ </td><td> $1 8 . 9 7 _ { \pm 2 . 5 4 }$ </td><td> $4 8 . 5 7 _ { \pm 0 . 0 2 }$ </td></tr><tr><td>DyGFormer</td><td>一</td><td></td><td> $2 1 . 6 8 _ { \pm 0 . 2 0 }$ </td><td> $1 8 . 3 9 _ { \pm 0 . 0 2 }$ </td><td> $4 9 . 5 8 _ { \pm 2 . 8 7 }$ </td><td> $4 6 . 0 8 _ { \pm 3 . 4 4 }$ </td><td> $1 4 . 2 0 _ { \pm 2 . 9 3 }$ </td><td></td></tr><tr><td>TSNN</td><td> $\underline { { 3 5 . 4 7 } } _ { \pm 0 . 0 5 }$ </td><td> $6 8 . 2 5 _ { \pm 0 . 1 7 }$ </td><td> $6 9 . 1 3 _ { \pm 0 . 2 9 }$ </td><td> $6 1 . 4 3 _ { \pm 0 . 5 9 }$ </td><td> $6 0 . 8 4 _ { \pm 0 . 1 3 }$ </td><td> $\underline { { 5 9 . 7 8 } } _ { \pm 0 . 3 2 }$ </td><td> $1 7 . 2 1 _ { \pm 0 . 1 6 }$ </td><td> $6 9 . 3 7 _ { \pm 0 . 2 2 }$ </td></tr><tr><td colspan="9">Structural and sequential pattern modeling Training per target graph</td></tr><tr><td>CAWN</td><td> $1 2 . 3 1 _ { \pm 0 . 0 2 }$ </td><td></td><td> $2 5 . 7 1 _ { \pm 0 . 0 9 }$ </td><td> $1 8 . 2 6 _ { \pm 0 . 0 2 }$ </td><td> $4 8 . 6 9 _ { \pm 6 . 0 8 }$ </td><td> $4 7 . 5 5 _ { \pm 1 . 0 8 }$ </td><td> $1 2 . 3 4 _ { \pm 0 . 4 7 }$ </td><td></td></tr><tr><td>CRAFT</td><td> $3 4 . 4 2 _ { \pm 0 . 6 8 }$ </td><td> $\underline { { 6 9 . 1 3 } } _ { \pm 0 . 4 5 }$ </td><td> $\underline { { 7 1 . 1 2 } } _ { \pm 0 . 5 8 }$ </td><td> $6 0 . 8 1 _ { \pm 0 . 4 9 }$ </td><td> $\underline { { 6 0 . 8 7 } } _ { \pm 0 . 8 1 }$ </td><td> $5 7 . 3 9 _ { \pm 0 . 1 9 }$ </td><td> $1 8 . 4 8 _ { \pm 0 . 7 2 }$ </td><td> $\underline { { 7 3 . 9 2 } } _ { \pm 0 . 8 6 }$ </td></tr><tr><td>TAMI</td><td> $3 1 . 5 8 _ { \pm 0 . 0 3 }$ </td><td> $6 7 . 1 1 _ { \pm 0 . 2 4 }$ </td><td> $6 7 . 8 0 _ { \pm 0 . 4 1 }$ </td><td> $\underline { { 6 1 . 3 5 } } _ { \pm 0 . 4 8 }$ </td><td> $5 8 . 6 2 _ { \pm 0 . 2 2 }$ </td><td> $5 8 . 1 1 _ { \pm 0 . 2 0 }$ </td><td> $1 6 . 5 3 _ { \pm 0 . 3 5 }$ </td><td> $6 7 . 8 9 _ { \pm 0 . 4 2 }$ </td></tr><tr><td colspan="9">Dynamic graph foundation model One shared model; in-context prediction; no target-graph fine-tuning</td></tr><tr><td>CL-DGFM</td><td> $3 6 . 6 4 _ { \pm 0 . 6 3 }$ </td><td> $6 9 . 6 7 _ { \pm 0 . 0 3 }$ </td><td> $7 2 . 4 8 _ { \pm 0 . 0 8 }$ </td><td> $5 7 . 3 6 _ { \pm 0 . 2 3 }$ </td><td> $7 6 . 1 1 _ { \pm 0 . 2 1 }$ </td><td> $7 1 . 4 7 _ { \pm 0 . 0 1 }$ </td><td> $1 9 . 3 2 _ { \pm 0 . 6 1 }$ </td><td> $7 4 . 2 5 _ { \pm 0 . 1 7 }$ </td></tr></table>

Table 3: Predictor comparison using the same tokenizer (100× MRR; higher is better). Column-wise best results are red and bold; second-best results are underlined.
<table><tr><td>Predictor</td><td>Wiki</td><td>Google Local</td><td>Flickr</td><td>YouTube</td></tr><tr><td>Machine learning</td><td>Training per graph</td><td></td><td></td><td></td></tr><tr><td>Logistic regression</td><td>79.73</td><td>54.31</td><td>70.07</td><td>69.32</td></tr><tr><td>Random forest</td><td>79.25</td><td>56.10</td><td>71.08</td><td>69.51</td></tr><tr><td>CatBoost</td><td>80.65</td><td>56.13</td><td>72.20</td><td>69.95</td></tr><tr><td>LightGBM</td><td>81.21</td><td>54.12</td><td>71.94</td><td>66.63</td></tr><tr><td>Deep learning</td><td>Training per graph</td><td></td><td></td><td></td></tr><tr><td>MLP</td><td>80.37</td><td>54.07</td><td>70.87</td><td>69.19</td></tr><tr><td>ResNet</td><td>81.76</td><td>54.56</td><td>70.31</td><td>67.62</td></tr><tr><td>FT-Transformer</td><td>81.94</td><td>55.70</td><td>75.07</td><td>69.74</td></tr><tr><td colspan="3">Pretrained foundation models No fine-tuning Fixed weights across graphs; graph-specific labeled context</td><td></td><td></td></tr><tr><td colspan="2">TabICLv2 77.19</td><td>56.63</td><td>73.63</td><td>71.16</td></tr><tr><td colspan="2">TabPFNv2 81.12</td><td>56.14</td><td>74.83</td><td>69.00</td></tr><tr><td colspan="2">LimiX-2M 79.82</td><td>56.41</td><td>72.72</td><td>69.91</td></tr><tr><td colspan="2">CL-DGFM 82.24</td><td>57.36</td><td>76.11</td><td>71.47</td></tr></table>

![](images/2f00f6a463835549c3cc0c101a46b214435bbd5a47c38d135b5ab43d0373b7f7.jpg)  
Figure 3: Representation comparison with a frozen backbone. Error bars show standard deviations over three paired context draws; dashed lines mark the full tokenizer.

## 5.3 Efectiveness of the Common Interaction Language (RQ2)

foundation predictor to transfer across real networks. Labeled historical context conditions prediction without target-graph parameter updates or an extra trainable graph encoder. CL-DGFM remains slightly below TPNet on Wiki and shows larger gaps on GoogleLocal and Patent. Shared descriptors may omit useful historical details, while the synthetic prior and limited context may not suficiently cover some target dynamics. This suggests that a common representation supports transfer, but efectiveness depends on learning and identifying relevant history-outcome relationships.

We represent each candidate interaction as a feature vector and stack these vectors into a table. Using this shared input format, we evaluate conventional machine learning models, deep learning models, and pretrained tabular foundation models. We also compare alternative representations using the same frozen backbone and assess tokenizer components by removing each component while keeping model weights fixed.

Obs. 2. The common language brings tabular foundation models to dynamic graph prediction. Table 3 shows that directly using tokenizer outputs as feature vectors allows ordinary machine learning and neural models to predict temporal links with competitive performance. More importantly, arranging these outputs as tabular data enables direct use of pretrained tabular foundation models on dynamic graphs. TabICLv2, TabPFNv2, and LimiX-2M perform prediction with fixed parameters and labeled historical context, without an additional graph encoder. Thus, the common language makes dynamic graph prediction accessible to existing predictors and allows capabilities learned from tabular tasks to be reused on interaction histories.

<table><tr><td rowspan=1 colspan=10>(a)                    (b)</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1>Full</td><td rowspan=1 colspan=1>w/o</td><td rowspan=1 colspan=1>w/o</td><td rowspan=1 colspan=1>w/o</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>w/o</td><td rowspan=1 colspan=1>w/o</td><td rowspan=1 colspan=1>w/o</td><td rowspan=1 colspan=1>w/o</td></tr><tr><td rowspan=1 colspan=1>IM</td><td rowspan=1 colspan=1>HC</td><td rowspan=1 colspan=1>ODT</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>gpref</td><td rowspan=1 colspan=1>grec</td><td rowspan=1 colspan=1>gseq</td><td rowspan=1 colspan=1>gstr</td></tr><tr><td rowspan=1 colspan=1>Wiki</td><td rowspan=1 colspan=1>82.24</td><td rowspan=1 colspan=1>-4.72</td><td rowspan=1 colspan=1>-5.38</td><td rowspan=1 colspan=1>-7.51</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.17</td><td rowspan=1 colspan=1>-0.73</td><td rowspan=1 colspan=1>-0.44</td><td rowspan=1 colspan=1>-0.27</td></tr><tr><td rowspan=1 colspan=1>GoogleLocal</td><td rowspan=1 colspan=1>57.36</td><td rowspan=1 colspan=1>-2.88</td><td rowspan=1 colspan=1>-29.37</td><td rowspan=1 colspan=1>-0.90</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.91</td><td rowspan=1 colspan=1>-0.61</td><td rowspan=1 colspan=1>-0.40</td><td rowspan=1 colspan=1>-0.57</td></tr><tr><td rowspan=1 colspan=1>ML-20M</td><td rowspan=1 colspan=1>36.64</td><td rowspan=1 colspan=1>-9.57</td><td rowspan=1 colspan=1>-5.41</td><td rowspan=1 colspan=1>-2.10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.59</td><td rowspan=1 colspan=1>-0.03</td><td rowspan=1 colspan=1>-0.48</td><td rowspan=1 colspan=1>-0.81</td></tr><tr><td rowspan=1 colspan=1>Flickr</td><td rowspan=1 colspan=1>76.11</td><td rowspan=1 colspan=1>-5.56</td><td rowspan=1 colspan=1>-22.82</td><td rowspan=1 colspan=1>-1.69</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.58</td><td rowspan=1 colspan=1>-0.45</td><td rowspan=1 colspan=1>-0.86</td><td rowspan=1 colspan=1>-0.75</td></tr><tr><td rowspan=1 colspan=1>YouTube</td><td rowspan=1 colspan=1>71.47</td><td rowspan=1 colspan=1>-2.17</td><td rowspan=1 colspan=1>-12.32</td><td rowspan=1 colspan=1>-0.57</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.49</td><td rowspan=1 colspan=1>-0.54</td><td rowspan=1 colspan=1>-0.51</td><td rowspan=1 colspan=1>-0.68</td></tr><tr><td rowspan=1 colspan=10>∆MRR (pp)-30 -20-i0  0-1.0       -0.5        0.0</td></tr></table>

Figure 4: MRR changes from ablating (a) tokenizer components (RQ2) and (b) prior mechanisms (RQ3).

Obs. 3. Explicit interaction patterns outperform combined topology and temporal statistics. Figure 3 shows that the complete tokenizer outperforms topology statistics, temporal statistics, and their concatenation on all four graphs. Merely combining topology and time does not consistently improve prediction. The tokenizer instead describes how contacts repeat, how endpoints connect indirectly, and which destinations follow earlier visits. These explicit interaction patterns provide more useful evidence to the same frozen predictor, showing that the benefit comes from how histories are represented, beyond a change in input format. Obs. 4. Diferent graphs rely on diferent combinations of memory, connectivity, and transitions. Figure 4(a) shows that removing connectivity causes the largest drops on GoogleLocal, Flickr, and YouTube. Indirect paths can distinguish candidates with little or no direct contact, which pair memory alone cannot resolve. Memory matters most on ML-20M, consistent with repeated contacts and their recency providing useful evidence. Transitions matter most on Wiki, suggesting that destination order adds information beyond contact frequency and connectivity. These patterns explain the components’ complementary roles: a common language must preserve multiple historical signals to support graphs with diferent interaction dynamics.

## 5.4 Factors Supporting Cross-Graph Transfer (RQ3)

We ablate prior mechanisms with matched retraining budgets, vary the number ofdistinct synthetic tasks at a fixed budget, and compare predictors across context sizes. These experiments examine which training dynamics support transfer and how additional historical examples afect prediction.

Obs. 5. The value of each training mechanism depends on the target dynamics. Figure 4(b) shows that removing recurrence causes the largest drop on Wiki, but barely afects ML-20M. Structural proximity matters most on ML-20M and YouTube, while preference and transitions have the largest efects on GoogleLocal and Flickr, respectively. Thus, no single training mechanism provides the best preparation for every graph. Although all variants retain the same tokenizer, removing a mechanism reduces transfer performance. This supports learning across diferent history–outcome relationships, rather than relying on a common representation alone.

![](images/fa8dbc2af76e96b40a3ab691fa4c146636fcd6704c682db2a77e497c94dbb017.jpg)  
Figure 5: Training on more distinct synthetic graphs versus repeatedly using fewer graphs, with the same number of training updates and graph samples. Shades indicate graphpool sizes; endpoints report MRR.

Obs. 6. Diverse synthetic graphs improve transfer beyond repeated training on a small graph pool. Figure 5 compares pools containing 256, 1,024, 4,096, and 16,384 distinct synthetic graphs. Each pool covers the same prior configurations, with variation in sampled graph structures and interaction dynamics. Every run uses 4,096 updates with four graphs per update, totaling 16,384 graph samples. The four settings therefore use each graph approximately 64, 16, 4, and 1 times, respectively. Expanding the pool improves MRR on all five real graphs, including from 68.56 to 82.18 on Wiki and from 62.22 to 76.02 on Flickr. These gains show that learning from more realizations of the same dynamic mechanisms supports better transfer than repeatedly training on a small graph pool. This supports generating varied interaction histories to prepare the shared predictor for diferent real networks.

Obs. 7. Additional context helps some graphs substantially, but does not guarantee better prediction. Figure 6 shows strong context gains on Flickr and GoogleLocal, whereas Wiki and YouTube improve less and saturate earlier. On Wiki, frozen TabICLv2 even declines as context grows toward 512 positive events before partially recovering. Thus, more labeled history does not automatically produce better interpretation of interaction patterns. CL-DGFM remains strongest across the tested sizes, with a clear advantage on Flickr but a narrowing gap on GoogleLocal as context grows. These results suggest that learning across dynamics and supplying more target history ofer diferent benefits: context can reduce some transfer gaps, while the learned predictor remains important when additional examples yield limited gains.

## 6 Conclusion

We introduced CL-DGFM, combining a common interaction language with learning across diverse synthetic dynamics. The language enables existing tabular predictors to process dynamic graphs, while labeled historical context supports prediction without targetgraph parameter updates. Experiments on TGB and TGB-Seq demonstrate competitive performance across predictor families. These results support existing predictive architectures across dynamic networks through shared representations of interaction histories.

![](images/24baa9a01c116e7062098baf785fa579b0ffe1ee050b77f84d953b6f07c13ebd.jpg)  
Figure 6: Efect of context size on prediction (100×MRR). � counts positive context events, each paired with one negative. Dashed lines mark the main setting $K = 5 1 2 ;$ vertical-axis ranges difer across datasets.

## References

[1] Md Sadek Hossain Asif, Tanzila Khan, and Md. Mosaddek Khan. 2026. Temporal Sheaf Neural Networks with Dynamic Orthogonal Transport. arXiv preprint arXiv:2606.10071 (2026).

[2] Jeongwhan Choi, Jongwoo Kim, Woosung Kang, and Noseong Park. 2026. Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors. In International Conference on Learning Representations.

[3] Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, and Mehrdad Mahdavi. 2023. Do We Really Need Complicated Model Architectures for Temporal Networks?. In International Conference on Learning Representations.

[4] Fabrizio De Castelli, Alessio Gravina, Davide Bacciu, and Moshe Eliasof. 2026. Disentangling Dynamics: A Compositional Framework for Temporal Graph Foundation Models. In ICML Workshop on Graph Foundation Models.

[5] Dmitry Eremeev, Gleb Bazhenov, Oleg Platonov, Artem Babenko, and Li udmila Prokhorenkova. 2025. Turning Tabular Foundation Models into Graph Foundation Models. arXiv preprint. Also presented at the NeurIPS 2025 Workshop on New Perspectives in Advancing Graph Machine Learning. arXiv:2508.20906 [cs.LG]

[6] Dmitry Eremeev, Oleg Platonov, Gleb Bazhenov, Artem Babenko, and Liudmila Prokhorenkova. 2026. GraphPFN: A Prior-Data Fitted Graph Foundation Model. In Proceedings ofthe 43rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 306). 28149–28165.

[7] Ruiyi Fang, Bingheng Li, Zhao Kang, Qiuhao Zeng, Nima Hosseini Dashtbayaz, Ruizhi Pu, Boyu Wang, and Charles Ling. 2025. On the benefits ofattribute-driven graph domain adaptation. The Thirteenth International Conference on Learning Representations (2025).

[8] Ruiyi Fang, Bingheng Li, Jingyu Zhao, Ruizhi Pu, Qiuhao Zeng, Gezheng Xu, Charles Ling, and Boyu Wang. 2025. Homophily Enhanced Graph Domain Adaptation. Forty-Second International Conference on Machine Learning (2025).

[9] Jian Gao, Jianshe Wu, and JingYi Ding. 2025. HyperEvent: A Strong Baseline for Dynamic Link Prediction via Relative Structural Encoding. arXiv preprint. arXiv:2507.11836 [cs.LG]

[10] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. 2021. Revisiting Deep Learning Models for Tabular Data. In Advances in Neural Information Processing Systems, Vol. 34.

[11] Alessio Gravina, Giulio Lovisotto, Claudio Gallicchio, Davide Bacciu, and Claas Grohnfeldt. 2024. Long Range Propagation on Continuous-Time Dynamic Graphs. In Proceedings of the 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235). 16206–16225.

[12] Alan G. Hawkes. 1971. Spectra of Some Self-Exciting and Mutually Exciting Point Processes. Biometrika 58, 1 (1971), 83–90.

[13] Yufei He, Yuan Sui, Xiaoxin He, and Bryan Hooi. 2025. UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 448–459.

[14] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. 2025. Accurate Predictions on Small Data with a Tabular Foundation Model. Nature 637 (2025), 319–326.

[15] Shi Bin Hoo, Samuel Müller, David Salinas, and Frank Hutter. 2026. From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting. Transactions on Machine Learning Research (July 2026).

[16] Shenyang Huang, Farimah Poursafaei, Jacob Danovitch, Matthias Fey, Weihua Hu, Emanuele Rossi, Jure Leskovec, Michael Bronstein, Guillaume Rabusseau, and Reihaneh Rabbany. 2023. Temporal Graph Benchmark for Machine Learning on Temporal Graphs. In Advances in Neural Information Processing Systems.

[17] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A Highly Eficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems, Vol. 30.

[18] Lecheng Kong, Jiarui Feng, Hao Liu, Chengsong Huang, Jiaxin Huang, Yixin Chen, and Muhan Zhang. 2025. GOFA: A Generative One-For-All Model for Joint Graph Language Modeling. In International Conference on Learning Representations.

[19] Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

[20] Tianyin Liao, Chunyu Hu, Yicheng Sui, Xingxuan Zhang, Peng Cui, Jianxin Li, and Ziwei Zhang. 2026. TFMLinker: Universal Link Predictor by Graph In-Context Learning with Tabular Foundation Models. arXiv preprint. arXiv:2602.08592 [cs.LG]

[21] Hao Liu, Jiarui Feng, Lecheng Kong, Ningyue Liang, Dacheng Tao, Yixin Chen, and Muhan Zhang. 2024. One for All: Towards Training One Graph Model for All Classification Tasks. In International Conference on Learning Representations.

[22] Xiaodong Lu, Leilei Sun, Tongyu Zhu, and Weifeng Lv. 2024. Improving Temporal Link Prediction via Temporal Walk Matrix Projection. In Advances in Neural Information Processing Systems, Vol. 37.

[23] Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Hamidreza Kamkari, Jesse C. Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L. Caterini, and Maksims Volkovs. 2025. TabDPT: Scaling Tabular Foundation Models on Real Data. In Advances in Neural Information Processing Systems, Vol. 38.

[24] Yijun Ma, Zehong Wang, Weixiang Sun, and Yanfang Ye. 2026. Temporal Graph Pattern Machine. arXiv preprint. Presented at the ICML 2026 Workshop on Graph Foundation Models. arXiv:2601.22454 [cs.LG]

[25] Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. 2022. Transformers Can Do Bayesian Inference. In International Conference on Learning Representations.

[26] Farimah Poursafaei, Shenyang Huang, Kellin Pelrine, and Reihaneh Rabbany. 2022. Towards Better Evaluation for Dynamic Link Prediction. In Advances in Neural Information Processing Systems, Vol. 35.

[27] Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: Unbiased Boosting with Categorical Features. In Advances in Neural Information Processing Systems, Vol. 31.

[28] Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. 2026. TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model. In Proceedings ofthe 43rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 306). 102477–102555.

[29] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein. 2020. Temporal Graph Networks for Deep Learning on Dynamic Graphs. arXiv preprint. arXiv:2006.10637 [cs.LG]

[30] Rakshit Trivedi, Mehrdad Farajtabar, Prasenjeet Biswal, and Hongyuan Zha. 2019. DyRep: Learning Representations over Dynamic Graphs. In International Conference on Learning Representations.

[31] Lu Wang, Xiaofu Chang, Shuang Li, Yunfei Chu, Hui Li, Wei Zhang, Xiaofeng He, Le Song, Jingren Zhou, and Hongxia Yang. 2021. TCL: Transformer based Dynamic Graph Modelling via Contrastive Learning. arXiv preprint. arXiv:2105.07944 [cs.LG]

[32] Yanbang Wang, Yen-Yu Chang, Yunyu Liu, Jure Leskovec, and Pan Li. 2021. Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. In International Conference on Learning Representations.

[33] Zehong Wang, Zheyuan Zhang, Tianyi Ma, Nitesh V. Chawla, Chuxu Zhang, and Yanfang Ye. 2025. Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 267). 65518– 65555.

[34] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2020. Inductive Representation Learning on Temporal Graphs. In International Conference on Learning Representations.

[35] Linjie Xu, Yanlin Zhang, Quan Gan, Minjie Wang, and David Wipf. 2026. No Need to Train Your RDB Foundation Model. In Proceedings ofthe 43rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 306). 142705–142726.

[36] Lu Yi, Runlin Lei, Fengran Mo, Yanping Zheng, Zhewei Wei, and Yuhang Ye. 2025. Future Link Prediction Without Memory or Aggregation. In Advances in Neural Information Processing Systems, Vol. 38.

[37] Lu Yi, Jie Peng, Yanping Zheng, Fengran Mo, Zhewei Wei, Yuhang Ye, Zixuan Yue, and Zengfeng Huang. 2025. TGB-Seq Benchmark: Challenging Temporal GNNs with Complex Sequential Dynamics. In The Thirteenth International Conference on Learning Representations.

[38] Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. 2023. Towards Better Dynamic Graph Learning: New Architecture and Unified Library. In Advances in Neural Information Processing Systems, Vol. 36.

[39] Zhongyi Yu, Jianqiu Wu, Zhenghao Wu, Shuhan Zhong, Weifeng Su, Chul-Ho Lee, and Weipeng Zhuo. 2025. TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction. In Advances in Neural Information Processing Systems, Vol. 38.

[40] Haonan Yuan, Qingyun Sun, Junhua Shi, Xingcheng Fu, Jianxin Li, and Philip S. Yu. 2026. Decoupled and Divergence-Conditioned Prompt for Multi-domain Dynamic Graph Foundation Models. arXiv preprint. arXiv:2605.13540 [cs.LG]

[41] Xingxuan Zhang, Gang Ren, Han Yu, Hao Yuan, Hui Wang, Jiansheng Li, Jiayun Wu, Lang Mo, Li Mao, Mingchao Hao, Ningbo Dai, Renzhe Xu, Shuyang Li, Tianyang Zhang, Yue He, Yuanrui Wang, Yunjia Zhang, Zijing Xu, Dongzhe Li, Fang Gao, Hao Zou, Jiandong Liu, Jiashuo Liu, Jiawei Xu, Kaijie Cheng, Kehan Li, Linjun Zhou, Qing Li, Shaohua Fan, Xiaoyu Lin, Xinyan Han, Xuanyue Li, Yan Lu, Yuan Xue, Yuanyuan Jiang, Zimu Wang, Zhenlei Wang, and Peng Cui. 2025. LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence. arXiv preprint. arXiv:2509.03505 [cs.LG]

[42] Xiyuan Zhang, Danielle Maddix Robinson, Junming Yin, Nick Erickson, Ab dul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Tony Hu, Huzefa Rangwala, George Karypis, and Yuyang Wang. 2025. Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models. In Advances in Neural Information Processing Systems, Vol. 38.

[43] Xiaohui Zhang, Yanbo Wang, Xiyuan Wang, and Muhan Zhang. 2025. Eficient Neural Common Neighbor for Temporal Graph Link Prediction. In Proceedings of the Fourth Learning on Graphs Conference.

## A Datasets and Experimental Settings

To facilitate reproducibility, we provide our code and experimental configurations in the following anonymous repository: https:// anonymous.4open.science/r/CL-DGFM-C10D/.

## A.1 Dataset Statistics and Splits

Table 4 reports dataset statistics, test-query counts, and negativecandidate counts. Node counts include only participating entities. We use v2 of Wiki, Review, Coin, and Flight, and v1 of Comment. TGB [16] uses chronological splits with approximately 70% for training, 15% for validation, and 15% for testing, whereas TGB-Seq [37] uses dataset-specific splits. Together, these datasets evaluate whether a shared predictor transfers across interaction semantics, graph structures, and scales without target-graph parameter updates. They cover six application domains:

• Online interactions. Wiki records user–page edits, Comment captures social replies, and Flickr and YouTube record social links. These datasets cover user–content and user– user interactions in bipartite and homogeneous graphs.

• Recommendation. Review, GoogleLocal, and Yelp record product or business reviews, ML-20M movie ratings, and Taobao product clicks. These bipartite graphs capture user preferences through feedback and clicks.

• Financial transactions. Coin records directed transactions between accounts. Each event identifies a source account, a destination account, and a timestamp, forming a temporal history of financial exchanges.

• Transportation. Flight records flights between airports, representing physical movements through a transport network. With approximately 67 million events, it contains the largest interaction history among our datasets.

• Citations. Patent records directed citations between patents. Its history also includes events that are not evaluation targets but remain observable to strictly later queries.

• Hyperlinks. WikiLink records links between Wikipedia pages, capturing changes in document connectivity. Its pageto-page relations complement the user–page editing activity recorded in Wiki.

Our model is trained exclusively on synthetic interaction streams, and no real benchmark split is used to update its parameters. Evaluation uses the oficial test splits, with preceding real interactions providing observable history and labeled context without parameter updates.

## A.2 Evaluation Details

Observable history includes earlier validation and test events, but excludes interactions at or after the query timestamp. We rank the observed destination against the benchmark-provided negatives, retaining repeated candidate positions. MRR averages its reciprocal rank over test queries, with average ranks assigned to ties.

All target-graph contexts are sampled from the final 10% of the training interval. Within each comparison, representations and predictors share context samples, labels, and test candidates. The context-size study varies only the context size, keeping each predictor’s parameters fixed. Within each evaluation run, labeled context stays fixed, while observable history advances with query time. New interactions update subsequent candidate features without parameter updates or context reselection.

Alternative representations. Representation comparisons use the same frozen TabICLv2 predictor. Topology contains 12 statistics: endpoint in/out neighborhood sizes, common neighbors, Adamic– Adar, resource allocation,Jaccard similarity, preferential attachment, forward/reverse pair counts, and three-hop path counts.

Time contains 20 statistics: the time since the latest event and event counts in four age intervals (0–0.03, 0.03–0.3, 0.3–3, and above 3, measured in calibrated time units) for each of the forward-pair, reverse-pair, source-incident, and destination-incident histories. Topology+Time concatenates both groups. Appending candidaterelative ranks yields 24, 40, and 64 input dimensions, respectively. Topology summarizes accumulated connectivity, whereas Time describes when interactions occurred. Their concatenation tests whether combining conventional structural and temporal statistics sufices to match the tokenizer’s explicit interaction patterns.

Ablations. With model parameters fixed, tokenizer ablations separately remove memory, connectivity, or transition features, including the associated normalized features and matching candidaterelative rank columns, from both context and query inputs. Prior ablations remove one generation mechanism at a time and retrain on the resulting synthetic streams, using the same tokenizer and matched training budgets.

## A.3 Baseline Sources and Implementation

We reproduce CRAFT, TAMI, and TSNN using their public implementations and reported settings. Other temporal baseline results are taken from the benchmarks. In tokenizer-based predictor comparisons, machine learning and deep learning predictors take $\mathbf { \hat { X } } ^ { M L \& D L } = \mathbf { x }$ as input, fit the labeled context, and select configurations by validation MRR. For foundation models, we stack the same vectors x into an input table �<sup>tab</sup>, comprising labeled context rows and unlabeled query rows, and perform prediction with fixed parameters. Main experiments run on NVIDIA A40 GPUs with Python 3.10, PyTorch 2.6.0, CUDA 12.4, and TabICL 2.2.0.

Table 4: Statistics of synthetic training data and real evaluation datasets. B/H denote bipartite/homogeneous graphs. Events counts all observed interactions; Test counts evaluation queries. Neg. denotes negative destinations per test query, excluding the positive destination.
<table><tr><td>Dataset</td><td>Domain (type)</td><td>Nodes</td><td>Events</td><td>Test</td><td>Neg.</td></tr><tr><td colspan="6">Synthetic training data</td></tr><tr><td>Generated streams</td><td>Synthetic interactions (B/H)</td><td>≈ 1.30M</td><td>≈ 28.50M(training data)</td><td></td><td></td></tr><tr><td colspan="6">TGB evaluation datasets</td></tr><tr><td>Wiki</td><td>Page edits (B)</td><td>9,227</td><td>157,474</td><td>23,621</td><td>999</td></tr><tr><td>Review</td><td>Product reviews (B)</td><td>352,637</td><td>4,873,540</td><td>728,919</td><td>100</td></tr><tr><td>Coin</td><td>Transactions (H)</td><td>638,486</td><td>22,809,486</td><td>3,421,411</td><td>20</td></tr><tr><td>Flight</td><td>Flights (H)</td><td>18,143</td><td>67,169,570</td><td>10,026,943</td><td>20</td></tr><tr><td>Comment</td><td>Social replies (H)</td><td>994,790</td><td>44,314,507</td><td>6,647,176</td><td>20</td></tr><tr><td colspan="6">TGB-Seq evaluation datasets</td></tr><tr><td>GoogleLocal</td><td>Business reviews (B)</td><td>473,580</td><td>1,913,967</td><td>43,546</td><td>100</td></tr><tr><td>ML-20M</td><td>Movie ratings (B)</td><td>110,431</td><td>14,494,325</td><td>129,291</td><td>100</td></tr><tr><td>Flickr</td><td>Social links (H)</td><td>233,836</td><td>7,223,559</td><td>609,679</td><td>100</td></tr><tr><td>YouTube</td><td>Social links (H)</td><td>402,422</td><td>3,288,028</td><td>232,865</td><td>100</td></tr><tr><td>Yelp</td><td>Business reviews (B)</td><td>1,743,769</td><td>19,760,293</td><td>1,032,354</td><td>100</td></tr><tr><td>Taobao</td><td>Product clicks (B)</td><td>1,623,633</td><td>18,853,792 12,749,824</td><td>2,406,071</td><td>100</td></tr><tr><td>Patent†</td><td>Citations (H)</td><td>2,241,784</td><td></td><td>683,323</td><td>100</td></tr><tr><td>WikiLink</td><td>Hyperlinks (H)</td><td>1,361,972</td><td>34,163,774</td><td>3,073,045</td><td>100</td></tr></table>

<sup>†</sup>Patent includes 1,931,005 history-only events in its event total; these events are not evaluation targets and are visible only to strictly later queries.

## B Dynamic Interaction Tokenizer

## B.1 Time Calibration and Feature Settings

We calibrate time using the training history of each real graph or the pre-context history of each synthetic task. Let � count events in this history, $n _ { \mathrm { s r c } }$ count distinct sources, and $t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } }$ denote its earliest and latest timestamps. The time scale is:

$$
s = \operatorname* { m a x } \left\{ \frac { ( t _ { \mathrm { m a x } } - t _ { \mathrm { m i n } } ) n _ { \mathrm { s r c } } } { n } , \epsilon _ { s } \right\} .\tag{15}
$$

We use $\epsilon _ { s } ~ = ~ 1 0 ^ { - 6 }$ for real graphs and $1 0 ^ { - 8 }$ for synthetic tasks, keeping � fixed per graph or task.

All feature histories strictly precede the represented timestamp �. Let $C _ { u v }$ count directed pair events and $d _ { x }$ count all events incident to node �, including repeated interactions. We log-compress these counts using $\ell _ { + } ( x ) = \log ( 1 + x ) . I _ { x }$ retains the latest 256 incident timestamp entries of �, in either direction. $M _ { D } ( h )$ and $W _ { \ell } ( w , h )$ denote the uncompressed sums underlying Eqs. (3) and (5). Recency is $R _ { D } = - \log ( 1 + ( t - \operatorname* { m a x } D ) / s )$ , with $R _ { \emptyset } = - 3 0 .$ . Empty sums are zero. Table 5 follows the main-text concatenation order, with rows 9–18 forming $\mathbf { a } ( u , v , t )$ . Rows 19–25 instantiate $\Omega _ { P }$ , and rows 26–30 instantiate Ω<sub>�</sub>.

## B.2 History Retention

To construct $\mathcal { P } _ { \ell } ( u , v , t )$ , we expand paths through each node’s latest 256 incident entries and count distinct event combinations separately. The second hop excludes returns to �, and the third excludes � and the first intermediate node, with no further deduplication.

To construct $O _ { L } ( u , v , t )$ , we inspect the latest 48 incident entries of each reference destination � for starting visits $b \to a$ from other sources $b \neq u .$ For each starting visit at $\tau _ { a } ,$ we scan 24 incident entries of � from its first timestamp $\geq \tau _ { a } .$ . We retain the first four outgoing visits strictly between $\tau _ { a }$ and � to destinations other than �.

Table 5: The 31-dimensional history descriptor. $\begin{array} { r l } { \mathcal { B } } & { { } = } \end{array}$ �{0.03, 0.3, 3, 30} is ordered increasingly. Ratios precede log compression, and ℎ = ∞ removes decay.  
```latex
No. Features and settings
Interaction memory: $: \mathbf { z } ^ { \mathrm { { m e m } } } \in \mathbb { R } ^ { 1 8 }$
1–4 $\ell _ { + } ( M _ { E u v } \left( h \right) ) , h $ ∈ B, using all past � → � events
5–8 $\ell _ { + } ( M _ { I _ { v } } ( h ) ) , h \in \mathcal { B }$
9–11 $\ell _ { + } ( C _ { u v } ) , \ell _ { + } ( d _ { u } ) , \ell _ { + } ( d _ { v } )$
12–14 $R _ { E u v } , R _ { I u } , R _ { I v }$
15–17 $\ell _ { + } ( C _ { v u } ) , ^ { - 1 } \{ C _ { u v } > 0 \} , 1 \{ d _ { v } = 0 \}$
18 $\ell _ { + } \left( \frac { { \cal M } _ { E u v } \left( 0 . 3 s \right) } { 1 + { \cal M } _ { I v } \left( 0 . 3 s \right) } \right)$
Historical connectivity: $\mathbf { z } ^ { \mathrm { c o n n } } \in \mathbb { R } ^ { 7 }$
19–21 $\ell _ { + } \left( W _ { 2 } \left( 1 , \infty \right) \right) , \stackrel { \cdot } { \ell } _ { + } \left( W _ { 2 } \left( w _ { \mathrm { a c t } } , \infty \right) \right) , \ell _ { + } \left( W _ { 2 } \left( w _ { \mathrm { a c t } } , s \right) \right)$
22–24 ℓ (� (1, ∞) ), ℓ (� (� , ∞) ), ℓ (� (� , � ) )
$\begin{array} { r l r l } { { 2 5 } } & { { } } & { \ell _ { + } \left( W _ { 3 } \left( w _ { \mathrm { d s t } } , \infty \right) \right) } \end{array}$
$w _ { \mathrm { a c t } } ( \rho ) = \prod _ { k = 1 } ^ { \ell - 1 } \operatorname* { m a x } ( 1 , d _ { x _ { k } } ) ^ { - 1 }$ for intermediate nodes $x _ { k } .$ , and $w _ { \mathrm { d s t } } ( \rho ) = ( 1 + d _ { v } ) ^ { - 1 / 2 } .$
Ordered transitions: $\mathbf { z } ^ { \mathrm { s e q } } \in \mathbb { R } ^ { 6 }$
26–27 ℓ<sub>+</sub> (�<sub>1</sub> (�<sub>pos</sub>, ∞) ), ℓ<sub>+</sub> (�<sub>3</sub> (�<sub>pos</sub>, ∞) )
28 ℓ<sub>+</sub> (�<sub>3</sub> (�<sub>pos</sub>, � ) )
29 $\ell _ { + } ( T _ { 3 } ( \dot { \omega _ { \mathrm { s r c } } } , \infty ) )$
30 $\ell _ { + } \left( T _ { 3 } \left( \omega _ { \mathrm { d s t } } , \infty \right) \right)$
31 $1 \{ v = v _ { u } ^ { \mathrm { l a s t } } \}$ , zero without outgoing history
�<sub>pos</sub> = 1/(��), �<sub>src</sub> = �<sub>pos</sub>/√︁max(1, �<sub>�</sub> ), and �<sub>dst</sub> $= \omega _ { \mathrm { p o s } } / \sqrt { 1 + d _ { v } } ,$ where � indexes refer
ences from most recent, � indexes eligible successors, and � is their source.
```

The successor index � in � (�) follows this retained order, regardless of which destinations appear in the query’s candidate list.

## B.3 Input Transformation

We use $c _ { x } = 4 \mathrm { i n } \mathrm { E q . } ( 9 )$ . Let ran $\mathfrak { c } _ { q , k } ( j )$ be candidate $j ^ { \prime } s$ ascending rank on feature � within query $q ,$ starting at one and averaging ties. The rank vector in the main text is:

$$
\mathbf { r } _ { q , j } = \left[ \frac { 2 \operatorname { r a n k } _ { q , k } ( j ) - 1 } { K _ { q } } - 1 \right] _ { k = 1 } ^ { 3 1 } .\tag{16}
$$

Repeated candidate positions are retained. Context ranks are computed within each complete candidate list before row selection.

Algorithm 1 CL-DGFM: Dynamic-Prior Learning and Cross-Graph   
Inference   
Require: Prior Π, pretrained backbone, updates �, target graph �, target   
queries Q   
Ensure: Candidate scores $\{ \pmb { s } _ { q } ^ { G } \} _ { q \in Q }$   
1: Initialize the backbone and initialize the Contextual Value Adapter.   
2: Freeze input encoders.   
Dynamic-prior learning   
3: for $k = 1 , \dots , U$ do   
4: Select generator configurations $\xi$ according to Π.   
5: Generate sequences E using the source rates and destination distri   
bution $\pi _ { \xi }$ in Section 4.2.   
6: Construct a batch of tasks T from earlier context events and strictly   
later queries.   
7: for all tasks $\mathcal { T }$ do   
8: Construct $X _ { C } , \mathbf { y } _ { C }$ and $\{ X _ { q } \} _ { q \in \mathcal { I } _ { Q } }$ using the tokenizer.   
9: Apply one sampled, context-fitted preprocessing view.   
10: Score task queries with $F _ { \theta }$ using Eq. (13).   
11: end for   
12: Estimate the batch ranking loss using Eq. (14).   
13: Update the in-context predictor and adapter with AdamW.   
14: end for   
Cross-graph inference   
15: Fix all model parameters.   
16: Select historical context from $G \ ' s$ training interval and construct   
$X _ { C } , \mathbf { y } _ { C } .$   
17: Fit the backbone’s two preprocessing views on $X _ { C }$   
18: for all $q \in { \cal Q }$ do   
19: Construct $X _ { q }$ for the given candidate list using the history visible   
before �.   
20: Run the adapted backbone under both views.   
21: Obtain $s _ { q } ^ { G }$ using prediction fusion and identical-descriptor averag   
ing as specified in Appendix D   
22: end for   
23: return $\{ s _ { q } ^ { G } \} _ { q \in Q } .$

## C Dynamic Interaction Prior

## C.1 Event Generation

For the source rates $\lambda _ { u } ( t )$ , let $\mu _ { u } > 0$ be baseline rates with $\begin{array} { r } { \sum _ { u } \mu _ { u } = } \end{array}$ $1 , \eta \ge 0$ the excitation strength, and $\beta > 0$ its decay rate. We use:

$$
\lambda _ { u } ( t ) = \mu _ { u } + \eta \beta \sum _ { i : u _ { i } = u , \ t _ { i } < t } e ^ { - \beta ( t - t _ { i } ) } .\tag{17}
$$

Starting from empty histories, we generate event times by thinning.

## C.2 Destination Mechanisms

For ${ g } _ { \mathrm { p r e f } }$ and $g _ { \mathrm { r e c } } ,$ let $c _ { u } , c _ { v }$ denote sampled community labels, � a positive community-afinity matrix, and $a _ { v }$ a positive destination propensity. With two sampled decay scales $h _ { 1 } , h _ { 2 } \ > \ 0$ and smoothing constant $\varepsilon = 1 0 ^ { - 9 }$ , their base scores are:

$$
\begin{array} { l } { { g _ { \mathrm { p r e f } } ( u , v , t ) = B _ { c _ { u } , c _ { v } } a _ { v } , } } \\ { { \displaystyle g _ { \mathrm { r e c } } ( u , v , t ) = \sum _ { \tau \in E _ { u v } } \sum _ { k = 1 } ^ { 2 } e ^ { - ( t - \tau ) / h _ { k } } + \varepsilon . } } \end{array}\tag{18}
$$

Variants of $g _ { \mathrm { p r e f } }$ incorporate destination-specific boosts, accumulated popularity, periodic modulation, or changes midway through the sequence.

Table 6: Task construction and training settings. Intervals refer to positions in the generated stream.  
Setting Value   
Stream length {3,000, 6,000, 12,000} events   
Positive context events {64, 128, 256, 512} with probabilities (0.3, 0.3, 0.3, 0.1)   
Context / query interval 30–65% / 65–95%   
Queries / candidates 8 / {32, 64, 128}, capped by the destination pool   
Optimizer / learning rates AdamW, 10<sup>−5</sup> for the predictor and 10<sup>−3</sup> for the adapter   
Updates / tasks per update 1,024 / 4, with 4,096 updates for the diversity study   
Random seeds Three   
Target context 512 positive events from the final 10% of the training   
interval, each paired with one negative

For $g _ { \mathrm { s e q } }$ in $\operatorname { E q } .$ (10), each row of � combines a weak background weighted by destination propensity with a few enhanced destinations, followed by normalization. Without a previous destination, $g _ { \mathrm { s e q } } = g _ { \mathrm { p r e f } } .$ . For $g _ { \mathrm { s t r } } .$ , we sample $S _ { u } ( t )$ uniformly without replacement from eligible other sources and add � to the structural score. We fall back to ${ g } _ { \mathrm { p r e f } }$ when closure is disabled or no eligible source or previous destination exists.

In Eq. (11), normalization before mixing uses the full destination pool, and repetition inhibition is clipped to exp $[ - \operatorname* { m i n } ( \gamma C _ { u v } ( t ) , 4 0 ) ]$ Self-links in homogeneous graphs are excluded before final normalization; bipartite graphs use a separate destination set.

## D Contextual Modeling and Training

## D.1 Tasks and Training

We construct T from E using the settings in Table 6, requiring all queries to occur strictly after the context events. Negative candidates exclude self-links and destinations observed for the source at the represented timestamp. Tasks alternate uniform negative sampling with a mixture drawing up to half the negatives from previously visited destinations. After tokenizing complete candidate lists, we retain one positive and one negative per context event.

We train the in-context predictor and adapter with Eq. (14), using positive-minus-negative class logits as training scores and freezing input encoders. One context-fitted backbone preprocessing view is sampled per task.

## D.2 Contextual Value Adapter

To specify ValueAdapter(·) in Eq. (12), we summarize the bounded memory-scale, connectivity, and transition features in $X _ { C }$ by their classwise means and standard deviations under y<sub>�</sub>. These statistics condition softmax-weighted combinations of the four pair and four destination memory features. The combined pair memories, their contrasts with destination memories, and the connectivity and transition features are standardized across context rows. A learned afine map followed by 1 + tanh produces a gate $\mathbf { g } _ { r } \in ( 0 , 2 ) ^ { 1 6 }$ for each context row �. Using h<sup>¯</sup><sub>�</sub>, the corresponding row representation from $\bar { H } _ { C }$ , the projected correction is:

$$
\begin{array} { r l } & { \Delta \mathbf { v } _ { r } = W _ { V } W _ { \mathrm { u p } } \left[ \left( W _ { \mathrm { d o w n } } \bar { \mathbf { h } } _ { r } \right) \odot \mathbf { g } _ { r } \right] , } \\ & { \Delta V _ { C } = [ \Delta \mathbf { v } _ { 1 } , \dots , \Delta \mathbf { v } _ { n _ { C } } ] ^ { \top } . } \end{array}\tag{19}
$$

Here, $W _ { \mathrm { d o w n } }$ and $W _ { \mathrm { u p } }$ project into and out of a 16-dimensional subspace, ⊙ denotes elementwise multiplication, and � is the backbone’s original value projection. This gives the correction added to $V _ { C }$ in the main text.

## D.3 Target-Graph Inference

Each target graph reuses its labeled context with fixed model parameters. The backbone averages aligned logits from two contextfitted preprocessing views, none and power, and applies binary softmax with temperature 0.9. Before ranking, we average the resulting probabilities among candidates with exactly identical 31-dimensional descriptors within each query. This assigns equal scores to tokenizer-indistinguishable candidates without using labels or removing candidate positions. Ties receive average ranks when computing MRR.

## E Overall Algorithm

Algorithm 1 connects sequence generation, task construction, and contextual prediction. All input tables follow Eq. (9), using only history strictly before each row’s timestamp. Candidate-relative ranks are computed before context-row selection. The predictor $F _ { \theta }$ includes the Contextual Value Adapter.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009