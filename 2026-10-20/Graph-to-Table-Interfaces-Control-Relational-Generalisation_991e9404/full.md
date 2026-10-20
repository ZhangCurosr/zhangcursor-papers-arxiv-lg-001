# Graph-to-Table Interfaces Control Relational Generalisation

Alessandro Cagiano University of Milan alessandro.cagiano@unimi.it

Gianluca Moro University of Bologna gianluca.moro@unibo.it

Lorenzo Molfetta University of Bologna lorenzo.molfetta@unibo.it

Matteo Zignani University of Milan matteo.zignani@unimi.it

Pasquale Minervini University of Edinburgh Miniml.AI p.minervini@ed.ac.uk

## Abstract

Graph-to-table systems combine three design choices: an encoder that maps relational evidence to feature rows, a context constructor that selects labelled examples, and a tabular decoder. We investigate whether these components can be evaluated independently through matched comparisons that change one component while holding the rest of the pipeline and the evaluation constant. Across logistic regression, CATBOOST, and TABPFN-3 on CLUTRR, 57 GraphLog worlds, and 57 knowledge graphs, we show that conclusions about one component can change when the others change. On CLUTRR, absolute positional encodings maintain all distinctions among the evaluated queries and fit the short-path setting nearly perfectly; yet, they generalise worse than relation-count features because the supported prefix acts as a shortcut that breaks down on longer paths. On GraphLog, encoder preference depends on the decoder, while the relation-count representation degrades as CatBoost tree depth increases. Across different knowl edge graphs, adding the frozen ULTRA score to local features is consistently useful, whereas adding local features to the score has a smaller and less stable effect. A pointwise TABPFN-3 readout using the full encoder underperforms native ULTRA on 48 of 57 datasets. These results show that encoder assessment depends jointly on context and decoder, while panel-level conclusions can also depend on evaluation and aggregation protocols.

## 1 Introduction

Prior-data fitted networks (PFNs) make predictions from a small labelled table without target-specific fine-tuning [10, 12]. Their use is becoming increasingly common in graph-to-table approaches to relational prediction. In this setting, a graph-to-table encoder ϕ maps each relational query and its surrounding graph evidence to a feature row, a context constructor S selects labelled rows from the target graph to form the prediction context, while a tabular decoder D infers the labels of held-out queries. We refer to the combination (ϕ, S, D) as the graph-to-table interface.

Recent work explores related approaches through graph-derived feature representations, graph-aware architectures, and learnt alignment between graph and tabular representations [5, 6, 8, 13, 20]. When such solutions are evaluated only through final predictive performance, it is difficult to determine where gains and failures originate. The three components of this interface impose distinct constraints. The encoder determines which relational distinctions are preserved in each query row. The context determines which of these distinctions are associated with the target labels. The decoder determine which dependencies between the resulting features and labels can be exploited. Consequently, preserving more detail does not by itself guarantee better generalisation. Patterns predictive in the labelled context may not generalise to held-out queries, and the decoder may change the relative ranking of encoders. End-to-end performance alone cannot distinguish among these effects.

We study how the encoder, context, and decoder jointly determine relational generalisation. We ask:

When does a graph-to-table representation generalise beyond its labelled context?

We investigate this question on controlled composition tasks and on a 57-knowledge-graph panel spanning multiple decoders, context constructions, and graph-derived encoders.<sup>1</sup>

Our findings are as follows:

• We use paired diagnostics to determine whether an encoder preserves target-relevant features, whether those features transfer from the labelled support examples to held-out queries, and whether the decoder can utilise them.

• We show on CLUTRR that absolute positions exploit a prefix shortcut that fails on longer paths, and on GraphLog that encoder preference changes across decoders and is sensitive to tree capacity.

• On 57 knowledge graphs, we find that adding the frozen ULTRA score to local features is consistently useful, whereas adding local features to the score has a smaller and less stable effect. Native-ranking results depend on panel weighting, while encoder effects depend on context construction.

## 2 Related Work

Graph–table foundation models. Building on tabular in-context learning with TABPFN and TABPFN-3 [10, 12], several methods expose graph structure to tabular predictors. TabGFM [11] augments node attributes with graph-derived structural features, whereas G2T-FM [5] constructs tabular representations through neighbourhood aggregation and structural encodings. GTAlign [13] instead learns an explicit alignment between graph and tabular representations. These approaches establish interfaces between graph and tabular representations, while our focus is to isolate which query-level relational information must be preserved for link-level transfer.

In-context link prediction. In-context link prediction adapts a shared predictor to a target graph through labelled demonstrations, keeping the predictor’s parameters fixed. UniLP [4] reasons directly over graph neighbourhoods, TFMLinker [20] represents candidate links as topology-aware rows with local and global contexts, and KGPFN [8] combines transferable relation representations with query-local features and global context. These methods differ in both the underlying predictor and where relational evidence is introduced: through graph encoding, row construction, or demonstration selection. Aggregate performance therefore leaves the contribution of each component unresolved. We address this distinction by varying path encoding and labelled context independently under a fixed evaluation protocol.

Length generalisation and contextual shortcuts. CLUTRR [24] evaluates relational composition over held-out kinship rules and chain lengths, whereas GraphLog [25] evaluates logical generalisation across graph worlds governed by different relational rules. Such settings instantiate compositional generalisation [15], requiring learned components to remain productive under unseen combinations. For Transformer models, length extrapolation is likewise sensitive to positional representation [18]. These sensitivities relate to shortcut learning, where predictive regularities learned in one regime fail under distributional shift [9]. Within tabular in-context learning, VIP-COP [1] studies the selection of informative rows and features under a limited context budget, while OpenRFM [2] identifies relation-level label coverage as a critical factor for relational inference. We examine these sources of failure jointly by evaluating across path lengths while independently controlling the encoder and labelled context.

Systematic component analyses. Several studies show that model comparisons can change substantially when architecture, training, and evaluation choices are controlled separately. In knowledge graph completion, Ruffinelli et al. [23] systematically study the effects of model architecture, training strategy, and hyperparameter optimisation, while changes in the training regime can substantially alter comparisons among knowledge graph embedding models [16]. In graph learning, GraphGym [29] studies a modular graph neural network (GNN) design space under a unified evaluation setup, separating architectural choices that are often compared only through complete models. Our work follows this controlled-comparison perspective, but applies it to graph-to-table prediction by varying the encoder, labelled context, and decoder while keeping the remaining pipeline and evaluation protocol fixed.

![](images/816f72429165f313b94a248d9f56331aa753907fe27f275f2a9febc83cf16a14.jpg)  
Figure 1: Graph-to-table prediction pipeline. Given a relational query and its surrounding graph, the encoder ϕ constructs a feature row, the context constructor S assembles labelled examples from the target graph, and the tabular decoder D produces prediction scores for the held-out query.

Representation expressivity and distinguishability. Relational expressivity has been studied through supported relational patterns and structural motifs [14, 27], while Weisfeiler–Leman analyses characterise graph representations through structures they cannot distinguish [22, 28]. Our rowequivalence diagnostic applies the same principle after graph-to-table conversion. If two labelled queries induce identical rows within the same prediction context, they are necessarily indistinguishable to the tabular decoder. The converse does not hold: distinguishable encodings merely remove this representational obstruction, since successful transfer additionally depends on preserving task-relevant information and maintaining adequate source–target support [17].

Inductive and zero-shot knowledge graph reasoning. Inductive knowledge graph methods use transferable structural computations, including enclosing subgraphs in GraIL [26], path-based reasoning in NBFNet [31], and relation graphs in InGram [19]. ULTRA [7] extends structural reasoning to zero-shot transfer across knowledge graphs with previously unseen relation vocabularies, while TRIX [30] develops more expressive fully inductive triplet representations for entity and relation prediction. Our setting keeps ULTRA frozen and exposes its candidate score as a tabular feature, testing whether local graph evidence adds information beyond the native score and whether in-context tabular decoding improves the resulting ranking.

## 3 Graph-to-Table Diagnostic Framework

A graph-to-table pipeline comprises three components: an encoder ϕ that maps graph evidence to a feature row, a context constructor S that assembles labelled rows, and a decoder D that predicts from the resulting context (see Figure 1). End-to-end performance conflates the effects of these components. We therefore organise the analysis around four diagnostic questions concerning row distinguishability, feature alignment, decoder dependence, and query-level ranking effects. Each question either establishes an encoder-induced ceiling or compares matched conditions while holding the remaining pipeline and evaluation protocol fixed.

## 3.1 Problem Formulation

Let $G = ( V , R , E )$ be an observed relational graph, where V is the set of entities, R is the set of relation types, and $E \subseteq V \times R \times V$ is the set of observed edges. A candidate query $q = ( h , r , t )$ associates a head entity $h \in V$ and a tail entity $t \in V$ through relation $r \in R .$ . Let ${ \mathcal { G } } , { \mathcal { Q } } , { \mathcal { X } }$ , and Y denote the graph, query, feature-row, and label spaces, respectively. The encoder

$$
\phi : \mathcal { G } \times \mathcal { Q }  \mathcal { X }
$$

maps a graph and query to a feature row $x = \phi ( G , q )$ . Given a pool of labelled rows and a requested budget $n \in \mathbb { N } , \mathrm { i . e . }$ ., the number of labelled rows to include per class, the context constructor

$$
S : ( x \times y ) ^ { * } \times \mathbb { N } \to \mathcal { C }
$$

assembles a labelled context $S _ { n }$ , where C denotes the space of possible contexts. Finally, the decoder

$$
D : \mathcal { C } \times \mathcal { X } \to \mathbb { R } ^ { | \mathcal { V } | }
$$

maps the labelled context and a query row to class scores. We obtain a predicted label from the highest-scoring class, while in binary link prediction we can use the positive-class score for candidate ranking. For example, in our path-based experiments, ϕ may encode a relational path through relation counts or absolute positions, S selects labelled rows from the target graph, and D is instantiated by logistic regression, CatBoost, or TabPFN-3. The resulting pipeline can be written as

$$
\begin{array} { r } {  { \mathcal { S } } _ { n } =  { S } \big ( \{ ( \phi ( G , q _ { i } ) , y _ { i } ) \} _ { i \in \mathbb { Z } } , n \big ) , \qquad { \mathbf { s } } = D (  { \mathcal { S } } _ { n } , x ) , \qquad \hat { y } = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } s _ { y } . } \end{array}\tag{1}
$$

## 3.2 Diagnostic Questions

Ã Does the row distinguish queries with different labels? Before considering any particular decoder, we ask whether the encoder maps queries with different labels to distinguishable rows. For a fixed graph $G ,$ , the encoder partitions queries according to row identity: we write $q \sim _ { \phi } q ^ { \prime }$ if and only $\mathbf { i f } \ \phi ( { \bar { G } } , { \dot { q } } ) = \phi ( G , q ^ { \prime } )$ . Queries in the same equivalence class expose the same row to the decoder and, with the context and remaining inputs fixed, cannot be distinguished from that row alone. To quantify this representational limitation independently of the decoder, we compute an oracle accuracy ceiling on the finite set of evaluated queries, each paired with its label. Let N be the number of such queries, $n _ { x y }$ the number of queries mapped to row x with label $y ,$ and $\begin{array} { r } { n _ { x } = \sum _ { y } n _ { x y } } \end{array}$ the number of queries mapped to row x. Any predictor that observes only x must make the same prediction for all examples sharing that row, and can therefore classify at most ma ${ \bf x } _ { y } n _ { x y }$ of them correctly. The resulting empirical ceiling is

$$
A ^ { \star } ( \phi ) = \frac { 1 } { N } \sum _ { x } \operatorname* { m a x } _ { y } n _ { x y } = \sum _ { x } p ( x ) \operatorname* { m a x } _ { y } p ( y \mid x ) ,\tag{2}
$$

where $p ( x )$ and $p ( y \mid x )$ are empirical frequencies. Under zero–one loss, this is the empirical Bayes optimal accuracy obtained by predicting the majority label within each row class [3]. Evaluation labels are used only to quantify ambiguity introduced by the encoder. Successful transfer additionally requires the decoder to infer an appropriate rule from the available labelled context.

A ceiling of one does not necessarily imply that the representation resolves repeated query patterns: it may also arise because every evaluated query maps to a unique row. We therefore report the singleton mass

$$
\sigma _ { \phi } = \sum _ { x } p ( x ) \mathbf { 1 } [ n _ { x } = 1 ] ,\tag{3}
$$

where $n _ { x }$ is the number of queries mapped to row x. A value of $\sigma _ { \phi } = 1$ indicates that every evaluated query occupies a singleton equivalence class. Since singleton rows are necessarily assigned a ceiling accuracy of one in Equation (2), they can mask ambiguity among rows that occur more than once. We therefore condition the ceiling on repeated rows:

$$
A _ { \mathrm { n s } } ^ { \star } ( \phi ) = \frac { \sum _ { x : n _ { x } > 1 } p ( x ) \operatorname* { m a x } _ { y } p ( y \mid x ) } { 1 - \sigma _ { \phi } } , \qquad A ^ { \star } ( \phi ) = \sigma _ { \phi } + ( 1 - \sigma _ { \phi } ) A _ { \mathrm { n s } } ^ { \star } ( \phi ) .\tag{4}
$$

The conditional quantity is undefined when $\sigma _ { \phi } = 1$ ; it measures the maximum row-only accuracy among examples belonging to repeated equivalence classes, separating ambiguity within repeated rows from the trivial contribution of singleton rows.

For GraphLog, evaluation uses macro balanced accuracy, which weights every class equally. For a row x, predicting class y contributes $n _ { x y } / n _ { y }$ to the class-normalised number of correct predictions. The optimal row-only prediction therefore chooses the class maximising this quantity, yielding the corresponding class-reweighed ceiling:

$$
B ^ { \star } ( \phi ) = \frac { 1 } { K } \sum _ { x } \operatorname* { m a x } _ { y } \frac { n _ { x y } } { n _ { y } } ,\tag{5}
$$

where K is the number of observed classes, $n _ { x y }$ counts examples with row x and label $y ,$ and $n _ { y }$ is the number of examples in class $y .$

$\pmb { \mathcal { O } }$ Does feature alignment improve predictions on unseen queries? Let $b ( q )$ denote an additional block of related feature columns associated with query q. We compare three matched conditions: (i) the base encoder excludes the block, (ii) the permuted encoder includes the same collection of complete blocks but permutes their correspondence to queries, and (iii) the aligned encoder restores the original query-to-block correspondence. The base–permuted comparison measures the effect of adding the block without preserving its query alignment. The permuted–aligned comparison measures the effect of restoring that alignment while holding fixed the block dimensionality, the collection of block values, context construction, and decoder. This comparison separates sensitivity to query–feature alignment from the mere presence of the additional feature block.

Ô Does encoder preference change across decoders? For two encoders $\phi _ { 1 }$ and $\phi _ { 2 } .$ , we compare their performance under the same context construction and evaluation queries. For a decoder D, let

$$
\Delta _ { D } = M ( \phi _ { 1 } , S , D ) - M ( \phi _ { 2 } , S , D ) ,
$$

where M denotes the task metric: a change in the sign of $\Delta _ { D }$ across decoders indicates a reversal in the relative ranking of the two encoders. We also inspect each encoder’s absolute performance, since a reversal can arise from degradation in one representation or improvement in the other.

 For which query regimes does graph evidence improve ranking? Aggregate ranking metrics may conceal heterogeneous effects across query types. We therefore partition evaluation queries using graph evidence observable before prediction, such as the presence of direct or reverse edges between the query endpoints. Within each stratum, we compare reciprocal rank across encoder variants on the same queries. These comparisons keep the context construction, decoder, candidate set, filtering protocol, and random seed fixed. The resulting stratification identifies the query regimes in which particular graph-derived feature groups are associated with ranking improvements.

## 4 Experimental Design

We evaluate the diagnostic framework in three regimes: synthetic calibration, controlled compositional extrapolation, and heterogeneous cross-graph ranking. Whenever the decoder D is varied, logistic regression, CATBOOST, and frozen TABPFN-3 receive the same queries, labels, context examples, and feature values before decoder-specific categorical encoding. Decoder configurations remain fixed within each matched comparison.

## 4.1 Calibration and Compositional Evaluation

Controlled calibration. We generate four binary tasks governed by reciprocity, typed relation composition, local structural roles, and shared graph-community membership. For each task, ten independently generated graphs define disjoint context and evaluation populations. Because the relevant relational distinctions are known by construction, these tasks validate the row-equivalence ceiling, singleton mass, and matched-permutation diagnostics before applying them to benchmark data. Appendix B provides the complete generation procedure and results.

In CLUTRR, GraphLog, and the dihedral control, each row represents a supplied or generated symbolic sequence. The encoders differ only in which properties of that sequence are exposed to the decoder.

CLUTRR. We use the official gen-train23-test2to10 split [24]. Contexts contain training paths of lengths 2–3 with nested budgets of 5, 10, 20, 40, 80 examples per target relation, while evaluation covers lengths 4–10. We compare three encoders of the same symbolic path: a relation-multiset encoder, recording path length and relation counts; an absolute-position encoder, preserving the relation at each position; and a directed-transition encoder, counting ordered adjacent relation pairs without absolute positions. For the alignment test, we append the absolute-position block to the relation-multiset row and compare the correctly aligned block with a condition in which complete blocks are permuted among paths of equal length. This isolates the effect of query–feature alignment while preserving the block dimensionality and collection of values. A separate suffix ablation zeros absolute-position coordinates beyond position 3, testing whether unsupported positions contribute to the observed behaviour. To isolate length coverage from context volume, the same 80-example length 2–3 base context receives either 20 further short examples or 20 length-4 examples, yielding 100 examples per target relation in both conditions. Appendix C.1 reports the schemas, dimensionalities, and split statistics.

GraphLog. We use the official symbolic path descriptors from all 57 worlds [25]. Contexts contain paths of lengths 2–3 with nested budgets of 5, 10, 20, 40 examples per target class, while evaluation uses all test paths of lengths 4–5. All worlds share the same vocabulary of 20 directed relation symbols but differ in their relational rules. We compare relation-multiset and absolute-position encoders. Neither includes entity identifiers or the target relation, so the comparison isolates how the supplied symbolic path is represented. Performance is measured using macro balanced accuracy together with the corresponding class-balanced row-equivalence ceiling. Appendix C gives the complete schemas, alignment controls, and decoder sensitivities.

Dihedral boundary control. The dihedral control contains ten independently generated populations over the 16 symmetries of a regular octagon. Context sequences have lengths 2–3, evaluation sequences have lengths 4–10, and the target is their ordered group product. We compare an absoluteposition encoder with a coarse encoder that removes sequence order and merges group elements separated by four rotations. This provides a boundary case in which coarsening discards task-relevant information, testing whether the preference observed for a coarser representation on CLUTRR persists when the removed information is required by the target.

## 4.2 Knowledge Graph Panel

Graphs and ranking protocol. The panel contains all 57 knowledge graphs, grouped into 16 provenance families, with ULTRA [7] frozen throughout. For each graph, we sample 64 official test triples stratified by relation and evaluate both head and tail prediction by ranking all entities under standard filtered evaluation. For every candidate entity, the resulting candidate triple is mapped by the encoder to one feature row. All graph-derived features are computed from the observed support graph, which excludes the held-out target triples, and context positives are disjoint from the sampled evaluation queries. This keeps the candidate-ranking problem fixed while varying the graph evidence exposed to the tabular decoder.

Encoder variants. We evaluate the nine encoder variants summarised in Appendix Table 12. They include Entity tokens, exposing dataset-scoped endpoint tokens; Dyadic, based on evidence directly connecting the query endpoints; and Paths, based on two-hop path evidence. The broader Local encoder combines relation identity with endpoint, pair, path, and community statistics, whereas Global exposes graph-wide topological and spectral information. ULTRA score exposes only the candidate score produced by the frozen graph-native ULTRA model. Composite encoders concatenate these fixed blocks as Local + Global, Local + ULTRA score, and Full. The complete encoder ladder is evaluated with nested context budgets of 20, 50, 100, 200 examples per binary class. Its baseline negatives are endpoint corruptions without an all-known-positive rejection step. Appendix D separately isolates the effect of this filtering choice with a matched control. Whenever the frozen ULTRA score is included, we also report the native ULTRA candidate ranking as a reference.

Pointwise-ranking control. The tabular decoders are trained with a pointwise binary objective but are evaluated by filtered ranking within each directional query. At 200 examples per class, we therefore cross the Local, ULTRA score, and Local + ULTRA score encoders with raw, within-query z-score, and within-query percentile representations of their numerical columns. The transformations are label-free and applied by the same rule to context groups and evaluation candidate blocks. This tests whether query-relative normalisation changes the behaviour of the pointwise readouts under an otherwise fixed ranking protocol. Appendix D.2 gives the grouping and results.

Negative-context construction. For the negative-context comparison, we fix the budget at 200 examples per class and vary only the context constructor S, while keeping context positives and evaluation queries fixed. Filtered-uniform negatives replace the missing endpoint with a uniformly sampled entity outside the known-positive set. Structurally plausible negatives favour entities observed in the appropriate relation role and, when possible, in the same graph partition as the fixed endpoint, followed by degree similarity. Selection uses only graph-based criteria, so the comparison changes the difficulty and structure of the labelled negative context without using evaluation labels or the frozen ULTRA score. Appendix D gives the complete construction and audit checks.

Positive-context construction. We additionally compare pool-random positives with evaluationrelation-matched positives while keeping context size and evaluation queries fixed. The latter selects positives to better match the relation distribution represented by the evaluation queries, allowing us to vary relation-level support without changing the evaluation set.

## 4.3 Statistical Analysis

For the 57-KG panel, the metric is averaged over queries and seeds within datasets, then over datasets within provenance families, with families weighted equally. We refer to this estimand as the equal-family metric and additionally report equal-dataset aggregation as a sensitivity analysis. Provenance families are also the unit of inference: bootstrap sensitivity intervals resample families with replacement, and the exact max-|t| multiple-comparison correction in Appendix D.8 flips the sign of each family’s mean paired effect as a block. Because the panel is fixed rather than sampled from a population of graphs, we read bootstrap intervals as sensitivity to panel composition rather than as confidence intervals, and leave-one-family-out ranges show how much each estimate depends on any single family. For CLUTRR, effects are paired across context seeds on the same official split. GraphLog comparisons are paired across worlds under shared matched conditions. The dihedral control uses the independently generated populations as replication units. Query-level ranking analyses stratify the same evaluation queries by graph evidence observable before prediction and report reciprocal rank within each stratum.

## 5 Results

## 5.1 Aligned Prefixes Form a Shortcut Under Length Shift

At 80 context examples per target relation, the absolute-position encoder reaches 0.156/0.109/0.103 accuracy for logistic regression, CATBOOST, and TABPFN-3, whereas the relation-multiset encoder reaches 0.429/0.272/0.289. The absolute-position ceiling of 1.000 follows from its singleton mass of 1.000 and therefore carries no information about ambiguity among repeated rows. For the relation multiset, the raw ceiling is 0.930, while the repeated-row mass is 0.243 and its conditional ceiling is 0.712. The lower absolute-position accuracy therefore occurs despite injective rows, while its perfect ceiling follows from that injectivity. The multiset advantage also holds across all declared metrics and for at least 15 of the 18 target relations (Appendix C.1, Tables 3 to 6).

Figure 2 reports the matched alignment and length-exposure interventions under the short-to-long support shift. Relative to the relation-multiset encoder, adding an absolute-position block permuted among paths of equal length changes accuracy by only $- 0 . 0 3 0 / - 0 . 0 \dot { 1 } 7 / + 0 . 0 0 2$ for logistic regression, CATBOOST, and TABPFN-3. Restoring the correct path-level alignment reduces accuracy by a further 0.133/0.092/0.104. Context paths have lengths 2–3, so positions 4–10 have no labelled support. Evaluation paths have lengths 4–10. Setting these unsupported suffix coordinates to zero changes predictions by less than $3 \times 1 0 ^ { - 1 7 }$ . The aligned encoders reach 0.990–0.995 accuracy on the labelled context rows, while transferring substantially worse to longer paths. These matched interventions therefore support a shortcut interpretation in which alignment exposes a highly predictive supported prefix that does not transfer to longer paths.

The length-exposure intervention produces a similarly local effect. Replacing 20 additional short examples with 20 length-4 examples improves length 4 by +0.489/+0.372/+0.758, but changes lengths 5–10 by only -0.001/+0.010/+0.033. The directed-transition encoder reaches 0.408/0.285/0.156, with its ordering relative to the multiset changing across decoders. A shared-parameter gated recurrent unit (GRU) with 32-dimensional relation embeddings, 64 hidden units, one recurrent layer, and context-only early stopping improves from 0.346 to 0.705 OOD accuracy across the five context budgets. At the largest budget it decreases from 0.898 at length 4 to 0.527 at length 10, with aggregate accuracy 0.705. The GRU serves as an existence control for the split, showing that substantially stronger short-to-long transfer is attainable. Together, these results support a shortcut interpretation for the tested tabular interfaces under the known length-support shift.

## 5.2 A Decoder-Dependent Failure Regime

GraphLog reverses the encoder ordering observed on CLUTRR for the nonlinear decoders. The performance difference between the relation multiset and absolute positions is +0.065 for logistic regression, -0.157 for CATBOOST, and -0.079 for TABPFN-3. Complete absolute levels are reported in Table 7. Both encoders retain high class-balanced ceilings of 0.965 and 0.983, while every world uses the same global vocabulary of 20 directed relations. Dimensionality and coordinate meaning are therefore fixed across worlds, enabling matched comparison of the two representations across decoders.

(a) OOD accuracy  
![](images/6fb80f682bb504f863319922c16cea45ceaaa1ec47ec1aa0647201ad22c7dc2f.jpg)

(b) Paired changes  
![](images/fc89b65d97ee6676b2e3f32ec7785914b51187da8a02ccc6e51197df4e90775d.jpg)

(c) Length exposure  
![](images/ecdddb51bdf8095423af89190067beaca1d11f81b5a653b5cf23e8758b19ed14.jpg)  
Figure 2: Effect of path-feature alignment and length exposure on CLUTRR. (a) Out-ofdistribution (OOD) accuracy for the relation multiset alone, with a permuted absolute-position block, and with the correctly aligned block. (b) Paired accuracy changes from adding the permuted block and restoring its path-level alignment. (c) Accuracy change after replacing 20 additional length-2–3 examples with 20 length-4 examples, evaluated at length 4 and lengths 5–10. Panels (a) and (b) use 80 context examples per target relation and evaluation paths of lengths 4–10, while panel (c) uses the same 80-example base plus 20 additional examples, for 100 examples per target relation in both arms. In (b), positive block-addition values indicate higher OOD accuracy after adding the permuted block, while positive alignment values indicate accuracy lost after restoring the correct path-level alignment. In (c), positive values indicate higher OOD accuracy after adding length-4 context examples. Bar labels are rounded to three decimals.

(a) CLUTRR  
![](images/4c1137af0573c269226c6ad0459367bb39ce398e68d296e17090c01330115f90.jpg)  
Performance difference

(b) GraphLog  
![](images/6e9ab1add744a801cc65b60e0e45985822840bfea451ee7abe1faa66555917c3.jpg)  
Performance difference

(c) Dihedral composition  
![](images/d0ea6b15d823e48e1407f43a25b29dba571d29a720bb8c74bf67e19312f01bef.jpg)  
Figure 3: Encoder effects across decoders. At the largest context budget, values report relation multiset minus absolute-position performance for (a) CLUTRR and (b) GraphLog, and coarse minus absolute-position performance for (c) the dihedral control. CLUTRR and the dihedral control use accuracy, while GraphLog uses macro balanced accuracy. Positive values favour the coarser encoder. Bars in (a) and (c) denote means, with capped intervals spanning paired context seeds and generated populations, respectively. In (b), light points denote paired worlds, diamonds denote means, and capped intervals denote sensitivity intervals. GraphLog world-level wins/ties/losses are 41/0/16 for logistic regression, 6/0/51 for CATBOOST, and 11/0/46 for TABPFN-3. Panels use separate horizontal scales.

For CATBOOST and TABPFN-3, absolute positions yield higher balanced accuracy than the relation multiset, while the multiset enters a failure regime in many worlds. At 40 examples per class, the multiset is at or below each world’s chance level in 30 and 12 of the 57 worlds for CATBOOST and TABPFN-3, compared with 1 and 4 worlds for absolute positions. To characterise this regime, a CATBOOST tree-depth sweep uses all 57 worlds and holds all other settings fixed across depths 1–7.

![](images/40a0980e72969d2e51739e860af585d2bc38e5fff5ce9f6e2dec54445fc239a4.jpg)  
Figure 4: Paired encoder effects across 57 knowledge graphs. Equal-family ∆MRR at 200 examples per class for (a) local features minus dataset-scoped entity identifiers, (b) local plus score minus local, and (c) local plus score minus score. Light-brown bars denote filtered-uniform negatives and dark-brown bars denote structurally plausible negatives. Capped error bars show 95% provenancefamily bootstrap sensitivity intervals; positive values favour the first encoder in each contrast. Positive context examples and evaluation queries are fixed across the two negative-context constructions.

Evaluation balanced accuracy changes from 0.201/0.271 at depth one to 0.125/0.282 at depth seven, widening the gap from -0.070 to -0.157. Over the same sweep, context fit rises from 0.830/0.910 to 0.948/0.991. The complete curve in Appendix C shows progressively lower multiset evaluation performance as tree depth increases, while absolute-position performance remains comparatively stable. Together, these results show that the GraphLog encoder ordering changes across decoders and that the CATBOOST gap is depth-sensitive, driven mainly by multiset degradation.

The dihedral control limits how broadly the CLUTRR ordering can be interpreted. Its coarse encoder removes sequence order while also merging distinct group elements. At chance level 0.0625, coarse/absolute accuracies are 0.0618/0.0622, 0.0770/0.0643, and 0.0614/0.0626. When coarsening also removes task-relevant identity, both encoders therefore remain at or near chance apart from the small CATBOOST contrast.

## 5.3 Feature Contributions Are Asymmetric and Context-Dependent

Across the 57-KG panel, adding the frozen ULTRA score to the Local encoder improves equal-family mean reciprocal rank (MRR) by +0.048–+0.129 across all decoder–context combinations. Five of the six contrasts remain supported after joint correction. The structurally plausible TABPFN-3 contrast remains positive but falls just outside the correction threshold. The reverse addition, adding Local features to the score, ranges from −0.039 to +0.064, and none of the six contrasts remains supported after correction. As a reference comparison, Local also outperforms dataset-scoped entity identifiers by +0.177–+0.262 across all six combinations. Removing relation identity from Local changes equalfamily MRR by at most 0.0031 in magnitude, and none of the tested contrasts remains supported after joint correction (Appendix D.4). Figure 4 summarises these three matched comparisons.

The native ranking provides a rank-preservation sanity check. Score-only logistic regression reproduces the native ULTRA ranks exactly across all 684 combinations of dataset, context budget, and seed, as expected when the one-dimensional frozen score is read out with positive orientation. For TABPFN-3, adding global PageRank, component, heat, and diffusion features after the score changes MRR by only +0.005 ([-0.001,+0.010]), with the contrast falling below the joint-correction threshold. TABPFN-3 with the Full encoder underperforms native ULTRA on 48 of 57 datasets and 14 of 16 provenance families. Its full-minus-native contrast is +0.006 under equal-family weighting but -0.035 under equal-dataset weighting. The corresponding medians are -0.034 and -0.039. The leaveone-family-out range is [-0.041,+0.015], with the minimum obtained by removing MetaFam. The change between equal-family and equal-dataset aggregation shows that this panel-level comparison is sensitive to weighting.

Because the tabular decoders are pointwise while evaluation ranks candidates within each query, we test their sensitivity to label-free within-query z-score and percentile transformations. These improve local-only logistic regression by +0.081/+0.076 MRR, but yield negative point estimates for every tested CATBOOST and TABPFN-3 transformation. All six CATBOOST reductions and five of the six TABPFN-3 reductions remain supported after joint correction. Score-only logistic regression preserves all 21,888 native query ranks exactly under both maps. Query-relative scaling therefore affects the tested pointwise readouts in a decoder-specific way. The analysis does not alter the underlying ranking objective (Appendix D.2). A 512-query sensitivity preserves the sign of all five panel-level encoder contrasts, with mean changes no larger than 0.0064 MRR (Appendix D).

Changing only the negative context also affects performance. Structurally plausible negatives reduce MRR by 0.020–0.122 across all nine informative encoder–decoder combinations, whereas identifier baselines change by less than 0.001. The selector uses only graph-based criteria, yet its selected negatives have a mean frozen score of −1.238, compared with −6.929 under filtered-uniform sampling, consistent with a harder negative distribution under frozen ULTRA. Decoder ordering also varies with the encoder: logistic regression is strongest for the score-only row, whereas TABPFN-3 is strongest for the local and local-plus-score rows (Appendix D).

A separate matched control isolates all-known-positive rejection from the other differences between the two context constructions. Filtering changes equal-family MRR by +0.0081 for Local, -0.0007 for Local + score, and +0.0022 for score-only TABPFN-3, with all three contrasts below the jointcorrection threshold. The matched control therefore indicates that filtering accounts for only a small part of the differences between the two context constructions (Appendix D.7). We also vary positivecontext selection. Matching positives to the evaluation relation distribution increases relation-type coverage from 0.667 to 0.871 and relation mass from 0.693 to 0.891, while reducing distributional total variation from 0.447 to 0.190. The corresponding MRR effects are generally small. Only CATBOOST with the Local encoder without relation identity remains supported after joint correction (+0.0139; Appendix D.3).

## 5.4 Local Feature Utility Varies with Endpoint Connectivity

Aggregate MRR conceals substantial differences among query regimes. We stratify the 7,296 directional queries by observed endpoint connectivity in the support graph, excluding the held-out target edge. 1,622 have a direct or reverse endpoint-pair edge.

Appendix Figure 8 shows that dyadic features achieve higher aggregate reciprocal rank than boundedpath features in each endpoint-connected stratum. The ordering reverses for the 5,674 queries without a pair edge. Dyadic reciprocal rank falls to 0.001–0.002, whereas path features reach 0.045–0.052 and the Local encoder reaches 0.043–0.094. The broader Local encoder therefore retains higher ranking performance for the majority of queries where direct pair evidence is absent.

The both-direction and direct-only groups contain 244 and 414 queries, so we treat their absolute levels as fixed-panel descriptions. Complete counts are reported in Appendix D.10, and the descriptive, order-dependent variance decomposition in Appendix E. The stratification therefore identifies query regimes in which different graph-derived feature groups are associated with higher ranking performance.

## 6 Conclusion

We introduced a paired diagnostic framework for the graph-to-table encoder ϕ, context constructor S, and decoder D. On CLUTRR, preserving query distinctions is insufficient for length transfer: unsupported absolute coordinates have negligible influence, while the aligned representation achieves near-perfect accuracy on the labelled context rows yet transfers worse to longer paths. On GraphLog, encoder preference changes across decoders, and a CATBOOST tree-depth sweep shows that the gap between absolute positions and relation counts widens primarily through degradation of the latter. Across 57 knowledge graphs, adding a frozen ULTRA score to local features provides consistent gains, whereas adding local features to the score has a smaller and less stable effect. A pointwise TABPFN-3 readout with the Full encoder underperforms native ULTRA on 48 of 57 datasets, while its aggregate comparison changes from positive under equal-family weighting to negative under equal-dataset weighting. Changing the labelled context also changes the measured contribution of the same encoder. Together, these results show that graph-to-table representations cannot be assessed through the encoder alone. Their behaviour depends jointly on the labelled context, decoder, evaluation protocol, and panel aggregation.

## References

[1] Yilong Chen, Xueying Ding, and Leman Akoglu. VIP-COP: Context optimization for tabular foundation models, 2026. URL https://arxiv.org/abs/2605.12904. 2

[2] Zhikai Chen, Junyu Yin, Jialiang Gu, Siheng Xiong, Xiaoze Liu, Ruowang Zhang, Keren Zhou, and Kai Guo. OpenRFM: Dissecting relational In-Context learning, 2026. URL https: //arxiv.org/abs/2606.04320. 2

[3] Luc Devroye, László Györfi, and Gábor Lugosi. A Probabilistic Theory of Pattern Recognition, volume 31 of Stochastic Modelling and Applied Probability. Springer, 1996. doi: 10.1007/ 978-1-4612-0711-5. 4

[4] Kaiwen Dong, Haitao Mao, Zhichun Guo, and Nitesh V. Chawla. Universal link predictor by In-Context learning on graphs. Trans. Mach. Learn. Res., 2025, 2025. URL https: //openreview.net/forum?id=EYpqmoejB8. 2

[5] Dmitry Eremeev, Gleb Bazhenov, Oleg Platonov, Artem Babenko, and Liudmila Prokhorenkova. Turning tabular foundation models into graph foundation models. In NeurIPS Workshop on New Perspectives in Graph Machine Learning, 2025. URL https://openreview.net/forum? id=KyiooU4ECI. Poster. 1, 2

[6] Dmitry Eremeev, Oleg Platonov, Gleb Bazhenov, Artem Babenko, and Liudmila Prokhorenkova. GraphPFN: A Prior-Data fitted graph foundation model. In ICML, volume 306 of Proceedings of Machine Learning Research, pages 28149–28165. PMLR, 2026. URL https://proceedings. mlr.press/v306/eremeev26a.html. 1

[7] Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation models for knowledge graph reasoning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=jVEoydFOl9. 3, 6

[8] Yisen Gao, Jiaxin Bai, Haoyu Huang, Zhongwei Xie, Yufei Li, Hong Ting Tsang, Sirui Han, and Yangqiu Song. KGPFN: Unlocking the potential of knowledge graph foundation model via In-Context learning, 2026. URL https://arxiv.org/abs/2605.14907. 1, 2

[9] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665–673, 2020. doi: 10.1038/s42256-020-00257-z. 2

[10] Léo Grinsztajn, Klemens Flöge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Manium, Shi Bin Hoo, Magnus Bühler, Anurag Garg, Dominik Safaric, Jake Robertson, Benjamin Jäger, Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp Singer, Alan Arazi, Julien Siems, Jan Hendrik Metzen, Georg Grab, Nick Erickson, Siyuan Guo, Eliott Kalfon, Simon Bing, David Salinas, Clara Cornu, Lilly Charlotte Wehrhahn, Diana Kriuchkova, Kursat Kaya, Lydia Sidhoum, Marie Salmon, Jerry Chen, Madelon Hulsebos, Yann LeCun, Samuel Müller, Bernhard Schölkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. TabPFN-3: Technical report, 2026. URL https://arxiv.org/abs/2605.13986. 1, 2

[11] Adrian Hayler, Xingyue Huang, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Michael M. Bronstein, and Ben Finkelshtein. Of graphs and tables: Zero-Shot node classification with tabular foundation models. In NeurIPS Workshop on New Perspectives in Graph Machine Learning, 2025. URL https://openreview.net/forum?id=u3vyZf5Jv2. Oral presentation. 2

[12] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 2025. doi: 10.1038/s41586-024-08328-6. 1, 2

[13] Chunyu Hu, Tianyin Liao, Ge Lan, Xingxuan Zhang, Jianxin Li, Peng Cui, and Ziwei Zhang. Surprisingly simple and effective Multi-Domain graph foundation model through Graph-to-Table alignment, 2026. URL https://arxiv.org/abs/2607.11374. 1, 2

[14] Xingyue Huang, Pablo Barcelo, Michael M. Bronstein, Ismail Ilkan Ceylan, Mikhail Galkin, Juan L. Reutter, and Miguel Romero Orth. How expressive are knowledge graph foundation models? In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of

Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025. URL https: //proceedings.mlr.press/v267/huang25a.html. 3

[15] Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. Compositionality decomposed: How do neural networks generalise? Journal ofArtificial Intelligence Research, 67:757–795, 2020. doi: 10.1613/jair.1.11674. 2

[16] Prachi Jain, Sushant Rathi, Mausam, and Soumen Chakrabarti. Knowledge base completion: Baseline strikes back (again), 2020. URL https://arxiv.org/abs/2005.00804. 2

[17] Fredrik D. Johansson, David Sontag, and Rajesh Ranganath. Support and invertibility in Domain-Invariant representations. In Kamalika Chaudhuri and Masashi Sugiyama, editors, The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pages 527–536. PMLR, 2019. URL https://proceedings.mlr.press/v89/ johansson19a.html. 3

[18] Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL https://openreview.net/forum?id=Drrl2gcjzl. 2

[19] Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang. InGram: Inductive knowledge graph embedding via relation graphs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 18796–18809. PMLR, 2023. URL https://proceedings.mlr.press/v202/lee23c.html. 3

[20] Tianyin Liao, Chunyu Hu, Yicheng Sui, Xingxuan Zhang, Peng Cui, Jianxin Li, and Ziwei Zhang. TFMLinker: Universal link predictor by graph In-Context learning with tabular foundation models, 2026. URL https://arxiv.org/abs/2602.08592. 1, 2, 20

[21] Aditya Krishna Menon and Robert C. Williamson. Bipartite ranking: A Risk-Theoretic perspective. Journal of Machine Learning Research, 17(195):1–102, 2016. URL https: //jmlr.org/papers/v17/14-265.html. 14, 21

[22] Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-Order graph neural networks. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications ofArtificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019, pages 4602–4609. AAAI Press, 2019. doi: 10. 1609/aaai.v33i01.33014602. URL https://doi.org/10.1609/aaai.v33i01.33014602. 3

[23] Daniel Ruffinelli, Samuel Broscheit, and Rainer Gemulla. You CAN teach an old dog new tricks! on training knowledge graph embeddings. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=BkxSmlBFvr. 2

[24] Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4506–4515, Hong Kong, China, 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1458. URL https://aclanthology. org/D19-1458/. 2, 5

[25] Koustuv Sinha, Shagun Sodhani, Joelle Pineau, and William L. Hamilton. Evaluating logical generalization in graph neural networks, 2020. URL https://arxiv.org/abs/2003.06560. 2, 6

[26] Komal K. Teru, Etienne Denis, and William L. Hamilton. Inductive relation prediction by subgraph reasoning. In Proceedings of the 37th International Conference on Machine Learning,

ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 9448–9457. PMLR, 2020. URL https://proceedings.mlr.press/v119/ teru20a.html. 3

[27] Théo Trouillon, Éric Gaussier, Christopher R. Dance, and Guillaume Bouchard. On inductive abilities of latent factor models for relational learning. Journal ofArtificial Intelligence Research, 64:21–53, 2019. doi: 10.1613/jair.1.11305. 3

[28] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https://openreview.net/ forum?id=ryGs6iA5Km. 3

[29] Jiaxuan You, Rex Ying, and Jure Leskovec. Design space for graph neural networks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ c5c3d4fe6b2cc463c7d7ecba17cc9de7-Abstract.html. 2

[30] Yucheng Zhang, Beatrice Bevilacqua, Mikhail Galkin, and Bruno Ribeiro. TRIX: A more expressive model for Zero-Shot domain transfer in knowledge graphs. In LoG, volume 269 of Proceedings ofMachine Learning Research, pages 12:1–12:28. PMLR, 2025. URL https: //proceedings.mlr.press/v269/zhang25a.html. 3

[31] Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. Neural Bellman-Ford networks: A general graph neural network framework for link prediction. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 29476–29490, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html. 3

## A Limitations

Our experiments audit the graph-to-table interface alone. CLUTRR and GraphLog use supplied symbolic paths, so they evaluate encoding and decoding after relational evidence has been extracted. The observed encoder, context, and decoder effects are limited to the tested configurations, while the 57-KG study uses constructed feature groups and one frozen upstream checkpoint whose pretraining lineage overlaps with part of the panel. Query-level groups and the sequential variance decomposition are descriptive and do not establish causal feature importance or a deployable routing rule. Finally, the family-weighted KG panel uses 64 sampled test triples per dataset. All-entity filtered ranking and the 512-query sensitivity support the reported contrast directions, but the reported sensitivities are conditional on this fixed panel.

## B Controlled Calibration of the Audit Diagnostics

This appendix provides the complete protocols, robustness analyses, and reproducibility details supporting Section 5. The controlled calibration first tests whether the diagnostics in Section 3 recover known properties under graph-disjoint evaluation, before applying them to the benchmarkspecific analyses in Section 5. We generate four binary tasks governed by reciprocity, typed relation composition, local structural roles, and shared graph-community membership. For each task, ten independently generated graphs define disjoint context and evaluation populations. We compute the latent-label ceiling before applying a deterministic 10% symmetric label flip, and define mixed-label collision mass as the fraction of examples assigned to a row-equivalence class containing both latent labels.

## B.1 Finite-Population Diagnostics

Figure 5 reports fitted ROC-AUC (area under the receiver operating characteristic curve) after the controlled label perturbation, whereas the finite-population diagnostics below are decoder-independent properties of the encoder.

We first verify that the ceiling, singleton-mass, and collision diagnostics recover the distinctions imposed by each generated property. For row-equivalence class $c ,$ let $n _ { c } ^ { + }$ and $n _ { c } ^ { - }$ denote its latent positive and negative counts. We define mixed-label collision mass as

$$
\kappa _ { \phi } = \frac { 1 } { N } \sum _ { c } ( n _ { c } ^ { + } + n _ { c } ^ { - } ) { \bf 1 } [ n _ { c } ^ { + } > 0 ] { \bf 1 } [ n _ { c } ^ { - } > 0 ] ,\tag{6}
$$

where $\begin{array} { r } { N = \sum _ { c } ( n _ { c } ^ { + } + n _ { c } ^ { - } ) } \end{array}$ . Equation (6) is the fraction of examples assigned to a row-equivalence class containing both latent labels.

Table 1: Finite-population row-equivalence diagnostics. Results are averaged over ten independently generated evaluation graphs. The ceiling and collision mass use latent labels before the deterministic 10% symmetric label flip. A ceiling and singleton mass of one indicate that every evaluated query has a unique row, characterising distinguishability on the evaluated population.
<table><tr><td>Encoder / property</td><td>Latent  $A ^ { \star }$ </td><td>Singleton mass</td><td>Collision mass</td></tr><tr><td>Relation only / reciprocity</td><td>0.520</td><td>0.000</td><td>1.000</td></tr><tr><td>Reverse edge / reciprocity</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>Typed paths / composition</td><td>1.000</td><td>0.067</td><td>0.000</td></tr><tr><td>Local roles / structural role</td><td>1.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Community affinity / community</td><td>0.924</td><td>0.655</td><td>0.191</td></tr></table>

For binary ranking, the finite-population AUC ceiling follows the optimal ordering of row-equivalence classes by posterior positive probability [21]. For class c, let $n _ { c } ^ { + } , n _ { c } ^ { - }$ , and $\eta _ { c } = n _ { c } ^ { + } / ( n _ { c } ^ { + } + n _ { c } ^ { - } )$ ) denote its positive count, negative count, and positive rate. With $\begin{array} { r } { \breve { N } ^ { + } = \sum _ { c } n _ { c } ^ { + } } \end{array}$ and $\begin{array} { r } { N ^ { - } = \sum _ { c } n _ { c } ^ { - } } \end{array}$ the ceiling is

$$
\mathrm { A U C } ^ { \star } ( \phi ) = \frac { \sum _ { c , d } n _ { c } ^ { + } n _ { d } ^ { - } \left[ \mathbf { 1 } ( \eta _ { c } > \eta _ { d } ) + \frac { 1 } { 2 } \mathbf { 1 } ( \eta _ { c } = \eta _ { d } ) \right] } { N ^ { + } N ^ { - } } .\tag{7}
$$

Ordering classes by $\eta _ { c }$ is optimal because swapping an inverted pair increases the number of correctly ordered positive–negative pairs. Equation (7), like the accuracy ceiling in Section $^ { 3 , }$ is an oracle diagnostic on a fixed labelled population.

Each encoder begins with a four-way query-relation one-hot vector. The reciprocity encoder adds direct- and reverse-edge indicators and endpoint in- and out-degree. The typed-composition encoder adds path counts and existence indicators for fixed ordered two-relation templates. The structural-role encoder adds relation-specific endpoint degrees. The community-affinity encoder adds endpointdegree contrasts, common-neighbour count, Jaccard similarity, Adamic–Adar, preferential attachment, and component status. Feature construction is independent of the perturbed target label.

The calibration flips every tenth positive and negative candidate after an independent within-class permutation determined by the split seed. Query index, candidate order, and flip status are not encoded. Graph generation, relabelling, splitting, feature construction, noise, and decoder seeds are distinct. Because the perturbation mechanically caps sufficient repeated-row encoders near 0.9, Table 1 reports latent pre-flip ceilings and collisions for interpreting encoder distinguishability.

## B.2 Encoder Ladder and Alignment Control

We next test whether the encoder ladder and row-permutation control distinguish task-specific evidence from dimension-matched but misaligned features. Figure 5 reports that the constructed evidence is property-specific. Local-role counts solve the structural-role task but remain near chance on reciprocity and typed composition, which require reverse-edge and ordered-path evidence. The off-diagonal cells therefore test whether a feature family generalises beyond its construction.

(a) 20 context examples per class  
![](images/ccbacd5de2c5412f4014d0aa7590ba8ab26145e7f177d9fa79a1b43c20a93446.jpg)  
Figure 5: Graph-disjoint encoder ladder. Frozen TABPFN-3 ROC-AUC for the four generated relational properties (rows) and candidate encoders (columns). Panel (a) uses 20 labelled context examples per class and panel (b) uses 400. Each cell reports the mean over ten graph seeds, and both panels share the same colour scale. Matched property evidence is the aligned evidence bank constructed for each property. Hatched cells denote encoders not instantiated for the corresponding property.

Figure 6 compares aligned rows with a joint row-permutation null that preserves dimensionality and the covariance structure of complete feature blocks while breaking query-to-row correspondence. Aligned performance increases with the context budget, whereas both permuted conditions remain near chance. At 20 examples per class, the pre-specified TABPFN-3–CATBOOST difference-indifferences is +0.0343, with fixed-panel sensitivity [−0.0128, +0.1106]. The calibration therefore validates the alignment control.

A separately audited independent-column permutation preserves the marginal distribution of each aligned column while destroying cross-column covariance and yields the same near-chance conclusion.

![](images/560f4d3e6467ea33e0144b9e1f932fe0cb678b6f9df5cd95387ba73cccc1c581.jpg)  
Figure 6: Effect of row alignment across context budgets. Bars report the equal-property macro ROC-AUC gain of correctly aligned rows over a joint row permutation of the complete feature block, for TABPFN-3 and CATBOOST at each number of labelled context examples per class, averaged over ten regenerated graphs per property. Under the permutation, ROC-AUC stays between 0.497 and 0.508, close to chance (0.5). Bar labels are rounded to three decimals.

Because either permutation can also alter dependence between the permuted block and unpermuted columns, we interpret these variants as broad alignment nulls.

## C Composition Benchmarks and Complete Results

This section provides the complete encoder schemas, absolute performance levels, and robustness checks for the compositional results in Section 5.1 and Section 5.2. The ordering follows the main analysis. We first document the CLUTRR alignment and length-exposure controls, then report the GraphLog decoder sensitivities and the dihedral boundary control.

## C.1 CLUTRR Protocol and Encoder Schemas

The official gen-train23-test2to10 split contains 9,074 training, 2,020 validation, and 1,146 test stories. Its 18 target classes, in configured index order, are {aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece}. The path-edge vocabulary is the union over the fixed train, validation, and test files and contains {aunt, brother, daughter, father, granddaughter, grandfather, grandmother, grandson, husband, mother, sister, son, uncle, wife}.

Table 2: CLUTRR encoder schemas. All encoders use the same 14-symbol path-edge vocabulary and exclude entity identifiers. Combined variants retain the 15-column relation-multiset block and differ only in the treatment of the 140-column absolute-position block.
<table><tr><td>Encoder</td><td>Dim.</td><td>Contents</td></tr><tr><td>Relation multiset</td><td>15</td><td>Path length and one count per relation symbol</td></tr><tr><td>Absolute position</td><td>141</td><td>Path length and ten 14-way positional one-hot blocks</td></tr><tr><td>Directed transitions</td><td>197</td><td>Path length and counts of ordered adjacent relation pairs</td></tr><tr><td>Aligned absolute + multiset</td><td>155</td><td>Relation multiset and the aligned absolute-position block</td></tr><tr><td>Suffix-zeroed absolute + multiset</td><td>155</td><td>Aligned positions 1–3, with positions 4–10 set to zero</td></tr><tr><td>Permuted absolute + multiset</td><td>155</td><td>Relation multiset and a complete absolute block reassigned within length</td></tr></table>

The permutation acts on complete absolute-position rows separately within each path length. For each paired context seed s, deterministic within-length permutations are generated separately for context and evaluation using seeds s and s + 1, respectively. This preserves dimensionality and the empirical joint distribution of complete absolute-position blocks within each length while breaking their assignment to the original paths. All decoders receive identical stories, labels, context indices, and evaluation queries.

## C.2 CLUTRR Complete Results and Robustness

Table 3 reports the absolute levels underlying Figure 2. At 80 examples per target relation, the relation-multiset encoder outperforms the absolute-position encoder for all three decoders. The directed-transition encoder retains local order without absolute positions, but its relative performance changes across decoders.

Table 3: CLUTRR OOD accuracy. Results use 80 context examples per target relation and are averaged equally across evaluation lengths 4–10 and then across the three paired context seeds. Values in parentheses denote encoder dimensionality.
<table><tr><td>Decoder</td><td>Multiset (15)</td><td>Transitions (197)</td><td>Absolute (141)</td></tr><tr><td>Logistic</td><td>0.429</td><td>0.408</td><td>0.156</td></tr><tr><td>CATBOOST</td><td>0.272</td><td>0.285</td><td>0.109</td></tr><tr><td>TABPFN-3</td><td>0.289</td><td>0.156</td><td>0.103</td></tr></table>

<table><tr><td>Decoder</td><td>Aligned + multiset (155)</td><td>Suffix-zeroed (155)</td><td>Permuted + multiset (155)</td></tr><tr><td>Logistic</td><td>0.266</td><td>0.266</td><td>0.399</td></tr><tr><td>CATBOOST</td><td>0.163</td><td>0.163</td><td>0.255</td></tr><tr><td>TABPFN-3</td><td>0.187</td><td>0.187</td><td>0.291</td></tr></table>

The suffix-zeroed and fully aligned encoders yield indistinguishable OOD accuracy, with a maximum paired accuracy change of $2 . { \overset { - } { 8 } } \times 1 0 ^ { - 1 7 }$ The suffix-zeroed ceiling is 0.998 with singleton mass 0.991, so only 0.009 of the equal-length evaluation mass contributes repeated rows. Its conditional repeated-row ceiling is 0.721. The absolute-position and combined encoders fit the length-2–3 context at 0.990–0.995 accuracy, as reported in Table 4.

Table 4: CLUTRR context fit and finite-population diagnostics. Results use 80 context examples per target relation. OOD diagnostics average lengths 4–10 equally. Conditional ceilings use only repeated-row mass and are undefined when every row is a singleton. Context accuracy is averaged across the same three paired context seeds used for OOD evaluation.
<table><tr><td>Encoder</td><td>OOD ceiling</td><td>Singleton</td><td>Non-singleton</td><td>Conditional</td><td>Logistic</td><td>CATBOOST</td><td>TABPFN-3</td></tr><tr><td>Relation multiset</td><td>0.930</td><td>0.757</td><td>0.243</td><td>0.712</td><td>0.764</td><td>0.774</td><td>0.791</td></tr><tr><td>Absolute position</td><td>1.000</td><td>1.000</td><td>0.000</td><td></td><td>0.990</td><td>0.991</td><td>0.991</td></tr><tr><td>Supported prefix + multiset</td><td>0.998</td><td>0.991</td><td>0.009</td><td>0.721</td><td>0.994</td><td>0.995</td><td>0.995</td></tr></table>

The collision structure also changes with path length. The relation-multiset singleton mass increases from 0.453 at length 4 to 0.933 at length 10, whereas the absolute-position encoder has singleton mass 1.000 at every evaluated length. High finite-population ceilings therefore increasingly reflect query-level lookup as paths become longer.

Table 5 separates block addition, block alignment, and length exposure. The aligned and permuted conditions differ only in path-to-block assignment. The exposure intervention retains the same nested length-2–3 base in both arms and adds 20 examples per target relation from either lengths 2–3 or length 4.

Table 6 shows that the multiset-minus-absolute ordering holds across every declared classification endpoint. For log loss and Brier score, positive values denote the reduction obtained by the relation multiset.

The relation multiset outperforms absolute positions on 16 of 18 target relations for logistic regression and CATBOOST, and on 15 of 18 for TABPFN-3. Restricting evaluation to lengths 4–5, matching the magnitude of the GraphLog length shift, retains effects of $+ 0 . 3 4 3 / + 0 . 1 \hat { 7 } 7 / + 0 . 3 3 8$ . These analyses are robustness checks within one official population. The shared-parameter GRU uses a 32-dimensional relation embedding, 64 hidden units, one recurrent layer, and context-only early stopping. Its accuracy increases from 0.346 to 0.705 across the five context budgets and decreases from 0.898 at length 4 to 0.527 at length 10 at the largest budget. These controls support the separation in Section 5.1 between row distinguishability, context fit, and short-to-long transfer.

Table 5: Matched CLUTRR interventions. Block addition reports permuted absolute plus multiset minus multiset only, alignment reports permuted absolute plus multiset minus aligned absolute plus multiset, and length exposure replaces 20 additional length-2–3 examples with 20 length-4 examples. Positive values denote higher OOD accuracy for block addition and length exposure. For alignment, they denote the accuracy lost after restoring path-level alignment.
<table><tr><td>Decoder</td><td>Block addition</td><td>Alignment</td><td>Exposure at L4</td><td>Exposure at L5–10</td></tr><tr><td>Logistic</td><td>-0.030</td><td>0.133</td><td>+0.489</td><td>-0.001</td></tr><tr><td>CATBOOST</td><td>-0.017</td><td>0.092</td><td>+0.372</td><td>+0.010</td></tr><tr><td>TABPFN-3</td><td>+0.002</td><td>0.104</td><td>+0.758</td><td>+0.033</td></tr></table>

Table 6: CLUTRR metric robustness. Results use 80 context examples per target relation and are averaged equally across OOD lengths 4–10 and then across the three paired context seeds. Positive values favour the relation multiset.
<table><tr><td>Decoder</td><td>Accuracy</td><td>Bal. acc.</td><td>Macro-F1</td><td>Top-3</td><td>Log loss</td><td>Brier</td></tr><tr><td>Logistic</td><td>+0.273</td><td>+0.236</td><td>+0.224</td><td>+0.403</td><td>+4.208</td><td>+0.482</td></tr><tr><td>CATBOOST</td><td>+0.163</td><td>+0.132</td><td>+0.155</td><td>+0.293</td><td>+2.002</td><td>+0.459</td></tr><tr><td>TABPFN-3</td><td>+0.186</td><td>+0.201</td><td>+0.153</td><td>+0.304</td><td>+2.510</td><td>+0.540</td></tr></table>

## C.3 GraphLog Complete Results and Decoder Sensitivities

All 57 GraphLog worlds use the same global vocabulary of 20 directed relation symbols. Encoder dimensionality and coordinate meaning therefore remain fixed across worlds. Table 7 provides the absolute balanced-accuracy levels behind Figure 3, together with the absolute levels for the dihedral boundary control.

Table 7: GraphLog and dihedral-control results. GraphLog reports equal-world macro balanced accuracy at the largest context budget, with the contrast defined as relation multiset minus absolute position. The dihedral control reports accuracy over the 16 symmetries of a regular octagon and has chance accuracy 1/16 = 0.0625.
<table><tr><td></td><td colspan="3">GraphLog</td><td colspan="2">Dihedral control</td></tr><tr><td>Decoder</td><td>Multiset</td><td>Absolute</td><td>Difference</td><td>Coarse</td><td>Absolute</td></tr><tr><td>Logistic</td><td>0.298</td><td>0.234</td><td>+0.065</td><td>0.0618</td><td>0.0622</td></tr><tr><td>CATBOOST</td><td>0.125</td><td>0.282</td><td>-0.157</td><td>0.0770</td><td>0.0643</td></tr><tr><td>TABPFN-3</td><td>0.159</td><td>0.238</td><td>-0.079</td><td>0.0614</td><td>0.0626</td></tr></table>

The multiset–absolute comparison is complemented by a blockwise alignment control. At 40 examples per class, the block-permuted-minus-aligned contrast is small for logistic regression, whereas both nonlinear decoders prefer the correctly aligned ordered block. Because the permutation removes both supported prefix coordinates and evaluation-only suffix coordinates, this contrast measures broad ordered-block dependence.

The 57 worlds draw subsets from 76 rules. Their mean pairwise rule-set Jaccard similarity is 0.230 and the maximum is 0.905. Similarity-cluster sensitivities preserve the direction of the logistic and nonlinear-decoder effects. Leave-cluster-out regressions based on rule overlap and simple world descriptors yield negative R<sup>2</sup>, indicating poor prediction of the observed encoder contrast from these descriptors. These checks describe robustness within the complete benchmark, with worlds treated as fixed benchmark units.

The class-reweighted ceilings used for the main GraphLog comparison are 0.965 for the relation multiset and 0.983 for absolute positions. Table 9 reports the capacity sweep. Every depth is rerun on all 57 worlds with the same context rows, evaluation queries, 500 trees, learning rate 0.05, and three seeds. Increasing depth raises context fit for both encoders while progressively reducing multiset evaluation accuracy. This pattern supports capacity-amplified overfitting of the multiset row under this CATBOOST configuration.

![](images/55bcc4eaafaa6d1180c1ae8ab25464aa5e46ed567bf4a20dbe23ccb72e1e5bf8.jpg)

(b) Encoder contrasts  
![](images/47983b81190e9dc69e30a6c08acbdd7ed6d7c26a534fb935acecb2c3996721ff.jpg)  
Context-tuned CatBoost (40 rows)  
Figure 7: GraphLog encoder effects across context budgets. (a) Equal-world macro balanced accuracy for the relation-multiset (solid) and absolute-position (dashed) encoders. (b) Relationmultiset-minus-absolute-position (solid) and block-permuted-minus-aligned (dashed) contrasts. The diamonds mark context-tuned CATBOOST at 40 rows for the two contrasts, drawn slightly to the right of the 40-row position. Context rows per relation are shown on a logarithmic axis.

Table 8: GraphLog row-equivalence diagnostics. Ordinary-accuracy ceilings and singleton mass are averaged equally across worlds at each OOD length. The main decoder results use macro balanced accuracy and the corresponding class-reweighted ceiling.
<table><tr><td>Encoder</td><td>Ceiling L4</td><td>Singleton L4</td><td>Ceiling L5</td><td>Singleton L5</td></tr><tr><td>Length only</td><td>0.290</td><td>0.000</td><td>0.269</td><td>0.000</td></tr><tr><td>Relation multiset</td><td>0.963</td><td>0.005</td><td>0.974</td><td>0.163</td></tr><tr><td>Directed transitions</td><td>0.981</td><td>0.007</td><td>0.990</td><td>0.188</td></tr><tr><td>Absolute position</td><td>0.981</td><td>0.007</td><td>0.992</td><td>0.188</td></tr><tr><td>Absolute + multiset</td><td>0.981</td><td>0.007</td><td>0.992</td><td>0.188</td></tr></table>

Table 9: GraphLog CATBOOST capacity sweep. Evaluation values are equal-world means of equal-length macro balanced accuracy over lengths 4–5. Context values are equal-world macro balanced accuracy. The effect is relation multiset minus absolute position.
<table><tr><td>Depth</td><td>Eval. multiset</td><td>Eval. absolute</td><td>Effect</td><td>Context multiset</td><td>Context absolute</td><td>Multiset at/below chance</td></tr><tr><td>1</td><td>0.201</td><td>0.271</td><td>-0.070</td><td>0.830</td><td>0.910</td><td>5</td></tr><tr><td>2</td><td>0.169</td><td>0.264</td><td>-0.096</td><td>0.930</td><td>0.980</td><td>9</td></tr><tr><td>3</td><td>0.150</td><td>0.263</td><td>-0.112</td><td>0.945</td><td>0.988</td><td>20</td></tr><tr><td>4</td><td>0.138</td><td>0.268</td><td>-0.129</td><td>0.947</td><td>0.990</td><td>25</td></tr><tr><td>5</td><td>0.129</td><td>0.278</td><td>-0.149</td><td>0.948</td><td>0.991</td><td>25</td></tr><tr><td>6</td><td>0.126</td><td>0.282</td><td>-0.156</td><td>0.948</td><td>0.991</td><td>27</td></tr><tr><td>7</td><td>0.125</td><td>0.282</td><td>-0.157</td><td>0.948</td><td>0.991</td><td>30</td></tr></table>

At 40 examples per class, multiset and absolute-position performance are at or below each world’s chance level in 0 and 3 worlds for logistic regression, 30 and 1 for depth-7 CATBOOST, and 12 and 4 for TABPFN-3.

## C.4 Dihedral Boundary Control

The independent control samples ten generated populations over the 16 symmetries of a regular octagon. Context sequences have lengths 2–3, evaluation sequences have lengths 4–10, and the ordered group product is the target. The absolute-position encoder preserves the element at each sequence position. The coarse encoder removes order and merges elements separated by four rotations, so it discards both ordering information and part of the group-element identity.

Across generated populations, the 95% sensitivity ranges for coarse-minus-absolute accuracy are [−0.0028, 0.0020], [0.0106, 0.0147], and [−0.0042, 0.0023] for logistic regression, CATBOOST, and TABPFN-3. Both encoders remain at or near chance apart from the small CATBOOST contrast. The 257-column directed-transition encoder also remains at floor, reaching 0.0622/0.0617/0.0648 for the three decoders. This boundary control limits the CLUTRR interpretation: when coarsening also removes task-relevant identity, both representations remain at floor apart from the small CATBOOST contrast.

## D Knowledge Graph Panel

This section documents the panel, encoders, context constructions, and attribution checks underlying Section 5.3 and Section 5.4. The presentation separates the fixed evaluation protocol from the encoder and context variants so that every reported contrast can be traced to one changed component.

## D.1 Panel Composition and Evaluation Protocol

The panel contains 57 knowledge graphs grouped into 16 metric-free provenance families. Family assignment follows source lineage and is independent of decoder outcomes. Table 10 lists the complete panel.

Table 10: Summary of the 57-KG panel. Dataset names reproduce the fixed asset identifiers. Provenance families define the equal-weight aggregation and resampling units used in the main analysis.

<table><tr><td>Family</td><td>Dataset identifiers</td></tr><tr><td>AristoV4</td><td>aristov4</td></tr><tr><td>CoDEx</td><td>codex_large, codex_medium, codex_small</td></tr><tr><td>ConceptNet100K DBpedia100K</td><td>conceptnet100k dbpedia100k</td></tr><tr><td>FB15k-237</td><td>fb15k237, fb15k237_10, fb15k237_20, fb15k237_50, fb15k237_ind_v1, fb15k237_ind_v2,</td></tr><tr><td>FBNELL</td><td>fb15k237_ind_v3,fb15k237_ind_v4,fb_ingram_25,fb_ingram_50,fb_ingram_75,fb_ingram_100</td></tr><tr><td>Hetionet</td><td>fbnell hetionet</td></tr><tr><td>H&amp;M</td><td>hm_1k, hm_3k, hm_5k, hm_indigo</td></tr><tr><td>ILPC2022</td><td>ilpc2022_large, ilpc2022_small</td></tr><tr><td>MetaFam</td><td>metafam</td></tr><tr><td>NELL</td><td>nell23k, nell995, nell_ind_v1, nell_ind_v2, nell_ind_v3, nell_ind_v4, nl_ingram_0,</td></tr><tr><td>WDSinger</td><td>nl_ingram_25, nl_ingram_50, nl_ingram_75, nl_ingram_100 wdsinger</td></tr><tr><td>Wiki-Topics</td><td>wiki_mt1_health, wiki_mt1_tax, wiki_mt2_org, wiki_mt2_sci, wiki_mt3_art, wiki_mt3_infra, wiki_mt4_health,wiki_mt4_sci</td></tr><tr><td>WK-InGram</td><td>wk_ingram_25, wk_ingram_50, wk_ingram_75, wk_ingram_100</td></tr><tr><td>WN18RR</td><td>wn18rr, wn18rr_ind_v1, wn18rr_ind_v2, wn18rr_ind_v3, wn18rr_ind_v4</td></tr><tr><td>YAGO3-10</td><td>yago310</td></tr></table>

For each dataset and seed, the source pool for the complete encoder ladder contains 400 positives sampled without replacement from the official held-out target triples and 400 endpoint corruptions that retain the positive relation. The budgets of 20, 50, 100, and 200 examples per binary class are deterministic nested subsets. No relation is dropped or duplicated to satisfy a per-relation quota. These baseline corruptions are not guaranteed to exclude every known positive triple and are distinct from the explicitly filtered condition in Appendix D.7.

The largest 400-row local context matches the inference-time local-context size of TFMLinker [20], which uses 200 positive and 200 negative local rows together with 80 learned global prototypes. Our setup retains the 400-row local context while freezing upstream models and auditing labelled target-graph context, leaving the learned cross-graph prototypes outside the comparison.

A lower-bound audit finds that 32 base negative rows coincide with one of the selected 64-query evaluation positives (0.047% of the 68,400 base negative rows). After expansion over the four nested context budgets, 19 selected incidences occur across 8 of 684 dataset–seed–budget cells, corresponding to 0.030% of selected negative-row incidences. This audit covers overlap with the sampled evaluation positives.

Constructed features and frozen ULTRA scores use only the observed support graph, which excludes every held-out target triple. Neither the queried target edge nor an artificial inverse copy is inserted when constructing its row, while naturally observed reverse-relation edges remain available. Across all 684 dataset–seed–budget contexts, selected positive context triples and the 64 sampled evaluation triples have zero overlap. Evaluation samples 64 official test triples stratified by relation and ranks both head and tail prediction against all entities under standard filtered masks. Every paired comparison uses the same queries, candidates, filtering, and tie handling.

Let $M _ { d , s }$ denote the query-averaged metric for dataset d and seed $s ,$ and let $\mathcal { D } _ { f }$ be the datasets assigned to provenance family $f .$ The equal-family estimate is

$$
\overline { { M } } = \frac { 1 } { | \mathcal { F } | } \sum _ { f \in \mathcal { F } } \frac { 1 } { | \mathscr { D } _ { f } | } \sum _ { d \in \mathscr { D } _ { f } } \frac { 1 } { | S | } \sum _ { s \in \mathscr { S } } M _ { d , s } .\tag{8}
$$

Equation (8) prevents large provenance families from dominating the panel average. Provenance families are also the resampling and sign-flip units.

Because eight of the sixteen families contain a single dataset, we report equal-dataset aggregation alongside the registered equal-family estimand. For the full-TABPFN-3 minus native-ULTRA contrast, the equal-family mean is +0.006 and the equal-dataset mean is -0.035. Their medians are -0.034 and -0.039. Dataset wins/ties/losses are 5/4/48 under the declared ±0.005 band. The leave-one-family-out range is [-0.041,+0.015], and removing MetaFam yields -0.041. These are fixed-panel sensitivities.

## D.2 Query-Relative Pointwise Control

The tabular decoders are trained as pointwise binary classifiers on a shared labelled context but are evaluated by filtered ranking within each directional query. Their training loss targets pointwise classification, whereas MRR depends on within-query candidate ordering; classification and ranking losses therefore require distinct analyses [21]. Cross-query calibration is irrelevant to MRR, which is computed separately within each query. We therefore repeat the 200-per-class comparison using two label-free, query-relative feature maps. For each numerical column, within-query z-score subtracts the group mean and divides by its standard deviation, mapping constant columns to zero. Within-query percentile replaces values by tie-aware midranks scaled to [0, 1]. The maps are applied independently to every directional group: a context group contains a source positive and its paired endpoint corruption, while an evaluation group contains all entity candidates for one head- or tail-prediction query. Thus context labels and evaluation targets are never used by the transformation.

The control crosses the raw, z-score, and percentile maps with the Local, score, and Local + score encoders and all three decoders, using the same 57 datasets, three seeds, pool-random positive contexts with filtered-uniform negatives, queries, candidates, and ranking masks. These positive contexts differ from the evaluation-relation-matched positive contexts used in Table 15. Because an affine or rank transform of the one-column frozen score is order preserving within each query, score-only logistic regression must reproduce native ULTRA whenever the fitted coefficient has positive orientation. Exact native-rank equality is checked cell by cell. The remaining contrasts test whether query-relative inputs remedy the mismatch for a nonlinear or multivariate pointwise readout.

Table 11 reports all 4,617 completed cells. Within-query scaling improves the Local row for logistic regression, but not the multivariate Local + score row. It lowers MRR for all six CATBOOST transformations and all six TABPFN-3 transformations. All six CATBOOST reductions and five of the six TABPFN-3 reductions remain supported after joint correction. The remaining TABPFN-3 percentile contrast is negative but does not survive the correction. For score-only logistic regression, all 171 dataset–seed cells reproduce native ULTRA exactly under every map, covering 21,888 directional query rankings per map. These controls show that the effect of query-relative scaling is decoder-specific. The analysis is scoped to the tested pointwise readouts under the fixed ranking protocol.

## D.3 Positive-Context Relation Coverage

We additionally vary the positive context while keeping its size and the evaluation queries fixed. Pool-random positives are sampled from the available source pool, whereas evaluation-relationmatched positives are selected to better match the relation distribution of the evaluation queries. Relation matching increases equal-family relation-type coverage from 0.667 to 0.871 and relationmass coverage from 0.693 to 0.891, while reducing the total-variation distance between context and evaluation relation distributions from 0.447 to 0.190.

Table 11: Query-relative control. Entries are equal-family MRR. Parentheses report the transformedminus-raw effect. Z-score and percentile maps are applied by the same label-free rule to context groups and evaluation candidate blocks.
<table><tr><td>Decoder</td><td>Encoder</td><td>Raw</td><td>Within-query z-score</td><td>Within-query percentile</td></tr><tr><td rowspan="3">Logistic</td><td>Local</td><td>0.198</td><td>0.279 (+0.081)</td><td>0.274 (+0.076)</td></tr><tr><td>Score</td><td>0.335</td><td>0.335 (0.000)</td><td>0.335 (0.000)</td></tr><tr><td>Local + score</td><td>0.326</td><td>0.285 (-0.041)</td><td>0.274 (-0.053)</td></tr><tr><td rowspan="3">CATBOOST</td><td>Local</td><td>0.245</td><td>0.199 (-0.046)</td><td>0.181 (-0.064)</td></tr><tr><td>Score</td><td>0.299</td><td>0.006 (-0.293)</td><td>0.004 (-0.295)</td></tr><tr><td>Local + score</td><td>0.301</td><td>0.159 (-0.142)</td><td>0.145 (-0.156)</td></tr><tr><td rowspan="3">TABPFN-3</td><td>Local</td><td>0.265</td><td>0.002 (-0.262)</td><td>0.142 (-0.123)</td></tr><tr><td>Score</td><td>0.329</td><td>0.002 (-0.326)</td><td>0.186 (-0.142)</td></tr><tr><td>Local + score</td><td>0.387</td><td>0.002 (-0.384)</td><td>0.222 (-0.165)</td></tr></table>

Despite these large changes in relational coverage, the corresponding MRR effects are generally modest across the tested encoder and decoder configurations. Among the 12 evaluation-relationmatched-minus-pool-random contrasts, only the CATBOOST local encoder without relation identity remains supported after the joint multiple-comparison correction, with an equal-family MRR change of +0.0139 $( p _ { \operatorname* { m a x } T } = 0 . 0 3 6 9 )$ . Increasing relation coverage therefore changes the labelled support substantially, while the corresponding MRR effects remain modest.

## D.4 Relation-Identity Ablation

Because the Local encoder includes relation identity whereas the entity-token control does not, we separately remove the relation-identity column and repeat the matched comparisons across decoders, upstream-score conditions, and positive-context constructions. Across the relation-identity contrasts, the absolute equal-family MRR effect never exceeds 0.0031, and none remains supported after the joint max-|t| correction. The relation-identity column therefore accounts for little of the observed local-versus-identifier difference under these matched conditions.

## D.5 Encoder Schemas

Table 12 defines the nine encoders used in the complete encoder ladder and in the matched additions reported in Section 5.3.

Table 12: Graph-to-table encoder schemas. The nine encoders are grouped by the evidence exposed by their columns. Composite encoders concatenate the named blocks without recomputation and inherit the icons of their constituent evidence families.
<table><tr><td></td><td>Encoder</td><td>Dim.</td><td>Columns or blocks</td></tr><tr><td>#</td><td>Entity tokens</td><td>2</td><td>Dataset-scoped head and tail tokens</td></tr><tr><td></td><td>Dyadic</td><td>3</td><td>Other-relation pair count, reverse-pair count, same entity</td></tr><tr><td></td><td>Paths</td><td>4</td><td>Two-hop, start-relation, end-relation, reverse-two-hop counts</td></tr><tr><td>VYY</td><td>Local</td><td>18</td><td>Relation identity; endpoint, relation, pair, path, community statistics</td></tr><tr><td></td><td>Global</td><td>13</td><td>PageRank, component, heat, diffusion</td></tr><tr><td>中</td><td>ULTRA score</td><td>1</td><td>Frozen ULTRA-4G candidate score</td></tr><tr><td></td><td>Local + global</td><td>31</td><td>18 local ⊕ 13 global</td></tr><tr><td></td><td>Local + score</td><td>19</td><td>18 local ⊕ frozen score</td></tr><tr><td></td><td>Full table</td><td>32</td><td>18 local ⊕ frozen score ⊕ 13 global</td></tr></table>

<sup>º</sup> = opaque identifiers; $\pmb { \mathscr { L } }$ = local relational structure; <sup></sup> = global topological and spectral structure; Ô = learned-model score.

The 18 local columns are relation identity; total in- and out-degree for each endpoint; head relationout and tail relation-in degree; relation frequency, functionality, and inverse functionality; otherrelation pair and reverse-pair counts; two-hop, start-relation, end-relation, and reverse-two-hop counts; shared community assignment; and community relation density. The 13 global columns are head and tail PageRank, membership in the same weakly connected component, head and tail log component fraction, and truncated low-frequency heat-projector similarity and diffusion distance at $t \in \bar { \{ 0 . 1 , 1 , 1 0 , 1 0 0 \} }$ . The community and weak-component indicators are distinct, and no column is duplicated.

## D.6 Complete Encoder Ladder

Table 13 reports the absolute TABPFN-3 levels behind the nested comparisons in Section 5.3. Dyadic evidence captures much of the narrow endpoint-pair signal, paths improve with context but remain weaker in aggregate, global features alone remain limited, and the frozen score is the strongest individual feature group. Composite levels combine these sources and are therefore interpreted through the paired additions in Appendix D.8.

Table 13: TABPFN-3 performance across the encoder ladder. Entries report equal-family MRR under filtered candidate evaluation on the 57-KG panel. The unfiltered-uniform context negatives are target-excluding endpoint corruptions and are not filtered by the all-known-positive mask. Contexts contain 20, 50, 100, or 200 examples per binary class.
<table><tr><td>Encoder</td><td>20</td><td>50</td><td>100</td><td>200</td></tr><tr><td>Dataset-scoped entity IDs</td><td>0.002</td><td>0.002</td><td>0.003</td><td>0.004</td></tr><tr><td>Dyadic</td><td>0.119</td><td>0.136</td><td>0.149</td><td>0.152</td></tr><tr><td>Paths</td><td>0.064</td><td>0.068</td><td>0.071</td><td>0.076</td></tr><tr><td>Local</td><td>0.167</td><td>0.208</td><td>0.228</td><td>0.237</td></tr><tr><td>Global</td><td>0.053</td><td>0.059</td><td>0.069</td><td>0.068</td></tr><tr><td>ULTRA score</td><td>0.299</td><td>0.305</td><td>0.307</td><td>0.301</td></tr><tr><td>Local + global</td><td>0.134</td><td>0.189</td><td>0.217</td><td>0.233</td></tr><tr><td>Local + score</td><td>0.297</td><td>0.323</td><td>0.334</td><td>0.336</td></tr><tr><td>Full</td><td>0.287</td><td>0.321</td><td>0.339</td><td>0.341</td></tr></table>

## D.7 Negative-Context Construction and Audit Checks

The context-robustness experiment fixes 200 positive and 200 negative context rows for each dataset and seed. Context positives and the 64 sampled evaluation triples are disjoint. Positive rows and evaluation queries are identical across the two negative-context conditions.

Filtered-uniform negatives replace the missing endpoint with a uniformly sampled candidate excluded by the all-known-positive mask. Structurally plausible negatives first restrict candidates to the appropriate relation-specific endpoint role and, when possible, the same graph partition as the fixed endpoint. They then sample among the 32 closest candidates under relative in- and out-degree. Empty role and partition pools trigger explicit fallbacks, observed for 1,399 and 14,446 selected negatives (4.1% and 42.2% of the 34,200). The selector uses only the graph-based criteria above. Its selected negatives nevertheless receive a mean frozen score of −1.238, compared with −6.929 under filtered-uniform sampling, indicating that the two conditions differ in upstream-model difficulty despite sharing positives and evaluation.

All 171 dataset–seed context audits contain zero known-positive negative rows, and all positiverow hashes agree between conditions. Evaluation indices, candidate sets, and filtered masks are unchanged. The paired contrast therefore isolates the negative context distribution.

The complete encoder ladder in Table 13 uses endpoint corruptions without an all-known-positive rejection step, whereas the context-robustness experiment uses filtered negatives. We isolate filtering with a separate matched asset constructed from one deterministic corruption stream. Across its unfiltered and filtered arms, positive rows, directional query identities, seeds, evaluation candidates, masks, targets, and feature schemas are identical. Only rejection of candidates in the all-knownpositive set changes. The unfiltered arm contains 456 known-positive negatives among 34,200 rows (1.33%), distributed across 88 of 171 dataset–seed contexts. Contamination reaches 39.5% for one seed of nell\_ind\_v1. We rerun TABPFN-3 on the Local, score, and Local + score encoders in all 1,026 matched cells.

Filtering yields a modest +0.0081 equal-family improvement for Local, with sensitivity [+0.0022,+0.0147], but it falls just outside the jointly corrected threshold $( p _ { \operatorname* { m a x } T } = 0 . 0 5 7 8 )$ . The score-containing effects are smaller and unsupported. The contamination audit is therefore operationally important, especially for NELL-ind-v1, while the matched filtering effects are substantially smaller than the cross-table differences of 0.030–0.055 MRR.

Table 14: Matched all-known-positive filtering control. The effect is filtered-uniform minus unfiltered-uniform at 200 examples per class. Levels and effects are MRR; sensitivities resample the 16 provenance families. The last column is the exact family-level max-|t| correction across the three contrasts.
<table><tr><td>Encoder</td><td>Unfiltered</td><td>Filtered</td><td>Equal-family</td><td>Equal-dataset</td><td>95% sensitivity</td><td> $p _ { \operatorname* { m a x } { T } }$ </td></tr><tr><td>Local</td><td>0.257</td><td>0.265</td><td>+0.0081</td><td>+0.0056</td><td>[+0.0022,+0.0147]</td><td>0.0578</td></tr><tr><td>Score</td><td>0.326</td><td>0.329</td><td>+0.0022</td><td>+0.0021</td><td>[-0.0022,+0.0074]</td><td>0.7739</td></tr><tr><td>Local + score</td><td>0.387</td><td>0.387</td><td>-0.0007</td><td>+0.0040</td><td>[-0.0085,+0.0062]</td><td>0.9972</td></tr></table>

## D.8 Paired Context and Encoder Effects

Table 15 gives the absolute equal-family levels associated with Figure 4. The best decoder depends on the encoder. Logistic regression is strongest for the score-only row, while TABPFN-3 is strongest for local and local-plus-score rows under both context constructions.

Table 15: Context-robustness results. Entries report equal-family MRR at 200 examples per class for filtered-uniform and structurally plausible negative contexts. Evaluation-relation-matched positive rows and evaluation queries are identical across conditions. Bold indicates the best decoder within each encoder and context.
<table><tr><td></td><td colspan="3">Filtered uniform</td><td colspan="3">Structurally plausible</td></tr><tr><td>Encoder</td><td>Logistic</td><td>CATBOOST</td><td>TABPFN-3</td><td>Logistic</td><td>CATBOOST</td><td>TABPFN-3</td></tr><tr><td>Local</td><td>0.199</td><td>0.262</td><td>0.267</td><td>0.179</td><td>0.199</td><td>0.213</td></tr><tr><td>Score</td><td>0.335</td><td>0.299</td><td>0.332</td><td>0.301</td><td>0.197</td><td>0.245</td></tr><tr><td>Local + score</td><td>0.328</td><td>0.310</td><td>0.391</td><td>0.262</td><td>0.261</td><td>0.269</td></tr></table>

For each contrast $c ,$ let $\delta _ { f , c }$ be its mean paired effect within provenance family $f .$ The exact multiplecomparison procedure enumerates all $2 ^ { \bar { 1 } 6 }$ family-level sign assignments, recomputes the one-sample t-statistic for every contrast, and records $T = \operatorname* { m a x } _ { c } \left| t _ { c } \right|$ . The adjusted $p _ { \operatorname* { m a x } T }$ in Table 16 is the fraction of sign assignments whose maximum statistic is at least as large as the observed $\left| t _ { c } \right|$ . This preserves the family as the inferential unit and controls the 30 tested panel contrasts jointly.

Table 16: Paired encoder effects. Effects are equal-family ∆MRR with 95% provenance-family bootstrap sensitivity intervals. Positive values favour the first term in each contrast. Dataset wins, ties, and losses use a ±0.005 band. The final column reports the exact max-|t| family-level sign-flip correction across 30 contrasts.
<table><tr><td>Contrast</td><td>Context</td><td>Decoder</td><td>Effect</td><td>95% sensitivity</td><td>W/T/L</td><td> $p _ { \operatorname* { m a x } T }$ </td></tr><tr><td rowspan="6">Local minus entity IDs</td><td>Filtered uniform</td><td>Logistic</td><td>+0.197</td><td>[+0.117,+0.314]</td><td>57/0/0</td><td>0.018</td></tr><tr><td>Filtered uniform</td><td>CATBOOST</td><td>+0.258</td><td> $[ + 0 . 1 7 7 , + 0 . 3 7 2 ]$ </td><td>57/0/0</td><td>0.002</td></tr><tr><td>Filtered uniform</td><td>TABPFN-3</td><td>+0.262</td><td>[+0.184,+0.378]</td><td>57/0/0</td><td>0.001</td></tr><tr><td>Structural</td><td>Logistic</td><td>+0.177</td><td> $[ + 0 . 0 9 7 , + 0 . 3 0 1 ]$ </td><td>57/0/0</td><td>0.045</td></tr><tr><td>Structural</td><td>CATBOOST</td><td>+0.194</td><td> $[ + 0 . 1 1 8 , + 0 . 3 1 4 ]$ </td><td>57/0/0</td><td>0.019</td></tr><tr><td>Structural</td><td>TABPFN-3</td><td>+0.208</td><td>[+0.128,+0.329]</td><td>57/0/0</td><td>0.013</td></tr><tr><td rowspan="6">Score added to local</td><td>Filtered uniform</td><td>Logistic</td><td>+0.129</td><td>[+0.093,+0.162]</td><td>56/0/1</td><td>&lt;0.001</td></tr><tr><td>Filtered uniform</td><td>CATBOOST</td><td>+0.048</td><td>[+0.033,+0.065]</td><td>48/3/6</td><td>&lt;0.001</td></tr><tr><td>Filtered uniform</td><td>TABPFN-3</td><td>+0.124</td><td>[+0.096,+0.152]</td><td>54/3/0</td><td>&lt;0.001</td></tr><tr><td>Structural</td><td>Logistic</td><td>+0.083</td><td>[+0.041,+0.129]</td><td>41/0/16</td><td>0.017</td></tr><tr><td>Structural</td><td>CATBOOST</td><td>+0.061</td><td>[+0.030,+0.096]</td><td>36/5/16</td><td>0.018</td></tr><tr><td>Structural</td><td>TABPFN-3</td><td>+0.056</td><td>[+0.023,+0.092]</td><td>38/3/16</td><td>0.051</td></tr><tr><td rowspan="6">Local added to score</td><td>Filtered uniform</td><td>Logistic</td><td>-0.007</td><td>[-0.065,+0.095]</td><td>4/6/47</td><td>1.000</td></tr><tr><td>Filtered uniform</td><td>CATBOOST</td><td>+0.012</td><td>[-0.054,+0.125]</td><td>10/2/45</td><td>1.000</td></tr><tr><td>Filtered uniform</td><td>TABPFN-3</td><td>+0.059</td><td>[+0.005,+0.158]</td><td>28/12/17</td><td>0.978</td></tr><tr><td>Structural</td><td>Logistic</td><td>-0.039</td><td>[-0.115,+0.076]</td><td>10/3/44</td><td>1.000</td></tr><tr><td>Structural</td><td>CATBOOST</td><td>+0.064</td><td>[-0.003,+0.168]</td><td>39/3/15</td><td>0.955</td></tr><tr><td>Structural</td><td>TABPFN-3</td><td>+0.024</td><td>[-0.038,+0.128]</td><td>18/6/33</td><td>1.000</td></tr></table>

Table 16 formalises the asymmetry discussed in Section 5.3. Adding the frozen score to local features is positive in all six conditions and remains supported in five after joint correction. Adding local features to the score is smaller and unstable, with no adjusted contrast supported. The direction of a feature addition must therefore be retained when interpreting nested encoders.

Table 17: Effect of negative-context construction. Filtered-uniform negatives are replaced with structurally plausible negatives while positive rows and evaluation queries remain fixed. Entries report equal-family ∆MRR with 95% provenance-family bootstrap sensitivity intervals and dataset wins, ties, and losses under a ±0.005 band. Positive values favour the structurally plausible context.
<table><tr><td>Encoder</td><td>Decoder</td><td>Effect</td><td>95% sensitivity</td><td>W/T/L</td></tr><tr><td>Local</td><td>Logistic</td><td>-0.020</td><td>[-0.032,-0.010]</td><td>10/8/39</td></tr><tr><td></td><td>CATBOOST</td><td>-0.063</td><td>[-0.084,-0.042]</td><td>4/2/51</td></tr><tr><td></td><td>TABPFN-3</td><td>-0.054</td><td>[-0.075,-0.033]</td><td>6/3/48</td></tr><tr><td>Score</td><td>Logistic</td><td>-0.033</td><td>[-0.073,-0.005]</td><td>0/45/12</td></tr><tr><td></td><td>CATBOOST</td><td>-0.102</td><td>[-0.144,-0.060]</td><td>8/1/48</td></tr><tr><td></td><td>TABPFN-3</td><td>-0.086</td><td>[-0.132,-0.043]</td><td>4/13/40</td></tr><tr><td>Local + score</td><td>Logistic</td><td>-0.066</td><td>[-0.102,-0.032]</td><td>10/4/43</td></tr><tr><td></td><td>CATBOOST</td><td>-0.050</td><td>[-0.087,-0.014]</td><td>13/1/43</td></tr><tr><td></td><td>TABPFN-3</td><td>-0.122</td><td>[-0.164,-0.081]</td><td>1/2/54</td></tr></table>

Table 17 shows that structurally plausible negatives lower MRR for every informative encoder– decoder combination. The identifier baselines change by less than 0.001 and are omitted from the table. Since positives and evaluation remain fixed, these effects quantify sensitivity to the negative-context construction S.

## D.9 Upstream-Score Lineage and Native-Ranking Checks

The frozen ultra\_4g checkpoint was pretrained on FB15k-237, WN18RR, CoDEx-Medium, and NELL-995. These lineages occur in the 57-KG panel. Comparisons involving the frozen score therefore audit a fixed upstream model on a panel that includes some of its pretraining lineages.

As a direct-lineage sensitivity, we remove the CoDEx, FB15k-237, FBNELL, NELL, and WN18RR provenance families. The local-versus-identifier and score-given-local effects remain positive in every retained family. This sensitivity specifically addresses known dataset lineage.

For score-only logistic regression, every fitted coefficient has positive orientation. Ranking the original ULTRA score directly consequently preserves its ordering and ties by construction. The complete rerun verifies exact equality between logistic and native ranks across all 684 combinations of dataset, context budget, and seed, with maximum rank difference zero.

Table 18: Attribution and sensitivity checks for the frozen ULTRA score. Effects report equalfamily MRR differences. Dataset and family counts indicate how often TABPFN-3 with the Full encoder performs below native ULTRA ranking. A dash denotes a quantity not used in the corresponding comparison.
<table><tr><td>Comparison</td><td>Effect</td><td>95% sensitivity</td><td>Datasets below native</td><td>Families below native</td></tr><tr><td>Global added after local + score</td><td>+0.005</td><td>[-0.001,+0.010]</td><td></td><td></td></tr><tr><td>Full TABPFN-3 minus native ULTRA</td><td>+0.006</td><td>[-0.053,+0.108]</td><td>48/57</td><td>14/16</td></tr><tr><td>Full minus native, excluding MetaFam</td><td>-0.041</td><td></td><td></td><td></td></tr></table>

Table 18 provides the attribution checks summarised in Section 5.3. Global PageRank, component, heat, and diffusion features add only +0.005 once the score is present and do not remain supported after correction. TABPFN-3 with the Full encoder underperforms native ranking on 48 of 57 dataset and 14 of 16 families. Its positive aggregate contrast is driven by MetaFam and becomes -0.041 when that family is excluded.

A paired sensitivity analysis using 512 sampled test triples per dataset preserves the sign of all five panel-level encoder contrasts used in the paper. Mean changes relative to the 64-triple evaluation range from −0.0015 to +0.0064 MRR. The 64-triple sample is therefore used for paired panel conclusions.

## D.10 Query-Evidence Strata

Section 5.4 groups directional evaluation queries by endpoint connectivity in the observed support graph. For query $q = ( h , r , t )$ , let $a _ { h  t } = 1$ when the support graph contains at least one observed edge $( h , r ^ { \prime } , t )$ with $r ^ { \prime } \neq r ,$ , and let $a _ { t  h }$ be defined analogously for $( t , r ^ { \prime } , h )$ . The held-out target edge is removed before these indicators are computed.

![](images/2f4729f59517ac7db0cde9057361a6d2d73210490a7a2135fabc53b5bc078c7e.jpg)  
(a) Logistic  
(b) CatBoost  
(c) TabPFN-3  
Figure 8: Reciprocal rank by observed endpoint connectivity. Panels (a)–(c) report equal-family MRR across 57 knowledge graphs for logistic regression, CATBOOST, and TABPFN-3. Queries are grouped as both directions, reverse only (candidate tail to head), direct only (head to candidate tail), or no pair edge after removing the held-out target edge. Within each group, bars report the Dyadic, Paths, and Local encoders from top to bottom. Error bars show 95% provenance-family bootstrap sensitivity intervals.

Table 19: Endpoint-connectivity strata. $a _ { h  t }$ and $a _ { t  h }$ indicate whether another observed relation connects the endpoints in the corresponding direction after removing the held-out target edge. Counts cover the 7,296 directional queries obtained from head and tail prediction for 64 test triples in each of 57 knowledge graphs. Percentages are computed over the fixed panel.
<table><tr><td>Group</td><td> $a _ { h  t }$ </td><td> $a _ { t  h }$ </td><td>Queries</td><td>Share</td></tr><tr><td>Both directions</td><td>1</td><td>1</td><td>244</td><td>3.3%</td></tr><tr><td>Reverse only</td><td>0</td><td>1</td><td>964</td><td>13.2%</td></tr><tr><td>Direct only</td><td>1</td><td>0</td><td>414</td><td>5.7%</td></tr><tr><td>No pair edge</td><td>0</td><td>0</td><td>5,674</td><td>77.8%</td></tr></table>

The Dyadic encoder contains other-relation pair count, reverse-pair count, and a same-entity indicator. The Paths encoder contains bounded two-hop and relation-conditioned path counts. The Local encoder combines these groups with relation, endpoint-degree, and community features. Figure 8 reports equal-family reciprocal rank within each stratum under the same evaluation queries, context, and seeds.

The connected strata contain 1,622 queries in total. Dyadic features outperform Paths when direct or reverse pair evidence is present. The no-pair-edge stratum contains 5,674 queries, for which Dyadic reciprocal rank falls to 0.001–0.002, Paths reaches 0.045–0.052, and Local reaches 0.043–0.094. These fixed-panel comparisons identify where feature groups carry predictive evidence and remain descriptive at the feature-group level.

## E Exploratory Analyses

The main analysis attributes performance through matched decoder comparisons. This section complements the paper’s pre-specified contrasts with post-hoc analyses of shared difficulty and query heterogeneity. These analyses provide additional context for the interpretations in Section 5.3 and Section 5.4.

![](images/6082ad9367ace3afc7101662e9b6d3ef19c16f2a11fc0214b419439034ff10eb.jpg)  
Figure 9: Context separability and candidate-pool size across the 57-KG panel. Each point denotes one knowledge graph under the filtered-uniform context with 200 examples per class. The horizontal axis reports candidate-pool size on a logarithmic scale, the vertical axis reports the strongest oriented single-column context ROC-AUC, and colour indicates observed TABPFN-3 MRR. Curves show equal values of the context-predicted rank.

## E.1 Context-Based Rank Heuristic

For numeric column $j ,$ let $n _ { + }$ and n denote the numbers of positive and negative context rows. We compute its context separation as

$$
A _ { j } = \frac { 1 } { n _ { + } n _ { - } } \sum _ { p = 1 } ^ { n _ { + } } \sum _ { q = 1 } ^ { n _ { - } } \left[ \mathbf { 1 } \big ( x _ { p , j } ^ { + } > x _ { q , j } ^ { - } \big ) + \frac { 1 } { 2 } \mathbf { 1 } \big ( x _ { p , j } ^ { + } = x _ { q , j } ^ { - } \big ) \right] ,\tag{9}
$$

where $x _ { p , j } ^ { + }$ and $x _ { q , j } ^ { - }$ are the values of column j for positive and negative context rows. We remove orientation by setting $A _ { j } ^ { * } = \operatorname* { m a x } ( A _ { j } , 1 - A _ { j } )$ . With m the mean number of valid evaluation candidates per query, we define

$$
\widehat { R } = ( 1 - \operatorname* { m a x } _ { j } A _ { j } ^ { * } ) m .\tag{10}
$$

Equation (10) combines the strongest single-column separation observed in the labelled context with candidate-set size. It is a descriptive difficulty heuristic. It uses only labelled-context features and candidate-set size, excluding constant and nominal identifier columns.

Figure 9 shows the two quantities entering the heuristic. Graphs with similar single-column separation can differ in predicted difficulty because they rank different numbers of candidates, while stronger separation can offset a larger candidate pool.

Applied without refitting at 200 examples per class, Equation (10) orders observed mean rank for TABPFN-3 at Spearman 0.957 across 455 dataset–encoder strata. The corresponding correlations are 0.973 for logistic regression and 0.957 for CATBOOST. Of the 456 possible numeric dataset–encoder strata, wiki\_mt2\_org with the Dyadic encoder is excluded because all three numerical columns are constant across all three seeds. Figure 10 shows that the same envelope holds when the strata are separated by decoder.

The heuristic captures a substantial component of shared difficulty, while encoder and decoder selection remain separate questions. None of the 56 benchmark-summary–configuration associations survives the joint max-|ρ| family-permutation sensitivity, and leave-one-family-out selectors do not improve on their in-fold majority baseline.

## E.2 Single-Column Context Separability

The column-level analysis in Figure 11 follows the evidence-family taxonomy in Table 12 and helps explain why the frozen score remains useful when added to local features. It is the strongest

![](images/6660889008ae8d7e095f7a5af5b4267f7543aed497eec5b5205d7d9a45768b75.jpg)  
Figure 10: Transport of the context-predicted rank. Panel (a) compares predicted and observed mean rank across 455 dataset–encoder strata for TABPFN-3; the solid curve and band summarise the observed strata. Panel (b) shows the same predicted-versus-observed comparison separately for TABPFN-3, logistic regression, and CATBOOST, left to right. Ranks are expressed as a percentage of the candidate pool on logarithmic axes, and dashed diagonals denote equality.

individual context separator on 55 of 57 graphs, whereas most constructed local and global columns lie closer to chance. This is a single-column diagnostic, while the main results concern multivariate and query-conditional effects beyond the strongest individual separator.

every input column's separability, ranked  
![](images/376d228e9399d2fe6acacbc66e8f3342ac9970a77e937183eb3e9d38fc423d5e.jpg)  
Figure 11: Single-column context separability across the 57-KG panel. Oriented ROC-AUC for every column of the Full encoder at 200 examples per class. Each point denotes one knowledge graph and heavy ticks denote column medians. Columns are ordered by median separability and grouped into the frozen ULTRA score, local relational structure, and global topological or spectral structure. The dashed reference marks chance separation at 0.5.

## E.3 Query-Level Variance and Structural Difficulty

The endpoint-connectivity analysis in Section 5.4 conditions on graph evidence available before prediction. We additionally describe how much reciprocal-rank variation is associated with query identity and the remaining experimental factors. A sequential group-mean decomposition is computed in the declared order query, encoder, decoder, context, and seed. Median shares across datasets are 0.250, 0.147, 0.003, 0.001, and less than 0.001, with 0.581 residual. Because the decomposition is sequential, these values are order-dependent descriptions.

The query–decoder interaction adds only 0.028 median explained variance, with maximum 0.076. This provides limited evidence for per-query decoder selection under the present experimental setup. Within datasets, target-side relation degree correlates positively with reciprocal rank at $\rho = + 0 . 5 1 \mathrm { t o } + 0 . 5 8$ , while anchor-side relation degree correlates negatively at $\rho = - 0 . 3 2 \mathrm { t o } - 0 . 2 3$ MetaFam is the only dataset in which both directions reverse.

![](images/7a21d9eceb9be6fb64c9d6bf7b4198d6ae76dc1341512849a7f077ed5675f6d8.jpg)  
Figure 12: Query-level variance and structural difficulty across the 57-KG panel. Panel (a) reports the sequential reciprocal-rank shares assigned to query identity, encoder, decoder, context, and seed, together with the residual share and the additional query–decoder interaction. The context and seed shares are combined only for display. Panel (b) reports within-dataset Spearman correlations between reciprocal rank and target- or anchor-side relation degree. The decomposition in panel (a) is descriptive and order-dependent.

## F Audit Reporting Checklist

Table 20 condenses the comparisons used throughout the paper into a reporting checklist. It links each attribution question in Section 3 to the failure mode that an end-to-end score would leave unresolved.

Table 20: Reporting checklist for graph-to-table audits. Each row links an attribution question to the failure left unresolved by an end-to-end score and the minimum matched comparison required to address it.
<table><tr><td>Question</td><td>Failure to avoid</td><td>Required comparison</td></tr><tr><td>Choose an encoder</td><td>Unique but non-reusable rows</td><td>Ceiling and singleton mass, with identifier rela- belling where relevant</td></tr><tr><td>Add a feature group</td><td>Supported shortcut or nuisance information</td><td>Base, permuted, and aligned encoders</td></tr><tr><td>Choose a decoder</td><td>Decoder-specific encoder ranking</td><td>Same rows, context, and evaluation across de- coders</td></tr><tr><td>Add an upstream score</td><td>Gain credited to the tabular readout</td><td>Nested encoders and native graph-model ranking</td></tr><tr><td>Interpret an average gain</td><td>Opposing query regimes hidden by averages</td><td>Paired query analysis grouped by observed graph evidence</td></tr></table>

## G Reproducibility

## G.1 Decoder Configurations

We report the default fixed decoder configurations used unless a decoder parameter is explicitly varied in a sensitivity analysis. Logistic regression uses StandardScaler fitted on context rows followed by C = 1 logistic regression. CATBOOST uses 500 iterations, depth 7, learning rate 0.05, and log loss. TABPFN-3 uses package version 8.1.0, the TabPFN-3 classifier checkpoint pinned by SHA-256 in the artefact, four estimators, fit\_with\_cache, and prediction batches capped at 50,000 rows. Automatic estimator scaling is disabled. A context-only sensitivity tunes logistic $C \in \{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ and CATBOOST depth {4, 7, 10} crossed with $\ell _ { 2 }$ leaf regularisation {3, 10}. Hyperparameter selection uses context data only.

## G.2 Categorical Encoding and Ranking

We next document the decoder-specific transformations applied after the shared feature matrices are constructed. In the context-robustness experiment, relation ID and the same-partition indicator are treated as nominal for every decoder. Logistic regression uses one-hot encoding, CATBOOST uses native categorical features, and TABPFN-3 receives categorical indices. The same-partition value is a binary indicator of whether the two entities share a partition. Scaling and categorical transformations are fitted on context rows only.

Ranking uses the least-compressed positive-class score exposed by each decoder. For logistic regression with the one-column ULTRA-score encoder, the fitted coefficient determines orientation and the original upstream score is ranked directly, preserving native ties exactly. Every paired decoder comparison receives identical feature matrices, labels, context indices, and evaluation indices.

## G.3 Recorded Artefacts and Statistical Units

We finally record the artefacts and replication units used to reproduce the reported estimates. The composition and 57-KG experiments record source revisions, model identities, seed streams, feature schemas, context and evaluation hashes, and prediction checksums. Reported uncertainty uses the replication unit declared in the corresponding experimental section, with graph, world, or provenancefamily units.

## GenAI Usage Statement

We used generative AI tools to assist with manuscript revision, experiment planning, and implementation of additional analysis code. All AI-assisted code was reviewed and tested, numerical results were regenerated under frozen experimental plans, and AI-assisted text and claims were checked and revised by the authors. The authors take full responsibility for the final manuscript and accompanying artefacts.