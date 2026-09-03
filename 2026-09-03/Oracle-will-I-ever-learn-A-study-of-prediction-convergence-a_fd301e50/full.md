# Oracle, will I ever learn? A study of prediction convergence and complementarity across link prediction models

Guillaume M´erou´e<sup>[0009−0004−7111−3202]</sup>, Fabien Gandon<sup>[0000−0003−0543−1232]</sup>, and Pierre Monnin<sup>[0000−0002−2017−8426]</sup>

Universit´e Cˆote d’Azur, Inria, CNRS, I3S, France firstname.lastname@inria.fr

Abstract. Knowledge graphs have become an important source of structured knowledge for Web applications, including search, question answering, and recommender systems. In these applications, link prediction can serve either as a prediction task itself or as a means to enrich incomplete knowledge graphs for downstream tasks. Interestingly, diferent link prediction models, or even diferent training runs of the same model, can produce substantially diferent predictions for the same query. This suggests a variability in the capture of the underlying knowledge by models, thus raising a fundamental question: to what extent do diferent models capture complementary knowledge, and how much of this knowledge could be recovered by combining them? We propose to measure model complementarity through the performance of an oracle that, for each query, selects the best prediction among a considered set of models, hence providing an upper bound on the performance achievable through model combination. Across several architectures and benchmarks, we find a substantial gap between individual models and their oracle, revealing that diferent models capture complementary knowledge. Yet, this complementarity rapidly saturates as more models are added, leaving a persistent subset of queries unsolved even by a large number of models. These findings reveal both the potential of model complementarity and a fundamental limit to what current link prediction models can collectively recover; thereby highlighting the need for further research to build robust Web applications.

## 1 Introduction

The Web exposes users to an ever-growing amount of information, making structuring, identifying, and ranking relevant information from large and heterogeneous collections of data a central challenge of modern Web systems (e.g., search, recommendation). That is why, knowledge graphs (KGs) are increasingly used as a source of structured knowledge in Web applications, e.g. in semantic search [12], question answering [33,34], and recommender systems [29,28]. KGs explicitly models entities and the relationships between them [11], and are typically defined within standard frameworks such as RDF [15], where knowledge is represented as triples $( h , r , t )$ , indicating that a relation r links a head entity h to a tail entity t.

However, real-world KGs inherently incomplete [13], limiting their usefulness and motivating several refinement tasks. Among them, link prediction (LP) aims to infer missing entities in incomplete triples of the form $( h , r , ? )$ or $( ? , r , t )$ within a learning-to-rank setting. A wide range of methods have been developed for this task, including KG embedding models [4,31,8], other neural approaches [7,37,32], and rule-based methods [9,18]. Over the years, continuous improvements in model architectures and training strategies, such as loss design, regularization, and negative sampling, have led to improvements in performance as measured by metrics such as MRR [22].

Despite substantial progress in link prediction, individual models remain imperfect and, more importantly, do not necessarily capture the same knowledge. Recent work [20,36] has shown that training the same model with diferent random seeds can lead to diferent model instances producing substantially diferent predictions. This phenomenon has been quantified through metrics such as ambiguity [36] and prediction overlap [20], with results respectively showcasing that some instances may successfully rank the ground-truth entity among their top-K predictions while others fail to do so, and that a substantial portion of the top-K predictions may difer.

This suggests that diferent models and model instances may capture complementary parts of the knowledge that can be inferred from a KG, making model combination a potential means of increasing knowledge coverage. This intuition is consistent with the broader literature on ensemble learning for link prediction, where combining multiple predictors can exploit complementary predictions and improve performance [14,30,10]. However, despite these improvements, ensemble methods remain far from perfect performance. Besides, to the best of our knowledge, no prior work has quantified how much additional performance could theoretically be obtained by fully exploiting the diversity of predictions across instances and models, nor how this potential evolves as an increasing number of instances is considered. This motivates the following research questions aiming at assessing model complementarity through performance gains:

RQ1. To what extent can multiple instances of a single model provide performance gains over a single instance?

RQ2. To what extent can diferent models provide performance gains beyond those obtained from multiple instances of a single model?

RQ3. How does the performance gain evolve as the number of instances increases?

To address these questions, we introduce an oracle framework, defined as an idealized setting that selects, for each test triple, the best prediction among a set of instances. This framework provides an upper bound on the performance achievable through instance selection and allows us to quantify the potential performance gains of ensemble methods.

With this oracle framework, we identify three distinct performance gaps. For RQ1, the Instance–Oracle Gap highlights that exploiting seed-induced diversity alone is suficient to yield substantial gains over individual instances. For RQ2, the Model–Cross-model Gap shows that combining diferent model architectures provides additional advantages, as cross-model ensembles reach the performance of the best single-model oracle without prior model selection. Finally, for RQ3, the Asymptotic Gap captures the performance levels that remain unattainable as more instances are added. Indeed, oracle performance quickly saturates, leaving a subset of queries that none of the considered models can solve.

The remainder of this paper is organized as follows. We first review related work in Section 2, before formalizing our oracle framework in Section 3 and describing the experimental setup in Section 4. Section 5 then presents the results, structured around the three research questions and their corresponding performance gaps. Section 6 discusses the findings, while Section 7 concludes the paper.

## 2 Related Work

## 2.1 Link Prediction Models

Link prediction is a fundamental task for completing knowledge graphs and supporting downstream applications such as search, question answering, and recommendation [34,28].

Models for link prediction can be unified under the common formulation of a scoring function $\varphi ( h , r , t )$ , which evaluates the plausibility of a triple $( h , r , t )$

Over the past decade, the dominant approach has been KGEMs, which aim to embed a KG within a d-dimensional continuous vector space while preserving its structural regularities. The scoring function operates over this embedding space, which is learned during training by maximizing the scores of observed triples while minimizing those of negative ones generated by corrupting positive triples. KGEMs are commonly categorized into three families based on the design of their scoring functions: geometric models $( \mathrm { e . g . }$ , TransE [4], RotatE [25], BoxE [1]), tensor factorization models $( \mathrm { e . g . }$ , RESCAL [21], DistMult [31], ComplEx [27]), and neural network models that learn the scoring function (e.g., ConvE [8], HittER [6]).

Beyond the KGEMs introduced above, several approaches adopt distinct mechanisms while still relying on learned parameters. NBFNet [37] formulates the task as a path search problem, generalising the Bellman-Ford algorithm within a graph neural network. MINERVA [7] models the task as a sequential decision process, where an agent learns to navigate the knowledge graph by following relation paths from a source entity to a target entity, using reinforcement learning. Alternatively, NeuralLP [32] compiles logical inference into diferentiable tensor operations, enabling the joint learning of parameters and logical rule structures through gradient descent.

Alternatively to these neural approaches, symbolic rule mining methods extract explicit patterns directly from the graph. Among them, AMIE [9] follows a top-down search guided by confidence measures, while AnyBURL [18] adopts a bottom-up strategy based on random walks.

## 2.2 Ensemble Learning for Link Prediction

Recent work has explored ensemble learning strategies to improve link prediction performance. Xu et al. [30] study ensembles composed of multiple instances of the same model and show that variability induced by diferent random seeds is suficient to outperform a single model with an equivalent number of parameters. In particular, they exhibit that such ensembles can collectively capture relational patterns that are theoretically inaccessible to individual instances, such as symmetry in TransE. SnapE [24] extends this idea by constructing ensembles from intermediate training snapshots, aiming to retain the benefits of ensembling without increasing training time.

These observations relate to the fact that KGEMs difer in the relational patterns they can theoretically capture (e.g., symmetry, intersection, composition), as a direct consequence of their scoring functions [25,1]. This has motivated ensemble approaches that aim to overcome the limitations of individual models by combining their strengths. Krompass and Tresp [14], Gregucci et al. [10], and Yue et al. [35] argue that such combinations lead to improved predictive performance. More broadly, Meilicke et al. [19,17] show that KGEMs and rule-based methods can also efectively complement each other.

While these studies consistently report gains over the best single instance, the improvements in terms of MRR remain relatively modest [10,35]. Moreover, this body of work leaves open the questions of a potential upper bound in performance when aggregating model instances, whether this upper bound is achievable, and whether existing approaches already operate close to this limit. This motivates our proposed oracle framework to characterize such an upper bound, and measure the gaps between individual model instances and their oracle, opening further research on model aggregation strategies.

## 3 Measuring Performance Upper Bounds with Oracles

Let us denote a KG as $\mathcal { K } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ , where $\mathcal { E }$ is a set of entities, R a set of relations, and $\mathcal { T } \subseteq \mathcal { E } \times \mathcal { R } \times \mathcal { E }$ a set of triples encoding facts. Each triple $( h , r , t )$ consists of a head entity $h .$ a relation r, and a tail entity t, e.g., (France, hasCapital, Paris). In link prediction, $\tau$ is partitioned into fixed training $( \mathcal { T } _ { \mathrm { t r a i n } } )$ , validation $( \mathcal { T } _ { \mathrm { v a l } } )$ and test $( T _ { \mathrm { t e s t } } )$ sets. Given a test triple $( h , r , t ) \in \mathcal { T } _ { \mathrm { t e s t } }$ , the task consists in predicting the missing entity in queries of the form $( h , r , ? )$ or $( ? , r , t )$ by scoring and ranking all entities, such that the ground-truth entity is ranked as highly as possible. Performance is thus evaluated using rank-based metrics such as Mean Rank (MR), Mean Reciprocal Rank (MRR), and Hits@K (typically $K \in \{ 1 , 3 , 1 0 \} ,$ ).

In the literature, several models have been proposed to address this task; in this work, we restrict our study to a subset of models denoted $\mathcal { M }$ . The training of a model depends on the chosen architecture, a hyperparameter configuration that is often chosen as the one maximizing the rank-based metrics on the validation set, and a random seed to control stochastic factors such as negative sampling or embedding initialization. We call a trained model given these characteristics an instance, that we formally define as follows:

![](images/20c3d0635265850c21e11d98e27e71adaf87ceef9a5356c8582b709cb5116a78.jpg)  
Fig. 1: Per-query selection mechanism of oracles. Although individual instances achieve similar aggregate MRR, their performance difers at the query level $( q _ { 1 } , q _ { 2 } , q _ { 3 } )$ . By selecting the best rank per query across instances, the oracle constructs optimal pseudo-instances at the model group level $\left( \mathcal { O } _ { \mathrm { M o d e l } } \right)$ and crossmodel group level $\left( \mathcal { O } _ { \mathcal { M } } \right)$ , leading to higher overall performance.

Definition 1 (Instance).

An instance ${ \cal I } = { \cal I } n s t a n c e ( M , 7 , C , \mathfrak { S } )$ corresponds to a model $M \in \mathcal { M }$ trained on dataset T with hyperparameter configuration C and random seed S. When unambiguous, we write $I _ { M } ^ { \mathfrak { S } }$

Even with M, T , and C fixed, training is stochastic: the learned representations depend on the random seed S [20]. In standard practice, diferent instances obtained with distinct seeds are commonly treated as interchangeable, since aggregate link prediction metrics usually exhibit only limited variation across runs [20]. Consequently, experimental results are often reported from a single run or averaged over a few runs, and the specific influence of the seed is largely disregarded.

However, prior work has shown that such instances can yield divergent predictions [20,36]. This phenomenon can be illustrated in Figure 1, where two instances of Model 1 achieve the same MRR (0.4), yet difer at the query level: the first instance ranks the first query correctly while the second does not, and the opposite holds for the second query. As previously introduced, this variability motivates the use of an ensemble of models to benefit from the diferent knowledge captured by the diferent instances [30].

In our work, we further investigate the performance gain brought by such groups of models. To address the two settings of in-model and cross-model complementarity, we consider groups of models that can involve instances of the same model (Definition 2), or instances of diferent models (Definition 3), respectively.

Definition 2 (Model Group). Let $M \in \mathcal { M }$ be a model. A model group associated with M is a finite set of independently trained instances of M:

$$
G _ { M } = \{ I _ { M } ^ { \mathfrak { S } _ { 1 } } , \dots , I _ { M } ^ { \mathfrak { S } _ { l } } \} ,
$$

where each instance $I _ { M } ^ { \mathfrak { S } _ { i } }$ is obtained using a diferent random seed ${ \mathfrak { S } } _ { i }$

Definition 3 (Cross-Model Group). Let $\mathcal { M } ^ { \prime } \subseteq \mathcal { M }$ be a set of models. A cross-model group is any finite set of instances of models in $\mathcal { M } ^ { \prime }$ :

$$
G _ { \mathcal { M } ^ { \prime } } = \{ I _ { M _ { 1 } } ^ { \mathfrak { S } _ { 1 } } , \ldots , I _ { M _ { n } } ^ { \mathfrak { S } _ { n } } \} , \quad w i t h \ M _ { i } \in \mathcal { M } ^ { \prime } \ \forall i ,
$$

To characterize the theoretical performance upper-bound of such groups, we model their theoretical best performing behavior as an oracle formally defined as follows.

Definition 4 (Oracle). Let us denote Q the set of queries. Given a group G, the oracle associated with G is the mapping

$$
\begin{array} { r l } & { \mathcal { O } _ { G } : \mathcal { Q } \longrightarrow \mathbb { Z } ^ { + } } \\ & { ~ \quad \quad \quad q \longmapsto \mathcal { O } _ { G } ( q ) = \underset { I \in G } { \operatorname* { m i n } } r a n k ( I , q ) . } \end{array}
$$

where rank $( I , q )$ returns the rank assigned by instance I to the ground-truth entity for a query q.

In other words, the oracle can be interpreted as a pseudo-instance that, for each query, achieves the best rank obtained by the instances in G. For clarity, we denote the oracle of a model group $G _ { M }$ or a cross-model group $G _ { \mathcal { M } ^ { \prime } }$ as $\mathcal { O } _ { M } \equiv \mathcal { O } _ { G _ { M } }$ and $\mathcal { O } _ { \mathcal { M } ^ { \prime } } \equiv \mathcal { O } _ { G _ { \mathcal { M } ^ { \prime } } }$ , respectively.

Figure 1 provides examples of oracle predictions for Model 1. For the first query, the oracle selects the first instance, which ranks the ground-truth entity higher than the second instance (1 vs. 10). The opposite occurs for the second query. For the third query, both instances yield identical ranks, and the oracle selection is therefore arbitrary. The same procedure applies to Model 2. Figure 1 also illustrates cross-model group oracles with 2 and 4 instances per group. The selection mechanism remains identical, but the oracle can now choose among instances from diferent models. For example, ${ \mathcal { O } } _ { { \mathcal { M } } } ^ { 4 }$ will first select instance 1 and then instance 2 from Model 1, and then select the prediction of instance 1 from Model 2, which ranks higher than the Model 1 instances (2 vs. 10).

This pseudo-instance interpretation allows us to evaluate $\mathcal { O } _ { G }$ using the same metrics as for standard instances. In particular, the Mean Reciprocal Rank of the oracle is defined as

$$
\mathrm { M R R } _ { \mathscr { O } _ { G } } = \frac { 1 } { | \mathscr { Q } | } \sum _ { q \in \mathscr { Q } } \frac { 1 } { \mathscr { O } _ { G } ( q ) } .
$$

$\mathrm { M R R } _ { { \mathscr { G } } }$ therefore represents the upper-bound performance achievable if, for each query, one could a priori select the best-performing instance in G. This upper-bound thus characterizes the theoretical performance limit of a group of instances w.r.t. perfect prediction, and enables to assess the gap between instances and this upper-bound, paving the way toward enhanced strategies for result aggregation.

To illustrate, in Figure 1, although the instances of Model 1 and Model 2 individually obtain MRR values of 0.40 and 0.44, respectively, the collective performance measured by the oracle is higher. In particular, two instances of Model 1 can jointly achieve an MRR of 0.70, while Model 2 reaches 0.50. When combining the first two instances of Model 1 and Model 2, the resulting group achieves an MRR of 0.61, and this value further increases to 0.83 when considering the full set of instances.

## 4 Experimental Settings

Models. In our experiments, we consider a diverse set of representative models for link prediction, spanning several methodological families. First, we include a collection of widely adopted knowledge graph embedding models:

$$
\mathcal { M } _ { \mathrm { K G E } } = \{ \mathrm { T r a n s E , C o m p l E x , D i s t M u l t , C o n v E , R o t a t E , R E S C A L } ,
$$

BoxE}.

We then extend this set with neural approaches relying on alternative formalisms, such as path-based reasoning using reinforcement learning, GNN, or

neural rule induction:

$$
\begin{array} { r l r l r l } { \mathcal { M } _ { \mathrm { N e u r a l } } } & { } & & { = } & & { \quad \mathcal { M } _ { \mathrm { K G E } } } & { \cup } & { } & { \{ \mathrm { N e u r a l L P , N B F N e t , M I N E R V A } \} . } \end{array}
$$

In addition, we consider two symbolic rule-based methods for link prediction:

$$
{ \mathcal { M } } _ { \mathrm { S y m b o l i c } } = \{ { \mathrm { A M I E , A n y B U R L } } \} .
$$

Finally, the complete set of models considered in this work is defined as:

$$
\mathcal { M } = \mathcal { M } _ { \mathrm { N e u r a l } } \cup \mathcal { M } _ { \mathrm { S y m b o l i c } } .
$$

For model implementation, we rely on reference frameworks and implementations, depending on model availability and in decreasing order of priority: LibKGE [5] (for TransE, DistMult, ComplEx, ConvE, RotatE, and RESCAL), PyKEEN [2] (for BoxE), TorchDrug<sup>1</sup> (for NBFNet and NeuralLP), PyClause [3] (for AMIE and AnyBURL). Finally, we use the original implementation of MIN-ERVA [7], as it is not available in the aforementioned frameworks. Note that the original implementation of MINERVA trains exclusively for tail prediction. To allow comparison with the other considered models, we choose to report results for both head and tail prediction. Specifically, we reformulate each headprediction query $( ? , r , t )$ as the equivalent tail-prediction query $( t , r ^ { - 1 } , ? )$ . This formulation for head prediction is valid since MINERVA learns to navigate the graph using both relations and their inverses. Also note that AMIE is included in cross-model comparisons, but difers fundamentally from the other approaches as it relies on a fully deterministic rule mining process.

Datasets. We consider three standard datasets from the literature: WN18RR [8], a subset of the lexical KG WordNet characterized by its hierarchical structure; and FB15k-237 [26] and CoDEx-S [23], derived from Freebase and Wikidata, respectively, two prominent Web KGs containing large collections of real-world facts. We use the standard splits of these datasets, with PyKEEN’s filtering to remove test triples involving entities unseen during training, which removes 28 out of 20, 466 test triples from FB15k-237, 210 out of 3, 134 from WN18RR, and none from CoDEx-S.

Instances. For each (model, dataset) pair, we train 20 independent instances using random seeds ranging from 0 to 19. Instances that fail to converge, defined as achieving a validation MRR strictly below 0.05, are discarded from the analysis. Consequently, we exclude 1, 9, and 9 NeuralLP instances on WN18RR, FB15k-237, and CoDEx-S, respectively.

For TransE, DistMult, ComplEx, ConvE, RotatE, and RESCAL, we use the oficial configurations provided in the LibKGE repository for FB15k-237 and WN18RR. These configurations result from the hyperparameter tuning procedure conducted by Rufinelli et al. [22]. For CoDEx-S, we use the configurations provided in the same repository for TransE, ComplEx, and ConvE, obtained following the hyperparameter tuning procedure as in Safavi et al. [23]. As no dataset-specific configurations are provided for DistMult and RotatE on CoDEx-S, we reuse their FB15k-237 configurations, given the similarity between the two datasets.

For BoxE, MINERVA, and NBFNet, we follow the same principle. We use the oficial configurations provided by their respective authors for FB15k-237 and WN18RR [1,7,37], and reuse the FB15k-237 configurations for CoDEx-S. For NeuralLP, we use the same configuration across all datasets, following the configuration provided in the oficial TorchDrug tutorial<sup>2</sup>.

Finally, AnyBURL and AMIE, implemented with PyClause, are run using the framework’s default configurations.

Evaluation Metrics. To evaluate instance performance, we focus on the standard Mean Reciprocal Rank (MRR), Hit@1 and Hit@10 metrics. We use filtered ranks, removing all other known true triples from the candidate set, and the pessimistic tie convention, assigning tied candidates the worst rank in their tie group.

Experimental Protocols. We devise specific experimental protocols to answer each of the research questions.

RQ1 (Section 5.1). For each model $M \in { \mathcal { M } }$ , we consider its associated group $G _ { M }$ composed of 20 independently trained instances. We compare the average performance across instances, ${ \overline { { \mathrm { M R R } } } } _ { G _ { M } } ,$ , with the oracle performance over the group, $\mathrm { M R R } _ { { \cal M } _ { N } }$

RQ2 (Section ${ \bf 5 . 2 } )$ . We evaluate the performance of cross-model groups. To ensure a fair comparison with single-model groups, we control the total number of instances. Specifically, we sample the same number of instances from each model in a given family $\mathcal { M } ^ { \prime }$ , such that the total number of selected instances k remains close to 20. The corresponding oracle performance is denoted $\mathrm { M R R } _ { \mathcal { O } _ { M ^ { \prime } } } ^ { k }$ As this sampling is stochastic, we repeat the process 100 times and report the average oracle performance.

To relax the uniform allocation constraint, we construct a group of 20 instances using a greedy procedure over M, iteratively selecting the instance that yields the largest increase in oracle performance, regardless of its model. Once the greedy procedure is completed, we repeatedly sample new instances while preserving the model proportions of the selected group. This allows us to disentangle the efect of the resulting model allocation from that of the particular instances selected during the greedy procedure, and thus assess whether the observed gain is attributable to a promising model allocation rather than to specific instances. We perform 100 such samplings and report the resulting average oracle performance as MRR<sup>20,greedy</sup><sub>O</sub> .

RQ3 (Section $\mathbf { 5 . 3 } )$ . We analyze how oracle performance evolves as the number of instances increases. Given the instances of a group $G ,$ we evaluate $\mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { k }$ for $k \in \{ 1 , \ldots , | G | \}$ by progressively considering subsets of size $k .$ . Since the results depend on the order in which instances are added, we sample 100 random permutations and report the average oracle performance for each value of k.

Our code is available on GitHub.<sup>3</sup>

## 5 Uncovering Performance Gaps

Our objective is to quantify how much can be gained by moving from a single instance to groups of instances and models. To this end, we structure our analysis around three gaps. We first study the Instance–Oracle Gap, measuring how much a group of instances of the same model can collectively outperform a single instance (Section 5.1). We then examine the Model–Cross-Model Gap, capturing the additional gains brought by combining diferent models (Section 5.2). Finally, we analyze the Asymptotic Gap, highlighting that performance converges below perfect accuracy as the number of instances increases (Section 5.3).

## 5.1 Instance–Oracle Gap: Performances of Instances Versus Their Oracle

Given a model M, we compare the average MRR of the instances within a group, $\overline { { \mathrm { M R R } } } _ { G _ { M } }$ , with the MRR of the oracle of the same group, $\mathrm { M R R } _ { { \cal M } }$ . This enables to assess the impact of seed-induced variability on performance. Results are reported in Table 1 (and in Appendix C for Hit@1). When aggregating individual MRR scores, the standard deviation across seeds is close to zero for most models. Such performance stability could suggest limited diversity across runs and therefore negligible room for improvement beyond single-instance performance. However, substantial gaps consistently appear between $\overline { { \mathrm { M R R } } } _ { G _ { M } }$ and $\mathrm { M R R } _ { { \cal M } }$ revealing a markedly diferent picture. For example, on CoDEx-S, ConvE improves from 0.45 to 0.64, DistMult from 0.42 to 0.56, and ComplEx from 0.46 to 0.60. Across all datasets, these gains often exceed 10 points and can reach up to nearly 20 points, demonstrating that instances trained with diferent seeds solve partially disjoint subsets of queries.

Moreover, although strong instance-level performance tends to be associated with higher oracle performance, this relationship does not always hold. For instance, on CoDEx-S, ComplEx achieves a higher averaged MRR than ConvE (0.46 vs. 0.45), yet exhibits a lower oracle performance (0.60 vs. 0.64). This suggests that oracle gains depend not only on instance-level performance alone but also on the diversity of correctly predicted queries across instances.

Our results thus reveal a clear performance gap between the predictive ability of a single instance and that of a collection of instances. The potential gain from exploiting this collection is substantially larger than what is achieved by current ensemble learning methods [10,35]. This gap highlights the potential for improved aggregation and ensemble strategies that can better exploit the complementary predictions of diferent instances, motivating further work toward practical methods that approach the oracle’s performance.

Table 1: Averaged and Oracle MRR across datasets, model groups and crossmodel groups.
<table><tr><td rowspan=1 colspan=7>Model          WN18RR FB15k-237 CoDEx-S</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=6 colspan=6> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { T r a n s E } } }$      $. 2 4 \pm . 0 0$     $. 3 1 \pm . 0 0$     $. 3 5 \pm . 0 0$ .45        .43.34 ± .00  .42 ± .00.46        .56.34 ±.00  .45 ± .00.47        .64</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=5>MRROTransE.27</td></tr><tr><td rowspan=1 colspan=5>MRRGDistMult .48 ± .00</td></tr><tr><td rowspan=5 colspan=1>KGE</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>MRRODistMult.53</td></tr><tr><td rowspan=1 colspan=5>MRRGConvE .47 ±.00</td></tr><tr><td rowspan=1 colspan=5>MRROConvE.54</td></tr><tr><td rowspan=2 colspan=5> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { R o t a t E } } }$     .51 ± .00.56</td><td rowspan=2 colspan=1>.33 ± .00  .42 ± .00.42        .53</td></tr><tr><td rowspan=1 colspan=1>MRRORotatE</td></tr><tr><td rowspan=9 colspan=1>Neua</td><td rowspan=1 colspan=2> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { C o m p l E x } } }$ </td><td rowspan=1 colspan=3>.51 ± .00</td><td rowspan=1 colspan=1>)</td></tr><tr><td rowspan=1 colspan=5> $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { C o m p l e x } } }$       .60</td><td rowspan=1 colspan=1>.46        .60</td></tr><tr><td rowspan=1 colspan=5> $\operatorname { M R R } _ { G _ { \mathrm { R E S C A L } } }$    .50 ± .00</td><td rowspan=1 colspan=1>.36 ± .00  .40 ± .00</td></tr><tr><td rowspan=1 colspan=5> $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { R E S C A L } } }$       .59</td><td rowspan=1 colspan=1>.52        .54</td></tr><tr><td rowspan=1 colspan=5> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { B o x E } } }$       .48 ± .00</td><td rowspan=1 colspan=1>.35 ± .00  .42 ± .01</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { M R R } _ { { \mathcal { O } } _ { \mathrm { B o x E } } }$ </td><td></td><td rowspan=1 colspan=3>.55</td><td rowspan=1 colspan=1>.42        .50</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { M I N E R V A } } }$ </td><td></td><td rowspan=2 colspan=3>.43 ± .01</td><td rowspan=5 colspan=1>.17 ± .00  .29 ± .01.32        .47.21 ± .01  .27 ± .01.38        .44.41 ± .00  .36 ± .01.58        .53</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { M I N E R V A } } }$ </td><td></td><td rowspan=2 colspan=3>.49.34 ± .04</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { N e u r a l L P } } }$ </td><td></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=5> $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { N e u r a l L P } } }$      .53 $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { N B F N e t } } }$    .59 ± .00.68</td></tr><tr><td rowspan=1 colspan=1>MRRONBFNet</td></tr><tr><td rowspan=1 colspan=7>Symqoie                       .48 ± .00  .25 ± .00  .30 ± .00 $\underline { { \mathrm { M R R } } } \mathcal { O } _ { \mathrm { A n y B U R L } }$ .49.34.31 $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { A M I E } } }$ .43 ± .00.20 ± .00.23 ± .00 $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { A M I E } } }$ .43.20.23</td></tr><tr><td rowspan=2 colspan=7> $\mathrm { M R R } _ { \mathcal { O } _ { K G E } } ^ { 2 1 }$        $. 6 2 \pm . 0 0$     $. 5 1 \pm . 0 0$     $. 6 3 \pm . 0 0$ Croooms-mdel     $\mathrm { M R R } _ { \mathcal { O } _ { N e u r a l } } ^ { 2 0 }$      $. 6 7 \pm . 0 0$     $. 5 8 \pm . 0 0$     $. 6 6 \pm . 0 0$  $\mathrm { M R R } _ { \mathcal { O } _ { S y m b o l i c } } ^ { 2 0 }$  $. 4 9 \pm . 0 0$  $. 3 3 \pm . 0 0$  $. 3 2 \pm . 0 0$  $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 4 }$  $. 6 7 \pm . 0 0$  $. 5 8 \pm . 0 0$  $. 6 6 \pm . 0 0$  $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 0 , \mathrm { g r e e d y } }$  $. 6 9 \pm . 0 0$  $. 6 0 \pm . 0 0$  $. 6 7 \pm . 0 0$  $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 4 0 }$            .73        .69        .76</td></tr><tr><td rowspan=1 colspan=2></td></tr></table>

## 5.2 Model–Cross-Model Gap: Benefits of Mixing Models

Prior work on ensemble learning [10] assumes that structural diferences across models constitute a complementary source of diversity. Building on this observation, this section investigates the gains associated with cross-model grouping, as well as the complementarity between models. Results are reported in the last section of Table 1.

Cross-model groups generally achieve higher oracle performance than most model groups. However, they do not systematically outperform the best singlemodel oracle. For example, on FB15k-237, the oracle performance obtained with 20 instances of NBFNet is identical (0.58) to that obtained with 24 instances (three per model) drawn from M. In some cases, cross-model oracles may underperform the best individual model group; for instance, ConvE achieves a higher oracle score than the M<sub>KGE</sub> cross-model group on CoDEx-S (0.64 vs. 0.63).

Nevertheless, cross-model group oracles tend to closely approach the best single-model oracle. Interestingly, the latter can vary across datasets: within M<sub>KGE</sub>, the best single-model oracle is ComplEx on WN18RR, RESCAL on FB15k-237, and ConvE on CoDEx-S. This observation highlights the practical relevance of cross-model groups. Indeed, in practice, it is not possible to determine in advance which specific model will perform best on a given dataset. Adopting a cross-model approach thus allows to expect the best performance without searching for the best performing single model, under the assumption that oracle performance can be attained

We also observe a remarkably low variance in cross-model oracle scores despite the use of sampling. This suggests that, as with the stability of MRR across instances, diversity is uniformly distributed: instances appear largely interchangeable, and substituting one instance for another has limited impact on overall complementarity.

Enforcing a uniform composition of models within cross-model groups may be restrictive. To relax this constraint, we consider $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 0 , \mathrm { g r e e d y } }$ , which constructs a group by greedily selecting instances so as to maximize oracle performance (see the experimental protocol for RQ2 in Section 4). Across all datasets, the oracle of the resulting group consistently outperforms (albeit with a modest margin) the best single-model oracle (0.69 vs. 0.68 on WN18RR, 0.60 vs. 0.58 on FB15k-237, and 0.67 vs. 0.64 on CoDEx-S). This result confirms that cross-model groups benefit from model complementarity, while suggesting that a heterogeneous composition of models may bring additional performance gains, with models providing more diversity than others. This phenomenon is illustrated in Figure 2, which depicts the composition of the greedily constructed group. A clear predominance of NBFNet emerges, consistent with its strong individual performance on FB15k-237 and WN18RR. However, other models, including lower-performing ones such as NeuralLP, TransE, and MINERVA, are also selected on FB15k-237. This highlights that performance is not suficient to characterize the complementary contribution of a model to a group: a model with lower individual performance may still provide substantial complementarity by correctly predicting queries that are missed by stronger models.

![](images/1a00cc53d8f27626af40f531fab622168e78dcec6e700caa781a0a05e7fb9dd6.jpg)  
(a) WN18RR

![](images/fa60f1366a7c830800dcabc9009e9b47ce79b0183ecb78fed5f29dc831cd866c.jpg)  
(b) FB15k-237

![](images/a065631386fe9fa1c5f3a0db0a622815e8549efbb05d5ab49d505982e803b4e6.jpg)  
(c) CoDEx-S  
Fig. 2: Greedy composition across datasets.

To better qualify this diversity, we analyze whether diferent model oracles succeed on the same queries. To this end, we reuse model-group oracles and binarize their predictions using Hit@1, indicating whether a given oracle correctly answers a query. For each query, we then count the number of model oracles that successfully resolve it, while also tracking the contribution of each model. The results are presented in Figure 3.

![](images/895fb1aa18485f143efd84d0d2ce6cd1ea82651cf8e5c0879bafdcaff15c976b.jpg)  
Fig. 3: Distribution of queries according to the number of model oracles achieving Hit@1

First, we observe that a substantial fraction of queries, approximately one third across all datasets, are not solved by any model. Furthermore, the proportion of queries solved by only a small number of model oracles (one to three) is significant, accounting for nearly 20% on FB15k-237 and CoDEx-S, and about 15% on WN18RR. We also observe that WN18RR exhibits a bimodal distribution, with queries either unsolved by all models or solved by nearly all (except TransE). Finally, the distribution further reveals that, although NBFNet dominates on FB15k-237 and WN18RR, other models also contribute in nonnegligible proportions. A more fine-grained view of these interactions is provided by the upset plots in Appendix B. Overall, these observations confirm that, from an oracle perspective, complementarity can ensure strong performance, and even yield additional gains when suficiently diverse groups are considered.

The previous results motivated an exploratory analysis of diversity across instances. More precisely, we measure the redundancy between instances as the Jaccard distance computed over the sets of queries for which each instance achieves a Hit@1 of 1 on FB15k-237. We then visualize this redundancy using a dendrogram (Figure 4), where the distance between two instances reflects the similarity of their predictions. Instances merging closer to leafs are thus more similar. Pairwise distances between instances are computed once, followed by agglomerative hierarchical clustering (UPGMA, average linkage).

This visualization highlights several key patterns. First, instances from the same model form tight clusters, indicating that identical architectures tend to learn similar queries. Second, NBFNet clearly separates from other models, suggesting that it captures distinct queries. The remaining branch further splits into two groups: one containing the full $\mathcal { M } _ { \mathrm { K G E } }$ family, where DistMult and its generalization ComplEx appear particularly close, and a second cluster mixing symbolic and neural models. The latter further divides into two subgroups: one corresponding to $\mathcal { M } _ { \mathrm { S y m b o l i c } } ,$ where AMIE exhibits no variation across instances, and another grouping NeuralLP and MINERVA. The proximity of NeuralLP to symbolic models can be explained by its joint learning of embeddings and rules, while the position of MINERVA suggests that path-based reasoning shares similarities with rule-based approaches.

The observation that the distance between two NBFNet instances is comparable to that between instances of diferent KGE models, or even between instances from the cluster mixing symbolic and neural approaches, indicates substantial diversity among NBFNet instances. This diversity may explain why NBFNet accounts for a large proportion of the models selected by the greedy procedure, in addition to its strong individual performance. A similar trend is observed for RESCAL within the KGE family, where it appears to be among the most diversified models.

In this section, we have shown that cross-model grouping provides a reliable way to approach the best single-model oracle, without requiring dataset-specific model selection. We further observed that models exhibit some degree of complementarity in their predictions, which can be leveraged to yield additional performance gains. This complementarity therefore provides a natural basis for developing ensemble methods that exploit the diversity of predictions across different model architectures. A further research avenue lies in identifying a priori highly complementary groups.

![](images/db2cfa45d2b9891d585ef39c09792b5d1e5cc0dccea683e1723fd9a54ac393fe.jpg)  
Fig. 4: Dendrogram of instance similarities on FB15k-237.

## 5.3 Asymptotic Gap: Performance Limit under Large Groups

Until now, our analysis has focused on groups with a fixed and limited number of instances, ensuring comparability across models and groups. However, the behavior of $\dot { \mathrm { M R R } } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 4 0 }$ suggests that increasing the number of instances can further improve performance, raising the question of how oracle performance evolves as the group size grows.

To address this, we study the evolution of $\mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { k }$ as a function of the number of instances k. Figure 5 shows the observed evolution of $\mathrm { M R R } _ { \mathcal { O } _ { M } } ^ { k }$ on FB15k-237 for each model $M \in \mathcal { M }$ (solid lines) up to our empirical limit of $k \leq 2 0$ . We additionally report $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 1 2 }$ and $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 4 }$ , corresponding to cross-model groups with one and two instances per model, respectively. Similar trends are observed on other datasets (see $\mathrm { G i t H u b ^ { 3 } } )$ . Because multiple instance orderings are possible, we sample 100 diferent permutations and report the mean trajectory, along with the best and worst cases. Results indicate that oracle performance depends primarily on the number of instances rather than their ordering, reinforcing the view that diversity is uniformly distributed across runs rather than concentrated in a few specific instances.

![](images/8ec318d37de5705321ebcca642ede1ab7c9d0c08eb177a570d9bcf463c4c7a30.jpg)  
Fig. 5: Evolution of oracle MRR as a function of the number of instances k on FB15k-237.

Overall, the figure highlights a clear diminishing-return efect: the marginal gain of each additional instance decreases rapidly as k increases. To characterize this trend beyond the observed $k \leq 2 0$ , we model the marginal performance gain using a power law:

$$
\begin{array} { r } { \varDelta \mathrm { M R R } ( k ) = \mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { k + 1 } - \mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { k } = { A k ^ { - m } } . } \end{array}
$$

Taking the logarithm of both sides gives log ∆MRR(k) = log A − m log k. We therefore estimate log A and m using ordinary least-squares regression on the log-transformed observations. The resulting fits obtain $R ^ { 2 } > 0 . 9 9$ across almost all considered models<sup>4</sup>, as illustrated in Figure 6.

For a given number of instances k, the oracle performance can then be reconstructed by summing the marginal gains:

$$
\mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { k } = \mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { 1 } + \sum _ { i = 1 } ^ { k - 1 } \varDelta \mathrm { M R R } ( i ) \approx \mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { 1 } + A \sum _ { i = 1 } ^ { k - 1 } i ^ { - m } .
$$

This provides a simple extrapolation of oracle performance beyond the observed range.

Crucially, we fit $( A , m )$ separately for each of the 100 sampled permutations rather than on their mean trajectory. This turns the extrapolation into a distribution of possible performance trajectories, from which we report the median and 95% interval. The resulting extrapolations are shown as dashed curves in Figure 5.

![](images/98cae4dfdb0b8c69f1f39481f364ad4988ed80de1d1d1e2d3e746574e20698b8.jpg)  
Fig. 6: Power-law fits of ∆MRR(k) (y-axis) w.r.t. instance number (x-axis) in log-log scale across models on FB15k-237.

The extrapolations suggest that an unrealistically large number of instances would be required to approach a performance of 1. To further highlight this phenomenon, we use this extrapolation in Table 2 to estimate the oracle performance under an extreme scenario in which the number of instances reaches $k = 1 0 , 0 0 0$ . The extrapolated results indicate that, for most groups, including the cross-model group $G _ { \mathcal { M } } .$ , even such an extreme number of instances would not bring oracle performance close to 1. A few exceptions are observed, notably for MINERVA and NeuralLP on FB15k-237 and for NeuralLP and NBFNet on CoDEx-S. In these cases, the observed gains vary significantly across instances, resulting in a high sensitivity of the extrapolation to the permutations used for fitting the scaling law and consequently in wide prediction intervals. Some extrapolations therefore reach 1, which should not be interpreted as evidence that perfect performance would actually be achieved. ConvE on CoDEx-S constitutes the notable exception, with high performance being extrapolated.

The extrapolation nevertheless only provides an indication of how the observed scaling trend may evolve beyond the number of instances that we actually evaluate. The underlying phenomenon is already directly observable with the 240 instances considered in the largest group, $G _ { \mathcal { M } } \colon$ the oracle fails to resolve a non-negligible subset of queries. Thus, even when exploiting substantial stochastic and architectural diversity, some queries remain unresolved by all of the considered models.

These findings highlight the third gap identified in this paper: the knowledge captured collectively by the considered models remains incomplete. The unresolved queries may reflect limitations in the expressivity of the considered approaches or deficiencies in the benchmarks, which may contain information that is fundamentally unrecoverable by any statistical model.

In Appendix A, we investigate what distinguishes these hard-to-predict queries from the others, particularly those that are commonly easy to solve.

Table 2: Median estimated value of $\mathrm { M R R } _ { \mathcal { O } _ { G } } ^ { 1 0 , 0 0 0 }$ using the marginal-gain scaling over 100 permutations, with the 95% interval in brackets. † marks groups for which some permutations extrapolate above 1.
<table><tr><td>Model</td><td>FB15k-237</td><td>WN18RR</td><td></td><td>CoDEx-S</td></tr><tr><td>TransE</td><td>0.70 [0.67, 0.72]</td><td>0.31 [0.30, 0.33]</td><td></td><td>0.51 [0.49, 0.58]</td></tr><tr><td>DistMult</td><td>0.69 [0.67, 0.72]</td><td>0.60 [0.58, 0.65]</td><td></td><td>0.77 [0.71, 0.90]</td></tr><tr><td>ConvE</td><td>0.68 [0.66, 0.72]</td><td>0.72 [0.68, 0.77]</td><td></td><td>0.92† [0.85, 0.99]</td></tr><tr><td>RotatE</td><td>0.56 [0.54, 0.59]</td><td>0.63 [0.60, 0.66]</td><td></td><td>0.70 [0.64, 0.79]</td></tr><tr><td>ComplEx</td><td>0.66 [0.65, 0.69]</td><td>0.77 [0.71, 0.87]</td><td></td><td>0.79 [0.76, 0.84]</td></tr><tr><td>RESCAL</td><td>0.77 [0.72, 0.85]</td><td>0.83 [0.77, 0.94]</td><td></td><td>0.71 [0.62, 0.95]</td></tr><tr><td>BoxE</td><td>0.51 [0.49, 0.53]</td><td>0.61 [0.59, 0.67]</td><td></td><td>0.57 [0.54, 0.73]</td></tr><tr><td>MINERVA</td><td>0.76† [0.62, 1.10]</td><td>0.54 [0.52, 0.63]</td><td></td><td>0.71 [0.59, 0.91]</td></tr><tr><td>NeuralLP</td><td>0.90† [0.62, 1.73]</td><td>0.56 [0.55, 0.58]</td><td></td><td>0.84† [0.57, 2.66]</td></tr><tr><td>NBFNet</td><td>0.82 [0.78, 0.86]</td><td>0.75 [0.73, 0.79]</td><td></td><td>0.78† [0.66, 1.49]</td></tr><tr><td>AnyBURL</td><td>0.45 [0.42, 0.51]</td><td>0.50 [0.49, 0.51]</td><td></td><td>0.31 [0.31, 0.32]</td></tr><tr><td>M</td><td>0.84 [0.82, 0.85]</td><td>0.81 [0.79, 0.84]</td><td></td><td>0.87 [0.83, 0.91]</td></tr></table>

## 6 Discussion

Our proposed oracle-framework provides upper bounds that could be reached by ensemble of instances. While we do not evaluate existing ensemble methods on our instances, the improvements reported in prior work [10,35] appear smaller than those suggested by oracles. This observation indicates that additional gains may still be attainable through more efective selection or combination strategies. In this view, oracles provide useful reference points for future ensemble learning benchmarks, as they correspond to an ideal routing mechanism. In principle, combining predictions could even surpass this bound, as certain aggregation mechanisms (e.g., voting or score weighting and aggregation between instances) may recover correct rankings that are not produced by any individual instance for a given query.

In our experimental setting, instance diversity is induced through diferent random seeds or model architectures, while keeping hyperparameter configurations fixed. Traditionally, hyperparameter tuning is designed to optimize the performance of individual instances, whereas our analysis focuses on the collective performance of groups. Hyperparameter selection could thus be alternatively designed to promote diversity at the group level: either by identifying a single configuration that promotes diversity across instances, or by intentionally varying configurations to generate complementary behaviors. Exploring these alternatives could provide an additional source of diversity for improving ensemble performance.

Regarding our observation that all models tend to fail on the same subset of queries, one may wonder whether this is a consequence of training all instances on the same data distribution. In such a setting, models may learn a shared core signal, with diversity emerging only at the margins. Alternative strategies could thus be envisioned to further enhance ensemble efectiveness, such as encouraging specialization by training instances on diferent regions of the knowledge graph or introducing coordination mechanisms that promote complementary behaviors. Alternatively, this could also constitute a potential limitation of current benchmarks, which may include queries that are inherently unanswerable given the available training signal. In such cases, achieving perfect performance would be impossible, regardless of the modeling approach.

Finally, approaching oracle performance requires identifying a priori which instance is most likely to produce the best prediction for a given query. The next step therefore lies in identifying performance and complementarity signals to guide instance selection or weighting in ensemble approaches.

## 7 Conclusion

In this paper, we propose an oracle framework as an idealized setting that provides an upper-bound on ensemble learning performance for link prediction. This framework reveals three performance gaps. First, ensembling multiple instances of the same model, trained with diferent random seeds, yields substantial performance gains, showcasing that isolated instances are not representative of the attainable performance of a model. Second, aggregating instances across diferent model architectures matches or outperforms the best single-model oracle, illustrating that no approach provides alone the diversity of combining various architectures, regardless of their individual performance. Third, even with increasingly large ensembles, performance consistently converges below perfect accuracy, indicating limitations in knowledge capture by current approaches or in learnable knowledge in current benchmarks. These gaps underscore the value of ensemble learning for link prediction in knowledge graphs, while opening several directions for future work: designing single-model or cross-model aggregation strategies to attain oracle performance, increasing diversity in model instances to boost performance, and identifying unlearnable queries in benchmark datasets.

## References

1. Abboud, R., Ceylan, I., Lukasiewicz, T., Salvatori, T.: Boxe: A box embedding model for knowledge base completion. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 9649–9661. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 6dbbe6abe5f14af882ff977fc3f35501-Paper.pdf

2. Ali, M., Berrendorf, M., Hoyt, C.T., Vermue, L., Sharifzadeh, S., Tresp, V., Lehmann, J.: PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings. Journal of Machine Learning Research 22(82), 1–6 (2021), http://jmlr.org/papers/v22/20-825.html

3. Betz, P., Galarraga, L., Ott, S., Meilicke, C., Suchanek, F.M., Stuckenschmidt, H.: Pyclause-simple and eficient rule handling for knowledge graphs. In: IJCAI, demo track. Ijcai.org (2024)

4. Bordes, A., Usunier, N., Garc´ıa-Dur´an, A., Weston, J., Yakhnenko, O.: Translating embeddings for modeling multi-relational data. In: Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States. pp. 2787–2795 (2013), https://proceedings.neurips.cc/ paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html

5. Broscheit, S., Rufinelli, D., Kochsiek, A., Betz, P., Gemulla, R.: LibKGE - A knowledge graph embedding library for reproducible research. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. pp. 165–174 (2020), https://www.aclweb.org/anthology/ 2020.emnlp-demos.22

6. Chen, S., Liu, X., Gao, J., Jiao, J., Zhang, R., Ji, Y.: HittER: Hierarchical transformers for knowledge graph embeddings. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 10395–10407. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic (Nov 2021). https://doi.org/10.18653/v1/2021.emnlp-main.812, https://aclanthology.org/2021.emnlp-main.812/

7. Das, R., Dhuliawala, S., Zaheer, M., Vilnis, L., Durugkar, I., Krishnamurthy, A., Smola, A., McCallum, A.: Go for a walk and arrive at the answer: Reasoning over paths in knowledge bases using reinforcement learning. In: ICLR (2018)

8. Dettmers, T., Minervini, P., Stenetorp, P., Riedel, S.: Convolutional 2d knowledge graph embeddings. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018. pp. 1811–1818. AAAI Press (2018). https://doi.org/10.1609/AAAI.V32I1.11573, https://doi.org/10.1609/aaai.v32i1.11573

9. Gal´arraga, L.A., Teflioudi, C., Hose, K., Suchanek, F.M.: AMIE: association rule mining under incomplete evidence in ontological knowledge bases. In: Schwabe, D., Almeida, V.A.F., Glaser, H., Baeza-Yates, R., Moon, S.B. (eds.) 22nd International World Wide Web Conference, WWW ’13, Rio de Janeiro, Brazil, May 13-17, 2013. pp. 413–422. International World Wide Web Conferences Steering Committee / ACM (2013). https://doi.org/10.1145/2488388.2488425, https: //doi.org/10.1145/2488388.2488425

10. Gregucci, C., Nayyeri, M., Hern´andez, D., Staab, S.: Link prediction with attention applied on multiple knowledge graph embedding models. In: Ding, Y., Tang, J., Sequeda, J.F., Aroyo, L., Castillo, C., Houben, G. (eds.) Proceedings of the ACM Web Conference 2023, WWW 2023, Austin, TX, USA, 30 April 2023 - 4 May 2023. pp. 2600–2610. ACM (2023). https://doi.org/10.1145/3543507.3583358, https://doi.org/10.1145/3543507.3583358

11. Hogan, A., Blomqvist, E., Cochez, M., d’Amato, C., Melo, G.D., Gutierrez, C., Kirrane, S., Gayo, J.E.L., Navigli, R., Neumaier, S., et al.: Knowledge graphs. ACM Computing Surveys (Csur) 54(4), 1–37 (2021)

12. Hogan, A., Harth, A., Umbrich, J., Kinsella, S., Polleres, A., Decker, S.: Searching and browsing linked data with swse: The semantic web search engine. Journal of Web Semantics 9(4), 365–401 (2011). https://doi.org/https: //doi.org/10.1016/j.websem.2011.06.004, https://www.sciencedirect.com/ science/article/pii/S1570826811000473, jWS special issue on Semantic Search

13. Krompaß, D., Baier, S., Tresp, V.: Type-constrained representation learning in knowledge graphs. In: The Semantic Web - ISWC 2015 - 14th International Semantic Web Conference, Bethlehem, PA, USA, October 11-15, 2015, Proceedings, Part I. Lecture Notes in Computer Science, vol. 9366, pp. 640–655. Springer (2015). https://doi.org/10.1007/978-3-319-25007-6\_37, https://doi.org/10.1007/ 978-3-319-25007-6\_37

14. Krompass, D., Tresp, V.: Ensemble solutions for link-prediction in knowledge graphs (2015), https://api.semanticscholar.org/CorpusID:15603094

15. Lanthaler, M., Cyganiak, R., Wood, D.: RDF 1.1 concepts and abstract syntax. W3C recommendation, W3C (Feb 2014), https://www.w3.org/TR/2014/RECrdf11-concepts-20140225/

16. Liu, S., Cuenca Grau, B., Horrocks, I., Kostylev, E.V.: Revisiting Inferential Benchmarks for Knowledge Graph Completion. In: Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning. pp. 461–471 (8 2023). https://doi.org/10.24963/kr.2023/45, https://doi.org/10.24963/ kr.2023/45

17. Meilicke, C., Betz, P., Stuckenschmidt, H.: Why a naive way to combine symbolic and latent knowledge base completion works surprisingly well. In: Conference on Automated Knowledge Base Construction (2021), https://api. semanticscholar.org/CorpusID:247296302

18. Meilicke, C., Chekol, M.W., Betz, P., Fink, M., Stuckenschmidt, H.: Anytime bottom-up rule learning for large-scale knowledge graph completion. VLDB Journal 33(1), 131–161 (2024). https://doi.org/10.1007/S00778-023-00800-5, https://doi.org/10.1007/s00778-023-00800-5

19. Meilicke, C., Fink, M., Wang, Y., Rufinelli, D., Gemulla, R., Stuckenschmidt, H.: Fine-grained evaluation of rule- and embedding-based systems for knowledge graph completion. In: The Semantic Web – ISWC 2018: 17th International Semantic Web Conference, Monterey, CA, USA, October 8–12, 2018, Proceedings, Part I. p. 3–20. Springer-Verlag, Berlin, Heidelberg (2018), https://doi.org/10.1007/ 978-3-030-00671-6\_1

20. M´erou´e, G., Gandon, F., Monnin, P.: Link prediction or perdition: The seeds of instability in knowledge graph embeddings. In: Acosta, M., van Erp, M., Rudolph, S., Hartig, O., Spahiu, B., Rula, A., Garijo, D., Osborne, F. (eds.) The Semantic Web. pp. 198–216. Springer Nature Switzerland, Cham (2026)

21. Nickel, M., Tresp, V., Kriegel, H.P.: A three-way model for collective learning on multi-relational data. In: Proceedings of the 28th International Conference on

International Conference on Machine Learning. p. 809–816. ICML’11, Omnipress, Madison, WI, USA (2011)

22. Rufinelli, D., Broscheit, S., Gemulla, R.: You CAN teach an old dog new tricks! on training knowledge graph embeddings. In: 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. Open-Review.net (2020), https://openreview.net/forum?id=BkxSmlBFvr

23. Safavi, T., Koutra, D.: CoDEx: A Comprehensive Knowledge Graph Completion Benchmark. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 8328–8350. Association for Computational Linguistics, Online (Nov 2020). https://doi.org/10.18653/v1/2020. emnlp-main.669, https://www.aclweb.org/anthology/2020.emnlp-main.669

24. Shaban, A., Paulheim, H.: Snape–training snapshot ensembles of link prediction models. In: International Semantic Web Conference. pp. 3–22. Springer (2024)

25. Sun, Z., Deng, Z., Nie, J., Tang, J.: Rotate: Knowledge graph embedding by relational rotation in complex space. In: 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net (2019), https://openreview.net/forum?id=HkgEQnRqYQ

26. Toutanova, K., Chen, D.: Observed versus latent features for knowledge base and text inference. In: Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality, CVSC 2015, Beijing, China, July 26-31, 2015. pp. 57–66. Association for Computational Linguistics (2015). https://doi.org/ 10.18653/V1/W15-4007, https://doi.org/10.18653/v1/W15-4007

27. Trouillon, T., Welbl, J., Riedel, S., Gaussier, E., Bouchard, G.: Complex embed-<sup>´</sup> dings for simple link prediction. In: Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016. JMLR Workshop and Conference Proceedings, vol. 48, pp. 2071–2080. JMLR.org (2016), http://proceedings.mlr.press/v48/trouillon16.html

28. Wang, H., Zhang, F., Wang, J., Zhao, M., Li, W., Xie, X., Guo, M.: Ripple network: Propagating user preferences on the knowledge graph for recommender systems. CoRR abs/1803.03467 (2018), http://arxiv.org/abs/1803.03467

29. Wang, H., Zhang, F., Wang, J., Zhao, M., Li, W., Xie, X., Guo, M.: Ripplenet: Propagating user preferences on the knowledge graph for recommender systems. In: Proceedings of the 27th ACM International Conference on Information and Knowledge Management. p. 417–426. CIKM ’18, Association for Computing Machinery, New York, NY, USA (2018). https://doi.org/10.1145/3269206.3271739, https://doi.org/10.1145/3269206.3271739

30. Xu, C., Nayyeri, M., Vahdati, S., Lehmann, J.: Multiple run ensemble learning with low-dimensional knowledge graph embeddings. In: International Joint Conference on Neural Networks, IJCNN 2021, Shenzhen, China, July 18-22, 2021. pp. 1– 8. IEEE (2021). https://doi.org/10.1109/IJCNN52387.2021.9533372, https: //doi.org/10.1109/IJCNN52387.2021.9533372

31. Yang, B., Yih, W., He, X., Gao, J., Deng, L.: Embedding entities and relations for learning and inference in knowledge bases. In: 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings (2015), http://arxiv.org/abs/1412.6575

32. Yang, F., Yang, Z., Cohen, W.W.: Diferentiable learning of logical rules for knowledge base reasoning. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017), https://proceedings.neurips.cc/paper\_files/paper/2017/file/ 0e55666a4ad822e0e34299df3591d979-Paper.pdf

33. Yasunaga, M., Ren, H., Bosselut, A., Liang, P., Leskovec, J.: QA-GNN: Reasoning with language models and knowledge graphs for question answering. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. pp. 535–546. Association for Computational Linguistics, Online (Jun 2021). https://doi.org/10. 18653/v1/2021.naacl-main.45, https://aclanthology.org/2021.naacl-main. 45/

34. Yasunaga, M., Ren, H., Bosselut, A., Liang, P., Leskovec, J.: QA-GNN: reasoning with language models and knowledge graphs for question answering. CoRR abs/2104.06378 (2021), https://arxiv.org/abs/2104.06378

35. Yue, L., Zhang, Y., Yao, Q., Li, Y., Wu, X., Zhang, Z., Lin, Z., Zheng, Y.: Relationaware ensemble learning for knowledge graph embedding. In: Bouamor, H., Pino, J., Bali, K. (eds.) Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023. pp. 16620–16631. Association for Computational Linguistics (2023). https:// doi.org/10.18653/V1/2023.EMNLP-MAIN.1034, https://doi.org/10.18653/v1/ 2023.emnlp-main.1034

36. Zhu, Y., Potyka, N., Nayyeri, M., Xiong, B., He, Y., Kharlamov, E., Staab, S.: Predictive multiplicity of knowledge graph embeddings in link prediction. In: Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 334–354. Association for Computational Linguistics, Miami, Florida, USA (Nov 2024). https://doi.org/10.18653/v1/2024.findings-emnlp.19, https:// aclanthology.org/2024.findings-emnlp.19/

37. Zhu, Z., Zhang, Z., Xhonneux, L.P., Tang, J.: Neural bellman-ford networks: A general graph neural network framework for link prediction. Advances in Neural Information Processing Systems 34 (2021)

## A Characterizing Dificult Queries

To understand why some queries remain unsolved by all instances, we analyze whether dificult queries exhibit diferent structural properties from queries that are consistently solved.

Query dificulties. We classify queries by how consistently instances in $G _ { \mathcal { M } }$ recover the correct answer at rank 1. A query is Easy if every instance ranks the correct answer first, Hard if none does, and Medium otherwise. To prevent low-performing models from shrinking the global Easy set $( \mathrm { e . g . }$ , TransE on WN18RR), for each dataset, we compute the Easy set of each model separately and exclude those whose Easy ratio is below 15% .

Query features. We characterize each query using two families of features computed from the training graph: graph connectivity and relation structure. For graph connectivity, consider a query $( h , r , ? )$ whose ground-truth triple is $( h , r , t )$ We measure how frequently the target entity t has been observed as an object of relation $r$ in the training graph (target occurrence, per relation), and how frequently the source entity $h$ has been observed as a subject of relation $r$ (source occurrence, per relation). The latter is equivalent to the number of valid answers for the query in the training graph $\mathrm { a s } ,$ for $( h , r , ? )$ , it counts the distinct entities $t ^ { \prime }$ such that $\left( h , r , t ^ { \prime } \right)$ occurs in the training set. For relation structure, we consider simple relation patterns (see, e.g., Liu et al. [16]). In particular, a query $( h , r , ? )$ whose ground-truth triple is $( h , r , t )$ is considered symmetric if the relation satisfies the symmetry pattern $( x , r , y ) \Rightarrow ( y , r , x )$ with confidence greater than 80% in the training graph, and the corresponding mirror triple $( t , r , h )$ is observed in the training set. We also characterize each relation by its cardinality type (N-1, or N-N). Note that we considered additional features, including global entity occurrence, 2- and 3-hop paths, relation composition, 1-1 and 1-N cardinality types. We focus here on the features with the clearest signal in our analysis, but provide the full set of features on $\mathrm { G i t H u b ^ { 3 } }$ . For each feature, we report its value separately for Easy, Medium, and Hard queries, as well as over all queries (Total). Numerical features are summarized by their median, while relation patterns are reported as the percentage of queries exhibiting the corresponding property.

To quantify how strongly a feature distinguishes Easy from Hard queries, we compute the area under the receiver operating characteristic curve (AUC). For a feature $f ,$ the AUC can be interpreted <sup>5</sup> as the probability that a randomly selected Easy query receives a higher feature value than a randomly selected Hard query, with ties counted as half:

$$
\operatorname { A U C } ( f ) = \operatorname* { P r } \left[ f ( q _ { E } ) > f ( q _ { H } ) \right] + \frac { 1 } { 2 } \operatorname* { P r } \left[ f ( q _ { E } ) = f ( q _ { H } ) \right] ,
$$

where $q _ { E }$ and $q _ { H }$ are independently sampled Easy and Hard queries. An AUC of 1 therefore means that the feature perfectly separates the two groups, with larger values systematically associated with Easy queries. Conversely, an AUC of 0 corresponds to perfect separation in the opposite direction, with larger values associated with Hard queries. An AUC of 0.5 indicates that the feature provides no discrimination between the two groups. The AUC furthest from 0.5 for each dataset is shown in bold in Table 3.

Table 3: Characteristics of Easy, Medium, and Hard queries.
<table><tr><td>Dataset</td><td>Statistic</td><td>Easy</td><td>Medium</td><td>Hard</td><td>Total</td><td>AUC</td></tr><tr><td rowspan="6">FB15k-237</td><td>% of queries</td><td>10%</td><td>49%</td><td>41%</td><td>100%</td><td></td></tr><tr><td>Target occ. (rel.)</td><td>966</td><td>22</td><td>5</td><td>12</td><td>0.952</td></tr><tr><td>Source occ. (rel.)</td><td>1</td><td>11</td><td>27</td><td>12</td><td>0.135</td></tr><tr><td>Symmetry</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.500</td></tr><tr><td>N-1</td><td>57.5%</td><td>11.9%</td><td>6.2%</td><td>14.1%</td><td>0.757</td></tr><tr><td>N-N</td><td>41.2%</td><td>76.0%</td><td>71.8%</td><td>70.8%</td><td>0.347</td></tr><tr><td rowspan="6">WN18RR</td><td>% of queries</td><td>21%</td><td>47%</td><td>32%</td><td>100%</td><td></td></tr><tr><td>Target occ. (rel.)</td><td>2</td><td>2</td><td>1</td><td>2</td><td>0.589</td></tr><tr><td>Source occ. (rel.)</td><td>2</td><td>2</td><td>1</td><td>2</td><td>0.501</td></tr><tr><td>Symmetry</td><td>97.8%</td><td>33.4%</td><td>0.0%</td><td>36.0%</td><td>0.989</td></tr><tr><td>N-1</td><td>1.2%</td><td>33.1%</td><td>43.7%</td><td>30.0%</td><td>0.287</td></tr><tr><td>N-N</td><td>97.0%</td><td>36.3%</td><td>4.6%</td><td>38.6%</td><td>0.962</td></tr><tr><td rowspan="6">CoDEx-S</td><td>% of queries</td><td>6%</td><td>56%</td><td>37%</td><td>100%</td><td></td></tr><tr><td>Target occ. (rel.)</td><td>516</td><td>38</td><td>8</td><td>22</td><td>0.973</td></tr><tr><td>Source occ. (rel.)</td><td>3</td><td>18</td><td>53</td><td>22</td><td>0.138</td></tr><tr><td>Symmetry</td><td>0.0%</td><td>24.5%</td><td>0.3%</td><td>13.9%</td><td>0.499</td></tr><tr><td>N-1</td><td>42.8%</td><td>12.6%</td><td>3.3%</td><td>11.1%</td><td>0.697</td></tr><tr><td>N-N</td><td>57.2%</td><td>79.7%</td><td>77.6%</td><td>77.5%</td><td>0.398</td></tr></table>

Analysis. The factors associated with query dificulty difer substantially across datasets. On FB15k-237 and CoDEx-S, dificulty is primarily driven by entity connectivity. Target occurrence is the most discriminative feature, with median values dropping from 966 to 5 on FB15k-237 (AUC 0.95) and from 516 to 8 on CoDEx-S (AUC 0.97) between Easy and Hard queries. Thus, queries are easier when the ground-truth target has frequently been observed as an object of the query relation in the training graph. Conversely, the number of valid answers increases from 1 to 27 and from 3 to 53, respectively (AUC 0.14 in both cases), indicating that queries with more possible answers are harder to solve. Relation patterns are substantially less discriminative on these datasets, with maximum AUCs of 0.76 and 0.70, respectively.

WN18RR exhibits the opposite pattern: connectivity features are weak (maximum AUC 0.62), while relation structure is highly discriminative. Symmetry reaches an AUC of 0.99, with 97.8% of Easy queries having a training-set witness for the mirror triple (t, r, h), compared with none of the Hard queries. N-N cardinality also reaches an AUC of 0.96.

Overall, query dificulty depends strongly on the structure of the underlying KG, with entity connectivity dominating on FB15k-237 and CoDEx-S, and relation structure dominating on WN18RR.

## B UpSet plot

The UpSet plots in Fig. 7 provide a complementary perspective to Fig. 3, ofering a more fine-grained analysis of model behavior. Specifically, they depict the proportion of queries successfully answered (i.e., Hit@1 = 1 or Hit@10 = 1) for the 20 most represented combinations of models. In other words, each bar corresponds to the set of queries that are jointly solved by a given subset of models and by no others, thereby revealing patterns of overlap, exclusivity, and complementarity among models. Results for CoDEx-S are available in our GitHub repository under oracles outputs/histogram.

For instance, considering Hit@1 on FB15k-237, we observe that 38% of the queries are not correctly answered by any model, while 10% are successfully solved by all models. Furthermore, 7% of the queries are uniquely solved by NBFNet, highlighting its distinctive contribution. Interestingly, adding any of RESCAL, MINERVA, NeuralLP, or even TransE to the other models increases the overall number of solved queries by more than 1%, despite their overall lower performance. This suggests that even weaker models can capture complementary signals that are not exploited by stronger ones.

## C Results for Hit@1

## GenAI Usage Disclosure

During the preparation of this paper, ChatGPT was used as a writing assistant to improve language. In addition, Windsurf Editor, powered by Claude Sonnet 4.6, was used as a coding assistant. All scientific contributions, analyses, and experimental results presented in this paper are the authors’ own.

Table 4: Averaged and Oracle Hit@1 across datasets and models
<table><tr><td>Model</td><td> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { T r a n s E } } }$ </td><td>WN18RR FB15k-237</td><td></td><td>codex-s</td></tr><tr><td rowspan="8">KGE Neua Syolie Croosmdel</td><td> $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { T r a n s E } } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { D i s t M u l t } } }$   $\underline { { \mathrm { M R R } } } \mathcal { O } _ { \mathrm { D i s t M u l t } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { C o n v E } } }$ </td><td> $. 0 6 \pm . 0 0$  .07 .44 ± .00 .48 .44 ± .00</td><td> $. 2 2 \pm . 0 0$  .35 .25 ± .00 .36 .25 ± .00</td><td> $. 2 2 \pm . 0 0$  .27 .30 ± .01 .46 .34 ± .01</td></tr><tr><td> $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { C o n v E } } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { R o t a t E } } }$   $\underline { { \mathrm { M R R } } } _ { \mathcal { O } _ { \mathrm { R o t a t E } } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { C o m p l E x } } }$   $\underline { { \mathrm { M R R } } } \mathcal { O } _ { \mathrm { C o m p l e x } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { R E S C A L } } }$   $\underline { { \mathrm { M R R } } } \mathcal { O } _ { \mathrm { R E S C A L } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { B o x E } } }$ </td><td>.49 .47 ± .00 .51 .47 ±.00 .55 .47 ±.00 .55</td><td>.38 .24 ± .00 .33 .25 ± .00 .36 .26 ± .00 .43</td><td>.55 .30 ± .00 .43 .37 ± .00 .51 .29 ± .00 .44</td></tr><tr><td> $\mathrm { M R R } _ { { \mathcal { O } } _ { \mathrm { B o x E } } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { M I N E R V A } } }$   $\underline { { \mathrm { M R R } } } \mathcal { O } _ { \mathrm { M I N E R V A } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { N e u r a l L P } } }$   $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { N e u r a l L P } } }$ </td><td>.43 ± .00 .50 .39 ± .01 .45 .31 ± .05 .50</td><td>.25 ± .00 .32 .11 ± .00 .25 .16 ± .01</td><td>.31 ± .01 .39 .19 ± .02 .35 .19 ± .01</td></tr><tr><td> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { N B F N e t } } }$   $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { N B F N e t } } }$   $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { A n y B U R L } } }$   $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { A n y B U R L } } }$ </td><td>.53 ± .00 .63 .43 ± .00 .44</td><td>.30 .32 ± .00 .50 .18 ± .00 .24</td><td>.35 .26 ± .01 .42 .20 ± .00 .20</td></tr><tr><td> $\overline { { \mathrm { M R R } } } _ { G _ { \mathrm { A M I E } } }$   $\mathrm { M R R } _ { \mathcal { O } _ { \mathrm { A M I E } } }$   $\mathrm { M R R } _ { \mathcal { O } _ { K G E } } ^ { 2 1 }$ </td><td>.39 ± .00 .39</td><td>.14 ±.00 .14</td><td>.15 ± .00 .15</td></tr><tr><td> $\mathrm { M R R } _ { \mathcal { O } _ { N e u r a l } } ^ { 2 0 }$   $\mathrm { M R R } _ { \mathcal { O } _ { S y m b o l i q u e } } ^ { 2 0 }$ </td><td> $. 5 7 \pm . 0 0$   $. 6 2 \pm . 0 0$ </td><td> $. 4 2 \pm . 0 0$   $. 4 9 \pm . 0 0$ </td><td> $. 5 5 \pm . 0 0$   $. 5 8 \pm . 0 0$ </td></tr><tr><td> $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 4 }$   $\mathrm { M R R } _ { \mathcal { O } _ { \mathcal { M } } } ^ { 2 0 , \mathrm { g r e e d y } }$ </td><td> $. 4 4 \pm . 0 0$   $. 6 2 \pm . 0 0$   $. 6 4 \pm . 0 0$ </td><td> $. 2 4 \pm . 0 0$   $. 5 0 \pm . 0 0$ </td><td> $. 2 1 \pm . 0 0$   $. 5 8 \pm . 0 0$ </td></tr></table>

![](images/541eea76cc21a79da02fd100b44f714cef9f9b5fd7bb3045f2041d490df834af.jpg)

![](images/c2c8ada91956df0daa13107c556fab41f629083a70f1501a15bebf56372e181b.jpg)  
(a) FB15k-237 (Hit@1)  
(b) FB15k-237 (Hit@10)

![](images/59845ea0c749530b16456fd11963fb62813fc647b8fa4b9c662f22b55b315dff.jpg)  
(c) WN18RR (Hit@1)

![](images/ecc6d1f2bedf15893451a664f21721fe75226ee1322870e2e1c331bc5d553b11.jpg)  
(d) WN18RR (Hit@10)

Fig. 7: UpSet plots showing the distribution of correctly answered queries across the 20 most frequent combinations of models for FB15k-237 and WN18RR. Each bar represents the proportion of queries solved by a specific subset of models, enabling the analysis of overlap and model complementarity at diferent evaluation thresholds (Hit@1 and Hit@10).