# One Pretrained Decision Model Is Enough for Training-Free Generalist Graph Anomaly Detection

Xinyu Zhao   
School of Computer Science and   
Engineering, Beihang University   
Beijing, China   
xyzhao@buaa.edu.cn   
Tianchen Zhu   
School of Reliability and Systems   
Engineering, Beihang University   
Beijing, China   
catezi@buaa.edu.cn   
Qingyun Sun<sup>∗</sup>   
School of Computer Science and   
Engineering, Beihang University   
Beijing, China   
sunqy@buaa.edu.cn   
Daqing Li   
School of Reliability and Systems   
Engineering, Beihang University   
Beijing, China   
daqingl@buaa.edu.cn   
Ziwei Zhang   
School of Computer Science and   
Engineering, Beihang University   
Beijing, China   
zwzhang@buaa.edu.cn   
Jianxin Li   
School of Computer Science and   
Engineering, Beihang University   
Beijing, China   
lijx@buaa.edu.cn

## Abstract

Graph anomaly detection (GAD) aims to identify irregular entities in graph-structured data, yet heterogeneous anomaly patterns and distributional diferences across graphs challenge generalizable detection. Existing training-free approaches avoid task-specific op timization but primarily rely on predefined numerical scoring rules, leaving the semantic interpretation of heterogeneous anomaly evidence largely unexplored. In this paper, we propose Jev-GAD, a training-free generalist graph anomaly detection framework that bridges heterogeneous numerical evidence and pretrained decision mechanisms. To address heterogeneous anomaly patterns, we construct complementary structural, attribute, and neighborhood evidence and transform heterogeneous statistics into graph-relative anomaly percentiles. To move beyond predefined numerical scoring rules, we leverage a single frozen pretrained language model for evidence-conditioned anomaly assessment. To bridge the semantic gap between numerical evidence and pretrained models, we design a semantic Graph-to-State (G2S) transformation that organizes graph context and semantically labeled node evidence into structured decision states for non-generative option-logit readout. The resulting decision scores are integrated with four complementary statistical scores through equal-weight rank-based fusion, without GAD-specific gradient-based optimization. Extensive experiments on seven graph anomaly detection benchmarks spanning citation, social, and co-review networks demonstrate the efectiveness of Jev-GAD, achieving average AUROC and AUPRC scores of 89.18% and 52.40%, respectively. Further analyses validate the contributions of pretrained decision readout and semantic evidence grounding.

## CCS Concepts

• Information systems → Data mining; • Computing methodologies → Machine learning.

## Keywords

Graph Anomaly Detection, Training-Free Detection, Pretrained Decision Models, Heterogeneous Evidence Fusion

## 1 Introduction

Graph Anomaly Detection (GAD) plays a crucial role in maintaining the trustworthiness of Web platforms by identifying suspicious entities and irregular interactions in online social networks, e-commerce systems, and recommendation services [12, 30, 43]. In these environments, anomalous behaviors may manifest as unusual structural positions, atypical node attributes, or inconsistencies with neighboring entities [23, 28]. For instance, fraudulent users may disguise their attributes or relationships to evade detection [4, 49], while abnormal relational patterns can exhibit varying degrees of neighborhood heterophily [7]. Such heterogeneous anomaly characteristics, together with substantial distributional diferences across Web graphs, make it challenging to develop detectors that generalize across domains. Traditional GAD methods typically rely on reconstruction or contrastive learning objectives [3, 5, 22], while recent generalist approaches explore transferring anomaly detection capabilities to unseen graphs [20, 25, 26, 44, 45]. Despite their efectiveness, many existing methods still require source-domain training or specialized adaptation. Training-free approaches [46, 50] ofer an appealing alternative by exploiting intrinsic graph characteristics without GAD-specific parameter training, particularly when target-domain anomaly labels are unavailable and graph distributions vary across domains.

However, eliminating task-specific training does not necessarily eliminate the limitations of predefined anomaly scoring mechanisms. As illustrated in Fig. 1, heterogeneous graph anomalies give rise to distinct forms of structural, attribute, and neighborhood evidence, each reflecting diferent characteristics of anomalous behavior. Existing training-free methods commonly transform such evidence into anomaly scores through predefined numerical criteria [46, 50]. Although these criteria can efectively capture particular irregularities, they do not explicitly condition their assessments on the semantic meanings of diferent evidence types. For instance, structural rarity and neighborhood inconsistency may both indicate anomalous behavior, yet their implications can vary with the underlying graph context. Consequently, a fixed numerical scoring mechanism may struggle to accommodate heterogeneous, contextdependent anomaly patterns. This limitation raises a central question: how can training-free GAD leverage pretrained decision models for evidence-conditioned anomaly scoring?

![](images/b624cc77ddc136819734e0eac9fa1ee2f85c3bc307a78f1e531c01f6dbba9240.jpg)  
Figure 1: Motivation ofJev-GAD: heterogeneous anomalies, fixed scoring rules, and missing semantic grounding.

Meanwhile, pretrained language models have shown promising capabilities in evaluation, classification, and structured decisionmaking beyond conventional text generation [2, 10, 31, 48]. These advances suggest the potential of leveraging a single frozen pretrained model for graph anomaly assessment. However, directly applying such models introduces a fundamental semantic gap: graph anomaly evidence primarily consists of numerical statistics whose interpretations depend on structural, attribute, and relational contexts. Even after graph-relative normalization, anonymous numer ical values cannot distinguish structural rarity from attribute deviation or neighborhood inconsistency, making it dificult for pretrained models to interpret the anomaly characteristics represented by diferent evidence dimensions. This observation motivates our key insight: by associating graph-relative numerical evidence with explicit anomaly semantics, frozen pretrained models can perform evidence-conditioned anomaly assessment without graph-specific parameter learning. Such semantic grounding provides a bridge between heterogeneous numerical graph evidence and pretrained decision mechanisms.

Building upon this insight, we propose Jev-GAD, a training-free generalist graph anomaly detection framework that bridges heterogeneous graph evidence and pretrained decision mechanisms. To accommodate diverse anomaly patterns and graph distributions, Jev-GAD constructs complementary structural, attribute, and neigh borhood evidence and transforms heterogeneous statistics into graph-relative anomaly percentiles. To move beyond predefined numerical scoring rules, we leverage a single frozen pretrained language model for evidence-conditioned anomaly assessment. To bridge the semantic gap between numerical evidence and pretrained models, we introduce a semantic Graph-to-State (G2S) transformation that deterministically organizes graph context and semantically labeled node evidence into structured decision states. The frozen model subsequently evaluates anomalous and normal alternatives through non-generative option-logit readout, producing continuous anomaly scores without task-specific parameter updates. In parallel, four complementary statistical scorers assess numerical irregularities, and their scores are integrated with pretrained decision scores through equal-weight rank-based fusion. Extensive experiments on seven benchmarks spanning citation, social, and co-review networks demonstrate the efectiveness ofJev-GAD, achieving average AUROC and AUPRC scores of 89.18% and 52.40%, respectively, surpassing the strongest baseline macro-averages by 7.09 and 15.44 percentage points. Further analyses validate the contributions of pretrained decision readout, semantic evidence grounding, and complementary statistical scoring. Our contributions are summarized as follows:

• We proposeJev-GAD, a training-free generalist graph anomaly detection framework that leverages a single frozen pretrained model for evidence-conditioned anomaly scoring beyond predefined numerical rules.

• We introduce Graph-to-State (G2S), which transforms heterogeneous numerical evidence into semantically grounded decision states, enabling non-generative pretrained decision readout and its integration with complementary statistical scoring.

• Extensive experiments on seven graph anomaly detection benchmarks demonstrate the efectiveness of Jev-GAD, while comprehensive analyses validate the contributions of pretrained decisions, semantic grounding, and complementary statistical evidence.

## 2 Related Work

## 2.1 Graph Anomaly Detection

Traditional GAD methods identify anomalous nodes through residual modeling, reconstruction, or contrastive learning, as exemplified by Radar, DOMINANT, AnomalyDAE, CoLA, ANEMONE, CONAD, and GAD-NR [3, 5, 9, 14, 22, 32, 41]. Cluster-aware methods further exploit community-level patterns for graph anomaly detection [47]. Together, these approaches capture diferent levels of anomaly evidence, ranging from node attributes and neighborhood relations to higher-order community structure. Large-scale benchmarking eforts further reveal the diverse characteristics and performance of unsupervised graph anomaly detectors across attributed networks [19]. Such diversity motivates combining complementary structural, attribute, and neighborhood evidence. Recent generalist methods, including ARC, UNPrompt, AnomalyGFM, IA-GGAD, and GAD-MoRE, further explore transferable anomaly detection across diverse graph domains [20, 25, 26, 44, 45]. However, many of these approaches still require task-specific training or adaptation. To reduce this dependency, training-free methods such as TFGAD and FreeGAD leverage intrinsic graph statistics and predefined scoring mechanisms to identify anomalies [46, 50]. Despite their efectiveness, these methods primarily rely on statistical scoring criteria, which may struggle to capture context-dependent anomaly patterns, particularly when diferent evidence types interact.

## 2.2 Pretrained Decision Models

Recent advances in large language models (LLMs) have enabled automated evaluation and decision-making beyond text generation [10, 48]. Unlike free-form generative judging, non-generative de cision readout directly obtains option-level scores conditioned on predefined questions and contextual states. Recent studies have systematically evaluated Jev’s zero-shot decision capabilities across diverse classification and reasoning tasks [2, 29]. Further investigations demonstrate its potential for eficient preference evaluation, scientific decision-making, and semantic query processing [1, 16, 40]. Meanwhile, Open-Jev explores single-pass decision readout through option logits, while recent research examines the stability of such non-generative prediction mechanisms [15, 31]. Option-level scores also allow pretrained decisions to be integrated directly with numerical anomaly scorers. Despite these advances, applying pretrained decision mechanisms to heterogeneous graph evidence remains relatively underexplored.

## 3 Problem Formulation

We consider node-level graph anomaly detection on an attributed graph $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \mathbf { X } _ { 0 } )$ , where V and E denote the node and edge sets, respectively, and $\mathbf { X } _ { 0 } \in \mathbb { R } ^ { N \times D _ { 0 } }$ is the node feature matrix. Let $\mathbf { A } \in \{ 0 , \bar { 1 } \} ^ { N \times N }$ denote the adjacency matrix, where $N = | \mathcal { V } |$ . The objective is to construct an anomaly scoring function $f : \mathcal { V } $ R that assigns higher scores to anomalous nodes. We focus on the training-free generalist setting, where the detector is directly applied to unseen target graphs $\mathcal { D } _ { \mathrm { t e s t } } = \{ \mathcal { G } ^ { ( t ) } \} _ { t = 1 } ^ { T }$ without access to their anomaly labels $\mathbf { y } ^ { ( t ) } \in \{ 0 , 1 \} ^ { N _ { t } }$ or GAD-specific gradient-based training. The pretrained decision model �<sub>�</sub> remains frozen, and $\mathbf { y } ^ { ( t ) }$ is used solely for evaluation.

## 4 Methodology

As illustrated in Figure 2, Jev-GAD constructs graph-relative anomaly evidence from structure, attributes, and neighborhoods, organizes selected evidence into semantically grounded states for pretrained decision inference, and integrates the resulting decision scores with complementary statistical assessments through rank-based fusion.

## 4.1 Heterogeneous Graph Evidence Construction

Graph anomalies manifest as structural irregularities, attribute deviations, and neighborhood inconsistencies. We construct complementary graph evidence to characterize these heterogeneous anomaly patterns.

Complementary Graph Evidence. Given an attributed graph $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \mathbf { X } _ { 0 } )$ , we consider three categories of anomaly evidence.

Structural evidence characterizes how a node difers from the typical topological organization of the graph, including its connec tivity, local clustering, core membership, and centrality. Attribute evidence measures deviations from the global feature distribution, while neighborhood evidence captures inconsistencies between a node and its local relational context.

To complement the original attributes, we additionally construct an aligned feature view using the training-free multi-curvature alignment procedure of [45]. Attribute and neighborhood evidence are extracted from both the original and aligned views, allowing diferent representations to expose potentially distinct anomaly characteristics. The detailed evidence configuration is provided in Appendix A.1.

Graph-Relative Evidence. A central dificulty in combining heterogeneous evidence is that diferent statistics have incompatible scales and interpretations. For instance, a large feature distance may indicate an anomaly, while structural irregularities can appear at either extreme of a statistic’s distribution. Moreover, the same numerical value may have diferent implications across graphs with diferent structural and attribute distributions.

We address this issue by converting each statistic into a graphrelative measure of atypicality. Let $q _ { j } ( \boldsymbol { v } )$ denote the �-th statistic of node �, and let $\delta _ { j } ( \boldsymbol { v } )$ be its anomaly-oriented deviation. For statistics with potentially anomalous upper and lower tails, $\delta _ { j } ( \boldsymbol { v } )$ measures the deviation from the graph-wide median; for distance-based statistics, larger deviations are retained directly. The normalized evidence is defined as

$$
e _ { j } ( v ) = { \mathcal { R } } _ { G } { \big ( } \delta _ { j } ( v ) { \big ) } ,\tag{1}
$$

where ${ \mathcal { R } } _ { G }$ computes the empirical percentile rank among all nodes in the target graph.

This transformation provides a common numerical interpretation: evidence values lie in [0, 1], with larger values indicating greater atypicality relative to the graph population. The resulting 16-dimensional numerical evidence representation is

$$
\mathbf { e } _ { v } = \left[ { \bf e } _ { v } ^ { \mathrm { s t r } } ; \mathbf { e } _ { v } ^ { \mathrm { r a w } } ; \mathbf { e } _ { v } ^ { \mathrm { a l i g n } } \right] ,\tag{2}
$$

where the components comprise four structural indicators and six attribute and neighborhood indicators from each feature view. This representation captures complementary anomaly patterns and provides the numerical basis for statistical scoring.

## 4.2 Semantic Graph-to-State Representation

Although graph-relative normalization makes heterogeneous evidence numerically comparable, numerical comparability alone does not make the evidence interpretable to a pretrained model. For example, a high percentile may describe either an unusual structural position or a strong disagreement with neighboring nodes. These observations can have diferent implications for anomaly assessment, yet their meanings cannot be recovered from anonymous numerical values alone.

This motivates a semantic interface that preserves the correspondence between each numerical observation and the graph characteristic it describes. We introduce a semantic Graph-to-State (G2S) transformation that organizes graph evidence into structured decision states with explicit semantic correspondence.

Semantic Evidence Grounding. Let $\boldsymbol { \mathcal { I } } = \left( j _ { 1 } , \ldots , j _ { 7 } \right)$ denote the ordered indices of the seven indicators selected for decision readout. For each selected indicator $j _ { k }$ , we associate its normalized value $e _ { j _ { k } } ( \boldsymbol { \upsilon } )$ with a semantic descriptor $\psi _ { j _ { k } }$ specifying the corresponding anomaly characteristic.

The resulting evidence state is represented as

$$
\begin{array} { r } { \mathcal { B } _ { v } = \left[ ( \psi _ { j _ { k } } , e _ { j _ { k } } ( v ) ) \right] _ { k = 1 } ^ { 7 } . } \end{array}\tag{3}
$$

Unlike an anonymous feature vector, $\mathcal { B } _ { v }$ retains both the magnitude and interpretation of each observation. Its seven entries follow a fixed order across neighborhood, structural, and attribute groups, allowing the decision model to distinguish diferent sources of irregularity.

![](images/c71ca8a8be3cdd437e216919348563e3f9f92e469f10b0b5fa4d095f6f09592e.jpg)  
Figure 2: Overview of Jev-GAD. Heterogeneous anomaly evidence is organized into semantically grounded decision states for a single frozen pretrained model. Complementary statistical scoring and rank-based fusion produce the final anomaly ranking without GAD-specific gradient-based optimization.

For example, a node may exhibit ordinary structural characteristics but strong neighborhood inconsistency. Preserving these distinctions enables the pretrained model to evaluate the supplied evidence with explicit knowledge of what each numerical value represents, rather than relying on feature positions alone.

Graph-to-State Construction. Node-level evidence is further accompanied by auxiliary graph-level context $\mathbf { c } _ { G } ,$ which summarizes properties of the observed graph, including its size, connectivity, and feature characteristics. We define the resulting state as

$$
S _ { v } = \Phi _ { \mathrm { G 2 S } } ( G , v ) = \left( \mathbf { c } _ { G } , \mathcal { B } _ { v } \right) ,\tag{4}
$$

where $\Phi _ { \mathrm { G 2 S } }$ denotes the Graph-to-State transformation.

The state is serialized into a structured input with a fixed structure across nodes and graphs. Its node-level entries are grouped by anomaly characteristic and retain their descriptive semantics. The graph-level context provides auxiliary information about the observed graph.

Two properties are particularly important. First, semantic descriptors and numerical evidence remain aligned across graphs. Second, graph-relative normalization ensures that the values share a consistent direction and represent relative atypicality rather than graph-specific raw magnitudes.

These properties enable a frozen pretrained model to access heterogeneous graph evidence without graph-specific representation learning. Semantic correspondence is examined in Sec. 5.4.

## 4.3 Pretrained Decision Readout

Given the semantically grounded state $\textstyle S _ { v } ,$ , Jev-GAD employs a frozen pretrained model to make evidence-conditioned anomaly decisions through non-generative option-logit readout.

Evidence-Conditioned Decision. We formulate anomaly assessment as a binary choice between anomalous and normal behavior. Each decision query combines a fixed decision criterion, two candidate options, and the node-specific state $S _ { v }$

The criterion specifies how the evidence should be interpreted: larger normalized values represent stronger graph-relative deviations, while the accompanying semantic descriptions identify the nature of those deviations. The decision criterion and candidate options remain unchanged across nodes and graphs, whereas the evidence state varies with the node being evaluated.

This formulation allows the pretrained model to receive multiple anomaly characteristics as explicit decision evidence, rather than requiring an additional graph-specific classifier to be trained from labeled examples.

Direct Option Readout. Instead of generating a textual judgment, we adopt a non-generative decision readout inspired by singlepass decision mechanisms [31]. Specifically, a frozen pretrained language model processes the decision query, and we extract the output logits corresponding to the two fixed, single-token options.

Let $\ell _ { v } ^ { a }$ and $\ell _ { v } ^ { n }$ denote the logits associated with anomalous and normal decisions, respectively. The decision score is

$$
s _ { \mathrm { d e c } } ( v ) = \sigma \left( \ell _ { v } ^ { a } - \ell _ { v } ^ { n } \right) ,\tag{5}
$$

where $\sigma ( \cdot )$ is the sigmoid function.

A larger score indicates a stronger preference for the anomalous option and provides a continuous signal for anomaly ranking.

A single forward pass produces the decision score without autoregressive generation or parameter updates.

## 4.4 Statistical Scoring and Fusion

In parallel with pretrained decision readout, Jev-GAD employs complementary statistical scorers to capture diferent forms of numerical irregularity in graph evidence.

Statistical Evidence Assessment. We apply four statistical scoring mechanisms to the heterogeneous evidence representation.

A fixed aggregation rule captures the overall magnitude of anomaly evidence, while an ECOD-style upper-tail criterion [17] emphasizes extreme deviations in individual evidence dimensions. Isolation Forest [18] evaluates multivariate isolation tendencies, and an

Table 1: AUROC comparison on seven graph anomaly detection benchmarks (%). Results are reported as mean ± standard deviation. Best and second-best mean values are highlighted in bold and underline, respectively.
<table><tr><td>Method (Venue)</td><td>ACM</td><td>Amazon</td><td>BlogCatalog</td><td>CiteSeer</td><td>Cora</td><td>Facebook</td><td>Weibo</td><td> $\mathbf { A v g . }$ </td></tr><tr><td>Supervised Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GCN (ICLR&#x27;17)</td><td> $4 7 . 6 9 { \scriptstyle \pm 1 . 4 8 }$ </td><td> $4 1 . 1 0 { \scriptstyle \pm 4 . 2 4 }$ </td><td> $4 3 . 8 3 { \scriptstyle \pm 2 . 3 9 }$ </td><td> $5 3 . 8 3 { \scriptstyle \pm 4 . 0 1 }$ </td><td> $5 1 . 9 9 { \scriptstyle \pm 3 . 2 0 } $ </td><td> $4 6 . 0 9 { \scriptstyle \pm 1 8 . 8 8 }$ </td><td> $3 7 . 6 4 \pm 1 3 . 8 4$ </td><td>46.02</td></tr><tr><td>GAT (ICLR&#x27;18)</td><td> $4 6 . 5 4 { \scriptstyle \pm 4 . 7 1 }$ </td><td> $5 5 . 3 5 { \scriptstyle \pm 5 . 6 9 }$ </td><td> $4 9 . 3 2 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $4 4 . 5 8 { \scriptstyle \pm 8 . 2 5 }$ </td><td> $4 4 . 6 5 { \scriptstyle \pm 6 . 2 0 }$ </td><td> $5 3 . 9 6 { \scriptstyle \pm 2 6 . 1 4 }$ </td><td> $3 5 . 5 0 { \scriptstyle \pm 6 . 9 1 }$ </td><td>47.13</td></tr><tr><td>BWGNN (ICML&#x27;22)</td><td> $6 7 . 4 3 { \scriptstyle \pm 2 . 4 8 }$ </td><td> $5 0 . 0 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $6 9 . 9 1 { \scriptstyle \pm 4 . 6 6 }$ </td><td> $4 2 . 5 8 { \scriptstyle \pm 2 . 8 6 }$ </td><td> $3 9 . 7 5 { \scriptstyle \pm 3 . 1 6 }$ </td><td> $6 4 . 8 4 { \scriptstyle \pm 4 . 1 7 }$ </td><td> $5 0 . 0 8 { \scriptstyle \pm 0 . 0 3 }$ </td><td>54.94</td></tr><tr><td>GHRN (WWW&#x27;23)</td><td> $5 2 . 7 2 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $6 3 . 5 2 { \scriptstyle \pm 1 . 9 2 }$ </td><td> $6 2 . 2 8 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $5 4 . 1 1 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $4 7 . 6 5 { \scriptstyle \pm 2 . 0 3 }$ </td><td> $5 4 . 0 4 { \scriptstyle \pm 3 . 8 1 }$ </td><td> $1 7 . 7 9 { \scriptstyle \pm 5 . 0 0 }$ </td><td>50.30</td></tr><tr><td>Unsupervised Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AnomalyDAE (ICASSP&#x27;20)</td><td> $7 7 . 6 0 { \scriptstyle \pm 1 . 0 3 }$ </td><td> $5 8 . 3 8 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $5 7 . 5 3 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $8 2 . 0 0 { \scriptstyle \pm 2 . 7 8 }$ </td><td> $8 4 . 0 1 { \scriptstyle \pm 1 . 1 9 }$ </td><td> $1 0 . 8 7 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $1 9 . 7 7 { \scriptstyle \pm 1 . 1 4 }$ </td><td>55.74</td></tr><tr><td>CoLA (TNNLS&#x27;21)</td><td> $4 9 . 3 0 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $4 7 . 5 7 { \pm } 3 . 0 2 $ </td><td> $4 9 . 5 0 { \scriptstyle \pm 1 . 2 7 } $ </td><td> $5 2 . 5 6 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $4 9 . 3 5 { \scriptstyle \pm 1 . 5 2 }$ </td><td> $5 0 . 7 0 { \scriptstyle \pm 5 . 4 9 }$ </td><td> $4 5 . 2 5 { \scriptstyle \pm 2 . 3 9 }$ </td><td>49.17</td></tr><tr><td>TAM (NeurIPS&#x27;23)</td><td> $5 0 . 3 9 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $5 2 . 3 7 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $3 3 . 0 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $4 6 . 4 4 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $4 8 . 7 8 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $5 4 . 5 6 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $2 1 . 5 0 { \scriptstyle \pm 0 . 3 0 }$ </td><td>43.87</td></tr><tr><td>Generalist Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ARC (10-shot) (NeurIPS&#x27;24)</td><td> $7 8 . 3 3 { \scriptstyle \pm 1 . 1 3 }$ </td><td> $5 7 . 9 5 { \scriptstyle \pm 6 . 1 3 }$ </td><td> $7 2 . 5 9 { \pm } 1 . 1 0 $ </td><td> $8 2 . 3 1 { \scriptstyle \pm 2 . 8 9 }$ </td><td> $8 2 . 2 2 { \scriptstyle \pm 2 . 3 4 }$ </td><td> $6 4 . 9 4 { \scriptstyle \pm 3 . 7 2 }$ </td><td>88.65±0.82</td><td>75.28</td></tr><tr><td>AnomalyGFM (KDD&#x27;25)</td><td> $3 8 . 9 1 { \scriptstyle \pm 1 . 2 1 }$ </td><td> $6 1 . 2 4 { \scriptstyle \pm 5 . 2 1 }$ </td><td> $4 0 . 9 3 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $4 0 . 3 7 { \scriptstyle \pm 1 . 3 7 }$ </td><td> $4 0 . 9 3 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $7 4 . 4 5 { \scriptstyle \pm 2 . 3 6 }$ </td><td> $4 4 . 4 3 { \scriptstyle \pm 4 . 8 9 }$ </td><td>48.75</td></tr><tr><td>UNPrompt (IJCAI&#x27;25)</td><td> $7 3 . 2 0 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $7 2 . 1 4 { \scriptstyle \pm 2 . 9 4 }$ </td><td> $7 0 . 8 5 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $5 9 . 5 3 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 3 . 1 8 { \scriptstyle \pm 1 . 2 9 } $ </td><td> $6 0 . 6 4 { \scriptstyle \pm 1 . 9 4 }$ </td><td> $6 6 . 2 9 { \scriptstyle \pm 1 . 3 8 }$ </td><td>66.55</td></tr><tr><td>IA-GGAD (NeurIPS&#x27;25)</td><td> $7 8 . 4 4 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $5 3 . 3 8 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $7 3 . 0 3 { \scriptstyle \pm 1 . 2 0 }$ </td><td> $9 1 . 5 4 { \scriptstyle \pm 0 . 3 2 } $ </td><td> $8 5 . 6 9 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $6 9 . 5 9 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $8 7 . 3 5 { \scriptstyle \pm 0 . 4 0 }$ </td><td>77.00</td></tr><tr><td>ARCzero (TKDE&#x27;26)</td><td> $7 9 . 0 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $6 2 . 5 9 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $7 0 . 4 6 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $9 0 . 0 1 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $\underline { { 8 7 . 5 5 } } { \pm } 0 . 2 6$ </td><td> $6 7 . 6 3 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $8 8 . 5 7 { \scriptstyle \pm 0 . 1 5 }$ </td><td>77.98</td></tr><tr><td>GAD-MoRE (ICDM&#x27;26)</td><td> $\underline { { 8 1 . 1 7 } } { \pm } 0 . 0 5$ </td><td> ${ \underline { { 7 6 . 9 0 { \pm } 1 . 8 0 } } }$ </td><td> $7 3 . 0 9 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $9 0 . 2 8 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $8 6 . 3 9 { \scriptstyle \pm 0 . 2 0 }$ </td><td> ${ \underline { { 7 5 . 7 5 } } } { \pm 0 . 9 5 }$ </td><td> $9 1 . 0 3 { \pm } 0 . 1 1$ </td><td>82.09</td></tr><tr><td>Training-free Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TFGAD (WWW&#x27;25)</td><td> $7 6 . 0 4 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $7 0 . 5 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 4 . \mathbf { 0 } 7 { \pm } 0 . 0 5$ </td><td> $7 4 . 7 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $8 0 . 4 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $3 6 . 4 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $7 7 . 5 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td>69.98</td></tr><tr><td>FreeGAD (CIKM&#x27;25)</td><td> $3 7 . 4 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 9 . 4 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 5 . 4 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $7 1 . 8 5 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 3 . 6 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $6 0 . 4 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 . 8 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td>46.02</td></tr><tr><td>Jev-GAD (ours)</td><td> $\mathbf { 9 5 . 2 5 { \scriptstyle \pm 0 . 1 0 } }$ </td><td> $\mathbf { 8 6 . 9 0 } { \scriptstyle \pm 0 . 0 6 }$ </td><td> $7 3 . 3 6 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $\mathbf { 9 8 . 4 1 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $9 6 . 4 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $\mathbf { 8 0 . 2 1 { \scriptstyle \pm 0 . 1 1 } }$ </td><td> $9 3 . 6 6 { \scriptstyle \pm 0 . 0 6 }$ </td><td>89.18</td></tr></table>

RBF-based One-Class SVM [34] characterizes deviations from the estimated distribution boundary.

These scoring mechanisms operate on the same evidence representation but emphasize diferent forms of numerical irregularity. The statistical estimators are fitted using unlabeled target-graph evidence, without anomaly-specific supervision.

Rank-Based Score Fusion. Since the decision and statistical scores arise from diferent mechanisms, their raw values are not directly comparable. We therefore normalize each score by its empirical percentile rank within the target graph before aggregation.

Let M denote the set of statistical scoring mechanisms. The final anomaly score is

$$
s ( v ) = \frac { 1 } { \vert \mathcal { M } \vert + 1 } \left[ \mathcal { R } _ { G } \big ( s _ { \mathrm { d e c } } ( v ) \big ) + \sum _ { m \in \mathcal { M } } \mathcal { R } _ { G } \big ( s _ { m } ( v ) \big ) \right] ,
$$

## 5 Experiments

We evaluate Jev- ${ \cdot } \mathrm { G A D } ^ { 1 }$ on diverse attributed graphs and investigate the following research questions:

• RQ1: How efective is Jev-GAD across graph domains and anomaly patterns? (▷ Section 5.2)

• RQ2: Do pretrained decision readout and statistical scoring provide complementary signals? (▷ Section 5.3)

(6)

• RQ3: Does semantic grounding improve the interpretation of numerical graph evidence? (▷ Section 5.4)

• RQ4: How do evidence views and statistical backends afect performance? (▷ Section 5.5)

where $s _ { m } ( \boldsymbol { v } )$ is the anomaly score produced by statistical mechanism �.

With four statistical scorers, the decision score and each statistical score contribute equally with weight $1 / 5 .$ . This fusion combines pretrained decisions with complementary numerical assessments without label-dependent weight estimation.

• RQ5: How robust is pretrained decision readout to prompt variations and to its fusion weight? (▷ Section 5.6)

• RQ6: What are the inference-time and memory costs of Jev-GAD? (▷ Section 5.7)

## 5.1 Experimental Settings

Datasets and Metrics. We evaluate Jev-GAD on seven attributed graph anomaly detection benchmarks, following established evaluation protocols [19, 20, 45]: three citation graphs (ACM [37], CiteSeer and Cora [35]), three social networks (BlogCatalog [38], Facebook and Weibo [20]), and one co-review network (Amazon [12, 24]). These benchmarks span diverse structures, feature distributions, and anomaly patterns. We report AUROC [6], AUPRC [33], and their seven-dataset macro-averages.

Table 2: AUPRC comparison on seven graph anomaly detection benchmarks (%). Results are reported as mean ± standard deviation. Best and second-best mean values are highlighted in bold and underline, respectively.
<table><tr><td>Method (Venue)</td><td>ACM</td><td>Amazon</td><td>BlogCatalog</td><td>CiteSeer</td><td>Cora</td><td>Facebook</td><td>Weibo</td><td> $\mathbf { A v g . }$ </td></tr><tr><td>Supervised Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GCN (ICLR&#x27;17)</td><td> $3 . 6 8 { \pm } 0 . 1 6$ </td><td> $5 . 5 4 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $5 . 7 1 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 . 2 1 { \scriptstyle \pm 0 . 8 9 }$ </td><td> $5 . 8 5 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $4 . 5 5 { \scriptstyle \pm 3 . 0 6 }$ </td><td> $2 5 . 9 7 { \scriptstyle \pm 1 5 . 8 4 }$ </td><td>8.07</td></tr><tr><td>GAT (ICLR&#x27;18)</td><td> $3 . 3 8 { \pm } 0 . 4 5 $ </td><td> $8 . 1 0 { \scriptstyle \pm 1 . 2 4 }$ </td><td> $6 . 6 4 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $4 . 3 5 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $4 . 9 3 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $6 . 1 8 { \scriptstyle \pm 5 . 8 3 }$ </td><td> $1 5 . 8 6 { \scriptstyle \pm 5 . 8 3 }$ </td><td>7.06</td></tr><tr><td>BWGNN (ICML&#x27;22)</td><td> $1 9 . 4 8 { \scriptstyle \pm 2 . 2 0 }$ </td><td> $6 . 7 9 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $2 8 . 3 7 { \scriptstyle \pm 4 . 3 1 }$ </td><td> $4 . 0 2 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $5 . 3 0 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $5 . 7 1 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $1 0 . 4 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td>11.45</td></tr><tr><td>GHRN (WWW&#x27;23)</td><td> $9 . 2 1 { \pm } 2 . 9 4$ </td><td> $1 1 . 1 8 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $2 4 . 5 7 { \scriptstyle \pm 3 . 6 0 }$ </td><td> $5 . 3 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $5 . 2 3 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $3 . 6 1 { \pm } 0 . 6 8$ </td><td> $8 . 3 5 { \scriptstyle \pm 2 . 4 4 }$ </td><td>9.65</td></tr><tr><td>Unsupervised Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AnomalyDAE (ICASSP&#x27;20)</td><td> $1 1 . 6 7 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $9 . 2 6 { \pm } 1 . 5 1 $ </td><td> $7 . 1 7 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $3 1 . 3 7 { \scriptstyle \pm 1 . 0 9 }$ </td><td> $3 2 . 6 5 { \scriptstyle \pm 1 . 4 6 }$ </td><td> $1 . 3 2 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $6 . 1 0 { \scriptstyle \pm 0 . 0 8 }$ </td><td>14.22</td></tr><tr><td>CoLA (TNNLS&#x27;21)</td><td> $3 . 5 5 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $6 . 5 5 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $6 . 1 1 { \scriptstyle \pm 0 . 3 6 }$ </td><td> $4 . 7 6 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 . 8 6 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $3 . 4 1 { \pm } 1 . 3 8 $ </td><td> $9 . 6 7 { \scriptstyle \pm 0 . 5 0 }$ </td><td>5.70</td></tr><tr><td>TAM (NeurIPS&#x27;23)</td><td> $3 . 7 4 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $7 . 9 0 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $5 . 6 6 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $4 . 1 5 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $5 . 5 8 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $5 . 3 7 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $6 . 2 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td>5.51</td></tr><tr><td>Generalist Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ARC (10-shot) (NeurIPS&#x27;24)</td><td> $3 6 . 5 3 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $7 . 9 8 { \pm } 1 . 1 8$ </td><td> $\underline { { 3 7 . 7 7 } } { \pm } 0 . 3 5$ </td><td> $3 0 . 9 3 { \scriptstyle \pm 4 . 2 4 }$ </td><td> $3 9 . 2 3 { \scriptstyle \pm 4 . 4 0 }$ </td><td> $4 . 9 2 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $5 9 . 4 7 { \scriptstyle \pm 2 . 3 8 }$ </td><td>30.98</td></tr><tr><td>AnomalyGFM (KDD&#x27;25)</td><td> $3 . 1 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $9 . 1 8 { \pm } 1 . 6 9$ </td><td> $4 . 8 5 { \scriptstyle \pm 0 . 4 0 }$ </td><td> $3 . 6 9 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $4 . 3 6 { \pm } 0 . 0 9$ </td><td> $6 . 0 0 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $9 . 0 5 { \scriptstyle \pm 1 . 0 3 }$ </td><td>5.75</td></tr><tr><td>UNPrompt (IJCAI&#x27;25)</td><td> $2 4 . 0 2 { \scriptstyle \pm 1 . 3 8 }$ </td><td> $1 2 . 2 8 { \scriptstyle \pm 1 . 6 3 }$ </td><td> $3 5 . 2 9 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $9 . 4 7 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $1 0 . 2 6 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $3 . 4 4 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $2 8 . 4 1 { \scriptstyle \pm 2 . 2 5 }$ </td><td>17.59</td></tr><tr><td>IA-GGAD (NeurIPS&#x27;25)</td><td> $3 6 . 3 3 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $6 . 9 0 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $3 6 . 6 4 { \scriptstyle \pm 0 . 6 6 }$ </td><td> $4 6 . 9 6 { \pm } 1 . 1 6$ </td><td> $4 5 . 0 3 { \scriptstyle \pm 2 . 0 6 }$ </td><td> $8 . 6 5 { \scriptstyle \pm 1 . 7 4 }$ </td><td> $6 0 . 1 6 { \scriptstyle \pm 0 . 9 3 }$ </td><td>34.38</td></tr><tr><td>ARCzero (TKDE&#x27;26)</td><td> $\underline { { 4 0 . 6 2 } } { \pm 0 . 1 2 }$ </td><td> $8 . 5 3 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $3 8 . 0 3 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $4 5 . 0 3 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $\underline { { 4 8 . 2 5 { \pm } 0 . 5 1 } }$ </td><td> $\underline { { 1 0 . 7 4 } } { \pm 1 . 6 0 }$ </td><td> $\underline { { 6 1 . 8 1 } } \pm 0 . 4 1$ </td><td>36.14</td></tr><tr><td>GAD-MoRE (ICDM&#x27;26)</td><td> $3 7 . 1 7 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $\underline { { 3 1 . 5 5 { \pm } 4 . 8 4 } }$ </td><td> $3 5 . 3 1 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $4 0 . 1 5 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $4 5 . 2 5 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $9 . 0 0 { \pm } 1 . 1 2 $ </td><td> $6 0 . 2 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td>36.96</td></tr><tr><td>Training-free Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TFGAD (WWW&#x27;25)</td><td> $2 4 . 5 1 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $1 9 . 4 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 2 . 5 8 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $2 4 . 3 4 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 0 . 6 7 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $1 . 7 6 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $6 5 . 3 2 { \scriptstyle \pm 0 . 0 0 }$ </td><td>28.37</td></tr><tr><td>FreeGAD (CIKM&#x27;25)</td><td> $3 . 0 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 . 1 4 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $3 . 8 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 9 . 3 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 6 . 3 9 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $2 . 9 8 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $5 . 5 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>9.46</td></tr><tr><td>Jev-GAD (ours)</td><td> ${ \bf 6 1 . 6 8 { \scriptstyle \pm 0 . 2 9 } }$ </td><td> $\mathbf { 4 1 . 2 6 { \scriptstyle \pm 0 . 3 0 } }$ </td><td> $3 3 . 5 0 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $\mathbf { 8 8 . 0 2 } { \scriptstyle \pm 0 . 1 9 }$ </td><td> $7 6 . 1 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $\mathbf { 1 1 . 0 8 } { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 5 . 1 5 { \scriptstyle \pm 0 . 3 4 }$ </td><td>52.40</td></tr></table>

aggregation, ECOD-style upper-tail scoring [17], Isolation Forest (IF) [18], and RBF-based One-Class SVM (OCSVM) [34]. IF uses 1,000 trees, and its stochastic results are averaged over five random seeds. All five scores undergo graph-relative rank normalization before equal-weight fusion.

Baselines. We compare against 15 methods in four categories: Supervised GAD (GCN [11], GAT [39], BWGNN [36], GHRN [7]); Unsupervised GAD (AnomalyDAE [5], CoLA [22], TAM [27]); Generalist GAD (ARC (10-shot) [20], UNPrompt [25], AnomalyGFM [26], IA-GGAD [44], ARCzero [21], GAD-MoRE [45]); and Training-free GAD (TFGAD [50], FreeGAD [46]). Source-trained comparisons use disjoint source and target graphs following established crossgraph protocols [45], with source datasets detailed in Appendix B.1. Previously reported and reproduced results follow their respective protocols. In particular, ARC (10-shot) uses target labels and serves as a label-assisted reference rather than a strictly label-free baseline. Evaluation Protocol and Implementation. Jev-GAD requires neither target anomaly labels nor GAD-specific gradient-based optimization at inference time. Anomaly labels are used only for metric computation, while statistical estimators operate on unlabeled target-graph evidence. We use Qwen3-4B [42] as the frozen backbone for non-generative decision readout, extracting scores from two fixed candidate decision tokens without autoregressive generation or parameter updates.

## 5.2 RQ1: Overall Efectiveness

Tables 1 and 2 compare Jev-GAD against the baselines in terms of AUROC and AUPRC. The results reveal the following findings. (1) Strong overall performance. Jev-GAD achieves average AU-ROC and AUPRC scores of 89.18% and 52.40%, outperforming the strongest baseline macro-averages by 7.09 and 15.44 percentage points, respectively. It obtains the highest AUROC on six of seven graphs and the highest AUPRC on five, demonstrating consistently strong performance across diverse graph benchmarks.

(2) Pronounced gains on selected graph domains. On CiteSeer and Cora, Jev-GAD obtains AUPRC scores of 88.02% and 76.10%, respectively. These improvements demonstrate the efectiveness of integrating heterogeneous anomaly evidence with pretrained decision readout to capture diverse anomaly characteristics.

The evidence representation combines structural, attribute, and neighborhood statistics from the raw and aligned feature views, with the aligned dimension set to 32. The decision state retains structural rarity, aligned-view attribute deviation, and aligned-view neighborhood inconsistency. The statistical branch consists of fixed

## 5.3 RQ2: Component Ablation

We compare three configurations: (1) Decision-only, using the pretrained score; (2) Statistical-only, combining four numerical scorers; and (3) Jev-GAD, fusing both pathways. Figure 3(a) reports their AUROC and AUPRC on four diagnostic datasets.

![](images/2d198d055937f8fc59d0463df7b90e3b4467a09d8aa02c99538a3477b657a558.jpg)  
Correct Semantic State Anonymous Fields Mismatched Semantics

![](images/a190b8dafa9ead404dfb192e24c666c53375783cfa689a35abd9acb4f7a770d3.jpg)

![](images/84732fc6354890aa90f291b770835440724b5956733e10e984debdc80a335c72.jpg)

![](images/02d07b797e0176b5a5362959fd10d58f10a17288174f2b1c93a1e9950f36977a.jpg)

![](images/70a0188d6df704524515cbd678fbe2e924de8caec7e2ba4a1523e9e044c36f17.jpg)  
(c) Semantic grounding: Full Jev-GAD  
Original Four Scorers IF to kNN IF to LOF OCSVM to Mahalanobis

![](images/901316a0d92b8e5a4771938427bb75f54e0e40ccbee1861de570f8f93f3de1d9.jpg)  
(d) Multi-view evidence

![](images/51f7ad74300b20061e548705f0fcc30eb1ad0098e01f54be628117e2cdf12d05.jpg)  
(e) Statistical backends (I)

![](images/961d061e1de240f03caaf123a86809d082ecba22815c246cc604595e243b5d09.jpg)  
(f) Statistical backends (II)  
Figure 3: Diagnostic analyses on four graphs. (a) Component ablation. (b, c) Semantic grounding under Decision-only and full Jev-GAD scoring; Mismatched Semantics averages three fixed cyclic shifts. (d) Raw, aligned, and combined evidence views. (e, f) One-for-one replacement of statistical scorers. Y-axes do not all start at zero; numbers above bars are exact scores (%).

(1) Decision readout and numerical scoring are complementary. Jev-GAD outperforms both pathways on all four datasets. On CiteSeer, Decision-only, Statistical-only, and Jev-GAD achieve AUPRC scores of 74.04%, 71.06%, and 88.02%; on Cora, the scores are 51.89%, 67.34%, and 76.10%. These gains confirm complementary anomaly-ranking information from both pathways.

(2) The value of pretrained decisions varies by dataset. Compared with Statistical-only, Jev-GAD improves AUPRC by 6.59, 0.91, 16.96, and 8.76 percentage points on Amazon, BlogCatalog, CiteSeer, and Cora, respectively. These varying gains reflect the dataset-dependent contribution of pretrained decisions across het erogeneous anomaly patterns.

(3) Pretrained decisions provide unique value beyond ad ditional numerical scorers. To distinguish semantic decisionmaking from the efect of introducing an additional scoring signal, we retain the four statistical scorers and compare the pretrained decision score against three numerical alternatives derived from the same seven-dimensional evidence: Mean-7D, Max-7D, and LOF-7D. Each five-score configuration uses equal-weight rank fusion. Across the four diagnostic graphs, Jev-GAD achieves a macro-AUPRC of 59.72%, outperforming Statistical-only (51.41%), Mean-7D (49.98%), Max-7D (50.04%), and LOF-7D (42.14%). The corresponding macro-AUROC reaches 88.78%, compared with 88.14% for Statistical-only. These results demonstrate that pretrained semantic decisions contribute complementary anomaly-ranking information beyond that provided by additional numerical scoring mechanisms. Detailed comparisons are reported in Table 6.

## 5.4 RQ3: Semantic Evidence Grounding

We test whether pretrained decisions depend on the correspondence between numerical evidence and its semantic description. Two controlled variants are considered: (1) Anonymous Fields, which replace evidence descriptors with anonymous identifiers while preserving all values; and (2) Mismatched Semantics, which cyclically reassign values to incorrect descriptors. The latter is averaged over three predefined permutations. Figure 3(b) and (c) compare the pretrained decision pathway and the final fused scores, respectively. (1) Semantic correspondence matters for decision readout. On CiteSeer, Decision-only AUPRC declines from 74.04% with the original descriptions to 19.79% with Anonymous Fields and 41.90% with Mismatched Semantics. Similar declines across other datasets highlight the importance of semantic descriptions and their correct alignment with numerical evidence.

![](images/22fad2a65223b89359998fd5dadc27eeff155c67cb5519db29a070fe4289ccfc.jpg)  
Figure 4: Prompt robustness. The first row of each block gives the original score (%); the other rows give changes relative to it in percentage points.

![](images/eb8a7c320f7597d4cf00a2caf923647198a63d11912a5e30c9ed033cc1c47451.jpg)

![](images/1c45bfc85270cf071610dc30c0b819f1bc9848629d1efec0549643480369ff3e.jpg)  
(b) AUPRC sensitivity  
Figure 5: Sensitivity to decision weight � on four graphs. The dashed line marks equal-weight fusion (� = 0.2), while shaded bands indicate ±1 standard deviation across five Isolation Forest seeds. The endpoints represent Statistical-only (� = 0) and Decision-only (� = 1).

(2) The benefit survives statistical fusion. On Amazon, the com plete Jev-GAD model reaches 41.26% AUPRC with the original state, compared with 25.76% and 27.95% under Anonymous Fields and Mismatched Semantics. The observed trends across the four diagnostic graphs support the role of explicit semantic grounding beyond its efect on the standalone decision branch. Because the numerical evidence and four statistical scores remain unchanged under both perturbations, the performance degradation isolates the contribution of semantic grounding to the fused ranking.

![](images/db7cf488f493375d0f60df42f4408f704c44c4ac141bff3b187a907c099c8646.jpg)  
Figure 6: Computational eficiency. Numbers above bars are measured total times (s).

## 5.5 RQ4: Evidence and Scoring Analysis

We assess two design choices that afect how heterogeneous numerical evidence is used: feature-view composition and the statistical scoring backend.

(1) Raw and aligned views contain complementary evidence. Figure 3(d) compares Raw View, Aligned View, and Both Views while keeping pretrained decision readout unchanged. Combining both views produces the highest AUPRC on Amazon, BlogCatalog, and CiteSeer. On Amazon, the three configurations obtain 22.70%, 37.23%, and 41.26% AUPRC, respectively. Combining both views achieves the highest AUPRC on three of the four datasets, demonstrating the efectiveness of complementary anomaly evidence.

(2) Performance depends on the statistical backend. We compare the original ensemble with three alternative configurations: replacing IF with kNN or LOF, and replacing OCSVM with Mahalanobis distance. As shown in Figure 3(e) and (f), the original backend yields higher AUROC and AUPRC than these variants on all four diagnostic datasets. For example, its AUPRC on CiteSeer is 88.02%, compared with 51.01%, 54.01%, and 71.39% for the tested alternatives. These results demonstrate the efectiveness of the proposed statistical scoring configuration.

## 5.6 RQ5: Decision Robustness

We examine prompt sensitivity by swapping option order or using two alternative decision instructions while keeping the evidence states fixed.

(1) Robustness to prompt variations. Figure 4 shows that Jev-GAD achieves 88.02% AUPRC on CiteSeer with the original prompt and 86.81%, 88.20%, and 85.01% under three variants. On Cora, AUPRC ranges from 75.55% to 77.04%. These results demonstrate stable performance across the tested prompt formulations and option orderings.

(2) Sensitivity to decision weight. We vary the decision weight $\lambda \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 5 , 0 . 7 , 1 \}$ , distributing the remaining weight equally among four statistical scorers. Here, � = 0 and $\lambda \ : = \ : 1$ correspond to Statistical-only and Decision-only, while $\lambda = 0 . 2$ recovers equal-weight fusion. Figure 5 shows peak AUPRC at � = 0.2 on Amazon and BlogCatalog, 0.3 on Cora, and 0.5 on CiteSeer. Increasing � from 0.2 to 0.3 changes macro-AUPRC by only 0.41 points (59.72% to 60.13%), whereas the endpoints lose 8.31 and 18.90 points relative to the default. The default also achieves the highest macro-AUROC.

## 5.7 RQ6: Computational Eficiency

We measure graph evidence processing and frozen-model decision inference separately, as summarized in Figure 6.

(1) Runtime includes preprocessing and model inference. On Amazon (10,224 nodes), Jev-GAD takes approximately 116 seconds, including 31 seconds of CPU processing and 85 seconds of pretrained decision inference. On Cora (2,708 nodes), it takes approximately 31 seconds.

(2) Pretrained decisions introduce a measurable cost. Frozen Qwen3-4B readout processes approximately 120 nodes per second with about 9.6 GiB of peak allocated GPU memory. This is the primary trade-of relative to numerical training-free scoring, which requires no language-model inference.

## 6 Conclusion

We presented Jev-GAD, a training-free generalist graph anomaly detection framework connecting heterogeneous anomaly evidence with a single frozen pretrained model. The semantic Graph-to-State (G2S) transformation organizes graph-relative evidence into semantically grounded inputs for non-generative decision readout, while rank-based fusion integrates pretrained decisions with complementary statistical scores. Across seven benchmarks, Jev-GAD achieves average AUROC and AUPRC of 89.18% and 52.40%, respectively. Component ablations and robustness analyses validate the efectiveness of pretrained decision readout and semantic grounding for training-free graph anomaly detection.

## References

[1] Boyuan Deng, Shuyi Fan, Hongyang Zhang, and Xinhong Xie. 2026. Jev for scientific decisions: Evaluating semantic choices and their consequences. arXiv preprint arXiv:2609.24965 (2026).

[2] Tobias Deußer, Lorenz Sparrenberg, and Rafet Sifa. 2026. Evaluating and Bench marking the System One Model Jev. arXiv preprint arXiv:2609.37647 (2026).

[3] Kaize Ding, Jundong Li, Rohit Bhanushali, and Huan Liu. 2019. Deep anomaly detection on attributed networks. In Proceedings ofthe 2019 SIAM international conference on data mining. SIAM, 594–602.

[4] Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, and Philip S Yu. 2020. Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In Proceedings of the 29th ACM international conference on information & knowledge management. 315–324.

[5] Haoyi Fan, Fengbin Zhang, and Zuoyong Li. 2020. Anomalydae: Dual autoencoder for anomaly detection on attributed networks. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 5685–5689.

[6] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern recognition letters 27, 8 (2006), 861–874.

[7] Yuan Gao, Xiang Wang, Xiangnan He, Zhenguang Liu, Huamin Feng, and Yong dong Zhang. 2023. Addressing heterophily in graph anomaly detection: A

perspective of graph spectrum. In Proceedings ofthe ACM web conference 2023. 1528–1538.

[8] Xiaofei He, Deng Cai, and Partha Niyogi. 2005. Laplacian score for feature selection. Advances in neural information processing systems 18 (2005).

[9] Ming Jin, Yixin Liu, Yu Zheng, Lianhua Chi, Yuan-Fang Li, and Shirui Pan. 2021. Anemone: Graph anomaly detection with multi-scale contrastive learning. In Proceedings of the 30th ACM international conference on information & knowledge management. 3122–3126.

[10] Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. 2024. Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models. In International Conference on Learning Representations.

[11] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016).

[12] Srijan Kumar, Bryan Hooi, Disha Makhija, Mohit Kumar, Christos Faloutsos, and VS Subrahmanian. 2018. Rev2: Fraudulent user prediction in rating platforms. In Proceedings ofthe eleventh ACM international conference on web search and data mining. 333–341.

[13] Srijan Kumar, Xikun Zhang, and Jure Leskovec. 2019. Predicting dynamic em bedding trajectory in temporal interaction networks. In Proceedings ofthe 25th ACM SIGKDD international conference on knowledge discovery & data mining. 1269–1278.

[14] Jundong Li, Harsh Dani, Xia Hu, and Huan Liu. 2017. Radar: Residual analysis for anomaly detection in attributed networks.. In Ijcai, Vol. 17. 2152–2158.

[15] Ran Li and Lei Chen. 2026. Readout Stability in Prefill-Only Decision Models: Zero-Label Prediction and Inference-Time Compute Allocation. arXiv preprint arXiv:2610.07716 (2026).

[16] Yubo Li, Yidi Miao, Ramayya Krishnan, and Rema Padman. 2026. JEV-as-a-judge: Accept when confident, escalate when unsure. arXiv preprint arXiv:2609.26550 (2026).

[17] Zheng Li, Yue Zhao, Xiyang Hu, Nicola Botta, Cezar Ionescu, and George H Chen. 2022. Ecod: Unsupervised outlier detection using empirical cumulative distribution functions. IEEE Transactions on Knowledge and Data Engineering 35, 12 (2022), 12181–12193.

[18] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation forest. In 2008 eighth ieee international conference on data mining. Ieee, 413–422.

[19] Kay Liu, Yingtong Dou, Yue Zhao, Xueying Ding, Xiyang Hu, Ruitong Zhang, Kaize Ding, Canyu Chen, Hao Peng, Kai Shu, et al. 2022. Bond: Benchmarking unsupervised outlier node detection on static attributed graphs. Advances in neural information processing systems 35 (2022), 27021–27035.

[20] Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, and Shirui Pan. 2024. Arc: A generalist graph anomaly detector with in-context learning. Advances in Neural Information Processing Systems 37 (2024), 50772–50804.

[21] Yixin Liu, Shiyuan Li, Yu Zheng, Qingfeng Chen, Chengqi Zhang, Philip S Yu, and Shirui Pan. 2026. From few-shot to zero-shot: Towards generalist graph anomaly detection. IEEE Transactions on Knowledge and Data Engineering (2026).

[22] Yixin Liu, Zhao Li, Shirui Pan, Chen Gong, Chuan Zhou, and George Karypis. 2021. Anomaly detection on attributed networks via contrastive self-supervised learning. IEEE transactions on neural networks and learning systems 33, 6 (2021), 2378–2392.

[23] Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z Sheng, Hui Xiong, and Leman Akoglu. 2021. A comprehensive survey on graph anomaly detection with deep learning. IEEE transactions on knowledge and data engineering 35, 12 (2021), 12012–12038.

[24] Julian John McAuley and Jure Leskovec. 2013. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. In Proceedings ofthe 22nd international conference on World Wide Web. 897–908.

[25] Chaoxi Niu, Hezhe Qiao, Changlu Chen, Ling Chen, and Guansong Pang. 2024. Zero-shot generalist graph anomaly detection with unified neighborhood prompts. arXiv preprint arXiv:2410.14886 (2024).

[26] Hezhe Qiao, Chaoxi Niu, Ling Chen, and Guansong Pang. 2025. Anomalygfm: Graph foundation model for zero/few-shot anomaly detection. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 2326–2337.

[27] Hezhe Qiao and Guansong Pang. 2023. Truncated afinity maximization: Oneclass homophily modeling for graph anomaly detection. Advances in Neural Information Processing Systems 36 (2023), 49490–49512.

[28] Hezhe Qiao, Hanghang Tong, Bo An, Irwin King, Charu Aggarwal, and Guansong Pang. 2025. Deep graph anomaly detection: A survey and new perspectives. IEEE Transactions on Knowledge and Data Engineering 37, 9 (2025), 5106–5126.

[29] Amir Rafe and Subasish Das. 2026. Benchmarking System One decision models against trained classifiers and language models for automated decision gates. arXiv preprint arXiv:2610.00346 (2026).

[30] Shebuti Rayana and Leman Akoglu. 2015. Collective opinion spam detection: Bridging review networks and metadata. In Proceedings ofthe 21th acm sigkdd international conference on knowledge discovery and data mining. 985–994.

[31] Simiao Ren, Kidus Zewde, Xingyu Shen, Yuchen Zhou, Dennis Ng, Ankit Raj, Tommy Duong, Yuxin Zhang, and Neo Tiangratanakul. 2026. Open-Jev Judg ments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model. arXiv preprint arXiv:2609.23959 (2026).

[32] Amit Roy, Juan Shu, Jia Li, Carl Yang, Olivier Elshocht, Jeroen Smeets, and Pan Li. 2024. Gad-nr: Graph anomaly detection via neighborhood reconstruction. In Proceedings ofthe 17th ACM international conference on web search and data mining. 576–585.

[33] Takaya Saito and Marc Rehmsmeier. 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PloS one 10, 3 (2015), e0118432.

[34] Bernhard Schölkopf, John C Platt, John Shawe-Taylor, Alex J Smola, and Robert C Williamson. 2001. Estimating the support of a high-dimensional distribution. Neural computation 13, 7 (2001), 1443–1471

[35] Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad. 2008. Collective classification in network data. AI magazine 29, 3 (2008), 93–106.

[36] Jianheng Tang, Jiajin Li, Ziqi Gao, and Jia Li. 2022. Rethinking graph neural networks for anomaly detection. In International conference on machine learning. PMLR, 21076–21089.

[37] Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. 2008. Arnetminer: extraction and mining of academic social networks. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 990–998.

[38] Lei Tang and Huan Liu. 2009. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining. 817–826.

[39] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 (2017).

[40] Zhengle Wang, Hanxu Yan, Fuheng Zhao, and Chunwei Liu. 2026. Prune First, Decide Fast: Scalable Semantic Query Processing with JEVDB. arXiv preprint arXiv:2610.02046 (2026).

[41] Zhiming Xu, Xiao Huang, Yue Zhao, Yushun Dong, and Jundong Li. 2022. Contrastive attributed network anomaly detection with data augmentation. In Pacific-Asia conference on knowledge discovery and data mining. Springer, 444–457.

[42] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

[43] Ge Zhang, Zhao Li, Jiaming Huang, Jia Wu, Chuan Zhou, Jian Yang, and Jianliang Gao. 2022. efraudcom: An e-commerce fraud detection system via competitive graph neural networks. ACM Transactions on Information Systems (TOIS) 40, 3 (2022), 1–29.

[44] Xiong Zhang, Zhenli He, Changlong Fu, and Cheng Xie. 2025. Ia-ggad: Zero-shot generalist graph anomaly detection via invariant and afinity learning. Advances in Neural Information Processing Systems 38 (2025), 83728–83760.

[45] Xinyu Zhao, Qingyun Sun, Jiayi Luo, Xingcheng Fu, and Jianxin Li. 2026. Zeroshot generalizable graph anomaly detection with mixture of riemannian experts. arXiv preprint arXiv:2602.06859 (2026).

[46] Yunfeng Zhao, Yixin Liu, Shiyuan Li, Qingfeng Chen, Yu Zheng, and Shirui Pan. 2025. Freegad: A training-free yet efective approach for graph anomaly detection. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management. 4379–4389.

[47] Lecheng Zheng, John Birge, Haiyue Wu, Yifang Zhang, and Jingrui He. 2025. Cluster aware graph anomaly detection. In Proceedings of the ACM on Web Conference 2025. 1771–1782.

[48] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems 36 (2023), 46595–46623.

[49] Xiangping Zheng, Bo Wu, Xun Liang, and Wei Li. 2024. Friend or foe? mining suspicious behavior via graph capsule infomax detector against fraudsters. In Proceedings of the ACM Web Conference 2024. 2684–2693.

[50] Cheng Zhou, Guangxia Li, Hao Weng, and Yiyu Xiang. 2025. Training-free graph anomaly detection: A simple approach via singular value decomposition. In Proceedings ofthe ACM on Web Conference 2025. 4196–4205.

## A Additional Method Details

## A.1 Graph Evidence Construction

Jev-GAD first converts graph statistics into directionally consistent, graph-relative evidence. We remove self-loops and symmetrize the adjacency matrix, treating an edge in either direction as an undirected connection. For a statistic $q _ { j } ( \boldsymbol { \upsilon } )$ , we use the empirical

mid-rank percentile

$$
R _ { G } ( q _ { j } ( v ) ) = \frac { \mathrm { r a n k } _ { \mathrm { a v g } } ( q _ { j } ( v ) ) - 1 } { | V | - 1 } ,\tag{7}
$$

where tied values receive their average rank. For structural quantities whose atypicality can occur at either tail, we rank $\lvert q _ { j } ( \upsilon ) -$ media $\ v { 1 } _ { u \in V } q _ { j } ( u ) \ v { | }$ instead.

Table 3 summarizes the seven evidence entries passed to the pretrained decision model. All features are measured on the input graph, and larger percentiles represent stronger deviations on the corresponding statistic.

Table 3: Node-level evidence fields in the semantic decision state.

![](images/d77884716c82fb71f0205ec1e8c733fd779205ea1ca48508bb0a8a64db4b76d0.jpg)

The decision state uses the aligned feature view, while the numerical scorers receive a more extensive evidence matrix. Specifically, the scoring matrix contains four structural columns, six attribute and neighborhood columns from the original features, and six from the aligned features, yielding 4 + 6 + 6 = 16 evidence columns. The extra attribute measurements include feature magnitude or cosine deviation, whitened distance, maximum standardized deviation, and residual energy, as applicable to the feature view. All columns are converted into graph-relative percentiles before statistical scoring.

Feature alignment. The 32-dimensional aligned representation contains a 16-dimensional Euclidean view and four 4-dimensional curvature views, with curvatures {−1, +1, −0.5, +0.5}. Each curvature view is constructed using manifold projection, exponential mapping, logarithmic mapping, and candidate selection, following the alignment construction in [45]. Candidate features are obtained by PCA when necessary and ranked by normalized graph Laplacian scores [8]; the lowest scores are retained for each view. This preprocessing uses a fixed random seed and does not optimize anomaly-specific model parameters.

## A.2 Semantic Graph-to-State Interface

Each node � is represented as a pair $\boldsymbol { S _ { v } } = \left( c _ { G } , B _ { v } \right)$ , corresponding to the graph\_context and node\_evidence objects. The graph-level context $c _ { G }$ contains the node and edge counts, edge density, mean degree, aligned feature dimension, and mean node–neighbor cosine similarity. The seven node-level percentiles in $B _ { v }$ are grouped in the fixed order neighborhood, structural, and attribute. Numeric values are serialized to six decimal places. This fixed organization preserves the correspondence between an observation and its anomaly-related meaning.

```latex
Algorithm A1 Training-Free Scoring with Jev-GAD
Input: Attributed graph $G = \left( V , E , X _ { 0 } \right)$ ; frozen decision model ��
Output: Node-level anomaly scores {� (�) } <sub>∈�</sub>
Stage I: Graph Evidence Construction
1: $\widetilde { X } \gets \mathrm { A l i g n } _ { 3 2 } ( G , X _ { 0 } )$
2: E ← BuildEvidence $( G , X _ { 0 } , { \widetilde { X } } )$
Stage II: Semantic Decision Readout
3: for all � ∈ � do
4: $S _ { v } \gets ( c _ { G } , \mathrm { S e l e c t } _ { 7 } ( \mathbf { E } _ { 1 6 } [ v ] ) )$
5: �<sub>dec</sub> (�) ← Readout<sub>�/�</sub> (�<sub>�</sub>, �<sub>�</sub> )
6: end for
Stage III: Statistical Scoring and Fusion
7: M ← {Fixed, ECOD, IF, OCSVM}
8: for all � ∈ M do
9: $s _ { m } \gets \mathrm { S c o r e } _ { m } ( \mathbf { E } _ { 1 6 } )$
10: end for
11: for all $v \in V _ { \mathbf { \sigma } _ { \mathbf { \sigma } } } \mathbf { d o } _ { \mathbf { \sigma } _ { \mathbf { \sigma } } }$
12: � (�) ← $\begin{array} { r } { \begin{array} { r } { \dot { R _ { G } } ( s _ { \mathrm { d e c } } ( v ) ) + \sum _ { m \in \mathcal { M } } R _ { G } ( s _ { m } ( v ) ) } \end{array} } \end{array}$
5
13: end for
```

Decision query and readout. The serialized state forms the evidence part of a fixed structured query. The criterion instructs the model to interpret all node-level values as graph-relative anomaly percentiles. Two labeled alternatives are provided: A: anomalous behavior, and B: ordinary node variation. The query presents the criterion, the candidate options, and then the node-specific evidence. The frozen Qwen/Qwen3-4B model processes the query with thinking disabled and without autoregressive answer generation. We extract the logits $\ell _ { A } ( \boldsymbol { v } )$ and $\ell _ { B } ( \boldsymbol { v } )$ at the answer position, yielding

$$
s _ { \mathrm { d e c } } ( v ) = \frac { \exp ( \ell _ { A } ( v ) ) } { \exp ( \ell _ { A } ( v ) ) + \exp ( \ell _ { B } ( v ) ) } .\tag{8}
$$

The backbone is loaded in FP16; option logits are computed in FP32. A common query prefix is reused across nodes to reduce repeated forward computation. Neither the backbone nor the decision interface is adapted with GAD-specific gradients.

## A.3 Statistical Scoring and Rank Fusion

The four numerical scoring rules operate on the 16-dimensional percentile evidence matrix $\boldsymbol { E } \in [ 0 , \bar { 1 } ] ^ { N \times 1 6 }$ : Fixed averages the 16 evidence values; ECOD [17] aggregates their upper-tail evidence, $- \textstyle \sum _ { j } \log ( 1 - E _ { v j } + 0 . 5 / N )$ ; Isolation Forest [18] uses 1,000 estimators; and OCSVM [34] uses an RBF kernel with gamma=scale. Isolation Forest uses five seeds {0, 1, 2, 3, 4}; the other scorers are deterministic under the evaluation protocol. Every score is individually converted into its empirical percentile rank before fusion. The final score is the arithmetic mean of the decision rank and the four numerical ranks, as defined in Eq. (6). Thus the default decision weight is 1/5, as examined in the fusion sensitivity experiment in the main text (Figure 5).

## B Experiment Details

## B.1 Dataset Details

We evaluate Jev-GAD on seven attributed graph anomaly detection benchmarks spanning three domains, following established

GAD benchmark protocols [20, 27, 45]. For source-trained baseline comparisons, the cross-graph protocol additionally uses four disjoint source graphs: PubMed [35], Flickr [3, 38], Reddit [13], and YelpChi [24, 30]. These source graphs are used only by the applicable baselines and are not used to train or configure Jev-GAD. Table 4 reports statistics of the seven target graphs used for evaluation.

• Citation Domain: ACM, CiteSeer, and Cora [3, 20], where nodes represent papers and edges denote citation links. Node attributes are derived from paper content, and the benchmark versions contain injected structural and attribute anomalies.

• Social Domain: BlogCatalog [27] is a social-blogging network with user profile and content features and injected anomalies; Facebook and Weibo [20] are social networks with annotated anomalous accounts or behaviors.

• Co-review Domain: Amazon [12, 30] connects reviewers through co-review activity, and its anomalies arise from suspicious review patterns.

Anomaly labels are used only for metric computation on the seven evaluation graphs.

Table 4: Statistics of the seven benchmark graphs. Edges are undirected and self-loop-free; Ratio is the percentage of anomalous nodes and Deg. the average degree.
<table><tr><td>Dataset</td><td>Domain</td><td>#Node</td><td>#Edge</td><td>#Feat.</td><td>#Anom.</td><td>Ratio</td><td>Deg.</td></tr><tr><td>ACM</td><td>Citation</td><td>16,484</td><td>74,073</td><td>8,337</td><td>597</td><td>3.62</td><td>8.99</td></tr><tr><td>CiteSeer</td><td>Citation</td><td>3,327</td><td>5,077</td><td>3,703</td><td>150</td><td>4.51</td><td>3.05</td></tr><tr><td>Cora</td><td>Citation</td><td>2,708</td><td>5,803</td><td>1,433</td><td>150</td><td>5.54</td><td>4.29</td></tr><tr><td>BlogCatalog</td><td>Social</td><td>5,196</td><td>172,783</td><td>8,189</td><td>300</td><td>5.77</td><td>66.51</td></tr><tr><td>Facebook</td><td>Social</td><td>1,081</td><td>27,552</td><td>576</td><td>25</td><td>2.31</td><td>50.98</td></tr><tr><td>Weibo</td><td>Social</td><td>8,405</td><td>377,271</td><td>400</td><td>868</td><td>10.33</td><td>89.77</td></tr><tr><td>Amazon</td><td>Co-review</td><td>10,224</td><td>175,608</td><td>25</td><td>693</td><td>6.78</td><td>34.35</td></tr></table>

## B.2 Baseline Details

We compare Jev-GAD with 15 baselines from four categories.

• Supervised GAD: GCN [11] and GAT [39], representative graph neural networks based on neighborhood aggregation and attention, and BWGNN [36] and GHRN [7], two GAD-oriented graph learning methods. All four are trained with labeled examples.

• Unsupervised GAD: AnomalyDAE [5], which reconstructs graph structure and attributes; CoLA [22], which contrasts nodes with their contextual subgraphs; and TAM [27], which models node–neighbor afinity.

• Generalist GAD: UNPrompt [25], AnomalyGFM [26], and IA-GGAD [44], which learn transferable anomaly detection capabilities across graph domains; ARCzero [21], evaluated in its zero-shot transfer setting; GAD-MoRE [45], which models crossgraph anomaly evidence with a mixture of geometric experts; and ARC (10-shot) [20], which adapts across graphs with in-context examples and serves as a label-assisted reference rather than a strict label-free baseline. Methods with prior training on source graphs are grouped separately from training-free methods.

• Training-free GAD: TFGAD [50] and FreeGAD [46], the two dedicated training-free baselines, which perform no anomalyspecific parameter training on the target graph.

Source-Graph Training Protocol. Source-trained baselines following the cross-graph protocol of [45] are trained on the four source graphs described in Appendix B.1 and evaluated on seven disjoint target graphs. Supervised baselines use source-graph anomaly labels, whereas unsupervised baselines train without them. Previously reported results are taken from their respective published evaluations where applicable. We reproduce ARCzero, TFGAD, and FreeGAD with their available implementations on the graph versions used by Jev-GAD, keeping the implementation settings instead of tuning individual test graphs; ARCzero uses five trials, and TFGAD and FreeGAD use five random seeds.

## B.3 Implementation Details

We introduce the general implementation details below.

Decision readout. The frozen Qwen/Qwen3-4B backbone is loaded in FP16 with thinking disabled. The two option logits are computed in FP32 at the answer position without autoregressive decoding, and decision inference uses a batch size of 32. The inference cache is keyed by the backbone, numerical precision, query template, and node-specific prompt digest, so the rank fusion and the weightsensitivity analysis can be recomputed from saved scores without repeating language-model inference.

Scoring and evaluation. Graph preprocessing, feature alignment, evidence construction, and statistical scoring run independently for each graph on the CPU; feature alignment and residual-energy computation are restricted to one thread. Isolation Forest results are averaged over five seeds, and the main tables report mean and standard deviation. AUROC and AUPRC are computed from node-level anomaly scores; labels are not required to construct target-graph evidence, decision states, or inference scores.

Environment. Experiments are conducted with:

• Operating System: Ubuntu 24.04.3 LTS (Linux kernel 6.8).

• CPU: two AMD EPYC 7742 64-core processors (256 threads) with 1007 GiB of memory.

• GPU: eight NVIDIA A100-SXM4-80GB GPUs (driver 580.65.06), one GPU per inference process. The eficiency measurements in Section 5.7 use one GPU, processing about 120 nodes per second with about 9.6 GiB of peak allocated memory.

• Software: CUDA 13.0, Python 3.12.3, PyTorch 2.14.1, Transformers 5.19.0, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, NetworkX 3.7, and Geoopt 0.5.1.

## C Additional Experimental Analysis

## C.1 Decision-State Evidence Granularity

Following the analysis of semantic grounding in the main text, we further investigate the number of evidence entries supplied to the decision model. The 5D state retains two neighborhood cues, two structural rarity cues (degree and PageRank), and one aligned attribute-distance cue. The default 7D state adds clustering and core-number rarity. The 10D state additionally introduces whitened distance, maximum standardized deviation, and residual energy. All variants retain the same scoring branch and fusion protocol.

Table 5: AUPRC (%) ofthe fused method with diferent semantic state sizes. The 7D configuration is the default Jev-GAD setting.
<table><tr><td>Dataset</td><td>5D</td><td>7D</td><td>10D</td></tr><tr><td>Amazon</td><td>37.48</td><td>41.26</td><td>42.34</td></tr><tr><td>BlogCatalog</td><td>32.86</td><td>33.50</td><td>33.73</td></tr><tr><td>CiteSeer</td><td>85.29</td><td>88.02</td><td>86.23</td></tr><tr><td>Cora</td><td>74.12</td><td>76.10</td><td>73.58</td></tr><tr><td>Macro average</td><td>57.44</td><td>59.72</td><td>58.97</td></tr></table>

As shown in Table 5, the seven-field representation achieves the highest macro-AUPRC across the four diagnostic datasets. Expanding the state from 5D to 7D improves all four datasets. Extending it to 10D provides small additional gains on Amazon and BlogCatalog but reduces performance on CiteSeer and Cora. The analysis illustrates that decision quality depends on organizing the most useful semantic evidence, rather than simply increasing the number of numerical fields passed to the pretrained model.

## C.2 Cumulative Anomaly Retrieval

We examine how pretrained decision scores improve anomaly prioritization by comparing cumulative anomaly retrieval under Statistical-only scoring and Jev-GAD on CiteSeer. As shown in Figure 7, Jev-GAD retrieves 122 true anomalies among the top-150 ranked nodes, compared with 105 for Statistical-only. This improvement of 17 additional detections demonstrates the benefit of incorporating pretrained decisions into anomaly ranking.

![](images/7a50a787c9e89e64f5185c569878a4f103cce4acc6936f2e24a6870a6909bd6d.jpg)  
Figure 7: Cumulative anomaly hits at diferent ranking depths on CiteSeer. Jev-GAD retrieves 122 true anomalies at Top-150, compared with 105 for Statistical-only.

## C.3 Alternative Fifth-Scorer Controls

We examine whether pretrained decision scores can be replaced by additional numerical scoring mechanisms. The original four statistical scorers remain fixed, while the fifth score is selected from Mean-7D, Max-7D, LOF-7D, or pretrained semantic decisions. All five-score configurations employ equal-weight rank fusion, while Statistical-only uses the original four-score aggregation.

Table 6: Alternative fifth-scorer controls on four diagnostic datasets (AUPRC, %).
<table><tr><td>Dataset</td><td>Stat.</td><td>Mean</td><td>Max</td><td>LOF Semantic</td></tr><tr><td>Amazon</td><td>34.67</td><td>28.11</td><td>25.80 30.93</td><td>41.26</td></tr><tr><td>BlogCatalog</td><td>32.59</td><td>30.23</td><td>33.80 28.87</td><td>33.50</td></tr><tr><td>CiteSeer</td><td>71.06</td><td>70.29</td><td>71.36 56.13</td><td>88.02</td></tr><tr><td>Cora</td><td>67.34</td><td>71.29</td><td>69.20 52.63</td><td>76.10</td></tr><tr><td>Macro Avg.</td><td>51.41</td><td>49.98</td><td>50.04 42.14</td><td>59.72</td></tr></table>

As shown in Table 6, pretrained semantic decisions achieve the highest macro-AUPRC of59.72%, exceeding Statistical-only (51.41%) and all three numerical fifth-scorer alternatives. The gains are particularly pronounced on CiteSeer and Cora, while Max-7D performs slightly better on BlogCatalog. Overall, pretrained decisions provide complementary anomaly-ranking information beyond additional numerical scoring.

## C.4 Conditional Ranking Beyond Statistical Scores

We further examine whether pretrained decisions distinguish anomalous nodes with similar statistical anomaly scores. For each graph, nodes are partitioned into ten equal-frequency bins according to their Statistical-only scores, with tied scores assigned to the same bin. Within bins containing both anomalous and normal nodes, we compute pairwise ranking AUC for each scoring method and aggregate the results weighted by the number of valid anomaly–normal pairs.

Table 7: Pair-weighted within-bin AUC conditioned on Statistical-only anomaly scores.
<table><tr><td>Dataset</td><td>Stat. Mean</td><td>Max</td><td>LOF</td><td>Semantic</td></tr><tr><td>Amazon</td><td>0.592</td><td>0.353</td><td>0.314 0.511</td><td>0.611</td></tr><tr><td>BlogCatalog</td><td>0.613</td><td>0.495</td><td>0.566 0.544</td><td>0.593</td></tr><tr><td>CiteSeer</td><td>0.783</td><td>0.660</td><td>0.703 0.491</td><td>0.914</td></tr><tr><td>Cora</td><td>0.750</td><td>0.747</td><td>0.679 0.512</td><td>0.813</td></tr><tr><td>Macro Avg.</td><td>0.684</td><td>0.564</td><td>0.565 0.514</td><td>0.733</td></tr></table>

Table 7 shows that pretrained semantic decisions achieve a macro within-bin AUC of 0.733, compared with 0.684 for Statistical-only and 0.564, 0.565, and 0.514 for Mean-7D, Max-7D, and LOF-7D, respectively. The improvements are most pronounced on CiteSeer and Cora, demonstrating that semantic decisions provide additional discriminative information even among nodes with similar statistical scores.

The analysis considers only bins containing both anomalous and normal nodes, covering 6.68%–8.99% of all anomaly–normal pairs across the four graphs.