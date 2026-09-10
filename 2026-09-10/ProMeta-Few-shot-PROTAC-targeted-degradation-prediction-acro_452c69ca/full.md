doi: Preprint version Preprint posted: September 2026 Preprint

PREPRINT

# ProMeta: Few-shot PROTAC-targeted degradation prediction across E3 ligases

Yuansheng Liu<sup>1,2</sup>, Yufei Ye<sup>1,2</sup>, Tao Tang<sup>3</sup>, Jiawei Luo<sup>1,2</sup>, Wen Tao<sup>4,∗</sup>, Xiao Luo<sup>5,∗</sup>

<sup>1</sup>College of Computer Science and Electronic Engineering, Hunan University, Changsha, Hunan 410086, China <sup>2</sup>Yuelushan Laboratory, Changsha, Hunan 410128, China

<sup>3</sup>School of Modern Posts, Nanjing University of Posts and Telecommunications, Nanjing, Jiangsu 210023, China

<sup>4</sup>College of Computing and Data Science, Nanyang Technological University, Singapore 639798, Singapore

<sup>5</sup>Hunan Research Center of the Basic Discipline for Cell Signaling, College of Biology, Hunan University, Changsha, Hunan 410086, China \*Corresponding author: taowen228@gmail.com; xluo@hnu.edu.cn

## Abstract

Motivation: Proteolysis-targeting chimeras (PROTACs) have emerged as a transformative therapeutic strategy that selectively degrades historically “undruggable” targets via the ubiquitin–proteasome system. Despite growing eforts to develop computational predictors of PROTAC degradation activity, existing supervised approaches remain severely challenged by data scarcity and imbalance across E3 ligases, limiting their ability to generalize beyond well-studied ligase contexts. In practice, labeled data are heavily concentrated on a few ligases (e.g., CRBN and VHL), while the majority of E3 ligases remain underexplored yet are critical for expanding the design space of targeted degraders. Developing methods that enable robust cross-ligase generalization with minimal labeled data is therefore essential for improving the practical utility of computational PROTAC discovery.

Results: We reformulate PROTAC degradation activity prediction across E3 ligases as a few-shot meta-learning problem and present ProMeta, a prototype-based graph neural network trained through episodic meta-learning on source-E3 tasks and evaluated on held-out target-E3 tasks through support-conditioned inference. ProMeta performs inference without updating the encoder by dynamically estimating class prototypes from minimal target-ligase support samples. On the CRBN-to-VHL benchmark, ProMeta achieves AUROC values of 0.796 under K = 2, Q = 3 and 0.883 under K = 2, Q = 5, improving by 19.9% and 6.8%, respectively, over the corresponding supervised GNN baseline. Reverse VHL-to-CRBN transfer under the same protocol yielded AUROC values of 0.702 (K = 2, Q = 3) and 0.821 (K = 2, Q = 5), confirming bidirectional applicability while revealing direction and data-regime dependence. Under one-shot evaluation on rare E3 ligases, ProMeta attains an AUROC of 0.700. Case-study analyses further demonstrate predictive consistency on VZ185- derived candidates and two additional retrospective chemical series. Together, these results support ProMeta as a practical framework for cross-ligase few-shot prediction under the evaluated support/query protocols.

Availability and implementation: The source code is available at https://github.com/yeyufeiyyf/prometa, and the datasets, experimental splits, model checkpoints, and released prediction results are archived on Zenodo at https: //doi.org/10.5281/zenodo.21371599.

## Introduction

Proteolysis-targeting chimeras (PROTACs) have emerged as a promising therapeutic strategy for targeted protein degradation [1, 2]. Unlike conventional small-molecule inhibitors that rely on occupancy-driven mechanisms, PROTACs recruit an E3 ubiquitin ligase to a protein of interest (POI), inducing ternary complex formation and subsequent ubiquitination and proteasomal degradation [3]. Operating through an event-driven pharmacological mechanism, PROTACs act catalytically, enabling sustained target depletion at substoichiometric doses and potentially reducing side efects [4, 5]. Importantly, this paradigm allows engagement of proteins traditionally considered “undruggable” [6, 7]. Collectively, these properties position PROTACs as a transformative modality with the potential to fundamentally reshape therapeutic strategies for challenging protein targets.

Despite their considerable promise, the rational design of efective PROTAC molecules remains challenging. A typical PROTAC comprises three components: a warhead targeting the protein of interest, a ligand recruiting an E3 ubiquitin ligase, and a linker connecting the two. Degradation eficacy depends on ligand binding afinity and linker length and flexibility [7]; productive degradation further depends on the cooperativity and structural compatibility of ternary-complex recognition [8]. Experimental screening of large PROTAC libraries is both timeconsuming and costly [9, 10]. With the increasing availability of curated datasets such as PROTAC-DB [11], researchers have explored machine learning approaches for activity prediction.

Structure-aware models, including graph neural networks (GNNs) [12], DeepPROTAC [13], and PROTAC-STAN [14], as well as multimodal frameworks such as DegradeMaster [15], have demonstrated encouraging performance by capturing molecular and protein-level interactions. A recent geometryaware predictor, SE(3)-PROTACs, combines an SE(3)- equivariant molecular encoder with POI and E3 proteinsequence representations [16]. Nevertheless, whether these prediction frameworks transfer to heterogeneous, label-scarce cross-ligase settings remains insuficiently established.

Despite promising progress, existing approaches face critical limitations. Most models rely heavily on large annotated datasets and exhibit limited generalization across E3 ligases, target proteins, and chemical scafolds, while real-world PROTAC data are typically scarce and highly imbalanced. In our raw PROTAC-DB export, over 80% of entries lack degradation activity annotations $( \mathrm { D C } _ { 5 0 }$ and $\operatorname { D } _ { \operatorname { m a x } } ;$ data source: PROTAC-DB 3.0 [11]), and the labeled data used here are predominantly concentrated in CRBN and VHL (Supplementary Fig. S1). This concentration motivates, but does not itself demonstrate, the challenge posed by rare-ligase settings. Therefore, developing methods capable of robust cross-ligase generalization under minimal labeled data is critical for enabling data-eficient exploration of underrepresented ligase contexts.

To address this challenge, we reformulate PROTAC degradation activity prediction across E3 ligases as a fewshot meta-learning problem. This formulation follows the support/query episodic paradigm used by Matching Networks [17] and Prototypical Networks [18], while difering from gradient-based fast-adaptation methods such as MAML [19]; broader meta-learning taxonomies are reviewed by Hospedales et al. [20]. Each E3-ligase-defined prediction context is treated as a task. E3 identity defines the meta-training/meta-test task split: source-E3 episodes are used for episodic meta-training, whereas held-out target-E3 tasks are instantiated at meta-test from labeled support compounds and evaluated on disjoint query compounds. The class labels remain active/inactive within each task, and target-E3 query labels are never used for prototype construction or model fitting. Most ligases beyond CRBN and VHL have only limited labeled samples available, and the objective is to generalize to unseen ligases given only a small support set. This setting departs from conventional supervised learning, which assumes a shared data distribution and suficient annotations, and instead requires learning transferable representations that remain useful under a ligase-defined task shift.

Motivated by this perspective, we propose ProMeta, a prototype-based graph neural network trained with episodic meta-learning. ProMeta encodes each molecule as a graph and augments the molecular representation with protein sequence embeddings of both the POI and the E3 ligase, providing biologically informed context. At inference, this learned representation is complemented by a fixed ECFP4 molecular fingerprint before prototype construction. By learning across diverse source-E3 episodes, the model is optimized for support-conditioned classification on held-out E3 tasks. Class prototypes are then estimated on the fly from a minimal targetligase support set, enabling prediction for held-out E3 ligases without encoder updates. We evaluate ProMeta on crossligase transfer benchmarks, including bidirectional transfer between CRBN and VHL and an exploratory extreme lowdata evaluation across rare E3 ligases, supporting ProMeta as a practical framework for data-scarce PROTAC degradation prediction.

## Materials and Methods

## Dataset collection and preprocessing

Data source and activity metrics. To evaluate the proposed framework and all baseline methods, we utilized data from PROTAC-DB [11], a publicly available repository of experimentally characterized PROTAC molecules. The raw local PROTAC-DB export used in this study contains 9,384 rows; this approximately 9.38K-row raw export should be distinguished from the smaller filtered labeled tables used for model training and evaluation. Records provide molecular structures, POI and E3 annotations, and, where available, degradation evidence including $\mathrm { D C } _ { 5 0 }$ and $\operatorname { D } _ { \operatorname* { m a x } } .$ These metrics serve as standard quantitative measures of degradation eficiency [21, 22]: a lower $\mathrm { D C } _ { 5 0 }$ indicates greater potency, and a higher $\mathrm { D } _ { \mathrm { m a x } }$ indicates more complete target elimination.

Binary labeling and data filtering. Binary activity labels were assigned following the protocol of Li et al. [13], incorporating both quantitative $\mathrm { D C _ { 5 0 } / D _ { m a x } }$ values and qualitative activity calls derived from experimental descriptions. For rows with complete quantitative evidence, a compound was designated as high degradation activity when $\mathrm { D C } _ { 5 0 } \quad < \quad 1 0 0$ nM and $\begin{array} { r l r } { \operatorname { D } _ { \mathrm { { m a x } } } } & { { } \ge } & { 8 0 \% ; } \end{array}$ qualitativeonly rows were retained only when an explicit activity call was available. Other retained labeled rows were assigned to the low degradation activity class. Raw rows without usable quantitative measurements or qualitative activity calls were excluded rather than treated as negative examples. A highquality labeled subset was curated through three sequential filtering steps: (i) entries with missing SMILES, UniProt IDs, or activity labels were discarded; (ii) all structures were validated with RDKit [23]; and (iii) duplicates were removed via canonical SMILES comparison. For the strict CRBN/VHL transfer experiments, duplicate compound–E3– target contexts with conflicting binary labels were removed before support/query split generation, yielding 1,386 final clean CRBN/VHL contexts (842 CRBN and 544 VHL; 690 positive and 696 negative labels).

Class imbalance adjustment. The raw export exhibits pronounced imbalance across E3 ligases: CRBN (6,041 rows) and VHL (2,858 rows) account for most records before strict activity-evidence filtering (Supplementary Fig. S1). To mitigate class imbalance in the labeled modeling table, random majority-class down-sampling [24] was applied independently to CRBN and VHL. The balanced labeled dataset shown in Supplementary Fig. S1 and Table 1 comprises 860 CRBN samples (430 high-activity / 430 low-activity) and 560 VHL samples (280 high-activity / 280 low-activity). This distribution summary precedes the additional context-level duplicate/conflict cleanup that produced the stricter 1,386- context CRBN/VHL analysis set used for the main transfer experiments (Supplementary Table S1).

Protein sequence retrieval. For each compound in the final dataset, the amino acid sequences of both the POI and the recruited E3 ligase were retrieved from UniProt. POI sequences were obtained using the provided UniProt identifiers, with 89 entries failing to map to a valid sequence and subsequently removed. E3 ligase sequences were retrieved via a manually curated name-to-UniProt mapping covering all ligases present in the dataset. Sequences were truncated to a maximum length of 2,000 residues. To clarify how the final analysis set was obtained without overloading the main text, Supplementary Table S1 summarizes the preprocessing flow from the raw PROTAC-DB export to the strict CRBN/VHL context-level dataset used for the main transfer experiments. The 1,420-record CRBN/VHL subset is reported only as the candidate pool before duplicate/conflict cleanup; the main strict CRBN/VHL transfer experiments use the final 1,386 unique contexts.

Table 1. Sample distribution across major E3 ligases in the balanced labeled dataset. Ligases not listed individually are pooled as “Others”.
<table><tr><td>E3 ligase</td><td>Low-activity</td><td>High-activity</td><td>Total</td></tr><tr><td>CRBN</td><td>430</td><td>430</td><td>860</td></tr><tr><td>VHL</td><td>280</td><td>280</td><td>560</td></tr><tr><td>cIAP1</td><td>8</td><td>8</td><td>16</td></tr><tr><td>IAP</td><td>7</td><td>7</td><td>14</td></tr><tr><td>XIAP</td><td>3</td><td>3</td><td>6</td></tr><tr><td>FEM1B</td><td>2</td><td>2</td><td>4</td></tr><tr><td>MDM2</td><td>2</td><td>2</td><td>4</td></tr><tr><td>Others</td><td>129</td><td>8</td><td>137</td></tr></table>

## Episodic meta-learning dataset construction

Ligase-disjoint data partitioning. To ensure unbiased assessment of cross-ligase generalization, we adopted a ligasedisjoint episodic data partition. For CRBN→VHL and rare-E3 transfer, CRBN-associated molecules formed the source pool; for the reverse VHL→CRBN benchmark, VHL-associated molecules formed the source pool. Each source pool was split into training (80%) and validation (20%) partitions, and the target ligase was excluded from both. The rare-E3 targets were cIAP1, IAP, MDM2, XIAP, and FEM1B. Ligases without both positive and negative samples were excluded. No compound overlap existed between source and target partitions, and all experiments were conducted under three random seeds (42, 2025, 3407).

Benchmark design and justification. CRBN-recruiting and VHL-recruiting PROTACs employ chemically distinct ligands that occupy non-overlapping regions of chemical space. Their POI targets are also largely non-overlapping, ensuring the CRBN-to-VHL benchmark reflects genuine cross-distribution transfer. We selected CRBN as the primary source E3 ligase because it is the most data-rich ligase in PROTAC-DB, reflecting a realistic transfer from a data-rich source to held-out target-ligase contexts. We additionally evaluated the reverse VHL-to-CRBN direction under the same ligase-disjoint episodic protocol as a complementary test of direction dependence. In each transfer direction, the target ligase was excluded from source training and used only through labeled support compounds for prototype construction, while query compounds were reserved exclusively for evaluation.

## Framework overview

An overview of the architecture of ProMeta and its key components is illustrated in Fig. 1. As depicted in Fig. 1, ProMeta integrates three principal components: a molecular representation module combining learned graph features [12] with protein features and a fixed molecular fingerprint at inference, a prototype-based classifier, and an episodic metalearning training scheme [18]. Consistent with prototypical few-shot learning, the encoder is meta-trained on source-E3 episodes, whereas cross-ligase meta-testing on a held-out target E3 is performed by constructing prototypes from that ligase’s support set. Class prototypes are then estimated from a small labeled support set, and a query molecule is assigned to the class whose prototype is nearest in the learned representation space. This design confers two key advantages over conventional supervised classifiers: (i) the episodic metatraining objective directly matches the few-shot inference setting, promoting the acquisition of broadly transferable molecular representations; and (ii) the classification mechanism is non-parametric and requires no parameter update when the target E3 ligase changes, enabling immediate deployment to newly characterized ligase contexts.

## Molecular graph representation

Graph construction. Each PROTAC molecule is represented as an attributed graph $\textit { G } = \ : ( V , E )$ , where nodes $ { \boldsymbol { v } } \in  { \boldsymbol { V } }$ correspond to heavy atoms and edges e ∈ E to covalent bonds. Graphs were constructed from SMILES strings using RDKit [23]. Node features are encoded as a 9-dimensional one-hot vector representing the atom type, covering the nine most prevalent elements in PROTAC structures: C, N, O, F, P, S, Cl, Br, and I. Edges are included for all covalent bonds and stored as undirected pairs (two directed edges per bond); no edge features are used. All graphs were validated prior to GNN input, and molecules that failed RDKit parsing were excluded during preprocessing.

GNN encoder. We adopt a Graph Convolutional Network (GCN) [25] as the molecular encoder. Let $h _ { v } ^ { ( l ) } \in \mathbb { R } ^ { d }$ denote the representation of node v at layer l, initialized with the atom feature vector $( l = 0 )$ . Each layer updates node representations through neighborhood aggregation:

$$
h _ { v } ^ { ( l + 1 ) } = \phi \Bigl ( h _ { v } ^ { ( l ) } , \mathrm { \ A G G } \Bigl ( \bigl \{ h _ { u } ^ { ( l ) } : u \in \mathcal { N } ( v ) \bigr \} \Bigr ) ,\tag{1}
$$

where $\mathcal { N } ( v )$ denotes the set of atoms bonded to v, AGG(·) is a permutation-invariant aggregation function (mean pooling in this work), and $\phi ( \cdot )$ is a linear transformation followed by ReLU activation. Stacking L such layers integrates structural information from the L-hop neighborhood of each atom.

A graph-level representation is obtained by global mean pooling over all final node embeddings:

$$
\mathbf { z } = \frac { 1 } { | V | } \sum _ { v \in V } h _ { v } ^ { ( L ) } \in \mathbb { R } ^ { d } ,\tag{2}
$$

where z serves as the molecular embedding for all downstream operations. Mean pooling was used as a size-normalizing design choice to reduce sensitivity to molecular size, which varies considerably across PROTAC compounds.

Protein sequence encoding and feature fusion. To incorporate biological context, the molecular embedding z is augmented with protein sequence embeddings of the POI and the E3 ligase. For each compound, the amino acid sequence is encoded by a lightweight sequence encoder consisting of a learnable embedding layer over the 20 standard amino acids, followed by mean pooling across residue positions and a linear projection into $\mathbb { R } ^ { d }$ . Let $\mathbf { z } ^ { \mathrm { { p o i } } }$ and $\mathbf { z } ^ { \mathrm { { e 3 } } }$ denote the resulting sequence embeddings for the POI and E3 ligase, respectively.

![](images/ecb62d011d2464fd5503a3aac4c966a39df070a7fba8e1e9c3abaf3b0cad0a6c.jpg)  
Fig. 1. Overview of ProMeta, a prototype-based few-shot meta-learning framework for cross-ligase PROTAC degradation activity prediction. PROTAC molecules are constructed as molecular graphs from SMILES strings via RDKit. During episodic meta-training on a source E3, a shared GNN encoder maps molecules into an embedding space, from which class prototypes are computed as the mean embeddings of support-set molecules per class. Query molecules are classified by distance to the prototypes, and encoder parameters are updated through the episodic query loss. At cross-ligase meta-test, the encoder is frozen and transferred to a held-out target E3, enabling support-conditioned prediction without encoder updates. Before prototypes and query distances are computed, the L2-normalized learned representation is concatenated with a weighted L2-normalized 512-bit ECFP4 fingerprint.

The fused molecular representation is defined as:

$$
{ \bf \tilde { z } = z + \alpha _ { \mathrm { p o i } } \mathrm { \bf z ^ { \mathrm { p o i } } } } + \alpha _ { \mathrm { e 3 } } \mathrm { \bf z ^ { \mathrm { e 3 } } } ,\tag{3}
$$

where $\alpha _ { \mathrm { p o i } }$ and $\alpha _ { \mathrm { e 3 } }$ are learnable scalar weights initialized to 0.1, allowing the model to adaptively balance molecular structure and biological sequence signals. The fused embedding z˜ forms the learned molecular/protein block of the representation used for prototype construction.

Molecular fingerprint fusion. To complement the learned representation with a fixed descriptor of local chemical substructures, we used RDKit [23] to generate a 512-bit extended-connectivity fingerprint with diameter four (ECFP4; Morgan radius = 2) [26] from each molecule’s SMILES string. Let f<sub>i</sub> denote the fingerprint of molecule i. At inference, the learned molecular/protein embedding and fingerprint are independently L2-normalized and concatenated as

$$
\hat { \bf z } _ { i } = \left[ \frac { \tilde { \bf z } _ { i } } { \lVert \tilde { \bf z } _ { i } \rVert _ { 2 } } \right] \bigg | \lambda _ { \mathrm { f p } } \frac { { \bf f } _ { i } } { \lVert { \bf f } _ { i } \rVert _ { 2 } } \bigg ] ,\tag{4}
$$

where ∥ denotes concatenation and $\lambda _ { \mathrm { f p } } ~ = ~ 0 . 5$ controls the contribution of the fingerprint block. The fingerprint is fixed rather than learned and is introduced only when forming support and query features for prototype-based inference on the target ligase.

## Prototype-based classification

For an episodic task with support set $\begin{array} { r } { { \cal { S } } ~ = ~ \{ ( { \bf x } _ { i } , y _ { i } ) \} } \end{array}$ , the prototype of class $c \in \{ 0 , 1 \}$ is defined as the centroid of its

support-set embeddings:

$$
\mathbf { p } _ { c } = \frac { 1 } { | S _ { c } | } \sum _ { ( \mathbf { x } _ { i } , y _ { i } ) \in S _ { c } } f ( \mathbf { x } _ { i } ) ,\tag{5}
$$

where $S _ { c } ~ = ~ \{ ( \mathbf { x } _ { i } , y _ { i } ) ~ \in ~ \mathcal { S } ~ : ~ y _ { i } ~ = ~ c \}$ . The representation $f ( \mathbf { x } _ { i } )$ is $\tilde { \mathbf { z } } _ { i }$ during episodic meta-training and the fingerprintaugmented representation $\hat { \mathbf { z } } _ { i }$ (Eq. 4) during target-ligase metatest inference. For a query molecule $\mathbf { x } _ { q } .$ the Euclidean distance to each prototype is computed as:

$$
d _ { c } = \left\| f ( \mathbf { x } _ { q } ) - \mathbf { p } _ { c } \right\| _ { 2 } ,\tag{6}
$$

and class assignment probabilities are obtained via a softmax over negative distances, where $c ^ { \prime }$ denotes all classes in {0, 1}:

$$
P ( y = c \mid \mathbf { x } _ { q } ) = \frac { \exp ( - d _ { c } ) } { \sum _ { c ^ { \prime } } \exp ( - d _ { c ^ { \prime } } ) } .\tag{7}
$$

Euclidean distance was selected over cosine similarity because it directly reflects absolute displacement in the embedding space, which is more meaningful when prototype positions carry geometric significance. During inference, encoder weights are frozen, and prototypes are recomputed solely from the available support set, requiring no gradient updates for novel E3 ligase tasks.

## Episodic meta-learning

Training follows the episodic meta-learning protocol of Snell et al. [18]. At each iteration, an episode is constructed by sampling a 2-way activity classification problem (active vs. inactive) from the relevant source-E3 training pool, with K support samples and Q query samples per class. The class labels within an episode are active/inactive, while E3 identity defines the metatraining/meta-test task split: source-E3 episodes are used to meta-train the encoder, and held-out target-E3 support/query episodes define the meta-test tasks. The encoder processes all $2 ( K + Q )$ molecules, prototypes are estimated from the 2K support embeddings (Eq. 5), and the cross-entropy loss over the 2Q query predictions is minimized:

$$
\mathcal { L } = - \sum _ { ( \mathbf { x } _ { q } , y _ { q } ) \in Q } \log P ( y _ { q } \mid \mathbf { x } _ { q } ) .\tag{8}
$$

$\mathrm { B y }$ sampling diverse episodes throughout training, the encoder is optimized to produce embeddings in which withinepisode class prototypes are well separated—a representational property intended to support prototype-based few-shot prediction when the E3-ligase-defined task changes at test time.

For evaluation, episodes are drawn exclusively from heldout target E3 ligases (VHL, CRBN, or a rare ligase, depending on the transfer setting) that were never observed during the corresponding source-ligase training run. Target-ligase support compounds are used only to compute active/inactive prototypes, and target-ligase query compounds are used only for evaluation. In the bidirectional CRBN/VHL fixed-query benchmarks, target episodes use K = 2 support samples per class with either Q = 3 or Q = 5 query samples per class. In the rare-E3 benchmark, each held-out ligase/seed episode uses K = 1 support sample per class (one active and one inactive compound), followed by an all-remaining query protocol in which all eligible compounds from the same ligase that are not selected as support are used as query candidates. Thus, the rare-E3 benchmark contains one episode per rare ligase per seed, yielding five ligases × three seeds = 15 episodes before common-query coverage matching across baselines. Additional K/Q sensitivity analyses are provided in the Supplementary Information. This strict cross-ligase partition ensures that reported performance metrics reflect transfer to target-ligase chemical space rather than interpolation within the sourceligase chemical space.

## Evaluation metrics

Performance was evaluated using five complementary metrics: accuracy (ACC), macro F1-score (F1), balanced accuracy (BALACC), AUROC, and AUPRC. AUROC and AUPRC are designated as the primary metrics throughout, as they are threshold-independent and robust to class imbalance [24]. Accuracy, BALACC, and F1 are reported as secondary metrics for reference; under the class-imbalanced episodic setting, macro F1 may favour methods with a tendency toward majority-class prediction and should be interpreted with caution. For episodic experiments, each metric was averaged across all tasks and episodes, and over three independent random seeds; results are reported as the mean and standard deviation.

## Implementation details

All models were implemented in Python 3.9 using PyTorch [27] 2.0 and PyTorch Geometric 2.3. Molecular graph construction and 512-bit ECFP4 fingerprint generation (Morgan radius = 2) were performed using RDKit 2023.03 [23]. For ProMeta inference, the learned and fingerprint feature blocks were independently L2-normalized and concatenated using a fingerprint weight of 0.5. The GNN encoder comprised $L = 3$ message-passing layers with hidden dimension d = 128, followed by global mean pooling. Parameters were optimized using Adam [28] $( \beta _ { 1 } ~ = ~ 0 . 9 , ~ \beta _ { 2 } ~ = ~ 0 . 9 9 9 )$ with a learning rate of $1 \times 1 0 ^ { - 3 }$ and weight decay of $1 \times 1 0 ^ { - 5 }$ . Hyperparameters were selected using the validation partition containing 20% of the corresponding source-ligase POI targets, withheld from the training pool.

For each transfer direction, episodic meta-training tasks were sampled exclusively from the corresponding source-E3 molecules: CRBN for CRBN→VHL and VHL for VHL→CRBN $( K = 2 , Q = 3$ per class during meta-training). The meta-trained encoder was then evaluated on support/query episodes from the held-out target E3 at meta-test. Evaluation configurations varied by benchmark and are described in Section 3. In each direction, the target E3 ligase was completely excluded from meta-training. All evaluations used the same predefined support/query split-generation protocol and fixed seeds (42, 2025, and 3407); within each direction and K/Q setting, all compared methods received identical support/query rows to ensure a fair and reproducible comparison.

## Baseline evaluation protocol

A key methodological challenge in evaluating cross-ligase few-shot generalization is that existing PROTAC activity predictors were not originally designed for cross-ligase support/query inference. To enable comparison within this setting, we re-evaluated or adapted all baselines in our pipeline using shared support/query splits. Our baselines include PROTAC-STAN [14], DegradeMaster [15], a support-trained supervised GNN [25], and an ECFP feature representation [26] coupled to a random forest [29]. We emphasize that these baselines were developed for diferent prediction settings: in particular, DegradeMaster targets general supervised or semisupervised PROTAC prediction with 3D E(3)-equivariant modeling, whereas ProMeta is designed for cross-ligase few shot prediction. For $\mathrm { R F + \ E C F P } .$ , a 500-tree RF is trained only on source-ligase ECFP4, RDKit5, and POI–E3 ACC features. Each compound is then represented by the frozen vector of per-tree positive-class probabilities concatenated with ECFP4, and an L2-regularized logistic head is fitted using only the target support set. For the supervised GNN baseline, model parameters are randomly initialized and trained exclusively on the support set for each episode, with POI and E3 ligase sequence features incorporated via elementwise fusion. For PROTAC-STAN and DegradeMaster—which do not natively support this E3-separated episodic protocol— we retrain each model on the corresponding source-ligase training pool following its original procedure and default hyperparameters, then adapt the classifier head to each episode’s support set via 50 gradient steps using AdamW, with all backbone parameters frozen. Query labels are used only for metric calculation. For each episode, the same available support and query rows—defined by shared split files generated under three random seeds (42, 2025, and 3407)—were supplied to every method; common-query subsets were used when an external model lacked complete input coverage. ProMeta instead concatenates the learned representation with the fixed ECFP4 block and computes class prototypes directly from target support compounds, requiring no encoder update at inference. Executable implementation details and frozen split files are supplied in the reproducibility repository.

A. AUROC
<table><tr><td rowspan=3 colspan=1>ProMetaRF + ECFP</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.883</td><td rowspan=1 colspan=1>0.702</td><td rowspan=1 colspan=1>0.821</td></tr><tr><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>0.613</td><td rowspan=1 colspan=1>0.702</td></tr><tr><td rowspan=1 colspan=1>Supervised GNN</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.827</td><td rowspan=1 colspan=1>0.487</td><td rowspan=1 colspan=1>0.541</td></tr><tr><td rowspan=1 colspan=1>PROTAC-STAN</td><td rowspan=1 colspan=1>0.560</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>0.624</td><td rowspan=1 colspan=1>0.560</td></tr><tr><td rowspan=1 colspan=1>DegradeMaster</td><td rowspan=1 colspan=1>0.588</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.608</td><td rowspan=1 colspan=1>0.668</td></tr><tr><td rowspan=2 colspan=3>K2Q3  K2Q5CRBN-&gt;VHL</td><td rowspan=1 colspan=2>K2Q3  K2Q5</td></tr><tr><td rowspan=1 colspan=2>VHL-&gt;CRBN</td></tr></table>

B. AUPRC  
![](images/d47958874e6a4a7b7e25c3a53b9371f96bbd45c8972abd15febe3722f59c0e8a.jpg)

C. BALACC  
![](images/f15ba883311e1968f448412197329bfd965dbc60b19384969eb4cd06eb66d0e5.jpg)  
Fig. 2. Bidirectional CRBN/VHL transfer performance under strict episodic evaluation. Heatmaps show mean (A) AUROC, (B) AUPRC, and (C) balanced accuracy (BALACC) across three random seeds for the CRBN-to-VHL and VHL-to-CRBN $K = 2 , Q = 3$ and $K = 2 , Q = 5$ settings. The same predefined support/query split-generation protocol was used throughout, and all methods received identical support/query rows within each direction and K/Q setting. Darker colors indicate higher scores, and bold cell values indicate the best-performing method within each transfer setting and metric. Complete mean ± standard deviation values are provided in Supplementary Table S2.

As part of the ablation analyses, we further isolated the efect of the decision rule from the efect of the learned representation. In this matched-head analysis, GNN and PROTAC-STAN embeddings and DegradeMaster E(3)- equivariant latent features were evaluated with the same support-set prototype classifier; ECFP4 fingerprints were evaluated with nearest-centroid inference, and ProMeta was also evaluated with a support-trained linear head. DegradeMaster latent-feature results use the subset for which its required inputs are available. This reciprocal comparison tests whether ProMeta’s performance is attributable only to non-parametric prototype inference or to the combination of episodic representation learning and support-based prototype construction.

## Results

## Bidirectional cross-ligase transfer between CRBN and VHL

We first evaluate all methods on the CRBN to VHL transfer benchmark, in which models trained exclusively on CRBNassociated PROTAC molecules are assessed on episodic tasks constructed from VHL-associated molecules. Scafold analysis confirmed zero overlap between the CRBN training set and both evaluation sets, with cross-set Tanimoto similarity substantially lower than within-CRBN similarity (0.145 and 0.163 vs. 0.289; Supplementary Fig. S2), confirming that both benchmarks present genuine out-of-distribution generalization challenges at the molecular level. UMAP visualization of the learned molecular embeddings further reveals that CRBN and VHL compounds occupy largely distinct regions of the embedding space (Supplementary Fig. S3), with degradation-active compounds forming loosely separable clusters within each ligase context, as further confirmed by VHL-specific embedding analysis (Supplementary Fig. S4). This setting directly probes out-of-distribution generalization across E3 ligases, a central challenge in computational PROTAC modeling.

In this primary CRBN-to-VHL benchmark (Fig. 2), ProMeta retained the main conclusion of the original experiment. Under $K \ = \ 2 , Q \ = \ 3 .$ ProMeta achieved an AUROC of 0.796, compared with 0.785 for $\mathrm { R F + E C F P }$ and 0.664 for the supervised GNN baseline. Under $K \ = \ 2 , Q \ =$ 5, ProMeta achieved the highest AUROC of 0.883, while RF + ECFP achieved 0.842. These results indicate that a representation learned from CRBN episodes can support fewshot prediction on VHL using only a small VHL support set.

To test whether this result was specific to one transfer direction, we additionally evaluated the reverse VHL-to-CRBN setting under the same strict cross-ligase episodic protocol. Reverse-transfer performance remained lower than CRBN-to-VHL transfer, consistent with the smaller VHL source set and the larger, more diverse CRBN target space. ProMeta achieved an AUROC of 0.702 under $K = 2 , Q = 3$ and 0.821 under $K = 2 , Q = 5 ,$ , compared with 0.613 and 0.702, respectively, for RF + ECFP. Thus, the bidirectional benchmark preserves the original CRBN-to-VHL interpretation while showing that few-shot cross-ligase generalization is empirically directionand data-regime-dependent under substantial distribution shift. Complete mean ± standard deviation values are provided in Supplementary Table S2.

To assess whether the stronger CRBN-to-VHL direction was attributable only to the larger CRBN source pool, we performed an independent 10-seed full-source versus VHL-size-matched retraining control. Under $K = 2 , Q = 3 ,$ the full-source and size-matched conditions achieved AUROCs of 0.731 ± 0.079 and $0 . 7 1 5 \pm 0 . 1 0 4$ , respectively; under $K \ = \ 2 , Q \ = \ 5 ,$ the corresponding values were 0.883±0.044 and 0.869±0.081. These descriptive sensitivity results are reported in Supplementary Table S3 and indicate that matching the source-pool size did not produce a consistent performance collapse.

![](images/202d2a9fcc1f2e7f8f37ab2ebf9db43c4c083c77b8a0d7ed2c3057d8ff58c62c.jpg)

![](images/650e8775aec21543b5b63a908673266a03ea998bfc13baec456f743cf0047024.jpg)  
Fig. 3. Rare-E3 common-query evaluation under the K = 1 all-remaining-query protocol. (A) Overall AUROC, AUPRC, and balanced accuracy (BALACC) across 15 held-out rare-E3 episodes, constructed with one active and one inactive support compound per episode, and 90 strict common query label-agree rows after coverage matching. (B) Per-ligase AUROC, with n indicating the number of strict common-query label-agree rows after coverage matching. Dashed lines indicate the random AUROC reference of 0.5. For individual ligase–method pairs with very small query pools, an AUROC below 0.5 can result from only a few reversed positive–negative rankings and is not interpreted as evidence of stable inverse prediction. Full per-ligase AUROC/AUPRC/BALACC values are provided in Supplementary Table S5.

Additional K/Q sensitivity analyses, including larger fixedquery and all-remaining-query variants, are reported in Supplementary Table S4.

## Cross-ligase transfer: CRBN to rare E3 ligases

After evaluating bidirectional CRBN/VHL transfer, we next examine a more challenging rare-E3 setting with extremely limited labeled data. We construct a benchmark comprising five low-frequency ligases (FEM1B, IAP, MDM2, XIAP, and cIAP1), each with only a small number of labeled compounds in the curated dataset. This setting reflects a realistic deployment scenario in which only a few experimentally labeled compounds are available for novel ligase targets. For each rare ligase and random seed, the support set contains K = 1 compound per activity class (one active and one inactive compound), and all remaining eligible compounds from the same ligase are used as query candidates after support selection rather than sampling a fixed numeric Q. This yields one episode per ligase per seed, or 15 held-out rare-E3 episodes across five ligases and three seeds. To ensure matched comparison across baselines, we report a strict common-query evaluation covering these 15 episodes and 90 strict common-query label-agree rows after coverage matching. The rare-E3 counts in Table 1 denote candidate labeled pools before episodic support/query construction, whereas the query rows reported here are episodelevel evaluated rows after support selection and common-query coverage matching; unique query counts and reuse rates are audited in Supplementary Table S6.

As this benchmark targets cross-ligase generalization under extreme data scarcity, we focus on methods that can be applied to new E3-defined tasks with minimal supervision. Methods like RF and GNN trained from scratch are not well-suited to this setting, as the limited support samples provide insuficient signal to reliably learn task-specific decision boundaries. Therefore, we compare against PROTAC-STAN and DegradeMaster, two representative PROTAC degradation activity predictors evaluated here under the rare-E3 episodic protocol.

As shown in Fig. 3, ProMeta achieves the best aggregate rare-E3 common-query performance, with an AUROC of 0.700 across the 90 strict common-query label-agree rows. Under the same coverage-matched evaluation, PROTAC-STAN obtains an AUROC of 0.550, and DegradeMaster obtains an AUROC of 0.564. These results indicate that support-set prototype construction can preserve useful discriminative structure even when target-ligase labels are scarce, although the margin over adapted baselines remains modest in this extremely low-data setting.

The complete per-ligase AUROC values and query-row counts are provided in Supplementary Table S5, and the corresponding episode-level confidence intervals and support/query reuse audit are provided in Supplementary Table S6. The per-ligase results are heterogeneous, but ProMeta achieves the highest AUROC across all five rare ligases. Some adapted-baseline point estimates fall below the random AUROC reference of 0.5. This behavior should not be interpreted as a stable tendency to predict the opposite biological label: AUROC is a pairwise ranking statistic, and in the smallest groups—for example, FEM1B with 5 episodelevel query rows representing only 3 unique query compounds, and XIAP with 12 rows representing 6 unique compounds— one or two reversed positive–negative rankings can change the estimate substantially. Moreover, under K = 1, the supportconditioned decision rule is highly sensitive to which single active and inactive compounds define the episode, and query compounds may recur across the three seeds. The below-0.5 values therefore reflect unstable realized rankings under an extreme one-shot stress test and should be read together with the confidence intervals and reuse audit in Supplementary Table S6. Because several rare-E3 groups contain only a few common-query rows, we interpret this benchmark as an informative low-data stress test rather than as evidence for broad rare-ligase superiority.

Table 2. Core ablation results. Each cell reports mean AUROC averaged over the $K = 2 , Q = 3$ and $K = 2 , Q = 5$ settings.
<table><tr><td>Method</td><td>CRBN→VHL VHL→CRBN</td><td></td></tr><tr><td>ProMeta</td><td>0.840</td><td>0.761</td></tr><tr><td>ProMeta + linear head</td><td>0.814</td><td>0.691</td></tr><tr><td>ProMeta w/o episodic training</td><td>0.818</td><td>0.554</td></tr><tr><td>GNN encoder + prototype head</td><td>0.806</td><td>0.520</td></tr><tr><td>ECFP4 nearest-centroid</td><td>0.799</td><td>0.689</td></tr></table>

## Ablation study

To isolate the contributions of episodic training and the support-set prototype decision rule, we expanded the original ablation beyond the previous GNN/GNN+MAML/ProMeta comparison. The revised ablation includes ProMeta, ProMeta + linear head, ProMeta without episodic training, supervised GNN encoder + prototype head, and ECFP4 nearestcentroid controls. Table 2 summarizes the core mean AUROC results averaged over the two K/Q settings within each transfer direction. Consistent with Supplementary Tables S2 and S7, all $\begin{array} { r c l r c l } { K } & { = } & { 2 , Q } & { = } & { 3 } & { } \end{array}$ and $\begin{array} { r c l r } { K } & { = } & { 2 , Q } & { = } & { 5 } \end{array}$ evaluations use the same predefined support/query splitgeneration protocol and the same three split seeds; within each direction and K/Q setting, all compared methods receive identical support/query rows. Full per-setting mean ± standard deviation values and additional matched-head controls are provided in Supplementary Tables S7 and S8.

The comparison with ProMeta without episodic training shows a direction-dependent contribution. Removing episodic training changed mean AUROC from 0.840 to 0.818 in CRBNto-VHL transfer and from 0.761 to 0.554 in VHL-to-CRBN transfer. The larger reverse-transfer gap indicates that the benefit of episodic representation learning is not uniform across data regimes.

The matched-head controls further show that the prototype decision rule is beneficial but not suficient by itself. The ProMeta + linear head experiment provides the direct classification-head control. Replacing prototype inference with a support-trained linear head reduced mean AUROC in both transfer directions, while applying prototype inference to supervised GNN embeddings or ECFP4 fingerprints also remained below ProMeta. Taken together, these results indicate that neither episodic training nor prototype inference alone fully accounts for the observed cross-ligase performance. Instead, the combination of episodic representation learning with a support-set, non-parametric prototype decision mechanism provides a suitable inductive bias for the evaluated data-scarce transfer setting.

## Case study: VZ185-derived PROTAC candidates

Following the evaluation protocol of DeepPROTAC [13], we assessed the practical utility of ProMeta beyond episodic benchmarks in a case study of 12 structurally related PROTAC candidates derived from the VZ185 scafold [30]. These compounds all recruit VHL and share the same warhead and E3 ligase ligand substructures, difering mainly in linker composition. This controlled setting examines each model’s sensitivity to linker-associated structure– activity relationships. As shown in Fig. 4, ProMeta achieves 91.7% classification accuracy (11 out of 12 compounds), outperforming DeepPROTAC, PROTAC-STAN, and DegradeMaster on this small retrospective task. To examine performance beyond the VZ185 series, we additionally evaluated two independent retrospective medicinal-chemistry series using the prespecified main-experiment checkpoint family. ProMeta correctly classified 11 of 14 CRBN-recruiting CDK6 degraders (78.6%) and all 6 VHL-recruiting BCLxL degraders (100.0%) [31, 32]; compound-level experimental annotations and predictions are provided in Supplementary Table S9 and the reproducibility files. Given their small sample sizes, these additional series provide complementary scafoldlevel evidence and should not be interpreted as evidence of broad chemical-series superiority. ProMeta correctly classifies several active VZ185-series compounds that all three baseline models misclassify, which is consistent with sensitivity to linkerassociated structural variation. Linker length and geometry are known to influence ternary-complex formation and the spatial positioning required for productive ubiquitin transfer [8, 30]; the present retrospective result suggests, but does not establish mechanistically, that ProMeta may capture related structure–activity signals.

## Conclusion and discussion

ProMeta addresses a specific limitation that is not the primary focus of most existing PROTAC activity predictors: crossligase degradation prediction under limited labeled data. Unlike conventional supervised approaches, ProMeta reformulates this setting as a few-shot learning problem for cross-ligase generalization and predicts target-ligase query compounds by constructing active/inactive prototypes from a small target support set without updating model parameters. The revised bidirectional CRBN/VHL experiments show that ProMeta is not limited to the original CRBN-to-VHL direction, but also reveal that cross-ligase few-shot generalization is directionand data-regime-dependent under substantial distribution shift. The K/Q sensitivity and classification-head controls further indicate that the conclusion is not driven by a single support/query split or by a conventional linear head.

A notable observation is that RF + ECFP, despite relying largely on 2D fingerprints, achieves competitive AUROC values in selected settings. This result indicates that representation complexity alone does not determine performance under the evaluated protocol and that source– target distribution mismatch remains important. CRBN and VHL recruit structurally distinct chemical scafolds and engage diferent POI distributions, which provides a plausible chemical and biological basis for the observed direction dependence; the present predictive experiments do not, however, isolate a single mechanistic cause. The lower performance of PROTAC-STAN and DegradeMaster in our episodic evaluation should therefore be interpreted in light of their original objectives: they were designed for general supervised or semisupervised PROTAC prediction, with DegradeMaster further emphasizing 3D E(3)-equivariant modeling, rather than for cross-ligase fewshot support/query prediction. Thus, our results do not claim general superiority over these models in their native prediction setting; instead, they show that architectures optimized for data-rich prediction do not automatically solve cross-ligase few-shot generalization.

Despite its advantages, ProMeta currently relies on complementary 2D molecular graph and ECFP4 fingerprint representations together with protein-sequence context, without explicit 3D conformational or ternary-complex information.

<table><tr><td>Compound ID</td><td>Linker</td><td>Linker Length</td><td>E3 ligase ligand</td><td colspan="4">Prediction Correctness</td></tr><tr><td></td><td></td><td></td><td></td><td>DeepPROTAC PROTAC-STAN DegradeMaster ProMeta</td><td></td><td></td><td></td></tr><tr><td>C1</td><td></td><td>8</td><td>VHL4</td><td>1</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C2</td><td></td><td>8</td><td>VHL3</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C3</td><td></td><td>9</td><td>VHL4</td><td>1</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C4</td><td></td><td>13</td><td>VHL4</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C5</td><td></td><td>11</td><td>VHL4</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td>C6</td><td></td><td>5</td><td>VHL4</td><td>1</td><td>1</td><td>0</td><td>1</td></tr><tr><td>C7(VZ185)</td><td></td><td>5</td><td>VHL4</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C8</td><td></td><td>5</td><td>VHL4</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>C9</td><td></td><td>8</td><td>VHL4</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C10</td><td></td><td>11</td><td>VHL4</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>C11</td><td></td><td>5</td><td>VHL4</td><td>0</td><td>1</td><td>0</td><td>1</td></tr><tr><td>C12</td><td></td><td>8</td><td>VHL4</td><td>0</td><td>0</td><td>1</td><td>0</td></tr></table>

Fig. 4. Degradability prediction of VZ185-derived PROTAC candidates [30]. Compounds C13–C16 lack a POI-binding warhead and were excluded; C1–C12 were used for evaluation. Each cell indicates whether a model’s prediction matches the experimental label (1 = correct, 0 = incorrect). Heatmap columns correspond to DeepPROTAC, PROTAC-STAN, DegradeMaster, and ProMeta.

Lightweight 3D augmentation controls and an intersectiontrained, coverage-limited diagnostic comparison with the geometry-aware SE(3)-PROTACs baseline [16] are provided in Supplementary Tables S10 and S11, respectively. In addition, the available source-ligase data remain concentrated in CRBN and VHL, and the rare-E3 analyses are necessarily limited by small query sets. Therefore, the cross-ligase generalization evidence should be interpreted within the controlled fewshot support/query evaluation used here rather than as exhaustive validation across all E3 ligase families. Future work will incorporate E(3)-equivariant encoders and structurebased ternary complex features, and extend meta-training to additional source-ligase contexts as PROTAC-DB continues to grow.

In summary, we present ProMeta, a prototype-based framework for cross-ligase few-shot PROTAC degradation prediction. Across the evaluated protocols, ProMeta provides a practical strategy for predicting held-out E3 ligase contexts with minimal labeled support data, with the strongest evidence in CRBN-to-VHL transfer and more heterogeneous but informative behavior in the reverse VHL-to-CRBN and rare-E3 settings. These results support ProMeta as a few-shot framework for cross-ligase generalization. As therapeutically relevant E3 ligases continue to expand beyond well-characterized cases, methods that can generalize with minimal labeled data will be important for accelerating nextgeneration PROTAC discovery.

## Supplementary Data

Supplementary data are available with this preprint.

Conflict of interests

No conflict of interest is declared.

## Acknowledgements

This study was supported by Yuelushan Laboratory Breeding Program (No. YLS-2025-ZY03024); the National Natural Science Foundation of China (Grant No. 62372159, 32400506); the Natural Science Foundation of Hunan Province (Grant No. 2024JJ4008); Fundamental Research Funds for the Central Universities (Grant No. 541109030062).

## References

1. Kathleen M Sakamoto, Kyung B Kim, Akiko Kumagai, Frank Mercurio, Craig M Crews, and Raymond J Deshaies. Protacs: Chimeric molecules that target proteins to the skp1–cullin–f box complex for ubiquitination and degradation. Proceedings of the National Academy of Sciences, 98(15):8554–8559, 2001.

2. Daniel P Bondeson, Alina Mares, Ian ED Smith, Eunhwa Ko, Sebastien Campos, Afjal H Miah, Katie E Mulholland, Natasha Routly, Dennis L Buckley, Jefrey L Gustafson, et al. Catalytic in vivo protein knockdown by smallmolecule protacs. Nature chemical biology, 11(8):611–617, 2015.

3. Georg E Winter, Dennis L Buckley, Joshiawa Paulk, Justin M Roberts, Amanda Souza, Sirano Dhe-Paganon, and James E Bradner. Phthalimide conjugation as a strategy for in vivo target protein degradation. Science, 348(6241):1376–1381, 2015.

4. Momar Toure and Craig M Crews. Small-molecule protacs: new approaches to protein degradation. Angewandte Chemie International Edition, 55(6):1966–1973, 2016.

5. Taavi K Neklesa, James D Winkler, and Craig M Crews. Targeted protein degradation by protacs. Pharmacology & therapeutics, 174:138–144, 2017.

6. George M Burslem and Craig M Crews. Small-molecule modulation of protein homeostasis. Chemical reviews, 117(17):11269–11301, 2017.

7. Mikl´os B´ek´es, David R Langley, and Craig M Crews. PROTAC targeted protein degraders: the past is prologue. Nature Reviews Drug Discovery, 21(3):181–200, 2022.

8. Morgan S Gadd, Andrea Testa, Xavier Lucas, Kwok-Ho Chan, Wenzhang Chen, Douglas J Lamont, Michael Zengerle, and Alessio Ciulli. Structural basis of PROTAC cooperative recognition for selective protein degradation. Nature Chemical Biology, 13(5):514–521, 2017.

9. Jiantao Hu, Biao Hu, Mingliang Wang, Fuming Xu, Bukeyan Miao, Chao-Yie Yang, Mi Wang, Zhaomin Liu, Daniel F Hayes, Krishnapriya Chinnaswamy, et al. Discovery of ERD-308 as a highly potent proteolysis targeting chimera (PROTAC) degrader of estrogen receptor (ER). Journal of Medicinal Chemistry, 62(3):1420–1442, 2019.

10. Chien-Ting Kao, Chieh-Te Lin, Cheng-Li Chou, and Chu-Chung Lin. Fragment linker prediction using the deep encoder-decoder network for protacs drug design. Journal of chemical information and modeling, 63(10):2918–2927, 2023.

11. Jingxuan Ge, Shimeng Li, Gaoqi Weng, Huating Wang, Meijing Fang, Huiyong Sun, Yafeng Deng, Chang-Yu Hsieh, Dan Li, and Tingjun Hou. PROTAC-DB 3.0: an updated database of PROTACs with extended pharmacokinetic parameters. Nucleic Acids Research, 53(D1):D1510– D1515, 2025.

12. Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In International conference on machine learning, pages 1263–1272. Pmlr, 2017.

13. Fenglei Li, Qiaoyu Hu, Xianglei Zhang, Renhong Sun, Zhuanghua Liu, Sanan Wu, Siyuan Tian, Xinyue Ma, Zhizhuo Dai, Xiaobao Yang, et al. Deepprotacs is a deep learning-based targeted degradation predictor for protacs. Nature communications, 13(1):7133, 2022.

14. Zhenglu Chen, Chunbin Gu, Shuoyan Tan, Xiaorui Wang, Yuquan Li, Mutian He, Ruiqiang Lu, Shijia Sun, Chang-Yu Hsieh, Xiaojun Yao, et al. Interpretable PROTAC Degradation Prediction With Structure-Informed Deep Ternary Attention Framework. Advanced Science, 12(47):e08138, 2025.

15. Jie Liu, Michael J Roy, Luke Isbel, and Fuyi Li. Accurate protac-targeted degradation prediction with degrademaster. Bioinformatics, 41(Supplement 1):i342– i351, 2025.

16. Akash Reddy Kothakapu, Sharanya Madugula, Saketh Bharadwaj Sharma Gandeed, Kavya Sri Sai Yadlapati, Sharanya Sury, and Vani Kondaparthi. SE(3)-PROTACs: geometric deep learning for PROTAC degradation prediction. Briefings in Bioinformatics, 27(3):bbag228, 2026.

17. Oriol Vinyals, Charles Blundell, Timothy Lillicrap, and Daan Wierstra. Matching networks for one shot learning. In Advances in Neural Information Processing Systems, volume 29, pages 3630–3638, 2016.

18. Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, volume 30, pages 4077– 4087, 2017.

19. Chelsea Finn, Pieter Abbeel, and Sergey Levine. Modelagnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

20. Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5149–5169, 2022.

21. Mariell Pettersson and Craig M Crews. Proteolysis targeting chimeras (protacs)—past, present and future. Drug Discovery Today: Technologies, 31:15–27, 2019.

22. Zi Liu, Mingxing Hu, Yu Yang, Chenghao Du, Haoxuan Zhou, Chengyali Liu, Yuanwei Chen, Lei Fan, Hongqun Ma, Youling Gong, et al. An overview of protacs: a promising drug discovery paradigm. Molecular biomedicine, 3(1):46, 2022.

23. RDKit: Open-source cheminformatics. https://www.rdkit. org, 2023.

24. Haibo He and Edwardo A Garcia. Learning from imbalanced data. IEEE Transactions on knowledge and data engineering, 21(9):1263–1284, 2009.

25. Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks, 2016.

26. David Rogers and Mathew Hahn. Extended-connectivity fingerprints. Journal of chemical information and modeling, 50(5):742–754, 2010.

27. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An open source machine learning framework. Advances in Neural Information Processing Systems, 32, 2019.

28. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

29. Leo Breiman. Random forests. Machine learning, 45(1):5– 32, 2001.

30. Vittoria Zoppi, Scott J Hughes, Chiara Maniaci, Andrea Testa, Teresa Gmaschitz, Corinna Wieshofer, Manfred Koegl, Kristin M Riching, Danette L Daniels, Andrea Spallarossa, et al. Iterative design and optimization of initially inactive proteolysis targeting chimeras (protacs) identify vz185 as a potent, fast, and selective von hippel– lindau (vhl) based dual degrader probe of brd9 and brd7. Journal of medicinal chemistry, 62(2):699–726, 2019.

31. Shiqing Su, Zhan Yang, Hui Gao, Hua Yang, Sijin Zhu, Ziyu An, Jing Wang, Qing Li, Sarat Chandarlapaty, Hu Deng, Weiping Wu, and Yao Rao. Potent and preferential degradation of CDK6 via proteolysis targeting chimera degraders. Journal of Medicinal Chemistry, 62(16):7575– 7582, 2019.

32. Xuan Zhang, Dhanusha Thummuri, Xiang Liu, Wei Hu, Ping Zhang, Sajid Khan, Yong Yuan, Daohong Zhou, and Guangrong Zheng. Discovery of PROTAC BCL-XL degraders as potent anticancer agents with low ontarget platelet toxicity. European Journal of Medicinal Chemistry, 192:112186, 2020.