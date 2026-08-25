# Hierarchy-Aware Semantic Losses for Knowledge Graph Link Prediction

Filip Kronstr¨om

filipkro@chalmers.se

Department of Computer Science and Engineering,

Chalmers University of Technology and University of Gothenburg, Sweden

Ross D. King

rossk@chalmers.se

Department of Computer Science and Engineering,

Chalmers University of Technology and University of Gothenburg, Sweden

Department of Chemical Engineering and Biotechnology, University of Cambridge, United Kingdom

Editors: Alessandra Mileo, Andrea Passerini and Cogan Shimizu

## Abstract

Knowledge graphs are often accompanied by ontological class hierarchies that encode valuable semantic information, yet many link prediction methods either ignore such hierarchies or incorporate them indirectly through additional graph edges. Recent work introduced hierarchy-aware graph neural networks (GNNs), which use semantic losses derived from box embeddings to encourage satisfaction of subclass relationships during GNN-based representation learning. While this approach has shown promise for biological regression tasks, its efectiveness for knowledge graph link prediction has not been investigated.

In this paper we evaluate hierarchy-aware semantic losses on link prediction across three benchmark datasets: AIFB, CoDEx, and BioKG. We combine graph neural network encoders with box-embedding-based semantic losses that encourage learned representations to better satisfy ontology-derived class hierarchies, and compare this approach to both standard link prediction models and models incorporating subclass relations as graph edges. Across all datasets, hierarchy-aware semantic losses significantly improve mean reciprocal rank (MRR) and consistently outperform models that incorporate hierarchy information through additional subclass edges. Relative to the baseline GNN models, MRR improved by 7.6%, 2.4%, and 15.5% on AIFB, CoDEx, and BioKG, respectively. Furthermore, semantic losses consistently outperform the alternative of augmenting the graph with subclass edges.

These results are consistent with ontology-derived class hierarchies providing comple mentary information to graph structure, and suggest that encouraging hierarchical consistency through semantic losses is an efective and comparatively parameter-eficient mech anism for improving knowledge graph link prediction.

## 1. Introduction

Knowledge graphs (KGs) are widely used to represent structured relational knowledge across domains such as biomedicine, recommender systems, and semantic web applications. Many KGs can be enriched with ontological information, where entities are organised into class hierarchies through relations such as subClassOf. These hierarchies encode valuable background knowledge about conceptual similarity and inheritance structure, and can therefore provide useful inductive bias for representation learning.

A common task that motivates knowledge graph representation learning is link prediction: estimating missing relations between entities based on the observed graph structure.

Contemporary link prediction methods typically learn low-dimensional embeddings of entities and relations by optimising reconstruction objectives over observed triples. Translational methods such as TransE (Bordes et al., 2013) and RotatE (Sun et al., 2019), bilinear models such as DistMult (Yang et al., 2015) and ComplEx (Trouillon et al., 2016), and graph neural network (GNN) approaches such as R-GCNs (Schlichtkrull et al., 2018) have all achieved strong empirical performance on benchmark datasets. However, these methods generally learn representations from the observed graph structure and do not explicitly enforce semantic consistency with ontological hierarchies. As a result, learned embeddings may violate known class inclusion relationships even when such information is available in the data.

Recent work has explored several approaches for incorporating hierarchical information into representation learning. Within knowledge graph embedding, methods such as HAKE (Zhang et al., 2020) and hyperbolic embedding approaches (Nickel and Kiela, 2017) introduce geometric inductive biases that capture hierarchical structure and improve link prediction performance. However, these methods capture hierarchical patterns through geometric properties of the embedding space and do not directly incorporate ontology-derived concepts or logical axioms.

A complementary line of research focuses on embedding ontologies formulated in description logics. Box embeddings (Vilnis et al., 2018) provide a geometric representation in which concepts are modelled as axis-aligned hyperrectangles and subsumption relations correspond naturally to geometric containment. This idea has inspired several ontology embedding methods, including ELEm (Kulmanov et al., 2019), ELBE (Peng et al., 2022), BoxEL (Xiong et al., 2022), and $\mathrm { B o x ^ { 2 } E L }$ (Jackermeier et al., 2024), which construct geometric representations that approximate model-theoretic interpretations of ontology axioms. These methods show that ontological constraints can be encoded efectively in continuous vector spaces, particularly for biomedical ontologies formulated in lightweight logics such as $\mathcal { E } \mathcal { L } ^ { + + }$

More generally, semantic loss functions provide a mechanism for incorporating symbolic knowledge into neural models by penalising predictions that violate logical constraints during training (Xu et al., 2018). The geometric representations described above provide a natural way of expressing such constraints in terms of hierarchical relations between concepts.

Building on these ideas, Kronstr¨om et al. (2026) introduced a hierarchy-aware graph neural network framework that combines GNN-based message passing with semantic loss terms derived from ontological class hierarchies. The method represents classes using box embeddings and optimises semantic losses encouraging subclass containment while simultaneously training on a downstream prediction task. Applied to a heterogeneous biological KG describing the yeast Saccharomyces cerevisiae, this approach improved predictive performance on gene deletion fitness prediction and produced embeddings that better reflected the underlying ontology structure.

In this paper, we investigate whether the hierarchy-aware semantic loss framework introduced by Kronstr¨om et al. (2026) generalises beyond biological regression to knowledge graph link prediction. Our contribution is a systematic empirical evaluation of this framework across three benchmark datasets, together with an analysis of how hierarchy characteristics relate to the observed performance gains. We evaluate the framework on BioKG (Walsh et al., 2020) from the Open Graph Benchmark (OGB) (Hu et al., 2020), CoDEx (Safavi and Koutra, 2020), and AIFB (Bloehdorn and Sure, 2007), spanning diferent domains, scales, and degrees of ontological structure.

![](images/4113135b25d79aa072ba5dcc65dff50d642a884c307d8bc5280c5c5f09829e21.jpg)  
Figure 1: Overview of the hierarchy-aware GNN architecture. Semantic losses are applied to the GNN encoder to produce embeddings encouraged to better satisfy subsumptions defined in ontologies. This is optimised jointly with a task-specific decoder, e.g., for link prediction.

## 2. Hierarchy-aware GNNs

Inspired by Kronstr¨om et al. (2026), we incorporate semantic losses derived from ontological class hierarchies into GNN-based KG embeddings. The approach combines a GNN encoder with semantic losses derived from class hierarchies. This encoder is learnt along with a taskspecific decoder, jointly optimising the semantic losses describing the hierarchical constraints and the task-specific loss. An overview of the architecture can be seen in Figure 1.

We represent concepts as axis-aligned hyperrectangles in a latent space,

$$
\operatorname { B o x } = \prod _ { i = 1 } ^ { n } [ z _ { i } , Z _ { i } ] ,\tag{1}
$$

where $z _ { i }$ and $Z _ { i }$ denote lower and upper coordinates along dimension i. Following Chheda et al. (2021), valid boxes are constructed from latent variables, θ, as

$$
z _ { i } ( \theta _ { i } ) = \theta _ { i } ^ { z } , \qquad Z _ { i } ( \theta _ { i } ) = z _ { i } + \mathrm { s o f t p l u s } ( \theta _ { i } ^ { Z } ) .\tag{2}
$$

Instead of the lower and upper coordinates, boxes can also be represented by their centerpoint, $c _ { i } .$ , and ofset, $o _ { i } .$ , along each dimension i.

Hierarchical relations are modelled geometrically through containment: for a subsumption relation $C \subseteq D$ , the box corresponding to C is encouraged to lie inside the box corresponding to D. These constraints are incorporated through a distance-based semantic loss operating on box embeddings, inspired by Peng et al. (2022); Jackermeier et al. (2024) and successfully used as semantic loss by Kronstr¨om et al. (2026). For two boxes C and D, we define the signed element-wise box distance:

$$
d ( C _ { i } , D _ { i } ) = | c _ { i } ^ { C } - c _ { i } ^ { D } | - o _ { i } ^ { C } - o _ { i } ^ { D } ,\tag{3}
$$

where negative values indicate overlap between the boxes in a given dimension, while positive values indicate separation. The semantic inclusion loss is then defined as

$$
\mathcal { L } _ { \boldsymbol { \Xi } } ( \boldsymbol { C } , D ) = \Big | \Big | \Big ( \operatorname* { m a x } ( 0 , d ( C _ { i } , D _ { i } ) + 2 o _ { i } ^ { C } ) \Big ) _ { i = 1 } ^ { n } \Big | \Big | _ { 2 } ,\tag{4}
$$

which penalises violations of subclass containment. Intuitively, the loss measures how far the child box lies outside the parent box in each dimension. When the child box is fully contained within the parent box, the loss is zero; otherwise it increases in proportion to the extent of the containment violation. Kronstr¨om et al. (2026) also used a negative loss term to penalise disjoint entities. In this work we omit this term since the taxonomies considered lack such axioms and the link prediction task seems to provide separation between entities. This is further discussed in Appendix A. The overall semantic loss, denoted $\mathcal { L } _ { \subseteq }$ , is obtained by summing $\mathcal { L } _ { \subseteq } ( C , D )$ over all subclass axioms in the hierarchy.

At each GNN layer, node embeddings are transformed into box representations using (2). The semantic inclusion loss is then evaluated over the ontology hierarchy to encourage consistency with the ontology-derived subclass relations. The framework is architectureagnostic and can be combined with arbitrary downstream objectives. While Kronstr¨om et al. (2026) combined the semantic loss with an MSE loss for regression, in this work we jointly optimise the binary cross-entropy loss for link prediction and the semantic loss:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \alpha \mathcal { L } _ { \stackrel { } { = } } , } \end{array}\tag{5}
$$

where α controls the contribution of the semantic loss. Because $\mathcal { L } _ { \subseteq }$ is jointly optimised with the link prediction objective, the hierarchy constraints are encouraged rather than guaranteed to hold exactly.

This formulation allows the encoder to exploit both local graph structure and global ontological information during representation learning, while remaining compatible with standard end-to-end optimisation. The decoder is simultaneously optimised for the downstream link prediction task.

## 3. Experimental setup

## 3.1. Datasets and hierarchy construction

We evaluate the proposed hierarchy-aware semantic losses on three benchmark knowledge graph datasets: AIFB (Bloehdorn and Sure, 2007), CoDEx (Safavi and Koutra, 2020), and BioKG (Walsh et al., 2020). The datasets vary substantially in scale, domain, and the availability of hierarchical information, allowing us to investigate the efect of semantic constraints under diferent conditions. Details about the datasets can be found in Tables 1 and 2.

AIFB A semantic web benchmark containing entities such as persons, projects, organizations, and publications. Since the dataset is derived from a single ontology, class hierarchy information is readily available through explicit subClassOf relations, which are directly used in our experiments. For link prediction, we consider only the publication relation linking persons to publications.

Table 1: Overview of dataset and hierarchy characteristics. |E| and |R| denote the number of entities and relations in the original KG. Hierarchy entities and subclass axioms denote the number of additional entities and subclass relations introduced by the hierarchy. Coverage indicates the percentage of KG entities represented in the hierarchy. Domain-specific hierarchy characteristics for BioKG are reported in Table 2.
<table><tr><td>Dataset</td><td>|ε|</td><td>|R|</td><td>Train triples</td><td>Val/Test triples</td><td>Hierarchy entities</td><td>Subclass axioms</td><td>Coverage (%)</td><td>Max depth</td><td>Median depth</td></tr><tr><td>AIFB</td><td>2,431</td><td>36</td><td>30,102</td><td>416</td><td>172</td><td>3,890</td><td>100.0</td><td>5</td><td>2</td></tr><tr><td>CoDEx</td><td>77,951</td><td>69</td><td>551,193</td><td>30,622</td><td>1,986</td><td>69,974</td><td>80.2</td><td>22</td><td>10</td></tr><tr><td>BioKG</td><td>93,773</td><td>51</td><td>4,762,678</td><td>162,870</td><td>64,985</td><td>455,560</td><td>84.7</td><td>58</td><td>6</td></tr></table>

Table 2: Hierarchy characteristics for the five domains in BioKG.
<table><tr><td>Domain</td><td>|ε|</td><td>Hierarchy entities</td><td>Subclass axioms</td><td>Coverage (%)</td><td>Max depth</td><td>Median leaf node depth</td></tr><tr><td>function</td><td>45,085</td><td>618</td><td>93,040</td><td>100.0</td><td>16</td><td>6</td></tr><tr><td>disease</td><td>10,687</td><td>25,804</td><td>163,606</td><td>83.4</td><td>58</td><td>24</td></tr><tr><td>sideeffect</td><td>9,969</td><td>28,217</td><td>166,898</td><td>91.2</td><td>57</td><td>16</td></tr><tr><td>protein</td><td>17,499</td><td>2,040</td><td>21,906</td><td>82.9</td><td>6</td><td>2</td></tr><tr><td>drug</td><td>10,533</td><td>8,306</td><td>10,110</td><td>17.1</td><td>11</td><td>6</td></tr></table>

CoDEx A multi-domain knowledge graph completion benchmark derived from Wikidata and Wikipedia, covering entities from domains such as writing, entertainment, academia, and science. We use CoDEx-L, the largest variant of the benchmark. Since the dataset does not provide explicit hierarchy information, we construct a class hierarchy using the instanceOf (P31) and subClassOf (P279) properties extracted from Wikidata5M (Wang et al., 2021).

BioKG A heterogeneous biomedical knowledge graph containing entities from five domains: functions, diseases, side-efects, proteins, and drugs. Each domain is associated with a domain-specific class hierarchy. Function entities are directly described by the Gene Ontology (GO) (Ashburner et al., 2000). The protein hierarchy was derived from the HGNC gene family taxonomy (Gray et al., 2016), where proteins were assigned to gene families via Entrez Gene identifiers and connected through hierarchical family relations. Drug entities were mapped to the ChemOnt chemical taxonomy using structural classifications obtained through the ClassyFire API (Djoumbou Feunang et al., 2016). Disease and side-efect hierarchies were derived from the UMLS Metathesaurus (Bodenreider, 2004), using hierarchical relations among concepts originating from multiple biomedical vocabularies, primarily SNOMED CT and MeSH.

BioKG is particularly heterogeneous, both in terms of relation types and entity domains. The graph contains interactions between functions, diseases, side-efects, proteins, and drugs, each associated with its own ontology-derived hierarchy. This provides a useful setting for evaluating whether hierarchy-aware semantic losses can exploit multiple sources of ontological information within a single knowledge graph.

## 3.2. Models

For the GNN encoder we consider both R-GCN (Schlichtkrull et al., 2018) and GraphSAGE (Hamilton et al., 2017) architectures. Preliminary experiments indicated that R-GCN performed best on AIFB, while GraphSAGE was more efective on CoDEx and BioKG. We therefore report results using the strongest encoder for each dataset. This diference is consistent with the contrasting characteristics of the datasets: AIFB is comparatively small and contains a limited number of relation types, whereas CoDEx and BioKG are substantially larger and more heterogeneous.

For BioKG, the graph is represented as a heterogeneous graph with separate node and edge types for each domain and relation. Message passing is implemented using a HeteroConv architecture, where relation-specific GraphSAGE convolutions are applied independently and aggregated by mean pooling. Hierarchy-aware semantic losses are applied separately to the ontology-derived embedding spaces associated with each domain.

The output node embeddings generated by the encoder are provided to a link prediction decoder. We evaluate two decoder types:

• ComplEx (Trouillon et al., 2016): a bilinear scoring function commonly used for knowledge graph completion.

• Neural network (NN): a small multilayer perceptron operating on concatenated head and tail embeddings. Separate network parameters are learned for each relation type, providing relation-specific decoding while sharing the underlying node representations.

Preliminary experiments showed that the ComplEx decoder performed best on AIFB, whereas the neural network decoder was more efective on CoDEx and BioKG. This mirrors the encoder selection: the comparatively small AIFB dataset appears to benefit from the additional inductive bias provided by the structured bilinear scoring function, while the larger datasets benefit from the greater flexibility of the neural network decoder.

To assess the efect of incorporating hierarchy information, we compare three configurations:

• Baseline: original graph without hierarchy information.

• SC-Edges: subClassOf relations added directly as graph edges.

• Semantic loss: subClassOf relations incorporated through hierarchy-aware semantic losses introduced in Section 2.

Unlike SC-Edges, the semantic loss approach preserves the original graph topology and instead constrains the learned embedding space. We additionally compare against a conventional ComplEx model without a GNN encoder.

## 3.3. Training and evaluation

CoDEx and BioKG use the standard train, validation, and test splits provided with the benchmarks. Since AIFB does not provide a link prediction split for the publication relation, we randomly partition the corresponding triples into training, validation, and test sets using an 80/10/10 ratio.

Models are trained using binary cross-entropy (BCE) loss for link prediction, with randomly sampled negative examples. For hierarchy-aware models, the semantic loss in (5) is jointly optimised with the task-specific loss. Evaluation is based on MRR. For AIFB and CoDEx, each positive validation and test triple is ranked against 1,000 randomly sampled negative triples generated by corrupting the head or tail entity, while BioKG uses the benchmark-provided evaluation negatives.

## 4. Results

We evaluate hierarchy-aware semantic losses on AIFB, CoDEx, and BioKG. The reported results are based on 10 training runs. We compare standard link prediction models, models incorporating subclass relations as graph edges (SC-Edges), and hierarchy-aware semantic losses. Full hyperparameter settings are provided in Appendix B.

Table 3: Link prediction performance (MRR) and number of trainable parameters. Results are reported as mean ± standard deviation over 10 runs. All models trained with the semantic loss significantly outperform the corresponding baseline models (Welch’s two-sided t-test, p < 0.05).
<table><tr><td></td><td></td><td colspan="2">ComplEx</td><td colspan="3">R-GCN / GraphSAGE</td></tr><tr><td>Dataset</td><td>Metric</td><td>Baseline</td><td>SC-Edges</td><td>Baseline</td><td>SC-Edges</td><td>SemLoss</td></tr><tr><td rowspan="3">AIFB</td><td>MRR</td><td>0.4056</td><td>0.4336</td><td>0.4758</td><td>0.4872</td><td>0.5118</td></tr><tr><td></td><td>±0.0085</td><td>±0.0062</td><td>±0.0237</td><td>±0.0257</td><td>±0.0151</td></tr><tr><td>#Params (M)</td><td>1.32</td><td>1.35</td><td>0.51</td><td>0.51</td><td>0.51</td></tr><tr><td rowspan="3">CoDEx</td><td>MRR</td><td>0.3734</td><td>0.3700</td><td>0.3999</td><td>0.4035</td><td>0.4095</td></tr><tr><td></td><td>±0.0010</td><td>±0.0014</td><td>±0.0069</td><td>±0.0074</td><td>±0.0025</td></tr><tr><td>#Params (M)</td><td>39.95</td><td>40.96</td><td>6.63</td><td>6.74</td><td>6.69</td></tr><tr><td rowspan="3">BioKG</td><td>MRR</td><td>0.7440</td><td>0.6963</td><td>0.7232</td><td>0.7644</td><td>0.8354</td></tr><tr><td></td><td>±0.0044</td><td>±0.0223</td><td>±0.0785</td><td>±0.0393</td><td>±0.0144</td></tr><tr><td>#Params (M)</td><td>96.08</td><td>181.05</td><td>42.30</td><td>53.41</td><td>50.62</td></tr></table>

Table 3 and Table 5 in Appendix C summarise the results on the three benchmark datasets. Across all datasets, the semantic loss models achieved the strongest performance and significantly outperformed both the corresponding baseline and SC-Edges variants (Welch’s two-sided t-test, $p < 0 . 0 5 )$ . The largest improvement was observed on BioKG, where semantic losses increased MRR from 0.723 to 0.835, corresponding to a relative improvement of approximately 15%. Improvements were also observed on AIFB and CoDEx, with relative gains of 7.6% and 2.4%, respectively.

The behaviour of the ComplEx baseline on BioKG is also noteworthy. The absolute performance is somewhat lower than reported in leaderboard evaluations, likely reflecting the smaller model size used here. In addition, augmenting the graph with subclass relations decreases ComplEx performance. This may be due to the large number of hierarchy entities introduced into the graph, which participate primarily in taxonomic relations and substantially increase graph size without directly contributing to the link prediction objective.

As shown in Table 3, the semantic-loss models generally require fewer parameters than the corresponding SC-Edges variants, since hierarchy information is incorporated through the loss function rather than entirely through graph augmentation. While representing hierarchy entities increases model size, the resulting parameter overhead remains smaller than that of SC-Edges and depends on the size of the hierarchy. The increase is negligible for AIFB but more substantial for BioKG, which incorporates several large biomedical ontologies and therefore many more hierarchy entities. Despite using fewer parameters, semantic losses consistently achieved stronger performance than SC-Edges, most notably on BioKG where MRR increases from 0.764 to 0.835 while reducing the parameter count from 53.41M to 50.62M.

The strongest gains were obtained on BioKG, which contains extensive ontology-derived hierarchies spanning multiple biomedical domains. In contrast, the improvements on CoDEx were more modest. This suggests that the efectiveness of semantic regularisation depends on characteristics of the available hierarchy information, an observation we investigate further in the following analyses.

![](images/e6b439efd6c0df5947f735aa89b1245f39b3ef1c27a9bfcb454867e61aa8e9ae.jpg)

![](images/b0a9054d1ef6f410702cea6d82dadb0455dbe10c05a5ca7655392fe075293cd9.jpg)

![](images/47b90b2981a7a52037692b6544f977d8fa0fa1ad53ac46cb122e0919a54d9830.jpg)

![](images/573b36e1b056b540f7aeda6a3bc927f5517065489f1c567ba3883073da9d92a4.jpg)

![](images/68d32100d3ef8072e07cc56b47a134ddd2d6a656060e066b4d6f4f4a4b839078.jpg)

![](images/5086f12a108487163718bdb3381ad75c1f6f575b3d87d7a7d8b59217e69c597f.jpg)  
Figure 2: Evolution of domain-specific semantic losses during training on BioKG. Curves show the median semantic loss over 10 training runs and shaded regions indicate the interquartile range. Losses are reported separately for each ontology domain and GNN layer. The semantic losses decrease throughout training for all domains, indicating increasing satisfaction of ontology-derived hierarchy constraints, although the magnitude of improvement varies between domains. Layer 0 corresponds to the initial node embeddings prior to message passing.

To investigate how the hierarchy constraints are incorporated during training, Figure 2 shows the evolution of the semantic losses for the five BioKG ontology domains. The losses decrease consistently throughout training, indicating increasing satisfaction of the ontologyderived hierarchy constraints, although complete satisfaction is not guaranteed because the semantic loss is optimised jointly with the link prediction objective. Reductions are observed across all domains and GNN layers, with the largest decreases occurring for the function, protein, and drug hierarchies. In contrast, the side-efect hierarchy exhibits greater variability and smaller relative improvements. The abrupt reductions observed around epochs 20, 100, and 200 coincide with the scheduled learning-rate decreases described in Appendix B.

To further investigate the difering improvements across datasets, Figure 3 compares performance gains with two descriptive hierarchy characteristics. We define hierarchy density as the number of subclass axioms per hierarchy entity and hierarchy availability as the number of hierarchy entities per hierarchy-covered graph entity. Together, these metrics characterise both the local connectivity of the hierarchy and the amount of hierarchy structure available to constrain the learned embeddings. Although only three datasets are available, the figure suggests that larger performance gains are associated with greater hierarchy availability, whereas hierarchy density exhibits the opposite trend.

![](images/2739f15b074a179b3b480cd934c1f69854bfcfc2c8860706cb88cfe10b72ab93.jpg)  
Figure 3: Exploratory comparison between performance improvements obtained by hierarchy-aware semantic losses and hierarchy characteristics across the three evaluated datasets. The x-axis shows the improvement in mean reciprocal rank (∆MRR) relative to the corresponding baseline model. Hierarchy density is defined as the number of subclass axioms divided by the number of hierarchy entities. Hierarchy availability is defined as the number of hierarchy entities divided by the number of hierarchy-covered graph entities and reflects the amount of hierarchy information available per covered graph entity.

## 5. Discussion

The results demonstrate that hierarchy-aware semantic losses can improve link prediction performance across a range of knowledge graph domains and scales. Across all three benchmark datasets, semantic losses significantly outperformed both the corresponding baseline models and variants incorporating subclass relations directly as graph edges (SC-Edges). Although the semantic-loss models require additional parameters to represent hierarchy entities, they remain more parameter-eficient than the corresponding SC-Edges variants while achieving stronger predictive performance.

The comparison with SC-Edges is particularly informative because adding subclass relations as additional message-passing edges is a common strategy for incorporating ontology information into graph neural networks. The consistent advantage of semantic losses suggests that explicitly constraining the embedding space to satisfy hierarchical relations may be more efective than relying on message passing alone to propagate taxonomic information. Moreover, semantic losses preserve the original graph topology and avoid introducing large numbers of hierarchy edges into the graph.

Figure 2 provides additional insight into how the hierarchy-aware semantic losses evolve during training. Across all five BioKG domains, the semantic losses decrease substantially during training, indicating that the learned embeddings increasingly satisfy the ontologyderived hierarchy constraints. Although reductions are observed across all domains, the side-efect hierarchy exhibits both smaller decreases and lower initial loss values. This suggests that its constraints are already comparatively well satisfied before training, leaving less room for further optimisation.

The magnitude of the improvements varied substantially across datasets. Figure 3 provides an exploratory comparison between performance gains and selected hierarchy characteristics. Although only three datasets are available, the observed trends suggest that performance gains are associated with both higher hierarchy availability and lower hierarchy density. In particular, BioKG exhibits substantially higher hierarchy availability than AIFB and CoDEx while also achieving the largest performance improvements. Conversely, hierarchy density decreases with increasing gains, suggesting that local hierarchy connectivity is not the sole determinant of performance gains.

However, these metrics provide only a simplified view of the hierarchies. The usefulness of hierarchy-derived supervision is likely to depend strongly on how the hierarchies are constructed and how well they capture meaningful semantic relationships. For example, the AIFB hierarchy is derived directly from an ontology, while several BioKG hierarchies originate from curated biomedical ontologies such as the Gene Ontology. In contrast, the CoDEx hierarchy was derived from Wikidata, whose category structure was not primarily designed for ontological reasoning and may therefore provide weaker semantic constraints. Consequently, hierarchy density and hierarchy availability should be viewed as coarse descriptive measures rather than direct indicators of hierarchy quality. Nevertheless, the substantially larger gains observed on BioKG suggest that both the quantity and quality of available hierarchical information influence the efectiveness of hierarchy-aware semantic losses.

Only three datasets were considered in this study, making it dificult to draw strong conclusions regarding which hierarchy characteristics are most important. Future work should evaluate hierarchy-aware semantic losses across a broader range of knowledge graphs and ontology sources, enabling a more systematic investigation of the relationship between hierarchy structure, hierarchy quality, and downstream performance.

More generally, these results indicate that class hierarchies provide complementary information to the relational structure captured by graph neural networks. By encouraging representations to better satisfy ontology-derived hierarchy constraints throughout the encoder, hierarchy-aware semantic losses promote representations that reflect both graph connectivity and conceptual organisation more faithfully. The consistent improvements observed across all datasets suggest that this principle generalises beyond the biological regression task originally studied by Kronstr¨om et al. (2026) and can be applied efectively to knowledge graph link prediction.

## Availability

Code for generating hierarchies and training models can be found at https://github.com/ filipkro/hgnn-lp

## Acknowledgments

The computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725. This work was supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Alice Wallenberg Foundation, the UK Engineering and Physical Sciences Research Council (EPSRC) [EP/R022925/2, EP/W004801/1 and EP/X032418/1], and the Chalmers AI Research Centre.

## References

Michael Ashburner, Catherine A. Ball, Judith A. Blake, David Botstein, Heather Butler, J. Michael Cherry, Allan P. Davis, Kara Dolinski, Selina S. Dwight, Janan T. Eppig, Midori A. Harris, David P. Hill, Laurie Issel-Tarver, Andrew Kasarskis, Suzanna Lewis, John C. Matese, Joel E. Richardson, Martin Ringwald, Gerald M. Rubin, and Gavin Sherlock. Gene ontology: tool for the unification of biology. Nature Genetics, 25(1): 25–29, 2000. ISSN 1546-1718. doi: 10.1038/75556. Number: 1.

Stephan Bloehdorn and York Sure. Kernel methods for mining instance data in ontologies. In Karl Aberer, Key-Sun Choi, Natasha Noy, Dean Allemang, Kyung-Il Lee, Lyndon Nixon, Jennifer Golbeck, Peter Mika, Diana Maynard, Riichiro Mizoguchi, Guus Schreiber, and Philippe Cudr´e-Mauroux, editors, The Semantic Web, pages 58–71. Springer, 2007. ISBN 978-3-540-76298-0. doi: 10.1007/978-3-540-76298-0 5.

Olivier Bodenreider. The unified medical language system (UMLS): integrating biomedical terminology. Nucleic Acids Research, 32:D267–D270, 2004. ISSN 0305-1048. doi: 10. 1093/nar/gkh061.

Antoine Bordes, Nicolas Usunier, Alberto Garcia-Dur´an, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2, volume 2 of NIPS’13, pages 2787–2795. Curran Associates Inc., 2013.

Tejas Chheda, Purujit Goyal, Trang Tran, Dhruvesh Patel, Michael Boratko, Shib Sankar Dasgupta, and Andrew McCallum. Box embeddings: An open-source library for representation learning using geometric structures, 2021.

Yannick Djoumbou Feunang, Roman Eisner, Craig Knox, Leonid Chepelev, Janna Hastings, Gareth Owen, Eoin Fahy, Christoph Steinbeck, Shankar Subramanian, Evan Bolton, Russell Greiner, and David S. Wishart. ClassyFire: automated chemical classification with a comprehensive, computable taxonomy. Journal of Cheminformatics, 8(1):61, 2016. ISSN 1758-2946. doi: 10.1186/s13321-016-0174-y.

Kristian A Gray, Ruth L Seal, Susan Tweedie, Mathew W Wright, and Elspeth A Bruford. A review of the new HGNC gene family resource. Human Genomics, 10:6, 2016. ISSN 1473-9542. doi: 10.1186/s40246-016-0062-6.

William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 1025–1035. Curran Associates Inc., 2017. ISBN 978- 1-5108-6096-4.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: datasets for machine learning on graphs. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, pages 22118–22133. Curran Associates Inc., 2020. ISBN 978-1-7138-2954-6.

Mathias Jackermeier, Jiaoyan Chen, and Ian Horrocks. Dual box embeddings for the description logic EL++. In Proceedings of the ACM Web Conference 2024, WWW ’24, pages 2250–2258. Association for Computing Machinery, 2024. ISBN 979-8-4007-0171-9. doi: 10.1145/3589334.3645648.

Filip Kronstr¨om, Alexander H. Gower, Daniel Brunns˚aker, Ievgeniia A. Tiukova, and Ross D. King. Graph neural network based hierarchy-aware embeddings of knowledge graphs: Applications to yeast phenotype prediction. In arXiv, 2026. doi: 10.48550/arXiv.2605.03690.

Maxat Kulmanov, Wang Liu-Wei, Yuan Yan, and Robert Hoehndorf. EL embeddings: Geometric construction of models for the description logic EL++. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pages 6103– 6109. International Joint Conferences on Artificial Intelligence Organization, 2019. ISBN 978-0-9992411-4-1. doi: 10.24963/ijcai.2019/845.

Maximilian Nickel and Douwe Kiela. Poincar´e embeddings for learning hierarchical representations. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 6341–6350. Curran Associates Inc., 2017. ISBN 978- 1-5108-6096-4.

Xi Peng, Zhenwei Tang, Maxat Kulmanov, Kexin Niu, and Robert Hoehndorf. Description logic EL++ embeddings with intersectional closure, 2022.

Tara Safavi and Danai Koutra. CoDEx: A comprehensive knowledge graph completion benchmark. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8328–8350. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-main.669.

Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In Aldo Gangemi, Roberto Navigli, Maria-Esther Vidal, Pascal Hitzler, Rapha¨el Troncy, Laura Hollink, Anna Tordai, and Mehwish Alam, editors, The Semantic Web, pages 593– 607. Springer International Publishing, 2018. ISBN 978-3-319-93417-4. doi: 10.1007/ 978-3-319-93417-4 38.

Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. RotatE: Knowledge graph embedding by relational rotation in complex space. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019.

Th´eo Trouillon, Johannes Welbl, Sebastian Riedel, Eric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In Proceedings of The 33rd International Conference on Machine Learning, pages 2071–2080. PMLR, 2016.

Luke Vilnis, Xiang Li, Shikhar Murty, and Andrew McCallum. Probabilistic embedding of knowledge graphs with box lattice measures. In Iryna Gurevych and Yusuke Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 263–272. Association for Computational Linguistics, 2018. doi: 10.18653/v1/P18-1025.

Brian Walsh, Sameh K. Mohamed, and V´ıt Nov´aˇcek. BioKG: A knowledge graph for relational learning on biological data. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, CIKM ’20, pages 3173–3180. Association for Computing Machinery, 2020. ISBN 978-1-4503-6859-9. doi: 10.1145/3340531.3412776.

Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. KEPLER: A unified model for knowledge embedding and pre-trained language representation. Transactions of the Association for Computational Linguistics, 9:176–194, 2021. doi: 10.1162/tacl a 00360.

Bo Xiong, Nico Potyka, Trung-Kien Tran, Mojtaba Nayyeri, and Stefen Staab. Faithful embeddings for EL++ knowledge bases. In The Semantic Web – ISWC 2022: 21st International Semantic Web Conference, Virtual Event, October 23–27, 2022, Proceedings, pages 22–38. Springer-Verlag, 2022. ISBN 978-3-031-19432-0. doi: 10.1007/ 978-3-031-19433-7 2.

Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy Broeck. A semantic loss function for deep learning with symbolic knowledge. In Proceedings of the 35th International Conference on Machine Learning, pages 5502–5511. PMLR, 2018.

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http: //arxiv.org/abs/1412.6575.

Zhanqiu Zhang, Jianyu Cai, Yongdong Zhang, and Jie Wang. Learning hierarchy-aware knowledge graph embeddings for link prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 34(3):3065–3072, 2020. ISSN 2374-3468, 2159-5399. doi: 10.1609/ aaai.v34i03.5701.

## Appendix A. Negative loss

In addition to the positive semantic inclusion loss, Kronstr¨om et al. (2026) employed a negative semantic loss to model concepts that should be kept separate in the embedding space. The primary motivation for this loss is to represent disjointness axioms in ontologies. A secondary benefit is that it discourages degenerate solutions in which many concepts collapse into identical or highly overlapping boxes while still satisfying the positive hierarchy constraints.

The negative loss can be applied either to explicitly disjoint concepts or to randomly sampled negative examples:

$$
\mathcal { L } _ { \Xi } ^ { - } ( C , D ) = \Big | \Big | \Big ( \operatorname* { m a x } ( 0 , - d ( C _ { i } , D _ { i } ) ) \Big ) _ { i = 1 } ^ { n } \Big | \Big | \cdot \prod _ { i = 1 } ^ { n } \mathbb { I } \big [ d ( C _ { i } , D _ { i } ) < 0 \big ] .\tag{6}
$$

Intuitively, the loss penalises overlap between boxes corresponding to concepts that are assumed to be distinct. Combined with the task-specific and positive semantic loss, the overall objective becomes

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { T A S K } } ( y , \hat { y } ) + \alpha ( \mathcal { L } _ { \Xi } + \beta \mathcal { L } _ { \Xi } ^ { - } ) ,\tag{7}
$$

where $\beta$ controls the contribution of the negative examples in relation to the positive hierarchy examples.

In the datasets considered in this work, no explicit disjointness axioms are available. Consequently, applying a negative semantic loss would require generating artificial negative examples by randomly sampling pairs of classes. Unlike true disjointness axioms, however, such sampled pairs are not guaranteed to be semantically unrelated. The negative loss may therefore penalise embeddings that correctly place related concepts close together in the latent space, introducing constraints that are not supported by the underlying ontology.

Furthermore, the link prediction objective already provides a strong discriminative signal. To predict relations accurately, the model must learn to distinguish between entities and classes that participate in diferent parts of the graph, which may reduce the risk of the embedding collapse that motivated the negative loss in the original work. In this setting, the negative loss may provide limited additional information while simultaneously introducing potentially incorrect constraints.

Empirically, preliminary experiments showed no benefit from including the negative loss, and in several cases performance was slightly improved when it was omitted. For these reasons, all experiments reported in this paper use only the positive semantic inclusion loss.

## Appendix B. Hyperparameters

Hyperparameters were selected based on validation performance. All models were trained using the Adam optimiser with an L2 regularisation weight of $1 0 ^ { - 3 }$ . Models incorporating semantic loss used a semantic loss weight of $\alpha = 1 0 ^ { - 1 }$

AIFB was trained for 120 epochs using a learning rate of $1 0 ^ { - 3 }$ . CoDEx and BioKG were trained for 150 and 250 epochs, respectively. For these datasets, the initial learning rate was set to $5 \times 1 0 ^ { - 2 }$ and reduced by a factor of two at epochs 10, 40, and 75 for CoDEx, and at epochs 20, 100, and 200 for BioKG. For each run, the model achieving the best validation performance was retained for evaluation on the test set.

Negative examples for link prediction training were generated by uniformly sampling random triples rather than by corrupting heads or tails of positive triples. AIFB used a negative-to-positive ratio of 1000:1 with a positive class weight of 500. CoDEx used a negative-to-positive ratio of 150:1 with a positive class weight of 250, while BioKG used a negative-to-positive ratio of 100:1 with a positive class weight of 200. These settings were selected based on validation performance.

Dataset-specific architectural hyperparameters are summarised in Table 4.

Table 4: Hyperparameters for the diferent datasets.
<table><tr><td>Dataset</td><td>Embedding dimension</td><td>Hidden dimensions</td><td>R-GCN bases</td><td>NN decoder units</td></tr><tr><td>AIFB</td><td>32</td><td>[64, 64, 64]</td><td>40</td><td></td></tr><tr><td>CoDEx</td><td>32</td><td>[64, 128]</td><td></td><td>[32, 8, 1]</td></tr><tr><td>BioKG</td><td>128</td><td>[256, 256]</td><td></td><td>[64, 32, 8, 1]</td></tr></table>

The baseline ComplEx models were trained using the Adam optimiser with a learning rate of $1 0 ^ { - 3 }$ . Model selection was performed using early stopping based on validation performance with a patience of 20 epochs. Following preliminary hyperparameter tuning, embedding dimensions of 256 were used for AIFB and CoDEx, while BioKG used an embedding dimension of 512.

## Appendix C. Hits@K

Table 5: Link prediction performance (Hits@K). Results are reported as mean ± standard deviation over 10 runs. § indicates significant improvement for the model trained with the semantic loss over all baseline models. † and ‡ indicate significant improvements compared to the ComplEx and ComplEx with subClassOf-edges respectively. \* indicates significant improvement compared to the GNN baseline. Improvements are tested with Welch’s two-sided t-test $\left( p < 0 . 0 5 \right)$
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="2">ComplEx</td><td colspan="3">R-GCN/ GraphSAGE</td></tr><tr><td>Baseline</td><td>SC-Edges</td><td>Baseline</td><td>SC-Edges</td><td>SemLoss</td></tr><tr><td rowspan="4">AIFB</td><td>Hits@1</td><td>0.2389</td><td>0.2764</td><td>0.3478</td><td>0.3565</td><td>0.38618</td></tr><tr><td rowspan="2">Hits@3</td><td>±0.0156</td><td>±0.0113</td><td>±0.0227</td><td>±0.0289</td><td>±0.0157</td></tr><tr><td>0.5041 ±0.0103</td><td>0.5303 ±0.0106</td><td>0.5334 ±0.0311</td><td>0.5454 ±0.0275</td><td>0.5671*†‡ ±0.0190</td></tr><tr><td>Hits@10</td><td>0.7327 ±0.0113</td><td>0.7334</td><td>0.7409</td><td>0.7558</td><td>0.7687*†$</td></tr><tr><td rowspan="6">CoDEx</td><td>Hits@1</td><td>0.3020</td><td>±0.0042 0.2971</td><td>±0.0306</td><td>±0.0289</td><td>±0.0127 0.33808</td></tr><tr><td></td><td>±0.0019</td><td>±0.0020</td><td>0.3266 ±0.0078</td><td>0.3310 ±0.0082</td><td>±0.0030</td></tr><tr><td rowspan="2">Hits@3</td><td>0.4111</td><td>0.4085</td><td>0.4423</td><td>0.4448</td><td>0.4478†‡</td></tr><tr><td>±0.0008</td><td>±0.0016</td><td>±0.0066</td><td>±0.0070</td><td>±0.0045</td></tr><tr><td rowspan="2">Hits@10</td><td>0.4914</td><td>0.4895</td><td>0.5268</td><td>0.5274</td><td>0.5336*†‡</td></tr><tr><td>±0.0011</td><td>±0.0027</td><td>±0.0064</td><td>±0.0079</td><td>±0.0050</td></tr><tr><td rowspan="5">BioKG</td><td>Hits@1</td><td>0.6125</td><td>0.5563</td><td>0.6059</td><td>0.6349</td><td>0.71578</td></tr><tr><td rowspan="2">Hits@3</td><td>±0.0060</td><td>±0.0227</td><td>±0.0685</td><td>±0.0524</td><td>±0.0302</td></tr><tr><td>0.8556</td><td>0.8187</td><td>0.8253</td><td>0.8768</td><td>0.92408</td></tr><tr><td rowspan="2"></td><td>±0.0036</td><td>±0.0238</td><td>±0.0904</td><td>±0.0348</td><td>±0.0184</td></tr><tr><td>0.9603</td><td>0.9167</td><td>0.8858</td><td>0.9512</td><td>0.9622‡</td></tr><tr><td rowspan="4"></td><td>Hits@10</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>±0.0017</td><td>±0.0222</td><td>±0.1062</td><td>±0.0240</td><td>±0.0239</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>