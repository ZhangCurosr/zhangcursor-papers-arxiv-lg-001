# A Drosophila Whole-Connectome Network Can Learn Human-Designed Cognitive Tasks

Joonghui Cho Minchan Kang Daeshik Kim Korea Advanced Institute of Science and Technology (KAIST) paul26@kaist.ac.kr mc.kang@kaist.ac.kr daeshik@kaist.ac.kr <sup>∗</sup>Corresponding author.

## Abstract

Can a biological wiring diagram serve as a useful computational substrate beyond the behaviors for which it evolved? We use the publicly released MaleCNS v1.0 connectome, reconstructed from a single adult male Drosophila specimen, as the fixed recurrent topology of an artificial network. We train separate models for bounded addition and for a controlled grounded relational language task built from a fixed 100-word lexicon. In both models, one scalar is learned per anatomical edge. The anatomical graph reaches 92.77% mean accuracy on held-out addition, compared with 67.93% for directed degree-preserving rewires. On the strict paired language endpoint, which matches original and order-reversed scenes to their corresponding descriptions, it reaches 61.59% across four fixed interfaces, compared with 44.17% for matched rewires. At the canonical interface, it ranks first in a fixed 21-graph comparison. On the matched 48-group intervention subset, shufling task-defined sensory features reduces its score from 60.94% to 19.27%. Together, these results show that higher-order MaleCNS wiring provides a reusable inductive bias for bounded addition and grounded relational language.

## 1 Introduction

Connectomes are anatomical records, but they can also be read as candidate architectures. Synapse-resolution reconstructions now cover nearly the entire adult fly brain and the complete adult male central nervous system (CNS), revealing recurrent organizations shaped by an animal’s behavior (Dorkenwald et al., 2024; Berg et al., 2026). We ask whether that organization remains useful when connection strengths are relearned for problems outside the animal’s natural repertoire.

Most connectome-constrained models have pursued biological prediction in visual and sensorimotor systems, causal-efect estimation, or whole-brain activity fitting (Lappalainen et al., 2024; Shiu et al., 2024; Pospisil et al., 2024; Wang et al., 2026). Work on artificial tasks is smaller and has adapted fly wiring to image classification, chess, simulated locomotion, and robot navigation (Yu et al., 2026; Jin et al., 2026; Wang and Chen, 2026). Evidence about the topology itself is mixed. Random structures performed similarly to biological ones in one comparison, and another study found that shared initialization and degree-preserving nulls removed much of an apparent advantage (Pircher et al., 2021; Dhiman, 2026). Task performance by itself does not show whether the biological arrangement of edges matters.

We use the MaleCNS adjacency as the recurrent mask of an artificial network, with one learned scalar on each retained edge. Addition and grounded relational language are trained as separate models. Addition requires exact answers for unseen operand pairs. The language model receives controlled English alongside task-defined sensory events and must distinguish descriptions of held-out channel pairs and reversed event sequences. Orthogonal token codes contain no pretrained lexical semantics, so the mapping must be acquired from the generated corpus and sensory stream.

The anatomical graph outperforms degree-preserving rewires on both tasks, with the language separation appearing at every tested interface and in an additional 20-rewire ensemble. Sensory shufling sharply reduces language performance. Figure 1 introduces the graph, tasks, and main controls.

## 2 Related Work

Whole-connectome resources and models. Fly-Wire reconstructed a whole adult female brain with 139,255 neurons and about 50 million chemical synapses (Dorkenwald et al., 2024). MaleCNS extends synapseresolution reconstruction across the complete adult male CNS (Berg et al., 2026). Whole-brain analyses identify rich-club organization and overrepresented recurrent motifs (Lin et al., 2024). Functional models use this anatomical detail to predict visual activity (Lappalainen et al., 2024), model feeding and grooming transformations (Shiu et al., 2024), and formulate estimators for causal neural efects (Pospisil et al., 2024). BrainTrace, for example, applied its online learning method to a FlyWire-constrained spiking model of resting-state regional activity and interregional functional connectivity (Wang et al., 2026).

![](images/907d18f3484c943bddaa97d6a6e35e6f7c68cfc1874f76ed11ebc098fb0a4c9c.jpg)  
Figure 1: Study design. (a) The processed MaleCNS v1.0 adjacency fixes the recurrent support, and each retained directed edge carries one learned scalar. The anatomical rendering shows a deterministic sample of 3,000 neuron skeletons. The overlaid nodes and arrows are schematic. Directed-degree-preserving rewires retain each node’s in- and out-degree. (b) Addition and grounded relational language are separate fits on the same topology. Language receives sensory events and fixed token codes and predicts the next symbol through a fixed readout. Held-out evaluation pairs original and order-reversed scenes with their corresponding equal-length descriptions. Orange marks the sensory-shufle control used only at evaluation.

Topology as an inductive bias. Connectomeconstrained reservoir studies have examined time-series prediction and multifunctional dynamics across hyperparameter regimes (Morra et al., 2025). In small fly-connectome subgraphs, raw task performance was generally comparable to random reservoirs, but weightcost-normalized performance and robustness to neuron loss or parameter variation were higher (McAllister et al., 2026). Simplified whole-brain communication models likewise produced simulated region-selective activation patterns that were sensitive to degree-preserving rewiring (Zhang et al., 2026). A recent fly-visual-system study found that advantages over a weak random control largely disappeared with shared from-scratch initialization and degree-preserving nulls (Dhiman, 2026).

Artificial-task studies use a complete larval connectome as a reservoir for image classification and chess (Yu et al., 2026), an adult-brain connectome as a learned controller for simulated fly locomotion (Jin et al., 2026), and a fly-brain-derived recurrent network for robot navigation (Wang and Chen, 2026). FlyGM also compares its locomotion controller with a degree-preserving rewire. The present experiments move away from embodied behavior to two symbolic mappings and ask whether their performance survives changes in token-input and output placement and graph draw.

Grounding, composition, and arithmetic. Flyinspired language work has used a mathematical mushroom-body motif to learn sparse word representations from text (Liang et al., 2021). Associations between words and referents can also be learned from paired egocentric visual and linguistic experience (Vong et al., 2024), and pretrained instruction embeddings can support compositional generalization in sensorimotor networks (Riveland and Pouget, 2024). Benchmarks such as gSCAN, COGS, ReaSCAN, and Winoground test whether familiar components remain interpretable in new combinations or orders (Ruis et al., 2020; Kim and Linzen, 2020; Wu et al., 2021; Thrush et al., 2022). Our language experiment uses orthogonal token codes, channel-pair holdouts, sensory interventions, and relation reversal for the same general reason. The addition experiment reports exact answers and carry-sensitive results, following work that separates predictive performance from mechanistic recovery (Nikankin et al., 2025).

## 3 Methods

## 3.1 Experimental design

The two task models are trained independently on the anatomical topology and on directed degree-preserving rewires. Addition provides a repeated-run comparison across several graphs, whereas language uses a crossed design of five topologies and four input-output interfaces followed by a larger 21-graph comparison at interface I1. Rewiring changes the adjacency while retaining node and edge counts, every node’s directed degree, the task interface, parameter count, initialization distribution, optimizer, and training budget.

## 3.2 Connectome network and topology controls

We use the public MaleCNS v1.0 release (Berg et al., 2026) and retain every downloaded neuron assigned a superclass. The processed graph has 166,700 nodes and 25,582,938 directed neuron-to-neuron edges, representing 124,177,617 synaptic contacts. Appendix A gives the specimen, selection, orientation, and provenance details.

Let $G = ( V , E )$ denote the directed graph. For a batch state $h _ { t } \in \dot { \mathbb { R } ^ { | V | \times B } }$ , both task models use a leaky bounded recurrence

$$
h _ { t + 1 } = ( 1 - \alpha ) h _ { t } + \alpha \operatorname { t a n h } ( u _ { t } + W _ { E } h _ { t } ) ,\tag{1}
$$

where α is the leak coeficient, $u _ { t }$ is the task-specific input drive, and $W _ { E }$ has nonzeros only on $E .$ . For an edge $e = ( j  i ) , ( W _ { E } ) _ { i j } = \theta _ { e } / \sqrt { \mathrm { m a x } ( d _ { i } ^ { \mathrm { i n } } , 1 ) }$ , where $d _ { i } ^ { \mathrm { i n } }$ is the in-degree of node i. The recurrent gain is one in both tasks. Synaptic contact counts determine only the mask, and edge scalars are initialized independently. The edge vector $\bar { \boldsymbol { \theta } } \in \mathbb { R } ^ { | E | }$ is the only trainable tensor, with input and readout maps and codebooks held fixed.

Rewired controls are generated by directed double-edge swaps that preserve every node’s unweighted in-degree and out-degree, retain the original self-loops, and introduce no duplicate edges. After five accepted swaps per original edge, only 0.588–0.594% of source edges remain across the 20-graph ensemble. The comparison tests the collective contribution of higher-order properties, including reciprocity, motifs, paths, communities, and spectra, beyond directed degree. Appendix D gives the construction details and structural diagnostics.

## 3.3 Bounded-addition protocol

The task contains every ordered pair of nonnegative integers satisfying $a + b \leq 1 0 0$ . An expression $a + b$ is encoded in 47 fixed channels: two 22-dimensional digit codes and three operator channels. The network receives 64 recurrent updates, and logits for the 101 answers from 0 through 100 are averaged over the final four steps. Training minimizes cross-entropy plus a normalized ordinal CDF loss with unit coeficient. The 5,151 expressions are divided into 3,584 training, 775 validation, and 792 test examples. Commuted expressions are grouped before splitting, so no unordered operand pair occurs in more than one split. Each epoch visits all training expressions once without replacement, and the checkpoint with the lowest validation loss is selected before a single evaluation of the test set.

The comparison contains four optimization runs on the anatomical graph, four on the first rewire, and two each on two additional rewires, yielding an unbalanced graph-by-initialization design. We report the individual runs and their descriptive aggregate.

## 3.4 Grounded relational language protocol

The language task pairs sensory events with a question and a controlled English answer. In one held-out example, strong pressure on the right middle leg occurs before weak movement on the right front leg. Asked “What did I feel?”, the target answer begins “The strong pressure on the right middle leg happened before the weak movement on the right front leg.” Table 3 gives the complete record and its order reversal.

Each event combines one of six leg locations with one of four signal kinds and has binary strength, duration, and speed attributes. The 24 location–kind channels drive disjoint groups of anatomically annotated MaleCNS leg aferents, comprising 2,837 neurons. Event attributes determine the amplitudes and timing of the sensory input. Answers are rendered from a 100-word lexicon using authored task conventions, while the model vocabulary also includes three punctuation marks and six control symbols. Two fixed 128 × 109 codebooks assign orthonormal input and output codes to these 109 symbols.

Let $a _ { t } \in \mathbb { R } ^ { 2 4 }$ contain the task-defined sensory amplitudes and let $q _ { t }$ be the one-hot indicator for the current symbol. With fixed sensory and token gains $g _ { s }$ and $g _ { w } .$ the language drive is

$$
u _ { t } = g _ { s } P _ { s } a _ { t } + g _ { w } P _ { w } E _ { \mathrm { i n } } q _ { t } .\tag{2}
$$

For sensory group $S _ { g } , ( P _ { s } ) _ { i g } = | S _ { g } | ^ { - 1 / 2 }$ when $i \in S _ { g }$ and zero otherwise, while each code dimension is copied by $P _ { w }$ to eight token-input nodes. Token nodes are sampled without replacement from nodes with outgoing edges after excluding sensory nodes, and eight output nodes per code dimension are sampled from nodes with incoming edges after excluding both earlier sets. The sensory, token-input, and output sets are therefore disjoint. The four interfaces reuse the sensory groups and codebooks but place token and output nodes diferently.

If $O _ { d }$ is the eight-node output group for code dimension $d ,$ the readout is

$$
z _ { t , d } = \frac { 1 } { 8 } \sum _ { i \in O _ { d } } h _ { t , i } , \qquad \ell _ { t } = \frac { E _ { \mathrm { o u t } } ^ { \top } z _ { t } } { \tau \operatorname* { m a x } ( \Vert z _ { t } \Vert _ { 2 } , 1 0 ^ { - 6 } ) } .\tag{3}
$$

Here $\ell _ { t }$ contains the symbol logits and $\tau$ is the temperature.

The network first encodes the sensory sequence, then retains its recurrent state while reading the prompt and predicting the answer. During training, we supply a start symbol followed by the correct preceding answer symbols and minimize cross-entropy over answer symbols and the end symbol. Controlled text records instead provide next-symbol training without sensory input. Appendix B specifies the temporal schedule and loss normalization.

To test new combinations of familiar sensory channels, we partition the 276 unordered channel pairs into 228 training, 24 composition-validation, and 24 compositiontest pairs before rendering any text. All orders and wordings of a pair inherit its partition. The deterministic generator contains 2 report families, 18 question-answer families, and 11 text families. It materializes 200,000 training records and 4,096 records for each of familiar-family validation, composition validation, and composition test. Appendix B gives the split rule and template families.

Each full run uses the same budget of four million processed tokens. The first quarter is a report-only warm-up, and the remainder follows a deterministic 60:20:20 token mixture of reports, question answering, and text. Primary evaluation uses the final checkpoint. The factorial crosses the four interfaces with the anatomical graph and four rewires under one shared initialization. Both this comparison and the 20-rewire ensemble reuse settings established during method development at the canonical interface I1.

Primary endpoint. For each of 192 held-out two-event groups, we pair the original scene and its order reversal with their corresponding equal-length descriptions while leaving the question unchanged. Scoring each scene with each candidate gives four teacher-forced sequence log probabilities, summed over target tokens. A group is correct only when both matched scores strictly exceed both mismatched scores. Thus the model must favor the appropriate description for each scene and the appropriate scene for each description. Ties are failures. The primary endpoint is the fraction of correct groups. Under four continuous exchangeable scores, its chance reference is $1 / 6 = 1 6 . 6 7 \%$ , calibrated by score-position permutations in Appendix C.

## 3.5 Controls and statistical analysis

The sensory-shufle intervention leaves prompts and candidates unchanged but replaces the complete task-defined sensory-amplitude tensor with that of the next evaluation group in a deterministic cycle. The intervention uses two groups from each of 24 held-out families, for 48 groups in total, and compares normal and shufled evaluation on the same records. A parameter-matched dense recurrent control has width round $( \sqrt { | E | } ) = 5 { , } 0 5 8$ and approximately the same number of recurrent parameters, but far fewer recurrent units. Two additional fits use the anatomical graph at I1 and are trained and evaluated without the task-defined sensory channel.

For the pair-preserving counterfactual endpoint, let $r ^ { ( c ) }$ change one factor of record $r ,$ with rendered answers y and $y ^ { ( c ) }$ . Using mean teacher-forced log probability per target token, denoted m, a counterfactual pair is correct when $m ( r , y ) > m ( r , y ^ { ( c ) } )$ and $m ( r ^ { ( c ) } , y ^ { ( c ) } ) > m ( r ^ { ( c ) } , y )$ . We average the resulting accuracies across strength, duration, speed, and order, each evaluated on 48 groups. For comparison, we also report the stricter four-inequality score obtained by adding the two cross-scene comparisons from the primary endpoint. Location and signal-kind changes may alter channel-pair membership and are therefore excluded from the held-out aggregate.

The language factorial estimand is the mean withininterface diference between the anatomical graph and four rewires. We report a 20,000-sample two-way crossed bootstrap that independently resamples the four interface levels and the four rewire identities, preserving the crossed design. With four levels on each axis, the interval summarizes variation within the executed design. A separate analysis at I1 ranks the anatomical graph against 20 independently generated directed degree-preserving rewires under a shared initialization. Writing $T ( G )$ for a graph’s strict paired score, its finite-ensemble upper-tail fraction is

$$
\widehat { r } _ { 2 0 } = \frac { 1 + \# \{ T ( G _ { b } ^ { * } ) \geq T ( G ) \} } { 2 0 + 1 } .\tag{4}
$$

The minimum attainable rank fraction is $1 / 2 1$ . Two additional initializations test optimization robustness for the anatomical graph and two rewires.

## 4 Results

Table 1 summarizes the primary topology comparisons, and the accompanying figures show the individual optimization runs and graph draws.

## 4.1 The connectome topology improves held-out addition

The anatomical graph reaches a mean held-out accuracy of 92.77% across four optimization runs, compared with 67.93% across eight fits on three rewired graphs, a diference of 24.84 pp (Table 1 and Figure 2). On examples with a ones-digit carry and a sum below 100, the corresponding means are 92.96% and 69.09%. The weakest anatomical run reaches 78.79%, indicating substantial optimization variability. The constant-answer test baseline is 0.51%. Appendix F reports every run, carry stratum, and mean absolute error.

## 4.2 The topology advantage extends to grounded relational language

Varying the interface tests whether the topology diference persists across token-input and output placements. Across four interfaces, the anatomical graph obtains 61.59% on the strict paired language endpoint. The four rewires paired with each interface average 44.17%, yielding a $1 7 . 4 2 p p$ diference and a 95% crossed-bootstrap interval of $[ 1 1 . 3 9 , 2 4 . 9 0 ] p p$ . The separation is positive at every placement tested (Figure 3).

Table 1: Primary topology comparisons.
<table><tr><td>Endpoint</td><td>Anat.</td><td>Rewired</td><td>∆</td></tr><tr><td>Addition overall</td><td>92.77</td><td>67.93</td><td>+24.84</td></tr><tr><td>Addition carry (&lt; 100)</td><td>92.96</td><td>69.09</td><td>+23.88</td></tr><tr><td>Language factorial</td><td>61.59</td><td>44.17</td><td>+17.42</td></tr><tr><td>Language ensemble</td><td>71.35</td><td>46.56</td><td>+24.79</td></tr></table>

Values are percentages. Addition averages four anatomical fits and eight fits over three rewired graphs. The language factorial crosses four interfaces with four rewires; the ensemble row com pares one anatomical run with 20 independent rewires. The last column is descriptive.

![](images/5f3f62cde3f0122f46fd4894b4820f46313c493d3b94b1ffba7875062177dd53.jpg)

![](images/35290e8ba4e6b8df29e2966af0176c2f3a5ce57639e543034ee9128309df7aca.jpg)  
Figure 2: Bounded-addition results. (a) Held-out accuracy for four anatomical runs and eight runs on three degree-preserving rewires. Horizontal segments mark topology-class means. (b) Accuracy without and with a ones-digit carry among examples whose sum is below 100. Lines connect the two strata within each run. Sum-100 examples are reported separately in Appendix F.

The larger ensemble examines variation across graph draws at I1, where the anatomical graph scores 71.35%. Twenty independent rewires average 46.56% (SD 6.99 pp, range 36.98–58.85%), and none reaches the anatomical score (Figure 4). The resulting finite-ensemble upper-tail rank fraction is $\widehat { r } _ { 2 0 } = 1 / 2 1 = 0 . 0 4 7 6$

Across three optimization runs at I1, the anatomical graph averages 71.01% (SD 2.62 pp), compared with 46.53% for Rewire 1 (SD 8.87 pp) and 46.18% for Rewire 2 (SD 11.22 pp). Optimization variance is smaller for the anatomical language runs than for either rewire, although the anatomical addition runs show a wider spread.

We next examined whether diferences in access to the output nodes could explain the language gap. Every output node is reachable from both token and sensory inputs at all four interfaces. The largest strongly connected component contains 99.17% of nodes in both topology classes, and the mean minimum token-to-output distance is 1.51 in both. The anatomical graph has far more reciprocal directed edges (29.89% versus 0.514–0.520%) and longer minimum sensory-to-output paths (mean 2.55 versus 1.38). Table 7 gives the full structural comparison.

## 4.3 Sensory alignment and relational generalization

On the 48-group intervention subset, shufling the taskdefined sensory amplitudes reduces the anatomical score from 60.94% to 19.27%, a 41.67 pp drop. The mean rewired score falls from 41.80% to 13.02%, a 28.78 pp drop. Two models trained without sensory input score 0%: original and reversed scenes then produce identical inputs, making the endpoint’s four strict inequalities impossible to satisfy. The shufle result connects language performance to the alignment between the sensory stream and the rendered descriptions.

The counterfactual evaluation asks whether changing one scene factor changes which answer the model prefers. On the two-preference pair-preserving endpoint for strength, duration, speed, and order, the anatomical graph reaches 34.11% versus 8.66% for rewires. The corresponding strict four-comparison means are 16.41% and 4.69%. Anatomical accuracy on the two-preference endpoint is highest for duration and strength and lowest for speed (Table 13). For comparison, the parametermatched dense recurrent model obtains 8.72% on the primary endpoint, 1.56% under sensory shufle, and 3.39% on the two-preference counterfactual endpoint.

## 5 Discussion

Across the two tasks, the anatomical graph’s advantage indicates that directed degree does not capture all of the useful organization in this connectome. The language gap persists across token-input and output placements and graph draws while sensory aferents remain in their anatomical groups. Because each task is fitted independently, what carries across tasks is the recurrent architecture rather than trained weights.

The language results also depend on the correspondence between sensory input and text. The attribute-level counterfactual scores show that this grounding is uneven (Table 13). The anatomical model distinguishes strength and duration changes more reliably than changes in speed.

The structural diagnostics make a simple reachability explanation unlikely, since both topology classes retain access from the inputs to every output node. Rewiring changes the organization of those routes even when access is preserved. Null models that retain reciprocity, community structure, or sensory-to-output path statistics could distinguish their contributions to the performance gap.

![](images/d092fc21ca10981925b4686804922ba5fda45fd4d37504f781ce9ea449d04a53.jpg)

![](images/b0a586f50828774d6e8a2fa713e3d636bbf54fd56bd123fc71f8f86f1e4fa85d.jpg)

Figure 3: Grounded relational language results. (a) Strict paired scores at the canonical interface I1 and three alternative placements. Each anatomical fit is compared with four rewires. Horizontal segments mark rewire means. (b) On the same 48 groups, sensory shufling reduces the anatomical score from 60.94% to 19.27% and the rewire mean from 41.80% to 13.02%. The dotted line marks the 1/6 chance reference.  
![](images/3c26a6401e617120d097a6d6b050ffb6dc4fa9a1c5b9dcf0a6dae23ab0c56d5a.jpg)  
Figure 4: Rewire ensemble at interface I1. The 20 degreepreserving rewire scores are sorted for display. The anatomical graph scores 71.35%, above every rewire and the ensemble mean of 46.56%. Its finite-ensemble rank fraction is 1/21. The dotted line marks the 1/6 chance reference.

## 6 Limitations

We evaluate one anatomical graph and two synthetic tasks. The addition runs form an unbalanced graph-byinitialization design, and the 20-graph language ensemble limits rank resolution to 1/21. Four interfaces sample only a small part of the possible placement space.

The compact vocabulary, authored semantics, and controlled grammar make relational generalization measurable but leave open scaling to natural language. The network omits synaptic signs, neurotransmitter identities, delays, cell dynamics, and biological plasticity. Its dense control matches recurrent parameter count rather than state dimension or sparsity.

## 7 Conclusion

A recurrent network with the processed MaleCNS topology can learn bounded addition and controlled grounded relational language through separate fits. Degree-preserving rewiring reduces performance on both tasks, with the language gap persisting across the tested token-input and output placements and graph draws. Sensory shufling sharply reduces language performance, linking the learned descriptions to the task-defined input stream. These results identify the connectome as a useful architectural prior and motivate isolating the structural properties responsible for its advantage.

## AI Use Statement

We used OpenAI generative AI tools to develop and review the synthetic data generator and to polish the manuscript’s prose. Their use for the training code, equations, literature, methodology, result interpretation, figures, and translations was limited to review. The authors take responsibility for the final content.

## References

Berg, S., et al. (2026). Sexual dimorphism in the complete Drosophila male central nervous system connectome. Cell, 189:5504–5526.e15.

Dhiman, N. (2026). Topological sensitivity in connectome-constrained neural networks. arXiv preprint arXiv:2604.04033.

Dorkenwald, S., Matsliah, A., Sterling, A. R., et al. (2024). Neuronal wiring diagram of an adult brain. Nature, 634:124–138.

Jin, Z., Zhu, Y., Zhang, C., and Sui, Y. (2026). Whole-brain connectomic graph model enables wholebody locomotion control in fruit fly. arXiv preprint arXiv:2602.17997.

Kim, N. and Linzen, T. (2020). COGS: A compositional generalization challenge based on semantic interpretation. In Proceedings of EMNLP.

Lappalainen, J. K., Tschopp, F. D., Prakhya, S., et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. Nature, 634:1132–1140.

Liang, Y., Ryali, C. K., Hoover, B., Grinberg, L., Navlakha, S., Zaki, M. J., and Krotov, D. (2021). Can a fruit fly learn word embeddings? In International Conference on Learning Representations.

Lin, A., Yang, R., Dorkenwald, S., et al. (2024). Network statistics of the whole-brain connectome of Drosophila. Nature, 634:153–165.

McAllister, J., Houghton, C., Wade, J., and O’Donnell, C. (2026). Non-random brain connectome wiring enables robust and eficient neural network function under high sparsity. bioRxiv preprint 2026.03.30.715411.

Morra, J., Fouke, K., Naumann, E. A., and Daley, M. (2025). Connectomes inform function: from timevarying dynamics to animal behaviour. Natural Computing, 24(3):511–528.

Nikankin, Y., Reusch, A., Mueller, A., and Belinkov, Y. (2025). Arithmetic without algorithms: Language models solve math with a bag of heuristics. In International Conference on Learning Representations.

Pircher, T., Pircher, B., Schlücker, E., and Feigenspan, A. (2021). The structure dilemma in biological and artificial neural networks. Scientific Reports, 11:5621.

Pospisil, D. A., Aragon, M. J., Dorkenwald, S., et al. (2024). The fly connectome reveals a path to the efectome. Nature, 634:201–209.

Riveland, R. and Pouget, A. (2024). Natural language instructions induce compositional generalization in networks of neurons. Nature Neuroscience, 27:988–999.

Ruis, L., Andreas, J., Baroni, M., Bouchacourt, D., and Lake, B. M. (2020). A benchmark for systematic generalization in grounded language understanding. In Advances in Neural Information Processing Systems.

Shiu, P. K., Sterne, G. R., Spiller, N., et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. Nature, 634:210–219.

Thrush, T., Jiang, R., Bartolo, M., et al. (2022). Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings of CVPR.

Vong, W. K., Wang, W., Orhan, A. E., and Lake, B. M. (2024). Grounded language acquisition through the eyes and ears of a single child. Science, 383:504–511.

Wang, B. and Chen, J. (2026). FLYNN: Robust neural network for robot navigation using fly brain topology. arXiv preprint arXiv:2607.00025.

Wang, C., Dong, X., Ji, Z., Xiao, M., Jiang, J., Liu, X., Huan, Y., and Wu, S. (2026). Model-agnostic linearmemory online learning in spiking neural networks. Nature Communications, 17:1745.

Wu, Z., Kreiss, E., Ong, D. C., and Potts, C. (2021). ReaS-CAN: Compositional reasoning in language grounding. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.

Yu, S., Qin, Z., Liu, T., Xu, B., Vogelstein, R. J., Brown, J., and Vogelstein, J. T. (2026). Biological processing units: Leveraging an insect connectome to pioneer biofidelic neural architectures. In Artificial General Intelligence: 18th International Conference, AGI 2025, Lecture Notes in Computer Science, vol. 16058, pp. 361–369.

Zhang, X., Yang, P., Feng, J., Wen, K., Yan, G., Luo, Q., Lin, W., and Lu, X. (2026). Network structure governs Drosophila brain functionality. Fundamental Research, 6(3):1859–1868.

## A Data and Graph Provenance

MaleCNS v1.0 reconstructs the CNS of a single adult male and was obtained from the oficial release page (Berg et al., 2026). The MaleCNS v1.0 dataset is available under CC BY 4.0.

We selected annotation rows with an assigned superclass, sorted retained body IDs, and kept a directed connection only when both endpoints survived selection. Repeated records for the same ordered neuron pair were aggregated by summing their contact counts. Contact counts define edge presence rather than initial weight magnitudes. Our rule retains 166,700 unique body IDs, matching the total reported by Berg et al. (2026). Their connectivity graph contains 166,483 neurons after excluding 217 neurons without synaptic connections. Table 2 summarizes the source and processed graph counts.

Table 2: Processed graph and source statistics.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Source-article neuron total</td><td>166,700 neurons</td></tr><tr><td>Source-article connected neurons</td><td>166,483 neurons</td></tr><tr><td>Selection</td><td>Superclass annotation assigned</td></tr><tr><td>Selected neurons in processed graph</td><td>166,700</td></tr><tr><td>Directed aggregated edges</td><td>25,582,938</td></tr><tr><td>Synaptic contacts represented</td><td>124,177,617</td></tr><tr><td>Self-edges</td><td>101</td></tr><tr><td>Edge threshold</td><td>None</td></tr><tr><td>Orientation</td><td>postsynaptic rows, presynaptic columns</td></tr></table>

Graph orientation follows the implemented sparse matrix: rows are postsynaptic neurons and columns are presynaptic neurons. Thus a stored edge $( i , j )$ contributes from neuron j to neuron i. The graph contains 101 self-edges, which remain unchanged under rewiring. Appendix D gives the rewire algorithm and structural diagnostics.

The anatomical rendering in Figure 1 uses a deterministic sample of 3,000 retained neurons with nonnull soma locations from the oficial precomputed skeleton service. The overlaid nodes and arrows illustrate the network schematically.

## B Task, Model, and Training Details

## B.1 Shared recurrent parameterization

In both tasks, the edge scalars are initialized independently from $\mathcal { N } ( 0 , 1 )$ using a run-specific random state. The factor $1 / \sqrt { \operatorname* { m a x } ( d _ { i } ^ { \mathrm { i n } } , 1 ) }$ normalizes every edge entering node i, and recurrent states begin at zero. Input groups, output groups, graph indices and language codebooks are fixed, so only the edge scalars are optimized.

## B.2 Addition task and optimization

Input groups contain 64 graph nodes per channel, and output groups contain 32 nodes per class. The groups are sampled without replacement from nodes with outgoing or incoming edges and are disjoint. For an encoded expression $x \in \mathbb { R } ^ { 4 7 }$ , channel $x _ { c }$ is copied with amplitude 1 to every node in its input group at each recurrent update. If $R _ { k }$ is the output group for answer k, its logit is

$$
\ell _ { k } = \frac { 1 } { 4 \tau } \sum _ { t = T - 3 } ^ { T } \frac { 1 } { | R _ { k } | } \sum _ { i \in R _ { k } } h _ { t , i } , \qquad T = 6 4 ,
$$

with temperature $\tau = 0 . 1$ . The readout has no learned weight or bias.

We use $\alpha = 0 . 5 ,$ unit recurrent gain, and AdamW with learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 4 }$ . The gradient norm is clipped at 1, and the microbatch size is 128. Runs train for at most 100 epochs, with a minimum of 20 epochs and patience 15. If $p = \operatorname { s o f t m a x } ( \ell )$ and $y \in \{ 0 , \ldots , 1 0 0 \}$ , the training loss is

$$
{ \mathcal { L } } _ { \mathrm { a d d } } = - \log p _ { y } + { \frac { 1 } { 1 0 0 } } \sum _ { c = 0 } ^ { 9 9 } \left( \sum _ { k = 0 } ^ { c } p _ { k } - \mathbf { 1 } \{ y \leq c \} \right) ^ { 2 } .
$$

The second term penalizes disagreement between the predicted and target cumulative distributions over the ordered answer classes.

## B.3 Language task specification

An event is a tuple $e = ( l , k , v , d , s )$ . Here l is one of six leg locations, k is one of four signal kinds, and $v , d ,$ and s are binary strength, duration, and speed attributes. The channel index is $4 l + k ,$ giving 24 sensory channels. Complete report answers describe all five attributes. Table 3 gives a held-out composition-test record. The primary paired endpoint renders both descriptions in a fixed relation-first form (“ X happened before $Y ^ { \prime \prime } )$ and leaves the question unchanged after reversing event order.

Table 3: Worked example of a paired language group.
<table><tr><td>Component</td><td>Content</td></tr><tr><td>Question</td><td>What did I feel?</td></tr><tr><td>Original scene r Candidate y</td><td>strong, long, slow pressure on the right middle leg; then weak, short, slow movement on the right front leg The strong pressure on the right middle leg happened before the weak movement on the right front leg. The first signal started slowly and stopped after a long time. The last signal started slowly and stopped</td></tr><tr><td>Reversed scene  $r ^ {  }$  Candidate  $y ^ {  }$ </td><td>after a short time. weak, short, slow movement on the right front leg; then strong, long, slow pressure on the right middle leg The weak movement on the right front leg happened before the strong pressure on the right middle leg. The first signal started slowly and stopped after a short time. The last signal started slowly and stopped after a long time.</td></tr></table>

Table 4 lists the additional meaning and action conventions used by a subset of question-answer and text families.

Table 4: Authored signal conventions in the language generator.
<table><tr><td>Signal kind</td><td>Rendered meaning</td><td>Rendered action</td></tr><tr><td>touch</td><td>a signal from the body</td><td>follow the signal</td></tr><tr><td>taste</td><td>food</td><td>approach the food</td></tr><tr><td>movement</td><td>a change in the ground</td><td>follow the movement</td></tr><tr><td>pressure</td><td>pain and danger</td><td>avoid that place</td></tr></table>

The complete lexical vocabulary is shown in Table 5. Punctuation and the six control symbols are separate model symbols and are not counted among the 100 words. The input and output codebooks have orthonormal columns and are generated once for use at all four interface placements. Sensory waveforms are deterministic functions of the split index and record identifier.

<table><tr><td>Category Words</td><td>Table 5: The 100-word lexicon.</td></tr><tr><td>Nouns</td><td>fly, body, leg, legs, side, strength, touch, taste, movement, pressure, signal, sensation, event, events, pain, food,</td></tr><tr><td>Attributes</td><td>ground, danger, place, time, pulse, change left, right, front, middle, hind, weak, strong, short, long, quickly, slowly, first, last, same, different, one, two,</td></tr><tr><td>Verbs</td><td>three, four feel, felt, detect, detected, happen, happened, start, started, stop, stopped, mean, meant, avoid, avoided,</td></tr><tr><td>Function words</td><td>approach, approached, follow, followed, do  ${ \mathrm { a } } ,$  the, that, my, its, it, i, they, both, another, is, are, was, were, did, in, on, to, from, of, and, but, so, because, before, after, while, then, not, no, yes, what, where, when, which, why, how, many, there, or</td></tr></table>

The generator contains the families summarized in Table 6. Report families express complete event descriptions and supply the composition endpoint. Question-answer families isolate attributes, comparisons, temporal order, meanings, and actions. Text families provide next-symbol training on short controlled documents.

Table 6: Template families used by the deterministic corpus generator.
<table><tr><td>Mode</td><td>Count</td><td>Content</td></tr><tr><td>Report</td><td>2</td><td>First- and third-person descriptions of complete event sequences</td></tr><tr><td>Question answering</td><td>18</td><td>Event identity and attributes, temporal order, location, comparisons, counts, meaning, action, and causal explanation</td></tr><tr><td>Controlled text</td><td>11</td><td>Temporal and causal statements, contrasts, simultaneous events, counting, side descriptions, pulses, and short dialogue</td></tr></table>

## B.4 Corpus generation and split construction

The generator indexes the 24 sensory channels from 0 to 23. It assigns the 24 adjacent cyclic pairs $\{ i , ( i + 1 )$ mod 24} to composition validation and the 24 ofset-five pairs $\{ i , ( i + 5 )$ mod 24} to composition test after canonicalizing pair order, leaving the remaining 228 unordered pairs for training. Scenes are sampled before wording is rendered, so every event order and paraphrase inherits the same pair partition. Familiar-family validation uses a deterministic hash reservation over attributed scenes and additionally reserves prompt identities for text records.

Training records consist of reports (60%), question answering (20%), and controlled text (20%). Training reports contain one or two events, whereas composition-test records contain two. Selected text families contain three or four events. Two-event reports and the two report-based question-answer families use four surface variants, while one-event reports use a single form. Other question-answer and text families use a single template, except dialogue records that combine report variants with question families. The materialized bank contains 200,000 training records and 119,040 distinct conditional examples after exact deduplication. The first quarter of the token budget draws only from reports. The remainder selects the currently under-exposed pool to maintain the same 60:20:20 ratio by processed tokens.

## B.5 Language interface and optimization

The sensory interface contains 24 disjoint anatomical aferent groups comprising 2,837 neurons selected using the side, leg entry nerve, class and subclass annotations. The task channels correspond to tactile mechanosensory bristles (touch), gustatory leg bristles (taste), chordotonal organs (movement) and campaniform sensilla (pressure). Each group is scaled by the inverse square root of its size. Token codes have dimension 128, with eight token-input nodes and eight output nodes per dimension. Token-input and output groups follow the sampling procedure described in Section 3.4.

Each sensory event occupies eight encoding steps. Its active channel lasts two steps for a short event or four for a long event. Weak and strong events have base amplitudes 0.4 and 0.8. Slow events use a linear envelope from 0.6 to 1, whereas fast events use a flat envelope. Deterministic Gaussian noise with standard deviation 0.003 is added and clipped to [0, 1]. Events are encoded in scene order.

After sensory encoding, the grounded-mode symbol is applied for one recurrent transition. The network then processes the prompt followed by a separator and predicts the answer from a start symbol. Each prompt symbol and teacher-forced previous-target symbol receives four recurrent transitions with a one-step input pulse. Text records use the text-mode symbol and predict the complete document from the start symbol without a conditioning prompt or sensory input. Five non-output control symbols are masked from the logits, while punctuation, lexical words, and the end symbol remain available. The recurrence uses leak 0.5, token gain 4, sensory gain 1, and temperature 0.1.

AdamW uses learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , global batch 192, microbatch 24, and gradient clipping at 1. Microbatch gradients are accumulated with summed cross-entropy normalized by the total number of supervised symbols in the global batch, excluding padding. Primary results use the final checkpoint after the common four-million-token training budget.

## B.6 Hyperparameter selection and interfaces

Method development established the language recurrence, optimizer, curriculum, codebook construction, and canonica interface I1. A training-only diagnostic checked task learnability. The subsequent four-interface factorial and 20-rewire ensemble reused these settings and the same evaluation protocol and training budget.

## B.7 Language controls

The dense control replaces the sparse recurrence with a fully connected 5,058-unit state and a $5 , 0 5 8 \times 5 , 0 5 8$ trainable recurrent matrix, initialized from $\mathcal { N } ( 0 , 1 )$ and scaled by $1 / \sqrt { 5 , 0 5 8 }$ . It shares the codebooks, task generator, encoding schedule, objective, optimizer settings, curriculum, and four-million-token budget of the connectome models. Its sensory interface consists of 24 singleton input groups. Token and output groups retain eight units per code dimension.

Two no-sensory controls use the anatomical graph at I1 and were trained from scratch with sensory input disabled.   
Evaluation also holds the sensory tensor at zero.

## C Evaluation Metrics and Statistical Procedures

## C.1 Strict paired language endpoint

For a held-out two-event record r, let $r ^ {  }$ reverse the event order without changing the question. Let $y$ and $y ^ {  }$ be the equal-length answers rendered from the two scenes. If $s ( x , y )$ is the teacher-forced sequence log probability of candidate y under scene and prompt x, summed over target symbols, the group is correct exactly when

$$
\begin{array} { l } { { s ( r , y ) > s ( r , y ^ {  } ) , } } \\ { { s ( r , y ) > s ( r ^ {  } , y ) , } } \end{array}
$$

$$
\begin{array} { l } { { s ( r ^ {  } , y ^ {  } ) > s ( r ^ {  } , y ) , } } \\ { { s ( r ^ {  } , y ^ {  } ) > s ( r , y ^ {  } ) . } } \end{array}
$$

Thus both matched scores must exceed both mismatched scores, and ties fail. The primary score is the proportion correct over 192 held-out groups. With four continuous exchangeable scores, four of the 4! = 24 orderings satisfy this criterion, yielding the analytic reference $1 / 6 .$ . Candidate lengths are equal within each group, so summed and mean log probabilities give the same ordering for this endpoint.

The sensory intervention uses two records from each of the 24 held-out pair families. For each of these 48 group IDs, the normal evaluation is compared with a shufle that substitutes the full sensory-amplitude tensor from the next group in a deterministic cycle. The prompt and candidates do not change, so both scores are computed on the same records.

For a pair-preserving counterfactual $r ^ { ( c ) }$ , the two-preference criterion requires

$$
m ( r , y ) > m ( r , y ^ { ( c ) } ) \quad \mathrm { a n d } \quad m ( r ^ { ( c ) } , y ^ { ( c ) } ) > m ( r ^ { ( c ) } , y ) ,
$$

where m is mean teacher-forced log probability per target symbol. The strict version adds the two cross-scene inequalities used above. Strength, duration, speed, and order each use 48 groups. Location and signal-kind substitutions are excluded because they may move a record to a diferent held-out channel-pair family.

## C.2 Factorial uncertainty and graph ranking

Let $D _ { i j }$ be the score diference between the anatomical graph and rewire $j$ at interface i. The reported factorial efect is $\begin{array} { r } { \bar { D } = 1 6 ^ { - 1 } \sum _ { i = 1 } ^ { 4 } \sum _ { j = 1 } ^ { 4 } D _ { i j } } \end{array}$ . For each of 20,000 bootstrap replicates, four interfaces and four rewires are independently resampled with replacement, and the mean of the resulting crossed $4 \times 4$ table is recorded. Its 2.5th and 97.5th percentiles give the reported interval for variation within this $4 \times 4$ design.

The I1 ensemble compares one anatomical graph with 20 independently constructed rewires. Its upper-tail rank fraction is

$$
\widehat { r } _ { 2 0 } = \frac { 1 + \# \{ T ( G _ { b } ^ { * } ) \geq T ( G ) \} } { 2 1 } .
$$

The smallest attainable value is $1 / 2 1$ , and we report the quantity as a finite-ensemble rank without assuming uniformity of the double-edge-swap chain. Endpoint calibration separately permutes the four score positions within each group 2,000 times, recovering the analytic 16.67% reference at all four anatomical interfaces.

## D Rewire Construction and Structural Diagnostics

Each control begins from the processed directed graph. A proposed double-edge swap replaces $( a , b )$ and $( c , d )$ with $( a , d )$ and (c, b). Proposals that introduce a duplicate edge or change a self-edge are rejected. All other accepted swaps preserve every node’s in-degree and out-degree. Each rewire uses five accepted swaps per source edge. The acceptance rate is approximately 98.66% across the 20 controls.

We computed Table 7 directly from the saved edge arrays. Reciprocity is the fraction of directed edges whose reverse is present. Strongly connected components follow the presynaptic-to-postsynaptic direction. For each interface, the path statistic averages across output nodes the minimum unweighted directed distance from any node in the named input set. The sensory set contains 2,837 anatomical aferents; the token set contains 1,024 input nodes. Anatomical values average four interfaces. Rewire path values pool four interfaces and 20 graphs. Every output node was reachable.

Rewiring replaces almost every edge but leaves the giant strongly connected core and token-to-output distance nearly unchanged. It greatly reduces reciprocity and shortens sensory-to-output distances. Shared reachability but divergent higher-order statistics motivate more selective null models.

Table 7: Structural diagnostics for the anatomical graph and rewired controls. Brackets give the minimum and maximum across 20 rewires.
<table><tr><td>Diagnostic</td><td>Anatomical</td><td>20 rewires</td></tr><tr><td>Source-edge overlap (%)</td><td>100</td><td>0.591 [0.588, 0.594]</td></tr><tr><td>Accepted swaps per edge</td><td>0</td><td>5</td></tr><tr><td>Reciprocal directed edges (%)</td><td>29.893</td><td>0.517 [0.514, 0.520]</td></tr><tr><td>Largest SCC (% nodes)</td><td>99.169</td><td>99.176 [99.175, 99.176]</td></tr><tr><td>Token→output min. distance</td><td>1.507</td><td>1.506 [1.448, 1.546]</td></tr><tr><td>Sensory→output min. distance</td><td>2.552</td><td>1.378 [1.343, 1.412]</td></tr><tr><td>Output reachability (%)</td><td>100</td><td>100</td></tr></table>

## E Compute and Reproducibility

Both tasks were implemented in $\mathrm { P y }$ Torch and trained in FP32, with each process using one NVIDIA RTX 3090 GPU. Sparse recurrent propagation costs $O ( | E | B )$ per step and uses $O ( | E | + | V | B )$ principal storage aside from autograd state.

Code and data availability. MaleCNS v1.0 source tables and skeletons are available from the oficial release page. Code is available at https://github.com/joonghui0926/drosophila-connectome-cognitive-tasks. The repository includes compact result records, checksums and instructions for rebuilding the graph and generated corpus. MaleCNS data and full training checkpoints are not redistributed.

## F Detailed Numerical Results and Controls

Unless otherwise noted, accuracy and score entries in Tables 8–14 are percentages. MAE is measured in answer-value units, and SD is reported in percentage points.

## F.1 Addition runs

Table 8 reports every completed addition run used in the paper. The no-carry and carry columns partition the 772 expressions whose sum is below 100; the 20 expressions that equal 100 form a separate stratum. Means weight completed optimization runs equally.

Table 8: Every bounded-addition test run.
<table><tr><td>Topology</td><td>Run</td><td>Overall (792)</td><td>No carry (438)</td><td>Carry (334)</td><td>Sum 100 (20)</td><td>MAE</td></tr><tr><td>Anatomical</td><td>1</td><td>97.22</td><td>96.80</td><td>97.60</td><td>100</td><td>0.078</td></tr><tr><td>Anatomical</td><td>2</td><td>95.96</td><td>96.12</td><td>95.51</td><td>100</td><td>0.110</td></tr><tr><td>Anatomical</td><td>3</td><td>99.12</td><td>98.86</td><td>99.40</td><td>100</td><td>0.040</td></tr><tr><td>Anatomical</td><td>4</td><td>78.79</td><td>78.54</td><td>79.34</td><td>75</td><td>0.266</td></tr><tr><td>Rewire 1</td><td>1</td><td>66.41</td><td>67.35</td><td>67.37</td><td>30</td><td>0.413</td></tr><tr><td>Rewire 1</td><td>2</td><td>63.64</td><td>63.24</td><td>65.27</td><td>45</td><td>0.448</td></tr><tr><td>Rewire 1</td><td>3</td><td>72.47</td><td>74.66</td><td>70.06</td><td>65</td><td>0.355</td></tr><tr><td>Rewire 1</td><td>4</td><td>69.57</td><td>67.58</td><td>70.96</td><td>90</td><td>0.393</td></tr><tr><td>Rewire 2</td><td>1</td><td>68.81</td><td>67.35</td><td>68.86</td><td>100</td><td>0.385</td></tr><tr><td>Rewire 2</td><td>2</td><td>64.90</td><td>66.67</td><td>65.27</td><td>20</td><td>0.434</td></tr><tr><td>Rewire 3</td><td>1</td><td>65.40</td><td>61.19</td><td>68.86</td><td>100</td><td>0.443</td></tr><tr><td>Rewire 3</td><td>2</td><td>72.22</td><td>70.78</td><td>76.05</td><td>40</td><td>0.359</td></tr><tr><td>Anatomical mean</td><td>一</td><td>92.77</td><td>92.58</td><td>92.96</td><td>93.75</td><td>0.124</td></tr><tr><td>Rewire mean</td><td>一</td><td>67.93</td><td>67.35</td><td>69.09</td><td>61.25</td><td>0.404</td></tr></table>

## F.2 Language factorial and controls

Table 9 reports every cell in the crossed topology-interface comparison. Table 10 separately lists the 20 rewire scores used for the graph-level rank statistic at I1.

Table 9: Every cell in the primary language factorial. R1–R4 denote the four prespecified rewired graphs.
<table><tr><td>Interface</td><td>Anatomical</td><td>R1</td><td>R2</td><td>R3</td><td>R4</td><td>Rewire mean</td><td>∆</td></tr><tr><td>I1 (canonical)</td><td>71.35</td><td>38.02</td><td>58.85</td><td>42.71</td><td>40.10</td><td>44.92</td><td>26.43</td></tr><tr><td>I2</td><td>57.29</td><td>39.58</td><td>47.92</td><td>35.42</td><td>41.15</td><td>41.02</td><td>16.28</td></tr><tr><td>I3</td><td>61.46</td><td>50.52</td><td>51.04</td><td>35.42</td><td>47.40</td><td>46.09</td><td>15.36</td></tr><tr><td>I4</td><td>56.25</td><td>43.75</td><td>47.92</td><td>43.75</td><td>43.23</td><td>44.66</td><td>11.59</td></tr><tr><td>Mean</td><td>61.59</td><td>42.97</td><td>51.43</td><td>39.32</td><td>42.97</td><td>44.17</td><td>17.42</td></tr></table>

Table 10: I1 rewire scores, ordered from lowest to highest.
<table><tr><td>Position</td><td>Score</td><td>Position</td><td>Score</td><td>Position</td><td>Score</td></tr><tr><td>1</td><td>36.98</td><td>8</td><td>42.71</td><td>15</td><td>51.56</td></tr><tr><td>2</td><td>38.02</td><td>9</td><td>44.27</td><td>16</td><td>54.17</td></tr><tr><td>3</td><td>39.06</td><td>10</td><td>44.79</td><td>17</td><td>55.21</td></tr><tr><td>4</td><td>39.06</td><td>11</td><td>46.88</td><td>18</td><td>55.73</td></tr><tr><td>5</td><td>40.10</td><td>12</td><td>47.40</td><td>19</td><td>57.81</td></tr><tr><td>6</td><td>40.10</td><td>13</td><td>47.92</td><td>20</td><td>58.85</td></tr><tr><td>7</td><td>42.19</td><td>14</td><td>48.44</td><td></td><td></td></tr></table>

Table 11: Strict paired scores at I1 across optimization runs.
<table><tr><td>Topology</td><td>Run 1</td><td>Run 2</td><td>Run 3</td><td>Mean</td><td>SD</td></tr><tr><td>Anatomical</td><td>71.35</td><td>68.23</td><td>73.44</td><td>71.01</td><td>2.62</td></tr><tr><td>Rewire 1</td><td>38.02</td><td>55.73</td><td>45.83</td><td>46.53</td><td>8.87</td></tr><tr><td>Rewire 2</td><td>58.85</td><td>37.50</td><td>42.19</td><td>46.18</td><td>11.22</td></tr></table>

Table 12: Sensory intervention by interface. Normal and shufled values use the same records; rewire entries average four graphs.
<table><tr><td colspan="3">Anatomical</td><td colspan="2">Rewire mean</td></tr><tr><td>Interface</td><td>Normal</td><td>Shuffled</td><td>Normal</td><td>Shuffled</td></tr><tr><td>I1</td><td>68.75</td><td>25.00</td><td>44.79</td><td>12.50</td></tr><tr><td>I2</td><td>56.25</td><td>18.75</td><td>39.58</td><td>13.02</td></tr><tr><td>I3</td><td>62.50</td><td>20.83</td><td>42.71</td><td>14.58</td></tr><tr><td>I4</td><td>56.25</td><td>12.50</td><td>40.10</td><td>11.98</td></tr><tr><td>Mean</td><td>60.94</td><td>19.27</td><td>41.80</td><td>13.02</td></tr></table>

Table 13: Pair-preserving counterfactual accuracy by changed factor. Each factor uses 48 groups per run.
<table><tr><td colspan="3">Two-preference</td><td colspan="2">Strict four-inequality</td></tr><tr><td>Factor</td><td>Anatomical</td><td>Rewired</td><td>Anatomical</td><td>Rewired</td></tr><tr><td>Strength</td><td>46.35</td><td>13.02</td><td>17.19</td><td>1.82</td></tr><tr><td>Duration</td><td>59.38</td><td>4.69</td><td>22.40</td><td>0.52</td></tr><tr><td>Speed</td><td>4.17</td><td>0.52</td><td>0.00</td><td>0.13</td></tr><tr><td>Order</td><td>26.56</td><td>16.41</td><td>26.04</td><td>16.28</td></tr><tr><td>Macro mean</td><td>34.11</td><td>8.66</td><td>16.41</td><td>4.69</td></tr></table>

The counterfactual result is concentrated in strength and duration. Speed remains near zero.

Table 14: Dense-control results by optimization run. Primary, Normal, and Shufled report strict paired accuracy. Counterfactual reports the two-preference macro accuracy over strength, duration, speed, and order. All values are percentages.
<table><tr><td>Run</td><td>Primary</td><td>Normal (48)</td><td>Shuffled (48)</td><td>Counterfactual</td></tr><tr><td>1</td><td>5.73</td><td>4.17</td><td>0.00</td><td>2.08</td></tr><tr><td>2</td><td>8.33</td><td>6.25</td><td>0.00</td><td>3.65</td></tr><tr><td>3</td><td>17.71</td><td>22.92</td><td>6.25</td><td>6.25</td></tr><tr><td>4</td><td>3.13</td><td>4.17</td><td>0.00</td><td>1.56</td></tr><tr><td>Mean</td><td>8.72</td><td>9.38</td><td>1.56</td><td>3.39</td></tr></table>

Both no-sensory checkpoints score 0% on the primary and counterfactual endpoints and under sensory shufle.