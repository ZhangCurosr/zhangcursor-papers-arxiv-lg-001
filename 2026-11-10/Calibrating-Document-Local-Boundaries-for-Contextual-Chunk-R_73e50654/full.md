# Calibrating Document-Local Boundaries for Contextual Chunk Retrieval

Yuanxiang Wang Hui Jiang Da Zhu Zhihua Chang

Qwen Business Unit, Alibaba Group

Abstract. Late Chunking encodes an entire document before pooling at chunk boundaries, preserving document context in local retrieval units. Chunk-level post-training improves context-dependent retrieval, but shared context can make semantically similar same-document negatives dificult to distinguish. We study the local decision boundary between a gold chunk and its hardest sibling. Starting from InSeNT, we analyze InSeNT’s within-document gradients and show that the entropy of the conditional negative allocation decreases monotonically with the scaling factor at fixed logits. We propose a two-part framework for calibrating document-local chunk boundaries. The first component adapts the discriminative scale from within-document margins, while the second exposes chunk partitions to attention during document encoding. Experiments on ConTEB and NanoBEIR show that the resulting model improves contextual chunk retrieval across the evaluated encoder settings while retaining competitive passage-level retrieval quality; ablations further characterize the distinct roles of adaptive scaling and chunk-aware routing.

Keywords: Contextual embeddings · Dense retrieval · Contrastive learning · Document representation · Late Chunking

## 1 Introduction

Long-document retrieval usually returns local evidence that answers a query, rather than the whole document [1]. Its meaning often depends on context: pronouns refer to things mentioned earlier, numbers belong to particular years, and similar expressions may describe diferent facts [2]. Independent chunk encoding loses these cues, whereas a single document vector cannot localize evidence. Prior work improves contextual retrieval through input enrichment, long-context encoding, and chunk-level post-training [3, 4, 2]. These approaches preserve more document context, but retrieval still requires a model to separate locally similar chunks that share that context.

Because multiple chunks in a document share the same context, retrieval must distinguish the answer-bearing chunk from its same-document siblings; chunk boundaries therefore directly shape the retrieval objective. A useful training signal must preserve global context while distinguishing chunks that share entities, topics, or surface form. Therefore, we treat the gold chunk and its strongest same-document sibling as the basic unit of analysis, and evaluate both contextual and passage-level retrieval. We focus on contrastive post-training with same-document negative chunks. Of 19,974 InSeNT evaluation queries, 6,976 fail to rank gold first. Among these 6,976 queries, the highest-scoring sibling exceeds the gold score in 82.21% of cases, illustrating unresolved local factual confusion. Figure 1 asks for goals in 2019–20: both passages discuss the same player, European competitions, and scoring records, but the hard negative concerns another season.

<table><tr><td colspan="2">Query</td><td colspan="2">How many goals did Gonçalo Ramos score in the 2019–20 UEFA Youth League?</td></tr><tr><td colspan="3">Gold chunk During the 2019–20 UEFA Youth League, [he] helped Benfica reach the final, scoring a brace in the final and finishing as joint-top scorer with eight goals.</td><td>Same-document hard negative In 2022–23, [he] scored two goals in the Lisbon derby and two goals in a Champions League knockout match; [he] finished the season with 27 goals.</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Gold global rank 80</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="3">Fixed negative global rank 16</td></tr><tr><td colspan="3">11 Fixed-pair cosine margin -0.05</td></tr><tr><td colspan="3">+0.04</td></tr></table>

Figure 1: Season confusion in Football: the same gold–negative pair under InSeNT and DcLBC.

We identify two complementary gaps at this local boundary. First, InSeNT’s fixed inverse temperature jointly controls row-wise total efort and its allocation among siblings, despite heterogeneous query–document margins. AMS instead computes a detached scale from each pair’s gold–hardest-sibling margin and applies it only to the InSeNT within-document loss. Second, distinguishing sibling chunks requires contextual representations that preserve each chunk’s local evidence while incorporating shared document context. Although InSeNT supervises chunks, standard Late Chunking uses partitions only during post-encoder pooling; final-layer attention does not explicitly observe retrieval-unit boundaries.

We propose Document-Chunk Local Boundary Calibration (DcLBC), combining Adaptive Margin Scaling (AMS) and Chunk-Aware Key Routing (CKR). AMS adapts the discriminative scale to focus conditional efort on dificult negatives. CKR injects the chunk partition into post-RoPE Keys, allowing chunk-wise context routing during encoding. Figure 2 summarizes the pipeline from CKR-enabled encoding through chunk pooling and scoring to the AMS loss. Our contributions are:

• We analyze InSeNT at the query–document row level by separating total score-space efort from its conditional allocation among same-document negatives, and derive how competitor entropy changes with the scale.

• We introduce DcLBC, combining AMS, which derives a detached instance-level scale from the gold–hardest-sibling margin, with CKR, which injects chunk routing into post-RoPE Keys.

• We evaluate DcLBC on ConTEB and NanoBEIR with ModernBERT and ModernColBERT backbones, and use component ablations and CKR design tests to measure the contribution of each mechanism.

## 2 Related Work

Contextualized Chunk Representations.. Standard passage retrieval encodes each chunk independently, which can discard antecedents, definitions, and other evidence located elsewhere in the source document. Several approaches therefore enrich retrieval units with broader context. Contextual Retrieval prepends document-specific explanatory text to each chunk, whereas RAPTOR recursively summarizes and clusters passages into a hierarchical retrieval tree [3, 5]. Landmark Embedding learns landmark tokens that organize a long document into semantically retrievable spans rather than relying only on fixed boundaries [6]. Contextual Document Embeddings condition an item on related documents, extending contextualization beyond a single document [7]. Situated Embedding instead trains passage representations to retain information from their surrounding document context [8]. Late Chunking encodes the full text before pooling token states at chunk boundaries, allowing each pooled chunk to incorporate both left and right context [4]. Context is Gold further introduces In-Sequence Negative Training, in which chunks from the same document serve as competing candidates during contextual embedding post-training [2]. Together, these studies span textual augmentation, hierarchical organization, learned segmentation, contextual encoding, and chunk-level post-training as complementary routes to context-aware retrieval units.

Long-Context and Retrieval-Oriented Encoders.. Contextualized retrieval also depends on encoder architectures that can represent long inputs while preserving fine-grained matching signals. Longformer established an eficient bidirectional design based on sparse local and global attention for processing long sequences [9], while ModernBERT revisits bidirectional encoder pretraining with modern architectural and training choices [10]. In dense retrieval, DPR represents queries and passages with dual encoders and scores them by vector similarity, and Contriever demonstrates that retrieval representations can also be learned through unsupervised contrastive pretraining [1, 11]. Late-interaction models retain token-level vectors: ColBERT performs eficient fine-grained matching with a MaxSim operator, and ColBERTv2 improves its quality and storage eficiency through denoised supervision and residual compression [12, 13]. A parallel line adapts decoder-only language models into embedding encoders. Echo Embedding repeats the input to expose token representations to preceding context, LLM2Vec combines bidirectional attention with unsupervised and supervised adaptation, and NV-Embed introduces latent attention pooling together with retrieval-oriented training [14, 15, 16]. These families provide diferent tradeofs among context length, representation granularity, scoring expressiveness, and retrieval eficiency.

Contrastive Learning, Hard Negatives, and Temperature.. Modern embedding models are commonly optimized with contrastive objectives that raise the similarity of matched query– passage pairs relative to negative candidates. SimCSE shows how contrastive learning can produce strong sentence representations using either dropout-based positive pairs or supervised entailment data [17]. For retrieval, negative selection is especially important because randomly sampled passages may become uninformative early in training. ANCE addresses this issue by periodically mining dificult candidates from an approximate nearest-neighbor index, thereby coupling representation learning with the evolving retrieval space [18]. Temperature provides a second control over contrastive optimization: smaller temperatures sharpen the softmax distribution and concentrate gradients on harder negatives [19]. Dual Temperature separates scaling efects associated with diferent components of contrastive learning to reduce sensitivity to negative-sample quality [20]. MACL further adjusts contrastive training according to model-dependent alignment and uniformity signals rather than using a single fixed optimization regime [21]. This body of research establishes negative construction and logit scaling as central factors governing the strength and distribution of contrastive learning signals.

## 3 Analysis and Method

## 3.1 Within-document chunk competition

We first recall InSeNT’s in-sequence loss [2]. Index query– document pairs by � and chunks within pair � by $j .$ . The normalized query $q _ { i } \in \mathbb { R } ^ { d }$ has $n _ { i }$ normalized document chunks $c _ { i j } \in \mathbb { R } ^ { d }$ , collected as $\boldsymbol { C } _ { i } = [ c _ { i 1 } , \ldots , c _ { i n _ { i } } ] ^ { \top }$ , with gold index $g _ { i } .$ Let $a > 0$ be the similarity scale and $\tau > 0$ the within-document temperature. Holding the cross-document term fixed,

$$
s _ { i j } = \frac { a } { \tau } q _ { i } ^ { \top } c _ { i j } , \qquad \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } = - \log \frac { \exp ( s _ { i g _ { i } } ) } { \sum _ { j = 1 } ^ { n _ { i } } \exp ( s _ { i j } ) } .\tag{1}
$$

Here $s _ { i j }$ is the scaled query–chunk logit. The hardest samedocument negative and its margin from gold are

$$
h _ { i } = \arg \operatorname* { m a x } _ { j \neq g _ { i } } s _ { i j } , \quad m _ { i } = s _ { i g _ { i } } - s _ { i h _ { i } } .\tag{2}
$$

![](images/371fae4da8faf0100b5226d0ada53b90ff0743ad7f46244c95215c4c6ebed3d9.jpg)  
Figure 2: Overview of DcLBC. CKR redistributes attention across chunks, while AMS adaptively scales within-document scores.

Smaller $m _ { i }$ means stronger competition. Let $p _ { i j }$ be the withindocument softmax probability and I[·] the indicator function:

$$
\begin{array} { c } { { p _ { i j } = \displaystyle \frac { \exp ( s _ { i j } ) } { \sum _ { k = 1 } ^ { n _ { i } } \exp ( s _ { i k } ) } , } } \\ { { \log \displaystyle \frac { p _ { i g _ { i } } } { p _ { i h _ { i } } } = m _ { i } , \qquad \displaystyle \frac { \partial \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } } { \partial s _ { i j } } = p _ { i j } - \mathbb { I } [ j = g _ { i } ] . } } \end{array}\tag{3}
$$

Thus $m _ { i }$ is the gold–hard-negative posterior log-odds around which within-document optimization operates.

## 3.2 Document-Chunk Local Boundary Calibration

Local margins vary across query–document pairs. DcLBC addresses this heterogeneity through AMS and CKR.

## 3.2.1 Adaptive Margin Scaling (AMS)

AMS applies a detached instance-level multiplier $\kappa _ { i }$ to withindocument scores, with $p _ { i j } ^ { ( \kappa ) } ~ = ~ \exp ( \kappa _ { i } s _ { i j } ) / { \sum _ { k = 1 } ^ { n _ { i } } \exp ( \kappa _ { i } s _ { i k } ) }$ The local log-odds become

$$
\rho _ { i } ( \kappa _ { i } ) = \log \frac { p _ { i g _ { i } } ^ { ( \kappa ) } } { p _ { i h _ { i } } ^ { ( \kappa ) } } = \kappa _ { i } m _ { i } .\tag{4}
$$

At training step �, let $\mu _ { t } > 0$ be the detached EMA reference margin and $\widetilde { m } _ { i } = \operatorname* { m a x } ( m _ { i } , \epsilon )$ the floored margin. Minimizing $( \kappa \widetilde { m } _ { i } - \mu _ { t } ) ^ { 2 }$ over $1 \le \kappa \le K$ yields

$$
\kappa _ { i } = \mathrm { c l i p } \left( \frac { \mu _ { t } } { \widetilde { m } _ { i } } , 1 , K \right) .\tag{5}
$$

Here $\mathrm { c l i p } ( x , l , u ) = \operatorname* { m i n } ( \operatorname* { m a x } ( x , l ) , u ) , K \geq 1$ caps the scale, and $\epsilon > 0$ prevents divergence for small or negative margins. Appendix B specifies the EMA update.

Relation to temperature adjustment. For untempered scores $u _ { i j } = a q _ { i } ^ { \top } c _ { i j }$ , fixed and instance-level temperatures give $s _ { i j } =$ $u _ { i j } / \tau$ and $u _ { i j } / \tau _ { i }$ . AMS is also instance-level temperature

adjustment, specifically controlled by local same-document competition:

$$
\tau _ { i } ^ { \mathrm { A M S } } = \frac { \tau } { \kappa _ { i } } ,\tag{6}
$$

where $\kappa _ { i }$ in Eq. (5) adapts the temperature to each query– document pair’s gold–hardest-sibling margin. This adjustment applies only to the sequence loss; cross-document scores retain their original scale. Since $\rho _ { i } = \kappa _ { i } m _ { i }$ , AMS yields $\rho _ { i } = \mu _ { t }$ whenever $m _ { i } \geq \epsilon$ and $\mu _ { t } / K \le m _ { i } \le \mu _ { t }$

Optimization pressure. Define the total weight and conditional allocation over sibling negatives as

$$
W _ { i } = \kappa _ { i } \big ( 1 - p _ { i g _ { i } } ^ { ( \kappa ) } \big ) , \qquad r _ { i j } = \frac { p _ { i j } ^ { ( \kappa ) } } { 1 - p _ { i g _ { i } } ^ { ( \kappa ) } } , \quad j \ne g _ { i } .\tag{7}
$$

Let $\mathcal { L } _ { \mathrm { D c L B C } } ^ { ( i ) } = - \log p _ { i g _ { i } } ^ { ( \kappa ) }$ be the rescaled row loss, and let $\nabla = \nabla _ { \theta }$ diferentiate with respect to trainable parameters � while treating $\kappa _ { i }$ as constant. Its negative gradient is

$$
- \nabla \mathcal { L } _ { \mathrm { D c L B C } } ^ { ( i ) } = W _ { i } \left( \nabla s _ { i g _ { i } } - \sum _ { j \neq g _ { i } } r _ { i j } \nabla s _ { i j } \right) .\tag{8}
$$

$W _ { i }$ controls total efort and $r _ { i }$ its allocation over sibling negatives. At fixed logits, replace $\kappa _ { i }$ in $r _ { i }$ by a generic continuous scale $\kappa > 0$ and denote the resulting allocation by $r _ { i } ( \boldsymbol { \kappa } )$ Let $\begin{array} { r } { H ( r _ { i } ( \boldsymbol { \kappa } ) ) = - \sum _ { j \neq g _ { i } } r _ { i j } ( \boldsymbol { \kappa } ) } \end{array}$ log $r _ { i j } ( \boldsymbol { \kappa } )$ be its entropy. The sibling-only log-partition has second derivative $\mathrm { V a r } _ { r _ { i } ( \kappa ) } \big ( s _ { i j } \big )$ giving

$$
\frac { \mathrm { d } H ( r _ { i } ( \boldsymbol { \kappa } ) ) } { \mathrm { d } \boldsymbol { \kappa } } = - \kappa \operatorname { V a r } _ { r _ { i } ( \boldsymbol { \kappa } ) } \bigl ( s _ { i j } \bigr ) \leq 0 .\tag{9}
$$

At fixed logits, increasing $\kappa _ { i }$ concentrates negative gradients on higher-scoring competitors, providing a more focused update direction for high-competition documents.

## 3.2.2 Chunk-Aware Key Routing (CKR)

CKR introduces chunk boundaries into attention by adding a shared ofset to the keys of each chunk. Consider one attention head (index omitted), with post-RoPE token queries $q _ { s } \in \mathbb { R } ^ { d _ { h } }$ keys $\boldsymbol { k } _ { t } \in \mathbb { R } ^ { d _ { h } }$ , and head dimension $d _ { h } ; s$ and � index source and target tokens. From hidden states $h _ { t } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ in chunk $^ { c , }$ we compute a routing vector and update its keys:

Table 1: ConTEB and NanoBEIR NDCG@10 (%). Parentheses give base-model parameters in millions (M); bold/underline: best/second best. “–” indicates no additional post-training in this study.
<table><tr><td>Model</td><td>Post-training</td><td>MLDR</td><td>SQuAD</td><td>NQA</td><td>COVID</td><td>Rest.</td><td>Foot.</td><td>Cities</td><td>EIOPA</td><td>ConTEB Avg</td><td>NanoBEIR</td></tr><tr><td colspan="10">Retrieval baselines without additional post-training</td><td></td><td></td><td></td></tr><tr><td>BM25 [22]</td><td></td><td>69.43</td><td>56.18</td><td>74.68</td><td>53.75</td><td>19.91</td><td>12.20</td><td>45.61</td><td>0.00</td><td>41.47</td><td>43.41</td></tr><tr><td>Contriever-MSMARCO (109M) [11]</td><td></td><td>75.12</td><td>70.65</td><td>77.00</td><td>53.79</td><td>42.79</td><td>24.41</td><td>67.42</td><td>14.77</td><td>53.24</td><td>59.54</td></tr><tr><td>Anthropic Contextual</td><td></td><td>85.40</td><td>77.10</td><td>77.70</td><td>60.70</td><td>34.80</td><td>53.90</td><td>89.40</td><td>100.00</td><td>72.40</td><td>63.20</td></tr><tr><td colspan="10">Encoder models with additional post-training</td><td></td><td></td></tr><tr><td>ModernBERT-base (149M) [10]</td><td>InfoNCE</td><td>80.09</td><td>72.62</td><td>76.84</td><td>48.54</td><td>35.48</td><td>45.84</td><td>76.47</td><td>99.69</td><td>66.95</td><td>58.70</td></tr><tr><td></td><td>InSeNT</td><td>85.81</td><td>80.11</td><td>80.33</td><td>55.81</td><td>44.66</td><td>58.25</td><td>82.92</td><td>100.00</td><td>73.49</td><td>58.97</td></tr><tr><td></td><td>DcLBC</td><td>85.28</td><td>80.67</td><td>81.00</td><td>54.59</td><td>44.53</td><td>66.19</td><td>86.83</td><td>100.00</td><td>74.89</td><td>59.41</td></tr><tr><td>ModernBERT-large</td><td>InfoNCE</td><td>81.89</td><td>72.89</td><td>78.70</td><td>51.46</td><td>44.14</td><td>63.07</td><td>85.23</td><td>100.00</td><td>72.17</td><td>59.69</td></tr><tr><td>(395M) [10]</td><td>InSeNT</td><td>88.92</td><td>80.80</td><td>83.51</td><td>59.53</td><td>48.25</td><td>65.49</td><td>87.10</td><td>100.00</td><td>76.70</td><td>59.75</td></tr><tr><td></td><td>DcLBC</td><td>89.20</td><td>82.92</td><td>84.66</td><td>57.80</td><td>51.19</td><td>71.51</td><td>91.63</td><td>100.00</td><td>78.61</td><td>60.56</td></tr><tr><td>ModernColBERT</td><td>InfoNCE</td><td>74.88</td><td>63.49</td><td>78.81</td><td>47.05</td><td>27.99</td><td>48.18</td><td>82.52</td><td>21.53</td><td>55.56</td><td>54.25</td></tr><tr><td>(149M) [23]</td><td>InSeNT</td><td>89.32</td><td>77.24</td><td>83.32</td><td>64.78</td><td>49.00</td><td>64.55</td><td>89.66</td><td>51.42</td><td>71.16</td><td>57.96</td></tr><tr><td></td><td>DcLBC</td><td>89.70</td><td>74.31</td><td>82.61</td><td>65.54</td><td>46.58</td><td>68.02</td><td>91.37</td><td>70.73</td><td>73.61</td><td>56.79</td></tr><tr><td colspan="10">LLM-based embedding models (causal attention)</td><td></td><td></td><td></td></tr><tr><td>Qwen3-Embedding (596M) [24]</td><td></td><td>57.12</td><td>63.98</td><td>58.66</td><td>19.52</td><td>11.75</td><td>53.08</td><td>82.15</td><td>62.97</td><td>51.15</td><td>64.91</td></tr><tr><td>Harrier (268M) [25]</td><td></td><td>65.77</td><td>62.30</td><td>64.36</td><td>16.83</td><td>7.71</td><td>39.30</td><td>81.74</td><td>40.20</td><td>47.28</td><td>62.64</td></tr><tr><td>ML-Embed (596M) [26]</td><td></td><td>52.25</td><td>56.21</td><td>60.19</td><td>16.15</td><td>11.14</td><td>27.91</td><td>71.54</td><td>22.00</td><td>39.67</td><td>63.18</td></tr><tr><td colspan="10">LLM-based embedding models (bidirectional attention)</td><td></td></tr><tr><td>pplx-embed-v1 (596M) [27]</td><td></td><td>84.68</td><td>74.61</td><td>81.69</td><td>53.76</td><td>26.57</td><td>62.78</td><td>82.88</td><td>86.56</td><td>69.19</td><td>56.07</td></tr><tr><td>BidirLM (596M) [28]</td><td></td><td>58.29</td><td>68.90</td><td>66.40</td><td>36.07</td><td>28.87</td><td>60.68</td><td>92.76</td><td>23.79</td><td>54.47</td><td>57.45</td></tr><tr><td>Nano-Em1-0.6B-v2 (596M) [29]</td><td></td><td>49.87</td><td>41.98</td><td>60.90</td><td>26.67</td><td>22.22</td><td>20.37</td><td>74.63</td><td>33.57</td><td>41.28</td><td>49.18</td></tr></table>

$$
\begin{array} { r l } & { z _ { c } = \mathrm { G E L U } ( W _ { \mathrm { d o w n } } \mathrm { L N } ( \mathrm { M e a n } _ { t \in c } h _ { t } ) ) , } \\ & { k _ { t } ^ { \prime } = k _ { t } + U _ { k } z _ { c } , \qquad t \in c . } \end{array}\tag{10}
$$

Here LN denotes layer normalization, Mean averages token states, and GELU is the activation. The learned projections $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { r \times d _ { \mathrm { m o d e l } } }$ and $U _ { k } \in \mathbb { R } ^ { d _ { h } \times r }$ map through routing dimension �, so $z _ { c } \in \mathbb { R } ^ { r }$ . Adding the ofset after RoPE keeps it identical across positions within a chunk. Thus, for source token �, every token in � receives the same additive bias to its attention logit $\ell _ { s t } = q _ { s } ^ { \top } k _ { t } / \sqrt { d _ { h } }$

$$
\ell _ { s t } ^ { \prime } = \ell _ { s t } + b _ { s c } , \qquad b _ { s c } = \frac { q _ { s } ^ { \top } U _ { k } z _ { c } } { \sqrt { d _ { h } } } .\tag{11}
$$

Let $a _ { s t } = \mathrm { s o f t m a x } _ { t } ( \ell _ { s t } )$ be the original attention weight and $\textstyle A _ { s c } = \sum _ { t \in c } a _ { s t }$ the total attention mass assigned to chunk �. Applying softmax with the same mask gives

$$
\begin{array} { r l } & { \boldsymbol { A } _ { s c } ^ { \prime } = \cfrac { e ^ { b _ { s c } } \boldsymbol { A } _ { s c } } { \boldsymbol { A } _ { s 0 } + \sum _ { c ^ { \prime } } e ^ { b _ { s c ^ { \prime } } } \boldsymbol { A } _ { s c ^ { \prime } } } , } \\ & { \frac { \boldsymbol { a } _ { s t } ^ { \prime } } { \boldsymbol { A } _ { s c } ^ { \prime } } = \frac { \boldsymbol { a } _ { s t } } { \boldsymbol { A } _ { s c } } , \qquad t \in c , \ A _ { s c } > 0 . } \end{array}\tag{12}
$$

Here $A _ { s 0 }$ is the original mass of special or unassigned tokens, whose keys remain unchanged; all sums respect the attention

mask. CKR therefore multiplies each chunk’s mass by $e ^ { b _ { s c } }$ before renormalization, while preserving its internal attention proportions for fixed queries, token states, and routing vectors.

## 3.3 Joint training objective

For a batch of � query–document pairs, DcLBC retains In-SeNT’s cross-document loss $\mathcal { L } _ { \mathrm { { i n t e r } } }$ and rescales the withindocument loss:

$$
\mathcal { L } _ { \mathrm { s e q } } ^ { ( \kappa ) } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \kappa _ { i } s _ { i g _ { i } } ) } { \sum _ { j = 1 } ^ { n _ { i } } \exp ( \kappa _ { i } s _ { i j } ) } .\tag{13}
$$

With $\lambda _ { \mathrm { s e q } } \in [ 0 , 1 ]$ denoting the within-document loss weight, the full objective is

$$
\mathcal { L } _ { \mathrm { D c L B C } } = ( 1 - \lambda _ { \mathrm { s e q } } ) \mathcal { L } _ { \mathrm { i n t e r } } + \lambda _ { \mathrm { s e q } } \mathcal { L } _ { \mathrm { s e q } } ^ { ( \kappa ) } .\tag{14}
$$

The detached $\kappa _ { i }$ only controls within-document competition. CKR adds no loss: its representations enter both terms, and its parameters are jointly optimized with the encoder. Complete mathematical proofs of all propositions and identities in this section are provided in Appendix A.

## 4 Experiments

We evaluate the overall retrieval performance of DcLBC and several baseline methods on ConTEB and NanoBEIR, and examine the contributions of AMS and CKR through component ablations and CKR design analyses.

Table 2: Component ablation on ModernBERT-large: ConTEB NDCG@10, mean ± standard deviation (SD, %); Δ in percentage points (pp).
<table><tr><td>AMS</td><td>CKR</td><td>ConTEB Avg</td><td>∆ vs. InSeNT</td></tr><tr><td>X</td><td>X</td><td> $7 6 . 1 2 \pm 0 . 5 9$ </td><td>一</td></tr><tr><td>X</td><td>√</td><td> $7 6 . 6 8 \pm 0 . 4 3$ </td><td>+0.56</td></tr><tr><td>√</td><td>X</td><td> $7 7 . 5 9 \pm 0 . 3 3$ </td><td>+1.47</td></tr><tr><td>√</td><td>√</td><td> ${ \bf 7 8 . 3 8 \pm 0 . 2 3 }$ </td><td>+2.26</td></tr></table>

## 4.1 Experimental setup

Following the training data and two-epoch schedule of Context Is Gold [2], all post-training methods use the MLDR, NarrativeQA, and SQuAD corpora. Our runs use 8\*A800 GPUs with BF16 and a learning rate of $2 \times 1 0 ^ { - 4 }$ , and all methods are evaluated at their final checkpoint. Our primary metric is ConTEB eight-task macro normalized discounted cumulative gain at rank 10 (NDCG@10, abbreviated 8DS); NanoBEIR passage retrieval assesses performance without document context. ModernBERT models use normalized 1024-dimensional cosine scores and ModernColBERT uses MaxSim [12, 13].

## 4.2 Main results

Table 1 compares retrieval models on 8DS and NanoBEIR. DcLBC consistently outperforms InSeNT on 8DS across the evaluated encoder and multi-vector backbones. The gains reach 1.40 and 1.91 points on ModernBERT-base and ModernBERTlarge, respectively, and peak at 2.45 points on ModernColBERT. For ModernBERT-large, the largest task-level improvements occur on Football by 6.02 points and Cities by 4.53 points, while COVID is the only contextual task that declines. This contrast reflects ConTEB’s construction: Football and Cities suppress explicit mentions of the document subject to require coreference resolution, whereas COVID-QA often provides strong local matches through extractive medical terminology [2]. This concentration suggests that DcLBC is particularly efective when retrieval depends on distinguishing related chunks within a shared document.

For the two ModernBERT encoders, the NanoBEIR results further show that DcLBC does not trade away retrieval quality when document context is absent. The base and large models retain advantages of 0.44 and 0.81 points over InSeNT, reaching 59.41 and 60.56, respectively. Their concurrent gains on context-dependent ConTEB and passage-level NanoBEIR suggest that local boundary calibration is not merely exploiting extra document context.

## 4.3 Ablation and design analysis

We analyze DcLBC components and CKR design, reporting ConTEB eight-task macro NDCG@10 (%). Table 2 first isolates the contributions of AMS and CKR, while Table 3 examines how the routing configuration afects CKR. Because the two studies use separate controlled comparisons, we interpret each configuration relative to the corresponding baseline in its own table.

Table 3: CKR design analysis on ModernBERT-large: ConTEB NDCG@10, mean ± standard deviation (SD, %). Alignment tests use Last 2.
<table><tr><td>Configuration</td><td>ConTEB Avg</td></tr><tr><td>Layer coverage</td><td></td></tr><tr><td>None</td><td> $7 7 . 6 0 \pm 0 . 3 2$ </td></tr><tr><td>Last 1 (27)</td><td> $7 7 . 9 2 \pm 0 . 4 2$ </td></tr><tr><td>Last 2 (24, 27)</td><td> ${ \bf 7 8 . 0 5 \pm 0 . 5 2 }$ </td></tr><tr><td>Last 3 (21, 24, 27)</td><td> $7 7 . 7 6 \pm 0 . 3 7$ </td></tr><tr><td>Last 4 (18, 21, 24, 27)</td><td> $7 8 . 0 5 \pm 0 . 3 9$ </td></tr><tr><td>Chunk alignment</td><td></td></tr><tr><td>Correct alignment (Last 2)</td><td> $7 8 . 0 5 \pm 0 . 5 2$ </td></tr><tr><td>Cyclic permutation</td><td> $7 7 . 6 2 \pm 0 . 4 4$ </td></tr></table>

None disables CKR; other settings share the rank-32 router.

## 4.3.1 Component ablation

Table 2 shows that AMS raises the mean by 1.47 points to 77.59, while CKR alone reaches 76.68, improving over InSeNT by 0.56 points. When added to AMS, CKR contributes a further 0.79-point gain and brings the combined model to 78.38. The joint improvement of 2.26 points is 0.23 points larger than the sum of the two isolated gains, indicating a modest positive interaction rather than simple redundancy. The combined configuration also has the smallest reported standard deviation (0.23) among the four settings. Thus, both components improve retrieval individually, and their combination performs best without increasing the observed across-run variability.

These results are consistent with the distinct roles suggested by our analysis: AMS controls total optimization efort and, at fixed logits, concentrates negative gradient weight on harder siblings, whereas CKR redistributes attention across chunks during encoding. In the population-wide training diagnostics reported in Appendix C.2, AMS raises the query-row mean calibration efort $\begin{array} { r } { \dot { \overline { { W } } } = N ^ { - 1 } \sum _ { i = 1 } ^ { N } } \end{array}$ �<sub>�</sub> over the � recorded training rows from 0.374 to 0.535, a 43.1% increase, further supporting its optimization-side role.

## 4.3.2 CKR design analysis

Table 3 compares layer coverage and chunk alignment within the AMS-enabled design sweep. Relative to the no-CKR reference at 77.60, routing the last one, two, three, and four selected layers changes the mean to 77.92, 78.05, 77.76, and 78.05, respectively. The non-monotonic pattern shows that increasing the number of routed layers alone does not guarantee a better result. Last 2 and Last 4 attain the same mean, but Last 2 uses fewer routing layers; neither Last 1 nor Last 3 surpasses it. We therefore use Last 2 as the compact default.

Normally, the representation $z _ { c }$ aggregated from chunk � routes the Keys of that same chunk. Cyclically permuting chunk representations within a document breaks this correspondence while retaining the model parameters and other settings. This controlled mismatch reduces the mean from 78.05 to 77.62, a 0.43-point drop. The result indicates that the gain depends not only on adding a routing module, but also on preserving the correspondence between each chunk representation and the chunk whose Keys it modifies.

![](images/49584a2e77d8d1c6317420d6daceb98659a692dcd248bf0f2aeb7d20fd18a67f.jpg)  
Figure 3: Query margins and adaptive scales during DcLBC training. Solid: example queries; dashed: training means.

## 4.4 AMS training dynamics

We track the gold–hardest-sibling margin $m _ { i }$ and adaptive scale $\kappa _ { i }$ during two epochs of DcLBC training with ModernBERTlarge. Figure 3 compares four example queries, measured every ten updates in evaluation mode using the current training EMA, with query-weighted training-batch means.

Q1 and Q2 retain smaller margins and scales above one, whereas Q3’s scale settles at one and Q4’s fluctuates near this lower bound. This behavior is consistent with Eq. (5): AMS assigns stronger scaling to queries with smaller margins relative to the evolving reference $\mu _ { t }$ . Meanwhile, the mean margin increases, suggesting improved average gold–negative separation, and the mean scale stabilizes near 1.44. These trends are compatible because the reference evolves with training: overall separation can improve while relatively dificult queries retain stronger scaling. These trajectories illustrate AMS’s adaptive response; at fixed logits, stronger scaling concentrates negative gradient weight on harder siblings.

## 5 Conclusion

We proposed DcLBC for chunk-level retrieval by combining adaptive margin scaling with chunk-aware context routing. On ConTEB, DcLBC consistently outperforms InSeNT across the evaluated encoder and multi-vector backbones, with gains of up to 2.45 NDCG@10 points. Both ModernBERT variants also improve on NanoBEIR, indicating that stronger contextual retrieval need not compromise passage-level performance. Component ablations and CKR design analyses further support the complementary roles of AMS and CKR in calibrating local chunk boundaries.

## References

[1] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih, “Dense passage retrieval for open-domain question answering,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020, pp. 6769–6781, Association for Computational Linguistics.

[2] Max Conti, Manuel Faysse, Gautier Viaud, Antoine Bosselut, Céline Hudelot, and Pierre Colombo, “Context is gold to find the gold passage: Evaluating and training contextual document embeddings,” in Proc. EMNLP. 2025, pp. 22594–22608, Association for Computational Linguistics.

[3] Anthropic, “Contextual retrieval in AI systems,” Anthropic Engineering, September 2024.

[4] Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, and Han Xiao, “Late chunking: Contextual chunk embeddings using long-context embedding models,” arXiv preprint arXiv:2409.04701, 2024.

[5] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning, “RAPTOR: Recursive abstractive processing for tree-organized retrieval,” in Proc. ICLR. 2024, OpenReview.net.

[6] Kun Luo, Zheng Liu, Shitao Xiao, Tong Zhou, Yubo Chen, Jun Zhao, and Kang Liu, “Landmark embedding: A chunking-free embedding method for retrieval augmented long-context large language models,” in Proc. ACL (Long Papers). 2024, pp. 3268–3281, Association for Computational Linguistics.

[7] John Xavier Morris and Alexander M. Rush, “Contextual document embeddings,” in Proc. ICLR. 2025, OpenReview.net.

[8] Junjie Wu, Jiangnan Li, Yuqing Li, Lemao Liu, Liyan Xu, Jiwei Li, Dit-Yan Yeung, Jie Zhou, and Mo Yu, “Situated embedding models for context-aware dense retrieval,” in Proc. ACL (Short Papers). 2026, pp. 37–49, Association for Computational Linguistics.

[9] Iz Beltagy, Matthew E. Peters, and Arman Cohan, “Longformer: The long-document transformer,” arXiv preprint arXiv:2004.05150, 2020.

[10] Benjamin Warner, Antoine Chafin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Grifin Adams, Jeremy Howard, and Iacopo Poli, “Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory eficient, and long context finetuning and inference,” arXiv preprint arXiv:2412.13663, 2024.

[11] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave, “Unsupervised dense information retrieval with contrastive learning,” CoRR, vol. abs/2112.09118, 2021.

[12] Omar Khattab and Matei Zaharia, “ColBERT: Eficient and efective passage search via contextualized late interaction over BERT,” in Proc. ACM SIGIR, 2020, pp. 39–48.

[13] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia, “ColBERTv2: Efective and eficient retrieval via lightweight late interaction,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2022, pp. 3715–3734, Association for Computational Linguistics.

[14] Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan, “Repetition improves language model embeddings,” in Proc. ICLR. 2025, OpenReview.net.

[15] Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy, “LLM2Vec: Large language models are secretly powerful text encoders,” CoRR, vol. abs/2404.05961, 2024.

[16] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping, “NV-Embed: Improved techniques for training LLMs as generalist embedding models,” in Proc. ICLR. 2025, OpenReview.net.

[17] Tianyu Gao, Xingcheng Yao, and Danqi Chen, “SimCSE: Simple contrastive learning of sentence embeddings,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021, pp. 6894–6910, Association for Computational Linguistics.

[18] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk, “Approximate nearest neighbor negative contrastive learning for dense text retrieval,” in International Conference on Learning Representations, 2021.

[19] Feng Wang and Huaping Liu, “Understanding the behaviour ofcontrastive loss,” in Proc. IEEE/CVF CVPR, 2021, pp. 2495–2504.

[20] Chaoning Zhang, Kang Zhang, Trung X. Pham, Axi Niu, Zhinan Qiao, Chang D. Yoo, and In So Kweon, “Dual temperature helps contrastive learning without many negative samples: Towards understanding and simplifying MoCo,” in Proc. IEEE/CVF CVPR, 2022, pp. 14441–14450.

[21] Zizheng Huang, Haoxing Chen, Ziqi Wen, Chao Zhang, Huaxiong Li, Bo Wang, and Chunlin Chen, “Model-aware contrastive learning: Towards escaping the dilemmas,” in Proc. ICML. 2023, vol. 202 of Proceedings ofMachine Learning Research, pp. 13774–13790, PMLR.

[22] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford, “Okapi at TREC-3,” in Proc. TREC-3, 1994, pp. 109–126.

[23] Antoine Chafin, “GTE-ModernColBERT-v1 model card,” Hugging Face model repository, 2025.

[24] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou, “Qwen3 Embedding: Advancing text embedding and reranking through foundation models,” arXiv preprint arXiv:2506.05176, 2025.

[25] Microsoft, “Harrier OSS v1 270M model card,” Hugging Face model repository, 2026.

[26] CodeFuse AI, “ML-Embed-0.6B model card,” Hugging Face model repository, 2026.

[27] Sedigheh Eslami, Maksim Gaiduk, Markus Krimmel, Louis Milliken, Bo Wang, and Denis Bykov, “Difusion-pretrained dense and contextual embeddings,” arXiv preprint arXiv:2602.11151, 2026.

[28] Nicolas Boizard, Théo Deschamps-Berger, Hippolyte Gisserot-Boukhlef, Céline Hudelot, and Pierre Colombo, “BidirLM: From text to omnimodal bidirectional encoders by adapting and composing causal LLMs,” arXiv preprint arXiv:2604.02045, 2026.

[29] KiteFish AI, “Nano-Em1-0.6B-v2 model card,” Hugging Face model repository, 2026.

[30] AI Sage, “Giga Embeddings Instruct 480M model card,” Hugging Face model repository, 2026.

## Appendix Contents

A Supplementary Proofs . . . . 8   
A.1 Row-Level Properties of In-Sequence Negative Train  
ing. 8   
A.2 Adaptive Margin Scaling: Scale Calibration, Efort   
Decomposition, and Entropy Monotonicity . 8   
A.3 Chunk-Aware Key Routing: Exact Attention Factor  
ization . . 9   
A.4 Boundary-Aligned Geometric Work . 10   
B Implementation Details . . 10

## A Supplementary Proofs

This section provides supplementary proofs for Section 3.

## A.1 Row-Level Properties of In-Sequence Negative Training

Proposition A.1 (Posterior Log-Odds and Score-Space Gradient of InSeNT; Eq. (3)). For normalized query $q _ { i } \in \mathbb { R } ^ { d }$ with document chunks $C _ { i } = [ c _ { i 1 } , . . . , c _ { i n _ { i } } ] ^ { \intercal } \in \mathbb { R } ^ { n _ { i } \times d }$ , gold index $g _ { i } ,$ , similarity scale $a > 0 ;$ , and temperature $\tau > 0 ,$ , let $s _ { i j } =$ $( a / \tau ) q _ { i } ^ { \top } c _ { i j }$ and $h _ { i } \in$ arg max $\_ f \neq _ { g _ { i } } s _ { i j } .$ . Define $m _ { i } = s _ { i g _ { i } } - s _ { i h }$ and $p _ { i j } = \exp ( s _ { i j } ) / { \sum _ { k = 1 } ^ { n _ { i } } \exp ( s _ { i k } ) }$ . Then:

$$
\log \frac { p _ { i g _ { i } } } { p _ { i h _ { i } } } = m _ { i } , \qquad \frac { \partial \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } } { \partial s _ { i j } } = p _ { i j } - \mathbb { I } [ j = g _ { i } ] .\tag{A.1}
$$

Proof. Let $\begin{array} { r } { Z _ { i } = \sum _ { k = 1 } ^ { n _ { i } } \exp ( s _ { i k } ) } \end{array}$ denote the within-document partition function, so that $p _ { i j } = \exp ( s _ { i j } ) / Z _ { i }$ . Forming the ratio between the gold probability $p _ { i g _ { i } }$ and the hardest-sibling probability $p _ { i h _ { i } }$ , the shared normalizer $Z _ { i }$ cancels:

$$
\log \frac { p _ { i g _ { i } } } { p _ { i h _ { i } } } = \log \frac { \exp ( s _ { i g _ { i } } ) / Z _ { i } } { \exp ( s _ { i h _ { i } } ) / Z _ { i } } = s _ { i g _ { i } } - s _ { i h _ { i } } = m _ { i } .\tag{A.2}
$$

Next, write the sequence loss in Eq. (1) as $\mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } = - \log p _ { i g _ { i } } =$ $- s _ { i g _ { i } }$ + log $Z _ { i }$ . Diferentiating with respect to each scaled logit $s _ { i j }$ yields

$$
\begin{array} { l } { \displaystyle \frac { \partial \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } } { \partial s _ { i j } } = - \mathbb { I } \big [ j = g _ { i } \big ] + \frac { 1 } { Z _ { i } } \frac { \partial Z _ { i } } { \partial s _ { i j } } } \\ { \displaystyle = - \mathbb { I } \big [ j = g _ { i } \big ] + \frac { \exp ( s _ { i j } ) } { Z _ { i } } } \\ { \displaystyle = p _ { i j } - \mathbb { I } \big [ j = g _ { i } \big ] . } \end{array}\tag{A.3}
$$

Because $p _ { i g _ { i } } / p _ { i h _ { i } } = \exp ( m _ { i } )$ , a smaller margin $m _ { i }$ directly corresponds to weaker posterior discrimination between the gold chunk and its strongest same-document competitor. □

Proposition A.2 (Row Shift Invariance and Zero-Sum Relative Operator). For a single query �, let $\mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } ( s _ { i } ) = - s _ { i g _ { i } } +$ log $\textstyle \sum _ { j = 1 } ^ { n _ { i } } \exp ( s _ { i j } )$ , and let ${ \mathbf { 1 } } _ { n _ { i } }$ be the length-�<sub>�</sub> all-ones vector.

C Boundary and Optimization Diagnostics . . . . . . 10   
C.1 Competition-Stratified Retrieval Analysis. 10   
C.2 Optimization Dynamics . 11   
D Robustness and Baseline Comparisons 11   
D.1 Temperature Baseline Comparisons 11   
E Additional Encoder Evidence 11   
E.1 Controlled-3 Context-Necessity Subset and Sub-1B   
LLM Encoders . 11

For any scalar shift � ∈ R, $\begin{array} { r } { \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } ( s _ { i } + b \mathbf { 1 } _ { n _ { i } } ) = \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } ( s _ { i } ) } \end{array}$ , and the score-space gradient satisfies

$$
\sum _ { j = 1 } ^ { n _ { i } } \frac { \partial \mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) } } { \partial s _ { i j } } = 0 .\tag{A.4}
$$

Consequently, the sequence loss acts purely on relative differences among same-document chunks and is invariant to document-wide score translations.

Proof. Replacing $s _ { i j }$ with $s _ { i j } + b$ multiplies both numerator and denominator inside the logarithm by $e ^ { b }$ , leaving $\mathcal { L } _ { \mathrm { s e q } } ^ { ( i ) }$ unchanged. Summing Proposition A.1 over $j \in \{ 1 , \ldots , n _ { i } \}$ gives $\begin{array} { r } { \sum _ { j = 1 } ^ { n _ { i } } ( p _ { i j } - \mathbb { I } [ j = g _ { i } ] ) = \sum _ { j = 1 } ^ { n _ { i } } p _ { i j } - 1 = 0 . } \end{array}$ □

## A.2 Adaptive Margin Scaling: Scale Calibration, Efort Decomposition, and Entropy Monotonicity

Proposition A.3 (Constrained Margin-Scale Calibration; Eqs. $\left( 4 \right) - \left( 5 \right) )$ . Given reference scale $\mu _ { t } \ > \ 0 ,$ , margin floor $\epsilon > 0 ,$ , and truncated margin $\widetilde { m } _ { i } \ = \ \operatorname* { m a x } ( m _ { i } , \epsilon ) \ > \ 0 ,$ , the box-constrained quadratic calibration problem

$$
\kappa _ { i } = \underset { \kappa \in [ 1 , K ] } { \arg \operatorname* { m i n } } ( \kappa \widetilde { m } _ { i } - \mu _ { t } ) ^ { 2 }\tag{A.5}
$$

admits the unique closed-form solution $\kappa _ { i } = \mathrm { c l i p } ( \mu _ { t } / \widetilde { m } _ { i } , 1 , K )$ Furthermore, whenever $m _ { i } ~ \geq ~ \epsilon$ and $\mu _ { t } / K \le m _ { i } \le \mu _ { t }$ , the calibrated posterior log-odds between the gold chunk and the hardest sibling satisfies $\rho _ { i } ( \kappa _ { i } ) = \kappa _ { i } m _ { i } = \mu _ { t }$

Proof. Because $\widetilde { m } _ { i } ~ > ~ 0$ , the objective $f ( \kappa ) ~ = ~ ( { \widetilde m } _ { i } ) ^ { 2 } ( \kappa ~ -$ $\mu _ { t } / \widetilde { m } _ { i } ) ^ { 2 }$ is strictly convex in � with unconstrained minimizer $\kappa ^ { \star } = \mu _ { t } / \widetilde m _ { i }$ . Projecting $\kappa ^ { \star }$ onto the closed interval [1, �] yields � = min(max $( \mu _ { t } / \widetilde { m } _ { i } , 1 ) , K ) = \mathrm { c l i p } ( \mu _ { t } / \widetilde { m } _ { i } , 1 , K )$ . Under scaled logits $\boldsymbol { K } _ { i } \boldsymbol { s } _ { i j }$ , the posterior log-odds is

$$
\rho _ { i } ( \kappa _ { i } ) = \log \frac { p _ { i g _ { i } } ^ { ( \kappa ) } } { p _ { i h _ { i } } ^ { ( \kappa ) } } = \kappa _ { i } s _ { i g _ { i } } - \kappa _ { i } s _ { i h _ { i } } = \kappa _ { i } m _ { i } .\tag{A.6}
$$

If $m _ { i } \geq \epsilon$ and $\mu _ { t } / K \le m _ { i } \le \mu _ { t }$ , then $\widetilde { m } _ { i } = m _ { i }$ and $\mu _ { t } / \widetilde { m } _ { i } \in$ $[ 1 , K ] ,$ , so $\kappa _ { i } = \mu _ { t } / m _ { i }$ and $\rho _ { i } ( \kappa _ { i } ) = \mu _ { t }$ . For easy instances $( m _ { i } > \mu _ { t } ) , \kappa _ { i } = 1$ preserves the baseline scale; for unresolved or inverted margins $( m _ { i } < \mu _ { t } / K ) , \kappa _ { i } = K$ caps the maximum sharpening multiplier. □

Proposition A.4 (Optimization Efort and Conditional Negative Allocation; Eq. (8)). For a detached row multiplier $\kappa _ { i } > 0$ , let $\begin{array} { r } { p _ { i j } ^ { ( \kappa ) } = \exp ( \kappa _ { i } s _ { i j } ) / { \sum _ { k = 1 } ^ { n _ { i } } \exp ( \kappa _ { i } s _ { i k } ) } , \ W _ { i } = \kappa _ { i } ( 1 - p _ { i g _ { i } } ^ { ( \kappa ) } ) } \end{array}$ , and $r _ { i j } = p _ { i j } ^ { ( \kappa ) } / ( 1 - p _ { i g _ { i } } ^ { ( \kappa ) } ) f o r j \ne g _ { i }$ . Define $\mathcal { L } _ { \mathrm { D c L B C } } ^ { ( i ) } = - \log p _ { i g _ { i } } ^ { ( \kappa ) }$ and $\nabla \overset { \smile } { = } \nabla _ { \theta }$ with respect to trainable parameters �, holding $\kappa _ { i } ~ { f i x e d . }$ Then $r _ { i }$ is a probability distribution over sibling negatives, and

$$
- \nabla \mathcal { L } _ { \mathrm { D c L B C } } ^ { ( i ) } = W _ { i } \left( \nabla s _ { i g _ { i } } - \sum _ { j \neq g _ { i } } r _ { i j } \nabla s _ { i j } \right) .\tag{A.7}
$$

Proof. Every $p _ { i j } ^ { ( \kappa ) }$ is positive, and the probabilities sum to one. Hence $\begin{array} { r } { \sum _ { j \neq g _ { i } } p _ { i j } ^ { ( \kappa ) } = 1 - p _ { i g _ { i } } ^ { ( \kappa ) } > 0 } \end{array}$ , which gives $r _ { i j } > 0$ and $\begin{array} { r } { \sum _ { j \neq g _ { i } } r _ { i j } = 1 } \end{array}$ . Because $\kappa _ { i }$ is detached in the computation graph, applying the chain rule to $\mathcal { L } _ { \mathrm { D c L B C } } ^ { ( i ) } = - \kappa _ { i } s _ { i g _ { i } } +$ log $\begin{array} { r } { \sum _ { j = 1 } ^ { n _ { i } } \exp ( \kappa _ { i } s _ { i j } ) } \end{array}$ yields

$$
\begin{array} { r l } { - \nabla \mathcal { L } _ { \mathrm { D C L B C } } ^ { ( i ) } = - \displaystyle \sum _ { j = 1 } ^ { n _ { i } } K _ { i } \left( p _ { i j } ^ { ( \kappa ) } - \mathbb { I } [ j = g _ { i } ] \right) \nabla s _ { i j } } & { } \\ { = \kappa _ { i } \left( 1 - p _ { i g _ { i } } ^ { ( \kappa ) } \right) \nabla s _ { i g _ { i } } - \displaystyle \sum _ { j \neq g _ { i } } \kappa _ { i p } ( { \bf { \Sigma } } _ { i j } ^ { \kappa } \nabla s _ { i j } } & { } \\ { = W _ { i } \nabla s _ { i g _ { i } } - W _ { i } \displaystyle \sum _ { j \neq g _ { i } } \frac { p _ { i j } ^ { ( \kappa ) } } { 1 - p _ { i g _ { i } } ^ { ( \kappa ) } } \nabla s _ { i j } } & { } \\ { = W _ { i } \left( \nabla s _ { i g _ { i } } - \displaystyle \sum _ { j \neq g _ { i } } r _ { i j } \nabla s _ { i j } \right) . } & { } \end{array}\tag{A.8}
$$

Thus $W _ { i }$ governs the total magnitude of the score-space update, while $r _ { i }$ governs how repulsive efort is distributed across same-document negative chunks. □

Proposition A.5 (Monotonic Decrease of Conditional Competitor Entropy; Eq. (9)). At fixed logits $\{ s _ { i j } \} _ { j \neq g _ { i } }$ , let $H ( r _ { i } ) =$ $\begin{array} { r } { - \sum _ { j \neq g _ { i } } r _ { i j } \log r _ { i j } } \end{array}$ be the Shannon entropy ofthe conditional negative allocation $r _ { i } ( \boldsymbol { \kappa } )$ . Then

$$
\frac { \mathrm { d } H ( r _ { i } ) } { \mathrm { d } \kappa } = - \kappa \mathrm { V a r } _ { r _ { i } } ( s _ { i j } ) \leq 0 ,\tag{A.9}
$$

with strict inequality whenever the sibling negative logits $\{ s _ { i j } \} _ { j \neq g _ { i } }$ are not all identical. Moreover, for any pair of sibling negatives $j , k \neq g _ { i }$ with $s _ { i j } > s _ { i k }$ , the relative weight ratio $r _ { i j } / r _ { i k } = \exp ( \kappa ( s _ { i j } - s _ { i k } ) )$ grows strictly with �.

Proof. Define the negative-only log-partition function $\Phi _ { i } ( \kappa ) =$ log $\begin{array} { r } { \sum _ { j \neq g _ { i } } \exp ( \kappa s _ { i j } ) } \end{array}$ . Because the gold term cancels when conditioning on $j ~ \neq ~ g _ { i }$ , the conditional distribution is an exponential family in canonical parameter �:

$$
r _ { i j } ( \kappa ) = \frac { \exp ( \kappa s _ { i j } ) } { \sum _ { k \neq g _ { i } } \exp ( \kappa s _ { i k } ) } = \exp \left( \kappa s _ { i j } - \Phi _ { i } ( \kappa ) \right) .\tag{A.10}
$$

Standard cumulant-generating properties of exponential families give the first and second derivatives of $\Phi _ { i } ( \boldsymbol { \kappa } )$

$$
\begin{array} { r l } & { \Phi _ { i } ^ { \prime } ( \kappa ) = \displaystyle \sum _ { j \neq g _ { i } } r _ { i j } s _ { i j } = \mathbb { E } _ { r _ { i } } \big [ s _ { i j } \big ] , } \\ & { \Phi _ { i } ^ { \prime \prime } ( \kappa ) = \displaystyle \sum _ { j \neq g _ { i } } r _ { i j } \left( s _ { i j } - \mathbb { E } _ { r _ { i } } \big [ s _ { i j } \big ] \right) ^ { 2 } } \\ & { \qquad = \mathrm { V a r } _ { r _ { i } } \big ( s _ { i j } \big ) . } \end{array}\tag{A.11}
$$

Substituting log $r _ { i j } = \kappa s _ { i j } - \Phi _ { i } ( \boldsymbol { \kappa } )$ into the Shannon entropy definition yields the Legendre relation

$$
H ( \boldsymbol { r } _ { i } ) = - \sum _ { j \neq g _ { i } } r _ { i j } \left( \kappa s _ { i j } - \Phi _ { i } ( \boldsymbol { \kappa } ) \right) = \Phi _ { i } ( \boldsymbol { \kappa } ) - \kappa \Phi _ { i } ^ { \prime } ( \boldsymbol { \kappa } ) .\tag{A.12}
$$

Diferentiating with respect to � via the product rule cancels the $\Phi _ { i } ^ { \prime } ( \kappa )$ terms:

$$
\frac { \mathrm { d } H ( r _ { i } ) } { \mathrm { d } \kappa } = \Phi _ { i } ^ { \prime } ( \kappa ) - \Phi _ { i } ^ { \prime } ( \kappa ) - \kappa \Phi _ { i } ^ { \prime \prime } ( \kappa ) = - \kappa \mathrm { V a r } _ { r _ { i } } ( s _ { i j } ) \leq 0 .\tag{A.13}
$$

For $\kappa > 0 , \nabla \mathrm { a r } _ { r _ { i } } ( s _ { i j } ) > 0$ whenever the non-gold scores are not identical, establishing strict monotonicity. □

## A.3 Chunk-Aware Key Routing: Exact Attention Factor ization

Proposition A.6 (Post-RoPE Chunk-Mass Factorization and Within-Chunk Ratio Invariance; Eqs. (11)–(12)). Consider a single attention head with post-RoPE query $q _ { s } \in \mathbb { R } ^ { d _ { h } }$ at source token � and post-RoPE keys $\boldsymbol { k } _ { t } \in \mathbb { R } ^ { d _ { h } }$ at target tokens �. Suppose CKR shifts keys inside each routed chunk � by a chunk-shared ofset $k _ { t } ^ { \prime } = k _ { t } + U _ { k } z _ { c } \ ( t \in c ) ,$ , while leaving special or unassigned tokens $( t \in c _ { 0 } )$ unrouted $( k _ { t } ^ { \prime } = k _ { t } )$ . Let $\ell _ { s t } = q _ { s } ^ { \top } k _ { t } / \sqrt { d _ { h } } , a _ { s t } = \mathrm { s o f t m a x } _ { t } ( \ell _ { s t } )$ , and $\textstyle A _ { s c } = \sum _ { t \in c } a _ { s t }$ Then under CKR:

1. The routed chunk attention mass satisfies

$$
A _ { s c } ^ { \prime } = \frac { e ^ { b _ { s c } } A _ { s c } } { A _ { s 0 } + \sum _ { c ^ { \prime } } e ^ { b _ { s c ^ { \prime } } } A _ { s c ^ { \prime } } } , \qquad b _ { s c } = \frac { q _ { s } ^ { \top } U _ { k } z _ { c } } { \sqrt { d _ { h } } } .\tag{A.14}
$$

2. For any chunk � with $A _ { s c } > 0$ and any token $t \in c ,$ the conditional within-chunk token weight is invariant:

$$
\begin{array} { l } { \displaystyle \frac { a _ { s t } ^ { \prime } } { A _ { s c } ^ { \prime } } = \frac { a _ { s t } } { A _ { s c } } , } \\ { \displaystyle \frac { a _ { s t } ^ { \prime } } { a _ { s u } ^ { \prime } } = \frac { a _ { s t } } { a _ { s u } } , \qquad t , u \in { c } . } \end{array}\tag{A.15}
$$

Proof. For any target token $t \in c$ , linearity of the inner product after RoPE gives

$$
\ell _ { s t } ^ { \prime } = \frac { q _ { s } ^ { \top } ( k _ { t } + U _ { k } z _ { c } ) } { \sqrt { d _ { h } } } = \frac { q _ { s } ^ { \top } k _ { t } } { \sqrt { d _ { h } } } + \frac { q _ { s } ^ { \top } U _ { k } z _ { c } } { \sqrt { d _ { h } } } = \ell _ { s t } + b _ { s c } .\tag{A.16}
$$

Let $\begin{array} { r } { Z _ { s } = \sum _ { \nu } \exp ( \ell _ { s \nu } ) } \end{array}$ be the original row normalizer, where � ranges over valid target tokens. Then $\begin{array} { r } { Z _ { s c } = \sum _ { t \in c } \exp ( \ell _ { s t } ) = } \end{array}$

$Z _ { s } A _ { s c }$ and $\begin{array} { r } { Z _ { s 0 } = \sum _ { t \in c _ { 0 } } \exp ( \ell _ { s t } ) = Z _ { s } A _ { s 0 } } \end{array}$ . Under the routed logits $\ell _ { s t } ^ { \prime } .$ , the unnormalized chunk sum factors as

$$
\begin{array} { l } { \displaystyle Z _ { s c } ^ { \prime } = \sum _ { t \in c } \exp ( \ell _ { s t } + b _ { s c } ) } \\ { = \displaystyle e ^ { b _ { s c } } \sum _ { t \in c } \exp ( \ell _ { s t } ) } \\ { = \displaystyle e ^ { b _ { s c } } Z _ { s c } = Z _ { s } e ^ { b _ { s c } } A _ { s c } , } \end{array}\tag{A.17}
$$

while $Z _ { s 0 } ^ { \prime } = Z _ { s 0 } = Z _ { s } A _ { s 0 }$ . Dividing $Z _ { s c } ^ { \prime }$ by the new total normalizer $\begin{array} { r } { Z _ { s } ^ { \prime } = Z _ { s 0 } ^ { \prime } + \sum _ { c ^ { \prime } } Z _ { s c ^ { \prime } } ^ { \prime } } \end{array}$ and canceling the common positive factor $Z _ { s }$ yields the chunk-mass formula in Claim 1. Finally, for any $t \in c$ with $A _ { s c } > 0$

$$
\frac { a _ { s t } ^ { \prime } } { A _ { s c } ^ { \prime } } = \frac { \exp ( \ell _ { s t } ^ { \prime } ) / Z _ { s } ^ { \prime } } { Z _ { s c } ^ { \prime } / Z _ { s } ^ { \prime } } = \frac { e ^ { b _ { s c } } \exp ( \ell _ { s t } ) } { e ^ { b _ { s c } } Z _ { s c } } = \frac { \exp ( \ell _ { s t } ) / Z _ { s } } { Z _ { s c } / Z _ { s } } = \frac { a _ { s t } } { A _ { s c } } ,\tag{A.18}
$$

which establishes Claim 2. Note that applying the translation after rotary position embedding is essential: if $U _ { k } z _ { c }$ were added before position-dependent rotation $\mathcal { R } _ { t }$ , the efective bias $q _ { s } ^ { \top } \mathcal { R } _ { t } ( U _ { k } z _ { c } ) / \sqrt { d _ { h } }$ would vary with token position �, breaking within-chunk ratio invariance. □

## A.4 Boundary-Aligned Geometric Work

Proposition A.7 (Boundary-Aligned Work Identity and Magnitude Bound). Let $C _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ contain the normalized chunk embeddings ofdocument $i ,$ let $q _ { i } \in \mathbb { R } ^ { d }$ satisfy $\| q _ { i } \| _ { 2 } = 1$ , and let $\boldsymbol { e } _ { j } \in \mathbb { R } ^ { n _ { i } }$ denote the �th standard basis vector. Define $\nu _ { i } = e _ { g _ { i } } - e _ { h _ { i } }$ . For updated embeddings $C _ { i } ^ { \prime }$ and displacement $\Delta C _ { i } = C _ { i } ^ { \prime } - C _ { i }$ at fixed $( q _ { i } , g _ { i } , h _ { i } )$ , the cosine-margin change $\Delta m _ { i } ^ { \mathrm { c o s } } = \dot { ( } c _ { i g _ { i } } ^ { \prime } - c _ { i h _ { i } } ^ { \prime } ) ^ { \top } q _ { i } - ( c _ { i g _ { i } } - c _ { i h _ { i } } ) ^ { \top } q _ { i }$ satisfies

$$
\begin{array} { r l } & { \Delta m _ { i } ^ { \mathrm { c o s } } = \nu _ { i } ^ { \top } \Delta C _ { i } q _ { i } } \\ & { \quad \quad = \left. \Delta C _ { i } , \nu _ { i } q _ { i } ^ { \top } \right. _ { F } , } \\ & { \left| \Delta m _ { i } ^ { \mathrm { c o s } } \right| \leq \sqrt { 2 } \left\| \Delta C _ { i } \right\| _ { F } . } \end{array}\tag{A.19}
$$

Proof. Writing the cosine margin as $m _ { i } ^ { \mathrm { c o s } } = \nu _ { i } ^ { \top } C _ { i } q _ { i }$ , linearity yields $\Delta m _ { i } ^ { \mathrm { c o s } } = \nu _ { i } ^ { \top } \Delta C _ { i } q _ { i } = \mathrm { t r } ( ( \nu _ { i } q _ { i } ^ { \top } ) ^ { \top } \Delta C _ { i } ) \stackrel { \cdot } { = } \langle \Delta C _ { i } , \nu _ { i } q _ { i } ^ { \top } \rangle _ { F }$ Applying the Frobenius Cauchy–Schwarz inequality gives

$$
\begin{array} { r l } & { \left| \Delta m _ { i } ^ { \mathrm { c o s } } \right| \leq \| \Delta C _ { i } \| _ { F } \left\| \nu _ { i } q _ { i } ^ { \top } \right\| _ { F } } \\ & { \qquad = \| \Delta C _ { i } \| _ { F } \left\| \nu _ { i } \right\| _ { 2 } \| q _ { i } \| _ { 2 } } \\ & { \qquad = \sqrt { 2 } \left\| \Delta C _ { i } \right\| _ { F } , } \end{array}\tag{A.20}
$$

since $\| e _ { g _ { i } } - e _ { h _ { i } } \| _ { 2 } = \sqrt { 2 }$ for $g _ { i } \neq h _ { i }$ and $\| q _ { i } \| _ { 2 } = 1$ . Moreover, because $\nu _ { i } ^ { \top } \mathbf { 1 } _ { n _ { i } } = 0 .$ any common shift $\boldsymbol { \delta } \in \mathbb { R } ^ { d }$ applied as $\Delta C _ { i } =$ $\mathbf { 1 } _ { n _ { i } } \delta ^ { \top }$ contributes zero to $\Delta m _ { i } ^ { \mathrm { c o s } }$ . Thus retrieval improvement depends on signed alignment with the rank-one boundary operator $\nu _ { i } q _ { i } ^ { \intercal }$ rather than raw representation movement or global cluster expansion. □

## B Implementation Details

All ModernBERT-large variants use the same pre-tokenized training documents, data order, and evaluation pipeline. Documents are capped at 8,192 tokens and encoded once before Late

Table C.1: NDCG@10 gains in percentage points (pp) by baseline InSeNT margin quartile. $Q _ { 1 }$ denotes the highest competition queries.
<table><tr><td>Comparison</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td></tr><tr><td>AMS – InSeNT</td><td>+4.14</td><td>+2.01</td><td>-1.00</td><td>-0.09</td></tr><tr><td>CKR increment</td><td>+1.32</td><td>+0.56</td><td>+0.33</td><td>+0.32</td></tr><tr><td>DcLBC – InSeNT</td><td>+5.46</td><td>+2.57</td><td>-0.67</td><td>+0.23</td></tr></table>

Chunking mean pooling. Post-training runs for two epochs with a global batch of 64 in BF16. We use AdamW with a peak learning rate of $2 \times 1 0 ^ { - 4 }$ , cosine decay, a short warmup, and no weight decay. The inter-document and within-document objectives are combined with $\lambda _ { \mathrm { s e q } } = 0 . 1$ ; the similarity scale is $a = 2 0$ , and the within-document temperature is 1.0 for InSeNT and 0.8 for AMS-based variants.

AMS uses $K = 2$ , margin floor $\epsilon = 1 0 ^ { - 4 }$ , and EMA decay $\gamma = 0 . 9 9$ . Let $\mathcal { B } _ { t }$ be the query rows local to the device at the �th within-document loss call and

$$
\bar { m } _ { t } = \frac { 1 } { \vert \mathcal { B } _ { t } \vert } \sum _ { i \in \mathcal { B } _ { t } } \operatorname* { m a x } ( m _ { i } , \epsilon ) .\tag{B.1}
$$

The first call sets $\mu _ { 0 } = { \bar { m } } _ { 0 } ;$ subsequent calls update

$$
\mu _ { t } = \gamma \mu _ { t - 1 } + ( 1 - \gamma ) \bar { m } _ { t } .\tag{B.2}
$$

This EMA is not synchronized across devices, and both $\mu _ { t }$ and $\kappa _ { i }$ are detached from gradient computation. CKR is applied to the final two global-attention layers through one shared rank-32 bottleneck. Its bias-free Key up-projection is zeroinitialized, so post-training begins from the pretrained attention distribution.

The resulting chunk vectors are L2-normalized and retrieved with a single cosine score. AMS changes only the training loss and therefore introduces no inference-time operation. CKR adds only the shared bottleneck while leaving the embedding dimension and retrieval procedure unchanged.

## C Boundary and Optimization Diagnostics

## C.1 Competition-Stratified Retrieval Analysis

We partition queries within each ConTEB dataset into quartiles ordered by the baseline InSeNT gold–hardest-sibling margin. The first quartile contains the most dificult local competitions. Table C.1 gives the compact numerical summary, while Figure C.1 shows both the concentration of gains in this regime and the query-level association between boundary-margin changes and retrieval changes.

AMS contributes most strongly in $Q _ { 1 }$ and $Q _ { 2 }$ , where the baseline gold–sibling margin is smallest, rather than uniformly shifting all retrieval scores. CKR adds a positive increment in every quartile, including those for which the AMS-only change is small or negative. Their profiles are therefore complementary: AMS supplies the largest dificult-regime correction, whereas CKR provides a smaller but broader refinement. Figure C.1 also shows dataset-level uncertainty and the centered association between changes in boundary margin and retrieval quality.

![](images/2d232ea73216bf3d0092693f9b7e63380da834c0344954b61d3a7824ddc4a225.jpg)

![](images/6493e6a6ce9ff6a6bf7d25f0213c97533eb1e9874a12bc7c8ec7ac1835368a00.jpg)  
Figure C.1: Boundary diagnostics. (a) AMS and full DcLBC gains by baseline margin quartile; connectors show the CKR increment and error bars show the dataset-level standard error of the mean (SEM). (b) Within-dataset centered margin and NDCG@10 changes, aggregated into equal-frequency bins.

## C.2 Optimization Dynamics

Figure C.2 shows that, relative to fixed-scale InSeNT, AMS keeps the adaptive scale above one, increases calibration efort, and reduces conditional sibling-allocation entropy. This is consistent with Proposition A.5: dificult rows receive stronger, more concentrated updates. Routed-layer reconstructions further indicate that CKR redistributes attention across chunks and sharpens final-layer target selection.

![](images/cf5c6c55661c0edc009db168bd77b88bf1135499ea0f27ac5cbeade268ed2feb.jpg)

![](images/bd555b432dde4914878ea9330de195dd4c035df4efc4818d4d120f8f83124603.jpg)

![](images/01db0a978c916742193092e934245085bd31be1973b7153d2817889ffac032ca.jpg)  
Figure C.2: Population training dynamics: adaptive scale, calibration efort, and conditional negative-allocation entropy.

## D Robustness and Baseline Comparisons

## D.1 Temperature Baseline Comparisons

Table D.1 compares AMS and DcLBC with fixed and adaptive temperature baselines under the same protocol [20, 21]. Controlled-3 is not a separate dataset: it is the macro average over Football, Cities, and EIOPA, the three ConTEB tasks that most directly require document context to resolve a chunk. The comparison shows that global temperature schedules do not match AMS’s query-specific document-local calibration.

## E Additional Encoder Evidence

## E.1 Controlled-3 Context-Necessity Subset and Sub-1B LLM Encoders

Controlled-3 is not a standalone dataset but the unweighted macro average of NDCG@10 on Football, Cities, and EIOPA [2]. Football and Cities require document-level entity resolution, whereas EIOPA requires disambiguation among structurally similar regulatory sections. This subset focuses on retrieval that requires document context. DcLBC outperforms InSeNT on this subset across all three evaluated backbones.

Table D.1: Temperature baselines on ModernBERT-large (NDCG@10, %). Controlled-3 is the macro average over the Football, Cities, and EIOPA datasets in ConTEB.
<table><tr><td>Method</td><td>ConTEB Avg</td><td>Controlled-3</td></tr><tr><td>InSeNT (τ = 1.0)</td><td>76.70</td><td>84.20</td></tr><tr><td>Fixed τ = 0.8</td><td>77.15</td><td>85.08</td></tr><tr><td>Dual Temperature [20]</td><td>62.44</td><td>64.28</td></tr><tr><td>MACL (temp.-only) [21]</td><td>76.15</td><td>82.71</td></tr><tr><td>MACL (full) [21]</td><td>63.96</td><td>65.84</td></tr><tr><td>AMS (Ours)</td><td>77.96</td><td>87.05</td></tr><tr><td>DcLBC (AMS + CKR, Ours)</td><td>78.61</td><td>87.71</td></tr></table>

Table E.1: Aggregate zero-shot comparison of Direct and Late Chunking (LC) modes (NDCG@10, %). Each score pair is reported as Direct / LC.
<table><tr><td>Model</td><td>ConTEB</td><td>Controlled-3</td></tr><tr><td colspan="3">Causal encoders</td></tr><tr><td>Qwen3-Embedding [24]</td><td>51.37/51.15</td><td>30.85/66.07</td></tr><tr><td>Harrier OSS [25]</td><td>51.54/47.28</td><td>35.66/53.75</td></tr><tr><td>ML-Embed [26]</td><td>45.91/39.67</td><td>25.10/40.48</td></tr><tr><td colspan="3">Bidirectional encoders</td></tr><tr><td>Giga Embeddings [30]</td><td>55.75/63.58</td><td>35.22/68.43</td></tr><tr><td>pplx-embed-v1 [27]</td><td>46.79/69.19</td><td>26.08/77.41</td></tr><tr><td>BidirLM [28]</td><td>43.02/54.47</td><td>23.30/59.08</td></tr><tr><td>Nano-Em1-v2 [29]</td><td>39.17/41.28</td><td>21.81/42.86</td></tr></table>

Table E.1 compares independent chunk encoding (Direct) with Late Chunking (LC) for zero-shot sub-1B LLM encoders. LC is inconsistent for causal last-token encoders but improves all evaluated bidirectional mean-pooling encoders on both aggregate columns, consistent with the directionality of context propagation.