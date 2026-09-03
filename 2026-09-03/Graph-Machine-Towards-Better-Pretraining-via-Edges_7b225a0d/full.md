# Graph Machine: Towards Better Pretraining via Edges

Lintai Hou

lintai@iterlabs.ai

## Abstract

We introduce the Graph Machine (GM), an architecture that maintains an $O ( n )$ sized state and accesses it through sparse, dynamic routing. Unlike methods with fixed-size states or sparse but static routing, GM preserves $O ( n )$ complexity in its sparse layers without restricting the potentially accessible state size to O(1). Instead, GM uses edges—pointer-like objects updated differentiably by a referral mechanism resembling pointer chasing. We replace 75% of the dense Transformer layers in Qwen3-0.6B with GM sparse layers and pretrain from scratch on 15.7B tokens. With only 2 of 4,096 tokens retrieved per KV head in each sparse layer, loss degrades only slightly; with 4, the best model marginally improves loss.

![](images/4918bcf17b53d3dba11e0fb4041700766ba94132b178bf8cd1c20dfd194a3cd2.jpg)  
Attention

![](images/faf0d1aff1a199b347aa58968a25f5e8718d5ffa553840cc5edae40f801d65e4.jpg)  
Figure 1: Attention and referral viewed as pointer operations.

![](images/ff2856ec7bc1957ad5f21710f2c65a6c0b39b94e33f9983a13e126bc876e232e.jpg)  
Figure 2: Different referral approaches with ℓ = 2 (simplified).

## 1 Introduction

We classify sequence modeling approaches by the sizes of their state, access, and dynamic addresses.

Recurrent models such as $\mathrm { R N N s } ^ { 1 }$ and $\mathsf { S S M s } ^ { 2 }$ maintain a Θ(1)-sized state and must therefore compress their history.

Transformers<sup>3</sup> avoid this compression by retaining a $\Theta ( n )$ -sized state in their keys and values. However, vanilla attention allows each step (token) to access the entire state, leading to $\Theta ( n ^ { 2 } )$ complexity.

Methods such as sliding-window attention<sup>4</sup> reduce this access to Θ(1). Yet the accessible positions are fixed independently of the new step’s content: all but Θ(1) positions can be excluded in advance. Thus, their access is sparse but static.

A simple information-theoretic argument shows that dynamic, Θ(1)-sized access to a $\Theta ( n )$ -sized state requires each new token to provide Θ(log n) bits that can be used directly to address—identify and retrieve—a constant-sized portion of the state. Together, these properties define a fourth category.

Table 1: A sequence-model taxonomy by state size, per-step access size, and per-step dynamic address bits
<table><tr><td>Type</td><td>Example(s)</td><td>State</td><td>Access</td><td>Dynamic address bits</td></tr><tr><td>1</td><td>RNN, SSM</td><td>Θ(1)</td><td>Θ(1)</td><td>1</td></tr><tr><td>2</td><td>Vanilla transformer</td><td>Θ(n)</td><td>Θ(n)</td><td></td></tr><tr><td>3</td><td>Sliding-window transformer</td><td>Θ(n)</td><td>Θ(1)</td><td>O(1)</td></tr><tr><td>4</td><td>GM</td><td>Θ(n)</td><td>Θ(1)</td><td>Θ(log n)</td></tr></table>

Although each dynamic address theoretically contains $\Theta ( \log n )$ bits, at practical sequence lengths it fits in a fixed-width integer such as int64, making its storage cost effectively constant. The Graph Machine therefore instantiates this fourth class: it maintains $\Theta ( n )$ state entries, accesses Θ(1) entries per step, and uses Θ(log n) bits of dynamic addressing per step. Thus, GM stores both floating-point hidden states, which we call nodefeatures, and integer edge targets which we call edge indices. At each new step, stored edge indices are used to retrieve Θ(1) positions for information aggregation by attention (Figure 1).

To update the stored edges across layers, we use a $r e f e r r a l ^ { 5 }$ mechanism that constructs the next neighborhoods from the current ℓ-hop neighborhoods. Referral can be understood as neighbors recursively referring to their neighbors for ℓ rounds, as composing ℓ-hop edges into one-hop edges, or as taking an adjacency matrix A and producing $A ^ { \ell }$ (Figure 1). If each node retains s neighbors, referral can produce up to $s ^ { \ell }$ candidate paths. Keeping the neighborhood size fixed at s therefore requires a hard selection that reduces these candidates to s neighbors. To learn the selection, a probabilistic policy is needed in the form of

$$
A _ { \mathrm { b i n a r y } } ^ { \prime } \sim \pi _ { \theta } \big ( \cdot \mid A _ { \mathrm { b i n a r y } } ^ { \ell } \big ) ,
$$

which would introduce credit assignment difficulties.

Instead, we soften the edges rather than the policy, reformulating the top-s selection into a top-s approximation over soft weights. We start with a soft and sparse adjacency matrix, take its ℓ-th power, and then sparsify the result to form the new adjacency matrix (Figure 2). I.e., we approximate

$$
A _ { \mathrm { d e n s e } } ^ { \prime } = A _ { \mathrm { d e n s e } } ^ { \ell }
$$

with

$$
A _ { \mathrm { s p a r s e } } ^ { \prime } = \mathrm { S p a r s i f y } _ { s } ( A _ { \mathrm { s p a r s e } } ^ { \ell } ) .
$$

The resulting edge weights then contribute to the attention weights alongside the usual query–key factors in a product-of-experts<sup>6</sup> manner.

Intuitively, referral seeks to provide useful relational representations to attention through sparsityconstrained intermediaries. Rather than learning to shift the support directly, it learns from redistributing weight mass within the support. From another perspective, each edge is now jointly represented by integer edge indices and floating-point edge weights, the latter carrying gradients that discrete indices cannot.

Viewed globally, the Graph Machine maintains and updates a graph with soft directed edges. Sparse attention passes features along edges to update nodes, while referral passes addresses along edges to form new ones. In this paper, we hybridize GM and Transformer layers and train the resulting Graph Language Machines (GLMs) under a standard language-model pretraining setup. Throughout the paper, we move between the vocabularies of adjacency matrices, edges, neighbors, addresses, pointers, and indices as appropriate.

## 2 Architecture

We first provide an overview of the GM architecture, then introduce two common operations, and finally describe its two new sparse edge submodules.

## 2.1 Overview

![](images/db4898ee9da253848f8ccb57ddca811ffa4d58092ee892a50c23c0e8b9e577a2.jpg)  
Figure 3: GM states.

![](images/6d22eb1851306edf9a2313455b337830df8a7d0b38a99e51d01c1b934d998ecb.jpg)  
Figure 4: Adjacency matrices → edge states.

States. GM maintains three state tensors (Figure 3). Alongside the $n \times d$ node features, it maintains edge indices and edge weights, both of shape $n \times k \times s _ { 1 }$ . We obtain this representation by generalizing the adjacency matrix to k copies and representing them in an $s _ { 1 }$ -sparse coordinate format, giving each node k edges and each edge $s _ { 1 }$ member positions (Figure 4). Across the $s _ { 1 }$ member positions, the edgeindex tensor specifies target nodes, while the edge-weight tensor provides corresponding nonnegative weights that sum to one. We call the edge states passed between layers stored edges, distinguishing them from the mixed edges used directly by referral and attention within the submodules.

![](images/a1e5f300fab9643c46baccc21bade7f1af4df6769ac117d616f341d33f161026.jpg)  
Figure 5: GM layers.

![](images/58f0c5839e73d53261bddaa7577cd5e478d1ae18567b3e01a2c9cb86ed3b902f.jpg)  
Figure 6: A sparse GM layer.

Layers. Token embeddings initialize the node features in our GLMs. Each token’s initial input edges point to itself and its k − 1 nearest preceding tokens, clamped to the first token and cached for reuse. Each input edge places all its weight on its single target and fills its remaining member positions with zero indices and weights. The dense layers are standard Transformer layers. Each sparse layer applies r edge-referral submodules, each performing two-hop referral, followed by an edge-attention submodule and an MLP. Referral updates the edges, while attention and the MLP update the node features. Edges remain causal because they are derived from previous causal edges or masked attention. As usual, both dense and sparse layers use pre-normalization and residual connections for node features. A final output projection produces prediction logits from node features.

## 2.2 Operations

Sparsify. A conceptual weighted average of adjacency-matrix rows requires several additional steps in a sparse-coordinate implementation. Duplicate indices must be coalesced, the highest-weight entries selected, and their weights renormalized to achieve the target sparsity. For example, equally averaging ([0, 3], [0.4, 0.6]) and ([2, 3], [0.8, 0.2]) first produces

$$
( [ 0 , 3 , 2 , 3 ] , [ 0 . 2 , 0 . 3 , 0 . 4 , 0 . 1 ] ) .
$$

Coalescing the two entries for index 3, dropping the lowest-weight entry, and renormalizing gives ([2, 3], [0.5, 0.5]).

Since we use hard top-s selection, only the retained weights receive gradients. Sparsification occurs when stored edges are mixed before referral and attention, and when the resulting edges are reconciled after referral.

Mix. We mix the k stored edges of each node before using them for referral or attention, much as a pointwise convolution mixes channels in a CNN. For each output edge, mixing logits are projected from the node features and normalized over the input edges with a softmax, producing a mixing matrix M. The resulting coefficients are used to compute a weighted average of the input edges. Weighted averaging ${ \widetilde { A } } = M A$ alone would keep new edges approximately within the convex hull of earlier edges. We therefore introduce nonlinearity through temperature scaling both before and after the average. Separate input- and output-edge temperatures are parameterized as the softplus or exponential of projections from the node features; each acts as an exponent on the corresponding edge weights, which are then renormalized. I.e., for an edge-weight vector a, the temperature-scaling is

$$
\mathrm { T } _ { \tau } ( a ) = \frac { a ^ { \tau } } { \sum _ { j } a _ { j } ^ { \tau } } , \quad \tau > 0 ,
$$

and mixing follows

$$
\widetilde { A } = \mathrm { T } _ { \tau _ { \mathrm { o u t } } } \left( M \mathrm { T } _ { \tau _ { \mathrm { i n } } } ( A ) \right) .
$$

Sparsification coalesces the mixed edges $\widetilde { A }$ and reduces them to the desired sparsity.

## 2.3 Submodules

Sparse edge referral (SER). During sparse edge referral, rather than using a single adjacency matrix $A ,$ we take two matrices $A _ { 1 }$ and $A _ { 2 }$ and perform two-hop referral via

$$
A ^ { \prime } = \mathrm { S p a r s i f y } _ { s _ { 1 } } ( A _ { 1 } A _ { 2 } ) .
$$

At a path level, referral composes a first edge $e _ { 1 } = ( n _ { 1 } \ { \xrightarrow { \ w _ { 1 } } } \ n _ { 2 } )$ with a second edge $e _ { 2 } = ( n _ { 2 } \stackrel { w _ { 2 } } { \longrightarrow } )$ $n _ { 3 } )$ , omitting sparsification:

$$
\begin{array} { r } { e _ { 1 } \circ e _ { 2 } = \left( n _ { 1 } \xrightarrow { w _ { 1 } w _ { 2 } } n _ { 3 } \right) . } \end{array}
$$

Concretely, to construct k new edges, we first mix the k stored edges into 2k channels, forming k pairs that represent the two legs of the new edges. Within each pair, $e _ { 1 }$ and $e _ { 2 }$ are sparsified to $s _ { 2 }$ and $s _ { 3 } ,$ respectively. The $e _ { 1 }$ indices are then used to retrieve the corresponding $e _ { 2 }$ indices and weights. This produces up to $s _ { 2 } s _ { 3 }$ members per edge before sparsification to $s _ { 1 }$ . Optionally, refresh edges augment the stored edges before mixing. These refresh edges can come from the initial input edges or from the top-k indices and normalized weights of the most recent dense-attention layer.

Sparse edge attention (SEA). During sparse edge attention, we mix the k stored edges into g channels, one per KV head, and sparsify each to $s _ { 4 }$ . The resulting edge indices specify which positions provide the keys and values. We compute the usual scaled query–key scores over these positions and combine them with the mixed edge weights as a product of experts. Specifically, we call the query–key scores nodefactors and scale them using per-head temperatures parameterized analogously to the output-edge temperatures in mixing. We call the logarithms of the mixed edge weights edge factors. Adding the node and edge factors and applying softmax produces the final attention weights. Formally, for one query head and its associated KV head, let $\mathcal { N } ( i )$ denote the positions retrieved for node $i ,$ and let $w _ { i j }$ denote their mixed edge weights. The final attention weights are

$$
a _ { i j } = \mathrm { s o f t m a x } _ { j \in \mathcal { N } ( i ) } \left( \tau \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d _ { h } } } + \log w _ { i j } \right) \propto w _ { i j } \exp \left( \tau \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d _ { h } } } \right) .
$$

Here, addition in logit space corresponds to a product of experts in probability space. Value aggregation and output projection then follow as in standard attention. Optionally, we use the retrieved indices and final attention weights to replace a subset of the stored edges. This realigns the pointer distribution after approximate referral using the feature-based evidence obtained during attention.

## 3 Results

We use $\mathrm { Q w e n } 3 { - } 0 . 6 \mathbf { B } ^ { 7 }$ as our baseline and as a representative modern dense LLM. For a controlled comparison, all models are trained from scratch using the same backbone and training hyperparameters, codebase, and random seed. We use a conventional LLM training recipe based on established practice, without tuning it to our specific conditions.

For GM, we use a sparse-to-dense layer ratio of 3:1, scheduled as $[ \mathrm { S } , \mathrm { S } , \mathrm { D } , \mathrm { S } ] \times 7 .$ We use 16–32 edges, 0–6 referral steps, sparsity budgets of (2, 4, 4, 2) or (4, 4, 4, 4), 8 input-refresh edges, and 0–8 dense-refresh edges, with realignment enabled. Temperatures are parameterized by the horizontally shifted softplus $\bar { T } ( x ) = \mathrm { s o f t p l u s } ( x + \log ( e - 1 ) )$ , such that $T ( 0 ) \dot { = } 1 . \mathrm { R o P E } ^ { 8 }$ is omitted from SEA.

We name each GLM by its sparsity budget, number of stored edges, number of referral steps, and whether it uses eight dense-refresh edges. Theia<sup>9</sup> models use a sparsity budget of $( 2 , 4 , 4 , 2 )$ , whereas Hyperion models use (4, 4, 4, 4). Because $s _ { 4 }$ determines attention sparsity, Theia and Hyperion retrieve 2 and 4 positions per KV head during SEA, respectively. For example, Hyperion-K16-R3-S uses a (4, 4, 4, 4) sparsity budget, 16 edges, three referral steps, and eight dense-refresh edges. In the tables, we group conditions by sparsity and then order them by the number of referral steps.

Table 2: Parameter counts and estimated training FLOPs. Only multiplications and accumulations are counted, each as one FLOP; normalization, softmax, activations, RoPE, and sparsification are excluded. SEA KV access is measured over a 4,096-token sequence relative to dense causal attention.
<table><tr><td>Model</td><td>Parameters</td><td>Total compute</td><td>Ref + att compute</td><td>SEA att KV access</td></tr><tr><td>Qwen3</td><td>596M</td><td>78.4 EFLOPs</td><td>38.8 EFLOPs</td><td>100.000%</td></tr><tr><td>Theia-K24</td><td>601M</td><td>62.3 EFLOPs</td><td>22.7 EFLOPs</td><td>0.098%</td></tr><tr><td>Theia-K32-R2</td><td>711M</td><td>72.7 EFLOPs</td><td>33.0 EFLOPs</td><td>0.098%</td></tr><tr><td>Theia-K24-R3</td><td>700M</td><td>71.6 EFLOPs</td><td>32.0 EFLOPs</td><td>0.098%</td></tr><tr><td>Theia-K24-R4</td><td>733M</td><td>74.7 EFLOPs</td><td>35.1 EFLOPs</td><td>0.098%</td></tr><tr><td>Theia-K16-R4-S</td><td>686M</td><td>73.1 EFLOPs</td><td>33.5 EFLOPs</td><td>0.098%</td></tr><tr><td>Theia-K16-R6</td><td>700M</td><td>71.6 EFLOPs</td><td>32.0 EFLOPs</td><td>0.098%</td></tr><tr><td>Hyperion-K32-R1</td><td>657M</td><td>67.5 EFLOPs</td><td>27.9 EFLOPs</td><td>0.195%</td></tr><tr><td>Hyperion-K24-R3</td><td>700M</td><td>71.6 EFLOPs</td><td>32.0 EFLOPs</td><td>0.195%</td></tr><tr><td>Hyperion-K16-R3</td><td>650M</td><td>66.9 EFLOPs</td><td>27.3 EFLOPs</td><td>0.195%</td></tr><tr><td>Hyperion-K16-R3-S</td><td>664M</td><td>71.1 EFLOPs</td><td>31.4 EFLOPs</td><td>0.195%</td></tr><tr><td>Hyperion-K16-R4</td><td>666M</td><td>68.5 EFLOPs</td><td>28.8 EFLOPs</td><td>0.195%</td></tr></table>

We train on a randomly sampled, overprovisioned subset of FineWeb-Edu<sup>10</sup>, which is processed for less than one epoch. Documents are packed into 4,096-token sequences and padded as needed. A padding token is placed at the start of each sequence to absorb underflowing edge indices. Across a random sample of 100K documents, document length has a mean of 1,035 and a standard deviation of 1,909 Qwen3 tokens. A sequence length of 4,096 therefore does not imply a comparable effective context length within each document.

The implementation is primarily written in generic PyTorch<sup>11</sup>, with a custom Triton<sup>12</sup> kernel used only for sparsification. Each training run uses a single H100 SXM and takes 53–236 hours, with Qwen3 requiring 53 hours and most GLMs clustering around 150–160 hours. Under our prototype implementation, GLMs are generally several times slower than Qwen3 on this hardware. Preliminary experiments on an RTX 4090 show that some configurations approach Qwen3’s training throughput, suggesting that relative performance depends strongly on hardware and kernel implementation.

With the backbone dimensions held fixed, the referral and temperature parameters increase the parameter counts of referral-equipped GLMs by 9%–23%. Nevertheless, these models reduce total estimated training compute by 5%–15% and referral-plus-attention compute by 10%–30% relative to Qwen3. Most of the additional parameters belong to referral projections, which support relatively inexpensive operations. Over a length-n sequence, dense causal attention accesses an average of (n + 1)/2 KV positions per query. At n = 4096, retrieving at most two or four positions therefore corresponds to 0.098% or 0.195% of dense causal KV access, respectively.

Table 3: Test loss throughout training. Qwen3 entries give absolute loss, whereas GLM entries give the difference from Qwen3. Highlighting marks the model with the lowest final loss in each sparsity category.
<table><tr><td rowspan="2">Model</td><td colspan="4">Test loss / ∆ test loss</td></tr><tr><td>@ 25%</td><td>@ 50%</td><td>@ 75%</td><td>@ 100%</td></tr><tr><td>Qwen3</td><td>2.869</td><td>2.744</td><td>2.660</td><td>2.587</td></tr><tr><td>Theia-K24</td><td>+0.041</td><td>+0.039</td><td>+0.040</td><td>+0.040</td></tr><tr><td>Theia-K32-R2</td><td>+0.016</td><td>+0.017</td><td>+0.018</td><td>+0.020</td></tr><tr><td>Theia-K24-R3</td><td>+0.012</td><td>+0.012</td><td>+0.013</td><td>+0.014</td></tr><tr><td>Theia-K24-R4</td><td>+0.013</td><td>+0.013</td><td>+0.014</td><td>+0.015</td></tr><tr><td>Theia-K16-R4-S</td><td>+0.010</td><td>+0.013</td><td>+0.014</td><td>+0.016</td></tr><tr><td>Theia-K16-R6</td><td>+0.013</td><td>+0.014</td><td>+0.014</td><td>+0.015</td></tr><tr><td>Hyperion-K32-R1</td><td>+0.008</td><td>+0.008</td><td>+0.009</td><td>+0.009</td></tr><tr><td>Hyperion-K24-R3</td><td>-0.002</td><td>-0.003</td><td>-0.003</td><td>-0.002</td></tr><tr><td>Hyperion-K16-R3</td><td>0.000</td><td>0.000</td><td>-0.001</td><td>0.000</td></tr><tr><td>Hyperion-K16-R3-S</td><td>-0.005</td><td>-0.005</td><td>-0.005</td><td>-0.003</td></tr><tr><td>Hyperion-K16-R4</td><td>-0.004</td><td>-0.003</td><td>-0.003</td><td>-0.002</td></tr></table>

All models are evaluated on the same fixed held-out random subset of FineWeb-Edu. Final test losses are around 2.60, consistent with broader LLM pretraining experience at this scale.

Increasing the number of edges beyond the 16 attention heads provides a moderate improvement in the matched Hyperion comparison: K24-R3 outperforms K16-R3 by 0.003 at the end of training. This improvement comes at additional parameter and compute cost because full cross-edge mixing scales as k<sup>2</sup>.

Referral clearly improves performance from zero to a few steps: Theia-K24-R3 improves over the non-referral Theia-K24 baseline by 0.026 at 100%. Additional referral steps can also help under some conditions, with Hyperion-K16-R4 improving over Hyperion-K16-R3 by 0.003. However, this trend does not hold in every setting. Under the sparser Theia budget with more edges than heads, Theia-K24-R4 performs 0.001 worse than Theia-K24-R3.

Preliminary experiments suggest benefits from realignment and input refresh. We observe a similar benefit from dense refresh: Hyperion-K16-R3-S improves over Hyperion-K16-R3 by 0.004 at 100%.

Within the tested configurations, neither parameter count nor estimated compute is a strong predictor of performance. Hyperion-K16-R3-S achieves the best loss despite being among the less expensive GLMs, with 11% more parameters and 19% less referral-plus-attention compute than Qwen3.

Overall, the best Theia model increases final loss by approximately 0.014, while the best Hyperion model reduces it by approximately 0.003. These results show that most Transformer layers can be replaced by sparse layers retrieving only 2 or 4 of 4,096 positions per KV head without materially sacrificing quality, while meaningfully reducing estimated compute.

## 4 Related work

This work builds directly on the original GM work<sup>5</sup>; we refer to the two papers as GM-1 and GM-2.

1. GM-1 uses a bespoke Sudoku benchmark, whereas GM-2 adopts a standard language-model pretraining setting, making its results easier to contextualize.

2. GM-1 primarily studies the inductive bias introduced by edges, leaving computational practicality to future work. Its dense edge representation incurs cubic time and quadratic space complexity, limiting experiments to a few hundred nodes. GM-2 instead uses sparse edge representations and operations. Its edge indices and weights form a sparse coordinate representation of GM-1’s edge addresses, while SER and SEA are sparse counterparts of GM-1’s referral and attention operations.

3. GM-2 simplifies the state by unifying node and edge features, removing the need for carefully engineered interactions among separate representations.

4. GM-2 introduces refresh and realignment, allowing referral to reuse initial or dense-attentionderived edges and sparse attention to update stored edges.

GM is related to efficient sequence models and their hybrids<sup>13,14,15,16</sup>. GM is also situated within the sparse-attention literature <sup>17,4,18,19,20</sup>.

## 5 Limitations and conclusion

This work demonstrates the GM architecture while leaving substantial room for architectural and implementation improvements. More efficient custom kernels are one such important direction.

Our experiments are small-scale in both model and training, and evaluate only a pretraining setup using test loss as an aggregate metric. We leave richer evaluations across scales, setups, and downstream capabilities to future work.

Although inductive bias was a central focus of GM-1, it remains to be examined for the updated architecture and in the context of language modeling. As part of this investigation, but also as an independent direction, mechanistic interpretation could take advantage of GM’s highly selfinterpretable relational states and operations.

In conclusion, although some amount of global dense attention likely remains valuable under current language-modeling settings, our results show—at least at our scale, under our setup, and by our measure—that it is nonetheless highly trimmable. If part of attention’s role is fundamentally to pass and resolve addresses, then we can make the core machinery—logarithmic-sized identification bits and direct retrieval through them—primitives of the architecture. GM now carries and transmits addresses via indices rather than features, and resolves them through indexed gathering rather than dense scoring.

This creates new degrees of freedom for architectural design. On the efficiency side, dense computation can be traded off against sparse memory operations. On the inductive-bias side, architectures can trade off relational traversal against global search, and address passing against content passing. Finding the optimal balance will require further exploration.

## References

[1] Jeffrey L. Elman. Finding structure in time. Cognitive Science, 14(2):179–211, 1990.

[2] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022.

[3] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Informa tion Processing Systems, volume 30, pages 5998–6008. Curran Associates, Inc., 2017.

[4] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer, 2020.

[5] Lintai Hou. Graph machine: Exploring edge mechanisms as an inductive bias, 2026.

[6] Geoffrey E. Hinton. Training products of experts by minimizing contrastive divergence. Neural Computation, 14(8):1771–1800, 2002.

[7] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.

[8] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Ro-Former: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[9] IO Interactive. 007 first light. Video game, May 2026.

[10] Guilherme Penedo, Hynek Kydlícek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell,ˇ Colin Raffel, Leandro von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, volume 37, pages 30811–30849. Curran Associates, Inc., 2024.

[11] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, pages 8024–8035. Curran Associates, Inc., 2019.

[12] Philippe Tillet, H. T. Kung, and David Cox. Triton: An intermediate language and compiler for tiled neural network computations. In Proceedings ofthe 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, Mapl ’19, pages 10–19, New York, NY, USA, 2019. Association for Computing Machinery.

[13] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Proceedings ofthe First Conference on Language Modeling, 2024.

[14] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, 2025.

[15] Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avshalom Manevich, Nir Ratner, Noam Rozen, Erez Schwartz, Mor Zusman, and Yoav Shoham. Jamba: A hybrid transformer-mamba language model, 2024.

[16] Kimi Team. Kimi Linear: An expressive, efficient attention architecture, 2025.

[17] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers, 2019.

[18] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions ofthe Associationfor Computational Linguistics, 9:53–68, 2021.

[19] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23078–23097, Vienna, Austria, July 2025. Association for Computational Linguistics.

[20] DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models, 2025.