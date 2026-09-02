# MULTI-HEAD SELF ATTENTION IS A PARAMETER IDENTIFICATION MECHANISM

W ROSS MORROW

Abstract. We prove that a multi-head scaled dot product attention can be viewed as a parameter identification strategy. The ratio of unidentified parameters to the total number of parameters scales like the reciprocal of the number of heads $( 1 / 2  1 / ( 2 H ) )$ , meaning models with more heads are structurally more identified. A subtle side efect of the mathematics observation that attention can never be fully identified. Similarly we also show that some bias terms can have no efect on softmax-based attention layers in both the single- and multiple-head settings, though this is mostly a curiosity that should have a marginal efect on model size and model training/prediction eficiency. We also touch on modern improvements to transformers including RoPE and GQA from this perspective, illustrating how those as well can improve the ratio of “meaningful” parameters to all parameters. Simple numerical examples demonstrate that training can indeed involve updates that overlap model-invariant subspaces that arise from a lack of identification. As part of our experiments we use a “rebalancing” approach that can $\mathrm { { \^ { 6 } f i x } } ^ { \mathfrak { P } }$ updates that overlap unindentified subspaces but do not try to present evidence this should actually be adopted. Instead we simply view our numerical results as exploring and confirming the theoretical results. As a whole we discuss a purely mathematical/statistical explanation, identification, for why specific architectural choices in transformers may have improved performance.

## Contents

1. Introduction 2   
2. Related Literature 4   
3. Definitions 6   
3.1. Embeddings. 6   
3.2. Queries, Keys, and Values. 7   
3.3. Probabilities. 7   
3.4. Value Weighting. 8   
3.5. Multi-Head Attention. 8   
4. Parameter Identification in Attention 9   
4.1. A Trivial, Yet Illustrative, Example. 9   
4.2. Invariances in Query-Key Products. 10   
4.3. A “Principled” Approach. 11   
4.4. Directional Derivatives. 12   
4.5. Multi-Head Attention Improves Identification. 14   
4.6. Invariance with Query-Key Bias Terms. 15   
4.7. Irrelevant Bias Terms. 15   
4.8. Extensions. 17   
4.9. Grouped Query Attention. 18   
4.10. Equivalence. 19   
5. Numerical Examples 20   
5.1. Gradient Flow. 20   
5.2. Training Runs. 22   
5.3. Training Results. 23   
6. Conclusions 25   
Declaration of Generative AI and AI-assisted technologies in the   
writing process 26   
References 26

## 1. Introduction

A fundamental part of the current success of transformer architectures is the multi-head attention introduced in Vaswani et al. (2023). This work popularized not only using scaled dot products as the basis of a mixing operator to train relevant context for token prediction, but also the branching strategy of using multiple low-rank heads in a batched fashion followed by “averaging” of their efects into the embedding dimension. The formal form of these interactions raises a question about parameter identification: whether all parameters (weights and biases) introduced into a modeling architecture actually represent trainable efects. While sidestepping such discussions is understandable and even probably practical in the face of the complexities of modern neural networks like transformers, using dot products in token mixing is a relatively obvious source of potential reduction in the efective predictive power of each individual parameter.

In this short note we focus on and examine the interplay between identification and dot product attention. We observe how by multiplying weight matrices together we lose identification, both by demonstrating explicit invariances and identifying where directional derivatives vanish. Importantly these results show that it is impossible for all query-key value weights to have meaningful efects and, for similar parameter counts, a multi-head attention model has a higher fraction of identified parameters than a single-head attention. Specifically the unidentified fraction of parameters is related to the reciprocal of the number of heads $( 1 / 2  1 / ( 2 H )$ , though there are other practical and competing consequences to increasing the number of heads). The observation of a symmetry that implies a reduction in identification is not in itself so surprising or novel. However we are unaware of the derived observation that splitting into multiple heads has the byproduct of (perhaps dramatically) improving identification, or the fact that this basic efect is also further enhanced by RoPE and GQA.

We also undertake and report on some small scale and simple numerical experiments to suggest how this might afect model training in practice. Gradients with respect to weights will avoid symmetries that are connected to unidentified parameters by definition. Better optimizers add in higher order or momentum terms break this symmetry and can overlap invariant subspaces in directional derivatives. Our numerical experiments are mostly pointed at confirmation of the theoretical results, not to proposing specific methods to improve training. We do though present and use a “rebalancing” method as part of our confirmations that could be used to “canonicalize” weight iterates in training.

Our overall perspective is that identification is not just a theoretical curiosity, nor a classical statistical consideration outdated for deep learning. When parameters or parameter combinations are unidentified, we both store data on disk and in memory that doesn’t influence outcomes as well as can train with updates to weights or biases that have reduced (or no) efect on model outputs. For efective and eficient training and inference, modelers should ideally aim to ensure that as many parameters as possible are actually influencing model predictions; in mathematical language we should aim for injective representations over the parameter space.

In itself this is probably an uncontroversial if idealistic perspective. However architectural innovations in deep networks like transformers have been made that have the byproduct of improving parameter identification, without this being a directly stated motivation for those architectures. Presuming more identification indeed means more predictive capability for the same model size, at least in parameter counts, changing identification raises a confound for what is making more identified architectures better. “Structural” reasons to introduce multiple heads (or RoPE, or GQA) can be valid at the same time as predictive performance is improved simply because more of the parameters have predictive influence or meaning.

Any potential improvements in identification presented by multi-head attention are, surely, not the only mechanism driving model quality improvement. A multi-head attention setup is functionally distinct from a singlehead attention, in the sense that these two generate distinct functional classes mapping one set of embedded token sequences to another. Along this line of thinking we also show there is no efective-parameter-comparable single-head attention we can build for a given multi-head attention without lifting to a larger embedding space for the entire model.

This note starts with a short review of recent literature related to transformers and identification. We follow that with a careful definitions section to properly set the stage for results we derive. The third section focuses on our identification results, particularly that the architecture of multi-head attention alone improves identification. Our final substantive section reviews our numerical experiments to explore and confirm the theory. The note closes with brief review of the main conclusions we draw.

## 2. Related Literature

Of course the basis for our discussion is the pivotal 2017-but-updated paper Vaswani et al. (2023) introducing the main components of self attention and the transformer architecture. Here is the motivation for a multihead attention as stated there:

[W]e found it beneficial to linearly project the queries, keys and values h times with diferent, learned linear projections

... Multi-head attention allows the model to jointly attend to information from diferent representation subspaces at different positions. With a single attention head, averaging inhibits this.

Having efectively popularized the architecture enabling much of today’s deep networks there have been extensive discussions about, investigations into, and extensions to this architecture. We’ll review several related lines of enquiry here.

The author originally learned about transformers and worked on this note through experiments with nanoGPT (Karpathy, 2022), reading a formal report from Google Brain (Phuong and Hutter, 2022), and of course Vaswani et al. (2023). By today our results are not entirely novel nor are really intended to be. Zhao, Walters, and Yu (2025) discuss group symmetries in detail identifying a $G L _ { d } ( \mathbb { R } )$ symmetry in adjacent linear layers, including self-attention’s query-key products, as well as Softmax invariance under certain translations. Both appear here with constructive proofs. Zhao, Walters, and Yu (2025) in fact also elaborate on parameter identifiability and training dynamics, and cite numerous articles where the completeness of symmetry groups with respect to identifiability is known. Our contribution (if any) is to elaborate on how these results specifically relate to the introduction of multiple heads in attention as described in Vaswani et al. (2023), and how by doing so we improve parameter identification “out of the box” in a manner counfounding motivation from “representationality”.

Zhang et al. (2025), motivated by well-known permutation symmetries in MLPs, study rotational symmetries in attention. This is a close subclass of the invariances we study: coarsely, and in our language, they focus on something like half the unidentified degrees of freedom we do. Importantly Zhang et al., 2025 don’t simply observe such symmetries but also propose a method to “quotient out” the associated invariances in training (Zhao, Walters, and Yu, 2025). We do something similar with weight rebalancing in AdamW though do not formally propose or analyze this rigorously as a serious method for training.

Fundamentally multi-head attention achieves higher identification through lowering the rank of any specific attention applied, at least between the queries and keys. Bhojanapalli et al. (2020) leverage the “representation power” argument for multiple low-rank heads and propose setting the attention head output projection size to the input sequence length (but experiment only within the uppser bound of the related embedding dimension). Their theoretical motivation is a representation theorem stating that to get an arbitrary stochastic output matrix out of an attention head one must choose an inner head dimensionality (here denoted D) at least the sequence length (here E). They run experiments with multi-head attentions with “inflated” output dimensions larger than the layer’s embedding dimension divided by the number of heads as in Vaswani et al. (2023). Our results are compatible with improved performance in this architecture and also present a tradeof: the “most identified” extreme of rank-1 heads $( D = 1 )$ ) is the least representationally powerful, but the “most representational” extreme of full-rank heads $( D = E )$ is the most wasteful with parameters, and the original proposal $( D = E / H$ where H divides E) efectively trades of these two efects of representational rank and parameter eficiency.

As of writing many successful transformer models for language also adopt Rotary Positional Encodings (RoPE) following the work in Su et al. (2023). Importantly this technique partially breaks the symmetries in the original multi-head attention. We study a sketch of RoPE, concluding this related architecture again significantly improves identification $( { } ^ { 6 4 } D ^ { 2 } \ \to \ D$ identified”) by efectively inducing a positional family commutativity constraint. Identification, though, is seemingly not why RoPE was proposed. Much like multi-head attention RoPE appears to have a representational motivation with a very similar side efect on parameter eficiency. There is also recent interest in no positional encoding (NoPE) for learned long-context positional relationships (Wang et al., 2024) which would recenter the ${ } ^ { 6 \ell } G L _ { D } ( \mathbb { R } ) ^ { 3 }$ idenfication considerations we discuss.

Further enhancements like grouped query attention (GQA) (Ainslie et al., 2023) or cross attention (Brandon et al., 2024) have also appeared. With the exception of RoPE or any other modifier “inside” query-key products these would change the quantitative, not qualitative, results. Like RoPE we review GQA specifically, and how it can tie together head symmetries to fewer groups to improve identification.

As a final note related to existing literature, we could also consider any relationship identification may have with the Lottery Ticket Hypothesis (LTH) (Frankle and Carbin, 2019). Experiments in that work suggest far fewer meaningful parameters than we formally show are identified. Specifically our results suggest for some characterizations of now-plain multi-head attention networks about 90% of parameters have “meaning”, whereas the LTH results are more like 10-20%. This misalignment means should a lack of identification alone be a prominent feature in the LTH many more sources are missing.

## 3. Definitions

Let’s say we have inputs $\mathbf { x } \in \{ 1 , . . . , T \} ^ { L } \subset \mathbb { R } ^ { L }$ where $L \in \mathbb { N }$ is a uniform token sequence length and $\{ 1 , \dots , T \} \subset \mathbb { \tilde { \Gamma } }$ N is the token vocabulary (or indices of them), and embed tokens as vectors in $\mathbb { R } ^ { E }$ for some $E \in \mathbb { N }$ . We focus on attention heads motivated by the Vaswani et al. (2023)-popularized scaled dot product formulation stylized as

$$
\mathbf { V } \operatorname { S o f t m a x } \left( \mathbf { K } ^ { \top } \mathbf { Q } \right)
$$

of attention with query-key-value structure as a basic layer unit. Note here we aren’t using the row-centric standards common in deep learning, in which some “broadcast” logic is more natural, but rather a maybe more mathematical column vector orientation. This is an equivalent representation with the right transposing that is a bit more consistent with general linear algebra knowledge but mainly just more comfortable for the author. We also exclude a scaling factor, as this is a computational convenience with no unique impact on theory, as we can just absorb scaling into the weights for our purposes. We describe the details of how this layer works in this section to provide us with the foundation for analysis.

Just for clarity, denoting a generic pytorch-style linear layer as

$$
\mathcal { L } ( \mathbf { X } | \mathbf { W } , \mathbf { b } ) = \mathbf { X } \mathbf { W } ^ { \top } + \mathbf { 1 6 } ^ { \top }
$$

we would implement

$$
\mathrm { S o f t m a x } _ { d } \left( \mathcal { L } _ { Q } ( \mathbf { X } ) \mathcal { L } _ { K } ( \mathbf { X } ) ^ { \top } \right) \mathcal { L } _ { V } ( \mathbf { X } )
$$

The literal inputs X could difer or be batched (or both), as long as the dimensions work out. Perhaps importantly though the pytorch form has exactly the same inner matrix products we identify invariances in and require no modifications to understand.

3.1. Embeddings. Our first step, token embedding, actually occurs before attention and any layers: we are given a “table” $\mathbf { e } : \{ 1 , \dots , T \}  \mathbb { R } ^ { E }$ that maps tokenized sequences into a vector space for some $E \in \mathbb { N }$ , and construct a matrix representation of the input sequence as

$$
\mathbf { X } = \left( \begin{array} { l l l } { | } & { } & { | } \\ { \mathbf { e } ( x _ { 1 } ) } & { \cdots } & { \mathbf { e } ( x _ { L } ) } \\ { | } & { } & { | } \end{array} \right) \in \mathbb { R } ^ { E \times L }
$$

This embedding can be trained, or it can provide a level of pre-“trained” semantic similarity between tokens for words or word parts. Regardless we generally consider inputs $\mathbf { X } \in \mathbb { R } ^ { E \times L }$ of embedded tokens throughout layers. There are other properties satisfied by inputs to attention heads, such as layer normalization, that we won’t consider here.

3.2. Queries, Keys, and Values. With values in embedding space and choosing some $D \in \mathbb { N }$ we compute “queries” “keys” and “values” Q, K, V from linear (possibly afine) layers:

$$
\mathbf { Q } ( \mathbf { X } ) = \mathbf { W } _ { Q } \mathbf { X } \in \mathbb { R } ^ { D \times L }
$$

$$
\mathbf { K } ( \mathbf { X } ) = \mathbf { W } _ { K } \mathbf { X } \in \mathbb { R } ^ { D \times L }
$$

$$
\mathbf { V } ( \mathbf { X } ) = \mathbf { W } _ { V } \mathbf { X } \in \mathbb { R } ^ { E \times L }
$$

$$
\mathrm { w h e r e } \mathbf { W } _ { Q } , \mathbf { W } _ { K } \in \mathbb { R } ^ { D \times E } , \mathbf { W } _ { V } \in \mathbb { R } ^ { E \times E }
$$

Later we’ll consider bias terms, but for now we ignore them. Note also that

$$
\mathbf { Q } ( \mathbf { X } ) = \mathbf { W } _ { Q } \mathbf { X } = \left( \begin{array} { c c c } { \big | } & { } & { \big | } \\ { \mathbf { W } _ { Q } \mathbf { x } _ { 1 } } & { \cdots } & { \mathbf { W } _ { Q } \mathbf { x } _ { L } } \\ { \big | } & { } & { \big | } \end{array} \right) \in \mathbb { R } ^ { E \times L }
$$

(similarly for $\mathbf { K } ( \mathbf { X } )$ and $\mathbf { V } ( \mathbf { X } ) )$ and thus column ordering in queries, keys, and values is preserved relative to positional ordering in any original input sequence x (prior to embedding to lift up to X which also preserves ordering). Efectively, the keys and queries “project” embeddings into $\mathbb { R } ^ { D }$ for some $D ,$ whereas the values (in a single head) are functionally a post-attention linear layer.

We’ve assumed the value layer is “square” $( E \times E )$ instead or rectangular, and ignore biases here which are easy to add back. The setup described here has $2 D E + E ^ { 2 }$ parameters. If $D = E$ we have $3 E ^ { 2 }$ parameters but the queries and keys can be low rank with $D < E$ . These symbols and counts will get slightly more complicated for layers that “lift” or “contract” the embedding dimension, but not significantly meaningful way.

3.3. Probabilities. Using a query-key multiplication as above is suficient to efect “embedded token mixing” interpreted as any operation across tokens in a sequence. More commonly, transformer architectures also take a Softmax over columns of the query-key product matrix:

$$
\operatorname { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ( \mathbf { X } ) \right) \in \mathbb { R } ^ { L \times L }
$$

again ignoring scaling, forming a set of column-specific probabilities. For “causal” modeling we would also include a mask whose efect is to make the Softmax respect ordering in the token embeddings. Formally, the $( d , \ell ) ^ { . }$ th element is defined as

$$
\operatorname { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ( \mathbf { X } ) \right) _ { d , l } = \frac { \exp \{ \mathbf { k } _ { d } ^ { \top } \mathbf { q } _ { \ell } \} } { \sum _ { c } \exp \{ \mathbf { k } _ { c } ^ { \top } \mathbf { q } _ { \ell } \} }
$$

for some sum over c. The result is a left-stochastic matrix, i.e. one with positive entries whose column sums are all one.

Left multiplication by X forms convex combinations of the columns of X in the result:

$$
\begin{array} { r l } { \left( \sum _ { c = 1 } ^ { L } p _ { c , 1 } \mathbf { x } _ { c } } & { \sum _ { c = 1 } ^ { L } p _ { c , 2 } \mathbf { x } _ { c } \quad \cdots \quad \sum _ { c = 1 } ^ { L } p _ { c , L } \mathbf { x } _ { c } \right) } \\ { \qquad | \qquad | \qquad | } \end{array}
$$

where for simplicity we denote the $\operatorname { S o f t m a x } _ { \mathrm { c o l } }$ results as $p$ (for “probabil-$\mathrm { i t y ^ { \dag } }$ of course). A “causal” attention, wherein we “mask” out any “future” embeddings when computing probabilities, results in a form

$$
\begin{array} { r } { \left( \begin{array} { l l l l l } { | } & { | } & & { | } & \\ { \mathbf { x } _ { 1 } } & { \sum _ { c = 1 } ^ { 2 } p _ { c , 2 } \mathbf { x } _ { c } } & { \cdots } & { \sum _ { c = 1 } ^ { L } p _ { c , L } \mathbf { x } _ { c } } \\ { | } & { | } & & { | } \end{array} \right) } \end{array}
$$

embodied by a stochastic upper triangular $\operatorname { S o f t m a x } _ { \mathrm { c o l } }$ matrix, with entries depending only on forward relationships derived from sequence ordering. We’ll leave masking out of the notation most of the time, as what we discuss will not depend on masking.

Finally, note we are not considering “cross attentions” wherein the queries and keys are determined by diferent inputs, as might be useful in translation oriented encoder-decoder networks. Formally, we’re focusing on “self attention” layers, though our basic results about identification in matrix products hold for cross attention as well because they don’t relate to the inputs.

3.4. Value Weighting. We then compose those probabilities with a linear layer using $\mathbf { V } ( \mathbf { X } )$ ，

$$
\mathbf { V } ( \mathbf { X } ) \mathrm { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ( \mathbf { X } ) \right) = \mathbf { W } _ { V } \mathbf { X } \mathrm { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ( \mathbf { X } ) \right)
$$

We could view the attention head as a layer computing

$$
\mathbf { X } \operatorname { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } \right)
$$

with the convex combinations mentioned above followed by a linear layer with weights $\mathbf { W } _ { V }$ , where each column is the “expected” embedding for each token in the sequence using the probabilities from the $\operatorname { S o f t m a x } _ { \mathrm { c o l } }$

3.5. Multi-Head Attention. The Vaswani et al. (2023)-popularized $H _ { - }$ head attention generalizes the above choosing $H , D , C \in \mathbb { N }$ and taking

$$
\mathbf { Q } _ { h } ( \mathbf { X } ) = \mathbf { W } _ { Q } ^ { h } \mathbf { X } \in \mathbb { R } ^ { D \times L }
$$

$$
\mathbf { K } _ { h } ( \mathbf { X } ) = \mathbf { W } _ { K } ^ { h } \mathbf { X } \in \mathbb { R } ^ { D \times L }
$$

$$
\mathbf { V } _ { h } ( \mathbf { X } ) = \mathbf { W } _ { V } ^ { h } \mathbf { X } \in \mathbb { R } ^ { E \times L }
$$

$$
\mathrm { w h e r e } \mathbf { W } _ { Q } ^ { h } , \mathbf { W } _ { K } ^ { h } \in \mathbb { R } ^ { D \times E } , \mathbf { W } _ { V } ^ { h } \in \mathbb { R } ^ { C \times E }
$$

for $h = 1 , \ldots , H$ and adding an additional afine operation

$$
\sum _ { h = 1 } ^ { H } { \bf W } _ { O } ^ { h } { \bf V } ^ { h } ( { \bf X } ) \mathrm { S o f t m a x } _ { \mathrm { c o l } } \left( { \bf K } _ { h } ( { \bf X } ) ^ { \top } { \bf Q } _ { h } ( { \bf X } ) \right) + { \bf b } _ { O } { \bf 1 } ^ { \top }
$$

for parameters $\mathbf { W } _ { O } ^ { h } \in \mathbb { R } ^ { E \times C } , \mathbf { b } _ { O } \in \mathbb { R } ^ { E }$ and $\mathbf { 1 } \in \mathbb { R } ^ { L }$ a vector of ones. This form is the same as “concatenate and project” common in the literature and code like nanoGPT:

$$
\begin{array} { r } { \left( \mathbf { W } _ { O } ^ { 1 } \cdots \cdots \mathbf { W } _ { O } ^ { H } \right) \left( \begin{array} { c } { \mathbf { V } ^ { 1 } ( \mathbf { X } ) \mathrm { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } _ { 1 } ( \mathbf { X } ) ^ { \top } \mathbf { Q } _ { 1 } ( \mathbf { X } ) \right) } \\ { \vdots } \\ { \mathbf { V } ^ { H } ( \mathbf { X } ) \mathrm { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } _ { H } ( \mathbf { X } ) ^ { \top } \mathbf { Q } _ { H } ( \mathbf { X } ) \right) } \end{array} \right) + \mathbf { b } _ { O } \mathbf { 1 } ^ { \top } } \end{array}
$$

Note there is a fan-out/fan-in efect between value and output weighting, with strong similarity to adding up low-rank outer products. We project $\mathrm { ^ { 6 6 } d o w n ^ { 9 9 } }$ to $\mathbb { R } ^ { C }$ from $\textstyle { \mathrm { ~ \mathbb { R } ^ { E } ~ } }$ with the head-specific value weights (presuming $C < E )$ , and “lift” back up to $\mathbb { R } ^ { E }$ with the “output” weights introduced enabling us to “pool” efects from the diferent heads.

Counting up we have $2 H E ( D + C )$ weights, A convenient choice for the dimensions here is $D = C = E / H$ choosing an H that divides E which would imply we have a total of 4H $E D = 4 E ^ { \tilde { 2 } }$ weights.

## 4. Parameter Identification in Attention

Executing matrix-matrix products between weights results in a loss of degrees of freedom identifiable through training. Specifically, when our model multiplies weights directly inside attention heads, we form sums of products of elements that will be invariant to specific perturbations in the individual weights. This means two things: the model is “over-specified” in the specific sense that it has more parameters than degrees-of-freedom or “uniquely meaningful” efects, and there are linear subspaces over which layers including weight matrix products have vanishing directional derivatives with respect to weights (for any observations). This section refines these ideas and proves our basic results: (i) there are $D ^ { 2 }$ unidentified degrees of freedom in commonly used attention heads, and (ii) using multiple heads improves (increases) the ratio of identified degrees of freedom to parameters. Note that if there are always $D ^ { 2 }$ unidentified parameters (or combinations), an attention head can never be fully identified only asymptotically well identified.

A side note about our style of discourse here: we purposefully use probably overly deliberate and constructive derivations. Hopefully that doesn’t obscure our basic points which could be stated more tersely.

4.1. A Trivial, Yet Illustrative, Example. Suppose we forgo the embedding. That is, set $E = 1$ and $\mathbf { e } ( x ) = x$ in which case $\mathbf { X } = \mathbf { \bar { x } } ^ { \top } \in \mathbb { R } ^ { 1 \times L }$

Then $\mathbf { W } _ { Q } , \mathbf { W } _ { K } \in \mathbb { R } ^ { D \times 1 }$ are column vectors, the queries and keys

$$
\mathbf { Q } ( \mathbf { x } ) = \mathbf { W _ { \boldsymbol { Q } } } \mathbf { x } ^ { \top } \in \mathbb { R } ^ { D \times L }
$$

$$
\mathbf { K } ( \mathbf { x } ) = \mathbf { W } _ { K } \mathbf { x } ^ { \top } \in \mathbb { R } ^ { D \times L }
$$

are rank-1 transformations (outer products), and

$$
\mathbf { K } ( \mathbf { x } ) ^ { \top } \mathbf { Q } ( \mathbf { x } ) = \mathbf { x } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { x } = \omega \mathbf { x } \mathbf { x } ^ { \top } \qquad \omega = \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \in \mathbb { R }
$$

Even though this is entirely artificial (especially as we would likely constrain $D \leq E )$ it raises an important point: for all the 2D variables we admit with ${ \bf W } _ { Q } , { \bf W } _ { K } \in \mathbb { R } ^ { D \times 1 }$ we only have a single efect, the inner product of these two vectors. Even if $D = 1$ , we model with two numbers but only their product can have an efect. This is the prototype of an “identification” problem, in that we have $2 D - 1$ out of $2 D ^ { \circ }$ trainable” values that can’t influence the layer output.

4.2. Invariances in Query-Key Products. In general a lack of parameter identification arises from any query-key products as defined here. The fundamental operation is

$$
\mathbf { K } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ( \mathbf { X } ) = \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X }
$$

which we’ll sometimes refer to as a “quadratic form” even though that’s not technically precise to the usual meaning. Elhage et al. (2021) call this the “QK-circuit”. The now-eponomous “key-value cache” (Brandon et al., 2024) for eficient inference stores results of the layer application, possibly “hiding” some of these efects.

Multiplying two matrices has easily discoverable, trivial, invariances: First, any shared row permutation defined by $\mathbf { P } \in \mathbb { R } ^ { D \times D } , \mathbf { P } ^ { \top } \mathbf { P } = \mathbf { I } .$ , can be injected:

$$
\left( \mathbf { P } \mathbf { W } _ { \mathbf { K } } \right) ^ { \top } \left( \mathbf { P } \mathbf { W } _ { Q } \right) = \mathbf { W } _ { \mathbf { K } } ^ { \top } \mathbf { P } ^ { \top } \mathbf { P } \mathbf { W } _ { Q } = \mathbf { W } _ { \mathbf { K } } ^ { \top } \mathbf { W } _ { Q }
$$

Of course this also shows any orthonormal matrix can be injected in the same way (of which permutation matrices are just a special case). Also any nonsingular row scaling $\mathbf { A } = \mathrm { d i a g } ( \mathbf { a } )$ for any vector $\mathbf { a } \in \mathbb { R } ^ { D }$ with no zeros is an invariant via

$$
\left( \mathbf { A } ^ { - 1 } \mathbf { W } _ { \mathbf { K } } \right) ^ { \top } \left( \mathbf { A } \mathbf { W } _ { Q } \right) = \mathbf { W _ { K } } ^ { \top } \mathbf { A } ^ { - \top } \mathbf { A } \mathbf { W } _ { Q } = \mathbf { W _ { K } } ^ { \top } \mathbf { A } ^ { - 1 } \mathbf { A } \mathbf { W } _ { Q } = \mathbf { W _ { K } } ^ { \top } \mathbf { W } _ { Q }
$$

We’ve more or less shown the following general invariance:

Lemma 4.1. For any pair $( \mathbf { W } _ { Q } , \mathbf { W } _ { K } ) \in \mathbb { R } ^ { D \times E }$ and any nonsingular $\mathbf { S } \in$ $\mathbb { R } ^ { D \times D }$ , the pair $( \mathbf { S } \mathbf { W } _ { Q } , \mathbf { S } ^ { - \top } \mathbf { W } _ { K } )$ creates the same quadratic form.

$$
P r o o f . ~ ( \mathbf { S } ^ { - \top } \mathbf { W } _ { K } ) ^ { \top } ( \mathbf { S } \mathbf { W } _ { Q } ) = \mathbf { W } _ { K } \mathbf { S } ^ { - 1 } \mathbf { S } \mathbf { W } _ { Q } = \mathbf { W } _ { K } \mathbf { W } _ { Q }
$$

Our examples above the lemma furnish specific examples: we can swap the rows of our weights around without efect, or more generally rotate them in any way, or scale up one set of weights as long as we comparably shrink the other. This fact is corroborated by an analysis of directional derivatives, which is actually a distinct result about locally linear approximations relevant to training. Generally speaking, this invariance should be a reasonable concern for gradient style optimization methods, unless there are some other (natural or imposed) corrective mechanisms at play.

This also means there are $D ^ { 2 }$ out of $2 D E$ “unidentified” degrees of freedom in any $D \times E$ attention head. That is, only a fraction

$$
\frac { 2 D E - D ^ { 2 } } { 2 D E } = \frac { 2 E - D } { 2 E } = 1 - \frac { 1 } { 2 } \frac { D } { E }
$$

of the parameters have “uniquely meaningful” efects, though we can’t necessarily find ”the specific ones”. If we used a full rank attention head with $D = E$ , only $1 / 2$ of the parameters are “uniquely meaningful” in this sense; but note that the smaller $D { \mathrm { ~ i s } } ,$ i.e. the lower rank the query-key transformation ${ \mathrm { i s } } ,$ the more identification we can obtain. But it can never be “all of the parameters”, because the best we can do is $1 - 1 / ( 2 E )$ (with $D = 1 )$ .

We should comment that it is hard to envision how one could directly and efectively capitalize on this fact by training over only “uniquely meaningful” parameters. This point hopefully gets clearer with the additional exposition below.

4.3. A “Principled” Approach. We can view this efect a diferent if laborious way:

Lemma 4.2. For any ${ \bf W } _ { { Q } } , { \bf W } _ { K } \in \mathbb { R } ^ { D \times E }$ there are matrices $\bar { \mathbf { U } } _ { Q } , \bar { \mathbf { U } } _ { K } \in$ $\mathbb { R } ^ { E \times D }$ with orthonormal columns and a vector ${ \pmb { \sigma } } \in \mathbb { R } ^ { D }$ with non-negative entries such that

$$
\begin{array} { r } { \bar { \bf W } _ { Q } = \mathrm { d i a g } ( { \pmb \sigma } ) \bar { \bf U } _ { Q } \quad , \quad \bar { \bf W } _ { K } = \mathrm { d i a g } ( { \pmb \sigma } ) \bar { \bf U } _ { K } } \end{array}
$$

has the same quadratic form as does $( \mathbf { W } _ { Q } , \mathbf { W } _ { K } )$

Proof. Take an SVD of the (square) product $\mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } = \bar { \mathbf { U } } \bar { \Sigma } \bar { \mathbf { V } } ^ { \top }$ where $\bar { \mathbf { U } } , \bar { \mathbf { V } } \in \mathbb { R } ^ { E \times E }$ have orthonormal columns and $\bar { \Sigma } \in \bar { \mathbb { R } } ^ { E \times E }$ is a diagonal matrix with non-negative elements non-decreasing in index the last $E - D$ of which must be zero. The result follows immediately taking $\begin{array} { r } { \bar { \bf U } _ { Q } = \bar { \bf U } _ { 1 : D , : } , } \end{array}$ $\bar { \mathbf { U } } _ { K } = \bar { \mathbf { V } } _ { 1 : D , }$ <sub>:</sub> where $1 : D _ { : }$ , : notates taking first D rows and $\sigma _ { d } ^ { 2 } = \bar { \Sigma } _ { d , d }$ for $d = 1 , \ldots , D$ □

Note that the resulting “factorization” here is still signed-permutation invariant. That is, we can change signs and ordering without influencing the result product. However neither of those are linear efects, and thus perhaps not relevant from the perspective of gradient-based training.

Imagine for a moment we used this as a definition of ${ \bf W } _ { Q } , { \bf W } _ { K }$ . We would train over $2 D E + D$ parameters while constraining the matrices to be composed of orthonormal columns. Putting aside the practicality of doing this, we efectively reduce the number of degrees of freedom in each matrix by

$$
{ \frac { D ( D - 1 ) } { 2 } } + D = { \frac { D ( D + 1 ) } { 2 } }
$$

from the orthonormality constraints on the D columns. In total, we would have $2 D E + D$ parameters with only

$$
2 D E + D - 2 { \frac { D ( D + 1 ) } { 2 } } = 2 D E + D - D ( D + 1 ) = 2 D E - D ^ { 2 }
$$

actual degrees of freedom. Note the same outcome: $D ^ { 2 }$ fewer degrees of freedom.

Actually following through on this thought experiment, probably by keeping the query-key weights orthonormal and adding scaling values as an other term in the product, is likely impractical. Projected gradient methods might, in principle, allow us to maintain orthonormality over weight up dates but at the cost of 2 SVDs per attention layer update. Literally executing repeated factorizations of that sort during training would not scale.

4.4. Directional Derivatives. Unsurprisingly these invariances overlap with diferentiation. Our lemmas above formally apply to direct perturbations of weights but not local linearity of the query-keyquery-key quadratic form, which might behave diferently. Given that training updates are closely tied to diferentiation, it’s worth considering the same sort of efect from that viewpoint.

The $( \delta \mathbf { W } _ { Q } , \delta \mathbf { W } _ { K } )$ )-directional derivative at $( \mathbf { W } _ { Q } , \mathbf { W } _ { K } )$ of the quadratic form in attention is

$$
\mathbf { X } ^ { \top } \left( \mathbf { W } _ { K } ^ { \top } \delta \mathbf { W } _ { Q } + \delta \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \right) \mathbf { X }
$$

Such a derivative vanishes for any embedded token sequence (or it’s propagation through a network) when

$$
\mathbf { W } _ { K } ^ { \top } \delta \mathbf { W } _ { Q } + \delta \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } = \mathbf { 0 }
$$

For which subspaces of $( \delta \mathbf { W } _ { Q } , \delta \mathbf { W } _ { K } )$ does that hold? One trivial zero is obvious:

$$
( \delta { \bf W } _ { Q } , \delta { \bf W } _ { K } ) = ( { \bf W } _ { Q } , - { \bf W } _ { K } )
$$

as can be immediately checked. But we should ask for which other choices that could hold, and how many lost “degrees of freedom” that implies.

Lemma 4.3. Let $D \ \leq \ E$ and presume $\mathbf { W } _ { Q } , \mathbf { W } _ { K } \in \mathbb { R } ^ { D \times E }$ are full-rank (that $i s ,$ are rank $D )$ . There are directions $( \bar { \partial } \mathbf { W } _ { K } , \delta \mathbf { W } _ { Q } ) \in \mathbb { R } ^ { D \times E }$ derived from linear transformations of any $\mathbf { S } \in \mathbb { R } ^ { D \times D }$ in which $\mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q }$ has zero directional derivative.

Note here we do not require S to be invertible, unlike the result above.

Proof. Take (wide) SVDs

$$
\begin{array} { r } { { \bf W } _ { K } = { \bar { \bf U } } _ { K } { \bar { \boldsymbol \Sigma } } _ { K } \bar { \bf V } _ { K } ^ { \top } \qquad { \bf W } _ { Q } = \bar { \bf U } _ { Q } \bar { \boldsymbol \Sigma } _ { Q } \bar { \bf V } _ { Q } ^ { \top } } \end{array}
$$

where $\bar { \mathbf { U } } _ { K } , \bar { \pmb { \Sigma } } _ { K } \in \mathbb { R } ^ { D \times D }$ and $\bar { \mathbf { V } } _ { K } \in \mathbb { R } ^ { E \times D }$ with $\bar { \bf U } _ { K } ^ { \top } \bar { \bf U } _ { K } = { \bf I }$ and $\Sigma _ { K }$ diagonal and also invertible (the same holds for the $Q$ subscript). For any $\mathbf { S } \in \mathbb { R } ^ { D \times D }$

define

$$
\delta { \mathbf W } _ { Q } ( { \mathbf S } ) = \bar { \mathbf { U } } _ { K } \bar { \boldsymbol { \Sigma } } _ { K } ^ { - 1 } { \mathbf S } \bar { \mathbf { V } } _ { Q } ^ { \top } , \delta { \mathbf W } _ { K } ( { \mathbf S } ) = - \bar { \mathbf { U } } _ { Q } \bar { \boldsymbol { \Sigma } } _ { Q } ^ { - 1 } { \mathbf S } ^ { \top } \bar { \mathbf { V } } _ { K } ^ { \top }
$$

Then

$$
{ \bf W } _ { K } ^ { \top } \delta { \bf W } _ { Q } ( { \bf S } ) = \bar { \bf V } _ { K } \bar { \bf \Sigma } _ { K } \bar { \bf U } _ { K } ^ { \top } \bar { \bf U } _ { K } \bar { \bf \Sigma } _ { K } ^ { - 1 } { \bf S } \bar { \bf V } _ { Q } ^ { \top } = \bar { \bf V } _ { K } { \bf S } \bar { \bf V } _ { Q } ^ { \top }
$$

and

$$
\delta \mathbf { W } _ { K } ( \mathbf { S } ) ^ { \top } \mathbf { W } _ { Q } = - \bar { \mathbf { V } } _ { K } \mathbf { S } \bar { \mathbf { \Sigma } } _ { Q } ^ { - 1 } \bar { \mathbf { U } } _ { Q } ^ { \top } \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \Sigma } } _ { Q } \bar { \mathbf { V } } _ { Q } ^ { \top } = - \bar { \mathbf { V } } _ { K } \mathbf { S } \bar { \mathbf { V } } _ { Q } ^ { \top }
$$

and thus

$$
\mathbf { W } _ { K } ^ { \top } \delta \mathbf { W } _ { Q } ( \mathbf { S } ) + \delta \mathbf { W } _ { K } ( \mathbf { S } ) ^ { \top } \mathbf { W } _ { Q } = \mathbf { 0 }
$$

This proves the claim that the directional derivatives of $\mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q }$ are zero in the directions $( \delta { \bf W } _ { K } ( { \bf S } ) , \delta { \bf W } _ { Q } ( { \bf S } ) )$ , for any S, for all inputs X. □

The following addition is trivially verifiable:

Corollary 4.4. The set of perturbations $\mathbf { S } \  \ ( \pmb { \delta } \mathbf { W } _ { Q } ( \mathbf { S } ) , \pmb { \delta } \mathbf { W } _ { K } ( \mathbf { S } ) )$ that have zero directional derivative derived in the proof of the lemma is a linear subspace of dimension $D ^ { 2 }$

Also, our trivial zero is included in the subspace of zeros:

Corollary 4.5. $( \mathbf { W } _ { Q } , - \mathbf { W } _ { K } )$ is in the subspace of zeros of the directional derivative of the quadratic form derived above.

Proof. Set $\begin{array} { r } { \mathbf { S } = \bar { \Sigma } _ { K } \bar { \mathbf { U } } _ { K } ^ { \top } \bar { \mathbf { U } } _ { Q } \bar { \Sigma } _ { Q } } \end{array}$ and verify

$$
\begin{array} { r l } & { \delta { \mathbf { W } } _ { K } ( { \mathbf { S } } ) = \bar { \mathbf { U } } _ { K } \bar { \mathbf { S } } _ { K } ^ { - 1 } \bar { \mathbf { \bar { \mathbf { \Xi } } } } _ { K } \bar { \mathbf { U } } _ { K } ^ { \top } \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \bar { \mathbf { \Xi } } } } _ { Q } \bar { \mathbf { V } } _ { Q } ^ { \top } } \\ & { \qquad = \bar { \mathbf { U } } _ { K } \bar { \mathbf { U } } _ { K } ^ { \top } \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \Xi } } _ { Q } \bar { \mathbf { V } } _ { Q } ^ { \top } = \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \Xi } } _ { Q } \bar { \mathbf { V } } _ { Q } ^ { \top } = \mathbf { W } _ { Q } } \\ & { \delta { \mathbf { W } } _ { Q } ( { \mathbf { S } } ) = - \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \Xi } } _ { Q } ^ { - 1 } ( \bar { \mathbf { \Xi } } _ { K } \bar { \mathbf { U } } _ { K } ^ { \top } \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \Xi } } _ { Q } ) ^ { \top } \bar { \mathbf { V } } _ { K } ^ { \top } } \\ & { \qquad = - \bar { \mathbf { U } } _ { Q } \bar { \mathbf { \Xi } } _ { Q } ^ { - 1 } \bar { \mathbf { \Xi } } _ { Q } \bar { \mathbf { U } } _ { Q } ^ { \top } \bar { \mathbf { U } } _ { K } \bar { \mathbf { \Xi } } _ { K } \bar { \mathbf { V } } _ { K } ^ { \top } } \\ &  \qquad = - \bar { \mathbf { U } } _ { Q } \bar { \mathbf { U } } _ { Q } ^ { \top } \bar { \mathbf { U } } _ { K } \bar { \mathbf { \Xi } } _ { K } \bar { \mathbf { V } } _ { K } ^ { \top } = - \bar { \mathbf { U } } _ { K } \bar { \mathbf { \Xi } } _  \end{array}
$$

to prove the result.

Importantly, this result does not say the actual value of the quadratic form is invariant for steps in this direction, but rather only that it is a higherthan-first-order efect. In particular, the actual change in the quadratic form is determined by characteristically messy second order efects in these directions:

$$
\begin{array} { r l } & { \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } + \mathbf { W } _ { K } ^ { \top } \delta \mathbf { W } _ { Q } ( \mathbf { S } ) + \delta \mathbf { W } _ { K } ( \mathbf { S } ) ^ { \top } \mathbf { W } _ { Q } + \delta \mathbf { W } _ { K } ( \mathbf { S } ) ^ { \top } \delta \mathbf { W } _ { Q } ( \mathbf { S } ) } \\ & { \quad \quad \quad = \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } + \delta \mathbf { W } _ { K } ( \mathbf { S } ) ^ { \top } \delta \mathbf { W } _ { Q } ( \mathbf { S } ) \qquad \mathrm { ( f o r ~ t h e ~ l i n e a r l y ~ i n v a r i a n t ~ s t e p s ) } } \\ & { \quad \quad = \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } + \bar { \mathbf { V } } _ { K } \mathbf { S } \bar { \mathbf { \Sigma } } _ { Q } ^ { - 1 } \bar { \mathbf { U } } _ { Q } ^ { \top } \bar { \mathbf { U } } _ { K } \bar { \mathbf { \Sigma } } _ { K } ^ { - 1 } \mathbf { S } \bar { \mathbf { V } } _ { Q } ^ { \top } } \end{array}
$$

4.5. Multi-Head Attention Improves Identification. The results above apply immediately to each head in a multi-head attention and imply our main result. Suppose we split into H heads where H divides E and set $D = E / H$ . This is, in fact, the basic idea in Vaswani et al. (2023) as well as implementations like nanoGPT. By the same logic as following our invariance lemma, there are $2 H D E = 2 E ^ { 2 }$ parameters in all the attention head querykey products (that is, not including value and output pooling weights), of which $H D ^ { 2 } = \dot { E } ^ { 2 } / H$ are unidentified, $D ^ { 2 }$ from each head. Equivalently we have a fraction

$$
1 - { \frac { H D ^ { 2 } } { 2 H D E } } = 1 - { \frac { D } { 2 E } } = 1 - { \frac { E } { 2 H E } } = 1 - { \frac { 1 } { 2 H } }
$$

identified parameters or degrees-of-freedom. Hence we can increase the identified fraction of parameters just by using more heads. Ignoring all other considerations, we can use up to E rank-one heads and again obtain “mean-$\mathrm { i n g } ^ { , , }$ for a fraction $1 - 1 / ( 2 E )$ of parameters at most. That is likely a poor actual strategy due to the computationally intensive nature of a Softmaxbased attention head, and some batching $( D > 1 )$ is probably valuable.

Importantly it shouldn’t take a large number of heads to see real benefits. For example with 6 heads $1 1 / 1 2$ or about 83% of the parameters have identified efects, and with 8 heads already $1 5 / 1 6$ or about 94% of the parameters have identified efects.

Again this result doesn’t directly account for the output projections involving the multiplication of $\mathbf { W } _ { O } ^ { h }$ and $\mathbf { W } _ { V } ^ { h }$ weights. But we again have weight matrix products like $\mathbf { W } _ { O } ^ { h } \mathbf { W } _ { V } ^ { h }$ which, even if low rank, have invariances that reduce degrees of freedom. This is the ${ } ^ { 4 \cdot } \mathrm { O V - C i r c u i t } { } ^ { 3 \cdot }$ in Elhage et al. (2021), and exactly the same situation occurs for these matrix products (Zhao, Walters, and Yu, 2025):

Lemma 4.6. For any pair $\mathbf { W } _ { O } \in \mathbb { R } ^ { E \times C }$ and $\mathbf { W } _ { V } \in \mathbb { R } ^ { C \times E }$ and any nonsingular S $\in \mathbb { R } ^ { C \times C }$ , the pair $( \mathbf { W } _ { O } \mathbf { S } ^ { - 1 } , \mathbf { S } \mathbf { W } _ { V } )$ has the same product as $( \mathbf { W } _ { O } , \mathbf { W } _ { V } )$ . Moreover the directional derivatives of the product $\mathbf { W } _ { O } \mathbf { W } _ { V }$ with respect to directions in both weight matrices vanish on a subspace of dimension $C ^ { 2 }$

Proof. The first claim is trivial. The second is proved with SVDs in the same manner as before, with a minor notational change:

$$
\begin{array} { r } { \delta { \bf W } _ { O } = \bar { \bf U } _ { O } { \bf S } \bar { \boldsymbol { \Sigma } } _ { V } ^ { - 1 } { \bf U } _ { V } ^ { \top } \qquad \delta { \bf W } _ { V } = \bar { \bf V } _ { O } { \bf S } \bar { \boldsymbol { \Sigma } } _ { O } ^ { - 1 } { \bf V } _ { V } ^ { \top } } \end{array}
$$

where $\mathbf { W } _ { O } = \bar { \mathbf { U } } _ { O } \bar { \boldsymbol { \Sigma } } _ { O } \mathbf { U } _ { O } ^ { \top }$ and $\mathbf { W } _ { V } = \bar { \mathbf { U } } _ { O } \bar { \boldsymbol { \Sigma } } _ { V } \mathbf { U } _ { V } ^ { \top }$ are SVDs.

In an approach like that encoded in nanoGPT, $C = D$ and we again have $D ^ { 2 }$ unidentified degrees of freedom. In total, a multi-head attention of that sort with $D = E / H$ (for an H that divides E) batches $4 E ^ { 2 } = 4 E H D$ parameters for queries, keys, values, and output projection. But $2 H D ^ { 2 } =$ $2 E ^ { 2 } / H$ of the parameters are unidentified, so that we can still only really

train a fraction

$$
1 - \frac { 2 E ^ { 2 } / H } { 4 E ^ { 2 } } = 1 - \frac { 1 } { 2 H }
$$

of these into real efects. In other words, the overall result about the fraction of meaningful parameters has not changed.

As an aside, we could consider $H = 1 , D = C = E$ a formulation of a single-head attention, with of course the same outcome about the identified fraction of parameters.

4.6. Invariance with Query-Key Bias Terms. If we include query-key biases we have to modify our invariance results slightly. Observe

$$
\begin{array} { r l } & { ( \mathbf { W } _ { K } \mathbf { X } + \mathbf { b } _ { K } \mathbf { 1 } ^ { \top } ) ^ { \top } ( \mathbf { W } _ { Q } \mathbf { X } + \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } ) } \\ & { \qquad = \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } + \mathbf { 1 } \mathbf { b } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \big ( \mathbf { b } _ { K } ^ { \top } \mathbf { b } _ { Q } \big ) \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}
$$

The mixtures mean if we perturb $( \mathbf { W } _ { Q } , \mathbf { W } _ { K } )$ we also have to perturb $\left( \mathbf { b } _ { Q } , \mathbf { b } _ { K } \right)$ Specifically consider

$$
( \mathbf { \sigma } ( \mathbf { W } _ { Q } , \mathbf { b } _ { Q } ) , \mathbf { \sigma } ( \mathbf { W } _ { K } , \mathbf { b } _ { K } ) )  ( \mathbf { \sigma } ( \mathbf { S } \mathbf { W } _ { Q } , \mathbf { b } _ { Q } ) , \mathbf { \sigma } ( \mathbf { S } ^ { - \top } \mathbf { W } _ { K } , \mathbf { b } _ { K } ) )
$$

resulting in query-key products

$$
\mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { S } ^ { - 1 } \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } + \mathbf { 1 } \mathbf { b } _ { K } ^ { \top } \mathbf { S } \mathbf { W } _ { Q } \mathbf { X } + \left( \mathbf { b } _ { K } ^ { \top } \mathbf { b } _ { Q } \right) \mathbf { 1 } \mathbf { 1 } ^ { \top }
$$

which is not an invariant result. But

$$
( \ ( \mathbf { W } _ { Q } , \mathbf { b } _ { Q } ) , \ ( \mathbf { W } _ { K } , \mathbf { b } _ { K } ) \ )  ( \ ( \mathbf { S } \mathbf { W } _ { Q } , \mathbf { S } \mathbf { b } _ { Q } ) , \ ( \mathbf { S } ^ { - \top } \mathbf { W } _ { K } , \mathbf { S } ^ { - \top } \mathbf { b } _ { K } ) \ )
$$

for which we would have

$$
\begin{array} { r l } & { \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { S } ^ { - 1 } ( \mathbf { S } \mathbf { b } _ { Q } ) \mathbf { 1 } ^ { \top } + \mathbf { 1 } ( \mathbf { S } ^ { - \top } \mathbf { b } _ { K } ) ^ { \top } \mathbf { S } \mathbf { W } _ { Q } \mathbf { X } + \mathbf { 1 } ( \mathbf { S } ^ { - \top } \mathbf { b } _ { K } ) ^ { \top } ( \mathbf { S } \mathbf { b } _ { Q } ) \mathbf { 1 } ^ { \top } } \\ & { \qquad = \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } ( \mathbf { S } ^ { - 1 } \mathbf { S } ) \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } + \mathbf { 1 } \mathbf { b } _ { K } ^ { \top } ( \mathbf { S } ^ { - 1 } \mathbf { S } ) \mathbf { W } _ { Q } \mathbf { X } + \mathbf { 1 } \mathbf { b } _ { K } ^ { \top } ( \mathbf { S } ^ { - 1 } \mathbf { S } ) \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } } \\ & { \qquad = ( \mathbf { W } _ { K } \mathbf { X } + \mathbf { b } _ { K } \mathbf { 1 } ^ { \top } ) ^ { \top } ( \mathbf { W } _ { Q } \mathbf { X } + \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } ) } \end{array}
$$

4.7. Irrelevant Bias Terms. Most of the above has ignored bias terms. Vaswani et al. (2023) don’t mention bias terms, perhaps suggesting they can be excluded in transformations related to attention. But code like nanoGPT (Karpathy, 2022) includes them, at least optionally, and the review from Google Brain (Phuong and Hutter, 2022) explicitly admits biases on all terms. Generally speaking tooling like pytorch (Paszke et al., 2019) is pretty liberal in allowing trainable linear parameters in various operations, in principle admitting identification-confounding linear operations like those discussed above.

The purpose of this section is to cover some basic results related in character but not substance: bias terms related to the key values are irrelevant in any Softmax-based attention head, and bias related to value projections are irrelevant in any multi-head attention head. While much of the above derives from analysis of the quadratic form implied by query-key products, the Softmax has it’s own “invariant subspaces”. In particular, a Softmax is invariant to value changes that are constant over columns with one of the biases we might use having only a constant efect on columns. Multi-head attention introduces another invariance in biases with output pooling.

The number of parameters we can omit from these observations about biases is small relative to the ”second order” invariances in query-key products. We never have to include $H D = E$ (when $C = D$ and H divides $E )$ key biases or HC value biases per multi-head attention layer, or when $C = D$ only 2E parameters per layer in total.

4.7.1. Key Bias Has No Efect. As is well known any uniform shift in all exponentiated ”logits” doesn’t afect a logistic/Softmax:

Fact 4.7. For any numbers $u _ { 1 } , \ldots , u _ { N }$

$$
{ \frac { \exp \{ u _ { n } + \delta \} } { \sum _ { m } \exp \{ u _ { m } + \delta \} } } = { \frac { e ^ { \delta } \exp \{ u _ { n } \} } { \sum _ { m } e ^ { \delta } \exp \{ u _ { m } \} } } = { \frac { e ^ { \delta } \exp \{ u _ { n } \} } { e ^ { \delta } \sum _ { m } \exp \{ u _ { m } \} } } = { \frac { \exp \{ u _ { n } \} } { \sum _ { m } \exp \{ u _ { m } \} } }
$$

This is actually the basis of a classical computational trick (used in pytorch and most serious implementations of logistics, expit, or softmax) to assert the exponents are never larger than one: just subtract away the maximum value from each value prior to exponentiating to avoid overflows.

Inside a So $\operatorname { \dot { t m a x } } _ { \operatorname { c o l } }$ operation we’re going to efectively do something like

$$
\frac { \exp \{ \mathbf { k } _ { d } ^ { \top } \mathbf { q } \varrho \} } { \sum _ { c } \exp \{ \mathbf { k } _ { c } ^ { \top } \mathbf { q } \varrho \} }
$$

which will have the same property. We leave of the sum’s upper limit to emphasize this could be “causal” or not. Again suppose we added query-key biases,

$$
\begin{array} { r l } & { \mathbf { K } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ( \mathbf { X } ) = \left( \mathbf { W } _ { K } \mathbf { X } + \mathbf { b } _ { K } \mathbf { 1 } ^ { \top } \right) ^ { \top } \left( \mathbf { W } _ { Q } \mathbf { X } + \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } \right) } \\ & { \qquad = \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { b } _ { Q } \mathbf { 1 } ^ { \top } + \mathbf { 1 } \mathbf { b } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } + \left( \mathbf { b } _ { K } ^ { \top } \mathbf { b } _ { Q } \right) \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}
$$

Note that the last term, $( \mathbf { b } _ { K } ^ { \top } \mathbf { b } _ { Q } ) \mathbf { 1 1 } ^ { \top }$ , is a constant applied to all rows and columns. Note also the third term, $\mathbf { 1 } \mathbf { b } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X }$ , is constant over columns but may difer over rows. So when we take $\operatorname { S o f t m a x } _ { \mathrm { c o l } }$ over this result, those two terms will be irrelevant (drop out of the exponential fractions), and only the query bias ${ \bf b } _ { Q }$ will have an impact on the attention layer output.

4.7.2. Value Bias is Unidentified in a Multi-Head Attention. Suppose we use multi-head attention and include value biases,

$$
\mathbf { W } _ { V } ^ { h } \mathbf { Y } ^ { h } + \mathbf { b } _ { V } ^ { h } \mathbf { q } ^ { \top } \quad \quad \mathbf { Y } ^ { h } ( \mathbf { X } ) = \mathbf { X } \operatorname { S o f t m a x } _ { \mathrm { c o l } } \left( \mathbf { K } ^ { h } ( \mathbf { X } ) ^ { \top } \mathbf { Q } ^ { h } ( \mathbf { X } ) \right)
$$

keeping in mind we will then pool head efects in an afine form

$$
\sum _ { h = 1 } ^ { H } { \mathbf W } _ { O } ^ { h } \left( { \mathbf W } _ { V } ^ { h } { \mathbf Y } ^ { h } ( { \mathbf X } ) + { \mathbf b } _ { V } ^ { h } { \mathbf 1 } ^ { \top } \right) + { \mathbf b } _ { O } { \mathbf 1 } ^ { \top }
$$

This is obviously expanded to

$$
\sum _ { h = 1 } ^ { H } \mathbf { W } _ { O } ^ { h } \mathbf { W } _ { V } ^ { h } \mathbf { Y } ^ { h } ( \mathbf { X } ) + \mathbf { b } _ { V , O } \mathbf { 1 } ^ { \top } \qquad \mathbf { b } _ { V , O } = \sum _ { h = 1 } ^ { H } \mathbf { W } _ { O } ^ { h } \mathbf { b } _ { V } ^ { h } + \mathbf { b } _ { O }
$$

Numerical training will not be able to estimate anything other than $\mathbf { b } _ { V , O }$ not the individual head (or output) biases. In other words, we might as well simple drop the head-value biases and just use $\mathbf { b } _ { O }$

4.8. Extensions. In this short section we discuss a few extensions related to relevant literature.

4.8.1. Using Larger Heads. Suppose, say following Bhojanapalli et al. (2020), we let the per-head outputs be larger than the “conventional” $E / H$ We could look at this as a “representational fan $\mathrm { o u t } ^ { \dag }$ in the heads. The per head identification result about invariance is the same. Along with larger heads we get “less efect per parameter” in this sense. So let’s take an extreme: $D = E$ for all H heads. This is actually the limiting extreme, as we can’t have $D > E$ without purposefully introducing unidentified interactions. Because the fraction of unidentified parameters is like $D / ( 2 E )$ , with $1 / ( 2 H )$ falling out of the architectural choice $D = E / H$ . When $D = E$ this becomes $1 / 2$ just like a single-head attention. This potentially exemplifies a balance the literature has not yet observed: with $H = E$ and $D = E / H = 1$ we would have the highest fraction of identified parameters but have the most low rank representations; with $D = E$ and any H we have “high rank” representations but also the lowest fraction of identified parameters. Somewhere in between these efects likely balance each other. A coarse review of their models used for ablation studies is in ${ \mathrm { F i g . } }$ 1 shows that most of the incremental benefit happens with more identified parameters, not with significantly larger-than-conventional heads.

<table><tr><td> $d _ { p }$ </td><td>Total Params</td><td>Attn.</td><td>Unidentified</td><td> $\overline { { \triangle \mathrm { { \ F 1 } } } }$ </td><td> $\bigtriangleup$  EM</td></tr><tr><td>32</td><td>130</td><td>12.6</td><td>0.26M (0.20%)</td><td></td><td></td></tr><tr><td>64</td><td>142</td><td>25.2</td><td>1.05M (0.74%)</td><td>0.98</td><td>1.22</td></tr><tr><td>128</td><td>168</td><td>50.4</td><td>4.20M (2.50%)</td><td>0.09</td><td>0.32</td></tr><tr><td>256</td><td>218</td><td>100.9</td><td>16.82M (7.71%)</td><td>0.73</td><td>0.63</td></tr></table>

Figure 1. Review of SQuAD ablation studies in Bhojanapalli et al. (2020) from our lens of attention parameter identification. Note $d _ { p }$ is their notation, and $d _ { p } = 6 4$ is the “conventional” setting $D = E / H$

4.8.2. RoFormer/RoPE.. RoPE (Su et al., 2023) is efectively the following modification:

$$
\begin{array} { r } { \left( \mathbf { X } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } \mathbf { X } \right) _ { m , n } \mapsto \mathbf { x } _ { m } ^ { \top } \mathbf { W } _ { K } ^ { \top } \mathbf { R } _ { n - m } \mathbf { W } _ { Q } \mathbf { x } _ { n } } \end{array}
$$

for a family of orthogonal matrices $\left\{ \mathbf { R } _ { n - m } \right\}$ . Of course the specific elements and encoding grounded in $2 \times 2$ rotations is critical to the approach. Each sequence pair mapping will appear to have a similar invariance to that discussed above,

$$
\mathbf { W } _ { K } ^ { \top } \mathbf { R } _ { n - m } \mathbf { W } _ { Q } \equiv ( \mathbf { S } ^ { - \top } \mathbf { R } _ { n - m } ^ { \top } \mathbf { W } _ { K } ) ^ { \top } \mathbf { R } _ { n - m } ( \mathbf { R } _ { n - m } ^ { - 1 } \mathbf { S } \mathbf { W } _ { Q } )
$$

but not all valid at once. So we can consider group transformations like $( \mathbf { W } _ { Q } , \mathbf { W } _ { K } ) \mapsto ( \mathbf { A } \mathbf { W } _ { Q } , \mathbf { B } \mathbf { W } _ { K } )$ for which we must have

$$
\mathbf { W } _ { K } ^ { \top } \mathbf { R } _ { n - m } \mathbf { W } _ { Q } = \mathbf { W } _ { K } ^ { \top } \mathbf { B } ^ { \top } \mathbf { R } _ { n - m } \mathbf { A } \mathbf { W } _ { Q }
$$

and thus ${ \bf A } = { \bf R } _ { n - m } ^ { - 1 } { \bf B } ^ { - \top } { \bf R } _ { n - m }$ for all valid $n , m$ . A can’t depend on $n - m$ itself, and thus A (efectively) has to satisfy a set of commutation relationships.

So, suppose $\mathbf { S } \in \mathbb { R } ^ { D \times D }$ commutes with all ${ \bf R } _ { n - m } \left( { \bf R } _ { n - m } { \bf S } = { \bf S } { \bf R } _ { n - m } \right)$ ; then

$$
\begin{array} { r l } { ( \mathbf S ^ { - \top } \mathbf W _ { K } ) ^ { \top } \mathbf R _ { n - m } ( \mathbf S \mathbf W _ { Q } ) = \mathbf W _ { K } ^ { \top } \mathbf S ^ { - 1 } \mathbf R _ { n - m } \mathbf S \mathbf W _ { Q } } & { } \\ { = \mathbf W _ { K } ^ { \top } \mathbf S ^ { - 1 } \mathbf S \mathbf R _ { n - m } \mathbf W _ { Q } } & { ( \mathrm { c o m m u t a t i v i t y } ) } \\ { = \mathbf W _ { K } ^ { \top } \mathbf R _ { n - m } \mathbf W _ { Q } } \end{array}
$$

So our now-valid family of $D \times D$ matrices has additional constraints related to commutativity. For the particular $D / 2$ planar rotations $\mathbf { R } _ { n - m }$ in RoPE commuting matrices have two free numbers: any other rotation over the same subspace and a scaling. Handwaving a bit, that means an S in an efectively similar $2 \times 2$ block-diagonal form encoded in complex notation as $\alpha _ { d } e ^ { i \theta _ { d } }$ for $D / 2$ scalings $\alpha _ { d }$ and rotations $\theta _ { d }$ , having a total of $D$ not $D ^ { 2 }$ degrees of freedom.

We’ve efectively shown:

Proposition 4.8. Using a RoPE encoding in a transformer we have Ddimensional invariance in query-key products $( ( \mathbb { C } \backslash \{ 0 \} ) ^ { D / 2 }$ instead of $G L _ { D } ( \mathbb { R } ) )$ of the form

$$
\mathbf { S } = \left( \begin{array} { c c c c } { \alpha _ { 1 } e ^ { i \theta _ { 1 } } } & { \mathbf { 0 } } & { \cdot \cdot \cdot } & { \mathbf { 0 } } \\ { \mathbf { 0 } } & { \alpha _ { 2 } e ^ { i \theta _ { 2 } } } & { \cdot \cdot \cdot } & { \mathbf { 0 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \mathbf { 0 } } & { \mathbf { 0 } } & { \cdot \cdot \cdot } & { \alpha _ { D / 2 } e ^ { i \theta _ { D / 2 } } } \end{array} \right)
$$

4.9. Grouped Query Attention. We can sketch GQA as defining G key groups for H heads using

$$
( \mathbf { W } _ { K } ^ { g ( h ) } ) ^ { \top } \mathbf { W } _ { Q } ^ { h } \qquad h = 1 , \dots , H \ g : \{ 1 , \dots , H \} \to \{ 1 , \dots , G \}
$$

There will thus similar symmetry except every group G has its own. That is,

$$
( \mathbf { W } _ { K } ^ { g ( h ) } ) ^ { \top } \mathbf { W } _ { Q } ^ { h } \equiv ( ( \mathbf { S } ^ { g ( h ) } ) ^ { - \top } \mathbf { W } _ { K } ^ { g ( h ) } ) ^ { \top } \mathbf { S } ^ { g ( h ) } \mathbf { W } _ { Q } ^ { h }
$$

So we have the same identification properties per group, not per head. The identifications are then

$$
\mathbf { W } _ { Q } ^ { h } \mapsto \mathbf { S } ^ { g ( h ) } \mathbf { W } _ { Q } ^ { h } \quad , \quad \mathbf { W } _ { K } ^ { g } \mapsto ( \mathbf { S } ^ { g } ) ^ { - \top } \mathbf { W } _ { K } ^ { g }
$$

for $h = 1 , \ldots , H$ and $g = 1 , \ldots , G$ . This is a complete description of the invariance, any other transformation pairs $\{ \mathbf A ^ { h } \} _ { h }$ and $\{ \mathbf { B } _ { g } \} _ { g }$ would need to satisfy $A _ { h } = B _ { g ( h ) } ^ { - \top }$ . When $G < H$ this is a proper constraint and will decrease the count of unidentified parameters from $H D ^ { 2 }$ to $G D ^ { 2 }$ . GQA thus cuts unidentified parameters like $H / G$

RoPE reduces symmetry and thus improves identification by a rotation not all transformations commute with, while GQA reduces it by tying heads together so they can no longer transform independently. Some models use both RoPE and GQA meaning group-identified invariance that also commute with the rotations, where the two efects compose: a $\mathrm { R o P E + G Q A }$ layer has GD unidentified parameters against $H \cdot D ^ { \widehat { 2 } }$ for more basic transformers. Specifically for Llama-2-70B parameters $( H = 6 4 , G = 8 , D = 1 2 8 )$ that is factor of 8 diference: $8 \times 1 2 8 ^ { 2 } = 1 3 1 , 0 7 2$ versus 1, 048, 576.

4.10. Equivalence. Returning to multi-head attention (MHA), we could ask the architectural question of whether there is a comparable single-head attention (SHA) from a purely identification standpoint. In other words, given $( E , H , D )$ (fixing $C = D )$ are there choices of $( E _ { 0 } , D _ { 0 } )$ that have the same “efective” parameter counts? Here we will ignore bias for simplicity, and look to equate

$$
( 2 D _ { 0 } + E _ { 0 } ) E _ { 0 } - D _ { 0 } ^ { 2 } = 4 E H D - 2 H D ^ { 2 }
$$

Consider first the default case where $D = C = E / H$ , H divides $E ,$ and $D _ { 0 } = E _ { 0 }$ . Then we efectively parameterize the MHA by $( E , H )$ , have $4 E ^ { 2 }$ parameters with $2 E ^ { 2 } / H$ unidentified. The SHA has $3 E _ { 0 } ^ { 2 }$ parameters with $E _ { 0 } ^ { 2 }$ unidentified. We can equate these as

$$
E _ { 0 } = \left\lceil E \sqrt { 2 - \frac { 1 } { H } } \right\rceil
$$

For $H = 1$ we obviously recover SHA, for

$$
E _ { 0 } = { \Bigg \lceil } E { \sqrt { 2 - { \frac { 1 } { 1 } } } } { \Bigg \rceil } = { \Big \lceil } E { \sqrt { 1 } } { \Big \rceil } = E
$$

For $H > 1$ this also plainly means we need a larger embedding dimension in SHA than in MHA. While using a larger embedding dimension is possible that will influence the entire network which we have mostly ignored. In the simplified case the inflation in model size could be large, roughly 60% larger for 12 heads. If executed that will result in a false comparison between the MHA-matched SHA because the capacity of other components like the residual skip layers and $\mathrm { M L P ^ { \prime } s }$ in a full transformer will have altered capacity as well.

Imposing this constraint, $E _ { 0 } = E$ , more generally we can get a no-go or impossibility result. Set $D _ { 0 } = \eta E _ { 0 }$ to prove this. We would have

$$
( 1 + 2 \eta - \eta ^ { 2 } ) E ^ { 2 } = 4 E H D - 2 H D ^ { 2 }
$$

or

$$
1 + 2 \eta - \eta ^ { 2 } = 2 H \left( 2 - \frac { D } { E } \right) \left( \frac { D } { E } \right)
$$

When $D = E / H , D / E = 1 / H$ and

$$
1 + 2 \eta - \eta ^ { 2 } = 2 \left( 2 - \frac { 1 } { H } \right)
$$

The LHS is a concave quadratic whose maximum is $2 \ ( \mathrm { a t } \ \eta = 1 )$ , whereas the RHS takes the value 3 at $H = 2$ (and increases from there); $H = 1$ again is a special case recovering the obvious equivalence. In other words, there is no comparable SHA for a given MHA with more than one head without changing, particularly increasing, the embedding dimension.

## 5. Numerical Examples

We have shown that there are specific “global” as well as locally linearly invariances in attention, it is not clear whether gradient-based updates avoid these subspaces. After all, shouldn’t a “direction of greatest increase” in some loss translate to implicitly avoiding subspaces over which there is no change? Moreover, the increase in relative identification from using a multi-head attention means more parameters are more useful to predictions. Moreover, can we evaluate how influential the improved identification from multi-head attention alone ${ \mathrm { i s } } ,$ to gauge how much else we should be further “reading into” the specific functional form of the attention?

## 5.1. Gradient Flow. Consider a gradient flow like

$$
\frac { d } { d t } ( { \bf W } _ { Q } , { \bf W } _ { K } ) = - \frac { \partial \mathcal { L } } { \partial ( { \bf W } _ { K } , { \bf W } _ { Q } ) }
$$

for a loss function ${ \mathcal { L } } .$

Lemma 5.1. $\mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } - \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top }$ is an invariant under gradient flow for a diferentiable loss.

This also dates back to Arora, Cohen, and Hazan (2018) as “imbalance” in a cross-layer setting.

Proof. Recognizing that the loss only depends on $\mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q }$ and letting $\mathbf { G } =$ $\partial \mathcal { L } / \partial ( \mathbf { W } _ { K } ^ { \top } \mathbf { W } _ { Q } )$ the chain rule implies

$$
{ \frac { \partial { \mathcal { L } } } { \partial \mathbf { W } _ { Q } } } = \mathbf { W } _ { K } \mathbf { G } \qquad { \frac { \partial { \mathcal { L } } } { \partial \mathbf { W } _ { K } } } = \mathbf { W } _ { Q } \mathbf { G } ^ { \intercal }
$$

Thus

$$
{ \frac { d } { d t } } ( \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } ) = - \left( { \frac { \partial { \mathcal { L } } } { \partial \mathbf { W } _ { Q } } } \mathbf { W } _ { Q } ^ { \top } + \mathbf { W } _ { Q } { \frac { \partial { \mathcal { L } } } { \partial \mathbf { W } _ { Q } } } ^ { \top } \right) = - \left( \mathbf { W } _ { K } \mathbf { G } \mathbf { W } _ { Q } ^ { \top } + \mathbf { W } _ { Q } \mathbf { G } ^ { \top } \mathbf { W } _ { K } ^ { \top } \right)
$$

and similarly

$$
{ \frac { d } { d t } } ( \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top } ) = - \left( { \frac { \partial { \mathcal { L } } } { \partial \mathbf { W } _ { K } } } \mathbf { W } _ { K } ^ { \top } + \mathbf { W } _ { K } { \frac { \partial { \mathcal { L } } } { \partial \mathbf { W } _ { K } } } ^ { \top } \right) = - \left( \mathbf { W } _ { Q } \mathbf { G } ^ { \top } \mathbf { W } _ { K } ^ { \top } + \mathbf { W } _ { K } \mathbf { G } \mathbf { W } _ { Q } ^ { \top } \right)
$$

using gradient flow for $d / d t \mathbf { W } _ { Q }$ and $d / d t { \bf W } _ { K }$ . This obviously implies

$$
\frac { d } { d t } ( \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } ) = \frac { d } { d t } ( \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top } )
$$

from which the result is simply this as

$$
\frac { d } { d t } ( \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } - \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top } ) = \mathbf { 0 } .
$$

□

Ignoring numerical concerns, this means we should be able to quantify the degree to which an actual training sequence “uses” unidenfitied parameters via computing $\begin{array} { r } { \frac { d } { d t } | | \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } - \mathbf { W } _ { K } \mathbf { \bar { W } } _ { K } ^ { \top } | | _ { F } } \end{array}$ so to speak. Using the theoretical gradients this should be zero (the norm should be constant-ish). But adaptive methods can deviate from this “conservation law”.

So consider a least squares problem

$$
\operatorname* { m i n } _ { \mathbf { S } \in \mathbb { R } ^ { D \times D } } \frac { 1 } { 2 } \Bigg \{ | | \delta \mathbf { W } _ { Q } - \mathbf { S W } _ { Q } | | _ { F } ^ { 2 } + | | \delta \mathbf { W } _ { K } - \mathbf { S } ^ { \top } \mathbf { W } _ { K } | | _ { F } ^ { 2 } \Bigg \}
$$

Stationarity would require

$$
\mathbf { S W } _ { Q } \mathbf { W } _ { Q } ^ { \top } + \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top } \mathbf { S } = \left( \delta \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } - \mathbf { W } _ { K } \delta \mathbf { W } _ { K } ^ { \top } \right)
$$

with a computable RHS yielding a Sylvester equation. There is a unique solution when the Gram matrices $\mathbf { G } _ { Q } = \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top }$ and $- \mathbf { G } _ { K } = - \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top }$ do not share eigenvalues. If we then compute eigendecompositions

$$
\mathbf { G } _ { Q } \mathbf { U } _ { Q } = \mathbf { U } _ { Q } \mathbf { \Lambda } _ { Q } \qquad \mathbf { G } _ { K } \mathbf { U } _ { K } = \mathbf { U } _ { K } \mathbf { \Lambda } \mathbf { \Lambda } \mathbf { \Lambda } _ { K }
$$

we have some eigenvalues $\lambda _ { Q , d } , \lambda _ { K , d } \geq 0$ for $d = 1 , \ldots , D$ from positive semidefiniteness of Gram matrices with strict inequality when the weights are full rank. Then there is a unique S when

$$
\lambda _ { Q , d } - \left( - \lambda _ { K , d ^ { \prime } } \right) = \lambda _ { Q , d } + \lambda _ { K , d ^ { \prime } } \neq 0
$$

for all $d , d ^ { \prime }$ . This will hold so long as at least one of the weight matrices is full rank, because then min $, \lambda _ { Q , d } > 0$ or min $_ { \cdot d } \lambda _ { K , d } > 0$ . We thus report a value

$$
\mu = \operatorname* { m i n } _ { d } \lambda _ { Q , d } + \operatorname* { m i n } _ { d } \lambda _ { K , d }
$$

which quantifies rank deficiency that should correlate to invalidating $D ^ { 2 }$ unidentified degrees of freedom.

Moreover when $\mu > 0$ we can solve for $\mathbf { S } = \mathbf { U } _ { K } \tilde { \mathbf { S } } \mathbf { U } _ { Q } ^ { \top }$ via

$$
\tilde { \mathbf { S } } \mathbf { \Lambda } _ { Q } + \mathbf { \Lambda } _ { K } \tilde { \mathbf { S } } = \mathbf { R } \quad , \quad \mathbf { R } = \mathbf { U } _ { K } ^ { \top } \left( \delta \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } - \mathbf { W } _ { K } \delta \mathbf { W } _ { K } ^ { \top } \right) \mathbf { U } _ { Q }
$$

which is a calculation of the RHS followed by elementwise divisions:

$$
\left( \tilde { \mathbf { S } } \right) _ { d , d ^ { \prime } } = \frac { ( \mathbf { R } ) _ { d , d ^ { \prime } } } { \lambda _ { Q , d } + \lambda _ { K , d ^ { \prime } } }
$$

Thus we have a formal solution S to the least squares problem, a measure

$$
\rho = \frac { | | \mathbf { S } \mathbf { W } _ { Q } | | _ { F } ^ { 2 } + | | \mathbf { S } ^ { \top } \mathbf { W } _ { K } | | _ { F } ^ { 2 } } { | | \delta \mathbf { W } _ { Q } | | _ { F } ^ { 2 } + | | \delta \mathbf { W } _ { K } | | _ { F } ^ { 2 } } \leq 1
$$

that describes a normalized projection length, and even an update modification if we were interested.

Note that eigendecompositions of the Gram matrices are relatively feasible being sized by $D$ . In our specific numerical examples from training we have $D = 6 4$ for which these operations are trivial on modern computers.

We will also report

$$
\boldsymbol { \beta } = | | \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \intercal } - \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \intercal } | | _ { F }
$$

which over a training run measures how conserved the conserved quantity is and

$$
\boldsymbol { \tau } = \operatorname { t r a c e } \left( \mathbf { W } _ { Q } \mathbf { W } _ { Q } ^ { \top } - \mathbf { W } _ { K } \mathbf { W } _ { K } ^ { \top } \right)
$$

which, alongside $\beta ,$ shows if deviation from conservation is signed.

5.2. Training Runs. We run training experiments on relatively “stock” nanoGPT (Karpathy, 2022) with the tiny shakespeare dataset using character tokens. This is to avoid questions that could arise from a completely custom implementation as well as include efects like dropout to show their influence (or not). Scale should not be an important factor for empirical validation of our results, so small scale tests are enough.

5.2.1. Training Cases. We run four cases to empirically explore our theoretical results:

(1) SGD with no weight decay and a learning rate of 0.25

(2) SGD with small weight decay $4 \times 1 0 ^ { - 4 }$ and a learning rate of 0.25 (to roughly match case 3)

(3) AdamW (the default) with weight decay 0.1 and a learning rate of $1 \times 1 0 ^ { - 3 }$

(4) AdamW with weight decay 0.1 and a learning rate of $1 \times 1 0 ^ { - 3 }$ but “rebalancing” weights every 200 iterations.

In each case we train for 5,000 steps and store sets of metrics every 10 steps including generic training metrics, step-specific bias gradient norms, and the query/key weight metrics discussed above $( \rho , \beta , \tau , \mathrm { a n d } \mu )$ . Case (1) is worth including because it should run entirely orthogonally to unidentified parameter subspaces, by definition of the gradient. However nanoGPT will include numerical errors, dropout, and additional features that could break this result. Case (2) adds weight decay which formally adds a diagonal multiplier into weight updates we have not analyzed. Case (3) should have the best training performance but also has the highest potential to make changes changing unidentified parameters. Case (4) is an experiment to uncover if “fixing” AdamW step overlap with unidentified parameter subspaces has any efect on training. In principle the rebalancing we use destroys the higher order moment data so some degradation in training performance should be expected though overall our theory predicts that the efect should be minor.

We also run a sweep of case (1) with various learning rates to sketch out the influence of the related numerical discretization error that would occur. A priori we would expect larger steps to deviate further from idealistic gradient flow conservation roughly like the learning rate squared and can review results from that point of view. We run this sweep with learning rates 0.5, 0.25, 0.125, and 0.0625.

5.2.2. Rebalancing. We should clarify though what our rebalancing step really is to avoid confusion. Note that we could consider this a form of “quotienting out” the symmetry, which has a precedent in the literature (Zhao, Walters, and Yu, 2025). Our process is as follows: When rebalancing we reset

$$
\begin{array} { r l } & { \mathbf { M } = ( \mathbf { W } _ { K } ^ { ( h ) } ) ^ { \top } \mathbf { W } _ { Q } ^ { ( h ) } = \mathbf { Q } _ { K } ^ { ( h ) } \mathbf { R } _ { K } ^ { ( h ) } ( \mathbf { R } _ { Q } ^ { ( h ) } ) ^ { \top } ( \mathbf { Q } _ { Q } ^ { ( h ) } ) ^ { \top } } \\ & { \quad = \mathbf { Q } _ { K } ^ { ( h ) } \mathbf { U } _ { M } \Sigma \mathbf { V } _ { M } ^ { \top } ( \mathbf { Q } _ { Q } ^ { ( h ) } ) ^ { \top } = \left( \Sigma ^ { 1 / 2 } ( \mathbf { Q } _ { K } ^ { ( h ) } ) ^ { \top } \mathbf { U } _ { M } ^ { \top } \right) ^ { \top } \left( \Sigma ^ { 1 / 2 } \mathbf { V } _ { M } ^ { \top } ( \mathbf { Q } _ { Q } ^ { ( h ) } ) ^ { \top } \right) } \end{array}
$$

taking thin QR decompositions of $\mathbf { W } _ { Q } ^ { ( h ) } , \mathbf { W } _ { K } ^ { ( h ) }$ and an SVD for $\mathbf { R } _ { K } ^ { ( h ) } ( { \mathbf { R } _ { Q } ^ { ( h ) } } ) ^ { \top }$ That gives weight resets

$$
\bar { \mathbf { W } } _ { Q } ^ { ( h ) } = \Sigma ^ { 1 / 2 } \mathbf { V } _ { M } ^ { \top } ( \mathbf { Q } _ { Q } ^ { ( h ) } ) ^ { \top } \quad \quad \bar { \mathbf { W } } _ { K } ^ { ( h ) } = \Sigma ^ { 1 / 2 } ( \mathbf { Q } _ { K } ^ { ( h ) } ) ^ { \top } \mathbf { U } _ { M } ^ { \top }
$$

which provably zeros out the conserved quantity from gradient flow because

$$
\bar { \mathbf { W } } _ { Q } ^ { ( h ) } ( \bar { \mathbf { W } } _ { Q } ^ { ( h ) } ) ^ { \top } = \Sigma ^ { 1 / 2 } \mathbf { V } _ { M } ^ { \top } ( \mathbf { Q } _ { Q } ^ { ( h ) } ) ^ { \top } \mathbf { Q } _ { Q } ^ { ( h ) } \mathbf { V } _ { M } \Sigma ^ { 1 / 2 } = \Sigma
$$

$$
\bar { \mathbf { W } } _ { K } ^ { ( h ) } ( \bar { \mathbf { W } } _ { K } ^ { ( h ) } ) ^ { \top } = \boldsymbol { \Sigma } ^ { 1 / 2 } ( \mathbf { Q } _ { K } ^ { ( h ) } ) ^ { \top } \mathbf { U } _ { M } ^ { \top } \mathbf { U } _ { M } \mathbf { Q } _ { K } ^ { ( h ) } \boldsymbol { \Sigma } ^ { 1 / 2 } = \boldsymbol { \Sigma }
$$

Because there are still invariant rotations and permutations (we don’t solve for a projection) we also solve

$$
\operatorname* { m i n } _ { \mathbf { Z } ^ { \top } \mathbf { Z } = \mathbf { I } } | | \mathbf { Z } \bar { \mathbf { W } } _ { Q } ^ { ( h ) } - \mathbf { W } _ { Q } ^ { ( h ) } | | _ { F } ^ { 2 } + | | \mathbf { Z } \bar { \mathbf { W } } _ { K } ^ { ( h ) } - \mathbf { W } _ { K } ^ { ( h ) } | | _ { F } ^ { 2 }
$$

with another $\operatorname { S V D }$ to get a specific choice of the update closest to prerebalanced step.

5.3. Training Results. A summary of the results for a single representative run is presented in Fig. 2.

Note that, in the second column, ρ is plotted along with $1 / ( 2 H )$ and the max update norm. The closer $\rho$ is to $1 / ( 2 H )$ the more “unidentified” actual steps are. We can clearly see pure SGD steps far away from that boundary, decaying weight steps become closer, and AdamW is closest to the boundary with some layer head updates efectively covering the unidentified invariant space. We also show the update norm to clarify “spikes” in $\rho$ towards the end of training. The smaller the update norm, the more ill-posed computing $\rho$ becomes, and thus these spikes are explained as a computational artifact. See also Fig. 4.

The third column shows $\beta ,$ which should be conserved for gradient flow whose orbits are orthogonal to symmetries creating unidentified parameters. SGD keeps $\beta$ constant (nonzero simply from initialization), SGD with weight decay slowly decreases $\beta ,$ , while AdamW clearly does not keep $\beta$ conserved.

The fourth column shows $\tau$ which is related to a signed $\beta .$ . For SGD with and without weight decay this is efectively zero, suggestive of a mean-zero initial state for the conserved diference in Gram matrices. The story is entirely diferent for AdamW. $\tau$ spreads over training, in both positive and negative directions (sign related to queries vs keys), with the largest changes in the earliest layer.

The fifth column with $\mu$ exists just to show we aren’t seeing rank deficiency in these training runs. Rank deficiency would appear as lines dropping to zero, which does not occur. Across all runs this is consistent with our results showing the existence of unidentified parameters when not paired with rank deficiency. Again $\mu$ is stable for SGD, while AdamW is showing significantly diferent behavior for this metric of the spectrum of the query and key weights.

The rebalanced AdamW in the last row proves out the idea that a “proximal orbit correction” during training to bring iterates back to a canonical set of weights with respect to attention weight symmetries has no deleterious efect on training. Comparing “norma $\mathrm { l } ^ { \dag \dag }$ AdamW to periodic corrections does not change the overall loss but (by construction) regularly zeros out the conserved quantities derived from the unidentified parameter symmetries. Specifically $\beta$ exhibits a sawtooth pattern around the rebalances (with a decaying envelope related to the learning rate) and $\tau ,$ the signed deviation from conservation using trace, looks much more like plain SGD. As an aside, we ran but do not display the rebalanced AdamW training with optimizer clearing but no weight changes and recovered efectively “plain” AdamW behavior. That is, the change in behavior is from the rebalancing, not optimizer state clearing.

Fig. 3 supports our theoretical results about bias terms. For simplicity, we focus on the query-key biases, as the value-output claims in our theory are both harder to compute (coupled together by weights with output bias) and also may be influenced by dropout in training (while the claims are unafected by that in forward inference). The conclusion from this figure is simply confirmation: regardless of the optimizer approach the key layer bias has numerically zero gradients, while the query layer bias has small but not likely numerically zero gradients. This is exactly what we claimed to prove.

Results of our sweep of SGD without weight decay at descending learning rates are shown in Fig. 4. One efect we can again see is the spiking of $\rho$ with smaller updates. Another is that “drift”, the norm of the step-wise diference in the conserved quantity, is larger for earlier layers, stabilizes over training, and for all layers/heads decreases with learning rate. This suggests a story about our simplest optimizer in which discretization error from the learning rate itself is driving lack of conservation.

## 6. Conclusions

The original Vaswani et al. (2023) paper introduced transformers and specifically motivated the introduction of multi-head attention as a way to attend to distinct representational subspaces. In this note we have proved, both theoretically and empirically (albeit on a popular small somewhat toy implementation), that introducing additional heads also takes a transformer architecture from $1 / 2$ of parameters unidentified to $1 / ( 2 H )$ (50% to 80- 90% identified in published architectures). We have also shown matching the identification of the multi-head architecture is not possible in a naive single-head design. Part of our framing in the latter was to rule out an experiment we could otherwise do, but it has the side-efect of implying that the architectural choice is vital to obtaining the property.

We’ve generally motivated our work generally by stating “identification matters” which is easier said than done at the level of complexity in useful neural networks. However reflecting on the mathematics of attention based layer designs proves this out implicitly for architectures adopted for reasons other than purely statistical well-posedness with identification. Our main result is that using multiple heads implicitly improves identification, but was considered for representationality. RoPE improves it further via commutativity constraints but intends to impose pre-defined structure on the attended tokens suitable for language. GQA is presumably somewhere in between. In other words, while literature and practice has been innovating architectures to change how transformer networks build and weight features they have also changed how many network parameters have “meaning” in a basic statistical sense. This is a basic and seemingly unacknowledged confound for understanding performance improvement.

In the process this note has provided buttressing evidence for some other aspects of the study of transformers. We have strengthened the dimensionality of invariance (Zhang et al., 2025), shown that in the specific case of multi-head attention the architectural influence on identification is nontrivial, and that the conventional choice $( D = E / H$ where H divides $E )$ is likely a very good tradeof between representationality, eficiency, and identification (Bhojanapalli et al., 2020).

## Declaration of Generative AI and AI-assisted technologies in the writing process

In the preparation of this note the author(s) used Anthropic’s Claude (Opus 5, medium thinking) to prepare a formal referee-style review, suggest targeted related literature, provide suggestions for metrics and plots in the numerical experiements, and simplify the coding process for the numerical experiments. No direct text written by Claude was used in this note, and after using this tool, the author reviewed, edited, and re-derived the content as needed for both substance and style and takes full responsibility for the content.

## References

Ainslie, Joshua et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv: 2305.13245 [cs.CL]. url: https://arxiv.org/abs/2305.13245.

Arora, Sanjeev, Nadav Cohen, and Elad Hazan (2018). On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization. arXiv: 1802.06509 [cs.LG]. url: https://arxiv.org/abs/1802.06509.

Bhojanapalli, Srinadh et al. (2020). Low-Rank Bottleneck in Multi-head Attention Models. arXiv: 2002.07028 [cs.LG]. url: https://arxiv.org /abs/2002.07028.

Brandon, William et al. (2024). “Reducing Transformer Key-Value Cache Size with Cross-Layer Attention”. In: Advances in Neural Information Processing Systems. Ed. by A. Globerson et al. Vol. 37. Curran Associates, Inc., pp. 86927–86957. doi: 10.52202/079017-2758. url: https://pro ceedings.neurips.cc/paper\_files/paper/2024/file/9e23d020c18e 4c40d81c6a0fc7a46f68-Paper-Conference.pdf.

Elhage, Nelson et al. (2021). “A Mathematical Framework for Transformer Circuits”. In: Transformer Circuits Thread. url: https://transformer -circuits.pub/2021/framework/index.html.

Frankle, Jonathan and Michael Carbin (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. arXiv: 1803 . 03635 [cs.LG]. url: https://arxiv.org/abs/1803.03635.

Karpathy, Andrej (2022). nanoGPT. https://github.com/karpathy/nan ogpt.

Paszke, Adam et al. (2019). “PyTorch: An Imperative Style, High-Performance Deep Learning Library”. In: Advances in Neural Information Processing Systems. Ed. by H. Wallach et al. Vol. 32. Curran Associates, Inc., pp. 8024–8035. url: https://neurips.cc.

Phuong, Mary and Marcus Hutter (2022). Formal Algorithms for Transformers. arXiv: 2207.09238 [cs.LG]. url: https://arxiv.org/abs/22 07.09238.

Su, Jianlin et al. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv: 2104.09864 [cs.CL]. url: https://arxiv .org/abs/2104.09864.

Vaswani, Ashish et al. (2023). Attention Is All You Need. arXiv: 1706.03762 [cs.CL]. url: https://arxiv.org/abs/1706.03762.

Wang, Jie et al. (2024). Length Generalization of Causal Transformers without Position Encoding. arXiv: 2404.12224 [cs.CL]. url: https://arxi v.org/abs/2404.12224.

Zhang, Binchi et al. (2025). Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion. arXiv: 2502 . 00264 [cs.LG]. url: https://arxiv.org/abs/2502.00264.

Zhao, Bo, Robin Walters, and Rose Yu (2025). Symmetry in Neural Network Parameter Spaces. arXiv: 2506.13018 [cs.LG]. url: https://arxiv.o rg/abs/2506.13018.

Email address: morrowwr@gmail.com

URL: github.com/wrossmorrow

![](images/a6c716247496ef78ef58683167066dc08f6dc05145a95fdc9e62d03a56281477.jpg)  
Figure 2. Numerical results from 5000 training iterations on the $^ { \mathrm { { \sc ~ 5 9 } } } \mathrm { { t i n y } }$ shakespeare” datasets using character tokens with nanoGPT. First row: using plain SGD with no weight decay and a 0.25 base learning rate, where unidentified subspaces should be entirely avoided outside of numerical error. Second row: using plain SGD with 0.0004 weight decay matched to a base learning rate 0.25, exhibiting a compressive efect on the query-key trace imbalance discussed in the text. Third row: using AdamW with weight decay, showing in-training efects on both $\beta$ and the trace imbalance from training steps overlapping unidentified subspaces. Even ${ \mathrm { s o } } ,$ AdamW makes the most progress in reducing loss over the iteration range. Fourth row: AdamW with rebalancing.

![](images/47637b089753b3ad4f9d762284c8f522c530fc6cd7cf893db03fbdcf49afc3cb.jpg)  
Figure 3. Envelopes (min, max, median, and mean) of the key and query bias for three training runs as in Fig. 2. Optimizer or settings don’t significantly efect the bias gradient norms. Moreover the key bias norm is numerically zero, as required by our theoretical result.

![](images/e93b7bffbecc1420308daa61d712a7bae186660d68c5e4c6fad563847267cb27.jpg)  
Figure 4. Results for the SGD (no decay) sweep of training runs with descending learning rates. Drift here refers to the conserved quantity drift over a step. There is a steady decrease from higher learning rates to smaller learning rates suggesting influence of discretization error.