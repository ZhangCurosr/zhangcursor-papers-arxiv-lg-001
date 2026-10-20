# DELTATRACE: EFFICIENT SIGNED ATTRIBUTIONFOR REASONING LANGUAGE MODELS

Chen Liang<sup>1,2</sup> Daoping Wu<sup>2</sup> Donghua Yang<sup>1</sup> Zhiyu Liang<sup>1</sup> Ying Zhang<sup>2</sup> Shuguang Xiong<sup>2</sup> Hongzhi Wang<sup>1</sup>

<sup>1</sup>Harbin Institute of Technology <sup>2</sup>Zhejiang Lab

23B903050@stu.hit.edu.cn daopingwu@outlook.com yang.dh@hit.edu.cn {zyliang,wangzh}@hit.edu.cn {zhangy,xiongshuguang}@zhejianglab.org

## ABSTRACT

We study how words in the context support or oppose a language model’s response and present DELTATRACE, a novel attribution framework that explains the effect of input tokens on a fixed model response. DELTATRACE measures how the response’s log-probability changes when the context tokens being explained are masked. To distribute this difference among source tokens, it constructs operatorspecific local rules for finite changes and composes them backward through the computational graph using effective secant slopes. Each token is assigned a signed attribution as the inner product of its embedding displacement and the corresponding backward-propagated slope. Local conservation makes these contributions sum to the measured score difference and preserves their meaning when aggregated over sentences or passages. DELTATRACE supports both attention and linear-attention models, with auxiliary memory growing linearly with sequence length at fixed model dimensions and chunk size. On the evaluated attention model, DELTATRACE achieves the lowest deletion-faithfulness scores (RISE and MAS) on most retrieval and reasoning tasks and the highest needle-in-a-haystack Recall, exceeding FlashTrace by 5.67-24.33 percentage points. On the evaluated hybrid attention and linear-attention model, DELTATRACE lowers RISE on 9 tasks and MAS on 12 relative to FlashTrace.

## 1 INTRODUCTION

As people increasingly use large language models (LLMs) to support decisions, they need to know what information the models rely on. Input attribution helps answer this question by tracing a response back to the words in its context. A reasoning response depends on both the information available and the computations that select and combine it (Ferrando & Voita, 2024). Figure 1(d) shows a passage containing a playwright, William Shakespeare, and a composer, William Walton. The names supply candidate answers, and the phrase “the author of the play” specifies which role to use. Attribution should account for how both the candidate names and the requested role affect the response. It should also distinguish words that support the response from words that oppose it.

We ask how masking selected context tokens changes a language model’s score for a fixed response. We compare the original prompt with a reference in which the selected tokens are replaced by EOS, while the response and its preceding tokens remain fixed. The resulting score difference is divided into signed contributions for the selected tokens.

![](images/f606b5dfd3b780a7d32bc8c62650df5d360f724b04a07b2b32186e4619fe54d9.jpg)

![](images/8ddc247217f974ef14aa71bbe153b19a6a7a678fa2cf95a3b1c97c83f2446844.jpg)

![](images/34d10d7ea48817e82b06810c850b955907580dcc5426620276e83904f5271b5a.jpg)  
Figure 1: DeltaTrace assigns a response-score difference to input tokens. (a) Two executions score the same fixed response $y ,$ with source tokens kept in the original input and masked with EOS in the reference. Each operation’s formula and values from both runs determine local maps $D _ { f }$ that express output changes as weighted sums of input changes. Applying their transposes in reverse yields m, the effective secant slopes from the indicated activations to the response logprobability F. At the inputs, multiplying these slopes by embedding displacements and summing within each token gives contributions whose total recovers the measured $\Delta F .$ $\Delta$ denotes original minus reference. (b, c) Attention and linear attention (Gated DeltaNet) are examples within this reverse pass. Gray arrows show forward operations, and purple arrows return these slopes to their inputs. Other operations appear in Appendix A.3. (d) Positive and negative scores identify support and opposition. Name annotations sum token scores, colors saturate at ±2.5 nats, and ellipses omit text. The target is the full stored response plus EOS.

Attribution methods can explain prediction differences relative to a reference (Sundararajan et al., 2017; Shrikumar et al., 2017). Relevance propagation and representation decomposition trace information through attention and recurrent states (Ali et al., 2022; Achtibat et al., 2024; Rezaei Jafari et al., 2024; Modarressi et al., 2023; Pitorro & Treviso, 2025). FlashTrace follows source importance through multi-token responses (Pan et al., 2026). For the fixed-response question above, the target is a finite log-probability difference. Computing token contributions requires relating the response-score difference to activation differences throughout the model.

Existing attribution methods often propagate local derivatives or relevance. For the counterfactual above, this leaves a gap. Masking creates a finite score change, and several operands can change together inside one operation. In attention, both the weights and values change. In Gated DeltaNet, values, keys, queries, and gates jointly affect memory updates and reads (Yang et al., 2025). The missing rule is one that expresses the observed output change in terms of these joint input changes before carrying it backward.

DELTATRACE traces the finite score difference $\Delta F$ back to its source tokens. It modifies back propagation because derivatives evaluated at the original execution generally cannot reconstruct the finite response-score change caused by masking. From each operation’s formula and the two exe cutions, it constructs local finite-change rules and composes them in one reverse pass into effective secant slopes from input embeddings to the response score. Just as a secant slope converts horizontal displacement into vertical displacement, multiplying each embedding component’s change by its effective slope gives its signed share of the score change. Summing these shares within a token gives its attribution; local conservation makes the total across all tokens equal the measured difference and preserves this interpretation when tokens are grouped into sentences or passages. Figure 1(a) shows the complete method, while (b) and (c) illustrate its backward calculations for two operations.

## Contributions.

• Signed evidence for the reasoning and answer together. DT combines a source token’s effects across response positions into one contribution to the full response’s score difference. The signed scores identify support and opposition and can be summed to explain sentences or passages (Section 2.2).

• One procedure for attention and linear attention. We derive backward calculations for attention products, normalization, and Gated DeltaNet as examples of the operations handled by DELTATRACE. These calculations account for simultaneous changes in their inputs, allowing the same attribution procedure to follow attention and recurrent memory (Sections 2.3–2.4).

• Auxiliary storage linear in sequence length. FlashAttention-style tiles (Dao et al., 2022) and Flash Linear Attention (FLA) chunks (Yang & Zhang, 2024) bound the workspace used for intermediate products. At fixed model dimensions and chunk size, longer inputs and responses require only linear growth in auxiliary storage (Section 3).

On Qwen3-8B retrieval and reasoning tasks, DELTATRACE achieves the lowest deletion-faithfulness scores on most tasks and the highest evidence recovery across the six needle-in-a-haystack tasks (Section 4). On Qwen3.5-9B, DELTATRACE lowers RISE on 9 of 13 tasks and MAS on 12 relative to FlashTrace.

## 2 DELTATRACE

## 2.1 THE COUNTERFACTUAL QUESTION AND ATTRIBUTION TARGET

We explain how masking selected prompt tokens changes the score of a fixed target span. The inputs are a trained language model with fixed parameters $\theta ,$ a prompt x containing the context and question, and a stored response $y = ( y _ { 1 } , \dots , y _ { L } )$ of $L$ tokens. Let $\tau$ be the positions of the target span within $y .$ Both executions score these positions using the same preceding response tokens. The scalar target score is

$$
F ( x ; y ) = \sum _ { t \in \mathcal { T } } \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) ,\tag{1}
$$

where $p _ { \theta }$ is the model’s next-token probability and $y _ { < t }$ is the response prefix before position t. A higher score means the model assigns greater probability to the target span given its preceding tokens. For full-response attribution, $y$ includes the reasoning, answer, and end-of-sequence (EOS) token, and $\mathcal { T } = \{ 1 , \ldots , L \}$ . The response tokens and target positions stay fixed, while their hidden representations can change as they read the different prompts.

The source positions are the context or question tokens chosen for explanation. Let $x _ { 1 }$ be the original prompt. Its reference $x _ { 0 }$ masks all source tokens together with EOS at the same positions (Figure 1(a)), keeping all other prompt tokens fixed. Both executions process the full input sequence and score the same target span, giving

$$
\begin{array} { r } { \Delta F = F ( x _ { 1 } ; y ) - F ( x _ { 0 } ; y ) . } \end{array}
$$

A positive $\Delta F$ means masking these tokens lowers the target score. The attribution task is to divide this difference among the source tokens. Each token receives a signed contribution $A _ { i } ,$ and adding the contributions recovers $\Delta F .$ . Positive and negative scores indicate support and opposition under this reference (Figure 1(d)).

Subscripts 0 and 1 denote reference and original activations, and $\Delta U = U _ { 1 } - U _ { 0 }$ denotes their difference. For source token $\boldsymbol { i } , \boldsymbol { e } _ { i , 0 }$ is the EOS embedding and $e _ { i , 1 }$ is its original embedding, so $\Delta e _ { i } = e _ { i , 1 } - e _ { i , 0 }$ . Tokens held fixed have zero embedding difference. The reverse pass converts these embedding differences into token contributions.

## 2.2 FROM THE SCORE DIFFERENCE TO TOKEN CONTRIBUTIONS

DELTATRACE replaces tangent slopes with secant slopes to trace the measured output change back to the inputs. For a scalar activation, the secant connects its reference and original points: its slope converts the horizontal input displacement into the vertical output displacement. $\bar { \mathsf { A } }$ tangent slope gives only a first-order approximation to this finite change. Equal inputs use the local derivative.

The reverse pass carries m for each activation entry, its effective secant slope to the final response score under the chosen allocation rules. Multiplying this slope by the entry’s displacement gives its assigned contribution. Subscripts identify the associated activation. The pass starts with $m _ { F } = 1$ since the full score difference is to be allocated. Table 1 contrasts tangent and secant propagation through the same activation.

Table 1: The same activation, two backward calculations. The activation has input a and output $h = f ( a )$ . Here d denotes an infinitesimal change and $\Delta$ the observed difference between the two executions.
<table><tr><td></td><td>Ordinary backpropagation Tangent</td><td colspan="2">DELTATRACE Secant</td></tr><tr><td>Local slope</td><td> $f ^ { \prime } ( a _ { 1 } )$ </td><td></td><td> $\Delta h _ { \mathrm { ~ - ~ } } h _ { 1 } - h _ { 0 }$   ${ \Delta a } ^ { - } \ a _ { 1 } - a _ { 0 }$ </td></tr><tr><td>Output change</td><td> $d h = f ^ { \prime } ( a _ { 1 } ) d a$ </td><td></td><td> $\Delta h = { \frac { \Delta h } { \Delta a } } \Delta a$ </td></tr><tr><td>Backward calculation</td><td> $\frac { \partial F } { \partial a } = f ^ { \prime } ( a _ { 1 } ) \frac { \partial F } { \partial h }$ </td><td></td><td> $m _ { a } = \frac { \Delta h } { \Delta a } m _ { h }$ </td></tr></table>

For array operations, we use the operation’s formula and both executions to express output changes as weighted sums of input changes. These local slopes form the matrix $D _ { f }$ . Linear operations use their existing weights. For a product, splitting the term involving both input changes equally between them gives equal weight to the two possible change orders. This produces the mean attention weights and mean values in Section 2.3.

The same local map relates observed differences and propagates m backward through its transpose,

$$
\Delta h = D _ { f } \Delta a , m _ { a } = D _ { f } ^ { \top } m _ { h } .\tag{2}
$$

The transpose multiplies each output’s effective slope by the corresponding local slopes and sums the products for each input. This expresses the same contribution in terms of input changes. The local slopes replace derivatives, and the propagated slopes replace gradients. The two products have the forms of a Jacobian–vector product (JVP) and a vector–Jacobian product (VJP), respectively.<sup>1</sup>

At the input embeddings, the propagated $m _ { e _ { i } }$ give the effective secant slopes for each token. Summing their products with the corresponding embedding displacements yields the token contributions,

$$
A _ { i } = \langle m _ { e _ { i } } , \Delta e _ { i } \rangle , \qquad \sum _ { i } A _ { i } = \Delta F .\tag{3}
$$

Every backward calculation preserves the contribution being divided, so the token scores add to the measured response-score difference. Positive and negative scores identify support and opposition across the chosen response span. They use natural-log units (nats) and can be summed into name,

sentence, or passage contributions. A fixed token’s embedding does not change, so its contribution is zero. Appendix A.1 gives the full derivation.

We next apply these rules to attention and Gated DeltaNet (Figure $1 ( \boldsymbol { \mathbf { b } } , \boldsymbol { \mathbf { c } } ) )$ . Appendix A.3 gives the formulas for linear layers, additions, activation functions, and normalizations in the same reverse pass.

## 2.3 ATTENTION EXAMPLE

For attention, $P _ { i j }$ is the weight with which position i reads source j, and the rows of $V$ contain source value vectors. The output $Y = P V$ is their weighted sum, with $P$ obtained by softmax normalization of query–key similarity scores.

We use the two-run means so that the weighted input changes sum exactly to the observed output change. Let $\Delta P , \Delta V ,$ , and $\Delta Y$ denote the differences in weights, values, and output between the two executions. Splitting the interaction term $\Delta P \Delta V$ equally between the inputs gives equal weight to the two possible change orders:

$$
\begin{array} { r l } & { \Delta Y = P _ { 0 } \Delta V + \Delta P V _ { 0 } + \underbrace { \Delta P \Delta V } _ { \mathrm { s p l i t \ : e q u a l l y } } } \\ & { \qquad = \underbrace { \left( P _ { 0 } + \frac { 1 } { 2 } \Delta P \right) } _ { \bar { P } } \Delta V + \Delta P \underbrace { \left( V _ { 0 } + \frac { 1 } { 2 } \Delta V \right) } _ { \bar { V } } . } \end{array}\tag{4}
$$

Here $\bar { P }$ and $\bar { V }$ are the means of the original and reference weights and values. They provide the two sets of local slopes in $D _ { f }$ (Equation 2): the mean weights multiply the value changes, and the mean values multiply the weight changes.

The incoming slopes m weight the two terms in the second line of Equation 4, dividing the output’s assigned contribution between value and weight changes. The backward calculation uses the same two products as ordinary backpropagation, with means replacing the original activations,

$$
\begin{array} { r l r } { \mathrm { B a c k p r o p a g a t i o n } } & { { } \displaystyle { \frac { \partial F } { \partial V } = P _ { 1 } ^ { \top } \frac { \partial F } { \partial Y } , \quad } } & { \displaystyle { \frac { \partial F } { \partial P } = \frac { \partial F } { \partial Y } V _ { 1 } ^ { \top } , } } \\ { \mathrm { D e l t a T r a c e } } & { { } \displaystyle m _ { V } = \bar { P } ^ { \top } m _ { Y } , \quad } & { \displaystyle m _ { P } = m _ { Y } \bar { V } ^ { \top } . } \end{array}\tag{5}
$$

The returned slopes continue through preceding operations to the input embeddings.

Softmax makes each attention row sum to one, so increasing one source’s weight reduces the total assigned to the others. The backward calculation for softmax carries this competition through a weighted row average subtracted from the incoming slopes. Its weights depend on the probabilities in both executions. The slopes then pass through the query–key scores to their input representations. The same construction handles the vocabulary normalization used to score each response token. Appendix A.2 gives the formula and its derivation.

## 2.4 GATED DELTANET EXAMPLE

Gated DeltaNet implements linear attention through recurrent memory that stores key–value associations across token positions (Yang et al., 2025). Figure 1(c) shows one step from the previous memory $S _ { t - 1 }$ to the updated memory $S _ { t }$ and its readout $o _ { t }$ . A retention gate controls how much earlier memory is kept, and a write gate controls a correction toward the current value at its key. A query reads from $S _ { t }$ to produce $o _ { t } .$ . Normalization, an output gate, and a linear projection then produce the layer output. Appendix A.4 gives the state equations.

Information written by an earlier token can affect a later readout, so its contribution must pass backward through the intervening memory updates. The readout slopes $m _ { o _ { t } }$ pass backward to the memory and query. At $S _ { t } ,$ the contribution from this readout joins those from later memory states. The resulting $m _ { S _ { t } }$ passes through the update to $S _ { t - 1 }$ , the value, the key, and the retention and write gates (Figure 1(c)). The returned slopes continue through preceding operations to the input embeddings, where Equation 3 gives token scores.

Within each GDN layer, the original and reference activations define two decompositions of the same readout changes over the complete memory sequence. Both backward calculations receive the same readout slopes. Their returned slopes for each value, key, query, and gate are averaged with equal weight before propagation continues to earlier layers. This averaging stays within the model’s single reverse pass. The separate output-gating product also splits contributions between its two changing inputs using their two-run means. Appendix A.4 gives these calculations.

## 3 EFFICIENT COMPUTATION

Storing dense attention matrices requires space for all token pairs. DELTATRACE computes local transpose products in small blocks and accumulates their results directly into the input slopes. The attention and memory rules in Sections 2.3–2.4 support tiled and chunked implementations.

Attention in tiles. A tile is a small block of query and source positions. Following FlashAttention (Dao et al., 2022), DELTATRACE reconstructs the two executions’ attention probabilities for one tile, uses them to update query, key, and value slopes, and reuses the workspace for the next tile. The softmax rule needs a weighted average over the whole row. One sweep collects those row statistics. Subsequent sweeps use them when propagating through each tile. For N input-plus-response posi tions and head dimension d, the stored vectors and row statistics occupy $O ( N d )$ auxiliary space per head. Dense attention requires $O ( N ^ { 2 } d )$ arithmetic.

Memory in chunks. A chunk is a consecutive group of tokens. The memory state at a chunk boundary summarizes what earlier tokens passed forward. In reverse, the slopes for that state pass back to the preceding chunk. Within each chunk, matrix products propagate the slopes through its reads, writes, and gates. This organization uses Flash Linear Attention (FLA) (Yang & Zhang, 2024). At fixed state dimensions and chunk size, the stored token slopes and boundary states grow linearly with sequence length.

The complete attribution call includes two forward executions and one reverse pass. It recomputes intermediate activations one layer at a time during the reverse pass, so each layer’s workspace can be reused for the next. Appendix B details the storage accounting and accelerated operations. Section 4 measures the complete call’s time and memory.

## 4 EVALUATION

We evaluate evidence recovery, deletion faithfulness, and attribution cost.

Tasks and methods. We first evaluate Qwen3-8B on the 1,243 examples released with FlashTrace (Pan et al., 2026). These comprise ten RULER tasks with 100 examples each, HotpotQA with 48, MATH with 100, and MoreHopQA with 95.<sup>2</sup> Needle-in-a-haystack (NIAH) tasks retrieve information among distractors. Multi-query tasks (MQ-Q2/Q4/Q8) ask for 2, 4, or 8 keys, while multi-value tasks (MV-V2/V4/V8) ask for 2, 4, or 8 values of one key. Variable Tracking (VT) follows variable assignments. Here H is the number of binding hops and C the number of parallel chains. HotpotQA and MoreHopQA require combining facts, while MATH contains mathematical reasoning problems. The full stored reasoning and answer plus EOS is scored for faithfulness. We compare DELTATRACE (DT) with FlashTrace (FT), Perturbation, REAGENT, Context Length Probing (CLP), Information Flow Routes (IFR), and Attention-aware Layer-wise Relevance Propagation (AttnLRP). Table 7 in Appendix C summarizes their principles and DT’s advantages. DT and FT use the same evaluation examples. VT/HotpotQA recovery compares all seven methods.

Evidence recovery. Recall is the fraction of annotated evidence recovered among the highestranked sources. NIAH evidence consists of the inserted key–value sentences, and VT evidence consists of the assignment statements along the target variable chain (Pan et al., 2026). A token budget limits how much input the ranking may select. For NIAH, DT attributes the full response plus EOS. We select the top 10% of eligible input tokens ranked by positive-part attribution scores. Eligible tokens are the source positions marked for evaluation in the released examples. VT measures token Recall in the passage body, with budgets of 10%, 10%, 20%, and 30% for H2, H4, H6, and H10, and attributes the answer tokens. HotpotQA reports Full-support Recall@30%. An example scores one if every official supporting fact is selected, and zero otherwise. It attributes the full response, ranks passage sentences by the sum of their positive token contributions, and selects the longest ranking prefix within 30% of all body tokens. FT uses three recursive tracing steps (K3) for recovery. Table 4 reports these values.

Table 2: RISE ↓ on the complete released Qwen3-8B tasks. FT is FlashTrace (Pan et al., 2026) and DT is DELTATRACE. DT uses signed ranking. FT uses the same evaluation examples as DT. n is the released task size. Bold marks the lowest observed mean across all seven methods.
<table><tr><td>Task</td><td>n</td><td>Perturbation</td><td>REAGENT</td><td>CLP</td><td>IFR</td><td>AttnLRP</td><td>FT</td><td>DT</td></tr><tr><td>MQ-Q2</td><td>100</td><td>0.0945</td><td>0.1174</td><td>0.0977</td><td>0.0747</td><td>0.1955</td><td>0.0682</td><td>0.0723</td></tr><tr><td>MQ-Q4</td><td>100</td><td>0.2389</td><td>0.2599</td><td>0.2534</td><td>0.1147</td><td>0.2628</td><td>0.1134</td><td>0.1448</td></tr><tr><td>MQ-Q8</td><td>100</td><td>0.4986</td><td>0.4865</td><td>0.5096</td><td>0.3709</td><td>0.3766</td><td>0.3519</td><td>0.4286</td></tr><tr><td>MV-V2</td><td>100</td><td>0.1343</td><td>0.1876</td><td>0.1559</td><td>0.0691</td><td>0.1403</td><td>0.0686</td><td>0.0597</td></tr><tr><td>MV-V4</td><td>100</td><td>0.1862</td><td>0.2107</td><td>0.2166</td><td>0.0726</td><td>0.1934</td><td>0.0704</td><td>0.0587</td></tr><tr><td>MV-V8</td><td>100</td><td>0.3514</td><td>0.3685</td><td>0.3929</td><td>0.2051</td><td>0.2852</td><td>0.1828</td><td>0.1159</td></tr><tr><td>VT-H2-C3</td><td>100</td><td>0.3844</td><td>0.4378</td><td>0.3742</td><td>0.1609</td><td>0.3191</td><td>0.1317</td><td>0.1003</td></tr><tr><td>VT-H4-C1</td><td>100</td><td>0.3538</td><td>0.3971</td><td>0.3277</td><td>0.1018</td><td>0.3239</td><td>0.1102</td><td>0.1013</td></tr><tr><td>VT-H6-C1</td><td>100</td><td>0.4584</td><td>0.4860</td><td>0.4225</td><td>0.1245</td><td>0.3378</td><td>0.1224</td><td>0.1109</td></tr><tr><td>VT-H10-C1</td><td>100</td><td>0.4663</td><td>0.4953</td><td>0.4510</td><td>0.1526</td><td>0.3573</td><td>0.1427</td><td>0.1403</td></tr><tr><td>HotpotQA</td><td>48</td><td>0.1333</td><td>0.1445</td><td>0.1013</td><td>0.0745</td><td>0.1548</td><td>0.0331</td><td>0.0275</td></tr><tr><td>MATH</td><td>100</td><td>0.3802</td><td>0.3878</td><td>0.3616</td><td>0.3537</td><td>0.3684</td><td>0.3487</td><td>0.2321</td></tr><tr><td>MoreHopQA</td><td>95</td><td>0.2489</td><td>0.2652</td><td>0.2487</td><td>0.1455</td><td>0.2857</td><td>0.1281</td><td>0.0703</td></tr></table>

Table 3: MAS ↓ on the same released tasks and seven methods as Table 2. DT uses its positive source contributions. Bold marks the lowest observed mean.
<table><tr><td>Task</td><td>n</td><td>Perturbation</td><td>REAGENT</td><td>CLP</td><td>IFR</td><td>AttnLRP</td><td>FT</td><td>DT</td></tr><tr><td>MQ-Q2</td><td>100</td><td>0.1442</td><td>0.1966</td><td>0.1663</td><td>0.1397</td><td>0.3261</td><td>0.1628</td><td>0.1176</td></tr><tr><td>MQ-Q4</td><td>100</td><td>0.3270</td><td>0.3574</td><td>0.3203</td><td>0.1770</td><td>0.4509</td><td>0.1937</td><td>0.2011</td></tr><tr><td>MQ-Q8</td><td>100</td><td>0.7093</td><td>0.6942</td><td>0.6566</td><td>0.4600</td><td>0.6020</td><td>0.4271</td><td>0.6047</td></tr><tr><td>MV-V2</td><td>100</td><td>0.1870</td><td>0.2756</td><td>0.2159</td><td>0.1343</td><td>0.2290</td><td>0.1569</td><td>0.1046</td></tr><tr><td>MV-V4</td><td>100</td><td>0.2436</td><td>0.2905</td><td>0.2804</td><td>0.1416</td><td>0.3248</td><td>0.1605</td><td>0.1058</td></tr><tr><td>MV-V8</td><td>100</td><td>0.4580</td><td>0.4938</td><td>0.5114</td><td>0.2746</td><td>0.4752</td><td>0.2477</td><td>0.1717</td></tr><tr><td>VT-H2-C3</td><td>100</td><td>0.5513</td><td>0.6680</td><td>0.4900</td><td>0.2309</td><td>0.5214</td><td>0.1873</td><td>0.1486</td></tr><tr><td>VT-H4-C1</td><td>100</td><td>0.5168</td><td>0.6031</td><td>0.4199</td><td>0.1479</td><td>0.5475</td><td>0.1656</td><td>0.1406</td></tr><tr><td>VT-H6-C1</td><td>100</td><td>0.6837</td><td>0.7453</td><td>0.5648</td><td>0.1727</td><td>0.5723</td><td>0.1787</td><td>0.1537</td></tr><tr><td>VT-H10-C1</td><td>100</td><td>0.7013</td><td>0.7405</td><td>0.5967</td><td>0.2011</td><td>0.5915</td><td>0.1983</td><td>0.1842</td></tr><tr><td>HotpotQA</td><td>48</td><td>0.2203</td><td>0.2351</td><td>0.1895</td><td>0.1655</td><td>0.2493</td><td>0.1276</td><td>0.1002</td></tr><tr><td>MATH</td><td>100</td><td>0.5202</td><td>0.5358</td><td>0.4907</td><td>0.4897</td><td>0.5244</td><td>0.4461</td><td>0.3323</td></tr><tr><td>MoreHopQA</td><td>95</td><td>0.3470</td><td>0.3679</td><td>0.3478</td><td>0.2277</td><td>0.4579</td><td>0.2049</td><td>0.1546</td></tr></table>

Deletion faithfulness. The released deletion test masks high-ranked source tokens with EOS in 20 steps, rescoring the same full response after each step. The score is normalized between the full-input and fully-masked endpoints. A good ranking causes an early score drop because it masks evidence the model uses. The RISE deletion score summarizes the area under this curve. Magnitude Aligned Scoring (MAS) also compares the remaining response score with the remaining share of positive attribution (Pan et al., 2026). Lower values are better for both metrics. DT uses signed scores for RISE. For MAS, it sets negative scores to zero, ranking sources by their positive support for the response. Appendix C specifies the ranking and normalization.

Attribution quality. On Qwen3-8B, DT has the lowest RISE on 10 of 13 tasks and the lowest MAS on 11 (Tables 2 and 3). The largest absolute RISE reduction against FT is on MATH, from 0.3487 to 0.2321. DT achieves the highest Recall on all six NIAH tasks and all four VT tasks. At the stated VT budgets, its macro Recall is 87.75%, compared with 76.57% for FT K3. On HotpotQA, DT reaches 85.42% Full-support Recall, compared with 75.00% for FT K3 (Table 4).

Table 4: Evidence recovery (%, higher is better) at the listed token budgets. NIAH/VT report token Recall. HotpotQA reports Full-support Recall@30%, the fraction of examples for which every official supporting fact is selected. Sentence scores sum positive token contributions. All body tokens are included in the budget calculation. VT macro averages its four rows. FT uses K3 for recovery. Bold marks the highest observed mean.
<table><tr><td>Task</td><td>n</td><td></td><td>Budget Perturbation REAGENT</td><td></td><td>CLP</td><td></td><td>IFR AttnLRP</td><td>FT</td><td>DT</td></tr><tr><td>MQ-Q2</td><td>100</td><td>10%</td><td>39.05</td><td>24.35</td><td></td><td>39.85 47.10</td><td>21.49</td><td>46.32 67.24</td><td></td></tr><tr><td>MQ-Q4</td><td>100</td><td>10%</td><td>9.04</td><td>8.49</td><td></td><td>8.58 32.84</td><td>20.42</td><td>39.5847.16</td><td></td></tr><tr><td>MQ-Q8</td><td>100</td><td>10%</td><td>0.97</td><td>0.47</td><td>0.84</td><td>1.19</td><td>7.63</td><td></td><td>7.72 13.59</td></tr><tr><td>MV-V2</td><td>100</td><td>10%</td><td>25.48</td><td>18.02</td><td></td><td>20.67 57.50</td><td>25.44</td><td>53.4677.79</td><td></td></tr><tr><td>MV-V4</td><td>100</td><td>10%</td><td>16.05</td><td>15.60</td><td></td><td>14.6145.24</td><td>24.25</td><td>49.3756.79</td><td></td></tr><tr><td>MV-V8</td><td>100</td><td>10%</td><td>8.02</td><td>7.37</td><td>7.29</td><td>17.87</td><td>15.88</td><td>19.52</td><td>25.19</td></tr><tr><td>VT-H2-C3</td><td>100</td><td>10%</td><td>26.69</td><td>26.66</td><td></td><td>29.10 81.16</td><td>83.37</td><td>72.41 93.81</td><td></td></tr><tr><td>VT-H4-C1</td><td>100</td><td>10%</td><td>18.86</td><td></td><td></td><td>17.58 20.53 70.09</td><td></td><td>70.5468.3271.31</td><td></td></tr><tr><td>VT-H6-C1</td><td>100</td><td>20%</td><td>33.57</td><td></td><td></td><td>32.47 33.8583.74</td><td>84.57</td><td>82.1595.37</td><td></td></tr><tr><td>VT-H10-C1</td><td>100</td><td>30%</td><td>40.70</td><td>42.48</td><td></td><td>44.0482.66</td><td>77.33</td><td></td><td>83.41 90.51</td></tr><tr><td>VT macro</td><td>400</td><td>mixed</td><td>29.96</td><td>29.80</td><td>31.88</td><td>79.41</td><td>78.95</td><td></td><td>76.57 87.75</td></tr><tr><td>HotpotQA</td><td>48</td><td>30%</td><td>35.42</td><td></td><td></td><td>35.4264.5885.42</td><td></td><td>87.5075.0085.42</td><td></td></tr></table>

Table 5: Attribution quality on Qwen3.5-9B. All tasks use the full set of released examples. Recovery is token Recall for NIAH/VT and Full-support Recall@30% for HotpotQA, as defined in Table 4. FT denotes FlashTrace, with K3 for Recall and K1 for RISE/MAS. DT uses signed RISE and positive-part MAS. Bold marks the better mean. Dashes indicate undefined recovery.
<table><tr><td>Task</td><td rowspan="2">n</td><td rowspan="2">Budget</td><td colspan="2">Recall (%) ↑</td><td colspan="2">RISE↓</td><td colspan="2">MAS↓</td></tr><tr><td></td><td>FT</td><td>DT</td><td>FT</td><td>DT</td><td>FT</td><td>DT</td></tr><tr><td>MQ-Q2</td><td>100</td><td>10%</td><td>52.92</td><td>71.94</td><td>0.0982</td><td>0.0996</td><td>0.2549</td><td>0.1306</td></tr><tr><td>MQ-Q4</td><td>100</td><td>10%</td><td>38.56</td><td>54.18</td><td>0.1414</td><td>0.1579</td><td>0.2651</td><td>0.2242</td></tr><tr><td>MQ-Q8</td><td>100</td><td>10%</td><td>10.72</td><td>14.63</td><td>0.3704</td><td>0.4028</td><td>0.4390</td><td>0.5891</td></tr><tr><td>MV-V2</td><td>100</td><td>10%</td><td>60.51</td><td>83.53</td><td>0.0974</td><td>0.0804</td><td>0.2467</td><td>0.1080</td></tr><tr><td>MV-V4</td><td>100</td><td>10%</td><td>46.19</td><td>61.42</td><td>0.1284</td><td>0.1130</td><td>0.2538</td><td>0.1510</td></tr><tr><td>MV-V8</td><td>100</td><td>10%</td><td>22.27</td><td>26.36</td><td>0.2508</td><td>0.2206</td><td>0.3264</td><td>0.2992</td></tr><tr><td>MATH</td><td>100</td><td></td><td></td><td></td><td>0.3777</td><td>0.2638</td><td>0.4837</td><td>0.3368</td></tr><tr><td>MoreHopQA</td><td>95</td><td></td><td></td><td></td><td>0.2093</td><td>0.1429</td><td>0.2894</td><td>0.2001</td></tr><tr><td>VT H2-C3</td><td>100</td><td>10%</td><td>67.80</td><td>94.20</td><td>0.1514</td><td>0.1095</td><td>0.2224</td><td>0.1295</td></tr><tr><td>VT H4-C1</td><td>100</td><td>10%</td><td>69.07</td><td>71.89</td><td>0.1161</td><td>0.1013</td><td>0.1888</td><td>0.1343</td></tr><tr><td>VT H6-C1</td><td>100</td><td>20%</td><td>83.06</td><td>97.35</td><td>0.1300</td><td>0.1154</td><td>0.2051</td><td>0.1504</td></tr><tr><td>VT H10-C1</td><td>100</td><td>30%</td><td>81.90</td><td>94.23</td><td>0.1537</td><td>0.1561</td><td>0.2307</td><td>0.2040</td></tr><tr><td>HotpotQA</td><td>48</td><td>30%</td><td>58.33</td><td>95.83</td><td>0.2154</td><td>0.1470</td><td>0.3125</td><td>0.2367</td></tr></table>

Qwen3.5-9B results. We also evaluate DT and FT on the hybrid Qwen3.5-9B model using 13 tasks comprising 1,243 examples, including all 400 VT and 48 HotpotQA examples (Table 5). DT has lower RISE on 9 tasks and lower MAS on 12. Recall is higher on all six NIAH and all four VT tasks. On HotpotQA, DT reaches 95.83% Full-support Recall, compared with 58.33% for FT K3, and lowers both RISE and MAS.

Across both models, DT recovers more annotated MQ evidence within the same 10% token budget, with higher Recall for 2, 4, and 8 queried keys.

Attribution time and memory. The first-page timing panel compares DT and one-hop FT as response length increases on Qwen3-8B, using one MetaX C550. Crosses mark out-of-memory runs. In the 16-example Qwen3-8B comparison, DT has comparable attribution time to one-hop FT (11.40 versus 12.16 seconds, summing per-example medians) with lower peak allocated memory (18.76 versus 21.49 GB). Times cover complete attribution calls. Appendix D gives the timing protocol.

Reading the token scores. The signed scores distinguish support from opposition in the Qwen3.5- 9B playwright example (Figure 1(d)). Figures 2 and 3 show retrieval and multi-hop examples with each model’s own scores. Teal marks positive contributions and coral negative contributions to the full response score. The retrieval example links source keys to their numbers and the query naming them. The multi-hop example follows a voice actor to a film and its director, whose surname determines the answer. Each deletion curve compares DT with one-hop FT on the same model and response.

## 5 RELATED WORK

Reference-based attribution. Integrated Gradients integrates input sensitivities between a reference and the input, while DeepLIFT propagates activation differences into source contributions (Sundararajan et al., 2017; Shrikumar et al., 2017). Conservative Propagation and AttnLRP develop relevance rules for transformers (Ali et al., 2022; Achtibat et al., 2024). DT constructs local slopes from each operation’s formula and two executions, then composes them in a single reverse pass to attribute the response’s log-probability difference to input tokens.

Tracing source information. ALTI aggregates token interactions across layers, and DecompX propagates decomposed representations to the prediction head (Ferrando et al., 2022; Modarressi et al., 2023). Information Flow Routes traces influential computations, while FlashTrace aggregates response spans and recursively follows their source importance (Ferrando & Voita, 2024; Pan et al., 2026). DT sums the target response tokens’ log-probabilities into one scalar score and attributes its difference from the reference to input tokens.

Recurrent models. MambaLRP conserves relevance through selective state-space computation, and LaTIM decomposes token interactions in Mamba models (Rezaei Jafari et al., 2024; Pitorro & Treviso, 2025). The Gated DeltaNet example applies DT’s backward calculation to retained states, write corrections, and query reads (Yang et al., 2025). The same reverse pass traverses attention and memory layers.

## 6 CONCLUSION

DT assigns signed token contributions to the change in a full response’s log-probability between reference and original inputs. Its reverse pass replaces local derivatives with finite-change local slopes and composes them into effective secant slopes. Their products with embedding displacements give token contributions that sum to the response-score difference. Tiled attention and chunked memory make the auxiliary storage linear in sequence length at fixed model dimensions and chunk size. On the evaluated attention and hybrid models, DT improves evidence recovery and deletion faithfulness over FlashTrace on most tasks. The same counterfactual view has an action-level analogue: under a fixed continuation policy, averaging the return difference over a reference action recovers the advantage $Q ^ { \pi } ( h , a ) - \dot { V } ^ { \pi } ( h )$ , linking finite counterfactual credit to the standard policy-gradient signal (Appendix F).

## AI USE STATEMENT

AI tools assisted with method development, derivations, coding, experiments, literature review, and manuscript preparation.

## ETHICS STATEMENT

The experiments use publicly released benchmark examples and do not involve human subjects or private personal data. The work presents a research tool for analyzing model behavior, and we have no additional conflicts of interest, sponsorship, or legal or ethical concerns to disclose.

## REPRODUCIBILITY STATEMENT

The paper and appendix specify the attribution procedure, finite propagation rules, attention and Gated DeltaNet cases, evaluation metrics, and measurement protocols. The supplementary code package provides the implementation, frozen result tables, figure-generation scripts, environment records, and reproduction instructions for the reported results.

## REFERENCES

Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain, Thomas Wiegand, Sebastian Lapuschkin, and Wojciech Samek. AttnLRP: Attention-aware layer-wise relevance propagation for transformers. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pp. 135–168. PMLR, 2024. URL https://proceedings.mlr.press/v235/achtibat24a.html.

Ameen Ali, Thomas Schnake, Oliver Eberle, Gregoire Montavon, Klaus-Robert M´ uller, and Lior¨ Wolf. XAI for transformers: Better explanations through conservative propagation. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162, pp. 435–451. PMLR, 2022. URL https://proceedings.mlr.press/v162/ali22a.html.

Ondˇrej C´ıfka and Antoine Liutkus. Black-box language model explanation by context length probing. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pp. 1067–1079. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-short.92. URL https://aclanthology.org/2023.acl-short.92/.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: Fast and´ memory-efficient exact attention with IO-awareness. arXiv preprint arXiv:2205.14135, 2022. URL https://arxiv.org/abs/2205.14135.

Javier Ferrando and Elena Voita. Information flow routes: Automatically interpreting language models at scale. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pp. 17432–17445. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.965. URL https://aclanthology.org/2024.emnlp-main.965/.

Javier Ferrando, Gerard I. Gallego, and Marta R. Costa-juss´ a. Measuring the mixing of contextual\` information in the transformer. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pp. 8698–8714. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.595. URL https://aclanthology.org/2022.emnlp-main.595/.

Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32, pp. 2974–2982, 2018. doi: 10.1609/aaai.v32i1.11794. URL https://ojs.aaai.org/index.php/AAAI/article/view/11794.

Ali Modarressi, Mohsen Fayyaz, Ehsan Aghazadeh, Yadollah Yaghoobzadeh, and Mohammad Taher Pilehvar. DecompX: Explaining transformers decisions by propagating token decomposition. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 2649–2664. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.149. URL https://aclanthology.org/2023.acl-long.149/.

Wenbo Pan, Zhichao Liu, Xianlong Wang, Haining Yu, and Xiaohua Jia. Towards long-horizon interpretability: Efficient and faithful multi-token attribution for reasoning LLMs. arXiv preprint arXiv:2602.01914, 2026. URL https://arxiv.org/abs/2602.01914.

Hugo Pitorro and Marcos Vinicius Treviso. LaTIM: Measuring latent token-to-token interactions in Mamba models. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor

Computational Linguistics (Volume 1: Long Papers), pp. 24478–24493. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.acl-long.1194. URL https://aclanthology.org/2025.acl-long.1194/.

Farnoush Rezaei Jafari, Gregoire Montavon, Klaus-Robert M´ uller, and Oliver Eberle. MambaLRP:¨ Explaining selective state space sequence models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://papers.neurips.cc/paper\_files/paper/2024/hash/ d6d0e41e0b1ed38c76d13c9e417a8f1f-Abstract-Conference.html.

Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70, pp. 3145–3153. PMLR, 2017. URL https://proceedings.mlr.press/v70/shrikumar17a.html.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 3319–3328. PMLR, 2017. URL https://proceedings.mlr.press/v70/sundararajan17a.html.

Richard S. Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, volume 12, pp. 1057–1063, 1999. URL https://proceedings.neurips.cc/paper/1999/hash/ 464d828b85b0bed98e80ade0a5c43b0f-Abstract.html.

Songlin Yang and Yu Zhang. FLA: A triton-based library for hardware-efficient implementations of linear attention mechanism. Software repository, January 2024. URL https://github.com/fla-org/flash-linear-attention.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2412.06464.

Zhixue Zhao and Boxuan Shan. ReAGent: A model-agnostic feature attribution method for generative language models. arXiv preprint arXiv:2402.00794, 2024. URL https://arxiv.org/abs/2402.00794.

## A FORMAL RULES AND DERIVATIONS

## A.1 FINITE PROPAGATION AND CONTRIBUTION CONSERVATION

For an operation $v = f ( u _ { 1 } , \ldots , u _ { k } )$ , DELTATRACE constructs linear maps $D _ { j }$ from the two executions such that

$$
\Delta v = \sum _ { j = 1 } ^ { k } D _ { j } \Delta u _ { j } .\tag{6}
$$

The incoming slopes $m _ { v }$ propagate to input $j$ as $D _ { j } ^ { \top } m _ { v } .$ . Slopes from multiple consumers add at their shared input. Starting with slope 1 at the scalar target, the reverse pass produces the source slopes in Equation 3. For a scalar nonlinearity $f ,$ the local slope is the divided difference $( f ( u _ { 1 } ) -$ $f ( u _ { 0 } ) ) / ( u _ { 1 } - u _ { 0 } )$ , with its derivative as the coincident-input limit. Linear transformations retain their weights, and addition sends the slope to each summand.

Theorem 1 (Contribution conservation). For an acyclic computation over real-valued operations satisfying Equation $\delta ,$ reverse propagation from a scalar target yields the conservation identity in Equation 3.

Proof. An acyclic computation can be expanded into elementary operations with explicit branches. At an operation $v = f ( u _ { 1 } , \ldots , u _ { k } )$ , Equation 6 gives

$$
\langle m _ { v } , \Delta v \rangle = \sum _ { j } \left. D _ { j } ^ { \top } m _ { v } , \Delta u _ { j } \right. .\tag{7}
$$

Replacing a node’s contribution by the contributions of its parents therefore preserves the total on a cut through the graph. At shared parents, slopes from different consumers sum. Repeated substitution reaches the input cut and yields Equation 3, since the slope of the scalar target is 1. Constant inputs have zero difference and contribute zero to this sum.

## A.2 FINITE ATTENTION NORMALIZATION

For an attention row $p _ { b } = \mathrm { s o f t m a x } ( z _ { b } ) , b \in \{ 0 , 1 \}$ , the finite rule first converts changes in logprobabilities into changes in probabilities. The slope of this conversion is the logarithmic mean $\ell _ { i }$ We define it and its sum as

$$
\ell _ { i } = \frac { p _ { 1 , i } - p _ { 0 , i } } { \log p _ { 1 , i } - \log p _ { 0 , i } } , \qquad \tau = \sum _ { i } \ell _ { i } ,\tag{8}
$$

with $\ell _ { i } = p _ { 0 } ,$ <sub>i</sub> at coincident probabilities. Dividing by τ gives weights that sum to one for the row averages below. All quantities are evaluated on the common unmasked support.

Proposition 1 (Finite attention normalization). For finite logits on a common support,

$$
\Delta p = \left( \mathrm { d i a g } ( \ell ) - \frac { \ell \ell ^ { \top } } { \tau } \right) \Delta z .\tag{9}
$$

Proof. The common support contains positive probabilities. For endpoint $b ,$ log $p _ { b } = z _ { b } - a _ { b } \mathbf { 1 }$ where $\begin{array} { r } { a _ { b } = \log \sum _ { i } \exp ( z _ { b , i } ) } \end{array}$ . The definition of the logarithmic mean gives

$$
\Delta p = \ell \odot ( \Delta z - \Delta a \mathbf { 1 } ) .\tag{10}
$$

Probability normalization implies $\mathbf { 1 } ^ { \top } \Delta p = 0$ , so

$$
\Delta a = \frac { \ell ^ { \top } \Delta z } { \tau } .\tag{11}
$$

Substitution proves Equation 9. For a target coordinate y,

$$
\Delta \log p _ { y } = \left( e _ { y } - \frac { \ell } { \tau } \right) ^ { \top } \Delta z .\tag{12}
$$

The coincident-probability definition follows from the continuous limit of the logarithmic mean.

The operator is symmetric. For incoming slopes $m _ { p }$ at probabilities $p ,$ the slope at query–key score $z _ { i }$ is

$$
m _ { z , i } = \ell _ { i } ( m _ { p , i } - \bar { m } _ { p } ) , \qquad \bar { m } _ { p } = \frac { \sum _ { j } \ell _ { j } m _ { p , j } } { \sum _ { j } \ell _ { j } } .\tag{13}
$$

The slopes for a log-softmax target are $m _ { z } = e _ { y } - \ell / \tau$ , where $e _ { y }$ is the one-hot vector selecting the target vocabulary coordinate.

## A.3 FINITE OPERATOR RULES

For the attention example with output $Y = P V$ , Equation 4 gives the slopes $m _ { V } = \bar { P } ^ { \top } m _ { Y }$ and $m _ { P } = m _ { Y } \bar { V } ^ { \top }$ for an incoming slope matrix $m _ { Y }$

Query–key scores use the symmetric product identity,

$$
\Delta ( Q K ^ { \top } ) = \Delta Q \bar { K } ^ { \top } + \bar { Q } \Delta K ^ { \top } , \qquad \bar { Q } = ( Q _ { 0 } + Q _ { 1 } ) / 2 ,\tag{14}
$$

with $\bar { K }$ defined in the same way. Table 6 lists the finite decompositions by operator. Normalization, rotary position transformations, and grouped heads follow their respective model definitions.

Table 6: Finite propagation rules by operator. Bars denote endpoint averages. The gated-memory recurrence is given in Appendix A.4.  
Operation Change decomposition   
Linear and residual $\Delta ( W x + b ) = W \Delta x ; \Delta ( a + b ) = \Delta a + \Delta b$   
Attention values $\Delta ( P V ) = \bar { P } \Delta V + \Delta P \vec { V }$   
Attention scores $\Delta ( Q K ^ { \top } ) = \Delta Q \bar { K } ^ { \top } + \bar { Q } \Delta K ^ { \top }$   
SwiGLU $\Delta ( u s ) = \bar { s } \odot \Delta u + \bar { u } \odot \Delta s , s = \mathrm { S i L U } ( g )$   
GDN output gating $\Delta ( n \odot s ) = \bar { s } \odot \Delta n + \bar { n } \odot \Delta s$   
RMS normalization Analytic finite rule for wx $/ { \sqrt { \mathrm { m e a n } ( x ^ { 2 } ) + \epsilon } }$

## A.4 BACKWARD CALCULATION FOR THE GATED DELTANET EXAMPLE

With column vectors and a key-by-value state $S _ { t } \in \mathbb R ^ { d _ { k } \times d _ { v } }$ , its per-token computation is

$$
\begin{array} { l } { T _ { t } = \alpha _ { t } S _ { t - 1 } , } \\ { c _ { t } = v _ { t } - r _ { t } , } \\ { S _ { t } = T _ { t } + k _ { t } u _ { t } ^ { \top } , } \end{array}
$$

$$
\begin{array} { r l } & { \boldsymbol { r } _ { t } = \boldsymbol { T } _ { t } ^ { \top } \boldsymbol { k } _ { t } , } \\ & { \boldsymbol { u } _ { t } = \beta _ { t } \boldsymbol { c } _ { t } , } \\ & { \boldsymbol { o } _ { t } = \boldsymbol { S } _ { t } ^ { \top } \boldsymbol { q } _ { t } . } \end{array}\tag{15}
$$

The query includes the model’s fixed attention scale. $T _ { t }$ is the retained state, $r _ { t }$ is its readout at key $k _ { t } .$ , and $u _ { t }$ is the gated difference between the incoming value and that readout. One endpoint ordering constructs a finite map $D _ { \mathrm { G D N } } ^ { 0 1 }$ for the complete memory recurrence through

$$
\begin{array} { r l } & { \Delta T _ { t } = \alpha _ { t , 1 } \Delta S _ { t - 1 } + S _ { t - 1 , 0 } \Delta \alpha _ { t } , } \\ & { \Delta r _ { t } = ( \Delta T _ { t } ) ^ { \top } k _ { t , 1 } + T _ { t , 0 } ^ { \top } \Delta k _ { t } , } \\ & { \Delta u _ { t } = \beta _ { t , 1 } ( \Delta v _ { t } - \Delta r _ { t } ) + c _ { t , 0 } \Delta \beta _ { t } , } \\ & { \Delta S _ { t } = \Delta T _ { t } + k _ { t , 1 } \Delta u _ { t } ^ { \top } + \Delta k _ { t } u _ { t , 0 } ^ { \top } , } \\ & { \Delta o _ { t } = ( \Delta S _ { t } ) ^ { \top } q _ { t , 1 } + S _ { t , 0 } ^ { \top } \Delta q _ { t } . } \end{array}\tag{16}
$$

Swapping the reference and original activations used as local slopes gives $D _ { \mathrm { G D N } } ^ { 1 0 }$ . Within each GDN layer, we average these memory-recurrence maps and propagate the readout slopes as

$$
\bar { D } _ { \mathrm { G D N } } = \frac { D _ { \mathrm { G D N } } ^ { 0 1 } + D _ { \mathrm { G D N } } ^ { 1 0 } } { 2 } , \qquad m _ { \mathrm { i n } } = \bar { D } _ { \mathrm { G D N } } ^ { \top } m _ { o } .\tag{17}
$$

Here $m _ { o }$ collects the slopes of all readouts $o _ { t }$ in the layer, and $m _ { \mathrm { i n } }$ collects those returned to the recurrence inputs. Equivalently, apply each transpose map to the same $m _ { o }$ and average the returned slopes. Both maps express the same finite output difference, so their mean preserves it. The recurrence returns slopes for values, keys, queries, and both gates, and propagates state slopes to earlier token positions. Scalar gate nonlinearities and query/key normalization use the surrounding network’s finite rules. Output gating uses the symmetric product in Table $^ { 6 , }$ with $n _ { t }$ the normalized readout and $s _ { t }$ its activated output-gate vector.

## A.5 PRODUCT AND NORMALIZATION IDENTITIES

For a bilinear map $B ,$ both

$$
\Delta B ( a , b ) = B ( \Delta a , \bar { b } ) + B ( \bar { a } , \Delta b ) ,\tag{18}
$$

$$
\Delta B ( a , b ) = B ( \Delta a , b _ { 1 } ) + B ( a _ { 0 } , \Delta b )\tag{19}
$$

follow by expansion. Equation 4 and the SwiGLU rule use the first, symmetric identity. The second identity, with the displayed argument ordering, gives the products in Equation 16. Its swapped ordering gives the other layer map averaged in Equation 17. Subtraction and state addition are linear.

For RMS normalization $f ( x ) = w \odot x / r ( x ) , r ( x ) = \sqrt { \| x \| ^ { 2 } / d + \epsilon }$ , the symmetric product rule gives an analytic local map. With $\bar { x } = ( x _ { 0 } + x _ { 1 } ) / 2 , r _ { b } = r ( x _ { b } )$ , and $v = w \odot m$ , its transpose action is

$$
D _ { f } ^ { \top } m = \frac { r _ { 0 } ^ { - 1 } + r _ { 1 } ^ { - 1 } } { 2 } v - \frac { 2 \bar { x } \langle \bar { x } , v \rangle } { d r _ { 0 } r _ { 1 } ( r _ { 0 } + r _ { 1 } ) } .\tag{20}
$$

The identity follows from $\Delta r = ( 2 \left. \bar { x } , \Delta x \right. / d ) / ( r _ { 0 } + r _ { 1 } )$ and $\Delta ( r ^ { - 1 } ) = - \Delta r / ( r _ { 0 } r _ { 1 } )$ . The same construction covers fixed learned scale parameters and the query/key normalization used in the mem ory path.

## B IMPLEMENTATION AND STORAGE ACCOUNTING

Attention. Native endpoint queries, keys, values, and log-normalizers determine the probabilities within each FlashAttention tile. The first sweep accumulates the logarithmic-mean normalizers and weighted row averages in Equation 13. Subsequent sweeps accumulate query, key, and value slopes. Endpoint averages reside in linear-size buffers or are computed inside the tiles.

Memory. The implementation processes 64-token chunks using FLA’s native chunk-state adjoints and batched contractions with saved endpoint states. An associative affine scan computes the finite decay contribution. Convolutions and linear projections propagate slopes through their transpose operations.

Storage. For sequence length $T ,$ , head dimension $d ,$ and tile or chunk width $C ,$ , attention vectors and row statistics occupy $O ( \breve { T } d )$ storage per head. Pairwise products reside in bounded tiles. Dense attention arithmetic is ${ \dot { O } } ( T ^ { 2 } d )$ . Chunked memory occupies $O ( T d + T C + ( T / C ) d ^ { 2 } )$ auxiliary storage per head for token slopes, within-chunk products, and boundary states. Both storage expressions are linear in $T$ at fixed d and $C .$ Layer replay supplies intermediate activations, with checkpoint storage accounted for separately.

## C EVALUATION SPECIFICATION

Both models use the released prompts, stored responses, eligible source positions, and evidence annotations. Table 5 specifies the Qwen3.5 examples for each task. An eligible position is an input position included in the task’s attribution and deletion domain. For NIAH, this domain follows the released token selection. VT and HotpotQA recovery select from the passage body.

Baseline comparisons. VT and HotpotQA recovery compare all seven methods on the same 448 inputs. The paired development comparison uses the same inputs and stored responses for DT and FT on each model, with one FT recursive step for faithfulness and three for recovery.

Table 7: Attribution methods.
<table><tr><td>Method</td><td>Principle</td><td>Strength</td><td>DT advantage</td></tr><tr><td>Perturbation (Pan et al., 2026)</td><td>Measure effects of separate masks</td><td>Direct intervention</td><td>Additive allocation of the joint effect</td></tr><tr><td>REAGENT (Zhao &amp; Shan, 2024)</td><td>Iteratively replace tokens and Black-box attribution update importance</td><td></td><td>Signed scores without iterative sampling</td></tr><tr><td>CLP (Cífka &amp; Liutkus, 2023)</td><td>Measure prediction changes as context expands</td><td>Reveals newly added information</td><td>Allocates the joint effect of selected inputs</td></tr><tr><td>IFR (Ferrando &amp; Voita, 2024)</td><td>Trace contributions to internal representations</td><td>paths</td><td>Identifies information Includes changes in attention weights</td></tr><tr><td>AttnLRP (Achtibat et al., 2024)</td><td>Propagate output relevance backward</td><td>Input and neuron attribution</td><td>Explains the measured finite score change</td></tr><tr><td>FlashTrace (Pan et al., 2026)</td><td>Aggregate spans and recursively trace sources</td><td>Efficient multi-token attribution</td><td>Signed, additive response-score contributions</td></tr></table>

Deletion scores. For token attribution scores A, DT ranks tokens by A for RISE and by max(A , 0) for MAS. The evaluator records 21 response scores across 20 deletion steps. Each score sums log-probabilities over the full fixed response plus EOS. Masking all eligible tokens with EOS gives the attribution reference. Full-input and fully-masked scores define the normalization endpoints. The normalized values are clipped to [0, 1] and made nonincreasing by a cumulative minimum. For MAS, the remaining attribution share is the positive-score sum over unmasked tokens divided by the initial positive-score sum over eligible tokens. We add its absolute difference from the normalized response score to that score, clip to [0, 1], and rescale the resulting curve to [0, 1] by its minimum and maximum. MAS is the trapezoidal area under this curve. The case-study curves use the positive ranking and share these endpoints between DT and FT for each model and example.

Recovery targets and budgets. For NIAH, DT attributes the full response plus EOS. We select the top 10% of eligible input tokens ranked by positive-part attribution scores. VT scores the answer tokens and ranks positive body-token scores. Its H2/H4/H6/H10 budgets are 10%/10%/20%/30%. HotpotQA scores the full response, ranks passage sentences by the sum of their positive token contributions, and selects the longest ranked sentence prefix within 30% of all body tokens. NIAH/VT report the fraction of annotated evidence tokens recovered. HotpotQA reports Full-support Recall@30%. Each example scores one if every official supporting fact is selected and zero otherwise. The reported value is the mean across examples.

Table 8: Paired development results on the first eight MQ-Q2 and eight MoreHopQA examples per model. This gives 32 model–example pairs. FT uses one recursive hop for RISE/MAS and three for recovery. RISE is signed, while MAS and recovery use positive scores.
<table><tr><td>Model</td><td>Task</td><td>Method</td><td>RISE↓</td><td>MAS↓</td><td>Recovery@10% ↑</td></tr><tr><td>Qwen3-8B</td><td>MQ-Q2</td><td>FlashTrace</td><td>0.0685</td><td>0.1547</td><td>56.45</td></tr><tr><td></td><td></td><td>DELTATRACE</td><td>0.0574</td><td>0.0806</td><td>77.52</td></tr><tr><td></td><td>MoreHopQA</td><td>FlashTrace</td><td>0.1111</td><td>0.1969</td><td></td></tr><tr><td></td><td></td><td>DELTATRACE</td><td>0.0585</td><td>0.1173</td><td></td></tr><tr><td>Qwen3.5-9B</td><td>MQ-Q2</td><td>FlashTrace</td><td>0.0942</td><td>0.2485</td><td>63.92</td></tr><tr><td></td><td></td><td>DELTATRACE</td><td>0.1115</td><td>0.1518</td><td>73.35</td></tr><tr><td></td><td>MoreHopQA</td><td>FlashTrace</td><td>0.2356</td><td>0.2797</td><td></td></tr><tr><td></td><td></td><td>DELTATRACE</td><td>0.1688</td><td>0.2426</td><td></td></tr></table>

## D IMPLEMENTATION COST MEASUREMENTS

DT and one-hop FT use the same inputs and responses in each timing comparison. Time covers the complete attribution call, including returning the source scores. For DT, this includes both forward executions and the reverse pass. Peak allocated device memory includes model weights.

Paired Qwen3 cost. Each method has one warm call followed by three measured calls in rotated order on each of 16 development examples. Aggregate times sum the per-example medians.

Response-length comparison. The first-page timing panel measures Qwen3-8B on one MetaX C550. Its horizontal axis counts response tokens. Each point is the mean of three complete attribution calls after two warm calls, with a fresh process for each method and length. Error bars show the minimum and maximum measured times.

## E DETAILED SIGNED EVIDENCE VIEWS

![](images/7bb3d2b04f5b4f11f182d496f8812d9605d2aaed36bab2a0532313d84008bdb2.jpg)  
Figure 2: Signed evidence for retrieving two numbers. (a,b) The same niah mq q2 input on Qwen3-8B and Qwen3.5-9B. Teal and coral show positive and negative DT contributions in nats. Marks locate excerpts. (c) Each model’s DT (solid) and one-hop FT (dashed) curves show the normalized log-probability of the full fixed response plus EOS under 20-step deletion. Color identifies the model. The released normalization uses that model’s full-input and fully-masked scores, clipping and a cumulative minimum. DT deletion uses positive ranking. Answer summarizes the full stored response.

![](images/d34d310bad93b6e5c0a7a29a1f384163a4ed331310ba23139463468c7ee202c9.jpg)  
Figure 3: Signed evidence across a multi-hop question. (a,b) The MoreHopQA input links Townsend Coleman to Sing, Garth Jennings, and his surname’s consonants. (c) Normalized log probability of the full fixed response plus EOS, comparing each model’s DT with its own one-hop FT baseline. Model colors, method line styles, and per-model normalization follow Figure 2. Both heatmaps share the signed scale within this example.

## F COUNTERFACTUAL CREDIT AND POLICY LEARNING

Under a fixed continuation policy, the expected return difference between an action and a reference action drawn from that policy equals the action’s advantage. This identity expresses the actionindependent-baseline principle (Sutton et al., 1999) in counterfactual form. COMA uses the related construction to assign credit by averaging one agent’s action while fixing the others (Foerster et al., 2018).

Let h be a decision history and π a fixed continuation policy. Write $G ^ { \pi } ( h , a ; \xi )$ for the integrable return obtained by setting the current action to a and subsequently following π, with $\xi$ supplying environment and policy randomness. Define

$$
\begin{array} { r l r } { Q ^ { \pi } ( h , a ) = \mathbb { E } _ { \xi } [ G ^ { \pi } ( h , a ; \xi ) ] , } & { { } } & { V ^ { \pi } ( h ) = \mathbb { E } _ { a ^ { \prime } \sim \pi ( \cdot | h ) } [ Q ^ { \pi } ( h , a ^ { \prime } ) ] . } \end{array}\tag{21}
$$

The reference action $a ^ { \prime }$ is sampled independently of the selected action $^ { a , }$ conditional on $h .$ The two rollouts may share randomness, with each retaining its correct marginal distribution under the corresponding action intervention.

Proposition 2 (Exactly averaged counterfactual credit). Under these definitions, the policyaveraged counterfactual credit satisfies

$$
C ^ { \pi } ( h , a ) = \mathbb { E } _ { a ^ { \prime } , \xi } [ G ^ { \pi } ( h , a ; \xi ) - G ^ { \pi } ( h , a ^ { \prime } ; \xi ) ] = Q ^ { \pi } ( h , a ) - V ^ { \pi } ( h ) .\tag{22}
$$

For a differentiable policy π with positive probabilities over a fixed finite action set, the scorefunction learning signals obey

$$
\begin{array} { r } { { \mathbb E } _ { a \sim \pi _ { \theta } } [ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid h ) C ^ { \pi _ { \theta } } ( h , a ) ] = { \mathbb E } _ { a \sim \pi _ { \theta } } [ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid h ) Q ^ { \pi _ { \theta } } ( h , a ) ] . } \end{array}\tag{23}
$$

Proof. Linearity of expectation gives Equation 22. The baseline term in Equation 23 is

$$
V ^ { \pi _ { \theta } } ( h ) \sum _ { a } \pi _ { \theta } ( a \mid h ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid h ) = V ^ { \pi _ { \theta } } ( h ) \nabla _ { \theta } \sum _ { a } \pi _ { \theta } ( a \mid h ) = 0 .\tag{24}
$$

Subtracting this term proves the result. Credit and action value serve as weights in the score-function update. □

This equality provides a concrete target for extending finite attribution to learning. The averaged action credit of a reward-level construction should recover the advantage.