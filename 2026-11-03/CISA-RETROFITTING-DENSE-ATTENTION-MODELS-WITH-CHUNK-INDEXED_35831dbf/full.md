# CISA: RETROFITTING DENSE ATTENTION MODELS WITH CHUNK-INDEXED SPARSE ATTENTION

Mohammed Dabbah, Izhak Golan, Oren Tropp, Daniel Afrimi, Vladimir Anisimov, Tomer Asida, Nave Assaf, Ran El-Yaniv, Ido Galil, Yonatan Geifman, Omri Puny, Ido Shahaf, and Akhiad Bercovich

NVIDIA, {mdabbah, izikg, otropp, dafrimi, vanisimov, tasida, nassaf, relyaniv, igalil, ygeifman, opuny, ishahaf, abercovich}@nvidia.com

## ABSTRACT

As context grows, LLM inference becomes increasingly expensive because attention repeatedly reads a growing key-value (KV) history. We introduce Chunk-Indexed Sparse Attention (CISA), a method for retrofitting pretrained dense attention models without gradient updates. CISA selects fixed-size chunks of the KV history using compact summaries derived from the model’s existing keys, avoiding token-level indexing and a separate full-length index cache. For indexing, CISA uses a single query per KV group and shares the selected chunks across the query in that group. During prefill, we reuse chunk selections across adjacent queries to reduce indexing cost and improve KV reuse and GPU utilization. We call this technique QShare.

We evaluate CISA on the NVIDIA Nemotron 3 Nano and Ultra models, as well as on Inkling-Small. In one-layer BF16 benchmarks at batch size 32 using Ultra’s perdevice attention dimensions, CISA accelerates core attention by 11.7× for prefill queries near the end of a 1M-token prompt and 13.0× during decode. Including chunk scoring, selection, and all other measured GPU work, the corresponding speedups are 8.6× and 3.0×. On Ultra NVFP4, at k<sub>chunks</sub> = 4,096, CISA keeps aggregate scores within 1.5 percentage points of dense attention on MRCR, Long-Bench v2, and GraphWalks. RULER sweeps on Nano and Ultra BF16 up to one million tokens show scores within 2.05 percentage points of dense attention with 8–16× fewer KV reads.

When serving million-token prompts with 32 concurrent requests and 4K output tokens per request, CISA achieves 2.29× the end-to-end throughput of dense attention using Nemotron 3 Ultra NVFP4 on four NVIDIA B300 GPUs. Selecting fewer chunks increases throughput to 2.61× compared with dense attention; in separate GraphWalks evaluations, the smaller budget widens the aggregate score gap from 1.49 to 3.59 percentage points. These results expose a practical quality– speed tradeoff without requiring sparse training.

## 1 INTRODUCTION

Long-context models can process code repositories, extended conversations, and million-token documents without an external retrieval step. This capability is expensive to serve. During dense prefill, each query attends to an increasingly long prefix, while autoregressive decode repeatedly reads the accumulated key–value (KV) cache. IO-aware kernels reduce data movement within exact attention (Dao et al., 2022), but they do not remove the growth in attended history. At long context lengths, KV-cache data movement is a major bottleneck for latency and throughput.

Sparse attention reduces this cost by selecting a small query-dependent subset of history and applying ordinary attention only to that subset. The challenge is to turn fewer attended tokens into faster GPU execution despite the cost of indexing and scattered KV-cache reads. Token-level sparse attention, such as DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025), indexes and selects individual positions from the history. Although attention accesses fewer tokens, the indexer still scans the full history, which can limit the overall speedup. Serving also requires writing fine-grained scores and indices, issuing irregular K/V reads, and often performing an additional gather or materialization step before attention begins. The nominal reduction in dot products can therefore overstate the realizable speedup.

A second challenge is that several strong sparse architectures are trained natively or introduced through substantial continued training (Yuan et al., 2025; DeepSeek-AI, 2025; Lai et al., 2026). That path is valuable when training a new model, but it does not directly address the large installed base of available dense attention models.

We introduce Chunk-Indexed Sparse Attention (CISA), a sparse-attention design that enables efficient long-context inference through chunk-level indexing and direct KV-cache access (Figure 1). CISA makes three coordinated choices. First, it indexes and selects fixed-size chunks rather than individual tokens, recalling a fixed budget of $k _ { \mathrm { c h u n k s } }$ historical chunks at each deployment operating point. Second, we initialize the index-query projections from the model’s existing query heads using a DSA-inspired attention-distribution matching criterion, without any gradient-based training. Third, we share one selected chunk set across G adjacent prefill query positions. This performs indexing once per group and improves attention efficiency by reusing the selected K/V data across more query rows. We call this mechanism QShare. It is compatible with multi-token prediction (MTP).

The design is motivated by deployment topology as much as approximation quality. MiniMax Sparse Attention (MSA) selects blocks by scanning a separately projected token-level index-key stream shared across KV groups (Lai et al., 2026); under tensor parallelism, that stream must be replicated or communicated. CISA instead derives one representative per 16-token chunk from each KV group’s main keys and stores it with the corresponding KV shard. Index storage and scanning scale with chunks; selected page IDs feed exact attention over original Q/K/V without a gather buffer.

We evaluate NVIDIA Nemotron 3 Nano 30B-A3B and Ultra 550B-A55B, together with Inkling-Small. NVFP4 weights on Ultra and Inkling-Small are used for the deployment evaluations; BF16 Nano and Ultra runs provide model-scaling and context-scaling controls. We evaluate four complementary long-context benchmarks—MRCR, LongBench v2, GraphWalks, and RULER—as well as AA-LCR reasoning, SWE-bench Verified, Terminal-Bench, and GDPval (Vodrahalli et al., 2024; Hsieh et al., 2024; Bai et al., 2024; OpenAI, 2025; Artificial Analysis Team, 2025; Jimenez et al., 2024; Merrill et al., 2026; Patwardhan et al., 2025). For MRCR, LongBench v2, and GraphWalks on Ultra NVFP4, we apply $k _ { \mathrm { c h u n k s } } = 2 , 0 4 8$ and $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ uniformly and report both aggregate and prompt-length-stratified quality. We also derive an attention IO-traffic model and measure controlled, equal-work serving throughput at fixed operating points. We report theoretical component ceilings separately from measured application throughput.

This paper makes three contributions:

![](images/a1b92a469716d87982fd97c84eb49a63349f9fefe75b7ecefd7deb6da0acf8c3.jpg)  
Figure 1: CISA retrofit and online sparse execution. Within each KV group, QShare reuses one page plan across adjacent queries; attention uses the original Q/K/V.

• We introduce CISA, a method for retrofitting pretrained dense attention layers with fixedbudget chunk selection, without gradient-based training.

• We develop a tensor-parallel-optimized sparse-attention layout with one indexer per KV group, chunk summaries from the main keys, and QShare for efficient prefill.

• We characterize quality and generation length across two model families, chunk budgets, precisions, and context lengths up to one million tokens, linking an IO-traffic model to measured serving throughput.

## 2 METHOD: CHUNK-INDEXED SPARSE ATTENTION

## 2.1 CHUNK-INDEXED ATTENTION FOR A SINGLE QUERY HEAD

For one query head at position $t ,$ let $q _ { t }$ be the query and $\{ k _ { u } , v _ { u } \} _ { u = 1 } ^ { t }$ its causal keys and values, with head dimension d. Let $K _ { 1 : t }$ and $V _ { 1 : t }$ stack these keys and values as rows. Dense attention applies a softmax to query–key scores and computes a weighted sum of values:

$$
o _ { t } ^ { \mathrm { d e n s e } } = \mathrm { s o f t m a x } \bigg ( \frac { q _ { t } K _ { 1 : t } ^ { \top } } { \sqrt { d } } \bigg ) V _ { 1 : t } .\tag{1}
$$

CISA selects chunks of history, then computes attention using their original keys and values.

We partition the token sequence into chunks $\mathcal { C } _ { j }$ of C consecutive tokens, indexed from $j = 1$ . For chunk selection, each full chunk $\mathcal { C } _ { j }$ is represented by the mean $r _ { j }$ of its C key vectors. For a query at position t, attention always includes the tokens within the chunk containing t, which are at positions $\dot { \mathcal { T } } _ { t } = \{ ( \lceil t / C \rceil - 1 ) C + \dot { \mathrm { 1 } } , \ldots , t \}$ , and the tokens ${ \ w } _ { t }$ in up to W immediately preceding chunks. The eligible set $\mathcal { E } _ { t }$ contains the indices of the full historical chunks outside the recent window. A separate query projection produces $q _ { t } ^ { I }$ to score their representatives and select a fixed budget of $k _ { \mathrm { c h u n k s } }$ chunks:

$$
r _ { j } = \frac { 1 } { C } \sum _ { u \in \mathcal { C } _ { j } } k _ { u } , \quad s _ { t , j } = \left. q _ { t } ^ { I } , r _ { j } \right. , \qquad j \in \mathcal { E } _ { t } = \left\{ 1 , \dots , \operatorname* { m a x } \left( 0 , \left\lceil \frac { t } { C } \right\rceil - W - 1 \right) \right\} ,\tag{2}
$$

$$
\begin{array} { r } {  { \boldsymbol { S } } _ { t } =  { \operatorname { T o p K } } _ { j \in  { \mathcal { E } } _ { t } } \left( s _ { t , j } , k _ { \mathrm { c h u n k s } } \right) . } \end{array}\tag{3}
$$

An upper bound of zero denotes an empty eligible set. The index query serves only selection; its projection is initialized in Section 2.4. We use all eligible chunks if fewer than $k _ { \mathrm { c h u n k s } }$ are available. The set of attended token positions is $\mathcal { U } _ { t } = ( \bigcup _ { j \in \mathcal { S } _ { t } } \mathcal { C } _ { j } ^ { \top } ) \cup \mathcal { W } _ { t } \cup \mathcal { T } _ { t }$ . Attention uses the original Q/K/V and normalizes over this set:

$$
o _ { t } ^ { \mathrm { C I S A } } = \mathrm { s o f t m a x } \Bigg ( \frac { q _ { t } K _ { \mathcal { U } _ { t } } ^ { \top } } { \sqrt { d } } \Bigg ) V _ { \mathcal { U } _ { t } } ,\tag{4}
$$

where $K _ { U _ { t } } , V _ { U _ { t } }$ denote the key and value rows at positions $\mathcal { U } _ { t }$ . When all eligible chunks fit within the budget, this reduces to dense attention.

## 2.2 ONE INDEXER PER KV GROUP

In grouped-query attention, let ${ \mathcal { H } } _ { a }$ denote the set of query heads sharing the keys and values of KV group a, and let $H = | \mathcal { H } _ { a } |$ . CISA uses one index query $q _ { t , a } ^ { I }$ for the group: Equations 2–3 form representatives $r _ { j , a }$ from the KV group’s keys and select one chunk set $\boldsymbol { \mathcal { S } } _ { t , a }$ , which is shared by all H query heads. Each head still computes Equation 4 with its own query $q _ { t , h }$ and the group’s original keys and values $k _ { u , a } , v _ { u , a }$ . Thus, the heads read the same tokens but assign their own attention weights.

Representatives follow the main keys’ tensor-parallel ownership and have dimension $d _ { I } = d$ . For a sequence of N tokens, the representative cache stores approximately $N / C$ vectors per local KV group, each of dimension $d _ { I }$ , avoiding a separate token-level index-key cache.

## 2.3 QShare: SHARING CHUNK SELECTION ACROSS QUERY POSITIONS

During prefill, QShare uses one chunk selection for multiple adjacent query positions within each KV group. For a prefix of $L _ { \mathrm { p r e f i x } }$ tokens, QShare-G partitions the query positions into disjoint groups $\bar { \mathcal { G } _ { m } } = \{ ( m - 1 ) G + 1 , \bar { \ldots } , \operatorname* { m i n } ( m G , L _ { \mathrm { p r e f i x } } ) \}$ , where $m = 1 , \bar { \cdot } \cdot \cdot , \bar { \lceil } \bar { L _ { \mathrm { p r e f i x } } } / G \rceil$ . Each group contains $G$ consecutive query positions, except the final group, which may be shorter. Standard autoregressive decoding contributes one new query position per sequence, so $G = 1$

Within each group $\mathcal { G } _ { m }$ and KV group a, we form a shared index query by averaging the projected index queries:

$$
\bar { q } _ { m , a } ^ { I } = \frac { 1 } { | { \mathcal G } _ { m } | } \sum _ { t \in { \mathcal G } _ { m } } q _ { t , a } ^ { I } .\tag{5}
$$

We use $\bar { q } _ { m , a } ^ { I }$ in Equations 2–3 to score the chunk representatives and select one historical chunk set for $\mathcal { G } _ { m }$ . Only chunk indices in $\textstyle \bigcap _ { t \in { \mathcal { G } } _ { m } } { \mathcal { E } } _ { t }$ are considered, so every selected chunk is eligible for every position in the group. Each head at each position then computes attention using its original query and the original K/V from the selected chunks, recent window, and current-chunk tail, with a position-specific causal mask.

Sharing the chunk selection amortizes representative scoring and top-k selection across positions and lets more query rows reuse the selected K/V. The resulting gains in KV reuse and GPU tile utilization are examined in Section 3.3.

## 2.4 INITIALIZING THE INDEXER FROM DENSE ATTENTION

We initialize each group’s index-query projection from a pretrained dense attention layer without gradient-based training.

For each converted attention layer and KV group a, the candidate set $\scriptstyle { { \mathcal { C } } _ { a } }$ contains $H + 1$ query projections: the projections of the H heads in ${ \mathcal { H } } _ { a }$ and their rowwise average.

For each sequence x and query position t in the calibration set $\mathcal { D } _ { \mathrm { c a l } }$ , we define the indexer’s target by averaging the dense attention distributions of the H query heads in KV group a:

$$
p _ { x , t , a } ^ { \star } = \frac { 1 } { H } \sum _ { h \in \mathcal { H } _ { a } } \mathrm { s o f t m a x } \left( \frac { q _ { x , t , h } K _ { x , 1 : t , a } ^ { \top } } { \sqrt { d } } \right) .\tag{6}
$$

We include the averaged query projection as a candidate for approximating this target. Because softmax is nonlinear, however, the attention distribution produced by the averaged query is not generally equal to the average of the heads’ attention distributions. Each candidate projection $c \in { \mathcal { C } } _ { a }$ produces a query whose attention distribution over the same causal keys is $p _ { x , t , a } ^ { ( c ) } .$ Following DSA’s KL-matching objective (DeepSeek-AI, 2025), we select the candidate whose distribution most closely matches the target on the calibration set:

$$
c _ { a } ^ { \star } = \arg \operatorname* { m i n } _ { c \in \mathcal { C } _ { a } } \mathbb { E } _ { ( \boldsymbol { x } , t ) \sim \mathcal { D } _ { \mathrm { c a l } } } [ D _ { \mathrm { K L } } ( p _ { x , t , a } ^ { \star }  p _ { x , t , a } ^ { ( c ) } ) ] .\tag{7}
$$

The selected projection weights are copied into the group’s indexer.<sup>1</sup> The base-model weights remain unchanged.

## 3 SYSTEM REALIZATION AND EFFICIENCY

## 3.1 PAGED EXECUTION IN VLLM

CISA reads selected chunks directly from the existing KV cache. In vLLM, this cache is organized into fixed-size memory blocks called pages. These pages need not be adjacent in memory; a block table maps their positions in the sequence to their physical storage locations (Kwon et al., 2023). We set $C = 1 6$ to match the page size, so each selected chunk corresponds to one cache page. CISA supplies the selected physical page IDs to the existing attention kernel, which reads the original K/V in place without gathering or copying them into a separate buffer.

The serving implementation also caches the chunk representatives. As full chunks become available, it updates this cache and reuses the stored summaries during selection. For equal index and attention head dimensions, one representative per $C = 1 6$ tokens requires $1 / ( 2 C ) = \dot { 1 } / 3 2$ as many elements as the corresponding main K+V state. The full KV history remains stored so later queries can select different chunks. CISA reduces cache reads rather than cache capacity.

The runtime skips representative scoring and top-k selection when all eligible chunks fit within the budget $k _ { \mathrm { c h u n k s } }$ . Prefill query blocks crossing this threshold are split into a full-attention prefix and a sparse-attention suffix.

## 3.2 ATTENTION-TRAFFIC MODEL

Let d be the attention head dimension and $d _ { I }$ the representative dimension. We analyze traffic per KV group, whose K/V loads are reused across its assigned query heads. For a query with history length $N$ , dense attention reads approximately $2 N d$ elements from the K/V cache. Once sparse selection is active, CISA reads $2 k _ { \mathrm { c h u n k s } } C d$ selected K/V elements and scans $( N / C ) d _ { I }$ representative elements. With $Q S h a r e – G$ , the representative scan is amortized across G adjacent queries. Ignoring the recent window, current-chunk tail, projections, metadata, and representative construction, we express CISA’s estimated memory-read volume for a query near the end of prefill as a fraction of that required by dense attention:

$$
\rho _ { \mathrm { e n d } } ( N , k _ { \mathrm { c h u n k s } } ) \approx \underbrace { \frac { k _ { \mathrm { c h u n k s } } C } { N } } _ { \mathrm { K / V \ r e a d s } } + \underbrace { \frac { d _ { I } } { 2 C G d } } _ { \mathrm { r e p r e s e n t a t i v e \ r e a d s } } .\tag{8}
$$

This per-query expression does not describe a complete causal prefill: early queries still attend to all available history until $k _ { \mathrm { c h u n k s } } C$ tokens are available. Let $\alpha = k _ { \mathrm { c h u n k s } } C / N$ denote the selected historical token budget as a fraction of the full prefix length. Integrating traffic over the full causal triangle gives

$$
\rho _ { \mathrm { p r e f l l } } \approx \underbrace { 2 \alpha - \alpha ^ { 2 } } _ { \mathrm { K / V \ r e a d s } } + \underbrace { ( 1 - \alpha ^ { 2 } ) \frac { d _ { I } } { 2 C G d } } _ { \mathrm { r e p r e s e n t a t i v e \ r e a d s } } .\tag{9}
$$

The inverse of either ratio is a bandwidth-only speedup estimate based on modeled memory reads, not a bound on measured kernel or end-to-end speedup.

For $C = 1 6 , d _ { I } = d ,$ and $G = 4 ,$ , selecting $\begin{array} { r } { k _ { \mathrm { c h u n k s } } = \frac { N } { 8 C } } \end{array}$ , corresponding to 8x nominal compression of the selected remote-history span, gives a 7.53× estimate near the end of prefill and a 4.13× fullprefill estimate. Selecting $\begin{array} { r } { k _ { \mathrm { c h u n k s } } = \frac { N } { 1 6 C } \left( 1 6 \mathrm { x } \right. } \end{array}$ compression) yields $1 4 . 2 2 \times$ and $7 . 7 6 \times$ , respectively. Conventional single-token decode has $G = 1$ , reducing the corresponding per-query ceilings to 6.40× and $1 0 . 6 7 \times$ . The fixed recent window and at-most-one-chunk causal tail lower these ceilings by a small additive term whose relative contribution decreases with context length.

## 3.3 FROM COMPONENT SAVINGS TO THROUGHPUT

The traffic reductions above do not by themselves guarantee proportional time savings: selected pages must feed well-utilized attention kernels, and scanning the representatives must be efficient.

Ultra has 32 query heads per KV head, so $G = 4$ provides $4 \times 3 2 = 1 2 8$ query rows per selected KV set, filling the $M = 1 2 8$ FlashAttention tile used in our implementation. Inkling-Small has four query heads per KV head: $G = 4$ supplies 16 rows and underfills the tile, whereas $G = 3 2$ fills its $3 2 \times 4 = 1 2 8$ rows. Larger sharing groups also reduce the number of separate selections, which may reduce quality.

Figure 2 illustrates these effects using one-layer BF16 Inkling-Small benchmarks at 1M context, batch size 16, and $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ . Relative to dense attention, terminal-prefill GPU-busy speedups reach $1 . 2 1 \times { ( G = 4 ) } , 3 . 3 0 \times { ( G = 1 6 ) }$ , and $5 . 1 1 \times ( G = 3 2 )$ . Single-token decode, with effective positional sharing $G = 1$ , achieves 3.05×.

![](images/318a52656fcd3823cb99e29ff45263d63cc0d9248f49d4d7c9c4fd394529207d.jpg)

![](images/36787c2611e1a7a4ee90fc31af42c20372dd3854b49b8bce8139e64d92558f50.jpg)  
Figure 2: Query sharing and attention-path efficiency on Inkling-Small at approximately one million context tokens, batch size 16, and $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ . (a) Terminal prefill with $G \in \{ 4 , 1 6 , 3 2 \}$ . Query sharing increases reuse of loaded KV blocks across query positions and can improve attention-tile utilization, depending on the number of query heads per KV head. (b) Single-token decode with effective positional sharing $G = 1$ . Bars show mean GPU busy time over ten retained forwards; labels give total milliseconds and speedup relative to dense attention. Measurements use a one-layer BF16 model. Component definitions, full timings, and setup appear in Appendix Table 11.

For Ultra’s per-rank TP2 attention geometry at the same context and budget, one-layer BF16 benchmarks at batch size 32 yield core-attention speedups of 11.70× for terminal prefill and 12.99× for decode. Including scoring, selection, and other measured GPU work in this one-layer benchmark, the corresponding speedups are 8.61× and 2.98×.

Appendix D provides component timings and experimental details. These measurements isolate attention-path efficiency; Section 4.4 evaluates how the gains translate into full-model serving throughput.

## 4 EVALUATION

We evaluate CISA in four stages: budget-matched RULER scaling, heterogeneous long-context workloads, capability preservation, and controlled quality–throughput measurements. Unless marked otherwise, sparse runs use 16-token chunks, a 128-page recent window, and $Q S h a r e – 4 ;$ each result states its fixed chunk budget $k _ { \mathrm { c h u n k s } } .$ tables and figures use k or K as shorthand for $k _ { \mathrm { c h u n k s } }$ . Model configurations, sampling, cohort construction, and detailed statistics appear in Appendix A.

## 4.1 BUDGET-MATCHED RULER SCALING

We evaluate RULER scaling on Nano and Ultra with BF16 weights and on Inkling-Small with NVFP4 weights (Figure 3). Nano and Ultra use $G = 4 .$ , while Inkling-Small uses $G = 1 6$ for this RULER sweep. All three models follow the same budget schedule: $k _ { \mathrm { c h u n k s } } = 1 , 0 2 4 , 2 , 0 4 8 , 4 , 0 9 6$ , and 4,096 at 128K, 256K, 512K, and 1M, respectively, with each budget fixed throughout its run. At the nominal context endpoints, these budgets select 8× fewer remote historical tokens through 512K and 16× fewer at 1M, before adding the recent window and current-page tail.

On Nano and Ultra, CISA remains within 2.05 points of dense attention across all eight model–length comparisons, with smaller gaps at 1M than at 512K. With G = 16, Inkling-Small trails dense attention by 2.94 points at 128K and 2.95 points at 256K, then scores 1.93 points above dense at 512K and 4.71 points above at 1M. Because the budget changes with context length, this sweep measures the combined operating point rather than the independent effect of either variable. Additional RULER results across recall budgets and query-sharing groups are in Appendix Table 5.

## 4.2 HETEROGENEOUS WORKLOADS WITH VERY LONG CONTEXTS

RULER provides a controlled scaling sweep, but heterogeneous workloads place different demands on attention. We therefore evaluate Ultra and Inkling-Small on MRCR, LongBench v2, and GraphWalks using NVFP4 weights at $k _ { \mathrm { c h u n k s } } ~ \in ~ \{ 2 , 0 4 8 , 4 , 0 9 6 \}$ . We use $G = 4$ for Ultra and $G = 3 2$ for Inkling-Small, following the hardware considerations in Section 3.3. Within each model, dense and sparse comparisons use the same tasks and repeats, with prompt-length cohorts fixed by the dense reference. Table 1 summarizes aggregate accuracy and changes in mean generation length.

![](images/8c89d68517297120da01a73435ed94477f6fb4fdba9eb1064fa75c4bc8b6ded3.jpg)  
Figure 3: Budget-matched RULER results for Nano and Ultra (BF16, G = 4) and Inkling-Small (NVFP4, G = 16). Each sparse run keeps the labeled budget fixed. Panels share the same accuracy scale; signed labels show CISA minus dense accuracy in percentage points. Exact scores, budgets, and remote-compression factors appear in Appendix Table 5.

Table 1: Ultra and Inkling-Small NVFP4 long-context accuracy and generation length at two fixed budgets. Ultra uses $G = 4 ;$ Inkling-Small uses $G = 3 2$ . Scores are percentages, and score $\Delta$ is CISA minus the corresponding dense baseline in percentage points. Verbosity $\Delta$ is the relative change in mean generated tokens. The two budgets select 32,768 and 65,536 remote historical tokens, respectively, in addition to the fixed recent window and current-page tail. Sampling and cache settings appear in Appendix A.1.
<table><tr><td>Benchmark</td><td>Dense</td><td> $\mathrm { C I S A } \ k = 2 { , } 0 4 8$ </td><td>∆</td><td>Verb. ∆</td><td> $\mathrm { C I S A } \ : k = 4 { , } 0 9 6$ </td><td></td><td> $\Delta$ </td><td>Verb.  $\Delta$ </td></tr><tr><td colspan="3">Ultra NVFP4  $( G = 4 )$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MRCR</td><td>47.34</td><td></td><td>46.64</td><td>-0.70</td><td>+2.9%</td><td>46.96</td><td>-0.38</td><td>+3.1%</td></tr><tr><td>LongBench v2</td><td>61.38</td><td></td><td>61.38</td><td>+0.00</td><td>+8.0%</td><td>62.48</td><td>+1.09</td><td>+4.7%</td></tr><tr><td>GraphWalks</td><td>53.42</td><td></td><td>49.84</td><td>-3.59</td><td>+14.1%</td><td>51.93</td><td>-1.49</td><td>+10.6%</td></tr><tr><td colspan="3">Inkling-Small NVFP4 (G = 32)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MRCR</td><td>41.46</td><td></td><td>43.33</td><td>+1.87</td><td>-3.8%</td><td>42.41</td><td>+0.94</td><td>-3.6%</td></tr><tr><td>LongBench v2</td><td>63.27</td><td></td><td>64.31</td><td>+1.04</td><td>+24.3%</td><td>63.37</td><td>+0.10</td><td>+3.9%</td></tr><tr><td>GraphWalks</td><td>57.33</td><td></td><td>56.81</td><td>-0.52</td><td>+128.3%</td><td>57.05</td><td>-0.28</td><td>+78.2%</td></tr></table>

Across both models, MRCR and LongBench v2 remain close to dense attention at both budgets. GraphWalks is more sensitive to the budget on Ultra: increasing $k _ { \mathrm { c h u n k s } }$ from 2,048 to 4,096 reduces the accuracy deficit from 3.59 to 1.49 points. Inkling-Small’s GraphWalks scores remain closer to dense, with deficits of 0.52 and 0.28 points at the smaller and larger budgets, respectively.

The breakdown by prompt length shows where Ultra benefits from the larger budget (Appendix Figure 5 and Table 4). In the 256–512K GraphWalks cohort, increasing $k _ { \mathrm { c h u n k s } }$ from 2,048 to $4 { , } 0 9 6$ narrows the gap from 7.24 to 1.79 points. At one million tokens and above, however, dense GraphWalks accuracy falls to 7.08%, and both sparse settings remain below it. MRCR tracks dense attention across the length range. LongBench v2 remains within five points through 512K, while its longer cohorts have wide uncertainty because they contain only 12, 7, and 33 prompts. Appendix Figure 6 reports the corresponding breakdown for Inkling-Small.

Table 2: Capability preservation for Nano BF16 and Ultra NVFP4 at their target budgets. Scores use each benchmark’s native percentage metric except GDPval, reported in Elo; score $\Delta$ is in percentage points or Elo, respectively, and verbosity $\Delta$ is the relative change in mean generated tokens.
<table><tr><td>Model</td><td>Benchmark</td><td>k</td><td>Dense</td><td>CISA</td><td> $\Delta$ </td><td>Verbosity  $\Delta$ </td></tr><tr><td>Nano BF16</td><td>AA-LCR</td><td>1,024</td><td>34.94</td><td>34.25</td><td>-0.69</td><td>-9.1%</td></tr><tr><td rowspan="4">Ultra NVFP4</td><td>AA-LCR</td><td>1,024</td><td>65.00</td><td>65.88</td><td>+0.88</td><td>+1.8%</td></tr><tr><td>SWE-bench Verified</td><td>2,048</td><td>69.20</td><td>69.93</td><td>+0.73</td><td>-0.2%</td></tr><tr><td>Terminal-Bench 2.0</td><td>2,048</td><td>48.03</td><td>49.44</td><td>+1.40</td><td>-0.7%</td></tr><tr><td>Terminal-Bench Hard</td><td>1,024</td><td>35.94</td><td>35.42</td><td>-0.52</td><td>+2.8%</td></tr><tr><td colspan="2">GDPval</td><td>2,048</td><td>1395.59</td><td>1408.71</td><td>+13.12</td><td>-2.8%</td></tr></table>

On Inkling-Small’s GraphWalks evaluation, CISA generates 128.3% more tokens on average than dense attention at $k _ { \mathrm { c h u n k s } } = 2$ ,048 and 78.2% more at $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ . Inspection reveals longer reasoning, accompanied by some repetitive generations that exhaust the output budget.

## 4.3 REASONING, AGENTIC, AND PROFESSIONAL CAPABILITY

Beyond long-context retrieval, we evaluate whether CISA preserves the model’s reasoning, agentic, and professional-task capabilities. Table 2 reports the Nano control and the primary Ultra NVFP4 capability results, including generation-length changes; the Ultra BF16 controls and full matrix remain in Appendix Table $^ { 6 . }$

Across the percentage-scored Ultra NVFP4 evaluations in Table 2, CISA changes the point estimates by −0.52 to +1.40 points; the Terminal-Bench rows use context summarization and therefore do not establish unchanged compaction behavior. On GDPval the failure-adjusted Elo point estimates are 1395.59 for dense attention and 1408.71 for CISA, a difference of +13.12. Appendix A.4 details the adjustment and the matched-rollout comparison.

## 4.4 QUALITY, THROUGHPUT, AND OUTPUT LENGTH

We evaluate how recall budget and query sharing jointly affect quality, measured throughput, and mean output length on Ultra and Inkling-Small. We define quality as the equal-weight mean score across RULER 1M, MRCR200, GraphWalks-200, and AA-LCR. The base profile selects 4,096 historical chunks for the first three benchmarks and 1,024 for AA-LCR. The reduced profile halves these budgets. We vary G across profiles. Both models use the same tasks and reference answers, and within each model quality and output length are measured on the same generations.

We measure controlled fixed-output total token throughput on MRCR200, the 200 longest prompts in our MRCR corpus, with contexts around one million tokens. Every configuration serves the same prompts, and each request generates exactly 4,096 tokens, fixing the input and output workload across configurations. We report speedup S relative to the same model’s dense baseline.

In the quality evaluations, some CISA configurations generate more output tokens per response on average than dense attention. In deployment, longer responses could offset part of the benefit of faster token generation. For each benchmark, we divide CISA’s mean output token count, including reasoning tokens, by that of the dense baseline. We average the four ratios with equal weight to obtain R, then define the verbosity-adjusted speedup proxy $S _ { \mathrm { a d j } } = S / R .$ . This simple adjustment assumes that workload cost scales linearly with output length. It does not model fixed prefix cost, nonlinear batching effects, or differences between the quality and throughput workloads.

Figure 4 shows the resulting trade-offs. For Ultra, increasing sharing from G = 4 to G = 8 at the base budgets gives a higher mean score point estimate and a higher adjusted speedup proxy than halving the budgets at G = 4. For Inkling-Small at $G = 3 2$ , the base profile’s measured 1.54× speedup corresponds to an adjusted proxy of 1.12×, while its mean score remains above the dense reference. The reduced profile achieves a larger measured speedup of 1.64×, but its higher mean output token count lower the adjusted proxy to 0.93×. Together, these results illustrate why recall budget, query sharing, and output length should be considered jointly. Appendix Figure 7 reports the unadjusted speedups, and Appendix B.1 gives the evaluation subsets and measurement details.

![](images/776f08d29af43b00ed682ef67ea290e592ded2286a9786d739e2d2bc1426ba0c.jpg)  
Figure 4: Quality–cost operating points for (a) Nemotron Ultra and (b) Inkling-Small, both NVFP4. Quality is the equal-weight mean score across RULER 1M, MRCR200, GraphWalks-200, and AA-LCR. The horizontal axis divides controlled, fixed-output total token throughout speedup by the mean benchmark-relative output-length ratio. It is an estimate that accounts for differences in output length, not a direct measurement of natural completion throughput. Reduced profiles halve the base recall budgets. Lines connect point estimates as G increases within each profile. Exact coordinates and measurement details appear in Appendix Table 7 and Appendix B.1.

## 5 RELATED WORK

FlashAttention tiles exact attention to reduce IO (Dao et al., 2022), while PagedAttention and vLLM manage non-contiguous KV pages and continuous batching (Kwon et al., 2023). CISA retains paged KV management but supplies a query-dependent sparse block table.

Quest scores KV pages using coordinatewise key bounds and produces a separate plan per query head (Tang et al., 2024). This design is natural for MAH, where each query head has its own KV states. Under GQA, however, separate plans for each query head can reduce reuse among query heads that share the same KV head; CISA instead creates one plan per KV group, and QShare reuses that plan across query positions. On Nano, at a 1M-token context, the hybrid Quest-bound + CISA index control scores 84.95 at G = 1 and 84.41 at G = 4 compared with 85.87 for the budget-aligned CISA reference (Appendix Table 5).

Longformer and BigBird train with fixed sparse structure (Beltagy et al., 2020; Zaheer et al., 2020); NSA, DSA, and MSA learn selectors with GPU-aware sparse execution (Yuan et al., 2025; DeepSeek-AI, 2025; Lai et al., 2026). CISA does not claim block selection, per-group indexing, or training-free sparsification in isolation; it combines fixed-budget recall, a dense-initialized indexer, TP-local main-key summaries, and cross-position reuse in one paged-attention path.

## 6 LIMITATIONS

Initialization requires no gradient updates but uses calibration activations and candidate selection; it is not data-free. CISA reduces attention reads but retains the main K/V state for future selection; it is not KV-cache eviction or capacity compression. Quality also remains workload- and budget-dependent: at Ultra NVFP4, $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ leaves the ≥ 1M GraphWalks cohort 4.62 points below dense attention. We test two model families, one serving stack, and a limited budget grid.

The controlled serving workload combines prefill and decode, while the analytical ceilings omit selection, metadata. Finally, conventional autoregressive decode exposes one query at a time, so QShare currently benefits prefill only. Extending selection reuse to MTP (Gloeckle et al., 2024) or speculative decoding remains future work.

## 7 CONCLUSION

Retrofitting dense attention is both an approximation problem and a systems problem. CISA addresses them together through chunk-level recall, KV-group-local indexing, and shared prefill selection, without requiring gradient updates. Results on Nemotron and Inkling-Small show that this design can keep many benchmark scores close to dense attention while improving fixed-output throughput. The trade-off remains workload-dependent: longer generations can reduce the serving benefit even though benchmark scores remain close. These results support considering recall budget and query sharing jointly during deployment tuning. Future work includes learned indexers, adaptive budgets, and decode-side selection reuse.

## REFERENCES

Artificial Analysis Team. Artificial analysis long context reasoning benchmark. Hugging Face dataset, 2025. URL https://huggingface.co/datasets/ArtificialAnalysis/ AA-LCR.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. arXiv preprint arXiv:2412.15204, 2024. URL https://arxiv.org/abs/2412.15204.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2205.14135.

DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025. URL https://arxiv.org/abs/2512.02556.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024. URL https://arxiv.org/abs/2404.19737.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024. URL https://arxiv.org/abs/2404. 06654.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=VTF8yNQM66.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023. URL https://arxiv.org/abs/2309.06180.

Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, and Pengyu Zhao. MiniMax Sparse Attention. arXiv preprint arXiv:2606.13392, 2026. URL https://arxiv.org/abs/2606.13392.

Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, et al. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. arXiv preprint arXiv:2601.11868, 2026. URL https://arxiv.org/abs/2601.11868.

OpenAI. GraphWalks: A multi-hop reasoning long-context benchmark. Hugging Face dataset, 2025. URL https://huggingface.co/datasets/openai/graphwalks.

Tejal Patwardhan, Rachel Dias, Elizabeth Proehl, Grace Kim, Michele Wang, Olivia Watkins, Simón Posada Fishman, Marwan Aljubeh, Phoebe Thacker, Laurance Fauconnet, Natalie S. Kim, Patrick Chao, Samuel Miserendino, Gildas Chabot, David Li, Michael Sharman, Alexandra Barr, Amelia Glaese, and Jerry Tworek. GDPval: Evaluating AI model performance on realworld economically valuable tasks. arXiv preprint arXiv:2510.04374, 2025. URL https: //arxiv.org/abs/2510.04374.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity for efficient long-context LLM inference. arXiv preprint arXiv:2406.10774, 2024. URL https://arxiv.org/abs/2406.10774.

Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui, Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, Rohan Anil, Ethan Dyer, Siamak Shakeri, et al. Michelangelo: Long context evaluations beyond haystacks via latent structure queries. arXiv preprint arXiv:2409.12640, 2024. URL https://arxiv.org/abs/2409.12640.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics, 2025. URL https://aclanthology.org/2025.acl-long.1126/.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. In Advances in Neural Information Processing Systems, 2020.

## A ADDITIONAL EVALUATION DETAILS

This appendix specifies the evaluation settings and cohorts, then reports the length-stratified results and capability controls underlying Section 4.

## A.1 SAMPLING AND EXECUTION SETTINGS

Model configurations. Nano and Ultra are hybrid Mamba–Transformer MoE models. Inkling-Small combines global and sliding-window attention with MoE and short-convolution components. Their configurations are summarized below; query and KV head counts refer to global attention, with head dimension 128 throughout.

<table><tr><td>Model</td><td>Total blocks/layers</td><td>Global-attention layers</td><td>Hidden dimension</td><td>Query heads</td><td>KV heads</td></tr><tr><td>Nano</td><td>52</td><td></td><td>2,688</td><td>32</td><td>2</td></tr><tr><td>Ultra</td><td>108</td><td>12</td><td>8,192</td><td>64</td><td>2</td></tr><tr><td>Inkling-Small</td><td>42</td><td></td><td>4,096</td><td>32</td><td>8</td></tr></table>

Inkling-Small’s remaining 35 layers use sliding-window attention; CISA replaces only its seven global-attention layers. Across all three models, CISA preserves the base-model weights and adds initialized index-query projections.

Sampling. For Ultra and Inkling-Small, MRCR, LongBench v2, and GraphWalks evaluations use top-p = 1, four repeats per task, and a 65,536-token output cap. RULER uses temperature $1 0 ^ { - 5 }$ and top-p = 0.99, with 100 examples per subtask across 13 subtasks.

The capability evaluations use temperature 1 and the following task counts, repeats, and top-p settings.

<table><tr><td>Benchmark</td><td>Tasks</td><td>Repeats per task</td><td>Top-p</td></tr><tr><td>AA-LCR</td><td>100</td><td></td><td>16 0.95</td></tr><tr><td>SWE-bench Verified</td><td>500</td><td>3</td><td>0.95</td></tr><tr><td>Terminal-Bench 2.0</td><td>89</td><td>8</td><td>0.95 (BF16); 1 (NVFP4)</td></tr><tr><td>Terminal-Bench Hard</td><td>48</td><td>8</td><td>0.95</td></tr><tr><td>GDPval</td><td>220</td><td>2</td><td>0.95</td></tr></table>

The separate AA-LCR operating-point experiments retain 100 tasks and 16 repeats, using top-p = 1 for Ultra and $\mathrm { t o p } { - } p = 0 . 9 5$ for Inkling-Small. GDPval’s treatment of incomplete rollouts is specified in Appendix A.4.

Execution settings. The Ultra NVFP4 GraphWalks runs in Table 1 use BF16 KV and Mamba caches, with FP16 Mamba SSM state and the FlashInfer Mamba backend.

## A.2 FIXED BUDGETS BY INPUT LENGTH

We report results by input length to examine how quality preservation varies with context length at a fixed chunk budget. Cohorts use each model’s dense-reference prompt-token counts and remain fixed across dense and CISA runs. Ultra uses the same cohorts for BF16 and NVFP4; Inkling-Small uses its own token counts, so cohort membership can differ between models.

Figures 5 and 6 show NVFP4 results for Ultra and Inkling-Small, respectively. Each CISA configuration keeps its budget of $k _ { \mathrm { c h u n k s } } = 2 , 0 4 8$ or 4,096 fixed throughout the run. Tables 3 and 4 provide Ultra’s BF16 and NVFP4 scores and cohort sizes.

![](images/a8774ed09a119c86bae73b95555546e46fba2cb9f623abc5158ce6889c44423d.jpg)

![](images/cbe85685ad14501153c63d53a725e0c1022c844f4420d625ef069278a3b86033.jpg)

![](images/638bb6e312d281454f75bb8f9f86a924ac213e235f76788c303e6f3cfb6e0509.jpg)  
Figure 5: Ultra NVFP4 scores by dense-reference prompt length, comparing dense attention with CISA at $k _ { \mathrm { c h u n k s } } \in \{ 2 , 0 4 8 , 4 , 0 9 6 \}$ and $G = 4$ . Labels give the number of distinct prompts. Error bars show 95% task-bootstrap confidence intervals, retaining all four repeats of each sampled task. Table 4 reports exact scores and cohort sizes.

![](images/6c0bd97215f0c0d0259ca45d25d3ceb6a3548ab67197c5c29e63c348e44edf9f.jpg)

![](images/8cfbfe17d8022f207d770702937c94e680400ff637127dee590f9448f3a66b49.jpg)

![](images/61a639e5c9dcd175ccf68109210d4f1f6e44ee8663f24b000367948395638e8c.jpg)  
Figure 6: Inkling-Small NVFP4 scores by dense-reference prompt length, comparing dense attention with CISA at $k _ { \mathrm { c h u n k s } } \in \{ 2 , 0 4 8 , 4 , 0 9 6 \}$ and $G = 3 2$ . Labels give the number of distinct prompts; shaded buckets contain no prompts. Error bars show pointwise 95% confidence intervals from $2 0 { , } 0 0 0$ task-bootstrap resamples, retaining all four repeats of each sampled task.

Table 3: Ultra BF16 scores by dense-reference prompt length (%). Both CISA configurations use 16-token chunks, a 128-page recent window, and $\bar { G } = \bar { 4 } . \Delta$ is CISA minus dense in percentage points. Empty MRCR final answers are scored as zero: one for dense attention and two for $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$
<table><tr><td>Benchmark</td><td>Input length</td><td>Prompts</td><td>Rollouts</td><td>Dense</td><td> $k = 2 , 0 4 8$ </td><td> $\Delta$ </td><td> $k = 4 , 0 9 6$ </td><td> $\Delta$ </td></tr><tr><td rowspan="7">MRCR</td><td>&lt; 32K</td><td>863</td><td>3,452</td><td>84.67</td><td>84.85</td><td>+0.18</td><td>84.73</td><td>+0.06</td></tr><tr><td>32-128K</td><td>601</td><td>2,404</td><td>46.58</td><td>46.47</td><td>-0.11</td><td>46.07</td><td>-0.51</td></tr><tr><td>128-256K</td><td>282</td><td>1,128</td><td>18.36</td><td>17.78</td><td>-0.58</td><td>18.34</td><td>-0.02</td></tr><tr><td>256–512K</td><td>316</td><td>1,264</td><td>10.36</td><td>7.64</td><td>-2.72</td><td>9.50</td><td>-0.86</td></tr><tr><td>512-768K</td><td>136</td><td>544</td><td>5.66</td><td>4.96</td><td>-0.70</td><td>5.20</td><td>-0.46</td></tr><tr><td>768K-1M</td><td>153</td><td>612</td><td>4.76</td><td>4.07</td><td>-0.69</td><td>4.69</td><td>-0.07</td></tr><tr><td>≥ 1M</td><td>49</td><td>196</td><td>5.02</td><td>4.43</td><td>-0.59</td><td>4.74</td><td>-0.27</td></tr><tr><td rowspan="7">LongBench v2</td><td>&lt; 32K</td><td>113</td><td>452</td><td>61.95</td><td>62.17</td><td>+0.22</td><td>62.17</td><td>+0.22</td></tr><tr><td>32-128K</td><td>188</td><td>752</td><td>62.90</td><td>62.23</td><td>-0.66</td><td>63.43</td><td>+0.53</td></tr><tr><td>128-256K</td><td>100</td><td>400</td><td>65.50</td><td>68.50</td><td>+3.00</td><td>66.50</td><td>+1.00</td></tr><tr><td>256-512K</td><td>50</td><td>200</td><td>63.50</td><td>64.50</td><td>+1.00</td><td>68.00</td><td>+4.50</td></tr><tr><td>512-768K</td><td>12</td><td>48</td><td>41.67</td><td>54.17</td><td>+12.50</td><td>50.00</td><td>+8.33</td></tr><tr><td>768K-1M</td><td>7</td><td>28</td><td>53.57</td><td>57.14</td><td>+3.57</td><td>64.29</td><td>+10.71</td></tr><tr><td>≥ 1M</td><td>33</td><td>132</td><td>53.79</td><td>50.76</td><td>-3.03</td><td>61.36</td><td>+7.58</td></tr><tr><td rowspan="7">GraphWalks</td><td>&lt; 32K</td><td>450</td><td>1,800</td><td>93.82</td><td>93.37</td><td>-0.45</td><td>92.99</td><td>-0.83</td></tr><tr><td>32-128K</td><td>200</td><td>800</td><td>53.39</td><td>50.92</td><td>-2.47</td><td>54.09</td><td>+0.69</td></tr><tr><td>128-256K</td><td>100</td><td>400</td><td>38.82</td><td>34.74</td><td>-4.07</td><td>35.11</td><td>-3.70</td></tr><tr><td>256-512K</td><td>200</td><td>800</td><td>23.98</td><td>18.87</td><td>-5.11</td><td>20.84</td><td>-3.15</td></tr><tr><td>512-768K</td><td>0</td><td>0</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>768K-1M</td><td>0</td><td>0</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>≥ 1M</td><td>200</td><td>800</td><td>7.36</td><td>4.23</td><td>-3.12</td><td>5.35</td><td>-2.01</td></tr></table>

For Ultra, the three longest LongBench v2 cohorts contain only 12, 7, and 33 prompts and should be interpreted cautiously. Its two intermediate GraphWalks cohorts between 512K and 1M are empty.

Table 4: Ultra NVFP4 scores by dense-reference prompt length (%), with BF16 KV state. Both CISA configurations use 16-token chunks, a 128-page recent window, and G = 4. ∆ is CISA minus dense in percentage points.
<table><tr><td>Benchmark</td><td>Input length</td><td>Prompts</td><td>Rollouts</td><td>Dense</td><td> $k = 2 , 0 4 8$ </td><td>∆</td><td> $k = 4 , 0 9 6$ </td><td>Δ</td></tr><tr><td rowspan="7">MRCR</td><td>&lt; 32K</td><td>863</td><td>3,452</td><td>84.32</td><td>84.30</td><td>-0.02</td><td>84.64</td><td>+0.32</td></tr><tr><td>32-128K</td><td>601</td><td>2,404</td><td>45.28</td><td>44.13</td><td>-1.15</td><td>44.64</td><td>-0.64</td></tr><tr><td>128-256K</td><td>282</td><td>1,128</td><td>19.65</td><td>17.93</td><td>-1.72</td><td>18.39</td><td>-1.26</td></tr><tr><td>256-512K</td><td>316</td><td>1,264</td><td>14.87</td><td>13.17</td><td>-1.70</td><td>13.17</td><td>-1.70</td></tr><tr><td>512-768K</td><td>136</td><td>544</td><td>11.09</td><td>10.70</td><td>-0.39</td><td>10.60</td><td>-0.50</td></tr><tr><td>768K-1M</td><td>153</td><td>612</td><td>9.59</td><td>10.13</td><td>+0.54</td><td>10.33</td><td>+0.74</td></tr><tr><td>≥ 1M</td><td>49</td><td>196</td><td>8.59</td><td>8.80</td><td>+0.21</td><td>9.18</td><td>+0.59</td></tr><tr><td rowspan="7">LongBench v2</td><td>&lt; 32K</td><td>113</td><td>452</td><td>62.17</td><td>62.39</td><td>+0.22</td><td>63.50</td><td>+1.33</td></tr><tr><td>32-128K</td><td>188</td><td>752</td><td>61.04</td><td>61.70</td><td>+0.66</td><td>63.43</td><td>+2.39</td></tr><tr><td>128-256K</td><td>100</td><td>400</td><td>63.75</td><td>65.50</td><td>+1.75</td><td>64.00</td><td>+0.25</td></tr><tr><td>256–512K</td><td>50</td><td>200</td><td>63.50</td><td>65.50</td><td>+2.00</td><td>68.50</td><td>+5.00</td></tr><tr><td>512-768K</td><td>12</td><td>48</td><td>52.08</td><td>43.75</td><td>-8.33</td><td>47.92</td><td>-4.17</td></tr><tr><td>768K-1M</td><td>7</td><td>28</td><td>64.29</td><td>67.86</td><td>+3.57</td><td>67.86</td><td>+3.57</td></tr><tr><td>≥ 1M</td><td>33</td><td>132</td><td>53.03</td><td>42.42</td><td>-10.61</td><td>43.94</td><td>-9.09</td></tr><tr><td rowspan="8">GraphWalks</td><td>&lt; 32K</td><td>450</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>32-128K</td><td>200</td><td>1,800</td><td>92.23</td><td>91.70</td><td>-0.53</td><td>91.76</td><td>-0.47</td></tr><tr><td>128-256K</td><td>100</td><td>800 400</td><td>49.60</td><td>45.39</td><td>-4.21</td><td>50.31</td><td>+0.71</td></tr><tr><td>256–512K</td><td>200</td><td></td><td>37.48</td><td>30.31</td><td>-7.18</td><td>33.87</td><td>-3.61</td></tr><tr><td>512-768K</td><td>0</td><td>800 0</td><td>24.22</td><td>16.99</td><td>-7.24</td><td>22.43</td><td>-1.79</td></tr><tr><td>768K-1M</td><td>0</td><td>0</td><td>一</td><td>1 一</td><td>一 一</td><td>一</td><td>一</td></tr><tr><td>≥ 1M</td><td>200</td><td>800</td><td>7.08</td><td>2.70</td><td>-4.38</td><td>2.46</td><td>一</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-4.62</td></tr></table>

## A.3 RULER CONTROLS

Table 5 reports the scores underlying Figure 3 and additional budget and sharing controls. Inkling-Small’s dense and six sparse 128K runs use identical questions and reference answers across three budgets at $G = 1 6$ and $\bar { G } = 3 2$

## A.4 FULL CAPABILITY RESULTS

Nano and Ultra use $k _ { \mathrm { c h u n k s } } = 1 , 0 2 4$ for AA-LCR and Terminal-Bench Hard, whose dense-reference 99th-percentile input lengths are approximately 123K and 75K. SWE-bench Verified, Terminal-Bench 2.0, and GDPval use $k _ { \mathrm { c h u n k s } } = 2 , 0 4 8$ . Inkling-Small’s separate AA-LCR budget and sharing sweep is included in Appendix B.1.

Terminal-Bench and cache precision. The Terminal-Bench runs use context summarization; their scores do not establish unchanged compaction frequency or fidelity. All NVFP4 rows use BF16 KV state. Ultra’s BF16 dense Terminal-Bench references use FP8 KV caches, so these comparisons change both sparsity and KV precision.

GDPval scoring. Each rollout is compared with Kimi-K2.5-Thinking (vadams r2) using four Gemini 3.1 Pro Preview judgments with alternating response positions. A rollout missing any of the four valid judgments at the reporting cutoff is assigned four losses. Failure-adjusted Elo therefore includes all 440 rollouts and 1,760 comparison slots. Complete judgments are available for 427/440 dense and 418/440 CISA rollouts in BF16, and 420/440 dense and 430/440 CISA rollouts in NVFP4. Restricting the calculation to the 420 NVFP4 rollouts with complete judgments for both methods gives Elo 1419.57 for dense and 1421.94 for CISA (+2.38, calculated before rounding). NVFP4 verbosity counts generated tokens across all model calls in one retained trajectory per rollout for all 440 rollouts, excluding judge calls and superseded generation attempts.

Table 5: RULER accuracy across context lengths, recall budgets, and query-sharing groups. (a) Nano and Ultra BF16 controls at $G = 4 ; \Delta$ is CISA minus dense. (b) Inkling-Small NVFP4 results at $G = 1 6$ and $G = 3 2 ; \Delta _ { 1 6 - 3 2 }$ is the score at $G = 1 6$ minus that at $G = 3 2$ . K counts selected remote 16-token chunks; remote compression excludes the recent window and current tail. Scores are percentages and differences are percentage points, calculated before rounding. Daggers identify the $G = 1 6$ settings plotted in Figure $_ { 3 ; }$ dashes denote unavailable results. The Nano 1M Quest-bound controls use the same serving build, budget, window, and page size at $G = 1$ and $4 ;$ the CISA reference uses a different build.  
(a) Nano and Ultra BF16: matched budgets, $G = 4$
<table><tr><td>Model</td><td>Context</td><td>k (chunks)</td><td>Remote comp.</td><td>Dense</td><td>CISA</td><td>∆</td></tr><tr><td>Nano</td><td>128K</td><td>1,024</td><td>8×</td><td>94.61</td><td>92.93</td><td>-1.68</td></tr><tr><td>Nano</td><td>256K</td><td>2,048</td><td>8×</td><td>92.75</td><td>90.70</td><td>-2.05</td></tr><tr><td>Nano</td><td>512K</td><td>4,096</td><td>8×</td><td>91.37</td><td>89.52</td><td>-1.85</td></tr><tr><td>Nano</td><td>1M</td><td>4,096</td><td>16×</td><td>86.60</td><td>85.87</td><td>-0.74</td></tr><tr><td colspan="5">Quest bound + CISA index, Nano 1M</td><td> $G = 1 \colon 8 4 . 9 5 ; G = 4 \colon 8 4 . 4 1$ </td><td></td></tr><tr><td>Ultra</td><td>128K</td><td>1,024</td><td>8×</td><td>97.69</td><td>97.31</td><td>-0.38</td></tr><tr><td>Ultra</td><td>256K</td><td>2,048</td><td>8×</td><td>96.15</td><td>95.31</td><td>-0.85</td></tr><tr><td>Ultra</td><td>512K</td><td>4,096</td><td>8×</td><td>95.92</td><td>94.31</td><td>-1.62</td></tr><tr><td>Ultra</td><td>1M</td><td>4,096</td><td>16×</td><td>94.76</td><td>93.70</td><td>-1.07</td></tr></table>

<table><tr><td colspan="7">(b) Inkling-Small NVFP4: budget and sharing controls</td></tr><tr><td>Context</td><td>K (chunks)</td><td>Remote comp.</td><td>Dense</td><td> $G = 1 6$ </td><td> $G = 3 2$ </td><td> $\Delta _ { 1 6 - 3 2 }$ </td></tr><tr><td rowspan="3">128K</td><td>1,024</td><td>8×</td><td rowspan="3">92.48</td><td> $8 9 . 5 5 ^ { \dagger }$ </td><td>86.83</td><td>+2.71</td></tr><tr><td>2,048</td><td>4×</td><td>91.71</td><td>90.16</td><td>+1.55</td></tr><tr><td>4,096</td><td>2×</td><td>92.99</td><td>92.22</td><td>+0.77</td></tr><tr><td rowspan="2">256K</td><td>2,048</td><td>8×</td><td rowspan="2">89.13</td><td>86.18†</td><td>86.53</td><td>-0.35</td></tr><tr><td>4,096</td><td>4×</td><td></td><td>87.71</td><td></td></tr><tr><td rowspan="2">512K</td><td>2,048</td><td>16×</td><td rowspan="2">80.85</td><td></td><td>81.27</td><td></td></tr><tr><td>4,096</td><td>8×</td><td> $8 2 . 7 8 ^ { \dagger }$ </td><td>81.33</td><td>+1.45</td></tr><tr><td rowspan="2">1M</td><td>2,048</td><td>32×</td><td rowspan="2">73.81</td><td>76.94</td><td>74.53</td><td>+2.41</td></tr><tr><td>4,096</td><td>16×</td><td>78.52†</td><td>76.03</td><td>+2.49</td></tr></table>

Table 6: Nano and Ultra capability results at fixed chunk budgets. Scores are percentages except GDPval, for which we report failure-adjusted Elo. Verbosity $\Delta \mathrm { i } \mathrm { \bar { s } 1 0 0 } ( \bar { L } _ { C I S A } / \bar { L } _ { \mathrm { d e n s e } } ^ { \mathrm { ~ \scriptsize ~ - ~ } } - 1 \bar { ) }$ , where $\bar { L }$ is mean generated tokens per rollout on the matched cohort. For multi-turn agents, rollout length sums generated tokens across all model calls. <sup>2</sup> Unavailable results reflect incomplete evaluations or unrecoverable token counts for the matched cohort. <sup>3</sup> GDPval includes all 220 tasks and two rollouts per task under the failure-adjustment rule below.
<table><tr><td>Scale</td><td>Precision</td><td>Benchmark</td><td>Metric</td><td>k</td><td>Dense</td><td>CISA</td><td> $\Delta$ </td><td>Verbosity  $\Delta$ </td></tr><tr><td>Nano</td><td>BF16</td><td>AA-LCR</td><td>pass@1 (%)</td><td>1,024</td><td>34.94</td><td>34.25</td><td>-0.69</td><td>-9.1%</td></tr><tr><td>Ultra</td><td>BF16</td><td>AA-LCR</td><td>pass@1 (%)</td><td>1,024</td><td>65.13</td><td>65.38</td><td>+0.25</td><td>+5.3%</td></tr><tr><td>Ultra</td><td>NVFP4</td><td>AA-LCR</td><td>pass@1 (%)</td><td>1,024</td><td>65.00</td><td>65.88</td><td>+0.88</td><td>+1.8%</td></tr><tr><td>Ultra</td><td>BF16</td><td>SWE-bench Verified</td><td>resolved (%)</td><td>2,048</td><td>70.00</td><td>69.07</td><td>-0.93</td><td>-4.2%</td></tr><tr><td>Ultra</td><td>NVFP4</td><td>SWE-bench Verified</td><td>resolved (%)</td><td>2,048</td><td>69.20</td><td>69.93</td><td>+0.73</td><td>-0.2%</td></tr><tr><td>Ultra Ultra</td><td>BF16</td><td>Terminal-Bench 2.0</td><td>pass@1 (%)</td><td>2,048</td><td>49.86</td><td>49.30</td><td>-0.56</td><td>-19.1%</td></tr><tr><td></td><td>NVFP4</td><td>Terminal-Bench 2.0</td><td>pass@1 (%)</td><td>2,048</td><td>48.03</td><td>49.44</td><td>+1.40</td><td>-0.7%</td></tr><tr><td>Ultra Ultra</td><td>BF16</td><td>Terminal-Bench Hard</td><td>pass@1 (%)</td><td>1,024</td><td>37.24</td><td>36.98</td><td>-0.26</td><td>-6.9%</td></tr><tr><td></td><td>NVFP4</td><td>Terminal-Bench Hard</td><td>pass@1 (%)</td><td>1,024</td><td>35.94</td><td>35.42</td><td>-0.52</td><td>+2.8%</td></tr><tr><td>Ultra</td><td>BF16</td><td> $\mathrm { G D P v a l } ^ { 3 }$ </td><td>Adj. Elo</td><td>2,048</td><td>1422.62</td><td>1414.72</td><td>-7.90</td><td>_2</td></tr><tr><td>Ultra</td><td>NVFP4</td><td> $\mathrm { G D P v a l } ^ { 3 }$ </td><td>Adj. Elo</td><td>2,048</td><td>1395.59</td><td>1408.71</td><td>+13.12</td><td>-2.8%</td></tr></table>

## B OPERATING-POINT ABLATIONS

We detail the quality–throughput profiles in Section 4.4, then report no-sharing baselines and controls for the recent window and query sharing.

## B.1 QUALITY AND GENERATION-COST SENSITIVITY

Figure 4 combines benchmark quality, controlled serving throughput, and generation length. For profile p, let $A _ { p , b }$ be benchmark b’s percentage score and $\bar { L } _ { p , b }$ its mean generated tokens per response. For B = {RULER 1M, MRCR200, GraphWalks-200, AA-LCR}, define

$$
A _ { p } = \frac { 1 } { 4 } \sum _ { b \in \mathcal { B } } A _ { p , b } , \qquad R _ { p } = \frac { 1 } { 4 } \sum _ { b \in \mathcal { B } } \frac { \bar { L } _ { p , b } } { \bar { L } _ { \mathrm { d e n s e } , b } } , \qquad S _ { \mathrm { a d j } , p } = \frac { S _ { p } } { R _ { p } } ,\tag{10}
$$

where $S _ { p }$ is total-token throughput relative to the same model’s dense reference. Every profile includes all four benchmarks, weighted equally. Output-length ratios are computed from each benchmark’s mean, including reasoning tokens and outputs that reach the generation cap. The base profile uses chunk budgets {4096, 4096, 4096, 1024} in the order above; the reduced profile halves each budget. Each profile uses one sharing factor G throughout, with a 128-page recent window and the current-page tail in addition to the selected chunks.

Evaluation subsets. Both models use identical tasks and reference answers. RULER uses its 1M-context evaluation, with 100 examples in each of 13 subtasks. MRCR200 contains the longest 200 of the 2,400 MRCR prompts, selected before model-specific tokenization using source lengths of 763,613–1,118,929 tokens. GraphWalks-200 contains the 200 prompts from Ultra’s 256K–512K bucket, chosen to examine budget sensitivity; their framed lengths are approximately 360K tokens for Ultra and 255K for Inkling-Small. MRCR200 and GraphWalks-200 each use four repeats per task; AA-LCR uses 100 tasks with 16 repeats. For each benchmark, quality and output length use the same tasks and repeats.

Throughput measurements. Each model serves the same ordered MRCR200 requests across its configurations, generating exactly 4,096 tokens per request with EOS ignored. Ultra uses four B300 GPUs in TP2DP2 with concurrency 32 and temperature zero; each point averages three independent allocations with fresh servers. The dense and base-G = 4 runs share one campaign, while reduced-G = 4 and base-G = 8 use separate campaigns under the same workload and allocation protocol. The G = 8 campaign uses a newer serving build. Inkling-Small uses eight B300 GPUs in TP8DP1, with concurrency and maximum sequence count 16, a maximum of 32,768 batched tokens, and one run per setting. Throughput is the client-reported total-token rate from vllm bench serve, checked against token totals and elapsed time. Speedups are normalized within each model; the different topologies do not support a controlled efficiency comparison between models.

Interpretation and uncertainty. Dividing by $R _ { p }$ assumes cost is proportional to output length, omitting prefill cost, nonlinear batching effects, and workload differences. The adjusted value is therefore a sensitivity proxy rather than measured natural-completion speedup. Quality and throughput are measured separately, sometimes with different serving builds. Ultra’s dense RULER reference also uses an earlier build and topology, with task and reference identities matched to the sparse runs. Inkling-Small’s quality and throughput runs use different builds.

Quality intervals use 20,000 task-bootstrap resamples, stratified within RULER’s 13 subtasks and retaining all repeats of each sampled task for the other benchmarks. Each resample recomputes the four-benchmark mean, using seed 20260901 for Ultra and 20260912 for Inkling-Small. Intervals are pointwise, and Inkling-Small’s single throughput run per setting provides no estimate of variation across runs. Small differences between plotted points do not establish statistically resolved dominance. Table 7 reports both speedup coordinates and the quality intervals; Table 8 gives Ultra’s component scores.

![](images/b3fea64b7ad7a6ce0b82003a46611638b122930ca8b3ba965a37e6431814cec4.jpg)  
Measured fixed-output end-to-end speedup  
Figure 7: Unadjusted quality–throughput results for Ultra and Inkling-Small NVFP4. Quality scores and profiles match Figure 4; the horizontal axis shows measured total-token throughput relative to each model’s dense baseline, using MRCR200 with 4,096 output tokens per request. Lines connect increasing G within each budget profile. Table 7 reports the exact coordinates.

Table 7: Operating points for Figures 4 and 7. Mean is the equally weighted four-benchmark score (%) with its pointwise 95% task-bootstrap interval. $S _ { p }$ is measured fixed-output throughput relative to dense; $\Delta L \stackrel { - } { = } 1 0 0 ( R _ { p } - 1 )$ is the mean benchmark-relative output-length change. $S _ { \mathrm { a d j } , p } = S _ { p } / R _ { p }$ is the generation-cost sensitivity proxy. Base budgets are {4096, 4096, 4096, 1024} chunks for RULER 1M, MRCR200, GraphWalks-200, and AA-LCR, respectively; reduced budgets halve these values.
<table><tr><td>Profile</td><td>G</td><td>Mean [95% CI] Measured</td><td> $S _ { p }$   $\Delta L$ </td><td>(%) Adjusted</td><td> $S _ { \mathrm { a d j } , p }$ </td></tr><tr><td colspan="6">Ultra NVFP4</td></tr><tr><td>Dense</td><td></td><td>48.04 [45.75, 50.30]</td><td>1.000×</td><td>+0.0</td><td>1.000×</td></tr><tr><td>Base</td><td>4</td><td>47.68 [45.39, 49.93]</td><td>2.285×</td><td>+17.9</td><td>1.939×</td></tr><tr><td>Reduced</td><td>4</td><td>45.40 [43.14, 47.64]</td><td>2.615×</td><td>+34.1</td><td>1.950×</td></tr><tr><td>Base</td><td>8</td><td>46.77 [44.48, 49.02]</td><td>2.594×</td><td>+13.9</td><td>2.277×</td></tr><tr><td colspan="6">Inkling-Small NVFP4</td></tr><tr><td>Dense</td><td></td><td>42.73 [40.34, 45.10]</td><td>1.000×</td><td>+0.0</td><td>1.000×</td></tr><tr><td>Reduced</td><td>4</td><td>44.95 [42.65, 47.23]</td><td>1.141×</td><td>+75.1</td><td>0.652×</td></tr><tr><td>Reduced</td><td>16</td><td>44.61 [42.28, 46.96]</td><td>1.522×</td><td>+76.6</td><td>0.862×</td></tr><tr><td>Reduced</td><td>32</td><td>44.30 [41.93, 46.66]</td><td>1.636×</td><td>+76.6</td><td>0.926×</td></tr><tr><td>Base</td><td>4</td><td>44.65 [42.34, 46.93]</td><td>0.898×</td><td>+39.7</td><td>0.643×</td></tr><tr><td>Base</td><td>16</td><td>44.38 [42.04, 46.73]</td><td>1.403×</td><td>+38.6</td><td>1.013×</td></tr><tr><td>Base</td><td>32</td><td>43.68 [41.32, 46.02]</td><td>1.544×</td><td>+37.3</td><td>1.124×</td></tr></table>

Table 8: Ultra NVFP4 benchmark scores underlying the operating-point means in Table 7. Scores use each benchmark’s native percentage metric; the mean weights all four equally.
<table><tr><td>Profile</td><td>RULER</td><td>MRCR200</td><td>GraphWalks-200</td><td>AA-LCR</td><td>Mean</td></tr><tr><td>Dense</td><td>94.53</td><td>9.21</td><td>24.22</td><td>64.19</td><td>48.04</td></tr><tr><td>G = 4, base</td><td>93.76</td><td>9.99</td><td>22.43</td><td>64.56</td><td>47.68</td></tr><tr><td>G = 4, reduced</td><td>91.88</td><td>10.48</td><td>16.99</td><td>62.25</td><td>45.40</td></tr><tr><td>G = 8, base</td><td>92.69</td><td>9.81</td><td>20.69</td><td>63.88</td><td>46.77</td></tr></table>

## B.2 NO QUERY SHARING (G = 1)

To document performance without positional query sharing, Table 9 reports Ultra NVFP4 quality and throughput separately. The $G = 1$ quality run uses $k _ { \mathrm { c h u n k s } } = 2 , 0 4 8$ and an implementation without a representative cache; its throughput run uses $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ . These measurements describe different operating points and are not combined into one quality–throughput coordinate.

Table 9: Ultra NVFP4 results with and without query sharing. (a) GraphWalks-200 node-set F1 (%) with pointwise 95% intervals from 20,000 task-bootstrap resamples (seed 20260901). (b) MRCR200 total-token throughput on four B300 GPUs in TP2DP2 with concurrency 32 and 4,096 generated tokens per request. The $G = 1$ throughput result is provisional and uses one allocation; other rows average three, with the observed range shown.
<table><tr><td colspan="4">(a) Quality</td></tr><tr><td>Setting</td><td>k (chunks)</td><td>Score</td><td>95% CI</td></tr><tr><td>Dense</td><td></td><td>24.22</td><td>[20.02, 28.64]</td></tr><tr><td> $G = 1$ </td><td>2,048</td><td>18.38</td><td>[14.67, 22.30]</td></tr><tr><td> $G = 4$ </td><td>2,048</td><td>16.99</td><td>[13.44, 20.73]</td></tr><tr><td> $G = 4$ </td><td>4,096</td><td>22.43</td><td>[18.32, 26.73]</td></tr><tr><td> $G = 8$ </td><td>4,096</td><td>20.69</td><td>[16.61, 24.90]</td></tr><tr><td colspan="4"></td></tr><tr><td></td><td></td><td>(b) Throughput</td><td></td></tr><tr><td>Setting</td><td>k (chunks)</td><td>Tokens/s</td><td>Range</td></tr><tr><td>Dense</td><td></td><td>14,677</td><td>14,518-14,823</td></tr><tr><td> $G = 1$ </td><td>4,096</td><td>18,063</td><td></td></tr><tr><td> $G = 4$ </td><td>2,048</td><td>38,375</td><td>38,083–38,767</td></tr><tr><td> $G = 4$ </td><td>4,096</td><td>33,543</td><td>33,091–34,196</td></tr><tr><td> $G = 8$ </td><td>4,096</td><td>38,075</td><td>37,656–38,434</td></tr></table>

Additional G = 1 diagnostics give an AA-LCR score of 65.4 over 15 of the planned 16 repeats and an MRCR200 score of 9.18 after recovering all saved responses; two MRCR200 servers failed their post-run health checks. Earlier RULER and MRCR200 runs from before an implementation repair are excluded. Without a validated RULER result and replicated throughput measurements, G = 1 is excluded from the four-benchmark operating-point comparison.

## B.3 RECENT-WINDOW AND QUERY-SHARING CONTROLS

Matched recent-window control. On MRCR200 with four repeats per task, removing the 128-page recent window at $G = 4 , k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ changes the score from 9.99 to 9.64. The paired difference is −0.34 percentage points before rounding, with a 95% task-bootstrap interval of [−1.44, 0.71] (20,000 resamples, seed 20260901). This experiment does not resolve an effect of the recent window on MRCR200 quality.

Held-out routing diagnostics. Figure 8 probes the two routing mechanisms directly on held-out activations from all 12 Ultra attention layers and both KV groups. At G = 4, QShare retains, on aver age, 92.7% of the pages selected independently at each position while losing 0.35 percentage points of total dense-attention probability mass. The overlap between independent selections, measured as their intersection divided by the smaller set, is 88.3% at a one-token lag and 87.4% at lag 32. For the recent window, adding $\dot { W } = 1 2 8$ pages to $k _ { \mathrm { c h u n k s } } = 1 , 0 2 4$ increases captured dense-attention mass by 1.08 percentage points, whereas replacing 128 selected historical pages at the same total page count adds 0.07 points. In this diagnostic, most of the gain from adding the recent window is attributable to reading more pages.

![](images/911c7db24229935350891dbcba1dcb51126d5352214af5c60fe3e1777e64a845.jpg)

![](images/f9ba1824a62cb44966b6772bde2b0398df443b3ef8ad9161c0d43d2a4f90a67b.jpg)  
Figure 8: Offline routing diagnostics on 64 held-out 131K-token Ultra documents $( k _ { \mathrm { c h u n k s } } = 1 , 0 2 4$ 16-token pages). (a) Dense-attention-mass loss when one selection with $W = 1 2 8$ is shared across G adjacent queries over the final 2,048 positions. (b) Mass gain relative to $W = 0$ when recent pages are added to $k _ { \mathrm { c h u n k s } }$ selected pages or replace the same number at a fixed total, sampling within the final 8,192-position region. Whiskers are pointwise 95% document-bootstrap intervals (2,000 draws, seed 42); window differences are paired with $W = 0$ . These are mechanism diagnostics, not benchmark-quality or throughput measurements.

## C TRAFFIC DERIVATION

We derive the traffic ratios in Section 3, omitting the recent window, current-page tail, and execution overheads.

For one KV group and one query with history $n ,$ dense attention reads 2nd K/V elements. After the sparse threshold $\dot { S } = k _ { \mathrm { c h u n k s } } \dot { C }$ , CISA reads 2Sd selected K/V elements and $( n / C ) d _ { I } / G$ amortized representative elements. Dividing by the dense read at $n = N$ gives Equation 8.

For a complete causal prefill, dense K/V traffic is proportional to $\begin{array} { r } { \int _ { 0 } ^ { N } 2 n d d n = N ^ { 2 } d . } \end{array}$ . CISA attends densely through S and then reads 2Sd selected K/V elements per query, giving $S ^ { 2 } d + 2 S ( N - S ) d .$ The representative scan is active only over the indexed-history region and integrates to $( N ^ { 2 } -$ $S ^ { 2 } ) d _ { I } \bar { / } ( 2 C G )$ . Normalizing by dense traffic and substituting $\alpha = S / N$ gives Equation 9. An additive recent window and current-page tail introuduce additional traffic that is omitted from these expressions. The inverse ratios are therefore bandwidth-only estimates, not strict bounds on measured speedup.

The representative cache has the following capacity relative to the main K+V state:

$$
\frac { ( N / C ) d _ { I } } { 2 N d } = \frac { d _ { I } } { 2 C d } .\tag{11}
$$

For $C = 1 6$ and $d _ { I } = d ,$ , this is $1 / 3 2$ . This capacity is local to the KV group and follows its tensor-parallel ownership.

## D ATTENTION-ONLY SPEEDUP MEASUREMENTS

These one-layer experiments measure the attention-path costs discussed in Section 3.3 for Nano, Ultra, and Inkling-Small.

## D.1 NANO AND ULTRA HEAD GEOMETRIES

We compare dense and sparse one-layer Nemotron-H models with dummy weights on one B300 GPU at TP1. All runs use CUDA graphs, BF16 model state and K/V, head dimension 128, 32 query heads, and a maximum of 8,192 batched tokens. Two KV heads reproduce Nano’s TP1 attention geometry; one KV head with 32 query heads reproduces Ultra’s local geometry at TP2, excluding inter-GPU communication. CISA uses 16-token chunks, $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6$ , a 128-page recent window, and $G = 4$ for prefill; single-token decode has effective $G = 1$

Table 10: CISA speedup over matched dense GQA at one million tokens (higher is better). The geometric mean is across the three executed batch sizes. GPU busy is the union of kernel, memcpy, and memset intervals, with overlaps counted once. Core attention includes only the FlashAttention forward and combine kernels; GPU busy also includes projections, scoring, selection, metadata, and copies.
<table><tr><td>Local model shape</td><td>Phase</td><td>Metric</td><td> $B = 1 6$ </td><td> $B = 2 4$ </td><td> $B = 3 2$ </td><td>Geom. mean</td></tr><tr><td rowspan="3">Nano TP1  $( H _ { \mathrm { K V } } = 2 )$ </td><td>Terminal prefill</td><td>GPU busy Core attention</td><td>4.49× 5.95×</td><td>4.57× 6.40×</td><td>4.56× 6.07×</td><td>4.54× 6.13×</td></tr><tr><td>Steady decode</td><td>GPU busy Core attention</td><td>2.72× 12.37×</td><td>3.24×</td><td>3.39×</td><td>3.11×</td></tr><tr><td>Terminal prefill</td><td>GPU busy Core attention</td><td>8.61×</td><td>13.18× 8.09×</td><td>12.94× 8.61×</td><td>12.83× 8.43×</td></tr><tr><td rowspan="2">Ultra TP2/rank  $( H _ { \mathrm { K V } } = 1 )$ </td><td></td><td>GPU busy</td><td>11.63× 2.56×</td><td>11.57× 2.96×</td><td>11.70× 2.98×</td><td>11.63× 2.83×</td></tr><tr><td>Steady decode</td><td>Core attention</td><td>13.26×</td><td>13.79×</td><td>12.99×</td><td>13.34×</td></tr></table>

For batch sizes $B \in \{ 1 6 , 2 4 , 3 2 \}$ , we retain ten terminal-prefill or steady-decode forwards near $N = 1 { , } 0 4 8 { , } 5 7 6$ . Profiler annotations verify the phase, batch size, and per-request history. Table 10 reports speedups over dense GQA for each batch size and their geometric mean.

At this context length, the selected K/V span is $k _ { \mathrm { c h u n k s } } C = 6 5 , 5 3 6 = N / 1 6 .$ The ideal K/Vonly reduction is therefore 16×, or 14.22× after accounting for the representative scan at $G = 4$ (Equation 8). Ultra’s geometric-mean terminal-prefill speedup is 11.63× for core attention and 8.43× for all GPU activity. The remaining costs include the recent window, current-page tail, routing, metadata, projections, and copies. Nano’s lower prefill speedup reflects an implementation that processes its two local KV groups serially on one CUDA stream, while dense attention uses a single kernel.

During decode, the representative scan cannot be shared across positions, giving an ideal combined traffic reduction of $1 0 . 6 7 \times \mathrm { a t } G = 1$ . Core-attention speedups are 12.83× for Nano and 13.34× for Ultra, but including all GPU activity reduces them to 3.11× and 2.83×. These gaps illustrate the cost of work outside the core attention kernel.

At B = 24, the 8,192-token batching cap gives 341 prefill tokens per request and a final one-token tail. Dense and sparse runs execute the same shapes, including this irregular tail.

## D.2 INKLING-SMALL COMPONENT BREAKDOWN

Table 11 gives the timings behind Figure 2, including the decode implementation before coalescing representative loads. The prefill sweep varies query sharing at a fixed chunk budget; the decode optimization preserves the scoring rule and selection policy.

Model and hardware. We use one B300 GPU at TP1 with a one-layer dummy-weight model containing global attention, a dense MLP, and short convolution. Four query heads and one KV head reproduce Inkling-Small’s local attention geometry at TP8. Hidden and MLP intermediate dimensions are 4,096, convolution width is 4, and attention and representative dimensions are 128. The model omits MoE, sliding-window layers, and TP8 communication. Model state, activations, representatives, and K/V are BF16; execution uses FlashAttention 4, CUDA graphs, and the baseline top-k implementation.

Workload and measurements. The workload comprises sixteen 1,048,576-token prompts, with a maximum of 8,192 batched tokens and a 64 GiB KV reservation. Sparse runs use 16-token chunks, $k _ { \mathrm { c h u n k s } } = 4 { , } 0 9 6 ,$ a 128-page recent window, and the current-page tail. Prefill varies $G \in \{ 4 , 1 6 , 3 2 \}$ decode uses the $G = 3 2$ configuration with one query per sequence, giving effective positional sharing $G = 1$ . Each mean uses ten forwards from one capture, rather than independent trials. Terminal-prefill forwards contain 512 queries per sequence at histories 1,043,968–1,048,576 in steps of $5 1 2 ;$ decode histories span 1,048,694–1,048,703. The baseline and coalesced-load decode implementations are measured on the same B300; the dense decode reference is captured on another B300.

Table 11: Inkling-Small attention-path timings for Figure 2. Values are mean milliseconds over ten forwards from one capture. GPU busy counts overlapping kernel and memory-operation intervals once and excludes idle gaps. Component sums may differ from totals because of rounding.
<table><tr><td>Phase</td><td>Variant</td><td>Attention</td><td>Scoring</td><td>Top-k</td><td>Setup</td><td>Other</td><td>GPU busy</td></tr><tr><td>Terminal prefill</td><td>Dense</td><td>14.176</td><td>0.000</td><td>0.000</td><td>0.018</td><td>0.912</td><td>15.106</td></tr><tr><td>Terminal prefill</td><td>CISA G4</td><td>9.154</td><td>1.174</td><td>1.004</td><td>0.152</td><td>0.972</td><td>12.456</td></tr><tr><td>Terminal prefill</td><td>CISA G16</td><td>2.626</td><td>0.589</td><td>0.325</td><td>0.069</td><td>0.968</td><td>4.577</td></tr><tr><td>Terminal prefill</td><td>CISA G32</td><td>1.198</td><td>0.587</td><td>0.147</td><td>0.060</td><td>0.962</td><td>2.954</td></tr><tr><td>Steady decode</td><td>Dense</td><td>1.218</td><td>0.000</td><td>0.000</td><td>0.019</td><td>0.092</td><td>1.329</td></tr><tr><td>Steady decode</td><td>CISA</td><td>0.104</td><td>0.536</td><td>0.118</td><td>0.047</td><td>0.112</td><td>0.918</td></tr><tr><td>Steady decode</td><td>CISA + coalesced loads</td><td>0.104</td><td>0.057</td><td>0.118</td><td>0.045</td><td>0.111</td><td>0.436</td></tr></table>

Components. Attention contains core attention kernels; Scoring contains representative scoring; Top-k contains selection. Setup includes scorer-plan construction, query sharing, representative maintenance, page construction, and block-table gather. Other contains the remaining kernels and copies, with overlaps counted once.