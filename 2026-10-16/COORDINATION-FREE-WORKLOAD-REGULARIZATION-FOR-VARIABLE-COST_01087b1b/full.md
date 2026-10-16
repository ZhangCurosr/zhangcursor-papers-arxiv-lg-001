# COORDINATION-FREE WORKLOAD REGULARIZATION FOR VARIABLE-COST DISTRIBUTED TRAINING

Tao Huang <sup>1</sup> <sup>†</sup> Chen Zhu <sup>1</sup> Antonio Alonso <sup>1</sup> Daming Li <sup>1</sup> Adarsh Goura <sup>1</sup> <sup>2</sup> <sup>∗</sup> Jaideep Ray <sup>1</sup> Yitong Zhou <sup>1</sup>

<sup>1</sup> LinkedIn Corporation, Sunnyvale, CA, USA

<sup>2</sup> The University of Texas at Dallas, Richardson, TX, USA

## ABSTRACT

Synchronous data-parallel training usually assigns every rank the same number of records, even when record costs vary by orders of magnitude. For variable-length models, this converts input variation into idle time because each step waits for its most expensive rank-local batch. We present Batch Cost Lookahead (BCL), a coordination-free data-loading policy that buffers a bounded rank-local window and repartitions it into fixed-cardinality packs with balanced estimated cost. BCL preserves the trainer’s record-count contract and requires neither cross-rank metadata exchange nor runtime modification. In simulation over public workload distributions, enabling BCL reduces per-rank estimated batch-cost variability by up to 82.8%. Across 13 NeMo-RL SFT corpora on eight H200 GPUs, rank-local BCL runs 1.050–1.167× faster than pooled MFFD. Compiled H200 measurements on an internal sequence workload show 1.197–1.343× speedup over packed execution from one to four nodes. We also identify a memory-feasibility boundary: a p99 cap for high-variance or OOM-prone data bounds outliers before BCL balances the rest. These results show that substantial variable-cost imbalance can be removed at the data-loader boundary, while clarifying when data preparation or broader coordination is still necessary.

## 1 INTRODUCTION

In synchronous data-parallel training, each rank processes a local batch, computes gradients, and participates in collective synchronization (Li et al., 2020; Goyal et al., 2017). Workload imbalance is pronounced for variable-length sequences. The cost of token-wise operations grows with sequence length, while dense self-attention includes a quadratic sequence-length component (Vaswani et al., 2017; Dao et al., 2022). Consequently, two batches with the same number of records can have very different costs when one contains longer sequences. In synchronous training, this imbalance lies on the critical path because each step waits for the slowest rank (Chen et al., 2016). The resulting idle time tends to grow with workload variability and the number of participating ranks (Dean & Barroso, 2013; Chen et al., 2016).

Sequence packing and variable-length kernels reduce padding work within a batch (Krell et al., 2021; Dao et al., 2022), but they do not determine how records should be grouped across rank-local batches in a synchronized step. Cross-worker balancing can use global metadata to make that decision, but recent designs require additional communication or runtime coordination (Wang et al., 2026; Wan et al., 2026). This paper asks a narrower systems question:

How much variable-cost imbalance can be removed using only bounded information already available to each rank-local data loader?

We study Batch Cost Lookahead (BCL), a local workloadregularization policy. A worker buffers K future fixedcardinality batches, estimates a cost for each record, and partitions the KB records into K packs of exactly B records. The objective is to reduce the maximum pack cost while preserving the existing fixed-cardinality interface. Each input reader schedules independently, without exchanging record costs with other readers or changing the model and distributed collectives.

Our production trainer expects a fixed record count, so BCL retains that interface. Token-budget batching can improve utilization, but varying the number of records per step requires careful loss normalization (Ott et al., 2018; Krell et al., 2021). BCL regroups the same records within a bounded window while emitting exactly B records per pack. This makes BCL complementary to padding elimination and packed execution: packing removes wasted work within a pack, while BCL reduces cost dispersion among packs that must synchronize. Fixed-token packing can also leave unequal attention work across ranks (Wang et al., 2026);

globally coordinated assignment offers another response to this skew (Li et al., 2026).

Our evaluation separates three forms of evidence. First, a self-contained simulation over six checked-in length arrays derived from public datasets isolates the scheduling effect. With batch size 128, eight ranks, and a quadratic cappedlength cost, increasing lookahead from $K = 1$ to $K = 3 2$ reduces per-rank estimated batch-cost variability by up to 82.8%. Second, NeMo-RL SFT runs (NVIDIA, n.d.) compare independent per-rank input with pooled MFFD. At $B = 8$ on eight H200 GPUs, BCL improves wall-clock step time by 1.050–1.167× across 13 corpora; this includes both scheduling and input-path differences. Third, compiled H200 measurements on an internal Feed sequential-ranking workload whose model setting is described publicly by Hertel et al. (2026) compare BCL at K = 32 with packed execution from one to four nodes. The measured speedup ranges from 1.197× to 1.343×.

BCL cannot split an overlong record, so we set the length policy before comparing packers. Earlier feasibility probes found five corpora that passed uncapped smoke tests, four that used p99 proactively for high length variability, and four for which p99 enabled runs that had OOMed uncapped. Every NeMo comparison arm uses the same per-corpus cap, separating the effect of grouping from changes in truncation.

This paper makes four contributions:

• We formulate a coordination-free operating point for variable-cost balancing: rank-local scheduling under a fixed-cardinality and bounded-reordering data-loader contract.

• We implement fixed-cardinality balancing within bounded reader windows.

• We evaluate BCL in a public-distribution simulation, NeMo-RL SFT on 13 corpora, and internal H200 runs on one to four nodes.

• We identify regimes in which local scheduling is insufficient and length capping or broader coordination is required.

Paper organization. Section 2 motivates the cost model and distinguishes cross-rank imbalance from padding waste. Section 3 describes BCL and its fixed-cardinality implementation. Section 4 develops the reducible and irreducible regimes. Sections 5–7 present the public-distribution, NeMo, and multi-node internal evaluations; each introduces its workload, setup, reporting provenance, and results together. Section 8 situates BCL against related work, Section 9 summarizes deployment boundaries, and Section 10 concludes.

## 2 MOTIVATION AND COST MODEL

## 2.1 Fixed cardinality is not fixed work

Consider synchronous data-parallel training with R ranks and a prescribed per-rank batch cardinality B. At step t, every rank processes exactly B records before participating in the same synchronization barrier. This fixed-count contract controls how many records contribute to an optimizer step, but it does not control how much computation those records require.

For variable-length data, the record lengths assigned to a rank determine its workload. Two ranks can therefore receive the same number of records while one processes substantially more tokens or more attention work. Because the synchronized step waits for the slowest rank, this cost difference becomes idle time on the faster ranks (Li et al., 2020; Chen et al., 2016). The problem addressed in this paper is not unequal record count; it is unequal work under an intentionally equal-record-count interface. This interface describes our production streaming use case; token-budget and packed batches with variable record counts can exhibit the same synchronization imbalance. The formal metrics here use fixed B; Section 6 compares token-budget packing using measured wall-clock time.

## 2.2 Padding waste and synchronization imbalance

Padding and cross-rank imbalance are separate inefficiencies. In a padded batch, shorter sequences execute against the longest sequence’s shape. Sequence packing and variable-length kernels reduce this intra-batch waste by avoiding computation on padding (Krell et al., 2021; Dao et al., 2022). They do not, however, guarantee that the useful work assigned to each rank is equal. A packed rank containing several long records can still take longer than another packed rank containing the same number of shorter records.

BCL targets this residual synchronization imbalance. It is complementary to packed execution: packing changes how the records inside a pack are represented, whereas BCL changes which records are grouped into packs that will execute at the same synchronization boundary. The evaluation therefore reports packed baselines separately rather than attributing padding removal to BCL.

## 2.3 Per-record and per-pack cost

Let ℓ denote a record’s sequence length and let $\ell _ { \mathrm { m a x } }$ be an optional feasibility cap. We use the effective length $\bar { \ell } = \operatorname* { m i n } ( \ell , \ell _ { \mathrm { m a x } } )$ and the first-order proxy

$$
c ( \ell ) = \alpha \bar { \ell } + \beta \bar { \ell } ^ { 2 } ,\tag{1}
$$

where $\alpha , \beta \ge 0$ . The linear term represents token-wise work such as projections and feed-forward layers, while the quadratic term represents dense self-attention (Vaswani et al., 2017; Dao et al., 2022). Setting $( \alpha , \beta ) = ( 1 , 0 )$ gives the linear, or L1, mode; (0, 1) gives the quadratic, or L2, mode; and nonzero coefficients give a coefficient-weighted blend.

For rank r at step t, the modeled pack cost is additive:

$$
C _ { r , t } = \sum _ { i = 1 } ^ { B } c ( \ell _ { r , t , i } ) .\tag{2}
$$

Only relative costs matter to the scheduler, so multiplying every cost by the same positive constant does not change an assignment or a normalized balance metric. The cap and coefficients are workload policy, not properties inferred by BCL. The public-distribution simulation uses capped L2 cost to isolate the scheduling effect; each measured evaluation states its length policy and controls alongside its results.

The cost estimate in Equation 2 guides how BCL groups records, but GPU step time also depends on kernels, communication, data loading, and compilation. We measure step time on hardware rather than treating improved estimated balance as a speedup.

## 2.4 Critical-path metrics

The modeled critical-path cost of synchronized step t is

$$
C _ { t } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { 1 \leq r \leq R } C _ { r , t } .\tag{3}
$$

We report two complementary normalized metrics. Packcost coefficient of variation measures the relative temporal spread of emitted pack costs:

$$
\mathrm { C V } _ { \mathrm { p a c k } } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \frac { \mathrm { s t d } _ { t } ( C _ { r , t } ) } { \mathrm { m e a n } _ { t } ( C _ { r , t } ) } .\tag{4}
$$

A lower value means each rank receives more consistent modeled work over time. The synchronization straggler ratio measures how much modeled work lies on the crossrank critical path:

$$
\rho = \frac { \mathrm { m e a n } _ { t } ( C _ { t } ^ { \operatorname* { m a x } } ) } { \mathrm { m e a n } _ { r , t } ( C _ { r , t } ) } .\tag{5}
$$

Here $\rho = 1$ is perfect modeled balance, while $\rho = 1 . 1 0$ means the per-step bottleneck pack is, on average, about 10% costlier than the average pack. The excess over perfect modeled balance is $\rho - 1 ;$ only part of it may be removed by rank-local scheduling.

These metrics answer different questions. Pack-cost variability measures regularity within each rank’s emitted stream; the straggler ratio measures the synchronized penalty across ranks. BCL’s fixed-cardinality objective, developed next, reduces the maximum pack cost within each bounded local window without changing the records consumed or the number of records per step.

## 3 BCL DESIGN AND IMPLEMENTATION

## 3.1 Rank-local deployment boundary

BCL runs independently inside each rank’s data-loader path. Let B be the required number of records per emitted batch and K the lookahead in batches. A worker stages a bounded window of

$$
W = K B\tag{6}
$$

records from its own stream, assigns those records to K output packs, and then resumes filling the next window. It reads no other rank’s stream, exchanges no per-record metadata, and moves no records across ranks.

The deployment contract has three invariants. First, every staged record is emitted exactly once. Second, every output pack contains exactly B records. Third, reordering is confined to the current $W \cdot$ -record window. BCL therefore preserves the optimizer’s record-count cadence and the existing distributed collectives while trading a bounded amount of host-side staging and reordering for better balance. Length capping, rendering, and tokenization occur before scheduling; BCL never silently truncates an alreadyrendered record.

## 3.2 Fixed-cardinality objective

For one window, let $P _ { 1 } , \ldots , P _ { K }$ be a disjoint partition of the record indices, with $| P _ { k } | = B$ for every pack. Using the cost function from Equation 1, the fixed-cardinality objective is

$$
\operatorname* { m i n } _ { P _ { 1 } , \dots , P _ { K } } \operatorname* { m a x } _ { 1 \leq k \leq K } \sum _ { i \in P _ { k } } c ( \ell _ { i } ) .\tag{7}
$$

This is a cardinality-constrained makespan objective: it directly targets the most expensive pack that may become a synchronized straggler. Lower variance is a useful consequence, but not the optimization target.

The decision version contains strongly NP-complete 3- PARTITION when packs have three records and their number can vary (Garey & Johnson, 1979); dynamic programming applies only to restricted cases with few packs and bounded integer costs. BCL starts with a longest-processingtime-first ordering, assigning each record to the least-loaded pack that is not full (Graham, 1969). Classical LPT bounds do not cover exact cardinality and local refinement, so we evaluate balance empirically rather than claiming an approximation bound.

Algorithm 1 Fixed-cardinality BCL on one rank-local win  
dow   
Require: Lengths $\ell _ { 1 } , \ldots , \ell _ { W }$ with ${ \overline { { W = K B } } } ;$ cost $c ;$ tie   
key τ; rounds $Q$   
1: I ← sort indices by $( - c ( \ell _ { i } ) , \tau ( i ) )$   
2: Initialize K empty packs $P _ { k }$ with loads $L _ { k } \gets 0$   
3: for each i in I do   
4: k ← least-loaded pack with $| P _ { k } | < B ,$ , breaking ties   
by index   
5: $P _ { k }  P _ { k } \cup \{ i \} ; L _ { k }  L _ { k } + c ( \ell _ { i } )$   
6: end for   
7: for $q = 1 , \ldots , Q$ do   
8: while a heavy–light exchange strictly lowers   
max $L _ { k }$ do   
9: Select a deterministic improving exchange   
10: Apply the exchange and update the two affected   
loads   
11: end while   
12: end for   
13: return $P _ { 1 } , \ldots , P _ { K }$

## 3.3 Bounded scheduling procedure

Algorithm 1 gives the fixed-cardinality procedure. A deterministic per-record tie key makes the assignment reproducible. A scan of the non-full packs selects the least-loaded pack while enforcing cardinality. Refinement considers exchanges between the current heaviest and lightest packs and accepts one only when it strictly reduces the maximum pack cost. Because refinement uses exchanges rather than one-way moves, it cannot change pack cardinality.

The construction conserves records and cardinality by design.

## 3.4 Streaming integration

The public simulation, NeMo input path, and multi-node loader share the fixed-cardinality scheduling contract. In the NeMo run, each GPU worker reads only its own shards and schedules KB records into K packs of exactly B. The worker consumes its own packs; neither records nor cost metadata are exchanged between ranks.

For the pooled reference, we use NeMo-RL’s modified firstfit decreasing (MFFD) packer (Johnson & Garey, 1985; NVIDIA, n.d.) from the earlier global-input campaign. It repacks the current RB documents into R token-budget rows; per-rank BCL instead looks ahead across batches without seeing other ranks’ records. Because the loaders differ, the measured speedup reflects both packing and inputpipeline costs.

## 3.5 Cost and systems overhead

For $W = K B$ records, stable sorting costs O(W log W) and scanning the K packs for each assignment costs $O ( W K )$ . Two refinement rounds repeat improving heavy– light exchanges until none remains; storage is $O ( W )$ for the staged records, costs, and assignments. The scheduler operates on compact cost metadata after rendering, so it adds no GPU collective and does not increase the optimizer batch or the number of active records in a training step.

Larger K stages more records on the host, delays the first emitted pack, and permits farther reordering. It cannot make an infeasible record fit or remove kernel and communication work shared by all ranks.

## 4 REDUCIBILITY AND COORDINATION BOUNDARIES

## 4.1 A single-record lower bound

Scheduling can only rearrange the records available in a window; it cannot split a record or make that record cheaper. For one rank-local window, let

$$
S = \sum _ { i = 1 } ^ { W } c ( \ell _ { i } ) , \qquad \bar { C } = \frac { S } { K } , \qquad m = \operatorname * { m a x } _ { i } c ( \ell _ { i } ) ,\tag{8}
$$

where $\bar { C }$ is the average cost of a B-record pack. Any valid partition has

$$
C _ { \mathrm { m a x } } ^ { \star } \geq \operatorname* { m a x } ( \bar { C } , m ) .\tag{9}
$$

The first term is the average-load bound; the second follows because the largest record must appear in some pack. Normalizing by $\bar { C }$ gives the single-record irreducibility index

$$
\eta = \frac { m } { \bar { C } } = \frac { K m } { S } .\tag{10}
$$

When $\eta \ll 1$ , no individual record forces a large residual and the window is potentially reducible by assignment. When $\eta \geq 1$ , one record alone costs at least as much as an average pack, so even an optimal scheduler has a normalized maximum of at least η.

This diagnostic is a lower bound, not a complete attainability test. Multiple large records, exact-cardinality constraints, and a small window can leave additional discrete imbalance even when $\eta \ : < \ : 1$ . Conversely, increasing K can expose more complementary records, but it cannot lower the floor imposed by an unchanged m. A p99 cap can reduce the dominant record cost m. If the capped window has $\eta < 1$ and its packs fit in memory, the single-record bound no longer forces a straggler; BCL can then reduce the assignment-induced part of the critical path.

## 4.2 Why bounded lookahead helps

Let the per-record cost random variable $X = c ( \ell )$ have mean $\mu ,$ standard deviation σ, and coefficient of variation $\operatorname { C V } _ { X } = \sigma / \mu .$ . Under an independent random-assignment approximation, a B-record pack has mean $B \mu$ and coefficient of variation

$$
\mathrm { C V } _ { B } \approx { \frac { \mathrm { C V } _ { X } } { \sqrt { B } } } .\tag{11}
$$

If R rank-local pack costs have similar distributions, an order-statistics approximation gives

$$
\rho _ { \mathrm { r a n d o m } } \approx 1 + a _ { R } { \frac { \mathrm { C V } _ { X } } { \sqrt { B } } } ,\tag{12}
$$

where $a _ { R }$ is the standardized expected maximum and increases with the number of synchronized ranks (Dean $\&$ Barroso, 2013). This explains why the same workload variation creates more straggler opportunity at larger scale.

If record costs are independent and each rank can divide its KB records into K nearly equal-cost packs of B, those packs approach their rank-window mean. The relative spread of that mean scales as $\mathrm { C V } _ { X } / \sqrt { K B } .$ , suggesting

$$
\rho _ { \mathrm { B C L } } \sim 1 + a _ { R } \frac { \mathrm { C V } _ { X } } { \sqrt { K B } } .\tag{13}
$$

Reusing $a _ { R }$ for rank-window means is an approximation; its value depends on the distribution of the costs being maximized. The relation suggests diminishing gains as $K$ grows. Correlated streams and scheduling-dependent pack costs can change that trend. Section 5 tests modeled balance; Sections 6–7 measure step time on hardware.

## 4.3 The rank-divergence floor

Local lookahead also cannot change the total cost already assigned to a rank. For a block of K synchronized steps, let rank r receive window total $S _ { r }$ and average pack cost $\bar { C } _ { r } = S _ { r } / K$ . Define the cross-rank mean

$$
\bar { C } _ { \mathrm { a l l } } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \bar { C } _ { r }\tag{14}
$$

and the rank-divergence index

$$
\delta = \frac { \operatorname* { m a x } _ { r } \bar { C } _ { r } } { \bar { C } _ { \mathrm { a l l } } } .\tag{15}
$$

For aligned steps with one fixed window per GPU rank, local permutations cannot reduce the block’s normalized average critical path below δ: they cannot move cost between rank windows.

When ranks receive shuffled shards of the same distribution, δ is typically close to one and temporal lookahead can address most assignment-induced variation. When shards or user populations differ systematically, δ grows and the missing information is genuinely cross-rank. A global scheduler may then improve balance by exchanging metadata or moving samples, at the cost of a larger communication domain and tighter runtime coupling (Wang et al., 2026; Wan et al., 2026).

## 4.4 Choosing the required intervention

Table 1 connects the single-record and rank-divergence bounds with the remaining modeled straggler excess. Memory feasibility is checked before scheduling. Both δ and $\rho$ are at least one; 1.10 represents a 10% modeled excess over the corresponding average.

Table 1. Workload boundary and appropriate intervention.
<table><tr><td>Diagnostic</td><td>Implication</td></tr><tr><td> $\eta < 1 , \delta \approx 1$ </td><td>Local variation is potentially reducible; in- crease bounded lookahead until marginal gains saturate.</td></tr><tr><td> $\eta \geq 1$ </td><td>A record meets or exceeds mean modeled pack cost; cap, split, or change its representa-</td></tr><tr><td> $\delta > 1$ </td><td>tion before scheduling. Rank-window means differ; if material, re- shard or use cross-rank coordination.</td></tr><tr><td> $\rho \approx 1$ </td><td>Little modeled imbalance remains; limited modeled scheduling headroom.</td></tr></table>

Notation: η compares the largest record cost with the mean pack cost in one window; δ compares the heaviest rank-window mean with the mean across ranks; ρ (Eq. 5) compares the average synchronized maximum pack cost with the average pack cost. Equations 12 and 13 approximate $\rho$ under random assignment and idealized BCL, respectively.

Memory infeasibility is the strongest form of the singlerecord boundary. If one rendered record cannot fit within the available memory, no assignment can make it feasible. The length policy must therefore be established before BCL. Earlier uncapped probes distinguish proactive caps from caps prompted by an OOM (Section 9); the streaming comparison fixes p99 caps across arms. Earlier $B = 8$ no-lookahead runs on CodeParrot and OpenWebMath OOMed on oversized rows; per-rank BCL completes on both. Scheduling cannot make an individually infeasible record fit.

## 5 PUBLIC-DISTRIBUTION SCHEDULING SIMULATION

## 5.1 Question and public inputs

Using the same sampled records in both arms, we test whether BCL steadies modeled pack costs and lowers the synchronized maximum. GPU step time is measured separately.

The six checked-in arrays contain 165,011 integer token lengths derived from Stanford Alpaca (Taori et al., 2023), the train split of the CodeParrot clean-valid dataset (Hugging Face, n.d.), Dolly (Conover et al., 2023), OpenHermes 2.5 (Teknium, 2023), OpenWebMath (Paster et al., 2023), and the HuggingFaceH4 UltraChat 200k subset (Ding et al., 2023). They contain no source text. The public source identifier, rendering rule, sample count, and SHA-256 checksum for every array are retained with the experiment. Table 2 reports the uncapped length statistics.

Table 2. Public length arrays used by the scheduling simulation. Percentiles are token lengths before the 1,024-token cost cap.
<table><tr><td>Dataset</td><td>Records</td><td> $\mathsf { p 5 0 }$ </td><td> $\mathsf { p } 9 9$ </td></tr><tr><td>Alpaca</td><td>30,000</td><td>59</td><td>270</td></tr><tr><td>CodeParrot</td><td>30,000</td><td>975</td><td>20,423</td></tr><tr><td>Dolly</td><td>15,011</td><td>101</td><td>1,146</td></tr><tr><td>OpenHermes</td><td>30,000</td><td>237</td><td>2,468</td></tr><tr><td>OpenWebMath</td><td>30,000</td><td>763</td><td>16,900</td></tr><tr><td>UİtraChat</td><td>30,000</td><td>1,051</td><td>3,004</td></tr></table>

## 5.2 Paired simulation design

For each dataset, the benchmark draws one cost stream for each of $R = 8$ ranks and reuses that exact stream for both scheduling arms. Each arm therefore consumes the same sampled input stream on every rank; only windowlocal grouping and emitted order change. We draw each rank’s records independently, with replacement. The results describe modeled workload balance, not measured GPU step time. The run contains 1,000 synchronized steps with perrank cardinality $B = 1 2 8$ , seed zero, and capped quadratic cost

$$
c ( \ell ) = \operatorname* { m i n } ( \ell , 1 0 2 4 ) ^ { 2 } .\tag{16}
$$

The BCL-disabled arm emits arrival-order packs with $K =$ 1. The BCL arm uses $K = 3 2$ with cardinality-constrained LPT initialization and two iterative refinement rounds. Because 1,000 is not divisible by $^ { 3 2 , }$ the last eight steps use a smaller eight-pack window rather than drawing or discarding additional records. Constructed pack order is deterministically shuffled within each rank window before ranks are paired into synchronized steps.

The primary metrics are the pack-cost coefficient of variation and straggler ratio defined in Section 2. We report their relative reductions as

$$
\Delta _ { \mathrm { C V } } = 1 - { \frac { \mathrm { C V } _ { K = 3 2 } } { \mathrm { C V } _ { K = 1 } } } ,\tag{17}
$$

and

$$
\Delta _ { \rho } = 1 - \frac { \rho _ { K = 3 2 } - 1 } { \rho _ { K = 1 } - 1 } .\tag{18}
$$

The second expression measures the fraction of modeled straggler excess removed; it does not treat the unavoidable unit baseline as removable work.

## 5.3 Results

Table 3 shows the paired result. BCL reduces pack-cost variability by 81.4–82.8% on every retained distribution. The modeled straggler ratio falls from 1.050–1.345 under arrival-order batching to 1.009–1.066 with BCL, removing 81.0–83.9% of modeled straggler excess.

Table 3. Paired public-distribution simulation. Both arms consume the exact same sampled record stream on every rank. $\Delta _ { \rho }$ is the reduction in ρ − 1.
<table><tr><td>Dataset</td><td>CV, K = 1</td><td> $\mathrm { C V } , K = 3 2$ </td><td> $\Delta _ { \mathrm { C V } }$ </td></tr><tr><td>Alpaca</td><td>0.187</td><td>0.032</td><td>82.8%</td></tr><tr><td>CodeParrot</td><td>0.051</td><td>0.009</td><td>81.5%</td></tr><tr><td>Dolly</td><td>0.228</td><td>0.041</td><td>82.1%</td></tr><tr><td>OpenHermes</td><td>0.133</td><td>0.024</td><td>82.2%</td></tr><tr><td>OpenWebMath</td><td>0.066</td><td>0.012</td><td>81.4%</td></tr><tr><td>UltraChat</td><td>0.036</td><td>0.006</td><td>82.3%</td></tr><tr><td>Dataset</td><td> $\rho , K = 1$ </td><td> $\rho , K = 3 2$ </td><td> $\Delta _ { \rho }$ </td></tr><tr><td>Alpaca</td><td>1.300</td><td>1.048</td><td>83.9%</td></tr><tr><td>CodeParrot</td><td>1.072</td><td>1.014</td><td>81.2%</td></tr><tr><td>Dolly</td><td>1.345</td><td>1.066</td><td>81.0%</td></tr><tr><td>OpenHermes</td><td>1.195</td><td>1.036</td><td>81.4%</td></tr><tr><td>OpenWebMath</td><td>1.094</td><td>1.017</td><td>82.4%</td></tr><tr><td>UltraChat</td><td>1.050</td><td>1.009</td><td>81.3%</td></tr></table>

The relative reduction is consistent even though the starting opportunity differs substantially. Dolly and Alpaca have the largest decreases in the modeled straggler ratio $\rho .$ UltraChat begins much closer to balance and has a smaller decrease in $\rho$ points despite a similar proportional reduction in $\rho - 1$ . CodeParrot and OpenWebMath have long uncapped tails. The 1,024-token cap limits their modeled costs; BCL then reduces the remaining assignment variation in the simulation.

## 6 NEMO-RL STREAMING EVALUATION

## 6.1 Setup and controls

We fine-tune Meta’s pretrained Llama-3.2-1B checkpoint (Meta, 2024) with NeMo-RL v0.7.0 SFT (NVIDIA, n.d.) on one eight-H200 node (FSDP2, bf16, FlashAttention 3 (Shah et al., 2024)). The 13 public corpora are Alpaca (Taori et al., 2023), Dolly (Conover et al., 2023), OASST1 (Kopf¨ et al., 2023), OASST2 (Open-Assistant contributors, 2023), OpenHermes (Teknium, 2023), ShareGPT (Community contributors, n.d.), ShareGPT-GPT4 (Xu, n.d.), Ultra-Chat (Ding et al., 2023), CodeParrot (Hugging Face, n.d.), FineWeb (Penedo et al., 2024), OpenWebMath (Paster et al., 2023), OpenWebText (Gokaslan & Cohen, 2019), and SlimPajama (Soboleva et al., 2023). Each corpus is processed by NeMo’s SFT preprocessing with the same perdocument p99 length cap in every arm. The study-specific overlength truncation is shared across arms.

Table 4. NeMo-RL SFT comparisons. Only the first two arms share the per-rank input path. MFFD is a pooled reference from the preceding campaign; the default path was rerun in the current campaign.
<table><tr><td>Arm</td><td>Assignment</td></tr><tr><td>Default packing</td><td>Per-rank reader and collector; next B documents from each rank, no lookahead.</td></tr><tr><td>BCL</td><td>Same per-rank path with K = 32; exactly B documents per rank.</td></tr><tr><td>Pooled MFFD</td><td>Earlier global-input path; per-step token-budget packing into one row per rank.</td></tr><tr><td>NeMo default path</td><td>Own loader and MFFD packer; potentially multiple rows per rank.</td></tr></table>

For the per-rank experiment, each of the eight NeMo-RL workers owns eight of 64 Parquet shards. Two Ray readers per worker (Moritz et al., 2018) tokenize that rank’s records and schedule KB-document windows. A collector inside the worker takes one pack per step and builds one unpadded row of B documents for NeMo’s training worker. No documents or scheduling metadata pass between ranks or through the driver. BCL uses $K = 3 2 .$ , capped squared-length cost, LPT initialization, two swap passes, and a seeded pack shuffle; default packing of input documents<sup>1</sup> uses the same per-rank pipeline without lookahead. Both use the same model, loss, optimizer, global batch size, FSDP2 training path, and train microbatch size of one. Both disable per-step CUDA cache clearing and freeze Python garbage collection after setup to avoid asymmetric allocation stalls.

## 6.2 Measured step time

Each arm has three 300-step repeats, excluding the first 20 steps. We time the wall-clock interval between consecutive batch fetches, covering input and training in both paths. NeMo’s built-in step timer includes per-rank fetch but excludes the earlier driver’s fetch. Table 5 reports the percentage speedups. Against pooled MFFD, rank-local BCL shows +5.0% to +16.7% speedup across all 13 corpora. This compares complete input paths, not packing algorithms in isolation. On the same per-rank path, $K = 3 2$ shows +0.4% to +30.5% mean speedup over default packing on 11 corpora, isolating lookahead. The K = 1 arm was not rerun on CodeParrot or OpenWebMath at B = 8 after earlier oversized-row OOMs. NeMo’s default loader and packer were rerun as a separate reference.

Reruns of NeMo’s default path differ by at most 1.5% between campaigns, providing a check on node drift. Moving the same BCL arm from driver-merged to per-rank input gives +2.8% to +7.5% speedup on 12 corpora $( + 0 . 2 \%$ on Alpaca). Per-step audits confirm one unpadded B-document row per rank and matching loader and trainer token counts. Two readers per rank usually keep input ready after warmup.

## 7 MULTI-NODE H200 EVALUATION

## 7.1 Workload and reporting boundary

The final evaluation asks whether the scheduling benefit persists from one to four nodes on an internal sequence workload associated with the Feed Sequential Recommender training setting described by Hertel et al. (2026). The training records are capped at 1,024 tokens before scheduling. We report only length statistics and aggregate performance; no member content, raw records, production model code, or execution harness is included.

The retained full characterization covers 7,187 shards and 320,137,728 records. Mean capped length is 215.6 tokens, and 10.44% of records lie at the cap. A separately retained 16-shard sample contains 327,680 records, with mean 213.08, median 28, and 10.29% at the cap. The sample has similar mean length and cap rate.

## 7.2 Length-sample scheduler probe

Before measuring the model, a length-only probe applies the same fixed-cardinality scheduler to the 16-shard sample at per-rank $B = 3 8 4$ . Table 6 shows the modeled straggler ratio at eight and sixteen ranks. Moving from K = 1 to $K = 3 2$ removes approximately 82% of straggler excess at both scales; $K = 6 4$ provides a smaller additional reduction.

Table 6. Modeled straggler ratios from scheduling 327,680 measured capped lengths, not GPU step times.
<table><tr><td>Ranks</td><td> $K = 1$ </td><td> $K = 3 2$ </td><td> $K = 6 4$ </td></tr><tr><td>8</td><td>1.1438</td><td>1.0261</td><td>1.0192</td></tr><tr><td>16</td><td>1.1931</td><td>1.0363</td><td>1.0286</td></tr></table>

The higher arrival-order ratio at 16 ranks is consistent with the rank-count effect in Equation 12. BCL keeps the residual close to one without moving samples between ranks.

## 7.3 Compiled H200 setup

The hardware comparison uses compiled execution (Ansel et al., 2024) on H200 GPUs, per-rank batch size 384, and BCL window $K = 3 2$ . One, two, and four nodes correspond to 8, 16, and 32 ranks and derived global batch sizes of 3,072, 6,144, and 12,288 records. The baseline is the same packed execution path with BCL disabled. Reported values are post-warm-up seconds per step, aggregated across repeated runs. Global batch size rises with rank count; this is weak scaling at fixed per-rank batch size.

Table 5. NeMo-RL SFT wall-clock speedup at $B = 8 ,$ defined as $1 0 0 ( t _ { \mathrm { b a s e l i n e } } / t _ { \mathrm { B C L } } - 1 )$ . Values are percentages; ± terms are standard deviations in percentage points over three repeats. Pooled MFFD uses a cross-campaign ratio of mean per-step times.
<table><tr><td>Corpus</td><td>vs. pooled MFFD (%)</td><td>vs. default packing (%)</td><td>vs. NeMo default (%)†</td></tr><tr><td>Alpaca</td><td>+5.0</td><td> $+ 0 . 4 \pm 1 . 4$ </td><td> $+ 1 2 2 . 8 \pm 2 . 5$ </td></tr><tr><td>Dolly</td><td>+12.3</td><td> $+ 4 . 3 \pm 0 . 4$ </td><td> $+ 7 7 . 1 \pm 2 . 0$ </td></tr><tr><td>OpenHermes</td><td>+11.3</td><td> $+ 7 . 5 \pm 0 . 5$ </td><td> $+ 7 4 . 8 \pm 0 . 8$ </td></tr><tr><td>OASST1</td><td>+9.5</td><td> $+ 4 . 9 \pm 1 . 2$ </td><td> $+ 1 0 9 . 1 \pm 1 . 0$ </td></tr><tr><td>OASST2</td><td>+11.1</td><td> $+ 5 . 5 \pm 0 . 3$ </td><td> $+ 1 0 4 . 2 \pm 2 . 1$ </td></tr><tr><td>FineWeb</td><td>+16.7</td><td> $+ 2 2 . 7 \pm 0 . 2$ </td><td> $+ 5 8 . 5 \pm 0 . 9$ </td></tr><tr><td>SlimPajama</td><td>+14.1</td><td> $+ 3 0 . 5 \pm 1 . 4$ </td><td> $+ 3 9 . 4 \pm 0 . 5$ </td></tr><tr><td>OpenWebText</td><td>+15.6</td><td> $+ 3 0 . 2 \pm 0 . 1$ </td><td> $+ 5 5 . 8 \pm 1 . 3$ </td></tr><tr><td>UltraChat</td><td>+10.4</td><td> $+ 1 2 . 5 \pm 0 . 9$ </td><td> $+ 7 4 . 7 \pm 1 . 9$ </td></tr><tr><td>ShareGPT-GPT4</td><td>+13.7</td><td> $+ 1 6 . 3 \pm 0 . 9$ </td><td> $+ 7 1 . 0 \pm 1 . 2$ </td></tr><tr><td>ShareGPT</td><td>+13.3</td><td> $+ 1 6 . 4 \pm 0 . 9$ </td><td> $+ 8 7 . 1 \pm 1 . 7$ </td></tr><tr><td>OpenWebMath</td><td>+15.2</td><td></td><td> $+ 3 5 . 6 \pm 0 . 6$ </td></tr><tr><td>CodeParrot</td><td>+13.0</td><td></td><td> $+ 3 4 . 3 \pm 1 . 0$ </td></tr></table>

<sup>\*</sup> Pooled MFFD uses the earlier global-input path; this wall-clock comparison includes input-path changes and is not paired by repeat. <sup>†</sup> The NeMo default path uses its own loader and may execute multiple rows per rank; its speedup is not an isolated scheduling effect. Dashes indicate that the $K = 1$ arm was not rerun after earlier B=8 OOMs.

The comparison includes three BCL and four Packed repeats at 8 and 16 ranks, and three per arm at 32 ranks.

## 7.4 Compiled step-time scaling

Table 7 reports the primary measured comparison. BCL is faster than packed execution at every topology. Step time falls by 16.45%, 19.51%, and 25.52% at one, two, and four nodes, corresponding to 1.197×, 1.242×, and 1.343× speedups. This rise is consistent with the larger straggler effect expected as more ranks synchronize (Equation 12).

Table 7. Primary measured compiled H200 step time. Values are mean ± sample standard deviation in seconds per step across repeated runs.
<table><tr><td>Nodes / ranks</td><td>Packed</td><td></td><td>BCL Speedup</td></tr><tr><td>1/8</td><td> $0 . 3 9 1 5 \pm 0 . 0 0 3 5$ </td><td> $0 . 3 2 7 1 \pm 0 . 0 0 0 5$ </td><td>1.197×</td></tr><tr><td>2/ 16</td><td> $0 . 4 3 9 8 \pm 0 . 0 0 3 5$ </td><td> $0 . 3 5 4 0 \pm 0 . 0 0 0 5$ </td><td>1.242×</td></tr><tr><td>4/32</td><td> $0 . 5 4 3 6 \pm 0 . 0 2 3 3$ </td><td> $0 . 4 0 4 9 \pm 0 . 0 0 3 4$ </td><td>1.343×</td></tr></table>

BCL reorders records only within bounded rank-local windows rather than globally sorting by length, which can impose a training curriculum (Bengio et al., 2009). In Feed sequence-model evaluations, mean AUC differed from the reference by just 0.07% on average, with no observed loss in this aggregate metric.

## 7.5 Peak-memory evidence

Single-node telemetry at B = 384 provides a supporting memory result. The busiest-rank peak is $1 3 8 . 7 \pm 0 . 8$ GiB for packed execution and 124.9 ± 4.2 GiB with BCL, approximately 13.8 GiB or 9.9% lower.

## 7.6 Calibrated sensitivity analysis

Calibrated length-distribution replays extend the K and B sweeps beyond the measured H200 setup. Table 8 shows the trends. At B = 384, the modeled straggler ratio falls rapidly through K = 16 and reaches 1.027 at $K = 3 2 ;$ the change to $K = 6 4$ is smaller. At fixed $K = 3 2 ,$ , larger packs move closer to the calibrated global assignment target because they average more per-record variation.

Table 8. Calibrated Feed workload sensitivity. These values are model/replay outputs, not direct hardware measurements.
<table><tr><td>K</td><td>1</td><td>2 4</td><td>8</td><td>16</td><td>32</td><td>64</td></tr><tr><td>ρ</td><td>1.159 1.107</td><td>1.076</td><td>1.054</td><td>1.038</td><td>1.027</td><td>1.019</td></tr><tr><td> $B { \mathrm { ~ a t ~ } } K = 3 2$ </td><td></td><td>32</td><td>96</td><td>256</td><td>384</td><td>1,024</td></tr><tr><td>% modeled target</td><td></td><td>94.5</td><td>96.7</td><td>98.0</td><td>98.3</td><td>98.9</td></tr></table>

From $K = 3 2$ and $B = 3 8 4 .$ , increasing either parameter adds little in the replays. The reported percentages measure proximity to a modeled assignment target, not measured H200 speedup.

## 8 RELATED WORK

Batching, packing, and padding elimination. Tokenbudget and length-aware batching reduce variation in the amount of padding or tokens processed per batch, but may vary record cardinality and therefore require explicit normalization and optimization policies (Ott et al., 2018). Sequence packing and variable-length attention kernels remove computation on padding within a packed batch (Krell et al., 2021; Dao et al., 2022). BCL is complementary: it preserves record cardinality and changes only which records share packs that meet at a synchronization barrier.

Variable-length distributed training. Hydraulis, first released in 2024, jointly selects heterogeneous parallel strategies and assigns data within and across model replicas (Li et al., 2026). WLB-LLM combines workload-aware document packing with per-document context-parallel sharding (Wang et al., 2025). DynaPipe builds variable-length microbatches and dynamically schedules pipeline execution and communication (Jiang et al., 2024). These systems demonstrate that length-aware execution can recover substantial throughput, but they alter parallelism, microbatch structure, or cross-worker data assignment. BCL instead fixes the existing parallel plan and collective structure, and exposes a rank-local data-loader option with exact record cardinality.

Scheduling foundations. The assignment core is not a new scheduling heuristic. Its initialization follows classical longest-processing-time list scheduling (Graham, 1969); the NeMo comparison uses modified first-fit decreasing (Johnson & Garey, 1985) as a per-step reference. BCL’s systems contribution is the bounded, streaming deployment contract and its coordination boundary. The motivation follows the broader synchronous-straggler literature: each step waits for its slowest worker, and tail effects become more consequential as more workers participate (Li et al., 2020; Chen et al., 2016; Dean & Barroso, 2013).

Recent approaches. Several recent systems address related workload skew through different forms of coordination. Libra uses a bounded sequence pool to address long-context attention-workload skew (Wang et al., 2026); ODC revisits parameter-server-style decoupling for LLM post-training (Wan et al., 2026); and RoutePack co-plans mixture-ofexperts placement with attention-aware data packing over an optimizer-step window (Shen et al., 2026). RoutePack plans globally for a mixture-of-experts workload, while BCL reorders fixed-size packs within independent reader windows. If rank-window totals differ substantially, coordinated assignment may be needed (Section 4).

## 9 DEPLOYMENT BOUNDARIES

Before scheduling, render and tokenize records as training will consume them and choose a length cap and batch size that fit in memory. Earlier probes used raw rendered-length CV ≥ 2 to trigger proactive p99 capping; lower-CV corpora also OOMed uncapped, so CV is a heuristic rather than a feasibility test. With feasible records, a length-only sweep can suggest a lookahead window, but its benefit and staging cost must be measured on the target trainer. A dominant record needs a shorter cap or a different representation; persistently heavier rank windows call for resharding or coordination. The NeMo comparison holds p99 caps fixed and does not evaluate their effect on training quality.

BCL changes which records share an optimizer step. Workloads with ordering constraints must define legal permutations before applying it. An exact mid-stream restart would need to restore reader positions, staged packs, shuffle state, and the per-rank reader merge. The current NeMo experiment does not test exact restart or held-out quality under that reordering.

## 10 CONCLUSION

Equal record counts do not guarantee equal work in variablelength distributed training. BCL groups each reader’s bounded window into balanced fixed-cardinality packs without exchanging cost metadata between readers or changing the model and collectives.

Paired simulation over six public length distributions reduces per-rank modeled pack-cost variability by 81.4– 82.8%. In NeMo-RL SFT, rank-local BCL improves wallclock step time by 1.050–1.167× over pooled MFFD across 13 corpora, including input-path differences. Against matched no-lookahead packing, BCL is 1.004–1.305× faster on 11 corpora and completes the other two, where earlier B = 8 no-lookahead runs OOMed. On the internal H200 workload, compiled measurements show 1.197– 1.343× speedup from one to four nodes.

BCL is most useful when records fit and the reader windows have similar total costs. Outlying records need capping or a change in representation; consistently heavier windows call for resharding or coordination.

## ACKNOWLEDGMENTS

We thank our leaders Animesh Singh and Dave Burgess for their invaluable expertise in the ML infrastructure stack and open-source strategy.

## REFERENCES

Ansel, J., Yang, E., He, H., et al. PyTorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 929–947, 2024. doi: 10. 1145/3620665.3640366. URL https://doi.org/ 10.1145/3620665.3640366.

Bengio, Y., Louradour, J., Collobert, R., and Weston, J. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning (ICML), pp. 41–48, 2009. doi: 10.1145/1553374. 1553380. URL https://doi.org/10.1145/

1553374.1553380.

Chen, J., Pan, X., Monga, R., Bengio, S., and Jozefowicz, R. Revisiting distributed synchronous SGD. arXiv preprint arXiv:1604.00981, 2016. URL https: //arxiv.org/abs/1604.00981. Presented at the ICLR 2016 Workshop track.

Community contributors. ShareGPT vicuna unfiltered, community mirror. https://huggingface. co/datasets/Aeala/ShareGPT\_Vicuna\_ unfiltered, n.d. Exact public source used in the NeMo evaluation, split train. Community-scraped conversations with no official release, dataset paper, or pinned upstream revision.

Conover, M., Hayes, M., Mathur, A., Xie, J., Wan, J., Shah, S., Ghodsi, A., Wendell, P., Zaharia, M., and Xin, R. Free dolly: Introducing the world’s first truly open instruction-tuned LLM. https://huggingface.co/datasets/ databricks/databricks-dolly-15k, 2023. Official Databricks dataset release. The public simulation uses the train split.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Re, C.´ FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems 35 (NeurIPS), 2022. URL https://arxiv.org/abs/2205.14135.

Dean, J. and Barroso, L. A. The tail at scale. Communications of the ACM, 56(2):74–80, 2013. doi: 10. 1145/2408776.2408794. URL https://doi.org/ 10.1145/2408776.2408794.

Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M., and Zhou, B. Enhancing chat language models by scaling high-quality instructional conversations. arXiv preprint arXiv:2305.14233, 2023. URL https://arxiv.org/abs/2305.14233. The public simulation and NeMo evaluation use https://huggingface.co/datasets/ HuggingFaceH4/ultrachat\_200k, split train sft.

Garey, M. R. and Johnson, D. S. Computers and Intractability: A Guide to the Theory ofNP-Completeness. W. H. Freeman, San Francisco, 1979.

Gokaslan, A. and Cohen, V. OpenWebText corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019. The NeMo evaluation uses the official mirror https://huggingface. co/datasets/Skylion007/openwebtext, config plain text, split train.

Goyal, P., Dollar, P., Girshick, R., Noordhuis, P.,´ Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv preprint arXiv:1706.02677, 2017. URL https://arxiv.org/abs/1706.02677.

Graham, R. L. Bounds on multiprocessing timing anomalies. SIAM Journal on Applied Mathematics, 17(2):416–429, 1969. doi: 10.1137/0117039. URL https://doi. org/10.1137/0117039.

Hertel, L., Srivastava, G., Naqvi, S. A., Kumar, S., Zhang, Y., Ocejo, B., Zelditch, B., Englhardt, A., Cheng, H., Hu, A., Alonso, A., Li, D., Dangi, S., Zhu, C., Zhou, M., Li, W., Huang, T., Borisyuk, F., Parameswaran, G., Tiwana, B. S., Sankar, S., Lan, Q., Choi, J., and Ghosh, S. An industrial-scale sequential recommender for LinkedIn feed ranking. arXiv preprint arXiv:2602.12354, 2026. URL https://arxiv.org/abs/2602.12354.

Hugging Face. CodeParrot clean, validation split. https://huggingface.co/datasets/ codeparrot/codeparrot-clean-valid, n.d. The public simulation uses the dataset’s train split. Accessed: 2026-08-24.

Jiang, C., Jia, Z., Zheng, S., Wang, Y., and Wu, C. DynaPipe: Optimizing multi-task training through dynamic pipelines. In Proceedings of the Nineteenth European Conference on Computer Systems (EuroSys), 2024. doi: 10.1145/3627703.3629585. URL https://arxiv. org/abs/2311.10418.

Johnson, D. S. and Garey, M. R. A 71/60 theorem for bin packing. Journal ofComplexity, 1(1):65–106, 1985. doi: 10.1016/0885-064X(85)90022-6. URL https://doi. org/10.1016/0885-064X(85)90022-6.

Kopf, A., Kilcher, Y., von R ¨ utte, D., Anagnos-¨ tidis, S., et al. OpenAssistant conversations— democratizing large language model alignment. In Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track, 2023. URL https://arxiv.org/abs/2304.07327. The NeMo evaluation uses https://huggingface.co/ datasets/OpenAssistant/oasst1, English reviewed leaf paths from the train split.

Krell, M. M., Kosec, M., Perez, S. P., and Fitzgibbon, A. Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance. arXiv preprint arXiv:2107.02027, 2021. URL https://arxiv.org/abs/2107.02027.

Li, H., Fu, F., Lin, S., Ge, H., Wang, X., Niu, J., Xue, J., Tao, Y., Wang, D., Jiang, J., and Cui, B. Hydraulis: Balancing large transformer model training via

co-designing parallel strategies and data assignment. In Proceedings ofthe ACM on Management ofData (SIG-MOD), 2026. doi: 10.1145/3769802. URL https: //arxiv.org/abs/2412.07894. First posted as arXiv:2412.07894 in December 2024.

Li, S., Zhao, Y., Varma, R., Salpekar, O., Noordhuis, P., Li, T., Paszke, A., Smith, J., Vaughan, B., Damania, P., and Chintala, S. PyTorch distributed: Experiences on accelerating data parallel training. Proceedings of the VLDB Endowment, 13(12):3005–3018, 2020. doi: 10.14778/3415478.3415530. URL https://arxiv. org/abs/2006.15704.

Meta. Llama 3.2-1b model card. https: //huggingface.co/meta-llama/Llama-3. 2-1B, 2024.

Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, Z., Paul, W., Jordan, M. I., and Stoica, I. Ray: A distributed framework for emerging AI applications. In Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation, 2018. URL https://arxiv.org/ abs/1712.05889.

NVIDIA. NeMo RL: A scalable and efficient post-training library. https://github.com/NVIDIA-NeMo/ RL, n.d. Accessed: 2026-08-24.

Open-Assistant contributors. OpenAssistant conversations release 2 (OASST2). https://huggingface.co/ datasets/OpenAssistant/oasst2, 2023. Second data release of the OpenAssistant project. The NeMo evaluation uses English reviewed leaf paths from the train split; see Kopf et al. (2023) for methodology.¨

Ott, M., Edunov, S., Grangier, D., and Auli, M. Scaling neural machine translation. In Proceedings of the Third Conference on Machine Translation: Research Papers (WMT), pp. 1–9, Brussels, Belgium, 2018. Association for Computational Linguistics. doi: 10.18653/v1/ W18-6301. URL https://aclanthology.org/ W18-6301/.

Paster, K., Dos Santos, M., Azerbayev, Z., and Ba, J. OpenWebMath: An open dataset of high-quality mathematical web text. arXiv preprint arXiv:2310.06786, 2023. URL https://arxiv.org/abs/2310. 06786. Dataset: https://huggingface.co/ datasets/open-web-math/open-web-math.

Penedo, G., Kydl´ıcek, H., Ben Allal, L., Lozhkov,ˇ A., Mitchell, M., Raffel, C., Von Werra, L., and Wolf, T. The FineWeb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024. URL

https://arxiv.org/abs/2406.17557. The NeMo evaluation uses https://huggingface. co/datasets/HuggingFaceFW/fineweb, config sample-10BT, split train.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. arXiv preprint arXiv:2407.08608, 2024. URL https:// arxiv.org/abs/2407.08608.

Shen, Y., Han, X., Zhu, X., Li, G., and Pan, Z. RoutePack: Expert placement and attention-aware data packing for MoE reinforcement learning. arXiv preprint arXiv:2608.12146, 2026. URL https://arxiv. org/abs/2608.12146. Concurrent work posted in August 2026.

Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hestness, J., and Dey, N. SlimPajama: A 627b token cleaned and deduplicated version of RedPajama. Cerebras dataset release, June 2023. URL https://huggingface.co/datasets/ cerebras/SlimPajama-627B. Origin of SlimPajama. The NeMo evaluation uses the community 6B-token subset https://huggingface.co/ datasets/DKYoon/SlimPajama-6B, split train, rather than the full Cerebras release.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following LLaMA model. https://github.com/tatsu-lab/ stanford\_alpaca, 2023. The public simulation uses https://huggingface.co/datasets/ tatsu-lab/alpaca, split train.

Teknium. OpenHermes 2.5. https://huggingface. co/datasets/teknium/OpenHermes-2.5, 2023. The public simulation uses the train split of this synthetic and curated instruction-and-chat compilation.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NeurIPS), 2017. URL https://arxiv.org/abs/1706.03762.

Wan, X., Qi, P., Huang, G., Ruan, C., Lin, M., and Li, J. Revisiting parameter server in LLM post-training. arXiv preprint arXiv:2601.19362, 2026. URL https:// arxiv.org/abs/2601.19362. Accepted at ICLR 2026.

Wang, Y., Yuan, X., Yang, K., Peng, J., Lu, P., Li, M., Zhang, Z., Si, C., Ruan, Z., Chen, H., Jiang, L., Wang, S., Chen, L., Men, R., Yuan, M., Tan, G., Li, Y., Jia, W.,

and Zhou, J. Libra: Taming attention workload skew in long-context LLM training with bounded sequence pool. arXiv preprint arXiv:2607.23250, 2026. URL https://arxiv.org/abs/2607.23250.

Wang, Z., Cai, A., Xie, X., Pan, Z., Guan, Y., Chu, W., Wang, J., Li, S., Huang, J., Cai, C., Hao, Y., and Ding, Y. WLB-LLM: Workload-balanced 4d parallelism for large language model training. arXiv preprint arXiv:2503.17924, 2025. URL https://arxiv. org/abs/2503.17924.

Xu, M. ShareGPT-GPT4, community dataset. https://huggingface.co/datasets/ shibing624/sharegpt\_gpt4, n.d. Exact public source used in the NeMo evaluation, split train. Community-curated conversations with no official dataset paper or pinned upstream revision.