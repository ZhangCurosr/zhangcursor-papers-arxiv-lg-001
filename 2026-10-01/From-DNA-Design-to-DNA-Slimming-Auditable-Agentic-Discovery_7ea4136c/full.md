# From DNA Design to DNA Slimming: Auditable Agentic Discovery of a Deletion-Only Designer

Joel Shor AI Bio Design at the Allen Institute & Move37 Labs joel.shor@alleninstitute.org

## Abstract

Compact regulatory DNA can free up space in vector payloads, reduce synthesis and assay burden, and expose which sequence features drive predicted activity. Yet most model-based nucleic-acid designers optimize fixed-length sequences through substitutions; they do not ask which bases of an existing functional element can be removed while retaining predicted activity. We define the task of sequence slimming as selecting an exact-length, order-preserving subsequence while retaining activity. Modeled on the design benchmark NucleoBench, we propose a quantitative evaluation for slimming that balances sequence reduction with maintaining function. Each slimmer must return both the subsequence and its source indices, which can be used to verify that the slimmer obeyed task requirements. To our knowledge, this is the first dedicated benchmark of this deletion-only problem. The coding agent Empirical Research Assistant (ERA) then searched over executable designer programs. ERA received the task prompt and a successful substitutiononly designer GrAdaBeam as a starting program, and it modified the designer to produce GRADASLIM. We report held-out evaluations for five transcription-factor binding targets, comparing random, greedy, and ERA-guided slimming at 400 and 100 bp. ERA has the highest mean in 9/10 settings. Paired bootstrap intervals for ERA minus greedy are above zero in all five 400-bp settings, below zero in one 100-bp setting, and overlap zero in the remaining four.

## 1 Introduction

Regulatory elements that are compact have practical benefits. Viral vectors are limited in the length of the payload they can deliver, and more compact payloads can show improved properties, such as transducibility [8]. However, natural flanking sequences sometimes contain desirable properties [5], making the slimming problem a complex one. Slimming therefore needs to be considered as a size/function tradeoff.

Model-based DNA designers take a deep sequence-to-function model (oracle) and iteratively make substitutions to maximize the oracle-predicted activity. When the oracles are trained to predict transcription-factor binding and motif syntax, such as BPNet [1], this iterative approach designs for regulatory behavior. This technique has been validated biologically [13, 3]. Only recently has there been a quantitative comparison of model-based, substitution-only designers [11]. This comparison explored a number of substitution designers [4, 10, 12, 11], but no deletion-based designers.

GrAdaBeam is a particularly relevant starting point for sequence slimming because it was designed as a hybrid of gradient guidance and adaptive discrete search [11]. It operates over proposed discrete actions. In principle, the action space can therefore simply be changed from substitutions to deletions. The gradient-based components do not transfer as directly. For example, a substitution modifies one position in a fixed-length representation, whereas deleting a base shifts every downstream coordinate and creates a new sequence junction. Consequently, the substitution gradients used by GrAdaBeam,

Ledidi, and FastSeqProp cannot be interpreted directly as deletion gradients [10, 4]. Purely gradientbased designers would require a new variable-length relaxation or alignment-aware parameterization, whereas GrAdaBeam provides a natural discrete search architecture in which to test deletion actions.

Previous work has explored short enhancer design in the wet lab [5]. Earlier computational enhancer design used fixed-length substitutions or mixed substitution, insertion, and deletion paths [6]. To our knowledge, prior work has not defined and benchmarked the dedicated problem studied here: return an exact-length, deletion-only, order-preserving subsequence of an existing functional regulatory sequence.

Agentic program search such as FunSearch, AlphaEvolve, and the Empirical Research Assistant (ERA) use language models plus automated evaluation to search over executable programs [9, 7, 2]. In our setting, ERA searches over executable designer programs, while each proposed program searches over DNA subsequences. This nested problem decomposition is attractive because the output of the outer loop is ordinary code that can be inspected, versioned, and tested directly on biological tasks.

This short paper makes three contributions. First, we define a machine-verifiable evaluation framework for deletion-only slimming. Second, we describe an algorithm discovered by ERA in a deletiononly sequence-slimming task: a two-phase search that learns a salience map from scored discrete candidates. Third, we report five held-out BPNet studies comparing the discovered program with random and greedy controls under a common, target-separated protocol. The full algorithm appears in Appendix A.

## 2 Defining sequence slimming

Problem definition. Given a source DNA sequence x of length $L$ and a target length $m < L ,$ a slimmer must return two objects: a candidate sequence $\widehat { x }$ and its claimed retained source positions $I = ( i _ { 1 } , \ldots , i _ { m } )$ . An output is valid only if both objects have length $m , 1 \leq i _ { 1 } < \cdot \cdot \cdot < i _ { m } \leq L ,$ and $\widehat { x } _ { j } = x _ { i _ { j } }$ for every j. Equivalently, the candidate must equal the indexed subsequence $x _ { I } =$ $x _ { i _ { 1 } } \cdots x _ { i _ { m } }$ . The slimmer may delete bases, but it may not substitute, insert, reuse a source position, reorder, or invert them.

The BPNet oracle model expects a fixed input length $L _ { \mathrm { m i n } }$ that is larger than m in our evaluations. For any sequence y with $| y | \leq L _ { \mathrm { m i n } } .$ , Embed(y, b) replaces the centered $| y | – \mathsf { b p }$ region of an $L _ { \mathrm { m i n } } – \mathrm { b p }$ background b with $y .$ . When $| y | = L _ { \mathrm { m i n } }$ , Embed $( y , b ) = y$ . All evaluation start sequences have $L = \mathrm { ^ { - } 1 , 1 8 8 } < L _ { \mathrm { m i n } }$ . Let $\mathcal { E } ( z )$ denote the oracle energy of input z, where lower is better, and let $B _ { \mathrm { d e s i g n } }$ be the set of backgrounds available during optimization; it is a singleton in this evaluation. The design problem is

$$
I ^ { * } = \operatorname * { a r g m i n } _ { 1 \leq i _ { 1 } < \cdots < i _ { m } \leq L } \frac { 1 } { | \mathcal { B } _ { \mathrm { d e s i g n } } | } \sum _ { b \in \mathcal { B } _ { \mathrm { d e s i g n } } } \mathcal { E } ( \mathrm { E m b e d } ( x _ { i _ { 1 } } \cdot \cdot \cdot x _ { i _ { m } } , b ) ) .\tag{1}
$$

This two-part output converts the deletion-only requirement into a directly verifiable, executable contract. The evaluator independently reconstructs $x _ { I }$ from the source and rejects the output unless $\widehat { \boldsymbol { x } } = \boldsymbol { x } _ { I }$ . Substitutions and insertions fail this reconstruction check. Duplicated, out-of-range, or nonincreasing indices fail the index checks, while reordering or inversion fails the index or reconstruction check.

Alternatives and benchmark choices. We considered allowing length at most m (rather than exactly m) or permitting arbitrary substitutions. Both depart from the intended behavior of a slimmer: at-most length confounds algorithms that return different sizes, and allowing substitutions incentivizes the model to favor them over deletions. We therefore require exactly m increasing source indices and evaluate several fixed values of m to obtain a length–activity curve.

Context and activity. Rather than pad with ambiguous bases or flank with random sequences, we center each candidate in low-activity backgrounds mined from the human genome. Design and test backgrounds are disjoint, and every selected background has lower predicted activity than every embedded start. For minimized energy E, held-out background $b ,$ source $x ,$ and candidate $x _ { I }$ , the

primary metric is

$$
R = \frac { \mathcal { E } ( b ) - \mathcal { E } ( \mathrm { E m b e d } ( x _ { I } , b ) ) } { \mathcal { E } ( b ) - \mathcal { E } ( \mathrm { E m b e d } ( x , b ) ) } .\tag{2}
$$

Thus $R = 1$ retains the source’s predicted lift over background and $R > 1$ improves it. Raw candidate energy was rejected as the primary measure because its scale and baseline vary across starts and contexts.

Compute and uncertainty. We follow the NucleoBench protocol and use a wall-clock limit because batching, gradients, and proposal bookkeeping have different costs. Each designed candidate is evaluated in five fixed test backgrounds to measure robustness of the slimmed subsequence. We average the sequence scores across background contexts to obtain one value for each starting sequence, method, and target length. We then compare methods using paired differences on the same starting sequences and bootstrap over starting sequences to obtain nominal confidence intervals.

## 3 Agentic discovery of a deletion-only designer

Constraining ERA to produce valid DNA slimmers. We ran an ERA program-search task to produce a deletion-only sequence designer with the best R. We used hard checks in addition to naturallanguage instructions. ERA received the task prompt and GrAdaBeam as a starting program, and was asked to modify the designer for deletion-only sequence slimming. For every proposed program, the evaluator reconstructed the returned sequence from those indices, and rejected substitutions, rearrangements, and candidates above the requested length. For accepted subsequences, it centered each variable-length sequence in one target-specific low-activity background before scoring it. The instructions prioritized reaching the length constraint before optimizing activity. Together, these choices prevented ERA from “cheating” the deletion-only constraint, while leaving it free to explore algorithm space.

ERA program search. Each proposed designer was evaluated on one high-scoring ATAC starting sequence and one high-scoring MYC starting sequence. For each case, the designer optimized the corresponding BPNet-lite oracle using that target’s low-activity background. The returned sequence was scored with the same target-specific oracle, and the two resulting energies were averaged to produce the scalar objective supplied to ERA. Failures, timeouts, and invalid subsequences received the worst objective value, while successful programs were expanded through ERA’s program-search procedure.

The discovered slimmer. The best program, GRADASLIM, maintains a salience value for each source position. Above the target length, it repeatedly removes approximately half of the remaining length gap, sampling deletions preferentially from low-salience positions and keeping the best of 64 proposals. At exact length, it proposes membership swaps, single-index shifts, and contiguous block shifts. It updates salience by crediting the positions retained in better-than-average candidates and diffuses a small amount of credit to neighboring positions. A cooling simulated-annealing rule occasionally accepts worse exact-length states, while an archive preserves the best valid candidate [14]. ERA’s resulting program combines rapid deletion, exact-length refinement, and persistent empirical salience learned from scored candidate batches. The search and constants are reported in Appendix A.

## 4 Five-target BPNet evaluation

Protocol. The evaluation consists of slimming tasks for transcription factors E2F3, ELF4, MAX, MECOM, and RAD21, none of which were used during ERA program search. Each study compares random exact deletion, batched greedy deletion with exact-length swap refinement, and ERA GRADASLIM on five high-activity 1,188-bp starts at target lengths 400 and 100 bp. Every evaluation run receives a 600-second wall-clock budget on an H200 GPU to produce the best target-length sequence. Each target has one fixed target-specific design context visible during optimization. Thus, each study contains 30 unique design cells and 150 background-level evaluation rows. The five test-context scores are averaged first, yielding one value per start, method, and length. Method differences are paired by start, and bootstrap intervals resample the five starts. Each target and length is reported separately.

Table 1: Retained effect (higher is better). Means average five starts after averaging five test contexts within each start. ERA−greedy intervals and wins are paired by start.
<table><tr><td>Target</td><td>bp</td><td>Random</td><td>Greedy</td><td>ERA</td><td>ERA—greedy [95% CI]</td><td>Wins/5</td></tr><tr><td>E2F3</td><td>400</td><td>0.862</td><td>3.918</td><td>5.735</td><td>1.817 [1.320, 2.321]</td><td>5/5</td></tr><tr><td>E2F3</td><td>100</td><td>0.613</td><td>3.012</td><td>2.713</td><td>-0.299 [-0.516, -0.072]</td><td>1/5</td></tr><tr><td>ELF4</td><td>400</td><td>0.636</td><td>1.533</td><td>1.999</td><td>0.466 [0.398, 0.533]</td><td>5/5</td></tr><tr><td>ELF4</td><td>100</td><td>0.348</td><td>1.083</td><td>1.152</td><td>0.069 [-0.009, 0.129]</td><td>4/5</td></tr><tr><td>MAX</td><td>400</td><td>1.068</td><td>1.810</td><td>2.211</td><td>0.401 [0.321, 0.484]</td><td>5/5</td></tr><tr><td>MAX</td><td>100</td><td>0.748</td><td>1.448</td><td>1.519</td><td>0.071 [-0.027, 0.170]</td><td>3/5</td></tr><tr><td>MECOM</td><td>400</td><td>0.645</td><td>2.474</td><td>4.065</td><td>1.590 [1.412, 1.785]</td><td>5/5</td></tr><tr><td>MECOM</td><td>100</td><td>0.385</td><td>1.781</td><td>1.919</td><td>0.138 [-0.026, 0.322]</td><td>3/5</td></tr><tr><td>RAD21</td><td>400</td><td>0.475</td><td>1.148</td><td>1.754</td><td>0.606 [0.365, 0.848]</td><td>5/5</td></tr><tr><td>RAD21</td><td>100</td><td>0.280</td><td>0.955</td><td>1.142</td><td>0.187 [-0.008, 0.367]</td><td>4/5</td></tr></table>

Results. ERA has the highest mean retained effect in nine of ten settings and exceeds random deletion in all ten. At 400 bp, ERA outperforms greedy for every target and all 25 paired starts, with every target-specific interval above zero. At 100 bp, ERA is numerically higher for ELF4, MAX, MECOM, and RAD21, but all four intervals overlap zero. Note that this regime is significantly shorter than what ERA used during program search (ERA focused on slimming to 300 bp). Greedy outperforms ERA for E2F3 at 100 bp (mean difference −0.299, 95% CI [−0.516, −0.072]). ERA’s mean retained effect exceeds 1 in every setting, indicating that its slimmed sequences improve the predicted lift over background relative to their full-length sources.

## 5 Limitations and conclusions

The evaluation design has five starts and one optimizer seed per target and tests only two output lengths. Targets were selected using results from NucleoBench, and the starts were selected for high activity. Contexts are fixed and target-specific. The same target oracle is used for optimization and evaluation, so the study does not test an orthogonal model; there is also no wet-lab validation. ERA program search did not include 100-bp targets, so the 100-bp evaluation tests substantially more aggressive slimming than was used during algorithm discovery. The frequent R > 1 values, including for 100-bp sequences, may reflect exploitation of the BPNet oracle or artificial deletion junctions rather than preservation of biological function. Bootstrap intervals are based on only five starting sequences per target and therefore provide limited-resolution uncertainty estimates.

Together, this work establishes deletion-only slimming as a distinct, machine-verifiable design problem. The preliminary results show a consistent ERA advantage at 400 bp and mixed performance at 100 bp.

## References

[1] Ziga Avsec, Melanie Weilert, Avanti Shrikumar, et al. Base-resolution models of transcriptionfactor binding reveal soft motif syntax. Nature Genetics, 53:354–366, 2021. doi: 10.1038/ s41588-021-00782-6.

[2] Eser Aygun, Anastasiya Belyaeva, Gheorghe Comanici, et al. An ai system to help scientists write expert-level empirical software. Nature, 654:909–916, 2026. doi: 10.1038/ s41586-026-10658-6.

[3] Sager J. Gosai, Rodrigo I. Castro, Natalia Fuentes, et al. Machine-guided design of celltype-targeting cis-regulatory elements. Nature, 634:1211–1220, 2024. doi: 10.1038/ s41586-024-08070-z.

[4] Johannes Linder and Georg Seelig. Fast activation maximization for molecular sequence design. BMC Bioinformatics, 22, 2021. doi: 10.1186/s12859-021-04437-5.

[5] Francheska López-Rivera, Olivia K. Foster Rhoades, Ben J. Vincent, Edward C. G. Pym, Meghan D. J. Bragdon, Javier Estrada, Angela H. DePace, and Zeba Wunderlich. A mutation in the Drosophila melanogaster eve stripe 2 minimal enhancer is buffered by flanking sequences. G3: Genes, Genomes, Genetics, 10(12):4473–4482, 2020. doi: 10.1534/g3.120.401777.

[6] Carlos A. Martinez, Kenneth Barr, Ah-Ram Kim, and John Reinitz. A synthetic biology approach to the development of transcriptional regulatory models and custom enhancer design. Methods, 62(1):91–98, 2013. doi: 10.1016/j.ymeth.2013.05.014.

[7] Alexander Novikov, Ngan Vu, Marvin Eisenberger, et al. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025. URL https: //arxiv.org/abs/2506.13131.

[8] Nikoletta Psatha, Peter Sova, Grigorios Georgolopoulos, et al. Large-scale discovery of potent, compact and erythroid specific enhancers for gene therapy vectors. Nature Communications, 16:4325, 2025. doi: 10.1038/s41467-025-59235-x.

[9] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, et al. Mathematical discoveries from program search with large language models. Nature, 625:468–475, 2024. doi: 10.1038/s41586-023-06924-6.

[10] Jacob Schreiber, Yang Young Lu, and William Stafford Noble. Ledidi: Designing genomic edits that induce functional activity. bioRxiv, 2020. doi: 10.1101/2020.05.21.109686.

[11] Joel Shor, Erik Strand, and Cory Y. McLean. Gradabeam: Combining model gradients with evolutionary search for generalizable nucleic acid design. bioRxiv, 2026. doi: 10.1101/2025.06. 20.660785.

[12] Sam Sinai, Richard Wang, Alexander Whatley, Stewart Slocum, Elina Locane, and Eric D. Kelsic. Adalead: A simple and robust adaptive greedy search algorithm for sequence design. arXiv preprint arXiv:2010.02141, 2020. URL https://arxiv.org/abs/2010.02141.

[13] Ibrahim I. Taskiran, Katina I. Spanier, Hannah Dickmänken, et al. Cell-type-directed design of synthetic enhancers. Nature, 626:212–220, 2024. doi: 10.1038/s41586-023-06936-2.

[14] Peter J. M. van Laarhoven and Emile H. L. Aarts. Simulated Annealing: Theory and Applications. Springer, 1987. doi: 10.1007/978-94-015-7744-1.

## A Complete discovered algorithm

Algorithm 1 gives the exact-length evaluation implementation of GRADASLIM. ERA discovered its proposal and credit-assignment rules.

## Algorithm 1: ERA-discovered GRADASLIM

Require: source x of length L; target m; minimized oracle f; budget T; seed q; batch size $B = 6 4$   
1: t ← CLOCK   
2: initialize isolated RNG with q   
3: $s _ { i } \sim \mathcal { N } ( 0 , 1 0 ^ { - 5 } )$ for $i = 1 , \ldots , L$   
4: score exact fallback $F = ( \overset { \cdot } { 1 } , \dots , m ) ; ( I ^ { * } , e ^ { * } ) \gets ( F , f ( x _ { F } ) )$   
5: $I \gets ( 1 , \dots , L ) ; e \gets f ( { \boldsymbol { x } } _ { I } )$   
6: while $\mathrm { C L O C K } - t _ { 0 } < T$ do   
7: $\mathbf { i f } \mid I \mid > m$ then ▷ Phase 1: reach the constraint quickly   
8: $\dot { d } \gets \operatorname* { m a x } \{ 1 , \lceil ( | I | - m ) / 2 \rceil \}$   
9: for $j = 1 , \cdot \cdot \cdot , \stackrel { \cdot } { B }$ do   
10: $\begin{array} { r } { w _ { k } \gets \operatorname* { m a x } _ { r \in I } s _ { r } - s _ { k } + 0 . 1 \operatorname { s d } ( s _ { I } ) + 1 0 ^ { - 7 } } \end{array}$   
11: sample d distinct positions $D _ { j } \subset I$ proportional to w   
12: $J _ { j }  I \backslash D _ { j }$   
13: end for   
14: else ▷ Phase 2: exact-length refinement   
15: $U \gets \{ 1 , \dots , L \} \setminus I$   
16: for $j \doteq 1 , \ldots , \dot { B }$ do   
17: $J _ { j }  I { \rangle }$ draw r ∼ Uniform(0, 1)   
18: $\mathbf { i f } r < 0 . 3 5$ and $U \neq \emptyset$ then ▷ salience-biased swap   
19: sample $a \in I$ proportional to max ${ ( s _ { I } ) - s _ { a } + 1 0 ^ { - 7 } }$   
20: sample $u \in U$ proportional to $s _ { u } - \mathrm { m i n } ( s _ { U } ) + 1 0 ^ { - 7 }$   
21: $J _ { j } \gets \mathrm { s o r t } ( ( I \setminus \{ \bar { a } \} ) \cup \{ u \} )$

22: else if $r < 0 . 6 5$ then ▷ point shift   
23: sample retained rank k uniformly   
24: replace $I _ { k }$ uniformly within its order-preserving neighbor bounds   
25: else ▷ contiguous block shift   
26: sample a valid block start and a block length $\ell \in \{ 1 , \ldots , 1 1 \}$   
27: shift the block uniformly within its order-preserving bounds   
28: end if   
29: end for   
30: end if   
31: $e _ { j } \gets f ( x _ { J _ { j } } ) \operatorname { f o r } j = 1 , \dots , B$ in one batch; $\bar { e }  B ^ { - 1 } \sum _ { j } e _ { j }$   
32: $e ^ { * , \mathrm { o l d } }  e ^ { * }$ ▷ archive best before this iteration   
33: for $j = 1 , \dots , B$ and $k \in J _ { j }$ do   
34: $s _ { k }  s _ { k } + ( \bar { e } - e _ { j } )$ ▷ credit bases in good candidates   
35: end for   
36: for $k = 2 , \ldots , L - 1$ do   
37: $s _ { k }  s _ { k } + 0 . 0 5 ( s _ { k - 1 } + s _ { k + 1 } )$ ▷ neighbor diffusion   
38: end for   
39: $j ^ { * } \gets \arg \operatorname* { m i n } _ { j } e _ { j }$   
40: $\mathbf { \dot { i } } \mathbf { f } \left| I \right| > m$ then   
41: $\left( I , e \right) \gets \left( J _ { j ^ { * } } , e _ { j ^ { * } } \right)$   
42: else   
43: $p \gets \operatorname* { m i n } \{ 1 , ( \mathbf { C } \mathrm { L O C K } - t _ { 0 } ) / T \} ; \tau \gets \operatorname* { m a x } \{ 0 . 0 0 1 , 0 . 4 ( 1 - p ) ^ { 3 } \}$   
44: i $\mathfrak { i } e _ { j ^ { * } } < \dot { e } \mathrm { o r } \mathrm { U n i f o r m } ( 0 , 1 ) < \mathrm { e x p } ( - ( e _ { j ^ { * } } \overset { \cdot } { - } e ) / \tau )$ then   
45: $\mathbf { \Phi } ^ { \prime } ( I , e ) \gets ( J _ { j ^ { * } } , e _ { j ^ { * } } )$   
46: end if   
47: end if   
48: if any $| J _ { j } | = m$ and $e _ { j } < e ^ { * } ,$ , set $( I ^ { * } , e ^ { * } )$ to the best such $( J _ { j } , e _ { j } )$   
49: if $| I | = m$ and $e < e ^ { * , \mathrm { o l d } }$ then ▷ reinforce a newly accepted global best   
50: $\dot { s } _ { I }  s _ { I } + 0 . 1 | e |$   
51: end if   
52: end while   
53: return $( x _ { I ^ { * } } , I ^ { * } )$

Interpretation. The salience vector is a persistent credit-assignment reservoir over source coordinates, not a model gradient. For each scored batch, every retained index receives reward $\bar { e } - e _ { j } { : }$ positions occurring in below-average-energy candidates gain salience. Phase 1 uses the inverse of this value to target deletions. Phase 2 removes low-salience retained positions and restores high-salience omitted positions, while point and block shifts search local source geometry. Neighbor diffusion biases nearby bases together, which may preserve motifs but may also amplify correlated artifacts. Simulated annealing allows the single incumbent to cross local barriers, whereas the archive prevents loss of the best exact solution.