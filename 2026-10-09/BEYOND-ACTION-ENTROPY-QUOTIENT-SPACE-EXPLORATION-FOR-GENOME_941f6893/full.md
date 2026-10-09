# BEYOND ACTION ENTROPY: QUOTIENT-SPACE EXPLORATION FOR GENOME-SCALE METABOLIC MODEL REPAIR

Xuan Gong<sup>⋆,1</sup>, Hanbo Huang<sup>⋆,1</sup>, Wenbin Dai<sup>⋆,1</sup>, Jing Wang<sup>1</sup>, Lei Bai<sup>2</sup>, Xiang Xiao<sup>1</sup>, Weishu Zhao<sup>1</sup>, Shiyu Liang<sup>†,1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Shanghai AI Laboratory {gongxuan0610, lsy18602808513}@sjtu.edu.cn

## ABSTRACT

Repairing scientific models from functional observations differs fundamentally from supervised prediction: feedback may certify a solution without revealing which structural correction is responsible. We study this setting for genome-scale metabolic model (GEM) repair, where multiple reaction edits can explain the same phenotypes and many apparently distinct edits correspond to the same biological mechanism. This many-to-one structure creates a hidden failure mode for conventional exploration: diversity in the output space need not translate into diversity of scientific hypotheses. We introduce QUOTIENTPO, which collapses equivalent repairs into canonical mechanisms and optimizes exploration directly over the resulting quotient space. To make quotient exploration informative under finite rollouts, we derive a kernelized Renyi estimator that resolves graded crowding among´ distinct repair cores beyond coarse exact-match counts. On 2,212 held-out GEMs, QUOTIENTPO improves Success@32 from 17.93% to 20.10% (+12.1% relative) while consistently increasing distinct successful-core discovery under the same sampling budget. These results establish quotient-space exploration as a principled approach to mechanism-level discovery under verifier-induced equivalence.

## 1 INTRODUCTION

AI4Sci (Wang et al., 2023) is increasingly moving beyond predicting scientific observations toward reasoning over, generating, and modifying scientific models and structures (Abramson et al., 2024; Merchant et al., 2023). Unlike ordinary prediction targets, these scientific objects act as hypotheses about underlying mechanisms whose validity is assessed indirectly through physical laws (Karniadakis et al., 2021) or experimental feedback (Boiko et al., 2023). Evaluating candidates against functional constraints rather than gold targets characterizes verifier-guided scientific AI, posing the core challenge of learning structural corrections from indirect evidence alone. This challenge directly materializes in genome-scale metabolic models (GEMs) (Mardinoglu & Palsson, 2025), which operationalize genomic sequences into executable metabolic machinery. To overcome this barrier, we introduce phenotype-supervised GEM repair: an amortized framework that replaces costly manual curation to scale predictive mechanistic modeling to uncharacterized organisms.

This repair paradigm is fundamentally challenged by partial identification: phenotype observations validate models without uniquely determining underlying mechanisms (Raue et al., 2009; Benedict et al., 2014). This underdetermination creates a core tension: while multiple distinct mechanisms can satisfy identical phenotypes, combinatorial edits often implement the exact same mechanism, rendering surface diversity deceptive. Prior work addresses non-surface redundancy via semantic classes, kernels, or symmetries (Kuhn et al., 2023; Nikitin et al., 2024; Kim et al., 2025; Li et al., 2026). Yet in verifier-guided repair, a single functional intervention admits diverse raw realizations, meaning output similarity cannot define which candidates constitute the same scientific hypothesis.

![](images/2f383bdaa35cc72b1e6b27c5b2faf78883ae652bdd07de7dc4f457f73d05e0e6.jpg)  
Figure 1: Overview. QUOTIENTPO canonicalizes equivalent surface edits to explore distinct mechanistic hypotheses rather than redundant syntactic variants for GEM repair.

Here, identity is induced operationally by the verifier itself. This raises a key question: How should exploration be structured when diversification is defined up to verifier-preserving transformations?

Our answer is to quotient before exploring. Conceptually, we partition repair edits into verifierpreserving equivalence classes through canonicalization, establishing scientific identity instead of syntactic multiplicity as the unit of diversity. Computationally, exact identity provides only a coarse finite-sample exploration signal, becoming especially degenerate when exact mechanism collisions are sparse. We therefore relax binary collision counting into a graded, kernelized Renyi entropy´ objective (Principe et al., 2000; Giraldo et al., 2014), allowing mechanistic crowding to be detected even among non-identical repairs. Based on these principles, we introduce QUOTIENTPO (Fig. 1), an algorithm combining group-relative policy optimization with quotient-space exploration. Together, our theory and experiments establish quotient-space exploration as a principled route to broader valid-mechanism coverage, advancing a Sci4AI perspective on verifier-guided learning.

Our contributions are threefold:

• Formulation and Benchmark for Scientific Repair: We formulate phenotype-supervised GEM repair to generalize edits across organisms without test-time phenotypes, and establish the first large-scale benchmark for amortized GEM correction.

• Theoretical Foundation of Quotient Exploration: We identify representation multiplicity as an intrinsic failure mode in which standard policy entropy rewards syntactic redundancy, and establish theoretical connections between quotient-space diversity and deployed hypothesis coverage.

• Tractable Algorithm and Empirical Validation: We develop QUOTIENTPO, making quotient exploration practical via kernelized Renyi estimation of graded mechanistic crowding, and demon-´ strate consistent gains in multi-budget repair and distinct mechanism recovery.

## 2 BACKGROUND AND RELATED WORK

Genome-scale metabolic reconstruction and model repair. GEMs support mechanistic phenotype simulation and metabolic engineering, yet incomplete biological knowledge and reconstruction choices can leave missing and spurious reactions (Thiele & Palsson, 2010; Machado et al., 2018; Zimmermann et al., 2021; Hsieh et al., 2024; Huang et al., 2026). Classical constraint-based methods repair each GEM independently through gap filling or phenotype reconciliation, without sharing repair knowledge across organisms. The same phenotypic constraints may admit multiple valid repairs (Satish Kumar et al., 2007; Kumar & Maranas, 2009; Thiele et al., 2014; Hartleb et al., 2016). Recent learning-based methods recover missing reactions through link or hypergraph prediction (Yadati et al., 2020; Chen et al., 2023; Huang et al., 2024; Liu et al., 2024; Zhao et al., 2026), but largely target synthetically removed reactions rather than functional bidirectional repair. We instead learn a shared repair policy across organisms, using phenotypes only as training supervision to predict functionally validated bidirectional edits without test-time phenotype access.

Diversity-aware reinforcement learning for structured prediction. Reinforcement learning (RL) optimizes structured outputs under non-differentiable functional rewards, including molecular design (You et al., 2018), executable code (Le et al., 2022), and outcome-based reasoning (Shao et al., 2024). Diversity has been pursued through maximum-entropy policies (Haarnoja et al., 2018), unsupervised skill discovery (Eysenbach et al., 2018; Sharma et al., 2020), quality-diversity optimization (Grillotti et al., 2024), and GFlowNets for reward-weighted object generation (Bengio et al., 2021; Jain et al., 2022). Recent work further considers semantic equivalence and kernelized set diversity (Kuhn et al., 2023; Li et al., 2026). However, these approaches do not resolve the verifier-induced many-to-one mapping central to GEM repair, where distinct edit sets can collapse to the same repair core. Raw-output diversity can therefore reward representation multiplicity without expanding distinct repair hypotheses, motivating quotienting before exploration.

## 3 PRELIMINARIES AND PROBLEM FORMULATION

Genome-scale metabolic models. A GEM represents an organism’s metabolic network as a directed bipartite graph $M = ( \mathcal { C } , \mathcal { R } , S )$ between metabolites C and reactions R (Klamt et al., 2009; Smart et al., 2008; Basler et al., 2011). The stoichiometric matrix $S \in \mathbb { R } ^ { | \mathcal { C } | \times | \mathcal { R } | }$ defines weighted edges between them, specifying the consumption and production coefficients of each reaction.

Formulation of GEM repair. We frame GEM correction as constrained graph editing through reaction additions and deletions. Given an initial model $M _ { 0 }$ with reaction set $\mathcal { R } _ { 0 }$ and a candidate reaction universe U, a repair $E = ( A , D )$ adds $A \subseteq \mathcal { U } \setminus \mathcal { R } _ { 0 }$ and removes $D \subseteq \mathcal { R } _ { 0 } \cap \mathcal { U }$ , yielding the repaired model $M _ { E }$ . For each tested phenotype i, flux balance analysis (FBA;Orth et al., 2010) over its nutrient-specific assay(s) computes optimal biomass flux $v _ { i } ^ { * } ( \dot { M } _ { E } )$ , giving predicted phenotype $\widehat { y } _ { i } ( M _ { E } ) = \mathbb { I } \big [ \boldsymbol { v } _ { i } ^ { * } ( M _ { E } ) > \overline { { \delta } } \big ]$ under growth threshold $\delta > 0$ . Given observed BacDive phenotypes $y \in$ $\{ 0 , 1 \} ^ { N }$ , repair quality is measured by phenotype consistency $\begin{array} { r } { \Phi _ { y } ( M _ { E } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } [ \widehat { y } _ { i } ( M _ { E } ) = y _ { i } ] } \end{array}$ FBA therefore serves directly as the functional verifier that maps a repaired GEM to phenotype predictions. The finite repair space $\mathcal { F }$ contains all legal reaction edits over candidate universe U.

Learning problem. Let G denote the organism-specific protein representation derived from its genome-encoded proteins. We assume a training distribution D over organisms, where for each training instance $\begin{array} { r } { ( \boldsymbol { \bar { M } _ { 0 } } , \boldsymbol { G } , \boldsymbol { y } ) \sim \mathcal { D } , } \end{array}$ y contains the available wet-lab phenotypes. Neither a corrected GEM nor a gold reaction edit set is available, and the observed phenotypes generally do not uniquely identify the underlying repair. We therefore learn a stochastic repair policy $\overline { { { \pi } } } _ { \boldsymbol { \theta } } ( E \mid \dot { M } _ { 0 } , G )$ over valid repairs. Importantly, the phenotypes y provide training supervision through $\Phi _ { y } ( M _ { E } )$ only and are never given as model input. The resulting learning objective is

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( M _ { 0 } , G , y ) \sim \mathcal { D } } \mathbb { E } _ { E \sim \pi _ { \theta } ( \cdot \vert M _ { 0 } , G ) } \left[ \Phi _ { y } ( M _ { E } ) \right] \quad \mathrm { s . t . } \quad E \in \mathcal { F } .\tag{1}
$$

At test time, only $( M _ { 0 } , G )$ is available and the learned policy outputs a repair E and the corresponding $M _ { E }$ . Because no organism-specific corrected GEM or gold edit set is available, evaluation uses only held-out phenotype consistency target.

## 4 METHODOLOGY

The formulation in Eq. (1) directly motivates policy-gradient $\mathbf { R L } ^ { 1 }$ with explicit multi-solution exploration: (1) Bypassing non-differentiable verifiers and solver bias: Repair quality is evaluated by non-differentiable FBA-based verifiers, preventing direct backpropagation. While offline mixedinteger linear programming (MILP) (Satish Kumar et al., 2007; Benedict et al., 2014; Schroeder & Saha, 2020) can provide demonstrations for supervised fine-tuning (SFT), it arbitrarily breaks ties across non-unique feasible sets, biasing SFT toward solver heuristics. We therefore adopt RL to optimize verifier rewards directly beyond offline demonstrations. (2) Preventing collapse under partial identification: Across a combinatorial graph edit space, observed phenotypes only partially identify the metabolic network, creating a landscape with multiple discrete optima where mechanis tically distinct repairs achieve identical rewards. A policy collapsed onto any single valid repair is already reward-optimal, leaving standard reward maximization with zero gradient signal to discover alternative mechanisms. Under finite rollout budgets, this wastes samples on redundant variation around a single mode, motivating explicit exploration to preserve diverse scientific hypotheses.

![](images/d52490187e55b477539c1054cfc2075c8113972879132e932ab596cff480537e.jpg)  
Figure 2: Theoretical intuition. (a) Each core m contains $n _ { m }$ equivalent raw edits contracted by κ into a single point, preventing large $n _ { m }$ from distorting exploration. (b) Bar heights show exploration credit. Exact matching gives identical credit to equal-count cores, whereas kernel similarity favors isolated discoveries $( m _ { 1 } )$ over clustered candidates $( m _ { 2 } \ – m _ { 4 } )$

## 4.1 REPRESENTATION-INVARIANT EXPLORATION WITH QUOTIENT ENTROPY

Action entropy rewards representation multiplicity, not mechanistic diversity. A natural exploration bonus is the Shannon entropy over raw repair actions $H ( \pi _ { \theta } )$ , leading to the standard entropy-regularized objective $\mathcal { T } _ { \mathrm { a c t } } ( \theta ) \overset {  } { = } \mathbb { E } _ { ( M _ { 0 } , G , y ) \sim \mathcal { D } } \mathbf { \hat { \mathbb { E } } } _ { E \sim \pi _ { \theta } } [ \Phi _ { y } ( M _ { E } ) ] ^ { \sim } + \beta H ( \hat { \pi _ { \theta } } ) ]$ , where $\beta > 0$ controls the exploration strength. The problem is that entropy counts representations, not repair hypotheses. If two equally rewarded hypotheses differ only in that one admits n equivalent action realizations, action entropy gives it an additional preference of β log n. Thus, a hypothesis is favored purely by representation redundancy, rather than any distinct biological mechanism. We formalize this multiplicity bias in Appendix B.2.

Exploring distinct mechanisms requires quotienting equivalent repairs. To remove this bias, equivalent repairs should contribute only once to exploration. We define a deterministic, verifierpreserving map $\kappa _ { y }$ that removes redundant edits, contracting the $n _ { m }$ equivalent raw realizations into a canonical repair core $m = \kappa _ { y } ( E )$ (Fig. 2a). Let $\mathcal { M } _ { y }$ denote the set of repair cores, serving as operational representations of underlying mechanisms. Repairs sharing the same core are treated as equivalent hypotheses, and the policy induces the quotient distribution $q = q _ { \theta , y } ,$ , where $q ( m ) =$ $\scriptstyle \sum _ { E : \kappa _ { y } ( E ) = m } { \bar { \pi _ { \theta } ( E \mid M _ { 0 } , G ) } }$ aggregates probability mass across all equivalent raw realizations.

Theorem 4.1 (Within-core invariance characterizes quotient exploration). An explorationfunctional D is invariant to within-core redistribution ifand only ifit depends on the raw policy π only through its induced core distribution $q _ { \pi } .$ . Equivalently, there exists afunctional D such that ${ \mathfrak { D } } ( \pi ) = { \overline { { \mathfrak { D } } } } ( q _ { \pi } )$

Theorem 4.1 formalizes why exploration should depend only on the induced distribution over repair cores. The intuition is direct: splitting probability across equivalent realizations inflates action entropy, whereas the aggregated core probability $q ( m )$ remains unchanged. Evaluating entropy over this aggregated distribution removes multiplicity bias by construction, leading to the quotientregularized objective: $\begin{array} { r } { \mathcal { T } _ { \mathrm { q u o t } } ( \theta ) = \mathbb { E } _ { ( M _ { 0 } , G , y ) \sim \hat { \mathcal { D } } } \left[ \mathbb { E } _ { E \sim \pi _ { \theta } } \left[ \hat { \Phi _ { y } } ( M _ { E } ) \right] + \beta H ( q ) \right] } \end{array}$ , where the quotient entropy is $\begin{array} { r } { H ( q ) = - \sum _ { m \in \mathcal { M } _ { u } } q ( m ) \log { q ( m ) } } \end{array}$ . Crucially, optimizing $\mathcal { J } _ { \mathrm { q u o t } }$ weights distinct hypotheses strictly by task utility rather than raw representation multiplicity (Appendix B.2).

## 4.2 GRADED QUOTIENT ESTIMATION WITH RENYI ´ KERNELS

Exact matching gives coarse finite-sample quotient estimates. Because the core probability $q ( m )$ is an unobserved fiber mass rather than a closed-form likelihood, the quotient diversity must be estimated from sampled cores. Exact-frequency estimation, however, provides only binary crowding information: structurally different cores with the same empirical multiplicity receive identical credit. This limitation becomes extreme in the sparse-collision regime. With L sampled cores, the expected number of exact pairwise collisions is $\textstyle { \binom { \bar { L } } { 2 } } \sum _ { m } q ( m ) ^ { 2 }$ . When $\begin{array} { r } { \binom { L } { 2 } \sum _ { m } q ( m ) ^ { 2 } \ll 1 } \end{array}$ , a rollout group contains few or no repeated cores, making distinct samples indistinguishable under exact matching (Appendix B.3). Thus, exact-count estimation provides only a coarse finite-sample exploration signal, even before collisions vanish entirely.

Renyi kernels turn binary identity into graded mechanistic crowding.´ The key limitation is the knife-edge nature of exact matching: it distinguishes identical from non-identical cores, but provides no graded notion of crowding among distinct repairs. Indeed, the exact pairwise collision probabil-$\mathrm { i t y } \sum _ { m } q ( m ) ^ { 2 }$ is precisely the quadratic Renyi concentration´ $\exp [ - H _ { 2 } ( q ) ]$ (Principe et al., 2000; Giraldo et al., 2014). This suggests replacing quotient Shannon entropy with quadratic Renyi en-´ tropy, whose exact collision form admits a graded relaxation while preserving the quotient identity. Representing each repair core as a signed edit set, we replace the exact-match collision operator with the graded kernel $\bar { k ( m , m ^ { \prime } ) } = \mathrm { e x p } ( \bar { - } ( 1 - J ( m , m ^ { \prime } ) ) / \bar { h } )$ , where J is Jaccard similarity and $h > 0$ controls smoothness. $\mathrm { \bf A s } h  0 ,$ k recovers exact matching, whereas finite h assigns progressively larger collision mass to increasingly overlapping repairs. This kernel relaxation exposes mechanistic crowding before exact duplicates dominate (Figure 2b), discounting exploration credit for clustered repairs while rewarding genuinely distinct bypasses. Replacing exact-collision Renyi entropy with´ the kernelized surrogate $\mathbf { \bar { { H } } } _ { 2 , k } ( q ) = - \log \mathbb { E } _ { m , m ^ { \prime } \sim q } [ k ( m , m ^ { \prime } ) ]$ gives the crowding-aware objective:

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { r e n y i } } ( \theta ) = \mathbb { E } _ { ( M _ { 0 } , G , y ) \sim \mathcal { D } } \left[ \mathbb { E } _ { E \sim \pi _ { \theta } } \big [ \Phi _ { y } ( M _ { E } ) \big ] + \beta H _ { 2 , k } ( q ) \right] . } \end{array}\tag{2}
$$

Kernelized Renyi estimation therefore preserves the quotient notion of identity while replacing´ coarse binary collisions with a graded finite-sample exploration signal.

## 4.3 QUOTIENTPO: QUOTIENT-SPACE POLICY OPTIMIZATION

QuotientPO operationalizes quotient exploration within GRPO by decomposing verifier feedback and quotient diversity into candidate-level credit. We optimize a verifier-shaped relaxation where each component provides a principled finite-sample realization of Eq. (2).

Verifier-shaped task reward. For a candidate repair $E ,$ we define the terminal task reward as $R _ { y } ( E ) = \bar { \Phi } _ { y } ( M _ { E } ) - \Phi _ { y } ( M _ { 0 } ) + 2 \mathbb { I } [ \mathrm { a l l }$ l satisfied]. Since $\Phi _ { y } ( M _ { E } ) - \Phi _ { y } ( M _ { 0 } ) \leq 1$ , the success weight of 2 enforces a strict margin ensuring that any fully verified repair $( R _ { y } \geq 2 )$ strictly outranks any partial improvement $( R _ { y } < 1 )$ . This rules out reward-level satisficing on partial repairs while retaining dense shaping rewards during early exploration.

Group-relative quotient credit. For each organism, we sample L repairs and canonicalize them into repair cores ${ m } _ { 1 } , \ldots , { m } _ { L }$ . While the kernelized Renyi entropy in Section 4.2 evaluates diversity´ collectively across the rollout, policy optimization requires candidate-level credit. A leave-one-out linearization of this group entropy around the empirical mean gives the normalized crowding cost

$$
c _ { i } = 2 L \frac { \sum _ { j \neq i } k ( m _ { i } , m _ { j } ) } { \sum _ { a \neq b } k ( m _ { a } , m _ { b } ) } - 2 , \qquad \sum _ { i = 1 } ^ { L } c _ { i } = 0 .\tag{3}
$$

A positive $c _ { i }$ indicates that a repair core is crowded by similar candidates, while a negative value indicates that it expands quotient-space coverage. The zero-sum form aligns this diversity signal with group-relative optimization. Its derivation is given in Appendix B.4.

QuotientPO objective. Let $\rho _ { i } ( \theta ) = \pi _ { \theta } ( E _ { i } \mid M _ { 0 } , G ) / \pi _ { \theta _ { \mathrm { o l d } } } ( E _ { i } \mid M _ { 0 } , G )$ , and let $A _ { i } = ( R _ { y } ( E _ { i } ) -$ $\bar { R } ) / ( \sigma _ { R } + \varepsilon )$ denote the group-relative advantage with sample mean R<sup>¯</sup> and standard deviation $\sigma _ { R }$ The complete objective is

$$
\mathcal { I } ( \theta ) = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \operatorname* { m i n } \Bigl ( \rho _ { i } ( \theta ) A _ { i } , \operatorname { c l i p } ( \rho _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } \Bigr ) - \frac { \beta } { L } \sum _ { i = 1 } ^ { L } \rho _ { i } ( \theta ) c _ { i } + \mathcal { R } _ { \mathrm { a u x } } ( \theta ) .\tag{4}
$$

The GRPO term optimizes verifier quality, while the quotient term shifts probability away from crowded cores toward distinct repair hypotheses. The auxiliary term $\mathcal { R } _ { \mathrm { a u x } }$ combines QPI credit for newly discovered successful cores and an optional redundancy penalty for edits outside the canonical core. More details are given in Appendix A.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Datasets. We pair strain-level metabolic phenotypes from BacDive (Schober et al., 2025) with matched NCBI (Sayers et al., 2022) proteomes, yielding 11,275 organisms spanning 2,454 genera and 40 phyla. Each proteome is reconstructed under 4 CarveMe (Machado et al., 2018) metabolic universes in the BiGG (King et al., 2016) namespace. Mapping and filtering phenotype annotations onto 16 core BiGG metabolite families produces 3,978 phenotype-consistent candidate GEMs for synthetic pretraining and 28,842 phenotype-inconsistent GEMs for post-training repair and evaluation. We first partition the post-training pool into train, validation, and test splits with an 85:5:10 ratio. Subsequent phenotype-assay filtering and certified-teacher selection leave 2,504/294 training/validation GEMs for SFT, while the RL split retains 9,196/1,087 GEMs and the final sealed test set contains 2,212 GEMs. All splits are species-disjoint, with 5 held-out genera demanding out-ofdistribution generalization rather than taxonomic interpolation. Throughout training and evaluation, BacDive phenotypes are never provided as policy inputs, but are used to generate expert repairs via phenotype-constrained MILP for SFT, define RL rewards, and evaluation (Appendix C).

Table 1: Main results. MILP is an oracle with test-time phenotype access. SFT methods use checkpoints selected on the validation split, while RL methods are evaluated at the final checkpoint after an identical 864-update training budget. All results are evaluated on the test set, reporting mean ± standard deviation across three seed runs.
<table><tr><td>Method</td><td>Greedy</td><td>Success@1</td><td>Success@8</td><td>Success@32</td><td>Improved (mean@32)</td><td>Success Cores@32</td></tr><tr><td>MILP (Oracle)</td><td>15.01</td><td></td><td></td><td></td><td>25.63</td><td>0.15</td></tr><tr><td>SFT-Hard</td><td> $1 . 7 2 _ { \pm 0 . 1 6 }$ </td><td> $0 . 5 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 9 6 _ { \pm 0 . 1 1 }$ </td><td> $4 . 4 9 _ { \pm 0 . 0 5 }$ </td><td> $1 . 6 4 _ { \pm 0 . 0 3 }$ </td><td> $0 . 0 9 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>SFT-Soft</td><td> $2 . 6 5 _ { \pm 0 . 3 5 }$ </td><td> $2 . 4 0 _ { \pm 0 . 0 5 }$ </td><td> $5 . 2 7 _ { \pm 0 . 4 1 }$ </td><td> $7 . 5 2 _ { \pm 0 . 4 9 }$ </td><td> $9 . 9 2 _ { \pm 0 . 1 8 }$ </td><td> $0 . 1 6 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>Vanilla GRPO</td><td> $7 . 9 7 _ { \pm 0 . 2 9 }$ </td><td> $7 . 4 3 _ { \pm 0 . 0 3 }$ </td><td> $1 3 . 7 9 _ { \pm 1 . 0 9 }$ </td><td> $1 7 . 9 3 _ { \pm 1 . 5 4 }$ </td><td> $3 0 . 2 1 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 0 1 }$ </td></tr><tr><td>MaxEnt</td><td> $7 . 2 9 _ { \pm 1 . 1 6 }$ </td><td> $5 . 9 7 _ { \pm 0 . 8 6 }$ </td><td> $1 2 . 6 9 _ { \pm 1 . 2 7 }$ </td><td> $1 7 . 1 2 _ { \pm 1 . 4 1 }$ </td><td> $2 9 . 3 5 _ { \pm 2 . 0 3 }$ </td><td> $0 . 4 8 _ { \pm 0 . 0 5 }$ </td></tr><tr><td>GFlowNet</td><td> $3 . 2 0 _ { \pm 0 . 2 3 }$ </td><td> $3 . 2 3 _ { \pm 0 . 1 5 }$ </td><td> $8 . 3 5 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $1 3 . 1 0 _ { \pm 0 . 1 7 }$ </td><td> $1 4 . 9 1 _ { \pm 0 . 4 9 }$ </td><td> $0 . 3 4 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>QuotientPO</td><td> $\mathbf { 8 . 4 7 _ { \pm 0 . 2 5 } }$ </td><td> $7 . 7 6 _ { \pm 0 . 2 3 }$ </td><td> $\mathbf { 1 5 . 3 0 _ { \pm 0 . 3 0 } }$ </td><td> $\mathbf { 2 0 . 1 0 _ { \pm 0 . 5 2 } }$ </td><td> ${ \bf 3 1 . 5 9 _ { \pm 0 . 8 2 } }$ </td><td> ${ \bf 0 . 5 7 } _ { \pm 0 . 0 2 }$ </td></tr></table>

Verifier. We define 18 standardized FBA assays across 16 BacDive nutrient families, where each tested nutrient serves as the sole elemental source against an elementdepleted background to preclude metabolic bypass (Fig. 3). Nutrients and oxygen uptake bounds are 10 and 20, respectively, with growth defined by biomass flux > $1 0 ^ { - 3 }$ . The terminal verifier rewards any repair reproducing the observed phenotypes.

![](images/4abac4d78a99e22c3835b08d8324a252f405e91270d86c6168490db8aaa72d25.jpg)  
Figure 3: Verifier design.

Evaluation Metrics. We use

Success@k(%) as our primary metric, defined as the probability that at least one of k sampled repairs achieves full phenotype recovery $( \Phi _ { y } ( M _ { E } ) \bar { = } 1 )$ , following multi-sample evaluation in RL and generative modeling (Hong et al., 2026; Chen et al., 2021; Feng et al., 2026). We report $k \in \{ 1 , 8 , \bar { 3 } 2 \}$ alongside greedy decoding (Greedy), the rate of partial progress $\Phi _ { y } ( M _ { E } ) > \Phi _ { y } ( \dot { M } _ { 0 } )$ (Improved), and the macro-averaged number of distinct canonical repair cores recovered among successful repairs within the same 32 samples (Success Cores@32).

Baselines. We compare against 6 baselines across 3 paradigms: (1) domain oracle: MILP with testtime phenotype access; (2) standard policy learning: imitation across probabilistic and discretethreshold architectures (SFT-Soft/Hard) alongside verifier RL (Vanilla GRPO); and (3) diversityoriented baselines: action-entropy GRPO (MaxEnt) and GFlowNet (Bengio et al., 2021) (rewardproportional raw-space sampling without quotient equivalence). Appendix D details all baselines.

Implementation Details. Following prior methods of GEM GNN modeling (Hasibi et al., 2024; Huang et al., 2024), we train a 92.85M-parameter GNN policy that fuses frozen ESM-2 protein embeddings (Lin et al., 2023) into the global GEM representation, predicting one-shot repair probabilities over a candidate universe of 28,301 reactions. For RL training, we initialize from SFT-Soft and train for 3 epochs using AdamW with a fixed learning rate of $2 e { \mathrm { - } } 6 ,$ group size $L = 3 2 ,$ , bandwidth $h = 0 . 2 5$ , and the quotient term weight $\beta = 0 . 0 2$ . All FBA simulations use HiGHS (Huangfu & Hall, 2015). See Appendix A and E for more details.

![](images/be85ebaa4dabaad8e930aec8ebbfcdfd442b19bb1738cd14ce7b04483d4c7b77.jpg)  
(a) Test-time scaling.

![](images/1267590b42dc88100cf0a4896cf9913c68680a8c0803670e65860c6d0a8aba44.jpg)  
(b) Training dynamics.  
Figure 4: Test-time scaling and training dynamics. (a) Repair success (Success@k) and distinct successful-core discovery (Success cores@k). (b) Validation curves across different RL methods.

## 5.2 MAIN RESULTS

Quotient exploration improves repair coverage. Table 1 shows a consistent separation between mechanism-aware exploration and raw-space diversity objectives. QuotientPO achieves the strongest performance across all learned metrics, improving Success@32 from 17.93% (Vanilla GRPO) to 20.10% and core discovery from 0.50 to 0.57. These absolute success rates reflect task difficulty: even the phenotype-informed MILP oracle, despite direct test-label access, reaches 15.01% full repair within its fixed 600 s search budget. In contrast, MaxEnt does not improve over vanilla, and GFlowNet underperforms despite explicitly encouraging diversity, confirming that syntactic variety fails to expand mechanistic diversity. The large gap over SFT further shows that supervised imitation alone cannot optimize verifier-defined outcomes.

The advantage emerges through test-time scaling. Figure 4a shows that quotient exploration improves both repair success and successful-core discovery as the sampling budget increases. Relative to Vanilla GRPO, the Success@k margin grows from +0.33 points at k = 1 to +2.17 at $k = 3 2 .$ while the gap in Success cores@k expands from +0.003 to +0.068 cores per GEM. Thus, the growing deployment advantage reflects genuine coverage of distinct successful repair cores rather than raw scaling alone. Training dynamics (Fig. 4b) mirror this pattern: differences in Success@1 remain modest, whereas QuotientPO separates clearly under Success@8 as training progresses. Together, these results show that quotient exploration becomes increasingly effective as more samples are allocated to uncover alternative valid repairs.

## 5.3 EMPIRICAL ANALYSIS OF QUOTIENTPO’S MECHANISM

Verifier feedback moves repair beyond the supervised teacher prior. SFT is trained to amortize MILPgenerated repairs, but a phenotype-consistent repair is not uniquely identified by its teacher edit set. Following Tokar et al. (2020), we project sampled repair sets via metric MDS on pairwise signed-Jaccard distances under shared coordinate axes (Fig. 5). While SFT-Soft clusters tightly around the teacher, QuotientPO discovers a well-separated region containing more successful solutions. This divergence also holds quantitatively across the test set: mean distance to the MILP teacher increases from 0.784 for SFT-Soft to 0.980 for QuotientPO (0.780 vs. 0.961 among successful repairs). Thus, the gain from verifier-guided optimization is not explained by more faithful imitation of the teacher, as successful QuotientPO repairs are substantially farther from it. This supports the partialidentification view of GEM repair, where the teacher provides one admissible correction but verifier feedback reveals alternative successful repairs beyond the supervised mode.

![](images/8a2d21c3714e87b013e613366ca02e42ffc1b4037f098409182a835ddf819bc6.jpg)  
Figure 5: Landscape of sampled repairs (Streptomyces axinellae).

![](images/52e5c780ff65c0facf65e5a43cfb3bccbc62df69ce4a0b30ef87931935e7be9d.jpg)  
(a) Raw diversity does not imply mechanism diversity.

![](images/9653c182557b66047c70f6877d1a579764be9c4c8df8b1695d9d2b819bc3480d.jpg)  
(b) Raw-to-core collapse and success filtering.  
Figure 6: Exploration across raw and quotient spaces. (a) MDS projections of sampled repairs before (top) and after (bottom) canonicalization on (S. griseoflavus). (b) Funnel transitions from distinct raw repairs $( k = 3 2 )$ to verifier-validated cores across the test set.

Quotient exploration converts diversity into useful mechanism coverage. Figure 6 exposes a key failure mode: although MaxEnt and GFlowNet explicitly diversify raw repair sets, over 80% of their sampled representations collapse after canonicalization, and neither improves successful-core yield over Vanilla GRPO. In contrast, QuotientPO applies diversity pressure directly in quotient space, reaching 6.44 distinct cores and 0.57 successful cores per GEM under the same 32-sample budget, compared with 5.98 and 0.50 for Vanilla GRPO. To disentangle mechanism coverage from the higher repair success rate itself, we further compare the two methods on paired GEMs where both produce at least r successful repairs and equalize the successful-sample budget. QuotientPO recovers more distinct successful cores at $r \in \{ 2 , 4 , 8 \}$ : 1.46 vs. 1.43, 2.05 vs. 1.96, and 2.78 vs. 2.66, respectively (Appendix F.4). Thus, its coverage advantage persists after controlling for the number of successful samples, rather than arising solely from a higher repair-success frequency.

Table 2: Finite-sample credit resolution and successfulcore diversity. Training diagnostics use all rollouts.

Kernel relaxation recovers graded diversity signals from coarse finitesample counts. Under Exact Renyi-´ 2, the exact-count tie rate rises from 7.34% to 10.23% over training (Table 2), while 78.2% of these final ties are resolved by kernel credit. In contrast, Kernel Renyi-2 reduces´ the exact-count tie rate of its rollout groups to 7.99%, while achieving the lowest successful-core colli-

<table><tr><td></td><td>Training (200→864)</td><td>Successful Cores</td><td></td></tr><tr><td>Method</td><td>Exact ties (%) Resolved (%) Collision ↓ Mean</td><td></td><td> $d _ { J } \ ^ { \prime }$  个</td></tr><tr><td>Vanilla</td><td> $8 . 1 9  8 . 5 9$   $6 0 . 5 5 \to 8 4 . 8 6$ </td><td>0.5769</td><td>0.2408</td></tr><tr><td>Shannon</td><td> $8 . 6 0  9 . 7 2$   $5 9 . 2 3  8 0 . 2 0$ </td><td>0.5687</td><td>0.2431</td></tr><tr><td>Exact R2</td><td> $7 . 3 4  1 0 . 2 3$   $5 9 . 7 1  7 8 . 2 0$ </td><td>0.5285</td><td>0.2861</td></tr><tr><td>Kernel R2</td><td> $8 . 5 7  7 . 9 9$   $6 3 . 8 1  7 2 . 7 8$ </td><td>0.5176</td><td>0.3008</td></tr></table>

sion (0.518) and largest mean edit Jaccard distance $d _ { J } \ = \ 1 - \ J \ ( 0 . 3 0 1 )$ . Consistently, it also achieves the best Success@32 and successful-core (Table 12). These results show that kernel relaxation provides finer exploration credit precisely where exact multiplicity counts lose resolution.

Structured initialization turns sparse verifier feedback into a learnable signal. Figure 7 separates the roles of representation learning and supervised policy formation. During single-edit imitation (SFT-A), which trains the model on individual MILP teacher operations, operation-level Hit@1 drops from 20.7% to 11.4% without structural pretraining and to 14.3% without ESM-2 conditioning, showing that both improve local repair prediction. Yet representation alone provides almost no terminal learning signal: before any RL update, only

![](images/0a8038cc5e62e0f48eb7b054e082d3963312b246bcad16b804c58bb808bbad2a.jpg)

![](images/5141062b9ab766f270609da94b6bc492a1fad9b7a38574706c03269fdd6c3640.jpg)  
Figure 7: Building an exploration-ready policy. (a) Pretraining and ESM-2 enhance the local repair prior. (b) Set-level SFT unlocks rollout reward variance for RL. Error bars denote standard deviation across 8 rollout batches.

Table 3: Robustness and ablation studies. S@32 denotes Success@32 (%).
<table><tr><td colspan="2">(a) RLOO Base</td></tr><tr><td>Method</td><td>S@32</td></tr><tr><td>Vanilla</td><td>15.75</td></tr><tr><td>MaxEnt</td><td>14.29</td></tr><tr><td>GFlowNet</td><td>11.57</td></tr><tr><td>QuotientPO 17.21</td><td></td></tr></table>

<table><tr><td colspan="2">(b) Policy Architecture</td></tr><tr><td>Arch. RL</td><td>S@32</td></tr><tr><td>Vanilla Soft</td><td>17.93 QuotientPO 20.10</td></tr><tr><td>Hard</td><td>Vanilla 5.92 QuotientPO 6.65</td></tr></table>

(c) Design Ablation
<table><tr><td>Variant</td><td>S@32</td></tr><tr><td>Default</td><td>20.10</td></tr><tr><td>Raw-space kernel 18.16</td><td></td></tr><tr><td>w/o Quotient term 17.54</td><td></td></tr><tr><td>w/o QPI term</td><td>19.62</td></tr><tr><td>w/o Red. penalty</td><td>19.94</td></tr></table>

(d) Sensitivity
<table><tr><td></td><td>Param. Setting S@32</td></tr><tr><td></td><td>Default 20.10</td></tr><tr><td>h(0.25)</td><td>0.125 19.41 0.5 18.57</td></tr><tr><td>β(0.02)</td><td>0.01 19.05 0.04 20.00</td></tr></table>

0.4% and 0.8% of rollout groups exhibit within-group reward variation under random and pretrained policies. Set-level supervision changes this regime sharply, increasing the rate to 48.1% without SFT-A and to 51.6% with the complete supervised pipeline. Thus, while pretraining and genomic conditioning provide a local repair prior, set-level supervision is crucial to unlock group reward variance for RL.

## 5.4 ROBUSTNESS AND ABLATION ANALYSIS

Robustness across RL objectives and policy architectures. As shown in Table 3(a,b), QuotientPO consistently outperforms baselines across RL formulations and architectures. Under RLOO (Ahmadian et al., 2024), it reaches 17.21 Success@32 (S@32, vs. 15.75 Vanilla), while under GRPO, it improves both Soft (20.10 vs. 17.93) and Hard (6.65 vs. 5.92) architectures. These results show that the benefit of quotient-space exploration is not tied to a specific RL formulation or architecture.

Design ablation and sensitivity. QPI and the redundancy penalty yield smaller mean improvements, while the quotient term accounts for the dominant performance gain: removing it incurs the steepest drop in S@32 (from 20.10 to 17.54; Table 3c). Critically, with the same kernel and training budget, quotient-space exploration improves both Success@32 (20.10 vs. 18.16) and successfulcore discovery (0.57 vs. 0.52) over its raw-space counterpart. Sensitivity analysis (Table 3d) shows broad robustness to the diversity weight β, but sensitivity to bandwidth h: a larger h over-smooths distant cores and blunts crowding selectivity, while leaving quotient identity unchanged.

## 6 CASE STUDY

Figure 8 grounds the mechanistic failure modes in concrete biological repairs. On S. mashuensis (Fig. 8a), Vanilla GRPO collapses entirely onto boundary entry blockade (NO3abc, NO3tex), whereas QuotientPO discovers an orthogonal intracellular mechanism: jointly knocking out the NADH- and menaquinol-dependent reductases (NITR, NTR3B). On S. palmae (Fig. 8b), syntactically different Vanilla proposals repeatedly retain the same extracellular bypass (SUCRe), so surrounding edits vary without resolving the sucrose phenotype, whereas QuotientPO recovers clean successful cores. Counterfactual replays confirm that toggling SUCRe directly flips repair outcomes. These cases expose two failures of raw exploration: viable mechanisms can be missed, while sur face variation around the same outcome-determining core can be mistaken for useful diversity. QuotientPO resolves both by directing exploration toward functionally distinct mechanisms.

![](images/46f1a7908f4b922d7e9f2cab4dad2f6c49e5b2cbea95eb3e9b77c6c4fe53547a.jpg)  
(a) S. mashuensis. Nitrate utilization.  
(b) S. palmae. Sucrose utilization.  
Figure 8: Case studies of mechanistic repair.

## 7 CONCLUSION

Repairing GEMs from indirect functional feedback faces a structural mismatch: combinatorial candidate edits can collapse to the same underlying repair mechanism, causing standard exploration to mistake syntactic multiplicity for scientific diversity. We address this mismatch by changing the unit of exploration from raw reaction-level actions to invariant repair mechanisms. By canonicaliz ing verifier-preserving edits, QuotientPO explores directly in quotient space, with a kernelized Renyi´ relaxation replacing coarse exact-match statistics with graded finite-sample crowding signals. This formulation removes representation-multiplicity bias while empirically improving repair success and the coverage of distinct valid mechanisms. More broadly, our results suggest that when scientific solutions admit equivalent representations, effective exploration should be defined over equivalence classes rather than surface representations (limitations and future directions in Appendix H).

## AI USE STATEMENT

We used generative AI tools to assist with writing refinement and code generation. For writing, these tools were used to improve the clarity, grammar, conciseness, and presentation of authorwritten text. For code, generative AI was used to assist with generating and revising portions of the implementation. All AI-assisted code was manually reviewed, tested, and verified by the authors before use. All AI-assisted writing was likewise reviewed and edited by the authors. The authors take full responsibility for the correctness and final content of this work.

## ETHICS STATEMENT

This work uses publicly available microbial genomic and phenotype data from NCBI (Sayers et al., 2022) and BacDive (Schober et al., 2025), together with established computational resources for genome-scale metabolic modeling (CarveMe (Machado et al., 2018), BiGG (King et al., 2016), etc.), in accordance with the terms and policies of their respective sources. All experiments are computational and involve no human participants, personally identifiable information, animal experiments, or experimental manipulation of living organisms. Our method operates only on in sil ico metabolic models and is intended for scientific modeling and computational research. While improved metabolic models may support downstream applications in metabolic engineering or synthetic biology, such uses should follow appropriate institutional biosafety procedures, regulatory requirements, and responsible-use practices.

## REPRODUCIBILITY STATEMENT

We provide detailed materials to support reproduction of our results. The complete QuotientPO algorithm and optimization procedure are described in Appendix A, with the core implementation and usage instructions included in the supplemental materials. The benchmark construction and data-processing pipeline, based on publicly available databases, are documented in Appendix C; the benchmark construction pipeline and evaluation code are also provided in the supplemental materials. Model architecture and training details for all stages are specified in Appendix E, and the corresponding code is included in the supplemental materials. Implementation details for all baselines are provided in Appendix D, with their core implementations likewise included in the supplemental materials.

## REFERENCES

Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J. Ballard, Joshua Bambrick, Sebastian W. Bodenstein, David A. Evans, Chia-Chun Hung, Michael O’Neill, David Reiman, Kathryn Tunyasuvunakool, Zachary Wu, Akvile˙ Zemgulyt<sup>ˇ</sup> e, Eirini Arvaniti, Charles Beattie, Ottavia Bertolli, Alex˙ Bridgland, Alexey Cherepanov, Miles Congreve, Alexander I. Cowen-Rivers, Andrew Cowie, Michael Figurnov, Fabian B. Fuchs, Hannah Gladman, Rishub Jain, Yousuf A. Khan, Caroline M. R. Low, Kuba Perlin, Anna Potapenko, Pascal Savy, Sukhdeep Singh, Adrian Stecula,

Ashok Thillaisundaram, Catherine Tong, Sergei Yakneen, Ellen D. Zhong, Michal Zielinski, Augustin Z<sup>ˇ</sup> ´ıdek, Victor Bapst, Pushmeet Kohli, Max Jaderberg, Demis Hassabis, and John M. Jumper. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, Jun 2024. ISSN 1476-4687. doi: 10.1038/s41586-024-07487-w. URL https://doi.org/10.1038/s41586-024-07487-w.

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin,´ Ahmet Ust<sup>¨</sup> un, and Sara Hooker. Back to basics: Revisiting REINFORCE-style optimiza-¨ tion for learning from human feedback in LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.662. URL https://aclanthology.org/2024.acl-long.662/.

Georg Basler, Oliver Ebenhoh, Joachim Selbig, and Zoran Nikoloski. Mass-balanced randomization¨ of metabolic networks. Bioinformatics, 27(10):1397–1403, 05 2011. ISSN 1367-4803. doi: 10.1093/bioinformatics/btr145. URL https://doi.org/10.1093/bioinformatics/ btr145.

Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning, 2017. URL https://arxiv.org/abs/1611. 09940.

Matthew N. Benedict, Michael B. Mundy, Christopher S. Henry, Nicholas Chia, and Nathan D. Price. Likelihood-based gene annotations for gap filling and quality assessment in genome-scale metabolic models. PLOS Computational Biology, 10(10):1–14, 10 2014. doi: 10.1371/journal. pcbi.1003882. URL https://doi.org/10.1371/journal.pcbi.1003882.

Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 27381–27394. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/ file/e614f646836aaed9f89ce58e837e2310-Paper.pdf.

Daniil A. Boiko, Robert MacKnight, Ben Kline, and Gabe Gomes. Autonomous chemical research with large language models. Nature, 624(7992):570–578, Dec 2023. ISSN 1476-4687. doi: 10.1038/s41586-023-06792-0. URL https://doi.org/10.1038/ s41586-023-06792-0.

A.Z. Broder. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No.97TB100171), pp. 21–29, 1997. doi: 10.1109/ SEQUEN.1997.666900.

Can Chen, Chen Liao, and Yang-Yu Liu. Teasing out missing reactions in genome-scale metabolic networks through hypergraph learning. Nature Communications, 14(1):2375, 2023. ISSN 2041-1723. doi: 10.1038/s41467-023-38110-7. URL https://doi.org/10.1038/ s41467-023-38110-7.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function, 2018. URL https://arxiv.org/abs/1802. 06070.

Bin Feng, Jiying Zhang, Xinni Zhang, Zijing Liu, and Yu Li. BioMD: All-atom generative model for biomolecular dynamics simulation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=LQDeJk6NOr.

Luis G. Sanchez Giraldo, Murali Rao, and Jose C. Principe. Measures of entropy from data using infinitely divisible kernels, 2014. URL https://arxiv.org/abs/1211.2459.

Luca Grillotti, Maxence Faldor, Borja G. Leon, and Antoine Cully. Quality-diversity actor-critic:´ Learning high-performing and diverse behaviors via value and successor features critics. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 16416–16459. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/grillotti24a. html.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1861–1870. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/haarnoja18b.html.

Daniel Hartleb, Florian Jarre, and Martin J. Lercher. Improved metabolic models for e. coli and mycoplasma genitalium from globalfit, an algorithm that simultaneously matches growth and non-growth data sets. PLOS Computational Biology, 12(8):1–22, 08 2016. doi: 10.1371/journal. pcbi.1005036. URL https://doi.org/10.1371/journal.pcbi.1005036.

Ramin Hasibi, Tom Michoel, and Diego A. Oyarzun. Integration of graph neural networks and´ genome-scale metabolic models for predicting gene essentiality. npj Systems Biology and Applications, 10(1):24, Mar 2024. ISSN 2056-7189. doi: 10.1038/s41540-024-00348-2. URL https://doi.org/10.1038/s41540-024-00348-2.

Matthew M. Hong, Jesse Zhang, Anusha Nagabandi, and Abhishek Gupta. TMRL: Diffusion Timestep-Modulated Pretraining Enables Exploration for Efficient Policy Finetuning. In Proceedings of Robotics: Science and Systems, Sydney, Australia, July 2026. doi: 10.15607/RSS. 2026.XXII.208.

Yunli Eric Hsieh, Kshitij Tandon, Heroen Verbruggen, and Zoran Nikoloski. Comparative analysis of metabolic models of microbial communities reconstructed from automated tools and consensus approaches. npj Systems Biology and Applications, 10(1):54, 2024. ISSN 2056-7189. doi: 10.1038/s41540-024-00384-y. URL https://doi.org/10.1038/ s41540-024-00384-y.

Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks, 2020. URL https://arxiv. org/abs/1905.12265.

Hanbo Huang, Xuan Gong, Jing Wang, Lei Bai, Xiang Xiao, Weishu Zhao, and Shiyu Liang. Ggbound: A genome-grounded agent for microbial life-boundary prediction, 2026. URL https: //arxiv.org/abs/2605.14442.

Weihong Huang, Feng Yang, Qiang Zhang, and Juan Liu. A dual-scale fused hypergraph convolution-based hyperedge prediction model for predicting missing reactions in genome-scale metabolic networks. Briefings in Bioinformatics, 25(5):bbae383, 09 2024. ISSN 1477-4054. doi: 10.1093/bib/bbae383. URL https://doi.org/10.1093/bib/bbae383.

Q. Huangfu and J. A. J. Hall. Parallelizing the dual revised simplex method, 2015. URL https: //arxiv.org/abs/1503.01889.

Moksh Jain, Emmanuel Bengio, Alex Hernandez-Garcia, Jarrid Rector-Brooks, Bonaventure F. P. Dossou, Chanakya Ajit Ekbote, Jie Fu, Tianyu Zhang, Michael Kilgour, Dinghuai Zhang, Lena Simine, Payel Das, and Yoshua Bengio. Biological sequence design with GFlowNets. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 9786–9801. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/jain22a.html.

George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, Jun 2021. ISSN 2522-5820. doi: 10.1038/s42254-021-00314-5. URL https://doi.org/10.1038/ s42254-021-00314-5.

Hohyun Kim, Seunggeun Lee, and Min hwan Oh. Symmetry-aware gflownets, 2025. URL https: //arxiv.org/abs/2506.02685.

Zachary A. King, Justin Lu, Andreas Drager, Philip Miller, Stephen Federowicz, Joshua A. Lerman,¨ Ali Ebrahim, Bernhard O. Palsson, and Nathan E. Lewis. Bigg models: A platform for integrating, standardizing and sharing genome-scale models. Nucleic Acids Research, 44(D1):D515–D522, 01 2016. ISSN 0305-1048. doi: 10.1093/nar/gkv1049. URL https://doi.org/10.1093/ nar/gkv1049.

Steffen Klamt, Utz-Uwe Haus, and Fabian Theis. Hypergraphs and cellular networks. PLOS Computational Biology, 5(5):1–6, 05 2009. doi: 10.1371/journal.pcbi.1000385. URL https: //doi.org/10.1371/journal.pcbi.1000385.

Wouter Kool, Herke van Hoof, and Max Welling. Attention, learn to solve routing problems!, 2019. URL https://arxiv.org/abs/1803.08475.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= VD-AYtP0dve.

Vinay Satish Kumar and Costas D. Maranas. Growmatch: An automated method for reconciling in silico/in vivo growth predictions. PLOS Computational Biology, 5(3):1–13, 03 2009. doi: 10.1371/journal.pcbi.1000308. URL https://doi.org/10.1371/journal. pcbi.1000308.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 21314–21328. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1549. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/8636419dea1aa9fbd25fc4248e702da4-Paper-Conference.pdf.

Chenyi Li, Yuan Zhang, Bo Wang, Guoqing Ma, Wei Tang, Haoyang Huang, and Nan Duan. Setpo: Set-level policy optimization for diversity-preserving llm reasoning, 2026. URL https:// arxiv.org/abs/2602.01062.

Ping Li, Anshumali Shrivastava, Joshua Moore, and Arnd Konig. Hashing algorithms for large-¨ scale learning. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger (eds.), Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc., 2011. URL https://proceedings.neurips.cc/paper\_files/paper/2011/ file/0a1bf96b7165e962e90cb14648c9462d-Paper.pdf.

Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Salvatore Candido, and Alexander Rives. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379(6637):1123–1130, 2023. doi: 10.1126/ science.ade2574. URL https://www.science.org/doi/abs/10.1126/science. ade2574.

Xiaoyi Liu, Hongpeng Yang, Chengwei Ai, Ruihan Dong, Yijie Ding, Qianqian Yuan, Jijun Tang, and Fei Guo. A generalizable framework for unlocking missing reactions in genome-scale metabolic networks using deep learning, 2024. URL https://arxiv.org/abs/2409. 13259.

Daniel Machado, Sergej Andrejev, Melanie Tramontano, and Kiran Raosaheb Patil. Fast automated reconstruction of genome-scale metabolic models for microbial species and communities. Nucleic Acids Research, 46(15):7542–7553, 09 2018. ISSN 0305-1048. doi: 10.1093/nar/gky537. URL https://doi.org/10.1093/nar/gky537.

Adil Mardinoglu and Bernhard Ø. Palsson. Genome-scale models in human metabologenomics. Nature Reviews Genetics, 26(2):123–140, Feb 2025. ISSN 1471-0064. doi: 10.1038/ s41576-024-00768-0. URL https://doi.org/10.1038/s41576-024-00768-0.

Amil Merchant, Simon Batzner, Samuel S. Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin Dogus Cubuk. Scaling deep learning for materials discovery. Nature, 624(7990):80–85, Dec 2023. ISSN 1476-4687. doi: 10.1038/s41586-023-06735-9. URL https://doi.org/ 10.1038/s41586-023-06735-9.

Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. Kernel language entropy: Finegrained uncertainty quantification for llms from semantic similarities, 2024. URL https:// arxiv.org/abs/2405.20003.

Jeffrey D. Orth, Ines Thiele, and Bernhard Ø Palsson. What is flux balance analysis? Nature Biotechnology, 28(3):245–248, Mar 2010. ISSN 1546-1696. doi: 10.1038/nbt.1614. URL https://doi.org/10.1038/nbt.1614.

Jose C. Principe, Dongxin Xu, Qun Zhao, and John W. Fisher. Learning from examples with information theoretic criteria. Journal of VLSI signal processing systems for signal, image and video technology, 26(1):61–77, Aug 2000. ISSN 0922-5773. doi: 10.1023/A:1008143417156. URL https://doi.org/10.1023/A:1008143417156.

A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmuller, and J. Timmer. Struc-¨ tural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics, 25(15):1923–1929, 08 2009. ISSN 1367-4803. doi: 10.1093/bioinformatics/btp358. URL https://doi.org/10.1093/bioinformatics/ btp358.

Vinay Satish Kumar, Madhukar S. Dasika, and Costas D. Maranas. Optimization based automated curation of metabolic reconstructions. BMC Bioinformatics, 8(1):212, 2007. doi: 10.1186/1471-2105-8-212. URL https://doi.org/10.1186/1471-2105-8-212.

Eric W Sayers, Evan E Bolton, J Rodney Brister, Kathi Canese, Jessica Chan, Donald C Comeau, Ryan Connor, Kathryn Funk, Chris Kelly, Sunghwan Kim, Tom Madej, Aron Marchler-Bauer, Christopher Lanczycki, Stacy Lathrop, Zhiyong Lu, Francoise Thibaud-Nissen, Terence Murphy, Lon Phan, Yuri Skripchenko, Tony Tse, Jiyao Wang, Rebecca Williams, Barton W Trawick, Kim D Pruitt, and Stephen T Sherry. Database resources of the national center for biotechnology information. Nucleic Acids Research, 50(D1):D20–D26, 01 2022. ISSN 0305-1048. doi: 10.1093/nar/gkab1112. URL https://doi.org/10.1093/nar/gkab1112.

Isabel Schober, Julia Koblitz, Joaquim Sarda Carbasse, Christian Ebeling, Marvin Leon Schmidt,\` Adam Podstawka, Rohit Gupta, Vinodh Ilangovan, Javad Chamanara, Jorg Overmann, and¨ Lorenz Christian Reimer. Bacdive in 2025: the core database for prokaryotic strain data. Nucleic Acids Research, 53(D1):D748–D756, 01 2025. ISSN 1362-4962. doi: 10.1093/nar/gkae959. URL https://doi.org/10.1093/nar/gkae959.

Wheaton L. Schroeder and Rajib Saha. Optfill: A tool for infeasible cycle-free gapfilling of stoichiometric metabolic models. iScience, 23(1):100783, 2020. ISSN 2589-0042. doi: https://doi. org/10.1016/j.isci.2019.100783. URL https://www.sciencedirect.com/science/ article/pii/S2589004219305280.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=HJgLZR4KvH.

Ashley G. Smart, Luis A. N. Amaral, and Julio M. Ottino. Cascading failure and robustness in metabolic networks. Proceedings of the National Academy of Sciences, 105(36):13223–13228, 2008. doi: 10.1073/pnas.0803571105. URL https://www.pnas.org/doi/abs/10. 1073/pnas.0803571105.

Ines Thiele and Bernhard Ø Palsson. A protocol for generating a high-quality genome-scale metabolic reconstruction. Nature Protocols, 5(1):93–121, Jan 2010. ISSN 1750-2799. doi: 10.1038/nprot.2009.203. URL https://doi.org/10.1038/nprot.2009.203.

Ines Thiele, Nikos Vlassis, and Ronan M. T. Fleming. fastgapfill: efficient gap filling in metabolic networks. Bioinformatics, 30(17):2529–2531, 09 2014. ISSN 1367-4803. doi: 10.1093/bioinformatics/btu321. URL https://doi.org/10.1093/bioinformatics/ btu321.

Tomas Tokar, Chiara Pastrello, and Igor Jurisica. Gsoap: a tool for visualization of gene set overrepresentation analysis. Bioinformatics, 36(9):2923–2925, 05 2020. ISSN 1367-4803. doi: 10. 1093/bioinformatics/btaa001. URL https://doi.org/10.1093/bioinformatics/ btaa001.

Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, Anima Anandkumar, Karianne Bergen, Carla P. Gomes, Shirley Ho, Pushmeet Kohli, Joan Lasenby, Jure Leskovec, Tie-Yan Liu, Arjun Manrai, Debora Marks, Bharath Ramsundar, Le Song, Jimeng Sun, Jian Tang, Petar Velickoviˇ c,´ Max Welling, Linfeng Zhang, Connor W. Coley, Yoshua Bengio, and Marinka Zitnik. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, Aug 2023. ISSN 1476-4687. doi: 10.1038/s41586-023-06221-2. URL https://doi.org/10.1038/ s41586-023-06221-2.

Naganand Yadati, Vikram Nitin, Madhav Nimishakavi, Prateek Yadav, Anand Louis, and Partha Talukdar. Nhp: Neural hypergraph link prediction. In Proceedings ofthe 29th ACM International Conference on Information & Knowledge Management, CIKM ’20, pp. 1705–1714, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450368599. doi: 10.1145/ 3340531.3411870. URL https://doi.org/10.1145/3340531.3411870.

Jiaxuan You, Bowen Liu, Zhitao Ying, Vijay Pande, and Jure Leskovec. Graph convolutional policy network for goal-directed molecular graph generation. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/d60678e8f2ba9c540798ebbde31177e8-Paper.pdf.

Yanlong Zhao, Yixiao Chen, Yi Yu, Xiang Liu, Jiawen Du, Jun Wen, Quan Sun, Ren Wang, and Can Chen. A multi-way SMILES-based hypergraph inference network for metabolic model reconstruction. Communications Biology, 9(1):531, 2026. ISSN 2399-3642. doi: 10.1038/ s42003-026-09761-1. URL https://doi.org/10.1038/s42003-026-09761-1.

Johannes Zimmermann, Christoph Kaleta, and Silvio Waschina. gapseq: informed prediction of bacterial metabolic pathways and reconstruction of accurate metabolic models. Genome Biology, 22(1):81, 2021. ISSN 1474-760X. doi: 10.1186/s13059-021-02295-1. URL https://doi. org/10.1186/s13059-021-02295-1.

## A ALGORITHM DETAILS OF QUOTIENTPO

This section gives the implementation details of QuotientPO. For compactness, let $x = ( M _ { 0 } , G , y )$ denote a training instance, while emphasizing that the repair policy $\pi _ { \boldsymbol { \theta } } \mathbf { \bar { ( } } E \mid M _ { 0 } , G )$ conditions only on the initial GEM and genomic representation. The phenotype labels y are used exclusively by the verifier. QuotientPO fine-tunes the supervised one-shot repair policy using four signals: verifier reward, kernelized quotient crowding (Quotient term), finite-panel successful-core discovery credit (QPI term), and a redundancy penalty outside the canonical repair core.

## A.1 ONE-SHOT REPAIR POLICY

Let $\mathcal { A } ( M _ { 0 } )$ denote the legal signed reaction edits for an initial GEM, where $( \mathrm { A D D } , r )$ and $( \mathrm { R E M O V E } , r )$ are treated as distinct actions. The policy predicts the complete unordered repair set in one shot. For each legal edit $a \in \mathcal { A } ( M _ { 0 } )$ , the actor produces a logit $z _ { \theta , a }$ and an operationspecific threshold $\tau _ { \theta , o ( a ) }$ , giving the inclusion probability $p _ { \theta , a } = \mathrm { s i g m o i d } \left( z _ { \theta , a } - \tau _ { \theta , o ( a ) } \right)$ . The complete edit-set likelihood is the product of the corresponding Bernoulli decisions,

$$
\log \pi _ { \theta } ( E \mid M _ { 0 } , G ) = \sum _ { a \in A ( M _ { 0 } ) } \left[ \mathbb { I } [ a \in E ] \log p _ { \theta , a } + \mathbb { I } [ a \not \in E ] \log ( 1 - p _ { \theta , a } ) \right] .
$$

Algorithm 1 Verifier-preserving canonicalization $\kappa _ { y }$   
Require: Repair set $E ,$ phenotype targets y, deterministic verifier $\mathrm { F B A } _ { y } ,$ maximum parallelism $B \geq 1$   
Ensure: Canonical repair core $m = \kappa _ { y } ( E )$   
1: $\pmb { \sigma } ^ { \star } \gets \mathrm { F B A } _ { y } ( E )$   
2: C ← sort<sub>(operation,reaction)</sub>(E)   
3: while |C| > 0 do   
4: $( c _ { 1 } , \ldots , c _ { n } )  C$   
5: removed ← false   
6: i ← 1   
7: w ← 1   
8: while $i \leq n$ do   
9: r ← min(n, i + w − 1)   
10: for $j = i , \ldots , r \mathbf { d o }$   
11: $\mathbf { \bar { \mathit { S } } } _ { j }  C \setminus \{ \mathit { c } _ { j } \}$   
12: $\overset { \triangledown } { h _ { j } ^ { + } }  \mathrm { S U B M I T } ( \mathrm { F B A } _ { y } ( S _ { j } ) = \pmb { \sigma } ^ { \star } )$ ▷ Asynchronous verifier check   
13: end for   
14: for $j = i , \dots , r$ do   
15: a<sub>j</sub> ← AWAIT(h<sub>j</sub>) ▷ Consume results in the fixed order   
16: if a = true then   
17: ${ \bf \bar { \theta } } _ { C  C \backslash \{ c _ { j } \} }$   
18: removed ← true   
19: CANCELORDISCARD(h<sub>j+1:r</sub>)   
20: break   
21: end if   
22: end for   
23: if removed then   
24: break ▷ Restart after the committed deletion   
25: end if   
26: i ← r + 1   
27: w ← min(2w, B) ▷ Expand only after the entire batch fails   
28: end while   
29: if not removed then   
30: break ▷ No verifier-preserving singleton deletion remains   
31: end if   
32: end while   
33: return C

## A.2 VERIFIER-PRESERVING CANONICALIZATION

For a sampled repair E, let ${ \pmb \sigma } _ { y } ( E ) = ( { \mathbb I } [ { \widehat { y } } _ { i } ( M _ { E } ) = y _ { i } ] ) _ { i = 1 } ^ { N }$ denote the complete phenotypesatisfaction vector across all $\bar { N }$ conditions. The quotient map $\kappa _ { y }$ (Section 4.1) is implemented as a deterministic, verifier-preserving reduction (Algorithm 1) that systematically removes redundant edits while preserving $\sigma _ { y } ( E )$ . The resulting deletion-irreducible repair core $m = \kappa _ { y } ( E ) \subseteq E$ defines the canonical functional representation of E under the verifier and induces the redundancy cost $| E | - | \kappa _ { y } ( E )$ |. Because $\kappa _ { y }$ is deterministic, every repair is assigned to exactly one canonical core, yielding a well-defined quotient partition over the repair space.

## A.3 KERNELIZED QUOTIENT CROWDING (QUOTIENT TERM)

For each organism, we sample a rollout group $E _ { 1 : L } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid M _ { 0 } , G ) $ and obtain their canonical cores $m _ { i } = \kappa _ { y } ( E _ { i } )$ . All cores participate in quotient exploration, irrespective of whether they achieve complete phenotype recovery. As in Section 4.2, signed repair cores are compared via the kernel

$$
k ( m _ { i } , m _ { j } ) = \exp \left( - \frac { 1 - J ( m _ { i } , m _ { j } ) } { h } \right) , \qquad J ( m _ { i } , m _ { j } ) = \frac { | m _ { i } \cap m _ { j } | } { | m _ { i } \cup m _ { j } | } ,
$$

with $J ( \emptyset , \emptyset ) = 1$ , and with ADD and REMOVE edits treated as distinct signed elements.

For $L \geq 2 .$ , the empirical off-diagonal kernel concentration is

$$
\widehat { I } = \frac { 1 } { L ( L - 1 ) } \sum _ { a \neq b } k ( m _ { a } , m _ { b } ) .
$$

The candidate-level crowding cost is directly given by Eq. (3):

$$
c _ { i } = 2 L \frac { \sum _ { j \neq i } k ( m _ { i } , m _ { j } ) } { \sum _ { a \neq b } k ( m _ { a } , m _ { b } ) } - 2 .
$$

For numerical robustness, the implemented score is clipped: $\widetilde { c } _ { i } = \mathrm { c l i p } ( c _ { i } , - 8 , 8 )$ . The analysis in Section 4.3 characterizes the unclipped crowding score $c _ { i } ;$ implementation clipping is used only for finite-sample numerical stability. A positive crowding cost suppresses probability assigned to locally redundant cores, while a negative cost favors isolated cores.

## A.4 FINITE-PANEL SUCCESSFUL-CORE DISCOVERY CREDIT (QPI TERM)

Kernelized Renyi credit encourages geometric diversity over the entire space of repair cores, but ´ finite rollout groups provide sparse signals regarding which distinct cores are functionally useful. We therefore supplement it with a finite-panel quotient policy iteration (QPI) credit that explicitly rewards successful-core discovery.

For the reported setting, each rollout group of $L = 3 2$ samples is deterministically repartitioned $R \ : = \ : 4$ times into panels of size $K \ = \ 8 .$ For a panel $P ,$ let $\begin{array} { r } { S _ { P } = \sum _ { i \in P } s _ { y } ( \dot { E } _ { j } ) } \end{array}$ denote the number of successful samples, and $\begin{array} { r } { N _ { P } ( m ) = \sum _ { i \in P } \mathbb { I } [ m _ { j } = m ] } \end{array}$ denote the multiplicity of core m within the panel. Each training organism maintains a bounded archive $\mathcal { H } _ { x }$ of previously discovered successful cores. Let $\mathcal { H } _ { x } ^ { - }$ denote the archive before processing the current rollout group, and let $\nu _ { i } = \mathbb { I } [ s _ { y } ( E _ { i } ) = 1 , m _ { i } \notin \mathcal { H } _ { x } ^ { - } ]$ indicate that sample i discovers a previously unseen successful core. The multiplicity of novel samples realizing core m in panel $P$ is $\begin{array} { r } { \bar { N } _ { P } ^ { \mathrm { n e w } } ( m ) = \sum _ { i \in P } \nu _ { j } \mathbb { I } [ m _ { j } = m ] } \end{array}$

For sample i in panel P, the credit decomposes into three terms:

$$
\begin{array} { r l r } & { } & { a _ { i } ( P ) = \left\{ \displaystyle { s _ { y } ( E _ { i } ) / S _ { P } } , \quad S _ { P } > 0 , \right. } \\ & { } & { \left. S _ { P } = 0 , \right. } \\ & { } & { b _ { i } ( P ) = \displaystyle { \frac { 1 } { K } \mathbb { I } [ N _ { P } ( m _ { i } ) = 1 ] } , } \\ & { } & { n _ { i } ( P ) = \left\{ \displaystyle { \frac { 1 } { K N _ { P } ^ { \mathrm { n e w } } ( m _ { i } ) } } , \quad \nu _ { i } = 1 , \right. } \\ & { } & { \left. \nu _ { i } = 0 , \right. } \end{array}
$$

successful coverage,

within-panel core diversity,

archive novelty.

The first term shares credit among successful candidates in the panel; the second rewards cores that occur uniquely within the current panel; and the third rewards successful cores not previously archived. Averaging over the R deterministic repartitions gives the detached QPI credit

$$
u _ { i } = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \left[ a _ { i } ( P _ { i } ^ { ( r ) } ) + \alpha _ { t } \left( b _ { i } ( P _ { i } ^ { ( r ) } ) + n _ { i } ( P _ { i } ^ { ( r ) } ) \right) \right] ,
$$

Algorithm 2 QuotientPO: quotient-space policy optimization   
Require: Training set $\mathcal { D } ;$ initialized policy $\pi _ { \boldsymbol { \theta } _ { 0 } } ;$ verifier FBA; rollout size $L ;$ panel size $K$   
Ensure: Trained policy $\pi _ { \boldsymbol { \theta } _ { T } }$   
1: $\theta  \theta _ { 0 }$   
2: Initialize successful-core archives $\{ \mathcal { H } _ { x } \} _ { x \in \mathcal { D } }$   
3: for $t = 1 , \dots , T$ do   
4: Select a training batch, including archive-guided context replay   
5: $\theta _ { \mathrm { o l d } }  \theta$   
6: Set coefficients $\left( \beta _ { t } , \lambda _ { t } , \alpha _ { t } , \eta _ { t } \right)$   
7: for each training instance $x = ( M _ { 0 } , G , y )$ in the batch do   
8: Sample $E _ { 1 : L } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid M _ { 0 } , G ) $   
9: for $i = 1 , \dots , L$ do   
10: Verify $M _ { E _ { i } }$ to obtain $\mathrm { F B A } _ { y } ( E _ { i } ) , \pmb { \sigma } _ { y } ( E _ { i } )$ , and $\Phi _ { y } ( M _ { E _ { i } } )$   
11: Compute terminal reward ${ \dot { R } } _ { y } ( E _ { i } )$   
12: $m _ { i } \ : \dot {  } \ : \mathrm { C A N O N I C A L I Z E } ( E _ { i } , \mathcal { Y } , \mathrm { F B A } _ { y } )$ ▷ Algorithm 1   
13: Compute redundancy $( | \dot { E } | - | \kappa _ { y } ( E ) | )$   
14: end for   
15: Compute group-relative advantages $A _ { 1 : L }$   
16: Compute kernel crowding costs $\widetilde { c } _ { 1 : L }$ from the valid cores   
17: Compute finite-panel QPI credits $u 1 { : } L$ against the pre-update archive $\mathcal { H } _ { x } ^ { - }$   
18: Insert newly discovered successful cores into $\mathcal { H } _ { x }$   
19: end for   
20: Rescore all sampled edit sets under $\pi \theta$ and compute policy ratios $\rho _ { i } ( \theta )$   
21: Form $\mathcal { T } _ { t } ( \theta )$   
22: Take one gradient update on $\theta$   
23: end for   
24: return $\pi _ { \boldsymbol { \theta } _ { T } }$

where $P _ { i } ^ { ( r ) }$ is the panel containing sample i in repartition r. Newly discovered successful cores are subsequently inserted into $\mathcal { H } _ { x }$ (bounded to eight distinct cores per organism). The archive is used strictly to evaluate novelty and prioritize under-explored instances; archived repairs are never replayed as actions, and no distillation loss is applied.

## A.5 COMPLETE QUOTIENTPO OBJECTIVE

Within each rollout group, the verifier reward is normalized using the GRPO baseline, $A _ { i } \ =$ $\frac { R _ { y } ( E _ { i } ) - \overline { { R } } } { \sigma _ { R } + \varepsilon }$ , with $A _ { i } = 0$ when the group reward variance vanishes. For a sampled repair, the policy ratio is $\begin{array} { r } { \rho _ { i } ( \theta ) = \frac { \pi _ { \theta } ( E _ { i } | M _ { 0 } , G ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( E _ { i } | M _ { 0 } , G ) } } \end{array}$ . The implemented maximization objective is

$$
\begin{array} { c } { \displaystyle \mathcal { J } _ { t } ( \theta ) = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \operatorname* { m i n } \Bigl ( \rho _ { i } ( \theta ) A _ { i } , ~ \mathrm { c l i p } ( \rho _ { i } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } \Bigr ) } \\ { \displaystyle - \frac { \beta _ { t } } { L } \sum _ { i = 1 } ^ { L } \rho _ { i } ( \theta ) ~ \mathrm { s g } ( \widetilde { c } _ { i } ) } \\ { \displaystyle + \frac { \lambda _ { t } K } { L } \sum _ { i = 1 } ^ { L } \rho _ { i } ( \theta ) ~ \mathrm { s g } ( u _ { i } ) - \frac { \eta _ { t } } { L } \sum _ { i = 1 } ^ { L } \rho _ { i } ( \theta ) ~ \mathrm { s g } ( | E | - | \kappa _ { y } ( E ) | ) , } \end{array}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient. The first term is verifier-guided GRPO. The second implements the kernelized quotient-diversity objective by shifting probability away from crowded repair cores. The third is the finite-panel QPI term that promotes successful-core discovery and retention. The final term suppresses surface edits removed by canonicalization. All verifier outcomes, advantages, core identities, crowding costs, QPI credits, and redundancy costs are detached from the actor graph. Gradients therefore flow only through the complete edit-set likelihood $\pi _ { \boldsymbol { \theta } } ( E \mid M _ { 0 } , G )$ . PPO clipping is applied to the GRPO task term, whereas the quotient and auxiliary terms use the unclipped likelihood ratio. The full algorithm process is Algorithm 2.

In all experiments, the hyperparameters balance verifier reward optimization against mechanismlevel diversity: we use a rollout group size of $L = 3 2$ with PPO clipping threshold $\epsilon = 0 . 2$ . For exploration, we set the kernel crowding weight to $\beta = 0 . 0 2$ to penalize mechanism collapse, the finite-panel QPI discovery weight to $\lambda = 0 . 1 0$ with panel size ${ \bar { K } } = 8$ and internal novelty scale $\alpha = 0 . 0 5$ , and the canonical redundancy penalty to $\eta = 0 . 0 0 1$ to gently suppress extraneous surface edits without impeding phenotype recovery.

## B THEORETICAL DETAILS

## B.1 FORMULATION OF GEM MODELING

A genome-scale metabolic model (GEM) is a directed, weighted bipartite graph $M = ( \mathcal { C } , \mathcal { R } , S )$ where metabolites C and reactions R form the two node types and $S \in \mathbb { R } ^ { | \mathcal { C } | \times | \mathcal { R } | }$ encodes their stoichiometric edges. For metabolite c and reaction $r , S _ { c r } < 0$ denotes consumption $( c  r )$ $S _ { c r } > 0$ denotes production $( r  c )$ , and $| S _ { c r } |$ gives the stoichiometric coefficient. Reaction addition or deletion therefore corresponds to adding or removing a reaction node and its associated column of S. Flux balance analysis (FBA) assigns reaction fluxes v and computes

$$
v ^ { * } ( M ) = \operatorname* { m a x } _ { v } v _ { \mathrm { b i o } } \qquad \mathrm { s . t . } \qquad S v = 0 , \quad \ell \leq v \leq u ,
$$

where $S v \ : = \ : 0$ enforces steady-state mass balance and $[ \ell , u ]$ specifies reaction directionality and the reference environment. Wet-lab phenotypes provide organism-specific functional constraints through $\widehat { y } _ { z } ( M )$ , and $v ^ { * } ( M ) > \delta$ enforces a non-negligible biomass-supporting feasible flux.

## B.2 THEORY OF REPRESENTATION-INVARIANT EXPLORATION

Notation and setup. Given an initial draft metabolic network $M _ { 0 }$ , genomic context G, and experimental phenotype target y, the repair policy generates a raw repair candidate $E \in { \mathcal { F } }$ (a set of reaction additions or deletions). The deterministic verifier-preserving reduction $\kappa _ { u } : { \mathcal { F } }  { \mathcal { F } }$ maps each raw repair $E$ to an operational repair core $m = \kappa _ { y } ( E )$ . We denote by $\tilde { \mathcal { M } _ { y } } = \kappa _ { y } ( \mathcal { F } ) \subseteq \mathcal { F }$ the quotient mechanism space of all such canonical repairs. Throughout this subsection, we fix an arbitrary training instance $( M _ { 0 } , G , y )$ and suppress this conditioning from the raw policy π<sub>θ</sub> and the induced quotient policy $q _ { \theta , y }$ to streamline notation, writing $\pi ( E )$ and $q ( m )$ respectively. For each mechanism $m \in \mathcal { M } _ { y }$ , we define:

• the repair fiber $\Omega _ { m } = \{ E \in \mathcal { F } : \kappa _ { y } ( E ) = m \}$ , which collects all syntactically distinct reaction edits mapped to the same canonical repair mechanism m.

• the fiber multiplicity $n _ { m } = | \Omega _ { m } |$ , quantifying the syntactic degeneracy (the number of redundant representations) of mechanism m.

• the quotient policy marginal $\begin{array} { r } { q ( m ) = \sum _ { E \in \Omega _ { m } } \pi ( E ) } \end{array}$ , representing the aggregate probability assigned by the policy to deploying mechanism m.

Because $\kappa _ { y }$ preserves the complete phenotype response, any raw edit $E \in \Omega _ { m }$ achieves an identical post-repair phenotype evaluation $\mathring { \Phi _ { y } } ( M _ { E } ) \mathring { = } \mathring { \Phi _ { y } } ( M _ { m } )$ . Consequently, the mechanism-level class utility $\bar { U } _ { y } ( \bar { m } ) = \bar { \Phi _ { y } ( M _ { E } ) } , E \in \Omega _ { m }$ , is well defined and invariant to the choice of raw representative within $\bar { \Omega _ { m } }$ . Since the candidate edit space $\mathcal { F }$ is formed by reaction additions and deletions over a finite reaction database, it is finite. Consequently, the quotient mechanism space $\mathcal { M } _ { y }$ and every repair fiber $\Omega _ { m }$ are finite. We denote by $\Delta ( \dot { \mathcal { F } } )$ and $\Delta ( \mathcal { M } _ { y } )$ the probability simplices over candidate edits and repair mechanisms, respectively, and assume an exploration coefficient $\beta > 0$ throughout.

Lemma B.1 (Entropy decomposition over repair fibers). For any probability distribution $\pi \in \Delta ( \mathcal { F } )$ define the conditional distribution $\pi ( E \mid m ) ^ { \backslash } = \pi ( E ) / q ( m ) f o r \bar { E } \in \Omega _ { m }$ whenever $q ( m ) > 0$ . Then

$$
H _ { \mathrm { a c t } } ( \pi ) = H ( q ) + \sum _ { m \in \mathcal { M } _ { y } } q ( m ) H ( \pi ( \cdot \mid m ) ) ,\tag{5}
$$

where zero-mass fibers contribute zero. Moreover,

$$
H ( \pi ( \cdot \mid m ) ) \leq \log n _ { m } ,\tag{6}
$$

with equality, for $q ( m ) > 0$ , if and only $i f \pi ( \cdot \mid m )$ is uniform on $\Omega _ { m }$

Proof. Because $\mathcal { F } = \textstyle \sqcup _ { m \in \mathcal { M } _ { y } } \Omega _ { m }$ , for every $E \in \Omega _ { m }$ with $q ( m ) > 0$ we may write $\pi ( E ) =$ $q ( m ) \pi ( E \mid m )$ . Therefore,

$$
\begin{array} { l } { { \displaystyle { \cal H } _ { \mathrm { a c t } } ( \pi ) = - \sum _ { m \in \mathcal { M } _ { y } } \displaystyle { \sum _ { { \cal E } \in \Omega _ { m } } q ( m ) \pi ( { \cal E } \mid m ) \log \bigl ( q ( m ) \pi ( { \cal E } \mid m ) \bigr ) } } \ ~ } \\ { { \displaystyle ~ = - \sum _ { m } q ( m ) \log q ( m ) \sum _ { { \cal E } \in \Omega _ { m } } \pi ( { \cal E } \mid m ) - \sum _ { m } q ( m ) \sum _ { { \cal E } \in \Omega _ { m } } \pi ( { \cal E } \mid m ) \log \pi ( { \cal E } \mid m ) } \ ~ } \\ { { \displaystyle ~ = { \cal H } ( q ) + \sum _ { m } q ( m ) { \cal H } ( \pi ( \cdot \mid m ) ) } , } \end{array}
$$

which proves Eq. (5).

For the second claim, let $u _ { m }$ denote the uniform distribution on $\Omega _ { m } , \mathrm { s o } u _ { m } ( E ) = 1 / n _ { m }$ . Nonnegativity of the Kullback–Leibler divergence gives

$$
0 \leq D _ { \mathrm { K L } } ( \pi ( \cdot \mid m ) \parallel u _ { m } ) = \sum _ { E \in \Omega _ { m } } \pi ( E \mid m ) \log \frac { \pi ( E \mid m ) } { 1 / n _ { m } } = - H ( \pi ( \cdot \mid m ) ) + \log n _ { m } .
$$

Hence $H ( \pi ( \cdot \mid m ) ) \leq \log n _ { m }$ . Equality in the KL inequality holds if and only if $\pi ( \cdot \mid m ) = u _ { m } .$ proving the equality condition. □

Proposition B.2 (Action-space multiplicity bias). For a fixed training instance, let $\{ \Omega _ { m } \} _ { m \in \mathcal { M } _ { u } }$ partition the feasible repair space into verifier-equivalent classes, with common utility $U _ { y } ( m ) =$ $\Phi _ { y } ( M _ { E } )$ for $E \in \Omega _ { m }$ and multiplicity $n _ { m } = | \Omega _ { m } |$ . Optimizing $\mathcal { I } _ { \mathrm { a c t } }$ over the unrestricted prob ability simplex induces $q _ { \mathrm { a c t } } ^ { \star } ( m ) \propto n _ { m } \exp ( U _ { y } ( m ) / \beta )$ . Equivalently, action entropy introduces an implicit class-level utility $\beta \log n _ { m }$ in addition to biological utility $U _ { y } ( m )$

Proof. For a fixed raw repair distribution π, phenotype utility depends only on its quotient marginal:

$$
\mathbb { E } _ { E \sim \pi } [ \Phi _ { y } ( M _ { E } ) ] = \sum _ { m \in \mathcal { M } _ { y } } \sum _ { E \in \Omega _ { m } } \pi ( E ) U _ { y } ( m ) = \sum _ { m \in \mathcal { M } _ { y } } q ( m ) U _ { y } ( m ) .\tag{7}
$$

Combining Eq. (7) with Lemma B.1, the fixed-instance action-entropy objective can be written as

$$
\mathcal { I } _ { \mathrm { a c t } } ( \pi ) = \sum _ { m } q ( m ) U _ { y } ( m ) + \beta H ( q ) + \beta \sum _ { m } q ( m ) H ( \pi ( \cdot \mid m ) ) .\tag{8}
$$

Now fix an arbitrary quotient marginal $q .$ . The first two terms in $\operatorname { E q . } \ ( 8 )$ are independent of the conditional distributions inside the fibers. By Eq. (6), the final term is maximized independently in every positive-mass fiber by

$$
\pi ^ { \star } ( E \mid m ) = \frac { 1 } { n _ { m } } , \qquad E \in \Omega _ { m } .
$$

Therefore optimization over the raw simplex $\Delta ( \mathcal { F } )$ is equivalent to

$$
\operatorname* { m a x } _ { \pi \in \Delta ( \mathcal { F } ) } \mathcal { I } _ { \mathrm { a c t } } ( \pi ) = \operatorname* { m a x } _ { q \in \Delta ( M _ { y } ) } \left\{ \sum _ { m } q ( m ) U _ { y } ( m ) + \beta H ( q ) + \beta \sum _ { m } q ( m ) \log n _ { m } \right\} .\tag{9}
$$

Define

$$
a _ { m } = n _ { m } \exp \left( \frac { U _ { y } ( m ) } { \beta } \right) , \qquad Z _ { \mathrm { a c t } } = \sum _ { m \in \mathcal { M } _ { y } } a _ { m } ,
$$

and

$$
\boldsymbol { q } _ { \mathrm { a c t } } ^ { \star } ( m ) = \frac { \boldsymbol { a } _ { m } } { Z _ { \mathrm { a c t } } } .
$$

Since every fiber is nonempty, $n _ { m } \ge 1$ , and all utilities are finite, $q _ { \mathrm { a c t } } ^ { \star } ( m ) > 0$ for every m. Using log $a _ { m } = \log n _ { m } + U _ { y } ( m ) / \beta$ , the objective in Eq. (9) becomes

$$
\begin{array} { r l } {  { \sum _ { m } q ( m ) U _ { y } ( m ) + \beta \sum _ { m } q ( m ) \log n _ { m } - \beta \sum _ { m } q ( m ) \log q ( m ) } } \\ & { = \beta \sum _ { m } q ( m ) \log \frac { a _ { m } } { q ( m ) } } \\ & { = \beta \log Z _ { \mathrm { a c t } } - \beta D _ { \mathrm { K L } } ( q \parallel q _ { \mathrm { a c t } } ^ { \star } ) . } \end{array}\tag{10}
$$

Because KL divergence is nonnegative and vanishes only when its two arguments coincide, Eq. (10) proves that the unique quotient marginal maximizing the objective is

$$
q _ { \mathrm { a c t } } ^ { \star } ( m ) = \frac { n _ { m } \exp ( U _ { y } ( m ) / \beta ) } { \sum _ { m ^ { \prime } \in { \mathcal { M } _ { y } } } n _ { m ^ { \prime } } \exp ( U _ { y } ( m ^ { \prime } ) / \beta ) } .\tag{11}
$$

Furthermore, because this optimum assigns positive mass to every fiber, Lemma B.1 requires the unique optimal within-fiber conditional to be uniform. Thus the corresponding raw-policy optimum satisfies

$$
\pi ^ { \star } ( E ) = \frac { q _ { \mathrm { a c t } } ^ { \star } ( m ) } { n _ { m } } = \frac { \exp ( U _ { y } ( m ) / \beta ) } { Z _ { \mathrm { a c t } } } , \qquad E \in \Omega _ { m } .
$$

Hence the multiplicity factor arises only when raw action probabilities are aggregated at the class level: each representative receives the same Boltzmann weight for a fixed utility, and a class containing $n _ { m }$ representatives accumulates $n _ { m }$ such contributions. Equivalently, action entropy contributes the implicit class utility $\beta$ log $n _ { m }$ claimed in the proposition. □

Lemma B.3 (Surjectivity of the quotient marginalization map). Let $T : \Delta ( \mathcal { F } )  \Delta ( \mathcal { M } _ { y } )$ be defined by

$$
[ T \pi ] ( m ) = \sum _ { E \in \Omega _ { m } } \pi ( E ) ,
$$

where $\Omega _ { m } = \{ E \in \mathcal { F } : \kappa _ { y } ( E ) = m \}$ . Then $T$ is surjective.

Proof. Because $\mathcal { M } _ { y } = \kappa _ { y } ( \mathcal { F } )$ , every $m \in \mathcal { M } _ { y }$ has at least one representative $r ( m ) \in \mathcal { F }$ satisfying $\kappa _ { y } ( r ( m ) ) = m$ . For any $\bar { q } \in \Delta ( \mathcal { M } _ { y } )$ , define

$$
\pi _ { q } ( E ) = { \left\{ \begin{array} { l l } { q ( m ) , } & { E = r ( m ) { \mathrm { ~ f o r ~ s o m e ~ } } m \in { \mathcal { M } } _ { y } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Since distinct fibers are disjoint, the selected representatives are distinct and $\pi _ { q } \in \Delta ( \mathcal { F } )$ . Moreover,

$$
[ T \pi _ { q } ] ( m ) = \sum _ { E \in \Omega _ { m } } \pi _ { q } ( E ) = \pi _ { q } ( r ( m ) ) = q ( m ) .
$$

Hence $T \pi _ { q } = q .$ , proving that $T$ is surjective.

Remark B.4 (Biological and probabilistic intuition). In probabilistic terms, the operator $T$ simply formalizes marginalization over functionally equivalent repairs: it aggregates the probabilities of all distinct reaction edit sets mapped to the same canonical repair mechanism, yielding the total probability $q ( m )$ of deploying mechanism $m .$ Biologically, surjectivity ensures that the quotient mechanism space $\Delta ( \mathcal { M } _ { y } )$ has no “ghost states”: any target distribution over metabolic repair mechanisms is physically realizable by an underlying reaction-level policy.

ProofofTheorem 4.1. Let $T : \Delta ( \mathcal { F } )  \Delta ( \mathcal { M } _ { y } )$ be the quotient marginalization operator from Lemma B.3.

Sufficiency. Suppose there exists $\overline { { \mathfrak { D } } } : \Delta ( \mathcal { M } _ { y } )  \mathbb { R }$ such that ${ \mathfrak { D } } ( \pi ) = { \overline { { \mathfrak { D } } } } ( T \pi )$ , for all $\pi \in \Delta ( \mathcal { F } )$ If two raw policies $\pi , \pi ^ { \prime }$ induce the same quotient marginal, $T \pi = \dot { T } \pi ^ { \prime }$ , then

$$
\mathfrak { D } ( \pi ) = \overline { { \mathfrak { D } } } ( T \pi ) = \overline { { \mathfrak { D } } } ( T \pi ^ { \prime } ) = \mathfrak { D } ( \pi ^ { \prime } ) .
$$

Hence $\mathfrak { D }$ is invariant to all probability redistributions that leave the quotient law unchanged.

Necessity. Conversely, assume that

$$
T \pi = T \pi ^ { \prime } \quad \Longrightarrow \quad \mathfrak { D } ( \pi ) = \mathfrak { D } ( \pi ^ { \prime } )
$$

for all $\pi , \pi ^ { \prime } \in \Delta ( { \mathcal { F } } )$ . By Lemma B.3, for every $q \in \Delta ( \mathcal { M } _ { y } )$ there exists at least one lift $\pi _ { q } \in \Delta ( \mathcal { F } )$ satisfying $T \pi _ { q } = q$ . Define

$$
\overline { { \mathfrak { D } } } ( q ) = \mathfrak { D } ( \pi _ { q } ) .\tag{12}
$$

We first verify that this definition is independent of the chosen lift. $\operatorname { I f } \pi _ { q }$ and $\widetilde { \pi } _ { q }$ both satisfy $T \pi _ { q } =$ $T \widetilde { \pi } _ { q } = q$ , then the assumed invariance gives $\mathfrak { D } ( \pi _ { q } ) = \mathfrak { D } ( \widetilde { \pi } _ { q } )$ . Thus Eq. (12) defines a well-defined functional on the entire quotient simplex.

Now take an arbitrary $\pi \in \Delta ( \mathcal { F } )$ and set $q = T \pi$ . Both π and the selected lift $\pi _ { q }$ induce the same quotient law. Therefore,

$$
{ \mathfrak { D } } ( \pi ) = { \mathfrak { D } } ( \pi _ { q } ) = { \overline { { { \mathfrak { D } } } } } ( q ) = { \overline { { { \mathfrak { D } } } } } ( T \pi ) ,
$$

which proves the required factorization.

It remains to prove uniqueness. Suppose two functionals $\overline { { \mathfrak { D } } } _ { 1 }$ and $\overline { { \mathfrak { D } } } _ { 2 }$ satisfy $\mathfrak { D } = \overline { { \mathfrak { D } } } _ { 1 } \circ T = \overline { { \mathfrak { D } } } _ { 2 } \circ T .$ For any $q \in \Delta ( \mathcal { M } _ { y } )$ , surjectivity of T provides a lift $\pi _ { q }$ with $T \pi _ { q } = q$ . Hence

$$
\overline { { \mathfrak { D } } } _ { 1 } ( q ) = \overline { { \mathfrak { D } } } _ { 1 } ( T \pi _ { q } ) = \mathfrak { D } ( \pi _ { q } ) = \overline { { \mathfrak { D } } } _ { 2 } ( T \pi _ { q } ) = \overline { { \mathfrak { D } } } _ { 2 } ( q ) .
$$

Therefore $\overline { { \mathfrak { D } } } _ { 1 } = \overline { { \mathfrak { D } } } _ { 2 } \mathrm { o n } \Delta ( \mathcal { M } _ { y } )$ . This establishes both existence and uniqueness.

Remark B.5 (Preservation under a representative lift). Beyond existence and uniqueness, the quotient factorization preserves basic analytical properties of the exploration functional on the probability simplex. Fix, for each $m \in \mathcal { M } _ { y } ,$ an arbitrary representative $r ( m ) \in \Omega _ { m } ,$ , so that $\kappa _ { y } ( r ( m ) ) = m$ as in Lemma B.3. This representative selection induces a lift $L _ { r } : \Delta ( \mathcal { M } _ { y } )  \Delta ( \mathcal { F } )$ defined by

$$
[ L _ { r } q ] ( E ) = \left\{ { q ( m ) } , \quad E = r ( m ) { \mathrm { ~ f o r ~ s o m e ~ } } m \in \mathcal { M } _ { y } , \quad \right.
$$

By construction, $T L _ { r } q = q$ for every $q \in \Delta ( \mathcal { M } _ { y } )$ . Moreover, $L _ { r }$ is linear:

$$
L _ { r } ( \alpha q _ { 1 } + ( 1 - \alpha ) q _ { 2 } ) = \alpha L _ { r } q _ { 1 } + ( 1 - \alpha ) L _ { r } q _ { 2 } , \qquad \alpha \in [ 0 , 1 ] .
$$

Therefore, for the quotient functional $\bar { \mathcal D }$ characterized by Theorem 4.1,

$$
\begin{array} { r } { \bar { \mathcal { D } } ( q ) = \mathcal { D } ( L _ { r } q ) , } \end{array}
$$

and continuity and convexity or concavity of D restrict directly to $\bar { \mathcal D }$ through this linear lift.

Corollary B.6 (Quotient Shannon entropy removes the multiplicity factor). For a fixed training instance, unrestricted maximization of $\textstyle \sum _ { m \in { \mathcal { M } } _ { y } } q ( m ) U _ { y } ( m ) { \dot { + } } \beta { \dot { H } } ( q )$ over $q \in \Delta ( \mathcal { M } _ { y } )$ has the unique solution

$$
q _ { \mathrm { q u o t } } ^ { \star } ( m ) = \frac { \exp ( U _ { y } ( m ) / \beta ) } { \sum _ { m ^ { \prime } \in { \mathcal { M } } _ { y } } \exp ( U _ { y } ( m ^ { \prime } ) / \beta ) } .\tag{13}
$$

Proof. Define

$$
Z _ { \mathrm { q u o t } } = \sum _ { m \in \mathcal { M } _ { y } } \exp \left( \frac { U _ { y } ( m ) } { \beta } \right) , \qquad q _ { \mathrm { q u o t } } ^ { \star } ( m ) = \frac { \exp ( U _ { y } ( m ) / \beta ) } { Z _ { \mathrm { q u o t } } } .
$$

For arbitrary $q \in \Delta ( \mathcal { M } _ { y } )$ ,

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { m } q ( m ) U _ { y } ( m ) + \beta H ( q ) = \beta \sum _ { m } q ( m ) \log \frac { \exp ( U _ { y } ( m ) / \beta ) } { q ( m ) } } \\ & { } & { \quad \quad = \beta \log Z _ { \mathrm { q u o t } } - \beta D _ { \mathrm { K L } } \big ( q \parallel q _ { \mathrm { q u o t } } ^ { \star } \big ) . } \end{array}\tag{14}
$$

The KL divergence is nonnegative and equals zero if and only if $q = q _ { \mathrm { q u o t } } ^ { \star }$ . Therefore Eq. (13) is the unique maximizer. □

Comparing this expression with the action-space optimum, $q _ { \mathrm { a c t } } ^ { \star } ( m ) \propto n _ { m } \exp ( U _ { y } ( m ) / \beta )$ , shows that quotient Shannon entropy removes the representation-volume factor $n _ { m }$ exactly.

Scope of the results. Lemma B.1 applies to any distribution on the finite feasible repair space under the partition induced by $\kappa _ { y }$ . The Gibbs solutions characterize unrestricted simplex optima and serve as variational references for the parameterized policy. The factorization theorem holds for any fixed verifier-preserving $\kappa _ { y } ,$ establishing rigorous representation invariance directly over the operational mechanism classes induced by the verifier across diverse biological hypotheses.

## B.3 COLLISION SPARSITY AND KERNEL RENYI´ ESTIMATION

Notation and setup. This subsection provides the formal results supporting Sec. 4.2, adopting the quotient setup from Appendix B.2. Unless stated otherwise, $m _ { 1 } , \ldots , m _ { L }$ are drawn IID from the quotient distribution q on the finite space $\mathcal { M } _ { y }$ . We write

$$
C ( q ) = \sum _ { m \in \mathcal { M } _ { y } } q ( m ) ^ { 2 } = \exp [ - H _ { 2 } ( q ) ]
$$

for the collision concentration, where $H _ { 2 }$ denotes categorical Renyi entropy of order two.´

We first characterize the collision-free regime as an extreme case of the broader finite-sample coarseness of exact-count statistics.

Proposition B.7 (Sparse-collision barrier). For L independently sampled repair cores $m _ { 1 } , \ldots , m _ { L } \sim q _ { \theta , y } ,$ , the pairwise collision count $\begin{array} { r } { N _ { \mathrm { c o l } } = \sum _ { i < j } { \bf 1 } [ m _ { i } \stackrel { \cdot } { = } m _ { j } ] } \end{array}$ satisfies $\mathbb { P } ( N _ { \mathrm { c o l } } >$ $\begin{array} { r } { 0 ) \leq \mathbb { E } [ N _ { \mathrm { c o l } } ] = \binom { L } { 2 } \sum _ { m \in \mathcal { M } _ { u } } q _ { \theta , y } ( m ) ^ { 2 } } \end{array}$ . Consequently, along any sequence of instances and policies for which $\begin{array} { r } { L ^ { 2 } \sum _ { m \in \mathcal { M } _ { u } } \bar { q _ { \theta , y } } ( m ) ^ { 2 } \to 0 , } \end{array}$ , the collision probability vanishes $( \mathbb { P } ( N _ { \mathrm { c o l } } > 0 )  0 )$ leaving all sampled candidates with identical zero exploration credit.

Proof. Writing $\begin{array} { r } { N _ { \mathrm { c o l } } = \sum _ { 1 < i < i < L } \mathbf { 1 } [ m _ { i } = m _ { j } ] } \end{array}$ , independence gives

$$
\mathbb { P } ( m _ { i } = m _ { j } ) = \sum _ { m \in \mathcal { M } _ { y } } q ( m ) ^ { 2 } = C ( q ) .
$$

Hence, by linearity of expectation,

$$
\mathbb { E } [ N _ { \mathrm { c o l } } ] = { \binom { L } { 2 } } C ( q ) .
$$

Since $N _ { \mathrm { c o l } }$ is nonnegative and integer valued, Markov’s inequality yields

$$
\mathbb { P } ( N _ { \mathrm { c o l } } > 0 ) \le \operatorname* { m i n } \left\{ 1 , \binom { L } { 2 } C ( q ) \right\} .
$$

More formally, along any sequence of quotient laws $q _ { n }$ and rollout sizes $L _ { n }$ satisfying

$$
L _ { n } ^ { 2 } C ( q _ { n } ) \longrightarrow 0 ,
$$

the preceding bound implies $\mathbb { P } ( N _ { \mathrm { c o l } } > 0 )  0$

If the group is collision-free, its empirical distribution assigns mass $1 / L$ to each sampled core, so both its plug-in Shannon entropy and categorical Renyi-2 entropy equal ´ log L. After removing any candidate i, the corresponding entropy satisfies

$$
H _ { - i } = \log ( L - 1 ) \qquad { \mathrm { f o r ~ a l l ~ } } i .
$$

Since this holds for every i, the mean across the group is $\begin{array} { r } { \bar { H } _ { - } \ = \ \frac { 1 } { L } \sum _ { j = 1 } ^ { L } H _ { - j } \ = \ \log ( L - 1 ) } \end{array}$ yielding a leave-one-out entropy contrast

$$
\bar { H } _ { - } - H _ { - i } = \log ( L - 1 ) - \log ( L - 1 ) = 0\tag{15}
$$

identically for all $i \in \{ 1 , \ldots , L \}$ . The same argument applies to any symmetric equality-only statistic whose leave-one-out value depends solely on collision patterns. Hence, under collision-free rollout groups, exact-match leave-one-out evaluation provides zero exploration credit to all sampled candidates. □

We now formalize the kernel relaxation used to replace exact collisions. Each repair core is identified with its canonical finite set of signed reaction edits, so equality of cores coincides with equality of their signed-edit sets. Define

$$
J ( m , m ^ { \prime } ) = \frac { | m \cap m ^ { \prime } | } { | m \cup m ^ { \prime } | } , \qquad J ( \emptyset , \emptyset ) = 1 ,
$$

and, for fixed bandwidth $h > 0$

$$
k _ { y } ( m , m ^ { \prime } ) = \exp \left( - \frac { 1 - J ( m , m ^ { \prime } ) } { h } \right) .
$$

The kernel rule is shared across instances and the subscript y indicates its restriction to the instancespecific quotient space $\mathcal { M } _ { y }$

Proposition B.8 (Validity and consistency of the signed-Jaccard relaxation). The kernel $k _ { y }$ is positive semidefinite and satisfie

$$
e ^ { - 1 / h } \leq k _ { y } ( m , m ^ { \prime } ) \leq 1 , \qquad k _ { y } ( m , m ) = 1 .
$$

Moreover, on thefinite quotient space $\mathcal { M } _ { y } , k _ { y } ( m , m ^ { \prime } )$ converges uniformly to the exact-match kernel $\mathbf { 1 } [ m = m ^ { \prime } ]$ as $h  0$

Define the population kernel concentration and its associated kernelized quadratic Renyi functional ´ by

$$
V _ { k } ( q ) = \mathbb { E } _ { m , m ^ { \prime } \sim q } [ k _ { y } ( m , m ^ { \prime } ) ] , \qquad H _ { 2 , k } ( q ) = - \log V _ { k } ( q ) .\tag{16}
$$

Then $C ( q ) \leq V _ { k } ( q ) \leq 1 , 0 \leq H _ { 2 , k } ( q ) \leq H _ { 2 } ( q )$ , and as $h  0 ,$

$$
V _ { k } ( q ) \longrightarrow C ( q ) , \qquad H _ { 2 , k } ( q ) \longrightarrow H _ { 2 } ( q ) .
$$

Proof. We first establish positive semidefiniteness of the Jaccard similarity. Let $\ d \mathcal { U } _ { \pm }$ denote the finite universe of signed reaction edits, and let ζ be a uniformly random permutation of $\ d \mathcal { U } _ { \pm }$ . For each nonempty core m, let $b _ { \zeta } ( m )$ be its minimum element under ζ, assigning the empty core a distinguished sentinel symbol.

For any finite collection $m _ { 1 } , \ldots , m _ { n }$ , the matrix

$$
G _ { a b } ^ { \zeta } = { \bf 1 } [ b _ { \zeta } ( m _ { a } ) = b _ { \zeta } ( m _ { b } ) ]
$$

is a Gram matrix of one-hot vectors and is therefore positive semidefinite. For two nonempty cores, the standard minwise-hashing identity gives

$$
\mathbb { E } _ { \zeta } [ G _ { a b } ^ { \zeta } ] = \frac { \left| m _ { a } \cap m _ { b } \right| } { \left| m _ { a } \cup m _ { b } \right| } = J ( m _ { a } , m _ { b } ) ,
$$

with the sentinel convention giving the stated values involving the empty core (Broder, 1997; Li et al., 2011). Hence the Jaccard Gram matrix Q is positive semidefinite.

For the same collection, the Gram matrix K of $k _ { y }$ can be expressed via the Hadamard exponential as

$$
K = e ^ { - 1 / h } \exp ^ { \odot } ( Q / h ) = e ^ { - 1 / h } \sum _ { r = 0 } ^ { \infty } \frac { Q ^ { \odot r } } { h ^ { r } r ! } .
$$

By the Schur product theorem, every Hadamard power $Q ^ { \odot r }$ is positive semidefinite. Thus every finite partial sum is positive semidefinite; since the positive-semidefinite cone is closed and the series converges in finite dimension, $K \succeq 0$ . The bounds follow directly from $0 \leq J \leq 1$

For the small-bandwidth limit, the case $| \mathcal { M } _ { y } | = 1$ is immediate. I $\mathrm { ~ f ~ } | \mathcal { M } _ { y } | > 1$ , finiteness of $\mathcal { M } _ { y }$ and the canonical signed-edit representation guarantee

$$
\gamma _ { y } = \operatorname* { m i n } _ { m \neq m ^ { \prime } } \bigl ( 1 - J ( m , m ^ { \prime } ) \bigr ) > 0 .
$$

Therefore,

$$
\operatorname* { s u p } _ { m \neq m ^ { \prime } } k _ { y } ( m , m ^ { \prime } ) \leq e ^ { - \gamma _ { y } / h } \longrightarrow 0 \quad \mathrm { a s ~ } h  0 ,\tag{17}
$$

while $k _ { y } ( m , m ) = 1$ for all $h > 0$ , proving uniform convergence to the exact-match kernel.

Finally, expanding $V _ { k } ( q )$ into diagonal and off-diagonal terms gives

$$
V _ { k } ( q ) = \sum _ { m } q ( m ) ^ { 2 } + \sum _ { m \neq m ^ { \prime } } q ( m ) q ( m ^ { \prime } ) k _ { y } ( m , m ^ { \prime } ) = C ( q ) + \sum _ { m \neq m ^ { \prime } } q ( m ) q ( m ^ { \prime } ) k _ { y } ( m , m ^ { \prime } ) \geq C ( q ) ,
$$

while $V _ { k } ( q ) \leq 1$ . Taking negative logarithms yields $0 \leq H _ { 2 , k } ( q ) \leq H _ { 2 } ( q )$

Note that the total off-diagonal probability mass satisfies

$$
\sum _ { m \neq m ^ { \prime } } q ( m ) q ( m ^ { \prime } ) = \left( \sum _ { m } q ( m ) \right) ^ { 2 } - \sum _ { m } q ( m ) ^ { 2 } = 1 - C ( q ) .
$$

Consequently, factoring out the off-diagonal supremum yields

$$
\begin{array} { r l r } {  { 0 \leq V _ { k } ( q ) - C ( q ) = \displaystyle \sum _ { m \neq m ^ { \prime } } q ( m ) q ( m ^ { \prime } ) k _ { y } ( m , m ^ { \prime } ) } } \\ & { } & { \leq ( \displaystyle \operatorname* { s u p } _ { m \neq m ^ { \prime } } k _ { y } ( m , m ^ { \prime } ) ) \displaystyle \sum _ { m \neq m ^ { \prime } } q ( m ) q ( m ^ { \prime } ) } \\ & { } & { = ( 1 - C ( q ) ) \displaystyle \operatorname* { s u p } _ { m \neq m ^ { \prime } } k _ { y } ( m , m ^ { \prime } ) } \\ & { } & { \leq \displaystyle \operatorname* { s u p } _ { m \neq m ^ { \prime } } k _ { y } ( m , m ^ { \prime } ) \leq e ^ { - \gamma _ { y } / h } . } \end{array}\tag{18}
$$

As $h  0 .$ , this upper bound vanishes by Eq. (17), so ${ \cal V } _ { k } ( q ) \to { \cal C } ( q )$ . Because $\mathcal { M } _ { y }$ is finite and nonempty, Cauchy–Schwarz ensures $\begin{array} { r } { C ( q ) = \sum _ { m } q ( m ) ^ { 2 } \geq 1 / | { \mathcal M } _ { y } | > 0 ; } \end{array}$ ; thus continuity of − log on (0, 1] implies $H _ { 2 , k } ( q ) \to H _ { 2 } ( q )$ □

Theorem B.9 (Finite-group estimation and consistency). For $L \geq 2 ,$ , define

$$
\widehat V _ { L } = \frac { 1 } { L ( L - 1 ) } \sum _ { i \neq j } k _ { y } ( m _ { i } , m _ { j } ) , \qquad \widehat H _ { 2 , k } = - \log \widehat V _ { L } , \qquad J _ { L } ( q ) = { \mathbb E } [ \widehat H _ { 2 , k } ] .
$$

Then $\widehat { V } _ { L }$ is an unbiased estimator of the population kernel concentration,

$$
\mathbb { E } [ \widehat { V } _ { L } ] = V _ { k } ( q ) ,
$$

and,for everyfixed bandwidth $h > 0$

$$
0 \leq J _ { L } ( q ) - H _ { 2 , k } ( q ) \leq B _ { L } ( h ) , \qquad B _ { L } ( h ) = \frac { ( e ^ { 1 / h } - 1 ) ^ { 2 } } { 4 L } .
$$

Hence,forfixed $h , J _ { L } ( q )$ converges uniformly over quotient laws to $H _ { 2 , k } ( q )$ at rate $O ( L ^ { - 1 } )$

Proof. The statistic $\widehat { V } _ { L }$ is the order-two U-statistic

$$
\widehat V _ { L } = \frac { 2 } { L ( L - 1 ) } \sum _ { i < j } k _ { y } ( m _ { i } , m _ { j } ) ,
$$

so independence immediately gives $\mathbb { E } [ \widehat { V } _ { L } ] = V _ { k } ( q )$ . Since − log is convex, Jensen’s inequality implies

$$
J _ { L } ( q ) = \mathbb { E } [ - \log \widehat { V } _ { L } ] \geq - \log V _ { k } ( q ) = H _ { 2 , k } ( q ) .
$$

Let $X = \widehat { V } _ { L } , V = V _ { k } ( q )$ , and recall that X, $V \in [ e ^ { - 1 / h } , 1 ]$ . For $f ( v ) = - \log v , f ^ { \prime \prime } ( v ) = 1 / v ^ { 2 }$ By Taylor’s theorem, for some ξ between X and V,

$$
f ( X ) = f ( V ) + f ^ { \prime } ( V ) ( X - V ) + \frac { f ^ { \prime \prime } ( \xi ) } { 2 } ( X - V ) ^ { 2 } .
$$

Since $\xi \in [ e ^ { - 1 / h } , 1 ]$ , we have $f ^ { \prime \prime } ( \xi ) \leq e ^ { 2 / h }$ , so

$$
0 \leq f ( X ) - f ( V ) - f ^ { \prime } ( V ) ( X - V ) \leq \frac { e ^ { 2 / h } } { 2 } ( X - V ) ^ { 2 } .
$$

Taking expectations and using $\mathbb { E } [ X ] = V$ yields

$$
0 \leq J _ { L } ( q ) - H _ { 2 , k } ( q ) \leq { \frac { e ^ { 2 / h } } { 2 } } \mathrm { V a r } ( \widehat { V } _ { L } ) .\tag{19}
$$

To bound the variance, let $g ( m , m ^ { \prime } ) = k _ { y } ( m , m ^ { \prime } ) , g _ { 1 } ( m ) = \mathbb { E } _ { m ^ { \prime } } [ g ( m , m ^ { \prime } ) ] - V$ , and define the degenerate second-order projection $g _ { 2 } ( \tilde { m , } \tilde { m ^ { \prime } } ) = g ( m , \tilde { m ^ { \prime } } ) - V { \bar { - } } g _ { 1 } ( m ) { \bar { - } } g _ { 1 } ( m ^ { \prime } )$ . Let $\zeta _ { 1 } =$ $\mathbb { E } [ \breve { g } _ { 1 } ( m ) ^ { 2 } ] , \zeta _ { 2 } = \mathbb { E } [ g _ { 2 } ( m , \breve { m ^ { \prime } } ) ^ { 2 } ]$ . The Hoeffding decomposition for an order-two U-statistic gives

$$
\mathrm { V a r } ( \widehat { V } _ { L } ) = \frac { 4 } { L } \zeta _ { 1 } + \frac { 2 } { L ( L - 1 ) } \zeta _ { 2 } ,
$$

while orthogonality of the projections implies $2 \zeta _ { 1 } + \zeta _ { 2 } = \mathrm { V a r } ( g ( m , m ^ { \prime } ) )$ . Therefore, for $L \geq 2$

$$
\mathrm { V a r } ( \widehat { V } _ { L } ) \le \frac { 2 } { L } ( 2 \zeta _ { 1 } + \zeta _ { 2 } ) = \frac { 2 } { L } \mathrm { V a r } ( k _ { y } ( m , m ^ { \prime } ) ) \le \frac { ( 1 - e ^ { - 1 / h } ) ^ { 2 } } { 2 L } ,\tag{20}
$$

where the last inequality follows from Popoviciu’s inequality since $k _ { y } ( m , m ^ { \prime } ) \in [ e ^ { - 1 / h } , 1 ]$ . Substituting Eq. (20) into Eq. (19) gives

$$
J _ { L } ( q ) - H _ { 2 , k } ( q ) \leq \frac { e ^ { 2 / h } ( 1 - e ^ { - 1 / h } ) ^ { 2 } } { 4 L } = \frac { ( e ^ { 1 / h } - 1 ) ^ { 2 } } { 4 L } .
$$

Therefore, for every fixed $h > 0 .$

$$
\operatorname* { s u p } _ { \boldsymbol { q } } | J _ { L } ( \boldsymbol { q } ) - H _ { 2 , k } ( \boldsymbol { q } ) | = O ( L ^ { - 1 } ) ,
$$

which establishes uniform consistency over quotient laws.

Scope of the result. Theorem B.9 establishes uniform $O ( L ^ { - 1 } )$ consistency for fixed bandwidth h. The distribution-free bound is a conservative worst-case guarantee and is not intended as a tight characterization of finite rollout groups. The limit $H _ { 2 , k } ( q ) \stackrel { - } {  } H _ { 2 } ( q )$ is a population result, while signed-Jaccard similarity quantifies structural overlap rather than biological equivalence.

## B.4 QUOTIENTPO OPTIMIZATION THEORY

This subsection establishes the theoretical role of the quotient-diversity term in Eq. (4). We deliberately separate this analysis from the verifier-shaped reward, GRPO clipping, and auxiliary discovery and retention mechanisms, which are optimization choices detailed in Appendix A. Our goal here is twofold: to characterize the candidate-level crowding cost in $\mathbf { E q } . ( 3 )$ , and to relate its policy update to the population kernel Renyi functional introduced in Appendix B.3.´

Notation and setup. In this subsection, fix a training instance and suppress its conditioning. For a rollout group $m _ { 1 : L }$ , recall the group kernel concentration $\begin{array} { r } { \widehat { V } _ { L } = \frac { 1 } { L ( L - 1 ) } \sum _ { a \neq b } k _ { y } ( m _ { a } , m _ { b } ) } \end{array}$ from Theorem B.9, and define the candidate-level average kernel crowding

$$
\bar { k } _ { i } = \frac { 1 } { L - 1 } \sum _ { j \neq i } k _ { y } ( m _ { i } , m _ { j } ) , \qquad \mathrm { s o ~ t h a t } \quad \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \bar { k } _ { i } = \widehat { V } _ { L } .\tag{21}
$$

Proposition B.10 (Candidate crowding from leave-one-out kernel entropy). For $L \geq 3 ,$ let ${ \widehat { H } } _ { - i } =$ $- \log { \widehat { V } } _ { - i }$ be the kernel Renyi entropy of the group after removing candidate ´ i, where

$$
\widehat { V } _ { - i } = \frac { L \widehat { V } _ { L } - 2 \bar { k } _ { i } } { L - 2 } .
$$

Then the leave-one-out contrast $\Delta _ { i } = \widehat { H } _ { 2 , k } - \widehat { H } _ { - i }$ satisfies

$$
\Delta _ { i } = \log \left( 1 - \frac { 2 ( \bar { k } _ { i } - \widehat { V } _ { L } ) } { ( L - 2 ) \widehat { V } _ { L } } \right) .\tag{22}
$$

As $L  \infty ,$ the first-order Taylor expansion around the group mean $\widehat { V } _ { L }$ yields

$$
\Delta _ { i } = - \frac { c _ { i } } { L - 2 } + \mathcal { O } \left( \frac { 1 } { L ^ { 2 } } \right) ,\tag{23}
$$

where $\begin{array} { r } { c _ { i } = \frac { 2 \bar { k } _ { i } } { \widehat { V } _ { L } } - 2 } \end{array}$ is the crowding cost in Eq. (3). Moreover,

$$
\sum _ { i = 1 } ^ { L } c _ { i } = 0 , \quad c _ { i } > 0 \Longleftrightarrow \bar { k } _ { i } > \widehat { V } _ { L } , \quad c _ { i } < 0 \Longleftrightarrow \bar { k } _ { i } < \widehat { V } _ { L } .
$$

Proof. By the kernel symmetry $\begin{array} { r c l } { k _ { y } ( m , m ^ { \prime } ) } & { = } & { k _ { y } ( m ^ { \prime } , m ) } \end{array}$ , removing candidate i deletes $\begin{array} { r } { 2 \sum _ { j \neq i } k _ { y } ( m _ { i } , m _ { j } ) = 2 ( L - 1 ) \bar { k } _ { i } } \end{array}$ off-diagonal mass from the total group mass $L ( L - 1 ) \widehat { V } _ { L }$ . The remaining $( L - 1 ) ( L - 2 )$ ordered pairs have average concentration

$$
\widehat V _ { - i } = { \frac { L ( L - 1 ) \widehat V _ { L } - 2 ( L - 1 ) \bar { k } _ { i } } { ( L - 1 ) ( L - 2 ) } } = { \frac { L \widehat V _ { L } - 2 \bar { k } _ { i } } { L - 2 } } = \widehat V _ { L } - { \frac { 2 ( \bar { k } _ { i } - \widehat V _ { L } ) } { L - 2 } } .\tag{24}
$$

By Proposition B.8, $k _ { y } ( m , m ^ { \prime } ) \geq e ^ { - 1 / h } > 0$ holds uniformly, guaranteeing that $\widehat { V } _ { - i } \geq e ^ { - 1 / h } > 0$ and the logarithm is strictly well-defined. Dividing Eq. (24) by $\widehat { V } _ { L }$ and evaluating the leave-one-out entropy contrast gives

$$
\Delta _ { i } = \widehat { H } _ { 2 , k } - \widehat { H } _ { - i } = - \log \widehat { V } _ { L } - ( - \log \widehat { V } _ { - i } ) = \log \left( \frac { \widehat { V } _ { - i } } { \widehat { V } _ { L } } \right) = \log \left( 1 - \frac { 2 ( \bar { k } _ { i } - \widehat { V } _ { L } ) } { ( L - 2 ) \widehat { V } _ { L } } \right) ,
$$

which establishes Eq. (22).

Next, let $\begin{array} { r } { x _ { i } = \frac { 2 ( \bar { k } _ { i } - \widehat { V } _ { L } ) } { ( L - 2 ) \widehat { V } _ { L } } } \end{array}$ . Since $| \bar { k } _ { i } - \widehat { V } _ { L } | \leq 1 - e ^ { - 1 / h }$ and $\widehat { V } _ { L } \geq e ^ { - 1 / h }$ , we have $x _ { i } = \mathcal { O } ( L ^ { - 1 } )$ Applying the Taylor expansion log $( 1 - x ) = - x + \mathcal { O } ( x ^ { 2 } )$ around $x = 0$ yields

$$
\Delta _ { i } = - x _ { i } + \mathcal { O } ( x _ { i } ^ { 2 } ) = - \frac { 2 ( \bar { k } _ { i } - \widehat { V } _ { L } ) } { ( L - 2 ) \widehat { V } _ { L } } + \mathcal { O } ( L ^ { - 2 } ) = - \frac { c _ { i } } { L - 2 } + \mathcal { O } ( L ^ { - 2 } ) ,
$$

which establishes Eq. (23) and shows that the candidate crowding cost $\begin{array} { r } { c _ { i } = \frac { 2 \bar { k } _ { i } } { \widehat { V } _ { L } } - 2 } \end{array}$ governs the leading leave-one-out entropy rate.

Finally, using the definition $\begin{array} { r } { \sum _ { i = 1 } ^ { L } \bar { k } _ { i } = L \widehat { V } _ { L } } \end{array}$ from Eq. (3), the group sum of the costs satisfies

$$
\sum _ { i = 1 } ^ { L } c _ { i } = { \frac { 2 } { \widehat { V } _ { L } } } \sum _ { i = 1 } ^ { L } \bar { k } _ { i } - 2 L = { \frac { 2 } { \widehat { V } _ { L } } } ( L \widehat { V } _ { L } ) - 2 L = 0 .\tag{25}
$$

Rewriting the cost as $\begin{array} { r } { c _ { i } = \frac { 2 } { \widehat { V } _ { I } } ( \bar { k } _ { i } - \widehat { V } _ { L } ) } \end{array}$ with strictly positive normalizer $\widehat V _ { L } > 0$ , the signs satisfy $c _ { i } > 0 \Longleftrightarrow \bar { k } _ { i } > \widehat { V } _ { L }$ and $c _ { i } < 0 \Longleftrightarrow \bar { k } _ { i } < \widehat { V } _ { L }$ directly by affine monotonicity. □

The preceding result gives a finite-group interpretation of the implemented cost. We next show that the same quantity converges to the score coefficient of the population kernel Renyi objective. Define´ the population kernel density of a repair core m by

$$
d _ { q } ( m ) = \mathbb { E } _ { m ^ { \prime } \sim q } [ k _ { y } ( m , m ^ { \prime } ) ] ,
$$

so that

$$
V _ { k } ( q ) = \mathbb { E } _ { m \sim q } [ d _ { q } ( m ) ] .
$$

Define the corresponding population crowding coefficient

$$
c _ { q } ^ { k } ( m ) = 2 \frac { d _ { q } ( m ) } { V _ { k } ( q ) } - 2 .\tag{26}
$$

Theorem B.11 (Population score identity and local consistency). Assume that π<sub>θ</sub> is differentiable with locally fixed support and integrable score $\mathbb { E } _ { E \sim \pi _ { \theta } } [ \lVert g _ { \theta } ( E ) \rVert ] < \infty$ , and that $\kappa _ { y }$ and $k _ { y }$ remain fixed during the policy update. Let $g _ { \theta } ( E ) = \nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( E \mid M _ { 0 } , G )$ and let $q _ { \theta , y }$ denote the distribution of repair cores $m = \kappa _ { y } ( E )$ induced by $E \sim \pi _ { \theta }$ (abbreviated as q). Then:

(i) Score Identity: The population kernel Renyi functional satisfies´

$$
\nabla _ { \boldsymbol { \theta } } H _ { 2 , k } ( q _ { \boldsymbol { \theta } , \boldsymbol { y } } ) = - \mathbb { E } _ { E \sim \pi _ { \boldsymbol { \theta } } } \left[ c _ { q _ { \boldsymbol { \theta } , \boldsymbol { y } } } ^ { k } ( \kappa _ { \boldsymbol { y } } ( E ) ) g _ { \boldsymbol { \theta } } ( E ) \right] .\tag{27}
$$

(ii) Local Consistency: For the empirical crowding cost $c _ { i }$ in $E q . ( 2 3 ) ,$ we have $| c _ { i } - c _ { q } ^ { k } ( m _ { i } ) | $ 0 in probability as $L  \infty$ . Consequently, for the detached quotient-diversity surrogate $\begin{array} { r } { \mathcal { T } _ { k , L } ( \theta ) = - \frac { \beta } { L } \sum _ { i = 1 } ^ { L } \rho _ { i } ( \theta ) \mathrm { s g } ( c _ { i } ) } \end{array}$ in Eq. (4), its expected gradient at the behavior policy satisfies

$$
\operatorname* { l i m } _ { L \to \infty } \mathbb { E } \left[ \nabla _ { \theta } \mathcal { T } _ { k , L } ( \theta ) \big | _ { \theta = \theta _ { \mathrm { o l d } } } \right] = \beta \nabla _ { \theta } H _ { 2 , k } ( q _ { \theta _ { \mathrm { o l d } } , y } ) .\tag{28}
$$

Thus the QuotientPO kernel term is a locally consistent score surrogate for population quotient diversity as the rollout group grows.

Proof. Using the deterministic core map $m = \kappa _ { y } ( E )$ , the population concentration can be written as

$$
V _ { k } ( q _ { \theta , y } ) = \mathbb { E } _ { E , E ^ { \prime } \sim \pi _ { \theta } } \left[ k _ { y } ( \kappa _ { y } ( E ) , \kappa _ { y } ( E ^ { \prime } ) ) \right] .
$$

Differentiating the two IID policy factors gives

$$
\nabla _ { \theta } V _ { k } ( q _ { \theta , y } ) = 2 \mathbb { E } _ { E \sim \pi _ { \theta } } \left[ d _ { q } ( \kappa _ { y } ( E ) ) g _ { \theta } ( E ) \right] .
$$

Since $H _ { 2 , k } ( q ) = - \log V _ { k } ( q )$ , we have

$$
\nabla _ { \theta } H _ { 2 , k } ( q _ { \theta , y } ) = - \frac { 2 } { V _ { k } ( q _ { \theta , y } ) } \mathbb { E } _ { E \sim \pi _ { \theta } } \left[ d _ { q } ( \kappa _ { y } ( E ) ) g _ { \theta } ( E ) \right] .
$$

The score identity $\mathbb { E } _ { E \sim \pi _ { \theta } } [ g _ { \theta } ( E ) ] = 0$ allows baseline subtraction without altering the expectation, yielding

$$
\nabla _ { \theta } H _ { 2 , k } ( q _ { \theta , y } ) = - \mathbb { E } _ { E \sim \pi _ { \theta } } \left[ \left( 2 \frac { d _ { q } ( \kappa _ { y } ( E ) ) } { V _ { k } ( q _ { \theta , y } ) } - 2 \right) g _ { \theta } ( E ) \right] = - \mathbb { E } _ { E \sim \pi _ { \theta } } \left[ c _ { q } ^ { k } ( \kappa _ { y } ( E ) ) g _ { \theta } ( E ) \right] ,
$$

which proves Eq. (27).

Next, we establish the local consistency of the empirical surrogate. Conditional on $m _ { i }$ , the remaining $L - 1$ cores are IID draws from $q . \mathrm { \ B y }$ the weak law of large numbers, $\bar { k } _ { i }  d _ { q } ( m _ { i } )$ in probability as $L  \infty$ . In addition, the bounded order-two U-statistic satisfies $\widehat { V } _ { L } \to V _ { k } ( q )$ in probability. Because $V _ { k } ( q ) \geq e ^ { - 1 / h } > 0$ 0 is bounded away from zero, the continuous mapping theorem implies that $| c _ { i } - c _ { q } ^ { k } ( { \dot { m } } _ { i } ) | \to 0$ in probability as $L \to \infty$

Finally, evaluating the surrogate gradient at the behavior policy gives $\nabla _ { \theta } \rho _ { i } ( \theta ) | _ { \theta = \theta _ { \mathrm { o l d } } } = g _ { \theta _ { \mathrm { o l d } } } ( E _ { i } )$ By the exchangeability of the IID rollout sample $( E _ { 1 } , \dots , E _ { L } )$

$$
\mathbb { E } \left[ \nabla _ { \theta } \mathcal { I } _ { k , L } ( \theta ) \big | _ { \theta = \theta _ { \mathrm { o l d } } } \right] = - \beta \mathbb { E } \left[ \frac { 1 } { L } \sum _ { i = 1 } ^ { L } c _ { i } g _ { \theta _ { \mathrm { o l d } } } ( E _ { i } ) \right] = - \beta \mathbb { E } \left[ c _ { 1 } ^ { k } g _ { \theta _ { \mathrm { o l d } } } ( E _ { 1 } ) \right] .
$$

Since $k _ { y } \in [ e ^ { - 1 / h } , 1 ]$ , the statistics $\bar { k } _ { 1 }$ and $\widehat { V } _ { L }$ both lie in $[ e ^ { - 1 / h } , 1 ]$ , so $c _ { 1 } ^ { k }$ is uniformly bounded in L by $2 ( e ^ { 1 / h } - 1 )$ . Combined with score integrability $\mathbb { E } [ \| g _ { \theta _ { \mathrm { o l d } } } ( E _ { 1 } ) \| ] < \infty$ , the dominated convergence theorem implies

$$
\operatorname* { l i m } _ { L \to \infty } \mathbb { E } \left[ c _ { 1 } ^ { k } g _ { \theta _ { \mathrm { o l d } } } ( E _ { 1 } ) \right] = \mathbb { E } \left[ c _ { q } ^ { k } ( \kappa _ { y } ( E _ { 1 } ) ) g _ { \theta _ { \mathrm { o l d } } } ( E _ { 1 } ) \right] .
$$

Substituting this limit back into the surrogate expectation and applying Eq. (27) directly yields Eq. (28). □

Relation to deployment-time success and mechanism coverage. Because $k$ is already used for the similarity kernel, let $K$ denote the number of independently generated repairs in the evaluation metric success@K. For a fixed evaluation instance x, let $s _ { x } ( E ) \in \{ 0 , 1 \}$ denote the evaluation success predicate and define the single-draw success probability

$$
Z _ { x } ( \theta ) = \mathbb { P } _ { E \sim \pi _ { \theta } ( \cdot | x ) } [ s _ { x } ( E ) = 1 ] .
$$

Proposition B.12 (Successful quotient mass determines Success@K). For $K \geq 1$ IID repairs sampled from $\pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ , the probability that at least one repair succeeds is

$$
P _ { K } ( x ; \theta ) = 1 - \big ( 1 - Z _ { x } ( \theta ) \big ) ^ { K } .\tag{29}
$$

Therefore $P _ { K } \left( x ; \theta \right)$ is monotonically increasing in $Z _ { x } ( \theta )$ , and strictly increasing whenever $Z _ { x } ( \theta ) <$ 1. If the evaluation success predicate is invariant within repair cores, i.e., there exists $s _ { x } ^ { \mathrm { c o r e } }$ such that $s _ { x } ( E ) = s _ { x } ^ { \mathrm { c o r e } } ( \kappa _ { y } ( E ) )$ , then

$$
Z _ { x } ( \theta ) = \sum _ { m \in \mathcal { M } _ { y } } q _ { \theta , y } ( m ) s _ { x } ^ { \mathrm { c o r e } } ( m ) .\tag{30}
$$

Thus success@K is determined by the total quotient probability mass assigned to successful repair cores.

Proof. A single repair fails with probability $1 - Z _ { x } ( \theta )$ . Under IID sampling, the probability that all K repairs fail is therefore

$$
\left( 1 - Z _ { x } ( \theta ) \right) ^ { K } .
$$

Taking the complement proves Eq. (29). Moreover,

$$
\frac { \partial P _ { K } ( x ; \theta ) } { \partial Z _ { x } } = K ( 1 - Z _ { x } ) ^ { K - 1 } ,
$$

which is positive for $Z _ { x } < 1$

Under core invariance of the success predicate,

$$
\begin{array} { l } { \displaystyle Z _ { x } ( \theta ) = \sum _ { E } \pi _ { \theta } ( E \mid x ) s _ { x } ^ { \mathrm { c o r e } } ( \kappa _ { y } ( E ) ) } \\ { = \displaystyle \sum _ { m \in \mathcal { M } _ { y } } s _ { x } ^ { \mathrm { c o r e } } ( m ) \sum _ { E : \kappa _ { y } ( E ) = m } \pi _ { \theta } ( E \mid x ) } \\ { = \displaystyle \sum _ { m \in \mathcal { M } _ { y } } q _ { \theta , y } ( m ) s _ { x } ^ { \mathrm { c o r e } } ( m ) , } \end{array}
$$

which proves Eq. (30).

Proposition B.13 (Expected distinct successful-core coverage). Let

$$
\mathcal { M } _ { x } ^ { \mathrm { s u c c } } = \{ m \in \mathcal { M } _ { y } : s _ { x } ^ { \mathrm { c o r e } } ( m ) = 1 \}
$$

denote the successful repair cores for instance x. For K IID repairs $E _ { 1 } , \dots , E _ { K } \sim \pi _ { \theta } ( \cdot \mid x )$ , let

$$
D _ { K } = | \{ \kappa _ { y } ( E _ { t } ) : 1 \leq t \leq K , s _ { x } ( E _ { t } ) = 1 \} |
$$

be the number ofdistinct successful cores discovered within the K samples. Then

$$
\mathbb { E } [ D _ { K } ] = \sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } \left[ 1 - \left( 1 - q _ { \theta , y } ( m ) \right) ^ { K } \right] .
$$

Thus, unlike Success@K, distinct successful-core coverage depends not only on the total successful mass

$$
Z _ { x } ( \theta ) = \sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } q _ { \theta , y } ( m ) ,
$$

but also on how that mass is distributed across successful cores.

Proof. For each successful core $m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } }$ , define

$$
I _ { m } = \mathbf { 1 } \left[ m \in \{ \kappa _ { y } ( E _ { 1 } ) , \dots , \kappa _ { y } ( E _ { K } ) \} \right] .
$$

Then

$$
D _ { K } = \sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } I _ { m } .
$$

Because each sampled repair induces core m with probability $q _ { \theta , y } ( m )$ , independence gives

$$
\mathrm { P r } ( I _ { m } = 0 ) = \left( 1 - q _ { \theta , y } ( m ) \right) ^ { K } .
$$

Hence

$$
\mathbb { E } [ I _ { m } ] = 1 - \left( 1 - q _ { \theta , y } ( m ) \right) ^ { K } .
$$

Applying linearity of expectation yields

$$
\mathbb { E } [ D _ { K } ] = \sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } \left[ 1 - \left( 1 - q _ { \theta , y } ( m ) \right) ^ { K } \right] .
$$

Corollary B.14 (Successful-mass equalization improves mechanism coverage). Fix an instance x, a sampling budget $K > 1$ , and a set of S successful repair cores. For a fixed total successful mass

$$
\sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } q _ { \theta , y } ( m ) = Z ,
$$

the expected distinct successful-core coverage

$$
\mathbb { E } [ D _ { K } ] = \sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } \left[ 1 - \left( 1 - q _ { \theta , y } ( m ) \right) ^ { K } \right]
$$

is Schur-concave in the successful-core probability vector. Consequently, any mean-preserving redistribution that equalizes probability mass across successful cores weakly increases expected coverage. The maximum is attained at the uniform allocation

$$
q _ { \theta , y } ( m ) = \frac { Z } { S } ,
$$

yielding

$$
\mathbb { E } [ D _ { K } ] = S \left[ 1 - \left( 1 - \frac { Z } { S } \right) ^ { K } \right] .
$$

Meanwhile, Success@ $K$ remains

$$
1 - ( 1 - Z ) ^ { K } ,
$$

and therefore depends only on the total successful mass $Z .$

Proof. For $K > 1$ , define

$$
f _ { K } ( p ) = 1 - ( 1 - p ) ^ { K } .
$$

Since

$$
f _ { K } ^ { \prime \prime } ( p ) = - K ( K - 1 ) ( 1 - p ) ^ { K - 2 } \leq 0 ,
$$

$f _ { K }$ is concave on [0, 1]. Therefore,

$$
\mathbb { E } [ D _ { K } ] = \sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } f _ { K } ( q _ { \theta , y } ( m ) )
$$

is a symmetric Schur-concave function of the successful-core probability vector. Hence, any meanpreserving equalization of this vector weakly increases expected coverage. Under the fixed-sum constraint

$$
\sum _ { m \in \mathcal { M } _ { x } ^ { \mathrm { s u c c } } } q _ { \theta , y } ( m ) = Z ,
$$

the maximum is attained at the uniform allocation

$$
q _ { \theta , y } ( m ) = \frac { Z } { S } .
$$

Substituting this value gives

$$
\mathbb { E } [ D _ { K } ] = S \left[ 1 - \left( 1 - \frac { Z } { S } \right) ^ { K } \right] .
$$

By Proposition B.12, Success@K is

$$
1 - ( 1 - Z ) ^ { K } ,
$$

which depends only on the total successful mass $Z .$

Interpretation and scope. Propositions B.12–B.13 reveal two complementary dimensions of deployment-time performance. Success@K is governed solely by the total probability mass $Z _ { x }$ assigned to successful repair cores, whereas distinct-core coverage additionally rewards how broadly this mass is distributed across successful mechanisms. At fixed $Z _ { x }$ , equalizing probability across successful cores monotonically improves expected mechanism coverage, with uniform allocation achieving the optimum. Thus, successful-mass concentration and successful-mass allocation control different aspects of test-time scaling: increasing $Z _ { x }$ raises the probability of obtaining a valid repair, while distributing that mass across distinct successful cores increases the number of mechanisms recoverable under the same sampling budget.

This characterization motivates quotient-space exploration at the mechanism level. By shifting probability away from crowded regions of the repair-core distribution, the kernelized quotient objective promotes broader allocation across candidate mechanisms, while verifier-guided optimization concentrates probability on functionally successful regions. Their combination therefore aligns naturally with the two quantities identified above: successful probability mass and successful-core coverage. Empirically, QuotientPO improves both, yielding higher Success@K and greater distinct successful-core recovery under the same sampling budget.

## C DATASET AND BENCHMARK CONSTRUCTION

## C.1 CONSTRUCTION PIPELINE

Construction of initial GEM and proteome matching. BacDive phenotype records were linked to NCBI (Sayers et al., 2022) proteomes via prefix/suffix-normalized INSDC assembly accessions. Retaining matches with available protein.faa(.gz) files yielded 12,303 proteome inputs across 11,238 unique assemblies. We constructed candidate GEMs under four CarveMe settings: default, standard Gram-positive, standard Gram-negative, and a custom archaeal universe (augmenting CarveMe with archaeal biomass and ATP maintenance), producing 36,855 SBML Level 3/FBC2 GEMs as uncurated starting hypotheses. Concurrently, raw BacDive records were filtered for unambiguous binary phenotypes and mapped to BiGG identifiers. Retaining organisms with at least one mapped phenotype label produced 11,275 organisms and 37,864 organism–model entries (as multiple BacDive strains may share an assembly). This cohort was filtered and bifurcated into two downstream pipelines: 3,978 phenotype-consistent GEMs under the selected assays were identified as candidates for self-supervised pre-training, 28,842 canonical error-bearing GEMs formed the post-training cohort. Records sharing the same normalized accession were assigned to the same split to prevent genome-level leakage.

Table 4: Nutrient-growth assay coverage.
<table><tr><td>Assay category</td><td>BacDive families</td></tr><tr><td>Carbon source</td><td>glucose, maltose, sucrose, glycerol, acetate, lactate, pyruvate, succinate, citrate, glutamate</td></tr><tr><td>Nitrogen source</td><td>ammonium, urea, nitrate, nitrite</td></tr><tr><td>Sulfur source</td><td>sulfate</td></tr><tr><td>Phosphorus source</td><td>phosphate</td></tr></table>

Construction of BacDive targets. Binary BacDive nutrient-utilization phenotypes were mapped to BiGG metabolites across 16 nutrient families (Table 4), with stereospecific assays for lactate and glutamate yielding 18 total assay configurations. In each assay, the tested nutrient s serves as the sole elemental source for its macro-element class (C, N, S, or P) against an element-depleted minimal background to preclude metabolic bypass, with flux bounds set to 10 for the tested substrate, 20 for oxygen, and 1,000 for non-limiting element-free inorganics and water. To ensure consistent boundary conditions across draft GEMs, native exchanges were replaced by standardized boundary reactions (excluded from the editable action space), and all unsupplemented exchange uptakes were strictly disabled. Growth is predicted when optimal biomass flux exceeds $1 0 ^ { - 3 }$ . For multi-assay labels, positive predictions require growth in at least one assay, while negative predictions require absence across all. Requiring at least one decidable target and one initial phenotype error yields 21,615 post-training eligible GEMs across all partitions.

![](images/acdb188212c5577addb8fc9aba51e09457bcf3f14cdfc1b17acae49ecb821958.jpg)

![](images/3e781700af85d3f5b50f633703700397cd11e8e07cea00223107c3d914d12168.jpg)

<table><tr><td colspan="5">C. Core-16 target composition by native split</td><td rowspan="7">0.20 Shias i    wvidit 0.15 0.10 0.05 - 0.00</td></tr><tr><td>glucose</td><td>18.25%</td><td>18.21%</td><td>17.78%</td><td>20.39%</td></tr><tr><td>nitrate</td><td>13.50%</td><td>13.72%</td><td>13.96%</td><td>12.29%</td></tr><tr><td>maltose</td><td>11.50%</td><td>11.53%</td><td>10.79%</td><td>11.99%</td></tr><tr><td>sucrose</td><td>10.17%</td><td>10.31%</td><td>10.22%</td><td>14.11%</td></tr><tr><td>urea -</td><td>8.49%</td><td>8.46%</td><td>8.63%</td><td>9.99%</td></tr><tr><td>citrate -</td><td>8.64%</td><td>8.65%</td><td>8.57%</td><td>7.40%</td></tr><tr><td>nitrite -</td><td>7.72%</td><td>7.91%</td><td>7.83%</td><td>10.61%</td></tr><tr><td>glycerol -</td><td>6.56%</td><td>6.52%</td><td>6.22%</td><td>7.17%</td></tr><tr><td>glutamate</td><td>4.07%</td><td>4.10%</td><td>4.19%</td><td>2.32%</td></tr><tr><td>acetate</td><td>3.44%</td><td>3.28%</td><td>3.93%</td><td>0.99%</td></tr><tr><td>lactate</td><td>2.70%</td><td>2.73%</td><td>2.86%</td><td>1.08%</td></tr><tr><td>succinate</td><td>2.31%</td><td>2.03%</td><td>2.32%</td><td>0.50%</td></tr><tr><td>pyruvate</td><td>2.08%</td><td>1.93%</td><td>2.07%</td><td>0.69%</td></tr><tr><td>ammonium -</td><td>0.29%</td><td>0.32%</td><td>0.30%</td><td>0.26%</td></tr><tr><td>sulfate </td><td>0.28%</td><td>0.30%</td><td>0.33%</td><td>0.15%</td></tr><tr><td>phosphate -</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.06%</td></tr><tr><td></td><td>SFT</td><td>RL</td><td>Val</td><td>Test</td></tr></table>

![](images/49fd043bfbabd282a3fdae7c799388acb74a74a132d52facc2231243575e524f.jpg)  
Figure 9: Dataset characterization and split diagnostics.

Certified repair supervision. Teacher repair sets were generated offline via mixed-integer optimization (Appendix D.1) and verified by FBA under identical assay conditions. We distinguish complete repairs (correcting all phenotype errors) from partial repairs (strictly reducing net errors while preserving initially correct phenotypes). Partial repairs provide action-level supervision for SFT-A (masking actions that temporarily degrade phenotype accuracy), whereas complete repairs supervise set-level prediction in SFT-B. Training states are constructed from unordered subsets of verified repairs to eliminate arbitrary edit-order assumptions, retaining multiple valid solutions with per-GEM weight normalization. This supervision covers 1,426 GEMs (10,837 states) for SFT-A and 1,078 GEMs (40,151 rows) for SFT-B after augmentation. Crucially, phenotype labels, error diagnostics, and teacher edits are never exposed as policy inputs.

Data partitioning and leakage prevention. Dataset partitions were fixed before post-training eligi bility filtering. Entries connected by normalized organism identity, proteome identity, GEM provenance, or model checksum were merged by transitive closure into 8,568 indivisible identity components. A genus-held-out test set contains Bacillus, Clostridium, Lactobacillus, Mycobacterium, and Streptomyces. The remaining identity components were deterministically assigned to the SFT, RL, and validation pools: $2 8 , 8 4 \bar { 2 } = 1 2 , 2 2 3 _ { \mathrm { S F T } } + 1 2 , 2 2 6 _ { \mathrm { R L } } + 1 , 4 4 5 _ { \mathrm { v a l i d a t i o n } } + 2 , 9 4 8 _ { \mathrm { t e s t } }$ . After phenotype-assay filtering and certified-teacher selection, SFT-A retained 1,426/177 training/validation GEMs and SFT-B retained 1,078/117. The RL split retained 9,196/1,087 training/validation GEMs, while the final repair test set contained 2,212 GEMs. Eligible models in the SFT pool without certified supervision were left unused rather than reassigned to RL. All augmented states inherit the partition of their source GEM. Beyond the component-level partitioning of the post-training cohort, the 3,978 clean GEMs allocated to self-supervised pre-training were strictly quarantined from the outset. Pre-training explicitly excludes all five held-out evaluation genera and shares zero normalized INSDC accessions or proteomes with the test split. Auditing confirmed zero biological identity overlap across the entire lifecycle (Pre-training, SFT, RL, Validation and Test), preventing models from acquiring prior familiarity with test-split metabolic networks.

## C.2 DATASET CHARACTERIZATION AND QUALITY CONTROL

Figure 9 summarizes the biological breadth and split-level properties of the post-training collection. The linked dataset is taxonomically long-tailed, with 11,275 model-linked organisms spanning 2,454 genera and four GEM reconstruction variants (Fig. 9A,D). The core-16 supervision covers 153,869 model-level BacDive target groups across the 28,842 retained GEMs (Fig. 9B). These counts measure phenotype-target coverage rather than repair errors: the same organism contributes separately through different GEM variants, and distinct compound-level annotations remain independent targets. Across the native SFT, RL, and validation pools, major phenotype partitions are broadly consistent, while the sealed genus-held-out test set exhibits the expected distribution shifts induced by taxonomic holdout (Fig. 9C). Together, these diagnostics establish broad biological coverage, explicit bidirectional phenotype supervision, and a leakage-controlled evaluation shift.

## C.3 BENCHMARK VALIDATION

The sealed benchmark (2,212 GEMs) evaluates generalization across biological taxa, reconstruction templates, and phenotype error modes (Figure 10). Taxonomically, it spans five held-out genera: Streptomyces (926), Bacillus (443), Lactobacillus (403), Mycobacterium (266), and Clostridium (174) (Fig. 10A). It balances four draft reconstruction templates: 697 default, 517 archaeal, 507 Gram-negative, and 491 Gram-positive models (Fig. 10B). Across the 16 core metabolites, initial errors display dominant false negatives alongside non-trivial false positives (Fig. 10C), requiring policies to coordinate reaction additions and deletions rather than relying on monotonic gap-filling.

![](images/60d2126b044bb5bdf3740df27bb771a184fbf2704ffa78d80dfdabd5ad40c1c5.jpg)

![](images/417e819294dff7dceae3beea755894d005c95df11f5fdcc2c1aa35b502d9b7cb.jpg)

Figure 10: Benchmark validation.  
![](images/c3adabbd9894e41dc155c8f2059a264da77421ad3e66d65634e754be60a3e299.jpg)

## D BASELINE DETAILS

This section details the baselines that require bespoke algorithmic reformulation to operate over metabolic networks: the phenotype-informed MILP oracle (§D.1) and the object-level GFlowNet (§D.2), both adapted from external paradigms to phenotype-supervised GEM repair. The remaining baselines build directly upon our neural architecture and training pipeline: imitation models (SFT-Soft and SFT-Hard) are described alongside architectural details in Appendix E.3, while reinforcement learning policies (Vanilla GRPO, MaxEnt, and QuotientPO) are detailed in Appendix A and E.4.

## D.1 MILP

Mathematical formulation and phenotype-guided separation. We construct a phenotypeinformed mixed-integer linear programming (MILP) oracle that couples discrete reaction edits with continuous intracellular fluxes. For an initial GEM $M _ { 0 }$ with reaction set $\mathcal { R } _ { 0 }$ and a candidate reaction universe U, a binary decision vector $x \in \{ 0 , 1 \} ^ { | \mathcal { U } | }$ specifies toggled reactions, defining the repair edit set $E ( x ) = ( A \dot { ( } x ) , D ( x ) )$ ) with additions $\hat { A ( x ) } = \hat { \{ r \in \mathcal { U } \setminus \hat { \mathcal { R } } _ { 0 } : x _ { r } = 1 \} }$ and deletions $D ( x ) = \{ r \in \mathscr { R } _ { 0 } \cap \mathscr { U } : x _ { r } = 1 \}$ , such that $\| x \| _ { 1 } = | E ( \dot { x } ) |$ . For each nutrient assay s, biomass flux is evaluated via flux balance analysis (FBA), formulated as a continuous linear programming (LP) problem:

$$
v _ { s } ^ { * } ( x ) = \operatorname* { m a x } _ { v } \left\{ c ^ { \top } v : S v = 0 , \ell _ { s } ( x ) \leq v \leq u _ { s } ( x ) \right\} .
$$

Given target phenotype labels $y _ { i } \in \{ 0 , 1 \}$ and mapped assay sets $\mathcal { A } _ { i }$ , the predicted phenotype is $\widehat { y } _ { i } ( x ) = 1 [ \mathrm { m a x } _ { s \in { \mathcal A } _ { i } } v _ { s } ^ { * } ( x ) > 1 0 ^ { - 3 } ]$ , yielding the total phenotype error $\begin{array} { r } { \mathrm { E r r } ( x ) = \sum _ { i } { \bf 1 } [ \widehat { y } _ { i } ( x ) \neq } \end{array}$ $y _ { i } ]$ . The oracle accesses test-time labels to lexicographically minimize phenotype errors before edit cardinality: $\mathrm { m i n } _ { x } ( \mathrm { E r r } ( x ) , \| x \| _ { 1 } )$ ). We implement this hierarchy via binary violation indicators

z<sub>i</sub> ∈ {0, 1}:

$$
\operatorname* { m i n } _ { x , z } \quad \sum _ { r \in \mathcal { U } } x _ { r } + ( | \mathcal { U } | + 1 ) \sum _ { i } z _ { i } ,
$$

where $z _ { i } ~ = ~ 0$ requires $\widehat { y _ { i } } ( x ) = y _ { i }$ . Because $\| x \| _ { 1 } \leq | \mathcal { U } | .$ , the weight $( | \mathcal { U } | + 1 )$ guarantees that resolving any phenotype strictly dominates toggling candidate reactions. Rather than embedding continuous fluxes into a large big-M formulation, the discrete master optimizes $( x , z )$ and generates phenotype-guided cuts through FBA separation. At iteration $k ,$ a violated phenotype induces

$$
D _ { H } ( x , x ^ { ( k ) } ) + z _ { i } \geq 1 , \qquad D _ { H } ( x , x ^ { ( k ) } ) = \sum _ { r \in H } \left| x _ { r } - x _ { r } ^ { ( k ) } \right| ,
$$

where $H \subseteq U$ concentrates the separation on phenotype-relevant reactions identified from LP dual information for failed positive phenotypes or active flux support for failed negative phenotypes. These phenotype-guided restricted-support cuts concentrate the finite-budget search on functionally relevant edit configurations, while every retained candidate is independently certified by complete FBA evaluation under the original assay conditions.

Solver implementation and verification pipeline. Each GEM receives a 600-second search budget using double-precision HiGHS (Huangfu & Hall, 2015) with one thread per GEM, LP feasibility tolerance 1e−8, MIP integrality tolerance 1e−9, and zero requested MIP gap. The search allocates 120 seconds to greedy subset repair (up to 1,000 separation iterations) and the remainder to lexicographic search (up to 4,000 iterations). Each candidate is independently evaluated by the complete runtime FBA verifier, which determines its phenotype error under the original assay conditions. Across the 2,212 sealed test GEMs, we retain the FBA-verified candidate with the best lexicographic objective $( \operatorname { E r r } ( x ) , \| x \| _ { 1 } )$ encountered within the search budget, with unimproved cases retaining $M _ { 0 } ^ { \mathbf { \bar { \Phi } } }$ . This phenotype-informed oracle therefore provides a strong label-access baseline for evaluating amortized repair policies.

## D.2 GFLOWNET

We adapt GFlowNet (Bengio et al., 2021) to one-shot GEM repair as an object-level diversity baseline. It is to test whether distributing probability across high-reward raw repair sets, without identifying equivalent repairs, is sufficient to obtain the multi-solution benefits targeted by QuotientPO.

Object-level GFlowNet objective. Let $\pi _ { 0 }$ denote the frozen SFT policy used to initialize RL. For a complete repair set E, we define the reward-tilted target distribution

$$
p ^ { \star } ( E \mid M _ { 0 } , G , y ) \propto \pi _ { 0 } ( E \mid M _ { 0 } , G ) ^ { \gamma } \exp ( R _ { y } ( E ) / \tau ) ,\tag{31}
$$

where $\tau$ controls reward temperature and $\gamma$ controls the strength of the SFT base distribution. The target therefore favors high-reward repairs while retaining a structured prior over plausible edits. Crucially, its support is defined over raw edit sets: two repairs remain distinct GFlowNet objects whenever their edit sets differ, even if they canonicalize to the same repair core. This provides a direct object-space counterpart to the quotient-space exploration of QuotientPO.

One-shot adaptation. Unlike standard trajectory-based GFlowNets, our policy predicts the entire unordered repair set in a single forward pass via independent Bernoulli decoders, requiring no intermediate states, action orderings, or learned backward policies. Flow consistency under target (31) then requires policy probabilities to match target weights up to a partition constant. For a rollout group of $L$ candidates, defining the log flow residual $\bar { z } _ { i } = \bar { R } _ { y } ( E _ { i } ) \bar { / } \tau + \gamma \log \pi _ { 0 } ( E _ { i } ) - \log \pi _ { \theta } ( E _ { i } )$ and eliminating the unknown instance offset yields the log-partition variance (LPV) loss,

$$
\mathcal { L } _ { \mathrm { L P V } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } ( z _ { i } - \bar { z } ) ^ { 2 } ,
$$

which enforces reward-proportional probabilities without explicit partition-function estimation.

SFT anchor and task adaptation. A uniform base distribution over complete edit sets concentrates probability on unrealistically large cardinalities (averaging half of all available actions), causing unanchored exploration $( \gamma = 0 )$ to rapidly degenerate into thousands of spurious ADD edits (Table 5). The $\bar { \mathrm { S F T } }$ anchor in Eq. (31) resolves this combinatorial mismatch by tempering the target distribution with the structured prior $\pi _ { 0 } ^ { \gamma }$ while keeping policy initialization matched. Comparing completed candidates across $\gamma \in \{ 0 . 4 , 0 . 6 , 1 . 0 \}$ under identical training recipes, the intermediate anchor $\gamma = 0 . 6$ achieves the highest validation Success@8 (9.38%) while maintaining sparse, plausible repairs (10.98 ADD edits); we therefore fix $\gamma = 0 . 6$ before running replicated seeds without consulting test data. In contrast, an identity-uniform diagnostic that matched expected edit counts but removed reaction-specific SFT preferences collapsed toward nearly empty repairs (0.065 edits, 0.276% validation success), confirming that GFlowNet requires biological reaction-level preferences rather than mere cardinality constraints.

Table 5: SFT-anchor selection for the GFlowNet baseline. Anchored variants use the same final training recipe and complete all 864 updates. The unanchored run is incomplete and is included only to show object-space cardinality degeneration. Success values are validation Success@8 (%).
<table><tr><td>Anchor γ</td><td>Best</td><td>Best step</td><td>Last</td><td>Last ADD</td></tr><tr><td>0 (no anchor)</td><td>7.91</td><td>15</td><td>5.80</td><td>7367.57</td></tr><tr><td>0.4</td><td>8.46</td><td>745</td><td>7.64</td><td>23.05</td></tr><tr><td>0.6</td><td>9.38</td><td>864</td><td>9.38</td><td>10.98</td></tr><tr><td>1.0</td><td>8.00</td><td>715</td><td>7.27</td><td>6.05</td></tr></table>

Implementation details. To isolate object-level exploration under matched task optimization, we train the baseline using the joint objective $\mathcal { L } _ { \mathrm { G F l o w N e t } } = \mathcal { L } _ { \mathrm { G R P O } } + \lambda _ { \mathrm { L P V } } \mathcal { L } _ { \mathrm { L P V } }$ , combining standard verifier-guided GRPO on group-normalized advantages with LPV distribution matching on raw rewards. GFlowNet uses no quotient-space mechanisms during training, with repair cores computed strictly post hoc for evaluation. The baseline shares the exact policy backbone, candidate action masks, FBA verifier, and frozen SFT initialization $\mathbf { \Pi } ^ { \left( \pi _ { 0 } \right) }$ with all other RL methods. We use rollout size $L = 3 2$ , anchor strength $\gamma = 0 . 6$ , reward temperature $\tau = 1$ , LPV weight $\lambda _ { \mathrm { L P V } } = 0 . 0 2$ , and PPO clipping threshold 0.2. Each update batches 32 organisms with 32 fresh on-policy samples for a fixed budget of 864 updates (matching three epochs) via AdamW (learning rate $2 e { \mathrm { - } } 6 ,$ weight decay $1 e { \mathrm { - } } 4 ,$ global gradient clipping at 1.0). Across seeds 42, 43, and 44, test metrics are reported from the final checkpoint matching Table 1.

## E EXPERIMENTAL DETAILS

Our training follows a three-stage Pretrain–SFT–RL pipeline. Since GEM repair is evaluated only through non-differentiable terminal feedback, direct RL from random initialization is ineffective in the large and sparse reaction space. We therefore first pretrain the model to learn GEM structure and reaction compatibility, then use phenotype-supervised SFT to learn a repair prior, and finally optimize complete edit sets with verifier-grounded RL. These stages correspond to structural literacy, repair prior learning, and joint consequence optimization.

## E.1 BASE MODEL ARCHITECTURE.

All post-training methods use the same pretrained backbone (Fig. 11). Each GEM is represented as a metabolite–reaction bipartite graph with 19-dimensional node features and 3-dimensional stoichiometric edge features, and encoded by a 12-layer edge-aware GINE (Hu et al., 2020) with hidden dimension 768, residual connections, LayerNorm, and dropout. Reaction and metabolite nodes use type embeddings together with shared biochemical identity embeddings, while separate attention pooling over the two node types produces a global GEM representation. Candidate ADD reactions are encoded by a universal reaction encoder over 28,301 reactions, which combines a shared reaction-ID embedding, pooled substrate/product metabolite embeddings, and reaction-level structural features into the same 768-dimensional latent space. Organism-level proteomic context is extracted by a frozen $_ { \mathrm { E S M 2 - t 1 } 2 - 3 5 \mathrm { M } \ ^ { 2 } }$ encoder, whose 480-dimensional protein embeddings are deterministically reduced to 256 dimensions, attention-pooled over at most 512 proteins, and mapped to 768 dimensions by a learnable gated MLP before fusion with the global GEM representation. Based on this shared representation, separate ADD and REMOVE heads score absent and existing reactions, respectively: the ADD head combines graph–reaction matching, pairwise interactions, and local metabolite compatibility, whereas the REMOVE head scores interactions between the global GEM state and reaction-node embeddings. During post-training, the ESM-2 backbone remains frozen, while the GINE encoder, universal reaction encoder, proteome conditioner, and ADD/REMOVE scoring heads are jointly fine-tuned.

![](images/966781de3875f63ff16334164d0fa50764458a4a1977416b0d279a09a85a0116.jpg)  
Figure 11: Base model architecture.

## E.2 PRETRAINING DETAILS

Dataset construction. Of the 3,978 candidate GEMs, 143 were excluded by the pretraining leakagecontrol criteria, leaving 3,806 training GEMs and 29 held-out validation GEMs. We pretrain the GEM backbone through single-reaction denoising on these 3,806 clean GEMs. Each training example contains exactly one corruption: either a valid reaction is removed and recovered by the ADD head, or an absent reaction is inserted and identified by the REMOVE head. The two tasks are sampled with equal probability. To prevent synthetic shortcuts, injected reactions must be absent from the original GEM, avoid duplicate stoichiometric signatures, and connect metabolites already represented in the graph; artifact-prone metadata features are masked during pretraining. Each epoch contains 125,000 examples, comprising 100,000 dynamically generated corruptions and 25,000 fixed atomic examples, for a total of five million training exposures over 40 epochs. This objective trains reaction-level compatibility without imposing a synthetic prior over downstream repair-set cardinality or co-occurrence.

Model training and evaluation. Each pre-training example contains exactly one corrupted reaction, corresponding to either an ADD or REMOVE target. For action type a ∈ {ADD, REMOVE}, let $\mathcal { C } _ { i } ^ { a }$ denote the legal candidate set for example i and $y _ { i } ^ { a }$ its target reaction. ADD uses up to 4,096 sampled absent reactions, always including the positive target and frequency-based hard negatives, whereas REMOVE scores all reactions currently present in the corrupted GEM. Given logits $s _ { i r } ^ { a }$ the reaction-identification loss is $\begin{array} { r } { \mathcal { L } _ { \mathrm { C E } } ^ { a } = - \frac { 1 } { | \mathcal { B } _ { a } | } \sum _ { i \in \mathcal { B } _ { a } } \log \frac { \exp ( s _ { i y _ { i } ^ { a } } ^ { a } ) } { \sum _ { r \in \mathcal { C } _ { i } ^ { a } } \exp ( s _ { i r } ^ { a } ) } } \end{array}$ . We further emphasize difficult confounders using the eight highest-scoring incorrect candidates, H<sup>a</sup> = Top8 $\{ s _ { i r } ^ { a } : r \in$ $\mathcal { C } _ { i } ^ { a } \setminus \{ y _ { i } ^ { a } \} ]$ }, with pairwise ranking loss $\begin{array} { r } { \mathcal { L } _ { \mathrm { r a n k } } ^ { a } = \frac { 1 } { | \mathcal { B } _ { a } | } \sum _ { i \in \mathcal { B } _ { a } } \frac { 1 } { | \mathcal { H } _ { i } ^ { a } | } \sum _ { r \in \mathcal { H } _ { i } ^ { a } } } \end{array}$ softplus $\left( s _ { i r } ^ { a } - s _ { i y _ { i } ^ { a } } ^ { a } \right)$ The complete objective is

$$
\mathcal { L } _ { \mathrm { p r e t r a i n } } = \mathcal { L } _ { \mathrm { C E } } ^ { \mathrm { A D D } } + \mathcal { L } _ { \mathrm { C E } } ^ { \mathrm { R E M O V E } } + 0 . 2 \left( \mathcal { L } _ { \mathrm { r a n k } } ^ { \mathrm { A D D } } + \mathcal { L } _ { \mathrm { r a n k } } ^ { \mathrm { R E M O V E } } \right) .
$$

Models are trained with AdamW using a learning rate of 1e−4, weight decay 1e−5, global batch size 384 across eight GPUs<sup>3</sup>, BF16 mixed precision, and gradient clipping at 1.0. Validation uses 580 fixed corruptions generated from 29 held-out GEMs, including multi-edit corruptions, and scores the full reaction vocabulary. Checkpoint selection uses the macro-average of ADD and REMOVE recall on the operation-specific single-target subsets.

Scaling of model size and data size. We study scaling along both model capacity and cumulative pretraining exposure. Specifically, we compare 256(hidden size)×4(layer), 384×6, and 768×12 GINE backbones containing 11.16M, 18.98M, and 58.45M parameters, respectively, at 0.25M, 0.5M, 1M, 2.5M, and 5M training examples. Scaling is evaluated using oracle-count edit recall on 580 fixed validation examples generated from 29 held-out GEMs. Unlike the single-reaction corruptions used for training, validation examples may contain multiple target edits. For each example and operation, reactions are ranked by model score and the top $k _ { i }$ predictions are retained, where $k _ { i }$ equals the number of ground-truth targets for that operation. Recall is then micro-averaged over target reactions separately for ADD and REMOVE. This metric isolates reaction-ranking quality from threshold calibration and cardinality prediction. As shown in Figure 12, both additional data and model capacity improve recovery, with the largest gains appearing for ADD. At five million examples, the 768×12 model reaches 63.49% ADD recall and 80.73% REMOVE recall. We therefore use the $7 6 8 \times 1 2$ checkpoint as the shared pretrained backbone for all downstream experiments.

![](images/c54ed0931e24e8ae33fed73f00f9f7ff714531cd76fd44fb0280b398deb50d97.jpg)  
Figure 12: Scaling behavior of GEM pretraining.

## E.3 SUPERVISED FINE-TUNING DETAILS

Supervised fine-tuning bridges single-reaction pretraining and one-shot RL by establishing a directed biochemical repair prior. Because directly predicting complete repairs conflates reaction prioritization with combinatorial set sizing under underdetermined phenotypes, we divide SFT into two sequential stages. SFT-A learns local reaction rankings via multi-positive partial-label supervision, while SFT-B calibrates state-conditioned thresholds to decode complete edit sets in one shot, stably initializing downstream policy optimization.

## E.3.1 SFT-A: PARTIAL-LABEL REACTION-EDIT RANKING

Algorithm design. SFT-A bridges pretrained metabolic representations and downstream completeset decoding by learning which reaction edits are promising from partially edited states. Because phenotype supervision is partially identifying and admits multiple valid repair paths, treating a single teacher edit as unique penalizes alternative valid choices, while treating unlisted reactions as negatives overstates supervisory completeness. We therefore formulate $\bar { \mathrm { S F T - A } }$ as multi-positive partial-label ranking. Evaluated by FBA verification on partially edited states, candidate edits are categorized into strictly improving actions (immediate phenotype gain, forming set $P ^ { + } )$ , required neutral actions (zero immediate gain but necessary for multi-step completions, forming set $P ^ { 0 } )$ , and regressive actions (degrading agreement, masked from gradients). Let $C ^ { - }$ denote the regressive actions and define the effective candidate set ${ \cal \tilde { C } } = { \cal { C } } \backslash { \cal { C } } ^ { - } ;$ all losses below are evaluated only over ${ \widetilde { C } } .$ By construction, each SFT-A state contains at least one certified positive $( P \neq \emptyset )$ . When $\mathbf { \bar { \boldsymbol { P } } ^ { + } } = \boldsymbol { \mathcal { O } }$ the strict-tier term is omitted $( L _ { \mathrm { s t r i c t } } = 0 )$ , and when $H = \varnothing , L _ { \mathrm { t a i l } } = 0 .$ . Improving and neutral edits jointly form the certified positive set $P = P ^ { + } \cup P ^ { 0 }$ . For candidate logits z over legal candidates ${ \mathcal { C } } ,$ the complete training objective combines partial-label ranking, strict-tier enhancement, hard-tail margin separation against the top unlabeled candidates $\mathcal { H } \subseteq \mathcal { C } \setminus \overline { { P } } \left( \left| \mathcal { H } \right| \leq 3 2 \right)$ , and a logit regularizer:

$$
{ \mathcal { L } } _ { \mathrm { S F T - A } } = { \mathcal { L } } _ { \mathrm { u n i o n } } + 0 . 1 5 { \mathcal { L } } _ { \mathrm { s t r i c t } } + 0 . 1 5 { \mathcal { L } } _ { \mathrm { t a i l } } + 1 0 ^ { - 5 } { \mathcal { L } } _ { \mathrm { l o g i t } } ,
$$

$$
\begin{array} { r l } { \mathrm { w h e r e } } & { { } \mathcal { L } _ { \mathrm { u n i o n } } = \mathrm { L S E } _ { b \in \widetilde { C } } z _ { b } - \mathrm { L S E } _ { a \in P } z _ { a } , } \end{array}
$$

$$
{ \mathcal { L } } _ { \mathrm { s t r i c t } } = \mathrm { L S E } _ { b \in { \widetilde { C } } } z _ { b } - \mathrm { L S E } _ { a \in P ^ { + } } z _ { a } ,
$$

$$
\mathcal { L } _ { \mathrm { t a i l } } = \frac { 1 } { \vert \mathcal { H } \vert } \sum _ { b \in \mathcal { H } } \mathrm { s o f t p l u s } \big ( z _ { b } - \operatorname* { m a x } _ { a \in P } z _ { a } + 0 . 5 \big ) .
$$

Table 6: Direct SFT-A development results. Hit rates are operation-macro, model-equal retrieval scores on certified partial labels (seed 42).
<table><tr><td>Configuration</td><td>Hit@1↑</td><td>dev-select loss↓</td></tr><tr><td>Pretrained + ESM2</td><td>0.207</td><td>3.248</td></tr><tr><td>Random + ESM2</td><td>0.114</td><td>4.524</td></tr><tr><td>Pretrained, no ESM2</td><td>0.143</td><td>3.922</td></tr></table>

Here, $\begin{array} { r c l } { \mathrm { L S E } _ { i \in { \mathcal S } } z _ { i } } & { = } & { \log \sum _ { i \in { \mathcal S } } \exp ( z _ { i } ) } \end{array}$ denotes the LogSumExp operator, and $\begin{array} { r l } { \mathcal { L } _ { \mathrm { l o g i t } } } & { { } = } \end{array}$ $\frac { 1 } { 2 | \widetilde { C } | } \sum _ { b \in \widetilde { C } } z _ { b } ^ { 2 }$ regularizes finite logits to prevent probability saturation. This multi-positive objective equips downstream policy learning with a directional biochemical prior toward functionally viable reactions while strictly preserving alternative valid repair pathways.

Engineering implementation. The reaction-edit actor initializes from our pretraining checkpoint using a 12-layer, 768-dimensional GINE encoder over bipartite reaction–metabolite graphs, containing 59.3M trainable parameters out of 92.7M total. Organism context is supplied by frozen ESM2 representations (Fig.11): up to 512 uniformly sampled proteins per proteome are represented by fixed 256-dimensional truncations of the normalized 480-dimensional ESM2 embeddings, aggregated by trainable attention, and mapped into the 768-dimensional graph embedding via a zeroinitialized gated MLP residual, preserving pretrained reaction logits at initialization without learning explicit protein-to-reaction alignments. For scalable ADD training, each local batch constructs a shared candidate panel of nominal size 2,048 that retains all supervised batch positives and draws up to 75% of remaining slots from cached topology-based hard candidates (256 per graph) alongside vocabulary negatives; validation scores the complete ADD vocabulary without candidate sampling, while REMOVE scoring evaluates only present reaction nodes. Training uses 10,837 partial states weighted inversely by each organism’s state count $( w _ { s } = 1 / n _ { x } )$ to balance supervisory influence. Optimization spans 8 GPUs with local batch size 4 (effective global batch size 32) for up to five epochs using AdamW (learning rate $3 \times 1 0 ^ { - 5 }$ , weight decay $\overline { { 1 0 ^ { - 4 } } }$ , gradient clipping at 1.0) under BF16 autocast with FP32 loss reductions. We evaluate checkpoints on 410 development states and select checkpoints lexicographically by operation-macro, model-equal hit rate (Hit@1). SFT-A strictly outputs reaction candidate rankings, leaving decision thresholds, edit cardinality, and complete repair decoding to the downstream SFT-Soft stage.

Key results. Table 6 summarizes reaction-retrieval performance. The full Pretrained + ESM2 configuration achieves the strongest accuracy and likelihood fit, lifting operation-macro Hit@1 from 0.016 on pretrained initialization to 0.207 while attaining the lowest development loss (3.248). Removing pretraining degrades Hit@1 to 0.114 and inflates loss to 4.524, while removing ESM2 drops Hit@1 to 0.143 with loss 3.922. This advantage is concentrated in ADD retrieval, reaching 0.264 Hit@1 against 0.068 without pretraining and 0.125 without ESM2. Both metabolic pretraining and proteome context thus prove essential for prioritizing viable reactions and minimizing partial-label ranking loss.

## E.3.2 SFT-B: COMPLETE SET-LEVEL REACTION-EDIT SUPERVISION

Algorithm design. SFT-B converts the single action ranking actor learned in SFT-A into a one shot predictor of the complete remaining repair set. Suppressing state and operation indices, let C be the legal candidate set, $Y \subseteq { \mathcal { C } }$ the target edits from one certified completion, $N = \mathcal { C } \setminus Y$ the non target candidates. Unlike SFT-A, SFT-B treats each teacher completion as a complete binary target. Alternative valid completions remain separate training examples rather than being merged. Both variants share a class balanced binary cross entropy loss and a hard negative ranking loss. Let v denote the candidate score vector, and let $H _ { h } \subseteq N$ be the top h negatives. With softplus(u) = log(1 + e<sup>u</sup>),

$$
\mathcal { L } _ { \mathrm { B C E } } ( v ) = \frac { 1 } { 2 } \left[ \frac { 1 } { | Y | } \sum _ { a \in Y } \mathrm { s o f t p l u s } ( - v _ { a } ) + \frac { 1 } { | N | } \sum _ { b \in N } \mathrm { s o f t p l u s } ( v _ { b } ) \right] ,
$$

$$
{ \mathcal { L } } _ { \mathrm { r a n k } } ( v ) = { \frac { 1 } { | Y | | H _ { h } | } } \sum _ { a \in Y } \sum _ { b \in H _ { h } } { \mathrm { s o f t p l u s } } ( v _ { b } - v _ { a } ) .
$$

The balanced BCE prevents the large negative vocabulary from dominating the objective, while the ranking loss separates target edits from competitive negatives. When one class is empty, BCE uses only the available class. Likewise, boundary losses are evaluated only when both their positive and negative aggregates are nonempty; otherwise the corresponding boundary term is set to zero. Ranking is applied only when both classes are present. Hard-SFT evaluates these losses on raw logits z, while Soft-SFT evaluates them on threshold adjusted scores u.

Hard-SFT factorizes set prediction into reaction ranking and discrete cardinality prediction. The actor outputs reaction logits z and a categorical count distribution $q ( k )$ Decoding predicts ${ \widehat k } = \arg \operatorname* { m a x } _ { k } q ( k )$ and returns $\widehat { Y } \ = \ \mathrm { T o p K } ( z , \widehat { k } )$ The count head supports up to $K _ { \mathrm { m a x } }$ edits (13 for ADD, 1 for REMOVE), trained with cross-entropy $\mathcal { L } _ { \mathrm { c o u n t } } ~ = ~ - \log q ( m )$ and normalized expected error $\begin{array} { r } { \mathcal { L } _ { \mathrm { d i s t } } = | \sum _ { k } k q ( k ) - m | / K _ { \operatorname* { m a x } } } \end{array}$ . Hard-SFT also enforces separation around the decision boundary at temperature $t = 0 . 2 5$ via ${ \mathcal { L } } _ { \mathrm { b d r y } } = { \mathrm { : } }$ t softplus $( ( b ^ { - } - b ^ { + } + 1 ) / t )$ , where $\begin{array} { r } { b ^ { + } = - t \log ( | Y | ^ { - 1 } \sum _ { a \in Y } e ^ { - z _ { a } / t } ) } \end{array}$ and $\begin{array} { r } { b ^ { - } = t \log ( | H _ { 3 2 } | ^ { - 1 } \sum _ { b \in H _ { 3 2 } } e ^ { z _ { b } / t } ) } \end{array}$ smoothly aggregate target and top-32 negative logits. The complete objective is

$$
\mathcal { L } _ { \mathrm { H a r d } } = \mathcal { L } _ { \mathrm { B C E } } ( z ) + 0 . 4 0 \mathcal { L } _ { \mathrm { r a n k } } ( z ) + 0 . 2 0 \mathcal { L } _ { \mathrm { b d r y } } + 0 . 2 5 \mathcal { L } _ { \mathrm { c o u n t } } + 0 . 5 0 \mathcal { L } _ { \mathrm { d i s t } } .
$$

Hard-SFT therefore learns reaction identity and repair size through separate outputs. This gives explicit cardinality control, but count errors directly cause under or over editing and the finite support imposes a fixed upper bound on repair size.

Soft-SFT replaces the discrete count head with a state conditioned inclusion threshold. The actor outputs reaction logits $z _ { a } ,$ while an auxiliary head predicts an operation specific threshold $\tau .$ The adjusted score is $u _ { a } = z _ { a } - \tau$ , giving inclusion probability $p _ { a } = \sigma ( u _ { a } )$ and deterministic prediction $\widehat { Y } = \{ a \in \mathcal { C } : u _ { a } \geq 0 \}$ . Repair size therefore emerges from the candidate scores and the learned threshold rather than from a separate count prediction. Soft-SFT applies the shared BCE and ranking losses to $u ,$ with $H _ { 8 }$ used for ranking. Its expected cardinality is $\bar { k } = \textstyle \sum _ { a \in { \mathcal { C } } } p _ { a }$ . Using smooth $L _ { 1 }$ loss $\rho ,$ cardinality is supervised by $\mathcal { L } _ { \mathrm { l o g c o u n t } } = \rho ( \log ( 1 + \bar { k } ) - \log ( 1 + m ) )$ and $\mathcal { L } _ { \mathrm { r e l a t i v e } } =$ $\rho ( ( \bar { k } - m ) / ( m + 1 ) )$ . The former stabilizes supervision across different set sizes, while the latter remains sensitive to small targets. Soft-SFT also directly trains the zero inclusion boundary using $\begin{array} { r } { \mathcal { L } _ { \mathrm { b d r y } } = \frac { 1 } { 2 } [ | Y | ^ { - 1 } \sum _ { a \in Y } } \end{array}$ softplus $( 0 . 5 - u _ { a } ) + | H _ { 6 4 } | ^ { - 1 } \dot { \sum } _ { b \in H _ { 6 4 } }$ softplus(u<sub>b</sub> + 0.5)]. The complete objective is

$$
\mathcal { L } _ { \mathrm { S o f t } } = \mathcal { L } _ { \mathrm { B C E } } ( u ) + 0 . 2 0 \mathcal { L } _ { \mathrm { r a n k } } ( u ) + 0 . 1 5 \mathcal { L } _ { \mathrm { b d r y } } + 0 . 5 0 \mathcal { L } _ { \mathrm { l o g c o u n t } } + 0 . 0 2 \mathcal { L } _ { \mathrm { r e l a t i v e } } .
$$

By coupling reaction scoring and repair sizing within continuous scores u, Soft-SFT removes the discrete cardinality limit and directly produces the Bernoulli inclusion probabilities for downstream one-shot RL.

Engineering implementation. Both variants initialize their shared 12-layer, 768-dimensional GNN actor, ADD/REMOVE scoring heads, and frozen ESM2 conditioning from the identical selected SFT-A checkpoint across seeds 42–44, randomly initializing only the newly added count or threshold module. Training views are matched to their decoding assumptions across 1,078 shared GEMs: Hard-SFT trains on 5,874 early-subset states with ADD-loss reweighting, whereas Soft-SFT trains on 40,151 all-subset states. Hard-SFT restricts training to early-subset states $( | S | \le 2 )$ to prevent its categorical count head from developing a small-cardinality prior collapse induced by late-stage combinatorial subsets, aligning the training cardinality distribution with initial-state deployment. Conversely, Soft-SFT leverages all-subset states to train its continuous threshold across diverse residual repair scales. Operation-specific Soft thresholds are initialized by maximizing edit F1 on initial training examples and recalibrated per-epoch on a fixed training subset without consulting evaluation splits. Both methods score the full ADD vocabulary under per-GEM legality masks, restrict REMOVE scoring to present reactions, and evaluate or deploy directly from original unedited GEMs. Optimization runs on eight GPUs with effective batch size 128 using AdamW (weight decay 1e−4, gradient clipping at 1.0) with layer-wise learning rates of 5e−6 for the backbone, 2e−5 for actor heads, and 1e−4 for the count or threshold module.

Key results. For downstream evaluation, checkpoints across seeds are selected on the development split by joint edit F1, with final held-out test evaluations presented in Table 1. Crucially, a matchedbatch diagnostic before any RL update confirms that Soft-SFT provides a far more informative initialization for policy optimization, yielding nonzero within-group reward variance for 51.6% ± 4.9% of sampled organism groups compared with $3 2 . 0 \% \pm 5 . \mathsf { \bar { 1 \% } }$ for Hard-SFT. This verifies that the adaptive-threshold policy exposes a richer verifier signal to kick-start policy exploration.

Table 7: Shared RL configuration.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Actor initialization</td><td>Soft-SFT</td></tr><tr><td>Total / trainable parameters</td><td>92.85M / 59.46M</td></tr><tr><td>Training / validation GEMs</td><td>9,196 / 1,087</td></tr><tr><td>RL seeds</td><td>42, 43, 44</td></tr><tr><td>Training devices</td><td>8 GPU ranks 32</td></tr><tr><td>GEMs per update</td><td></td></tr><tr><td>Rollouts per GEM</td><td>32</td></tr><tr><td>Training budget</td><td>3 nominal epochs, 864 updates</td></tr><tr><td>Actor updates per rollout batch</td><td>1</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>2e-6</td></tr><tr><td>Weight decay</td><td>1e-4</td></tr><tr><td>Gradient norm clipping</td><td>1.0</td></tr><tr><td>GRPO clipping</td><td>€ = 0.2</td></tr><tr><td>Advantage stabilizer</td><td>1e-6</td></tr><tr><td>Bernoulli sampling temperature</td><td>1.0</td></tr><tr><td>FBA growth threshold</td><td> $1 e { - 3 }$ </td></tr><tr><td>Validation cadence</td><td>Every 5 updates and at update 864</td></tr></table>

## E.4 REINFORCEMENT LEARNING DETAILS

Shared RL training setup. Alongside the GFlowNet (Appendix D.2), Vanilla GRPO, MaxEnt, and QuotientPO (Appendix A) share a unified one-shot policy optimization framework with detailed configurations summarized in Table 7. Initialized from the selected Soft-SFT checkpoint, the policy samples a complete repair set in a single forward pass via independent Bernoulli decisions over legal edits using fine-tuned inclusion thresholds, dispensing with value critics, reference KL penalties, or discrete cardinality heads. Vanilla GRPO optimizes this verifier-guided objective directly, serving as the common backbone for exploration ablations.

Action-entropy baseline. MaxEnt augments GRPO with the exact Shannon entropy of the factorized Bernoulli policy, summed analytically in nats over all legal ADD and REMOVE decisions without vocabulary normalization, Monte Carlo estimation, or additional verifier overhead. Because summing unnormalized entropy across a large reaction vocabulary risks simultaneously elevating many individually unlikely actions and causing severe repair cardinality explosion, its coefficient requires explicit calibration. In development experiments, coefficients of $1 e { - 3 }$ and $5 e ^ { - 4 }$ triggered severe ADD cardinality inflation, whereas 1e−4 remained stable but yielded insufficient exploration. We therefore fixed the intermediate coefficient of 2e−4 across all seeds based strictly on validation behavior, ensuring that the hyperparameter choice was locked before test evaluation without consulting held-out splits.

Runtime overhead. We benchmark one complete optimizer update on a single node with 8× NVIDIA RTX PRO 6000 Blackwell GPUs, using the same RL training settings for all methods. All runs use the same HiGHS backend, 64 single-threaded FBA workers, FP32 execution, and no validation or external logging. QuotientPO requires 23.22 s per update versus 21.32 s for Vanilla GRPO, corresponding to an 8.9% overhead. Most additional cost comes from verifier-side processing, including canonicalization, QPI and kernel scoring.

Table 8: Per-update runtime under identical hardware and rollout settings.
<table><tr><td>Method</td><td>Update (s)</td><td>FBA drain (s)</td><td>Overhead</td></tr><tr><td>Vanilla GRPO</td><td>21.323</td><td>2.229</td><td></td></tr><tr><td>QuotientPO</td><td>23.218</td><td>4.186</td><td>+8.9%</td></tr></table>

Table 9: Performance across GEM reconstruction templates. Results are mean ± standard deviation across three seeds.
<table><tr><td>Template Method</td><td></td><td></td><td>Success@1 Success@8 Success@32</td><td>Greedy</td><td></td><td>Improved Success Cores@32</td></tr><tr><td colspan="7">Default (N = 697)</td></tr><tr><td></td><td></td><td>Vanilla GRPO 7.37±0.5812.72±0.7916.79±1.087.84±0.36 28.83±1.08</td><td></td><td></td><td></td><td> $0 . 5 1 6 { \pm } 0 . 0 0 3$ </td></tr><tr><td>MaxEnt</td><td></td><td></td><td></td><td></td><td>5.69±0.9611.81±1.58 15.93±1.28 7.32±1.52 28.04±2.33</td><td> $0 . 4 6 2 { \pm } 0 . 0 8 8$ </td></tr><tr><td>GFlowNet</td><td></td><td></td><td></td><td></td><td>1.77±0.22 6.07±0.54 10.38±0.33 2.54±0.3011.48±0.70</td><td> $0 . 2 6 5 { \scriptstyle \pm 0 . 0 1 7 }$ </td></tr><tr><td>QuotientPO</td><td></td><td></td><td></td><td></td><td>7.89±0.5014.25±0.2218.75±0.82 7.80±0.33 30.01±1.19</td><td> $\mathbf { 0 . 5 8 2 { \scriptstyle \pm 0 . 0 0 6 } }$ </td></tr><tr><td colspan="7">Gram-positive (N = 491)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Vanilla GRPO 7.13±0.2013.04±0.35 16.70±2.12 8.15±0.41 30.14±0.72</td><td> $0 . 5 7 3 { \scriptstyle \pm 0 . 0 1 9 }$ </td></tr><tr><td>MaxEnt</td><td></td><td></td><td></td><td></td><td>5.16±1.73 10.93±1.01 16.02±0.85 7.13±1.43 30.17±1.06</td><td> $0 . 4 8 3 { \pm } 0 . 0 8 7$ </td></tr><tr><td>GFlowNet</td><td></td><td></td><td></td><td></td><td>2.58±0.31 7.06±0.31 10.93±0.12 2.72±0.82 14.33±0.64</td><td> $0 . 2 9 7 { \scriptstyle \pm 0 . 0 1 7 }$ </td></tr><tr><td>QuotientPO</td><td></td><td></td><td></td><td></td><td>6.93±0.2014.39±0.6219.48±0.779.03±0.7731.44±1.08</td><td> $\mathbf { 0 . 6 0 4 } \pm \mathbf { 0 . 0 0 3 }$ </td></tr><tr><td colspan="7">Gram-negative (N = 507)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Vanilla GRPO 7.43±0.89 14.79±0.99 19.26±0.99 8.02±0.41 30.21±0.49</td><td>0.507±0.027</td></tr><tr><td>MaxEnt</td><td></td><td></td><td></td><td></td><td>6.11±0.5913.48±1.4017.82±2.436.77±0.97 29.50±1.88</td><td>0.550±0.027</td></tr><tr><td>GFlowNet</td><td></td><td></td><td></td><td></td><td>3.81±0.30 8.22±0.11 12.10±0.30 2.89±0.11 13.89±0.46</td><td>0.377±0.008</td></tr><tr><td>QuotientPO</td><td></td><td></td><td></td><td></td><td>8.15±0.30 15.98±0.20 20.97±1.018.55±0.23 32.08±0.56</td><td> $\mathbf { 0 . 6 1 7 \pm 0 . 0 3 2 }$ </td></tr><tr><td colspan="7">Archaeal (N = 517)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Vanilla GRPO 7.80±0.30 14.96±2.34 19.34±2.42 7.93±0.70 32.11±0.23</td><td> $0 . 4 0 9 { \scriptstyle \pm 0 . 0 5 7 }$ </td></tr><tr><td>MaxEnt</td><td></td><td></td><td></td><td></td><td>6.96±0.8914.77±1.24 19.08±1.45 7.93±1.01 30.17±2.76</td><td> $0 . 4 2 7 { \scriptstyle \pm 0 . 0 2 2 }$ </td></tr><tr><td>GFlowNet</td><td></td><td></td><td></td><td></td><td>5.22±0.1912.77±0.51 19.79±0.404.84±0.39 21.08±0.15</td><td> $0 . 4 3 3 { \pm } 0 . 0 1 7$ </td></tr><tr><td>QuotientPO</td><td></td><td></td><td></td><td></td><td>8.00±0.30 16.89±0.81 21.66±0.398.77±0.40 33.38±0.36</td><td> $\mathbf { 0 . 4 7 6 { \overset { . } { \bot } } 0 . 0 3 0 }$ </td></tr></table>

## E.5 EVALUATION DETAILS

All methods are evaluated on 2,212 held-out initial GEMs. For each GEM, we generate 32 stochastic repair sets in FP32 with dropout disabled under fixed per-organism seeds, alongside a separate greedy prediction. Canonical repair cores are uniformly recomputed for all baselines using the deterministic reduction described in Appendix A. Success@K is computed over nested prefixes of the 32 rollouts, greedy success is evaluated separately, and diversity metrics are calculated over these frozen outputs. All metrics are computed per GEM and macro-averaged, reporting the mean and sample standard deviation across three training seeds.

## F ADDITIONAL EXPERIMENTAL RESULTS

## F.1 ROBUSTNESS ACROSS RECONSTRUCTION TEMPLATES.

QuotientPO consistently improves Success@32 over Vanilla GRPO across all four CarveMe templates, with gains of +1.96, +2.78, +1.71, and +2.32 percentage points on default, Gram-positive, Gram-negative, and archaeal models, respectively (Table 9). Successful-core coverage also im proves in every template, indicating that the overall gain is not driven by any single reconstruction setting, including the archaeal subset.

## F.2 SOLUTION SPACE GEOMETRY: SFT VERSUS POLICY OPTIMIZATION

To analyze exploration geometry, each raw repair is represented by its signed reaction edits under pairwise Jaccard distance $d ( A , \dot { B } ) = 1 - | A \cap \mathbf { \hat { B } } | / | A \cup \mathbf { \hat { B } } | ( d ( \infty , \infty ) = 0 )$ , and 32 stochastic rollouts from SFT-Soft and QuotientPO are jointly embedded with the unedited GEM and a certified MILP teacher reference via 2D metric multidimensional scaling (MDS)<sup>4</sup>. Identical edits share averaged coordinates on aligned axes, with all reported teacher distances evaluated in the ambient Jaccard space. Across six representative organisms (Figure 13), SFT-Soft rollouts yield few successful completions and cluster tightly around the certified teacher reference. In contrast, QuotientPO substantially increases repair success and disperses into multiple disconnected clusters, moving beyond the supervised demonstration to recover valid repairs across separated regions of the edit space.

Table 10: Success-controlled mechanism coverage. Expected distinct canonical cores $( D _ { r } )$ evaluated across r successful draws on paired cohorts.
<table><tr><td>r</td><td>Paired GEMs (42/43/44)</td><td>Vanilla GRPO</td><td>QuotientPO</td></tr><tr><td>2</td><td>295/269/323</td><td> $1 . 4 2 5 \pm 0 . 0 1 9$ </td><td> $\mathbf { 1 . 4 6 1 \pm 0 . 0 2 0 }$ </td></tr><tr><td>4</td><td>247/228/258</td><td> $1 . 9 6 0 \pm 0 . 0 3 7$ </td><td> $\mathbf { 2 . 0 5 0 \pm 0 . 0 5 6 }$ </td></tr><tr><td>8</td><td>209/186/206</td><td> $2 . 6 5 9 \pm 0 . 0 7 6$ </td><td> $\mathbf { 2 . 7 8 2 \pm 0 . 1 1 4 }$ </td></tr><tr><td>16</td><td>147/123/130</td><td> $3 . 2 5 6 \pm 0 . 0 3 3$ </td><td> $\mathbf { 3 . 5 0 2 \pm 0 . 2 0 3 }$ </td></tr></table>

## F.3 EXTENDED SOLUTION-SPACE VISUALIZATIONS ACROSS RL BASELINES

Figure 14 compares exploration behavior across Vanilla GRPO, MaxEnt, GFlowNet, and Quo tientPO projected into both raw edit and quotient mechanism spaces over six representative GEMs. Standard RL baselines and action-entropy exploration suffer from severe mode collapse, frequently rediscovering identical redundant mechanisms despite surface-level action variations. In contrast, QuotientPO consistently disperses across the quotient landscape, discovering substantially more distinct, successful canonical repair cores across diverse biological targets.

## F.4 SUCCESS-CONTROLLED MECHANISM COVERAGE

Successful-core yield conflates solve rates with conditional mechanism diversity. To isolate the latter, we measure the expected distinct canonical cores under an equalized budget of r successful repairs, restricted to paired cohorts where both methods achieve at least r successes among 32 rollouts. For M successful repairs with core a occurring $s _ { a }$ times, the expected distinct core count across r draws without replacement is

$$
D _ { r } = \sum _ { a } \left[ 1 - \frac { \binom { M - s _ { a } } { r } } { \binom { M } { r } } \right] .
$$

Table 10 shows that QuotientPO consistently outperforms Vanilla GRPO across all budgets, with the margin widening from +0.035 at r = 2 to +0.246 at $r = 1 6$ . This confirms that the diversity gain does not merely reflect a higher solve rate. We truncate at $r = 1 6$ because requiring all 32 rollouts to succeed leaves only 10/8/4 paired GEMs across seeds 42/43/44.

## F.5 FINITE-SAMPLE CREDIT RESOLUTION ANALYSIS

Metric definitions. We compute the training diagnostics from each rollout group of $N = 3 2$ repairs. Let $n _ { a }$ denote the multiplicity of canonical core a in the group. Under exact Renyi credit, two´ distinct cores receive identical credit if and only if they have the same multiplicity, so we identify ties directly by $n _ { a } = n _ { b }$ rather than by floating-point comparison. Weighting each distinct-core pair by the number of corresponding sampled pairs, ${ w _ { a b } } = n _ { a } n _ { b }$ , the exact-credit tie rate is

$$
{ \mathrm { T i e R a t e } } = { \frac { \sum _ { a < b } w _ { a b } \mathbb { I } [ n _ { a } = n _ { b } ] } { \sum _ { a < b } w _ { a b } } } .
$$

Importantly, TieRate is an exact-count diagnostic for every method: it measures how often distinct cores receive identical multiplicity-based credit, $n _ { a } = n _ { b }$ , within rollout groups generated by that policy. Thus, the Kernel Renyi-2 row does not measure ties under the kernel credit itself. Instead,´ it quantifies how often exact counting would still fail to distinguish cores sampled from the kerneltrained policy. The complementary ResolvedRate measures the fraction of these exact-count ties that are separated by the graded kernel credit.

To measure how much structure is hidden within these ties, we recompute a kernel credit from the same full rollout group. With signed-Jaccard similarity $K _ { a b } = \exp [ - ( \bar { 1 } - J ( a , b ) ) / 0 . 2 5 ]$ , the local kernel density of core a is $\begin{array} { r } { \rho _ { a } = \frac { \mathsf { \bar { \rho } } ( n _ { a } - 1 ) + \sum _ { b \neq a } n _ { b } K _ { a b } } { N - 1 } , \dot { P } _ { K } = \frac { 1 } { N } \sum _ { a } n _ { a } \rho _ { a } \rho _ { a } } \end{array}$ , giving the normalized kernel cost $c _ { a } ^ { K } = 2 ( \rho _ { a } / P _ { K } - 1 )$ . Following the training scale, we compare $\widetilde { c } _ { a } ^ { K } = 0 . 0 2 \ \mathrm { c l i p } ( c _ { a } ^ { K } , - 8 , 8 )$ and define the kernel-resolved rate as the fraction of exact-credit ties satisfying $\displaystyle { | \widetilde { c } _ { a } ^ { K } - \bar { \widetilde { c } } _ { b } ^ { K } | } > 0 . 0 0 1 $

$$
\mathrm { R e s o l v e d R a t e } = \frac { \sum _ { a < b } w _ { a b } \mathbb { I } [ n _ { a } = n _ { b } ] \mathbb { I } [ | \widetilde { c } _ { a } ^ { K } - \widetilde { c } _ { b } ^ { K } | > 0 . 0 0 1 ] } { \sum _ { a < b } w _ { a b } \mathbb { I } [ n _ { a } = n _ { b } ] } .
$$

Thus, TieRate measures the coarseness of exact finite-sample credit, whereas ResolvedRate is a conditional diagnostic of graded structure hidden inside exact ties and is not a higher-is-better performance metric. The successful-core statistics in Table 2 are computed separately from successful stochastic repairs only. For a GEM with $M \geq 2$ successful draws, let $s _ { a }$ be the number assigned to successful core a. Exact collision and mean pairwise Jaccard distance are

$$
\mathrm { C o l l i s i o n } = \frac { \sum _ { a } s _ { a } ( s _ { a } - 1 ) } { M ( M - 1 ) } , \qquad { \overline { { d } } } _ { J } = \frac { \sum _ { a < b } 2 s _ { a } s _ { b } [ 1 - J ( a , b ) ] } { M ( M - 1 ) } .
$$

Repeated draws retain their empirical multiplicity, so duplicate cores increase collision and contribute zero Jaccard distance rather than being deduplicated. All quantities are first computed per GEM and then macro-averaged over GEMs for which the corresponding statistic is defined. Undefined cases are excluded rather than imputed as zero.

Full training trajectory. Table 11 traces credit resolution across updates 0 to 864. As training proceeds, Exact Renyi-2 becomes increasingly coarse under finite rollout groups, with its exact-count ´ tie rate rising from 5.67% to 10.23%. Kernel Renyi-2 instead reshapes the rollout distribution to-´ ward finer exact-count structure, ending at 7.99%. More importantly, exact multiplicity substantially understates the available geometric signal: at step 864, kernel credit resolves 78.20% of exact-count ties in Exact Renyi-2 and 84.86% in Vanilla GRPO. These results demonstrate that kernel relaxation´ promotes a less degenerate rollout geometry and extracts graded credit among otherwise indistinguishable cores, translating directly into superior downstream performance (Table 12).

Table 11: Full training trajectory of finite-sample credit resolution. Exact-count tie rates measure distinct-core pairs indistinguishable under multiplicity credit within rollout groups; kernelresolved rates report the fraction of these ties separated by graded kernel credit.
<table><tr><td rowspan="2">Method</td><td colspan="4">Exact-credit ties (%)</td><td colspan="4">Kernel-resolved ties (%)</td></tr><tr><td>0</td><td>200</td><td>400</td><td>864</td><td>0</td><td>200</td><td>400</td><td>864</td></tr><tr><td>Vanilla GRPO</td><td>5.67</td><td>8.19</td><td>8.31</td><td>8.59</td><td>30.61</td><td>60.55</td><td>57.50</td><td>84.86</td></tr><tr><td>Discrete Shannon</td><td>5.67</td><td>8.60</td><td>9.99</td><td>9.72</td><td>30.61</td><td>59.23</td><td>59.77</td><td>80.20</td></tr><tr><td>Exact Rényi-2</td><td>5.67</td><td>7.34</td><td>9.06</td><td>10.23</td><td>30.61</td><td>59.71</td><td>63.71</td><td>78.20</td></tr><tr><td>Kernel Rényi-2</td><td>5.67</td><td>8.57</td><td>8.60</td><td>7.99</td><td>30.61</td><td>63.81</td><td>57.23</td><td>72.78</td></tr></table>

Table 12: Downstream performance of quotient exploration objectives.
<table><tr><td>Method</td><td>Success@32 ↑</td><td>Success Cores@32 ↑</td></tr><tr><td>Vanilla GRPO</td><td> $1 7 . 9 3 \pm 1 . 5 4$ </td><td> $0 . 5 0 \pm 0 . 0 1$ </td></tr><tr><td>Discrete Shannon</td><td> $1 8 . 6 1 \pm 1 . 0 9$ </td><td> $0 . 5 2 \pm 0 . 0 1$ </td></tr><tr><td>Exact Rényi-2</td><td> $1 8 . 0 4 \pm 1 . 0 8$ </td><td> $0 . 5 4 \pm 0 . 0 3$ </td></tr><tr><td>Kernel Rényi-2</td><td> ${ \bf 2 0 . 1 0 \pm 0 . 5 2 }$ </td><td> $\mathbf { 0 . 5 7 \pm 0 . 0 2 }$ </td></tr></table>

## F.6 MULTI-SEED ROBUSTNESS AND ABLATION RESULTS

Table 13 provides the complete evaluations corresponding to Table 3, confirming consistent improvements for QuotientPO.

Table 13: Multi-seed robustness and ablation results. Success@32 (%) are reported as mean ± standard deviation over three seeds.  
(a) RLOO Base
<table><tr><td>Method</td><td>Success@32</td></tr><tr><td>Vanilla</td><td> $1 5 . 7 5 \pm 1 . 2 0$ </td></tr><tr><td>MaxEnt</td><td> $1 4 . 2 9 \pm 0 . 9 5$ </td></tr><tr><td>GFlowNet</td><td> $1 1 . 5 7 \pm 0 . 1 4$ </td></tr><tr><td>QuotientPO</td><td> ${ \bf 1 7 . 2 1 \mu _ { \pm 0 . 8 1 } }$ </td></tr></table>

(c) Design Ablation

(b) Policy Architecture
<table><tr><td>Architecture RL</td><td>Success@32</td></tr><tr><td>Vanilla Soft</td><td> $1 7 . 9 3 \pm 1 . 5 4$  QuotientPO  $2 0 . 1 0 \pm 0 . 5 2$ </td></tr><tr><td>Hard</td><td>Vanilla  $5 . 9 2 \pm 0 . 0 9$  QuotientPO  $6 . 6 5 \pm 0 . 2 3$ </td></tr></table>

<table><tr><td>Variant</td><td>Success@32</td></tr><tr><td>Default</td><td> $2 0 . 1 0 \pm 0 . 5 2$ </td></tr><tr><td>Raw-space kernel</td><td> $1 8 . 1 6 \pm 0 . 6 1$ </td></tr><tr><td>w/o Quotient term</td><td> $1 7 . 5 4 \pm 1 . 8 1$ </td></tr><tr><td>w/o QPI term</td><td> $1 9 . 6 2 \pm 0 . 7 2$ </td></tr><tr><td>w/o Redundancy penalty</td><td> $1 9 . 9 4 \pm 1 . 7 9$ </td></tr></table>

(d) Hyperparameter Sensitivity
<table><tr><td>Param. Setting</td><td>Success@32</td></tr><tr><td>Default  $h = 0 . 2 5$   $\beta = 0 . 0 2$  ，</td><td> $2 0 . 1 0 \pm 0 . 5 2$ </td></tr><tr><td>0.125 h 0.5</td><td> $1 9 . 4 1 \pm 0 . 7 0$  18.57 ±0.74</td></tr><tr><td>0.01</td><td></td></tr><tr><td>β 0.04</td><td> $1 9 . 0 5 \pm 1 . 1 2$   $2 0 . 0 0 \pm 0 . 7 4$ </td></tr></table>

## G BIOLOGICAL ANALYSIS AND CASE STUDIES

Six additional cases complement the main-text sucrose and nitrate examples, spanning two cases across three progressive coverage tiers. Each comparison evaluates the identical GEM, checkpoint, and seed over $\bar { N = 3 2 }$ stochastic rollouts per method (V: Vanilla GRPO; Q: QuotientPO). For each focal alternative (the second pathway in each case), we track its sample-level representation along the three-stage evaluation pipeline: raw proposals containing the focal module $( R ) ,$ verified successful raw repairs containing the focal module (S), and successful canonicalized repairs in which the focal module remains present (C), satisfying $R \geq S \geq C .$ Importantly, R, S, and C count sample occurrences rather than distinct canonical cores; the total number of distinct successful canoni cal cores is reported separately as $K _ { \mathrm { c o r e } } .$ . The three coverage tiers characterize the stage at which Vanilla fails to retain the focal mechanism while QuotientPO does $( C _ { Q } > 0 )$ : Tier 1 (retention gap: $S _ { V } > 0 , C _ { V } = 0 )$ captures focal alternatives that already occur in successful Vanilla proposals but are eliminated during canonicalization; Tier 2 (viability gap: $R _ { V } > 0 , S _ { V } = 0 )$ captures focal modules proposed by Vanilla only within unsuccessful repairs; and Tier 3 (discovery gap: $R _ { V } = 0 )$ captures focal modules absent from Vanilla raw proposals.

## H LIMITATIONS AND FUTURE WORK

Limitations. Our study adopts a controlled setting to isolate phenotype-supervised GEM repair. All GEMs are reconstructed with CarveMe in the BiGG reaction system, which ensures a consistent reaction vocabulary and modeling convention across organisms but does not yet test transfer across reconstruction pipelines. We also use standardized category-specific media as reproducible proxies for BacDive growth conditions. This abstraction is necessary because many BacDive assays provide heterogeneous or incomplete culture descriptions, but it does not capture all strain-specific environmental details. Finally, our verifier focuses on steady-state metabolic feasibility and does not model regulatory or kinetic effects.

Future work. Future work can extend the framework to GEMs reconstructed by other pipelines, richer or uncertainty-aware culture-condition models, and additional biological constraints beyond FBA. More broadly, combining quotient-space repair with targeted experimental validation could connect phenotype-consistent computational repairs to testable metabolic hypotheses. Finally, quotient-space exploration may extend well beyond metabolic modeling to broader scientific domains such as molecular design, reaction mechanism discovery, and symbolic physical modeling, where macroscopic observations underdetermine underlying structures and search will benefit from operating directly over functional equivalence classes.

Success-to-MILP mean Jaccard distance: SFT-Soft 0.65, QuotientPO 0.98  
Success-to-MILP mean Jaccard distance: SFT-Soft 0.70, QuotientPO 1.0  
Success-to-MILP mean Jaccard distance: SFT-Soft 0.76, QuotientPO 0.97  
![](images/6d0ba43976a5cf4f80bf46c9131112cc30d4bb2b4e507a42cc52b45c2cea4c29.jpg)  
Success-to-MILP mean Jaccard distance: SFT-Soft 0.75, QuotientPO 1.00

![](images/9493039f89356b30704af6feea62cc6193549614690c97123a45d7aa98d76310.jpg)  
Success-to-MILP mean Jaccard distance: SFT-Soft 0.17, QuotientPO 0.95

![](images/718538408652dd2271abcfa01e5c2468b8e60454d1c85cf54be3bdb23dd018e9.jpg)

![](images/1f57407ec6e5c56bbebd28297c693b5dddcd7f23873a2e5815f2104967e48681.jpg)

![](images/999131e335bd89c9ffc6d23039e9edc0cf645b747a196f386ec18a88eefe9435.jpg)

![](images/30428fa879b07cf2164fddef3fd6d978c73e668a17dfbf91e2acd739873a9724.jpg)  
Other samples Improved, not successful Fully successful MILP teacher  
Figure 13: Cases of SFT versus RL Optimization.

Other samples Improved, not successful Fully successful

![](images/a9c9c6fe6f9f772936ea8d1aeb31a7998adad6f042b97a82b74f0359950cd4d0.jpg)  
Figure 14: Cases of solution-space visualization across RL baselines.

Table 14: Tier 1: retained-solution coverage. The focal alternative appears in successful Vanilla raw proposals $( S _ { V } ~ > ~ 0 )$ but is retained only by QuotientPO after canonical reduction $( C _ { V } = 0 ,$ $C _ { Q } > 0 )$  
Repair alternatives & encoded chemistry Functional verification & policy diagnosis   
Case 1 (C1): Mycobacterium terrae   
Target: Urea utilization   
Focal pipeline $( R  S  C ) \colon V \colon 1 3  2  0$ vs $Q \colon 1 3 \to 2 \to 1$   
Yield $\bar { ( N _ { \mathrm { s u c c } } \mid K _ { \mathrm { c o r e } } ) } \colon V \colon 4 \mid 1 \qquad Q \colon 4 \mid 2$   
Shared deletions (3): Phenotypic verification   
$- \mathtt { H E X 1 } , - \mathtt { M A L T a b c } , - \mathtt { N A R K }$ Both cores reduce urea-supported growth from 0.202   
Pathway 1: Uptake blockade to zero while preserving nitrate-supported growth of   
-UREAt 0.545 and all remaining targets. Reintroducing the   
focal deletion restores the failed urea-growth   
urea −→ urea phenotype, confirming its functional necessity.   
Pathway 2: Carboxylation blockade (focal)   
Search behavior diagnosis   
-UREASE   
Vanilla produces successful raw proposals containing   
urea $+ \mathrm { A T P } + \mathrm { H C O _ { 3 } ^ { - } }$ this intracellular branch $( S _ { V } = \hat { 2 } )$ , but completely   
−→ allophanate eliminates it during canonical reduction $( { \dot { C _ { V } } } = { \dot { 0 } } ) .$   
In contrast, QuotientPO successfully retains both   
$+ \mathrm { A D P + P _ { i } + H ^ { + } }$ functional intervention sites $( C _ { Q } = 1 )$   
Note: UREASE encodes urea carboxylase in the   
GEM. Both canonical cores contain exactly four   
deletions.   
Case 2 (C2): Streptomyces inusitatus   
Target: Alternative proline supply   
Focal pipeline $( R \hat {  } S  C ) \hat { : } \hat { V } \colon 3 2  3 2  0$ vs $Q \colon 3 2 \to 3 0 \to 2$   
Yield $\mathrm { \bf { \bar { ( } } } \hat { N } _ { \mathrm { s u c c } } \ | \ \cal K _ { \mathrm { c o r e } } ) \colon V \colon 3 2 \ | \ 2 \ \qquad Q \colon 3 0 \ | \ 2$   
Shared deletion (1): -UREAt Phenotypic verification   
Pathway 1: Direct conversion Both two-edit cores restore proline supply and   
+ORNCD recover citrate-supported growth from zero to 0.831   
while preserving the inactive urea target. Flux   
ornithine −→ proline $+ \mathrm { N H _ { 4 } ^ { + } }$ witnesses independently confirm active throughput   
Pathway 2: Semialdehyde/P5C route (focal) along both pathways.   
+G5SADs   
Search behavior diagnosis   
$\mathrm { G S A } \longrightarrow \mathrm { P 5 C } + \mathrm { H _ { 2 } O } + \mathrm { H ^ { + } }$ Vanilla already contains the focal G5SADs route in   
$\xrightarrow { \mathrm { e x i s t i n g ~ P 5 C R / P 5 C R x } }$ successful proposals but loses it after   
P5C proline canonicalization $( C _ { V } = 0 ) ;$ QuotientPO retains it   
Note: GSA denotes glutamate-5-semialdehyde; P5C $( C _ { Q } = 2 )$ . Thus, this is a retention rather than   
denotes $\Delta ^ { 1 } .$ -pyrroline-5-carboxylate. Both canonical discovery gap. ORNCD and ORNCD 2 are treated   
cores contain two edits. as one functional pathway family.

Table 15: Tier 2: successful-combination coverage. Vanilla proposes the focal module $( R _ { V } > 0 )$ but does not place it in a successful complete repair $( S _ { V } = 0 )$ , whereas QuotientPO does.  
Repair alternatives & encoded chemistry Functional verification & policy diagnosis   
Case 3 (C3): Lactobacillusfarraginis   
Target: Alternative phosphate donors for UTP synthesis   
Focal pipeline $( R \hat {  } S \hat {  } C ) \colon V \colon 4  0  0$ vs $Q \colon 1 0 \to 1 \to 1$   
Yield $\bar { ( \boldsymbol { N _ { \mathrm { s u c c } } } }$ | K<sub>core</sub>): $V \colon 0 \mid 0$ $Q \colon 2 \mid 2$   
Shared edits (3): Phenotypic verification   
+MOHMT, +NTRI $\mathtt { R 2 y , - N T R I R 3 p p }$ The alternatives provide UDP-to-UTP   
Pathway 1: ATP donor phosphorylation using different phosphate donors.   
+NDPK2 Both QuotientPO cores satisfy all six targets, with   
$\mathrm { A T P + U D P \longrightarrow A D P + U T P }$ glucose growth 0.540171 and 0.539515,   
respectively, and nitrate growth 0.888279.   
Pathway 2: PEP donor (focal)   
Flux witnesses carry approximately 0.16655 through   
+PYK2   
NDPK2 or 0.16635 through PYK2.   
$\mathrm { P E P + U D P + H ^ { + } }$   
$\longrightarrow \mathrm { p y r u v a t e } + \mathrm { U T P }$ Search behavior diagnosis Vanilla proposes PYK2 four times, but these   
Both complete cores contain four edits. proposals lack the coordinated repair required for   
success. Adding NTRIR2y restores the   
nitrate-positive target.   
Case 4 (C4): Streptomycesferralitis   
Target: Alternative nitrogen donors for asparagine   
Focal pipeline $( R \to S { \stackrel { \smile } { \to } } C ) \colon$ V: 24 → 0 → 0 vs $Q \colon 2 9 \to 3 \to 1$   
Yield $\bar { ( N _ { \mathrm { s u c c } } }$ | K<sub>core</sub>): V: 0 | 0 Q: 3 | 3   
Shared edits (3): Phenotypic verification   
$+ \mathtt { A R G D C } , - \mathtt { F F S D } , - \mathtt { S U C R t e x }$ The two reactions complete asparagine synthesis   
Pathway 1: Glutamine donor using different nitrogen donors.   
+ASNS1 Both QuotientPO cores satisfy the glucose-positive   
Asp $+ \mathrm { G l n + A T P + H _ { 2 } O }$ and sucrose-negative targets, with glucose growth   
1.342551 and 1.345126.   
−→ Asn + Glu   
$+ \mathrm { A M P + P P _ { i } + H ^ { + } }$ Search behavior diagnosis   
Vanilla frequently proposes the focal ASNS2 branch,   
Pathway 2: Ammonium donor (focal) but the corresponding repair sets remain   
+ASNS2 unsuccessful. One such proposal also introduces   
$\mathrm { A s p + N H _ { 4 } ^ { + } + A T P }$ extracellular sucrose hydrolysis through SUCRe;   
removing this additional bypass restores the   
−→ Asn + AMP   
sucrose-negative target.   
$+ \mathrm { P P _ { i } + H ^ { + } }$ Thus, successful use of the focal biosynthetic branch   
depends on its surrounding repair combination

Table 16: Tier 3: raw-module coverage. The complete focal module is absent from Vanilla raw proposals (R<sub>V</sub> = 0) but appears in successful QuotientPO repairs.  
Repair alternatives & encoded chemistry Functional verification & policy diagnosis   
Case 5 (C5): Streptomyces olivochromogenes   
Target: Alternative control of citrate-supported growth   
Focal pipeline (R → S → C): V: 0 → 0 → 0 vs Q: 2 → 2 → 2   
Yield (N<sub>succ</sub> | K<sub>core</sub>): V: 2 | 1 Q: 8 | 3   
Shared edit (1): Phenotypic verification   
-UREAt Both cores eliminate the initial urea and citrate false   
Alternative 1: Entry restriction positives while preserving glucose- and   
-CITt14, -FE3t sucrose-supported growth. The entry-restriction core   
CITt14 directly supports citrate uptake, while FE3t retains glucose/sucrose growth at   
0.630548/1.261097, whereas the SUCOAS core   
provides a reversible iron route coupled to the   
remaining FEDCabc citrate-entry pathway. retains 0.436023/0.872046.   
Alternative 2: Intracellular node (focal) Restoring either CITt14 or FE3t breaks only the   
-SUCOAS citrate-negative target in the first core. Restoring   
SUCOAS does the same in the focal core. Restoring   
succinyl-CoA+ADP+P<sub>i</sub> −→ succinate+CoA+ATP.<sub>UREAt affects only the urea-negative target.</sub>   
Search behavior diagnosis   
Vanilla never proposes -SUCOAS in this rollout   
panel. QuotientPO discovers it twice. Both samples   
are successful and canonicalize to the same focal   
core. Thus, this comparison exposes an intracellular   
repair alternative absent from Vanilla raw proposals,   
rather than two distinct mechanisms.   
Case 6 (C6): Streptomyces durbertensis   
Target: Alternative quinone-precursor supply   
Focal pipeline (R → S → C): V: 0 → 0 → 0 vs Q: 5 → 1 → 1   
Yield (N<sub>succ</sub> | K<sub>core</sub>): V: 1 | 1 Q: 7 | 5   
Shared edits (5): Phenotypic verification   
+HSERTA, +HSST, +THZPSN2 SC, -NO3abc, Both branches converge on OSB-CoA and   
-NO3tex downstream menaquinone-precursor supply.   
Pathway 1: Aromatic branch The resulting cores satisfy all four targets and yield   
+PHPYROX glucose growth of approximately 0.454946.   
phenylpyruvate −→ benzaldehyde Minimum-total-flux witnesses use the corresponding   
−→ benzoate −→ benzoyl-CoA precursor branch at approximately 4.55 × 10<sup>−5</sup>,   
matching the quinone demand induced by biomass   
reverse BSCT OSB-CoA production.   
Pathway 2: Three-ADD branch (focal) Search behavior diagnosis   
+SHCHCS, +SUCBZS, +SUCBZL 1 The complete three-ADD module does not occur in   
isochorismate + SSA Vanilla raw proposals, whereas QuotientPO proposes   
it five times and retains one successful canonical   
−→ SHCHC −→ OSB   
core. Its five total distinct cores therefore correspond   
−→ OSB-CoA   
to multiple edit sets, rather than five distinct   
Note: SSA denotes succinic semialdehyde and OSB quinone-supply mechanisms.   
denotes o-succinylbenzoate.