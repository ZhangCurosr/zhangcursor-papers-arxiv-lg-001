# CLIQUE CONSTRAINTS FOR EXACT CAUSAL STRUCTURELEARNING

Xiaoyu He AI Center Czech Technical University in Prague Karlovo namesti 13, Prague 12135 hexiaoyu@fel.cvut.cz

Petr Ryšavý AI Center Czech Technical University in Prague Karlovo namesti 13, Prague 12135 petr.rysavy@fel.cvut.cz

Pavel Rytírˇ AI Center Czech Technical University in Prague Karlovo namesti 13, Prague 12135 rytirpav@fel.cvut.cz

Jakub Marecekˇ AI Center Czech Technical University in Prague Karlovo namesti 13, Prague 12135 jakub.marecek@fel.cvut.cz

November 10, 2026

## ABSTRACT

Exact score-based causal structure learning supports explicit structural constraints and optimisation certificates, but its branch-and-cut search can be computationally demanding. We augment exact causal structure learning methods with an upper bound on the maximum clique size of the learned undirected skeleton, enforced at integer candidates through lazy cycle and clique constraints. On static linear-Gaussian Erdos–Rényi and scale-free DAGs, we evaluate callback order, alternative clique-cut˝ policies, incumbent repair, fractional separation, and Big-M bounds. We report causal-recovery metrics jointly with optimality gaps, primal and dual bounds, node throughput, MIPSOL events, and callback cost. Two variants receive particular attention: B3 selects the most violated minimal (k + 1)-clique inequality at each callback, whereas C3 repairs an infeasible integer candidate by deleting violating edges and refitting affected regression columns using QR-based least squares. The results do not indicate that clique constraints uniformly accelerate optimisation. However, their benefit depends on graph density, sample size, and cut-management policy, while large collections of correlated constraints can reduce node throughput. Tighter valid Big-M bounds reduced the fixed-budget optimality gap in most tested settings, without necessarily changing SHD or F1 at a fixed time limit. These findings characterize the computational trade-offs of clique-constrained exact causal learning, and identify selective separation and incumbent repair as promising directions.

## 1 Introduction

Causal structure learning from observational data is a fundamental problem in statistics and machine learning. Continuous methods such as NOTEARS represent acyclicity through a differentiable equality constraint [1]. Exact score-based methods instead introduce binary edge-selection variables and impose graph validity through integer or mixed-integer constraints, thereby providing globally optimal or certifiably near-optimal solutions under the specified model and stopping tolerance [2, 3, 4]. Related exact formulations apply this principle to causal structure learning under different graphical assumptions, including extensions that represent latent confounding [5, 6, 7, 8]. Although these formulations provide a principled mechanism for encoding graph validity, sparsity, degree restrictions, and prior knowledge, their computational cost remains substantial: the number of admissible structures grows exponentially, and solver performance depends on the formulation, relaxation strength, branching behavior, and management of constraints generated during the search.

Graph inequalities and on-demand separation are established techniques in combinatorial optimisation. Rather than enumerating every structural constraint in the initial formulation, a branch-and-cut method identifies a violated inequality at the current solution and adds it during the search [9]. Following this principle, we impose an upper bound on the clique size of the learned undirected skeleton. Unlike a global edge-count or degree restriction, a maximum-clique constraint directly excludes locally complete subgraphs and therefore serves as a local-density prior.

Enumerating all forbidden node subsets in advance is generally impractical, so we enforce the maximum-clique condition lazily. At a MIPSOL event, the callback examines an integer candidate and submits a lazy inequality through cbLazy() when the candidate violates the prescribed clique bound. This mechanism rejects structurally invalid integer candidates but does not directly strengthen fractional node relaxations. Strengthening a fractional relaxation instead requires a valid user cut submitted through cbCut() at MIPNODE. The computational effect of lazy clique separation is therefore empirical: a selected inequality may prevent repeated integer violations, whereas separation and additional model rows may increase callback overhead and node-relaxation cost.

The proposed condition should be interpreted specifically as a bound on the maximum clique of the learned skeleton. It is not a bound on clique-width or treewidth, and subsequent moralization or triangulation may introduce cliques that are absent from the original skeleton. We consequently do not assume that the restriction necessarily reduces downstream chordal complexity or accelerates optimisation. Instead, we examine when the structural prior improves causal-structure recovery or redirects the branch-and-cut search, and when its additional separation cost outweighs the reduction in the feasible graph class.

Our contributions are fourfold. First, we formulate and implement a lazy maximum-clique-size restriction for exact DAG structure learning. Second, we compare callback order and controlled clique-cut selection, including first and mostviolated minimal-(k + 1) policies. Third, we evaluate complementary strategies involving incumbent repair, fractional separation, and tighter Big-M bounds. Fourth, we introduce detailed instrumentation that separates primal progress, dual-bound progress, MIPSOL events, node throughput, cut submission, and callback time. These measurements expose when clique constraints improve DAG recovery, when they merely redirect the search, and when their additional optimisation cost outweighs the reduction in the feasible graph class

## 2 Graph knowledge background

An (undirected) graph $G = ( V , E )$ consists of a set of nodes $V$ and a set of edges $E \subseteq V \times V$ , where $( v _ { i } , v _ { j } ) \in E$ indicates that nodes $v _ { i }$ and $v _ { j }$ are adjacent. For a graph G, we use $V ( G )$ and $E ( { \bar { G } } )$ to denote its node set and edge set, respectively.

The adjacency matrix of a graph G is denoted by $B _ { G }$ , where we put ones on its diagonal. For two graphs $G$ and H, we say that G is a subgraph of H if

$$
V ( G ) \subseteq V ( H ) \quad { \mathrm { a n d } } \quad E ( G ) \subseteq E ( H ) ,
$$

which is denoted by $G \subseteq H$

For a graph $G = ( V , E )$ , a cycle oflength k is a sequence of distinct nodes $v _ { 1 } , v _ { 2 } , \dotsc , v _ { k } \in V$ such that $( v _ { i } , v _ { i + 1 } ) \in E$ for $i = 1 , \ldots , k - 1$ and $( v _ { k } , v _ { 1 } ) \in E . \mathrm { ~ A ~ }$ clique $\bar { C } \subseteq V$ of a graph $G = ( V , E )$ is a subset of vertices where every two distinct nodes are adjacent, i.e., $( v _ { i } , v _ { j } ) \in E$ for all $v _ { i } , v _ { j } \in C$ with $v _ { i } \neq v _ { j }$

## 3 Formulation

We formulate causal structure learning as a mixed-integer optimisation problem. Let n be the number of observations, d the number of variables, $\ b { X } \in \mathbb { R } ^ { n \times d }$ the contemporaneous data matrix and $Y ^ { ( t ) } \in \mathbb { R } ^ { n \times d } , t = 1 , \dots , p .$ the lagged data matrices. The optimisation variables are the contemporaneous weighted adjacency matrix $W \in \mathbb { R } ^ { d \times d }$ , lagged coefficient matrices $\ \dot { \boldsymbol { A } } ^ { ( t ) }$ , and binary edge-selection variables $z _ { i j } \in \{ 0 , 1 \}$ indicating whether edge $i  j$ is selected. For each sample i and node $j ,$ , the residual is

$$
r _ { i j } = X _ { i j } - \sum _ { k \ne j } X _ { i k } W _ { k j } - \sum _ { t = 1 } ^ { p } \sum _ { k = 1 } ^ { d } Y _ { i k } ^ { ( t ) } A _ { k j } ^ { ( t ) } .
$$

This yields the following regularized least-squares objective:

$$
\operatorname* { m i n } _ { W , A , z } \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } r _ { i j } ^ { 2 } + \frac { \lambda _ { 1 } } { d } \operatorname { R e g } _ { W } + \frac { \lambda _ { 2 } } { d } \operatorname { R e g } _ { A } ,\tag{1}
$$

where

$$
{ \mathrm { R e g } } _ { W } = { \left\{ \begin{array} { l l } { \lVert W \rVert _ { F } ^ { 2 } , } & { { \mathrm { i f ~ } } { \mathrm { R e g } } { \mathrm { T y p e ~ i s ~ } } l _ { 2 } , } \\ { \lVert z \rVert _ { 1 } , } & { { \mathrm { i f ~ } } { \mathrm { R e g } } { \mathrm { T y p e ~ i s ~ } } l _ { 1 } , } \end{array} \right. }
$$

$$
\mathrm { R e g } _ { A } = \left\{ \begin{array} { l l } { \displaystyle { \sum _ { t = 1 } ^ { p } \| A ^ { ( t ) } \| _ { F } ^ { 2 } , } } & { \mathrm { i f ~ } \mathrm { R e g } \mathrm { T y p e } \mathrm { i s } l _ { 2 } , } \\ { \displaystyle { \sum _ { t = 1 } ^ { p } \| A ^ { ( t ) } \| _ { 1 } , } } & { \mathrm { i f ~ } \mathrm { R e g } \mathrm { T y p e } \mathrm { i s } l _ { 1 } . } \end{array} \right.
$$

With squared loss, the formulation is a mixed-integer quadratic program (MIQP). An MILP variant is obtained with absolute loss by introducing auxiliary variables $u _ { i j } \geq 0$ , satisfying $- u _ { i j } \le r _ { i j } \le u _ { i j }$ and replacing (1) by

$$
\operatorname* { m i n } _ { W , A , z , u } \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { d } u _ { i j } + \frac { \lambda _ { 1 } } { d } \operatorname { R e g } _ { W } + \frac { \lambda _ { 2 } } { d } \operatorname { R e g } _ { A } ,
$$

which linearizes the absolute residuals.

Big-M constraints Binary variables are linked with edge weights through the standard big-M constraints,

$$
- M z _ { i j } \le W _ { i j } \le M z _ { i j } , \qquad i \ne j ,\tag{2}
$$

where M is a valid upper bound on the absolute edge weight. A smaller valid value strengthens the continuous relaxation, whereas an excessively large value permits a small fractional $z _ { i j }$ to support a comparatively large $W _ { i j }$ Reducing M without validating the coefficient range can, however, exclude feasible solutions. To exclude two-node cycles, we impose

$$
z _ { i j } + z _ { j i } \leq 1 , \qquad i \neq j .\tag{3}
$$

Longer directed cycles are eliminated lazily during branch-and-bound. Whenever the callback detects a directed cycle $C ,$ it generates the cut

$$
\sum _ { ( i , j ) \in C } z _ { i j } \leq | C | - 1 ,\tag{4}
$$

forcing at least one edge in every directed cycle to be removed.

Lazy clique constraints Let Q be a clique in this skeleton such that $| Q | > k ,$ , where k is the maximum permitted clique size (set to five in our experiments), and let

$$
P ( Q ) = \{ ( i , j ) : i < j , \ i , j \in Q \}
$$

denote the set of unordered node pairs in $Q .$ . Since $| P ( Q ) | = { \binom { | Q | } { 2 } }$ , the callback submits the lazy inequality

$$
\sum _ { ( i , j ) \in P ( Q ) } \left( z _ { i j } + z _ { j i } \right) \leq \binom { | Q | } { 2 } - 1 ,\tag{5}
$$

which removes the violating clique while preserving all skeletons whose maximum clique size is at most k. The callback separates violated cliques during branch-and-cut. The complete optimisation and callback workflow is summarized in Figure 1.

## 4 Methodology

Strength and selection of clique cuts. A limitation of Constraint (5) is that it only breaks the particular complete clique detected by the callback; it does not force the remaining graph to satisfy the maximum clique size k immediately. For example, an eight-node clique contains

$$
| P ( Q ) | = { \binom { 8 } { 2 } } = 2 8\tag{6}
$$

undirected edges, so the callback adds

$$
\sum _ { \{ i , j \} \subseteq Q } ( z _ { i j } + z _ { j i } ) \leq 2 7 .\tag{7}
$$

This cut requires only one of the 28 edges to be removed. It excludes the complete 8-clique, but the resulting graph may still contain a 7-clique or a 6-clique. When $k = 5 ,$ such candidates remain infeasible even though they satisfy Constraint (7). The original maximal-clique cut is therefore exact for the detected clique but weak with respect to all forbidden subcliques, and several similar infeasible MIPSOL candidates may be generated before repeated callback rounds eliminate them.

A stronger alternative separates minimal forbidden sets of size $k + 1$ . For each $S \subseteq Q$ with $| S | = k + 1$ , the valid inequality is

$$
\sum _ { \{ i , j \} \subseteq S } ( z _ { i j } + z _ { j i } ) \leq { \binom { k + 1 } { 2 } } - 1 .\tag{8}
$$

For the 8-clique example with $k = 5 ,$ there are ${ \binom { 8 } { 6 } } = 2 8$ six-node subsets, each requiring at least one of its 15 skeleton edges to be absent. Adding every row excludes more related integer candidate at once, but can increase separation, model updating, and subsequent node-processing costs. We therefore evaluate policies that vary the number and diversity of submitted rows, including violation-based ranking, per call cut limits, duplicate suppression, Jaccard diversity, and search history [10, 11, 12].

B3: most-violated minimal clique cut. At a MIPSOL callback, B3 first detects each oversized clique Q in the integer candidate and generates its minimal forbidden subsets $S \subseteq Q , | S | = k + 1$ . For each subset it evaluates the violation

$$
v ( S ) = \sum _ { \{ i , j \} \subseteq S } ( z _ { i j } + z _ { j i } ) - \left[ { \binom { k + 1 } { 2 } } - 1 \right] .\tag{9}
$$

The subsets are ranked by $v ( S )$ and the highest-ranked row is submitted through cbLazy $\left( \right)$ (subject to the configured per-callback limit). B3 is consequently an integer-candidate cut-selection policy, not a MIPNODE fractional separator. At a binary clique candidate, multiple minimal subsets can have identical violation; in that case B3 reduces to a deterministic tie-breaking rule and may behave similarly to the first- $( k + 1 )$ policy in method B1 (included in the Appendix). The Clique-Most-Violated (B3) implementation proceeds as follows.

Step 1. Read the integer edge-selection candidate and detect every maximal clique $Q$ whose size exceeds k.

Step 2. Generate the unique minimal forbidden node sets $S \subseteq Q$ with $| S | = k + 1$ . If several maximal cliques contain the same S, that subset appears only once in the candidate list.

Step 3. For every candidate $S ,$ construct its unordered node pairs and evaluate in Equation 9. The implementation also computes the normalized efficiency $v ( S ) / { \binom { k + 1 } { 2 } }$

Step 4. Rank candidates lexicographically by decreasing violation, decreasing efficiency, and then increasing sorted node tuple. The node tuple therefore provides deterministic tie-breaking.

Step 5. Select the highest-ranked row and apply the configured callback cut limit. In the B3 experiments, max\_clique\_cuts\_per\_callback=1 and clique-cut deduplication is disabled, so at most one row is submitted during the callback.

Step 6. Submit the selected inequality through cbLazy(), reject the current structurally invalid integer candidate, return control to Gurobi, and resume branch-and-cut search.

C3: QR-based incumbent repair. C3 supplements lazy rejection with a primal heuristic. When a MIPSOL candidate violates acyclicity or the clique bound, it retains the candidate support and weights initially, greedily deletes edges implicated in the violations, and records the parent sets that changed. Only regression columns associated with those parent sets are refitted by QR-based least squares. Before submission, the repaired assignment is validated against integrality, Big-M linking, acyclicity, and the maximum-clique constraint. A valid repair is submitted with cbSetSolution() and evaluated with cbUseSolution(), while the original infeasible candidate remains rejected through the required lazy constraints. C3 can improve the primal search by reusing an otherwise rejected candidate, but it neither adds fractional user cuts nor directly strengthens the dual bound.

## 5 Experiments

This section describes the data-generation procedure, solver configuration, experimental design, and empirical results.

![](images/775f9a4ea95e8a6eba0e5dad875a72884081e27fa8c44aed5303d70899ab35e7.jpg)  
Figure 1: Workflow of clique-constrained exact causal structure learning. The solver alternates between branchand-cut node processing, cycle separation, clique separation, incumbent acceptance, and bound updates until optimality or a stopping limit is reached.

## 5.1 Synthetic Data Generation

We use synthetic data because they provide a known ground-truth adjacency matrix. The benchmark contains cliquefiltered scale-free (cliquefree-SF) and Erdos–Rényi (cliquefree-ER) DAGs in a static linear-Gaussian setting˝ $( p = 0 )$ following the general simulation protocol of NOTEARS [1]. The SF generator produces sparse, hub-dominated graphs. For ER data, an undirected graph is sampled at the target edge ratio and oriented according to a random topological ordering. In both families, oversized cliques are detected and low-weight edges are removed until the prescribed clique bound is satisfied. Observations are then sampled from the resulting linear structural equation model (SEM).

Let $W \in \mathbb { R } ^ { d \times d }$ denote the weighted adjacency matrix of the ground-truth DAG, where $W _ { i i } = 0$ and $W _ { i j } \neq 0$ if and only if the edge $i  j$ is present in the data-generating graph. A sample $x \in \mathbb { R } ^ { d }$ is generated according to

$$
\begin{array} { r } { x = W ^ { \top } x + \varepsilon , \qquad \varepsilon \sim \mathcal { N } ( 0 , I _ { d } ) . } \end{array}\tag{10}
$$

Because the ground-truth graph is acyclic, $I - W ^ { \top }$ is invertible.

Samples are generated in topological order, with nonzero coefficients drawn from the configured weight intervals and unit-variance Gaussian noise. We consider $d \in \{ 1 1 , 1 5 , 2 0 , 2 5 , 3 0 , 3 5 , 4 0 , 4 5 , 5 0 \}$ nodes and $n \in \mathsf { \bar { \{ 2 0 0 , 2 0 0 0 \} } }$ observations. The nominal ER edge ratio is 4 or 5. Because the reported experiments are static, the inter-slice edge-ratio parameter is inactive. Detailed density and linearity statistics are reported in the supplementary material.

## 5.2 Solver Configuration

We use Gurobi with lazy cycle separation enabled. Unless otherwise stated, each run has a 1,800-second time limit, a target relative MIP gap of $\mathrm { 1 0 ^ { - 3 } } , \mathrm { \dot { M } } = 1 0 0$ , squared loss, and the regularization parameters $\lambda _ { 1 } = \lambda _ { 2 } = 1$ . Dedicated Big-M experiments evaluate alternative valid bounds. Hydra manages configurations and MLflow records parameters, metrics, and run metadata [13, 14]. Experiments were executed on MetaCentrum [15] using jobs allocated two CPUs, 32 GB RAM, and 30 GB of local scratch storage. The software environment uses Python 3.10, Gurobi, NetworkX, NumPy, and SciPy. Detailed implementation and environment information are provided in the supplementary code-anddata package.

<table><tr><td>Method</td><td>Integer-separation policy</td><td>Repair</td><td>M</td></tr><tr><td>ExDBN</td><td>None: the baseline enforces acyclicity but None imposes no clique-size restriction.</td><td></td><td>100</td></tr><tr><td>Clique</td><td>At each infeasible integer candidate, sub- None mit lazy cycle constraints and all detected violated clique constraints.</td><td></td><td>100</td></tr><tr><td>Clique ·Most-Violated</td><td>At each infeasible integer candidate, sub- None mit lazy cycle constraints and only the most-violated minimal (k + 1) clique con- straint.</td><td></td><td>100</td></tr><tr><td>Clique · QR Repair</td><td>Use the default Clique integer separation: Delete violation-causing edges, locally submit lazy cycle and violated clique con- refit affected regressions by QR, validate straints.</td><td>the repaired assignment, and submit it via cbSetSolution().</td><td>100</td></tr><tr><td>Clique · M = 20</td><td>Use the default Clique integer separation.</td><td>None</td><td>20</td></tr></table>

Table 1: Compared methods. Integer separation is performed at MIPSOL events through lazy constraints; fractional MIPNODE separation is evaluated separately.

![](images/06ee6389f1bb3bd215831c5112b62ad0a6d0d1b0d379f718e7c95034634434f4.jpg)

![](images/3e8a1c57e32a1d214b4c1003d14eb6a5810d1c9f3ae6066b10d8cf0ea09b7a54.jpg)  
Figure 2: Comparison of the SHD score and number of lazy-constraint submissions between Clique and ExDBN under different time limits setting and various dimensions d. The solid line represents the mean across multiple trials, with the shaded region indicating the standard deviation.

## 5.3 Experiment Setup

Table 1 summarizes ExDBN, Clique, and the principal variants evaluated on static cliquefree-SF and cliquefree-ER data. For each graph family and each $( d , n )$ pair, we use four seeds (1, 2, 3, 4). We record structural Hamming distance (SHD), F1 score, the relative MIP gap, submitted lazy constraints, primal and dual bounds, and solver diagnostics. We evaluate the best incumbent available at time limits of 5, 60, 120, 300, 600, 1200, 1800, 2700, 3600, 5400, and 7200 seconds. Thus, each fixed-time comparison concerns the best solution found within the specified budget rather than a final globally optimal solution.

## 5.4 Observations

Most large instances terminate at the time limit with a nonzero optimality gap. Accordingly, every fixed-time comparison concerns the best incumbent and best bound available at that budget, not a proven global optimum. A higher SHD for Clique at a fixed time therefore indicates weaker time-limited recovery on that run; it does not establish that the structural constraint reduces the accuracy of the fully optimized model. Under the fixed 1000-second computational budget, the reported gaps and node throughputs reflect solver progress rather than certified final-optimum quality.

![](images/1a58e4aafba6fac3af1b444305bae114aed9716b59ac7ce3148cf089d232fc01.jpg)  
Figure 3: SHD results over the ER and scale-free synthetic graph families. Solid lines and shaded areas denote the mean and standard deviation of multiple runs, respectively.

SHD and F1 evolve only when a newly accepted incumbent changes the learned graph. They can therefore remain constant while the solver continues to improve the dual bound and close the MIP gap. This stepwise behavior is visible in the time-limit experiments: recovery metrics often stabilize before the optimality certificate. It would nevertheless be incorrect to infer a monotone relation between gap and SHD. A smaller gap certifies the current objective more tightly, whereas SHD measures agreement with an external ground-truth graph; either quantity can improve without the other.

Thus, the poorer SHD of Clique in Figure 2 cannot be directly interpreted as “the constraints reduce accuracy.” A more accurate statement is that, under the same time budget, Clique has not yet found an incumbent of the same quality as ExDBN. For $d = 2 0$ , after 1,200 seconds, the gaps of the Clique and ExDBN optimisation models continue to decrease gradually, while the SHD score (and F1 score) of the predicted causal graphs tend to stabilize. For $d = 2 5$ , the overall change in SHD slows after 2,700 seconds. For $d = 3 { \bar { 0 } }$ , the overall change in SHD score slows after 3,000 seconds.

The distance of the corresponding graphical causal-model solution gradually decreases, but the numbers of callbacks are different when the solutions reach a stable state. For $d = 3 0$ , after the optimisation process reaches a stable regime, the stable number of lazy callbacks for ExDBN is larger than that for Clique, while their SHDs are similar. It cannot be concluded that ExDBN produces a more accurate causal model. Because Clique introduces violation checks for cliques larger than 5, when the clique and cycle violation checks do not overlap, Clique should have more callbacks than ExDBN. One plausible explanation, consistent with the callback diagnostics, is adding clique callbacks reduces the number of cycle callbacks, and the number of added clique callbacks is smaller than the corresponding reduction in cycle callbacks.

Figure 3 shows the Structural Hamming Distance (SHD) of four different methods as the number of nodes (d) increases from 5 to 55. In the cliquefree-er-4-0 and cliquefree-er-5-0 graph settings, the SHD of all methods remains low when $d \leq 3 0$ , but increases when $d > 3 5 .$ . Among them, Clique · QR Repair generally results in a lower SHD than ExDBN and standard Clique in these two settings when the graph scale grows larger. For the cliquefree-sf-5-0 setting, the SHD values of all four methods stay below 50 across all tested values of d, which is lower than the values observed in the other two graph types. For $n = 2 0 0$ , Clique · QR Repair attains the lowest SHD on both ER settings at $d \geq 4 0 ;$ for $n = 2 0 0 0$ , this advantage disappears and the methods are not consistently ordered.

In Figure 4, clique-5 provides the most stable performance, while the optimal clique bound remains dependent on graph size and sample size.

Big-M and fractional separation. Table 2 reports fixed-budget optimisation diagnostics. Under the 1,000-second budget, the effect of the linking bound is dataset-dependent. For ER-4-0, M = 20 achieved the smallest mean optimality gap (10.03%, compared with 17.60% for $M = 1 0 \bar { 0 } )$ and the highest node throughput. This advantage did not transfer uniformly to ER-5-0: $M = 1 0$ obtained the smallest gap (11.05%), while the results for $M = 2 0$ and $M = 1 0 0$ were close (11.39% and 11.65%, respectively). Thus, the results support using a tighter valid linking bound, but do not identify $M = 2 0$ as uniformly optimal across instances.

![](images/3f35dc9e38215b8fe1a02b784cc20b406b30aba92c3cf672b87c18104ec45606.jpg)  
Figure 4: Comparison between ExDBN and Cliques with maximum clique sizes $k \in \{ 3 , 4 , 5 \}$ under different time limit settings.

<table><tr><td>Study</td><td>Setting</td><td>ER-4-0</td><td>ER-5-0</td></tr><tr><td rowspan="4">Big-M</td><td>M = 10</td><td>16.75% / 2147</td><td>11.05% / 2733</td></tr><tr><td>M = 20</td><td>10.03% / 2621</td><td>111.39% / 2779</td></tr><tr><td>M = 50</td><td>13.81% / 2412</td><td>12.03% / 2755</td></tr><tr><td>M = 100</td><td>17.60% / 2453</td><td>11.65% / 2774</td></tr><tr><td rowspan="4">Fractional separation</td><td>No cbCut Root exhaustive</td><td>20.20% / 2110 20.24% / 2096</td><td>11.66% / 2760 11.69% /2718</td></tr><tr><td>Root greedy</td><td>20.27% / 2088</td><td>11.69% / 2721</td></tr><tr><td>Root + sparse</td><td></td><td></td></tr><tr><td>non-root</td><td>20.18% / 1297</td><td>12.82% / 1330</td></tr><tr><td></td><td>History-guided</td><td>20.53% / 200111.79% / 2582</td><td></td></tr></table>

Table 2: Big-M and fractional-separation ablation. Each cell reports Gap (%) / nodes per second under the fixed 1,000-second budget. D2, D3, and D5 submitted no user cuts. D4 submitted 146.5 / 108.5 user cuts on ER-4-0 / ER-5-0, respectively, while spending 41.09% / 47.38% of runtime in callbacks.

Figure 5 complements these optimisation diagnostics with graph-recovery results on ER-4-0 over increasing numbers of variables. The SHD curves show that Clique with M = 20 is generally competitive with the default Clique formulation and ExDBN, particularly at larger problem sizes; however, the uncertainty intervals widen substantially as the number of variables increases, and no method dominates uniformly for both sample sizes. Therefore, the improvement observed for M = 20 in the fixed-budget ER-4-0 optimisation experiment should not be interpreted as a universal improvement in structural-recovery accuracy.

Fractional clique separation did not improve performance in the current implementation. The root-exhaustive, rootgreedy, and history-guided variants (D2, D3, and D5) submitted no user cuts in these runs and therefore remained close to the no-fractional-cut baseline (D1), as shown in Table 2. The sparse non-root variant (D4) submitted, on average, 146.5 and 108.5 user cuts on ER-4-0 and ER-5-0, respectively; however, callback time increased to 41.09% and 47.38% of total runtime, while node throughput decreased substantially. In particular, its ER-5-0 gap increased from 11.66% for D1 to 12.82% for D4. These results do not support enabling the current fractional-separation procedure by default.

Callback-order comparison. Table 3 compares Cycle-only, Cycle-then-Clique, and Clique-then-Cycle under a fixed 1,800-second budget. Since the remaining MIP gaps are large, the reported quantities reflect intermediate solver progress rather than certified optimality.

For ER-4-0, the mixed policies explore substantially fewer nodes than Cycle-only (348,106 and 379,203 versus 1,914,601). Both mixed policies also submit fewer lazy constraints on ER-4-0 and ER-5-0; when cycles are separated first, the later clique stage submits relatively few clique constraints. However, this search-tree reduction does not improve the fixed-budget Gap: Cycle-only has the smallest Gap and largest best bound on both datasets. Thus, mixed callback orders reduce constraint submissions and may reduce node exploration, but do not uniformly improve dual-bound progress in the current implementation.

![](images/43b3043258e2ac3b8354417e34c3ec07d578e821c38da9eb9c63d3f8a3192d11.jpg)  
Figure 5: SHD on ER-4-0 as the number of variables increases. Clique with M = 20 is competitive with ExDBN and the default Clique formulation, but its advantage is not uniform across node counts or sample sizes. Shaded regions show variability across seeds.

<table><tr><td>Ratio Metric</td><td></td><td>Cycle-only Cycle→Clique Clique→Cycle</td><td></td></tr><tr><td rowspan="5">4</td><td>Submitted (103)</td><td>23.72</td><td>18.42 20.16</td></tr><tr><td>Nodes  $( 1 0 ^ { 6 } )$ </td><td>1.915</td><td>0.348 0.379</td></tr><tr><td>Gap (%)</td><td>52.81</td><td>59.69 59.91</td></tr><tr><td>Best bound</td><td>11.234</td><td>9.600 9.545</td></tr><tr><td>MIPSOL events</td><td>532</td><td>398 433</td></tr><tr><td rowspan="6">5</td><td>Submitted (103)</td><td>31.18</td><td>29.23 29.56</td></tr><tr><td>Nodes  $( 1 0 ^ { 6 } )$ </td><td>1.586</td><td>1.128 1.014</td></tr><tr><td>Gap (%)</td><td>53.68</td><td>55.32 56.29</td></tr><tr><td>Best bound</td><td>11.336</td><td>11.017 10.779</td></tr><tr><td>MIPSOL events</td><td>546</td><td>529 580</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 3: Callback-order diagnostics under a fixed budget. All experiments use static Erdos–Rényi synthetic data˝ with p = 0 and edge ratios of 4 and 5. Results are averages over seeds 1–4 with d = 20, n = 2000, M = 100, and a 1,800-second time limit. Submitted constraints and explored nodes are reported in thousands and millions, respectively.

## 6 Conclusion

We studied an explicit maximum-clique-size restriction for exact causal structure learning in the ExDBN mixed-integer formulation. Lazy constraints exclude violating integer candidates without enumerating all forbidden subsets in advance. When the solver terminates with a nonzero gap, the returned incumbent is feasible under the generated structural constraints but is not claimed to be globally optimal.

The experiments show that clique constraints introduce a trade-off rather than a uniform speedup. Selective minimal cuts can reduce redundant model growth, but integer violation scores often tie, which limits the distinction between B3 and a deterministic first-(k + 1) rule. C3 reuses rejected candidates through QR-based repair and can improve primal search on some datasets, but its benefit is not consistent across all graph families. Fractional user cuts must be evaluated separately from MIPSOL lazy cuts because only the former directly strengthen node relaxations.

Finally, the Big-M results support tighter validated coefficient bounds: relative to M = 100, smaller valid values generally improve relaxation quality and dual-bound progress, although the best tested value depends on the dataset. Future work should validate edge-specific bounds, evaluate B3 with fractional-aware scores, and report repair acceptance and incumbent-improvement rates for C3. These conclusions are limited to synthetic linear-Gaussian instances and runs that frequently terminate before proven optimality; evaluation on real, nonlinear, and larger-scale data remains necessary.

## Acknowledgements

This work was supported by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254) project.

## References

[1] Xun Zheng, Bryon Aragam, Pradeep K. Ravikumar, and Eric P. Xing. Dags with no tears: Continuous optimization for structure learning. In Advances in Neural Information Processing Systems 31, 2018.

[2] James Cussens. Integer programming for bayesian network structure learning. Quality Technology & Quantitative Management, 11(1):99–110, 2014.

[3] Mark Bartlett and James Cussens. Integer linear programming for the bayesian network structure learning problem. Artificial Intelligence, 244:258–271, 2017.

[4] James Cussens. Gobnilp: Learning bayesian network structure with integer programming. In Proceedings ofthe 10th International Conference on Probabilistic Graphical Models, pages 605–608, 2020.

[5] Pavel Rytíˇr, Aleš Wodecki, and Jakub Marecek. Exdag: an miqp algorithm for learning dags. ˇ arXiv preprint arXiv:2406.15229, 2024.

[6] Pavel Rytíˇr, Aleš Wodecki, Georgios Korpas, and Jakub Marecek. Exdbn: Learning dynamic bayesian networksˇ using extended mixed-integer programming formulations. Transactions on Machine Learning Research, 2025. arXiv:2410.16100.

[7] Rui Chen, Sanjeeb Dash, and Tian Gao. Integer programming for causal structure learning in the presence of latent variables. In Proceedings of the 38th International Conference on Machine Learning, pages 1550–1560, 2021.

[8] Petr Ryšavý, Pavel Rytíˇr, Xiaoyu He, Georgios Korpas, and Jakub Marecek. Exmag: Learning of maximallyˇ ancestral graphs. arXiv preprint arXiv:2503.08245, 2025.

[9] Sibel B. Sonuc, J. Cole Smith, and Illya V. Hicks. A branch-and-price-and-cut method for computing an optimal bramble. Discrete Optimization, 18:168–188, 2015.

[10] Paul Jaccard. The distribution of the flora in the alpine zone. New Phytologist, 11(2):37–50, 1912.

[11] Zeren Huang, Kerong Wang, Furui Liu, Hui-Ling Zhen, Weinan Zhang, Mingxuan Yuan, Jianye Hao, Yong Yu, and Jun Wang. Learning to select cuts for efficient mixed-integer programming. Pattern Recognition, 123:108353, 2022.

[12] Mark Turner, Thorsten Koch, Felipe Serrano, and Michael Winkler. Adaptive cut selection in mixed-integer linear programming. Open Journal ofMathematical Optimization, 4:1–28, 2023.

[13] Omry Yadan. Hydra - a framework for elegantly configuring complex applications. Github, 2019.

[14] Matei Zaharia, Andrew Chen, Aaron Chow, David Chu, Robert Grandl, Ali Kulkarni, Li Li, Siddharth Murching, Tomas Nykodym, Paul Ogilvie, Mani Parkhe, Areg Podgorny, Patrick Reynolds, Tatiana Tang, and Corey Zumar. Developments in MLflow: A system to accelerate the machine learning lifecycle. In Proceedings of the 2020 International Conference on Management ofData (SIGMOD/DEEM), pages 1–4. ACM, 2020.

[15] Jiˇrí Sitera, Petr Mukaˇrovský, Miroslav Kuda, Hana Rudová, and Ludek Matyska. Metacentrum—The Czechˇ National Grid Infrastructure. In Proceedings of the International Conference on High Performance Computing and Simulation, pages 42–50, 2011.

<table><tr><td colspan="4">ER-4-0</td><td colspan="4">ER-5-0</td><td colspan="4">SF-5-0</td></tr><tr><td>Nodes</td><td>Nom.</td><td>Real.</td><td> $R ^ { 2 }$ </td><td>Nodes</td><td>Nom.</td><td>Real.</td><td> $R ^ { 2 }$ </td><td>Nodes</td><td>Nom.</td><td>Real.</td><td> $R ^ { 2 }$ </td></tr><tr><td>11</td><td>80.00%</td><td>75.45% / 77.27%</td><td>0.847 / 0.875</td><td>11</td><td>100.00%</td><td>57.58% / 58.18%</td><td>0.737 / 0.808</td><td>11</td><td>100.00%</td><td>70.18% / 70.38%</td><td>0.890 / 0.884</td></tr><tr><td>15</td><td>57.14%</td><td>56.90% / 56.90%</td><td>0.928 / 0.923</td><td>15</td><td>71.43%</td><td>62.82% / 63.33%</td><td>0.897 / 0.916</td><td>15</td><td>71.43%</td><td>55.57% / 55.60%</td><td>0.875 / 0.874</td></tr><tr><td>20</td><td>42.11%</td><td>41.97% / 42.01%</td><td>0.900 / 0.897</td><td>20</td><td>52.63%</td><td>48.31% / 49.25%</td><td>0.907 / 0.926</td><td>20</td><td>52.63%</td><td>43.87% / 42.63%</td><td>0.874 / 0.879</td></tr><tr><td>25</td><td>33.33%</td><td>33.33% / 33.33%</td><td>0.910 / 0.913</td><td>25</td><td>41.67%</td><td>41.16% / 41.16%</td><td>0.929 / 0.930</td><td>25</td><td>41.67%</td><td>35.05% / 35.28%</td><td>0.866 / 0.865</td></tr><tr><td>30</td><td>27.59%</td><td>27.59% / 27.59%</td><td>0.909 / 0.908</td><td>30</td><td>34.48%</td><td>34.03% / 33.93%</td><td>0.918 / 0.921</td><td>30</td><td>34.48%</td><td>29.90% / 30.06%</td><td>0.875 / 0.881</td></tr><tr><td>35</td><td>23.53%</td><td>23.53% / 23.53%</td><td>0.927 / 0.925</td><td>35</td><td>29.41%</td><td>29.41% / 29.41%</td><td>0.941 / 0.941</td><td>35</td><td>29.41%</td><td>26.06% / 26.11%</td><td>0.871 / 0.874</td></tr><tr><td>40</td><td>20.51%</td><td>20.51% / 20.51%</td><td>0.902 / 0.902</td><td>40</td><td>25.64%</td><td>25.64% / 25.64%</td><td>0.951 / 0.951</td><td>40</td><td>25.64%</td><td>23.03% / 23.12%</td><td>0.861 / 0.864</td></tr><tr><td>45</td><td>18.18%</td><td>18.18% / 18.18%</td><td>0.929 / 0.930</td><td>45</td><td>22.73%</td><td>22.73% / 22.73%</td><td>0.943 / 0.943</td><td>45</td><td>22.73%</td><td>20.66% / 20.73%</td><td>0.866 / 0.866</td></tr><tr><td>50</td><td>16.33%</td><td>16.33% / 16.33%</td><td>0.924 / 0.926</td><td>50</td><td>20.41%</td><td>20.41% / 20.41%</td><td>0.956 / 0.957</td><td>50</td><td>20.41%</td><td>18.70% / 18.68%</td><td>0.887 / 0.875</td></tr></table>

Table S4: Synthetic-data density and linearity. Nom. is the nominal skeleton density, and Real. is the mean realized skeleton density after clique-free filtering. Each entry is reported as $n = 2 0 0 / n = 2 0 0 0$ . All values are means across the available generated instances.

This supplementary appendix extends the main paper “Clique Constraints for Exact Causal Structure Learning” by providing detailed method explanations, technical clarifications, and additional empirical results. Specifically, we evaluate synthetic graph densities and linear $R ^ { 2 }$ , dissect the diagnostic timeline while defining the core MIPSOL and MIPNODE callback quantities, catalog the experiments A-G based on improving directions and attach a table of mathematical notations and optimization terminology . Collectively, these materials serve as diagnostic support for the theoretical and empirical claims established in the main manuscript.

## A Synthetic Data Density and Linearity

<table><tr><td>Edge Ratio</td><td> $d _ { \mathrm { m a x } }$ </td><td>Target Edges</td><td>Max Capacity</td></tr><tr><td>2</td><td>5</td><td>10</td><td>10</td></tr><tr><td>3</td><td>7</td><td>21</td><td>21</td></tr><tr><td>4</td><td>9</td><td>36</td><td>36</td></tr><tr><td>5</td><td>11</td><td>55</td><td>55</td></tr></table>

Table S5: Theoretical limits for complete network saturation. The maximum node number $d _ { \mathrm { m a x } }$ represents the upper bound where an edge ratio r can force a 100% nominal skeleton density.

In the main results, we observed that the clique-constrained callback enhances search efficiency more significantly in denser causal graphs. To dig deeper into the performance of these constraints across different scenarios, we examine three key dataset characteristics: nominal skeleton density, realized skeleton density, and linear $R ^ { 2 }$ . The nominal skeleton density is calculated during the initial random graph generation step and remains invariant across random seeds. Conversely, the realized density represents the ratio of surviving edges in the true adjacency matrix $W _ { \mathrm { t r u e } }$ to the maximum possible skeleton capacity after applying the clique-free filter (which removes cliques larger than k). These metrics are formalized as follows:

$$
\rho _ { \mathrm { n o m } } = \operatorname* { m i n } \left( 1 , \frac { 2 r } { d - 1 } \right) , \quad \rho _ { \mathrm { r e a l } } = \frac { | E ( W _ { \mathrm { t r u e } } ) | } { \binom { d } { 2 } } ,\tag{S11}
$$

where r denotes the target edge ratio and d represents the number of nodes.

As reported in Table S4, while ER-5-0 generates denser causal graphs than ER-4-0 prior to filtering, its realized skeleton density drops below that of ER-4-0 at $d = 1 1$ . This structural inversion occurs because a higher initial edge ratio paired with a low node count severely exacerbates the formation of large, dense cliques; consequently, the clique-free filter must aggressively remove a substantially larger volume of violating edges to satisfy the structural constraint. This phenomenon highlights why comparative analyses should rely on the realized density rather than the nominal edge-ratio label alone. Complementing this finding, a comparison between the realized skeleton densities of ER-5-0 and SF-5-0 highlights the structural impact of their underlying generation algorithms. Specifically, unlike the random graph model, the scale-free model (SF-5-0) does not exhibit an unusually depressed realized density at Nodes = 11. This stability arises because its power-law degree distribution naturally distributes edges in a way that minimizes the dense, localized clique clusters typically found in Erdos-Rényi configurations. Consequently, structural˝ evaluations must distinguish between different graph typologies rather than relying solely on the edge-ratio baseline.

This structural behavior at low dimensions is directly linked to network saturation boundaries. As established previously, for a given edge ratio r, a graph achieves full saturation—reaching a 100% nominal skeleton density where the target edge count equals or exceeds the total capacity of a complete graph, i.e.,

$$
\lfloor r \cdot d \rfloor \geq { \binom { d } { 2 } }\tag{S12}
$$

—only when the node count d is sufficiently small. Because $\lfloor r \cdot d \rfloor \leq r \cdot d ,$ this condition implies $\textstyle r \cdot d \geq { \frac { d ( d - 1 ) } { 2 } }$ , which analytically simplifies to the deterministic boundary $d \leq \bar { 2 } r + \bar { 1 } . \ \mathrm { A s }$ d grows beyond this threshold, the quadratic capacity $\frac { d \mathsf { \check { ( } } d - 1 ) } { 2 }$ quickly outpaces the linear growth of the target edge allocation. The exact maximum node numbers $( d _ { \mathrm { m a x } } )$ for which each edge ratio completely saturates a graph are summarized in Table S5. Given these boundary constraints, cross-model performance comparisons should avoid configurations starting at or below these minimal saturation thresholds to ensure fair evaluation.

Additionally, to quantify the degree of linearity and data-generating fidelity of our synthetic graphs, we evaluate the mean linear $\dot { R } ^ { 2 }$ across all variables. For a specific node ${ \bar { \boldsymbol { \jmath } } } \in V$ , let $X _ { j }$ denote its observed data vector and $\mathbf { P a } ( j )$ represent the set of its true structural parents determined by $W _ { \mathrm { t r u e } } .$ . The linear $R ^ { 2 }$ for node j is formalized as:

$$
R _ { j } ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } \left( X _ { i j } - \hat { X } _ { i j } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { n } \left( X _ { i j } - \bar { X } _ { j } \right) ^ { 2 } } = 1 - \frac { \mathrm { R S S } _ { j } } { \mathrm { T S S } _ { j } } ,\tag{S13}
$$

where $\hat { X } _ { i j }$ is the value predicted by an ordinary least squares (OLS) linear regression of $X _ { j }$ on $X _ { \mathbf { P a } ( j ) } , \bar { X } _ { j }$ is the empirical mean of $X _ { j } , { \bar { \mathsf { R S } } } { \mathsf { S } } _ { j }$ is the residual sum of squares, and TSS<sub>j</sub> is the total sum of squares.

The globally reported linear $R ^ { 2 }$ profile in Table S4 represents the macro-average across all non-root nodes with non-empty parent sets, computed as:

$$
R _ { \mathrm { m e a n } } ^ { 2 } = \frac { 1 } { | V _ { \mathrm { n o n - r o o t } } | } \sum _ { j \in V _ { \mathrm { n o n - r o o t } } } R _ { j } ^ { 2 } .\tag{S14}
$$

The linear $R ^ { 2 }$ values, which strictly range between 0 and 1, directly quantify the capability of true structural parents to linearly explain the variance of their corresponding child variables. In our experiments, the consistently high baseline $( R ^ { 2 } \geq 0 . 8 \dot { 5 }$ , with many configurations exceeding 0.90) demonstrates that a vast majority of the variable variance is fully accounted for by linear combinations of true causal ancestors, signifying a robust causal signal-to-noise ratio with minimal noise interference. Conversely, an exceptionally depressed $\breve { R ^ { 2 } } \ : ( \mathrm { e . g . } , < 0 . 4 0 )$ would imply either a dominance of random error or severe underlying non-linearities, under which standard linear Mixed-Integer Programming (MIP) optimization models would encounter immense difficulty in accurately recovering the true graph topology.

When scaling the sample size from $n = 2 0 0 \mathrm { t o } n = 2 0 0 0$ , the observed narrow oscillations and marginal shifts in $R ^ { 2 }$ (such as the transition from 0.737 to 0.808 for ER-5-0 at $d = 1 1 )$ reflect standard asymptotic behavior rather than structural instability. In smaller samples $( n = 2 0 0 )$ coupled with dense local configurations, ordinary least squares (OLS) estimations are susceptible to minor small-sample overfitting biases and random noise fluctuations. Expanding the horizon to $n = 2 0 0 0$ invokes the law of large numbers, allowing the empirical sample $R ^ { 2 }$ to cleanly converge toward its true population parameter dictated by the underlying coefficient matrix $W _ { \mathrm { t r u e } }$ and error variance $\sigma ^ { 2 }$ . This asymptotic alignment provides a more precise and stable empirical characterization of the robust causal linear pathways surviving the clique-free filtering process, particularly within low-dimensional saturated boundaries.

## B Callback Events, Timing, and MIPSOL Interpretation

In this section, we formally define the MIPSOL, MIPNODE, cut, bound, timing, and checkpoint quantities used throughout our analysis, illustrating exactly where they are recorded during the branch-and-bound search in Figure S6. To support the empirical results presented in the main manuscript regarding the distinction between cycle-only, clique-only, and jointly violating candidates in the callback analysis, as well as the fractional user cuts findings.

The callback mechanism operates via two main event types: MIPSOL and MIPNODE, corresponding to integer candidate solutions and node LP relaxations, respectively. As distinguished in Table S6, these events delineate the computational workload triggered inside a callback from the solver’s subsequent independent search. At a MIPSOL event, the callback intercepts an integer candidate, verifies cycle and clique compliance, and rejects infeasible configurations by submitting lazy constraints via cbLazy(). Once the callback returns, Gurobi resumes its standard branch-and-bound routine until the next candidate is generated. Consequently, callback duration precisely measures the computational cost of separation and constraint submission, whereas the remaining runtime reflects baseline model processing, LP relaxations, branching decisions, primal heuristics, and node search.

![](images/1cdf392a131ee3474e6a3d618273726d28ebe51d3328c149adac706d3fd0c234.jpg)  
Figure S6: Timing points in branch-and-bound search. For each integer candidate, the callback first performs cycle and clique separation and may submit lazy constraints. After control returns to Gurobi, the solver continues branch-and-bound search until a subsequent candidate, bound update, or termination event.

When analyzing these events, callback frequencies, detected violations, and successfully submitted constraints must be treated as distinct metrics. A MIPSOL record represents a downstream outcome of the global search process rather than a direct measure of immediate relaxation strength. Similarly, the volume of MIPNODE invocations, the frequency of violated fractional clique inequalities, and the total of submitted user cuts describe separate structural dimensions of the optimization trajectory.

Interpreting MIPSOL Counts. A larger MIPSOL count is neither inherently beneficial nor detrimental; its utility must be evaluated jointly with incumbent improvements, dual bound progress, rejection rates, and node pruning efficiency. An elevated count is productive when accepted candidates repeatedly update the incumbent, thereby tightening the objective cutoff and allowing more branch-and-bound nodes to be pruned. Conversely, it becomes an optimization bottleneck when lazy cycle or clique constraints continuously reject candidates without advancing either the primal or dual bounds. For example, under the configuration M = 10 with seed 3, Clique generated 741 MIPSOL events compared to 735 for ExDBN. However, both models converged to nearly identical objective/bound pairs (24.8/8.2 versus 24.6/8.3) and relative gaps (0.67 versus 0.66). In this scenario, the six additional MIPSOL events yielded negligible optimization progress, primarily incurring redundant separation and search overhead.

Incorporating fractional user cuts introduces a severe performance bottleneck. The intensive separation overhead outweighs its dual-bound strengthening benefits, with the Root + sparse non-root configuration spending a substantial portion of runtime (41.09% on ER-4-0 and 47.38% on ER-5-0) confined within callbacks. Under the fixed 1, 000-second budget, the computational burden of active separation drastically reduces node exploration speeds. For instance, the throughput drops from 2, 110 to 1, 297 nodes/second compared to the baseline No cbCut configuration, yielding negligible improvements or even slight degradation in the final relative optimality gap. These trends demonstrate that exhaustive or heuristic fractional cut generation at the root or non-root nodes fails to alleviate the dual-bound bottleneck efficiently. Consequently, standard integer-candidate lazy constraints via MIPSOL remain a computationally superior strategy over fractional MIPNODE separation for this causal structure learning framework.

Table S6: Operational distinction between MIPSOL and MIPNODE events.
<table><tr><td>Event Type</td><td>Description</td><td>Callback Action</td></tr><tr><td>MIPSOL</td><td>Triggered when the solver finds an integer can- didate, originating either from a branch-and- bound node or an internal primal heuristic.</td><td>cbLazy() may reject the candidate by appending valid lazy constraints.</td></tr><tr><td>MIPNODE</td><td>Triggered when a continuous LP relaxation is available at a branch-and-bound node. The so- lution is typically fractional.</td><td>cbCut () may submit globally valid user cuts; root-node diagnostics are also logged.</td></tr></table>

![](images/b88e13cb9d91b3fa88156a4259bf3323135f452d5d8976a1940a6413b7635fd0.jpg)

![](images/dfb44dc430b705737c7601f7ff2bdd5148240d5feea082bf4a82e975e8f54479.jpg)  
Figure S7: Relative MIP gap and F1 score for Clique and ExDBN across time limits and dimensions d. Lines show averages over seeds and shaded regions report variability across runs.

## C Additional Results

This section presents the supplementary figures and detailed experimental protocols for Improvements A–G discussed in the main manuscript.

Figure S7 illustrates that the relative MIP gap often continues to decrease monotonically long after the F1 score has stabilized. This trajectory profile reflects a fundamental decoupling between bounding progress and structural recovery: the gap evolves whenever either the primal or dual bound improves, whereas the F1 score adapts exclusively when a newly accepted incumbent alters the underlying graph topology. Consequently, these trends demonstrate that causal graph recovery quality is bounded by computational convergence, yet further numerical optimization faces diminishing returns regarding structural refinement once the search layout settles.

Stability Diagnostics In addition to the clique-constrained MILP formulation, we evaluate an edge-limit MILP variant to analyze stability, where structural complexity is controlled by static linear constraints imposed at model construction time rather than dynamically separated clique cuts. These constraints restrict both the in-degree and out-degree of every node, remaining invariant throughout the optimization trajectory. To match our experimental envelope where the maximum dimensionality reaches $d = 3 5$ , we evaluate degree bounds ranging from 0 to 35 formalized as:

$$
\sum _ { i \in V } z _ { i j } \le d _ { \mathrm { i n } } ^ { \mathrm { m a x } } , \quad \forall j \in V ,\tag{S15}
$$

$$
\sum _ { j \in V } z _ { i j } \le d _ { \mathrm { o u t } } ^ { \operatorname* { m a x } } , \quad \forall i \in V ,\tag{S16}
$$

where $z _ { i j } \in \{ 0 , 1 \}$ denotes the binary edge-selection variable associated with the learned adjacency matrix. By enforcing a global degree envelope through a priori linear structures rather than via runtime lazy cuts, this formulation establishes a controlled environment to study algorithmic behavior, where relaxing the edge boundaries expands the feasible search space of graph topologies but simultaneously exacerbates the combinatorial complexity of the underlying branch-and-bound tree.

<table><tr><td>Improvements</td><td>Methods</td><td>Experiments</td></tr><tr><td rowspan="2">Reduce redundant searches</td><td>A. Delayed Separation</td><td>Reduce clique separation, increase solving nodes per sec and close gap.</td></tr><tr><td>C. Repair Incumbent</td><td>Repair an infeasible MIPSOL, locally refitting support with ridge/QR.</td></tr><tr><td rowspan="2">Control adding constraint number</td><td>B. Controlled Cut Addition</td><td>First oversized cliques (original), First size-min-k+1-clique most-violated size-min-k+1-clique.</td></tr><tr><td>F. Stronger Clique Cuts</td><td>Top-3-size-min-k+1-cliques, Top-5-size-min-k+1-cliques, All- unique-size-min-k+1-cliques (adding  $\binom { 8 } { 6 }$  constraints).</td></tr><tr><td rowspan="2">Strengthen the LP relaxation and dual bound</td><td>D. Fractional Separation</td><td>Inspect the fractional MIPNODE relaxation and submit valid violated clique inequalities through cbCut ().</td></tr><tr><td>E. Big-M Relaxation</td><td>M ∈ {10, 15, 20, 25, 50, 100} and edge-specific  $M _ { i j }$  linking bounds.</td></tr><tr><td>Checkpoints</td><td>G. Time-Limit Optimization</td><td>Run one continuous solve and record metrics at fixed check- points.</td></tr></table>

Table S7: Improvements and experimental directions. Variants B1–F2, B3, C3, and E2 are systematically compared in the main manuscript.

The continuous empirical trajectories and exact structural plateaus resulting from this setup are collectively systematized in Figure S8. As illustrated in the graphical profiles (a and b), relaxing the edge limits initially enhances statistical recovery if a tighter bound previously excluded true causal relationships, yet it eventually slows down dual bound progression and optimality-gap closure, capturing a clear statistical–computational trade-off. The empirical stability boundaries cataloged in the unified subtable (c) further confirm this trade-off; for lower dimensions $( d \leq 1 5 )$ , expanding the horizon from n = 200 to n = 2000 cleanly reduces the minimal stable SHD bound from 11.5 to an exceptionally tight value of 2.0. These combined diagnostics demonstrate that causal graph recovery quality is fundamentally driven by computational convergence, showing that further structural refinement faces diminishing returns long before the full optimality gap is closed.

## D Notation Table

Notations and terminologies are listed in Table S8.

![](images/47b83eaf012ed9cd12e4fb9b15d53c15dc2c06962a7b0ebd8c74b004b0da5c9f.jpg)  
(a) Structural Hamming Distance (SHD) trends

![](images/c1f032f2b30627667af2ae08c50257fa2e859887a69747ee227decf0d9c963e7.jpg)  
(b) Relative MIP gap progression

<table><tr><td rowspan="3">Nodes (d)</td><td colspan="4">Minimal Stable SHD Bound</td><td colspan="4">Minimal Stable GAP Bound</td></tr><tr><td colspan="2">Sample n = 200</td><td colspan="2">Sample  $n = 2 0 0 0$ </td><td colspan="2">Sample n = 200</td><td colspan="2">Sample n = 2000</td></tr><tr><td>Degree</td><td>SHD Value</td><td>Degree</td><td>SHD Value</td><td>Degree</td><td>GAP Value</td><td>Degree</td><td>GAP Value</td></tr><tr><td>5</td><td>3</td><td>3.0</td><td>3</td><td>2.0</td><td>1</td><td>0.0000</td><td>1</td><td>0.0000</td></tr><tr><td>10</td><td>7</td><td>7.0</td><td>8</td><td>5.0</td><td>9</td><td>0.0100</td><td>9</td><td>0.0099</td></tr><tr><td>15</td><td>10</td><td>11.5</td><td>11</td><td>2.0</td><td>20</td><td>0.27*</td><td>14</td><td>0.29*</td></tr><tr><td>20</td><td>20</td><td>2.0</td><td>14</td><td>0.0</td><td>25</td><td>0.61*</td><td>18</td><td>0.56*</td></tr><tr><td>25</td><td>18</td><td>1.0</td><td>18</td><td>0.0</td><td>23</td><td>0.63*</td><td>24</td><td>0.65*</td></tr><tr><td>30</td><td>20</td><td>0.0</td><td>20</td><td>0.0</td><td>28</td><td>0.79*</td><td>29</td><td>0.78*</td></tr><tr><td>35</td><td>25</td><td>1.0</td><td>24</td><td>0.0</td><td>34</td><td>0.8517</td><td>19</td><td>0.82*</td></tr></table>

(c) Empirical thresholds for structural and optimization stability  
Figure S8: Edge-limit stability trajectories and empirical plateaus across dimension regimes. Panels (a) and (b) trace the global continuous optimization dynamics, while the unified data subtable (c) catalogs the exact minimal degree boundaries $( d _ { \mathrm { m a x } } )$ beyond which additional metric relaxation yields marginal structural variation. Asterisks (∗) denote approximate plateaus resolved to two decimal places. These thresholds are descriptive diagnostic references and do not represent formal asymptotic convergence guarantees.

<table><tr><td>Concept</td><td>Notation / Gurobi term</td><td>Definition</td></tr><tr><td>Branch-and-bound</td><td>B&amp;B</td><td>Tree-search for mixed-integer optimization; nodes define subproblems whose continuous relaxations supply bounds.</td></tr><tr><td>Search tree</td><td> $\tau$ </td><td>Collection of generated B&amp;B nodes; a root-to-node path records the</td></tr><tr><td>Node queue</td><td>Q; MIP_NODLFT</td><td>branching decisions. Unexplored B&amp;B nodes remaining in the solver queue.</td></tr><tr><td>Current node</td><td> $v \in \mathcal T$ </td><td>The B&amp;B subproblem currently processed. Its LP relaxation can be fractional.</td></tr><tr><td>Callback</td><td>MIPSOL; MIPNODE</td><td>User code at solver events. MIPSOL inspects integer candidates; MIPNODE</td></tr><tr><td>Incumbent</td><td> $z ^ { \mathrm { i n c } } ; \mathtt { M I P } .$  _OBJBST</td><td>inspects fractional relaxations. Best globally feasible integer solution found so far.</td></tr><tr><td>Primal bound / upper bound</td><td> $U B = f ( z ^ { \mathrm { i n c } } )$ </td><td>Objective value of the current incumbent (minimization convention).</td></tr><tr><td>Dual bound / lower bound</td><td>LB; MIP_OBJBND</td><td>Global lower bound on the optimal objective, derived from active node relaxations.</td></tr><tr><td>Relative optimality gap</td><td> $U B - L B$ </td><td>Normalized separation between primal and dual bounds.</td></tr><tr><td>Clique separation</td><td> $\overline { { \operatorname* { m a x } \{ | U B | , \epsilon \} } }$   $\mathcal { S } _ { \mathrm { c l i q u e } }$ </td><td>Detection of violated cliques. cbLazy() rejects candidates at MIPSOL;</td></tr><tr><td>Clique lazy constraint</td><td> $\mathtt { c b L a z y ( ) }$ </td><td>cbCut () adds cuts at MIPNODE. Dynamically generated constraint enforcing integer feasibility without</td></tr><tr><td>Clique user cut</td><td> $\mathtt { c b c u t ( ) }$ </td><td>directly strengthening fractional relaxations. Globally valid inequality submitted at fractional MIPNODE relaxations to</td></tr><tr><td>Samples, variables, and lags</td><td> $n , d , p$ </td><td>strengthen them. Numbers of observations, graph variables, and temporal lags  $( p = 0$ </td></tr><tr><td>Data matrices</td><td> $X ; Y ^ { ( t ) }$ </td><td>represents static data). Contemporaneous and lag-t data matrices.</td></tr><tr><td>Coefficient matrices</td><td> $W ; A ^ { ( t ) }$ </td><td>Contemporaneous adjacency matrix  $( W _ { i j } \ne 0 \implies i \to j )$  and lag-t</td></tr><tr><td>Edge-selection variable</td><td> $z _ { i j } \in \{ 0 , 1 \}$ </td><td>coefficient matrix. Binary indicator whether directed edge  $i  j$  is selected.</td></tr><tr><td>Residual and absolute-loss</td><td> $r _ { i j } ; u _ { i j }$ </td><td>Regression residual and its linearized absolute-value auxiliary variable.</td></tr><tr><td>Regularization</td><td> $\lambda _ { 1 } , \lambda _ { 2 } ;$ </td><td>Penalty weights and regularizers for contemporaneous and lagged coeffi-</td></tr><tr><td>Big-M bound</td><td> $\mathrm { R e g } _ { W } , \mathrm { R e g } _ { A }$  M</td><td>cients.</td></tr><tr><td></td><td></td><td>Bound in  $- M z _ { i j } \ \leq \ W _ { i j } \ \leq \ M z _ { i j }$  . Tighter valid values strengthen relaxation .</td></tr><tr><td>Graph Support</td><td> $G = ( V , E )$ </td><td>Selected directed graph.</td></tr><tr><td>Cycle</td><td>C</td><td>Edge skeleton. Directed cycle .</td></tr><tr><td>Clique and clique bound</td><td> $Q ; k ; \omega ( G )$ </td><td>Clique  $Q ,$  maximum permitted clique size  $k ,$  and graph clique number </td></tr><tr><td></td><td></td><td> $( \omega ( \bar { G } ) \leq k )$ </td></tr><tr><td>Clique-pair set Minimal forbidden subset</td><td> $P ( Q )$   $S \subseteq Q , | S | = k { + } 1$ </td><td>Unordered pairs in  $Q ,$  with  $| P ( Q ) | = { \binom { | Q | } { 2 } }$ </td></tr><tr><td>Clique-cut violation</td><td> $v ( S )$ </td><td>Smallest subset violating the clique bound.</td></tr><tr><td>B3</td><td>Most-violated mini-</td><td>Violation score used by B3 to rank minimal  $( k + 1 )$  clique inequalities. MIPSOL policy submitting highest-ranked minimal  $( k \bar { + } 1 )$  inequalities</td></tr><tr><td></td><td>mal clique cut</td><td>via cbLazy() under cut limit.</td></tr><tr><td>C3</td><td>QR-based incum- bent repair</td><td>Primal heuristic that deletes violating edges, refits columns via QR, and submits via cbSetSolution().</td></tr><tr><td>Recovery metrics</td><td>SHD; F1</td><td>SHD measures graph-edit distance to ground truth; F1 is the harmonic mean of precision and recall.</td></tr></table>

Table S8: Notation used throughout the paper. The table consolidates optimization terminology, statistical-model notation, and graph, method, and evaluation notation