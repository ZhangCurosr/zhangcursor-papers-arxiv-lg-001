# Beyond Prediction Residuals: Conformal Regret Calibration for Robust Contextual Linear Optimization

Lingjie Zhao<sup>1</sup>, Hansheng Jiang<sup>2</sup>, Wei Qi<sup>1</sup>

<sup>1</sup>Department of Industrial Engineering, Tsinghua University <sup>2</sup>Rotman School of Management, University of Toronto

## Abstract

Predict-then-optimize systems convert contextual forecasts into decisions, yet conformal robust optimization typically calibrates prediction residuals rather than downstream loss. For linear optimization, this parameter-space geometry can be badly misaligned with decision risk: large objective perturbations may preserve the optimizer, while small boundarycrossing perturbations may cause substantial regret. We introduce Conformal Regret Sets, a post-hoc, model-agnostic framework that conformalizes an optimization-derived regret surrogate and uses its calibrated level set for minimax-regret refinement. We first show that directly conformalizing exact regret, although suficient to certify the nominal predict-thenoptimize decision, yields a degenerate robust problem: absent an exogenous bounded support, every alternative decision has unbounded worst-case regret. We therefore use SPO+, a pointwise upper bound on exact regret, as the conformal score, obtaining distribution-free finite-sample marginal regret guarantees under exchangeability. For contextual linear programs, the resulting set decomposes into polyhedral pieces aligned with the normal fan of the feasible polytope. A vertex is active if and only if its predicted suboptimality gap is at most half the calibrated threshold η, enabling local, outputsensitive generation of all relevant pieces. Moreover, if Γ is the predicted gap to the closest competing vertex, the nominal and refined decisions admit the sharper worst-case-regret certificate $( \eta - 2 \Gamma ) _ { + }$ , with an exact early exit when η ≤ 2Γ. Experiments on contextual fractional knapsack validate the geometry and certificates: continuous refinement reduces the 0.95-quantile and $\mathrm { C V a R _ { 0 . 9 5 } }$ of regret by 9.1% and 13.1% relative to nominal PTO, while vertex-restricted refinement changes only 10.0% of nominal decisions.

## 1 Introduction

Predictive models are increasingly embedded in downstream optimization systems. In the standard predict-then-optimize (PTO) pipeline, observable features are mapped to uncertain problem parameters, and the resulting point prediction is inserted into a deterministic optimization problem. This modular paradigm is natural for contextual linear programs (LPs) arising in routing, allocation, and portfolio selection (Bertsimas and Tsitsiklis 1998; Bertsimas and Kallus 2020). Yet a point forecast alone ofers no protection against prediction error, and optimizing an estimated objective can amplify modest errors into substantial post-decision loss (Smith and Winkler 2006).

A common remedy is to replace the point prediction with a data-driven uncertainty set and solve a robust optimization problem. Conformal prediction (CP) is attractive for this purpose because it is model-agnostic and provides distributionfree, finite-sample marginal coverage under exchangeability (Angelopoulos and Bates 2023). Existing conformal approaches to contextual optimization typically calibrate residual scores, such as coordinatewise errors or norm distances between predicted and realized objective vectors (Sun, Liu, and Li 2023; Patel, Rayan, and Tewari 2024). The resulting boxes, ellipsoids, or norm balls are statistically valid in parameter space, but need not be aligned with downstream decision risk.

For an LP, residual coverage is not decision calibration. The objective space is partitioned by the normal fan of the feasible polytope: all costs in one cone select the same optimal vertex. A large perturbation can remain within a cone and incur no decision loss, whereas a smaller perturbation can cross a nearby boundary, change the optimizer, and cause substantial regret. Some directions also shift every feasible objective value equally and are therefore irrelevant to both the optimizer and pairwise regret. Residual-based scores do not distinguish these decision-invariant directions from decisionsensitive ones. This leads to the following question: Can conformal uncertainty instead be calibrated directly in units of downstream regret while still supporting nontrivial robust re-optimization?

Conformal validity alone is not the obstacle: the score sublevel set must also support meaningful comparison of alternative actions. The most direct score is exact decision regret, or the Smart Predict-then-Optimize (SPO) loss (Elmachtoub and Grigas 2022). Although its conformal quantile immediately certifies the nominal PTO decision, we prove that, without an independently imposed bounded objective support, its sublevel set is degenerate for re-optimization: every alternative decision has infinite worst-case regret. Thus, a score that is ideal for certifying one fixed action can be unusable for comparing actions.

We therefore introduce Conformal Regret Sets based on SPO+, a pointwise upper bound on exact regret. Splitconformal calibration produces a threshold η in objectivevalue units, yielding a finite-sample marginal regret guarantee and an objective-space set for minimax-regret refinement. The construction is post hoc and compatible with any blackbox predictor. For contextual LPs, the SPO+ level set has an exact normal-fan structure that makes this otherwise nonconvex robust problem analyzable and amenable to outputsensitive solution methods.

Our contributions are threefold.

• Regret calibration and the need for a robustifiable score. We formalize conformal calibration with optimization regret, establish the degeneracy of exact-SPO sublevel sets for robust refinement, and show that SPO+ calibration provides a distribution-free finite-sample guarantee for the regret of the nominal PTO decision.

• Exact normal-fan geometry and local active-set generation. We derive a closed polyhedral representation of the SPO+ conformal regret set, exact whenever the hindsight optimizer is unique, and show that its unbounded directions are decision-invariant. A vertex generates a nonempty piece if and only if its predicted suboptimality gap is at most $\eta / 2 .$ . These active vertices form a connected subgraph of the polytope’s one-skeleton, enabling outputsensitive generation from the nominal solution rather than enumeration of all vertices.

• Margin-certified minimax-regret refinement. Let Γ be the predicted gap between the nominal optimizer and its closest competing vertex. We prove that the certified worst-case regret of the nominal, vertex-restricted, and continuous refinements is at most $( \eta - 2 \Gamma ) _ { + }$ . With $\eta = \eta _ { \alpha }$ , the same bound controls realized regret with probability at least $1 - \alpha$ . Hence $\eta \leq 2 \Gamma$ gives a zero worst-case-regret certificate over the calibrated set and an exact early-exit test. Otherwise, adjacency-based generation and a fixed-piece LP oracle evaluate the regret adversary without enumerating all comparator vertices; once the active pieces are generated, the continuous refinement admits a finite LP formulation.

## 2 Related Work

Conformal prediction for optimization. CP provides distribution-free prediction regions under exchangeability (Angelopoulos and Bates 2023), and a growing literature uses these regions as uncertainty sets for contextual robust optimization (Zhao, Jiang, and Qi 2026). Early methods primarily calibrate parameter-space errors, including ellipsoidal sets (Johnstone and Cox 2021), residual-based sets for contextual LPs (Sun, Liu, and Li 2023), and generative, potentially nonconvex regions (Patel, Rayan, and Tewari 2024). Conformal ideas have also been combined with inverse optimization and adherence-aware prescription (Chan, Delage, and Lin 2024). More broadly, conformal risk control extends conformal guarantees beyond miscoverage to generic monotone losses (Angelopoulos et al. 2024).

Decision-aware conformal score design. Beyond residual calibration, recent work makes uncertainty quantification for optimization more decision-aware. End-to-end conditional robust optimization trains predictive and uncertainty models using downstream decision loss (Chenreddy and Delage 2024), while conformal methods learn calibrated convex or polyhedral set geometry from the induced robust loss (Yeh et al. 2025; Chen, Zhou, and Zhu 2026). In particular, Chen,

Zhou, and Zhu (2026) learn geometry within a parameterized polyhedral family and then recalibrate on independent data. Utility-directed CP incorporates a user-specified downstream utility into conformal score design (Cortes-Gomez et al. 2025); Hu et al. (2026) design sets under explicit robustness constraints, and Zhou and Zhu (2026) calibrate downstream robustness levels. CREDO combines conformal and inverse-optimization geometry to audit the probability that a fixed candidate decision is suboptimal (Zhou, Orfanoudaki, and Zhu 2026). Our construction derives set geometry directly from the optimization-based SPO+ score. The resulting nonconvex, normal-fan-aligned sets admit exact activepiece screening, margin-adaptive certificates, and certified minimax-regret refinement for contextual LPs.

Smart predict-then-optimize and decision-focused learning. Decision-focused learning (DFL) addresses the mismatch between prediction accuracy and downstream performance by diferentiating through, or otherwise exploiting, the optimization problem during training (Donti, Amos, and Kolter 2017; Wilder, Dilkina, and Tambe 2019; Mandi et al. 2022; Gupta and Huang 2024). The Smart PTO framework uses decision regret as the training objective and introduces SPO+ as a tractable convex surrogate for linear optimization (Elmachtoub and Grigas 2022). Liu and Grigas (2021) study surrogate-risk calibration, relating excess SPO+ risk to excess expected SPO risk to support predictor learning. Subsequent work develops optimization-aware losses and training procedures across linear, combinatorial, probabilistic, and causal settings (Qi, Grigas, and Shen 2025; Berden et al. 2025; Zhang et al. 2025); robust-constraint-aware training is considered through the SPO-RC loss (Im, Benslimane, and Grigas 2025). Our calibration is post hoc: we apply splitconformal calibration to SPO+ scores from a fixed predictor, obtaining finite-sample marginal coverage and regret guarantees for minimax-regret refinement.

## 3 Problem Setup and Regret Calibration

## 3.1 Contextual Linear Optimization and Decision Regret

Consider a linear program (LP) that takes the following standard form:

$$
\operatorname* { m i n } c ^ { \top } { x } ,\tag{1}
$$

$$
\mathrm { s . t . } A x = b , x \geq 0 ,\tag{2}
$$

where $x \in \mathbb { R } ^ { n }$ is the decision variable, $c \in \mathbb { R } ^ { n }$ is the uncertain parameter, and $A \in \mathbb { R } ^ { m \times n }$ and $b \in \mathbb { R } ^ { m } ( m \leq n )$ define the deterministic system constraints. We assume throughout that $\mathcal { X } : = \{ x \in \mathbb { R } ^ { n } : A x = b , x \geq 0 \}$ is a nonempty compact polytope. Let $\mathcal { V } : = \operatorname { e x t } ( \mathcal { X } ) = \big \{ v _ { 1 } , . . . , v _ { K } \big \}$ be its complete set of distinct geometric vertices, and write $[ K ] \ : = \ \{ 1 , \ldots , K \}$ . For any realized cost vector $c ,$ let $x ^ { * } ( c ) \in \arg \operatorname* { m i n } _ { x \in \mathcal { X } } c ^ { \top } x$ be a fixed deterministic measurable vertex-valued optimal selector. Throughout the main text, we assume that every cost vector under consideration induces a unique optimizer, so the selector is immaterial there; tie-boundary extensions are treated in the appendix.

For a feasible decision $x \in \mathcal { X }$ and a realized objective

vector c, define its decision regret as

$$
\operatorname { R e g } ( x , c ) : = c ^ { \top } x - c ^ { \top } x ^ { * } ( c ) .\tag{3}
$$

It measures the excess realized cost of x relative to the hindsight optimum and is therefore nonnegative.

In the Predict-Then-Optimize paradigm, a predictor $\hat { f }$ $\mathbb { R } ^ { d } \to \mathbb { R } ^ { n }$ maps observable features z to a point estimate cˆ of the uncertain parameter c. The PTO simply prescribes $x ^ { * } ( \hat { c } )$ , but the fragility of decisions based on point forecasts motivates robust methods, i.e., replacing cˆ with an uncertainty set and optimizing against its worst-case realization.

## 3.2 Split Conformal Calibration

To quantify predictive uncertainty, we leverage split conformal prediction (Angelopoulos and Bates 2023). Let $\mathcal { D } _ { c a l } =$ $\{ ( z _ { i } , \bar { c } _ { i } ) \} _ { i = 1 } ^ { N }$ be a held-out calibration sample set, and define a conformal score function $s ( z , c )$ measuring the discrepancy between the prediction $\hat { f } ( z )$ and the realization c. Evaluating this score on $\mathcal { D } _ { c a l }$ yields empirical scores ${ \cal { S } } = \{ s _ { i } \} _ { i = 1 } ^ { N }$ and the order statistics $s _ { ( 1 ) } \leq \cdot \cdot \cdot \leq s _ { ( N ) }$ . For a target miscoverage level $\alpha \in ( 0 , 1 )$ , set

$$
q _ { \alpha } : = \lceil ( N + 1 ) ( 1 - \alpha ) \rceil , \qquad \eta _ { \alpha } : = s _ { ( q _ { \alpha } ) } ,\tag{4}
$$

with $s _ { ( N + 1 ) } = + \infty$ . For any new test instance z, the conformal uncertainty set is constructed as the level set

$$
\mathcal { U } _ { \alpha } ( z ) = \left\{ c \in \mathbb { R } ^ { n } : s ( z , c ) \leq \eta _ { \alpha } \right\} .\tag{5}
$$

Under the standard exchangeability assumption, the foundational result in conformal prediction yields a distributionfree finite-sample marginal coverage guarantee,

$$
\begin{array} { r } { \mathbb { P } \big ( c _ { \mathrm { t e s t } } \in \mathcal { U } _ { \alpha } ( z _ { \mathrm { t e s t } } ) \big ) \geq 1 - \alpha . } \end{array}
$$

The choice of score determines the geometry and downstream meaning of the resulting set. We next specialize the score to decision-aware optimization settings.

## 3.3 SPO+ Conformal Regret Set and Optimization

Existing conformal robust optimization methods typically calibrate generic parameter-space sets, such as norm balls, boxes, ellipsoids, and budgeted sets. Although these sets admit well-established deterministic reformulations, they often ignore downstream decision geometry. This mismatch motivates constructing uncertainty sets by directly calibrating decision regret. For the nominal PTO decision, the natural score is its realized regret, namely the exact SPO loss (Elmachtoub and Grigas 2022):

$$
\begin{array} { r } { s _ { \mathrm { S P O } } ( c , \hat { c } ) = c ^ { \top } x ^ { * } ( \hat { c } ) - c ^ { \top } x ^ { * } ( c ) . } \end{array}\tag{6}
$$

For any threshold $\eta \geq 0 .$ , define the SPO uncertainty set as

$$
\mathcal { U } _ { \mathrm { S P O } } ( \hat { c } ; \eta ) : = \left\{ c \in \mathbb { R } ^ { n } : s _ { \mathrm { S P O } } ( c , \hat { c } ) \leq \eta \right\} .\tag{7}
$$

By construction, every $c \in \mathcal { U } _ { \mathrm { S P O } } ( \hat { c } ; \eta )$ certifies that the nominal PTO decision incurs regret at most η. However, this fixed-decision certificate does not necessarily provide a useful uncertainty representation for selecting alternative prescriptions: as the following proposition shows, regret-robust refinement over the exact-SPO uncertainty sets is degenerate.

Proposition 1 (Degeneracy of exact-SPO refinement). $S u p $ pose that X contains at least two distinct points and that $\hat { x } : = x ^ { \ast } ( \hat { c } )$ is a vertex ofX. For everyfinite $\eta \geq 0 _ { \mathrm { { ; } } }$

$$
\operatorname* { s u p } _ { c \in \mathcal { U } _ { \mathrm { S P O } } ( \hat { c } ; \eta ) } \operatorname { R e g } ( \hat { x } , c ) \leq \eta ,
$$

whereas, for every $x \in \mathcal { X } \setminus \{ \hat { x } \}$

$$
\operatorname* { s u p } _ { c \in \mathcal { U } _ { \mathrm { S P O } } ( \hat { c } ; \eta ) } \operatorname { R e g } ( x , c ) = + \infty .
$$

Consequently, xˆ is the unique minimizer of the corresponding worst-case regret problem.

Proposition 1 shows that exact SPO is suitable for certifying the nominal PTO decision but cannot support nontrivial refinement: every alternative prescription has infinite worstcase regret over its sublevel set. An independently justified compact objective support can restore boundedness, but this boundedness is supplied externally rather than by the exact-SPO score. Detailed proofs and a comprehensive treatment of the theoretical extensions are provided in the appendix.

This motivates using the SPO+ function for robustification. Unlike the action-specific exact-SPO certificate, SPO+ retains comparator-wise information across alternative decisions. For a realized parameter c and prediction ${ \hat { c } } ,$ define

$$
\boldsymbol { s } _ { \mathrm { S P O + } } ( \boldsymbol { c } , \hat { \boldsymbol { c } } ) = \operatorname* { m a x } _ { \boldsymbol { x } \in \mathcal { X } } \{ ( \boldsymbol { c } - 2 \hat { \boldsymbol { c } } ) ^ { \top } \boldsymbol { x } \} + ( 2 \hat { \boldsymbol { c } } - \boldsymbol { c } ) ^ { \top } \boldsymbol { x } ^ { * } ( \boldsymbol { c } ) ,\tag{8}
$$

which pointwise bounds the exact regret (SPO function). Applying the split-conformal procedure in Section 3.2 to the SPO+ scores yields the threshold $\eta _ { \alpha }$ and the conformal regret set

$$
\mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta _ { \alpha } ) = \{ c \in \mathbb { R } ^ { n } : s _ { \mathrm { S P O } + } ( c , \hat { c } ) \leq \eta _ { \alpha } \} .\tag{9}
$$

As formalized below, this construction translates conformal validity into a finite-sample guarantee on the regret of the nominal PTO decision.

Theorem 1 (Finite-sample regret validity). Let $\hat { c } _ { \mathrm { t e s t } } : =$ $\hat { f } ( z _ { \mathrm { t e s t } } )$ . If the calibration and test observations are exchangeable and the predictor <sup>ˆ</sup>f is trained on data independent ofthe joint calibration–test sample, then

$$
\begin{array} { r } { \mathbb { P } \big \{ c _ { \mathrm { t e s t } } \in \mathcal { U } _ { S P O + } \big ( \widehat { c } _ { \mathrm { t e s t } } ; \eta _ { \alpha } \big ) \big \} \geq 1 - \alpha . } \end{array}\tag{10}
$$

Moreover, the nominal PTO decision satisfies

$$
\begin{array} { r } { \mathbb { P } \left( c _ { \mathrm { t e s t } } ^ { \top } x ^ { * } ( \hat { c } _ { \mathrm { t e s t } } ) - c _ { \mathrm { t e s t } } ^ { \top } x ^ { * } ( c _ { \mathrm { t e s t } } ) \leq \eta _ { \alpha } \right) \geq 1 - \alpha . } \end{array}\tag{11}
$$

Theorem 1 provides an interpretable operational certificate by bounding the regret of a prescribed action before deployment. In conventional robust optimization, conservatism is controlled through geometric parameters, such as uncertainty-set radii or budgets, whose implications for decision quality may be dificult to interpret. In contrast, our framework links the coverage level 1 − α directly to a regret threshold $\eta _ { \alpha } .$ , allowing practitioners to assess the trade-of between statistical reliability and tolerated suboptimality.

Downstream regret refinement. The conformal set can also be used to select a decision that is less exposed to adverse objective realizations than the nominal PTO prescription. For any $x \in \mathcal { X }$ and a generic threshold $\eta \geq 0$ , define its worstcase regret over this set as

$$
\rho _ { \eta } ( x ; \hat { c } ) : = \operatorname* { s u p } _ { c \in \mathcal { U } _ { S P O + } ( \hat { c } ; \eta ) } \mathrm { R e g } ( x , c ) .\tag{12}
$$

Given a nonempty compact admissible decision domain $\mathcal { D } \subseteq$ X , the corresponding regret-robust refinement is

$$
\boldsymbol { x } _ { \mathcal { D } } ^ { \mathrm { R R } } ( \hat { \boldsymbol { c } } ) \in \arg \operatorname* { m i n } _ { \boldsymbol { x } \in \mathcal { D } } \rho _ { \eta _ { \alpha } } ( \boldsymbol { x } ; \hat { \boldsymbol { c } } ) .\tag{13}
$$

Later, we consider the two principal choices $\mathcal { D } = \mathcal { V }$ , which restricts the refinement to feasible vertices, and $\mathcal { D } = \mathcal { X }$ which permits any feasible point.

## 4 Geometry and Regret Refinement

Fix a prediction cˆ and a threshold $\eta ~ \geq ~ 0 .$ , and write $\rho _ { \eta } ( x ) : = \rho _ { \eta } ( x ; \hat { c } )$ . Because the SPO+ conformal regret set is generally nonconvex, we first exploit its geometry rather than optimize over it monolithically. We first derive a vertexindexed polyhedral decomposition and establish finite worstcase regret on each piece. We then characterize and locally generate the nonempty pieces, with certified truncation based on predicted gaps. Finally, we develop margin-based certificates and computational formulations for regret refinement without requiring a complete vertex description.

## 4.1 Polyhedral Decomposition of Set Geometry

Within a region where a fixed vertex $v _ { k }$ is hindsight optimal, the SPO+ score is convex and piecewise linear in c, so its sublevel set is polyhedral. The union of these regional sets need not be convex. Under the main-text unique-optimizer assumption, this yields the following exact decomposition; the tie-safe extension is given in the appendix.

Theorem 2 (Polyhedral Decomposition). At any cost vector c with unique optimizer $v _ { k }$

$$
c \in \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta ) \quad \Longleftrightarrow \quad c \in P _ { k } ,
$$

where

$$
\begin{array} { c } { P _ { k } = \Big \{ c \in \mathbb { R } ^ { n } : c ^ { \top } ( v _ { j } - v _ { k } ) \leq \eta + 2 \hat { c } ^ { \top } ( v _ { j } - v _ { k } ) , } \\ { c ^ { \top } ( v _ { k } - v _ { j } ) \leq 0 , \forall j \in \{ 1 , \ldots , K \} \Big \} . } \end{array}\tag{14}
$$

Although $\textstyle \bigcup _ { k = 1 } ^ { K } P _ { k }$ additionally includes cost vectors with tied optimizers, it is the closure of the unique-optimizer portion of $\mathcal { U } _ { \mathrm { S P O + } } ( \hat { c } ; \eta )$ and, by continuity of regret, yields the same worst-case regret values. The second group of inequalities makes $v _ { k }$ optimal under c, while the first imposes the SPO+ threshold conditional on $v _ { k }$ being hindsight optimal. Hence, on $P _ { k }$ $\mathrm { R e g } ( x , c ) = c ^ { \top } ( x - v _ { k } )$ , and worst-case regret can be evaluated piecewise over the union.

An unbounded piece $P _ { k }$ can still induce finite worst-case regret. The following result characterizes its recession cone $\mathrm { r e c } ( P _ { k } )$ and the worst-case regret $\phi _ { k } ( x )$ ofa feasible decision x over that piece.

Proposition 2 (Finite regret over each piece). For every nonempty piece $P _ { k }$

$$
\operatorname { r e c } ( P _ { k } ) = \operatorname { s p a n } ( \mathcal { X } - \mathcal { X } ) ^ { \perp } .\tag{15}
$$

Consequently, for every $x \in { \mathcal { X } } ,$

$$
\phi _ { k } ( x ) : = \operatorname* { m a x } _ { c \in P _ { k } } c ^ { \top } ( x - v _ { k } )\tag{16}
$$

is finite and attained.

Proposition 2 shows that the ambient unboundedness of $P _ { k }$ is harmless for regret evaluation: its recession directions are decision invariant, so every active piece induces a finite and attained adversarial problem. Consequently, worst-case regret over the nonconvex set can be evaluated piecewise over the active pieces.

Let

$$
\mathcal { A } : = \{ k \in [ K ] : P _ { k } \neq \emptyset \}\tag{17}
$$

denote the active-piece index set. It follows that, for every $x \in { \mathcal { X } } .$

$$
\rho _ { \eta } ( x ) = \operatorname* { m a x } _ { k \in \mathcal { A } } \phi _ { k } ( x ) .\tag{18}
$$

The decomposition is exact but may still be computationally large: K can be exponential, and each $P _ { k }$ in (14) contains constraints indexed by all comparator vertices. The next subsection addresses the first issue through active-piece screening and local generation, while Section 4.3 addresses the second through a constraint-based fixed-piece problem reformulation (oracle).

## 4.2 Active-Piece Screening and Generation

We first address the number of potential pieces. For each $v _ { k } \in \mathcal V$ , define its predicted gap relative to $\hat { v } = x ^ { * } ( \hat { c } )$ as

$$
\Delta _ { k } : = \hat { c } ^ { \top } ( v _ { k } - \hat { v } ) \geq 0 .\tag{19}
$$

The following result gives an exact characterization of the nonempty polyhedral pieces.

Theorem 3 (Exact active-piece screening). For every vertex $v _ { k } \in \mathcal V$ and everyfinite threshold $\eta \geq 0 ,$

$$
P _ { k } \neq \emptyset \quad \Longleftrightarrow \quad \Delta _ { k } \leq \frac { \eta } { 2 } .\tag{20}
$$

Therefore,

$$
\mathcal { A } = \left\{ k \in [ K ] : \Delta _ { k } \leq \frac { \eta } { 2 } \right\} .\tag{21}
$$

The relevant complexity is determined by the number of $\eta / 2 .$ -near-optimal vertices rather than by $| \nu |$ . It may be much smaller when the nominal optimizer is well separated, although it can remain exponential in the worst case.

Screening alone would still require evaluating $\Delta _ { k }$ for every vertex. We next show that the active vertices can be generated locally. Let $\mathcal { G } _ { \mathcal { X } }$ denote the one-skeleton of $x ,$ , and for any $\tau \geq 0$ define

$$
\mathcal { V } _ { \tau } : = \left\{ v \in \mathcal { V } : \hat { c } ^ { \top } ( v - \hat { v } ) \leq \tau \right\} .\tag{22}
$$

Proposition 3 (Connectivity of predicted near-optimal vertices). For every $\tau \geq 0 ,$ , the subgraph of $\mathcal { G } _ { \mathcal { X } }$ induced by $\mathcal { V } _ { \tau }$ is connected. More precisely, for every $v \in \mathcal V _ { \tau }$ , there exists an edge path $\hat { v } = v ^ { \hat { 0 } } , v ^ { 1 } , \ldots , \bar { v } ^ { L } = v$ such that

$$
\hat { c } ^ { \top } v ^ { 0 } \leq \hat { c } ^ { \top } v ^ { 1 } \leq \cdot \cdot \cdot \leq \hat { c } ^ { \top } v ^ { L } ,\tag{23}
$$

and every vertex on the path belongs to $\mathcal { V } _ { \tau }$

Setting $\tau = \eta / 2$ , Theorem 3 and Proposition 3 yield a direct local generation procedure. Starting from vˆ, we enumerate adjacent vertices, retain those satisfying $\hat { c } ^ { \top } ( v - \hat { v } ) \leq \eta / 2$ and recursively expand every newly retained vertex. Because the induced active subgraph is connected, exhaustive traversal recovers every active piece without expanding an inactive vertex. We call this procedure a margin-truncated adjacency search.

Given an adjacency oracle, the search expands only active vertices while inspecting their incident edges. Its computational efort therefore scales with the explored active subgraph and the oracle cost, rather than with the full vertex set. A best-first search prioritizes vertices by $\Delta ( v ) : = \hat { c } ^ { \top } ( v - \hat { v } )$ thereby enumerating active vertices in nondecreasing predicted gap and enabling certified early termination.

This ordering also enables a certified early-stopping rule. For $0 \le \tau \le \eta / 2$ , define the truncated active set and its restricted regret adversary as

$$
\begin{array} { r } { \begin{array} { r l } & { \mathscr { A } _ { \tau } : = \{ k \in \mathcal { A } : \Delta _ { k } \leq \tau \} , } \\ & { \rho _ { \eta , \tau } ( x ) : = \displaystyle \operatorname* { m a x } _ { k \in \mathcal { A } _ { \tau } } \phi _ { k } ( x ) . } \end{array} } \end{array}\tag{24}
$$

Corollary 1 (Certified gap-ordered truncation). For every $x \in { \mathcal { X } } ,$ , the contribution ofthe omitted active pieces satisfies

$$
\operatorname* { m a x } _ { k \in \mathcal A \setminus \mathcal A _ { \tau } } \phi _ { k } ( x ) \leq B _ { \tau } ( x ) : = \big ( \eta + 2 \hat { c } ^ { \top } ( x - \hat { v } ) - 2 \tau \big ) _ { + } ,
$$

where the maximum over an empty index set is defined as zero.

Let $x _ { \tau } \in { }$ arg min $\mathbf { \Psi } _ { x \in \mathcal { D } } \rho _ { \eta , \tau } ( x ) , R _ { \tau } : = \rho _ { \eta , \tau } ( x _ { \tau } )$ , and let $\begin{array} { r } { R _ { \mathcal { D } } ^ { \star } : = \operatorname* { m i n } _ { x \in \mathcal { D } } \rho _ { \eta } ( x ) } \end{array}$ denote the full refinement value over the same decision domain. Then

$$
R _ { \tau } \leq R _ { D } ^ { \star } \leq \rho _ { \eta } ( x _ { \tau } ) \leq \operatorname* { m a x } \{ R _ { \tau } , B _ { \tau } ( x _ { \tau } ) \} .\tag{25}
$$

Consequently, $B _ { \tau } ( x _ { \tau } ) \leq R _ { \tau }$ certifies that $x _ { \tau }$ is optimal for the full refinement problem.

Corollary 1 turns best-first enumeration into a certified anytime procedure. At any threshold τ , it provides a lower bound $\bar { R } _ { \bar { \imath } }$ on the full optimum and an upper bound max $\{ R _ { \tau } , B _ { \tau } ( x _ { \tau } ) \}$ on the full-set regret of the current truncated solution. The search may terminate with a certificate of exact optimality when $B _ { \tau } ( x _ { \tau } ) \leq R _ { \tau }$ , or with an approximate certificate when the resulting certified optimality gap falls below a prescribed tolerance.

## 4.3 Certified Regret Refinement

The connectivity result also yields a locally computable margin certificate. Let $\operatorname { A d j } ( \hat { v } )$ denote the vertices adjacent to vˆ, and define

$$
\Gamma : = \operatorname* { m i n } _ { v \in \mathcal { V } \backslash \{ \hat { v } \} } \hat { c } ^ { \top } ( v - \hat { v } ) = \operatorname* { m i n } _ { v \in \mathrm { A d j } ( \hat { v } ) } \hat { c } ^ { \top } ( v - \hat { v } ) .\tag{26}
$$

The second equality follows from Proposition 3. Hence, Γ is available from the initial adjacency expansion and measures the predicted separation from the closest competing vertex.

We consider two instances of (13): the pure refinement, restricted to vertices, and the continuous refinement over the full feasible polytope:

$$
x ^ { \mathrm { p u r e } } \in \arg \operatorname* { m i n } _ { x \in \mathcal { V } } \rho _ { \eta } ( x ) , \quad R _ { \mathrm { p u r e } } : = \rho _ { \eta } ( x ^ { \mathrm { p u r e } } ) ,\tag{27}
$$

$$
x ^ { \mathrm { c o n t } } \in \arg \operatorname* { m i n } _ { x \in \mathcal { X } } \rho _ { \eta } ( x ) , ~ R _ { \mathrm { c o n t } } : = \rho _ { \eta } ( x ^ { \mathrm { c o n t } } ) .\tag{28}
$$

For $q \in \{ \mathrm { p u r e } , \mathrm { c o n t } \}$ , let $x _ { \mathrm { t e s t } } ^ { q }$ and $R _ { \mathrm { t e s t } } ^ { q }$ denote the corresponding refined decision and robust regret value at $\hat { c } _ { \mathrm { t e s t } }$

Theorem 4 (Margin-certified regret refinement). The two refinement problems attain their optima and satisfy

$$
0 \leq R _ { \mathrm { c o n t } } \leq R _ { \mathrm { p u r e } } \leq \rho _ { \eta } ( \hat { v } ) \leq ( \eta - 2 \Gamma ) _ { + } \leq \eta ,\tag{29}
$$

where $( a ) _ { + } : = \operatorname* { m a x } \{ a , 0 \}$ . Moreover, setting $\eta = \eta _ { \alpha }$ , under the exchangeability condition ofTheorem 1,

$$
\begin{array} { r l } & { \mathbb { P } \Big \{ \operatorname { R e g } ( x _ { \mathrm { t e s t } } ^ { q } , c _ { \mathrm { t e s t } } ) \leq R _ { \mathrm { t e s t } } ^ { q } } \\ & { \quad \leq ( \eta _ { \alpha } - 2 \Gamma _ { \mathrm { t e s t } } ) _ { + } , \quad \forall q \in \{ \mathrm { p u r e } , \mathrm { c o n t } \} \Big \} \geq 1 - \alpha . } \end{array}
$$

The deterministic ordering shows that both refinements weakly improve the nominal worst-case certificate, while the larger continuous decision domain cannot perform worse than the vertex-restricted domain. The predicted margin further tightens the calibrated threshold from η to $( \eta - 2 \Gamma ) _ { + }$ In particular,

$$
\eta \leq 2 \Gamma \quad \Longrightarrow \quad \rho _ { \eta } ( \hat { v } ) = 0 .\tag{30}
$$

Because Γ is available from the initial adjacency expansion, this condition provides an exact early exit: the nominal decision is already certified and refinement is unnecessary. Otherwise, $( \eta - 2 \Gamma ) .$ <sub>+</sub> quantifies the remaining worst-case certificate. With $\eta = \eta _ { \alpha } ,$ , the same bound controls the realized regret of both refinements with probability at least $1 - \alpha$

Computational Formulations. Continuous refinement is natural for fractional or splittable decisions, whereas pure refinement preserves vertex-valued prescriptions. Since $\nu \subseteq$ $\chi , R _ { \mathrm { c o n t } } \leq R _ { \mathrm { p u r e } } ;$ however, this worst-case ordering need not extend to realized mean or tail regret. Moreover, x<sup>pure</sup> need not be active, because active vertices index adversarial pieces rather than candidate decisions.

Suppose piece $P _ { k }$ has constraint matrix $H _ { k }$ and righthand-side vector $h _ { k } \mathrm { . }$ :

$$
P _ { k } = \{ c \in \mathbb { R } ^ { n } : H _ { k } c \leq h _ { k } \} .\tag{31}
$$

With nonnegative dual multipliers $y _ { k }$ , strong duality gives

$$
\phi _ { k } ( x ) = \operatorname* { m i n } _ { y _ { k } \geq 0 } \left\{ h _ { k } ^ { \top } y _ { k } : H _ { k } ^ { \top } y _ { k } = x - v _ { k } \right\} .\tag{32}
$$

Aggregating these dual representations over the active pieces, with a scalar upper bound t on worst-case regret, yields the following finite continuous master.

Proposition 4 (Finite continuous master). The continuous refinement value $R _ { \mathrm { c o n t } }$ equals the optimal value ofthe linear program

$$
\begin{array} { r } { \mathrm { s . t . } \quad H _ { k } ^ { \top } y _ { k } = x - v _ { k } , \qquad k \in \mathcal { A } , } \end{array}\tag{33}
$$

$$
h _ { k } ^ { \top } y _ { k } \leq t , \qquad k \in { \mathcal { A } } ,
$$

$$
y _ { k } \geq 0 , \qquad k \in \mathcal { A } .
$$

![](images/481a1660ad21b420114832f386cb531bc3aa77c212e0e3800eff75a6c8f88f7d.jpg)  
Figure 1: Geometry of residual and regret sets in cost coordinates $( c _ { 1 } , c _ { 2 } )$ . The panels show normal-fan regions, a residual ${ \boldsymbol { \mathbf { \ell } } } _ { \mathbf { \ell } } - { \boldsymbol { \ell } } _ { 2 }$ ball, an exact-regret set, and the active pieces of an SPO+ set.

Thus, once the active pieces are explicitly represented, continuous refinement reduces to a single linear program whose size depends on the active set.

When the explicit piece descriptions are unavailable or prohibitively large, the original constraint representation of $\begin{array} { r } { \dot { \mathcal { X } } = \{ x \in \mathbb { R } _ { + } ^ { n } : A x = b \} } \end{array}$ yields a fixed-piece oracle that avoids all comparator-vertex constraints.

Proposition 5 (Constraint-based fixed-piece oracle). For any fixed $\bar { x } \in \mathcal { X }$ and active vertex $v _ { k } , \ \phi _ { k } ( \bar { x } )$ equals the optimal value of

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { c , { \mu } , \lambda } } & { c ^ { \top } ( \bar { x } - v _ { k } ) } \\ { \mathrm { s . t . } } & { A ^ { \top } { \mu } \leq c , \quad \quad b ^ { \top } { \mu } = c ^ { \top } v _ { k } , } \\ & { A ^ { \top } \lambda \geq c - 2 \hat { c } , } \\ & { b ^ { \top } \lambda - ( c - 2 \hat { c } ) ^ { \top } v _ { k } \leq \eta , } \\ & { { \mu } , \lambda \in { \mathbb R } ^ { m } . } \end{array}\tag{34}
$$

Here $\mu$ and $\lambda$ are dual variables for the hindsight minimization and the maximization in the SPO+ score, certifying optimality of $v _ { k }$ under c and the SPO+ threshold, respectively. At a master incumbent $( { \bar { x } } , { \bar { t } } )$ , an optimizer $c _ { k } ^ { \star }$ of (34) yields the valid cut $t \geq c _ { k } ^ { \star \top } ( x - v _ { k } )$ whenever ${ c _ { k } ^ { \star } } ^ { \top } ( \bar { x } - v _ { k } ) > \bar { t } .$

Hence, Propositions 4 and 5 are complementary: explicit pieces yield a single LP, whereas the oracle supports constraint generation for either the continuous or vertexrestricted master directly from (A, b). Combined with screening and adjacency generation, they avoid monolithic optimization over the nonconvex conformal set.

## 5 Experiments

We evaluate the proposed framework using an illustrative example and contextual fractional-knapsack experiments, focusing on set geometry, finite-sample validity, decision performance, and certified active-piece generation.

## 5.1 Illustrative Geometry of Calibrated Sets

We consider a two-dimensional decision problem with feasible set V given by the eight vertices of a regular octagon and objective mi $\mathbf { n } _ { v \in \mathcal { V } } c ^ { \top } v$ after c is realized. In Figure 1, each point $c = ( c _ { 1 } , c _ { 2 } )$ denotes a possible ground-truth cost, while the black cross marks the prediction $\bar { \hat { c } } = ( - 1 , - 0 . 3 5 )$ , which selects the nominal action vˆ. From left to right, the panels show (i) the hindsight-optimal action for each c, partitioning cost space into normal-fan regions; (ii) the conventional residual ${ \boldsymbol { \mathbf { \ell } } } _ { \mathbf { \ell } } - { \boldsymbol { \ell } } _ { 2 }$ ball of radius 0.70 centered at cˆ; (iii) the costs for which vˆ incurs exact regret at most 1.10; and (iv) the conformal regret set with SPO+ score at the same illustrative threshold, with colors identifying its active normalfan pieces. The comparison illustrates both the piecewiseconvex structure of the SPO+ set and why only predicted near-optimal vertices need to be generated.

## 5.2 Experiments on the Fractional Knapsack Problem

Following Ho-Nguyen and Kılınç-Karzan (2022), we generate nonlinear context-dependent item values with multiplicative noise and use their negatives as minimization costs. Complete implementation details are provided in the appendix. Small instances permit exhaustive geometric validation, while larger instances evaluate decision performance and computational eficiency.

Active Pieces and Margin Early Exit We first use $n = 8$ instances, for which all vertices can be enumerated. Although the feasible polytope contains 320.3 vertices on average, the mean number of active pieces ranges from only 5.33 to 1.00 across the calibration levels; see Figure 2(a).

The adjacency procedure matches exhaustive enumeration for the active set and second-best margin in all context–level comparisons. $\mathrm { A t } \ \alpha = 0 . 1 0$ , all nominal certificates satisfy the margin bound within numerical tolerance, and 30.8% of contexts receive a zero certificate (see Theorem 4), allowing the robust master to be skipped. These results validate the exactness and local sparsity of the procedure.

Coverage of Calibrated Sets. Parameter coverage measures inclusion of the realized cost in the calibrated set; decision-regret coverage measures whether realized regret is bounded by the set-based certificate. Table 1 reports both metrics on the exact $n = 8$ test split, with the decisions for the “Regret set” obtained using continuous regret refinement (RR). Detailed definitions are given in Appendix B.6.

All four methods attain parameter coverage consistent with the nominal levels. The conventional sets nevertheless give essentially 100% decision-regret coverage, which ofers little information about decision risk because their full-set regret certificates are substantially wider: parameter inclusion implies decision-regret coverage, whereas the converse need not hold. By contrast, the regret set keeps the two coverage notions more closely aligned, providing a more decisioninformative description of uncertainty rather than achieving high coverage through conservative bounds. Complete fivelevel results and certificate scales are reported in appendix.

![](images/3cfb390b216571125463ed8c32845d0c00ec21cde129d94eb80aea024b941835.jpg)

![](images/57918cdb45ffce484c502f84b8672285adb479580f8150db0ea7381ea83baf4f.jpg)  
Figure 2: Exact geometry validation for $n = 8 .$ (a) Activepiece counts across calibration levels. (b) Nominal certificate versus the margin upper bound at $\alpha = 0 . 1 0 ;$ ; “Skip” denotes the fraction certified at zero without solving a robust master.

Table 1: Empirical parameter / decision-regret coverage.
<table><tr><td>Method</td><td> $\alpha = . 0 5$ </td><td> $\alpha = . 1 0$ </td><td> $\alpha = . 2 0$ </td></tr><tr><td>Regret set</td><td>0.96/0.97</td><td>0.91/0.930.83/0.86</td><td></td></tr><tr><td>Residual  $\mathbf { \nabla } \cdot \ell _ { 2 }$  set</td><td></td><td>0.94/1.000.89/1.000.80/1.00</td><td></td></tr><tr><td>Box set</td><td></td><td>0.94/1.000.90/1.00 0.80/1.00</td><td></td></tr><tr><td>KNN-Ellipsoid 0.94/1.00 0.87/1.00 0.78/1.00</td><td></td><td></td><td></td></tr></table>

Decision Performance against Baselines We next evaluate decision performance on larger instances with $n = 2 0$ and $\alpha = 0 . 1 0$ . We compare Nominal PTO with pure and continuous SPO+ regret refinement, as well as conformal Residual-ℓ , conformal contextual Box, and KNN–Ellipsoid Cost-RO. Here we follow each method’s intended objective: worst-case cost for RO and minimax regret for SPO+. Adapting the RO baselines to minimax regret goes beyond their standard formulations and can require more involved optimization. To isolate the role of the calibrated set, Appendix B.8 reports objective-matched comparisons, which retain the performance advantage of SPO+ refinement.

Continuous RR provides the strongest tail protection, reducing Q95 and $\mathrm { \bar { C V a R } _ { 0 . 9 5 } }$ by 9.1% and 13.1% relative to Nominal PTO, at the cost of higher mean regret. Pure RR changes only 10.0% of decisions, remains close to the nominal mean, and achieves the best average performance among all robust methods. These results suggest using $\mathrm { P T O }$ when average performance is the priority, Pure RR for selective intervention with limited disruption to nominal decisions, and Continuous RR when upper-tail protection is worth a higher mean regret. Additional experiments on real solar-grid data and sparse transportation LPs are reported in Appendix C.

Table 2: Decision performance at $\alpha = 0 . 1 0 ;$ ; lower regret is better. Q95 is the 0.95 quantile, CVaR95 is the average regret in the upper 5% tail, and Change is the fraction of decisions difering from nominal PTO.
<table><tr><td>Method</td><td>Mean</td><td>Q95</td><td>CVaR95</td><td>Change</td></tr><tr><td>Nominal PTO</td><td>7.605</td><td>38.664</td><td>60.557</td><td>0.000</td></tr><tr><td>Pure RR</td><td>7.763</td><td>38.956</td><td>61.814</td><td>0.100</td></tr><tr><td>Continuous RR</td><td>11.117</td><td>35.137</td><td>52.618</td><td>0.990</td></tr><tr><td>Residual.  $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$  RO</td><td>20.001</td><td>57.090</td><td>77.027</td><td>0.956</td></tr><tr><td>Box RO</td><td>8.824</td><td>40.673</td><td>64.430</td><td>0.284</td></tr><tr><td>KNN RO</td><td>624.111</td><td>1374.573</td><td>1682.087</td><td>1.000</td></tr></table>

![](images/e20b5273ace1446210d066009cefdceac8238449378824713efae90012671cc8.jpg)  
Figure 3: Certified gap-ordered generation on 500 test contexts. (a) Exact-certification rate. (b) Median and 0.90 quantile of the certified optimality gap normalized by η.

Certified Early Termination. We finally evaluate the practical efectiveness of the gap-ordered truncation certificate in Corollary 1 on the main $n = 2 0$ , α = 0.10 instances. On average, each feasible polytope has $1 . 1 6 \times 1 0 ^ { 6 }$ vertices, of which only 44.96 generate active pieces. Let M denote the number of generated active pieces. As shown in Figure 3, at $M = 2 5$ , exact optimality is already certified for 99.6% of Pure RR and 91.4% of Continuous RR instances, although only 56.2% of active sets have been fully enumerated. The rates increase to 99.8% and 98.4%, respectively, at $M = 5 0 .$ while both the median and 0.90 quantile of the normalized certified gap are already zero at $M = 2 5$ . Thus, the certificate is typically tight enough to establish global optimality well before complete active-set generation. Detailed budget-wise results and runtimes are reported in Appendix B.5.

## 6 Conclusion

We introduced Conformal Regret Sets, a decision-focused framework that calibrates SPO+ scores to control downstream regret rather than residuals. Split-conformal calibration provides a finite-sample marginal guarantee for the nominal PTO regret, while the induced nonconvex set admits a normal-fan-aligned polyhedral decomposition with exact active-piece screening, local generation, certified truncation, and margin-based certificates. These properties support certified pure and continuous regret-robust refinement without monolithic optimization. Fractional-knapsack experiments validate the theory: pure refinement preserves nearnominal average performance, continuous refinement improves upper-tail protection, and gap-ordered generation often certifies optimality before the full active set is recovered.

## References

Angelopoulos, A. N.; and Bates, S. 2023. Conformal Prediction: A Gentle Introduction. Found. Trends Mach. Learn., 16(4): 494–591.

Angelopoulos, A. N.; Bates, S.; Fisch, A.; Lei, L.; and Schuster, T. 2024. Conformal Risk Control. In The Twelfth International Conference on Learning Representations.

Berden, S.; Mahmutoğulları, A. İ.; Tsouros, D.; and Guns, T. 2025. Solver-Free Decision-Focused Learning for Linear Optimization Problems. In Advances in Neural Information Processing Systems.

Bertsimas, D.; and Kallus, N. 2020. From predictive to prescriptive analytics. Management Science, 66(3): 1025– 1044.

Bertsimas, D.; and Tsitsiklis, J. 1998. Introduction to Linear Optimization. Athena Scientific.

Chan, T.; Delage, E.; and Lin, B. 2024. Conformal inverse optimization for adherence-aware prescriptive analytics. Available at SSRN 4970102.

Chen, S.; Zhou, W.; and Zhu, S. 2026. Learning Polyhedral Conformal Sets for Robust Optimization. arXiv preprint arXiv:2605.08506.

Chenreddy, A. R.; and Delage, E. 2024. End-to-end conditional robust optimization. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, 736– 748.

Cortes-Gomez, S.; Patiño, C.; Byun, Y.; Wu, Z. S.; Horvitz, E.; and Wilder, B. 2025. Utility-Directed Conformal Prediction: A Decision-Aware Framework for Actionable Uncertainty Quantification. In International Conference on Learning Representations.

Donti, P. L.; Amos, B.; and Kolter, J. Z. 2017. Task-based End-to-end Model Learning in Stochastic Optimization. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Elmachtoub, A. N.; and Grigas, P. 2022. Smart “Predict, then Optimize”. Management Science, 68(1): 9–26.

Gupta, V.; and Huang, M. 2024. Decision-focused learning with directional gradients. Advances in Neural Information Processing Systems, 37: 79194–79220.

Ho-Nguyen, N.; and Kılınç-Karzan, F. 2022. Risk Guarantees for End-to-End Prediction and Optimization Processes. Management Science, 68(12): 8680–8698.

Hu, Y.; Tan, J.; Zou, C.; Bao, Y.; and Ren, H. 2026. Conformal Robustness Control: A New Strategy for Robust Decision. In The Fourteenth International Conference on Learning Representations.

Im, H.; Benslimane, W.; and Grigas, P. 2025. Smart Surrogate Losses for Contextual Stochastic Linear Optimization with Robust Constraints. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Johnstone, C.; and Cox, B. 2021. Conformal uncertainty sets for robust optimization. In Carlsson, L.; Luo, Z.; Cherubin, G.; and An Nguyen, K., eds., Proceedings of the Tenth Symposium on Conformal and Probabilistic Prediction and Applications, volume 152 of Proceedings ofMachine Learning Research, 72–90. PMLR.

Liu, H.; and Grigas, P. 2021. Risk Bounds and Calibration for a Smart Predict-then-Optimize Method. In Advances in Neural Information Processing Systems, volume 34, 22083– 22094.

Mandi, J.; Bucarey, V.; Tchomba, M. M. K.; and Guns, T. 2022. Decision-focused learning: Through the lens of learning to rank. In International conference on machine learning, 14935–14947. PMLR.

Patel, Y. P.; Rayan, S.; and Tewari, A. 2024. Conformal contextual robust optimization. In International Conference on Artificial Intelligence and Statistics, 2485–2493. PMLR.

Qi, M.; Grigas, P.; and Shen, Z.-J. M. 2025. Integrated Conditional Estimation-Optimization. Operations Research.

Smith, J. E.; and Winkler, R. L. 2006. The optimizer’s curse: Skepticism and postdecision surprise in decision analysis. Management Science, 52(3): 311–322.

Sun, C.; Liu, L.; and Li, X. 2023. Predict-then-calibrate: A new perspective of robust contextual LP. Advances in Neural Information Processing Systems, 36: 17713–17741.

Wilder, B.; Dilkina, B.; and Tambe, M. 2019. Melding the data-decisions pipeline: decision-focused learning for combinatorial optimization. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and Thirty-First Innovative Applications ofArtificial Intelligence Conference and Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’19/IAAI’19/EAAI’19. AAAI Press. ISBN 978-1-57735-809-1.

Yeh, C.; Christianson, N.; Wu, A.; Wierman, A.; and Yue, Y. 2025. End-to-End Conformal Calibration for Optimization Under Uncertainty. Transactions on Machine Learning Research.

Zhang, S.; Zhou, H.; Zheng, J.; Jiang, G.; Cheng, B.; Lin, W.; and Chen, G. 2025. Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization: Bridging Observational and Experimental Data. In Advances in Neural Information Processing Systems.

Zhao, L.; Jiang, H.; and Qi, W. 2026. Conformal Robustness in Prediction-Driven Decision-Making. Available at SSRN 5338354.

Zhou, W.; Orfanoudaki, A.; and Zhu, S. 2026. Conformalized Decision Risk Assessment. In The Fourteenth International Conference on Learning Representations.

Zhou, W.; and Zhu, S. 2026. Calibrating decision robustness via inverse conformal risk control. In Forty-third International Conference on Machine Learning.

Zhou, W.; Zhu, S.; Qiu, F.; and Wu, X. 2025. Hierarchical Spatio-Temporal Uncertainty Quantification for Distributed Energy Adoption. In 2025 IEEE Power & Energy Society General Meeting (PESGM), 1–5.

## A Proofs and extended analysis

Unless stated otherwise, the geometric arguments below fix a prediction ${ \hat { c } } ,$ a finite threshold $\eta \geq 0 ,$ , and a nonempty compact polytope $\mathcal { X } = \mathrm { c o n v } ( \mathcal { V } )$ with complete vertex set $\mathcal { V } = \bar { \{ v _ { 1 } , . . . , \bar { v _ { K } } \} }$ . We write $[ \acute { K } ] : = \{ 1 , \ldots \acute { , } K \}$ , let $\hat { v } : =$ $x ^ { * } ( \hat { c } )$ , and let $\hat { k } \in [ K ]$ satisfy $\boldsymbol { v } _ { \hat { k } } = \boldsymbol { \hat { v } }$ . We use $\operatorname { A d j } ( v )$ for the vertices adjacent to v in the one-skeleton of X. In proofs concerning (14), η denotes the generic threshold fixed in the main paper’s geometry section; it is specialized to $\eta _ { \alpha }$ for the calibrated set. All dimensions below follow the main paper: costs and decisions lie in $\mathbb { R } ^ { n }$ . Throughout the Main Paper subsection, the main text’s unique-optimizer convention is in force, and we abbreviate $\rho _ { \eta } ( x ; \hat { c } ) \operatorname { a s } \rho _ { \eta } ( x )$ . Full optimizer-tie statements are deferred to the Extended Analysis subsection; tie-boundary closures are used below only to justify the exact worst-case values.

## A.1 Main Paper

## Exact-SPO degeneracy and SPO+ dominance.

ProofofProposition 1. The nominal bound is immediate from the definition of the exact-SPO sublevel set. For every $c \in \mathcal { U } _ { \mathrm { S P O } } ( \hat { c } ; \eta )$

$$
\mathrm { R e g } ( \hat { x } , c ) = c ^ { \top } \hat { x } - \operatorname* { m i n } _ { u \in \mathcal { X } } c ^ { \top } u = s _ { \mathrm { S P O } } ( c , \hat { c } ) \leq \eta .
$$

Taking the supremum proves the nominal bound in the proposition.

It remains to show that every alternative decision has unbounded regret. Because xˆ is a vertex of the polyhedron $x ,$ it is an exposed point: there exists $d \in \mathbb { R } ^ { n }$ such that

$$
\begin{array} { r } { d ^ { \top } ( x - \hat { x } ) > 0 \qquad \mathrm { f o r e v e r y } x \in \mathcal { X } \setminus \{ \hat { x } \} . } \end{array}\tag{35}
$$

For completeness, choose d in the relative interior of the minimization normal cone of xˆ. The face–normal-cone correspondence for polyhedra states that the minimizers exposed by any vector in this relative interior form exactly the face associated with that normal cone. Since the normal cone is the one associated with the vertex ${ \hat { x } } ,$ the exposed face is the singleton {xˆ}, which proves (35).

For $t > 0$ , set $c _ { t } : = t d .$ . Equation (35) makes xˆ the unique optimizer under $c _ { t } .$ , and hence

$$
s _ { \mathrm { S P O } } ( c _ { t } , \hat { c } ) = c _ { t } ^ { \top } \hat { x } - \operatorname* { m i n } _ { u \in \mathcal { X } } c _ { t } ^ { \top } u = 0 .
$$

Thus $c _ { t } \in \mathcal { U } _ { \mathrm { S P O } } ( \hat { c } ; \eta )$ for every $t > 0$ . For any fixed $x \neq { \hat { x } } ,$

$$
\mathrm { R e g } ( x , c _ { t } ) = t d ^ { \top } ( x - \hat { x } ) \longrightarrow + \infty .
$$

This proves the asserted unboundedness. The robust value at xˆ is finite, whereas it is infinite at every other feasible decision, so xˆ is the unique robust minimizer. □

We next record explicitly the pointwise inequality used by the conformal argument. Let $v : = x ^ { * } ( c )$ and $\hat { v } : \dot { = } x ^ { \ast } ( \hat { c } ) \dot { }$ Their optimality gives

$$
c ^ { \top } ( \hat { v } - v ) \geq 0 , \qquad \hat { c } ^ { \top } ( \hat { v } - v ) \leq 0 .\tag{36}
$$

Selecting $x \ = \ v$ in the maximum defining $s _ { \mathrm { S P O + } }$ gives $s _ { \mathrm { S P O } + } ( c , \hat { c } ) \geq 0$ . Selecting $x = \hat { v }$ instead gives

$$
\begin{array} { r l } & { s _ { \mathrm { S P O } + } ( c , \hat { c } ) \geq ( c - 2 \hat { c } ) ^ { \top } ( \hat { v } - v ) } \\ & { \qquad = \mathrm { R e g } ( \hat { v } , c ) - 2 \hat { c } ^ { \top } ( \hat { v } - v ) } \\ & { \qquad \geq \mathrm { R e g } ( \hat { v } , c ) = s _ { \mathrm { S P O } } ( c , \hat { c } ) . } \end{array}\tag{37}
$$

The last inequality has the stated direction because the second inner product in (36) is nonpositive.

ProofofTheorem 1. We use the standard split-sample interpretation of the theorem: the training data, and hence the fitted predictor, are independent of the joint calibration–test tuple. Condition on those training data. The fitted predictor and all deterministic measurable optimization maps are then fixed functions, while the calibration and test observations remain exchangeable. Applying the same SPO+ score map to them therefore produces exchangeable scores $S _ { 1 } , \ldots , \bar { S _ { N } } , S _ { N + 1 }$ Independence from the calibration sample alone would not sufice if the fitted predictor were allowed to depend on the test observation.

First suppose these $N + 1$ scores are distinct and $q _ { \alpha } \leq N$ The rank R of $S _ { N + 1 }$ among all $N + 1$ scores is uniform on $\{ 1 , \ldots , N + 1 \}$ . Consequently,

$$
\mathbb { P } \{ R \leq q _ { \alpha } \} = \frac { q _ { \alpha } } { N + 1 } \geq 1 - \alpha .
$$

If $R \leq q _ { \alpha }$ , fewer than $q _ { \alpha }$ calibration scores are smaller than $S _ { N + 1 } , \mathrm { s o } \ S _ { N + 1 } \le S _ { ( q _ { \alpha } ) } = \eta _ { \alpha }$ . Hence

$$
\mathbb { P } \{ S _ { N + 1 } \leq \eta _ { \alpha } \} \geq 1 - \alpha ,
$$

which is exactly (10). When $q _ { \alpha } = N + 1$ , the convention $\eta _ { \alpha } = + \infty$ makes the inclusion event certain. This last branch is a valid but vacuous statistical statement: all recession, screening, and finite-refinement results below require $\eta _ { \alpha } <$ $+ \infty$ . Calibration-score ties are handled without a continuity assumption in the extended analysis below.

On the same score-inclusion event, (37) yields

$$
\mathrm { R e g } ( x ^ { * } ( \hat { c } _ { \mathrm { t e s t } } ) , c _ { \mathrm { t e s t } } ) \leq s _ { \mathrm { S P O } + } ( c _ { \mathrm { t e s t } } , \hat { c } _ { \mathrm { t e s t } } ) \leq \eta _ { \alpha } .
$$

The event on the left therefore has probability at least $1 - \alpha ,$ proving the second assertion. □

Polyhedral decomposition under the main-paper convention.

Proof of Theorem 2. Consider first a cost c having the unique optimizer $v _ { k }$ . Since $\mathcal { X } = \mathrm { c o n v } ( \mathcal { V } )$ , a linear objective attains the same maximum over $\mathcal { X }$ and over V. Thus

$$
s _ { \mathrm { S P O + } } ( c , \hat { c } ) = \operatorname* { m a x } _ { 1 \leq j \leq K } ( c - 2 \hat { c } ) ^ { \top } ( v _ { j } - v _ { k } ) .\tag{38}
$$

The score is at most η if and only if, for every $j ,$

$$
c ^ { \top } ( v _ { j } - v _ { k } ) \leq \eta + 2 \hat { c } ^ { \top } ( v _ { j } - v _ { k } ) .
$$

Meanwhile, optimality of $v _ { k }$ under the minimization objec tive is equivalent to

$$
c ^ { \top } ( v _ { k } - v _ { j } ) \leq 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } j .
$$

These are precisely the two families of inequalities defining $P _ { k }$ in (14). Hence, at every cost with a unique optimizer,

$$
c \in \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta ) \quad \Longleftrightarrow \quad c \in \bigcup _ { k = 1 } ^ { K } P _ { k } .
$$

This proves the pointwise tie-free decomposition.

We next justify the use of the closed pieces in the subsequent worst-case regret values. Let $\mathcal { U } _ { \eta } ^ { \mathrm { u n i } }$ denote the subset of $\mathcal { U } _ { \mathrm { S P O + } } ( \hat { c } ; \eta )$ consisting of costs whose linear objective has a unique minimizer over X. Under the main paper’s uniquenominal-optimizer assumption,

$$
\mathrm { { c l } } \left( { \mathcal { U } } _ { \eta } ^ { \mathrm { { u n i } } } \right) = \bigcup _ { k = 1 } ^ { K } P _ { k } .\tag{39}
$$

The inclusion from left to right follows from the pointwise equivalence above and the fact that a finite union of closed polyhedra is closed. For the reverse inclusion, fix a nonempty $P _ { k }$ and write $\Delta _ { j } : = \hat { c } ^ { \top } ( v _ { j } - \hat { v } )$ . Using $\boldsymbol { v } _ { j } = \boldsymbol { \hat { v } }$ in the two families defining $P _ { k }$ gives

$$
0 \leq c ^ { \top } ( \hat { v } - v _ { k } ) \leq \eta - 2 \Delta _ { k } \qquad ( c \in P _ { k } ) ,
$$

so $\Delta _ { k } \leq \eta / 2 .$

Suppose first that $\eta > 0$ and $\Delta _ { k } < \eta / 2$ . Choose an exposing cost $d _ { k }$ for which $v _ { k }$ is the unique minimizer, so $\bar { d } _ { k } ^ { \top } ( v _ { j } - v _ { k } ) > 0$ for every $j \neq k$ . The right-hand side of each upper piece inequality satisfies

$$
\eta + 2 \hat { c } ^ { \top } ( v _ { j } - v _ { k } ) = \eta + 2 ( \Delta _ { j } - \Delta _ { k } ) \geq \eta - 2 \Delta _ { k } > 0 .
$$

Consequently, for all suficiently small $\varepsilon > 0 , \varepsilon d _ { k } \in P _ { k }$ , and all its lower piece inequalities are strict for $j \neq$ k. For any $c \in P _ { k }$ and $t \in ( 0 , 1 ]$ , convexity of $P _ { k }$ then gives

$$
c _ { t } : = ( 1 - t ) c + t \varepsilon d _ { k } \in P _ { k } ,
$$

with $v _ { k }$ uniquely optimal under $c _ { t }$ . The pointwise equivalence already proved implies $c _ { t } \in \mathcal { U } _ { n } ^ { \mathrm { u n i } }$ , and $c _ { t }  c \mathrm { a s } t \downarrow 0 .$ If $\eta > 0$ and $\Delta _ { k } = \eta / 2$ , the displayed two-sided bound forces $c ^ { \top } ( \hat { v } - v _ { k } ) = 0$ throughout $P _ { k }$ . Thus vˆ is also optimal under every $c \in P _ { k }$ . Moreover, for every j,

$$
\begin{array} { r l } & { c ^ { \top } ( v _ { j } - \hat { v } ) = c ^ { \top } ( v _ { j } - v _ { k } ) } \\ & { \qquad \leq \eta + 2 \hat { c } ^ { \top } ( v _ { j } - v _ { k } ) = 2 \hat { c } ^ { \top } ( v _ { j } - \hat { v } ) } \\ & { \qquad \leq \eta + 2 \hat { c } ^ { \top } ( v _ { j } - \hat { v } ) , } \end{array}
$$

so $P _ { k } \subseteq P _ { \hat { k } }$ . Since $\Delta _ { \hat { k } } = 0 < \eta / 2$ , the preceding argument applies to $\ddot { P } _ { \hat { k } }$

Finally, if $\eta = 0 .$ , nonemptiness forces $\Delta _ { k } = 0$ . Uniqueness of vˆ under cˆ then implies $k = { \hat { k } } .$ . The cost cˆ itself belongs to $P _ { \hat { k } }$ and uniquely exposes $\hat { v } ,$ so mixing any $\boldsymbol { c } \in P _ { \boldsymbol { \hat { k } } }$ with a positive amount of cˆ again produces points of $\mathcal { U } _ { \eta } ^ { \mathrm { u n i } }$ converging to c. This proves (39).

For any deterministic vertex-valued extension of $x ^ { * } ( c )$ to tie costs, the same inequality calculation used above gives

$$
\mathcal { U } _ { \eta } ^ { \mathrm { u n i } } \subseteq \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta ) \subseteq \bigcup _ { k = 1 } ^ { K } P _ { k } .
$$

Thus all three sets have the same closure. The regret $c ^ { \top } x -$ min $\mathfrak { u } \in \mathcal { X } ^ { c ^ { \top } }$ u is continuous in c, so taking its supremum over any of these sets gives the same value. Finally, every $c \in P _ { k }$ makes $v _ { k }$ optimal and hence satisfies $\mathrm { R e g } ( \dot { x } , c ) = \dot { c } ^ { \top } ( x -$ $v _ { k } )$ . It follows that

$$
\rho _ { \eta } ( x ) = \operatorname* { m a x } _ { k \in \mathcal { A } } \phi _ { k } ( x ) ,
$$

which therefore justifies (18). The unclosed sets themselves can difer on optimizer-tie boundaries; the precise extension is given in Proposition 6. □

Recession directions, screening, and active-piece generation.

Proof of Proposition 2. Fix a nonempty piece $P _ { k }$ and abbreviate

$$
a _ { j } : = v _ { j } - v _ { k } , \qquad b _ { j } : = \eta + 2 \hat { c } ^ { \top } ( v _ { j } - v _ { k } ) .
$$

The two families of inequalities in (14) can be written as

$$
P _ { k } = \{ c : - a _ { j } ^ { \top } c \leq 0 , a _ { j } ^ { \top } c \leq b _ { j } , j = 1 , \ldots , K \} .\tag{40}
$$

Because $P _ { k }$ is nonempty, d is a recession direction if and only if it satisfies the homogeneous version of every row in (40). Thus

$$
- a _ { j } ^ { \top } d \leq 0 \quad \mathrm { a n d } \quad a _ { j } ^ { \top } d \leq 0 \qquad \mathrm { f o r e v e r y } \ j ,
$$

which is equivalent to

$$
d ^ { \top } ( v _ { j } - v _ { k } ) = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } j .\tag{41}
$$

We next verify the required span identity. The inclusion

$$
\mathrm { s p a n } \{ v _ { j } - v _ { k } : j \in [ K ] \} \subseteq \mathrm { s p a n } ( \mathcal { X } - \mathcal { X } )
$$

is immediate. Conversely, write any $x , y \in \mathcal { X } \mathrm { ~ a s ~ } x \mathrm { ~ = ~ }$ $\sum _ { j } \lambda _ { j } v _ { j }$ and $\begin{array} { r } { y = \sum _ { j } \mu _ { j } v _ { j } } \end{array}$ , where both coeficient vectors are nonnegative and sum to one. Then

$$
x - y = \sum _ { j } ( \lambda _ { j } - \mu _ { j } ) ( v _ { j } - v _ { k } ) ,
$$

because $\begin{array} { r } { \sum _ { j } ( \lambda _ { j } - \mu _ { j } ) = 0 } \end{array}$ . Hence

$$
\operatorname { s p a n } \{ v _ { j } - v _ { k } : j \in [ K ] \} = \operatorname { s p a n } ( \mathcal { X } - \mathcal { X } ) .
$$

Combining this identity with (41) proves (15).

Now fix $x \in \mathcal { X }$ and set $\boldsymbol { q } : = \boldsymbol { x } - \boldsymbol { v } _ { k }$ . Then $q \in \operatorname { s p a n } ( \mathcal { X } -$ $\mathcal { X } ) , \mathrm { { s o } } q ^ { \top } d = 0$ for every d $\in \mathrm { r e c } ( P _ { k } )$ . To establish finiteness and attainment rather than only the absence of an improving ray, take a finite-generator representation

$$
P _ { k } = { \mathrm { c o n v } } \{ p _ { 1 } , \ldots , p _ { r } \} + { \mathrm { c o n e } } \{ r _ { 1 } , \ldots , r _ { s } \} + L _ { k } ,
$$

where $r \geq 1 ,$ , every $r _ { \ell }$ is a recession direction, and the lineality space $L _ { k }$ is contained in rec ${ \bf \nabla } ( P _ { k } )$ . For arbitrary $c \in P _ { k }$

$$
c = \sum _ { i } \theta _ { i } p _ { i } + \sum _ { \ell } \gamma _ { \ell } r _ { \ell } + \ell .
$$

Here $\theta _ { i } \geq 0 , \sum _ { i } \theta _ { i } = 1 , \gamma _ { \ell } \geq 0 \quad$ , and $\ell \in L _ { k }$ . Orthogonality to the recession cone yields

$$
q ^ { \top } c = \sum _ { i } \theta _ { i } q ^ { \top } p _ { i } \leq \operatorname* { m a x } _ { 1 \leq i \leq r } q ^ { \top } p _ { i } .
$$

The upper bound is finite and is attained by a maximizing generator $p _ { i } \in P _ { k }$ . This proves that $\phi _ { k } ( x )$ in (16) is finite and attained. The same representation also gives

$$
\phi _ { k } ( x ) = \operatorname* { m a x } _ { 1 \leq i \leq r } p _ { i } ^ { \top } ( x - v _ { k } ) ,\tag{42}
$$

so $\phi _ { k }$ is a continuous polyhedral convex function on $\mathcal { X } . \quad \sqsubset$

ProofofTheorem 3. Suppose first that $P _ { k }$ is nonempty and choose $c \in P _ { k }$ . Use $\boldsymbol { v } _ { j } = \boldsymbol { \hat { v } }$ in the lower and upper inequalities defining the piece. Since $\Delta _ { k } = \hat { c } ^ { \top } ( v _ { k } - \hat { v } )$

$$
\begin{array} { r l } & { 0 \leq c ^ { \top } ( \hat { v } - v _ { k } ) } \\ & { \quad \leq \eta + 2 \hat { c } ^ { \top } ( \hat { v } - v _ { k } ) = \eta - 2 \Delta _ { k } . } \end{array}\tag{43}
$$

The two endpoints imply $\Delta _ { k } \leq \eta / 2$

Conversely, suppose $\Delta _ { k } \leq \eta / 2$ and set $c = 0$ . Nominal optimality of vˆ gives $\hat { c } ^ { \top } ( v _ { j } - \hat { v } ) \ge 0$ for every j. Therefore

$$
\begin{array} { c } { \eta + 2 \hat { c } ^ { \top } ( v _ { j } - v _ { k } ) = \eta + 2 \hat { c } ^ { \top } ( v _ { j } - \hat { v } ) - 2 \Delta _ { k } } \\ { \geq \eta - 2 \Delta _ { k } \geq 0 . } \end{array}\tag{44}
$$

$\mathbf { A } \mathbf { t } c = 0 $ , both $c ^ { \top } ( v _ { j } - v _ { k } )$ and $c ^ { \top } ( v _ { k } - v _ { j } )$ are zero, and (44) verifies every remaining upper bound. Hence $0 \in P _ { k } .$ proving the reverse implication and (20). In particular, $P _ { \hat { k } }$ is always nonempty because $\Delta _ { \hat { k } } = 0$ . This argument concerns the closed algebraic pieces $P _ { k } . \mathrm { A t } \Delta _ { k } = \eta / 2$ , the zero witness can lie entirely on an optimizer-tie boundary; Proposition 6 explains when these pieces are exact score pieces and when they form a conservative outer representation.

ProofofProposition 3. At a vertex v, the tangent cone of a polytope is generated by the directions of its incident edges:

$$
T _ { \mathcal { X } } ( v ) = \mathrm { c o n e } \{ u - v : u \in \mathrm { A d j } ( v ) \} .
$$

Suppose $\boldsymbol { v } \neq \boldsymbol { \hat { v } }$ and every adjacent vertex u satisfied $\hat { c } ^ { \top } ( u -$ $v ) \geq 0 ,$ . Every vector $x - v , x \in \mathcal { X }$ , belongs to $T _ { \mathcal { X } } ( v )$ , so the displayed conic representation would imply $\hat { c } ^ { \top } ( x - v ) \ge 0$ for every $x \in \mathcal { X }$ . Then v would be a global minimizer of ${ \hat { c } } ^ { \top } x ,$ contradicting uniqueness of $\hat { v } .$ Thus every v $\neq \hat { v }$ has an adjacent vertex with strictly smaller predicted objective.

Starting at any vertex $v ,$ repeatedly take such a decreasing edge. Because V is finite and the objective decreases strictly at each step, the path terminates; its terminal vertex has no decreasing neighbor and therefore must be the unique minimizer vˆ. Reversing the path gives

$$
\hat { v } = v ^ { 0 } , v ^ { 1 } , \ldots , v ^ { L } = v , \qquad \hat { c } ^ { \top } v ^ { 0 } \leq \cdots \leq \hat { c } ^ { \top } v ^ { L } .
$$

If $v \in \mathcal V _ { \tau }$ , every preceding vertex on this path has predicted objective no larger than that of v, and hence also belongs to $\mathcal { V } _ { \tau }$ . The induced subgraph is therefore connected, proving (23).

The same path proves the local-margin identity (26) when $| \nu | \geq 2$ . Indeed, because $\operatorname { A d j } ( { \hat { v } } ) \subseteq { \mathcal { V } } \setminus \{ { \hat { v } } \}$

$$
\Gamma \leq \operatorname* { m i n } _ { u \in \mathrm { A d j } ( \hat { v } ) } \hat { c } ^ { \top } ( u - \hat { v } ) .
$$

Conversely, the first step $v ^ { 1 }$ on a monotone path from vˆ to any $\boldsymbol { v } \neq \boldsymbol { \hat { v } }$ is adjacent to vˆ and satisfies $\hat { c } ^ { \top } ( v ^ { 1 } - \hat { v } ) \leq \hat { c } ^ { \top } ( v - \hat { v } ) \Big )$ Minimizing over $\boldsymbol { v } \neq \boldsymbol { \hat { v } }$ gives the reverse inequality. □

## Certified truncation.

ProofofCorollary 1. For the minimum notation in the statement to be literal, the decision domain D is assumed nonempty and compact; this includes both V and X. More generally, the displayed argument is conditional on the stated minimizers existing. Fix ${ \bar { x } } \in { \mathcal { X } }$ and write $\begin{array} { r } { x = \sum _ { j } \lambda _ { j } v _ { j } } \end{array}$ with $\lambda _ { j } \ \geq \ 0$ and $\textstyle \sum _ { j } \lambda _ { j } = 1$ . For $c \in P _ { k }$ , summing the upper piece inequalities with these weights gives

$$
\begin{array} { c } { { c ^ { \top } ( x - v _ { k } ) \leq \eta + 2 \hat { c } ^ { \top } ( x - v _ { k } ) } } \\ { { = \eta + 2 \hat { c } ^ { \top } ( x - \hat { v } ) - 2 \Delta _ { k } . } } \end{array}\tag{45}
$$

The lower piece inequalities make v<sub>k</sub> optimal under c, so the left-hand side is nonnegative. Taking the maximum over $P _ { k }$ and then taking the positive part in (45) yields

$$
\begin{array} { r } { \phi _ { k } ( x ) \leq \left( \eta + 2 \hat { c } ^ { \top } ( x - \hat { v } ) - 2 \Delta _ { k } \right) _ { + } . } \end{array}
$$

Every omitted active piece has $\Delta _ { k } > \tau$ . Monotonicity of the positive-part map therefore gives

$$
\operatorname* { m a x } _ { k \in \mathcal A \backslash \mathcal A _ { \tau } } \phi _ { k } ( x ) \leq \left( \eta + 2 \hat { c } ^ { \top } ( x - \hat { v } ) - 2 \tau \right) _ { + } = B _ { \tau } ( x ) .
$$

If the omitted set is empty, its contribution is defined as zero, and the same inequality holds.

Because $\mathcal { A } _ { \tau } \subseteq \mathcal { A } , \rho _ { \eta , \tau } ( x ) \leq \rho _ { \eta } ( x )$ for every x. Consequently,

$$
R _ { \tau } \leq R _ { \mathcal { D } } ^ { \star } \leq \rho _ { \eta } ( x _ { \tau } ) .
$$

At $x _ { \tau }$ , the retained pieces contribute $\rho _ { \eta , \tau } ( x _ { \tau } ) = R _ { \tau }$ , while all omitted pieces are bounded by $B _ { \tau } ( \dot { x _ { \tau } } )$ . Hence

$$
\rho _ { \eta } ( x _ { \tau } ) \leq \operatorname* { m a x } \{ R _ { \tau } , B _ { \tau } ( x _ { \tau } ) \} ,
$$

which proves the full chain. If $B _ { \tau } ( x _ { \tau } ) \leq R _ { \tau }$ , then

$$
R _ { \tau } \leq R _ { D } ^ { \star } \leq \rho _ { \eta } ( x _ { \tau } ) \leq R _ { \tau } .
$$

Thus $x _ { \tau }$ is optimal for the full problem. Since $\mathbf { \mathcal { A } } _ { \tau }$ uses the inclusive condition $\Delta _ { k } \le \tau$ , a certified gap-ordered implementation must exhaust the entire gap plateau at the cutof; stopping in the middle of a tied plateau does not produce the set $\boldsymbol { A } _ { \ u { \tau } }$ used in this proof. This identification with $\rho _ { \eta }$ uses the main-text unique-optimizer convention; Proposition 6 gives the corresponding tie-safe statement. □

## Margin certification and finite formulations.

Proof of Theorem 4. For every active k and every $c \in P _ { k } .$ the lower piece inequalities make $v _ { k }$ optimal under $c .$ Therefore, for all $x \in \mathcal { X }$

$$
c ^ { \top } ( x - v _ { k } ) = \operatorname { R e g } ( x , c ) \geq 0 .
$$

Taking the attained maximum over $P _ { k }$ and then the finite maximum over $k \in { \mathcal { A } }$ shows that $\rho _ { \eta } ( x ) \geq 0$

Equation (42) shows that every ϕ is continuous. Hence $\rho _ { \eta } = \operatorname* { m a x } _ { k \in \mathcal { A } } \phi _ { k }$ is continuous as a finite maximum. The pure domain V is finite and the continuous domain $\mathcal { X }$ is compact, so both refinement problems attain their minima. Since $\hat { v } \in \mathcal { V } \subseteq \mathcal { X }$ , minimizing the same function over these nested domains gives

$$
0 \leq R _ { \mathrm { c o n t } } \leq R _ { \mathrm { p u r e } } \leq \rho _ { \eta } ( \hat { v } ) .\tag{46}
$$

We now bound the last term piece by piece. The nominal piece contributes

$$
\phi _ { \hat { k } } ( \hat { v } ) = \operatorname* { m a x } _ { c \in P _ { \hat { k } } } c ^ { \top } ( \hat { v } - \hat { v } ) = 0 .
$$

For a nonnominal active vertex $v _ { k }$ , use $\boldsymbol { v } _ { j } = \boldsymbol { \hat { v } }$ in the upper piece inequality. For every $c \in P _ { k }$

$$
\begin{array} { r l } & { 0 \leq c ^ { \top } ( \hat { v } - v _ { k } ) = \mathrm { R e g } ( \hat { v } , c ) } \\ & { \quad \leq \eta + 2 \hat { c } ^ { \top } ( \hat { v } - v _ { k } ) = \eta - 2 \Delta _ { k } \leq \eta - 2 \Gamma . } \end{array}\tag{47}
$$

The last comparison is used only for $v _ { k } \neq \hat { v } ;$ it is not valid for the nominal index. Taking the maximum of the zero nominal contribution and the nonnominal bounds proves

$$
\rho _ { \eta } ( \hat { v } ) \leq ( \eta - 2 \Gamma ) _ { + } \leq \eta .
$$

Together with (46), this proves (29). The singleton convention needed when $| \nu | = { \bar { 1 } }$ is given in the extended analysis.

The equality boundary is included. If $\eta = 2 \Gamma < + \infty$ every active nonnominal vertex satisfies $\dot { \Delta } _ { k } \geq \Gamma$ and, by Theorem $3 , \Delta _ { k } \leq \eta / 2 = \Gamma$ . Hence equality holds throughout and (47) forces every nonnominal contribution at vˆ to be zero. The nominal contribution is also zero, so $\rho _ { \eta } ( \hat { v } ) = 0$ and all terms preceding it in (29) are zero.

Finally set $\eta ~ = ~ \eta _ { \alpha } ~ < ~ + \infty$ and consider the conformal inclusion event from Theorem 1. On this event, $c _ { \mathrm { t e s t } } \in \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } _ { \mathrm { t e s t } } ; \eta _ { \alpha } )$ . Therefore, for $q \in \{ \mathrm { p u r e } , \mathrm { c o n t } \}$ ,

$$
\begin{array} { r l r } & { } & { \mathrm { R e g } ( x _ { \mathrm { t e s t } } ^ { q } , c _ { \mathrm { t e s t } } ) \le \rho _ { \eta _ { \alpha } } ( x _ { \mathrm { t e s t } } ^ { q } ) = R _ { \mathrm { t e s t } } ^ { q } } \\ & { } & { \qquad \le ( \eta _ { \alpha } - 2 \Gamma _ { \mathrm { t e s t } } ) _ { + } . \qquad } \end{array}
$$

Both choices of $\mid q$ are controlled on the same inclusion event, whose probability is at least $1 - \alpha$ □

Proof of Proposition $^ { 4 . }$ Fix an active piece $P _ { k } = \{ c :$ $H _ { k } c \le h _ { k } \}$ and a decision $x \in \mathcal { X }$ . The support problem defining $\phi _ { k } ( x )$ is

$$
\operatorname* { m a x } _ { c } \{ ( x - v _ { k } ) ^ { \top } c : H _ { k } c \leq h _ { k } \} .
$$

It is feasible because $P _ { k }$ is active, and it has a finite optimum attained by Proposition 2. Its LP dual is

$$
\operatorname* { m i n } _ { y _ { k } \geq 0 } \{ h _ { k } ^ { \top } y _ { k } : H _ { k } ^ { \top } y _ { k } = x - v _ { k } \} .
$$

The equality has the displayed sign because the primal objective coeficient of c is $x - v _ { k }$ . Strong duality gives equality of the two attained optimal values and proves (32).

For fixed x and t, weak duality shows that the existence of $y _ { k } \geq 0$ satisfying

$$
H _ { k } ^ { \top } y _ { k } = x - v _ { k } , \qquad h _ { k } ^ { \top } y _ { k } \le t
$$

implies $\phi _ { k } ( x ) \leq t .$ . Conversely, $\mathrm { i f } \phi _ { k } ( x ) \leq t .$ , an optimal dual solution exists and satisfies these two relations. Imposing this equivalence for every $k \in { \mathcal { A } }$ gives exactly the epigraph

$$
t \geq \operatorname* { m a x } _ { k \in \mathcal { A } } \phi _ { k } ( x ) = \rho _ { \eta } ( x )
$$

over $x \in \mathcal { X }$ . Minimizing t therefore yields $R _ { \mathrm { c o n t } }$ , proving (33). □

ProofofProposition 5. Fix ${ \bar { x } } \in { \mathcal { X } }$ and $v _ { k } \in \mathcal V$ . For a candidate cost $c ,$ consider first the standard-form minimization problem

$$
\operatorname* { m i n } _ { z \geq 0 } \{ c ^ { \top } z : A z = b \} .
$$

Its dual is

$$
\operatorname* { m a x } _ { \mu \in \mathbb { R } ^ { m } } \{ b ^ { \top } \mu : A ^ { \top } \mu \leq c \} .
$$

Because $v _ { k }$ is primal feasible, weak duality implies that the existence of $\mathbf {  { \mu } } _ {  { \mu } }$ satisfying

$$
A ^ { \top } \mu \leq c , \qquad b ^ { \top } \mu = c ^ { \top } v _ { k }
$$

certifies optimality of $v _ { k }$ under c. Conversely, if $v _ { k }$ is optimal, LP strong duality supplies such a dual-optimal $\mu .$

Next set $d : = c - 2 \hat { c }$ . Conditional on $v _ { k }$ being hindsightoptimal, the SPO+ threshold is

$$
\operatorname* { m a x } _ { z \in \mathcal { X } } d ^ { \top } ( z - v _ { k } ) \leq \eta .\tag{48}
$$

The dual of max $\{ d ^ { \top } z : A z = b , z \geq 0 \}$ is

$$
\operatorname* { m i n } _ { \lambda \in \mathbb { R } ^ { m } } \{ b ^ { \top } \lambda : A ^ { \top } \lambda \geq d \} .
$$

Nonemptiness and compactness of X make the primal optimum finite and attained, so strong duality shows that (48) holds if and only if there is $\boldsymbol { \lambda } \in \mathbb { R } ^ { m }$ such that

$$
A ^ { \top } \lambda \geq c - 2 \hat { c } , \qquad b ^ { \top } \lambda - ( c - 2 \hat { c } ) ^ { \top } v _ { k } \leq \eta .
$$

The $\mu$ constraints therefore describe exactly the costs for which v is optimal, while the λ constraints describe exactly the SPO+ threshold for that same vertex. Their projection onto the c-coordinates is precisely $P _ { k }$ . The objective in (34) $\mathrm { i s } c ^ { \top } ( \bar { x } - v _ { k } )$ , so maximizing over the extended formulation has the same value as maximizing over $c \in P _ { k } ,$ , namely $\phi _ { k } ( \bar { x } )$ . An optimal c exists by Proposition $^ { 2 , }$ and strong duality supplies corresponding $\mu$ and λ, so the oracle also attains this value. □

## A.2 Extended Analysis

Calibration-score ties and the augmented threshold. The split-conformal guarantee does not require a continuous score distribution. Conditional on training data that are independent of the joint calibration–test tuple, let $S _ { 1 } , \ldots , S _ { N } , \dot { S } _ { N + 1 }$ be the exchangeable scores. Introduce independent $U _ { 1 } , \ L _ { \cdot } \ L _ { \cdot } , U _ { N + 1 } \sim \mathrm { U n i f } ( 0 , 1 )$ and rank $( S _ { i } , U _ { i } )$ lexicographically. These pairs are exchangeable and almost surely distinct, so the rank R of the test pair is uniform on $\{ 1 , \ldots , N + 1 \}$ }. For $q _ { \alpha } \leq N$

$$
\mathbb { P } \{ R \leq q _ { \alpha } \} = \frac { q _ { \alpha } } { N + 1 } \geq 1 - \alpha .
$$

If $S _ { N + 1 } > S _ { ( q _ { \alpha } ) }$ , at least $q _ { \alpha }$ calibration scores are strictly smaller than $S _ { N + 1 }$ . Their pairs all precede the test pair, so $R > q _ { \alpha }$ . Taking the contrapositive gives

$$
\{ R \leq q _ { \alpha } \} \subseteq \{ S _ { N + 1 } \leq S _ { ( q _ { \alpha } ) } \} .
$$

Thus the inclusive conformal event has probability at least $1 - \alpha$ , including when calibration scores are tied.

If $q _ { \alpha } ~ = ~ N + 1$ , the augmented threshold is +∞ and coverage is one. This is only a vacuous statistical branch. Even when X is compact and every SPO+ score is finite, its +∞-sublevel set is all of $\mathbb { R } ^ { n } ;$ the upper piece inequalities no longer restrict recession directions, and worst-case regret can be infinite. Accordingly, all screening, margin, and finite master results are invoked only when $\eta _ { \alpha } < + \infty$

## Optimizer ties and the closed piecewise adversary.

Proposition 6 (Tie-safe SPO+ representation and certificates). Retain the main paper’s unique nominal optimizer vˆ, but allow realized costs to have optimizer ties and fix any deterministic measurable vertex-valued hindsight selector. For this extension only, define

$$
\overline { { \mathcal { U } } } _ { \eta } : = \bigcup _ { k = 1 } ^ { K } P _ { k } , \qquad \overline { { \rho } } _ { \eta } ( \boldsymbol { x } ) : = \underset { k \in \mathcal { A } } { \operatorname* { m a x } } \underset { c \in P _ { k } } { \operatorname* { m a x } } c ^ { \top } ( \boldsymbol { x } - \boldsymbol { v } _ { k } ) .
$$

For finite η,

$$
\mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta ) \subseteq \overline { { \mathcal { U } } } _ { \eta } ,\tag{49}
$$

the two sets agree at every cost having a unique optimizer, and

$$
\operatorname { c l } ( \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta ) ) = \overline { { \mathcal { U } } } _ { \eta } .\tag{50}
$$

Moreover, for every $x \in { \mathcal { X } } ,$

$$
0 \leq \rho _ { \eta } ( x ) = \overline { { \rho } } _ { \eta } ( x ) < + \infty .\tag{51}
$$

Thus passing to the closed union does not change any worstcase regret value, although it can add boundary costs and ensure that the piecewise adversarial maximum is attained. In particular, every additional cost in $\overline { { \mathcal { U } } } _ { \eta } \setminus \mathcal { U } _ { \mathrm { S P O + } } ( \hat { c } ; \eta )$ lies on an optimizer-tie boundary.

Define instead a compatible selector as follows: among all vertices minimizing $\bar { c } ^ { \top } v ,$ , first minimize $\hat { c } ^ { \top } v ,$ and break any remaining tie by a fixed deterministic vertex order. If the $S P O +$ score and its calibration scores use this same rule, then

$$
\mathcal { U } _ { \mathrm { S P O + } } ( \hat { c } ; \eta ) = \overline { { \mathcal { U } } } _ { \eta } , \qquad \rho _ { \eta } ( x ) = \overline { { \rho } } _ { \eta } ( x ) \quad f o r e \nu e r y \ x \in \mathcal { X } .\tag{52}
$$

Proof. Take $c \in \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta )$ and let $v _ { k } = x ^ { * } ( c )$ be the vertex returned by the fixed selector. Its optimality gives $c ^ { \top } ( v _ { k } - v _ { j } ) \leq 0$ for every j. The score inequality is equivalent to

$$
( c - 2 { \hat { c } } ) ^ { \top } ( v _ { j } - v _ { k } ) \leq \eta \qquad { \mathrm { f o r ~ e v e r y ~ } } j ,
$$

which is the upper family defining $P _ { k }$ . Hence $c \in P _ { k }$ , proving (49). If the optimizer under c is unique, any $c \in P _ { k }$ must use that same $v _ { k }$ in the score, so the upper piece inequalities also prove the reverse inclusion.

The selector-specific set contains $\mathcal { U } _ { \eta } ^ { \mathrm { u n i } }$ and is contained in ${ \overline { { \mathcal { U } } } } _ { \eta } .$ Taking closures and applying (39) proves (50).

For any $c \in P _ { k }$ , the lower inequalities make $v _ { k }$ optimal even if other vertices are tied. Regret is invariant to which optimizer is used, and therefore

$$
\mathrm { R e g } ( x , c ) = c ^ { \top } ( x - v _ { k } ) \qquad \mathrm { f o r ~ e v e r y ~ } x \in \mathcal { X } .\tag{53}
$$

The regret is continuous in $c ,$ so (50), (53), and Proposition 2 give (51).

It remains to prove exactness of the compatible selector. For a fixed $c ,$ all its optimal vertices share the same value $\begin{array} { r } { m _ { c } : = \operatorname* { m i n } _ { j } c ^ { \top } v _ { j } } \end{array}$ . Let

$$
M ( c ) : = \operatorname* { m a x } _ { j } ( c - 2 \hat { c } ) ^ { \top } v _ { j } .
$$

If $v$ is any c-optimal vertex, the SPO+ score obtained by using v is

$$
M ( c ) - ( c - 2 \hat { c } ) ^ { \top } v = M ( c ) - m _ { c } + 2 \hat { c } ^ { \top } v .\tag{54}
$$

The compatible selector minimizes the final term among all c-optimal vertices and therefore gives the smallest possible tied-optimizer score. Now take $c \in P _ { k }$ . The vertex $v _ { k }$ is $c -$ optimal and its upper piece inequalities say that its score is at most η. By (54), the compatible selector has a score no larger than this one, so $c \in \mathcal { U } _ { \mathrm { S P O } + } ( \hat { c } ; \eta )$ . This proves the reverse inclusion and hence the set equality in (52). The equality of robust values then follows from (53). □

For completeness, Proposition 6 also completes the arguments concerning attainment and probability statements under an arbitrary selector. By (51), the selector-specific $\rho _ { \eta }$ is the finite maximum of the continuous functions $\phi _ { k }$ , and hence is continuous. It therefore attains a minimum on compact X, while attainment over finite V is immediate. Furthermore,

$$
\rho _ { \eta } ( \hat { v } ) = \overline { { \rho } } _ { \eta } ( \hat { v } ) \leq ( \eta - 2 \Gamma ) _ { + } .
$$

The nested-domain ordering in Theorem 4 therefore holds for the selector-specific refinements as well. On the conformal event, the realized cost belongs to the selector-specific score set by definition, so its regret is bounded directly by the corresponding $\rho _ { \eta } .$ . Under any deterministic selector, the piecewise representations, truncation bounds, and continuous master compute the exact worst-case value. A closedpiece maximizer may nevertheless be merely a limit point of the selector-specific set; the compatible selector additionally makes the underlying set equality and closed-piece adversarial attainment exact.

Nominal ties, equality boundaries, and singleton polytopes. Continue to use the closed-union notation introduced above. Suppose now that vˆ is any fixed vertex selected from a non-singleton optimal face under cˆ. Then $\Delta _ { k } \geq 0$ still holds for every vertex, and all vertices of the nominal optimal face have zero gap. The graph of a face of a polytope is connected. From vˆ, first traverse zero-gap edges inside the nominal face as follows. For a target vertex $v ,$ apply the decreasing-edge argument from the proof of Proposition 3 until it terminates at some optimal vertex w. The nominal face contains both vˆ and $w ,$ , and its one-skeleton gives a zero-gap path from vˆ to w. Concatenating this path with the reversal of the descent from v to w constructs a nondecreasing path from the selected vˆ to v. Hence the induced sublevel graph remains connected without nominal uniqueness.

If the nominal face contains another vertex, the first edge of a path to that vertex has zero gap. Consequently,

$$
\Gamma = \operatorname* { m i n } _ { v \in \mathcal { V } \backslash \{ \hat { v } \} } \hat { c } ^ { \top } ( v - \hat { v } ) = \operatorname* { m i n } _ { v \in \mathrm { A d j } ( \hat { v } ) } \hat { c } ^ { \top } ( v - \hat { v } ) = 0 .
$$

The margin certificate remains valid but reduces to the unstrengthened bound $\overline { { \rho } } _ { \eta } ( \hat { v } ) \leq \eta$

The inclusive screening boundary is also substantive. If $\Delta _ { k } = \eta / 2$ , the witness $c = 0$ keeps $P _ { k }$ nonempty, while (43) forces $c ^ { \top } ( \hat { v } - v _ { k } ) = 0$ throughout the piece. Such a piece may be lower-dimensional. For example, with $\chi = [ 0 , 1 ] ,$ $\hat { c } \doteq 1 , \hat { v } = 0 , \eta = 2$ , and $v _ { k } = 1$ , its constraints reduce to $0 \leq - c \leq 0 ,$ so $P _ { k } = \{ 0 \}$ . Likewise, at $\eta = 2 \Gamma <$ $+ \infty .$ closest nonnominal pieces are active but (47) forces their regret at vˆ to be zero. This is why the zero-certificate condition is inclusive.

$\operatorname { I f } | \mathcal { V } | = 1$ , define $\Gamma : = + \infty$ and, for finite $\eta , ( \eta - 2 \Gamma ) _ { + } : =$ 0. There is no nonnominal or adjacent vertex, so the localneighbor formula is not invoked. Every feasible decision equals $\hat { v } ,$ every regret and refinement value is zero, and all margin conclusions hold under this convention. If a pure or continuous refinement has multiple minimizers in a nonsingleton problem, every minimizer has the same certificate value; a deterministic measurable tie rule is needed only to make the selected decision a well-defined random variable.

Context-dependent feasible sets and nonpositive costs. The rank argument remains valid when the feasible polytope varies across observations, provided the full observations

$$
( Z _ { i } , A _ { i } , b _ { i } , C _ { i } ) , \qquad i = 1 , \ldots , N + 1 ,
$$

are exchangeable, the training data are independent of this joint tuple, and every realized $\mathcal { X } _ { i } = \{ x : A _ { i } x = b _ { i } , x \ge 0 \}$ is nonempty and compact. Assume also that the same jointly measurable optimization and tie-selection maps are used to construct every calibration and test score. Conditional on training, the resulting scores are then exchangeable. All geometric proofs are applied pointwise to each realized polytope. This formulation includes instance-specific prices and capacities in the fractional-knapsack experiments after the usual slack-variable standard-form lifting.

Corollary 2 (Validity with the nonpositive-cost support). Assume in addition that $C \in \mathbb { R } _ { - } ^ { n }$ almost surely and replace every closed piece by $P _ { k } \cap \mathbb { R } _ { - } ^ { n }$ . For finite $\eta ,$ exact screening of these supported pieces, finiteness and attainment of piecewise regret, the truncation bound, and the margin certificate remain valid. On the SPO+ conformal event, the realized regret is bounded by the corresponding supported closed-union certificate with probability at least $1 - \alpha$

Proof. The screening necessity argument is unchanged because a supported piece is a subset of $P _ { k }$ . For suficiency, the witness $c = 0$ used in (44) also belongs to $\mathbb { R } _ { - } ^ { n }$ , so

$$
P _ { k } \cap \mathbb { R } _ { - } ^ { n } \neq \emptyset \quad \Longleftrightarrow \quad \Delta _ { k } \leq \eta / 2 .
$$

For every nonempty intersection of these polyhedra,

$$
\begin{array} { r } { \mathrm { r e c } ( P _ { k } \cap \mathbb { R } _ { - } ^ { n } ) = \mathrm { r e c } ( P _ { k } ) \cap \mathbb { R } _ { - } ^ { n } \subseteq \mathrm { s p a n } ( \mathcal { X } - \mathcal { X } ) ^ { \perp } . } \end{array}
$$

Thus every regret direction $x - v _ { k }$ remains orthogonal to the recession cone, and the finite-generator proof of finiteness and attainment applies unchanged. Intersecting a piece can only decrease its support value, so all truncation and margin upper bounds remain valid.

On the conformal score event, Proposition 6 places $C$ in a closed piece. The almost-sure sign condition also gives $C \in \mathbb { R } ^ { n }$ , so C belongs to the corresponding supported piece and its realized regret is bounded by the supported closedunion maximum. The event has probability at least $1 - \alpha$ In the experiments this support condition follows from the construction $C = - V$ with $\bar { V } \geq 0$ . In the fixed-piece oracle, the same intersection is implemented simply by adding $c \leq$ 0; all other duality constraints are unchanged. □

## B Experiment Details

## B.1 Data-Generating Process and Split Protocol

All fractional-knapsack experiments use the feasible region

$$
{ \mathcal { X } } ( p , b ) = \{ x \in [ 0 , 1 ] ^ { n } : p ^ { \top } x \leq b \} ,
$$

where $x _ { j }$ is the fraction of item $j$ selected, $p _ { j }$ is its price, and $b$ is the budget. We use a global seed of 42 and deterministic child random streams for the signal matrix and observations. We report one fixed split rather than averages over multiple top-level seeds.

For context dimension $d _ { z }$ , a shared signal matrix $\Theta \ \in$ $\mathbb { R } _ { + } ^ { n \times d _ { z } }$ is generated once for each experimental configuration. Its final two columns are zero. For the remaining columns, we generate a signal-inclusion indicator $D _ { j \ell }$ and a coeficient magnitude $A _ { j \ell }$ and set

$$
\begin{array} { l } { { \Theta _ { j \ell } = D _ { j \ell } A _ { j \ell } , } } \\ { { D _ { j \ell } \sim \mathrm { B e r n o u l l i } ( 0 . 5 ) , } } \\ { { A _ { j \ell } \sim \mathrm { U n i f } [ 0 . 8 , 1 . 2 ] . } } \end{array}
$$

The same $\Theta$ is used for the training, calibration, and test observations. The context $Z ,$ , conditional mean item value $\mu _ { j } ( Z )$ , realized value $V _ { j }$ , and multiplicative noise $E _ { j }$ satisfy

$$
\begin{array} { r l r } & { } & { Z \sim \mathrm { U n i f } ( [ 0 , 4 ] ^ { d _ { z } } ) , \qquad \mu _ { j } ( Z ) = ( Z ^ { \top } \Theta _ { j \cdot } ) ^ { 2 } , } \\ & { } & \\ & { } & { V _ { j } = \mu _ { j } ( Z ) E _ { j } , \qquad E _ { j } \overset { \mathrm { i i d } } { \sim } \mathrm { U n i f } [ 0 . 8 , 1 . 2 ] . } \end{array}
$$

The optimization models use the minimization-cost convention $\dot { C } = - V$

Each item price is sampled independently and uniformly from the integers 100 through 1000. Let $S { \dot { = } } \sum _ { j } p _ { j }$ be the total price and $M = \operatorname* { m a x } _ { j } p _ { j }$ the largest item price. With independent $U _ { 1 } , U _ { 2 } \sim \mathrm { U n i f } [ 0 , 1 ]$ , define an upper budget limit <sup>¯</sup>b and sample b as follows:

$$
\begin{array} { l } { { \bar { b } = \operatorname* { m i n } \{ S - 1 0 ^ { - 9 } , } } \\ { { \operatorname* { m a x } \{ M + 1 0 ^ { - 9 } , S - U _ { 1 } M \} \} , } } \\ { { b = M + U _ { 2 } ( \bar { b } - M ) . } } \end{array}
$$

A single deterministic observation stream is generated and then partitioned into disjoint training, calibration, and test subsets. All predictive models are trained on values. Predictions are transformed back to the original value scale and clipped below at zero before conversion to costs. No finite lower bound inferred from the synthetic value distribution is supplied to an uncertainty set.

## B.2 Calibration and Evaluation Definitions

For calibration scores $S _ { 1 } , \ldots , S _ { n _ { \mathrm { c a l } } }$ and miscoverage level $\alpha ,$ the split-conformal threshold is the exact order statistic

$$
k _ { \alpha } = \lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) \rceil , \qquad \eta _ { \alpha } = S _ { ( k _ { \alpha } ) } .
$$

Thus, we do not substitute an interpolated empirical quantile for the conformal order statistic.

Let $\begin{array} { r } { v ( C ) \in \arg \operatorname* { m i n } _ { u \in \mathcal { X } ( p , b ) } C ^ { \top } u } \end{array}$ . The SPO+ calibration score is

$$
s _ { + } ( C , \widehat { C } ) = \operatorname* { m a x } _ { u \in \mathcal { X } ( p , b ) } ( C - 2 \widehat { C } ) ^ { \top } \{ u - v ( C ) \} .
$$

Nominal, hindsight, and SPO+ score problems are solved as linear programs. Optimal-solution ties use a deterministic secondary objective so that the returned solution is a reproducible polytope vertex.

Reported regret uses the equivalent value convention

$$
R ( x ; V , p , b ) = \operatorname* { m a x } _ { u \in \mathcal { X } ( p , b ) } V ^ { \top } u - V ^ { \top } x .
$$

For test regrets $r _ { 1 } , \ldots , r _ { n _ { \mathrm { t e s t } } }$ , the performance summaries are the arithmetic mean, the empirical 0.95 quantile, and

$$
\mathrm { C V a R } _ { 0 . 9 5 } = \operatorname* { m i n } _ { t \in \mathbb { R } } \left\{ t + \frac { 1 } { 0 . 0 5 n _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } ( r _ { i } - t ) _ { + } \right\} ,
$$

which assigns exactly $5 \%$ empirical probability mass to the upper tail, using fractional weight at a tied boundary when necessary. A decision is recorded as changed when its $\ell _ { \infty }$ distance from the nominal PTO decision exceeds $1 0 ^ { - 6 }$

## B.3 Illustrative Objective-Space Geometry

The geometry figure is illustrative and is not constructed from a calibration sample. Its action set contains the eight unit-circle points

$$
v _ { k } = ( \cos ( k \pi / 4 ) , \sin ( k \pi / 4 ) ) , \qquad k = 0 , \dots , 7 .
$$

The predicted cost is $\widehat { C } = ( - 1 , - 0 . 3 5 )$ . Scores are evaluated on a 241×241 grid over $[ - 2 . 7 5 , 2 . 7 5 ] ^ { 2 }$ , using the illustrative values $\eta = 1 . 1 0$ and residual radius 0.70. The four panels show the closed normal-fan regions, the residual set $\{ C :$ $\Vert C - \widehat { C } \Vert _ { 2 } \leq 0 . 7 0 \}$ , the exact-regret sublevel set, and the SPO+ sublevel set. In the last panel, a normal-fan piece is active precisely when

$$
\widehat { C } ^ { \top } ( v _ { k } - \widehat { v } ) \leq \eta / 2 .
$$

These numerical values serve only to visualize the diferent objective-space geometries.

## B.4 Active Pieces and Margin Early Exit

The exact-validation configuration is

$$
\begin{array} { r } { n = 8 , \quad \quad d _ { z } = 8 , \quad } \\ { n _ { \mathrm { t r a i n } } = 2 0 0 0 , \quad n _ { \mathrm { c a l } } = 1 0 0 0 , \quad } \\ { n _ { \mathrm { t e s t } } = 5 0 0 , \quad \quad } \end{array}
$$

with $\alpha \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 , 0 . 4 0 \}$ . The predictor is kernel ridge regression with an RBF kernel, regularization 0.1, and kernel parameter 0.03. Both features and multi-output targets are standardized using training data only.

Exact Enumeration Audit For every test context, all vertices of the fractional-knapsack polytope are explicitly enumerated. The complete active set obtained by enumeration is compared with the adjacency-based search. The local margin

$$
\Gamma = \operatorname* { m i n } _ { v \in \mathrm { A d j } ( \widehat { v } ) } \widehat { C } ^ { \top } ( v - \widehat { v } )
$$

is also compared with complete enumeration. No activepiece cap is imposed in this experiment.

The histogram uses complete active-piece counts. The margin diagnostic uses $\alpha ~ = ~ 0 . 1 0$ , places $( \eta \mathrm { ~ - ~ } 2 \Gamma ) _ { + }$ on the horizontal axis and the nominal set-relative certificate on the vertical axis, and records the frequency with which the margin rule accepts the nominal decision without solving a robust master.

## B.5 Certified Anytime Active-Piece Generation

We empirically examine the certified early-stopping rule in Corollary 1 under the main $n = 2 0 , \alpha = 0 . 1 0$ setting. The predictor, calibration threshold, and 500 test contexts are held fixed. The feasible polytopes contain $1 . 1 5 9 \times 1 0 ^ { 6 }$ vertices on average, while the complete active sets contain only 44.956; the full local traversal evaluates 380.472 distinct vertices on average. For each context, we enumerate active vertices in nondecreasing predicted gap and consider requested prefix budgets

$$
M \in \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 \} .
$$

If the Mth vertex lies on a predicted-gap plateau, the prefix is expanded through the entire plateau. Thus the retained set is exactly $\mathcal { A } _ { \tau } = \{ \bar { k } : \Delta _ { k } \leq \tau \}$ for the recorded boundary τ ; we denote its size by $M _ { \mathrm { e f f } }$ . This tie-safe convention altered only two of the 500 contexts at one grid budget each, changing $M _ { \mathrm { e f f } }$ from 10 to 11 and from 25 to 27.

For each tie-safe prefix and each decision domain, we solve the restricted refinement and record

$$
\begin{array} { l l } { { R _ { \tau } = \rho _ { \eta , \tau } ( x _ { \tau } ) , } } & { { B _ { \tau } = \left( \eta + 2 \widehat { C } ^ { \top } ( x _ { \tau } - \widehat { v } ) - 2 \tau \right) _ { + } , } } \\ { { U _ { \tau } = \operatorname* { m a x } \{ R _ { \tau } , B _ { \tau } \} . } } & { { } } \end{array}
$$

We declare exact certification only when $B _ { \tau } \ \leq \ R _ { \tau }$ <sub>τ</sub> up to the prespecified scale-aware tolerance, or when the complete active set has been exhausted. The reported optimality gap is zero in either certified case and $U _ { \tau } - R _ { \tau }$ otherwise, normalized by η. As an independent check, every row satisfies $R _ { \tau } \leq R _ { \mathcal { D } } ^ { \star } \leq U _ { \tau }$ . The reference $R _ { \mathcal { D } } ^ { \star }$ is the full-set certificate established in the main experiment: 472 contexts are exhausted within 100 pieces, and the separate full-set audit verifies the remaining 28 contexts.

Figure 3(a) shows that exact certification arrives substantially before exhaustive enumeration. With $M \ : = \ : 2 5$ , the certification rates are 99.6% for Pure RR and 91.4% for Continuous RR, whereas only 56.2% of the active sets have been fully enumerated. $\mathbf { A } \mathbf { t } M = 5 0$ , these rates rise to 99.8% and 98.4%, respectively; at $M = 1 0 0$ , they are 100% and 99.8%. Equivalently, the first requested grid budget certifying exactness is at most 25 for 498 Pure RR instances and 457 Continuous RR instances. These are first certifying points on the chosen grid, not theoretical minimum prefix sizes.

As shown in Figure 3(b), the certified gap contracts rapidly even before exact termination. For Pure RR, its median is already zero at $M = 5$ , while its 0.90 quantile falls from 0.700 at $M = 5$ to 0.128 at $M = 1 0$ and zero at $M = 2 5$ For Continuous RR, the corresponding median/0.90-quantile values are $0 . 3 1 4 / 0 . 7 3 9 , 0 . 0 3 0 / 0 . 3 9 5$ , and $0 / 0$ . Hence the gap bound remains informative in the instances that are not yet exactly certified.

Table 3 separates the cost of prefix generation from the robust master. The best-first search takes a median of 15.1 ms and a 0.90 quantile of 51.8 ms by $M = 1 0 0$ , whereas the paired master solve has corresponding times of 1.045 and 2.959 seconds. Moreover, the median $M _ { \mathrm { e f f } }$ saturates at 22, because more than half of the active sets are already complete. These results support the output-sensitive interpretation of the method on this benchmark: a small gap-ordered prefix usually sufices for a rigorous full-problem certificate, and the search itself is not the computational bottleneck.

The single Continuous RR instance not certified by $M =$ 100 is a 4,770-piece extreme case; its normalized certified gap is 0.111. Thus, rather than recommending $M = 1 0 0$ as a general rule, the experiment further supports our conclusion that the number of active pieces required in practice is typically small relative to the total number of vertices.

## B.6 Coverage of Calibrated Sets

We compare what the calibrated sets cover in parameter space with what their induced certificates cover at the decision level. This experiment uses the same $n = 8$ split, RBF kernel-ridge predictor, and five calibration levels as the exactvalidation experiment. For method $m ,$ let $\mathcal { U } _ { m , \alpha } ( Z _ { i } )$ denote its calibrated uncertainty set and let $x _ { m , \alpha , i }$ be the decision returned from that set. We record

$$
I _ { m , \alpha , i } ^ { \mathrm { p a r } } = \mathbf { 1 } \{ C _ { i } \in \mathcal { U } _ { m , \alpha } ( Z _ { i } ) \}
$$

and

$$
\begin{array} { r l } & { I _ { m , \alpha , i } ^ { \mathrm { r e g } } = \mathbf { 1 } \{ \mathrm { R e g } ( x _ { m , \alpha , i } , C _ { i } ) \leq \rho _ { m , \alpha , i } \} , } \\ & { \rho _ { m , \alpha , i } = \underset { C ^ { \prime } \in \mathcal { U } _ { m , \alpha } ( Z _ { i } ) } { \operatorname* { s u p } } \mathrm { R e g } ( x _ { m , \alpha , i } , C ^ { \prime } ) . } \end{array}
$$

The SPO+ row uses the continuous regret-refinement decision. The other three rows use their method-specific Cost-RO decisions, after which $\rho _ { m , \alpha , i }$ is evaluated over the complete corresponding uncertainty set. Thus, the experiment compares the certificates naturally attached to the decisions each method actually prescribes; it does not replace the Cost-RO objectives of the conventional baselines by regret minimization.

As shown in Table 4, parameter coverage generally follows the nominal sequence, with the modest departures expected from one fixed calibration–test split. The largest Bernoulli standard error for 500 test observations is 0.022, so small differences from $1 - \alpha$ should not be interpreted as evidence for or against the marginal guarantee. More importantly, parameter inclusion deterministically implies decision-regret coverage for every method, but the converse need not hold. The conventional sets consequently attain decision-regret coverage between 0.992 and 1.000, even when their parameter coverage is close to the nominal target. The corresponding gap is smaller for the $\mathrm { S P O + s e t . }$ especially at the lower miscoverage levels. High decision-regret coverage alone, however, may result from a loose certificate rather than a more informative characterization of decision risk.

To distinguish these explanations, Table 5 reports the scale of the same full-set certificates. The SPO+ certificate has a median of 4.990 at $\alpha = . 0 5$ and 1.407 at $\alpha = . 1 0 \mathrm { { ; } }$ , and its median is exactly zero from $\alpha = . 2 0$ onward. Its 0.95 quantile falls from 7.697 to zero across the calibration levels. By contrast, the Residual- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$ and KNN–Ellipsoid certificates remain much larger throughout. The contextual box has a slightly smaller median at $\alpha = . 0 5$ , but its 0.95 quantile is 28.944, compared with 7.697 for $\mathrm { S P O + }$ . Hence, the near-universal decision-regret coverage of the conventional sets is accompanied by substantially wider regret bounds. This comparison supports the distinction motivating our construction: comparable marginal coverage in parameter space does not by itself imply a decision-informative certificate, whereas calibration in the $\mathrm { { \cal S P O + } }$ geometry more closely couples coverage to the downstream regret scale. These comparisons concern marginal coverage; they do not assert conditional coverage or uniform dominance in realized decision performance.

## B.7 Main Decision-Performance Comparison

Experimental Setup and Refinement Implementation The main comparison uses

$$
\begin{array} { r l r } & { n = 2 0 , } & { d _ { z } = 1 5 , \qquad \alpha = 0 . 1 0 , } \\ & { n _ { \mathrm { t r a i n } } = 5 0 0 0 , } & { n _ { \mathrm { c a l } } = 2 0 0 0 , } \\ & { n _ { \mathrm { t e s t } } = 5 0 0 . } \end{array}
$$

All methods use the same observations and test contexts. The principal predictor is polynomial kernel ridge regression, selected by training-only five-fold cross-validation. The tuning grid contains ten logarithmically spaced regularization strengths λ from $1 0 ^ { - 3 } \mathrm { t o } 1 0 ^ { 3 }$ and polynomial degrees $d \in \{ 2 , 3 , 4 \}$ . The kernel scale is $\gamma = 1 / d _ { z }$ , and the ofset is coe $\mathrm { f 0 } = 1$ The selected configuration is degree 2 with $\lambda = 2 . 1 5 4 ;$ its mean cross-validation MSE is 999.114. Feature and multi-output target standardization are fitted inside each training fold, after which the selected model is refitted on all 5000 training observations.

The compared methods are Nominal PTO, SPO+ Pure RR, $\mathrm { { \cal S P O + } }$ Continuous RR, Conformal Residual- $\cdot \ell _ { 2 }$ Cost-RO, Conformal Box Cost-RO, and KNN–Ellipsoid Cost-RO. The two SPO+ methods minimize worst-case regret over the calibrated SPO+ set. Pure RR restricts the returned decision to a vertex of the fractional-knapsack polytope, whereas Continuous RR optimizes over the full fractional polytope.

The active-piece search starts at the nominal vertex and uses a priority queue ordered by the predicted objective gap

$$
\delta ( v ) = { \widehat C } ^ { \top } ( v - \widehat v ) .
$$

Canonical vertex keys, rounded to ten decimal places, break equal-gap ties. Only vertices satisfying

$$
\delta ( v ) \leq \eta / 2 + 1 0 ^ { - 9 } \operatorname* { m a x } \{ 1 , | \eta | \}
$$

are expanded. For computational eficiency, the pure and continuous masters are first solved using the same best-first prefix of at most 100 active pieces. Every returned decision is then reevaluated against the complete active set, so the reported certificates use the full set rather than the retained prefix. The complete set contains at most 100 pieces in 472 of the 500 test contexts. In each of the remaining 28 contexts, the full-set certificate is exactly equal to its prefix certificate for both refinements; hence the prefix solution is also optimal for the full active set. The worst-case pieces have maximum prefix ranks 28 and 27 for Pure and Continuous RR, respectively. Thus, the 100-piece limit is only a computational device and does not alter any reported decision or certificate. When $\eta \leq 2 \Gamma$ , up to the scale-aware numerical tolerance, the nominal decision is returned directly.

Table 3: Certified anytime behavior by requested prefix budget. Exact, full-set, and tie-expansion columns are percentages; efective prefix sizes and runtimes are reported as median / 0.90 quantile. Search time is in milliseconds and paired-solve time is in seconds.
<table><tr><td>Domain</td><td>M</td><td>Exact</td><td>Full set</td><td>Tie exp.</td><td> $M _ { \mathrm { e f f } }$ </td><td>Search (ms)</td><td>Solve (s)</td></tr><tr><td>Pure RR</td><td>1</td><td>0.8</td><td>0.8</td><td>0.0</td><td>1.0/1.0</td><td>0.769/1.231</td><td>0.049/0.108</td></tr><tr><td>Pure RR</td><td>5</td><td>63.2</td><td>5.8</td><td>0.0</td><td>5.0/5.0</td><td>3.418/4.577</td><td>0.312/0.606</td></tr><tr><td>Pure RR</td><td>10</td><td>86.6</td><td>21.6</td><td>0.2</td><td>10.0/10.0</td><td>6.668/8.487</td><td>0.483/1.045</td></tr><tr><td>Pure RR</td><td>25</td><td>99.6</td><td>56.2</td><td>0.2</td><td>22.0/25.0</td><td>14.931/18.760</td><td>0.833/1.557</td></tr><tr><td>Pure RR</td><td>50</td><td>99.8</td><td>83.2</td><td>0.0</td><td>22.0/50.0</td><td>15.055/33.920</td><td>1.045/2.107</td></tr><tr><td>Pure RR</td><td>100</td><td>100.0</td><td>94.4</td><td>0.0</td><td>22.0/72.1</td><td>15.055/51.759</td><td>1.045/2.959</td></tr><tr><td>Continuous RR</td><td>1</td><td>0.8</td><td>0.8</td><td>0.0</td><td>1.0/1.0</td><td>0.769/1.231</td><td>0.049/0.108</td></tr><tr><td>Continuous RR</td><td>5</td><td>17.8</td><td>5.8</td><td>0.0</td><td>5.0/5.0</td><td>3.418/4.577</td><td>0.312/0.606</td></tr><tr><td>Continuous RR</td><td>10</td><td>46.4</td><td>21.6</td><td>0.2</td><td>10.0/10.0</td><td>6.668/8.487</td><td>0.483/1.045</td></tr><tr><td>Continuous RR</td><td>25</td><td>91.4</td><td>56.2</td><td>0.2</td><td>22.0/25.0</td><td>14.931/18.760</td><td>0.833/1.557</td></tr><tr><td>Continuous RR</td><td>50</td><td>98.4</td><td>83.2</td><td>0.0</td><td>22.0/50.0</td><td>15.055/33.920</td><td>1.045/2.107</td></tr><tr><td>Continuous RR</td><td>100</td><td>99.8</td><td>94.4</td><td>0.0</td><td>22.0/72.1</td><td>15.055/51.759</td><td>1.045/2.959</td></tr></table>

Table 4: Marginal parameter and decision-regret coverage on the fixed $n = 8$ test split. Each entry is parameter coverage / decision-regret coverage; the nominal target is $1 - \alpha$
<table><tr><td>Method</td><td> $\alpha = . 0 5$ </td><td> $\alpha = . 1 0$ </td><td> $\alpha = . 2 0$ </td><td> $\alpha = . 3 0$ </td><td> $\alpha = . 4 0$ </td></tr><tr><td>SPO+ regret set (Continuous RR)</td><td>0.964/0.974</td><td>0.912/0.926</td><td>0.826/0.862</td><td>0.706/0.802</td><td>0.656/0.816</td></tr><tr><td>Residual  ${ \bf - } \ell _ { 2 }$  Cost-RO</td><td>0.938/0.998</td><td>0.892/0.998</td><td>0.804/0.998</td><td>0.696/0.996</td><td>0.572/0.992</td></tr><tr><td>Contextual Box Cost-RO</td><td>0.942/1.000</td><td>0.904/1.000</td><td>0.796/1.000</td><td>0.714/1.000</td><td>0.608/0.998</td></tr><tr><td>KNN-Ellipsoid Cost-RO</td><td>0.936/1.000</td><td>0.868/1.000</td><td>0.778/1.000</td><td>0.678/1.000</td><td>0.570/1.000</td></tr></table>

Table 5: Scale of the method-specific full-set regret certificates in Table 4. Each entry is the median / 0.95 quantile over the 500 test observations.
<table><tr><td>Method</td><td> $\alpha = . 0 5$ </td><td> $\alpha = . 1 0$ </td><td> $\alpha = . 2 0$ </td><td> $\alpha = . 3 0$ </td><td> $\alpha = . 4 0$ </td></tr><tr><td>SPO+ regret set (Continuous RR)</td><td>4.990/7.697</td><td>1.407/3.031</td><td>0.000/0.751</td><td>0.000/0.073</td><td>0.000/0.000</td></tr><tr><td>Residual  $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$  Cost-RO</td><td>35.783/55.768</td><td>30.142/47.248</td><td>23.427/37.042</td><td>18.935/30.515</td><td>14.934/25.445</td></tr><tr><td>Contextual Box Cost-RO</td><td>4.033/28.944</td><td>3.531/26.607</td><td>2.792/21.779</td><td>2.488/20.421</td><td>2.132/18.512</td></tr><tr><td>KNN-Ellipsoid Cost-RO</td><td>88.626/244.809</td><td>50.288/165.904</td><td>28.770/90.569</td><td>20.315/60.454</td><td>15.750/46.285</td></tr></table>

Cost-RO Baselines Unlike the proposed refinements, the three conventional baselines solve

$$
\operatorname* { m i n } _ { x \in \mathcal { X } ( p , b ) } \operatorname* { m a x } _ { C \in \mathcal { U } ( Z ) } { C ^ { \top } x }
$$

and do not optimize worst-case regret. Each baseline has its own calibration score and conformal threshold. Since item values are nonnegative, the minimization costs $C = - V$

satisfy $C _ { j } \leq 0 .$ . For solver implementation, strict negativity is represented numerically by

$$
C _ { 0 } ^ { \epsilon } = \{ C : C _ { j } \leq - \epsilon _ { s } , \ j = 1 , \ldots , n \} , \qquad \epsilon _ { s } = 1 0 ^ { - 6 } .
$$

Replacing the boundary value 0 with $- \epsilon _ { s }$ is primarily intended to improve numerical stability while providing a solver-compatible implementation of $C _ { j } ~ < ~ 0$ . The SPO+ fixed-piece models use the closed limiting support $C _ { j } \leq 0 .$

For Residual $\mathbf { \mathcal { - l } } _ { 2 } \mathbf { C o s t - R O }$ , the score and calibrated set are

$$
\begin{array} { r l } & { S _ { \ell _ { 2 } } ( C , \widehat { C } ) = \| C - \widehat { C } \| _ { 2 } , } \\ & { \qquad \mathcal { U } _ { \ell _ { 2 } } = \{ C : \| C - \widehat { C } \| _ { 2 } \leq \eta _ { \ell _ { 2 } } \} } \\ & { \qquad \cap C _ { 0 } ^ { \epsilon } . } \end{array}
$$

The direct SOCP reformulation uses dual variables $\lambda \in \mathbb { R } _ { + } ^ { n }$ for the cost-support constraints and solves

$$
\operatorname* { m i n } _ { x , \lambda } \widehat { C } ^ { \top } ( x - \lambda ) + \eta \varepsilon _ { 2 } \| x - \lambda \| _ { 2 } - \epsilon _ { s } { \bf 1 } ^ { \top } \lambda \quad \mathrm { s . t . } x \in \mathcal { X } ( p , b ) .
$$

For the contextual Box baseline, a separate gradientboosting quantile regressor is fitted to the absolute training residuals of each item. Its settings are $\mathtt { l o s s } { = } \mathtt { q u a n t i l }$ e, $\mathtt { a l p h a } { = } 0 . 8 , \mathtt { n } _ { - } { \in } \mathtt { s t i m a t o r s } { = } 4 0 , \mathtt { m a x \_ d e p t h } { = } 2$ , and random\_ $s \ t a t e = 4 2$ . Its predicted scale is truncated below at $1 0 ^ { - 6 }$ . With scale vector ${ \overline { { H } } } ( Z )$ , the calibration score is

$$
S _ { \mathrm { B o x } } ( C , \widehat { C } , Z ) = \operatorname* { m a x } _ { j } \frac { | C _ { j } - \widehat { C } _ { j } | } { H _ { j } ( Z ) } .
$$

The calibrated coordinate bounds are

$$
\widehat { C } _ { j } - \eta _ { \mathrm { B o x } } H _ { j } ( Z ) \leq C _ { j } \leq \widehat { C } _ { j } + \eta _ { \mathrm { B o x } } H _ { j } ( Z ) ,
$$

intersected with $C _ { 0 } ^ { \epsilon }$ . Since $x \geq 0$ , the robust problem is the LP

$$
\operatorname* { m i n } _ { x \in \mathcal { X } ( p , b ) } \sum _ { j } \operatorname* { m i n } \{ \widehat { C } _ { j } + \eta _ { \mathrm { B o x } } H _ { j } ( Z ) , - \epsilon _ { s } \} x _ { j } .
$$

For KNN–Ellipsoid Cost-RO, features are standardized using the training split. Each calibration or test context uses its $k = 1 0 0$ nearest training observations. Their observed costs define a local mean $\mu _ { K } ( Z ) ;$ ; the local covariance $\Sigma _ { K } ( Z )$ is estimated using Ledoit–Wolf shrinkage. If its smallest eigenvalue is below $\mathrm { \overline { { 1 0 } } ^ { - 1 0 } }$ , the implementation adds the following multiple of the $n \times n$ identity matrix $I \colon$

$$
1 0 ^ { - 8 } \frac { \mathrm { t r } ( \Sigma _ { K } ) } { n } I .
$$

The calibration score and uncertainty set are

$$
S _ { K } ( Z , C ) = \sqrt { ( C - \mu _ { K } ) ^ { \top } \Sigma _ { K } ^ { - 1 } ( C - \mu _ { K } ) } ,
$$

$$
\mathcal { U } _ { K } ( Z ) = \{ C : ( C - \mu _ { K } ) ^ { \top } \Sigma _ { K } ^ { - 1 } ( C - \mu _ { K } ) \leq \eta _ { K } ^ { 2 } \} \cap C _ { 0 } ^ { \epsilon } .
$$

With covariance factor $L$ satisfying $\Sigma _ { K } = L L ^ { \top }$ , the direct SOCP reformulation is

$$
\begin{array} { r l } { \underset { x , \lambda \geq 0 } { \mathrm { m i n } } } & { \mu _ { K } ^ { \top } ( x - \lambda ) + \eta _ { K } \| L ^ { \top } ( x - \lambda ) \| _ { 2 } } \\ & { - \epsilon _ { s } \mathbf { 1 } ^ { \top } \lambda } \\ { \mathrm { s . t . } } & { x \in \mathcal { X } ( p , b ) . } \end{array}
$$

Regret-Distribution Profiles The summary statistics in Table 2 do not identify where the tail benefit of refinement begins. We therefore compare the empirical regret distributions at $\alpha = 0 . 1 0$ over 100 evenly spaced quantile levels from 0.50 through 0.99. Figure $4 ( \mathrm { a } )$ displays the quantile curves, while panel (b) subtracts the corresponding Nominal PTO quantile to make the location and direction of each diference explicit.

Continuous RR performs worse over the center of the distribution but improves upon Nominal PTO from approximately the 0.94 quantile onward, thereby reducing severe regret. Pure RR closely follows the nominal curve and intervenes only in the subset of contexts identified as vulnerable by the calibrated regret model. Hence, continuous refinement emphasizes tail protection, while pure refinement adds robustness with limited disruption to nominal performance.

![](images/ece948ca3e441decc8ffb4bcc5cad01101e2d90ee945fcc811246568f5be0ddc.jpg)

![](images/e252cd1d147662e3203d59269dde6c9844e3b20a5ef55c6727ca90ab821db701.jpg)  
Figure 4: Realized-regret distribution at $\alpha = 0 . 1 0 . ( \mathrm { a } )$ Empirical regret quantiles. (b) Quantile-wise gap to Nominal $\mathrm { P T O } .$ where negative values are better. KNN–Ellipsoid is omitted for scale and retained in Table 2.

## B.8 Objective-Matched Regret Comparison

We complement the main comparison with minimax-regret variants of the three residual-based RO baselines. The experiment uses the same $n = 2 0$ fractional-knapsack instances, predictor, calibration sample, and 500 test observations at $\alpha = 0 . 1 0$ as Table 2. For each calibrated set $\mathcal { U } _ { m } ( Z )$ , where m denotes Residual- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$ , Box, or KNN–Ellipsoid, we solve

$$
\operatorname* { m i n } _ { x \in \mathcal { X } ( p , b ) } \operatorname* { m a x } _ { c \in \mathcal { U } _ { m } ( Z ) } \left\{ c ^ { \top } x - \operatorname* { m i n } _ { y \in \mathcal { X } ( p , b ) } c ^ { \top } y \right\} .\tag{55}
$$

Each baseline retains its original calibrated set and support constraints, changing only the downstream objective. The decision domain is the full fractional polytope, as in Continuous RR. All three baseline variants attain a relative optimality gap of at most $1 0 ^ { - 6 }$ on all 500 test instances.

Table 6: Objective-matched performance at $n \ = \ 2 0$ and $\alpha = 0 . 1 0$ on 500 test observations. Change is the decisionchange rate; Time is mean recorded time per instance (one solver thread for PTO, SPO+, and Box; up to twenty for Residual- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 2 }$ and KNN).

<table><tr><td>Method</td><td>Mean</td><td>Q95</td><td>CVaR95</td><td>Change</td><td>Time (s)</td></tr><tr><td>Nominal PTO</td><td>7.605</td><td>38.664</td><td>60.557</td><td>0.000</td><td>0.00124</td></tr><tr><td>Pure RR</td><td>7.763</td><td>38.956</td><td>61.814</td><td>0.100</td><td>1.41800</td></tr><tr><td>Continuous RR</td><td>11.117</td><td>35.137</td><td>52.618</td><td>0.990</td><td>1.88969</td></tr><tr><td>Box RR</td><td>17.890</td><td>57.051</td><td>73.152</td><td>0.986</td><td>0.10363</td></tr><tr><td>Residual  ${ \bf - } \ell _ { 2 }$ </td><td>RR  38.348</td><td>64.456</td><td>78.628</td><td>1.000</td><td>13.15628</td></tr><tr><td>KNN RR</td><td>52.924</td><td>119.828</td><td>146.999</td><td>1.000</td><td>14.37106</td></tr></table>

Table 6 shows that both SPO+ refinements achieve lower mean and tail regret than all three objective-matched baselines. Switching KNN from worst-case cost to minimax regret improves its mean regret from 624.111 to 52.924, while the advantage of SPO+ remains. These results support the contribution of the SPO+-calibrated set beyond the choice of downstream objective.

Pure and Continuous RR are run consecutively with shared active-piece search and model construction; their reported times include this common work and their respective optimization and certification stages.

## B.9 Additional Validation of Marginal Guarantees

As a supplementary check using the same $n = 8$ exactvalidation split, we evaluate the prespecified marginal events for the proposed method across all five calibration levels.

For each α, we evaluate the six empirical events

$$
\begin{array} { r l } & { s _ { + } ( C , \widehat { C } ) \leq \eta _ { \alpha } , \qquad \mathrm { R e g } ( \widehat { v } , C ) \leq \eta _ { \alpha } , } \\ & { \qquad \mathrm { R e g } ( x ^ { \mathrm { p u r e } } , C ) \leq \rho _ { \eta _ { \alpha } } ( x ^ { \mathrm { p u r e } } ) , } \\ & { \qquad \mathrm { R e g } ( x ^ { \mathrm { c o n t } } , C ) \leq \rho _ { \eta _ { \alpha } } ( x ^ { \mathrm { c o n t } } ) . } \\ & { \qquad \mathrm { R e g } ( x ^ { \mathrm { p u r e } } , C ) \leq ( \eta _ { \alpha } - 2 \Gamma ) _ { + } , } \\ & { \qquad \mathrm { R e g } ( x ^ { \mathrm { c o n t } } , C ) \leq ( \eta _ { \alpha } - 2 \Gamma ) _ { + } . } \end{array}
$$

An event $a \leq b$ is accepted numerically when

$$
a \leq b + 1 0 ^ { - 6 } \operatorname* { m a x } \{ 1 , | b | \} .
$$

The validation table reports the empirical frequencies with target $1 - \alpha$

Table 7 reports the six prespecified marginal events. SPO+ score inclusion frequencies are 0.964, 0.912, 0.826, 0.706, and 0.656, compared with targets 0.95, 0.90, 0.80, 0.70, and 0.60, respectively. The nominal-regret, set-relative, and margin-certificate events are likewise at or above their corresponding targets. These fixed-seed empirical frequencies are consistent with the marginal guarantees up to ordinary finite-sample variation.

## B.10 Additional Predictor-Quality Comparison

Predictors and Model Selection This experiment uses the same $n = 2 0$ data, fixed split, test contexts, seed, and $\alpha = 0 . 1 0$ as the main comparison. It includes only Nominal PTO, SPO+ Pure RR, and SPO+ Continuous RR. Each predictor has its own SPO+ calibration scores and conformal threshold; a threshold calibrated for one predictor is never reused for another.

All new predictors are selected by deterministic five-fold cross-validation on the training split. The folds are shufled using seed 42, and selection minimizes mean multi-output validation MSE. Grid search uses four parallel workers, with estimator-level parallelism restricted to one worker. The selected model is refitted on the complete training split. Polynomial KRR reuses the model, predictions, and calibration evidence from the main comparison.

Ridge. The regularization grid contains nine logarithmically spaced values of λ from $\overline { { 1 0 ^ { - 4 } } } \tan 1 0 ^ { 4 }$ . Feature and target standardization are fitted within each fold. The selected value is $\lambda = 1$ , with mean cross-validation MSE 1151.114.

Polynomial KRR. The kernel and tuning grid follow $\mathsf { A p - }$ pendix B.7. The reused configuration is $d = 2$ and $\lambda =$ 2.154, with mean cross-validation MSE 999.114.

RBF KRR. The grid is

$$
\begin{array} { r l } & { \lambda \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 \} , } \\ & { \gamma \in d _ { z } ^ { - 1 } \{ 0 . 2 5 , 0 . 5 , 1 , 2 \} . } \end{array}
$$

Features and targets are standardized within each fold. The selected configuration is $\lambda = 1$ and $\gamma = 1 / 6 0 = 0 . 0 1 6 6 7 .$ with mean cross-validation MSE 1033.416.

MLP. The candidate hidden-layer configurations are (64) and (128, 64), with weight-decay values $\{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ and initial learning rates $\{ 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ . All candidates use ReLU activations, Adam, at most 500 epochs, an early-stopping validation fraction of 0.1, a 30-epoch noimprovement rule, and seed 42. Features and targets are standardized within each fold. The selected configuration has one 64-unit hidden layer, weight decay $1 0 ^ { - 3 }$ , and initial learning rate $1 0 ^ { - 3 }$ ; its mean cross-validation MSE is 1032.420.

Random Forest. The grid is

$$
\begin{array} { r l } & { \mathfrak { n _ { - } e s t i m a t o r s \in \{ 2 0 0 , 4 0 0 \} } , } \\ & { \mathfrak { m i n _ { - } s a m p l e s \_ l e a f \in \{ 1 , 2 , 5 \} } , } \\ & { \qquad \mathfrak { m a x \_ f e a t u r e s \in \{ 0 . 5 , 1 . 0 \} } . } \end{array}
$$

The model uses raw features and raw multi-output values, seed 42, and one estimator worker. The selected configuration has 400 trees, mi $\mathrm { . n _ { - } s a m p l e s _ { - } l e a f ~ = ~ 1 }$ , and $\mathrm { m a x \_ f e a t u r e s = 1 . 0 }$ , with mean cross-validation MSE 3106.874.

The independently calibrated SPO+ thresholds, in the row order of Table 8, are 110.280, 88.482, 90.922, 91.778, and 202.496. Test RMSE is computed jointly over all test observations and items. For each predictor, the reported mean active-piece count comes from a complete, uncapped adjacency search. The Pure and Continuous RR masters subsequently use at most the first 150 active pieces in best-first order. The raw output stores the complete count, used count, and truncation indicator separately. The margin-skip rate is the fraction of test contexts satisfying $\eta \leq 2 \bar { \Gamma }$ . No Cost-RO baseline is rerun in this experiment.

Exploratory Results Table 8 summarizes prediction accuracy, mean decision regret, and complete active-piece counts for the five predictors. The refinement masters use at most the first 150 pieces in best-first order, so this computational cap does not truncate the reported geometric statistic.

Prediction accuracy and downstream performance are related but not interchangeable. Polynomial KRR has the lowest test RMSE (31.55), whereas RBF KRR has the lowest Nominal and Pure-RR mean regrets (7.56 and 7.70); polynomial KRR instead has the lowest Continuous-RR mean regret (11.12). Pure RR remains close to Nominal PTO for every predictor and slightly improves Ridge from 10.08 to 9.97, but is slightly worse for the other four. Continuous RR has higher mean regret than Nominal PTO throughout, consistent with the mean–tail trade-of observed in the main comparison rather than a uniform improvement in average performance. Random forest is weakest in both RMSE and regret, although the remaining prediction and decision rankings are not identical.

Table 7: Empirical probabilities for the six validation events; nominal target $1 - \alpha$ . These marginal frequencies summarize one fixed-seed experiment and do not claim conditional coverage.
<table><tr><td>Event</td><td> $\alpha = 0 . 0 5$ </td><td> $\alpha = 0 . 1 0$ </td><td> $\alpha = 0 . 2 0$ </td><td> $\alpha = 0 . 3 0$ </td><td> $\alpha = 0 . 4 0$ </td></tr><tr><td>SPO+ score inclusion</td><td>0.964</td><td>0.912</td><td>0.826</td><td>0.706</td><td>0.656</td></tr><tr><td>Nominal regret</td><td>0.994</td><td>0.978</td><td>0.928</td><td>0.850</td><td>0.816</td></tr><tr><td>Pure RR set-relative certificate</td><td>0.978</td><td>0.944</td><td>0.882</td><td>0.818</td><td>0.816</td></tr><tr><td>Continuous RR set-relative certificate</td><td>0.974</td><td>0.926</td><td>0.862</td><td>0.802</td><td>0.816</td></tr><tr><td>Pure RR margin certificate</td><td>0.980</td><td>0.946</td><td>0.888</td><td>0.820</td><td>0.816</td></tr><tr><td>Continuous RR margin certificate</td><td>0.982</td><td>0.942</td><td>0.882</td><td>0.806</td><td>0.816</td></tr></table>

Table 8: Prediction quality, decision regret, and refinement geometry. Active counts use the complete uncapped search; Lower values are better except for the skip rate.
<table><tr><td>Predictor</td><td>RMSE</td><td>Nominal</td><td>Pure RR</td><td>Continuous RR</td><td>Active</td><td> $\mathrm { S k i p } \left( \% \right)$ </td></tr><tr><td>Ridge</td><td>33.87</td><td>10.08</td><td>9.97</td><td>14.60</td><td>370.63</td><td>0.20</td></tr><tr><td>Poly. KRR</td><td>31.55</td><td>7.60</td><td>7.76</td><td>11.12</td><td>44.96</td><td>0.80</td></tr><tr><td>RBF KRR</td><td>32.15</td><td>7.56</td><td>7.70</td><td>11.35</td><td>45.47</td><td>0.80</td></tr><tr><td>MLP</td><td>31.96</td><td>7.75</td><td>7.84</td><td>11.74</td><td>45.66</td><td>0.60</td></tr><tr><td>Random forest</td><td>53.94</td><td>20.17</td><td>20.88</td><td>29.50</td><td>181.42</td><td>0.20</td></tr></table>

Geometric complexity is even less closely related to prediction quality. Polynomial KRR, RBF KRR, and the MLP have similar complete mean active-piece counts (44.96, 45.47, and 45.66), whereas Random Forest has 181.42 and Ridge has the largest mean, 370.63, despite substantially outperforming Random Forest in prediction and decision quality. Margin certificates trigger on only 0.2%–0.8% of contexts. The 150-piece refinement cap binds in 232 of the 2,500 predictor–context pairs, including 151 Random-Forest pairs; the reported active counts nevertheless use the complete search. These fixed-split comparisons are descriptive: the predictor jointly changes its errors, calibrated score distribution, and local decision geometry, so RMSE alone does not determine regret or search burden.

## B.11 Optimization and Numerical Settings

The main knapsack experiments use

$$
\begin{array} { r l } & { \tt { O u t p u t F 1 a g } = 0 , \quad \tt { T h r e a d s } = 1 , } \\ & { \tt { S e e d } = 4 2 . } \\ & { \tt { M I P G a p } = 1 0 ^ { - 6 } , } \\ & { \tt { F e a s i b i l i t y T o l } = 1 0 ^ { - 7 } , } \\ & { \tt { O p t i m a l i t y T o l } = 1 0 ^ { - 7 } . } \end{array}
$$

Fractional-knapsack LPs use deterministic dual simplex and a secondary tie-breaking optimization that preserves the primary objective. The pure vertex master additionally uses $\mathtt { I n t F e a s T o l } = 1 0 ^ { - 9 }$ . The auxiliary mixed-integer regret oracle used to evaluate the $\mathrm { C o s t - R O }$ certificates in Table 5 uses $\mathtt { F e a s i b i l i t y T o l } = 1 0 ^ { - 9 }$ and $\mathtt { I n t F e a s T o l } =$ $1 0 ^ { - 7 }$ . The tighter primal tolerance prevents a feasible-vertex decomposition error from being amplified by the cost coeficients.

The SPO+ masters use constraint generation with at most 200 iterations and add all currently violated cuts at each iteration. The final certificate is independently recomputed over all retained fixed-piece oracles. A solve is accepted only if Gurobi returns OPTIMAL, any applicable relative gap is at most $1 0 ^ { - 6 } .$ , all certificates are finite, and the normalized physical feasibility residual is at most $1 0 ^ { - 6 } .$ . The continuous, pure, and nominal certificates must also satisfy their expected hierarchy up to a scale-aware tolerance of $\mathrm { 1 0 ^ { - 6 } }$ . Failures are treated as errors; no failed instance is silently replaced by its nominal decision or removed from the test sample. The Cost-RO decisions themselves use one LP or SOCP and do not optimize through the bilinear adversarial-regret oracle; that oracle is invoked only afterward to evaluate the supplementary full-set certificates.

## B.12 Computing Environment

All experiments were run on a PC with an Intel Core Ultra 9 275HX processor and 32 GB of RAM, using 64-bit Microsoft Windows 11. The implementation used Python 3.12.7, Gurobi 13.0.1, NumPy 2.4.3, pandas 2.3.2, SciPy 1.16.0, scikit-learn 1.7.1, joblib 1.5.2, and Matplotlib 3.10.6. GPU acceleration is not used. Although the processor provides 24 cores, the main knapsack experiments used one Gurobi thread per solve.

## C Additional Experiments

We complement the fractional-knapsack experiments with two additional LP cases: solar-grid budget allocation using real installation data and synthetic sparse transportation.

Both cases compare the same six methods as the main experiment.

## C.1 Real-Data Solar-Grid Allocation

Data and Rolling Evaluation Protocol We follow the solar-grid investment setting of Zhou, Orfanoudaki, and Zhu (2026), using publicly released monthly aggregates of the real solar-installation data studied by Zhou et al. (2025).<sup>1</sup> The series covers four substations from January 2011–December 2024, with 168 monthly observations of a four-dimensional count vector and 182 installations in total. The monthly sequence has no missing observations, and zero-installation months are retained.

For each evaluation year $y \in \{ 2 0 1 3 , \ldots , 2 0 2 4 \}$ , we use a 24-month window from January of year $y - 1$ through December of year y. The three chronological splits are:

• training: January–August of year $y - 1 ;$

• calibration: September of year y − 1–April of year y;

• testing: May–December of year y.

Each split contains eight months. Advancing the window by twelve months produces twelve evaluation windows and 96 distinct test months. The miscoverage level is $\alpha = 0 . 2 0$ throughout.

Let $N _ { t j }$ denote the installation count at station j in month t, and let $\mathcal { T } _ { y } ^ { \mathrm { t r } }$ be the training months for year y. We define the station-specific threshold $\tau _ { j , y } ,$ , the smoothed overload-loss coeficient $c _ { t j } ^ { ( y ) }$ , and the sigmoid function σ as follows:

$$
\begin{array} { l } { \displaystyle \tau _ { j , y } = \frac { 1 } { 8 } \sum _ { t \in \mathbb { Z } _ { y } ^ { \mathrm { t r } } } N _ { t j } , } \\ { \displaystyle c _ { t j } ^ { ( y ) } = \sigma ( 0 . 5 ( N _ { t j } - \tau _ { j , y } ) ) , } \\ { \displaystyle \sigma ( a ) = \frac { 1 } { 1 + \exp ( - a ) } . } \end{array}
$$

Thus, each station’s threshold is determined by its trainingperiod installation history, and larger installation counts yield larger loss coeficients. The same threshold is used to transform the training, calibration, and test observations within that window.

The point predictor is the historical mean of the transformed training costs:

$$
\hat { c } _ { j , y } = \frac { 1 } { 8 } \sum _ { t \in \mathcal { T } _ { y } ^ { \mathrm { t r } } } c _ { t j } ^ { ( y ) } .
$$

This historical-mean setting uses the four station histories directly. Predictions and calibrated sets are held fixed over the eight test months and updated when the window advances.

Allocation Model and Regret Before the monthly loss coeficients are observed, a planner allocates a divisible upgrade budget of two station-equivalents. An upgrade fraction $u _ { j } \in [ 0 , 1 ]$ reduces station j’s exposure by the same fraction, subject to $\textstyle \sum _ { j } u _ { j } \leq 2$ . Writing $x _ { j } = 1 - u _ { j }$ for the remaining exposure gives the linear allocation model

$$
\begin{array} { r l } & { \underset { x \in \mathcal { X } _ { \mathrm { s o l } } } { \operatorname* { m i n } } c _ { t } ^ { \top } x , } \\ & { \mathcal { X } _ { \mathrm { s o l } } = \{ x \in [ 0 , 1 ] ^ { 4 } : \mathbf { 1 } ^ { \top } x \geq 2 \} . } \end{array}\tag{56}
$$

Here the window superscript on $c _ { t }$ is suppressed. The objective is a dimensionless smoothed overload loss, and realized regret is

$$
R _ { t } ( x ) = c _ { t } ^ { \top } x - \operatorname* { m i n } _ { v \in \mathcal { X } _ { \mathrm { s o l } } } c _ { t } ^ { \top } v .
$$

The feasible region has the eleven vertices

$$
\mathcal { V } _ { \mathrm { s o l } } = \{ v \in \{ 0 , 1 \} ^ { 4 } : \mathbf { 1 } ^ { \top } v \geq 2 \} .
$$

Nominal PTO optimizes the predicted loss over this region. Pure RR chooses a vertex, corresponding to full upgrades at selected stations, while Continuous RR permits fractional upgrades across stations.

Calibration and Refinement Each method computes its score on the eight calibration months using the fixed trainingperiod predictor. The conformal order-statistic index is

$$
k _ { \alpha } = \lceil ( 8 + 1 ) ( 1 - 0 . 2 0 ) \rceil = 8 ,
$$

so the threshold is the largest of that method’s eight calibration scores. Pure and Continuous RR share the same SPO+ threshold and uncertainty set.

The Residual-ℓ<sub>2</sub>, Box, and KNN–Ellipsoid baselines follow Appendix B.7. For this historical-mean setting, their context input is constant and the calibrated sets are used over R<sup>4</sup>. KNN uses all eight training observations, $k \mathbf { \Psi } =$ min $( 1 0 0 , n _ { \mathrm { t r a i n } } ) = 8 ;$ ; its small-sample covariance uses an eigenvalue floor of $1 0 ^ { - 1 2 }$ . The score definitions and other baseline parameters are unchanged.

The eleven vertices are explicitly available, so we retain every active SPO+ piece identified by the screening rule. A fixed-piece LP evaluates the worst-case regret of each candidate decision. Pure RR compares all eleven feasible vertices. Continuous RR combines the dual representations of all active pieces into the finite LP of Proposition 4, and its returned decision is evaluated again using the primal fixed-piece oracles. The master objective and recomputed worst-case regret agree within $1 0 ^ { - 6 }$ . Solver settings follow Appendix B.11.

Hindsight optimizer ties are resolved by minimizing predicted cost among the tied vertices, followed by a fixed vertex order. Among equally good Pure-RR candidates, the implementation first minimizes the predicted objective gap and then uses the same fixed order.

Decision Performance All methods are evaluated on the same 96 test months. We pool their monthly regrets to compute the mean, linearly interpolated Q95, and empirical CVaR95 using the definitions in Appendix B.2. For this sample size, the upper 5% corresponds to 4.8 observations, so CVaR95 uses the four largest regrets and 0.8 of the fifthlargest regret, divided by 4.8.

Pure and Continuous RR are run consecutively and share the piece models constructed in the Pure stage; the table reports their separately recorded stage times.

In Table 9, Continuous RR reduces CVaR95 from 0.3718 to 0.2464, a 33.7% reduction, with a 1.7% increase in mean regret. It also lowers Q95 from 0.2606 to 0.2059 and attains the lowest tail regret among the six methods. Residual-ℓ RO produces a similar tail profile, while Pure RR preserves the nominal decisions and their mean regret.

In all 96 test months, Continuous RR spreads the budget across all four stations, with station-level upgrade fractions ranging from 0.493 to 0.507. Nominal PTO and Pure RR fully upgrade two stations. The application therefore gives a direct allocation interpretation of the mean–tail trade-of.

Table 9: Decision performance for real-data solar-grid allocation $( \alpha = 0 . 2 0$ , 96 test months). Change is the decisionchange rate; Time is the mean recorded solver-stage time. Lower regret is better.

<table><tr><td>Method</td><td>Mean</td><td>Q95</td><td>CVaR95</td><td>Change</td><td>Time (s)</td></tr><tr><td>Nominal PTO</td><td>0.07869 0.2606</td><td></td><td>0.3718</td><td>0.000</td><td>0.00043</td></tr><tr><td>Pure RR</td><td>0.07869 0.2606</td><td></td><td>0.3718</td><td>0.000</td><td>0.01193</td></tr><tr><td>Continuous RR</td><td>R0.080030.2059</td><td></td><td>0.2464</td><td>1.000</td><td>0.00547</td></tr><tr><td>Residual  $\mathbf { \nabla } \cdot \ell _ { 2 }$ </td><td>RO 0.08006 0.2062</td><td></td><td>0.2471</td><td>1.000</td><td>0.00140</td></tr><tr><td>Box RO</td><td>0.11770.3541</td><td></td><td>0.4488</td><td>1.000</td><td>0.00038</td></tr><tr><td>KNN RO</td><td>0.093530.2365</td><td></td><td>0.3075</td><td>1.000</td><td>0.00130</td></tr></table>

## C.2 Sparse Transportation Linear Programs

Network and Data Generation The network has eight origins, eight destinations, and 32 permitted shipping arcs, half of the 64 possible origin–destination connections. We use five independent seeds, 42–46. For each seed, supply and demand vectors are drawn independently from Dirichlet $( 2 , \ldots , 2 )$ and then fixed, each summing to one. Draws are screened to keep distinct proper supply and demand subset sums more than $1 0 ^ { - 9 }$ apart, giving nondegenerate transportation vertices.

To construct a feasible sparse network, we first solve the complete-network transportation problem with arc costs sampled independently from Unif[0, 1]. We retain the resulting spanning-tree support of fifteen arcs and add seventeen arcs sampled without replacement from the remaining connections. The resulting arc set, supply vector, and demand vector are shared by all training, calibration, and test observations for that seed.

We use the quadratic multiplicative-noise generator from Appendix B.1. For each arc $e ,$ let $\theta _ { e }$ denote its featurecoeficient vector, $\mu _ { e } ( Z )$ its conditional mean cost, and $\xi _ { e }$ its multiplicative noise:

$$
\begin{array} { r l r } & { } & { Z \sim \mathrm { U n i f } ( [ 0 , 4 ] ^ { 1 5 } ) , \phantom { Z Z Z Z } } \\ & { } & { \mu _ { e } ( Z ) = ( \theta _ { e } ^ { \top } Z ) ^ { 2 } , \phantom { Z Z Z Z } } \\ & { } & { C _ { e } = \mu _ { e } ( Z ) \xi _ { e } , \phantom { Z Z Z Z } } & { \xi _ { e } \overset { \mathrm { i i d } } { \sim } \mathrm { U n i f } [ 0 . 8 , 1 . 2 ] . } \end{array}
$$

For the first thirteen features, the coeficients satisfy

$$
\begin{array} { r } { \theta _ { e \ell } = D _ { e \ell } A _ { e \ell } , ~ } \\ { D _ { e \ell } \sim \mathrm { B e r n o u l l i } ( 0 . 5 ) , ~ } \\ { A _ { e \ell } \sim \mathrm { U n i f } [ 0 . 8 , 1 . 2 ] , ~ } \end{array}
$$

independently across arcs and features; the final two coefficients are zero. The signal matrix is generated once per seed and held fixed across the three data splits. Contexts and multiplicative noises are independent across observations. Costs are nonnegative and are used directly as minimization coeficients.

Transportation Model Let $\mathcal { E }$ be the set of permitted arcs. For $e \overset { \cdot } { = } ( i , j ) \in \mathcal { E } ,$ , let $\boldsymbol { x } _ { i j }$ denote the shipment from origin i to destination j and $c _ { i j }$ its realized unit cost. With supplies $a _ { i }$ and demands $b _ { j } .$ , the transportation LP is:

$$
\begin{array} { r l } { \underset { x \geq 0 } { \operatorname* { m i n } } } & { \displaystyle \sum _ { ( i , j ) \in \mathcal { E } } c _ { i j } x _ { i j } , } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { j : ( i , j ) \in \mathcal { E } } x _ { i j } = a _ { i } , \quad i = 1 , \ldots , 8 , } \\ & { \displaystyle \sum _ { i : ( i , j ) \in \mathcal { E } } x _ { i j } = b _ { j } , \quad j = 1 , \ldots , 8 . } \end{array}\tag{57}
$$

Total shipment volume is one. Removing one redundant demand equality gives the standard form

$$
\mathcal { X } _ { \mathrm { t r } } = \{ x \in \mathbb { R } _ { + } ^ { 3 2 } : A x = h \} , \qquad A \in \mathbb { R } ^ { 1 5 \times 3 2 } ,
$$

where A is the balance-constraint matrix of rank 15 and $h \in$ $\mathbb { R } ^ { 1 5 }$ contains the eight supplies and seven retained demands. The feasible region has dimension 17. Each nondegenerate vertex has fifteen positive arcs forming a spanning tree. Pure RR selects such a vertex; Continuous RR allows any feasible shipment plan. For a realized arc-cost vector c, regret is

$$
R ( x ; c ) = c ^ { \top } x - \operatorname* { m i n } _ { v \in \mathcal { X } _ { \mathrm { t r } } } c ^ { \top } v .
$$

Prediction and Calibration Each seed uses 5000 training observations, 2000 calibration observations, and 500 test observations. All six methods share these observations and sample splits. The predictor is multi-output polynomial kernel ridge regression with the kernel, tuning grid, and trainingonly five-fold selection procedure of Appendix B.7. Feature and target standardization are fitted within each training fold. All five seeds select degree 2 and $\lambda = 2 . 1 5 4$ , after which the model is refitted on all 5000 training observations. Predicted costs are returned to the original scale and clipped below at zero.

We use $\alpha = 0 . 1 0$ . Each score has its own threshold, given by order statistic

$$
k _ { \alpha } = \lceil ( 2 0 0 0 + 1 ) ( 1 - 0 . 1 0 ) \rceil = 1 8 0 1
$$

of its calibration scores. Both SPO+ refinements use the same SPO+ scores and threshold. The Residual- $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 } .$ , Box, and KNN–Ellipsoid baselines follow Appendix B.7, with the known support $c \geq 0$ replacing the knapsack sign constraint. Their score definitions, training procedures, and hyperparameters, including $k = 1 0 0$ for KNN, are unchanged.

Active-Piece Generation and Refinement Active pieces are generated from the nominal transportation vertex in nondecreasing predicted-gap order. The adjacency oracle adds a nonbasic arc to the current spanning tree, follows the resulting cycle, and adjusts alternating flows until one tree arc leaves the support. Positive-arc supports identify distinct vertices. As in the knapsack experiment, only vertices satisfying the active-piece screening condition are expanded.

Table 10: Decision performance on the $8 \times 8$ transportation network with 32 arcs $( \alpha = 0 . 1 0 )$ . Entries average per-seed summaries over five seeds, each with 500 test observations. Change is the decision-change rate; Time is mean online time.
<table><tr><td>Method</td><td>Mean Q95</td><td>CVaR95</td><td>Change</td><td>Time (s)</td></tr><tr><td>Nominal PTO</td><td>0.843 3.720</td><td>5.379</td><td>0.000</td><td>0.00202</td></tr><tr><td>Pure RR</td><td>0.851 3.726</td><td>5.380</td><td>0.070</td><td>0.15305</td></tr><tr><td>Continuous RR</td><td>1.2353.466</td><td>4.875</td><td>0.997</td><td>0.54145</td></tr><tr><td>Residual-  $\mathbf { \nabla } \cdot \ell _ { 2 }$  RO</td><td>3.1546.813</td><td>7.967</td><td>0.963</td><td>0.00378</td></tr><tr><td>Box RO</td><td>0.9134.066</td><td>5.661</td><td>0.240</td><td>0.00190</td></tr><tr><td>KNN RO</td><td>3.235 9.609</td><td>12.874</td><td>0.851</td><td>0.00777</td></tr></table>

The generator uses requested prefix budgets

$$
M \in \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 \} ,
$$

extending through a complete equal-gap plateau when needed. Each retained piece supplies a fixed-piece LP adversary with $c \geq 0 .$ . Continuous RR uses an LP master with scenario cuts. Pure RR uses the same cuts and binary arcselection variables $y _ { i j }$ to represent a transportation vertex:

$$
\sum _ { ( i , j ) \in \mathcal { E } } ^ { 0 \leq x _ { i j } } \leq \operatorname* { m i n } ( a _ { i } , b _ { j } ) y _ { i j } , \quad
$$

together with the balance constraints in (57). Thus Pure RR ranges over all transportation vertices and Continuous RR over the full feasible polytope; the prefix restricts the adversarial pieces.

Let LB denote the master lower bound and UB the fullproblem upper bound obtained by combining the adversarial upper bound with the omitted-piece bound in Corollary 1. Refinement stops when UB − LB is at most

$$
1 0 ^ { - 6 } \operatorname* { m a x } \{ 1 , | \mathrm { L B } | , | \mathrm { U B } | \} ,
$$

or the final prefix budget is reached. The constraintgeneration loops have per-instance time budgets of 30 seconds for Pure RR and 20 seconds for Continuous RR. Solver settings follow Appendix B.11, with feasibility and integrality tolerances of $\mathrm { \dot { 1 } 0 ^ { - 9 } }$ for the pure master. Nominal and hindsight LPs use deterministic dual simplex and a fixed secondary objective to resolve optimizer ties.

At the final budget, Pure RR is certified on all 2500 test instances and Continuous RR on 2493 of 2500; neither method reaches its time budget. The performance summaries include the returned feasible decisions for all 500 test instances per seed.

Decision Performance For each seed, we compute mean regret, Q95, CVaR95, the decision-change rate, and mean online time over its 500 test observations. Table 10 reports the arithmetic mean of these per-seed summaries over the five seeds. Time includes amortized prediction and baselinequery costs together with optimization, excluding ofline training and calibration. Pure and Continuous RR run consecutively within each generation round, sharing active-piece search and model construction; their reported times include this common work and their respective optimization stages.

Continuous RR reduces Q95 from 3.720 to 3.466 and CVaR95 from 5.379 to 4.875, a 9.4% reduction in CVaR95, while mean regret increases from 0.843 to 1.235. Pure RR remains close to nominal average performance and has the lowest mean regret among the robust methods. It changes 6.96% of nominal shipment plans, illustrating selective intervention under coupled supply and demand constraints.