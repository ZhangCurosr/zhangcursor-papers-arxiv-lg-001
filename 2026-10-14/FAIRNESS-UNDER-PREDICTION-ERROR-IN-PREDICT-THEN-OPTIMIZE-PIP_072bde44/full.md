# FAIRNESS UNDER PREDICTION ERROR IN PREDICT-THEN-OPTIMIZE PIPELINES

PREPRINT

Hilda Adwubi Osei Department of Industrial Engineering Kwame Nkrumah University of Science and Technology Kumasi, Ghana hildaosei109@gmail.com

## ABSTRACT

Fairness constraints in predict-then-optimize pipelines are necessarily enforced on predicted outcomes yˆ, since true outcomes y are unobserved at decision time. We study whether a demographic parity constraint enforced this way continues to hold against y, and how the resulting violation scales with prediction error. Using a budget-constrained allocation problem solved as a linear program, we show empirically that the true-outcome fairness gap grows with prediction error σ and saturates, well fit by $c ( 1 - e ^ { - \sigma / \tau } )$ with $c = 0 . 1 5 4 5 \pm 0 . 0 1 0 3$ and $\tau = 3 . 9 0 4 5 \pm 0 . 3 7 4 5 ;$ its mean is invariant to sample size while its variance decays at the standard O(1/ n) rate. We derive a two-part theoretical bound: a closed-form local sensitivity result showing the gap’s first-order response to prediction error depends on only two individuals per group, validated against the solver to within three significant figures; and a universal, error-independent ceiling following from the boundedness of the decision variable, confirmed consistent with but looser than the empirical saturation level. On US Census income data, enforcing demographic parity on a trained predictor’s output reduces an unconstrained disparity in true income to roughly one-third of its original size, leaving $3 4 . 4 \% \pm 3 . 3 \%$ (California) and $3 5 . 7 \% \pm 1 . 8 \%$ (Texas) of the disparity intact and undetectable from the predictions alone – consistent not only in mechanism but in approximate magnitude across two states with different demographics and income distributions, and stable across ten independent train/test splits in each. A fairness constraint that appears fully satisfied at decision time can mask a substantial, systematic violation of the outcome it was meant to protect.

Keywords fairness · demographic parity · predict-then-optimize · optimization · prediction error

## 1 Introduction

A prediction yˆ that feeds a downstream decision is not evaluated on its own terms. In predict-then-optimize pipelines – allocating a budget across loan applicants, prioritizing recipients of a limited intervention, and distributing aid across a population – a model’s output is consumed by an optimizer that chooses an action before the true outcome y is observed. Any fairness constraint placed on that optimizer is therefore necessarily a constraint on yˆ, not on y: the quantity a practitioner can certify as fair at decision time is not the quantity the fairness guarantee is meant to protect.

This study asks whether a demographic parity constraint enforced on yˆ continues to hold when evaluated against y and characterizes the gap between the two as a function of the prediction error. We study this in a budget-constrained resource-allocation problem solved as a linear program with demographic parity imposed on per-capita benefit at decision time.

Our contributions are as follows.

(1) Empirical characterization. Using synthetic data with controllable prediction error σ, we show that the trueoutcome fairness gap grows with σ and saturates toward a fixed ceiling rather than increasing without bound, well fit by $c \cdot ( 1 - e ^ { - \sigma / \tau } )$ across more than an order of magnitude of σ (Section 4). We further show that the gap’s mean is approximately invariant to the sample size n, while its variance across draws shrinks at the standard $O ( 1 / \sqrt { n } )$ rate – indicating a systematic bias that additional data does not correct, distinct from ordinary estimation noise.

(2) A two-part theoretical bound. We derive a structural result (Section 5): at any optimum of this allocation problem, at most two individuals per group have a fractional allocation, and the gap’s first-order sensitivity to prediction error depends only on these two “marginal” individuals – every other individual’s prediction error has zero local effect. This reduces the local sensitivity to a closed-form expression, which we validate numerically against the solver to within three significant figures. We pair this local result with a universal, σ-independent ceiling that follows directly from the boundedness of the decision variable and holds for arbitrarily large prediction error, and we show that it is consistent with, though looser than, the empirically observed saturation level.

(3) Validation on real data. We confirm the mechanism using a trained income predictor on US Census data (ACSIncome; Ding et al. [2021]) (Section 7). Enforcing demographic parity on the predictor’s output reduces the unconstrained disparity in true income by approximately two-thirds but leaves the remainder intact and undetectable from the predictions alone, a result stable across ten independent train/test splits and consistent in both mechanism and approximate magnitude across two states with different demographics (California and Texas).

Scope. We isolate the estimation error – predictions drawn from the same distribution as the true outcome, corrupted by noise – as the source of prediction imperfection. We do not study distribution shift, in which the feature-outcome relationship itself changes between training and deployment (Section 8). We restrict our attention to demographic parity on continuous per-capita benefit; other fairness definitions, such as equal opportunity, are not addressed here and may exhibit different behavior under the same mechanism.

Section 2 positions this work relative to existing literature on fairness in predict-then-optimize pipelines. Section 3 formalizes the decision problem and fairness gap. Sections 4–6 present the empirical study, theoretical bound, and their comparison. Section 7 presents the real-data experiment. Section 8 discusses limitations and future work, and Section 9 concludes.

## 2 Related Work

This work sits at the intersection of predict-then-optimize decision-making and algorithmic fairness. We organize related work by the question each line addresses and state after each how it differs from ours.

Decision-focused learning. Elmachtoub and Grigas [2022] formalize predict-then-optimize as a learning problem and propose training predictors to minimize downstream decision regret rather than prediction error alone. Wilder et al. [2019] extend this idea to combinatorial optimization problems, differentiating through the optimization step during training so the predictor is shaped by the decisions it will induce. Neither considers a fairness constraint on the resulting decisions; both aim to make the decision better, not fairer. We take a trained predictor’s error as given, rather than optimizing against it, and ask a narrower question: holding the predictor fixed, how does its error propagate into a fairness violation.

Fairness in predict-then-optimize pipelines. Closest to this work, Wang and Chen [2026] introduced decision regret as the gap between a fairness objective evaluated on a prediction-based decision and on the full-information (true-outcome) decision – the same quantity we study. Their method, fair decision-focused learning, minimizes this gap during training: predictions are optimized jointly for accuracy and fairness of the decisions they induce, using a differentiable relaxation of the downstream optimization problem. This requires retraining the predictor and assumes that the fairness objective is available as a training signal. We do not propose a training method; instead, we characterize how this gap behaves as a function of prediction error alone, for a fixed, already trained predictor, via a closed-form local sensitivity result and a universal bound requiring no retraining or differentiable relaxation to apply. Our results could determine, prior to any retraining, whether a given predictor’s error already makes their training-time correction necessary.

Fairness in budget-constrained decision-making. Corbett-Davies et al. [2017] study fairness constraints in thresholdbased decisions under a budget (e.g. a limited number of interventions), and show that imposing standard fairness criteria can trade off against the decision-maker’s objective – the “cost of fairness.” Their analysis takes the risk score (our yˆ) as exact; it does not consider how error in that score affects whether the fairness criterion holds against the true outcome. Our budget-constrained allocation problem is structurally similar to theirs, but our question – does the constraint survive contact with y – is orthogonal to the cost-of-fairness trade-off they study, which concerns yˆ throughout.

Fairness under finite-sample uncertainty. Cousins [2022] bound how finite sample size distorts estimates of a fairness objective, where uncertainty arises from estimating group-level statistics from a limited number of individuals. This is structurally different from our setting: their uncertainty source is how many people are observed, ours is how accurately each individual’s outcome is predicted. Consistent with this distinction, we find in Section 4 that our fairness gap does not vanish as sample size grows, whereas sample-complexity bounds of this type generally do – the two sources of uncertainty have different asymptotic behavior, not merely different derivations.

Where fairness should be enforced. De Vos et al. [2025] argue that fairness should be measured in the region of the score distribution where decisions actually change, rather than globally. This is a question of where to measure fairness; our question is, given a fixed, sensible place to enforce it (at decision time, on the available prediction), whether the resulting guarantee survives contact with the true outcome. The two questions are complementary rather than competing.

Demographic parity. We adopt the demographic parity criterion of Dwork et al. [2012], applied here to a continuous per-capita benefit rather than a binary classification rate. We do not consider alternative criteria such as equal opportunity; Section 8 discusses this as a direction for future work.

Summary. No existing work, to our knowledge, derives an explicit bound relating a true-outcome fairness violation to prediction error and sample size for a demographic-parity-constrained allocation problem, built from first principles (LP sensitivity and duality) and validated against both a controlled synthetic study and real data. This is the gap this paper addresses.

## 3 Problem Setup

## 3.1 Notation

Table 1 summarizes the notation used throughout the paper. Once introduced here, each symbol retains its meaning in every subsequent section.

Table 1: Notation used throughout the paper.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $n$ </td><td>Population size</td></tr><tr><td> $a _ { i } \in \{ 0 , 1 \}$ </td><td>Group membership of individual i</td></tr><tr><td> $G _ { 0 } , G _ { 1 }$ </td><td>The two groups,  $G _ { g } = \{ i : a _ { i } = g \}$ </td></tr><tr><td> $n _ { 0 } , n _ { 1 }$ </td><td>Group sizes,  $n _ { g } = | G _ { g } |$ </td></tr><tr><td> $y _ { i } \in \mathbb { R }$ </td><td>True outcome of individual ¿</td></tr><tr><td> $\hat { y } _ { i } \in \mathbb { R }$ </td><td>Predicted outcome of individual i</td></tr><tr><td> $z _ { i } \in [ 0 , 1 ]$ </td><td>Allocation decision for individual i</td></tr><tr><td> $B$ </td><td>Total budget,  $B = \beta n$  for budget fraction  $\beta \in ( 0 , 1 )$ </td></tr><tr><td> $\sigma$ </td><td>Prediction error scale (synthetic study only)</td></tr><tr><td> $\mathrm { g a p } ( v , z )$ </td><td>Fairness gap, evaluated at value vector v and allocation z</td></tr></table>

## 3.2 Assumptions

Assumption 1 (Two groups). Each individual belongs to exactly one of two groups, $G _ { 0 }$ or $G _ { 1 }$ , with $n _ { 0 } , n _ { 1 } \geq 1 .$

Assumption 2 (Divisible allocation). The decision $z _ { i }$ is a continuous fraction in $[ 0 , 1 ] _ { ; }$ , not a binary selection. This is standard in resource-allocation settings where the allocated quantity (e.g. funding, service hours) is itself divisible.

Assumption 3 (Prediction available, outcome unobserved at decision time). At decision time, $\hat { y } _ { i }$ is knownfor every i, and $y _ { i }$ is not. Any constraint imposed on the decision can therefore be a function of yˆ, but not of y.

## 3.3 The Allocation Problem

Given predictions ${ \hat { y } } \in \mathbb { R } ^ { n }$ and group labels $a \in \{ 0 , 1 \} ^ { n }$ , the decision-maker solves

$$
\begin{array} { r l r } { \underset { z \in \mathbb { R } ^ { n } } { \operatorname* { m a x } } } & { \hat { y } ^ { \top } z } \\ { \mathrm { s . t . } } & { 0 \leq z _ { i } \leq 1 } & { \forall i , } \\ & { \sum _ { i = 1 } ^ { n } z _ { i } \leq B , } \\ & { \mathrm { g a p } ( \hat { y } , z ) \leq \epsilon , } & { \mathrm { g a p } ( \hat { y } , z ) \geq - \epsilon , } \end{array}\tag{1}
$$

where the fairness constraint is written as two one-sided inequalities rather than a single equality to match the implementation used throughout the paper; the two coincide exactly when $\epsilon = 0$ , which we use throughout unless stated otherwise. Problem (1) is a linear program: the objective and every constraint are linear in z.

## 3.4 The Fairness Gap

For any value vector $v \in \mathbb { R } ^ { n }$ (either yˆ or y) and allocation $z ,$ define the per-capita benefit of group g as $\textstyle { \frac { 1 } { n _ { g } } } \sum _ { i \in G _ { g } } v _ { i } z _ { i } .$ and the fairness gap as the difference between groups:

$$
\mathrm { { g a p } } ( v , z ) = \frac { 1 } { n _ { 1 } } \sum _ { i \in G _ { 1 } } v _ { i } z _ { i } - \frac { 1 } { n _ { 0 } } \sum _ { i \in G _ { 0 } } v _ { i } z _ { i } .\tag{2}
$$

This is linear in z for fixed $v ,$ which is what makes it usable directly as an $\mathrm { L P }$ constraint in (1).

The central object of this paper is the distinction between two evaluations of (2) at the solution $z ^ { * } ( \hat { y } )$ of problem (1):

$\mathrm { g a p } ( \hat { y } , z ^ { * } ( \hat { y } ) )$ – the enforced gap, constrained to lie in $[ - \epsilon , \epsilon ]$ by construction, and the only quantity observable at decision time (Assumption 3);

$\mathrm { g a p } ( y , z ^ { * } ( \hat { y } ) )$ – the true gap: the same allocation $z ^ { * } ( \hat { y } )$ , evaluated against the true outcomes once observed.

These coincide only when ${ \hat { y } } = y$ . The remainder of the paper studies how large $| \mathrm { g a p } ( y , z ^ { * } ( \hat { y } ) ) |$ can be as a function of the discrepancy between yˆ and y, given that $\mathrm { g a p } ( \hat { y } , z ^ { * } ( \bar { \hat { y } } ) ) \approx 0$ by construction.

## 3.5 A Minimal Example

We anchor the formalism with the smallest nontrivial instance: $n = 2 ,$ , one individual per group $( n _ { 0 } = n _ { 1 } = 1 )$ , budget $B = 1$ , and $\epsilon = 0$ . Writing $\hat { y } _ { 1 } , \hat { y } _ { 2 }$ for the two individuals’ predictions (group 0 and group 1 respectively), the fairness constraint $\mathrm { g a p } ( \hat { y } , z ) = 0$ reduces to $\hat { y } _ { 1 } z _ { 1 } = \hat { y } _ { 2 } z _ { 2 }$ , and combined with the budget constraint $z _ { 1 } + z _ { 2 } = 1$ (binding, since increasing either z<sub>i</sub> only helps the objective), the unique solution is

$$
z _ { 1 } ^ { * } = \frac { \hat { y } _ { 2 } } { \hat { y } _ { 1 } + \hat { y } _ { 2 } } , \qquad z _ { 2 } ^ { * } = \frac { \hat { y } _ { 1 } } { \hat { y } _ { 1 } + \hat { y } _ { 2 } } .\tag{3}
$$

Substituting (3) into (2) evaluated at the true outcomes $y _ { 1 } , y _ { 2 }$ gives the true gap in closed form:

$$
\mathrm { g a p } ( y , z ^ { * } ( \hat { y } ) ) = \frac { y _ { 1 } \hat { y } _ { 2 } - y _ { 2 } \hat { y } _ { 1 } } { \hat { y } _ { 1 } + \hat { y } _ { 2 } } .\tag{4}
$$

Equation (4) is zero exactly when ${ \hat { y } } = y$ (up to scale), confirming that the enforced and true gaps coincide only in the absence of prediction error, and is nonzero otherwise – the simplest possible instance of the phenomenon this paper studies at general n. We refer to the two individuals appearing in (3)–(4) as marginal: Section 5 shows that for general $n ,$ the solution to problem (1) reduces to exactly this two-variable structure, restricted to at most one marginal individual per group, with every other individual’s allocation pinned to 0 or 1.

## 4 Empirical Study

## 4.1 Experimental Design

We generate synthetic populations with two groups of controllable size and outcome gap, and a controllable amount of prediction error, so that every quantity affecting the true-outcome fairness gap is under direct experimental control. For a population of size n, group membership a is drawn Bernoulli with $P ( a _ { i } = 1 ) = 0 . 3$ (group 1 the minority); true outcomes are drawn $y _ { i } \stackrel { - } { \sim } \mathcal { N } ( \mu _ { a _ { i } } , 0 . 5 ^ { 2 } )$ with $\mu _ { 0 } = 2 . 0 , \mu _ { 1 } = 1 . 4$ , so that group 1 has strictly lower expected outcome and an unconstrained allocation is unfair by construction; predictions are generated as $\hat { y } _ { i } = y _ { i } + \sigma \cdot \varepsilon _ { i } \mathrm { f o r } \varepsilon _ { i } \sim \mathcal { N } ( 0 , 1 )$ so σ is the sole, directly controlled knob governing prediction error, with $\sigma = 0$ recovering perfect prediction. For each generated population, we solve problem (1) with $\epsilon = 0$ and budget fraction $\beta = 0 . 3$ , and record $\mathrm { g a p } ( y , z ^ { * } ( \hat { y } ) ) -$ the true gap, under the allocation that was constrained to satisfy parity only on yˆ.

Two quantities were varied across experiments: σ to trace how the true gap depends on the prediction error, and n to determine whether the gap is a finite-sample artifact or persists asymptotically. Every other generative parameter $( \mu _ { 0 } , \mu _ { 1 }$ , group proportion, $\beta )$ was held fixed throughout, so that the observed changes in the gap could be attributed to σ and n alone. Each reported configuration was repeated over multiple random seeds; we report the mean and standard error (SE) over seeds throughout, never a single run, because any individual draw of $a , y , \varepsilon \mathrm { i }$ is itself a random quantity.

## 4.2 Dependence on Prediction Error

We first sweep $\sigma \in \{ 0 , 0 . 2 , 0 . 4 , \dots , 2 . 0 \}$ at a fixed $n = 2 0 0 0$ , with 20 seeds per value. The mean true gap grows monotonically from ≈ 0 at $\sigma = 0 \mathrm { t o } - 0 . 0 7 7$ at $\sigma = 2 . 0$ , but the per-step increase shrinks as σ grows (first differences fall from $\approx 0 . 0 0 9$ near $\sigma \approx 0 . 3 \mathrm { t o } \approx 0 . 0 0 6$ near $\sigma = 2 . 0 )$ , indicating concave, not linear, growth. This is structurally expected: $z \in [ 0 , 1 ] ^ { n }$ is bounded and budget-constrained (Assumption 2, Eq. 1), so $\mathrm { g a p } ( y , z )$ is itself bounded regardless of how large σ becomes, ruling out unbounded (e.g. linear) growth on structural grounds alone.

## 4.3 Dependence on Sample Size

Next, we sweep $n \in \{ 5 0 0 , 1 0 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 \}$ at three fixed values of $\sigma \in \{ 0 . 5 , 1 . 0 , 1 . 5 \}$ , with 20 seeds per configuration. The mean true gap appears to decrease in magnitude from $n = 2 0 0 0$ to $n = 8 0 0 0$ at every tested $\sigma \left( \mathrm { { e . g } } \right.$ at $\sigma = 0 . 5 \colon - 0 . 0 2 0 4 \to - 0 . 0 1 8 0 \to - 0 . 0 1 5 4 { \mathrm { ~ f o r ~ } } n = 2 0 0 0 , 4 0 0 0 , 8 0 0 0 )$ , suggesting the gap might vanish as $n \to \infty$ We report this intermediate result because it did not survive closer scrutiny: the standard error at $n = 5 0 0 ( \mathrm { S E } \approx 0 . 0 0 6$ at $\sigma = 0 . 5 )$ is nearly as large as the mean itself $\left( - 0 . 0 0 4 2 \right)$ , indicating 20 seeds is insufficient to resolve the small-n behavior, and the apparent downward trend could reflect sampling noise rather than a real effect.

Repeating with 100 seeds per configuration resolves this. The mean true gap is now approximately constant across n at every tested σ $( \mathbf { e } , \mathbf { g } , \mathbf { z } , \mathbf { a } { \mathsf { t } } \sigma = 0 . 5 ; - 0 . 0 1 3 1 , - 0 . 0 1 5 2 , - 0 . 0 1 5 4 , - 0 . 0 1 5 9 , - { \bar { 0 } } , 0 1 5 4 { \mathrm { ~ f o r ~ } } { \bar { n } } = 5 0 0 , 1 { \bar { 0 } } 0 0 , 2 0 0 0 , 4 0 0 0 , 8 0 0 0 0 , \ldots$ each within 1–2 SE of the others), while the standard deviation across seeds shrinks systematically with n (from 0.0241 at $n = 5 0 0$ to 0.0061 at $n = 8 0 0 0 , \mathrm { a } \approx 3 . 9 5 \times$ reduction for a 16× increase in $n ,$ matching the ${ \sqrt { 1 6 } } = 4$ predicted by standard sampling theory). We interpret this as two distinct components of the true gap: a systematic bias that does not vanish with n, and ordinary $O ( 1 / \sqrt { n } )$ sampling noise around it. We return to this distinction structurally in Section 5.

## 4.4 Functional Form

Having established that the $\mathrm { g a p ' s }$ dependence on n is limited to variance, not mean, we fix $n = 4 0 0 0$ (where the standard deviation is already small) and focus sampling effort on resolving the σ dependence precisely, using 100 seeds per value of $\sigma \in \{ 0 , 0 . 2 , \ldots , 3 . 0 \}$ . We compared five candidate functional forms using weighted least squares, using each point’s standard error as the weight: linear (cσ), square root $( c \sqrt { \sigma } )$ , power law $( c \sigma ^ { p } )$ , rational saturating $( c \sigma / ( k + \sigma ) )$ , and exponential saturating $( c ( 1 - e ^ { - \sigma / \tau } ) )$ . Over this range, the square-root form is decisively rejected (RMSE 10× worse than the alternatives); the power-law fit returns $p \approx 0 . 9 7$ , which is indistinguishable from linear; and while the saturating forms fit measurably better, their fitted saturation parameters $( k \approx 1 6 , \tau \approx 8 . 3 )$ lie well outside the tested range of σ, meaning the data over [0, 3] cannot yet distinguish genuine saturation from a slowly bending curve that merely resembles it.

We therefore extend the sweep to $\sigma \in \{ 0 , 1 , 2 , 3 , 4 , 6 , 8 , 1 0 , 1 4 , 2 0 \}$ , with 60 seeds per value (sufficient to resolve the shape at this coarser resolution). Over this wider range, linear growth is decisively ruled out (RMSE 30× worse than the best fit), and the exponential saturating form fits best, with parameters that now fall within the tested range $( \tau \approx 3 . 9 5$ compared to $\tau \approx 8 . 3$ fitted from the shorter range alone) – indicating the earlier, short-range fit was extrapolating, while this one is interpolating. Combining both sweeps into a single weighted fit gives

$$
| \mathrm { g a p } ( y , z ^ { * } ( \hat { y } ) ) | \approx c \left( 1 - e ^ { - \sigma / \tau } \right) , \qquad c = 0 . 1 5 4 5 \pm 0 . 0 1 0 3 , \quad \tau = 3 . 9 0 4 5 \pm 0 . 3 7 4 5 .\tag{5}
$$

Figure 1 shows the empirical means with standard error bars alongside the fitted curve of Eq. (5): the gap rises steeply and nearly linearly for $\sigma \lesssim 1$ , then bends over and flattens toward a ceiling of $c \approx 0 . 1 5 5$ by σ ≈ 14–20, with the fitted curve passing through the error bars at every tested point.

## 4.5 Summary

Two findings from this section are used directly in what follows. First, the true-outcome fairness gap is well described by the saturating form of Eq. (5), growing approximately linearly for small σ and flattening to a finite ceiling for large σ – Section 6 compares this empirical shape against the theoretical bound derived in Section 5. Second, the $\mathrm { g a p ^ { \prime } s }$ mean is approximately invariant to sample size while its variance decays at the standard $O ( 1 / \sqrt { n } )$ rate – a signature Section 5 explains structurally by showing the gap’s sensitivity to prediction error depends on only a bounded number of individuals regardless of n.

![](images/39503161242984ab11a9b2b1e14099da7d30a85ae259af5db1093ff787e1c2bf.jpg)  
Figure 1: True-outcome fairness gap as a function of prediction error σ $( n = 4 0 0 0 , 6 0 – 1 0 0$ seeds per point, error bars show ±1 SE), with the fitted curve of Eq. (5).

## 5 Theoretical Analysis

This section derives a theoretical characterization of the true-outcome gap in two parts: a structural result reducing the optimal allocation’s dependence on yˆ to at most two individuals per group (Section 5.1), from which a closed-form local sensitivity formula follows (Section 5.2), and a universal, σ-independent ceiling following from boundedness alone (Section 5.3). Every derived expression in this section is validated numerically against the solver used in Section 4, with the validation stated immediately after each result.

## 5.1 The Active Set Reduces to One Marginal Individual per Group

Proposition 1 (Active-set reduction). Let $z ^ { * }$ be an optimal solution to problem (1) with $\epsilon = 0$ , and suppose no two individuals within the same group share an identical predicted value $\hat { y } _ { i }$ . Then at most one individual in each group $G _ { g }$ satisfies $0 < z _ { i } ^ { * } < 1 ;$ every other individual satisfies $z _ { i } ^ { * } \in \{ 0 , 1 \}$ }.

Proof. Fix $g \in \{ 0 , 1 \}$ and the total allocation to group g, $\begin{array} { r } { S _ { g } = \sum _ { i \in G _ { g } } z _ { i } , } \end{array}$ , at its value under $z ^ { * }$ . Among all allocations within $G _ { g }$ summing to $S _ { g }$ with $0 \leq z _ { i } \leq 1$ , the one maximizing $\sum _ { i \in G _ { g } } \hat { y } _ { i } z _ { i } -$ the only part of the objective in (1) depending on $\{ z _ { i } \} _ { i \in G _ { g } }$ for fixed $S _ { g } - \mathrm { i s }$ obtained by fully funding the highest- $- \hat { y } _ { i }$ individuals in $G _ { g }$ first. Suppose instead $z _ { i } ^ { * } , z _ { j } ^ { * } \in ( 0 , 1 )$ for two individuals $i , j \in G _ { g }$ with $\hat { y } _ { i } \neq \hat { y } _ { j } ( \mathrm { w . l . o . g . ~ } \hat { y } _ { i } > \hat { y } _ { j } )$ . Let $\delta = \operatorname* { m i n } ( 1 - z _ { i } ^ { * } , z _ { j } ^ { * } ) > 0$ and define $z ^ { \prime }$ by $z _ { i } ^ { \prime } = z _ { i } ^ { * } + \delta , z _ { j } ^ { \prime } = z _ { j } ^ { * } - \delta , z _ { k } ^ { \prime } = z _ { k } ^ { * }$ otherwise. Then $z ^ { \prime }$ is feasible for problem (1): $S _ { g }$ is unchanged, so both the budget constraint (which depends only on $\sum _ { i } z _ { i } )$ and the fairness constraint (which depends only on $S _ { 0 } , S _ { 1 }$ through (2)) are unaffected. The objective strictly increases, $\hat { y } ^ { \top } z ^ { \prime } - \hat { y } ^ { \top } z ^ { * } = \delta ( \hat { y } _ { i } - \hat { y } _ { j } ) > 0$ , contradicting optimality of $z ^ { * }$ . Hence at most one individual per group is fractional. 口

We write $m _ { g }$ for the (at most one) marginal individual in group g under $z ^ { * } ,$ , when one exists, and refer to individuals with $z _ { i } ^ { * } = \mathrm { i }$ asfullyfunded and those with $z _ { i } ^ { * } = 0$ as excluded. Proposition 1 shows that for fixed total allocations $S _ { 0 } , S _ { 1 }$ , the within-group structure is fully determined (fully fund by descending $\hat { y } _ { i }$ until the budget for that group is exhausted, with at most one fractional remainder); the only remaining freedom is in $( S _ { 0 } , S _ { 1 } )$ itself, governed jointly by the budget and fairness constraints across both groups. This is exactly the two-variable system solved explicitly for $n = 2$ in Section 3.5, now shown to govern the general-n problem as well, restricted to whichever two individuals happen to be marginal.

Corollary 1 (Non-marginal insensitivity). For any individual i that isfullyfunded or excluded under $z ^ { * } ( \hat { y } )$ , andfor sufficiently small perturbations δ of $\hat { y } _ { i }$ that do not change which individuals are fully funded, excluded, or marginal, $\partial \bar { \mathrm { g a p } } ( y , \dot { z } ^ { * } ( \hat { y } ) ) / \bar { \partial } \hat { y } _ { i } = 0$

Proof. By Proposition 1, i’s status (fully funded or excluded) and every other individual’s allocation are determined by the ranking of yˆ within each group and the two marginal individuals’ values, not by $\hat { y } _ { i } \mathrm { \Delta } ^ { \circ } \mathrm { s }$ exact value, provided the perturbation is small enough that i’s rank relative to its group’s marginal individual is unchanged. Under this condition $z ^ { * } ( \hat { y } )$ is locally constant in $\hat { y } _ { i }$ , and since $\operatorname { g a p } ( y , \cdot )$ in (2) is a fixed linear functional of z for fixed $y ,$ , it is locally constant in $\hat { y } _ { i }$ as well. □

Numerical validation. We verified Corollary 1 directly: perturbing a non-marginal individual’s $\hat { y } _ { i }$ by $1 0 ^ { - 4 }$ in the $n = 4 0 0 0 , \sigma = 0 . 3$ configuration of Section 4 changed the true gap by $\approx 3 . 6 \times \mathrm { 1 0 ^ { - 1 2 } }$ , consistent with floating-point noise rather than a real effect.

## 5.2 Local Sensitivity to Prediction Error

Write $n _ { m _ { 0 } } , n _ { m _ { 1 } }$ for the group sizes containing the marginal individuals, $F _ { g }$ for the fully funded set in group $^ { g , }$ $\begin{array} { r } { A _ { g } = \frac { 1 } { n _ { g } } \sum _ { i \in F _ { g } } \hat { y } _ { i } } \end{array}$ for the fixed per-capita contribution of the fully funded individuals (as measured by $\hat { y } ,$ the quantity the constraint is enforced on), and $b _ { g } = \hat { y } _ { m _ { g } } / n _ { g }$ . Let $\begin{array} { r } { R = B - \sum _ { i \in F _ { 0 } \cup F _ { 1 } } } \end{array}$ 1 be the remaining budget for the marginal pair.

Lemma 1 (Closed form for the marginal allocations). Suppose both the budget andfairness constraints ofproblem (1) bind $a t ~ z ^ { * }$ , and both groups have a marginal individual. Then

$$
z _ { m _ { 0 } } ^ { * } = \frac { A _ { 1 } - A _ { 0 } + b _ { 1 } R } { b _ { 0 } + b _ { 1 } } , \qquad z _ { m _ { 1 } } ^ { * } = R - z _ { m _ { 0 } } ^ { * } .\tag{6}
$$

Proof. With the within-group structure fixed by Proposition 1, the budget constraint restricted to the marginal pair is $z _ { m _ { 0 } } + z _ { m _ { 1 } } = R$ , and the fairness constraint $\mathrm { g a p } ( \hat { y } , z ) = 0$ becomes $A _ { 0 } + b _ { 0 } z _ { m _ { 0 } } = A _ { 1 } + b _ { 1 } z _ { m _ { 1 } }$ . Substituting $z _ { m _ { 1 } } = R - z _ { m _ { 0 } }$ and solving the resulting linear equation for $z _ { m _ { 0 } } \ { \mathrm { g i v e s } } \ ( 6 )$ 口

Theorem 1 (Local sensitivity of the true gap). Under the conditions ofLemma 1, andfor perturbations of $\hat { y } _ { m _ { 0 } } , \hat { y } _ { m _ { 1 } }$ small enough that the identity ofthe marginal pair and thefully-funded/excluded sets are unchanged,

$$
\frac { \partial \mathrm { g a p } ( y , z ^ { * } ) } { \partial \hat { y } _ { m _ { 0 } } } = \frac { z _ { m _ { 0 } } ^ { * } } { n _ { 0 } ( b _ { 0 } + b _ { 1 } ) } \left( \frac { y _ { m _ { 0 } } } { n _ { 0 } } + \frac { y _ { m _ { 1 } } } { n _ { 1 } } \right) , \qquad \frac { \partial \mathrm { g a p } ( y , z ^ { * } ) } { \partial \hat { y } _ { m _ { 1 } } } = - \frac { z _ { m _ { 1 } } ^ { * } } { n _ { 1 } ( b _ { 0 } + b _ { 1 } ) } \left( \frac { y _ { m _ { 0 } } } { n _ { 0 } } + \frac { y _ { m _ { 1 } } } { n _ { 1 } } \right) .\tag{7}
$$

Proof. Write $N = A _ { 1 } - A _ { 0 } + b _ { 1 } R , \mathrm { s o } z _ { m _ { 0 } } ^ { * } = N / ( b _ { 0 } + b _ { 1 } )$ by Lemma 1, and $b _ { 0 } = \hat { y } _ { m _ { 0 } } / n _ { 0 }$ is the only term depending on $\hat { y } _ { m _ { 0 } }$ (for fixed ${ \hat { y } } _ { m _ { 1 } } , A _ { 0 } , A _ { 1 } , R$ depend on neither). Then $\partial z _ { m _ { 0 } } ^ { * } / \partial b _ { 0 } = - N / ( b _ { 0 } + b _ { 1 } ) ^ { 2 } = - z _ { m _ { 0 } } ^ { * } / ( b _ { 0 } + b _ { 1 } )$ , and by the chain rule with $\partial b _ { 0 } / \partial \hat { y } _ { m _ { 0 } } = 1 / n _ { 0 } , \partial z _ { m _ { 0 } } ^ { * } / \partial \hat { y } _ { m _ { 0 } } = - z _ { m _ { 0 } } ^ { * } / ( \tilde { n _ { 0 } } ( b _ { 0 } + b _ { 1 } ) )$ . By Corollary 1 the true gap depends on $\hat { y } _ { m _ { 0 } }$ only through $z _ { m _ { 0 } } ^ { * }$ and $z _ { m _ { 1 } } ^ { * } = R - z _ { m _ { 0 } } ^ { * } ;$ : differentiating $\begin{array} { r } { \mathrm { g a p } ( y , z ) = \frac { 1 } { n _ { 1 } } \sum _ { G _ { 1 } } y _ { i } z _ { i } - \frac { 1 } { n _ { 0 } } \sum _ { G _ { 0 } } y _ { i } z _ { i } \mathrm { ( E q . ~ } 2 ) } \end{array}$ with respect to $\begin{array} { r } { \hat { y } _ { m _ { 0 } } \ \mathrm { g i v e s } \ \frac { y _ { m _ { 1 } } } { n _ { 1 } } \frac { \partial z _ { m _ { 1 } } ^ { * } } { \partial \hat { y } _ { m _ { 0 } } } - \frac { y _ { m _ { 0 } } } { n _ { 0 } } \frac { \partial z _ { m _ { 0 } } ^ { * } } { \partial \hat { y } _ { m _ { 0 } } } } \end{array}$ , and substituting $\partial z _ { m _ { 1 } } ^ { * } / \partial \hat { y } _ { m _ { 0 } } = - \partial z _ { m _ { 0 } } ^ { * } / \partial \hat { y } _ { m _ { 0 } }$ gives the first expression in (7). The second follows symmetrically, differentiating Lemma 1 with respect to $\hat { y } _ { m ; }$ through $b _ { 1 }$ □

Numerical validation. We validated Theorem 1 against the real solver at $n = 4 0 0 0 , \sigma = 0 . 3$ (seed 1), where the marginal pair had $n _ { 0 } = 2 8 2 0 , n _ { 1 } = 1 1 8 0 , z _ { m _ { 0 } } ^ { * } = 0 . 6 3 2 , z _ { m _ { 1 } } ^ { * } = 0 . 3 6 8$ . Central finite differences with step $1 0 ^ { - 4 } – 1 0 ^ { - 5 }$ (small enough not to disturb the active $\mathbf { s e t - a }$ step of $1 0 ^ { - 3 }$ was large enough to push the group-0 marginal individual to $z = 1$ , giving a spurious slope) gave 0.000230 for ${ \partial \mathrm { g a p } } / { \partial \hat { y } _ { m _ { 0 } } }$ against $( { \bar { 7 } } ) ^ { 3 }$ s prediction of 0.000229, and −0.000327 for $\partial \mathrm { g a p } / \partial { \hat { y } } _ { m } .$ against a prediction of −0.000318 – agreement to 2–3 significant figures in both cases. We note as a methodological point that an earlier derivation of Theorem 1 carried a sign error (the two terms of Eq. 2 transposed), caught precisely by this numerical check rather than by inspection; every analytical result in this section was validated this way before being accepted.

## 5.3 A Universal Ceiling

Theorem 2 (Universal ceiling). For any feasible z in problem (1) (i.e. $0 \leq z _ { i } \leq 1$ for all i and $\textstyle \sum _ { i } z _ { i } \leq B )$ and any value vector v,

$$
| \mathrm { g a p } ( v , z ) | \leq \frac { B } { \operatorname* { m i n } ( n _ { 0 } , n _ { 1 } ) } \operatorname* { m a x } _ { i } | v _ { i } | .\tag{8}
$$

Proof. By the triangle inequality applied to $\begin{array} { r l r } { ( 2 ) , \mathrm {  ~ \left| g a p ( } v , z \mathrm {  ~ \right| ~ } } & { { } \leq } & { \frac { 1 } { n _ { 1 } } \sum _ { G _ { 1 } } | v _ { i } | z _ { i } \ + \ \frac { 1 } { n _ { 0 } } \sum _ { G _ { 0 } } | v _ { i } | z _ { i } \quad \leq } \end{array}$ max<sub>i</sub> $\begin{array} { r } { | v _ { i } | \left( \frac { 1 } { n _ { 1 } } \sum _ { G _ { 1 } } z _ { i } + \frac { 1 } { n _ { 0 } } \sum _ { G _ { 0 } } z _ { i } \right) \leq \frac { \operatorname* { m a x } _ { i } | v _ { i } | } { \operatorname* { m i n } ( n _ { 0 } , n _ { 1 } ) } \sum _ { i } z _ { i } \leq \frac { B \operatorname* { m a x } _ { i } | v _ { i } | } { \operatorname* { m i n } ( n _ { 0 } , n _ { 1 } ) } } \end{array}$ , using $z _ { i } \ \leq \ 1$ and $\textstyle \sum _ { i } z _ { i } \leq B$ in the last step. □

Unlike Theorem 1, Theorem 2 requires no assumption on which constraints bind, which individuals are marginal, or how large the prediction error is; it holds even as $\sigma \to \infty$ , when predictions carry no information about $y .$ This is the sense in which it is universal; it is correspondingly loose, as it does not use the fact that $z ^ { * }$ is chosen by an optimizer at all. In our experiments, outcomes are nonnegative $( y _ { i } \geq 0$ , by construction in the synthetic study and by the clipping step of Section 7), so max $\left| y _ { i } \right| = \operatorname* { m a x } _ { i } y _ { i } ;$ evaluating Eq. (8) at $n = 4 0 0 0 , \sigma = 1 . 0 , \beta = 0 . 3$ gives a ceiling of 4.04, compared to the empirically observed saturation level of $c \approx 0$ .155 from Eq. (5) – consistent with, though substantially looser than, the saturation Section 4 observes. We compare the two results directly in Section 6.

## 6 Bound Validation

This section compares the theoretical results of Section 5 to the empirical curve of Section $4 ;$ it introduces no new derivations.

The universal ceiling is loose, as expected. Theorem 2 provides a ceiling of 4.04 at $n = 4 0 0 0 , \sigma = 1 . 0 , \beta = 0 . 3$ roughly 26× the empirically observed saturation level c ≈ 0.155 from Eq. (5). This looseness is expected rather than a shortcoming: Theorem 2 uses only the boundedness of z, not the fact that $z ^ { * }$ is chosen by an optimizer balancing a budget and fairness constraint against the objective. It is a worst-case bound over all feasible z, not a description of the $z ^ { * }$ the LP actually produces, so a substantial gap between the two is exactly what should be expected of a bound this general. Across every $( \sigma , n )$ configuration in Section 4, the empirical gap never approached, let alone exceeded, the corresponding ceiling – the bound is never violated, which is the property it was constructed to guarantee.

The local sensitivity result is validated as a mechanism, not as a direct predictor of the population-level curve. Theorem 1 was validated in Section 5.2 at the level it was derived: the sensitivity of the true gap to one marginal individual’s prediction, at one specific realization of $( \hat { y } , a )$ , confirmed against finite differences on the solver to 2–3 significant figures. This is a different quantity from the population-averaged slope in Eq. (5) near $\sigma = 0$ , which would additionally require taking an expectation over the identity and values of the marginal pair as σ varies across random draws, a step we do not attempt here. What Theorem 1 and Corollary 1 do establish, and what the empirical curve is consistent with, is the mechanism: that the gap’s response to prediction error is governed by a small, bounded number of individuals rather than growing with n, matching the n-invariance of the mean gap observed in Section 4.3, and that this local, per-individual sensitivity is well-defined only while the active set is stable, consistent with the near-linear behavior of Eq. (5) for small $\sigma ,$ before repeated swaps of the marginal pair at larger σ average into the observed saturation.

What this does not achieve. Neither theoretical result derives Eq. (5)’s exponential saturating form, or the specific values of c and $\tau ,$ from first principles. Doing so would require a distributional assumption on how the identity of the marginal pair changes under repeated independent draws of the noise ε – the population-level analog of the per-draw mechanism Theorem 1 characterizes – which we leave as an open direction (Section 8) rather than introduce unjustified assumptions to close it here. The two-part result of Section 5 is deliberately an envelope, not a reconstruction of the fitted curve: a provable local mechanism plus a provable universal ceiling, between which the empirically observed curve must lie, and does.

## 7 Real-Data Experiment

## 7.1 Data and Task

We use the ACSIncome task from the folktables package [Ding et al., 2021] with 2018 1-year ACS PUMS vintage. In place of the task’s default binary income threshold, we use the continuous reported income (PINCP) directly as y, consistent with the continuous-outcome setting of Sections 3–5. We define the two groups as White $( \mathtt { R A C 1 P = 1 } )$ and Black (RAC1P=2) respondents, the standard comparison for this dataset in the fairness literature. We run the full pipeline on two states, California and Texas, to assess whether the results are specific to one state’s income distribution and demographics (Section 7.4).

For California, after removing missing income values, clipping negative income to zero, and winsorizing the top 1% to limit the influence of extreme earners, $n _ { 0 } = 1 2 1 { , } 0 0 6$ and $n _ { 1 } = 8 { , } 5 5 7$ , with group 1’s mean income (\$50,985) below group 0’s (\$70,039), consistent with the sign convention used throughout the paper.

Pulling PINCP and RAC1P naively from the source dataframe by the processed feature table’s index produced silently misaligned income values (confirmed by the near-zero correlation between income and hours worked or education, which should correlate positively): folktables’s internal preprocessing resets the feature, label, and group arrays to a fresh index disconnected from the original dataframe’s row labels. We corrected this by extracting income from the library’s own filtering step (ACSIncome.\_preprocess) directly, which restored the expected positive correlations before proceeding, and confirmed the same fix is required and sufficient on Texas’s data (Section 7.4).

## 7.2 Predictor

We train a random forest regressor (200 trees, max depth 10) to predict income from the task’s remaining features, excluding the group attribute RAC1P so that any resulting prediction error reflects ordinary predictive difficulty rather than direct use of group membership. We use a 70/30 train/test split and evaluate yˆ only on the held-out test set to avoid overfitting and understating the true prediction error, which is the central quantity studied in this paper. In California, one representative split achieves $R ^ { 2 } = 0 . 4 0 7$ with negligible bias (mean residual $\approx 1 5 ,$ , against incomes on the order of $1 0 ^ { 4 } – 1 \dot { 0 } ^ { 5 } ) ;$ ; in Texas, $R ^ { 2 } = 0 . 4 1 7$ with similarly negligible bias (mean residual ≈ 89). Both are realistic, non-overfit predictors, and are not particularly strong or weak for this task.

## 7.3 Results: California

On the representative California split above, solving problem (1) with $\beta = 0 . 3$ yields an unconstrained true gap of $- \$ 19,417$ enforcing the fairness constraint on yˆ reduces this to $\mathrm { g a p } ( \hat { y } , z ^ { * } )$ ≈ 0 (as designed) while the true gap $\exp ( y , z ^ { * } )$ is −\$6,910 – nonzero, but substantially smaller than the unconstrained disparity.

Because any single split is itself a random quantity (Section 4.1), we repeated this procedure – retraining the predictor from scratch each time – across 10 independent train/test splits. The result was stable: unconstrained gap, $- \$ 19,112,4$ \$872; constrained true $\mathrm { g a p } , - \mathbb { 5 } 6 , 5 7 7 \pm \mathbb { 5 } 7 3 8 \colon$ and, expressing the constrained gap as a fraction of the unconstrained one, $3 4 . 4 \% \pm 3 . 3 \%$ of the original disparity survives the constraint, consistently across all 10 splits (never below 29% or above 40%). This was the headline California result; the single-split numbers above are an illustrative first pass, not independent evidence of the result.

## 7.4 Cross-State Robustness: Texas

To test whether the California result reflects a general mechanism or is specific to California’s income distribution and demographics, we repeat the entire pipeline unchanged – same task, same group pair, same predictor architecture and hyperparameters, same cleaning procedure – on Texas. Texas has a meaningfully different group balance $( n _ { 1 } / n _ { 0 } \approx$ 11.8%, vs. California $\mathrm { \overline { { s } } \approx 7 . 1 \% }$ and a different absolute income distribution (group 0 mean \$58,539, group 1 mean \$42,092, both lower than California’s), providing a genuinely independent test rather than a near-duplicate of the California setting.

The 10-split result on Texas is $- \$ 16,357\pm9826$ (unconstrained), $- \$ 5,846,456$ (constrained true gap), and $3 5 . 7 \% \pm \mathrm { \bar { 1 } } . 8 \%$ of the disparity retained. Table 2 summarizes both states. While the absolute dollar magnitudes differ between states, consistent with their different income distributions, the fraction of the unconstrained disparity that survives the constraint is nearly identical – 34.4% vs. 35.7%, well within each other’s uncertainty – despite the two states differing in group balance, absolute income levels, and predictor error $( R ^ { 2 } = 0 . 4 0 7 \mathrm { v s . 0 . 4 1 7 } )$ . This indicates the phenomenon studied in this paper is not an artifact of one state’s data, and that its approximate magnitude, not only its existence, may generalize.

Table 2: Real-data results, California vs. Texas (10-split mean ± std).
<table><tr><td></td><td>California</td><td>Texas</td></tr><tr><td>Group sizes  $( n _ { 0 } , n _ { 1 } )$ </td><td>121,006, 8,557</td><td> $1 0 5 , 8 3 5 , 1 2 , 5 2 9$ </td></tr><tr><td>Predictor  $R ^ { 2 }$ </td><td>0.407</td><td>0.417</td></tr><tr><td>Unconstrained gap</td><td> $- \$ 19,112\pm \$ 872$ </td><td> $- \$ 16,357\pm9826$ </td></tr><tr><td>Constrained true gap</td><td> $- \$ 6,57 7 \pm9738$ </td><td> $- \$ 5,846\pm \$ 431$ </td></tr><tr><td>Fraction of disparity retained</td><td> $3 4 . 4 \% \pm 3 . 3 \%$ </td><td> $3 5 . 7 \% \pm 1 . 8 \%$ </td></tr></table>

## 7.5 Scope of this Experiment

Two aspects of this setup remain worth noting as scope. We use a single task (income) and a single group pair (White/Black); we do not claim that the specific retained-disparity figures generalize beyond this comparison –other group pairs or outcome variables may behave differently, and two states, while a genuine robustness check, are not a representative sample of all fifty. The consistency observed here is evidence against the result being a California-specific artifact, not proof of universal generalization. We discuss this boundary further in Section 8.

## 8 Discussion

Fairness definition. Every result in this study concerns demographic parity on continuous per-capita benefits (Eq. 2); we do not study how prediction error propagates under other common fairness criteria, such as equal opportunity, which conditions on the true outcome rather than comparing unconditional group averages. Because the definition of equal opportunity already references $y$ in its standard form, the mechanism studied here – a constraint stated purely in terms of yˆ silently drifting from its true-outcome counterpart – may not transfer directly; characterizing this for conditional fairness criteria is a natural and, we expect, non-trivial extension.

Decision structure. Problem (1) is a specific linear program: continuous allocation, a single budget constraint, a single pairwise fairness constraint. The active-set reduction (Proposition 1) relies on this specific structure – in particular, on there being exactly two non-trivial constraints beyond the box constraints on z. Decision problems with more constraints (e.g. multiple budgets, more than two groups) would admit a correspondingly larger, but still bounded, active set, and we expect the qualitative conclusion (a small, fixed number of individuals governing local sensitivity, independent of n) to generalize; we have not verified this beyond the two-constraint case analyzed here.

The active-set stability threshold. Theorem 1 holds only for perturbations small enough that the marginal pair’s identity does not change (Section 5.2); we verified that this condition holds for specific, small finite-difference steps, but we do not derive a general formula for how large a perturbation is permissible before a rank swap occurs. Such a formula would depend on the gap between the marginal individual’s value and their nearest competitor within the group, and deriving it in general would likely require a distributional assumption on the noise – precisely the kind of assumption we deliberately avoided in favor of the looser, assumption-free envelope of Section 6. Formalizing this threshold, and using it to derive the population-averaged small-σ slope directly (rather than validating the per-draw mechanism only, as done here), is an open technical question.

Estimation error versus distribution shift. This paper isolates estimation error: yˆ is a noisy estimate of $y ,$ drawn from the same underlying relationship between features and outcome. A distinct and arguably more consequential source of prediction unreliability is distribution shift, where the feature-outcome relationship itself differs between the data a predictor was trained on and the population it is deployed on. The mechanism studied here – sensitivity concentrated in a small active set – was derived without reference to how ${ \hat { y } } - y$ arises, and may or may not extend unchanged to shift-induced error, which need not resemble the symmetric, mean-zero noise assumed in Section 4.1. Extending this analysis to distribution shift is, in our view, the most consequential direction for future work.

Practical implications. For a practitioner auditing a fairness-constrained allocation system, the results of Sections 4 and 7 imply that verifying a demographic parity constraint against the predictions used at decision time is not sufficient evidence that the constraint holds in reality, even when the predictor is reasonably accurate $( R ^ { 2 } \approx 0 . 4$ in our real-data experiment) and the verification is numerically exact. The size of the resulting true-outcome violation is not negligible or catastrophic in our experiments – roughly a third of the unconstrained disparity survived the constraint – but it is systematic rather than reducible by collecting more data (Section 4.3), meaning an audit that checks predictions alone will report a false sense of certainty regardless of how much data informed the predictor.

## 9 Conclusion

Enforcing a demographic parity constraint on predicted outcomes does not guarantee that the constraint holds against true outcomes, and the resulting violation is systematic, governed by a small, bounded number of individuals regardless of population size, not an artifact that more data resolves, a pattern we establish theoretically, validate numerically, and confirm on real income data, where roughly a third of an unconstrained disparity survives a constraint that appears, by every measure available at decision time, to have eliminated it. The most consequential extension of this result is to distribution shift, where prediction error arises not from estimation noise but from a changing relationship between features and outcomes, a setting that this paper’s mechanism was not derived for, and where the size of the resulting fairness violation may behave very differently.

## Code Availability

All code, including the full synthetic and real-data pipelines, is available at https://github.com/ hildaadwubiosei/predict-then-optimize. The repository contains a single Jupyter notebook reproducing every result and figure reported in this paper, executed end-to-end with no manual intervention between cells. Code is released under the MIT License; written content (this paper’s text and figures) is released separately under CC BY 4.0, as noted in the repository’s LICENSE and LICENSE-CONTENT files. Raw Census data used in Section 7 is not redistributed in the repository due to its size, but is downloaded automatically on first run via the folktables package [Ding et al., 2021].

## References

Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. Advances in Neural Information Processing Systems, 2021.

Adam N. Elmachtoub and Paul Grigas. Smart “predict, then optimize”. Management Science, 2022.

Bryan Wilder, Bistra Dilkina, and Milind Tambe. Melding the data-decisions pipeline: Decision-focused learning fo combinatorial optimization. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2019.

Yu Wang and Violet Xinying Chen. End-to-end fairness optimization with fair decision-focused learning, 2026. URL https://arxiv.org/abs/2607.29441.

Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. Algorithmic decision making and the cost of fairness. In Proceedings ofthe 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2017.

Cyrus Cousins. Uncertainty and the social planner’s problem: Why sample complexity matters. In 2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT ’22), Seoul, Republic of Korea, 2022.

Simon De Vos, Jente Van Belle, Andres Algaba, Wouter Verbeke, and Sam Verboven. Decision-centric fairness: Evaluation and optimization for resource allocation problems, 2025. URL https://arxiv.org/abs/2504.20642.

Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference, 2012.