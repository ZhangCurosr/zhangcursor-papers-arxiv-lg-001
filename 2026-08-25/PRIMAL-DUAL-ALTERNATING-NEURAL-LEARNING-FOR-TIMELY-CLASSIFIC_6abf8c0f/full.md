# PRIMAL–DUAL ALTERNATING NEURAL LEARNING FOR TIMELY CLASSIFICATION WITH PERFORMANCE GUARANTEES

BY JIAMING QIU<sup>1</sup>, YINGYE ZHENG<sup>1</sup> AND YING-QI ZHAO <sup>\*1,a</sup>

<sup>1</sup>Biostatistics Program, Public Health Sciences Division, Fred Hutchison Cancer Center, <sup>a</sup>yqzhao@fredhutch.org

Timely risk classification is essential in many clinical monitoring settings, where decisions must balance the benefit of classifying patients early for subsequent intervention against the value of observing additional data. Yet most existing statistical and machine-learning methods are designed for fully observed trajectories and offer limited control over key operating characteristics such as sensitivity, specificity, and monitoring cost. We cast the sequential classification problem within a multi-objective optimization framework targeting these three criteria. We characterize the optimal decision rule through a value recursion that quantifies, at each time point, the trade-off between immediate classification and continued monitoring. To estimate the rule from data, we formulate a constrained optimization problem that maximizes specificity while enforcing prespecified sensitivity and monitoring-cost constraints. We then develop an estimation procedure that employs a recurrent neural network to approximate the evolving value processes and a primal–dual updating scheme to satisfy the performance constraints. Through simulation studies and an application to continuous glucose monitoring for hypoglycemia risk prediction, we demonstrate that the proposed method yields accurate and timely sequential decision rules that adhere to the desired operating characteristics.

1. Introduction. Many diagnostic and monitoring tasks require sequential decision making, where longitudinal data are used to assess risk and guide timely intervention. In settings such as continuous glucose monitoring (CGM), chronic disease surveillance, and screening programs where sequential measurements accumulate over time, clinicians continually decide whether to act immediately based on the current trajectory or wait for additional observations. Unlike traditional risk prediction models that mainly use baseline information, these practices leverage longitudinally collected data to refine the decision making. Examples include multi-step diagnostic protocols (Yu et al., 2023), longitudinal risk prediction using electronic health records (Pan et al., 2023), and continuous sensor-based monitoring such as CGM for hypoglycemia risk. These practical problems can be cast as sequential binary classification tasks. However, acting too late increases the monitoring burden and may delay care that could prevent critical events, whereas acting too early may trigger unnecessary alarms or interventions. These trade-offs among sensitivity, specificity, and the cost of continued monitoring are prevalent across clinical and public health applications.

Despite the growing availability of high-frequency sensor data and electronic health records, most existing statistical and machine-learning approaches are not designed for sequential classification under explicit performance constraints. Dynamic treatment regimes and reinforcementlearning methods have gained substantial attention in clinical decision-making in the sequential setup (Murphy, 2003; Moodie, Richardson and Stephens, 2007; Laber et al., 2014; Zhao et al., 2015; Luckett et al., 2019; Yu et al., 2023; Zhou, Zhu and Qu, 2024; Shi et al., 2024), and recent work incorporates safety or risk constraints when learning adaptive treatment policies (Laber, Lizotte and Ferguson, 2014; Liu et al., 2024a,b). These approaches, however, aim to optimize cumulative reward over sequences of interventions that modify patient trajectories and influence patient outcomes. In contrast, our setting does not involve altering outcomes through treatment. Instead, the goal is to make timely classification decisions while balancing multiple clinical criteria, such as sensitivity and specificity, and adhering to constraints on monitoring cost. Thus, while methodologically related in spirit, methods on developing dynamic treatment regimes are not directly applicable to constrained sequential classification problems such as ours.

A more closely related line of work is early classification of time series (ECTS), where the goal is to make a class prediction before the entire trajectory is observed. At each time point, these methods decide whether to classify now or wait for more data. Existing ECTS methods typically balance classification accuracy against earliness or measurement cost (see Renault et al., 2025, for a survey). The decision to stop has been guided by prediction reliability (Mori et al., 2018; Schäfer and Leser, 2020), anticipated loss reduction (Dachraoui, Bondu and Cornuéjols, 2015; Achenchabe et al., 2021; Bilski and Jastrz˛ebska, 2023), or likelihood-ratio/posterior-risk stopping boundaries (Ebihara et al., 2021, 2024). Related ideas have also appeared in longitudinal risk prediction: Pan et al. (2023) proposed a reinforced risk prediction method that classifies patients over time while allowing an intermediate “uncertain” category for continued monitoring. However, these methods often target the accuracy–earliness trade-off. Even when multiple objectives are considered, they typically do not provide direct control of sensitivity and specificity as separate operating characteristics while also treating monitoring cost as a dedicated criterion.

These limitations motivate the need for sequential decision rules that (i) can issue early and interpretable decisions for informing future risks, such that monitoring cost is optimized; (ii) directly satisfy clinically meaningful performance targets, such as high sensitivity to avoid missed events and high specificity to avoid false alarms; and (iii) remain flexible enough to accommodate longer sequences of observations. In this work, we develop a multi-objective optimization (MOO) framework (e.g., Miettinen, 1999) for sequential classification. Our approach aligns with a Neyman–Pearson formulation (Tong, Feng and Li, 2018), where we maximize two key design criteria: a sensitivity ensuring protection against missing cases, and a specificity avoiding false positives. We also simultaneously aim to minimize the monitoring cost, which reflects the burden of data collection, delay, or clinical inaction. In applications such as glucose-based hypoglycemia prediction, these objectives translate into fewer missed hypoglycemic events (high sensitivity), fewer false alarms (high specificity), and earlier detection in the sense that low cost corresponds to seeing fewer observations before a decision.

We recast the multi-objective problem as a constrained optimization task that maximizes specificity while enforcing predefined sensitivity and monitoring-cost constraints. From this formulation, we derive a Pareto-optimal sequential decision rule that quantifies, at each time point, the trade-off between acting immediately and gathering additional information. The resulting rule has an intuitive and clinically interpretable form. It recommends providing a decision when the immediate benefit of doing so exceeds the expected benefit of waiting for further data.

To estimate this rule from data, we propose a learning procedure, termed as Timely Decisions with Primal–dual Alternating Neural Learning (TD-PANL), that uses a recurrent neural network model shared across all decision times to approximate the evolving value processes underlying the decision rule. Neural networks flexibly estimate the evolving conditional risk process and the value processes that dictate when to act. To satisfy the desired sensitivity and monitoring-cost constraints, we introduce a primal–dual alternating scheme that iteratively updates the Lagrange multipliers in the dual and the neural network parameters in the primal. This procedure avoids the need for extensive grid search over multipliers and scales readily to high-dimensional or nonlinear trajectories.

The remainder of this paper is organized as follows. In Section 2, we formalize the MOO problem and derive the Pareto-optimal sequential rules. Section 3 provides the details for the TD-PANL method, including the shared model and alternating-step algorithm. Section 4 reports numerical experiments that investigate the performance of the proposed method and demonstrate the trade-offs among sensitivity, specificity, and cost. The method is also applied to continuous glucose monitoring data to predict hypoglycemia risk. Finally, we provide a discussion in Section 5.

2. Optimal Sequential Decision Rules. Consider a sequence of observations as $X =$ $X _ { 1 : T }$ with $X _ { t } \in \mathbb { R }$ for $t = 1 , \dots , T$ and a binary outcome $Y \in \{ 0 , 1 \}$ , with $Y = 1$ indicating the presence of an adverse outcome. We observe the data over time and need to determine at each step whether to stop and classify or continue observing. Our goal is to construct optimal sequential rules under competing performance criteria, including specificity, sensitivity, and monitoring cost. In the following, we first define the sequential rules, introduce a constrained Neyman–Pearson formulation that captures the desired performance guarantees, and then derive an explicit form of the optimal rule.

2.1. Sequential rules and a constrained Neyman–Pearson problem. A sequential decision rule is a function that maps the history of $X _ { 1 : t }$ to a decision: “classify now” or “continue”, until the terminal time $T$ when a classification is due. To formalize this, we define $f _ { t } = f _ { t } \left( X _ { 1 : t } \right) \in$ R as the score at time t based on the history $X _ { 1 : t } = ( X _ { 1 } , \ldots , X _ { t } )$ . For simplicity, we write $f _ { 1 : t }$ as the scores up to time t with the convention that $f _ { 1 : t - 1 } = 0$ means $f _ { 1 } , \ldots , f _ { t - 1 }$ are all zero. We then define

$$
\varphi \left( X _ { 1 : t } \right) = \mathbb { 1 } \left( f _ { 1 } > 0 \right) + \mathbb { 1 } \left( f _ { 1 } = 0 , f _ { 2 } > 0 \right) + \dots + \mathbb { 1 } \left( f _ { 1 : t - 1 } = 0 , f _ { t } > 0 \right) ,\tag{2.1}
$$

where $t = 1 , \ldots , T$ , and $\mathbb { 1 } ( \cdot ) \in \{ 0 , 1 \}$ takes 1 if the condition is true, and 0 otherwise.

The decision process can stop before the terminal time $T$ once a non-zero score is received. Denote τ = inf $\{ 1 \leq t \leq T : f _ { t } \neq 0 \}$ as the associated stopping time at which rule (2.1) makes the decision, with $\varphi = 1$ indicating a positive decision at time $\tau$ and $\varphi = 0$ meaning negative decision. Note that the stopping time depends on the observed history $X _ { 1 : \tau }$ , allowing the sequential rule to adapt dynamically to individual trajectories, such as those of different patients.

Let $C _ { t }$ denote the accumulated operational cost of observing up to time t. Thus, $0 \leq$ $C _ { 1 } < \cdots < C _ { T }$ and $C _ { \tau }$ is the realized cost of the decision. The choice of $C$ is applicationspecific. If observations are equally spaced and similarly burdensome, a natural choice is $C _ { t } = ( t - 1 ) / ( T - 1 )$ , corresponding to normalized decision time, with zero for deciding at the first time point and one for waiting until the last time point. If late decisions are disproportionately undesirable, one may instead use a nonlinear cost such as $C _ { t } = ( ( t - 1 ) / ( T ^ { ' } - \hat { 1 } ) ) ^ { 2 }$ If observations differ in burden, $\mathrm { e . g . }$ ., in a diagnostic pathway where later time steps may involve more expensive tests such as imaging or biopsy, one may use $\begin{array} { r } { C _ { t } = \sum _ { s = 1 } ^ { t } \bar { c } _ { s } , } \end{array}$ with $c _ { s }$ denoting the incremental monetary expense or measurement burden at time s. Overall, C can flexibly encode elapsed time, cumulative measurement burden, or increasing penalty for delayed decision making.

We consider three characteristics of any sequential decision rules, including sensitivity $\mathbb { E } [ \varphi | Y = 1 ]$ , specificity $1 - \mathbb { E } [ \varphi \mid Y = 0 ]$ , and expected cost $\mathbb { E } [ C _ { \tau } ]$ . These quantities represent competing goals. High sensitivity reduces missed events, but it also leads to higher false positives (lower specificity). Lower cost may correspond to earlier decisions, which might result in less accurate decisions due to the use of less information.

Rather than optimizing a single scalar objective, we adopt a multi-objective optimization (MOO) perspective by identifying the sequential rule that

$$
{ \mathrm { m a x i m i z e ~ } } \operatorname { \mathbb { E } } [ \varphi \mid Y = 1 ] , 1 - \operatorname { \mathbb { E } } [ \varphi \mid Y = 0 ] , { \mathrm { ~ a n d ~ m i n i m i z e ~ } } \operatorname { \mathbb { E } } [ C _ { \tau } ]\tag{2.2}
$$

with expectations taken over $X _ { 1 : T }$ . This problem seeks to find the optimal trade-off among the aforementioned three competing criteria.

Usually multiple optima of an MOO exist, where none of the objectives can be further improved without sacrificing some other objectives. Such solutions are called Pareto optimal, and their collection is called the Paretofront (e.g., Pardalos, Žilinskas and Žilinskas, 2017). Pareto optimality provides a useful interpretation of this MOO formulation. Because cost, sensitivity, and specificity are competing objectives, one should not expect a single sequential rule to dominate across all aspects. A rule on the Pareto front is non-dominated: no other rule can increase sensitivity or specificity, or reduce expected observation cost, without worsening the remaining criteria. Therefore, the Pareto front summarizes the set of efficient operating trade-offs available to the practitioner. Different points on this front then correspond to different viable choices.

To obtain estimable rules that achieve desired operating levels, we adopt the ε-constraints approach. For user-specified targets $0 < \beta < 1$ on sensitivity and $0 < \gamma < 1$ on cost, we find the rule that achieves these targets with the highest specificity:

$$
{ \mathrm { m i n i m i z e ~ I E } } [ \varphi \mid Y = 0 ] ,\tag{2.3}
$$

$$
{ \mathrm { s u b j e c t ~ t o ~ } } \operatorname { \mathbb { E } } [ C _ { \tau } ] \leq \gamma { \mathrm { ~ a n d ~ } } \operatorname { \mathbb { E } } [ \varphi | Y = 1 ] \geq \beta .
$$

This mirrors the Neyman–Pearson paradigm, which maximizes specificity subject to fixed sensitivity, but incorporates cost as an additional operating constraint.

The Lagrangian function of this constraint optimization problem is

$$
\mathcal { L } \left( f _ { 1 : T } , a , b \right) = \mathbb { E } [ \varphi \mid Y = 0 ] + a \left( \mathbb { E } [ C _ { \tau } ] - \gamma \right) + b \left( \beta - \mathbb { E } [ \varphi \mid Y = 1 ] \right)\tag{2.4}
$$

for some non-negative Lagrangian multipliers $( a , b )$ . The Lagrangian multipliers represent the relative importance of the cost, sensitivity, and specificity. For any fixed a and $b ,$ minimizing $\mathcal { L }$ corresponds to minimizing a weighted sum of false positives, false negatives, and cost.

2.2. The optimal decision rule to the Lagrangian. Define $\mu _ { t } { : = } \mathbb { E } [ Y \mid X _ { 1 : t } ]$ and $\eta _ { t } { : = \left( b p _ { 1 } ^ { - 1 } + \dot { p } _ { 0 } ^ { - 1 } \right) \mu _ { t } - p _ { 0 } ^ { - 1 } }$ , where $p _ { 1 } = \mathbb { P } ( Y = 1 )$ and $p _ { 0 } = 1 - p _ { 1 }$ . Let $S _ { T } { : = } \eta _ { T } ^ { + } - a C _ { T }$ and recursively

$$
\nu _ { t } { : = } \mathbb { E } [ S _ { t + 1 } \mid X _ { 1 : t } ] , \quad S _ { t } { : = } \operatorname* { m a x } \big ( \eta _ { t } ^ { + } - a C _ { t } , \nu _ { t } \big )\tag{2.5}
$$

for $t = 1 , \dots , T - 1$ . Here $\eta _ { t } ^ { + } { : = } \operatorname* { m a x } ( \eta _ { t } , 0 )$ . See Qiu, Zheng and Zhao (2026, Supplement, Section S1) for the proof of the following proposition.

PROPOSITION 2.1. For any fixed Lagrange multipliers $a , b \geq 0$ , the rule minimizing the Lagrangian (2.4) is

$$
\tilde { f } _ { t } ( X _ { 1 : t } ; a , b ) = \left\{ \begin{array} { l l } { 1 , } & { i f \eta _ { t } > 0 a n d \zeta _ { t } > \nu _ { t } , } \\ { 0 , } & { i f \nu _ { t } \geq \zeta _ { t } , } \\ { - 1 , } & { i f \eta _ { t } \leq 0 a n d \zeta _ { t } > \nu _ { t } , } \end{array} \right.\tag{2.6}
$$

with $\zeta _ { t } { : = } \eta _ { t } ^ { + } - a C _ { t }$ and $\nu _ { t }$ as defined above.

Here, $\tilde { f } _ { t } = 1$ denotes stopping with a positive decision, $\tilde { f } _ { t } = 0$ denotes continuation, and $\tilde { f } _ { t } = - 1$ denotes stopping with a negative decision. This is consistent with (2.1), where the indicator sum in (2.1) records whether the stopping decision is positive, while the stopping time is determined by the first t such that $\tilde { f } _ { t } \neq \bar { 0 }$ . Thus, a negative value of $\tilde { f } _ { t }$ corresponds to a negative termination $\varphi = 0$ . Particularly, $\eta _ { t }$ is the evidence in favor of a positive classification, and its sign dictates whether stopping at time t leads to a positive or negative decision. The quantity $\zeta _ { t } = \eta _ { t } ^ { + } - a C _ { t }$ is the net immediate benefit of stopping at time t, after accounting for the monitoring cost already incurred. The quantity $\nu _ { t }$ is the expected benefit of continuing to observe the trajectory and making the decision later. Thus, the rule stops when the immediate benefit $\zeta _ { t }$ exceeds the continuation value $\nu _ { t }$ , and otherwise continues monitoring. Later, in Subsection $4 . 2$ , we plot $\eta _ { t } , \nu _ { t }$ , and $\zeta _ { t }$ along observation trajectories to illustrate how these quantities govern the decision process in practice.

2.3. Dual characterization and Pareto optimality. The optimal rule in Proposition 2.1 solves the Lagrangian (2.4) for any fixed pair of multipliers $( a , b )$ . A naive way to approximate the full Pareto front would be to compute the rule (2.6) over a grid of $( a , b )$ values and record the resulting specificity, sensitivity, and cost. However, the multipliers do not directly control these interpretable operating characteristics. A regular grid over $( a , b )$ therefore need not translate into a regular coverage of the Pareto front: it may produce many similar rules in some regions while leaving other regions sparsely covered (see Figure S2.4 of the Supplement).

A more principled approach is to examine the dual function of the Lagrangian function $\mathcal { L } .$ defined as $\begin{array} { r } { \mathfrak { G } ( { a } , { b } ) { : = } \operatorname* { i n f } _ { f _ { 1 : T } } \mathcal { L } \left( f _ { 1 : T } ; { a } , { b } \right) } \end{array}$ , which, by Proposition 2.1, is

$$
\begin{array} { r } { \mathcal { G } ( a , b ) = b \beta - a \gamma - \mathbb { E } [ S _ { 1 } ] . } \end{array}\tag{2.7}
$$

For any fixed pair $( a , b )$ , minimizing the Lagrangian corresponds to finding the best rule under a particular weighted trade-off among false positive rate, sensitivity, and monitoring cost. The dual function records the optimal value achievable under this trade-off. By construction, the dual function $\mathcal { G }$ is a lower bound on the minimal value of the constrained problem. Maximizing $\mathcal { G } ( a , b )$ over nonnegative $( a , b )$ therefore finds the trade-off weights that most tightly support the constrained optimum. We establish below that, the maximum of the dual function equals the minimum of the primal objective, i.e., the primal and dual problems have no duality gap. Consequently, the point at which $\mathcal { G } ( a , b )$ achieves its maximum corresponds exactly to the Lagrange multipliers that enforce the sensitivity and cost constraints.

Formally, we consider the dual problem as

$$
\mathrm { m a x i m i z e } ~ \mathcal { G } ( a , b ) ~ \mathrm { f o r } ~ a , b \geq 0 .\tag{2.8}
$$

We show that, there always exists some dual optimal $a ^ { * } , b ^ { * }$ with no duality gap, that is, the $\mathcal { L }$ and $\mathcal { G }$ equal. At the optimal multipliers $a ^ { * } , b ^ { * }$ , the optimal rule $\varphi ^ { * }$ satisfies the sensitivity and cost constraints in (2.3) with equality and achieves the lowest possible false positive rate among all rules satisfying those constraints. This also explains the connection to Pareto optimality. If another rule could reduce false positive rate without decreasing sensitivity or increasing cost, it would contradict the optimality of $\varphi ^ { * }$ for the constrained problem. Therefore, $\varphi ^ { * }$ is Pareto optimal for the MOO. These results are summarized in the following theorem.

THEOREM 2.1. Supposefor $t = 1 , \ldots , T$ , the $X _ { t }$ has a continuous density, and $X _ { 1 : t } \mapsto$ $\mu _ { 1 : t }$ are continuous functions bounded away from zero, then for any $\beta , \gamma \in ( 0 , 1 )$ there exists $a \varphi ^ { * }$ whose scores take theform of (2.6) such that

1. $\varphi ^ { * }$ solves the constrained problem (2.3) with equality;

2. $\varphi ^ { * }$ is Pareto optimalfor the MOO problem (2.2).

One key assumption here is that $\mu _ { t }$ is a continuous function of $X _ { 1 : t }$ for all t. The continuity assumption ensures that sensitivity varies continuously with the decision threshold, making it possible to attain any desired sensitivity level without randomization. Essentially, the optimal rule compares the value of stopping versus continuing, providing a generalization of the Neyman–Pearson paradigm to sequential settings. We refer to Section S1 of the Supplement for proof.

In this sense, the theorem links the constrained problem and the original MOO formulation. Each choice of constraint levels $( \beta , \gamma )$ corresponds to a particular point on the Pareto front. Solving the dual problem for this pair yields the Lagrange multipliers $( a ^ { * } , b ^ { * } )$ that enforce the constraints and the corresponding optimal rule. As $( \beta , \gamma )$ vary over their feasible ranges, the resulting solutions lead to the full collection of triples that cannot dominate each other. Thus, the ε-constraint formulation, combined with the dual characterization, provides a complete parametrization of the Pareto front. This optimal rule structure guides the estimation strategy developed in Section 3.

3. Estimating the Optimal Rule via Primal-Dual Alternating Neural Learning. Section 2 provides the theoretical form of the optimal sequential rule. For fixed Lagrange multipliers $( a , b )$ , the optimal rule depends on two processes, $\mu _ { t } = \mathbb { E } [ Y \mid X _ { 1 : t } ]$ and $\nu _ { t } = \mathbb { E } [ S _ { t + 1 } \mid X _ { 1 : t } ]$ $\operatorname { I f } \mu _ { t }$ and $\nu _ { t }$ were known, the optimal sequential rule as defined in (2.6) can be obtained. In real-world applications, these quantities must be estimated from data. We introduce a Timely Decisions with Primal–dual Alternating Neural Learning (TD-PANL) approach that uses neural networks as flexible estimators for $\mu _ { t }$ and $\nu _ { t } .$ , and alternates between

1. primal updates on model parameters, which refine the estimate of $\nu _ { t } ;$

2. dual updates, which adjust $( a , b )$ so that the sensitivity and cost constraints are satisfied.

This approach combines the constrained Neyman–Pearson formulation of Section 2 with neural network approximations of the value processes $\mu _ { t }$ and $\nu _ { t }$ . The process $\mu _ { t }$ is trained once to estimate the conditional class probability process, while the process $\nu _ { t }$ is updated iteratively to approximate the value driving the stopping rule. The dual parameters $( a , b )$ are updated in parallel to enforce sensitivity and cost constraints.

In particular, we will employ Recurrent Neural Networks (RNNs) in estimating $\mu _ { t }$ and $\nu _ { t }$ Designed for sequence-to-sequence learning, RNNs compute the output at time $t + 1$ as a function of the new observation $X _ { t + 1 }$ , and the hidden state (or output) carried over from time t. This recursive evaluation eliminates the need to stack a list of time-specific models. Moreover, the resulting output at time t will depend only on the observed history $X _ { 1 : t }$ . Combined with the well-known flexibility and expressiveness of neural networks, RNNs form an ideal estimator class for approximating the processes $\mu _ { 1 : T }$ and $\nu _ { 1 : T - 1 }$ required for our method.

3.1. Estimation of $\mu _ { t }$ and $\nu _ { t }$ . We first estimate the quantities in Proposition 2.1, assuming the Lagrangian multipliers $a , b$ are fixed.

For each t, the risk process $\mu _ { t } = \mathbb { E } [ Y \mid X _ { 1 : t } ]$ is the minimizer of the squared error

$$
\sum _ { t = 1 } ^ { T } \mathbb { E } [ ( g _ { t } - Y ) ^ { 2 } ]
$$

over all history-dependent functions $g _ { 1 : T }$ . We therefore can estimate $\mu _ { t }$ by minimizing the empirical analogue of this loss. Estimation of $\nu _ { t }$ is more challenging because it depends on its future through $S _ { t + 1 }$ as defined in (2.5). Using the recursion form of $\nu _ { t } = \mathbb { E } [ S _ { t + 1 } \mid X _ { 1 : t } ]$ we employ a temporal difference learning approach, where the process $\nu _ { 1 : T - 1 }$ is obtained by minimizing

$$
\sum _ { t = 1 } ^ { T - 1 } \mathbb { E } [ ( g _ { t } - S _ { t + 1 } ) ^ { 2 } ]\tag{3.1}
$$

which fits all $\nu _ { t }$ simultaneously. This temporal difference-based loss function aggregates across all time $t = 1 , \ldots , T$ without backward recursion or fitting a specific model at each time point.

We estimate both $\mu _ { t }$ and $\nu _ { t }$ using RNNs. Denote $\mathring \mu \left( X _ { 1 : t } , \theta _ { \mu } \right)$ and $\mathring { \nu } \left( X _ { 1 : t } , \theta _ { \nu } \right)$ as the RNN models for $\mu _ { t }$ and $\nu _ { t }$ respectively. Substituting these estimates, together with the sample class proportions $\hat { p } _ { 0 }$ and $\hat { p } _ { 1 }$ , into the definitions in Section 2 yields the plug-in quantities $\mathring \eta , \mathring \zeta , \mathring { S } , \mathring \tau$ and $\mathring \varphi .$ . The least squared errors for $\mu _ { t }$ and $\nu _ { t }$ are approximated by the corresponding empirical loss functions

$$
Q _ { \mu } \left( \theta _ { \mu } \right) = \sum _ { t = 1 } ^ { T } \mathbb { E } _ { n } [ \left( \overset { \circ } { \mu } \left( X _ { 1 : t } , \theta _ { \mu } \right) - Y \right) ^ { 2 } ] ,\tag{3.2}
$$

$$
Q _ { \nu } \left( \theta _ { \nu } | \theta _ { \mu } , a , b \right) = \sum _ { t = 1 } ^ { T - 1 } \mathbb { E } _ { n } [ \left( \bar { \nu } \left( X _ { 1 : t } , \theta _ { \nu } \right) - \mathring { S } _ { t + 1 } \right) ^ { 2 } ] ,
$$

where $\mathbb { E } _ { n } [ \cdot ]$ denotes the empirical expectation. The objective $Q _ { \mu }$ corresponds to a many-toone classification problem that uses the entire sequence $X _ { 1 : T }$ to predict Y. We first obtain $\hat { \theta } _ { \mu } \in$ arg min $Q _ { \mu }$ . For fixed $( a , b )$ , we compute $\widehat { \theta } _ { \nu } \in$ arg min $Q _ { \nu } \left( \cdot | \widehat { \theta } _ { \mu } , a , b \right)$ . Plugging-in the fitted networks, denoted as $\hat { \mu } = \mathring { \mu } \left( X _ { 1 : t } , \hat { \theta } _ { \mu } \right) , \hat { \nu } = \mathring { \nu } \left( X _ { 1 : t } , \hat { \theta } _ { \nu } \right)$ , into (2.6) provides the estimated optimal rule for the given Lagrangian multiplier $a , b .$

3.2. Alternating updates for model parameters and Lagrange multipliers. A key intermediate result in the proof of Theorem 2.1 is the following characterization of the dual gradient.

PROPOSITION 3.1. The gradient ofthe dualfunction is $\partial \mathcal { G } / \partial a = \mathbb { E } [ C _ { \tau } ] - \gamma$ and $\partial \mathcal { G } / \partial b =$ $\beta - \mathbb { E } [ \tilde { \varphi } | Y = 1 ]$ , where $\tilde { \varphi }$ is the rule (2.1) with scores $\tilde { f } = \tilde { f } ( \boldsymbol { X } ; a , b )$ of (2.6) plugged-in.

This shows that the dual gradient consists of performance gaps, that is, the difference between the achieved and desired cost and sensitivity. The empirical gradients can therefore be approximated by substituting empirical performance measures of the plug-in rule:

$$
\frac { \partial \mathcal { G } } { \partial a } \left( a , b | \hat { \theta } _ { \mu } , \hat { \theta } _ { \nu } \right) \approx \mathbb { E } _ { n } [ \hat { C } ] - \gamma , \mathrm { a n d } \frac { \partial \mathcal { G } } { \partial b } \left( a , b | \hat { \theta } _ { \mu } , \hat { \theta } _ { \nu } \right) \approx \beta - \mathbb { E } _ { n } [ \hat { \varphi } | Y = 1 ] .\tag{3.3}
$$

These expressions have an intuitive interpretation, where a should be increased if the classifier has too much cost, and $b$ should be increased if the sensitivity is lower than $\beta .$ Hence, any gradient-based optimizer can be applied to maximize $\mathring { \operatorname { \mathcal { G } } } \left( a , b | \hat { \theta } _ { \mu } , \hat { \theta } _ { \nu } \right)$ , provided we update $\nu$ as the dual variables change.

A direct gradient-based approach would require refitting ν to minimize $Q _ { \nu }$ for every update of $( a , b )$ , which is computationally intensive. However, modern gradient ascent uses small step sizes, controlled by learning rates and gradient clipping. Therefore, the change in $( a , b )$ from one iteration to the next is typically modest. Consequently, the corresponding drift in the optimal ν-network is also small, and it suffices to patch $\hat { \theta } _ { \nu }$ with a single gradient step on $Q _ { \nu }$ rather than re-estimating it from scratch. This leads to an alternating dual-ascent / primal-descent scheme:

1. Dual ascent: update $( a , b )$ using the empirical gaps in cost and sensitivity.

2. Primal descent: update $\theta _ { \nu }$ with one step of gradient descent on $Q _ { \nu }$

This alternating structure forms the basis of Algorithm 1.

The Lagrangian multipliers $( a , b )$ are updated using the empirical gaps in cost and sensitivity, so that the induced rule moves toward the target levels $( \beta , \gamma )$ . The update of $\theta _ { \nu }$ is nested within this process, allowing the rule and the multipliers to be learned jointly from the training data. All parameter updates are based on the training data, while a separate validation set can be used to monitor performance and stop training. In particular, training is stopped when either the maximum number of iterations K is reached or the norm of the dual gradient $\left\| \partial \mathring { \mathcal { G } } \right\|$ falls below a prespecified tolerance $\delta .$ By Proposition 3.1, the latter means that the empirical cost and sensitivity gaps are both sufficiently small. In our implementation, $K$ was chosen sufficiently large so that training typically stopped by the performance rather than by hitting the iteration limit.

Algorithm 1 Optimizing $\theta _ { \nu } , a ,$ and b given $\beta , \gamma$ for K iterations with $\overline { { \delta } }$ tolerance and learning   
rate $r .$   
Obtain $\hat { \theta } _ { \mu }$ ← arg min $Q \mu ,$ then initialize $a ^ { ( 0 ) }$ and $\boldsymbol { b } ^ { ( 0 ) }$ , set $k = 0 .$   
$\theta _ { \nu } ^ { ( 0 ) }$ ← arg min $Q _ { \nu } \left( \cdot | \hat { \theta } _ { \mu } , a ^ { ( 0 ) } , b ^ { ( 0 ) } \right)$   
while $k \leq K$ and $\left\| \partial { \dot { \boldsymbol { \mathfrak { S } } } } \right\| > \delta$ do   
Lagrangian update: $\left( a ^ { ( k + 1 ) } , b ^ { ( k + 1 ) } \right) \gets \left( a ^ { ( k ) } , b ^ { ( k ) } \right) + r \partial \mathring { \mathcal { G } } ( a , b | \hat { \theta } _ { \mu } , \theta _ { \nu } ^ { ( k ) } ) ;$   
ν-update: $\theta _ { \nu } ^ { ( k + 1 ) } \stackrel { \cdot \cdot } {  } \theta _ { \nu } ^ { ( k ) } - r \partial Q _ { \nu } ( \tilde { \theta _ { \nu } } | \hat { \theta } _ { \mu } , a ^ { ( k + 1 ) } , b ^ { ( \stackrel { \cdot } { k } + 1 ) } )$   
$k  k + 1 .$   
end while

In our implementation, the training samples are representative, and $p _ { 0 }$ and $p _ { 1 }$ are estimated by the sample class proportions. However, in applications where one class is rare, classdependent sampling, such as upsampling the minority class or downsampling the majority class, may be used to improve class representation. Such sampling changes the class proportions in the training sample. Supplement Section S3 describes an extension to class-dependent sampling, including corrections for estimating $\mu _ { t } , \nu _ { t }$ , and the dual updates.

4. Numerical Experiments. This section evaluates the empirical performance of the proposed TD-PANL method using two complementary examples. We begin with a synthetic study, which mimics the sequential monitoring of longitudinal trajectories in practice. These examples have increasing complexity to illustrate how sequential structure, nonlinearity, and signal-to-noise ratio affect the estimation of the value processes. We then apply TD-PANL to a real continuous glucose monitoring dataset (CGM, Weinstock et al., 2015) to evaluate early hypoglycemia detection, where timely decisions are clinically important. Across settings, we assess whether the learned rules satisfy the desired sensitivity and cost constraints.

For simplicity, across all experiments we set the cost function to $C _ { t } = ( t - 1 ) / ( T - 1 )$ yielding a unified cost scale of [0, 1] regardless of the sequence length T. For plug-in estimators for $\mu$ and $\nu ,$ the proposed method used two separate gated recurrent units (GRUs; Chung et al., 2014), each followed by a fully connected linear post-processing layer. The resulting decision rule contained approximately 2,000 to 3,000 trainable parameters depending on $T$ . Details of the network architecture are provided in Section S2 of the Supplement.

## 4.1. Synthetic Data Examples.

4.1.1. Data Generation. We begin with simulated studies designed to mimic sequential monitoring of longitudinal measurements. The data are generated under three temporal mechanisms: a Markov process, a probit model, and a bi-modal mixture model. These settings vary in temporal and signal complexity, thereby probing different aspects of learning the processes $\mu _ { t }$ and $\nu _ { t }$ . Particularly, we generate $X _ { 1 : T }$ as a stationary autoregressive time series with standard Gaussian innovations. We draw Y following $\mathbb { P } ( Y = 1 | X _ { 1 : T } ) = h ( V )$ , where $V$ is a linear combination of $X _ { 1 : T }$ and $h : \mathbb { R }  [ 0 , 1 ]$ maps this score to probability. We varied the auto-regressive parameters, the link h, and the coefficients of $V$ to create the three simulation settings, where the marginal event rates $\mathbb { P } ( Y = 1 )$ are both 0.5 for the Markov and probit settings, and 0.27 for the bi-modal setting. The detailed specification can be found in Section S2.1 of the Supplement.

The sequence length is set to $T = 5$ , allowing us to compute the true optimal by numerical integration, which becomes prohibitive for larger $T .$ . The proposed method itself does not require this restriction. Each setting was repeated 50 times. In each replicate, we generated $1 0 ^ { \hat { 4 } }$ and $1 0 ^ { 5 }$ series for training and testing respectively. An additional 2,500 validation series were generated for performance monitoring and stopping training. A grid of sensitivity and cost constraints $\beta , \gamma \in { 0 . 1 , . . . , 0 . 9 }$ were considered.

4.1.2. Methods for Comparison. For the simulated settings, we numerically computed the optimal rule using the known data-generating mechanism, providing a theoretical benchmark for the best achievable specificity–cost trade-off under the imposed sensitivity constraint.

We compared the proposed method to a simple myopic rule, which bases its decision solely on the currently observed data at a pre-specified time $t _ { 0 }$ and ignores any potential future information. Operationally, this places a Neyman–Pearson classifier (Tong, Feng and Li, 2018) at $t _ { 0 }$ , so the myopic rule can be constructed directly from the estimated probabilities $\hat { \mu } _ { t _ { 0 } }$ Because the myopic rule does not learn a data-dependent stopping time, its cost is fixed at $C _ { t _ { 0 } }$

We additionally compared the proposed method with FIRMBOUND (Ebihara et al., 2024), an ECTS method closely related to our optimal-stopping formulation. Additional details on FIRMBOUND are provided in Section S2.2 of the Supplement. Like TD-PANL, FIRM-BOUND treats early classification as a finite-horizon stopping problem and compares immediate stopping risk with expected continuation risk. However, FIRMBOUND performs this recursion using the posterior risk $\mu _ { t }$ as a low-dimensional summary of the observed history, whereas TD-PANL conditions on the full trajectory $X _ { 1 : t }$ . Additionally, FIRMBOUND optimizes a prespecified weighted combination of classification error and observation cost, rather than directly targeting sensitivity and cost constraints. Consequently, identifying operating points with desired sensitivity and cost levels requires tuning over a grid of weight parameters and selecting rules whose realized operating characteristics match the target constraints. Particularly, we first performed a broad coarse grid search over its weight parameters on $[ 0 , 5 ] \times [ 0 , 5 ]$ to identify the range producing sensitivity near 0.9. We then refined the search using a 20-by-20 grid over the smaller region $[ 0 . 0 1 , 1 ] \times [ 1 , 2 . 3 ]$ . Finally, we used post hoc matching: for each cost constraint considered, we selected FIRMBOUND rules whose realized sensitivity and cost were comparable, and then compared specificity.

4.1.3. Results. To illustrate the specificity–cost trade-off, Figure 4.1 constrains the sensitivity constraint at $\beta = 0 . 9$ and varies the cost constraint over $\gamma = 0 . 1 , \ldots , 0 . 9$ . As demonstrated in Figure 4.1a, the proposed method nearly coincides with the true optimal Pareto frontier in the Markov and probit settings, because both $\mu _ { t }$ and $\nu _ { t }$ are relatively easy to learn. In the bi-modal setting, TD-PANL still recovers the overall trade-off but exhibits a relatively larger gap from the optimum due to the increased difficulty of estimating $\mu _ { t }$ under a non-monotone, bimodal link $h .$ . Supplementary Section S2.1 shows that when the true $\mu _ { t }$ is supplied, the proposed method recovers the optimal frontier. Figure 4.1b further shows that the sensitivity and cost constraints are satisfied throughout the considered operating range. These results indicate that TD-PANL can attain performance close to the optimal stopping rule while remaining computationally feasible.

The resulting frontier also provides insight into the value of additional observations. Notably, the gain in specificity diminishes as cost increases, especially in the probit and bi-modal examples, indicating that once specificity is near its maximal value, additional observations yield limited gains. This is also reflected in the excess cost panel (bottom right of Figure 4.1b) where the rightmost bars are negative. Together, these patterns suggest an oracle-like behavior, in the sense that the learned rule often reaches a decision using only a shorter sequence of observations while achieving performance close to that attainable using the full trajectory.

The myopic rules’ performance was evaluated directly on the testing set using an ROC analysis of the predicted probabilities, so they always satisfy the requested sensitivity constraints. Relative to the myopic benchmark, TD-PANL consistently achieves higher specificity across the three simulation settings. These results demonstrate the benefit of adaptive stopping, compared to a fixed-time classification.

Compared with FIRMBOUND, the proposed TD-PANL achieved comparable specificity in the Markov and probit settings while maintaining similar sensitivity and observation cost. In the bi-modal setting, TD-PANL achieved higher specificity, with the advantage most pronounced under stringent cost constraints. The larger gap in the bi-modal setting may reflect that $\mu _ { t }$ does not summarize the stopping problem as effectively as in the other two settings. In the Markov and probit examples, the Bellman recursion (2.5) can be written using $\mu _ { t }$ alone rather than the full history $X _ { 1 : t }$ . In contrast, in the bi-modal setting, the same value of $\mu _ { t }$ may arise from different histories that imply different future evolution. As a result, conditioning only on $\mu _ { t }$ may discard information relevant to the stopping decision.

Additionally, we examined the full three-way trade-off among specificity, sensitivity, and cost in Figure 4.2. We solved the working problem over a grid of sensitivity and cost constraints spanned by $\beta , \gamma \in { 0 . 1 , . . . , 0 . 9 }$ . Each fitted rule was evaluated on the test set, producing realized sensitivity, cost, and specificity. In the heatmaps, the fitted rules were then grouped into bins according to their realized sensitivity and cost, and the color of each tile gives the average realized specificity within that bin. The figure therefore provides a visual summary of the three-way trade-off. Across the three examples, high specificity is maintained over a broad region of the sensitivity–cost plane. The loss of specificity is concentrated in the lower-right region requiring both high sensitivity and low cost, where the rule must stop early while still identifying most positive cases. Conversely, when the sensitivity requirement is relaxed or more observations are allowed, high specificity is easier to maintain. These results indicate that favorable operating characteristics can be achieved across a wide range of sensitivity and cost targets, rather than only at the fixed sensitivity level considered in Figure 4.1.

Figure S2.3 in the Supplement re-expresses the operating characteristics shown in Figure 4.2 in terms of the corresponding learned multipliers. The highly nonlinear relationship between the multipliers and the resulting operating characteristics provides additional motivation for the adaptive dual updates used by the proposed method.

Finally, we conducted a sensitivity analysis to assess the robustness of the simulation results to the choice of neural network architecture, model depth, and hidden dimension. We considered a deeper GRU model, a wider GRU model, and alternative models including a vanilla RNN, an LSTM, and a Transformer encoder with a causal mask. Overall, the empirical conclusions were broadly stable across these choices. See Section S2.3 of the Supplement for details.

![](images/8138603fcb055a3077f597c7a19e1e7139831f9b387b109b96aa7aabcfcbe997.jpg)

(a) Specificity achieved by different methods at varying monitoring-cost levels.  
![](images/647a3909e9e8a8911d13478745f5191fe8b9391284d5f5759512d9fe5fab0ea1.jpg)  
(b) Constraints on sensitivity (left) and cost (right) are satisfied by the proposed method.

Fig 4.1: Performance of the proposed sequential decision rules, FIRMBOUND competitor, the myopic baselines, and the true optimal, under a constraint of sensitivity at $\beta = 0 . 9$ and cost over $\gamma = 0 . 1 , \dots , 0 . 9$ . The top panels illustrate the specificity–cost trade-off under a sensitivity constraint. The bottom panels inspect whether the constraints were satisfied, where the over\_cost (the bottom right panel) is defined by the achieved cost minus the desired. The solid horizontal lines also represent the myopic baselines, which satisfy the sensitivity and cost constraints by construction. The means (dots) and 90% intervals (bars) over 50 repeated experiments are reported.

![](images/1972d0f72e46fa41c5a56b7e90ccf5405c7fb7984476262867d9c55fdefefd82.jpg)  
Fig 4.2: The operating characteristic heatmaps of the sequential rules illustrates the three-way trade-off among specificity, sensitivity, and cost. Each tile represents the binned and averaged performance over 50 repeated experiments.

4.2. Early Detection of Hypoglycemia through Continuous Glucose Monitoring. We applied the proposed sequential decision-rule framework to continuous glucose monitoring (CGM) data from a cohort of older adults $( \geq 6 0 { \mathrm { y e a r s } } )$ with long-standing Type 1 diabetes (Weinstock et al., 2015)<sup>1</sup>. The original cohort included 201 participants, with 101 cases who experienced at least one severe hypoglycemic episode in the prior 12 months and 100 controls without such events in the preceding three years. All participants wore blinded CGM for up to 14 days, during which glucose was recorded without additional intervention, capturing natural within-person variation.

Severe hypoglycemia poses significant risks, especially for older adults, including falls, cardiovascular events, and cognitive impairment. Early detection of impending hypoglycemia enables timely interventions, such as carbohydrate intake or insulin adjustment, which is easier to implement and more effective than treating an event once it has already occurred. Our task is to predict impending hypoglycemia using preceding CGM.

Each sample $X _ { 1 : T }$ is a one-hour CGM sequence recorded at 5-minute intervals $( T = 1 3 )$ The binary outcome $Y$ indicates whether a hypoglycemic event occurs within the subsequent 30 minutes, defined as glucose $\leq 6 0$ mg/dL sustained for at least 20 minutes. The one-hour window itself must contain no hypoglycemia. Figure 4.3a illustrates an example trajectory.

For each experiment, we constructed 8,192 monitoring windows by block-bootstrapping the original CGM traces, up-sampling windows preceding hypoglycemia to achieve an event rate of approximately 0.3. The data were split into training (70%), validation (10%), and testing (20%) sets, with windows derived from the same hypoglycemia episode kept within the same partition to avoid leakage. The training set was used for learning model parameters, while the validation set was used for performance monitoring and stopping training. The experiments were repeated 50 times. We estimated sequential rules targeting a high-sensitivity constraint of $\beta = 0 . 9 5$ , reflecting clinical priorities to avoid missed hypoglycemia.

Figure 4.4 shows that the CGM setting exhibits the same oracle-like behavior observed in our synthetic experiments. The left panel presents the elbow shape of the specificity–cost trade-off. At the same time, the two rightmost bars show that the sequential rule consistently underuses cost when $\gamma > 0 . 7$ , indicating that the full hour is unnecessary for many monitoring windows. Across experiments, the sequential rule achieved near-maximal specificity $( \approx 0 . 8 )$ at an average cost of around 0.7, corresponding to a lead time of approximately 18 minutes. By contrast, the myopic baseline required the entire one-hour window to reach similar specificity.

To illustrate how the learned rule operates in practice, Figure 4.3 presents two example CGM trajectories and their corresponding decision processes. In both cases, the decision rule was learned under a sensitivity constraint of 0.95 and a cost level of 0.7. Figure 4.3b illustrates a monitoring window with declining glucose that is followed by a hypoglycemia episode. Observing the steep dive in glucose level around 40 to 50 minutes, the benefit of making an immediate classification outweighs that of waiting $( \zeta > \nu )$ , triggering an early warning. Figure 4.3c, in contrast, shows a monitoring window after which no hypoglycemia occurs. Under this stable glucose pattern, the rule continues to wait until the end of the window and issues no warning. Moreover, Figure 4.5 shows how the decision processes vary under different sensitivity and cost constraints. Overall, $\eta ,$ the benefit associated with a positive decision, tends to increase with $\beta ,$ as a stricter sensitivity requirement induces a stronger tendency toward positive conclusions (Figure 4.5a). Likewise, $\nu ,$ the benefit associated with continuation, tends to increase with $\gamma$ , because a looser constraint on cost or timeliness makes delaying the decision more favorable (Figure 4.5b).

![](images/4d433faca3efde4c6bf1d5c40bf9928c85998d304831f55522dd737b26687e28.jpg)

(a) An example one-hour monitoring window (shaded), followed by a hypoglycemia episode within 30 minutes.  
![](images/2a1a6e6ebb84b62bb3c092811054a447de133ac6b5ad733ca986a95c258c4576.jpg)  
(b) The CGM and decision processes during the monitoring window that is shaded in Figure 4.3a.

![](images/c11958bd8c15b8f085da77b4a4eddc7b3c88c4ff01ae67340e1deb18847423ba.jpg)  
(c) Another monitoring window not followed by hypoglycemia.  
Fig 4.3: Example CGM trajectories and corresponding decision processes. The upper panel illustrates the construction of a monitoring window (shaded) and the definition of a hypoglycemia episode. The bottom panels illustrate the decision rule learned under a sensitivity of 0.95 and a cost of 0.7. The bottom-left panel shows how the proposed decision rule operates on a window with declining glucose. Once the benefit of an immediate classification (ζ) outweighs that of waiting (ν) at $t = 5 0$ minutes, the rule issues a hypoglycemia warning based on the positive imminent risk signal $( \eta > 0 )$ . (The $\eta$ and $\zeta$ curves overlap in this panel due to a small Lagrangian multiplier: $\hat { a } = 0 . 0 4 5 . )$ The bottom-right panel shows a monitoring window with stable CGM, where the rule continues to wait until the end and issues no warnings.

This example demonstrates that the proposed sequential decision-rule framework can be applied to real-world data beyond synthetic settings. It delivers early, accurate warnings, satisfies stringent sensitivity constraints, and requires substantially fewer observations than myopic rules.

5. Discussion. Our contributions are threefold. First, we provide a statistical formulation of optimal sequential classification under a three-fold trade-off among specificity, sensitivity, and monitoring cost. Second, we develop a scalable learning procedure that combines recurrent neural networks with a primal–dual updating scheme to estimate the associated value processes. Finally, we demonstrate through simulations and a CGM application that the method yields interpretable, constraint-satisfying decision rules with strong empirical performance.

![](images/e0ca74d9c915e622098b6dcd20f0ba91e2409ef7d3c7741d880a38159043db61.jpg)  
Fig 4.4: Performance of the proposed sequential decision rules and the myopic baseline under a sensitivity constraint of $\beta = 0 . 9 5$ across cost $\gamma = 0 . 1 , \dots , 0 . 9$ . The leftmost panel illustrates the specificity–cost trade-off under the sensitivity constraint. The middle and rightmost panels examine whether the constraints were satisfied; the over cost (the rightmost panel) is defined as the achieved cost minus the desired cost. Means (dots) and 90% intervals (bars and ribbons) are shown over 50 repeated cross-validation experiments. To reduce visual clutter, the dots and bars for the myopic baseline are replaced with dotted lines and ribbons.

![](images/dc3b81d89f80930e1acdc150f7bdb51149086cf21f2abe27a66173dda9ccd843.jpg)

(a) Set $\gamma = 0 . 7$ and vary $\beta = 0 . 8 5 , 0 . 9 , 0 . 9 5$  
![](images/d1bed9f9d456e66c008df9dd9c01285b92745da03b98b07193cd50ac2aad445c.jpg)  
(b) Set $\beta = 0 . 9 5$ and vary $\gamma = 0 . 3 , 0 . 5 , 0 . 7$  
Fig 4.5: Effects of different sensitivity (β) and cost (γ) constraints on the decision processes for the non-hypoglycemia CGM trajectory (Figure 4.3c).

Several limitations and directions for further work remain. First, although we use flexible recurrent neural network architectures as estimators, we still rely on parametric function classes. Consequently, performance can be sensitive to how well the network models are specified and estimated, as observed in the simulation examples. Developing systematic diagnostic tools and calibration checks to guide model refinement is therefore an important direction for future work. Second, we have focused on binary outcomes and a scalar cost structure. Many applications require handling multiple event types, time-to-event outcomes, or richer, patient-specific cost functions that incorporate downstream resource use or harm. Extending the formulation to such settings is a natural next step. We are currently pursuing these extensions.

Acknowledgments. The authors would like to thank the anonymous referees, the Associate Editor and the Editor for their careful review and constructive comments that improved the quality of this paper.

## SUPPLEMENTARY MATERIAL

## Supplement A: Proofs and Additional Experiment Details

This supplement contains proofs and additional details for the numeric experiments.

## Supplement B: Code

Source code implementing the proposed method.

## REFERENCES

ACHENCHABE, Y., BONDU, A., CORNUÉJOLS, A. and DACHRAOUI, A. (2021). Early Classification of Time Series: Cost-based Optimization Criterion and Algorithms. Mach. Learn. 110 1481–1504. https://doi.org/10. 1007/s10994-021-05974-z

BILSKI, J. M. and JASTRZ ˛EBSKA, A. (2023). CALIMERA: A New Early Time Series Classification Method. Information Processing & Management 60 103465. https://doi.org/10.1016/j.ipm.2023.103465

CHUNG, J., GULCEHRE, C., CHO, K. and BENGIO, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. https://doi.org/10.48550/arXiv.1412.3555

DACHRAOUI, A., BONDU, A. and CORNUÉJOLS, A. (2015). Early Classification of Time Series as a Non Myopic Sequential Decision Making Problem. In Machine Learning and Knowledge Discovery in Databases (A. APPICE, P. P. RODRIGUES, V. SANTOS COSTA, C. SOARES, J. GAMA and A. JORGE, eds.) 433–447. Springer International Publishing, Cham. https://doi.org/10.1007/978-3-319-23528-8\_27

EBIHARA, A. F., MIYAGAWA, T., SAKURAI, K. and IMAOKA, H. (2021). Sequential Density Ratio Estimation for Simultaneous Optimization of Speed and Accuracy. https://doi.org/10.48550/arXiv.2006.05587

EBIHARA, A. F., MIYAGAWA, T., SAKURAI, K. and IMAOKA, H. (2024). Learning the Optimal Stopping for Early Classification within Finite Horizons via Sequential Probability Ratio Test. In The Thirteenth International Conference on Learning Representations.

LABER, E. B., LIZOTTE, D. J. and FERGUSON, B. (2014). Set-valued dynamic treatment regimes for competing outcomes. Biometrics 70 53–61.

LABER, E. B., LIZOTTE, D. J., QIAN, M., PELHAM, W. E. and MURPHY, S. A. (2014). Dynamic treatment regimes: Technical challenges and applications. Electronic journal ofstatistics 8 1225.

LIU, M., WANG, Y., FU, H. and ZENG, D. (2024a). Learning Optimal Dynamic Treatment Regimens Subject to Stagewise Risk Controls. Journal ofMachine Learning Research 25 1–64.

LIU, M., WANG, Y., FU, H. and ZENG, D. (2024b). Controlling Cumulative Adverse Risk in Learning Optimal Dynamic Treatment Regimens. Journal ofthe American Statistical Association 119 2622–2633. https://doi.org/ 10.1080/01621459.2023.2270637

LUCKETT, D. J., LABER, E. B., KAHKOSKA, A. R., MAAHS, D. M., MAYER-DAVIS, E. and KOSOROK, M. R. (2019). Estimating Dynamic Treatment Regimes in Mobile Health Using V-learning. Journal ofthe American Statistical Association 115 692.

MIETTINEN, K. (1999). Nonlinear Multiobjective Optimization. International Series in Operations Research & Management Science 12. Kluwer Academic Publishers, Boston.

MOODIE, E. E., RICHARDSON, T. S. and STEPHENS, D. A. (2007). Demystifying optimal dynamic treatment regimes. Biometrics 63 447–455.

MORI, U., MENDIBURU, A., DASGUPTA, S. and LOZANO, J. A. (2018). Early Classification of Time Series by Simultaneously Optimizing the Accuracy and Earliness. IEEE Transactions on Neural Networks and Learning Systems 29 4569–4578. https://doi.org/10.1109/TNNLS.2017.2764939

MURPHY, S. A. (2003). Optimal Dynamic Treatment Regimes. Journal ofthe Royal Statistical Society: Series B (Statistical Methodology) 65 331–355. https://doi.org/10.1111/1467-9868.00389

PAN, Y., LABER, E. B., SMITH, M. A. and ZHAO, Y.-Q. (2023). Reinforced Risk Prediction With Budget Constraint Using Irregularly Measured Data From Electronic Health Records. Journal ofthe American Statistical Association 118 1090–1101. https://doi.org/10.1080/01621459.2021.1978467

PARDALOS, P. M., ŽILINSKAS, A. and ŽILINSKAS, J. (2017). Non-Convex Multi-Objective Optimization. Springer Optimization and Its Applications 123. Springer International Publishing, Cham. https://doi.org/10. 1007/978-3-319-61007-8

QIU, J., ZHENG, Y. and ZHAO, Y.-Q. (2026). Supplement to “Primal–Dual Alternating Neural Learning for Timely Classification with Performance Guarantees”. https://doi.org/10.1214/[providedbytypesetter]

RENAULT, A., BONDU, A., CORNUÉJOLS, A. and LEMAIRE, V. (2025). Early Classification of Time Series: A Survey and Benchmark. Transactions on Machine Learning Research.

SCHÄFER, P. and LESER, U. (2020). TEASER: Early and Accurate Time Series Classification. Data Mining and Knowledge Discovery 34 1336–1362. https://doi.org/10.1007/s10618-020-00690-z

SHI, C., LUO, S., LE, Y., ZHU, H. and SONG, R. (2024). Statistically efficient advantage learning for offline reinforcement learning in infinite horizons. Journal ofthe American Statistical Association 119 232–245.

TONG, X., FENG, Y. and LI, J. J. (2018). Neyman-Pearson Classification Algorithms and NP Receiver Operating Characteristics. Science Advances 4 eaao1659. https://doi.org/10.1126/sciadv.aao1659

WEINSTOCK, R. S., DUBOSE, S. N., BERGENSTAL, R. M., CHAYTOR, N. S., PETERSON, C., OLSON, B. A., MUNSHI, M. N., PERRIN, A. J. S., MILLER, K. M., BECK, R. W., LILJENQUIST, D. R., ALEPPO, G., BUSE, J. B., KRUGER, D., BHARGAVA, A., GOLAND, R. S., EDELEN, R. C., PRATLEY, R. E., PETERS, A. L., RODRIGUEZ, H., AHMANN, A. J., LOCK, J.-P., GARG, S. K., RICKELS, M. R., HIRSCH, I. B. and FOR THE T1D EXCHANGE SEVERE HYPOGLYCEMIA IN OLDER ADULTS WITH TYPE 1 DIABETES STUDY GROUP (2015). Risk Factors Associated With Severe Hypoglycemia in Older Adults With Type 1 Diabetes. Diabetes Care 39 603–610. https://doi.org/10.2337/dc15-1426

YU, Z., LI, Y., KIM, J., HUANG, K., LUO, Y. and WANG, M. (2023). Deep Reinforcement Learning for Cost-Effective Medical Diagnosis. https://doi.org/10.48550/ARXIV.2302.10261

ZHAO, Y.-Q., ZENG, D., LABER, E. B. and KOSOROK, M. R. (2015). New Statistical Learning Methods for Estimating Optimal Dynamic Treatment Regimes. Journal ofthe American Statistical Association 110 583–598. https://doi.org/10.1080/01621459.2014.937488

ZHOU, W., ZHU, R. and QU, A. (2024). Estimating optimal infinite horizon dynamic treatment regimes via pt-learning. Journal ofthe American Statistical Association 119 625–638.