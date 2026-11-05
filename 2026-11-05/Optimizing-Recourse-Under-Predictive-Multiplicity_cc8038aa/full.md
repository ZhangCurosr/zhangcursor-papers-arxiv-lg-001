# Optimizing Recourse Under Predictive Multiplicity

Shahin Jabbari<sup>∗</sup>

## Abstract

Algorithmic recourse identifies modifications that can change an unfavorable model prediction into a favorable one. Under predictive multiplicity, similarly performing models can disagree on individual predictions, so recourse valid for one model may fail for another. We formulate robust recourse under predictive multiplicity as a budget-constrained minimization of the worst-case validity loss across a competing set of models, which can include models with diferent architectures or from diferent hypothesis classes. We propose an oracle-eficient algorithm called ORPM and prove suficient conditions for its convergence. We evaluate ORPM on one synthetic and five real datasets, and competing sets consist of linear, neural, mixed, and decision-tree models. Our experiments show decreasing optimization-gap bounds and high validity in many tested settings, with favorable performance against baselines designed for specific model classes or architectures. We also quantify how requiring a common action across competing models changes recourse success at a given budget and examine disparities in this success across demographic groups.

## 1 Introduction

Algorithmic recourse provides individuals who receive unfavorable decisions from automated decision-making systems (e.g., loan denial) with guidance on modifying their features to achieve a favorable outcome (e.g., loan approval) [62, 64]. In practice, decision-makers may have access to competing set of models with similar predictive performance but diferent decisions for specific individuals, a phenomenon known as predictive multiplicity [7, 43]. Therefore, for an individual, the modifications can depend on which similarly accurate model a decision-maker selects. Recourse computed for one model may therefore become invalid for other models in the set, undermining its reliability [48].

Robust algorithmic recourse addresses this unreliability by accounting for model uncertainty [22, 44, 61]. Some approaches consider bounded changes in model parameters [61], while others target particular model classes [18]. Recently, Turbal et al. [60] studied recourse under model multiplicity and proposed an ellipsoidal approximation of the Rashomon set for linear models and neural networks sharing a fixed feature extractor, with variation confined to the last-layer parameters. However, competing models may have diferent architectures or even belong to diferent classes [5]. This motivates designing a framework that directly considers diverse competing sets without requiring a shared architecture or closeness in parameters.

Our Contributions and Results. We formulate robust recourse under predictive multiplicity as a budget-constrained minimization of the worst-case validity loss across competing models, allowing models with diferent architectures or from diferent classes. To solve this problem, we develop an oracle eficient algorithm, called Oracle-Eficient Robust Recourse under Predictive Multiplicity (ORPM), which is inspired by dynamics for computing equilibria in two-player games [21]. Under bounded losses and controlled global oracle error, we prove convergence to an approximate equilibrium for a uniform mixture of oracle responses. In the convex setting, the same guarantee applies to their average point. We show how an oracle can be designed for diferentiable and tree-based competing sets.

Empirically, using one synthetic and five real-world datasets, we evaluate ORPM on linear, neural, mixed, and tree-based competing sets. Our convergence experiments show that validity can be achieved before the prescribed optimization-gap target is certified. ORPM achieves high joint validity in many tested settings and often outperforms or matches prior approaches, even in settings where the competing set only contains models with the same architecture.

We further examine how the recourse budget and size or complexity of the competing set afect validity. We observe that larger budgets generally improve validity, while acceptance by more diverse models can substantially reduce it. On competing sets that contain models with the same model class and architecture, we show that ORPM competes favorably with prior approaches that are specialized for such competing sets. We also empirically study how multiplicity afects the validity and disparity of recourse. These results characterize both ORPM’s practical strengths and highlight the challenges of providing recourse that is robust to predictive multiplicity.

## 2 Related Work

Algorithmic Recourse. Algorithmic recourse provides low-cost modifications needed to change a model’s prediction for an input with an undesirable label [62, 64]. Since its introduction, researchers have proposed various recourse formulations and studied aspects such as feasibility, fairness, improvement, and competition [4, 10, 28, 32, 36, 37, 38, 49, 51, 54, 55]. See [33, 63] for surveys.

Predictive Multiplicity. Predictive multiplicity refers to the existence of multiple models that achieve similar predictive performance yet difer in their individual-level predictions [13, 19, 56, 57]. A Rashomon set is a set of models from the same model class that perform similarly to a reference model [7]. In this work, we call the finite collection of similar-performing models from potentially diferent model classes the competing set. Marx et al. [43] formalize this phenomenon for classification, and Watson-Daniels et al. [65] extend the analysis to probabilistic classifiers. Subsequent works quantify predictive multiplicity and the size of Rashomon sets [25, 57], eficiently explore Rashomon sets [27, 66] and study the efects of multiplicity in downstream tasks [17, 29].

Recourse Under Predictive Multiplicity. Several works study the challenges posed by predictive multiplicity for algorithmic recourse [2, 48, 51]. Turbal et al. [60] study robust recourse over an ellipsoidal approximation of the Rashomon set of linear models and neural networks with a fixed feature extractor and diferent last-layer parameters. Leofante et al. [41] define robust counterfactuals for models with compatible inputs and outputs. Their product-network construction focuses on homogeneous neural networks, with a complexity extension to piecewise-linear models. Noorani et al. [46] study ensemble recourse through cost and worst-case loss and introduce an entropic-risk formulation whose limiting case recovers worst-case robustness. We study budget-constrained worst-case loss; our approach is oracle-eficient and can be applied to a competing set that contains models from diferent model classes. We compare our approach to [60] and a baseline (MinL2) that adapts the joint neural feasibility formulation [41].

Robust Recourse. Robust recourse studies how the validity of recourse can be preserved under model uncertainty. An early influential paper proposes ROAR, designed to find valid recourses under bounded shifts in model parameters [61]. Similar to recourse, various formulations for robust recourse have been explored [8, 10, 22, 23, 30, 31, 34, 38, 39, 44, 45, 67]. Specifically, for tree-based models, OCEAN uses mixed-integer programs to generate recourse for a fixed tree ensemble [47], while FOCUS uses diferentiable approximations, enabling gradient-based search [42]. Robustness is addressed through a stability criterion (RobX) [18] and probabilistic constraints to target the randomness of the ensemble [20]. See [31] for a survey. Our formulation permits a specified finite set of compatible models from diferent classes without requiring proximity in parameter space. Our contribution concerns the budgeted formulation, oracle implementations, conditional guarantees, and empirical analysis, rather than introducing heterogeneous-model robustness itself. We compare our approach to many of these prior robust approaches in Section 5.

Equilibrium Computation in Two-Player Games. Computing equilibrium using dynamics involving best-response and no-regret approaches has a long history, including the celebrated work of Freund and Schapire [21] and subsequent works [3, 14, 52]. The oracle-eficient approach has also been extensively studied in the algorithmic fairness literature [1, 15, 35, 58]. Our analysis specializes the oracle-based robust-optimization framework of [9] to recourse.

## 3 Problem Formulation

Consider a binary classification task where the input space is $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ (representing individuals), and the label space is $\mathcal { V } = \{ 0 , 1 \}$ , where label 0 (1) denotes an unfavorable (favorable) outcome, $\mathrm { e . g . }$ , loan denial (loan approval). Let D denote the distribution over $\mathcal { X } \times \mathcal { V }$ . Let $\mathcal { H } \subseteq [ 0 , \mathrm { i } ] ^ { \mathcal { X } }$ be the hypothesis class of probabilistic models, and $h \in \mathcal H$ be a probabilistic classifier deployed by a decision-maker, where $h : \mathcal { X }  [ 0 , 1 ]$ The probability output of the classifier h can be converted to a deterministic label in $\mathcal { V }$ by thresholding, and we use ${ \hat { y } } _ { h } ( x )$ to denote this deterministic label.

Let $c : \mathcal { X } \times \mathcal { X } \to \mathbb { R } ^ { \geq 0 }$ be a cost function quantifying the cost of modifying any input to another. We assume $c ( x , x ) = 0$ . For any given $x _ { 0 } \in \mathcal { X }$ and $h \in \mathcal H$ , if the $\hat { y } _ { h } ( x _ { 0 } ) = 0 $ , then the decision-maker’s goal in algorithmic recourse [62, 64] is to find an $x ^ { \prime } \in { \mathcal { X } }$ such that ${ \hat { y } } _ { h } ( x ^ { \prime } ) = 1 \ ( \mathrm { i . e . } $ , a valid recourse) and $c ( x _ { 0 } , x ^ { \prime } ) \leq \delta _ { \mathrm { { f } } }$ where $\delta > 0$ . In this formulation, δ is a (potentially instance-dependent) budget parameter and demonstrates the maximum cost an individual is willing and able to incur to receive the favorable label. This formulation captures the limits on the resources available for implementing recourse, and even the least costly valid recourse may be unafordable. We treat $\delta$ as fixed for each optimization but vary $\delta$ across experiments to study the trade-of between cost and validity [50].

We consider a decision-maker that uses a budget-constrained relaxed formulation of recourse [61]:

$$
x ^ { \prime } \in \mathop { \arg \operatorname* { m i n } } _ { x \in \mathcal { X } } \ell ( h ( x ) , 1 ) \mathrm { ~ s . t . ~ } c ( x _ { 0 } , x ) \leq \delta ,\tag{1}
$$

where $\ell : \mathbb { R } \times \mathbb { R } \to \mathbb { R } ^ { \geq 0 }$ is a surrogate loss function chosen such that $\ell ( p , 1 )$ is strictly decreasing in $p \in [ 0 , 1 ]$ $( \mathrm { e . g . } ,$ squared loss). Therefore, minimizing the loss is a proxy for maximizing validity.

In the presence of model multiplicity [7, 43], there exists a set of competing models that can be used by the decision-maker. We assume this set has n models and use $H = \left\{ h _ { 1 } , \ldots , h _ { n } \right\} \subseteq { \mathcal { H } }$ to refer to it. Our formulation allows a finite but otherwise arbitrary competing set. In particular, we allow the models in H to belong to diferent model classes. Let $h \in H$ denote the current model used by the decision-maker to label individual $x _ { 0 } \in \mathcal { X }$ . When $\hat { y } _ { h } ( x _ { 0 } ) = 0$ , the goal in recourse in the presence of model multiplicity is ideally to find a robust recourse $x ^ { * } \in \mathcal { X }$ that is valid under every model in the competing set, i.e., to satisfy $\hat { y } _ { h } ( x ^ { * } ) = 1$ for all $h \in H$ , and again $x ^ { * }$ needs to satisfy $c ( x _ { 0 } , x ^ { * } ) \leq \delta$

However, such a valid recourse need not exist: either because the budget may be insuficient to reach a favorable point for all models or the favorable decision regions of diferent models in the competing set may have no common intersection. Despite this, similar to the case of a single classifier, a relaxed version of the problem can be written as:

$$
x ^ { * } \in \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { x \in \mathcal { X } } \operatorname* { m a x } _ { h \in H } \ell ( h ( x ) , 1 ) \mathrm { ~ s . t . ~ } c ( x _ { 0 } , x ) \le \delta ,\tag{2}
$$

where ℓ is the surrogate loss defined above. Minimizing the first term in Equation 2 is a proxy for maximizing the worst-case validity across all competing models. Under the monotonicity assumption, this is equivalent to maximizing the lowest score across all models in the competing set, within the budget. We emphasize that while the main motivation of our work is to generate a recourse robust to model multiplicity, our formulation does not require architectural similarity or parameter-space proximity among the models in H.

## 4 Algorithm and Analysis

To solve the robust recourse problem defined in Equation 2, we model the optimization as a zero-sum game between the decision-maker who generates the recourse and an adversary who selects the worst-case model for the generated recourse. Let $K : = \{ x \in \mathcal { X } : c ( x _ { 0 } , x ) \leq \delta \}$ denote the feasible recourse set and $\begin{array} { r } { \mathcal { L } ( x , w ) : = \sum _ { i = 1 } ^ { n } w _ { i } \ell ( h _ { i } ( x ) , 1 ) } \end{array}$ . We allow the adversary to select a distribution $w \in \Delta ( H ) = \{ w \in \mathbb { R } ^ { n } : w _ { i } \geq$ $0 , \ \textstyle \sum _ { i = 1 } ^ { n } w _ { i } = 1 \}$ over the competing models and solve $x ^ { * } \in$ arg min $_ { x \in K } \operatorname* { m a x } _ { w \in \Delta ( H ) } \mathcal { L } ( x , w )$ instead. This is without loss of generality, as the objective is linear in $w .$ so its maximum over the simplex is attained by placing all mass on a worst-case model. Our approach separates the game dynamics from the optimization of recourse for a fixed model distribution, utilizing an approximate oracle.

Assumption 4.1 (Approximate Oracle). Given any distribution $w \in \Delta ( H )$ , an ϵ-approximate oracle $\mathcal { O } _ { x _ { 0 } , \ell , c }$ returns a feasible recourse ${ \tilde { x } } \in K$ satisfying $\begin{array} { r } { \mathcal { L } ( \tilde { x } , w ) \leq \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , w ) + \epsilon . } \end{array}$ , where $\epsilon \geq 0$ bounds the additive global error.

For notational convenience, we suppress the dependence of the oracle on $H , \delta ,$ and ϵ. The error bound concerns global optimality and may be established analytically or using optimization certificates.

We propose the Oracle-Eficient Robust Recourse under Predictive Multiplicity (ORPM) algorithm. In each round, the decision-maker plays an approximate best response to $w _ { t }$ by invoking the oracle (line 4). The adversary then uses multiplicative weights [3] to shift probability mass toward models with higher loss at the current response (lines 5–7)

```latex
ALGORITHM 1: Oracle-Eficient Robust Recourse under Predictive Multiplicity (ORPM)
1: Inputs: Initial instance $x _ { 0 } ,$ , loss $\ell ,$ cost $c ,$ competing set $\overline { { H = \{ h _ { 1 } , \ldots , h _ { n } \} } }$ , oracle $\mathcal { O } _ { x _ { 0 } , \ell , c } ,$ budget δ, learning rate
$\eta ,$ number of rounds $T .$
2: Initialize: $w _ { 1 , i } = 1 / n$ for $i = 1 , \ldots , n .$
3: for $t = 1$ to $T$ do
4: $x _ { t }  \mathcal { O } _ { x _ { 0 } , \ell , c } ( w _ { t } ) .$ ▷ Query the oracle for the optimal recourse for current weights
5: $v _ { t , i } \gets \ell ( h _ { i } ( x _ { t } ) , 1 )$ for $i = 1 , \ldots , n .$ ▷ Compute the loss vector for the generated recourse
6: $\widetilde { w } _ { t + 1 , i } \gets w _ { t , i } \exp ( \eta v _ { t , i } )$ for $i = 1 , \ldots , n .$ ▷ Update weights to reward models with high loss
7: $\begin{array} { r } { w _ { t + 1 , i }  \widetilde { w } _ { t + 1 , i } / \sum _ { j = 1 } ^ { n } \widetilde { w } _ { t + 1 , j } } \end{array}$ for $i = 1 , \ldots , n .$ ▷ Normalize the weights to sum to 1
8: end for
9: Return: Response sequence $( x _ { 1 } , \dots , x _ { T } )$ and $\bar { w } = T ^ { - 1 } \sum _ { t = 1 } ^ { T }$ w<sub>t</sub>; obtain a deterministic recourse using the return
rules below.
```

ORPM makes $T$ oracle calls, $n T$ model-loss evaluations, and $O ( T ( n + d ) )$ additional arithmetic operations. The cost of each oracle call depends on the competing set and the weighted recourse problem.

Convergence Guarantees. Let $x _ { 1 : T } : = ( x _ { 1 } , \ldots , x _ { T } )$ denote the sequence of oracle responses, and let ${ \bar { x } } : = T ^ { - 1 } \sum _ { t = 1 } ^ { T } x _ { t }$ and ${ \bar { w } } : = T ^ { - 1 } \sum _ { t = 1 } ^ { T } w _ { t }$ . Also let $\epsilon _ { t }$ denote the global error bound for the oracle call at round t, i.e., $\begin{array} { r } { \mathcal { L } ( x _ { t } , w _ { t } ) \leq \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , w _ { t } ) + \epsilon _ { t } } \end{array}$ . We allow these bounds to vary across calls. If the same accuracy guarantee ϵ holds for every call, then $\epsilon _ { t } \ \leq \ \epsilon$ for every t. Define the mixture gap as: $\begin{array} { r } { \mathrm { G a p } _ { \operatorname* { m i x } } ( x _ { 1 : T } , \bar { w } ) : = \operatorname* { m a x } _ { w \in \Delta ( H ) } \frac { 1 } { T } \sum _ { t = 1 } ^ { T ^ { \prime } } \mathcal { L } ( x _ { t } , w ) - \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , \bar { w } ) } \end{array}$ . Consider the strategy pair in which the decision-maker selects one of $x _ { 1 } , \ldots , x _ { T }$ uniformly at random and the adversary independently selects a model according to w¯. The mixture gap measures the sum of the players’ possible improvements: the adversary can increase the expected loss by changing its distribution, while the decision-maker can decrease it by changing its recourse. In each case, the other player’s strategy remains fixed. We require the following assumption.

Assumption 4.2 (Bounded Loss). There exists $B > 0$ such that $\ell ( h _ { i } ( x ) , 1 ) \in [ 0 , B ]$ for every $x \in K$ and $i \in \{ 1 , \ldots , n \}$

For probabilistic predictions, squared loss satisfies this assumption with $B = 1$ . Our main result is as follows.

Theorem 4.3 (Oracle-Dependent Convergence). Under Assumptions 4.1 and 4.2, let $\bar { \epsilon } _ { T } : = T ^ { - 1 } \sum _ { t = 1 } ^ { T } \epsilon _ { t }$ . For $n \geq 2$ and any $\eta > 0$ , the responses and average weights produced by Algorithm 1 satisfy $\mathrm { G a p } _ { \operatorname* { m i x } } ( x _ { 1 : T } , \bar { w } ) \leq$ $\begin{array} { r } { \frac { \ln n } { \eta T } + \frac { \eta B ^ { 2 } } { 8 } + \bar { \epsilon } _ { T } } \end{array}$ . In particular, setting $\eta = B ^ { - 1 } { \sqrt { \frac { \ln n } { T } } }$ gives $\begin{array} { r } { \mathrm { G a p } _ { \operatorname* { m i x } } ( x _ { 1 : T } , \bar { w } ) \leq 2 B \sqrt { \frac { \ln n } { T } + \bar { \epsilon } _ { T } } } \end{array}$ . For $n = 1$ , the mixture gap is at most $\bar { \epsilon } _ { T }$ .

All proofs are deferred to Appendix A. Theorem 4.3 does not require convexity of the models’ losses or the feasible set. It applies to nonconvex problems whenever the oracle has controlled global error (e.g., oracles with suitable optimization certificates). With a uniform error bound $\epsilon ,$ the gap is at most $\epsilon + O ( \sqrt { \ln n / T } )$ Allowing the decision-maker to randomize can improve the objective relative to any single recourse, but this guarantee alone does not establish convergence to an optimal deterministic recourse. We next study the convex setting where we can guarantee convergence to deterministic recourse. We need the following assumption.

Assumption 4.4 (Convexity and Compactness). The feasible recourse set $K$ is compact and convex, and the mapping $x \mapsto \ell ( h _ { i } ( x ) , 1 )$ is continuous and convex on K for every $i \in \{ 1 , \ldots , n \}$

Norm-based cost constraints and convex feature restrictions yield convex feasible sets. Under Assumption 4.4, together with linearity in $w ,$ the minimax theorem guarantees an equilibrium for the deterministic recourse game [59]. Define the point gap as $\begin{array} { r } { \mathrm { G a p } ( x , w ) : = \operatorname* { m a x } _ { u \in \Delta ( H ) } \mathcal { L } ( x , u ) - \operatorname* { i n f } _ { z \in K } \mathcal { L } ( z , w ) } \end{array}$

Corollary 4.5 (Convergence of the Average Recourse). Suppose Assumptions 4.1, 4.2, and 4.4 hold. For $n \geq 2$ , run Algorithm 1 with $\eta = B ^ { - 1 } { \sqrt { \frac { \ln n } { T } } }$ . Then the average ${ \bar { x } } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } x _ { t }$ of the response sequence returned by Algorithm 1 is feasible and satisfies $\mathrm { G a p } ( \bar { x } , \bar { w } ) \leq \mathrm { G a p } _ { \operatorname* { m i x } } ( x _ { 1 : T } , \bar { w } ) \leq 2 B \sqrt { \frac { \ln n } { T } } + \bar { \epsilon } _ { T } .$

The distinction between Theorem 4.3 and Corollary 4.5 separates the role of global oracle accuracy from that of convexity: the former controls convergence of the response distribution, while the latter allows us to return its average responses with the same guarantee. In Corollary 4.5, if $\epsilon _ { t } \leq \epsilon$ for all t, the final term may be replaced by ϵ.

Oracle Design. We next describe how an oracle can be implemented for convex and specific non-convex settings. Additional details can be found in Appendix B.2.

1. Convex Setting: For afine logits, both squared and logistic loss are strictly decreasing functions of the logit. Consequently, their deterministic minimax problems have the same optimizers and admit the exact reformulation (maximize the smallest logit across the competing models) that, combined with an $L _ { 2 }$ budget and afine feature constraints, leads to a second-order cone program [6]. Logistic loss also makes each weighted response problem convex, allowing supporting-plane lower bounds to certify its numerical optimization error. Weighted squared-probability loss need not be convex, and replacing it changes the response problems even though the deterministic minimax optimizers coincide.

2. Diferentiable Nonconvex Setting: For diferentiable nonconvex competing sets, our implementation uses local SLSQP response optimization. These local solves do not establish the global oracle-error bounds required by Theorem 4.3. We therefore assess their performance empirically. See Section 5.

3. Tree-Based Nonconvex Setting: Decision trees have piecewise-constant predictions, so their input gradients do not provide useful search directions. Following the mixed-integer modeling approach of Parmentier and Vidal [47], for competing sets containing decision trees, our oracle encodes the competing tree models jointly and minimizes their weighted loss subject to budget and feature constraints. If a feasible response has weighted loss $u _ { t } = \mathcal { L } ( x _ { t } , w _ { t } )$ and the solver provides a valid global lower bound $b _ { t }$ , then $\epsilon _ { t } = u _ { t } - b _ { t }$ is a global oracle-error bound. Thus, even a time-limited solver can support Theorem 4.3, although the resulting bound may be loose. The solver’s bounds apply to the budget and feature constraints represented in the optimization problem, up to its numerical tolerances. For a deterministic tree-based recourse, we return the candidate with the smallest worst-model loss among the original input and all oracle responses, breaking ties by cost. Combining this candidate’s worst-model loss with the solver’s global lower bounds lets us bound how far its loss is from the optimal deterministic recourse loss, without requiring convexity. See Appendices A and B.3.

## 5 Experiments

We empirically study the performance of ORPM in convex and non-convex settings. Code and data are available here.

## 5.1 Datasets

We used six datasets: one synthetic dataset and five real-world datasets. The synthetic dataset is twodimensional. There are equal observations for each class, and they are generated from a multivariate norma distribution with means $\mu _ { 0 } = [ - 3 , - 3 ]$ and $\mu _ { 1 } = [ 3 , 3 ]$ , respectively, and a shared covariance matrix $\Sigma = 5 \mathbb { I }$ The real-world datasets include Correctional Ofender Management Profiling for Alternative Sanctions (COMPAS), Polish Companies Bankruptcy (Polish), German Credit (German), Give Me Some Credit (GMC), and ACSIncome [16] from Folktables. The datasets range from 2 to 67 features. For COMPAS, Polish, and German, we build on the preprocessing code associated with [60]. Detailed information about the datasets, splitting, and standardization procedures is described in Appendix B.1.

## 5.2 Implementation Details

We used logistic regression, neural networks with one or two hidden layers, and decision trees as models. We constructed models for the competing sets using randomized model training (on a subset of features to generate diversity) or from a Hessian-based ellipsoidal approximation to the Rashomon set [60]. We then filtered these models by validation accuracy to form competing sets. Evaluation used repeated stratified training, validation, and test splits, with feature standardization fitted on the training split. We generated recourse only for held-out instances rejected by at least half of the models in the competing set.

We used the $L ^ { 2 } \mathrm { - n o r m }$ in standardized feature space as the recourse cost $c ,$ enforcing no changes on dataset-specific immutable features. The diferentiable trade-of and same-architecture neural experiments use raw logistic loss and local SLSQP response optimization. Tree experiments use squared probability loss and time-limited mixed-integer response optimization. The convex duality-gap experiment uses normalized logistic loss. Budgets and compositions/size of the competing set are specified with each experiment. Unless stated otherwise, experiments use 10 seeds and 20 instances per seed. Results are averaged first across instances within each seed and then across seeds. For the stratified cost/disparity results, we weight subgroup success rates by the study-specific group proportions within each seed, then average equally across seeds. These weighted estimates need not occur in fixed increments of 0.05 or 0.005. See Appendix B.2 for more details.

Baselines. We next discuss the main baselines used in our trade-of experiments. We also use additional experiment-specific baselines, which we introduce throughout Section 5.3. For diferentiable competing sets, we compared ORPM with 3 baselines: (1) a budget-constrained implementation of ROAR [61], which optimizes against a bounded parameter uncertainty set, (2) ElliCE [60], which uses a Hessian-based ellipsoid to approximate the Rashomon set, and (3) ADV, a baseline built on the idea of adversarial training that optimizes against the largest loss among the competing models. For neural networks, ROAR uses local linear logit surrogates, whereas ElliCE considers last-layer parameter uncertainty with the feature extractor fixed. For tree-based competing sets, we compared ORPM with 4 baselines: (1) ADV-surrogate, which is a diferentiable tree approximations inspired by FOCUS [42], (2) ROAR-surrogate, which applies ROAR to local linear logit surrogates, (3) RobX-constrained which adapts RobX [18] to the budget and immutable-feature constraints, and (4) OCEAN-style which implements minimum-cost recourse for the deployed model using the mixed-integer formulation approach of [47]. See Appendix B.2 for more details.

## 5.3 Findings

Duality Gap. We ran ORPM on all datasets using 5 seeds and 5 instances per seed for 1000 rounds. We experimented on competing sets of size 6, 12, and 24 and three types of competing sets: the first type only contains logistic regression models (linear), the second contains an equal number of one-layer and two-layer neural networks (neural), and the third contains an equal number of logistic regression, one-layer, and two-layer neural networks (mixed). We used diferent budgets for each dataset. See Appendix B.3.1. Upon termination and after several checkpoint rounds, we measured the validity of recourse and upper and lower bounds on the averaged-point gap (using, e.g., convex bounds) with a gap of ϵ = 0.005 as the target.

![](images/395a5a712199d5fa675d9b7bada23260dcf9a1595fe1e2b7d4f25c41e318dd03.jpg)  
(a) COMPAS dataset, 6 competing models, δ = 2.

![](images/5e2166bf4df17ce8f11de201423331bcaea49601a5fb062ce34d0181ed12bad4.jpg)  
(b) Polish dataset, 24 competing models, δ = 2.  
Figure 1: Fraction of instances whose certified upper bound on the averaged-point gap is at most 0.005 at each checkpoint. Each subfigure corresponds to a diferent dataset. The number of models and budget are mentioned in the caption.

Figure 1 depicts the fraction of instances that achieve their upper bound on the gap and meet this stringent threshold. This fraction is averaged over seeds and instances and is depicted (Y-axis) versus a number of rounds (X-axis) for COMPAS (left panel) and Polish (right panel). In each subfigure, each curve tracks a competing set of linear, neural, or mixed models. The size of the competing set and the budget are mentioned in the caption. We observe that the fraction of recourses with a duality gap upper bound at or below the threshold is generally higher for linear competing sets by the end of 1000 rounds, even though other competing sets can initially achieve a higher success fraction. Across 54 dataset/competing set/budget configurations, when increasing the number of models from 6 to 24, the fraction of instances achieving the target gap has decreased in 22 comparisons, remained unchanged in 28 comparisons, and only increased in 4. The efect of budget is less uniform; increasing the budget increased the fraction in 23 comparisons, left it unchanged in 22, and decreased it in 9 comparisons. See Figure 8 in Appendix B.3.1 for other datasets.

To understand whether more rounds are required to improve the validity of ORPM, we compared the validity of recourse returned by ORPM at rounds 100 and 200 with the validity of recourse at round 1000. Outside GMC, the validity is identical from round 200 to round 1000 for 132 out of 135 dataset/model class/model count/budget configurations, and the increase in the other 3 configurations is around 4%. In 100 rounds, in 129 of the 135 configurations, validity is the same as in 1000 rounds. In GMC, however, the validity gains are significantly higher when scaling from 100 or 200 rounds to 1000 rounds. Based on these observations, we used a default choice of 200 for all datasets except GMC and 1000 for GMC for the remainder of the experiments on linear, neural network, and mixed models.

Tradeof Between Validity and Budget. We next study the validity of recourse returned by ORPM, where we define a recourse as jointly valid (or simply valid) if all the models provide a favorable prediction. In particular, we use the same combinations of models, model counts, and datasets as in the convergence experiments, but this time we use 10 seeds and compute recourse for 20 instances per seed, and all results are averaged over instances and seeds. We expand the set of budgets used compared to the convergence experiments (see Appendix B.3.2).

Figure 2 depicts the trade-of between the budget (X-axis) and the validity of recourse (Y-axis). Each subfigure corresponds to a combination of dataset, competing set, and the number of competing models as

![](images/86c777743287a930ba44637a550178708ad11012ac061727eb978e53231d7343.jpg)  
(a) German dataset, mixed competing set of 12 models.

![](images/ed9849b11d4bc56f060de79a41bc466f7b4f56035659efe974f1f5d2ef4ad7bb.jpg)  
(b) ACSIncome dataset, neural competing set of 24 models.

Figure 2: The trade-of between validity and budget. Each subfigure corresponds to a dataset and competing set as described in the caption. In each subfigure, the curves track the validity of ORPM and baselines (ROAR [61], ElliCE [60], ADV, and Reference).

indicated in the caption. In addition to ORPM, we measure the validity for baselines ROAR [61], ElliCE [60], and ADV. We also add another baseline called Reference, where we simply optimize for recourse for the reference model in the competing set without taking into account whether such recourse would be valid for other models in the competing set. Results for other datasets are in Figure 9 in Appendix B.3.2.

We observe that in all datasets except GMC, ORPM achieves the highest validity over the baseline across all budgets. For GMC, other approaches outperform ORPM for mixed and neural network models, but ORPM outperforms for linear models (see Figure 9). ADV generally has the second-highest validity after ORPM, while ElliCE performs better on linear than on nonlinear sets. ROAR and ElliCE address diferent uncertainty sets, and our results assess their stated adaptations on our finite competing sets.

![](images/39de8c7ec76473480cb5e1a24637c3b93a83f703d9605b71ab6216c6037435cb.jpg)  
(a) The efect of budget on ORPM’s validity for a mixed competing set of 12 models.

![](images/9615678acc801f5d352e0cd5a69a565fe6a5cfb51babd135f5775bf7470a40e5.jpg)  
(b) The efect of composition of the competing set of 6 models on ORPM’s validity.  
Figure 3: The efect of budget and composition of the competing set on validity. Each curve corresponds to a diferent dataset.

Increasing the budget will generally result in ORPM achieving perfect (or near-perfect) validity in several settings. An example of this behavior is provided in Figure 3a. In this figure, for a mixed competing set of

12 models, we track the validity of ORPM across each dataset as the budget increases. However, residual failures remain in some configurations, including ACSIncome with 24 neural models (see e.g., Figure 2b). For many settings, we observe 3 regimes: a low budget regime where the validity remains low, a medium budget regime where the validity sharply increases, and a high budget regime where the validity is near perfect and the rate of validity improvement is significantly lower compared to the medium budget regime. See Figure 10 in Appendix B.3.2 for more details.

The validity generally decreases when the competing set contains both linear and neural models compared to the case where it contains only linear or neural models. In general, achieving high validity for neural/mixed models is more dificult than for linear models. Figure 3b tracks the validity of recourse returned by ORPM (Y-axis) for diferent compositions of competing sets containing 6 classifiers (X-axis). We also observe that validity drops most significantly on GMC, but the drop is generally milder on other datasets. These observations are fairly robust across competing sets of diferent sizes. Observed validity generally decreases with competing-set size, although notable exceptions occur, including GMC. Moreover, the validity degradation is more pronounced for neural and mixed competing sets, compared to linear competing sets. See Figure 11 in Appendix B.3.2 for more details.

Homogeneous Competing Sets. We next study the validity of the provided recourse when the competing set contains homogeneous models and compare with prior approaches that are specifically designed to handle such competing sets. In particular, we focus on two types of competing sets: one only containing linear models and another consisting of neural networks with the same architecture but diferent weights.
<table><tr><td>Dataset</td><td>δ Set</td><td>ORPM</td><td>ElliCE</td></tr><tr><td>Synthetic</td><td>2 Seen</td><td> ${ \bf 7 9 . 0 \pm 8 . 4 }$ </td><td> $6 9 . 5 \pm 7 . 2$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 7 8 . 5 \pm 8 . 3 }$ </td><td> $7 0 . 0 \pm 7 . 5$ </td></tr><tr><td></td><td>5 Seen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td>COMPAS</td><td>2 Seen</td><td> ${ \bf 5 9 . 5 \pm 1 3 . 9 }$ </td><td> $3 7 . 5 \pm 1 5 . 8$ </td></tr><tr><td></td><td>Unseen</td><td> $3 0 . 5 \pm 1 4 . 6$ </td><td> ${ \bf 3 6 . 0 \pm 1 3 . 6 }$ </td></tr><tr><td></td><td>5 Seen</td><td> ${ \bf 9 4 . 0 \pm 4 . 4 }$ </td><td> $7 6 . 5 \pm 2 2 . 6$ </td></tr><tr><td></td><td>Unseen</td><td> $7 2 . 5 \pm 2 6 . 8$ </td><td> ${ \bf 8 9 . 5 \pm 7 . 2 }$ </td></tr><tr><td>Polish</td><td>2 Seen</td><td> ${ \bf 9 1 . 5 \pm 4 . 5 }$ </td><td> $3 3 . 5 \pm 7 . 2$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 8 8 . 0 \pm 4 . 5 }$ </td><td> $3 9 . 5 \pm 8 . 3$ </td></tr><tr><td></td><td>5 Seen</td><td> ${ \bf 9 7 . 0 \pm 2 . 5 }$ </td><td> $7 3 . 5 \pm 8 . 4$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 9 6 . 0 \pm 2 . 8 }$ </td><td> $7 4 . 0 \pm 7 . 7$ </td></tr><tr><td>German</td><td>2 Seen</td><td> ${ \bf 5 7 . 0 \pm 6 . 8 }$ </td><td> $5 . 0 \pm 2 . 4$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 3 5 . 5 \pm 1 0 . 6 }$ </td><td> $1 0 . 0 \pm 3 . 4$ </td></tr><tr><td></td><td>5 Seen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $2 5 . 5 \pm 6 . 4$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 9 9 . 0 \pm 1 . 5 }$ </td><td> $2 7 . 0 \pm 6 . 4$ </td></tr><tr><td>GMC</td><td>50 Seen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>Unseen</td><td> $0 . 0 \pm 0 . 0$ </td><td> ${ \bf 9 9 . 0 \pm 1 . 5 }$ </td></tr><tr><td></td><td>100 Seen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>Unseen</td><td> $0 . 0 \pm 0 . 0$ </td><td> ${ \bf 9 9 . 0 \pm 1 . 5 }$ </td></tr><tr><td>ACSIncome</td><td>3 Seen</td><td> ${ \bf 7 4 . 0 \pm 8 . 2 }$ </td><td> $6 0 . 0 \pm 8 . 1$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 6 5 . 0 \pm 7 . 9 }$ </td><td> $5 8 . 5 \pm 7 . 9$ </td></tr><tr><td></td><td>10 Seen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr></table>

(a) Linear: ORPM vs. ElliCE.

<table><tr><td>Dataset</td><td>δSet</td><td>ORPM MaxMargin</td><td>MinL2</td></tr><tr><td>Synthetic</td><td>1 Seen</td><td> $\mathbf { 5 4 . 0 \pm 1 9 . 1 5 4 . 0 \pm 1 9 . 1 }$ </td><td> ${ \bf 5 4 . 0 \pm 1 9 . 1 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 5 2 . 0 \pm 1 8 . 1 ~ 5 2 . 0 \pm 1 8 . 1 }$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td></td><td>3 Seen</td><td> $\mathbf { 1 0 0 . 0 \ : \pm 0 . 0 \ : 1 0 0 . 0 \ : \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>Unseen</td><td> $\mathbf { 1 0 0 . 0 \ : \pm 0 . 0 \ : 1 0 0 . 0 \ : \pm 0 . 0 }$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>COMPAS</td><td>1 Seen</td><td> $\mathbf { 4 2 . 0 \pm 1 2 . 5 4 2 . 0 \pm 1 2 . 5 }$ </td><td> ${ \bf 4 2 . 0 \pm 1 2 . 5 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 3 6 . 0 \pm 1 3 . 1 }$   ${ \bf 3 6 . 0 \pm 1 3 . 1 }$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td></td><td>3 Seen</td><td> ${ \bf 9 2 . 0 \pm 7 . 4 }$   ${ \bf 9 2 . 0 \pm 7 . 4 }$ </td><td> ${ \bf 9 2 . 0 \pm 7 . 4 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 9 0 . 0 \pm 7 . 5 }$   ${ \bf 9 0 . 0 \pm 7 . 5 }$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Polish</td><td>1 Seen</td><td> ${ \bf 7 8 . 0 \pm 1 0 . 6 }$   $6 0 . 0 \pm 6 . 7$ </td><td> ${ \bf 7 8 . 0 \pm 1 0 . 6 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 3 4 . 0 \pm 1 1 . 8 }$   $2 6 . 0 \pm 1 1 . 8$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td></td><td>3 Seen</td><td> ${ \bf 9 4 . 0 \pm 6 . 9 }$   $8 8 . 0 \pm 1 0 . 0 $ </td><td> ${ \bf 9 4 . 0 \pm 6 . 9 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 8 8 . 0 \pm 7 . 4 }$   $7 8 . 0 \pm 1 2 . 5$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>German</td><td>1 Seen</td><td> ${ \bf 2 0 . 0 \pm 9 . 5 }$   $1 2 . 0 \pm 7 . 4$ </td><td> ${ \bf 2 0 . 0 \pm 9 . 5 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 0 . 0 \pm 0 . 0 }$   ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>3 Seen</td><td> ${ \bf 9 4 . 0 \pm 6 . 9 }$   $7 6 . 0 \pm 1 4 . 8$ </td><td> ${ \bf 9 4 . 0 \pm 6 . 9 }$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 2 0 . 0 \pm 1 1 . 7 }$   $1 4 . 0 \pm 1 1 . 8$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>GMC</td><td>25 Seen</td><td> ${ \bf 9 0 . 0 \pm 1 0 . 1 }$   ${ \bf 9 0 . 0 \pm 1 0 . 1 }$ </td><td> $6 0 . 0 \pm 1 7 . 8$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 9 0 . 0 \pm 1 0 . 1 }$   $8 2 . 0 \pm 1 8 . 4$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td></td><td>75 Seen</td><td> $9 2 . 0 \pm 1 0 . 0$   ${ \bf 9 6 . 0 \pm 6 . 0 }$ </td><td> $6 6 . 0 \pm 1 9 . 1$ </td></tr><tr><td></td><td>Unseen</td><td> ${ \bf 9 2 . 0 \pm 1 0 . 0 }$   $8 8 . 0 \pm 1 5 . 4$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>ACSIncome</td><td>2 Seen</td><td> ${ \bf 5 8 . 0 \pm 1 9 . 6 }$   ${ \bf 5 8 . 0 \pm 1 9 . 6 }$ </td><td> ${ \bf 5 8 . 0 \pm 1 9 . 6 }$ </td></tr><tr><td></td><td></td><td>Unseen 52.0 ± 19.3 52.0 ± 19.3</td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td></td><td>7.5 Seen</td><td> $\mathbf { 1 0 0 . 0 \ : \pm 0 . 0 \ : 1 0 0 . 0 \ : \pm 0 . 0 }$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td></tr><tr><td></td><td>Unseen</td><td> $\mathbf { 1 0 0 . 0 \ : \pm 0 . 0 \ : 1 0 0 . 0 \ : \pm 0 . 0 }$ </td><td> $2 . 0 \pm 4 . 5$ </td></tr></table>

(b) Neural: ORPM vs MIQCP approaches.  
Table 1: Joint validity (%) on seen and unseen models. The left panel compares ORPM and ElliCE on 24 seen linear models and 50 unseen models. The right panel compares ORPM with the MaxMargin and MinL2 MIQCP references on 12 seen networks with the same architecture and 50 unseen models. Entries are means ± pointwise 95% Student-t intervals.

For linear competing sets, we created a competing set of 24 models and compared validity with ElliCE [60], which is designed to provide robust recourse over an ellipsoidal approximation of the Rashomon set. Table 1a summarizes the results (the rows labeled as seen in Table 1a), where for each dataset and budget value, we report the joint validity of ORPM and ElliCE. We observe that ORPM achieves higher or equal validity on all seen models. However, evaluating ElliCE strictly on the finite subset may penalize its intended design, which aims to protect against any model in the ellipsoidal approximation of the Rashomon set. To highlight the tradeof between true generalization and finite-set optimization, we drew 50 additional unseen models from the filtered ellipsoidal approximation of the Rashomon set and measured the validity of recourse provided by both ORPM and ElliCE on these unseen models. Results are summarized in Table 1a (the rows labeled as unseen in Table 1a). We see that ORPM’s validity generally drops on unseen models, while ElliCE is more robust. Still, ORPM achieves higher or equal unseen joint validity on four datasets, while ElliCE performs better on COMPAS and GMC. Thus, ORPM shows broader applicability than its intended design.

For the same-architecture neural experiment, we used 12 networks with one hidden layer of 20 ReLU units. We used 10 seeds and 5 instances per seed. We compared ORPM with two joint neural-optimization baselines. MinL2 adapts the minimum-distance formulation of [41], using $L _ { 2 }$ rather than $L _ { 1 }$ distance and requiring acceptance by every competing network. MaxMargin maximizes the minimum network logit within budget δ. Both jointly encode the ReLU networks, and the quadratic distance terms yield MIQCP formulations. MaxMargin receives one 60-second solve per positive budget; MinL2 receives one 60-second solve at the largest budget, and its result is reused for smaller budgets. Appendix B.2 specifies the formulations, numerical tolerances, and return rules. Table 1b reports joint validity for ORPM and these two references. On seen models (the rows labeled as seen in Table 1b), ORPM matches MaxMargin on Synthetic, COMPAS and ACSIncome at both displayed budgets, and has higher means on Polish and German. GMC at budget 75 is an exception: MaxMargin achieves 96% joint validity versus ORPM’s 92%. Similar to the linear case, to examine robustness, we drew 50 additional unseen models by using diferent initializations and feature masks. ORPM matches or exceeds MaxMargin’s mean unseen joint validity at every displayed setting (the rows labeled as unseen in Table 1b). MinL2 often matches ORPM on seen models but achieves only 0–2% unseen validity in these settings. Thus, joint feasibility on the competing set does not ensure transfer to unseen models.

The Cost of Multiplicity. We next study how predictive multiplicity afects the dificulty of providing valid recourse by measuring budget-feasibility curves and the loss of recourse validity when separate model-specific actions are replaced by one common action. In particular, for each dataset, budget, and competing model set, we compared ORPM’s joint success rate with three feasibility benchmarks: (1) average singleton, the fraction of models for which a budget-feasible successful action exists; (2) all separately, the fraction of instances for which every model admits a budget-feasible successful action, allowing diferent actions for diferent models; and (3) one common action, the fraction of instances for which a single budget-feasible action is accepted by every model simultaneously. These benchmarks impose increasingly stringent requirements, so their success rates weakly decrease in that order. We focused on homogeneous linear and neural competing sets, where we can better analyze the true success of one common action using convex quadratic optimization and MIQCP solvers. When numerical optimization could not resolve benchmark feasibility, we retained lower and upper bounds.

For a fixed combination of dataset and competing set, we computed the success rate of ORPM and the baselines as a function of allowable budget. The results for the German dataset on competing sets of 24 models are summarized in Figure 4. For linear models (left panel) at budget 2, average singleton success is 92.5%, whereas all-separate and common-action success are 64.4% and 57.2%, respectively. The first gap reflects the dificulty of accommodating each model individually; the additional 7.2-percentage-point drop reflects the added requirement that one action satisfy them simultaneously. ORPM achieves 57.1%, closely matching common-action feasibility and indicating little observed algorithmic shortfall at this budget. For neural models (right panel) at the same budget of 2, singleton success is bounded by [93.6, 97.7]%, while common-action success is bounded by [46.0, 67.0]% and ORPM achieves a success rate of 46.0%, so the algorithmic shortfall is bounded by 0–21 percentage points. The unresolved range does not establish that ORPM attains the common-action optimum. Increasing the budget to 5 raises ORPM success to 99.7% for linear models and 100% for neural models, showing that additional budget largely overcomes the observed burden in the German dataset. The appendix shows similar separations between singleton and joint feasibility across the remaining datasets, although their magnitudes vary. See Figures 12 and 13 in Appendix B.3.3 for more details.

![](images/4a12af2736aa1708db1dbf8dacb2b57eaf92e3eee230e224a2275951dd6c1de8.jpg)  
(a) German dataset, linear competing set of 24 models.

![](images/254204f39174b06fc6535906f9879ae038ca38608632adf9dc61be54c8ffa1fe.jpg)  
(b) German dataset, neural competing set of 24 models.  
Figure 4: The validity degradation to guard against multiplicity. Each subfigure corresponds to a pair of datasets and competing sets as indicated in the caption. Each curve tracks the success rate (Y-axis) for ORPM and baselines as a function of allowable budget (X-axis).

The Disparity of Multiplicity. We next examine demographic disparities in recourse success under predictive multiplicity. We focus on four datasets: COMPAS and ACSIncome, using the recorded female and male groups, and GMC and German, using individuals younger than 40 and those aged 40 or older. Synthetic and Polish are excluded because we do not designate a sensitive attribute for these datasets in our analysis. We considered competing sets of 24 linear models and compared ORPM with the three cost-of-multiplicity benchmarks: average singleton, all separately, and one common action. For each method or benchmark, we computed the signed diference in success rates between the two groups within each seed, average these diferences across ten seeds, and take the absolute value. This measures the magnitude of the average group success disparity, in percentage points.

![](images/14fa9d9d81797d94fd0c45fda49f8901682f9adcbd176ad2ca389c7b818fa544.jpg)  
(a) Linear competing set, small budget regime.

![](images/3f95ad2bbe71c9f1c3c46d31e6270f3d44579f2b4d8be05a920277e4c3ebdd5a.jpg)  
(b) Linear competing set, medium budget regime.  
Figure 5: The disparity of recourse success across diferent groups. Each subfigure corresponds to diferent budget values, with small budgets on the left and medium budgets on the right. In each subfigure, the bars track the disparity in recourse success (Y-axis) for ORPM and baselines on each dataset.

Figure 5 shows that ORPM closely tracks the common-action benchmark: their estimated disparity gap magnitudes difer by at most 0.5 percentage points across the displayed linear settings. ORPM’s disparity gaps are generally larger than those of average singleton, demonstrating that multiplicity can further amplify the disparities. On the other hand, all-separately can show larger or smaller gaps than ORPM. These comparisons concern benchmarks with diferent success requirements and do not imply an ordering of disparity by stringency. Increasing the budget does not uniformly reduce disparity: ORPM’s disparity gap on GMC decreases from 36.7 to 19.6 percentage points as the budget increases from 5 to 10, whereas its COMPAS gap increases from 7.0 to 16.5 points as the budget increases from 1 to 2. Thus, additional budget can narrow or widen observed group diferences. In these experiments, confidence intervals are often wide, limiting conclusions about method rankings. See Figure 14 in Appendix B.3.4 for neural competing sets.

![](images/48217584ac4d91a4974e397bca663f0c9e34c5dce2efb69525b5464c9c530155.jpg)  
(a) Convergence for Polish on tree-based competing sets.

![](images/1cf74172edc63e3220ab8f76066e460db3398b98c0aa2e85c0abcc6239377b0c.jpg)  
(b) Joint validity for COMPAS on tree-based competing sets.  
Figure 6: Experiments on tree-based competing sets: The left panel depicts the average upper bound on the mixture gap (Y-axis) as the number of rounds of ORPM (X-axis). The right panel tracks the joint validity (Y-axis) of ORPM and tree-based baselines as a function of recourse budget (X-axis).

Tree-Based Competing Sets. So far we have focused on convex and nonconvex but diferentiable competing sets. As our last experiments, we revisit our prior empirical evaluations by focusing on non-convex tree-based competing sets. For all experiments in this section, the competing sets contain trees of depth four.

We start with convergence experiments: we ran ORPM on diferent datasets and competing sets of diferent sizes for 200 rounds and measured the validity and the mixture gap of the returned recourse. Consistent with diferentiable experiments, we see that the mixture gap decreases when ORPM is run for more rounds. An example of such behavior is shown in Figure 6a, where we track mean upper bounds on the mixture gap for competing sets of diferent sizes on the Polish dataset using a budget of 2. We observe that the upper bound on the mixture gap is lower for competing sets with a smaller number of trees. Similar to diferentiable competing sets, joint validity can be achieved before the mixture gap reaches our target of 0.005. For example, for the competing set with 12 trees, we observe 88.5% joint validity after 100 rounds, despite no instances attaining the certified mixture-gap target. Increasing the number of rounds from 100 to 200 will increase the joint validity despite the smaller progress rate in decreasing the mixture gap.

We also examine how the available recourse budget afects joint validity across five competing decision trees on COMPAS, using 10 seeds and 20 instances per seed. Figure 6b summarizes the results, tracking the joint validity of ORPM and baselines (Y-axis) across diferent budget values (X-axis). ORPM achieves the highest observed joint-validity rate at every tested budget among the tree-based baselines, increasing from 25.0% at $\delta = 0 . 5$ to 78.5% at $\delta = 1 0$ . At δ = 2, ORPM achieves 62.5% joint validity, compared with 46.0% for RobX-constrained, 43.0% for OCEAN-style, 14.5% for ADV-surrogate, and 10.0% for ROAR-surrogate. ORPM’s advantage over the strongest baseline, RobX-constrained, is 16.5 percentage points. Increasing the budget also improves RobX-constrained and OCEAN-style, but their validity reaches only 53.5% and 49.5%, respectively, at $\delta = 1 0$ . ADV-surrogate remains at 14.5% across the tested budgets, while ROAR-surrogate reaches at most 13.0%. Compared to the baselines, ORPM benefits more from additional budget, although its improvement diminishes at larger budgets: increasing δ from 5 to 10 raises validity only from 76.5% to 78.5%.

![](images/7c48367f8a14df48c758412f2ebb7802c9a61dbda3732ed5d4ae5e4acd0c24d1.jpg)

![](images/cd0cb046b2b8c6ba82cc9aafe1591edfc19562d481e91ed8a2df19ad7f364d69.jpg)  
(a) Cost of multiplicity for the German dataset on (b) Disparity of multiplicity for the GMC dataset on tree-based tree-based competing sets. competing sets.  
Figure 7: Additional experiments on tree-based competing sets: The left panel depicts the joint validity (Y-axis) for each budget (X-axis) for ORPM and baselines that are afected by multiplicity to diferent degrees. In the right panel, the bars track the disparity in recourse success (Y-axis) for ORPM and baselines on the GMC dataset.

We next examine how multiplicity afects the validity. In Figure $^ { 7 \mathrm { a } , }$ we track the joint validity of ORPM and baselines (Y-axis) against the recourse budget (X-axis) for the German dataset with 12 competing decision trees. These baselines include average singleton, all separately, and all common action, with the same description provided in the diferentiable experiments. At $\delta = 2$ , average singleton success is 99.5% and all-separate feasibility is 94.5%, but common-action feasibility falls to $5 3 . 4 \%$ , a reduction of 41.0 percentage points. ORPM attains the same 53.4% joint-validity rate, indicating that the main dificulty at this budget is satisfying the competing trees with one action. Increasing the budget to $\delta = 3$ raises both common-action feasibility and ORPM validity to 90.9%. At $\delta = 5 ,$ a common action is feasible for every evaluated case, while ORPM achieves 98.4% validity. Thus, additional budget substantially reduces the validity loss from requiring a shared action, and ORPM closely tracks common-action feasibility in this setting.

Finally, we examine whether the requirement to provide recourse across competing trees afects demographic groups diferently. In Figure 7b, we depict the disparity in recourse success for ORPM and baselines on the GMC dataset for diferent sizes of competing sets. GMC gives a particularly clear example at budget δ = 1: the absolute success gap between the two subgroups (younger than 40 or aged 40 and older) is 6.7 percentage points for average-singleton success, 26.9 points for all-separately success, and 29.5 points for both common-action feasibility and ORPM success; the younger group has lower success in every case. Relative to average-singleton success, the common-action requirement therefore adds 22.7 points to the success gap. ORPM matches the common-action disparity in all three displayed settings: attaining the benchmark’s feasibility does not remove the unequal burden associated with multiplicity. The ORPM/common-action gap is 16.4, 29.5, and 33.5 points for 6, 12, and 24 trees, respectively, illustrating a larger observed gap for larger sets. Additional results for tree-based competing sets are provided in Appendix B.3.5.

## 6 Discussion and Future Work

We formulate recourse under predictive multiplicity as budget-constrained minimization of the worst-case validity loss over a specified competing model set. Requiring one action to satisfy every model restricts the set of successful actions relative to single-model recourse and can increase the minimum budget required for success, although a strict cost increase is not inevitable. This is consistent with prior work examining the relationship between robustness requirements and recourse cost [39, 61]. Our formulation makes this tension explicit by optimizing worst-case loss within an allowable budget, rather than directly minimizing cost. Nevertheless, successful recourse may remain unavailable at the chosen budget, and good performance on the competing set does not always establish validity for unseen models.

Our framework accommodates competing models with diferent architectures or model classes without requiring proximity in parameter space. However, models with similar aggregate predictive performance can impose conflicting requirements on individual recourse. Defining a meaningful competing set therefore requires explicit choices about predictive quality, admissible models, and deployment conditions. Our disparity results additionally show that robustness across models does not ensure equal success across groups, and increasing the budget need not reduce disparities.

Future work could extend the framework to continuous or infinite Rashomon sets [11], combining model exploration methods [26, 27] with procedures that search for models violating a proposed recourse. Establishing guarantees for such extensions would require controlling both recourse optimization error and approximation of the competing set. Another direction is to jointly learn competing models and recourse policies, balancing predictive accuracy, model diversity, and recourse feasibility. Incorporating equity objectives and action constraints is also left as future work. Finally, explicitly modeling how recourse changes individuals’ behavior and subsequent training data can address robustness under endogenous model updates.

Ethical Considerations. Recourse helps individuals with actionable recommendations on how to improve outcomes. Predictive multiplicity can create opportunities to identify less discriminatory models [12, 24, 40], though this is not always the case [29]. Understanding how to provide recourse that is robust to predictive multiplicity is an important reliability concern, though by itself it does not ensure equitable outcomes, as shown in our experiments. This is consistent with prior work on algorithmic recourse, which showed independently generating recourse for individuals might lead to disproportionate validity or burden across diferent subpopulations [62]. This is also consistent with prior work on predictive multiplicity, which showed multiplicity exposes some groups to greater prediction instability [12].

Robustness should therefore not be interpreted as an endorsement of the competing models or their decision criteria. Institutions remain responsible for assessing discriminatory model behavior and whether recommended changes are feasible, afordable, and appropriate for afected individuals. Our computational validity and cost measures do not establish these properties in practice, and observed group diferences are descriptive rather than causal findings. Future work should extend the analysis to larger and intersectional groups, incorporate individual circumstances and causal constraints, and develop methods that jointly address robustness, validity, and equitable recourse burden and success.

Acknowledgment. We thank Kshitij Kayastha for discussions during the early stages of this work.

LLM Usage. Gemini 3.1 Pro was used to write the initial version of the experiments from the problem formulation and algorithm. We used GPT-6 Astra to verify the implementation. We additionally checked all the code. GPT-6 Astra was used to proofread the paper, especially in the description of the implementation details and experimental results. LLMs were not used for problem formulation, algorithmic solutions, design of experiments, and description of related work.

## References

[1] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna M. Wallach. A reductions approach to fair classification. In 35th International Conference on Machine Learning, volume 80, pages 60–69, 2018.

[2] Gianvincenzo Alfano, Adam Gould, Francesco Leofante, Antonio Rago, and Francesca Toni. Counterfactual explanations under model multiplicity and their use in computational argumentation. In 34th International Joint Conference on Artificial Intelligence, pages 4321–4329, 2025.

[3] Sanjeev Arora, Elad Hazan, and Satyen Kale. The multiplicative weights update method: a metaalgorithm and applications. Theory of Computing, 8(6):121–164, 2012.

[4] Srikanth Avasarala, Varun Gupta, Shahin Jabbari, Saber Salehkaleybar, and Juba Ziani. The role of causality in algorithmic recourse. CoRR, abs/2607.28497, 2026.

[5] Tina Behzad, Sílvia Casacuberta, Emily Diana, and Alexander Tolbert. Reconciling predictive multiplicity in practice. In 8th ACM Conference on Fairness, Accountability, and Transparency, pages 3350–3369, 2025.

[6] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

[7] Leo Breiman. Statistical modeling: the two cultures. Statistical Science, 16(3):199–231, 2001.

[8] Ngoc Bui, Duy Nguyen, Man-Chung Yue, and Viet Anh Nguyen. Coverage-validity-aware algorithmic recourse. Operation Research, 73(6):3294–3310, 2025.

[9] Robert Chen, Brendan Lucier, Yaron Singer, and Vasilis Syrgkanis. Robust optimization for non-convex objectives. In Advances in Neural Information Processing Systems 30, pages 4705–4714, 2017.

[10] Seung Hyun Cheon, Anneke Wernerfelt, Sorelle Friedler, and Berk Ustun. Feature responsiveness scores: Model-agnostic explanations for recourse. In 13th International Conference on Learning Representations, 2025.

[11] Evzenie Coupkova and Mireille Boutin. On the Rashomon ratio of infinite hypothesis sets. CoRR, abs/2404.17746, 2024.

[12] Gordon Dai, Pavan Ravishankar, Rachel Yuan, Emily Black, and Daniel B. Neill. Be intentional about fairness!: Fairness, size, and multiplicity in the Rashomon set. In 5th ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, pages 42–73, 2025.

[13] Alexander D’Amour, Katherine A. Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D. Hofman, Farhad Hormozdiari, Neil Houlsby, Shaobo Hou, Ghassen Jerfel, Alan Karthikesalingam, Mario Lucic, Yi-An Ma, Cory Y. McLean, Diana Mincu, Akinori Mitani, Andrea Montanari, Zachary Nado, Vivek Natarajan, Christopher Nielson, Thomas F. Osborne, Rajiv Raman, Kim Ramasamy, Rory Sayres, Jessica Schrouf, Martin Seneviratne, Shannon Sequeira, Harini Suresh, Victor Veitch, Max Vladymyrov, Xuezhi Wang, Kellie Webster, Steve Yadlowsky, Taedong Yun, Xiaohua Zhai, and D. Sculley. Underspecification presents challenges for credibility in modern machine learning. Journal of Machine Learning Research, 23:226:1–226:61, 2022.

[14] Constantinos Daskalakis, Andrew Ilyas, Vasilis Syrgkanis, and Haoyang Zeng. Training gans with optimism. In 6th International Conference on Learning Representations, 2018.

[15] Emily Diana, Wesley Gill, Ira Globus-Harris, Michael Kearns, Aaron Roth, and Saeed Sharifi-Malvajerdi. Lexicographically fair learning: Algorithms and generalization. In 2nd Symposium on Foundations of Responsible Computing, pages 6:1–6:23, 2021.

[16] Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. In Advances in Neural Information Processing Systems 34, pages 6478–6490, 2021.

[17] Ally Yalei Du, Dung Daniel Ngo, and Zhiwei Steven Wu. Reconciling model multiplicity for downstream decision making. In 13th International Conference on Learning Representations, 2025.

[18] Sanghamitra Dutta, Jason Long, Saumitra Mishra, Cecilia Tilli, and Daniele Magazzeni. Robust counterfactual explanations for tree-based ensembles. In 39th International Conference on Machine Learning, volume 162, pages 5742–5756, 2022.

[19] Aaron Fisher, Cynthia Rudin, and Francesca Dominici. All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20:177:1–177:81, 2019.

[20] Alexandre Forel, Axel Parmentier, and Thibaut Vidal. Don’t explain noise: Robust counterfactuals for randomized ensembles. In Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pages 293–309, 2024.

[21] Yoav Freund and Robert Schapire. Game theory, on-line prediction and boosting. In 9th Annual Conference on Computational Learning Theory, pages 325–332, 1996.

[22] Hangzhi Guo, Feiran Jia, Jinghui Chen, Anna Squicciarini, and Amulya Yadav. RoCourseNet: Robust training of a prediction aware recourse model. In 32nd ACM International Conference on Information and Knowledge Management, pages 619–628, 2023.

[23] Faisal Hamman, Erfaun Noorani, Saumitra Mishra, Daniele Magazzeni, and Sanghamitra Dutta. Robust counterfactual explanations for neural networks with probabilistic guarantees. In 40th International Conference on Machine Learning, pages 12351–12367, 2023.

[24] Chris Hays, Benjamin Laufer, Solon Barocas, and Manish Raghavan. Statistical guarantees in the search for less discriminatory algorithms. In 14th International Conference on Learning Representations, 2026.

[25] Hsiang Hsu and Flavio du Pin Calmon. Rashomon capacity: A metric for predictive multiplicity in classification. In Advances in Neural Information Processing Systems 35, 2022.

[26] Hsiang Hsu, Ivan Brugere, Shubham Sharma, Freddy Lécué, and Richard Chen. RashomonGB: Analyzing the Rashomon efect and mitigating predictive multiplicity in gradient boosting. In Advances in Neural Information Processing Systems 37, 2024.

[27] Hsiang Hsu, Guihong Li, Shaohan Hu, and Chun-Fu Chen. Dropout-based Rashomon set exploration for eficient predictive multiplicity estimation. In 12th International Conference on Learning Representations, 2024.

[28] Shahin Jabbari. Algorithmic recourse under competition. CoRR, abs/2609.39877, 2026.

[29] Shomik Jain, Margaret Wang, Kathleen Creel, and Ashia Wilson. Allocation multiplicity: Evaluating the promises of the rashomon set. In 8th ACM Conference on Fairness, Accountability, and Transparency, pages 2040–2055. ACM, 2025.

[30] Junqi Jiang, Jianglin Lan, Francesco Leofante, Antonio Rago, and Francesca Toni. Provably robust and plausible counterfactual explanations for neural networks via robust optimisation. In 15th Asian Conference on Machine Learning, pages 582–597, 2023.

[31] Junqi Jiang, Francesco Leofante, Antonio Rago, and Francesca Toni. Robust counterfactual explanations in machine learning: A survey. In 33rd International Joint Conference on Artificial Intelligence, pages 8086–8094, 2024.

[32] Amir-Hossein Karimi, Gilles Barthe, Borja Balle, and Isabel Valera. Model-agnostic counterfactual explanations for consequential decisions. In 23rd International Conference on Artificial Intelligence and Statistics, pages 895–905, 2020.

[33] Amir-Hossein Karimi, Gilles Barthe, Bernhard Schölkopf, and Isabel Valera. A survey of algorithmic recourse: Contrastive explanations and consequential recommendations. ACM Computing Surveys, 55 (5):95:1–95:29, 2023.

[34] Kshitij Kayastha, Vasilis Gkatzelis, and Shahin Jabbari. Learning-augmented robust algorithmic recourse. Transactions on Machine Learning Research, 2026.

[35] Michael Kearns, Seth Neel, Aaron Roth, and Zhiwei Wu. Preventing fairness gerrymandering: Auditing and learning for subgroup fairness. In 35th International Conference on Machine Learning, pages 2569–2577, 2018.

[36] Eoin Kenny, Allan Anzagira, Tom Bewley, Freddy Lecue, and Manuela Veloso. An evaluation of cost functions for algorithmic recourse. In 29th Annual Conference on Artificial Intelligence and Statistics, 2026.

[37] Gunnar König, Timo Freiesleben, and Moritz Grosse-Wentrup. Improvement-focused causal recourse (ICR). In 37th AAAI Conference on Artificial Intelligence, pages 11847–11855, 2023.

[38] Gunnar König, Hidde Fokkema, Timo Freiesleben, Celestine Mendler-Dünner, and Ulrike von Luxburg. Performative validity of recourse explanations. In Advances in Neural Information Processing Systems 38, 2025.

[39] Phone Kyaw, Kshitij Kayastha, and Shahin Jabbari. Optimal robust recourse with L<sup>p</sup>-bounded model change. In 4th IEEE Conference on Secure and Trustworthy Machine Learning, 2026.

[40] Benjamin Laufer, Manish Raghavan, and Solon Barocas. What constitutes a less discriminatory algorithm? In 4th Symposium on Computer Science and Law, pages 136–151, 2025.

[41] Francesco Leofante, Elena Botoeva, and Vineet Rajani. Counterfactual explanations and model multiplicity: a relational verification view. In 20th International Conference on Principles of Knowledge Representation and Reasoning, pages 763–768, 2023.

[42] Ana Lucic, Harrie Oosterhuis, Hinda Haned, and Maarten de Rijke. FOCUS: flexible optimizable counterfactual explanations for tree ensembles. In 36th AAAI Conference on Artificial Intelligence, pages 5313–5322, 2022.

[43] Charles Marx, Flávio P. Calmon, and Berk Ustun. Predictive multiplicity in classification. In 37th International Conference on Machine Learning, volume 119, pages 6765–6774, 2020.

[44] Duy Nguyen, Ngoc Bui, and Viet Anh Nguyen. Distributionally robust recourse action. In 11th International Conference on Learning Representations, 2023.

[45] Tuan-Duy Nguyen, Ngoc Bui, Duy Nguyen, Man-Chung Sue, and Viet Anh Nguyen. Robust Bayesian recourse. In 38th Conference on Uncertainty in Artificial Intelligence, pages 1498–1508, 2022.

[46] Erfaun Noorani, Pasan Dissanayake, Faisal Hamman, and Sanghamitra Dutta. Counterfactual explanations for model ensembles using entropic risk measures. In 24th International Conference on Autonomous Agents and Multiagent Systems, pages 1566–1575, 2025.

[47] Axel Parmentier and Thibaut Vidal. Optimal counterfactual explanations in tree ensembles. In 38th International Conference on Machine Learning, pages 8422–8431, 2021.

[48] Martin Pawelczyk, Klaus Broelemann, and Gjergji Kasneci. On counterfactual explanations under predictive multiplicity. In 36th Conference on Uncertainty in Artificial Intelligence, pages 809–818, 2020.

[49] Martin Pawelczyk, Teresa Datta, Johannes van den Heuvel, Gjergji Kasneci, and Himabindu Lakkaraju. Probabilistically robust recourse: Navigating the trade-ofs between costs and robustness in algorithmic recourse. In 11th International Conference on Learning Representations, 2023.

[50] Martin Pawelczyk, Tobias Leemann, Asia Biega, and Gjergji Kasneci. On the trade-of between actionable explanations and the right to be forgotten. In 11th International Conference on Learning Representations, 2023.

[51] Rafael Poyiadzi, Kacper Sokol, Raúl Santos-Rodríguez, Tijl De Bie, and Peter Flach. FACE: feasible and actionable counterfactual explanations. In 3rd AAAI/ACM Conference on AI, Ethics, and Society, pages 344–350, 2020.

[52] Alexander Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Advances in Neural Information Processing Systems 26, pages 3066–3074, 2013.

[53] Marco Túlio Ribeiro, Sameer Singh, and Carlos Guestrin. “Why should I trust you?": Explaining the predictions of any classifier. In 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1135–1144, 2016.

[54] Chris Russell. Eficient search for diverse coherent explanations. In 2nd ACM Conference on Fairness, Accountability, and Transparency, pages 20–28. ACM, 2019.

[55] Meirav Segal, Anne-Marie George, and Christos Dimitrakakis. Threshold recourse for dynamic allocations. In 3rd European Workshop on Algorithmic Fairness, 2024.

[56] Lesia Semenova and Cynthia Rudin. A study in Rashomon curves and volumes: A new perspective on generalization and model simplicity in machine learning. CoRR, abs/1908.01755, 2019.

[57] Lesia Semenova, Cynthia Rudin, and Ronald Parr. On the existence of simpler machine learning models. In 5th ACM Conference on Fairness Accountability and Transparency, pages 1827–1858, 2022.

[58] Saeed Sharifi-Malvajerdi, Michael Kearns, and Aaron Roth. Average individual fairness: Algorithms, generalization and experiments. In Advances in Neural Information Processing Systems 32, pages 8240–8249, 2019.

[59] Maurice Sion. On general minimax theorems. Pacific Journal of Mathematics, 8(1):171–176, 1958.

[60] Bohdan Turbal, Iryna Voitsitska, and Lesia Semenova. ElliCE: Eficient and provably robust algorithmic recourse via the Rashomon sets. In Advances in Neural Information Processing Systems 38, 2025.

[61] Sohini Upadhyay, Shalmali Joshi, and Himabindu Lakkaraju. Towards robust and reliable algorithmic recourse. In Advances in Neural Information Processing Systems 34, pages 16926–16937, 2021.

[62] Berk Ustun, Alexander Spangher, and Yang Liu. Actionable recourse in linear classification. In 3rd ACM Conference on Fairness, Accountability, and Transparency, pages 10–19, 2019.

[63] Sahil Verma, John Dickerson, and Keegan Hines. Counterfactual explanations for machine learning: A review. CoRR, abs/2010.10596, 2020.

[64] Sandra Wachter, Brent Mittelstadt, and Chris Russell. Counterfactual explanations without opening the black box: automated decisions and the GDPR. Harvard Journal of Law and Technology, 31(2):841–887, 2018.

[65] Jamelle Watson-Daniels, David Parkes, and Berk Ustun. Predictive multiplicity in probabilistic classification. In 37th AAAI Conference on Artificial Intelligence, pages 10306–10314, 2023.

[66] Rui Xin, Chudi Zhong, Zhi Chen, Takuya Takagi, Margo Seltzer, and Cynthia Rudin. Exploring the whole Rashomon set of sparse decision trees. In Advances in Neural Information Processing Systems 35, 2022.

[67] Jayanth Yetukuri, Ian Hardy, Yevgeniy Vorobeychik, Berk Ustun, and Yang Liu. Providing fair recourse over plausible groups. In 38th AAAI Conference on Artificial Intelligence, pages 21753–21760, 2024.

## A Omitted Details from Section 4

Proof of Theorem 4.3. By the definition of the mixture gap, the definition of w¯, and linearity of $\mathcal { L } ( x , \cdot )$

$$
\begin{array} { l } { \displaystyle \mathrm { G a p } _ { \mathrm { m i x } } ( x _ { 1 : T } , \bar { w } ) = \displaystyle \operatorname* { m a x } _ { w \in \Delta ( H ) } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } ( x _ { t } , w ) - \displaystyle \operatorname* { i n f } _ { x \in K } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } ( x , w _ { t } ) } \\ { = R _ { \mathrm { a d v } } + R _ { \mathrm { r e c } } , } \end{array}
$$

where

$$
\begin{array} { r l } & { R _ { \mathrm { a d v } } : = \underset { w \in \Delta ( H ) } { \operatorname* { m a x } } \frac { 1 } { T } { \displaystyle \sum _ { t = 1 } ^ { T } } \mathcal { L } ( x _ { t } , w ) - \frac { 1 } { T } { \displaystyle \sum _ { t = 1 } ^ { T } } \mathcal { L } ( x _ { t } , w _ { t } ) , } \\ & { R _ { \mathrm { r e c } } : = \frac { 1 } { T } { \displaystyle \sum _ { t = 1 } ^ { T } } \mathcal { L } ( x _ { t } , w _ { t } ) - \underset { x \in K } { \operatorname* { i n f } } \frac { 1 } { T } { \displaystyle \sum _ { t = 1 } ^ { T } } \mathcal { L } ( x , w _ { t } ) . } \end{array}
$$

We bound the two terms separately.

First, suppose $n \geq 2$ . We bound $R _ { \mathrm { a d v } }$ using the standard exponential-weights analysis [3]. Recall that $v _ { t , i } = \ell ( h _ { i } ( x _ { t } ) , 1 ) \in [ 0 , B ]$ by Assumption 4.2. Write the updates in unnormalized form:

$$
q _ { 1 , i } = 1 , \qquad q _ { t + 1 , i } = q _ { t , i } \exp ( \eta v _ { t , i } ) , \qquad Z _ { t } = \sum _ { i = 1 } ^ { n } q _ { t , i } , \qquad w _ { t , i } = \frac { q _ { t , i } } { Z _ { t } } .
$$

Hoefding’s lemma gives

$$
\ln { \frac { Z _ { t + 1 } } { Z _ { t } } } = \ln \left( \sum _ { i = 1 } ^ { n } w _ { t , i } \exp ( \eta v _ { t , i } ) \right) \leq \eta { \mathcal { L } } ( x _ { t } , w _ { t } ) + { \frac { \eta ^ { 2 } B ^ { 2 } } { 8 } } .
$$

Summing over rounds and telescoping, using $Z _ { 1 } = n$ , yields

$$
\ln Z _ { T + 1 } - \ln n = \sum _ { t = 1 } ^ { T } \ln \frac { Z _ { t + 1 } } { Z _ { t } } \leq \eta \sum _ { t = 1 } ^ { T } \mathcal { L } ( x _ { t } , w _ { t } ) + \frac { T \eta ^ { 2 } B ^ { 2 } } { 8 } .
$$

On the other hand, since $\begin{array} { r } { q _ { T + 1 , i } = \exp ( \eta \sum _ { t = 1 } ^ { T } v _ { t , i } ) } \end{array}$ 2

$$
\ln Z _ { T + 1 } = \ln \left( \sum _ { i = 1 } ^ { n } \exp \left( \eta \sum _ { t = 1 } ^ { T } v _ { t , i } \right) \right) \geq \ln \left( \exp \left( \eta \operatorname* { m a x } _ { 1 \leq i \leq n } \sum _ { t = 1 } ^ { T } v _ { t , i } \right) \right) = \eta \operatorname* { m a x } _ { 1 \leq i \leq n } \sum _ { t = 1 } ^ { T } v _ { t , i } ,
$$

where the inequality holds because a sum of positive terms is at least its largest term. Combining these bounds gives

$$
\eta \operatorname* { m a x } _ { 1 \leq i \leq n } \sum _ { t = 1 } ^ { T } v _ { t , i } \leq \ln n + \eta \sum _ { t = 1 } ^ { T } \mathcal { L } ( x _ { t } , w _ { t } ) + \frac { T \eta ^ { 2 } B ^ { 2 } } { 8 } .
$$

Rearranging and dividing by $\eta T$ ,

$$
\frac { 1 } { T } \operatorname* { m a x } _ { 1 \leq i \leq n } \sum _ { t = 1 } ^ { T } v _ { t , i } - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } ( x _ { t } , w _ { t } ) \leq \frac { \ln n } { \eta T } + \frac { \eta B ^ { 2 } } { 8 } .
$$

Maximizing a linear function over $\Delta ( H )$ selects a model with the largest cumulative loss, so

$$
\operatorname* { m a x } _ { w \in \Delta ( H ) } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } ( x _ { t } , w ) = \frac { 1 } { T } \operatorname* { m a x } _ { 1 \leq i \leq n } \sum _ { t = 1 } ^ { T } v _ { t , i } .
$$

Thus, the left-hand side of the preceding regret inequality is $R _ { \mathrm { a d v } }$ , and

$$
R _ { \mathrm { a d v } } \leq { \frac { \ln n } { \eta T } } + { \frac { \eta B ^ { 2 } } { 8 } } .
$$

Second, Assumption 4.1 gives, for each $t ,$

$$
\mathcal { L } ( x _ { t } , w _ { t } ) \leq \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , w _ { t } ) + \epsilon _ { t } .
$$

Consequently,

$$
R _ { \mathrm { r e c } } \leq \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , w _ { t } ) - \operatorname* { i n f } _ { x \in K } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } ( x , w _ { t } ) + \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \epsilon _ { t } \leq \bar { \epsilon } _ { T } ,
$$

because the average of the infima is at most the infimum of the average. Combining the two bounds proves Equation 4.3. Substituting $\eta = B ^ { - 1 } \sqrt { \ln n / T }$ gives

$$
\mathrm { G a p } _ { \operatorname* { m i x } } ( x _ { 1 : T } , \bar { w } ) \leq \frac { 9 } { 8 } B \sqrt { \frac { \ln n } { T } } + \bar { \epsilon } _ { T } \leq 2 B \sqrt { \frac { \ln n } { T } } + \bar { \epsilon } _ { T } .
$$

For $n = 1$ , the adversary has only one action and $R _ { \mathrm { a d v } } = 0$ . The conclusion follows directly from $R _ { \mathrm { r e c } } \leq \bar { \epsilon } _ { T }$ . Proof of Corollary 4.5. Since each $x _ { t } \in K$ and K is convex, ${ \bar { x } } \in K$ . Also, $\bar { w } \in \Delta ( H )$ . Convexity of $\mathcal { L } ( \cdot , w )$ implies, for every $w \in \Delta ( H )$

$$
\mathcal { L } ( \bar { x } , w ) \leq \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { L } ( x _ { t } , w ) .
$$

Maximizing over $w$ and subtracting the same term $\operatorname* { i n f } _ { x \in K } { \mathcal { L } } ( x , { \bar { w } } )$ from both sides yields

$$
\mathrm { G a p } ( \bar { x } , \bar { w } ) \leq \mathrm { G a p } _ { \operatorname* { m i x } } ( x _ { 1 : T } , \bar { w } ) .
$$

The result now follows from Theorem 4.3.

Deterministic Certificate for Tree-Based Models.

Proposition A.1 (Deterministic Recourse Certificate). Let $C _ { T } \subseteq K$ be a finite set of feasible candidates and suppose $b _ { t } \leq \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , w _ { t } )$ for each t. Define

$$
L _ { T } : = \operatorname* { m a x } _ { 1 \leq t \leq T } b _ { t } , \quad U _ { T } : = \operatorname* { m i n } _ { x \in C _ { T } } \operatorname* { m a x } _ { 1 \leq i \leq n } \ell ( h _ { i } ( x ) , 1 ) , \quad V _ { \mathrm { p u r e } } : = \operatorname* { i n f } _ { x \in K } \operatorname* { m a x } _ { 1 \leq i \leq n } \ell ( h _ { i } ( x ) , 1 ) .
$$

Then $L _ { T } \leq V _ { \mathrm { p u r e } } \leq U _ { T }$ , and a candidate attaining $U _ { T }$ has deterministic objective suboptimality at most $U _ { T } - L _ { T }$

Proof. For any $x \in K$ and $w _ { t } \in \Delta ( H )$ ),

$$
\mathcal { L } ( x , w _ { t } ) \leq \operatorname* { m a x } _ { 1 \leq i \leq n } \ell ( h _ { i } ( x ) , 1 ) .
$$

Taking infima over x shows that

$$
b _ { t } \leq \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , w _ { t } ) \leq V _ { \mathrm { p u r e } } .
$$

Taking the maximum over t gives $L _ { T } \leq V _ { \mathrm { p u r e } }$ . Since $C _ { T } \subseteq K$ , every candidate’s worst-model loss is at least $V _ { \mathrm { p u r e } }$ , and hence $V _ { \mathrm { p u r e } } \leq U _ { T }$ . If xbT attains $U _ { T }$ , then

$$
0 \leq \operatorname* { m a x } _ { 1 \leq i \leq n } \ell ( h _ { i } ( \widehat { x } _ { T } ) , 1 ) - V _ { \mathrm { p u r e } } = U _ { T } - V _ { \mathrm { p u r e } } \leq U _ { T } - L _ { T } .
$$

This certificate is useful when the bounds are tight; the mixture convergence theorem does not imply that $U _ { T } - L _ { T }$ vanishes. Accordingly, the experiments distinguish the convex point gap, the tree response-mixture gap and deterministic certificates, and empirical recourse quality for the diferentiable nonconvex oracles. A small optimization gap does not by itself ensure validity across all competing models, which may be impossible for the given budget and competing set.

Evaluating Gaps with an Approximate Optimization Solver. Let $D ( \bar { w } ) : = \operatorname* { i n f } _ { x \in K } \mathcal { L } ( x , \bar { w } )$ , and suppose a separate solver provides bounds $b \leq D ( \bar { w } ) \leq u$ . For the mixture gap, set $P = \mathrm { m a x } _ { i } T ^ { - 1 } \textstyle \sum _ { t } \ell \bigl ( h _ { i } ( x _ { t } ) , 1 \bigr )$ for a feasible deterministic point ${ \widehat { x } } ,$ set $P = \operatorname* { m a x } _ { i } \ell ( h _ { i } ( { \widehat x } ) , 1 )$ . In either case, the corresponding gap belongs to $\lceil \operatorname* { m a x } \{ 0 , P - u \} , P - b \rceil$ . A feasible local solution to the weighted minimization problem provides an upper bound $u ,$ and therefore a lower bound on the gap; it cannot by itself certify that the gap is small. This optimization interval is distinct from a confidence interval across experimental seeds.

Convergence Bounds at Intermediate Checkpoints. For a run using a fixed learning rate $\eta ,$ the proof of Equation 4.3 applies to every prefix of $s \leq T$ rounds, giving the bound ln $n / ( \eta s ) + \eta B ^ { 2 } / 8 + s ^ { - 1 } \textstyle \sum _ { t = 1 } ^ { s } \epsilon _ { t }$ when $n \geq 2 . \mathrm { ~ A ~ }$ rate obtained by substituting $\eta = B ^ { - 1 } \sqrt { \ln n / s }$ instead describes a learning rate chosen for horizon s and should not be substituted into a fixed-horizon trajectory without changing its updates.

Why the Nonconvex Distinction Matters. Consider a feasible set with two recourses whose loss vectors across two models are (0, 1) and (1, 0). Every deterministic recourse has worst-model loss 1, while their uniform mixture has worst expected loss $1 / 2$ . Against the uniform model distribution, that mixture has zero gap, but either deterministic recourse has point gap $1 / 2$ despite being optimal for the deterministic problem. Thus, mixture convergence does not imply deterministic convergence, and a positive nonconvex point gap need not indicate a suboptimal deterministic recourse.

## B Omitted Details from Section 5

## B.1 Datasets

For the synthetic dataset, we used 2,000 observations for linear baselines and 20,000 observations from the same distribution for non-linear baselines. The goal is to predict which of the two Gaussian distributions generated the observation. For COMPAS, Polish, and German datasets (processed from the CSV files in the datasets folder), we removed non-feature index columns, converted entries to numeric values, and removed rows containing missing or non-numeric values. We then balanced the two labels by randomly undersampling the majority class without replacement, using a fixed seed of 42. The resulting datasets contain 6,392, 864, and 600 observations for each of the datasets, respectively. The German dataset originally contained 1,000 observations and 20 attributes. After balancing the labels and one-hot encoding, the observations decreased to 600 with 67 input coordinates. For the COMPAS dataset, the goal is to predict whether the defendant is arrested within two years. For the Polish dataset, the goal is to predict a company’s bankruptcy status. The dataset contains 864 observations on 64 financial indicators. For the German dataset, the goal is to classify an applicant with good or bad credit. For the Give Me Some Credit dataset, we retained the 10 predictors and removed rows with missing values, leaving 120,269 labeled observations. We did not balance this dataset. The goal is to predict serious delinquencies (being at least 90 days past due) within the following two years. For Folktables, we use the standard ACSIncome task on California’s 2018 one-year ACS person-level data. Its eligibility filter retains individuals older than 16, with annual personal income above \$100, positive usual weekly working hours, and a person weight of at least one. The goal is to predict whether annual personal income exceeds \$50,000. This produces 195,665 observations with 10 predictors. We retain the numeric feature codes without additional one-hot encoding and do not balance the classes. See Table 2 for a summary of the statistics of each dataset.

<table><tr><td>Dataset</td><td>Features before encoding</td><td>Model input features</td><td colspan="2">ObservationsFavorable label in source data</td><td>Immutable features</td></tr><tr><td>Synthetic</td><td>2</td><td>2</td><td>2,000*</td><td>y = 1 (generated positive class)</td><td>None</td></tr><tr><td>COMPAS</td><td>7</td><td>7</td><td></td><td>6,392 two_year_recid=0 (no recorded two-year recidivism)</td><td>Age (age); sex (sex=female)</td></tr><tr><td>Polish</td><td>64</td><td>64</td><td></td><td>864 Recorded target y = 1</td><td>None</td></tr><tr><td>German</td><td>20</td><td>67</td><td></td><td>600 Local y = 0 (good credit)</td><td>Age (Attribute13); dependents (Attribute18); personal status/sex (all Attribute9 indicators); foreign-worker status (all</td></tr><tr><td>Give Me Some Credit</td><td>10</td><td>10</td><td></td><td>120,269 SeriousD1qin2yrs= 0 (no serious delinquency dependents within two years)</td><td>Attribute20 indicators) Age (age); number of (NumberOfDependents)</td></tr><tr><td>Folktables (ACSIncome)</td><td>10</td><td>10</td><td></td><td>195,665 target= 1 (annual personal income &gt; $50,000)</td><td>Age (AGEP); marital status (MAR); place of birth (POBP); relationship to householder (RELP); sex (SEX); race</td></tr></table>

Table 2: A summary of datasets. <sup>∗</sup>The synthetic nonlinear-baseline comparison uses 20,000 observations. The listed source labels are mapped to model class 1; immutable features remain fixed during recourse, including every indicator in a listed one-hot group.

For each experimental seed, we construct stratified training, validation, and test splits containing approximately 64%, 16%, and 20% of the observations, respectively. We fit the feature standardization using only the training split and apply the same transformation to validation and test observations.

## B.2 Implementation Details

Models and Training. We implemented logistic regression and neural networks in PyTorch. Logistic regression consists of an afine logit followed by a sigmoid. The first neural architecture has one fully connected hidden layer with 20 units; the second has two fully connected hidden layers with 50 and 100 units, respectively. Both architectures use ReLU hidden activations and a sigmoid output. The main training procedure uses full-batch Adam for 100 epochs with learning rate 0.01, minimizing binary cross-entropy through its numerically stable logit formulation.

Decision trees were trained using Scikit-Learn. The larger tree experiments use maximum depth 4, a minimum of 5 training observations per leaf, and square-root feature subsampling.

Data Splitting and Instance Selection. For each seed, we first reserved 20% of the data for testing and then reserved 20% of the remaining data for validation, producing approximately 64% training, 16% validation, and 20% testing data. Both splits were stratified by label. Feature standardization was fitted exclusively on the training split. Models were trained on the training data and admitted to the competing set using validation accuracy.

The first model in the competing set serves as the reference model in the reference baseline. Recourse is generated only for test instances for which at least half of the models in the competing set assign a probability below 0.5. The same models and factual instances are used across methods and budgets within each experimental configuration. Unless a numerical acceptance convention is explicitly specified, a returned point is counted as jointly valid only if it satisfies the budget and feature constraints and every competing model assigns it probability at least 0.5. Table 1b uses the logit tolerance described below.

Construction of Diferentiable Competing Sets. In the linear baseline comparison, we trained a central logistic regression model and sampled additional parameter vectors from its local Hessian ellipsoid, following the geometric construction of Turbal et al. [60]. Including the intercept, the sampling region is $( \theta - \theta _ { 0 } ) ^ { \top } A ( \theta - \theta _ { 0 } ) \leq 2 \kappa$ , where A is the Hessian of the average training binary cross-entropy, regularized by $1 0 ^ { - 5 } I ,$ and $\kappa = 0 . 0 5$

The mixed-model experiments train logistic regressions and neural networks with diferent initializations. Feature-diverse replacements retain approximately 50% or 75% of predictor groups, including possible omission of actionable groups. Candidates must pass the validation-accuracy floor and lie within 0.02 accuracy of their slot’s control model. Greedy selection favors validation disagreement, with control-model fallback when no eligible replacement exists.

For both construction procedures, a candidate is admitted when its validation accuracy is no more than 0.05 below that of the central logistic regression model. This is a one-sided admission criterion; candidates with higher accuracy are also admitted. These are empirically filtered competing sets, rather than certified enumerations of a Rashomon set. The floor is relative to central-logistic accuracy, not to the best available model. The mixed-model trade-of experiments use (2, 2, 2), (4, 4, 4), and (8, 8, 8) logistic, single-hidden-layer, and double-hidden-layer models, respectively, giving competing sets of 6, 12, and 24 models.

Construction of Tree Competing Sets. Tree models are trained on the same training split using diferent random seeds. Trained models must have validation accuracy within 0.05 of the best model in the generated optimization pool. Separately generated evaluation models must lie within a two-sided 0.05 accuracy band around that reference and are never queried by the recourse optimizers.

Recourse Domain and Loss. We use standardized $L _ { 2 }$ distance as the cost, fixing the listed immutable coordinates. The diferentiable trade-of and same-architecture neural experiments use raw logistic loss $\ell _ { j } ( z ) = \log ( 1 + \exp [ - g _ { j } ( z ) ] )$ , where $g _ { j }$ is model $j ^ { \cdot } \mathrm { s }$ logit. Tree experiments use squared probability loss. The convex duality-gap experiment uses the separately specified normalized logistic loss.

Diferentiable ORPM Oracle. Neural ORPM starts with uniform weights, updates $w _ { t + 1 , j } \propto w _ { t , j } \exp ( \ell _ { j } ( z _ { t } ) )$ and returns the arithmetic mean of its responses. Each response uses $\mathrm { S L S Q P }$ with at most 100 iterations and tolerance $1 0 ^ { - 8 }$ , initialized at the previous response (the factual point initially). The best weighted-loss point among the factual point, initialization, and projected proposal is retained. We use 200 rounds for every dataset except GMC, where we use 1,000 rounds. These are local neural optimization runs with raw outer rate 1; they do not establish global-oracle assumptions.

Tree-based ORPM Oracle. Following the mixed-integer approach to tree recourse of Parmentier and Vidal [47], the oracle jointly encodes the competing models’ leaf selections, predictions, immutable features, and recourse budget. With squared probability loss and an $L ^ { 2 }$ budget, the implementation is a mixed-integer quadratically constrained optimization problem. It is solved using Gurobi, reusing the formulation across rounds and updating the model weights and warm start. The expanded tree experiments allow one second per ORPM response solve, use one solver thread, and request an absolute optimality gap of $1 0 ^ { - 5 }$ . ORPM uses 100 rounds in the expanded tree budget-validity comparison and 200 in the tree convergence experiments. These are computational limits and stopping targets, not guarantees that every call attains that accuracy. Returned candidates are checked against the original models and feasibility constraints. Solver status, objective bounds, and verified incumbent values are retained. Any resulting optimality certificate is subject to the solver’s numerical tolerances. Tree ORPM returns the candidate with the smallest worst-model loss among the factua point and oracle responses, breaking ties by cost. It does not average feature vectors. Consequently, the convex theorem for an averaged recourse point does not directly justify this return rule.

Diferentiable Baselines. We use the following approaches as baselines.

• ADV: ADV applies 500 projected Adam steps with learning rate 0.01δ, where δ is the recourse budget, directly to the largest loss among the competing models. Each step diferentiates through a currently worst-loss model. It returns its best observed worst-loss point, without tolerance-based early stopping.

• ROAR: Our ROAR implementation [61] uses a fixed $L ^ { \infty }$ uncertainty box around the deployed model’s afine parameters, including its intercept. The radius is the largest coordinatewise parameter diference between the deployed model and the competing models. For nonlinear models, these parameters come from factual-point LIME approximations of the logits [53], anchored to match each original model at that point. Recourse is computed by maximizing the box’s worst-case logit using SLSQP, subject to the common budget and feature constraints. ROAR uses SLSQP with at most 500 iterations and tolerance $1 0 ^ { - 4 }$ . For neural models, local logit surrogates use 5,000 samples.

• ElliCE: Our implementation of ElliCE [60] uses the training-loss Hessian ellipsoid centered on the deployed model. For neural networks, the feature extractor is fixed, and uncertainty is applied to the final afine layer. So unlike ROAR, LIME is not used for non-linear models. SLSQP first searches for a favorable worst-case ellipsoid score and, when successful and iterations remain, reduces $L ^ { 2 }$ cost while preserving that score. ElliCE uses SLSQP tolerance $1 0 ^ { - 4 }$ and experiment-specific iteration caps.

Neural Optimization Baseline in Table 1b. The neural references in Table 1b use the continuous action region

$$
\begin{array} { r } { \mathcal { K } _ { \delta } ( x ) = \{ z : \| z - x \| _ { 2 } \leq \delta , \ z _ { j } = x _ { j } \ \mathrm { f o r ~ i m m u t a b l e } \ j \} . } \end{array}
$$

Writing $g _ { j } ( z )$ for network $j ^ { \circ } \mathrm { s }$ logit, MaxMargin maximizes the minimum logit, while MinL2 minimizes the distance required for acceptance by every seen network. ReLU encodings and the quadratic budget make these MIQCPs. They are our implementation-specific optimization references, related to prior work on joint neural verification [41].

For the raw logistic loss used here,

$$
\operatorname* { m a x } _ { j } \log \bigl ( 1 + \exp [ - g _ { j } ( z ) ] \bigr ) = \log \bigl ( 1 + \exp [ - \operatorname* { m i n } _ { j } g _ { j } ( z ) ] \bigr ) .
$$

Thus MaxMargin and worst-logistic-loss minimization have the same global optimizers on ${ \ K } _ { \delta } ( x )$ . Comparisons concern the returned solutions under the stated computational limits: outperforming a time-limited MaxMargin output does not imply outperforming the global minimax optimum.

<table><tr><td></td><td>MaxMargin</td><td>MinL2</td></tr><tr><td>Objective</td><td> $\operatorname* { m a x } _ { z , t } t ,$  with  $g _ { j } ( z ) \geq t$  for every seen network</td><td> $\begin{array} { r } { \operatorname* { m i n } _ { z } \frac { 1 } { 2 } \| z - x \| _ { 2 } ^ { 2 } , } \end{array}$  with  $g _ { j } ( z ) \geq 0$  for every seen network</td></tr><tr><td>Domain Solver/cap</td><td> $z \in \mathcal { K } _ { \delta } ( x )$  Gurobi, one thread, 60 seconds per</td><td> $z \in \mathcal { K } _ { \delta _ { \operatorname* { m a x } } } ( x )$  Gurobi, one thread, 60 seconds once per</td></tr><tr><td></td><td>positive-budget solve</td><td>person/model set at the largest budget; reuse for smaller budgets</td></tr><tr><td>Return</td><td>Verified incumbent or factual point, whichever has the larger minimum logit</td><td>Verified incumbent when its distance fits the queried budget; otherwise mark the result unavailable/unresolved as appropriate</td></tr><tr><td>Unavailable output</td><td>Factual fallback, evaluated explicitly</td><td>Factual placeholder; primary validity is zero, and cost is missing, not zero</td></tr></table>

The native relative and absolute gap targets are $\overline { { 1 0 ^ { - 5 } \mathrm { ~ a n d ~ } 1 0 ^ { - 6 } } } ;$ ; feasibility and integrality tolerances are ${ \overline { { 1 0 ^ { - 9 } } } } .$ For Table 1b, primary acceptance uses $g _ { j } ( z ) \geq - 1 0 ^ { - 7 }$ for every evaluated network, with the same convention for ORPM, MaxMargin and MinL2 on both seen and unseen panels. Strict zero-threshold results are retained separately, and budget checks allow $1 0 ^ { - 8 }$ . Timeouts retain their native status and do not establish infeasibility. Solver bounds are numerical certificates, not exact-arithmetic proofs. No ORPM warm starts or replacement actions are used. The 60-second limit excludes formulation and shared-preparation time.

Tree-based Baselines. We include the following explicitly labeled adaptations:

• ADV-surrogate: projected optimization using smooth sigmoid approximations to tree splits, inspired by FOCUS [42], with temperature 0.1. Candidate quality is checked using the original tree models.

• ROAR-surrogate: the parameter-box method applied to anchored local logit surrogates.

• RobX-constrained: an adaptation of RobX [18] that refines a deployed-model counterfactual using a local stability score and projected training-data anchors. Defaults are 100 Gaussian probes with standard deviation 0.1, stability threshold 0.7, interpolation coeficient 0.1, and up to three anchors selected from 200 candidates.

• OCEAN-style: minimum- $L ^ { 2 } .$ -cost recourse for the deployed tree using our mixed-integer formulation, following Parmentier and Vidal [47].

• Direct minimax: a direct mixed-integer optimization reference for the finite competing-set minimax objective. Its objective and computational allowance are specified in Appendix B.3.5.

The adapted baselines retain their stated budget and search limits; theoretical guarantees from their original formulations are not automatically transferred to these adaptations.

Evaluation of Duality Gap. The convex convergence experiment uses afine-logit models with logistic loss $\widetilde { \ell } _ { i } ( x ) = \log \left( 1 + \exp ( - a _ { i } ^ { \top } x - b _ { i } ) \right)$ , divided by a common upper bound over the feasible region so that all losses lie in [0, 1]. Its oracle uses float64 SLSQP with at most 200 iterations and tolerance $1 0 ^ { - 1 0 }$ , together with a convex supporting-plane lower bound. At each diagnostic checkpoint, let P denote the primal objective and let $\begin{array} { r } { q ( \bar { w } ) : = \operatorname* { m i n } _ { x \in K } \mathcal { L } ( x , \bar { w } ) } \end{array}$ denote the dual objective. If the diagnostic solver supplies bounds $L \leq q ( \bar { w } ) \leq U$ we reportGap ∈ [max{0, P − U}, P − L] . These optimization-bound intervals are distinct from statistical confidence intervals. Linear convergence diagnostics use normalized logistic loss; neural/mixed diagnostics use raw logistic loss. The numerical target 0.005 therefore does not represent a common loss scale across these settings, and attainment rates are interpreted within each stated diagnostic.

For neural sets, we evaluate the saved responses after the trajectory has finished. At each checkpoint, we use the average pre-update weights w¯ and compute both $P _ { \mathrm { p o i n t } } = \operatorname* { m a x } _ { i } \ell _ { i } ( \bar { z } )$ and $P _ { \mathrm { m i x } } = \mathrm { m a x } _ { i } T ^ { - 1 } \sum _ { t = 1 } ^ { T } \ell _ { i } ( z _ { t } )$ A feasible SLSQP diagnostic response supplies an upper bound on $\begin{array} { r } { q ( \bar { w } ) = \operatorname* { i n f } _ { z \in \mathcal { K } _ { \delta } ( x ) } \sum _ { i } \bar { w } _ { i } \ell _ { i } ( z ) } \end{array}$ . Lower bounds combine analytic component bounds with a mixed-integer encoding of the saved ReLU networks and piecewise-linear tangent minorants of logistic loss, using a one-second diagnostic allowance per checkpoint and five seconds at the final checkpoint. All quantities use the same loss scale. We report $[ \operatorname* { m a x } \{ 0 , P - U \} , P - L ]$ separately for each primal quantity. Numerical lower bounds contradicted by saved feasible witnesses, including later responses, are discarded together with their derived bounds. Note that these are numerical optimization intervals, not sampling intervals or exact-arithmetic proofs.

Tree diagnostics evaluate both the returned candidate and the uniform distribution over oracle responses. For the latter, the primal objective is $\begin{array} { r } { P _ { \operatorname* { m i x } } = \operatorname* { m a x } _ { i \in \{ 1 , \dots , n \} } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \ell ( h _ { i } ( x _ { t } ) , 1 ) } \end{array}$ , rather than the loss at an averaged feature vector. Diagnostic solves are separate from the ORPM trajectory and do not influence its updates.

Cohorts and Aggregation for the Cost-of-Multiplicity Success Curves. For a fixed dataset, model count, and budget, let $n _ { s g }$ be the evaluated count and $\widehat { p } _ { s g }$ the mean instance-level success measure in group g of seed s. Our cost of multiplicity results use

$$
\widehat { p } = \frac { 1 } { 1 0 } \sum _ { s = 4 0 1 } ^ { 4 1 0 } \sum _ { g : n _ { s g } > 0 } \frac { W _ { s g } } { \sum _ { h } W _ { s h } } \widehat { p } _ { s g } ,
$$

as the estimated overall recourse success rate. For linear and tree models, $W _ { s g }$ counts all eligible held-out test instances in that group’s saved model bank: at least half the models reject the factual point in each of the nested 6/12/24-model sets. For neural models, $W _ { s g }$ counts group members in the original twenty-instance source cohort, whereas only four instances per seed are evaluated in the cost comparison. Synthetic and Polish have no designated demographic groups and use ordinary within-seed means: twenty instances per seed for linear/trees and four for neural models.

Linear selection retains the original twenty instances and appends instances in the saved eligible order until twenty per group or exhaustion; tree selection retains the earlier cost-study cases and fills to twenty; neural selection retains pilot cases, targets two per group within the original twenty, and fills shortages in saved order. No selected case is dropped from these success curves because recourse fails; unresolved benchmark feasibility remains a lower/upper success range, with the same aggregation applied to each endpoint. The only absent group is recorded women in neural COMPAS seed 409 $( n = 0 , W = 0 )$ , which contributes zero weight to its overall rate. Disparity uses $\begin{array} { r } { D = | S | ^ { - 1 } \sum _ { s \in S } ( \widehat { p } _ { s 1 } - \widehat { p } _ { s 2 } ) } \end{array}$ , without population weights, where $S$ contains seeds with both groups: nine for neural COMPAS (excluding 409), ten otherwise. Plots of gap magnitude use |D|. Pointwise 95% Student-t intervals use the contributing seed means (nine degrees of freedom for overal rates, eight for neural COMPAS disparity). These conventions concern the cost/multiplicity studies and their group contrasts, not the separate trade-of or unseen-model experiments.

For example, German tree seed 401 evaluates 14 instances under 40 and 6 aged 40+, but uses eligiblepopulation counts 41 and 17, hence weights $4 1 / 5 8$ and 17/58, rather than $1 4 / 2 0$ and 6/20. Across ten seeds, German ORPM success is 53.4272% at twelve trees and δ = 2 (unweighted: $1 0 7 / 2 0 0 = 5 3 . 5 \% ) ;$ the linear 24-model values are 57.0909% and 99.6822% at δ = 2, 5. Thus, the displayed 53.4%, 57.1% and 99.7% are weighted estimates, not raw percentages out of 200.

## B.3 Additional Experimental Results

## B.3.1 Duality-Gap

For Synthetic, COMPAS, Polish, and German datasets, we used a budget of {1, 2, 5}. For GMC, we used a budget of {50, 60, 75}. The set of budgets used for ACSIncome is {2, 5, 10}. Additional convergence results for other datasets not covered in the main body are provided in Figure 8.

We also compare three potential return rules for ORPM: the mean of the first T responses (as the theory suggests), the last response, and the recourse with the smallest worst loss among the factual point and the first T responses, breaking exact loss ties by cost. For this analysis, we use a competing set of twelve neural networks (six with one 20-ReLU hidden layer and six with 50/100-ReLU hidden layers), five seeds, and five instances per seed on every dataset, at δ = 2 for Synthetic, COMPAS, Polish and German, 60 for GMC and 5 for ACSIncome. We observe that joint validity is equal across the three rules on five datasets. On GMC and using $T = 2 0 0$ rounds, the joint validity is 72% for averaging and 76% for last and best, becoming 76% for all three at $T = 1 { , } 0 0 0$

![](images/9aff7c6b7f295bc6e81f3c2c5d0f47f223c59fed22b86113e7264de7c5f072db.jpg)  
(a) ASCIncome dataset, 6 competing models, $\delta = 2 .$

![](images/a096611e8ae8efcbcdd70ab429a82dca1489044279fa1e8996785b0a6cfe564c.jpg)  
(b) German dataset, 6 competing models, δ = 2.

![](images/04686e7a3119e1afb5af6f977765abc89447c3ec0ea8cef9aac6991b59d43029.jpg)  
(c) GMC dataset, 24 competing models, δ = 75.

![](images/b5704f467f9d7c0d2763a886a0c3b4c322e34a085f350354d21934521c6789b8.jpg)  
(d) Synthetic dataset, 24 competing models, δ = 2.  
Figure 8: Fraction of instances whose certified upper bound on the averaged-point gap is at most 0.005 at each checkpoint. Each subfigure corresponds to a diferent dataset. The number of models and budget are mentioned in the legend.

## B.3.2 Trade-of Between Validity and Budget

For Synthetic, COMPAS, and German datasets, we used a budget of {0.5, 1, 2, 3, 5}. For the Polish dataset, we used a budget of {0.25, 0.5, 1, 2, 3, 5}. For GMC, we used a budget of {5, 10, 25, 50, 60, 75, 100}. The budget used for ACSIncome is {0.5, 1, 2, 3, 5, 7.5, 10}. Additional trade-of results are provided in Figure 9. Additional results on how budget and the size or complexity of the competing set afect the validity are provided in Figures 10 and 11.

## B.3.3 The Cost of Multiplicity

Additional cost of multiplicity results for the remaining datasets are provided in Figures 12 and 13. These are the analogous results to Figure 4 in the main body, which focused on the German dataset.

![](images/fc4fe06a5ceb76ec69b52c81c6843b63a3c27f0dc534fb70f23d9e455ffcb48a.jpg)  
(a) ACSIncome dataset, mixed competing set with 12 models

![](images/6d69c2db7d5e2dab57601c7b6a3944a36141599f8a347574e124df4591fbc4d8.jpg)  
(b) Synthetic dataset, linear competing set with 12 models

![](images/497c95ea01f96f55ace323255d0a3a047c0214ba420794f92f5cd0a15465ad53.jpg)  
(c) Polish dataset, linear competing set with 6 models

![](images/944206eb0cee033db5b04751f58d1f1da2bb8dc23c6bc9a1ec22450296eba295.jpg)  
(d) COMPAS dataset, neural competing set with 6 models

![](images/5d7bf479a920fb58c7dbdf2199a142feca4119572d115edbe2b4378cbe1e9796.jpg)  
(e) GMC dataset, mixed competing set with 24 models

![](images/161094a20a95d3dc0706c8f222dcc12f7ccc458697b8dc15db74d5d5265b989c.jpg)  
(f) GMC dataset, linear competing set with 24 models

Figure 9: The trade-of between validity and budget. Each subfigure corresponds to a dataset and competing set as described in the legend. Each curve corresponds to a diferent approach to provide recourse.

## B.3.4 The Disparity of Multiplicity

Figure 14 depicts the disparity gap of ORPM and baselines for the neural settings. On ACSIncome, ORPM’s gap decreases from 15 percentage points at budget 1 to zero at budget 3, below the optimization ranges

![](images/1289d58ba81d60413d7764775124ad2fd990d0c603dfd9078a0c19a952a5db07.jpg)  
(a) The efect of budget on ORPM’s validity for a neural competing set of 24 models.

![](images/93d744c11bb0ed1d5852de8853694d77d7146e9b81b494dc765d6f2b3019095f.jpg)  
(b) The efect of budget on ORPM’s validity for a linear competing set of 12 models.

Figure 10: The efect of budget (X-axis) on the validity of the recourse returned by ORPM (Y-axis). Each curve corresponds to a diferent dataset.  
![](images/659cd0cdfe561073075521e2704133aac89955f459d52454e9002c37dd18738f.jpg)  
(a) The efect of the number of models in a neural competing set on ORPM’s validity.

![](images/ad7878051b21e666e09d728fa280f7ef4e45b7764db79e4e991bb81555e5cc3a.jpg)

![](images/376f515ff5eaa8d4848c64b33af9f36558a6c35b557b932f364cdf82a7295e2a.jpg)  
(c) The efect of the number of models in a linear competing set on ORPM’s validity.

(b) The efect of the number of models in a mixed competing set on ORPM’s validity.  
![](images/c98e4900fee976f58190afde7d6663fe20d6b6b5534f17e124ff05bd54ae0070.jpg)  
(d) The efect of composition of the competing set of 24 models on ORPM’s validity.  
Figure 11: The efect of composition or size of the competing set (X-axis) on validity of the recourse returned by ORPM (Y-axis). Each curve corresponds to a diferent dataset.

![](images/8a124e9560fe9dfed055741144df3461e0aa22253a6174a57c0897d43a5cdde3.jpg)  
(a) Synthetic dataset, linear competing set of 24 models.

![](images/8ed92bb423d416c365d95925e4ec5211c828d4c3aa21263c65129d9066c56de5.jpg)  
(b) Synthetic dataset, neural competing set of 24 models.

![](images/387bfb2a2c18a4578ee13688e48c4fcfaa576370eae8b20e787308f6478eb978.jpg)  
(c) Polish dataset, linear competing set of 12 models.

![](images/8f0811da9f0e9d8e588d6c195941d55d6f8bdd726956257b6c32acc5676f23f5.jpg)  
(d) Polish dataset, neural competing set of 12 models.

![](images/c8af20c87820225a2d19f3a55d85ef2cadc2b7f62f31a510d1bd142d431f4b15.jpg)

![](images/01fd0c103547885318643d927eb3950ac58420ec9b1fdf6a82dc1e1d71cc0380.jpg)  
(e) COMPAS dataset, linear competing set of 24 models. (f) COMPAS dataset, neural competing set of 24 models.  
Figure 12: The validity degradation to guard against multiplicity. Each subfigure corresponds to a pair of datasets and competing sets as indicated in the legend. Each curve tracks the success rate (Y-axis) for ORPM and baselines as a function of allowable budget (X-axis).

identified for the all-separately and common-action benchmarks at both budgets. The COMPAS gap decreases from 24.1 to 20.4 points as the budget increases from 1 to 2, whereas the German gap increases from 10 to 20 points. At the moderate budget, ORPM has larger gaps than the average singleton on both COMPAS

![](images/7fbc214a38fe45b03db12cd411385b03bd701f8844262b284699cb26052ec60b.jpg)  
(a) ACSIncome dataset, linear competing set of 12 models.

![](images/b64f9ee8f83352c01d4efc435a8771fe09dd82aa280831152dc4743edd87c879.jpg)  
(b) ACSIncome dataset, neural competing set of 12 models.

![](images/65a2e1421cf330c82659121ce4b1820ac82d1caabae66ec42c20f6f5b86626ca.jpg)  
(c) GMC dataset, linear competing set of 24 models.

![](images/6e9d9f6ff8973feae46d7edddfe8e78843543c77383ae2e134071dd392f14ce9.jpg)  
(d) GMC dataset, neural competing set of 24 models.  
Figure 13: The validity degradation to guard against multiplicity. Each subfigure corresponds to a pair of datasets and competing sets as indicated in the legend. Each curve tracks the success rate (Y-axis) for ORPM and baselines as a function of allowable budget (X-axis).

and German. On GMC, ORPM, all separately, and common action have zero estimated disparity at both displayed budgets, but this accompanies only 10% success in each group. Small gaps therefore need not indicate efective recourse.

![](images/055817dac352e3370a17e691be935caca3847ee21e9dbd1d9fcb652c6fd81fc4.jpg)  
(a) Neural competing set, small budget regime.

![](images/2f9caa0cadcadf58371af846da85386d71efab60102bab7aea2628abf6904730.jpg)  
(b) Neural competing set, medium budget regime.

Figure 14: The disparity of recourse success across diferent groups. Each subfigure corresponds to diferent budget values, with small budgets on the left and medium budgets on the right. In each subfigure, the bars track the disparity in recourse success for ORPM and baselines on each dataset.

## B.3.5 Tree-Based Competing Sets

Figures 15-17 extend the decision-tree results across datasets, using 10 seeds and 20 instances per seed. At the middle budget, mixture-gap certification varies markedly: Polish with twelve trees has no certified attainment of 0.005 by round 200, and COMPAS and ACSIncome also have none, despite positive deterministic validity in the budget experiments (see Figure 15). In the budget-validity comparison, increasing δ raises ORPM’s observed joint validity on COMPAS with twelve trees from 21.5% at $\delta = 0 . 5$ to 68.5% at $\delta = 5$ , whereas ACSIncome remains at 43.5% from $\delta = 5$ to 10. Larger budgets therefore improve success in some settings but do not uniformly eliminate failures (see Figure 16). We use additional baselines for the trade-of experiment. Let $p _ { i } ( z )$ be tree i’s favorable-class probability and $\kappa _ { \delta } ( x )$ the allowed changes within budget δ. The baseline uniform weighting minimizes the average squared probability loss: min $\begin{array} { r } { { \bf \Lambda } _ { \in \mathcal { K } _ { \delta } ( { x } ) } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( 1 - p _ { i } ( z ) ) ^ { 2 } } \end{array}$ . Every tree receives equal weight. Good average performance can still leave some trees rejecting the action. The baseline direct minimax minimizes the largest loss: min $z \in \mathcal { K } _ { \delta } ( x ) \operatorname* { m a x } _ { i } ( 1 - p _ { i } ( z ) ) ^ { 2 }$ . It directly targets the least favorable tree, the same objective as ORPM approaches through repeated weighted responses. Uniform weighting and direct minimax each receive one 30-second, single-thread solve, and their returned solutions are not necessarily globally optimal. The baseline single-tree reference generates an action using only the first tree, then evaluates it against the entire competing set. If that tree already accepts the factual point, it returns the factual point. Otherwise, it enumerates reachable leaves and selects the highest favorable-class probability, breaking ties by smaller $L _ { 2 }$ distance. Finally, requiring a common action can be substantially harder than allowing separate recourse: on German with twelve trees at $\delta = 2$ , average-singleton success is 99.5%, all-separate feasibility 94.5%, and common-action and ORPM success 53.4% (see Figure 17).

![](images/a596035a55af0112b54043b834c04d60b35864aebc533aba43ccd58f571df689.jpg)

![](images/df2860eae58ed5e99c84a9cbc413ec41850049e86f0ccfc9433a71c76146070b.jpg)

![](images/1d5e2c8bfc7ccd2e737b4b0e5de7a14913ad9b7f30079128a8fed7026a205cb7.jpg)

![](images/586bacf4424111fd26ee9cbffc2ac268e34322bb50c45a2d820e2738c4543f28.jpg)

![](images/a6442194d6fb9e668689e2fa47ab1157d565f26b3d57f38223402c8a17520bd6.jpg)

![](images/cf15e2424d40b3296d208a1aab7c06be7e082bee1414643bb0ae4181321b95b0.jpg)  
Figure 15: Decision-tree convergence at δ = 2 for Synthetic, COMPAS, Polish and German, $\delta = 1 0$ for GMC, and $\delta = 5$ for ACSIncome. Curves show the mean upper bound on the empirical-response-mixture gap for 6, $^ { 1 2 , }$ and 24 trees (lower is better), across 10 seeds and 20 instances per seed. The vertical scale is linear near zero and logarithmic above 0.005; the dotted horizontal line marks 0.005. Shaded bands are pointwise 95% Student-t intervals over seed means. The mixture objective averages response losses, not feature vectors, and is distinct from deterministic joint validity.

![](images/d9ed831752f07a09db10aff2ecb059ff6f0f32731e3a4f400530556c48361b6f.jpg)

![](images/b49b60302804b2e7036b942e4ef97d5ff9d354dc1ff3b18957e5c06f72366b92.jpg)

![](images/83036b5186e66fcb647e3c2f011da08c0399354be9b2098a8b95a5d97659b66e.jpg)

![](images/aa94a1567453c7a5af20b24b24ab74d3530c08bf95f96570be7cd6e282d3bf43.jpg)

![](images/76ef3fa4f2f9b4a7a3b2828fcacade98a349ace0536d51b41535c7787a5fc7ea.jpg)

![](images/d92a8dfe79787722652ed2b55e728cdaecbe8ae26648672a1d3f7d468ee899da.jpg)  
Figure 16: Expanded decision-tree budget-validity comparison on all six datasets with 12 competing trees, 10 seeds, and 20 instances per seed. ORPM uses 100 one-second response calls; uniform weighting and direct minimax use one 30-second solve each; the single-tree reference uses leaf enumeration. Shading is pointwise 95% Student-t uncertainty over seed means, clipped to [0, 100]% for display.

![](images/2fc0524cf9358489c3eac863572fa78304c041751639aee3adc8fdf8d4c7cb86.jpg)

![](images/690976b17ed41bcdb065cdbea00144403ce8c00ca7e23cbabacc326eea534638.jpg)

![](images/3cd3b86b267ea7e24194c1ab18d746e224feac5e53d43c9ca6936aa76df37f69.jpg)

![](images/487870ace09d60bca557e88814aca75ee70087f5ca2e5044cbdae8b19a6abfd7.jpg)

![](images/d68c3d0a6d1cee676f36f0f10a92638f9c200c755c0d64611f46447d6533dc5a.jpg)

![](images/42ee57a29ec59b80557ff354c1bda964482fac7a0120d5abda0f16233ff62b70.jpg)  
Figure 17: Cost-of-multiplicity success curves on all six datasets with twelve decision trees with 10 seeds and 20 instances per seed. Dark fill denotes optimization identification ranges and thin dotted boundaries denote pointwise 95% sampling limits at their endpoints. This cohort preserves eleven cases outside the trade-of study’s standard prefix (1,189/1,200 overlap), so the overall curves need not coincide with Figure 16.