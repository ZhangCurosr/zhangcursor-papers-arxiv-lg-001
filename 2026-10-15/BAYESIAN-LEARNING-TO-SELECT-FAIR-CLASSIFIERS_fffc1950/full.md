# BAYESIAN LEARNING TO SELECT FAIR CLASSIFIERS

Saatvik Kher University of California, Irvine

Hyungrok Do New York University

Judy Zhong Weill Cornell Medicine

Padhraic Smyth University of California, Irvine

## ABSTRACT

Evaluations of classifiers (AI or human or agents) often assume a batch setting in which every agent produces labels on every example in a test set, which can be expensive or impractical in many contexts. This motivates online policies that adaptively select agents to query as examples arrive and update policies as outcomes are observed. We study the problem of selecting, in an online stream, the agent that best balances (i) predictive performance and (ii) subgroup fairness of a sensitive attribute, and that allows for delayed observations of true class labels. We formulate this as a Bayesian multi-armed bandit problem using Thompson Sampling, explicitly balancing accuracy and fairness. We conduct empirical evaluations on real-world datasets spanning problems involving tabular classifiers, large language models, and human labelers, demonstrating substantial reductions in regret relative to baselines. Our theoretical contributions include guarantees on the performance of the selected agent that incorporate both the accuracy and fairness aspects of the problem.

## 1 Introduction

Machine learning systems are increasingly deployed as black-box decision-makers in high-stakes domains, such as healthcare diagnosis, content moderation, and credit decisions. This creates an operational need to evaluate individua models for accuracy as well as to select among multiple candidate decision-making agents (potentially including both ML models and human experts). In many applications, the criterion for selection is multi-faceted and needs to take into account not only prediction accuracy but also the fairness of an agent and (potentially) the cost of querying the agent.

Prior work in this general context often assumes a batch setting in which each agent labels each example in a test set. While useful for offline audits, this approach is costly and infeasible when running multiple models, soliciting human expert judgments, or waiting for ground truth. Many scenarios require that models be deployed and predictive tasks allocated sequentially, necessitating that estimation of model performance and identifying the best-performing model be performed in concert with making actual predictions on real cases. Motivated by such scenarios, in this paper, we focus on learning online selection protocols for such situations, balancing classification accuracy, fairness, and model querying cost, making sequential choices and learning from observed outcomes as they are observed.

As a motivating example for our work, consider a scenario where an insurance company wishes to select a large language model (LLM) to classify customer claims into pre-defined categories. Initially, the company has access to a number of different LLMs and perhaps a single human worker to process the claims, but for operational and cost reasons would like to converge on using only a single agent in the longer-term. As a second example, a hospital may wish to evaluate the effectiveness of re-admission prediction models: if released, will a patient be re-admitted within 30 days? The hospital might have 3 internal models available for evaluation, say a simple linear model, a tree-based model, and a deep learning model. It wishes to select a single model to use going forward for operational and legal reasons. For each scenario, the insurance company or the hospital face the same question: is there a better approach to finding the best model in an online setting, beyond randomly or deterministically assigning claims or patients to models? This general model selection problem is common across application scenarios, where the models (or agents) being selected could include ML models and/or human experts.

In this context, we formalize fair online model selection as a multi-armed bandit problem where each arm corresponds to an agent (a pre-trained classifier or a human expert), and the learner selects an arm for every instance it observes from the sequential stream, observing only that agent’s prediction, and the eventual true label when available. Fairness is encoded through subgroup statistics (for example, subgroup positive-rate gaps) and combined with accuracy into a utility so the selector explicitly trades off these objectives. A Bayesian framework is natural here, as maintaining conjugate posteriors lets the learner represent uncertainty, update beliefs incrementally, and use posterior-sampling to balance exploration and exploitation of the different agents. The approach organically allows for delayed reward, per-arm costs, missing observations, and introducing new agents at any time.

Our primary contributions are the following:

• We introduce the problem of online classifier selection in the context of balancing prediction accuracy, fairness, and querying cost. We formulate this within a multi-arm bandit (MAB) framework.

• We propose a Thompson sampling approach and develop new theoretical regret bounds that show how differences in sensitive attribute subgroup proportions directly affect convergence rates.

• We conduct a series of experiments across multiple real-world classification problems, including classifiers, LLMs, and human agents, and demonstrate significant superiority in efficiency of Thompson sampling (and other MAB approaches) over non-bandit methods such as round-robin querying.

For research reproducibility, we will make all our code and datasets publicly available on GitHub for the final version of our paper.

## 2 Related Work

Relevant prior research can broadly be grouped into two main categories, each of which we discuss below: (i) work in algorithmic fairness (largely focusing on offline settings), (ii) work on online model selection (distinct from fairness). In this context, a distinguishing novel feature of our work is that we investigate the intersection of these themes, namely, online model selection in the setting of algorithmic fairness.

Evaluating the Fairness of Prediction Models There has been a large body of work on the development of “inprocess" training algorithms that leverage fairness metrics to learn fair models (e.g., Pessach and Shmueli [2022]). However, less attention has been paid to the problem of statistical assessment of how fair pretrained models are, for example models that are being deployed “in the wild." In the batch setting, there has been some work on developing general approaches for statistical assessment of the fairness properties of a pre-trained model, emphasizing the need for uncertainty quantification in assessing if the model is fair or not [McNair, 2018, Ji et al., 2021, Miller et al., 2021, Barrainkua et al., 2024, Herlihy et al., 2024, Yin et al., 2026]. In the sequential online setting, Chugg et al. [2023] motivate the importance of statistical auditing of the fairness properties of a deployed prediction model and develop a family of sequential hypothesis testing approaches for this problem. Similarly, Maneriker et al. [2023] develop probabilistic bounding techniques for online monitoring of the fairness properties of a prediction model in a sample-efficient manner.

Our work differs from that described above in that we address the problem of sequential selection from a set of prediction agents (rather than monitoring a single agent) with the goal of identifying the agent that provides the best trade-off between accuracy and fairness (rather than establishing if a particular agent meets the definition of “fair" or not).

Fair Bandit Algorithms There has also been work on fair bandit algorithms that encourage fair selection of arms/contexts. Schumann et al. [2022] consider a setting in which the arms are partitioned into two or more sensitive groups, and the goal is to balance the arms pulled from both groups. Other examples include individual fairness in contextual bandits [Joseph et al., 2016], where lower-merit actions must not be chosen more than high-merit ones; calibrated fairness that matches selection probabilities to an agent’s chance of being best [Liu et al., 2017], with cumulative merit accounting and exposure fairness extensions [Wang et al., 2021]; unknown-metric individual fairness learned from weak regulator feedback in linear contextual bandits [Gillen et al., 2018]; and minimum pull-rate guarantees that ensure per-arm exposure for MABs [Chen et al., 2020, Patil et al., 2021]. These methods encourage fair arms selection but do not address fair predictions across sensitive groups as we do in our work. At a high level, the closest work to our setting is Huang et al. [2022], though it addresses user-side fairness exclusively within recommendation systems. Our approach provides regret bounds that explicitly incorporate sensitive subgroup probabilities and employs a Bayesian framework.

Another related line of work on fair policy algorithms [Jabbari et al., 2017, Wen et al., 2021, Puranik et al., 2022, Yin et al., 2023], studies fairness in reinforcement learning and Markov decision processes by accounting for long-term effects of decisions. However, these methods focus on policy fairness rather than our focus of predictive fairness across sensitive groups.

Online Model Selection with Pretrained Models Another body of research relevant to our approach is prior work on online model selection from a set of pretrained models $( \mathrm { e . g . }$ ., classifiers or LLMs). The focus of this work is often on the design of efficient sequential algorithms to select the models with the best performance (according to some loss metric) in a streaming online setting. For example, Hu et al. [2025] address the problem of online selection of text-based generative models, conditioned on different prompts, using a contextual-bandit framework with kernel-based feature representations and an upper-confidence-bound (UCB) bandit algorithm. Xia et al. [2024] also use a UCB bandit approach for the specific problem of selecting the best LLM in the context of non-stationarity, where the performance of models (LLMs) may be increasing over time. Wei et al. [2025] present an LLM routing algorithm trained online with bandits. Dai et al. [2024] develop combinatorial cost-aware bandit algorithms for selecting optimal combinations of multiple LLMs in the context of online user queries. The works of Karimi et al. [2021], Kay et al. [2025] and Liu et al. [2024] address the problem of online model selection from a different perspective, namely how to minimize the number of label queries needed to find the best-performing model. In our work, we focus on a different variant of model selection to the settings in the papers described above, in that we add a fairness dimension to the model selection problem, using a general MAB setting with querying costs and reward delays, with theoretical regret bounds to support our approach.

## 3 Problem Statement

Let $\mathcal { A } = \{ 1 , \cdots , N \}$ be an index set of agents. Given an input $X$ each agent a can produce a prediction of a class label, ${ \hat { y } } _ { a } .$ Let $c _ { a } \geq 0$ be the fixed deterministic query cost incurred to obtain a prediction label from agent a. At each round, $t = 1 , \cdots , T$ , a triplet $( X _ { t } , S _ { t } , Y _ { t } )$ is drawn IID from an unknown probability distribution over $\breve { \mathscr { X } } \times \mathscr { S } \times \mathscr { Y }$ where X is the feature space, S is the set of sensitive attributes, and Y is the set of possible outcomes. For now, we focus on two subgroups and binary classification, i.e., ${ \mathcal { S } } = \{ 0 , 1 \}$ and $\mathcal { V } = \{ 0 , 1 \}$ –we show that the methods and theory we develop can be extended straightforwardly to more than two subgroups and to multi-class problems using appropriate definitions of fairness [Shui et al., 2022, Xian et al., 2023]. We note that the sensitive attributes are distributed as $\mathbb { P } ( S _ { t } = 0 ) = \pi _ { 0 }$ and $\mathbb { P } ( S _ { t } = 1 ) = \pi _ { 1 }$ where $\pi _ { 0 } + \pi _ { 1 } = 1$ and $\pi _ { \operatorname* { m i n } } = \operatorname* { m i n } \{ \pi _ { 0 } , \pi _ { 1 } \} > 0$ . We assume that this does not change over t.

We formulate online selection of agents as a multi-armed bandit problem where each arm represents an ML classifier or human expert, and the goal is to identify the agent that optimally balances classification error, fairness disparity, and cost.

Formally, at round t, the policy learner observes both the classification features $X _ { t }$ and the sensitive attribute value $S _ { t }$ The learner must choose an arm $A _ { t }$ from the set of available arms at time $t , A _ { t }$ , informed by the history of past actions and outcomes. The prediction $\hat { y } _ { A _ { t } }$ for the selected arm is then observed with querying cost $c _ { A _ { t } }$ . Although features are observed, the learner’s selection policy is non-contextual.

The true label $Y _ { t }$ is revealed at time $t + L _ { t }$ where $L _ { t } \in \mathbb { Z } _ { > 0 }$ is assumed to be an IID random variable representing a nonnegative integer delay, reflecting real-world scenarios (such as medical image classification, loan approval, etc) where the true $Y$ might not be known until some time in the future after the prediction. This type of delayed reward has been analyzed in the past in the MAB literature, resulting in regret bounds that scale favorably over time horizons, subject to reasonable conditions on delay distributions [Wu and Wager, 2022].

The accuracy of each arm a (measured marginally over $( X , Y ) )$ is

$$
\rho _ { a } = \mathbb { P } \big ( \hat { y } _ { a } ( X ) = Y \big ) .
$$

The highest accuracy (over all arms/agents) is $\rho _ { * } = \operatorname* { m a x } _ { a } \rho _ { a }$

Fairness metrics can be expressed as functions of subgroup and label-specific conditional probabilities $\mu _ { a , s , y }$ . The resulting fairness gap (usually a difference or ratio) is a function $g _ { a } = f ( \mu _ { a , s , y } )$ of these conditional probabilities. Let the fairest arm be arg $\operatorname* { m i n } _ { a \in \mathcal { A } } g _ { a }$ and denote $g _ { * } = \operatorname* { m i n } _ { a \in \mathcal { A } } g _ { a }$

As an example, for demographic-parity (DP) gap, we may define the subgroup-positive rate $\mu _ { a , s } = \mathbb { P } \big ( \hat { y } _ { a } ( X ) = 1 \mid$ $S = s )$ for each arm $a \in { \mathcal { A } }$ and subgroup $s \in \{ 0 , 1 \}$ . The resulting DP gap of arm a is $g _ { a } = | \mu _ { a , 0 } - \mu _ { a , 1 } |$

Given the framework above, the utility per agent trades off accuracy, fairness, and cost and is defined as

$$
U _ { a } ( \lambda ) : = \rho _ { a } - \lambda g _ { a } - c _ { a } , \qquad \lambda \ge 0 .\tag{1}
$$

Here, λ is a user-defined (constant) weight that a decision-maker assigns to indicate the relative weight of the fairness gap compared to classification accuracy, with larger positive values of λ indicating greater weight on fairness over accuracy. In the context of algorithmic fairness, the weight λ represents a normative policy preference (how much accuracy the stakeholder is willing to sacrifice for fairness), rather than a latent statistical parameter. This functional form is similar to that used in learning fair agents [Kamishima et al., 2011, Menon and Williamson, 2018]. The querying cost $c _ { a }$ is assumed to be expressed in the same units as accuracy $\rho _ { a } , \mathbf { e . g . }$ , in some utility units. Our formulation follows the standard decision theoretic framework (Murphy [2012, pp. 176–180], Bishop [2006, p. 41]) where accuracy is the expected utility of making a correct decision and the cost is normalized relative to the value or utility of a correct prediction.

Let $U _ { * } ( \lambda ) = \operatorname* { m a x } _ { a } U _ { a } ( \lambda )$ , i.e., the utility of the optimal arm $a ^ { * }$ . The goal of the learner is to select $A _ { t }$ such that the selected arm’s utility is as close as possible to the optimum $U _ { * }$ up to time $T .$ This performance measure is formalized as the cumulative regret of the learner - a quantity that the learner aims to minimize

$$
R _ { \lambda } ( T ) = \sum _ { t = 1 } ^ { T } \big ( U _ { * } ( \lambda ) - U _ { A _ { t } } ( \lambda ) \big ) .\tag{2}
$$

This decomposes linearly into accuracy, fairness, and cost-regret terms:

$$
R _ { \lambda } ( T ) = \underbrace { \sum _ { t = 1 } ^ { T } \left( \rho _ { a ^ { * } } - \rho _ { A _ { t } } \right) } _ { \mathrm { a c c u r a c y r e g r e t } } + \lambda \underbrace { \sum _ { t = 1 } ^ { T } \left( g _ { A _ { t } } - g _ { a ^ { * } } \right) } _ { \mathrm { f a i r n e s s r e g r e t } } + \underbrace { \sum _ { t = 1 } ^ { T } \left( c _ { A _ { t } } - c _ { a ^ { * } } \right) } _ { \mathrm { c o s t r e g r e t } } .
$$

For clarity, we note that the accuracy term in this decomposition is formulated as the performance deficit relative to the optimal arm $( \mathrm { i . e . } \ \rho _ { a ^ { \ast } } - \rho _ { A _ { t } } )$ . This definition effectively flips the sign relative to the utility function to ensure that lower values indicate better performance, consistent with the standard definition of regret. For fairness and cost, the decomposition remains $g _ { A _ { t } } - g _ { a ^ { * } }$ and $c _ { A _ { t } } - c _ { a ^ { * } }$ respectively.

## 4 Technical Approach: Bayesian Bandits

Algorithm 1 Thompson Sampling for Online Selection   
Require: Arms ${ \mathcal { A } } ,$ horizon $T ,$ , trade-off $\lambda \geq 0 ,$ , per-arm costs $c _ { a } \geq 0 ,$ priors   
$\tilde { \mu } _ { a , s } \sim \mathrm { B e t a } ( \alpha _ { a , s } , \beta _ { a , s } )$ for $( a , s ) \in \mathcal { A } \overset { \vartriangle } { \times } \{ 0 , 1 \} ^ { 1 } .$   
$\tilde { \rho } _ { a } \sim \mathrm { B e t a } ( \alpha _ { a } , \beta _ { a } )$ for $a \in A .$   
1: for $t = 1 , \dots , T$ do   
2: for all $a \in \mathcal { A } _ { t }$ do   
3: Draw $\tilde { \mu } _ { a , 0 } \sim \mathrm { B e t a } ( \alpha _ { a , 0 } , \beta _ { a , 0 } ) , \ \tilde { \mu } _ { a , 1 } \sim \mathrm { B e t a } ( \alpha _ { a , 1 } , \beta _ { a , 1 } )$   
4: Draw $\tilde { \rho } _ { a } \sim \mathrm { B e t a } ( \alpha _ { \underline { { a } } } , \beta _ { a } ) .$   
5: $\tilde { g } _ { a } \gets | \tilde { \mu } _ { a , 0 } - \tilde { \mu } _ { a , 1 } | ^ { 1 }$   
6: $U _ { a } \gets \tilde { \rho } _ { a } - \lambda \tilde { g } _ { a } - c _ { a } .$   
7: Select $A _ { t } \in$ arg max $\ L _ { a \in \mathcal { A } _ { t } } U _ { a }$ (ties arbitrary).   
8: Observe $S _ { t } ; \mathrm { s e t } r _ { \mathrm { f a i r } } = \mathbb { 1 } \bar { \{ y _ { A _ { t } } ( X _ { t } ) = 1 \} } ^ { 1 }$   
9: Update fairness posterior: $\alpha _ { A _ { t } , S _ { t } }  \alpha _ { A _ { t } , S _ { t } } + r _ { \mathrm { f a i r } } , \beta _ { A _ { t } , S _ { t } }  \beta _ { A _ { t } , S _ { t } } + ( 1 - r _ { \mathrm { f a i r } } ) .$   
10: for all $\tau \leq t$ whose label Y arrives at round t do   
11: Set $r _ { \mathrm { a c c } } = \mathbb { 1 } \{ \hat { y } _ { A _ { \tau } } ( X _ { \tau } ) = Y _ { \tau } \}$   
12: Update accuracy posterior: $\dot { \alpha _ { A _ { \tau } } }  \alpha _ { A _ { \tau } } + r _ { \mathrm { a c c } } , \beta _ { A _ { \tau } }  \beta _ { A _ { \tau } } + ( 1 - r _ { \mathrm { a c c } } ) .$  
<sup>1</sup>For DP gap, we maintain subgroup positive-rate. $\scriptstyle { \tilde { g } } _ { a }$ and $r _ { \mathrm { f a i r } }$ follow.

Formulating online selection as an MAB problem allows us to use any standard MAB algorithm, including but not limited to UCB-style optimistic methods [Auer et al., 2002], (ϵ-)greedy heuristics [Sutton et al., 1998], and Thompson Sampling [Russo et al., 2018, Russo and Van Roy, 2014]. In this work, we focus on Thompson Sampling (TS): the Bayesian posterior view makes it straightforward to maintain and systematically update uncertainty beliefs. The TS approach handles feedback (immediate or delayed) naturally via conjugate updates, and posterior sampling results in an intuitive mechanism that naturally balances exploration and exploitation. The seminal work of Chapelle and Li [2011] also demonstrated highly competitive empirical results for TS in traditional MAB settings, such as online advertisement and recommendation systems.

Our work differs from standard work on MABs in that each arm induces two latent “sub-arms” corresponding to sensitive subgroups S. Algorithmically, we must maintain and update separate posteriors for each subgroup as well as marginal accuracy estimates. Analytically, this increases the number of parameters to learn and introduces a dependence on subgroup prevalence: regret bounds therefore decompose into accuracy and fairness components and depend on the smallest subgroup probability (and on label frequencies for conditional fairness notions); our theoretical guarantees presented in Section 5 reflect this structure. In Algorithm 1, we formalize Thompson Sampling for online selection of classifiers, balancing accuracy, fairness, and query cost with potential delays in true label observation. We consider a broader class of model selection tasks and frameworks:

Other fairness metrics In addition to TS for DP-gap, we can use other fairness metrics for this algorithm. For metrics such as Equalized Odds and Equality of Opportunity, we would maintain posterior beliefs $\tilde { \mu } _ { a , s , y } \sim \mathrm { B e t a } ( \alpha _ { a , s , y } , \beta _ { a , s , y } )$ for $( a , s , \bar { y } ) \in \mathcal { A } \times \{ 0 , 1 \} \times \{ \bar { 0 } , 1 \}$ that include the true label Y . Both $\tilde { g } _ { a }$ and $r _ { \mathrm { f a i t } }$ change accordingly, and the fairness update waits for $Y$

Multi-class classification $( | \mathcal { V } | = C > 2 )$ and Multi-subgroup sensitive attribute $( | S | = M > 2 )$ We continue to use Beta-Bernoulli posteriors for the accuracy estimate since the reward is still binary (assuming symmetric utilities for different misclassifications). To capture disparities in class predictions (e.g., unequal prediction rates across classes conditioned on sensitive attributes), we replace the subgroup Beta posteriors with M Dirichlet posteriors (under a Dirichlet-Multinomial model), parameterizing the probability vector over the C classes for each subgroup. We sample probability vectors $\tilde { \mu } _ { a , s }$ for all subgroups and use the maximum/average pairwise $L _ { 1 }$ distance computed on these samples to capture disparity [Xian et al., 2023, Barocas et al., 2023, Agarwal et al., 2018].

Real-valued reward The framework extends to bounded real-valued rewards, such as the Brier score of a probabilistic classifier or LLM evaluation scores [Liang et al., 2023]. Suppose arm a outputs a score $p _ { a } \in [ 0 , 1 ]$ instead of a label. With reward $1 - ( Y - p _ { a } ) ^ { 2 } , \rho _ { a }$ is one minus the arm’s Brier score, and the fairness gap $g _ { a }$ is the DP gap of the subgroup mean scores $\mu _ { a , s } = \mathbb { E } [ p _ { a } \mid S = s ]$ . For 0/1 scores these reduce to accuracy and the DP gap. If the scores are rounded to a grid $\{ 0 , 1 / K , \ldots , 1 \}$ , scores and rewards take finitely many values, and each Beta posterior is replaced by a Dirichlet posterior over these values (conjugate to the categorical likelihood). With one pseudo-count at 0 and one at 1, the prior is Beta(1, 1) for binary data. TS samples Dirichlet weights w and uses the weighted mean $\sum _ { k } w _ { k } v _ { k }$ of the values $v _ { k }$ as $\tilde { \rho } _ { a }$ or $\tilde { \mu } _ { a , s }$ . Up to the prior, this is non-parametric TS [Riou and Honda, 2020] with tied observations grouped.

## 5 Theoretical Results

We investigate the regret bounds of our Thompson sampling algorithm (Algorithm 1). Let $\Delta _ { a } = U _ { * } ( \lambda ) - U _ { a } ( \lambda )$ be the utility gap and $a ^ { * } \in \mathrm { a r g m a x } _ { a } U _ { a } ( \lambda )$ . Recall that the cumulative regret is defined as in Equation (2).

Theorem 5.1 (Regret Bound of Thompson Sampling with Fairness). For the Thompson sampling described in Section 4, with no delays and arbitrary (deterministic) querying costs,for a universal constant $C > 0$ , we have the gap-dependent regret:

$$
\begin{array} { r l r } {  { \mathbb { E } [ R _ { \lambda } ( T ) ] \le C \sum _ { a : \Delta _ { a } ( \lambda ) > 0 } \frac { 1 } { \Delta _ { a } ( \lambda ) } \Bigg [ \log T + \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } \log T } } \\ & { } & { + ( \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } + \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } ) ( \log T ) ^ { 7 / 2 } \Bigg ] . } \end{array}\tag{3}
$$

Moreover, the gap-free bound is

$$
\begin{array} { r } { \mathbb { E } [ R _ { \lambda } ( T ) ] \leq O \Bigg ( \sqrt { N T \log T } + \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } \sqrt { N T \log T } } \\ { + \sqrt { \bigg ( \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } + \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } \bigg ) T ( \log T ) ^ { 7 / 2 } } \Bigg ) . } \end{array}\tag{4}
$$

When $N \geq ( \log T ) ^ { 5 / 2 }$ , the third term of (4), the additional optimal-arm anti-concentration cost, is dominated by the first two and the gap-free bound reduces to $O \big ( \sqrt { N T }$ log $\begin{array} { r } { \overline { { T } } + \frac { \lambda } { \sqrt { \pi _ { \mathrm { m i n } } } } \sqrt { N T \log T } ) } \end{array}$ .

Without fairness, Thompson sampling has the bound $O ( { \sqrt { N T \log T } } )$ [Agrawal and Goyal, 2017]. Our bounds add two terms to this accuracy cost: the fairness-estimation cost and the optimal-arm anti-concentration cost. The $\pi _ { \mathrm { m i n } }$ factors quantify the statistical cost of estimating the demographic parity gap across subgroups: rarer groups encourage more exploration. However, incorporating fairness preserves the canonical gap-dependent and gap-free regret rates in T. The λ factors reflect the sensitivity of the utility to fairness-gap estimation error: heavier fairness weighting amplifies the impact of subgroup-rate uncertainty on arm misordering. The optimal-arm anti-concentration cost arises because noisy subgroup draws can inflate the optimal arm’s sampled fairness gap even when its accuracy draw is optimistic, which delays its selection. This cost vanishes when $\lambda = \bar { 0 }$ and does not grow with $N$

Here we provide a sketch of the proof: full proof in Appendix A. We follow the regret bound analysis of Agrawal and Goyal [2017] to the accuracy–fairness-cost utility. For any suboptimal arm, we consider rounds in which (I) the optimal arm’s sampled utility is too low, (II) the suboptimal arm’s posterior draw is overly optimistic, or (III) its posterior-mean utility is overly optimistic. The selected arm does not depend on the group of the incoming sample, each observation of an arm comes from group s with probability $\pi _ { s }$ . Thus, the first n observations of any arm contain about $\pi _ { s } n$ observations from group s, and the smallest group contributes about $\pi _ { \mathrm { m i n } } n$ . Concentration of both accuracy and subgroup rates then controls (II) and (III). Known querying costs enter the utility thresholds and gaps as fixed offsets.

Case (I) requires an additional argument. The exchange step of Agrawal and Goyal [2017] still applies, but their scalar Beta-Binomial anti-concentration time bound does not carry over as the optimal arm’s sampled utility combines an accuracy draw with the absolute difference of two subgroup draws. Moreover, the accuracy and subgroup estimates of the optimal arm can be dependent because they are computed from the same observation. Thus, we bound this joint anti-concentration cost using a positive utility margin. We then group arms by utility gap so that anti-concentration is charged once per group in the gap-free bound. $\mathbf { A } \mathbf { t } \ \lambda = 0$ , the joint anti-concentration step disappears and the scalar accuracy argument reduces to the standard rate.

Theorem 5.2 (Regret Bound of Thompson Sampling with Fairness under Delayed Reward). Consider Algorithm 1 under the assumptions ofTheorem 5.1. For each arm, assume that label delays are IID. Fairness posteriors are updated immediately, whereas accuracy posteriors are updated when labels arrive. Choose $q _ { a } \in ( 0 , 1 ]$ such that $d _ { a } ( q _ { a } ) < \infty ,$ and let $q _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { a } q _ { a }$ . Then, the gap-free regret bound becomes:

$$
\begin{array} { r } { \mathbb { E } [ R _ { \lambda } ( T ) ] \le O \Bigg ( \sqrt { \frac { N T \log T } { q _ { \mathrm { m i n } } } } + \frac { \lambda } { \sqrt { \pi _ { \mathrm { m i n } } } } \sqrt { N T \log T } } \\ { + \left( 1 + \lambda \right) \displaystyle \sum _ { a = 1 } ^ { N } d _ { a } ( q _ { a } ) + \sqrt { T H _ { D } ( T ) } \Bigg ) , } \end{array}
$$

where $H _ { D } ( T )$ summarizes optimal-arm anti-concentration caused by pessimistic utility draws under delayedfeedback.   
Itsformal definition is given in Appendix A.3.

As accuracy updates require the true labels, their estimation cost is inflated by $q _ { \mathrm { m i n } } ^ { - 1 / 2 }$ . Demographic parity feedback is available immediately, so the fairness estimation term scales as $\lambda / \sqrt { \pi _ { \operatorname* { m i n } } }$ without an additional delay factor. The quantile argument also produces the additive contribution $\begin{array} { r } { ( 1 + \lambda ) \sum _ { a = 1 } ^ { N } d _ { a } ( q _ { a } ) } \end{array}$ . The remaining term, $\sqrt { T H _ { D } ( T ) }$ covers the optimal arm’s anti-concentration, both the accuracy-based part and the part due to joint accuracy-fairness uncertainty. Proof sketch (full proof in Appendix A.3). We combine the delay-quantile counting of Wu and Wager [2022] with the posterior concentration and exchange arguments of Theorem 5.1. Concentration controls overly optimistic draws of a suboptimal arm, using the arrived labels for accuracy and the predictions observed at selection for fairness. The exchange argument reduces the remaining pulls to the anti-concentration cost of the optimal arm, $H _ { D } ( T )$ which we charge once per utility-gap group, and the choice of the gap cutoff yields the stated bound. Without delays, the anti-concentration bound in Theorem 5.1 controls $H _ { D } ( T )$ and recovers its regret rate.

## 6 Experiments

## 6.1 Datasets

We conduct experiments on real-world datasets involving classification problems that have a sensitive attribute (see Table 1). For ML Classifiers as Arms we use 4 tabular datasets where the arms are pre-trained classifiers. For each dataset we train models on 60% of the data and use the remaining 40% to create an online stream. On every dataset we train the same 18 classifiers. They range from standard models (e.g., Gradient Boosted Trees and Logistic Regression), to models trained with a fairness penalty and deliberately biased models; most do not use the sensitive attribute (detail in Appendix B.1). The first dataset is Bank Marketing (Bank) [Moro et al., 2014], where the classification goal is to predict if the client will subscribe to a bank term deposit, with 19 features and where marital status is the sensitive attribute. For the Adult dataset [Becker and Kohavi, 1996], the goal is to predict whether a person’s income exceeds \$50K, with 11 features and where race (white vs. non-white) is the sensitive attribute. The Credit dataset [Yeh and Lien, 2009] is a credit default classification task, with 22 features, and customer sex is the sensitive attribute. For a Multi-subgroup Sensitive Attribute $( M > 2 )$ we use the updated (ACSIncome) [Ding et al., 2021], with the same prediction task as Adult, using 9 features. The sensitive attribute is race/ethnicity with five groups and the DP gap is the largest gap between any two groups. Additional experiments with the Coupon dataset [Wang et al., 2017] and COMPAS [Larson et al., 2016] are included in Appendix B.1.

<table><tr><td>Dataset</td><td>Type of Arms (N)</td><td>C</td><td>Sensitive Attribute (M)</td><td>πmin</td></tr><tr><td>Bank</td><td>ML classifiers (18)</td><td>2</td><td>Marital Status (2)</td><td>0.396</td></tr><tr><td>Adult</td><td>ML classifiers (18)</td><td>2</td><td>Race (2)</td><td>0.146</td></tr><tr><td>Credit</td><td>ML classifiers (18)</td><td>2</td><td>Sex (2)</td><td>0.392</td></tr><tr><td>ACSIncome</td><td>ML classifiers (18)</td><td>2</td><td>Race (5)</td><td>0.036</td></tr><tr><td>Toxicity</td><td>LLMs (56)</td><td>2</td><td>Race (2)</td><td>0.500</td></tr><tr><td>Bios</td><td>LLMs (16)</td><td>28</td><td>Gender (2)</td><td>0.461</td></tr><tr><td>Twitter</td><td>Human Experts (10)</td><td>2</td><td>Dialect (2)</td><td>0.494</td></tr></table>

Table 1: Datasets: N arms, C classes and M subgroups

For LLMs as Arms we use the CivilComments Toxicity detection dataset (Toxicity) [Borkan et al., 2019]. For this dataset, the arms correspond to 56 LLMs accessed through API queried on 2000 prompts, where the classification task is whether a passage is toxic or not [Liang et al., 2023]. Race is the sensitive attribute. We include a Multi-class Classification $\left( C > 2 \right)$ setup with LLMs, using the Bias in Bios (Bios) dataset [De-Arteaga et al., 2019], where the goal is to predict a person’s occupation (28 classes) from their online biography. Gender is the sensitive attribute. The arms are 16 open instruction-tuned LLMs (0.36B–9B parameters) queried zero-shot on 30,000 biographies.

For Humans as Arms we use the TwitterAAE dataset (Twitter) [Keswani et al., 2021] with Tweets annotated using Mechanical Turk for whether they contain hate-speech or not [Davidson et al., 2017] The sensitive attribute is whether the post contains African-American English [Blodgett et al., 2017]. The original dataset contained annotations from 170 experts, each covering only 4% of the data. To reduce sparsity in the response, we grouped experts into 10 arms by randomly combining 18 experts per arm and using majority voting, resulting in each arm classifying ≈60% of the examples.

## 6.2 Baseline Methods

Round-Robin The learner cycles deterministically through available arms, guaranteeing even coverage and predictable scheduling; it yields unbiased exploration but no exploitation of observed rewards. It is equivalent, in expectation, to a random policy.

ϵ-Greedy Under the same Bayesian setup as Thompson Sampling, we consider a greedy strategy that selects the arm with the largest posterior mean. For the subgroup posterior estimate (Algorithm 1, Step 3) $\tilde { \mu } _ { a , s } = \mathbb { E } [ \mathrm { B e t a } ( \alpha _ { a , s } , \beta _ { a , s } ) ]$ for $( a , s ) \in \bar { \mathcal { A } } \times \{ 0 , 1 \}$ , and for accuracy (Algorithm 1, Step 4) $\tilde { \rho } _ { a } = \mathbb E [ \mathrm { { \bar { B } e t a } } ( \alpha _ { a } , \beta _ { a } ) ]$ ] for $a \in A .$ . At each round, with probability 1 − ϵ the arm with the largest mean is selected, and (to encourage exploration) with probability ϵ an arm is selected uniformly at random.

We compare against UCB1 where the learner selects the arm with the largest optimistic utility [Auer et al., 2002] and Bayes-UCB [Kaufmann et al., 2012], where, under the same Bayesian setup as TS, posterior samples are replaced with posterior quantiles at level 1 − 1/t. Additional implementation details in Appendix B.2.

## 6.3 Methodology

We average results over 10,000 runs. Each run is a stream of $T = 5 0 { , } 0 0 0$ instances sampled with replacement from the test dataset (results with fewer rounds are in Appendix B.3), and all algorithms and values of λ see the same stream. We report the mean regret per example, $R _ { \lambda } ( \bar { T } ) / T$ in %, computed from the arms’ true accuracies and gaps on the test dataset. On Twitter, regret is measured against the best available arm in each round. Bayesian methods are always initialized with uninformative priors Beta(1, 1) or Dirichlet $( 1 , \ldots , 1 )$ .For the experiments, the fairness gap is defined as the demographic parity gap (for Bios, half the $L _ { 1 }$ distance between the subgroups’ class distributions; for ACSIncome, the largest gap between any two subgroups) and traded off with the utility score (Equation 1). λ = 0 represents accuracy-only selection $( U _ { a } = \rho _ { a } - c _ { a } )$ and $\lambda = \infty$ fairness-only selection $( U _ { a } = - g _ { a } - c _ { a } )$ . We use 29–38 values of λ per dataset (Table 3), log-spaced and denser near the values at which the best arm changes. For all results in

the main paper, the query cost is the same for all arms (so the cost term drops out of the utility) and the delay $L _ { t }$ is zero.   
Results with non-zero costs and with delays are in Appendices B.4 and B.5.

## 6.4 Results

In Table 2 we compare Thompson Sampling and the 5 baselines across the 7 datasets in Table 1 at the two “endpoint” special cases of (a) λ = 0 (where regret only depends on model accuracy) and (b) $\lambda = \infty$ (where regret only depends on fairness). Thompson sampling has the smallest regret in 12 of the 14 cases, and in the other two it is within 0.1 of the best.
<table><tr><td>Algorithm</td><td>Twitter</td><td></td><td></td><td>Bank Adult Credit Toxicity</td><td></td><td>Bios</td><td>ACSIncome</td></tr><tr><td colspan="6">(a) Accuracy (λ = 0)</td><td></td><td></td></tr><tr><td>Round-Robin UCB1 Bayes-UCB</td><td>4.141 1.924 0.864</td><td>0.819 0.563</td><td>3.4943.521 1.544 0.679</td><td>3.155 0.975 0.615</td><td>16.73 4.930 2.229</td><td>17.80 3.344 0.669</td><td>7.115 2.476 0.845</td></tr><tr><td>€-Greedy (0) €-Greedy (0.1) TS</td><td>1.200 0.915 0.770</td><td>0.836 0.634 0.507</td><td>2.505 0.552 0.548</td><td>1.109 0.530 0.532</td><td>4.151 2.406 1.948</td><td>8.327 2.236 0.523</td><td>5.255 1.001 0.676</td></tr><tr><td colspan="6">(b) Fairness  $( \lambda = \infty )$ </td><td></td><td></td></tr><tr><td>Round-Robin</td><td>3.706</td><td>5.612</td><td>9.266</td><td>4.956</td><td>12.89</td><td>10.44</td><td>21.37</td></tr><tr><td>UCB1</td><td>3.739</td><td>2.344</td><td>6.472</td><td>1.612</td><td>11.80</td><td>10.44</td><td></td></tr><tr><td></td><td></td><td>1.627</td><td></td><td>0.764</td><td></td><td></td><td>15.12</td></tr><tr><td>Bayes-UCB</td><td>2.619</td><td></td><td>5.181</td><td></td><td>6.902</td><td>7.820</td><td>3.136</td></tr><tr><td>€-Greedy (0)</td><td>2.010</td><td></td><td>1.6245.335</td><td>0.815</td><td>1.874</td><td>2.860</td><td>8.378</td></tr><tr><td>€-Greedy (0.1)</td><td>1.715</td><td></td><td>1.1232.435</td><td>0.653</td><td>1.453</td><td>2.039</td><td>5.151</td></tr><tr><td>TS</td><td>1.367</td><td></td><td>1.023 2.416</td><td>0.179</td><td>0.548</td><td>2.109</td><td>1.495</td></tr></table>

Table 2: Regret per example (%) after $T = 5 0 \mathrm { k }$ rounds for each algorithm (rows) and dataset (columns), for (a) accuracy $( \lambda = 0 )$ and (b) fairness $( \lambda = \infty )$ , averaged over 10,000 runs (the endpoints of the curves in Figure 1). Minimum regret bolded.

Figure 1 shows the trade-off for λ values between these two endpoints, indicating the cumulative accuracy regret (x-axis) and cumulative fairness regret (y-axis) for that λ value as a percentage of total rounds, averaged over 10,000 runs each of $T = 5 0 \mathrm { k }$ rounds. The endpoints of each curve are the values in Table 2. Additional details are provided in Appendix B.1.

The non-MAB method (Round-Robin) is clearly not competitive with the MAB methods. In fact for the Bank, Credit and Toxicity datasets, Round-Robin is off-the-scale to the upper-right. Among the MAB methods, Thompson Sampling (TS) is closest to the oracle frontier on every dataset (in some cases by a large margin), with occasional exceptions for some values of λ (which is not surprising given that none of these methods are provably dominant empirically for all MAB problems and their relative performance is dataset-dependent [Russo et al., 2018]). The UCB methods are far from the frontier at large λ, because their lower bound on the DP gap stays at zero until the subgroup confidence intervals separate. The overall picture from Figure 1 is that (i) MAB methods significantly outperform non-MAB methods, and (ii) TS is more reliable in achieving good solutions than the ϵ-Greedy and UCB methods.

Table 3 extends the comparison of Table 2 to every value of λ. It summarizes Figure 1 in tabular form how often each algorithm has the best point. TS has the lowest regret at all values on four datasets and at all but one on Credit. On ACSIncome, Bayes-UCB is lower where two arms have nearly equal utility. On Bios, ϵ-Greedy $( \epsilon = 0 . 1 )$ is lower at large λ: with 28 classes, the flat Dirichlet prior inflates the sampled gap of a rarely selected arm, so TS can under-explore the fairest LLM.

We report further experiments with real-valued rewards (Appendix B.7), multi-class classification on a second dataset (Appendix B.6), fewer rounds (Appendix B.3), fewer arms (Appendix B.9), delayed feedback (Appendix B.5), query costs (Appendix B.4) and smaller values of $\pi _ { \mathrm { m i n } }$ (Appendix B.8). TS is consistently the best-performing method across scenarios, with the lowest regret at most values of λ. The main exceptions are short horizons $( T \le 1 0 \mathbf { k } )$ and long delays.

![](images/7690ba6a20cb5a901f58e78b8d37085a62c14b2c85d57ba23825d2a05081564f.jpg)

Figure 1: Trade-off between accuracy regret (x-axis) and fairness regret (y-axis) after $T = 5 0 \mathrm { k }$ rounds, as average regret per example (%) over 10,000 runs. Each point on a curve is one value of λ, from $\lambda = 0$ (all weight on accuracy) to $\lambda = \infty$ (all weight on fairness). Datasets (a)–(e) are binary, (f) is multi-class and (g) is multi-subgroup. Grey crosses are the individual arms and the dotted line is the oracle frontier. Round-Robin does not depend on λ and is omitted from (b), (d) and (e), where it is off-scale.
<table><tr><td>Algorithm</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Twitter Bank Adult Credit Toxicity Bios ACSIncome</td></tr><tr><td rowspan="2">Round-Robin UCB1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Bayes-UCB</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>18</td></tr><tr><td>€-Greedy (0)</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>€-Greedy (0.1)</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>38</td><td>0</td></tr><tr><td>TS</td><td>100</td><td>100</td><td>100</td><td>97</td><td>100</td><td>62</td><td>82</td></tr></table>

Table 3: % of λ values at which each algorithm has the lowest average cumulative regret after $T = 5 0 \mathrm { k }$ rounds.

## 7 Discussion and Conclusion

In terms of limitations, our theoretical proofs focus on binary classifiers, non-contextual bandits, and difference-based fairness metrics for binary sensitive attributes. Nonetheless, we believe our results represent an important step in model selection that balances accuracy, fairness, and cost, providing a basis for a variety of directions for future work in terms of theory, algorithms, and experiments. An obvious direction is model selection via contextual bandits [Chapelle and Li, 2011], where the selected arm $A _ { t }$ is dependent on the instance $X _ { t } , \mathbf { e . g . }$ ., modeling the expected reward via GLMs conditioned on features $X _ { t }$ and utilizing Laplace approximations to maintain a tractable Gaussian posterior over the parameters. This could result in policies that allocate tasks to arms that are better for specific regions of the context space. Additional potential directions for future research include: extensions to time-to-event models, multi-play algorithms to provide additional exploration of small subgroups (small $\pi _ { \mathrm { m i n } } )$ , and leveraging probabilistic information from ML models. The algorithmic template would remain largely unchanged for many of these extensions: maintain per-arm/per-subgroup posteriors for the quantities that define utility, sample those posteriors at decision time, and select based on a utility that trades-off accuracy, fairness, and cost; only the choice of conjugate family and the disparity functional differ.

## References

A. Agarwal, A. Beygelzimer, M. Dudík, J. Langford, and H. Wallach. A reductions approach to fair classification. In International conference on machine learning, pages 60–69. PMLR, 2018.

S. Agrawal and N. Goyal. Near-optimal regret bounds for Thompson sampling. Journal ofthe ACM (JACM), 64(5): 1–24, 2017.

P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47 (2):235–256, 2002.

S. Barocas, M. Hardt, and A. Narayanan. Fairness and machine learning: Limitations and opportunities. MIT press, 2023.

A. Barrainkua, P. Gordaliza, J. A. Lozano, and N. Quadrianto. Uncertainty matters: stable conclusions under unstable assessment of fairness results. In International Conference on Artificial Intelligence and Statistics, pages 1198–1206. PMLR, 2024.

B. Becker and R. Kohavi. Adult. UCI Machine Learning Repository, 1996. DOI: https://doi.org/10.24432/C5XW20.

C. M. Bishop. Pattern recognition and machine learning, volume 4. Springer, 2006.

S. L. Blodgett, J. Wei, and B. O’Connor. A dataset and classifier for recognizing social media english. In Proceedings ofthe 3rd Workshop on Noisy User-generated Text, pages 56–61, 2017.

D. Borkan, L. Dixon, J. Sorensen, N. Thain, and L. Vasserman. Nuanced metrics for measuring unintended bias with real data for text classification. In Companion Proceedings ofthe 2019 World Wide Web Conference, 2019.

O. Chapelle and L. Li. An empirical evaluation of Thompson sampling. Advances in Neural Information Processing Systems, 24, 2011.

Y. Chen, A. Cuellar, H. Luo, J. Modi, H. Nemlekar, and S. Nikolaidis. Fair contextual multi-armed bandits: Theory and experiments. In Conference on Uncertainty in Artificial Intelligence, pages 181–190. PMLR, 2020.

B. Chugg, S. Cortes-Gomez, B. Wilder, and A. Ramdas. Auditing fairness by betting. Advances in Neural Information Processing Systems, 36:6070–6091, 2023.

X. Dai, J. Li, X. Liu, A. Yu, and J. Lui. Cost-effective online multi-LLM selection with versatile reward models. arXiv preprint arXiv:2405.16587, 2024.

T. Davidson, D. Warmsley, M. Macy, and I. Weber. Automated hate speech detection and the problem of offensive language. In Proceedings ofthe International AAAI Conference on Web and Social Media (ICWSM), volume 11, pages 512–515, 2017.

M. De-Arteaga, A. Romanov, H. Wallach, J. Chayes, C. Borgs, A. Chouldechova, S. Geyik, K. Kenthapadi, and A. T. Kalai. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings of the Conference on Fairness, Accountability, and Transparency, pages 120–128, 2019.

F. Ding, M. Hardt, J. Miller, and L. Schmidt. Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34:6478–6490, 2021.

S. Gillen, C. Jung, M. Kearns, and A. Roth. Online learning with an unknown fairness metric. Advances in Neural Information Processing Systems, 31, 2018.

C. Herlihy, K. Truong, A. Chouldechova, and M. Dudík. A structured regression approach for evaluating model performance across intersectional subgroups. In The 2024 ACM Conference on Fairness, Accountability, and Transparency, pages 313–325, 2024.

X. Hu, H.-f. Leung, and F. Farnia. An online learning approach to prompt-based selection of generative models and LLMs. In Forty-second International Conference on Machine Learning, 2025.

W. Huang, K. Labille, X. Wu, D. Lee, and N. Heffernan. Achieving user-side fairness in contextual bandits. Human-Centric Intelligent Systems, 2(3):81–94, 2022.

S. Jabbari, M. Joseph, M. Kearns, J. Morgenstern, and A. Roth. Fairness in reinforcement learning. In International Conference on Machine Learning, pages 1617–1626. PMLR, 2017.

D. Ji, R. L. Logan, P. Smyth, and M. Steyvers. Active Bayesian assessment of black-box classifiers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 7935–7944, 2021.

M. Joseph, M. Kearns, J. H. Morgenstern, and A. Roth. Fairness in learning: Classic and contextual bandits. Advances in Neural Information Processing Systems, 29, 2016.

F. Kamiran and T. Calders. Data preprocessing techniques for classification without discrimination. Knowledge and information systems, 33(1):1–33, 2012.

T. Kamishima, S. Akaho, and J. Sakuma. Fairness-aware learning through regularization approach. In 2011 IEEE 11th international conference on data mining workshops, pages 643–650. IEEE, 2011.

M. R. Karimi, N. M. Gürel, B. Karlaš, J. Rausch, C. Zhang, and A. Krause. Online active model selection for pre-trained classifiers. In International Conference on Artificial Intelligence and Statistics, pages 307–315. PMLR, 2021.

E. Kaufmann, O. Cappé, and A. Garivier. On bayesian upper confidence bounds for bandit problems. In Artificial intelligence and statistics, pages 592–600. PMLR, 2012.

J. Kay, G. Van Horn, S. Maji, D. Sheldon, and S. Beery. Consensus-driven active model selection. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4594–4604. IEEE, 2025.

V. Keswani, M. Lease, and K. Kenthapadi. Towards unbiased and accurate deferral to multiple experts. In Proceedings ofthe 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 154–165, 2021.

J. Larson, J. Angwin, L. Kirchner, and S. Mattu. How we analyzed the compas recidivism algorithm, May 2016. URL https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm.

P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, B. Newman, B. Yuan, B. Yan, C. Zhang, C. A. Cosgrove, C. D. Manning, C. Re, D. Acosta-Navas, D. A. Hudson, E. Zelikman, E. Durmus, F. Ladhak, F. Rong, H. Ren, H. Yao, J. WANG, K. Santhanam, L. Orr, L. Zheng, M. Yuksekgonul, M. Suzgun, N. Kim, N. Guha, N. S. Chatterji, O. Khattab, P. Henderson, Q. Huang, R. A. Chi, S. M. Xie, S. Santurkar, S. Ganguli, T. Hashimoto, T. Icard, T. Zhang, V. Chaudhary, W. Wang, X. Li, Y. Mai, Y. Zhang, and Y. Koreeda. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=iO4LZibEqW. Featured Certification, Expert Certification.

X. Liu, F. Xia, R. Stevens, and Y. Chen. Contextual active model selection. Advances in Neural Information Processing Systems, 37:49914–49956, 2024.

Y. Liu, G. Radanovic, C. Dimitrakakis, D. Mandal, and D. C. Parkes. Calibrated fairness in bandits. arXiv preprint arXiv:1707.01875, 2017.

P. Maneriker, C. Burley, and S. Parthasarathy. Online fairness auditing through iterative refinement. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1665–1676, 2023.

D. S. McNair. Preventing disparities: Bayesian and frequentist methods for assessing fairness in machine learning decision-support models. New Insights into Bayesian Inference, 71, 2018.

A. K. Menon and R. C. Williamson. The cost of fairness in binary classification. In Proceedings of the Conference on Fairness, Accountability and Transparency, pages 107–118. PMLR, 2018.

A. C. Miller, L. A. Gatys, J. Futoma, and E. Fox. Model-based metrics: Sample-efficient estimates of predictive model subpopulation performance. In Machine Learningfor Healthcare Conference, pages 308–336. PMLR, 2021.

S. Moro, P. Rita, and P. Cortez. Bank Marketing. UCI Machine Learning Repository, 2014. DOI: https://doi.org/10.24432/C5K306.

K. P. Murphy. Machine learning: a probabilistic perspective. MIT press, 2012.

V. Patil, G. Ghalme, V. Nair, and Y. Narahari. Achieving fairness in the stochastic multi-armed bandit problem. Journal ofMachine Learning Research, 22(174):1–31, 2021.

P. Perrault, E. Boursier, M. Valko, and V. Perchet. Statistical efficiency of thompson sampling for combinatorial semi-bandits. Advances in Neural Information Processing Systems, 33:5429–5440, 2020.

D. Pessach and E. Shmueli. A review on fairness in machine learning. ACM Computing Surveys (CSUR), 55(3):1–44, 2022.

B. Puranik, U. Madhow, and R. Pedarsani. A dynamic decision-making framework promoting long-term fairness. In Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society, pages 547–556, 2022.

C. Riou and J. Honda. Bandit algorithms based on thompson sampling for bounded reward distributions. In Algorithmic Learning Theory, pages 777–826. PMLR, 2020.

D. Russo and B. Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research, 39(4): 1221–1243, 2014.

D. J. Russo, B. Van Roy, A. Kazerouni, I. Osband, Z. Wen, et al. A tutorial on Thompson sampling. Foundations and Trends® in Machine Learning, 11(1):1–96, 2018.

C. Schumann, Z. Lang, N. Mattei, and J. P. Dickerson. Group fairness in bandits with biased feedback. In 21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2022, 2022.

C. Shui, G. Xu, Q. Chen, J. Li, C. X. Ling, T. Arbel, B. Wang, and C. Gagné. On learning fairness and accuracy on multiple subgroups. Advances in Neural Information Processing Systems, 35:34121–34135, 2022.

R. S. Sutton, A. G. Barto, et al. Reinforcement Learning: An Introduction, volume 1. MIT Press, Cambridge, 1998.

L. Wang, Y. Bai, W. Sun, and T. Joachims. Fairness of exposure in stochastic bandits. In International Conference on Machine Learning, pages 10686–10696. PMLR, 2021.

T. Wang, C. Rudin, F. Doshi-Velez, Y. Liu, E. Klampfl, and P. MacNeille. A Bayesian framework for learning rule sets for interpretable classification. Journal ofMachine Learning Research, 18(70):1–37, 2017.

W. Wei, T. Yang, H. Chen, Y. Zhao, F. Dernoncourt, R. A. Rossi, and H. Eldardiry. Learning to route llms from bandit feedback: One policy, many trade-offs. arXiv preprint arXiv:2510.07429, 2025.

M. Wen, O. Bastani, and U. Topcu. Algorithms for fairness in sequential decision making. In International Conference on Artificial Intelligence and Statistics, pages 1144–1152. PMLR, 2021.

H. Wu and S. Wager. Thompson sampling with unrestricted delays. In Proceedings of the 23rd ACM Conference on Economics and Computation, pages 937–955, 2022.

Y. Xia, F. Kong, T. Yu, L. Guo, R. A. Rossi, S. Kim, and S. Li. Which LLM to play? convergence-aware online model selection with time-increasing bandits. In Proceedings ofthe ACM Web Conference 2024, pages 4059–4070, 2024.

R. Xian, L. Yin, and H. Zhao. Fair and optimal classification via post-processing. In International conference on machine learning, pages 37977–38012. PMLR, 2023.

I.-C. Yeh and C.-h. Lien. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2):2473–2480, 2009.

T. Yin, R. Raab, M. Liu, and Y. Liu. Long-term fairness with unknown dynamics. Advances in Neural Information Processing Systems, 36:55110–55139, 2023.

Z. Yin, Z. Wang, A. Palikhe, and W. Zhang. Fairness definitions in language models explained. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 16(1):e70063, 2026.

## A Regret Bound Proofs for Thompson Sampling

We assume the standard setting where outcomes and sensitive attributes are independent across arms and IID over time. The utility $U _ { a } ( \lambda )$ depends on the underlying Bernoulli means through the accuracy $\rho _ { a }$ and the fairness gap $g _ { a } = | \mu _ { a , 0 } - \mu _ { a , 1 } |$ . Since the absolute difference is a 1-Lipschitz function, the overall utility is Lipschitz continuous with respect to the parameters. This allows us to apply standard posterior concentration results (e.g., Agrawal and Goyal [2017]).

## A.1 Regret Bound of the Thompson Sampling with No Delays (Theorem 5.1)

We follow the proof structure of Agrawal and Goyal [2017]. We first bound the expected number of selections of each suboptimal arm, then derive the gap-free regret bound. The high-level picture is:

1. Decompose selections on the utility scale: we define utility thresholds and split the selections of a suboptimal arm into three cases: (I) an ordinary competitor beats a pessimistic draw of the optimal arm, (II) the competitor’s posterior draw is overly optimistic, or (III) its empirical estimate is overly optimistic.

2. Bound the cases in the order of Agrawal and Goyal [2017]: we use the exchange step and the optimal-arm observation epochs for (I), empirical concentration for (III), and then conditional Beta concentration for (II). (II) and (III) can be addressed the same way as in Agrawal and Goyal [2017]; however (I) needs a new optimism estimate.

3. Combine and group by gap: adding three cases proves the selection-count theorem. For the gap-free regret, we charge small gaps directly and charge the anti-concentration once per group of larger gaps. This keeps the additional anti-concentration contribution free of a factor $\sqrt { N }$

We note that, at $\lambda = 0$ , the joint anti-concentration step reduces to the scalar argument of Agrawal and Goyal [2017].

We consider N fixed arms that are always available. Cases are IID, the policy is noncontextual, and feedback is immediate. Algorithm 1 keeps its original independent Beta(1, 1) priors on each coordinate, its posterior draws, and its updates from each selected case. Let $C _ { a }$ be the correctness, $Z _ { a }$ the binary prediction, and $S \in \{ \bar { 0 } , 1 \}$ the subgroup. We define

$$
\begin{array} { r l r l r l r l r } & { \rho _ { a } = \mathbb { P } ( C _ { a } = 1 ) , } & & { \mu _ { a , s } = \mathbb { P } ( Z _ { a } = 1 | S = s ) , } & & { g _ { a } = \left| \mu _ { a , 0 } - \mu _ { a , 1 } \right| , } & & { U _ { a } = \rho _ { a } - \lambda g _ { a } - c _ { a } . } \end{array}\tag{5}
$$

The costs $c _ { a } \geq 0$ are known, $\lambda \geq 0$ is finite, and $\pi _ { s } = \mathbb { P } ( S = s ) > 0$ . We choose $a ^ { * } \in$ arg ma $\operatorname { x } _ { a } U _ { a }$ , set $\Delta _ { a } = U _ { * } - U _ { a }$ and let $K _ { a } ( T )$ count the selections of arm a. We write

$$
\pi _ { \mathrm { m i n } } = \displaystyle \operatorname* { m i n } _ { s } \pi _ { s } , \quad \quad V _ { \lambda } = 1 + \frac { \lambda ^ { 2 } } { \pi _ { \mathrm { m i n } } } .\tag{6}
$$

Then $\begin{array} { r } { R _ { \lambda } ( T ) = \sum _ { a } \Delta _ { a } K _ { a } ( T ) } \end{array}$ . All bounds below are for $T \geq 2$

For positive fairness weights, we impose the following two population conditions. First, the three true means of the optimal arm lie in $\left[ \kappa _ { 0 } , 1 - \kappa _ { 0 } \right]$ with the fixed numerical margin $\kappa _ { 0 } = 1 / 1 0 0$ . Second, let $P _ { * }$ be the joint law of $( S , Z _ { * } , C _ { * } )$ , and define

$$
\begin{array} { l } { { \displaystyle \mathcal { U } ( \rho , \mu _ { 0 } , \mu _ { 1 } ) = \rho - \lambda | \mu _ { 0 } - \mu _ { 1 } | - c _ { a ^ { * } } , } } \\ { { \displaystyle K ( Q , \theta ) = d ( q _ { C } , \rho ) + \sum _ { s } q _ { s } d ( q _ { Z , s } , \mu _ { s } ) , } } \\ { { \displaystyle J _ { \lambda } ( Q ) = \operatorname* { m i n } _ { \theta : \mathcal { U } ( \theta ) \geq U _ { * } } K ( Q , \theta ) . } } \end{array}
$$

Here $d$ is the Bernoulli KL divergence, $q _ { C } = Q ( C = 1 ) , q _ { s } = Q ( S = s )$ , and $q _ { Z , s } = Q ( Z = 1 | S = s )$ . A conditional term with zero weight is zero. We assume

$$
J _ { \lambda } ( Q ) \leq D ( Q \| P _ { * } ) \quad { \mathrm { f o r e v e r y ~ } } Q { \mathrm { ~ w i t h ~ f i n i t e ~ d i v e r g e n c e } } .\tag{7}
$$

Both sides of (7) are exponential rates for n observations of the optimal arm with empirical law $Q .$ . The product Beta posterior has density proportional to $e ^ { - n K ( Q , \theta ) }$ , so it draws a parameter with utility at least $U _ { * }$ with probability $e ^ { - n J _ { \lambda } ^ { \bf * } ( Q ) }$ , up to polynomial factors in n. A history with empirical law $Q$ occurs with probability at most $e ^ { \frac { \cdot } { - } n D ( Q | | P _ { * } ) }$ The condition therefore says that a history that makes the optimal arm pessimistic is at least as rare as the pessimism it causes, so such histories cannot dominate the expected anti-concentration for an optimistic draw. $\mathbf { A } \mathbf { t } \ \lambda = 0$ , the condition holds automatically by (13). For $\lambda > 0 ,$ , K adds the accuracy and subgroup costs separately, whereas $D ( Q \| P _ { * } )$ ) is computed from the joint law of correctness and prediction, so (7) is an additional condition on $P _ { * }$

Throughout, $C \geq 1$ denotes a universal numerical constant that may change from line to line.

Theorem A.1. Under the conditions above, for every arm a with $\Delta _ { a } > 0$ and $T \geq 2 ,$ , Algorithm 1 satisfies

$$
\mathbb { E } [ K _ { a } ( T ) ] \le \underbrace { C } _ { \Delta _ { a } ^ { 2 } } \biggl [ \underbrace { \log T } _ { \mathrm { a c u r a c y } } + \underbrace { \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } \log T } _ { \mathrm { f a i m e s s \ e s t i m a t i o n } } + \underbrace { \biggl ( \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } + \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } \biggr ) ( \log T ) ^ { 7 / 2 } } _ { \mathrm { a d d i t i o n a l \ o p t i m a l - a r m \ a n t i - c o n c e n t r a i o n } } \biggr ] .\tag{8}
$$

At $\lambda = 0 ,$ , the two additional population conditions are not needed. Moreover, we also have:

$$
\mathbb { E } [ R _ { \lambda } ( T ) ] \leq C \Big [ \underbrace { \sqrt { N T \log T } } _ { \mathrm { a c c u r a c y } } + \underbrace { \frac { \lambda } { \sqrt { \pi _ { \mathrm { m i n } } } } \sqrt { N T \log T } } _ { \mathrm { f a i m e s s e s t i m a t i o n } } + \underbrace { \sqrt { \left( \frac { \lambda } { \sqrt { \pi _ { \mathrm { m i n } } } } + \frac { \lambda ^ { 2 } } { \pi _ { \mathrm { m i n } } } \right) T ( \log T ) ^ { 7 / 2 } } } _ { \mathrm { a d d i t i o n a l ~ o p i m a l - a m ~ a n t i - c o r c e n t r a i o n } } \Big ] .\tag{9}
$$

The displayed bounds separate three contributions: accuracy, fairness estimation from subgroup observations, and additional optimal-arm anti-concentration. The accuracy term already includes the scalar anti-concentration. The additional anti-concentration cost is shared across competitors, so its gap-free contribution has no factor $\sqrt { N }$ . Both fairness contributions vanish at $\lambda = 0$ . We do not change the algorithm or its sampling rule.

The anti-concentration term in (9) is absorbed into the first two terms when the number of arms is large or the fairness weight is small. Either of the two conditions

$$
N \geq ( \log T ) ^ { 5 / 2 } \quad \mathrm { o r } \quad \lambda \leq \frac { N \sqrt { \pi _ { \operatorname* { m i n } } } } { ( \log T ) ^ { 5 / 2 } }\tag{10}
$$

implies

$$
\mathbb { E } [ R _ { \lambda } ( T ) ] \le C \left( 1 + \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } \right) \sqrt { N T \log T } .\tag{11}
$$

The ratio of the anti-concentration term to the accuracy and fairness-estimation terms in (9) is at most $( \log T ) ^ { 5 / 4 } / \sqrt { N } .$ so the anti-concentration term is relatively negligible whenever $N / ( \log T ) ^ { 5 / 2 } \to \infty$ . This compares contributions to the upper bound only. It does not imply that the absolute anti-concentration term decreases with N, or that a fixed N suffices as $T \to \infty$ . We also have no lower bound showing that the extra logarithmic factors are necessary.

Lemma 2.9 of Agrawal and Goyal [2017] bounds the expected inverse odds of a scalar Beta optimism event, averaging over a binomial success count. Our exchange argument remains valid, but for $\lambda > 0$ the corresponding event is

$$
\tilde { \rho } _ { \ast } - \rho _ { \ast } + \epsilon > \lambda ( \tilde { g } _ { \ast } - g _ { \ast } ) .\tag{12}
$$

This event involves accuracy and the absolute difference of two subgroup draws with random observation counts. At $g _ { * } = 0$ , a disagreement in either direction creates a fairness penalty. Although the posterior draws are conditionally independent, their empirical histories can be dependent because accuracy and subgroup observations come from the same cases. Consequently, coordinatewise concentration bounds alone do not control the required expected inverse odds, and products of conditional probabilities cannot generally be averaged by multiplying their expectations.

The anti-concentration proof first combines the scalar bound with concentration on histories where the fairness error can be controlled within the utility margin. The remaining histories require a multivariate argument. The information condition compares the minimum KL cost of an optimistic parameter, $n J _ { \lambda } ( Q )$ , with the relative-entropy cost of its history, $n D ( Q \| P _ { * } )$ . Fixed interiority of the optimal arm supports a finite-sample lower bound on the corresponding posterior mass. The positive utility margin strengthens the exponential bound and supplies decay when summing over observation counts. Weighted information-moment bounds then control averaging over the dependent empirical histories. Thus, the information comparison alone does not establish the anti-concentration estimate. Interiority and information dominance are population assumptions, whereas the margin is a positive proof tolerance chosen from the utility gap.

For $\lambda = 0$ , the information comparison holds automatically:

$$
J _ { 0 } ( Q ) \leq d ( q _ { C } , \rho _ { * } ) \leq D ( Q \| P _ { * } ) .\tag{13}
$$

The first inequality follows by setting $\rho = \rho ,$ <sub>∗</sub> and the unused subgroup coordinates to their empirical values. The second follows from data processing. In this case, the optimism event is scalar, so Lemma 2.9 of Agrawal and Goyal [2017] applies directly without either additional population assumption. We invoke (7) and fixed interiority only in the multivariate argument.

Related reciprocal-probability arguments appear in combinatorial Thompson sampling. For example, Perrault et al. [2020] analyze Beta sampling under independent base-arm outcomes and Gaussian sampling under dependence. These results motivate a related proof strategy, but do not directly establish the required anti-concentration bound for our product-Beta sampler with dependent accuracy and subgroup observations.

## Proof

We follow the decomposition and the bounds in [Agrawal and Goyal, 2017, Section 2.1 and Lemmas 2.8–2.12] (A&G), with thresholds on the utility. The joint anti-concentration estimate below replaces their scalar Lemma 2.9.

1. Thresholds and three cases. Arms with $c _ { a } - c _ { a ^ { * } } > 1 + \lambda$ are never selected, and every remaining arm has $\Delta _ { a } \leq 2 ( 1 + \lambda )$ . For such a suboptimal arm, we set

$$
\epsilon _ { a } = \Delta _ { a } / 3 , \qquad x _ { a } = U _ { a } + \epsilon _ { a } , \qquad y _ { a } = U _ { * } - \epsilon _ { a } .\tag{14}
$$

Let $\overline { { U } } _ { a } ( t )$ be the utility evaluated at the coordinate posterior means, and let $\tilde { U } _ { a } ( t )$ be the sampled utility. We define

$$
E _ { a } ^ { \mu } ( t ) = \{ \overline { { U } } _ { a } ( t ) \leq x _ { a } \} , \qquad E _ { a } ^ { \theta } ( t ) = \{ \tilde { U } _ { a } ( t ) \leq y _ { a } \} .\tag{15}
$$

Since $U _ { a } < x _ { a } < y _ { a } < U _ { * }$ , the decomposition of Agrawal and Goyal is exact:

$$
\begin{array} { r l r } {  { \mathbb { E } [ K _ { a } ( T ) ] = \underbrace { \sum _ { t \leq T } \mathbb { P } ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) ) } _ { \mathrm { { I } } _ { a } } } } \\ & { } & { + \underbrace { \sum _ { t \leq T } \mathbb { P } ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) , E _ { a } ^ { \theta } ( t ) ^ { c } ) } _ { \mathrm { { I I } } _ { a } } + \underbrace { \sum _ { t \leq T } \mathbb { P } ( A _ { t } = a , E _ { a } ^ { \mu } ( t ) ^ { c } ) } _ { \mathrm { { I I I } } _ { a } } . } \end{array}\tag{16}
$$

2. Exchange inequality and optimal-arm epochs (Lemma 2.8 of A&G). Let $\mathcal { F } _ { t - 1 }$ be the observed history before round t. For a set B of suboptimal arms and a common threshold $y < U _ { * }$ , let $p _ { t } ( y ) = \mathbb { P } ( \tilde { U } _ { * } ( t ) > y | \mathcal { F } _ { t - 1 } )$ ). Since the arm draws are conditionally independent,

$$
\begin{array} { r l } & { \mathbb { P } ( A _ { t } \in B , \overline { { U } } _ { A _ { t } } ( t ) \leq x _ { A _ { t } } , \tilde { U } _ { A _ { t } } ( t ) \leq y | \mathcal { F } _ { t - 1 } ) } \\ & { \quad \leq \displaystyle \frac { 1 - p _ { t } ( y ) } { p _ { t } ( y ) } \mathbb { P } ( A _ { t } = a ^ { * } | \mathcal { F } _ { t - 1 } ) . } \end{array}\tag{17}
$$

To see this, we fix all draws except that of the optimal arm, whose conditional CDF is $F _ { * }$ . Let b maximize the other sampled utilities, with maximum value M. The event on the left requires $b \in B , \overline { { U } } _ { b } \leq x _ { b } ,$ , and $M \leq y$ . On these draws, its probability divided by the probability of selecting $a ^ { * }$ is $F _ { * } ( M ) \bar { / } ( 1 - F _ { * } ( M ) ) \leq ( 1 - p _ { t } ( y ) ) / p _ { t } ( y )$ . Integrating and dropping the restrictions on the right proves the inequality. Ties have probability zero.

Let ${ \mathcal { H } } _ { n }$ be the first n observations of the optimal arm in the IID observation stream. We define

$$
p _ { n } ( \epsilon ) = \mathbb { P } ( \tilde { U } _ { * } > U _ { * } - \epsilon \vert \mathcal { H } _ { n } ) ,\tag{18}
$$

$$
W ( \epsilon , T ) = \sum _ { n < T } \mathbb { E } \left[ \operatorname* { m i n } \left\{ T , \frac { 1 - p _ { n } ( \epsilon ) } { p _ { n } ( \epsilon ) } \right\} \right] .\tag{19}
$$

The probability $p _ { t } ( y )$ does not change between successive selections of the optimal arm. Conditional on the starting history of an epoch, summing (17) bounds the expected case-I count of the epoch by the inverse odds, since the epoch contains at most one selection of the optimal arm. This count is also at most T. We then sum over epochs and drop their reach indicators using the pre-generated IID streams. Thus the event on the left of (17) has expected count at most $W ( U _ { * } - y , T )$ ). In particular, $\boldsymbol { \mathrm { I } _ { a } } ^ { - } \le W ( \epsilon _ { a } , T )$

## 3. Inverse odds (replacement for Lemmas 2.9–2.10 of A&G).

Lemma A.2 (Optimal-arm anti-concentration). Assume that the optimal arm satisfies fixed interiority, $\rho _ { * } , \mu _ { * , 0 } , \mu _ { * , 1 } \in$ $\left[ \kappa _ { 0 } , 1 - \kappa _ { 0 } \right]$ with $\kappa _ { 0 } = 1 / 1 0 0$ , and the information condition $J _ { \lambda } \bar { ( } Q ) \leq D ( Q \| P _ { * } )$ for every Q with finite divergence. Set

$$
B _ { \lambda } ( T ) = \operatorname* { m i n } \left\{ \frac { \lambda ^ { 2 } } { \pi _ { \mathrm { m i n } } } ( \log T ) ^ { 9 / 2 } , V _ { \lambda } ( \log T ) ^ { 5 / 2 } \right\} .\tag{20}
$$

For $T \geq 2$ and $0 < \epsilon \leq 1 + \lambda$

$$
W ( \epsilon , T ) \leq { \frac { C } { \epsilon ^ { 2 } } } \left[ \log T + B _ { \lambda } ( T ) \right] .\tag{21}
$$

$A t \lambda = 0 ,$ this reduces to the scalar bound from Lemma 2.9 of Agrawal and Goyal [2017], and neither population condition is required.

We prove this lemma separately. This is the only step that uses the interiority and information assumptions on the optimal arm. Since $\epsilon _ { a } \leq 1 + \lambda$ , the lemma gives $\mathbf { \dot { I } } _ { a } \le \mathbf { \dot { C } } [ \log T + B _ { \lambda } ( T ) ] / \epsilon _ { a } ^ { 2 }$

4. Empirical error (Lemma 2.11 of A&G). Hoeffding’s inequality, the smoothing error of the posterior mean, and the binomial subgroup counts give, at the fixed prefix of the first n observations of arm a,

$$
\begin{array} { r } { \mathbb { P } ( \overline { { U } } _ { a , n } - U _ { a } > \epsilon _ { a } ) \le C e ^ { - n \epsilon _ { a } ^ { 2 } / ( C V _ { \lambda } ) } . } \end{array}\tag{22}
$$

Each prefix is charged at most once by a selection of the arm. Therefore,

$$
\Pi \boldsymbol { \mathrm { I I I } } _ { a } \leq \sum _ { n < T } C e ^ { - n \epsilon _ { a } ^ { 2 } / ( C V _ { \lambda } ) } \leq C V _ { \lambda } / \epsilon _ { a } ^ { 2 } .\tag{23}
$$

5. Posterior error (Lemma 2.12 of A&G). On $E _ { a } ^ { \mu } ( t ) \cap E _ { a } ^ { \theta } ( t ) ^ { c }$ , we have $\tilde { U } _ { a } ( t ) - \overline { { U } } _ { a } ( t ) > \epsilon _ { a }$ . Let $\begin{array} { r l } { m _ { a } } & { { } = } \end{array}$ ⌈CV<sub>λ</sub> log $T / \epsilon _ { a } ^ { 2 ^ { - } }$ ⌉. At most $m _ { a }$ selections occur before the arm reaches this pull count. For $n \geq m _ { a }$ and subgroup counts $\tilde { N _ { a , s } } \geq n \pi _ { s } / 2 .$ , conditional Beta concentration gives

$$
\mathbb { P } ( \tilde { U } _ { a } - \overline { { U } } _ { a } > \epsilon _ { a } | \mathcal { F } _ { t - 1 } ) \leq C / T .\tag{24}
$$

Here we allocate the tolerance $\epsilon _ { a } / 2$ to accuracy and $\epsilon _ { a } / ( 4 \lambda )$ to each subgroup. Summing this conditional bound over rounds costs at most $C .$ The selected prefixes with a deficient subgroup count cost at most $\begin{array} { r } { \sum _ { n > 0 } \sum _ { s } e ^ { - n \pi _ { s } / 8 } \le } \end{array}$ $C / \pi _ { \operatorname* { m i n } }$ . We need the subgroup control only if $\epsilon _ { a } \leq 4 \lambda$ , in which case $1 / \pi _ { \mathrm { m i n } } \leq 1 6 V _ { \lambda } / \epsilon _ { a } ^ { 2 }$ . Otherwise, the fairness deviation is at most $\lambda < \epsilon _ { a } / 4$ , and the accuracy concentration alone is enough. Consequently,

$$
\Pi _ { a } \leq C V _ { \lambda } \log T / \epsilon _ { a } ^ { 2 } .\tag{25}
$$

6. Add the three bounds (final combination of $\mathbf { A } \mathbf { \delta } \mathbf { \& } \mathbf { \delta } \mathbf { G } )$ . Substituting the three bounds into (16) proves the sharper estimate

$$
\mathbb { E } [ K _ { a } ( T ) ] \leq \frac { C } { \Delta _ { a } ^ { 2 } } \left[ V _ { \lambda } \log T + B _ { \lambda } ( T ) \right] .\tag{26}
$$

The elementary inequality min $\{ b _ { 1 } , b _ { 2 } \} \le \sqrt { b _ { 1 } b _ { 2 } }$ gives

$$
B _ { \lambda } ( T ) \le \frac { \lambda \sqrt { V _ { \lambda } } } { \sqrt { \pi _ { \operatorname* { m i n } } } } ( \log T ) ^ { 7 / 2 } \le \left( \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } + \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } \right) ( \log T ) ^ { 7 / 2 } .\tag{27}
$$

Expanding $V _ { \lambda }$ proves (8). $\mathbf { A } \mathbf { t } \lambda = 0$ , the proof uses only the scalar accuracy bounds.

## Gap-free regret: the cutoff of A&G with shared anti-concentration

Fix $d > 0$ . As in [Agrawal and Goyal, 2017, Section 2.2], gaps at most d contribute at most T d. We group the arms with larger gaps as

$$
d _ { j } = 2 ^ { j } d , \qquad B _ { j } = \{ a : d _ { j } < \Delta _ { a } \leq 2 d _ { j } \} , \qquad y _ { j } = U _ { * } - d _ { j } / 3 .\tag{28}
$$

We use the preceding decomposition with the same $x _ { a }$ and the common threshold $y _ { j }$ in group $B _ { j }$ . Since $y _ { j } - x _ { a } =$ $( 2 \Delta _ { a } - d _ { j } ) / 3 \ge \Delta _ { a } / 3$ , Steps 4 and 5 contribute at most $\begin{array} { r } { C V _ { \lambda } \log T \sum _ { { \Delta _ { a } } > d } { \Delta _ { a } ^ { - 1 } } } \end{array}$ . By Step 2, the case-I count of the whole group is bounded once by $W ( d _ { j } / 3 , T )$ . All selectable arms have $\Delta _ { a } \leq 2 ( 1 + \lambda )$ , so the anti-concentration lemma applies in every nonempty group. Therefore, the combined case-I regret is at most

$$
\sum _ { j : B _ { j } \neq \emptyset } 2 d _ { j } W ( d _ { j } / 3 , T ) \leq C [ \log T + B _ { \lambda } ( T ) ] \sum _ { j \geq 0 } \frac { 1 } { d _ { j } } = \frac { C } { d } [ \log T + B _ { \lambda } ( T ) ] .\tag{29}
$$

This grouping is the only refinement of the gap cutoff of Agrawal and Goyal: it avoids charging the same optimal-arm anti-concentration cost separately to every competitor. We obtain

$$
\mathbb { E } \big [ R _ { \lambda } ( T ) \big ] \leq T d + C V _ { \lambda } \log T \sum _ { \Delta _ { a } > d } \frac { 1 } { \Delta _ { a } } + \frac { C } { d } [ \log T + B _ { \lambda } ( T ) ] .\tag{30}
$$

We bound the sum by $N / d ,$ , absorb the scalar log T term into $N V _ { \lambda }$ log $T ,$ , and optimize over d. This gives

$$
\mathbb { E } [ R _ { \lambda } ( T ) ] \le C \sqrt { T [ N V _ { \lambda } \log T + B _ { \lambda } ( T ) ] } .\tag{31}
$$

Separating the square root and using (27) proves (9). Under either condition in (10), we have $\frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } ( \log T ) ^ { 5 / 2 } \leq$ $\begin{array} { r } { \left( 1 + \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } \right) N } \end{array}$ , so the anti-concentration term in (9) is at most the sum of the first two terms, which proves (11). Finally, $B _ { 0 } ( { \cal T } ) = 0$ gives the scalar gap-free bound without the additional population conditions.

## A.2 Proof of Lemma A.2

Let $Q _ { n }$ be the empirical joint law, $N _ { s }$ the subgroup counts, and $\hat { \rho } , \hat { \mu } _ { s }$ the empirical means. Set

$$
\begin{array} { r l } & { \mathcal { L } _ { n } ( \theta ) = n d ( \hat { \rho } , \rho ) + \displaystyle \sum _ { s } N _ { s } d ( \hat { \mu } _ { s } , \mu _ { s } ) , } \\ & { \qquad X _ { n } = \displaystyle \operatorname* { i n f } _ { \stackrel { \mathrm { i n f } } { Q _ { s } = Q _ { n , S } \pi } n D ( Q \| P _ { * } ) , } } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { b _ { \delta } ( m ) = \operatorname* { m i n } \{ 1 , \sqrt { \delta ( m + 1 ) } \} , } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}\tag{32}
$$

Zero-count likelihood terms vanish and $X _ { 0 } = j _ { 0 } = 0$ . The conditional density is $e ^ { - { \mathcal { L } } _ { n } } / Z _ { n }$ . Information dominance and data processing imply

$$
n J _ { \boldsymbol { \lambda } } ( Q _ { n } ) \le X _ { n } , \qquad \mathcal { L } _ { n } ( \theta _ { * } ) \le 2 X _ { n } , \qquad X _ { n } \ge n D ( Q _ { n , S Z } \| P _ { * , S Z } ) .\tag{33}
$$

The first inequality is interpreted as zero at $n = 0 .$

Lemma A.3 (History moments). For $0 < \delta < 1$ and $0 \leq u \leq 1 - \delta ,$

$$
\mathbb { E } [ D _ { n , \delta } e ^ { u X _ { n } } ] \leq C \delta ^ { - 2 } .\tag{34}
$$

For $n \geq 1$ and $\delta \leq 1 / 6 4$ , also

$$
\mathbb { E } \left[ \frac { D _ { n , \delta } e ^ { u X _ { n } } } { \sqrt { \operatorname* { m a x } \{ M _ { n } , \delta ^ { - 1 } \} } } \right] \leq C \left[ \frac { \delta ^ { - 2 } } { \sqrt { 1 + n \pi _ { \operatorname* { m i n } } } } + \delta ^ { - 3 / 2 } e ^ { - \delta n \pi _ { \operatorname* { m i n } } / C } \right] .\tag{35}
$$

Proof. First, for a sum $Y$ of at most four independent binomials with m total trials and $I ( x ) = \operatorname* { s u p } _ { t } \{ t x - \log \mathbb { E } e ^ { t Y } \}$ we claim

$$
\mathbb { E } \Big [ e ^ { u I ( Y ) } \Big ] \leq C \operatorname* { m i n } \{ \delta ^ { - 1 / 2 } , \sqrt { m + 1 } \} .\tag{36}
$$

Remove deterministic trials, translating $Y$ and redefining m. For the remaining groups $( m _ { j } , p _ { j } )$ , exponential tilting gives $\begin{array} { r } { q _ { j } ( t ) = p _ { j } e ^ { t } / ( 1 - p _ { j } + p _ { j } e ^ { t } ) , \bar { x ^ { } } = \sum _ { j } m _ { j } q _ { j } } \end{array}$ and $\begin{array} { r } { \dot { v } = \sum _ { j } m _ { j } q _ { j } \big ( 1 - q _ { j } \big ) } \end{array}$ . Thus dx $\langle j \dot { d } t = v , | \dot { d } v / d x | \leq 1$ , and $\begin{array} { r } { I ( x ( t ) ) = \sum _ { i } m _ { j } d ( q _ { j } ( t ) , p _ { j } ) } \end{array}$ . The Bernoulli-sum atom bound yields $\mathbb { P } ( Y = k ) \le C e ^ { - I ( k ) } / \sqrt { 1 + v ( k ) }$ , including the endpoint tilts by continuity. Compare each summand with the unit interval toward EY, leaving at most two central terms. Monotonicity of I on either side of its minimum and $| d v / d x | \leq 1$ give

$$
\begin{array} { l } { \displaystyle \mathbb { E } \Big [ e ^ { u I ( Y ) } \Big ] \leq C + C \int _ { 0 } ^ { m } \frac { e ^ { - \delta I ( x ) } } { \sqrt { v ( x ) } } d x } \\ { \displaystyle \leq C + C \sum _ { j } \sqrt { m _ { j } } \int _ { 0 } ^ { 1 } \frac { e ^ { - \delta m _ { j } d ( q , p _ { j } ) } } { \sqrt { q ( 1 - q ) } } d q \leq C \operatorname* { m i n } \{ \delta ^ { - 1 / 2 } , \sqrt { m + 1 } \} . } \end{array}
$$

The second line uses $d q _ { j } / d t = q _ { j } ( 1 - q _ { j } )$ and $\begin{array} { r } { \sqrt { \sum _ { j } a _ { j } } \le \sum _ { j } \sqrt { a _ { j } } } \end{array}$ . The final inequality follows from $q = \sin ^ { 2 } \alpha$ $d ( q , p _ { j } ) \geq c ( \alpha -$ arcsin $\sqrt { p _ { j } } ) ^ { 2 }$ , and at most four groups. The deterministic case is immediate, and restoring removed trials only enlarges the bound.

Write $n _ { s z }$ for the $( S , Z _ { * } )$ counts, $K _ { C }$ for total correctness, and $\gamma _ { s z } = \mathbb { P } ( C _ { * } = 1 | S = s , Z _ { * } = z )$ . The KL chain rule and information projection give

$$
X _ { n } = n d ( N _ { 0 } / n , \pi _ { 0 } ) + \sum _ { s } N _ { s } d ( { \hat { \mu } } _ { s } , \mu _ { * , s } ) + I _ { n } ( K _ { C } ) ,
$$

$$
I _ { n } ( k ) = \operatorname* { s u p } _ { t } \left\{ t k - \sum _ { s , z } n _ { s z } \log ( 1 - \gamma _ { s z } + \gamma _ { s z } e ^ { t } ) \right\} .\tag{37}
$$

Restrict to the support of $P _ { * }$ and use limiting tilts at boundaries. Conditional on $\left( n _ { s z } \right)$ , correctness is a sum of four independent binomials. Apply (36) and pair this moment with $b _ { \delta } ( n ) ^ { - 1 }$ using

$$
b _ { \delta } ( m ) ^ { - 1 } \operatorname* { m i n } \{ \delta ^ { - 1 / 2 } , \sqrt { m + 1 } \} = \delta ^ { - 1 / 2 } .
$$

Conditional on $( N _ { 0 } , N _ { 1 } )$ , apply the same argument to the two independent prediction counts, pairing with $b _ { \delta } ( N _ { s } ) ^ { - 1 }$ The subgroup-count moment contributes a fourth $\delta ^ { - 1 / 2 }$ factor, proving (34). At $n = 0 , D _ { 0 , \delta } = \delta ^ { - 3 / 2 } \leq \delta ^ { - 2 }$

For (35), split at min<sub>s</sub> $N _ { s } = n \pi _ { \operatorname* { m i n } } / 2$ . On the upper event, use (34). On the lower event, $X _ { n } \geq n \pi _ { \operatorname* { m i n } } / 8$ and the inverse-count factor is at most $\sqrt { \delta }$ , so its contribution is bounded by

$$
\sqrt { \delta } e ^ { - \delta n \pi _ { \mathrm { m i n } } / 1 6 } \mathbb { E } [ D _ { n , \delta / 2 } e ^ { ( u + \delta / 2 ) X _ { n } } ] \le C \delta ^ { - 3 / 2 } e ^ { - \delta n \pi _ { \mathrm { m i n } } / 1 6 } .
$$

Lemma A.4 (Conditional posterior anti-concentration). Under the population conditions, for $0 < \delta \leq 1 / 6 4$

$$
\sqrt { j _ { n } } \leq \left( \sqrt { X _ { n } } - \frac { \epsilon \sqrt { n } } { C \sqrt { V _ { \lambda } } } \right) _ { + } ,\tag{38}
$$

$$
p _ { n } ( \epsilon ) \geq \frac { \delta ^ { 3 / 2 } } { C D _ { n , \delta } } e ^ { - ( 1 - \delta ) j _ { n } - 4 \delta X _ { n } } \operatorname* { m i n } \left\{ 1 , \frac { \epsilon \sqrt { \operatorname* { m a x } \{ M _ { n } , \delta ^ { - 1 } \} } } { \lambda \sqrt { \delta } } \right\} .\tag{39}
$$

The minimum is one at $\lambda = 0 .$

Proof. For any Q, let $D = D ( Q \| P _ { * } )$ and $\hat { \boldsymbol { \theta } } = ( q _ { C } , q _ { Z , 0 } , q _ { Z , 1 } )$ . Pinsker and the split $q _ { s } \geq \pi _ { s } / 2$ give $h = ( U _ { * } -$ $\mathcal { U } ( \hat { \theta } ) ) _ { + } \leq C \sqrt { V _ { \lambda } D }$ . On the complementary subgroup event use $D \ge \pi _ { s } / 8$ and error at most one. Let $J _ { \lambda , r }$ denote the cost with utility threshold $U _ { * } - r .$ . It vanishes if $h \leq r$ . Otherwise, interpolate an unrelaxed minimizer toward $\hat { \theta }$ by $r / h$ Concavity of utility, convexity of KL, and information dominance imply

$$
J _ { \lambda , r } ( Q ) \leq ( 1 - r / h ) D \leq D - \frac { r \sqrt { D } } { C \sqrt { V _ { \lambda } } } .
$$

Take square roots, set $r = \epsilon / 2 ,$ , and minimize D over the recorded marginals to obtain (38).

For (39), let ${ \mathcal { L } } _ { n } ( \theta ^ { \circ } ) = j _ { n } ,$ , put $m = \left( n , N _ { 0 } , N _ { 1 } \right)$ , and take $v = \theta _ { * } + w$ with $| w _ { j } | \leq r _ { j } = C ^ { - 1 } \operatorname* { m i n } \{ 1 , [ \delta ( m _ { j } + 1 ) ] ^ { - 1 / 2 } \}$ Fixed interiority and Taylor’s formula give $\delta \mathcal { L } _ { n } ( v ) \leq 4 \delta X _ { n } + 1$ . Hence convexity bounds the likelihood cost at $\theta = ( 1 - \delta ) \theta ^ { \circ } + \delta v { \bf b y } \bar { ( } 1 - \delta ) j _ { n } + 4 \delta \bar { X } _ { n } + 1$ . Retain $w _ { \rho } \geq 0$ and $| w _ { 0 } - w _ { 1 } | \leq \epsilon / ( 4 \delta \lambda )$ . The image has utility at least $U _ { * } - ( 1 - \delta ) \epsilon / 2 - \epsilon / 4 > U _ { * } - \epsilon ,$ and the retained box fraction is at least $\mathrm { \bar { { c } } m i n } \{ 1 , \epsilon / ( \delta \lambda \operatorname* { m a x } _ { s } r _ { s } ) \}$ . The strip is omitted at $\lambda = 0$ . Since the Jacobian is $\delta ^ { 3 }$ and $\begin{array} { r } { Z _ { n } \leq C \prod _ { i } ( m _ { j } + 1 ) ^ { - 1 / 2 } } \end{array}$ by Pinsker, the identities

$$
\delta r _ { j } \sqrt { m _ { j } + 1 } = C ^ { - 1 } \sqrt { \delta } b _ { \delta } ( m _ { j } ) , \qquad \delta \operatorname* { m a x } _ { s } r _ { s } = \frac { \sqrt { \delta } } { C \sqrt { \operatorname* { m a x } \{ M _ { n } , \delta ^ { - 1 } \} } }
$$

prove (39), including zero counts.

ProofofLemma A.2. Write $p _ { n } = p _ { n } ( \epsilon ) , O _ { n } = ( 1 - p _ { n } ) / p _ { n }$ and $q _ { n } = \mathbb { P } ( \tilde { \rho } _ { * } > \rho _ { * } - \epsilon / 2 \mid \mathcal { H } _ { n } )$ . The Beta-Binomial identity and Chernoff bounds [Agrawal and Goyal, 2017, Appendix A], together with Agrawal and Goyal [2017, Lemma 2.9], give

$$
\begin{array} { l } { \displaystyle \mathbb { E } \big [ ( 1 - p _ { n } ) \big ] \leq C e ^ { - n \epsilon ^ { 2 } / ( C V _ { \lambda } ) } , \qquad \mathbb { E } \Big [ M _ { n } ^ { - 1 / 2 } \Big ] \leq C ( 1 + n \pi _ { \operatorname* { m i n } } ) ^ { - 1 / 2 } , } \\ { \displaystyle \sum _ { n < T } \mathbb { E } \Bigg [ \frac { 1 - q _ { n } } { q _ { n } } \Bigg ] \leq C \log T / \epsilon ^ { 2 } . } \end{array}\tag{40}
$$

For the first bound allocate $\epsilon / 2$ to accuracy and, when $\lambda > 0 , \epsilon / ( 4 \lambda )$ to each subgroup. Average the binomial counts, omitting tolerances above one. The second uses a split at half the expected count. Boundary correctness means follow directly from the Beta formula. $\mathrm { I f } \epsilon \geq 2 \lambda$ , then $p _ { n } \geq q _ { n }$ , proving the claim. Otherwise set $\nu _ { f } = \lambda ^ { 2 } / ( \pi _ { \mathrm { m i n } } \epsilon ^ { 2 } )$ and $\nu = V _ { \lambda } / \epsilon ^ { 2 }$ , so $\nu \ge \nu _ { f } > 1 / 2$

Posterior means and variances, Pinsker, and (33) yield

$$
\mathbb { P } ( \lambda ( \tilde { g } _ { * } - g _ { * } ) \leq h _ { n } | \mathcal { H } _ { n } ) \geq 3 / 4 , \qquad h _ { n } = \operatorname* { m i n } \left\{ \lambda , \frac { C \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } \sqrt { \frac { X _ { n } + 1 } { n + 1 } } \right\} .
$$

Indeed, Chebyshev gives posterior error at most $3 / \sqrt { N _ { s } + 1 }$ from the empirical mean with probability $1 5 / 1 6$ for each positive count. If $\bar { N _ { s } } \geq \bar { n \pi } _ { s } / 2 .$ , use $X _ { n } \geq N _ { s } d ( \dot { \hat { \mu } } _ { s } , \mu _ { * , s } )$ . Otherwise $\bar { X _ { n } } \geq n \pi _ { s } / 8$ and the error is at most one. The

case $n = 0$ is immediate. On $h _ { n } \leq \epsilon / 2$ , conditional independence gives $p _ { n } \geq 3 q _ { n } / 4$ . Splitting at $p _ { n } = 1 / 2$ and $q _ { n } = 1 / 2$ , and using (40), therefore gives

$$
W ( \epsilon , T ) \le C \nu + C \log T / \epsilon ^ { 2 } + \sum _ { n < T } \mathbb { E } [ \operatorname* { m i n } \{ T , O _ { n } \} \mathbb { 1 } _ { F _ { n } } ] ,\tag{41}
$$

$$
F _ { n } = \{ h _ { n } > \epsilon / 2 , p _ { n } < 1 / 2 \} \quad \Longrightarrow \quad n + 1 \leq C \nu _ { f } ( X _ { n } + 1 ) .
$$

On $F _ { n } \cap \{ x \leq X _ { n } < 2 x \}$ with $x \ge 1 , n \le C \nu _ { f } x$ and (38) gives $j _ { n } \le X _ { n } - c \sqrt { n X _ { n } / \nu }$ after decreasing c. Invert (39) with $\delta = ( 6 4 x ) ^ { - 1 }$ , use $e ^ { X _ { n } } \leq C e ^ { ( 1 - \delta ) X _ { \tau } }$ <sup>n</sup> on this shell, and apply Lemma A.3. For $n \geq 1$ this gives

$$
\mathbb { E } [ O _ { n } \mathbb { 1 } _ { F _ { n } \cap \{ x \leq X _ { n } < 2 x \} } ] \leq C e ^ { - c \sqrt { n x / \nu } } \left[ x ^ { 7 / 2 } + \frac { \lambda x ^ { 3 } } { \epsilon \sqrt { 1 + n \pi _ { \mathrm { m i n } } } } + \frac { \lambda x ^ { 5 / 2 } } { \epsilon } e ^ { - n \pi _ { \mathrm { m i n } } / ( C x ) } \right] .\tag{42}
$$

Only $n \leq C \nu _ { f } x$ contribute. Integral comparison bounds the three sums, after removing their prefactors, by

$$
C \operatorname* { m i n } \{ \nu _ { f } x , \nu / x \} , \quad \frac { C } { \sqrt { \pi _ { \mathrm { m i n } } } } \operatorname* { m i n } \{ \sqrt { \nu _ { f } x } , \sqrt { \nu / x } \} , \quad C \operatorname* { m i n } \{ \nu _ { f } x , \nu / x , x / \pi _ { \mathrm { m i n } } \} ,
$$

respectively. Using $\lambda / \epsilon = \sqrt { \pi _ { \operatorname* { m i n } } \nu _ { f } }$ and min $\{ a , b \} \leq { \sqrt { a b } }$ for the last term yields

$$
\sum _ { n < T } \mathbb { E } [ O _ { n } \mathbb { 1 } _ { F _ { n } \cap \{ x \leq X _ { n } < 2 x \} } ] \leq C \operatorname* { m i n } \{ \nu _ { f } x ^ { 9 / 2 } , \nu x ^ { 5 / 2 } \} .\tag{43}
$$

For $F _ { n } \cap \{ X _ { n } \mid < 1 \}$ , use $n \le C \nu _ { f } , ( 3 9 )$ at $\delta \ : = \ : 1 / 6 4$ , and (40). The contribution is at most $C \nu _ { f } + C \lambda / \epsilon +$ $C \lambda \sqrt { \nu _ { f } } / ( \epsilon \sqrt { \pi _ { \mathrm { m i n } } } ) \le C \nu _ { f }$ . For $X _ { n } > 8$ log $T , D _ { n , 1 / 2 } \geq 1$ and (34) give $\begin{array} { r } { T \sum _ { n < T } \mathbb { P } ( X _ { n } > 8 \log T ) \le C \dot { T } ^ { - 2 } \le C \nu _ { f } } \end{array}$ Finally, sum (43) over dyadic shells up to 8 log T. Each geometric sum is dominated by its largest scale. Substitute into (41) and use $\nu = \epsilon ^ { - 2 } + \nu _ { f }$ to conclude. □

## A.3 Regret Bound of the Thompson Sampling with Delays (Theorem 5.2)

Theorem A.5 (Regret Bound of Thompson Sampling with Fairness under Delayed Reward). Consider Algorithm 1 under the assumptions of Theorem 5.1. For each arm, assume that label delays are IID and independent of the cases and algorithmic randomization. Fairness posteriors are updated immediately, whereas accuracy posteriors are updated when labels arrive. Choose $q _ { a } \in ( 0 , 1 ]$ such that $d _ { a } ( q _ { a } ) < \infty ,$ , and let $q _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { a } q _ { a }$ . Then, the gap-free regret bound becomes

$$
\mathbb { E } [ R _ { \lambda } ( T ) ] \leq O \left( \sqrt { \frac { N T \log T } { q _ { \operatorname* { m i n } } } } + \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } \sqrt { N T \log T } + ( 1 + \lambda ) \sum _ { a = 1 } ^ { N } d _ { a } ( q _ { a } ) + \sqrt { T H _ { D } ( T ) } \right) ,
$$

where $H _ { D } ( T )$ summarizes optimal-arm anti-concentration caused by pessimistic utility draws under delayedfeedback.   
Itsformal definition is given in Appendix A.3.

## Proof

We use the notation and decomposition from the proof of Theorem 5.1. In particular, $U _ { a } , \Delta _ { a } , K _ { a } ( T ) , \overline { { U } } _ { a } ( t ) , \tilde { U } _ { a } ( t )$ and $\mathcal { F } _ { t - 1 }$ have exactly the same meanings. We retain

$$
V _ { \lambda } = 1 + { \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } } , \qquad B _ { \lambda } ( T ) = \operatorname* { m i n } \left\{ { \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } } ( \log T ) ^ { 9 / 2 } , V _ { \lambda } ( \log T ) ^ { 5 / 2 } \right\} .\tag{44}
$$

Recall that $W ( \epsilon , T )$ in that proof averages the truncated inverse optimism odds over the first n complete observations of the optimal arm, and its anti-concentration lemma gives

$$
W ( \epsilon , T ) \le \frac { C } { \epsilon ^ { 2 } } \left[ \log T + B _ { \lambda } ( T ) \right] , \qquad 0 < \epsilon \le 1 + \lambda .\tag{45}
$$

Thus $B _ { \lambda } ( T )$ is the additional fairness contribution. The scalar log $T$ contribution is absorbed into $N V _ { \lambda } \log T$ in Theorem 5.1.

We first specify the two changes needed for delayed labels. A label generated at round u with integer delay $D _ { u }$ is used at the end of round $u + D _ { u }$ . Write

$$
K _ { a } ^ { \mathrm { o b s } } ( t - 1 ) = \sum _ { u < t } \mathbb { 1 } \{ A _ { u } = a , u + D _ { u } < t \} ,
$$

$$
N _ { a , s } ( t ) = \sum _ { u < t } \mathbb { 1 } \{ A _ { u } = a , S _ { u } = s \} .\tag{46}
$$

Here $d _ { a } ( q _ { a } )$ is the $q _ { a }$ -quantile of arm a’s delay distribution. The accuracy posterior uses $K _ { a } ^ { \mathrm { o b s } } ( t - 1 )$ labels. The subgroup posterior uses the immediate count $N _ { a , s } ( t )$ , as in Theorem 5.1. In particular, $K _ { a } ( T )$ continues to count selections.

For the anti-concentration term, retain the optimism probability from the main proof:

$$
p _ { t } ( y ) = \mathbb { P } \left( \tilde { U } _ { * } ( t ) > y | \mathcal { F } _ { t - 1 } \right) .\tag{47}
$$

Partition the decision rounds into maximal consecutive epochs during which all three optimal-arm coordinate posteriors remain unchanged, including the subgroup posteriors when $\lambda = 0$ . Let $\sigma _ { e }$ be the first round of epoch e. Define

$$
\mathcal { W } _ { D } ( \epsilon , T ) = \mathbb { E } \left[ \sum _ { e : \sigma _ { \epsilon } \leq T } \operatorname* { m i n } \left\{ T , \frac { 1 - p _ { \sigma _ { e } } ( U _ { * } - \epsilon ) } { p _ { \sigma _ { e } } ( U _ { * } - \epsilon ) } \right\} \right] ,\tag{48}
$$

and

$$
H _ { D } ( T ) = \operatorname* { s u p } _ { 0 < \epsilon \le 1 + \lambda } \epsilon ^ { 2 } \mathcal { W } _ { D } ( \epsilon , T ) .\tag{49}
$$

Here $\mathcal { W } _ { D }$ plays the role of W in the exchange argument, with the actual delayed posterior histories. Consequently, $H _ { D } ( T )$ bounds the total optimal-arm anti-concentration coefficient. Its immediate-feedback counterpart is log $T +$ $B _ { \lambda } ( T )$ , rather than $B _ { \lambda } ( T )$ alone. We establish this connection explicitly below.

As in Theorem 5.1, arms with $c _ { a } - c _ { a ^ { * } } > 1 + \lambda$ are never selected, and every selectable arm satisfies $\Delta _ { a } \leq 2 ( 1 + \lambda )$ The cases $N = 1$ and $N > T$ follow from zero regret and the trivial bound $2 ( 1 + \lambda ) T$ , respectively. We therefore take $2 \leq N \leq T$ . All arm groups below contain selectable arms.

Use exactly the thresholds and events of the main proof:

$$
\begin{array} { r l r } & { } & { \epsilon _ { a } = \Delta _ { a } / 3 , \qquad x _ { a } = U _ { a } + \epsilon _ { a } , \qquad y _ { a } = U _ { * } - \epsilon _ { a } , } \\ & { } & { E _ { a } ^ { \mu } ( t ) = \{ \overline { { U } } _ { a } ( t ) \leq x _ { a } \} , \qquad E _ { a } ^ { \theta } ( t ) = \{ \tilde { U } _ { a } ( t ) \leq y _ { a } \} . } \end{array}\tag{50}
$$

The same three-case decomposition gives

$$
\begin{array} { r l } { \mathbb { E } [ K _ { \alpha } ( T ) ] = \underbrace { \displaystyle \sum _ { t \leq T } \mathbb { P } ( A _ { t } = a , E _ { \alpha } ^ { \mu } ( t ) , E _ { \alpha } ^ { \theta } ( t ) ) } _ { \mathrm { \normalfont ~ l } _ { \mathrm { = } } } } & { } \\ & { \quad \quad + \underbrace { \sum _ { t \leq T } \mathbb { P } ( A _ { t } = a , E _ { \alpha } ^ { \mu } ( t ) , E _ { \alpha } ^ { \theta } ( t ) ^ { c } ) } _ { \mathrm { \normalfont ~ l } _ { \mathrm { = } } } } \\ & { \quad \quad + \underbrace { \sum _ { t \leq T } \mathbb { P } ( A _ { t } = a , E _ { \alpha } ^ { \mu } ( t ) ^ { c } ) } _ { \mathrm { \normalfont ~ l l l } _ { \mathrm { = } } } . } \end{array}\tag{51}
$$

The exchange inequality in Step 2 of the main proof applies conditionally on $\mathcal { F } _ { t - 1 }$ without modification. It is the same conditional-draw argument as Agrawal and Goyal [2017, Lemma 2.8]. For a group B and a common threshold $y < U _ { * }$ it gives

$$
\mathbb { P } \left( A _ { t } \in B , E _ { A _ { t } } ^ { \mu } ( t ) , \tilde { U } _ { A _ { t } } ( t ) \leq y | \mathcal { F } _ { t - 1 } \right) \leq \frac { 1 - p _ { t } ( y ) } { p _ { t } ( y ) } \mathbb { P } ( A _ { t } = a ^ { * } | \mathcal { F } _ { t - 1 } ) .\tag{52}
$$

Within epoch $e , p _ { t } ( y ) = p _ { \sigma _ { e } } ( y )$ . Each epoch contains at most one optimal-arm selection, because that selection immediately updates a subgroup posterior. An accuracy-label arrival can end the epoch earlier. Conditional on its starting history, summing (52) bounds the expected group case-I count by the inverse odds, and also by T. Averaging and summing over the reached epochs gives

$$
\mathbb { E } \Bigg [ \sum _ { t \leq T } \mathbb { 1 } \{ A _ { t } \in B , E _ { A _ { t } } ^ { \mu } ( t ) , \tilde { U } _ { A _ { t } } ( t ) \Bigg ] \leq y \} \leq \mathcal { W } _ { D } ( U _ { * } - y , T ) .\tag{53}
$$

In particular, $\mathsf { I } _ { a } \leq \mathcal { W } _ { D } ( \epsilon _ { a } , T )$ . This is precisely the use of $W ( \epsilon _ { a } , T )$ in the main proof, with delayed posterior epochs.

For cases II and III, we use the coordinate concentration calculations in Steps 4 and 5 of that proof. The observationcounting change is

$$
\begin{array} { l } { \displaystyle \mathbb { E } \Bigg [ \sum _ { t \leq T } \mathbb { 1 } \{ A _ { t } = a , K _ { a } ^ { \mathrm { o b s } } ( t - 1 ) < m \} \Bigg ] \leq \frac { m } { q _ { a } } + d _ { a } ( q _ { a } ) , } \\ { \displaystyle \mathbb { E } \Bigg [ \sum _ { t \leq T } \mathbb { 1 } \{ A _ { t } = a , N _ { a , s } ( t ) < m \} \Bigg ] \leq \frac { m } { \pi _ { s } } } \end{array}\tag{54}
$$

for every integer $m \geq 1$ . For the first inequality, pre-generate the arm’s IID delays in selection order. The expected index of the mth delay at most $d _ { a } ( q _ { a } )$ is at most $m / q _ { a }$ . After that selection, all m designated labels arrive within $d _ { a } ( q _ { a } )$ ) rounds. At most $d _ { a } ( q _ { a } )$ further selections can occur during those rounds. The second inequality is the same negative-binomial argument for immediate subgroup observations. This is the delay-quantile counting principle of Wu and Wager [2022].

For completeness, the accuracy concentration calculation remains valid for the observed count in (46). Fix a calendar round t. Let $C _ { u }$ be the correctness observed for the case selected at round u, and let $\hat { \rho } _ { a } ( t )$ be the empirical accuracy from arm a’s arrived labels. For analysis, expose complete past cases, then the current action and delay, before the fresh case. The mask $w _ { u } = \mathbb { 1 } \{ A _ { u } = a , u + D _ { u } < t \}$ is predictable. Delay independence and noncontextual selection give

$$
\mathbb { E } \Bigg [ \exp \left( \eta \sum _ { u < t } w _ { u } ( C _ { u } - \rho _ { a } ) - \frac { \eta ^ { 2 } } { 8 } \sum _ { u < t } w _ { u } \right) \Bigg ] \leq 1 .\tag{55}
$$

Taking η = 4v and η = −4v yields

$$
\begin{array} { r } { \mathbb { P } \left( K _ { a } ^ { \mathrm { o b s } } ( t - 1 ) \geq m , | \hat { \rho } _ { a } ( t ) - \rho _ { a } | > v \right) \leq 2 e ^ { - 2 m v ^ { 2 } } . } \end{array}\tag{56}
$$

The subgroup argument is identical after exposing the current group. Posterior-mean smoothing and conditional Beta concentration are unchanged. No independence between accuracy and subgroup empirical histories is used.

As in the main proof, allocate tolerance $\epsilon _ { a } / 2$ to accuracy and $\epsilon _ { a } / ( 4 \lambda )$ to each subgroup. Choose observation thresholds of orders log $\dot { T } / \epsilon _ { a } ^ { 2 }$ and $\lambda ^ { 2 }$ log $T / \epsilon _ { a } ^ { 2 }$ , respectively. Omit a coordinate condition when its tolerance is at least one, and omit the subgroup conditions at $\lambda = 0$ . Above these thresholds, the empirical and posterior error probabilities at each fixed round are at most $C / T ^ { 3 }$ . Below them, apply (54). Combining the two cases before charging the insufficient-count events gives

$$
\begin{array} { r l r } & { } & { \mathsf { I I } _ { a } + \mathsf { I I I } _ { a } \le \frac { C \log T } { \Delta _ { a } ^ { 2 } } \left( q _ { a } ^ { - 1 } + \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } \right) + d _ { a } ( q _ { a } ) } \\ & { } & { \quad = \frac { C \log T } { \Delta _ { a } ^ { 2 } } \left( V _ { \lambda } + q _ { a } ^ { - 1 } - 1 \right) + d _ { a } ( q _ { a } ) . } \end{array}\tag{57}
$$

Threshold ceilings and the summed failure probabilities are absorbed into the leading term, using $\Delta _ { a } ^ { 2 } \le 8 ( q _ { a } ^ { - 1 }$ + $\lambda ^ { 2 } / \pi _ { \mathrm { m i n } } )$ . Thus only the accuracy contribution 1 in $V _ { \lambda }$ becomes $q _ { a } ^ { - 1 }$ . The fairness contribution remains $\widehat { \lambda } ^ { 2 } / \pi _ { \mathrm { m i n } } .$

Adding the three cases gives the delayed counterpart of the selection-count bound in Theorem 5.1:

$$
\mathbb { E } [ K _ { a } ( T ) ] \le \frac { C ( V _ { \lambda } + q _ { a } ^ { - 1 } - 1 ) \log T } { \Delta _ { a } ^ { 2 } } + d _ { a } ( q _ { a } ) + \mathcal { W } _ { D } ( \Delta _ { a } / 3 , T ) .\tag{58}
$$

For the gap-free bound, use the same cutoff d and groups $d _ { j } , B _ { j } , y _ { j }$ as in the main proof:

$$
d _ { j } = 2 ^ { j } d , \qquad B _ { j } = \{ a : d _ { j } < \Delta _ { a } \leq 2 d _ { j } \} , \qquad y _ { j } = U _ { * } - d _ { j } / 3 .\tag{59}
$$

Keep $x _ { a } = U _ { a } + \Delta _ { a } / 3$ . Since $y _ { j } - x _ { a } \ge \Delta _ { a } / 3$ for $a \in B _ { j } , ( 5 7 )$ applies with the common threshold. Charge gaps at most d by T d and use (53) once per nonempty group. This gives exactly the main proof’s grouped cutoff with its delayed terms:

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { \lambda } ( T ) ] \le T d + C \log T \displaystyle \sum _ { \Delta _ { a } > d } \frac { V _ { \lambda } + q _ { a } ^ { - 1 } - 1 } { \Delta _ { a } } } \\ & { \quad \quad \quad + \displaystyle \sum _ { \Delta _ { a } > d } \Delta _ { a } d _ { a } ( q _ { a } ) + \displaystyle \sum _ { j : B _ { j } \neq \emptyset } 2 d _ { j } \mathcal { W } _ { D } ( d _ { j } / 3 , T ) . } \end{array}\tag{60}
$$

Every nonempty group has $d _ { j } / 3 \leq 1 + \lambda .$ The definition of $H _ { D } ( T )$ therefore gives

$$
\sum _ { j : B _ { j } \neq \emptyset } 2 d _ { j } \mathcal { W } _ { D } ( d _ { j } / 3 , T ) \leq 1 8 H _ { D } ( T ) \sum _ { j \geq 0 } \frac { 1 } { 2 ^ { j } d } = \frac { 3 6 H _ { D } ( T ) } { d } .\tag{61}
$$

Also,

$$
\sum _ { \Delta _ { a } > d } \frac { V _ { \lambda } + q _ { a } ^ { - 1 } - 1 } { \Delta _ { a } } \leq \frac { N ( V _ { \lambda } + q _ { \operatorname* { m i n } } ^ { - 1 } - 1 ) } { d } , \qquad \sum _ { \Delta _ { a } > d } \Delta _ { a } d _ { a } ( q _ { a } ) \leq 2 ( 1 + \lambda ) \sum _ { a = 1 } ^ { N } d _ { a } ( q _ { a } ) .\tag{62}
$$

Optimizing over d as in Theorem 5.1 yields

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { \lambda } ( T ) ] \le C \sqrt { T \left[ N ( V _ { \lambda } + q _ { \operatorname* { m i n } } ^ { - 1 } - 1 ) \log T + H _ { D } ( T ) \right] } } \\ & { \qquad + C ( 1 + \lambda ) \displaystyle \sum _ { a = 1 } ^ { N } d _ { a } ( q _ { a } ) . } \end{array}\tag{63}
$$

Expanding $V _ { \lambda } + q _ { \mathrm { m i n } } ^ { - 1 } - 1 = q _ { \mathrm { m i n } } ^ { - 1 } + \lambda ^ { 2 } / \pi _ { \mathrm { m i n } }$ and separating the square root proves the stated theorem.

Finally, the connection to Theorem 5.1 uses its original W and $B _ { \lambda }$ . When labels arrive immediately, take $q _ { a } = 1$ and $d _ { a } ( 1 ) = 0$ . Each reached epoch is then the posterior after a prefix of the optimal-arm IID stream. Dropping its reach indicator gives

$$
\begin{array} { r } { \mathcal { W } _ { D } ( \epsilon , T ) \leq W ( \epsilon , T ) , \qquad H _ { D } ( T ) \leq C \left[ \log T + B _ { \lambda } ( T ) \right] . } \end{array}\tag{64}
$$

Substitution into (63) gives

$$
\begin{array} { r l } & { \mathbb { E } [ R _ { \lambda } ( T ) ] \leq C \sqrt { T \left[ N V _ { \lambda } \log T + \log T + B _ { \lambda } ( T ) \right] } } \\ & { \qquad \leq C \sqrt { T \left[ N V _ { \lambda } \log T + B _ { \lambda } ( T ) \right] } . } \end{array}\tag{65}
$$

This is the sharper gap-free bound in the main proof. Applying its same inequality

$$
B _ { \lambda } ( T ) \le \left( { \frac { \lambda } { \sqrt { \pi _ { \operatorname* { m i n } } } } } + { \frac { \lambda ^ { 2 } } { \pi _ { \operatorname* { m i n } } } } \right) ( \log T ) ^ { 7 / 2 }\tag{66}
$$

recovers all three displayed terms of Theorem 5.1. In particular, $B _ { 0 } ( T ) = 0$ recovers its scalar result. For positive delays, (63) retains $\dot { H _ { D } } ( \dot { T } )$ as defined in (49). The no-delay anti-concentration lemma is used only in (64).

non-IID $L _ { t }$

In practice, delays may correlate with subgroup identity, for example, when each subgroup has its own stationary feedback-delay distribution. We believe the proof would extend naturally to this case. The key point is that our analysis already tracks per-subgroup effective sample sizes separately — the delayed-filtration argument of Wu and Wager [2022] can be applied per-subgroup rather than globally. If delays are correlated with S, the bottleneck becomes the subgroup with the worst combination of low prevalence and long delays: intuitively, $\pi _ { \operatorname* { m i n } } \cdot q _ { m i n }$ in Theorem 5.2 would be replaced by min<sub>s</sub> ${ \bf \Xi } ( \pi _ { s } \cdot q _ { s } )$ , where $q _ { s }$ is the subgroup-specific delay quantile. The rest of the posterior concentration argument proceeds as before, since updates remain order-invariant within each subgroup. In the worst case, the minority subgroup also experiences the longest delays, compounding the $\pi _ { \mathrm { m i n } }$ factor. For other non-IID delays, the impact on the theoretical guarantees depends on the structure of the dependence.

## B Additional Details About Experiments

## B.1 Dataset Details

For the ML datasets, the arms are trained on 60% of the data (split stratified on Y), and the remaining 40% is the stream population. Twitter, Toxicity and Bios have no training step, and all their instances form the stream population. Each of the 10,000 runs samples $T = 5 0 { , } 0 0 0$ instances with replacement from the stream population (Section 6.3).

The grid of λ values depends on the oracle breakpoints, the values of λ at which the optimal arm changes. With $b _ { \mathrm { m i n } }$ and $b _ { \mathrm { m a x } }$ the smallest and largest breakpoint, the grid contains 0 and ∞, 21 log-spaced values from $b _ { \mathrm { m i n } } / 1 0 \mathrm { t o } 1 0 b _ { \mathrm { m a x } }$ and 0.9b, b and 1.1b for every breakpoint b.

For each dataset we give the λ grid, a table of the arms and a figure with four panels. Panels (a)–(c) show the regret per example, $R _ { \lambda } ( t ) / \bar { ( } ( 1 + \lambda ) t )$ , against the round t at $\lambda = 0 ,$ , at an intermediate λ and at $\lambda = \infty$ . The intermediate λ is the grid value closest to $\sqrt { b _ { \mathrm { m i n } } b _ { \mathrm { m a x } } }$ . Panel (d) shows the same quantity after $T = 5 0 \mathrm { k }$ rounds against λ, with the breakpoints as dotted vertical lines. Dividing by $1 + \lambda$ makes the values at $\overset { \cdot } { \lambda } = 0$ and $\lambda = \infty$ the accuracy and fairness regret of Figure 1. All values are means over 10,000 runs; standard errors are smaller than the markers. As in Figure 1, Random is omitted because it coincides with Round-Robin.

## B.1.1 Classifier Arms for the Tabular Datasets

Bank, Adult, Credit, ACSIncome, Coupon and COMPAS use the same 18 arms. Numeric features are standardized and categorical features one-hot encoded. All models are fitted on the training split.

• Standard models (do not use S). Gradient boosted trees (GBM), a random forest (RF, 200 trees of depth at most 12), logistic regression (LR) and a decision tree of depth 3 (Tree-d3).

• GBM with S as a feature (GBM+S).

• Operating points ofthe GBM score (GBM-OP1 to GBM-OP6). Each uses one threshold for all subgroups: GBM-OPk uses the most accurate threshold whose DP gap is at most $( 1 - k / 7 )$ times the GBM’s. The thresholds are set on the stream population, so the six arms have evenly spaced gaps.

• Fairness in-processing (S is used only in training). LR- $. \mathrm { D P } \eta , \eta \in \{ 3 , 1 0 , 3 0 , 1 0 0 \}$ , adds η times the squared difference between the subgroups’ mean predicted probabilities to the logistic loss (for $M > 2$ , the sum over all pairs of subgroups). GBM-RW is a GBM trained with the reweighing of Kamiran and Calders [2012].

• Biased arms. GBM-BiasedLabels is a GBM+S trained on labels in which 40% of the positives of the subgroup with the lowest base rate are flipped to 0. Biased predicts ${ \hat { y } } = S$ with probability 0.7 and is otherwise a fair coin (for $M > 2 ,$ , it predicts 1 for the subgroup with the highest base rate).

The fairness arms do not use S at decision time. With subgroup-specific thresholds, parity costs at most 0.1 accuracy points on Bank, Credit and Coupon, so such arms tie and there is no trade-off to learn.

## B.1.2 Twitter

The TwitterAAE dataset [Keswani et al., 2021] includes 1471 Twitter posts (a subset of a larger dataset from Davidson et al. [2017]) labeled for containing hate speech by 170 Mechanical Turk workers. The sensitive attribute - dialect (African American English (AAE) and non-AAE) - is annotated using a model from Blodgett et al. [2017], and this is treated as ground truth. Since each worker doesn’t label all the tweets the dataset is sparse - each tweet is labeled by around 4% of experts. We randomly combine 18 expert predictions into an arm a, taking the majority vote as their prediction ${ \hat { y } } _ { a }$ . This results in 10 arms, each with around 40% missingness. Unavailable arms are dynamically excluded from the action space for all algorithms, and regret is measured against the best available arm in each round.

The 29 values of λ are $\{ 0 , 0 . 0 2 3 5 , 0 . 0 3 1 8 , 0 . 0 4 3 , 0 . 0 5 8 3 , 0 . 0 7 8 9 , 0 . 1 0 7 , 0 . 1 4 4 , 0 . 1 9 6 , 0 . 2 1 1 , 0 . 2 3 5 , 0 . 2 5 8 , 0 . 2 6 5 , 0 . 1 1 , 0 . 2 6 5 , 0 . 1 2 , 0 . 1 1 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 0 . 1 2 , 1 . 0 , 0 . 1 2 , 1 . 0 , 0 . 1 2 , 1 . 0 , 0 . 1 2 , 1 . 0 , 0 . 1 2 , 1 . 0 , 1 . 0 , 0 . 1 2 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 . 0 , 1 . 0 , 1 . 0 , 1 . 0 , 1 . 0 . 0 , 1 . 0 , 1 . 0 . 0 , 1 . 0 , 1 . 0 . 0 , 1 . 0 , 1 . 0 . 1 .$ 0.358, 0.485, 0.656, 0.889, 0.901, 1, 1.1, 1.2, 1.63, 2.2, 2.98, 4.04, 5.46, 7.4, 10, ∞}.

Table 4: The 10 arms of Twitter: accuracy, DP gap and share of tweets without a prediction (%). <sup>∗</sup>: optimal for some λ.
<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td><td>% Missing</td></tr><tr><td>Arm 0</td><td>75.1</td><td>6.4</td><td>39.4</td></tr><tr><td>Arm 1</td><td>79.5</td><td>5.2</td><td>37.5</td></tr><tr><td> $\mathrm { A r m } 2 ^ { * }$ </td><td>81.5</td><td>6.2</td><td>45.8</td></tr><tr><td> $\boldsymbol { \mathrm { A r m } 3 ^ { * } }$ </td><td>76.5</td><td>1.2</td><td>45.4</td></tr><tr><td> $\mathrm { A r m } 4 ^ { \ast }$ </td><td>83.0</td><td>12.6</td><td>39.8</td></tr><tr><td> $\mathbf { A } \mathrm { r m } 5$ </td><td>80.4</td><td>11.5</td><td>34.2</td></tr><tr><td> $\mathrm { A r m } \ : 6$ </td><td>75.4</td><td>4.5</td><td>41.4</td></tr><tr><td> $\mathrm { A r m } 7$ </td><td>74.0</td><td>5.0</td><td>39.7</td></tr><tr><td> $\mathrm { A r m } \ 8$ </td><td>77.5</td><td>4.4</td><td>38.5</td></tr><tr><td>Arm 9</td><td>76.8</td><td>3.4</td><td>46.4</td></tr></table>

![](images/50c58b6bc63cf854e56371dd862cab035997c72ad66d7e9b0701fbb078fb05f2.jpg)

![](images/5f3b6f4c6719e7d8e8fce4c4af8d75062f23e27c052a33a32263c6b3a939ca11.jpg)

![](images/19f6155f5fec63eb770a4eafce2a01f6a937ad092772b4c2a826815289cbe934.jpg)

![](images/aa531c5ae8a959ff6332a1fbc938b4db5e394b668b8a9c83346f08db124674ff.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 2: Twitter: regret per example (%) against the round t at three values of λ (a)–(c), and after $T = 5 0 \mathrm { k }$ rounds against λ (d).

## B.1.3 Bank

The Bank Marketing dataset [Moro et al., 2014] collects data from direct marketing campaigns of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit. The dataset does not contain any missing values. There are a total of 41,188 instances (24,712 used for training the arms and 16,476 used to create the online stream). The sensitive attribute is Marital status (Married vs Single/Divorced) - 60.4% of the individuals are married and 39.6% are Single/Divorced. 88.7% have true label Y = 0 and 11.3% have true label Y = 1. The arms use 19 features.

The 32 values of λ are {0, 0.0211, 0.0281, 0.0373, 0.0495, 0.0658, 0.0874, 0.116, 0.154, 0.19, 0.205, 0.211, 0.232, 0.272, 0.361, 0.457, 0.48, 0.508, 0.556, 0.559, 0.618, 0.638, 0.68, 0.847, 1.13, 1.49, 1.99, 2.64, 3.5, 4.65, 6.18, ∞}.

Table 5: The 18 arms of Bank: accuracy and DP gap (%) on the stream population. <sup>∗</sup>: optimal for some λ. Arms are described in Appendix B.1.1.
<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td></tr><tr><td>GBM*</td><td>91.90</td><td>2.52</td></tr><tr><td>RF</td><td>91.26</td><td>1.75</td></tr><tr><td>LR</td><td>91.10</td><td>1.77</td></tr><tr><td>Tree-d3</td><td>90.94</td><td>2.63</td></tr><tr><td>GBM+S</td><td>91.74</td><td>2.24</td></tr><tr><td>GBM-OP1</td><td>91.67</td><td>2.16</td></tr><tr><td>GBM-OP2</td><td>91.56</td><td>1.77</td></tr><tr><td>GBM-OP3</td><td>91.44</td><td>1.43</td></tr><tr><td>GBM-OP4</td><td>90.62</td><td>1.06</td></tr><tr><td>GBM-OP5</td><td>90.26</td><td>0.70</td></tr><tr><td>GBM-OP6</td><td>89.31</td><td>0.33</td></tr><tr><td>LR-DP3</td><td>91.14</td><td>1.37</td></tr><tr><td>LR-DP10</td><td>91.07</td><td>0.85</td></tr><tr><td>LR-DP30*</td><td>91.01</td><td>0.33</td></tr><tr><td>LR-DP100*</td><td>90.80</td><td>0.00</td></tr><tr><td>GBM-RW*</td><td>91.74</td><td>1.77</td></tr><tr><td>GBM-BiasedLabels</td><td>90.44</td><td>8.68</td></tr><tr><td>Biased</td><td>43.25</td><td>69.71</td></tr></table>

![](images/34eb495a59b117b7fa21d3bb1fa65b398bc3a270a0cbf0448335c16a7b787750.jpg)

![](images/79addc33d6e80d8826178e0f515a194926f52f3f55b4a10749fcc32401a1b5a1.jpg)

![](images/9c553d58d81c3a94885a9d66ea0e3590a62186d1d544602b35dc9cca233e3a57.jpg)

![](images/090a639bd1f948cd2c7a10cc5a52c668d73f090c7a65e9c59e67847508781fa2.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 3: Bank: regret per example (%) against the round t at three values of λ (a)–(c), and after $T = 5 0 \mathrm { k }$ rounds against λ (d).

## B.1.4 Adult

The Adult dataset [Becker and Kohavi, 1996] contains 48,842 records from the 1994 US Census (the UCI training and test files combined). The goal is to predict whether income exceeds \$50K. The sensitive attribute is race, white $( S = 0 )$ vs. non-white (S = 1). We drop race, the census sampling weight and the categorical education feature (it duplicates the numeric one), which leaves 11 features. 29,305 instances are used for training and 19,537 for the stream. 14.6% of the stream population is non-white and 23.9% has Y = 1.

The 32 values of λ are {0, 0.0023, 0.00332, 0.0048, 0.00694, 0.01, 0.0145, 0.0207, 0.021, 0.023, 0.0253, 0.0303, 0.0439, 0.0634, 0.0917, 0.133, 0.192, 0.235, 0.262, 0.277, 0.288, 0.33, 0.366, 0.401, 0.403, 0.58, 0.838, 1.21, 1.75, 2.53, 3.66, ∞}.

Table 6: The 18 arms of Adult: accuracy and DP gap (%) on the stream population. <sup>∗</sup>: optimal for some λ. Arms are described in Appendix B.1.1.
<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td></tr><tr><td>GBM</td><td>87.47</td><td>9.46</td></tr><tr><td>RF</td><td>86.54</td><td>7.45</td></tr><tr><td>LR</td><td>85.43</td><td>9.15</td></tr><tr><td>Tree-d3</td><td>84.62</td><td>6.23</td></tr><tr><td>GBM+S*</td><td>87.49</td><td>9.92</td></tr><tr><td>GBM-OP1*</td><td>87.33</td><td>8.09</td></tr><tr><td>GBM-OP2</td><td>86.74</td><td>6.75</td></tr><tr><td>GBM-OP3</td><td>85.72</td><td>5.40</td></tr><tr><td>GBM-OP4</td><td>84.10</td><td>4.04</td></tr><tr><td>GBM-OP5</td><td>81.82</td><td>2.69</td></tr><tr><td>GBM-OP6</td><td>79.50</td><td>1.34</td></tr><tr><td>LR-DP3</td><td>85.13</td><td>3.56</td></tr><tr><td>LR-DP10</td><td>84.77</td><td>1.23</td></tr><tr><td>LR-DP30*</td><td>84.49</td><td>0.35</td></tr><tr><td>LR-DP100</td><td>84.37</td><td>0.74</td></tr><tr><td>GBM-RW*</td><td>87.45</td><td>8.58</td></tr><tr><td>GBM-BiasedLabels</td><td>87.23</td><td>17.81</td></tr><tr><td>Biased</td><td>61.15</td><td>70.29</td></tr><tr><td></td><td></td><td></td></tr></table>

![](images/f2592d1b2b6ae4327202ffcecd68790906a1508d7e234d0d691be0c1d084c1ec.jpg)

![](images/8bee7f41ee677c7380c32d11fa5c61fd0d70462faa5edd49551d692df7a1b4e9.jpg)

![](images/92dd15244c181e09ac26bcd74bd2b1a2d33940a0814ee667a8bf7ed06cc39a51.jpg)

![](images/038e0c67228a72d0990c686d79389444c03a1b53b13caeb5b28879f35e797f8b.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 4: Adult: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.1.5 Credit

The Default of Credit Card Clients dataset [Yeh and Lien, 2009] was created to predict customers’ credit default. There are a total of 30,000 instances. We use 18,000 to train arms and 12,000 to create the online stream. 60.8% have sensitive attribute S = 1 (female). 22.1% have true label Y = 1. The arms use 22 features.

The 29 values of λ are {0, 0.274, 0.376, 0.515, 0.706, 0.967, 1.33, 1.82, 2.47, 2.49, 2.74, 3.02, 3.41, 4.67, 6.4, 8.77, 12, 13.4, 14.9, 16.4, 16.5, 22.6, 30.9, 42.4, 58.1, 79.6, 109, 149, ∞}.

Table 7: The 18 arms of Credit: accuracy and DP gap (%) on the stream population. <sup>∗</sup>: optimal for some λ. Arms are described in Appendix B.1.1.
<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td></tr><tr><td>GBM</td><td>81.84</td><td>0.93</td></tr><tr><td>RF</td><td>81.57</td><td>0.87</td></tr><tr><td>LR</td><td>80.90</td><td>1.03</td></tr><tr><td>Tree-d3</td><td>81.73</td><td>0.94</td></tr><tr><td>GBM+S</td><td>81.88</td><td>0.72</td></tr><tr><td>GBM-OP1*</td><td>81.88</td><td>0.72</td></tr><tr><td>GBM-OP2</td><td>81.47</td><td>0.57</td></tr><tr><td>GBM-OP3</td><td>80.82</td><td>0.52</td></tr><tr><td>GBM-OP4</td><td>79.89</td><td>0.39</td></tr><tr><td>GBM-OP5</td><td>78.42</td><td>0.23</td></tr><tr><td>GBM-OP6*</td><td>77.94</td><td>0.08</td></tr><tr><td>LR-DP3</td><td>80.82</td><td>0.76</td></tr><tr><td>LR-DP10</td><td>80.83</td><td>0.53</td></tr><tr><td>LR-DP30*</td><td>80.61</td><td>0.26</td></tr><tr><td>LR-DP100</td><td>80.12</td><td>0.50</td></tr><tr><td>GBM-RW</td><td>81.79</td><td>0.73</td></tr><tr><td>GBM-BiasedLabels</td><td>80.08</td><td>10.45</td></tr><tr><td>Biased</td><td>44.50</td><td>70.42</td></tr></table>

![](images/5a7f6c7220cac84dd0e4cd9463572a150c4f55c7c9904701bf65f92fc918e4f3.jpg)

![](images/1286c7a1e8169ec8a05b7588604ce6065ee3627717b1a5a4fbd194a837e4e842.jpg)

![](images/b8e99459518a1e2fe8aa7b20e3e8287da5694c36dc75a2f3785026b4eb136888.jpg)

![](images/25a042a6f63465351d8a87d295a71a315cd4b159300b6baffec87822b304bf98.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 5: Credit: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.1.6 Toxicity

We parse raw runs from Liang et al. [2023] available at https://crfm.stanford.edu/helm/classic/latest/ #/runs. Here, 66 different LLMs were queried - the inputs start with a comment (either toxic or non-toxic from the CivilComments dataset) followed by “\n Question: Is the passage above toxic?”. Outputs are either “True” or “False.” Accuracy is measured using quasi-exact match. Any LLMs with missing responses were discarded. We do not include perturbed instances. This leaves us with 56 LLMs (arms) and 2000 prompts (1000 prompts mention Black demographics and 1000 mention White). The motivation behind the dataset is that toxicity classifiers pick up on biases in the training data and spuriously associate toxicity with the mention of certain demographics. A fair classifier would have a low DP-gap since classifications of Toxic should be equal for both subgroups. 29.4% of the prompts have true label Toxic. The marginal accuracy and DP-gap of the 56 LLMs used for experiments are summarized in Table 8.

The 29 values of λ are {0, 0.0134, 0.0175, 0.023, 0.0301, 0.0395, 0.0518, 0.0679, 0.089, 0.117, 0.12, 0.134, 0.147, 0.153, 0.2, 0.263, 0.27, 0.3, 0.33, 0.344, 0.451, 0.591, 0.775, 1.02, 1.33, 1.75, 2.29, 3, ∞}.

Table 8: The 56 LLMs of Toxicity: accuracy and DP gap (%). <sup>∗</sup>: optimal for some λ.
<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td><td>Arm</td><td>Accuracy</td><td>DP gap</td></tr><tr><td>together_opt-66b</td><td>29.5</td><td>0.2</td><td>cohere_command-xlarge-beta</td><td>58.7</td><td>5.1</td></tr><tr><td>cohere_medium-20220720</td><td>29.7</td><td>1.0</td><td>ai21_j2-large</td><td>59.7</td><td>7.4</td></tr><tr><td>together_opt-175b</td><td>30.9</td><td>2.1</td><td>together_redpajama-incite-instruct-7b</td><td>60.7</td><td>9.3</td></tr><tr><td>together_gpt-neox-20b</td><td>31.2</td><td>1.3</td><td>tiiuae_falcon-40b-instruct</td><td>60.9</td><td>28.7</td></tr><tr><td>together_gpt-j-6b</td><td>32.1</td><td>2.1</td><td>together_bloom</td><td>61.8</td><td>21.0</td></tr><tr><td>openai_davinci</td><td>34.9</td><td>11.4</td><td>eleutherai_pythia-12b-v0</td><td>62.9</td><td>4.3</td></tr><tr><td>cohere_large-20220720</td><td>37.3</td><td>16.8</td><td>openai_text-davinci-003</td><td>65.5</td><td>10.7</td></tr><tr><td>cohere_xlarge-20220609</td><td>38.5</td><td>3.2</td><td>tiiuae_falcon-40b</td><td>65.8</td><td>27.6</td></tr><tr><td>together_redpajama-incite-base-7b</td><td>38.8</td><td>38.2</td><td>AlephAlpha_luminous-supreme</td><td>66.5</td><td>17.6</td></tr><tr><td>cohere_command-medium-beta</td><td>38.8</td><td>6.9</td><td>openai_ada</td><td>67.8</td><td>22.5</td></tr><tr><td>ai21_j2-jumbo</td><td>40.1</td><td>11.2</td><td>ai21_j1-large</td><td>67.8</td><td>2.5</td></tr><tr><td>ai21_j1-jumbo</td><td>41.5</td><td>9.7</td><td>together_yalm</td><td>68.8</td><td>4.7</td></tr><tr><td>meta_llama-13b</td><td>43.0</td><td>16.7</td><td>cohere_xlarge-20221108</td><td>69.2</td><td>0.3</td></tr><tr><td>AlephAlpha_luminous-base</td><td>43.1</td><td>43.3</td><td>ai21_j1-grande</td><td>69.3</td><td>6.5</td></tr><tr><td>meta_llama-7b</td><td>43.8</td><td>44.0</td><td>AlephAlpha_luminous-extended</td><td>69.6</td><td>0.8</td></tr><tr><td>lmsys_vicuna-13b-v1.3</td><td>45.1</td><td>2.5</td><td>openai_babbage</td><td>69.8</td><td>0.4</td></tr><tr><td>mosaicml_mpt-30b</td><td>45.1</td><td>20.2</td><td>tiiuae_falcon-7b</td><td>70.2</td><td>3.3</td></tr><tr><td>openai_curie</td><td>46.2</td><td>24.2</td><td>cohere_medium-20221108</td><td>70.5</td><td>0.1</td></tr><tr><td>eleutherai_pythia-6.9b</td><td>46.5</td><td>14.2</td><td>openai_text-ada-001*</td><td>70.6</td><td>0.0</td></tr><tr><td>lmsys_vicuna-7b-v1.3</td><td>47.2</td><td>27.2</td><td>together_glm*</td><td>70.6</td><td>0.0</td></tr><tr><td>meta_llama-2-13b</td><td>47.9</td><td>15.5</td><td>cohere_small-20220720*</td><td>70.6</td><td>0.0</td></tr><tr><td>microsoft_TNLGv2_530B</td><td>48.7</td><td>0.3</td><td>openai_text-babbage-001*</td><td>70.6</td><td>0.0</td></tr><tr><td>meta_llama-2-70b</td><td>51.4</td><td>1.3</td><td>ai21_j2-grande</td><td>70.8</td><td>3.8</td></tr><tr><td>microsoft_TNLGv2_7B</td><td>51.7</td><td>53.7</td><td>meta_llama-30b</td><td>71.2</td><td>4.8</td></tr><tr><td>together_redpajama-incite-base-3b-v1</td><td>53.8</td><td>76.1</td><td>tiiuae_falcon-7b-instruct*</td><td>71.2</td><td>2.0</td></tr><tr><td>meta_llama-65b</td><td>55.5</td><td>9.5</td><td>ai21_j1-grande-v2-beta</td><td>71.3</td><td>3.8</td></tr><tr><td>anthropic_stanford-online-all-v4-s3</td><td>56.2</td><td>10.2</td><td>meta_llama-2-7b</td><td>72.0</td><td>23.1</td></tr><tr><td>together_t5-11b</td><td>58.0</td><td>39.2</td><td>together_redpajama-incite-instruct-3b-v1*</td><td>72.2</td><td>9.1</td></tr></table>

(a) λ = 0  
![](images/743003adf6cea64a0c6d45f9ae46e7525df4ab54390039b78bf4107122e01685.jpg)

(b) λ = 0.2  
![](images/cbd283aaa9457f3ede87890e5b9b58024591a1271988c013d43a5316f729bb59.jpg)

(c) λ = ∞  
![](images/efb56546ff76b0662e720455b0b02aa299926c4c3e3866bbfe0c65b1f0a9d50e.jpg)

![](images/8e50915c1db6277dad64b2f87d63699f4dce4466e761702784c2ecc359d5b8c4.jpg)  
Figure 6: Toxicity: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.1.7 Bios

Bias in Bios [De-Arteaga et al., 2019] contains online biographies labeled with one of 28 occupations, with the first sentence (which names the occupation) removed. The sensitive attribute is gender. We sample 30,000 biographies at random from the test split; 46.1% are women, and the most frequent occupation (professor) accounts for 30.0%. The arms are 16 open instruction-tuned LLMs with 0.36B to 9B parameters, fixed before any model was run. Each model receives a zero-shot prompt with the biography and the list of the 28 occupations, and its greedy decoding is constrained to one of the occupation names. The gap is the total variation distance between the subgroups’ distributions of predicted occupations.

The 29 values of λ are {0, 0.277, 0.36, 0.467, 0.607, 0.788, 1.02, 1.33, 1.72, 2.24, 2.49, 2.77, 2.91, 3.05, 3.77, 4.63, 4.9, 5.14, 5.65, 6.36, 8.26, 10.7, 13.9, 18.1, 23.5, 30.5, 39.6, 51.4, ∞}.

Table 9: The 16 arms of Bios: accuracy and TV gap (%) on the stream population. <sup>∗</sup>: optimal for some λ.
<table><tr><td>Arm</td><td>Accuracy</td><td>TV gap</td></tr><tr><td>SmolLM2-360M</td><td>13.9</td><td>24.3</td></tr><tr><td>SmolLM2-1.7B*</td><td>28.7</td><td>13.3</td></tr><tr><td>Qwen2.5-0.5B</td><td>47.1</td><td>28.3</td></tr><tr><td>Gemma3-1B</td><td>49.9</td><td>22.3</td></tr><tr><td>Qwen2.5-1.5B</td><td>53.0</td><td>25.7</td></tr><tr><td>OLMo2-7B</td><td>56.8</td><td>28.7</td></tr><tr><td>Gemma3-4B</td><td>60.7</td><td>22.3</td></tr><tr><td>Qwen3-4B</td><td>61.9</td><td>24.7</td></tr><tr><td>Qwen2.5-3B</td><td>63.6</td><td>24.5</td></tr><tr><td>Phi4-mini</td><td>65.0</td><td>22.6</td></tr><tr><td>Mistral-7B</td><td>65.2</td><td>24.7</td></tr><tr><td>Qwen3-8B*</td><td>66.8</td><td>20.7</td></tr><tr><td>Gemma2-2B</td><td>66.9</td><td>27.7</td></tr><tr><td>Phi3.5-mini</td><td>67.4</td><td>21.6</td></tr><tr><td>Qwen2.5-7B</td><td>68.1</td><td>24.6</td></tr><tr><td>Gemma2-9B*</td><td>74.6</td><td>23.5</td></tr></table>

![](images/1c3dee8fe6e72a53415b0c2ff1164d064f42a218beb89811fe074521a847819d.jpg)

(b) λ = 3.77  
![](images/c80b668b63a030467690f6a085efaa71c3adb789b5c12460abc5952775156417.jpg)

(c) λ = ∞  
![](images/6d6c1079145a05f8fe44f5112cdd6c7a6292b39295b54097a511125bc8174d4e.jpg)

(d) T= 50k  
![](images/990f85c29e021614bde6f682962c90c1df9d9c914be2e49008ef5c65a6b5164f.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 7: Bios: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.1.8 ACSIncome

ACSIncome [Ding et al., 2021] is built from the 2018 American Community Survey (California), with 195,665 instances after the standard ACSIncome filter. The goal is to predict whether income exceeds \$50K (41.1% of the stream population). The 9 features are age, class of worker, education, marital status, occupation, place of birth, relationship, hours worked per week and sex. The sensitive attribute is race/ethnicity in five groups: White (42.5% of the stream population), Hispanic (33.1%), Asian (16.6%), Black (4.2%) and Other (3.6%). 117,399 instances are used for training and 78,266 for the stream. The DP gap is the largest difference in positive prediction rate between any two groups.

The 38 values of λ are {0, 0.00893, 0.0132, 0.0194, 0.0287, 0.0423, 0.0624, 0.0804, 0.0893, 0.092, 0.0982, 0.136, 0.2, 0.222, 0.247, 0.272, 0.295, 0.372, 0.414, 0.436, 0.455, 0.643, 0.672, 0.746, 0.821, 0.948, 1.4, 1.91, 2.06, 2.13, 2.34, 3.04, 4.49, 6.63, 9.77, 14.4, 21.3, ∞}.

Table 10: The 18 arms of ACSIncome: accuracy and DP gap (%) on the stream population. <sup>∗</sup>: optimal for some λ. Arms are described in Appendix B.1.1.
<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap (max-min)</td></tr><tr><td>GBM*</td><td>82.22</td><td>35.33</td></tr><tr><td>RF</td><td>79.32</td><td>33.02</td></tr><tr><td>LR</td><td>81.84</td><td>36.06</td></tr><tr><td>Tree-d3</td><td>74.65</td><td>31.81</td></tr><tr><td>GBM+S</td><td>82.20</td><td>40.51</td></tr><tr><td>GBM-OP1</td><td>80.50</td><td>30.29</td></tr><tr><td>GBM-OP2</td><td>77.53</td><td>25.22</td></tr><tr><td>GBM-OP3</td><td>73.94</td><td>20.18</td></tr><tr><td>GBM-OP4</td><td>70.28</td><td>15.14</td></tr><tr><td>GBM-OP5</td><td>66.38</td><td>10.08</td></tr><tr><td>GBM-OP6</td><td>61.91</td><td>5.02</td></tr><tr><td>LR-DP3*</td><td>76.48</td><td>8.03</td></tr><tr><td> $\mathrm { L R - D P 1 0 ^ { * } }$ </td><td>75.08</td><td>4.64</td></tr><tr><td> $\mathrm { L R - D P } 3 0 ^ { * }$ </td><td>74.60</td><td>4.00</td></tr><tr><td> $\mathrm { L R - D P l 0 0 ^ { \ast } }$ </td><td>74.42</td><td>3.91</td></tr><tr><td>GBM-RW*</td><td>81.65</td><td>28.97</td></tr><tr><td>GBM-BiasedLabels</td><td>80.92</td><td>52.53</td></tr><tr><td>Biased</td><td>58.00</td><td>70.43</td></tr><tr><td></td><td></td><td></td></tr></table>

(a) λ = 0  
![](images/ba156ed21735f42f3f4c20f408da96e5e86736ae26219eca0ccbd534c28cc41f.jpg)

(b) λ = 0.436  
![](images/5f50c22f797b4f72269ae04bff26bf726387919072f46d96119e219647ff45e6.jpg)

(c) λ = ∞  
![](images/d1abe0070dfff36ec07e576919039478051c4a4f50d72ae9b0425c36df51921d.jpg)

![](images/562f92287b3f5d7ac3732253611544e324a67ae0a6e7814a19b618b179ad3103.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 8: ACSIncome: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.1.9 Coupon

For the In-Vehicle Coupon Recommendation [Wang et al., 2017] dataset, driving scenarios are presented to Mechanical Turk workers, and they are asked whether they would accept a coupon if they were the driver. There are a total of 12,684 instances. 7,610 are used for training arms, and 5,074 are used for creating an online stream. 49.7% of the test data have sensitive attribute S = 1 (male). 56.8% have true label Y = 1. The arms use 23 features.

The 29 values of λ are {0, 0.395, 0.548, 0.759, 1.05, 1.46, 2.02, 2.8, 3.56, 3.88, 3.95, 4.35, 5.38, 7.46, 10.3, 14.3, 19.9, 24.3, 27, 27.5, 29.7, 38.1, 52.9, 73.3, 102, 141, 195, 270, ∞}.

Table 11: The 18 arms of Coupon: accuracy and DP gap (%) on the stream population. <sup>∗</sup>: optimal for some λ. Arms are described in Appendix B.1.1.  
![](images/a2852ac8c3b6bed3d53a9c6dbf379673133600e58a211674d981d970e4a18824.jpg)

<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td></tr><tr><td>GBM</td><td>73.93</td><td>4.04</td></tr><tr><td>RF</td><td>73.51</td><td>2.47</td></tr><tr><td>LR</td><td>68.23</td><td>3.13</td></tr><tr><td>Tree-d3</td><td>63.84</td><td>3.98</td></tr><tr><td>GBM+S</td><td>74.24</td><td>5.53</td></tr><tr><td>GBM-OP1</td><td>73.18</td><td>3.37</td></tr><tr><td>GBM-OP2</td><td>72.23</td><td>2.86</td></tr><tr><td>GBM-OP3</td><td>69.49</td><td>2.22</td></tr><tr><td>GBM-OP4</td><td>66.30</td><td>1.70</td></tr><tr><td>GBM-OP5</td><td>65.10</td><td>1.08</td></tr><tr><td>GBM-OP6*</td><td>58.32</td><td>0.57</td></tr><tr><td>LR-DP3</td><td>67.95</td><td>2.11</td></tr><tr><td>LR-DP10</td><td>67.93</td><td>1.52</td></tr><tr><td>LR-DP30</td><td>67.90</td><td>1.04</td></tr><tr><td>LR-DP100*</td><td>67.88</td><td>0.92</td></tr><tr><td>GBM-RW*</td><td>74.28</td><td>2.54</td></tr><tr><td>GBM-BiasedLabels</td><td>65.92</td><td>48.86</td></tr><tr><td>Biased</td><td>51.46</td><td>69.60</td></tr></table>

Figure 9: Trade-off between accuracy and fairness regret on Coupon after T = 50k rounds, as in Figure 1. Round-Robin is off-scale and omitted.

(a) λ = 0  
![](images/99c308997d48b9c63154ecf09aaf8e930577a3554ab0c25df2993f6e85291ccc.jpg)

(b) λ = 10.3  
![](images/4d586a7247639f4ecf6149511a3e3804e656766f25a05c67e909d76fd84b1a5f.jpg)

(c) λ = ∞  
![](images/2d92f9112d6159bc14074b14bca212119917ab7adaa274d1443d3cd247dbfdcd.jpg)

(d) T = 50k  
![](images/f49d0930d734dc6caf3322ccbef0cdf366219b8052a27c12983e29a99063d72c.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 10: Coupon: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.1.10 COMPAS

The COMPAS dataset [Larson et al., 2016] contains criminal defendants from Broward County, Florida. The goal is to predict recidivism within two years. We apply the filters of the ProPublica analysis and keep African-American $( S ^ { ' } = 1 )$ and Caucasian (S = 0) defendants, which leaves 5,278 instances. The 7 features are age, sex, number of prior offenses, three juvenile offence counts and the charge degree; the COMPAS score is not used. 3,166 instances are used for training and 2,112 for the stream. 39.6% of the stream population is Caucasian and 47.1% has $Y = 1$

The 44 values of λ are {0, 0.00208, 0.00314, 0.00475, 0.00719, 0.0109, 0.0165, 0.0187, 0.0208, 0.0228, 0.0249, 0.0377, 0.046, 0.0512, 0.0563, 0.057, 0.0863, 0.131, 0.198, 0.222, 0.246, 0.271, 0.299, 0.34, 0.378, 0.416, 0.452, 0.479, 0.532, 0.585, 0.652, 0.684, 0.725, 0.739, 0.797, 0.821, 0.903, 1.04, 1.57, 2.37, 3.59, 5.43, 8.21, ∞}.

Table 12: The 18 arms of COMPAS: accuracy and DP gap (%) on the stream population. <sup>∗</sup>: optimal for some λ. Arms are described in Appendix B.1.1.  
![](images/4fbbfa59affda399fa027bd33dc8bd54e4511e271b88f06fd707a5ca5376436f.jpg)

<table><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td></tr><tr><td>GBM</td><td>66.48</td><td>22.20</td></tr><tr><td>RF</td><td>66.34</td><td>22.92</td></tr><tr><td>LR*</td><td>67.05</td><td>26.20</td></tr><tr><td> ${ \mathrm { T r e e } } { \cdot } { \mathrm { d } } 3 ^ { \ast }$ </td><td>67.00</td><td>23.92</td></tr><tr><td>GBM+S*</td><td>66.95</td><td>23.00</td></tr><tr><td>GBM-OP1</td><td>65.10</td><td>18.88</td></tr><tr><td> $\mathrm { G B M - O P } 2 ^ { \ast }$ </td><td>64.63</td><td>15.52</td></tr><tr><td>GBM-OP3*</td><td>63.07</td><td>12.58</td></tr><tr><td>GBM-OP4</td><td>60.75</td><td>9.47</td></tr><tr><td> $\mathrm { G B M - O P } 5 ^ { \ast }$ </td><td>58.38</td><td>6.12</td></tr><tr><td> $_ \mathrm { G B M - O P 6 ^ { * } }$ </td><td>55.87</td><td>3.06</td></tr><tr><td> $\mathrm { L R - D P } 3 ^ { \ast }$ </td><td>66.00</td><td>19.15</td></tr><tr><td>LR-DP10</td><td>63.49</td><td>15.04</td></tr><tr><td>LR-DP30</td><td>61.51</td><td>13.07</td></tr><tr><td>LR-DP100</td><td>57.53</td><td>6.39</td></tr><tr><td>GBM-RW</td><td>65.67</td><td>19.60</td></tr><tr><td>GBM-BiasedLabels</td><td>64.11</td><td>43.19</td></tr><tr><td>Biased</td><td>54.02</td><td>70.68</td></tr></table>

Figure 11: Trade-off between accuracy and fairness regret on COM-PAS after T = 50k rounds, as in Figure 1.

![](images/0ac0d780ab919fbaf72322c468efc7760a8207485d9ea14d54d987d9df86b5a8.jpg)  
(a) λ = 0

![](images/36aa21dc06c5d1634e8cdd304791ab6ced4d9c75b3dec997353285f673767ec0.jpg)  
(b) λ = 0.131

(c) λ = ∞  
![](images/71a7178ed54a3f86a8910a5714bacbac6718f2f6a6b23f47cff537d9f7f213aa.jpg)

(d) T = 50k  
![](images/b003ea585b2272920f0cc7a9c56bc1a0e569e4efd68f84c9297fb46f8b3a3560.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 12: COMPAS: regret per example (%) against the round t at three values of λ (a)–(c), and after T = 50k rounds against λ (d).

## B.2 UCB Baselines

Both UCB baselines replace the unknown accuracy $\rho _ { a }$ with an upper bound $\rho _ { a } ^ { + }$ and the ${ \tt g a p } g _ { a }$ with a lower bound $g _ { a } ^ { - }$ In round t, the learner selects the available arm with the largest optimistic utility

$$
\hat { U } _ { a } ( t ) = \rho _ { a } ^ { + } ( t ) - \lambda g _ { a } ^ { - } ( t ) - c _ { a } ,
$$

with $\hat { U } _ { a } ( t ) = - g _ { a } ^ { - } ( t ) - c _ { a }$ for $\lambda = \infty$ . Ties are broken uniformly at random.

UCB1 Let $n _ { a }$ be the number of times arm a has been selected, $n _ { a , s }$ the number of those rounds with $S = s ,$ and $\hat { \rho } _ { a }$ and $\hat { \mu } _ { a , s }$ the empirical accuracy and subgroup-positive rate. With the Hoeffding radii $r _ { a } = \sqrt { 2 \log { t / n _ { a } } }$ and $r _ { a , s } = \sqrt { 2 \log t / n _ { a , s } }$ [Auer et al., 2002],

$$
\begin{array} { r } { \rho _ { a } ^ { + } = \operatorname* { m i n } \big ( 1 , \hat { \rho } _ { a } + r _ { a } \big ) , \qquad g _ { a } ^ { - } = \operatorname* { m a x } \big ( 0 , | \hat { \mu } _ { a , 0 } - \hat { \mu } _ { a , 1 } | - r _ { a , 0 } - r _ { a , 1 } \big ) . } \end{array}
$$

An arm that has not been selected has an infinite index, so every arm is tried once. Until both subgroups of an arm have been observed, its gap bound is zero.

Bayes-UCB Bayes-UCB keeps the Beta posteriors of TS (Algorithm 1) and replaces each posterior sample with a quantile [Kaufmann et al., 2012]. The accuracy bound $\rho _ { a } ^ { + }$ is the $1 - 1 / t$ quantile of $\mathrm { B e t a } ( \alpha _ { a } , \beta _ { a } )$ . For the gap, the $1 / t$ and $1 - 1 / t$ quantiles of Bet $\mathrm { a } ( \alpha _ { a , s } , \bar { \beta } _ { a , s } )$ give an interval $[ \ell _ { a , s } , u _ { a , s } ]$ for each subgroup, and $\begin{array} { r l } { g _ { a } ^ { - } } & { { } = } \end{array}$ max(0, $\ell _ { a , 0 } - u _ { a , 1 } , \ell _ { a , 1 } - u _ { a , 0 } )$ is the smallest gap consistent with both intervals. We use the level $1 - 1 / t , \mathrm { i . e . } \bar { c } = 0$ in the level $1 - 1 / ( t ( \log T ) ^ { c } )$ of Kaufmann et al. [2012]. Their regret bound requires $c \geq 5$ , but $c = 0$ gave the best results in their simulations. To save computation, for $t > 2 0 0$ we round t in the level down to a geometric grid with ratio 1.01, so a quantile is recomputed only when the arm’s counts or the grid point change. With delayed labels, the posteriors are updated as in Algorithm 1.

Multi-class and multi-subgroup Both methods bound each rate separately. With C classes, UCB1 uses the radii above for every class rate $\mu _ { a , s , k }$ , and Bayes-UCB uses the quantiles of the Beta marginals $\mathrm { B e t a } ( 1 + n _ { a , s , k } , C - 1 + n _ { a , s } - n _ { a , s , k } )$ of the Dirichlet posterior. The gap bound is half the sum over classes of the smallest difference consistent with the per-class intervals. With M subgroups, $g _ { a } ^ { - } = \operatorname* { m a x } ( 0$ , max<sub>s</sub> $\ell _ { a , s } - \operatorname* { m i n } _ { s } u _ { a , s } )$ , where $[ \ell _ { a , s } , u _ { a , s } ]$ are the subgroup intervals of either method. For real-valued rewards, see Appendix B.7.

## B.3 Fewer Rounds

Figures 13–15 repeat Figure 1 after $T = 2 , 5 0 0 , 5 , 0 0 0$ and 10,000 rounds, the first rounds of the same runs. At these horizons ϵ-Greedy has the lowest regret at most values of λ on Twitter, Adult and Toxicity and at some values on Bio and ACSIncome, while TS has the lowest regret at nearly all values on Bank and Credit. The regret per example of TS keeps decreasing with T (Appendix B.1), and after $T = { \dot { 5 } } 0 \mathrm { k }$ rounds TS has the lowest regret at every λ on four datasets and at all but one on Credit (Table 3).

(a) Twitter  
![](images/f459bced0d3165984abcdfbb89826ac1129a9f139e330befe128a3054991a698.jpg)  
(e) Toxicity

(b) Bank  
![](images/9ec2de1d472847afa41bb2586bc4076bbf7672578bf1c3cfc9a40071c6bb3de3.jpg)  
(f) Bias in Bios

(c) Adult  
![](images/43c9330feaa29cc038669c9b890f5bb94361acafa631a2fd3438d1c82f617912.jpg)  
(g) ACSIncome

(d) Credit  
![](images/005a3a1dc779a30c9740f21c2876bef4dffb2b6d184664d243a12988683c6a29.jpg)

![](images/fc130c6d2865a0598a83e430f9c1ca415359828d6d99aac054f810ded2ace424.jpg)

![](images/0d7d478a5c335bb590b3e85e57add27a20268c59dcd85a85f5f31cfbe308aed7.jpg)

![](images/80dfe7d0b3373f5662d81ed345d8b4d25f2b402242a397c3ac156921e6965668.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin Individual arms Oracle frontier  
Accuracy regret per example (%)  
Figure 13: Trade-off between accuracy and fairness regret after $T = 2 { , } 5 0 0$ rounds, as in Figure 1.

(a) Twitter  
![](images/570bfcb9a309d12f6fbb6286061e85374bf2cc8707000276dadee5e9ae305680.jpg)

(b) Bank  
![](images/d921ecbc481a7783e23fbc78fd19ea155610f628fad868ad7cdc682f0f5f3a4f.jpg)  
(f) Bias in Bios

(c) Adult  
![](images/a899998bd3b372c19f96e93a97d844cb70f0c28944c31ea9f359662ec053ce93.jpg)  
(e) Toxicity  
(g) ACSIncome

(d) Credit  
![](images/da016a28a473ac111548afd3e014ac6918cb839b904b09b5cee2e954e56c71a8.jpg)

![](images/458dd4341286795657395138d8814f8674222fc8368efca4406fa213be9cd81d.jpg)

![](images/6479a8fbbab10bc2cda7394ab764c7a8eaac9ac2ac4b67716920b83796ede0ab.jpg)

![](images/9fd69a0c8c16b5a32b7ad538b293b38f12e2008cc98981bdfc48aaf3e44aad02.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin Individual arms Oracle frontier

Accuracy regret per example (%)  
Figure 14: Trade-off between accuracy and fairness regret after $T = 5 { , } 0 0 0$ rounds, as in Figure 1.  
(a) Twitter  
![](images/32c448e9217ff183632187b095e39f8d10bdc5acf38bc09d62571cb52b5bbc63.jpg)

(b) Bank  
![](images/b24f2ba09fec222fa5f3ae687ed7e4d3112e9b926bc5e5b5d3366f55112a1f7b.jpg)

(c) Adult  
![](images/dae09ba2a0e01fee2c7d97278e871421c352e923ade34f04e84c09694be4b6c7.jpg)

(d) Credit  
![](images/87adcdee7c54661d4310df92773c6d8e742fac64e521ec9c89b59d5f91bf0106.jpg)

(e) Toxicity  
(f) Bias in Bios  
![](images/30a3abc77ab07fb9e0c53aeb52ab58e026be04f866406ef818e9e16e20585845.jpg)

(g) ACSIncome  
![](images/4dc42817098f32e0f0fdb7715734eb305bbda209ae8a18bf19d2a8b6f1abc03a.jpg)

![](images/3d854062c7aaf5de028a464bf73e6edf2cf4409525a9ad3eba8c36b208bb3a44.jpg)  
Accuracy regret per example (%)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin Individual arms Oracle frontier  
Figure 15: Trade-off between accuracy and fairness regret after $T = 1 0 , 0 0 0$ rounds, as in Figure 1.

## B.4 Cost

Query costs $c _ { a }$ are in the units of accuracy (%) and are subtracted from the utility (Eq. 1). We use Coupon with $\lambda = 1$ and five arms: four classifiers that use S as a feature, and the Biased arm (Appendix B.1.1). Without costs, RandomForest has the highest utility, just above XGBoost (Table 13). A cost of 5 for RandomForest and 1 for every other arm makes XGBoost the best arm by a clear margin.

TS follows the change (Figure 16). Without costs, it splits its pulls between the two near-tied arms, mostly RandomForest. With costs, it selects XGBoost in 93% of the last 25k rounds. In both settings, TS has significantly lower regret than every baseline (paired Wilcoxon signed-rank test, $p < 0 . 0 1 )$ .

Table 13: Coupon arms: accuracy, DP gap and utility $U _ { a }$ at λ = 1 (%), without and with costs $c _ { a }$ . Bold: best arm.
<table><tr><td></td><td></td><td></td><td>No cost</td><td colspan="2">With cost</td></tr><tr><td>Arm</td><td>Accuracy</td><td>DP gap</td><td> $U _ { a }$ </td><td> $c _ { a }$ </td><td> $U _ { a }$ </td></tr><tr><td>XGBoost</td><td>73.99</td><td>6.60</td><td>67.38</td><td>1</td><td>66.38</td></tr><tr><td>RandomForest</td><td>75.11</td><td>6.88</td><td>68.23</td><td>5</td><td>63.23</td></tr><tr><td>LogisticRegression</td><td>68.13</td><td>8.38</td><td>59.75</td><td>1</td><td>58.75</td></tr><tr><td>DecisionStump</td><td>61.67</td><td>0.36</td><td>61.31</td><td>1</td><td>60.31</td></tr><tr><td>Biased</td><td>50.57</td><td>69.88</td><td>-19.31</td><td>1</td><td>-20.31</td></tr></table>

![](images/05298fe2b8143fac6f225984bb7de0abf21f162e192ff97500b2f09a543050b8.jpg)  
Figure 16: Share of rounds in which TS selects each arm over rounds 25k–50k, averaged over 10,000 runs.

## B.5 Delayed Reward

We draw each label delay $L _ { t }$ from a geometric distribution with mean $\mathbb { E } [ L _ { t } ] \in \{ 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } \}$ ; labels that would arrive after $T = 5 0 { , } 0 0 0$ rounds are never observed. TS follows Algorithm 1, and ϵ-Greedy and Bayes-UCB use the same updates. Random and Round-Robin do not use feedback and are unaffected, and UCB1 is not included. Results are averages over 10,000 runs, as in Section 6.3.

Delays do not change the fairness regret of TS at $\lambda = \infty$ , and up to $\mathbb { E } [ L _ { t } ] = 1 0 ^ { 3 }$ they increase its accuracy regret at $\lambda = 0 \mathrm { b y }$ at most $0 . { \bar { 0 5 } }$ points (Figure 17). TS has the lowest regret at most values of λ on Twitter, Bank and Credit and at half or more on Adult; on Toxicity, ϵ-Greedy $( \epsilon = 0 )$ has lower regret at most values for $\mathbb { E } [ L _ { t } ] \geq 1 0 0$ , consistent with delayed labels acting as forced exploration for the greedy method.

![](images/ff1cbbea35cb0ec40921e1572efb3a0267fa1a237e2c9688aca4e14f3efafce4.jpg)  
Figure 17: Trade-off between accuracy and fairness regret after $T = 5 0 \mathrm { k }$ rounds with geometric label delays of mean $\mathbb { E } [ \bar { L } _ { t } ]$ (columns), using Algorithm 1. Round-Robin does not use feedback and is taken from the runs without delay; it is omitted where it is off-scale.

## B.6 Multi-Class Classification

The accuracy reward is still binary $( \hat { y } = y$ or not), so the accuracy posterior stays Beta-Bernoulli. For fairness, $\mu _ { a , s }$ is the distribution of arm a’s predicted classes in subgroup s, with a Dirichlet posterior over the C classes. We sample $\tilde { \mu } _ { a , 0 }$ and $\tilde { \mu } _ { a , 1 }$ and take as the gap their total variation distance, half their $L _ { 1 }$ distance [Xian et al., 2023]. For $C = 2$ this is the binary DP gap.

Bios (Figure 1(f), Appendix B.1.7) has LLMs as arms. For our second multi-class dataset, ACSIncome-5C, we use trained classifiers. We use the ACSIncome data of Appendix B.1.8 with sex as the sensitive attribute and income in five classes, as in Xian et al. [2023]: below \$15K, \$15K–30K, \$30K–48.6K, \$48.6K–78K, and at least \$78K. The 9 features are those of Appendix B.1.8, with race in place of sex. The 60/40 split is stratified on the class. The stream population has 78,266 instances, 47.0% of them women.

The 18 arms follow Appendix B.1.1, adapted to five classes:

• each operating point adds a fixed offset per class to the GBM’s log-probabilities, the same in both subgroups, and predicts the largest;

$\mathrm { L R - D P } \eta$ is multinomial logistic regression with the penalty $\eta \sum _ { k } ( \bar { p } _ { k , 0 } - \bar { p } _ { k , 1 } ) ^ { 2 }$ , where $\bar { p } _ { k , s }$ is the mean predicted probability of class k in subgroup s;

• GBM-BiasedLabels is trained after 40% of the lower-income subgroup’s labels above the lowest class are moved down one class;

• Biased predicts the highest class for one subgroup and the lowest for the other with probability 0.7, and a uniformly random class otherwise.

The 35 values of λ are {0, 0.0065, 0.0109, 0.0184, 0.0309, 0.052, 0.0585, 0.065, 0.0715, 0.0873, 0.147, 0.247, 0.415, 0.469, 0.521, 0.573, 0.698, 1.17, 1.97, 3.32, 5.57, 6.06, 6.73, 7.41, 9.37, 15.8, 19, 21.2, 23.3, 26.5, 44.5, 74.8, 126, 212, ∞}.

After $T = 5 0 \mathrm { k }$ rounds, TS has significantly lower regret than every baseline at all 35 values of λ (paired Wilcoxon signed-rank test, $p < 0 . 0 1 )$ . At $\lambda = \infty$ its fairness regret is 0.35%, against 1.55% for the next best method, ϵ-Greedy (ϵ = 0.1). ϵ-Greedy has lower regret at some values of λ only in the first $1 0 { , } 0 0 0$ rounds (Figure 18(a)–(c)).

(a) λ = 0  
![](images/4218036c18ad904579d1158e2653b2e9400636a87d15bfd75a2188b4b8d41420.jpg)

(b) λ = 1.17  
![](images/8a3cb5fcc43d67d68c9ca97a88916f17359f91649a3cb7d7316332d0603844cf.jpg)

(c) λ = ∞  
![](images/e76cd0cd0ca9413cb86c8fb7e2cf489e15e31f2e297dfc900bd56c81a5fcb249.jpg)

(d) T = 50k  
![](images/0c179c67552eb8435bf2af8f0e66995f5132bc24477ccac9e5cfed2f45fd34fa.jpg)  
Thompson Sampling Bayes-UCB UCB1 ε-Greedy (ε=0) ε-Greedy (ε=0.1) Round-Robin  
Figure 18: ACSIncome- $\cdot 5 \mathrm { C } \colon$ regret per example (%) against the round t at three values of λ (a)–(c), and after $T = 5 0 \mathrm { k }$ rounds against λ (d), as in Appendix B.1.

Table 14: The 18 arms of ACSIncome-5C: accuracy and TV gap (%) on the stream population. <sup>∗</sup>: optimal for some λ.
<table><tr><td>Arm</td><td>Accuracy</td><td>TV gap</td></tr><tr><td>GBM</td><td>53.17</td><td>9.37</td></tr><tr><td>RF</td><td>46.14</td><td>7.42</td></tr><tr><td>LR</td><td>51.75</td><td>8.85</td></tr><tr><td>Tree-d3</td><td>43.80</td><td>13.75</td></tr><tr><td>GBM+S*</td><td>53.61</td><td>14.80</td></tr><tr><td>GBM-OP1</td><td>49.13</td><td>8.00</td></tr><tr><td>GBM-OP2</td><td>41.96</td><td>6.69</td></tr><tr><td>GBM-OP3</td><td>37.51</td><td>5.35</td></tr><tr><td>GBM-OP4</td><td>34.22</td><td>3.93</td></tr><tr><td>GBM-OP5</td><td>31.86</td><td>2.67</td></tr><tr><td>GBM-OP6*</td><td></td><td></td></tr><tr><td></td><td>28.67</td><td>1.31</td></tr><tr><td>LR-DP3</td><td>51.33</td><td>6.08</td></tr><tr><td>LR-DP10</td><td>50.53</td><td>3.78</td></tr><tr><td>LR-DP30*</td><td>50.05</td><td>2.34</td></tr><tr><td>LR-DP100*</td><td>49.82</td><td>2.31</td></tr><tr><td>GBM-RW*</td><td>53.20</td><td>8.39</td></tr><tr><td>GBM-BiasedLabels</td><td>50.46</td><td>31.55</td></tr><tr><td>Biased</td><td>24.93</td><td>70.10</td></tr></table>

![](images/64faca8a1654943e2998eb0e15b239d217ce3a7374d7e69ff5e80c3f5e3533c1.jpg)  
Figure 19: Trade-off between accuracy and fairness regret on ACSIncome-5C after T = 50k rounds, as in Figure 1.

## B.7 Real-Valued Reward

Setup We use Adult with the same sensitive attribute as in the main experiments. Each arm outputs a risk score p rounded to the grid $\{ 0 , 0 . 0 5 , \ldots , 1 \} ( K = 2 0 )$ , so scores and rewards each take 21 values. The reward is $1 - ( Y - p ) ^ { \dot { 2 } } .$ one minus the Brier score, and the fairness gap is the DP gap of the mean scores (Section 4). The 17 arms are the 11 classifiers of the main Adult arm set that output probabilities, and 6 arms that shrink the gradient boosted trees’ score toward its mean by increasing amounts, ending with a constant predictor (zero gap). TS and ϵ-Greedy use the grid Dirichlet posteriors of Section 4, with one pseudo-count at 0 and one at 1, rather than continuous non-parametric TS. UCB1 adds Hoeffding radii to the empirical means of the rewards and scores, and Bayes-UCB uses quantiles of the same Dirichlet posteriors, computed with a saddlepoint approximation. As in the main experiments, results are averaged over 10,000 runs of $T = 5 0 { , } 0 0 0$ rounds, with 29 values of λ and no costs or delays.

![](images/d9e835f4368cb1f2d32c9570c47b9c11a0e4741d33337f21eab1eaa3aefdc4cc.jpg)  
Brier-score regret per example (%)  
Figure 20: Trade-off between Brier-score regret (x-axis) and fairness regret (y-axis) on Adult with risk-score arms after $T = 5 0 \mathrm { k }$ rounds, as average regret per example (%) over 10,000 runs. Each point on a curve is one value of λ. Grey crosses are the individual arms and the dotted line is the oracle frontier.

Results TS has the lowest regret at both ends of the trade-off (Figure 20); at $\lambda = \infty$ its fairness regret is 0.39%, against 1.32% for the next best method, ϵ-Greedy (ϵ = 0.1). At every value of λ, TS has significantly lower regret than UCB1, Bayes-UCB, ϵ-Greedy (ϵ = 0) and Round-Robin. ϵ-Greedy $( \epsilon = 0 . 1 )$ has lower regret than TS only for large λ near the value at which the optimal arm changes from a DP-penalized logistic regression to the constant predictor; there the two arms have nearly equal utility, so separating them requires many samples. UCB1 and Bayes-UCB stay far from the frontier at large λ, because their lower bound on the gap remains zero until the subgroup confidence intervals separate.

## B.8 $\pi _ { \mathrm { m i n } }$ Simulations

The regret bounds of Theorem 1 grow with $1 / \pi _ { \operatorname* { m i n } }$ . To test this dependence, each round draws an instance of the smaller subgroup with probability $\pi _ { \mathrm { m i n } }$ and of the larger subgroup otherwise, uniformly from the stream population. We use $1 / \pi _ { \mathrm { m i n } } \in \{ 2 , 5 , 1 0 , 2 0 , 3 0 , 5 0 , 1 0 0 \}$ and $\lambda = 2 .$ , and compare TS with Bayes-UCB and ϵ-Greedy $( \epsilon = 0$ and 0.1) over 10,000 runs of $T = 5 0 \mathrm { k }$ rounds. On real data, reweighting the subgroups also changes the arms’ utilities. We therefore add a synthetic instance with four arms whose accuracy is the same in both subgroups, so their utilities do not depend on $\pi _ { \mathrm { m i n } }$

The regret of TS increases with $1 / \pi _ { \mathrm { m i n } } ,$ consistent with Theorem 1, except for a small dip at $1 / \pi _ { \mathrm { m i n } } = 5 $ on Twitter and Credit (Figure 21). TS has significantly lower regret than all three baselines in 37 of the 56 settings. The exception is ϵ-Greedy $( \epsilon = 0 . 1 )$ at small $\pi _ { \mathrm { m i n } }$ on Twitter, Adult and COMPAS. There, few instances of the smaller subgroup arrive in $T = 5 0 \mathrm { k }$ rounds, and no method has converged.

(a) Twitter  
![](images/c220830702371bc1dde417663773f107e0ff6a805f9aab174f00ba6c106aa7a6.jpg)

(b) Bank  
![](images/509624f2fcc26498c8f41fb7ef40ecd36ebeaa59f7fcd297b6a775503a8276ab.jpg)

(c) Adult  
![](images/51f8e0a2335539c4108b6d5c2eb908da778e33a14b8c5bf8c898b8eb5d87ec9b.jpg)

(d) Credit  
![](images/d52813a12707319ac3f9cc7aed09cf71ef682fd278cb077a7238a8a8bc1f5240.jpg)

(e) Toxicity  
![](images/88798e8c55e8f4b5ece030dce5588c9431ac1039be202324d842e1d77df2892a.jpg)

(f) Coupon  
![](images/a10807bb9b90a3732673d591ef2a650a8b7e13a81886e8a3a7a85fc4a87d5f6e.jpg)

(g) COMPAS  
![](images/32432bd00a1f68f55d5a929034e622f7ef5de90da37d8a468cd4e6f6bef5cc18.jpg)

(h) Synthetic (fixed gaps)  
![](images/243defe98805830261f0c4418f533cbd16da9b3d016f534b0766e6a8a4ae1343.jpg)  
Figure 21: Regret per example after $T = 5 0 \mathrm { k }$ rounds $( \lambda = 2 )$ against $1 / \pi _ { \operatorname* { m i n } }$ , averaged over 10,000 runs. (a)–(g) Datasets of Appendix B.1; (h) synthetic instance.

## B.9 Fewer Arms

To test whether the comparison depends on the choice of arms, we draw 30 random subsets of 3 and 30 of 5 of the 18 tabular arms (Appendix B.1.1) and run each on Bank, Adult and Credit (400 runs, T = 50,000). The number of subsets that contain a biased arm (GBM-BiasedLabels or Biased) is fixed at its expected value. TS is compared with Bayes-UCB and both ϵ-Greedy variants.

Figure 22 shows, for each subset, the percentage of λ values at which TS has the lowest regret. For the median subset, TS has the lowest regret at most values of λ on Bank and Adult, and at fewer than half on Credit. The share is highest when the subset contains a biased arm. ϵ-Greedy (0.1) pulls a random arm in 10% of the rounds, so it keeps paying for the poor arm, while TS stops pulling it early. Without a biased arm, ϵ-Greedy (0.1) or Bayes-UCB often has lower regret: forced exploration costs little when all arms are close to optimal, and TS needs many pulls to separate arms whose accuracies and DP gaps nearly tie.

Figure 23 shows this hard case for three arms that are all vertices of the oracle frontier: GBM, GBM-OP4 and GBM-OP6 (10,000 runs; λ grid from the breakpoints of these arms). All learners stay close to the frontier. TS has the lowest regret at λ = 0 on all three datasets, but over all values of λ it is lowest at only 14% of them on Bank and 21% on Credit (72% on Adult), less often than for most random 3-arm subsets.

![](images/4d3d8c33cc43891fe1ba54748b562b5820099acdd3ca71590bcc68de3c220ea0.jpg)  
% of λ values at which TS has the lowest regret  
Subset without a biased arm Subset with a biased arm Median  
Figure 22: Percentage of λ values at which TS has a lower mean regret than Bayes-UCB, ϵ-Greedy (0) and ϵ-Greedy (0.1), for 30 random subsets of 3 and of 5 arms (one point per subset; 400 runs, T = 50,000). Filled points: the subset contains a biased arm. Bars: medians.

(a) Bank  
![](images/c2783b034dbde9e564ead38ac67790267944f13a8765f46008d7066a7a57f23d.jpg)

(b) Adult  
![](images/5f29aaa978f4db9b7a03f5187f81ae1f7b1600d9a90cc150bdbdbfa5e4f913c4.jpg)

(c) Credit  
![](images/8b1578339c229f29930ea6bd6da819a03d08610b579c7af32639a88dcfbdd8bc.jpg)  
Figure 23: Trade-off between accuracy and fairness regret after $T = 5 0 { , } 0 0 0$ rounds with three frontier arms (GBM, GBM-OP4, GBM-OP6), as in Figure 1. Means over 10,000 runs.