# Circuits of LLM Classifiers

Hadi Khalaf<sup>\*ab</sup> Sajani Vithana Kunal Talreja<sup>c</sup> Haewon Jeong<sup>de</sup> Viveck R. Cadambe<sup>c</sup> Flavio Calmon<sup>a</sup>

<sup>a</sup>Harvard School of Engineering and Applied Science <sup>b</sup>Kempner Institute for the Study of Natural & Artificial Intelligence <sup>c</sup>School of Electrical and Computer Engineering, Georgia Institute of Technology <sup>d</sup>Department of Electrical and Computer Engineering, University of California, Santa Barbara <sup>e</sup>Center for Computational Mathematics, Flatiron Institute

## Abstract

Language models make it easy to build classifiers by simply prompting a model with the sample to classify. However, their predictions are neither deterministic nor uniformly reliable across samples. How can we combine predictions from different models to improve reliability? We view LLM classifiers as noisy measurement instruments of an underlying target label. Our goal is to maximize average classification accuracy at a given average query cost. To this end, we introduce circuits that organize LLM classifiers into an adaptive sequence of queries and decisions. We then characterize the optimal circuit, obtain the fundamental limits of achievable accuracy, and provide practical insights on realizing the optimal circuit. Across seven classification tasks spanning safety, health, political science, human preference, agent evaluation, decision-making and reasoning, our method improves the cost–accuracy frontier over routing and cascading baselines. We provide our code here.

## 1 Introduction

Large language models (LLMs) are increasingly used in prediction and classification tasks as (auto-) raters (Vu et al., 2024), verifiers (Cobbe et al., 2021; Kwok et al., 2026), judges (Zheng et al., 2023; Chiang et al., 2024), and more. A main draw of LLMs as classifiers is their ease of implementation and deployment (Brown et al., 2020; Liu et al., 2023a). By simply specifying a new classification task in natural language, we can prompt these systems to assign labels in a range of applications such as content moderation (OpenAI et al., 2024), scientific review (Liu & Shah, 2023), guardrailing (Inan et al., 2023), and qualitative data analysis in social sciences (Halterman & Keith, 2025). Here, a “classifier” is defined by a combination of backend model choice, instruction prompt, grading rubric, decoding parameters (e.g., temperature, chainof-thought), and more. “Fitting” a classifier then boils down to navigating these choices during inference.

Consider first the task of designing a single LLM classifier. Even after fixing the model and instructions, repeated evaluations of the same prompt can return conflicting labels (Haldar & Hockenmaier, 2025; Wang et al., 2023a)<sup>1</sup>. It is therefore useful to view an LLM classifier as a noisy measurement instrument (Brennan, 1992; Kahneman et al., 2021). Naturally, if an instrument is noisy, we should collect more than one measurement with our tool, or measure with multiple (independent) instruments to control error. When using

![](images/67bb52d04b10244ede38e937005613b237dd3c00a5ccdbbcd5f016fa79facf8e.jpg)  
Figure 1: Classification circuits and their cost–accuracy tradeoffs. Majority voting, routing, and cascades are examples of circuits, differing in how they select and combine LLM classifiers. For any prompt, our method searches over a broad class of circuits and uses a dynamic program to determine which classifier to query next and when to stop. The right panel shows such circuits for the prompt z: starting from its embedding, the circuit may predict immediately (as in A) or sequentially acquire additional LLM measurements as needed (as in B and C). Across prompts, these decisions produce the cost–accuracy frontier for the BeaverTails safety dataset.

LLMs to classify a given prompt, this intuition suggests we can improve classification accuracy by collecting predictions from different models and instruction configurations. Of course, querying multiple models incurs a cost (e.g., API calls add up). A key challenge is deciding which classifier, if any, to query next as we collect measurements. These decisions create a tradeoff between accuracy and expected inference cost. In this paper, we ask:

## How should we combine LLM classifiers to optimize the cost-accuracy trade-off in a given task?

Strategies for combining classifiers, such as bagging (Breiman, 1996) and boosting (Schapire & Freund, 2013), are familiar in machine learning. More recently, this question has received growing attention in work on LLM routing (Ding et al., 2024; Ong et al., 2025), cascading (Chen et al., 2024b; Jung et al., 2025; Bouchard, 2026), and multi-judge evaluation (Verga et al., 2024; Turkmen et al., 2026). These methods show that combining multiple LLM classifiers outperforms any single model.

We refer to any procedure that queries one or more LLM classifiers and combines their outputs into a final decision as a classification circuit, or a circuit for short. The name reflects how these strategies use LLMs. The classifiers act as components, while decision rules determine how their outputs are passed forward, combined, or used to select the next query. Changing either the components or how they are connected produces a different circuit. For example, routing queries a single selected model, cascades query models sequentially in a fixed order, and voting schemes like majority voting query several models in parallel and aggregate their outputs. Different circuits using the same set of classifiers can therefore have different accuracy and cost, as shown in Figure 1.

Given the many ways of combining LLM classifiers, which circuit is best? What is the “optimal” circuit when samples come from a known distribution, such as a large safety classification dataset? What properties should we promote across classifiers to improve accuracy, such as independence or decorrelated errors? And what is the best achievable tradeoff between cost and accuracy?

We show that existing ways of combining LLM classifiers are facets of a common decision-theoretic formulation. This formulation maximizes average classification accuracy while minimizing the average cost, and is essentially a Bayesian sequential decision problem (Wald, 1992; DeGroot, 1962), closely related to sequential hypothesis testing (Chernoff, 1959; Naghshvar & Javidi, 2013). The fundamental challenge is to determine the value of an additional query given the observed measurements. For this formulation, an optimal circuit is given by the solution to a dynamic program.

Analyzing this dynamic program offers several insights into combining LLM classifiers. A first insight is that a query is valuable not only for its label, but also for what it reveals about how hard the sample is to classify. When classifiers disagree, the circuit learns to spend more queries on difficult prompts while resolving easier ones quickly. A second insight concerns how accuracy improves with additional query cost. For a fixed sample, asymptotically, the optimal error decreases with increasing cost at a rate determined by the most informative classifier per unit cost. A final, more obvious insight is that the error of any circuit lies between two limits: the Bayes error before any query is made and the Bayes error given the outputs of all classifiers. The latter is the accuracy ceiling for any way of combining a given set of classifiers.

Solving for this dynamic program requires a joint distribution between the ground truth label and instruments for a prompt. In practice, however, the joint distribution is unknown. We show that this optimal circuit can nevertheless be approximated in two steps. First, we fit a single probabilistic model of the joint distribution of the label and classifier outputs using a labeled training set. Then, we solve the dynamic program with the learned model, which results in a prompt-dependent policy that determines which LLM to query and when to stop. In a range of classification tasks, this two-step approach Pareto-dominates routing and cascading baselines in the cost–accuracy tradeoff.

Contributions. Our contributions are as follows:

• Optimal classification circuits. We introduce classification circuits as a general framework for combining LLM classifiers. Under a cost–accuracy tradeoff, we show that the optimal circuit for a given prompt can be solved exactly through a dynamic program.

• Properties of classification circuits. We identify the accuracy ceiling for a fixed pool of classifiers, quantify the value of adding complementary classifiers, and show how the distribution of prompt difficulty determines how average accuracy scales with cost.

• Learning circuits from data. We show how to approximate the optimal circuit from a labeled training set by learning the distributions required by the dynamic program from prompt embeddings and classifier outputs. Across seven classification tasks, circuits improve the cost–accuracy frontier over routing and cascading baselines.

## 2 Problem Formulation

Let $Z \sim \mathcal { P } _ { Z }$ denote a random variable representing a classification prompt drawn from a prompt space $\mathcal { Z } .$ In our applications, a prompt consists of the text to be classified together with task-specific information defining the classification problem. Each prompt z has a binary target label $y ( z ) \in \{ 0 , 1 \}$ . To predict $y ( z )$ for a realized prompt $Z = z$ , we query a subset of measurement instruments from a pool of available instruments denoted by H.

Definition 1 (Measurement instrument). A measurement instrument, indexed by $i \in \mathcal { H } .$ , is characterized by a conditional distribution $Q _ { X \mid Z } ^ { [ i ] }$ that takes a prompt z as input and outputs a measurement $X ^ { ( i ) } \in \mathcal { X } _ { i }$ . For a prompt z, instrument i has a query cost $c _ { i } ( z ) > 0 ^ { 2 }$

![](images/46f7025d632a228050aaef5c8143d115b7541b15a32f78708358b5e40d1117cb.jpg)  
Figure 2: Illustration of a few known circuits within the framework of Definition 2.

An instrument can correspond to a particular configuration of an LLM. For example, two instruments may differ in the underlying model, instruction prompt or rubric, reasoning configuration, or inference harness. Each such configuration defines a distinct distribution of the form in Definition 1.

We next study how to compose LLMs as measurement instruments into classification circuits that improve prediction accuracy. Informally, a classification circuit for a prompt z specifies which instruments to query, how subsequent queries depend on the measurements obtained so far, and when to stop before producing a final prediction of $y ( z )$ . Because this circuit may vary with the prompt, we use π to denote the circuit policy and $\pi ( z )$ for the circuit instantiated on $z . \mathrm { A }$ few example circuits are given in Figure 2.

Definition 2 (Classification circuit). Fix a finite instrument pool H, where querying instrument $i \in \mathcal { H }$ returns an outcome in $\mathcal { X } _ { i }$ . For $t \geq 0 .$ , let

$$
\mathcal T _ { t } = \Big \{ \big ( ( A _ { 1 } , x _ { 1 } ) , \dots , ( A _ { t } , x _ { t } ) \big ) : \emptyset \neq A _ { r } \subseteq \mathcal H , x _ { r } \in \{ 0 , 1 \} ^ { A _ { r } } , A _ { r } \cap A _ { s } = \emptyset \mathrm { ~ f o r ~ } r \neq s \Big \}
$$

denote the set of all histories of length t. Here $A _ { r }$ is the set of instruments queried in round $r , x _ { r }$ records their responses, and the disjointness constraint encodes that no item in H is queried twice. We set $\mathcal { T } _ { 0 } = \{ \emptyset \}$ and write $\begin{array} { r } { A ( h _ { t } ) = \bigcup _ { r \leq t } A _ { \iota } } \end{array}$ for the instruments already queried in $h _ { t } \in \mathcal T _ { t }$

A classification circuit is a pair $\pi = ( \pi _ { \mathsf { q u e r y } } , \pi _ { \mathsf { p r e d i c t } } )$ . The query policy $\pi _ { \mathsf { q u e r y } }$ is a family $\{ \pi _ { \mathsf { q u e r y } , t } \} _ { t \geq 0 }$ of maps

$$
\pi _ { \mathsf { q u e r y } , t } : \mathcal { Z } \times \mathcal { T } _ { t } \longrightarrow 2 ^ { \mathcal { H } \setminus A ( h _ { t } ) } , \qquad A _ { t + 1 } = \pi _ { \mathsf { q u e r y } , t } ( z , h _ { t } ) ,
$$

where $2 ^ { \mathcal { H } \backslash A ( h _ { t } ) }$ is the power set of the instruments left unqueried in $h _ { t } .$ . Thus the query policy determines, for a given prompt z, which instruments are queried in each round: starting from $h _ { 0 } = \emptyset ,$ , the process terminates if $A _ { t + 1 } = \emptyset$ , and otherwise queries the instruments in $A _ { t + 1 }$ , observes $x _ { t + 1 } \in \{ 0 , 1 \} ^ { A _ { t + 1 } }$ , and appends $( A _ { t + 1 } , x _ { t + 1 } )$ to the history $h _ { t }$

Call $h = \left( ( A _ { 1 } , x _ { 1 } ) , \ldots , ( A _ { \tau } , x _ { \tau } ) \right) \in \mathcal { T } _ { \tau }$ terminalfor z if $A _ { r + 1 } = \pi _ { \mathsf { q u e r y } , r } ( z , h _ { r } )$ for every $0 \leq r < \tau$ , where $h _ { r }$ is the length-r prefix of h, and $\pi _ { \mathsf { q u e r y } , \tau } ( z , h ) = \emptyset$ . Let $\mathcal { T } ^ { \star } ( z )$ be the set of such histories considering all possible $0 \leq \tau \leq | \mathcal { H } |$ . The decision rule is a map

$$
\pi _ { \mathrm { p r e d i c t } } : { \mathcal Z } \times T ^ { \star } ( z ) \longrightarrow \{ 0 , 1 \} , \qquad \widehat { Y } _ { \pi } ( z ) = \pi _ { \mathrm { p r e d i c t } } ( z , h ) .
$$

Remark 1. Definition 2 permits repeated independent queries to the same instrument. In particular, items $i , j ~ \in ~ { \mathcal { H } }$ may correspond to two independent queries of the same instrument, with $Q _ { X | Z } ^ { [ i ] } = Q _ { X | Z } ^ { [ j ] }$ and $c _ { i } ( z ) = c _ { j } ( z )$ . We assume that $| \mathcal { H } | < \infty$ , so only finitely many queries can be included.

We focus on the binary setting throughout this work, i.e., $\mathcal { X } _ { i } = \{ 0 , 1 \}$ for all $i \in \mathcal { H }$ . For a fixed prompt $Z = z ,$ , executing a circuit $\pi ( z )$ in Definition 2 produces the collected measurements $h = ( A _ { i } , x _ { i } ) _ { i \leq \tau }$ for some τ. Let $\mathcal { A } = \cup _ { i \leq \tau } A _ { i } \subseteq \mathcal { H }$ denote the subset of instruments queried by $\pi ( z )$ and let $X \in \{ 0 , \overset { - } { 1 } \} ^ { | \mathcal { A } | }$ be the corresponding measurements. Then, the circuit’s prediction on the target label is given by $\hat { Y } _ { \pi } ( z ) =$ $\pi _ { \sf p r e d i c t } ( z , \boldsymbol { X } )$ . Over the prompt distribution, the probability of error of a circuit policy π is

$$
\mathsf { E } ( \pi ) = \operatorname* { P r } \left( \hat { Y } _ { \pi } ( Z ) \neq y ( Z ) \right) = \mathbb { E } _ { Z } \left[ \mathbb { E } _ { X \mid Z } \left[ \mathbf { 1 } _ { \left\{ \hat { Y } _ { \pi } ( Z ) \neq y ( Z ) \right\} } \middle | Z \right] \right] ,\tag{1}
$$

and the expected query cost is given by

$$
\mathsf { C } ( \pi ) = \mathbb { E } _ { Z } \left[ \mathbb { E } _ { X \mid Z } \Bigl [ \sum _ { i \in \mathcal { A } } c _ { i } ( Z ) \mid Z \Bigr ] \right]\tag{2}
$$

where the inner expectation is over the instrument measurements, conditional on a fixed prompt.

Since each LLM query uses fresh randomness, we model measurements as independent conditional on a prompt.

Assumption 1 (Conditional independence of measurements). Conditional on $Z = z ,$ , the measurements $\{ X ^ { ( i ) } \} _ { i \in \mathcal { H } }$ are jointly independent. Thus, for any $( x _ { i } ) _ { i \in \mathcal { H } }$

$$
\operatorname* { P r } \Big ( X ^ { ( i ) } = x _ { i } , \forall i \in { \mathcal { H } } \mid Z = z \Big ) = \prod _ { i \in { \mathcal { H } } } Q _ { X \mid Z } ^ { [ i ] } ( x _ { i } \mid z ) .
$$

At first glance, this may seem to rule out correlated errors observed across language models (Kim et al., 2025). The distinction is that independence holds within a prompt, whereas error correlation is measured across prompts. We also validate this empirically in Figure 7.

## 3 Optimizing Classification Circuits

Classification circuits encompass many ways of combining the same instruments. We study one particular criterion for choosing among them, the expected cost–accuracy tradeoff, and show how to construct the resulting circuits for any prompt. We present all proofs in Section B.

Our goal is to find a circuit policy π that best trades off classification error against query cost. We control this tradeoff with a parameter $\lambda \geq 0$ that weights cost relative to error, and define the objective ${ \mathcal { L } } _ { \lambda } ( \pi ) =$ $\mathsf E ( \pi ) + \lambda \mathsf C ( \pi )$ . For a given $\lambda ,$ the optimal policy is

$$
\pi _ { \lambda } ^ { \star } \in \arg \operatorname* { m i n } _ { \pi \in \Pi } \mathcal L _ { \lambda } ( \pi )\tag{3}
$$

where Π denotes the set of all circuit policies described in Definition 2. We first show that the class Π can be restricted to a smaller class without loss of optimality with respect to this objective.

Proposition 1. Any circuit that queries multiple instruments simultaneously has a sequential counterpart that achieves the same probability of error in Equation (1) at the same cost in Equation (2). Hence an optimal circuitfor Equation (3) can always be chosen to query one instrument at a time.

The optimal solution to Equation (3) can hence be viewed as a sequence of decisions. Among all possible sequential circuits, the dynamic program resulting from the Bellman equations (Bellman, 1966) gives the optimal solution to Equation (3).

## 3.1 Optimal Circuit Setup

Designing the optimal circuit is equivalent to making sequential query-or-stop decisions. At each step, given the responses observed so far, the policy either selects the next model to query or stops and predicts. The resulting policy defines the optimal circuit. We first introduce the objects that appear in the dynamic program: 1) the state of a circuit, 2) the actions available in each state, and the transitions between states.

State. Fix a prompt z. At any point during the execution of a circuit, the information gathered so far is the set of (instrument, response) pairs observed,

$$
s = \{ ( j , x ^ { ( j ) } ) : j \in \mathcal { M } ( s ) \} ,\tag{4}
$$

where $\mathcal { M } ( s ) \subseteq \mathcal { H }$ is the set of instruments queried so far. We call s the state, with $\boldsymbol { \mathcal { S } }$ denoting the set of all possible states. $\vert { \mathcal { S } } \vert = 3 ^ { \vert { \mathcal { H } } \vert }$ since we assume that each instrument indexed $i \in \mathcal { H }$ can only be queried once <sup>3</sup>. $s _ { 0 } = \emptyset$ denotes the initial state in which nothing has been queried. Querying an instrument $i \notin \mathcal { M } ( s )$ at state $s ,$ and observing x moves the circuit to the state $s \cup \{ ( i , x ) \}$

Actions. In a state $s ,$ the circuit chooses one of two actions: (a) STOP, and output the Bayes prediction $\hat { y } _ { \pi } ( z ) = \arg \operatorname* { m a x } _ { y \in \{ 0 , 1 \} } P _ { D } ( Y = y \mid z , s )$ , where $Y$ denotes the unobserved label $y ( z )$ , and $P _ { D }$ denotes the probabilistic model used to estimate the posterior belief, or (b) QUERY an instrument i $\neq \mathcal { M } ( s )$ at cost $c _ { i } ( z )$ , observe its response $x ^ { ( i ) }$ , and move to the state $s \cup \{ ( i , x ^ { ( i ) } ) \} . \mathrm { I f } \mathcal { M } ( s ) = \mathcal { H }$ , only STOP is available.

For a fixed prompt z at state s, the circuit’s posterior belief on the target label $y ( z ) = 1$ is given by,

$$
q _ { z } ( s ) = P _ { D } \left( Y = 1 \mid z , s \right)\tag{5}
$$

If the circuit stops at $s ,$ the optimal prediction is the more likely label, with Bayes error

$$
e _ { z } ( s ) = \operatorname* { m i n } \{ q _ { z } ( s ) , 1 - q _ { z } ( s ) \} .\tag{6}
$$

For a given $\lambda \geq 0$ , let $V _ { \lambda , z } ( s )$ denote the minimum ${ \mathcal { L } } _ { \lambda } ( \pi )$ (for prompt z) at state s. The Bellman equations (Bellman, 1966), together with Assumption 1, characterize $V _ { \lambda , z } ( s )$ as:

$$
V _ { \lambda , z } ( s ) = \operatorname* { m i n } \left\{ \underbrace { e _ { z } ( s ) , \operatorname* { m i n } _ { i \notin \mathcal { M } ( \{ s \} } ) } _ { \mathrm { p r e d i c t } } \underbrace { \left[ \lambda c _ { i } ( z ) + \sum _ { x \in \{ 0 , 1 \} } Q _ { X | Z } ^ { [ i ] } ( x | z ) V _ { \lambda , z } \big ( s \cup ( i , x ) \big ) \right] } _ { \mathrm { q u e r y } ( i ) } \right\}\tag{7}
$$

The first term in Equation (7) is the terminal error for stopping and predicting the label at state s. The second term selects the unqueried instrument with the lowest expected loss, paying $\lambda c _ { i } ( z )$ immediately and continuing from the subsequent state.

$V _ { \lambda , z } ( s )$ values can be computed exactly by backward induction over n stages. For a fixed prompt z and weight λ, solving Equation (7) yields the optimal circuit policy: 1) which models to query, 2) in what order, and 3) when to stop. This optimal policy is the solution $\pi _ { \lambda } ^ { * }$ in Equation (3).

## 3.2 Generating circuits for unseen prompts: LMCIRCUIT

Notice that the optimal policy is determined entirely by the label posterior at each state $q _ { z } ( s )$ and the instrument output distributions $Q _ { X \mid Z } ^ { [ i ] }$ for a given prompt z. This suggests a two-step approach for solving Equation (7): (i) estimate the probabilities $q _ { z } ( s ) = P _ { D } ( Y = 1 | s , z ) , \forall s \in \mathcal { S }$ and $Q _ { X | Z } ^ { [ i ] } ( \cdot | z )$ , for a given z from a training set and (ii) plug the estimates into the dynamic program solver. The training set contains prompts, their ground truth labels, and measurements from instruments in H. To estimate the probabilities, we represent each prompt z by an embedding $\phi ( z ) \in \mathbb { R } ^ { d }$ and fit a model $f _ { \theta }$ with outputs,

$$
f _ { \theta } ( \phi ( z ) , s ) = \Big ( \eta _ { \theta } ( \phi ( z ) , s ) , \nu _ { \theta } ^ { ( 1 ) } ( \phi ( z ) , s ) , \dots , \nu _ { \theta } ^ { ( n ) } ( \phi ( z ) , s ) \Big ) , \quad \forall s \in S
$$

![](images/f6e5c2a49a6967e891f5609b090785352551897b79c5b47bd509a49d40f88df0.jpg)  
Figure 3: Generating circuits. We first fit a probabilistic model from labeled prompts and instrument measurements (Section 3.2). At inference time, a prompt embedding is passed through this model, and a dynamic program generates a circuit that adaptively queries instruments and decides when to stop (Equation (7)).

where $\eta _ { \boldsymbol \theta } ( \phi ( z ) , s ) ~ \approx ~ \operatorname* { P r } ( Y = 1 \mid \phi ( z ) , s )$ is an estimate of the label posterior, while $\nu _ { \theta } ^ { ( i ) } ( \phi ( z ) , s )$ ≈ $\operatorname* { P r } \bigl ( X ^ { ( i ) } = 1 \mid \phi ( z ) , s \bigr )$ is an estimate of the instrument probabilities. We then use the above estimates to compute the dynamic program for any new prompt. Since the distributions do not depend on $\lambda ,$ the same fitted model generates circuits that operate at different points of the cost–accuracy frontier.

Remark 2 (Zero-cost prediction). At the initial state $s _ { 0 } = \emptyset$ , the LMCIRCUIT can give a prediction on the output label for a prompt z without querying any instrument, i.e., at zero cost, with $\operatorname* { P r } ( Y = 1 \mid z ) \approx$ $\eta _ { \theta } ( \phi ( z ) , s _ { 0 } )$ , where $\eta _ { \theta } ( \phi ( z ) , s _ { 0 } )$ is an output of $f _ { \theta } ( \phi ( z ) , s _ { 0 } )$

The fitted model $f _ { \theta }$ can then run completely on CPU. Hence, it provides a practical and lightweight way to organize how inference is allocated and optimized over LLM classifiers.

## 4 Analysis of the Optimal Circuit

This section studies the properties of optimal circuits. For a fixed instrument pool H, we first characterize the two endpoints of the error–cost tradeoff: (i) when it is optimal to make no queries, and (ii) what the best possible accuracy is using all the instruments. We then ask how adding one additional instrument to an existing pool improves this maximum accuracy, which yields practical guidance for assembling a pool of instruments. Finally, we derive scaling laws for circuits, i.e., the rate at which the error of the optimal circuit decreases with query cost on a prompt, and how this rate changes once averaged over a population of prompts with varying difficulty. All proofs are provided in Section B of the appendix.

## 4.1 Fundamental Limits of Circuits for a Fixed and Finite Pool

In this section, we consider the two extreme circuits for a given prompt z and an instrument pool H: 1) Zero-cost circuit, where no instrument is queried and the output is predicted based on the prior belief, and 2) Full-cost circuit, where all instruments in H are queried.

For a prompt z, consider the circuit $\pi _ { 0 } ( z )$ that stops at state $s _ { 0 } = \emptyset , \mathrm { i . e . }$ ., before any instrument is queried. It then predicts the more likely label according to its prior belief $q _ { z } ( \emptyset ) = P _ { D } ( Y = 1 \mid z )$ , and incurs the Bayes error $\begin{array} { r } { \mathsf { E } _ { \pi _ { 0 } } ( z ) = \operatorname* { m i n } _ { y \in \{ 0 , 1 \} } P _ { D } ( Y = y \mid z ) } \end{array}$

Proposition 2 (Optimality of the zero-cost circuit). The zero-cost circuit $\pi _ { 0 } ( z )$ is optimal for prompt $z \ i f$ and only $\begin{array} { r } { i f \lambda \geq \lambda _ { \mathrm { L B } } ( z ) : = \operatorname* { s u p } _ { \pi : \mathsf C _ { \pi } ( z ) > 0 } \frac { \left[ \mathsf E _ { \pi _ { 0 } } ( z ) - \mathsf E _ { \pi } ( z ) \right] _ { + } } { \mathsf C _ { \pi } ( z ) } } \end{array}$ . Letting $\begin{array} { r } { c _ { \operatorname* { m i n } } ( z ) : = \operatorname* { m i n } _ { i \in \mathcal { H } } c _ { i } ( z ) } \end{array}$ , we have that $\begin{array} { r } { \lambda \ge \frac { 1 } { 2 c _ { \operatorname* { m i n } } ( z ) } } \end{array}$ is sufficient for π<sub>0</sub> to be optimal.

Proposition 2 states that for all λ exceeding a certain threshold, it is optimal to make no queries. At the other endpoint, $\lambda = 0$ , the circuit uses all measurements in H to maximize accuracy, and its minimum error is the Bayes error after observing the full instrument pool. Let $\ b X ^ { ( \mathcal { H } ) }$ denote the measurements obtained by querying every instrument in H.

Proposition 3 (Minimum error for full-cost circuit). For a prompt z and instrument pool H, the minimum achievable error is

$$
\mathsf { E } _ { \operatorname* { m i n } } ( z , \mathcal { H } ) = \mathbb { E } _ { { \pmb X } ^ { ( \mathcal { H } ) } | Z = z } \left[ \operatorname* { m i n } _ { y \in \{ 0 , 1 \} } P _ { D } \Big ( Y = y \Big | z , \pmb X ^ { ( \mathcal { H } ) } \Big ) \right]\tag{8}
$$

As a result, $\mathsf { E } _ { \mathrm { m i n } } ( z , \mathcal { H } )$ is a lower bound on the error of any circuit constructed from H for any λ. This limit allows us to answer how much performance can improve when another instrument becomes available. Denote this instrument by $i _ { \mathsf { n e w } } .$ , and let the corresponding distribution generating observations be $Q _ { X | Z } ^ { [ i _ { \mathsf { n e w } } ] } ( \cdot | z )$ Then, the new pool is denoted by $\mathcal { H } _ { \sf n e w } = \mathcal { H } \cup i _ { \sf n e w }$ . The next result characterizes the value of adding a new instrument to the existing pool.

Proposition 4 (Gain from adding one extra instrument). Over $Z \sim { \mathcal { P } } _ { Z } ,$

$$
\mathbb { E } _ { Z } [ \mathsf { E } ( Z , \mathcal { H } ) - \mathsf { E } ( Z , \mathcal { H } _ { \mathrm { n e w } } ) ] \leq \sqrt { \frac { 1 } { 2 } I \big ( Y ; X ^ { ( i _ { \mathrm { n e w } } ) } \big | X ^ { ( \mathcal { H } ) } , Z \big ) }\tag{9}
$$

where the mutual information is computed under the joint distribution of $( Y , X ^ { ( \mathcal { H } ) } , X ^ { ( i _ { \mathrm { n e w } } ) } , Z )$

What makes a new instrument useful? Its value is not determined by the instrument’s accuracy but by the information it provides about the target beyond the measurements already available.

Practical significance of Proposition 4. To generate the circuit for any prompt z, we use a labeled training set to estimate the joint distribution of $( Y , { \bar { \pmb X } } ^ { ( \mathcal { H } ) } )$ , conditional on z. As the pool size |H| increases, this joint distribution becomes more difficult to estimate, which can degrade the performance of the resulting dynamic program. At the same time, a larger pool provides more opportunities to obtain complementary measurements, enabling more powerful circuits for a wider range of prompts. Pool construction must therefore balance the additional information supplied by new instruments against the statistical cost of estimating their joint behavior. Proposition 4 suggests a greedy strategy for constructing a task-specific instrument pool. Given a candidate pool of N instruments, we can initialize H with a low-cost instrument and iteratively add the candidate that maximizes the conditional-information bound in Equation (9), until the additional gains are non-significant. The resulting pool can be used in the dynamic program.<sup>4</sup> Figure 12 shows that greedy selection by conditional information reaches the full-pool frontier with 2–3 instruments on the four tasks shown, and dominates accuracy-first, cheapest-first, and random ordering methods of building the pool.

## 4.2 Scaling Laws for Classification Circuits

In this section, we characterize the resulting relationship between optimal error and expected query cost through its error–cost scaling law. In particular, we study how the error probability decreases with query cost as $\lambda \to 0$ and consider its implications in a given population of prompts.

For each instrument $i \in \mathcal { H } ,$ , its accuracy on prompt z is $Q _ { X \mid Z } ^ { [ i ] } ( y ( z ) \mid z )$ , which we denote by $a _ { i } ( z )$ . For the asymptotic analysis, we make the following assumptions: (1) each instrument can be queried arbitrarily many times, (2) every instrument is at least as accurate as random guessing on every prompt, $\mathrm { i . e . , } a _ { i } ( z ) \in$ $[ 1 / 2 , 1 )$ for all $i \in \mathcal { H }$ and $z \in { \mathcal { Z } } .$ , and (3) the response distribution is symmetric with respect to the groundtruth label: conditional on $H _ { y } : y ( z ) = y .$ , an independent call to instrument i returns the correct label y with probability $a _ { i } ( z )$ . While the optimal query and stopping decisions continue to satisfy the Bellman equation in Equation (7), they can no longer be realized using backward induction in this asymptotic regime.

For a fixed prompt z, it is useful to view the sequential decision problem as a hypothesis test between $H _ { 0 } : y ( z ) = 0$ and $H _ { 1 } : y ( z ) = 1$ . Each collected measurement provides information about which label is correct. A measurement from instrument i is distributed as Ber $\cdot ( 1 - a _ { i } ( z ) )$ under $H _ { 0 }$ and $\mathrm { B e r } ( a _ { i } ( z ) )$ under $H _ { 1 }$ . Then, the KL-divergence $d _ { i } ( z ) = D _ { \mathrm { K L } } ( \mathrm { B e r } ( a _ { i } ( z ) )$ ∥ Ber $\left( 1 - a _ { i } ( z ) \right) )$ quantifies the expected information provided by one measurement for distinguishing the two hypotheses. Because instruments differ in cost, we normalize this quantity and define the information rate of instrument i to be: $\begin{array} { r } { I _ { i } ( z ) : = \frac { d _ { i } ( z ) } { c _ { i } ( z ) } } \end{array}$ and maximum information rate of the model pool to be: $I ( z ) : = \operatorname* { m a x } _ { i \in { \mathcal { H } } } I _ { i } ( z )$ . Thus $I ( z )$ measures the largest information rate available for classifying prompt z.

For a fixed prompt z, let $\mathsf E _ { \lambda } ( z )$ and $\mathsf C _ { \lambda } ( z )$ denote the error probability and expected query cost of the optimal circuit for parameter λ. As λ approaches zero, the circuit places progressively less weight on query cost and can spend more to reduce its error. For $0 < I ( z ) < \infty$ , the optimal error approaches zero, but doing so requires an expected cost that grows without bound. This raises the question of how quickly error decreases as more cost is incurred. The following result shows that the decay is exponential with cost, at rate $I ( z )$

Theorem 1 (Scaling law for a fixed prompt). For every prompt z with $0 < I ( z ) < \infty ,$ , we have that

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } { \frac { - \log \mathsf E _ { \lambda } ( z ) } { \mathsf C _ { \lambda } ( z ) } } = I ( z ) .\tag{10}
$$

The information rate also determines how the optimal circuit allocates its queries.

How does the optimal circuit allocate its queries? As $\lambda \downarrow 0 .$ , the optimal circuit spends nearly all of its expected cost on instruments with the largest information rate, $I _ { i } ( z ) = I ( z )$ ; see Proposition 5.

For a fixed λ, the optimal next query can still depend on the measurements collected so far and on its cost, so the optimal circuit may query several instruments. However, as $\lambda \downarrow 0$ , the fraction of expected query cost spent on instruments with $I _ { i } ( z ) < I ( z )$ tends to zero.

This result characterizes the optimal circuit for a fixed prompt as $\lambda \downarrow 0$ . The information rate $I ( z )$ determines the rate of error decay. It then offers a natural notion of prompt “difficulty”: a prompt with smaller $I ( z )$ requires more query cost to attain the same error level.

Now consider a population of prompts $Z \sim \mathcal { P } _ { Z }$ , and let $\mathsf { E } _ { \lambda }$ and $\mathsf { C } _ { \lambda }$ denote the average error probability and expected query cost of the optimal circuit. While Theorem 1 establishes exponential error decay for each prompt with $I ( z ) > 0$ , the population may contain prompts on which every instrument performs only slightly better than random guessing. Thus, information rates can be arbitrarily close to zero across the population. The population error–cost tradeoff therefore depends on the distribution of these rates, as characterized by the following theorem.

Theorem 2 (Scaling law across prompts). Assume that, for almost every prompt z, the prior Bayes error $\mathsf { E } _ { \pi _ { 0 } } ( z )$ is bounded below by a uniform positive constant, query costs are uniformly bounded above, and $0 < I ( z ) < \infty$ . Then

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \frac { - \log \mathsf E _ { \lambda } } { \mathsf C _ { \lambda } } = \left( \mathbb E \left[ \frac { 1 } { I ( Z ) } \right] \right) ^ { - 1 } ,\tag{11}
$$

where the right-hand side is zero ifthe expectation is infinite.

An important consequence is that exponential error decay for almost every prompt need not imply exponential decay with average cost across the population. To characterize the resulting error–cost tradeoff even further, we can consider how the probability of small information rates decreases as they approach zero. Corollary 1 (Scaling law across prompts under a tail assumption). Under the conditions of Theorem 2, suppose that, for some $\alpha > 0$

$$
\operatorname* { P r } ( I ( Z ) \leq r ) \asymp r ^ { \alpha } \qquad \mathrm { a s } r \downarrow 0 ,
$$

where $r > 0 ^ { 5 }$ . In this case, $\mathbb { E } [ 1 / I ( Z ) ]$ is finite for $\alpha > 1$ . Then, as $\lambda \downarrow ($ 0:

(i) $\mathrm { I f } \ \alpha > 1$ , the error decays exponentially with cost, at the rate given in Theorem 2.

(ii) If $\alpha = 1$ , the error decays at a stretched-exponential rate, $- \log \mathsf E _ { \lambda } \asymp \sqrt { \mathsf C _ { \lambda } }$

(iii) If $0 < \alpha < 1$ , the error decays polynomially, $\mathsf E _ { \lambda } \asymp \mathsf C _ { \lambda } ^ { - \frac { \alpha } { 1 - \alpha } }$

The results point to an obvious limit of inference-time computation: even an optimal circuit cannot compensate for a lack of informative measurements. Improving the instruments on the hardest prompts can therefore change the scaling law itself.

## 5 Experiments

Setup. We consider seven tasks: AgentJudgeBench (Verma et al., 2026) for judging whether an AI agent’s planned tool calls are correct, BeaverTails (Ji et al., 2023) for safety classification, BIG-Bench Hard (Suzgun et al., 2023) for reasoning questions, Decision Index (Multimodal Art, 2026) for decision tasks, Health-Bench (Arora et al., 2025) to evaluate whether a text satisfies a physician’s rubric, LMArena (Chiang et al., 2024) for classifying human rater preference, and PolitiCause (Garcia-Corral et al., 2024) for detecting causality in political text. For each task, a training set is used to estimate the LMCIRCUIT (as described in Section 3.2) required by the dynamic program. We then vary the parameter λ on a validation set to obtain circuits at different operating points and report test accuracy against average query cost. Within each task, costs are normalized relative to the most expensive instrument. Additional details are in Section C.

We benchmark against (1) Predictive routers, including ROUTELLM (Ong et al., 2025), HYBRIDLLM (Ding et al., 2024), and CARROT (Somerstep et al., 2025); (2) Model cascades, including CASCADE ROUTING (Dekoninck et al., 2025), CONSISTENCY CASCADE (Yue et al., 2024), CASCADED SELECTIVE (Jung et al., 2025), and FRUGALGPT (Chen et al., 2024b); (3) Sampling methods, namely ADAPTIVE-CONSISTENCY (Aggarwal et al., 2023); and (4) individual instruments and majority voting.

## 5.1 Results

LMCIRCUIT improves the cost–accuracy frontier. Across the seven tasks, the learned circuits attain higher accuracy than every individual instrument and than the routing and cascade baselines at nearly all budgets (Figure 4). The choice of instruments that the circuit uses depends on the task and is not predictable from cost or accuracy alone. On LMArena, the two mid-cost Gemma models (26B and 31B) are the most informative per unit cost, and the circuit queries almost exclusively these two for most budgets. On Beaver-Tails, the cheapest model, Gemma 4B, carries the most information per unit cost (Figure 11). Hence, the circuit queries it on most prompts and reserves the expensive models for the highest budgets.

<table><tr><td></td><td>Individual judge</td><td>LMCircuit (Ours)</td><td>★</td><td>Majority voting</td><td>RouteLLM</td><td></td><td>HybridLLM</td><td>FrugalGPT</td></tr><tr><td></td><td>Cascaded Selective</td><td>Cascade Routing</td><td>---</td><td>CARROT</td><td></td><td>Consistency Cascade</td><td>…</td><td>Adaptive-Consistency</td></tr></table>

![](images/a358cb80199b717db5f4026ef1482eeffd09778cd2c5d2519885bf66c67654e2.jpg)  
Figure 4: LMCircuit improves the cost-accuracy frontier over routing and cascading baselines. As λ decreases and additional queries become less costly, the circuits make more calls and also change which instruments they use (see lower panel). The allocations differ substantially across tasks, indicating that the value of an instrument depends on the classification problem and on other measurements available to the circuit.

![](images/3e08379cd745951f24ad8347a1b9b0174a748d90411e250a95633c4bf5e4b1c9.jpg)

![](images/a3bc8acb1e0337bcdd4871af40df03a666ab8e732abbbdfad76d14f273a42687.jpg)

![](images/d5c4e0f8973780c87ecab5183e28a5ee3186425a6b92417bc9184096e3c6dd6e.jpg)

![](images/a0c2d2440f10e6ae79be2e9312002ffd8a4681f2eb5361c327672828d9ced49d.jpg)

![](images/23b1c9743426764983bb416d2fece6dbb4a00f8db4624ac009fb89f83fd05efb.jpg)

![](images/53acced76e1c2db929661bd965530df56de5caf74bcac9803fe6226b0894ae2a.jpg)

![](images/ae45c1a7262c51d4ce813f1ac62bdf070a6f0367b2c4f1eb2ca52647af283fd4.jpg)  
Figure 5: LMCircuit is substantially faster to construct than most routing and cascading baselines. Across all seven tasks, LMCircuit is trained in 1–3 minutes, while some baselines require minutes to hours.

LMCIRCUIT provides calibrated decisions. For a given prompt z, the distribution estimator outputs an estimate of $\operatorname* { P r } ( Y = 1 \mid z )$ . Based on its confidence, LMCIRCUIT assesses whether the prompt is easy or hard, selects the instrument path accordingly, and updates its posterior belief after each measurement. As expected, under the same cost-accuracy parameter λ, harder prompts require more model calls (Figure 10). An advantage of our method is that LMCIRCUIT’s prior alone can yield a label prediction without querying any instrument. Therefore, at the low-cost end of the frontier, the average query cost can fall below that of the cheapest individual instrument. Figure 6 further shows that LMCIRCUIT produces better calibrated decisions on unseen data.

LMCIRCUIT learns whether more queries help and how reliable each instrument is. For any given prompt, initially, the circuit predicts its label from the prompt embedding alone, without querying any instrument (Remark 2). We call this the embedding prediction. To investigate the impact of LLM calls on top of embedding predictions, we organize unseen prompts by the fraction of recorded calls that return the correct label (Figure 9). On noisy prompts, where this fraction lies between (1/2, 1), the circuit at budget 1 improves the accuracy of embedding prediction by 5%–44%. On prompts where every call is correct, it matches majority voting (within 1.3% accuracy difference) while using 38%–86% fewer calls.

The fitted model of LMCIRCUIT is accurate about the instruments. The dynamic program relies on the fitted model’s estimates of how each instrument responds on a given prompt. Figure 8 shows that, without access to test labels, the fitted model predicts 1) each instrument’s accuracy on unseen prompts, 2) the correlations of the instruments’ answers and errors across prompts. Across the seven tasks, the implied correlations between instruments’ answers, and between their errors, are within 4%–17% of the measured values, and the implied accuracy of each instrument is within 1.2%–3.6% of its measured accuracy.

LMCIRCUIT is easy and fast to implement. Our method is less complex than many baselines (Figure 5), attaining significantly shorter training times. The estimation step uses a simple neural network together with pretrained embeddings of prompts. We do not require token probabilities or training larger models (such as DistilBERT or DeBERTa) for each instrument as in other methods, and the dynamic program is cheap for moderate pools of models. We note, however, that the dynamic program grows exponentially with the number of models. In many applications, this is less restrictive than it may seem, since a small, well-chosen pool of classifiers is often sufficient (see Section 4.1 for guidance on model pool selection).

Types of noise in measurements: A measurement instrument is subject to two types of noise: (1) stochastic noise, errors that can be reduced through repeated or correlated measurements, and (2) systematic noise, errors that are inherent to how the instrument was trained. Our results show that the dynamic program automatically adapts to both. It selects highly correlated models to reduce (1) when queried models are moderately accurate (∼75–80%, Figure 13), and less correlated models to reduce (2) when queried models are less accurate (∼63–66%, Figure 14).

## 6 Conclusion

As LLM classifiers become components of larger systems, a practical question is how inference should be organized and allocated among them to maximize accuracy at the lowest expected cost. We take a decision-theoretic view of this problem, using a lightweight probabilistic model to characterize the available classifiers and a dynamic program to decide what to query and when to stop. We are then able to generate circuits for any prompt while offering control on the error–cost tradeoff. One challenge is to extend this to larger pools of general measurement instruments with potential feedback among them. More broadly, our framework can broaden access to high-quality and automated decision-making for practitioners without frontier-model budgets.

Acknowledgments This work was supported in part by the National Science Foundation under Grant No: 2506573, CIF-2231707, CIF-2312667, and a Gift by Coefficient Giving. Hadi Khalaf is supported by a Kempner Graduate Research Fellowship which has been made possible by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University. The Flatiron Institute is funded by the Simons Foundation.

## References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Pranjal Aggarwal, Aman Madaan, Yiming Yang, et al. Let’s sample step by step: Adaptive-consistency for efficient reasoning and coding with llms. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 12375–12396, 2023.

Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, et al. Automix: Automatically mixing language models. Advances in Neural Information Processing Systems, 37:131000–131034, 2024.

Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, and Karan Singhal. Healthbench: Evaluating large language models towards improved human health, 2025.

Richard Bellman. Dynamic programming. Science, 153(3731):34–37, 1966.

Alexandre Belloni, Yan Chen, and Yehua Wei. Online pandora’s box for contextual llm cascading, 2026.

Dylan Bouchard. Is escalation worth it? a decision-theoretic characterization of llm cascades, 2026.

Leo Breiman. Bagging predictors. Machine learning, 24(2):123–140, 1996.

Robert L Brennan. Generalizability theory. Educational Measurement: Issues and Practice, 11(4):27–34, 1992.

Jean Bretagnolle and Catherine Huber. Estimation des densités: risque minimax. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 47(2):119–137, 1979.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020.

Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, and James Zou. Are more LLM calls all you need? towards the scaling properties of compound AI systems. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a.

Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024b. ISSN 2835-8856.

Herman Chernoff. Sequential design of experiments. In Breakthroughs in Statistics: Methodology and Distribution, pp. 345–360. Springer, 1959.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios N. Angelopoulos, Tianle Li, Dacheng Li, Banghua Zhu, Hao Zhang, Michael I. Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: an open platform for evaluating llms by human preference. In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.

Alexander Philip Dawid and Allan M Skene. Maximum likelihood estimation of observer error-rates using the em algorithm. Journal of the Royal Statistical Society: Series C (Applied Statistics), 28(1):20–28, 1979.

Morris H DeGroot. Uncertainty, information, and sequential experiments. The Annals of Mathematical Statistics, 33(2):404–419, 1962.

Jasper Dekoninck, Maximilian Baader, and Martin Vechev. A unified approach to routing and cascading for LLMs. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 12987–13010. PMLR, 13–19 Jul 2025.

Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple classifier systems, pp. 1–15. Springer, 2000.

Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Ruhle, Laks V. S. Lakshmanan, and Ahmed Hassan Awadallah. Hybrid LLM: Cost-efficient and quality-aware query routing. In International Conference on Learning Representations, 2024.

Tao Feng, Yanzhen Shen, and Jiaxuan You. Graphrouter: A graph-based router for llm selections. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 26186–26203, 2025.

Paulina Garcia-Corral, Hannah Béchara, Ran Zhang, and Slava Jankin. Politicause: An annotation scheme and corpus for causality in political texts. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 12836– 12845, 2024.

Shashwat Goel, Joschka Strüber, Ilze Amanda Auzina, Karuna K Chandra, Ponnurangam Kumaraguru, Douwe Kiela, Ameya Prabhu, Matthias Bethge, and Jonas Geiping. Great models think alike and this undermines AI oversight. In Forty-second International Conference on Machine Learning, 2025.

Daniel Golovin and Andreas Krause. Adaptive submodularity: Theory and applications in active learning and stochastic optimization. Journal ofArtificial Intelligence Research, 42:427–486, 2011. doi: 10.1613/ jair.3278.

Y. Guo, Stefanus Jasin, and C. A. A. Lin. Congestion-aware static LLM cascades: Analysis of a steady-state framework. SSRNpreprint 6759238, 2026.

Rajarshi Haldar and Julia Hockenmaier. Rating roulette: Self-inconsistency in LLM-as-a-judge frameworks. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 24986–25004, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1361.

Andrew Halterman and Katherine A. Keith. Codebook llms: Evaluating llms as measurement tools for political science concepts. Political Analysis, 34(2):188–204, 2025. ISSN 1476-4989. doi: 10.1017/pan. 2025.10017.

Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms. Advances in neural information processing systems, 37:8093–8131, 2024.

Nicholas Hay, Stuart Russell, David Tolpin, and Solomon Eyal Shimony. Selecting computations: Theory and applications, 2012.

Qitian Jason Hu, Jacob Bieker, Xiuyu Li, Nan Jiang, Benjamin Keigwin, Gaurav Ranganath, Kurt Keutzer, and Shriyash Kaustubh Upadhyay. Routerbench: A benchmark for multi-LLM routing system. In Agentic Markets Workshop at ICML 2024, 2024.

Baihe Huang, Shanda Li, Tianhao Wu, Yiming Yang, Ameet Talwalkar, Kannan Ramchandran, Michael I. Jordan, and Jiantao Jiao. Sample complexity and representation ability of test-time scaling paradigms. In The Fourteenth International Conference on Learning Representations, 2026a.

Hui Huang, Xingyuan Bu, Hongli Zhou, Yingqi Qu, Jing Liu, Muyun Yang, Bing Xu, and Tiejun Zhao. An empirical study of llm-as-a-judge for llm evaluation: Fine-tuned judge model is not a general substitute for gpt-4. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 5880–5895, 2025.

Jingkai Huang, Will Ma, and Zhengyuan Zhou. Optimal bayesian stopping for efficient inference of consistent LLM answers. In Forty-third International Conference on Machine Learning, 2026b.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. Llama guard: Llm-based input-output safeguard for human-ai conversations, 2023.

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. Beavertails: Towards improved safety alignment of llm via a humanpreference dataset. Advances in Neural Information Processing Systems, 36:24678–24704, 2023.

Jaehun Jung, Faeze Brahman, and Yejin Choi. Trust or escalate: LLM judges with provable guarantees for human agreement. In The Thirteenth International Conference on Learning Representations, 2025.

Daniel Kahneman, Olivier Sibony, and Cass R Sunstein. Noise: A flaw in human judgment. 2021.

Elliot Myunghoon Kim, Avi Garg, Kenny Peng, and Nikhil Garg. Correlated errors in large language models. In Forty-second International Conference on Machine Learning, 2025.

Seungone Kim, Jay Shin, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Ryan Shin, Sungdong Kim, James Thorne, Minjoon Seo, et al. Prometheus: Inducing fine-grained evaluation capability in language models. In International Conference on Learning Representations, volume 2024, pp. 29927– 29962, 2024.

Guneet Kohli. Nine judges, two effective votes: Correlated errors undermine LLM evaluation panels. arXiv preprint arXiv:2605.29800, 2026.

Jacky Kwok, Shulu Li, Pranav Atreya, Yuejiang Liu, Yixing Jiang, Chelsea Finn, Marco Pavone, Ion Stoica, and Azalia Mirhoseini. Llm-as-a-verifier: A general-purpose verification framework. arXiv preprint arXiv:2607.05391, 2026.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. RLAIF vs. RLHF: Scaling reinforcement learning from human feedback with AI feedback. In Forty-first International Conference on Machine Learning, 2024.

Guokai Li, Alys Liang, Mo Liu, Murray Lei, Stefanus Jasin, Fenghua Yang, and Preet Baxi. Sequential sampling for binary classification: Two llms are (almost) all you need, 2026.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM computing surveys, 55(9):1–35, 2023a.

Ryan Liu and Nihar B. Shah. Reviewergpt? an exploratory study on using large language models for paper reviewing, 2023.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2511–2522. Association for Computational Linguistics, 2023b. doi: 10.18653/v1/2023.emnlp-main.153.

Keming Lu, Hongyi Yuan, Runji Lin, Junyang Lin, Zheng Yuan, Chang Zhou, and Jingren Zhou. Routing to the expert: Efficient reward-guided ensemble of large language models, 2023. URL https://arxiv. org/abs/2311.08692.

Anqi Mao, Christopher Mohri, Mehryar Mohri, and Yutao Zhong. Two-stage learning to defer with multiple experts. Advances in neural information processing systems, 36:3578–3606, 2023.

Yasmin Moslem and John D. Kelleher. Dynamic model routing and cascading for efficient LLM inference: A survey. arXiv preprint arXiv:2603.04445, 2026.

Multimodal Art. Jev decision index. Hugging Face Spaces, 2026. URL https://huggingface.co/ spaces/multimodalart/jev-decision-index. Version 0.3.2, accessed October 13, 2026.

Mohammad Naghshvar and Tara Javidi. Active sequential hypothesis testing. The Annals of Statistics, 41 (6):2703–2738, 2013. ISSN 00905364, 21688966.

Sirin Nitinawarat, George K. Atia, and Venugopal V. Veeravalli. Controlled sensing for multihypothesis testing. IEEE Transactions on Automatic Control, 58(10):2451–2464, 2013. doi: 10.1109/TAC.2013. 2261188.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. RouteLLM: Learning to route LLMs from preference data. In The Thirteenth International Conference on Learning Representations, 2025.

OpenAI, Josh Achiam, Steven Adler, and Sandhini Agarwal et. al. Gpt-4 technical report, 2024.

Alexander J. Ratner, Christopher De Sa, Sen Wu, Daniel Selsam, and Christopher Ré. Data programming: Creating large training sets, quickly. Advances in neural information processing systems, 29:3567 – 3575, 2016.

Robert E Schapire and Yoav Freund. Boosting: Foundations and algorithms, 2013.

Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, and Anna Rohrbach. When to solve, when to verify: Compute-optimal problem solving and generative verification for llm reasoning, 2025.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations, 2025.

Seamus Somerstep, Felipe Maia Polo, Allysson Flavio Melo de Oliveira, Prattyush Mangal, Mírian Silva, Onkar Bhardwaj, Mikhail Yurochkin, and Subha Maity. Carrot: A cost aware rate optimal router. arXiv preprint arXiv:2502.03261, 2025.

Wei Song, Zhenya Huang, Cheng Cheng, Weibo Gao, Bihan Xu, GuanHao Zhao, Fei Wang, and Runze Wu. Irt-router: Effective and interpretable multi-llm routing via item response theory, 2025.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, 2023.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Tang, Alejandro Cuadron, Chenguang Wang, Raluca Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges. In International Conference on Learning Representations, volume 2025, pp. 63277–63303, 2025.

Gemma Team, Sherif El Abd, and Vaibhav Aggarwal et. al. Gemma 4 technical report, 2026.

Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. Judging the judges: Evaluating alignment and vulnerabilities in LLMs-as-judges. In Proceedings ofthe Fourth Workshop on Generation, Evaluation and Metrics, 2025.

Yigit Turkmen, Baturalp Buyukates, and Melih Bastopcu. Don’t always pick the highest-performing model: An information theoretic view of LLM ensemble selection. In ICML 2026 Statistical Frameworks for Uncertainty in Agentic Systems, 2026.

Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. Replacing judges with juries: Evaluating llm generations with a panel of diverse models. arXiv preprint arXiv:2404.18796, 2024.

Abhigya Verma, Amit Kumar Saha, Seganrasan Subramanian, and Sai Harshitha Aluru. Agentjudgebench: A multi-difficulty benchmark for evaluating llm judges on agentic tool-calling, 2026.

George Vershinin, Asaf Cohen, and Omer Gurewitz. Active sequential hypothesis testing with nonhomogeneous costs. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 91–95, 2026. doi: 10.1109/ICASSP55912.2026.11463839.

Tu Vu, Kalpesh Krishna, Salaheddin Alzubi, Chris Tar, Manaal Faruqui, and Yun-Hsuan Sung. Foundational autoraters: Taming large language models for better automatic evaluation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 17086–17105, November 2024.

Abraham Wald. Sequential tests of statistical hypotheses. In Breakthroughs in statistics: Foundations and basic theory, pp. 256–298. Springer, 1992.

Xinyuan Wang, Yanchi Liu, Wei Cheng, Xujiang Zhao, Zhengzhang Chen, Wenchao Yu, Yanjie Fu, and Haifeng Chen. Mixllm: Dynamic routing in mixed large language models, 2025.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023a.

Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, Wei Ye, Shikun Zhang, and Yue Zhang. Pandalm: An automatic evaluation benchmark for LLM instruction tuning optimization. arXiv preprint arXiv:2306.05087, 2023b.

An Yang, Anfeng Li, and Baosong Yang et. al. Qwen3 technical report, 2025.

Murong Yue, Jie Zhao, Min Zhang, Liang Du, and Ziyu Yao. Large language model cascades with mixture of thought representations for cost-efficient reasoning. In International Conference on Learning Representations, volume 2024, pp. 21691–21728, 2024.

Haozhen Zhang, Tao Feng, and Jiaxuan You. Router-r1: Teaching llms multi-round routing and aggregation via reinforcement learning. arXiv preprint arXiv:2506.09033, 2025.

Xuechen Zhang, Zijian Huang, Ege Onur Taga, Carlee Joe-Wong, Samet Oymak, and Jiasi Chen. Efficient contextual llm cascades through budget-constrained policy learning, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-ajudge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. In The Thirteenth International Conference on Learning Representations, 2025.

## A Related Work

This work is closely related to the following fields: 1) Sequential information acquisition (decision-theoretic foundations), 2) Routing and cascading methods, 3) LLMs as classifiers, judges, and annotators, 4) Theory of LLM orchestration and test-time compute, and 5) Aggregating Noisy Classifiers.

Sequential information acquisition: Our formulation is based on sequential hypothesis testing. Wald’s sequential probability ratio test (Wald, 1992) established that stopping adaptively, as soon as the accumulated evidence crosses a threshold, requires fewer samples on average than any fixed-sample test with the same error probabilities. Chernoff extended this to the sequential design of experiments (Chernoff, 1959), where the decision-maker chooses which experiment to run next, and DeGroot cast the problem as Bayesian decision-making in which each experiment is valued by its expected reduction in posterior uncertainty (De-Groot, 1962). These fundamental ideas are the principle behind our circuits. Since each LLM query is a costly experiment, the circuit selects the next instrument by the expected improvement in its posterior belief, and it stops once that belief is confident enough that further queries are not worth their cost. The modern literature on active hypothesis testing and controlled sensing (Naghshvar & Javidi, 2013; Nitinawarat et al., 2013; Vershinin et al., 2026) develops this framework further, deriving information-theoretic lower bounds on expected sample size under multiple hypotheses and heterogeneous action costs, and showing that simple posterior-driven policies attain the optimal error exponents. These works solve a single testing instance in which the observation model is known in advance. In our setting, each prompt is one such instance, but there are countably many possibilities of prompts, and the observation model of each instrument for any given prompt must be learned from data.

Routing and Cascading Methods: In routing, a prompt is assigned to a single model in a pool of models that is predicted to handle it best. For example, RouteLLM (Ong et al., 2025) trains binary routers between a strong and a weak model, while HybridLLM (Ding et al., 2024) trains a classifier to predict whether a cheap model’s answer will be of acceptable quality and routes to the large model only when it is needed. CARROT (Somerstep et al., 2025) routes by predicting both expected quality and expected cost per model for a query and minimizing a weighted combination, with a minimax-rate-optimality guarantee for the resulting router. More routing mechanisms (Lu et al., 2023; Feng et al., 2025; Wang et al., 2025; Zhang et al., 2025; Song et al., 2025; Hu et al., 2024) can be found in the survey (Moslem & Kelleher, 2026). The circuits proposed in this work generalize routers since a circuit decides after each observation whether the evidence collected so far is sufficient or whether another model should be queried.

In cascading, models are queried one after another in a fixed order, typically from cheapest to most expensive, and the process stops as soon as some confidence or quality criterion is met. FrugalGPT (Chen et al., 2024b) chains LLM APIs from cheap to expensive and uses a learned scorer to decide whether the current answer is reliable enough to stop. Automix (Aggarwal et al., 2024) has a small model self-verify its own answer and uses this verification signal to decide whether to escalate to a larger model. Yue et al. (2024) escalate based on agreement between a weak model’s answers under different reasoning formats, treating disagreement as the trigger for the strong model. Jung et al. (2025) use judges of increasing size with a conformal-style confidence threshold that guarantees a target level of agreement with humans. More cascading methods can be found in the survey (Moslem & Kelleher, 2026). Our approach differs from these cascades in two ways. First, a cascade queries models in a fixed order and only decides whether to escalate, whereas a circuit chooses which model to query next based on the prompt and the outputs observed so far. Second, a cascade typically returns the output of the model at which it stops, discarding earlier outputs once they have triggered escalation, whereas a circuit combines all observed outputs into a posterior belief on the label and predicts from that belief.

Among cascading methods, TREACLE (Zhang et al., 2024) and cascade-routing (Dekoninck et al., 2025) do not follow a fixed order of models. Using reinforcement learning, TREACLE (Zhang et al., 2024) learns a policy that selects the next LLM and prompting scheme based on the question and the consistency of responses so far, under a budget shared across a dataset of questions. Our circuits also adapt the next query to past model outputs, but rather than training the policy itself, we learn a probabilistic model of how each instrument responds on a given prompt and compute the optimal query-and-stop policy from it by dynamic programming. The resulting circuit is optimal with respect to the learned model and its error–cost tradeoff can be analyzed, neither of which holds for a policy trained end-to-end by reinforcement learning. Cascade routing (Dekoninck et al., 2025) selects the next model after each output by estimating, for each set of remaining models, the quality of the best output it would produce net of its cost, and finally returns the output with the highest estimated quality. Our circuits also choose the next model from the full pool after each observation, but differ in two respects. The decision to continue is solved exactly by dynamic programming rather than by a one-shot estimate of what the remaining models would yield, and the prediction combines all observed outputs through the posterior on the label rather than selecting one output.

LLMs as classifiers, judges, and annotators. Prompting turns an LLM into a classifier without training (Brown et al., 2020; Liu et al., 2023a), and this has made LLMs default raters in many pipelines. They judge model outputs (Zheng et al., 2023; Liu et al., 2023b; Wang et al., 2023b), either as prompted frontier models or as judges fine-tuned for the task (Kim et al., 2024; Zhu et al., 2025; Vu et al., 2024; Huang et al., 2025). They provide feedback for alignment (Lee et al., 2024), act as safety classifiers and guardrails (Inan et al., 2023; Han et al., 2024; OpenAI et al., 2024), and annotate text in the social sciences and in scientific review (Halterman & Keith, 2025; Liu & Shah, 2023). A parallel line of work measures the reliability of raters. This includes documenting position and verbosity biases, disagreement with humans, and vulnerability to adversarial inputs (Tan et al., 2025; Thakur et al., 2025), as well as low intra-rater reliability, where the same judge returns different labels across runs (Haldar & Hockenmaier, 2025).

Theory of LLM orchestration and test-time compute: A growing body of work analyzes how to allocate inference across LLM calls. For a single model, the test-time scaling literature studies how accuracy improves with the number of LLM calls or the amount of verification (Snell et al., 2025; Singhi et al., 2025). Within this line, Chen et al. (2024a) shows that accuracy under majority voting can be non-monotone in the number of calls, Huang et al. (2026a) give sample-complexity bounds for voting and verifier-based selection, and Huang et al. (2026b) characterize when to stop sampling through Bayesian optimal stopping.

For multiple models, routing has been analyzed as a cost-aware prediction problem (Somerstep et al., 2025). Cascades have been studied from several angles: as a sequence of linear programs (Dekoninck et al., 2025), as a Pandora’s Box search problem (Belloni et al., 2026), and as a queueing system with congestion (Guo et al., 2026). Information-theoretic criteria have also been proposed for choosing which models to include in an ensemble (Turkmen et al., 2026). These rest on older foundations in metareasoning (Hay et al., 2012), adaptive submodularity (Golovin & Krause, 2011), and learning to defer (Mao et al., 2023). Closest to our analysis, concurrent work by Li et al. (2026) studies sequential binary classification over heterogeneous LLMs with known accuracies, costs, and latencies. They show that in the high-confidence limit a policy alternating between two hypothesis-specific specialists is asymptotically optimal. Our asymptotic analysis agrees with this: under symmetric accuracies the two specialists coincide, and the optimal circuit concentrates its cost on the instrument with the highest information rate per unit cost. These works, however, analyze a single instance with known model parameters and unlimited repeated queries. Our main focus is on the practical problem of a finite pool of instruments whose reliabilities vary across prompts and must be learned from data, for which we solve the exact dynamic program and show that the resulting circuits improve the cost–accuracy frontier in practice.

Aggregating Noisy Classifiers: We draw from work on combining the outputs of multiple noisy classifiers to determine a true label. In the context of LLMs, aggregating the outputs of smaller models can be used to approximate the judgement of frontier models at lower cost (Verga et al., 2024). Previous work in crowdsourcing and weak supervision (Dawid & Skene, 1979; Ratner et al., 2016) learn the error rate of each individual classifier and aggregate them for the distribution over the true label, assuming independence between the errors of each classifier. Ensemble diversity (Dietterich, 2000) extends this to a machine learning setting by accounting for the correlation of errors between models, which remains an important hurdle for combining the outputs of large language models (Goel et al., 2025; Kim et al., 2025; Kohli, 2026). In this spirit, our probabilistic model learns a distribution over their responses and prompt embeddings, enabling the dynamic program to account for correlation between model outputs.

## B Proofs

ProofofProposition 1. Consider a circuit that queries a batch $( i _ { 1 } , \ldots , i _ { k } )$ . Replace the batch by the same k queries in any fixed order, without allowing the intermediate responses to affect subsequent queries. After all k responses have been observed, the circuit proceeds exactly as the original circuit would after observing the batch. The resulting circuit therefore has the same measurements, the same final prediction, and the same total query cost. Repeating this replacement for every batch gives the desired sequential circuit.

Proof of Proposition 2. Consider the objective function for a fixed prompt z:

$$
\mathcal { L } _ { \lambda } ( \pi , z ) = \mathsf { E } _ { \pi } ( z ) + \lambda \mathsf { C } _ { \pi } ( z )\tag{12}
$$

Then, the zero-cost circuit is optimal when:

$$
\mathsf { E } _ { \pi _ { 0 } } ( z ) \le \mathsf { E } _ { \pi } ( z ) + \lambda \mathsf { C } _ { \pi } ( z )\tag{13}
$$

since $\mathsf C _ { \pi _ { 0 } } ( z ) = 0$ for the zero-cost circuit. This, together with $\lambda \geq 0$ directly results in proving Proposition 2.

Now consider the expression:

$$
\lambda _ { L B } = \operatorname* { s u p } _ { \pi : C _ { \pi } ( z ) > 0 } \frac { [ \mathsf { E } _ { \pi _ { 0 } } ( z ) - \mathsf { E } _ { \pi } ( z ) ] _ { + } } { \mathsf { C } _ { \pi } ( z ) }\tag{14}
$$

The upper bound on $\lambda _ { L B }$ is obtained using the following bounds: $\mathsf { E } _ { \pi _ { 0 } } ( z ) \leq \frac { 1 } { 2 } , \mathsf { E } _ { \pi } ( z ) \geq 0$ , and ${ \mathsf C } _ { \pi } ( z ) \geq$ $C _ { \mathrm { m i n } } ( z )$

Proof of Proposition 4. Fix a prompt z, and write $X = X ^ { ( \mathcal { H } ) }$ and $X _ { \mathrm { n e w } } = X ^ { ( i _ { \mathrm { n e w } } ) }$ . Define

$$
\eta _ { z } ( x ) = P _ { D } ( Y = 1 \mid z , X = x ) , \qquad \eta _ { z } ( x , u ) = P _ { D } ( Y = 1 \mid z , X = x , X _ { \mathrm { n e w } } = u ) .
$$

Since the Bayes accuracy for a binary label is $g ( q ) = \operatorname* { m a x } \{ q , 1 - q \}$ , we have

$$
E _ { \operatorname* { m i n } } ( z , \mathcal { H } ) - E _ { \operatorname* { m i n } } ( z , \mathcal { H } _ { \operatorname* { m e w } } ) = \mathbb { E } \left[ g ( \eta _ { z } ( X , X _ { \mathfrak { m e w } } ) ) - g ( \eta _ { z } ( X ) ) ~ | ~ Z = z \right] \leq \mathbb { E } \left[ \left| \eta _ { z } ( X , X _ { \mathfrak { m e w } } ) - \eta _ { z } ( X ) \right| ~ | ~ Z = z \right] ,
$$

where the inequality follows because g is 1-Lipschitz.

Since Y is binary, conditioning on $X = x .$ , the inner expectation equals

$$
\mathrm { T V } \left( P _ { D } ( Y , X _ { \mathrm { n e w } } \mid z , x ) , P _ { D } ( Y \mid z , x ) P _ { D } ( X _ { \mathrm { n e w } } \mid z , x ) \right) .
$$

By Pinsker’s inequality and Jensen’s inequality,

$$
E _ { \mathrm { m i n } } ( z , { \mathcal { H } } ) - E _ { \mathrm { m i n } } ( z , { \mathcal { H } } _ { \mathrm { n e w } } ) \leq \sqrt { \frac { 1 } { 2 } I ( Y ; X _ { \mathrm { n e w } } \mid X , Z = z ) } .
$$

Averaging over $Z$ and applying Jensen’s inequality again,

$$
\mathbb { E } _ { Z } \left[ E _ { \operatorname* { m i n } } ( Z , { \mathcal { H } } ) - E _ { \operatorname* { m i n } } ( Z , { \mathcal { H } } _ { \mathrm { n e w } } ) \right] \leq { \sqrt { { \frac { 1 } { 2 } } I \left( Y ; X ^ { ( i _ { \mathrm { n e w } } ) } \mid X ^ { ( { \mathcal { H } } ) } , Z \right) } } .
$$

Nonnegativity follows because additional measurements can always be ignored.

## B.1 Proofs for Fixed-Prompt Scaling

We first show that any circuit achieving error probability ε requires expected cost at least log $( 1 / \varepsilon ) / I ( z ) -$ $O ( 1 )$ , and then construct a sequential likelihood-ratio test that attains this bound up to an additive constant.

Preliminaries Fix a prompt z and suppress its dependence in the notation. Write $a _ { i } : = a _ { i } ( z )$ for an instrument’s accuracy on prompt z and $c _ { i } : = c _ { i } ( z )$ for its cost. For the asymptotic analysis, each instrument $i \in \mathcal { H }$ may be queried arbitrarily many times, with independent randomness across calls.

Recall that classification reduces to testing $H _ { 0 } : y ( z ) = 0$ against $H _ { 1 } : y ( z ) = 1$ , where a query to instrument i produces

$$
X \mid H _ { 1 } \sim \operatorname { B e r } ( a _ { i } ) , \qquad X \mid H _ { 0 } \sim \operatorname { B e r } ( 1 - a _ { i } ) .
$$

The log-likelihood ratio contributed by a measurement $x \in \{ 0 , 1 \}$ is

$$
L _ { i } ( x ) = \log \frac { \mathrm { P r } ( X = x \mid H _ { 1 } , i ) } { \mathrm { P r } ( X = x \mid H _ { 0 } , i ) } = ( 2 x - 1 ) \ell _ { i } , \qquad \ell _ { i } : = \log \frac { a _ { i } } { 1 - a _ { i } } .\tag{15}
$$

By our assumption in this analysis, an instrument’s measurement distribution is symmetric under the two hypotheses, so

$$
\mathbb { E } _ { 1 } [ L _ { i } ( X ) ] = d _ { i } , \qquad \mathbb { E } _ { 0 } [ L _ { i } ( X ) ] = - d _ { i } ,
$$

where $\mathbb { E } _ { y }$ denotes expectation under $H _ { y }$ and $d _ { i } : = D _ { \mathrm { K L } } ( \operatorname { B e r } ( a _ { i } ( z ) ) \parallel \operatorname { B e r } ( 1 - a _ { i } ( z ) ) ) = ( 2 a _ { i } - 1 ) \ell _ { i } .$

A sequential circuit chooses an instrument $A _ { t } \in { \mathcal { H } }$ at each step t, observes its measurement $X _ { t } ,$ , and stops at a stopping time τ. The choice of $A _ { t }$ and the decision to continue depend only on the preceding history $\mathcal { F } _ { t - 1 } = \sigma ( A _ { 1 } , X _ { 1 } , \ldots , A _ { t - 1 } , X _ { t - 1 } )$ . Let $\begin{array} { r } { N _ { i } = \sum _ { t = 1 } ^ { \tau } \mathbf { 1 } \{ A _ { t } = i \} } \end{array}$ denote the number of queries to i.

Write $q = q _ { z } ( \emptyset ) \in ( 0 , 1 )$ for the prior probability of $H _ { 1 }$ and $m = \operatorname* { m i n } \{ q , 1 - q \}$ . For a circuit π, let $P _ { y } ^ { \pi }$ and $\mathbb { E } _ { y } ^ { \pi }$ denote probability and expectation under $H _ { y }$ , and define its prior-averaged query counts by

$$
\begin{array} { r } { \bar { N } _ { i } ^ { \pi } = q \mathbb { E } _ { 1 } ^ { \pi } [ N _ { i } ] + ( 1 - q ) \mathbb { E } _ { 0 } ^ { \pi } [ N _ { i } ] . } \end{array}
$$

The error probability and expected cost, conditional on the fixed prompt, are

$$
\mathsf E ( \pi ) = q P _ { 1 } ^ { \pi } ( \hat { Y } = 0 ) + ( 1 - q ) P _ { 0 } ^ { \pi } ( \hat { Y } = 1 ) ,\tag{16}
$$

$$
{ \mathsf { C } } ( \pi ) = \sum _ { i \in \mathcal { H } } c _ { i } { \bar { N } } _ { i } ^ { \pi } .\tag{17}
$$

For an optimal circuit at price λ, these are $\mathsf E _ { \lambda } ( z )$ and $\mathsf C _ { \lambda } ( z )$ , respectively. Lemma 1 (Information lower bound). Every sequential circuit with finite expected cost satisfies

$$
D _ { \mathrm { K L } } ( P _ { y } ^ { \pi } \| P _ { 1 - y } ^ { \pi } ) = \sum _ { i \in \mathcal { H } } d _ { i } \mathbb { E } _ { y } ^ { \pi } [ N _ { i } ] , \qquad y \in \{ 0 , 1 \} ,\tag{18}
$$

where $P _ { y } ^ { \pi }$ denotes the distribution of the complete terminated history under $H _ { y }$ . Consequently,

$$
\log \frac { m } { 2 \mathsf { E } ( \pi ) } \le \sum _ { i \in \mathcal { H } } d _ { i } \bar { N } _ { i } ^ { \pi } \le I \mathsf { C } ( \pi ) .\tag{19}
$$

Proof. The querying and stopping rules depend only on the observed history and are identical under both hypotheses. Their contributions therefore cancel from the likelihood ratio of the terminated history. By the chain rule for relative entropy,

$$
D _ { \mathrm { K L } } ( P _ { 1 } ^ { \pi } \Vert P _ { 0 } ^ { \pi } ) = \mathbb { E } _ { 1 } ^ { \pi } \left[ \sum _ { t = 1 } ^ { \tau } L _ { A _ { t } } ( X _ { t } ) \right] = \mathbb { E } _ { 1 } ^ { \pi } \left[ \sum _ { t = 1 } ^ { \tau } d _ { A _ { t } } \right] = \sum _ { i \in \mathcal { H } } d _ { i } \mathbb { E } _ { 1 } ^ { \pi } [ N _ { i } ] .
$$

The second equality follows by conditioning on the history before each query, since $\mathbb { E } _ { 1 } [ L _ { A _ { t } } ( X _ { t } ) \mid { \mathcal { F } } _ { t - 1 } ] =$ $d _ { A _ { t } }$ . Reversing the hypotheses gives $\begin{array} { r } { D _ { \mathrm { K L } } ( P _ { 0 } ^ { \pi } \| P _ { 1 } ^ { \pi } ) = \sum _ { i } d _ { i } \mathbb { E } _ { 0 } ^ { \pi } [ N _ { i } ] } \end{array}$ , proving Equation (18).

To relate the acquired information to classification error, let $\alpha _ { \pi } = P _ { 1 } ^ { \pi } ( \hat { Y } = 0 )$ and $\beta _ { \pi } = P _ { 0 } ^ { \pi } ( \hat { Y } = 1 )$ . By Equation (16),

$$
\mathsf E ( \pi ) = q \alpha _ { \pi } + ( 1 - q ) \beta _ { \pi } \ge m ( \alpha _ { \pi } + \beta _ { \pi } ) .
$$

The Bretagnolle–Huber inequality (Bretagnolle & Huber, 1979) gives, in either direction,

$$
D _ { \mathrm { K L } } ( P _ { y } ^ { \pi } \| P _ { 1 - y } ^ { \pi } ) \geq \log \frac { 1 } { 2 ( \alpha _ { \pi } + \beta _ { \pi } ) } \geq \log \frac { m } { 2 \mathsf { E } ( \pi ) } .
$$

Averaging these inequalities with weights $q$ and $1 - q .$ , and applying Equation (18), yields

$$
\log \frac { m } { 2 \mathsf { E } ( \pi ) } \leq \sum _ { i } d _ { i } \bar { N } _ { i } ^ { \pi } .
$$

Finally, $d _ { i } \leq I c _ { i }$ for every instrument, so $\begin{array} { r } { \sum _ { i } d _ { i } \bar { N } _ { i } ^ { \pi } \le I \sum _ { i } c _ { i } \bar { N } _ { i } ^ { \pi } = I \mathsf { C } ( \pi ) } \end{array}$

Lemma 2 (Likelihood-ratio test). Suppose $0 < I < \infty ,$ and choose $i ^ { \star } \in \arg \operatorname* { m a x } _ { i } I _ { i }$ . For every $B \geq 0$ there exists a sequential circuit $\pi _ { B }$ using only instrument $i ^ { \star }$ such that

$$
\mathsf E ( \pi _ { B } ) \le e ^ { - B } , \qquad I \mathsf C ( \pi _ { B } ) \le B + \ell _ { i ^ { \star } } .\tag{20}
$$

Proof. Query instrument $i ^ { \star }$ repeatedly, and let

$$
S _ { t } = \log \frac { q } { 1 - q } + \sum _ { k = 1 } ^ { t } L _ { i ^ { \star } } ( X _ { k } )
$$

denote the posterior log-odds after t queries. Write $b = \log ( q / ( 1 - q ) )$ , so $S _ { 0 } = b$

Consider the stopping rule

$$
\tau _ { B } = \operatorname* { i n f } \{ t \geq 0 : | S _ { t } | \geq B \} ,
$$

which predicts $H _ { 1 }$ if $S _ { \tau _ { B } } \geq 0$ and $H _ { 0 }$ otherwise. The posterior probability of error at any stopping state with log-odds $S _ { t }$ is

$$
\operatorname* { m i n } \{ \operatorname* { P r } ( H _ { 1 } \mid { \mathcal { F } } _ { t } ) , \operatorname* { P r } ( H _ { 0 } \mid { \mathcal { F } } _ { t } ) \} = { \frac { 1 } { 1 + e ^ { | S _ { t } | } } } .
$$

Thus, once we establish that $\tau _ { B }$ is finite, the stopping rule immediately gives $\mathsf { E } ( \pi _ { B } ) \leq e ^ { - B }$

It remains to bound the expected stopping time. $\mathrm { I f ~ } | b | \geq B$ , the circuit stops without querying, and both inequalities hold.

Suppose $| b | < B$ . Each query changes $S _ { t }$ by $\pm \ell _ { i ^ { \star } }$ , so for $T _ { n } = \tau _ { B } \wedge n$ , we have that

$$
\left. S _ { T _ { n } } \right. \leq B + \ell _ { i ^ { \star } } .
$$

Under $H _ { y }$ , the process $( 2 y - 1 ) ( S _ { t } - b ) - t d _ { i } ,$ ⋆ is a martingale. Since $\tau _ { B }$ is not yet known to have finite expectation, we first apply optional stopping at the bounded stopping time $T _ { n }$ . Using $| S _ { T _ { n } } | \leq B + \ell _ { i ^ { \star } }$ gives

$$
d _ { i ^ { \star } } \mathbb { E } _ { y } ^ { \pi _ { B } } [ T _ { n } ] = \mathbb { E } _ { y } ^ { \pi _ { B } } \left[ ( 2 y - 1 ) ( S _ { T _ { n } } - b ) \right] \leq B + \ell _ { i ^ { \star } } - ( 2 y - 1 ) b .
$$

Since $T _ { n } \uparrow \tau _ { B }$ , monotone convergence gives the same bound for $d _ { i ^ { \star } } \mathbb { E } _ { y } ^ { \pi _ { B } } [ \tau _ { B } ]$ . In particular, $\tau _ { B }$ has finite expectation under both hypotheses.

Every query has cost $c _ { i ^ { \star } }$ , and $d _ { i ^ { \star } } = I c _ { i ^ { \star } }$ . Averaging over the prior therefore yields

$$
\begin{array} { r } { I \mathsf { C } ( \pi _ { B } ) = d _ { i ^ { \star } } \left( q \mathbb { E } _ { 1 } ^ { \pi _ { B } } [ \tau _ { B } ] + ( 1 - q ) \mathbb { E } _ { 0 } ^ { \pi _ { B } } [ \tau _ { B } ] \right) \le B + \ell _ { i ^ { \star } } + ( 1 - 2 q ) b \le B + \ell _ { i ^ { \star } } , } \end{array}
$$

where the last inequality follows because $b = \log ( q / ( 1 - q ) )$ has the same sign as $2 q - 1$

Since $\tau _ { B }$ is finite almost surely and $| S _ { \tau _ { B } } | \ge B$ , the posterior error bound above gives $\mathsf { E } ( \pi _ { B } ) \leq e ^ { - B }$ □

Together, the two lemmas bound the optimal error–cost tradeoff from both sides. The information lower bound applies to every adaptive circuit, while the likelihood-ratio test achieves the same leading rate using only an instrument maximizing $I _ { i }$

Lemma 3 (Optimal error and cost for a fixed prompt). Suppose $0 < I < \infty$ . Let $\mathsf { E } _ { \lambda }$ and $\mathsf { C } _ { \lambda }$ denote the error and expected cost ofany circuit minimizing $\mathsf { E } ( \pi ) + \lambda \mathsf { C } ( \pi ) . ~ A s ~ \lambda \downarrow 0 ,$

$$
\mathsf E _ { \lambda } = \Theta ( \lambda / I ) ,\tag{21}
$$

$$
I { \mathsf { C } } _ { \lambda } = \log ( I / \lambda ) + O ( 1 ) .\tag{22}
$$

Proof. For $0 < \lambda < I$ , set $\eta = \lambda / I$ and choose $B = \log ( 1 / \eta )$ in Lemma 2. By optimality,

$$
\mathsf { E } _ { \lambda } + \lambda \mathsf { C } _ { \lambda } \leq \eta \left( 1 + \log \frac { 1 } { \eta } + \ell _ { i ^ { \star } } \right) .\tag{23}
$$

To bound the error separately, combine Equation (23) with the information lower bound Equation (19). Writing $x _ { \lambda } = \mathsf E _ { \lambda } / \eta$ gives

$$
x _ { \lambda } + \log \frac { m } { 2 \eta x _ { \lambda } } \leq 1 + \log \frac { 1 } { \eta } + \ell _ { i ^ { \star } } ,
$$

or equivalently,

$$
x _ { \lambda } - \log x _ { \lambda } \leq 1 + \ell _ { i ^ { \star } } + \log { \frac { 2 } { m } } .\tag{24}
$$

For a fixed prompt, the right-hand side is constant. Since $x -$ log x diverges as either $x \downarrow 0$ or $x \to \infty$ , the sequence $x _ { \lambda }$ is bounded above and away from zero. Hence $\mathsf E _ { \lambda } = \Theta ( \eta ) = \Theta ( \lambda / I )$ .

For the cost, Equation (19) and Equation (23) imply

$$
\log \frac { m } { 2 \mathsf { E } _ { \lambda } } \leq I \mathsf { C } _ { \lambda } \leq 1 + \log \frac { 1 } { \eta } + \ell _ { i ^ { \star } } .
$$

Since $\mathsf E _ { \lambda } = \Theta ( \eta )$ , the lower bound equals log $( 1 / \eta ) + { \cal O } ( 1 )$ and we obtain our result.

Proof of Theorem 1. Restoring the dependence on $z ,$ Lemma 3 gives

$$
- \log \mathsf E _ { \lambda } ( z ) = \log \frac { I ( z ) } { \lambda } + O ( 1 ) = I ( z ) \mathsf C _ { \lambda } ( z ) + O ( 1 ) .
$$

Allocation of queries. The same information bound also characterizes which instruments an optimal circuit uses. In particular, any expected cost spent on instruments with strictly lower information rates remains bounded as $\lambda \downarrow 0$

Proposition 5 (Asymptotic query allocation). Fix a prompt with $0 < I < \infty ,$ , and let $\bar { N } _ { i } ^ { \lambda }$ be the expected number ofqueries to instrument i under an optimal circuit. Then

$$
\sum _ { i : I _ { i } < I } c _ { i } \bar { N } _ { i } ^ { \lambda } = { \cal O } ( 1 ) .
$$

Consequently,

$$
\frac { \sum _ { i : I _ { i } < I } c _ { i } \bar { N } _ { i } ^ { \lambda } } { \mathsf { C } _ { \lambda } } \longrightarrow 0 \qquad a s \lambda \downarrow 0 .
$$

Proof. By Equation (19),

$$
\sum _ { i } d _ { i } \bar { N } _ { i } ^ { \lambda } \geq \log \frac { m } { 2 \mathsf { E } _ { \lambda } } .
$$

Using $d _ { i } = I _ { i } c _ { i }$ and $\begin{array} { r } { \mathsf C _ { \lambda } = \sum _ { i } c _ { i } \bar { N } _ { i } ^ { \lambda } } \end{array}$ , we obtain

$$
\sum _ { i } ( I - I _ { i } ) c _ { i } { \bar { N } } _ { i } ^ { \lambda } = I { \mathsf { C } } _ { \lambda } - \sum _ { i } d _ { i } { \bar { N } } _ { i } ^ { \lambda } \leq I { \mathsf { C } } _ { \lambda } - \log { \frac { m } { 2 \mathsf { E } _ { \lambda } } } = O ( 1 ) ,\tag{25}
$$

where the last equality follows from Lemma 3.

If any instrument has $I _ { i } < I .$ , define $\Delta = \mathrm { m i n } _ { i : I _ { i } < I } ( I - I _ { i } ) > 0$ . Equation (25) then gives

$$
\Delta \sum _ { i : I _ { i } < I } c _ { i } \bar { N } _ { i } ^ { \lambda } \leq O ( 1 ) ,
$$

so the total expected cost allocated to strictly suboptimal instruments is $O ( 1 )$ . If every instrument attains I, the sum is zero. Finally, Equation (22) implies $\mathsf C _ { \lambda } \to \infty$ . Dividing the bounded suboptimal expenditure by the total expected cost proves the result. □

## B.2 Proof of the Population Scaling Law

We now consider the prompt distribution $Z \sim { \mathcal { P } } _ { Z }$ . To extend the fixed-prompt results to a population, we need bounds that hold uniformly across prompts.

Assumption 2 (Uniformity across prompts). There exist constants $m _ { 0 } , c _ { \mathrm { m a x } } > 0$ such that, for almost every prompt z,

$$
\operatorname* { m i n } \{ q _ { z } ( \emptyset ) , 1 - q _ { z } ( \emptyset ) \} \geq m _ { 0 } , \qquad \operatorname* { m i n } _ { i : I _ { i } ( z ) = I ( z ) } c _ { i } ( z ) \leq c _ { \operatorname* { m a x } } .\tag{26}
$$

The first condition ensures that the error probability before querying is bounded away from zero, while the second ensures that at least one instrument with the highest information rate has bounded query cost.

We first establish uniform error bounds and use them to prove the population scaling law. We then obtain the finer error–cost rates under the polynomial tail assumption.

Lemma 4 (Uniform fixed-prompt error bounds). Under Assumption 2, for every fixed $r _ { 0 } > 0 ,$ there exist positive constants, independent ofz and $\lambda ,$ such that,for almost every z and all sufficiently small λ,

$$
\mathsf E _ { \lambda } ( z ) \asymp \operatorname* { m i n } \left\{ 1 , \frac { \lambda } { I ( z ) } \right\} ,
$$

$$
0 < I ( z ) \leq r _ { 0 } ,\tag{27}
$$

$$
\mathsf E _ { \lambda } ( z ) = { \cal O } ( \lambda ) ,
$$

$$
I ( z ) > r _ { 0 } .\tag{28}
$$

Proof. For each prompt, choose an information-rate-maximizing instrument $i ^ { \star }$ with $c _ { i ^ { \star } } ( z ) \leq c _ { \mathrm { m a x } }$ . We first bound the overshoot term appearing in Equation (24).

For $a _ { i } \in [ 1 / 2 , 1 )$ , we have

$$
\ell _ { i } - d _ { i } = 2 ( 1 - a _ { i } ) \log \frac { a _ { i } } { 1 - a _ { i } } \leq 2 ( 1 - a _ { i } ) \log \frac { 1 } { 1 - a _ { i } } \leq \frac { 2 } { e } ,
$$

where the last inequality uses u $\log ( 1 / u ) \leq 1 / e$ for $u \in ( 0 , 1 )$ . Since $d _ { i ^ { \star } } = I ( z ) c _ { i ^ { \star } } ( z )$

$$
\ell _ { i ^ { \star } } ( z ) \leq \frac { 2 } { e } + c _ { \mathrm { m a x } } I ( z ) .\tag{29}
$$

Suppose first that $\lambda < I ( z ) \le r _ { 0 }$ . By Equation (24),

$$
x _ { \lambda } - \log x _ { \lambda } \leq 1 + { \frac { 2 } { e } } + c _ { \operatorname* { m a x } } r _ { 0 } + \log { \frac { 2 } { m _ { 0 } } } .
$$

The right-hand side is uniform over this range. As x − log x diverges at both zero and infinity, $x _ { \lambda }$ lies in a fixed compact interval inside $( 0 , \infty )$ . Consequently,

$$
\mathsf E _ { \lambda } ( z ) \asymp \frac { \lambda } { I ( z ) } \qquad \mathrm { i f } ~ \lambda < I ( z ) \le r _ { 0 } .
$$

Now suppose $0 < I ( z ) \leq \lambda$ . Stopping without querying gives error at most $1 / 2$ , so optimality implies

$$
\lambda \mathsf C _ { \lambda } ( z ) \le \frac { 1 } { 2 } , \qquad I ( z ) \mathsf C _ { \lambda } ( z ) \le \frac { 1 } { 2 } .
$$

The information lower bound Equation (19) therefore yields $\begin{array} { r } { \mathsf { E } _ { \lambda } ( z ) \ge \frac { m _ { 0 } } { 2 \sqrt { e } } } \end{array}$ . Since $\mathsf { E } _ { \lambda } ( z ) \le 1 / 2$ , the error is bounded above and below by positive constants. Combining the two ranges proves Equation (27).

Finally, consider $I ( z ) > r _ { 0 }$ . For sufficiently small λ, we have $\lambda < I ( z )$ . By Equation (24) and Equation (29),

$$
x _ { \lambda } - \log x _ { \lambda } \leq K _ { 0 } + c _ { \operatorname* { m a x } } I ( z ) , \qquad K _ { 0 } = 1 + { \frac { 2 } { e } } + \log { \frac { 2 } { m _ { 0 } } } .
$$

Since $x - \log x \geq x / 2$ for every $x > 0$

$$
x _ { \lambda } \leq 2 K _ { 0 } + 2 c _ { \operatorname* { m a x } } I ( z ) .
$$

It follows that

$$
\mathsf E _ { \lambda } ( z ) = \frac { \lambda x _ { \lambda } } { I ( z ) } \le 2 \lambda \left( \frac { K _ { 0 } } { r _ { 0 } } + c _ { \mathrm { m a x } } \right) ,
$$

proving Equation (28).

ProofofTheorem 2. Write $L _ { \lambda } = \log ( 1 / \lambda )$ . For each fixed prompt z with $0 < I ( z ) < \infty$ , Lemma 4 gives $\mathsf E _ { \lambda } ( z ) = \Theta ( \lambda )$ . Hence

$$
\frac { - \log \mathsf E _ { \lambda } ( z ) } { L _ { \lambda } } \longrightarrow 1 .
$$

Combining this with Theorem 1 yields

$$
\frac { { \mathsf C } _ { \lambda } ( z ) } { L _ { \lambda } } \longrightarrow \frac { 1 } { I ( z ) } .\tag{30}
$$

To average this limit across prompts, we first establish a uniform bound on the expected cost.

For each z, choose an instrument $i ^ { \star }$ attaining $I ( z )$ with $c _ { i ^ { \star } } ( z ) \leq c _ { \operatorname* { m a x } }$ . Consider the sequential likelihoodratio test that repeatedly queries $i ^ { \star }$ and stops when the log-likelihood ratio exits $( - B , B )$ . We have established that this test has error probability at most $e ^ { - B }$ and expected query cost at most

$$
\frac { B + \ell _ { i ^ { \star } } ( z ) } { I ( z ) } .
$$

Setting $B = L _ { \lambda }$ and using optimality gives

$$
\mathsf E _ { \lambda } ( z ) + \lambda \mathsf C _ { \lambda } ( z ) \le \lambda + \lambda \frac { L _ { \lambda } + \ell _ { i ^ { \star } } ( z ) } { I ( z ) } .
$$

By Equation (29), $\ell _ { i ^ { \star } } ( z ) \leq 2 / e + c _ { \mathrm { m a x } } I ( z )$ . Consequently,

$$
0 \le \frac { \mathsf C _ { \lambda } ( z ) } { L _ { \lambda } } \le \left( 1 + \frac 2 { e L _ { \lambda } } \right) \frac 1 { I ( z ) } + \frac { 1 + c _ { \operatorname* { m a x } } } { L _ { \lambda } } .\tag{31}
$$

If $\mathbb { E } [ 1 / I ( Z ) ] < \infty$ , the right-hand side provides an integrable bound for sufficiently small λ. Dominated convergence and Equation (30) therefore yield

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \frac { \mathsf C _ { \lambda } } { L _ { \lambda } } = \mathbb E \left[ \frac { 1 } { I ( Z ) } \right] .
$$

If E $[ 1 / I ( Z ) ] = \infty$ , Fatou’s lemma gives

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \operatorname* { l n f } _ { L _ { \lambda } } \geq \mathbb { E } \left[ { \frac { 1 } { I ( Z ) } } \right] = \infty .
$$

Thus, in either case,

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \frac { \mathsf C _ { \lambda } } { L _ { \lambda } } = \mathbb E \left[ \frac { 1 } { I ( Z ) } \right] ,\tag{32}
$$

where the limit is interpreted as infinite when the expectation diverges.

We next bound the population error. Since $0 < I ( Z ) < \infty$ almost surely, there exist $0 < a < b <$ ∞ such that

$$
p : = \operatorname* { P r } ( a \leq I ( Z ) \leq b ) > 0 .
$$

By Lemma 4, for sufficiently small λ and some constant $k > 0$

$$
\mathsf E _ { \lambda } \ge \mathbb E \left[ \mathsf E _ { \lambda } ( Z ) \mathbf 1 _ { \{ a \le I ( Z ) \le b \} } \right] \ge \frac { k p } { b } \lambda .
$$

Therefore,

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \operatorname* { s u p } { \frac { - \log { \mathsf E } _ { \lambda } } { L _ { \lambda } } } \leq 1 .\tag{33}
$$

Suppose first that $\mathbb { E } [ 1 / I ( Z ) ] < \infty$ . For any fixed $r _ { 0 } > 0$ , Lemma 4 gives

$$
\mathsf E _ { \lambda } \le K \lambda \mathbb E \left[ \frac { 1 } { I ( Z ) } \mathbf 1 _ { \{ I ( Z ) \le r _ { 0 } \} } \right] + K ^ { \prime } \lambda = O ( \lambda ) ,
$$

for constants $K , K ^ { \prime } > 0$ . Together with the lower bound above, this implies $\mathsf E _ { \lambda } = \Theta ( \lambda )$ , and hence

$$
\frac { - \log { \mathsf E } _ { \lambda } } { L _ { \lambda } } \longrightarrow 1 .
$$

Combining this with Equation (32), we obtain

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \frac { - \log \mathsf E _ { \lambda } } { \mathsf C _ { \lambda } } = \left( \mathbb E \left[ \frac { 1 } { I ( Z ) } \right] \right) ^ { - 1 } .
$$

Finally, if $\mathbb { E } [ 1 / I ( Z ) ] = \infty .$ , Equation (32) gives $\mathsf C _ { \lambda } / L _ { \lambda } \to \infty$ , while Equation (33) shows that $- \log \mathsf { E } _ { \lambda } / L _ { \lambda }$ remains bounded. Therefore,

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \frac { - \log \mathsf { E } _ { \lambda } } { \mathsf { C } _ { \lambda } } = 0 ,
$$

completing the proof.

We now prove Corollary 1. The tail assumption allows us to determine the orders of both the population error and expected cost. We first establish an identity that recovers expected cost from the optimal error. Lemma 5 (Recovering cost from error). Let Π be afixedpolicy class containing a zero-costpolicy with error at most $1 / 2 .$ For each $\lambda > 0$ , let $( \mathsf { E } _ { \lambda } , \mathsf { C } _ { \lambda } )$ denote the error and cost of a policy minimizing $\mathsf { E } ( \pi ) + \lambda \mathsf { C } ( \pi )$ over Π. Then

$$
{ \sf C } _ { \lambda } = \int _ { \lambda } ^ { \infty } \frac { { \sf E } _ { s } } { s ^ { 2 } } d s - \frac { { \sf E } _ { \lambda } } { \lambda } .\tag{34}
$$

Ifminimizers are not unique, the identity holdsfor any measurable selection.

Proof. Let

$$
V ( \lambda ) = \operatorname* { m i n } _ { \pi \in \Pi } \{ \mathsf E ( \pi ) + \lambda { \mathsf C } ( \pi ) \} .
$$

As the infimum of affine functions of λ, V is concave and hence differentiable almost everywhere. At every point of differentiability, the cost of any optimal policy equals $V ^ { \prime } ( \lambda )$ . Therefore,

$$
{ \frac { d } { d \lambda } } { \frac { V ( \lambda ) } { \lambda } } = { \frac { \lambda { \mathsf { C } } _ { \lambda } - V ( \lambda ) } { \lambda ^ { 2 } } } = - { \frac { \mathsf { E } _ { \lambda } } { \lambda ^ { 2 } } } .
$$

The zero-cost policy gives $0 \leq V ( \lambda ) \leq 1 / 2$ , so $V ( \lambda ) / \lambda \to 0$ as $\lambda \to \infty$ . Integrating from λ to infinity gives

$$
\frac { V ( \lambda ) } { \lambda } = \int _ { \lambda } ^ { \infty } \frac { \mathsf E _ { s } } { s ^ { 2 } } d s .
$$

Substituting $V ( \lambda ) = \mathsf E _ { \lambda } + \lambda \mathsf C _ { \lambda }$ proves Equation (34).

Proof of Corollary 1. Let $D = 1 / I ( Z )$ denote the inverse information rate. The assumed lower tail $\Pr ( I ( Z ) \leq$ $r ) \asymp r ^ { \alpha }$ implies

$$
\operatorname* { P r } ( D > t ) \asymp t ^ { - \alpha } \qquad ( t \to \infty ) .\tag{35}
$$

By the tail-integral formula,

$$
\mathbb { E } [ D ] = \int _ { 0 } ^ { \infty } \operatorname* { P r } ( D > t ) d t .
$$

Thus, $\mathbb { E } [ D ] < \infty$ if and only if $\alpha > 1$ . In this case, the exponential error-decay rate follows directly from Theorem 2. To obtain the finer scaling for the remaining cases, we first determine the population error.

Choose $d _ { 0 } > 0$ sufficiently large that the tail bounds in Equation (35) hold for $t \geq d _ { 0 }$ , and define

$$
M _ { \lambda } = \mathbb { E } \left[ \operatorname* { m i n } \{ 1 , \lambda D \} \mathbf { 1 } _ { \{ D \geq d _ { 0 } \} } \right] .
$$

By Lemma $^ { 4 , }$ the expected error over prompts with $D \geq d _ { 0 }$ is of order $M _ { \lambda }$ , while prompts with $D < d _ { 0 }$ contribute at most $O ( \lambda )$ . Since

$$
M _ { \lambda } \geq \lambda d _ { 0 } \operatorname* { P r } ( D \geq d _ { 0 } ) = \Omega ( \lambda )
$$

for sufficiently small λ, the latter contribution does not change the asymptotic order. Hence $\mathsf E _ { \lambda } \asymp M _ { \lambda }$

For $\lambda < 1 / d _ { 0 }$ , the tail-integral formula gives

$$
M _ { \lambda } = \lambda d _ { 0 } \operatorname* { P r } ( D \geq d _ { 0 } ) + \lambda \int _ { d _ { 0 } } ^ { 1 / \lambda } \operatorname* { P r } ( D > t ) d t \asymp \lambda \left( 1 + \int _ { d _ { 0 } } ^ { 1 / \lambda } t ^ { - \alpha } d t \right) .
$$

Evaluating the integral yields

$$
\mathsf { E } _ { \lambda } \asymp \left\{ \begin{array} { l l } { \lambda , } & { \alpha > 1 , } \\ { \lambda \log ( 1 / \lambda ) , } & { \alpha = 1 , } \\ { \lambda ^ { \alpha } , } & { 0 < \alpha < 1 . } \end{array} \right.\tag{36}
$$

We next determine the expected cost. For $\alpha > 1$ , Equation (32) already establishes

$$
\mathsf C _ { \lambda } \sim \mathbb E [ D ] \log ( 1 / \lambda ) .
$$

If $\alpha = 1$ , then $\mathsf E _ { s } \asymp s \log ( 1 / s )$ near zero. Consequently,

$$
\int _ { \lambda } ^ { \infty } \frac { \mathsf { E } _ { s } } { s ^ { 2 } } d s \asymp \int _ { \lambda } ^ { s _ { 0 } } \frac { \log ( 1 / s ) } { s } d s \asymp ( \log ( 1 / \lambda ) ) ^ { 2 } ,
$$

where $s _ { 0 } > 0$ is fixed. Since $\mathsf E _ { \lambda } / \lambda = \Theta ( \log ( 1 / \lambda ) )$ is of lower order, Equation (34) gives

$$
\mathsf C _ { \lambda } \asymp ( \log ( 1 / \lambda ) ) ^ { 2 } .
$$

Finally, suppose $0 < \alpha < 1$ . Since $\mathsf E _ { s } \asymp s ^ { \alpha }$ , Equation (34) gives

$$
{ \mathsf { C } } _ { \lambda } = O ( \lambda ^ { \alpha - 1 } ) .
$$

For the matching lower bound, choose constants $u , v > 0$ such that, for sufficiently small $\lambda ,$

$$
u \lambda ^ { \alpha } \leq \mathsf { E } _ { \lambda } \leq v \lambda ^ { \alpha } .
$$

Fix $a > 1$ large enough that $u a ^ { \alpha } > v$ . Optimality of the circuit at price aλ implies

$$
\mathsf { E } _ { a \lambda } + a \lambda \mathsf { C } _ { a \lambda } \leq \mathsf { E } _ { \lambda } + a \lambda \mathsf { C } _ { \lambda } .
$$

Therefore,

$$
\mathsf C _ { \lambda } \ge \frac { \mathsf E _ { a \lambda } - \mathsf E _ { \lambda } } { a \lambda } \ge \frac { u a ^ { \alpha } - v } { a } \lambda ^ { \alpha - 1 } .
$$

Hence ${ \mathsf { C } } _ { \lambda } \asymp \lambda ^ { \alpha - 1 }$

Combining the three cases,

$$
\mathsf C _ { \lambda } \asymp \left\{ \begin{array} { l l } { \log ( 1 / \lambda ) , } & { \alpha > 1 , } \\ { ( \log ( 1 / \lambda ) ) ^ { 2 } , } & { \alpha = 1 , } \\ { \lambda ^ { \alpha - 1 } , } & { 0 < \alpha < 1 . } \end{array} \right.\tag{37}
$$

It remains to eliminate λ between Equation (36) and Equation (37).

For $\alpha > 1$ , Theorem 2 gives

$$
\operatorname* { l i m } _ { \lambda \downarrow 0 } \frac { - \log \mathsf E _ { \lambda } } { \mathsf C _ { \lambda } } = \frac { 1 } { \mathbb E [ D ] } .
$$

For $\alpha = 1$

$$
- \log \mathsf E _ { \lambda } = \log ( 1 / \lambda ) - \log \log ( 1 / \lambda ) + O ( 1 ) \asymp \sqrt { \mathsf C _ { \lambda } } .
$$

For $0 < \alpha < 1$

$$
\mathsf E _ { \lambda } \asymp \lambda ^ { \alpha } \asymp \mathsf C _ { \lambda } ^ { - \alpha / ( 1 - \alpha ) } .
$$

These establish the three regimes in Corollary 1.

## C Additional Experimental Details

## C.1 Data

In this work, we consider the following seven classification datasets.

AgentJudgeBench (Verma et al., 2026). Each prompt is a user request, the tools available to an AI agent and the tool calls the agent planned, together with one of two criteria: whether the plan uses exactly the right set of tools (tool selection), or whether it covers every part of the request (query coverage). An instrument answers whether the plan fully meets the criterion without seeing the reference plan. We draw 1,000 requests, giving 20,850 prompts. The seven instruments are: Prometheus-2-7B, gpt-oss-20b, gpt-oss-120b, QwQ-32B, Gemini-2.5-Pro, GPT-5.4 and Claude Sonnet 4.5.

BeaverTails (Ji et al., 2023). Each prompt is a user request together with a model’s response. An instrument answers whether the response is unsafe under the benchmark’s harm categories.

BIG-Bench Hard (Suzgun et al., 2023). Each prompt is a question with two answer options from one of eight tasks (boolean expressions, causal judgement, formal fallacies, hyperbaton, navigate, snarks, sports understanding and web of lies), with 1,865 questions in total. An instrument answers which option is correct. The 46 instruments are models from 0.36B to 123B parameters.

Decision Index (Multimodal Art, 2026). Each prompt is a request from one of six binary decision tasks: whether clinical-trial reports entail a statement (NLI4CT), a yes/no question about cause and effect (CLadder), which of two limericks is the original (BPoMP), whether Wikipedia introductions support a multi-hop claim (HoVer), whether a tweet is sarcastic (iSarcasmEval) and which of two headline edits is funnier (Humicroedit), with 24,928 questions in total. The 18 instruments are entrants of the Decision Index leaderboard, from 0.4B to 36B parameters.

HealthBench (Arora et al., 2025). Each prompt is a conversation between a user and an AI assistant together with one rubric criterion written by physicians, and an instrument answers whether the last response meets the criterion, using the benchmark’s grading instruction. Each prompt was judged by two to five physicians. Therefore, the target is the probability that a randomly chosen physician judges the criterion met. The eight instruments are configurations of three reasoning models: Granite 4.2 3B with reasoning off, low and default, and gpt-oss-20b (low, medium and high effort) and gpt-oss-120b (low and medium effort) (Agarwal et al., 2025).

LMArena (Chiang et al., 2024). Each prompt is a user message together with two responses and an instrument answers which response is more likely to be preferred (A or B). The order in which the two responses are shown is drawn at random for every run of the prompt. We consider 34,952 single-turn English comparisons.

PolitiCAUSE (Garcia-Corral et al., 2024). Each prompt is one sentence from political text, and an instrument answers whether the sentence states a causal relation.

On BeaverTails, HealthBench, LMArena and PolitiCAUSE, we query the instruments ourselves. Each instrument is given the dataset’s official classification instruction, with the model’s default decoding (unless we state otherwise). BeaverTails, LMArena and PolitiCAUSE share seven instruments: Gemma 4 (Team et al., 2026) (E2B, E4B, 26B-A4B and 31B, instruction-tuned) and Qwen3 (Yang et al., 2025) (4B, 8B and 32B). At their defaults, the Gemma models answer directly and the Qwen3 models reason before answering. On AgentJudgeBench, BIG-Bench Hard and Decision Index, we use the answers each benchmark recorded. No call costs were recorded there, so on BIG-Bench Hard and Decision Index a call costs \$0.01 per million tokens per billion active parameters, and on AgentJudgeBench we price each call at the model’s API price with the prompt’s input tokens and an assumed output length.

Each dataset is split into five folds, keeping prompts that share a source together. Each fold serves once as the test set, and the remaining prompts are divided into training and validation.

Embedding model. The embedding $\phi ( z )$ is built from Qwen3-Embedding-0.6B (Yang et al., 2025) with last-token pooling, which gives a 1024-dimensional vector per text.

## C.2 Training the Probabilistic Model

Architecture. The model $f _ { \theta }$ assumes that each prompt belongs to one of $T$ latent types. It first maps the embedding to $h ( z ) = W \phi ( z ) \in \mathbb { R } ^ { 6 4 }$ with a learned matrix $W$ , and three networks take $h ( z )$ as input:

1. a type head, a two-layer network of width 128, gives the probability $\rho _ { t } ( z )$ that z is of type t;

2. a measurement head, a two-layer network of width 128, gives the probability $\mu _ { i t } ( z )$ that instrument i returns 1 on a prompt of type $t ;$

3. a label head gives the probability $\eta _ { t } ( z , s )$ that $Y = 1$ on a prompt of type t at state s. It encodes each observed pair $( j , x ) \in s \mathrm { ~ a s ~ } g ( e _ { j } + e _ { x } ^ { \prime } + e _ { t } ^ { \prime \prime } )$ , where $e _ { j } , e _ { x } ^ { \prime }$ and $e _ { t } ^ { \prime \prime }$ are learned 64-dimensional embeddings of the instrument, the measurement and the type, and $g$ is a two-layer network. It sums these vectors over the state and passes $\begin{array} { r } { [ h ( z ) + e _ { t } ^ { \prime \prime } , \sum _ { ( j , x ) \in s } g ( e _ { j } + e _ { x } ^ { \prime } + e _ { t } ^ { \prime \prime } ) , | \mathcal { M } ( s ) | / n ] } \end{array}$ through a two-layer network of width 128 and a linear output.

The measurements observed so far update the type by Bayes’ rule,

$$
w _ { t } ( z , s ) \propto \rho _ { t } ( z ) \prod _ { ( j , x ) \in s } \mu _ { j t } ( z ) ^ { x } \big ( 1 - \mu _ { j t } ( z ) \big ) ^ { 1 - x } ,\tag{38}
$$

and the outputs of $f _ { \theta }$ average over the types:

$$
\eta _ { \theta } ( \phi ( z ) , s ) = \sum _ { t } w _ { t } ( z , s ) \eta _ { t } ( z , s ) , \qquad \nu _ { \theta } ^ { ( i ) } ( \phi ( z ) , s ) = \sum _ { t } w _ { t } ( z , s ) \mu _ { i t } ( z ) .\tag{39}
$$

Loss function. For a prompt z, let $\mathcal { D } ( z )$ be the multiset of its recorded measurements $( i , x )$ , and let $a _ { 1 } ( z )$ and $a _ { 0 } ( z )$ be the numbers of annotations that give label 1 and label 0. For a hard label, $a _ { y ( z ) } ( z ) = 1$ and the other count is zero. On HealthBench, they count the physicians who judged the criterion met and not met. For each prompt z in a minibatch B we draw a state $s _ { z } \colon$ a number k uniformly from $\{ 0 , \ldots , n \}$ , a uniformly random set $\boldsymbol { \mathcal { M } } ( \boldsymbol { s } _ { z } )$ of k instruments, and for each $i \in \mathcal { M } ( s _ { z } )$ one of its recorded measurements on $z ,$ chosen uniformly. With $\ell ( p , x ) = - x \log p - ( 1 - x ) \log ( 1 - p )$ the binary cross-entropy, the label loss is

$$
\mathcal { L } _ { Y } ( \theta ) = \frac { \sum _ { z \in B } \left[ a _ { 1 } ( z ) \ell \big ( \eta _ { \theta } ( \phi ( z ) , s _ { z } ) , 1 \big ) + a _ { 0 } ( z ) \ell \big ( \eta _ { \theta } ( \phi ( z ) , s _ { z } ) , 0 \big ) \right] } { \sum _ { z \in B } \big ( a _ { 0 } ( z ) + a _ { 1 } ( z ) \big ) } .\tag{40}
$$

For the measurement loss we draw a second state $s _ { z } ^ { \prime }$ in the same way and score the prediction for every unqueried instrument against all its recorded measurements,

$$
\mathcal { L } _ { X } ( \theta ) = \frac { 1 } { | B | } \sum _ { z \in B } \frac { 1 } { N ^ { \prime } ( z ) } \sum _ { ( i , x ) \in D ( z ) , i \notin \mathcal { M } ( s _ { z } ^ { \prime } ) } \ell \big ( \nu _ { \theta } ^ { ( i ) } ( \phi ( z ) , s _ { z } ^ { \prime } ) , x \big ) ,\tag{41}
$$

where $N ^ { \prime } ( z )$ is the number of terms in the inner sum. We minimize $\mathcal { L } _ { Y } + \mathcal { L } _ { X }$

Optimization and calibration. We use AdamW (learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 256) with a cosine schedule over at most 300 epochs. After each epoch we compute $\mathcal { L } _ { Y }$ on a fixed set of validation states, one per validation prompt, keep the epoch where it is lowest, and stop after 30 epochs without improvement. We train one model for each $T \in \{ 1 , 2 , 4 , 8 \}$ and keep the one with the lowest validation loss $\mathcal { L } _ { Y } + \mathcal { L } _ { X }$ . We then fit one temperature per head on ten states per validation prompt, $\eta =$ $\sigma ( u _ { \eta } / \tau _ { \eta } )$ and $\nu ^ { ( i ) } = \sigma ( u _ { \nu , i } / \tau _ { \nu } )$ on the logits $u ,$ each chosen to minimize the validation negative log likelihood. We train five models per fold by changing the random seed.

Generating circuits. We solve Equation (7) with $q _ { z } ( s ) = \eta _ { \theta } ( \phi ( z ) , s )$ and with $\nu _ { \theta } ^ { ( i ) } ( \phi ( z ) , s )$ in place of $Q _ { X \mid Z } ^ { [ i ] } ( 1 \mid z )$ . On BIG-Bench Hard and Decision Index, which have a large set of instruments, we keep the eight whose per-task cost–accuracy frontier on the training and validation prompts comes closest to that of all instruments. The program uses a cost known before the query. On the datasets we query ourselves, $c _ { i } ( z )$ is the input tokens of z times the input price plus the 95th percentile of instrument i’s output length on training prompts times the output price. On the other three, $c _ { i } ( z )$ is instrument i’s mean recorded cost on training prompts. Costs are divided by the mean cost of the most expensive instrument. We solve Equation (7) for $\lambda = 0$ , for 201 values of λ spaced geometrically from $1 0 ^ { - 8 } \mathrm { t o } 1 0 ^ { 2 }$ , and for $\lambda = \infty$ (the zero-cost prediction). For each budget, we choose the λ with the best validation metric among those whose mean validation cost is within the budget and report its test metric. Since some benchmarks have class imbalance, we always optimize for balanced accuracy and report balanced accuracy in our plots.

## C.3 Full list of baselines

• Predictive routers: ROUTELLM (Ong et al., 2025), HYBRIDLLM (Ding et al., 2024), and CAR-ROT (Somerstep et al., 2025)

• Model cascades: CASCADE ROUTING (Dekoninck et al., 2025), CONSISTENCY CASCADES (Yue et al., 2024), CASCADED SELECTIVE (Jung et al., 2025), and FRUGALGPT (Chen et al., 2024b)

• Sampling methods: ADAPTIVE-CONSISTENCY (Aggarwal et al., 2023)

RouteLLM and Consistency Cascade route between two instruments, and we sweep every pair. Since HY-BRIDLLM fine-tunes a separate router for each pair of instruments, we use a single pair: the cheapest instrument and the most accurate of the costlier instruments on the training prompts.

## C.4 Additional Experimental Results

In this subsection, we provide additional experimental analysis of our learned circuits.

## C.4.1 Calibration of the circuit’s beliefs.

Our method returns a predictive distribution over the ground truth label, allowing us to assess whether its confidence is well calibrated. The circuit’s belief at the terminal state $s T$ is $\hat { p } ( z ) ~ = ~ \eta _ { \theta } ( \phi ( z ) , s _ { T } )$ . We evaluate these probabilities with the Brier score

$$
\mathsf { B r i e r } ( \pi ) = \mathbb { E } _ { Z } \mathbb { E } _ { X \mid Z } \Big [ \big ( \hat { p } ( Z ) - Y \big ) ^ { 2 } \Big ] ,\tag{42}
$$

which is small only when the forecasts are both calibrated and sharp. A forecast is calibrated ${ \mathrm { i f } } ,$ among the prompts forecast at probability $p ,$ a fraction $p$ have label 1. While the dynamic program never optimizes for the Brier metric directly, it attains very competitive calibration across different budgets; see Figure 6. We also report the calibration of majority voting, treating the fraction of the instruments’ calls that give label 1 as its forecast probability.

![](images/2a0e443e528ca8dbe46e834b8da1a2c04bfc83e50d031b192a01ce7823f6f566.jpg)  
Figure 6: The circuit’s beliefs are calibrated at every budget. For each task, left: Brier score on the test set (lower is better) against expected cost per item; right: reliability diagrams of the circuits at different budgets and of majority voting. The dashed line is perfect calibration. The circuit’s expected calibration error is at most 2.1% at every budget on six tasks, and at most 5.2% on BIG-Bench Hard, whereas majority voting’s is 4.8% to 24.6% and its miscalibration differs in direction by task.

## C.4.2 Joint distribution between instruments

To test the assumption that LLMs are conditionally independent given any prompt (Assumption 1), we queried eight instruments 100 times each on 20 prompts from BeaverTails. For any two instruments, we compute the Pearson correlation $\phi$ of their answers in two ways: within a prompt, across its 100 calls, averaged over prompts; and across prompts, pooling all 2,000 calls. Under the assumption, the first is zero, while the second can be large because the instruments share the (latent) difficulty of each prompt. To test the joint independence of all eight instruments on a prompt, we use their total correlation

$$
\mathsf { T C } ( z ) = \sum _ { i } H \bigl ( X ^ { ( i ) } \mid Z = z \bigr ) - H \bigl ( X ^ { ( 1 ) } , \ldots , X ^ { ( 8 ) } \mid Z = z \bigr ) ,\tag{43}
$$

estimated in bits from the 100 calls, which is zero if and only if the answers are jointly independent. We compare it with a permutation null that shuffles each instrument’s calls independently, preserving each instrument’s answer frequencies while removing any dependence between instruments. We present these results in Figure 7.

Given labeled data, the fitted model treats measurements as independent given the embedding and the latent type (Section C.2). Since the type is unobserved, a measurement changes the model’s prediction of the other instruments’ measurements, so the model can represent dependence between instruments that the embedding alone does not explain. We use this joint to estimate correlations on the test prompts, without the true labels, and compare them with those measured from the recorded calls and labels on the same set of prompts, pooling the five test folds. Write $\nu _ { x } ^ { ( i ) } ( \phi , s )$ for the model’s probability that instrument i answers x at state s (so $\nu _ { 1 } ^ { ( i ) } = \nu _ { \theta } ^ { ( i ) } ) , \eta _ { x } ( \phi , s )$ for its belief that $Y = x$ at state s, and $e _ { i } = \mathbf { 1 } \{ X ^ { ( i ) } \neq Y \}$ for the error of instrument i. Then

![](images/44b4c3e68f98ea5deb42cf121eaa78160c9de498249a778705a7bf3dc4ad6a87.jpg)

![](images/f6b83ae2b53d8d3243d7184adbe35a53bcc3c0af11e7dad237c30ada4c0c9be3.jpg)  
Figure 7: Instrument answers are independent within a prompt but correlated across prompts. (a) ϕ coefficient of two instruments’ answers within a prompt, averaged over prompts on which both instruments’ answers vary. (b) The same coefficient across prompts, pooling all calls. (c) Total correlation of the eight answers on each prompt (Equation (43)) against the 95th percentile of its permutation null. Red marks a rejection at the 5% level. Within a prompt the coefficients lie between −0.10 and 0.15, while across prompts they reach 0.69. Only one of the 20 prompts rejects joint independence, the rate expected by chance.

$$
\begin{array} { l } { \displaystyle \operatorname* { P r } \left( X ^ { ( i ) } = x , X ^ { ( j ) } = x ^ { \prime } \mid \phi \right) = \nu _ { x } ^ { ( i ) } ( \phi , \emptyset ) \nu _ { x ^ { \prime } } ^ { ( j ) } \left( \phi , \{ ( i , x ) \} \right) , } \\ { \displaystyle \qquad \operatorname* { P r } ( e _ { i } = 1 \mid \phi ) = \sum _ { x } \nu _ { x } ^ { ( i ) } ( \phi , \emptyset ) \left[ 1 - \eta _ { x } ( \phi , \{ ( i , x ) \} ) \right] , } \\ { \displaystyle \qquad \operatorname* { P r } ( e _ { i } = e _ { j } = 1 \mid \phi ) = \sum _ { x } \operatorname* { P r } \left( X ^ { ( i ) } = x , X ^ { ( j ) } = x \mid \phi \right) \left[ 1 - \eta _ { x } ( \phi , \{ ( i , x ) , ( j , x ) \} ) \right] , } \end{array}\tag{44}
$$

where two binary instruments err together only when they give the same answer. Averaging these moments over the test prompts gives the implied correlations, and the implied accuracy of instrument i is $1 - \mathbb { E } [ \operatorname* { P r } ( e _ { i } = 1 \mid \phi ) ]$

Consider Figure 8. The fitted model recovers nearly all of the dependence between instruments: on six of the seven tasks, the implied correlations of answers and of errors are within 4% of the measured ones on average, and on BIG-Bench Hard, the smallest task, they are 83% to 85% of them. The model also predicts each instrument’s accuracy on unseen prompts with high fidelity: the predicted accuracies are within 1.2 points of the measured ones on average on five tasks, 2.1 points on AgentJudgeBench and 3.6 points on BIG-Bench Hard.

## C.4.3 How does the circuit allocate its queries?

We first examine how the value of additional queries depends on the reliability of the instrument pool for a given prompt; see Figure 9. For a given prompt z, let r(z) be the proportion of the recorded calls of the instruments the circuit can query that return the label y(z). A prompt is (1) all right if $r ( z ) = 1 , ( 2 )$ noisy if $1 / 2 < r ( z ) < 1$ , (3) mostly wrong if $0 < r ( z ) \leq 1 / 2 .$ , and (4) all wrong if $r ( z ) = 0$ . On HealthBench, prompts on which the physicians disagree are left out. On noisy prompts, correct measurements occur more often, so additional queries provide further evidence for the correct label. On mostly wrong and all wrong prompts, the measurements favor the incorrect label, so additional queries to the same pool tend to reinforce the error.

![](images/e78923c80e0260c8cc55b3d0d126d1985c0c985a9a909e6c73d933bfad7c9016.jpg)  
Figure 8: The fitted model predicts each instrument’s accuracy on unseen data without labels and recovers nearly all of the dependence between instruments. Correlations and accuracies implied by the fitted model on unlabeled test prompts (Equation (44)) against those measured from recorded calls and labels. Left: correlation of two instruments’ answers across prompts (one point per pair). Middle: correlation of their errors. Right: accuracy of each instrument. The dashed line is equality.

We next examine how the circuit allocates those measurements across prompts; see Figure 10. Let $s ( z )$ be the fraction of these recorded calls on z that return 1, and define the instruments’ disagreement by $d ( z ) = 1 - | 2 s ( z ) - 1 |$ , so that $d ( z ) = 0$ when all calls agree and $d ( z ) = 1$ when they split evenly. We also record whether the embedding prediction $\mathbf { 1 } \{ \eta _ { \theta } ( \phi ( z ) , s _ { 0 } ) \ \geq \ 1 / 2 \}$ agrees with the majority of these calls. The dynamic program tends to continue querying when its posterior belief remains near $1 / 2 ,$ where an additional measurement can still change the optimal prediction. Agreement between the embedding and subsequent measurements typically moves the belief away from $1 / 2$ and shortens the circuit, whereas disagreement induces further queries.

Lastly, Section 4.2 shows that, as $\lambda \downarrow 0 ,$ the optimal circuit concentrates its queries on instruments with the largest prompt-specific information rate $I _ { i } ( z ) = d _ { i } ( z ) / c _ { i } ( z )$ . Since $d _ { i } ( z )$ depends on the unknown label, we construct an analogue using the fitted model that measures the information provided by the first query to instrument i, beyond the embedding, per unit cost:

$$
\hat { r } _ { i } = \frac { \hat { I } _ { i } } { \bar { c } _ { i } } , \qquad \hat { I } _ { i } = \widehat { \mathbb { E } } \Big [ h ( \eta _ { \theta } ( \phi ( Z ) , s _ { 0 } ) ) - h \Big ( \eta _ { \theta } ( \phi ( Z ) , \{ ( i , X ^ { ( i ) } ) \} ) \Big ) \Big ] ,\tag{45}
$$

where $h ( p ) = - p \log _ { 2 } p - ( 1 - p ) \log _ { 2 } ( 1 - p ) , { \bar { c } } _ { i }$ is the mean cost of instrument i, and the expectation is taken over validation prompts, their recorded responses, and the fitted models of every fold and seed. Thus $\hat { I } _ { i }$ estimates $I ( Y ; X ^ { ( i ) } \mid \phi ( Z ) ) ,$ ) in bits, and $\boldsymbol { { \hat { r } } _ { i } }$ measures information per unit cost. Unlike $I _ { i } ( z )$ , however, $\boldsymbol { { \hat { r } } _ { i } }$ is averaged over prompts and describes only the value of the first query.

We define the entry point of instrument i as the smallest expected cost per item at which the circuit makes at least 0.05 expected calls per prompt to i. The Spearman rank correlation between entry order and $\boldsymbol { { \hat { r } } _ { i } }$ is 1.00, 0.96, and 0.79 on Arena, BeaverTails, and PolitiCAUSE, respectively. For comparison, the corresponding correlations are 0.75, 0.89, and 0.93 for inverse cost; 0.79, 0.21, and 0.71 for accuracy; and −0.79, 0.04, and 0.43 for low redundancy, measured as the negative mean error correlation with the remaining instruments.

![](images/4cca42e6a6f0203f6d344df5f2ffb5b66bbbe1b461a3d4ba8d3e858a3dc0688f.jpg)  
Figure 9: Additional queries improve accuracy when instrument errors vary across calls. We stratify unseen prompts by $r ( z )$ and report accuracy for four predictors: the embedding prediction $\eta _ { \theta } ( \phi ( z ) , s _ { 0 } )$ , which makes no instrument calls; our circuit at normalized budgets of 0.03 and 1; and majority voting over the instruments the circuit can query. Numbers above the bars give the mean number of calls per prompt, and percentages below each group give its share of the test set. Relative to the embedding prediction, the circuit at budget 1 improves accuracy on noisy prompts by 5–44 percentage points. On all right prompts, it come within 1.3 percentage points of majority voting on four tasks while using 38–86% fewer calls.

## C.5 Circuit Examples

We present two examples of circuits our method builds for individual test prompts at three values of λ; see Figure 13 and Figure 14. Using our trained probabilistic model, the prompt’s embedding gives the initial belief $\eta _ { \theta } ( \phi ( z ) , s _ { 0 } )$ , shown as “Embedding @ $p \% ^ { \dag }$ . For each λ, the circuit either predicts immediately or asks an instrument. Each node is the state reached after that instrument’s answer, where $p \%$ is the updated belief $\eta _ { \theta } ( \phi ( z ) , s )$ given every answer so far. Instrument names are colored by cost, from green (cheapest) to red. The highlighted path is the one followed with the prompt’s recorded answers. At stopping, the circuit predicts the label with belief above $1 / 2 .$

![](images/7cb69fe87c6fd0b378f61a4dd163d6b1f1a2e7550c933c4fb7dc3060ec8e96a5.jpg)

![](images/7bd38e05ed2891dbd0da9be79c87e086547ba0abe70a0d7ca3b46163731c1b73.jpg)  
Figure 10: The circuit allocates more queries when the prior prediction and instrument measurements disagree. We present the expected number of calls per test prompt for circuits at normalized budgets 0.03, 0.1, and 0.2 on Arena, BeaverTails and PolitiCAUSE, grouped by prompt type (left), instrument disagreement d(z) (middle: 0, (0, 1/3], (1/3, 2/3], and (2/3, 1]), and whether the embedding prediction agrees with the majority of the recorded calls (right). Percentages below each group give its share of the test set. Mean calls increase from all right to mostly wrong prompts on every task. The clearest separation occurs when conditioning on agreement with the embedding prediction: at budget 0.2, the circuit makes 1.7–1.9× as many calls when the embedding and instrument majority disagree on Arena (1.84 vs. 1.08), BeaverTails (1.77 vs. 0.95) and PolitiCAUSE (1.85 vs. 1.11).  
Figure 11: Instruments with higher information per unit cost enter the circuit earlier. Expected calls per unseen prompt to each instrument as a function of the circuit’s expected cost per item. Instruments are colored by their rank under the estimated information rate rˆ<sub>i</sub> in Equation (45), with the corresponding value in bits per unit cost shown in parentheses. Across the three tasks, instruments with larger $\hat { r } _ { i }$ tend to begin receiving calls at lower expected cost. The highest-rate instrument enters first on every task, including Arena, where Gemma 26B is not the cheapest instrument.

![](images/827f966b5f53e1886ca586590bf4d92e06ca366893bb38657409e8c61d0388de.jpg)  
Figure 12: Building the instrument pool. We build the instrument pool one instrument at a time according to different strategies: by estimated conditional information given the embedding and the pool (greedy algorithm), by accuracy on the training calls (most accurate first), cheapest first, or in random order (mean of three runs). Every point is our circuit restricted to that pool, using a single fitted model (an earlier version of the model without latent types). The circled point is the circuit with all instruments. Top: frontier area of test accuracy. Middle: test accuracy when every instrument in the pool is queried $( \lambda = 0 )$ . Bottom: the upper bound $\sqrt { I / 2 }$ of Proposition 4 on the drop in Bayes error from adding each instrument (I its estimated conditional information, in nats), and the drop in the model’s Bayes error $\mathbb { E } [ \operatorname* { m i n } \{ \eta , 1 - \eta \} ]$ that the instrument’s answers actually produce, both on test.

![](images/be202839d4e0680fdf55676f5014d07eecd5bf898e525fae06ab844f5a352e69.jpg)  
Figure 13: Example of generated circuit and instrument’s error correlations on BeaverTails

![](images/b85440b6aef2514c87b4d15db55b7c1d859925f740d0abf8f8ffd52a637c2496.jpg)  
Figure 14: Example of generated circuit and instrument’s error correlations on LMArena