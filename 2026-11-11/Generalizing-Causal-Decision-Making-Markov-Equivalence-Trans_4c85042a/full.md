# Generalizing Causal Decision-Making: Markov Equivalence, Transportability and Counterfactuals

Min Woo Park

alsdn0110@snu.ac.kr

Sanghack Lee

sanghack@snu.ac.kr

Seoul National University

Seoul, Republic of Korea

## Abstract

Structural causal bandits use causal knowledge to guide exploration when reward mechanisms are unknown. This paper extends the approach to partially known causal structure, information from other environments, and realizable counterfactual actions. First, we develop graphical criteria for minimal intervention sets under Markov equivalence and characterize possibly-optimal minimal intervention sets for maximal ancestral graphs. A search algorithm finds sets that are possibly optimal in at least one graph represented by a partial ancestral graph. Second, we derive target reward bounds from source distributions through partial transportability, accounting for non-manipulable variables. Dominance relations provide further bounds for action pruning and restricted exploration, with a regret guarantee under valid prior bounds. Third, we formulate counterfactual structural causal bandits over actions supported by specified experimental capabilities. Graphical characterizations of minimality and possible optimality yield algorithms that remove equivalent or dominated choices while preserving an optimal action. Experiments with specified structural causal models examine how these reductions and transferred bounds afect learning. Together, these results clarify how causal knowledge and experimental capabilities determine which actions merit exploration and guide learning within the retained action space.

Keywords: Causal Inference, Causal Decision-Making, Markov Equivalence, Transportability, Counterfactual Inference, Multi-Armed Bandits

## 1 Introduction

Sequential decision-making requires an agent to choose actions while learning their consequences. In a multi-armed bandit (MAB), the agent repeatedly selects an action and observes its reward, balancing the value of gathering information against the value of choosing a promising action (Thompson, 1933; Robbins, 1952; Lai and Robbins, 1985; Auer et al., 2002; Lattimore and Szepesv´ari, 2020). An unstructured bandit treats the reward distributions of diferent actions as unrelated. In many applications, however, actions intervene on variables in a common causal system, and their efects are therefore connected.

For example, a physician choosing treatments must consider how they afect the same underlying biological mechanisms. An education policy may change several factors that influence later employment, such as access to training and acquired skills. In both cases, treating every combination of interventions as an unrelated action can require unnecessary exploration. Causal knowledge helps determine which interventions can afect the outcome and which choices can be excluded without losing an optimal decision (Bareinboim et al., 2024).

![](images/1c473ddad4c493124c19c30108c55c8c8f9eb81fba4ee8620229325228a72eb2.jpg)  
Figure 1: Structural causal bandits. Ofline, the agent uses the causal knowledge (e.g., causal diagram) to reduce the action space while retaining at least one optimal action. Online, it selects interventions from this reduced space and uses rewards from the environment’s SCM to guide subsequent choices.

Structural causal models (SCMs) describe such systems through mechanisms that generate each variable from its causes and unobserved inputs (Pearl, 2000). An intervention replaces selected mechanisms while leaving the others intact. This representation connects the efects of diferent actions and supports questions that a list of unrelated reward distributions does not answer. Causal decision-making uses these relationships to inform learning and action selection (Zhang and Bareinboim, 2019, 2020; Zhang et al., 2020; Kumor et al., 2021; Zhang and Bareinboim, 2022; Ruan et al., 2024; Zhang and Bareinboim, 2025; Richens and Everitt, 2024; Richens et al., 2025; Mueller and Pearl, 2023; Bareinboim et al., 2024).

Early work on causal bandits demonstrated how causal information can improve exploration (Bareinboim et al., 2015; Lattimore et al., 2016; Forney et al., 2017). Subsequent studies considered additional information or restrictions, including access to observational and interventional distributions and assumptions on causal mechanisms (Zhang and Bareinboim, 2017; Lu et al., 2020; Bilodeau et al., 2022; De Kroon et al., 2022; Feng and Chen, 2023; Varici et al., 2023). Structural causal bandits (SCBs) (Lee and Bareinboim, 2018, 2019) provide the starting point of this paper. In an SCB, an action specifies both the variables to intervene on and the values assigned to them.

Fig. 1 separates the SCB framework into two stages. Ofline action space optimization uses graphical equivalence and dominance relations in the known causal structure to remove unnecessary interventions while retaining at least one optimal action in every compatible SCM. This optimization requires neither the mechanisms nor the reward means to be known. During online learning, the agent selects an intervention from the reduced action space, observes the reward generated by the environment’s SCM, and uses the accumulated observations to guide subsequent choices.

Contributions. The central question of this paper is which actions remain worth exploring given the agent’s causal knowledge and experimental capabilities. We extend SCBs in three complementary directions, each addressing a diferent limitation of the starting formulation.

▷ Markov equivalence. Observational data generally do not determine a unique causal diagram. Sec. 4 develops action space optimizations when the agent knows only a

Markov equivalence class, represented by a partial ancestral graph. We characterize minimal intervention sets for ancestral graphs, distinguish whether their minimality can be witnessed by one compatible model, and establish a graphical characterization of possibly-optimal minimal intervention sets for maximal ancestral graphs. A search algorithm identifies sets that are possibly optimal in at least one represented maximal ancestral graph (Park et al., 2025).

▷ Transportability. Data collected elsewhere can inform a target decision problem, but diferences between environments and restrictions on manipulation limit what can be transferred. Sec. 5 first reviews the existing formulation of structural causal bandits with non-manipulable variables. It then develops partial transportability and dominance bounds for target rewards, combines information from available source experiments, and uses the resulting bounds to prune actions and restrict exploration. A regret guarantee describes the behavior of the proposed policy under valid prior bounds (Park and Lee, 2026a).

▷ Counterfactuals. Some systems allow diferent mechanisms to receive diferent versions of an input on the same unit. Sec. 6 uses realizability to specify the counterfactual actions supported by these capabilities. We extend minimality and possible optimality to this action space, derive graphical characterizations, and construct a reduced collection that preserves an optimal action (Park and Lee, 2026b). The required capabilities are stated explicitly, because a mathematically defined counterfactual need not be experimentally available.

Relation to earlier publications. This paper brings together and revises our studies of structural causal bandits under Markov equivalence (Park et al., 2025), transportability (Park and Lee, 2026a), and counterfactual interventions (Park and Lee, 2026b). The revisions include corrected graphical statements, algorithms, and proofs, together with new or recomputed experiments. The opening paragraphs of Secs. 4 to 6 identify the changes relative to each publication; the statements and experimental results presented here are the revised versions.

Organization. Sec. 2 introduces the causal and graphical foundations. Sec. 3 develops the bandit objective and action space optimization. Secs. 4 to 6 present the three extensions, and Sec. 7 brings the results together. The appendices provide supporting proofs and experimental model specifications.

## 2 Causal and Graphical Preliminaries

This section introduces the notation used throughout the paper, the semantic framework represented by structural causal models (SCMs), and the necessary background results on which our work builds. A large part of this section follows Jaber (2022); Bareinboim et al. (2022); Zhang (2023) and Bareinboim et al. (2024).

The SCM and intervention notation supports every subsequent section. Readers can proceed to Sec. 3 after the material on SCMs and causal diagrams. The material on maximal ancestral graphs (MAGs), partial ancestral graphs (PAGs), and Fast Causal Inference (FCI) prepares the partial-structure setting in Sec. 4.

## 2.1 Basic Notation

Following standard conventions, we use a capital letter, such as X, to represent a variable, with its corresponding lowercase letter, x, denoting a realization of the variable. Boldface letters are used to represent sets of variables or values, denoted by X or x. The domain $\Omega _ { X }$ is the set of possible values of X, and $\Omega _ { \mathbf { X } } = \bigtimes _ { X \in \mathbf { X } } \Omega _ { X }$ . For a discrete distribution $P ,$ its support consists of the assignments in the domain that have positive probability; it may vary across environments and interventions. We consistently use $P ( \mathbf { x } )$ as an abbreviation for $P ( \mathbf { X } = \mathbf { x } )$ . We denote by ${ \mathbb { 1 } } \{ { \bf X } = { \bf x } \}$ the indicator function. Two values x and z are consistent if they share common values for $\mathbf { X } \cap \mathbf { Z }$ . We denote by $\mathbf { x } \setminus \mathbf { Z }$ the value of $\mathbf { X } \setminus \mathbf { Z }$ consistent with x and by $\mathbf { x } \cap \mathbf { Z }$ the restriction of x to the variables in Z.

A graph $\mathcal { G } = \langle \mathbf { V } , \mathbf { E } \rangle$ is defined over a set of nodes (vertices) V and a set of edges E. A graph is directed if all its edges are directed (→). We use mixed graphs whose edges may be directed or bidirected (↔); either type may be absent.

## 2.2 Structural Causal Models

A structural causal model (SCM) is a versatile and expressive framework that provides a principled way to represent and reason about causal relationships. Its generality makes it applicable across a wide range of domains. We use an SCM as the semantic framework to represent the environment in which the agent operates.

Throughout this paper, we consider recursive SCMs, so that each exogenous realization determines a unique value of every endogenous variable. Unless stated otherwise, all variables and mechanisms considered below satisfy this convention.

Definition 1 (Structural Causal Model (Pearl, 2000)). An SCM M is a quadruple $\langle \mathbf { U } , \mathbf { V } , { \mathcal { F } } , P ( \mathbf { U } ) _ { \mathfrak { \Omega } } ^ { \backslash }$ ⟩, where:

• U is a set of exogenous variables that are determined by factors outside the model and follow a joint distribution P(U);

• V is a set of endogenous variables representing observed variables inside the model;

${ \mathcal { F } } = \{ f _ { V } \} _ { V \in \mathbf { V } }$ is a set of functions (mechanisms) such that $v = f _ { V } ( \mathbf { p a } _ { V } , \mathbf { u } _ { V } )$ where $\mathbf { P a } _ { V } \subseteq \mathbf { V } \setminus \{ V \}$ and $\mathbf { U } _ { V } \subseteq \mathbf { U }$ ;

$P ( \mathbf { U } )$ is a probability distribution over the exogenous variables.

## 2.3 Pearl’s Causal Hierarchy

An SCM induces all quantities within Pearl’s Causal Hierarchy (PCH) (Pearl and Mackenzie, 2018; Bareinboim et al., 2022). The three layers of the PCH correspond to distinct regimes of reasoning about an environment: seeing, doing, and imagining (Table 1).

<table><tr><td></td><td>Layer (Symbolic)</td><td>Typical Activity</td><td>Typical Question</td><td>Example</td><td>Machine Learning</td></tr><tr><td> $\mathcal { L } _ { 1 }$ </td><td>Observational  $P ( y \mid x )$ </td><td>Seeing</td><td>What is? How would seeing X change my belief in Y?</td><td>What does a symptom tell us about the disease?</td><td>Supervised / Unsupervised Learning</td></tr><tr><td> $\mathcal { L } _ { 2 }$ </td><td>Interventional  $P ( y \mid d o ( x ) )$ </td><td>Doing</td><td>What if? What if I do X?</td><td>If I take aspirin, will my headache be cured?</td><td>Multi-armed bandits</td></tr><tr><td> $\mathcal { L } _ { 3 }$ </td><td>Counterfactual  $P ( y _ { x } \mid x ^ { \prime } , y ^ { \prime } )$ </td><td>Imagining</td><td>Why? What if I had acted differently?</td><td>Was it the aspirin that stopped my headache?</td><td></td></tr></table>

Table 1: Pearl’s Causal Hierarchy (Bareinboim et al., 2022).

The first layer $\mathcal { L } _ { 1 }$ represents observational distributions, such as $P ( \boldsymbol { Y } \mid \boldsymbol { x } )$ ; the second layer $\mathcal { L } _ { 2 }$ represents interventional distributions, e.g., $P ( Y \mid d o ( x ) )$ , using the do-operator (Pearl, 1995). The highest layer $\mathcal { L } _ { 3 }$ represents counterfactual distributions, such as $P ( Y _ { x } \mid X = x ^ { \prime } )$ (Heckman and Robb Jr, 1985; Heckman and Robb, 1986): the distribution of Y under intervention $d o ( X = x )$ among units whose observed treatment is $x ^ { \prime } .$ . Such distributions underlie quantities such as the efect of treatment on the treated (Heckman and Robb Jr, 1985; Heckman and Robb, 1986).

## 2.3.1 Observational Distributions

An agent perceives the deployed environment through its sensors by “seeing”, which can be formalized through the first layer of the PCH $( \mathcal { L } _ { 1 } )$

Definition 2 (Observational distributions (Bareinboim et al., 2022)). An SCM $\mathcal { M } =$ $\langle \mathbf { U } , \mathbf { V } , { \mathcal { F } } , P ( \mathbf { U } ) \rangle$ defines a joint probability distribution $P ( \mathbf { V } )$ such that for each $\mathbf { Y } \subseteq \mathbf { V }$

$$
P ( \mathbf { y } ) = \sum _ { \mathbf { u } } \mathbb { 1 } \{ \mathbf { Y } ( \mathbf { u } ) = \mathbf { y } \} P ( \mathbf { u } ) ,\tag{1}
$$

where $\mathbf { Y } ( \mathbf { u } )$ is the solution for Y after evaluating F with $\mathbf { U } = \mathbf { u } ^ { 1 }$

Therefore, observational distributions are computed using the following procedure:

1. For each exogenous realization $\mathbf { U } = \mathbf { u } ^ { 2 }$ , the environment evaluates functions in F according to a topological order (i.e., any variable on the left-hand side is evaluated after those on the right-hand side); and

2. The probability $P ( \mathbf { u } )$ is accumulated for each realization U = u consistent with $\mathbf { Y } = \mathbf { y }$ Example 1 (Clinical system (Zhang, 2023; Bareinboim et al., 2024)). In a clinic treating patients with a chronic disease, physicians must choose a treatment. There are two possible

treatments, $X = 0$ and $X = 1$ . The agent’s goal is to identify the treatment that maximizes the probability of patient recovery (Y ). Consider an SCM M that models this system:

$$
{ \mathcal { M } } = { \{ \begin{array} { l l } { \mathbf { U } = \{ U \} } \\ { \mathbf { V } = \{ X , Y \} } \\ { { \mathcal { F } } = \{ f _ { X } ( u ) = 1 \{ u < 0 . 8 \}  } \\ {  f _ { Y } ( x , u ) = 1 \{ u < 0 . 4 - \eta \cdot x \}  } \\ {  P ( \mathbf { U } ) = \{ U \sim { \mathrm { U n i f } } ( 0 , 1 ) \} , } \end{array}  }\tag{2}
$$

where the coeficient η is a real number bounded in (0, 0.4).

In words, the physician prescribes treatment $X = 0$ for senior patients $( U \ge 0 . 8 )$ ; otherwise, the alternative treatment $X = 1$ is prescribed. The recovery Y depends on the treatment X and the patient’s age U.

By passively observing patient outcomes and collecting data, the agent can estimate the mean recovery rate as follows:

$$
P ( Y = 1 ) = P ( X = 0 , Y = 1 ) + P ( X = 1 , Y = 1 )\tag{3}
$$

$$
= P ( X = 0 , Y = 1 ) + P ( Y = 1 \mid X = 1 ) P ( X = 1 )\tag{4}
$$

$$
= \underbrace { P ( U \geq 0 . 8 , U < 0 . 4 - \eta \cdot X ) } _ { = 0 } + P ( Y = 1 \mid X = 1 ) P ( U < 0 . 8 )\tag{5}
$$

$$
= 0 . 8 \cdot P ( Y = 1 | X = 1 ) .\tag{6}
$$

The agent may also estimate the recovery rate among patients who received each treatment. Among patients who received treatment X = 0, the recovery rate is:

$$
P ( Y = 1 \mid X = 0 ) = P ( U < 0 . 4 - \eta \cdot X \mid U \geq 0 . 8 )\tag{7}
$$

$$
\leq P ( U < 0 . 4 \mid U \geq 0 . 8 )\tag{8}
$$

$$
\mathit { \Theta } = 0 .\tag{9}
$$

Similarly, the recovery rate for X = 1 is:

$$
\begin{array} { c } { { P ( Y = 1 \mid X = 1 ) = P ( U < 0 . 4 - \eta \cdot X \mid U < 0 . 8 ) } } \\ { { { } } } \\ { { = 0 . 5 - 1 . 2 5 \eta . } } \end{array}\tag{10}
$$

(11)

Therefore, the mean recovery rate is given by:

$$
\begin{array} { c } { { P ( Y = 1 ) = 0 . 8 \cdot P ( Y = 1 \mid X = 1 ) } } \\ { { = 0 . 4 - \eta . } } \end{array}\tag{12}
$$

(13)

Since the coeficient η lies in (0, 0.4), we can conclude:

$$
P ( Y = 1 \mid X = 0 ) < P ( Y = 1 \mid X = 1 ) .\tag{14}
$$

This result from an observational study suggests that treatment $X = 1$ is better than treatment X = 0.

## 2.3.2 Interventional Distributions

The agent interacts with the environment through the notion of “doing”; that is, by intervening (acting) on the SCM (environment), thereby modifying the original system. We formally define the environment that arises under such interventions using submodels.

Definition 3 (Submodel (Bareinboim et al., 2024)). Let $\mathcal { M } = \langle \mathbf { U } , \mathbf { V } , \mathcal { F } , P ( \mathbf { U } ) \rangle$ ⟩ be an SCM, and let x be a particular realization of a set of variables $\mathbf { X } \subseteq \mathbf { V }$ . A submodel $\mathcal { M } _ { \mathbf { x } }$ of M is an SCM $\langle \mathbf { U } , \mathbf { V } , \mathcal { F } _ { \mathbf { x } } , P ( \mathbf { U } ) \rangle$ ⟩ where:

$$
{ \mathcal { F } } _ { \mathbf { x } } = \{ f _ { V } : V \in \mathbf { V } \setminus \mathbf { X } \} \cup \{ f _ { X } = x : X \in \mathbf { X } , x { \mathrm { ~ i s ~ t h e ~ v a l u e ~ o f ~ } } X { \mathrm { ~ i n ~ } } \mathbf { x } \} .\tag{15}
$$

In words, an intervention is modeled by replacing the original mechanisms associated with variables X with constant values $\mathbf { x } ,$ , as represented by the do-operator, i.e., $d o ( \mathbf { X } = \mathbf { x } )$ (for short, $d o ( \mathbf { x } ) )$ ). The outcome under the intervention is called the potential response (Pearl, 2000).

Definition 4 (Potential response (Bareinboim et al., 2022)). Given an SCM ${ \mathcal { M } } = \langle { \bf U } , { \bf V } , { \mathcal { F } } , P ( { \bf U } ) \rangle$ , let X and Y be two sets of variables in V, and let u be a realization of U. The potential response ${ \bf Y } _ { \bf x } ( { \bf u } )$ is defined as the solution for Y obtained from the system $\mathcal { F } _ { \mathbf { x } }$ with $\mathbf { U } = \mathbf { u }$ in $\mathcal { M }$

For a fixed intervention $\mathbf { x } ,$ the potential response becomes a random variable by evaluating it at the random exogenous variables: $\mathbf { Y } _ { \mathbf { x } } = \mathbf { Y } _ { \mathbf { x } } ( \mathbf { U } )$ , with ${ \bf U } \sim { \cal P } ( { \bf U } ) ^ { 3 }$ . Its distribution is induced by $P ( \mathbf { U } ) ;$ ; averaging a scalar potential response gives its expectation. We next define the resulting interventional distribution.

Definition 5 (Interventional distribution (Bareinboim et al., 2022)). An SCM $\mathcal { M } =$ $\langle \mathbf { U } , \mathbf { V } , { \mathcal { F } } , P ( \mathbf { U } ) \rangle$ induces a collection of joint probability distributions over V, one for each intervention x. For each $\mathbf { Y } \subseteq \mathbf { V }$

$$
P ( \mathbf { y _ { x } } ) = \sum _ { \mathbf { u } } \mathbb { 1 } \{ \mathbf { Y _ { x } } ( \mathbf { u } ) = \mathbf { y } \} P ( \mathbf { u } ) .\tag{16}
$$

Hence, the interventional distribution can be evaluated as follows:

1. Replace the mechanism of each $X \in \mathbf { X }$ with its value in $\mathbf { x } ;$ the resulting $\mathcal { F } _ { \mathbf { x } }$ defines the submodel $\mathcal { M } _ { \mathbf { x } }$

2. For each realization $\mathbf { U } = \mathbf { u }$ , the environment evaluates functions in $\mathcal { F } _ { \mathbf { x } }$ according to a topological order.

3. The probability $P ( \mathbf { u } )$ is accumulated for each realization $\mathbf { U } = \mathbf { u }$ consistent with $\mathbf { Y } = \mathbf { y }$ in the submodel $\mathcal { M } _ { \mathbf { x } }$

In words, the interventional distributions induced by $d o ( \mathbf { x } )$ in an SCM M are distributions over the endogenous variables in the submodel $\mathcal { M } _ { \mathbf { x } }$ . We denote the interventional distribution over Y induced by $d o ( \mathbf { x } )$ interchangeably as $P _ { \mathbf { x } } ( \mathbf { y } ) , P ( \mathbf { y } _ { \mathbf { x } } )$ , or $P ( \mathbf { y } \mid d o ( \mathbf { x } ) ) ^ { 4 }$

Example 2 (Continued). Suppose that the patient’s age U is unavailable to the agent, so the agent observes only the treatment X and the outcome Y. Consider an intervention $d o ( X = x )$ that sets treatment X to a constant $x \in \{ 0 , 1 \}$ . The submodel $\mathcal { M } _ { x }$ induced by $d o ( X = x )$ is given by:

$$
\mathcal { M } _ { x } = \left\{ \begin{array} { l l } { \mathbf { U } = \{ U \} } \\ { \mathbf { V } = \{ X , Y \} } \\ { \mathcal { F } _ { x } = \left\{ { f _ { X } = x } \atop { f _ { Y } ( x , u ) = 1 \} \{ u < 0 . 4 - \eta \cdot x \} } \right. } \\ { \quad \left. P ( \mathbf { U } ) = \{ U \sim \mathrm { U n i f } ( 0 , 1 ) \} . \right. } \end{array} \right.\tag{17}
$$

Evaluating the patient’s recovery Y in the submodel $\mathcal { M } _ { x }$ gives:

$$
\begin{array} { r } { P _ { X = 0 } ( Y = 1 ) = P ( Y = 1 \mid d o ( X = 0 ) ) } \end{array}\tag{18}
$$

$$
= P ( U < 0 . 4 - \eta \cdot X \mid d o ( X = 0 ) )\tag{19}
$$

$$
= P ( U < 0 . 4 )\tag{20}
$$

$$
= 0 . 4 .\tag{21}
$$

Similarly, the patient’s recovery rate under treatment do(X = 1) is:

$$
P _ { X = 1 } ( Y = 1 ) = P ( Y = 1 \mid d o ( X = 1 ) )\tag{22}
$$

$$
= P ( U < 0 . 4 - \eta \cdot X \mid d o ( X = 1 ) )\tag{23}
$$

$$
= P ( U < 0 . 4 - \eta )\tag{24}
$$

$$
= 0 . 4 - \eta .\tag{25}
$$

Given that $\eta > 0$ , we can conclude:

$$
P _ { X = 0 } ( Y = 1 ) > P _ { X = 1 } ( Y = 1 ) .\tag{26}
$$

Thus, treatment X = 0 yields a higher recovery rate under intervention. This reverses the observational ranking in Eq. (14): patients receiving the two treatments difer in U, whereas an intervention fixes the treatment while preserving the population distribution of U. The example therefore distinguishes observational association from the efect of intervening on treatment.

## 2.4 Causal Diagrams

While SCMs provide precise semantics for the distributions underlying the PCH, they are typically not directly observable by the learning agent in practice. As a result, reasoning about the system requires explicit assumptions about the underlying causal model. These assumptions are commonly represented using a causal diagram.

Definition 6 (Causal diagram (Pearl, 1995; Bareinboim et al., 2022)). Consider an SCM $\mathcal { M } = \langle \mathbf { U } , \mathbf { V } , \mathcal { F } , P ( \mathbf { U } ) \rangle$ . Then G is said to be a causal diagram of M if it is constructed as follows:

![](images/79e808bae9128f33812862e018ef378cccafa21270ac49a48553493266bc4933.jpg)  
Figure 2: (a, b) Exogenous-explicit diagrams, where each directed edge indicates the input-output relation of the mechanisms in $\mathcal { F } ;$ (c) causal diagram ${ \mathcal { G } } .$

1. Add a vertex for every endogenous variable in V.

2. Add a directed edge $( V \to W )$ if V appears as an argument of $f _ { W } \in \mathcal { F }$

3. Add a bidirected edge $( V  W )$ if the corresponding $\mathbf { U } _ { V } , \mathbf { U } _ { W } \subseteq \mathbf { U }$ are dependent<sup>5</sup> or the corresponding functions $f _ { V } , f _ { W }$ share some $U _ { V W } \in \mathbf { U }$ as an argument.

We focus on recursive SCMs, i.e., acyclic causal diagrams. Intuitively, there is a directed edge from an endogenous variable V to $W \ ( V \to W )$ whenever $W$ “listens to” V in determining its value. Similarly, a bidirected edge between $V$ and W $( V  W )$ indicates the presence of shared unobserved factors that influence both variables.

While an SCM $\mathcal { M } = \langle \mathbf { U } , \mathbf { V } , \mathcal { F } , P ( \mathbf { U } ) \rangle$ contains explicit quantitative information about the structural mechanisms $\mathcal { F }$ and the exogenous distribution $P ( \mathbf { U } )$ , a causal diagram $\mathcal { G }$ encodes only qualitative information about which variables $m a y ^ { 6 }$ serve as inputs to the functions $f _ { V } \in \mathcal { F }$ and how the exogenous variations are related. Hence, the diagram abstracts away the specific forms of the mechanisms $\mathcal { F }$ and the distribution $P ( \mathbf { U } )$ , retaining only qualitative information about their arguments and independence structure.

Example 3 (Continued). Let us consider the running example M. Fig. 2a illustrates an abstraction of M. The causal mechanisms $\mathcal { F }$ are structural functions of the form:

$$
\mathcal { F } = \left\{ \begin{array} { l l } { \displaystyle x = f _ { X } ( \boldsymbol { u } ) } \\ { \displaystyle y = f _ { Y } ( \boldsymbol { u } , \boldsymbol { x } ) . } \end{array} \right.\tag{27}
$$

To apply the graphical construction, we start the process by examining each of the endogenous variables $\mathbf { V } = \{ X , Y \}$ and adding them as nodes in the causal diagram $\mathcal { G }$ . The corresponding diagram is illustrated in Fig. $\mathrm { 2 c ^ { 7 } }$

For concreteness, we consider an additional instance of the following form:

$$
\mathcal { M } ^ { \prime } = \left\{ \begin{array} { l l } { \mathbf { U } = \{ U _ { X } , U _ { Y } , U _ { 1 } , U _ { 2 } \} } \\ { \mathbf { V } = \{ X , Y \} } \\ { \mathcal { F } = \left\{ \begin{array} { l l } { x = f _ { X } ( u _ { X } , u _ { 1 } , u _ { 2 } ) } \\ { y = f _ { Y } ( x , u _ { Y } , u _ { 1 } , u _ { 2 } ) } \\ { P ( \mathbf { U } ) = \{ U \sim \mathrm { U n i f } ( 0 , 1 ) : \forall U \in \mathbf { U } \} . } \end{array} \right. } \end{array} \right.\tag{28}
$$

Take the exogenous variables to be mutually independent, so the displayed uniform marginals specify their joint distribution. In this setting, $\mathcal { M } ^ { \prime }$ induces the same causal diagram $\mathcal { G }$ , where $\mathbf { U } _ { X } = \{ U _ { X } , U _ { 1 } , U _ { 2 } \}$ and $\mathbf { U } _ { Y } = \{ U _ { Y } , U _ { 1 } , U _ { 2 } \}$ . The diagram gives a compact summary of the possible directed and confounding relationships between $X$ and $Y .$ , while omitting the identities and number of exogenous variables and the quantitative forms of the mechanisms.

## 2.4.1 Graphical Notation

If there is an edge between two vertices X and Y in ${ \mathcal { G } } ,$ we say that the two vertices are adjacent in ${ \mathcal { G } } ,$ denoted by $Y \in { \mathsf { A d j } } ( X ) _ { \mathcal { G } }$ or $X \in \mathsf { A d j } ( Y ) _ { \mathcal { G } }$ . A path is a sequence of distinct vertices, with a specified edge connecting each consecutive pair. When two vertices are joined by both a directed and a bidirected edge, the path specifies which edge is traversed. If a path consists of directed edges with the same orientation, we say the path is directed. A non-endpoint vertex is a collider on a path when both adjacent path edges have arrowheads at that vertex, as in $A \to B  C .$ . Otherwise, it is a non-collider on that path. A path is a collider path if all its non-endpoint vertices are colliders.

We use kinship notation for graphical relationships: parents $\left( \mathsf { P a } \right)$ , children (Ch), ancestors (An), and descendants (De).

$\mathtt { P a } ( X ) _ { \mathcal { G } }$ (parents of X in $\mathcal { G } )$ : the set $\left\{ Z _ { i } \right\}$ such that there exists a directed edge $Z _ { i } \to X$ in $\mathcal { G } .$

• Ch $( X ) _ { \mathcal { G } }$ (children of $X$ in $\mathcal { G } )$ : the set $\left\{ Z _ { i } \right\}$ such that there exists a directed edge $X  Z _ { i }$ in $\mathcal { G }$

$\mathtt { A n } ( X ) _ { \mathcal { G } }$ (ancestors of X in $\mathcal { G } )$ : the set $\{ Z _ { i } \}$ such that there exists a directed path $Z _ { i } \to \cdots \to X$ in $\mathcal { G }$

$\mathtt { D e } ( X ) _ { \mathcal { G } }$ (descendants of X in $\mathcal { G } )$ : the set $\left\{ Z _ { i } \right\}$ such that there exists a directed path $X  \cdots  Z _ { i }$ in $\mathcal { G }$

Ancestors and descendants include the variable itself. For a set of variables, we define ${ \begin{array} { r } { \mathtt { A n } ( \mathbf { X } ) _ { \mathcal { G } } = \bigcup _ { X \in \mathbf { X } } \mathtt { A n } ( X ) _ { \mathcal { G } } } \end{array} }$ , and similarly for other relationships.

The X-lower-manipulation of $\mathcal { G }$ removes all directed edges out of variables in X, denoted as $\mathcal { G } _ { \mathbf { X } }$ , while the X-upper-manipulation of $\mathcal { G }$ removes all edges with an arrowhead at a variable in X, including bidirected edges, denoted as $\mathcal { G } _ { \overline { { \mathbf { X } } } }$ . We denote the set of variables in $\mathcal { G }$ by $\mathbf { V } ( { \mathcal { G } } )$ A subgraph $\mathcal { G } [ \mathbf { V } ^ { \prime } ]$ , where $\mathbf { V } ^ { \prime } \subseteq \mathbf { V } ( { \mathcal { G } } )$ , is defined as a vertex-induced subgraph in which all edges among the vertices in $\mathbf { V } ^ { \prime }$ are preserved. We define $\mathcal { G } \backslash \mathbf { X }$ as ${ \mathcal { G } } [ \mathbf { V } ( { \mathcal { G } } ) \setminus \mathbf { X } ]$ for $\mathbf { X } \subseteq \mathbf { V } ( { \mathcal { G } } )$

Definition 7 (Latent projection (Verma and Pearl, 1990)). The latent projection of a causal diagram $\mathcal { G }$ over V onto $\mathbf { X } \subseteq \mathbf { V }$ is a mixed graph over X. It retains the edges in $\mathcal { G } [ \mathbf { X } ]$ and, for every pair of distinct vertices $V _ { i } , V _ { j } \in \mathbf { X }$ , includes the following additional edges:

1. Add a directed edge $V _ { i } \to V _ { j }$ if there exists a directed path from $V _ { i }$ to $V _ { j }$ in $\mathcal { G }$ such that every non-endpoint vertex on the path is not in X.

2. Add a bidirected edge $V _ { i }  V _ { j }$ if there exists a path between $V _ { i }$ and $V _ { j }$ in $\mathcal { G }$ with arrowheads at both endpoints, such that every non-endpoint vertex is a non-collider and lies outside X.

We denote by $\mathcal { G } \langle { \bf X } \rangle$ the latent projection of $\mathcal { G }$ onto $\mathbf { X }$

## 2.4.2 Graphical Constraints

The structure of a causal diagram imposes constraints on the data generated by any causal model that is consistent with the diagram. One such constraint is conditional independence (CI), which can be inferred from the causal diagram via the d-separation criterion.

Definition 8 (d-separation (Pearl, 2000)). In a causal diagram G, let X and Y be distinct vertices and let $\mathbf { Z } \subseteq \mathbf { V } \setminus \{ X , Y \}$ . A path p between vertices X and Y is a d-connecting (active) path relative to a set Z if:

1. every non-collider on $p$ is not a member of $\mathbf { Z } ;$ and

2. every collider on p is an ancestor of some member of $\mathbf { Z } .$

Two variables X and Y are said to be d-separated by Z if there is no d-connecting path between X and Y relative to Z. Furthermore, two disjoint sets X and Y are said to be d-separated by Z if every variable in X is d-separated from every variable in Y by $\mathbf { Z } ;$ this is denoted as $( \mathbf { X } \perp \perp \mathbf { Y } \mid \mathbf { Z } ) _ { \mathcal { G } } ^ { 8 }$

We now present a key result that relates d-separation statements derived from a causal diagram $\mathcal { G }$ to constraints on the distribution $P ( \mathbf { V } )$

Theorem 1 (Probabilistic implications of d-separation (Pearl, 2000)). If X and Y are d-separated by Z in a causal diagram ${ \mathcal { G } } ,$ then X is independent of Y given Z in every distribution P compatible with G. Conversely, if X and Y are not d-separated by Z in $\mathcal { G }$ then X and Y are dependent given Z in at least one distribution P compatible with $\mathcal { G }$

For concreteness, we consider the causal diagram in Fig. 3. The path $p = \langle X , W , T , Y \rangle$ is active relative to $\mathbf { Z } = \{ W \}$ since T /∈ Z is a non-collider and $W \in \mathbf { Z }$ is a collider. Alternatively, the path is blocked relative to $\mathbf { Z } ^ { \prime } = \{ W , T \}$ since ${ \boldsymbol { T } } \in \mathbf { Z ^ { \prime } }$ is a non-collider along $p .$ Moreover, $p ^ { \prime } = \langle X , W , Y \rangle$ is blocked relative to $\mathbf { Z } ^ { \prime }$ since $W \in \mathbf { Z } ^ { \prime }$ is a non-collider along $p ^ { \prime }$ Hence, X and Y are d-separated relative to $\mathbf { Z } ^ { \prime } = \{ W , T \}$ . Therefore, Thm. 1 states that:

$$
( X \ \perp Y \mid W , T ) _ { \mathcal { G } } \implies ( X \perp Y \mid W , T ) _ { P } .\tag{29}
$$

![](images/5d00e0874f87b4b2d7cb62bb508d8daf77acfb44d476db5be6b4d5d96be21efd.jpg)  
Figure 3: An example causal diagram for illustrating d-separation.

## 2.5 Ancestral Graphs and Causal Discovery

Recovering causal structures. Causal discovery addresses the reverse direction: inferring from P(V) which causal diagrams could have generated it (Pearl, 2000; Spirtes et al., 2000). In practice, assumptions regarding model simplicity (\`a la Occam’s razor) are used to avoid situations in which a saturated model would otherwise be preferred. A growing literature focuses on constructing equivalence classes of causal diagrams, acknowledging the dificulty of identifying a single true causal diagram from data (Zhang, 2008a; Hauser and B¨uhlmann, 2012; Kocaoglu et al., 2017; Greenewald et al., 2019; Squires et al., 2020; Choo and Shiragur, 2023; Zhou et al., 2025).

The simplest and most abundant form of data is observational, and conditional independence is a key property that is imprinted in the data by the underlying causal diagram via dseparation, as discussed earlier. Consequently, causal diagrams that encode the same set of d-separations imply the same CIs in the generated data. Hence, the goal is to leverage the CIs in the data to learn the equivalence class of the true, but unknown, causal diagram.

Markov equivalence class. The following definition formally introduces this equivalence class, known as the Markov equivalence class (MEC) (Richardson and Spirtes, 2002; Zhang, 2006, 2008a).

Definition 9 (Markov equivalence). Two graphs are in the same Markov equivalence class (MEC) if they imply the same set of conditional independence statements.

We use E to denote a Markov equivalence class. To denote a particular equivalence class to which a model G belongs, we write E(G).

Ancestral graphs. Ancestral graphs are designed to capture graph structures without explicitly modeling latent variables. While directed edges between vertices in a causal diagram represent a possible direct causal link between them, in ancestral graphs, directed edges instead represent ancestral relationships. Similar to the absence of directed cycles in causal diagrams, ancestral graphs do not permit an almost directed cycle, which occurs when X ↔ Y is present while X is an ancestor of Y .

Maximal ancestral graphs. We consider the case without selection variables, so undirected edges are excluded. A mixed graph is called a maximal ancestral graph (MAG) if:

1 (Ancestral). It does not contain any directed or almost directed cycles; and

2 (Maximal). There is no inducing path relative to the empty $\mathrm { s e t ^ { 9 } }$ between any two non-adjacent vertices.

A MAG entails a conditional independence model via m-separation, defined below. Its path-blocking conditions are the same as those used for causal diagrams above; the diferent name indicates that we now apply them to ancestral graphs<sup>10</sup>.

Definition 10 (m-separation). Let X and Y be distinct vertices and let $\mathbf { Z } \subseteq \mathbf { V } \setminus \{ X , Y \}$ A path p between vertices X and Y is an m-connecting (active) path relative to a set Z if:

1. Every non-collider on p is not a member of $\mathbf { Z } ;$ and

2. Every collider on p is an ancestor of some member of $\mathbf { Z }$

Two variables X and Y are m-separated by Z if there is no m-connecting path between X and Y relative to Z. Two disjoint sets X and Y are said to be m-separated by Z if every variable in X is m-separated from every variable in Y by Z.

In general, a MAG represents a set of causal diagrams that entail the same conditiona independence and ancestral relations among the observed variables. For each causal diagram, there is a unique corresponding MAG over the observed variables that preserves their ancestral relations and the conditional independence relations implied by the graph after marginalizing the latent variables. Distinct MAGs can imply the same conditional independence relations among the observed variables. To illustrate, consider the causal diagrams $\mathcal { G } _ { 1 }$ and $\mathcal { G } _ { 2 }$ in Fig. 4. While they yield the same conditional independence relations:

$$
( A \perp B ) _ { P } { \mathrm { ~ a n d ~ } } ( A \perp Y \mid B , C ) _ { P }\tag{30}
$$

they correspond to distinct MAGs, $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ , respectively. The following graphical constructs are key to deriving a graphical characterization for when two MAGs are Markov equivalent.

The skeleton is obtained by ignoring all endpoint marks. A triple $\langle A , B , C \rangle$ is unshielded if A and B are adjacent, B and C are adjacent, and A and $C$ are not adjacent. A path is uncovered if every consecutive triple on the path is unshielded. This condition concerns vertices two steps apart; it does not, by itself, prohibit every edge between nonconsecutive vertices.

Definition 11 (Discriminating path). A path $p = \langle X , \cdot \cdot \cdot , W , Z , Y \rangle$ is a discriminating path for Z if:

• p includes at least three edges;

• $Z$ is a non-endpoint vertex on p, and is adjacent to $Y$ on $p ;$ and

• X is not adjacent to $Y .$ , and every vertex between X and $Z$ is a collider on $p$ and is a parent of $Y$

(e) P  
![](images/1d5e117fb6244f28343561fea43d41e90d03c23a71a7e32f6d4dd02aed058637.jpg)  
(a) G<sub>1</sub>

![](images/2b5161a124742941d44c6e4f05ffab2acab436f1e62eab9679c9a58a10a5f52e.jpg)  
(b) G<sub>2</sub>

![](images/118eb6599e2d882005e87982695c8826974b987a72b743d4d21b7255d9da6d47.jpg)  
(c) M<sub>1</sub>

![](images/4fdf5a826c4f0a6a6381e41cd23f1db2efd7751ee0ab407aa2216597e0b28095.jpg)  
(d) M<sub>2</sub>

![](images/1d76a3eb4478413b2d8fd3d7a5905fb192e0874312fa762749acb1110c9b6573.jpg)  
Figure 4: Causal diagrams (a, b) with corresponding (c, d) MAGs and (e) a PAG. Purple directed edges are visible.

For example, consider the MAG in Fig. 4c. The path $p = \langle A , C , B , Y \rangle$ is a discriminating path for B between A and Y. The path includes at least three edges, A is not adjacent to Y , and C is a collider along p and a parent of Y . Using this structure, we have the following graphical characterization for when two MAGs are Markov equivalent.

Lemma 1 (Graphical characterization of MEC (Spirtes and Richardson, 1997; Zhang, 2007)). Two MAGs over the same set of vertices are Markov equivalent if and only if:

1. they have the same adjacencies;

2. they have the same unshielded colliders; and

3. for every path that is discriminating for the same vertex in both graphs, that vertex has the same collider status on the path in the two graphs.

In words, two MAGs are Markov equivalent if they have the same skeleton and the same unshielded colliders (also known as v-structures), and if every discriminating path common to both graphs identifies a collider in one graph if and only if it identifies a collider in the other.<sup>11</sup>

For example, $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ in Figs. 4c and 4d are Markov equivalent because:

1. they have the same adjacencies A∗→ C, B ∗→ C, B ∗→ Y and $C \ast  Y$

2. they have the same unshielded collider $A * \to C  * B$

3. the path $p = \langle A , C , B , Y \rangle$ is a discriminating path for B in both graphs, and B is a non-collider on p in both graphs.

This characterization of equivalent MAGs motivates the formulation of a partial ancestral graph (PAG), which is a partial mixed graph (PMG) representing an equivalence class of MAGs. A PAG shares the same adjacencies as every MAG in $\mathcal { E } ( \mathcal { M } )$ and displays all and only the invariant edge marks. To make this precise, we first introduce PMGs and then PAGs.

Partial mixed graphs. A partial mixed graph (PMG) has endpoint marks drawn from tails (−), arrowheads (>), and circles (◦). A circle mark implies an undetermined mark that can be either an arrowhead or a tail. We use the ? mark to denote a wildcard that represents either a tail $( - )$ or a circle $( \circ )$ , but not an arrowhead (>). In addition, we use an asterisk (∗) as a wildcard to denote any possible mark.

In a PMG, if every edge mark on a path consists of circles, the path is called a circle path, and each edge is called a circle edge (◦−◦). An edge is a partially directed edge (◦→) if it has both a circle and an arrowhead. A circle component is a maximal set of vertices connected by circle paths. A vertex incident to no circle edge forms a singleton circle component. Furthermore, [Q] denotes the set of MAGs represented by the PMG Q, and similarly [M] denotes the set of causal diagrams conforming to the MAG M.

A path $\langle V _ { 0 } ( = X ) , V _ { 1 } , \cdot \cdot \cdot , V _ { k } ( = Y ) \rangle$ is possibly directed from X to Y if, for every $i < k .$ , the edge between $V _ { i }$ and $V _ { i + 1 }$ has no arrowhead at $V _ { i }$ and no tail at $V _ { i + 1 }$ . We use kinship notation for graphical relationships in PMGs, with the prefix Poss: possible ancestors (PossAn) and possible descendants (PossDe).

• PossAn $( X ) _ { \mathcal { Q } }$ (possible ancestors of X in Q): the set $\left\{ Z _ { i } \right\}$ such that there exists a possibly directed path from $Z _ { i }$ to X in Q.

• PossDe $( X ) _ { \mathcal { Q } }$ (possible descendants of X in $\mathcal { Q } )$ : the set $\{ Z _ { i } \}$ such that there exists a possibly directed path from X to $Z _ { i }$ in $\mathcal { Q } .$

Both possible ancestors and possible descendants include the variable itself. For a set of variables X, define:

$$
\operatorname { P o s s A n } ( \mathbf { X } ) _ { \mathcal { Q } } = \bigcup _ { X \in \mathbf { X } } \operatorname { P o s s A n } ( X ) _ { \mathcal { Q } } ,\tag{31}
$$

and define PossDe $( \mathbf { X } ) _ { \Omega }$ analogously.

Partial ancestral graphs. A partial ancestral graph (PAG) (Richardson and Spirtes, 2002; Zhang, 2006, 2008a,b; Ali et al., 2005), denoted by P, is a PMG that represents a Markov equivalence class of MAGs. Every MAG M represented by a PAG has the same skeleton as $\mathcal { P } _ { \cdot }$ , and the non-circle marks in P are identical to those in M. Every circle in P corresponds to a non-invariant mark among the represented MAGs. The PAG $\mathcal { P }$ in Fig. 4e, for instance, encodes every MAG obtained by orienting circle marks incident to A and B as either $> \mathrm { o r } -$ , including both $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ . Therefore, $\{ \mathcal { M } _ { 1 } , \mathcal { M } _ { 2 } \} \subseteq [ \mathcal { P } ] = \mathcal { E } ( \mathcal { M } _ { 1 } ) = \mathcal { E } ( \mathcal { M } _ { 2 } )$

Fast Causal Inference (FCI). Assume the causal Markov condition and faithfulness with respect to the true MAG $\mathcal { M } ^ { \ast }$ , with no selection bias. The causal Markov condition states that graphical separation implies conditional independence in the observational distribution; faithfulness requires the converse. For pairwise disjoint sets X, Y, $\mathbf { Z } \subseteq \mathbf { V }$ , faithfulness states that:

$$
( \mathbf { X } \perp \mathbf { Y } \mid \mathbf { Z } ) _ { P } \implies ( \mathbf { X } \perp \mathbf { Y } \mid \mathbf { Z } ) _ { \mathcal { M } ^ { * } } .\tag{32}
$$

```latex
Algorithm 1: An oracle-level schematic of FCI for learning a PAG.
Input: Observational distribution P(V)
Output: A PAG P
// Phase 1: Learn separating sets and adjacencies.
1 Form a complete graph P over V composed of circle edges (◦−◦)
2 for each unordered pair of distinct vertices $X , Y \in \mathbf { V }$ do
3 Set Sepflag(X, Y ) = False.
4 for $\mathbf { Z } \subseteq \mathbf { V } \setminus \{ X , Y \}$ do
5 if $( X \perp \perp Y \mid \mathbf { Z } ) _ { P }$ then
6 Set Sepset(X, Y) = Sepset(Y, X) = Z.
7 Set Sepflag(X, Y ) = True.
8 break
9 if Sepflag(X, Y ) then
10 Remove the edge between X and Y from P.
// Phase 2: Learn v-structures.
11 Apply $\textstyle { \mathcal { R } } _ { 0 }$ until none applies.
// Phase 3: Apply orientation rules.
12 Apply the seven rules $\mathcal { R } _ { 1 } { - } \mathcal { R } _ { 4 } , \mathcal { R } _ { 8 } { - } \mathcal { R } _ { 1 0 }$ until none applies.
```

Given an oracle for these conditional independence relations, FCI (Spirtes et al., 2000; Zhang, 2008a) recovers the PAG representing the Markov equivalence class of $\mathcal { M } ^ { * }$ . With finite data, conditional independence tests estimate this oracle information.

Alg. 1 gives an oracle-level schematic of FCI, using an exhaustive search for separating sets to simplify the presentation. Its three phases describe the logical steps of recovering adjacencies, identifying unshielded colliders, and completing invariant orientations.

In Phase 1, we initialize a complete graph over the nodes in V consisting of circle edges (◦−◦). For every pair of nodes $X , Y$ , we remove the edge between X and Y if there exists a separating set Z such that $( X \perp \perp Y \mid \mathbf { Z } ) _ { P }$ . By the end of Phase 1, we obtain the skeleton of the resulting PAG P.

In Phase 2, we identify all v-structures by testing $\mathcal { R } _ { 0 }$

$\mathcal { R } _ { 0 }$ . For each unshielded triple ⟨X, Z, Y ⟩ in P, orient it as a collider $X * \to Z  * Y$ if and only if Z is not in Sepset(X, Y ).

Fig. 5 illustrates $\mathcal { R } _ { 0 }$ and the arrowhead orientation rules $\mathcal { R } _ { 1 } - \mathcal { R } _ { 4 }$

Finally, in Phase 3, we apply the orientation rules of Zhang (2008a) other than $\mathcal { R } _ { 5 } { - } \mathcal { R } _ { 7 }$ until none applies. These three rules are omitted because selection bias is excluded. Fig. 6 illustrates the tail orientation rules $\mathcal { R } _ { 8 } \mathcal { R } _ { 1 0 }$

$\mathcal { R } _ { 1 }$ . If $X * \to Z \circ - * Y$ , and X and Y are not adjacent, then orient $Z \circ - * Y$ as $Z \to Y$

$\mathcal { R } _ { 2 } .$ . If $X \to Z * \to Y$ or $X * \to Z \to Y$ , and $X { \ast { - } \circ } Y$ , then orient $X { \ast { \mathrm { - } } \circ Y }$ as $X \not \to Y$ $\mathcal { R } _ { 3 }$ . If $X * \to Z  * Y$ 2 $X * - \circ W \circ - * Y$ , X and $Y$ are not adjacent, and $W \mathrel { \ast } \mathrm { - } \mathrm { o } Z$ , then orient $W { \ast } { - } \circ Z$ as $W \not \to Z$

![](images/0127bd53ce98007dd239aca67fc629f69eb862de1e3a302f923ca36f55b49b7e.jpg)  
Figure 5: Illustration of the FCI arrowhead (>) orientation rules $\mathcal { R } _ { 0 } { - } \mathcal { R } _ { 4 }$

$\mathcal { R } _ { 4 }$ . If $\langle X , \cdots , W , Z , Y \rangle$ is a discriminating path between X and Y for Z, and $Z \circ - * Y$ ， then orient $Z \ \circ { \mathrm { - * } } \ Y$ as $Z \to Y$ if $Z \in { \mathsf { S e p s e t } } ( X , Y )$ ; otherwise orient the triple $\langle W , Z , Y \rangle$ as $W  Z  Y$

$\mathcal { R } _ { 8 }$ . If $X  Z  Y$ , and $X \circ  Y$ , orient $X \circ  Y$ as $X  Y$

$\mathcal { R } _ { 9 }$ . If $X \circ \to Y$ , and $p = \langle X , Z , W , \cdot \cdot \cdot , Y \rangle$ is an uncovered possibly directed path from X to $Y$ such that Z and Y are not adjacent, then orient $X \circ  Y$ as $X  Y$

$\mathcal { R } _ { 1 0 }$ . Suppose $X \circ  Y , Z  Y  W , p _ { 1 }$ is an uncovered possibly directed path from X to $Z ,$ and $p _ { 2 }$ is an uncovered possibly directed path from X to W. Let U be the vertex adjacent to X on $p _ { 1 }$ (U could be $Z )$ , and V be the vertex adjacent to X on $p _ { 2 }$ (V could be W). If U and V are distinct, and not adjacent, then orient $X \circ  Y$ as $X  Y$

![](images/820d2c814e406183b9aa8bdb116601a88bd1db5e77738b47d07f090f78a3696c.jpg)

![](images/159366b66b978245b161def0dd64a080519ff4879b9ac30363c64649239ab116.jpg)  
(c) $\mathcal { R } _ { 1 0 }$

Figure 6: Illustration of the FCI tail (−) orientation rules $\mathcal { R } _ { 8 } \mathcal { R } _ { 1 0 }$ . In $\mathcal { R } _ { 9 }$ , Z is the first vertex after X on the displayed path and is not adjacent to Y. In $\mathcal { R } _ { 1 0 }$ , U and V are the first vertices after X on the two displayed paths; they are distinct and nonadjacent.

![](images/fffbfb5d7aae4fa72316faa46cfd4371f0cba629b286fa5e52d84481e4f5d488.jpg)  
(a)

![](images/cafd5e3172031eeb769cde9d49112d11bc0226d5b5e82a583d301ecc37ed4635.jpg)  
(b) Phase 1

![](images/43c8241f4cba9a3e7e253c2475111dca224aefd3a077a518e06431077658fe4c.jpg)  
(c) Phase 2

![](images/05fba9b9bed3827e7f3ea5563d82462a709f50c3886eae64f16265e3784ed25d.jpg)  
(d) $\mathcal { R } _ { 1 }$

![](images/164385ff790e510260c02cae3650183cd43f4118d1813403585088525fa8da34.jpg)  
(e) $\mathcal { R } _ { 4 }$  
Figure 7: Resulting graphs at each phase of FCI: (a) initial complete graph with circle edges; (b) after Phase 1; (c) after Phase 2; and (d, e) after applying $\mathcal { R } _ { 1 }$ and $\mathcal { R } _ { 4 }$ in Phase 3.

Fig. 7 illustrates how FCI works. We can learn the PAG P in Fig. 7e from observational data generated by the causal diagram in Fig. 4a. Nodes A and B are not adjacent in P because $( A \perp B ) _ { P }$ , and A and Y are not adjacent since $( A \perp Y \mid B , C ) _ { P }$ . Hence, Phase 1 results in the skeleton in Fig. 7b.

In Phase 2, the v-structure $A \circ \to C  \circ B$ is identified since $C \not \in { \mathsf { S e p s e t } } ( A , B )$ , leading to Fig. 7c. In Phase 3, $C \circ { \mathrm { - } } \circ Y$ is oriented as $C  Y$ by $\mathcal { R } _ { 1 }$ since the triple $A \circ  C \circ - \circ Y$ is unshielded, which gives Fig. 7d. Moreover, $B \circ { \mathrm { - } } \circ Y$ is oriented as $B  Y$ by $\mathcal { R } _ { 4 }$ since $\langle A , C , B , Y \rangle$ is a discriminating path between A and Y for B, and $B \in { \mathsf { S e p s e t } } ( A , Y )$ . This yields the PAG in Fig. 7e.

Under the oracle, Markov, and faithfulness assumptions stated above, FCI returns the PAG of the true MAG. Moreover, its orientation rules are sound and complete in the sense that they identify all and only the edge marks that are invariant within the represented Markov equivalence class. To state these properties precisely, we first define orientation soundness and completeness for a general PMG.

Soundness and completeness of orientations. Fix a nonempty target class $\mathcal { I }$ of MAGs with a common skeleton, determined by the input PAG and any additional orientations. Let Q be a PMG on this skeleton.

![](images/ad31f91a734fd6ddc99553bd74fb119a06352faf858398427fe1692281a8d49d.jpg)  
Figure 8: Graphical characterization of visible edges.

• Soundness. Every tail or arrowhead in Q agrees with the corresponding mark in every MAG in $\mathcal { I } .$

• Completeness. Every endpoint mark that is invariant across $\mathcal { I }$ is displayed as a tail or arrowhead in Q.

Thus, when orientations are sound and complete, each circle in Q has both a tail witness and an arrowhead witness in $\mathcal { I } .$ For the PMGs used below, the represented class [Q] is understood relative to the input PAG and the incorporated additional orientations.

For example, consider the PAG P in Fig. 4e, with target class $\mathcal { I } = [ \mathcal { P } ]$ . The arrowheads at $C$ on $A \circ  C$ and $B \circ  C$ are invariant, as are both endpoint marks of $B  Y$ and $C  Y$ Soundness means that these displayed arrowheads and tails agree with every MAG in $\mathcal { I }$ Completeness means that all such invariant marks have been displayed.

The MAGs $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ in Figs. 4c and 4d are two members of the class that witness the variation represented by the remaining circles. In $\mathcal { M } _ { 1 }$ , the edges $A  C$ and $B  C$ have arrowheads at A and $B ,$ whereas in $\mathcal { M } _ { 2 }$ , the corresponding edges $A  C$ and $B  C$ have tails at those endpoints. Thus, neither endpoint mark is invariant, and the PAG correctly retains circles at A and B. Its orientations are nevertheless complete: every remaining circle reflects variation within the target class rather than a missing invariant orientation.

Visibility. In the context of ancestral graphs, it is crucial to be able to identify edges that are not confounded. A directed edge $X  Y$ is visible if no DAG whose corresponding MAG is the given MAG (or belongs to the class represented by the given PAG) contains an inducing path between X and Y, relative to its latent vertices, that is into X. We refer to any edge that is not visible as invisible. For induced subgraphs, we retain the model class specified by the full PMG Q. Let G be a DAG over observed vertices V and latent vertices U whose corresponding MAG belongs to [Q]. A directed edge $X  Y$ in $\mathcal { Q } [ \mathbf { A } ]$ is visible relative to this class if no such DAG contains an inducing path into X between X and Y in $\mathcal { G } [ \mathbf { A } \cup \mathbf { U } ]$ , relative to U. The latent vertices are retained when the observed vertex set is restricted to A.

Lemma 2 (Graphical characterization of visibility (Zhang, 2006; Maathuis and Colombo, 2015)). A directed edge $X  Y$ in a MAG or PAG is visible if and only if there exists a vertex Z not adjacent to Y for which at least one of the following conditions holds:

1. Z and X are joined by an edge with an arrowhead at X.

2. A collider path from Z to X has an arrowhead at X, and all its non-endpoint vertices are parents of Y.

The criterion is evaluated in the full MAG or PAG, relative to its represented causal model class. For a PAG, a graphical visibility witness is preserved in every represented MAG. Conversely, if the criterion fails in the PAG, there is a represented MAG in which the directed edge is invisible (see Lemma 24). For an induced subgraph retaining the full graph’s model class, a visibility witness may lie outside the retained observed vertices; the criterion should therefore not be recomputed using only the induced subgraph.

Fig. 8 illustrates the conditions. Throughout this paper, purple directed edges indicate visibility. For concreteness, $C  Y$ in the PAG in Fig. 7e is visible, indicating that the two nodes are not confounded in any causal diagram in the MEC represented by the PAG. Note that $A \circ  C  Y$ satisfies the first condition of Lemma 2. Moreover, consider the edge $B  Y$ in the MAG of Fig. 4c, which is visible by the second condition of Lemma 2. The path $\langle A , C , B \rangle$ forms a collider path into $B ,$ and $C$ on this path is a parent of Y.

## 3 Structural Causal Bandits

Many canonical decision-making settings, such as multi-armed bandits (MABs), Markov decision processes (MDPs), and dynamic treatment regimes (DTRs), can be represented within the semantic framework of structural causal models (SCMs) (Zhang, 2023; Bareinboim et al., 2024). In this section, we introduce the MAB framework, examine it through a causal lens, and generalize it within the SCM framework, resulting in what is known as structural causal bandits (SCBs) (Lee and Bareinboim, 2018, 2019).

## 3.1 Multi-Armed Bandits

The multi-armed bandit (MAB; simply, bandit) problem is a sequential decision-making problem. One can view this problem as a sequential game between an agent and an environment (i.e., a bandit instance). The game is played over $T$ rounds. In each round $t \in \{ 1 , 2 , \cdots , T \}$ 2 the agent first chooses an action $A _ { t }$ from a given set $\mathcal { A } = \{ a _ { i } \} _ { i = 1 } ^ { | \mathcal { A } | }$ , and the environment then reveals a reward Y<sub>t</sub>. The agent cannot access future information when choosing actions; its action distribution may depend on the observed history:

$$
H _ { t - 1 } = \langle A _ { 1 } , Y _ { 1 } , \cdot \cdot \cdot , A _ { t - 1 } , Y _ { t - 1 } \rangle .\tag{33}
$$

A policy $\pi = \{ \pi _ { t } \} _ { t = 1 } ^ { T }$ assigns an action distribution $\pi _ { t } ( \cdot \mid H _ { t - 1 } )$ to each history. At round $t ,$ the agent samples $A _ { t }$ from this distribution using internal randomness independent of the environment’s randomness. Deterministic policies are included as a special case. The agent interacts with the environment by following a policy. The objective of the agent is to maximize the total reward, which is a random quantity depending on both the agent’s actions and the rewards generated by the environment. Conditional on the selected action $A _ { t } = a$ and the past history, $Y _ { t }$ has distribution $\nu _ { a }$ , which does not change with t. All reward means are assumed finite. In the SCM formulation, each round uses a fresh independent draw of the exogenous variables from the same distribution $P ( \mathbf { U } )$

Unstructured vs. structured bandits. The fundamental challenge in bandit problems is that the environment is unknown to the agent. The agent only knows that the true environment lies in some set C, called the environment class.

![](images/b7701f1077886fb3135f45de83445482f4c9371a62901a265118569a4ff5135e.jpg)  
Figure 9: (a) A multi-armed bandit environment; (b) the submodel induced by the action do(x), representing the post-action model perceived by the agent; and (c) a structural causal bandit environment.

An environment class C is called unstructured if A is finite and playing an action provides no information about the reward distributions of other actions (Lattimore and Szepesv´ari, 2020). Environment classes that are not unstructured are called structured. An important feature of structured bandits is that the agent can often gain information about some actions without explicitly playing them.

Cumulative regret. Let $\nu = \{ \nu _ { a } : a \in \mathcal { A } \}$ be a stochastic bandit instance, and let $\mu _ { a }$ denote the mean reward of action $a \in { \mathcal { A } }$ . Let $a ^ { \star }$ be an optimal action, i.e., $\mu _ { a ^ { \star } } = \operatorname* { m a x } _ { a \in \mathcal { A } } \mu _ { a }$ The cumulative (pseudo-)regret of a policy π on instance ν is defined as:

$$
R _ { T } ^ { A } ( \pi , \nu ) = T \mu _ { a ^ { \star } } - \sum _ { t = 1 } ^ { T } \mu _ { A _ { t } } ,\tag{34}
$$

which compares the cumulative mean reward of always playing an optimal action with the sum of the mean rewards of the selected actions over T rounds.

Define the suboptimality gap as $\Delta _ { a } \triangleq \mu _ { a ^ { \star } } - \mu _ { a }$ for $a \in { \mathcal { A } }$ . By the standard regret decomposition (Lattimore and Szepesv´ari, 2020), we have:

$$
R _ { T } ^ { A } ( \pi , \nu ) = \sum _ { a \in \mathcal { A } } \Delta _ { a } N _ { T } ( a ) ,\tag{35}
$$

where $N _ { T } ( a )$ denotes the number of times action a is selected up to time T. This decomposition shows that minimizing expected cumulative regret amounts to controlling the expected number of pulls of suboptimal actions, weighted by their gaps:

$$
\mathbb { E } R _ { T } ^ { A } ( \pi , \nu ) = \sum _ { a \in \mathcal { A } } \Delta _ { a } \mathbb { E } N _ { T } ( a ) .\tag{36}
$$

The expectation is taken with respect to the probability measure induced by the interaction between π and ν. When clear from context, we omit the dependence on π and ν.

## 3.2 Structural Causal Bandits: Where to Intervene?

We begin by noting that a standard bandit can be viewed as the simple causal model shown in Fig. 9a. Interventions on X correspond to actions with $\Omega _ { X } = \{ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { k } \}$ , and the action space $\mathcal { A } = \{ a _ { i } \} _ { i = 1 } ^ { k }$ is defined as:

$$
\begin{array} { r } { \mathcal { A } = \{ a _ { i } = d o ( x _ { i } ) : x _ { i } \in \Omega _ { X } \} . } \end{array}\tag{37}
$$

The variable Y denotes the reward, and U represents an unobserved confounder. Then $\nu _ { a _ { i } }$ can be written as follows:

$$
\nu _ { a _ { i } } = P ( Y \mid d o ( x _ { i } ) ) = P _ { x _ { i } } ( Y ) .\tag{38}
$$

Eq. (38) states that a standard bandit can be modeled as an SCM associated with the two-node causal diagram in Fig. 2c, highlighting that the classical formulation is a special case of a causal model.

Example 4 (Clinical system; continued). Revisit the clinical-system example. This can be seen as an example of a bandit problem. The environment corresponds to SCM M in Eq. (2) and the action space is defined as:

$$
\mathcal { A } = \{ d o ( X = 0 ) , d o ( X = 1 ) \} .\tag{39}
$$

Furthermore, the bandit instance can be written as follows:

$$
\nu = \{ P _ { X = 0 } ( Y ) = \mathtt { B e r n } ( 0 . 4 ) , P _ { X = 1 } ( Y ) = \mathtt { B e r n } ( 0 . 4 - \eta ) \} ,\tag{40}
$$

as shown in Eqs. (21) and (25).

A careful reader may naturally ask:

## “Why should we restrict ourselves to this simple setting?”

In particular, there is no need to limit the analysis to a two-node structure with a single manipulable variable X. In what follows, we generalize the bandit formulation to arbitrary SCMs, which we refer to as structural causal bandits.

Structural causal bandits. We now formalize the structural causal bandit (SCB) (Lee and Bareinboim, 2018, 2019) problem in which an agent interacts with an environment modeled by an SCM $\mathcal { M } = \langle \mathbf { U } , \mathbf { V } , \mathcal { F } , P ( \mathbf { U } ) \rangle$ that includes a reward variable $Y \in \mathbf { V }$ . In this setting, each action corresponds to an intervention on some set of variables $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ , with an assigned value $\mathbf { x } \in \Omega _ { \mathbf { X } }$ . That is:

$$
{ \mathcal { A } } = \{ d o ( \mathbf { x } ) : \mathbf { x } \in \Omega _ { \mathbf { X } } , \mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \} \} .\tag{41}
$$

We refer to each X as an intervention set (IS). We use the terms arm, action, and intervention interchangeably, depending on the context. The mean reward of an action is denoted by $\mu _ { \mathbf { x } } = \mathbb { E } [ Y \mid d o ( \mathbf { x } ) ]$ . Whenever a maximizing action is used, we assume that the maximum is attained and select one maximizer using a fixed tie-breaking rule. This is automatic for finite action spaces. For a fixed intervention set X, we denote a best action by:

$$
\mathbf { x } ^ { \ast } \in \arg \operatorname* { m a x } _ { \mathbf { x } \in \Omega _ { \mathbf { X } } } \mu _ { \mathbf { x } } .\tag{42}
$$

Similarly, we denote the optimal action in A by:

$$
\mathbf { x } ^ { \star } \in \underset { \mathbf { x } \in \Omega _ { \mathbf { X } } } { \arg \operatorname* { m a x } } ~ \mu _ { \mathbf { x } } .\tag{43}
$$

For convenience, we identify $d o ( \mathbf { x } )$ with the variable-indexed assignment x whenever the intervention target is clear. Assignments to diferent variable sets remain distinct actions, even when their numerical values coincide. The empty assignment denotes $d o ( \emptyset )$

Let $A _ { t } \in { \mathcal { A } }$ denote the random intervention selected at round t, using the assignment notation above. We define the suboptimality gap by $\Delta _ { \mathbf { x } } = \mu _ { \mathbf { x } ^ { \star } } - \mu _ { \mathbf { x } }$ and the cumulative regret as follows:

$$
R _ { T } ^ { A } = T \mu _ { { \bf x } ^ { \star } } - \sum _ { t = 1 } ^ { T } \mu _ { A _ { t } } = \sum _ { { \bf x } \in \mathcal { A } } \Delta _ { { \bf x } } N _ { T } ( { \bf x } ) ,\tag{44}
$$

where $N _ { T } ( \mathbf { x } )$ denotes the number of times $d o ( \mathbf { x } )$ is selected up to time $T .$

In this setting, the agent knows the causal diagram $\mathcal { G }$ and the reward variable $Y .$ The diagram $\mathcal { G }$ specifies the environment class ${ \mathcal { C } } ,$ consisting of SCMs conforming to $\mathcal { G }$ , while the structural functions $\mathcal { F }$ and the exogenous distribution $P ( \mathbf { U } )$ remain unknown. Thus, the agent has access to qualitative structural information but not to the quantitative parameterization that determines the reward distributions. Graphical characterizations range over compatible SCMs, with domains chosen as part of the model; a particular bandit instance fixes these domains.

Example 5 (Two manipulable variables). Consider an MAB environment induced by the following SCM $\mathcal { M }$

$$
\mathcal { M } = \left\{ \begin{array} { l l } { \mathbf { U } = \{ U _ { 1 } , U _ { 2 } \} } \\ { \mathbf { V } = \{ X _ { 1 } , X _ { 2 } , Y \} } \\ { \mathcal { F } = \left\{ \begin{array} { l l } { f _ { X _ { 1 } } ( u _ { 1 } ) = u _ { 1 } } \\ { f _ { X _ { 2 } } ( x _ { 1 } , u _ { 2 } ) = x _ { 1 } \oplus u _ { 2 } } \\ { f _ { Y } ( x _ { 2 } , u _ { 2 } ) = x _ { 2 } \oplus u _ { 2 } } \\ { P ( \mathbf { U } ) = \{ U \sim \mathrm { B e r n } ( 0 . 5 ) : \forall U \in \mathbf { U } \} } \end{array} \right. } \end{array} \right.\tag{45}
$$

where $\bigoplus$ denotes the exclusive-or operator. Assume that $U _ { 1 }$ and $U _ { 2 }$ are mutually independent. The causal diagram $\mathcal { G }$ associated with $\mathcal { M }$ is shown in Fig. 9c. Consider an agent deployed in M whose goal is to minimize expected cumulative regret:

$$
\mathbb { E } R _ { T } ^ { \mathcal { A } } = \sum _ { \mathbf { x } \in \mathcal { A } } \Delta _ { \mathbf { x } } \mathbb { E } N _ { T } ( \mathbf { x } )\tag{46}
$$

by intervening on the manipulable variables $\mathbf { V } \setminus \{ Y \} = \{ X _ { 1 } , X _ { 2 } \}$ . The collection of intervention sets is given by:

$$
\{ \emptyset , \{ X _ { 1 } \} , \{ X _ { 2 } \} , \{ X _ { 1 } , X _ { 2 } \} \} .\tag{47}
$$

This collection is precisely the power set of the manipulable variables, and therefore contains $2 ^ { | \{ X _ { 1 } , X _ { 2 } \} | } = 4$ possible intervention sets. This setting can be translated into a traditional MAB instance in which each action corresponds to assigning specific values to a subset of $\{ X _ { 1 } , X _ { 2 } \}$ under intervention. In this example, this yields $9 ^ { 1 2 }$ actions:

$$
\mathcal { A } = \{ \mathbf { x } \in \{ 0 , 1 \} ^ { | \mathbf { X } | } , \mathbf { X } \in \{ \varnothing , \{ X _ { 1 } \} , \{ X _ { 2 } \} , \{ X _ { 1 } , X _ { 2 } \} \} \} .\tag{48}
$$

We provide two structural properties that arise in the SCB setting: equivalence and partial orders. These properties capture dependencies among diferent actions and can be understood through do-calculus (Pearl, 1995) and related tools.

Theorem 2 (do-calculus (Pearl, 1995)). Let G be a causal diagram of M, with endogenous variables V. For any disjoint sets X, Y, W, $\mathbf { Z } \subseteq \mathbf { V } .$ , the following rules are valid whenever the displayed conditional distributions are defined.

$$
\begin{array} { r l r } & { R u l e ~ \boldsymbol { \mathrm { \boldsymbol { \mathit { 1 } } } } . ~ P ( \mathbf { \boldsymbol { y } } \mid d o ( \mathbf { w } ) , \mathbf { x } , \mathbf { z } ) = P ( \mathbf { \boldsymbol { y } } \mid d o ( \mathbf { w } ) , \mathbf { z } ) } & { \qquad } & { i f \left( \mathbf { \mathbf { X } } \perp \mathbf { Y } \mid \mathbf { W } \cup \mathbf { Z } \right) _ { \mathcal { G } _ { \mathbf { W } } ^ { \mathbf { \mathbf { \boldsymbol { \Sigma } } } } } , } \\ & { R u l e ~ \boldsymbol { \mathrm { \boldsymbol { 2 } } } . ~ P ( \mathbf { \boldsymbol { y } } \mid d o ( \mathbf { w } , \mathbf { x } ) , \mathbf { z } ) = P ( \mathbf { \boldsymbol { y } } \mid d o ( \mathbf { w } ) , \mathbf { x } , \mathbf { z } ) } & { \qquad } & { i f \left( \mathbf { X } \perp \mathbf { Y } \mid \mathbf { W } \cup \mathbf { Z } \right) _ { \mathcal { G } _ { \mathbf { W } \mathbf { X } } ^ { \mathbf { \boldsymbol { \Sigma } } } } , } \\ & { R u l e ~ \boldsymbol { \mathrm { \boldsymbol { 3 } } } . ~ P ( \mathbf { \boldsymbol { y } } \mid d o ( \mathbf { w } , \mathbf { x } ) , \mathbf { z } ) = P ( \mathbf { \boldsymbol { y } } \mid d o ( \mathbf { w } ) , \mathbf { z } ) } & { \qquad } & { i f \left( \mathbf { X } \perp \mathbf { Y } \mid \mathbf { W } \cup \mathbf { Z } \right) _ { \mathcal { G } _ { \mathbf { W } } , \mathbf { X } ( \mathbf { Z } ) } } \\ & { w h e r e ~ \mathbf { X } ( \mathbf { Z } ) \triangleq \mathbf { X } \setminus \mathrm { A n } ( \mathbf { Z } ) _ { \mathcal { G } [ \mathbf { V } \backslash \mathbf { W } ] } . } \end{array}
$$

To provide intuition, let us consider the following example, where the action space can be reduced using graphical information derived from these properties. The first reduction corresponds to removing intervention sets that are equivalent to some proper subset, while the second corresponds to removing intervention sets that are never strictly better than some other intervention set.

Example 6 (Continued). The main question is whether one can do better by exploiting the causal invariance implied by M and represented by G.

Property 1. First, using Rule 3 of do-calculus, we have:

$$
P ( y \mid d o ( x _ { 1 } , x _ { 2 } ) ) = P ( y \mid d o ( x _ { 2 } ) ) ,\tag{49}
$$

since $( X _ { 1 } \perp \perp Y ) _ { \mathcal { G } _ { \overline { { X _ { 2 } } } } }$ . In words, intervening on $X _ { 1 }$ has no additional efect on $Y$ once $X _ { 2 }$ is intervened upon. Therefore, the corresponding expected rewards are equal:

$$
\mu _ { x _ { 1 } , x _ { 2 } } = \mu _ { x _ { 2 } } ,\tag{50}
$$

for any $x _ { 1 } \in \Omega _ { X _ { 1 } }$ and $x _ { 2 } \in \Omega _ { X _ { 2 } }$ . Consequently, actions of the form $d o ( x _ { 1 } , x _ { 2 } )$ are redundant, and it sufices to consider only the 5 actions induced by the remaining three intervention sets:

$$
\mathcal { A } = \{ \mathbf { x } \in \{ 0 , 1 \} ^ { | \mathbf { X } | } , \mathbf { X } \in \{ \varnothing , \{ X _ { 1 } \} , \{ X _ { 2 } \} \} \} .\tag{51}
$$

Property 2. Next, the causal structure implies that the observational reward distribution $P ( y ) = P ( y \mid d o ( \emptyset ) )$ can be written as a convex combination of $\{ P _ { x _ { 1 } } ( y ) \} _ { x _ { 1 } \in \Omega _ { X _ { 1 } } }$ :

$$
P ( y ) = \sum _ { x _ { 1 } } P ( y \mid x _ { 1 } ) P ( x _ { 1 } )\tag{52}
$$

$$
= \sum _ { x _ { 1 } } { P _ { x _ { 1 } } ( y ) P ( x _ { 1 } ) } ,\tag{53}
$$

by Rule 2 of do-calculus with $( X _ { 1 } \perp \perp Y ) _ { \mathcal { G } _ { X _ { 1 } } }$ . Equivalently, $\begin{array} { r } { \mu _ { \emptyset } = \sum _ { x _ { 1 } } \mu _ { x _ { 1 } } P ( x _ { 1 } ) } \end{array}$ . Thus, we can derive:

$$
\mu _ { \emptyset } = \sum _ { x _ { 1 } } \mu _ { x _ { 1 } } P ( x _ { 1 } )\tag{54}
$$

$$
\leq \sum _ { x _ { 1 } } \mu _ { x _ { 1 } ^ { * } } P ( x _ { 1 } )\tag{55}
$$

$$
\boldsymbol { \mathbf { \mu } } = \mu _ { x _ { 1 } ^ { * } } .\tag{56}
$$

Hence, $d o ( \emptyset )$ is never better than $d o ( x _ { 1 } ^ { * } )$ and can therefore be excluded from consideration. Thus, it sufices to consider only the 4 actions induced by the remaining two intervention sets:

$$
\mathcal { A } = \{ \mathbf { x } \in \{ 0 , 1 \} ^ { | \mathbf { X } | } , \mathbf { X } \in \{ \{ X _ { 1 } \} , \{ X _ { 2 } \} \} \} .\tag{57}
$$

This conclusion holds for any SCM consistent with ${ \mathcal { G } } ,$ showing how causal structure can reduce the action space while retaining at least one optimal action. Eliminated actions may tie with retained actions, but cannot achieve a higher reward than the best retained action.

We now formalize these two reductions, beginning with equivalence among actions.

## 3.2.1 Property 1: E<sub>q</sub>uivalence among Actions

We begin by recalling Rule 3 of do-calculus, which gives conditions under which an intervention has no efect on the outcome variable. Since actions correspond to interventions (including the null intervention), we consider whether:

$$
P ( y \mid d o ( \mathbf { x } , \mathbf { w } ) ) = P ( y \mid d o ( \mathbf { x } ) )\tag{58}
$$

holds. By Rule 3 of do-calculus, this equality is valid if $( Y \perp \mathbf { W } \mid \mathbf { X } ) _ { \mathcal { G } _ { \overline { { \mathbf { X } \cup \mathbf { W } } } } } .$ , which implies that $\mu _ { \mathbf { x } \cup \mathbf { w } } = \mu _ { \mathbf { x } }$ . When this condition holds, the actions $d o ( \mathbf { x } , \mathbf { w } )$ and $d o ( \mathbf { x } )$ are equivalent in terms of their induced reward distributions. Consequently, it sufices to play one action from each equivalence class.

Minimal intervention sets. Ideally, one would like to restrict attention to a smaller subset of actions, provided that it still contains an optimal action. Motivated by this observation, we introduce the notion of a minimal intervention set (MIS).

Definition 12 (Minimal intervention set (Lee and Bareinboim, 2018)). A set $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is an MIS relative to $\langle \mathcal { G } , Y \rangle$ if there is no proper subset $\mathbf { Z } \subsetneq \mathbf { X }$ such that, for every SCM conforming to $\mathcal { G }$ and every assignment $\mathbf { x } \in \Omega _ { \mathbf { X } } , \mu _ { \mathbf { x } } = \mu _ { \mathbf { x } \cap \mathbf { Z } } ^ { 1 3 }$

We denote by $\mathbb { M } _ { \mathcal { G } , Y }$ the collection of MISs with respect to $\langle \mathcal { G } , Y \rangle$ . In words, an MIS is a set for which no proper subset is equivalent in terms of the induced reward. Whether a set is an MIS can be determined through a simple procedure based on ancestral relationships.

Proposition 1 (Proposition 1 in Lee and Bareinboim (2018)). $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is an MIS with respect to $\langle \mathcal { G } , Y \rangle$ if and only if $\mathbf { X } \subseteq \mathsf { A n } ( Y ) _ { \mathcal { G } _ { \overline { { \mathbf { X } } } } }$

Intuitively, this characterization states that, for X to be an MIS, each variable $X \in \mathbf { X }$ must have a directed path to Y that does not pass through any other variables in $\mathbf { X } \setminus \{ X \}$ . Such a path is called proper with respect to X: only its starting vertex belongs to X. Note that ∅ and every singleton {X} with $X \in \mathsf { A n } ( Y ) _ { \mathcal { G } } \setminus \{ Y \}$ are MISs.

Example 7 (Continued). Let us consider the causal diagram $\mathcal { G }$ shown in Fig. 9c. Since ∅, $\{ X _ { 1 } \}$ , and $\{ X _ { 2 } \}$ are MISs in this graph, it remains only to test $\{ X _ { 1 } , X _ { 2 } \}$ :

$$
\{ X _ { 1 } , X _ { 2 } \} \not \subseteq \mathtt { A n } ( Y ) _ { { \mathcal G } _ { \overline { { X _ { 1 } , X _ { 2 } } } } } = \{ X _ { 2 } , Y \} .\tag{59}
$$

Therefore, it follows that:

$$
{ \mathbb M } _ { { \mathcal G } , Y } = \{ \emptyset , \{ X _ { 1 } \} , \{ X _ { 2 } \} \} ,\tag{60}
$$

and the corresponding action set A is given in Eq. (51).

## 3.2.2 Property 2: Partial Orders among MISs

Possibly-optimal minimal intervention sets. We now explore partial orders among MISs. Given a causal diagram G, it is possible that interventions on certain variables always perform at least as well as interventions on others, regardless of the parameterization of the underlying model. Formally, there may exist two sets of variables W and X such that:

$$
\mu _ { \mathbf { w } ^ { * } } \leq \mu _ { \mathbf { x } ^ { * } }\tag{61}
$$

for every SCM consistent with G. In this case, it is unnecessary to consider actions over $\Omega _ { \mathbf { W } }$ Motivated by this observation, we define the notion of possibly-optimal MIS, which captures this partial order among MISs.

Definition 13 (POMIS (Lee and Bareinboim, 2018)). Let $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ be an MIS with respect to $\langle \mathcal { G } , Y \rangle$ . If there exists an SCM conforming to $\mathcal { G }$ such that $\mu _ { \mathbf { x } ^ { * } } > \mu _ { \mathbf { w } ^ { * } }$ for all $\mathbf { W } \in \mathbb { M } _ { \mathcal { G } , Y } \setminus \{ \mathbf { X } \}$ where $\mathbb { M } _ { \mathcal { G } , Y }$ is the collection of MISs with respect to $\langle \mathcal { G } , Y \rangle$ , then X is a possibly-optimal minimal intervention set (POMIS) with respect to ⟨G, Y⟩.

We denote by $\mathbb { P } _ { \mathcal { G } , Y }$ the collection of POMISs with respect to $\langle \mathcal { G } , Y \rangle$ . Possible optimality means that the graph alone cannot rule out the intervention set; the best action in the actual SCM must still be learned.

Consider an MIS $\mathbf { W } \in \mathbb { M } _ { \mathcal { G } , Y }$ and a disjoint set of variables X. It follows that:

$$
\mu _ { \mathbf { w } } = \sum _ { \mathbf { x } } \mathbb { E } _ { \mathbf { w } } [ Y \mid \mathbf { x } ] P _ { \mathbf { w } } ( \mathbf { x } ) .\tag{62}
$$

Suppose that, by Rule 2 of do-calculus, conditioning on x can be exchanged with intervention. Then, we can derive the following for every fixed w:

$$
\mu _ { \mathbf { w } } = \sum _ { \mathbf { x } } \mathbb { E } _ { \mathbf { w } } [ Y \mid \mathbf { x } ] P _ { \mathbf { w } } ( \mathbf { x } )\tag{63}
$$

$$
= \sum _ { \mathbf { x } } \mu _ { \mathbf { x w } } P _ { \mathbf { w } } ( \mathbf { x } )
$$

$$
\mathbf { R u l e \ 2 }\tag{64}
$$

$$
\leq \sum _ { \mathbf { x } } \mu _ { \{ \mathbf { x w } \} * } P _ { \mathbf { w } } ( \mathbf { x } )\tag{65}
$$

$$
= \mu _ { \{ { \bf x w } \} } \ast .\tag{66}
$$

Let $\mathbf { Z } \subseteq \mathbf { X } \cup \mathbf { W }$ be an MIS equivalent to $\mathbf { X } \cup \mathbf { W }$ . Then, we have:

$$
\mu _ { \mathbf { w } ^ { * } } \leq \mu _ { \mathbf { z } ^ { * } } .\tag{67}
$$

To identify such $\mathbf { X } ,$ , we examine the condition for Rule 2 of do-calculus $( Y \perp \perp \mathbf { X } \mid \mathbf { W } ) _ { \mathcal { G } _ { \overline { { \mathbf { W } } } \underline { { X } } } } ,$ which is equivalent to:

$$
( Y \perp \perp \mathbf { X } ) _ { \mathcal { G } _ { \mathbf { \underline { { X } } } } ^ { \prime } } ,\tag{68}
$$

where $\mathcal { G } ^ { \prime } = \mathcal { G } \setminus \mathbf { W }$ . Therefore, if every path from X to $Y$ in $\mathcal { G } _ { \underline { { \mathbf { X } } } } ^ { \prime }$ is blocked, then such an MIS Z satisfies:

$$
\mu _ { \mathbf { w } ^ { * } } \leq \mu _ { \{ \mathbf { x } , \mathbf { w } \} ^ { * } } = \mu _ { \mathbf { z } ^ { * } } .\tag{69}
$$

Building on this observation, we introduce two graphical concepts for characterizing POMISs. Let $\mathsf { c c } ( X ) _ { \mathcal G }$ be the c-component (Tian and Pearl, 2003) containing X in ${ \mathcal { G } } _ { : }$ , namely, the maximal set of vertices connected to X by bidirected paths, and $\begin{array} { r } { \mathsf { c c } ( \mathbf { X } ) _ { \mathcal { G } } = \bigcup _ { X \in \mathbf { X } } \mathsf { c c } ( X ) _ { \mathcal { G } } } \end{array}$

Definition 14 (Unobserved confounders’ territory (Lee and Bareinboim, 2018)). Given information $\langle \mathcal { G } , Y \rangle$ , let $\mathcal { H } = \mathcal { G } [ \mathtt { A n } ( Y ) _ { \mathcal { G } } ]$ . A set of variables $\mathbf { T } \subseteq \mathbf { V } ( { \mathcal { H } } )$ containing Y is called a UC-territory on $\mathcal { G }$ with respect to Y if $\mathsf { D e } ( \mathbf { T } ) _ { \mathcal { H } } = \mathbf { T }$ and ${ \mathsf { c c } } ( \mathbf { T } ) _ { \mathcal { H } } = \mathbf { T }$ . A UC-territory T is said to be minimal if no $\mathbf { T } ^ { \prime } \subsetneq \mathbf { T }$ is a UC-territory.

Definition 15 (Interventional border (Lee and Bareinboim, 2018)). Let T be a minimal UC-territory on the causal diagram $\mathcal { G }$ with respect to $Y$ . Then $\mathbf { W } = \mathsf { P a } ( \mathbf { T } ) _ { \mathcal { G } } \setminus \mathbf { T }$ is called an interventional border (IB) for $\mathcal { G }$ with respect to $Y$

The minimal unobserved confounders’ territory (MUCT) is the smallest set containing Y that is closed under descendants and c-components within $\mathcal { G } [ \mathtt { A n } ( Y ) _ { \mathcal { G } } ]$

To compute it, start from $\{ Y \}$ in $\mathcal { H } = \mathcal { G } [ \mathtt { A n } ( Y ) _ { \mathcal { G } } ]$ and repeatedly add the c-components and descendants of the current set in H until the set stops changing. Every UC-territory contains each stage of this construction, so its limit is the unique MUCT. Its closure properties support the use of Rule 2 to exchange conditioning on its border with intervention.

We denote by ${ \mathsf { M U C T } } ( { \mathcal { G } } , Y )$ and $| \mathsf { B } ( \mathcal { G } , Y )$ the MUCT and IB with respect to $\langle \mathcal { G } , Y \rangle$ , respectively. The following theorem provides a complete characterization of POMISs.

Theorem 3 (Theorem 6 in Lee and Bareinboim (2018)). $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is a POMIS with respect to ⟨G, Y ⟩ if and only $i f 1 8 ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \mathbf { X }$

Example 8 (Continued). Again, consider the causal diagram $\mathcal { G }$ shown in Fig. 9c. We now test whether each MIS in ${ \mathbb M } _ { { \mathcal G } , Y } = \{ \emptyset , \{ X _ { 1 } \} , \{ X _ { 2 } \} \}$ is a POMIS as follows:

$$
\mathrm { F o r } \ \mathbf { X } = \varnothing , \mathsf { l B } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \{ X _ { 1 } \} \ne \mathbf { X } ;\tag{70}
$$

![](images/2c12afa875144683dac991618ad56a442efd2d49ff50e1ffdd6aedebb34c884d.jpg)  
(a) G

![](images/c9bf4fd85235dcd94e98c4afe814e64a57aebc23b37f5dd76737c95f9086171f.jpg)  
(b) Under intervention on B

![](images/571f22d1a330e380d408e9565e139f6891ee67ecd5385503f1bc8868f65c4777.jpg)  
(c) Under intervention on E

![](images/22c47e189c14647d09feb4a55e5afe7ef4e2f6602dd37911e0043202368f8e21.jpg)  
(d) Under intervention on F

Figure 10: MUCTs and IBs. Panels (b)–(d) show the ancestral subgraphs of the manipulated graphs. Light blue regions mark MUCTs, green fill marks IB variables, and thick outlines mark intervention targets.  
![](images/3488f999a250792c1d3373c7f3c99e40a402ac2a9c30335dc52ea801205a4dc4.jpg)  
(a) Inclusion

![](images/24dcfb61e6ead96e3825dc2a2925a09677e054965366d40489e22e4e69a96e7d.jpg)  
(b) Regret  
Figure 11: (a) Inclusion of POMISs in MISs and of MISs in intervention sets. (b) Schematic comparison of expected cumulative regret for action spaces defined by all intervention sets (IS), MISs, and POMISs. Solid and dashed curves represent distinct policies, e.g., UCB and Thompson sampling.

$$
\mathrm { F o r } \ \mathbf { X } = \{ X _ { 1 } \} , \mathsf { l B } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \{ X _ { 1 } \} = \mathbf { X } ;\tag{71}
$$

$$
\mathrm { F o r } \ \mathbf { X } = \{ X _ { 2 } \} , \mathsf { l B } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \{ X _ { 2 } \} = \mathbf { X } .\tag{72}
$$

Therefore, the collection of POMISs is:

$$
\mathbb { P } _ { \mathcal { G } , Y } = \{ \{ X _ { 1 } \} , \{ X _ { 2 } \} \} ,\tag{73}
$$

and the corresponding action set A is given in Eq. (57).

The following example applies the MUCT–IB criterion to several intervention sets in one confounded system. Consider G in Fig. 10a. For each intervention set X, we form $\mathcal { G } _ { \overline { { \mathbf { X } } } }$ and

compute its MUCT in the subgraph induced by the ancestors of Y in that manipulated graph.

In Fig. 10b, A is no longer an ancestor of Y . Starting from {Y }, the bidirected edge $C  Y$ adds C, the directed edge $C  E$ adds E, and $E  F$ then adds F. Consequently:

$$
\mathrm { F o r } \ \mathbf { X } = \{ B \} , \mathsf { M U C T } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \{ C , E , F , Y \}\tag{74}
$$

$$
\Rightarrow \mathsf { I B } ( \mathcal G _ { \overline { { \mathbf { X } } } } , Y ) = \{ D \} \neq \mathbf { X } .\tag{75}
$$

The border gives the comparison $\mu _ { b ^ { * } } \leq \mu _ { d ^ { * } }$ , i.e., intervening on D can achieve at least the best reward available by intervening on B. Indeed, Rule 2 gives:

$$
\mathbb { E } [ Y \mid d o ( b ) , d ] = \mathbb { E } [ Y \mid d o ( b , d ) ] ,\tag{76}
$$

and Rule 3 removes the intervention on B once D is intervened upon. Therefore, $\mu _ { b } =$ $\begin{array} { r } { \sum _ { d } \mu _ { d } P ( d \mid d o ( b ) ) \le \mu _ { d ^ { * } } } \end{array}$

Only E, F, and Y remain in the ancestral subgraph in Fig. 10c. Thus,

$$
\mathrm { F o r } ~ \mathbf { X } = \{ E \} , \mathsf { M U C T } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \{ Y \}
$$

$$
\Rightarrow \mathsf { I B } ( \mathcal G _ { \overline { { \mathbf { X } } } } , Y ) = \{ E , F \} \neq \mathbf { X } .\tag{77}
$$

(78)

Here the comparison is $\mu _ { e ^ { * } } \leq \mu _ { \{ e , f \} ^ { * } }$ , so an optimal joint intervention on E and F performs at least as well as an optimal intervention on E alone.

The territory in Fig. 10d contains all vertices except F:

$$
\mathrm { F o r } ~ \mathbf { X } = \{ F \} , \mathsf { M U C T } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \{ A , B , C , D , E , Y \}\tag{79}
$$

$$
\Rightarrow { \sf I B } ( \mathcal G _ { \overline { { \bf X } } } , Y ) = \{ F \} = { \bf X } .\tag{80}
$$

By Thm. $3 , \{ F \}$ is a POMIS, whereas {B} and {E} are not. This illustrates how the MUCT and IB support action space pruning before the agent estimates reward distributions: they identify intervention sets that can be excluded and those that must remain candidates for optimality.

Equipped with Thm. 3, one can systematically reduce the action space using the information ⟨G, Y ⟩. Restricting attention to POMISs preserves the optimal achievable reward while reducing the candidate action space. The schematic comparison in Fig. 11 illustrates the potential reduction in expected cumulative regret from restricting the action space to MISs or POMISs.

## 4 Structural Causal Bandits under Markov Equivalence

The SCB formulation in the preceding section assumes that the agent knows the causal diagram. Under the causal Markov and faithfulness assumptions, observational conditional independences generally identify only a Markov equivalence class. Causal suficiency, the absence of unobserved common causes of the observed variables, restricts the graph class but does not, by itself, resolve all orientations (Verma and Pearl, 1990; Spirtes et al., 2000; Chickering, 2002; Tsamardinos et al., 2006). Additional assumptions on functional forms, noise distributions, or variation across environments may provide further identification (Perry et al., 2022; Peters et al., 2016; Ghassami et al., 2017; Heinze-Deml et al., 2018; Huang et al., 2020; Ghassami et al., 2018; Zeng et al., 2021). In the presence of latent variables and without selection bias, we represent the observational Markov equivalence class by a PAG.

Under these assumptions, observational conditional-independence information identifies a PAG, which may encode a super-exponential number of maximal ancestral graphs (MAGs). Each such MAG, in turn, represents an infinite number of causal diagrams over supersets of the observed variables. Therefore, considering all causal diagrams consistent with a PAG is computationally prohibitive. Directly identifying conditions for MIS and POMIS at the level of ancestral graphs would circumvent this dificulty. Our key contributions in this section are as follows:

• We generalize MIS and develop graphical criteria for ancestral graphs, enabling an agent to identify and exclude variables that have no efect on the reward.

• We extend POMIS to ancestral graphs and establish a necessary and suficient graphical condition for MAGs.

• We present a search algorithm that is sound and complete for POMIS membership in at least one represented MAG.

Relation to the published version. This section revises Park et al. (2025). The counterexample in Sec. 4.1.2 disproves the suficiency of Theorem 2 of Park et al. (2025). For Proposition 6 of Park et al. (2025), only the implication from PAG-level POMIS membership to membership in some represented MAG is retained (Prop. 8). The search associated with Theorem 4 of Park et al. (2025) instead characterizes the union of the POMIS collections of the represented MAGs (Sec. 4.2.2). Definition 6 of Park et al. (2025) is corrected by removing X before computing possible ancestors, excluding paths that pass through intervened variables (Def. 20). Our revision of Algorithm 1 of Park et al. (2025) uses an MIS precheck, omits the preliminary orientation step, and tests proper paths and the corrected IB (Alg. 3 and prop. 10). Accordingly, Sec. 4.3 reports corrected action spaces and new experiments with horizon 50,000 and 100 repetitions; these replace the published arm counts and regret results.

## 4.1 Generalizing Minimal Intervention Sets

In this section, we generalize the notion of a minimal intervention set (MIS) to ancestral graphs, so that it applies not only to causal diagrams but also to MAGs and PAGs. We first provide complete graphical conditions for MIS in MAGs and PAGs. We then show that MIS membership in a PAG need not have a common witnessing MAG. To address this issue, we introduce definitely minimal intervention sets (DMISs), which require a single compatible SCM to witness the absence of equivalence to every proper subset. Here, definitely refers to this common witness, not to minimality in every MAG represented by the PAG.

## 4.1.1 Minimal Intervention Sets for Ancestral Graphs

We use D to denote either a causal diagram or an ancestral graph (MAG or PAG) over V. The existence arguments below quantify over the graphical model class, without fixing an observational distribution or a parametric family.

Definition 16 (MIS). Given $\langle { \mathcal { D } } , Y \rangle$ , a set $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is an MIS ${ \mathrm { i f } } ,$ for every $\mathbf { Z } \subsetneq \mathbf { X }$ there exist an SCM S conforming to D and an assignment $\mathbf { x } \in \Omega \mathbf { x }$ such that $\mu _ { \mathbf { x } } ^ { S } \neq \mu _ { \mathbf { x } \cap \mathbf { Z } } ^ { S } { ^ { 1 4 } }$

For any causal diagram, we can explicitly construct a compatible binary SCM ${ \boldsymbol { S } } ^ { * }$ to provide the witnesses used below (Lemma 35). Each variable equals one when its private uniform noise falls below a threshold that increases with its parents’ values, while shared latent variables represent the bidirected edges. Setting a previously unintervened variable to one strictly increases its mean, and positive parental efects propagate this increase along directed paths. Along a proper directed path to Y, no other intervention interrupts this propagation, so the reward mean strictly increases.

Minimal intervention sets for MAGs. We now turn to the graphical characterization of MIS for MAGs. As in the case of causal diagrams, this characterization relies on the explicit ancestral relations among variables in MAGs and on Rule 3 of do-calculus for MAGs (Zhang, 2008b):

$$
\begin{array} { r l } & { \mathrm { { \bf ~ R u l e ~ 3 . } } \ P ( { \bf \underline { { y } } } \mid d o ( { \bf w } , { \bf x } ) , { \bf z } ) = P ( { \bf \underline { { y } } } \mid d o ( { \bf w } ) , { \bf z } ) \qquad \qquad \quad \mathrm { i f ~ ( { \bf ~ X } ~ \bot ~ { \bf ~ Y } ~ \mid ~ { \bf ~ W } \cup { \bf ~ Z } ) _ { \mathcal { M } _ { \overline { { { \bf w } } } , { \bf { x } } ( { \bf \ Z } ) } } ~ } } \\ & { \quad \quad \quad \quad \quad \mathrm { { w h e r e ~ } } \ \mathrm { { \bf ~ X } } ( { \bf Z } ) \triangleq { \bf { X } } \ \backslash \ \mathrm { { \bf ~ A n } } ( { \bf Z } ) _ { \mathcal { M } [ \mathrm { { \bf ~ V } } \backslash { \bf W } ] } . \qquad } \end{array}
$$

Theorem 4 (Characterization of MIS for MAGs). Given information $\langle { \mathcal { M } } , Y \rangle , \mathbf { X } \subseteq \mathbf { V } \backslash \{ Y \}$ is an MIS relative to $\langle { \mathcal { M } } , Y \rangle$ if and only if $\mathbf { X } \subseteq \mathrm { A n } ( Y ) _ { \mathcal { M } _ { \overline { { \mathbf { X } } } } }$ holds.

Proof of Thm. $\it 4 .$ (If) Let G have exactly the directed and bidirected edges of M, interpreted as a causal diagram. Its corresponding MAG is M, since replacing each bidirected edge by a separate latent common parent preserves its separation model, and maximality and ancestrality recover the same adjacencies and marks. The two graphs have the same directed paths after deleting incoming edges to X.

If $\mathbf { X } \subseteq \mathsf { A n } ( Y ) _ { \mathcal { M } _ { \overline { { \mathbf { X } } } } } .$ , every $X \in \mathbf { X }$ has a proper directed path to $Y .$ . Fix the SCM ${ \boldsymbol { S } } ^ { * }$ provided by Lemma 35, which conforms to $\mathcal { G }$ and hence to M. For any proper subset $\mathbf { Z } \subsetneq \mathbf { X }$ , choose $W \in { \bf X } \backslash { \bf Z }$ . Because W has a proper directed path to $Y$ with respect to $\mathbf { X } ,$ , the lemma gives $\mu _ { \mathbf { X } = \mathbf { 1 } } ^ { S ^ { * } } > \mu _ { \mathbf { Z } = \mathbf { 1 } } ^ { S ^ { * } }$ where the subscripts denote interventions that set all variables in the indicated set to one. This proves MIS membership. If $\mathbf { X } = \varnothing$ , there is no proper subset to check.

(Only if) Suppose $\mathbf { Z } : = \mathbf { X } \backslash \mathtt { A n } ( Y ) _ { \mathcal { M } _ { \overline { { \mathbf { X } } } } }$ is nonempty, and let ${ \bf W } = { \bf X } \backslash { \bf Z }$ . We show:

$$
( Y \perp \perp \mathbf { Z } \mid \mathbf { W } ) _ { \mathcal { M } _ { \overline { { \mathbf { X } } } } } .\tag{81}
$$

Otherwise, an m-connecting path from $Z \in \mathbf { Z }$ to $Y$ would start with an outgoing edge, since all arrowheads into X have been removed. It could contain no collider: every conditioned vertex in W has no incoming edge, so no collider can be its ancestor or coincide with it. Such a path must therefore be directed towards Y and avoid W. This contradicts the choice of Z.

Rule 3 for MAGs now gives $P ( y \mid d o ( \mathbf { x } ) ) = P ( y \mid d o ( \mathbf { x } \cap \mathbf { W } ) )$ for every assignment x and every SCM conforming to M. Since W ⊊ X, X is not an MIS. □

Proposition 2. $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is an MIS relative to $\langle { \mathcal { M } } , Y \rangle$ if and only if there exists a causal diagram $\mathcal { G } \in [ M ]$ such that X is an MIS relative to ⟨G, Y ⟩.

Proof of Prop. 2. (If) Suppose that X is an MIS for a causal diagram $\mathcal { G } \in [ M ]$ . For every $\mathbf { Z } \subsetneq \mathbf { X }$ , there exist an SCM conforming to G and an assignment x for which $\mu _ { \mathbf { x } } \neq \mu _ { \mathbf { x } \cap \mathbf { Z } }$ This SCM also conforms to M; thus, the same witnesses establish that X is an MIS for M. (Only if) By Thm. 4, $\mathbf { X } \subseteq \mathrm { A n } ( Y ) _ { \mathcal { M } _ { \overline { { \mathbf { X } } } } }$ . Take the causal diagram G with exactly the directed and bidirected edges of M, as in that theorem’s proof. Its corresponding MAG is M, and the proper directed paths from X to Y are unchanged. Thus, Prop. 1 implies that X is an MIS for G. □

This proposition guarantees that if X is an MIS relative to ⟨M, Y ⟩ for a given MAG M, then there exists a causal diagram G conforming to M such that X is an MIS relative to $\langle \mathcal { G } , Y \rangle$

Remark 1. Even if X is an MIS with respect to $\langle { \mathcal { M } } , Y \rangle$ , there is no guarantee that it is also an MIS with respect to ⟨G, Y⟩ for every causal diagram $\mathcal { G }$ consistent with M.

Minimal intervention sets for PAGs. We proceed to the characterization of MIS for PAGs. Rule 3 for PAGs (Jaber et al., 2022) can establish interventional equivalence throughout the represented model class:

$$
\begin{array} { r l r } & { } & { \mathrm { { \bf ~ R u l e ~ 3 . } ~ } P ( { \bf y } \mid d o ( { \bf w } , { \bf x } ) , { \bf z } ) = P ( { \bf y } \mid d o ( { \bf w } ) , { \bf z } ) \quad } \\ & { } & { \mathrm { w h e r e ~ } { \bf X } ( { \bf Z } ) \triangleq { \bf X } \setminus \mathrm { P o s s A n } ( { \bf Z } ) _ { \mathcal P [ { \bf V \backslash W } ] } . } \end{array}
$$

$$
\mathbf { i f } \ ( \mathbf { X } \ \bot \mathbf { Y } \mid \mathbf { W } \cup \mathbf { Z } ) _ { \mathcal { P } _ { \overline { { \mathbf { W } } } , \overline { { \mathbf { X } ( \mathbf { Z } ) } } } }
$$

For example, in the PAG $X { \circ { \mathrm { - } } \circ Z \circ { \mathrm { - } } \circ Y }$ , the only path from X to Y has Z as a definite non-collider<sup>15</sup>. Thus, $( X \perp Y \mid Z ) _ { \mathcal { P } _ { \overline { { \{ X , Z \} } } } } .$ , and Rule 3 gives $P ( y \mid d o ( x , z ) ) = P ( y \mid d o ( z ) )$ This excludes {X, Z} from MIS membership. To establish a necessary and suficient characterization, we also need compatible SCMs witnessing the inequalities required for MIS membership.

For this purpose, we extend the definition of a proper path in Sec. 3 to possibly directed paths from $X \in \mathbf { X }$ to Y. Each such path is not disturbed by other intervening variables, and thus aligns with the characterizations of MIS for causal diagrams and MAGs. Let P be a PAG over the set of variables V.

15. Let ⟨X, Z, Y⟩ be any consecutive triple. We say that Z is a definite collider if both edges have arrowheads at Z. If (1) one of the edges is out of Z; or (2) both edges have a circle mark at $Z \ ( { \mathrm { i . e . , } } \ X * { \mathrm { - } } \circ Z \circ { \mathrm { - } } * Y )$ and there is no edge between X and Y, then we say that Z is a definite non-collider. A path is said to have a definite status if every non-endpoint node along it is either a definite collider or a definite non-collider. In a PAG, m-separation is tested over definite-status paths: a path is m-connecting given Z if every non-collider is outside Z and every collider is an ancestor of Z. If no such path connects the queried sets, they are m-separated.

Algorithm 2: Enumerating minimal intervention sets by recursive extension.   
1 function MISs(D, Y )   
2 Set $\mathbf { A } = \mathsf { P o s s A n } ( Y ) _ { \mathcal { D } }$ and $\mathcal { H } = \mathcal { D } [ \mathbf { A } ]$   
3 Order $\mathbf { A } \setminus \{ Y \}$ as ℓ from sinks to sources as described in the text.   
4 return subMISs(H, Y, ∅, ℓ)   
5 function $\mathsf { s u b M } | \mathsf { S s } ( \mathcal { H } , Y , \mathbf { X } , \ell = [ W _ { 1 } , \cdot \cdot \cdot , W _ { r } ] )$   
6 Set $\mathbf { A } = \mathsf { P o s s A n } ( Y ) _ { \mathcal { H } \backslash \mathbf { X } } .$   
// Check every selected variable after each addition.   
7 if $\exists X \in \mathbf { X }$ such that PossCh $( X ) \varkappa \cap \mathbf { A } = \varnothing$ then return ∅   
8 Set $\mathbb { M } = \{ \mathbf { X } \}$   
9 for $i = 1 , \cdots , r$ do   
10 if $W _ { i } \in { \bf A }$ then   
11 Update $\mathbb { M } = \mathbb { M } \cup \mathsf { s u b } \mathsf { M } | \mathsf { S s } ( \mathcal { H } , Y , \mathbf { X } \cup \{ W _ { i } \} , [ W _ { i + 1 } , \cdot \cdot \cdot , W _ { r } ] )$   
12 return M

Proposition 3 (Graphical characterization of MIS for PAGs). $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is an MIS relative to $\langle \mathcal { P } , Y \rangle$ if and only if for every variable $X \in \mathbf { X }$ , there exists a proper possibly directed path from X to Y with respect to X in P.

Proof of Prop. 3. (If) Fix an arbitrary $\mathbf { Z } \subsetneq \mathbf { X }$ and choose $W \in { \bf X } \setminus { \bf Z }$ . By hypothesis and Lemma 15, there is an uncovered proper possibly directed path p from $W$ to Y relative to X. Taking a subsequence preserves properness.

Replace every ◦→ edge in P by a directed edge and orient its circle components into DAGs without unshielded colliders, orienting every circle edge incident to W out of W. Such a choice is possible by the construction in the proof of Lemma 23; Lemma 14 ensures that the resulting MAG M belongs to [P]. The first edge of p now points out of W. Each consecutive triple of p is unshielded and cannot become a collider, so the subsequent edges must also point towards Y . Thus, p is a proper directed path in M.

Use the causal diagram with the same directed and bidirected edges as M. By Lemma 35, it admits an SCM ${ S ^ { * } }$ with $\mu _ { \mathbf { X } = \mathbf { 1 } } ^ { S ^ { \ast } } > \mu _ { \mathbf { Z } = \mathbf { 1 } } ^ { S ^ { \ast } }$ . This SCM conforms to P. Since Z was arbitrary, X is an MIS. The witnessing MAG and SCM may depend on Z, as permitted by Def. 16. The case $\mathbf { X } = \varnothing$ is vacuous.

(Only if) Suppose that some $Z \in \mathbf { X }$ has no proper possibly directed path to Y . Any proper directed path in a represented MAG would appear as a proper possibly directed path in P. Hence no $\mathcal { M } \in [ \mathcal { P } ]$ contains such a directed path from Z. The separation argument in the only-if part of Thm. 4, applied to this vertex, gives $P ( y \mid d o ( \mathbf { x } ) ) = P ( y \mid d o ( \mathbf { x } \cap ( \mathbf { X } \setminus \{ Z \} ) ) )$ for every assignment and every SCM conforming to any such MAG. The equality therefore holds throughout the model class of P, so X is not an MIS. □

Enumerating minimal intervention sets. The characterizations in Thm. 4 and prop. 3 lead to MISs (Alg. 2), which returns the complete collection $\mathbb { M } _ { \mathcal { D } , Y }$ for a MAG or PAG D. The algorithm begins with the empty intervention set and adds one variable at a time. It stops extending a set as soon as one of its variables loses every proper possibly directed path to $Y$

First, let $\mathbf { A } = \mathsf { P o s s A n } ( Y ) _ { \mathcal { D } }$ and retain the induced graph $\mathcal { H } = \mathcal { D } [ \mathbf { A } ]$ . No MIS can contain a variable outside A. Moreover, every vertex on a possibly directed path to Y belongs to A, so this restriction preserves all paths needed by the MIS condition. The graph H is then kept fixed throughout the recursion.

The variables in $\mathbf { A } \setminus \{ Y \}$ are arranged in a fixed list $\ell :$

• For a MAG, we use a reverse topological order of its directed edges. That is, whenever $U  V$ , the vertex V precedes U.

• For a PAG, we order its circle components by reversing the partial topological order returned by Alg. 12.

For the PAG case, this is the order in which Alg. 12 extracts the components. Within each component, variables are ordered by increasing shortest-path distance to Y in the underlying undirected graph of $\mathcal { D } ;$ ties can be resolved arbitrarily.

For a PAG, processing circle components in this reverse partial topological order supports early pruning before recursion. To see this, suppose that every possibly directed path from an upstream candidate W to $Y$ passes through already selected downstream variables. Deleting those selected variables excludes W from the remaining possible ancestors of $Y ,$ , so the algorithm discards this extension before making a recursive call. Processing downstream components first therefore allows such upstream candidates to be eliminated as soon as they are considered.

The same order also preserves paths across diferent circle components. Each step between distinct components on a possibly directed path leads to a component appearing earlier in the order. A variable selected later from a diferent component therefore cannot lie on a proper possibly directed path from an earlier selected variable to $Y$ . Adding that later variable cannot block the earlier variable’s path. Thus, reverse partial topological order prevents this source of failed recursive calls between components. Within a single component, however, a later choice can still block an earlier variable’s path. Thus, the algorithm retains the test for all selected variables.

For a current intervention set X, subMISs recomputes $\mathbf { A } = \mathtt { P o s s A n } ( Y ) _ { \# \backslash \mathbf { X } }$ . These are the vertices that can still reach Y without passing through any variable in X. Write PossCh(X)<sub>H</sub> for the set of vertices adjacent to X whose incident edge has no arrowhead at $X$ . Each selected X must have a first step into A, giving the test:

$$
\mathrm { P o s s C h } ( X ) _ { \mathcal { H } } \cap { \bf A } \neq \emptyset \quad \mathrm { f o r ~ e v e r y ~ } X \in { \bf X } .\tag{82}
$$

Possible ancestors include Y itself, so a single possibly directed edge from X to Y can satisfy this test. For a M $\mathrm { A G } ,$ the first step and the remaining path must be directed. The test is applied to all selected variables after every addition. Although a new variable may have a path to ${ \cal Y } ,$ selecting it can interrupt a path used by an earlier variable. Keeping H fixed allows the algorithm to inspect the first step from every selected variable; only the graph used to compute A has those variables removed.

If the test fails, the call returns an empty collection. Otherwise, it records X and tries each remaining variable $W _ { i } \in { \bf A }$ . The recursive call receives $\mathbf { X } \cup \{ W _ { i } \}$ and only the variables after $W _ { i }$ in the list. In particular, the initial call records $\varnothing ,$ , because its test has no selected variable to check.

Proposition 4. $M I S s ( \mathcal { D } , Y )$ returns exactly $\mathbb { M } _ { \mathcal { D } , Y }$ . Every MIS is generated once, and the recursion terminates.

Proof of Prop. 4. We first justify the test $\mathtt { P o s s c h } ( X ) _ { \mathcal { H } } \cap \mathbf { A } \neq \emptyset$ for every $X \in \mathbf { X }$ in Eq. (82). If $X \in \mathbf { X }$ has a proper possibly directed path to Y, its first adjacent vertex V is outside X. The remaining path avoids all of X, so $V \in \mathbf { A }$ . Conversely, if $V \in \mathtt { P o s s C h } ( X ) _ { \mathcal { H } } \cap \mathbf { A }$ , there is a possibly directed path from V to Y in $\mathcal { H } \backslash \mathbf { X }$ . Prepending X gives a proper path: X cannot already occur on the remaining path, and no other selected variable occurs there. This also covers $V = Y$ , when the remaining path has zero length. Thus, the test is equivalent to requiring a proper path for every selected variable. By Thm. 4 and prop. 3, every recorded set is an MIS.

Next, suppose an intervention set X fails the test at a variable $X \in \mathbf { X }$ . Adding more intervention variables only removes additional vertices from the graph in which the remainder of its path must lie. It cannot create a path for X that avoids the enlarged intervention set. Therefore, no superset of X is an MIS, and stopping this branch cannot omit a valid output. Likewise, if an unselected variable $W$ is outside $\mathbf { A } = \mathtt { P o s s A n } ( Y ) _ { \# \backslash \mathbf { X } }$ , it has no path to $Y$ avoiding X and cannot be added in any valid extension.

To establish completeness, fix an MIS Z. All its variables belong to $\mathtt { P o s s A n } ( Y ) _ { \mathcal { D } } \setminus \{ Y \}$ . List them in the order inherited from ℓ. Every prefix X of this list is an MIS: the proper paths for the variables of Z avoid Z except at their starting vertices, and hence also avoid the smaller set X except at their starting vertices. For the next variable $W \in \mathbf { Z } \backslash \mathbf { X }$ , its proper path for Z avoids X entirely. Thus $W \in \mathbf { A } = \mathtt { P o s s A n } ( Y ) _ { \mathcal { H } \backslash \mathbf { X } }$ , and the algorithm visits the branch that adds it. Inductively, it reaches and records Z.

Finally, each recursive call passes only the sufix after its newly selected variable. Selected indices therefore increase strictly. A given set has only one increasing sequence of indices, so it is generated on only one branch. The list becomes shorter at every recursive step, establishing termination. □

Enumeration cost. Let $\kappa = | \mathbb { M } _ { \mathcal { D } , Y } |$ be the total number of MISs returned, let $n = | \mathbf { V } |$ and m be the number of edges, and let $r = | \mathsf { P o s s A n } ( Y ) _ { \mathcal { D } } \setminus \{ Y \} |$ be the number of initial candidates. We first express the cost in terms of $\kappa .$ Each call that passes $\operatorname { E q . }$ (82) records exactly one distinct MIS. Thus, there are exactly κ such calls. Each starts at most r recursive calls, while a call that fails the test stops immediately. Thus, the total number of calls is at most $1 + r \kappa$ . For a MAG with the reverse topological order above, no call fails the test, and the number of calls is exactly κ.

With adjacency-list graph operations, computing the remaining possible ancestors and checking all selected variables takes $\mathcal { O } ( n + m )$ time per call. Copying a sufix or a selected set takes at most $\mathcal { O } ( r )$ time. We also account for the collection unions in the algorithm. Each output belongs to the returned collections of at most $r + 1$ calls along its branch, and each such collection can be copied at most r times while processing the remaining candidates.

![](images/b4b2ef71833de67495eb3e5b755403604703cc9cce014ba46e519c866deaf133.jpg)  
Figure 12: Three MAGs $\mathcal { M } _ { 1 } - \mathcal { M } _ { 3 }$ are represented by P. In contrast, $\mathcal { M } _ { 4 }$ is not represented by $\mathcal { P }$

Repeated copying therefore contributes at most $\mathcal { O } ( ( r + 1 ) ^ { 2 } \kappa )$ time. This is absorbed by the bound below because $r + 1 \leq n .$ A straightforward implementation of the preliminary ordering takes $\mathcal { O } ( n ^ { 3 } )$ time, giving the conservative total bound:

$$
\mathcal { T } _ { \mathrm { M I S s } } = \mathcal { O } ( n ^ { 3 } + ( r + 1 ) \kappa ( n + m ) ) .\tag{83}
$$

For a fixed number κ of output MISs, this bound is polynomial in the graph size. It counts both the unsuccessful calls and the cost of collecting the outputs, rather than only the cost of checking a single candidate.

However, κ itself can be exponential. For example, a MAG consisting of r candidate vertices $X _ { 1 } , \cdots , X _ { r }$ , the reward Y, and the edges $X _ { i } \to Y$ has $\kappa = 2 ^ { r }$ . Every subset of the candidate vertices is an MIS because each selected variable has a directed path to Y . Thus, polynomial dependence on a fixed output count does not imply polynomial time for all input graphs. The recursion reduces unnecessary exploration by rejecting an invalid set together with all of its extensions.

Possible vacuousness. One might expect that if X is an MIS relative to $\langle \mathcal { P } , Y \rangle$ , then it is also an MIS relative to ⟨M, Y⟩ for some MAG M conforming to the PAG P. However, this is not always the case, i.e., X need not be an MIS for any MAG represented by P.

For concreteness, consider the PAG P in Fig. 12 where $\mathbf { X } = \{ A , B \}$ is an MIS with respect to $\langle \mathcal { P } , Y \rangle$ since both A and B have proper possibly directed paths to $Y \ ( { \mathrm { i . e . , } } \ A \circ { \mathrm { - o } } \ Y$ and $B \circ { \mathrm { - } } \circ Y$ , respectively). However, we will demonstrate that at least one of A and B is irrelevant to the reward Y in every conforming MAG. To see this, consider binary SCMs with mutually independent exogenous variables $U _ { A } , U _ { B } , U _ { Y } \sim \mathtt { B e r n } ( \epsilon )$ , where $\begin{array} { r } { 0 < \epsilon < \frac { 1 } { 2 } } \end{array}$ First, for the proper subset $\mathbf { Z } _ { 1 } = \varnothing$ , we can construct an SCM $ { \boldsymbol { S } } _ { 1 }$ as follows:

$$
\mathcal { S } _ { 1 } = \left\{ \begin{array} { l l } { \mathbf { U } = \{ U _ { A } , U _ { B } , U _ { Y } \} } \\ { \mathbf { V } = \{ A , B , Y \} } \\ { \mathcal { F } = \left\{ \begin{array} { l l } { f _ { A } ( y , u _ { A } ) = y \oplus u _ { A } } \\ { f _ { B } ( u _ { B } ) = u _ { B } } \\ { f _ { Y } ( b , u _ { Y } ) = b \oplus u _ { Y } } \\ { P ( \mathbf { U } ) = \{ U _ { V } \sim \mathrm { B e r n } ( \epsilon ) : \forall U _ { V } \in \mathbf { U } \} . } \end{array} \right. } \end{array} \right.\tag{84}
$$

Its corresponding MAG is shown in Fig. 12b. Here, $\mu _ { \emptyset } = 2 \epsilon ( 1 - \epsilon )$ while $\mu _ { \{ a , b \} ^ { * } } = \mu _ { b ^ { * } } = 1 - \epsilon$ with $b ^ { * } = 1$ . Thus, $\mu _ { a , 1 } \neq \mu _ { \emptyset }$ for every $a \in \{ 0 , 1 \}$ in $S _ { 1 }$ . That is, there exists an SCM $S _ { 1 }$ such that:

$$
\mu _ { \mathbf { x } ^ { * } } \neq \mu _ { \mathbf { z } _ { 1 } ^ { * } }\tag{85}
$$

Next, for the proper subset $\mathbf { Z } _ { 2 } = \{ A \}$ , since $\mu _ { a } = \mu _ { \emptyset } = 2 { \epsilon } ( 1 - { \epsilon } )$ , we again have $\mu _ { a , 1 } \neq \mu _ { a }$ for every $a \in \{ 0 , 1 \}$ in $S _ { 1 }$ . That is, there also exists an SCM $S _ { 1 }$ such that:

$$
\mu _ { \mathbf { x } ^ { * } } \neq \mu _ { \mathbf { z } _ { 2 } ^ { * } }\tag{86}
$$

An analogous construction applies to ${ \bf Z } _ { 3 } = \{ B \}$ using $S _ { 2 }$ (Fig. 12c), defined as follows:

$$
\mathcal { S } _ { 2 } = \left\{ \begin{array} { l l } { \mathbf { U } = \{ U _ { A } , U _ { B } , U _ { Y } \} } \\ { \mathbf { V } = \{ A , B , Y \} } \\ { \mathcal { F } = \left\{ \begin{array} { l l } { f _ { A } ( u _ { A } ) = u _ { A } } \\ { f _ { B } ( y , u _ { B } ) = y \oplus u _ { B } } \\ { f _ { Y } ( a , u _ { Y } ) = a \oplus u _ { Y } } \\ { P ( \mathbf { U } ) = \{ U _ { V } \sim \mathrm { B e r n } ( \epsilon ) : \forall U _ { V } \in \mathbf { U } \} . } \end{array} \right. } \end{array} \right.\tag{87}
$$

Hence, there also exists an SCM $S _ { 2 }$ such that:

$$
\mu _ { \mathbf { x } ^ { * } } \neq \mu _ { \mathbf { z } _ { 3 } ^ { * } }\tag{88}
$$

These witnesses establish that $\{ A , B \}$ is an MIS for P. However, a common witnessing SCM would have a corresponding MAG in which neither intervention variable is redundant. By Thm. 4, both A and B would then have proper directed paths to $Y .$ . On this three-vertex skeleton, this requires $A \right. Y \left. B$ , creating an unshielded collider absent from $\mathcal { P } .$ . Hence no such MAG, and therefore no common witnessing SCM, exists.

## 4.1.2 Definitely Minimal Intervention Sets

To address this vacuousness, we propose the concept of definitely MIS, which ensures that an MIS in a PAG remains an MIS in some consistent MAG.

For an MIS, the witnessing SCM may depend on the proper subset being compared. A DMIS requires a single SCM to work for all proper subsets; the distinguishing assignment may still depend on the subset.

Definition 17 (Definitely minimal intervention set). Given $\langle { \mathcal { D } } , Y \rangle , \mathbf { X } \subseteq \mathbf { V } \backslash \{ Y \}$ is a DMIS if there exists an SCM ${ S ^ { * } }$ conforming to $\mathcal { D }$ such that, for every $\mathbf { Z } \subsetneq \mathbf { X }$ , there exists an assignment $\mathbf { x } \in \Omega _ { \mathbf { X } }$ satisfying $\mu _ { \mathbf { x } } ^ { S ^ { * } } \neq \bar { \mu _ { \mathbf { x } \cap \mathbf { Z } } ^ { S ^ { * } } }$

The collection of DMISs is denoted by $\mathbb { D } _ { \mathcal { D } , Y }$ . Since DMIS is stricter than MIS, their inclusion relation is as follows.

Proposition 5. If X is a DMIS with respect to $\langle { \mathcal { D } } , Y \rangle$ , then it is also an MIS.

Proof of Prop. 5. Let $S ^ { * }$ be the common SCM in Def. 17. For each $\mathbf { Z } \subsetneq \mathbf { X }$ , that definition supplies an assignment x with $\mu _ { \mathbf { x } } ^ { S ^ { * } } \neq \mu _ { \mathbf { x } \cap \mathbf { Z } } ^ { S ^ { * } }$ . Choosing the same $S ^ { * }$ for each subset satisfies exactly the quantifiers in Def. 16. □

For causal diagrams and MAGs, which have no circle marks, MIS and DMIS coincide.

![](images/046ec75e29bdc24f74f91fd30706eafd4cf337fbd900dabf18d349d7aeca68d6.jpg)  
Figure 13: The three MAGs $\mathcal { M } _ { 1 } - \mathcal { M } _ { 3 }$ are represented by the PAG P.

Proposition 6. Let D be either a causal diagram or a MAG. If X is an MIS with respect to $\langle { \mathcal { D } } , Y \rangle$ , then it is also a DMIS.

Proof of Prop. 6. If D is a causal diagram, let $\mathcal { G } = \mathcal { D }$ . If D is a MAG, let G have exactly its directed and bidirected edges, so its corresponding MAG is D. In either case, Prop. 1 and Thm. 4 imply that every $X \in \mathbf { X }$ has a proper directed path to Y in G.

Fix the SCM S<sup>∗</sup> supplied by Lemma 35. For any $\mathbf { Z } \subsetneq \mathbf { X }$ , choose $W \in { \bf X } \setminus { \bf Z }$ and apply that lemma to its proper path. Then $\mu _ { \mathbf { X } = \mathbf { 1 } } ^ { S ^ { \ast } } \neq \mu _ { \mathbf { Z } = \mathbf { 1 } } ^ { S ^ { \ast } }$ . The same SCM works for every proper subset and conforms to D. This establishes DMIS membership. If $\mathbf { X } = \varnothing$ , any conforming SCM witnesses the vacuous requirement. □

This equivalence between MIS and DMIS for causal diagrams and MAGs stems from the fact that the relevant ancestral relations are determined, namely, $\mathbf { X } \subseteq \mathsf { A n } ( Y ) _ { \mathcal { D } _ { \overline { { \mathbf { X } } } } }$ . We now turn to DMIS for PAGs, where ancestral relations are not fully determined, thereby creating a notable gap between MIS and DMIS. Recall the PAG P in Fig. 12a, where {A, B} is an MIS but not a DMIS.

Proposition 7. $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is a DMIS relative to $\langle \mathcal { P } , Y \rangle$ if and only if there exists a MAG M conforming to P such that X is an MIS relative to ⟨M, Y ⟩.

Proof of Prop. 7. (If) Suppose that X is an MIS for some $\mathcal { M } \in [ \mathcal { P } ]$ . By Prop. 6, there is one SCM ${ S ^ { * } }$ conforming to M such that every proper subset of X is distinguished by an assignment. Since this SCM also conforms to P, it witnesses DMIS membership for the PAG. (Only if) Let ${ \boldsymbol { S } } ^ { * }$ witness that X is a DMIS for P. Since ${ \boldsymbol { S } } ^ { * }$ conforms to P, choose a MAG $\mathcal { M } \in [ \mathcal { P } ]$ to which it conforms. The same SCM and assignments witness that X is a DMIS for M. By Prop. 5, it is an MIS for M. □

Thus, a DMIS has a common witnessing SCM, whereas an MIS for a PAG may rely on diferent SCMs for diferent proper subsets. The following examples illustrate the dificulty of expressing this common-witness requirement as a direct graphical criterion.

The dificulty for PAGs is that the proper paths required for diferent intervention variables must occur in the same represented MAG. In Fig. 12a, the vertices A and B cannot both be ancestors of Y in one represented MAG, because $A \right. Y \left. B$ would create an unshielded collider on the uncovered path $A \circ - \circ Y \circ - \circ B$

Consider the uncovered circle path $A \circ - \circ C \circ - \circ Y \circ - \circ D$ in the PAG P shown in Fig. 13a.   
Every consecutive triple on this path must remain a non-collider in every represented MAG.

![](images/7a1e6aae300527e8b8b4411080d88ce207ba6b10543954c5025897948d0de631.jpg)  
Figure 14: The intervention set {X, Z} is an MIS but not a DMIS for P. The MAGs $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ realize the two paths separately; the proper paths for {X, Z} cannot be jointly realized in a MAG represented by P.

This restricts the orientations that could make both A and D ancestors of $Y ,$ as follows. Consider any $\mathrm { M A G }$ represented by P (Figs. 13b to 13d) where $A { \circ { \mathrm { - } } \circ } C$ appears as a directed edge out of A $( \mathrm { e . g . } , \mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 } )$ . This results in $C  Y  D$ , as the path is of definite status. In contrast, if any MAG contains $D \to Y$ , this leads to $Y  C  A$ , as in $\mathcal { M } _ { 3 }$ . The important observation is that $A  C$ ensures that $D \not \in { \mathsf { A n } } ( Y ) _ { \mathcal { M } }$ , whereas $D \to Y$ ensures that $A \not \in \mathsf { A n } ( Y ) _ { \mathcal { M } }$ for any MAG $\mathcal { M } \in [ \mathcal { P } ]$ . This shows that A and D cannot simultaneously be ancestors of Y in M. Hence, $\{ A , D \}$ is not a DMIS relative to $\langle \mathcal { P } , Y \rangle$

We consider a more involved example in Fig. 14. In ${ \mathcal { P } } , \mathbf { X } = \{ X , Z \}$ is an MIS because X and $Z$ each have a proper possibly directed path to Y with respect to X, namely $X { \circ { \mathrm { - } } \circ A \circ { \mathrm { - } } \circ Y }$ and $Z \circ - \circ B \circ - \circ Y$ , respectively. These paths are realized as directed paths in $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 } .$ separately.

However, realizing both paths in a single MAG requires $X  A$ and $Z \to B ,$ , which forces $A  Z$ and $B  X$ to avoid new v-structures. These orientations produce the directed cycle $X \to A \to Z \to B \to X$ . The alternative choice of distinct first neighbors, $X  B$ and $Z  A .$ , is symmetric. A shared first neighbor instead creates a v-structure, either $X \right. A \left. Z$ or $X \right. B \left. Z$ . Since these cases exhaust the possible first neighbors, no MAG represented by P admits proper directed paths from both X and Z to Y with respect to X. Hence X is not a DMIS.

Remark 2 (On the graphical characterization of DMISs). The example in Fig. 14 provides a counterexample to the graphical characterization of DMISs for PAGs in Park et al. (2025). The paths $X \circ - \circ A \circ - \circ Y$ and $Z \circ - \circ B \circ - \circ Y$ meet the condition in Theorem 2 of Park et al. (2025): no uncovered path can have both starting edges as its end edges, because any connecting path contains a shielded triple involving the adjacent vertices A and B. Hence, $\{ X , Z \}$ satisfies the stated graphical condition but is not a DMIS, disproving its suficiency. We leave the development of a direct graphical criterion for DMIS membership for PAGs, together with a polynomial-time decision algorithm, to future work.

## 4.2 Generalizing Possibly-Optimal Minimal Intervention Sets

We now refine the notion of a possibly-optimal minimal intervention set (POMIS) by defining it over DMISs rather than MISs. This refined notion is consistent with the existing literature and serves as a natural extension, as supported by Props. 5 and 6, which imply that MIS and DMIS coincide for causal diagrams and MAGs.

Definition 18 (Possibly-optimal minimal intervention set (POMIS)). Let $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ be a DMIS relative to $\langle \mathcal { D } , Y \rangle$ . If there exists an SCM conforming to D such that $\mu _ { \mathbf { x } ^ { * } } > \mu _ { \mathbf { w } ^ { * } }$ for all $\mathbf { W } \in \mathbb { D } _ { \mathcal { D } , Y } \setminus \{ \mathbf { X } \}$ , we say that X is a possibly-optimal minimal intervention set (POMIS) relative to $\langle { \mathcal { D } } , Y \rangle$

For a PAG, this definition compares X against the DMISs of the entire represented model class within one SCM. All reward means and maximizing assignments in this comparison are evaluated in that same SCM. The following proposition relates this comparison to POMIS membership in a represented MAG.

Proposition 8. $I f \mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is a POMIS relative to $\langle \mathcal { P } , Y \rangle$ , then there exists a MAG M conforming to P such that X is a POMIS relative to $\langle { \mathcal { M } } , Y \rangle$

Proof of Prop. 8. Let ${ \boldsymbol { S } } ^ { * }$ witness that X is a POMIS relative to $\langle \mathcal { P } , Y \rangle$ . Since ${ S ^ { * } }$ conforms to ${ \mathcal { P } } ,$ choose a MAG $\mathcal { M } \in [ \mathcal { P } ]$ to which it conforms.

First, we will show that X is an MIS relative to $\langle { \mathcal { M } } , Y \rangle$ . For the sake of contradiction, suppose X is not an MIS relative to $\langle { \mathcal { M } } , Y \rangle$ . Then, repeatedly removing a graphically redundant subset yields an MIS $\mathbf { Z } \subsetneq \mathbf { X }$ such that $\mu _ { \mathbf { x } } = \mu _ { \mathbf { x } \cap \mathbf { Z } }$ for every assignment x and every SCM conforming to $\mathcal { M } .$ . In particular, under ${ \boldsymbol { S } } ^ { * }$ , maximizing over assignments gives $\mu _ { \mathbf { x } ^ { * } } = \mu _ { \mathbf { z } ^ { * } }$ . By Prop. 6, Z is a DMIS for M and hence for P. This contradicts the strict optimality of X against every other DMIS in $\mathbb { D } _ { \mathcal { P } , Y }$ under ${ \boldsymbol { S } } ^ { * }$

Thus, X is an MIS, and therefore a DMIS, for $\mathcal { M }$ . Moreover, $\mathbb { D } _ { \mathcal { M } , Y } \subseteq \mathbb { D } _ { \mathcal { P } , Y }$ , since an SCM witnessing DMIS membership for M also conforms to $\mathcal { P } .$ . The strict inequalities witnessed by ${ \boldsymbol { S } } ^ { * }$ therefore hold against every member of $\mathbb { D } _ { \mathcal { M } , Y } \setminus \{ \mathbf { X } \}$ . Consequently, X is a POMIS relative to $\langle { \mathcal { M } } , Y \rangle$ □

Failure of the converse. The converse of Prop. 8 does not hold. To see this, consider the PAG shown in Fig. 12a. The empty set is a POMIS for both MAGs shown in Figs. 12b and 12c. In each MAG, the edge entering $Y$ is invisible, so its endpoints may be confounded in a represented causal diagram. POMIS membership here asserts the existence of a compatible SCM in which the empty intervention is strictly optimal among MISs; it need not be optimal in the SCM displayed earlier. However, in every compatible SCM, at least one of A and B is not an ancestor of $Y$ , so intervening on that variable leaves the distribution of $Y$ unchanged; that is, either $P ( y \mid d o ( a ) ) = P ( y )$ or $P ( y \mid d o ( b ) ) = P ( y )$ . Since both singleton intervention sets are DMISs for the PAG, the empty set cannot be strictly better than every other DMIS.

MUCT and IB for ancestral graphs. We now investigate the graphical characterization of POMISs for MAGs and PAGs. To this end, we extend the notions of MUCT and IB to these graph classes. The main challenge is that ancestral graphs do not display the underlying latent structure or the inducing paths that give rise to their adjacencies. We therefore use edge visibility to identify directed edges that cannot be confounded in any represented causal diagram. To generalize the UC-territory, we introduce a possible c-component (Jaber et al., 2018), which provides a necessary condition for nodes to belong to the same c-component in an underlying causal diagram.

![](images/6f7f538f12893307b7aeafd44d4fa67733fc2b463ba88f3368de8cf1763b9658.jpg)  
Figure 15: (a, b) Causal diagram and corresponding MAG. (c) Induced graph M \ {A, B}.

Definition 19 (pc-component (Jaber et al., 2018)). Two nodes are in the same possible c-component (pc-component) if there is a path between them such that:

1. all non-endpoint nodes along the path are colliders; and

2. none of the edges are visible.

The pc-component relation need not be transitive. We use ${ \mathsf { p c } } ( X ) _ { \mathcal { Q } }$ for the set of vertices related to X by the path condition in the definition, including X itself, and define $\mathsf { p c } ( \mathbf { X } ) _ { \mathscr { Q } } \triangleq$ $\textstyle \bigcup _ { X \in \mathbf { X } } { \mathsf { p c } } ( X ) _ { \mathcal { Q } }$ . Thus, these sets need not form a partition.

For example, A and B are in the same pc-component in the PAG P in Fig. 4e because they are connected by the collider path $A \circ \to C  \circ B$ , whose edges are all invisible, and indeed $\mathsf { p c } ( A ) _ { \mathcal P } = \{ A , B , C \}$ . Furthermore, because $A \notin \mathsf { p c } ( Y ) _ { \mathcal { P } } = \{ B , Y \}$ , A and Y cannot belong to the same c-component in any causal diagram conforming to P. We now generalize MUCT and IB for PMGs.

Definition 20 (Unobserved confounders’ territory for PMGs). Given information $\langle \mathcal { Q } , Y \rangle$ and intervention set $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ , let $\mathcal { H } = \mathcal { Q } [ \mathsf { P o s s A n } ( Y ) _ { \mathcal { Q } \backslash \mathbf { X } } ]$ . A set of variables $\mathbf { T } \ \subseteq \ \mathtt { P o s s A n } ( Y ) _ { \mathcal { Q } \backslash \mathbf { X } }$ containing Y is called a UC-territory with respect to $\langle \mathcal { Q } , Y , \mathbf { X } \rangle$ if PossDe $( \mathbf { T } ) _ { \mathcal { H } } = \mathbf { T }$ and $\mathsf { p c } ( \mathbf T ) _ { \mathcal { H } } = \mathbf T$ . A UC-territory T is called a minimal UC-territory (MUCT) if no $\mathbf { T } ^ { \prime } \subsetneq \mathbf { T }$ is a UC-territory.

Definition 21 (Interventional border for PMGs). Let T be a minimal UC-territory with respect to $\langle \mathcal { Q } , Y , \mathbf { X } \rangle$ . Then $\mathsf { P a } ( \mathbf { T } ) _ { \mathcal { Q } } \setminus \mathbf { T }$ is called an interventional border (IB) with respect to $\langle \mathcal { Q } , Y , \mathbf { X } \rangle$

We denote MUCT and IB with respect to ⟨Q, Y, X⟩ as MUCT(Q, Y, X) and $| \mathsf { B } ( \mathcal { Q } , Y , \mathbf { X } )$ respectively. Note that the MUCT is unique. Starting from {Y } and repeatedly adding possible descendants and pc-related vertices in H yields a closed set contained in every UC-territory. Since the graph is finite, this process terminates at the unique minimal UC-territory.

For concreteness, consider M and $\mathbf { X } = \{ A , B \}$ in Fig. 15b. Here, we omit Poss because we are working with a MAG. Let H be the induced graph $\mathcal { M } [ \mathbb { A } \mathrm { n } ( Y ) _ { \mathcal { M } \backslash \mathbf { X } } ]$ (shown in Fig. 15c). In H, the nodes C and Y are in the same pc-component, and D is a descendant of C. This implies that $\mathbf { T } = \{ C , D , Y \}$ is the minimal closed set for De<sub>H</sub> and $\mathsf { p c } _ { \mathcal { H } }$ . Therefore, we have:

$$
\mathsf { M U C T } ( \boldsymbol { \mathcal { M } } , \boldsymbol { Y } , \mathbf { X } ) = \mathbf { T } = \{ C , D , Y \}\tag{89}
$$

$$
\mathsf { I B } ( \mathcal { M } , Y , \mathbf { X } ) = \mathsf { P a } ( \mathbf { T } ) _ { \mathcal { M } } \ \backslash \ \mathbf { T }\tag{90}
$$

$$
= \{ A , B \} .\tag{91}
$$

## 4.2.1 Characterization of POMIS for MAGs

The MUCT and IB construction gives a necessary and suficient condition for POMIS membership in a MAG.

Theorem 5 (Graphical characterization of POMIS for MAGs). $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ is a POMIS relative to $\langle { \mathcal { M } } , Y \rangle$ if and only if $\mathbf { X } = \mathsf { I B } ( \mathcal { M } , Y , \mathbf { X } )$

The proof is given in Sec. A.2.3. For the empty intervention in each MAG in Figs. 12b and 12c, the MUCT consists of Y and its parent, and the IB is empty. Thus, Thm. 5 confirms the POMIS membership stated above.

For any intervention set, we seek a causal diagram whose corresponding MAG is M and whose MUCT and IB after intervention agree with those computed from M. Lemma 36 establishes that such a diagram always exists. We also retain exactly the directed edges of ${ \mathcal { M } } ,$ so that the two graphs have the same proper directed paths and hence the same MISs. Start by copying all directed and bidirected edges of M into a causal diagram, and compute its MUCT after intervention. If this set is smaller than the MUCT computed from the MAG, choose a variable inside the current set that has invisible parents outside it but within $\dim ( Y ) _ { \mathcal { M } \backslash \mathbf { X } }$ . For every such parent, add a bidirected edge to the chosen variable, while keeping the original directed edge. All these edges to the chosen variable are added together. Recompute the MUCT and repeat until it agrees with the MAG’s MUCT. The proof explains both why a variable can always be chosen when the sets difer and why these additions preserve the corresponding MAG.

Fig. 16 applies the construction to the running example in Fig. 15b. With $\mathbf { X } = \{ A , B \}$ , the relevant non-intervened ancestors are $\mathtt { A n } ( Y ) _ { \mathcal { M } \backslash \mathbf { X } } = \{ C , D , Y \}$ . The copied causal diagram has no bidirected edges, so its MUCT after intervention is initially just {Y }. In the original MAG, however, both $C  Y$ and $D \to Y$ are invisible: every other vertex is adjacent to Y , so neither edge can have a visibility witness. The construction therefore adds $C  Y$ and $D  Y$ together. The three vertices are now connected by bidirected edges, giving IB $\{ A , B \} = \mathbf { X }$ , as required. Therefore, we conclude that $\mathbf { X } = \{ A , B \}$ is a POMIS with respect to $\langle { \mathcal { M } } , Y \rangle$

The constructed diagram retains exactly the directed edges of the MAG and need not coincide with G in Fig. 15a. That original diagram is also represented by M and has $\{ A , B \}$ as a POMIS: after intervention, $C  Y$ places C in the territory, descendant closure includes D, and the border is again $\{ A , B \}$

## 4.2.2 Searching for POMISs in Represented MAGs

The remainder of this section targets $\begin{array} { r } { \widehat { \mathbb { P } } _ { \mathcal { P } , Y } \triangleq \bigcup _ { \mathcal { M } \in \left[ \mathcal { P } \right] } \mathbb { P } _ { \mathcal { M } , Y } } \end{array}$ , where $\mathbb { P } _ { \mathcal { M } , Y }$ is the collection of POMISs of a represented MAG. By Prop. 8, the PAG-wide collection of POMISs in Def. 18 satisfies $\mathbb { P } _ { \mathcal { P } , Y } \subseteq \widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ , and this inclusion can be strict. The search below is sound and complete for the union.

![](images/4dd5e98f15c38a513eec7603a82783e6e94dd6ab66c3b05ef644434acabe3e12.jpg)  
(a) M

![](images/9a604a79b317c4bff17230697c857238d273d7b0971f9da4a315726d7837c832.jpg)  
(b) Initial diagram

![](images/8c1ce42c965f5d465dffa31508739fdbed1ad8ff0e5e0844cc37b128224b3ec7.jpg)  
(c) After addition  
Figure 16: Constructing a causal diagram for the MAG in Fig. 15b, with $\mathbf { X } = \{ A , B \}$ . Copying the MAG’s edges initially gives IB $\{ A , B , C , D \} \neq \mathbf { X }$ in (b). Adding $C  Y$ and $D  Y$ together gives IB $\{ A , B \} = \mathbf { X }$ in (c). Light blue regions mark MUCTs, green fill marks IB variables, and thick outlines mark intervention targets.

This union retains an optimal action for every compatible SCM. Fix such an SCM and its corresponding MAG M. For any intervention set X, let $\mathbf { T } = \mathsf { M U C T } ( \mathscr { M } , Y , \mathbf { X } )$ and $\mathbf { W } = \mathsf { I B } ( \mathcal { M } , Y , \mathbf { X } )$ . The necessity proof of Thm. 5 gives $\mu _ { \mathbf { x } ^ { * } } \leq \mu _ { \mathbf { w } ^ { * } }$ in this SCM. Moreover, $\mathrm { A n } ( Y ) _ { \mathcal { M } \backslash \mathbf { W } } = \mathbf { T }$ : descendant closure keeps a directed path from every member of T to Y inside T, while any directed path from outside T to $Y$ enters T through a parent in W. Every path used to generate T by descendant and pc closure also stays inside T. With visibility inherited from the same full MAG, restricting the closure construction to M[T] therefore leaves its result unchanged. Thus MUCT $( \mathcal { M } , Y , \mathbf { W } ) = \mathbf { T }$ and ${ \sf I B } ( \mathcal { M } , Y , { \bf W } ) = { \bf W }$ By Thm. 5, W is a POMIS of M. Applying this argument to an optimal intervention set proves that the union retains an optimal action.

To search for $\widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ , we partition [P] according to the possible orientations of circle marks at vertices in $\mathbf { X } \cup \{ Y \}$ . Each part of this partition is represented by a locally oriented PMG, whose graphical structure is used in the terminal IB test. To this end, we first review local transformations of Wang et al. (2023), which orient the circle marks incident to a given variable.

Local transformation. We define a local transformation (Wang et al., 2023) at $A \in \mathbf { V }$ as a set:

$$
\mathbf { C } _ { A } ^ { Q } \subseteq \{ V \in \operatorname { A d j } ( A ) _ { \mathscr { Q } } | A \circ - * V \} .\tag{92}
$$

The vertices in $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ are those whose edges with a circle mark at A will be oriented with arrowheads at A. That is, for $V \in \mathbf { C } _ { A } ^ { \mathcal { Q } }$ and $V ^ { \prime } \in \{ U \in \operatorname { A d j } ( A ) _ { \mathscr { Q } } : A \circ - * U \} \backslash \mathbf { C } _ { A } ^ { \mathscr { Q } } .$ , the local transformation places an arrowhead at A on the first edge and a tail at A on the second edge, with $\mathcal { R } _ { \mathtt { S B } }$ completing the latter as a directed edge.

$\mathcal { R } _ { \mathtt { S B } }$ If $X - \circ Y$ , then orient $X \mathrm { ~ { - o } } Y$ as $X  Y$

The rule $\mathcal { R } _ { \mathtt { S B } }$ naturally follows from the absence of selection bias<sup>16</sup>. The resulting local orientation is:

$$
V * - \circ A \circ - * V ^ { \prime } \implies V * \to A \to V ^ { \prime } .\tag{93}
$$

We say that a local transformation is valid if at least one MAG in $[ \mathcal { Q } ]$ is consistent with all of its chosen orientations. A local transformation can be viewed as a candidate statement: it resolves the circle marks at one vertex. In the search below, we consider alternative statements of this kind and retain those compatible with at least one represented MAG. Notably, Wang et al. (2022, 2023) and Wang et al. (2024a) demonstrated that the rules $\mathcal { R } _ { 1 } – \mathcal { R } _ { 3 } , \mathcal { R } _ { 4 } ^ { \prime } , \mathcal { R } _ { 8 } – \mathcal { R } _ { 1 0 }$ , and $\mathcal { R } _ { \mathtt { S B } }$ are complete for incorporating local transformations.

$\mathcal { R } _ { 4 } ^ { \prime } \mathrm { ~ I f ~ } \langle X , \cdot \cdot \cdot , W , Z , Y \rangle$ is a discriminating path between X and Y for $Z ,$ and $Z \circ - * Y$ then orient $Z \circ - * Y$ as $Z \to Y$

In other words, after applying a valid local transformation and then applying the rules until no further orientation is possible, the orientations in the resulting PMG are complete. The following theorem shows that the validity of a given local transformation can be verified in $\mathcal { O } ( | \mathbf { V } | ^ { 3 } )$ time.

Theorem 6 (Theorem 3 and Remark 3 in Wang et al. (2023)). Let $\mathcal { Q }$ be a PMG obtained from a $P A G ~ \mathcal { P }$ by applying valid local transformations and the orientation rules $\mathcal { R } _ { 1 } - \mathcal { R } _ { 3 }$ $\mathcal { R } _ { 4 } ^ { \prime } , \mathcal { R } _ { 8 } - \mathcal { R } _ { 1 0 }$ , and $\mathcal { R } _ { S B }$ to closure after each transformation. Given a local transformation $\mathbf { C } _ { A } ^ { \mathcal { Q } } ;$ , there exists a MAG $\mathcal { M } \in [ \mathcal { Q } ]$ with $A  * V$ for all $V \in \mathbf { C } _ { A } ^ { \mathcal { Q } }$ , and $A  V ^ { \prime }$ for all $V ^ { \prime } \in \{ V \in \operatorname { A d j } ( A ) _ { \mathcal { Q } } | A \circ - * V \} \backslash \mathbf { C } _ { A } ^ { \mathcal { Q } }$ if and only if $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ satisfies the following conditions:

1. PossDe $( A ) _ { \mathcal { Q } \backslash \mathbf { C } _ { A } ^ { \mathcal { Q } } } \cap \mathsf { P a } ( \mathbf { C } _ { A } ^ { \mathcal { Q } } ) _ { \mathcal { Q } } = \emptyset ,$

2. $\mathcal { Q } [ \mathbf { C } _ { A } ^ { \mathcal { Q } } ]$ is a complete graph;

3. Orient the subgraph Q[PossDe $( A ) _ { \mathcal { Q } \backslash \mathbf { C } _ { A } ^ { \mathcal { Q } } } \backslash \{ A \} ]$ as follows until no further orientations are required:

For any circle edge $V _ { l } \circ { \mathrm { - } } \circ V _ { j }$ , orient the edge as $V _ { l }  V _ { j } \ i f { \mathrm { : } }$

1. $\mathbf { F } _ { V _ { l } } \setminus \mathbf { F } _ { V _ { j } } \ne \emptyset ;$ or

2. $\mathbf { F } _ { V _ { l } } = \mathbf { F } _ { V _ { i } }$ and there exists $V _ { m } \in \mathtt { P o s s D e } ( A ) _ { \mathcal { Q } \backslash \mathbf { C } _ { A } ^ { \mathcal { Q } } } \backslash \{ A \}$ not adjacent to $V _ { j }$ such that $V _ { m }  V _ { l } \circ – \circ V _ { j }$

where $\mathbf { F } _ { V _ { l } } = \{ V \in \mathbf { C } _ { A } ^ { Q } \cup \{ A \} \mid V \ast - \circ V _ { l } \in { \mathcal { Q } } \}$ is computed before these orientations. These orientations must introduce no new v-structures and must not require both $V _ { l } \to V _ { j }$ and $V _ { j }  V _ { l }$ for the same original circle edge.

For concreteness, consider the PMG Q shown in Fig. 17a, in which $Z$ is a parent of every other vertex. The choice $\mathbf { C } _ { A } ^ { \mathcal { Q } } = \{ B , C \}$ supplies the local transformation shown in Fig. 17b:

$$
B \circ  A  \circ C \quad { \mathrm { a n d } } \quad A  V _ { i } \quad { \mathrm { f o r ~ } } i = 1 , 2 , 3 .\tag{94}
$$

The marks at $B$ and $C$ remain circles. Validity requires a MAG consistent with both $\mathcal { Q }$ and these local orientations.

The first condition of Thm. 6 checks whether the local transformation would create a directed or almost directed cycle.

$$
\mathrm { F o r } \ \mathbf { C } _ { A } ^ { Q } = \{ B , C \} , \ \mathtt { P o s s D e } ( A ) _ { \mathcal { Q } \backslash \mathbf { C } _ { A } ^ { \mathcal { Q } } } \cap \mathsf { P a } ( \mathbf { C } _ { A } ^ { \mathcal { Q } } ) _ { \mathcal { Q } }\tag{95}
$$

![](images/c67d06ce10df013eac88df9b76320847d13461f9c5586ab46b755a6839ead1af.jpg)

![](images/5aac7e73fbc6932ecd415a165ec587871a42d7fe73fb0f31a9cd390c67bd9608.jpg)

![](images/35ef3b66c739c8038bce747b01de6580de988ad87abe0e1d1915d946969a0e7d.jpg)  
(c) Condition 3

(b) C<sup>Q</sup><sub>A</sub> = {B, C}  
![](images/05ca380bcb0ad8a9dd3ba0e1561d4117f2f8ff2504ae1d5dc7352c34e24ec62f.jpg)  
(d) The resulting PMG  
Figure 17: Local transformation and the conditions in Thm. 6. Panels (a)–(b) specify the local transformation, (c) illustrates the third condition, and (d) shows the PMG after orientation closure. Blue edges show local orientations and their consequences. In (c), first and second refer to the two subconditions of condition 3.

$$
\begin{array} { l } { { = \left\{ A , V _ { 1 } , V _ { 2 } , V _ { 3 } \right\} \cap \left\{ Z \right\} } } \\ { { \ } } \\ { { = \varnothing . } } \end{array}\tag{96}
$$

(97)

All arrows incident to Z point away from Z, preventing a possibly directed path from A back to this parent of B and C. Thus, the local choice cannot close a directed or almost directed cycle through Z.

Next, the second condition checks whether the local transformation introduces a new vstructure at A. Since B and C are adjacent, $\mathcal { Q } [ \{ B , C \} ]$ is complete. Their adjacency shields the triple $B \circ \right. A \left. \circ C ,$ so the two new arrowheads do not create a v-structure at A.

Finally, the third condition checks whether orientations propagated to possible descendants of A create new v-structures or require conflicting directions. The induced subgraph in Fig. 17c is $V _ { 1 } \circ - \circ V _ { 2 } \circ - \circ V _ { 3 }$ , with:

$$
{ \mathbf { F } } _ { V _ { 1 } } = \{ A , C \} , \qquad { \mathbf { F } } _ { V _ { 2 } } = { \mathbf { F } } _ { V _ { 3 } } = \{ A \} .\tag{98}
$$

Here A is excluded from the induced subgraph but retained in the F sets. The first subcondition requires $V _ { 1 }  V _ { 2 }$ because $\mathbf { F } _ { V _ { 1 } } \setminus \mathbf { F } _ { V _ { 2 } } = \{ C \} \ne \emptyset$ . Since $\mathbf { F } _ { V _ { 2 } } = \mathbf { F } _ { V _ { 3 } }$ and $V _ { 1 }  V _ { 2 } \circ - \circ V _ { 3 }$ is unshielded, the second subcondition then requires $V _ { 2 }  V _ { 3 }$ . Both orientations avoid new v-structures and conflicting directions.

All three conditions hold, so the local transformation is valid. Orientation closure yields the PMG shown in Fig. 17d, with $C  V _ { 1 }  V _ { 2 }  V _ { 3 }$ fixed. The edges $B \circ - \circ C , B \circ  A$ , and $C \circ \to A$ retain four circle marks. Therefore, the resulting PMG represents multiple MAGs consistent with the local transformation.

The PMG incorporating local transformations satisfies the properties below.

Theorem 7 (Theorem 1 in Wang et al. (2023)). Let Q be a PMG obtained from a $P A G \mathcal { P }$ by applying valid local transformations and applying the orientation rules $\mathcal { R } _ { 1 } - \mathcal { R } _ { 3 } , \mathcal { R } _ { 4 } ^ { \prime } , \mathcal { R } _ { 8 } - \mathcal { R } _ { 1 0 }$ and $\mathcal { R } _ { S B }$ to closure after each transformation. Then Q satisfies the following properties:

• Closed. Q is closed under the orientation rules.

• Invariant. The arrowheads $( > )$ and tails (−) in Q are invariant in all the MAGs consistent with Q.

• Chordal.<sup>17</sup> Circle components in Q are chordal.

• Balanced. For any three nodes A, B, C in Q, if A ∗→ B ◦−∗ C, then there is an edge between A and C with an arrowhead at C, namely, $A \nrightarrow C$ . Furthermore, if the edge between A and B is $A  B$ , then the edge between A and C is either $A  C$ or $A \circ  C \ ( i . e .$ , it is not $A  C )$

• Complete. For each circle at A on any edge A ◦−∗ B in Q, there exist MAGs $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ consistent with Q such that $A  * B$ in $\mathcal { M } _ { 1 }$ and $A  B$ in $\mathcal { M } _ { 2 }$

Thus, PMGs obtained by valid local transformations retain the listed graphical properties of PAGs, while representing a class of MAGs restricted by the incorporated local transformations.

Properties: inherited from PAGs. Let P be a PAG, and let Q be obtained from P through a sequence of valid local transformations, applying the orientation rules $\mathcal { R } _ { 1 } - \mathcal { R } _ { 3 } ,$ $\mathcal { R } _ { 4 } ^ { \prime } , \mathcal { R } _ { 8 } - \mathcal { R } _ { 1 0 }$ , and $\mathcal { R } _ { \mathtt { S B } }$ to closure after each transformation. For any $\mathbf { A } \subseteq \mathbf { V }$ , let $\mathcal { Q } [ \mathbf { A } ]$ be the induced graph on A.

Lemma 3. For any distinct vertices $A , B , C \in \mathbf { A }$ , if A∗→ B ◦−∗C in $\mathcal { Q } [ \mathbf { A } ]$ , then $A \ast  C$ is also present. Furthermore, if $A  B$ , then the edge between A and C is either $A  C$ or $A \circ  C$

Proof of Lemma 3. By Thm. 7, Q satisfies the balanced property. Since A, $B , C \in \mathbf { A }$ , every edge among these vertices, together with its endpoint marks, is preserved in the induced graph $\mathcal { Q } [ \mathbf { A } ]$ . Both conclusions therefore hold in $\mathcal { Q } [ \mathbf { A } ]$ □

Lemma 4. Suppose that distinct vertices $A , B \in \mathbf { A }$ are connected by a circle path in $\mathcal { Q } [ \mathbf { A } ]$ Then the following properties hold:

1. If A and B are adjacent, their edge is $A \circ - \circ B$

2. For every $C \in \mathbf { A } \setminus \{ A , B \} , C * \to A$ if and only if $C \ast  B$ . Furthermore, $C  A$ if and only $i f C  B$

Proof of Lemma $\it 4 .$ Let $p : V _ { 0 } ( = A ) { \mathsf { o } } { - } \circ V _ { 1 } \circ { - } \circ \cdot \cdot \cdot \circ { - } \circ V _ { k } ( = B )$ be a circle path in $\mathcal { Q } [ \mathbf { A } ]$ For the first property, the case $k = 1$ is immediate. Suppose that $k \geq 2$ and the edge between A and B has an arrowhead at B. Applying Lemma 3 to $A { * } {  } V _ { k } \circ { - } \circ V _ { k - 1 }$ gives $A * \to V _ { k - 1 }$ Repeating this argument along $p$ yields $A * \to V _ { 1 }$ , contradicting $A { \circ { \mathrm { - } } \circ V _ { 1 } }$ . Reversing the roles of A and B also excludes an arrowhead at A. Since selection bias is absent and $\mathcal { Q }$ is closed under $\mathcal { R } _ { \mathtt { S B } }$ , the edge must therefore be $A \circ - \circ B$

For the second property, if C belongs to the same circle component as A and $B ,$ the first property excludes all four relations under consideration. Otherwise, repeated application of Lemma 3 along $p$ shows that $C \ast \to A$ implies $C \ast  B$ . Traversing $p$ in reverse gives the converse.

It remains to establish the equivalence for bidirected edges. Consider consecutive vertices $U , V$ on $p$ and suppose that $C  U$ . The balanced property gives $C \nrightarrow V$ . If this edge were $C  V$ , applying the stronger part of Lemma $3$ to $C  V \circ - \circ U$ would exclude $C  U$ . If it were $C \circ \to V .$ , applying the balanced property to $U  C \circ \to V$ would give $U \ast \to V$ , contradicting $U \circ { \mathrm { - } } \circ V$ . Thus, $C  V$ . Iterating along $p ,$ and then applying the same argument in reverse, proves the desired equivalence. □

Lemma 5. Let $\mathcal { G }$ be any causal diagram consistent with $\mathcal { Q } ,$ and let $X , Y \in \mathbf { A }$ . If $X \in$ $\mathtt { A n } ( Y ) _ { \mathcal { G } [ \mathbf { A } ] }$ , then $X \in \mathtt { P o s s A n } ( Y ) _ { \mathcal { Q } [ \mathbf { A } ] }$

Proof of Lemma 5. The case $X = Y$ is immediate. Otherwise, let $p$ be a directed path from X to $Y$ in $\mathcal { G } [ \mathbf { A } ]$ , and let $\mathcal { M }$ be the MAG corresponding to $\mathcal { G }$ . Each directed edge on $p$ is also present with the same orientation in $\mathcal { M }$ . Since $\mathcal { M } \in [ \mathcal { Q } ]$ , no endpoint mark in $\mathcal { Q }$ can contradict these orientations. Hence, the same vertex sequence forms a possibly directed path from $X$ to $Y$ in $\mathcal { Q } .$ . All its vertices belong to $\mathbf { A }$ , so this path is also present in $\mathcal { Q } [ \mathbf { A } ]$ Therefore, $X \in \mathtt { P o s s A n } ( Y ) _ { \mathcal { Q } [ \mathbf { A } ] }$ □

Lemma 6. For every directed edge $X  Y$ in ${ \mathcal { Q } } [ \mathbf { A } ] , i f X  Y$ is visible in $\mathcal { Q } ,$ then it remains visible in $\mathcal { Q } [ \mathbf { A } ]$

Proof of Lemma $\it 6 .$ Let $\mathcal { G }$ be any DAG with explicit latent vertices, whose corresponding MAG belongs $\mathrm { t o } \ [ \mathcal { Q } ]$ , and let U be its latent vertex set. Since $X  Y$ is visible in $\mathcal { Q } , \mathcal { G }$ contains no inducing path between X and Y , relative to U, that is into X.

Suppose that such a path existed in $\mathcal { G } [ \mathbf { A } \cup \mathbf { U } ]$ . Its edges would also be present in ${ \mathcal { G } } ,$ as would the directed paths from each collider to at least one endpoint. It would therefore be an inducing path in $\mathcal { G }$ relative to the same latent set U, contradicting visibility in Q. As $\mathcal { G }$ was arbitrary, $X  Y$ is visible in $\mathcal { Q } [ \mathbf { A } ]$ □

Lemma 7. Let G be any causal diagram consistent with $\mathcal { Q } ,$ and let $X , Y \in \mathbf { A }$ . If $X \in$ ${ \mathsf { c c } } ( Y ) _ { { \mathcal { G } } [ \mathbf { A } ] }$ , then $X \in { \mathsf { p c } } ( Y ) _ { \mathcal { Q } [ \mathbf { A } ] }$

Proof of Lemma 7. The case $X = Y$ is immediate. Otherwise, let $p$ be a bidirected path between X and Y in $\mathcal { G } [ \mathbf { A } ]$ , and let $\mathcal { M } \in [ \mathcal { Q } ]$ be the MAG corresponding to ${ \mathcal { G } } .$ . By Lemma 29, some subsequence of the vertices of $p$ forms a collider path $p ^ { \prime }$ in $\mathcal { M }$ whose directed edges, if any, are invisible in $\mathcal { M }$

Among the paths between X and Y in M whose vertex sequences are subsequences of $p ^ { \prime }$ choose a shortest path $q$ subject to the requirements that every internal vertex is a collider and every directed edge is invisible in M. Such a path exists because $p ^ { \prime }$ satisfies these requirements. The proof of Lemma 30 in Jaber et al. (2018) shows that the internal vertices of this chosen path are definite colliders in the corresponding PAG P. Their arrowheads are preserved when $\mathcal { P }$ is refined to Q. Moreover, every vertex of q belongs to A. Thus, the corresponding path in $\mathcal { Q } [ \mathbf { A } ]$ has only definite colliders as internal vertices.

We next show that no edge on this path is visible in $\mathcal { Q } [ \mathbf { A } ]$ . Consider any directed edge $V  W$ on the path in $\mathcal { Q } [ \mathbf { A } ]$ . Because $\mathcal { M } \in [ \mathcal { Q } ]$ , the corresponding edge in M is also $V  W$ and it is invisible by the choice of $q .$ By Lemma 21, a DAG represented by M can be chosen to contain $V \left. U \right. W$ for a latent vertex U. Since $V , W \in \mathbf { A }$ , this common-parent path survives restriction to A and the latent vertices. As $\mathcal { M } \in [ \mathcal { Q } ]$ , this DAG belongs to the class used to interpret visibility in $\mathcal { Q } [ \mathbf { A } ]$ . Hence, $V  W$ is invisible in $\mathcal { Q } [ \mathbf { A } ]$ . The witnessing DAG may be chosen separately for each directed edge, since visibility is defined edge by edge.

Every remaining edge on the path is non-directed and is therefore invisible by definition. Consequently, q is a path in $\mathcal { Q } [ \mathbf { A } ]$ whose internal vertices are colliders and whose edges are all invisible. By Def. 19, this proves $X \in { \mathsf { p c } } ( Y ) _ { \mathcal { Q } [ \mathbf { A } ] }$ □

Lemma 8. pto (Alg. 12) is sound over $\mathcal { Q } [ \mathbf { A } ]$

Proof of Lemma 8. Construct an auxiliary directed graph T whose vertices are the circle components of $\mathcal { Q } [ \mathbf { A } ]$ . For distinct components B and C, include the directed edge $\mathbf { B }  \mathbf { C }$ if $\mathcal { Q } [ \mathbf { A } ]$ contains $U  V$ or $U \circ \to V$ for some $U \in \mathbf { B }$ and $V \in \mathbf { C }$

We first show that T is acyclic. Suppose instead that it contains a directed cycle ${ \bf B } _ { 1 } $ $\cdots  \mathbf { B } _ { k }  \mathbf { B } _ { 1 }$ . For each directed edge on this cycle, choose an edge $U _ { i } ~ \stackrel { ? } {  } ~ V _ { i }$ in $\mathcal { Q } [ \mathbf { A } ]$ with $U _ { i } \in \mathbf { B } _ { i }$ and $V _ { i } \in { \bf B } _ { i + 1 }$ , where indices are interpreted cyclically. Since $V _ { i }$ and $U _ { i + 1 }$ lie in the same circle component, the arrowhead and bidirected-edge equivalences in Lemma 4 imply $U _ { i } :  U _ { i + 1 }$ . Indeed, the arrowhead at $V _ { i }$ propagates to $U _ { i + 1 }$ , while a bidirected edge to $U _ { i + 1 }$ would imply a bidirected edge to $V _ { i } ,$ contrary to the choice of $U _ { i } \ { \stackrel { ? } { \to } } \ V _ { i }$

By the chordal property in Thm. 7 and Lemma 34, there exists a MAG $\mathcal { M } \in [ \mathcal { Q } ]$ obtained by replacing every $\hookrightarrow \to$ edge with a directed edge and orienting the circle components into DAGs without v-structures. The edges identified above would then form $U _ { 1 } \to U _ { 2 } \to \cdots \to U _ { k } \to U _ { 1 }$ in M, contradicting its ancestral property. Therefore, T is acyclic.

Every nonempty induced subgraph of T consequently has a vertex with no outgoing directed edge. Such a vertex corresponds to a remaining circle component B for which every edge to another remaining component has an arrowhead at its endpoint in B. Thus, B satisfies the condition for removal in pto. Removing a whole circle component leaves the other circle components unchanged. Repeatedly choosing and removing a component with this property therefore removes every component and terminates.

Finally, let $\mathcal { G }$ be any causal diagram consistent with Q. Every directed edge $U  V$ in $\mathcal { G } [ \mathbf { A } ]$ is also directed from U to V in its corresponding MAG. If U and V belong to diferent circle components of $\mathcal { Q } [ \mathbf { A } ]$ , their edge in Q must be $U \to V$ or $U \circ \to V$ . There is therefore a directed edge in $\tau$ from the component of U to the component of V. The component of V must be removed before that of $U _ { : }$ , because the latter still has an outgoing directed edge while both remain. Reversing the removal order thus places the component of U before that of $V$ , as required by every directed edge between components in $\mathcal { G } [ \mathbf { A } ]$ . The resulting partial topological order is therefore valid for $\mathcal { G } [ \mathbf { A } ]$ □

Building on these properties, we can treat PMGs obtained through local transformations in much the same way as PAGs.

Orienting according to local transformations. Fix an ordering of the vertices in $\mathbf { X } \cup \{ Y \}$ . Starting from ${ \mathcal { P } } _ { : }$ , process these vertices in this order, considering every valid local transformation at each step. After each transformation, apply the orientation rules $\mathcal { R } _ { 1 } - \mathcal { R } _ { 3 }$ $\mathcal { R } _ { 4 } ^ { \prime } , \mathcal { R } _ { 8 } - \mathcal { R } _ { 1 0 }$ , and $\mathcal { R } _ { \mathtt { S B } }$ until no further orientation is possible. Let $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ denote the resulting PMGs, where i indexes the sequences of local transformation choices. The constructed PMGs $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ inherit these graphical properties.

Corollary 1 (Inherited properties). Lemmas 3 to 8 apply with ${ \boldsymbol { \mathcal { Q } } } = { \mathcal { Q } } _ { \mathbf { X } } ^ { i }$

Proof of Cor. 1. By Thm. 7, the resulting PMG is orientation-complete. It also satisfies the common hypotheses of the lemmas. Their conclusions therefore hold for $\mathcal { Q } _ { \mathbf { X } } ^ { i } [ \mathbf { A } ]$ □

Proposition 9. Let $\mathcal { M } \in [ \mathcal { P } ]$ , and suppose that X is an MIS relative to $\langle { \mathcal { M } } , Y \rangle$ . Then there exists a PMG $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ representing M such that:

1. Every circle mark at a vertex in $\mathbf { X } \cup \{ Y \}$ in $\mathcal { P }$ is replaced by a tail or an arrowhead in $\mathcal { Q } _ { \mathbf { X } } ^ { i }$

2. $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ is closed under $\mathscr { R } _ { 1 } - \mathscr { R } _ { 3 } , \mathscr { R } _ { 4 } ^ { \prime } , \mathscr { R } _ { 8 } - \mathscr { R } _ { 1 0 }$ , and $\mathcal { R } _ { S B }$

3. For every ${ \boldsymbol { X } } \in \mathbf { X } , { \mathcal { Q } } _ { \mathbf { X } } ^ { i }$ contains a proper directed path from X to Y with respect to X.

Proof of Prop. 9. We first construct a sequence of local transformations that preserves M. Initially, $\mathcal { M } \in [ \mathcal { P } ]$ . At each step, let Q be the current PMG and let $A \in \mathbf { X } \cup \{ Y \}$ be the vertex being processed. Choose the local transformation at A so that every circle mark at A is oriented as in M. This choice is valid because $\mathcal { M } \in [ \mathcal { Q } ]$ provides a compatible MAG realizing all the selected marks simultaneously. The subsequent orientation rules are sound and therefore preserve compatibility with M. Inductively, the resulting PMG $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ satisfies $\mathcal { M } \in [ \mathcal { Q } _ { \mathbf { X } } ^ { i } ]$ . Every vertex in $\mathbf { X } \cup \{ Y \}$ is processed, and orientation closure is applied after each step. Hence, the first two conditions hold.

To prove the third condition, fix $X \in \mathbf { X }$ . By Thm. 4, choose a shortest proper directed path $p : V _ { 0 } ( = X ) \to V _ { 1 } \to \cdots \to V _ { k } ( = Y )$ with respect to X in M. This path is uncovered; otherwise, a chord between nonconsecutive vertices must point forward, since the earlier vertex is an ancestor of the later one and M is ancestral. Such a chord would give a shorter proper directed path, a contradiction.

Since M and $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ have the same adjacencies, the corresponding path is also uncovered in $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ . The mark at X on the edge between X and $V _ { 1 }$ is fixed in $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ and agrees with M. It is therefore a tail. Thus, closure under $\mathcal { R } _ { \mathtt { S B } }$ gives $X  V _ { 1 }$ in $\mathcal { Q } _ { \mathbf { X } } ^ { i }$

Now suppose that $V _ { j - 1 } \to V _ { j }$ has been established for some $1 \leq j < k$ . Because $\mathcal { M } \in [ \mathcal { Q } _ { \mathbf { X } } ^ { i } ]$ and $V _ { j } \to V _ { j + 1 }$ in $\mathcal { M }$ , the mark at $V _ { j }$ on the edge between $V _ { j }$ and $V _ { j + 1 }$ cannot be an arrowhead. If it is a circle, the uncovered configuration $V _ { j - 1 } \to V _ { j } \circ - { * } V _ { j + 1 }$ is oriented as $V _ { j } \to V _ { j + 1 }$ by $\mathcal { R } _ { 1 }$ , since $V _ { j - 1 }$ and $V _ { j + 1 }$ are not adjacent. If the mark is already a tail, the edge is $V _ { j } \to V _ { j + 1 }$ , using $\mathcal { R } _ { \mathtt { S B } }$ if necessary. Thus, closure implies $V _ { j } \to V _ { j + 1 }$ in either case. Thus, the path is a proper directed path from X to Y with respect to X in $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ . The same PMG $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ was constructed by preserving M throughout all local transformations. Hence, this conclusion holds simultaneously for every $X \in \mathbf { X }$ , establishing the third condition.

Algorithm 3: Recursive search over valid local transformations.   
1 function IsPOMIS(P, Y, X)   
$/ /$ Check whether the given set X is an MIS.   
2 if X does not satisfy Prop. 3 then return False   
3 return subIsPOMIS(P, X ∪ {Y}, Y, X)   
4 function subIsPOMIS(Q, A, Y, X)   
$/ /$ Apply the terminal IB test justified by Thm. 8.   
5 if A is empty then return ${ \sf I B } ( \mathcal { Q } , Y , { \bf X } ) = { \bf X }$   
6 Pick the first vertex $A \in \mathbf { A }$ in a fixed order.   
7 for each local transformation $\mathbf { C } _ { A } ^ { \mathcal { Q } } \subseteq \{ V \in \operatorname { A d j } ( A ) _ { \mathcal { Q } } \mid A \circ - * V \}$ do   
$/ /$ Check whether $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ is valid.   
8 if $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ satisfies Thm. 6 then   
$/ /$ Ensure a proper directed path after closure.   
9 if $A \in \mathsf { P o s s A n } ( Y ) _ { \mathscr { Q } \backslash \left( { \bf C } _ { A } ^ { \mathscr { Q } } \cup ( { \bf X } \backslash \{ A \} ) \right) }$ then   
10 Let $\mathcal { Q } ^ { \prime }$ be the PMG obtained by orienting the circle marks around A   
according to $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ and then applying the orientation rules $\mathcal { R } _ { 1 } - \mathcal { R } _ { 3 } , \mathcal { R } _ { 4 } ^ { \prime }$   
$\mathcal { R } _ { 8 } \mathcal { - R } _ { 1 0 } .$ , and $\mathcal { R } _ { \mathtt { S B } }$ until none applies.   
11 if subIs ${ \mathsf { P O M I S } } ( { \mathcal { Q } } ^ { \prime } , \mathbf { A } \setminus \{ A \} , Y , \mathbf { X } )$ then   
12 return True   
13 return False

Algorithmic characterization. Building on the theoretical results above, we present IsPOMIS (Alg. 3), which searches locally oriented PMGs and applies a terminal IB test. The following theorem establishes that the algorithm decides whether X is a POMIS in at least one MAG represented by P.

## Theorem 8. IsPOMIS $( A l g . ~ 3 )$ returns True if and only if $\mathbf { X } \in \widehat { \mathbb { P } } _ { \mathcal { P } , Y }$

The proof in Sec. A.2.5 combines Thm. 5 with the terminal PMG–MAG correspondence in Lemma 40. The latter constructs one compatible MAG with the required IB. The algorithm can therefore stop while circle marks remain at vertices outside $\mathbf { X } \cup \{ Y \}$

The algorithm first checks whether X is an MIS using Prop. 3. If this necessary condition fails, it returns False. Otherwise, it calls subIsPOMIS directly on ${ \mathcal { P } } ,$ with $\mathbf { A } = \mathbf { X } \cup \{ Y \}$ . When enumerating the full union, the candidates for these individual calls are the sets returned by MISs in Alg. 2.

At each recursive call, the algorithm selects the first vertex $A \in \mathbf { A }$ in a fixed order and considers local transformations $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ satisfying Thm. 6.

For each valid transformation, it additionally checks:

$$
A \in \mathsf { P o s s A n } ( Y ) _ { \mathscr { Q } \backslash \left( { \bf C } _ { A } ^ { \mathscr { Q } } \cup ( { \bf X } \backslash \{ A \} ) \right) } .\tag{99}
$$

Here, we set PossAn $( Y ) _ { \mathcal { H } } = \emptyset$ whenever $Y$ is not a vertex of H. When $A = Y$ , the condition holds automatically because possible ancestors include the vertex itself and $Y \not \in { \bf C } _ { Y } ^ { \mathcal { Q } } \cup { \bf X }$ When $A \in \mathbf { X }$ , the condition checks for a possibly directed path to $Y$ that avoids both $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ and the other intervention variables.

If the condition holds, the algorithm applies the selected local transformation and then the orientation rules $\mathcal { R } _ { 1 } { - } \mathcal { R } _ { 3 } , \mathcal { R } _ { 4 } ^ { \prime } , \mathcal { R } _ { 8 } { - } \mathcal { R } _ { 1 0 }$ , and $\mathcal { R } _ { \mathtt { S B } }$ until none applies, obtaining $\mathcal { Q } ^ { \prime } .$ . The following proposition shows that, for $A \in \mathbf { X }$ , this update establishes a proper directed path from A to $Y$

Proposition 10. Let Q be obtained from a PAG by valid local transformations and orientation closure, let $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ and let $A \in \mathbf { X } \cup \{ Y \}$ . Consider a MAG $\mathcal { M } \in [ \mathcal { Q } ]$ consistent with the valid local transformation $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ . Then the following equivalence holds:

$$
A \in \mathsf { P o s s A n } ( Y ) _ { \mathcal { Q } \backslash \left( \mathbf { C } _ { A } ^ { \mathcal { Q } } \cup ( \mathbf { X } \backslash \{ A \} ) \right) } \iff A \in \mathsf { A n } ( Y ) _ { \mathcal { M } \backslash ( \mathbf { X } \backslash \{ A \} ) } .\tag{100}
$$

If either graph does not contain $Y _ { + }$ , the corresponding ancestor set of Y is interpreted as empty.

Proof of Prop. 10. The case $A = Y$ is immediate because ancestors include the vertex itself. Hence assume $A \in \mathbf { X }$

(Only if) Suppose that the left-hand condition holds. There is a possibly directed path from A to Y in Q avoiding $\mathbf { C } _ { A } ^ { \mathcal { Q } } \cup ( \mathbf { X } \setminus \{ A \} )$ . By Lemma 33, the path has an uncovered possibly directed subsequence with the same endpoints. Since taking a subsequence introduces no vertices, the resulting path still avoids the deleted vertices.

Its first neighbor of A lies outside $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ . If the first edge already has a tail at A, it points out of A by closure under $\mathcal { R } _ { \mathtt { S B } }$ . Otherwise, its mark at A is a circle, which the selected local transformation replaces by a tail; $\mathcal { R } _ { \mathtt { S B } }$ then completes the directed edge. The rules $\mathcal { R } _ { 1 }$ and $\mathcal { R } _ { \mathtt { S B } }$ then propagate this orientation along the uncovered path. Consequently, the resulting PMG $\mathcal { Q } ^ { \prime }$ contains a proper directed path from A to Y with respect to $\mathbf { X }$ . The fixed MAG M agrees with the selected local transformations, so sound orientation closure preserves $\mathcal { M } \in [ \mathcal { Q } ^ { \prime } ]$ . The directed path in $\mathcal { Q } ^ { \prime }$ is therefore also present in this $\mathcal { M }$ , establishing the right-hand condition.

(If) Conversely, suppose that the right-hand condition holds. Then M contains a directed path from A to $Y$ avoiding $\mathbf { X } \setminus \{ A \}$ . Since $\mathcal { M }$ is consistent with the selected local transformation, the edge between A and every $C \in \mathbf { C } _ { A } ^ { \mathcal { Q } }$ has an arrowhead at A. By ancestrality, no such C can be a descendant of A. Hence, the directed path also avoids $\mathbf { C } _ { A } ^ { \mathcal { Q } }$ . Its corresponding path in Q is possibly directed, establishing the left-hand condition. □

The recursion continues with $\mathbf { A } \setminus \{ A \}$ . Once established, the proper directed paths are preserved by all subsequent valid transformations and orientation updates. Thus, when $\mathbf { A } = \varnothing$ , every intervention variable has a proper directed path to Y in the resulting orientationcomplete PMG $\mathcal { Q } _ { \mathbf { X } } ^ { i }$ . The algorithm then checks ${ \sf I B } ( \mathcal { Q } _ { \bf X } ^ { i } , Y , { \bf X } ) = { \bf X }$ . It returns True if any recursive branch satisfies this condition and False otherwise.

![](images/deed87630b9615ec599ea457bc20afcd601c6fb7e212fecefc11cb8b5d5df872.jpg)  
(a) P

![](images/83dd4d8d9d98a13bad0066bc70a2079b88659bfee01117f5141718e9ca2b3754.jpg)  
(b) After A

![](images/fdb3f84c71404185623ab256b81f4d046b5a7a92aba44f70f06f4b2c155e47bd.jpg)  
(c) Terminal PMG Q  
Figure 18: An IsPOMIS example for $\mathbf { X } = \{ A , B \}$ . (a) The input PAG P. (b) The PMG after the valid choice $\mathbf { C } _ { A } ^ { \mathcal { P } } = \varnothing$ . (c) The terminal PMG after choosing $\mathbf { C } _ { B } = \varnothing$ and then $\mathbf { C } _ { Y } = \varnothing$ , with two circle marks remaining. The light blue region marks the MUCT $\{ C , D , Y \}$ , green fill marks the $\mathrm { I B } \ \{ A , B \}$ and thick outlines mark the intervention targets. Purple directed edges are visible in their full PMG.

Corollary 2. Define IsDMIS by running Alg. 3 with the terminal return value $\begin{array} { r l } { { \sf I B } ( \mathcal { Q } , Y , { \bf X } ) = } & { { } } \end{array}$ X replaced by True, while retaining all other tests and recursive steps. For any $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ this procedure terminates and returns True if and only if $\mathbf { X } \in \mathbb { D } _ { \mathcal { P } , Y }$

Proof of Cor. 2. The search has finite branching, and each recursive descent removes one vertex from A, so it terminates.

If it accepts, the valid transformations and sound orientation closure yield a terminal PMG Q with $\emptyset \neq [ { \mathcal { Q } } ] \subseteq [ { \mathcal { P } } ]$ . By Prop. 10, every $X \in \mathbf { X }$ has a proper directed path to Y with respect to X in Q, preserved by subsequent orientations. Any $\mathcal { M } \in [ \mathcal { Q } ]$ therefore makes X an MIS by Thm. 4, and Prop. 7 gives DMIS membership.

Conversely, a DMIS has a witnessing MAG $\mathcal { M } \in [ \mathcal { P } ]$ in which it is an MIS, and it passes the initial MIS screen. Follow the local choices agreeing with this same ${ \mathcal { M } } .$ , as in Prop. 9. These choices are valid and preserve $\mathcal { M } ;$ Prop. 10 ensures that every proper path test passes. The test at Y holds by reflexivity. Thus an accepting terminal branch is reached. The argument includes $\mathbf { X } = \varnothing$ , for which the proper path requirements are vacuous. □

To illustrate the POMIS search, we return to the MAG in Fig. 16a and consider its PAG in Fig. 18a. The candidate is $\mathbf { X } = \{ A , B \}$ , and the processing order is $\langle A , B , Y \rangle$ . Both $A \circ  Y$ and $B \circ  Y$ are proper possibly directed paths, so this candidate belongs to the collection returned by MISs and passes the initial MIS check. The subsequent search determines whether this candidate is a POMIS in at least one represented MAG.

At A, choose $\mathbf { C } _ { A } ^ { \mathcal { P } } = \varnothing$ . This is valid, and the additional path test deletes only B, leaving the path $A \circ  Y$ . The transformation orients $A \to B , A \to C$ , and $A  Y$ . Orientation closure gives the PMG $\mathcal { Q } _ { A }$ in Fig. 18b. To see the role of the additional test, compare the valid choice $\mathbf { C } _ { A } ^ { \mathcal { P } } = \{ Y \}$ . This choice instead places an arrowhead at A on its edge to $Y _ { i }$ , but the path test deletes Y and therefore rejects the branch. Validity alone does not ensure a proper path to Y .

At B, the only remaining circle mark is on $B \circ  Y$ . Choose $\mathbf { C } _ { B } ^ { \mathcal { Q } _ { A } } = \varnothing$ . The path test deletes A and retains $B \circ  Y$ , so the branch continues and fixes $B  Y$

At Y , all incident marks are already arrowheads. Its only local choice is empty, and the reflexive possible ancestor condition holds after deleting $\{ A , B \}$ . No further orientations are needed. The resulting PMG Q is shown in Fig. 18c. After deleting X, the possible ancestors of Y are $\{ C , D , Y \}$ . The invisible edges $C \circ \to Y$ and $D \circ  Y$ place both $C$ and D in the territory reached from Y. Hence:

$$
\mathrm { F o r } { \bf X } = \{ A , B \} , \mathsf { M U C T } ( \mathscr { Q } , Y , { \bf X } ) = \{ C , D , Y \}\tag{101}
$$

$$
\Rightarrow | \mathsf { B } ( \mathcal { Q } , Y , \mathbf { X } ) = \{ A , B \} = \mathbf { X } .\tag{102}
$$

The algorithm therefore returns True. Two circle marks remain, one at C and one at $D$ Thus, the terminal graph is still a PMG. Replacing these two circles by tails gives the MAG in Fig. 16a. Its construction example in Fig. 16c realizes the same territory and border in a compatible causal diagram. Thus, the local search and the earlier MAG construction establish the same candidate’s POMIS membership from their respective graph representations.

Runtime analysis. Let $n = | \mathbf { V } |$ , let m be the number of edges in ${ \mathcal { P } } ,$ and let $k = | \mathbf { X } | + 1$ The initial MIS check can be implemented in $O ( n + m )$ time using a reverse reachability search from Y along possibly directed adjacencies. Vertices in X are marked when reached but are not expanded, ensuring that the corresponding paths avoid other intervention variables.

Let $p \leq n - 1$ bound the number of circle marks at any vertex being processed. Each recursive call considers at most $2 ^ { p }$ local transformations. Checking the validity of one transformation takes $\mathcal { O } ( n ^ { 3 } )$ time (Wang et al., 2023), while the condition in Prop. 10 can be evaluated by a reachability search in ${ \mathcal { O } } ( n + m )$ time.

The recursion has depth k and examines at most $k \cdot 2 ^ { k p }$ candidate transformations and $2 ^ { k p }$ terminal states. Let $\tau _ { \tt c l o s u r e }$ denote the worst-case cost of one orientation closure and $\mathcal { T } _ { \mathtt { I B } }$ the worst-case cost of evaluating the terminal IB condition. A conservative upper bound for one call to IsPOMIS is therefore:

$$
\mathcal { O } \left( n + m + 2 ^ { k p } \left[ k \left( n ^ { 3 } + \mathcal { T } _ { \mathrm { c l o s u r e } } \right) + \mathcal { T } _ { \mathrm { I B } } \right] \right) .\tag{103}
$$

To enumerate all members of $\widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ , first obtain $\mathbb { M } _ { \mathcal { P } , Y }$ using MISs, and then retain the sets accepted by IsPOMIS. Every POMIS in a represented MAG is an MIS for the PAG, so this restriction preserves every member of the union. There are κ candidates. The total cost is the MIS enumeration cost in Eq. (83), plus the sum of the individual IsPOMIS costs in Eq. (103) over these κ sets.

An alternative is to enumerate the MAGs represented by P and evaluate their IB conditions. For example, MagList (Wang et al., 2024a) enumerates these MAGs without duplication by recursively determining local structures throughout V. For a given candidate X, IsPOMIS branches only on local transformations at $\mathbf { X } \cup \{ Y \}$ , while applying orientation closure to the entire graph. A terminal PMG may therefore represent multiple MAGs. Its IB test is justified by Thm. 8.

For the PMGs visited by the algorithm, Cor. 11 shows that visibility is exactly characterized by the graphical criterion in Lemma 2, evaluated in the full PMG. For each directed edge

![](images/9a8c865e0407756c1a9bb2b3dd121ef50085e52719ec946da1b960ba9fac57ab.jpg)

![](images/76022d29caa36e1b8b11b6761b580cfe6ed31450046a35a1ab9513945df7b0ec.jpg)  
(a) Task 2

![](images/f81036c91a4cfad34c4dca604abc83e25684c16de96b46e5c29b99a6f3162a33.jpg)

![](images/ed9baf45b47f39f1397d8cc2d69d17e1c4575c357b99a861fa4f32d72fc67a08.jpg)  
(b) Task 3

PAG  
Causal diagram  
![](images/b464e0ff5bb9ef4dddb0bbf7d400f69682eaf99afdbc3e6ad4fc09bde17033ec.jpg)  
Causal diagram

![](images/1822d1dbd555aca2187d42de64ca9081feab9fd15aafe8a8f492824c28b18ee0.jpg)

![](images/1cab6b093bd539489ba18b0c1b33e1079f4c71532412bf22d7952a72f794ea7b.jpg)

![](images/f9af2e63299fb6771117dd5aef7ba9820df46c20870781c8ec2cb3b886c5e9d7.jpg)

(c) Task 4  
![](images/39756e642285ca441f8e5b178a853061d471b4293c61f7a6c437d070d08bc9ec.jpg)

(d) Task 5  
![](images/79473d2b748ad27b97b1298ad388d68f037cc6b33c14d092f8d71f9ddf264f9e.jpg)  
(e) Task 6  
Figure 19: Causal diagrams and oracle PAGs for Tasks 2–6.

$R \to B ,$ this criterion can be checked by finding the bidirected connections among parents of B and inspecting the edges entering those vertices. Starting from R, retain only parents of B along these bidirected connections, and check whether an edge entering a reached vertex starts at a vertex nonadjacent to B. Each search takes $\mathcal { O } ( n + m )$ time, so all directed edges can be checked in $\mathcal { O } ( m ( n + m ) )$ time. With their visibility fixed, the territory is reached from Y by traversing invisible edges in either direction and visible edges only from tail to arrowhead, within the possible-ancestor graph. Ancestor restriction, this traversal, and the border calculation each take $\mathcal { O } ( n + m )$ time. Thus $\mathcal { T } _ { \mathrm { I B } } = \mathcal { O } ( ( m + 1 ) ( n + m ) )$ is a valid bound.

## 4.3 Experiments

We evaluate six settings, each with a fixed binary SCM whose structural functions combine AND (∧), OR (∨), exclusive OR (⊕), and negation (¬), with independent Bernoulli exogenous variables. The complete structural functions and exogenous distributions are specified in

Table 2: Candidate action spaces. Each strategy column gives the number of intervention sets / arms.
<table><tr><td>Task</td><td> $| \mathbf { V } |$ </td><td>T</td><td>POMIS union</td><td>DMIS</td><td>BF</td></tr><tr><td>1</td><td>5</td><td>50k</td><td>5 / 11</td><td>7 / 17</td><td>16 / 81</td></tr><tr><td>2</td><td>6</td><td>50k</td><td>18 / 89</td><td>30 /195</td><td>32 / 243</td></tr><tr><td>3</td><td>10</td><td>50k</td><td>8 /54</td><td>120 / 1,593</td><td>512 / 19,683</td></tr><tr><td>4</td><td>6</td><td>50k</td><td>5 / 11</td><td>12 / 41</td><td>32  / 243</td></tr><tr><td>5</td><td>8</td><td>50k</td><td>16 / 81</td><td>32 / 189</td><td>128 / 2,187</td></tr><tr><td>6</td><td>8</td><td>50k</td><td>37 /211</td><td>70 / 627</td><td>128 / 2,187</td></tr></table>

![](images/2b19e087ea160e2bea238a210220ac72fdaa7a9252d45cd7012a4e1f6f1bb6c6.jpg)  
(a) Task 1

![](images/38e3a71d254092aa06853e42d5f6776358e84f36244c20a8770131aa1d73ccfd.jpg)  
(b) Task 2

Thompson sampling  
![](images/009fdc26c6109e0c4e915b47bda63760051b6bb1d01b1168267efb0cf8b8e43e.jpg)  
(c) Task 3

![](images/0e8c7b815d880e159ce33ce51d9aad2ceb01fe32c8148725ccff8aa9cff68f8d.jpg)  
(d) Task 4

![](images/0e59340f6757a41d8587fe517000f33325a131b6a4da51c9eb25baa51be8d078.jpg)  
(e) Task 5

![](images/3a351e04eb9a81e902773d342a73588d99c72c28978118b10b8559e4484f864b.jpg)  
(f) Task 6  
Figure 20: Experiments on Tasks 1–6. Curves show mean cumulative regret over 50,000 rounds; shaded regions show one standard deviation across 100 independent runs. Pink, purple, and green denote the POMIS union, DMIS, and brute force, respectively. Solid and dashed lines denote KL-UCB and Thompson sampling.

Sec. A.3. The agent receives the oracle PAG of the model’s causal diagram, and observes the reward of its selected intervention at each round. This PAG is obtained from graphical separation relations in the causal diagram, rather than from conditional independence tests on samples of the fixed SCM. Task 1 uses the PAG already shown in Fig. 13a. Fig. 19 places the causal diagrams of the SCMs for Tasks 2–6 beside their oracle PAGs.

Table 3: Final cumulative regret at $T = 5 0 { , } 0 0 0$ (mean ± standard deviation across 100 runs).
<table><tr><td>Task</td><td>Solver</td><td>POMIS union</td><td>DMIS</td><td>BF</td></tr><tr><td rowspan="2">1</td><td>TS</td><td> ${ \bf 2 3 . 3 \pm 9 . 4 }$ </td><td> $6 3 . 4 { \pm } 2 0 . 9$ </td><td> $1 9 9 . 6 { \scriptstyle \pm 2 2 . 6 }$ </td></tr><tr><td>KL-UCB</td><td> ${ \bf 5 4 . 2 \pm 1 2 . 6 }$ </td><td> $1 1 1 . 8 { \scriptstyle \pm 2 1 . 4 }$ </td><td> $3 5 2 . 5 { \pm } 2 5 . 9$ </td></tr><tr><td rowspan="2">2</td><td>TS</td><td> ${ \bf 6 2 5 . 3 \pm 4 3 . 9 }$ </td><td> $8 8 2 . 9 { \scriptstyle \pm 3 9 . 4 }$ </td><td> $9 7 0 . 2 { \scriptstyle \pm 4 0 . 0 }$ </td></tr><tr><td>KL-UCB</td><td> $\mathbf { 1 0 1 7 . 8 \div 3 4 . 7 }$ </td><td> $1 2 7 7 . 4 { \pm } 3 9 . 1 $ </td><td> $1 3 6 3 . 9 { \scriptstyle \pm 3 0 . 1 }$ </td></tr><tr><td rowspan="2">3</td><td>TS</td><td> $\mathbf { 3 1 . 0 } \pm \mathbf { 4 . 5 }$ </td><td> $7 7 0 . 7 { \scriptstyle \pm 1 9 . 2 }$ </td><td> $4 9 2 9 . 9 { \scriptstyle \pm 4 0 . 1 }$ </td></tr><tr><td>KL-UCB</td><td> $\mathbf { 3 0 . 5 \pm 4 . 1 }$ </td><td> $8 3 5 . 2 { \scriptstyle \pm 2 1 . 6 }$ </td><td> $6 2 2 2 . 6 { \scriptstyle \pm 2 6 . 8 }$ </td></tr><tr><td rowspan="2">4</td><td>TS</td><td> $\mathbf { 1 4 . 6 } \pm 3 . 8$ </td><td> $3 8 . 6 { \pm } 4 . 8 $ </td><td> $1 9 2 . 9 { \scriptstyle \pm 9 . 6 }$ </td></tr><tr><td>KL-UCB</td><td> $\mathbf { 1 0 . 5 \pm 2 . 8 }$ </td><td> $3 5 . 3 { \pm } 4 . 0$ </td><td> $1 9 0 . 4 { \pm } 9 . 8 $ </td></tr><tr><td rowspan="2">5</td><td>TS</td><td> $\mathbf { 1 0 7 1 . 3 } \pm 6 9 . 7$ </td><td> $2 2 1 6 . 2 { \scriptstyle \pm 9 9 . 6 }$ </td><td> $7 5 3 1 . 8 { \pm } 4 5 . 8$ </td></tr><tr><td>KL-UCB</td><td> $\mathbf { 2 3 5 9 . 8 \bot 9 9 . 3 }$ </td><td> $4 1 4 7 . 6 { \scriptstyle \pm 1 1 4 . 7 }$ </td><td> $8 2 2 4 . 0 { \scriptstyle \pm 2 5 . 4 }$ </td></tr><tr><td rowspan="2">6</td><td>TS</td><td> $\mathbf { 2 5 4 8 . 8 \pm 1 1 0 . 3 }$ </td><td> $3 8 0 4 . 5 { \scriptstyle \pm 6 8 . 4 }$ </td><td> $3 8 5 2 . 4 { \pm } 4 1 . 6 $ </td></tr><tr><td>KL-UCB</td><td> ${ \bf 3 7 4 5 . 7 { \scriptstyle \pm 7 4 . 5 } }$ </td><td> $4 3 7 4 . 8 { \scriptstyle \pm 4 2 . 0 }$ </td><td> $4 1 9 4 . 8 { \scriptstyle \pm 3 0 . 4 }$ </td></tr></table>

We compare three candidate action spaces. Brute force (BF) includes every assignment to every subset of $\mathbf { V } \setminus \{ Y \}$ , including the empty intervention. With binary variables, its size is $\begin{array} { r } { | \mathcal { A } _ { \mathtt { B F } } | = \sum _ { k = 0 } ^ { | \mathbf { V } | - 1 } { \binom { | \mathbf { V } | - 1 } { k } } 2 ^ { k } = 3 ^ { | \mathbf { V } | - 1 } } \end{array}$ . The DMIS strategy retains all assignments to DMISs. We obtain these sets by applying IsDMIS in Cor. 2 to the MIS candidates. The POMIS union strategy retains all assignments to the intervention sets in the union $\widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ over represented MAGs, as returned by the POMIS membership procedure. Thus, the label POMIS union in the figures and tables denotes this union, rather than $\mathbb { P } _ { \mathcal { P } , Y }$ . Table 2 reports the number of intervention sets and arms in each action space.

Each action space is paired with Thompson sampling (TS) (Thompson, 1933) and Bernoulli KL-UCB (Garivier and Capp´e, 2011; Capp´e et al., 2013). TS uses a Beta(1, 1) prior for each arm. KL-UCB initially samples arms once in a random order and then uses Bernoulli KL upper confidence indices with exploration function $f ( s ) = \log s + 3 \log \log s$ for $s \geq 3$ and $f ( s ) = 1$ otherwise, where s is the number of completed rounds. Thus, after initialization, the decision at round t uses $f ( t - 1 )$ . All tasks use $T = 5 0 { , } 0 0 0$ rounds, so each candidate action space completes this initial exploration. The reward mean of each intervention is computed from the fixed SCM to simulate Bernoulli rewards and evaluate regret. Arm selection uses only the selected arms and observed rewards from previous rounds.

The performance measure is the cumulative (pseudo-)regret defined in Sec. 3:

$$
R _ { T } = \sum _ { t = 1 } ^ { T } ( \mu ^ { \star } - \mu _ { A _ { t } } ) , \qquad \mu ^ { \star } = \operatorname* { m a x } _ { a \in \mathcal { A } _ { \mathtt { B F } } } \mu _ { a } ,\tag{104}
$$

using the same full action space benchmark for all strategies. For each of the six fixed SCMs, we verify that all three candidate action spaces retain an arm attaining this optimal mean. Each of the 36 configurations—six tasks, three strategies, and two solvers—is run 100 times. In Fig. 20, curves show the mean cumulative pseudo-regret and shaded regions show one standard deviation across these runs.

At T = 50,000, POMIS union has the lowest mean cumulative regret in all twelve task– solver comparisons in Table 3. Task 3 illustrates the benefit of substantial action reduction. POMIS union retains 54 arms, compared with 19,683 for BF, and both solvers incur much less regret. However, fewer arms do not guarantee lower finite-horizon regret. In Task 6, KL-UCB incurs mean regret 4,374.8 with DMIS, compared with 4,194.8 with BF. These results illustrate the benefit of the POMIS union action space on the six fixed SCMs with oracle PAGs.

## 5 On Transportability for Structural Causal Bandits

Secs. 3 and 4 used causal structure to reduce the action space while preserving an optimal action. Substantial exploration may nevertheless remain, and each intervention can be costly. A complementary strategy is to use data already collected from related source environments. Such data can narrow the possible rewards before interaction begins and reduce the cold-start problem.

However, an intervention that performs well in one environment may perform poorly in another. Transferring its reward distribution therefore requires accounting for the mechanisms that may change.

This section considers a known causal diagram and addresses these environmental discrepancies through transportability (Pearl and Bareinboim, 2011; Bareinboim and Pearl, 2012, 2016). Zhang and Bareinboim (2017) studied the transfer of prior observations for particular structures, including bow and instrumental-variable graphs. Bellot et al. (2023) and Deng et al. (2025) considered more general causal diagrams with single-variable interventions. Here, the available source experiments may involve several variables, and some variables may be non-manipulable in the target environment.

We first explain how non-manipulable variables change the POMIS action space. Then, we introduce the following contributions of this section.

• We derive causal bounds on individual action rewards by combining source distributions.

• We next provide dominance bounds, which compare the best rewards under diferent intervention constraints.

• Finally, we use these bounds to guide exploration in the target environment.

Relation to the published version. This section revises Park and Lee (2026a). The assumptions of Proposition 1 of Park and Lee (2026a) are corrected to allow $\mathbf { C } \subseteq \mathbf { C } ^ { \prime } \subseteq \mathbf { D }$ with C ancestral in G[C<sup>′</sup>] (Prop. 11). For the online policy, the variance interval uses the action-specific lower bound $\ell _ { \mathbf { x } } .$ , and a Bernstein index and its corresponding regret guarantee replace the published index and bound (Eqs. (166) and (169) and thm. 12). The experiments in Sec. 5.7 use this revised policy. Task 2 also uses a revised reward mechanism and controlled comparisons to isolate the efect of the upper dominance bound: the improvement reported for the original Task 2 did not establish that efect. The resulting regret measurements replace the published numerical results.

## 5.1 Structural Causal Bandits with Non-manipulable Variables

Sec. 3 allowed interventions on every observed variable except the reward. In many applications, an observed variable can be measured but cannot be set directly. For example, a treatment can be assigned while a patient’s physiological response must follow its natural mechanism. Let $\mathbf { N } \subseteq \mathbf { V } \setminus \{ Y \}$ denote these non-manipulable variables. The agent may now intervene only on sets $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \} \setminus \mathbf { N }$ (Lee and Bareinboim, 2019). The variables in N continue to respond to their causes, including interventions on other variables. Thus, this restriction changes which actions are available while preserving the SCM and reward notation introduced earlier. As before, the agent chooses an intervention based on the past history, before observing the variables generated in the current round.

MISs and POMISs under constraint. The definitions of MIS and POMIS in Sec. 3 extend by restricting their candidate intervention sets to the manipulable variables. We denote the resulting collections by $\mathbb { M } _ { \mathcal { G } , Y } ^ { \mathbf { N } }$ and $\mathbb { P } _ { \mathcal { G } , Y } ^ { \mathbf { N } }$ , respectively. Minimality is unchanged because every subset of a feasible intervention set is also feasible. Consequently:

$$
\mathbb { M } _ { \mathcal { G } , Y } ^ { \mathbf { N } } = \{ \mathbf { X } \in \mathbb { M } _ { \mathcal { G } , Y } : \mathbf { X } \cap \mathbf { N } = \emptyset \} .\tag{105}
$$

In particular, the ancestral criterion in Sec. 3 applies directly after checking that X is feasible.

Possible optimality also depends on which competing intervention sets remain available. A feasible MIS X belongs to $\bar { \mathbb { P } } _ { \mathcal { G } , Y } ^ { \mathbf { N } }$ if some SCM conforming to $\mathcal { G }$ satisfies:

$$
\mu _ { \mathbf { x } ^ { * } } > \mu _ { \mathbf { w } ^ { * } } \quad \mathrm { f o r ~ e v e r y ~ } \mathbf { W } \in \mathbb { M } _ { \mathcal { G } , Y } ^ { \mathbf { N } } \backslash \{ \mathbf { X } \} .\tag{106}
$$

An unconstrained POMIS that remains feasible still satisfies this requirement, because its witnessing SCM has fewer competitors to beat. Therefore:

$$
\{ \mathbf { X } \in \mathbb { P } _ { \mathcal { G } , Y } : \mathbf { X } \cap \mathbf { N } = \varnothing \} \subseteq \mathbb { P } _ { \mathcal { G } , Y } ^ { \mathbf { N } } \subseteq \mathbb { M } _ { \mathcal { G } , Y } ^ { \mathbf { N } } .\tag{107}
$$

The first inclusion can be strict. An MIS previously dominated by an intervention on a nonmanipulable variable may become a necessary candidate once that intervention is unavailable. Hence, simply removing infeasible sets from $\mathbb { P } _ { \mathcal { G } , Y }$ can discard an optimal feasible action.

Graphical characterization for POMISs under constraint. To account for this change, we use the latent projection from Def. 7 of $\mathcal { G }$ onto V \ N, denoted by ${ \mathcal { H } } = { \mathcal { G } } \langle \mathbf { V } \left. \mathbf { N } \right.$ The reason for projection is that an unavailable intervention cannot interrupt a causal pathway through N. Algebraically, each non-manipulable variable’s structural function can be substituted into the functions of its children. Repeating this substitution gives a model on H with the same reward distribution for every feasible intervention.

The graphical characterization ranges over SCMs compatible with the diagram, allowing the domains of non-manipulable variables to vary so their mechanisms can encode the inputs needed by their children.

Theorem 9 (Constrained POMISs (Lee and Bareinboim, 2019)). With $\mathcal { H } = \mathcal { G } \langle \mathbf { V } \mid \mathbf { N } \rangle$ $\mathbb { P } _ { \mathcal { G } , Y } ^ { \mathbf { N } } = \mathbb { P } _ { \mathcal { H } , Y }$ , and a feasible set X is a constrained POMIS if and only if $\mathsf { I B } ( \mathcal { H } _ { \overline { { \mathbf { X } } } } , Y ) = \mathbf { X }$

![](images/729a4d393ea4879c7dda53497f79639be7a6ffe5bdb22045d5b81c8628dafaad.jpg)  
Figure 21: A non-manipulable mediator changes which intervention sets are POMISs. The shaded variable $X _ { 2 }$ is observed but cannot be intervened on. The unconstrained POMISs are $\{ \emptyset , \{ X _ { 2 } \} \}$ whereas the constraint $\mathbf { N } = \{ X _ { 2 } \}$ yields $\{ \emptyset , \{ X _ { 1 } \} \}$ , as obtained from the latent projection in (b).

Proof of Thm. 9. Latent projection preserves directed paths between retained vertices that avoid an intervention set. The ancestral MIS criterion therefore gives $\mathbb { M } _ { \mathcal { G } , Y } ^ { \mathbf { N } } = \mathbb { M } _ { \mathcal { H } , Y }$ . Substituting the mechanisms in N preserves every feasible reward distribution, so each constrained POMIS witness on $\mathcal { G }$ is also a witness on H.

Conversely, the constructive proof of Thm. 3 provides a POMIS witness on H whose independent exogenous variables each afect at most two observed variables (Lee and Bareinboim, 2018). The reverse construction of Proposition 3 in Lee and Bareinboim (2019) expands this witness to $\mathcal { G }$ by letting restored variables encode the inputs needed by their children. The allowed choice of their domains permits this construction. It preserves the feasible reward distributions and hence the strict comparisons against the common MIS collection. This proves equality; the IB criterion follows from Thm. 3. □

For instance, consider the causal diagram in Fig. 21a. Its MISs are:

$$
{ \mathbb M } _ { { \mathcal G } , Y } = \{ \emptyset , \{ X _ { 1 } \} , \{ X _ { 2 } \} \} .\tag{108}
$$

The joint set $\{ X _ { 1 } , X _ { 2 } \}$ is redundant because fixing $X _ { 2 }$ removes the efect of $X _ { 1 }$ on $Y$ When both $X _ { 1 }$ and $X _ { 2 }$ are manipulable, the minimal UC-territory of $\mathcal { G }$ contains all three variables, and thus its border is empty. Hence ∅ is a POMIS. After intervening on $X _ { 2 }$ , the border is $\{ X _ { 2 } \}$ . By contrast, intervening on $X _ { 1 }$ removes its bidirected edge and leaves the border $\{ X _ { 2 } \}$ . Hence:

$$
\mathbb { P } _ { \mathcal { G } , Y } = \{ \emptyset , \{ X _ { 2 } \} \} .\tag{109}
$$

Now suppose that $X _ { 2 }$ is non-manipulable, so $\mathbf { N } = \{ X _ { 2 } \}$ . Projecting out $X _ { 2 }$ gives $X _ { 1 }  Y$ together with $X _ { 1 }  Y$ , as shown in Fig. 21b. In this projected graph, the border is empty before intervention and becomes $\{ X _ { 1 } \}$ after intervening on $X _ { 1 }$ . Therefore:

$$
{ \mathrm { F o r ~ } } \mathbf { N } = \{ X _ { 2 } \} , \quad \mathbb { P } _ { \mathcal { G } , Y } ^ { \mathbf { N } } = \{ \varnothing , \{ X _ { 1 } \} \} .\tag{110}
$$

Filtering the unconstrained POMISs would retain only ∅ and miss $\{ X _ { 1 } \}$

To see this, consider the following binary SCM conforming to the causal diagram in Fig. 21a:

$$
\mathcal { M } = \left\{ \begin{array} { l l } { \mathbf { U } = \{ U , U _ { 1 } , U _ { 2 } , U _ { Y } \} } \\ { \mathbf { V } = \{ X _ { 1 } , X _ { 2 } , Y \} } \\ { \mathcal { F } = \left\{ \begin{array} { l l } { f _ { X _ { 1 } } ( u , u _ { 1 } ) = u \oplus u _ { 1 } } \\ { f _ { X _ { 2 } } ( x _ { 1 } , u _ { 2 } ) = x _ { 1 } \oplus u _ { 2 } } \\ { f _ { Y } ( x _ { 2 } , u , u _ { Y } ) = x _ { 2 } \oplus u \oplus u _ { Y } } \\ { P ( \mathbf { U } ) = \{ U \sim \mathsf { B e r n } ( \frac { 2 } { 5 } ) , U _ { 1 } , U _ { 2 } , U _ { Y } \sim \mathsf { B e r n } ( \frac { 1 } { 4 } ) \} . } \end{array} \right. } \end{array} \right.\tag{111}
$$

The exogenous variables are mutually independent. The shared variable U induces $X _ { 1 }  Y$ 2 while $U _ { 1 } , U _ { 2 }$ , and $U _ { Y }$ supply separate random variation in the three mechanisms. In particular, $\begin{array} { r } { P ( X _ { 2 } = 1 \mid X _ { 1 } = 0 ) = \frac { 1 } { 4 } } \end{array}$ and $\begin{array} { r } { P ( X _ { 2 } = 1 \mid X _ { 1 } = 1 ) = \frac { 3 } { 4 } } \end{array}$ . Every assignment to $( X _ { 1 } , X _ { 2 } , Y )$ has positive observational probability. The observational and interventional means are:

$$
\mu _ { \emptyset } = \frac { 7 } { 1 6 } , \qquad \mu _ { X _ { 1 } = 0 } = \frac { 1 9 } { 4 0 } , \qquad \mu _ { X _ { 1 } = 1 } = \frac { 2 1 } { 4 0 } ,\tag{112}
$$

$$
\mu _ { X _ { 2 } = 0 } = \frac { 9 } { 2 0 } , \qquad \mu _ { X _ { 2 } = 1 } = \frac { 1 1 } { 2 0 } .\tag{113}
$$

Thus, when both variables can be manipulated, $d o ( X _ { 2 } = 1 )$ attains the optimal mean $\frac { 1 1 } { 2 0 }$ When $\mathbf { N } = \{ X _ { 2 } \}$ , the optimal feasible action is $d o ( X _ { 1 } = 1 )$ , with mean $\textstyle { \frac { 2 1 } { 4 0 } }$

For the remainder of this section, we write the actions induced by constrained POMISs as:

$$
\mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } } = \{ \mathbf { x } : \mathbf { X } \in \mathbb { P } _ { \mathcal { G } , Y } ^ { \mathbf { N } } , \mathbf { x } \in \Omega _ { \mathbf { X } } \} .\tag{114}
$$

This reduction preserves the best achievable expected reward under the constraint. It does not determine which feasible actions are most informative during learning: observations from an excluded action may still help estimate rewards of retained actions. In the transportability setting considered next, the same distinction applies to prior experiments, including experiments on variables that cannot be manipulated in the target environment.

## 5.2 Structural Causal Bandits under Transportability

Let $\pi ^ { * }$ denote the target environment and $\mathcal { M } ^ { * }$ its SCM, with reward ${ \cal Y } ~ \in ~ [ 0 , 1 ]$ and non-manipulable variables $\mathbf { N } ^ { * }$ . The agent also has information from source environments $\Pi = \{ \pi ^ { 1 } , \cdots , \pi ^ { n } \}$ , associated with SCMs $\mathcal { M } ^ { 1 } , \cdots , \mathcal { M } ^ { n }$ over the same observed variables and domains, all conforming to the known causal diagram ${ \mathcal { G } } .$ We use $P _ { \mathbf { x } } ^ { * }$ and $P _ { \mathbf { x } } ^ { i }$ for the distributions under $d o ( \mathbf { x } )$ in the target and source $\pi ^ { i }$ , respectively. Thus, $\mu _ { \mathbf { x } } ^ { * } = \mathbb { E } _ { P _ { \mathbf { x } } ^ { * } } Y$ is the target mean reward. A superscript ∗ on a distribution or mean specifies the target environment; the superscripts on assignments $\mathbf { x } ^ { * }$ and $\mathbf { x } ^ { \star }$ retain their meanings from Sec. 3. The target action space is $\mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } ^ { * } }$ . At round t, the agent selects $A _ { t }$ from this space and observes the outcome of its intervention. The cumulative pseudo-regret from Sec. 3, evaluated in the target, is:

$$
R _ { T } = T \mu _ { \mathbf { x } ^ { \star } } ^ { * } - \sum _ { t = 1 } ^ { T } \mu _ { A _ { t } } ^ { * } = \sum _ { \mathbf { x } \in A _ { \mathcal { G } , Y } ^ { \mathbf { N } ^ { \star } } } \Delta _ { \mathbf { x } } N _ { T } ( \mathbf { x } ) ,\tag{115}
$$

where $\Delta _ { \mathbf { x } } = \mu _ { \mathbf { x } ^ { \star } } ^ { * } - \mu _ { \mathbf { x } } ^ { * }$ and $\mathbf { x } ^ { \star }$ is optimal under the target constraint. The objective is to minimize $\mathbb { E } R _ { T }$ . Source data do not contribute rewards to this objective; they provide information for choosing target interventions.

Environmental discrepancies. For source $\pi ^ { i }$ , let $\mathbb { Z } ^ { i }$ contain the intervention sets for which experimental distributions are available. The empty set denotes observational data. We write:

$$
\mathbb { Z } = \{ \mathbb { Z } ^ { i } \} _ { i = 1 } ^ { n } , \qquad \mathcal { P } _ { \mathbb { Z } } ^ { \mathrm { I I } } = \{ P _ { \mathbf { z } } ^ { i } : \pi ^ { i } \in \Pi , \ \mathbf { Z } \in \mathbb { Z } ^ { i } , \ \mathbf { z } \in \Omega _ { \mathbf { Z } } \} .\tag{116}
$$

![](images/b3a9145a3b172bf3c9148d4f1477013c03d342ec653c3d47014ceea8ee8aa360.jpg)  
Figure 22: (a) The distribution of the unobserved variable U difers between source and target populations. (b) The corresponding collective selection diagram marks the afected mechanisms at $W$ and $X _ { 2 } .$ . (c) A collective selection diagram with $\Delta ^ { 1 } = \bar { \{ A \} }$ (red) and $\Delta ^ { 2 } = \{ C \}$ (blue). Squares mark selection variables.

A source may permit interventions that are unavailable in the target. For example, a laboratory experiment may directly control a biological mediator that cannot be assigned in a deployment population. Such experiments may still inform the rewards of feasible target actions.

For the probability bound algorithms below, observed variables have finite domains, with $\Omega _ { Y } \subseteq [ 0 , 1 ]$ . Source probabilities are treated as exact population quantities.

Definition 22 (Environment discrepancy and selection diagram). Let $\Delta ^ { i } = \Delta ^ { * , i 1 8 }$ contain the variables whose mechanisms may difer between $\pi ^ { * }$ and $\pi ^ { i }$ . For each $V \in \Delta ^ { i }$ , add a root node $S _ { V }$ and an edge $S _ { V }  V$ to ${ \mathcal { G } } .$ The resulting graph is the selection diagram $\mathcal { G } ^ { \Delta ^ { i } }$

A selection node marks a possible discrepancy between environments. Its absence asserts invariance of the corresponding mechanism, including the relevant exogenous variation. When a shared unobserved cause changes, its efects on all afected mechanisms must be represented. Equality of observed marginal distributions alone does not establish this invariance.

Let $\pmb { \Delta } = \{ \Delta ^ { i } \} _ { i = 1 } ^ { n } , \mathbf { S } ^ { i } = \{ S _ { V } : V \in \Delta ^ { i } \}$ , and $\mathbf { S } = \cup _ { i } \mathbf { S } ^ { i }$ . Adding all these selection edges gives the collective selection diagram $\mathcal { G } ^ { \Delta }$ . We assume that the environments admit a unified SCM with this diagram, in which selection nodes are independent roots controlling the environment-specific mechanisms. With $\mathbf { S } ^ { - i } = \mathbf { S } \setminus \mathbf { S } ^ { i }$ , its distributions satisfy:

$$
P _ { \mathbf { x } } ^ { \Delta } ( \mathbf { y } \mid \mathbf { S } = \mathbf { 0 } ) = P _ { \mathbf { x } } ^ { * } ( \mathbf { y } ) ,\tag{117}
$$

$$
P _ { \mathbf { x } } ^ { \mathbf { \Delta } } ( \mathbf { y } \mid \mathbf { S } ^ { i } = \mathbf { i } , \mathbf { S } ^ { - i } = \mathbf { 0 } ) = P _ { \mathbf { x } } ^ { i } ( \mathbf { y } ) .\tag{118}
$$

The assignment $\mathbf { S } ^ { i } = \mathbf { i }$ sets every selection node in $\mathbf { S } ^ { i }$ to i. In this representation, do-calculus can compare distributions across environments by determining whether the relevant selection nodes afect the queried outcome.

Example 9 (Treatment selection). Consider the schematic treatment system in Fig. 22a. Let $X _ { 1 }$ indicate an antihypertensive treatment, $X _ { 2 }$ an anti-diabetic treatment, W blood pressure, and $Y$ a favorable health outcome. The unobserved variable U influences both W and $X _ { 2 }$

Suppose that the agent can assign the two treatments but cannot directly set blood pressure, so $\mathbf { N } ^ { * } = \{ W \}$ . Prior data from another population can help evaluate treatment choices. However, if the distribution of $U$ difers across populations, both mechanisms involving W and $X _ { 2 }$ may change. The collective selection diagram in Fig. 22b represents this possibility by ${ \boldsymbol { \Delta } } ^ { 1 } = \{ { \boldsymbol { W } } , X _ { 2 } \}$ . The mechanism for Y, conditional on its causes and exogenous inputs, remains shared. We will return to this distinction when comparing feasible treatments with hypothetical interventions on $W$

## 5.3 Partial Transportability and Causal Bounds

A target mean $\mu _ { \mathbf { x } } ^ { * }$ is transportable from $\mathcal { P } _ { \mathbb { Z } } ^ { \Pi }$ if it has the same value in every collection of SCMs compatible with both these source distributions and the collective selection diagram (Lee et al., 2020). Equivalently, the available information uniquely determines the target mean. Diferent factors in a formula may come from diferent environments.

For the collective selection diagram shown in Fig. 22c, suppose that source $\pi ^ { 1 }$ supplies observational data and source $\pi ^ { 2 }$ supplies experiments on $\{ A , C \}$ , i.e., $\mathbb { Z } ^ { 1 } = \{ \emptyset \}$ and $\mathbb { Z } ^ { 2 } = \{ \{ A , C \} \}$ }. Let $\mathbf { N } ^ { * } = \{ C \}$ . Although $C$ cannot be manipulated in the target, experiments on C in $\pi ^ { 2 }$ can still identify the target reward of $d o ( a )$ . Specifically:

$$
P _ { a } ^ { * } ( y ) = \sum _ { b , c } P _ { a } ^ { * } ( b ) P _ { a } ^ { * } ( c \mid b ) P _ { a } ^ { * } ( y \mid b , c )\tag{119}
$$

$$
\stackrel { \mathrm { ( a ) } } { = } \sum _ { b , c } P _ { a , c } ^ { * } ( b ) P _ { \emptyset } ^ { * } ( c \mid a , b ) P _ { a , c } ^ { * } ( y \mid b )
$$

$$
{ \mathrm { R u l e s ~ 2 ~ a n d ~ 3 } }\tag{120}
$$

$$
= \sum _ { b , c } P _ { \emptyset } ^ { * } ( c \mid a , b ) P _ { a , c } ^ { * } ( b , y )\tag{121}
$$

$$
\stackrel { \mathrm { ( b ) } } { = } \sum _ { b , c } P _ { \emptyset } ^ { 1 } ( c \mid a , b ) P _ { a , c } ^ { 2 } ( b , y ) .\tag{122}
$$

Intermediate conditional probabilities are evaluated on conditioning events with positive probability; the final expression requires only the displayed source conditionals. Step (a) uses Rules 2 and 3 of do-calculus: the intervention on C does not afect B, conditioning on C can be exchanged with intervention in the reward factor, and the conditional mechanism for C is recovered observationally given A, B. In step (b), source $\pi ^ { 1 }$ supplies this conditional mechanism because its discrepancy is confined to A. Source $\pi ^ { 2 }$ supplies the joint distribution of $B , Y$ under $d o ( a , c )$ because intervening on $C$ removes its changed mechanism. Neither source needs to reproduce the entire target system. Multiplying these two contributions and summing over B, C gives the target reward distribution. The example illustrates why heterogeneous experiments can be useful even when they cannot be repeated in the target.

## 5.3.1 Partial Transportability

Exact transportability need not hold for every action. We call a mean partially transportable when the source information restricts it to an informative interval $[ \ell _ { \mathbf { x } } , u _ { \mathbf { x } } ] \subsetneq [ 0 , 1 ]$ that is valid for every compatible target SCM. To construct such intervals, we decompose the query into causal factors and bound the factors that cannot be determined exactly.

![](images/d064e87340b1f78626919389ca64a84a0129ed5292bcad2976fc44d8503577b6.jpg)  
(a) Pieces

![](images/8be4d1a67d9220724d431bfe9678ff650f2fe226a453b666dcb4e512d4da8d4b.jpg)  
(b) Transportable

![](images/36ef98efed902cbd9aa94af392b0b01b697e5fa44089249a406ab478c6e10776.jpg)  
(c) Non-transportable  
Figure 23: A jigsaw abstraction of the decomposition in $\operatorname { E q . }$ (125). Each piece represents a factor, and each color denotes a source environment supplying that factor. All pieces are available in (b). In (c), $Q _ { 1 }$ remains unavailable, motivating partial transportability bounds.

Q-decomposition. Recall the c-components from Sec. 3, and write ${ \mathsf { C C } } _ { \mathcal { G } }$ for their collection. For $\mathbf { C } \subseteq \mathbf { V }$ , define the causal factor:

$$
Q [ \mathbf { C } ] ( \mathbf { v } ) = P _ { \mathbf { v } \backslash \mathbf { C } } ( \mathbf { c } ) .\tag{123}
$$

It is the distribution of C when every observed variable outside C is fixed by intervention. Thus, $Q [ \mathbf { C } ]$ is shorthand for an interventional distribution; its displayed arguments also record those fixed values. We usually suppress its assignment argument v. Superscripts distinguish target and source factors, as in $Q ^ { * } [ \mathbf { C } ]$ and $Q ^ { i } [ \mathbf { C } ]$

Let $\mathbf { Y } ^ { + } = \mathsf { A n } ( Y ) _ { \mathcal { G } \backslash \mathbf { X } }$ . Only these remaining ancestors can afect Y after $d o ( \mathbf { x } )$ . The $\mathrm { Q - }$ decomposition (Tian and Pearl, 2003) gives:

$$
P _ { { \bf x } } ^ { * } ( y ) = \sum _ { { \bf y } ^ { + } \backslash \{ Y \} } \prod _ { { \bf C } \in { \bf c } { \bf c } _ { \mathcal { G } [ { \bf Y } ^ { + } ] } } Q ^ { * } [ { \bf C } ] ,\tag{124}
$$

$$
\mu _ { \mathbf { x } } ^ { * } = \sum _ { \mathbf { y } ^ { + } } y \prod _ { \mathbf { C } \in \mathfrak { c c } _ { g [ \mathbf { Y } ^ { + } ] } } Q ^ { * } [ \mathbf { C } ] .\tag{125}
$$

The fixed intervention arguments in each factor agree with x. Other fixed variables outside $\mathbf { Y } ^ { + } \cup \mathbf { X }$ do not afect these remaining mechanisms.

If $\mathbf { C } \cap \Delta ^ { i } = \emptyset$ , then all mechanisms retained in $Q [ \mathbf { C } ]$ are shared, and ${ \cal Q } ^ { * } [ { \bf C } ] = { \cal Q } ^ { i } [ { \bf C } ]$ When the latter factor is identifiable from an available source distribution and its graph, this equality transports the factor. Transporting every factor in Eq. (125) is suficient to determine the reward distribution and hence its mean.

Fig. 23 depicts the decomposition as a jigsaw puzzle. The query determines the pieces needed, and diferent source experiments may provide diferent pieces. A missing exact piece does not require discarding the entire query: a lower or upper bound for that piece may still constrain the final reward.

Bounds for unavailable factors. Return to the collective selection diagram in Fig. 22c, but now suppose that $\mathbb { Z } ^ { 1 } = \{ \emptyset , \{ C \} \}$ and $\mathbb { Z } ^ { 2 } = \{ \{ B \} \}$ . The target mean decomposes into:

$$
\mu _ { a } ^ { * } = \sum _ { b , c , y } y Q ^ { * } [ C ] Q ^ { * } [ B , Y ] .\tag{126}
$$

Source $\pi ^ { 1 }$ still identifies $Q ^ { * } [ C ] = P _ { \varnothing } ^ { 1 } ( c \mid a , b )$ . It does not identify $Q ^ { 1 } [ B , Y ]$ : within the available factor on $\{ A , B , Y \}$ , latent confounding prevents removal of the mechanism for A. Source $\pi ^ { 2 }$ intervenes on B and therefore does not provide the required joint factor on $B , Y$ Nevertheless, $Q ^ { * } [ B , Y ] = Q ^ { 1 } [ B , Y ]$ , since neither $B$ nor $Y$ is in $\Delta ^ { 1 }$

The natural bounds (Manski, 1990; Robins, 1989),

$$
P ( \mathbf { x } , \mathbf { y } ) \leq P _ { \mathbf { x } } ( \mathbf { y } ) \leq P ( \mathbf { x } , \mathbf { y } ) + 1 - P ( \mathbf { x } ) ,\tag{127}
$$

give the starting point. For units whose natural value is $\mathbf { x } ,$ replacing the corresponding mechanisms by x leaves the values received by all other mechanisms unchanged. Their outcome is therefore unchanged, so their observed joint probability supplies the lower bound. The remaining units can contribute at most their total probability, $1 - P ( \mathbf { x } )$ , to the upper bound. The same argument applies within an available causal factor.

For the next proposition, take $\mathbf { C } \subseteq \mathbf { C } ^ { \prime } \subseteq \mathbf { D }$ , where C is ancestral in $\mathcal { G } [ \mathbf { C } ^ { \prime } ] , \mathbf { C } \cap \Delta ^ { i } = \emptyset$ , and $Q ^ { i } [ \mathbf { D } ]$ is available or identifiable from source information. Ancestral marginalization gives $\begin{array} { r } { Q ^ { i } [ \mathbf { C } ] = \sum _ { \mathbf { c } ^ { \prime } \backslash \mathbf { C } } Q ^ { i } [ \mathbf { C } ^ { \prime } ] } \end{array}$

Proposition 11 (Bounds for a shared causal factor). The target factor satisfies:

$$
\sum _ { \mathbf { c } ^ { \prime } \backslash \mathbf { C } } Q ^ { i } [ \mathbf { D } ] \leq Q ^ { * } [ \mathbf { C } ] \leq \sum _ { \mathbf { c } ^ { \prime } \backslash \mathbf { C } } \left\{ Q ^ { i } [ \mathbf { D } ] + 1 - \sum _ { \mathbf { c } ^ { \prime } } Q ^ { i } [ \mathbf { D } ] \right\} .\tag{128}
$$

If $Q ^ { i } [ { \bf { C } } ^ { \prime } ]$ is identifiable, then $\begin{array} { r } { Q ^ { * } [ \mathbf { C } ] = \sum _ { \mathbf { c } ^ { \prime } \backslash \mathbf { C } } Q ^ { i } [ \mathbf { C } ^ { \prime } ] } \end{array}$

Proof of Prop. 11. Fix the arguments outside D and apply Eq. (127) within the resulting SCM, with the intervention variables $\mathbf { D } \setminus \mathbf { C ^ { \prime } }$ . This gives, at every assignment:

$$
Q ^ { i } [ \mathbf { D } ] \leq Q ^ { i } [ \mathbf { C } ^ { \prime } ] \leq Q ^ { i } [ \mathbf { D } ] + 1 - \sum _ { \mathbf { c } ^ { \prime } } Q ^ { i } [ \mathbf { D } ] .\tag{129}
$$

The inner sum varies only $\mathbf { C ^ { \prime } }$ , holding all other arguments fixed. Because C is ancestral in $\mathcal { G } [ \mathbf { C } ^ { \prime } ]$ , ancestral marginalization gives $\begin{array} { r } { Q ^ { i } [ \mathbf { C } ] = \sum _ { \mathbf { c ^ { \prime } } \backslash \mathbf { C } } Q ^ { i } [ \mathbf { C ^ { \prime } } ] } \end{array}$ by Lemma 3 in Tian and Pearl (2003). Also $Q ^ { * } [ \mathbf { C } ] = Q ^ { i } [ \mathbf { C } ]$ by the shared mechanisms. Summing the pointwise bounds proves the claim; the same marginalization of an identified $Q ^ { i } [ { \bf { C } } ^ { \prime } ]$ gives the exact case.

In the example, use $\mathbf { C } = \mathbf { C } ^ { \prime } = \{ B , Y \}$ and $\mathbf { D } = \{ A , B , Y \}$ . The enclosing factor is observed under the source experiment on C:

$$
Q ^ { 1 } [ A , B , Y ] = P _ { c } ^ { 1 } ( a , b , y ) .\tag{130}
$$

Combining its bounds with the exact factor for $C$ yields:

$$
\mu _ { a } ^ { * } \geq \sum _ { b , c , y } y P _ { \varnothing } ^ { 1 } ( c \mid a , b ) P _ { c } ^ { 1 } ( a , b , y )\tag{131}
$$

$$
\mu _ { a } ^ { * } \leq \sum _ { b , c , y } y P _ { \varnothing } ^ { 1 } ( c \mid a , b ) \left\{ P _ { c } ^ { 1 } ( a , b , y ) + 1 - P _ { c } ^ { 1 } ( a ) \right\} .\tag{132}
$$

Thus, the experiment on $C$ contributes information about a target intervention on $A ,$ even though it does not identify its reward exactly.

Algorithm 4: Partial transportability.   
1 function paTR(y, x, G, ∆, type)   
$/ /$ Use an available experiment when its discrepancy is irrelevant.   
2 if some $\mathbf { Z } \in \mathbb { Z } ^ { i }$ satisfies $\mathbf { X } = \mathbf { Z } \cap \mathbf { V }$ and $( \mathbf { S } ^ { i } \perp \perp \mathbf { Y } ) _ { \mathcal { G } ^ { \Delta } \backslash \mathbf { X } }$ then   
3 yield $P _ { \mathbf { z } \backslash \mathbf { V } , \mathbf { x } \cap \mathbf { Z } } ^ { i } ( \mathbf { y } )$ for every such source experiment; return   
$/ /$ Remove variables outside the ancestral query.   
4 if $\mathbf { V } ^ { \prime } = \mathbf { V } \setminus \mathsf { A n } ( \mathbf { Y } ) _ { \mathcal { G } } \neq \emptyset$ then   
5 yield $\mathrm { P A T R } ( \mathbf { y } , \mathbf { x } \setminus \mathbf { V } ^ { \prime } , \mathcal { G } \setminus \mathbf { V } ^ { \prime } , \{ \Delta ^ { i } \setminus \mathbf { V } ^ { \prime } \} _ { i } , \mathrm { t y p e } ) ;$ return   
$/ /$ Intervene on irrelevant remaining variables by Rule 3.   
6 ${ \bf i f } \ { \bf V } ^ { \prime } = ( { \bf V } \setminus { \bf X } ) \setminus \mathrm { A n } ( { \bf Y } ) _ { \mathcal { G } \backslash { \bf X } } \neq \emptyset$ then   
7 for each $\mathbf { v } ^ { \prime } \in \Omega _ { \mathbf { V } ^ { \prime } }$ do   
8 yield $\mathrm { P A T R } ( \mathbf { y } , \mathbf { x } \cup \mathbf { v } ^ { \prime } , \mathcal { G } , \pmb { \Delta } , \mathrm { t y p e } )$   
9 return   
$/ /$ Combine expressions for the c-components.   
10 if $| \mathsf { c c } _ { \mathcal { G } \backslash \mathbf { X } } | > 1$ then   
11 yield $\sum \Pi$ pa $\mathrm { T R } ( { \bf c } , { \bf v } \setminus { \bf c } , \mathcal { G } , \Delta , { \sf t y p e } ) ;$ return   
$\mathbf { v } \backslash ( \mathbf { X } \cup \mathbf { Y } ) \mathbf { C } { \in } { \mathbf { \mathfrak { c } } } _ { \mathcal { G } \backslash \mathbf { X } }$   
$/ /$ Try each source that preserves the remaining mechanisms.   
12 for $\pi ^ { i } \in \Pi$ with $\Delta ^ { i } \cap ( \mathbf { V } \setminus \bar { \mathbf { X } } ) = \emptyset .$ and $\mathbf { Z } \in \mathbb { Z } ^ { i }$ with $\mathbf { Z } \cap \mathbf { V } \subseteq \mathbf { X }$ do   
13 yield $\mathrm { P A I D } ( \mathbf { y } , \mathbf { x } \mid \mathbf { Z } , P _ { \mathbf { z } \mid \mathbf { v } , \mathbf { x } \cap \mathbf { Z } } ^ { i } , \mathcal { G } \setminus ( \mathbf { Z } \cap \mathbf { X } ) , \mathrm { t y p e } )$

Algorithmic partial transportability. paTR in Alg. 4 constructs lower or upper expressions for a target query, following the recursive structure of general transportability (Lee et al., 2020).

Its arguments are the outcome assignment ${ \mathbf { y } } ,$ intervention assignment x, current graph G, current discrepancy sets $\Delta$ , and bound direction type $\in \{ \ell , u \}$ . In every call, $\mathbf { V } = \mathbf { V } ( { \mathcal { G } } )$ denotes the current vertex set, and X, Y ⊆ V are the disjoint variable sets of $\mathbf x , \mathbf y$ . The sources Π, experimental sets $\mathbb { Z } ,$ and available distributions $\mathcal { P } _ { \mathbb { Z } } ^ { \Pi }$ are fixed global information. Each yielded expression is a candidate lower bound when $\mathtt { t y p e } = \ell$ or upper bound when type = u for the current target probability; an exact expression serves both directions.

A yield emits an expression, and a recursive yield forwards every expression obtained by that call. A following return ends the current call. If no source supplies a required expression, that call yields nothing. Whenever a source experiment is used, all available assignments z agreeing with the current query on $\mathbf { Z } \cap \mathbf { V }$ are considered. Values z \ V remain fixed external arguments. A source distribution supplied to a recursive call is first marginalized to the vertices of that call’s graph, with its intervention arguments held fixed.

The lines of Alg. 4 have the following roles:

• Use a direct source expression (lines 2–3). The experiment must set exactly X among the current vertices. The separation condition checks that its source discrepancy does not afect Y after those vertices are removed. Every qualifying source expression is then exact for the current target query, and the call ends.

• Remove non-ancestors (lines 4–5). Variables outside $\mathtt { A n } ( \mathbf { Y } ) _ { \mathcal { G } }$ cannot afect the query. The recursive call removes them from the graph, intervention assignment, and discrepancy sets, then forwards its expressions.

• Add irrelevant interventions (lines 6–9). Among the variables remaining after X is removed, those outside the ancestors of Y may be fixed to any assignment $\mathbf { v } ^ { \prime }$ by Rule 3. For every assignment $\mathbf { v } ^ { \prime } \in \Omega _ { \mathbf { V } ^ { \prime } }$ , the target query satisfies $P _ { \mathbf { x } } ^ { * } ( \mathbf { y } ) = P _ { \mathbf { x } \mathbf { v } ^ { \prime } } ^ { * } ( \mathbf { y } )$ The algorithm forwards the candidate expressions from all these assignments before ending the call. Although the target probability is invariant to this choice, the available source factors used to bound it may difer. CausalBound intersects the resulting bounds, which can tighten the interval without claiming sharpness.

• Decompose the query (lines 10–11). If several c-components remain, the algorithm queries the causal factor of each component C by fixing the other current variables. It forms every combination of one yielded expression per component, multiplies them, and sums over variables outside X ∪ Y. Components may use diferent sources; a combination contributes no expression if any required component yields none.

• Choose a source for the remaining component (lines 12–13). A source is eligible when its discrepancies avoid all remaining mechanisms, V \ X, and its experiment intervenes only within X among the current vertices. The variables already fixed by that experiment are removed from the graph and from the intervention still to be performed. The resulting source distribution and query are passed to paID, which supplies exact expressions or natural bounds.

The subroutine paID in Alg. 5 works within the selected source. Its arguments are y, x, P, G, type, with the same local graph, assignment, and bound-direction conventions. Here P is the source distribution or a causal factor over the current vertices, with outside assignments held fixed. The reductions in paTR ensure that each call satisfies:

$$
\mathbf { C } = \mathbf { V } \setminus \mathbf { X } = \mathsf { A n } ( \mathbf { Y } ) _ { \mathcal { G } \backslash \mathbf { X } } , \qquad \mathsf { c c } _ { \mathcal { G } \backslash \mathbf { X } } = \{ \mathbf { C } \} .\tag{133}
$$

The recursive calls of paID preserve this condition. The subroutine yields lower or upper expressions for the intervention query within $P ;$ the source-invariance check in paTR makes these expressions valid for the corresponding target query. In the factor products in Alg. 5, $\mathbf { v } _ { \prec j }$ denotes values preceding $V _ { j }$ in a topological ordering of the current graph, and the conditionals are formed from the current P while holding its outside assignments fixed.

• Finish an observational query (lines 2–3). If no intervention remains, marginalizing P to Y gives an exact expression.

• Restrict to ancestors (lines 4–6). Set $\mathbf { D } = \mathbb { A } \mathbf { n } ( \mathbf { Y } ) _ { \mathcal { G } }$ , sum out variables outside D from P, and retain only the graph and intervention on D.

• Bound an unavailable factor (lines 7–13). Line 7 names the unique remaining c-component C. If the whole current graph is one c-component, exact extraction stops. Lines 9–12 apply the lower or upper natural bound from Prop. 11 and marginalize to Y, after which the call ends. In the preceding example, this is the step that bounds $Q ^ { 1 } [ B , Y ]$ using $Q ^ { 1 } [ A , B , Y ]$

Algorithm 5: Partial identification subroutine for paTR.   
1 function paID(y, x, P, G, type)   
$/ /$ Marginalize when no intervention remains.   
2 if $\mathbf { X } = \dot { \boldsymbol { \varnothing } }$ then   
3 yield $\begin{array} { r } { \sum _ { \mathbf { v } \backslash \mathbf { Y } } P ( \mathbf { v } ) { : } } \end{array}$ ; return   
4 if $\mathbf { V } \neq \mathtt { A n } ( \mathbf { Y } ) _ { \mathcal { G } }$ then   
5 Set $\mathbf { D } = \mathrm { A n } ( \mathbf { Y } ) _ { \mathcal { G } } .$   
6 yield $\begin{array} { r l } { \mathrm { P A I D } ( \mathbf { y } , \mathbf { x } \cap \mathbf { D } , \sum _ { \mathbf { v } \backslash \mathbf { D } } P ( \mathbf { v } ) , \mathcal { G } [ \mathbf { D } ] } & { { } } \end{array}$ , type); return   
7 Let $\{ \mathbf { C } \} = \mathsf { c c } _ { \mathcal { G } \backslash \mathbf { X } } .$   
$/ /$ Use natural bounds when exact identification stops.   
8 if $\mathsf { c c } _ { \mathcal G } = \{ \mathbf V \}$ then   
9 if type $= \ell$ then   
10 yield $\begin{array} { r l } { \sum _ { \mathbf { v } \backslash ( \mathbf { X } \cup \mathbf { Y } ) } P ( \mathbf { v } ) } \end{array}$   
11 else   
12 yield $\begin{array} { r } { \sum _ { \mathbf { v } \backslash ( \mathbf { X } \cup \mathbf { Y } ) } \{ P ( \mathbf { v } ) + 1 - \sum _ { \mathbf { v } \backslash \mathbf { X } } P ( \mathbf { v } ) \} } \end{array}$   
13 return   
$/ /$ Extract an identifiable factor or recurse inside its c-component.   
14 if $\mathbf { C } \in \mathsf { c c } _ { \mathcal G }$ then   
15 yield $\begin{array} { r } { \sum _ { \mathbf { c } \backslash \mathbf { Y } } \prod _ { V _ { j } \in \mathbf { C } } P ( v _ { j } \mid \mathbf { v } _ { \prec j } ) } \end{array}$ ; return   
16 Let $\mathbf { C } ^ { \prime } \in \mathsf { c c } _ { \mathcal { G } }$ contain C.   
17 Set $\begin{array} { r } { Q [ \mathbf { C } ^ { \prime } ] = \prod _ { V _ { j } \in \mathbf { C } ^ { \prime } } P ( v _ { j } \mid \mathbf { v } _ { \prec j } ) . } \end{array}$   
18 yield $\mathrm { P A I D } ( \mathbf { y } , \dot { \mathbf { x } } \cap \mathbf { C } ^ { \prime } , Q [ \mathbf { C } ^ { \prime } ] , \mathcal { G } [ \mathbf { C } ^ { \prime } ] , \mathfrak { t } \mathbf { y } \mathsf { p } \mathsf { e } )$

• Extract an exact factor (lines 14–15). If C is already a c-component of the current graph, its Q-factor is the product of the displayed conditionals for $V _ { j } \in \mathbf { C }$ . Summing over C \ Y gives the exact query.

• Reduce to an enclosing component (lines 16–18). Otherwise, locate the ccomponent $\mathbf { C ^ { \prime } }$ containing C and extract its identifiable factor $Q [ \mathbf { C } ^ { \prime } ]$ from P. Recurse using this factor, the induced graph on $\mathbf { C ^ { \prime } }$ , and the interventions still inside $\mathbf { C ^ { \prime } }$ Assignments outside $\mathbf { C ^ { \prime } }$ remain fixed arguments of the extracted factor.

Theorem 10. Every expression yielded by paTR is a valid lower or upper bound, according to type, for the queried target probability.

The proof is given in Sec. B.1.1. Several expressions can bound the same probability. The theorem guarantees that these bounds are valid, although the source information may permit tighter bounds.

## 5.3.2 Causal Bounds

CausalBound in Alg. 6 takes the reward $Y \in [ 0 , 1 ]$ , an intervention assignment x, the diagram ${ \mathcal { G } } ,$ and discrepancies $\pmb { \Delta }$ , together with the fixed source experiments $\mathbb { Z }$ and distributions $\mathcal { P } _ { \mathbb { Z } } ^ { \Pi }$ . It returns an interval $[ \ell _ { \mathbf { x } } , u _ { \mathbf { x } } ]$ containing the target mean $\mu _ { \mathbf { x } } ^ { * }$ . Its calls to paTR use the fixed source information specified above.

```latex
Algorithm 6: CausalBound: partially transportable reward bounds.
// Retain the trivial bounds when no source gives an expression.
1 Initialize $[ \ell _ { { \bf x } } , u _ { { \bf x } } ] = [ 0 , 1 ]$
// Intersect all lower bounds after evaluating the source quantities.
2 for each expression $P _ { \mathbf { x } } ^ { \ell } ( y )$ yielded by p $\scriptstyle \mathrm { \scriptscriptstyle { \Lambda } } \mathrm { T R } ( y , \mathbf { x } , { \mathcal { G } } , \Delta , \ell )$ do
3 if after the stated chain-rule simplifications, a required source conditional is undefined at
an evaluated assignment then continue
4 Set $\begin{array} { r } { \ell _ { \mathbf { x } } = \operatorname* { m a x } \{ \ell _ { \mathbf { x } } , \sum _ { y } y P _ { \mathbf { x } } ^ { \ell } ( y ) \} } \end{array}$
// A probability upper bound can be truncated at one.
5 for each expression $P _ { \mathbf { x } } ^ { u } ( y )$ yielded by $\mathrm { P A T R } ( y , \mathbf { x } , \mathcal { G } , \Delta , u )$ do
6 if after the stated chain-rule simplifications, a required source conditional is undefined at
an evaluated assignment then continue
7 Set $\begin{array} { r } { u _ { \mathbf { x } } = \operatorname* { m i n } \{ u _ { \mathbf { x } } , \sum _ { y } y \operatorname* { m i n } \{ 1 , P _ { \mathbf { x } } ^ { u } ( y ) \} \} } \end{array}$
8 return $[ \ell _ { \mathbf { x } } , u _ { \mathbf { x } } ]$
```

Before checking support, we combine products of source probabilities by the chain rule, preserving their source and fixed experimental arguments. For example, for disjoint assignments $a , b , c , d ,$ the product $P ( a , b \mid c , d ) P ( c \mid d )$ is evaluated through the merged expression $P ( a , b , c \mid d )$ . The merged conditional requires $P ( d ) > 0$ and is defined even when $P ( c , d ) = 0$ This rule uses the joint-probability expression supplied by the chain rule; it does not permit arbitrary cancellation of zero-denominator ratios or numerical evaluation of undefined conditionals.

After these simplifications, we evaluate a formula only when every remaining source conditional it uses has a conditioning event with positive probability. Otherwise, that formula is omitted; another formula may still provide a bound. If none is usable, we retain $0 \leq \mu _ { \mathbf { x } } ^ { * } \leq 1$ Thus, the graph determines which source mechanisms can be transferred, while the source probabilities determine whether a particular formula can be evaluated.

Line 1 starts from the valid interval [0, 1]. Lines 2–4 evaluate each available lower expression, sum it against the nonnegative reward values, and retain the largest resulting lower bound. Lines 5–7 apply the same operation to upper expressions, truncating each probability bound at one before taking the smallest reward upper bound. Lines 3 and 6 apply the support test after the stated chain-rule simplifications and discard a formula if any remaining source conditional required for its evaluation is undefined. Thus a missing formula does not invalidate the other bounds, and a direction with no usable formula retains its trivial endpoint. Line 8 returns the intersection of the usable bounds. These operations are sound because multiplying pointwise probability bounds by $y \geq 0$ and summing preserves their direction.

Corollary 3. CausalBound returns $\ell _ { \mathbf { x } } \leq \mu _ { \mathbf { x } } ^ { * } \leq u _ { \mathbf { x } }$ . If this interval is a proper subinterval of [0, 1], the target mean is partially transportable.

Proof. By Thm. 10, every usable lower and upper expression satisfies $0 \leq P _ { \mathbf { x } } ^ { \ell } ( y ) \leq P _ { \mathbf { x } } ^ { * } ( y ) \leq$ $\operatorname* { m i n } \{ 1 , P _ { \mathbf { x } } ^ { u } ( y ) \}$ . Multiplying by $y \geq 0$ and summing over y gives valid bounds on $\mu _ { \mathbf { x } } ^ { * }$ . Taking the maximum of these lower bounds and the minimum of these upper bounds, starting from [0, 1], preserves validity. A proper resulting subinterval is informative by the definition of partial transportability. □

Computational cost. The cost includes graph operations, expression generation, and evaluation of source probabilities. Enumeration can be large: if each of d components has z candidate expressions, their product can yield $z ^ { d }$ combinations. Each Rule 3 branch also visits $| \Omega _ { \mathbf { V } ^ { \prime } } |$ assignments; these finite-domain branching factors compound across recursive calls. The total cost must therefore account for the number and size of returned expressions, as well as the assignments summed over in their evaluation. Source quantities and repeated subqueries can be reused. These computations occur before target interaction; once the bounds are available, the online policy uses their numerical endpoints.

## 5.4 Dominance Bounds

Causal bounds constrain individual rewards. The graph also relates the best rewards attainable from diferent action spaces. We say that one action space dominates another when its maximum expected reward is at least as large. This comparison concerns the maxima over the two spaces, rather than every pair of their actions.

## 5.4.1 Lower Dominance Bound

Under the target constraint $\mathbf { N } ^ { * }$ , the POMIS action space attains at least the reward of every feasible MIS action. Therefore, any lower bound for a feasible MIS action also lower-bounds the target optimum. Define:

$$
\begin{array} { r } { \boldsymbol { \ell } ^ { \star } = \underset { \mathbf { W } \in \mathbb { M } _ { \mathcal { G } , Y } ^ { \star } } { \operatorname* { m a x } } \ell _ { \mathbf { W } } . } \\ { \mathbf { W } \ell _ { \mathbf { W } } ^ { \star } } \end{array}\tag{134}
$$

If $\mathbf { w } ^ { \dagger }$ attains this maximum, then:

$$
\ell ^ { \star } = \ell _ { \mathbf { w } ^ { \dagger } } \leq \mu _ { \mathbf { w } ^ { \dagger } } ^ { * } \leq \mu _ { \mathbf { x } ^ { \star } } ^ { * } .\tag{135}
$$

The maximizing lower bound may come from an MIS that is not a POMIS. Its reward is still achievable, so it can certify that another action is suboptimal. In particular, $u _ { \mathbf { x } } < \ell ^ { \star }$ implies $\mu _ { \mathbf { x } } ^ { * } < \mu _ { \mathbf { x } } ^ { * }$ <sub>⋆</sub> and permits safe removal of x.

## 5.4.2 Upper Dominance Bound

A non-manipulable variable restricts the interventions available to the agent. Allowing an additional intervention cannot reduce the best attainable reward. The constrained POMIS construction makes this comparison explicit.

Proposition 12. A POMIS under ⟨G, Y, N⟩ is an MIS under ⟨G, Y ⟩.

Proof of Prop. 12. A constrained POMIS X is a minimal feasible intervention set. If it were not an unconstrained MIS, a proper subset of X would be interventionally equivalent for Y in every compatible SCM. That subset is also feasible because $\mathbf { X } \cap \mathbf { N } = \emptyset$ , contradicting constrained minimality. □

A constrained POMIS can nevertheless fail to be an unconstrained POMIS: an intervention that dominates it may become available when the constraint is removed.

Example 10 (Continued). Return to Example 9, with $\mathbf { N } ^ { * } = \{ W \}$ . Its constrained POMISs are:

$$
\mathbb { P } _ { \mathcal { G } , Y } ^ { \{ W \} } = \{ \{ X _ { 1 } \} , \{ X _ { 1 } , X _ { 2 } \} \} .\tag{136}
$$

Thus, an optimal feasible treatment can be selected among interventions on $X _ { 1 }$ alone or on both treatments. Now consider the unconstrained intervention set $\mathbf { R } = \{ W , X _ { 2 } \}$ . Fixing both direct causes of Y can match or improve the mean attained by any distribution over their natural values. For the best intervention on $X _ { 1 }$ :

$$
\mu _ { x _ { 1 } ^ { * } } ^ { * } = \sum _ { \mathbf { r } } \mathbb { E } _ { P _ { x _ { 1 } ^ { * } } ^ { * } } [ Y \mid \mathbf { r } ] P _ { x _ { 1 } ^ { * } } ^ { * } ( \mathbf { r } )\tag{137}
$$

$$
\stackrel { \mathrm { ( a ) } } { = } \sum _ { \mathbf { r } } \mathbb { E } _ { P _ { x _ { 1 } ^ { * } , \mathbf { r } } ^ { * } } [ Y ] P _ { x _ { 1 } ^ { * } } ^ { * } ( \mathbf { r } )
$$

Rule 2

$$
\underline { { \underline { { \mathbf { \Pi } } } } } \equiv \sum _ { \mathbf { r } } \mu _ { \mathbf { r } } ^ { * } P _ { x _ { 1 } ^ { * } } ^ { * } ( \mathbf { r } )\tag{138}
$$

Rule 3

(139)

$$
\le \mu _ { \mathbf { r } ^ { * } } ^ { * } .\tag{140}
$$

Step (a) uses Rule 2, since the relevant paths from $W ,$ $X _ { 2 }$ to $Y$ are unconfounded after fixing $X _ { 1 } . ~ \mathrm { S t e p ~ ( b ) }$ uses Rule 3: once W is fixed, $X _ { 1 }$ no longer afects Y . The last inequality bounds an average by its largest term.

The same argument applies to an intervention on $\{ X _ { 1 } , X _ { 2 } \}$ , averaging only over W. Consequently, the best reward under the target constraint is no greater than the best reward under $d o ( w , x _ { 2 } )$ . The latter interventions remain unavailable to the target agent, but an upper bound on their rewards can still constrain every feasible treatment.

Theorem 11. Let $\mathbf { N } \subsetneq \mathbf { N } ^ { * }$ , and let $\mathbf { r } ^ { \star }$ be optimal in $\mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } }$ . Then $\mu _ { \mathbf { x } ^ { \star } } ^ { * } \leq \mu _ { \mathbf { r } ^ { \star } } ^ { * }$

Proof of Thm. 11. Every intervention feasible under $\mathbf { N } ^ { * }$ is feasible under $\mathbf { N } \subsetneq \mathbf { N ^ { * } }$ . The maximum target mean over all feasible interventions therefore cannot decrease when the constraint is relaxed. POMIS reduction preserves each maximum, giving $\mu _ { \mathbf { x } ^ { \star } } ^ { * } \leq \mu _ { \mathbf { r } ^ { \star } } ^ { * }$ □

The optimal action $\mathbf { r } ^ { \star }$ under the weaker constraint is unknown. To obtain a computable upper bound, maximize the available upper bounds over all POMIS actions under the weaker constraint:

$$
\mu _ { \mathbf { x } ^ { \star } } ^ { * } \leq \operatorname* { m a x } _ { \mathbf { r } \in \mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } } } \mu _ { \mathbf { r } } ^ { * } \leq \operatorname* { m a x } _ { \mathbf { r } \in \mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } } } u _ { \mathbf { r } } .\tag{141}
$$

Combining these bounds, including the original constraint, gives:

$$
u ^ { \star } = \operatorname* { m i n } _ { { \bf N } \subseteq { \bf N } ^ { \ast } } \operatorname* { m a x } _ { { \bf r } \in \mathcal { A } _ { \mathcal { G } , Y } ^ { { \bf V } } } u _ { \bf r } .\tag{142}
$$

Every reward in this calculation is a target reward. Source experiments are used to bound those rewards; the optimum of a source environment is not itself a bound on the target optimum.

```latex
Algorithm 7: Upper dominance bound by traversing weaker constraints.
1 function udb(G, Y, N, L, U)
// Share stored results across recursive calls.
2 Return a stored result for N if available; otherwise store each return value below.
// Construct the POMIS action space for the current constraint.
3 Set $\mathcal { H } = \mathcal { G } \langle \mathbf { V } \mid \mathbf { N } \rangle$
4 Set $\mathcal { A } _ { \mathcal { H } , Y } = \{ \mathbf { r } : \mathbf { R } \in \mathbb { P } _ { \mathcal { H } , Y } , \ \mathbf { r } \in \Omega _ { \mathbf { R } } \}$
5 Choose $\mathbf { r } ^ { \dagger } \in \arg \operatorname* { m a x } _ { \mathbf { r } \in \mathcal { A } _ { \mathcal { H } , Y } } u _ { \mathbf { r } }$ preferring larger lower bounds in a tie.
6 Set $u ^ { \star } = u _ { \mathbf { r } ^ { \dagger } }$
// An exact maximizing bound determines this constrained optimum.
7 if $u _ { \mathbf { r } ^ { \dagger } } = \ell _ { \mathbf { r } ^ { \dagger } }$ then return $u ^ { \star }$
8 if $\mathbf { N } = \varnothing$ then return $u ^ { \star }$
// Every weaker constraint supplies another valid upper bound.
9 for each $\mathbf { \bar { N } } _ { \neq } ^ { \prime } \mathbf { N }$ with $| \mathbf { N } ^ { \prime } | = | \mathbf { N } | - \mathbf { \bar { \alpha } } _ { }$ do
10 Set $u ^ { \star } \gets \operatorname* { m i n } \{ u ^ { \star } , \operatorname { U D B } ( \mathcal { G } , Y , \mathbf { N } ^ { \prime } , \mathbb { L } , \mathbb { U } ) \}$
11 return $u ^ { \star }$
```

## 5.4.3 Traversing the Constraint Hierarchy

Removing one variable from N gives an immediately weaker constraint. Repeating this step reaches every subset of N. Exact optimal rewards increase weakly along these paths, but their computed upper bounds need not: source experiments can constrain a hypothetical intervention more tightly than the original feasible actions.

udb in Alg. 7 takes G, Y, the current non-manipulable set N, and valid target mean bounds $\mathbb { L } = \{ \boldsymbol { \ell } _ { \mathbf { r } } \}$ and $\mathbb { U } = \{ u _ { \mathbf { r } } \}$ for the actions queried during recursion. It returns an upper bound on the best target reward under N. The diagram, reward, and causal bounds remain fixed, and an initially empty collection of stored results is shared across the recursive calls.

The lines of Alg. 7 have the following roles:

• Construct the current candidates (lines 2–6). Reuse a stored result if this constraint was visited before. Otherwise, project out N, enumerate its POMIS actions, and select an action $\mathbf { r } ^ { \dagger }$ with the largest upper bound. Ties are resolved by preferring the larger lower bound, with a fixed rule for any remaining tie. Its upper endpoint supplies the local candidate.

• Stop when no improvement is possible (lines 7–8). If the maximizing action has $\ell _ { \mathbf { r } ^ { \dagger } } = u _ { \mathbf { r } ^ { \dagger } }$ , the exact constrained optimum is known. An exact interval for another action does not sufice. At N = ∅, no constraint remains to relax.

• Visit weaker constraints (lines 9–11). Remove one non-manipulable variable at a time and take the minimum of the resulting bounds and the local candidate. Store the returned value for later visits.

Corollary 4. udb returns the bound in Eq. (142); in particular, $\mu _ { \mathbf { x } ^ { \star } } ^ { * } \leq u ^ { \star }$

Proof of Cor. 4. At a constraint N, the local value $\operatorname* { m a x } _ { \mathbf { r } \in \mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } } } u _ { \mathbf { r } }$ bounds its constrained optimum. Every recursive value bounds an optimum under a weaker constraint, which is at least the current optimum by Thm. 11. Taking their minimum is therefore valid. Removing one variable recursively visits every proper subset, so induction on $| \mathbf { N } |$ gives the subsetminimum in Eq. (142). For an early return, the maximizing action $\mathbf { r } ^ { \dagger }$ satisfies $\ell _ { \mathbf { r } ^ { \dagger } } = u _ { \mathbf { r } ^ { \dagger } }$ , so its known mean equals the local maximum upper bound and hence the exact constrained optimum. Every bound at a weaker constraint is at least this value; omitting those calls therefore leaves the subset-minimum unchanged. □

![](images/7d0488c3693ea46b5871de9dae40d309951110df5bf307d054aab0114ca068dc.jpg)  
Figure 24: Collective selection diagrams for the worked examples. Gray circles mark variables that are non-manipulable in the target. Red and blue squares denote source 1 and source 2 selections, respectively: (a) $\Delta ^ { 1 } = \{ A \} ; ( \mathrm { b } ) \ \bar { \Delta } ^ { 1 } = \{ A \}$ and $\Delta ^ { 2 } = \{ B \}$ ; (c) $\Delta ^ { 1 } = \{ T \}$ and $\Delta ^ { 2 } = \{ R \}$

The same constraint can be reached by removing variables in diferent orders. Reusing its result limits the traversal to at most $2 ^ { | \mathbf { N } ^ { * } | }$ distinct constraints and $| \mathbf { N } ^ { * } | 2 ^ { | \mathbf { N } ^ { * } | - 1 }$ one-variable relaxations when $\mathbf { N } ^ { * } \neq \varnothing$ . The cost at each constraint additionally includes latent projection, POMIS enumeration, and any causal bounds not already computed. Early termination can reduce this cost.

Combining the bounds. The lower dominance bound certifies an achievable target reward; the upper dominance bound certifies a ceiling on the target optimum. Together with an action’s causal bounds, they provide two ways to reduce exploration. If $u _ { \mathbf { x } } < \ell ^ { \star }$ , the action can be removed. Otherwise, its reward still satisfies:

$$
\ell _ { \mathbf { x } } \leq \mu _ { \mathbf { x } } ^ { * } \leq \operatorname* { m i n } \{ u _ { \mathbf { x } } , u ^ { \star } \} .\tag{143}
$$

The upper dominance bound can therefore improve the interval even when no additional action can be pruned.

## 5.5 Worked Examples of Bound Construction

Every endogenous and exogenous variable is binary, and the exogenous variables within each SCM are mutually independent. For these worked examples, the source environments use the displayed structural functions and exogenous distributions, and their distributions are known at the population level. The discrepancy sets specify mechanisms that are allowed to difer, without requiring an actual change. The agent uses the collective selection diagram and available source distributions, while the displayed target means serve only to check the bounds. Numerical results are rounded to four decimal places; all comparisons use unrounded values.

Table 4: Bounds for all six initial POMIS actions in Fig. 24a. The lower dominance bound is $\ell ^ { \star } = 0 . 8 3 2 5$
<table><tr><td>Action</td><td> $\ell _ { \mathbf { x } }$ </td><td> $\mu _ { \mathbf { x } } ^ { * }$ </td><td> $u _ { \mathbf { x } }$ </td><td>Decision</td></tr><tr><td> $d o ( B = 0 )$ </td><td>0.0000</td><td>1.0000</td><td>1.0000</td><td>Retain</td></tr><tr><td> $d o ( B = 1 )$ </td><td>0.0000</td><td>0.5900</td><td>1.0000</td><td>Retain</td></tr><tr><td> $d o ( A = 0 , B = 0 )$ </td><td>0.1675</td><td>1.0000</td><td>1.0000</td><td>Retain</td></tr><tr><td> $d o ( A = 0 , B = 1 )$ </td><td>0.2965</td><td>0.5450</td><td>0.7940</td><td>Remove</td></tr><tr><td> $d o ( A = 1 , B = 0 )$ </td><td>0.8325</td><td>1.0000</td><td>1.0000</td><td>Retain</td></tr><tr><td> $d o ( A = 1 , B = 1 )$ </td><td>0.2935</td><td>0.5900</td><td>0.7960</td><td>Remove</td></tr></table>

Eliminating actions before interaction. Consider the diagram in Fig. 24a and the following SCM. Let $\mathbf { V } = \{ A , B , Y \}$ . The exogenous variables are:

$$
\mathbf { U } = \{ U _ { A } , U _ { B } , U _ { Y } , U _ { A Y } \} .\tag{144}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } \lor ( b \oplus u _ { A Y } ) } \\ & { f _ { B } = u _ { B } } \\ & { f _ { Y } = ( 1 - b ) \lor ( ( u _ { A Y } \lor a ) \oplus u _ { Y } ) . } \end{array}\tag{145}
$$

The exogenous distributions are:

$$
\begin{array} { c c } { { U _ { A } \sim \tt B e r n ( 0 . 3 3 ) } } & { { U _ { B } \sim \tt B e r n ( 0 . 3 8 ) } } \\ { { U _ { Y } \sim \tt B e r n ( 0 . 4 1 ) } } & { { U _ { A Y } \sim \tt B e r n ( 0 . 7 5 ) . } } \end{array}\tag{146}
$$

Here $\mathbf { N ^ { * } } = \varnothing , \Delta ^ { 1 } = \{ A \}$ , and $\mathbb { Z } ^ { 1 } = \{ \{ B \} \}$ . The constrained POMIS collection is $\mathbb { P } _ { \mathcal { G } , Y } ^ { \varnothing } =$ $\{ \{ B \} , \{ A , B \} \}$ , giving six initial actions.

For $d o ( B = b )$ , the remaining c-component $\{ A , Y \}$ intersects the discrepancy set at A, so paTR supplies no informative bound and retains $[ \ell _ { b } , u _ { b } ] = [ 0 , 1 ]$ . For $d o ( A = a , B = b )$ , the mechanism at A is replaced by intervention. Applying the natural bounds within the source experiment on B gives:

$$
\ell _ { a , b } = P _ { b } ^ { 1 } ( a , Y = 1 ) , \qquad \quad u _ { a , b } = P _ { b } ^ { 1 } ( a , Y = 1 ) + 1 - P _ { b } ^ { 1 } ( a ) .\tag{147}
$$

Table 4 gives all six intervals. The largest feasible lower bound is $\ell ^ { \star } = \ell _ { A = 1 , B = 0 } = 0 . 8 3 2 5$ while $u ^ { \star } = 1$ . The two actions $d o ( A = 0 , B = 1 )$ and $d o ( A = 1 , B = 1 )$ satisfy $u _ { \mathbf { x } } < \ell ^ { \star }$ and are removed. The four retained actions are $d o ( B = 0 ) , d o ( B = 1 ) , d o ( A = 0 , B = 0 )$ , and $d o ( A = 1 , B = 0 )$ . For each retained action, min $\{ u _ { { \bf x } } , u ^ { \star } \} = 1$ , so the upper dominance bound does not further change its interval.

Clipping bounds through weaker constraints. Consider the diagram in Fig. 24b and the following SCM. Let $\mathbf { V } = \{ A , B , C , Y \}$ . The exogenous variables are:

$$
{ \bf { U } } = \{ U _ { A } , U _ { B } , U _ { C } , U _ { Y } , U _ { B C } , U _ { B Y } \} .\tag{148}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } } \\ & { f _ { B } = \left( u _ { B Y } \wedge u _ { B C } \right) \oplus u _ { B } } \\ & { f _ { C } = b \oplus \left( u _ { B C } \wedge \left( \left( 1 - u _ { C } \right) \oplus a \right) \right) } \\ & { f _ { Y } = c \oplus \left( u _ { B Y } \wedge \left( \left( 1 - u _ { Y } \right) \oplus a \right) \right) . } \end{array}\tag{149}
$$

The exogenous distributions are:

$$
\begin{array} { r l r } & { U _ { A } \sim \mathtt { B e r n ( 0 . 2 8 ) } } & { U _ { B } \sim \mathtt { B e r n ( 0 . 4 2 ) } } & { U _ { C } \sim \mathtt { B e r n ( 0 . 5 2 ) } } \\ & { U _ { Y } \sim \mathtt { B e r n ( 0 . 4 7 ) } } & { U _ { B C } \sim \mathtt { B e r n ( 0 . 6 2 ) } } & { U _ { B Y } \sim \mathtt { B e r n ( 0 . 4 9 ) } . } \end{array}\tag{150}
$$

The target constraint is $\mathbf { N } ^ { * } = \{ A , C \}$ . Source 1 supplies observational data, with $\Delta ^ { 1 } = \{ A \}$ and $\mathbb { Z } ^ { 1 } = \{ \emptyset \}$ ; source 2 supplies experiments on $C ,$ with $\Delta ^ { 2 } = \{ B \}$ and $\mathbb { Z } ^ { 2 } = \{ \{ C \} \}$ . The initial POMIS collection is $\{ \emptyset , \{ B \} \}$ , giving three target actions.

For any fixed experimental assignment $c ^ { \prime } ,$ the observational action is exactly transportable:

$$
\mu _ { \emptyset } ^ { * } = \sum _ { a , b , c , y } y P _ { c ^ { \prime } } ^ { 2 } ( a ) P _ { \emptyset } ^ { 1 } ( b , c , y \mid a ) \approx 0 . 4 8 4 4 .\tag{151}
$$

For $d o ( B = b )$ , the factor $Q ^ { * } [ C ]$ is not identified from the available sources. Its natural bounds combine with the exact factors for A and $Y :$

$$
\ell _ { b } = \sum _ { a , c } P _ { c ^ { \prime } } ^ { 2 } ( a ) P _ { \emptyset } ^ { 1 } ( b , c \mid a ) P _ { c } ^ { 2 } ( Y = 1 \mid a ) ,\tag{152}
$$

$$
u _ { b } = \operatorname* { m i n } \Biggl \{ 1 , \sum _ { a , c } P _ { c ^ { \prime } } ^ { 2 } ( a ) P _ { c } ^ { 2 } ( Y = 1 \mid a ) \bigl [ P _ { \emptyset } ^ { 1 } ( b , c \mid a ) + 1 - P _ { \emptyset } ^ { 1 } ( b \mid a ) \bigr ] \Biggr \} .\tag{153}
$$

Thus $[ \ell _ { 0 } , u _ { 0 } ] \approx [ 0 . 2 0 9 7 , 0 . 6 7 8 3 ]$ and $[ \ell _ { 1 } , u _ { 1 } ] \approx [ 0 . 2 7 5 2 , 0 . 8 0 6 6 ]$ . The lower dominance bound is the exact observational mean, $\ell ^ { \star } \approx 0 . 4 8 4 4$ , and every initial action satisfies $u _ { \mathbf { x } } \geq \ell ^ { \star }$

The upper dominance calculation visits the four constraints in Table $6 ;$ their relations are shown in Fig. 25. At $\mathbf { N } = \{ A \}$ , the exact action $d o ( C = 1 )$ has mean approximately 0.7485. However, $d o ( B = 1 )$ has the larger upper bound approximately 0.8066, and its interval does not collapse. Hence the presence of an exact action does not justify early termination. At $\mathbf { N } = \{ C \}$ , the largest upper bound is approximately 0.8070, attained by $d o ( A = 0 , B = 1 )$ , whose interval also does not collapse. Both branches reach $\mathbf { N } = \varnothing$ , where:

$$
\mu _ { A = 1 , C = 1 } ^ { * } = P _ { C = 1 } ^ { 2 } ( Y = 1 \mid A = 1 ) = 0 . 7 6 9 7 .\tag{154}
$$

This action has the largest upper bound among the unconstrained POMIS actions, and its lower and upper bounds agree. Therefore the call returns 0.7697; the other branch reuses this stored result. The final upper dominance bound is:

$$
u ^ { \star } = \mu _ { A = 1 , C = 1 } ^ { * } = 0 . 7 6 9 7 .\tag{155}
$$

All three target actions remain, but the upper limit for $d o ( B = 1 )$ decreases from approximately 0.8066 to 0.7697, as shown in Table 5. The true best target mean is $\mu _ { B = 1 } ^ { * } \approx 0 . 5 9 6 9$ the intervention $d o ( A = 1 , C = 1 )$ is used only to bound the target optimum because A and C cannot be manipulated in the target.

![](images/6b6daaa1b7d78d2bafc1922b2cb598b235f3ca863f437319993096d6be70e08f.jpg)  
Figure 25: Dominance among POMIS action spaces for the graph in Fig. 24b. Each arrow points toward a weaker manipulability constraint, whose action space has an optimal expected reward at least as large as that of its predecessor. The labels (a)–(d) identify the four constraint settings.

Table 5: The three target actions in Fig. 24b. All are retained. The dominance bound reduces the upper limit for $d o ( B = 1 )$ to $u ^ { \star } = 0 . 7 6 9 7$
<table><tr><td>Action</td><td> $\ell _ { \mathbf { x } }$ </td><td> $\mu _ { \mathbf { x } } ^ { * }$ </td><td> $u _ { \mathbf { x } }$ </td><td> $b _ { \mathbf { x } } = \operatorname* { m i n } \{ u _ { \mathbf { x } } , u ^ { \star } \}$ </td></tr><tr><td> $d o ( \emptyset )$ </td><td>0.4844</td><td>0.4844</td><td>0.4844</td><td>0.4844</td></tr><tr><td> $d o ( B = 0 )$ </td><td>0.2097</td><td>0.4031</td><td>0.6783</td><td>0.6783</td></tr><tr><td> $d o ( B = 1 )$ </td><td>0.2752</td><td>0.5969</td><td>0.8066</td><td>0.7697</td></tr></table>

An exact optimal value from source experiments. Consider the diagram in Fig. 24c and the following SCM. Let $\mathbf { V } = \{ R , T , W , X , Z , Y \}$ . The exogenous variables are:

$$
\mathbf { U } = \{ U _ { V } : V \in \mathbf { V } \} \cup \{ U _ { R W } , U _ { R Y } , U _ { X Y } , U _ { W X } , U _ { W Z } \} .\tag{156}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { R } = u _ { R } \vee \left( ( 1 - u _ { R W } ) \wedge ( t \wedge u _ { R Y } ) \right) } \\ & { f _ { T } = u _ { T } } \\ & { f _ { W } = u _ { W } \vee \left( ( 1 - u _ { W X } ) \wedge ( u _ { R W } \wedge u _ { W Z } ) \right) } \\ & { f _ { X } = w \vee \left( ( 1 - u _ { W X } ) \wedge ( u _ { X Y } \wedge u _ { X } ) \right) } \\ & { f _ { Z } = x \vee \left( ( 1 - u _ { W Z } ) \wedge ( r \wedge u _ { Z } ) \right) } \\ & { f _ { Y } = z \vee \left( ( 1 - u _ { X Y } ) \wedge ( u _ { R Y } \wedge u _ { Y } ) \right) . } \end{array}\tag{157}
$$

The exogenous distributions are:

$$
\begin{array} { r l r } { U _ { R } \sim \mathtt { B e r n } ( 0 . 5 3 ) } & { { } \ U _ { T } \sim \mathtt { B e r n } ( 0 . 6 3 ) } & { \ U _ { W } \sim \mathtt { B e r n } ( 0 . 3 8 ) } \\ { U _ { X } \sim \mathtt { B e r n } ( 0 . 2 7 ) } & { { } \ U _ { Z } \sim \mathtt { B e r n } ( 0 . 4 0 ) } & { \ U _ { Y } \sim \mathtt { B e r n } ( 0 . 2 6 ) } \\ { U _ { R W } \sim \mathtt { B e r n } ( 0 . 5 2 ) } & { { } \ U _ { R Y } \sim \mathtt { B e r n } ( 0 . 6 3 ) } & { \ U _ { X Y } \sim \mathtt { B e r n } ( 0 . 7 9 ) } \\ { U _ { W X } \sim \mathtt { B e r n } ( 0 . 7 4 ) } & { { } \ U _ { W Z } \sim \mathtt { B e r n } ( 0 . 3 1 ) . } \end{array}\tag{158}
$$

Let $\mathbf { N } ^ { * } = \{ T , W \} , { \Delta } ^ { 1 } = \{ T \} , { \Delta } ^ { 2 } = \{ R \} , \mathbb { Z } ^ { 1 } = \{ \varnothing , \{ Z \} \}$ , and $\mathbb { Z } ^ { 2 } = \{ \{ Z \} \}$ . The initial POMIS collection is $\{ \emptyset , \{ R \} , \{ X \} , \{ Z \} \}$ , giving seven actions.

The observational mean and both interventions on $Z$ are exactly transportable:

$$
\mu _ { \emptyset } ^ { * } = \sum _ { \substack { r , t , w , x , z , y } } y P _ { z ^ { \prime } } ^ { 2 } ( t ) P _ { \emptyset } ^ { 1 } ( r , w , x , z , y \mid t ) \approx 0 . 5 5 1 4 ,\tag{159}
$$

Table 6: The four distinct constraints visited by udb in Fig. 24b. Both one-variable constraints share the stored result for $\mathbf { N } = \varnothing$ . The local upper bound is maximized over the displayed POMIS actions.
<table><tr><td>N</td><td> $\mathbb { P } _ { \mathcal { G } , Y } ^ { \mathbf { N } }$ </td><td>Local upper bound</td><td>Returned bound</td></tr><tr><td> $\{ A , C \}$ </td><td> $\{ \emptyset , \{ B \} \}$ </td><td>0.8066</td><td>0.7697</td></tr><tr><td>{A}</td><td> $\{ \varnothing , \hat { \{ B \} } , \{ C \} \}$ </td><td>0.8066</td><td>0.7697</td></tr><tr><td>{c}</td><td> $\{ \{ A \} , \{ A , B \} \}$ </td><td>0.8070</td><td>0.7697</td></tr><tr><td>0</td><td> $\{ \{ A \} , \{ A , C \} \}$ </td><td>0.7697</td><td>0.7697</td></tr></table>

Table 7: Bounds for the seven initial POMIS actions in Fig. 24c. The exact source value for $d o ( Z = 1 )$ gives $\ell ^ { \star } = u ^ { \star } = 1$ , so three actions are removed.
<table><tr><td>Action</td><td> $\ell _ { \mathbf { x } }$ </td><td> $\mu _ { \mathbf { x } } ^ { * }$ </td><td> $u _ { \mathbf { x } }$ </td><td>Decision</td></tr><tr><td>do(∅)</td><td>0.5514</td><td>0.5514</td><td>0.5514</td><td>Remove</td></tr><tr><td> $d o ( R = 0 )$ </td><td>0.2140</td><td>0.4553</td><td>1.0000</td><td>Retain</td></tr><tr><td> $d o ( R = 1 )$ </td><td>0.4092</td><td>0.6110</td><td>1.0000</td><td>Retain</td></tr><tr><td>do(X = 0)</td><td>0.0713</td><td>0.1990</td><td>0.7901</td><td>Remove</td></tr><tr><td> $d o ( X = 1 )$ </td><td>0.2176</td><td>1.0000</td><td>1.0000</td><td>Retain</td></tr><tr><td>do(Z = 0)</td><td>0.0344</td><td>0.0344</td><td>0.0344</td><td>Remove</td></tr><tr><td> $d o ( Z = 1 )$ </td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>Retain</td></tr></table>

$$
\mu _ { z } ^ { * } = P _ { z } ^ { 1 } ( Y = 1 ) = P _ { z } ^ { 2 } ( Y = 1 ) .\tag{160}
$$

Here $z ^ { \prime }$ is any fixed source assignment. The joint source factor also has the form $P _ { \emptyset } ^ { 1 } ( r , w , x , z , y \mid t ) = P _ { \emptyset } ^ { 1 } ( w ) P _ { \emptyset } ^ { 1 } ( r , x , z , y \mid t , w )$ , since T and W are independent in the diagram.

For $d o ( X = x )$ , two required factors are identified:

$$
Q ^ { * } [ T ] = P _ { z ^ { \prime } } ^ { 2 } ( t ) , \ ~ \ ~ \ Q ^ { * } [ R , Y ] = P _ { z } ^ { 1 } ( r , y \mid t ) .\tag{161}
$$

The remaining factor $Q ^ { * } [ Z ]$ is bounded, for each fixed $w ^ { \prime } .$ , by

$$
P _ { \emptyset } ^ { 1 } ( r , w ^ { \prime } , x , z \mid t ) \le Q ^ { * } [ Z ] \le P _ { \emptyset } ^ { 1 } ( r , w ^ { \prime } , x , z \mid t ) + 1 - P _ { \emptyset } ^ { 1 } ( r , w ^ { \prime } , x \mid t ) .\tag{162}
$$

Multiply these bounds by the two exact factors, sum over $( R , T , Z )$ with $Y = 1$ , and take the largest lower bound and smallest upper bound over $w ^ { \prime } \in \{ 0 , 1 \}$ , as enumerated by the Rule 3 loop of paTR. The target reward under $d o ( X = x )$ is unchanged by additionally fixing $W ;$ the natural bounds may difer because they use diferent slices of the source distribution. The resulting intervals appear in Table 7. This calculation illustrates the component-product step of paTR: source 2 supplies T, while source 1 supplies the remaining exact factor and a bound on Z.

For do $( R = r )$ , the natural-bound step is applied to:

$$
Q ^ { 1 } [ R , W , X , Z , Y ] = P _ { \varnothing } ^ { 1 } ( r , w , x , z , y \mid t ) .\tag{163}
$$

For this query, the Rule 3 loop enumerates $t ^ { \prime } \in \{ 0 , 1 \}$ , yielding:

$$
\ell _ { r } = \operatorname* { m a x } _ { t ^ { \prime } } P _ { \varnothing } ^ { 1 } ( r , Y = 1 \mid t ^ { \prime } ) ,\tag{164}
$$

$$
u _ { r } = \operatorname* { m i n } \Big \{ 1 , \operatorname* { m i n } _ { t ^ { \prime } } \big [ P _ { \varnothing } ^ { 1 } ( r , Y = 1 \mid t ^ { \prime } ) + 8 \{ 1 - P _ { \varnothing } ^ { 1 } ( r \mid t ^ { \prime } ) \} \big ] \Big \} = 1 .\tag{165}
$$

The coeficient 8 comes from the eight assignments to $( W , X , Z )$ in the outer sum. The inner marginal in the pointwise upper bound sums over all of $( W , X , Z , Y )$

Table 7 lists the resulting intervals. The source experiment $d o ( Z = 1 )$ identifies a mean of one, so $\ell ^ { \star } = u ^ { \star } = 1$ . Among actions tied at the largest upper bound, the tie rule prefers the larger lower bound and selects $d o ( Z = 1 )$ . Its interval collapses, so udb returns at the initial constraint. The actions $d o ( \emptyset )$ ， $d o ( X = 0 )$ , and $d o ( Z = 0 )$ have upper bounds below one and are removed. The four retained actions are do $( R = 0 )$ $d o ( R = 1 )$ $d o ( X = 1 )$ , and $d o ( Z = 1 )$ , all with clipped upper limit one. Thus source information identifies the optimal target value even though several other action means remain only partially transportable.

## 5.6 Online Learning with Transferred Bounds

We now use the bounds in a UCB policy. The clipping principle (Zhang and Bareinboim, 2017; Gong et al., 2023) limits an optimistic empirical estimate by a valid prior upper bound. trUCB in Alg. 8 first computes the causal and dominance bounds, removes actions with $u _ { \mathbf { x } } < \ell ^ { \star }$ , and then interacts only through the retained action space $\ b { A } ^ { * }$

Its inputs are $\mathcal { G } , Y , \mathbf { N } ^ { * }$ , the source information $( \Delta , \mathbb { Z } , \mathcal { P } _ { \mathbb { Z } } ^ { \Pi } )$ , and an integer horizon $T \geq 1$ It produces the target actions $A _ { 1 } , \cdots , A _ { T }$ and records their rewards. The causal queries include feasible MIS actions used for $\ell ^ { \star }$ and POMIS actions under the weaker constraints visited by udb; their bounds can be computed when first needed and reused.

Clipped exploration. Let $\kappa = | { \mathcal { A } } ^ { * } |$ and suppose that the horizon $T$ is known. The analysis uses the stationary, independent reward draws from Sec. 3. For each retained action, set:

$$
b _ { \mathbf { x } } = \operatorname* { m i n } \{ u _ { \mathbf { x } } , u ^ { \star } \} ,
$$

$$
\sigma _ { \mathbf { x } } ^ { 2 } = \operatorname* { m a x } _ { \mu \in [ \ell _ { \mathbf { x } } , b _ { \mathbf { x } } ] } \mu ( 1 - \mu ) .\tag{166}
$$

Because $Y \in [ 0 , 1 ]$ , its variance is no greater than $\mu _ { \mathbf { x } } ^ { * } ( 1 - \mu _ { \mathbf { x } } ^ { * } )$ , and hence no greater than $\sigma _ { \mathbf { x } } ^ { 2 }$ . For example, an interval lying near zero or one gives a smaller variance bound than an interval containing $\frac { 1 } { 2 }$

Using the counts $N _ { t } ( \mathbf { x } )$ from Sec. 3, the empirical target mean after at least one pull is:

$$
\widehat { \mu } _ { \mathbf { x } , t } ^ { * } = \frac { 1 } { N _ { t } ( \mathbf { x } ) } \sum _ { s = 1 } ^ { t } Y _ { s } \mathbb { 1 } \{ A _ { s } = \mathbf { x } \} .\tag{167}
$$

Each retained action is initially played once. With $\zeta _ { T } = \log ( 2 \kappa T ^ { 2 } )$ , a Bernstein confidence radius gives the index:

$$
U _ { \mathbf { x } } ( t ) = \widehat { \mu } _ { \mathbf { x } , t - 1 } ^ { * } + \sqrt { \frac { 2 \sigma _ { \mathbf { x } } ^ { 2 } \zeta _ { T } } { N _ { t - 1 } ( \mathbf { x } ) } } + \frac { 2 \zeta _ { T } } { 3 N _ { t - 1 } ( \mathbf { x } ) } ,\tag{168}
$$

$$
\begin{array} { r } { \bar { U } _ { \mathbf { x } } ( t ) = \operatorname* { m i n } \{ U _ { \mathbf { x } } ( t ) , b _ { \mathbf { x } } \} . } \end{array}\tag{169}
$$

```latex
Algorithm 8: trUCB: online learning with transferred bounds.
// Compute source and structural bounds before interaction.
1 Initialize $\mathcal { A } ^ { \ast } = \mathcal { A } _ { \mathcal { G } , Y } ^ { \mathbf { N } ^ { \ast } } .$
2 Compute $[ \ell _ { \mathbf { x } } , u _ { \mathbf { x } } ]$ for the required actions using CausalBound.
3 Compute $u ^ { \star }$ using udb(G, Y, N<sup>∗</sup>, L, U).
4 Set $\begin{array} { r } { \ell ^ { \star } = \operatorname* { m a x } _ { \mathbf { W } \in \mathbb { M } _ { \mathcal { G } , Y } ^ { \mathbf { N } ^ { * } } , \mathbf { w } \in \Omega _ { \mathbf { W } } } \ell _ { \mathbf { W } } . } \end{array}$
// Remove actions that cannot attain the target optimum.
5 Remove every $\mathbf { x } \in \mathcal { A } ^ { * }$ with $u _ { \mathbf { x } } < \ell ^ { \star }$
6 Set $\kappa = | { \mathcal { A } } ^ { * } |$ and $\zeta _ { T } = \log ( 2 \kappa T ^ { 2 } )$
7 Set $b _ { \mathbf { x } } = \operatorname* { m i n } \{ u _ { \mathbf { x } } , u ^ { \star } \}$ and $\sigma _ { \mathbf { x } } ^ { 2 } = \operatorname* { m a x } _ { \mu \in [ \ell _ { \mathbf { x } } , b _ { \mathbf { x } } ] } \mu ( 1 - \mu )$ for each retained action.
// Initialize the empirical means using one pull per retained action.
8 Play each retained action once in a fixed order, recording its reward and stopping after T
total pulls.
9 for each round $t = \kappa + 1 , \cdot \cdot \cdot , T$ do
// Use only observations available before the current decision.
10 Set $\begin{array} { r } { U _ { \mathbf { x } } ( t ) = \hat { \mu } _ { \mathbf { x } , t - 1 } ^ { * } + \sqrt { \frac { 2 \sigma _ { \mathbf { x } } ^ { 2 } \zeta _ { T } } { N _ { t - 1 } ( \mathbf { x } ) } } + \frac { 2 \zeta _ { T } } { 3 N _ { t - 1 } ( \mathbf { x } ) } } \end{array}$ for each $\mathbf { x } \in \mathcal { A } ^ { * }$
11 Set $\begin{array} { r } { \bar { U } _ { \mathbf { x } } ( t ) = \operatorname* { m i n } \{ U _ { \mathbf { x } } ( t ) , b _ { \mathbf { x } } \} . } \end{array}$
12 Choose $A _ { t } \in$ arg ma $\mathbf { \boldsymbol { x } _ { x \in \mathcal { A } ^ { * } } } \ \bar { U } _ { \mathbf { x } } ( t )$ using a fixed tie-breaking rule.
13 Intervene with $d o ( A _ { t } )$ and observe $Y _ { t } \sim P _ { A _ { t } } ^ { * } ( Y )$
14 Update the count and empirical reward mean of $A _ { t } .$
```

The square-root term uses the variance information supplied by the bounds. The linear term ensures the confidence guarantee for general bounded rewards. Both the count and the empirical mean use observations through round $t - 1$ , as required when selecting $A _ { t }$

Lines 1–5 perform the action reduction and bound construction before interaction. Lines 6–8 set the confidence parameter and variance bounds, then obtain one reward per retained action. At each subsequent round, lines 10–14 form the indices, clip them, select an action maximizing $\bar { U } _ { \mathbf { x } } ( t )$ , observe its target reward, and update only that action’s empirical mean and count. Directly evaluating the κ indices costs ${ \mathcal { O } } ( \kappa )$ per round; updating the selected mean and count costs $\mathcal { O } ( 1 )$ . Since $b _ { \mathbf { x } } ~ \geq ~ \mu _ { \mathbf { x } } ^ { * }$ , clipping preserves optimism whenever the empirical confidence bounds are valid. On the event that all confidence bounds hold, an action whose prior upper bound lies below the optimal mean need not be explored after initialization.

Regret guarantee. Assume valid population bounds and $T \geq \kappa$ . If the horizon is shorter, initialization stops after T pulls.

Theorem 12. The expected regret satisfies:

$$
\mathbb { E } R _ { T } \leq \sum _ { \mathbf { x } \in \mathcal { A } ^ { * } } \Delta _ { \mathbf { x } } + \zeta _ { T } \sum _ { \mathbf { x } \in \mathcal { A } ^ { * } \dot { \mathcal { \Delta } } _ { \mathbf { x } } > 0 } \left( \frac { 3 2 \sigma _ { \mathbf { x } } ^ { 2 } } { \Delta _ { \mathbf { x } } } + \frac { 8 } { 3 } \right) + 1 .\tag{170}
$$

The proof is given in Sec. B.1.2. The three terms have direct interpretations. The first accounts for playing each retained action once. The logarithmic term accounts for suboptimal actions whose upper bounds still permit optimality. The last term accounts for failure of the simultaneous confidence event, whose probability is at most $\textstyle { \frac { 1 } { T } }$ . In particular, initialization can incur regret even when no suboptimal action remains in the logarithmic sum.

More explicitly, actions removed before interaction are never selected. For a retained action with $b _ { \mathbf { x } } < \mu _ { \mathbf { x } ^ { \star } } ^ { * }$ , the expected number of pulls is at most two. For the other retained suboptimal actions, it is at most $\begin{array} { r } { 2 + \frac { 3 2 \sigma _ { \mathbf { x } } ^ { 2 } \zeta _ { T } } { \Delta _ { \mathbf { x } } ^ { 2 } } + \frac { 8 \zeta _ { T } } { 3 \Delta _ { \mathbf { x } } } } \end{array}$ . Thus, transferred bounds can remove actions entirely, prevent repeated exploration, or reduce the variance term for actions that remain competitive. Since $u ^ { \star } \geq \mu _ { \mathbf { x } ^ { \star } } ^ { * }$ , the condition $b _ { \mathbf { x } } \geq \mu _ { \mathbf { x } } ^ { * }$ <sub>⋆</sub> is equivalent to $u _ { \mathbf { x } } \geq \mu _ { \mathbf { x } ^ { \star } } ^ { * }$ . The dominance bound nevertheless afects the indices and $\sigma _ { \mathbf { x } } ^ { 2 } .$

## 5.7 Experiments

We evaluate the bounds and online policy on three binary SCMs. Tasks 1 and 3 use the models in the previous examples with Figs. 24a and 24c, respectively. Task 2 uses the diagram, source experiments, discrepancy sets, and target constraint of Fig. 24b, with the reward mechanism and exogenous probabilities specified below.

Task 2: upper dominance and interval variance. The agent cannot manipulate A or C in the target environment. Source 1 supplies observational data with $\Delta ^ { 1 } = \{ A \}$ , and source 2 supplies experiments on $C$ with $\Delta ^ { 2 } = \{ B \}$ . All exogenous variables are mutually independent. The SCM is specified as follows: Let $\mathbf { V } = \{ A , B , C , Y \}$ . The exogenous variables are:

$$
\mathbf { U } = \{ U _ { A } , U _ { B } , U _ { C } , U _ { Y } , U _ { B C } , U _ { B Y } , U _ { G } \} .\tag{171}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } } \\ & { f _ { B } = \left( u _ { B Y } \wedge u _ { B C } \right) \oplus u _ { B } } \\ & { f _ { C } = b \oplus \left( u _ { B C } \wedge \left( \left( 1 - u _ { C } \right) \oplus a \right) \right) } \\ & { f _ { Y } = u _ { G } \wedge \left[ \left( c \wedge \left( 1 - u _ { Y } \right) \right) \oplus \left( a \wedge u _ { B Y } \right) \right] . } \end{array}\tag{172}
$$

The exogenous distributions are:

$$
\begin{array} { r l r } & { U _ { A } \sim \mathtt { B e r n ( 0 . 8 ) } \quad \quad U _ { B } \sim \mathtt { B e r n ( 0 . 9 ) } } & { U _ { C } \sim \mathtt { B e r n ( 0 . 6 ) } } \\ & { U _ { Y } \sim \mathtt { B e r n ( 0 . 6 ) } } & { U _ { B C } \sim \mathtt { B e r n ( 0 . 1 ) } } & { U _ { B Y } \sim \mathtt { B e r n ( 0 . 4 ) } } \\ & { U _ { G } \sim \mathtt { B e r n ( 0 . 2 ) } . } \end{array}\tag{173}
$$

The additional private noise $U _ { G }$ enters only $f _ { Y }$ , so the causal diagram is unchanged. The three feasible actions are observation, $d o ( B = 0 )$ , and $d o ( B = 1 )$ , with $d o ( B = 1 )$ attaining the largest mean. Their source-derived intervals appear in Table 8.

For the hypothetical intervention d $\textstyle ( A = a , C = c )$ , source 2 identifies:

$$
\mu _ { A = a , C = c } ^ { * } = P _ { C = c } ^ { 2 } ( Y = 1 \mid A = a ) = 0 . 2 { \big ( } 0 . 4 c + 0 . 4 a - 0 . 3 2 a c { \big ) } .\tag{174}
$$

These means are 0, 0.08, 0.08, and 0.096 for $( a , c ) = ( 0 , 0 ) , ( 0 , 1 ) , ( 1 , 0 ) , ( 1 , 1 )$ , respectively. The other unconstrained POMIS, {A}, gives means approximately 0.0689 and 0.0931. The largest local upper bounds under the other constraints all exceed 0.096. Thus udb returns $u ^ { \star } = 0 . 0 9 6$ , reducing the upper limits of both interventions on B to this value. All three actions remain after pruning. Since their intervals lie below $\frac { 1 } { 2 }$ , the reduced upper limits also tighten the Bernoulli variance bounds in Eq. (166): from approximately 0.1241 to 0.0868 for $d o ( B = 0 )$ and from 0.0903 to 0.0868 for $d o ( B = 1 )$ . The larger reduction applies to the suboptimal action.

Table 8: Source bounds and target means for experimental Task 2. All three actions are retained, and $u ^ { \star } = 0 . 0 9 6$ . Values are rounded to four decimal places.
<table><tr><td>Action</td><td> $\ell _ { \mathbf { x } }$ </td><td> $\mu _ { \mathbf { x } } ^ { * }$ </td><td> $u _ { \mathbf { x } }$ </td><td> $b _ { \mathbf { x } }$ </td></tr><tr><td> $d o ( \emptyset )$ </td><td>0.0883</td><td>0.0883</td><td>0.0883</td><td>0.0883</td></tr><tr><td> $d o ( B = 0 )$ </td><td>0.0090</td><td>0.0654</td><td>0.1451</td><td>0.0960</td></tr><tr><td> $d o ( B = 1 )$ </td><td>0.0797</td><td>0.0914</td><td>0.1004</td><td>0.0960</td></tr></table>

Table 9: Candidate action spaces. Each strategy column gives the number of intervention sets $/$ arms. The retained action space is used by trUCB.
<table><tr><td>Task</td><td> $| \mathbf { V } |$ </td><td> $T$ </td><td>UCB</td><td>poUCB</td><td>trUCB</td></tr><tr><td>1</td><td>3</td><td>10k</td><td>4/9</td><td>2 / 6</td><td>2/4</td></tr><tr><td>2</td><td>4</td><td>10k</td><td>2/3</td><td> $2 ~ / ~ 3$ </td><td>2/3</td></tr><tr><td>3</td><td>6</td><td>10k</td><td>8/27</td><td>4/7</td><td>3/4</td></tr></table>

Source information and target interaction. The agent receives the causal diagram, the target constraint, and the permitted population source distributions. The source and target SCMs share the displayed numerical parameters in these tasks; the discrepancy sets specify which mechanisms may difer without requiring an actual change. Thus, these experiments study the use of source information under the asserted invariances. They do not assess uncertainty from finite source datasets. Table 9 reports the numbers of intervention sets and arms considered by the three main strategies in each task.

At every target round, the agent observes a fresh reward from its chosen intervention. We compute each target mean by summing over the SCM’s exogenous assignments and generate an independent Bernoulli reward with that mean. This gives the same reward distribution as simulating the full binary SCM. Exact target means are used only for reward generation and evaluation; the policy uses its permitted prior bounds and previously observed rewards. All methods are evaluated against the same optimum over target-feasible interventions: 1, approximately 0.0914, and 1 in Tasks 1–3. In particular, hypothetical interventions used by udb do not enlarge the agent’s feasible actions or change its regret benchmark.

Compared policies. We first compare UCB, poUCB, and trUCB. The first uses all assignments to all manipulable subsets, including the empty intervention. The second restricts these assignments to constrained POMISs. After playing each candidate action once, both use the standard UCB1 index (Auer et al., 2002), $\begin{array} { r } { \widehat { \mu } _ { \mathbf { x } , t - 1 } ^ { * } + \sqrt { \frac { 2 \log t } { N _ { t - 1 } ( \mathbf { x } ) } } } \end{array}$ . The third uses Alg. 8, including the Bernstein index in Eq. (169). These main comparisons therefore combine diferences in action selection and confidence rules.

Table 10: Final cumulative pseudo-regret at $T = 1 0 , 0 0 0$ (mean ± standard deviation across 1,000 independent runs). The lower block gives Bernstein-index ablations. Bold entries mark the lowest mean for each task.
<table><tr><td>Policy</td><td>Task 1</td><td>Task 2</td><td>Task 3</td></tr><tr><td>UCB</td><td> $2 4 7 . 1 { \pm } 1 9 . 6 $ </td><td> $6 7 . 1 { \pm } 5 . 9 $ </td><td> $2 4 8 . 6 { \pm } 1 1 . 3 $ </td></tr><tr><td>POUCB</td><td> $9 4 . 8 { \pm } 1 2 . 1 $ </td><td> $6 7 . 1 { \pm } 5 . 9 $ </td><td> $1 3 2 . 7 { \scriptstyle \pm 1 3 . 2 }$ </td></tr><tr><td>TRUCB</td><td> $\mathbf { 1 3 . 0 { \pm } 2 1 . 6 }$ </td><td> ${ \bf 5 5 . 4 } \pm 6 0 . 7 $ </td><td> $\mathbf { 3 2 . 2 \bot 3 5 . 2 }$ </td></tr><tr><td>Bernstein-POMIS</td><td> $3 8 . 1 { \pm } 4 3 . 5 $ </td><td> $6 3 . 8 { \pm } 6 . 7$ </td><td> $7 4 . 1 { \pm } 6 3 . 4 $ </td></tr><tr><td>Pruning only</td><td> $\mathbf { 1 3 . 0 { \pm } 2 1 . 6 }$ </td><td> $6 3 . 8 { \pm } 6 . 7$ </td><td> $\mathbf { 3 2 . 2 \bot 3 5 . 2 }$ </td></tr><tr><td> $\mathrm { P r u n i n g } + \mathrm { c l i p p i n g }$ </td><td> $\mathbf { 1 3 . 0 { \pm } 2 1 . 6 }$ </td><td> $1 2 1 . 7 { \scriptstyle \pm 1 2 8 . 1 }$ </td><td> $\mathbf { 3 2 . 2 \bot 3 5 . 2 }$ </td></tr><tr><td>TRUCB without UDB</td><td> $\mathbf { 1 3 . 0 { \pm } 2 1 . 6 }$ </td><td> $1 2 3 . 9 { \scriptstyle \pm 2 2 . 7 }$ </td><td> $\mathbf { 3 2 . 2 \bot 3 5 . 2 }$ </td></tr></table>

To distinguish the efects of the transferred bounds, we also compare four policies using the same Bernstein rule as trUCB. Bernstein-POMIS uses all POMIS actions with upper limit one and variance bound ${ \frac { 1 } { 4 } } .$ . Pruning only removes actions with $u _ { \mathbf { x } } < \ell ^ { \star }$ but otherwise keeps these trivial bounds. Pruning + clipping additionally uses $b _ { \mathbf { x } } = \operatorname* { m i n } \{ u _ { \mathbf { x } } , u ^ { \star } \}$ , while retaining variance bound $\textstyle { \frac { 1 } { 4 } }$ . Finally, trUCB without UDB uses the causal upper bound $u _ { \mathbf { x } }$ in both clipping and the variance interval, omitting only the upper dominance bound. Each policy sets κ to its number of retained actions when computing $\zeta _ { T }$

Evaluation. Every task–policy configuration is run for $T = 1 0 { , } 0 0 0$ rounds and repeated 1,000 times. Before interaction, each repetition draws an action order independently of the reward draws and keeps it fixed for initialization and index ties. All policies in that repetition use the same order restricted to their candidate actions and share the same sequence of independent uniform draws for generating rewards. The random sequences used for this evaluation were not used to choose the parameters of Task 2. We record cumulative pseudo-regret. Curves in Fig. 26 show its mean, and shaded regions show one population standard deviation across repetitions, matching the presentation in Sec. 4.3. Table 10 reports the final mean and standard deviation.

Performance. In Task 1, POMIS reduction decreases the number of actions from nine to six, and the source bounds remove two more. At $T = 1 0 { , } 0 0 0$ , the mean regret decreases from 247.1 for UCB to 94.8 for poUCB and 13.0 for trUCB. The controlled comparison isolates the additional efect of pruning: Bernstein-POMIS incurs mean regret 38.1, whereas Pruning only incurs 13.0. Since every retained upper limit is one, clipping does not further tighten these limits.

In Task 2, all methods retain the same three actions. Thus UCB and poUCB coincide, with mean regret 67.1, and pruning alone does not change the Bernstein policy, whose mean regret is 63.8. Full trUCB gives 55.4, compared with 123.9 without upper dominance. However, Pruning + clipping, which fixes the variance bound at $\textstyle { \frac { 1 } { 4 } }$ , gives 121.7. A common clipping level can produce index ties, so tightening a cap alone need not improve finite-horizon regret. The full policy also uses the smaller variance bounds supplied by the intervals. The additional comparison below isolates how upper dominance contributes through these two uses.

![](images/f8d3712251a8d529058e4503787630e08c08c2d56f46495a6637c1beeca9d1e2.jpg)  
(a) Task 1

![](images/f71d990d7a4dae0d53842eff7c883988b94c25953fe576a785ca805093693889.jpg)  
(b) Task 2

![](images/dfb708b4efc3518d10cac7d5b41ff5826239ccad1a90292557f9c1cbf4aae12f.jpg)  
(c) Task 3

Bernstein-POMIS Pruning only Pruning + clipping trUCB without UDB trUCB  
![](images/e256df4d8b68a568dc563900c568c9297a579cb694c8d7fd2974f744f7f0c907.jpg)  
(d) Task 1

![](images/3e6e58fefee9dccda2d15235e2cac31be99e8f3d41ad337d09b80df918177d35.jpg)  
(e) Task 2

![](images/71b9c4ea5972c0585a37c7eed509d3f425d06d83797daeb8d887abf6e31b3c9d.jpg)  
(f) Task 3  
Figure 26: Mean cumulative pseudo-regret over 10,000 rounds and 1,000 independent runs; shaded regions show one standard deviation. The upper row compares UCB and poUCB (standard UCB indices) with trUCB (Bernstein indices); the lower row compares controlled Bernstein-index ablations. UCB and poUCB overlap in Task 2. The curves for trUCB and trUCB without UDB overlap in Tasks 1 and 3.

In Task 3, the action counts decrease from 27 to 7 after POMIS reduction and to 4 after pruning. The corresponding main-policy regrets are 248.6, 132.7, and 32.2. With the Bernstein rule held fixed, pruning reduces mean regret from 74.1 to 32.2. The source experiment identifies an optimal mean of one, but several retained actions still have uncertain means. Consequently, initialization and subsequent exploration can incur regret even though the optimal value is known.

In Tasks 1 and 3, trUCB without UDB and Pruning + clipping select exactly the same actions as full trUCB under the shared random draws. Task 2 instead exhibits an additional gain from upper dominance and interval variance information. Together, the tasks distinguish action reduction from changes to the exploration of retained actions.

Separating the two uses of upper dominance. For Task 2, Fig. 27 further separates the efect of $u ^ { \star }$ on clipping from its efect on the variance interval. All four policies retain the same three actions and use the same Bernstein rule, initialization, reward draws, and action order. Starting from trUCB without UDB, UDB cap only replaces $u _ { \mathbf { x } }$ by $b _ { \mathbf { x } }$ only in the clipping step, while retaining the variance bound computed from $[ \ell _ { \mathbf { x } } , u _ { \mathbf { x } } ]$ . Conversely, UDB variance only uses the variance bound computed from $[ \ell _ { { \bf x } } , b _ { { \bf x } } ]$ but clips at $u _ { \mathbf { x } }$ . Full trUCB applies both changes. The cap-only comparison therefore difers from Pruning + clipping, which fixes the variance bound at $\textstyle { \frac { 1 } { 4 } }$

![](images/abb47d42b87abe3b7e4355792b61930b2a82883b091da04b362afd052ded297f.jpg)  
Figure 27: Separating the two uses of upper dominance in Task 2. All policies use the same three actions and Bernstein indices. UDB cap only adjusts the clipping level, UDB variance only adjusts the interval variance bound, and Full UDB applies both adjustments. Curves show mean cumulative pseudo-regret over 1,000 repetitions; shaded regions show one standard deviation.

The mean regrets are 123.9 without upper dominance, 75.2 with the cap adjustment alone, 91.2 with the variance adjustment alone, and 55.4 with both. Full trUCB reduces mean regret by 55.3% relative to trUCB without UDB. The mean paired reduction is 68.5, with a 95% confidence interval of [64.8, 72.3] across repetitions. Both uses of upper dominance therefore contribute to the improvement. The common cap removes the larger upper limit initially assigned to the suboptimal action, and the tighter variance bound further reduces its exploration. These efects do not require removing an action.

The improvement also persists when the horizon is changed: the mean regrets without and with upper dominance are 51.9 and 24.7 at $T = 2 { , } 0 0 0$ , and 145.4 and 64.6 at $T = 5 0 { , } 0 0 0$ Each horizon uses its own $\zeta _ { T }$ and 1,000 repetitions. In a further sensitivity check, changing each of the seven exogenous success probabilities by ±0.05, one at a time, gives 14 settings. Using 256 new repetitions per setting at $T = 1 0 { , } 0 0 0$ , the mean reductions range from 40.0% to 58.2%. These results illustrate a setting in which upper dominance improves learning; they do not imply an improvement for every SCM or every reward sequence.

## 6 Counterfactual Structural Causal Bandits

The preceding sections use causal structure to reduce the search space of a decision-making agent. This section extends the available interventions themselves.

Example 11 (AI-assisted clinical decision support system). Consider a hospital aiming to improve the quality of patient care. Let Y denote the treatment outcome, X a patient’s report of their health status, W an expert physician’s assessment, and $Z$ the score from an AI-assisted clinical tool. Fig. 28a illustrates the setting, where bidirected edges represent unobserved confounders such as historical bias (Peifer-Smadja et al., 2020).

![](images/851a7db176122bcdf64072d7b67737a1580631fe161c4414d9751edaf1529f82.jpg)  
Figure 28: (a) A structural causal bandit environment. (b) The submodel induced by do(x). (c) The submodel induced by the counterfactual action $d o ( W = W _ { x } , Z = Z _ { x ^ { \prime } } )$

For fixed report values x and $x ^ { \prime } { \mathrm { . } }$ , a standardized symptom-reporting interface can supply x to the expert assessment and $x ^ { \prime }$ to the AI tool. Supplying x to both assessments gives the same expected outcome as $d o ( X = x )$ , namely $\mathbb { E } Y _ { x }$ (Fig. 28b). Alternatively, x can describe a current health condition, while $x ^ { \prime }$ describes a prospective condition, such as the anticipated health benefits of beginning regular exercise (Fig. 28c). These report values are selected for the experiment before the patient’s natural report is observed. Supplying the two values separately gives $\mathbb { E } Y _ { W _ { x } , Z _ { x ^ { \prime } } }$ , allowing one to ask counterfactual-level questions:

## “What if the AI assistant had perceived the patient’s prospective health status $x ^ { \prime }$ rather than the current one $x ? ^ { \mathfrak { s } mathfrak { s } }$

and to design decision strategies that improve patient outcomes. Here $W _ { x }$ is the physician’s assessment under report x, and $Z _ { x ^ { \prime } }$ is the AI score under report $x ^ { \prime } .$ . Both concern the same patient, so their evaluation retains the same unobserved characteristics. The reporting interface changes the information received by each assessment; it does not itself change the patient’s physical health.

The example raises two questions.

• First, which counterfactuals can actually be produced through interaction with the system?

• Second, among the available actions, which can be removed without losing an optimal decision?

The first question requires a notion of realizability. The second extends the equivalence and partial-order arguments developed for ordinary interventions in Sec. 3.

We extend the SCB framework to accommodate a broader class of decision-making scenarios. Our key contributions in this section are as follows:

• We first formulate the counterfactual structural causal bandit problem, enabling an agent to consider actions beyond $\mathcal { L } _ { 2 } .$ -level interventions.

• We investigate equivalence and partial-order relations among these actions and develop graphical characterizations for reducing the action space.

• Building on these results, we present an algorithm that removes redundant or weakly dominated choices while retaining an optimal action.

Relation to the published version. This section revises Park and Lee (2026b). The counterfactual regime graph in Section 3.2 of Park and Lee (2026b) is corrected by deleting directed edges whose inputs are fixed to constants, retaining bidirected edges, and then restricting to reward ancestors (Thm. 14). The suficiency proof of Theorem 2 of Park and Lee (2026b) is replaced by a construction using reserved values and propagation of invalid inputs (Sec. C.1.3). The illustrative SCM in Section 3.2 of Park and Lee (2026b) is also corrected: $f _ { X }$ is set to $u _ { X } \vee z$ , and $f _ { Z }$ is specified as $u _ { Z } \vee ( w \oplus u _ { Z Y } )$ . Finally, Sec. 6.6 uses distinct realizable parent-input assignments, corrected ctf-POMIS arm counts, and a ctf-MIS baseline that includes nested actions. Its action counts and regret results replace those in Tables 1 and 2 of Park and Lee (2026b).

## 6.1 Background: Counterfactuals and Realizability

## 6.1.1 Counterfactual Distributions

We use the SCM, submodel, and PCH notation of Sec. 2. The counterfactual $X _ { \mathbf { w } } ( \mathbf { u } )$ evaluates X in the submodel $\mathcal { M } _ { \mathbf { w } }$ . Its random counterpart is $X _ { \mathbf { w } } = X _ { \mathbf { w } } ( \mathbf { U } )$ . Layer 3 of the PCH permits joint events involving counterfactuals from several such submodels. All counterfactuals are evaluated at the same exogenous realization (Bareinboim et al., 2022).

Definition 23 (Counterfactual distribution (Bareinboim et al., 2022)). An SCM M induces joint distributions over counterfactual events by:

$$
P ( \mathbf { y _ { x } } , \cdot \cdot \cdot , \mathbf { z _ { w } } ) = \sum _ { \mathbf { u } } \mathbb { 1 } \{ \mathbf { Y _ { x } } ( \mathbf { u } ) = \mathbf { y } , \cdot \cdot \cdot , \mathbf { Z _ { w } } ( \mathbf { u } ) = \mathbf { z } \} P ( \mathbf { u } ) .\tag{175}
$$

Shared exogenous inputs preserve the dependence between counterfactuals, including dependence induced by unobserved confounders. In the clinical example, $W _ { x }$ and $Z _ { x ^ { \prime } }$ describe two assessments of one patient under diferent reports. Separately sampling patients for the two assessments generally produces a diferent joint distribution. A fully specified SCM determines the joint counterfactual distribution, but its semantic definition alone supplies neither an identification formula from available data nor an experiment that measures the counterfactuals.

We write a collection of counterfactuals as $\mathbf { X } _ { * } = \{ X _ { i [ w _ { i } ] } \} _ { i = 1 } ^ { k }$ and its targets as $\begin{array} { r } { \mathbf { X } = \mathbf { V } ( \mathbf { X } _ { * } ) = } \end{array}$ $\{ X _ { i } \} _ { i = 1 } ^ { k }$ . The brackets separate the variable index i from its subscript assignment, and the symbol ∗ indexes the collection of counterfactual subscripts. Targets are distinct and exclude $Y ;$ every specified counterfactual has a nonempty subscript set. For example, $\mathbf { X } _ { * } = \{ W _ { x } , Z _ { x ^ { \prime } } \}$ has target set $\mathbf { X } = \{ W , Z \}$ . Diferent terms may use diferent values, including $x \neq x ^ { \prime }$ . Within a single counterfactual $Z _ { w }$ , however, the same assignment w is used throughout its defining submodel.

Nested counterfactuals. Counterfactuals can themselves supply the values of an intervention. For instance, $Y _ { \mathbf { Z _ { x } } } ( \mathbf { u } )$ , equivalently $Y _ { \mathbf { Z } = \mathbf { Z } _ { \mathbf { x } } ( \mathbf { u } ) } ( \mathbf { u } )$ , first computes ${ \bf Z _ { x } ( u ) }$ and then substitutes those values for Z in the mechanism for the outer counterfactual (Correa et al.,

2021). The inner and outer counterfactuals use the same u even though they may be evaluated under diferent interventions. More generally, $Y _ { \mathbf { X } _ { * } } ( \mathbf { u } )$ evaluates the specified counterfactuals and substitutes their values for the corresponding targets.

We write $\mathbf { X } _ { * * }$ when the assigned counterfactuals may themselves have counterfactuals in their subscripts. The double asterisk allows arbitrary nesting and includes $\mathbf { X } _ { * }$ as a special case. This distinction concerns the assigned counterfactuals; the reward $Y _ { \mathbf { X } } .$ already involves a counterfactual substitution.

Counterfactual calculus. The ctf-calculus provides rules for manipulating counterfactual probability quantities (Correa and Bareinboim, 2025). We recall the consistency and exclusion rules used below.

CTF-Rule 1 (Consistency). Once a counterfactual already takes the value x, explicitly setting it to that same value does not change the remaining counterfactuals:

$$
P ( \mathbf { y _ { T _ { * * } , x } } , \mathbf { x _ { T _ { * * } } } , \mathbf { w _ { * } } ) = P ( \mathbf { y _ { T _ { * * } } } , \mathbf { x _ { T _ { * * } } } , \mathbf { w _ { * } } )\tag{176}
$$

This rule allows a computed counterfactual to replace an explicit assignment when their values agree.

The extension from $\mathbf { T } _ { * }$ in the original rule to $\mathbf { T } _ { * } ,$ follows from unit-level composition: at a fixed exogenous realization u, evaluate the nested assignments recursively, apply consistency to the resulting values, and then average over u. All nested counterfactuals use that same exogenous realization.

CTF-Rule 3 (Exclusion). An assignment can be removed when its variables cannot afect the counterfactual after the other interventions:

$$
P ( \mathbf { y _ { x z } } , \mathbf { w _ { * } } ) = P ( \mathbf { y _ { z } } , \mathbf { w _ { * } } ) , \qquad { \mathrm { i f ~ } } \mathbf { X } \cap \mathrm { A n } ( \mathbf { Y } ) _ { \mathcal { G } _ { \overline { { \mathbf { z } } } } } = \emptyset .\tag{177}
$$

The ancestor condition is checked after cutting the incoming edges of Z. In the clinical example, fixing both $W$ and $Z$ blocks every directed path from $X$ to $Y$ , so an additional assignment to $X$ cannot change $Y _ { w z }$ . Consistency and exclusion will be used to identify counterfactual assignments that do not change the reward.

Counterfactual unnesting theorem. Nested distributions can also be expanded into joint events involving counterfactuals with less nesting. This is useful because each term then states explicitly which values must occur together on the same unit.

Lemma 9 (Counterfactual Unnesting Theorem (CUT); Theorem 4 in Correa et al. (2021)). For a collection of counterfactuals $\mathbf { T } _ { * * }$

$$
P ( Y _ { \mathbf { T } _ { * * } , X _ { \mathbf { z } } } = y ) = \sum _ { x } P ( Y _ { \mathbf { T } _ { * * } , x } = y , X _ { \mathbf { z } } = x ) .\tag{178}
$$

For example, applying CUT to the two assessments gives:

$$
P ( Y _ { W _ { x } , Z _ { x ^ { \prime } } } = y ) = \sum _ { w z } P ( Y _ { w z } = y , W _ { x } = w , Z _ { x ^ { \prime } } = z ) .\tag{179}
$$

For each possible pair $w , z ,$ , the event requires both assessments to produce that pair and the outcome under those values to equal y. Summing over the pairs recovers the nested distribution. These counterfactuals cannot generally be sampled independently because they retain the same exogenous realization. CUT is an identity between distributions; the expansion alone does not supply an experiment for the joint events or a single unnested action equivalent to the nested regime.

## 6.1.2 Realizability

Throughout this section, endogenous variables have finite discrete domains, as in Raghavan and Bareinboim (2025). These domains are fixed within each bandit instance but may vary across the compatible SCMs used in a graphical characterization, as specified in Sec. 3. The exogenous variables may have general distributions.

Experimental capabilities. Layer 1 distributions can be sampled through passive observation of the system’s natural behavior. Layer 2 distributions require interventions, such as Fisherian randomization (Fisher, 1935). Counterfactual randomization further permits changing the value transmitted to selected child mechanisms while allowing other children to receive the original value (Raghavan and Bareinboim, 2025).

Definition 24 (Feasible operations). An operation collection A specifies which of the following physical operations are available:

1. Select: select a unit from the target population.

2. Read(V ): observe the output of the mechanism generating V on that unit.

3. Rand(X): replace the natural mechanism of X by an externally randomized value whose support is $\Omega _ { X }$

4. ${ \mathrm { C T F - R A N D } } ( X \to \mathbf { Z } )$ : randomize the value of X received by selected directed children $\mathbf { Z } \subseteq \mathbb { C } \mathrm { { h } } ( X ) _ { \mathcal { G } }$ , using suitable counterfactual mediators.

A counterfactual mediator for $X  Z$ is a manipulable interface through which $f _ { Z }$ receives the information about X. Its ordinary states must encode X so that the input received by $f _ { Z }$ agrees with the original mechanism; changing the interface reproduces the intended alternative input. The reporting interface plays this role in the introductory example. Availability of such an interface is an experimental capability assumption: observing the edge $X  Z$ alone does not establish it. The formal mediator conditions are given by Raghavan and Bareinboim (2025).

Realizability. Counterfactual semantics specify a counterfactual even when no available experiment can measure it. Realizability asks whether physical operations can produce samples from the required distribution, including the dependence among counterfactuals on the same unit.

Definition 25 (Realizability). A counterfactual distribution is realizable from A if operations in A produce samples from it for every SCM conforming to $\mathcal { G }$ and the specified experimental capabilities.

Throughout this section, we use the maximal collection $\mathbb { A } ^ { \dagger }$ , which permits the operations above at every relevant variable and parent-child interface. Each external randomizer assigns positive probability to every value in the corresponding finite domain. The randomizers for separately selected inputs are independent of one another and of the unit’s exogenous variables. Assigned inputs can therefore be obtained by retaining units whose randomized inputs match the selected assignment. Retention depends only on these randomized inputs. As in Sec. 3, each bandit round uses a fresh independent retained unit. Discarded randomization attempts are outside the round count. Even with these capabilities, a mechanism cannot be evaluated twice under conflicting inputs on the same unit. Diferent reports sent to diferent mechanisms, as in $W _ { x }$ and $Z _ { x ^ { \prime } }$ in the clinical example, are allowed. The dificulty arises when the requested counterfactuals require two diferent versions of the same mechanism. Counterfactual ancestors make these requirements explicit.

Counterfactual ancestors. Before listing required mechanisms, we simplify the counterfactuals. Efectiveness gives $X _ { x \mathbf { w } } = x$ , so a directly assigned constant requires no evaluation of $f _ { X }$ . For other counterfactuals, interventional minimization removes assignments that cannot afect the counterfactual (Correa et al., 2021):

$$
\| X _ { \mathbf { w } } \| = X _ { \mathbf { w } \cap \mathrm { A n } ( X ) \mathcal { G } _ { \overline { { \mathbf { W } } } } } .\tag{180}
$$

For a collection, $\| \mathbf { X } _ { * } \|$ applies this operation to each term. We apply this simplification before checking realizability.

The counterfactual ancestors of $X _ { \mathbf { w } } .$ , denoted $\mathtt { A n } ( X _ { \mathbf { w } } ) _ { \mathcal { G } }$ , record the counterfactuals needed to compute it. A counterfactual $T _ { \mathbf { z } }$ belongs to this set when:

1. $T \in \mathsf { A n } ( X ) _ { \mathcal { G } _ { \overline { { \mathbf { W } } } } } \ \backslash$ W, and

$$
2 . \ \mathbf { z } = \mathbf { w } \cap \mathtt { A n } ( T ) _ { \mathcal { G } _ { \overline { { \mathbf { W } } } } } .
$$

The first condition traces the mechanisms that remain upstream of X after fixing W. The second retains only the assignments that can afect each such mechanism. For a collection, $\mu \mathrm { n } ( \mathbf { X } _ { * } ) _ { \mathcal { G } }$ is the union of the counterfactual ancestor sets of its terms. Natural variables may occur as intermediate ancestors; they need not be among the assigned counterfactuals. Two required versions conflict when their subscript assignments difer after interventional minimization, including the variables to which those assignments apply.

For instance, the causal diagram in Fig. 29a with $\mathbf { X } _ { * } = \{ W _ { x } , Z _ { x ^ { \prime } } \}$ gives:

$$
\mathtt { A n } ( \mathbf { X } _ { * } ) _ { \mathcal { G } } = \mathtt { A n } ( W _ { x } ) _ { \mathcal { G } } \cup \mathtt { A n } ( Z _ { x ^ { \prime } } ) _ { \mathcal { G } }\tag{181}
$$

$$
= \{ W _ { x } , Z _ { x } \} \cup \{ Z _ { x ^ { \prime } } \} = \{ W _ { x } , Z _ { x } , Z _ { x ^ { \prime } } \} .\tag{182}
$$

Computing $W _ { x }$ first requires $Z _ { x }$ , because Z lies between X and W. The separate counterfactual $Z _ { x ^ { \prime } }$ then asks for a second version of Z. If $x ^ { \prime } = x$ , these requirements coincide and one evaluation of $f _ { Z }$ supplies both. If $x ^ { \prime } \neq x$ , however, they require conflicting inputs to that mechanism. This leads to the graphical criterion below.

Lemma 10 (Realizability; Corollary 3.7 of Raghavan and Bareinboim (2025)). Under $\mathbb { A } ^ { \dagger }$ , a counterfactual distribution is realizable if and only if its ancestor set contains no conflicting required versions of the same mechanism.

![](images/dae40b89fe41709355466c42ac1da09fbc03b1c98e954cdedce974d1899ad5aa.jpg)  
Figure 29: Conflicting mechanism evaluations. (a) For $x \neq x ^ { \prime } .$ , realizing $\{ W _ { x } , Z _ { x ^ { \prime } } \}$ requires both $Z _ { x }$ and $Z _ { x ^ { \prime } } . \ ( \mathrm { b } )$ The counterfactual $Z _ { w }$ requires $X _ { w }$ , while the outer reward also uses the natural X.

Thus $P ( W _ { x } , Z _ { x ^ { \prime } } )$ in Fig. 29a is not realizable when $x \neq x ^ { \prime }$ . The obstacle concerns producing both counterfactuals on one unit, even though either counterfactual can be measured separately.

For an action, realizability concerns the mechanisms actually required to evaluate its reward on the same unit, including the inner counterfactuals that the reward uses. In Fig. 29b, $Z _ { w }$ requires $X _ { w }$ as an input. However, the outer counterfactual $Y _ { Z _ { w } }$ also receives the natural X through the direct edge $X  Y$ . Producing the required counterfactuals would therefore require both versions of X. If the action instead supplies $X _ { w }$ to the outer reward as well, giving $Y _ { X _ { w } , Z _ { w } }$ , the conflicting requirement disappears. This example explains why checking the assigned counterfactuals alone is insuficient: the complete nested evaluation determines which mechanisms are needed.

## 6.2 Counterfactual Structural Causal Bandits

We now formalize the counterfactual structural causal bandit (ctf-SCB) problem. As in Sec. 3, an agent interacts with a target system modeled by an SCM with a reward variable Y. The agent knows $\mathcal { G }$ and Y, while the structural functions and the exogenous distribution remain unknown. Traditional SCB (Lee and Bareinboim, 2018) optimizes the mean reward of an ordinary intervention:

$$
\mu _ { \mathbf { x } } = \mathbb { E } Y _ { \mathbf { x } } = \mathbb { E } [ Y \mid d o ( \mathbf { X } = \mathbf { x } ) ] .\tag{183}
$$

A counterfactual action $\mathbf { X } _ { * * }$ can instead supply computed counterfactuals to the system. We write its mean reward as:

$$
\mu _ { \mathbf { X } _ { * * } } = \mathbb { E } Y _ { \mathbf { X } _ { * * } } .\tag{184}
$$

When the action is given by counterfactual assignments, each $X _ { i [ \mathbf { w } _ { i } ] }$ is computed under the indicated submodel and substituted for its target $X _ { i }$ . The required mechanisms and the reward are evaluated on the same unit.

For concreteness, return to Fig. 28. The mean reward $\mu _ { x }$ is obtained by setting $X = x ,$ whereas $\mu _ { W _ { x } , Z _ { x ^ { \prime } } }$ uses the physician’s counterfactual under x and the AI counterfactual under $x ^ { \prime }$ . Since W and Z are the only children of X in this graph, $\mu _ { x } = \mu _ { W _ { x } , Z _ { x } }$ . Indeed, setting $X = x$ sends x to both assessments, and assigning x separately to the two inputs produces the same reward. Allowing $x ^ { \prime }$ to difer from x makes additional decisions available. The ordinary intervention is recovered when both input values agree.

Algorithm 9: Constructing the SCM for an assigned counterfactual action.   
1 Initialize $\mathcal { F } _ { \mathbf { X } _ { * * } }$ as the collection of original mechanisms ${ \mathcal F } .$   
2 for each $X \in \mathbf { V }$ in topological order do   
3 for each $W \in \mathsf { p a } [ X ]$ do   
4 Substitute the selected value for the argument $W$ in $f _ { X } \in \mathcal { F } _ { \mathbf { X } _ { * * } } .$   
5 Retain every other parent argument and every exogenous argument of $f _ { X }$   
6 return $\mathcal { M } _ { \mathbf { X } _ { * * } } = \langle \mathbf { U } , \mathbf { V } , \mathcal { F } _ { \mathbf { X } _ { * * } } , P ( \mathbf { U } ) \rangle$

Parent inputs of counterfactual actions. In the preceding example, counterfactuals describe the action. For computation, we describe the same experiment through the inputs received by each mechanism. At $f _ { X }$ , the map pa[X] records the parents whose inputs are assigned constants. Each selected input has a separately selectable value, and every other input receives the output of its parent’s mechanism on the same unit. Inputs to $f _ { Y }$ may also be fixed, while the reward mechanism itself is retained. Each required mechanism is evaluated once, in a topological order of ${ \mathcal { G } } .$

We use ${ \bf X } _ { * * }$ for this counterfactual action specified by these inputs and $Y _ { \mathbf { X } _ { * * } }$ for its reward. The parent-input specification preserves the constant inputs and counterfactual substitutions of the original action. Values within each counterfactual subscript remain unchanged. Inputs that cannot afect Y may be discarded.

Let $\mathcal { M } _ { \mathbf { X } _ { * } }$ be the SCM obtained by fixing these selected parent arguments, retaining every other argument and the original exogenous inputs. Furthermore, let $\mathcal { G } _ { \mathbf { X } _ { * * } }$ be a graph obtained from G by deleting $W  V$ whenever a required evaluation of $f _ { V }$ receives a constant in place of its W argument. All vertices, other directed edges, and bidirected edges are retained.

A directed path in $\mathcal { G } _ { \mathbf { X } _ { * } }$ therefore follows inputs that still receive their parent’s computed counterfactual. In particular, a vertex loses a parent precisely when that incoming edge is replaced by a constant input.

Definition 26 (Action space). The action space $\mathcal { A } _ { \mathrm { C T F } }$ consists of all counterfactual actions realizable under $\mathbb { A } ^ { \dagger }$ , with their selected input values specified.

This space includes realizable nested counterfactual actions. Their constant inputs can be recorded at the receiving mechanisms, as established below.

Proposition 13. Every realizable nested counterfactual action can be implemented by the parent-input choices described above.

Proof of Prop. 13. Expand $Y _ { \mathbf { X } _ { * * } }$ into its required structural equations, applying efectiveness and identifying equal counterfactuals. Realizability gives one evaluation of each required mechanism. Record in pa[X] every parent input fixed to a constant at $f _ { X }$ , retaining its value. All other arguments are outputs of parent mechanisms on the same unit. These choices define an action in $\mathcal { A } _ { \mathrm { C T F } }$

To compare the two evaluations, proceed in topological order. Each first required mechanism receives the same fixed inputs and exogenous realization in both evaluations, and hence produces the same output. At a subsequent mechanism, the parent outputs already agree, and the fixed inputs and exogenous realization also agree. Its output is therefore unchanged. Continuing to Y proves equality of the reward on every unit. □

Constructing the SCM for a counterfactual action. Alg. 9 starts from the original mechanisms and replaces only the selected parent arguments. Each remaining argument receives its parent’s output on the same unit, and every exogenous input is retained. Evaluating these mechanisms in topological order gives the intended reward $Y _ { \mathbf { X } _ { * * } }$ . Selecting an input of $f _ { Y }$ changes that argument while leaving its other inputs available; the reward mechanism itself is retained.

The construction specifies how an action is evaluated in an SCM. As in the previous sections, the bandit agent need not know these mechanisms. Instead, it performs the corresponding experiment and observes the reward.

Counterfactual structural causal bandits objective. As in Sec. 3, at round t the agent chooses $A _ { t } \in \mathcal { A } _ { \mathrm { C T F } }$ , performs the corresponding experiment on a fresh unit, and observes its reward $Y _ { t } .$ . Repeated selections provide samples for estimating the mean of that action, just as ordinary interventions do in SCB.

We use the best-action convention of Sec. 3: a superscript ∗ optimizes the subscript values of a fixed set of counterfactuals, and ⋆ selects an optimal action over the entire space. The cumulative pseudo-regret is:

$$
R _ { T } ^ { \mathcal { A } _ { \mathrm { C T F } } } = T \mu _ { \mathbf { X } _ { * * } ^ { \star } } - \sum _ { t = 1 } ^ { T } \mu _ { A _ { t } } = \sum _ { \mathbf { X } _ { * * } \in \mathcal { A } _ { \mathrm { C T F } } } \Delta _ { \mathbf { X } _ { * * } } N _ { T } ( \mathbf { X } _ { * * } ) ,\tag{185}
$$

where $\Delta _ { \mathbf { X } _ { * * } } = \mu _ { \mathbf { X } _ { * * } ^ { \star } } - \mu _ { \mathbf { X } , }$ ∗∗ and $N _ { T } ( \mathbf { X } _ { * * } )$ counts the rounds $t \leq T$ with $A _ { t } = \mathbf { X } _ { * * }$ . The learning objective has the same form as before; the available experiments and the benchmark reward have been extended.

Corollary 5. The optimal ordinary intervention cannot outperform the optimal counterfactual action, i.e., $\mu _ { \mathbf { X } ^ { \star } } \leq \mu _ { \mathbf { X } _ { * * } ^ { \star } }$

Proof of Cor. 5. An ordinary intervention is obtained by assigning the same value to every outgoing input of each intervened variable. Hence its reward is represented in $\mathcal { A } _ { \mathrm { C T F } }$ □

A larger space can ofer a higher reward, but it also gives the agent more actions to explore. The following sections use causal structure to reduce this exploration burden. We first ask when removing a counterfactual assignment leaves the reward unchanged, and then compare the best rewards attainable by non-equivalent counterfactual actions.

## 6.3 Equivalence among Actions

In Sec. 3, an ordinary intervention set is reduced by removing variables whose assigned values cannot afect the reward once the other interventions are in place. For example, if every directed path from $Z$ to Y passes through an intervened variable W, the additional assignment to $Z$ does not change Y. An MIS retains the intervention variables needed to distinguish that intervention set from its proper subsets.

Counterfactual minimal intervention sets. The same question arises for a set of counterfactuals ${ \bf X } _ { * * }$ with outer targets $\mathbf { X } = \mathbf { V } ( \mathbf { X } _ { * * } )$ . Each target now receives a counterfactual, which may itself contain nested counterfactuals. Such an assignment can be redundant for two reasons:

1. The target may have no remaining directed path to the reward.

2. The counterfactuals assigned to the other targets may already induce the same counterfactual at that target.

We examine both possibilities using the consistency and exclusion rules of ctf-calculus.

We first apply efectiveness and interventional minimization within the nested counterfactuals. For the special case $\mathbf { X } _ { * } ~ = ~ \{ X _ { i [ w _ { i } ] } \} _ { i }$ , this gives $\mathbf { X } _ { * } ~ = ~ \| \mathbf { X } _ { * } \|$ . We then ask whether the assignment to a target variable can be removed altogether. Retained counterfactuals keep all their inner substitutions and constant assignments. As in the MIS definition of Sec. 3, this comparison concerns an expression with variable assignments: retained terms keep their assignments, and equivalence must hold for all admissible values and all compatible SCMs.

Definition 27 (Counterfactual minimal intervention set). A set of counterfactuals $\mathbf { X } _ { * * }$ is a counterfactual minimal intervention set (ctf-MIS) if no proper subset ${ \mathbf { Z } } _ { \ast \ast } \subseteq { \mathbf { X } } _ { \ast \ast }$ satisfies $\mu _ { \mathbf { X } _ { * * } } = \mu _ { \mathbf { Z } _ { * * } }$ for every compatible SCM and every admissible subscript assignment.

Thus, deleting a term requires equality for every admissible assignment and every compatible SCM. The following examples show how the causal diagram can justify removing an assignment for all such values and SCMs.

For example, consider the causal diagram shown in Fig. $2 9 \mathrm { a } .$ and take the realizable counterfactual $\mathbf { X } _ { * } = \{ W _ { x } , Z _ { x } \}$ . The two assigned counterfactuals use the same assignment $x .$ To generate the reward, the outer system receives $W _ { x }$ at $W$ and $Z _ { x }$ at $Z .$ . However, once W receives its assigned counterfactual, the value supplied separately at $Z$ has no directed path to Y . CUT makes this explicit:

$$
\mu _ { W _ { x } , Z _ { x } } = \sum _ { y w z } y P ( y _ { w z } , w _ { x } , z _ { x } )\tag{186}
$$

$$
= \sum _ { y w z } y P ( y _ { w } , w _ { x } , z _ { x } ) = \sum _ { y w } y P ( y _ { w } , w _ { x } ) = \mu _ { W _ { x } } .\tag{187}
$$

The second line uses ctf-Rule 3 and then sums over z. Removing the outer assignment $Z _ { x }$ still leaves the evaluation of $Z _ { x }$ needed inside the submodel that computes $W _ { x }$ . It is the separate assignment at the outer target $Z$ that is redundant. Therefore $\{ W _ { x } , Z _ { x } \}$ is not a ctf-MIS.

In Fig. 29b, consider $\mathbf { X } _ { * } = \{ X _ { w } , Z _ { w ^ { \prime } } \}$ . Both targets have directed edges to $Y .$ , so neither is excluded by the ordinary MIS condition. Nevertheless, computing $Z _ { w ^ { \prime } }$ requires $X _ { w ^ { \prime } }$ , while the first assignment requires $X _ { w }$ . Realizability therefore requires $w ^ { \prime } = w$

With this shared value, assigning $X _ { w }$ already makes Z behave as $Z _ { w }$ since the only directed path from $W$ to $Z$ passes through X. ctf-Rule 1 permits substituting $X _ { w }$ into the submodel for $Z _ { w } .$ , and ctf-Rule 3 then removes the assignment to W: $Z _ { w } = Z _ { w , X _ { w } } = Z _ { X _ { w } }$ and thus $Y _ { X _ { w } , Z _ { w } } = Y _ { X _ { w } }$

![](images/82aba27fdf15b43da33585dde29599324991ae1a512ea37c9d955ae73eb12772.jpg)  
Figure 30: (a) A causal diagram. (b) Equivalent counterfactual actions induce the same propagation of x and $x ^ { \prime }$ to Y .

The other parent T of Z retains its natural value on the same unit in both evaluations. Thus every admissible assignment satisfies $\mu _ { X _ { w } , Z _ { w } } = \mu _ { X _ { w } }$ , and the assignment $Z _ { w }$ can be removed even though $Z$ has a directed edge to Y.

Graphical characterization for CTF-MISs. Throughout the remainder of this section, we assume that $\mathbf { X } _ { * * }$ is realizable and every assigned counterfactual is nonconstant, i.e., after efectiveness, it still requires its target mechanism. Constant values are free apart from equalities required by realizability.

The graph $\mathcal G _ { \mathbf { X } , \mathbf { \Lambda } }$ deletes the directed edges replaced by constant inputs. Thus a vertex V receives a fixed input exactly when its parent set in this graph is strictly smaller than in ${ \mathcal { G } } .$ In the second condition below, deleting the other outer targets from this graph ensures that the entire path, including its starting vertex, avoids those targets.

Theorem 13 (Graphical characterization of ctf-MIS). A counterfactual action $\mathbf { X } _ { * * }$ is a ctf-MIS if and only if the following two conditions hold:

1. $\mathbf { X } \subseteq \mathsf { A n } ( Y ) _ { \mathcal { G } _ { \overline { { \mathbf { X } } } } }$

2. For every $X _ { i } \in \mathbf { X }$ , there exists $V \in \mathsf { A n } ( X _ { i } ) _ { \mathcal { G } _ { \mathbf { X } _ { * * } } \backslash ( \mathbf { X } \backslash \{ X _ { i } \} ) }$ such that $\mathtt { P a } ( V ) _ { \mathcal { G } _ { \mathbf { X } _ { * * } } } \subsetneq \mathtt { P a } ( V ) _ { \mathcal { G } }$

The proof is given in Sec. C.1.1. The first condition is the ordinary MIS condition applied to the outer target set X. Each target must have a directed path to Y that avoids the other targets. In the first example, Z fails this test because its only path to Y passes through W. In contrast, $\{ W _ { x } , Z _ { x ^ { \prime } } \}$ in Fig. 30a has the two paths $W  T  Y$ and $Z \to R \to Y$ Therefore, both targets satisfy the first condition.

The second condition traces the counterfactual’s computation in the other direction. A directed edge deleted from $\mathcal { G }$ must enter a vertex with a retained directed path to $X _ { i }$ avoiding every other outer target. The path may have length zero: a fixed input received by $X _ { i }$ itself sufices. A fixed input received only at another outer target does not sufice, because that target’s counterfactual is still supplied when the assignment to $X _ { i }$ is removed.

![](images/2b144e5a1ca31c6e80512fc564bd72fdd335b0045083909ef8bbb41c04163376.jpg)  
Figure 31: (a) A causal diagram and (b) the graph for $\mathbf { X } _ { * * } = \{ X _ { w } , Z _ { w ^ { \prime } , X _ { w } } \}$

In the second example, the fixed input $W  X$ enters X. The counterfactual $X _ { w }$ therefore satisfies the second condition. For $Z _ { w }$ , however, this input reaches $Z$ only through the other target X. Deleting X from the graph removes that path, and $Z _ { w }$ fails the condition. In contrast, for $\{ W _ { x } , Z _ { x ^ { \prime } } \}$ in Fig. 30a, both W and Z receive a fixed input directly, so both conditions hold.

Corollary 6. For $\mathbf { X } _ { * } = \{ X _ { i \lceil w _ { i } \rceil } \} _ { i }$ satisfying the first condition, the second condition is equivalent to $\mathbf { W } _ { i } \cap \mathop { \mathrm { A n } } ( X _ { i } ) \mathop { \mathcal { G } _ { \overline { { \mathbf { X } \backslash \{ X _ { i } \} } } } } { } \ne \emptyset$ for every $X _ { i }$

Proof of Cor. 6. The first condition in Thm. 13 ensures that every assigned target computation is required by the reward. For the second condition, suppose a vertex of $\mathbf { W } _ { i }$ has a directed path to $X _ { i }$ in $\mathcal { G } _ { \overline { { \mathbf { X } \backslash \{ X _ { i } \} } } }$ . Take the last vertex of $\mathbf { W } _ { i }$ on this path. Its input to the next mechanism is fixed, whereas the remaining path to $X _ { i }$ contains no further subscript intervention or other outer target. Realizability preserves these inputs in $\mathcal { G } _ { \mathbf { X } _ { * } }$ , giving the mechanism and path required by the general condition.

Conversely, a fixed input received by a mechanism on such a path to $X _ { i }$ must be an assignment from $\mathbf { W } _ { i } { \mathrm { : } }$ these are exactly the constants used in the submodel computing $X _ { i [ \mathbf { w } _ { i } ] }$ Prepending that input edge to the path gives a directed path from $\mathbf { W } _ { i }$ to $X _ { i }$ whose internal vertices avoid the other outer targets. Consequently, the general condition is equivalent to $\mathbf { W } _ { i } \cap \mathop { \mathrm { A n } } ( X _ { i } ) _ { \mathcal { G } _ { \overline { { \mathbf { X } \backslash \{ X _ { i } \} } } } } \not = \emptyset$ , proving the specialization. □

Consider the causal diagram in Fig. 31a with $\mathbf { X } _ { * * } = \{ X _ { w } , Z _ { w ^ { \prime } , X _ { w } } \}$ . The first counterfactual gives $f _ { X }$ the input $W = w$ . The second gives f<sub>Z</sub> the input $W = w ^ { \prime }$ together with the previously computed $X _ { w }$ . Consequently, $W  X$ and $W \to Z$ are deleted, while $X  Z$ remains in $\mathcal { G } _ { \mathbf { X } } .$ (Fig. 31b). Both targets have a directed edge to $Y$ and receive a fixed input themselves. Thus, $\mathbf { X } _ { * * }$ is a ctf-MIS.

Reducing an action to a CTF-MIS. ctf-MISify (Alg. 10) applies these conditions to a supplied counterfactual set. It first removes all targets blocked from Y by other outer targets. A reverse search from Y, stopping at each outer target, identifies exactly the assignments to retain. It then examines each remaining target once, deleting its assignment if the second condition fails. The nested counterfactuals and constant values inside every retained term are unchanged.

Each deletion preserves the reward on every unit. The initial deletions are valid together: every path from an omitted target to $Y$ meets a retained target. Moreover, these deletions preserve the required mechanism evaluations and their fixed inputs, so $\mathcal { G } _ { \mathbf { X } _ { * * } }$ is unchanged. A target that passes either condition continues to pass after other targets are deleted, because its witnessing path remains available. Thus a single pass sufices, and the returned set satisfies both conditions.

Algorithm 10: Reducing a counterfactual action to a ctf-MIS.   
1 function $\mathrm { C T F - M I S I F Y } ( \mathcal { G } , Y , \mathbf { X } _ { * * } )$   
2 Compute $\mathcal G _ { \mathbf { X } _ { * } }$ and keep it fixed throughout.   
3 Set $\mathbf { X } = \mathbf { V } ( \mathbf { X } _ { * * } )$   
$/ /$ Apply the first condition in Thm. 13.   
4 Retain only the outer assignments whose targets belong to $\mathtt { A n } ( Y ) _ { \mathcal { G } _ { \overline { { \mathbf { X } } } } }$   
$/ /$ Apply the second condition in Thm. 13.   
5 Update $\mathbf { X } = \mathbf { V } ( \mathbf { X } _ { * * } )$   
6 for each remaining target $X _ { i }$ in any fixed order do   
7 Search backwards along directed edges from $X _ { i }$ in ${ \mathcal G } _ { \mathbf { X } _ { * * } } \ \backslash \ ( \mathbf { X } \setminus \{ X _ { i } \} )$   
including $X _ { i }$ itself.   
8 if every visited vertex has the same parents in $\mathcal { G } _ { \mathbf { X } _ { * * } }$ and $\mathcal { G }$ then   
9 Delete only the outer assignment to $X _ { i }$ from ${ \bf X } _ { * * }$   
10 Update $\mathbf { X } = \mathbf { V } ( \mathbf { X } _ { * * } )$   
11 return $\mathbf { X } _ { * * }$

Corollary 7. ctf-MISify returns an equivalent and realizable ctf-MIS subset of its input.

Proof of Cor. 7. In the first step, every directed path from a discarded target to Y meets a retained target: the last outer target on that path satisfies the first condition. Thus these assignments can be removed together, while keeping every retained nested counterfactual unchanged.

By the necessity proof of Thm. 13, both deletion steps preserve the reward on every unit, for every compatible SCM and every admissible assignment. They also preserve the required mechanism evaluations and their fixed inputs. Hence realizability and the initially computed graph $\mathcal G _ { \mathbf { X } _ { * } }$ are preserved.

Deleting other outer targets cannot destroy a path witnessing either condition for a retained target. Therefore, a target that passes a condition continues to pass it throughout the algorithm. After the single pass, every remaining target satisfies both conditions. Therefore, Thm. 13 gives the conclusion. □

Computational cost. Let $n = | \mathbf { V } |$ and m be the number of edges in ${ \mathcal { G } } ,$ and let $k = | \mathbf { X } | + 1$ for the initial outer target set. Comparing the two graphs’ parent sets and performing the initial reverse search take $\mathcal { O } ( n + m )$ time. Each remaining target requires at most one further reverse search, also taking $O ( n + m )$ time. Therefore ctf-MISify uses $\mathcal { O } ( k ( n + m ) )$ time and $\mathcal { O } ( n + m )$ auxiliary space. These bounds concern the graphical reduction after the action graph has been formed; retained nested terms are reused without expanding them.

Equivalent CTF-MISs. Diferent ctf-MISs can still describe the same input action even when neither target set contains the other. Two counterfactual sets are equivalent if every admissible subscript assignment to either set has a corresponding assignment to the other, independent of the SCM, that gives the same reward on every unit. To see this, consider the causal diagram shown in Fig. 30a. The counterfactual sets $\{ T _ { x } , Z _ { x ^ { \prime } } \}$ and $\{ W _ { x } , Z _ { x ^ { \prime } } \}$ satisfy both ctf-MIS conditions. The first assigns $T _ { x }$ directly to $T ;$ ; the second assigns $W _ { x }$ to W, which induces the same counterfactual $T _ { x }$ through the mechanism for T. In particular, W receives the fixed input $x ,$ and the retained path $W  T$ avoids the other target Z. This verifies the second condition for $T _ { x }$ , and its edge $T  Y$ verifies the first condition.

To compare the two actions, first compute $W _ { x }$ on a unit. Feeding this value into the unchanged mechanism for T produces $T _ { x }$ . The counterfactual $Z _ { x ^ { \prime } }$ on the other branch is identical under both actions, as is the natural value of X received directly by Y. Thus the reward mechanism receives the same inputs on the same unit. These substitutions give:

$$
\mu _ { T _ { x } , Z _ { x ^ { \prime } } } = \sum _ { y , t , z } y P ( y _ { t z } , t _ { x } , z _ { x ^ { \prime } } )
$$

$$
\mathrm { C U T }\tag{188}
$$

$$
= \sum _ { y , t , z , w } y P ( y _ { t z } , t _ { x } , z _ { x ^ { \prime } } , w _ { x } )
$$

$$
{ \mathrm { m a r g . ~ o v e r ~ } } W _ { x }\tag{189}
$$

$$
= \sum _ { y , t , z , w } y P ( y _ { t w z } , t _ { w x } , z _ { x ^ { \prime } } , w _ { x } )
$$

$$
_ { \mathrm { C T F - R u l e s \ 1 \ \mathrm { a n d \ 3 } } }\tag{190}
$$

$$
= \sum _ { y , t , z , w } y P ( y _ { t w z } , t _ { w } , z _ { x ^ { \prime } } , w _ { x } )
$$

$$
\mathrm { C T F  – R u l e \ 3 }\tag{191}
$$

$$
= \sum _ { y , t , z , w } y P ( y _ { t w z } , t _ { w z } , z _ { x ^ { \prime } } , w _ { x } )
$$

$$
\mathrm { C T F  – R u l e \ 3 }\tag{192}
$$

$$
= \sum _ { y , t , z , w } y P ( y _ { w z } , t _ { w z } , z _ { x ^ { \prime } } , w _ { x } )
$$

$$
\mathrm { C T F  – R u l e \ 1 }\tag{193}
$$

$$
= \sum _ { y , z , w } y P ( y _ { w z } , z _ { x ^ { \prime } } , w _ { x } )
$$

$$
\mathrm { s u m . \ o v e r \ } T _ { w z }\tag{194}
$$

$$
\mathbf { \Psi } = \mu _ { W _ { x } , Z _ { x ^ { \prime } } } .
$$

$$
\mathrm { C U T }\tag{195}
$$

Similarly, $\{ T _ { x } , R _ { x ^ { \prime } } \}$ has the same reward. These ctf-MISs assign the same inputs x on $X  W$ and $x ^ { \prime }$ on $X  Z ,$ as shown in Fig. 30b. The action space $A _ { \mathrm { C T F } }$ identifies these sets through their common input choices.

## 6.4 Counterfactual Possibly-Optimal Minimal Intervention Sets

The preceding section identifies redundant outer assignments in a realizable set $\mathbf { X } _ { * * }$ . Removing these assignments preserves the reward for every admissible choice of values. We now compare non-equivalent counterfactual actions after optimizing their subscript values. As in Sec. 3, the causal diagram can imply that one optimized counterfactual always performs at least as well as another, regardless of the parameterization of the underlying SCM. These partial-order relations allow the agent to exclude further actions before estimating their rewards.

![](images/d06befd7883f12ff5e40f3a74e2bf2b1d74730be494f298dd1fa568010fe75b2.jpg)  
Figure 32: (a) A causal diagram. (b) A counterfactual action with separately chosen values $w ^ { * }$ and $w ^ { \dagger }$ on the two outgoing edges of W.

From ordinary interventions to counterfactual interventions. First consider ordinary interventions in the causal diagram of Fig. 32a. Let $w ^ { * }$ maximize the reward under $d o ( W = w )$ , and choose $w ^ { \dagger }$ to maximize $\mu _ { W _ { w ^ { * } } , Z _ { w } }$ . Choosing $w = w ^ { * }$ in $\{ W _ { w ^ { * } } , Z _ { w } \}$ reproduces the ordinary intervention by efectiveness and ctf-Rule 1. Hence:

$$
\mu _ { w ^ { * } } = \mu _ { W _ { w ^ { * } } , Z _ { w ^ { * } } } \leq \mu _ { W _ { w ^ { * } } , Z _ { w ^ { \dagger } } } .\tag{196}
$$

The action $\{ W _ { w ^ { * } } , Z _ { w ^ { * } } \}$ sends the same value to both children of $W .$ . In contrast, for $\{ W _ { w ^ { * } } , Z _ { w ^ { \dagger } } \}$ , the input received directly by $Y$ remains $w ^ { * }$ , while $Z$ may listen to $w ^ { \dagger }$ . The best reward cannot decrease, since the choice $\boldsymbol { w ^ { \intercal } } = \boldsymbol { w ^ { \ast } }$ is still available. A diferent value may improve the reward, depending on the underlying mechanisms.

For a concrete example, consider the following binary SCM with independent exogenous variables:

![](images/694ea17ebb9220abf3fef96ef6aa4b1b3a75b99045011fe3eb07d738bcfae436.jpg)

(197)

Then $w ^ { * } = 0$ and $w ^ { \dagger } = 1$ give:

$$
\mu _ { \emptyset } = 0 . 2 2 4 6 4 < \mu _ { w ^ { * } } = 0 . 2 3 6 8 < \mu _ { W _ { w ^ { * } } , Z _ { w ^ { \dagger } } } = 0 . 8 8 4 8 .\tag{198}
$$

The first increase is obtained by replacing the natural value of W with its best constant. The much larger second increase is obtained by keeping this zero at the input of $Y$ and sending one to the input of $Z _ { i }$ , as in Fig. 32b.

Counterfactual possibly-optimal minimal intervention sets. We now extend possible optimality from Sec. 3 to counterfactuals. The comparisons below concern counterfactual sets with each retained parent input independently selectable; equivalent sets are treated as the same action.

Definition 28 (Counterfactual possibly-optimal minimal intervention set). A counterfactual action $\mathbf { X } _ { * * }$ is a ctf-POMIS if there exists an SCM conforming to $\mathcal { G }$ such that $\mu _ { \mathbf { X } _ { \ast \ast } ^ { \ast } } > \mu _ { \mathbf { Z } _ { \ast } ^ { \ast } }$ ∗ for every non-equivalent counterfactual action $\mathbf { Z } _ { \ast \ast }$

![](images/e1a4a55200c4e93bedac07d2e3ead0d6ee2233fe7f36c2d7137addbe79d2f55d.jpg)  
Figure 33: In a Markovian causal diagram, intervening on all parents of Y sufices for optimality.

For a retained counterfactual set, one compatible SCM makes its best subscript assignment outperform every assignment to any set that is not equivalent to it. Several counterfactual sets may satisfy this condition, with diferent SCMs establishing their possible optimality. The agent must still learn their rewards to determine which action is best in the actual environment.

## 6.4.1 CTF-POMIS under Unconfounded Rewards

We first consider the case in which the exogenous inputs of Y are independent of those generating its other ancestors. For any upstream intervention, the reward mechanism receives values from its parents. These values may be random. The agent can instead supply a fixed assignment directly to the same parent inputs. If conditioning on the received values does not change the distribution of the reward’s exogenous inputs, the resulting mean is a weighted average of the means under these direct assignments. Choosing the best parent assignment is therefore at least as good as that average.

Fig. 33 illustrates this comparison. An upstream intervention changes the distribution of the inputs reaching Y. An intervention on $\mathtt { P a } ( Y ) _ { \mathcal { G } }$ can instead fix those inputs to the values most favorable to the reward.

Proposition 14. If Y is not confounded with $\operatorname { A n } ( Y ) _ { \mathcal { G } } \setminus \{ Y \}$ , the ordinary intervention set $\mathtt { P a } ( Y ) _ { \mathcal { G } }$ is the only ctf-POMIS up to equivalence.

The proof is given in Sec. C.1.2.

Corollary 8. If G is Markovian, meaning that it has no bidirected edges, the ordinary intervention set $\mathtt { P a } ( Y ) _ { \mathcal { G } }$ is the only ctf-POMIS up to equivalence.

Proof of Cor. 8. Prop. 14 applies because a Markovian diagram has no unobserved confounding. □

Thus, in the Markovian case it sufices to intervene on the parents of Y. With unobserved confounding, a parent counterfactual may carry information about the exogenous inputs of the reward. Fixing that parent to a constant removes this dependence, so the preceding averaging argument need not apply. For example, in Fig. 32, the best ordinary intervention on the parents $\mathtt { P a } ( Y ) _ { \mathcal { G } } = \{ W , X \}$ has mean 0.82, below the mean 0.8848 of $\{ W _ { w ^ { * } } , Z _ { w ^ { \dagger } } \}$

![](images/bb8dffc0d60f64bd185c77bd2917079742131301e76764ade68f19487aeb7ba5.jpg)  
Figure 34: (a) A causal diagram. (b)–(e) Counterfactual regime graphs. Light blue regions mark ${ \mathrm { M U C T s } } ,$ and green fill marks IB variables. The empty borders in (b)–(d) make these actions ctf-POMISs; the border {X} in (e) excludes that action.

## 6.4.2 Graphical Characterization of CTF-POMISs

Counterfactual regime graph. Recall that $\mathcal { M } _ { \mathbf { X } _ { * * } }$ fixes the selected parent inputs, and its graph $\mathcal { G } _ { \mathbf { X } }$ <sub>∗∗</sub> deletes the corresponding directed edges while retaining the bidirected edges. We now define $\begin{array} { r } { \mathcal { H } _ { \mathbf { X } _ { * * } } \triangleq \mathcal { G } _ { \mathbf { X } _ { * * } } [ \mathtt { A n } ( Y ) _ { \mathcal { G } _ { \mathbf { X } _ { * * } } } ] } \end{array}$

For concreteness, consider the causal diagram in Fig. 34a and the action $\mathbf { X } _ { * * } = \{ W _ { x } , Z _ { x ^ { \prime } } \}$ . The two counterfactuals assign separate inputs from X, giving the substitutions $f _ { W } ( X , { \mathbf { U } } _ { W } ) \Rightarrow$ $f _ { W } ( x , { \bf { U } } _ { W } )$ and $f _ { Z } ( X , { \bf U } _ { Z } ) \Rightarrow f _ { Z } ( x ^ { \prime } , { \bf U } _ { Z } )$ , respectively. Consequently, $X  W$ and $X  Z$ are deleted. The natural output of X no longer reaches Y. Hence, the ancestral restriction removes X, as illustrated in Fig. 34b.

Figs. 34b to 34e show how the same construction changes with the action. Under ${ \bf X } _ { * * } = \{ T _ { x } \}$ in Fig. 34c, only the input $X  W$ is fixed to produce $T _ { x } .$ , while the natural value of X still reaches Y through Z and R. Under $\mathbf { X } _ { * * } = \{ R _ { x } \}$ in Fig. 34d, only $X  Z$ is fixed, while $X  W  T  Y$ remains. Thus X is retained in both graphs. Under $\mathbf { X } _ { * * } = \{ R _ { z } \}$ in Fig. 34e, the substitution is instead $f _ { R } ( Z , { \bf U } _ { R } ) \Rightarrow f _ { R } ( z , { \bf U } _ { R } )$ . This removes $Z \to R$ and makes the natural output of Z irrelevant to Y. Its vertex is omitted, while the exogenous inputs of R remain unchanged.

MUCT and IB for counterfactual actions. We reuse the c-components, minimal unobserved confounders’ territory MUCT, and interventional border IB from Definitions 14 and 15. For $\mathbf { X } _ { * * } = \{ W _ { x } , Z _ { x ^ { \prime } } \}$ in Fig. 34b, the c-component containing Y is $\mathtt { c c } ( Y ) _ { \mathcal { H } _ { \mathbf { X } _ { \ast \ast } } } =$ $\{ Y , R , Z , W \}$ . Including descendants adds T, thus:

$$
\mathrm { F o r } ~ { \mathbf { X } } _ { * * } = \{ W _ { x } , Z _ { x ^ { \prime } } \} , \mathsf { M U C T } ( \mathcal { H } _ { { \mathbf { X } } _ { * * } } , Y ) = \{ W , Z , T , R , Y \}\tag{199}
$$

$$
\Rightarrow \mathbb { B } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) = \emptyset .\tag{200}
$$

In Figs. 34c and 34d, the retained bidirected connection $X  Z$ also includes X in the c-component of Y . Adding T by descendant closure therefore gives all six vertices, again leaving an empty border.

In contrast, consider $\mathbf { X } _ { * * } = \{ R _ { z } \}$ in ${ \mathrm { F i g . } }$ . 34e. After removing Z, we obtain $\subset \alpha ( Y ) _ { \mathcal { H } _ { \mathbf { X } _ { * * } } } =$ $\{ Y , R , W \}$ . Including descendants adds T, and neither closure operation adds X. Thus:

$$
\operatorname { F o r } \mathbf { X } _ { * * } = \{ R _ { z } \} , \mathsf { M U C T } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) = \{ W , T , R , Y \}\tag{201}
$$

$$
\Rightarrow \mathbb { B } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) = \{ X \} .\tag{202}
$$

The remaining input from X enters the territory through W. Fixing it to x adds the counterfactual $W _ { x }$ to the action. Since the territory’s exogenous inputs are independent of those generating X, conditioning on $X = x$ gives the same reward mean as supplying this value directly. Hence:

$$
\mu _ { R _ { z } } = \sum _ { x } \mathbb { E } [ Y _ { R _ { z } } \mid X = x ] P ( x )\tag{203}
$$

$$
= \sum _ { x } \mu _ { R _ { z } , W _ { x } } P ( x ) \leq \mu _ { \{ R _ { z } , W _ { x } \} ^ { * } } .\tag{204}
$$

This shows why a nonempty border provides an additional input choice that weakly improves the best reward. An empty border leaves no such external input to fix. These examples suggest that possible optimality can be characterized by the absence of an interventional border.

Theorem 14 (Graphical characterization of ctf-POMIS). A counterfactual action $\mathbf { X } _ { * * }$ is a ctf-POMIS with respect to ⟨G, Y⟩ if and only if $\vert \mathsf { B } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) = \emptyset$

The proof is given in Sec. C.1.3. When the border is nonempty, fixing its inputs gives a counterfactual action whose optimized mean is at least as large. When the border is empty, the retained mechanisms and shared exogenous inputs can make this counterfactual uniquely optimal up to equivalence.

When Y is unconfounded with its ancestors, its minimal UC-territory is {Y}. Its border is empty precisely when every incoming reward argument has been fixed. These counterfactuals are equivalent to intervening on $\mathtt { P a } ( Y ) _ { \mathcal { G } }$

## 6.5 Enumerating the CTF-POMIS Action Space

The graphical characterization in Thm. 14 identifies counterfactual sets that can be optimal.   
As in the preceding sections, their subscript assignments are then treated as actions.

Enumerating counterfactual sets. Alg. 11 enumerates each choice of children in $\mathcal { H } =$ $\mathcal { G } [ \mathtt { A n } ( Y ) _ { \mathcal { G } } ]$ . For a parent $W _ { i }$ , the choice $\mathbf { X } _ { W } \subseteq \mathtt { C h } ( W ) _ { \mathcal { H } }$ lists the mechanisms that will receive a fixed value in place of their input from W. The map $\mathtt { p a } [ X ]$ collects these choices at the receiving mechanism $f _ { X }$ . Thus the children choices enumerate the selected edges, and $\mathtt { p a } [ X ]$ records how to execute the resulting action. Deleting those edges gives the graph of the remaining causal dependence. Restricting this graph to the ancestors of Y then removes mechanisms that cannot afect the reward.

An accepted choice. Consider the causal diagram in Fig. 34a. Choose $\mathbf { X } _ { X } = \{ W , Z \}$ and choose no children for the other parents. The resulting map has pa $[ W ] = \mathsf { p a } [ Z ] = \{ X \}$ with every other entry empty. The algorithm deletes $X  W$ and $X  Z$ . Since X then has no directed path to $Y .$ , the ancestral restriction removes X, giving Fig. 34b. Both selected inputs remain in the specification: their receiving mechanisms, $f _ { W }$ and $f _ { Z } ,$ still afect Y. The rule for pruning a selected input therefore concerns its receiving mechanism, even when its source vertex has disappeared from the graph.

Algorithm 11: Constructing the ctf-POMIS action space.   
Input: A causal diagram G and a reward variable $\overline { { Y } }$   
Output: ctf-POMISs with their selected-input specifications   
$/ /$ Retain mechanisms that can affect the reward.   
1 Set $\begin{array} { r } { \mathcal { H } = \mathcal { G } [ \mathtt { A n } ( Y ) _ { \mathcal { G } } ] . } \end{array}$   
$/ /$ Choose which child mechanisms receive fixed inputs.   
2 for each children choice $( \mathbf { X } _ { W } ) _ { W \in \mathbf { V } ( \mathcal { H } ) \backslash \{ Y \} } \in \big \times _ { W \in \mathbf { V } ( \mathcal { H } ) \backslash \{ Y \} } 2 ^ { \mathtt { C h } ( W ) _ { \# } }$ do   
$/ /$ Record each selected input at its receiving mechanism.   
3 Initialize $\mathtt { p a } [ X ] = \varnothing$ for every $X \in \mathbf { V } ( \mathscr { H } )$   
4 for each W and each $X \in \mathbf { X } _ { W }$ do   
5 Set $\mathtt { p a } [ X ]  \mathtt { p a } [ X ] \cup \{ W \}$   
$/ /$ Cut selected edges and remove irrelevant receivers.   
6 Form $\mathcal { H } ^ { \prime }$ by deleting ${ \bar { W } } \to X$ from H whenever $W \in \mathsf { p a } [ X ]$   
7 Replace $\mathcal { H ^ { \prime } }$ by $\mathcal { H ^ { \prime } } [ \mathtt { A n } ( Y ) _ { \mathcal { H ^ { \prime } } } ]$   
8 Set $\mathtt { p a } [ X ] = \varnothing$ for each $\mathbf { \Delta } _ { X } \dot { \notin } \mathbf { V } ( \mathbf { \mathcal { H } ^ { \prime } } )$   
$/ /$ Keep exactly the choices satisfying Thm. 14.   
9 if IB $\mathsf { \Omega } ^ { \prime } ( \mathcal { H } ^ { \prime } , Y ) \neq \emptyset$ then   
10 continue   
$/ /$ Leave retained input values separately selectable.   
11 Initialize the selected-input specification of $\mathbf { X } _ { * * } .$   
12 for each $X \in \mathbf { V } ( \mathcal { H } ^ { \prime } )$ in topological order do   
13 for each $W \in { \mathfrak { p a } } [ X ]$ do   
14 Assign a separately selectable value to the argument W of $f _ { X }$   
15 Retain every other parent input and every exogenous input of $f _ { X }$   
$/ /$ Return each retained specification only once.   
16 if this selected-input specification has not been returned then   
17 yield $\mathbf { X } _ { * * }$ with $\mathtt { p a } [ X ]$

The retained graph has an empty interventional border, so this choice is returned. Assigning x to the input of $f _ { W }$ and $x ^ { \prime }$ to the input of $f _ { Z }$ gives the counterfactuals $\{ W _ { x } , Z _ { x ^ { \prime } } \}$ . These two values are independently selectable.

A rejected choice and a redundant specification. Now choose only ${ \bf X } _ { Z } = \{ R \}$ . This gives ${ \tt p a } [ R ] = \{ Z \}$ and, after assigning its value, the action $\{ R _ { z } \}$ . Cutting $Z \to R$ removes $Z$ from the reward ancestry but retains its fixed input at $f _ { R } .$ . The resulting graph is Fig. 34e. Since its border is not empty, the algorithm rejects this counterfactual. Adding the choice $\mathbf { X } _ { X } = \{ W \}$ fixes the remaining input from this border into the territory. The resulting counterfactual set $\{ W _ { x } , R _ { z } \}$ has empty border and is returned.

If a retained specification passes the condition of Thm. 14, the algorithm returns it only once, regardless of how many original choices produce it. The returned action $\mathbf { X } _ { * * }$ retains its selected-input specification and may contain nested counterfactuals. All subscript assignments of these returned counterfactuals form $\mathcal { A } _ { \mathrm { C T F } } ^ { \star }$

Theorem 15. Alg. 11 returns exactly the ctf-POMISs with respect to $\langle \mathcal { G } , Y \rangle$

Proof of Thm. 15. Every selection of parent inputs occurs in the children-choice loop. Deleting its selected edges and restricting to the ancestors of Y gives exactly $\mathcal { H } _ { \mathbf { X } _ { * * } }$ . Removing fixed inputs at mechanisms outside this ancestral graph preserves the reward counterfactual on every unit, and each remaining selected input can afect Y along an uncut path. The characterization in Thm. 14 therefore accepts exactly the ctf-POMISs. Identifying repeated specifications changes no available action. □

Corollary 9. No two counterfactual sets returned by Alg. 11 are equivalent.

Proof of Cor. 9. The final check compares all retained entries $\mathtt { p a } [ X ]$ and returns each map once. It remains to show that distinct retained maps are not equivalent.

Choose an edge selected in one map but not the other. Its receiving child has a retained directed path to Y in the first map. If the other map retains this entire path, the chosen edge is followed by a path retained in both maps. Otherwise, take the last edge of that path selected by the other map. Again, the edge is fixed in one map and retained in the other, and its remaining path to Y is retained in both.

Choose a binary SCM in which the source of this edge outputs a private $\mathtt { B e r n } ( \mathtt { \mathtt { i } } )$ variable, and each mechanism along the edge and remaining path copies the preceding variable. Other inputs do not afect these mechanisms. Fixing the edge then gives a constant reward, whereas retaining it gives a non-constant reward on the same units, regardless of the other selected values. Thus the two maps cannot be equivalent. □

Corollary 10. For every SCM compatible with G:

$$
\operatorname* { m a x } _ { \mathbf { X } _ { * * } \in \mathcal { A } _ { \mathrm { C T F } } ^ { \star } } \mu _ { \mathbf { X } _ { * * } } = \operatorname* { m a x } _ { \mathbf { X } _ { * * } \in \mathcal { A } _ { \mathrm { C T F } } } \mu _ { \mathbf { X } _ { * * } } .\tag{205}
$$

The proof is given in Sec. C.1.4.

Computational cost. The initial ancestral restriction takes time linear in the size of G. Let n and m be the numbers of vertices and edges in H, and let mˆ count its directed edges. The children-choice loop examines $2 ^ { \hat { m } }$ choices. For each choice, deleting edges, finding reward ancestors, and computing the territory and border take $O ( n + m )$ time.

If κ distinct counterfactual sets are returned, comparing a retained specification against all previously returned specifications takes at most $\mathcal { O } ( \kappa ( n + m ) )$ time per choice. Thus a direct implementation takes $\mathcal { O } ( 2 ^ { \hat { m } } ( \kappa + 1 ) ( n + m ) )$ time and $\mathcal { O } ( ( \kappa + 1 ) ( n + m ) )$ space, including the returned specifications. These bounds enumerate counterfactual sets; assigning their constant values to form bandit arms is a subsequent step.

## 6.6 Experiments

We evaluate how the counterfactual action space and its graphical reduction afect bandit learning. Each task uses a fixed binary SCM with independent Bernoulli exogenous variables, and the agent observes the reward of its selected action on a fresh unit at each round. Here the agent is given the causal diagram and may perform the counterfactual experiments admitted by $A _ { \mathrm { C T F } }$ . Task 1 uses the causal diagram in Fig. 32a, and Task 2 uses the diagram in Fig. 29b. Tasks $\mathrm { 3 ( a ) }$ and 3(b) share the diagram in Fig. 35. We use one SCM for each of the first two tasks and two SCMs for Task 3, giving four experimental cases. Their complete specifications appear in Sec. C.2.

![](images/8b27bd58ef6aa45da1cf09ea6880ae8aa73982bce6b8a0f0ebf1f6f52a65f306.jpg)  
Figure 35: Causal diagram for Tasks $\mathrm { 3 ( a ) }$ and $\mathrm { 3 ( b ) }$ . The two tasks use diferent structural functions and exogenous distributions on this graph.

Table 11: Candidate action spaces. Each strategy column gives the number of sets / arms. For the counterfactual strategies, sets are distinct selections of reward-relevant parent inputs. Tasks 3(a) and 3(b) use the same causal diagram and therefore have the same counts.
<table><tr><td>Task</td><td>|V|</td><td>T</td><td>CTF-POMIS</td><td>CTF-MIS</td><td>POMIS</td></tr><tr><td>1</td><td>4</td><td>10k</td><td>2/8</td><td>8 /  21</td><td>2/6</td></tr><tr><td>2</td><td>5</td><td>10k</td><td>6/36</td><td>17 / 67</td><td>4 / 16</td></tr><tr><td>3(a), 3(b)</td><td>7</td><td>100k</td><td>5 / 52</td><td>35 / 187</td><td>5 / 36</td></tr></table>

Candidate action spaces. We compare three strategies. The CTF-POMIS strategy uses the actions returned by Alg. 11; each selected parent argument has its own binary value. Identical choices of fixed parent inputs and their assigned values are counted once.

The CTF-MIS baseline covers all realizable ctf-MIS actions, including nested counterfactuals. It enumerates the selected parent inputs as in Alg. 11, but omits the interventional border test. Every selection entering a mechanism required to evaluate Y is retained, together with all binary assignments. This includes ordinary interventions and the empty action. By Prop. 13, every realizable nested action can be evaluated through these parent-input choices; removing redundant outer assignments preserves its reward. Diferent expressions giving the same fixed inputs and values are counted once. Both counterfactual strategies count the same choice of fixed inputs and values only once. Their comparison measures the exploration savings from the graphical ctf-POMIS condition over the complete nested ctf-MIS baseline.

Finally, POMIS retains all assignments to ordinary POMISs, as in Sec. 3. Table 11 reports the numbers of sets and arms.

Solvers and evaluation. Each action space is paired with Thompson sampling (TS) (Thompson, 1933) and Bernoulli KL-UCB (Garivier and Capp´e, 2011; Capp´e et al., 2013), using the same settings as Sec. 4.3. TS uses a Beta(1, 1) prior for each arm. KL-UCB initially selects each arm once and subsequently uses $f ( s ) = \log s + 3$ log log s for $s \geq 3$ , with $f ( s ) = 1$ otherwise, where s is the number of completed rounds. Thus, after initialization, the decision at round t uses $f ( t - 1 )$ . Tasks 1 and 2 use $T = 1 0 { , } 0 0 0$ rounds, and Tasks 3(a) and 3(b) use $T = 1 0 0 , 0 0 0$ . Each of the 24 task–strategy–solver configurations is repeated 1,000 times.

![](images/829c20053b77ed9d5be3693902cd7f570127080ba3fcac5fe33c385933ab2ffb.jpg)  
(a) Task 1

![](images/2ed7ce3fc9d929e1bb02351b60201ef1be03db650e8df093fdfef71b909b5060.jpg)  
(b) Task 2

![](images/fb3adecbc28283580491169aba0c5cb71693b5f164a292c8147c2797c0cae899.jpg)  
(c) Task 3(a)

![](images/298b97b672a2b2506a6f3d913341b1f070322a90ce4e3de6007ccfdd5463dde9.jpg)  
(d) Task 3(b)  
Figure 36: Curves show mean cumulative regret over 1,000 independent runs; shaded regions show one standard deviation. Every strategy uses the same counterfactual optimum as its regret benchmark.

The reward means are computed exactly from the fixed SCMs by summing over their exogenous assignments. These means generate Bernoulli rewards and evaluate cumulative regret; the solvers use only the selected arms and observed rewards. All strategies are evaluated against the same optimal counterfactual mean $\mu \mathbf { { X } } _ { * * } ^ { \star }$ , using the cumulative regret defined earlier in this section. In particular, the ordinary POMIS baseline is compared with the counterfactual optimum even when its own best arm has a smaller mean. Curves in Fig. 36 show mean cumulative regret, and shaded regions show one population standard deviation across repeated runs. Displayed reward means in Sec. C.2 are rounded to at most three decimal places.

Tasks 1 and 2: enlarging and reducing the action space. Task 1 corresponds to the causal diagram in Fig. 32. Its best ordinary intervention has mean 0.82, while separately choosing the two values supplied by W raises the optimal mean to 0.8848. The CTF-POMIS strategy retains this optimum with 8 arms, compared with 21 for the CTF-MIS baseline. Task 2 uses the causal diagram in Fig. 29b, where realizability constrains the simultaneous use of counterfactuals. For its fixed SCM, the optimal counterfactual and ordinary means are approximately 0.6759 and 0.5797, respectively. Both counterfactual strategies retain the former mean, while CTF-POMIS reduces the number of arms from 67 to 36.

Table 12: Final cumulative regret (mean ± standard deviation across 1,000 runs). Tasks 1 and 2 use T = 10,000; Tasks 3(a) and 3(b) use $T = 1 0 0 , 0 0 0$ . Bold entries mark the lowest mean for each task–solver pair.
<table><tr><td>Task</td><td>Solver</td><td>CTF-POMIS</td><td>CTF-MIS</td><td>POMIS</td></tr><tr><td rowspan="2">1</td><td>TS</td><td> ${ \bf 5 1 . 4 } \pm 2 5 . 7$ </td><td> $1 3 6 . 2 { \scriptstyle \pm 3 1 . 7 }$ </td><td> $6 6 6 . 2 { \scriptstyle \pm 3 . 5 }$ </td></tr><tr><td>KL-UCB</td><td> $\mathbf { 1 1 7 . 7 { \pm } } 2 0 . 2 $ </td><td> $3 1 4 . 2 { \scriptstyle \pm 3 3 . 6 }$ </td><td> $6 8 5 . 9 { \scriptstyle \pm 5 . 5 }$ </td></tr><tr><td rowspan="2">2</td><td>TS</td><td> $\mathbf { 3 7 4 . 7 \pm } 4 5 . 2$ </td><td> $5 7 0 . 3 { \scriptstyle \pm 4 5 . 7 }$ </td><td> $1 2 5 5 . 1 { \scriptstyle \pm 7 7 . 4 }$ </td></tr><tr><td>KL-UCB</td><td> ${ \bf 7 0 6 . 1 \pm 4 6 . 9 }$ </td><td> $9 0 1 . 4 { \scriptstyle \pm 3 9 . 7 }$ </td><td> $1 4 8 8 . 8 { \pm } 4 7 . 2$ </td></tr><tr><td rowspan="2"> $\mathrm { 3 ( a ) }$ </td><td>TS</td><td> $\mathbf { 1 2 5 2 . 4 } { \scriptstyle \pm 1 6 1 . 5 }$ </td><td> $4 1 9 5 . 1 { \scriptstyle \pm 2 5 6 . 2 }$ </td><td> $3 5 0 2 . 2 { \scriptstyle \pm 1 7 1 . 9 }$ </td></tr><tr><td>KL-UCB</td><td> $\mathbf { 3 2 7 5 . 9 } \pm 1 8 5 . 2$ </td><td> $8 9 2 4 . 4 { \scriptstyle \pm 2 4 6 . 9 }$ </td><td> $5 1 6 6 . 2 { \scriptstyle \pm 1 8 7 . 9 }$ </td></tr><tr><td rowspan="2">3(b)</td><td>TS</td><td> $1 9 7 9 . 8 { \scriptstyle \pm 2 8 2 . 8 }$ </td><td> $4 4 9 5 . 5 { \pm } 1 6 4 . 4$ </td><td> $\mathbf { 1 4 3 2 . 5 \pm } 2 5 2 . 2$ </td></tr><tr><td>KL-UCB</td><td> $3 9 8 5 . 4 { \scriptstyle \pm 1 7 4 . 5 }$ </td><td> $5 2 2 9 . 9 { \scriptstyle \pm 5 7 . 0 }$ </td><td> $\mathbf { 3 1 4 3 . 4 } \pm 1 8 0 . 0$ </td></tr></table>

Task 3: whether ordinary interventions attain the optimum. The two sevenvariable SCMs share a causal diagram and therefore the same candidate action spaces: 52 arms for CTF-POMIS, 187 for the CTF-MIS baseline, and 36 for POMIS. Their structural functions determine whether ordinary interventions can attain the counterfactual optimum. In Task 3(a), the optimal counterfactual and ordinary means are approximately 0.6180 and 0.5932, respectively, leaving a positive gap of about 0.0248 per round. In Task 3(b), all three spaces attain the same optimal mean, approximately 0.4365. The latter case isolates the exploration cost of the larger counterfactual spaces when ordinary interventions already sufice.

For a strategy restricted to ordinary interventions, every selected arm has mean at most $\mu _ { \mathbf { x } ^ { \star } }$ Consequently, $R _ { T } ^ { A _ { \mathrm { C T F } } } \geq T ( \mu _ { \mathbf { X } _ { * * } ^ { \star } } - \mu _ { \mathbf { X } ^ { \star } } )$ . The positive gaps in Tasks 1, 2, and 3(a) therefore impose a linear contribution to the ordinary strategy’s regret, regardless of how quickly it learns its own best arm. This contribution vanishes in Task 3(b).

Observed learning performance. Table 12 reports the simulation results. The CTF-POMIS strategy has lower mean cumulative regret than the CTF-MIS baseline in all eight task–solver comparisons. In Task 1, its regret is 37.7% and 37.5% of the baseline under TS and KL-UCB, respectively. In Task 3(a), these ratios are 29.9% and 36.7%. The CTF-POMIS strategy also has lower mean regret than ordinary POMIS under both solvers in Tasks 1, 2, and 3(a). Ordinary POMIS has the lowest mean regret under both solvers in Task 3(b), where it retains an optimal arm with fewer candidates. These results illustrate both the gain from additional counterfactual actions and the exploration savings from graphical reduction on the fixed SCMs considered here.

## 7 Conclusion

Across the three settings studied in this paper, causal structure clarifies which decisions merit exploration even when their rewards remain unknown. Under structural uncertainty, graphical criteria support action reduction over compatible models. Information from source environments constrains target rewards through transferable mechanisms and dominance relations. Counterfactual actions expand the available decisions when the required experiments are realizable. These results separate the construction of an action space from learning the rewards of its members, while preserving an optimal action under the stated assumptions.

The scope of these guarantees is determined by their inputs and assumptions. The Markov equivalence experiments use oracle PAGs, and the transportability analysis assumes correct structural information and valid population reward bounds. The counterfactual formulation uses finite endogenous domains and specified experimental capabilities. The numerical studies use fixed finite SCMs and do not establish performance across arbitrary environments. Moreover, graphical action reduction does not eliminate the cost of constructing the reduced space: candidate enumeration and searches over compatible structures can still grow exponentially.

Several questions remain open. A direct graphical criterion for DMIS membership in PAGs and the complexity of deciding that membership deserve further study, as discussed in Sec. 4.1.2. Another direction is to propagate uncertainty from learned causal structures and estimated source distributions into action pruning and online confidence bounds. Extending the analysis to richer domains and to settings in which counterfactual experiments have unequal costs would help connect these guarantees to a wider range of decision problems.

## Appendix A. Supporting Results for Markov Equivalence

## A.1 Background Results

Throughout this section, P is a PAG without selection bias, $\mathcal { M } \in [ \mathcal { P } ]$ is a represented MAG, and G is a causal diagram whose corresponding MAG is M. We denote the observed vertex set by V and take $\mathbf { A } \subseteq \mathbf { V }$ whenever an induced graph is used. Visibility in an induced graph is interpreted with the full graph’s represented model class held fixed, as defined in the main text. A PMG Q is assumed to be obtained from P by valid local transformations, applying the orientation rules in Thm. 7 until no further orientation is possible after each transformation.

## A.1.1 Background Results in Zhang (2006, 2008a,b)

Lemma 11 (A lemma in the proof of Lemma 5.1.7 in Zhang (2006); see also the proof of Lemma 7 in Jaber et al. (2018)). $I f p = \langle X , \cdots , Z , V , Y \rangle$ is a discriminating path from X to Y for V in a MAG M, and the corresponding subpath between X and V in P is (also) a collider path, then the path corresponding to p in P is also a discriminating path for V .

Lemma 12 (Lemma A.1 in Zhang (2008a) and Lemma 5 in Jaber et al. (2018)). For any three nodes A, B, C ∈ A, if $A * \to B \circ - * C$ in $\mathcal { P } [ \mathbf { A } ]$ , then there is an edge between A and C with an arrowhead at C, namely, A∗→ C. Furthermore, if the edge between A and B is $A  B$ , then the edge between A and C is either $A  C$ or A◦→ C (i.e., it is not $A  C )$

Lemma 13 (Lemma 3.3.2 in Zhang (2006)). For any two distinct vertices A and B connected by a circle path in P, the following properties hold:

1. If there is an edge between A and B, the edge is not into A or B;

2. For any other node C, C ∗→ A if and only if C ∗→ B. Furthermore, $C  A$ if and only if $C  B$

Lemma 14 (Theorem 2 in Zhang (2008a)). Let M be the graph resulting from the following procedure applied to P.

Step 1. Replace all partially directed edges ( ◦→) in P with directed edges (→).

Step 2. Orient every circle component of P into a DAG with no unshielded colliders.

Then, the resulting graph M conforms to P.

Lemma 15 (Lemma B.1 in Zhang (2008a)). If p is a possibly directed path from A to B in P, then some subsequence of p forms an uncovered possibly directed path from A to B in P.

Lemma 16 (Lemma B.2 in Zhang (2008a)). $I f p = \langle V _ { 0 } ( = A ) , \cdots , V _ { n } ( = B ) \rangle$ with $n \geq 2$ is an uncovered possibly directed path from A to B in P, and $V _ { i - 1 } *  V _ { i }$ for some $i \in \{ 1 , \cdots , n \}$ then $V _ { j - 1 } \to V _ { j }$ for all $j \in \{ i + 1 , \cdots , n \}$

Lemma 17 (Lemma B.4 in Zhang (2008a)). If there is a possibly directed path from A to B in P, then the edge between A and B, if any, is not into A.

Lemma 18 (Lemma B.5 in Zhang (2008a)). If there is a possibly directed path from A to B that is into B in P, then every uncovered possibly directed path from A to B is into B.

Lemma 19 (Lemma B.7 in Zhang (2008a)). If there is a circle path between two adjacent vertices in P, then the edge between the two vertices is a circle edge.

Lemma 20 (Lemma 9 in Zhang (2008b)). Let G be a DAG over V ∪ U, and M be the MAG over V that represents the DAG. For any $A , B \in \mathbf { V } , \ i f \ A \in \mathtt { A n } ( B ) \mathcal { G } .$ , and there is an inducing path relative to U between A and B that is into A in $\mathcal { G }$ , then there is a directed edge A → B in M that is invisible.

Lemma 21 (Lemma 10 in Zhang (2008b)). Let M be any MAG over V, and $A  B$ be any directed edge in M. If A → B is invisible in M, then there is a DAG G over V ∪ U whose MAG is M in which A and B share a latent parent. That is, there is a latent variable U in the DAG such that $A \left. U \right. B$ is a subgraph of the DAG G.

Lemma 22 (Contrapositive of Corollary 13 in Zhang (2008b)). Let X, Y ∈ V be distinct, let $\mathbf { Z } \subseteq \mathbf { V } \setminus \{ X , Y \}$ , and let $\mathbf { X } \subseteq \mathbf { V }$ . If there is a d-connecting path between X and Y given Z in $\mathcal { G } _ { \overline { { \mathbf { X } } } }$ , then there is an m-connecting path between X and Y given Z in $\mathcal { M } _ { \overline { { \mathbf { X } } } }$

## A.1.2 Background Results in Maathuis and Colombo (2015)

Lemma 23 (Lemma 7.6 in Maathuis and Colombo (2015)). Let P be a PAG with exactly k incident edges carrying an arrowhead at X, where $k \geq 0$ . There exists a MAG M ∈ [P] with exactly the same k arrowheads at $X ;$ in particular, no circle at X is replaced by an arrowhead.

## A.1.3 Background Results in Perkovic et al.<sup>´</sup> (2018)

Lemma 24 (Lemma 48 in Perkovi´c et al. (2018)). Fix a vertex X in a PAG P. Construct $\mathcal { M } \in [ \mathcal { P } ]$ by replacing all ◦→ edges with directed edges and orienting every circle component as a DAG without unshielded colliders, with every circle edge incident to X oriented out of X. Then every edge that is $X \circ - \circ Y , X \circ - Y$ , or an invisible $X  Y$ in P becomes an invisible $X  Y$ in M.

## A.1.4 Background Results in Jaber et al. (2018, 2022)

Lemma 25 (Proposition 1 in Jaber et al. (2018)). Let G be any causal diagram consistent with P, and let X, Y ∈ A. If $X \in \mathsf { A n } ( Y ) _ { \mathcal { G } [ \mathbf { A } ] }$ , then $X \in \mathtt { P o s s A n } ( Y ) _ { \mathcal { P } [ \mathbf { A } ] }$

Lemma 26 (Lemma 4 in Jaber et al. (2018)). For every directed edge $X  Y$ in $\mathcal { P } [ \mathbf { A } ]$ , if it is visible in P, then it remains visible in $\mathcal { P } [ \mathbf { A } ]$

Lemma 27 (Proposition 2 in Jaber et al. (2018)). Let G be any causal diagram consistent with P, and let X, Y ∈ A. If $X \in \mathsf { c c } ( Y ) _ { \mathcal { G } [ \mathbf { A } ] }$ , then $X \in \mathsf { p c } ( Y ) _ { \mathcal { P } [ \mathbf { A } ] }$

Algorithm 12: Partial topological order (pto) (Jaber et al., 2018).   
Input: P, A ⊆ V(P)   
Output: Partial topological order over $\mathcal { P } [ \mathbf { A } ]$   
1 while there exists a circle component B in P[A] such that every edge between B   
and A \ B has an arrowhead at its endpoint in B do   
2 Extract B from $\mathcal { P } [ \mathbf { A } ]$   
3 A ← A \ B   
4 end   
5 The partial order is $\mathbf { B } ^ { 1 } \prec \dots \prec \mathbf { B } ^ { m }$ in reverse order of the circle component   
extraction, i.e., B<sup>1</sup> is the last circle component extracted and B<sup>m</sup> is the first.

Lemma 28 (Proposition 4 in Jaber et al. (2018)). pto (Alg. 12) is sound over $\mathcal { P } [ \mathbf { A } ]$ 2 in the sense that the partial order is valid with respect to G[A], for any causal diagram G represented by P.

Lemma 29 (Lemma 6 in Jaber et al. (2018)). If there is a bidirected path p between X and Y in G, then there is a path p<sup>′</sup> between X and Y in M over a subsequence of p such that:

1. all non-endpoint nodes are colliders; and

2. all directed edges on p<sup>′</sup> are invisible.

Lemma 30 (Lemma 7 in Jaber et al. (2018)). If there is a path p between X and Y in M such that:

1. all non-endpoint nodes are colliders; and

2. all directed edges, if any, are invisible,

then there is a path p<sup>∗</sup> between X and Y in P over a subsequence of p such that:

1. all non-endpoint nodes along the path are definite colliders; and

2. none of the edges are visible.

Lemma 31 (Lemma 6 in Jaber et al. (2019)). In M[A], the following property holds:

For any three vertices A, B, C ∈ A, if A∗→ B → C and both edges are invisible, then we have A∗→ C and the edge is invisible.

Lemma 32 (Lemma 18 in Jaber et al. (2022)). In P[A], the following property holds:

For any three vertices A, B, C ∈ A, if A∗→ B ?→ C and both edges are invisible, then we have A∗→ C and the edge is invisible.

## A.1.5 Background Results in Wang et al. (2023)

Proposition 15 (Lemma 4 in Wang et al. (2023)). Consider a MAG M consistent with both Q and a valid local transformation $\mathbf { C } _ { X } ^ { \mathcal { Q } }$ . Then $Y \in \mathsf { P o s s D e } ( X ) _ { \mathcal { Q } \backslash \mathbf { C } _ { X } ^ { \mathcal { Q } } }$ if and only if $Y \in \mathsf { D e } ( X ) _ { \mathcal { M } }$

Lemma 33 (Lemma 2 in Wang et al. (2023)). Let Q be obtained from a PAG by valid local transformations, applying the orientation rules in Thm. 7 to closure after each transformation. Every possibly directed path in $\mathcal { Q }$ has an uncovered possibly directed subsequence with the same endpoints.

Lemma 34 (Lemma 16.1 in Wang et al. (2023)). Let $\mathcal { Q }$ be obtained from a PAG by valid local transformations and orientation closure, as specified above. Replace every ◦→ edge in Q with a directed edge and orient each circle component into a DAG without unshielded colliders. Every graph obtained by these simultaneous choices is a MAG in [Q].

## A.2 Omitted Proofs

## A.2.1 Constructing a Binary SCM

Lemma 35 (Binary SCM construction). Let $\mathcal { G }$ be a causal diagram. There exists a binary SCM ${ S ^ { * } }$ conforming to $\mathcal { G }$ with the following property. $I f \textbf { Z } \subsetneq \textbf { X } \subseteq \textbf { V } \backslash \{ Y \}$ and some $W \in { \bf X } \setminus { \bf Z }$ has a proper directed path to Y with respect to X, then $\mu _ { \mathbf { X } = \mathbf { 1 } } ^ { S ^ { \ast } } > \mu _ { \mathbf { Z } = \mathbf { 1 } } ^ { S ^ { \ast } }$

Proof of Lemma 35. We first specify one SCM using only the causal diagram $\mathcal { G } ;$ none of its functions or distributions will depend on the intervention sets being compared. Here, binary refers to the observed variables; the exogenous variables need not be binary. Give each observed vertex V a private exogenous variable $U _ { V } \sim \mathbb { U n i f } ( 0 , 1 )$ , which appears directly only in the equation for V. For each bidirected edge $e ,$ introduce a separate exogenous variable $L _ { e } \sim \mathtt { B e r n } ( \frac { 1 } { 2 } )$ , which appears in the equations for both endpoints of $e .$ Thus each $U _ { V }$ is used only for V, whereas the $L _ { e }$ variables represent the unobserved confounders shown by the bidirected edges. Choose all these exogenous variables to be mutually independent.

In the following sums, e ranges over the bidirected edges of G. For each V, choose strictly positive coeficients $\alpha _ { V } , \beta _ { P V } , \gamma _ { e V }$ such that $\begin{array} { r } { \alpha _ { V } + \sum _ { P \in \mathtt { P a } ( V ) _ { \mathcal { G } } } \beta _ { P V } + \sum _ { e : V \in e } \gamma _ { e V } < 1 } \end{array}$ . For example, assign each coeficient in this inequality the value $\frac { 1 } { 2 ( 1 + | \mathsf { P a } ( V ) g | + | \{ e : V \in e \} | ) }$ . Their sum is then $\frac { 1 } { 2 }$ . Define the structural equation for V by:

$$
V = f _ { V } ( \mathsf { P a } ( V ) _ { \mathcal { G } } \cup \{ U _ { V } \} \cup \{ L _ { e } \} _ { e : V \in e } )\tag{206}
$$

$$
= 1 \big \{ U _ { V } \le \alpha _ { V } + \mathrm { ~ \displaystyle ~ \sum ~ } \beta _ { P V } P + \mathrm { ~ \displaystyle ~ \sum ~ } \gamma _ { e V } L _ { e } \big \} .\tag{207}
$$

$$
P \in \mathsf { P a } ( V ) _ { \mathcal { G } } \qquad e : V \in e
$$

The argument set in the first line specifies the variables whose values are supplied to $f _ { V }$ Every parent value and every $L _ { e }$ is either zero or one. Consequently, the threshold on the right-hand side is at least $\alpha _ { V } > 0$ and at most the sum of the coeficients, which is strictly less than one. Acyclicity lets us evaluate these equations in a topological order, with each parent evaluated before its child, and every resulting observed variable is binary.

The equation for V contains exactly its directed parents in ${ \mathcal { G } } ,$ each with a positive coeficient. The only exogenous variables shared by diferent equations are the $L _ { e }$ variables, one for each specified bidirected edge. Equivalently, each bidirected edge can be represented by an unobserved common parent of its two endpoints. Adding the directed edges $U _ { V } \to V$ for the separate exogenous variables gives the DAG underlying this causal diagram. The SCM therefore conforms to ${ \mathcal { G } } .$ . Fix this SCM as ${ \boldsymbol { S } } ^ { * }$ for all comparisons below.

Expected values under an intervention. For any intervention set S, write $\eta _ { V } ^ { \mathbf { S } } = \mathbb { E } [ V \ |$ $d o ( \mathbf { S } = \mathbf { 1 } ) ]$ ] in the fixed SCM. If $V \in \mathbf { S }$ , its equation is replaced by the constant one, so $\eta _ { V } ^ { \mathbf { S } } = 1 . \operatorname { I f } V \notin \mathbf { S }$ , its original mechanism remains. The private variable $U _ { V }$ is independent of both the parent values and the shared variables appearing in that equation: it is independent of all other exogenous variables, and acyclicity prevents any parent of $V$ from depending on $U _ { V }$ . Conditional on those parent values and shared variables, the probability that $V = 1$ thus equals its threshold.

An intervention changes structural equations but leaves the exogenous distributions unchanged. Hence each $L _ { e }$ still has mean $\frac { 1 } { 2 }$ . For $V \not \in { \bf S }$ , taking the expectation of the threshold gives:

$$
\eta _ { V } ^ { \mathbf { S } } = \alpha _ { V } + \sum _ { P \in \mathbb { P } \mathrm { \Xi } ( V ) _ { \mathcal { G } } } \beta _ { P V } \eta _ { P } ^ { \mathbf { S } } + \frac { 1 } { 2 } \sum _ { e : V \in e } \gamma _ { e V } , \qquad 0 < \eta _ { V } ^ { \mathbf { S } } < 1 .\tag{208}
$$

Only the independence of the private variable $U _ { V }$ was needed for the conditional probability calculation. In particular, a parent of V may depend on a shared $L _ { e }$ and therefore be correlated with it. Linearity of expectation still gives the displayed formula; it does not require independence between the terms in the threshold. The two strict inequalities hold because the threshold stays strictly between zero and one for every assignment of the parents and shared variables.

Comparing two intervention sets. Now choose any $\mathbf { Z } \subsetneq \mathbf { X }$ satisfying the condition in the lemma, and let $\delta _ { V } = \eta _ { V } ^ { \mathbf { X } } - \eta _ { V } ^ { \mathbf { Z } }$ . There are three cases. If $V \in \mathbf { Z }$ , both interventions set V to one, thus $\delta _ { V } = 0 . \mathrm { I f } V \in \mathbf { X } \backslash \mathbf { Z }$ , only the first intervention sets V to one, whereas its mean under the second is strictly below one; thus $\delta _ { V } = 1 - \eta _ { V } ^ { \mathbf { Z } } > 0$ . Finally, if V $\not \in \mathbf { X } ,$ its structural equation is unchanged under both interventions. Subtracting the two expectation formulas cancels the constant and the terms involving the shared exogenous variables, leaving:

$$
\delta _ { V } = \sum _ { P \in \mathsf { P a } ( V ) g } \beta _ { P V } \delta _ { P } .\tag{209}
$$

In a topological order, the diferences for the parents have already been evaluated when this last formula is used. The first two cases give nonnegative diferences directly, and the last case is a sum of nonnegative diferences for the parents with positive coeficients. Induction therefore gives $\delta _ { V } \geq 0$ at every vertex.

By the condition in the lemma, some $W \in { \bf X } \setminus { \bf Z }$ has a proper directed path $W ( = V _ { 0 } ) $ $V _ { 1 } \to \cdots \to V _ { k } ( = Y )$ with respect to X. Every vertex after W on this path lies outside X and hence also outside Z. The displayed equation for the diference in expected values therefore applies at each such vertex. The terms for its other parents are nonnegative, thus $\delta _ { V _ { i } } \geq \beta _ { V _ { i - 1 } V _ { i } } \delta _ { V _ { i - 1 } }$ for $i = 1 , \cdots , k .$ . Applying these inequalities successively yields the explicit

bound:

$$
\delta _ { { \cal Y } } \geq \delta _ { W } \prod _ { i = 1 } ^ { k } \beta _ { V _ { i - 1 } V _ { i } } = ( 1 - \eta _ { W } ^ { \mathbf { Z } } ) \prod _ { i = 1 } ^ { k } \beta _ { V _ { i - 1 } V _ { i } } > 0 .\tag{210}
$$

The final inequality is strict because $\eta _ { W } ^ { \mathbf { Z } } < 1$ and every coeficient on the path is positive. Since $\delta _ { Y } = \mu _ { \mathbf { X } = \mathbf { 1 } } ^ { S ^ { * } } - \mu _ { \mathbf { Z } = \mathbf { 1 } } ^ { S ^ { * } }$ , this is the required reward comparison. The argument applies to every pair X, Z satisfying the condition in the lemma within the same fixed SCM, and all reward means are bounded because Y is binary. □

## A.2.2 Constructing a Causal Diagram from a MAG

Lemma 36 (Causal diagram with the same MUCT and IB). Let M be a MAG and let $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ . There exists a causal diagram G whose corresponding MAG is M and whose directed edges are exactly those of M such that MUCT $( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \mathsf { M U C T } ( \mathcal { M } , Y , \mathbf { X } )$ and ${ \sf I B } ( \mathcal G _ { \overline { { \mathbf { X } } } } , Y ) = { \sf I B } ( \mathcal M , Y , \mathbf X )$

Proof of Lemma 36. Let $\mathbf { H } = \mathrm { A n } ( Y ) _ { \mathcal { M } \backslash \mathbf { X } }$ and $\mathbf { R } = \mathsf { M U C T } ( \mathscr { M } , Y , \mathbf { X } )$ . The set H contains exactly the vertices that can reach Y by a directed path avoiding X. The set R is the MUCT of the MAG that we want to obtain as the MUCT of a causal diagram G. We will retain every edge of M and add selected bidirected edges. The proof has two tasks:

• showing that these additions give the desired territory; and

• showing that they preserve the corresponding MAG.

The second task requires considering all the added edges together.

All decisions about visibility refer to the original MAG. For a directed edge retained in $\mathcal { M } [ \mathbf { H } ]$ we use the causal model class represented by the full M and retain its latent variables, as in the definition of visibility for induced subgraphs. A visible edge in M remains visible when the observed vertices are restricted to H. An invisible directed edge remains invisible because Lemma 21 provides a corresponding DAG in which its endpoints share a latent parent. This parent and its two edges remain when observed vertices outside H are removed.

An observation about invisible directed edges. We first establish a fact that will be used when checking preservation of the MAG. Suppose that $A  B$ is invisible and that there is a bidirected path from C to A in M whose vertices are all parents of B. Then $C  B$ is also invisible.

To see this, suppose instead that $C  B$ were visible. By the graphical visibility criterion in Lemma 2, there would be a vertex V nonadjacent to B and a collider path from V into C whose non-endpoint vertices are parents of B. This includes the possibility of a single edge from V into C. Starting from V , follow this path up to its first intersection with the bidirected path, and then follow the latter path to A. The resulting path has no repeated vertices: the part before the first intersection is disjoint from the bidirected path.

If the intersection is not A, both incident edges have arrowheads there; if it is A, the resulting path ends with an arrowhead at A. Every non-endpoint vertex of the resulting path is a parent of B. Thus it is a collider path from V into A satisfying the visibility criterion for $A  B$ . This contradicts the invisibility of $A  B$ and proves the observation.

Constructing the causal diagram. Begin with $\mathcal { G } = \mathcal { M }$ , viewed as a causal diagram. For the current G, start with Y and repeatedly include directed descendants and vertices connected by bidirected paths in $\mathcal { G } [ \mathbf { H } ]$ , until no more vertices can be included. Call the resulting set T. Equivalently, T is the least subset of H containing Y that is closed under these two operations. As shown below, after intervention on X this is precisely the MUCT of the current causal diagram.

If $\mathbf { T } \neq \mathbf { R }$ , choose a vertex $B \in \mathbf { T }$ for which there is an invisible directed edge $C  B$ with $C \in \mathbf { H } \ \backslash$ T. For this one chosen B, add the bidirected edge $C  B$ for every such C at the same time, retaining the original directed edges $C  B$ and all other edges. Recompute T and repeat this step until $\mathbf { T } = \mathbf { R } $

The requirement to include every eligible parent of the chosen B is part of the construction. Each such parent becomes connected to a vertex already in T and therefore enters T in this step. Once a vertex has entered T, it remains there, because edges are only added. Consequently, a vertex C can occur as the parent in an addition of $C  B$ only once. In a later step, it cannot qualify as a parent outside the current territory.

For example, Fig. 16 illustrates this step with $\mathbf { X } = \{ A , B \}$ and $\mathbf { H } = \{ C , D , Y \}$ . Initially the current territory is $\mathbf { T } = \{ Y \}$ . The directed edges $C  Y$ and $D \to Y$ are invisible, thus choosing Y requires adding both $C  Y$ and $D  Y$ before recomputing the territory. It then becomes $\mathbf { T } = \{ C , D , Y \}$ , whose parents outside the set are $\{ A , B \}$ . All original directed edges are retained in the constructed causal diagram; only the intervention removes edges into $\{ A , B \}$ in the manipulated panels.

The construction reaches the required territory. We verify both that T never exceeds R and that another addition is possible whenever T is smaller than R. Initially $\mathbf { T } \subseteq \mathbf { R } .$ because R contains Y and is closed under directed descendants and original bidirected connections in $\mathcal { M } [ \mathbf { H } ]$ . Suppose this inclusion holds before a step choosing $B \in \mathbf { T }$ . For every added $C  B .$ the original directed edge $C  B$ is invisible, which gives:

$$
C \in { \mathsf { p c } } ( B ) _ { \mathcal { M } [ \mathbf { H } ] } \subseteq \mathbf { R } .\tag{211}
$$

Thus both endpoints of every added edge belong to R. The directed descendant relation is unchanged, and the original bidirected connections also remain within R whenever they start there. Recomputing T therefore preserves $\mathbf { T } \subseteq \mathbf { R }$

Now suppose that $\mathbf { T } \neq \mathbf { R }$ . By construction, T is already closed under directed descendants and original bidirected connections in $\mathcal { M } [ \mathbf { H } ]$ . It cannot also satisfy $\mathsf { p c } ( \mathbf { T } ) _ { \mathcal { M } [ \mathbf { H } ] } = \mathbf { T }$ . Otherwise it would be a UC-territory in $\mathcal { M } [ \mathbf { H } ]$ containing Y and properly contained in R, contrary to the minimality of R. Hence, by Def. 19, there is a collider path in $\mathcal { M } [ \mathbf { H } ]$ from a vertex of T to a vertex outside T, and none of its edges are visible. Traverse it from the vertex in T and consider the first edge crossing to the complement of T. This edge cannot be bidirected, because T is closed under original bidirected connections. It cannot be directed from T to its complement, because T is closed under directed descendants. It must therefore be an invisible directed edge from a vertex outside T into a vertex $B \in \mathbf { T }$ . This supplies the next choice required by the construction.

Every step adds at least one vertex to $\mathbf { T } ,$ and $Y \in \mathbf { T } \subseteq \mathbf { R } \subseteq \mathbf { H }$ throughout. The procedure therefore terminates with $\mathbf { T } = \mathbf { R } $ after at most $| \mathbf { H } | - 1$ steps. Since each vertex outside the initial territory can occur as C in an added edge $C  B$ with original directed edge $C  B$ at most once, at most $| \mathbf { H } | - 1$ bidirected edges are added.

The corresponding MAG is preserved. We next check that the final $\mathcal { G }$ still has corresponding MAG M. The following argument extends the single-edge argument in the proof of Lemma 21 to all edges added by the construction. All original adjacencies and ancestral relations among observed vertices are retained because no original edge is removed and no directed edge is added. It remains to show that the added bidirected edges do not create an adjacency between vertices that were nonadjacent in $\mathcal { M } .$ . Such a new adjacency would require an inducing path between those vertices in ${ \mathcal { G } } .$

For the sake of contradiction, suppose that $p$ is an inducing path between vertices $S , T$ nonadjacent in $\mathcal { M } .$ . For a causal diagram, such a path has only colliders as non-endpoint vertices, each of which is an ancestor of $S$ or $T$ . Among all these inducing paths, choose one with the fewest vertices; among paths with that number of vertices, choose one using the fewest added bidirected edges. Both parts of this choice will be used: a path with fewer vertices contradicts the first part, and a path with the same vertices but fewer added edges contradicts the second. The chosen path uses an added edge. Otherwise the same path, with every non-endpoint collider still an ancestor of $S$ or $T$ , would already be an inducing path in $\mathcal { M } .$ , contradicting maximality.

We first rule out a particular arrangement of two consecutive added edges on $p .$ Here consecutive means that no other added edge occurs between them; original edges may occur between them. Since all vertices between these two edges are non-endpoint colliders, the intervening original edges must be bidirected. The arrangement we rule out is:

$$
B  A ( = U _ { 0 } )  U _ { 1 }  \cdots  C ( = U _ { k } )  D , \qquad A  B { \mathrm { ~ a n d ~ } } C  D { \mathrm { ~ i n ~ } } { \mathcal { M } } ,\tag{212}
$$

where the first and last bidirected edges were added and the middle ones were already in $\mathcal { M }$ . Thus the two original directed edges point toward B and D, away from the intervening bidirected segment. The case $k = 0$ , with $A = C$ , is included. The vertices B and $D$ may be endpoints of $p ,$ , but they are distinct because p has no repeated vertices.

If $k = 0$ , the two added edges have the same original parent $A = C$ . They cannot occur because each parent participates in at most one addition, as shown above. Hence assume $k \geq 1$ . Suppose first that $A  B$ was added before $C  D$ . We show that every $U _ { i }$ is a parent of B in $\mathcal { M }$ . The first vertex $U _ { 0 } ( = A )$ is a parent by assumption. For the induction step, suppose $U _ { 0 } , \cdots , U _ { i - 1 }$ are parents of B. If $U _ { i }$ were nonadjacent to $B _ { : }$ , the original collider path $U _ { i }  \cdots  A _ { } $ , whose non-endpoint vertices are parents of $B ,$ would make $A  B$ visible. Therefore, $U _ { i }$ is adjacent to B.

The edge cannot be $B  U _ { i }$ : together with $U _ { i - 1 } \to B$ , it would make $U _ { i - 1 }$ an ancestor of $U _ { i }$ despite the original edge $U _ { i - 1 }  U _ { i }$ . The edge cannot be $U _ { i }  B$ either: replacing the segment from B through A to $U _ { i }$ by this original bidirected edge would shorten $p .$ All retained non-endpoint vertices would still be colliders and ancestors of S or $T _ { i }$ thus the shortened path would still be inducing. The only remaining possibility is $U _ { i } \to B$ , completing the induction.

In particular, $C$ and A are joined by an original bidirected path whose vertices are all parents of B. The observation at the beginning of the proof therefore implies that $C  B$ is invisible. Consider the step in which $A  B$ was added. If C was already in the territory at that time, it could not be eligible for a later addition of $C  D$ . If C was outside the territory, then $C \in \mathbf { H } \backslash \mathbf { T }$ and its invisible directed edge into the chosen B required $C  B$ to be added in that same step. This would place $C$ in the territory, again preventing the later addition $C  D$ . Both possibilities contradict the construction. If $C  D$ was added first, the same argument read from D towards B gives the symmetric contradiction. Finally, the two edges cannot have been added in the same step, because that step chooses a single vertex, whereas $B \neq D$ . Thus the displayed arrangement cannot occur on $p .$

Now list the added edges in their order from S to T, and consider the direction of each one’s original directed edge in M. The arrangement just excluded is exactly a left-pointing original directed edge followed by a right-pointing original directed edge at consecutive positions in this list. Consequently, if the first directed edge points to the left, the last directed edge must also point to the left; otherwise such a change would occur somewhere in the list. We can therefore read p from one of its endpoints so that the first added edge is traversed from A to B with $A  B$ in M. Indeed, use the endpoint S if its first directed edge points to the right, and otherwise read the path from T. Rename the chosen starting endpoint S and the other endpoint T. All edges preceding this first added $A  B$ are original edges.

If $A = S$ , replace $A  B$ on the path by the original directed edge $A  B$ . The mark at A is irrelevant because A is an endpoint, and the arrowhead at B is unchanged. Hence every non-endpoint vertex remains a collider and an ancestor of S or T. The result is an inducing path with the same number of vertices but one fewer added edge, contradicting the second part of our choice of $p .$ This replacement is used only when the original parent A is the path endpoint; it does not assume that every added edge touching an endpoint can be replaced in this way.

It remains to consider $A \ \ne \ S$ . Write the original path segment from $S$ to $A$ as $\langle S , O _ { 1 } , \cdot \cdot \cdot , O _ { k } ( = A ) \rangle$ . It is a collider path into A, and all edges between the $O _ { i }$ are original bidirected edges. We show backwards that every $O _ { i }$ is a parent of B. Starting with $O _ { k } = A$ suppose that $O _ { i + 1 } , \cdots , O _ { k }$ are parents of B. The invisibility of $A  B$ makes $O _ { i }$ adjacent to $B ,$ , since otherwise $O _ { i }  \cdots  A$ would satisfy the visibility criterion for $A  B$ . The edge $B \to O _ { i }$ would create an almost directed cycle consisting of $O _ { i + 1 }  B  O _ { i }$ and $O _ { i }  O _ { i + 1 }$ . The edge $O _ { i }  B$ would shorten $p ,$ and all remaining non-endpoint vertices would still be colliders and ancestors of S or T. Therefore $O _ { i } \to B$ . This proves the claim for all vertices $O _ { 1 } , \cdots , O _ { k }$

The same visibility criterion implies that S is adjacent to B. Otherwise, the path $\langle S , O _ { 1 } , \cdots , O _ { k } \rangle$ would establish visibility of $A  B$ . This adjacency cannot be $B  S$ . If the first original edge is $S  O _ { 1 }$ , then $S  O _ { 1 }  B  S$ would be a directed cycle. If it is $S  O _ { 1 }$ , then $O _ { 1 }  B  S$ together with that bidirected edge would form an almost directed cycle. Thus the edge between S and B has an arrowhead at B. If $B = T$ , this is already a contradiction, since S and T were assumed nonadjacent in M. Otherwise replace the path segment $\langle S , O _ { 1 } , \cdot \cdot \cdot , O _ { k } ( = A ) , B \rangle$ by this original edge from S into B. The arrowhead at B is preserved, and all other retained non-endpoint vertices remain colliders and ancestors of S or T. We have obtained a shorter inducing path, again a contradiction.

We conclude that no inducing path in $\mathcal { G }$ creates a new adjacency. Its corresponding MAG therefore has exactly the adjacencies of $\mathcal { M }$ . It also has exactly the same endpoint marks, because the ancestral relations used to determine those marks have not changed. Hence its corresponding MAG is $\mathcal { M }$

Identifying the resulting territory and border. Finally, we connect the set T used in the construction with the ordinary MUCT after intervention on X. The directed edges of $\mathcal { G }$ and M coincide, and a directed path from a vertex outside X to Y survives this intervention exactly when it avoids X. Thus:

$$
{ \tt A n } ( Y ) _ { \mathcal { G } _ { \overline { { \mathbf { X } } } } } \setminus \mathbf { X } = { \tt A n } ( Y ) _ { \mathcal { M } \backslash \mathbf { X } } = \mathbf { H } .\tag{213}
$$

Some intervened vertices may themselves remain ancestors of Y. However, they cannot enter the territory grown from Y since the intervention removes every edge into them, so neither a bidirected connection nor a directed path from the current territory can reach a vertex in X. The ordinary MUCT computation in the ancestral subgraph of $\mathcal { G } _ { \overline { { \mathbf { X } } } }$ therefore takes place entirely within H and uses exactly the two closure operations defining T. Since the construction ended with $\mathbf { T } = \mathbf { R }$ , this proves MUCT $( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \mathsf { M U C T } ( \mathcal { M } , Y , \mathbf { X } )$

Every vertex of R is outside X, so intervention on X removes none of the directed edges entering R. Together with the fact that no directed edge was changed during the construction, this shows that the parents of R outside R are the same in $\mathcal G _ { \overline { { \mathbf { X } } } }$ and $\mathcal { M } .$ Consequently, $1 \mathsf { B } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \mathsf { P a } ( \mathbf { R } ) _ { \mathcal { M } } \backslash \mathbf { R } = 1 \mathsf { B } ( \mathcal { M } , Y , \mathbf { X } )$ □

## A.2.3 Proof of Theorem 5

Proof of Thm. 5. Necessity. Fix X and write:

$$
\begin{array} { r l } & { \mathcal { H } = \mathcal { M } [ \mathrm { A n } ( Y ) _ { \mathcal { M } \backslash \mathbf { X } } ] , \qquad \mathbf { T } = \mathbb { M } \mathsf { U } \mathsf { C } \mathsf { T } ( \mathcal { M } , Y , \mathbf { X } ) , } \\ & { \mathbf { W } = \mathsf { P a } ( \mathbf { T } ) _ { \mathcal { M } } \setminus \mathbf { T } , \qquad \mathbf { W } ^ { \prime } = \mathbf { W } \setminus \mathbf { X } , \qquad \mathbf { X } ^ { \prime } = \mathbf { X } \setminus \mathbf { W } . } \end{array}
$$

Thus $\mathbf { W } = \mathsf { I B } ( \mathcal { M } , Y , \mathbf { X } )$ . The defining closures are $\mathsf { D e } ( \mathbf { T } ) _ { \mathcal { H } } = \mathbf { T }$ and $\mathsf { p c } ( \mathbf { T } ) _ { \mathcal { H } } = \mathbf { T } ;$ these are closures in H, not in the full MAG. Also, $\mathbf { T } \cap \mathbf { X } = \emptyset$

Our aim is to show that intervening on W can achieve at least the optimal reward attainable by intervening on X, in every conforming SCM. If $\mathbf { W } \neq \mathbf { X }$ , this will provide a competing DMIS that prevents X from being strictly optimal. The set $\mathbf { W } ^ { \prime }$ consists of the border variables that are missing from the current intervention, whereas $\mathbf { X } ^ { \prime }$ consists of intervention variables that are not on the border. The two separation claims below will allow us first to intervene on $\mathbf { W } ^ { \prime }$ and then to remove $\mathbf { X } ^ { \prime }$ from the intervention.

First, W is an MIS for M. For each $W \in \mathbf { W }$ , choose $W  T$ with $T \in \mathbf { T }$ . Since T is a vertex of $\mathcal { H } ,$ , it has a directed path to Y within H. Descendant closure keeps that path entirely in T, so the path from $W$ to Y avoids every other member of W. Now apply Thm. 4. Thus, W is also a DMIS by Prop. 6. This includes $\mathbf { W } = \varnothing$

For $\mathrm { M A G s }$ , lower manipulation at $\mathbf { W } ^ { \prime }$ deletes visible directed edges out of $\mathbf { W } ^ { \prime }$ and replaces invisible directed edges out of $\mathbf { W } ^ { \prime }$ by bidirected edges. Apply this operation before upper manipulation at $\mathbf { X } ,$ as required by the MAG manipulation convention (Zhang, 2008b).

Claim 1. $( Y \perp \perp \mathbf { W } ^ { \prime } \mid \mathbf { X } ) _ { \mathbf { \overline { { X } } \underline { { W } } } ^ { \prime } }$

Proof of Claim 1. Suppose that an m-connecting path p in $M _ { \overline { { \mathbf { X } } } \mathbf { W } ^ { \prime } }$ joins $W \in \mathbf { W } ^ { \prime }$ to $Y$ given X. All incoming edges to X have been removed, so a collider on p could neither belong to X nor be an ancestor of a vertex in X. It would therefore violate the condition for m-connection. Thus, p has no colliders, and every non-endpoint vertex lies outside X. Lower manipulation leaves only arrowheads at vertices in $\mathbf { W } ^ { \prime }$ . A non-endpoint vertex in $\mathbf { W } ^ { \prime }$ would therefore be a collider. Consequently, p has no non-endpoint vertex in W and its first edge has an arrowhead at W.

Every member of $\mathbf { W } ^ { \prime }$ is a vertex of H: it is outside X and has a child in T followed by a directed path to Y in H. Consider first a path whose first edge $W  V$ was created from an invisible $W  V$ . If $V \neq Y$ , the next edge must point away from $V ;$ otherwise, V would be a collider. Applying the same argument at each subsequent vertex shows that the remaining path is directed from $V$ to $Y$ . The case $V = Y$ is included. Starting at $Y \in \mathbf { T }$ and considering each preceding vertex on this directed path, any parent outside T would lie in W. Since there is no non-endpoint vertex of $p$ in W, V must belong to T.

The edge $W  V$ is invisible in H under the inherited interpretation of visibility. Indeed, Lemma 21 gives a DAG whose corresponding MAG is M and which contains $W \left. U \right. V$ for a latent vertex U. This path remains when only observed vertices outside H are removed. Hence $W \in { \mathsf { p c } } ( V ) _ { \mathcal { H } } \subseteq \mathbf { T }$ , contradicting $W \in \mathbf { W }$

Otherwise, all edges of $p$ are original MAG edges. Any added bidirected edge would have to meet a member of $\mathbf { W } ^ { \prime }$ , but no non-endpoint vertex of p belongs to that set. Since $p$ has no colliders and its first edge has an arrowhead at W, it has one of the following forms:

$$
W \left. \cdot \cdot \left. L \right. \cdot \cdot \cdot \right. Y ,\tag{214}
$$

$$
W  \cdot \cdot \cdot  L  R  \cdot \cdot \cdot  Y .\tag{215}
$$

To see this, start at $W$ and follow the path. The edges may initially point towards W. Once an edge points towards the next vertex, or a bidirected edge is encountered, every subsequent edge must point towards Y to avoid a collider. In either form, the directed path ending at Y may have length zero and consist only of Y . In the second form, the directed path from L to W may also have length zero, with $L = W$ . In the first form, however, $L \neq W$ , because the first edge has an arrowhead at W.

All vertices of $p$ are in the vertex set of H. The directed path ending at Y consists of ancestors of Y and avoids X. The directed path from L to W also avoids X, and ${ \cal W } \in { \bf V } ( \mathscr { H } )$ has a directed path to Y avoiding X. Its vertices are therefore also ancestors of Y in ${ \mathcal { M } } \backslash \mathbf { X }$

We now show that both forms imply $W \in \mathbf { T }$ . In the first form, start at $Y \in \mathbf { T }$ and consider each preceding vertex on the directed path from L to Y. As above, a parent outside T would belong to W, which is impossible for a non-endpoint vertex of $p .$ Thus, $L \in \mathbf { T }$ . In the second form, the same argument along the directed path from R to Y gives $R \in \mathbf { T }$ . The bidirected edge $L  R$ is invisible, so $L \in { \mathsf { p c } } ( R ) _ { \mathcal { H } } \subseteq \mathbf { T }$ . In both forms, W is a descendant of L along a directed path in H. Since $\mathsf { D e } ( \mathbf { T } ) _ { \mathcal { H } } = \mathbf { T }$ , we obtain $W \in \mathbf { T }$ , contradicting $W \in \mathbf { W }$ □

Claim 2. $( Y \perp \perp \mathbf { X } ^ { \prime } \mid \mathbf { W } ) _ { \mathcal { M } _ { \overline { { \mathbf { W } \cup \mathbf { X } ^ { \prime } } } } } .$

Proof of Claim 2. An m-connecting path from $X \in \mathbf { X } ^ { \prime }$ would start with an outgoing edge. No collider can be activated by conditioning on W, whose vertices have no incoming edges after manipulation. The path must therefore be directed towards $Y$ and avoid W. Since $Y \in \mathbf { T }$ and $X \not \in \mathbf { T }$ , this directed path has a first edge entering T from outside. The vertex just before that first member of T lies in $\mathsf { P a } ( \mathbf { T } ) _ { \mathcal { M } } \setminus \mathbf { T } = \mathbf { W }$ , a contradiction. □

For any assignment x in any conforming SCM, marginalization and Rules 2 and 3 for MAGs yield:

$$
\mu _ { \mathbf { x } } = \sum _ { \mathbf { w } ^ { \prime } } \mathbb { E } [ Y \mid d o ( \mathbf { x } ) , \mathbf { w } ^ { \prime } ] P ( \mathbf { w } ^ { \prime } \mid d o ( \mathbf { x } ) )
$$

$$
= \sum _ { \mathbf { w } ^ { \prime } } \mathbb { E } [ Y \mid d o ( \mathbf { x } , \mathbf { w } ^ { \prime } ) ] P ( \mathbf { w } ^ { \prime } \mid d o ( \mathbf { x } ) )\tag{216}
$$

$$
\begin{array} { r l } & { = \displaystyle \sum _ { \mathbf { w } ^ { \prime } } \mathbb { E } [ Y \mid d o ( \mathbf { x } \cap \mathbf { W } , \mathbf { w } ^ { \prime } ) ] P ( \mathbf { w } ^ { \prime } \mid d o ( \mathbf { x } ) ) } \\ & { \leq \mu _ { \mathbf { w } ^ { * } } . } \end{array}\tag{217}
$$

The first rule application exchanges observation of $\mathbf { W } ^ { \prime }$ for intervention, using Claim 1. The second removes the interventions on $\mathbf { X } ^ { \prime } ,$ using Claim 2. Both rule applications are within the same fixed SCM. What remains is an average of rewards under assignments to $\mathbf { W } = ( \mathbf { X } \cap \mathbf { W } ) \cup \mathbf { W } ^ { \prime }$ . Each term is at most the best reward attainable on W, and the nonnegative averaging weights sum to one. This gives the last inequality. For general domains, replace the sum by the corresponding integral. The bound holds for every x, so it also holds after maximizing over assignments to X. If $\mathbf { W } \neq \mathbf { X }$ , the DMIS W is therefore a distinct competitor whose optimal reward is never smaller. Thus X cannot be a POMIS, proving necessity.

Suficiency. Suppose $\mathbf { X } = \mathsf { I B } ( \mathcal { M } , Y , \mathbf { X } )$ . By Lemma 36, there is a causal diagram $\mathcal { G }$ with corresponding MAG M, exactly the same directed edges, and $\mathsf { I B } ( \mathcal { G } _ { \overline { { \mathbf { X } } } } , Y ) = \mathbf { X }$ . By Thm. $^ { 3 , }$ X is a POMIS, and hence an MIS, for G. Thus there is one SCM ${ S ^ { * } }$ conforming to $\mathcal { G }$ in which the optimal reward attainable on X is strictly greater than that attainable on every other MIS of $\mathcal { G }$

Because $\mathcal { G }$ and M have exactly the same directed edges, their proper directed paths agree for every intervention set. Hence Prop. 1 and Thm. 4 imply:

$$
\mathbb { M } _ { \mathcal { G } , Y } = \mathbb { M } _ { \mathcal { M } , Y } = \mathbb { D } _ { \mathcal { M } , Y } .\tag{218}
$$

The final equality follows from Props. 5 and 6. The same ${ S ^ { * } }$ conforms to M and supplies all strict inequalities required by Def. 18. Therefore, X is a POMIS for M. □

## A.2.4 Terminal PMG–MAG Correspondence

Let $\mathcal { P }$ be a PAG on V, and let $\mathbf { X } \subseteq \mathbf { V } \setminus \{ Y \}$ . The PMG Q is obtained from $\mathcal { P }$ by valid local transformations and the orientation rules in Thm. 7 applied to closure after each transformation. We first explain how to compare the territories of Q and its represented MAGs. We then show how to choose the remaining orientations so that one represented MAG has the required interventional border. Visibility in every induced graph retains the model class of its full PMG.

Visibility in induced graphs. Consider a directed edge $A  B$ in Q and an observed vertex set A containing both endpoints. Visibility relative to the inherited model class is the same in $\mathcal { Q }$ and $\mathcal { Q } [ \mathbf { A } ]$ . We verify both directions because the ancestor sets used later depend on the graph.

First suppose that $A  B$ is visible in the full PMG. If a represented DAG restricted to A and its latent vertices had an inducing path into A between A and $B ,$ the same path would be present in the full DAG. The directed paths that establish the ancestry of any colliders would also remain present in the full DAG. This contradicts visibility in $\mathcal { Q } ,$ so the edge is visible after restriction.

Conversely, suppose that $A  B$ is invisible in the full PMG. There is then a represented DAG with an inducing path between A and B that is into A. Its corresponding MAG has an invisible $A  B$ by Lemma 20. By Lemma 21, a DAG represented by this same MAG can be chosen to contain a latent common parent $A \left. U \right. B$ . The vertices $A , B , U$ and these two edges survive restriction to A and the latent vertices. They therefore establish invisibility after restriction as well. Non-directed edges are invisible by definition. For each fixed full graph and its represented model class, these observations allow us to use the same visibility status when an edge appears in diferent induced ancestor graphs.

Expressing the territory using individual edges. Let D be $\mathcal { Q }$ or a MAG in [Q], let $\mathbf { A } _ { \mathcal { D } } = \mathsf { P o s s A n } ( Y ) _ { \mathcal { D } \backslash \mathbf { X } }$ and let $\mathbf { T } _ { \mathcal { D } } = \mathsf { M U C T } ( \mathcal { D } , Y , \mathbf { X } )$ . Then $\mathbf { T } _ { \mathcal { D } }$ is the set reachable from $Y$ in $\mathcal { D } [ \mathbf { A } _ { \mathcal { D } } ]$ when invisible edges can be traversed in both directions and visible edges can be traversed from tail to arrowhead.

An invisible edge is itself a path satisfying the possible c-component $\mathrm { ( p c ) }$ condition: it has no internal vertices, so its internal-collider condition is vacuous. Consequently, if a set is closed under pc and contains one endpoint of an invisible edge, it must also contain the other endpoint.

Conversely, suppose that a set is closed under this operation on individual invisible edges. A pc path consists entirely of invisible edges. Starting from an endpoint in the set, we can follow its edges one at a time and conclude that every vertex of the path, including the other endpoint, is in the set. Thus closure under pc is equivalent to closure under traversing individual invisible edges in either direction. This argument does not require the pc relation itself to be transitive.

Possible descendant closure additionally permits movement along a possibly directed path. Each invisible edge on such a path is already covered by the preceding operation, pc. Any visible edge is directed, and a possibly directed path can use it only from tail to arrowhead. It follows that the set reached from Y by the movements in the statement is closed under both operations. Every set containing Y and closed under both operations must contain all vertices reached in this way. Hence the reached set is precisely the minimal such set, $\mathbf { T } _ { \mathcal { D } }$

Lemma 37. Let D be Q or a MAG in [Q], let $\mathbf { A } _ { \mathcal { D } } = \mathsf { P o s s A n } ( Y ) _ { \mathcal { D } \backslash \mathbf { X } }$ and let ${ \bf T } _ { \mathcal { D } } =$ MUCT(D, Y, X). If every $X \in \mathbf { X }$ has a proper directed path to Y in $\mathcal { D }$ , then $\begin{array} { r } { { | \mathsf { B } ( \mathcal { D } , Y , \mathbf { X } ) = \mathbf { X } } } \end{array}$ $i f$ and only i $\mathbf { \sigma } ^ { \mathrm { ~ r ~ } } _ { \mathcal { D } } = \mathbf { A } _ { \mathcal { D } }$

Proof of Lemma 37. An unchanged border requires the entire ancestor set. Suppose that $\mathsf { I B } ( \mathcal { D } , Y , \mathbf { X } ) = \mathbf { X }$ , and assume for a contradiction that some $W$ belongs to $\mathbf { A } _ { \mathcal { D } } \setminus \mathbf { T } _ { \mathcal { D } }$

By the definition of $\mathbf { A } _ { \mathcal { D } }$ , there is a possibly directed path from W to Y avoiding X. Every vertex on this path belongs to $\mathbf { A } _ { D } ,$ , since the remaining part of the path is also a possibly directed path to Y avoiding X.

The path starts outside $\mathbf { T } _ { \mathcal { D } }$ and ends at $Y \in \mathbf { T } _ { \mathcal { D } }$ . Let $U * - * V$ be the first edge with $U \notin \mathbf { T } _ { \mathcal { D } }$ and $V \in \mathbf { T } _ { \mathcal { D } }$ . This edge cannot be invisible. Otherwise, U belongs to $\mathbf { T } _ { \mathcal { D } }$ . It must therefore be visible. Because the path is possibly directed from U toward $V ,$ its orientation is $U  V$ Thus U is a parent outside the territory and belongs to $| \mathsf { B } ( \mathcal { D } , Y , \mathbf { X } )$ . But U lies on a path avoiding X, so $U \not \in \mathbf { X }$ . This contradicts the assumed border equality $| \mathsf { B } ( \mathcal { D } , Y , \mathbf { X } ) = \mathbf { X }$ Therefore, $\mathbf { T } _ { \mathcal { D } } = \mathbf { A } _ { \mathcal { D } }$

Recover the border from the entire ancestor set. Now suppose that $\mathbf { T } _ { \mathcal { D } } = \mathbf { A } _ { \mathcal { D } }$ . Let $W  V$ be any edge with $V \in \mathbf { A } _ { \mathcal { D } }$ and $W \notin \mathbf { A } _ { \mathcal { D } }$ . If $W \not \in \mathbf { X }$ , this edge can be followed by a possibly directed path from V to Y avoiding X. Such a path cannot already contain $W$ since its remaining part from W would make W a member of $\mathbf { A } _ { \mathcal { D } }$ . The concatenation is therefore a possibly directed path from W to Y avoiding X, again a contradiction. Hence every parent outside the territory belongs to X, giving I $\mathsf { B } ( \mathcal { D } , Y , \mathbf { X } ) \subseteq \mathbf { X }$

For the reverse inclusion, fix $X \in \mathbf { X }$ and take its proper directed path to Y with respect to X. The first child on this path is an ancestor of Y along a path avoiding X, and thus belongs to $\mathbf { A } _ { \mathcal { D } } = \mathbf { T } _ { \mathcal { D } }$ . Since X itself is excluded from $\mathbf { A } _ { \mathcal { D } }$ , it is a parent outside the territory. This proves $\mathbf { X } \subseteq \mathsf { I B } ( \mathcal { D } , Y , \mathbf { X } )$ and completes the equivalence. □

Choosing the remaining orientations. We will obtain a MAG from Q by replacing every ◦→ edge with a directed edge and orienting each circle component as a DAG without unshielded colliders. Lemma 34 applies to the simultaneous orientation of all these components and ensures that the resulting graph belongs to [Q].

Within a circle component, we can require any specified vertex R to have no incoming edges from that component. To see this, choose R first, then repeatedly choose a remaining vertex with the largest number of already chosen neighbors, breaking ties arbitrarily. Direct every edge from the earlier vertex to the later vertex in this order. For a chordal graph, each vertex’s earlier neighbors form a clique (Tarjan and Yannakakis, 1984). They therefore cannot form an unshielded collider at that vertex. The ordering also excludes a directed cycle.

Moreover, every vertex of the component is a descendant of R in this orientation. Indeed, while a connected component is only partly visited, some unvisited vertex has a visited neighbor. The stated choice rule therefore selects a vertex with at least one earlier neighbor. Its edge from that neighbor is directed into the newly visited vertex. Starting with R and repeating this argument gives a directed path from R to every vertex of the component. The choices of R made below can difer between components; the same cited lemma (Lemma 34) ensures that the resulting orientations together give one compatible MAG.

For example, consider the circle component in Fig. 37a, formed by the two triangles on $\{ R , A , B \}$ and $\{ B , C , D \}$ . Choose R first. Both A and B then have one already chosen neighbor, so we may choose A. Now B has two already chosen neighbors, R and A, whereas C and D have none; hence B is chosen next. Both C and D now have B as an already chosen neighbor. We may choose C and then D, giving the order $R \prec A \prec B \prec C \prec D$ . Directing each edge from the earlier vertex to the later vertex gives Fig. 37b. The blue edges provide the paths: $R \to A , R \to B , R \to B \to C$ , and $R \to B \to D$ . Thus even C and $D ,$ which are not adjacent to R, are descendants of R. Each newly chosen vertex receives an edge from an earlier vertex that is already reachable from R. This is the step repeated in the general argument above. The two incoming edges at B have adjacent parents $R , A ,$ and those at D have adjacent parents $B , C ;$ the resulting orientation therefore has no unshielded colliders.

![](images/63efa5491d78d424b508aae03e51d157390a885deb8d4138bbd5d6c70fe29b54.jpg)  
Figure 37: Orienting a circle component starting at R. In (b), each edge points from the earlier vertex to the later vertex in the order $R \prec A \prec B \prec C \prec D$ . The blue edges give directed paths from R to every other vertex.

For an edge $R \to B$ , a visibility witness is a path into R whose first vertex is not adjacent to B and whose internal vertices are colliders and parents of B. A path consisting of one edge is allowed. These are the conditions in Lemma 2, which we use below to establish visibility.

Lemma 38. Let M be obtained from Q by replacing every ◦→ edge with a directed edge and orienting each circle component as a DAG without unshielded colliders. Suppose that R has no incoming edges within its circle component in this orientation. If $R \to B$ is visible in M, then the edge between R and B is already oriented as $R \to B$ in $\mathcal { Q }$ and has a graphical visibility witness there. In particular, an invisible edge of Q that becomes $R \to B$ remains invisible in M.

Proof of Lemma 38. Step 1: Find a path that establishes visibility in the MAG. By Lemma 2, visibility of $R  B$ supplies a path $p = \langle C _ { 0 } , C _ { 1 } , \cdots , C _ { k } ( = R ) \rangle$ such that $C _ { 0 }$ is not adjacent to B, the path has an arrowhead at $R ,$ and all internal vertices are colliders on the path and parents of B. The endpoint R is also a parent of B by the edge under consideration. Thus $C _ { i } \to B$ in M for $1 \leq i \leq k$ . All these vertices are distinct from B. When $k \geq 2$ , every edge from $C _ { 1 }$ through R has arrowheads at both endpoints, so this part of the path is bidirected.

Step 2: The path’s arrowheads were already in the PMG. The procedure used to obtain M creates no bidirected edges. Consequently, the bidirected part from $C _ { 1 }$ through R was already present in Q. If the first edge is bidirected, it too was already present. If the first edge is directed into $C _ { 1 }$ , its arrowhead at $C _ { 1 }$ could be new only when $C _ { 0 } { \circ { \mathrm { - } } \circ } C _ { 1 }$ was oriented as $C _ { 0 }  C _ { 1 }$ . For $k = 1$ , such a new arrowhead would give R an incoming edge from another vertex in its circle component, contrary to the assumption on R. For $k \geq 2 .$ , suppose that this possibility does occur. We then have $C _ { 0 } { \circ { - } \circ } C _ { 1 }  C _ { 2 }$ in Q. The bidirected edge equivalence in Lemma 4 gives $C _ { 0 }  C _ { 2 }$ . We may therefore replace the beginning $\langle C _ { 0 } , C _ { 1 } , C _ { 2 } \rangle$ by $\langle C _ { 0 } , C _ { 2 } \rangle$ . This deletes one vertex and preserves the collider condition, the arrowhead at $R ,$ and all the remaining parent conditions. It also leaves the same endpoint $C _ { 0 }$ , which is still not adjacent to B. After this possible deletion, every arrowhead required by the resulting path is already present in Q.

Step 3: Every required parent mark was already fixed. Relabel the resulting path as $\langle C _ { 0 } , C _ { 1 } , \cdot \cdot \cdot , C _ { k } ( = \textit { R } ) \rangle$ . We prove in order, for $i = 1 , \cdots , k$ , that the edge is already oriented as $C _ { i } \to B$ in Q. Because the corresponding edge is directed from $C _ { i }$ to B in $\mathcal { M } .$ the possibilities in Q are $C _ { i }  B , C _ { i } \circ  B$ , or $C _ { i } \circ - \circ B$

For $i = 1$ , the circle edge possibility $C _ { 1 } \circ - \circ B$ is excluded by balance: together with the already present $C _ { 0 } * \mathrm {  } C _ { 1 }$ , it would make $C _ { 0 }$ adjacent to B. If the remaining possibility is $C _ { 1 } \circ  B$ , the triple $C _ { 0 } * \to C _ { 1 } \circ \to B$ is unshielded, and $\mathcal { R } _ { 1 }$ orients $C _ { 1 } \to B$ . Since $\mathcal { Q }$ is closed under this rule, the tail at $C _ { 1 }$ must already be displayed in $\mathcal { Q } .$

Now take $i > 1$ and assume that $C _ { 1 } , \cdots , C _ { i - 1 }$ have already been shown to be parents of B in Q. If $C _ { i } \circ - \circ B$ , the edge $C _ { i - 1 }  C _ { i }$ and circle balance would imply $C _ { i - 1 }  B$ . This contradicts the orientation $C _ { i - 1 } \to B$ in Q just established. Hence only $C _ { i } \to B$ or $C _ { i } \circ  B$ remains.

To exclude the latter possibility, consider $\langle C _ { 0 } , C _ { 1 } , \cdots , C _ { i } , B \rangle$ . Its first endpoint $C _ { 0 }$ is not adjacent to B, while $C _ { i }$ is adjacent to B. Every intermediate vertex $C _ { 1 } , \cdots , C _ { i - 1 }$ is a collider on this path, by Step 2, and a parent of B in Q, by the induction hypothesis. The vertices are distinct. Thus this is a discriminating path for $C _ { i } ,$ and $\mathcal { R } _ { 4 } ^ { \prime }$ would orient $C _ { i } \circ  B$ as $C _ { i } \to B$ Closure again implies that this tail is already fixed in Q. This completes the induction.

Step 4: Apply the visibility criterion in the PMG. The path now has all its required arrowheads in Q, and all its internal vertices are parents of B in Q. Its endpoint $C _ { 0 }$ remains nonadjacent to B. It therefore establishes visibility of the edge $R  B$ , already oriented in $\mathcal { Q } ,$ in every represented MAG and in the inherited causal model class of $\mathcal { Q } .$ The last assertion of the lemma is the contrapositive of this conclusion. □

Corollary 11. A directed edge $R  B$ is visible in $\textit { \textbf { Q } } i f$ and only if it has a graphical visibility witness in the full graph Q.

Proof of Cor. 11. Suppose first that $R \to B$ is visible in Q. Choose R first when orienting its circle component as described above, and orient all other circle components as DAGs without unshielded colliders. Replace every ◦→ edge by a directed edge. By Lemma 34, these simultaneous choices give a MAG $\mathcal { M } \in [ \mathcal { Q } ]$ in which R has no incoming edges from its circle component. Every DAG represented by M belongs to the model class specified by Q, so $R \to B$ is visible in M. By Lemma 38, it has a graphical visibility witness in Q.

Conversely, suppose that such a witness is present in Q. Its nonadjacency, required arrowheads, and directed parent edges are preserved in every MAG in [Q]. The criterion in Lemma 2 therefore makes $R \to B$ visible in every such MAG, and hence in the full causal model class specified by Q. □

Lemma 39. Let B, C lie in the same circle component of Q and let R lie outside that component. If $R \to B$ has a graphical visibility witness in Q, then the edge between R and C is already oriented as $R \to C$ in Q and has a graphical visibility witness there.

Proof of Lemma 39. If $B = C$ , the conclusion is immediate. Otherwise, it sufices to prove the result for adjacent vertices $B \circ { \ - - } \circ C$ and repeat it along a circle path from B to C.

Step 1: Obtain the possible parent edges into C. Choose a visibility witness $\langle C _ { 0 } , C _ { 1 } , \cdot \cdot \cdot , C _ { k } ( = R ) \rangle$ for $R  B$ . For every $1 \leq i \leq k$ , Q already contains the directed edge $C _ { i } \to B$ . The stronger balanced property applied to $C _ { i } \to B \circ - \circ C$ therefore gives $C _ { i } \to C { \mathrm { ~ o r ~ } } C _ { i } \circ \to C$ . In particular, each $C _ { i }$ is adjacent to $C ,$ with an arrowhead at $C .$ Each $C _ { i }$ lies outside the circle component of $B \colon$ if it lay in that component, its adjacency to B would have to be a circle edge by Lemma 4(1), contradicting $C _ { i } \to B$ . Thus none of these vertices equals $C .$

Step 2: The first endpoint stays nonadjacent to $C .$ For the sake of contradiction, suppose that $C _ { 0 }$ is adjacent to $C .$ We consider all possible forms of this edge.

If $C _ { 0 } \circ { \mathrm { - } } \circ C ,$ , then $C _ { 0 }$ lies in the same circle component as $B .$ Since $C _ { 1 }$ is outside that component, the arrowhead equivalence in Lemma $4 ( 2 )$ , applied to $C _ { 1 }  B ,$ gives an arrowhead at $C _ { 0 }$ on the $C _ { 1 } * { \mathrm { - } } * C _ { 0 }$ edge. Its bidirected edge equivalence also excludes $C _ { 1 }  C _ { 0 }$ , because $C _ { 1 }  B$ is not bidirected. The edge must therefore be $C _ { 1 }  C _ { 0 }$ or $C _ { 1 } \circ  C _ { 0 }$ . Both possibilities contradict the witness arrowhead at $C _ { 1 }$

Next suppose that the edge has no arrowhead at C and is not a circle edge. In any MAG obtained by replacing all $\hookrightarrow \to$ edges with directed edges and orienting the circle components as DAGs without unshielded colliders, this edge is $C  C _ { 0 }$ . Step 1 also gives $C _ { 1 } \to C$ in that MAG. The first edge of the witness still points into $C _ { 1 }$ . If it is $C _ { 0 }  C _ { 1 }$ , these three edges form $C  C _ { 0 }  C _ { 1 }  C$ . If it is $C _ { 0 }  C _ { 1 }$ , then $C _ { 1 } \to C \to C _ { 0 }$ together with $C _ { 0 }  C _ { 1 }$ is an almost directed cycle. Both conclusions contradict the ancestral property of the MAG, whose existence follows from Lemma 34.

The remaining possibility is an arrowhead at $C$ on the $C _ { 0 } * \mathrm { - } * C$ edge. The balanced property applied to $C _ { 0 } * \to C \circ - \circ B$ then gives an adjacency between $C _ { 0 }$ and $B .$ . This contradicts the defining nonadjacency of the visibility witness. All possibilities lead to contradictions; therefore, $C _ { 0 }$ is not adjacent to $C .$

Step 3: Fix the parent marks into $C$ in order. For $C _ { 1 }$ , Step 1 leaves $C _ { 1 }  C$ or $C _ { 1 } \circ  C$ . Since $C _ { 0 }$ is not adjacent to $C ,$ the triple $C _ { 0 } * { \longrightarrow } C _ { 1 } \circ { \longrightarrow } C$ would trigger $\mathcal { R } _ { 1 }$ . Since $\mathcal { Q }$ is closed under the rule, $C _ { 1 } \to C$ is already oriented in $\mathcal { Q } .$

For $i > 1$ , assume that the vertices $C _ { 1 } , \cdots , C _ { i - 1 }$ have already been shown to be parents of C in $\mathcal { Q } .$ . The path $\langle C _ { 0 } , \cdots , C _ { i } , C \rangle$ has distinct vertices, its first endpoint is nonadjacent to $C ,$ and its intermediate vertices are colliders and parents of $C$ in $\mathcal { Q }$ . Thus it is a discriminating path for $C _ { i }$ . If $C _ { i } \circ  C$ remained, $\mathcal { R } _ { 4 } ^ { \prime }$ would orient $C _ { i } \to C$ . Closure shows that this tail is already fixed. The induction reaches $C _ { k } = R$

The edge $R \to C$ is therefore already oriented in $\mathcal { Q } .$ The same collider path into $R$ now has all its internal vertices as parents of $C$ in $\mathcal { Q } .$ Step 2 supplies the required nonadjacency of its first endpoint. This is the desired visibility witness for $R \to C$ □

Lemma 40. Suppose that every circle mark at a vertex in $\mathbf { X } \cup \{ Y \}$ has been resolved in Q and that every $X \in \mathbf { X }$ has a proper directed path to $Y$ with respect to X in $\mathcal { Q }$ . Then ${ \sf I B } ( \mathcal { Q } , Y , { \bf X } ) = { \bf X }$ if and only if there exists a MAG $\mathcal { M } \in [ \mathcal { Q } ]$ satisfying $| \mathsf { B } ( \mathcal { M } , Y , \mathbf { X } ) = \mathbf { X }$

Proof of Lemma $ \mathit { 4 0 } .$ . Let $\mathbf { A } _ { \mathcal { M } } = \mathsf { A n } ( Y ) _ { \mathcal { M } \backslash \mathbf { X } }$ and let $\mathbf { T } _ { \mathcal { M } } = \mathsf { M U C T } ( \mathcal { M } , Y , \mathbf { X } )$ . Similarly, let $\mathbf { A } _ { \mathcal { Q } } = \mathsf { P o s s A n } ( Y ) _ { \mathcal { Q } \backslash \mathbf { X } }$ and let $\mathbf { T } _ { \mathcal { Q } } = \mathsf { M U C T } ( \mathcal { Q } , Y , \mathbf { X } )$ . For a represented MAG, $\mathbf { A } _ { \mathcal { M } }$ consists of the actual ancestors of $Y$ after X is deleted. For the PMG, $\mathbf { A } _ { \mathcal { Q } }$ consists of possible ancestors under the same deletion. We always have $\mathbf { A } _ { \mathcal { M } } \subseteq \mathbf { A } _ { \mathcal { Q } }$ , since a directed path in $\mathcal { M }$ is possibly directed in $\mathcal { Q } .$ . This inclusion can be strict. The argument below keeps the two sets separate.

## From a Represented MAG to the PMG

Step 1: Transfer membership in the MAG territory to the PMG. Suppose that $\mathcal { M } \in [ \mathcal { Q } ]$ satisfies ${ \sf I B } ( \mathcal { M } , Y , { \bf X } ) = { \bf X }$ . The proper directed paths from X in $\mathcal { Q }$ remain directed in M. Thus Lemma 37 gives $\mathbf { T } _ { \mathcal { M } } = \mathbf { A } _ { \mathcal { M } }$ . We next show that $\mathbf { T } _ { \mathcal { M } } \subseteq \mathbf { T } _ { \mathcal { Q } }$

Every vertex used to reach $\mathbf { T } _ { \mathcal { M } }$ lies in $\mathbf { A } _ { \mathcal { M } }$ and hence in $\mathbf { A } _ { \mathcal { Q } }$ . If an edge traversed in M is invisible, its counterpart in Q is invisible as well. If its counterpart is not directed, it is invisible by definition. If its counterpart is directed, the same edge is directed in $\mathcal { M } .$ , and a DAG represented by M that establishes its invisibility also belongs to the causal model class specified by Q; the visibility-under-restriction observation at the beginning of this subsection applies to both induced graphs. Such an edge can therefore also be traversed in Q. If a traversed edge in M is visible, it is traversed in its directed direction. Its counterpart in Q is either invisible, allowing either direction, or is a visible directed edge with the same orientation, allowing that forward direction. Thus every step used to reach a vertex of $\mathbf { T } _ { \mathcal { M } }$ can also be made in $\mathcal { Q } [ \mathbf { A } _ { \mathcal { Q } } ]$ . This proves the claimed inclusion $\mathbf { T } _ { \mathcal { M } } \subseteq \mathbf { T } _ { \mathcal { Q } }$

Step 2: Include the remaining possible ancestors in the PMG territory. Fix $W \in \mathbf { A } _ { \mathcal { Q } }$ . The case $W = Y$ is immediate. Otherwise, Lemma 33 supplies an uncovered possibly directed path from W to Y avoiding X. Before its first arrowhead, all edges are circle edges. Once an arrowhead appears, each following uncovered triple triggers $\mathcal { R } _ { 1 }$ if the next edge still has a circle at its starting vertex. Closure therefore makes the rest of the path directed. The path has the form $W \circ – \circ \cdot \cdot \cdot \circ - \circ V \ ^ { \circ } \to U \to \cdot \cdot \cdot \to Y$ , where $V { \mathrm { ~ } } ? \to U$ denotes either $V  U$ or $V \circ \to U$ , and the initial circle segment or the final directed segment may have length zero. A nontrivial path consisting entirely of circle edges is excluded because the marks at $Y$ are resolved.

The vertex U has a directed path to Y avoiding X in every represented MAG. In particular:

$$
U \in { \bf A } _ { \mathcal { M } } = \mathbf { T } _ { \mathcal { M } } \subseteq \mathbf { T } _ { \mathcal { Q } } .\tag{219}
$$

If $V  U$ , then V also has a directed path to Y avoiding X, so the same reasoning puts V in $\mathbf { T } _ { \mathcal { Q } }$ . If $V \circ \to U$ , this edge is invisible in Q. Since $U \in \mathbf { T } _ { \mathcal { Q } }$ and both endpoints lie in $\mathbf { A } _ { \mathcal { Q } }$ closure along invisible edges also puts V in $\mathbf { T } _ { \mathcal { Q } }$ . Each preceding circle edge is also invisible. Following them back toward W gives $W \in \mathbf { T } _ { \mathcal { Q } }$

Since W was arbitrary, $\mathbf { A } _ { \mathcal { Q } } \subseteq \mathbf { T } _ { \mathcal { Q } }$ . The reverse inclusion is part of the definition of the territory, so these sets are equal. Lemma 37 gives ${ \sf I B } ( \mathcal { Q } , Y , { \bf X } ) = { \bf X }$

## From the PMG to One Represented MAG

Step 1: Choose one vertex in each relevant circle component. Suppose that ${ \sf I B } ( \mathcal { Q } , Y , { \bf X } ) = { \bf X }$ . By Lemma 37, $\mathbf { T } _ { \mathcal { Q } } = \mathbf { A } _ { \mathcal { Q } }$

The set $\mathbf { A } _ { \mathcal { Q } }$ is a union of entire circle components. To see this, suppose a component contains a possible ancestor of Y in ${ \mathcal { Q } } \setminus \mathbf { X }$ . Any other vertex in the component can follow a circle path to that vertex and then its possibly directed path to Y . Removing repeated vertices, if necessary, leaves a possibly directed path. This path avoids X: every vertex in X has all its marks resolved and therefore lies in a singleton circle component.

![](images/2692b31922f7be242fb9f2262a3bda7d1fb708a9b81e1733f1fb0e5e91cb3545.jpg)  
Figure 38: Recording edges and completing the PMG in (a), with X = {X}. Gray backgrounds group the two nontrivial circle components of Q; X lies outside the ancestor set after deleting X. Panels (b)–(c) record existing edges in blue, numbered in visit order, and outline the chosen vertices A and C. These panels retain the PMG’s endpoint marks. A recorded edge only specifies how a new component was reached. Panel (d) orients each component from its chosen vertex, giving one compatible MAG.

Consider the components contained in A<sub>Q</sub>. By the closure characterization, every vertex in them can be reached from Y using invisible edges in either direction and visible edges in their directed direction. Start with the singleton component {Y }. If some components remain unvisited, choose an allowed path from Y to one of them and take its first edge leaving the already visited components. Let T be its endpoint in a visited component and R its endpoint in the new component. Record this edge and choose R as the vertex from which to orient the new component. Mark the new component as visited and repeat. Every repetition adds one component, so eventually every component contained in $\mathbf { A } _ { \mathcal { Q } }$ has been visited. Each receives exactly one such choice.

Step 2: Make all the choices into one MAG. Within each component, orient its circle edges as a DAG without unshielded colliders, with no incoming component edges at its chosen vertex R. The preceding discussion of chordal components shows that this can be done and that R then has a directed path to every vertex of its component. The component of Y is a singleton, so there is no internal choice there. Orient any components outside A<sub>Q</sub> as DAGs without unshielded colliders as well, and replace every ◦→ edge by a directed edge.

These choices specify the orientation of every remaining circle mark. Lemma 34 applies to the simultaneous choices and gives a single MAG $\mathcal { M } \in [ \mathcal { Q } ]$ . We use this same MAG throughout the rest of the proof.

For concreteness, consider the PMG in Fig. 38a, with $\mathbf { X } = \{ X \}$ . Its relevant ancestor set and territory are both $\mathbf { A } _ { \mathcal { Q } } = \mathbf { T } _ { \mathcal { Q } } = \{ A , B , C , D , Y \}$ , and its relevant circle components are $\{ Y \} , \{ A , B \}$ , and $\{ C , D \}$ . The singleton {X} is outside this set because X has been deleted when computing ancestors.

Start with $\{ Y \}$ visited. To visit $\{ A , B \}$ , use the invisible edge $A \circ \to Y$ in the direction from Y to A. Record this edge with $T = Y$ and $R = A$ , and choose A as the vertex from which to orient the new component (Fig. 38b). The visited vertices are now $\{ Y , A , B \}$ . Next use $C \circ  Y$ in the direction from Y to C, record it with $T = Y$ and $R = C$ , and choose $C$ for the component $\{ C , D \}$ (Fig. 38c). All relevant components have now been visited. Recording these two edges does not change their endpoint marks; it specifies the connection and the chosen vertex for each new component.

Make both choices in the same graph: orient $A  B$ and $C  D$ , and replace $V \circ \to Y$ by $V  Y$ for every $V \in \{ A , B , C , D \}$ , while keeping $X  Y$ . These choices give the single MAG M in Fig. 38d. The selected vertex A reaches both vertices of its component, and C does the same for its component. Every vertex other than X has a directed path to $Y$ avoiding X, so in this example $\mathbf { A } _ { \mathcal { M } } = \mathbf { A } _ { \mathcal { Q } }$

The recorded edges become $A \  \ Y$ and $C  Y$ and remain invisible in M. Indeed, Y is adjacent to every other vertex, so no visibility witness can have the required first vertex nonadjacent to $Y .$ . Starting from Y , invisible edge closure therefore includes A and $C ;$ descendant closure along $A \  \ B$ and $C  D$ then includes B and D. Thus $\mathbf { T } _ { \mathcal { M } } = \left\{ A , B , C , D , Y \right\}$ and its only outside parent is X, giving $\mathsf { I B } ( \mathcal { M } , Y , \{ X \} ) = \{ X \}$ . This illustrates the reverse traversal of invisible edges in Step 4 below. The remaining argument also covers completions in which $\mathbf { A } _ { \mathcal { M } }$ is strictly smaller than $\mathbf { A } _ { \mathcal { Q } }$

Step 3: Verify ancestry for the chosen vertices. For every component contained in $\mathbf { A } _ { \mathcal { Q } } .$ , its chosen vertex R belongs to $\mathbf { A } _ { \mathcal { M } }$ . This is immediate for $R = Y$ . For any other $R ,$ take an uncovered possibly directed path from R to Y avoiding X. As in the first direction of this proof, it consists of circle edges, then one directed or partially directed edge, followed by directed edges to $Y$ . The initial circle segment stays in the component of R. In ${ \mathcal { M } } ,$ , there is a directed path from R to its last vertex because of the orientation chosen in Step 2. The next edge is directed in ${ \mathcal { M } } ,$ , and the remaining part of the path is already directed in $\mathcal { Q } .$ Since M has no directed cycle, concatenating these edges gives a directed path from R to $Y$

It avoids X because the circle component is disjoint from X and the remaining path was chosen to avoid X. Hence $R \in \mathbf { A } _ { \mathcal { M } }$

A component may also contain vertices that are in $\mathbf { A } _ { \mathcal { Q } }$ but not in $\mathbf { A } _ { \mathcal { M } }$ . The following induction will require territory membership only for its vertices in $\mathbf { A } _ { \mathcal { M } }$

Step 4: Include the actual ancestors component by component. Consider the components in the order recorded in Step 1. We prove that every vertex in an already considered component that belongs to $\mathbf { A } _ { \mathcal { M } }$ also belongs to $\mathbf { T } _ { \mathcal { M } }$ . The first component is $\{ Y \}$ and $Y \in \mathbf { T } _ { \mathcal { M } }$ by definition.

For the next component, use its recorded edge $T * - * R$ , where T is in an earlier component and R is the chosen vertex of the new component. Let S be the chosen vertex of the component containing $T .$ . Step 3 gives $R , S \in \mathbf { A } _ { \mathcal { M } }$ . The induction hypothesis in particular gives $S \in \mathbf { T } _ { \mathcal { M } }$ . We show that R also belongs to $\mathbf { T } \boldsymbol { \mathcal { M } }$ by considering the recorded edge.

• The edge points from the earlier component to the new one. Suppose that the edge in Q is $T  R$ or $T \circ \to R$ . It is $T  R$ in M. Because R has a directed path to $Y$ avoiding X, so does T. Thus $T \in \mathbf { A } _ { \mathcal { M } }$ , and the induction hypothesis puts $T$ in $\mathbf { T } _ { \mathcal { M } }$ . The edge $T  R$ lies in the induced graph on $\mathbf { A } _ { \mathcal { M } } ,$ so descendant closure puts R in $\mathbf { T } _ { \mathcal { M } }$

• The edge is bidirected. Suppose that the recorded edge is $R  T$ . If $T = S$ , it directly connects R to the known territory vertex S. Otherwise, T and S are in the same circle component, and Lemma 4 gives $R  S$ . This edge is unchanged in M. Both endpoints belong to $\mathbf { A } _ { \mathcal { M } }$ , and S already belongs to $\mathbf { T } _ { \mathcal { M } }$ . Closure along this invisible edge therefore includes R.

• The edge points toward the earlier component and is invisible. The remaining possibility is an invisible $R \to T$ or $R \circ  T$ , used in the reverse direction when visiting the new component. If $T = S$ , retain this edge. Otherwise, the arrowhead equivalence in Lemma 4(2) gives an arrowhead at S on the $R { * } { - } { * } S$ edge. The bidirected edge equivalence excludes $R  S ,$ , since it would imply $R  T$ . Thus the edge to S is $R \to S$ or $R \circ  S$ and in either case it becomes $R \to S$ in M.

We must also show that $R  S$ is invisible in M. Suppose instead that it is visible. The choice of R in Step 2 satisfies Lemma 38, which then shows that $R \to S$ is already oriented in Q and has a graphical visibility witness there. If $T \neq S$ , Lemma 39 shows that $R \to T$ is also already oriented in Q and has such a witness. If $T = S$ , the same conclusion already holds.

In both cases the edge originally used to visit the component would be visible in $\mathcal { Q } ,$ contradicting its selection as an invisible edge. Therefore $R \to S$ is invisible in $\mathcal { M }$ . Since both endpoints are in $\mathbf { A } _ { \mathcal { M } }$ and $S \in \mathbf { T } _ { \mathcal { M } }$ , invisible edge closure includes R.

These cases exhaust the possibilities. Circle edges cannot join diferent circle components, and a visible edge was allowed only in its directed direction. We have shown that the chosen vertex R of the new component belongs to $\mathbf { T } \boldsymbol { \mathcal { M } }$

Now let V be any other vertex of this component with $V \in \mathbf { A } _ { \mathcal { M } }$ . Step 2 supplies a directed path from R to V inside the component. Every vertex on that path is also in $\mathbf { A } _ { \mathcal { M } } \colon$ it reaches V , and V reaches Y along a directed path avoiding X. The component itself contains no vertex of X. Thus the path from R to V lies entirely in the induced ancestor graph. Descendant closure puts V in $\mathbf { T } _ { \mathcal { M } }$ , completing the induction step.

All vertices of $\mathbf { A } _ { \mathcal { M } }$ lie in $\mathbf { A } _ { \mathcal { Q } }$ , whose components have all been considered. The induction therefore proves $\mathbf { A } _ { \mathcal { M } } \subseteq \mathbf { T } _ { \mathcal { M } }$ . The reverse inclusion is definitional, giving $\mathbf { T } _ { \mathcal { M } } = \mathbf { A } _ { \mathcal { M } }$ . Finally, the proper directed paths from X present in Q are preserved in this same M. Lemma 37 now gives ${ \sf I B } ( \mathcal { M } , Y , { \bf X } ) = { \bf X }$ , which proves the required existence statement. □

## A.2.5 Proof of Theorem 8

Proof of Thm. 8. Termination. A recursive call considers finitely many local transformations at its chosen vertex. Each recursive descent removes that vertex from the set still to be processed, so the depth is at most $| \mathbf { X } \cup \{ Y \} |$ . The orientation closure within each call also terminates: each efective rule application resolves at least one circle mark, and there are only finitely many marks. Thus every branch and the entire search terminate.

The terminal correspondence applies. By the time a branch reaches its terminal PMG, it has processed every vertex in X ∪ {Y }. The local transformations have therefore resolved all circle marks at these vertices.

Furthermore, at each processed $X \in \mathbf { X }$ the branch passed the proper path check. By Prop. 10, the resulting PMG contains a proper directed path from X to Y with respect to X. Later valid transformations preserve the tails and arrowheads already displayed along this path, so it remains directed and proper in the terminal PMG. Consequently, each terminal PMG satisfies both conditions of Lemma 40.

Soundness. Suppose that IsPOMIS returns True. Then there is an accepting terminal PMG Q for which the tested equality is ${ \sf I B } ( \mathcal { Q } , Y , \mathbf { X } ) = \mathbf { X }$ . Every local transformation on this branch was valid. It therefore retains at least one compatible MAG, and sound orientation closure preserves the MAGs consistent with the chosen transformations. Since the branch started from P, we have $\emptyset \neq [ { \mathcal { Q } } ] \subseteq [ { \mathcal { P } } ]$ . The terminal correspondence supplies a single $\mathcal { M } \in [ \mathcal { Q } ]$ with IB $( { \mathcal { M } } , Y , \mathbf { X } ) = \mathbf { X }$ . By Thm. 5, X is a POMIS for this MAG. Because $\mathcal { M } \in [ \mathcal { P } ]$ the set X belongs to $\widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ . Thus every True output has the claimed meaning.

Completeness. Suppose that $\mathbf { X } \in \widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ . Fix one MAG $\mathcal { M } \in [ \mathcal { P } ]$ for which X is a POMIS. In particular, X is an MIS for M, so each of its vertices has a proper directed path to Y in M. The corresponding paths in P are possibly directed and remain proper. Hence X passes the algorithm’s initial MIS screen.

Now follow choices that agree with this fixed M at every processed vertex. We verify that these choices define an admissible branch. Initially $\mathcal { M } \in [ \mathcal { P } ]$ . Whenever the current PMG represents M, the local choice agreeing with M is valid, since this same MAG realizes all the chosen marks. Sound orientation closure preserves it; thus, the resulting PMG again represents M. This argument applies successively throughout the branch, as in Prop. 9.

It remains to check that the possible ancestor test never rejects this branch. At a processed $X \in \mathbf { X }$ , the proper directed path already present in M avoids $\mathbf { X } \setminus \{ X \}$ . Since M also agrees with the local choice, the reverse implication in Prop. 10 shows that the required possible ancestor test holds in the current PMG. At the processed vertex Y, the test holds by reflexivity: neither X nor the local transformation set contains Y.

Thus the search either returns True on an earlier branch or reaches a terminal PMG Q representing this same M. The necessary direction of Thm. 5 gives I $\mathsf { B } ( \mathcal { M } , Y , \mathbf { X } ) = \mathbf { X }$ Applying Lemma 40 to the terminal PMG then gives IB $( { \mathcal { Q } } , Y , \mathbf { X } ) = \mathbf { X }$

This branch therefore returns True. Every member of $\widehat { \mathbb { P } } _ { \mathcal { P } , Y }$ is accepted, which completes the proof. □

## A.3 Experimental Structural Causal Models

This section specifies the six fixed SCMs used in Sec. 4.3. All observed and exogenous variables are binary, and the reward is Y . Within each task, all distinct exogenous variables are mutually independent, with the Bernoulli distributions specified in each model. A variable $U _ { V }$ appears only in the equation for V. Each variable $U _ { V W }$ appears in two structural functions. Sharing an exogenous variable in this way allows dependence between observed variables, even though distinct exogenous variables are independent.

Task 1. Let $\mathbf { V } = \{ A , B , C , D , Y \}$ and $\mathbf { U } = \{ U _ { V } : V \in \mathbf { V } \} \cup \{ U _ { B Y } , U _ { A C } \}$ . The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A C } \wedge u _ { A } } \\ & { f _ { C } = \left( u _ { C } \vee \neg a \right) \vee u _ { A C } } \\ & { f _ { B } = \neg u _ { B Y } \oplus \left( c \vee u _ { B } \right) } \\ & { f _ { Y } = u _ { Y } \vee \left( \neg u _ { B Y } \wedge \left( \neg b \vee c \right) \right) } \\ & { f _ { D } = \neg u _ { D } \wedge y . } \end{array}\tag{220}
$$

The exogenous distributions are:

$$
\begin{array} { r l r } { U _ { A } \sim \mathtt { B e r n } ( 0 . 5 7 ) } & { { } U _ { B } \sim \mathtt { B e r n } ( 0 . 5 7 ) } & { U _ { C } \sim \mathtt { B e r n } ( 0 . 7 7 ) } \\ { U _ { D } \sim \mathtt { B e r n } ( 0 . 6 1 ) } & { { } U _ { Y } \sim \mathtt { B e r n } ( 0 . 4 2 ) } & { U _ { B Y } \sim \mathtt { B e r n } ( 0 . 4 6 ) } \\ { U _ { A C } \sim \mathtt { B e r n } ( 0 . 6 2 ) . } \end{array}\tag{221}
$$

Task 2. Let $\mathbf { V } = \{ A , B , C , D , E , Y \}$ and $\mathbf { U } = \{ U _ { V } : V \in \mathbf { V } \} \cup \{ U _ { B Y } , U _ { A Y } \}$ . The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } \oplus u _ { A Y } } \\ & { f _ { B } = u _ { B Y } \vee u _ { B } } \\ & { f _ { C } = \lnot u _ { C } \vee b } \\ & { f _ { D } = c \wedge u _ { D } } \\ & { f _ { E } = \left( \lnot a \oplus u _ { E } \right) \wedge c } \\ & { f _ { Y } = u _ { A Y } \wedge \left( \left( \lnot u _ { B Y } \vee \lnot d \right) \wedge e \right) \oplus u _ { Y } \Big ) . } \end{array}\tag{222}
$$

The exogenous distributions are:

$$
\begin{array} { l l l } { { U _ { A } \sim \mathrm { B e r n } ( 0 . 4 4 ) } } & { { U _ { B } \sim \mathrm { B e r n } ( 0 . 4 8 ) } } & { { U _ { C } \sim \mathrm { B e r n } ( 0 . 3 6 ) } } \\ { { U _ { D } \sim \mathrm { B e r n } ( 0 . 3 8 ) } } & { { U _ { E } \sim \mathrm { B e r n } ( 0 . 4 7 ) } } & { { U _ { Y } \sim \mathrm { B e r n } ( 0 . 7 2 ) } } \\ { { U _ { B Y } \sim \mathrm { B e r n } ( 0 . 5 5 ) } } & { { U _ { A Y } \sim \mathrm { B e r n } ( 0 . 3 7 ) . } } & { { } } \end{array}\tag{223}
$$

Task 3. Let $\textbf { V } = \{ A , B , C , D , E , F , G , H , I , Y \}$ and $\textbf { U } = \{ U _ { V } : V \in \textbf { V } \} \cup$ ∪ $\left\{ U _ { F G } , U _ { H I } , U _ { A B } , U _ { B E } , U _ { I Y } \right\}$ . The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } \oplus u _ { A R } } \\ & { f _ { B } = - u _ { B R } \lor \left( a \lor \left( u _ { B E } \oplus u _ { A B } \right) \right) } \\ & { f _ { C } = - u _ { N } u _ { C } } \\ & { f _ { D } = c \land u _ { D } } \\ & { f _ { E } = - d \land \left( u _ { B E } \lor u _ { E } \right) } \\ & { f _ { F } = u _ { F G } \land \left( c \lor u _ { F } \right) } \\ & { f _ { H } = u _ { H } \oplus \left( u _ { H } \land \epsilon \right) } \\ & { f _ { G } = - \land \lor \left( e \lor \left( u _ { G } \lor u _ { F } G \right) \right) } \\ & { f _ { I } = \left( - \land u _ { H } \land \left( - \land u _ { Y } \odot u _ { I } \right) \right) \land f } \\ & { f _ { Y } = - \land \lor \left( u _ { H } , \lor \land \left( - f _ { N } \land \lor u _ { Y } \right) \lor h \right) \} . } \end{array}\tag{224}
$$

The exogenous distributions are:

$$
U _ { A } \sim \mathtt { B e r n ( 0 . 3 1 ) } U _ { B } \sim \mathtt { B e r n ( 0 . 7 9 ) } U _ { C } \sim \mathtt { B e r n ( 0 . 6 2 ) }
$$

$$
U _ { D } \sim \mathrm { B e r n } ( 0 . 6 5 ) \ U _ { E } \sim \mathrm { B e r n } ( 0 . 2 7 ) U _ { F } \sim \mathrm { B e r n } ( 0 . 2 5 )
$$

$$
U _ { G } \sim \mathrm { B e r n } ( 0 . 6 0 ) U _ { H } \sim \mathrm { B e r n } ( 0 . 4 0 ) U _ { I } \sim \mathrm { B e r n } ( 0 . 4 2 )\tag{225}
$$

$$
U _ { Y } \sim \mathrm { B e r n } ( 0 . 5 2 ) \quad U _ { F G } \sim \mathrm { B e r n } ( 0 . 4 5 ) \quad U _ { H I } \sim \mathrm { B e r n } ( 0 . 6 9 )
$$

$$
U _ { A B } \sim \mathrm { B e r n } ( 0 . 5 4 ) \quad U _ { B E } \sim \mathrm { B e r n } ( 0 . 7 3 ) \quad U _ { I Y } \sim \mathrm { B e r n } ( 0 . 6 8 ) .
$$

Task 4. Let $\mathbf { V } = \{ A , B , C , D , E , Y \}$ . The exogenous variables are:

$$
\mathbf { U } = \{ U _ { V } : V \in \mathbf { V } \} \cup \{ U _ { A Y } , U _ { A C } , U _ { B Y } \} .\tag{226}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { B } = u _ { B } \oplus u _ { B Y } } \\ & { f _ { D } = u _ { D } \vee b } \\ & { f _ { A } = u _ { A } \vee \left( \left( - d \oplus \left( u _ { A Y } \wedge u _ { A C } \right) \right) \wedge b \right) } \\ & { f _ { E } = \neg d \vee \left( \neg u _ { E } \oplus b \right) } \\ & { f _ { C } = \neg b \wedge \left( \neg u _ { A C } \oplus \left( u _ { C } \oplus e \right) \right) } \\ & { f _ { Y } = \left( \neg u _ { Y } \vee \left( b \vee \left( u _ { B Y } \vee u _ { A Y } \right) \right) \right) \oplus c . } \end{array}\tag{227}
$$

The exogenous distributions are:

$$
\begin{array} { r l r } { U _ { A } \sim \mathtt { B e r n } ( 0 . 3 7 ) } & { { } \ U _ { B } \sim \mathtt { B e r n } ( 0 . 3 7 ) } & { \ U _ { C } \sim \mathtt { B e r n } ( 0 . 4 7 ) } \\ { U _ { D } \sim \mathtt { B e r n } ( 0 . 3 2 ) } & { { } \ U _ { E } \sim \mathtt { B e r n } ( 0 . 3 2 ) } & { \ U _ { Y } \sim \mathtt { B e r n } ( 0 . 5 1 ) } \\ { U _ { A Y } \sim \mathtt { B e r n } ( 0 . 2 5 ) } & { { } U _ { A C } \sim \mathtt { B e r n } ( 0 . 4 9 ) } & { U _ { B Y } \sim \mathtt { B e r n } ( 0 . 4 2 ) . } \end{array}\tag{228}
$$

Task 5. Let $\mathbf { V } = \{ A , B , C , D , E , F , G , Y \}$ . The exogenous variables are:

$$
\mathbf { U } = \{ U _ { V } : V \in \mathbf { V } \} \cup \{ U _ { F G } , U _ { A B } , U _ { B Y } , U _ { C D } \} .\tag{229}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { C } = \lnot u _ { C D } \lor u _ { C } } \\ & { f _ { D } = \lnot u _ { C D } \lor u _ { D } \lnot \phi c } \\ & { f _ { E } = d \oplus u _ { E } } \\ & { f _ { A } = \lnot u _ { A } \lor \ l \bigl ( \lnot u _ { B } \oplus e \bigr ) } \\ & { f _ { F } = e \lor \ l \bigl ( u _ { F } \land u _ { F G } \bigr ) } \\ & { f _ { B } = \ l \bigl ( f \lor \ l ( \lnot u _ { B } \lor A _ { A B } \bigr ) \bigr ) \lor u _ { B } } \\ & { f _ { G } = u _ { G } \oplus \ l \bigl ( \ l ( \lnot u _ { F G } \lor a ) \bigr . \lor e \bigr ) } \\ & { f _ { Y } = u _ { B Y } \land \ l \bigl ( \ l \bigl ( f \lor h \bigr ) \oplus \bigl ( u _ { Y } \lor a \bigr ) \bigr ) . } \end{array}\tag{230}
$$

The exogenous distributions are:

$$
U _ { A } \sim \mathrm { B e r n } ( 0 . 7 5 ) \qquad U _ { B } \sim \mathrm { B e r n } ( 0 . 6 9 ) \qquad U _ { C } \sim \mathrm { B e r n } ( 0 . 2 6 )
$$

$$
U _ { D } \sim \mathtt { B e r n ( 0 . 4 2 ) } U _ { E } \sim \mathtt { B e r n ( 0 . 2 2 ) } U _ { F } \sim \mathtt { B e r n ( 0 . 5 3 ) }
$$

$$
U _ { G } \sim \mathrm { { B e r n } ( 0 . 6 8 ) } U _ { Y } \sim \mathrm { { B e r n } ( 0 . 2 3 ) } U _ { F G } \sim \mathrm { { B e r n } ( 0 . 3 1 ) }\tag{231}
$$

$$
U _ { A B } \sim { \tt B e r n } ( 0 . 4 2 ) U _ { B Y } \sim { \tt B e r n } ( 0 . 3 5 ) U _ { C D } \sim { \tt B e r n } ( 0 . 6 8 ) .
$$

Task 6. Let $\mathbf { V } = \{ A , B , C , D , E , F , G , Y \}$ . The exogenous variables are:

$$
\mathbf { U } = \{ U _ { V } : V \in \mathbf { V } \} \cup \{ U _ { A Y } , U _ { E F } , U _ { F G } \} .\tag{232}
$$

The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A Y } \wedge u _ { A } } \\ & { f _ { B } = u _ { B } } \\ & { f _ { F } = u _ { F G } \oplus \left( u _ { E F } \oplus u _ { F } \right) } \\ & { f _ { C } = \left( f \wedge \neg u _ { C } \right) \oplus a } \\ & { f _ { D } = u _ { D } \vee f } \\ & { f _ { E } = \neg u _ { E F } \oplus \left( \neg d \oplus \left( \neg b \wedge u _ { F } \right) \right) } \\ & { f _ { G } = \left( u _ { G } \wedge c \right) \wedge u _ { F G } } \\ & { f _ { Y } = \left( e \oplus u _ { Y } \right) \oplus \left( d \vee \left( u _ { A Y } \vee g \right) \right) . } \end{array}\tag{233}
$$

The exogenous distributions are:

$$
U _ { A } \sim \mathsf { B e r n } ( 0 . 4 8 ) \quad \quad U _ { B } \sim \mathsf { B e r n } ( 0 . 5 7 ) \quad \quad U _ { C } \sim \mathsf { B e r n } ( 0 . 4 0 )
$$

$$
U _ { D } \sim \mathsf { B e r n } ( 0 . 6 0 ) \qquad U _ { E } \sim \mathsf { B e r n } ( 0 . 3 9 ) \qquad U _ { F } \sim \mathsf { B e r n } ( 0 . 6 7 )
$$

$$
U _ { G } \sim \mathtt { B e r n } ( 0 . 7 7 ) U _ { Y } \sim \mathtt { B e r n } ( 0 . 6 0 ) U _ { A Y } \sim \mathtt { B e r n } ( 0 . 2 1 )\tag{234}
$$

$$
U _ { E F } \sim \mathtt { B e r n ( 0 . 5 7 ) } \quad U _ { F G } \sim \mathtt { B e r n ( 0 . 6 0 ) } .
$$

## Appendix B. Supporting Results for Transportability

## B.1 Omitted Proofs

This section proves the soundness and regret guarantees from Sec. 5. Source distributions and the resulting causal bounds are population quantities. All fixed arguments of a causal factor remain fixed during its marginalization or extraction.

## B.1.1 Proof of Theorem 10

The source-selection test has a useful equivalent form once the graphical reductions in paTR have been applied.

Lemma 41 (Proposition 1 in Lee et al. (2020)). Suppose that $\mathbf { V } \setminus \mathbf { X } = \mathrm { A n } ( \mathbf { Y } ) _ { \mathcal { G } \setminus \mathbf { X } }$ . Then $( \mathbf { S } ^ { i } \perp \perp \mathbf { Y } ) _ { \mathcal { G } ^ { \Delta } \backslash \mathbf { X } }$ if and only if $\Delta ^ { i } \cap ( \mathbf { V } \setminus \mathbf { X } ) = \emptyset$

Proof of Lemma 41. If $\Delta ^ { i } \cap ( \mathbf { V } \setminus \mathbf { X } ) = \emptyset$ , removing X removes every child of $\mathbf { S } ^ { i }$ , giving the separation. Otherwise, a remaining child $V \in \Delta ^ { i } \setminus \mathbf { X }$ has a directed path to Y by the assumed ancestry condition. Its selection edge followed by that path is unblocked, so separation fails. □

Proof of Thm. 10. We first check the transformations in paTR, then the expressions supplied by paID. The direct-source branch, ancestral restriction, addition of irrelevant interventions, and c-component decomposition follow the sound transformations of general transportability (Lee et al., 2020).

The first branch uses the source experiment only when its selection nodes are separated from the query. Removing non-ancestors leaves its distribution unchanged, as does fixing the remaining non-ancestors by Rule 3 of do-calculus. Decomposition expresses the query as a sum of products of causal factors. Source experiments may difer between factors because each expression bounds the corresponding target factor.

Each graphical reduction forwards its recursive expressions and ends the current call. Consequently, when the source loop is reached:

$$
\mathbf { C } = \mathbf { V } \setminus \mathbf { X } = \mathsf { A n } ( \mathbf { Y } ) _ { \mathcal { G } \backslash \mathbf { X } } , \qquad \mathsf { c c } _ { \mathcal { G } \backslash \mathbf { X } } = \{ \mathbf { C } \} .\tag{235}
$$

The source test ensures $\Delta ^ { i } \cap \mathbf { C } = \emptyset$ and hence $Q ^ { * } [ \mathbf { C } ] = Q ^ { i } [ \mathbf { C } ]$ . Lemma 41 also shows why this test agrees with the separation test at this stage. The selected experiment fixes only variables in X among the current vertices. Removing those variables therefore leaves $\mathbf { C } ,$ its induced graph, and Eq. (235) unchanged.

Consider the resulting call to paID. Its input P is an available source distribution marginalized to its current vertices, with the experimental arguments fixed. Subsequent ancestral marginalization and c-component extraction make each input distribution or causal factor an identifiable source factor $Q ^ { i } [ \mathbf { V } _ { l } ]$ , where the subscript l denotes the variables and graph local to that call. Ancestral marginalization and the products of topologically ordered conditionals are the causal factor identities of Tian and Pearl (2003).

The set of non-intervened vertices remains C throughout these calls. Indeed, each vertex in C has a directed path to Y entirely inside C, so an ancestral restriction keeps all of C. An enclosing c-component $\mathbf { C ^ { \prime } }$ contains C, so restriction to $\mathbf { C ^ { \prime } }$ also keeps these paths and the same induced graph on C. Thus the singleton component specified by paID is always well defined. The source invariance of Q[C] is preserved as well. In particular, the query at every such call is the marginal $\begin{array} { r } { \sum _ { \mathbf { c } \backslash \mathbf { Y } } Q ^ { i } [ \mathbf { C } ] = \sum _ { \mathbf { c } \backslash \mathbf { Y } } Q ^ { * } [ \mathbf { C } ] } \end{array}$ with its inherited arguments fixed.

With no remaining intervention, the marginal of $P$ is exact. Extraction and marginalization are also exact when C is a c-component of the current graph. It remains to check the branch where the whole current graph is one c-component. Write $P _ { l } = Q ^ { i } [ \mathbf { V } _ { l } ]$ and $\mathbf { C } = \mathbf { V } _ { l } \setminus \mathbf { X } _ { l }$ Apply Prop. 11 with both smaller sets equal to C and the enclosing set equal to $\mathbf { V } _ { l }$ . Therefore, the following pointwise inequality holds:

$$
P l ( \mathbf { v } _ { l } ) \leq Q ^ { * } [ \mathbf { C } ] = Q ^ { i } [ \mathbf { C } ] \leq P l ( \mathbf { v } _ { l } ) + 1 - \sum _ { \mathbf { c } } P l ( \mathbf { v } _ { l } ) .\tag{236}
$$

The inner sum varies only C, holding the local intervention and all external arguments fixed. Summing over C \ Y gives exactly the lower or upper expression yielded by the fallback branch.

Every yielded bound is nonnegative: the consistency upper expression adds to $P _ { l }$ the complement of a marginal probability. Products of nonnegative lower bounds, or of nonnegative upper bounds, therefore preserve their inequality, as does summation. Induction through the c-component products and the preceding exact transformations proves soundness for every expression yielded by the original call. If a recursive call returns no expression, it contributes no candidate bound. Only expressions whose required source conditionals are defined are evaluated. These conventions can remove a candidate formula without invalidating any retained bound. □

## B.1.2 Proof of Theorem 12

Proof of Thm. 12. Let $\mathcal { A } ^ { \ast }$ be the retained action space, let $\kappa = | { \mathcal { A } } ^ { * } |$ , and assume $T \geq \kappa$ The causal and dominance bounds ensure that an optimal target action $\mathbf { x } ^ { \star }$ is retained and that, for every retained action:

$$
\ell _ { \mathbf { x } } \le \mu _ { \mathbf { x } } ^ { * } \le b _ { \mathbf { x } } = \operatorname* { m i n } \{ u _ { \mathbf { x } } , u ^ { \star } \} .\tag{237}
$$

Since $0 \leq Y \leq 1$ implies $Y ^ { 2 } \leq Y$

$$
\mathrm { V a r } [ Y \mid d o ( \mathbf { x } ) ] \leq \mu _ { \mathbf { x } } ^ { * } ( 1 - \mu _ { \mathbf { x } } ^ { * } ) \leq \sigma _ { \mathbf { x } } ^ { 2 } .\tag{238}
$$

Expose for each action an independent sequence of rewards with its target distribution. Each time that action is selected, the policy reveals the next element of its sequence. Fresh independent exogenous draws at each round justify this representation. Write $\widehat { m } _ { { \bf x } , n }$ for the average of the first n rewards in the sequence and set:

$$
\zeta _ { T } = \log ( 2 \kappa T ^ { 2 } ) , \qquad r _ { \mathbf { x } } ( n ) = \sqrt { \frac { 2 { \sigma } _ { \mathbf { x } } ^ { 2 } \zeta _ { T } } { n } } + \frac { 2 \zeta _ { T } } { 3 n } .\tag{239}
$$

Bernstein’s inequality gives:

$$
\begin{array} { r } { \operatorname* { P r } \left( | \widehat { m } _ { \mathbf { x } , n } - \mu _ { \mathbf { x } } ^ { * } | > r _ { \mathbf { x } } ( n ) \right) \leq 2 e ^ { - \zeta _ { T } } . } \end{array}\tag{240}
$$

Taking a union bound over all κ actions and $1 \leq n \leq T$ yields:

$$
\mathcal { E } = \bigcap _ { \mathbf { x } \in \mathcal { A } ^ { \ast } } \bigcap _ { n = 1 } ^ { T } \left\{ | \widehat { m } _ { \mathbf { x } , n } - \mu _ { \mathbf { x } } ^ { \ast } | \leq r _ { \mathbf { x } } ( n ) \right\} ,\tag{241}
$$

$$
\operatorname* { P r } ( \mathcal { E } ^ { c } ) \leq 2 \kappa T e ^ { - \zeta _ { T } } = \frac { 1 } { T } .\tag{242}
$$

This simultaneous event applies to the adaptively chosen counts $N _ { t - 1 } ( \mathbf { x } )$ , at which $\widehat { m } _ { \mathbf { x } , N _ { t - 1 } ( \mathbf { x } ) } = \widehat { \mu } _ { \mathbf { x } , t - 1 } ^ { * }$ . The linear term in $r _ { \mathbf { x } }$ is needed because the variance bound alone is not a sub-Gaussian variance proxy.

On E, both the empirical UCB and the clipping level of an optimal action are at least its mean. Thus:

$$
\bar { U } _ { \mathbf { x } ^ { \star } } ( t ) \geq \mu _ { \mathbf { x } ^ { \star } } ^ { \ast } \qquad ( t > \kappa ) .\tag{243}
$$

A retained action with $b _ { \mathbf { x } } < \mu _ { \mathbf { x } } ^ { * }$ <sub>⋆</sub> has a strictly smaller clipped index at every such round, so it is played only once during initialization on E. An action pruned before initialization is never played.

Now take a retained action with $b _ { \mathbf { x } } \geq \mu _ { \mathbf { x } } ^ { * }$ <sub>⋆</sub> and gap $\Delta _ { \mathbf { x } } > 0$ . If $A _ { t } = \mathbf { x }$ after initialization, index maximization and the confidence event imply, with $n = N _ { t - 1 } ( \mathbf { x } )$

$$
\begin{array} { r } { \mu _ { \mathbf { x } ^ { \star } } ^ { * } \leq \bar { U } _ { \mathbf { x } } ( t ) \leq U _ { \mathbf { x } } ( t ) \leq \mu _ { \mathbf { x } } ^ { * } + 2 r _ { \mathbf { x } } ( n ) . } \end{array}\tag{244}
$$

Selection therefore requires $\Delta _ { \mathbf { x } } \leq 2 r _ { \mathbf { x } } ( n )$

Define:

$$
q _ { \mathbf { x } } = \frac { 3 2 \sigma _ { \mathbf { x } } ^ { 2 } \zeta _ { T } } { \Delta _ { \mathbf { x } } ^ { 2 } } + \frac { 8 \zeta _ { T } } { 3 \Delta _ { \mathbf { x } } } .\tag{245}
$$

If $n > q _ { \mathbf { x } } .$ , the square-root and linear terms in $2 r _ { \mathbf { x } } ( n )$ are each strictly smaller than $\scriptstyle { \frac { \Delta \mathbf { x } } { 2 } }$ , so selection is impossible. Any selection after initialization has a preceding count of at most $\lfloor q _ { \mathbf { x } } \rfloor$ . Accounting also for the initialization pull therefore gives:

$$
N _ { T } ( \mathbf { x } ) \leq 1 + q _ { \mathbf { x } } \quad \mathrm { o n } ~ \mathcal { E } .\tag{246}
$$

On $\mathcal { E } ^ { c } , N _ { T } ( \mathbf { x } ) \leq T$ . Consequently, the expected number of pulls is at most two for a retained action with $b _ { \mathbf { x } } < \mu _ { \mathbf { x } } ^ { * } ,$ <sub>⋆</sub>, and at most $2 + q _ { \mathbf { x } }$ for any other retained suboptimal action.

For regret, charge failure once for the entire process. On $\mathcal { E } ,$ initialization contributes $\textstyle \sum _ { \mathbf { x } \in { \mathcal { A } } ^ { * } } \Delta _ { \mathbf { x } }$ , and only retained suboptimal actions with $b _ { \mathbf { x } } \geq \mu _ { \mathbf { x } } ^ { * }$ <sub>⋆</sub> can contribute further. Since $R _ { T } \leq T$ and $\begin{array} { r } { \operatorname* { P r } ( \mathcal { E } ^ { c } ) \leq \frac { 1 } { T } } \end{array}$ , the failure contribution is at most one. The regret decomposition therefore gives:

$$
\mathbb { E } R _ { T } \leq \sum _ { \mathbf { x } \in A ^ { * } } \Delta _ { \mathbf { x } } + \zeta _ { T } \sum _ { \mathbf { x } \in A ^ { * } \setminus \Omega _ { \mathbf { x } } \setminus \Omega _ { \mathbf { x } } \setminus \Omega } \left( \frac { 3 2 \sigma _ { \mathbf { x } } ^ { 2 } } { \Delta _ { \mathbf { x } } } + \frac { 8 } { 3 } \right) + 1 .\tag{247}
$$

This is the claimed bound. Since $u ^ { \star } \geq \mu _ { \mathbf { x } ^ { \star } } ^ { * }$ , the condition $b _ { \mathbf { x } } \geq \mu _ { \mathbf { x } } ^ { * }$ <sub>⋆</sub> is equivalent to $u _ { \bf x } \geq$ $\mu _ { \mathbf { x } ^ { \star } } ^ { * }$ □

## Appendix C. Supporting Results for Counterfactual Structural Causal Bandits

## C.1 Omitted Proofs

## C.1.1 Proof of Theorem 13

We assume that ${ \bf X } _ { * * }$ is realizable and every assigned counterfactual is nonconstant, i.e., after efectiveness, it still requires its target mechanism.

The proof follows the two parts of the graphical condition. If either part fails, we identify an assignment that can be removed without changing the reward. If both hold, we use the binary SCM of Lemma 35 to make every proper subset give a strictly smaller mean reward. One such SCM is enough to rule out equality in every compatible SCM.

Proof of Thm. 13. (Only if) We show that failure of either condition permits deleting one counterfactual from $\mathbf { X } _ { * * }$

Failure of the first condition. Suppose the first condition fails for $X _ { i }$ . Let $\mathbf { X } _ { * * } ^ { \prime }$ omit the counterfactual assigned to $X _ { i }$ , keeping all other terms. Every directed path from $X _ { i }$ to $Y$ then meets a target whose counterfactual is still assigned. Consequently, changing the value assigned to $X _ { i }$ cannot afect the reward once the values at those other targets have been supplied.

To apply ctf-Rule 3 explicitly, a path starting at $X _ { i }$ cannot use an edge entering $X _ { i }$ . The failed ancestor condition therefore gives $X _ { i } \notin { \frac { d } { \pi } }$ An $( Y ) _ { { \mathcal G } _ { \overline { { \mathbf { X } \backslash \{ X _ { i } \} } } } }$ . Therefore, we have:

$$
P ( Y _ { \mathbf { X _ { * * } } } = y ) = \sum _ { \mathbf { x } } P ( y _ { \mathbf { x } } , \mathbf { X _ { * * } } = \mathbf { x } )
$$

$$
\mathrm { C U T }\tag{248}
$$

$$
= \sum _ { \mathbf { x } } { P ( y _ { \mathbf { x } \backslash \{ X _ { i } \} } , \mathbf { X } _ { * * } = \mathbf { x } ) }
$$

$$
\mathrm { C T F  – R u l e \ 3 }\tag{249}
$$

$$
= \sum _ { \mathbf { x } \backslash \{ X _ { i } \} } P ( y _ { \mathbf { x } \backslash \{ X _ { i } \} } , \mathbf { X } _ { * * } ^ { \prime } = \mathbf { x } \backslash \{ X _ { i } \} )
$$

$$
{ \mathrm { m a r g . ~ o v e r ~ } } X _ { i }\tag{250}
$$

$$
\begin{array} { r } { = P ( Y _ { \mathbf { X } _ { * * } ^ { \prime } } = y ) . } \end{array}
$$

$$
\mathrm { C U T }\tag{251}
$$

Counterfactuals inside the remaining terms are unchanged. Only the separate assignment to $X _ { i }$ has disappeared. Thus the deletion preserves the reward on every unit and introduces no new required evaluation. The resulting action is realizable and has the same mean reward. If no assignment remains, its reward is simply $Y$

Failure of the second condition. Suppose the first condition holds but the second fails for $X _ { i }$ . Again, let $\mathbf { X } _ { * * } ^ { \prime }$ remove only the counterfactual assigned to $X _ { i }$ . The question is now whether computing $X _ { i }$ through its mechanism, with the remaining assignments in place, reproduces that counterfactual.

For example, consider $\{ X _ { w } , Z _ { w } \}$ in Fig. 29b. The mechanism of Z still receives $X _ { w } .$ , together with the same natural value of $T$ and the same exogenous realization. It therefore produces $Z _ { w }$ without a separate assignment to $Z .$ . As explained in the main text, $Z _ { w } = Z _ { X _ { w } }$ and thus $Y _ { X _ { w } , Z _ { w } } = Y _ { X _ { w } }$ . In this example, the second condition fails for $Z$ because the fixed value w reaches its computation only through the other target X.

For the general case, compare the computations in three steps:

1. Starting at $X _ { i } ,$ follow parent edges backwards in $\mathcal { G } _ { \mathbf { X } _ { * * } }$ . Stop a branch when another target in $\mathbf { X } \setminus \{ X _ { i } \}$ is reached. Its assigned counterfactual is still supplied under $\mathbf { X } _ { * * } ^ { \prime }$

2. At each vertex V visited before these stopping points, failure of the second condition gives $\mathtt { P a } ( V ) _ { \mathcal { G } _ { \mathbf { X } * * } } = \mathtt { P a } ( V ) _ { \mathcal { G } }$ . Thus no parent value received by $f _ { V }$ was replaced by a constant. Each such value is calculated by its parent mechanism or supplied by the counterfactual at a stopping point. At a stopping point, realizability ensures that the counterfactual needed by this calculation is the one assigned to that remaining target.

3. Evaluate the visited mechanisms in topological order. The values supplied at the stopping points are unchanged. Every other parent value is obtained from an earlier mechanism whose value already agrees, and the exogenous realization is the same. Hence each mechanism returns the same value, including $f _ { X _ { i } }$

Consequently, removing the assignment to $X _ { i }$ reproduces its assigned counterfactual through the same mechanisms. ctf-Rule 1 gives $Y _ { \mathbf { X } _ { * * } } = Y _ { \mathbf { X } _ { * * } ^ { \prime } }$ . The deletion preserves realizability because it reuses evaluations already needed by the original action. Both cases establish equality for every compatible SCM and every admissible subscript assignment, proving necessity.

(If) Suppose both conditions hold. We will use the second condition to obtain a strict diference at a removed target, and the first condition to carry that diference to $Y$

Step 1: Choose the binary SCM and set the subscript values. We first take M from Lemma 35 and set every constant subscript value to one. This choice respects the equalities required by realizability. Recall three properties of the construction:

• Every observed variable is binary. Increasing a parent value while keeping the exogenous realization fixed cannot decrease the child’s value.

• Every parent has a strictly positive coeficient $\beta _ { P V }$ in the expectation formula. A positive diference in a parent’s mean therefore increases the child’s mean when the other diferences are nonnegative.

• Every value computed by a mechanism has mean strictly between zero and one. In particular, it does not equal one with probability one, even when all its parent values are one.

Fix any proper subset $\mathbf { X } _ { * * } ^ { \prime } \subsetneq \mathbf { X } _ { * * }$ . We compare it with the original set in this same SCM, using the same exogenous realization and leaving every remaining counterfactual unchanged.

Step 2: Compare the values used in the two calculations. First consider a target whose counterfactual remains in $\mathbf { X } _ { * * } ^ { \prime }$ . Its assigned value is identical in the two rewards. For every other mechanism $f _ { V }$ required by the original action, $V$ is evaluated under the subset through its structural mechanism. Compare the values received from each parent as follows:

• If $\mathbf { X } _ { * * }$ fixes the value received from $P ,$ that value is one. The corresponding value under the subset is $P _ { \mathbf { X } _ { \ast } ^ { \prime } }$ , which is either zero or one.

• If the value received from P is calculated by its mechanism, compare that calculation with $P _ { \mathbf { X } _ { * * } ^ { \prime } }$ first. In topological order, this parent comparison has already been made before evaluating its child.

Starting with the earliest mechanisms, monotonicity therefore makes every value required by the original action at least as large as its counterpart in the subset’s reward calculation. In particular, for each target:

$$
X _ { i [ \mathbf { X } _ { * * } ] } ( \mathbf { u } ) \geq X _ { i [ \mathbf { X } _ { * * } ^ { \prime } ] } ( \mathbf { u } ) .\tag{252}
$$

For the next step, the notation $\mathbb { E } _ { M _ { \mathbf { X } _ { * * } } } V$ denotes the mean of the value returned by the mechanism for V when computing the original action. The notation $\mathbb { E } _ { \mathcal { M } } V _ { \mathbf { X } _ { \ast } ^ { \prime } }$ denotes the mean of V in the subset’s reward calculation. These may refer to diferent counterfactuals. For instance, computing $Z _ { w }$ may require $X _ { w }$ even when the reward after a deletion also receives the natural X. The comparison above keeps these calculations distinct. It remains valid when the subset is not realizable, because both counterfactuals are defined in the same SCM.

Step 3: Find a strict diference where a parent value was fixed. Consider a required mechanism $f _ { V }$ for which the subset supplies no assigned counterfactual at V. There are two contributions to the diference of its means. A parent edge deleted from $\mathcal { G } _ { \mathbf { X } } ,$ contributes the diference between the fixed value one and the calculated value $P _ { \mathbf { X } _ { \ast } ^ { \prime } }$ . A parent edge retained in that graph contributes the diference between the two calculated values of $P .$ The expectation formula from Lemma 35 gives:

$$
\mathbb { E } _ { \mathcal { M } _ { \mathbf { X } _ { * * } } } V - \mathbb { E } _ { \mathcal { M } } V _ { \mathbf { X } _ { * * } ^ { \prime } }\tag{253}
$$

$$
= \sum _ { P \in \mathsf { P a } ( V ) \varsigma \backslash \mathsf { P a } ( V ) \varsigma _ { \mathbf { X } _ { * * } } } \beta _ { P V } ( 1 - \mathbb { E } _ { \mathcal { M } } P _ { \mathbf { X } _ { * * } ^ { \prime } } )\tag{254}
$$

$$
+ \sum _ { P \in \mathsf { P a } ( V ) _ { \mathcal { G } _ { \mathbf { X } _ { * * } } } } \beta _ { P V } ( \mathbb { E } _ { \mathcal { M } _ { \mathbf { X } _ { * * } } } P - \mathbb { E } _ { \mathcal { M } } P _ { \mathbf { X } _ { * * } ^ { \prime } } ) .\tag{255}
$$

The first sum is over the parent values replaced by one; the second is over the parent values still computed by their mechanisms. The constant and shared latent terms cancel because the structural function and exogenous distribution are unchanged. Every summand is nonnegative by Step 2.

Moreover, every $P _ { \mathbf { X } _ { \ast } ^ { \prime } }$ has mean strictly below one. If P is a remaining target, its counterfactual still requires its mechanism by assumption. Otherwise, its mechanism is evaluated with the values produced under the subset. In both cases, the threshold of the binary construction stays strictly between zero and one. Therefore, whenever $f _ { V }$ receives a fixed parent value in the original action, the corresponding summand in the first sum is strictly positive.

The expectation calculation remains valid for nested counterfactuals. The private noise $U _ { V }$ is independent of all values supplied by parent mechanisms, since these use only ancestral mechanisms and their exogenous variables. Parents may be correlated with shared latent variables; the calculation uses independence only of $U _ { V }$

Step 4: Use the second condition to reach a removed target. Choose any target $X _ { i }$ whose counterfactual was removed. The second condition provides a vertex V that receives a fixed parent value and a retained directed path from V to $X _ { i }$ . This path, including V , avoids every other target in X, so none of its mechanisms is replaced by a remaining assignment. Step 3 gives a strictly positive diference at V . At each next vertex on the path, the contribution of the preceding parent appears in Eq. (255), multiplied by a strictly positive coeficient. All other contributions are nonnegative. The diference therefore stays strictly positive until it reaches $X _ { i }$ . If $V = X _ { i } ,$ , no propagation is needed.

Step 5: Use the first condition to reach the reward. The first condition provides a directed path from $X _ { i }$ to Y that avoids every other target in X. After $X _ { i } ,$ , these mechanisms are evaluated in both rewards with their original structural mechanisms. The strict diference at $X _ { i }$ therefore propagates along this path to Y by the same positive coeficient calculation. The other parent diferences are nonnegative by Eq. (252) and monotonicity. Hence $\mu _ { \mathbf { X } _ { * * } } >$ $\mu \mathbf { { X } } _ { * * } ^ { \prime }$

The proper subset was arbitrary: it may remove several assignments, and the resulting counterfactual need not be realizable. The same SCM and the same subscript values, all equal to one, distinguish the original set from every such subset. Thus no proper subset preserves its mean reward for every compatible SCM and every admissible assignment. This proves that $\mathbf { X } _ { * * }$ is a ctf-MIS. □

## C.1.2 Proof of Proposition 14

Proof of Prop. $1 \not \angle \cdot$ Fix an assigned counterfactual $\mathbf { X } _ { * * }$ and condition on the values received by the parent inputs of $f _ { Y }$ . These values are determined by upstream exogenous variables and fixed constants, independently of the exogenous inputs of Y. The conditional reward mean is therefore $\mathbb { E } Y _ { \mathbf { p a } _ { Y } }$ . Averaging over the received values gives:

$$
\mu _ { \mathbf { X } _ { * * } } \leq \operatorname* { m a x } _ { \mathbf { p a } _ { Y } } \mathbb { E } Y _ { \mathbf { p a } _ { Y } } = \mu _ { \mathbf { p a } _ { Y } ^ { * } } .\tag{256}
$$

Consequently, no non-equivalent counterfactual can be strictly better than the best parent intervention. To establish that the parent intervention set itself is a ctf-POMIS, we also need one SCM in which the comparison is strict against every non-equivalent counterfactual. Choose binary domains and independent private noises $U _ { V } \sim$ Bern(ϵ) for $V \neq Y$ and $U _ { Y } \sim \mathtt { B e r n } ( 1 - \epsilon )$ , where $0 < \epsilon < 1$ . Set $f _ { V } = U _ { V } \prod _ { P \in \mathsf { P a } ( V ) _ { G } } P$ for $V \in \mathbf { V }$ , where an empty product equals one. Fixing every parent input of Y to one gives mean $1 - \epsilon .$ Any non-equivalent counterfactual leaves some input $P  Y$ natural: if all these inputs are fixed, every upstream assignment is irrelevant to the reward. The natural output supplied by P is at most $U _ { P }$ , whatever its other inputs. Hence every choice of subscript values has mean at most $\epsilon ( 1 - \epsilon ) < 1 - \epsilon$ . This single SCM establishes strictness against all non-equivalent counterfactuals. When $\mathtt { P a } ( Y ) _ { \mathcal { G } } = \varnothing$ , every input assignment is irrelevant and every counterfactual is equivalent to the empty action. □

## C.1.3 Proof of Theorem 14

The two proof directions of Thm. 14 use diferent properties of the minimal UC-territory. When its border is empty, we construct an SCM in which the given counterfactual preserves useful dependence between the reward and its ancestors. Replacing a retained input by a constant breaks this dependence. When the border is nonempty, the inputs entering the territory are independent of its exogenous variables. Their random values can then be replaced by the best fixed values.

![](images/6e5034ddad4ddcf1fa8013b69b34583c486898ec4a9c7bf2492c1819a3d46bd0.jpg)  
Figure 39: The SCM construction for ctf-POMIS. (a) The original causal diagram. (b) B carries the blue value $u _ { 1 } .$ , C carries the red value $1 - u _ { 1 }$ , and the purple vertex Y checks the two values. (c) D and C carry the red value $1 - u _ { 2 } ; Y$ compares it with its blue exogenous input $u _ { 2 }$

<table><tr><td colspan="2"></td><td colspan="3"> $\mathcal { M } _ { 1 }$ </td><td colspan="3"> $\mathcal { M } _ { 2 }$ </td><td colspan="5">M</td></tr><tr><td> $U _ { 1 }$ </td><td> $U _ { 2 }$ </td><td> $B ^ { ( 1 ) }$ </td><td>C(1)</td><td> $Y ^ { ( 1 ) }$ </td><td> $D ^ { ( 2 ) }$ </td><td> $C ^ { ( 2 ) }$ </td><td> $Y ^ { ( 2 ) }$ </td><td>B</td><td>D</td><td>C</td><td>Y&#x27;</td><td>Y</td></tr><tr><td>0</td><td>0</td><td>0</td><td>1</td><td>2</td><td>1</td><td>1</td><td>2</td><td>00 00</td><td>00 01</td><td>01 01</td><td>1010</td><td>1</td></tr><tr><td>0</td><td>1</td><td>0</td><td>1</td><td>2</td><td>0</td><td>0</td><td>1</td><td>00 00</td><td>00 00</td><td>0100</td><td>1001</td><td>1</td></tr><tr><td>1</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td><td>2</td><td>0100</td><td>0001</td><td>00 01</td><td>0110</td><td>1</td></tr><tr><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>1</td><td>0100</td><td>0000</td><td>00 00</td><td>0101</td><td>1</td></tr></table>

Table 13: Values under $\{ B _ { a } , C _ { a ^ { \prime } , e } \}$ in Fig. 39, with $a \neq a ^ { \prime } .$ . Each coordinate is written using two bits, with the first component followed by the second; an inactive coordinate is 00. For example, $C = 0 1 0 1$ represents $( C ^ { ( 1 ) } , \bar { C } ^ { ( 2 ) } ) = ( 1 , 1 )$ . The intermediate value $Y ^ { \prime }$ records $( Y ^ { ( 1 ) } , Y ^ { ( 2 ) } )$ ). Both coordinates are 1 or 2 in every row, so the binary reward is $Y = 1$ on every unit.

An SCM construction. For each shared binary exogenous variable, blue and red vertices carry its value along the two sides of the corresponding bidirected edge. Purple vertices compare these values after the directed paths meet. The mechanisms retain a separate coordinate for each confounder, so all these comparisons take place in one SCM. Thus the colors specify the structural functions used in the proof.

Consider Fig. 39a. Fix the inputs $A \to B , A \to C$ , and $E  D$ to $a , a ^ { \prime }$ , and $e ,$ respectively. Choose $a \neq a ^ { \prime }$ and reserve these assigned values separately from the values naturally output by the mechanisms. Deleting the three selected edges and restricting to the ancestors of $Y$ leaves $\{ B , C , D , Y \}$ . Starting from Y, the bidirected edge $Y  D$ includes $D ,$ descendant closure includes C, and $C  B$ includes B. The retained graph is therefore its own minimal UC-territory, with empty border.

Let $U _ { 1 }$ be the fair bit shared by B and $C ,$ and let $U _ { 2 }$ be the independent fair bit shared by D and Y. In Fig. 39b, the blue vertex B carries $U _ { 1 }$ and the red vertex C carries $1 - U _ { 1 }$ . The purple vertex Y checks that these two values are complementary, recording 2 for (0, 1) and 1 for (1, 0). In Fig. 39c, the red vertices D and C carry $1 - U _ { 2 }$ . At Y, this value is checked against the genuine $U _ { 2 }$ available to its own mechanism. The same purple encoding records $2 - U _ { 2 }$ . Table 13 shows the calculations for $U _ { 1 }$ and $U _ { 2 }$ separately as $\mathcal { M } _ { 1 }$ and $\mathcal { M } _ { 2 }$ , then combines them into M. The two coordinates of B, D, and $C$ are $( U _ { 1 } , 0 ) , ( 0 , 1 - U _ { 2 } )$ , and $( 1 - U _ { 1 } , 1 - U _ { 2 } )$ , respectively; zero fills a coordinate unused at that vertex. The intermediate value $Y ^ { \prime }$ contains the two comparison results, and $Y = 1$ exactly when both are valid. Before making these calculations, the mechanisms of B, C, and D check their selected inputs against $a , a ^ { \prime } ,$ and e. Under the specified counterfactual, all checks succeed for every $( u _ { 1 } , u _ { 2 } )$ , giving reward one.

This example also explains why fixing a retained input can be harmful. If $B  Y$ is fixed while $C  Y$ remains natural, its first coordinate must agree with a fair bit carried through C. If $D  C$ is fixed while $C  Y$ remains natural, its second coordinate must agree with the genuine $U _ { 2 }$ at $Y$ . If $C  Y$ is fixed, its second coordinate must again agree with that $U _ { 2 }$ , even when the other inputs are fixed as well. Each comparison succeeds with probability at most $\textstyle { \frac { 1 } { 2 } }$ . If all retained inputs remain natural, omitting a selected input fails its assigned value check. The following construction applies these comparisons to arbitrary retained paths, including paths that merge.

Proof of Thm. $1 \not \angle \cdot$ (If) Suppose $\vert \mathsf { B } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) = \emptyset$ . Every vertex of the graph is ancestral to Y. If a vertex lay outside the minimal UC-territory, a directed path from it to Y would have a first edge entering the territory. The parent of this edge would belong to the border, a contradiction. Therefore:

$$
\mathbf { V } ^ { \dagger } = \mathbf { V } ( \mathcal { H } _ { \mathbf { X } _ { * * } } ) = \mathsf { M U C T } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) .\tag{257}
$$

Selected inputs whose children lie outside $\mathbf { V } ^ { \dagger }$ cannot afect Y and may be discarded. We compare the remaining selections of fixed inputs, allowing each selected value to vary freely and identifying equivalent counterfactual sets.

Choosing the assigned values. Choose the subscript values of $\mathbf { X } _ { * * }$ so that distinct selected occurrences of the same parent variable receive distinct values. Reserve separate values for the coordinate tuples used below and for an invalid output.

Each mechanism checks every input selected by $\mathbf { X } _ { * * }$ against its chosen value. No natural output will take one of these reserved assigned values. Consequently, a competing counterfactual cannot pass a selected-input check merely by leaving that input natural.

Constructing the mechanisms for one confounder. Let $U _ { 1 } , \cdots , U _ { k }$ be independent fair binary confounders, one for each bidirected edge of $\mathcal { H } _ { \mathbf { X } _ { * * } }$ . Denote the endpoints of $U _ { j }$ by $B ^ { ( j ) }$ and $R ^ { ( j ) }$ . Descendants in the following coloring are computed in the retained graph:

$$
\begin{array} { r l } { \mathtt { b l u e : } } & { { } \mathtt { D e } ( B ^ { ( j ) } ) \setminus \mathtt { D e } ( R ^ { ( j ) } ) , } \end{array}\tag{258}
$$

$$
\begin{array} { r l } { \mathtt { r e d } : } & { { } \mathtt { D e } ( R ^ { ( j ) } ) \setminus \mathtt { D e } ( B ^ { ( j ) } ) , } \end{array}\tag{259}
$$

$$
\begin{array} { r } { \mathrm { p u r p 1 e : ~ { \tt D e } } ( B ^ { ( j ) } ) \cap \mathtt { D e } ( R ^ { ( j ) } ) . } \end{array}\tag{260}
$$

The blue endpoint supplies $u _ { j }$ as an additional blue input to its mechanism, and the red endpoint supplies $1 - u _ { j }$ as an additional red input. This convention also applies when an endpoint is itself purple because it is a descendant of the other endpoint.

At a blue or red vertex, the mechanism returns the common value of its inputs of that color, provided they agree and belong to {0, 1}. At a purple vertex, the valid input combinations

and outputs are:

<table><tr><td>blue</td><td>red</td><td>purple</td><td>output</td></tr><tr><td>0</td><td>1</td><td>2</td><td>2</td></tr><tr><td>1</td><td>0</td><td>1</td><td>1</td></tr></table>

(261)

Each entry specifies the common value required of all inputs of that color; a color with no input imposes no condition. An unequal or malformed input, a failed selected-input check, or an invalid parent output gives the invalid value 3. Invalid outputs propagate along every retained directed edge. A vertex outside the two descendant sets has an unused zero coordinate.

These rules have a simple interpretation. Blue values encode $u _ { j } .$ , red values encode $1 - u _ { j }$ and purple values encode $2 - u _ { j }$ . Thus every mechanism compares copies of the same bit, with the encoding determined by the sender’s color. For example, a purple input equal to 2 is consistent with a blue input equal to 0 and a red input equal to 1. Keeping the purple value allows subsequent mechanisms to make the same comparison after paths have already merged.

Combining the mechanisms into one SCM. Represent each variable in $\mathbf { V } ^ { \dagger } \setminus \{ Y \}$ by the k coordinates just constructed, using two bits per coordinate as in Table 13. Its mechanism applies all selected-input checks and all coordinate comparisons. A coordinate tuple with an incorrect number of coordinates or a nonzero unused coordinate is invalid. Any failure makes the whole output invalid. The mechanism of $Y$ makes the corresponding comparisons, including any exogenous bits it receives directly, and returns one exactly when every check succeeds. Natural mechanisms outside $\mathbf { V } ^ { \dagger }$ also avoid the reserved assigned values. Other allowed exogenous inputs may be ignored.

Under the chosen subscript values, every coordinate carries the genuine shared bit in its designated encoding, and every selected-input check succeeds. Hence:

$$
\mu _ { \mathbf { X } _ { * * } } = \mu _ { \mathbf { X } _ { * * } ^ { * } } = 1 .\tag{262}
$$

If $\mathbf { V } ^ { \dagger } = \{ Y \}$ , there are no colored coordinates: checking the selected parent inputs of Y is suficient. In either case, this fixes one SCM before any competing counterfactual is considered.

Comparing every competing counterfactual. Consider an assigned counterfactual $\mathbf { Z } _ { \ast } ,$ <sub>∗</sub> with mean greater than $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ . We first show that it fixes no input retained in $\mathcal { H } _ { \mathbf { X } _ { * * } }$

Start at $Y$ , whose length-zero path to itself is unchanged. Suppose every directed path from a vertex to Y is unchanged, and that vertex shares $U _ { j }$ with another endpoint. On every accepted outcome, its genuine copy of $U _ { j }$ reaches $Y$ . If a path from the other endpoint to Y contains an input fixed by $\mathbf { Z } _ { \ast \ast } ,$ take the last such input on that path. This input supplies a constant coordinate tuple. Its jth coordinate must propagate along the unchanged remainder of the path and agree with the genuine copy, either at an earlier merge or at $Y$ . A malformed coordinate is rejected. Otherwise, the constant can agree with the fair bit in only one of its two states, so the competing reward mean is at most $\frac { 1 } { 2 }$ , a contradiction.

Every directed path from the other endpoint to Y must therefore be unchanged. The same conclusion holds for each of its descendants: any fixed input on a descendant’s path would also lie on a path from that endpoint. Repeatedly applying these bidirected and descendant steps reaches precisely the minimal UC-territory. Since that territory is $\mathbf { V } ^ { \dagger }$ , all its retained inputs must remain natural under an assignment with mean greater than $\frac { 1 } { 2 }$

Every input selected by $\mathbf { X } _ { * * }$ must also retain its designated value. Omitting its assignment supplies a natural output, which never takes that value and therefore fails its check. Since all retained paths are unchanged, the resulting invalid output reaches Y. If the competing counterfactual keeps every selected input, any further selections outside $\mathbf { V } ^ { \dagger }$ are irrelevant: their paths to $Y$ already pass through fixed inputs. Removing them leaves the same selected inputs with the same unrestricted choices of values. The two counterfactuals are then equivalent. We have proved, in one SCM, that:

$$
\mu _ { \mathbf { Z } _ { \ast \ast } ^ { \ast } } \leq \frac { 1 } { 2 } < 1 = \mu _ { \mathbf { X } _ { \ast \ast } ^ { \ast } }\tag{263}
$$

for every non-equivalent counterfactual. This establishes possible optimality.

(Only if) Suppose $\mathbf { T } = \mathsf { I B } ( \mathcal { H } _ { \mathbf { X } _ { * * } } , Y ) \neq \emptyset$ , and fix the best assignment $\mathbf { X } _ { * * } ^ { * }$ in an arbitrary compatible SCM. The minimal UC-territory is closed under descendants and bidirected connections. Its exogenous inputs are therefore independent of those generating T. Indeed, if an ancestor of a border vertex shared an exogenous input with the territory, bidirected closure would include that ancestor, and descendant closure would then include the border vertex. This contradicts the definition of the border. All parents entering the territory from outside belong to T.

Construct $\mathbf { X } _ { * * } ^ { \prime }$ by retaining the original input assignments and fixing every remaining input from T into the territory. For any value t, supply its component for T to all newly selected inputs from that border variable, while retaining the original optimizing values. The resulting reward has the same mean as the original reward conditional on $\mathbf { T } = \mathbf { t } ;$ the inputs received by the territory are the same, and conditioning does not change its exogenous distribution. The border values here are evaluated in the modified SCM $\mathcal { M } _ { \mathbf { X } _ { * * } ^ { * } }$ . In the summand below, $\mathbf { X } _ { * * } ^ { \prime }$ denotes the assignment corresponding to t. Conditioning gives:

$$
\mu _ { \mathbf { X } _ { \ast \ast } ^ { \ast } } = \sum _ { \mathbf { t } } \mathbb { E } _ { \mathcal { M } _ { \mathbf { X } _ { \ast \ast } ^ { \ast } } } [ Y \mid \mathbf { T } = \mathbf { t } ] P _ { \mathcal { M } _ { \mathbf { X } _ { \ast \ast } ^ { \ast } } } ( \mathbf { T } = \mathbf { t } )\tag{264}
$$

$$
= \sum _ { \mathbf { t } } \mu _ { \mathbf { X } _ { \ast \ast } ^ { \prime } } P _ { \mathcal { M } _ { \mathbf { X } _ { \ast \ast } ^ { \ast } } } ( \mathbf { T } = \mathbf { t } ) \leq \mu _ { \mathbf { X } _ { \ast \ast } ^ { \prime } } \ast .\tag{265}
$$

The last optimization allows independent values at the newly selected inputs, including the common-value assignments appearing in the sum. The best such assignment is at least as good as their average. Thus $\mathbf { X } _ { * } ^ { \prime } ,$ <sub>∗</sub> dominates $\mathbf { X } _ { * * }$ in every compatible SCM.

The two counterfactuals are non-equivalent. Choose $T \in \mathbf { T }$ and a directed path from $T$ into the territory and onward to $Y .$ . Let $T$ be an independent fair bit and let the mechanisms along this path copy it, ignoring their other inputs. Every assignment of ${ \bf X } _ { * * }$ <sub>∗</sub> then has mean $\frac { 1 } { 2 }$ , whereas fixing the new input from $T$ to one gives mean one. Finally, discard selected inputs in $\mathbf { X } _ { * * } ^ { \prime }$ whose children are no longer ancestral to $Y$ . This preserves the reward and leaves the corresponding counterfactual action. The original counterfactual therefore cannot be strictly better than every non-equivalent counterfactual in any compatible SCM, so it is not a ctf-POMIS. □

## C.1.4 Proof of Corollary 10

Proof of Cor. 10. For a rejected counterfactual set, fix the remaining inputs from its interventional border into its minimal UC-territory, retaining the originally selected inputs and their values. The argument in Eq. (265) shows that optimizing the new inputs weakly improves the best mean. The remaining graph is the subgraph induced by that territory and has empty border, so the algorithm returns the resulting counterfactual set. Every omitted action is thus weakly dominated by an action in $\mathcal { A } _ { \mathrm { C T F } } ^ { \star }$ ; the reverse inequality follows from $\mathcal { A } _ { \mathrm { C T F } } ^ { \star } \subseteq \mathcal { A } _ { \mathrm { C T F } }$ □

## C.2 Experimental Structural Causal Models

This section specifies the fixed SCMs used in Sec. 6.6. All observed and exogenous variables are binary. Distinct exogenous variables are mutually independent. A variable $U _ { V }$ is specific to the mechanism of $V _ { : }$ , while a shared variable $U _ { V W }$ appears in the two mechanisms connected by the corresponding bidirected edge. The symbols ∧, ∨, ⊕, and ¬ denote AND, OR, exclusive OR, and negation. Parentheses specify the order of the logical operations.

Task 1. The graph is shown in Fig. 32a. The structural functions are:

$$
\begin{array} { r l r } & { f _ { W } = u _ { W } } & { f _ { X } = z \vee u _ { X } } \\ & { f _ { Z } = u _ { Z } \vee ( w \oplus u _ { Z Y } ) } & { f _ { Y } = \lnot ( u _ { Y } \oplus x \oplus ( \lnot w ) \oplus u _ { Z Y } ) . } \end{array}\tag{266}
$$

Each of $U _ { W } , U _ { X } , U _ { Z } , U _ { Y } , U _ { Z Y }$ follows Bern(0.1). The action supplying $w = 0$ to $f _ { Y }$ and $w = 1$ to $f _ { Z }$ has mean approximately 0.885, whereas the optimal ordinary intervention has mean 0.82.

## Task 2. The graph is shown in Fig. 29b. The structural functions are:

$$
\begin{array} { r l } & { f _ { T } = u _ { T } \qquad f _ { W } = u _ { W } } \\ & { f _ { X } = \left( \neg u _ { X Y } \right) \oplus \left( u _ { X } \vee w \right) } \\ & { f _ { Z } = \left( \neg ( t \oplus ( \neg u _ { Z } ) ) \vee x \right) \oplus u _ { Z Y } } \\ & { f _ { Y } = \neg \big ( u _ { Z Y } \vee ( \neg u _ { X Y } ) \big ) \oplus u _ { Y } \oplus ( x \vee z ) . } \end{array}\tag{267}
$$

The exogenous distributions are:

$$
\begin{array} { r l r } { U _ { T } \sim \mathtt { B e r n ( 0 . 6 5 ) } } & { { } U _ { W } \sim \mathtt { B e r n ( 0 . 6 5 ) } } & { U _ { X } \sim \mathtt { B e r n ( 0 . 7 1 ) } } \\ { U _ { Y } \sim \mathtt { B e r n ( 0 . 7 3 ) } } & { { } U _ { Z } \sim \mathtt { B e r n ( 0 . 4 3 ) } } & { U _ { X Y } \sim \mathtt { B e r n ( 0 . 7 6 ) } } \\ { U _ { Z Y } \sim \mathtt { B e r n ( 0 . 4 3 ) } . } \end{array}\tag{268}
$$

Exact summation gives optimal counterfactual and ordinary means approximately 0.676 and 0.58, respectively.

Task 3(a). The graph is shown in Fig. 35. The structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } \qquad f _ { E } = u _ { E } \qquad f _ { F } = u _ { F } } \\ & { f _ { B } = \neg \left( \left( \neg a \right) \wedge u _ { B } \right) \oplus u _ { B C } \oplus u _ { B Y } \oplus f } \\ & { f _ { C } = \neg \left( \left( \neg u _ { C } \right) \oplus \left( \neg a \right) \right) \wedge \left( d \oplus u _ { B C } \right) } \\ & { f _ { D } = \left( e \oplus u _ { D Y } \right) \wedge u _ { D } } \\ & { f _ { Y } = \neg \left( \left( \neg b \right) \wedge u _ { Y } \right) \oplus u _ { D Y } \oplus u _ { B Y } \oplus c . } \end{array}\tag{269}
$$

The exogenous distributions are:

$$
\begin{array} { l l l } { { U _ { A } \sim \mathrm { B e r n } ( 0 . 2 ) } } & { { U _ { B } \sim \mathrm { B e r n } ( 0 . 3 ) } } & { { U _ { C } \sim \mathrm { B e r n } ( 0 . 4 4 ) } } \\ { { U _ { D } \sim \mathrm { B e r n } ( 0 . 5 2 ) } } & { { U _ { E } \sim \mathrm { B e r n } ( 0 . 5 5 ) } } & { { U _ { F } \sim \mathrm { B e r n } ( 0 . 2 4 ) } } \\ { { U _ { Y } \sim \mathrm { B e r n } ( 0 . 6 7 ) } } & { { U _ { B Y } \sim \mathrm { B e r n } ( 0 . 7 4 ) } } & { { U _ { D Y } \sim \mathrm { B e r n } ( 0 . 3 5 ) } } \\ { { U _ { B C } \sim \mathrm { B e r n } ( 0 . 3 7 ) . } } \end{array}\tag{270}
$$

Here $f _ { Y }$ shares $U _ { B Y }$ with $f _ { B } ,$ , as required by the bidirected edge $B  Y ; U _ { B C }$ is shared only by $f _ { B }$ and $f _ { C }$ . The optimal counterfactual and ordinary means are approximately 0.618 and 0.593, respectively.

Task 3(b). The graph is unchanged, but the structural functions are:

$$
\begin{array} { r l } & { f _ { A } = u _ { A } \qquad f _ { E } = u _ { E } \qquad f _ { F } = u _ { F } } \\ & { f _ { B } = \left( \neg u _ { B Y } \right) \vee \left( \neg ( u _ { B C } \vee u _ { B } ) \wedge f \wedge a \right) } \\ & { f _ { C } = \left( \neg u _ { B C } \right) \vee \left( ( \neg a ) \wedge ( \neg u _ { C } ) \wedge d \right) } \\ & { f _ { D } = \left( u _ { D } \vee ( \neg u _ { D Y } ) \right) \wedge e } \\ & { f _ { Y } = \left( \neg u _ { B Y } \right) \vee \left( \neg ( u _ { D Y } \vee u _ { Y } ) \wedge c \wedge b \right) . } \end{array}\tag{271}
$$

The exogenous distributions are:

$$
\begin{array} { l l l } { { U _ { A } \sim \mathrm { B e r n } ( 0 . 3 4 ) } } & { { U _ { B } \sim \mathrm { B e r n } ( 0 . 5 2 ) } } & { { U _ { C } \sim \mathrm { B e r n } ( 0 . 7 5 ) } } \\ { { U _ { D } \sim \mathrm { B e r n } ( 0 . 4 7 ) } } & { { U _ { E } \sim \mathrm { B e r n } ( 0 . 4 6 ) } } & { { U _ { F } \sim \mathrm { B e r n } ( 0 . 7 6 ) } } \\ { { U _ { Y } \sim \mathrm { B e r n } ( 0 . 6 7 ) } } & { { U _ { B Y } \sim \mathrm { B e r n } ( 0 . 6 3 ) } } & { { U _ { D Y } \sim \mathrm { B e r n } ( 0 . 6 8 ) } } \\ { { U _ { B C } \sim \mathrm { B e r n } ( 0 . 2 6 ) . } } \end{array}\tag{272}
$$

Assigning $b = c = 1$ maximizes $f _ { Y }$ for every exogenous realization, giving:

$$
\mu _ { b = 1 , c = 1 } = 1 - 0 . 6 3 + 0 . 6 3 ( 1 - 0 . 6 8 ) ( 1 - 0 . 6 7 ) \approx 0 . 4 3 7 .\tag{273}
$$

This ordinary intervention attains the counterfactual optimum as well.

## References

R. Ayesha Ali, Thomas S. Richardson, Peter Spirtes, and Jiji Zhang. Towards characterizing Markov equivalence classes for directed acyclic graphs with latent variables. In Proceedings of the Twenty-First Conference on Uncertainty in Artificial Intelligence, pages 10–17, 2005.

Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47:235–256, 2002.

Elias Bareinboim and Judea Pearl. Transportability of causal efects: Completeness results. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 26, pages 698–704, 2012.

Elias Bareinboim and Judea Pearl. Causal inference and the data-fusion problem. Proceedings of the National Academy of Sciences, 113(27):7345–7352, 2016.

Elias Bareinboim, Andrew Forney, and Judea Pearl. Bandits with unobserved confounders: A causal approach. In Advances in Neural Information Processing Systems, volume 28, pages 1342–1350, 2015.

Elias Bareinboim, Juan D Correa, Duligur Ibeling, and Thomas Icard. On Pearl’s hierarchy and the foundations of causal inference. In Hector Gefner, Rina Dechter, and Joseph Y. Halpern, editors, Probabilistic and Causal Inference: The Works of Judea Pearl, pages 507–556. Association for Computing Machinery, 2022.

Elias Bareinboim, Junzhe Zhang, and Sanghack Lee. An introduction to causal reinforcement learning. Technical Report R-65, Causal Artificial Intelligence Lab, Columbia University, Dec 2024. URL https://causalai.net/r65.pdf.

Alexis Bellot, Alan Malek, and Silvia Chiappa. Transportability for bandits with data from diferent environments. In Advances in Neural Information Processing Systems, volume 36, pages 44356–44381, 2023.

Blair Bilodeau, Linbo Wang, and Dan Roy. Adaptively exploiting d-separators with causal bandits. In Advances in Neural Information Processing Systems, volume 35, pages 20381– 20392, 2022.

Olivier Capp´e, Aur´elien Garivier, Odalric-Ambrym Maillard, R´emi Munos, and Gilles Stoltz. Kullback-Leibler upper confidence bounds for optimal sequential allocation. The Annals of Statistics, 41(3):1516–1541, 2013. doi: 10.1214/13-AOS1119.

David Maxwell Chickering. Optimal structure identification with greedy search. Journal of Machine Learning Research, 3(Nov):507–554, 2002.

Davin Choo and Kirankumar Shiragur. Adaptivity complexity for causal graph discovery. In Uncertainty in Artificial Intelligence, pages 391–402. PMLR, 2023.

Juan Correa, Sanghack Lee, and Elias Bareinboim. Nested counterfactual identification from arbitrary surrogate experiments. In Advances in Neural Information Processing Systems, volume 34, 2021.

Juan D. Correa and Elias Bareinboim. Counterfactual graphical models: Constraints and inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 11245–11254. PMLR, 2025. URL https://proceedings.mlr.press/v267/correa25a.html.

Arnoud De Kroon, Joris Mooij, and Danielle Belgrave. Causal bandits without prior knowledge using separating sets. In Conference on Causal Learning and Reasoning, pages 407–427. PMLR, 2022.

Mingwei Deng, Ville Kyrki, and Dominik Baumann. Transfer learning in latent contextual bandits with covariate shift through causal transportability. In Proceedings of the Fourth Conference on Causal Learning and Reasoning, volume 275 of Proceedings of Machine Learning Research, pages 731–756. PMLR, 2025. URL https://proceedings.mlr.press/ v275/deng25a.html.

Shi Feng and Wei Chen. Combinatorial causal bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7550–7558, 2023.

Ronald A. Fisher. The Design of Experiments. Oliver & Boyd, Edinburgh, 1 edition, 1935.

Andrew Forney, Judea Pearl, and Elias Bareinboim. Counterfactual data-fusion for online reinforcement learners. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1156–1164. PMLR, 06–11 Aug 2017. URL https: //proceedings.mlr.press/v70/forney17a.html.

Aur´elien Garivier and Olivier Capp´e. The KL-UCB algorithm for bounded stochastic bandits and beyond. In Proceedings of the 24th Annual Conference on Learning Theory, pages 359–376. JMLR Workshop and Conference Proceedings, 2011.

AmirEmad Ghassami, Saber Salehkaleybar, Negar Kiyavash, and Kun Zhang. Learning causal structures using regression invariance. In Advances in Neural Information Processing Systems, volume 30, 2017.

AmirEmad Ghassami, Negar Kiyavash, Biwei Huang, and Kun Zhang. Multi-domain causal structure learning in linear systems. In Advances in Neural Information Processing Systems, volume 31, 2018.

Xueping Gong, Wei You, and Jiheng Zhang. Eficient transfer learning via causal bounds, 2023. URL https://arxiv.org/abs/2308.03572.

Kristjan Greenewald, Dmitriy Katz, Karthikeyan Shanmugam, Sara Magliacane, Murat Kocaoglu, Enric Boix Adsera, and Guy Bresler. Sample eficient active learning of causal trees. In Advances in Neural Information Processing Systems, volume 32, 2019.

Alain Hauser and Peter B¨uhlmann. Characterization and greedy learning of interventional Markov equivalence classes of directed acyclic graphs. Journal of Machine Learning Research, 13(79):2409–2464, 2012.

James J. Heckman and Richard Robb. Alternative methods for solving the problem of selection bias in evaluating the impact of treatments on outcomes. In Howard Wainer, editor, Drawing Inferences from Self-Selected Samples, pages 63–107. Springer New York, 1986. doi: 10.1007/978-1-4612-4976-4 7.

James J Heckman and Richard Robb Jr. Alternative methods for evaluating the impact of interventions: An overview. Journal of Econometrics, 30(1-2):239–267, 1985.

Christina Heinze-Deml, Jonas Peters, and Nicolai Meinshausen. Invariant causal prediction for nonlinear models. Journal of Causal Inference, 6(2), 2018.

Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Sch¨olkopf. Causal discovery from heterogeneous/nonstationary data. Journal of Machine Learning Research, 21(89):1–53, 2020.

Amin Jaber. Causal Reasoning in Equivalence Classes. PhD thesis, Purdue University, 2022.

Amin Jaber, Jiji Zhang, and Elias Bareinboim. Causal identification under Markov equivalence. In Proceedings of the 34th Conference on Uncertainty in Artificial Intelligence, pages 977–986, 2018. URL https://causalai.net/r35.pdf. Full version with supplemental material.

Amin Jaber, Jiji Zhang, and Elias Bareinboim. Causal identification under Markov equivalence: Completeness results. Technical Report R-42, Columbia CausalAI Laboratory, 2019. URL https://causalai.net/r42.pdf.

Amin Jaber, Adele Ribeiro, Jiji Zhang, and Elias Bareinboim. Causal identification under Markov equivalence: calculus, algorithm, and completeness. In Advances in Neural Information Processing Systems, volume 35, pages 3679–3690, 2022.

Murat Kocaoglu, Karthikeyan Shanmugam, and Elias Bareinboim. Experimental design for learning causal graphs with latent variables. In Advances in Neural Information Processing Systems, volume 30, 2017.

Daniel Kumor, Junzhe Zhang, and Elias Bareinboim. Sequential causal imitation learning with unobserved confounders. In Advances in Neural Information Processing Systems, volume 34, pages 14669–14680, 2021.

Tze Leung Lai and Herbert Robbins. Asymptotically eficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4–22, 1985.

Finnian Lattimore, Tor Lattimore, and Mark D Reid. Causal bandits: Learning good interventions via causal inference. In Advances in Neural Information Processing Systems, volume 29, 2016.

Tor Lattimore and Csaba Szepesv´ari. Bandit algorithms. Cambridge University Press, 2020.

Sanghack Lee and Elias Bareinboim. Structural causal bandits: Where to intervene? In Advances in Neural Information Processing Systems, volume 31, 2018.

Sanghack Lee and Elias Bareinboim. Structural causal bandits with non-manipulable variables. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4164– 4172, 2019.

Sanghack Lee, Juan Correa, and Elias Bareinboim. General transportability–synthesizing observations and experiments from heterogeneous domains. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 10210–10217, 2020.

Yangyi Lu, Amirhossein Meisami, Ambuj Tewari, and William Yan. Regret analysis of bandit problems with causal background knowledge. In Conference on Uncertainty in Artificial Intelligence, pages 141–150. PMLR, 2020.

Marloes H Maathuis and Diego Colombo. A generalized back-door criterion. The Annals of Statistics, 43(3):1060–1088, 2015.

Charles F Manski. Nonparametric bounds on treatment efects. The American Economic Review, 80(2):319–323, 1990.

Scott Mueller and Judea Pearl. Personalized decision making – a conceptual introduction. Journal of Causal Inference, 11(1):20220050, 2023.

Jerzy Neyman. On the application of probability theory to agricultural experiments. essay on principles. section 9. Statistical Science, 5(4):465–472, 1990. doi: 10.1214/ss/1177012031. Translated from the original work published in 1923.

Min Woo Park and Sanghack Lee. On transportability for structural causal bandits. In Emilija Perkovi´c and Daniel Malinsky, editors, Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence, volume 337 of Proceedings of Machine Learning Research, pages 5266–5288. PMLR, 17–21 Aug 2026a. URL https://proceedings.mlr. press/v337/park26a.html.

Min Woo Park and Sanghack Lee. Counterfactual structural causal bandits. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ da80970c6ca6967f2a4de6f2e13c4bae-Abstract-Conference.html.

Min Woo Park, Andy Arditi, Elias Bareinboim, and Sanghack Lee. Structural causal bandits under Markov equivalence. In Advances in Neural Information Processing Systems, volume 38, 2025.

Judea Pearl. Causal diagrams for empirical research. Biometrika, 82(4):669–710, 1995.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, New York, 2000.

Judea Pearl and Elias Bareinboim. Transportability of causal and statistical relations: A formal approach. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 25, pages 247–254, 2011.

Judea Pearl and Dana Mackenzie. The book of why: the new science of cause and efect. Basic Books, 2018.

Judea Pearl, Madelyn Glymour, and Nicholas P Jewell. Causal inference in statistics: a primer. John Wiley & Sons, 2016.

Nathan Peifer-Smadja, Timothy Miles Rawson, Raheelah Ahmad, Albert Buchard, P Georgiou, F-X Lescure, Gabriel Birgand, and Alison Helen Holmes. Machine learning for clinical decision support in infectious diseases: a narrative review of current applications. Clinical Microbiology and Infection, 26(5):584–595, 2020.

Emilija Perkovi´c, Johannes Textor, Markus Kalisch, and Marloes H. Maathuis. Complete graphical characterization and construction of adjustment sets in Markov equivalence classes of ancestral graphs. Journal of Machine Learning Research, 18(220):1–62, 2018.

Ronan Perry, Julius von K¨ugelgen, and Bernhard Sch¨olkopf. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. In Advances in Neural Information Processing Systems, volume 35, pages 10904–10917, 2022.

Jonas Peters, Peter B¨uhlmann, and Nicolai Meinshausen. Causal inference by using invariant prediction: identification and confidence intervals. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78(5):947–1012, 2016.

Arvind Raghavan and Elias Bareinboim. Counterfactual realizability. In The Thirteenth International Conference on Learning Representations, 2025.

Thomas Richardson and Peter Spirtes. Ancestral graph Markov models. The Annals of Statistics, 30(4):962–1030, 2002.

Jonathan Richens and Tom Everitt. Robust agents learn causal world models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=pOoKI3ouv1.

Jonathan Richens, Tom Everitt, and David Abel. General agents need world models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 51659–51687. PMLR, 2025. URL https://proceedings.mlr.press/v267/richens25a.html.

Herbert Robbins. Some aspects of the sequential design of experiments. Bull. Amer. Math. Soc., 58(5):527–535, 1952.

James M. Robins. The analysis of randomized and non-randomized AIDS treatment trials using a new approach to causal inference in longitudinal studies. In L. Sechrest, H. Freeman, and A. Mulley, editors, Health Services Research Methodology: A Focus on AIDS, pages 113–159. US Public Health Service, National Center for Health Services Research, 1989.

Kangrui Ruan, Junzhe Zhang, Xuan Di, and Elias Bareinboim. Causal imitation for Markov decision processes: A partial identification approach. In Advances in Neural Information Processing Systems, volume 37, pages 87592–87620, 2024.

Donald B. Rubin. Estimating causal efects of treatments in randomized and nonrandomized studies. Journal of Educational Psychology, 66:688–701, 1974.

Peter Spirtes and Thomas S. Richardson. A polynomial time algorithm for determining DAG equivalence in the presence of latent variables and selection bias. In David Madigan and Padhraic Smyth, editors, Proceedings of the Sixth International Workshop on Artificial Intelligence and Statistics, volume R1 of Proceedings of Machine Learning Research, pages 489–500. PMLR, 04–07 Jan 1997. URL https://proceedings.mlr.press/r1/ spirtes97b.html. Reissued by PMLR on 30 March 2021.

Peter Spirtes, Clark N Glymour, and Richard Scheines. Causation, prediction, and search. MIT Press, 2000.

Chandler Squires, Sara Magliacane, Kristjan Greenewald, Dmitriy Katz, Murat Kocaoglu, and Karthikeyan Shanmugam. Active structure learning of causal DAGs via directed clique trees. In Advances in Neural Information Processing Systems, volume 33, pages 21500–21511, 2020.

Robert E Tarjan and Mihalis Yannakakis. Simple linear-time algorithms to test chordality of graphs, test acyclicity of hypergraphs, and selectively reduce acyclic hypergraphs. SIAM Journal on Computing, 13(3):566–579, 1984.

William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3-4):285–294, 1933.

Jin Tian and Judea Pearl. On the identification of causal efects. Technical Report R-290-L, Cognitive Systems Laboratory, University of California, Los Angeles, 2003. URL https://ftp.cs.ucla.edu/pub/stat\_ser/R290-L.pdf.

Ioannis Tsamardinos, Laura E Brown, and Constantin F Aliferis. The max-min hill-climbing Bayesian network structure learning algorithm. Machine Learning, 65:31–78, 2006.

Burak Varici, Karthikeyan Shanmugam, Prasanna Sattigeri, and Ali Tajer. Causal bandits for linear structural equation models. Journal of Machine Learning Research, 24(297): 1–59, 2023.

Aparajithan Venkateswaran and Emilija Perkovi´c. Towards complete causal explanation with expert knowledge. arXiv preprint arXiv:2407.07338, 2024.

Thomas Verma and Judea Pearl. Equivalence and synthesis of causal models. In Proceedings of the Sixth Annual Conference on Uncertainty in Artificial Intelligence, pages 220–227, 1990.

Tian-Zuo Wang, Tian Qin, and Zhi-Hua Zhou. Sound and complete causal identification with latent variables given local background knowledge. In Advances in Neural Information Processing Systems, volume 35, pages 10325–10338, 2022.

Tian-Zuo Wang, Tian Qin, and Zhi-Hua Zhou. Sound and complete causal identification with latent variables given local background knowledge. Artificial Intelligence, 322:103964, 2023.

Tian-Zuo Wang, Wen-Bo Du, and Zhi-Hua Zhou. An eficient maximal ancestral graph listing algorithm. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 50353–50378. PMLR, 2024a. URL https://proceedings.mlr.press/v235/wang24o.html.

Tian-Zuo Wang, Lue Tao, and Zhi-Hua Zhou. New rules for causal identification with background knowledge, 2024b. URL https://arxiv.org/abs/2407.15259.

Tian-Zuo Wang, Wen-Bo Du, and Zhi-Hua Zhou. Polynomial-delay MAG listing with novel locally complete orientation rules. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 62804– 62824. PMLR, 2025. URL https://proceedings.mlr.press/v267/wang25y.html.

Marcel Wien¨obst, Max Bannach, and Maciej Li´skiewicz. A new constructive criterion for Markov equivalence of MAGs. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pages 2107–2116. PMLR, 2022. URL https://proceedings.mlr.press/v180/ wienobst22a.html.

Yan Zeng, Shohei Shimizu, Ruichu Cai, Feng Xie, Michio Yamamoto, and Zhifeng Hao. Causal discovery with multi-domain LiNGAM for latent factors. In Causal Analysis Workshop Series, pages 1–4. PMLR, 2021.

Jiji Zhang. Causal Inference and Reasoning in Causally Insuficient Systems. PhD thesis, Carnegie Mellon University, 2006.

Jiji Zhang. A characterization of Markov equivalence classes for directed acyclic graphs with latent variables. In Proceedings of the Twenty-Third Conference on Uncertainty in Artificial Intelligence, pages 450–457, 2007. URL https://arxiv.org/abs/1206.5282.

Jiji Zhang. On the completeness of orientation rules for causal discovery in the presence of latent confounders and selection bias. Artificial Intelligence, 172(16–17):1873–1896, 2008a. ISSN 0004-3702. doi: 10.1016/j.artint.2008.08.001. URL https://www.sciencedirect. com/science/article/pii/S0004370208001008.

Jiji Zhang. Causal reasoning with ancestral graphs. Journal of Machine Learning Research, 9:1437–1474, 2008b.

Junzhe Zhang. Towards Causal Reinforcement Learning. PhD thesis, Columbia University, 2023.

Junzhe Zhang and Elias Bareinboim. Transfer learning in multi-armed bandits: A causal approach. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, pages 1340–1346, 2017. doi: 10.24963/ijcai.2017/186. URL https://www. ijcai.org/proceedings/2017/186.

Junzhe Zhang and Elias Bareinboim. Near-optimal reinforcement learning in dynamic treatment regimes. In Advances in Neural Information Processing Systems, volume 32, 2019.

Junzhe Zhang and Elias Bareinboim. Designing optimal dynamic treatment regimes: A causal reinforcement learning approach. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 11012– 11022. PMLR, 2020. URL https://proceedings.mlr.press/v119/zhang20a.html.

Junzhe Zhang and Elias Bareinboim. Online reinforcement learning for mixed policy scopes. In Advances in Neural Information Processing Systems, volume 35, pages 3191–3202, 2022.

Junzhe Zhang and Elias Bareinboim. Causal eligibility traces for confounding robust ofpolicy evaluation. In Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence, volume 286 of Proceedings of Machine Learning Research, pages 4933–4942. PMLR, 2025. URL https://proceedings.mlr.press/v286/zhang25d.html.

Junzhe Zhang, Daniel Kumor, and Elias Bareinboim. Causal imitation learning with unobserved confounders. In Advances in Neural Information Processing Systems, volume 33, pages 12263–12274, 2020.

Hui Zhao, Zhongguo Zheng, and Baijun Liu. On the Markov equivalence of maximal ancestral graphs. Science in China Series A: Mathematics, 48:548–562, 2005.

Zihan Zhou, Muhammad Qasim Elahi, and Murat Kocaoglu. Characterization and learning of causal graphs from hard interventions. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/ 085713-2603. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/6ff3e124a89678abb0dd5ffc322f0700-Abstract-Conference.html.