# A GAME-THEORETIC FRAMEWORK FOR INCENTIVE-COMPATIBLE AI TRAINING UNDER RENEWABLE-ENERGY CONSTRAINTS

A PREPRINT

<sup>1</sup>Konstantinos <sup>2</sup>Ramin <sup>1</sup>Adamantia <sup>1</sup>George D. <sup>1</sup>Vasillios A. Varsos Khalili Stamou Stamoulis Siris

<sup>1</sup>Athens University of Economics and Business, Greece <sup>2</sup>Independent researcher, Munich Germany

{kvarsos, stamouad, gstamoul, vsiris}@aueb.gr, raminster@gmail.com

## ABSTRACT

As artificial intelligence systems increasingly rely on distributed and collaborative training, the energy footprint of these processes becomes a shared responsibility. Modern AI training often unfolds across heterogeneous compute nodes-ranging from cloud clusters to edge devices-whose energy availability is spatially and temporally variable. At the same time, renewable energy grids experience growing levels of excess generation, creating opportunities to align computational workloads with low-carbon energy supply. In this work, we develop a game-theoretic model of carbon-aware AI training in which autonomous agents strategically choose whether to participate and how intensively to train under limited renewable energy availability. Each agent balances diminishing learning returns, rewards for remaining within green-energy budgets, and penalties for grid consumption. While our framework applies broadly to distributed AI training, we examine Federated Learning as a representative case study due to its decentralized structure and flexible scheduling. We analyze equilibrium existence, efficiency, and adaptive dynamics, and provide simulation evidence that appropriately designed incentives can eliminate grid-based energy usage while preserving model performance. Our findings demonstrate how incentive-compatible training mechanisms can enhance energy efficiency and sharply reduce carbon emissions under renewable-energy constraints.

## 1 Introduction

Artificial intelligence (AI) training has become a significant and growing source of energy consumption. As models increase in scale and complexity, their computational demands translate directly into substantial electricity usage and carbon emissions [24, 32]. Addressing these sustainability challenges requires not only technical innovation but also systemic redesign of digital services. European research efforts, and in particular EXIGENCE [34], exemplify thi broader transition toward user-centric incentive models and energy-aware service design, i.e., see in [35, 36].

Many modern AI systems are trained in distributed environments, where data are generated at the network edge and cannot be centrally aggregated due to privacy, regulatory, or bandwidth constraints [8]. A prominent architectural paradigm in this setting is Federated Learning (FL) [18, 30, 37], in which multiple agents collaboratively train a shared model without transmitting raw data. Instead, agents perform local computation and periodically send model updates to a coordinating service provider.

In this work, we adopt the Federated Learning structure as a canonical model of distributed AI training. The roundbased nature of FL, the separation between local computation and aggregation, and the presence of heterogeneous edge participants make it a useful abstraction for studying energy consumption in distributed AI systems. Importantly, the energy challenges we analyze are not specific to FL per se, but arise more broadly in decentralized and edge-based training architectures.

From an energy perspective, distributed training introduces additional inefficiencies compared to centralized training, including repeated communication rounds and execution on heterogeneous, often less energy-efficient devices [26]. At the same time, renewable energy grids increasingly experience periods of excess generation, during which energy is curtailed because supply exceeds demand or transmission capacity [2]. This stranded renewable energy presents an opportunity: if AI training workloads can be temporally and spatially aligned with renewable availability, their carbon footprint can be significantly reduced [13, 24].

Carbon-aware computing has therefore emerged as a promising direction for adapting computational workloads to variations in renewable energy supply [27]. Distributed training architectures such as Federated Learning are particularly suitable for such adaptation because training proceeds in rounds and often lacks strict real-time constraints [3]. However, most existing work models participating agents as passive resources that execute tasks when scheduled. In practice, participants are autonomous entities with their own objectives, resource constraints, and operational considerations [10, 38]. When renewable availability fluctuates, agents must decide not only whether to participate in a given training round, but also how intensively to train [15].

This observation motivates a strategic perspective on carbon-aware Federated Learning. We consider a setting in which each agent is modeled as rational and self-interested. In every training round, agents decide whether to participate and, conditional on participation, how many local samples to process. These decisions are constrained by the availability of renewable excess energy in their local environment [33]. Agents aim to balance the benefits of contributing to the global model–such as monetary rewards, reputation gains, or improved local performance–against the cost of energy consumption and potential penalties for exceeding green energy availability [29].

The central problem we study is twofold. First, how can participation be incentivized when renewable energy availability is volatile and heterogeneous across agents? Second, how can training intensity be endogenously determined so that agents fully utilize available green energy without exceeding it, thereby avoiding carbon-intensive fallback energy sources? Naively ignoring strategic behavior may lead to inefficient outcomes: some agents may free-ride by under-participating, while others may over-consume resources when incentives are misaligned [19]. Moreover, uncoordinated participation decisions can increase training time, introduce bias, or create instability in the learning process [3].

To address this challenge, we propose modeling carbon-aware Federated Learning as a repeated strategic interaction. Within this framework, agents choose participation and training intensity subject to renewable energy constraints, while a special entity, called AI-service provider, determines penalties, rewards, and aggregation. The objective is to design incentive-compatible mechanisms that (i) encourage sufficient participation, and (ii) align local training effort with locally available green energy. By integrating strategic decision-making with carbon-aware scheduling, this approach aims to achieve sustainable federated training with near-zero operational emissions while preserving efficiency and robustness. Towards this direction, we consider the AI-service provider as a trustworthy mediator that publicly signals information to the agents. First, we examine the dynamics generated by decentralized learning rules–specifically, a fictitious-play heuristic [4]–to determine whether agents’ adaptive behavior converges to stable outcomes, that is the Nash equilibria of the interaction, and how quickly such convergence occurs.

Besides the decentralized heuristic phase, we propose a second approach in which the AI-service provider makes private recommendations to each agent on training-intensity plans, in parallel to the information that publicly signals. If agents condition their strategies on a common public signal, their decisions may become correlated, leading to outcomes that can be characterized as correlated equilibria [1]. Under this signaling-based mechanism, the AI-service provider can incorporate forecasts of renewable availability and cross-agent heterogeneity into its recommendations. Provided that adherence to the recommendation is incentive-compatible, agents optimally follow the signal. This coordinated approach can outperform purely decentralized adaptation by leveraging heterogeneity in renewable-energy availability. As a direct consequence, our incentive-compatible training mechanisms enhance overall energy efficiency leading to zero carbon emissions under renewable-energy constraints.

Outline. Section 2 summarizes the key related literature. Section 3 introduces the necessary preliminaries, including game-theoretic foundations, training-accuracy considerations, and the associated energy-consumption formula. Section 4 presents our framework, followed by two complementary solution concepts: (i) decentralized adaptive dynamics based on fictitious play, and (ii) a signaling-based mechanism-design approach that induces correlated strategies through a trusted coordinating intermediary. Section 5 reports our evaluation results, and Section 6 concludes the paper.

![](images/bc098e9fa9eef49f28832107d4e898529f00ef48bbdcc4ccc7e989e3f266a621.jpg)  
Figure 1: Schematic of a Federated Learning system where, the knowledge of devices $1 , . . . ,$ is shared through averaging the NNs’ weights. Each device trains on its disjoint local data $\mathcal { D } _ { 1 } , \ldots ,$ producing newly trained weights $\overline { { w _ { 1 } ^ { t + 1 } } }$ Every round, the new averaged weights w<sup>t+1</sup> are distributed to all devices. see in [25].

## 2 Related work

Game theory becomes increasingly important to mitigate the issues and challenges arise from the integration of Federated Learning into into energy-constrained environments [10]. Two common approaches are: (i) contract theory, and (ii) auction-based mechanisms. In contract theory models, the AI-service provider offers a list of contracts so that agents self-select according to their private types (e.g., energy budget, data quality), which yields incentive-compatible participation and improves overall learning performance, see in [10]. Reverse auctions let agents bid for participation; the aggregator selects participants based on bids to maximize utility while minimizing energy drain on weaker devices, see in [9].

A second stream of works, utilizes hierarchical leader–follower models (Stackelberg games) are widely used to capture the interaction between a central aggregator (leader), and distributed agents (followers) [7, 38]. In such models the leader (AI-service provider) typically announces rewards, precision targets, or pricing, and the followers respond by choosing local CPU frequencies, numbers of local iterations, or transmission power to trade off energy consumption against the server’s accuracy/latency goals, [12, 29].

A related line of work [31] shows how a central coordinator can disclose truthful signals to guide agents toward more efficient participation and training decisions. Building on this idea, we introduce two policies for the AI-service provider: one that sets only the non-participation penalty parameter γ, and another that additionally provides truthful information to promote both agent participation and training efficiency.

Several works formalize the trade-off between number of Federated Learning rounds, such as convergence speed and accuracy, and total energy consumption [21]; they show existence (and sometimes uniqueness) of equilibria where the global model converges within bounded rounds while device energy budgets are respected [14]. These analy ses are frequently embedded in joint optimization or dynamic-resource allocation frameworks. However, computing such equilibria is often intractable in practice. To address this limitation, we introduce two practical alternatives: (i) decentralized learning through fictitious play and (ii) coordination via correlated strategies. Additionally, while prior work mitigates inefficiencies by selectively excluding devices from each round using cooperative or evolutionary game-theoretic tools, our framework keeps all agents eligible to participate in every round.

In [16], Federated Learning is modeled as an extensive-form game that balances energy consumption, privacy preser vation, and model accuracy. In contrast, our setting assumes that agents make their decisions simultaneously rather than sequentially. Work in [39] analyzes the trade-off between energy consumption and Quality of Service (QoS) in FL and derives an evolutionary stable strategy that identifies the optimal behavior for all players; however, this line of work does not address privacy leakage. Complementarily, [17] studies the interaction between uploaded sensing data and task performance using a Stackelberg game to derive optimal user-privacy strategies, but focuses solely on the privacy–accuracy trade-off and does not incorporate energy consumption during training. Our framework extends these prior efforts by jointly modeling energy usage, training performance, and participation incentives within a unified game-theoretic setting. Unlike previous works, we capture simultaneous decision-making, explicitly integrate renewable-energy constraints, and design incentive mechanisms that promote both efficient training and sustainable energy use.

## 3 Preliminaries

In this section, we review some elementary notions and notation used in subsequent sections. For any natural number $d \in \mathbb { N } , [ d ]$ denotes the set $\{ 1 , 2 , \ldots , d \}$ . Vectors in $\mathbb { R } ^ { d }$ are denoted by bold, $\mathrm { e . g . , } x .$ , with coordinates $x _ { i } , i \in [ d ]$ , while scalar values by light, e.g., x. Further, $\mathbf { 1 } \{ \cdot \}$ denotes the indicator function.

We study the interaction between rational and self-interested agents through the lens of game theory [22]. Given a time horizon T, a N-player finite normal-form game at time $t \in [ T ]$ , denoted as $G ^ { t }$ , has a finite number of agents, $\mathcal { N } ^ { t } \in \mathbb { N }$ , and for each agent $i \in [ N ^ { t } ]$ a finite set $S _ { i } ^ { t }$ of actions. We assume that the number of agents and the number of actions per agent remain constant for all $t \in [ T ]$ , therefore, we simplify the notation to $\mathcal { N }$ and $S _ { i } ,$ , respectively.

The set of pure strategy profiles is the Cartesian product of $S _ { i } { } ^ { \ ' } \mathbf { s } .$ , and is denoted by S. We denote the set agents other than $i \ \mathrm { b y } - i .$ . Finally, for each agent i and $\mathbf { s } \in S ,$ , i receives a real value, called payoff, and denoted as $u _ { i } ( \mathbf { s } , t )$ at t. We refer to the set $\{ u _ { i } ( \mathbf { s } , t ) \} _ { \mathbf { s } \in S }$ as the payoff table of agent i, agglomerating all the payoff tables for all $i \in \mathsf { W }$ , we take the payoff table of the game, denoted as $\{ \mathbf { u } _ { i } ^ { t } \} _ { i \in \mathcal { N } }$ . Succinctly, a N-player finite normal-form game $G ^ { t }$ can be written as the tuple $\langle [ \mathcal { N } ] , S , \{ \bar { \mathbf { u } _ { i } ^ { t } } \} _ { i \in \mathcal { N } } \rangle$ for $t \in [ T ]$

For a finite set A, let $\Delta ( A )$ denote the set of probability distributions over A. For $i \in [ N ]$ , let $\Delta _ { i } : = \Delta ( S _ { i } )$ denote the set of mixed strategies available to agent $i .$ That is, the agent i randomizes over its actions $S _ { i } .$ Formally, a mixed strategy for agent i is a distribution on $S _ { i } ,$ that is, real numbers $x _ { i , j } \geq 0$ for each action $\begin{array} { r } { j \in S _ { i } \mathrm { ~ s . t . ~ } \sum _ { \substack { i \in S _ { i } } } x _ { i , j } = 1 } \end{array}$ . Let $\begin{array} { r } { \Delta : = \prod _ { i \in [ \mathcal { N } ] } \Delta _ { i } } \end{array}$ denote the set ofjoint mixed strategies. A mixed strategy profile at t is $\pmb { \sigma } ^ { t } = ( \pmb { \sigma } _ { 1 } ^ { t } , \pmb { \bar { \sigma } } _ { 2 } ^ { t } , \dots , \pmb { \sigma } _ { \mathcal { N } } ^ { t } ) \in \Delta$ where $\boldsymbol { \sigma } _ { i } ^ { t }$ is the mixed strategy of agent i. Similarly, the pure strategy of agent i at t is denoted as $s _ { i } ^ { t }$ and $s ^ { t }$ is the pure strategy profile at t.

Given a mixed strategy profile $\pmb { \sigma } ^ { t } \in \Delta$ , the expected payoff of agent i is given by

$$
u _ { i } ( \pmb { \sigma } ^ { t } , t ) = \sum _ { \pmb { s } ^ { t } \in S } u _ { i } ( \pmb { s } , t ) \prod _ { i \in [ \mathcal { N } ] } \pmb { \sigma } _ { i } ^ { t } ( \mathscr { s } _ { i } ) .
$$

For a mixed strategy profile $\sigma \in \Delta$ , the best response of agent i is given by the set-valued function $B R _ { i } : \Delta _ { - i } \times $ $[ 0 , T ]  \Delta _ { i }$

$$
B R _ { i } ( \pmb { \sigma } _ { - i } , t ) : = \arg \operatorname* { m a x } _ { \pmb { x } \in \Delta _ { i } } \{ u _ { i } ( ( \pmb { x } , \pmb { \sigma } _ { - i } ) , t ) \} .
$$

For a mixed strategy profile $\pmb { \sigma } ^ { t } \in \Delta$ , the joint best response is given by the set-valued function $B R : \Delta \times [ 0 , T ]  \Delta$

$$
B R ( { \pmb \sigma } , t ) : = B R _ { 1 } ( { \pmb \sigma } _ { - 1 } , t ) \times B R _ { 2 } ( { \pmb \sigma } _ { - 2 } , t ) \times . . . \times B R _ { | N | } ( { \pmb \sigma } _ { - | N } | , t ) .
$$

The fundamental solution concept in game theory is that of Nash equilibrium, stating that no agent has an incentive to deviate to another strategy.

Definition 1 (Nash equilibrium). Let $G ^ { t } = \langle [ M ] , S , \{ \mathbf { u } _ { i } ^ { t } \} _ { i \in \mathcal { N } } \rangle$ be a normal-form game at time t. A Nash equilibrium $o f G ^ { t }$ is a strategy profile $\sigma ^ { * } \in \Delta ( S )$ such that, for each $i \in \mathcal { N }$ and each $\sigma _ { i } ^ { \check { t } } \in \Delta \check { ( } S _ { i } )$

$$
\begin{array} { r } { u _ { i } ( \pmb { \sigma } ^ { * } , t ) \geq u _ { i } ( ( \pmb { \sigma } _ { i } ^ { t } ; \pmb { \sigma } _ { - i } ^ { * } ) , t ) . } \end{array}
$$

Nash’s theorem [20] asserts that every finite normal-form game exhibits a Nash equilibrium.

## 3.1 Fictitious play

In this section we introduce the basic definition of discrete-time fictitious play. Fictitious play refers to a dynamic process where at each stage, agents play a pure best response to the empirical distribution of their opponent’s play, see in [4] and [28].

Let $s _ { i } ^ { t }$ denote the action played by agent i at time t. The empirical frequency of agent i’s play up to time t is defined as

$$
m _ { i } ^ { t } ( s _ { i } ) = \sum _ { \tau = 0 } ^ { t - 1 } { \bf 1 } \{ s _ { i } ^ { \tau } = s _ { i } \} ,\tag{1}
$$

Thus, $\boldsymbol { m } _ { i } ^ { t }$ is an $| S _ { i } |$ -dimensional vector whose components count the number of times agent i has played each action. The empirical distribution of agent $i \ ' s$ play up to time t is given by

$$
\mu _ { i } ^ { t } ( s _ { j } ) = \frac { m _ { i } ^ { t } ( s _ { j } ) } { t } .\tag{2}
$$

Finally, let $\mu ^ { t }$ denote the joint distribution on $\Pi _ { i \in \mathcal { N } } S _ { i }$ obtained as the independent product of the individual empirical distributions $\mu _ { i } ^ { t }$

In discrete-time fictitious play, each agent selects an arbitrary action at time $t = 0$ . For every $t > 0 ,$ agent i plays a pure best response to the product of the marginal empirical distributions of its opponents. Formally, for all $t > 0$ and for each agent i, $s _ { i } ^ { t } \in \mathrm { B } \dot { \mathrm { R } _ { i } } ( \mu _ { - i } ^ { t } , t )$ , where $\mu _ { - i } ^ { t }$ denotes the product of the empirical distributions of all agents other than i, and $\operatorname { B R } _ { i } ( \cdot )$ denotes the set of pure best responses of agent i.

```latex
Algorithm 1 Fictitious play algorithm
Require: A finite normal-form game $\langle [ \mathcal { N } ] , S , \{ \mathbf { u } _ { i } ^ { t } \} _ { i \in [ \mathcal { N } ] } \rangle$ , an initial mixed strategy profile σ, a finite time $\tau .$
1: ${ \pmb { \mu } } ^ { 0 }  { \pmb { \sigma } } .$
2: while $1 \leq \tau \leq \tau$ do
3: Compute best responses, $s _ { i } ^ { t } \in \mathrm { B R } _ { i } \left( \mu _ { - i } ^ { \tau } , t \right)$
4: Compute empirical frequencies for each action, $\begin{array} { r } { m _ { i } ^ { t } ( s _ { i } ) = \sum _ { \ell = 0 } ^ { \tau - 1 } \mathbf { 1 } \{ s _ { i } ^ { \ell } = s _ { i } \} } \end{array}$
5: Update empirical distributions, $\begin{array} { r } { \mu _ { i } ^ { t } ( s _ { i } ) = \frac { m _ { i } ^ { t } ( s _ { i } ) } { \tau } } \end{array}$
6: end while
7: $\sigma ^ { t + 1 } \gets \mu ^ { \tau }$
8: return A mixed strategy profile $\pmb { \sigma } ^ { t + 1 }$
```

Implementing the fictitious play algorithm we consider two assumptions, first, we assume that agents move simultaneously, and second, the agents put equal weight on every play in the past.

## 3.2 Correlated equilibria

Now we twist the previous setting introducing a correlation between the strategies of the agents. To that end, a trusted mediator signals information and makes recommendations but not enforce behavior, acting as a correlation device. The distribution $\chi$ from which the correlation device samples recommendations is public knowledge, and in addition the agents only get private recommendations from the correlation device. This lead to correlated equilibrium solution concept, first introduced in [1]. Formally, a correlated equilibrium is defined as follows

Definition 2 (Correlated equilibrium). Given a finite normal-form game $G ^ { t } = \langle [ \mathcal { N } ] , S , \{ \mathbf { u } _ { i } ^ { t } \} _ { i \in [ \mathcal { N } ] } \rangle$ at time $t ,$ correlated equilibrium is a distribution $\chi$ over $S$ such that for all agents $i \in [ N ]$ and actions a, $a ^ { \prime } \in S _ { i }$

$$
\begin{array} { r } { \mathbb { E } _ { s \sim \chi } [ u _ { i } ( ( a , \mathbf { s } _ { - i } ) , t ) \mid s _ { i } = a ] \ge \mathbb { E } _ { s \sim \chi } [ u _ { i } ( ( a ^ { \prime } , \mathbf { s } _ { - i } ) , t ) \mid s _ { i } = a ] . } \end{array}
$$

Nash equilibria are also correlated equilibria, therefore every finite normal-form game has a correlated equilibrium. From the perspective of correlated equilibria, a Nash equilibrium is the special case in which each agent’s actions are drawn from an independent distribution, and hence conditioning on s provides no additional information about $\mathbf { s } _ { - i } .$

Importantly, we treat $\langle [ \mathcal { N } ] , S , \{ \mathbf { u } _ { i } ^ { t } \} _ { i \in \mathcal { N } } \rangle$ as a sequence of independent stage games. Our analysis focuses on the equilibrium outcome in each round t separately rather than on a dynamic solution across the horizon T. Consequently, we do not consider equilibrium refinements for sequential games.

## 3.3 Federated Learning & Training Component

Consider that a set of N agents interconnected with a power domain, each characterized by energy-related and learning-related attributes. In what follows, the proposed framework in Section 4 is agnostic to the underlying machine-learning methodology and does not depend on a specific optimizer, model architecture, or training algorithm. For analytical simplicity, we assume that the outcome of a local training step depends on the amount and utility of the local training data, as well as on the current state of the global model that is being optimized. Other factors, including the employed learning algorithm, optimizer configuration, learning rate, memory constraints, and computational resources, can be incorporated orthogonally via additional parameters and/or properly defined functions. We consider a finite training horizon T, divided into discrete time slots indexed by $t \in \{ 0 , \ldots , T \}$

Furthermore, we assume that the AI-service provider specifies a target global accuracy threshold, denoted as $\mathcal { \alpha } .$ that serves as the stopping criterion for the training process. Unlike a centralized setting, where the provider can explicitly terminate training once the desired accuracy is achieved, the proposed decenatralized framework does not allow the provider to enforce a strict training decision. Instead, the accuracy threshold represents a target that is attained through the collective actions and decisions of the participating users. Once the global model reaches this target, no further training is required, thereby avoiding unnecessary computation and energy consumption while maintaining the desired predictive performance. Limiting training beyond the required accuracy can also help mitigate overfitting and improve the model’s generalization capability. In privacy-preserving learning settings, it may additionally contribute to balancing the trade-off between model utility and privacy preservation, as discussed in [5].

We consider a setting where a central AI-service provider assigns a learning model to the $\mathcal { N }$ agents and sets the training accuracy threshold $\boldsymbol { \mathscr { a } } . \mathrm { \mathbf { A } } \mathrm { t }$ the beginning of each round $t ,$ the AI-service provider broadcasts model $\mathbf { w } ^ { t }$ , characterized by predictive accuracy $\bar { \alpha } ^ { t } \in ( 0 , 1 )$ , to all agents simultaneously. Initially, the training process starts from a baseline model with predictive accuracy $\bar { \alpha } ^ { 0 }$ . Upon receiving the global model, each agent i performs local training using its private data, and then returns the updated model to the AI-service provider in a synchronized manner. The AIservice provider then aggregates the received updates to construct a refined global model. This iterative procedure continues until the model reaches a predetermined threshold, e.g., a prescribed predictive accuracy or until exhausting the training horizon.

Each agent $i \in \mathcal N$ possesses a private bundle of data $\mathcal { D } _ { i }$ . During round t it chooses a subset of samples $d _ { i } ^ { t } \in \mathcal { D } _ { i }$ for local training. The quality and usefulness of the selected training data are represented through a time-dependent quality function $\mathbf { \bar { \theta } } _ { i } : \mathcal { D } _ { i } \times [ 0 , \mathbf { \dot { T } } ]  [ 0 , 1 ]$ . Intuitively, $\theta _ { i } ( d _ { i } ^ { t } , t )$ captures attributes such as data quality, informativeness, diversity, and relevance of the selected samples to the current model state at time t.

Starting from the global predictive accuracy $\bar { \alpha } ^ { t }$ , each agent performs local training and improves the received model according to its selected training effort and data quality. Specifically, the local predictive accuracy achieved by agent i after local training at round t+1 is modeled as ${ \alpha } _ { i } ( d _ { i } ^ { t } , \theta _ { i } ( d _ { i } ^ { t } , t + 1 ) , t + 1 ) = \bar { \alpha } ^ { t } + { \xi } _ { i } ( d _ { i } ^ { t + 1 } , \theta _ { i } ( d _ { i } ^ { t + 1 } , t + 1 ) , t + 1 )$ , where $\xi _ { i } ( \cdot ) \in [ 0 , 1 ]$ represents the training gain obtained by agent i at round $t + 1$ . The gain function $\xi ( \cdot )$ depends on the amount of local samples used, their quality, and potentially other training-related factors, such as the number of local iterations. Since $\xi _ { i } ( \bar { \cdot } ) \geq 0$ , the local accuracy is non-decreasing over time and satisfies $\alpha _ { i } ( \cdot , \theta _ { i } ( \cdot , t + 1 ) , t + 1 ) \geq \bar { \alpha } ^ { t }$ which is the average accuracy across agents at round t.

After local training, agent i submits its updated model $\mathbf { w } _ { i } ^ { t + 1 }$ to the AI-service provider. The provider aggregates the received local models to construct the global model $\mathbf { w } ^ { t + 1 }$ for the next round. Since agents may contribute different amounts of training data, the aggregation is weighted according to the number of samples used during local training.

We define the nominal local accuracy achieved by agent i after participating in training at round t as $\alpha _ { i } ^ { ( + ) } ( t ) =$ $\alpha _ { i } ( d _ { i } ^ { t } , \theta _ { i } ( d _ { i } ^ { t } , t ) , t )$ . Then the aggregated accuracy of the FL process is defined as

$$
\bar { \alpha } ^ { t } = \sum _ { i \in \mathcal { N } } \left( \frac { d _ { i } ^ { t } } { \sum _ { j \in \mathcal { N } } d _ { j } ^ { t } } \right) \alpha _ { i } ^ { ( + ) } ( t ) ,\tag{3}
$$

where $\frac { d _ { i } ^ { t } } { \sum _ { j \in \mathcal { N } } d _ { j } ^ { t } }$ denotes the normalized contribution of user i, proportional to the amount of data used in its local training. The FL training is considered complete once $\alpha _ { i } ^ { ( + ) } > \alpha$ for any $i \in \mathcal N$ at some finite iteration $t \leq T$ . If $\begin{array} { r } { \operatorname* { l i m } _ { t \to \infty } \bar { \alpha } ^ { t } \to \alpha } \end{array}$ , then we say that the FL process converges asymptotically to the target accuracy. Importantly, this criterion can be integrated into the incentivization scheme, see Section 4, by allowing the AI-service provider to reduce or discontinue rewards once the desired accuracy level has been achieved.

Note that in a decentralized FL environment, agents may not participate in every training round. When agent i does not train the model for several consecutive rounds, the local model performance may deteriorate due to changes in the underlying data distribution.

Concluding, the predictive accuracy evolves recursively across communication rounds: agents receive the global model from the previous round, improve it locally using private data, and the provider forms a new global model through sample-weighted aggregation. This iterative process continues until the training horizon is reached or a target performance threshold is satisfied. A schematic illustration of this process is shown in Figure 1.

## 3.4 Energy Consumption Component

In this section, we model the energy consumption of each agent. Specifically, we define the energy consumption function as a mapping $\varepsilon _ { i } : \mathcal { D } _ { i } \times [ 0 , T ] \to \mathbb { R } _ { \geq 0 }$ , where $\varepsilon _ { i } ( d _ { i } ^ { t } , t )$ denotes the energy consumed by agent i when processing $d _ { i } ^ { t } \in \bar { \mathcal { D } } _ { i }$ bundle of data samples at time t.

The total energy consumption consists of two components: the energy required for local computation, denoted by $\varepsilon _ { i } ^ { g } ( \cdot )$ , and the energy required for communication with the AI-service provider, denoted by $\varepsilon _ { i } ^ { t r } ( \cdot )$ . We assume that the communication between the agents and the AI-service provider is carried out over wired or fiber-optic links.

Table 1: Summary of notation.
<table><tr><td>Symbol</td><td>Name</td><td>Symbol</td><td>Name</td></tr><tr><td> $\overline { { T } }$ </td><td>Time horizon</td><td> $\overline { { \mathbf { w } _ { i } ^ { t } } }$ </td><td>Local model</td></tr><tr><td> $t \in [ T ]$ </td><td>Time index</td><td> $\mathcal { D } _ { i }$ </td><td>Local dataset</td></tr><tr><td> $G ^ { t }$ </td><td>Game at time t</td><td> $d _ { i } ^ { t }$ </td><td>Local training samples</td></tr><tr><td> $\mathcal { N }$ </td><td>Number of agents</td><td> $\hat { \alpha }$ </td><td>Global accuracy threshold</td></tr><tr><td> $S _ { i }$ </td><td>Action set</td><td> $\alpha _ { i } ( \cdot )$ </td><td>Accuracy function</td></tr><tr><td> $S$ </td><td>Strategy profile space</td><td> $\bar { \alpha } ^ { t } ( \cdot )$ </td><td>Global accuracy function</td></tr><tr><td> $s _ { i } ^ { t }$ </td><td>Pure strategy</td><td> $\alpha _ { i } ^ { ( + ) } ( \cdot )$ </td><td>Nominal local accuracy function</td></tr><tr><td> $\mathrm { \mathbf { s } } ^ { t }$ </td><td>Pure strategy profile</td><td> $\alpha _ { i } ^ { ( - ) } ( \cdot )$ </td><td>Effective local accuracy function</td></tr><tr><td> $\pmb { \sigma } _ { i } ^ { t }$ </td><td>Mixed strategy</td><td> $\widetilde { \alpha } _ { i } ( \cdot )$ </td><td>Previous local accuracy function</td></tr><tr><td> $\sigma ^ { t }$ </td><td>Mixed strategy profile</td><td> $\alpha _ { i } ^ { \mathrm { { m i n } } }$ </td><td>Minimum accuracy</td></tr><tr><td> $\mathbf { u } _ { i } ^ { t }$ </td><td>Payoff table</td><td> $\theta _ { i } ( \cdot )$ </td><td>Data quality function</td></tr><tr><td> $u _ { i } ( \cdot )$ </td><td>Payoff function</td><td> $\xi _ { i } ( \cdot )$ </td><td>Training gain function</td></tr><tr><td> $u _ { i } ( \cdot )$ </td><td>Expected payoff function</td><td> $\beta$ </td><td>Diminishing returns factor</td></tr><tr><td> $g { \mathcal W } ( \cdot )$ </td><td>Green well-being function</td><td> $H _ { i } ( \cdot )$ </td><td>Accuracy drift function</td></tr><tr><td> $\sigma ^ { * }$ </td><td>Nash equilibrium</td><td> $\mathcal { d } _ { i }$ </td><td>Drift rate</td></tr><tr><td> $x$ </td><td>Correlated equilibrium</td><td> $\ell _ { i } ( \cdot )$ </td><td>Idle time function</td></tr><tr><td> $\overline { { { \mathcal G } ^ { t } } }$ </td><td>Green energy</td><td> $h$ </td><td>Recovery factor</td></tr><tr><td> $\mathcal { U } ^ { t }$ </td><td>Grid energy</td><td> $\gamma$ </td><td>Non-participation penalty</td></tr><tr><td> $\varepsilon _ { i } ( \cdot )$ </td><td>Energy consumption function</td><td>η</td><td>Relative intensity parameter</td></tr><tr><td> $\varepsilon ^ { g } ( \cdot )$ </td><td>Computation energy function</td><td>π</td><td>Accuracy indicator</td></tr><tr><td> $\varepsilon ^ { t r } ( \cdot )$ </td><td>Communication energy function</td><td>C</td><td>Accuracy intensity parameter</td></tr><tr><td> $C _ { i }$ </td><td>CPU cycles</td><td> $\psi _ { i }$ </td><td>Strategy-sample mapping</td></tr><tr><td> $\kappa$ </td><td>Switched capacitance</td><td> $\operatorname { \dot { } } P _ { i } ( \cdot )$ </td><td>Profit function</td></tr><tr><td> $I _ { i , t }$ </td><td>Local iterations</td><td> $C _ { i } ( \cdot )$ </td><td>Cost function</td></tr><tr><td> $f _ { i }$ </td><td>Computational capacity</td><td> $R _ { i } ( \cdot )$ </td><td>Reward function</td></tr><tr><td> $\overline { { { \bf w } ^ { t } } }$ </td><td>Global model</td><td> $g _ { i } ( \cdot )$ </td><td>Allocated energy function</td></tr></table>

Since the number of exchanged bits in each communication round is fixed, the corresponding communication energy consumption can be reasonably approximated as constant, say $c ^ { t r }$ . Therefore, for simplicity, we assume that the communication energy between each agent and the AI-service provider satisfies $\varepsilon _ { i } ^ { t r } ( \cdot ) = \stackrel { . } { c } ^ { t r }$ for all i.

Regarding the energy availability, at round t, the system operates under limited green energy availability, denoted by $\mathcal { G } ^ { t }$ , and by grid energy availability, denoted by U<sup>t</sup>. We assume that the AI-service provider first allocates $\setminus { \mathcal { G } } ^ { t }$ according to some predefined and exogenous protocol among the users in ${ \mathcal { N } } .$ . If $\mathcal { G } ^ { t }$ is not sufficient to cover the energy required for all tasks, the system can consume grid energy $\mathcal { U } ^ { t }$ . In cases where both $\mathcal { G } ^ { t }$ and $\mathcal { U } ^ { t }$ are not sufficient to cover all tasks, then we consider these tasks impractical, and we drop them.

## 4 Carbon- and Energy-Aware Training Model

We model a synchronous carbon-aware Federated Learning situation as a finite normal-form game played at a finite horizon T, in which users strategically choose their local training intensity under limited renewable energy availability. In particular, each user chooses the subset of local data to employ for training, thereby determining the corresponding learning accuracy and energy expenditure. Clearly, the number of local iterations affects both the learning accuracy and energy expenditure. Nevertheless, we consider it as constant throughout the process. The model is designed to capture three fundamental aspects: (i) competition over shared green energy, (ii) aligning with environmental considerations, and (iii) learning returns.

Before formalizing the strategic interaction, we discuss the individual knowledge that each participant possesses. Every participant at each round of the interaction knows the model $\mathbf { w } ^ { t }$ and the bundle of samples that each agent has. The AI-service provider knows the total green energy availability $\mathcal { G } ^ { t }$ , at each round. Importantly, at each round t, the AI-service provider knows the available bundle of samples for each agent, but does not know their quality $\theta ( \mathbf { d } ^ { t } , t )$ Since the training process is federated, agents have access only to their own local data. In particular, each agent i knows privately the quality of its dataset, that is $\theta _ { i } ( d _ { i } ^ { t } , t )$ , as well as the amount of green energy it can utilize, formally defined in eq. (5). Additional assumptions regarding the information structure are introduced below.

For the strategic interaction, consider the case where $| \mathcal { N } |$ | agents participate in a Federated Learning environment. Each agent $i \in [ N ]$ selects the number of samples $d _ { i } ^ { t } ,$ considering it quality $\theta _ { i } ( d _ { i } ^ { t } , t )$ , it will use during training at round $t ,$ therefore $\bar { d } _ { i } ^ { t } \in S _ { i }$ . Participation is endogenous such that agent i participates in the training iff $s _ { i } ^ { t } > 0$ . Moreover, we assume that when agent i does not participate, that is $s _ { i } ^ { t } = 0$ , then the system imposes to it a penalty $\gamma > 0$ . Therefore, the set of actions for agent i is $S _ { i } = \{ 0 \} \cup S _ { i }$ and $s$ is the cartesian product of $S _ { i }$ for all $i \in \mathcal N$ . Observe that, the set $s _ { i }$ coincides with the set $\mathcal { D } _ { i }$ and there is a bijection correspondence between each bundle of samples $d _ { i } \in \mathcal { D } _ { i }$ and each action $s _ { i } \in S _ { i }$ . We formalize the latter relationship with the function $\psi _ { i } : S _ { i }  D _ { i }$ for each i. In case of a mixed strategy profile σ, then $\psi _ { i } : \Delta ( S _ { i } )  \Delta ( \mathcal { D } _ { i } )$ , with $\begin{array} { r } { \psi _ { i } ( \mathbf { \bar { \sigma } } ) : = \sum _ { j \in S _ { i } } \sigma _ { i , j } \cdot \psi _ { i } ( s _ { j } ) } \end{array}$

Naturally, agent i receives a profit $P _ { i }$ and a cost $C _ { i }$ according to the actions it takes during the training process. In particular, the payoff of agent i in a single turn of the procedure is given by the following formula,

$$
u _ { i } ( { \bf s } ^ { t } , t ) = { \bf 1 } \{ s _ { i } ^ { t } > 0 \} \cdot \left( P _ { i } ( { \bf s } ^ { t } , t ) - C _ { i } ( { \bf s } ^ { t } , t ) \right) - { \bf 1 } \{ s _ { i } ^ { t } = 0 \} \cdot \pi \cdot \gamma .\tag{4}
$$

Non-participation yields $- \gamma$ payoff, while active agents balance learning gains and renewable rewards against energy costs and grid penalties. The parameter $\pi$ captures the relationship between the global accuracy threshold $\hat { \alpha }$ and model’s accuracy at the previous round t. If $\bar { \alpha } ^ { t } \leq \alpha$ , then $\pi _ { i } = 1$ , otherwise, $\pi = 0 .$ . Therefore, once the global accuracy threshold has been reached the AI-service provider has no considerations regarding the non-participation of the agents. In what follows, we construct the profit and the cost functions for each agent.

Initially, we describe how training intensity translates into energy demand and how renewable energy is allocated. Processing $s _ { i } ^ { t }$ samples requires energy $\varepsilon _ { i } ( \dot { \psi } _ { i } ( s _ { i } ^ { t } ) , t )$ , eq. (8), where $\psi _ { i } ( s _ { i } ^ { t } ) ~ = ~ d _ { i } ^ { t }$ . Let $\mathcal { G } ^ { t }$ be the total renewable (green) energy available during round t. Then if agent i is an active participant, $. . . , s _ { i } ^ { t } > 0 .$ , can consume up to $g _ { i } ( s _ { i } ^ { t } , t )$ quantity of $\mathcal { G } ^ { t }$ . The total energy demand is $\begin{array} { r } { \mathbf { \bar { \mathcal { E } } } ( \mathbf { s } ^ { t } , t ) : = \sum _ { i \in [ N ] } \bar { \varepsilon } _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) } \end{array}$ , and the green energy allocated as follows,

$$
\begin{array} { r } { g _ { i } ( s _ { i } ^ { t } , t ) = \left\{ \begin{array} { l l } { \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) } & { \mathrm { , ~ i f } \mathcal { E } ( \mathbf { s } ^ { t } , t ) \leq \mathcal { G } ^ { t } , } \\ { \mathcal { G } ^ { t } \cdot \frac { \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) } { \mathcal { E } ( \mathbf { s } ^ { t } , t ) } } & { \mathrm { , ~ o t h e r w i s e } } \end{array} \right. , } \end{array}\tag{5}
$$

with $g _ { i } ( 0 , t ) = 0$ for each $t .$ In the case where $\varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) > g _ { i } ( s _ { i } ^ { t } , t )$ , we assume that the remaining energy, $\varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) - g _ { i } ( s _ { i } ^ { t } , t )$ , required for the completeness of agent’s i training is drawn from the grid, which has arbitrarily large availability. Importantly, we assume that each agent i at first consumes green energy, if any, and then it uses grid energy, if necessary. Notably, allocation rule (5) induces interdependence across users: increasing one’s training intensity reduces others’ renewable shares.

We next introduce the reward component, which is designed to align incentives with renewable usage. We define the renewable-alignment term at time t for the agent i as follows,

$$
R _ { i } ( \mathbf { s } ^ { t } , t ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { , ~ i f ~ } s _ { i } ^ { t } = 0 \mathrm { ~ o r ~ } \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) > g _ { i } ( s _ { i } ^ { t } , t ) , } \\ { \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) } & { \mathrm { , ~ i f ~ } s _ { i } ^ { t } > 0 \mathrm { ~ a n d ~ } \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) \leq g _ { i } ( s _ { i } ^ { t } , t ) . } \end{array} \right.
$$

This term rewards users who remain within renewable limits and penalizes users who exceed them. Therefore, the profit $P _ { i } ( \cdot )$ of agent i at time t is defined as

$$
P _ { i } ( { \bf s } ^ { t } , t ) = \pi _ { i } \cdot R _ { i } ( { \bf s } ^ { t } , t ) \cdot \left( 1 + \eta \cdot \frac { \psi _ { i } ( s _ { i } ^ { t } ) } { \operatorname* { m a x } _ { j \in \mathcal { N } } \{ \psi _ { j } ( s _ { j } ^ { t } ) \} } \right) + c \cdot \alpha _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , \theta _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) , t ) ^ { 1 } ,
$$

where $c , \eta$ are positive constants. Here, the parameter $\pi _ { i }$ plays a role complementary to that in eq. (4). Specifically, once the global accuracy threshold has been reached, the AI-service provider stops offering rewards to the users. The multiplicative factor $\begin{array} { r } { \eta \cdot \frac { \psi _ { i } ( s _ { i } ^ { t } ) } { \operatorname* { m a x } _ { j \in \mathcal { N } } \{ \psi _ { j } ( s _ { i } ^ { t } ) \} } } \end{array}$ rewards relative intensity and captures competitive effort incentives, discouraging free-riding. The parameter η controls the strength of competitive effort incentives. If $\eta \ : = \ : 0 ,$ , then we have a purely compliance reward. As η is increased then the multiplicative factor supports a more competitive interaction. The additive accuracy term, eq. (7), ensures that accuracy remains intrinsically valuable.

Although the profit structure contains multiple terms, each component serves a distinct conceptual role. The renewable-alignment reward internalizes the environmental externality of shared green energy, the relative effort factor captures competition among participants, and the accuracy term preserves the intrinsic value of learning progress.

On the other hand, the cost function, $C _ { i } ( \cdot )$ captures both renewable usage costs and strong convex penalties for grid consumption in each round of the process. The cost incurred to agent i w.r.t. to the joint decision s is,

$$
C _ { i } ( { \bf s } ^ { t } , t ) = c _ { 1 } \cdot \operatorname* { m a x } \{ 0 , g _ { i } ( s _ { i } ^ { t } , t ) - \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) \} + c _ { 2 } \cdot \left( \operatorname* { m a x } \{ 0 , \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t ) - g _ { i } ( s _ { i } ^ { t } , t ) \} \right) ^ { 2 } ,
$$

where $c _ { 1 }$ and $c _ { 2 }$ are positive constants. The linear term represents baseline renewable energy cost. The quadratic sharing term penalizes grid energy consumption, see in [11]. The convex grid penalty makes carbon-intensive energy strongly undesirable, hence, we assume that $c _ { 1 } \ll c _ { 2 }$ , ensuring that exceeding renewable limits is substantially more expensive. Observe that, the decisions of the −i agents implicitly influence functions $P _ { i } ( \mathbf { s } ^ { t } , t )$ and $C _ { i } ( \mathbf { s } ^ { t } , t )$ via the green energy split in eq. (5).

The above components define a finite normal-form game $G ^ { t } = \langle [ \mathcal { N } ] , \mathcal { S } , \{ \mathbf { u } _ { i } ^ { t } \} _ { i \in \mathcal { N } } \rangle$ , which captures the tension between competitive training effort and collective renewable constraints at time t. Agents compete for limited green energy while facing diminishing accuracy returns and convex penalties for grid energy usage.

![](images/dbb7e2cb918fa7e39c339361f47f098ba11a6555f7948d39caba7561926b12e9.jpg)  
Figure 2: Schematic representation of FL training with the heuristic phase.

Heuristic phase. Although Nash equilibrium is a universal and appealing solution concept, it is intractable for large-scale problems, see in [6]. Further, Nash equilibrium requires that the agents have full information about the situation, which violates Federated Learning. To mitigate these issues we introduce the following mechanics. Since the training process is federated, agents have access only to their own local data; consequently, for each t, game $G ^ { t }$ is one of incomplete information. For that, we consider that agents rely on an introspective decision-making process, referred to as the heuristic phase. In particular, at time t, the AI-service provider assigns the model $\mathbf { w } ^ { t }$ to each agent and publicly broadcasts the joint strategy profile $\sigma ^ { t }$ . Agent i then independently runs the fictitious play algorithm, Algorithm 1, over a finite horizon $\tau .$ In other words, at each round agents receive public signal (e.g., observed past profile) and run $\tau$ steps of Algorithm 1 over the induced one-shot game with estimated opponent mixed strategies $\pmb { \mu } _ { - i }$ to output ${ \pmb { \sigma } } _ { i } ^ { t + 1 }$ . This procedure constitutes the heuristic phase of the training process. The outcome of the heuristic phase determines the strategy that agent i applies in the subsequent training round. Finally, agent i returns the updated model $\mathbf { w } ^ { t + 1 }$ along with its strategy ${ \boldsymbol { \sigma } } _ { i } ^ { t } ,$ see Figure 2.

![](images/6a042bc1a83f92c229554656a910a2cca72878879464c91e10f3cc1acdbb99aa.jpg)  
Figure 3: Schematic representation of FL training with the correlated devise.

Correlation device. In this approach, given the game $G ^ { t }$ , we assume that the AI-service provider besides assigning the model $\mathbf { w } ^ { t }$ to each agent, it publicly signals a distribution $\chi$ over the strategy space $s ,$ and privately recommends actions $s _ { i } \in S _ { i }$ to each agent i, see Figure 3. Since the quality of the dataset is private information unknown to the mediator, it suggests distribution χ using partial information. Our objective is to compute a distribution $\chi \in \Delta ( S )$ that optimizes a given system-wide green well-being. For that we introduce the function $\begin{array} { r } { \mathcal { G W } ( \mathbf { s } ^ { t } , t ) = \sum _ { i \in \mathcal { N } } ( g _ { i } ( \hat { s } _ { i } ^ { t } , t ) - } \end{array}$ $\varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t )$ , capturing the green energy consumption in the system at round t. Clearly, when $\mathcal { G W } ( \mathbf { s } ^ { t } , t ) > 0$ the aggregated energy consumption, w.r.t. $\mathrm { \mathbf { s } } ^ { t }$ , does not overwhelm the amount of aggregated renewable energy at $t .$ Contrarily, when $\dot { \boldsymbol { \mathcal { G } } } \boldsymbol { \mathcal { W } } ( \mathbf { s } ^ { t } , t ) \mathbf { \bar { \Psi } } < 0$ , then agents use grid energy $\mathcal { U } ^ { t }$ at some round of the process. So, the AI-service provider wants to find the distribution $\chi$ that maximizes GW. For that, for each $t \in [ T ]$ , we device the following linear program,

$$
\begin{array} { r l r } { \underset { x \sim \Delta ( S ) } { \operatorname* { m a x } } } & { \mathcal { G W } ( \mathbf { s } ^ { t } , t ) } \\ { \mathrm { s . t . } } & { \sum _ { s ^ { t } = ^ { t } , i ^ { \prime } \in S _ { - i } } ( u _ { i } \big ( ( s _ { i } ^ { t } , \mathbf { s } _ { - i } ^ { t } ) , t \big ) - u _ { i } \big ( ( s _ { i } ^ { \prime } , \mathbf { s } _ { - i } ^ { t } ) , t \big ) \cdot \chi ( \mathbf { s } _ { - i } ^ { t } \mid s _ { i } ) \ge 0 , \quad \forall i \in [ N ] \quad , s _ { i } ^ { t } , s _ { i } ^ { \prime } \in S _ { i } , } \\ & { \chi ( \mathbf { s } ^ { t } ) \ge 0 } & { , \forall \mathbf { s } ^ { t } \in \mathcal { S } } \\ & { \sum _ { \mathbf { s } ^ { t } \in S } \chi ( \mathbf { s } ^ { t } ) = 1 . } \end{array}\tag{6}
$$

In summary, we propose two complementary solution approaches. The first, heuristic phase, is a fully distributed mechanism, where agents independently update their strategies, using locally available information and observations over the choices of the other agents. The second, correlated device, is a semi-centralized mechanism, where a coordinator recommends actions by signaling a distribution over strategies. Together, these two approaches capture the trade-off between decentralization and coordination: the former emphasizes autonomy while the latter leverages centralized guidance to improve global performance.

## 5 Evaluation

Evaluation assumptions. We begin this section by describing the assumptions used for the energy and accuracy models introduced in Subsections 3.3 and 3.4, respectively. These assumptions are not required by the proposed method in Section 4; they are introduced only to facilitate the experimental evaluation. Therefore, they can be replaced by alternative energy and accuracy modeling frameworks without affecting the generality of the proposed approach.

For the accuracy modeling, we adopt a common assumption in learning systems that increasing the amount of training data improves the model accuracy, while the marginal improvement decreases as more data becomes available. To capture this diminishing-return behavior and the iterative nature of the training process, we consider the following gain function,

$$
\xi _ { i } ( d _ { i } ^ { t } , \theta _ { i } ( d _ { i } ^ { t } , t ) , t ) = 1 - \exp ( - \theta _ { i } ( d _ { i } ^ { t } , t ) \cdot ( d _ { i } ^ { t } ) ^ { \beta } ) ,\tag{7}
$$

where the term $( d _ { i } ^ { t } ) ^ { \beta }$ captures the diminishing returns w.r.t. the amount of data. Observe that if $d _ { i } ^ { t } = 0$ at round $t ,$ then $\alpha _ { i } ( 0 , \theta _ { i } ( 0 , t + \dot { 1 } ) , \dot { t } + 1 \bar { ) } = \alpha _ { i } ( 0 , \theta _ { i } ( 0 , t ) , t )$

Moreover, as stated in Subsection 3.3, when agent i does not train the model for several consecutive rounds, the local model performance may deteriorate due to changes in the underlying data distribution. Let $\widetilde { \alpha } _ { i } ( t )$ denote the last nominal accuracy achieved by agent i before round t. The effective accuracy before retraining and after drift i modeled as $\alpha _ { i } ^ { ( - ) } ( t ) = \widetilde { \alpha } _ { i } ( t ) - H _ { i } ( t )$ , where $H _ { i } ( t ) = \left( \widetilde { \alpha } _ { i } ( t ) - \alpha _ { i } ^ { \mathrm { m i n } } \right) \cdot \left( 1 - e ^ { - \mathcal { A } _ { i } \mathcal { t } _ { i } ( t ) } \right)$ . Here, $\ell _ { i } ( t )$ represents the number of consecutive rounds during which agent i does not participate, $\alpha _ { i } ^ { \mathrm { { m i n } } }$ is the minimum achievable accuracy level, and $\mathscr { d } _ { i }$ controls the drift rate.

When agent i resumes training, its accuracy updated as $\alpha ^ { ( + ) } ( t ) \gets ( 1 - h ) \alpha _ { i } ^ { ( - ) } ( t ) + h \alpha ^ { ( + ) } ( t )$ , where $h \in [ 0 , 1 ]$ and controls how effectively local training compensates for drift. To be more clear, $\widetilde { \alpha } ( t )$ is the pre-drift reference, whereas $\alpha _ { i } ^ { ( - ) } ( t )$ is the post-drift and pre-trained accuracy. The sequence of quantities regarding accuracy is: $\alpha _ { i } ^ { ( + ) } ( t - 1 ) $ $\tilde { \alpha } ( t )  \alpha _ { i } ^ { ( - ) } ( t )  \alpha _ { i } ^ { ( + ) } ( t )$

For the energy consumption, the total energy consumption consists of two components: the energy required for local computation, denoted by $\varepsilon _ { i } ^ { g } ( \cdot )$ ), and the energy required for communication with the AI-service provider, denoted by $\varepsilon _ { i } ^ { t r } ( \cdot )$ . Inspired by [23], the energy consumed by agent i for computing the gradients of the local loss is given by

$$
\varepsilon _ { i } ^ { g } ( d _ { i } , t ) = \kappa \cdot C _ { i } \cdot I _ { i , t } \cdot d _ { i } \cdot f _ { i } ^ { 2 } ( t ) ,
$$

where $\kappa$ is the effective switched capacitance, $C _ { i }$ is the number of CPU cycles required for computing one sample data, and $f _ { i }$ is the computational capacity of agent i at time t. The term $I _ { i , t }$ is the number of local iteration at agent i at time t and we consider it constant in the work. In general, this quantity depends not only on the selected data subset $d _ { i } ,$ , but also on the current state of the global model $\mathbf { w } ^ { t }$ at round t. As discussed in Subsection 3.4, we assume that the communication energy between each agent and the AI-service provider is constant, i.e., $\varepsilon _ { i } ^ { t r } ( \cdot ) = c ^ { t r }$ for all i. Therefore, the energy consumption for agent i is

$$
\varepsilon _ { i } ( d _ { i } , t ) = \varepsilon _ { i } ^ { g } ( d _ { i } , t ) + c ^ { t r } .\tag{8}
$$

Regarding the energy availability at round t, the system operates under limited green energy availability, denoted by $\mathcal { G } ^ { t }$ . Green energy $\breve { \mathcal { G } } ^ { \bar { t } }$ is known to the AI-service provider; users may have only local information. For simplicity, we assume that the AI-service provider allocates this quantity uniformly among the users in ${ \mathcal { N } } ,$ , therefore, the latter can consume up to a specific amount of green energy $\mathcal { \hat { G } ^ { t } }$ . If $\dot { \mathcal { G } } ^ { t }$ is not sufficient to cover the energy required for all tasks, the system can consume as much grid energy $\mathcal { U } ^ { \tilde { t } }$ as possible.

Synthetic data. For our experiments, we evaluate the framework for a population of agents $\mathcal { N } = \{ 1 , 2 , 3 \}$ , where each agent $i \in \mathcal N$ has an action set $s _ { i }$ of size $| S _ { i } | = 4$ . Synthetic datasets are generated to simulate the bundle of samples available to each agent, the quality of these samples, the renewable energy availability, and the penalty parameter $\gamma .$ . For all the following experiments, the total available renewable energy $\dot { \boldsymbol { { g } } } ^ { t }$ in turn t is sampled as an integer uniformly from the interval [50, 100]. For each agent, the bundle of samples is generated from the distribution $U [ \breve { { \mathcal { G } } ^ { t } } - \delta _ { - } , \breve { { \mathcal { G } } ^ { t } } + \mathbf { \bar { \delta } } _ { + } ] ^ { 4 }$ , where scalars $\delta _ { - }$ and $\delta _ { + }$ capture the deviation of the agent’s dataset size w.r.t. the total available renewable energy $\bar { \mathcal { G } } ^ { t }$ . Here $\mathcal { G } ^ { t }$ serves only as a system-wide scaling parameter and does not imply a causal relationship between the ecosystem’s green energy availability and agents’ energy demand.

Next, the quality of agent’s i samples $\theta _ { i } ( \mathbf { s } _ { i } ^ { t } , t )$ is generated as a strictly increasing sequence w.r.t. $\mathbf { s } _ { i } ^ { t } ,$ with values sampled uniformly from $U [ 0 . 0 1 , 1 ]$ for each $t \in T$ . In all experiments, the fictitious play algorithm is executed for $\tau = 1 0 0 0$ iterations. The parameters are set as follows: $\tilde { I _ { i , t } } \sp { * } = 1 0 , \alpha = 0 . 9 , \bar { \alpha } \sp { 0 } = \hat { 0 } . 0 , \sp { * } \alpha _ { i } \sp { \mathrm { m i n } } = 0 . 2 5 , \alpha _ { i } = 0 . 0 5 ,$ $c = 1 0 , c ^ { \mathrm { t r } } = 0 , c _ { 1 } = 1 , c _ { 2 } = 1 0 0 0 , \eta = 0 . 5 2 , C = 1 0 ^ { 2 } , f = 1 0 ^ { 3 } , \zeta = 4 \mathrm { a n d } \kappa = 1 0 ^ { - 9 }$ , for any i and $t \leq T$ . The penalty parameter γ and the parameter h take several non-negative values across experiments.

For the correlation mechanism, we first compute the distribution χ and clip any probability value smaller than $1 0 ^ { - 3 }$ . To analyze the behavior of each player, we compute the marginal distributions induced by $\chi .$ Finally, the overall training process runs for T = 10 rounds, and in each experiment we simulate each round $t \in T$ for 100 turns, reporting the mean values.

We focus on three main quantities, for each round: (i) the total energy (kW h) consumed during training, $\mathcal { E } ( \sigma ^ { \ast } , t ) : =$ $\textstyle \sum _ { i \in [ N ] } \varepsilon _ { i } ( \psi _ { i } ( \pmb { \sigma } _ { i } ^ { * } ) , t )$ , where $s _ { i } ^ { * }$ is the solution for agent i provided by either the Algorithm 1 or eq. (6), (ii) the training accuracy, eq. (7), that each agent achieves, and (iii) the mean accuracy (± one standard deviation) achieved. For the case of energy agnostic agents, we compute the correlated equilibrium using the social welfare function $\textstyle \sum _ { i \in [ { \mathcal { N } } ] } u _ { i } ^ { t } ( s ^ { t } )$

In Figures 4-12, the y-axis in the left panels is the total energy consumption (kWh). The blue dashed curve denotes the total availability of green energy in each round of the procedure. The middle panels in Figures 4-12 demonstrate the energy consumed by each agent, with a solid curve denoting the energy consumption achieved via the heuristic approach, and the dashed curves denote the energy consumption achieved via the correlation device. The rightmost panels in Figures 4-12 demonstrate the training accuracy that each agent achieves and the system as a whole. The diamond-shaped points indicate the accuracy that each agent achieved using the heuristic approach, after concluding the local training in each round and before the AI service provider communicates the updated global model for the next round. The black curve denotes the weighted mean accuracy for the heuristic approach. The teal region illustrates the case where the global model’s accuracy follows the best(worst) accuracy achieved by agents in each round. Similarly, we use the pentagon-shaped points for the accuracy that each agent achieved using the correlation device. The gray dashed curve denotes the weighted mean accuracy for the correlation device. The magenta region illustrates the case where the global model’s accuracy follows the best(worst) accuracy achieved by agents in each round. Further, in all plots, the heuristic phase is denoted as NE and the correlation mechanism as CE. Finally, in case a setting reaches the desired global accuracy threshold before the completion of T rounds, then we depict its performance, e.g., energy consumption, accuracy, etc, with loosely dashed horizontal lines.

![](images/93854a9169ac2595e2905e368c03b5a6c2baf01f1dbb86e69dfbf5a20318785a.jpg)

![](images/6e110cd20a055b848226fae9f5f99977b82f8a170af1964fd93da8111a37b447.jpg)

![](images/1de06d40e57287d78cd0421ba8ea609a20e229b5fba36f673f9e0b125c2673cb.jpg)  
Figure 4: Energy-agnostic training: Total energy consumption (left), energy consumption per agent (middle), total training accuracy and total training accuracy per agent (right), for the case where agents are energy-agnostic with $\delta _ { - } = \delta _ { + } = 1 0$

![](images/842eaf6901a3f13b5b2a7f828dd3da668fe30ae1fca17e874e9e5a1ceeda9658.jpg)

![](images/8afe3e1218bc6ec11432b1c72b7fffe4eefdfc6ca6058647295a07d3af23c019.jpg)

![](images/21e34101d7cab2e4713b4caa014f6c4932969d804bf7d315164993ec2a166e03.jpg)

![](images/2f2e2cb7b9f35105b20cb29c0ee39352200ee0d02693b8c03b5999aa86a948ea.jpg)

![](images/07e00fd62752e34e8d6af1975f7bfa3bc7ecf92d8d27caf32ffd411cccd40249.jpg)

![](images/dbb6f5424bc8ea02b666f60deceec2e00fa0d6b170b613fc33038bc5eb91c246.jpg)

![](images/137d2191ff878b22c84edd4d4c1f3a6cfcd2ef7718d0b9bd2ee396faf95fdf0b.jpg)

![](images/5c57b91e6e51faf45b267864efd8df4d7e5c3bbc6bda1bc09c675bef73ea4d67.jpg)

![](images/bcb23c4c7f58e4264cb5452dd0034d9d30a90c9c19c7d7da1e2d0b372451e561.jpg)  
Figure 5: Energy-aware training: Total energy consumption (left column), energy consumption per agent (middle column), training accuracy (right column), for the case where agents are energy-aware, with $\gamma = 1$ (upper row), $\gamma = 1 0$ (middle row), and $\gamma = 1 0 0$ (lower row), with $h = 0 . 5$ and $\mathit { \bar { \delta } } _ { + } = \delta _ { - } = 1 \bar { 0 }$

Energy-aware mechanism vs energy-agnostic training. Figures 4 and 5 compare two different settings: in Figure 4 agents are energy-agnostic, while in Figure 5 agents are energy-aware $( \gamma \in \{ 1 , \bar { 1 0 } , 1 0 0 \} )$ ), and we set $\delta _ { - } = \delta _ { + } = 1 0$ and $h = 0 . 5$

In Figure 4, agents are energy-agnostic, meaning that $g _ { i } ( s _ { i } ^ { t } , t ) = \varepsilon _ { i } ( \psi _ { i } ( s _ { i } ^ { t } ) , t )$ in eq. (5) and $u _ { i } ( { \bf s } , t ) = { \bf 1 } \{ s _ { i } ^ { t } >$ $0 \} \cdot \ \breve { P } _ { i } ( \mathbf { s } ^ { t } , t )$ in eq. (4) for every i, so the parameter γ does not affect their decisions. In this setting, the training process is driven exclusively by accuracy considerations. Agents are incentivized only to maximize their predictive performance and therefore consistently select the most informative data samples available. As this behavior remains unchanged over the training rounds, the system rapidly reaches a steady energy consumption level, while the accuracy continues to improve due to the progressive refinement of the global model. The model reaches the desired global accuracy threshold of $\alpha \ : = \ : 0 . 9$ in round 5, and the training process is terminated for both the heuristic and the correlated device approaches.

Moreover, the availability of green energy does not influence the agents’ decisions or the resulting system dynamics. Consequently, the system converges to a stable regime in which total energy consumption remains constant across rounds, individual accuracies reach high values, and the mean accuracy exhibits negligible variation, right plot in Figure 4. This behavior indicates that, without energy-aware incentives, and the agents do not explicitly trade off energy consumption against model performance. These observations apply to both the heuristic approach and the correlation device approach.

In contrast, Figure 5 illustrates the case where agents are energy-aware. In this setting, total energy consumption exhibits noticeable fluctuations across rounds (see the left column of Figure 5), reflecting the continuous adaptation of agents’ strategies to balance energy expenditure and learning benefits. Unlike the energy-agnostic case, agents respond to the energy-related incentives and dynamically adjust their decisions rather than following a fixed strategy (see the middle column of Figure 5). This adaptive behavior also results in visible variations in the individual strategies.

The impact of this strategic adaptation is further reflected in the training accuracies (see the right column of Figure 5). Individual agents achieve different accuracy levels across rounds (represented by the diamond-shaped and pentagonshaped points); however, all agents contribute to a steady improvement of the global model accuracy (represented by the black solid curve and gray dashed curve). The differences in local accuracy levels, together with the variations in individual energy consumption, highlight the inherent trade-off between energy efficiency and model performance. Agents allocate computational resources when the expected improvement in accuracy justifies the associated energy cost. Comparing the energy efficiency of the two solutions, we observe that both approaches the training process remains within the available renewable-energy budget, in the energy-aware setting, while the heuristic approach better captures the evolution of $\mathcal { G } ^ { t }$ (see the left column of Figure 5).

![](images/a683db0ce2445e85352b8994a592867b1a562e3e295b7079ec50ad6ba14af5e1.jpg)  
Figure 6: Energy-aware training: Total energy consumption (left column), energy consumption per agent (middle column), training accuracy (right column), for the case where agents are energy-aware, with $\gamma = 1$ (upper row), $\gamma = 1 0$ (middle row), and γ = 100 (lower row), with $h = 0 . 5$ and $\delta _ { + } = \delta _ { - } = 2 0$

Moving to the learning dynamics, the independent updates produced by Algorithm 1 yield higher weighted accuracy but also introduce greater variability across agents (see the right column of Figure 5). In contrast, the correlation mechanism reduces variance by steering agents toward more homogeneous, moderate behaviors, at the cost of slightly lower average performance. This difference is further highlighted in the per-agent accuracy points (see right column of Figure 5), where fictitious play allows more diverse outcomes, some agents achieve higher accuracy while others limit participation, whereas correlation induces more synchronized behavior among agents with similar characteristics. As a result, fictitious play promotes efficiency through heterogeneity, while correlation emphasizes stability and coordination.

Finally, increasing γ, which makes the AI service provider’s policy stricter, leads the two approaches to behave more similarly in terms of energy consumption (see left column of Figure 5). Starting from $\gamma = 1$ , the variance of the performance very quickly narrows for the correlated device. Nevertheless, the black curve (heuristic approach) reaches a plateau<sup>2</sup>, meaning that small γ values do incentivize agents, but this may be insufficient to make them achieve the global accuracy and, instead, remain idle when the global model secures a “good” accuracy level. The same trend appears for the heuristic approach, but with a significantly lower pace.

As γ is increased non-participation is discouraged more aggressively, reducing performance variance within and be tween the heuristic phase and the correlation mechanism, see also Figures 6 and 7. Moreover, higher γ values dis courage prolonged non-participation and can therefore help the agents overcome the accuracy plateaus observed for smaller penalty values, as in case where $\gamma = 1$ . Interestingly, in all cases displayed in Figure 5, the heuristic approach achieves faster the global accuracy threshold $\textstyle \alpha = 0 . 9$ , achieving the faster convergence in round 5 for $\gamma = 1 0$

Energy demands. Now we alter the experimental setting by examining how the distribution of users’ energy demands affect the training dynamics and the resulting energy consumption. Specifically, we consider two scenarios where agents’ energy demands are sampled from $U [ \mathcal { G } ^ { t } - \mathrm { \ddot { 1 0 } } , \mathcal { G } ^ { t } + \mathrm { 1 \dot { 0 } } ] ^ { 4 }$ , Figure $5 ,$ and from $U [ \mathcal { G } ^ { t } - 2 0 , \mathcal { G } ^ { t } + 2 0 ] ^ { 4 }$ Figure 6, respectively.

From the system-level perspective, we observe that when narrower energy demand distributions are applied and the γ values are relatively small, the heuristic approach produces outcomes that are closer to the actual green energy per active training round. This is mainly because agents experience similar energy conditions and, therefore, their participation decisions remain relatively aligned with the actual energy availability. In contrast, the correlated device approach tends to synchronize agents’ behaviors through the provider’s signals, resulting in more homogeneous energy consumption trajectories across users. Although this coordination improves consistency, it may reduce the ability of individual agents to adapt to their own local energy conditions.

From the model-level perspective, the reduced participation frequency of some agents directly influences the evolution of local accuracy through the model drift mechanism. Agents that remain inactive for several consecutive round experience a gradual degradation of their local model performance, as their accuracy decreases from the last achieved value toward the baseline accuracy level, see the evaluation assumptions for more details. When these agents resume participation, the parameter h determines the extent to which the accumulated drift is compensated. Since $h < 1$ , the recovery is only partial, and agents require multiple participation rounds to fully restore their local accuracy. Therefore, under heterogeneous energy conditions, independent decision-making may lead to more irregular participation patterns, resulting in fluctuations in the local and aggregated accuracy trajectories.

However, as show in Figure 6 increasing the energy-demand variability does not necessarily degrade the learning performance. In fact, for wider energy-demand distributions, the two mechanisms exhibit more similar aggregate consumption profiles. This suggests that the larger variability in individual energy demands changes the participation decisions in a way that reduces the differences between the two approaches. This behavior is consistent with agents adjusting their participation more frequently in response to the mismatch between their energy demands and the available renewable energy.

Nevertheless, the heuristic approach generally achieves higher global accuracy than the correlated device approach across the two energy-demand distributions. This suggests that allowing agents to independently adapt their participation decisions enables a more effective selection of training contributions, whereas excessive coordination may constrain agents’ ability to exploit favorable individual energy conditions.

In contrast to the case of narrower energy-demand distributions, Figure 5, in Figure 6 the global accuracy threshold is never reached. Nevertheless, the heuristic approach maintains higher accuracy than the correlation mechanism throughout the experiment, while the two approaches exhibit more similar energy-consumption patterns than in the narrower-demand setting. Thus, increasing the heterogeneity of energy demands does not simply translate into lower energy consumption or higher accuracy; rather, it changes the participation dynamics and the resulting trade-off between energy consumption and learning performance. Within the considered horizon, the heuristic approach provides a more favorable accuracy outcome, although neither mechanism achieves the prescribed accuracy target.

For $\gamma = 1$ , in both energy-demand distributions, the heuristic approach reaches quickly a relatively good global accuracy. However, for $\delta _ { + } = \delta _ { - } = 2 0$ , the wider range of energy demands leads to less favorable participation patterns under the same energy incentive, and the global accuracy reaches a plateau below $\mathcal { \alpha }$ . In contrast, when $\delta _ { + } = \delta _ { - } = 1 0$ , the energy demands remain closer to the available renewable energy, allowing participation to be sustained and the global accuracy to continue improving until the target is reached.

For $\gamma = 1 0$ and wider energy-demand distributions, middle row in Figure $^ { 6 , }$ the correlated device exhibits a step-like accuracy evolution due to more synchronized participation, whereas the heuristic approach improves more smoothly through independent participation decisions. In contrast to $\gamma = 1$ , heuristic approach appears to overcome the saturation and continue improving toward the target accuracy.

For $\gamma = 1 0 0$ , the gap between the heuristic and correlated-device approaches further narrows, as in the previous case. The heuristic approach exhibits a more step-like accuracy evolution, while the correlated device produces a more gradual, almost linear improvement. The stronger participation incentive also enables agents to exit the saturation regime earlier, allowing agents to make further progress toward $\hat { \alpha } .$ . This behavior reflects the stronger pressure to participate when the expected training benefit justifies the energy cost, reducing prolonged periods of non-participation and making the two approaches behave more similarly.

Overall, the results highlight a trade-off between coordinated participation and individual adaptability. Wider energydemand distributions reduce differences in energy consumption across agents, but the learning performance depends on how participation decisions interact with model drift and recovery. While the correlated device approach can create more synchronized participation patterns, the heuristic approach benefits from decentralized adaptation, enabling agents to exploit their individual conditions and achieve higher accuracy. These findings emphasize that coordination mechanisms should balance energy alignment with preserving sufficient flexibility for agents to contribute when their participation is most beneficial.

![](images/ac78e4d462941efad305f76bd380c313f892cc79b5be28484852f1a7f860fd88.jpg)

![](images/2801ee1964eacb1f67d59842f5b651680099839814b100b4ac553dcf41f932dc.jpg)

![](images/3b26b2cc0937d7f9840e00472ef75e2dca98022920d6cac2cbca7d2541f8983f.jpg)

![](images/00a82d968df1b0a8740e38254d62e56ec16347104d861490ddff8bb460300ff0.jpg)

![](images/866e7e44135ca8342560d509d5cf553afee833758149f20aa00f15d6479f28ce.jpg)

![](images/4a6fbe4b11c9cc7f787f593e1e3d17d667528031ee521878cec50eec7b6d292f.jpg)

![](images/712b71f19854a0543cdc3337982ab93d964f02747b8c99ecc5d7ecf0fb14201e.jpg)

![](images/1f288a112b5c9995566d8121707c90b318708c8078f87002c2c6cc42edb06767.jpg)

![](images/12fed07ec5ff1c8691b8915d6e9faa0a65b4b1e8a9a84e0ea1c48c198169fe31.jpg)  
Figure 7: Energy-aware training in a heterogeneous population: Total energy consumption (left column), energy consumption per agent (middle column), training accuracy (right column), for the case where agents are energyaware, with $\gamma = 1$ (upper row), $\gamma = 1 0$ (middle row), and $\gamma = 1 0 0$ (lower row), and $h = 0 . 5$

Heterogeneous population. We next consider a heterogeneous population of agents. In particular, agent 1 has a significantly larger dataset bundle, with $S _ { 1 } \sim U [ \mathcal { G } ^ { t } - \delta _ { - } , \mathcal { G } ^ { t } + \delta _ { + } ] ^ { 4 }$ , where $\delta _ { - } = 2 0$ and $\delta _ { + } = 1 0 0$ . From eq. (8), agent 1 can thus be interpreted as a high energy-demand agent. The remaining agents have moderate dataset bundles, $\mathrm { i } \mathrm { \check { . e } . , } S _ { i } \sim U [ \mathcal { G } ^ { t } - 1 0 , \mathcal { G } ^ { t } + 1 0 ] ^ { 4 }$ for all $i \neq 1$ . Furthermore, all agents experience penalty $\gamma \in \{ 1 , 1 0 , 1 0 0 \}$

In Figure 7, the energy consumption trajectories (left column) remain comparable across the two approaches, particularly for moderate and high penalty values. For $\gamma \in \{ 1 0 , 1 0 0 \}$ , both mechanisms adapt their decisions to the available renewable energy and exhibit similar aggregate consumption patterns across training rounds. For small penalty values, the correlated device produces smoother energy trajectories compared with the heuristic approach. This occurs because the provider’s signal coordinates agents’ decisions, reducing uncertainty in individual participation choices and avoiding abrupt changes in energy usage. In contrast, the heuristic approach relies on decentralized decisions, where each agent independently evaluates its own trade-off between energy consumption and participation benefits, leading to larger fluctuations. As γ increases, the two approaches become more aligned since the higher cost of non-participation dominates individual preferences and encourages agents to adjust their behavior more conservatively.

Despite the similarity in aggregate energy consumption, the two approaches lead to different participation patterns. In contrast to the homogeneous case, where different agents may become dominant depending on the realization of energy demands, the heterogeneous setting creates a persistent asymmetry due to the larger energy requirements of agent 1 (blue curves in the middle column of Figure 7). For $\gamma = 1$ , agent 1 becomes an important contributor to the training process, but does not completely dominate the learning dynamics. Further for $\gamma = 1 0 0$ it does dominate the heuristic approach. The remaining agents either participate intermittently or reduce their contribution depending on their individual energy conditions. Consequently, agent 1 consumes a moderately larger share of the available green energy budget.

For $\gamma = 1 0$ , however, the medium non-participation penalty significantly alters the agents’ decisions. Although abstaining becomes costly, agents do not necessarily increase their contribution uniformly. In particular, agents 2 and 3 frequently abstain, while agent 1 also reduces its participation because its higher energy demand makes participation more costly. In this experimental setting, this reduces the attractiveness of training for the high-demand agent and may ultimately lead to reduced participation despite the penalty. This result illustrates that excessive penalties may create undesirable incentives: agents may participate less efficiently or reduce their training contribution rather than providing additional useful updates.

The accuracy trajectories (right column in Figure 7) further highlight the interaction between heterogeneous energy availability, participation decisions, and model drift. Overall, the heuristic approach achieves higher accuracy than the correlated device approach, although the difference depends on the penalty level. For $\gamma = 1$ , almost all agent are constantly engaged in the training and both approaches converge quickly to the $\mathcal { \alpha } ,$ and the accuracy trajectories of the two approaches are very close. However, as training progresses, the correlated approach exhibits larger variability across agents. The coordinated decisions may cause some agents to become systematically less active, leading to the accumulation of model drift at those agents. Since the recovery parameter satisfies $h < 1$ , returning agents only partially recover their lost accuracy after retraining, resulting in persistent differences across local models.

For $\gamma = 1 0$ , both approaches slow down their performance, with the correlated mechanism experience the larger degradation. For $\gamma = 1 0 0$ , the divergence between agents becomes more pronounced. Although the correlated device maintains a more coordinated participation structure, this coordination does not necessarily translate into a more balanced learning process. Instead, the provider’s signal can concentrate training activity among a subset of agents, causing other agents to contribute less frequently and experience larger drift effects. Conversely, in the heuristic approach, participation is determined independently, and although some agents may remain inactive, the learning process can be driven effectively by the most active contributors. In the considered setting, agent 1 becomes the major contributor, resulting in a global accuracy improvement primarily driven by its local training effort. For $\gamma \in \{ 1 0 , 1 0 0 \}$ the heuristic approach experiences saturation.

This behavior highlights an important distinction between coordination and fairness. The correlated device reduces randomness in participation decisions but may introduce a stronger form of participation bias, as agents respond similarly to the provider’s signal and some agents become systematically underrepresented. The heuristic approach, despite producing more heterogeneous individual behaviors, allows agents to exploit their own energy conditions and can avoid excessive concentration of training effort. From the perspective of model drift, this suggests that coordinated inactivity can be particularly harmful because multiple agents may simultaneously accumulate accuracy degradation, whereas decentralized decisions naturally distribute participation opportunities over time.

Overall, the heterogeneous experiments reveal a trade-off between coordinated energy management and learning diversity. The heuristic mechanism promotes selective participation based on individual incentives and energy availability, whereas the correlation device provides stronger coordination but may unintentionally amplify participation imbalance. Therefore, while coordination can improve predictability of energy consumption, it must be carefully designed to avoid reducing the representation of specific agents and increasing drift-related accuracy degradation.

Accuracy vs Green energy efficiency. In Figures $5 \textrm { -- } 7 .$ , we observe a trade-off between accuracy and total energy consumption, together with a strong dependence on the penalty parameter $\gamma$ . In both approaches, the energy contribution of grid energy remains zero. This indicates that the system consistently adapts to operate within the renewable energy budget rather than exceeding it. For small values of $\gamma , { \bf e . g . } , \gamma = 1$ , participation is selective and heterogeneous. In the energy-aware homogeneous case, see Figures 5 and $^ { 6 , }$ agents dynamically adjust their training intensity, leading to fluctuating energy consumption and diverse accuracy outcomes (the diamond- and pentagon-shaped points). In the heterogeneous case, see Figure 7, this effect is clearer: under the heuristic phase, some agents, $\mathrm { e . g . }$ , agent 1, are the only agents not abstaining, and take on most of the workload; under the correlation mechanism, participation is instead concentrated on all agents. In both cases, the outcomes reflect how the limited renewable-energy budget is allocated across agents with different energy demands and training incentives, which leads to variability in both individual and mean accuracy. $\mathbf { A s } \gamma$ increases, $\mathrm { e . g . , } \gamma = 1 0 0$ , abstaining becomes less favorable if the global accuracy threshold has not been achieved, and agents exhibit more consistent, but conservative, participation. This results in more homoge neous behavior across agents and reduces variability in both energy consumption and accuracy. However, rather than significantly increasing performance, the stricter penalty leads agents to operate cautiously within the renewable energy constraint, which keeps accuracy improvements modest. Overall, γ acts as a key control parameter that regulates participation incentives under energy constraints. Lower and intermediate values allow flexible, uneven participation that can improve efficiency but introduce variability, while higher values enforce more uniform behavior, reducing variance at the cost of limiting performance gains.

Global accuracy threshold effect. Here we examine the effect of global accuracy threshold a. In particular, we set $\gamma = 1 , h = 0 . 5 , \delta _ { + } = \delta _ { - } = 1 0 , \mathrm { a n d } \alpha \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$

The upper row of Figure 5 illustrates the case where $\alpha = 0 . 9$ . Under the heuristic approach, the target accuracy is achieved after five training rounds, at which point the provider discontinues incentives and the training process terminates. In contrast, under the correlated device, the aggregated accuracy reaches only approximately 77% within the available ten rounds. This behavior is explained by the coordinated recommendations, which induce some agents to postpone participation in specific rounds. As a result, model drift accumulates at inactive agents, and because $h \ : = \ : 0 . 5$ only partially compensates for the lost accuracy upon re-entry, the recovery of the aggregated accuracy becomes slower, preventing the system from reaching the desired threshold within the considered time horizon.

![](images/3bfc0d5f0015d5525c945e0356d81e39b13ffaeb7a3a93ce0a53a017c1174b76.jpg)

![](images/2500dfd9eff2ba06cd3987f6134bdc676b282fe6e5c17f6cfa38d0f1ce4227bc.jpg)

![](images/da82179f9582af2bd37a4561dde722ee7ab328440db98bcb146d1bf282e56829.jpg)

![](images/e165f994f027b1202f8bf0d20d3cc452ea841d53b6e67c9745293da59742113d.jpg)

![](images/c4136319045411696e4052d7a74418ff4c97d6e90fcb09af449eb7fce44ffaa1.jpg)

![](images/98ead5a219eda03426ba1a01c861b1429b33da8b2a4538d9c95410ccd8ad89a2.jpg)  
Figure 8: Global accuracy threshold. Total energy reduction (left column), training accuracy per agent (middle column), mean training accuracy (right column), for the cases where: (i) $\alpha = 0 . 5$ (upper row), and (ii) $\alpha = 0 . 7$ (lower row), with $h = 0 . 5 , \gamma = 1$ and $\delta _ { + } = \delta _ { - } = 1 0$

For $\alpha = 0 . 7$ , shown in the lower row of Figure 8, the heuristic approach reaches the target accuracy after three rounds. Further, the correlated device does not require the entire training horizon to attain the prescribed threshold, but only four rounds. Noteworthy, the aggregated accuracies increase monotonically exhibiting periods of different growth rate, due to the fact that in some rounds some agents may abstain from training. This behavior reflects the dynamic interaction between participation decisions and drift. During rounds with limited participation, the accumulated drift slows the improvement of the global model. When previously inactive agents resume training, the blending h partially restores their local accuracy, producing temporary accelerations in the convergence process. Consequently, the global accuracy evolves in a non-uniform manner before eventually reaching the prescribed threshold.

Clearly, for lower a values the training process is terminated in earlier rounds, e.g., round 2 (heuristic phase) and round 3 (correlated device) when $\alpha = 0 . 5$ , and round 3 (heuristic phase) and round 4 (correlated device) when $\alpha = 0 . 7 $ , in Figure 8.

For all considered values of $\mathcal { \alpha }$ , we observe that agents under the correlated device participate more frequently than under the heuristic approach. Nevertheless, the participation dynamics remain noticeably different for different value of $\mathcal { \alpha } .$ . While the heuristic approach exhibits relatively stable participation throughout the incentivized period, the correlated device leads several agents to temporarily suspend their participation and subsequently re-enter the training process. Owing to the relatively low target accuracy, these temporary inactivity periods do not significantly affect the stopping time. However, they demonstrate the ability of the correlated recommendations to coordinate participation while exploiting the recovery mechanism. In the considered experiments, this suggests that intermittent participation can reduce training activity without preventing the system from reaching moderate accuracy targets. As the required accuracy increases, however, the accumulated effects of model drift become more pronounced, making sustained participation increasingly important for convergence.

Drifting phenomenon. To evaluate the impact of intermittent agent participation on model performance, we investigate the emergence of drift under penalty levels $\gamma \in \{ 1 , 1 0 , 1 0 0 \} , \alpha = 0 . 9$ and $h = 0 . 5$ , Figure 9. Further, we consider two different regimes regarding the agents’ bundle of samples, $\delta _ { + } = \delta _ { - } = 1 0$ , and $\delta _ { + } = \delta _ { - } = 2 0$ . Specifically, we analyze how the incentive mechanism influences agents’ training decisions and how these decisions affect the evolution of local model accuracy over time.

![](images/a84b539bda538d5167cc9ab392763c80f8649afb4deae2df52338e8f89a401e8.jpg)

![](images/fceae78555047ca6d012c6c4d152d6637502716477e83804c2cacdb609a93122.jpg)

![](images/c9b29a17d4516089203a1150408e987a7263222424564133bea1bb125cdc847e.jpg)

![](images/41ebef7201d603e73e5cbd35b5c2695fd15089d48352989446aec0376fb2f969.jpg)

![](images/10e361573e038c1858c9ce42a1c569d77c847feb9d4c6b23ec0affe57d07f9c3.jpg)  
Figure 9: Drifting phenomenon: Here, each agent i takes: $( \mathrm { i } ) \gamma = 1$ (left column), (ii) $\gamma = 1 0$ (middle column), and (iii) $\gamma = 1 0 0$ (right column), and $\delta _ { + } = \delta _ { - } = 1 0$ (upper row), and $\delta _ { + } = \delta _ { - } = 2 0$ (lower row), for $h = 0 . 5$

For the case where the agents have more balanced data, that is $\delta _ { + } = \delta _ { - } = 1 0$ , increasing $\gamma$ leads to increasing participation for some agents, see also first row in Figure 5. When $\gamma = 1$ , for the heuristic approach, agent 1 is almost never engaged in the training, resulting into high drift. remain engaged until the global accuracy threshold a is reached. For $\gamma = 1 0$ , agents 3 abstains from the first round of the training, while all other agents participate, see second row in Figure 5, so only agents 3 experiences drifting. Finally, for $\gamma = 1 0 0$ , agents oscillate between participation and non-participation, see lower row in Figure 5. Hence, the drift follows the same pattern. Finally, for $\gamma = 1 0 0$ case, engagement to training is almost complete, so agents experience small drift. For all $\gamma$ values using the correlated device, all agents are almost fully committed to the training process, so the drifting phenomenon is negligible.

When agents have wider energy demand distributions, that is $\delta _ { + } = \delta _ { - } = 2 0$ , the participation follows different trend. The main difference is that when $\gamma = 1 0 $ , all agents may abstain from training, resulting into significant drift. For higher penalty values, $\gamma \in \{ 1 0 , 1 0 0 \}$ , we observe the same behavior as the case where $\delta _ { + } \bar { = } \delta _ { - } = \bar { 1 0 }$

These results highlight that the effect of the non-participation penalty is not independent of the local data distribution. When agents have highly uneven data availability, increasing γ to moderate values may discourage participation and amplify model drift. However, when data availability is more balanced, the same penalty can reinforce participation and improve model stability. Therefore, the effectiveness of incentive mechanisms depends not only on the penalty design but also on the distribution and relative contribution of local training data across agents.

Blending effect. Here we examine the effect of blending both from systemic and agent perspective, setting $h \in$ $\{ 0 . 5 , 0 . 7 \bar { 5 } , 1 \}$ , see Figures 5, 9, 10, and 11. When $h = 1$ we examine immediate recovery, Figure 11. Further, we set $\mathcal { \bar { d } } _ { i } = 0 . 0 5 , \mathcal { \bar { \alpha } } = 0 . 9 , \mathcal { \gamma } \in \{ 1 , 1 0 , 1 0 0 \}$ and $\delta _ { + } = \delta _ { - } = 1 0$

For $h = 0 . 5$ , the recovery from model drift is partial, meaning that when an agent resumes participation after a period of inactivity, only half of the accumulated degradation is compensated in the current training round. For $\gamma = 1$ , the low penalty for non-participation is insufficient to discourage agents from avoiding training, leading to more frequent participation decisions being skipped, see first row in Figure 5. So agents that skip several training rounds remain with lower accuracy even after returning, see agent 3 (green diamond) in Figure 5. Increasing the penalty to $\gamma = 1 0$ encourages more consistent participation, reducing the occurrence of drift and improving the convergence speed, see upper middle panel in Figure 9 and middle row in 5. For $\gamma = 1 0 0$ , the large penalty makes agents to choose either low intensity training or short abstain periods. Consequently, model drift accumulates, resulting in lower per-agent accuracy and requiring additional training rounds to reach the saturation level, see lower right panel in Figure 5.

In general, since h remains moderate, agents that experience inactivity require multiple rounds to fully recover their accuracy, leading to a longer incentivization period and higher energy consumption compared with larger values of $h ,$ see upper left and right panels in Figure 9.

![](images/86b7755a25dea59250c499f35432589ca61904c8ad58a047f957babf7dcf9f95.jpg)

![](images/825fa06e583749d6aec23386a64a17a631fdae57bc4ce1e28ebe63bc718c92ca.jpg)

![](images/94a821b4e381094dfb1c472cc2b59250e73eeaf4a7459cc4d152190cc20c85cc.jpg)

![](images/fcc778d61d1435fe079c9ad43d1a05f850164df1d017951f32c40f79c3ade931.jpg)

![](images/516d7db276157ffe4b2f8ed78bc809702b9d916b02defedd9b19a18d0ca303e0.jpg)

![](images/14dc668e2cec586ec4bf3bc2dce5a0bac06d74434ab136c363636c6ec29269c8.jpg)

![](images/48dc051b93e1d7b292015a4f0daeb2bd64f7027b9ec9742772e664f20b542ee8.jpg)

![](images/b27fd24399419e85930310c9e66639951ea1cabdf1a508aec4339204fc5c16a2.jpg)

![](images/7418888c0a1deec0286da76d0a8c3cb5bba7e628554d4cb278424a654bc13a3f.jpg)  
Figure 10: Blending effect: Drift per user (left column), energy consumption per agent (middle column), training accuracy (right column), for the cases where agent i takes: $( \mathrm { i } ) \gamma = 1$ (upper row), and (ii) $\gamma = 1 0$ (middle row), and (iii) $\gamma = 1 0 0$ (lower row), with $h = 0 . 7 5$

For $h = 0 . 7 5$ , agents recover a larger portion of their lost accuracy after participating in training. Under $\gamma = 1$ although some agents may still choose not to participate due to the lower penalty, the impact of such decisions is less severe compared with $h = 0 . 5$ , as the accumulated drift is compensated more effectively once agents return. This results in improved accuracy stability and fewer corrective training rounds, see upper row in Figure 10. For $\gamma = 1 0$ , the agents are more indifferent regarding the training process via the heuristic approach. Nevertheless, the high blending value allows them to recovery quickly and the model’s accuracy reaches a saturation point. For $\gamma = 1 0 0 :$ participation is already strongly encouraged; increasing γ hen further mitigates the accuracy cost of occasional non-participation by accelerating recovery.

For $h = 1$ , local training fully eliminates the impact of accumulated drift after a single participation round. With $\gamma = 1$ , agents may still avoid participation due to the low non-participation penalty, but the resulting degradation has a limited impact because agents completely recover when they return to training. Therefore, the accuracy remains more stable compared with $h = 0 . 5$ and $h = 0 . 7 5$ , although occasional non-participation may still increase the number of rounds required to reach the target accuracy. For $\gamma \in \{ 1 , 1 0 \}$ , we observe that the absence of drifting makes the agents to neglect more often the training process, but the full recovery allows them to terminate the training, see the upper row in 11, or to be really close, see the middle row in 11. For $\gamma = 1 0 0 $ , agents are strongly encouraged to participate, and the complete drift recovery mechanism ensures that local accuracies quickly return to their optimal values, see lower row in Figure 11. Consequently, the aggregated accuracy converges faster, the incentive phase terminates earlier, resulting in the lowest total energy consumption among the three recovery settings.

Interestingly, from the above analysis we observe a tension between $\gamma$ and $h , \mathrm { e . g . }$ , middle row in Figure 10 $( \gamma = 1 0$ and $h = 0 . 7 5 )$ . The reason is that they affect two different mechanisms, γ controls participation choice, while h models the consequences of non-participation. In the second row in Figure 10 we observe that a higher drift recovery h can partially compensate for lower participation incentives $\gamma ,$ creating a trade-off. Intuitively, when h is low, e.g., $h = 0 . 5 ,$ , missing training rounds has a significant cost because agents cannot fully recover from drift immediately. Therefore, the training process relies more heavily on γ to maintain participation. A low $\gamma$ may lead to frequent non-participation, accumulated drift, slower convergence, and higher energy consumption due to additional training rounds. When h is higher, agents can faster recover their accuracy after a single participation round. So, the system becomes more tolerant to intermittent participation. Therefore, the provider may not need to impose a high penalty γ because drift effects are quickly corrected. This creates a potential strategic tension: a high h reduces the negative impact of skipping training. Hence, agents may become more willing to avoid participation because they know tha their accuracy can be restored later. Therefore, increasing h may unintentionally reduce the effectiveness of $\gamma$ as an incentive mechanism. In a nutshell, the proper penalty γ also depends on the drifting value.

![](images/d8298ab17d1dfe87c63173d9e86e9556916f464d730caaac18cc2e127c2ef63a.jpg)  
Figure 11: Blending effect. Drift per user (left column), energy consumption per agent (middle column), training accuracy (right column), for the case where agents are energy-aware, with $\gamma = 1$ (upper row), $\gamma = 1 0$ (middle row), and $\gamma = 1 0 0$ (lower row), with $h = 1$

Furthermore, the interaction between $h$ and $\gamma$ may introduce representation bias in the FL process. Specifically, larger values of h allow agents to recover quickly from periods of inactivity, potentially reducing their incentive to participate consistently. Consequently, the global model may reach the target accuracy primarily through the contributions of a subset of frequently participating agents, while the data distributions of less active agents remain underrepresented, $\mathrm { e . g . }$ ., agents 1 and 2 in Figure 8 for $\gamma = 1$ and $\mathcal { \alpha }$ . This highlights a trade-off between learning efficiency and fairness, suggesting that stopping criteria based solely on the aggregated accuracy may not always guarantee balanced learning across all participating agents.

Real-world energy-availability scenario. Figure 12 illustrates the evolution of renewable energy shares of total energy production for Finland, and Portugal (Table 2), and their interaction with total energy consumption under the two mechanisms, heuristic phase and correlation device in Section 4. Importantly, the proposed framework operates in iterative rounds; the yearly data points should not be interpreted as the duration of individual training rounds. Instead, each year represents a different observation of the real-world environment, capturing the evolution of data characteristics over time. The experimental setup uses these yearly snapshots to simulate successive states encountered during deployment. At each state, local agents perform training using the available data representation, return model updates, and the central system aggregates these updates to obtain the next model state. Therefore, the evaluation focuses on the ability of the proposed approach to adapt to evolving real-world conditions rather than assuming that one training round corresponds to one calendar year.

For the proposes of our model, given the Table 2, we assume the following values for each country: Finland $\mathcal { G } _ { f } ^ { t } \sim$ $U [ 1 5 , 3 0 ]$ , and $S _ { i } \sim U [ \mathcal { G } _ { t } ^ { t } - 5 , \mathcal { G } _ { t } ^ { t } + 1 0 ] ^ { 4 }$ , and Portugal $\mathcal { G } _ { p } ^ { t } \sim U [ 1 0 , 2 0 ]$ , and $S _ { i } \sim U [ \mathcal { G } _ { p } ^ { t } - 1 0 , \mathcal { G } _ { p } ^ { t } + 2 0 ] ^ { 4 }$ , for each $i \in \{ 1 , 2 , 3 \}$ . For simplicity, we normalize the total energy production of each country to one unit. Therefore, the reported renewable energy percentages are directly used as normalized green energy availability values. Further, we set $\gamma = 1 0 , h = 0 . 5 , \mathcal { d } = 0 . 0 5$ , and $\alpha = 0 . 9$

<table><tr><td></td><td>1990</td><td>1995</td><td>2000</td><td>2005</td><td>2010</td><td>2015</td><td>2020</td><td>2021</td></tr><tr><td>Finland</td><td>16.3</td><td>18.5</td><td>18.9</td><td>16.8</td><td>18.6</td><td>22.4</td><td>25.4</td><td>27.5</td></tr><tr><td>Portugal</td><td>16.7</td><td>15.8</td><td>12.5</td><td>12.1</td><td>13.3</td><td>16.5</td><td>17.7</td><td>17.8</td></tr></table>

Table 2: Percentage of renewable energy in total energy calculated on the level of Energy Available for Consumption<sup>3</sup>.

From the perspective of the induced energy consumption, both mechanisms operate within a relatively comfortable margin relative to green supply. However, the gap between NE and CE persists, indicating that coordination continues to yield efficiency gains even when renewable penetration is substantial. Nevertheless, the independent behaviours of the agents are more sensitive to the changes in the availability of green energy.

![](images/1fa958c8253dbe781c4d3840062f0b276cc1c27cea0dc3447ae5690f47912f80.jpg)

![](images/4a5b3d76ba90b852cd1375863f67840e8b059597e76d9315363dde5ba1719297.jpg)

![](images/e38aa15001a387214b80693efec36b06f47b01f230dbcbb59857c196839624ae.jpg)

![](images/8b5a626a3c76b940889f6e533663df86d0967eefce906402cb261d20ced9e72d.jpg)

![](images/2bb61126439990fb2269b872f6be7c8ed548d2083b9d82af4425f2f54fc20e3a.jpg)

![](images/aaf7875ce986e75f9bb6562b7a9e8371f936038f2403caa02b06f7edce7c4116.jpg)  
Figure 12: Real-world scenario. Total energy reduction (left column), training accuracy per agent (middle column), mean training accuracy (right column), for the cases Finland (upper row) and Portugal (lower row).

Overall, the figure highlights three main insights. The results indicate that increasing renewable-energy availability does not by itself eliminate the differences between decentralized and coordinated decision-making in the considered scenarios. There is a trade-off between the policies an AI-service provider can follow: coordination is stable, but best responses are more sensitive. Second, the gains from best responses are particularly valuable in early-stage or low-renewable environments, where system constraints are tighter. Third, as renewable shares increase, the system becomes more robust, but strategic control mechanisms still influence stability and variance of total consumption.

Across several experimental settings, the training process may exhibit a saturation point at an accuracy level close to, but below, a, e.g., middle row in Figure 6 and lower row in Figure 7. Once the aggregated accuracy reaches this regime, further improvements become increasingly limited because the incentive to participate is determined jointly by the expected training benefit and the energy cost of participation. When the current accuracy is already relatively high, some agents may therefore find continued participation insufficiently attractive and temporarily abstain from training. Their inactivity, in turn, causes their local models to drift toward the baseline accuracy level, see lower row in Figure 9. When they subsequently re-enter the training process, the recovery mechanism only partially compensates for the accumulated drift when $h < 1$ , so several additional participation rounds may be required to recover the lost accuracy. Consequently, the gains obtained from newly available training contributions can be largely offset by the degradation accumulated during inactive periods, causing the aggregated accuracy to remain in a narrow region and giving rise to the observed saturation. This effect becomes more pronounced when the target accuracy is high, since the system must maintain participation for longer and even temporary inactivity can have a cumulative impact on the global model. Experimentally, we observed that larger penalty may mitigate this issue, see middle and lower row in Figure 6. Summarizing, the observed saturation is not necessarily caused by a lack of available learning capacity, but can emerge endogenously from the interaction between high current accuracy, participation incentives, model drift, and incomplete recovery.

## 6 Conclusions

In this work, we studied carbon-aware Federated Learning under limited renewable energy availability through a gametheoretic perspective. We modeled each training round as a finite normal-form game in which agents strategically select their local training intensity while competing for shared green energy resources. The proposed framework captures the trade-offs between learning accuracy, renewable energy utilization, and carbon-intensive grid consumption.

The payoff structure incorporates diminishing returns in training accuracy, proportional renewable energy allocation, and convex penalties for grid energy usage. This formulation creates strategic interdependence among agents, as changes in one agent’s training decision affect the energy resources available to others. To address the decentralized and information-limited nature of Federated Learning, we introduced a heuristic decision mechanism based on discrete-time fictitious play, allowing agents to adapt their strategies using empirical observations. In addition, we considered a provider-assisted correlation device that generates coordinated green-aware recommendations based on a system-level objective.

Our results demonstrate that appropriate incentive mechanisms can enable sustainable Federated Learning with zerogrid energy consumption under the considered renewable energy availability conditions, while maintaining competitive learning performance. The heuristic mechanism generally achieves higher accuracy by allowing agents to adapt their decisions to individual energy conditions, whereas the correlation device provides more coordinated energy consumption patterns. However, increased coordination may also lead to less balanced participation and amplify the effects of model drift when some agents become underrepresented. These results highlight the trade-off between coordinated energy management and preserving diverse contributions to the global model.

Overall, this work connects Federated Learning, renewable energy allocation, and non-cooperative game theory by providing a framework for analyzing sustainable distributed AI training. Future research directions include extending the model to dynamic renewable availability, heterogeneous energy pricing, asynchronous updates, and scenarios with incomplete observability of global energy conditions.

## Acknowledgment

This work has been partly developed in the scope of the project EXIGENCE, which has received funding from the Smart Networks and Services Joint Undertaking (SNS JU) under the European Union (EU) Horizon Europe research and innovation programme under Grant Agreement No 101139120. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the EU or SNS JU.

## References

[1] Robert J. Aumann. Subjectivity and correlation in randomized strategies. Journal of Mathematical Economics, 1:67–96, 1974.

[2] Lori Bird, Jaquelin M. Cochran, and Xi Wang. Wind and solar energy curtailment: Experience and practices in the united states. 2014.

[3] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe´ Kiddon, Jakub Konecny, Stefano Mazzocchi, H. B. McMahan, Timon Van Overveldt, David Petrou, Daniel´ Ramage, and Jason Roselander. Towards federated learning at scale: System design. ArXiv, abs/1902.01046, 2019.

[4] George W. Brown. Iterative solution of games by fictitious play. In Activity Analysis of Production and Allocation. Wiley, New York, 1951.

[5] Di Chai, Leye Wang, Liu Yang, Junxue Zhang, Kai Chen, and Qian Yang. A survey for federated learning evaluations: Goals and measures. IEEE Transactions on Knowledge and Data Engineering, 36:5007–5024, 2023.

[6] Constantinos Daskalakis, Paul W. Goldberg, and Christos H. Papadimitriou. The complexity of computing a nash equilibrium. Electron. Colloquium Comput. Complex., TR05, 2006.

[7] Yanbo Fang, Tengfei Cao, and Yiming Zhang. Loss-convergence- driven federated learning with energy optimization: A stackelberg game approach. 2024 10th International Conference on Computer and Communications (ICCC), pages 2189–2193, 2024.

[8] Eva Garcia-Martin, Niklas Lavesson, Hakan Grahn, Emiliano Casalicchio, and Veselka Boeva. Estimation of ˚ energy consumption in machine learning. Journal ofParallel and Distributed Computing, 134:75–88, 2019.

[9] Yang Han, Tasiu Muazu, Samuel Omaji, and Shiyu Miao. A federated learning-based selection and incentive system using blockchain technology. Pervasive Mob. Comput., 112:102091, 2025.

[10] Jiawen Kang, Zehui Xiong, Dusit Tao Niyato, Han Yu, Ying-Chang Liang, and Dong In Kim. Incentive design for efficient federated learning in mobile networks: A contract theory approach. 2019 IEEE VTS Asia Pacific Wireless Communications Symposium (APWCS), pages 1–5, 2019.

[11] Jared Kaplan, Sam McCandlish, T. J. Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeff Wu, and Dario Amodei. Scaling laws for neural language models. ArXiv, abs/2001.08361, 2020.

[12] Latif Ullah Khan, Nguyen H. Tran, Shashi Raj Pandey, Walid Saad, Zhu Han, Minh N. H. Nguyen, and Choong Seon Hong. Federated learning for edge networks: Resource optimization and incentive mechanism. IEEE Communications Magazine, 58:88–93, 2019.

[13] Alexandre Lacoste, Alexandra Sasha Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. ArXiv, abs/1910.09700, 2019.

[14] Tra Huong Thi Le, Nguyen Hoang Tran, Yan Kyaw Tun, Minh N. H. Nguyen, Shashi Raj Pandey, Zhu Han, and Choong Seon Hong. An incentive mechanism for federated learning in wireless cellular networks: An auction approach. IEEE Transactions on Wireless Communications, 20:4874–4887, 2020.

[15] Wei Yang Bryan Lim, Jer Shyuan Ng, Zehui Xiong, Jiangming Jin, Yang Zhang, Dusist Niyato, Cyril Leung, and Chunyan Miao. Decentralized edge intelligence: A dynamic resource allocation framework for hierarchical federated learning. IEEE Transactions on Parallel and Distributed Systems, 33:536–550, 2022.

[16] Xiao Lin, Ruolin Wu, Haibo Mei, and Kun Yang. A game incentive mechanism for energy efficient federated learning in computing power networks. Digital Communications and Networks, 10(6):1741–1747, 2024.

[17] Yang Liu, Hongsheng Wang, Mugen Peng, Jianfeng Guan, and Yu Wang. An incentive mechanism for privacypreserving crowdsensing via deep reinforcement learning. IEEE Internet ofThings Journal, 8:8616–8631, 2021.

[18] H. B. McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-¨ efficient learning of deep networks from decentralized data. In International Conference on Artificial Intelligence and Statistics, 2016.

[19] Jiajun Meng, Jing Chen, Dongfang Zhao, and Lin Liu. Federated learning and free-riding in a competitive market. ArXiv, abs/2410.12723, 2024.

[20] John F. Nash. Non-cooperative games. Classics in Game Theory, 1951.

[21] Tu Viet Nguyen, Nhan Duc Ho, Hieu Thien Hoang, Cuong Danh Do, and Kok-Seng Wong. Toward efficient hierarchical federated learning design over multi-hop wireless communications networks. IEEE Access, 10:111910– 111922, 2022.

[22] Martin J. Osborne and Ariel Rubinstein. A course in game theory. 1995.

[23] Theodora Panagea, Nikolaos Koursioumpas, Lina Magoula, and Ramin Khalili. Greenflag: A green agentic approach for energy-efficient federated learning. 2026.

[24] David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Llu´ıs-Miquel Mungu´ıa, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. ArXiv, abs/2104.10350, 2021.

[25] Kilian Pfeiffer, Martin Rapp, Ramin Khalili, and Jorg Henkel. Federated learning for computationally con- ¨ strained heterogeneous devices: A survey. ACM Computing Surveys, 55:1 – 27, 2023.

[26] Xinchi Qiu, Titouan Parcollet, Daniel J. Beutel, Taner Topal, Akhil Mathur, and Nicholas D. Lane. A first look into the carbon footprint of federated learning. ArXiv, abs/2102.07627, 2020.

[27] Ana Radovanovic, Ross Koningstein, Ian Schneider, Bokan Chen, Alexandre Nobrega Duarte, Binz Roy, Diyue Xiao, Maya Haridasan, Patrick Hung, Nick Care, Saurav Talukdar, E. Mullen, Kendal Smith, MariEllen Cottman, and Walfredo Cirne. Carbon-aware computing for datacenters. IEEE Transactions on Power Systems, 38:1270– 1280, 2021.

[28] Julia Jean Robinson. An iterative method of solving a game. Classics in Game Theory, 1951.

[29] Yunus Sarikaya and Ozgur Ercetin. Motivating workers in federated learning: A stackelberg game perspective. IEEE Networking Letters, 2:23–27, 2019.

[30] Stefano Savazzi, Monica Nicoli, Vittorio Rampa, and Sanaz Kianush. Federated learning with cooperating devices: A consensus approach for wireless iot networks. IEEE Transactions on Communications, 68(12):7644– 7659, 2020.

[31] Rachael Hwee Ling Sim, Yehong Zhang, Mun Choon Chan, Bryan Kian, and Hsiang Low. Collaborative ma chine learning with incentive-aware model rewards. In International Conference on Machine Learning, 2020.

[32] Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in nlp. ArXiv, abs/1906.02243, 2019.

[33] Dipanwita Thakur, Antonella Guzzo, Giancarlo Fortino, and Francesco Piccialli. Green federated learning: A new era of green aware ai. ACM Computing Surveys, 57:1 – 36, 2024.

[34] The EXIGENCE-project. Sustainable by Default: Driving Energy-Efficient Streaming and AI Services through Smart Incentives. [Online] Available: https://projectexigence.eu/sustainable-by-defau lt-driving-energy-efficient-streaming-and-ai-services-through-smart-incen tives/. Accessed: 12 Feb 2026.

[35] Konstantinos Varsos, Adamantia Stamou, George D. Stamoulis, and Vasillios A. Siris. Optimal energy-aware service management in future networks with a gamified incentives mechanism. 2026.

[36] Konstantinos Varsos, Adamantia Stamou, George D. Stamoulis, and Vasillios A. Siris. User acceptance model for smart incentives in sustainable video streaming towards 6g. ICC 2026 - IEEE International Conference on Communications, pages 1–6, 2026.

[37] Philipp Wiesner, Ramin Khalili, Dennis Grinwald, Pratik Agrawal, Lauritz Thamsen, and Odej Kao. Fedzero: Leveraging renewable excess energy in federated learning. Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems, 2023.

[38] Yufeng Zhan, Peng Li, Zhihao Qu, Deze Zeng, and Song Guo. A learning-based incentive mechanism for federated learning. IEEE Internet ofThings Journal, 7:6360–6368, 2020.

[39] Yuze Zou, Shaohan Feng, Dusit Tao Niyato, Yutao Jiao, Shimin Gong, and Wenqing Cheng. Mobile device training strategies in federated learning: An evolutionary game approach. 2019 International Conference on Internet of Things (iThings) and IEEE Green Computing and Communications (GreenCom) and IEEE Cyber, Physical and Social Computing (CPSCom) and IEEE Smart Data (SmartData), pages 874–879, 2019.

## A Additional Experiments

## Heuristic phase vs Nash equilibrium.

Now we consider the case where two agents are engaged in the game with three actions each, and they follow the heuristic phase, also we set $\gamma = 1 0$ . Therefore we have a $3 \times 3$ bimatrix game. In Figure 13, we compare the solution provided by the fictitious play algorithm 1 with the accurate solution of the game. The game has a Nash equilibrium with strategy profile $( ( 1 , \bar { 0 } , 0 \bar { ) } , ( \bar { 0 } , 0 , 1 ) )$ . We initialize the algorithm in the pure strategy profile ((0, 0, 1), (1, 0, 0)). We observed that very fast the fictitious play algorithm converges to Nash equilibrium of the game.

![](images/bfb0b1cc8434595a31bda07807480a27b642ac8c3fbb9cb98f8acae294b50c4f.jpg)

![](images/038fefb9607dbb67e6a43be77234b882f63047e9ae44271e69ea61fbfe3a5205.jpg)

![](images/385824a5bed890dec200ca2ef27d09c5f6a13b72f06d9f0e1aabab9991131a0f.jpg)

![](images/e219515e93f80ab8f6e0ad617efe54046dde9deb2576da9691c66ab83385e6b8.jpg)

![](images/d4e72bdfbc027f88feed268698c839b6611073ee0a0afe00411fc7f9c008996a.jpg)

![](images/69ffea04cf618ecd9fe6cb9c558df34cd281caf556ba07666f377c27d4f6036a.jpg)  
Figure 13: Energy flexibility: (i) $\gamma = 0 \mathrm { ( l e f t ) } .$ , (ii) $\gamma = 1 0$ (middle), and (iii) γ = 1000 (right).

## Rewards

Here, we consider a homogeneous population of agents, $\mathcal { N } = \{ 1 , 2 , 3 \}$ , s.t. $S _ { i } \sim U [ \mathcal { G } ^ { t } - \delta _ { - } , \mathcal { G } ^ { t } + \delta _ { + } ] ^ { 4 }$ , with $\delta _ { - } = \delta _ { + } = 1 0 , \alpha = 0 . 9 , h = 0 . 5 , \dot { \theta _ { i } } \sim U [ 0 , 1 ]$ , and $\gamma \in \{ 1 , 1 0 , 1 0 \bar { 0 } \}$

Figure 14 presents the evolution of rewards across training rounds, with three panels corresponding to $\gamma \in$ $\{ 1 , 1 0 , 1 0 0 \}$ . For $\gamma = 1$ , the rewards w.r.t. heuristic algorithm exhibit differences across agents. Agents 1 secures almost constant rewards, and agent 2 receives rewards partially. Further, they consume energy in all training rounds, see Figure 5. On the other hand, agent 3 takes zero rewards in the rounds where it does not participates in the training. Under the correlation mechanism, the rewards are generally higher, in total, than under the heuristic mechanism, reflecting the more coordinated participation pattern observed in the corresponding energy trajectories. For all $\gamma \in \{ 1 , 1 0 , 1 0 \bar { 0 } \}$ we observe that rewards follow the energy patterns in Figure 5.

![](images/d3b7e690a04ddfe29cc04ebbc8260d5fb77ce0021696ecfb47e2f8128cb52d5d.jpg)

![](images/82a1cdb6e6ec1cd2ea0f0bd2ef30a279f34aff1bb0681148db910ee45e877e92.jpg)

![](images/8bb2ac205d2f8aabb6e45861f3a0b99f9c089d1140419584e421a3be84c06918.jpg)  
Figure 14: Rewards and accuracy per agent: (i) γ = 1 (left), (ii) $\gamma = 1 0$ (middle), and (iii) $\gamma = 1 0 0 \mathrm { ( r i g h t ) }$

Comparing the two mechanisms, the correlation device generally assigns higher rewards in the considered experi ments, while the heuristic mechanism produces more differentiated reward profiles across agents. This difference is consistent with the respective participation structures: coordination tends to synchronize the agents’ decisions, whereas the heuristic mechanism allows agents to respond independently to their individual conditions. The reward results therefore complement the energy and accuracy experiments by showing that the two mechanisms do not only differ in their aggregate energy-consumption patterns, but also in how the available incentives are distributed across agents.

![](images/42df8a14ca2cd6a24f45fc9dfc647c36e727beb899fb720e26bcb5b8c9d946ba.jpg)

![](images/1a799b0fb95f9127075fc3066e27b5e686de5cd7fbe3b59c86f07243bd4bc466.jpg)

![](images/c8d72f4f97fd945a9464c5fa6d4dd556dcc9d4088ed8e94f05b44ce1904ed8e2.jpg)  
Figure 15: Rewards and accuracy per agent: (i) $h = 0 . 5 ( \mathrm { l e f t } )$ , (ii) $h = 0 . 7 5$ (middle), and (iii) h = 1 (right), when $\gamma = 1 0$

Finally, the reward trajectories should be interpreted jointly with the accuracy curves. In particular, a zero reward corresponds to a round in which an agent does not participate, and therefore does not imply poor learning performance by itself. Its effect on accuracy depends on the subsequent evolution of the local model and on the recovery parameter h, see Figure 15. With $h = 0 . 5$ , inactive agents only partially recover the accuracy lost through model drift when they return to training, so repeated periods of non-participation can affect the subsequent accuracy trajectory. This distinction is important because the reward experiment demonstrates the relationship between participation and incentives, whereas the learning consequences of those participation decisions are captured by the accuracy and drift experiments.

## Quality of data.

Here, we consider a homogeneous population of agents, $\mathcal { N } = \{ 1 , 2 , 3 \} , \mathrm { s . t . } \ S _ { i } \sim U [ \mathcal { G } ^ { t } - \delta _ { - } , \mathcal { G } ^ { t } + \delta _ { + } ] ^ { 4 }$ , with with $\delta _ { - } = \delta _ { + } = 1 0 , \alpha = 0 . 9 , h = 0 . 5$ , and $\gamma \in \{ 1 , 1 0 , 1 0 0 \}$ . We discuss three cases: (i) $\theta _ { i } \sim U [ 0 , 0 . 5 ]$ (relatively poor training data), (ii) $\theta _ { i } \sim U [ 0 . 2 5 , 0 . 7 5 ]$ (moderately heterogeneous data quality), and (iii) $\theta _ { i } \sim U [ 0 . 5 , 1 ]$ (relatively good training data).

Quality of data vs accuracy. In the right columns in Figures 16, 17, and 18 we illustrate the training performance of each agent for the three different data-quality regimes and two penalty values, $\gamma \in \{ 1 0 , 1 0 0 \}$ . We observe that the dispersion of agent performance is larger under the heuristic approach than under the correlation device. In the latter case, performances are more concentrated, which is consistent with the fact that agents with similar characteristics tend to follow more similar training behavior when coordination is imposed.

Figures 16 - 18 show that data quality primarily affects the speed of convergence, rather than the qualitative behavior of the two mechanisms. Under the heuristic approach, the agents generally reach higher final accuracy, with the improvement becoming faster and smoother as data quality increases. This is particularly evident in Figure 18, where the higher-quality local data allow the target accuracy to be approached with fewer training rounds. This observation is consistent with the previous experiments: when participation decisions are made independently, agents can adapt their training activity to their individual conditions, which generally results in higher accuracy but also greater dispersion across agents.

The correlation device produces more synchronized accuracy trajectories, as expected from the coordination of participation decisions. However, its relative performance varies with data quality. In particular, Figure 17 shows that the correlation mechanism performs best in the medium-quality regime, whereas the heuristic approach benefits most clearly from the high-quality data in Figure 18. We therefore do not observe a uniformly increasing performance advantage for the correlation mechanism as data quality improves. A possible explanation is that coordination imposes a more uniform participation pattern and may therefore prevent the mechanism from fully exploiting differences in the informativeness of individual updates. This interpretation is consistent with the earlier experiments, where coordina tion was found to produce more synchronized behavior, while decentralized decisions allowed agents to adapt more directly to their individual conditions. However, the experiments do not by themselves establish that this is the sole cause of the observed difference

![](images/62cccbe370e689cdc0e9d22878e17b53bfb685f5702b8a405b6075d0cc89eb70.jpg)

![](images/a247d6651a9b512be3e9e512643c8b0747fdf69411fa99fbb65a35371ddc6d7c.jpg)

![](images/679ab7b3538b3e17bff26d85846acd610da22b6a466fc47796950c5b4449161e.jpg)

![](images/7973618abd147b1e3bdff35f0883df50cb036ad4f3d1fd79b2778dec9a143e66.jpg)

![](images/843f7b42d655075feb0067251822265e0430bfe62967795539b3635bfbc8f8a7.jpg)

![](images/3d3d99c4b12be231d91b25ad653372d1f6f1938deb609d88baad7a932a7f21d8.jpg)

![](images/542fa5649c86eb5c33317cb96e0a6eb17ba6defe5802dbaa03a8ec8159a6e1ab.jpg)

![](images/f2ed3c4cda4e8c79149f5cf75cb60e2bfde4f32b386b5429abebf67c1efc65c0.jpg)

![](images/e01ba044688e12cfad686c240bb15848340178e5513a1c22f28633f87a1ede7c.jpg)  
Figure 16: Quality of data $\theta _ { i } \sim U [ 0 , 0 . 5 ] \colon \gamma = 1$ (upper row), $\gamma = 1 0$ (middle row), and $\gamma = 1 0 0$ (lower row).

Quality of data vs energy. The effect on energy consumption is considerably weaker. Across Figures 16, 17, and 18, the heuristic approach generally consumes more energy than the correlation mechanism, while the overall trajectories remain qualitatively similar across the three data-quality regimes. This is consistent with the previous energy experiments, where energy consumption was primarily determined by the agents’ participation decisions and the available renewable energy. Here, data quality affects the learning outcome more directly than the energy allocation, since energy is constrained through the equilibrium incentives rather than being explicitly optimized as a function of $\theta _ { i } .$ Thus, within the considered settings, improving data quality mainly translates into faster and more stable learning, while its effect on total energy consumption remains comparatively limited.

Overall, Figures 16 - 18 indicate that data quality and participation incentives interact primarily through the learning dynamics. Higher-quality data allow the heuristic mechanism to exploit individual contributions more effectively, whereas coordinated participation can produce different outcomes depending on the data-quality regime. This complements the earlier observations on energy heterogeneity and model drift: the quality of the available updates affects how valuable continued participation is, while the incentive mechanism determines how that participation is distributed across training rounds. The experiments therefore support an effect of data quality on convergence and accuracy, but they do not justify concluding that higher data quality necessarily reduces energy consumption or that one participation mechanism is universally preferable.

![](images/6572e6e77d6baf7657a2bd1fc0b88f52ea961c268b6273c6fd6a8f3c73ede60b.jpg)

![](images/574e65c3d3fc468d670e928b0cc7c80ecd18ce0b9cfd6ab12883a697fb135d0a.jpg)

![](images/7406be20baf2adadd8e73687da91bc7316643789610981bceb3052b342c08027.jpg)

![](images/7ba31854a8927c6fe6401b4e99f6be80b0b35041c0cde45b4456087c508b4895.jpg)

![](images/7745b2ca801b7d861cfd2bac73f0ee4c82026f059403e406eefea48e75d4e128.jpg)

![](images/a20424d3af45c1ba762be314dea63945d0eecd10a9155da4e1b985dc7e5891f4.jpg)

![](images/f8285c61855c279494d8e076dadbf169f205d1dfd7b0a584afa55b0c90d4536f.jpg)

![](images/fc90f2a76824088f436a4801e11ab316fea8886cbc1690e5563907f4d6f359f9.jpg)  
Figure 17: Quality of data θ<sub>i</sub> ∼ U[0.25, 0.75]: γ = 1 (upper row), $\gamma = 1 0$ (middle row), and $\gamma = 1 0 0$ (lower row).

![](images/999811c12707b3379aa062e9720372ca98244dec3890c03f058b243e74219e60.jpg)

![](images/7c84fc2d28ba96247711979e2aee3dcd99ee7fb21d5e2310c6398ea4a4053fb5.jpg)

![](images/06ee34fd6f0216dc94c503a19e966130eaee3a069cba8b085f12e4bf13443bfe.jpg)

![](images/fcee0b3c8cae8efc37cff43d4e26a7e60b3b730f3f653a39d76f1e3b95bfff3d.jpg)

![](images/bc124ac3cefcb3477679c215647b5ab0a47a5c361675b47bf2b8c31b30436f93.jpg)

![](images/7a9dbf359da9367db88a8e007aebe65928e141cd161762f0d252f8fdb852e5c0.jpg)

![](images/2948335ef4bc13cecc3f994e7ea845a7a856f25abe851cb554b7b17e462c3da4.jpg)

![](images/de4c57165fcbf133baea3b3c05bd98ba44da8236c2daef2379b34f8a575d23bf.jpg)

![](images/ecf1e07b0ec19d850bfa07ca193a153d346b8a71ef3d01056dcbb144972c7d42.jpg)

![](images/d2f8e84effbe58d601579f08bda652cbfcd67110bec80087dd3d09899e53b64e.jpg)  
Figure 18: Quality of data $\theta _ { i } \sim U [ 0 . 5 ,$ 1]: γ = 1 (upper row), γ = 10 (middle row), and $\gamma = 1 0 0$ (lower row).