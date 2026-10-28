# Hyperparameter and Neural Architecture Search via Evolutionary Optimization for Reinforcement Learning: HPNAS-RL

Nihal Acharya Adde<sup>[0009−0002−0966−0825]1,2\*</sup> and Hanno Gottschalk<sup>[0000−0003−2167−2028]2</sup>

<sup>1\*</sup>Volkswagen Group Innovation, Volkswagen AG, Berliner Ring 2, Wolfsburg, 38440, Lower Saxony, Germany. <sup>2</sup>Institute of Mathematics, TU Berlin, Straße des 17. Juni 136, Berlin, 10623, Berlin, Germany.

\*Corresponding author(s). E-mail(s): nihal.acharya.adde@campus.tu-berlin.de, nihal.acharya.adde@volkswagen.de; Contributing authors: gottschalk@math.tu-berlin.de;

## Abstract

In this study, we propose a dual-branch evolutionary framework that simultaneously performs Neural Architecture Search (NAS) and Hyperparameter Optimization (HPO) for Reinforcement Learning (RL) in Autonomous Driving (AD) tasks. While NAS has shown strong potential in automating architecture design for Convolutional Neural Networks (CNNs), its application to RL remains limited, especially in scenarios that require joint optimization of architectural and training dynamics. Building on our prior work on Evolutionary Multi-Objective NAS (EMNAS), the proposed framework evolves discrete network architectures and continuous hyperparameters in parallel under a unified multi-objective search strategy. In this setup, NAS targets the convolutional layers of the RL policy network - responsible for visual feature extraction, while HPO focuses on optimizing critical RL hyperparameters. To accelerate convergence and improve generalization, we incorporate parallel evaluation and a teacher-student policy transfer mechanism. Experiments in a high-fidelity driving simulator show that the joint NAS-HPO framework consistently outperforms standalone NAS, yielding policies with higher rewards, fewer parameters, and improved stability. These results highlight the benefits of jointly optimizing model structure and training dynamics for scalable and efective RL agent design.

Keywords: Multi-Objective Evolutionary Algorithm, Neural Architecture Search, Hyperparameter Optimization, Reinforcement Learning

## 1 Introduction

Reinforcement Learning (RL) has emerged as a powerful framework for Autonomous Driving (AD), enabling agents to learn sophisticated decision-making and control policies in complex environments. However, designing efective RL agents requires carefully crafting both the neural network architecture and the associated training hyperparameters, which together determine the agent’s performance, sample eficiency, and computational cost. Traditionally, these design decisions are made manually, relying on expert intuition and extensive trial-and-error tuning, which are time-consuming and often suboptimal.

To automate and scale this process, we previously introduced the Evolutionary Multi-Objective Neural Architecture Search (EMNAS) framework [1], building on EEEANet [2]. EMNAS employs a Genetic Algorithm (GA) to evolve compact convolutional architectures for the policy network of a Proximal Policy Optimization (PPO) based RL agent. By integrating early-exit evaluation, Optimized Transfer Learning (OTL), and parallelized training, EMNAS demonstrated superior performance over manually crafted models [1]. However, its search space was limited solely to architectural configurations, neglecting the significant impact of RL hyperparameters on RL convergence and stability.

In this paper, we extend EMNAS into a dual-branch evolutionary framework that jointly optimizes neural network architectures and training hyperparameters. Hyperparameter and Neural Architecture Search (HPNAS) framework combines discrete architecture evolution with continuous hyperparameter tuning using polynomial mutation and Simulated Binary Crossover (SBX), enabling coordinated search across both spaces. Neural Architecture Search (NAS) targets the convolutional layers of the RL policy network, which are responsible for visual feature extraction, while Hyperparameter Optimization (HPO) tunes RL hyperparameters such as the learning rate, batch size, discount factor and time horizon. Initial experiments demonstrate that the HPNAS strategy yields consistent improvements in cumulative reward, model compactness, and training stability over architecture-only baselines. A Bayesian surrogate optimization model is being developed to learn the relationship between architecture and hyperparameter configurations and their performance. Although not yet integrated in this work, it is designed to guide later stage search by predicting promising configurations and is discussed further in the outlook.

## 1.1 Autonomous Driving Task:

The AD task [3] is performed in a simulated environment built using the Unity 3D engine [4], where multiple agents are trained with the Proximal Policy Optimization (PPO) algorithm [5]. Agents navigate a three-lane oval track with the goal of learning smooth, collision-free driving while avoiding abrupt maneuvers. Each agent receives semantic segmentation images from a forward-facing camera and predicts future trajectory points, which are executed by an external black-box vehicle controller responsible for throttle, brake, and steering commands. The reward function promotes accurate trajectory prediction, penalizes collisions and lane deviations, and encourages stable motion with minimal jerk.

Figure 1 illustrates the simulated Unity3D environment used for agent training. This setup provides a realistic yet controlled platform for evaluating reinforcement learning strategies and their generalization across training runs. In this work, we focus on optimizing the convolutional network architecture and PPO hyperparameters underlying the policy to enhance driving performance and training eficiency in the RL-based AD task.

![](images/be705526ee55c9433d3f859cd92e689664dc5f3a96cd99148ecc261bffb1f7fe.jpg)  
Fig. 1: Simulated three-lane track used for RL-based AD training in Unity3D. Top left: Drone view of road with agent. Top right: Front camera view, later converted to a semantic segmentation image and used as the network input. Bottom: Driving behavior based on predicted trajectory points.

## 2 Related Work

NAS has evolved through RL-based search [6], gradient-based optimization [7], and Evolutionary Algorithms (EAs) [8]. While gradient-based methods such as DARTS [7] improve eficiency by making architecture selection diferentiable through continuous weighting of candidate operations, they often converge to locally optimal designs and are restricted to diferentiable objectives. RL-based NAS frameworks [6] have shown promising results but require extensive computational resources due to sequential policy evaluations. In contrast, EAs provide a more flexible search paradigm capable of exploring large, non-diferentiable spaces while maintaining population diversity, making them well-suited for complex architecture optimization problems. Early meth ods such as NeuroEvolution of Augmenting Topologies (NEAT) [9] demonstrated the ability of genetic algorithms to evolve both network topology and parameters. Later studies, including AmoebaNet [8], showed that evolutionary search can outperform reinforcement learning and random search strategies in large-scale vision tasks. Building on these ideas, Multi-Objective Evolutionary Algorithms (MOEAs) were introduced to balance accuracy, computational cost, and network size. Examples include NSGA-Net [10], which generates a Pareto front of eficient architectures, and Early Exit Evolutionary Neural Architecture Search (EEEANet) [2], which combines MOEAs with early-exit initialization using the NSGA-II algorithm [11] to improve convergence. Despite their efectiveness, most EA and MOEA-based NAS frameworks remain confined to supervised learning, ofering limited scalability and adaptability to reinforcement learning tasks where sample eficiency and stability are critical.

In recent years, researchers have recognized that architectural choices and hyperparameter settings are interdependent, motivating the development of joint optimization approaches that integrate NAS with HPO [12, 13]. Such methods have shown that jointly tuning both components can lead to more stable and higher-performing models, as hyperparameters directly influence architecture behavior during training. Bayesian optimization-based strategies [14] and weight-sharing techniques [15] have been proposed to accelerate search by reusing knowledge across configurations. Evolutionary approaches for joint NAS–HPO have also been explored in convolutional network settings [16, 17]. These methods demonstrated the potential of integrating architectural and hyperparameter optimization, using evolutionary multi-objective strategies such as non-dominated sorting and tournament selection to balance competing objectives. However, they were primarily designed for static supervised learning tasks, where evaluations are deterministic and data distributions remain fixed. Such frameworks do not address the unique challenges of reinforcement learning, including stochastic rewards, non-stationary environments, and the need for scalable exploration across parallel agents.

In this context, our work introduces a dual-branch evolutionary framework that jointly optimizes architectures and RL-specific hyperparameters. By combining a Directed Acyclic Graph (DAG)-based NASNet-style representation [18] with parallel population training and transfer learning mechanisms, the proposed approach achieves scalable and sample-eficient optimization in high-dimensional RL environments, bridging the gap between HPO, NAS and RL.

## 3 HPNAS Methodology

The proposed HPNAS framework extends our earlier EMNAS system [1] by introducing a dual-branch evolutionary approach that jointly optimizes neural architectures and reinforcement learning hyperparameters. As illustrated in Sections 3.1 and 3.2, the framework comprises two coordinated components: the EMNAS-based architecture optimization branch, which evolves convolutional network structures, and the hyperparameter optimization branch, which tunes PPO-related parameters such as learning rate, batch size, and discount factor. Both branches operate within a unified multiobjective evolutionary process, sharing fitness evaluations to jointly enhance policy performance and model eficiency.

## 3.1 EMNAS Framework

Our previous work, EMNAS [1], builds upon the EEEANet framework [2] to optimize convolutional neural architectures for RL tasks based on the PPO algorithm. Using a multi-objective genetic algorithm, EMNAS evolves candidate architectures to simultaneously maximize PPO cumulative reward while minimizing computational cost and model complexity. The optimization problem is formulated as:

$$
f ( x ) = \operatorname* { m i n } \left\{ - \mathbf { R e w a r d } ( x ) , \mathbf { F L O P S } ( x ) , \mathbf { P a r a m s } ( x ) \right\}\tag{1}
$$

where Reward(x) denotes the PPO training reward, FLOPS(x) represents the floating-point operations per second, and Params(x) indicates the number of learnable parameters in the network. All objectives are minimized to identify models that achieve a favorable balance between accuracy and eficiency. An overview of the EMNAS pipeline, including the genotype encoding and evolutionary process, is shown in Figure 2.

![](images/3f5ac2688b2163a867ea5be6d169c60aa0d8c930964746ac2aafe29160f562e8.jpg)  
Fig. 2: Overview of the EMNAS pipeline. Multi-objective evolutionary optimization combines Early-Exit Population Initialization (EEPI) and Optimized Transfer Learning (OTL) for eficient RL architecture discovery.

## 3.1.1 Genotype Encoding Scheme

Architectures in EMNAS are represented as chromosomes composed of alternating normal and reduction cells [1, 18]. A normal cell preserves the spatial resolution of the input feature map, while a reduction cell halves it to enable hierarchical feature extraction. Each cell consists of several genes, defined as $L A _ { 1 } L A _ { 2 } , L B _ { 1 } L B _ { 2 } , L C _ { 1 } L C _ { 2 } .$ and $L D _ { 1 } L D _ { 2 }$ , where each gene specifies a convolutional operation L and its corresponding connection indices $( A , B , C , D )$

$$
\mathrm { c h r o m o s o m e } ( x ) = L A _ { 1 } L A _ { 2 } , L B _ { 1 } L B _ { 2 } , L C _ { 1 } L C _ { 2 } , L D _ { 1 } L D _ { 2 }\tag{2}
$$

Here, L represents the operator selected from the predefined search space, while indices $( A , B , C , D )$ determine the connectivity pattern between feature maps. The operation set includes depth-wise separable, dilated, and inverted convolutions, pooling operations, and skip connections as shown in Table 1.

Table 1: Search space used for normal and reduction cells in EMNAS.
<table><tr><td>Operation Type</td><td>Details</td></tr><tr><td>Convolutional Operations</td><td>3×3 depth-wise separable convolution 5×5 depth-wise separable convolution</td></tr><tr><td></td><td>3×3 dilated convolution</td></tr><tr><td></td><td>5×5 dilated convolution</td></tr><tr><td></td><td>3×3 inverted convolution</td></tr><tr><td></td><td>5×5 inverted convolution</td></tr><tr><td></td><td>7×7 convolution</td></tr><tr><td>Pooling Operations</td><td>Max pooling</td></tr><tr><td></td><td>Average pooling</td></tr><tr><td>Other Operations</td><td>Skip connection</td></tr></table>

The connections of each cell are defined according to the index positions $A , B , C , D$ which indicate where the outputs of previous operations are concatenated. For instance, assuming nine operations indexed as [0, 1, 2, 3, 4, 5, 6, 7, 8], a gene $L A _ { 1 } L A _ { 2 } =$ 30, 80 implies that outputs from operations 3 and 8 are connected at index 0, while $L B _ { 1 } L B _ { 2 } = 0 1 , 2 1$ connects operations 0 and 2 at index 1. Genes $L C _ { 1 } L C _ { 2 }$ and $L D _ { 1 } L D _ { 2 }$ connect sequentially to outputs of earlier nodes to maintain acyclic connectivity and reduce redundant parameters.

The index of each connection is computed recursively as:

$$
\mathrm { P r e v i o u s I n d e x } = \mathrm { i n d e x } - 2\tag{3}
$$

The PreviousIndex defines the node from which the current operation receives its input, ensuring that connections always originate from earlier nodes. For example, a gene $L C _ { 1 } L C _ { 2 } = 1 2 , 0 2 \ ( \mathrm { i n d e x } = 2 )$ connects from index 0, while $L D _ { 1 } L D _ { 2 } = 6 3 , 7 3$ $\mathrm { ( i n d e x = 3 ) }$ connects from index 1. If diferent indices exist within a gene, such as $L D _ { 1 } L D _ { 2 } = 6 3 , 7 2$ , operation 6 connects from index 1 and operation 7 from index 0. This hierarchical indexing maintains directed acyclic structure and enforces valid feature flow within each cell.

This compact encoding jointly captures both the architectural structure and interlayer dependencies, enabling EMNAS to perform eficient, flexible search over diverse CNN cell configurations.

## 3.1.2 Multi-objective Evolutionary Algorithm

The optimization of CNN architectures in EMNAS is formulated as a multi-objective problem, aiming to maximize reinforcement learning performance while minimizing

computational cost and model complexity:

$$
\begin{array} { r l } { \operatorname* { m i n } } & { { } f _ { 1 } ( x ) , f _ { 2 } ( x ) , \ldots , f _ { k } ( x ) } \\ { \mathrm { s . t . } } & { { } x \in X } \end{array}\tag{4}
$$

In EMNAS, $k = 3$ , and the objective vector is defined as in Equation 1. A population of candidate architectures is evolved over multiple generations, where each individual is trained and evaluated using the defined objectives. Fitness values are then ranked using the NSGA-II algorithm [11], which applies non-dominated sorting and crowding distance to preserve diversity among Pareto-optimal solutions. This enables the search to discover a balanced set of architectures that achieve high PPO reward while remaining lightweight and computationally eficient. Equal weighting is applied to all objectives to ensure fair comparison across the trade-of surface.

## 3.1.3 Genetic Operations

Each architecture in EMNAS is represented as a Directed Acyclic Graph (DAG) defined by $G = ( V , E )$ , where V denotes the set of nodes corresponding to convolutional operators and E represents the directed edges encoding their input connections. Ofspring are generated through crossover, mutation, and survival operations, which introduce diversity while preserving the structural validity of the architecture.

## Crossover:

Uniform crossover combines two parent architectures $G _ { 1 } = ( V _ { 1 } , E _ { 1 } )$ and $G _ { 2 } = ( V _ { 2 } , E _ { 2 } )$ to form a child $G _ { c } = ( V _ { c } , E _ { c } )$ . Each gene $( v _ { i } , e _ { i } )$ is inherited from either parent with probability $p _ { c } = 0 . 5 \colon$

$$
( v _ { i } ^ { c } , e _ { i } ^ { c } ) = \left\{ { \begin{array} { l l } { ( v _ { i } ^ { 1 } , e _ { i } ^ { 1 } ) , } & { { \mathrm { i f ~ } } r _ { i } < p _ { c } , } \\ { ( v _ { i } ^ { 2 } , e _ { i } ^ { 2 } ) , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{5}
$$

where $r _ { i } \sim U ( 0 , 1 )$ . This operation promotes genetic mixing by combining convolutional operations and their input connections from both parents, generating ofspring that inherit key structural traits while introducing novel layer combinations.

## Mutation:

With probability $p _ { m } ,$ a randomly selected gene $( v _ { i } , e _ { i } )$ is replaced by $( v _ { i } ^ { \prime } , e _ { i } ^ { \prime } )$ drawn from the predefined index and operation sets:

$$
( v _ { i } ^ { \prime } , e _ { i } ^ { \prime } ) = \left\{ \begin{array} { l l } { \mathrm { R a n d o m } ( V , E ) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } p _ { m } , } \\ { ( v _ { i } , e _ { i } ) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6}
$$

Additionally, an inversion mutation with probability $p _ { i n v }$ reverses the order of a randomly selected subgraph while maintaining the DAG structure. Typical values are $p _ { m } = 0 . 1$ 1 and $p _ { i n v } = 0 . 1$ 1. Mutation introduces controlled variability in the population, enhancing exploration while preventing premature convergence.

## Survival:

A survival mechanism ensures that the best-performing architectures from the current generation are retained in the next. This strategy preserves strong solutions to maintain knowledge continuity while allowing new ofspring to introduce diversity. By combining exploitation of high-fitness models with exploration of novel candidates, the population converges more efectively toward the Pareto-optimal frontier.

## Early Exit Parameter Initialization (EEPI):

EMNAS applies a maximum parameter constraint $\beta _ { \mathrm { m a x } }$ during initialization to avoid oversized models:

$$
\operatorname { E E P I } ( \theta ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } \beta \leq \beta _ { \operatorname* { m a x } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{7}
$$

Architectures exceeding this limit are discarded and reinitialized, which improves early-stage eficiency by reducing computational overhead and restricting the search space to feasible model configurations.

## 3.1.4 Optimized Transfer Learning (OTL)

To improve sample eficiency and ensure stable convergence across generations, EMNAS employs an Optimized Transfer Learning (OTL) mechanism. In this approach, student architectures inherit weights and policies from the best-performing model of the previous generation through behavior cloning [19], allowing new populations to begin training from a well-initialized policy rather than from scratch.

At the end of each generation, expert trajectories $\mathcal { D } _ { E } = \{ ( s _ { t } , a _ { t } ) \} _ { t = 1 } ^ { T }$ are collected from the top policy $\pi _ { E } ( a \mid s )$ , where $s _ { t } \in S$ is the state (or observation) at time $t ,$ $a _ { t } \in \mathcal A$ is the action, and T is the trajectory length. Each architecture in the next generation is initialized by supervised imitation of this expert using the behavior cloning loss:

$$
\mathcal { L } _ { B C } ( \theta ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } _ { E } } \left[ - \log \pi _ { \theta } ( a \vert s ) \right]\tag{8}
$$

This ensures that both new ofspring and surviving individuals begin PPO training from a consistent policy baseline, maintaining a fair optimization trajectory across the population. OTL also retains the optimizer state from the previous generation, including the final learning rate and PPO clipping value, ensuring a smooth continuation of training. By reusing expert data and policy initialization, OTL reduces PPO training time while maintaining stability and fairness across evolving architectures.

In summary, EMNAS combines a compact genotype encoding, NSGA-II based multi-objective optimization, and eficiency mechanisms (EEPI and OTL) to discover CNN architectures that jointly optimize RL performance and computational cost. This forms the basis for the dual-branch HPNAS extension.

## 3.2 Dual-Branch Extension: Joint NAS and HPO

While EMNAS originally targeted architecture search, RL performance is also strongly influenced by hyperparameter choices. We therefore extend EMNAS into a dual-branch framework that jointly evolves architectures and hyperparameters using a coupled genetic algorithm.

## 3.2.1 Hyperparameter Search Space

Based on the hyperparameter optimization and sensitivity analysis presented in Chapter ??[3], we initially considered six RL hyperparameters for optimization. To reduce the dimensionality and computational cost of the joint search, PPO epochs and the entropy coeficient were fixed at 3 and 0.01, respectively. The remaining four: batch size, discount factor, learning rate, and time horizon form the HPO search space in this work, as they contributed most to performance variation [3].The final set of selected hyperparameters and their corresponding ranges are summarized in Table 2. The chosen ranges are informed by previous studies in PPO-based reinforcement learning [5, 20] and empirical observations in AD environments, ensuring coverage of commonly efective values while maintaining search feasibility.

Table 2: Tuned Hyperparameters and Ranges
<table><tr><td>Hyperparameter</td><td>Range</td><td>Description</td></tr><tr><td>Batch Size</td><td> $\overline { { 6 4 - 5 1 2 } }$ </td><td>Number of samples per gradient update</td></tr><tr><td>Discount Factor (γ)</td><td> $0 . 9 5 - 0 . 9 9 9$ </td><td>Weighting of future rewards</td></tr><tr><td>Learning Rate</td><td> $1 0 ^ { - 5 } - 1 0 ^ { - 2 }$ </td><td>Step size for gradient updates</td></tr><tr><td>Time Horizon</td><td> $3 2 - 2 5 6$ </td><td>Number of steps per trajectory segment</td></tr></table>

Our dual-branch framework jointly evolves network architectures and hyperparameters, employing a discrete encoding for NAS and a continuous representation for HPO. Both branches are evaluated using the same multi-objective fitness function as described in EMNAS (Section 3.1), which balances reward, FLOPs, and parameter count. In the initial generation, hyperparameters are uniformly sampled within predefined ranges (see Table 2). From the second generation onward, the best-performing configuration is retained through a survival mechanism, and from the remaining population, ofspring are generated using crossover or mutation with equal probability (0.5 each). To maintain diversity, 10% of new individuals are sampled purely at random.

## 3.2.2 Crossover and Mutation Mechanisms

The continuous hyperparameter space is evolved using Simulated Binary Crossover (SBX) and polynomial mutation, well established operators in multi-objective evolutionary optimization due to their efectiveness in balancing exploration and exploitation [11, 21].

## Polynomial mutation:

Polynomial mutation introduces small stochastic perturbations to continuous parameters, maintaining a balance between local refinement and global exploration. For each hyperparameter $h _ { i }$ , the mutated value $h _ { i } ^ { \prime }$ is computed as:

$$
h _ { i } ^ { \prime } = h _ { i } + \delta _ { i } \big ( h _ { i } ^ { \operatorname* { m a x } } - h _ { i } ^ { \operatorname* { m i n } } \big ) ,\tag{9}
$$

where $\delta _ { i }$ is a mutation ofset sampled from a polynomial probability distribution [21]:

$$
\delta _ { i } = \left\{ { \begin{array} { l l } { ( 2 u ) ^ { \frac { 1 } { \eta _ { m } + 1 } } - 1 , } & { { \mathrm { i f ~ } } u < 0 . 5 , } \\ { 1 - [ 2 ( 1 - u ) ] ^ { \frac { 1 } { \eta _ { m } + 1 } } , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{10}
$$

with $u \sim \mathcal { U } ( 0 , 1 )$ being a uniformly distributed random number, and $\eta _ { m }$ representing the mutation distribution index.

A larger $\eta _ { m }$ produces smaller perturbations centered around the parent value $h _ { i }$ promoting fine-grained local search, whereas a smaller $\eta _ { m }$ increases the likelihood of larger deviations, enhancing exploration. The scaling term $( h _ { i } ^ { \operatorname* { m a x } } - h _ { i } ^ { \operatorname* { m i n } } )$ ensures that the mutation magnitude is proportional to the allowable range of the hyperparameter. After mutation, values are typically clipped to the interval $[ h _ { i } ^ { \operatorname* { m i n } } , h _ { i } ^ { \operatorname* { m a x } } ]$ to maintain feasibility. This formulation makes polynomial mutation particularly efective for finetuning continuous design variables in evolutionary algorithms.

## Simulated Binary Crossover (SBX):

Simulated Binary Crossover (SBX) is a widely used recombination operator for continuous variables that mimics the behavior of single-point crossover in binary-coded genetic algorithms. For each hyperparameter $h _ { i }$ , SBX generates two ofspring $h _ { i } ^ { \prime }$ and $h _ { i } ^ { \prime \prime }$ from two parent values $h _ { i } ^ { 1 }$ and $h _ { i } ^ { 2 }$ as:

$$
\begin{array} { r } { h _ { i } ^ { \prime } = 0 . 5 \left[ ( 1 + \beta _ { i } ) h _ { i } ^ { 1 } + ( 1 - \beta _ { i } ) h _ { i } ^ { 2 } \right] , } \\ { h _ { i } ^ { \prime \prime } = 0 . 5 \left[ ( 1 - \beta _ { i } ) h _ { i } ^ { 1 } + ( 1 + \beta _ { i } ) h _ { i } ^ { 2 } \right] , } \end{array}\tag{11}
$$

where $\beta _ { i }$ is a spread factor sampled according to the SBX probability distribution [21]:

$$
\beta _ { i } = \left\{ \begin{array} { l l } { ( 2 u ) ^ { \frac { 1 } { \eta _ { c } + 1 } } , } & { \mathrm { i f ~ } u \leq 0 . 5 , } \\ { \left[ \frac { 1 } { 2 ( 1 - u ) } \right] ^ { \frac { 1 } { \eta _ { c } + 1 } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{12}
$$

with $u \sim \mathcal { U } ( 0 , 1 )$ being a uniformly distributed random number, and $\eta _ { c }$ denoting the crossover distribution index.

A larger $\eta _ { c }$ produces ofspring closer to the parents, promoting exploitation, while a smaller $\eta _ { c }$ allows for a wider spread of solutions, enhancing exploration. The symmetric nature of SBX ensures that both ofspring inherit information from both parents, maintaining diversity while preserving continuity in the search space. This property makes SBX particularly efective for real-coded evolutionary algorithms applied to hyperparameter optimization, where smooth variation between parent configurations enables eficient exploration of the search space.

For the architecture branch, we retain EMNAS’s discrete mutation and crossover [1]. Together, these operators enable eficient joint search over architecture and training dynamics, supporting scalable and efective RL agent discovery.

## 4 Experimental Setup

We evaluate the proposed dual-branch evolutionary framework across multiple autonomous driving experiments, designed to test its efectiveness in jointly optimizing architectures and hyperparameters under resource constrained RL settings. The experiments are conducted in the Unity3D simulation environment described in Section 1.1, where multiple agents are trained using the PPO algorithm. Each candidate is evaluated with a multi-objective fitness that balances policy reward, FLOPs, and parameter count, consistent with EMNAS [1].

The evolutionary algorithm employs a crossover probability of 0.5 and a mutation probability of 0.1 for the architecture branch, while the hyperparameter branch uses a higher mutation probability of 0.5 to encourage exploration in continuous search spaces. A survival rate of 20% (up to four top-performing individuals retained per generation) is applied across both branches. To maintain diversity, 10% of ofspring are generated through uniform random sampling. We schedule both the crossover and mutation probabilities, and the per-generation distribution index of crossover and mutation, to shift from exploration in early generations to exploitation in later ones. Candidate policies are trained and evaluated under low-fidelity settings to reduce computational cost, including 40 PPO epochs, input resolution of 84×84×3, and a network depth of 8. The depth corresponds to eight stacked cells alternating between normal and reduction cells, where the discovered architectures define the internal operations and connections within each cell type. These cells are then repeated to construct the final policy network. As we employ relative ranking within each generation, evaluation bias is reduced despite the low-fidelity settings. The architecture search is bounded using an EEPI threshold of 5 million parameters [1]. As in prior work, we employ an optimized transfer learning strategy where student models are initialized via behavior cloning [19] from the best policy of the previous generation [1].

We run six experiments with the following (generations, population size) settings: (6, 6), (10, 4), (10, 10), (10, 15), (15, 15), (15, 20). All evaluations are parallelized across NVIDIA Tesla H100 GPUs. We benchmark against the original EMNAS baseline [1] and do not compare against the standalone hyperparameter tuning baseline from previous work [3], as those models were trained for the full number of RL epochs and are incompatible with the low-fidelity setting used here. Finally, the winning architecture discovered through the dual-branch search is fully trained and compared with the results of the NAS-only and HPO-only baselines to evaluate the overall performance gains achieved by joint optimization.

## 5 Experimental Results

Table 3 summarizes the performance of the proposed HPNAS framework compared to our previous EMNAS approach across six experiments with varying generation and population settings. The results are reported in terms of final PPO cumulative reward, total parameter count (in millions) and FLOPs (gigaFLOPs). All reported rewards correspond to the final cumulative episode return achieved by the best-performing configuration in each experiment.

Table 3: Comparison of EMNAS and HPNAS across multiple experiments
<table><tr><td></td><td></td><td colspan="2">Reward</td><td colspan="2">Params (M)</td><td colspan="2">FLOPs (G)</td></tr><tr><td>Exp</td><td>Gen / Pop</td><td>NAS</td><td>HPNAS</td><td>NAS</td><td>HPNAS</td><td>NAS</td><td>HPNAS</td></tr><tr><td>1</td><td>6/6</td><td>538</td><td>233</td><td>1.29</td><td>1.27</td><td>1.26</td><td>1.25</td></tr><tr><td>2</td><td>10 /  4</td><td>585</td><td>415</td><td>1.34</td><td>0.94</td><td>1.32</td><td>0.93</td></tr><tr><td>3</td><td>10 / 10</td><td>569</td><td>621</td><td>0.94</td><td>0.87</td><td>0.97</td><td>0.84</td></tr><tr><td>4</td><td>10 / 15</td><td>753</td><td>745</td><td>1.45</td><td>0.77</td><td>1.44</td><td>0.76</td></tr><tr><td>5</td><td>15 / 15</td><td>584</td><td>682</td><td>0.60</td><td>0.84</td><td>0.52</td><td>0.82</td></tr><tr><td>6</td><td>15 / 20</td><td>604</td><td>787</td><td>0.51</td><td>0.53</td><td>0.42</td><td>0.44</td></tr></table>

We observe that in smaller scale experiments (e.g., Exp 1 and 2), results are more scattered for the combined approach. This is likely due to the limited number of generations despite the larger HPNAS search space, which makes convergence more dificult. In some cases, NAS achieves higher reward, likely due to early discovery of promising architectures. However, these often lead to larger models, suggesting overparameterization when hyperparameters are not co-optimized. For instance, in Exp. 4, NAS achieves a reward of 753 but with 1.45M parameters and 1.44 GFLOPs, whereas HPNAS achieves a comparable reward of 745 with only 0.77M parameters and 0.76 GFLOPs, indicating that HPNAS finds more compact alternatives with a minimal trade-of in reward.

As the search budget increases (Exp 5 and 6), HPNAS consistently outperforms NAS in reward. While the resulting models are not always smaller, they remain comparable in size, achieving similar or slightly higher parameter counts and FLOPs. These results suggest that the HPNAS framework becomes more efective with larger populations and longer searches, better exploiting the interaction between architecture and training dynamics. This demonstrates the value of integrating HPO directly into the NAS loop, particularly in high-dimensional RL tasks.

Figure 3 shows how reward and parameter count evolve over generations in two larger experiments (15 generations with populations of 15 and 20). Since parameter count and FLOPs exhibit strong correlation across all runs, we omit FLOPs for clarity and use a 2D plot to better highlight the trade-of between performance and model eficiency. Each point represents a model, with color indicating generation (light to dark) and method (blue for NAS, red for HPNAS). Over time, HPNAS increasingly occupies the upper-left region, indicating higher rewards and smaller models. These trends highlight HPNAS’s stronger exploration in early stages and its ability to converge to more eficient solutions as the search progresses.

Figure 4 shows the evolution of four key hyperparameters across 15 generations in Experiment 6 with a population size of 20. Each subplot plots sampled hyperparameter values: batch size, time horizon, learning rate, and discount factor against their corresponding PPO rewards. Color shading reflects the generation in which the sample was evaluated, from light (early) to dark (later). The distribution shows how the search increasingly concentrates around high-performing regions. The optimal configuration converged to: batch size of 216, time horizon of 88, learning rate of $2 . 8 \times 1 0 ^ { - 3 }$ 2 and discount factor of 0.976, indicating the search efectively refined hyperparameters toward performance-optimal zones.

![](images/ee7b38e3501770e9c27caf347ad13e0dc8e514f338af700a833857c4d13462eb.jpg)  
(a) NAS vs HPNAS (15 Gen, 15 Pop)

![](images/63cb152a38251bf8ee6b822f0a1864a453d9d758256348939c094b3a632dcc3d.jpg)  
(b) NAS vs HPNAS (15 Gen, 20 Pop)  
Fig. 3: Reward–Parameter evolution across generations. Blue and red points represent NAS and HPNAS respectively, shaded from light (early generations) to dark (later generations).

![](images/e6727a0a79c6539d1fd25ec05dc4440011c0b937b26349c7790eee2e5360964d.jpg)

![](images/23389616537f9a2461c3483c9c81c6d797641342f41493abc52dc7403933e6d4.jpg)

![](images/e25ed1c097aa3b8baf83c58c7b54d5790bef5bc098f49faa1e1d2487d8fc89a8.jpg)

![](images/9cf6c95280d962d1acd29b7efd1ebd26966a73ff61ac44435e075dfa624821f8.jpg)  
Fig. 4: Hyperparameter evolution over generations for Experiment 6 (15 generations, 20 population). Each subplot visualizes one hyperparameter, shaded from light (early generations) to dark (later generations).

Finally, the RL-based autonomous driving agent was fully trained using the bestdiscovered architecture and its corresponding optimal hyperparameters. The resulting policy achieved a peak cumulative reward of 1205, surpassing the NAS-only configuration (1190) [1], the HPO-only configuration (1190) [3], and the manually tuned baseline (1140). This final result confirms that jointly optimizing architecture and hyperparameters yields measurable performance gains in RL-based autonomous driving tasks.

## 6 Conclusion

We proposed HPNAS, a dual-branch evolutionary framework for jointly optimizing neural architectures and hyperparameters in reinforcement learning. Experiments across multiple search budgets show that HPNAS consistently outperforms standalone NAS in both cumulative reward and model eficiency. While NAS can occasionally discover high-reward models early, it tends to overparameterize due to the absence of hyperparameter adaptation. These findings highlight the efectiveness of joint optimization in achieving compact, high-performing RL policies, particularly in larger-scale experiments.

## Outlook

As ongoing work, we aim to integrate a Bayesian Neural Network (BNN) surrogate model to improve sample eficiency during the final stages of evolution. The surrogate learns to predict PPO reward from joint encodings of neural architecture and hyperparameters. Once trained on NAS–HPO–reward triplets, the surrogate model enables conditional optimization [22] by inferring the optimal hyperparameter configuration that maximizes predicted reward for a given fixed architecture encoding. While the current implementation only gathers training data, future versions will activate the surrogate in the last 20% of the search to guide sampling towards promising configurations once suficient predictive accuracy is reached.

Acknowledgements. H.G. would like to acknowledge financial support through the German Federal Ministry of Economic Afairs and Energy through the nxtAIM consortium under grant no. 19A23014Q.

Disclosure of Interests. The results, opinions and conclusions expressed in this publication are not necessarily those of Volkswagen Aktiengesellschaft.

## References

[1] Adde, N.A., Gianzina, A., Gottschalk, H., Ebert, A.: Robust evolutionary multi-objective neural architecture search for reinforcement learning (emnas-rl). In: Proceedings of the ESANN 2025 – European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, pp. 359–364. i6doc.com, Bruges, Belgium and online event, April 23–25 (2025). https://doi.org/10.14428/esann/2025.ES2025-122 . https://doi.org/10.14428/esann/2025.ES2025-122

[2] Termritthikun, C., Jamtsho, Y., Ieamsaard, J., Muneesawang, P., Lee, I.: Eeea-net: An early exit evolutionary neural architecture search. Engineering Applications of Artificial Intelligence 104, 104397 (2021)

[3] Adde, N.A., Gottschalk, H., Ebert, A.: Hyperparameter optimization for driving strategies based on reinforcement learning. In: Nicosia, G., Ojha, V., Giesselbach, S., Pardalos, M.P., Umeton, R. (eds.) Machine Learning, Optimization, and Data Science, pp. 24–38. Springer, Cham (2025)

[4] Unity3D. https://unity.com/

[5] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)

[6] Zoph, B., Le, Q.V.: Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578 (2016)

[7] Liu, H., Simonyan, K., Yang, Y.: Darts: Diferentiable architecture search. In: International Conference on Learning Representations (ICLR) (2019)

[8] Real, E., Aggarwal, A., Huang, Y., Le, Q.V.: Regularized evolution for image classifier architecture search. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, pp. 4780–4789 (2019)

[9] Stanley, K.O., Miikkulainen, R.: Evolving neural networks through augmenting topologies. Evolutionary computation 10(2), 99–127 (2002)

[10] Lu, Z., Whalen, I., Boddeti, V., Dhebar, Y., Deb, K., Goodman, E., Banzhaf, W.: Nsga-net: NAS using multi-objective genetic algorithm. In: Proceedings of the Genetic and Evolutionary Computation Conference (2019)

[11] Deb, K., Pratap, A., Agarwal, S., Meyarivan, T.: A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation 6(2), 182–197 (2002)

[12] Hirose, Y., Yoshinari, S., Shirakawa, S.: Nas-hpo-bench-ii: A benchmark dataset on joint optimization of network architecture and training hyperparameters. In: Proceedings of the 1st International Conference on Automated Machine Learning (AutoML). Proceedings of Machine Learning Research, vol. 157, pp. 1–14. PMLR, ??? (2021). https://proceedings.mlr.press/v157/hirose21a/hirose21a.pdf

[13] Bansal, A., White, C., Neiswanger, W., Savani, Y., et al.: Jahs-bench-201: A foundation for research on joint architecture and hyperparameter search. In: Advances in Neural Information Processing Systems Datasets and Benchmarks Track (2022)

[14] Kandasamy, K., Neiswanger, W., Schneider, J., P´oczos, B., Xing, E.: Neural architecture search with bayesian optimisation and optimal transport. In: Advances in Neural Information Processing Systems, vol. 31 (2018)

[15] Li, X., Chen, J., Liu, H., Zhang, Y., Wang, Y., Yu, Q.: Random architecture search for nas with resource constraints. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6749–6758 (2021)

[16] Bria, A., De Ciccio, P., D’Alessandro, T., Fontanella, F.: A novel evolutionary approach for neural architecture search. In: International Conference on Artificial Intelligence Applications and Innovations, pp. 149–161 (2023). Springer

[17] Guerrero-Viu, J., Hauns, S., Izquierdo, S., Miotto, G., Schrodi, S., Biedenkapp, A., Elsken, T., Deng, D., Lindauer, M., Hutter, F.: Bag of baselines for multiobjective joint neural architecture search and hyperparameter optimization. In: 8th ICML Workshop on Automated Machine Learning (AutoML) (2021). https://openreview.net/pdf?id=yEGlj93aLFY

[18] Zoph, B., Vasudevan, V., Shlens, J., Le, Q.V.: Learning transferable architectures for scalable image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8697–8710 (2018)

[19] Torabi, F., Warnell, G., Stone, P.: Behavioral cloning from observation, pp. 4950– 4957 (2018). https://doi.org/10.24963/ijcai.2018/687

[20] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.: Deep reinforcement learning that matters. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32 (2018)

[21] Deb, K., Agrawal, R.B.: Simulated binary crossover for continuous search space. Complex Systems 9(2), 115–148 (1995)

[22] Klein, A., Falkner, S., Bartels, S., Hennig, P., Hutter, F.: Learning curve prediction with bayesian neural networks. In: International Conference on Learning Representations (ICLR) (2017). URL: https://openreview.net/forum?id= S1OO0u5Mg