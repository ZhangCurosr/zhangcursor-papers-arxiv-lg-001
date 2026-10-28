# GNN-Guided Network Reduction for Centralized Urban Trafic Routing

Nikolai Antonov<sup>a,b,∗</sup>, Lukáš Chrpa<sup>a</sup>, Andrii Nyporko<sup>a,b</sup>, Jan Hůla<sup>a</sup>

<sup>a</sup>Czech Technical University in Prague, Czech Institute of Informatics, Robotics and Cybernetics, Jugoslávských partyzánů 1580/3, Prague, 160 00, Czech Republic   
<sup>b</sup>Czech Technical University in Prague, Faculty of Electrical Engineering, Technická 2, Prague, 166 27, Czech Republic

## Abstract

We consider the centralized trafic routing (CTR) problem, where a central planner assigns routes to origin-destination (OD) vehicle flows to minimize congestion-dependent segment cost. Prior work formulates CTR eficiently as a mixed-integer linear program (MILP), but scaling this formulation to the full road network is computationally prohibitive within an operational time budget. Bounded Routes and Smart Routes speed up optimization using precomputed reduced networks; however, these methods do not account for vehicle information within each OD flow. Motivated by this gap, we propose a learning-based urban network reduction approach. A dedicated graph neural network (GNN) first scores road segments for each OD flow using the road network and vehicle-flow information. Highly ranked segments are then incorporated into complete OD routes, and the original segment-flow MILP is solved on the resulting reduced road network. The GNN thus guides the reduction of the road network, while the solver enforces feasibility and determines the routing. Experiments on four real urban networks – Prague-Dejvice, London, New York, and Sydney – show that the proposed approach achieves the smallest relative objective diference to the 30-second full-graph reference in every city under both trafic regimes. Averaged across cities, this diference is more than six times lower than that of the best competing method. Moreover, trafic simulation results show that the optimization gains of the proposed method lead to more eficient routing, with the highest

aggregate arrival rate and a competitive trip duration per arrival percentage point.

Keywords: centralized trafic routing, trafic assignment, mixed-integer linear programming, graph neural networks, machine learning for combinatorial optimization, network reduction

## 1. Introduction

Trafic congestion in urban road networks imposes substantial economic (Serok et al., 2022; Gañan-Cardenas et al., 2024; Gunjo et al., 2024) and environmental (Khreis et al., 2024) costs, making urban trafic management a core objective for intelligent transportation systems. Unlike decentralized navigation, in which each driver selects a route independently, centralized trafic routing (CTR) jointly assigns routes to all vehicles simultaneously. Such coordination can reduce travel time, as studied both theoretically (Cominetti et al., 2024) and using real mobility data (Monnot et al., 2022). In an operational setting, drivers submit their destinations and receive coordinated routes from a central planner. At each planning step (episode), the planner processes the current set of requests and must compute routes jointly for all vehicles on a large road network within a short operational time budget, which poses a challenging computational task.

Our prior work introduced a segment-flow mixed-integer linear programming (MILP) formulation for CTR and showed its advantages over planningbased approaches (Nyporko et al., 2025, 2026). Despite the demonstrated eficiency, solving this MILP within a short time budget becomes increasingly dificult as the road network and the number of vehicle flows grow. A common way to accelerate solving in this case is to reduce the road network before optimization. Prominent fixed-reduction methods – Bounded Routes (BR) and Smart Routes (SR) (Švadlenka et al., 2023) – precompute a pool of routes for each origin-destination (OD) pair and use the union of their segments as a reduced network across episodes. However, this creates a tradeof: conservative reduction may leave the MILP size close to the full-graph formulation, whereas aggressive pruning can remove valuable alternatives needed for low-cost routing.

Since historical trafic data reveal regularities in road congestion (Wang et al., 2015), we come up with the idea of tackling CTR using a data-driven approach. In particular, we propose an urban network reduction guided by a machine learning model that evaluates the road segments based not only on the network structure, but also on the information about OD vehicle flows. For each episode, a dedicated graph neural network (GNN) scores segments separately for every OD flow and ranks them by predicted relevance for routing. Rather than predicting final routes or exact segment flows, these rankings guide the construction of an episode-specific reduced network, which is then used to restrict the original segment-flow MILP. Thus, the GNN accelerates solving without taking over the optimization model’s role: it identifies promising parts of the road network, while the solver enforces model feasibility and determines the final routing.

We evaluate the proposed approach on real road network topologies from Prague-Dejvice, London, New York, and Sydney, using scenarios with diferent numbers of OD flows and two trafic regimes. On these scenarios, we first compare our GNN-guided reduction with fixed-reduction baselines in terms of solution quality and runtime under the short operational solver budget of 10 seconds. To assess the contribution of learned ranking, we also test the same reduction procedure with non-learned ranking alternatives. In these comparisons, solution quality is measured relative to a 30-second run of the CTR MILP on the full road network, using the full episode length. Finally, we examine whether the diferences observed at the optimization level carry over to trafic by replaying the resulting routes in the simulator of urban mobility (SUMO) (Alvarez Lopez et al., 2018) and measuring arrival rate and trip duration relative to the achieved arrival rate.

Our contributions are as follows:

• A ranking-guided network reduction for CTR. We propose an algorithm that uses per-flow segment rankings to construct a reduced road network for each CTR episode. By connecting selected segments through complete OD paths and locally expanding their union, the algorithm reduces the number of routing variables in the original segment-flow MILP.

• A dedicated GNN for learning flow-specific segment rankings. To guide the road network reduction, we introduce a GNN that combines the full MILP representation, road-network characteristics, and current OD flows to rank segments separately for each flow. The model learns segment orderings from labelled routing solutions rather than predicting exact vehicle counts or final routes, while the solver determines the final routing and enforces feasibility. For each city and trafic regime, a single trained model generalizes to unseen scenarios with diferent numbers of OD flows.

• The best solution quality under a short solver budget. We evaluate the resulting GNN-guided reduction on four real urban road networks, comparing it with fixed-reduction baselines and with the same procedure using non-learned rankings. The proposed GNN-guided network reduction approach achieves the best solution quality in terms of the smallest mean relative objective diference compared to other methods in every city and trafic regime.

• Route recovery and simulation-level validation. To transform MILP solutions into individual vehicle routes, we present a dedicated route-recovery procedure and then evaluate the recovered routes in SUMO under diferent shares of rerouted vehicles to assess whether the optimization gains lead to improved trafic outcomes. Under the considered stress-test scenarios, our proposed approach yields the highest overall arrival rate across cities in both trafic regimes and remains competitive in trip duration relative to the achieved arrival rate.

The remainder of the paper is organized as follows. Section 2 reviews centralized trafic routing, network reduction, and learning-assisted optimization. Section 3 formulates the considered CTR problem as a MILP with turn restrictions. Section 4 describes the fixed reduction baselines and our proposed ranking-guided reduction framework, including GNN-based segment ranking and route recovery. Section 5 presents the experimental setup and evaluates scalability, ranking quality, and performance in optimization and microscopic simulation. Section 6 concludes the work.

## 2. Related Work

In this section, we first review decentralized, partially centralized, and centralized routing to place CTR within the broader literature (Section 2.1). We then review static and dynamic CTR alongside planning-based approaches (Section 2.2), emphasizing the role of mixed-integer CTR formulations. Finally, we discuss the network reduction methods (Section 2.3) and connect them with learning-assisted routing and optimization approaches most closely related to the proposed GNN-guided reduction (Section 2.4).

## 2.1. Decentralized, partially centralized, and centralized routing

When drivers choose routes independently, the resulting trafic pattern corresponds to a user equilibrium, which is generally less eficient than the system-optimal routing selected by a central planner (Roughgarden and Tardos, 2002). The relative diference between decentralized and system-optimal routing tends to be most pronounced at intermediate trafic levels (Colini-Baldeschi et al., 2020), while empirical studies on real road networks show that uncoordinated drivers can experience considerably longer travel times (Youn et al., 2008; Zhang et al., 2018). Thus, decentralized routing cannot generally be expected to achieve system-optimal performance.

Several partially centralized approaches aim to adjust the behavior of decentralized systems. Information-based approaches improve drivers’ visibility of road network conditions through real-time guidance and navigation apps, so that selfish routing avoids overused roads (Bagabaldo et al., 2024). Incentivebased approaches modify route choice through congestion pricing (Maheshwari et al., 2024) or controlled information release (Ambrogio et al., 2024), aligning selfish best-response with the system optimum. Compliance-aware approaches design system-optimal recommendations that remain efective when only a fraction of drivers follow them (Yun et al., 2024). Thus, partial centralization can narrow the gap between decentralized and centralized routing, but its efectiveness depends on how drivers respond to information, incentives, and route recommendations.

Unlike partially centralized approaches, a centralized planner jointly assigns routes to all vehicles to minimize the total travel cost across the road network (Morandi, 2024). This requires accounting for their interactions and the resulting congestion. However, the optimization problem becomes increasingly dificult as the road network and trafic volumes grow (Chrpa and Vallati, 2023), while practical applications require high-quality solutions within limited computation time. Thus, centralized routing directly targets system-optimal performance, but at a substantial computational cost. The following subsection reviews the main CTR formulations and solution approaches developed for this setting.

## 2.2. CTR formulations and static mixed-integer optimization

CTR is commonly studied within the framework of static trafic assignment, where routes are computed for a given number of vehicles traveling between specified origins and destinations. A representative formulation is the segmentflow MILP of Nyporko et al. (2025), which assigns integer vehicle flows to road segments and minimizes a congestion-dependent objective. Related formulations adapt the system-optimal model to practical requirements. For example, Jahn et al. (2005) introduce user constraints that limit how much longer an assigned route may be than the shortest route available to that user, while Angelelli et al. (2021) use a piecewise-linear approximation to obtain a linear formulation. A broader overview of static trafic assignment formulations is provided by Morandi (2024). Thus, static CTR formulations can combine system-wide optimization with practical user-level requirements. However, the works reviewed above primarily contribute by introducing new CTR formulations or extending existing ones, while our work addresses a complementary computational question: how to reduce the road network to solve a given segment-flow MILP more eficiently.

Dynamic trafic assignment (DTA) extends the static setting by modeling how trafic flows and travel times evolve over time. Existing system-optimal approaches include link-based formulations (Long and Szeto, 2019), pathbased formulations (Zhang and Qian, 2020), and models combining routing with trafic flow control (Como et al., 2016). Unlike these approaches, our formulation solves each episode as a static routing problem, while microscopic simulation is used only to evaluate the resulting routes; therefore, DTA lies outside the scope of this paper.

A separate line of work addresses centralized routing through automated planning. Chrpa et al. (2019) formulate CTR as a planning problem in which the generated plan defines a congestion-mitigating routing for all vehicles. This framework was evaluated in microscopic trafic simulation (Vallati and Chrpa, 2020) and later extended to hybrid and real-time settings (Vallati et al., 2021). Our previous research (Nyporko et al., 2025) compared the planning-based methods with a segment-flow MILP and found that the MILP outperformed them in scalability and microscopic simulation performance. We therefore continue to study CTR through the mixed-integer segment-flow formulation.

## 2.3. Road network reduction for mixed-integer CTR

The main obstacle to solving CTR through mixed-integer programming is the limited scalability of the resulting MILP model. Existing approaches address this issue either by accelerating the solver (Zhang et al., 2023) or by reducing the instance size. The latter strategy is commonly implemented through route-based preprocessing that constructs a reduced road network before optimization (Švadlenka et al., 2023). Prominent methods in this line of work include Bounded Routes (BR) and Smart Routes (SR) (Švadlenka et al., 2023), forming a part of a broader framework for intelligent urban trafic routing (Švadlenka and Chrpa, 2023). BR enumerates routes whose lengths are bounded relative to the shortest route, while SR additionally clusters their segment sets using the Jaccard distance and retains only representative alternatives. Our work also follows this general direction and focuses on reducing the road network before solving the MILP rather than modifying the solver itself.

A key limitation of BR and SR is that the reduced road network remains fixed across episodes (Švadlenka et al., 2023). However, the contribution of individual segments to routing quality depends on the OD flows in the current episode: a segment that is unnecessary in one episode may provide an important routing alternative in another. Road network vulnerability studies show that the impact of segment disruptions on total travel cost depends on trafic volumes and congestion, while network structure alone may not identify the most critical segments (Chen et al., 2012; Gauthier et al., 2018). Our previous experiments similarly found that the stronger pruning of SR produced longer simulated trip durations than BR, indicating that removing too many alternatives can introduce a significant suboptimality (Nyporko et al., 2025). Thus, fixed preprocessing creates a trade-of between reducing the MILP and preserving routes that may be useful in the current episode.

## 2.4. Learning-assisted optimization for CTR

Combinatorial optimization is known to incorporate machine learning as heuristics for computationally hard problems (Bengio et al., 2021; Angioni et al., 2025). In trafic routing, learned models can use information about the OD flows in the current episode to provide better routing decisions. For example, reinforcement learning approaches produce routing policies for a given set of OD flows: their training can be guided by the method of successive averages (Wang et al., 2025), or by teacher policies derived from physics-based trafic flow models (Ke et al., 2025). Other methods use surrogate models to approximate trafic flows (Lassen et al., 2025) or estimate congestion caused by road renovations (Bosch et al., 2025). However, the routes predicted by a learned model may violate the constraints of the original problem unless feasibility is enforced separately. A useful alternative is to use machine learning for guidance, for example, to identify informative variable subsets that simplify mixed-integer optimization (Dilkina et al., 2009; Ferber et al., 2022) or to select heuristics for modifying vehicle routes during a search (Johnn et al., 2024). These directions suggest using learning to guide the optimization rather than to predict the final routing directly.

Following this idea, machine learning has been applied to reduce the problem size in crew rostering (Quesnel et al., 2022) and electric bus scheduling (Gerbaux et al., 2025). Quesnel et al. (2022) use a neural network to rank pairings for each crew member and retain the most promising ones for generating schedules within a branch-and-price algorithm. In a transportation setting, Gerbaux et al. (2025) combine GNN-based and greedy arc selection to construct reduced problems within a column-generation heuristic. In both cases, learning identifies a smaller set of promising elements, while the downstream optimization procedure constructs the final solution. However, both approaches are tailored to their specific scheduling problems and the corresponding column-generation or branch-and-price frameworks. Our approach follows the same general principle but applies learned reduction directly to the variables of the segment-flow CTR MILP. It combines neural scoring of graph elements (Joshi et al., 2019) with learning-assisted mixed-integer reduction (Liu et al., 2024). Before optimization, the GNN processes the complete MILP instance, including its variables, constraints, objective, and current OD vehicle flows, and scores every flow-segment variable $x _ { r } ^ { i }$ . The scores though do not constitute a solution; they just estimate which routing variables should be retained. The subsequent ranking-guided procedure converts them into flow-specific segment sets, after which the original MILP is constructed and solved on the reduced network.

## 3. Problem formulation

We consider a centralized trafic routing problem on a congested urban road network. The goal is to jointly assign routes to all vehicles while minimizing congestion-dependent travel cost. The formulation builds on our previous segment-flow MILP (Nyporko et al., 2025). It retains the aggregate OD-flow representation and tiered segment costs, while adding constraints that enforce the turn restrictions.

## 3.1. Network, OD flows, and congestion model

The road network is modeled as a directed graph $G = ( V , R )$ , where V is the set of junctions and R is the set of directed road segments. A single segment aggregates the lanes and parallel roadways connecting the same two junctions in the same direction, so at most one segment connects any ordered pair of nodes. We therefore identify each $r \in R$ with its endpoint pair and write $r = ( u , v )$ for the segment from u to v. Each segment has a length $\ell _ { r }$ and an estimated travel time $\tau _ { r }$ . For $v \in V$ , let $\delta ^ { + } ( v ) = \{ ( v , w ) \in R \}$ and $\delta ^ { - } ( v ) = \{ ( u , v ) \in R \}$ denote the outgoing and incoming segments at v.

Some junctions are assumed to have turn restrictions. Let $V _ { T } \subseteq V$ denote the set of such junctions. For each $v \in V _ { T }$ , let $T _ { v } \subseteq \delta ^ { - } ( v ) \times \delta ^ { + } ( v )$ denote the allowed turns through v. A pair $( a , b ) \in T _ { \tau }$ means that vehicles on segment a may continue through v onto segment b. We write $T = \{ T _ { v } \} _ { v \in V _ { T } }$ for the collection of turn restrictions. Junctions outside $V _ { T }$ allow all turn pairs.

A CTR instance is specified by a set of OD vehicle flows, which we call an episode or, equivalently, a problem instance. Within an episode, each flow $i \in \{ 1 , \ldots , N _ { F } \}$ represents a group of identical vehicles traveling from origin $o ^ { i } \in V$ to destination $d ^ { i } \in V$ , where $n ^ { i } \in \mathbb N$ denotes the number of vehicles in the group. The set $R _ { i } \subseteq R$ contains the road segments available for routing flow i. Similar to Nyporko et al. (2025), we adopt a static, episode-level CTR setting in which every episode is solved independently of the others, the segment costs are not updated as trafic conditions change, and trafic is modeled at the level of aggregate OD flows rather than individual vehicle trajectories.

Each road segment $r \in R$ is associated with a load $s _ { r } .$ , defined as the number of vehicles routed through this segment. There is no hard restriction on the segment capacity (any number of vehicles may traverse it); instead, the segment cost increases with its load according to a piecewise-linear function with upward jumps at the tier thresholds. Let $\mathcal { H } = \{ 1 , \dots , H \}$ denote the set of congestion tiers. For the first $H - 1$ tiers, $\theta _ { r } ^ { k }$ denotes the upper load threshold, with $0 < \theta _ { r } ^ { 1 } < \cdot \cdot < \theta _ { r } ^ { H - 1 }$ , while the final tier extends to $+ \infty$ Each tier has a unit cost $c _ { r } ^ { k } ,$ with $c _ { r } ^ { 1 } < c _ { r } ^ { 2 } < \cdots < c _ { r } ^ { H }$ . The active tier is determined by the segment load:

$$
k ( s _ { r } ) = \left\{ \begin{array} { l l } { 1 , \quad 0 \leq s _ { r } \leq \theta _ { r } ^ { 1 } , } \\ { k , \quad \theta _ { r } ^ { k - 1 } < s _ { r } \leq \theta _ { r } ^ { k } , \quad k = 2 , \ldots , H - 1 , } \\ { H , \quad s _ { r } > \theta _ { r } ^ { H - 1 } . } \end{array} \right.\tag{1}
$$

For a given load $s _ { r }$ , the segment cost is defined as $g _ { r } ( s _ { r } ) = c _ { r } ^ { k ( s _ { r } ) } s _ { r }$ , and the total cost of an episode is given by $\sum _ { r \in R } g _ { r } ( s _ { r } )$

## 3.2. MILP formulation

Decision variables. Table 1 summarizes the decision variables.

Table 1: Decision variables of the CTR MILP.
<table><tr><td>Variable Domain</td><td></td><td>Description</td></tr><tr><td> $x _ { r } ^ { i }$ </td><td> $\{ 0 , 1 , \ldots , n ^ { i } \}$ </td><td>Number of vehicles of flow i assigned to segment r</td></tr><tr><td> $f _ { a , b } ^ { i }$ </td><td> $\mathbb { R } _ { \geq 0 }$ </td><td>Number of vehicles of flow i continuing from seg- ment a to segment b through an allowed turn</td></tr><tr><td> $s _ { r }$ </td><td> $\mathbb { R } _ { \geq 0 }$ </td><td>Load on segment r</td></tr><tr><td> $y _ { r }$ </td><td> $\mathbb { R } _ { \geq 0 }$ </td><td>Cost contribution of segment r</td></tr><tr><td> $z _ { r } ^ { k }$ </td><td> $\{ 0 , 1 \}$ </td><td>Indicator that tier k is active on segment r</td></tr></table>

We introduce three groups of variables: routing variables $x _ { r } ^ { i }$ for segments $r \ \in \ R _ { i }$ , segment variables $s _ { r } , \ y _ { r } , \ z _ { r } ^ { k }$ for segments $r \in R$ , and turn-flow variables $f _ { a , b } ^ { i }$ for allowed turns at junctions $v \in V _ { T }$ other than the origin and destination of flow i.

Constraints. Let ${ \bar { s } } = \sum _ { i } n ^ { i }$ denote the total number of vehicles in the episode. Since the load on any segment cannot exceed ${ \bar { s } } ,$ we use $M _ { s } = \bar { s } + 1$ for the upper bound of the final congestion tier and $\begin{array} { r } { M _ { y } = \bar { s } \operatorname* { m a x } _ { r \in R , k \in \mathcal { H } } c _ { r } ^ { k } + 1 } \end{array}$ for the cost linearization.

• Flow is conserved at each node:

$$
\sum _ { r \in \delta ^ { + } ( v ) \cap R _ { i } } x _ { r } ^ { i } - \sum _ { r \in \delta ^ { - } ( v ) \cap R _ { i } } x _ { r } ^ { i } = \left\{ \begin{array} { l l } { n ^ { i } , } & { v = o ^ { i } , } \\ { - n ^ { i } , } & { v = d ^ { i } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \forall v \in V , ~ \forall i .\tag{2}
$$

• No flow enters an origin and none leaves a destination:

$$
\sum _ { r \in \delta ^ { - } ( o ^ { i } ) \cap R _ { i } } x _ { r } ^ { i } = 0 , \sum _ { r \in \delta ^ { + } ( d ^ { i } ) \cap R _ { i } } x _ { r } ^ { i } = 0 , \quad \forall i .\tag{3}
$$

• The load on each segment is the total number of vehicles routed through it:

$$
s _ { r } = \sum _ { i : r \in R _ { i } } x _ { r } ^ { i } , \quad \forall r \in R .\tag{4}
$$

• Exactly one congestion tier is selected for each segment:

$$
\sum _ { k \in \mathcal { H } } z _ { r } ^ { k } = 1 , \quad \forall r \in R .\tag{5}
$$

• The segment load must satisfy the bounds of the selected tier:

$$
\begin{array} { l l } { { \displaystyle s _ { r } \leq \sum _ { k = 1 } ^ { H - 1 } \theta _ { r } ^ { k } z _ { r } ^ { k } + M _ { s } z _ { r } ^ { H } , } } & { { \forall r \in R , } } \\ { { \displaystyle s _ { r } \geq \sum _ { k = 2 } ^ { H } \theta _ { r } ^ { k - 1 } z _ { r } ^ { k } } , } & { { \forall r \in R . } } \end{array}\tag{6}
$$

At an exact threshold, either adjacent tier is feasible; since unit costs increase and $y _ { r }$ is minimized, the lower-cost tier is selected.

• The segment cost is linearized using the selected tier:

$$
y _ { r } \geq c _ { r } ^ { k } s _ { r } - M _ { y } ( 1 - z _ { r } ^ { k } ) , \quad \forall r \in R , \forall k \in \mathcal { H } .\tag{7}
$$

• At junctions with turn restrictions, the per-turn flows reproduce the incoming and outgoing segment flows:

$$
\sum _ { b \in R _ { i } : ( a , b ) \in T _ { v } } f _ { a , b } ^ { i } = x _ { a } ^ { i } , \forall v \in V _ { T } , \forall a \in \delta ^ { - } ( v ) \cap R _ { i } , \forall i : \ o ^ { i } \neq v , \ d ^ { i } \neq v ,\tag{8}
$$

$$
\sum _ { a \in R _ { i } : ( a , b ) \in T _ { v } } f _ { a , b } ^ { i } = x _ { b } ^ { i } , \forall v \in V _ { T } , \forall b \in \delta ^ { + } ( v ) \cap R _ { i } , \forall i : \ o ^ { i } \neq v , \ d ^ { i } \neq v .\tag{9}
$$

We remark that imposing integrality on $f _ { a , b } ^ { i }$ is unnecessary at this stage, as it could make the formulation harder to solve. For fixed integer segment flows $x _ { r } ^ { i }$ from a feasible MILP solution, the turn-flow constraints at each restricted junction can be represented as a maximumflow problem on a bipartite network whose capacities are given by integers $x _ { r } ^ { i }$ . Hence, an integer assignment to the variables $f _ { a , b } ^ { i }$ exists and can be recovered in polynomial time, as described in Section 4.4.

Objective. The objective minimizes the total segment cost:

$$
\operatorname* { m i n } \quad \sum _ { r \in R } y _ { r } .\tag{10}
$$

![](images/559215176dcccf7f965ec2f601cdca4df318a963149dc84e3ee8d37277fe0e5b.jpg)  
Figure 1: Fixed and episode-specific network reduction. Boxes show states, while arrow labels show operations. (a) The fixed procedure constructs the flow-specific segment sets $R _ { i } ^ { \prime }$ independently of the current episode. (b) Our approach predicts per-flow segment scores and constructs these sets for the current episode. Both approaches use the same segment-flow MILP.

## 4. Solution Methodology

This section presents our GNN-guided network reduction approach and the fixed-reduction methods used for comparison. In every case, the goal is to restrict the segment-flow MILP by replacing the set $R _ { i }$ available to each flow i with a flow-specific segment set $R _ { i } ^ { \prime } \subseteq R _ { i } { \mathrm { : } }$ ; the formulation in Section 3.2 then includes routing variables and associated constraints only for segments in $R _ { i } ^ { \prime } .$ . We use the term reduced network to refer to the resulting collection $\{ R _ { i } ^ { \prime } \} _ { i = 1 } ^ { N _ { F } }$ . However, the presented approaches difer in how the flow-specific segment sets $R _ { i } ^ { \prime }$ are obtained. Section 4.1 describes fixed reduction methods, which construct $R _ { i } ^ { \prime }$ without using any information about the vehicles in the current episode. Section 4.2 introduces our algorithm, which constructs $R _ { i } ^ { \prime }$ from per-flow segment scores provided by a ranking oracle. Section 4.3 describes the proposed GNN approach for predicting the segment scores. Finally, Section 4.4 explains how a segment-flow solution is converted into individual vehicle routes. Figure 1 illustrates the fixed reductions in panel (a) and our proposed approach in panel (b).

## 4.1. Fixed reduction methods

The fixed reduction scheme is illustrated in Figure 1(a). For each OD pair, the procedure generates a set of routes whose combined segments define the allowed set $R _ { i } ^ { \prime }$ – once constructed, these sets remain unchanged across episodes. The approaches presented below represent diferent levels of pruning aggressiveness, from no pruning on the full graph to retaining only one shortest path for each OD pair.

Full graph (FG). FG applies no network reduction and sets $R _ { i } ^ { \prime } = R _ { i }$ for every flow. The MILP is therefore solved on the complete road network, retaining all routing variables of the original formulation without prior segment removal. This yields the largest and most computationally demanding model among the evaluated approaches. We therefore use FG as the reference for measuring the objective quality of the reduced approaches.

Bounded Routes. For each OD pair, the BR approach (Švadlenka et al., 2023) precomputes a large pool of candidate routes. An $\mathrm { A } ^ { * }$ search enumerates routes whose length lies within a factor $c > 1$ of the shortest route, using the shortest-path length to the destination as the heuristic while respecting the allowed turns at each expansion. Every segment appearing in at least one enumerated route is then retained, and the union of these segments for the OD pair defines $R _ { i } ^ { \prime }$ . BR thus discards segments absent from all selected routes while preserving several routing alternatives for each OD pair.

Smart Routes. The SR approach (Švadlenka et al., 2023) starts from the BR route pool and removes near-duplicate routes. DBSCAN clusters all available routes based on the Jaccard distance between their segment sets. Within each cluster, SR retains the shortest route, while routes classified as noise are likewise preserved as distinct alternatives. The corresponding $R _ { i } ^ { \prime }$ is then the union of the segments of the selected routes. Thus, SR preserves structurally distinct alternatives while eliminating routes that traverse largely overlapping segment sets, typically yielding a smaller flow-specific segment set than BR.

Shortest Path (SP). The SP approach retains only the shortest route satisfying the turn restrictions for each OD pair. Thus, SP represents the most aggressive fixed reduction considered in the evaluation, eliminating all alternative routing choices for each OD pair. Unlike BR and SR, it requires neither route enumeration nor clustering. As the extreme case of network reduction, SP provides a useful baseline for assessing the trade-of between model size and the loss of routing flexibility. In particular, comparison with less restrictive reductions indicates how much solution quality is afected when each OD pair is confined to a single predefined route.

We emphasize that the fixed approaches described above ignore the information about the vehicles in the current instance, and all episodes use the same flow-specific segment sets. Our approach, introduced next, instead adapts $R _ { i } ^ { \prime }$ to each CTR instance using both the information about the road network and the OD flows in the current episode.

## 4.2. Ranking-guided network reduction

The proposed approach is illustrated in Figure 1(b) and summarized in Algorithm 1. For each OD flow $i ,$ a ranking oracle assigns a score $q _ { r } ^ { i } \in \mathbb { R }$ to every segment $r ,$ taking into account both the road network and the OD flow information in the current episode. The resulting scores therefore make the reduction specific to the current CTR episode. Given these scores, the procedure selects a limited set of ranked segments and constructs complete OD routes containing them. The union of these routes is then expanded locally to retain nearby routing alternatives. Once the procedure is finished, the original segment-flow MILP is solved on the union of all retained segments, with each flow i restricted to its corresponding segment set $R _ { i } ^ { \prime }$

Step 1: Initializing the route pool. For each OD flow i, we first initialize the pool of routes $\mathcal { P } _ { i }$ with the shortest (by length) path which satisfies the turn restrictions. We compute it using Dijkstra’s algorithm, tracking both the current and previous junctions to enforce the turn restrictions. This route is retained regardless of the ranking, ensuring that at least one complete OD route remains available even if no additional ones are constructed.

Step 2: Constructing OD routes using segment rankings. We start by considering the first $K _ { \mathrm { r } }$ segments in the ranking and perform a further selection among them. For each candidate, we search for a shortest (by estimated travel time) OD path that passes through this segment. We use forward and reverse Dijkstra searches from the origin and destination, respectively, reusing their results for all candidates of the same OD pair. A segment is selected only when we find a path that respects the turn restrictions, contains no repeated junctions, and has travel time at most $\lambda d _ { \tau } ^ { i }$ . Here, $d _ { \tau } ^ { i }$ is the shortest estimated travel time between the origin and destination without requiring the route to pass through the segment. Only the segments that do not fail this check proceed to the second phase. The routes found are stored separately in a pool $\widehat { \mathcal { P } } _ { i }$ , but are not yet added to $\mathcal { P } _ { i }$

In the second phase, we construct routes through the selected segments, starting with those ranked highest. Segments that are already included in at least one route of $\mathcal { P } _ { i }$ are skipped. We first convert the ranking positions of the segments into values $\rho _ { r } ^ { i } \in [ 0 , 1 ]$ : the highest-ranked segment receives $0 ,$ and the lowest-ranked segment receives 1. Using these values as costs, we connect the origin to the start of the selected segment and its end to the destination through minimum-cost routes. Together with the selected segment, these form a complete OD route. If this route passes the same checks as in the first phase, we add it to $\mathcal { P } _ { i } ;$ otherwise, we add the route through this segment stored in $\widehat { \mathcal { P } } _ { i }$ during the first phase.

Algorithm 1 Ranking-guided network reduction   
Require: CTR instance $I = ( G , T , \{ ( o ^ { i } , d ^ { i } , n ^ { i } , R _ { i } ) \} _ { i = 1 } ^ { N _ { F } } )$ , per-flow segment   
scores $\left\{ q _ { r } ^ { i } \right\}$ , rank cutof $K _ { \mathrm { r } } ,$ detour factor $\lambda$   
Ensure: Flow-specific segment sets $\{ R _ { i } ^ { \prime } \} _ { i = 1 } ^ { N _ { F } }$   
▷ In Step 2, ∅ means no route passing the checks was obtained.   
1: for all flows i do   
Step 1: Initializing the route pool   
2: $p _ { 0 } ^ { i } $ ShortestODPath $( R _ { i } , T , o ^ { i } , d ^ { i } ; \ell )$   
3: $\mathcal { P } _ { i } \gets \{ p _ { 0 } ^ { i } \}$   
Step 2: Constructing OD routes using segment rankings   
▷ Phase 1: Segment selection   
4: $d _ { \tau } ^ { i } \gets \mathrm { S h o r t e s t O D C o s t } ( R _ { i } , T , o ^ { i } , d ^ { i } ; \tau )$   
5: $S _ { i }  \emptyset$   
6: for all $r \in \mathrm { T o p K } _ { K _ { \mathrm { r } } } ( q ^ { i } | _ { R _ { i } } )$ in ranking order do   
7: $\hat { p } _ { r } ^ { i } \gets$ BuildODRouteThrough(r, $R _ { i } , T ; \tau , \lambda d _ { \tau } ^ { i } )$   
8: if $\hat { p } _ { r } ^ { i } \neq \varnothing$ then   
9: $S _ { i }  S _ { i } \cup \{ r \}$   
10: end if   
11: end for   
12: $\widehat { \mathcal { P } } _ { i } \gets \{ \hat { p } _ { r } ^ { i } : r \in S _ { i } \}$   
▷ Phase 2: Route construction   
13: $U _ { i } \gets p _ { 0 } ^ { i } ; \rho ^ { i } \gets$ PercentileRanks $( q ^ { i } | _ { R _ { i } } )$   
14: for all $r \in S _ { i }$ in increasing order of $\rho _ { r } ^ { i }$ do   
15: if $r \not \in U _ { i }$ then   
16: p ← RankedODPathThrough $( r , R _ { i } , T , \rho ^ { i } ; \tau , \lambda d _ { \tau } ^ { i } )$   
17: if $p = \emptyset$ then   
18: $p \gets \hat { p } _ { r } ^ { i }$   
19: end if   
20: ${ \mathcal { P } } _ { i } \gets { \mathcal { P } } _ { i } \cup \{ p \}$   
21: $U _ { i } \gets U _ { i } \cup p$   
22: end if   
23: end for   
Step 3: Constructing the flow-specific segment set   
24: E<sub>i</sub> ← ExpandOneTurn $( U _ { i } , R _ { i } , T )$   
25: R<sup>′</sup> ← PruneToCompleteOD $( E _ { i } , o ^ { i } , d ^ { i } , T )$   
26: end for   
27: return $\{ R _ { i } ^ { \prime } \} _ { i = 1 } ^ { N _ { F } }$

Step 3: Constructing the flow-specific segment set. Once the pool of routes is constructed, we combine all segments used by these routes into a set $U _ { i }$ We then expand this set by adding neighbouring segments that can directly precede or follow at least one segment in $U _ { i }$ without violating the turn restrictions. We denote the resulting set by $E _ { i }$ . We then prune $E _ { i } ,$ removing segments that are unreachable from the origin or from which the destination cannot be reached within this network. The remaining segments form the final set $R _ { i } ^ { \prime }$ available for routing flow $i .$

Optimization on the reduced network. Once the flow-specific segment sets $R _ { i } ^ { \prime }$ are obtained, their union $R ^ { \prime } = \cup _ { i } R _ { i } ^ { \prime }$ defines the reduced segment set of the original segment-flow MILP, with each flow i restricted to its corresponding $R _ { i } ^ { \prime }$ . Variables $x _ { r } ^ { i }$ are therefore created only for segments $r \in R _ { i } ^ { \prime } .$ , while the flow conservation and turn restrictions retain their original form. All other parts of the formulation remain unchanged. We note that the solver may combine segments in $R _ { i } ^ { \prime }$ into any routing allowed by the original constraints so the search space is not restricted to the routes in $\mathcal { P } _ { i }$

We remark that the reduction procedure does not depend on how the segment rankings are obtained. In this work, we propose to generate them using the GNN described in the following subsection.

## 4.3. GNN-based segment ranking

To obtain the segment rankings, we use a GNN that processes the full MILP instance before network reduction. The model is trained on FG solutions, where the number of vehicles on each segment determines the segment ordering for the corresponding flow (segments used by more vehicles rank higher). We next describe the construction and training of the GNN in detail.

![](images/6cf681842edfd6524078e9d93d4e6cdd4ec9b90dfac67bfd0d3370b3c47d5361.jpg)  
Figure 2: GNN segment ranking model. The current CTR MILP instance is represented by variable, constraint, and objective nodes connected through six directed relation types. Node features combine MILP (blue), road network (green), and OD-flow (orange) information. Four GATv2 layers perform message passing (purple), after which a readout head assigns a raw score $q _ { r } ^ { i }$ to each routing variable $x _ { r } ^ { i }$ . These scores are ranked separately within each OD flow and passed to the network reduction procedure.

Representation of the problem instance. The full MILP instance is represented as the heterogeneous graph shown in Figure 2. Following Gasse et al. (2019), each MILP variable and constraint is represented by a separate node. We additionally introduce a single objective node, resulting in three node types: variables, constraints, and objective. A variable node and a constraint node are connected if the variable has a nonzero coeficient in that constraint, and the coeficient is used as the edge attribute. Every variable and constraint node is also connected to the objective node; variable-objective edges carry the corresponding linear objective coeficients. All three connection types are represented in both directions, resulting in six directed relation types. Since the graph size is not fixed, the same GNN can process instances with diferent numbers of OD flows.

Node features. The graph is augmented with node features that describe the current CTR instance from three perspectives: its MILP representation, fixed characteristics of the transportation network, and the OD flows in the current episode. The complete set of features for each node type is reported in Table 2. MILP-related features encode the variables and constraints and their role in the formulation, using information such as variable bounds, linear objective coeficients, constraint coeficients, and right-hand sides. Network features capture the physical and routing properties of individual segments, including travel characteristics, capacity, shortest-path position, and allowed turns. OD-flow features reflect the episode-specific use of the network, including flow sizes, flow-size-to-capacity ratios, and the sizes of other flows competing for the same segments.

Table 2: Node features used in the GNN representation.
<table><tr><td>Node type</td><td>Features</td></tr><tr><td>Variable (77)</td><td>variable type, linear objective coefficient, and bounds; routing, network, and OD-flow features for  $x _ { r } ^ { i } \mathrm { ; }$  turn context for  $f _ { a , b } ^ { i } { \mathrm { ; } }$  segment features for  $s _ { r } , y _ { r }$  , and  $z _ { r } ^ { k }$  with</td></tr><tr><td></td><td>additional tier-specific features for  $z _ { r } ^ { k }$  Constraint (44) constraint type, right-hand side, and sense; flow and OD context, segment and OD-flow features, congestion-</td></tr><tr><td></td><td>tier information, and turn connectivity according to the constraint type</td></tr></table>

Message passing and training. The features of each node type are first projected into a common hidden space. Four message-passing layers then apply a separate GATv2 convolution (Brody et al., 2022) to each of the six directed relation types, using the corresponding edge attributes. The resulting messages are aggregated and combined with the previous node representation through a residual connection, followed by layer normalization, LeakyReLU, and dropout. A two-layer readout assigns a scalar score to every variable node, while only scores associated with routing variables $x _ { r } ^ { i }$ are retained as $q _ { r } ^ { i }$ . The model is trained using FG solutions, with normalized segment usage $\bar { x } _ { r } ^ { i } = x _ { r } ^ { i } / n ^ { i }$ defining the target ordering within each flow. For a sampled pair $( r , s )$ with $\bar { x } _ { r } ^ { i } > \bar { x } _ { s } ^ { i }$ , the loss is $\mathcal { L } _ { i , r , s } = \sqrt { \bar { x } _ { r } ^ { i } - \bar { x } _ { s } ^ { i } }$ softplus $( q _ { s } ^ { i } - q _ { r } ^ { i } )$ , which penalizes incorrect ordering and gives more weight to larger diferences in segment usage.

## 4.4. Route recovery

Any solution returned by the CTR MILP must still be converted into individual vehicle routes. Nyporko et al. (2025) extract the solution using a DFS-based algorithm. However, the turn restrictions considered here require a diferent procedure. We therefore propose the route recovery procedure summarized in Algorithm 2.

Algorithm 2 Route recovery with turn restrictions   
Require: MILP solution $( x , f )$   
Ensure: Vehicle routes $\{ \mathcal { Q } _ { i } \} _ { i = 1 } ^ { N _ { F } }$   
1: $\mathcal { Q } _ { i }  \mathcal { O }$ for all i   
2: for all flows i with $n ^ { i } > 0$ do   
3: for all $v \in V _ { T } \ \backslash \ \{ o ^ { i } , d ^ { i } \}$ do   
4: if some $f _ { a , b } ^ { i } , ( a , b ) \in T _ { v } .$ , is fractional then   
5: $( f _ { a , b } ^ { i } ) _ { ( a , b ) \in T _ { v } }  1$ MaximumFlow $( x ^ { i } , T _ { v } )$   
6: end if   
7: end for   
8: $x _ { \mathrm { r e s } } ^ { i }  x ^ { i } ; f _ { \mathrm { r e s } } ^ { i }  f ^ { i }$   
▷ DFS uses positive residual flows and never repeats a segment.   
▷ Backtrack over routes and vehicle counts if needed.   
9: while $| \mathcal { Q } _ { i } | < n ^ { i }$ do   
10: p ← DFSRoute $( i , x _ { \mathrm { r e s } } ^ { i } , f _ { \mathrm { r e s } } ^ { i } )$   
11: Choose $1 \leq h \leq n ^ { i } - | \mathcal { Q } _ { i } |$ supported by the residual flows along p   
12: Add h copies of $p$ to $\mathcal { Q } _ { i } ;$ subtract h from ${ x _ { \mathrm { r e s } } ^ { i } }$ and $f _ { \mathrm { r e s } } ^ { i }$ along p   
13: end while   
14: Discard any residual circulations   
15: end for   
16: return $\{ \mathcal { Q } _ { i } \} _ { i = 1 } ^ { N _ { F } }$

As noted in Section 3.2, an integrality requirement is not imposed on the turn-flow variables $f _ { a , b } ^ { i }$ . The solver may return integer values for them, but it is not required to do so. If a given assignment is fractional at some restricted junction, we replace it with an integer one. For each flow i and junction $v \in V _ { T } \ \backslash \ \{ o ^ { i } , d ^ { i } \}$ , we formulate a maximum-flow problem: the nodes represent incoming and outgoing segments; we additionally introduce a source and a sink. We then connect the source to the incoming segments and the outgoing segments to the sink, adding arcs between segments only for allowed turns. The capacities of the arcs adjacent to the source and sink equal the corresponding $x _ { r } ^ { i }$ values. Since all capacities are integer, the integrality property of maximum flow guarantees an integer solution. We remark that maximum flow can be solved in polynomial time.

Once an integer solution is available for all restricted junctions, we reconstruct the routes for each flow i. Initially, the residual flows equal the solution values $x _ { r } ^ { i }$ on segments and $f _ { a , b } ^ { i }$ on turns. A depth-first search finds a feasible OD route with no repeated segments using only positive residual flows. After assigning vehicles to the route, we subtract their number from the residual flows on its segments and turns and repeat the search using the updated values. The procedure continues until routes have been assigned to all $n ^ { i }$ vehicles. Any remaining flow forms circulations that carry no vehicles between the origin and the destination and is discarded.

## 5. Experimental Evaluation

This section evaluates the proposed approach from four complementary perspectives. We first study how the considered CTR MILP scales with the number of OD flows and identify the settings in which it becomes dificult to solve within the operational budget, while additional computation can still improve the solution quality (Section 5.2). Next, we assess how accurately the GNN ranks segments (Section 5.3) and then compare the proposed approach with the other methods in terms of objective quality and runtime (Section 5.4). Finally, we replay the obtained routes in a microscopic trafic simulator to assess how diferences in the optimization translate into trafic outcomes (Section 5.5).

## 5.1. Experimental setup

Road networks. The experiments are carried out on four real urban road networks – Prague-Dejvice, London, New York, and Sydney – spanning a range of sizes and topologies. Table 3 reports their number of junctions, directed segments, and junctions with turn restrictions, together with the total segment length. The same rule is used for determining congestiondependent segment costs in all four networks. Similar to Ekström et al. (2012) and Angelelli et al. (2021), who use piecewise-linear approximations of congestion-dependent travel times, we represent each segment cost by ten tiers with quadratic threshold ofsets and a fourth-power cost progression. To define its segment-specific thresholds, let $m _ { r }$ denote the number of lanes represented by segment r, and let $\ell _ { r j }$ be the length of lane $j .$ . Using an average vehicle spacing of 5 metres, we define the nominal segment capacity as $\begin{array} { r } { \kappa _ { r } = \sum _ { j = 1 } ^ { m _ { r } } [ \ell _ { r j } / 5 ] } \end{array}$ . The congestion thresholds are set relative to this capacity. The first nine tiers have finite upper thresholds $\theta _ { r } ^ { k } = \kappa _ { r } + ( k - 1 ) ^ { 2 }$ $k = 1 , \ldots , 9$ , while the tenth tier is unbounded. We next define the cost associated with each tier. The base cost is the expected traversal time of a lightly loaded segment. This time combines the travel time along the segment with the expected trafic-light delay at its downstream junction. We denote it by $\bar { c } _ { r }$ and set $c _ { r } ^ { 1 } = \operatorname* { m a x } \{ 1 , \lfloor \bar { c } _ { r } \rfloor \}$ . The remaining finite-tier costs are $c _ { r } ^ { k } = \operatorname* { m a x } \{ c _ { r } ^ { k - 1 } + 1 , \lfloor c _ { r } ^ { 1 } + ( 5 0 0 0 - c _ { r } ^ { 1 } ) ( ( k - 1 ) / 9 ) ^ { 4 } + 1 / 2 \rfloor \} , k = 2 , \ldots , 9$ , with $c _ { r } ^ { 1 0 } = 5 0 0 0$

Table 3: Topology of the four test cities.
<table><tr><td>City</td><td>|V|</td><td> $| R |$   $| T |$ </td><td>Length (km)</td></tr><tr><td>Dejvice</td><td>95</td><td>183 5</td><td>15.1</td></tr><tr><td>London</td><td>308</td><td>645 65</td><td>40.7</td></tr><tr><td>New York</td><td>212</td><td>453 9</td><td>51.8</td></tr><tr><td>Sydney</td><td>441</td><td>1058 20</td><td>106.4</td></tr></table>

Evaluation scenarios. We retain the scenario structure used in Nyporko et al. (2025) and extend it to four cities and a wider range of OD flows. Each scenario reflects one hour of peak-period trafic and consists of 120 episodes of length $\Delta = 3 0$ seconds. Each episode contains $N _ { F } \mathrm { \ O D }$ flows. How the number of vehicles in each flow evolves over the hour depends on the trafic regime. We consider two trafic regimes for each city: static and increasing. In the static regime, each flow independently follows either a constant or an increasing trafic pattern with equal probability; in the increasing regime, all flows follow the increasing trafic pattern. Both patterns are described below.

For a flow with a constant trafic pattern, the rate $\lambda _ { i }$ is randomly selected from a discrete set spanning 0.10–0.67 vehicles per second. With $\Delta = 3 0$ seconds, this corresponds to $\widetilde { n } _ { e } ^ { i } \in \{ 3 , \dots , 2 0 \}$ vehicles per episode, obtained by rounding $\lambda _ { i } \Delta$ . For city $c ,$ this number is scaled as $n _ { e } ^ { i } = \operatorname* { m a x } \{ 1 , \lfloor \gamma _ { c } \widetilde { n } _ { e } ^ { i } + 1 / 2 \rfloor \}$ where $\gamma _ { c }$ is the city-specific scaling factor. Since $\lambda _ { i }$ is fixed, the number of vehicles remains constant throughout the one-hour scenario.

For a flow with an increasing trafic pattern, one rate triple $\lambda _ { i } ^ { ( 1 ) } < \lambda _ { i } ^ { ( 2 ) } <$ $\lambda _ { i } ^ { ( 3 ) }$ is randomly selected and used over three consecutive 20-minute periods, indexed by $s \in \{ 1 , 2 , 3 \}$ . These rates lie between 0.10 and 0.44 vehicles per second. For each episode e in period $s ( e )$ , the number of vehicles $\widetilde { n } _ { e } ^ { i }$ is sampled independently as $\widetilde { n } _ { e } ^ { i } \sim \mathrm { P o i s s o n } \Big ( \lambda _ { i } ^ { ( s ( e ) ) } \Delta \Big )$ . Thus, the number of vehicles varies across episodes, while its expected value increases in three stages over the hour. The same city-specific scaling is then applied to positive values of $\widetilde { n } _ { e } ^ { i }$

We use ten scenarios per experiment for every (city, mode, ${ \cal N } _ { F } )$ combination; microscopic simulation reuses the final evaluation scenarios. The scaling factor $\gamma _ { c }$ equals 0.80, 0.72, 0.67, and 1.07 for Prague-Dejvice, London, New York, and Sydney, respectively. In the optimization comparison, all methods receive a 10-second operational solver budget. A separate 30-second FG run provides a reference solution for each episode and allows us to assess the benefit of additional computation. This longer budget matches the episode length $\Delta$ but is not proposed for operational use, since preparing the optimization problem, recovering routes, and communicating them to vehicles also require time.

Compared methods. The evaluation includes all methods introduced in Section 4.1 (FG, the fixed reduction methods SP, BR, and SR) as well as three ranking oracles that use the procedure described in Section 4.2: GNN, linear programming (LP), and Random. The proposed GNN oracle ranks the segments of each flow using its predicted scores, as described in Section 4.3. The LP oracle instead uses the fractional $x _ { r } ^ { i }$ values obtained from the continuous relaxation of the FG, while Random uses a random segment permutation for each flow. All three oracles – GNN, LP, and Random – use the same $K _ { \mathrm { r } }$ and $\lambda ,$ calibrated from the GNN training and validation data as described in Section 5.4.

To assess the reliability of conclusions about diferences between methods, we perform a statistical analysis of the obtained results. For each evaluated metric and trafic regime, we compare GNN with the best of the other methods using the All cities results, pairing their outcomes on the same scenarios. We construct 95% confidence intervals using bootstrap and test the significance of diferences using a two-sided paired randomization test. Since the best other method is selected from several alternatives rather than specified in advance, we apply Holm’s correction to the p-values to reduce the risk of falsely declaring diferences significant.

Implementation details. The implementation is done in Python 3.13 and uses Gurobi 13.0.2, PyTorch Geometric, and SUMO v1.27.1 (Alvarez Lopez et al., 2018). The code and data are available in a public repository.<sup>1</sup> BR uses a pool of at most 100 routes per OD pair and c = 1.3; SR applies DBSCAN with ε = 0.3 and a minimum of two samples. The GNN uses a hidden dimension of 128, four attention heads, and a dropout rate of 0.1. All experiments run on a NixOS server with an AMD EPYC 9124 processor (64 cores), 384 GB of RAM, and one NVIDIA L40S GPU with 48 GB of memory.

## 5.2. Scalability of the MILP formulation

This series of experiments studies how the considered CTR MILP scales as the number of OD flows increases. We are interested in the intermediate range of OD flows where the operational 10-second budget is often insuficient to approach an optimum, but extending the solver budget to 30 seconds still leads to substantially better results. This is the range where the solver can benefit the most from the proposed learned reduction: the problem is already nontrivial, so the solver cannot handle it within the operational budget; however, the improvement under the longer budget indicates that high-quality solutions remain attainable, motivating the use of these instances for GNN training. To identify the targeted intermediate range for each road network and trafic regime, we vary the number of origin-destination pairs, considering $N _ { F } \in \{ 5 $ , 10, 15, 20, 25, 30, 35, 40, 45, 50}.

We use two metrics to characterize the solver’s performance under both the operational and extended time budgets. The first is the fraction of episodes for which FG obtains a feasible solution with a primal–dual gap of at most 1%. The gap is defined as $g _ { \mathrm { P D } } = ( z _ { \mathrm { b e s t } } - z _ { \mathrm { b o u n d } } ) / | z _ { \mathrm { b e s t } } |$ , where $z _ { \mathrm { b e s t } }$ is the best feasible objective value found and $z _ { \mathrm { b o u n d } }$ is the proven lower bound. The second metric we use is the fraction of episodes for which FG obtains any feasible solution, regardless of its gap. Figure 3 demonstrates the results.

Discussion. We first analyze the relation between the solid and dashed curves, which report, respectively, the fractions of episodes in which FG obtains a solution with a primal–dual gap of at most 1% and any feasible solution. We make two observations. First, the values of $N _ { F }$ at which the curves begin to decline broadly follow network size: Prague-Dejvice starts to decrease only at large $N _ { F }$ , London and New York do so in the middle range, while Sydney becomes computationally hard already at small $N _ { F }$ . Second, the distance between the solid and dashed curves shows whether the main dificulty lies in improving an already found solution or finding a feasible solution at all. In Prague-Dejvice, a feasible solution is found in almost every episode (a minimum of 99.75%) over the entire evaluated range, so the decrease of the solid curves mainly reflects the increasing dificulty of reaching a small primal-dual gap. London and New York show a two-stage trend: the solid curves decrease first while feasible solutions are still found in most episodes.

![](images/519e2613b7fbd348ae3169ee2cc69a58624db8a641119c8f52474e69d0a41780.jpg)  
Figure 3: FG scalability. Solid: $\exp \leq 1 \% ;$ dashed: feasible solution; shading: selected $N _ { F }$

For Sydney, the distance between the curves is the smallest, so finding a feasible solution becomes almost as dificult as obtaining a high-quality one, and feasibility itself quickly becomes computationally hard even for moderate numbers of OD flows.

We next identify the settings in which the problem is hard to tackle within the operational 10-second budget, but extending the solver budget to 30 seconds can still improve the solution. The shaded regions in Figure 3 mark the corresponding ranges of five consecutive $N _ { F }$ values selected for each city and trafic regime. We place each region around the point where the 10-second solid curve starts to decline while the 30-second curve remains substantially higher. This is the regime in which the proposed reduction can provide the clearest benefit: the full formulation is already dificult to solve within the operational budget, but the longer runs show that high-quality solutions remain attainable. The selected ranges are 28–32 and 31–35 for Prague-Dejvice, 7–11 and 10–14 for London, 12–16 for both New York modes, and 5–9 and 6–10 for Sydney, with the static regime listed first. These city-specific ranges are used in all subsequent experiments.

## 5.3. GNN training results

This series of experiments demonstrates the ability of the proposed GNN to learn the flow-specific segment ordering. We train a separate GNN model (with the same architecture but diferent learned weights) for each city and trafic regime. Each model is trained jointly on the five corresponding $N _ { F }$ values selected in the previous scalability experiment, using ten scenarios (1200 episodes) for each $N _ { F }$ value. For every $N _ { F }$ , eight complete scenarios are used for training and two for validation. To obtain high-quality labels, we solve FG with a 300-second time limit and retain every episode for which a feasible solution is found. All labelling runs produced a feasible solution, yielding 6000 labelled MILP instances per model. The problem representation, target ordering, and training loss are described in Section 4.3. Training uses Adam with learning rate $5 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 8, and a cosine learning rate schedule for at most 50 epochs. Early stopping is applied with a patience of five epochs.

We evaluate ranking quality using pairwise accuracy and top-K recall. For each flow, we sample 128 segment pairs with diferent target values and weight them as defined in Section 4.3. A sampled pair with $\bar { x } _ { r } ^ { i } > \bar { x } _ { s } ^ { i }$ is correctly ordered when $q _ { r } ^ { i } > q _ { s } ^ { i }$ . Pairwise accuracy is thus defined as the weighted fraction of correctly ordered pairs. The second metric, top-K recall, considers the segments used by a flow i in the labelled solution. It measures the fraction of such segments that appear among the K highest-scoring segments for that flow. We report the mean top-K recall across flows and validation episodes for $K \in \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 \}$ . For model selection, validation pairwise accuracy is computed separately for each $N _ { F }$ , and the model with the highest mean across the five values is retained.

![](images/e47c10fa1fe1ea8dac2ed0b669060510515ffaa08df53a6d9578efb066a5eea9.jpg)  
Figure 4: GNN validation ranking performance in the static (S) and increasing (I) regimes; bar labels show cumulative recall.

Discussion. Figure 4 shows that all eight models learn the target ordering consistently, with pairwise accuracy ranging from 92.0% to 96.1% and comparable between the two trafic regimes within each city. The Prague-Dejvice models achieve the highest accuracy, consistent with this network being the smallest of all four. The lowest values occur for Sydney, which is the largest studied network, and London, where a lower accuracy may also reflect its specific topology. Overall, the accuracy remains above 92% across all cities and trafic regimes, with a spread of only about four percentage points, indicating that the GNN learns reliably and preserves the relative segment-usage ordering irrespective of network size or trafic intensity.

Similarly to pairwise accuracy, top-K recall also exposes several diferences among the city networks. In Prague-Dejvice, the top 50 positions already recover more than 99.5% of the labelled segments. New York reaches 95.9– 96.9% at $K = 1 0 0$ and almost 100% at $K = 2 0 0$ ; London reaches only 80.4–82.4% at $K = 1 0 0$ but approximately 99.5% at $K = 2 0 0$ , while Sydney reaches 90.9–94.4% at $K = 2 0 0$ . Thus, even the cities with comparable total road lengths can require diferent prefix sizes – for example, London has a shorter total road length than New York but more segments and turnrestricted junctions, and requires a larger K to achieve comparable recall. As the required prefix sizes notably difer across the networks, the results support calibrating $K _ { \mathrm { r } }$ separately for each city and trafic regime.

## 5.4. Optimization-level evaluation

This experiment evaluates whether the proposed approach outperforms other methods on the considered CTR MILP under the operational 10-second solver budget. For each city and trafic regime, we evaluate five $N _ { F }$ values selected in Section 5.2, using ten test scenarios (1200 episodes per method) for each $N _ { F }$ value. The reduction parameter $K _ { \mathrm { r } }$ is calibrated independently for each (city, mode, ${ \cal N } _ { F } )$ combination. To determine it, we first weight each ranked segment by the fraction of vehicles routed through it in the labelled solution; we then choose $K _ { \mathrm { r } }$ as the smallest number of top-ranked segments containing 90% of the total weight in the GNN validation data. The detour factor λ is the maximum of 2.0 and the 99th percentile of route travel times in the training solutions relative to the shortest travel time for the same OD pair. The SP routing provides a feasible initial solution for every instance. A separate run of FG with a 30-second time limit, denoted $\mathrm { F G } _ { \mathrm { r e f } } .$ , provides a reference solution for each episode.

We evaluate each method in terms of its relative objective diference from $\mathrm { F G } _ { \mathrm { r e f } }$ and runtime. The diference is defined as $d _ { a } = ( z _ { a } - z _ { \mathrm { r e f } } ) / | z _ { \mathrm { r e f } } | .$ where $z _ { a }$ is the best objective value found by the evaluated method and $z _ { \mathrm { r e f } }$ is the corresponding best objective value found by $\mathrm { F G } _ { \mathrm { r e f } }$ . We report the average diference across all episodes for each city and trafic regime. As for runtime, the total time includes computing the scores (where applicable) and constructing the reduced network, as well as constructing and solving the MILP. The one-time loading of a GNN model is excluded. We also separately report the Gurobi time spent for solving the MILP. For each of the two runtime metrics, we remove the lowest and highest 10% of values and average the remainder.

Discussion. Table 4 shows that GNN obtains the smallest average relative objective diference from $\mathrm { F G } _ { \mathrm { r e f } }$ across all four city networks and both trafic regimes. The advantage ranges from a small margin on the small and moderatesized networks (Prague-Dejvice static, London increasing) to larger values on the bigger and computationally harder networks (New York and Sydney, in both settings). A particularly strong result is the negative diference achieved by GNN in the increasing regime for New York and Sydney: the GNN-reduced model, under a 10-second budget, finds better solutions than $\mathrm { F G } _ { \mathrm { r e f } }$ , which is given a three-times longer, 30-second budget – it indicates that the GNN-based reduction is highly informative and can yield substantial time savings. We further note that every GNN run returned at least one feasible solution improving on the SP initialization. We note in addition that the $\mathrm { F G } _ { \mathrm { r e f } }$ is tightly bounded in most of the episodes: its primal-dual gap is no more than 1% in 94.4% of static and 84.2% of increasing-regime episodes. Thus, the mean certified gaps of GNN, computed using the full-graph lower bounds from $\mathrm { F G } _ { \mathrm { r e f } }$ , are 0.83% and 1.16%, respectively.

Table 4: Optimization-level evaluation results.
<table><tr><td colspan="8">Mean relative objective difference to  $\mathrm { F G } _ { \mathrm { r e f } }$  (%)</td></tr><tr><td>City</td><td>FG</td><td>GNN</td><td>LP</td><td>Random</td><td>SR</td><td>BR</td><td>SP</td></tr><tr><td colspan="8">Static regime</td></tr><tr><td>Prague-Dejvice</td><td>1.14</td><td>1.12</td><td>142.00</td><td>4.52</td><td>40.34</td><td>23.85</td><td>422.96</td></tr><tr><td>London</td><td>1.68</td><td>0.54</td><td>3.85</td><td>1.17</td><td>309.25</td><td>201.56</td><td>545.60</td></tr><tr><td>New York</td><td>106.53</td><td>0.67</td><td>16.48</td><td>4.50</td><td>67.10</td><td>5.12</td><td>545.35</td></tr><tr><td>Sydney</td><td>31.06</td><td>0.24</td><td>118.94</td><td>6.34</td><td>43.51</td><td>11.51</td><td>341.89</td></tr><tr><td>All cities</td><td>35.10</td><td>0.64</td><td>70.32</td><td>4.13</td><td>115.05</td><td>60.51</td><td>463.95</td></tr><tr><td colspan="8">Increasing regime</td></tr><tr><td>Prague-Dejvice</td><td>2.85</td><td>2.06</td><td>250.26</td><td>6.53</td><td>39.86</td><td>22.19</td><td>658.67</td></tr><tr><td>London</td><td>3.09</td><td>1.22</td><td>1.25</td><td>1.27</td><td>245.14</td><td>174.18</td><td>517.71</td></tr><tr><td>New York</td><td>87.15</td><td>-1.24</td><td>45.30</td><td>4.23</td><td>56.56</td><td>3.91</td><td>508.18</td></tr><tr><td>Sydney</td><td>49.10</td><td>-0.61</td><td>88.51</td><td>8.78</td><td>38.15</td><td>9.35</td><td>299.07</td></tr><tr><td>All cities</td><td>35.55</td><td>0.36</td><td>96.33</td><td>5.20</td><td>94.93</td><td>52.41</td><td>495.91</td></tr><tr><td colspan="8">Runtime over all test episodes</td></tr><tr><td>Method</td><td>FG</td><td>GNN</td><td>LP</td><td>Random</td><td>SR</td><td>BR</td><td>SP</td></tr><tr><td>Solve time (s)</td><td>9.08</td><td>4.88</td><td>3.02</td><td>6.99</td><td>0.42</td><td>0.99</td><td>&lt; 0.01</td></tr><tr><td>Total time (s)</td><td>9.60</td><td>8.43</td><td>4.24</td><td>7.63</td><td>0.52</td><td>1.13</td><td>0.05</td></tr></table>

Several observations arise when comparing the proposed GNN-based ranking with its Random and LP ablations. First, the comparison with Random shows the proposed network-reduction procedure remains useful regardless of the predicting oracle: Random, indeed, performs substantially better than the fixed reduction methods in nearly all cases and even improves on the 10-second FG runs for the large city networks. However, the GNNbased ranking plays a significant role: for all cities it reduces the mean relative objective diference compared with Random, the best other method in both regimes, by 3.49 and 4.85 percentage points in the static and increasing regimes, respectively. Both diferences are statistically significant (Holmadjusted $p = 0 . 0 0 0 3 6 )$ . Moreover, in cases of New York and Sydney (increasing regime) the GNN even surpasses the $\mathrm { F G } _ { \mathrm { r e f } }$ baseline. We also note that, despite using the same ranking prefix budget, GNN retains only 45% of the flow– segment pairs available in FG, compared with 61% for Random, which is reflected in a lower solving time of 4.88 s versus $6 . 9 9 \ \mathrm { s } ; \mathrm { a }$ part of Random’s performance can therefore be attributed to its larger reduced network, which increases the chance of containing a good solution. Finally, the comparison of GNN with the LP ablation shows GNN’s clear dominance; the values obtained by LP indicate that the fractional primal values of its relaxation provide little useful ranking information for the subsequent integer optimization (in contrast to Dilkina et al. (2009), Section 2.4).

## 5.5. Microscopic simulation evaluation

The final experiment examines how improvements observed at the optimization level translate into trafic outcomes in a microscopic simulation. For every method, we recover vehicle routes from each episode’s MILP solution obtained in Section 5.4 and replay the resulting assignments in SUMO (Alvarez Lopez et al., 2018). For each (city, mode, ${ \cal N } _ { F } )$ combination, all methods use the same scenarios, vehicle departure times, and SUMO settings.

Since the trafic state depends on factors beyond the initial routes, an important question is how to formulate a valid experiment demonstrating how an optimization result leads to simulated trafic performance. For example, without rerouting, vehicles cannot respond to conditions arising during simulation (e.g., accidents, queues, and delays), so initial routes alone cannot realistically determine the outcome. Conversely, frequent route updates repeatedly overwrite the initial routing, so the experiment rather measures the adaptation to current trafic rather than the influence of the initial routes. We therefore use the following procedure: we define the share p of rerouted vehicles and run each method on every scenario for $p \in \{ 0 , 0 . 0 5 , \ldots , 0 . 5 0 \}$ SUMO updates the routes of the selected vehicles every 30 seconds, matching the episode length. The results are then averaged over p as defined below.

We consider a one-hour simulation horizon, matching the period covered by the 120 routing episodes. Although only a share of dispatched vehicles complete their trips within this window, extending the horizon further would increasingly reflect the network’s natural clearance after dispatch ends, as queues dissipate and delayed trips complete regardless of routing quality. This keeps the comparison focused on diferences between methods rather than generic post-dispatch dynamics.

We evaluate trafic performance using the arrival rate and the average trip duration. The arrival rate $A ( p )$ is the percentage of dispatched vehicles that reach their destination within the one-hour operational horizon, and the average trip duration $D ( p )$ is computed over completed trips. Since duration is conditional on arrival, we instead report the ratio $R ( p ) = D ( p ) / A ( p )$ , in seconds per percentage point $\mathrm { ( s / p p ) }$ of arrival. We average both metrics over the eleven rerouting shares:

$$
\overline { { { A } } } = \frac { 1 } { 1 1 } \sum _ { i = 0 } ^ { 1 0 } A ( 0 . 0 5 i ) , \qquad \overline { { { R } } } = \frac { 1 } { 1 1 } \sum _ { i = 0 } ^ { 1 0 } R ( 0 . 0 5 i ) .
$$

We make two remarks on the interpretation of the simulation results. First, the evaluation scenarios are intended as trafic stress tests rather than representations of typical trafic conditions. We deliberately impose high trafic loads: each OD pair accounts for an average of about 500–900 vehicles over the hour, while the average total number of vehicles per scenario (given the selected numbers of OD pairs) is about 21000, 6300, 7700, and 6400 for Prague-Dejvice, London, New York, and Sydney, respectively. Such loads, combined with the one-hour simulation horizon, are expected to lead to low arrival rates, and the resulting SUMO values should therefore not be interpreted as estimates of normal city trafic performance; instead, they provide a basis for comparing the routing methods under the same dificult conditions. Second, the individual results of the methods should be interpreted within the limitations of the experiment. The proposed MILP aims to minimize tiered segment costs, whereas the simulation metrics depend on vehicle interactions, departure times, rerouting, queues, and other trafic conditions that the optimization does not model directly, so an exact correspondence between the two should not be expected. Nevertheless, the experiment is intended to show that the optimization-level gains of the proposed method remain beneficial overall under the simulation settings.

Table 5: Microscopic simulation across rerouting shares. Cells report average arrival (%) / duration-to-arrival (s/pp); metric bests are bold.
<table><tr><td>Method</td><td>Prague-Dejvice</td><td>London</td><td>New York</td><td>Sydney</td><td>All cities</td></tr><tr><td colspan="6">Static regime</td></tr><tr><td>FG</td><td>1.11/401.0</td><td>16.29/47.1</td><td>2.37/589.6</td><td>17.45/43.8</td><td>5.98/130.4</td></tr><tr><td>GNN</td><td>1.17/407.6</td><td>15.83/48.9</td><td>2.44/585.2</td><td>17.89/43.6</td><td>6.02/131.8</td></tr><tr><td>LP</td><td>1.21/518.2</td><td>15.94/48.6</td><td>2.81/571.3</td><td>12.24/85.5</td><td>5.27/177.0</td></tr><tr><td>Random</td><td>0.89/514.8</td><td>16.20/47.1</td><td>2.66/509.8</td><td>17.18/46.2</td><td>5.87/136.1</td></tr><tr><td>SR</td><td>0.45/691.9</td><td>10.63/99.8</td><td>1.75/987.4</td><td>15.36/54.4</td><td>4.40/216.2</td></tr><tr><td>BR</td><td>0.46/678.5</td><td>13.46/68.6</td><td>2.56/593.7</td><td>16.74/48.2</td><td>5.17/172.2</td></tr><tr><td>SP</td><td>0.46/605.7</td><td>9.64/116.4</td><td>1.06/1586.5</td><td>11.43/62.8</td><td>3.54/255.5</td></tr><tr><td colspan="6">Increasing regime</td></tr><tr><td>FG</td><td>0.84/482.5</td><td>13.81/54.7</td><td>2.59/533.6</td><td>16.53/43.7</td><td>5.81/130.4</td></tr><tr><td>GNN</td><td>0.97/454.7</td><td>13.80/54.9</td><td>2.64/513.1</td><td>18.04/40.1</td><td>6.12/123.2</td></tr><tr><td>LP</td><td>1.45/441.4</td><td>13.76/55.2</td><td>3.25/493.6</td><td>13.09/70.6</td><td>5.67/155.8</td></tr><tr><td>Random</td><td>0.93/472.6</td><td>13.86/54.2</td><td>2.89/458.1</td><td>16.13/48.7</td><td>5.85/133.7</td></tr><tr><td>SR</td><td>0.49/659.6</td><td>9.81/99.4</td><td>2.12/782.8</td><td>15.25/51.8</td><td>4.68/191.4</td></tr><tr><td>BR</td><td>0.49/635.7</td><td>11.78/75.3</td><td>2.86/492.9</td><td>16.85/44.9</td><td>5.39/155.7</td></tr><tr><td>SP</td><td>0.50/519.3</td><td>8.32/129.5</td><td>1.40/1165.9</td><td>11.47/57.8</td><td>3.71/230.0</td></tr></table>

Discussion. We first analyze how well the optimization-level results lead to the microscopic trafic performance in general. Table 5 shows that the ordering from Section 5.4 is not maintained precisely across separate methods, but rather preserves a clear diference between groups of methods: FG and the ranking-based approaches demonstrate stronger simulation results, while the fixed reduction methods SR, BR, and SP tend to give lower arrival rates and larger duration-to-arrival ratios. In particular, GNN and LP obtain higher arrival rates in the All cities results than every fixed reduction in both trafic regimes.

Among the individual methods, GNN demonstrates the strongest overall performance across cities: it shows the best arrival rate in both trafic regimes, while also ranking first in the increasing regime by R and second in the static regime by the same metric. We remark that the diferences are statistically significant only for the increasing regime for both arrival rate (the best other method is Random) and duration-to-arrival ratio (the best other method is

FG), with Holm-adjusted $p = 0 . 0 0 0 3 6$ for both comparisons. The GNN’s performance is also reflected across individual cities: it stays among the topthree methods in most city/regime combinations – five of eight by arrival rate and six of eight by R. In addition, the GNN performs remarkably in Sydney, where it achieves the highest arrival rate (17.89% static, 18.04% increasing) together with the lowest duration-to-arrival ratio in both the static (43.6 s/pp) and increasing (40.1 s/pp) regimes. Thus, the results confirm that GNN’s stronger optimization-level performance also translates into stronger overall simulation performance across cities.

## 6. Conclusion

This paper addressed the computational dificulty of solving the CTR MILP over an urban road network within short operational time budgets. We proposed a network reduction procedure that exploits not only the topology of the road network but also information about the vehicles present in the current episode. A dedicated GNN ranks road segments separately for each OD flow, and the highest-ranked segments are retained to form a reduced network for solving the segment-flow MILP. The trained GNN thus narrows the search space available to optimization without predicting the routing itself, while the MILP ensures feasibility and determines the final routing under the same objective.

Unlike fixed reductions, which reuse the same reduced road network across all episodes, the proposed method constructs a separate reduced network for every episode: highly ranked segments are incorporated into complete origin-destination paths, and a local expansion adds neighboring segments. Since the construction is agnostic to the source of the segment scores, it can also use other rankings, e.g., those obtained from the linear programming relaxation of the considered MILP or even from a random ordering. This allows us to separate the contribution of the reduction procedure from that of the learned ranking.

We evaluated the proposed approach on four real urban road-network topologies under two trafic regimes. The scalability experiment identified ranges of OD flows pairs where the full-graph formulation is dificult to solve within the operational budget, but additional solver time yields substantially better solutions. Thus, these ranges provide both a motivation for network reduction and suitable training instances for the GNN. Across the eight trained models, the validation accuracy ranges from 92.0% to 96.1%. Within each city and trafic regime, the same model remains efective across five consecutive problem sizes and previously unseen scenarios. At the optimization level, GNN achieves the smallest relative objective diference from the 30-second full-graph reference in every city and trafic regime, averaging at 0.4–0.6% across cities. The comparison with fixed reduction methods shows that the proposed reduction procedure improves solution quality already on its own; nevertheless, among the methods using this procedure, GNN demonstrates a noticeable advantage over both Random and LP, confirming the substantial contribution of the learned ranking. The final experiment, which evaluates how the optimization results translate into simulation performance, shows that GNN also obtains the highest average arrival rate in both trafic regimes.

An important limitation concerns the connection between the optimization model and trafic simulation. The considered MILP represents congestion through tiered segment costs, whereas SUMO additionally models trafic lights, queues, interactions between vehicles, lane changes, and rerouting over time. The optimization model therefore captures only a limited part of the resulting trafic dynamics. Although the simulation experiment shows that the gains observed at the optimization level remain useful under the considered simulation settings, it does not establish that every improvement in the MILP objective necessarily improves trafic performance. A natural direction for future work is to make the approach more responsive to the current trafic situation by incorporating information about trafic lights, queues, and observed travel times and by updating the ranking and routing as trafic conditions change. This would bring the optimization model closer to real-time trafic management while retaining the feasibility guarantees of the MILP.

## Acknowledgements

This research is supported by Czech Science Foundation (project no. 23- 05575S and 25-18003S), by the European Union under the OP JAK project ROBOPROX (reg. no. CZ.02.01.01/00/22\_008/0004590), and by the Grant Agency of the Czech Technical University (project no. SGS26/188/OHK3/ 3T/37 and project no. SGS25/144/OHK3/3T/13).

## References

Alvarez Lopez, P., Behrisch, M., Bieker-Walz, L., Erdmann, J., Flötteröd, Y.P., Hilbrich, R., Lücken, L., Rummel, J., Wagner, P., Wießner, E., 2018.

Microscopic trafic simulation using SUMO, in: The 21st IEEE International Conference on Intelligent Transportation Systems, IEEE. pp. 2575–2582. doi:10.1109/ITSC.2018.8569938.

Ambrogio, A., Cianfanelli, L., Como, G., 2024. Information design for congestion minimization in transportation networks. IFAC-PapersOnLine 58, 181–185. URL: http://dx.doi.org/10.1016/j.ifacol.2025.01.17 7, doi:10.1016/j.ifacol.2025.01.177.

Angelelli, E., Morandi, V., Savelsbergh, M., Speranza, M., 2021. System optimal routing of trafic flows with user constraints using linear programming. European Journal of Operational Research 293, 863– 879. URL: http://dx.doi.org/10.1016/j.ejor.2020.12.043, doi:10.1016/j.ejor.2020.12.043.

Angioni, D., Archetti, C., Speranza, M.G., 2025. Neural combinatorial optimization: A tutorial. Computers & Operations Research 182, 107102. URL: https://doi.org/10.1016/j.cor.2025.107102, doi:10.1016/j. cor.2025.107102.

Bagabaldo, A.R., Gan, Q., Bayen, A.M., González, M.C., 2024. Impact of navigation apps on congestion and spread dynamics on a transportation network. Data Science for Transportation 6, 12. URL: http://dx.doi.o rg/10.1007/s42421-024-00099-w, doi:10.1007/s42421-024-00099-w.

Bengio, Y., Lodi, A., Prouvost, A., 2021. Machine learning for combinatorial optimization: A methodological tour d’horizon. European Journal of Operational Research 290, 405–421. URL: http://dx.doi.org/10.1016 /j.ejor.2020.07.063, doi:10.1016/j.ejor.2020.07.063.

Bosch, R., van Heeswijk, W., Rogetzer, P., Mes, M., 2025. Machine learning predictions for trafic equilibria in road renovation scheduling. URL: https: //arxiv.org/abs/2506.05933, doi:10.48550/ARXIV.2506.05933.

Brody, S., Alon, U., Yahav, E., 2022. How attentive are graph attention networks?, in: International Conference on Learning Representations (ICLR). URL: https://openreview.net/forum?id=F72ximsx7C1.

Chen, B.Y., Lam, W.H., Sumalee, A., Li, Q., Li, Z.C., 2012. Vulnerability analysis for large-scale and congested road networks with demand

uncertainty. Transportation Research Part A: Policy and Practice 46, 501–516. URL: http://dx.doi.org/10.1016/j.tra.2011.11.018, doi:10.1016/j.tra.2011.11.018.

Chrpa, L., Vallati, M., 2023. Centralised vehicle routing for optimising urban trafic: A scalability perspective, in: 2023 IEEE Intelligent Vehicles Symposium (IV), IEEE. pp. 1–6. URL: http://dx.doi.org/10.1109/I V55152.2023.10186707, doi:10.1109/iv55152.2023.10186707.

Chrpa, L., Vallati, M., Parkinson, S., 2019. Exploiting automated planning for eficient centralized vehicle routing and mitigating congestion in urban road networks, in: Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing (SAC 2019), Association for Computing Machinery. pp. 191–194. doi:10.1145/3297280.3297539.

Colini-Baldeschi, R., Cominetti, R., Mertikopoulos, P., Scarsini, M., 2020. When is selfish routing bad? the price of anarchy in light and heavy trafic. Operations Research 68, 411–434. URL: http://dx.doi.org/10.1287/o pre.2019.1894, doi:10.1287/opre.2019.1894.

Cominetti, R., Dose, V., Scarsini, M., 2024. The price of anarchy in routing games as a function of the demand. Mathematical Programming 203, 531–558. URL: http://dx.doi.org/10.1007/s10107-021-01701-7, doi:10.1007/s10107-021-01701-7.

Como, G., Lovisari, E., Savla, K., 2016. Convexity and robustness of dynamic trafic assignment and freeway network control. Transportation Research Part B: Methodological 91, 446–465. URL: http://dx.doi.org/10.1016 /j.trb.2016.06.007, doi:10.1016/j.trb.2016.06.007.

Dilkina, B., Gomes, C.P., Malitsky, Y., Sabharwal, A., Sellmann, M., 2009. Backdoors to combinatorial optimization: Feasibility and optimality, in: Integration of AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems. Springer Berlin Heidelberg, pp. 56–70. doi:10.1007/978-3-642-01929-6\_6.

Ekström, J., Sumalee, A., Lo, H.K., 2012. Optimizing toll locations and levels using a mixed integer linear approximation approach. Transportation Research Part B: Methodological 46, 834–854. URL: http://dx.doi.org /10.1016/j.trb.2012.02.006, doi:10.1016/j.trb.2012.02.006.

Ferber, A., Song, J., Dilkina, B., Yue, Y., 2022. Learning pseudo-backdoors for mixed integer programs, in: Schaus, P. (Ed.), Integration of Constraint Programming, Artificial Intelligence, and Operations Research, Springer, Cham. pp. 91–102. URL: https://link.springer.com/chapter/10.100 7/978-3-031-08011-1\_8, doi:10.1007/978-3-031-08011-1\_8.

Gañan-Cardenas, E., Rios-Echeverri, D.C., Ballesteros, J.R., Branch-Bedoya, J.W., 2024. Estimating trafic congestion cost uncertainty using a bootstrap scheme. Transportation Research Part D: Transport and Environment 136, 104462. URL: http://dx.doi.org/10.1016/j.trd.2024.104462, doi:10.1016/j.trd.2024.104462.

Gasse, M., Chételat, D., Ferroni, N., Charlin, L., Lodi, A., 2019. Exact combinatorial optimization with graph convolutional neural networks, in: Advances in Neural Information Processing Systems. URL: https://pape rs.neurips.cc/paper\_files/paper/2019/hash/d14c2267d848abeb81 fd590f371d39bd-Abstract.html.

Gauthier, P., Furno, A., El Faouzi, N.E., 2018. Road network resilience: How to identify critical links subject to day-to-day disruptions. Transportation Research Record: Journal of the Transportation Research Board 2672, 54–65. URL: http://dx.doi.org/10.1177/0361198118792115, doi:10.1 177/0361198118792115.

Gerbaux, J., Desaulniers, G., Cappart, Q., 2025. A machine-learning-based column generation heuristic for electric bus scheduling. Computers & Operations Research 173, 106848. URL: https://doi.org/10.1016/j.co r.2024.106848, doi:10.1016/j.cor.2024.106848.

Gunjo, S.B., Guta, D.D., Damene, S., 2024. Modeling the economic cost of congestion in Addis Ababa City, Ethiopia. Environmental Systems Research 13, 16. URL: http://dx.doi.org/10.1186/s40068-024-00344-9, doi:10.1186/s40068-024-00344-9.

Jahn, O., Möhring, R.H., Schulz, A.S., Stier-Moses, N.E., 2005. Systemoptimal routing of trafic flows with user constraints in networks with congestion. Operations Research 53, 600–616. URL: http://dx.doi.org /10.1287/opre.1040.0197, doi:10.1287/opre.1040.0197.

Johnn, S.N., Darvariu, V.A., Handl, J., Kalcsics, J., 2024. A graph reinforcement learning framework for neural adaptive large neighbourhood search. Computers & Operations Research 172, 106791. URL: https://doi.org/ 10.1016/j.cor.2024.106791, doi:10.1016/j.cor.2024.106791.

Joshi, C.K., Laurent, T., Bresson, X., 2019. An eficient graph convolutional network technique for the travelling salesman problem. URL: https://arxiv.org/abs/1906.01227, doi:10.48550/ARXIV.1906.01227. arXiv:1906.01227.

Ke, Z., Zou, Q., Liu, J., Qian, S., 2025. Real-time system optimal trafic routing under uncertainties — can physics models boost reinforcement learning? Transportation Research Part C: Emerging Technologies 173, 105040. URL: http://dx.doi.org/10.1016/j.trc.2025.105040, doi:10 .1016/j.trc.2025.105040.

Khreis, H., Williams, H., Abdollahpour, S.S., van den Bosch, M., Mudu, P., Tainio, M., Poom, A., Sohrabi, S., Hankey, S., 2024. The nexus of transportation, the built environment, air pollution and health. Cities & Health , 1–20URL: http://dx.doi.org/10.1080/23748834.2024.23763 89, doi:10.1080/23748834.2024.2376389.

Lassen, O.B., Agriesti, S., Eldafrawi, M., Gammelli, D., Cantelmo, G., Gentile, G., Pereira, F.C., 2025. Learning trafic flows: Graph neural networks for metamodelling trafic assignment. URL: https://arxiv.org/abs/2505.1 1230, doi:10.48550/ARXIV.2505.11230.

Liu, C., Dong, Z., Ma, H., Luo, W., Li, X., Pang, B., Zeng, J., Yan, J., 2024. L2P-MIP: Learning to presolve for mixed integer programming, in: The Twelfth International Conference on Learning Representations. URL: https://openreview.net/forum?id=McfYbKnpT8.

Long, J., Szeto, W.Y., 2019. Link-based system optimum dynamic trafic assignment problems in general networks. Operations Research 67, 167–182. URL: http://dx.doi.org/10.1287/opre.2018.1775, doi:10.1287/opre .2018.1775.

Maheshwari, C., Kulkarni, K., Pai, D., Yang, J., Wu, M., Sastry, S., 2024. Congestion pricing for eficiency and equity: Theory and applications to

the San Francisco Bay Area. URL: https://arxiv.org/abs/2401.16844, doi:10.48550/ARXIV.2401.16844.

Monnot, B., Benita, F., Piliouras, G., 2022. Routing games in the wild: Eficiency, equilibration, regret, and a price of anarchy bound via long division. ACM Transactions on Economics and Computation 10, 1–26. URL: http://dx.doi.org/10.1145/3512747, doi:10.1145/3512747.

Morandi, V., 2024. Bridging the user equilibrium and the system optimum in static trafic assignment: a review. 4OR 22, 89–119. URL: http: //dx.doi.org/10.1007/s10288-023-00540-w, doi:10.1007/s10288-0 23-00540-w.

Nyporko, A., Švadlenka, M., Antonov, N., Rohaninejad, M., Chrpa, L., 2025. Centralised urban trafic routing using mixed-integer programming, in: Proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART, INSTICC. SciTePress. pp. 176–187. doi:10.5220/0013137100003890.

Nyporko, A., Švadlenka, M., Antonov, N., Rohaninejad, M., Chrpa, L., 2026. Multicommodity flow based centralized trafic routing, in: van den Herik, H.J., Rocha, A.P., Steels, L. (Eds.), Agents and Artificial Intelligence, Springer, Cham. pp. 40–64. doi:10.1007/978-3-032-25032-2\_3.

Quesnel, F., Wu, A., Desaulniers, G., Soumis, F., 2022. Deep-learningbased partial pricing in a branch-and-price algorithm for personalized crew rostering. Computers & Operations Research 138, 105554. URL: https://doi.org/10.1016/j.cor.2021.105554, doi:10.1016/j.cor.20 21.105554.

Roughgarden, T., Tardos, É., 2002. How bad is selfish routing? Journal of the ACM 49, 236–259. URL: http://dx.doi.org/10.1145/506147.506153, doi:10.1145/506147.506153.

Serok, N., Havlin, S., Blumenfeld Lieberthal, E., 2022. Identification, cost evaluation, and prioritization of urban trafic congestions and their origin. Scientific Reports 12, 13026. URL: http://dx.doi.org/10.1038/s4159 8-022-17404-8, doi:10.1038/s41598-022-17404-8.

Švadlenka, M., Chrpa, L., 2023. Towards a framework for intelligent urban trafic routing. The International FLAIRS Conference Proceedings 36. URL:

http://dx.doi.org/10.32473/flairs.36.133321, doi:10.32473/flair s.36.133321.

Švadlenka, M., Chrpa, L., Vallati, M., 2023. Improving the scalability of automated planning-based vehicle routing via smart routes identification, in: 2023 8th International Conference on Models and Technologies for Intelligent Transportation Systems (MT-ITS), IEEE. pp. 1–6. URL: http: //dx.doi.org/10.1109/MT-ITS56129.2023.10241639, doi:10.1109/mt -its56129.2023.10241639.

Vallati, M., Chrpa, L., 2020. Reducing trafic congestion in urban areas via real-time re-routing: A simulation study, in: AI 2020: Advances in Artificial Intelligence. Springer International Publishing, pp. 69–81. URL: http://dx.doi.org/10.1007/978-3-030-64984-5\_6, doi:10.1007/97 8-3-030-64984-5\_6.

Vallati, M., Scala, E., Chrpa, L., 2021. A hybrid automated planning approach for urban real-time routing of connected vehicles, in: 2021 IEEE International Intelligent Transportation Systems Conference (ITSC), IEEE. pp. 3821–3826. URL: http://dx.doi.org/10.1109/ITSC48978.2021.95 64838, doi:10.1109/itsc48978.2021.9564838.

Wang, J., Mao, Y., Li, J., Xiong, Z., Wang, W.X., 2015. Predictability of road trafic and congestion in urban areas. PLOS ONE 10, e0121825. URL: http://dx.doi.org/10.1371/journal.pone.0121825, doi:10.1371/jo urnal.pone.0121825.

Wang, L., Duan, P., Lyu, C., Ma, Z., 2025. Reinforcement learning-based sequential route recommendation for system-optimal trafic assignment. URL: https://arxiv.org/abs/2505.20889, doi:10.48550/ARXIV.2505. 20889.

Youn, H., Gastner, M.T., Jeong, H., 2008. Price of anarchy in transportation networks: Eficiency and optimality control. Physical Review Letters 101, 128701. URL: http://dx.doi.org/10.1103/PhysRevLett.101.128701, doi:10.1103/physrevlett.101.128701.

Yun, H., Kim, E.j., Ham, S.W., Kim, D.K., 2024. Navigating the noncompliance efects on system optimal route guidance using reinforcement learning. Transportation Research Part C: Emerging Technologies 165,

104721. URL: http://dx.doi.org/10.1016/j.trc.2024.104721, doi:10 .1016/j.trc.2024.104721.

Zhang, J., Pourazarm, S., Cassandras, C.G., Paschalidis, I.C., 2018. The price of anarchy in transportation networks: Data-driven evaluation and reduction strategies. Proceedings of the IEEE 106, 538–553. URL: http: //dx.doi.org/10.1109/JPROC.2018.2790405, doi:10.1109/jproc.2018 .2790405.

Zhang, K., Zhang, H., Cheng, Q., Chen, X., Wang, Z., Liu, Z., 2023. A customized two-stage parallel computing algorithm for solving the combined modal split and trafic assignment problem. Computers & Operations Research 154, 106193. URL: http://dx.doi.org/10.1016/j.cor.2023. 106193, doi:10.1016/j.cor.2023.106193.

Zhang, P., Qian, S., 2020. Path-based system optimal dynamic trafic assignment: A subgradient approach. Transportation Research Part B: Methodological 134, 41–63. URL: http://dx.doi.org/10.1016/j.trb.2 020.02.004, doi:10.1016/j.trb.2020.02.004.