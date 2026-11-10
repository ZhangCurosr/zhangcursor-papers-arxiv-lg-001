# DeFSM: Robust and Eficient Decentralization of Federated Submodular Maximization

Duc A. Tran, Dung Truong<sup>∗</sup>, Duy Le

Department ofComputer Science, University ofMassachusetts Boston, Boston, MA, USA

## Abstract

Submodular maximization seeks a small subset best representing a large dataset. In its federated setting, the server is a single point of failure and some clients may be Byzantine. Removing the server raises a trilemma of decentralization, eficiency, and security. We propose DeFSM, a Byzantine-robust and communication-eficient architecture that replaces the server with a lightweight smart contract and a pool of independent, consensus-free agents. Its aggregation rule, SafeGM, builds on two notions from combinatorial geometry: Tukey depth and safe points (the points lying in the convex hull of the good points wherever the bad points are). SafeGM makes the robust decision on 2D Gaussian random projections, which the contract can compute eficiently, and lifts it back as a convex combination of actual updates. Since lifting loses information, it uses the projected geometric median only when its depth certifies it, and otherwise a Tukey median. The share of bad updates is then bounded only by a count: how many there are, not what they contain. We prove, and confirm empirically, that this bound controls the solution quality. On CIFAR10 and PathMNIST, with up to 30% Byzantine clients and agents, DeFSM never collapses and retains on average abou 91% of the quality of the ideal reference, the geometric median at a trusted central server. The geometric median decentralized in its best setting would retain about 96%, but the contract cannot compute it, and so each agent would have to download six times more data, or each client twice as much.

Keywords: Submodular maximization, Federated learning, Blockchain, Decentralized computing, Byzantine robustness

## 1. Introduction

Edge devices generate vast datasets that can fuel machine learning for predictive analytics, anomaly detection, and resource optimization. Transmitting all of these data, however, strains bandwidth and the central server, and labeling them requires substantial labor. Since data are often redundant, with additional samples adding less benefit, we should collect only the most informative ones. This is one of many applications of Federated Submodular Maximization (FSM), the federated setting of submodular maximization, a classic problem with diminishing marginal returns. FSM is analogous to Federated Learning (FL), except that the unknown is a subset and the objective is a submodular set function. Whereas FL has gained great traction, FSM research is rare [1, 2, 3].

An FSM system consists of local machines, the clients, and a central aggregation server. In the basic protocol for learning top-r elements from a federated dataset [1], the server broadcasts a fractional global solution x, initially zero. Each client returns, as a binary vector, the r elements it would add under its local objective around x. The server aggregates these vectors, typically by averaging, and moves x a small step in the aggregated direction. After a fixed number of rounds, x is rounded to the final subset.

The federated setting requires a central server. In practice, such a server may not be feasible due to limited resources, lack of trust, or the risk of a single point of failure. Also, the solution quality relies on the clients following the protocol, yet they can submit fake local updates.

Problem. We address a new problem: decentralized FSM without a server and in the presence of Byzantine clients. As for decentralization, a smart contract on a blockchain can replace the server logic verifiably, but it can aford only light computation, communication, and storage. The aggregation must therefore be outsourced to ofchain workers, which we call agents. However, some agents may themselves be Byzantine. No FSM research exists to date in this direction. As for client misbehavior, robust aggregators exist for federated learning [4, 5, 6], but robust FSM is underexplored [3]. How well these aggregators work when decentralized in server-less FSM is unknown.

Challenges. Robust aggregators operate on high-dimensional vectors, which the contract cannot aford, so the agents must run them entirely of-chain. Each agent then downloads all other agents’ aggregates in every round, hence incurring substantial trafic. Moreover, the agents need to agree on a final aggregate, but vector consensus in high dimensions is intractable in general [7]. Decentralization, eficiency, and security thus form a trilemma.

Contributions. We propose DeFSM, a decentralized architecture for FSM, with its aggregation rule SafeGM. Together they resolve the trilemma, and no party needs to know how many clients or agents are Byzantine. The key idea is safety: an aggregate is safe if it lies in the convex hull of the good updates. Tukey depth can certify this: a point is safe if its depth exceeds the number of bad points. SafeGM projects the updates onto a random 2D plane and computes their geometric median; if its depth falls below our design threshold, a Tukey median is used instead. This 2D point is then lifted back to the original space R<sup>d</sup> as a convex combination of actual updates, in which bad updates receive a share bounded by their count. Safety controls the solution quality because of two properties specific to FSM, which federated learning lacks: monotone objectives and sparse binary updates. Our contributions:

• Novel architecture: DeFSM is, to our knowledge, the first FSM architecture without a trusted server. Aggregation is outsourced to a pool of independent agents, a role anyone, even a client, can take, and coordinated by a smart contract so light that any blockchain can host it: it stores only hashes, seeds, and a few numbers per agent. Clients share their updates through IPFS, and each round the contract itself, not a vote among agents, makes the robust decision.

• Provable robustness: Today’s robust aggregators judge updates by distances, which an attacker can shape. Our aggregation rule, SafeGM, ofers a guarantee that no manipulation of data can break: the share of bad updates in its aggregate is bounded only by how many there are, not by what they contain. We prove, and confirm empirically, that this bound controls the solution quality end to end, recovering the classic (1 − 1/e) guarantee up to client heterogeneity.

• Decentralization eficiency: We avoid the need for the agents to download all agent aggregates and reach consensus among themselves. Because SafeGM decides in a random 2D projection and lifts the decision back, the contract can compute the decision easily from only two numbers per agent. Each client downloads only the few aggregates the contract selects. Our work makes classical depth-based robustness usable in high dimension for the first time.

• Empirical evidence: On CIFAR10 and PathMNIST, with up to 30% Byzantine clients and agents, DeFSM retains on average about 91% of the quality of the geometric median, the most accurate robust aggregator in practice, at a trusted server, and never collapses, even under extreme heterogeneity. Decentralizing the geometric median, even in an idealized setting, reaches 96% but costs six times more download per agent, or twice as much for clients; the mean and trimmed mean sometimes fall much lower.

Related work is reviewed in Section 2 and FSM background in Section 3. DeFSM is detailed in Section 4, analyzed in Section 5, and evaluated in Section 6. Section 7 concludes the paper.

## 2. Related Work

Submodular Maximization (SM) is useful in machine learning, operations research, social networks, and network design, with applications such as active learning, feature selection, exemplar clustering, document summarization, object recognition, viral marketing, social welfare, sensor placement, and robotic data collection.

Conventional SM. SM has been studied under many constraints, e.g., none, cardinality, knapsack, and non-linear constraints. The celebrated greedy algorithm of Nemhauser et al. [8] achieves the (1 − 1/e) approximation ratio in polynomial time, the best possible unless P=NP [9]. Distributed SM has also been explored [10, 11, 1, 3, 12], mostly for computational eficiency under homogeneous conditions.

Federated SM. FSM research is rare [1, 2, 3]. Rafiey [1] proposed what appears to be the first FSM algorithm, a federated continuous greedy method, and proved its convergence. Wang et al. [2] addressed client privacy in FSM. Tran et al. [3] proposed an FSM design robust to practical client attacks, the only work on Byzantine-robust FSM to date. Client risks are widely studied in Federated Learning (FL), mostly as backdoor attacks [13], but FSM faces attacks unseen in FL that require new defenses [3].

Decentralized FSM. When the server is untrustworthy or cannot be deployed, blockchain is a natural candidate to decentralize aggregation. No prior work uses blockchain for FSM, but its adoption in FL, though early, is rising [14, 15, 16, 17, 18, 19]. Existing designs build a blockchain network dedicated to FL or implement the server logic as a smart contract. Neither scales to the workloads of federated computation and storage. DeFSM instead outsources aggregation and storage to decentralized agents, with minimal contract overhead.

Robust aggregation and vector consensus. Byzantine-robust aggregation rules for $\mathrm { F L } .$ , such as Krum [6], the coordinate-wise trimmed mean [5] and median [20], the geometric median [4], and median-of-means [21], work on full d-dimensional updates, rely on distances between updates or on coordinate-wise order statistics, and favor proximity to a majority; Krum and the trimmed mean also need the number of Byzantine updates as a parameter. Tailored attacks exploit this preference by keeping bad updates close to good ones while steering the aggregate [22, 23, 24]. In FSM, heterogeneous good clients’ local solutions are not mutually close, so proximity is an even weaker sign of a good update [3]. Byzantine vector consensus (BVC) [7, 25] instead requires the agreed vector to lie in the convex hull of the good agents’ inputs, a distance-free guarantee, but is solvable only in very low dimension. Such guarantees rely on deep points, whose Tukey depth exceeds the number of bad inputs; they exist by the centerpoint theorem [26] and are eficiently computable in the 2D plane [27]. We make deep points usable for high-dimensional federated aggregation by deciding in 2D projections and bounding the error of lifting back. Unlike BVC, the agents in DeFSM never exchange messages to agree: the smart contract computes the deep point.

## 3. Preliminaries

Consider a utility function $f : 2 ^ { E } $ R defined on all subsets of a finite ground set $E .$ Function $f$ is submodular if for any subset $S \subset E$ and any elements $e _ { 1 } , e _ { 2 } \in E$ we have

$$
f ( S \cup \{ e _ { 2 } \} ) - f ( S ) \geq f ( S \cup \{ e _ { 1 } , e _ { 2 } \} ) - f ( S \cup \{ e _ { 1 } \} ) .
$$

That is, $f$ exhibits diminishing returns: the marginal gain of adding an element decreases as the context set grows.

## 3.1. Centralized Submodular Maximization (SM)

The goal of SM is to find a subset $S \subset E$ that maximizes f(S ) subject to a set constraint $\mathcal { T } \subset 2 ^ { E }$

$$
\operatorname* { m a x } _ { s } \{ f ( S ) : S \in { \mathcal { I } } \} .
$$

Ground set E may be a collection of text documents for summarization or unlabeled images for annotation, while $f$ evaluates the quality of the chosen subset. Constraint I is often a matroid; for example, the cardinality constraint $J = \{ S \subset E : | S | \leq r \}$ is a matroid, whereas the knapsack constraint $\begin{array} { r } { \mathcal { I } \ : = \ : \{ S \ : \subset \ : E \ : \mid \ : \sum _ { s \in S } w _ { s } \ : \leq \ : b \} } \end{array}$ is not. Let $r \triangleq \operatorname* { m a x } \{ | I | : I \in I \}$ be the rank of constraint matroid ${ \boldsymbol { \mathcal { I } } } .$

Continuous greedy algorithm. SM is strongly NP-hard, but it can be solved approximately up to an optimal approximation ratio of $\left( 1 - { \begin{array} { l } { { \frac { 1 } { \rho } } } \end{array} } \right) \left[ 9 \right]$ . This is achieved for the cardinality constraint by Nemhauser’s seminal algorithm [8]. For arbitrary matroid constraints, the approach is to convert the discrete set function to a continuous version so that a gradient ascent method can be applied [28, 29]. A subset $S \subset E$ is represented by its indicator vector $\mathbf { x } _ { S } \in [ 0 , 1 ] ^ { | E | }$ with $\mathbf { x } [ e ] = 1 \Leftrightarrow e \in S$ ; conversely, an arbitrary vector $\mathbf { x } \in [ 0 , 1 ] ^ { | E | }$ corresponds to a random set S containing each e independently with probability x[e]. The multilinear extension of $f$ is

$$
F ( \mathbf { x } ) \triangleq \mathbb { E } _ { S \sim \mathbf { x } } [ f ( S ) ] = \sum _ { S \subset E } f ( S ) \prod _ { e \in S } \mathbf { x } [ e ] \prod _ { e \notin S } ( 1 - \mathbf { x } [ e ] ) .
$$

Because $f ( S ) = F ( { \bf x } _ { S } )$ , maximizing f over sets becomes maximizing $F$ over vectors. The gradient of $F$ has coordinates $\partial F / \partial \mathbf { x } [ e ] = \mathbb { E } _ { S \sim \mathbf { x } } [ f ( S \cup \{ e \} ) - f ( S \cup \{ e \} ) ]$ ], which can be estimated by sampling $f .$ Starting from $\mathbf { X } ^ { ( 0 ) } = \mathbf { 0 }$ , the algorithm iterates, for $t = 0 , 1 , \ldots ,$

$$
\mathbf { w } ^ { ( t ) } = \arg \operatorname* { m a x } _ { \mathbf { w } \in \mathcal { P } } \langle \mathbf { w } , \nabla F ( \mathbf { x } ^ { ( t ) } ) \rangle , \qquad \mathbf { x } ^ { ( t + 1 ) } = \mathbf { x } ^ { ( t ) } + \eta \mathbf { w } ^ { ( t ) } ,
$$

where $\eta \in ( 0 , 1 )$ is the learning rate and $\mathcal { P }$ is the matroid polytope, the convex hull of the indicator vectors of the independent sets of I. After $T = 1 / \eta$ steps, x is a convex combination of points of $\mathcal { P }$ and hence feasible. For the cardinality constraint $\mathcal { T } = \{ S \subset E : | S | \leq r \}$ , with $d \triangleq | E |$

$$
\begin{array} { r } { \mathcal { P } = \big \{ \mathbf { x } \in [ 0 , 1 ] ^ { d } : \sum _ { e \in E } \mathbf { x } [ e ] \leq r \big \} . } \end{array}
$$

Its vertices with exactly r ones,

$$
\begin{array} { r } { \mathcal { V } _ { r } \triangleq \Big \{ \mathbf { w } \in \{ 0 , 1 \} ^ { d } : \sum _ { e } \mathbf { w } [ e ] = r \Big \} , } \end{array}
$$

are the indicator vectors of the r-subsets of $E ;$ for monotone objectives, the linear maximization above can always be solved by a vector of $\mathcal { V } _ { r }$ . The final subset can be taken as the r largest coordinates of x, or obtained by pipage rounding [29], which preserves the approximation guarantee in expectation.

## 3.2. Federated Submodular Maximization (FSM)

Consider a standard federated setting with a server and N clients $C = \{ 1 , \ldots , N \}$ , each identified with its index. Client i has a local submodular function $f _ { i } : 2 ^ { E } \to \mathbb { R }$ to value a candidate subset S . The objective is to find a subset of E under constraint I that maximizes the average of these local functions, i.e.,

$$
\operatorname* { m a x } _ { S \in \mathcal { T } } f ( S ) \triangleq \frac { 1 } { N } \sum _ { i \in C } f _ { i } ( S ) .
$$

Objective $f$ is submodular as an average of submodular functions. Ground set E and constraint I are globally known, but $f _ { i }$ is private to client i, and clients communicate only with the server. The continuous greedy algorithm for SM is extended for FSM as follows [1].

Federated continuous greedy algorithm. Let F and $F _ { i }$ be the multilinear extensions of $f$ and $f _ { i } ,$ respectively. FSM runs $T$ “aggregation rounds”. In round t, it keeps a global solution $( \mathbf { x } ^ { ( t ) } , \mathbf { w } ^ { ( t ) } )$ at the server and a local one $( \mathbf { x } _ { i } ^ { ( t ) } , \mathbf { w } _ { i } ^ { ( t ) } )$ at each client $i ,$ all zero at $t = 0$ . To begin a new round $t = 1 , 2 , . . . ,$ the server broadcasts the global gradient $\mathbf { w } ^ { ( t - 1 ) }$ to a subset ${ \cal C } ^ { ( t ) } \subset { \cal C }$ of clients (typically a random subset of fixed size). Upon receipt, each client $i \in C ^ { ( t ) }$ runs one greedy step to improve the solution using local continuous function $F _ { i } \colon$

$$
\mathbf { x } ^ { ( t - 1 ) } = \mathbf { x } ^ { ( t - 2 ) } + \eta \mathbf { w } ^ { ( t - 1 ) } , \quad \mathbf { w } _ { i } ^ { ( t ) } = \arg \operatorname* { m a x } _ { \mathbf { w } \in \mathcal { P } } \mathbf { w } , \nabla F _ { i } ( \mathbf { x } ^ { ( t - 1 ) } ) \rangle , \quad \mathbf { x } _ { i } ^ { ( t ) } = \mathbf { x } ^ { ( t - 1 ) } + \eta \mathbf { w } _ { i } ^ { ( t ) } .\tag{1}
$$

Then client i uploads its (polytope-projected) gradient vector $\mathbf { w } _ { i } ^ { ( t ) }$ to the server. Having received these vectors from all clients, the server updates the global solution:

$$
\mathbf { w } ^ { ( t ) } = \mathsf { m e a n } ( \{ \mathbf { w } _ { i } ^ { ( t ) } : i \in C ^ { ( t ) } \} ) , \qquad \mathbf { x } ^ { ( t ) } = \mathbf { x } ^ { ( t - 1 ) } + \eta \mathbf { w } ^ { ( t ) } .\tag{2}
$$

The new gradient aggregate $\mathbf { w } ^ { ( t ) }$ is broadcast to clients again in round $( t + 1 )$

## 4. DeFSM: Decentralized FSM with SafeGM

FSM is exposed to misbehavior by both the server and the clients. We propose DeFSM, a server-less implemen tation of FSM in which the server is replaced with a lightweight smart contract and a pool of independent agents. The central question is how to compute the global aggregate of $\operatorname { E q . } \left( 2 \right)$ in the presence of Byzantine clients and Byzantine agents, with a smart contract that can aford only light computation. The arithmetic mean is not robust, and robust aggregators work in $\mathbb { R } ^ { d }$ , which the contract cannot aford. DeFSM therefore makes the robust decision in a 2D projection and lifts it back; safety, defined next, is what makes the lift trustworthy.

## 4.1. Threat Model

Participants. We call uncompromised (compromised) clients good (bad); likewise for agents. There are M agents, indexed like the clients, with set $\mathcal { A } = \{ 1 , \ldots , M \}$ . We mark good with + and bad with $- \colon C ^ { + } , C ^ { - } \subseteq C$ are the sets of good and bad clients and $N ^ { + } , N ^ { - }$ their numbers; likewise, ${ \mathcal { A } } ^ { + } , { \mathcal { A } } ^ { - } \subseteq { \mathcal { A } }$ are the sets of good and bad agents and $M ^ { + } , M$ <sup>−</sup> their numbers. The Byzantine client rate is $\beta _ { c } = N ^ { - } / N$ and the Byzantine agent rate is $\beta _ { a } = M ^ { - } / M$ . In a given round, let $\mathbf { W } = \{ \mathbf { w } _ { j } \} _ { j = 1 } ^ { N }$ be the submitted local updates, $\mathbf { w } _ { j }$ being the update of client $j ;$ for brevity, we also write $C ^ { + }$ and $C ^ { - }$ for the sets of good and bad updates, $\{ \mathbf { w } _ { j } : j \in C ^ { + } \}$ and $\{ \mathbf { w } _ { j } : j \in C ^ { - } \}$

Adversary. The adversary knows all protocol logic and coordinates all bad clients and bad agents. Bad clients may observe the good updates before submitting their own (rushing adversary) and choose their updates arbitrarily; bad agents may deviate arbitrarily from the protocol: post any agent aggregate and any 2D point, submit invalid answers to the contract, or withhold data. The adversary cannot alter the code run by good participants or the smart contract, forge signatures, or break the hash function, and the local functions $f _ { i }$ of good clients are private. The contract draws the agents of each round at random from a pool of registered participants with deposits, and so the adversary controls how many of its accounts are in the pool but not which are drawn.

Validity filter. Every good local update is the output of a linear maximization over $\mathcal { P } _ { : }$ , attained at a vertex; for the cardinality constraint with monotone objectives, it is the indicator of the r elements with the largest marginal gains (ties broken by index), a vector in $\textstyle \mathcal { V } _ { r }$ (Section 3). Agents therefore discard any submitted vector outside $\textstyle { \mathcal { V } } _ { r }$ , an $O ( d )$ check, so that, without loss of generality, every bad update is also a binary vector with exactly r ones: the adversary chooses only which r elements to select. For a general matroid, the filter accepts indicators of bases, checkable with an independence oracle.

## 4.2. Safety as the Aggregation Target

Safety target. Deciding in a low-dimensional projection and lifting back loses information. Therefore, we need a guarantee that the lift preserves, at least approximately, whatever the adversary does. Distances are a poor basis, since attacks can shape them. We target an aggregate that is safe: a convex combination of good updates only, i.e., a point of their convex hull. A robust aggregate may still give bad updates a positive share; a safe aggregate gives them none.

Definition 1 (Safe vector, safety deficit). A vector w is safe if $\mathbf { w } \in \mathrm { c o n v } ( C ^ { + } )$ . For $\mathbf { w } \in \mathcal { S } .$ , its safety deficit is

$$
\begin{array} { r } { \sigma _ { C ^ { + } } ( \mathbf { w } ) \triangleq \operatorname* { m i n } { \big \{ } s \in [ 0 , 1 ] : \mathbf { w } \in ( 1 - s ) \operatorname { c o n v } ( C ^ { + } ) + s \mathcal { P } \big \} } .  \end{array}
$$

The deficit is the smallest fraction of w that cannot be attributed to good clients; $\sigma _ { C ^ { + } } ( \mathbf { w } ) = 0$ if w is safe, and $\sigma _ { C ^ { + } } \leq 1$ always since conv $( C ^ { + } ) \subseteq { \mathcal { P } }$ . It does not depend on how w was computed. For example, the arithmetic mean of validity-filtered updates has $\sigma _ { C ^ { + } } \leq \beta _ { c }$ , with equality when bad updates lie outside $\operatorname { c o n v } ( C ^ { + } ) ;$ ; a selection rule such as Krum has $\sigma _ { C ^ { + } } \in \{ 0 , 1 \}$ in each round depending on whether it selects a good update.

Section 5 (Theorem 1) will later show why this is the right target for FSM: for monotone objectives, an aggregate with deficit σ can only dilute the greedy progress by a factor $( 1 - \sigma )$ , regardless of what the adversary does; a safe aggregate in every round recovers the $( 1 - 1 / \mathrm { e } )$ guarantee up to client heterogeneity. Exact safety in $\mathbb { R } ^ { d }$ , however, cannot be certified eficiently. The classical certificate is Tukey depth.

Definition 2 (Tukey depth, deep point, Tukey median). The Tukey depth of a point u $\in \mathbb { R } ^ { k }$ with respect to a finite multiset $\mathbf { U } \subset \mathbb { R } ^ { k }$ is the minimum number of points of U in a closed half-space containing u:

$$
\mathrm { d e p t h } ( \mathbf { u } ; \mathbf { U } ) \triangleq \operatorname* { m i n } _ { \mathbf { y } \in \mathbb { R } ^ { k } \setminus \{ \mathbf { 0 } \} } \big | \{ \mathbf { v } \in \mathbf { U } : \langle \mathbf { y } , \mathbf { v } \rangle \geq \langle \mathbf { y } , \mathbf { u } \rangle \} \big | .
$$

A point u is a deep point if its depth exceeds the number of bad points in U. A point of maximum depth is a Tukey median.

A deep point is always safe: otherwise, a closed half-space containing it would miss every good point and hold fewer points than its depth. In $\mathbb { R } ^ { d }$ , a deep point is guaranteed to exist only when $\beta _ { c } < 1 / ( d + 1 )$ , which is vacuous when d is the size of the ground set. We therefore target safety through a tractable proxy that is exact in a 2D Gaussian random projection, and controls what is lost when lifting back to $\mathbb { R } ^ { d }$ . We do not claim exact safety in $\mathbb { R } ^ { d }$

## 4.3. The SafeGM Aggregation Rule

Notation. A superscript + marks a safe (or safety-targeting) vector, ∗ a vector of the global (agent-level) aggregation, and a hat a vector lifted from the projected space, hence an approximation; symbols without a hat are exact. For instance, ${ \mathbf { u } } ^ { + }$ targets safety exactly in the projection, while its lift $\hat { \mathbf { w } } ^ { + }$ approximates a safe vector.

Let $\Pi : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ be the round’s 2D Gaussian random projection $( \Pi _ { i j } \sim { \cal N } ( 0 , 1 / 2 ) )$ . We choose the plane for two reasons. First, a Tukey median of n points is cheap to compute in 2D, taking O(n log n) expected time [27]. Second, by the centerpoint theorem [26], the plane tolerates a Byzantine client rate up to $1 / 3$ , higher than $1 / ( k + 1 )$ in $\mathbb { R } ^ { k }$ for $k > 2 .$ A Byzantine threshold $1 / 3$ is the standard assumption of Byzantine fault-tolerant systems [30, 31].

Our aggregation rule SafeGM is:

1. Filter, deduplicate, and project: discard ${ \mathbf w } _ { j } \notin \mathcal N _ { r }$ , keep one copy of each distinct update, and apply the 2D projection $\mathbf { u } _ { j } = \Pi \mathbf { w } _ { j } , \mathbf { U } = \{ \mathbf { u } _ { j } \}$

2. Depth test: compute the geometric median g of U. If depth(g; $\mathbf { U } ) \geq \lceil n / 3 \rceil$ , set ${ \bf { u } } ^ { + } = { \bf { g } } ;$ otherwise let ${ \mathbf { u } } ^ { + }$ be a Tukey median of U.

3. Capped lifting: solve the linear program

$$
\operatorname* { m i n } _ { \lambda , \tau } \ \tau \quad \mathrm { s . t . } \ \sum _ { j } \lambda _ { j } \mathbf { u } _ { j } = \mathbf { u } ^ { + } , \quad \sum _ { j } \lambda _ { j } = 1 , \quad 0 \leq \lambda _ { j } \leq \tau ,\tag{3}
$$

and output $\begin{array} { r } { \hat { \mathbf { w } } ^ { + } = \sum _ { j } \lambda _ { j } \mathbf { w } _ { j } } \end{array}$

Deduplication in Step 1 gives every distinct update one vote. Bad clients that coordinate on the same update, as in the Include attack [3], would otherwise stack their copies on a single projected point, which then gains depth and coeficient in proportion to the number of copies. Heterogeneous good clients rarely submit identical updates, and removing a copy of a good update cannot make the aggregate unsafe. Hereafter, $C ^ { + }$ and $C ^ { - }$ denote the distinct good and bad updates, $n = | C ^ { + } | + | C ^ { - } |$ |, and an update identical to a good one counts as good.

In Step 2, we use the geometric median first because it is the most accurate of the common robust aggregators in practice [4], and in the plane it is almost always deep: in our experiments its depth reached $\lceil n / 3 \rceil$ in about 96% of aggregations. The Tukey median is only a fallback that ensures depth at least $\lceil n / 3 \rceil$ in every case (see Lemma 1(i) below), which is all the analysis needs. The rule never uses the number of bad updates. Whenever that number is below the depth of $\mathbf { u } ^ { + } , \mathbf { u } ^ { + }$ is a deep point and hence safe, $\mathrm { i . e . , } \mathbf { u ^ { + } } \in \Pi \mathrm { c o n v } ( C ^ { + } )$ . Lemma 1 below bounds the share of bad updates in every case.

Lemma 1 (Depth and capped lifting). Let $D ^ { + } \triangleq \mathrm { d e p t h } ( \mathbf { u } ^ { + } ; \mathbf { U } )$ . Then: (i) $D ^ { + } \geq \lceil n / 3 \rceil ;$ (ii) the optimal value of the capped linear program (3) is at most $1 / D ^ { + }$ , hence every coeficient $\begin{array} { r } { \lambda _ { j } \le 1 / D ^ { + } ; } \end{array}$ ; and (iii) the safety deficit ofthe output $\hat { \mathbf { w } } ^ { + }$ satisfies

$$
\sigma _ { C ^ { + } } ( \hat { \mathbf { w } } ^ { + } ) \leq \lambda ( C ^ { - } ) \leq \frac { | C ^ { - } | } { D ^ { + } } ,
$$

where $\begin{array} { r } { \lambda ( C ^ { - } ) \triangleq \sum _ { j \in C ^ { - } } \lambda _ { j } } \end{array}$ is the share ofbad updates.

Proof. (i) If ${ \bf u } ^ { + } = { \bf g } ,$ this is the test in Step 2. Otherwise ${ \mathbf { u } } ^ { + }$ is a Tukey median, whose depth is at least $\lceil n / 3 \rceil$ by the centerpoint theorem in the plane [26].

(ii) Let $Q ~ = ~ \{ \sum _ { j } \lambda _ { j } \mathbf { u } _ { j } ~ : ~ \lambda ~ \in ~ \Delta _ { n } , ~ \lambda _ { j } ~ \leq ~ 1 / D ^ { + } \}$ , a compact convex set whose support function is $h _ { Q } ( \mathbf { y } ) \ =$ $\begin{array} { r } { \frac { 1 } { D ^ { + } } \sum _ { D ^ { + } \mathrm { l a r g e s t } } \langle \mathbf { y } , \mathbf { u } _ { j } \rangle } \end{array}$ (fill the cap greedily). By the definition of depth, for every $\textbf { y } \neq \textbf { 0 }$ at least $D ^ { + }$ points satisfy $\langle \mathbf { y } , \mathbf { u } _ { j } \rangle \geq \langle \mathbf { y } , \mathbf { u } ^ { + } \rangle$ , and so the average of the $D ^ { + }$ largest values is at least $\left. \mathbf { y } , \mathbf { u } ^ { + } \right.$ . Hence $\langle \mathbf { y } , \mathbf { u } ^ { + } \rangle \leq h _ { Q } ( \mathbf { y } )$ for all $\mathbf { y }$ (trivially for $\mathbf { y } = \mathbf { 0 } )$ , and so ${ \mathbf { u } } ^ { + } \in { \cal { Q } } .$ . The corresponding λ, with $\tau = 1 / D ^ { + }$ , is feasible for (3). The optimal value $\tau ^ { * }$ is therefore at most $1 / D ^ { + }$ , and the returned coeficients satisfy $\lambda _ { i } \le \tau ^ { * } \le 1 / D ^ { + }$

(iii) Summing (ii) over the bad updates gives $\lambda ( C ^ { - } ) \leq | C ^ { - } | / D ^ { + } . \operatorname { I f } \lambda ( C ^ { - } ) = 0 .$ , then $\hat { \mathbf { w } } ^ { + } \in \mathrm { c o n v } ( C ^ { + } )$ and $\sigma _ { C ^ { + } } ( \hat { \mathbf { w } } ^ { + } ) =$ 0. If $0 < \lambda ( C ^ { - } ) < 1$ , write $\hat { \mathbf { w } } ^ { + } = ( 1 - \lambda ( C ^ { - } ) ) \mathbf { g } ^ { \prime } + \lambda ( C ^ { - } ) \mathbf { p }$ with $\begin{array} { r } { \textbf { g } ^ { \prime } = \sum _ { j \in C ^ { + } } \lambda _ { j } \mathbf w _ { j } / ( 1 - \lambda ( C ^ { - } ) ) \in } \end{array}$ conv $( C ^ { + } )$ and $\begin{array} { r } { \mathbf { p } = \sum _ { i \in C ^ { - } } \lambda _ { j } \mathbf { w } _ { j } / \lambda ( C ^ { - } ) \in \mathcal { P } } \end{array}$ , since bad updates pass the validity filter; hence $\sigma _ { C ^ { + } } ( \hat { \mathbf { w } } ^ { + } ) \leq \lambda ( C ^ { - } )$ . The case $\lambda ( C ^ { - } ) = 1$ is trivial. □

Remark. The bound on the share of bad updates holds in $\mathbb { R } ^ { d } \colon$ whatever the bad updates are and wherever their projections lie, together they carry at most a $| C ^ { - } | / D ^ { + }$ share of $\hat { \mathbf { w } } ^ { + }$ . By Lemma 1(i), this fraction is at most three times the fraction of bad updates. The proof above uses only the depth of ${ \mathbf { u } } ^ { + }$ , and so the same bound, with D in place of $D ^ { + }$ holds for any point of depth $D \geq 1$ and any coeficients $\lambda _ { j } \leq 1 / D$ that reproduce it. In Section 5, we turn this bound into a guarantee on solution quality (Theorem 1) and apply it at the agent level (Theorem 2).

Unlike Krum- and trimming-based rules, which require knowing a priori how many updates to discard, SafeGM requires no knowledge of the Byzantine client rate: its tolerance adapts to the data through the depth $D ^ { + }$ , which is often well above the test threshold $\lceil n / 3 \rceil$ . In the global update, $\hat { \mathbf { w } } ^ { + }$ replaces the mean,

$$
\mathbf { w } ^ { ( t ) } = \mathsf { S a f e G M } ( \{ \mathbf { w } _ { i } ^ { ( t ) } \mid i \in C ^ { ( t ) } \} ) , \qquad \mathbf { x } ^ { ( t ) } = \mathbf { x } ^ { ( t - 1 ) } + \eta \mathbf { w } ^ { ( t ) } .\tag{4}
$$

## 4.4. Decentralization

We now decentralize the safe aggregation. DeFSM splits the work between the smart contract and the of-chain agents so that each party does only what it can do cheaply. Agents aggregate random samples of client updates, and the contract makes the robust decision on 2D projections of the agents’ results. This split is possible because SafeGM decides in the plane: the contract only needs 2M numbers (the 2D coordinates of each agent’s projection), a geometric median or Tukey median of M points in the plane, and a small linear program. In contrast, an aggregation rule that decides in $\mathbb { R } ^ { d }$ would need the contract to store Md numbers and run the rule on them. Its agents would have to compute it of-chain and vote on the result instead.

Agents. Being an agent is a role, not a separate kind of machine. Anyone can register for it by placing a deposit with the contract, including clients. In each round, the contract draws M agents at random from the registered pool. This draw takes place after the client deadline, and as the set of agents difers from round to round, an adversary cannot choose which of its accounts are drawn and no agent knows its sample in advance.

## 4.4.1. Protocol

Each round has three main tasks. In agent aggregation, each agent runs SafeGM on a random sample of $L \ll N$ client updates. In global aggregation, the contract computes the SafeGM point of the agents’ 2D projections and coeficients that reproduce it; no agreement among agents is needed. In reconstruction, each client computes the global aggregate as the weighted sum of the agent aggregates with positive coeficient. There are two projection matrices, Π for projecting client updates and Π<sup>′</sup> for projecting agent aggregates. Fig. 1 illustrates the steps. In detail, a round proceeds as follows.

1. Client commit. Each client uploads $\mathbf { w } _ { j } \in \mathcal { N } ,$ to IPFS and commits $h ( \mathbf { w } _ { j } )$ to the contract. The transaction closing the client deadline fixes the seed of the client-level projection Π. The contract then draws M random agents and assigns to each agent i a uniformly random sample $C _ { i } \subseteq C$ of L clients, drawn independently across agents.

2. Agent aggregation. Agent i downloads its assigned updates $\mathbf { W } _ { i } = \{ \mathbf { w } _ { i l } \} _ { l = 1 } ^ { L }$ from IPFS, verifies their hashes and the validity filter, and runs SafeGM with Π, obtaining the agent aggregate $\begin{array} { r } { \hat { \mathbf { v } } _ { i } ^ { + } = \sum _ { l } { a _ { i l } { \mathbf { w } _ { i l } } } \in \mathcal { P } } \end{array}$ . It uploads $\hat { \mathbf { v } } _ { i } ^ { + }$ to IPFS and commits $h ( \hat { \mathbf { v } } _ { i } ^ { + } )$ to the contract.

3. Projection. The transaction closing the agent deadline fixes the seed of the agent-level projection $\Pi ^ { \prime }$ . Each agent i posts $\mathbf { z } _ { i } = \Pi ^ { \prime } \hat { \mathbf { v } } _ { i } ^ { + }$ to the contract. The contract thus keeps two numbers (the 2D coordinates) per distinct commitment, and so agents with identical aggregates count once. Let ${ \bf Z } = \{ { \bf z } _ { j } \}$ be the kept points.

4. Global aggregation. The contract applies the 2D aggregate of SafeGM to $\mathbf { Z } \mathbf { : }$ the 2D aggregate $\mathbf { z } ^ { \ast }$ is the geometric median if its depth is at least $\lceil \rceil \mathbf { Z } \rceil / 3 \rceil$ , or a Tukey median otherwise. It then computes convex coeficients b of $\mathbf { z } ^ { \ast }$ that solve (3) with the cap $\tau = 1 / D$ , where $D \triangleq \operatorname { d e p t h } ( \mathbf { z } ^ { * } ; \mathbf { Z } )$ , using a basic solution of the linear program found with a fixed pivoting rule. This solution has at most three coeficients strictly between 0 and $1 / D ,$ one per equality constraint. Therefore, at most $D + 2$ coeficients are positive, and so each client downloads at most $D + 2$ of the M agent aggregates, saving communication. The contract posts b.

![](images/8fb6fa654e135bfc644f66e37a5a5014df1ed4da35bfc7fea1c070f8508f6631.jpg)  
Figure 1: One round of DeFSM. Solid arrows are of-chain data transfers (orange: to and from IPFS); dashed arrows are on-chain posts. Numbers match the steps of the protocol.

5. Reconstruction. Each client downloads the agent aggregates $\{ \hat { \mathbf { v } } _ { j } ^ { + } \mid b _ { j } > 0 \}$ , verifies their hashes, and computes the global aggregate

$$
\hat { \mathbf { w } } ^ { * } = \sum _ { j } b _ { j } \hat { \mathbf { v } } _ { j } ^ { + } = \sum _ { j } b _ { j } \sum _ { l } a _ { j l } \mathbf { w } _ { j l } ,\tag{5}
$$

and the global step $\mathbf { x } \gets \mathbf { x } { + } \eta \hat { \mathbf { w } } ^ { * }$ . The coeficients are on-chain and the sum is computed in fixed-point arithmetic, and so all good clients obtain the same $\hat { \mathbf { w } } ^ { * }$ and the same x. A client also checks that $\Pi ^ { \prime } \hat { \mathbf { v } } _ { j } ^ { + } = \mathbf { z } _ { j }$ and $\hat { \mathbf { v } } _ { i } ^ { + } \in \mathcal { S }$ for each downloaded aggregate. If a check fails, it posts the aggregate as a fraud proof; the contract removes the agent’s point, slashes its deposit, and the round repeats Step 4.

No step of the basic protocol requires the agents to communicate, reach consensus, or vote: the contract makes the robust decision at the agent level, on public data.

Bandwidth-limited clients. In Step 5, a client downloads about D agent aggregates. If clients cannot aford this, one agent can compute $\hat { \mathbf { W } } ^ { * }$ , upload it to IPFS, and commit its hash; the agents then vote on the hash, and each client downloads a single vector. This vote certifies only the arithmetic of a weighted sum whose coeficients are already fixed on-chain; it does not afect the robust decision. We use client-side reconstruction, which needs no vote.

## 4.4.2. Complexity

Consider a round, with sparse binary updates of r ones. For agents, each agent downloads L client updates and spends $O ( L d )$ time for the projection, O(L log L) time for the depth test, and O(L) time for the capped lift [32]; hence $O ( L d )$ time in total, since $d \geq \log L .$ Over all M agents, this is O(MLd) time and ML downloads. All these costs are independent of $N .$ The contract spends O(M log M) time, or $O ( M ^ { 3 }$ log M) in the rare fallback to a Tukey median, plus $O ( M )$ time for its linear program; storage is $O ( N + M )$ , independent of d. For the clients, each downloads at most $D + 2$ agent aggregates, each with at most $L r$ nonzero entries, that is, $O ( D L r )$ entries, and computes their weighted sum in the same time, independent of d and N. For example, in our experiments $( M = L = 2 0 , r = 1 0 , d = 2 0 0 )$ , a client downloads on average 9.3 agent aggregates with 487 nonzero entries per round in total, about 4 KB, less than a typical webpage.

## 4.4.3. Unpredictable projections

Each round uses two fresh projections: Π for SafeGM (Step 1) and Π<sup>′</sup> for the agent level (Step 3). Every party regenerates both from public 32-byte seeds fixed by the smart contract, from the chain’s built-in randomness or a verifiable random function, so they cost no extra communication. Each seed is fixed only after the inputs it projects have been committed. Hence bad clients choose their updates without knowing Π, so they cannot place their projected points where they would gain the largest share, such as at the Tukey median; likewise, bad agents commit their aggregates without knowing Π<sup>′</sup>. The consistency check in Step 5 keeps them from moving their 2D points after Π<sup>′</sup> is known. A projection reused across rounds would be known in advance.

A bad client also has little control over where its update lands in the plane. For the matrix $\mathbf { W } = [ \mathbf { w } _ { 1 } , \ldots , \mathbf { w } _ { n } ]$ of distinct updates, the rows of U = ΠW are i.i.d. N(0 G/2) with Gram matrix $\mathbf { G } = \mathbf { W } ^ { \top } \mathbf { W }$ , and every entry of G lies in [0 r] after the validity filter. Hence a bad client afects the projected points only through how many elements its update shares with the others, and cannot aim its update at the deep region. The same holds for Π<sup>′</sup> and the agents aggregates.

## 5. Analysis

We first show that the safety deficit controls the solution: the quality of the federated continuous greedy solution degrades gracefully with the deficit of the aggregates, independently of the attack (Section 5.1), so a bound on the deficit is all we need. We then bound the deficit of SafeGM’s aggregate in DeFSM by counts and depths, and combine the two results into a convergence guarantee (Section 5.2).

## 5.1. Safety Suficesfor Monotone FSM

Throughout, each good client’s $f _ { i }$ is monotone submodular with $\begin{array} { r } { f _ { i } ( \emptyset ) ~ = ~ 0 ; ~ F _ { C ^ { + } } ~ \stackrel { \triangle } { = } ~ \frac { 1 } { N ^ { + } } \sum _ { i \in C ^ { + } } F _ { i } } \end{array}$ is the multilinear extension of the good clients’ objective, $\begin{array} { r } { \mathrm { O P T } \ = \ \operatorname* { m a x } _ { S \in \mathcal { T } } \frac { 1 } { N ^ { + } } \sum _ { i \in C ^ { + } } f _ { i } ( S ) } \end{array}$ , and $m \ \triangleq \ \operatorname* { m a x } _ { i \in C ^ { + } , e \in E } f _ { i } ( \{ e \} )$ . The algorithm runs $T \ = \ 1 / \eta$ rounds from $\mathbf { X } ^ { ( 0 ) } \ = \ \mathbf { 0 }$ , with $\mathbf { x } ^ { ( t ) } = \mathbf { x } ^ { ( t - 1 ) } + \eta \mathbf { w } ^ { ( t ) }$ . In round $t ,$ each good client j returns ${ \bf w } _ { j } \in  \Psi _ { { j } }$ arg $\operatorname* { m a x } _ { \mathbf { w } \in \mathcal { P } } \langle \mathbf { w } , \nabla F _ { j } ( \mathbf { x } ^ { ( t - 1 ) } ) \rangle$ ⟩.

Like most analyses of federated optimization under non-iid data, we bound how far a client’s gradient can be from the average [33, 34].

Assumption 1 (Bounded heterogeneity). There is $\zeta \geq 0$ such that $\| \nabla F _ { i } ( { \mathbf { x } } ) - \nabla F _ { C ^ { + } } ( { \mathbf { x } } ) \| _ { \infty } \leq \zeta$ for all $\mathbf { x } \in \mathcal { P }$ and $i \in C ^ { + }$ (If clients use sampled gradient estimates, ζ also absorbs the estimation error.)

The next theorem shows that the adversary afects the solution only through the safety deficit of the aggregates.

Theorem 1 (Safety sufices). Suppose every aggregate satisfies $\mathbf { w } ^ { ( t ) } \in \mathcal { S }$ and has safety deficit $\sigma _ { { C ^ { + } } _ { t } } ( \mathbf { w } ^ { ( t ) } ) \leq \sigma _ { t }$ for some $\sigma _ { t } \in [ 0 , 1 ]$ , where $C ^ { + }$ is the set of good updates of round t, and let $\begin{array} { r } { \bar { \boldsymbol { \sigma } } = \frac { 1 } { T } \sum _ { t } \boldsymbol { \sigma } _ { t } } \end{array}$ be the average deficit bound. Then,for any behavior ofthe bad clients, the multilinear extension $F _ { C ^ { + } }$ ofthe good clients’ objective satisfies

$$
F _ { C ^ { + } } ( { \bf x } ^ { ( T ) } ) \geq \big ( 1 - \mathrm { e } ^ { - ( 1 - \bar { \sigma } ) } \big ) \mathrm { O P T } - 2 r \zeta - { \frac { \eta m r ^ { 2 } } { 2 } } .
$$

Proof. Feasibility. $\begin{array} { r } { \mathbf { x } ^ { ( t ) } = \eta \sum _ { s \leq t } \mathbf { w } ^ { ( s ) } } \end{array}$ is $t \eta \leq 1$ times an average of points of ${ \mathcal { P } } ,$ and $\mathbf { 0 } \in { \mathcal { S } } .$ , so $\mathbf { X } ^ { \left( t \right) } \in \mathcal { S }$

Good directions are good ascent directions. Let $\mathbf { x } = \mathbf { x } ^ { ( t - 1 ) }$ and O an optimal set. For $j \in C ^ { + } { } _ { t } ,$ , since $\mathbf { 1 } _ { O } \in \mathcal { S }$ and $\| \mathbf { w } \| _ { 1 } \leq r \mathrm { o n } \mathcal { P } _ { \cdot }$

$$
\begin{array} { r l } & { \langle \mathbf { w } _ { j } , \nabla F _ { C ^ { + } } ( \mathbf { x } ) \rangle \geq \langle \mathbf { w } _ { j } , \nabla F _ { j } ( \mathbf { x } ) \rangle - r \zeta \geq \langle \mathbf { 1 } _ { O } , \nabla F _ { j } ( \mathbf { x } ) \rangle - r \zeta } \\ & { \qquad \geq \langle \mathbf { 1 } _ { O } , \nabla F _ { C ^ { + } } ( \mathbf { x } ) \rangle - 2 r \zeta \geq \mathrm { O P T } - F _ { C ^ { + } } ( \mathbf { x } ) - 2 r \zeta , } \end{array}
$$

where the last step is the standard property of monotone submodular multilinear extensions [29]: $F _ { C ^ { + } } ( \mathbf { x } \vee \mathbf { 1 } _ { O } ) -$ $F _ { C ^ { + } } ( \mathbf { x } ) \ \leq \ \langle \mathbf { 1 } _ { O } , \nabla F _ { C ^ { + } } ( \mathbf { x } ) \rangle$ by concavity along nonnegative directions and $\nabla F _ { C ^ { + } } \geq \mathbf { 0 } .$ , and $F _ { C ^ { + } } ( \mathbf { x } \vee \mathbf { 1 } _ { O } ) \geq \mathrm { O P T }$ by monotonicity. By linearity the same bound holds for every $\mathbf { g } \in \operatorname { c o n v } ( \boldsymbol { C } ^ { + } { } _ { t } )$

Unsafe mass can only dilute. Write $\mathbf w ^ { ( t ) } = ( 1 - \sigma _ { t } ) \mathbf g + \sigma _ { t } \mathbf p$ with g ∈ conv $( C ^ { + } { } _ { t } ) , \ \mathbf { p } \in { \mathcal { P } }$ (possible since $( 1 ~ -$ $s ) \operatorname { c o n v } ( C ^ { + } { } _ { t } ) + s \mathcal { P }$ grows with s when conv $( C ^ { + } { } _ { t } ) \subseteq { \mathcal { P } } )$ . Monotonicity gives $\nabla F _ { C ^ { + } } \geq \mathbf { 0 } $ , and $\mathbf { p } \geq \mathbf { 0 } .$ , so $\langle \mathbf { p } , \nabla F _ { C ^ { + } } ( \mathbf { x } ) \rangle \geq 0$ and

$$
\langle \mathbf { w } ^ { ( t ) } , \nabla F _ { C ^ { + } } ( \mathbf { x } ) \rangle \geq ( 1 - \sigma _ { t } ) ( \mathrm { O P T } - F _ { C ^ { + } } ( \mathbf { x } ) ) - 2 r \zeta .
$$

Smoothness. For the multilinear extension of a monotone submodular function, $\partial ^ { 2 } F _ { C ^ { + } } / \partial x _ { e } ^ { 2 } = 0$ and $- m \leq$ $\partial ^ { 2 } F _ { C ^ { + } } / \partial x _ { e } \partial x _ { e ^ { \prime } } \leq 0 .$ so, using $\| \mathbf { w } \| _ { 1 } \leq r$

$$
\begin{array} { r } { F _ { C ^ { + } } ( \mathbf { x } + \eta \mathbf { w } ) \geq F _ { C ^ { + } } ( \mathbf { x } ) + \eta \langle \mathbf { w } , \nabla F _ { C ^ { + } } ( \mathbf { x } ) \rangle - \frac { \eta ^ { 2 } m r ^ { 2 } } { 2 } . } \end{array}
$$

Recursion. With $\gamma _ { t } = \mathrm { O P T } - F _ { C ^ { + } } ( \mathbf { x } ^ { ( t ) } )$ , the last two displays give

$$
\begin{array} { r } { \gamma _ { t } \leq \big ( 1 - \eta ( 1 - \sigma _ { t } ) \big ) \gamma _ { t - 1 } + 2 r \zeta \eta + \frac { \eta ^ { 2 } m r ^ { 2 } } { 2 } . } \end{array}
$$

Unrolling from $\gamma _ { 0 } = \mathrm { O P T }$ and using $1 - z \leq \mathrm { e } ^ { - z }$ and $T \eta = 1$

$$
\begin{array} { r } { \gamma _ { T } \leq \mathrm { e } ^ { - \eta \sum _ { t } ( 1 - \sigma _ { t } ) } \mathrm { O P T } + 2 r \zeta + \frac { \eta m r ^ { 2 } } { 2 } = \mathrm { e } ^ { - ( 1 - \bar { \sigma } ) } \mathrm { O P T } + 2 r \zeta + \frac { \eta m r ^ { 2 } } { 2 } . } \end{array}
$$

Theorem 1 formalizes our design philosophy. The bound depends on the adversary only through the deficit, not through any notion of distance, and the validity filter ensures that unsafe mass slows progress but never reverses it. For example, the filtered mean has $\bar { \sigma } \le \beta _ { c }$ , while a rule that selects a bad update has $\sigma _ { t } = 1$ and makes no guaranteed progress in that round, which is how distance-based selection collapses under targeted attacks. SafeGM aims at $\sigma _ { t } \to 0 .$ . Since $1 - \mathrm { e } ^ { - z }$ is concave, $1 - \mathrm { e } ^ { - ( 1 - \bar { \sigma } ) } \geq ( 1 - \bar { \sigma } ) ( 1 - 1 / \mathrm { e } )$ : an average deficit $\bar { \sigma }$ costs at most a factor $( 1 - { \bar { \sigma } } )$ , and a safe aggregate in every round $( \bar { \sigma } = 0 )$ recovers the centralized $( 1 - 1 / \mathrm { e } )$ guarantee up to heterogeneity and discretization, for any adversary. The guarantee carries over to the final set: pipage rounding [29] turns $\mathbf { x } ^ { ( T ) } \in \mathcal { S }$ into a set S with $| S | \le r$ and $\mathbb { E } [ f _ { C ^ { + } } ( S ) ] \geq F _ { C ^ { + } } ( \mathbf { x } ^ { ( T ) } )$

Remark (why FSM, not federated learning). Theorem 1 rests on structure that FSM has and federated learning (FL) lacks. First, monotonicity makes every coordinate of the gradient nonnegative, and every update lies in the nonnegative polytope $\mathcal { P } ,$ so the unsafe part of an aggregate can only dilute progress, never reverse it. In FL, gradients take both signs, so even a small unsafe share can push the model in a harmful direction, and a bounded deficit no longer bounds the damage. Second, the validity filter checks that each update is a vertex of $\mathcal { P } ;$ no comparable check exists for a real-valued gradient. Third, FSM updates are sparse binary vectors with r ones, so a client can aford to download the $D + 2$ agent aggregates it needs; dense FL models with millions of parameters would multiply this cost. A count-based bound is thus a natural fit for FSM, and its extension to FL would require new ideas.

In practice, the final set is often taken deterministically as the r largest coordinates of $\mathbf { x } ^ { ( T ) }$ (top-r rounding), which is not lossless. Coordinated attacks such as Include [3] exploit exactly this step: bad clients that all select the same r elements concentrate their mass on few coordinates, while non-iid good clients spread theirs. Theorem 1 does not cover top-r rounding, but the deficit still limits such attacks. Call an element unsupported if no good client selects it in any round; the next proposition bounds, by the deficit, how much mass such elements can receive.

Proposition 1 (Top-r rounding). Under the conditions of Theorem $^ { l , }$ every unsupported element e satisfies $x ^ { ( T ) } [ e ] \leq$ ${ \bar { \sigma } } ,$ and the supported elements together carry mass at least $( 1 - { \bar { \sigma } } ) r$

Proof. Write $\mathbf { w } ^ { ( t ) } = ( 1 - \sigma _ { t } ) \mathbf { g } + \sigma _ { t } \mathbf { p }$ with $\mathbf { g } \in \mathrm { c o n v } ( \boldsymbol { C ^ { + } } _ { t } ) , \mathbf { p } \in \mathcal { P }$ . For unsupported $e ,$ every good update has $w _ { j } [ e ] = 0$ , so $g [ e ] = 0$ and $w ^ { ( t ) } [ e ] = \sigma _ { t } p [ e ] \leq \sigma _ { t } ;$ summing, $\begin{array} { r } { x ^ { ( T ) } [ e ] \leq \eta \sum _ { t } \sigma _ { t } = \bar { \sigma } } \end{array}$ . Every good update is in $\boldsymbol { \mathcal { V } } _ { r }$ and has its r ones on supported elements, so g puts mass r on them, and the supported mass of $\mathbf { x } ^ { ( T ) }$ is at least $\eta \sum _ { t } ( 1 - \sigma _ { t } ) r = ( 1 - \bar { \sigma } ) r$ □

Hence, if at least r supported elements have $x ^ { ( T ) } [ e ] > \bar { \sigma }$ , the top-r set contains no unsupported element; with safe aggregates, unsupported elements receive no mass and are never selected. For the filtered mean, $\bar { \sigma }$ can reach $\beta _ { c } \colon$ identical Include updates give their target coordinates mass $\beta _ { c }$ , which outranks supported elements chosen by fewer than a $\beta _ { c } / ( 1 - \beta _ { c } )$ fraction of good clients in every round. Lowering $\bar { \sigma }$ below $\beta _ { c }$ is therefore what protects the final ranking.

## 5.2. Guarantee ofthe Decentralized Aggregate

By Theorem 1, it remains to bound the deficit of the global aggregate $\hat { \mathbf { w } } ^ { * }$ that the clients compute in Step 5 of the protocol. For agent $j ,$ let $C _ { j } \subseteq C$ be its sample, with good and bad parts $C ^ { + } { } _ { j }$ and $C ^ { - } { } _ { j } , N _ { i } ^ { - }$ the number of distinct bad updates in it, and $D _ { \boldsymbol { i } } ^ { + }$ the depth of the point chosen by SafeGM on its sample. Call a good agent in-tolerance if $N _ { j } ^ { - } < D _ { j } ^ { + }$ . Let $\tilde { \mathcal { A } } ^ { + } \subseteq \mathcal { A } ^ { + }$ be the in-tolerance good agents and $\tilde { \mathcal { A } } ^ { - }$ the remaining agents: the bad agents $\mathcal { A } ^ { - }$ and the out-of-tolerance good agents, which the analysis treats as bad; $\tilde { M } ^ { + }$ and $\tilde { M } ^ { - }$ denote the numbers of agents in these two sets. The next theorem bounds the deficit by counts and depths only. It needs no assumption on the network, because the agents do not need to agree with each other.

Theorem 2 (Decentralized safety bound). Let ${ \mathbf { z } } ^ { * } , D ,$ and b be the point, depth, and coeficients computed by the smart contract. Then $\hat { \mathbf { w } } ^ { * } \in \mathcal { P }$ and

$$
\sigma _ { C ^ { + } } ( \hat { \mathbf { w } } ^ { * } ) \leq \operatorname* { m a x } _ { j \in \mathcal { \tilde { A } } ^ { + } } \frac { N _ { j } ^ { - } } { D _ { j } ^ { + } } + \frac { \tilde { M } ^ { - } } { D } .
$$

Proof. Each agent aggregate that passes the checks of Step 5 lies in ${ \mathcal { P } } ,$ and b lies in the simplex, so $\hat { \mathbf { w } } ^ { * } \in \mathcal { S }$ . Each kept point $\mathbf { z } _ { j }$ corresponds to one distinct agent aggregate $\hat { \mathbf { v } } _ { j }$ . Write $\hat { \bf v } _ { j } = ( 1 - \sigma _ { j } ) { \bf g } _ { j } + \sigma _ { j } { \bf p } _ { j }$ with $\mathbf { g } _ { j } \in$ conv $( C ^ { + } )$ and $\mathbf { p } _ { j } \in \mathcal { P }$ . If an in-tolerance good agent produced $\hat { \mathbf { v } } _ { j } ,$ then $\sigma _ { j } \le \operatorname* { m a x } _ { i \in \tilde { \mathcal { A } } ^ { + } } N _ { i } ^ { - } / D _ { i } ^ { + }$ by Lemma 1; otherwise $\sigma _ { j } \leq 1$ , and at most $\tilde { M } ^ { - }$ distinct aggregates are of this kind. The contract’s coeficients satisfy $b _ { j } \leq 1 / D ,$ , so these aggregates carry a total share at most $\tilde { M } ^ { - } / D$ . Mixing the decompositions with coeficients $b _ { j }$ gives $\begin{array} { r } { \sigma _ { C ^ { + } } ( \hat { \mathbf { w } } ^ { * } ) \ \leq \ \sum _ { j } b _ { j } \sigma _ { j } \ \leq \ } \end{array}$ $\mathrm { m a x } _ { i \in \tilde { \mathcal { A } } ^ { + } } N _ { i } ^ { - } / D _ { i } ^ { + } + \tilde { M } ^ { - } / D$ □

Theorem 2 applies Lemma 1 at both levels: the first term bounds the lift from client updates to agent aggregates, and the second bounds the lift from the contract’s 2D point to $\mathbb { R } ^ { d }$ . The bound counts distinct vectors at both levels, so copies of a bad update, or of a bad agent aggregate, add nothing. Bad agents enter only through their number, whatever they post. By Lemma 1(i) applied to $\mathbf { Z } , D \geq \lceil | \mathbf { Z } | / 3 \rceil$ , and so the agent-level term is at most three times $\tilde { M } ^ { - } / | \mathbf { Z } | ,$ and it is smaller when the depth of the contract’s point is larger. Neither the clients, the agents nor the contract needs to know the Byzantine client or agent rate.

Remark (role of unpredictable projections). Lemma 1 and Theorem 2 hold even if the adversary knows the projections in advance. Fresh, unpredictable projections (Section 4.4.3) only make their worst case harder to reach.

Remark (why a deep point at the agent level). The mean of the distinct agent aggregates has a smaller worstcase bound, $\tilde { M } ^ { - } / | \mathbf { Z } |$ instead of $\tilde { M } ^ { - } / D$ . We use the capped deep point for two reasons. First, bad aggregates far from the deep region receive much smaller coeficients than the cap allows: at $\beta _ { c } = \beta _ { a } = 0 . 3$ , replacing it with the mean raised the realized deficit from 0.30 to 0.42 and lowered the quality from $0 . 7 9$ to 0.68 on average over both datasets (Section 6). Second, the coeficients are sparse: at most $D + 2$ agent aggregates are needed instead of all $| \mathbf { Z } | ,$ , which halves the client download in our experiments.

How likely is the tolerance condition? Suppose the good updates within a sample are distinct, as is the case for heterogeneous clients. Then $| C ^ { - } { } _ { j } | < L / 3$ implies $N _ { i } ^ { - } < D _ { i } ^ { + }$ : the deduplicated sample has $| C ^ { + } { } _ { j } | > 2 L / 3 >$ $2 N _ { j } ^ { - }$ good updates, so $N _ { j } ^ { - } < ( | C ^ { + } { } _ { j } | + N _ { j } ^ { - } ) / 3 \leq D _ { j } ^ { + }$ by the centerpoint theorem. The contract samples agents independently, so the number $M ^ { + } - \tilde { M } ^ { + }$ of outof-tolerance good agents is dominated by $\mathrm { B i n } ( M ^ { + } , \rho _ { L } )$ , where

Table 1: Worst-case probability $\rho _ { L }$ that an agent’s sample exceeds the planar tolerance.
<table><tr><td> $N$ </td><td> $\beta _ { c }$ </td><td> $L$ </td><td> $\rho _ { L }$ </td></tr><tr><td>60</td><td>1/6</td><td>20</td><td>0.012</td></tr><tr><td>60</td><td>1/3</td><td>20</td><td>0.534</td></tr><tr><td>200</td><td>1/6</td><td>30</td><td>0.011</td></tr><tr><td>200</td><td>1/4</td><td>40</td><td>0.079</td></tr></table>

$$
\begin{array} { r } { \rho _ { L } = \operatorname* { P r } \left[ \operatorname { H y p } ( N , N ^ { - } , L ) \geq \lceil L / 3 \rceil \right] \leq \mathrm { e } ^ { - 2 L \left( \frac { 1 } { 3 } - \beta _ { c } \right) ^ { 2 } } \qquad ( \beta _ { c } < \frac { 1 } { 3 } ) } \end{array}
$$

is the worst-case probability that an L-sample exceeds the planar tolerance; the inequality is Hoefding’s bound, which also holds for sampling without replacement [35]. Hence $\rho _ { L }$ decays exponentially in the sample size $L ,$ and the number $\tilde { M } ^ { - }$ of agents that the analysis treats as bad is dominated by $M ^ { - } + \mathrm { B i n } ( M ^ { + } , \rho _ { L } )$ . The agent-level term of Theorem 2 stays below one whenever $\tilde { M } ^ { - } < D ,$ which holds when $\tilde { M } ^ { - } < \lceil | \mathbf { Z } | / 3 \rceil$ . Table 1 gives exact values: $\rho _ { L }$ is small for moderate $\beta _ { c }$ but large near $\beta _ { c } = 1 / 3$ , where larger samples are needed and safety otherwise rests on the data-adaptive tolerance $N _ { i } ^ { - } < D _ { i } ^ { + }$

Combining Theorems 1 and 2 gives the end-to-end guarantee of DeFSM.

Corollary 1 (Convergence of DeFSM). After $T = 1 / \eta$ rounds,

$$
F _ { C ^ { + } } ( { \bf x } ^ { ( T ) } ) \geq ( 1 - \bar { \sigma } ) ( 1 - 1 / { \mathrm { e } } ) \mathrm { O P T } - 2 r \zeta - \frac { \eta m r ^ { 2 } } { 2 } ,
$$

where $\begin{array} { r } { \bar { \boldsymbol { \sigma } } = \frac { 1 } { T } \sum _ { t } \boldsymbol { \sigma } _ { t } } \end{array}$ and $\sigma _ { t }$ is the minimum of1 and the bound ofTheorem 2 in round t. Per round, $O ( N + M )$ numbers are stored on-chain, independent $o f d ,$ and each client downloads at most $D _ { t } + 2$ agent aggregates.

Proof. By Theorem 2 every aggregate lies in $\mathcal { P }$ and its deficit is at most $\sigma _ { t } ;$ Theorem 1 and the inequality $1 - \mathsf { e } ^ { - ( 1 - \bar { \sigma } ) } \geq$ $( 1 - \bar { \sigma } ) ( 1 - 1 / \mathrm { e } )$ then apply. The costs follow from the protocol. □

Corollary 1 guarantees graceful degradation under any attack: the guarantee depends on the adversary only through the deficit, and the deficit is bounded through counts and depths, never through distances that the adversary could manipulate. The quantity that SafeGM drives toward zero, and that we report empirically, is the realized deficit $\sigma _ { C ^ { + } } ( \mathbf { w } ^ { ( t ) } )$ . It is computable in simulation by a linear program, because $\mathcal { P }$ is a polytope and $C ^ { + }$ is known to the experimenter.

## 6. Numerical Evaluation

We evaluate DeFSM numerically to validate its analysis and compare it with baselines. All simulation code, data, and results are available at https: $: / / \mathfrak { g } \mathrm { i }$ thub.com/UMB-NCLab/defsm. Here DeFSM denotes the architecture with its rule SafeGM. We test four hypotheses.

H1 (Accuracy). Without any trusted party, DeFSM retains most of the quality of the geometric median at a trusted server, and stays close to the geometric median decentralized in an idealized setting.

H2 (Eficiency). DeFSM needs much less communication than the straightforward decentralization where the agents run the robust aggregator in $\mathbb { R } ^ { d }$ and either agree on the result or let the clients reconstruct it.

H3 (Robustness). DeFSM does not collapse, even under attacks designed against it, extreme data heterogeneity, and unknown Byzantine rates. In contrast, rate-dependent rules can collapse.

H4 (Design parameters). The deficit does not depend on the number of clients and decreases with the sample size, as predicted, and the client download grows linearly with the number of agents.

Baselines. Agent-X denotes aggregation rule X (geometric median, trimmed mean, or mean) run by the agents of DeFSM in place of SafeGM: it applies X at both levels, with the same agents, samples, and bad agents, and deduplicates as SafeGM does. Agent-GM, with the geometric median, the most accurate robust rule in practice, is our main comparison; since the contract cannot compute it, we idealize it: the agents compute it and their vote always returns the good result. Agent-Trim, with the coordinate-wise trimmed mean, knows the true numbers of bad clients and agents unless stated otherwise. Agent-Mean is the non-robust reference. FSM-GM, the geometric median applied by a trusted central server to all client updates, is the centralized ideal reference. We also tested Krum, but its accuracy collapsed in every setting we tried, and so we omit it from the results.

Metrics. The quality $\begin{array} { r } { Q = \frac { 1 } { N ^ { + } } \sum _ { i \in C ^ { + } } f _ { i } ( S ^ { * } ) } \end{array}$ of the final solution $S ^ { * }$ , obtained by pipage rounding (Section 5.1) and averaged over 10 roundings, is normalized so that 1 is the quality obtained by averaging the good clients’ updates only and 0 that of a random subset. The safety deficit σ (Definition 1) of each round’s aggregate is computed exactly by a linear program, since $C ^ { + }$ is known in simulation. Communication is the number of nonzero entries an agent and a client download per round.

## 6.1. Evaluation Setting

A simulator runs the protocol round by round. We first run the default matrix: all five attacks under all four combinations of Byzantine rates, with the other factors at their defaults. We then run ablations by varying one factor at a time: the heterogeneity, the numbers of clients and agents, the sample size, and the rule at the agent level. Table 2 lists the defaults and the varied values.

Task and data. Clients jointly select $r = 1 0$ images that best represent their image collections. We use two real-world datasets of 60K images each: CIFAR10 [36] (10 object classes) and PathMNIST [37] (9 classes of pathology tissue), both represented by standardized 512-dimensional ResNet18 features [38] pretrained on ImageNet. In each run, the ground set E consists of $d = 2 0 0$ images drawn uniformly at random. Each client holds 40 images drawn from a client-specific class mixture sampled from a Dirichlet distribution with concentration $\alpha \ = \ 0 . 5 ,$ so the data are non-iid. Client i’s valuation is the facilitylocation function $\begin{array} { r } { f _ { i } ( S ) = \sum _ { p \in \chi _ { i } } \operatorname* { m a x } _ { s \in S } \exp ( - \| p } \end{array}$

Table 2: Default values, and the values to which one factor at a time is varied.
<table><tr><td>Factor</td><td>Default</td><td>Varied</td></tr><tr><td>Datasets</td><td>CIFAR10, PathMNIST</td><td></td></tr><tr><td>Ground set, budget</td><td> $d = 2 0 0 , r = 1 0$ </td><td></td></tr><tr><td>Clients</td><td> $N = 6 0$ </td><td> $N \in \{ 2 0 0 , 5 0 0 \}$ </td></tr><tr><td>Heterogeneity</td><td> $\mathrm { D i r i c h l e t } \alpha = 0 . 5$ </td><td> $\alpha = 0 . 0 1$ </td></tr><tr><td>Agents per round</td><td> $M = 2 0$ </td><td> $M \in \{ 1 0 , 4 0 \}$ </td></tr><tr><td>Sample size</td><td> $L = 2 0$ </td><td> $L \in \{ 1 0 , 4 0 \}$ </td></tr><tr><td>Byzantine rates</td><td>all pairs of  $\beta _ { c } \in \{ 0 . 2 , 0 . 3 \} , \beta _ { a } \in \{ 0 . 1 , 0 . 3 \}$ </td><td></td></tr><tr><td>Client attacks</td><td>all five (see Attacks)</td><td></td></tr><tr><td>Rate told Agent-Trim true</td><td></td><td>true ±0.1</td></tr><tr><td>Agent-level rule</td><td>SafeGM point</td><td>mean</td></tr><tr><td>Step size, rounds</td><td> $\eta = 0 . 0 1 , \geq 1 0 0 \mathrm { r o u n d s }$ </td><td></td></tr><tr><td>Seeds</td><td>3</td><td> $1 0 ( \alpha , N ) , 5 ( M , L )$ </td></tr></table>

$s \| ^ { 2 } / ( 0 . 5 h ) ^ { 2 } )$ , where h is the median distance between ground-set images; it is monotone submodular. Clients estimate the gradient of their multilinear extension from 8 random samples and return the binary top-r vector. Each run per forms at least $1 / \eta = 1 0 0$ rounds, the horizon of Theorem 1, and stops once the global top-r set is unchanged for 15 rounds (at most 300); most stop at 100. Beyond $1 / \eta$ rounds, x may leave $\mathcal { P } ;$ it is then clipped to $[ 0 , 1 ] ^ { d }$ and rescaled to sum r before rounding.

Attacks. All bad client updates are valid binary vectors with r ones, and so the validity filter cannot reject them. We use five client attacks:

1. Include: every bad client submits the r elements ranked lowest by the current global solution, the strongest coordinated push toward a poor solution;

2. Camouflage: half of a good client’s choice plus the lowest-ranked elements, identical across bad clients, to look partly legitimate while still promoting poor elements;

3. Near-Include: Include with one element swapped per bad client, so that bad updates are distinct and deduplication cannot remove them; it targets SafeGM specifically and is the worst case for its deduplication;

4. Mimic: each bad client copies a diferent good client’s update and replaces half of its elements with lowestranked ones, to lie close to good updates in every projection, where depth cannot separate them; it also targets SafeGM and is the worst case for its depth test;

5. Random: each bad client submits r random elements, an uncoordinated adversary that only adds noise.

A bad agent, in every method, posts as its aggregate the Include vector with one element swapped, distinct for each bad agent so that deduplication cannot merge them. This is the strongest push toward poor elements; by Theorem 2, other behaviors cannot gain more than their count allows.

## 6.2. Numerical Results

## 6.2.1. Accuracy (H1)

Fig. 2 summarizes accuracy over the default matrix, and Table 3 gives the numbers. DeFSM and Agent-GM sit close together, with similar worst settings; Agent-Trim is higher on average but spreads widely on PathMNIST, and Agent-Mean is lowest. FSM-GM reaches 0.92 on both datasets and DeFSM 0.84, about 91% of it without any trusted party. Agent-GM, decentralized in an idealized setting, reaches 0.88–0.89, about 96%, with a worst run similar to that of DeFSM (0.64–0.66 against 0.61–0.63).

Hardest setting. Fig. 3 shows each attack at $\beta _ { c } = \beta _ { a } = 0 . 3$ . On PathMNIST, DeFSM ties with Agent-GM under Near-Include and is within 0.04–0.09 of it under the other attacks, ahead of Agent-Trim except under Mimic. On CIFAR10, it ranks third under every attack, behind Agent-Trim and Agent-GM, by 0.04–0.11 from Agent-GM. Agent-Mean is clearly behind in every case.

Convergence. Under every attack and on both datasets, DeFSM converges as fast as Agent-GM and Agent-Mean to a similar objective (Fig. 4). These results support H1: DeFSM gives up a few points of quality, about 0.05 against Agent-GM, in exchange for removing the trusted party.

![](images/8b172d6ac4b2ccfc7c02fab4691b906b71265ab5075e0ed43efa6dc56007ab65.jpg)  
Figure 2: Accuracy summary. Small points: one per setting of the default matrix (mean over 3 seeds); 10 settings for FSM-GM. Diamond: average; open triangle: worst setting.

Table 3: Reliability over all 20 attack settings (5 attacks × $\beta _ { c } \in \{ 0 . 2 , 0 . 3 \}$ × $\beta _ { a } \in \{ 0 . 1 , 0 . 3 \} )$ , 3 seeds, $\alpha = 0 . 5 $ . Agent-Trim is told the true Byzantine rates unless stated; FSM-GM has 10 settings (β only). Worst σ: largest deficit of any run. Worst run / worst setting / avg: smallest Q of any run, smallest average Q of any setting, average Q. Bold: best in column.
<table><tr><td></td><td colspan="3">CIFAR10</td><td colspan="3">PathMNIST</td></tr><tr><td>Method</td><td>worst worst σ</td><td>run</td><td>worst setting</td><td>avg Q σ</td><td>worst worst run</td><td>worst setting</td><td>avg Q</td></tr><tr><td>DeFSM</td><td>0.41</td><td>0.63</td><td>0.71</td><td>0.84</td><td>0.44 0.61</td><td>0.71</td><td>0.84</td></tr><tr><td>Agent-GM Agent-Trim</td><td>0.35</td><td>0.64</td><td>0.76</td><td>0.89 0.39</td><td>0.66</td><td>0.71</td><td>0.88</td></tr><tr><td>Agent-Mean</td><td>1.00 0.50</td><td>0.78 0.39</td><td>0.87 0.59</td><td>1.00</td><td>1.00 0.50</td><td>0.36 0.47</td><td>0.52 0.89</td></tr><tr><td></td><td></td><td></td><td></td><td>0.75</td><td></td><td>0.60</td><td>0.76</td></tr><tr><td>Centralized ideal reference FSM-GM</td><td>0.27</td><td>0.73</td><td>0.78</td><td>0.92</td><td>0.30</td><td>0.73</td><td>0.80 0.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Agent-Trim told a wrong</td><td></td><td></td><td>Byzantine rate</td><td>(true ±0.1)</td><td></td><td></td><td></td></tr><tr><td>Agent-Trim, -0.1 1.00</td><td></td><td>0.68</td><td>0.79</td><td>0.98</td><td>0.99</td><td>0.47</td><td>0.57</td></tr><tr><td>Agent-Trim, +0.1</td><td>1.00</td><td>0.60</td><td>0.75</td><td>0.95</td><td>1.00</td><td>0.06</td><td>0.87 0.35 0.78</td></tr></table>

![](images/b35da997acc155aebc67af9dc1eb16589ffb9fa19a3eb5517d45f69d1e42fd34.jpg)  
Figure 3: Normalized quality Q in the hardest setting $( \beta _ { c } = \beta _ { a } = 0 . 3 ) ,$ mean and standard deviation over 3 seeds.

![](images/cab05d954e93933cdd1fa92a1341a7baf012b5478c7a5ce7cd0d4a0447186e4b.jpg)  
Figure 4: Good clients’ objective of the top-r set of x per round $( \beta _ { c } = \beta _ { a } = 0 . 3 ,$ , mean over 3 seeds).

![](images/73cff4ceaff22c3076c7be8f09a8dd254bf5188561f4be77de72f0d5d7d5eceb.jpg)  
Figure 5: Eficiency summary: download per agent and per client in each round, averaged over all settings. Baselines are shown with the clients or the agents (vote) computing the global aggregate.

![](images/985c11fd7715393b575500e19a35f7176e0f649ef4cc4e22074c02d165650c88.jpg)

![](images/da7c40d2f8e528ac716bc5f56c234f0fc4c81f9ff8fda77069af01a3e16db3ef.jpg)  
Figure 6: Stability summary: average against worst-run quality for each method and attack, over all Byzantine rates at $\alpha = 0 . 5$ (filled, 3 seeds per setting) and at extreme heterogeneity $\alpha = 0 . 0 1 , \beta _ { c } = \beta _ { a } = 0 . 3$ (open, 10 seeds). Dashed: worst run equal to average.

Figure 7: Safety deficit against quality for every run of the default matrix (3 seeds; circles CIFAR10, triangles PathMNIST). Lines: average and worst quality of the DeFSM, Agent-GM, and Agent-Mean runs per deficit bin of width 0.1.  
Table 4: Extreme heterogeneity (Dirichlet $\alpha = 0 . 0 1 , \beta _ { c } = \beta _ { a } = 0 . 3 )$ , 10 seeds: avg / worst normalized quality Q. Bold: best worst case.
<table><tr><td>Dataset</td><td>Attack</td><td>DeFSM Agent-GM</td><td>Agent-Trim</td><td>Agent-Mean</td></tr><tr><td rowspan="2">CIFAR10</td><td>Include 0.85/ /0.75</td><td>0.91 / 0.82 0.83 / 0.62</td><td>0.70 / 0.56</td></tr><tr><td>Near-I. 0.65 / 0.55</td><td>0.63 / 0.39 0.59 / 0.05</td><td>0.50 / 0.40</td></tr><tr><td>PathMNIST Include</td><td>Mimic 0.78 / 0.64</td><td>0.84/ 0.72 0.94 / 0.79</td><td>0.62 / 0.47</td></tr><tr><td rowspan="2"></td><td>0.77 / 0.63</td><td>0.85 / 0.68 0.55 / 0.40</td><td>0.62 / 0.45</td></tr><tr><td>Near-I. 0.59 / 0.43 0.62 / 0.48 Mimic 0.72 / 0.580.81 / 0.65</td><td>30.49 / −0.04 0.61 / 0.16</td><td>0.53 / 0.42 0.63 / 0.53</td></tr></table>

## 6.2.2. Eficiency (H2)

Fig. 5 places each method on the plane of agent and client download. In DeFSM, the contract decides from $2 M = 4 0$ numbers, an agent downloads only its L client updates (200 entries), and a client downloads 9.3 agent aggregates on average (487 entries), within the bound D + 2 of Corollary 1. Since the contract cannot compute Agent-GM, either each agent also downloads all agent aggregates (1,226 entries, six times more) and the agents vote, or each client downloads all 19.9 of them (1,026 entries, twice as many). Agent-Mean costs about the same. Agent-Trim is cheaper because its agent aggregates are nearly binary, but it is the least reliable robust option, as we show next. In every method, on-chain storage is independent of d. These results support H2.

## 6.2.3. Robustness (H3)

Stability at a glance. In Fig. 6, the points of DeFSM and Agent-GM stay together in the upper band at both heterogeneity levels. Agent-Mean sits lower, and Agent-Trim reaches high averages but has the lowest worst runs, down to −0 04 at $\alpha = 0 . 0 1$

![](images/1523c384c4c9aa058e6cbc1bee4df77dec6e851417b583ff9fe74352a22510b0.jpg)

![](images/b2ab62cd9a43928447b89768f4cd99d80e1dd5a411e5346a544207e78b89aa85.jpg)

![](images/8ea8ea567c0a66084e955b2de12809cd99bd9bd900dcfb9daa9cc69dc214ce3f.jpg)

![](images/f08a840c5bd6b074f5b6ed2376cbdc738a1577a00d6abeb14fed1938b54c75fa.jpg)  
Figure 8: Efect of (a) the number of clients N (PathMNIST, 10 seeds), (b) the number of agents M, and (c) the sample size L (both datasets, 5 seeds) on the average and worst normalized quality of DeFSM and Agent-GM, at $\beta _ { c } = \beta _ { a } \stackrel { - } { = } 0 . 3$ under Near-Include and Mimic, varying one parameter at a time. (d) Agent aggregates a client downloads per round.

Attacks and Byzantine rates. Over the default matrix (Table 3), the worst run of DeFSM is 0.63 on CIFAR10 and 0.61 on PathMNIST. The other rules each fail somewhere. Agent-Mean loses up to 0.6 of the good clients’ quality. Agent-Trim’s worst run on PathMNIST is 0.36, although it knows the exact Byzantine rates. When the rate it is told is of by 0 1, its worst run on PathMNIST falls to 0.06. DeFSM and Agent-GM need no such knowledge.

Deficit drives quality. Theorem 1 predicts that the quality degrades only through the safety deficit. Fig. 7 confirms it across all runs of DeFSM, Agent-GM, and Agent-Mean: the lower the deficit, the higher the quality, with a rank correlation of −0 86 (−0 75 to −0 85 within each method). The average and worst quality fall steadily, from 0.97 and 0.90 below $\sigma = 0 . 1$ to 0.64 and 0.39 at $\sigma = 0 . 4 – 0 . 5 .$ The condition is only suficient: Agent-Trim’s deficit is near 1 because its aggregates are nearly zero, yet its quality varies widely. The deficit of DeFSM never exceeds 0.44.

Extreme heterogeneity. Table 4 sets the Dirichlet concentration to 0.01, so that most clients hold a single class and good updates difer widely; we use 10 seeds here. Agent-GM or Agent-Trim is the most accurate on average except under Near-Include on CIFAR10, where DeFSM leads in both average (0.65 against 0.63) and worst run (0.55 against 0.39). Over all our experiments with the default sample size L = 20 (Tables 3 and 4, and Fig. 8(a)), the worst run of DeFSM is 0.43, against 0.39 for Agent-GM and Agent-Mean and −0 04 for Agent-Trim; DeFSM and Agent-GM each have 4 runs out of 240 below 0.5, Agent-Mean 22, and Agent-Trim 27.

Deep point at the agent level. Replacing the SafeGM point at the agent level by the mean of the distinct agent aggregates, with SafeGM unchanged inside each agent, lowers the quality at $\beta _ { c } = \beta _ { a } = 0 . 3$ from 0.79 to 0.67 on CIFAR10 and from 0.79 to 0.68 on PathMNIST, and raises the deficit from 0.29–0.31 to 0.41–0.42. The mean has the smaller worst-case bound (Section 5.2), but the deep point gives bad agent aggregates a much smaller share in practice. These results support H3: DeFSM degrades gracefully, as stably as Agent-GM, while Agent-Mean and Agent-Trim collapse in some settings.

## 6.2.4. Design Parameters (H4)

Number of clients. Fig. 8(a) increases the number of clients to 200 and 500 with the sample size fixed at $L = 2 0$ The deficit of DeFSM stays essentially unchanged (0.42–0.43 under Near-Include and 0.31–0.34 under Mimic), as the analysis predicts: its tolerance depends on the Byzantine client rate and $L ,$ not on N. Its quality decreases slightly with N, since each agent sees a smaller fraction of the clients, and stays within 0.05 of Agent-GM on average; its worst run is higher than Agent-GM’s at N = 200 (0.45 against 0.39).

Number of agents. The quality of DeFSM hardly changes with M (Fig. 8(b)), while the number of agent aggregates a client downloads grows linearly with M (5.7, 9.3, and 17.7; Fig. 8(d)), since the coeficients spread over at least $D \ge \lceil M / 3 \rceil$ of them, whereas Agent-GM at the clients needs all of them (10, 20, and 39). M therefore only needs to be large enough to keep the agent-level term of Theorem 2 small.

Sample size. A larger sample lowers the deficit of DeFSM (0.42, 0.37, and 0.34 for $L = 1 0 , 2 0 ,$ , and 40), as the analysis of $\rho _ { L }$ predicts (Table 1). Small samples hurt DeFSM more than Agent-GM (Fig. 8(c)): with $L = 1 0$ its worst run drops to 0.40, against 0.48 for Agent-GM, because an agent’s sample then exceeds the planar tolerance more often. These results support H4 and suggest $L \geq 2 0$

![](images/1461b880f0dec4df263b8856ca5b700ca2042e9953ff0150b31ca2a634a7ae0f.jpg)

![](images/53d266fe06675a296fb47ee2167f63e097ad045edb4748e40720731cdc3d08e1.jpg)  
Figure 9: Quality against download per round. Small points: one per setting of the default matrix (mean over 3 seeds). Diamond: average; open triangle: worst setting. Baselines compute the global aggregate at (a) the clients or (b) the agents, with a vote. Dotted: Agent-Trim told a Byzantine rate 0.1 too high. Dashed: FSM-GM.

## 6.2.5. Overall Picture

Fig. 9 combines accuracy and cost, with one point per method and setting of the default matrix. DeFSM needs about half the client download of Agent-GM and Agent-Mean, and a sixth of their agent download when they vote, and its points stay in a narrow band: its worst setting matches Agent-GM’s and is well above Agent-Mean’s. Agent Trim is cheaper and more accurate on average, but spreads widest, and a misestimated rate drops its worst setting to 0.35. Only DeFSM combines low cost, stable quality, and no need to know the Byzantine rates. As for the design parameters, DeFSM scales to many clients, the number of agents trades only client download, and the sample size is the lever on robustness.

## 7. Conclusions

Submodular maximization is a core tool in machine learning, and its federated version must work where a trusted server is infeasible. Removing the server creates a tension between decentralization, eficiency, and security. Our DeFSM architecture resolves it with a lightweight smart contract and a pool of agents that need no consensus, and its aggregation rule SafeGM is provably robust. We learned that robustness need not be judged by distances in the full space: a depth-certified decision in two dimensions bounds the influence of bad updates by their count, this bound controls the solution quality, and in practice it comes close to the geometric median at a fraction of the communication, without consensus or knowledge of the Byzantine rates. The impact reaches beyond FSM. It makes robust federated optimization deployable on public blockchains, where anyone can serve as an agent and no party must be trusted; hospitals, for example, could jointly curate the most informative medical images without a common coordinator. It suggests a general pattern for verifiable decentralized computing: of-chain workers do the high-dimensional work, and the chain decides on a low-dimensional certificate. It also brings combinatorial tools such as Tukey depth, long confined to very low dimension, to high-dimensional systems, with guarantees that hold by counting rather than by distance. In future work, we will implement DeFSM on a real blockchain and measure its gas fees, storage and communication overheads, and latency per round to assess its feasibility in practice.

## CRediT authorship contribution statement

Duc A. Tran: Conceptualization, Methodology, Writing – original draft, Writing – review & editing. Dung Truong: Conceptualization, Software, Data curation. Duy Le: Conceptualization, Software.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Data availability

All simulation code, data, and results are available at https://github.com/UMB-NCLab/defsm.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work we used AI as a plotting tool for the illustrative diagram in Figure 1 and the evaluation figures, where we input our own conceptual description and numerical values. We also used Claude AI in part of our coding for the evaluation study, and to improve the language and presentation of the manuscript. After all this AI help, we reviewed and edited the content to ensure correctness, and take full responsibility for the content of this article. We confirm that all the main, intellectual contents (ideas, literature review, algorithms, analysis, evaluation) are our own.

## References

[1] A. Rafiey, Decomposable submodular maximization in federated setting, in: Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, OpenReview.net, 2024.

[2] Y. Wang, T. Zhou, C. Chen, Y. Wang, Federated submodular maximization with diferential privacy, IEEE Internet of Things Journal 11 (2) (2024) 1827–1839. doi:10.1109/JIOT.2023.3324801.

[3] D. A. Tran, D. Truong, D. Le, RobustFSM: Submodular maximization in federated setting with malicious clients, in: Proceedings of the IEEE International Conference on Big Data (IEEE BigData 2025), Macao - China, 2025.

[4] K. Pillutla, S. M. Kakade, Z. Harchaoui, Robust aggregation for federated learning, IEEE Transactions on Signal Processing 70 (2022) 1142–1154.

[5] D. Yin, Y. Chen, K. Ramchandran, P. L. Bartlett, Byzantine-robust distributed learning: Towards optimal sta tistical rates, in: Proceedings of the 35th International Conference on Machine Learning (ICML), Vol. 80 of Proceedings of Machine Learning Research, 2018, pp. 5650–5659.

[6] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, J. Stainer, Machine learning with adversaries: Byzantine tol erant gradient descent, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, Curran Associates Inc., Red Hook, NY, USA, 2017, p. 118–128.

[7] N. H. Vaidya, V. K. Garg, Byzantine vector consensus in complete graphs, in: Proceedings of the 2013 ACM Symposium on Principles of Distributed Computing, PODC ’13, Association for Computing Machinery, New York, NY, USA, 2013, p. 65–73. doi:10.1145/2484239.2484256.

[8] G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, An analysis of approximations for maximizing submodular set functions–I, Math. Program. 14 (1) (1978) 265–294. doi:10.1007/BF01588971.

[9] U. Feige, A threshold of ln n for approximating set cover, Journal of the ACM 45 (4) (1998) 634–652.

[10] R. Barbosa, A. Ene, H. Nguyen, J. Ward, The power of randomization: Distributed submodular maximization on massive datasets, in: F. Bach, D. Blei (Eds.), Proceedings of the 32nd International Conference on Machine Learning, Vol. 37 of Proceedings of Machine Learning Research, PMLR, Lille, France, 2015, pp. 1236–1244.

[11] B. Mirzasoleiman, A. Karbasi, R. Sarkar, A. Krause, Distributed submodular maximization, J. Mach. Learn. Res. 17 (1) (2016) 8330–8373.

[12] J. Vendrell, S. Kia, Optimality gap of decentralized submodular maximization under probabilistic communication (2024). arXiv:2409.09979. URL https://arxiv.org/abs/2409.09979

[13] T. D. Nguyen, T. Nguyen, P. L. Nguyen, H. H. Pham, K. Doan, K.-S. Wong, Backdoor attacks and defenses in federated learning: Survey, challenges and future research directions, Engineering Applications of Artificial Intelligence (2024).

[14] Y. Li, C. Chen, N. Liu, H. Huang, Z. Zheng, Q. Yan, A blockchain-based decentralized federated learning framework with committee consensus, IEEE Network 35 (1) (2021) 234–241. doi:10.1109/MNET.011.2000263.

[15] Z. Lin, Z. Wu, J. Zhu, J. Huang, A blockchain-based decentralized federated learning framework with dualcommittees consensus, in: ACAI ’22: Proceedings of the 2022 5th International Conference on Algorithms, Computing and Artificial Intelligence, ACAI ’22, Association for Computing Machinery, New York, NY, USA, 2023. doi:10.1145/3579654.3579700.

[16] E. W. R. Madill, B. Nguyen, C. K.-S. Leung, S. Rouhani, ScaleSFL: A sharding solution for blockchain-based federated learning, in: Proceedings of the Fourth ACM International Symposium on Blockchain and Secure Critical Infrastructure (BSCI), 2022, pp. 95–106. doi:10.1145/3494106.3528680.

[17] D. H. Nguyen, P. L. Nguyen, T. T. Nguyen, H. H. Pham, D. A. Tran, FedBlock: A blockchain approach to federated learning against backdoor attacks, in: Proceedings of the IEEE International Conference on Big Data (IEEE BigData 2024), Washington DC, 2024.

[18] Z. Yang, G. Li, J. Wu, W. Yang, Propagable backdoors over blockchain-based federated learning via samplespecific eclipse, in: GLOBECOM 2022 - 2022 IEEE Global Communications Conference, 2022, pp. 2579–2584. doi:10.1109/GLOBECOM48099.2022.10001370.

[19] R. A. Mallah, D. López, T. Halabi, Blockchain-enabled eficient and secure federated learning in IoT and edge computing networks, in: 2023 International Conference on Computing, Networking and Communications (ICNC), 2023, pp. 511–515. doi:10.1109/ICNC57223.2023.10074277.

[20] M. Cambus, D. Melnyk, T. Milentijevic, S. Schmid, Coordinate-wise median in Byzantine federated learning, in: Proceedings of the International Workshop on Secure and Eficient Federated Learning, FL-AsiaCCS ’25, Association for Computing Machinery, New York, NY, USA, 2025. doi:10.1145/3709023.3737691.

[21] M. Xie, J. Ma, G. Long, C. Zhang, Robust clustered federated learning with bootstrap median-of-means, in: Web and Big Data: 6th International Joint Conference, APWeb-WAIM 2022, Nanjing, China, November 25–27, 2022, Proceedings, Part I, Springer-Verlag, Berlin, Heidelberg, 2022, p. 237–250. doi:10.1007/ 978-3-031-25158-0\_19.

[22] E. M. El Mhamdi, R. Guerraoui, S. Rouault, The hidden vulnerability of distributed learning in Byzantium, in: J. G. Dy, A. Krause (Eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, Vol. 80 of Proceedings of Machine Learning Research, PMLR, 2018, pp. 3521–3530.

[23] X. Lyu, Y. Han, W. Wang, J. Liu, B. Wang, J. Liu, X. Zhang, Poisoning with Cerberus: stealthy and col luded backdoor attack against federated learning, in: Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’23/IAAI’23/EAAI’23, AAAI Press, 2023. doi:10.1609/aaai.v37i7.26083.

[24] Z. Gong, Y. Zhang, L. Y. Zhang, Z. Zhang, Y. Xiang, S. Pan, Not All Edges are Equally Robust: Evaluating the Robustness of Ranking-Based Federated Learning , in: 2025 IEEE Symposium on Security and Privacy (SP), IEEE Computer Society, Los Alamitos, CA, USA, 2025, pp. 2527–2545. doi:10.1109/SP61157.2025. 00251.

[25] H. Mendes, M. Herlihy, Multidimensional approximate agreement in Byzantine asynchronous systems, in: Pro ceedings of the Forty-Fifth Annual ACM Symposium on Theory of Computing, STOC ’13, Association for Computing Machinery, New York, NY, USA, 2013, p. 391–400. doi:10.1145/2488608.2488657.

[26] R. Rado, A theorem on general measure, Journal of the London Mathematical Society s1-21 (4) (1946) 291–300.

[27] T. M. Chan, An optimal randomized algorithm for maximum Tukey depth, in: Proceedings of the 15th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2004, pp. 430–436.

[28] J. Vondrak, Optimal approximation for the submodular welfare problem in the value oracle model, in: Proceedings of the Fortieth Annual ACM Symposium on Theory of Computing, STOC ’08, Association for Computing Machinery, New York, NY, USA, 2008, p. 67–74. doi:10.1145/1374376.1374389.

[29] G. Calinescu, C. Chekuri, M. Pál, J. Vondrák, Maximizing a monotone submodular function subject to a matroid constraint, SIAM J. Comput. 40 (6) (2011) 1740–1766. doi:10.1137/080733991.

[30] L. Lamport, R. Shostak, M. Pease, The byzantine generals problem, ACM Trans. Program. Lang. Syst. 4 (3) (1982) 382–401. doi:10.1145/357172.357176. URL https://doi.org/10.1145/357172.357176

[31] M. Castro, B. Liskov, Practical byzantine fault tolerance, in: Proceedings of the Third Symposium on Operating Systems Design and Implementation, OSDI ’99, USENIX Association, USA, 1999, p. 173–186.

[32] N. Megiddo, Linear programming in linear time when the dimension is fixed, Journal of the ACM 31 (1) (1984) 114–127.

[33] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith, Federated optimization in heterogeneous networks, in: I. S. Dhillon, D. S. Papailiopoulos, V. Sze (Eds.), Proceedings of Machine Learning and Systems 2020, MLSys 2020, Austin, TX, USA, March 2-4, 2020, mlsys.org, 2020.

[34] X. Li, K. Huang, W. Yang, S. Wang, Z. Zhang, On the convergence of fedavg on non-iid data, in: 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020, 2020.

[35] W. Hoefding, Probability inequalities for sums of bounded random variables, Journal of the American Statistical Association 58 (301) (1963) 13–30.

[36] A. Krizhevsky, G. Hinton, Learning multiple layers of features from tiny images, Tech. rep., University of Toronto, Technical Report CRG-TR-09-1 (2009). URL https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf

[37] J. Yang, R. Shi, D. Wei, Z. Liu, L. Zhao, B. Ke, H. Pfister, B. Ni, MedMNIST v2 – a large-scale lightweight benchmark for 2D and 3D biomedical image classification, Scientific Data 10 (2023) 41. doi:10.1038/ s41597-022-01721-8.

[38] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.