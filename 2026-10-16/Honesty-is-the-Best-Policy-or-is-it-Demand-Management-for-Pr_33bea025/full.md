# Honesty is the Best Policy ... or is it? Demand Management for Probabilistic Service Options

Daniela Sailer<sup>1</sup>, Robert Klein<sup>1</sup>, Claudius Steinhardt<sup>2</sup> daniela.sailer@wiwi.uni-augsburg.de, robert.klein@wiwi.uni-augsburg.de, claudius.steinhardt@unibw.de

<sup>1</sup> Chair of Analytics & Optimization, University of Augsburg <sup>2</sup> Chair of Business Analytics & Management Science, University of the Bundeswehr Munich

## Abstract

To enable well-informed customer choices and redistribute demand across service options, providers can display probabilities regarding predicted service levels at the time of booking, efectively ofering probabilistic service options. In many service settings, the underlying truthful probabilities endogenously depend on the system state, rather than being static product attributes controlled by the provider. While customers rely on this information when making decisions, they cannot readily verify its accuracy. Consequently, providers may be tempted to manipulate the displayed probabilities to guide customers’ choices in their favor, creating a novel lever for demand management. In this work, we model the resulting problem as a sequential decision process, define truthful probabilities recursively, and introduce transparency constraints that limit their distortion. Because the exact computation of truthful probabilities is intractable for realistic settings, we develop an adapted reinforcement learning algorithm to approximate them. Furthermore, we propose a sampling-based mixed-integer program to optimize the information communicated by the provider in each state. Our framework enables providers to gauge the benefits of dynamically disclosing probabilistic information at the time of booking and sheds light on incentives for manipulation. Our computational study is motivated by out-of-home delivery, i.e., the delivery to parcel lockers or shops, and reveals a trade-of between individual-level transparency and aggregate customer utilities.

Keywords: Probabilistic Products, Demand Management, Reinforcement Learning

## 1 Introduction

Services are fundamentally characterized by the active participation of customers in the value creation process (Bordoloi et al. 2023, Chapter 1). From the provider’s perspective, this introduces substantial uncertainty regarding the available capacity given that the amount or duration of resources being blocked may also depend on customers’ behavior. Consequently, guaranteeing service provision for every advance reservation would require a conservative definition of feasibility based on worst-case assumptions regarding resource availability, which entails high costs in the form of foregone revenue and underutilized capacity (Sailer et al. 2026).

To sidestep strict feasibility requirements at the time of booking, service providers have devised strategies to deal with insuficient capacity ex post at the time of service provision. On the one hand, providers may deny the originally promised service, resorting to backup options or compensation schemes. Examples include widespread overbooking practices in the travel industry, preemptible instances in cloud computing, or the rerouting of parcels in out-of-home delivery networks. On the other hand, providers may still accommodate all customers at the cost of degraded service quality, such as throttled speed at electric vehicle charging stations or standing passengers on public transit. In both cases, customers systematically prefer one service outcome (receiving service, higher quality) over the other.

While these strategies enable providers to eficiently manage uncertainty, they shift most of the associated risks and inconvenience to customers, causing dissatisfaction. To mitigate this, the provider can share information on predicted service levels at the time of booking, thereby unlocking two key advantages: First, communicating predicted congestion or availability supports well-informed customer choices that align with individual preferences and risk tolerance. Second, the information simultaneously serves as a demand-shaping instrument because it induces customers to redistribute themselves across service options, reducing ineficient demand concentration and the need for ex post interventions.

In this work, we provide such information in the form of probabilities, thereby constructing probabilistic service options. The probabilities are displayed to dynamically arriving requests at the time of booking and represent the likelihood of receiving a capacitated premium outcome (the higher quality level or not being denied service altogether) at the time of service provision. With the remaining probability, the customer receives the service option’s base outcome. Essentially, the provider ofers an assortment of probabilistic service options to each incoming request. In response, the customer selects one of them (possibly including the alternative not to book), turning the request into an order. At specific points in time, service outcomes are allocated to orders. We allow for resources to be reusable, i.e., customers make use of the capacity for a (potentially random) duration, and it becomes available again afterward. In total, we consider three sources of uncertainty: request arrival, customer choice, and usage duration.

From a conceptual perspective, these probabilistic service options resemble probabilistic products (Fay and Xie 2008, G¨onsch 2020), i.e., an ofering that comprises multiple component products, each associated with a probability. Existing research on probabilistic products investigates probabilities as a static design choice, set exogenously by the provider prior to selling. In contrast, the stochasticity of the realized outcome in service settings is not the provider’s deliberate choice, but naturally emerges from the uncertain availability of capacity. Such uncertainty may stem from random capacity usage durations, customer no-shows, or orders displacing earlier arrivals if allocation decisions do not follow the first-come-first-served principle. As a result, the truthful probabilities endogenously depend on the state of the system, potentially including the characteristics of the current request, and vary dynamically during the selling horizon

Unlike traditional supervised learning, simply estimating the probabilities from historical data would lead to biased predictions because the data reflects customers’ choices under probabilities displayed in the past. Once the provider communicates newly estimated probabilities, customers choices adapt in response, and hence the underlying true probabilities change, resulting in a feedback loop. To communicate accurate probabilities to sequentially arriving requests, the provider must forecast demand, anticipate choice behavior of future requests in response to probabilities expected to be displayed in subsequent periods, predict allocation decisions, and account for units already sold along with current utilization. For realistic problem sizes, a vast space of possible states paired with the continuous probability space and customers’ expectation to receive probabilities in rea time pose significant challenges.

## 1.1 Contributions

To the best of our knowledge, we are the first to study endogenously probabilistic products, thereby enabling providers to assess the benefits and implications of dynamically communicating probabilistic information. Moreover, because customers cannot readily verify their accuracy, providers may be tempted to manipulate the displayed probabilities to guide customers’ choices in their favor. Beyond controlling product availability or prices, this creates a novel, information-based lever for demand management that may call for regulation. In response, we propose transparency constraints that limit the deviation between displayed and truthful probabilities, with full transparency referring to the case where the provider is honest and always reports probabilities equal to the truthful ones. Our work provides methodology to quantify the extent of the provider’s incentive for distortion and assess how transparency constraints afect the involved stakeholders.

With our work, we make the following contributions:

• Modeling framework. We formalize the problem as an infinite-horizon sequential decision process. As key modeling contributions, we formulate a recursive equation to compute truthful, endogenous probabilities under a given policy and introduce transparency constraints.

• Solution methodology. As a theoretical insight, we establish that, for certain allocation schemes, there exists exactly one feasible respective solution for the displayed probabilities under full transparency. For realistic problems, the exact computation of truthful probabilities is intractable. In response, we propose an adapted reinforcement learning algorithm to train an approximation. To optimize displayed probabilities, we develop a mixed-integer program that incorporates customer choice behavior through a sampling approach.

• Managerial insights. Our computational study is motivated by out-of-home delivery (OOHD), i.e., the delivery to parcel lockers or shops instead of the recipient’s doorstep (Janinhof et al. 2024). In practice, providers reroute parcels to alternative facilities if the one selected by the customer is fully occupied, efectively following an ex post strategy to handle insuficient capacity. Given the substantial cost savings associated with OOHD, providers seek to encourage adoption by maximizing customer satisfaction. Our study yields three core insights:

First, while truthful probability disclosure already improves aggregate customer satisfaction substantially, demand management yields further gains through targeted adjustments of the displayed probabilities. Notably, marginal gains are diminishing with the permitted amount of probability distortion. Second, the value of information provision and demand management is greatest when customers strongly prefer scarce premium outcomes and for high-density networks that ofer numerous suitable alternatives. Third, these aggregate benefits must be weighed against systematic utility gains and losses among customers as well as potential longterm efects on credibility.

## 1.2 Related Work

Our work lies at the intersection of probabilistic selling, revenue management, and information provision. We concisely review the corresponding literature streams, delineating our contributions from existing work.

## 1.2.1 Probabilistic Selling

Existing studies on probabilistic selling exclusively take on a strategic perspective, comparing selling strategies (Fay and Xie 2008, Guo et al. 2023) or investigating product design (Chen et al. 2021, Zhang et al. 2015). With the exception of Mitter et al. (2026), publications predominantly address settings with a single probabilistic product, typically comprising two vertically diferentiated component products (Chen et al. 2024, Guo et al. 2023, Ren and Huang 2022, Zhang et al. 2015).

Across the entire literature stream, allocation probabilities are assumed to be static. They are either decided by the seller (Mitter et al. 2026) or directly result from the inventory at the start of the selling horizon (Ren and Huang 2022, Zhang et al. 2015). Such static probabilities are justified if there are no capacity constraints (Chen et al. 2021) or a fixed capacity is reserved for the probabilistic product without availability information being revealed over time (Yang et al. 2026).

Taking on a novel operational perspective, we extend probabilistic selling to settings with dynamic, endogenous probabilities for assortments comprising multiple probabilistic products, thereby transforming it into an instrument to improve transparency in the face of inherently uncertain capacity.

## 1.2.2 Revenue Management

Similar to probabilistic products, flexible products in revenue management allow sellers to allocate an associated component product after the sale, but without announcing probabilities beforehand (G¨onsch 2020). The literature treats such oferings as a deliberate design choice to gain operational flexibility (G¨onsch et al. 2014) and an additional dimension for market segmentation by targeting customers willing to accept uncertainty, typically in exchange for a lower price (Gallego and Phillips 2004).

Research in this domain focuses on dynamic decision making, controlling product availability (Cheung and Simchi-Levi 2016, Koch et al. 2017) or prices (Strauss et al. 2021) over time. The sale of a flexible product is permitted even when some of its component products are guaranteed to be unavailable, as long as at least one feasible allocation exists (Petrick et al. 2010). While allocation typically occurs once, i.e., within (Petrick et al. 2012) or at the end of the finite selling horizon, Zhu and Topaloglu (2024) also consider periodic allocation decisions.

Transferring the operational perspective of revenue management to probabilistic selling, we employ information provision in the form of allocation probabilities as a novel mechanism to dynamically steer demand instead of availability or prices.

## 1.2.3 Information Provision

Beyond probabilities, the literature proposes other forms of information that providers may share with customers:

• In queuing-based service systems, the current queue length or estimated waiting time help customers decide whether to join the queue (Ibrahim 2018). Going beyond settings where service is executed as soon as possible, our work extends to advance reservations, incorporating a deliberate time lag between booking and service provision commonly encountered in practice.

• A second branch of literature examines messages sent to all or a subset of customers, e.g., regarding remaining inventory. In the majority of publications, messages to all targeted customers are dispatched simultaneously at a single point in time rather than sequentially (Allon and Bassamboo 2011, Drakopoulos et al. 2021, Cui and Shin 2018). While Simhon and Starobinski (2018) dynamically display the total number of unsold seats over a finite booking horizon, our framework features uncertain and reusable capacity. As a result, the available capacity at the time of service provision cannot be determined with certainty in advance in our case because it does not directly follow from the number of units sold, necessitating more nuanced information.

• In the context of restaurant meal delivery, Hildebrandt and Ulmer (2022) focus on accurately predicting estimated arrival times for a given routing and assignment policy. In comparison, we additionally consider the notion of deliberately distorting information to manage demand.

## 1.3 Organization of the Paper

The remainder of this paper is structured as follows: In Section 2, we explain our modeling framework. Section 3 is dedicated to our solution methodology. We present the results of our computational study in Section 4 and conclude the paper with future research directions in Section 5.

## 2 Modeling Framework

In this section, we define our problem (Section 2.1) and formalize it as a sequential decision process (Section 2.2). As a novel modeling component, we propose a recursive equation to compute truthful allocation probabilities (Section 2.3). Based on them, we formulate transparency constraints, resulting in a constrained variant of the sequential decision process (Section 2.4). Appendix A provides an overview on the notation.

## 2.1 Problem Definition

We denote the set of all service outcomes by $\mathcal { T } = \mathcal { T } ^ { - } \cup \mathcal { T } ^ { + } = \{ 1 , \ldots , I \}$ , partitioned into base outcomes $\mathcal { T } ^ { - }$ and premium outcomes ${ \mathcal { T } } ^ { + }$ . The capacity $Q _ { i }$ of premium outcomes $i \in \mathcal { I } ^ { + }$ is limited, whereas capacity constraints for base outcomes are assumed to be negligible, either because capacity is unrestricted or has been verified as suficient ex ante. Each probabilistic service option $j \in \mathcal { I } ^ { P }$ corresponds to a unique pair $( i _ { j } ^ { - } , i _ { j } ^ { + } )$ of a base outcome $i _ { j } ^ { - } \in \mathcal { I } ^ { - }$ and a premium outcome $i _ { j } ^ { + } \in \mathcal { I } ^ { + }$ Across diferent application contexts, examples include indoor versus terrace seating at a restaurant, delivery to a backup parcel shop versus the originally selected parcel locker, and slow versus fast electric vehicle charging. Base outcomes may additionally be ofered as deterministic options $j \in \mathcal { I } ^ { D }$ with $i _ { j } ^ { + } = i _ { j } ^ { - } = i _ { j } \in \mathcal { T } ^ { - }$ . The possibility that the customer chooses not to book can also be modeled with an artificial deterministic option $j \in \mathcal { I } ^ { D }$ . Set $\mathcal { I } = \mathcal { I } ^ { P } \cup \mathcal { I } ^ { D }$ represents all ofered options.

The planning horizon is divided into time periods such that at most one request arrives per period, yielding a set of discrete points in time $t \in { \mathcal { T } } = \left\{ 1 , \dots , \infty \right\}$ . Whenever a new request arrives, the provider displays a probability $p _ { j }$ for every probabilistic service option $j \in \mathcal { I } ^ { \mathcal { P } }$ (e.g., restaurants, delivery locations, charging stations). A customer choosing $j \in \mathcal { I } ^ { \mathcal { P } }$ expects to receive the premium outcome $i _ { j } ^ { + }$ with probability $p _ { j } \in [ 0 , 1 ]$ and the base outcome $i _ { j } ^ { - }$ with the complementary probability $1 - p _ { j }$ . Based on the displayed probabilities, the customer chooses one of the options $j \in \mathcal { I }$ , and the request turns into an order.

At designated points in time $t \in \mathcal { T } ^ { A } \subset \mathcal { T }$ , the provider fulfills orders scheduled for service provision by allocating service outcomes. For each order $^ { O , }$ the point in time $t _ { o } ^ { A } \in \mathcal { T } ^ { A }$ at which allocation for o must occur is known and cannot be pre- or postponed. Allocating a premium outcome temporarily reduces the available capacity for a (stochastic) amount of time. Because our analysis focuses on the role of (in)transparency in displayed probabilities as a demand management lever, we assume that the provider does not strategically withhold available capacity during allocation and always allocates the premium outcome to an order if suficient capacity is available. Otherwise, we would introduce a second source of allocation-related intransparency. The provider seeks to maximize the expected average reward, e.g., profit or customer satisfaction.

## 2.2 Sequential Decision Process

Leveraging the framework by Powell (2022), we formalize the basic problem without transparency constraints as a sequential decision process.

## 2.2.1 Decision Epoch

A decision epoch $k = 1 , \ldots , \infty$ occurs whenever a new request arrives, prompting the provider to specify probabilities, or the system reaches a point in time for service provision, necessitating an allocation decision.

## 2.2.2 Pre-Decision State

The pre-decision state variable $S _ { k }$ comprises four components:

1. Time. The current point in time is denoted by $t _ { k }$

2. Request. We encode the newly arrived request with $r _ { k }$ . Beyond the finite point in time $t _ { r _ { k } } ^ { A } \in \mathcal { T } ^ { A }$ for allocation, the provider may observe further request attributes, such as the customer’s segment, that correlate with choice behavior or the capacity usage duration. We use $r _ { k } = 0$ to model the case of no request arrival.

3. Pending orders. The set $\mathcal { O } _ { k }$ represents all pending orders, i.e., orders that have not been allocated yet. For each pending order $^ { O , }$ we track the service option $j _ { o }$ chosen by the customer and the point in time $t _ { o } ^ { A } \in \mathcal { T } ^ { A }$ at which one of the eligible service outcomes $i _ { j _ { o } } ^ { - }$ and $i _ { j _ { o } } ^ { + }$ must be allocated. In addition, further information such as the order’s arrival time may be necessary (Section 4).

4. Capacity utilization. For each premium service outcome $i \in \mathcal { T } ^ { + }$ , the set $\mathcal { C } _ { i k }$ contains all orders that currently occupy its capacity as a result of previous allocation decisions, including information on the amount of blocked capacity and the elapsed usage duration. From this, we can infer the available capacity $q _ { i } ^ { k }$ in decision epoch k. We summarize the capacity utilization with $\mathcal { C } _ { k } = \{ \mathcal { C } _ { i k } : i \in \mathcal { I } ^ { + } \}$

In summary, we define the pre-decision state variable as $S _ { k } = ( t _ { k } , r _ { k } , \mathcal { O } _ { k } , \mathcal { C } _ { k } )$ for $k > 0$ with initial state $S _ { 0 }$ and model the set of all possible pre-decision states with the pre-decision state space S.

## 2.2.3 Decision

We generally denote decisions by $X _ { k } \ \in \ { \mathcal { X } } ( S _ { k } )$ with the decision space $\mathcal { X } ( S _ { k } )$ encompassing all feasible decisions in pre-decision state $S _ { k }$ . There are two types of decisions:

1. Probability display. In case of a request arrival $( r _ { k } \neq 0 )$ , the provider displays a vector of probabilities $P _ { k } = ( p _ { j } ^ { k } ) _ { j \in \mathcal { I } ^ { P } }$ with $p _ { j } ^ { k } \in [ 0 , 1 ]$ for the probabilistic service options.

2. Allocation. If $t _ { k } \in \mathcal { T } ^ { A }$ , the provider allocates service outcomes to the relevant pending orders $\mathcal { O } _ { k } ^ { A } = \{ o \in \mathcal { O } _ { k } : t _ { o } ^ { A } = t _ { k } \}$ . For orders with $j _ { o } \in \mathcal { I } ^ { P }$ , the binary decision variable $a _ { o } ^ { k }$ equals 1 if order $o \in \mathcal { O } _ { k } ^ { A }$ receives the premium outcome $i _ { j _ { o } } ^ { + }$ of the chosen probabilistic service option $j _ { o } ,$ and 0 otherwise. Customers who chose a deterministic option $j _ { o } \in \mathcal { I } ^ { D }$ directly receive the associated outcome $i _ { j } .$ , and we set $a _ { o } ^ { k } = 1$ . Due to the structure of service options, allocation decisions $A _ { k } = ( a _ { o } ^ { k } ) _ { o \in \mathcal { O } _ { k } ^ { A } }$ can be decomposed by premium outcome $i \in \mathcal { T } ^ { + }$ and are constrained by the respective available capacity $q _ { i } ^ { k }$

Although the two decision spaces are strongly interrelated (Section 3), we assume that they are decoupled in the sense that displayed probabilities only afect the option chosen by the customer. Accordingly, it sufices to track the customer’s choice in the state variable, without storing the display that led to that choice. In particular, the provider does not allocate the premium outcome in case of insuficient capacity even when the originally displayed probability equals one, nor does the allocation decision space compel the provider to allocate the base outcome when the displayed probability is zero. Instead, we ensure consistency between decision types through transparency constraints (Section 2.4). To avoid concurrent probability display and allocation in a single decision epoch, we assume $r _ { k } = 0 { \mathrm { ~ i f ~ } } t _ { k } \in T ^ { A }$ without loss of generality.

## 2.2.4 Transitions and Exogenous Information

The exogenous information $W _ { k + 1 } = ( j _ { r _ { k } } , \mathcal { C } _ { k + 1 } , r _ { k + 1 } , t _ { k + 1 } ) \in \mathcal { W }$ is revealed during the transition from decision epoch k to $k + 1$ , i.e., after the decision $X _ { k }$ . It follows a known stochastic process with probabilities $\mathbb { P } ( W _ { k + 1 } | S _ { k } , X _ { k } )$ and represents the three sources of uncertainty in our problem:

1. Customer choice. The customer belonging to request $r _ { k } \neq 0$ associates utilities $u _ { i } ^ { k }$ with the service outcomes $i \in \mathcal { Z }$ , which are private information not observed by the provider. Based on the provider’s announced probabilities $P _ { k }$ , the customer forms valuations $U _ { j } ^ { k } = U ( P _ { k } , u _ { i _ { j } } ^ { k } , u _ { i _ { j } } ^ { k } )$ for the probabilistic service options $j \in \mathcal { I } ^ { P }$ , with function U reflecting the customer’s risk preference. The valuation for deterministic options $j \in \mathcal { I } ^ { D }$ equals $U _ { j } ^ { k } = u _ { i _ { j } } ^ { k }$ . The customer chooses the option with the highest valuation $j _ { r _ { k } } \ = \ \arg \operatorname* { m a x } _ { j \in \mathcal { I } } U _ { j } ^ { k }$ . After revealing the customer’s choice $j _ { r _ { k } }$ , the request turns into an order, yielding an updated set of pending orders $\mathcal { O } _ { k + 1 }$ . In case of no request $( r _ { k } = 0 )$ , we set $j _ { r _ { k } } = 0$

2. Capacity utilization. Since usage durations are stochastic (and may even take on the value zero to model no-shows), the capacity utilization $\mathcal { C } _ { k + 1 }$ in decision epoch $k + 1$ is not merely the result of a preceding allocation decision $X _ { k } ~ ( \mathrm { i f } ~ t _ { k } \in \mathcal { T } ^ { A } )$ , but is also afected by exogenous information about capacity units becoming available again.

3. Request arrival (or lack thereof). The subsequent decision epoch $k + 1$ is either triggered by a new request arrival $( r _ { k + 1 } \neq 0 )$ or an allocation decision $( r _ { k + 1 } = 0 )$ , determining the next point in time $t _ { k + 1 }$

State transitions after allocation decisions additionally include a deterministic update of the pending orders $\mathcal { O } _ { k + 1 }$ by removing allocated orders $\mathcal { O } _ { k } ^ { A }$ from $\mathcal { O } _ { k }$

## 2.2.5 Reward

We express rewards as a function $R ( S _ { k } , X _ { k } , W _ { k + 1 } )$ . Depending on the objective, rewards may arise from the customer’s choice revealed in $W _ { k + 1 } , \ \mathrm { e . g . }$ , if the provider aims to maximize the revenue earned from the sale of service options. Alternatively, rewards may result from allocation decisions if the specific allocated service outcome is relevant, e.g., to maximize customer satisfaction. In our computational study in Section 4, both the displayed probabilities and allocation decisions shape the reward signal based on customers’ expected conditional utilities.

## 2.2.6 Policy and Objective Function

The solution to a sequential decision process is a policy $\pi \in \Pi$ that maps states to decisions, denoted by $X ^ { \pi } ( S _ { k } )$ . The optimal policy $\pi ^ { * }$ maximizes the objective, i.e., the expected average reward per decision epoch $\bar { R } ^ { \pi }$ (Sutton and Barto 2018, Chapter 10.3):

$$
\operatorname* { m a x } _ { \pi \in \Pi } \bar { R } ^ { \pi } = \operatorname* { m a x } _ { \pi \in \Pi } \left\{ \operatorname* { l i m } _ { K \to \infty } \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { E } ^ { \pi } \left[ R ( S _ { k } , X ^ { \pi } ( S _ { k } ) , W _ { k + 1 } ) | S _ { 0 } \right] \right\}\tag{1}
$$

## 2.3 Post-Choice State and Truthful Probabilities

To compute truthful allocation probabilities, we introduce the concept of a post-choice state $S _ { k } ^ { c } \in S ^ { c }$ that artificially splits the transition between two consecutive pre-decision states into two parts: We first observe the customer’s choice $W _ { k + 1 } ^ { c } = j _ { r _ { k } }$ from the outcome space $\mathcal { W }$ , leading to the post-choice state $S _ { k } ^ { c } = S ^ { M W ^ { c } } ( S _ { k } , X _ { k } , W _ { k + 1 } ^ { c } )$ with transition function $S ^ { M W ^ { c } }$ . Subsequently, the remaining exogenous information $\boldsymbol { W _ { k + 1 } ^ { s } } = ( \mathcal { C } _ { k + 1 } , r _ { k + 1 } , t _ { k + 1 } ) \in \mathcal { W } ^ { s }$ shapes the transition to the next pre-decision state $S _ { k + 1 } = S ^ { M W ^ { s } } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } )$ with transition function $S ^ { M W ^ { s } }$ . Compared to $S _ { k } =$ $\left( { { t } _ { k } } , { { r } _ { k } } , { { O } _ { k } } , { { \mathcal { C } } _ { k } } \right)$ , the post-choice state $S _ { k } ^ { c } = ( t _ { k } , r _ { k } , \mathcal { O } _ { k + 1 } , \mathcal { C } _ { k } , \mathcal { O } _ { k } ^ { A } , A _ { k } )$ already contains the updated set of pending orders $\mathcal { O } _ { k + 1 }$ (Section 2.2.4) and additional information about fulfilled orders $\mathcal { O } _ { k } ^ { A }$ together with the allocated service outcomes encoded by $A _ { k }$ (Section 2.2.3) in case of a prior allocation decision $( \mathcal { O } _ { k } ^ { A } = A _ { k } = \{ \} \mathrm { i f } t _ { k } \notin \mathcal { T } ^ { A } )$ . Essentially, this concept allows us to capture the first and last moment an order exists as a pending order in the system.

Based on this definition, the ‘truthful’ allocation probabilities $\psi _ { o } ^ { \pi } ( S _ { k } ^ { c } )$ represent the likelihood that the premium outcome $i _ { j _ { o } } ^ { + }$ will get allocated to order o with $j _ { o } \in \mathcal { I } ^ { P }$ under policy π when the system currently occupies post-choice state $S _ { k } ^ { c }$ . Keeping in mind that $S _ { k + 1 } = S ^ { M W ^ { s } } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } )$ and $S _ { k + 1 } ^ { c } = S ^ { M W ^ { c } } ( S _ { k + 1 } , X ^ { \pi } ( S _ { k + 1 } ) , W _ { k + 2 } ^ { c } )$ , they are formalized with a recursive equation:

$$
\psi _ { o } ^ { \pi } ( S _ { k } ^ { c } ) = \left\{ \begin{array} { l l } { \displaystyle \sum _ { W _ { k + 1 } ^ { s } \in W ^ { s } } \mathbb { P } ( W _ { k + 1 } ^ { s } | S _ { k } ^ { c } ) \sum _ { W _ { k + 2 } ^ { c } \in \mathcal { W } ^ { c } } \mathbb { P } ( W _ { k + 2 } ^ { c } | S _ { k + 1 } , X ^ { \pi } ( S _ { k + 1 } ) ) \psi _ { o } ^ { \pi } ( S _ { k + 1 } ^ { c } ) } & { \mathrm { i f ~ } o \in \mathcal { O } _ { k + 1 } } \\ { \displaystyle } \\ { \displaystyle \psi _ { o } ^ { \pi } ( S _ { k } ^ { c } ) = \left\{ \begin{array} { l l } { \displaystyle \sum _ { w _ { k + 1 } ^ { s } \in W ^ { s } } \mathbb { P } ( W _ { k + 1 } ^ { c } | S _ { k } ^ { c } ) \sum _ { W _ { k + 2 } ^ { c } \in \mathcal { W } ^ { c } } \mathbb { P } ( W _ { k + 2 } ^ { c } | S _ { k + 1 } , X ^ { \pi } ( S _ { k + 1 } ) ) \psi _ { o } ^ { \pi } ( S _ { k + 1 } ^ { c } ) } & { \mathrm { i f ~ } o \in \mathcal { O } _ { k + 1 } ^ { d } } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle \vphantom { \operatorname* { i d } _ { m _ { k + 1 } ^ { s } \in W ^ { s } } } } & { \mathrm { i f ~ } o \in \mathcal { O } _ { k } ^ { A } } \end{array} \right. } \end{array} \right.\tag{2}
$$

From each individual order’s perspective, we obtain a finite-horizon stochastic process that is embedded within the overarching infinite-horizon sequential decision process. Its initial state equals the post-choice state at which the order first arises as a result of the customer choosing a probabilistic service option, and the terminal states correspond to the order’s potential service outcomes.

The value function $V ( S )$ represents the expected sum of rewards (adjusted by $\bar { R } ^ { \pi ^ { * } } )$ if we start in pre-decision state $S \in S$ and apply the optimal policy $\pi ^ { * }$ from that point onward (Mahadevan 1996). We denote its counterpart for the post-choice states $S \in S ^ { c }$ by $V ^ { c } ( S )$ . To accommodate the artificially split transition to and from the post-choice state, we let $R ^ { c } ( S _ { k } , X _ { k } , W _ { k + 1 } ^ { c } )$ and $R ^ { s } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } )$ denote the reward functions associated with the two respective transition stages, yielding an adapted Bellman equation:

$$
V ( S _ { k } ) = \operatorname* { m a x } _ { X _ { k } \in \mathcal { X } ( S _ { k } ) } \left\{ \sum _ { W _ { k + 1 } ^ { c } \in \mathcal { W } ^ { c } } \mathbb { P } ( W _ { k + 1 } ^ { c } | S _ { k } , X _ { k } ) \bigg ( R ^ { c } ( S _ { k } , X _ { k } , W _ { k + 1 } ^ { c } ) + V ^ { c } ( S _ { k } ^ { c } ) \bigg ) \right\} - \bar { R } ^ { \pi ^ { * } }\tag{3}
$$

$$
V ^ { c } ( S _ { k } ^ { c } ) = \sum _ { W _ { k + 1 } ^ { s } \in \mathcal { W } ^ { s } } \mathbb { P } ( W _ { k + 1 } ^ { s } | S _ { k } ^ { c } ) \left( R ^ { s } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } ) + V ( S _ { k + 1 } ) \right)\tag{4}
$$

The notation allows rewards to arise during either stage of the decomposed transition. If the original reward function $R ( S _ { k } , X _ { k } , W _ { k + 1 } )$ jointly depends on $W _ { k + 1 } ^ { c }$ and $W _ { k + 1 } ^ { s }$ , we can set $R ^ { c } ( S _ { k } , X _ { k } , W _ { k + 1 } ^ { c } ) =$ 0, include $W _ { k + 1 } ^ { c }$ in the post-choice state $S _ { k } ^ { c } .$ , and assign the full reward to the second stage $R ^ { s } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } )$

## 2.4 Transparency Constraints

To limit the extent to which the provider can manipulate displayed probabilities, we introduce transparency constraints. These constraints apply to all pre-decision states in which a request arrives, which we capture with a subset of the pre-decision state space $S ^ { p } = \{ S _ { k } \in { \mathcal { S } } : r _ { k } \neq 0 \}$ We let $P ^ { \pi } ( S _ { k } ) = ( p _ { j } ^ { \pi } ( S _ { k } ) ) _ { j \in \mathcal { I } ^ { P } }$ denote the probabilities displayed under policy π for pre-decision state $S _ { k }$ . Because the truthful probabilities defined in Section 2.3 refer to post-choice states, we introduce $o _ { k j }$ as the hypothetical order that would arise if the request in $S _ { k }$ chooses probabilistic service option $W _ { k + 1 } ^ { c } = j$ , yielding the hypothetical post-choice state $S _ { k j } ^ { c } = S ^ { M W ^ { c } } ( S _ { k } , P ^ { \pi } ( S _ { k } ) , j )$ and the corresponding truthful probability $\psi _ { o _ { k j } } ^ { \pi } ( S _ { k j } ^ { c } )$

The level of intransparency $\tau \in [ 0 , 1 ]$ determines the maximum allowed deviation between the displayed probability $p _ { j } ^ { \pi } ( S _ { k } )$ and the truthful probability $\psi _ { o _ { k j } } ^ { \pi } ( S _ { k j } ^ { c } )$ for the policy π. Accordingly, the transparency constraints for the sequential decision process encapsulated by (1) are formalized as follows:

$$
\left| p _ { j } ^ { \pi } ( S _ { k } ) - \psi _ { o _ { k j } } ^ { \pi } \left( S _ { k j } ^ { c } \right) \right| \leq \tau
$$

$$
j \in \mathcal { I } ^ { P } , S _ { k } \in \mathcal { S } ^ { p }\tag{5}
$$

While $\tau = 0$ represents the case of full transparency without any deviation from the truth, $\tau = 1$ efectively deactivates the constraint. Unlike standard constraints that characterize the decision space (Section 2.2.3), the feasibility of a decision under the transparency constraints not only depends on the current decision epoch, but also on the expected downstream evolution of the sequential decision process as well as anticipated future decisions induced by the policy, which is all captured by $\psi _ { o _ { k j } } ^ { \pi } ( S _ { k j } ^ { c } )$

The level of intransparency τ afects the optimal objective value of the constrained sequential decision process defined by (1) and (5) as follows:

Property 1: Monotonicity in the level of intransparency. The sets of feasible policies are nested in $\tau ;$ that is, $\Pi ( \tau _ { 1 } ) \subseteq \Pi ( \tau _ { 2 } )$ for any $\tau _ { 1 } \leq \tau _ { 2 }$ . Hence, the optimal expected average reward $\bar { R } ^ { \pi ^ { * } } ( \tau )$ is nondecreasing in the level of intransparency τ , pointing to an inherent incentive to distort probabilities.

## 3 Solution Approach

Our sequential decision process is characterized by combinatorial state spaces, a continuous probability decision space, combinatorial allocation decisions, and policy-dependent transparency constraints, hindering the exact computation of the optimal policy $\pi ^ { * }$ with conventional dynamic programming methods. For a fixed policy π and a given post-choice state $S _ { k } ^ { c }$ , we could, in principle, compute the truthful probabilities $\psi _ { o } ^ { \pi } ( S _ { k } ^ { c } )$ by recursively enumerating all state transitions until the considered order o is allocated, essentially generating a finite tree. Its size, however, grows combinatorially with the problem size, making the exact evaluation computationally prohibitive for realistic instances.

Constructing a policy for the problem at hand requires two components, one for probability display and one for allocation, reflecting the two types of decisions. These components are interrelated: Probability displays influence customers’ choices and therefore the composition of future orders to be allocated. Conversely, allocation decisions shape the available capacity and control the eventually assigned service outcome. In combination, both determine the truthful allocation probabilities induced by the policy. Joint optimization therefore requires the simultaneous specification of probability display and allocation decisions while accounting for the truthful probabilities endogenously generated by the overall policy to ensure feasibility with respect to the transparency constraints.

In response, we decompose the problem by decision type. For the allocation component, we restrict our attention to non-anticipatory allocation schemes. We characterize an allocation scheme as non-anticipatory when each allocation decision $A _ { k }$ in states $S _ { k } \in \mathcal { S } : t _ { k } \in \mathcal { T } ^ { A }$ is independent of pending orders $o ^ { \prime } \in \mathcal { O } _ { k } : t _ { o ^ { \prime } } ^ { A } > t _ { k }$ to be allocated at a later point in time and establish the following property (proof in Appendix B):

Property 2: Uniqueness of fully transparent probability displays under non-anticipatory allocation. Consider a fixed, fully specified, and non-anticipatory allocation scheme α that uniquely determines the joint allocation outcome for all orders $o \in { \mathcal { O } } _ { k } ^ { A }$ in every pre-decision state $S _ { k } \in \mathcal { S } : t _ { k } \in \mathcal { T } ^ { A }$ independent of pending orders $o ^ { \prime } \in \mathcal { O } _ { k } : t _ { o ^ { \prime } } ^ { A } > t _ { k }$ . Under full transparency $( \tau = 0 )$ , α admits exactly one feasible and hence optimal solution for the displayed probabilities, invariant to the underlying reward function.

Two consequences follow from Property 2: First, the joint policy optimization problem for $\tau = 0$ within the class of non-anticipatory allocation schemes reduces to selecting an allocation scheme α. Second, a non-anticipatory allocation scheme implies the existence of a feasible policy for $\tau = 0$ and by the nesting of $\Pi ( 0 ) \subseteq \Pi ( \tau )$ (Property 1), the feasible set $\Pi ( \tau )$ is nonempty for every $\tau = [ 0 , 1 ]$

In particular, we employ priority rules based on order characteristics (Section 4) and hold the resulting non-anticipatory allocation schemes fixed across transparency levels τ . This enables us to isolate the efect of manipulating probabilistic information as a demand-steering mechanism because any observed diferences in metrics across transparency levels can be fully attributed to the probabilities displayed.

As a solution strategy, we develop an approximation-based approach. Specifically, instead of computing $\psi _ { o } ^ { \pi } ( S _ { k } ^ { c } )$ for each individual order o and post-choice state $S _ { k } ^ { c }$ , we train a neural network to predict approximated truthful probabilities $\hat { \psi } _ { o } ( S _ { k } ^ { c } )$ and propose a corresponding learning algorithm (Section 3.1). To eficiently search the continuous probability display decision space, we formulate a scenario-based mixed-integer program (MIP, Section 3.2).

## 3.1 Approximation of Truthful Probabilities

We first outline the architecture of the neural network in Section 3.1.1. Following a detailed description of our learning algorithm in Section 3.1.2, we highlight important algorithmic design choices in Section 3.1.3. Lastly, we explain how to evaluate the approximation quality in Section 3.1.4.

## 3.1.1 Approximation Architecture

Our neural network $\hat { \psi } _ { \pmb { \theta } }$ is parameterized by weights θ that are learned ofline in a simulation. The network takes a vector of features $\phi ( S _ { k } ^ { c } , o )$ as an input that encodes information about the postchoice state $S _ { k } ^ { c }$ and the (hypothetical or pending) order o for which we aim to predict the truthful probability $\hat { \psi } _ { o } ( S _ { k } ^ { c } )$ . We normalize all features to the interval [−1, 1] to mitigate diferences in scale and prevent features with larger numerical magnitudes from dominating the network’s weight updates. The output layer consists of a single neuron with an activation function whose range is bounded by [0, 1], ensuring that the network output can be interpreted as a probability. For a newly arrived request, we conduct one forward pass $\hat { \psi } _ { \pmb { \theta } } \big ( \phi ( S _ { k j } ^ { c } , o _ { k j } ) \big )$ through the neural network per probabilistic service option $j \in \mathcal { I } ^ { P }$ , each with the respective hypothetical order.

We explain the concrete features and detailed network architecture used for our computational study in Appendix C. While the design of the features $\phi ( S _ { k j } ^ { c } , o _ { k j } )$ is inherently application-specific, we provide general conceptual guidance: Natural choices include the amount of time remaining until the order’s allocation, the number of pending orders for probabilistic service option j, the presently occupied capacity of the associated premium outcome $i _ { j } ^ { + }$ along with elapsed usage durations, and attributes of the request and the probabilistic service option under consideration. We also include the expected demand share of the premium outcome $i _ { j } ^ { + }$ as a measure of its popularity.

Beyond this readily available information, we construct more sophisticated features with insights gained from a lookahead approach, inspired by the corresponding policy class in the framework by

Powell (2022). Specifically, we simulate several sample paths starting from $S _ { k j } ^ { c }$ under a fixed, stateindependent base policy to obtain rough estimates of the capacity expected to be available upon allocation as well as the truthful probability. Because the base policy remains fixed over time and its decisions do not depend on specific pre-decision states, most computations for the lookahead procedure can be performed in advance and reused eficiently. The neural network then learns to correct the biased baseline estimate towards the actual policy.

## 3.1.2 Learning Algorithm

At first glance, approximating the truthful probabilities may resemble a standard supervised learning task. However, these probabilities depend on the policy, while the policy itself is in turn linked to the probabilities through the transparency constraints, resulting in a feedback loop. To address this endogeneity, we develop an iterative, simulation-based learning procedure inspired by reinforcement learning (Sutton and Barto 2018) that is conducted ofline. Like value function approximation, our method replaces a quantity that depends on the downstream evolution of the sequential decision process, namely $\psi _ { o } ^ { \pi } ( S _ { k + 1 } ^ { c } )$ in (2), with a function that generalizes across states. The key diference is that we approximate the conditional distribution of a future outcome given the current postchoice state and the policy, yielding an approximation strategy that targets the sequential decision problem’s outcome space instead of the value or decision space.

The training procedure is summarized in Algorithm 1. Whenever a request arrives in a simulated decision epoch n, we make a probability display decision. More precisely, a weighted coin flip determines whether we exploit our current knowledge encapsulated by $\hat { \psi } _ { \pmb { \theta } }$ and solve the scenariobased MIP described in Section 3.2 or explore by randomly sampling a decision from the decision space. The probability of exploring is high at the beginning and exponentially decreases over the course of training. This exploration mitigates a self-reinforcing sampling bias that may arise from the interaction between the probability approximation and customers’ choices: If the network underestimates the truthful probability for a probabilistic service option, the resulting policy displays a correspondingly low probability. Customers may then select this option only rarely, generating few orders and, consequently, few allocation outcomes from which the underestimation could be corrected. Exploratory display decisions increase the likelihood of obtaining observations for otherwise underrepresented probabilistic options and outcomes.

During the simulation, we collect experiences in a replay memory $\mathcal { E } ,$ which stores the training data for updating the network weights θ. After a probability display decision, we generate new experiences e in the form of tuples $( o _ { e } , \phi _ { e } , a _ { e } ) = ( o , \phi ( S _ { n } ^ { c } , o ) , \cdot )$ from each pending order $o \in \mathcal { O } _ { n }$ based on the order’s feature vector in the post-choice state of the current decision epoch $\phi _ { e } = \phi ( S _ { n } ^ { c } , o )$ Once an order o is allocated, we update all experiences $e \in \mathcal { E } : o _ { e } = o$ associated with that order by including the realized allocation outcome $a _ { e } = a _ { o } ^ { n }$ . We deliberately create experiences for all pending orders, rather than only for the order newly confirmed in the current decision epoch, because hypothetical and actual pending orders provide equivalent training observations for the network.

Given that the observed allocation outcomes are binary and we want to predict the underlying probability, our learning problem closely resembles a binary classification task. Accordingly, we train the network by minimizing the binary cross-entropy loss. To this end, we sample a mini-batch B of experiences from the replay memory for which the allocated service outcome $a _ { e }$ has already been observed and update the network parameters by performing a gradient-based optimization step on the loss function:

$$
\mathcal { L } ( \pmb { \theta } , \pmb { \mathcal { B } } ) = - \frac { 1 } { | \mathscr { B } | } \sum _ { e \in \mathcal { B } } \Big ( a _ { e } \log \hat { \psi } _ { \pmb { \theta } } ( \phi _ { e } ) + ( 1 - a _ { e } ) \log ( 1 - \hat { \psi } _ { \pmb { \theta } } ( \phi _ { e } ) ) \Big )\tag{6}
$$

Algorithm 1 Ofline training of network weights θ   
Initialization: Network $\hat { \psi } _ { \boldsymbol { \theta } }$ with random weights θ, replay memory ${ \mathcal { E } } = \{ \}$ , initial pre-decision state $S _ { 1 }$   
$e = 0$   
1: for $n = 1 , \ldots , N$ do   
2: if $r _ { n } \neq 0$ then   
3: if exploration then   
4: uniformly sample $P _ { n }$ from $\textstyle { \mathcal { X } } ( S _ { n } )$   
5: else   
6: for probabilistic product $j \in \mathcal { I } ^ { P }$ do   
7: compute features $\phi ( S _ { n j } ^ { c } , o _ { n j } )$ for hypothetical order $o _ { n j }$   
8: approximate truthful probabilities $\hat { \psi } _ { j } ^ { n } = \hat { \psi } _ { \pmb { \theta } } \big ( \phi ( S _ { n j } ^ { c } , o _ { n j } ) \big )$   
9: end for   
10: make probability display decision $P _ { n }$ with MIP (Section 3.2) based on $\hat { \psi } _ { j } ^ { n }$ for $j \in \mathcal { I } ^ { P }$   
11: end if   
12: for pending order $o \in \mathcal { O } _ { n }$ do   
13: e ← e + 1   
14: generate experience $( o _ { e } , \phi _ { e } , a _ { e } ) = ( o , \phi ( S _ { n } ^ { c } , o ) , \cdot )$ and store it in replay memory E   
15: end for   
16: else   
17: make allocation decision   
18: for allocated order $o \in \mathcal { O } _ { n } ^ { A }$ do   
19: update relevant experiences $e \in \mathcal { E } : o _ { e } = o$ by setting $a _ { e }$ to allocation outcome $a _ { o } ^ { n }$   
20: end for   
21: end if   
22: update network weights θ   
23: sample exogenous information $W _ { n + 1 }$ and transition to $S _ { n + 1 }$   
24: end for

## 3.1.3 Design Choices

We highlight two distinctive design choices for our learning algorithm that proved to be crucial for performance:

1. No temporal diference learning. From a reinforcement learning perspective, the embedded finite stochastic process associated with an order constitutes an approximation task characterized by highly sparse, delayed, and binary ‘reward’ signals: For each order, the allocated service outcome is observed only once as a binary signal at the time of allocation, with no intermediate feedback from the simulation environment. As a result, we found that temporal diference learning methods, which exploit the recursive structure in (2) and update the network parameters using the predicted truthful probability in the subsequent state, lead to unsatisfactory performance. Instead, we employ a Monte Carlo-style update (Sutton and Barto 2018, Chapter 5) built on the observed allocation decision to compute the loss (6). Although basing the update on complete trajectories increases the variance, it ensures that each update is grounded in an observed outcome.

2. Sampling of experiences. Instead of drawing the mini-batch B uniformly and independently, we first sample a set of previously encountered post-choice states represented in the replay memory. For each sampled state, we include all associated orders with observed allocation outcomes in the update, grouping them into several batches if their number exceeds |B|. Incorporating pending orders associated with the same state jointly in the update is essential, as it allows the network to learn to diferentiate between orders in the same state that ultimately experience diferent allocation decisions.

## 3.1.4 Evaluating Approximation Quality

By design, any reinforcement learning algorithm is unlikely to achieve predictive accuracy across the entire state space and for all hypothetical orders in our setting. In particular, an identification problem arises for combinations of requests and service options that are never observed in the ofline simulation. For example, if some customers consistently prefer a deterministic option to certain probabilistic service options, the simulation never generates allocation decisions for this combination of customer and service option, leading to a lack of training data.

At the same time, prediction errors for such combinations have limited managerial relevance because customers likewise cannot evaluate the accuracy of probabilities displayed for service options they do not choose. Instead, long-term credibility depends both on the approximation quality for chosen options and the extent to which the provider manipulates disclosed information. Accordingly, the central question is whether the displayed probabilities appear trustworthy to customers over repeated interactions, given the allocation outcomes they actually observe, instead of counterfactual accuracy for service options not selected.

To assess the quality of the learned approximation, we generate a test dataset and restrict attention to the chosen service options. Because the exact truthful probabilities are unavailable for comparison, we estimate them via Monte Carlo simulation for post-choice states randomly selected during an independent simulation. Specifically, for each sampled post-choice state $S _ { k } ^ { c }$ we simulate F trajectories under the given policy, each starting from $S _ { k } ^ { c }$ and continuing until the allocation decisions for all relevant pending orders with $j _ { o } \in \mathcal { I } ^ { P }$ have been observed. For each trajectory $f = 1 , \ldots , F$ , we encode by $a _ { o f } \in \{ 0 , 1 \}$ whether pending order o received the premium outcome $i _ { j _ { o } } ^ { + }$ of the chosen probabilistic service option $j _ { o } \in \mathcal { I } ^ { P }$ . The Monte Carlo estimate of the truthful probability is then given by $\begin{array} { r } { \hat { \psi } _ { o } ^ { M C } ( S _ { k } ^ { c } ) = \frac { 1 } { F } \sum _ { f = 1 } ^ { F } a _ { o f } } \end{array}$ . We compare these estimates with the corresponding neural network predictions $\hat { \psi } _ { o } ( S _ { k } ^ { c } )$ and assess approximation quality using the mean absolute error across the evaluated state-order combinations. Note that because the Monte Carlo estimates are based on a finite number of trajectories, they are themselves subject to sampling error.

Importantly, this procedure allows us to evaluate probability predictions for pending orders at arbitrary stages, rather than only immediately after order confirmation. If desired, the evaluation can be restricted to newly confirmed orders by filtering the test data accordingly.

## 3.2 Probability Display

To decide which probabilities to display in a pre-decision state $S _ { k }$ , the provider must essentially evaluate the maximization operator in (3) subject to the transparency constraints (5). Similar to the truthful probabilities, the exact value $V ^ { c } ( S _ { k } ^ { c } )$ of a post-choice state is generally intractable to compute for realistic instances, so we likewise replace it with an approximation $\hat { V } ^ { c } ( S _ { k } ^ { c } )$ . To avoid nonlinearity, the approximations should only depend on the option selected by the customer and not on the specific probabilities $P _ { k }$ displayed prior to that choice. Reusing the notation introduced in Section 2.2.4, we write $\hat { V } ^ { c } ( S _ { k j } ^ { c } )$ for the approximate value of the hypothetical post-choice state resulting from the current request choosing service option $j \in \mathcal I$

Our methodology is compatible with any of the commonly applied value approximation strategies, spanning both online methods, such as a truncated rollout or lookahead model, and ofline, learning-based methods; we refer to Powell (2022) and Sutton and Barto (2018) for comprehensive introductions. Accordingly, the specific approach to obtain $\hat { V } ^ { c } ( S _ { k j } ^ { c } )$ is not the focus of this work. We describe the particular methods used for our computational study in Section 4.

In addition to the value approximation $\hat { V } ^ { c } ( S _ { k j } ^ { c } )$ , solving the maximization problem requires a tractable closed-form representation of the choice probabilities $\mathbb { P } ( W _ { k + 1 } ^ { c } | S _ { k } , X _ { k } )$ that captures their dependence on the continuous probability decision variables $P _ { k }$ (encapsulated by $X _ { k } )$ . Because such a representation is not generally available, we develop a scenario-based approach inspired by Paneque et al. (2021). Specifically, we generate scenarios $\omega \in \Omega$ by sampling the service outcome utilities $\tilde { u } _ { i \omega }$ for the current customer request from their known distributions. For deterministic products $j \in \mathcal { I } ^ { D }$ , this immediately yields the valuation $\tilde { U } _ { j \omega } = \tilde { u } _ { i _ { j } \omega }$ as a constant parameter since the base and premium outcome coincide. For probabilistic service options $j \in \mathcal { I } ^ { P }$ , the probability display decision $P _ { k }$ and the scenario $\omega \in \Omega$ jointly determine the valuation, resulting in a continuous auxiliary variable $\tilde { U } _ { j \omega }$ . The customer selects the option with the highest valuation. We capture the maximum valuation in scenario ω with the continuous variable $\widetilde { U } _ { \omega }$ and represent the customer’s choice with the binary decision variable $c _ { j \omega }$ , which equals one if the customer selects service option $j$ in scenario $\omega$ and zero otherwise. The corresponding choice probability is approximated by the sample frequency $\begin{array} { r } { \hat { \mathbb P } ( W _ { k + 1 } ^ { c } = j \mid S _ { k } , X _ { k } ) = \frac { 1 } { | \Omega | } \sum _ { \omega \in \Omega } c _ { j \omega } } \end{array}$

Piecing everything together, we obtain the following mixed integer program:

$$
\operatorname* { m a x } \sum _ { j \in \mathcal { I } } \frac { 1 } { | \Omega | } \sum _ { \omega \in \Omega } c _ { j \omega } \bigg ( R ( S _ { k } , P _ { k } , j ) + \hat { V } ^ { c } ( S _ { k j } ^ { c } ) \bigg )\tag{7}
$$

$$
\tilde { U } _ { j \omega } = U \left( p _ { j } ^ { k } , \tilde { u } _ { i _ { j } ^ { - } \omega } , \tilde { u } _ { i _ { j } ^ { + } \omega } \right)
$$

$$
j \in \mathcal { I } ^ { P } , \omega \in \Omega\tag{8}
$$

$$
\tilde { U } _ { \omega } \geq \tilde { U } _ { j \omega }
$$

$$
j \in \mathcal { I } , \omega \in \Omega\tag{9}
$$

$$
\tilde { U } _ { \omega } \leq \tilde { U } _ { j \omega } + M ( 1 - c _ { j \omega } )
$$

$$
j \in \mathcal { I } , \omega \in \Omega\tag{10}
$$

$$
\sum _ { j \in \mathcal { I } } c _ { j \omega } = 1
$$

$$
\omega \in \Omega\tag{11}
$$

$$
p _ { j } ^ { \pi } ( S _ { k } ) - \hat { \psi } _ { j } ^ { k } \le \tau
$$

$$
j \in \mathcal { I } ^ { P }\tag{12}
$$

$$
\hat { \psi } _ { j } ^ { k } - p _ { j } ^ { \pi } ( S _ { k } ) \le \tau
$$

$$
j \in \mathcal { I } ^ { P }\tag{13}
$$

$$
p _ { j } ^ { k } \in [ 0 , 1 ]
$$

$$
j \in \mathcal { I } ^ { P }\tag{14}
$$

$$
c _ { j \omega } \in \{ 0 , 1 \}
$$

$$
j \in \mathcal { I } , \omega \in \Omega\tag{15}
$$

$$
\tilde { U } _ { j \omega } , \tilde { U } _ { \omega } \in \mathbb { R }
$$

$$
j \in \mathcal { I } ^ { P } , \omega \in \Omega\tag{16}
$$

Objective (7) approximates the maximum operator in (3). Constraints (8) determine the scenario- and probability-dependent valuations of probabilistic service options. Constraints (9) and (10), together with a suficiently large constant M, identify the customer’s maximum valuation per scenario and ensure that the associated option is selected. Constraints (11) require the customer to choose one option in each scenario. Finally, (12) and (13) limit the deviation between displayed and approximated truthful probabilities with $\hat { \psi } _ { j } ^ { k } = \hat { \psi } _ { \pmb { \theta } } ( \phi ( S _ { k j } ^ { c } , o _ { k j } ) )$ . Note that if the immediate reward $R ( S _ { k } , P _ { k } , j )$ directly depends on $P _ { k }$ or function U models nonlinear risk preferences, linearization techniques may be necessary.

There are two sources of multiple optimal solutions in our model, which we address as follows:

• Indiference. If multiple service options achieve the maximum valuation in a scenario, the customer is indiferent among them. The optimization automatically resolves such ties in favor of the provider, which not necessarily aligns with actual customer behavior. A relevant structural tie arises if the base outcome of a probabilistic service option $j \in \mathcal { I } ^ { P }$ is also ofered as a deterministic option $j ^ { \prime } \in \mathcal { I } ^ { D }$ with $i _ { j } ^ { - } = i _ { j ^ { \prime } } . \mathrm { I f } p _ { j } ^ { k } = 0$ , the product valuations are identical since both products are expected to yield the same outcome from the customer’s perspective. We denote the subset of probabilistic service options with such a deterministic counterpart by $\mathcal { I } ^ { P ^ { \prime } }$ . Assuming that the customer only selects a probabilistic service option $j \in \mathcal { I } ^ { \mathcal { P } ^ { \prime } }$ if $\begin{array} { r } { p _ { j } ^ { k } \ge \frac { 1 } { M ^ { \prime } } } \end{array}$ , we can formulate an additional tie-breaking constraint with a suficiently large constant $M ^ { \prime }$

$$
c _ { j \omega } \leq M ^ { \prime } \cdot p _ { j } ^ { k }
$$

$$
j \in \mathcal { I } ^ { P ^ { \prime } } , \omega \in \Omega\tag{17}
$$

• Intransparency. For $\tau > 0 .$ , diferent probability displays may consistently provoke the same customer choices in all scenarios and hence yield an identical objective value. To ensure that the provider distorts probabilities only as much as necessary, we penalize deviations from the approximated probabilities with a small positive constant in the objective.

## 4 Computational Experiments

Our computational study investigates how, in the context of out-of-home delivery (OOHD), displayed probabilities can serve as an information-based demand management tool and how (in)transparency afects the involved stakeholders, namely the provider and customers.

## 4.1 Design of Experiments

In this section, we apply our general modeling framework to OOHD (Section 4.1.1) and specify the instance settings (Section 4.1.2), investigated policies (Section 4.1.3), and evaluation metrics (Section 4.1.4).

## 4.1.1 The Case of Out-of-Home Delivery

OOHD providers commonly operate parcel lockers and parcel shops simultaneously within their networks since both facility types ofer distinct advantages and disadvantages (Janinhof et al. 2024): While the opening hours of shops are restricted, fully automated lockers enable $2 4 / 7$ access for customers. In contrast, capacity constraints at shops are negligible in practice, whereas locker capacity is strictly limited by the fixed number of compartments. Additionally, the availability of locker capacity is uncertain because it depends on customers’ pickup behavior, rendering lockers a natural candidate for probabilistic service options. More precisely, we model parcel shops as base outcomes $i \in \mathcal { Z }$ and lockers as premium outcomes $i \in \mathcal { I } ^ { + }$ . Each probabilistic service option $j \in \mathcal { I } ^ { P }$ with $\vert \mathcal { I } ^ { P } \vert = \vert \mathcal { I } ^ { + } \vert$ pairs a locker with its closest parcel shop, to which the parcel gets redirected if the locker’s capacity is insuficient. Customers may also directly choose a parcel shop, included as deterministic service options $j \in \mathcal { I } ^ { D }$ with $| \mathcal { I } ^ { D } | = | \mathcal { I } ^ { - } |$

We divide the planning horizon into days and map each point in time $t _ { k }$ to a tuple $( t _ { k } ^ { d } , t _ { k } ^ { w } )$ where $t _ { k } ^ { d } = 1 , \ldots , \infty$ denotes the current day and $t _ { k } ^ { w } = 1 , \dots , T + 1$ the point in time within a day. Requests may arrive at $t _ { k } ^ { w } = 1 , \dots , T$ and are characterized by the customers’ locations. At the end of each day $( t _ { k } ^ { d } , t _ { k } ^ { w } = T + 1 )$ , the provider makes allocation decisions for all pending orders that arrived throughout day $t _ { k } ^ { d } .$

Following random utility theory (Train 2009), we decompose the utility $u _ { i } ^ { k }$ that a request $r _ { k }$ associates with facility $i \in \mathcal { Z }$ into a deterministic and a stochastic component. The deterministic component $v _ { i } ^ { k } = \beta _ { L } \cdot \mathbf { 1 } ( i \in \mathcal { I } ^ { + } ) - \beta _ { D } \delta _ { r _ { k } i }$ includes an additive constant $\beta _ { L } > 0$ for parcel lockers, reflecting their benefit of unrestricted opening hours, and linearly depends on the Euclidean distance $\delta _ { r _ { k } i }$ between the customer’s location and facility i with coeficient $\beta _ { D } > 0$ . The stochastic component $\varepsilon _ { i } ^ { k }$ follows a known distribution and captures individual preferences as well as other unobserved factors. In total, we obtain $u _ { i } ^ { k } = v _ { i } ^ { k } + \varepsilon _ { i } ^ { k }$

The provider’s objective is to maximize aggregate customer satisfaction, measured by the sum of customers’ utilities for the allocated facilities. Although the provider does not observe utilities $u _ { i } ^ { k }$ directly, the displayed probability vector $P _ { k }$ and the chosen service option $j _ { r _ { k } }$ provide information about them. Once request $r _ { k }$ becomes an order $^ { O , }$ we store the conditional expected utilities $\mathbb { E } [ u _ { i } ^ { k } | P _ { k } , j _ { r _ { k } } ]$ for service outcomes $i \in \{ i _ { j _ { r _ { k } } } ^ { - } , i _ { j _ { r _ { k } } } ^ { + } \}$ as order attributes $\hat { u } _ { o } ^ { - } , \hat { u } _ { o } ^ { + }$ in the state variable. For allocation decisions, the reward function thus equals $\begin{array} { r } { R ( S _ { k } , A _ { k } , W _ { k + 1 } ) = \sum _ { o \in \mathcal { O } _ { k } ^ { A } } ( a _ { o } ^ { k } \hat { u } _ { o } ^ { + } + ( 1 - a _ { o } ^ { k } ) \hat { u } _ { o } ^ { - } ) } \end{array}$ with $R ( S _ { k } , P _ { k } , W _ { k + 1 } ) = 0$ in case of probability display.

## 4.1.2 Settings

We begin by specifying the parameters held constant across all settings:

• Infrastructure. Following Akkerman et al. (2025), we use the RC benchmark instance with 600 customer locations by Gehring and Homberger (2002) and randomly designate a settingdependent number of locations as OOHD facilities. Of these facilities, 80% are parcel lockers with $Q _ { i } = 1 0$ compartments each, and 20% are parcel shops. The remaining locations represent the customer pool. We rescale the coordinates to an area of $1 0 \times 1 0$ km.

• Requests. At each within-day point in time $t _ { k } ^ { w } \in \{ 1 , \dots , T \}$ , a request arrives with probability 0.9, and its location is sampled uniformly from the customer pool.

• Utilities. We set the distance-related utility coeficient to $\beta _ { D } = 4$ . In line with the multinomial logit model (Train 2009), the stochastic utility components $\varepsilon _ { i } ^ { k }$ follow independent standard Gumbel distributions. We assume risk-neutral customers such that the valuations $U _ { j } ^ { k }$ for probabilistic service options $j \in \mathcal { I } ^ { P }$ equal the expected utility $U _ { j } ^ { k } = p _ { j } ^ { k } u _ { i _ { j } ^ { + } } ^ { k } + ( 1 - p _ { j } ^ { k } ) u _ { i _ { j } ^ { - } } ^ { k }$ Because the conditional expectations $\mathbb { E } [ u _ { i } ^ { k } | P ^ { k } , j _ { r _ { k } } ]$ cannot be computed in closed form, we estimate them through accept-reject sampling (Train 2009) using 100 accepted draws.

• Pickup behavior. Customers collect their parcels on the first, second, or third day after allocation with probabilities of 60%, 20%, and 20%, respectively. The pickup time within each day is distributed uniformly.

Starting from a base setting with the additive locker utility constant set to $\beta _ { L } = 2$ and $I = 1 0$ facilities, we vary the experimental setup along two dimensions:

1. We examine diferent levels of systematic utility diference between lockers and shops $\beta _ { L } \in$ {0, 4}.

2. To assess the scalability of our approach, we increase the number of OOHD facilities to $I \in \{ 1 5 , 2 0 , 2 5 \}$ while maintaining a locker share of 80%. The resulting facility networks are nested, i.e., each larger network contains all facilities of the smaller networks. Throughout our study, we scale demand proportionally with network size by setting the number of within-day periods to $T = 6 I$

## 4.1.3 Policies

We test two non-anticipatory allocation schemes. In both cases, we rank the orders $o \in \mathcal { O } _ { k } ^ { A } : i _ { j _ { o } } ^ { + } = i$ for each parcel locker $i \in \mathcal { T } ^ { + }$ according to a specific priority criterion and allocate them to the premium outcome (parcel locker) as long as suficient capacity is available. Remaining orders receive the base outcome (parcel shop). The priority criterion for the respective allocation scheme must be stored as an additional order attribute in the state variables.

1. FF (first come, first served): As a benchmark, we process the orders according to their arrival time.

2. MY (myopic): Following a regret-based rationale, we rank the orders in descending order of the diference between the deterministic utility components $v _ { i } ^ { k }$ for locker and shop. Essentially, this prioritizes customers who incur the largest deterministic utility loss if redirected to the parcel shop. We deliberately refrain form ranking orders based on their conditional expected utilities $\hat { u } _ { o } ^ { + }$ and $\hat { u } _ { o } ^ { - }$ because these quantities depend on the specific probabilities displayed upon request arrival, which would create a feedback loop between probability display and allocation decisions.

For probability display, we consider two variants of our approach that difer in the technique to obtain the approximated post-choice state values $\hat { V } ^ { c } ( S _ { k } ^ { c } )$ and compare them to two benchmarks:

1. EU (expected utility): Focusing entirely on the current request, we approximate $\hat { V } ^ { c } ( S _ { k } ^ { c } )$ by the hypothetical order’s expected future reward. Based on the estimated truthful probabilities $\hat { \psi } _ { j } ^ { k }$ and the sampled utilities $\tilde { u } _ { i \omega }$ , we compute the scenario-dependent approximation $\hat { V } ^ { c } ( S _ { k j } ^ { c } , \omega ) =$ $\hat { \psi } _ { j } ^ { k } \tilde { u } _ { i _ { j } ^ { + } \omega } + ( 1 - \hat { \psi } _ { j } ^ { k } ) \tilde { u } _ { i _ { j } ^ { - } \omega } .$

2. LA (lookahead model): To account for the potential displacement of future customers, we generate scenarios for the future evolution of the system, solve the resulting finite-horizon full information problems, and derive $\hat { V } ^ { c } ( S _ { k j } ^ { c } )$ based on the objective values (details in Appendix C).

3. NO (no information): To gauge the value of information provision, we include a benchmark where the provider does not communicate probabilities to customers, efectively ofering flexible products instead (Section 1.2.2). We assume that customers disregard allocation uncertainty in their valuations and select the service option $j \in \mathcal { I }$ whose premium outcome yields the highest utility $u _ { i _ { j } ^ { + } } ^ { k }$

4. GP (guaranteed premium): As an alternative to probabilistic service options, the provider may decide to ofer a premium outcome only if it can be guaranteed, entirely eliminating the case of parcels being redirected to shops. This requires checking whether a feasible allocation that includes all pending orders exists assuming that customers collect their parcels as late as possible. The displayed probability equals 1 if allocating the request to the premium outcome is feasible under this worst-case assumption and 0 otherwise.

Combining the approaches listed above yields $2 \cdot 4 = 8$ policies. We denote each policy by $\pi _ { \mathrm { : } }$ with probability display indicated in the superscript and allocation in the subscript, $\mathrm { e . g . , \pi _ { F F } ^ { E U } }$ . For policies compatible with transparency constraints (EU and $\mathrm { L A } )$ , we include the level of intransparency, $\mathrm { e . g . }$ $\pi _ { \mathrm { F F } } ^ { \mathrm { E U } } ( \tau )$ . Across all settings, we consider both full transparency $( \tau = 0 )$ , where the provider merely displays probabilities to inform customers, and completely deactivated transparency constraints $( \tau = 1 )$ , allowing the provider to freely leverage the probabilities as a demand management tool. In the setting with $\beta _ { L } = 4$ , we conduct a more detailed analysis by varying $\tau \in \{ 0 , 0 . 0 5 , 0 . 1 , \dots , 1 \}$

We implemented all algorithms in Python 3.13 and solved the MIPs for probability display decisions to optimality with Gurobi 13 if $\tau > 0$ . Under full transparency $\tau = 0$ , the unique feasible solution to the MIP is directly given by $p _ { j } ^ { k } = \hat { \psi } _ { j } ^ { k }$ . For EU and LA, we train the approximation $\hat { \psi } _ { \pmb { \theta } }$ for every setting and value of $\tau _ { \mathrm { { ; } } }$ including $\tau = 1$ . Although the transparency constraints are inactive for $\tau = 1$ , we still require $\hat { \psi } _ { \pmb { \theta } }$ to minimize deviations from the approximated truthful probabilities $\hat { \psi } _ { j } ^ { k }$ as a secondary objective (Section 3.2). Appendix C provides details on the features $\phi ( S _ { k } ^ { c } , o )$ and hyperparameters. The experiments were conducted on a server equipped with two Intel Xeon E7-8890 v3 processors (2.5 GHz, 18 cores) and 512 GB of RAM. The average computation time for probability display and allocation decisions remained below 0.5 seconds for all considered policies and settings, indicating suitability for real-time deployment.

## 4.1.4 Metrics

We use eight metrics to evaluate solution quality and derive managerial insights:

1. CS (customer satisfaction): As our primary performance metric, we report customer satisfaction, measured by the conditional expected utilities for allocated service outcomes.

2. LS (locker share): The share of customers that select a probabilistic service option.

3. RR (redirection rate): The percentage of customers receiving a parcel shop base outcome out of those that chose a probabilistic service option.

4. LU (locker utilization): The percentage of occupied locker capacity directly after allocation.

5. DC (displayed probability for chosen option): The probability displayed for the premium outcome if a customer selects a probabilistic service option.

6. IN (level of intransparency): The maximum absolute deviation between the displayed probabilities $p _ { j } ^ { \pi } ( S _ { k } )$ and the approximated truthful probabilities $\hat { \psi } _ { o } ( S _ { k } ^ { c } )$ per customer.

7. AE (approximation error): The mean absolute error between our neural network’s predictions $\hat { \psi } _ { o } ( S _ { k } ^ { c } )$ and the Monte Carlo estimates $\hat { \psi } _ { o } ^ { M C } ( S _ { k } ^ { c } )$

Table 1. Performance metrics for the base setting.
<table><tr><td rowspan="2" colspan="2">Policy π</td><td rowspan="2"></td><td colspan="3">CS</td><td colspan="5">(%)</td><td colspan="2">(pp)</td></tr><tr><td>Total</td><td>∆%</td><td>pc</td><td>LS</td><td>RR</td><td>LU</td><td>LU</td><td>DC</td><td>IN</td><td>AE</td></tr><tr><td>FF</td><td>NO</td><td></td><td>-19,996.5</td><td>0.0</td><td>-4.9</td><td>86.6</td><td>13.4</td><td>59.8</td><td>96.6</td><td></td><td>一</td><td>一</td></tr><tr><td></td><td>GP</td><td></td><td>-19,950.2</td><td>0.2</td><td>-4.9</td><td>78.9</td><td>0.0</td><td>75.5</td><td>90.4</td><td></td><td></td><td>一</td></tr><tr><td></td><td>EU/LA</td><td>0</td><td>-17,439.2</td><td>12.8</td><td>-4.3</td><td>85.6</td><td>5.5</td><td>69.1</td><td>96.7</td><td>94.9</td><td>0.0</td><td>0.8</td></tr><tr><td></td><td>EU</td><td>1</td><td>-17,408.6</td><td>12.9</td><td>-4.3</td><td>85.5</td><td>5.2</td><td>69.7</td><td>96.6</td><td>94.7</td><td>1.1</td><td>0.6</td></tr><tr><td></td><td>LA</td><td>1</td><td>-16,551.2</td><td>17.2</td><td>-4.1</td><td>82.3</td><td>2.7</td><td>76.3</td><td>92.0</td><td>91.4</td><td>19.9</td><td>0.4</td></tr><tr><td>MY</td><td>NO</td><td></td><td>-18,108.0</td><td>0.0</td><td>-4.5</td><td>86.6</td><td>13.4</td><td>59.8</td><td>96.4</td><td></td><td>一</td><td>一</td></tr><tr><td></td><td>GP</td><td></td><td>-19,950.2</td><td>-10.2</td><td>-4.9</td><td>78.9</td><td>0.0</td><td>75.5</td><td>90.4</td><td></td><td></td><td></td></tr><tr><td></td><td>EU/LA</td><td>0</td><td>-16,741.6</td><td>7.5</td><td>-4.1</td><td>86.5</td><td>8.6</td><td>74.4</td><td>95.8</td><td>90.0</td><td>0.0</td><td>3.4</td></tr><tr><td></td><td>EU</td><td>1</td><td>-16,699.1</td><td>7.8</td><td>-4.1</td><td>86.4</td><td>8.5</td><td>75.1</td><td>95.7</td><td>90.2</td><td>1.8</td><td>3.5</td></tr><tr><td></td><td>LA</td><td>1</td><td>-15,917.5</td><td>12.1</td><td>-3.9</td><td>84.7</td><td>5.5</td><td>72.4</td><td>93.3</td><td>89.6</td><td>11.1</td><td>2.3</td></tr></table>

8. CR (customer regret): The utility diference between the outcome allocated under full transparency (τ = 0) and deactivated transparency constraints (τ = 1) per customer.

Throughout our study, we report probabilities and the metrics LS, RR, LU, and DC as percentages (%), whereas IN and AE are measured in percentage points (pp). In each setting, we estimate the metrics with 30 randomly generated instances. For the sake of comparability, we consistently use the same 30 sequences of customer arrivals (including location, stochastic utility components, and pickup behavior). The policies start with empty facilities and no pending orders and complete a warm-up phase of 25 days that are discarded to eliminate temporary efects of the initial state. We calculate the metrics per instance (consisting of 75 days after the warm-up) and compute the average per setting. To evaluate approximation quality, we randomly select 10 decision epochs from each instance and obtain the Monte Carlo estimates $\hat { \psi } _ { o } ^ { M C }$ for all pending orders based on 100 sampled paths (Section 3.1.4).

## 4.2 Base Setting

We first report aggregate results (Section 4.2.1), followed by a detailed analysis of the distortion of probabilities (Section 4.2.2), the redistribution of demand (Section 4.2.3), the impact of demand management on individual customers (Section 4.2.4), and long-term credibility (Section 4.2.5).

## 4.2.1 Solution Quality

Table 1 summarizes the metrics defined in Section 4.1.4 (except CR, which we analyze in Section 4.2.4) per policy in the base setting. For customer satisfaction (CS), we include the average aggregate value over the entire evaluation period (Total), its relative improvement over NO under the same allocation scheme in percent (∆%), and the mean per customer (pc) to facilitate the comparison across settings. For locker utilization (LU), we report the minimum (LU) and maximum (LU) average utilization across all lockers.

Except for GP, allocation scheme MY consistently outperforms FF in terms of customer satisfaction given that its priority criterion aligns more closely with the objective. GP produces identical results under both FF and MY because every customer who selects a locker receives the premium outcome for certain, reducing the redirection rate (RR) to zero and rendering allocation decisions trivial. On the flip side, this guarantee is very conservative, lowering the locker share (LS) from 86.6% under NO to 78.9%. Although minimum average locker utilization (LU) increases, reflecting a more balanced use of locker capacity, the maximum average utilization across lockers (LU) decreases as a result of the restrictive policy. Overall, GP generates no meaningful improvement over NO under FF and even deteriorates performance under MY.

Table 2. Probability distortion.
<table><tr><td></td><td colspan="2">Negative</td><td colspan="2">Positive</td></tr><tr><td> $\pi ^ { \mathbf { L A } } ( 1 )$ </td><td>Share</td><td>Mean</td><td>Share</td><td>Mean</td></tr><tr><td>FF</td><td>11.3</td><td>-29.2</td><td>1.5</td><td>5.0</td></tr><tr><td>MY</td><td>9.4</td><td>-15.7</td><td>8.9</td><td>4.4</td></tr></table>

In contrast, providing truthful probabilistic information $( \tau = 0 )$ substantially raises customer satisfaction. Since the transparency constraints for $\tau = 0$ imply a unique feasible solution for the displayed probabilities, EU and LA yield identical results (Property 2). Full transparency improves CS by 12.8% under FF and 7.5% under MY relative to NO. Simultaneously, it lowers the RR from 13.4% to 5.5% and 8.6%, respectively, while keeping the LS nearly constant. Similar to GP, the increase in LU indicates that probability disclosure redistributes demand away from highly utilized lockers toward facilities with more available capacity.

Removing the transparency constraints (τ = 1) produces negligible additional gains under EU. Notably, the policy distorts probabilities only slightly by up to 1.1 and 1.8 pp on average (IN). Conversely, LA with τ = 1 achieves the highest customer satisfaction under both allocation schemes. Unlike EU, which considers only the expected utility of the current request, LA also accounts for the opportunity cost of consuming locker capacity and the potential displacement of other customers. Accordingly, it steers demand more proactively, as reflected in the lower LS and RR. At the same time, these gains are the product of probability distortion of up to 19.9 and 11.1 pp. Across all policies with probability display, customers tend to select options with a high displayed probability of receiving the premium outcome (DC).

Finally, approximation errors (AE) range from 0.4 to 3.5 pp and are consistently lower under FF than under MY. This is to be expected given that FF constitutes a simpler prediction task: Since request arrival times serve as the priority criterion, an order’s ranking relative to other pending orders is fixed and cannot be altered by later-arriving orders. Uncertainty only arises from the number of customers picking up their parcels before allocation. In comparison, later-arriving orders under MY may displace earlier ones, creating an additional source of uncertainty.

## 4.2.2 Distortion of Probabilities

Providing an aggregate overview, Table 2 presents the frequency and magnitude of probability distortions under policies $\pi _ { \mathrm { F F } } ^ { \mathrm { L A } } ( 1 )$ and $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( 1 )$ . More specifically, we separately consider negative and positive deviations between displayed and approximated truthful probabilities $p _ { j } ^ { \pi } ( S _ { k } ) - \hat { \psi } _ { j } ^ { k }$ (in both cases of at least 1 pp in absolute magnitude) and report the share of displayed probabilities afected in % as well as the mean deviation in pp.

![](images/b3bed58d2ebb07831e1ea431ed02aa2c1d1aeecf396d042d3024bf35ed32f961.jpg)  
Figure 1. State-dependent truthful probabilities and their distortion.  
Note. The rows of plots correspond to policies $\pi _ { \mathrm { F F } } ^ { \mathrm { L A } } ( 1 )$ and $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( 1 )$ , respectively.

Under both allocation schemes, negative distortions are more prevalent and larger (in absolute terms) than positive ones, meaning that the policies primarily steer demand by understating the likelihood of receiving the premium outcome. In case of FF, both negative and positive distortions are more aggressive in magnitude than under MY, indicating more substantial adjustments whenever demand management intervenes. By contrast, MY distorts probabilities more frequently overall (afecting 18.3% versus 12.8% of probabilities displayed), particularly due to more positive deviations. The remaining 87.2% and 81.7% of displayed probabilities closely match their approximated truthful counterparts, indicating that the performance gains achieved under τ = 1 relative to τ = 0 arise from targeted adjustments in a contained subset of decisions.

Digging deeper, we note that both the truthful and displayed probabilities under a policy depend on the high-dimensional system state $S _ { k }$ . To gain an understanding of how probabilities get distorted across diferent regions of the state space, we group the observations for each pair of pre-decision state $S _ { k }$ and hypothetical order $o _ { k j }$ with $j \in \mathcal { I } ^ { P }$ encountered during the evaluation periods along two key dimensions, within-day time $t _ { k } ^ { w }$ and system load. We define the system load as the sum of (i) the number of occupied compartments in locker $i _ { j } ^ { + }$ and (ii) the number of pending orders for option $j$ with a higher allocation priority than $o _ { k j }$ . Figure 1 presents the results in the form of heatmaps, each cell containing the average for the associated observations. The first and second rows of plots correspond to policies $\pi _ { \mathrm { F F } } ^ { \mathrm { L A } } ( 1 )$ and $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( 1 )$ , respectively. The middle column reports the approximated truthful probabilities in ${ \% } ,$ while the left and right columns report negative and positive deviations in pp. To ensure representativeness, we only show cells containing at least 30 observations.

The truthful probabilities are generally high when the system load remains below the total locker capacity of 10 compartments. Delivery to the locker is guaranteed under FF in these states, consistently resulting in a truthful probability of 100%. Once the system load exceeds locker capacity, the truthful probabilities for FF decrease as the allocation decision at $t _ { k } ^ { w } = 6 1$ approaches: With less time remaining, occupied compartments are less likely to become available through parcel pickups before allocation.

While early arrival grants a high allocation priority under FF, it provides no comparable protection under MY. As a result, the truthful probabilities for MY may fall below 100% even for system loads under 10. Interestingly, the within-day time has two opposing efects: On the one hand, less time until allocation reduces the likelihood for occupied compartments to become available. On the other hand, there are fewer opportunities for future orders to arrive that potentially receive a higher allocation priority. For system loads below 10, the latter efect dominates, such that the truthful probabilities under MY increase over the course of the day. At system loads above 10, the locker occupancy tends to be high, especially early in the day, and the allocation outcome of the considered order depends more strongly on capacity becoming available through parcel pickups, such that truthful probabilities decrease as the day progresses.

Consistent with the findings from Table 2, negative deviations are generally larger in absolute terms and more widespread than positive distortions under FF. In light of the already high truthful probabilities, there simply is little room for upward distortion. Negative distortion is strongest early in the day, when most of the demand is yet to come, and under higher system loads, when capacity is more scarce. Given the generally lower truthful probabilities in the majority of heatmap cells, negative deviations are weaker under MY and increase towards the end of the day, whereas positive deviations occur more frequently and strongly. Pronounced positive distortions towards the end of the day and under high system loads show that the provider may also steer customers by artificially inflating the attractiveness of lockers despite a substantial risk of redirection.

## 4.2.3 Redistribution of Demand

To analyze how probability display and allocation decisions jointly shape which service outcomes customers receive, we randomly select one of the 30 base setting instances and examine it in more detail. We focus on all customers whose preferred service outcome (with the highest utility $u _ { i } ^ { k } )$ is a parcel locker $i \in \mathcal { T } ^ { + }$ and divide them into three groups:

1. PP (preferred and premium): Customers who choose the probabilistic service option containing their preferred locker and receive the locker premium outcome upon allocation.

Table 3. Group shares (%).
<table><tr><td>Policy π</td><td></td><td>T</td><td>PP</td><td>PB</td><td>00</td></tr><tr><td>FF</td><td>NO</td><td>一</td><td>84.0</td><td>16.0</td><td>0.0</td></tr><tr><td></td><td>LA</td><td>0</td><td>80.4</td><td>6.5</td><td>13.1</td></tr><tr><td></td><td></td><td>1</td><td>75.0</td><td>2.9</td><td>22.0</td></tr><tr><td>MY</td><td>NO</td><td>一</td><td>84.0</td><td>16.0</td><td>0.0</td></tr><tr><td></td><td>LA</td><td>0</td><td>79.2</td><td>9.5</td><td>11.2</td></tr><tr><td></td><td></td><td>1</td><td>77.2</td><td>6.4</td><td>16.5</td></tr></table>

2. PB (preferred and base): Customers who choose the probabilistic service option containing their preferred locker but receive the associated parcel shop base outcome.

3. OO (other option): Customers who choose a service option that does not contain their preferred locker.

Table 3 reports the share of customers in each group relative to the total number of lockerpreferring customers for selected policies. Under NO, customers choose the service option with their favorite facility by design, such that the share of group OO equals 0. Moreover, customers choices do not difer under FF and MY, resulting in identical sets of pending orders throughout the evaluation period. Nevertheless, the allocation schemes may produce distinct locker-occupancy trajectories because pickup realizations may vary depending on which specific orders end up in the locker. In total, both schemes allocate the preferred locker to 84.0% of the considered customers (PP) and redirect the remaining 16.0% (PB).

The probabilistic information shared under LA and full transparency induces 13.1% (FF) and 11.2% (MY) of customers to forgo their preferred locker in favor of an alternative (OO). Truthful information therefore facilitates customer self-selection: Depending on the utilities associated with the preferred locker and backup parcel shop, the availability of other options, the displayed probabilities and the resulting valuations, customers proactively select alternative service options when the displayed probability for their preferred locker renders the probabilistic service option unattractive.

Removing the transparency constraints intensifies this redistribution. The share of group OO increases to 22.0% (FF) and 16.5% (MY). Beyond merely informing customers, probabilities displayed under τ = 1 act as an active demand management mechanism, guiding customers away from their preferred option (even when allocation might otherwise have been possible) to preserve capacity for other orders. FF relies on this mechanism more strongly than MY, leading to a higher share for OO and a lower share for PB.

Figure 2 provides a customer-level view of these mechanisms. In each plot, the horizontal and vertical axes represent the utilities for the backup shop and the preferred locker, respectively. Because the sample is restricted to customers whose preferred outcome is a locker, all observations lie strictly above the 45-degree diagonal, with customers positioned close to it being nearly indiferent between the two outcomes.

![](images/e76c941a8c0d8eb3fcb09965f55c491ef3d9c0d3a110f671a32e5b3a6445f779.jpg)  
Figure 2. Customer segmentation.  
Note. The rows of plots correspond to allocation schemes FF and MY, while the columns correspond to NO, LA under full transparency (τ = 0), and LA without transparency constraints (τ = 1).

Under NO, both FF and MY generate identical shares for PP and PB, but Figure 2 reveals that they systematically allocate the base outcome to diferent customers. Under FF, observations belonging to PB are distributed broadly across the utility space given that the priority for allocation solely depends on the request arrival sequence. Consequently, redirection frequently also impacts customers for whom the backup shop yields a substantial utility decrease compared to the backup shop. In contrast, MY shifts observations for PB closer toward the diagonal, targeting customers for whom the backup shop is less unattractive, which is also evident for the LA-based policies. Because customer choices under NO are independent of the allocation scheme, the improvement in customer satisfaction (Table 1) from FF to MY for NO stems entirely from allocating the locker capacity to customers who value it more highly.

Under full transparency (τ = 0), the composition of group OO similarly depends on the allocation scheme. Under FF, this group contains many customers with relatively low backup-shop utility. Because arrival order alone determines allocation priority, these customers may face a substantial risk of receiving an unattractive base outcome and therefore avoid the probabilistic option when this risk is disclosed. Under MY, a low backup-shop utility (ceteris paribus) increases the utility loss from redirection and, consequently, the customer’s allocation priority, tending to result in a high probability. Customers with low backup-shop utility are accordingly less likely to forgo their preferred-locker option under MY than under FF.

Removing the transparency constraints (τ = 1) expands group OO, for FF particularly among customers exhibiting small utility diferences between their preferred locker and its backup shop. Steering these nearly indiferent customers, located close to the diagonal, toward alternative options (such as the backup shop ofered as a deterministic option) incurs a comparatively small utility loss. Under MY, the allocation priority criterion automatically accounts for this, resulting in less need to to steer these customers through probability distortions. The figure also indicates increased steering among customers with low backup shop utilities, especially in case of MY, which is linked to the spatial patterns discussed in the subsequent section.

Table 4. Regret distribution.
<table><tr><td></td><td colspan="7">Utility loss  $\mathrm { ( C R > 0 ) }$ </td><td colspan="7">Utility gain  $\left( \left| \mathrm { C R } < 0 \right| \right)$ </td></tr><tr><td> $\pi ^ { \mathbf { L A } } ( 1 )$ </td><td>Total</td><td>Share</td><td>Mean</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Max.</td><td>Total</td><td>Share</td><td>Mean</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Max.</td></tr><tr><td>FF</td><td>1,257.2</td><td>12.2</td><td>2.5</td><td>0.8</td><td>1.7</td><td>3.3</td><td>20.4</td><td>2,139.5</td><td>7.2</td><td>7.3</td><td>3.1</td><td>6.6</td><td>10.4</td><td>24.4</td></tr><tr><td>MY</td><td>1,122.2</td><td>8.9</td><td>3.1</td><td>1.0</td><td>2.1</td><td>4.1</td><td>19.2</td><td>1,940.4</td><td>7.6</td><td>6.3</td><td>2.0</td><td>5.3</td><td>9.4</td><td>23.1</td></tr></table>

## 4.2.4 Customer Regret

While removing the transparency constraints boosts total customer satisfaction (Table 1), the aggregate metrics do not reveal how the resulting gains and losses are distributed across individual customers. For this purpose, Table 4 diferentiates between customers experiencing a utility loss $\mathrm { ( C R > 0 ) }$ and those profiting from a utility gain $\mathrm { ( C R < 0 ) }$ . To facilitate a direct comparison, utility gains are reported as absolute values. Averaging across evaluation streams, we report the aggregate regret (Total), the share of afected customers in percent (Share), as well as the mean, quartiles, and maximum of regret per afected customer (Mean, Q1, Q2, Q3, Max.).

Under both allocation schemes, aggregate utility gains outweigh the losses, aligning with the overall increase in customer satisfaction. Notably, these outcomes are unevenly distributed: The share of customers sufering a utility loss (12.2% for FF and 8.9% for MY) exceeds the share of those experiencing a utility gain (7.2% and 7.6%), and the mean gain per afected customer is more than twice as large as the mean loss. The remaining majority of customers (80.6% and 83.5%) receives the identical service outcome regardless of whether transparency constraints are applied. Comparing the quartiles underscores the asymmetry in the distribution of gains and losses. Across all quartiles (Q1, Q2, Q3), the utility loss per afected customer is substantially smaller than the utility gain. Overall, a larger proportion of customers incurs a modest individual utility loss, thereby enabling substantial utility gains for a smaller group of beneficiaries.

To assess whether demand management puts particular customers at a systematic (dis)advantage, Figure 3 depicts the average regret by customer location under policies $\pi _ { \mathrm { F F } } ^ { \mathrm { L A } } ( 1 )$ and $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( 1 )$ . Complementing the spatial analysis, Figure 4 links regret to the within-day request arrival time. In both figures, positive CR values indicate a utility loss compared to full transparency, whereas negative CR implies a utility gain.

Figure 3 reveals considerable spatial heterogeneity in CR. Under FF, customers who benefit from demand management tend to be located at the periphery of the network. These customers often face long distances to alternative facilities and would incur comparatively large utility losses if denied access to their preferred locker. By steering more centrally located customers toward alternative facilities, demand management preserves scarce capacity for peripheral customers with fewer attractive substitutes. Under MY, this spatial pattern is less evident. Because MY inherently

![](images/9b53afa5ffda076fba855df9fe65f1fdfa8ba3c7c39bfe45c3ae11a4fab3473b.jpg)  
Figure 3. Spatial regret distribution.

![](images/a055beb94f199dc4a3508960ed3e13293cdf487cc5bab2503f134633665afa0c.jpg)  
Figure 4. Temporal regret distribution.

Note. The plots represent policies $\pi _ { \mathrm { F F } } ^ { \mathrm { L A } } ( 1 )$ and $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( 1 )$ , respectively.

prioritizes orders according to the utility loss associated with redirection, customers located far from the backup shop already receive a high allocation priority. Consequently, there is less need to additionally protect them through demand management. Essentially, MY embeds spatial diferentiation directly into the allocation decision itself, whereas FF relies more heavily on demand management to achieve a comparable efect.

The upper-right corner of the network for policy $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( 1 )$ nicely demonstrates that the spatial efects of demand management depend on the availability of alternative service options. The red cluster of customers in this region typically prefers the locker located at (6.9, 6.9) and faces a severe utility loss if redirected to its backup shop at (8.8, 4.7), which grants their orders high allocation priority under MY. At the same time, demand management can steer these customers toward another facility (such as the parcel shop at (2.6, 7.8) or the locker at (5.0, 5.0)) at a comparatively small utility cost, freeing up capacity and producing utility gains for customers in the neighboring blue cluster. A second cluster of customers benefiting from demand management under MY emerges in the lower-left part of the network. These customers compete for locker capacity with peripheral customers that exhibit larger utility losses. Under full transparency, the policy tends to indiscriminately display a high probability to these peripheral customers since they have a high chance of displacing orders from customers located closer to the center. Demand management can trade of the utility loss from steering peripheral customers toward another option with the potential utility loss due to redirection, especially in case of already existing pending orders from central customers.

As a second key dimension for our regret analysis, Figure 4 illustrates systematic diferences in customer regret depending on the request arrival time. Under FF and full transparency, customers arriving early possess an inherent advantage: Once an order has arrived, the provider cannot alter its ranking for allocation. To counteract this, demand management intentionally steers selected early arrivals, e.g., with comparatively low locker utility (Section 4.2.3), toward alternative options, at the cost of utility losses for the current customer. In anticipation of later-arriving customers, the policy preserves capacity, ultimately generating utility gains later in the day. This manifests itself in a decreasing trend for the regret curve.

![](images/093f73a984d78261538d8ba0bb8d3e130956e34f58b853cc61483427546f6e4b.jpg)  
Figure 5. Redirection rates.  
Note. The rows of plots correspond to allocation schemes FF and MY, while the columns correspond to LA under full transparency (τ = 0) and without transparency constraints (τ = 1).

Under MY, the temporal dynamics are reversed. Later-arriving orders with larger utility losses from redirection can displace existing orders. However, even if the current customer is highly likely to receive the locker, encouraging them to select the underlying probabilistic service option by honestly communicating the probability is not necessarily optimal for aggregate customer satisfaction: Although the displacement benefits the latecomer, it can result in a greater total utility loss than if the latecomer had simply been steered toward an alternative. Demand management addresses this trade-of by guiding later-arriving customers’ choices toward other options, even when their truthful probability of securing the locker is high. Consequently, probability distortion under MY tends to protect earlier orders from displacement, resulting in utility gains during the early part of the day and utility losses towards the end of the day.

In both cases, the results illustrate that demand management compensates weaknesses of the underlying allocation approaches, thereby leading to improved aggregate satisfaction. On the flip side, gains and losses are not randomly distributed across the customer population and systematically difer based on location and request arrival time.

## 4.2.5 Long-Term Credibility

From a managerial perspective, long-term credibility is critical; in particular when customers repeatedly interact with the system, as is the case for OOHD. To this end, Figure 5 compares the observed (empirical) redirection rates (RR) with the (expected) rates implied by the communicated probabilities. We divide the observations into bins according to the probability displayed for the option chosen by the customer and additionally report the share out of all customers per bin. The rows of plots correspond to FF and MY, whereas the columns represent LA for τ = 0 and τ = 1.

Table 5. Performance metrics for varying locker utility constant $\beta _ { L }$
<table><tr><td></td><td></td><td></td><td colspan="3">CS</td><td colspan="5">(%)</td><td colspan="2">(pp)</td></tr><tr><td> $\beta _ { L }$ </td><td> $\pi ^ { \mathrm { L A } }$ </td><td>T</td><td>Total</td><td>∆%</td><td>pc</td><td>LS</td><td>RR</td><td>LU</td><td>LU</td><td>DC</td><td>IN</td><td>AE</td></tr><tr><td>0</td><td>FF</td><td>0</td><td>-23,076.3</td><td>7.6</td><td>-5.7</td><td>78.6</td><td>4.5</td><td>41.7</td><td>94.4</td><td>94.8</td><td>0.0</td><td>0.6</td></tr><tr><td></td><td></td><td>1</td><td>-22,449.2</td><td>10.1</td><td>-5.5</td><td>76.0</td><td>2.1</td><td>47.9</td><td>90.2</td><td>91.7</td><td>17.2</td><td>0.3</td></tr><tr><td></td><td>MY</td><td>0</td><td>-22,448.6</td><td>4.2</td><td>-5.5</td><td>79.3</td><td>7.3</td><td>44.6</td><td>94.0</td><td>90.1</td><td>0.0</td><td>2.9</td></tr><tr><td></td><td></td><td>1</td><td>-21,966.3</td><td>6.3</td><td>-5.4</td><td>78.2</td><td>4.8</td><td>44.1</td><td>92.3</td><td>89.8</td><td>9.9</td><td>2.0</td></tr><tr><td>4</td><td>FF</td><td>0</td><td>-11,356.1</td><td>23.3</td><td>-2.8</td><td>90.4</td><td>5.7</td><td>85.6</td><td>97.3</td><td>93.0</td><td>0.0</td><td>0.7</td></tr><tr><td></td><td></td><td>1</td><td>-10,375.0</td><td>29.9</td><td>-2.6</td><td>86.2</td><td>3.4</td><td>84.8</td><td>93.6</td><td>91.0</td><td>21.4</td><td>0.4</td></tr><tr><td></td><td>MY</td><td>0</td><td>-10,722.1</td><td>14.7</td><td>-2.6</td><td>91.6</td><td>10.6</td><td>81.1</td><td>96.4</td><td>88.3</td><td>0.0</td><td>3.7</td></tr><tr><td></td><td></td><td>1</td><td>-9,567.4</td><td>23.8</td><td>-2.4</td><td>89.6</td><td>7.0</td><td>85.2</td><td>94.1</td><td>88.4</td><td>12.6</td><td>2.8</td></tr></table>

Under full transparency, empirical and expected redirection rates mostly align, with limited discrepancies occurring in probability ranges with low customer shares. Without transparency constraints, the gap between the two rates widens substantially but continues to mostly afect sparsely populated bins. The expected RR consistently exceeds the empirical one, indicating that the policy systematically signals a higher risk of redirection than actually prevalent.

Essentially, the provider faces a trade-of between the gains from demand management and the credibility of the communicated probabilities. If customers repeatedly receive an outcome presented as unlikely, they may cease to regard the probabilities as reliable, undermining the efectiveness of probability display as a demand management lever. At the same time, customers’ ability to detect systematic distortions would require frequent exposure to probability bins with strong discrepancies between expected and empirical RR.

## 4.3 Impact of Setting Parameters

Building on the base setting, we examine how variations in key problem parameters afect the value of information provision and additional gains from demand management.

## 4.3.1 Systematic Utility Diference

To assess the role of customers’ systematic preference for locker delivery, we vary the additive locker utility constant from its base value of $\beta _ { L } = 2$ to $\beta _ { L } \in \{ 0 , 4 \}$ and present the results in Table 5. Essentially, $\beta _ { L } = 0$ implies that lockers ofer no systematic utility advantage over parcel shops, whereas $\beta _ { L } = 4$ raises the utility premium associated with lockers.

In combination with Table 1, the results show that customer satisfaction (CS) generally increases with $\beta _ { L }$ , reflecting the greater utility customers derive from locker delivery. More importantly, the value of providing truthful information $( \tau = 0 )$ likewise grows with the attractiveness of lockers. Full transparency under $\beta _ { L } = 0$ improves customer satisfaction relative to NO by 7.6% and 4.2% for FF and MY, respectively, compared with 12.8% and 7.5% in the base setting. These improvements soar to 23.3% and 14.7% for $\beta _ { L } = 4$

![](images/9ebc28853f0814acc7237ca11b6b0d4b26508c59315099a13caf1028e03e4cef.jpg)  
Figure 6. Performance of policy $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( \tau )$ in setting $\beta _ { L } = 4$

We trace this pattern back to three mechanisms: First, a higher locker utility premium $\beta _ { L }$ attracts more customers to lockers, as reflected in LS and LU, which intensifies competition for their limited capacity. Second, the potential utility loss from redirection to a parcel shop tends to increase because redirected customers forgo the systematic utility premium associated with lockers. Third, a higher $\beta _ { L }$ expands the set of lockers that constitute attractive alternatives for a customer despite being located farther away than parcel shops. This gives customers more options to respond to the disclosed probabilities by switching among facilities and, consequently, increases the provider’s ability to redistribute demand through information provision. Conversely, customers under $\beta _ { L } = 0$ select lockers less frequently, reducing the scarcity of locker capacity and, consequently, the value of information provision. For the same reasons, the amount of probability distortion (IN) and the additional gains in CS from deactivating the transparency constraints follow a similar pattern. Relative to full transparency, LA with $\tau = 1$ improves CS by 2.7% and 2.1% for FF and MY under $\beta _ { L } = 0$ , compared with 5.1% and 4.9% in the base setting as well as 8.6% and 10.8% for $\beta _ { L } = 4$

Given that we observe the largest demand management gains for $\beta _ { L } = 4$ and MY, we investigate this setting in more detail by varying τ in increments of 0.05 for policy $\pi _ { \mathrm { M Y } } ^ { \mathrm { L A } } ( \tau )$ . Figure 6 depicts the resulting evolution of aggregate CS. Consistent with the monotonicity established in Property 1, CS generally improves with rising τ. Notably, modest relaxations of the transparency constraints already yield substantial improvements, whereas the marginal gains diminish with τ. CS reaches a plateau at $\tau = 0 . 3 5$ , beyond which further relaxation produces no additional benefit. Overall, the results illustrate that moderate flexibility in probability displays may already sufice for efective demand management.

## 4.3.2 Network Size

To test the scalability of our method, we increase the number of facilities I in the network, thereby modifying and expanding the set of ofered service options. Because the total service area remains fixed, this efectively results in a denser network with shorter distances between customers and their closest facilities. Table 6 demonstrates that the mean satisfaction per customer (pc) improves in response. Note that we scale demand proportionally with I, leading to a decreasing value for total

Table 6. Performance metrics for varying network size I.
<table><tr><td></td><td></td><td></td><td colspan="3">CS</td><td colspan="6">(%)</td><td colspan="2">(pp)</td></tr><tr><td>I</td><td> $\pi ^ { \mathrm { L A } }$ </td><td>T</td><td>Total</td><td>∆%</td><td>pc</td><td>LS</td><td>RR</td><td></td><td>LU</td><td>LU</td><td>DC</td><td>IN</td><td>AE</td></tr><tr><td>15</td><td>FF</td><td>0</td><td>-18,561.6</td><td>15.4</td><td>-3.1</td><td>84.3</td><td></td><td>5.1</td><td>64.9</td><td>96.5</td><td>93.9</td><td>0.0</td><td>0.6</td></tr><tr><td></td><td></td><td>1</td><td>-17,552.3</td><td>20.0</td><td>-2.9</td><td></td><td>80.5</td><td>2.3</td><td>71.3</td><td>93.0</td><td>91.6</td><td>22.2</td><td>0.2</td></tr><tr><td></td><td>MY</td><td>0</td><td>-18,000.5</td><td>8.8</td><td>-3.0</td><td></td><td>84.3</td><td>8.7</td><td>55.7</td><td>95.5</td><td>88.7</td><td>0.0</td><td>3.4</td></tr><tr><td></td><td></td><td>1</td><td>-16,867.1</td><td>14.6</td><td></td><td>-2.8</td><td>84.2</td><td>6.1</td><td>67.1</td><td>94.9</td><td>88.4</td><td>12.6</td><td>2.3</td></tr><tr><td>20</td><td>FF</td><td>0</td><td>-21,359.6</td><td>21.3</td><td></td><td>-2.6</td><td>86.7</td><td>6.7</td><td>66.0</td><td>99.5</td><td>93.1</td><td>0.0</td><td>0.7</td></tr><tr><td></td><td></td><td>1</td><td>-20,020.6</td><td>26.2</td><td>-2.5</td><td></td><td>82.2</td><td>3.1</td><td>67.6</td><td>96.5</td><td>89.6</td><td>23.8</td><td>0.3</td></tr><tr><td></td><td>MY</td><td>0</td><td>-21,007.1</td><td>13.1</td><td>-2.6</td><td></td><td>86.9</td><td>10.7</td><td>62.7</td><td>99.4</td><td>88.4</td><td>0.0</td><td>4.6</td></tr><tr><td></td><td></td><td>1</td><td>-19,023.6</td><td>21.3</td><td>-2.4</td><td></td><td>86.3</td><td>7.4</td><td>66.0</td><td>98.6</td><td>86.6</td><td>15.2</td><td>2.8</td></tr><tr><td>25</td><td>FF</td><td>0</td><td>-24,878.4</td><td>27.1</td><td></td><td>-2.5</td><td>87.8</td><td>8.2</td><td>26.9</td><td>99.8</td><td>91.8</td><td>0.0</td><td>0.8</td></tr><tr><td></td><td></td><td>1</td><td>-22,553.7</td><td>33.9</td><td>-2.2</td><td></td><td>82.5</td><td>3.2</td><td>35.3</td><td>97.1</td><td>87.3</td><td>26.2</td><td>0.5</td></tr><tr><td></td><td>MY</td><td>0</td><td>-25,199.0</td><td>17.1</td><td>-2.5</td><td></td><td>88.1</td><td>12.9</td><td>16.5</td><td>99.6</td><td>86.5</td><td>0.0</td><td>5.5</td></tr><tr><td></td><td></td><td>1</td><td>-21,570.1</td><td>29.1</td><td>-2.1</td><td>87.4</td><td></td><td>8.5</td><td>32.4</td><td>99.3</td><td>83.4</td><td>17.6</td><td>3.2</td></tr></table>

CS (despite the improvement in mean utility per customer) because the provider serves a larger number of customers.

The value of truthful information provision increases with network size. Relative to NO, full transparency improves CS under FF by 15.4%, 21.3%, and 27.1% for $I \in \{ 1 5 , 2 0 , 2 5 \}$ , respectively. Under MY, the corresponding improvements climb from 8.8% to 13.1% and 17.1%. As the network becomes denser, customers choose from a broader set of attractive facilities, and truthful probabilities become increasingly useful for diferentiating among them. Notably, RR also rises with the network size under both allocation schemes. In a denser network, the greater number of parcel shops reduces the likelihood that the base outcome associated with a locker is located far from the customer, making redirection less inconvenient. Consequently, customers become more willing to accept this risk. These same aspects also lead to denser OOHD networks ofering more potential for demand management. Relative to full transparency, removing the constraints improves CS under FF by 5.4%, 6.3%, and 9.3% for $I \in \{ 1 5 , 2 0 , 2 5 \}$ , respectively, with the corresponding gains ranging from 6.3% to 9.4% and 14.4% under MY.

## 5 Conclusion and Outlook

In this work, we introduce probabilistic service options to dynamically communicate and manage uncertainty about future service outcomes. Upon the arrival of a request, the provider displays the probabilities with which the customer will receive a capacity-constrained premium outcome at the time of service provision. Unlike existing work on probabilistic products, which treats the probabilities as a static design choice, the truthful probabilities in our setting emerge endogenously from the interaction of future demand, customer choices, allocation decisions, and uncertain capacity utilization, efectively depending both on the system state and the policy applied by the provider. By demonstrating that probability displays can serve as a novel information-based demand management lever, our work raises a fundamental question regarding the extent to which providers should be permitted to distort probabilities to optimize aggregate system performance.

As novel modeling contributions, we formulate a recursive equation to compute the truthful probabilities and propose transparency constraints that limit the deviations between displayed and truthful probabilities. The underlying idea of recursively evaluating the state-dependent probability that a certain event will occur under a policy is not limited to our problem and can be transferred to any sequential decision process. Because the exact computation is intractable for realistic instances, we develop an adapted reinforcement learning procedure as an approximation. We are confident that our method of constructing policies based on approximated, policy-dependent probabilities of future events is a promising starting point for future work.

Managerially, our results indicate that dynamic information provision is a valuable tool to counteract imbalances in demand. Additionally, we find that targeted adjustments of probabilities and a slight relaxation of the allowed level of intransparency can already generate considerable improvements. However, certain customer groups may be put at a systematic disadvantage, and the provider must trade of the gains attainable through demand management with the potential erosion of credibility if distortions are detected.

To conclude, we highlight two compelling directions for future research:

1. Advanced allocation mechanisms. We restrict attention to non-anticipatory and pre-determined allocation schemes, which enables us to establish the uniqueness of fully transparent probability displays and to isolate the efects of demand management. Future research could study anticipatory allocation strategies where the provider takes the future evolution of the system into account when allocating service outcomes. Another important extension would be the joint optimization of probability display and allocation and the potential feedback loops between the two decision types.

2. Customer learning. We assume that customers do not update their beliefs about the accuracy of the communicated probabilities and the provider’s trustworthiness over time. Incorporating such dynamics in future work would uncover whether demand management gains persist under this long-term perspective.

## Appendices

## Appendix A Notation

$i \in { \mathcal { T } } = { \mathcal { T } } ^ { - } \cup { \mathcal { T } } ^ { + }$ Service outcomes, partitioned into base outcomes $\mathcal { Z } ^ { - }$ and premium outcomes ${ \mathcal { T } } ^ { + }$   
Q<sub>i</sub> Capacity of premium outcome $i \in \mathcal { T } ^ { + }$   
$j \in \mathcal { I } = \mathcal { I } ^ { P } \cup \mathcal { I } ^ { D }$ Service options, partitioned into probabilistic service options $\mathcal { I } ^ { P }$ and deterministic service options $\mathcal { I } ^ { D }$   
$( i _ { j } ^ { - } , i _ { j } ^ { + } )$ Base outcome $i _ { j } ^ { - } \in \mathcal { T } ^ { - }$ and premium outcome $i _ { j } ^ { + } \in \mathcal { I } ^ { + }$ of probabilistic service option $\boldsymbol { j } \in \mathcal { I } ^ { P }$   
$i _ { j }$ Service outcome $i _ { j } \in \mathcal { I } ^ { - }$ of deterministic service option $\boldsymbol { j } \in \mathcal { I } ^ { \boldsymbol { D } }$   
$t \in \mathcal T$ Point in time   
p<sub>j</sub> Displayed probability for premium outcome of probabilistic service option $\boldsymbol { j } \in \mathcal { I } ^ { P }$   
$t \in { \mathcal { T } } ^ { A } \subset { \mathcal { T } }$ Points in time for allocation   
$k = 1 , \ldots , \infty$ Decision epoch   
$S _ { k } = ( t _ { k } , r _ { k } , \mathcal { O } _ { k } , \mathcal { C } _ { k } ) \in \mathcal { S }$ Pre-decision state in decision epoch k   
$t _ { k }$ Point in time in decision epoch k   
$r _ { k }$ Request in decision epoch k   
$t _ { r _ { k } } ^ { A } \in \mathcal { T } ^ { A }$ Allocation point in time for request in decision epoch k   
$o \in \mathcal { O } _ { k }$ Pending orders in decision epoch k   
$j _ { o } \in \mathcal { I }$ Service option chosen by order o   
$t _ { o } ^ { A } \in \mathcal { T } ^ { A }$ Allocation point in time for order o   
$\mathcal { C } _ { k } = \{ \mathcal { C } _ { i k } : i \in \mathcal { I } ^ { + } \}$ Capacity utilization in decision epoch k   
$q _ { i }$ Available capacity of premium outcome $i \in \mathcal { T } ^ { + }$   
$S _ { 0 }$ Initial state   
$X _ { k } \in \mathcal { X } ( S _ { k } )$ Decision in decision epoch k   
$P _ { k } = ( p _ { j } ^ { k } ) _ { j \in \mathcal { I } ^ { P } }$ Probability display decision in decision epoch k   
$\mathcal { O } _ { k } ^ { A } = \{ o \in \mathcal { O } _ { k } : t _ { o } ^ { A } = t _ { k } \}$ Orders to be allocated in decision epoch k   
$A _ { k } = ( a _ { o } ^ { k } ) _ { o \in \mathcal { O } _ { k } ^ { A } }$ Allocation decision in decision epoch k   
$W _ { k + 1 } = \left( j _ { r _ { k } } , \mathcal { C } _ { k + 1 } , r _ { k + 1 } , t _ { k + 1 } \right) \in \mathcal { W }$ Exogenous information revealed during transition from decision epoch k to $k + 1$   
$\mathbb { P } ( W _ { k + 1 } | S _ { k } , X _ { k } )$ Probability distribution of exogenous information   
$u _ { i } ^ { k }$ Utility that request in decision epoch k associates with service outcome $i \in \mathcal { Z }$   
$U _ { j } ^ { k } = U ( P _ { k } , u _ { i _ { j } ^ { - } } , u _ { i _ { j } ^ { + } } )$ Valuation for service option $j \in \mathcal I$   
$j _ { r _ { k } } \in \mathcal I$ Service option chosen by the request in decision epoch k   
$R ( S _ { k } , X _ { k } , W _ { k + 1 } )$ Reward function   
$\pi \in \Pi$ Policy   
$X _ { k } ^ { \pi } ( S _ { k } )$ Decision in pre-decision state $S _ { k }$ under policy $\pi$   
$\pi ^ { * } \in \Pi$ Optimal policy   
R¯<sup>π</sup> Expected average reward per decision epoch under policy π   
$S _ { k } ^ { c } = ( t _ { k } , r _ { k } , \mathcal { O } _ { k + 1 } , \mathcal { C } _ { k } , \mathcal { O } _ { k } ^ { A } , A _ { k } ) \in \mathcal { S } ^ { c }$ Post-choice state in decision epoch k   
$W _ { k + 1 } ^ { c } = j _ { r _ { k } } \in \mathcal { W } ^ { c }$ Exogenous information for transition from pre-decision state to post-choice state   
$S _ { k } ^ { c } = S ^ { M W ^ { c } } ( S _ { k } , X _ { k } , W _ { k + 1 } ^ { c } )$ Transition function from pre-decision state to post-choice state   
$W _ { k + 1 } ^ { s } = ( \mathcal { C } _ { k + 1 } , r _ { k + 1 } , t _ { k + 1 } ) \in \mathcal { W } ^ { s }$ Exogenous information for transition from post-choice state to pre-decision state   
$S _ { k + 1 } = S ^ { M W ^ { s } } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } )$ Transition function from post-choice state to pre-decision state   
ψ<sup>π</sup><sub>o</sub> (S<sup>c</sup><sub>k</sub>) Truthful probability   
$\mathbb { P } ( W _ { k + 1 } ^ { c } | S _ { k } , X _ { k } ) , \mathbb { P } ( W _ { k + 1 } ^ { s } | S _ { k } ^ { c } )$ Probability distribution of exogenous information   
V (S) Value function in pre-decision state $S \in S$   
V<sup>c</sup>(S) Value function in post-choice state $S \in S ^ { c }$   
$R ^ { c } ( S _ { k } , X _ { k } , W _ { k + 1 } ^ { c } )$ Reward function for transition from pre-decision state $S _ { k }$ to post-choice state $S _ { k } ^ { c }$   
$R ^ { s } ( S _ { k } ^ { c } , W _ { k + 1 } ^ { s } )$ Reward function for transition post-choice state $S _ { k } ^ { c }$ to pre-decision state $S _ { k + 1 }$   
$S ^ { p } = \{ S _ { k } \in { \mathcal { S } } : r _ { k } \neq 0 \}$ Set of pre-decision states with probability display decision   
$P ^ { \pi } ( S _ { k } ) = ( p _ { j } ^ { \pi } ( S _ { k } ) ) _ { j \in \mathcal { I } ^ { P } }$ Probabilities displayed under policy π in pre-decision state $S _ { k }$   
o<sub>kj</sub> Hypothetical order for probabilistic service option $\boldsymbol { j } \in \mathcal { I } ^ { P }$ in pre-decision state $S _ { k }$   
$S _ { k j } ^ { c } = S ^ { M W ^ { c } } ( S _ { k } , P ^ { \pi } ( S _ { k } ) , j )$ Hypothetical post-choice state for probabilistic service option $j \in \mathcal { I } ^ { P }$ and pre-decision state $S _ { k }$   
τ ∈ [0, 1] Level of intransparency   
Π(τ ) Set of feasible policies for level of intransparency τ   
α Allocation scheme   
$\hat { \psi } _ { o } ( S _ { k } ^ { k } )$ Approximated truthful probability   
$\hat { \psi } _ { \pmb { \theta } }$ Neural network

<table><tr><td>θ</td><td>Parameter weights of neural network</td></tr><tr><td> $\phi ( S _ { k } ^ { c } , o )$ </td><td>Feature vector for neural network</td></tr><tr><td> $e \in { \mathcal { E } }$ </td><td>Experience in replay memory</td></tr><tr><td> $n = 1 , \ldots , N$ </td><td>Decision epoch simulated for offline training</td></tr><tr><td> $\hat { \psi } _ { j } ^ { n } = \hat { \psi } _ { \pmb { \theta } } \big ( \phi ( S _ { n j } ^ { c } , o _ { n j } ) \big )$ </td><td>Approximated truthful probability for probabilistic service option in decision epoch n</td></tr><tr><td> $\left( o _ { e } , \phi _ { e } , a _ { e } \right)$ </td><td>Order, feature vector, and allocation outcome associated with experience e</td></tr><tr><td> $B \subset { \mathcal { E } }$ </td><td>Sampled mini-batch of experiences</td></tr><tr><td> $\mathcal { L } ( \pmb \theta , \boldsymbol B )$ </td><td>Binary cross-entropy loss</td></tr><tr><td> $f = 1 , \ldots , F$ </td><td>Sampled trajectories</td></tr><tr><td> $a _ { o f }$ </td><td>Allocation outcome for order o in trajectory f</td></tr><tr><td> $\hat { \psi } _ { o } ^ { M C }$ </td><td>Monte Carlo estimate for truthful probability</td></tr><tr><td> $\hat { V } ^ { c } ( S _ { k } ^ { c } )$ </td><td>Approximated post-choice state value function</td></tr><tr><td> $\omega \in \Omega$ </td><td>Utility scenario</td></tr><tr><td> $\tilde { u } _ { i \omega }$ </td><td>Sampled utility for service outcome  $i \in \mathcal { Z }$  in utility scenario ω</td></tr><tr><td> $\tilde { U } _ { j \omega }$ </td><td>Valuation for service option  $j \in \mathcal I$  in utility scenario ω </td></tr><tr><td> $\tilde { U } _ { \omega }$ </td><td>Maximum valuation in utility scenario ω</td></tr><tr><td> $c _ { j \omega }$ </td><td>Customer choice in utility scenario  $\omega$ </td></tr><tr><td>国  ${ } ^ { \circ } ( W _ { k + 1 } ^ { c } = j | S _ { k } , X _ { k } )$ </td><td>Approximated choice probability</td></tr><tr><td> $M , M ^ { \prime } , M _ { \omega }$ </td><td>Sufficiently large constants</td></tr><tr><td> $t _ { k } ^ { d } = 1 , \dots , \infty$ </td><td>Day in decision epoch k</td></tr><tr><td> $t _ { k } ^ { w } = 1 , \dots , T + 1$ </td><td>Within-day time in decision epoch k</td></tr><tr><td> $v _ { i } ^ { k }$ </td><td>Deterministic utility that request in decision epoch k associates with service  $i \in \mathcal { Z }$ </td></tr><tr><td> $\beta _ { L }$ </td><td>outcome Additive locker utility constant</td></tr><tr><td> $\delta _ { r _ { k } i }$ </td><td>Euclidean distance between location of customer  $r _ { k }$  and facility  $i \in \mathcal { Z }$ </td></tr><tr><td> $\beta _ { D }$ </td><td>Distance utility coefficient</td></tr><tr><td> $\varepsilon _ { i } ^ { k }$ </td><td>Stochastic utility that request in decision epoch k associates with service  $i \in \mathcal { Z }$ </td></tr><tr><td> $\hat { u } _ { o } ^ { - } , \hat { u } _ { o } ^ { + }$ </td><td>Conditional expected utility for base and premium outcome of order o</td></tr><tr><td> $I = | \mathcal { I } |$ </td><td>Number of facilities</td></tr><tr><td> $\hat { V } ^ { c } ( S _ { k j } , \omega )$ </td><td>Approximated post-choice state value in utility scenario ω</td></tr><tr><td> $\pi _ { \alpha }$ </td><td>Policy for given allocation scheme α</td></tr><tr><td> $h _ { o }$ </td><td>Time horizon of order o</td></tr><tr><td> $\mathcal { U } ( h )$ </td><td>Induction statement</td></tr><tr><td> $l \in \mathcal L$ </td><td>Lookahead scenario</td></tr><tr><td> $\hat { V } _ { l } ^ { - }$ </td><td>Approximated post-choice state value without hypothetical order  $O _ { k j }$  in lookahead scenario l</td></tr><tr><td> $\hat { V } _ { j \omega l } ^ { + }$ </td><td>Approximated post-choice state value with hypothetical order  $O _ { k j }$  in utility scenario ω and lookahead scenario l</td></tr><tr><td> $\Delta _ { j \omega l }$ </td><td>Incremental value of service option  $j \in \mathcal { I }$  in utility scenario ω and looka-</td></tr><tr><td> $\Delta _ { j \omega }$ </td><td>head scenario l Average incremental value of service option  $j \in \mathcal I$  in utility scenario ω</td></tr><tr><td> $\hat { V } ^ { - }$ </td><td>Average approximated post-choice state value without hypothetical or-</td></tr><tr><td> $\hat { V } _ { j \omega } ^ { + }$ </td><td>der  $O _ { k j }$  Average approximated post-choice state value with hypothetical order  $O _ { k j }$ </td></tr><tr><td> $\mathcal { P } ( S _ { k } )$ </td><td>in utility scenario ω MIP solution space</td></tr><tr><td> $d _ { j l } ^ { * }$ </td><td>Displacement due to hypothetical order  $O _ { k j }$  in lookahead scenario l </td></tr></table>

## Appendix B Uniqueness of Fully Transparent Probability Displays Under Non-Anticipatory Allocation

Before establishing the property, we note that non-anticipatory allocation induces a one-directional dependence across allocation times $t \in { \mathcal { T } } ^ { A }$ : A pending order can afect the allocation outcome of another order only if it is allocated earlier or as part of the same allocation decision. Conversely, pending orders allocated strictly later do not afect earlier allocation decisions. Moreover, allocation generally depends on the realized customer choices, but not directly on the probabilities previously displayed to individual customers.

Property: Consider a fixed, fully specified, and non-anticipatory allocation scheme α. For every pre-decision state $S _ { k } \in \mathcal { S } : t _ { k } \in \mathcal { T } ^ { A }$ , the allocation scheme α uniquely determines the joint allocation outcome for all orders $o \in { \mathcal { O } } _ { k } ^ { A }$ independent of pending orders $o ^ { \prime } \in \mathcal { O } _ { k } : t _ { o ^ { \prime } } ^ { A } > t _ { k }$ . Then, conditional on α, the truthful probabilities and thus the probabilities displayed under full transparency $\tau = 0$ are unique for every pre-decision state $S \in S ^ { p }$ . Therefore, it sufices to identify the single policy feasible under the transparency constraints to find the optimal policy $\pi _ { \alpha } ^ { * }$ given α.

Proof. For an order o arising at decision epoch k, define its time horizon in the system by $h _ { o } = t _ { o } ^ { A } - t _ { k }$ . Because (i) the points in time $t _ { k }$ are discrete, (ii) $t _ { o } ^ { A }$ is finite, and (iii) probability display and allocation occur at distinct decision epochs, it holds that $h _ { o } \in \mathbb { N }$

For every $h \in \mathbb { N } .$ , let $\mathcal { U } ( h )$ denote the following statement: For every probability display predecision state $S _ { k } \in S ^ { p }$ and hypothetical order $o _ { k j }$ arising if request $r _ { k }$ chooses service option $j \in$ $\mathcal { I } ^ { P }$ with $h _ { o _ { k j } } ~ = ~ h$ , the truthful allocation probabilities $\psi _ { o _ { k j } } ^ { \pi } ( S _ { k j } ^ { c } )$ for $o _ { k j }$ exist and are uniquely determined. Consequently, under full transparency, the probability-display vector $P ^ { \pi _ { \alpha } } ( S _ { k } )$ exists and is unique. Note that the allocation time $t _ { r _ { k } } ^ { A }$ is an attribute of the request such that all service options $j \in \mathcal { I }$ lead to the same horizon $h _ { o _ { k j } }$ . We prove $\mathcal { U } ( h )$ for every $h \in \mathbb { N }$ by induction.

For the base case, consider $h = 1$ , meaning that the hypothetical order $o _ { k j }$ arising in k is allocated at the subsequent decision epoch $k + 1$ . By the prescribed ordering of events, no additional probabil-$\mathrm { i t y }$ display decision occurs between the creation and allocation of $o _ { k j }$ . Hence, every order that can afect its allocation is already represented in post-choice state $S _ { k j } ^ { c }$ , and the exogenous-information distribution $\mathbb { P } ( W _ { k + 1 } ^ { s } | S _ { k j } ^ { c } )$ together with the transition function $S _ { k + 1 } = S ^ { M W ^ { s } } ( S _ { k j } ^ { c } , W _ { k + 1 } ^ { s } )$ uniquely determine the distribution of the next pre-decision state $S _ { k + 1 }$ . Conditional on each realization of the latter, the fully specified allocation scheme α determines a unique joint allocation outcome for all orders $o \in \mathcal { O } _ { k + 1 } ^ { A }$ , which includes $o _ { k j }$ . Therefore, the truthful allocation probabilities of $o _ { k j }$ for every $j \in \mathcal { I } ^ { P }$ exist and are unique. Under full transparency, the displayed probabilities must coincide with the truthful probabilities, establishing $\mathcal { U } ( 1 )$

For the induction step, fix an arbitrary $h \geq 2$ and suppose that $\mathcal { U } ( h ^ { \prime } )$ holds for every $h ^ { \prime } < h$ Consider an arbitrary pre-decision state $S _ { k } \in S ^ { p }$ with hypothetical order $o _ { k j }$ satisfying $t _ { o _ { k j } } ^ { A } - t _ { k } = h$ Additional requests may arise before $t _ { o _ { k j } } ^ { A }$ and generate orders allocated before or jointly with $o _ { k j }$ Consider any such order $o ^ { \prime }$ arising at a later decision epoch $k ^ { \prime }$ with $t _ { k } < t _ { k ^ { \prime } } < t _ { o _ { k j } } ^ { A }$ . There are two relevant cases to distinguish.

• Case 1: $t _ { o ^ { \prime } } ^ { A } \leq t _ { o _ { k . } } ^ { A }$ . Because α is non-anticipatory, $o ^ { \prime }$ can afect the allocation outcome of $o _ { k j }$ only if $t _ { o ^ { \prime } } ^ { A } \leq t _ { o _ { k . } } ^ { A }$ . Its horizon $h _ { o ^ { \prime } }$ satisfies

$$
h _ { o ^ { \prime } } = t _ { o ^ { \prime } } ^ { A } - t _ { k ^ { \prime } } \leq t _ { o _ { k j } } ^ { A } - t _ { k ^ { \prime } } < t _ { o _ { k j } } ^ { A } - t _ { k } = h
$$

Since probability display and allocation occur at distinct decision epochs, $1 \le h _ { o ^ { \prime } } < h$ . Hence, every future order capable of afecting the allocation of $o _ { k j }$ has a horizon in $\{ 1 , \ldots , h - 1 \}$ and is covered by the induction hypothesis. The truthful allocation probabilities, and therefore the fully transparent display vector at epoch $k ^ { \prime }$ , are thus uniquely determined. Since the customerchoice model $\mathbb { P } ( W _ { k ^ { \prime } + 1 } ^ { c } | S _ { k ^ { \prime } } , P ^ { \pi _ { \alpha } } ( S _ { k ^ { \prime } } ) )$ uniquely maps the state and probability display vector into a choice probability distribution, the conditional distribution of the customer’s choice is unique.

• Case 2: $t _ { o ^ { \prime } } ^ { A } > t _ { o _ { k } j } ^ { A }$ . The order $o ^ { \prime }$ is allocated strictly after order $o _ { k j }$ . Following its creation, $o ^ { \prime }$ is included in the updated set of pending orders $O _ { k ^ { \prime } + 1 }$ , but it does not occupy capacity before its allocation. Moreover, by non-anticipativity of $\alpha ,$ neither the existence nor the characteristics of $o ^ { \prime }$ afect any allocation decision made at or before $t _ { o _ { k j } } ^ { A }$ . Additionally, the presence of $o ^ { \prime }$ does not influence the conditional distribution of the exogenous information relevant to the allocation of $o _ { k j }$ , including capacity availabili $\operatorname { t y } ,$ request arrivals, and customer choices that may afect allocations up to $t _ { o _ { k j } } ^ { A }$ . Consequently, conditional on any realization of the process up to $t _ { o _ { k j } } ^ { A } .$ , the allocation outcome of $o _ { k j }$ is invariant to $o ^ { \prime } .$ . Hence, when constructing the conditional distribution of the allocation state for $o _ { k j }$ , the customer choice and subsequent state information associated exclusively with $o ^ { \prime }$ can be marginalized out. Orders allocated strictly after $o _ { k j }$ therefore do not afect the truthful allocation probability of $o _ { k j }$

The finite composition of state transitions, which follow unique conditional distributions, together uniquely determines the probability distribution of the allocation state for order $o _ { k j }$ and the associated set of orders to be allocated. Conditional on each realization, scheme α uniquely determines the joint allocation outcome or its probability distribution. Hence, the truthful allocation probability $\psi _ { o _ { k j } } ^ { \pi _ { \alpha } } ( S _ { k j } ^ { c } )$ exists and has a unique value. Therefore, the probability display vector in pre-decision state $S _ { k }$ is unique, and $\mathcal { U } ( h )$ holds.

By induction, $\mathcal { U } ( h )$ holds for every $h \in \mathbb { N }$ . Since $S _ { k }$ and $j$ were arbitrary, the truthful allocation probabilities exist and are uniquely determined in every post-choice state. Consequently, under $\tau = 0$ , the probability display decision is unique in every pre-decision state. Therefore, the fulltransparency constraints admit a unique policy conditional on $\alpha ,$ which is the optimal policy $\pi _ { \alpha } ^ { * }$ given α.

## Appendix C Policies

We provide details on our lookahead-model-based approximation for the post-choice state values, the features for predicting truthful probabilities, and the hyperparameters used in our computational study.

## C.1 Lookahead Model

With our lookahead model, we approximate the post-choice state values $\hat { V } ^ { c } ( S _ { k j } ^ { c } )$ in (7) using a set of sampled lookahead scenarios $l \in \mathcal L$ that represent the possible future evolution of the system. More precisely, the lookahead scenarios extend until the end of the current day $( t _ { k } ^ { d } , T + 1 )$ and comprise sampled pickup times for parcels currently occupying locker capacity, drawn from their conditional distributions, as well as future requests and their utilities. In the following, we omit the dependence on $S _ { k j } ^ { c }$ for notational convenience.

As a reference, we first compute the value $\hat { V } _ { l } ^ { - }$ of the system without hypothetical order $o _ { k j }$ in each lookahead scenario l. For each service option $j \in \mathcal { I }$ and utility scenario $\omega \in \Omega$ , we then compare it to the corresponding value $\hat { V } _ { j \omega l } ^ { + }$ if $o _ { k j }$ is included. This value depends on ω because it contains the current request’s utility $\tilde { u } _ { i _ { j } ^ { - } \omega }$ or $\tilde { u } _ { i _ { j } ^ { + } \omega }$ sampled for the allocated service outcome (Section 3.2). To reduce the computational burden, we estimate $\hat { V } _ { l } ^ { - }$ <sup>−</sup> once per lookahead scenario l and then heuristically determine $\hat { V } _ { j \omega l } ^ { + }$ through incremental adaptations.

For the baseline $\hat { V } _ { l } ^ { - }$ , we formulate a full-information assignment problem without the hypothetical order. Specifically, the objective coeficients correspond to the conditional expected utilities of currently pending orders and the sampled utilities of future orders, while the sampled pickup times determine the available locker capacity. We prevent infeasible assignments of existing pending orders to service outcomes not contained in their chosen service options by setting the objective coeficients to −∞. We solve the assignment problem to optimality with a modified Jonker-Volgenant algorithm (Crouse 2016), implemented in SciPy 1.16.

We define the incremental value of service option $j$ in utility scenario $\omega$ and lookahead scenario l as $\Delta _ { j \omega l } = \hat { V } _ { j \omega l } ^ { + } - \hat { V } _ { l } ^ { - }$ . Thus, $\Delta _ { j \omega l }$ captures the additional utility generated by the hypothetical order as well as the potential displacement of orders. Averaging the incremental value for service option $j$ and utility scenario ω across lookahead scenarios yields $\begin{array} { r } { \Delta _ { j \omega } = \frac { 1 } { | \mathcal { L } | } \sum _ { l \in \mathcal { L } } \Delta _ { j \omega l } } \end{array}$ . With the average baseline value $\begin{array} { r } { \hat { V } ^ { - } = \frac { 1 } { | \mathcal { L } | } \sum _ { l \in \mathcal { L } } \hat { V } _ { l } ^ { - } } \end{array}$ , we obtain $\hat { V } _ { j \omega } ^ { + } = \hat { V } ^ { - } + \Delta _ { j \omega }$ . Recall that the immediate reward for probability display in our OOHD setting equals $R ( S _ { k } , P _ { k } , j ) = 0$ . Substituting $\hat { V } _ { j \omega } ^ { + } = \hat { V } ^ { - } + \Delta _ { j \omega }$ into the MIP objective (7) and using $\begin{array} { r } { \sum _ { j \in \mathcal { T } } c _ { j \omega } = 1 \ \forall \omega \in \Omega } \end{array}$ yields

$$
\sum _ { j \in \mathcal { I } } \frac { 1 } { | \Omega | } \sum _ { \omega \in \Omega } c _ { j \omega } \hat { V } _ { j \omega } ^ { + } = \hat { V } ^ { - } + \sum _ { j \in \mathcal { I } } \frac { 1 } { | \Omega | } \sum _ { \omega \in \Omega } c _ { j \omega } \Delta _ { j \omega }\tag{18}
$$

Because $\hat { V } ^ { \cdot }$ <sup>−</sup> is independent of the displayed probabilities $P _ { k }$ , it can be omitted from (18). Letting $\mathcal { P } ( S _ { k } )$ denote the solution space formalized by Constraints (8)-(16), we obtain the incrementalvalue-based formulation for the MIP:

$$
\operatorname* { m a x } _ { P _ { k } \in \mathcal { P } ( S _ { k } ) } \left\{ \frac { 1 } { \left. \Omega \right. } \sum _ { \omega \in \Omega } \sum _ { j \in \mathcal { I } } c _ { j \omega } \Delta _ { j \omega } \right\}\tag{19}
$$

Starting from the optimal baseline solution computed for $\hat { V } _ { l } ^ { - }$ , we heuristically derive the incre mental value $\Delta _ { j \omega l }$ based on the change in the objective value caused by including hypothetical order $o _ { k j }$ and allowing up to one reassignment. Currently pending orders $o \in \mathcal { O } _ { k }$ may be reassigned only to the base outcomes associated with their chosen service options, whereas sampled future orders may be reassigned to any facility with suficient capacity.

If the hypothetical order corresponds to a probabilistic service option $j \in \mathcal { I } ^ { P }$ , three cases may arise:

1. If locker $i _ { j } ^ { + }$ has available capacity in lookahead scenario $l ,$ the hypothetical order receives the premium outcome without displacing another order, and the incremental value equals $\Delta _ { j \omega l } = \widetilde { u } _ { i _ { j } ^ { + } \omega } .$

2. If locker $i _ { j } ^ { + }$ is fully occupied in the baseline solution for l but suficient capacity can be released by reassigning another order, we identify the feasible reassignment that causes the smallest loss in the objective value. Let $d _ { j l } ^ { * } \leq 0$ denote the corresponding objective-value change. The hypothetical order can either receive the premium outcome $i _ { j } ^ { + }$ at the cost of $d _ { j l } ^ { * }$ due to the reassignment or the base outcome $i _ { j } ^ { - }$ , depending on which alternative results in a higher value: $\Delta _ { j \omega l } = \operatorname* { m a x } \left\{ \widetilde { u } _ { i _ { j } ^ { + } \omega } + d _ { j l } ^ { * } , \widetilde { u } _ { i _ { j } ^ { - } \omega } \right\}$

Table 8. Feature overview.
<table><tr><td>Feature</td><td>Min.</td><td>Max.</td><td>Interpretation</td></tr><tr><td> $\overline { { t ^ { A } } }$ </td><td>1</td><td>T</td><td>Number of within-day time periods until allocation  $T + 1 - t _ { k }$ </td></tr><tr><td> $O ^ { + }$ </td><td>0</td><td> $_ T$ </td><td>Number of pending orders for service option j with a higher priority for allocation than order o</td></tr><tr><td> $C ^ { d }$ </td><td>0</td><td> $Q _ { i _ { j } ^ { + } }$ </td><td>Number of occupied capacity units in locker  $i _ { j } ^ { + }$  with elapsed storage duration of d days</td></tr><tr><td> $x ^ { l } , y ^ { l }$ </td><td> $\underline { { x } } , \underline { { y } }$ </td><td> $x , y$ </td><td>Coordinates of locker  $i _ { j } ^ { + }$ </td></tr><tr><td> $s$ </td><td> $\underline { { s _ { i } } }$ </td><td> $\overline { { s _ { i } } }$ </td><td>Expected demand share of locker  $i _ { j } ^ { + }$ </td></tr><tr><td> $\Delta v$ </td><td> $\underline { { \Delta v } }$ </td><td> $\overline { { \Delta v } }$ </td><td>Deterministic utility component difference  $v _ { i _ { j } ^ { + } } - v _ { i _ { j } ^ { - } }$  for order o</td></tr><tr><td> $x ^ { c } , y ^ { c }$ </td><td> $\underline { { x ^ { c } } } , \underline { { y } } ^ { c }$ </td><td> $x ^ { \bar { c } } , y ^ { \bar { c } }$ </td><td>Coordinates of customer location</td></tr><tr><td> $\tilde { q } ^ { B }$ </td><td>0</td><td> $Q _ { i _ { j } ^ { + } }$ </td><td>Expected available capacity in locker  $i _ { j } ^ { + }$  at the time of allocation  $( t _ { k } ^ { d } , T + 1 )$ </td></tr><tr><td> $\hat { \psi } ^ { B }$ </td><td>0</td><td> $^ 1$ </td><td>Baseline probability estimate</td></tr></table>

3. If locker $i _ { j } ^ { + }$ is full and no feasible reassignment exists, the hypothetical order receives the base outcome $i _ { j } ^ { - }$ , leading to an incremental value of $\Delta _ { j \omega l } = \widetilde { u } _ { i _ { j } ^ { - } \omega } .$

In case of a deterministic option $j \in \mathcal { I } ^ { D }$ , the hypothetical order receives the associated outcome, yielding $\Delta _ { j \omega l } = \widetilde { u } _ { i _ { j } \omega }$

Finally, we take the average across lookahead scenarios to obtain $\begin{array} { r } { \Delta _ { j \omega } = \frac { 1 } { | \mathcal { L } | } \sum _ { l \in \mathcal { L } } \Delta _ { j \omega l } } \end{array}$ and solve the incremental-value-based MIP (19). Without this incremental strategy, we would have to solve a separate assignment problem for every combination of service option, utility scenario, and lookahead scenario, which conflicts with the requirement that customers expect to view the displayed probabilities in real time.

## C.2 Features

Table 8 summarizes our features $\phi ( S _ { k } ^ { c } , o )$ for approximating the truthful probabilities $\hat { \psi } _ { \pmb { \theta } }$ . Beyond the information directly available from the state variables and global instance parameters, we estimate the expected demand share $s _ { i }$ for each locker $i \in \mathcal { I } ^ { + }$ ex ante by simulating 100 days with probability display NO (Section 4.1.3) and computing the percentage of customers choosing the corresponding service option. To construct the lookahead-based features $\tilde { q } ^ { B }$ and $\hat { \psi } ^ { B }$ , we use the same simulated order streams to generate pools of future orders for each within-day period $t _ { k } ^ { w }$ These pools are independent of the remaining state information and can therefore be generated and stored in advance. For a given post-choice state and order $^ { O , }$ we use the corresponding order pool to estimate the average number of future arriving orders with higher allocation priority than $^ { O , }$ yielding a baseline estimate for the probability that o receives the premium outcome. Additionally, we track the locker capacity expected to be available at the time of allocation.

## C.3 Hyperparameters

We use the following hyperparameters across all experimental settings:

To obtain the approximation $\hat { \psi } _ { \pmb { \theta } }$ , we simulate 5000 days of training. Starting from an initial value of 1, the probability for exploration decays exponentially to 0.05 over the first 4000 days and remains constant afterward. The neural network is implemented using PyTorch 2.5.1 and comprises 2 hidden layers with 32 neurons each. The hidden layers use rectified linear unit (ReLU) activation functions, while the output layer employs a sigmoid activation function. The replay memory E stores experiences for all pending orders corresponding to the 250 most recently encountered post-choice states. For each network update, we randomly sample 4 post-choice states represented in E and group all the associated experiences for the individual pending orders into mini-batches with at most 64 observations. We update the network parameters θ using the AdamW optimizer (Loshchilov and Hutter 2019) with a learning rate of 0.0005 and the gradient norm clipped to 1.

For probability display decisions, the MIP is based on $| \Omega | = 5 0$ utility scenarios. We replace the suficiently large constant M with a scenario-specific value $\begin{array} { r } { M _ { \omega } = \operatorname* { m a x } _ { i \in \mathcal { T } } \{ \tilde { u } _ { i \omega } \} - \operatorname* { m i n } _ { i \in \mathcal { T } } \{ \tilde { u } _ { i \omega } \} } \end{array}$ and set $M ^ { \prime } = \operatorname* { m i n } \{ 0 . 0 0 1 , \hat { \psi } _ { 1 } ^ { k } , \dots , \hat { \psi } _ { | \mathcal { I } | } ^ { k } \}$ . For LA, we approximate the post-choice state values with $| \mathcal { L } | = 1 0$ lookahead scenarios.

## References

Akkerman, F., Dieter, P., and Mes, M. (2025). Learning dynamic selection and pricing of out-of-home deliveries. Transportation Science, 59(2):250–278.

Allon, G. and Bassamboo, A. (2011). Buying from the babbling retailer? The impact of availability information on customer behavior. Management Science, 57(4):713–726.

Bordoloi, S., Fitzsimmons, J. A., and Fitzsimmons, M. J. (2023). Service Management: Operations, Strategy, Information Technology. McGraw Hill, New York, NY, 10<sup>th</sup> edition.

Chen, N., Elmachtoub, A. N., Hamilton, M. L., and Lei, X. (2021). Loot box pricing and design. Management Science, 67(8):4809–4825.

Chen, T., Yang, F., and Guo, X. (2024). Probabilistic selling in a distribution channel with vertically diferentiated products. Journal of the Operational Research Society, 75(6):1076–1091.

Cheung, W. C. and Simchi-Levi, D. (2016). Eficiency and performance guarantees for choice-based network revenue management problems with flexible products. Working paper, Massachusetts Institute of Technology.

Crouse, D. F. (2016). On implementing 2D rectangular assignment algorithms. IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679–1696.

Cui, R. and Shin, H. (2018). Sharing aggregate inventory information with customers: Strategic cross-selling and shortage reduction. Management Science, 64(1):381–400.

Drakopoulos, K., Jain, S., and Randhawa, R. (2021). Persuading customers to buy early: The value of personalized information provisioning. Management Science, 67(2):828–853.

Fay, S. and Xie, J. (2008). Probabilistic goods: A creative way of selling products and services. Marketing Science, 27(4):674–690.

Gallego, G. and Phillips, R. (2004). Revenue management of flexible products. Manufacturing & Service Operations Management, 6(4):321–337.

Gehring, H. and Homberger, J. (2002). Parallelization of a two-phase metaheuristic for routing problems with time windows. Journal of Heuristics, 8(3):251–276.

G¨onsch, J. (2020). How much to tell your customer? – A survey of three perspectives on selling strategies with incompletely specified products. European Journal of Operational Research, 280(3):793–817.

G¨onsch, J., Koch, S., and Steinhardt, C. (2014). Revenue management with flexible products: The value of flexibility and its incorporation into DLP-based approaches. International Journal of Production Economics, 153:280–294.

Guo, X., Bian, J., Wu, P., Shi, V., and Chen, H. (2023). Probabilistic product design with regret-anticipated consumers. International Journal of Production Economics, 263:108917.

Hildebrandt, F. D. and Ulmer, M. W. (2022). Supervised learning for arrival time estimations in restaurant meal delivery. Transportation Science, 56(4):1058–1084.

Ibrahim, R. (2018). Sharing delay information in service systems: A literature survey. Queueing Systems, 89(1):49–79.

Janinhof, L., Klein, R., Sailer, D., and Schoppa, J. M. (2024). Out-of-home delivery in last-mile logistics: A review. Computers & Operations Research, 168:106686.

Koch, S., G¨onsch, J., and Steinhardt, C. (2017). Dynamic programming decomposition for choice-based revenue management with flexible products. Transportation Science, 51(4):1046–1062.

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations.

Mahadevan, S. (1996). Average reward reinforcement learning: Foundations, algorithms, and empirical results. Machine Learning, 22:159–195.

Mitter, M., K¨unz, L., and G¨onsch, J. (2026). Opaque selling with arbitrary customer preferences. Working paper, University of Duisburg-Essen.

Paneque, M. P., Bierlaire, M., Gendron, B., and Azadeh, S. S. (2021). Integrating advanced discrete choice models in mixed integer linear optimization. Transportation Research Part B: Methodological, 146:26– 49.

Petrick, A., G¨onsch, J., Steinhardt, C., and Klein, R. (2010). Dynamic control mechanisms for revenue management with flexible products. Computers & Operations Research, 37(11):2027–2039.

Petrick, A., Steinhardt, C., G¨onsch, J., and Klein, R. (2012). Using flexible products to cope with demand uncertainty in revenue management. OR Spectrum, 34(1):215–242.

Powell, W. B. (2022). Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions. John Wiley & Sons, Hoboken, NJ.

Ren, H. and Huang, T. (2022). Opaque selling and inventory management in vertically diferentiated markets. Manufacturing & Service Operations Management, 24(5):2543–2557.

Sailer, D., Klein, R., and Steinhardt, C. (2026). Dynamic demand management for parcel lockers. Transportation Science. Articles in Advance.

Simhon, E. and Starobinski, D. (2018). On the impact of information disclosure on advance reservations: A game-theoretic view. European Journal of Operational Research, 267(3):1075–1088.

Strauss, A., G¨ulpınar, N., and Zheng, Y. (2021). Dynamic pricing of flexible time slots for attended home delivery. European Journal of Operational Research, 294(3):1022–1041.

Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction. The MIT Press, Cambridge, MA, 2<sup>nd</sup> edition.

Train, K. E. (2009). Discrete Choice Methods with Simulation. Cambridge University Press, Cambridge, 2<sup>nd</sup> edition.

Yang, Z., Jang, H., and Pan, X. A. (2026). Probabilistic selling for vertically diferentiated products in a decentralized channel. Manufacturing & Service Operations Management. Articles in Advance.

Zhang, Z., Joseph, K., and Subramaniam, R. (2015). Probabilistic selling in quality-diferentiated markets. Management Science, 61(8):1959–1977.

Zhu, W. and Topaloglu, H. (2024). Performance guarantees for network revenue management with flexible products. Manufacturing & Service Operations Management, 26(1):252–270.