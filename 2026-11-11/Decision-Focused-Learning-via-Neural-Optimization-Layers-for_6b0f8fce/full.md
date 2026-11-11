# Decision-Focused Learning via Neural Optimization Layers for Battery-Assisted Grid-Import Smoothing in AI Data Centers

Xinda Zheng<sup>1</sup>, Canchen Jiang<sup>1\*</sup>, Frits de Nijs<sup>1</sup>, Edward Lam<sup>1</sup>, Markus Wagner<sup>1</sup>

<sup>1</sup> Department of Data Science and AI, Faculty of Information Technology, Monash University, Australia

Abstract—Rapid AI data-center power fluctuations challenge grid integration. Sustained grid-import smoothing requires allocating limited battery power and energy under uncertain demand. We formulate risk-aware BESS scheduling that coordinates advance reserve commitments and demand-responsive dispatch, accounting for operating costs, grid-ramp penalties, and their conditional value-at-risk. We develop battery scheduling decision-focused learning (BS-DFL) with a constraint-aware neural optimization layer trained directly from this objective without optimal-solution labels. The trained layer is then frozen to guide demandpredictor learning through scheduling costs and ramping risk, enabling solver-free rolling-horizon scheduling. In a 10,000-minute simulation using loads derived from measured AI workloads, BS-DFL reduces mean absolute minute-to-minute grid-import changes by 71.70% and total scheduling cost by 18.01% relative to an MSE-trained predictor coupled with a mixed-integer linear programming solver, with an average decision time of 18.77 ms.

Index Terms—AI data centers, Battery energy storage systems, Grid-import ramp mitigation, Decision-focused learning.

## I. INTRODUCTION

The rapid growth of AI data centers introduces highly variable electricity loads, creating challenges for grid-compatible operation and power-ramp management [1]. Behind-the-meter battery energy storage systems (BESSs) can mitigate rapid load fluctuations without modifying computational workload schedules [2]. However, sustained grid-import smoothing is constrained by battery power and energy limits, requiring careful scheduling of charging and discharging flexibility. Predictive scheduling is therefore needed to mitigate current grid-import fluctuations while preserving sufficient power and energy headroom for future uncertain demand.

Existing studies have developed battery-assisted ramp-rate control for AI data centers [3]. At the scheduling level, coordinated workload shifting, battery dispatch, and reserve allocation have been investigated to balance operating costs and grid-ramping risk [4]. To guide advance battery scheduling, neural load forecasts have been combined with robust model predictive control for peak shaving [5], while probabilistic load forecasts support capacity allocation between peak shaving and frequency containment reserves [6]. Nevertheless, prediction errors of similar magnitude can lead to substantially different scheduling costs and grid-ramp violations, depending on the battery’s available power and energy flexibility.

Conventional forecasting approaches typically minimize statistical prediction losses, such as mean squared error, without explicitly considering their impact on downstream battery scheduling decisions. This motivates decision-focused learning (DFL), which incorporates downstream scheduling costs into predictor training [7]. DFL has been applied to BESS reserve-market participation [8], PV-battery scheduling [9], and AI data-center workload scheduling under distributiongrid constraints [10]. These methods rely on differentiable convex optimization layers, limiting their direct application to scheduling with discrete decisions. To accommodate such decisions, surrogate-loss and black-box differentiation methods have been applied to battery scheduling [11]. Other approaches include perturbation-based learning for energy storage arbitrage [12] and supervised neural optimizer for battery peak shaving and energy cost reduction [13]. However, these approaches often rely on optimization solvers during training or deployment, which can limit training scalability for large or complex problems and the responsiveness of real-time battery scheduling.

More recently, [14] trains a neural optimization layer (NOL) directly from optimization objectives and uses it to guide predictor learning, enabling solver-free training and deployment. This framework offers a way to align prediction with downstream decision while supporting real-time decision making. Nevertheless, the existing NOL architecture is not directly applicable to BESS scheduling, which involves intertemporal energy constraints, charging/discharging exclusivity, and uncertain demand. To address these challenges, we propose a battery scheduling decision-focused learning framework (BS-DFL), which uses a BESS-scheduling-specific NOL (BS-NOL) to train demand predictors for improved battery scheduling and generate feasible decisions for the mixed-integer formulation without optimization solvers during training and rolling-horizon deployment. The main contributions are summarized as follows:

• We formulate a risk-aware BESS scheduling framework for AI data centers that coordinates advance reserve commitments and demand-responsive dispatch, accounting for operating costs, grid-import ramp penalties, and their tail risks through CVaR.

• We develop BS-DFL with BS-NOL that combines scenario-aware encoding and sequential constraint-aware decoding to coordinate reserve commitments and demandresponsive dispatch. The layer is trained directly using the scheduling objective without optimal solution labels, then frozen to guide demand-predictor learning through downstream scheduling costs and ramping risks.

• Using facility load profiles generated from measured AI workloads, we demonstrate that BS-DFL reduces mean absolute grid-import changes by 71.70% and total operating costs by 18.01% relative to an MSE-trained predictor coupled with the Gurobi solver. It achieves a total cost within 4.39% of the perfect-information rolling-horizon reference while computing decisions in milliseconds.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

We consider an AI data center with a behind-the-meter BESS connected to the grid at a POI. The BESS is operated using a rolling-horizon two-stage stochastic scheduling framework to coordinate economic operation and mitigation of rapid POI import variations.

The horizon $\begin{array} { r c l } { \mathcal { T } } & { = } & { \{ 1 , \dots , T \} } \end{array}$ consists of one-minute intervals of duration $\Delta t .$ A finite scenario set $s$ represents demand uncertainty, with interval-average demand $\widehat { P } _ { t , s } ^ { \mathrm { D C } }$ and probability $\rho _ { s } \geq 0 ;$ , where $\textstyle \sum _ { s \in { \mathcal { S } } } \rho _ { s } = 1$ . Scenarios combine demand forecasts with historical variations (Section III).

## A. Stage 1: BESS Scheduling and Reserve Allocation

Let $b _ { t }$ denote the baseline net BESS power, with positive and negative values indicating discharging and charging, respectively. The upward and downward reserve power capacities are denoted by ${ q } _ { t } ^ { + }$ and $q _ { t } ^ { - }$ , satisfying

$$
- P _ { \operatorname* { m a x } } ^ { \mathrm { c h } } \leq b _ { t } \leq P _ { \operatorname* { m a x } } ^ { \mathrm { d i s } } ,\tag{1a}
$$

$$
0 \leq q _ { t } ^ { + } \leq P _ { \operatorname* { m a x } } ^ { \mathrm { d i s } } - b _ { t } ,\tag{1b}
$$

$$
0 \leq q _ { t } ^ { - } \leq P _ { \operatorname* { m a x } } ^ { \mathrm { c h } } + b _ { t } ,\tag{1c}
$$

where $P _ { \mathrm { m a x } } ^ { \mathrm { c h } }$ and $P _ { \mathrm { m a x } } ^ { \mathrm { d i s } }$ denote the BESS rated charging and discharging power. The corresponding extreme net outputs are

$$
v _ { t } ^ { + } = b _ { t } + q _ { t } ^ { + } , v _ { t } ^ { - } = b _ { t } - q _ { t } ^ { - } .\tag{2}
$$

To ensure that idle operation remains feasible within the committed power envelope, we impose

$$
v _ { t } ^ { - } \leq 0 \leq v _ { t } ^ { + } .\tag{3}
$$

## B. Stage 2: Scenario-Dependent BESS Operation

For scenario $s ,$ let $a _ { t , s }$ denote the corrective power activation. The net BESS power $p _ { t , s } ^ { \mathrm { B } }$ and its charging/discharging components satisfy

$$
p _ { t , s } ^ { \mathrm { d i s } } - p _ { t , s } ^ { \mathrm { c h } } = b _ { t } + a _ { t , s } ,\tag{4a}
$$

$$
- q _ { t } ^ { - } \leq a _ { t , s } \leq q _ { t } ^ { + } ,\tag{4b}
$$

$$
0 \leq p _ { t , s } ^ { \mathrm { d i s } } \leq z _ { t , s } P _ { \operatorname* { m a x } } ^ { \mathrm { d i s } } ,\tag{4c}
$$

$$
0 \leq p _ { t , s } ^ { \mathrm { c h } } \leq ( 1 - z _ { t , s } ) P _ { \operatorname* { m a x } } ^ { \mathrm { c h } } ,\tag{4d}
$$

where $p _ { t , s } ^ { \mathrm { c h } }$ and $p _ { t , s } ^ { \mathrm { d i s } }$ are the actual charging and discharging powers, and the binary variable $z _ { t , s } ~ \in ~ \{ 0 , 1 \}$ prevents simultaneous charging and discharging at time t for scenario s.

The stored battery energy $E _ { t , s }$ evolves according to

$$
E _ { t , s } = E _ { t - 1 , s } + \eta _ { \mathrm { c h } } p _ { t , s } ^ { \mathrm { c h } } \Delta t - \frac { p _ { t , s } ^ { \mathrm { d i s } } \Delta t } { \eta _ { \mathrm { d i s } } } ,\tag{5a}
$$

$$
E _ { \mathrm { m i n } } \leq E _ { t , s } \leq E _ { \mathrm { m a x } } ,\tag{5b}
$$

where $\eta _ { \mathrm { c h } }$ and $\eta _ { \mathrm { d i s } }$ denote charging and discharging efficiencies. The initial energy is fixed at $E _ { 0 , s } = E _ { 0 }$

To sustain either committed power limit for one interval, the starting energy must provide sufficient stored energy and charging headroom:

$$
E _ { t - 1 , s } \geq E _ { \mathrm { m i n } } + \frac { \Delta t } { \eta _ { \mathrm { d i s } } } v _ { t } ^ { + } ,\tag{6a}
$$

$$
\begin{array} { r } { E _ { t - 1 , s } \leq E _ { \mathrm { m a x } } + \eta _ { \mathrm { c h } } \Delta t v _ { t } ^ { - } . } \end{array}\tag{6b}
$$

## C. POI Import and Ramping Constraints

With grid export prohibited, the projected POI import $g _ { t , s } \geq$ 0 under forecast scenario s is

$$
g _ { t , s } = \widehat { P } _ { t , s } ^ { \mathrm { D C } } + p _ { t , s } ^ { \mathrm { c h } } - p _ { t , s } ^ { \mathrm { d i s } } .\tag{7}
$$

To mitigate rapid fluctuations at the POI, upward and downward grid-power changes satisfy

$$
- ( R ^ { \downarrow } + \delta _ { t , s } ^ { - } ) \leq g _ { t , s } - g _ { t - 1 , s } \leq R ^ { \uparrow } + \delta _ { t , s } ^ { + } ,\tag{8}
$$

where $R ^ { \uparrow }$ and $R ^ { \downarrow }$ are the specified upward and downward ramp limits, and $\delta _ { t , s } ^ { + } , \delta _ { t , s } ^ { - } ~ \geq ~ 0$ measure the corresponding exceedances. Initially, $g _ { 0 , s } ~ = ~ g _ { 0 }$ . The accumulated rampviolation cost under scenario s is

$$
V _ { s } = \sum _ { t \in \tau } \left( \lambda ^ { + } \delta _ { t , s } ^ { + } + \lambda ^ { - } \delta _ { t , s } ^ { - } \right) ,\tag{9}
$$

where $\lambda ^ { + }$ and $\lambda ^ { - }$ represent penalties for upward and downward violations, respectively.

## D. Risk-Aware Stochastic Optimization

The objective is to minimize reserve allocation costs, expected operating costs, and the risk of severe POI ramp violations.

The reserve allocation cost is defined as $\begin{array} { r l } { C ^ { \mathrm { { r e s } } } } & { { } = } \end{array}$ $\textstyle \sum _ { t \in { \mathcal { T } } } ( c _ { t } ^ { + } q _ { t } ^ { + } + c _ { t } ^ { - } q _ { t } ^ { - } ) \Delta t$ , where $c _ { t } ^ { + }$ and $c _ { t } ^ { - }$ denote the incremental capacity-commitment cost coefficients for upward and downward reserves, respectively. These coefficients penalize excessive reserve allocation separately from the energy-arbitrage costs of BESS operation.

The scenario-dependent operating cost is $\begin{array} { r l } { C _ { s } ^ { \mathrm { o p } } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { t \in \mathcal { T } } \left[ \pi _ { t } g _ { t , s } + c ^ { \mathrm { d e g } } \left( p _ { t , s } ^ { \mathrm { c h } } + p _ { t , s } ^ { \mathrm { d i s } } \right) \right] \Delta t } \end{array}$ , where $\pi _ { t }$ denotes the electricity price and $c ^ { \mathrm { d e g } }$ is a throughput-based battery degra dation cost coefficient.

The terminal penalty $C _ { s } ^ { \mathrm { t e r m } } = c ^ { \mathrm { t e r m } } | E _ { T , s } - E ^ { \mathrm { t a r } } |$ penalizes deviations from the target energy at the horizon’s end.

Let $\mathbf { x } ^ { ( 1 ) } = \{ b _ { t } , q _ { t } ^ { + } , q _ { t } ^ { - } \} _ { t \in \mathcal { T } }$ denote the first-stage decisions and let $\mathbf { x } ^ { \mathrm { r e c } } = \{ a _ { t , s } , p _ { t , s } ^ { \mathrm { c h } } , p _ { t , s } ^ { \mathrm { d i s } } , E _ { t , s } , z _ { t , s } , g _ { t , s } , \delta _ { t , s } ^ { + } , \delta _ { t , s } ^ { - } \} _ { t \in \mathcal { T } , s \in \mathcal { S } }$ collect scenario-dependent operating variables, including the binary operating modes. Let $\mathcal { F }$ denote the feasible set defined by (1)-(8).

CVaR measures the tail risk of the scenario ramp cost $V _ { s }$ in (9). With auxiliary variables $\zeta \in \mathbb R$ and $\xi _ { s } \ge 0$ , the optimization problem is

$$
\begin{array} { r l } { \underset { \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { \mathrm { r e c } } , \boldsymbol { \zeta } , \boldsymbol { \xi } } { \operatorname* { m i n } } } & { C ^ { \mathrm { r e s } } + \displaystyle \sum _ { s \in S } \rho _ { s } ( C _ { s } ^ { \mathrm { o p } } + V _ { s } + C _ { s } ^ { \mathrm { t e r m } } ) } \\ & { +  \beta [ \boldsymbol { \zeta } + \frac { 1 } { 1 - \alpha } \underset { s \in S } { \sum } \rho _ { s } \boldsymbol { \xi } _ { s } ] } \\ { \mathrm { s . t . } } & { ( \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { \mathrm { r e c } } ) \in \mathcal { F } , \quad \boldsymbol { \xi } _ { s } \geq V _ { s } - \boldsymbol { \zeta } , \quad \forall s . } \end{array}\tag{10}
$$

where $\alpha \in ( 0 , 1 )$ denotes the CVaR confidence level and $\beta \geq 0$ controls the emphasis on tail risk.

## E. Real-Time Dispatch and Rolling-Horizon Implementation

At each decision epoch, only the first-interval baseline and reserves $( b _ { 1 } ^ { * } , q _ { 1 } ^ { + , * } , q _ { 1 } ^ { - , * } )$ from (10) are committed. Once updated demand measurements become available, BESS dispatch is re-optimized over the operational horizon using the current demand estimate and updated future forecasts. The dispatch optimization minimizes the operating, ramping, and terminal costs defined above, subject to the BESS and POI constraints in (4)-(8), with the first-interval baseline and reserves fixed at their committed values. Only the first-interval dispatch is implemented, while future decisions remain provisional. The battery state and POI import are updated, and the stochastic scheduling problem is re-solved for the next interval.

## III. DECISION-FOCUSED LEARNING FRAMEWORK

BS-DFL combines a neural demand predictor with the constraint-aware BS-NOL, as shown in Fig. 1.

![](images/3965becb215d2034fff1f754a55977e19e72ddb61a1bd1920f6740f623c7aa0c.jpg)  
Fig. 1. BS-DFL framework. Forecast-based scenarios guide baseline and reserve commitments. Battery responses adapt to observed demand and state within the committed range. The two decoders convert network outputs into feasible scheduling decisions.

Given an input feature vector h containing historical demand, node occupancy, and time features, a neural predictor $f _ { \theta }$ produces the positive demand forecast vector $\mu _ { \theta } = f _ { \theta } ( \mathbf { h } ) =$ $( \mu _ { \theta , 1 } , \ldots , \mu _ { \theta , T } ) ^ { \top }$ over the planning horizon. To represent uncertainty around this forecast, we select historical demand profiles from different training days with similar recent load levels and variability, node occupancy, and time of day. Let $H _ { t , s }$ denote the demand at interval t in the sth selected historical demand profile. The deviations $\begin{array} { r } { d _ { t , s } = H _ { t , s } - \frac { 1 } { | S | } \sum _ { j \in S } H _ { t , j } } \end{array}$ from the mean of the selected profiles at each interval are combined with the forecast to form demand scenarios $\widehat { P } _ { t , s } ^ { \mathrm { D C } }$ with probabilities $\rho _ { s } = 1 / | S |$

$$
\widehat { P } _ { t , s } ^ { \mathrm { D C } } = \mathrm { s o f t p l u s } \left( \mathrm { s o f t p l u s } ^ { - 1 } ( \mu _ { \theta , t } ) + d _ { t , s } \right) .\tag{11}
$$

This transformation keeps demand positive and returns the original forecast when $d _ { t , s } = 0 . \ \mathrm { B S - N O I }$ generates first-stage commitments $\mathbf { x } ^ { ( 1 ) }$ , defining the committed power range in (2), and scenario-dependent operating decisions ${ \bf x } ^ { \mathrm { r e c } }$ , including battery responses within that range as in (4).

## A. Battery-Scheduling Neural Optimization Layer

The encoder processes demand values with their time and scenario indices. Attention learns how to combine information across times and scenarios into a shared summary, which is combined with the initial battery state to guide reserve commitments and battery responses.

A linear commitment head maps the shared summary to $\gamma _ { t } ^ { - } , \gamma _ { t } ^ { + } , \gamma _ { t } ^ { \mathrm { b } } ~ \in ~ [ 0 , 1 ]$ . The commitment decoder uses these coefficients to select the lower power limit, upper power limit, and baseline. From the starting energy levels, it first computes power bounds that satisfy the power and reserve energy constraints in every scenario:

$$
\ell _ { t } = \operatorname* { m a x } \Biggl \{ - P _ { \mathrm { m a x } } ^ { \mathrm { c h } } , \frac { \operatorname* { m a x } _ { s } E _ { t - 1 , s } - E _ { \mathrm { m a x } } } { \eta _ { \mathrm { c h } } \Delta t } \Biggr \} ,\tag{12a}
$$

$$
u _ { t } = \operatorname* { m i n } \Biggl \{ P _ { \operatorname* { m a x } } ^ { \mathrm { d i s } } , \frac { \eta _ { \mathrm { d i s } } \bigl ( \operatorname* { m i n } _ { s } { E _ { t - 1 , s } - E _ { \mathrm { m i n } } } \bigr ) } { \Delta t } \Biggr \} .\tag{12b}
$$

Within these bounds, $\gamma _ { t } ^ { - }$ selects the charging limit $\begin{array} { r l } { \boldsymbol { v } _ { t } ^ { - } } & { { } = } \end{array}$ $\ell _ { t } ( 1 - \gamma _ { t } ^ { - } )$ , and $\gamma _ { t } ^ { + }$ selects the discharging limit $v _ { t } ^ { + } = u _ { t } \gamma _ { t } ^ { + }$ The third coefficient places the baseline within this range: $b _ { t } = v _ { t } ^ { - } + ( v _ { t } ^ { + } - v _ { t } ^ { - } ) \gamma _ { t } ^ { \mathrm { b } }$ . The reserve capacities follow as $q _ { t } ^ { + } ~ = ~ v _ { t } ^ { + } - b _ { t }$ and $q _ { t } ^ { - } ~ = ~ b _ { t } - v _ { t } ^ { - }$ . The committed range includes zero, and either power limit can be sustained for one interval in every scenario.

The response head uses a multilayer perceptron (MLP) to produce a scalar adjustment $\epsilon _ { t , s }$ from the shared summary, observed demand history, battery state, commitments, and interval index. The response decoder adjusts the power needed to maintain the previous grid import, $\widehat { P } _ { t , s } ^ { \mathrm { D C } } - { g } _ { t - 1 , s } ,$ , with $g _ { 0 , s } ~ = ~ g _ { 0 }$ . The learned adjustment allows the response to balance smoothing with available energy and scheduling costs. For $v _ { t } ^ { + } > v _ { t } ^ { - }$ , the reference position within the committed range is $r _ { t , s } ~ = ~ ( \widehat { P } _ { t , s } ^ { \mathrm { D C } } - g _ { t - 1 , s } - v _ { t } ^ { - } ) / ( v _ { t } ^ { + } - v _ { t } ^ { - } )$ , clipped strictly between zero and one before applying the logit. The proposed net BESS power is

$$
\widetilde { p } _ { t , s } ^ { \mathrm { B } } = v _ { t } ^ { - } + \left( v _ { t } ^ { + } - v _ { t } ^ { - } \right) \mathrm { s i g m o i d } ( \log \mathrm { i t } ( r _ { t , s } ) + \epsilon _ { t , s } ) .\tag{13}
$$

If the two power limits coincide, the output equals that common value. The decoder then clips the proposed power to the part of the committed range that satisfies the power, energy, and non-export limits: $\begin{array} { r } { \boldsymbol { p } _ { t , s } ^ { \mathrm { B } ^ { \mathrm { ~ * ~ } } } = \mathrm { c l i p } _ { [ L _ { t , s } , U _ { t , s } ] } ( \boldsymbol { \widetilde { p } } _ { t , s } ^ { \mathrm { B } } ) } \end{array}$ . This net power determines the reserve activation $a _ { t , s } = p _ { t , s } ^ { \mathrm { B } } - b _ { t } ,$ , charging power $p _ { t , s } ^ { \mathrm { c h } } = [ - p _ { t , s } ^ { \mathrm { B } } ] _ { + }$ , and discharging power $p _ { t , s } ^ { \mathrm { d i s } } = [ p _ { t , s } ^ { \mathrm { B } } ] _ { + }$ where $[ p ] _ { + } = \operatorname* { m a x } \{ p , 0 \}$ . This split prevents simultaneous charging and discharging, with $z _ { t , s } ~ = ~ 1$ for positive net power and zero otherwise. The remaining variables in $\mathbf { x } ^ { \mathrm { r e c } }$ follow directly: energy and grid import are computed using (5) and (7), and $\delta _ { t , s } ^ { \pm }$ record the ramp changes exceeding the limits in (8). The updated energy levels determine the next interval’s commitment bounds.

## B. Training

Training first teaches BS-NOL to generate battery schedules, then teaches the predictor to improve the decisions made through this layer. Let $F ( \mathbf { x } ; \mathbf { H } , \pmb { \sigma } )$ be the objective in (10) with the CVaR auxiliary variables minimized out. Here, $\mathbf { x } ~ = ~ ( \mathbf { x } ^ { ( 1 ) } , \mathbf { x } ^ { \mathrm { r e c } } )$ contains the scheduling decisions, $\mathbf { H } = ( H _ { t , s } ) _ { t \in { \mathcal { T } } , s \in { \mathcal { S } } }$ contains the selected historical demand profiles, and $\pmb { \sigma } = ( E _ { 0 } , p _ { 0 } ^ { \mathrm { B } } , g _ { 0 } )$ is the initial state, with $p _ { 0 } ^ { \mathrm { B } }$ the initial net BESS power.

For each training instance i, BS-NOL with parameters ω uses historical demand $\mathbf { H } _ { i }$ and initial state $\sigma _ { i }$ to generate a schedule $\widehat { \mathbf { x } } _ { \omega , i }$ of commitments and battery responses. We train ω on N instances using

$$
\mathcal { L } _ { \mathrm { N O L } } ( \omega ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F ( \widehat { \mathbf { x } } _ { \omega , i } ; \mathbf { H } _ { i } , \pmb { \sigma } _ { i } ) .\tag{14}
$$

The loss jointly trains the encoder and both output heads by evaluating reserve costs, operating costs, and ramping risk under the input demand profiles, without optimal-solution labels.

Next, we freeze BS-NOL at $\omega ^ { \star }$ and use forecast-based scenarios from (11) to generate full-horizon commitments and the shared summary. Keeping both fixed, we recompute responses under historical demand $\mathbf { H } _ { i }$ from the same initial state using the same decoder. This evaluates the commitments when demand differs from the forecast.

With these commitments fixed, discharging too much now may leave insufficient energy for a later commitment, while charging too much may leave insufficient space for later charging. To meet later commitments, the decoder keeps the energy remaining after interval t within $\mathcal { E } _ { t } = [ \underline { { E } } _ { t } , \overline { { E } } _ { t } ]$ , where $\underline { { E } } _ { t }$ is the minimum energy needed for later discharging and $\overline { { E } } _ { t }$ leaves enough space for later charging. At the end of the planning horizon $( t = T )$ , no further commitments remain, so $\mathcal { E } _ { T } = [ E _ { \mathrm { m i n } } , E _ { \mathrm { m a x } } ]$ . For demand above the rated discharge power, we compute these ranges backward from $t = T$ to $t = 1$

$$
\mathcal { E } _ { t - 1 } = \mathcal { A } _ { t } \cap \left[ \underline { { E } } _ { t } + \eta _ { \mathrm { c h } } \Delta t v _ { t } ^ { - } , ~ \overline { { E } } _ { t } + \frac { \Delta t } { \eta _ { \mathrm { d i s } } } v _ { t } ^ { + } \right] .\tag{15}
$$

Here, $\boldsymbol { A } _ { t }$ contains the starting energy levels that support the current commitment under (6). The bracketed interval contains those from which charging or discharging within the committed limits can bring the battery into $\mathcal { E } _ { t } ,$ following (5). Their intersection therefore supports both the current and remaining commitments.

The responses recomputed within these energy ranges, together with the unchanged forecast-based commitments, form $\widehat { \mathbf { x } } _ { \theta , i }$ . We train θ to minimize its cost and ramping risk:

$$
\mathcal { L } _ { \mathrm { D F L } } ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F ( \widehat { \mathbf { x } } _ { \theta , i } ; \mathbf { H } _ { i } , \pmb { \sigma } _ { i } ) .\tag{16}
$$

Gradients pass through the frozen BS-NOL and piecewise differentiable decoder to update only θ. After training, both networks generate decisions through forward passes with fixed parameters and no optimization solver.

## IV. SIMULATION AND DISCUSSION

## A. Experimental Setup

We use AI facility profiles (10 MW, 2,840 nodes, 80% target occupancy) from [15]. Chronological splits yield 4,500 training and 128 validation instances. Simulation spans 10,000 oneminute decisions with 60-min horizons. Under a discrete-time assumption, each interval’s average demand is known after commitment and before dispatch. The 1-MW/1-MWh BESS has 10–90% SOC limits, 50% initial/target SOC, and charging/discharging efficiencies of 0.95/0.90. Illustrative smoothingfocused costs are $\pi _ { t } = 8 0 , c _ { t } ^ { \pm } = 2 0 , c ^ { \mathrm { d e g } } = 2 .$ , and $c ^ { \mathrm { t e r m } } = 1 0 0$ in weighted cost units (CU)/MWh, with $R ^ { \uparrow } ~ = ~ R ^ { \downarrow } ~ = ~ 0$ $\lambda ^ { \pm } \ = \ 1 \ \mathrm { C U / k W } , \ \alpha \ = \ 0 . 7 ,$ , and $\beta \ : = \ : 0 . 5$ . All predictors use a 32 → 32 → 60 MLP. MSE+Gurobi and MSE+BS-NOL pair MSE-trained forecasts with Gurobi and frozen BS-NOL, respectively, where the latter isolates decision-focused predictor training. BS-NOL uses 8 scenarios due to limited historical profiles available during training. MSE+Gurobi uses 32, which performed best among 8, 16, and 32 in preliminary tests. The Gurobi solver (version 13.0.3) has a 60-s solver-time budget per decision. Adam settings are batch size 16, early-stopping patience $^ { 1 0 , }$ and learning rates $1 0 ^ { - 2 }$ for MSE and $1 0 ^ { - 3 }$ for BS-NOL/DFL. All experiments run on an Apple M3 Pro CPU.

## B. Simulation Results and Discussion

TABLE I  
CONTINUOUS SCHEDULING PERFORMANCE COMPARISON
<table><tr><td>Metric</td><td>No BESS</td><td>Perf-Info.</td><td>MSE+Gurobi</td><td>MSE+BS-NOL</td><td>BS-DFL</td></tr><tr><td>Mean |∆g|</td><td>34.7274</td><td>0.8237</td><td>3.5293</td><td>1.1626</td><td>0.9988</td></tr><tr><td>P95  $| \Delta g |$ </td><td>86.6182</td><td>0.4066</td><td>23.4497</td><td>0.0971</td><td>0.0104</td></tr><tr><td>Maximum |∆g|</td><td>199.949</td><td>986.790</td><td>413.191</td><td>179.400</td><td>203.920</td></tr><tr><td>Total cost</td><td>439,557</td><td>100,832</td><td>128,382</td><td>106,500</td><td>105,257</td></tr><tr><td>Reserve cost</td><td>0</td><td>215</td><td>717</td><td>2,485</td><td>2,898</td></tr><tr><td>Forecast MAE</td><td>一</td><td>一</td><td>179.97</td><td>179.97</td><td>240.66</td></tr><tr><td>Mean latency</td><td>一</td><td>一</td><td>2,513.41</td><td>24.76</td><td>18.77</td></tr><tr><td>P95 latency</td><td></td><td>一</td><td>5,581.17</td><td>26.77</td><td>21.21</td></tr><tr><td>Maximum latency</td><td></td><td></td><td>31,566.21</td><td>83.51</td><td>64.07</td></tr></table>

All methods use the same 10,000-minute trace. $\Delta g$ denotes one-minute grid-import change; P95 is the 95th percentile; MAE (mean absolute error) covers 600,000 forecast-truth pairs from rolling 60-step forecasts, including overlapping horizons. Power and MAE: kW; costs: CU; latency: ms. Total cost includes energy, reserve, degradation, ramp, and final SOC penalties, excluding CVaR. Costs are rounded to the nearest CU. Latencies are controller wall times from the respective runs.

Fig. 2 and Table I compare scheduling performance over the 10,000-minute evaluation trace. No BESS denotes uncontrolled demand, while Perfect-Information assumes known future demand within each 60-minute planning horizon. In Fig. 2(a), BS-DFL exhibits extended periods of nearly constant grid import, resembling the perfect-information reference. The enlarged windows in Fig. 2(b)-(c) further illustrate its ability to smooth grid import fluctuations under both low- and highvariability conditions. Compared with MSE+Gurobi, BS-DFL reduces mean absolute grid-import changes by 71.70%. Its P95 change of 0.0104 kW indicates nearly constant grid import during most intervals, although occasional large changes remain, as reflected by the maximum of 203.92 kW. Under the zero-ramp-limit settings, this behavior is consistent with the cumulative absolute change penalty in (9). The largest grid-import changes across the BESS methods occur during the sharp demand decline in Fig. 2(c). Limited look-ahead may lead to these changes, as the subsequent demand trough and recovery may not yet be visible within the 60-minute prediction horizon.

![](images/1339b1bb238d02dda8a9b0cf6ad014898a88ef665d28ce4e9e3cccc0e38b334d.jpg)  
Fig. 2. Grid-import power over the 10,000-minute evaluation trace. (a) Full trajectory with a broken vertical axis. (b)-(c) 12-hour windows with the lowest and highest mean absolute minute-to-minute demand changes, respectively, shown on different vertical scales.

Despite an additional reserve expenditure of 2,181 CU, BS-DFL reduces total cost by 18.01% relative to MSE+Gurobi, bringing it within 4.39% of the perfect-information reference. The resulting reduction in ramp penalties more than offsets the additional reserve expenditure. MSE+BS-NOL improves scheduling performance over MSE+Gurobi through smoother grid import and lower total cost, demonstrating the benefit of the neural optimization layer. BS-DFL further reduces mean absolute changes by 14.09% and total cost by 1.17% relative to MSE+BS-NOL, while lowering P95 from 0.0971 to 0.0104 kW. These improvements, despite a higher forecast MAE, demonstrate the value of training predictors directly for downstream scheduling performance. BS-DFL also achieve a mean decision time of 18.77 ms, with a maximum of 64.07 ms, providing computational headroom for more frequent dispatch updates.

## V. CONCLUSION AND FUTURE WORK

This work developed BS-DFL to align demand forecasting with the operating costs and ramping risks of battery scheduling. Compared with an MSE-trained predictor coupled with a MILP solver, BS-DFL reduced mean absolute grid-import changes by 71.70% and total operating cost by 18.01%, while achieving an average decision time of 18.77 ms without online optimization. Future work will investigate learned scheduling strategies, electricity-price uncertainty, and robustness across battery configurations and operating conditions.

## REFERENCES

[1] M.-S. Ko and H. Zhu, “Wide-area power system oscillations from large scale AI workloads,” IEEE Trans. Power Syst., vol. 41, no. 5, pp. 3908– 3921, 2026.

[2] B. A. Ross, X. Lyu, U. C. Nwaneto, S. M. Mohiuddin, and A. B. Nassif, “Using grid-forming energy storage systems to provide dynamic active power support for hyperscale data center,” IEEE Access, vol. 14, pp. 29 250–29 259, 2026.

[3] N. B. Abera, P. Musilek, and Y. Chen, “Battery-assisted ramp rate control of AI data centers: Change-point detection and optimal algorithm design,” in Proc. ACM Sustainability Week, 2026, pp. 266–272.

[4] X. Xie, C. Huang, Q. Hu, Z. Wu, S. Bu, and F. Li, “Risk-aware coordinated scheduling of battery storage and AI data centers with decoupled multi-layer AI workload models,” Appl. Energy, vol. 427, p. 128836, 2027.

[5] N. Mary and L. A. Dessaint, “Robust model predictive control of battery energy storage with neural network forecasting for peak shaving in university campus,” J. Build. Eng., vol. 107, p. 112445, 2025.

[6] S. Henni, J. Becker, P. Staudt, F. vom Scheidt, and C. Weinhardt, “Industrial peak shaving with battery storage using a probabilistic forecasting approach: Economic evaluation of risk attitude,” Appl. Energy, vol. 327, p. 120088, 2022.

[7] H. Zhang et al., “Decision-focused learning for power system decisionmaking under uncertainty,” IEEE Trans. Power Syst., vol. 41, no. 1, pp. 307–323, 2026.

[8] A. Paredes, J.-F. Toubeau, J. A. Aguado, and F. Vall<sup>´</sup> ee, “On the´ participation of energy storage systems in reserve markets using decision focused learning,” Sustain. Energy Grids Netw., vol. 42, p. 101677, 2025.

[9] J. Depoortere, H. Kazmi, and J. Driesen, “Decision-focused learning for optimal PV-battery scheduling,” J. Energy Storage, vol. 154, p. 121152, 2026.

[10] S. Yan, J. Zhang, X. Yao, J. Huang, and Y. Shi, “A predict-then-schedule framework for power distribution networks with AI data centers,” in Proc. IEEE Power Energy Soc. Gen. Meeting (PESGM), 2026.

[11] N. Alkhulaifi et al., “Decision-focused learning enhanced by automated feature engineering for energy storage optimisation,” Expert Syst. Appl., vol. 302, p. 130554, 2026.

[12] M. Yi, S. Alghumayjan, and B. Xu, “Perturbed decision-focused learning for modeling strategic energy storage,” IEEE Trans. Smart Grid, vol. 16, no. 3, pp. 2574–2586, 2025.

[13] H. Kim, A. Das, and D. Wu, “A neural optimizer with decision-focused learning for optimal energy storage operation,” IEEE Trans. Ind. Informat., vol. 21, no. 12, pp. 9286–9296, 2025.

[14] X. Zheng, F. de Nijs, and E. Lam, “Solver-free decision-focused learning via neural optimization layers for edge selection,” arXiv:2611.09560, 2026.

[15] R. Vercellino et al., “Dataset of generative AI workload power profiles,” NLR Data Catalog, 2026, doi: 10.7799/3025227.