# Equilibria and Learning in Competitive Market Making

Matteo Onger

matteo.onger@unimi.it

Luigi Foscari

luigi.foscari@unimi.it

Department of Computer Science, University of Milan

## Abstract

We propose a market-making game played among N ≥ 2 strategic market makers and a single non-strategic representative trader. In a continuous quote space, standard finite-game arguments do not apply, while non-convex and discontinuous payofs rule out classical continuous existence results and complicate learning. Nonetheless, we characterize purestrategy Nash equilibria (PSNEs) and show that at least one PSNE always exists under minimal assumptions on the underlying asset value distribution, and that the equilibrium bid–ask spread widens strictly with asset value volatility. To obtain a tractable learning framework, we discretize prices and, under an additional regularity condition, show that the time-averaged play of no-internal-regret learners approaches the continuous Nash price envelope as the tick size vanishes. Finally, we empirically validate our theoretical results through numerical simulations.

Keywords: Market making, Online learning, Equilibrium selection, Market microstructure

## 1. Introduction

Classical models of price competition, such as the Bertrand oligopoly model (Bertrand, 1883), elegantly explain why competition may drive prices toward marginal cost. However, applying this one-sided logic directly to financial market making overlooks a fundamental structural feature of liquidity provision: market makers operate simultaneously on both the buy and sell sides of the market, and their quotes actively shape the direction of incoming order flow.

This two-sided interaction is particularly prominent in financial markets characterized by inelastic trading demand, such as commodity markets (Bogmans et al., 2024), where overal trading volume does not respond to price changes and market participants simply choose whether to buy or sell. In conventional Bertrand competition, a firm’s price reduction allows it to gain market share at the cost of a lower margin; in contrast, liquidity providers in inelastic markets post bid and ask quotes that establish a market mid-price, which acts as a threshold determining whether buying or selling is advantageous relative to the asset’s fundamental value, ultimately dictating trade direction and, indirectly, execution priority.

To study this mechanism, and how makers learn in it, we model liquidity provision as a static game played among N ≥ 2 strategic market makers and a single non-strategic representative trader. The former simultaneously post bid–ask quotes for a single asset with uncertain fundamental value, while the latter, upon observing the posted quotes, submits either a buy or a sell order according to a fixed decision rule that compares the realized fundamental value of the asset with the market mid-price induced by the quoted prices.

Contributions. By bridging Bertrand-style oligopoly (Bertrand, 1883) and classic marketmaking models \`a la Glosten and Milgrom (1985), our framework provides a tractable baseline for studying competition and learning dynamics in two-sided inelastic markets. To the best of our knowledge, this work provides the first theoretical and algorithmic analysis of market making in this setting. Our main contributions are summarized below:

• Nash Equilibrium Existence (Section 3): We characterize pure-strategy Nash equilibria (PSNEs) in continuous quote spaces and prove that at least one PSNE always exists with minimal assumptions on the underlying asset value distribution. This equilibrium establishes a zero-profit competitive benchmark that guarantees perfect parity among all market makers, serving as an explicit reference model for competitive liquidity provision.

• Impact of Volatility on Competitive Pricing (Section 3.1): We investigate how fundamental market uncertainty shapes equilibrium quotes. For location-scale families of asset distributions, we derive a structural scaling result: the equilibrium bid–ask spread scales linearly with asset value standard deviation. Crucially, this demonstrates that spread widenings can arise purely from competitive risk adjustments, rather than algorithmic collusion (Calvano et al., 2020; Harrington, 2018) or a higher share of informed traders (Glosten and Milgrom, 1985).

• Learning the Competitive Benchmark (Section 4): To obtain a tractable learning framework, we analyze learning over a discrete price grid with tick size ∆. Under an additional regularity condition, we prove that all grid correlated equilibria uniformly concentrate on the continuous Nash price envelope as $\Delta  0$ . Consequently, the timeaveraged play of adaptive market makers using no-internal-regret learning algorithms is guaranteed to asymptotically approach the continuous competitive benchmark.

Finally, theoretical predictions and learning dynamics are empirically validated through numerical simulations in Section 5.

## 1.1. Technical Challenges

Analyzing this market-making game presents distinct theoretical and algorithmic challenges:

• Two-Sided Coupling: Updating a quote shifts the implied mid-price, directly coupling execution risk across both sides of the book. As a result, the game cannot be reduced to two independent Bertrand competitions. Indeed, since strategies consist of quote pairs, market makers can execute cross-side deviations (e.g., aggressively lowering the ask while retreating on the bid), thereby expanding the space of potentially profitable strategies in ways that break standard single-side equilibrium arguments.

• Continuous vs. Discrete Tension: Payofs in this game are inherently non-convex and discontinuous. In continuous quote spaces, this rules out standard PSNE existence theorems and complicates learning dynamics. Discretizing prices resolves these hurdles, but it introduces grid-dependent equilibria that obscure the model’s true structure and parameter dependencies.

## 1.2. Related Work

This work connects to market microstructure, game theory, and learning in games; we review the most closely related literature.

Price Competition. Among classical oligopoly models (Cournot, 1838; Bertrand, 1883), Bertrand competition is the closest benchmark to our setting: a marginal price improvement causes a discontinuous jump in demand, which complicates equilibrium existence. Purestrategy equilibria can be recovered under additional assumptions, such as quasiconcave payofs (Dasgupta and Maskin, 1986; Reny, 1999) or suitable cost and demand conditions, possibly with multiple equilibria (Dastidar, 1995). Neither applies here: in particular, quasiconcavity fails for bid–ask pairs, since averaging two profitable one-sided deviations can leave a maker inactive on both sides. Our existence proof instead exploits the monotonicity of the midpoint operator via Tarski’s theorem (Tarski, 1955), as in games with strategic complementarities, where extremal equilibria likewise bracket the equilibrium set (Milgrom and Roberts, 1990; Vives, 1990), although our game is not supermodular.

Market Microstructure. Seminal models study how informed order flow (Glosten and Milgrom, 1985; Kyle, 1985; Easley and O’Hara, 1987) and inventory risk (Ho and Stoll, 1981) shape spreads, typically assuming that dealers break even. Our zero-profit quotes mirror the upper-tail pricing of Glosten (1994), but arise as a Nash outcome among finitely many strategic makers, as in models of competing dealers (Dennert, 1993; Biais et al., 2000; Back and Baruch, 2013). Unlike these works, which rely on noise traders to avoid no-trade outcomes (Milgrom and Stokey, 1982), we consider a fully informed trader with inelastic demand, so that quotes determine the direction of trade rather than its occurrence. Spreads are also well known to widen with volatility (Copeland and Galai, 1983; Ho and Stoll, 1981); we show that, within location-scale families, they scale exactly linearly. Finally, our grid game relates to the tick-size literature (Anshuman and Kalay, 1998; Cartea et al., 2022), whose small-tick limit we analyze.

Learning in Games and Market Making. Our guarantees build on online learning (Cesa-Bianchi and Lugosi, 2006; Orabona, 2023) and on the convergence of no-internalregret dynamics to correlated equilibria (Foster and Vohra, 1997; Hart and Mas-Colell, 2000; Blum and Mansour, 2007). Unlike Stoltz and Lugosi (2007), who learn over compact strategy sets with continuous payofs, we approximate our discontinuous game by grid games. In non-strategic environments, online learning has been used to learn market-making strategies (Abernethy and Kale, 2013; Cesa-Bianchi et al., 2025; Maran and Restelli, 2026), via reductions to first-price auctions (Achddou et al., 2021; Cesa-Bianchi et al., 2024; Han et al., 2024) and dynamic pricing (Cesa-Bianchi et al., 2019; Tullii et al., 2025), while agent-based studies train makers via reinforcement learning (Spooner et al., 2018; Ganesh et al., 2019); here, we characterize the equilibria that interacting learners approach.

Algorithmic Collusion. Learning agents can sustain supra-competitive prices (Calvano et al., 2020; Klein, 2021; Banchio and Skrzypacz, 2022; Hansen et al., 2021), also in dealer markets (Cont and Xiong, 2024; Colliard et al., 2022), raising regulatory concerns (Harrington, 2018); no-swap-regret learning rules out some of these outcomes (Arunachaleswaran et al., 2025), motivating our choice of learners. Hartline et al. (2024) detect collusion from prices alone, a test later shown to be confoundable (Hartline et al., 2025); in this vein, we identify one such confound: volatility-driven spread widening.

## 2. Model

In this section, we present the model of inelastic markets used throughout the paper. Let $N \geq 2$ and $\mathcal { N } : = \{ 1 , \dots , N \}$ denote the set of market makers. The fundamental value of the asset is modeled as a continuous, non-degenerate random variable $V ,$ governed by an unknown cumulative distribution function $F$ (with density $f$ where it exists) and bounded support $[ \underline { { v } } , \overline { { v } } ] \subseteq [ 0 , 1 ]$ . Crucially, market makers possess no prior knowledge of the distribution $F$ and must learn its structure over repeated interactions. Conversely, the trader observes the realized value v of V prior to placing an order.

At the start of each round, each market maker $i \in \mathcal N$ posts a pair $s _ { i } : = ( b _ { i } , a _ { i } )$ , consisting of a bid quote $b _ { i }$ and an ask quote ${ { a } _ { i } } ,$ with $0 \leq b _ { i } \leq a _ { i } \leq 1$ . The strategy space for market maker i is thus

$$
\begin{array} { r } { { \cal S } _ { i } : = \left\{ ( b , a ) \in [ 0 , 1 ] ^ { 2 } ~ \big | ~ 0 \leq b \leq a \leq 1 \right\} . } \end{array}
$$

Let $\begin{array} { r } { \textbf { s } : = \left( s _ { 1 } , \ldots , s _ { N } \right) \in \mathcal { S } : = \prod _ { i \in \mathcal { N } } S _ { i } } \end{array}$ denote the joint strategy profile, and let b := $( b _ { 1 } , \dotsc , b _ { N } )$ and $\mathbf { a } : = ( a _ { 1 } , \ldots , a _ { N } )$ denote the joint vectors of bid and ask quotes, respectively, so that ${ \bf s } = ( { \bf b } , { \bf a } )$ . We write s<sub>−i</sub> for the profile of all makers other than i. The prevailing market prices are determined by the best available quotes:

$$
a ^ { * } ( \mathbf { a } ) : = \operatorname* { m i n } _ { i \in \mathcal { N } } a _ { i } \qquad \mathrm { a n d } \qquad b ^ { * } ( \mathbf { b } ) : = \operatorname* { m a x } _ { i \in \mathcal { N } } b _ { i } ,
$$

which together induce the market mid-price

$$
m ^ { * } ( \mathbf { s } ) : = \frac { a ^ { * } ( \mathbf { a } ) + b ^ { * } ( \mathbf { b } ) } { 2 } .
$$

After observing the posted quotes, the trader chooses an action $d \in { \mathcal { D } } : = \{ - 1 , + 1 \}$ , where $d = + 1$ denotes a buy order and $d = - 1$ denotes a sell order. Specifically, upon observing the asset’s realized fundamental value v and comparing it with the market mid-price $m ^ { * } ( \mathbf { s } )$ 2 the trader selects an action $d \in \mathcal { D }$ according to the following decision rule:

$$
\mathbb { P } ( d = + 1 \mid v , \mathbf { s } ) : = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f ~ } } v > m ^ { * } ( \mathbf { s } ) , } \\ { 0 } & { { \mathrm { i f ~ } } v < m ^ { * } ( \mathbf { s } ) , } \\ { { \frac { 1 } { 2 } } } & { { \mathrm { i f ~ } } v = m ^ { * } ( \mathbf { s } ) , } \end{array} \right. }\tag{1}
$$

with $\mathbb { P } ( d = - 1 \mid v , \mathbf { s } ) = 1 - \mathbb { P } ( d = + 1 \mid v , \mathbf { s } )$

To complete the model specification, we define order allocation and execution payofs. Let $n _ { A } ( \mathbf { a } ) : = | \{ i \in \mathcal { N } | a _ { i } = a ^ { * } ( \mathbf { a } ) \} |$ and $n _ { B } ( \mathbf { b } ) : = | \{ i \in \mathcal { N } | b _ { i } = b ^ { * } ( \mathbf { b } ) \} |$ denote the number of market makers quoting the best ask and bid prices, respectively. If multiple market makers quote the same prevailing price, the incoming order is allocated equally among them.<sup>1</sup> Consequently, for a given strategy profile s, the expected payof of market maker $i \in \mathcal N$ is $u _ { i } ( \mathbf { s } ) : = \mathbb { E } _ { V , d \sim \mathbb { P } ( \cdot | V , \mathbf { s } ) } [ r _ { i } ( \mathbf { s } ; V , d ) ]$ ], where the ex-post payof $r _ { i }$ under realization v and trader order d is given by

$$
r _ { i } ( \mathbf { s } ; v , d ) : = \left\{ \begin{array} { l l } { \displaystyle \frac { a _ { i } - v } { n _ { A } ( \mathbf { a } ) } } & { \mathrm { i f } \ d = + 1 \ \mathrm { a n d } \ a _ { i } = a ^ { * } ( \mathbf { a } ) , } \\ { \displaystyle \frac { v - b _ { i } } { n _ { B } ( \mathbf { b } ) } } & { \mathrm { i f } \ d = - 1 \ \mathrm { a n d } \ b _ { i } = b ^ { * } ( \mathbf { b } ) , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{2}
$$

For completeness, the ex-post realized payof of the trader is equal to $\left( v - a ^ { * } ( \mathbf { a } ) \right) \mathbb { I } _ { \{ d = + 1 \} } +$ $\left( b ^ { * } ( \mathbf { b } ) - v \right) \mathbb { I } _ { \{ d = - 1 \} }$ , implying that the overall trade interaction between the aggregate pool of market makers and the trader is zero-sum.

Strategic Scope and Game Reduction. Given the optimal decision rule in (1), the trader’s action is fully determined by the realized fundamental value and the posted quotes, leaving liquidity providers as the sole strategic players. Isolating their interactions yields a reduced general-sum game among the N market makers, where a buy order occurs exactly when $V > m ^ { * } ( \mathbf { s } )$ , and a sell order exactly when $V < m ^ { * } ( \mathbf { s } ) . ^ { 2 }$ Each maker’s expected payof accordingly decomposes as $u _ { i } ( \mathbf { s } ) = u _ { i } ^ { \mathrm { a s k } } ( \mathbf { s } ) + u _ { i } ^ { \mathrm { b i d } } ( \mathbf { s } )$ , where

$$
u _ { i } ^ { \mathrm { a s k } } ( \mathbf { s } ) : = \frac { \mathbb { I } _ { \left\{ a _ { i } = a ^ { * } ( \mathbf { a } ) \right\} } } { n _ { A } ( \mathbf { a } ) } \mathbb { P } \left( V > m ^ { * } ( \mathbf { s } ) \right) \left( a ^ { * } ( \mathbf { a } ) - \mathbb { E } \left[ V \mid V > m ^ { * } ( \mathbf { s } ) \right] \right)
$$

is maker i’s expected payof from buy orders, and $u _ { i } ^ { \mathrm { b i d } } ( \mathbf { s } )$ is defined symmetrically for sell orders. This reduced N-agent formulation fully captures the competitive dynamics of the market, providing the exact setting analyzed in the following sections.

## 3. Equilibrium Concept

In this section, we analyze the equilibrium properties of the proposed market-making game. Our main goal is to prove that at least one pure-strategy Nash equilibrium always exists. Establishing existence is nontrivial in our setting, as the strategy space is continuous and the payof functions are non-convex and discontinuous due to Bertrand-style price competition and tie-breaking in order allocation, precluding a direct application of standard equilibrium existence results.

Definition 1 (Pure-Strategy Nash Equilibrium). A strategy profile s is a purestrategy Nash equilibrium (PSNE) if, for every market maker $i \in \mathcal N$ and every feasible deviation $\widetilde { s } _ { i } : = ( \widetilde { b } _ { i } , \widetilde { a } _ { i } ) \in S _ { i }$ , we have:

$$
u _ { i } ( \widetilde { s } _ { i } , \mathbf { s } _ { - i } ) \leq u _ { i } ( s _ { i } , \mathbf { s } _ { - i } ) .
$$

In our two-sided market setting, a deviation $\widetilde { s } _ { i }$ allows market maker i to unilaterally alter its ask quote, its bid quote or both, simultaneously. As previously mentioned, such a deviation can afect not only the execution price and the maker’s share of the existing order flow, but also the composition of that flow by shifting the induced mid-price and hence the trader’s buy–sell decision. This coupling between the two sides of the market enlarges the set of potentially profitable deviations and makes their analysis more complex than in standard Bertrand competition. Establishing that a profile is a PSNE therefore also requires ruling out deviations that may exploit this interdependence.

To establish existence, we first derive the necessary conditions that a Nash equilibrium profile must satisfy.

Observation 2. In any PSNE, the expected profit from each side of the book must be zero independently. To see this, first observe that a strategy yielding negative expected payof cannot be optimal, since quoting the boundary prices, i.e., $s _ { i } = ( 0 , 1 )$ , guarantees a non-negative payof for every realization of V. Hence, the expected profit must be nonnegative. On the other hand, strictly positive expected profits cannot be sustained: if a side of the book generated strictly positive expected profit, a competing maker could undercut the corresponding quote by an arbitrarily small amount and capture all of the resulting order flow at a slightly less favorable price. Finally, because makers can adjust their quotes on the two sides independently, positive profits on one side cannot compensate for losses on the other: a maker could withdraw from the loss-making side while retaining the profitable side.

Observation 3. As in standard Bertrand competition, any PSNE must have at least two market makers quoting the best ask and bid, i.e., $n _ { A } ( { \bf a } ) , n _ { B } ( { \bf b } ) \ge 2$ . The usual undercutting argument must account for the fact that quotes also afect the mid-price. If, for instance, $n _ { A } ( \mathbf { a } ) = 1$ , the sole ask maker can execute a joint deviation by raising its ask by a small $\varepsilon > 0$ , while improving its bid by a much smaller amount. By shifting the mid-price upward, this deviation increases the margin on buy orders and converts marginal buy orders into sell orders captured by its new bid, yielding a strictly positive expected profit.

Building on the observations above, consider a PSNE profile s, and let $a ^ { * }$ , $b ^ { * }$ and $m ^ { * } : = ( a ^ { * } + b ^ { * } ) / 2$ denote its best ask, best bid and induced mid-price, respectively. The zero-profit condition implies

$$
\operatorname { \mathbb { E } } \left[ a ^ { * } - V \mid V > m ^ { * } \right] = 0 , \qquad \operatorname { \mathbb { E } } \left[ V - b ^ { * } \mid V < m ^ { * } \right] = 0 ,
$$

and hence

$$
a ^ { * } = \mathbb { E } \left[ V \mid V > m ^ { * } \right] , \qquad b ^ { * } = \mathbb { E } \left[ V \mid V < m ^ { * } \right] .
$$

Thus, at any PSNE, the zero-profit condition uniquely determines the best quotes for its midprice $m ^ { * }$ . More generally, for any candidate equilibrium mid-price $\widehat { m }$ , internal consistency requires that the midpoint induced by the corresponding zero-profit quotes coincide with $\widehat { m }$ itself, motivating the following definition.

Definition 4 (Midpoint Operator). For any m $\in [ { \underline { { v } } } , { \overline { { v } } } ]$ , define the zero-profit quotes

$$
a ( m ) : = \mathbb { E } \left[ V \mid V > m \right] , \qquad b ( m ) : = \mathbb { E } \left[ V \mid V < m \right]
$$

whenever the conditioning events have positive probability, and set $a ( \overline { { v } } ) : = \overline { { v } }$ and $b ( \underline { { v } } ) : = \underline { { v } }$ otherwise, in accordance with their one-sided limits. We then define the midpoint operator $H : [ \underline { { v } } , \overline { { v } } ] \to [ \underline { { v } } , \overline { { v } } ]$ as

$$
H ( m ) : = \frac { a ( m ) + b ( m ) } { 2 } .
$$

By construction, $H ( m )$ is the midpoint induced by the zero-profit quotes associated with $m .$ . Consequently, the mid-price $m ^ { * }$ of any PSNE must be self-consistent, satisfying

$$
m ^ { * } = H ( m ^ { * } ) ,
$$

i.e., any PSNE mid-price is necessarily a fixed point of $H .$

We denote by ${ \mathcal { F } } : = \{ m \in [ \underline { { v } } , \overline { { v } } ] \mid m = H ( m ) \}$ the set of fixed points of H. Next, we show that $\mathcal { F }$ is nonempty and lies strictly between the lower and upper bounds of the support of $V$

Theorem 5 (Existence of a Zero-Profit Midpoint). The set of fixed points $\mathcal { F }$ of H is nonempty and satisfies ${ \mathcal { F } } \subset ( { \underline { { v } } } , { \overline { { v } } } )$

Proof. We first establish that H is an order-preserving self-map on the complete lattice $[ \underline { { v } } , \overline { { v } } ]$ . For every $m \in [ \underline { { v } } , \overline { { v } } ] , a ( m )$ and $b ( m )$ are well defined. Moreover, both functions are nondecreasing in m: increasing m removes lower values from the conditioning event defining $a ( m )$ , while adding higher values to the conditioning event defining $b ( m )$ . Consequently, $H ( m ) = ( a ( m ) + b ( m ) ) / 2$ is also nondecreasing, establishing that H is indeed an orderpreserving self-map on [v, v], as claimed.

By the Knaster–Tarski fixed-point theorem (Tarski, 1955), H admits at least one fixed point $\widehat { m } \in [ \underline { { v } } , \overline { { v } } ]$ satisfying ${ \widehat { m } } = H ( { \widehat { m } } )$ , so $\mathcal { F } \neq \emptyset$

It remains to rule out boundary fixed points. At the upper boundary, we have

$$
H ( \overline { { v } } ) = \frac { \overline { { v } } + b ( \overline { { v } } ) } { 2 } < \overline { { v } } ,
$$

since $b ( \overline { { v } } ) = \mathbb { E } [ V ] < \overline { { v } }$ . Symmetrically, at the lower boundary,

$$
H ( \underline { { v } } ) = \frac { a ( \underline { { v } } ) + \underline { { v } } } { 2 } > \underline { { v } } ,
$$

since $a ( \underline { { v } } ) = \mathbb { E } [ V ] > \underline { { v } } .$ . Consequently, neither boundary point can be a fixed point of $H _ { ; }$ , so ${ \mathcal { F } } \subset ( { \underline { { v } } } , { \overline { { v } } } )$ ■

Although the zero-profit condition is necessary for a PSNE, it is not suficient. For any ${ \widehat { m } } \in { \mathcal { F } }$ , we denote by $s ( \widehat { m } ) : = ( b ( \widehat { m } ) , a ( \widehat { m } ) )$ the zero-profit strategy at $\widehat { m }$ , and by $\mathbf { s } ( \widehat { m } ) : = ( s ( \widehat { m } ) , \dots , s ( \widehat { m } ) )$ the symmetric zero-profit profile where all market makers play $s ( \widehat { m } )$ . Even when all competitors stick to $\mathbf { s } _ { - i } ( \widehat { m } )$ , a maker i may still find a profitable unilateral deviation $\widetilde { s } _ { i } : = ( \widetilde { b } _ { i } , \widetilde { a } _ { i } )$ that exploits the two-sided coupling of the market.

By improving a quote on one side, e.g., undercutting the ask to $\widetilde { a } _ { i } < a ( \widehat { m } )$ paired with bid withdrawal $( { \widetilde { b } } _ { i } < b ( { \widehat { m } } ) )$ , as illustrated in Figure 1, maker i becomes the unique recipient of buy orders. This deviation shifts the induced midpoint to $y : = ( \widetilde { a } _ { i } + b ( \widehat { m } ) ) / 2 < \widehat { m }$ , yielding an expected ask payof of

$$
u _ { i } ^ { \mathrm { a s k } } ( \widetilde { s } _ { i } , { \bf s } _ { - i } ( \widehat { m } ) ) = \mathbb { P } ( V > y ) \left( \widetilde { a } _ { i } - a ( y ) \right) .
$$

This reveals the trade-of of two-sided competition: undercutting the ask narrows the pertrade margin, but the leftward shift of the midpoint (from $\widehat { m }$ to $y )$ extends execution to buyers with lower values, raising the execution probability $\mathbb { P } ( V > y )$ . A deviation is strictly profitable whenever this volume gain outweighs the lower margin. Crucially, this shift in the mid-price also impacts the opposite side of the book. Under the newly induced midpoint $y < \widehat { m }$ , maintaining the original bid quote $b ( \widehat { m } )$ , i.e., playing $( b ( \widehat { m } ) , \widetilde { a } _ { i } )$ , would subject maker i to an expected bid payof of

![](images/05b6c72ddd28e988d9812e48f3f61874344ed43e6237ed41561ef43e2837d6db.jpg)  
Figure 1: Mechanism of a unilateral ask undercut paired with bid withdrawal. Lowering the ask to $\widetilde { a } _ { i }$ narrows the per-trade execution margin but shifts the induced midpoint to $y < \widehat { m }$ , thereby expanding the captured buy order flow across $( y , { \overline { { v } } } ]$

$$
u _ { i } ^ { \mathrm { b i d } } \big ( ( b ( \widehat { m } ) , \widetilde { a } _ { i } ) , \mathbf { s } _ { - i } ( \widehat { m } ) \big ) = \frac { 1 } { N } \mathbb { P } ( V < y ) \left( b ( y ) - b ( \widehat { m } ) \right) \leq 0 ,
$$

since $b ( y ) \leq b ( \widehat { m } )$ . This exposes the maker to adverse selection on the bid side, making it strictly loss-making whenever $b ( y ) < b ( \widehat { m } )$ . To prevent these potential losses from reducing its ask profits, the deviating maker plays $\widetilde { s } _ { i }$ instead, actively withdrawing from the bid side, which ensures

$$
u _ { i } ^ { \mathrm { b i d } } ( \widetilde { s } _ { i } , { \bf s } _ { - i } ( \widehat { m } ) ) = 0 .
$$

The next step is therefore to establish whether at least one point of $\mathcal { F }$ is robust against such deviations. The following lemmas establish the properties needed to answer this question (their proofs are provided in Section A.1); combining them, we then identify such a point, which constitutes a PSNE mid-price.

## Lemma 6. Let ${ \widehat { m } } \in { \mathcal { F } }$ . If the associated zero-profit profile $\mathbf { s } ( \widehat { m } )$ admits a profitable unilateral deviation, then it admits either a profitable ask undercut with bid withdrawal or a profitable bid raise with ask withdrawal.

The preceding lemma shows that, without loss of generality, it sufices to analyze these single-sided deviations. Hereafter, an ask undercut paired with bid withdrawal is referred to simply as an ask undercut (and symmetrically for bid raises).

![](images/f61cdfd33726e9dbad185e33292344500ba23bf81ed7d0c375097fd18a483d12.jpg)  
Figure 2: Visual breakdown of the three case contradictions in the proof of Theorem 10.

## Lemma 7. Let ${ \widehat { m } } \in { \mathcal { F } }$ . Then:

(i) For any $x > \widehat { m }$ , the unique bid raise inducing mid-price x is profitable if and only if $2 x - b ( x ) < a ( \widehat { m } )$ , which implies $H ( x ) > x$

(ii) For any $y < \widehat { m }$ , the unique ask undercut inducing mid-price y is profitable if and only $i f 2 y - a ( y ) > b ( \widehat { m } )$ , which implies $H ( y ) < y$

Lemma 8. Let $m _ { \operatorname* { m i n } } : = \operatorname* { m i n } \mathcal { F }$ and $m _ { \mathrm { m a x } } : = \operatorname* { m a x } \mathcal { F }$ . Then (i) no profitable ask undercut exists at $m _ { \mathrm { m i n } } .$ , and (ii) no profitable bid raise exists at $m _ { \mathrm { m a x } }$

## Lemma 9. Let $m _ { 1 } , m _ { 2 } \in { \mathcal { F } }$ with $m _ { 1 } < m _ { 2 }$ . Then:

(i) If a profitable bid raise at $m _ { 1 }$ induces a mid-price $x > m _ { 2 }$ , then the bid raise inducing the same midpoint x at $m _ { 2 }$ is also profitable.

(ii) If a profitable ask undercut at $m _ { 2 }$ induces a mid-price $y < m _ { 1 }$ , then the ask undercut inducing the same midpoint y at $m _ { 1 }$ is also profitable.

As anticipated above, we now combine these properties to identify a fixed point that admits no profitable unilateral deviation and hence constitutes a PSNE mid-price.

Theorem 10 (Existence of a Pure-Strategy Nash Equilibrium). Define

$$
{ \mathcal { F } } _ { B } : = \{ m \in { \mathcal { F } } \mid n o \ p r o f t a b l e \ b i d \ r a i s e \ e x i s t s \ a t \ m \}
$$

and $m ^ { - } : = \operatorname* { i n f } { \mathcal { F } _ { B } }$ . Then $\mathcal { F } _ { B } \neq \emptyset$ , and $\mathbf { s } ( m ^ { - } )$ is a pure-strategy Nash equilibrium.

Proof sketch. Since H is continuous, F is compact: it has a minimum and maximum fixed point, $m _ { \mathrm { m i n } }$ and $m _ { \mathrm { m a x } }$ , and contains the infimum and supremum of any of its nonempty subsets. Lemma 8 makes $m _ { \mathrm { m a x } }$ immune to bid raises, so $\mathcal { F } _ { B } \neq \emptyset$ and $m ^ { - } = \operatorname* { i n f } { \mathcal { F } } _ { B } \in { \mathcal { F } } .$ . A short contagion argument (Lemma 9) shows $m ^ { - } \in \mathcal { F } _ { B }$ , so $m ^ { - } = \operatorname* { m i n } \mathcal { F } _ { B }$ is itself immune to bid raises.

It remains to show that m<sup>−</sup> is also immune to undercuts. Suppose, towards a contradiction, that a profitable undercut at $m ^ { - }$ induces $y < m ^ { - }$ . Lemmas 8 and 9 place y strictly between $m _ { \mathrm { m i n } }$ and $m ^ { - }$ , and compactness gives a largest fixed point $m _ { k } < y$ . Since m<sub>k</sub> < m<sup>−</sup> = min $\mathcal { F } _ { B } , \ : m _ { k }$ admits a profitable bid raise to some $x > m _ { k }$ ; Lemma 7 gives $H ( x ) > x ,$ which rules out $x = y$ and $x = m ^ { - }$ , leaving three cases for x:

1. Case $x < y \colon$ H then maps $[ x , y ]$ into itself, so Knaster–Tarski produces a fixed point strictly between x and y, contradicting the maximality of $m _ { k }$

2. Case $y < x < m$ <sup>−</sup>: Inequalities from Lemma 7 combine, via monotonicity of $a ( \cdot ) , b ( \cdot )$ into $x < y .$ , contradicting $y < x$

3. Case $x > m ^ { - }$ : Lemma 9 carries the bid raise at $m _ { k }$ forward to $m ^ { - }$ , contradicting $m ^ { - } \in \mathcal { F } _ { B }$

No case survives, so $m ^ { - }$ is also undercut-immune, and Lemma 6 concludes. The full proof is given in Section $\mathrm { A . 1 }$

While the existence result above focuses on the minimal fixed point immune to bid raises, a natural dual argument applies symmetrically to undercuts.

Theorem 11 (Dual Existence of a Pure-Strategy Nash Equilibrium). Define

${ \mathcal { F } } _ { A } : = \{ m \in { \mathcal { F } } \mid$ no profitable aks undercut exists at m}

and $m ^ { + } : = \operatorname* { s u p } { \mathcal { F } _ { A } }$ . Then $\mathcal { F } _ { A } \neq \emptyset$ , and $\mathbf { s } ( m ^ { + } )$ is a pure-strategy Nash equilibrium.

Theorems 10 and 11 thus yield two, possibly coinciding, equilibrium mid-prices, $m ^ { - }$ and $m ^ { + }$ , obtained from dual constructions. We now show that these are not just any two equilibria: they bracket every other one.

Theorem 12 (Structure of Equilibria). The equilibrium midpoints $m ^ { - }$ and $m ^ { + }$ satisfy $m ^ { - } \leq m ^ { + }$ , and are respectively the smallest and largest equilibrium midpoints. Hence, every equilibrium mid-price belongs to the interval $[ m ^ { - } , m ^ { + } ]$

Proof. Any PSNE mid-price $m ^ { * }$ admits neither a profitable ask undercut nor a profitable bid raise, so $m ^ { * } \in \mathcal { F } _ { A } \cap \mathcal { F } _ { B }$ and hence $\begin{array} { r l } { m ^ { - } = } & { { } } \end{array}$ inf ${ \mathcal { F } } _ { B } \leq m ^ { * } \leq$ sup $\mathcal { F } _ { A } = m ^ { + }$ . Since $m ^ { - }$ and $m ^ { + }$ are themselves PSNE mid-prices (Theorems 10 and 11), taking $m ^ { * } = m$ <sup>−</sup> yields $m ^ { - } \leq m ^ { + }$ , and the bounds show that they are the smallest and largest PSNE mid-prices, respectively.

Concretely, combining all preceding results, for any equilibrium mid-price $m ^ { * } \in [ m ^ { - } , m ^ { + } ]$ 」， let $a ( m ^ { * } )$ and $b ( m ^ { * } )$ be the corresponding zero-profit quotes. Then, any profile ${ \bf s } = ( { \bf b } , { \bf a } )$ in which

1. at least two market makers quote the best ask $a ( m ^ { * } )$ and at least two market makers quote the best bid $b ( m ^ { * } )$ (ensuring $n _ { A } ( \mathbf { a } ) \geq 2$ and $n _ { B } ( \mathbf { b } ) \geq 2 )$ , and

2. all remaining market makers k submit non-improving quotes, i.e., $a _ { k } \ \geq \ a ( m ^ { * } )$ and $b _ { k } \le b ( m ^ { * } )$ 2

constitutes a pure-strategy Nash equilibrium. Thus, equilibrium is robust to any passive configuration of non-competitively quoting market makers.

Out-of-Support Zero-Profit Profile (mb <sup>≤</sup> v)  
![](images/0b3f535070a1559eb5eb08b56ec1496996c8f480438f9c4709a44b569d3f5f48.jpg)  
Unilateral Deviation $( \widetilde { s } _ { i } = ( \widetilde { b } _ { i } , \widetilde { a } _ { i } ) )$  
Figure 3: Profitable boundary deviation at an out-of-support candidate midpoint $\widehat { m } \leq \underline { { v } } .$ Posting $\widetilde { b } _ { i } = \underline { { v } }$ while withdrawing the ask shifts the induced midpoint to $y > \underline { { v } } ,$ reactivating sell order flow on [v, y) with a strictly positive expected margin.

Out-of-Support Midpoints. Throughout our analysis, we restricted attention to interior midpoints $m \in ( \underline { { v } } , \overline { { v } } )$ , where both sides of the market are active with positive probability. Although candidate midpoints outside the support $( \widehat { m } \leq \underline { { v } }$ or $\widehat { m } \geq \overline { { v } } )$ can exist, they are economically less interesting and can never sustain a PSNE. Indeed, as illustrated in Figure 3, if $\widehat { m } \leq \underline { { v } } .$ , the bid side is completely inactive; a market maker can unilaterally quote $\dot { b _ { i } } = \underline { { v } }$ while withdrawing its ask, shifting the induced midpoint into the support and capturing sell orders with strictly positive expected profit, since $V > \underline { { v } }$ almost surely. The case $\widehat { m } \geq \overline { { v } }$ is symmetric, with $\widetilde { a } _ { i } = \overline { { v } }$ . Therefore, we can restrict the search for PSNE mid-prices to $( \underline { { v } } , \overline { { v } } )$

Beyond Pure-Strategy Nash Equilibria. Extending our analysis to richer solution concepts, such as mixed-strategy Nash equilibria (MSNEs) and correlated equilibria (CEs), is a natural and interesting next step. Randomization and correlation can readily be layered onto our results without generating new economic outcomes:

• Mixed Strategies: Passive makers, those quoting strictly non-competitively $( a _ { i } > a ( m ^ { * } )$ and $b _ { i } < b ( m ^ { * } ) ,$ , can randomize freely, because they remain non-executing, so the prevailing best quotes remain bound to the zero-profit competitive threshold.

• Correlated Strategies: A public correlation device can synchronize agents on one of the (possibly multiple) PSNEs of Theorem 12, again leaving the zero-profit structure intact.

What we cannot show in general is the converse: that these are the only routes to an MSNE or CE. The obstacle lies specifically in randomization: against a genuinely mixing opponent, a deviation’s profitability becomes an expectation over its randomized quotes, whose sign is hard to control without further structure on F. Indeed, under the regularity of Section 4 and uniqueness of the PSNE mid-price, we establish an asymptotic version of this characterization, but achieving it without such assumptions remains a nontrivial open problem.

## 3.1. Impact of Asset Variance on Equilibrium Spreads

Having established the existence and structure of pure-strategy Nash equilibria, we now examine how the equilibrium bid–ask spread responds to changes in the uncertainty surrounding the underlying asset value V . Intuitively, higher volatility expands the conditional expectations of extreme value realizations, increasing adverse selection risk and forcing makers to widen their quotes.

To formalize this intuition, consider a location-scale family of asset distributions generated by a standardized continuous random variable X supported on a bounded interval $[ \underline { { x } } , \overline { { x } } ]$ ， with zero mean $\mathbb { E } [ X ] = 0$ and unit variance $\operatorname { V a r } ( X ) = 1$ . For any location parameter $\mu \in \mathbb { R }$ and scale parameter $\sigma > 0$ , define the fundamental value as

$$
V _ { \sigma } : = \mu + \sigma X .
$$

Thus, $V _ { \sigma }$ has support $[ { \underline { { v } } } _ { \sigma } , { \overline { { v } } } _ { \sigma } ] = [ \mu + \sigma \underline { { x } } , \mu + \sigma \overline { { x } } ]$ , which we require to lie in $[ 0 , 1 ]$ , with $\mathbb { E } [ V _ { \sigma } ] = \mu$ and $\mathrm { V a r } ( V _ { \sigma } ) = \sigma ^ { 2 }$ . For any mid-price m $\in ( \underline { { v } } _ { \sigma } , \overline { { v } } _ { \sigma } )$ , an afine change of variable gives the zero-profit ask and bid functions for $V _ { \sigma }$ as

$$
\begin{array} { r l r } & { } & { a _ { \sigma } ( m ) : = { \mathbb E } \left[ V _ { \sigma } \mid V _ { \sigma } > m \right] = \mu + \sigma { \mathbb E } \left[ X \mid X > \frac { m - \mu } { \sigma } \right] , } \\ & { } & { b _ { \sigma } ( m ) : = { \mathbb E } \left[ V _ { \sigma } \mid V _ { \sigma } < m \right] = \mu + \sigma { \mathbb E } \left[ X \mid X < \frac { m - \mu } { \sigma } \right] . } \end{array}
$$

Hence the midpoint operator $H _ { \sigma } ( m ) : = ( a _ { \sigma } ( m ) + b _ { \sigma } ( m ) ) / 2$ satisfies

$$
H _ { \sigma } ( m ) = \mu + \sigma H _ { 1 } \left( \frac { m - \mu } { \sigma } \right) ,
$$

where $H _ { 1 } ( z ) : = ( \mathbb { E } [ X \mid X > z ] + \mathbb { E } [ X \mid X < z ] ) / 2$ is the normalized midpoint operator for $X$ , extended to $[ \underline { { x } } , \overline { { x } } ]$ as in Definition 4 so that $H _ { 1 }$ (hence $H _ { \sigma } )$ is a well-defined self-map of the closed interval.

Proposition 13 (Scaling of the Zero-Profit Spread). Let $\mathcal { F } _ { 1 } \subset ( \underline { { x } } , \overline { { x } } )$ denote the set of fixed points of $H _ { 1 }$ , and define the spread functions

$$
\begin{array} { r l } & { \boldsymbol { w } _ { 1 } ( z ) : = \mathbb { E } \left[ X \mid X > z \right] - \mathbb { E } \left[ X \mid X < z \right] , } \\ & { \boldsymbol { w } _ { \sigma } ( m ) : = a _ { \sigma } ( m ) - b _ { \sigma } ( m ) . } \end{array}
$$

For any $\sigma > 0$

(i) The set of fixed points of $H _ { \sigma }$ satisfies $\mathcal { F } _ { \sigma } = \mu + \sigma \mathcal { F } _ { 1 }$

(ii) For any $\widehat { m } _ { \sigma } \in { \mathcal F } _ { \sigma }$ , writing $\widehat { m } _ { \sigma } = \mu + \sigma z$ with $z \in \mathcal { F } _ { 1 }$ , the spread satisfies

$$
w _ { \sigma } ( \widehat { m } _ { \sigma } ) = \sigma \cdot w _ { 1 } ( z ) .
$$

Proof. A point $\widehat { m } _ { \sigma } \in [ \underline { { v } } _ { \sigma } , \overline { { v } } _ { \sigma } ]$ is a fixed point of $H _ { \sigma }$ if and only if $\widehat { m } _ { \sigma } = H _ { \sigma } ( \widehat { m } _ { \sigma } )$ Substituting the scaled expression for $H _ { \sigma }$ and setting $z : = ( \widehat { m } _ { \sigma } - \mu ) / \sigma$ , this is equivalent to

$$
\mu + \sigma z = \mu + \sigma H _ { 1 } ( z ) \iff z = H _ { 1 } ( z ) .
$$

Thus, $\widehat { m } _ { \sigma } \in { \mathcal { F } } _ { \sigma }$ if and only if $z \in { \mathcal { F } } _ { 1 }$ , establishing part (i).

For part (ii), evaluate the spread at $\widehat { m } _ { \sigma } = \mu + \sigma z :$

$$
\begin{array} { r l } & { w _ { \sigma } ( \widehat { m } _ { \sigma } ) = a _ { \sigma } ( \widehat { m } _ { \sigma } ) - b _ { \sigma } ( \widehat { m } _ { \sigma } ) } \\ & { \qquad = ( \mu + \sigma \mathbb { E } \left[ X \mid X > z \right] ) - ( \mu + \sigma \mathbb { E } \left[ X \mid X < z \right] ) } \\ & { \qquad = \sigma \left( \mathbb { E } \left[ X \mid X > z \right] - \mathbb { E } \left[ X \mid X < z \right] \right) = \sigma \cdot w _ { 1 } ( z ) , } \end{array}
$$

which concludes the proof.

Since every PSNE mid-price is, in particular, a fixed point of the corresponding midpoint operator (Section 3), Proposition 13 applies to it directly: the equilibrium bid–ask spread at any PSNE $\mu + \sigma z$ of the $V _ { \sigma } { \mathrm { - m a r k e t } }$ equals $\sigma w _ { 1 } ( z )$ , growing linearly in $\sigma .$

To put this into perspective, while in information-based models, like Glosten and Milgrom (1985), the bid–ask spread widens primarily as a defense mechanism against a higher proportion of informed traders, here the spread widens strictly due to the amplification of fundamental risk $\sigma ,$ even in a framework where the underlying information structure remains fixed. Since every competing maker responds to the same volatility shock in the same way, this widening is synchronized across the market, producing exactly the kind of correlated price movement that recent tests for algorithmic collusion are designed to catch. Hartline et al. (2024) propose flagging collusion directly from the sequence of played prices, while Hartline et al. (2025) show that such price-based tests can be confounded whenever non-collusive forces already drive agents toward the same correlated play. Our result supplies exactly one such force: volatility-driven spread widening is a benign, competitive source of precisely the price patterns these tests are built to flag.

## 4. Learning

We ask whether adaptive makers can recover the competitive price structure without knowing their payof functions. Continuous-action bandit guarantees typically require smooth payofs (Cesa-Bianchi et al., 2023, 2025), whereas best-quote allocation makes ours discontinuous. We therefore restrict quotes to a price grid with tick size $\Delta ,$ giving rise to the grid game $\mathcal { G } _ { \Delta }$ , and show that the market prices of its correlated equilibria, and hence the time-averaged market prices of no-internal-regret learners, approach the Nash price envelope of the continuous game as $\Delta \to 0$

For simplicity, we focus on two makers and normalize the support of $V$ to $[ 0 , 1 ]$ , but neither restriction is essential.<sup>3</sup> As anticipated in Section $^ { 3 , }$ this analysis requires additional regularity.

Assumption 14 (Regularity). The random variable V has a continuous density $f > 0$ on $( 0 , 1 )$ , and the zero-profit quotes of Definition $\it 4$ satisfy $a ^ { \prime } ( m ) + b ^ { \prime } ( m ) \leq 2$ for every $m \in ( 0 , 1 )$

This assumption holds throughout the rest of this section. The first condition makes the zero-profit quotes continuously diferentiable, with derivatives

$$
b ^ { \prime } ( m ) = \frac { f ( m ) } { F ( m ) } ( m - b ( m ) ) , \qquad a ^ { \prime } ( m ) = \frac { f ( m ) } { 1 - F ( m ) } ( a ( m ) - m ) ,
$$

and it also ensures $b ^ { \prime } ( m ) , a ^ { \prime } ( m ) > 0$ , making the zero-profit quotes strictly increasing. The second condition bounds how fast H can move: $H ^ { \prime } ( m ) = ( a ^ { \prime } ( m ) + b ^ { \prime } ( m ) ) / 2 \leq 1$ , so $G ( m ) : = m - H ( m )$ has $G ^ { \prime } ( m ) \geq 0$ and is nondecreasing.

Proposition 15 (Structure of Equilibria under Regularity). (i) The fixed-point set $\mathcal { F }$ is a nonempty closed interval $[ m ^ { - } , m ^ { + } ]$ ; and $( i i )$ every $m ^ { * } \in { \mathcal { F } }$ is a PSNE mid-price, with market prices $( b ( m ^ { * } ) , a ( m ^ { * } ) )$

Proof sketch. For (i), F is the zero set of G, which is continuous and nondecreasing; $\mathcal { F }$ is nonempty by Theorem 5, hence it is a closed interval. For (ii), at any $m ^ { * } \in { \mathcal { F } }$ , the same monotonicity, driven by $a ^ { \prime } ( m ) + b ^ { \prime } ( m ) \leq 2$ , rules out both one-sided and two-sided deviations from $( b ( m ^ { * } ) , a ( m ^ { * } ) )$ ), so $m ^ { * }$ is a PSNE mid-price. The full proof is given in Section A.2.

Define $b ^ { \pm } : = b ( m ^ { \pm } )$ and $a ^ { \pm } : = a ( m ^ { \pm } )$ . As $m ^ { * }$ ranges over $\mathcal { F } = [ m ^ { - } , m ^ { + } ]$ , the market prices $( b ( m ^ { * } ) , a ( m ^ { * } ) )$ trace out the Nash price pairs, a monotone curve from $( b ^ { - } , a ^ { - } )$ to $( b ^ { + } , a ^ { + } )$ , whose axis-aligned envelope we call the Nash price envelope:

$$
\begin{array} { r } { \mathcal { R } ^ { \mathrm { N E } } : = [ b ^ { - } , b ^ { + } ] \times [ a ^ { - } , a ^ { + } ] . } \end{array}
$$

## 4.1. Zero-Profit Responses

While $a ( \cdot )$ and $b ( \cdot )$ tie both zero-profit quotes to a common mid-price, we now fix the market quote on one side and ask which quote on the other side breaks even.

Definition 16 (Zero-Profit Responses). For any market bid $B \in [ 0 , 1 ) , \alpha ( B )$ is the unique solution $x \in ( B , 1 ) \ o f x = a ( ( x + B ) / 2 )$ . Symmetrically, for any market ask $A \in ( 0 , 1 ]$ 2 $\beta ( A )$ is the unique solution $x \in ( 0 , A )$ of $x = b ( ( A + x ) / 2 )$ . We extend them by $\alpha ( 1 ) : = 1$ and $\beta ( 0 ) : = 0$ , consistently with their limits.

In words, $\alpha ( B )$ is the only market ask that earns zero expected profit against the market bid B, given the mid-price it induces; $\beta ( A )$ is its bid-side counterpart. Both are well defined, since $x \mapsto x - a ( ( x + B ) / 2 )$ is strictly increasing, negative at $x = B$ and positive at $x = 1$ and symmetrically for $\beta ( A ) . ^ { 4 }$ By the implicit function theorem, they are $C ^ { 1 }$ on $[ 0 , 1 )$ and $( 0 , 1 ]$ , respectively, with

$$
\alpha ^ { \prime } ( B ) = \frac { a ^ { \prime } ( m ) } { 2 - a ^ { \prime } ( m ) } \qquad \mathrm { a n d } \qquad \beta ^ { \prime } ( A ) = \frac { b ^ { \prime } ( m ) } { 2 - b ^ { \prime } ( m ) }
$$

at the induced mid-prices $m = ( B + \alpha ( B ) ) / 2$ and $m = ( A + \beta ( A ) ) / 2$ . Hence, both are strictly increasing and Lipschitz on every compact subinterval of these intervals.

At any $m ^ { * } \in { \mathcal { F } }$ , since $m ^ { * } = H ( m ^ { * } )$ , the responses recover the equilibrium quotes: $\alpha ( b ( m ^ { * } ) ) = a ( m ^ { * } )$ and $\beta ( a ( m ^ { * } ) ) = b ( m ^ { * } )$ , and in particular $\alpha ( b ^ { \pm } ) = a ^ { \pm }$ and $\beta ( a ^ { \pm } ) = b ^ { \pm }$ Composing them yields the following property, proved in Section A.2.

Lemma 17. For every market ask $A \in ( 0 , 1 ]$

$$
\alpha ( \beta ( A ) ) \left\{ { \begin{array} { l l } { { > A } } & { { i f A < a ^ { - } , } } \\ { { = A } } & { { i f A \in [ a ^ { - } , a ^ { + } ] , } } \\ { { < A } } & { { i f A > a ^ { + } . } } \end{array} } \right.
$$

That is, if the bid side responds to an ask A with $\beta ( A )$ , and the ask side responds in turn with $\alpha ( \beta ( A ) )$ , the ask stays at A when $A \in [ a ^ { - } , a ^ { + } ]$ and otherwise moves strictly toward this interval; the same holds for bids, with $\beta$ ◦ α and $[ b ^ { - } , b ^ { + } ]$

## 4.2. The Grid Game

Fix a tick size $\Delta > 0$ with $1 / \Delta \in \mathbb { N }$ , and let $\Gamma _ { \Delta } : = \{ 0 , \Delta , 2 \Delta , \ldots , 1 \}$ . The grid game $\mathcal { G } _ { \Delta }$ is the market-making game of Section 2 with each maker’s strategy space restricted to

$$
S _ { i } ^ { \Delta } : = \big \{ ( b , a ) \in \Gamma _ { \Delta } ^ { 2 } \ \big | \ b \leq a \big \} \subset S _ { i } \qquad \mathrm { a n d } \qquad \mathcal { S } ^ { \Delta } : = \prod _ { i \in \mathcal { N } } S _ { i } ^ { \Delta } ,
$$

so that best quotes, mid-prices and payofs are exactly as defined there. The next two lemmas identify profitable deviations on the grid: a best ask more than one tick below $\alpha ( B )$ is profitably raised by one tick, while a rival ask suficiently many ticks above $\alpha ( B )$ is profitably undercut by one tick; symmetrically for bids.

Lemma 18. Let $\mathbf { s } = ( \mathbf { b } , \mathbf { a } ) \in \mathcal { S } ^ { \Delta } , \ A : = a ^ { * } ( \mathbf { a } )$ , and $B : = b ^ { * } ( { \mathbf { b } } )$ . If $a _ { i } = A$ and $a _ { i } +$ $\Delta < \alpha ( B )$ , then raising $a _ { i }$ to $a _ { i } + \Delta$ strictly increases $u _ { i }$ . Symmetrically, $i f b _ { i } = B$ and $b _ { i } - \Delta > \beta ( A )$ , then lowering $b _ { i }$ to $b _ { i } - \Delta$ strictly increases $u _ { i }$

Lemma 19. Fix $\overline { { B } } < 1$ and $\underline { { A } } > 0$ . There exists $C _ { 0 } > 1$ , depending only on $F , { \overline { { B } } }$ and $\underline { { A } } ,$ such that the following holds. Let $\mathbf { s } = ( \mathbf { b } , \mathbf { a } ) \in \mathcal { S } ^ { \Delta } , A : = a ^ { * } ( \mathbf { a } ) , B : = b ^ { * } ( \mathbf { b } )$ , and let i be a maker. If $B \leq \overline { { B } }$ and $\widetilde { a } \in \Gamma _ { \Delta }$ satisfies $a _ { - i } \leq \widetilde { a } \leq a _ { i }$ and $\widetilde { a } \ge \alpha ( B ) + C _ { 0 } \Delta$ , then lowering $a _ { i }$ to $\widetilde { a } - \Delta$ weakly increases $\overset { u _ { i } } { \underset { \sim } { \sim } }$ , strictly if a<sub>−</sub> $_ { - i } \geq \widetilde { a } - \Delta$ . Symmetrically, if $A \geq \underline { { A } }$ and $\widetilde { b } \in \Gamma _ { \Delta }$ satisfies $b _ { i } \leq \widetilde { b } \leq b _ { - i }$ and $\bar { b } \le \beta ( A ) - C _ { 0 } \Delta$ , then raising $b _ { i }$ to $\widetilde { b } + \Delta$ weakly increases $u _ { i }$ strictly if $b _ { - i } \leq \bar { b } + \Delta$

## 4.3. Correlated Equilibria

Since the time-averaged play of no-internal-regret learners approaches the set of correlated equilibria (Foster and Vohra, 1997), we study the latter.

Definition 20 (Correlated Equilibrium). For $\delta \geq 0$ , a distribution $\pi$ over $S ^ { \Delta }$ is a δ-correlated equilibrium (δ-CE) of $\mathcal { G } _ { \Delta }$ if, for every maker $i \in \mathcal N$ and every pair $s _ { i } , \widetilde { s } _ { i } \in \mathcal { S } _ { i } ^ { \Delta }$

$$
\sum _ { \mathbf { s } _ { - i } } \pi ( s _ { i } , \mathbf { s } _ { - i } ) \left( u _ { i } ( \widetilde { s } _ { i } , \mathbf { s } _ { - i } ) - u _ { i } ( s _ { i } , \mathbf { s } _ { - i } ) \right) \leq \delta .
$$

A 0-CE is simply called a correlated equilibrium (CE).

For a CE π, let

$$
a ^ { \mathrm { m i n } } ( \pi ) : = \operatorname* { m i n } _ { \mathbf { s } \in \operatorname { s u p p } ( \pi ) } a ^ { * } ( \mathbf { a } ) \qquad { \mathrm { a n d } } \qquad a ^ { \mathrm { m a x } } ( \pi ) : = \operatorname* { m a x } _ { \mathbf { s } \in \operatorname { s u p p } ( \pi ) } a ^ { * } ( \mathbf { a } )
$$

denote the extremal market asks on its support, and define $b ^ { \operatorname* { m i n } } ( \pi )$ and $b ^ { \mathrm { m a x } } ( \pi )$ similarly.

Lemma 21. There exist $C _ { 0 } > 1$ and $\Delta _ { 0 } > 0$ , depending only on $F _ { ; }$ , such that, for every $\Delta \le \Delta _ { 0 }$ , every $C E \pi \ o f \mathcal { G } _ { \Delta }$ satisfies

$$
\begin{array} { r l } & { a ^ { \mathrm { m i n } } ( \pi ) \geq \alpha ( b ^ { \mathrm { m i n } } ( \pi ) ) - \Delta , \qquad b ^ { \mathrm { m a x } } ( \pi ) \leq \beta ( a ^ { \mathrm { m a x } } ( \pi ) ) + \Delta , } \\ & { a ^ { \mathrm { m a x } } ( \pi ) \leq \alpha ( b ^ { \mathrm { m a x } } ( \pi ) ) + C _ { 0 } \Delta , \qquad b ^ { \mathrm { m i n } } ( \pi ) \geq \beta ( a ^ { \mathrm { m i n } } ( \pi ) ) - C _ { 0 } \Delta . } \end{array}
$$

In words, up to one tick below and $C _ { 0 }$ ticks above, the market asks of any CE are bracketed by the zero-profit responses to its extremal bids; symmetrically for bids.

## 4.4. Grid Correlated Equilibria Approach Nash

For a profile s, let $D _ { \mathrm { e n v } } ( \mathbf { s } ) : = \operatorname* { m i n } _ { ( x , y ) \in { \mathcal { R } } ^ { \mathrm { N E } } } \bigl ( | b ^ { * } ( \mathbf { b } ) - x | + | a ^ { * } ( \mathbf { a } ) - y | \bigr )$ be the $\ell _ { \mathrm { 1 } } { \mathrm { - d i s t a n c e } }$ between its market prices and the Nash price envelope, and define the grid approximation error

$$
\varepsilon _ { \Delta } : = \operatorname* { s u p } _ { \pi } \operatorname* { m a x } _ { \mathbf { s } \in \mathrm { s u p p } ( \pi ) } D _ { \mathrm { e n v } } ( \mathbf { s } ) ,
$$

where $\pi$ ranges over all CEs of $\mathcal { G } _ { \Delta }$ .

Theorem 22 (Convergence of Grid Correlated Equilibria). $\varepsilon _ { \Delta }  0 ~ a s \Delta  0$

Proof sketch. Combining the bounds of Lemma 21, and using that α is Lipschitz, the extremal asks of any CE satisfy the two-cycle condition $\alpha ( \beta ( A ) ) = A$ up to an error of order $\Delta$ . By Lemma $^ { 1 7 , }$ this forces their limit points into $[ a ^ { - } , a ^ { + } ]$ as $\Delta \to 0$ , and the continuity of $\beta$ carries the conclusion over to the extremal bids and $[ b ^ { - } , b ^ { + } ]$ . Since every supported market price lies between the extremal ones, the convergence is uniform over CEs. The full proof is given in Section A.2. 7

In particular, if $\mathcal { F } = \{ m ^ { * } \}$ , then $\mathcal { R } ^ { \mathrm { N E } }$ collapses to the PSNE prices $( b ( m ^ { * } ) , a ( m ^ { * } ) )$ , and the market prices of all grid CEs converge uniformly to them.

## 4.5. No-Internal-Regret Learning

Theorem 22 extends to approximate CEs, and hence to learning dynamics.

Lemma 23 (Approximate Correlated Equilibria). Fix $\Delta > 0$ with $1 / \Delta \in \mathbb { N }$ $I f \pi _ { n }$ is a $\delta _ { n } – C E \ o f \mathcal { G } _ { \Delta }$ for every n, with $\delta _ { n } \to 0$ , then

$$
\begin{array} { r } { \operatorname* { l i m } _ { n } \underset { { n } } { \operatorname* { s u p } } \mathbb { E } _ { \mathbf { s } \sim \pi _ { n } } [ D _ { \mathrm { e n v } } ( \mathbf { s } ) ] \leq \varepsilon _ { \Delta } \ . } \end{array}
$$

In the repeated grid game, at each round t, the makers’ learning algorithms select a profile $\mathbf { S } _ { t } = ( S _ { t , 1 } , S _ { t , 2 } ) \in S ^ { \Delta }$ , the value $V _ { t } \sim F$ is drawn independently of $\mathbf { S } _ { t }$ and of the past, and each maker observes only its realized payof $r _ { i } ( \mathbf { S } _ { t } ; V _ { t } , d _ { t } )$ , where $d _ { t }$ is the trader’s order. We use uppercase letters because both the profile and the value are random: the former through the learners’ randomization and its dependence on past feedback. Maker i has vanishing internal regret if, for every pair $s _ { i } , \widetilde { s } _ { i } \in \mathcal { S } _ { i } ^ { \Delta }$ ，

$$
R _ { T , i } ( s _ { i } , \widetilde { s } _ { i } ) : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left( u _ { i } ( \widetilde { s } _ { i } , \mathbf { S } _ { t , - i } ) - u _ { i } ( \mathbf { S } _ { t } ) \right) \mathbb { I } _ { \{ S _ { t , i } = s _ { i } \} } \right]
$$

satisfies lim su $) _ { T } R _ { T , i } ( s _ { i } , \widetilde { s } _ { i } ) \leq 0$ , as guaranteed, $\mathrm { e . g . }$ , by regret-matching procedures (Hart and Mas-Colell, 2000) and swap-regret minimizers (Blum and Mansour, 2007).

Corollary 24 (No-Internal-Regret Learning). Fix $\Delta > 0$ with $1 / \Delta \in \mathbb { N }$ . If both makers have vanishing internal regret, then

$$
\operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ D _ { \mathrm { e n v } } ( \mathbf { S } _ { t } ) \right] \leq \varepsilon _ { \Delta } ,
$$

and hence, by Theorem 22, the left-hand side vanishes as $\Delta \to 0$

Proof. Since $V _ { t }$ is independent of $\mathbf { S } _ { t } , R _ { T , i }$ coincides with the internal regret on realized payofs. Let $\bar { \pi } _ { T }$ be the average of the laws of $\mathbf { S } _ { 1 } , \ldots , \mathbf { S } _ { T }$ . Each sum in Definition 20 for $\bar { \pi } _ { T }$ equals the corresponding $R _ { T , i } ( s _ { i } , \widetilde { s } _ { i } )$ , so π¯<sub>T</sub> is a $\delta _ { T ^ { - } } \mathrm { C E }$ , with $\delta _ { T }  0$ the largest positive part of these regrets. By linearity, $\mathbb { E } _ { \mathbf { s } \sim \bar { \pi } _ { T } } [ D _ { \mathrm { e n v } } ( \mathbf { s } ) ]$ is the time average in the statement, and Lemma 23 concludes.

To summarize, on fine grids every CE, and hence the time-averaged play of no-internalregret learners, keeps market prices close to the envelope $\mathcal { R } ^ { \mathrm { N E } }$ . The envelope is only an outer bound: our results do not exclude that learners occasionally play price pairs away from the Nash curve, such as $( b ^ { - } , a ^ { + } )$ , but they cannot settle on one. Indeed, if their time-averaged play converges to a single profile, the point mass on it is a CE of $\mathcal { G } _ { \Delta }$ , whose extremal prices coincide, so Lemma 21 gives $a ^ { * } ( { \mathbf { a } } ) \approx \alpha ( b ^ { * } ( { \mathbf { b } } ) )$ and $b ^ { * } ( \mathbf { b } ) \approx \beta ( a ^ { * } ( \mathbf { a } ) )$ ), and Lemma 17 forces its market prices close to the Nash curve.

## 5. Simulations

We illustrate the theoretical results with two market makers running Exp3 inside the Blum– Mansour reduction to internal regret (Cesa-Bianchi and Lugosi, 2006; Blum and Mansour, 2007), against a Gaussian asset value with mean 0.5 clamped<sup>5</sup> to [0, 1]. Every expert uses a learning rate $\eta = 0 . 0 5$

Figure 4 (left) fixes $\sigma = 0 . 1$ and sweeps the tick size $\Delta .$ , learning each grid for $T = 1 0 0 , 0 0 0$ rounds and averaging over 100 runs. $\mathrm { A s } \ \Delta$ shrinks, the learned market bid and ask move toward the Nash price envelope; the non-monotonicity is not surprising, since even the exact equilibria of the grid game need not vary smoothly in $\Delta$ at this resolution. Figure 4 $( \mathrm { r i g h t } )$ fixes $\Delta = 0 . 1$ and sweeps $\sigma ,$ learning each value for $T = 1 , 5 0 0 , 0 0 0$ rounds and averaging over 15 runs. The learned spread grows monotonically and close to linearly with $\sigma ,$ , consistent with Proposition 13.

![](images/2e383a8230f4f3498cce79603bceec0935cd06bb83b3c744ddea16865373c77d.jpg)

![](images/b463d026f114067c19b7316b267b3063b6cfa30e0f8994674a79cc1808ec6cc5.jpg)  
Figure 4: Simulations of two learning market makers running swap-regret minimizers. $L e f t { \mathrm { : } }$ $\mathrm { A s } ~ \Delta$ shrinks, the learned market ask (orange) and bid (blue) move toward the continuous Nash envelope (dotted lines), see Corollary 24. Right: At a fixed tick size, the learned spread (purple) grows monotonically with the standard deviation $\sigma$ of the asset value, tracking the theoretical spread (black) while sitting consistently above it, see Proposition 13. Error bars are the standard deviation of the learned quotes across independent replications.

## 6. Conclusion and Future Work

We modeled market making as a game in which makers post bid–ask quotes and an inelastic trader, who knows the asset value, always trades at the best quotes, buying whenever the value exceeds the market mid-price and selling otherwise. In this game, pure-strategy Nash equilibria always exist, and all equilibrium mid-prices lie between two extremal ones. These equilibria also pin down how competitive spreads depend on risk: within locationscale families, they grow linearly with the volatility of the asset. Finally, we proved that, under an additional regularity condition, no-internal-regret learners recover this competitive benchmark, with time-averaged market prices that lie close to the Nash price envelope on a fine price grid.

This benchmark clarifies what competitive pricing looks like in adaptive markets. In particular, when the asset becomes more volatile, the spreads of all makers widen together, with no coordination among them. Since algorithmic-collusion tests are designed to flag exactly this kind of simultaneous widening across makers, they should control for volatility, which is nontrivial in markets with incomplete information, where volatility itself must be estimated.

Future work includes a finer analysis of learning: we only showed that time-averaged market prices approach the Nash price envelope, leaving open whether learners concentrate on the Nash curve and at what rate. Other directions are to weaken the regularity required in Section 4, to characterize mixed-strategy and correlated equilibria in the general model, and to extend the model to inventory, latency, heterogeneous order flow and time-varying fundamentals.

## Acknowledgments

LF acknowledges support by the MUR PRIN grant 2022EKNE5K (Learning in Markets and Society), by the FAIR (Future Artificial Intelligence Research) project, funded by the NextGenerationEU program within the PNRR-PE-AI scheme (M4C2, investment 1.3, line on Artificial Intelligence), and by the EU Horizon CL4-2022-HUMAN-02 research and innovation action under grant agreement 101120237, project ELIAS (European Lighthouse of AI for Sustainability).

MO gratefully acknowledges Intesa Sanpaolo Innovation Center for partially funding his doctoral fellowship, as well as the financial support from the EU Horizon CL4-2022- HUMAN-02 research and innovation action under grant agreement 101120237, project ELIAS (European Lighthouse of AI for Sustainability).

## References

Jacob Abernethy and Satyen Kale. Adaptive market making via online learning. In Proceedings ofthe 27th International Conference on Neural Information Processing Systems, pages 2058–2066. Curran Associates Inc., 2013.

Juliette Achddou, Olivier Capp´e, and Aur´elien Garivier. Fast rate learning in stochastic first price bidding. In Proceedings of the 13th Asian Conference on Machine Learning, volume 157 of Proceedings of Machine Learning Research, pages 1754–1769. PMLR, 2021.

V. Ravi Anshuman and Avner Kalay. Market making with discrete prices. The Review of Financial Studies, 11(1):81–109, 1998.

Eshwar Ram Arunachaleswaran, Natalie Collina, Sampath Kannan, Aaron Roth, and Juba Ziani. Algorithmic collusion without threats. In Proceedings of the 16th Innovations in Theoretical Computer Science Conference, volume 325 of Leibniz International Proceedings in Informatics (LIPIcs), pages 10:1–10:21. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik, 2025.

Kerry Back and Shmuel Baruch. Strategic liquidity provision in limit order markets. Econometrica, 81(1):363–392, 2013.

Martino Banchio and Andrzej Skrzypacz. Artificial intelligence and auction design. In Proceedings of the 23rd ACM Conference on Economics and Computation, pages 30–31. Association for Computing Machinery, 2022.

Joseph Bertrand. Review of “Theorie mathematique de la richesse sociale” and of “Recherches sur les principes mathematiques de la theorie des richesses”. Journal des savants, 67:499, 1883.

Bruno Biais, David Martimort, and Jean-Charles Rochet. Competing mechanisms in a common value environment. Econometrica, 68(4):799–837, 2000.

Avrim Blum and Yishay Mansour. From external to internal regret. Journal of Machine Learning Research, 8(47):1307–1324, 2007.

Christian Bogmans, Andrea Pescatori, Ivan Petrella, Ervin Prifti, and Martin Stuermer. The power of prices: How fast do commodity markets adjust to shocks? IMF Working Paper 2024/077, International Monetary Fund, 2024.

Emilio Calvano, Giacomo Calzolari, Vincenzo Denicolo, and Sergio Pastorello. Artificial intelligence, algorithmic pricing, and collusion. American Economic Review, 110(10): 3267–3297, 2020.

Alvaro Cartea, Patrick Chang, and Jos´e Penalva. Algorithms and supracompetitive prices <sup>´</sup> in electronic markets: The impact of tick size. SSRN Working Paper 4105954, 2022.

Nicol\`o Cesa-Bianchi and G´abor Lugosi. Prediction, Learning, and Games. Cambridge University Press, Cambridge, 2006.

Nicol\`o Cesa-Bianchi, Tommaso Cesari, and Vianney Perchet. Dynamic pricing with finitely many unknown valuations. In Proceedings of the 30th International Conference on Algorithmic Learning Theory, volume 98 of Proceedings of Machine Learning Research, pages 247–273. PMLR, 2019.

Nicol\`o Cesa-Bianchi, Tommaso R. Cesari, Roberto Colomboni, Federico Fusco, and Stefano Leonardi. Repeated bilateral trade against a smoothed adversary. In Proceedings of the 36th Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 1095–1130. PMLR, 2023.

Nicol\`o Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Federico Fusco, and Stefano Leonardi. The role of transparency in repeated first-price auctions with unknown valuations. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 225– 236. Association for Computing Machinery, 2024.

Nicol\`o Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Luigi Foscari, and Vinayak Pathak. Market making without regret. In Proceedings of the 38th Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 799–837. PMLR, 2025.

Jean-Edouard Colliard, Thierry Foucault, and Stefano Lovo. Algorithmic pricing and liquidity in securities markets. HEC Research Papers Series 1459, HEC Paris, 2022.

Rama Cont and Wei Xiong. Dynamics of market making algorithms in dealer markets: Learning and tacit collusion. Mathematical Finance, 34(2):467–521, 2024.

Thomas E. Copeland and Dan Galai. Information efects on the bid-ask spread. The Journal of Finance, 38(5):1457–1469, 1983.

Antoine Augustin Cournot. Recherches sur les principes mathematiques de la theorie des richesses. L. Hachette, Paris, 1838.

Partha Dasgupta and Eric Maskin. The existence of equilibrium in discontinuous economic games, I: Theory. The Review of Economic Studies, 53(1):1–26, 1986.

Krishnendu Ghosh Dastidar. On the existence of pure strategy Bertrand equilibrium. Economic Theory, 5(1):19–32, 1995.

J¨urgen Dennert. Price competition between market makers. The Review of Economic Studies, 60(3):735–751, 1993.

David Easley and Maureen O’Hara. Price, trade size, and information in securities markets. Journal of Financial Economics, 19(1):69–90, 1987.

Dean P. Foster and Rakesh V. Vohra. Calibrated learning and correlated equilibrium. Games and Economic Behavior, 21(1–2):40–55, 1997.

Sumitra Ganesh, Nelson Vadori, Mengda Xu, Hua Zheng, Prashant Reddy, and Manuela Veloso. Reinforcement learning for market making in a multi-agent dealer market. arXiv preprint arXiv:1911.05892, 2019.

Lawrence R. Glosten. Is the electronic open limit order book inevitable? The Journal of Finance, 49(4):1127–1161, 1994.

Lawrence R. Glosten and Paul R. Milgrom. Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. Journal of Financial Economics, 14(1): 71–100, 1985.

Yanjun Han, Zhengyuan Zhou, and Tsachy Weissman. Optimal no-regret learning in repeated first-price auctions. arXiv preprint arXiv:2003.09795, 2024.

Karsten T. Hansen, Kanishka Misra, and Mallesh M. Pai. Frontiers: Algorithmic collusion: Supra-competitive prices via independent algorithms. Marketing Science, 40(1):1–12, 2021.

Joseph E. Harrington. Developing competition law for collusion by autonomous artificial agents. Journal of Competition Law & Economics, 14(3):331–363, 2018.

Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127–1150, 2000.

Jason D. Hartline, Sheng Long, and Chenhao Zhang. Regulation of algorithmic collusion. In Proceedings of the 2024 Symposium on Computer Science and Law, pages 98–108. Association for Computing Machinery, 2024.

Jason D. Hartline, Chang Wang, and Chenhao Zhang. Regulation of algorithmic collusion, refined: Testing pessimistic calibrated regret. In Proceedings of the 2025 Symposium on Computer Science and Law, pages 108–120. Association for Computing Machinery, 2025.

Thomas Ho and Hans R. Stoll. Optimal dealer pricing under transactions and return uncertainty. Journal of Financial Economics, 9(1):47–73, 1981.

Timo Klein. Autonomous algorithmic collusion: Q-learning under sequential pricing. The RAND Journal of Economics, 52(3):538–558, 2021.

Albert S. Kyle. Continuous auctions and insider trading. Econometrica, 53(6):1315–1335, 1985.

Davide Maran and Marcello Restelli. Online market making and the value of observing the order book. arXiv preprint arXiv:2605.19584, 2026.

Paul Milgrom and John Roberts. Rationalizability, learning, and equilibrium in games with strategic complementarities. Econometrica, 58(6):1255–1277, 1990.

Paul Milgrom and Nancy Stokey. Information, trade and common knowledge. Journal of Economic Theory, 26(1):17–27, 1982.

Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2023.

Philip J. Reny. On the existence of pure and mixed strategy Nash equilibria in discontinuous games. Econometrica, 67(5):1029–1056, 1999.

Thomas Spooner, John Fearnley, Rahul Savani, and Andreas Koukorinis. Market making via reinforcement learning. In Proceedings of the 17th International Conference on Autonomous Agents and Multiagent Systems, pages 434–441. International Foundation for Autonomous Agents and Multiagent Systems, 2018.

Gilles Stoltz and G´abor Lugosi. Learning correlated equilibria in games with compact sets of strategies. Games and Economic Behavior, 59(1):187–208, 2007.

Alfred Tarski. A lattice-theoretical fixpoint theorem and its applications. Pacific Journal of Mathematics, 5(2):285–309, 1955.

Matilde Tullii, Solenne Gaucher, Nadav Merlis, and Vianney Perchet. Improved algorithms for contextual dynamic pricing. arXiv preprint arXiv:2406.11316, 2025.

Xavier Vives. Nash equilibrium with strategic complementarities. Journal of Mathematical Economics, 19(3):305–321, 1990.

## Appendix A. Appendix: Omitted Proofs

In this appendix, we provide the complete proofs of all technical results presented in the main text.

## A.1. Proofs of Section 3

Lemma 6. Let ${ \widehat { m } } \in { \mathcal { F } }$ . If the associated zero-profit profile $\mathbf { s } ( \widehat { m } )$ admits a profitable unilateral deviation, then it admits either a profitable ask undercut with bid withdrawal or a profitable bid raise with ask withdrawal.

Proof. Let $\widetilde { s } _ { i } : = ( \widetilde { b } _ { i } , \widetilde { a } _ { i } )$ be a profitable deviation against $\mathbf { s } _ { - i } ( \widehat { m } )$ , inducing mid-price z. Since total profit is the sum of ask-side and bid-side profits, at least one side must yield strictly positive expected profit. Suppose first that the ask side is strictly profitable. Then $\widetilde { a } _ { i } < a ( \widehat { m } )$ : if $\widetilde { a } _ { i } > a ( \widehat { m } )$ , maker i captures no buy orders, while if $\widetilde { \boldsymbol { a } } _ { i } = \boldsymbol { a } ( \widehat { \boldsymbol { m } } )$ , the market ask is unchanged and the market bid can only rise, so $z \geq { \widehat { m } }$ and its ask-side payof $\begin{array} { r } { \frac { 1 } { N } \mathbb { P } ( V > z ) ( a ( \widehat { m } ) - a ( z ) ) } \end{array}$ is non-positive. Hence, maker i alone quotes the market ask, the midpoint is

$$
z = \frac { \widetilde { a } _ { i } + \operatorname* { m a x } ( \widetilde { b } _ { i } , { b ( \widehat { m } ) } ) } { 2 } ,
$$

and

$$
u _ { i } ^ { \mathrm { a s k } } \big ( \widetilde { s } _ { i } , { \mathbf { s } _ { - i } } ( \widehat { m } ) \big ) = \mathbb { P } ( V > z ) \left( \widetilde { a } _ { i } - a ( z ) \right) > 0 ,
$$

so that, in particular, the margin satisfies $\widetilde { a } _ { i } - a ( z ) > 0$

Now consider the ask undercut $( 0 , \widetilde { a } _ { i } )$ , which keeps the same ask quote and withdraws the bid. Under this single-sided deviation, the market bid reverts to $b ( \widehat { m } )$ , shifting the induced midpoint to

$$
y : = \frac { \widetilde { a } _ { i } + b ( \widehat { m } ) } { 2 } \leq z ,
$$

since $b ( \widehat { m } ) \leq \operatorname* { m a x } ( \widetilde { b } _ { i } , b ( \widehat { m } ) )$ . Given that $a ( \cdot )$ is nondecreasing, $y \le z$ expands the margin,

$$
\widetilde { a } _ { i } - a ( y ) \geq \widetilde { a } _ { i } - a ( z ) > 0 ,
$$

as well as the execution probability,

$$
\mathbb { P } ( V > y ) \geq \mathbb { P } ( V > z ) .
$$

Since maker i has withdrawn from the bid side, its bid profit under $( 0 , \widetilde { a } _ { i } )$ is zero. Total profit under the new deviation therefore satisfies

$$
\begin{array} { r l } & { u _ { i } ( ( 0 , \widetilde { a } _ { i } ) , \mathbf { s } _ { - i } ( \widehat { m } ) ) = \mathbb { P } ( V > y ) \left( \widetilde { a } _ { i } - a ( y ) \right) } \\ & { \qquad \quad \geq \mathbb { P } ( V > z ) \left( \widetilde { a } _ { i } - a ( z ) \right) = u _ { i } ^ { \mathrm { a s k } } \big ( \widetilde { s } _ { i } , \mathbf { s } _ { - i } ( \widehat { m } ) \big ) > 0 . } \end{array}
$$

Hence, the ask undercut paired with bid withdrawal remains strictly profitable.

Symmetrically, if the bid side is strictly profitable, then $\widetilde { b } _ { i } > b ( \widehat { m } )$ , and withdrawing the ask, i.e., playing $( \widetilde { b } _ { i } , 1 )$ , shifts the midpoint from z to

$$
x : = \frac { a ( \widehat { m } ) + \widetilde { b } _ { i } } { 2 } \geq z ,
$$

which again enhances both margin and execution probability.

## Lemma 7. Let ${ \widehat { m } } \in { \mathcal { F } }$ . Then:

(i) For any $x > { \widehat { m } } ,$ , the unique bid raise inducing mid-price x is profitable if and only if $2 x - b ( x ) < a ( \widehat { m } )$ , which implies $H ( x ) > x$

(ii) For any $y < \widehat { m }$ , the unique ask undercut inducing mid-price y is profitable if and only $i f 2 y - a ( y ) > b ( \widehat { m } )$ , which implies $H ( y ) < y$

Proof. We first prove (i). Fix $x > { \widehat { m } }$ . Under a bid raise by maker i (paired with ask withdrawal), the prevailing market ask remains $a ( \widehat { m } )$ . The induced midpoint $x =$ $( a ( \widehat { m } ) + \widetilde { b } _ { i } ) / 2$ uniquely yields

$$
\widetilde { b } _ { i } = 2 x - a ( \widehat { m } ) ,
$$

which is strictly profitable if and only if

$$
b ( x ) - \widetilde { b } _ { i } > 0 \Longleftrightarrow 2 x - b ( x ) < a ( \widehat { m } ) .
$$

Substituting $b ( x ) = 2 H ( x ) - a ( x )$ yields

$$
2 \left( x - H ( x ) \right) + a ( x ) - a ( \widehat { m } ) < 0 .
$$

Since $x > \widehat { m }$ and $a ( \cdot )$ is nondecreasing, we have $a ( x ) - a ( \widehat { m } ) \geq 0$ . For the inequality to hold, it must be that

$$
2 \left( x - H ( x ) \right) < a ( \widehat { m } ) - a ( x ) \leq 0 ,
$$

which strictly requires $H ( x ) > x$

Symmetrically, for (ii), fix $y < \widehat { m }$ . Under an ask undercut by maker i (paired with bid withdrawal), the prevailing market bid remains $b ( \widehat { m } )$ . The induced midpoint $y = ( \widetilde { a } _ { i } + b ( \widehat { m } ) ) / 2$ uniquely determines the required ask quote:

$$
\widetilde { a } _ { i } = 2 y - b ( \widehat { m } ) .
$$

This undercut is strictly profitable if and only if

$$
\widetilde { a } _ { i } - a ( y ) > 0 \iff 2 y - a ( y ) > b ( \widehat { m } ) .
$$

Substituting $a ( y ) = 2 H ( y ) - b ( y )$ rewrites this profitability condition as

$$
2 \left( y - H ( y ) \right) + b ( y ) - b ( \widehat { m } ) > 0 .
$$

Since $y < \widehat { m }$ and $b ( \cdot )$ is nondecreasing, we have $b ( y ) - b ( \widehat { m } ) \leq 0$ . For the inequality to hold, it must be that

$$
2 \left( y - H ( y ) \right) > b ( \widehat { m } ) - b ( y ) \geq 0 ,
$$

which strictly requires $H ( y ) < y$

Lemma 8. Let $m _ { \operatorname* { m i n } } : = \operatorname* { m i n } \mathcal { F }$ and $m _ { \mathrm { m a x } } : = \operatorname* { m a x } \mathcal { F }$ . Then $( i )$ no profitable ask undercut exists at $m _ { \mathrm { m i n } } .$ , and $( i i )$ no profitable bid raise exists at $m _ { \mathrm { m a x } }$

Proof. By Knaster–Tarski, $\mathcal { F }$ is itself a complete lattice (Tarski, 1955), so $m _ { \mathrm { m i n } }$ and $m _ { \mathrm { m a x } }$ are well defined. It remains to rule out the corresponding one-sided deviation at each extremal point, which we do by contradiction.

We first prove (i). Suppose, by contradiction, that a profitable ask undercut exists at $m _ { \mathrm { m i n } }$ . By Lemma 7, the induced midpoint $y < m _ { \mathrm { m i n } }$ satisfies $H ( y ) < y$ . Since H is nondecreasing and $H ( \underline { { v } } ) > \underline { { v } }$ , for every $m \in [ \underline { { v } } , y ]$ we have

$$
\underline { { v } } < H ( \underline { { v } } ) \leq H ( m ) \leq H ( y ) < y .
$$

Hence, H maps the closed interval $[ \underline { { v } } , y ]$ into itself. Then, by the Knaster–Tarski fixed-point theorem, H admits a fixed point $\check { m } \in [ \underline { { v } } , y ]$ . Consequently, $\check { m } \in \mathcal { F }$ satisfies $\check { m } \leq y < m _ { \operatorname* { m i n } } =$ min ${ \mathcal { F } } _ { : }$ , a contradiction.

Symmetrically, for (ii), suppose a profitable bid raise exists at $m _ { \mathrm { m a x } }$ . By Lemma $^ { 7 , }$ the induced midpoint $x > m _ { \mathrm { m a x } }$ satisfies $H ( x ) > x$ . Since $H ( \overline { { v } } ) < \overline { { v } }$ , for every $m \in [ x , \overline { { v } } ]$ we have

$$
x < H ( x ) \leq H ( m ) \leq H ( \overline { { { v } } } ) < \overline { { { v } } } .
$$

Thus, H maps the closed interval $[ x , { \overline { { v } } } ]$ into itself, yielding a fixed point $\check { m } \in [ x , \overline { { v } } ]$ by Knaster–Tarski. Consequently, $\check { m } \in \mathcal { F }$ satisfies $\begin{array} { r } { \check { m } \geq x > m _ { \operatorname* { m a x } } = \operatorname* { m a x } \mathcal { F } } \end{array}$ , a contradiction.

Lemma 9. Let $m _ { 1 } , m _ { 2 } \in { \mathcal { F } }$ with $m _ { 1 } < m _ { 2 }$ . Then:

(i) If a profitable bid raise at $m _ { 1 }$ induces a mid-price $x > m _ { 2 }$ , then the bid raise inducing the same midpoint x at m<sub>2</sub> is also profitable.

(ii) If a profitable ask undercut at m<sub>2</sub> induces a mid-price $y < m _ { 1 }$ , then the ask undercut inducing the same midpoint y at m<sub>1</sub> is also profitable.

Proof. We first prove (i). Consider a profitable bid raise at $m _ { 1 }$ inducing a midpoint $x > m _ { 2 }$ . By Lemma 7, profitability implies

$$
2 x - b ( x ) < a ( m _ { 1 } ) .
$$

Since $m _ { 1 } < m _ { 2 }$ and the function $a ( \cdot )$ is nondecreasing, we have $a ( m _ { 1 } ) \leq a ( m _ { 2 } )$ , which directly yields

$$
2 x - b ( x ) < a ( m _ { 2 } ) .
$$

Since $x > m _ { 2 }$ , Lemma 7 guarantees that the bid raise inducing midpoint x at $m _ { 2 }$ is likewise strictly profitable.

Symmetrically, for (ii), a profitable ask undercut at $m _ { 2 }$ inducing midpoint $y \textless m _ { 1 }$ satisfies

$$
2 y - a ( y ) > b ( m _ { 2 } ) .
$$

Since $b ( \cdot )$ is nondecreasing and $m _ { 1 } < m _ { 2 }$ , we have $b ( m _ { 2 } ) \geq b ( m _ { 1 } )$ , implying

$$
2 y - a ( y ) > b ( m _ { 1 } ) .
$$

Since $y < m _ { 1 }$ , Lemma 7 confirms that the ask undercut inducing midpoint y at $m _ { 1 }$ is also strictly profitable. ■

## Theorem 10 (Existence of a Pure-Strategy Nash Equilibrium). Define

$$
{ \mathcal { F } } _ { B } : = \{ m \in { \mathcal { F } } \mid n o \ p r o f t a b l e \ b i d \ r a i s e \ e x i s t s \ a t \ m \}
$$

and $m ^ { - } : = \operatorname* { i n f } { \mathcal { F } _ { B } }$ . Then $\mathcal { F } _ { B } \neq \emptyset$ , and $\mathbf { s } ( m ^ { - } )$ is a pure-strategy Nash equilibrium.

Proof. Since V is continuous, $a ( \cdot ) , b ( \cdot )$ , and hence H are continuous, so $\mathcal { F }$ is closed and thus compact: it has minimum and maximum elements $m _ { \mathrm { m i n } }$ and $m _ { \mathrm { m a x } }$ , and contains the infimum and supremum of any of its nonempty subsets. Lemma 8 gives $m _ { \operatorname* { m a x } } \in { \mathcal { F } } _ { B }$ , so ${ \mathcal { F } } _ { B } \neq { \mathcal { O } } ;$ since ${ \mathcal { F } } _ { B } \subseteq { \mathcal { F } }$ is nonempty, $m ^ { - } = \operatorname* { i n f } { \mathcal { F } } _ { B }$ exists and lies in $\mathcal { F }$ . We show $m ^ { - } \in \mathcal { F } _ { B }$ Suppose not: a profitable bid raise at m<sup>−</sup> induces some mid-price $x > m$ <sup>−</sup>. Since m<sup>−</sup> is a lower bound of $\mathcal { F } _ { B }$ but x is not (else x would be a lower bound exceeding the infimum), there is $m _ { B } \in \mathcal { F } _ { B }$ with $m ^ { - } \le m _ { B } < x$ . If $m _ { B } = m ^ { - }$ , this contradicts $m ^ { - } \notin \mathcal { F } _ { B }$ outright; if $m ^ { - } < m _ { B } < x .$ , Lemma 9 makes the same bid raise profitable at $m _ { B }$ , contradicting $m _ { B } \in \mathcal { F } _ { B }$ Hence $m ^ { - } \in { \mathcal { F } } _ { B } , \operatorname { s o } m ^ { - } = \operatorname* { m i n } { \mathcal { F } } _ { B }$ and, in particular, admits no profitable bid raise.

To show that $m ^ { - }$ also admits no profitable undercut, suppose, for contradiction, that a profitable undercut at $m ^ { - }$ induces $y < m ^ { - }$ ; by Lemma $^ { 7 , }$

$$
2 y - a ( y ) > b ( m ^ { - } ) \qquad \mathrm { a n d } \qquad H ( y ) < y .
$$

Since $H ( y ) < y , y \notin { \mathcal { F } }$ , so $y \ne m _ { \mathrm { { m i n } } }$ . Moreover, $m ^ { - } \ne m _ { \mathrm { { m i n } } } \colon$ : otherwise the assumed undercut at $m ^ { - }$ would itself be a profitable undercut at $m _ { \mathrm { m i n } } .$ , contradicting Lemma 8. Since $\mathcal { F } _ { B } \subseteq \mathcal { F } , m ^ { - } =$ inf $\mathcal { F } _ { B } \geq$ inf ${ \mathcal { F } } = m _ { \operatorname* { m i n } }$ , so together with $m ^ { - } \ne m _ { \mathrm { m i n } }$ we get $m _ { \mathrm { m i n } } < m ^ { - }$ Neither can $y < m _ { \mathrm { m i n } } .$ : Lemma 9 (with $m _ { 1 } = m _ { \mathrm { m i n } } < m ^ { - } = m _ { \mathrm { 2 } } )$ would then make the same undercut profitable at $m _ { \mathrm { m i n } } ,$ contradicting Lemma 8. Hence

$$
m _ { \mathrm { m i n } } < y < m ^ { - } .
$$

Let ${ \mathcal { F } } _ { y } : = \{ m \in { \mathcal { F } } \mid m < y \}$ ; it is nonempty (it contains $m _ { \mathrm { m i n } } )$ , and its supremum lies in F. As $y \notin { \mathcal { F } } _ { : }$ , this supremum cannot equal y, so it is attained: $\mathcal { F } _ { y }$ has a maximum $m _ { k } : =$ max $\mathcal { F } _ { y } < y < m$ <sup>−</sup>. Since $m _ { k } < m ^ { - } =$ min ${ \mathcal { F } } _ { B } , m _ { k } \notin { \mathcal { F } } _ { B } ;$ : a profitable bid raise at $m _ { k }$ induces some $x > m _ { k }$ , and by Lemma $^ { 7 , }$

$$
2 x - b ( x ) < a ( m _ { k } ) \qquad \mathrm { a n d } \qquad H ( x ) > x .
$$

Since $H ( x ) > x , x$ is neither y, as $H ( y ) < y$ , nor m<sup>−</sup>, as m<sup>−</sup> $\ u \in \mathcal { F }$ . Thus, the remaining viable positions for x are $x < y , y < x < m$ <sup>−</sup>, $\mathrm { o r } \ x > m ^ { - }$ :

1. Case $x < y \colon$ Since $H ( x ) > x , H ( y ) < y .$ , and H is nondecreasing, H maps $[ x , y ]$ into itself; by Knaster–Tarski, H admits a fixed point $\check { m } \in [ x , y ]$ . Since $\check { m } \ge x > m _ { k }$ and $\check { m } \leq y$ with $\check { m } \neq y \ ( \mathrm { a s } \ y \notin \mathcal { F } )$ , we have $m _ { k } < \check { m } < y$ , so mˇ $\in \mathcal { F } _ { y }$ , contradicting the maximality of $m _ { k }$ there.

2. Case $y \textless x < m ^ { - }$ : As $m _ { k } ~ < ~ y$ and $x < m ^ { - }$ , monotonicity of $a ( \cdot )$ and $b ( \cdot )$ implies $2 x - b ( x ) < a ( m _ { k } ) \leq a ( y )$ and $b ( x ) \leq b ( m ^ { - } ) < 2 y - a ( y )$ . Combining these gives $2 x - b ( x ) < a ( y ) < 2 y - b ( x )$ , which reduces to $x < y ,$ contradicting $y < x$

3. Case $x > m ^ { - } { : }$ Since $m _ { k } < m ^ { - } < x$ , Lemma 9 implies that the bid raise inducing midpoint x is also profitable at $m ^ { - }$ , contradicting $m ^ { - } \in \mathcal { F } _ { B }$

Every case is contradictory, so m<sup>−</sup> admits no profitable undercut. By Lemma 6, m<sup>−</sup> is therefore immune to every profitable unilateral deviation, so $\mathbf { s } ( m ^ { - } )$ is a pure-strategy Nash equilibrium. 7

## Theorem 11 (Dual Existence of a Pure-Strategy Nash Equilibrium). Define

${ \mathcal { F } } _ { A } : = \{ m \in { \mathcal { F } } \mid$ no profitable aks undercut exists at m}

and $m ^ { + } : = \operatorname* { s u p } { \mathcal { F } _ { A } }$ . Then $\mathcal { F } _ { A } \neq \emptyset$ , and $\mathbf { s } ( m ^ { + } )$ is a pure-strategy Nash equilibrium.

Proof. By Lemma $8 , m _ { \mathrm { m i n } } \in { \mathcal { F } } _ { A }$ so ${ \mathcal { F } } _ { A } \neq \emptyset ;$ since $\mathcal { F }$ is compact (as noted in the proof of Theorem 10) and ${ \mathcal { F } } _ { A } \subseteq { \mathcal { F } }$ is nonempty, $m ^ { + } : = \operatorname* { s u p } { \mathcal { F } _ { A } }$ exists and lies in $\mathcal { F }$ . We show $m ^ { + } \in { \mathcal { F } } _ { A }$

Suppose not: a profitable undercut at $m ^ { + }$ induces some $y < m ^ { + }$ . Since $m ^ { + }$ is an upper bound of ${ \mathcal { F } } _ { A }$ but y is not (else y would be an upper bound below the supremum), there is $m _ { A } \in { \mathcal { F } } _ { A }$ with $y < m _ { A } \leq m ^ { + }$ . If $m _ { A } = m ^ { + }$ , this contradicts $m ^ { + } \notin \mathcal { F } _ { A }$ outright; if $y < m _ { A } < m ^ { + }$ , Lemma 9 makes the same undercut profitable at $m _ { A }$ , contradicting $m _ { A } \in { \mathcal { F } } _ { A }$ Hence $m ^ { + } \in { \mathcal { F } } _ { A }$ , so $m ^ { + } = \operatorname* { m a x } { \mathcal { F } _ { A } }$ and, in particular, admits no profitable undercut. It remains to exclude a profitable bid raise.

Suppose one exists at $m ^ { + }$ , inducing $x > m ^ { + }$ . If $m ^ { + } = m _ { \mathrm { m a x } }$ , this bid raise is itself profitable at $m _ { \mathrm { m a x } }$ , contradicting Lemma 8; hence $m ^ { + } < m _ { \mathrm { m a x } } .$ . Moreover $x \le m _ { \mathrm { m a x } } \colon$ otherwise, since $m ^ { + } < m _ { \mathrm { m a x } } < x$ , Lemma 9 would carry the bid raise to $m _ { \mathrm { m a x } }$ , again contradicting Lemma 8. Thus ${ \mathcal { F } } _ { x } : = \{ m \in { \mathcal { F } } \mid m \geq x \}$ contains $m _ { \mathrm { m a x } }$ and, being closed and bounded, is compact, so

$$
m _ { k } : = \operatorname* { m i n } \mathcal { F } _ { x }
$$

is well defined. Since Lemma 7 gives $H ( x ) > x ,$ x is not a fixed point, so $m _ { k } \neq x ,$ so $m _ { k } > x > m ^ { + } = \operatorname* { m a x } \mathcal { F } _ { A } ;$ in particular, $m _ { k } \notin { \mathcal { F } } _ { A }$ , so it admits a profitable undercut inducing some $y < m _ { k }$

This is the mirror image of the construction in Theorem 10: there, an undercut at $m ^ { - }$ produced y and a bid raise at an auxiliary $m _ { k }$ produced $x ;$ here, a bid raise at $m ^ { + }$ produces x and an undercut at the auxiliary $m _ { k }$ produces y. Since the undercut at $m _ { k }$ gives $H ( y ) < y$ $y$ is neither x, as $H ( x ) > x$ , nor $m ^ { + }$ , as $m ^ { + } \in \mathcal { F }$ , and the same three-way split on the position of y closes of identically:

1. Case $y < m ^ { + }$ : Lemma 9 carries the undercut at $m _ { k }$ to $m ^ { + }$ , contradicting $m ^ { + } \in { \mathcal { F } } _ { A }$

2. Case $m ^ { + } < y < x$ : Monotonicity of $a ( \cdot ) , b ( \cdot )$ collapses the two deviation inequalities to $x < y ,$ contradicting $y < x$

3. Case $x \ < \ y :$ H maps $[ x , y ]$ into itself, so Knaster–Tarski gives a fixed point strictly between x and $y ,$ contradicting the minimality of $m _ { k }$

Hence no profitable bid raise exists at $m ^ { + }$ , and Lemma 6 makes $\mathbf { s } ( m ^ { + } )$ a pure-strategy Nash equilibrium. ■

## A.2. Proofs of Section 4

Proposition 15 (Structure of Equilibria under Regularity). (i) The fixed-point set $\mathcal { F }$ is a nonempty closed interval $[ m ^ { - } , m ^ { + } ]$ ; and (ii) every $m ^ { * } \in { \mathcal { F } }$ is a PSNE mid-price, with market prices $( b ( m ^ { * } ) , a ( m ^ { * } ) )$ .

Proof. We first show (i). By Theorem 5, F is nonempty; equivalently, $G ( m ) = m - H ( m )$ has a zero. The function G is continuous and, by the second condition of Assumption 14, nondecreasing, so its zero set is a closed interval: if $G ( m _ { 1 } ) = G ( m _ { 2 } ) = 0$ for some $m _ { 1 } < m _ { 2 }$ monotonicity forces $0 = G ( m _ { 1 } ) \leq G ( m ) \leq G ( m _ { 2 } ) = 0$ for every m $\in [ m _ { 1 } , m _ { 2 } ]$ , so the zero set is convex, and continuity makes it closed. Hence $\mathcal { F }$ is a nonempty closed interval $[ m ^ { - } , m ^ { + } ]$

We now show (ii). Fix $m ^ { * } \in { \mathcal { F } }$ , and consider a unilateral deviation $( \widetilde { b } , \widetilde { a } )$ from the symmetric zero-profit profile $\mathbf { s } ( m ^ { * } )$ , where both makers play $( b ( m ^ { * } ) , a ( m ^ { * } ) )$ :

1. Case $\widetilde { a } \geq a ( m ^ { * } )$ and ${ \widetilde { b } } \leq b ( m ^ { * } )$ : The market prices, and hence the mid-price $m ^ { * }$ , are unchanged, so the deviator earns zero profit on each side, and the deviation is not profitable.

2. Case $\widetilde { a } < a ( m ^ { * } )$ and $\widetilde { b } \leq b ( m ^ { * } )$ : On the ask side, the deviator is now uniquely best, so the market mid-price becomes

$$
y : = \frac { \widetilde a + b ( m ^ { * } ) } { 2 } < m ^ { * } ,
$$

and its ask-side profit is $\mathbb { P } ( V > y ) ( \widetilde { a } - a ( y ) )$ ; we show this margin is negative. Since G is nondecreasing by the second condition of Assumption 14 and $y < m ^ { * } , G ( y ) \leq G ( m ^ { * } ) = 0$ i.e.,

$$
y \leq H ( y ) = { \frac { a ( y ) + b ( y ) } { 2 } } \iff a ( y ) \geq 2 y - b ( y ) .
$$

Since b is strictly increasing by the first condition of Assumption 14 and $y \ < \ m ^ { * }$ 2 $b ( y ) < b ( m ^ { * } )$ . Hence

$$
\widetilde { a } = 2 y - b ( m ^ { * } ) < 2 y - b ( y ) \leq a ( y ) ,
$$

so $\widetilde { a } - a ( y ) < 0$ , and the ask side is loss-making. On the bid side, the deviator is either inactive, or ties at $b ( m ^ { * } )$ and earns $\begin{array} { r } { \frac { 1 } { 2 } \mathbb { P } ( V < y ) \left( b ( y ) - b ( m ^ { * } ) \right) < 0 } \end{array}$ . Hence the deviation is not profitable.

3. Case $\widetilde { a } \geq a ( m ^ { * } )$ and $\widetilde { b } > b ( m ^ { * } )$ : Symmetric to the previous case, with the roles of a and b swapped.

4. Case $\widetilde { a } < a ( m ^ { * } )$ and $\widetilde { b } > b ( m ^ { * } )$ : The deviator is now uniquely best on both sides, so the market mid-price becomes

$$
z : = \frac { \widetilde { a } + \widetilde { b } } { 2 } ,
$$

and its total profit is

$$
\mathbb { P } ( V > z ) \left( \widetilde { a } - a ( z ) \right) + \mathbb { P } ( V < z ) \left( b ( z ) - \widetilde { b } \right) .
$$

We show that both margins are negative when $z \geq m ^ { * } ;$ ; the case $z < m ^ { * }$ is symmetric. Since $a ( \cdot )$ is strictly increasing by the first condition of Assumption 14 and $z \geq m ^ { * }$ 2 $a ( z ) \geq a ( m ^ { * } ) > \widetilde { a } .$ so the ask side is loss-making. For the bid side, since $\widetilde { b } = 2 z - \widetilde { a }$ and $b ( m ^ { * } ) = 2 m ^ { * } - a ( m ^ { * } )$

$$
\widetilde { b } - b ( m ^ { * } ) = 2 ( z - m ^ { * } ) + \underbrace { ( a ( m ^ { * } ) - \widetilde { a } ) } _ { > 0 } > 2 ( z - m ^ { * } ) .
$$

By Assumption 14, $b ^ { \prime } ( m ) \leq 2 - a ^ { \prime } ( m ) < 2 \mathrm { ~ o n ~ } ( 0 , 1 )$ , so the mean value theorem gives $b ( z ) - b ( m ^ { * } ) \leq 2 ( z - m ^ { * } )$ . Hence

$$
\widetilde b - b ( m ^ { * } ) > 2 ( z - m ^ { * } ) \ge b ( z ) - b ( m ^ { * } ) ,
$$

so $\widetilde { b } > b ( z )$ : the bid side is also loss-making, and the deviation is not profitable.

Hence $\mathbf { s } ( m ^ { * } )$ is a PSNE, and $m ^ { * }$ is a PSNE mid-price.

Lemma 17. For every market ask $A \in ( 0 , 1 ]$

$$
\alpha ( \beta ( A ) ) \left\{ { \begin{array} { l l } { { > A } } & { { i f A < a ^ { - } , } } \\ { { = A } } & { { i f A \in [ a ^ { - } , a ^ { + } ] , } } \\ { { < A } } & { { i f A > a ^ { + } . } } \end{array} } \right.
$$

Proof. Fix $A \in ( 0 , 1 ]$ , and let $m ( A ) : = ( A + \beta ( A ) ) / 2$ be the mid-price induced by A and $\beta ( A )$ , so that $\beta ( A ) = b ( m ( A ) )$ by Definition 16.

We first reduce the claim to the sign of G. As shown after Definition 16, the map $x \mapsto x - a ( ( x + \beta ( A ) ) / 2 )$ is strictly increasing and vanishes exactly at $x = \alpha ( \beta ( A ) )$ . At $x = A$ , it equals

$$
A - a ( m ( A ) ) = 2 m ( A ) - b ( m ( A ) ) - a ( m ( A ) ) = 2 G ( m ( A ) ) .
$$

Hence $\alpha ( \beta ( A ) ) > A , = A , \mathrm { o r } < A$ according to whether $G ( m ( A ) )$ is negative, zero, or positive.

We now locate $m ( A )$ . Since $\beta$ is strictly increasing, so is $m ( \cdot )$ . Moreover, $\beta ( a ^ { \pm } ) = b ^ { \pm }$ so $m ( a ^ { \pm } ) = H ( m ^ { \pm } ) = m ^ { \pm }$ . Hence $m ( A ) < m ^ { - } { \mathrm { ~ i f ~ } } A < a ^ { - } , m ( A ) \in [ m ^ { - } , m ^ { + } ] { \mathrm { ~ i f ~ } } A \in [ a ^ { - } , a ^ { + } ]$ and $m ( A ) > m ^ { + } \mathrm { i f } \ A > a ^ { + }$

Finally, $G$ is nondecreasing by the second condition of Assumption 14, and its zero set is exactly $\mathcal { F } = [ m ^ { - } , m ^ { + } ]$ by Proposition 15. Hence $G < 0 \mathrm { o n } [ 0 , m ^ { - } ) , G = 0 \mathrm { o n } [ m ^ { - } , m ^ { + } ]$ , and $G > 0 \mathrm { ~ o n ~ } ( m ^ { + } , 1 ]$ , which gives the three cases of the claim.

Lemma 18. Let $\mathbf { s } = ( \mathbf { b } , \mathbf { a } ) \in \mathcal { S } ^ { \Delta } , A : = a ^ { * } ( \mathbf { a } )$ , and $B : = b ^ { * } ( { \mathbf { b } } )$ $I f a _ { i } = A$ and $a _ { i } + \Delta < \alpha ( B )$ 2 then raising $a _ { i }$ to $a _ { i } + \Delta$ strictly increases $u _ { i } .$ . Symmetrically, $i f b _ { i } = B$ and $b _ { i } - \Delta > \beta ( A )$ , then lowering $b _ { i }$ to $b _ { i } - \Delta$ strictly increases $u _ { i }$

Proof. We prove the ask statement; the bid statement is symmetric. Only $a _ { i }$ changes, so the market bid B and maker i’s share of the bid side, $\lambda : = \mathbb { I } _ { \{ b _ { i } = B \} } / n _ { B } ( \mathbf { b } ) \in \{ 0 , \frac { 1 } { 2 } , 1 \}$ , remain fixed throughout. For $\widetilde { a } \in [ a _ { i } , a _ { i } + \Delta ]$ , let $y : = ( \widetilde { a } + B ) / 2$ , and let

$$
u _ { i } ( \widetilde { \boldsymbol { a } } ) : = u _ { i } ^ { \mathrm { a s k } } ( \widetilde { \boldsymbol { a } } ) + u _ { i } ^ { \mathrm { b i d } } ( \widetilde { \boldsymbol { a } } ) = ( 1 - F ( y ) ) ( \widetilde { \boldsymbol { a } } - \boldsymbol { a } ( y ) ) + \lambda F ( y ) ( b ( y ) - B )
$$

be maker i’s payof if it alone quoted the market ask ${ \widetilde { a } } ,$ all other quotes unchanged, split into its ask-side and bid-side components.

We first show that the ask side is loss-making, i.e., $u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) < 0$ , or equivalently $\widetilde a < a ( y )$ $\mathrm { I f } \ B < 1$ , the map $\varphi ( x ) : = x - a ( ( x + B ) / 2 )$ is strictly increasing on $[ 0 , 1 ]$ and vanishes at $x = \alpha ( B )$ , as noted after Definition 16. Since $\widetilde { a } \le a _ { i } + \Delta < \alpha ( B )$

$$
\widetilde { a } - a ( y ) = \varphi ( \widetilde { a } ) < \varphi ( \alpha ( B ) ) = 0 .
$$

If $B = 1$ , then $\widetilde { a } < \alpha ( 1 ) = 1$ , so $\widetilde { a } < y < 1$ and hence $\widetilde { a } < y < a ( y )$ We now show that $u _ { i }$ is strictly increasing on $[ a _ { i } , a _ { i } + \Delta ]$ . For $\widetilde { a } \in ( a _ { i } , a _ { i } + \Delta )$ , we have $0 < \widetilde { a } < \alpha ( B ) \leq 1$ and $0 \leq B \leq 1$ , so $y \in ( 0 , 1 )$ . Writing $\begin{array} { r } { \left( 1 - F ( y ) \right) a ( y ) = \int _ { y } ^ { 1 } v f ( v ) } \end{array}$ dv and $\begin{array} { r } { F ( y ) b ( y ) = \int _ { 0 } ^ { y } v f ( v ) d v } \end{array}$ , and using ${ d y } / { d \widetilde { a } } = 1 / 2$ and $\widetilde { a } - y = y - B = ( \widetilde { a } - B ) / 2$ , we obtain

$$
\begin{array} { l } { { u _ { i } ^ { \prime } ( \widetilde { a } ) = 1 - F ( y ) - \displaystyle \frac { 1 } { 2 } f ( y ) ( \widetilde { a } - y ) + \displaystyle \frac { \lambda } { 2 } f ( y ) ( y - B ) } } \\ { { \displaystyle ~ = 1 - F ( y ) - \frac { 1 - \lambda } { 4 } f ( y ) ( \widetilde { a } - B ) . } } \end{array}
$$

The first term is the margin gained on every executed buy order, while the second is the net loss from the marginal trader at $y .$ who switches from buying at $\widetilde { a }$ to selling at $B ,$ of which maker i captures only a share λ. If $\widetilde a \le B$ , then $u _ { i } ^ { \prime } ( \widetilde { a } ) \geq 1 - F ( y ) > 0$ . Otherwise, the ask-side loss shown above gives $\widetilde { a } - y < a ( y ) - y .$ , so, using $a ^ { \prime } ( y ) = f ( y ) ( a ( y ) - y ) / ( 1 - F ( y ) )$ ,

$$
\begin{array} { l } { \displaystyle \frac { 1 - \lambda } { 4 } f ( y ) ( \widetilde a - B ) \le \frac 1 2 f ( y ) ( \widetilde a - y ) } \\ { \displaystyle < \frac 1 2 f ( y ) ( a ( y ) - y ) = \frac 1 2 ( 1 - F ( y ) ) a ^ { \prime } ( y ) . } \end{array}
$$

Hence $u _ { i } ^ { \prime } ( \widetilde { a } ) > ( 1 - F ( y ) ) ( 1 - a ^ { \prime } ( y ) / 2 ) > 0 .$ , since $a ^ { \prime } ( y ) \leq 2 - b ^ { \prime } ( y ) < 2$ by Assumption 14. As $u _ { i }$ is continuous on $[ a _ { i } , a _ { i } + \Delta ]$ , the mean value theorem shows that it is strictly increasing there.

Finally, since $a _ { i } = A$ and prices lie on the grid, exactly one of the following holds:

1. Case $a _ { - i } \geq a _ { i } + 2 \Delta$ : Maker i alone quotes the market ask before and after the deviation, so its payof rises from $u _ { i } ( a _ { i } )$ to $u _ { i } ( a _ { i } + \Delta )$

2. Case $a _ { - i } = a _ { i } + \Delta$ : After the deviation, maker i ties at the market ask $a _ { i } + \Delta$ . The mid-price is the same as if it were alone there, so its bid-side payof is $u _ { i } ^ { \mathrm { b i d } } ( a _ { i } + \Delta )$ , while its ask-side payof is halved. Its payof is therefore $\begin{array} { r } { u _ { i } ( a _ { i } + \Delta ) - \frac { 1 } { 2 } u _ { i } ^ { \mathrm { a s k } } ( a _ { i } + \Delta ) > u _ { i } ( a _ { i } + \Delta ) > u _ { i } ( a _ { i } ) } \end{array}$

3. Case $a _ { - i } = a _ { i } \colon$ Before the deviation, maker i ties at $a _ { i }$ and earns $\textstyle { \frac { 1 } { 2 } } u _ { i } ^ { \mathrm { a s k } } ( a _ { i } ) + u _ { i } ^ { \mathrm { b i d } } ( a _ { i } )$ After it, the opponent alone quotes the market ask ${ { a } _ { i } } ,$ , so the mid-price, and hence maker i’s bid-side payof, is unchanged, while its ask-side payof becomes zero, for a gain of $\cdot \frac { 1 } { 2 } u _ { i } ^ { \mathrm { a s k } } ( a _ { i } ) > 0$

In every case, the deviation strictly increases $u _ { i } .$ , and it is feasible, since $b _ { i } \leq a _ { i } < a _ { i } + \Delta <$ $\alpha ( B ) \leq 1$ ■

Lemma 19. Fix $\overline { { B } } < 1$ and $\underline { { A } } > 0$ . There exists $C _ { 0 } > 1$ , depending only on $F , { \overline { { B } } }$ and $\underline { { A } } ,$ such that the following holds. Let $\mathbf { s } = ( \mathbf { b } , \mathbf { a } ) \in \mathcal { S } ^ { \Delta } , A : = a ^ { * } ( \mathbf { a } ) , B : = b ^ { * } ( \mathbf { b } )$ , and let i be a maker. If $B \leq \overline { { B } }$ and ea $\in \Gamma _ { \Delta }$ satisfies $a _ { - i } \leq \widetilde { a } \leq a _ { i }$ and $\widetilde { a } \ge \alpha ( B ) + C _ { 0 } \Delta$ , then lowering $a _ { i }$ to $\widetilde { a } - \Delta$ weakly increases $u _ { i } ,$ strictly if $a _ { - i } \geq \widetilde { a } - \Delta$ . Symmetrically, if $A \geq \underline { { A } }$ and $\widetilde { b } \in \Gamma _ { \Delta }$ satisfies $b _ { i } \leq \widetilde { b } \leq b _ { - i }$ and $\widetilde { b } \le \beta ( A ) - C _ { 0 } \Delta$ , then raising $b _ { i }$ to $\widetilde { b } + \Delta$ weakly increases $u _ { i . }$ strictly if $b _ { - i } \leq \bar { b } + \Delta$

Proof. We prove the ask statement; the bid statement is symmetric. The deviation changes only $a _ { i } .$ so the market bid $B$ and maker i’s share λ of the bid side, as in the proof of Lemma 18, are fixed throughout. As in that proof, if maker i alone quoted the market ask ${ \widetilde { a } } ,$ its ask-side and bid-side payofs would be

$$
u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) = ( 1 - F ( y ) ) ( \widetilde { a } - a ( y ) ) , \quad \mathrm { a n d } \quad u _ { i } ^ { \mathrm { b i d } } ( \widetilde { a } ) = \lambda F ( y ) ( b ( y ) - B ) ,
$$

where $y : = ( \widetilde { a } + B ) / 2$

We first fix the constants. For $\widetilde { a } \in [ \alpha ( B ) , 1 ]$ , since $\alpha ( 0 ) \leq \alpha ( B )$ and $B \leq \overline { { B } }$ , the mid-price $y$ lies in the compact interval

$$
K : = \left[ { \alpha ( 0 ) } / { 2 } , ( 1 + \overline { { B } } ) / 2 \right] \subset ( 0 , 1 ) .
$$

On $K ,$ , the functions $1 - F$ and $1 - a ^ { \prime } / 2$ are continuous and positive, the latter since $a ^ { \prime } ( m ) \leq 2 - b ^ { \prime } ( m ) < 2$ by Assumption 14, so they are bounded below by some $c _ { 1 } , c _ { 2 } > 0$ respectively; moreover, $f$ is bounded. Since the margin $\widetilde a - a ( y )$ vanishes at $\widetilde { a } = \alpha ( B )$ and has derivative $1 - a ^ { \prime } ( y ) / 2 \geq c _ { 2 }$ , every $\widetilde { a } \in [ \alpha ( B ) , 1 ]$ satisfies

$$
u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) \geq c ( \widetilde { a } - \alpha ( B ) ) , \qquad c : = c _ { 1 } c _ { 2 } .
$$

Moreover, diferentiating as in the proof of Lemma 18,

$$
\frac { d } { d \widetilde { a } } u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) = 1 - F ( y ) - \frac { 1 } { 4 } f ( y ) ( \widetilde { a } - B ) , \quad \mathrm { a n d } \quad \frac { d } { d \widetilde { a } } u _ { i } ^ { \mathrm { b i d } } ( \widetilde { a } ) = \frac { \lambda } { 4 } f ( y ) ( \widetilde { a } - B ) ,
$$

which are both bounded in absolute value by some $L > 0$ on $[ \alpha ( B ) , 1 ]$ . The constants c and L depend only on F and B, and we choose $C _ { 0 } > 1$ such that $c C _ { 0 } > 4 L ;$ for the bid statement, the symmetric construction yields constants depending only on F and $\underline { { A } } .$ and we take $C _ { 0 }$ large enough for both.

Consider now the deviation from $a _ { i }$ to $\widetilde { a } - \Delta$ . Since $C _ { 0 } > 1$ ,

$$
\widetilde a - \Delta \ge \alpha ( B ) + ( C _ { 0 } - 1 ) \Delta > \alpha ( B ) > B \ge b _ { i } ,
$$

so the deviation is feasible, and $u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } - \Delta ) > 0$ by the first bound above. Since prices lie on the grid and $a _ { - i } \leq \widetilde { a } \leq a _ { i }$ , one of the following cases holds:

1. Case $a _ { - i } \leq \widetilde { a } - 2 \Delta$ : Maker i quotes above the market ask before and after the deviation, and market prices are unchanged, so its payof is unchanged.

2. Case $a _ { - i } = \widetilde { a } - \Delta \mathrm { : }$ : Before the deviation, maker i quotes above the market ask, so its ask-side payof is zero. After the deviation, it ties at the market ask, which is unchanged, so its bid-side payof is unchanged and its ask-side payof becomes $\frac { 1 } { 2 } u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } - \Delta ) > 0$

3. Case $a _ { - i } = \widetilde { a } \colon$ Before the deviation, maker i’s ask-side payof is $\frac { 1 } { 2 } u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } )$ if $a _ { i } = \widetilde { a }$ and zero otherwise, hence at most $\frac { 1 } { 2 } u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } )$ , since $u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) \ge c C _ { 0 } \Delta > 0$ . After the deviation, it alone quotes the market ask $\widetilde { a } - \Delta$ ; by the mean value theorem, its ask-side payof is $u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } - \Delta ) \geq u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) - L \Delta$ , while its bid-side payof decreases by at most $L \Delta$ . Hence its total gain is at least

$$
\frac { 1 } { 2 } u _ { i } ^ { \mathrm { a s k } } ( \widetilde { a } ) - 2 L \Delta \ge \left( \frac { c C _ { 0 } } { 2 } - 2 L \right) \Delta > 0 .
$$

Hence, the deviation weakly increases $u _ { i } ,$ strictly if $a _ { - i } \geq \widetilde { a } - \Delta$

Lemma 21. There exist $C _ { 0 } > 1$ and $\Delta _ { 0 } > 0$ , depending only on $F _ { s }$ , such that, for every $\Delta \le \Delta _ { 0 }$ , every $C E \pi \ o f \mathcal { G } _ { \Delta }$ satisfies

$$
\begin{array} { r l } & { a ^ { \mathrm { m i n } } ( \pi ) \geq \alpha ( b ^ { \mathrm { m i n } } ( \pi ) ) - \Delta , \qquad b ^ { \mathrm { m a x } } ( \pi ) \leq \beta ( a ^ { \mathrm { m a x } } ( \pi ) ) + \Delta , } \\ & { a ^ { \mathrm { m a x } } ( \pi ) \leq \alpha ( b ^ { \mathrm { m a x } } ( \pi ) ) + C _ { 0 } \Delta , \qquad b ^ { \mathrm { m i n } } ( \pi ) \geq \beta ( a ^ { \mathrm { m i n } } ( \pi ) ) - C _ { 0 } \Delta . } \end{array}
$$

Proof. We prove the ask bounds; the bid bounds are symmetric. We drop the argument π of the extremal prices, and call a profile supported if it lies in $\operatorname { s u p p } ( \pi )$ . We repeatedly use the following consequence of Definition 20: if, for a maker i, a supported profile s, and some $\widetilde { s } _ { i } \in { \mathcal { S } } _ { i } ^ { \Delta }$ , switching from $s _ { i }$ to $\widetilde { s } _ { i }$ weakly increases $u _ { i }$ at every supported profile in which maker i plays $s _ { i } ,$ and strictly at ${ \mathbf { s } } ,$ then the CE constraint for $s _ { i }$ and $\widetilde { s } _ { i }$ is violated. Let $\Delta _ { 0 } : = \mathrm { m i n } \{ \alpha ( 0 ) , 1 - \beta ( 1 ) \} / 2$ , and let $C _ { 0 }$ be the constant of Lemma 19 for $\overline { { B } } : = ( 1 + \beta ( 1 ) ) / 2$ and $\underline { { A } } : = \alpha ( 0 ) / 2 $ ; both depend only on $F$

We first show that $a ^ { \mathrm { m i n } } \geq \alpha ( b ^ { \mathrm { m i n } } ) - \Delta$ . Suppose not, and let s be a supported profile with market ask $a ^ { \mathrm { m i n } }$ , quoted by maker i. Every supported profile in which maker i plays $s _ { i }$ has market ask at least $a ^ { \mathrm { m i n } } = a _ { i }$ , so maker i quotes the market ask there, and its market bid $B \geq b ^ { \mathrm { m i n } }$ satisfies

$$
a _ { i } + \Delta < \alpha ( b ^ { \mathrm { m i n } } ) \leq \alpha ( B ) .
$$

By Lemma 18, raising $a _ { i }$ to $a _ { i } + \Delta$ strictly increases $u _ { i }$ at every such profile, a contradiction. Symmetrically, $b ^ { \operatorname* { m a x } } \le \beta ( a ^ { \operatorname* { m a x } } ) + \Delta$

By these two bounds and $\Delta \le \Delta _ { 0 }$ , every supported profile satisfies

$$
B \leq b ^ { \operatorname* { m a x } } \leq \beta ( 1 ) + \Delta \leq \overline { { B } } \qquad \mathrm { a n d } \qquad A \geq a ^ { \operatorname* { m i n } } \geq \alpha ( 0 ) - \Delta \geq \underline { { A } } ,
$$

so Lemma 19 applies at every supported profile.

We now show that $a ^ { \mathrm { m a x } } \leq \alpha ( b ^ { \mathrm { m a x } } ) + C _ { 0 } \Delta$ . Suppose not; since α is increasing, every supported profile then satisfies $a ^ { \mathrm { m a x } } > \alpha ( B ) + C _ { 0 } \Delta$ . It sufices to find a supported profile s with market ask $a ^ { \mathrm { m a x } }$ and a maker i such that, at every supported profile in which maker i plays $s _ { i } .$ the opponent’s ask is at most $a ^ { \mathrm { m a x } }$ . Indeed, since $a _ { i } \geq a ^ { \operatorname* { m a x } }$ , Lemma 19 with $\widetilde { a } : = a ^ { \mathrm { m a x } }$ then shows that lowering $a _ { i }$ to $a ^ { \mathrm { m a x } } - \Delta$ weakly increases $u _ { i }$ at every such profile, and strictly at s, where $a _ { - i } \geq a ^ { \operatorname* { m a x } }$ , a contradiction. We find such s and i as follows:

1. Case some supported profile s with market ask $a ^ { \mathrm { m a x } }$ has a maker i with $a _ { i } > a ^ { \operatorname* { m a x } }$ Every supported profile in which maker i plays $s _ { i }$ has market ask at most $a ^ { \mathrm { m a x } } < a _ { i } ,$ by definition of $a ^ { \mathrm { m a x } }$ ; hence the market ask there is quoted by the opponent, whose ask is therefore at most $a ^ { \mathrm { m a x } }$

2. Case both makers quote $a ^ { \mathrm { m a x } }$ at every supported profile with market ask $a ^ { \mathrm { m a x } }$ : Take any such s and any maker i. If, at some supported profile in which maker i plays $s _ { i } ,$ the opponent’s ask exceeded $a ^ { \mathrm { m a x } }$ , that profile would have market ask $a _ { i } = a ^ { \operatorname* { m a x } }$ , quoted by maker i alone, contradicting the case assumption.

Hence $a ^ { \mathrm { m a x } } \leq \alpha ( b ^ { \mathrm { m a x } } ) + C _ { 0 } \Delta$

Theorem 22 (Convergence of Grid Correlated Equilibria). $\varepsilon _ { \Delta } \to 0$ as $\Delta \to 0$

Proof. It sufices to show that, for any tick sizes $\Delta _ { n }  0$ with $1 / \Delta _ { n } \in \mathbb { N }$ and any CEs $\pi _ { n }$ of ${ \mathcal { G } } _ { \Delta _ { n } }$

$$
\operatorname* { m a x } _ { \mathbf { s } \in \mathrm { s u p p } ( \pi _ { n } ) } D _ { \mathrm { e n v } } ( \mathbf { s } ) \to 0 ;
$$

choosing each $\pi _ { n }$ within $\Delta _ { n }$ of the supremum in $\varepsilon _ { \Delta _ { n } }$ then gives $\varepsilon _ { \Delta _ { n } } \to 0$ . Write $a _ { n } ^ { \mathrm { m i n } } : =$ $a ^ { \mathrm { m i n } } ( \pi _ { n } )$ , and similarly $a _ { n } ^ { \mathrm { m a x } } , b _ { n } ^ { \mathrm { m i n } }$ and $b _ { n } ^ { \mathrm { m a x } }$ . Let $C _ { 0 }$ and $\Delta _ { 0 }$ be the constants of Lemma 21; discarding finitely many terms, we may assume that $\Delta _ { n } \leq \Delta _ { 0 }$ and

$$
( 1 + C _ { 0 } ) \Delta _ { n } \leq \frac { 1 - \beta ( 1 ) } { 2 } .
$$

We first combine the four bounds of Lemma 21, namely

$$
\begin{array} { r l } { \beta ( a _ { n } ^ { \mathrm { m i n } } ) \leq b _ { n } ^ { \mathrm { m i n } } + C _ { 0 } \Delta _ { n } , \quad \quad \quad } & { \alpha ( b _ { n } ^ { \mathrm { m i n } } ) \leq a _ { n } ^ { \mathrm { m i n } } + \Delta _ { n } , } \\ { b _ { n } ^ { \mathrm { m a x } } \leq \beta ( a _ { n } ^ { \mathrm { m a x } } ) + \Delta _ { n } , \quad \quad \quad } & { a _ { n } ^ { \mathrm { m a x } } \leq \alpha ( b _ { n } ^ { \mathrm { m a x } } ) + C _ { 0 } \Delta _ { n } . } \end{array}
$$

In particular, $b _ { n } ^ { \operatorname* { m i n } } \leq b _ { n } ^ { \operatorname* { m a x } } \leq \beta ( 1 ) + \Delta _ { n } , \mathrm { \ s o }$ , by the choice of $\Delta _ { n } .$ , the points $b _ { n } ^ { \mathrm { m i n } } + C _ { 0 } \Delta _ { n }$ and $\beta ( a _ { n } ^ { \mathrm { m a x } } ) + \Delta _ { n } .$ and hence also $b _ { n } ^ { \mathrm { m i n } }$ and $\beta ( a _ { n } ^ { \mathrm { m a x } } )$ , lie in $J : = [ 0 , ( 1 + \beta ( 1 ) ) / 2 ] \subset [ 0 , 1 )$ on which α is Lipschitz with some constant $L _ { \alpha }$ . Since α is also increasing, combining the bounds in each row gives, for the first row,

$$
\begin{array} { r l r } {  { \alpha ( \beta ( a _ { n } ^ { \mathrm { m i n } } ) ) \le \alpha ( b _ { n } ^ { \mathrm { m i n } } + C _ { 0 } \Delta _ { n } ) } } \\ & { } & \\ & { } & { \le \alpha ( b _ { n } ^ { \mathrm { m i n } } ) + L _ { \alpha } C _ { 0 } \Delta _ { n } \le a _ { n } ^ { \mathrm { m i n } } + ( 1 + L _ { \alpha } C _ { 0 } ) \Delta _ { n } , } \end{array}
$$

and similarly, for the second row,

$$
a _ { n } ^ { \operatorname* { m a x } } \leq \alpha ( \beta ( a _ { n } ^ { \operatorname* { m a x } } ) ) + ( L _ { \alpha } + C _ { 0 } ) \Delta _ { n } .
$$

In words, the extremal asks are approximate fixed points of $\alpha \circ \beta$ , up to an error of order $\Delta _ { n }$

We now show that lim inf $a _ { n } ^ { \mathrm { m i n } } \geq a ^ { - }$ and lim su $\operatorname { p } _ { n } a _ { n } ^ { \mathrm { m a x } } \leq a ^ { + }$ . If lim inf<sub>n</sub> $a _ { n } ^ { \mathrm { m i n } } < a ^ { - }$ then $a _ { n } ^ { \mathrm { m i n } }  \bar { a } < a ^ { - }$ along a subsequence, with $\bar { a } \ge \alpha ( 0 ) > 0$ , since $a _ { n } ^ { \mathrm { m i n } } \geq \alpha ( b _ { n } ^ { \mathrm { m i n } } ) - \Delta _ { n } \geq$ $\alpha ( 0 ) - \Delta _ { n }$ by Lemma 21 and the monotonicity of α. Since $\alpha \circ \beta$ is continuous on (0, 1], the

first inequality gives $\alpha ( \beta ( \bar { a } ) ) \leq \bar { a }$ in the limit, contradicting Lemma 17. Symmetrically, if $a _ { n } ^ { \mathrm { m a x } } \to \bar { a } > a ^ { + }$ along a subsequence, with $\bar { a } \le 1$ , the second inequality gives $\bar { a } \le \alpha ( \beta ( \bar { a } ) )$ , again contradicting Lemma 17.

The bid bounds follow from the ask bounds. By Lemma 21, $b _ { n } ^ { \mathrm { m i n } } \geq \beta ( a _ { n } ^ { \mathrm { m i n } } ) - C _ { 0 } \Delta _ { n }$ and $b _ { n } ^ { \mathrm { m a x } } \leq \beta ( a _ { n } ^ { \mathrm { m a x } } ) + \Delta _ { n }$ . Since $\beta$ is increasing and continuous, and $\beta ( a ^ { \pm } ) = b ^ { \pm }$

$$
\operatorname* { l i m } _ { n } \operatorname* { i n f } b _ { n } ^ { \operatorname* { m i n } } \geq b ^ { - } , \qquad \operatorname* { l i m } _ { n } \operatorname* { s u p } b _ { n } ^ { \operatorname* { m a x } } \leq b ^ { + } .
$$

Finally, every s ∈ supp(π<sub>n</sub>) has $a _ { n } ^ { \mathrm { m i n } } \leq a ^ { * } ( \mathbf { a } ) \leq a _ { n } ^ { \mathrm { m a x } }$ and $b _ { n } ^ { \mathrm { m i n } } \leq b ^ { * } ( { \bf b } ) \leq b _ { n } ^ { \mathrm { m a x } }$ . Since $\mathcal { R } ^ { \mathrm { N E } }$ is a rectangle, $D _ { \mathrm { e n v } } ( \mathbf { s } )$ is the sum of the distances of $b ^ { * } ( \mathbf { b } )$ from $[ b ^ { - } , b ^ { + } ]$ and of $a ^ { \ast } ( \mathbf { a } )$ from $[ a ^ { - } , a ^ { + } ]$ , so

$$
\begin{array} { r l } & { \underset { { \mathbf s } \in { \mathrm { s u p p } } ( \pi _ { n } ) } { \operatorname* { m a x } } D _ { \mathrm { e n v } } ( { \mathbf s } ) \leq \operatorname* { m a x } \{ a ^ { - } - a _ { n } ^ { \operatorname* { m i n } } , 0 \} + \operatorname* { m a x } \{ a _ { n } ^ { \operatorname* { m a x } } - a ^ { + } , 0 \} } \\ & { \qquad + \operatorname* { m a x } \{ b ^ { - } - b _ { n } ^ { \operatorname* { m i n } } , 0 \} + \operatorname* { m a x } \{ b _ { n } ^ { \operatorname* { m a x } } - b ^ { + } , 0 \} , } \end{array}
$$

and the right-hand side tends to zero by the four limits above.

Lemma 23 (Approximate Correlated Equilibria). Fix $\Delta > 0$ with $1 / \Delta \in \mathbb { N }$ . If $\pi _ { n }$ is a $\delta _ { n } – C E ~ o f \mathcal { G } _ { \Delta }$ for every n, with $\delta _ { n } \to 0$ , then

$$
\begin{array} { r } { \operatorname* { l i m } _ { n } \underset { \ b { n } } { \operatorname* { s u p } } \mathbb { E } _ { \mathbf { s } \sim \pi _ { n } } [ D _ { \mathrm { e n v } } ( \mathbf { s } ) ] \leq \varepsilon _ { \Delta } \ . } \end{array}
$$

Proof. Pass to a subsequence along which $\mathbb { E } _ { \mathbf { s } \sim \pi _ { n } } [ D _ { \mathrm { e n v } } ( \mathbf { s } ) ]$ converges to its limit superior. Since $S ^ { \Delta }$ is finite, the set of distributions over it is compact, so, passing to a further subsequence, $\pi _ { n }  \pi$ . Both the inequalities of Definition 20 and the expectation of $D _ { \mathrm { e n v } }$ are linear in the distribution, so, since $\delta _ { n } \to 0$ , π is a CE, and

$$
\operatorname* { l i m } _ { n } \boldsymbol { \mathrm { s u p } } \mathbb { E } _ { \mathbf { s } \sim \pi _ { n } } [ D _ { \mathrm { e n v } } ( \mathbf { s } ) ] = \mathbb { E } _ { \mathbf { s } \sim \pi } [ D _ { \mathrm { e n v } } ( \mathbf { s } ) ] \leq \operatorname* { m a x } _ { \mathbf { s } \in \mathbf { s u p p } ( \pi ) } D _ { \mathrm { e n v } } ( \mathbf { s } ) \leq \varepsilon _ { \Delta } ,
$$

where the last inequality holds by definition of $\varepsilon \Delta$ .