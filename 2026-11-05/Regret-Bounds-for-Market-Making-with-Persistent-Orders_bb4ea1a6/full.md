# Regret Bounds for Market Making with Persistent Orders

Nicolò Cesa-Bianchi Università degli Studi di Milano, Italy

NICOLO.CESA-BIANCHI@UNIMI.IT

Tommaso Cesari University of Ottawa, Ottawa, Canada

Roberto Colomboni ROBE School ofMathematics, University ofBristol, Bristol, United Kingdom

Luigi Foscari Università degli Studi di Milano, Italy

Vinayak Pathak Computational Rational Agents Laboratory (CORAL)

TCESARI@UOTTOWA.CA

RTO.COLOMBONI@BRISTOL.AC.UK

LUIGI.FOSCARI@UNIMI.IT

PATH.VINAYAK@GMAIL.COM

## Abstract

We study adversarial market making with orders that remain active until execution. The maker operates on two markets: a primary market where orders are made and a secondary market where the resulting positions are hedged. In each round, the maker posts a quote on the primary market before observing both primary and secondary market prices. An order executes when the primary price reaches its quote, and the resulting position is immediately offset at the secondary price. When both price sequences are chosen by an adversary over T rounds, we give a simple deterministic policy with regret of order log T against the best fixed quote (static comparator) and $\sqrt { T }$ against the best quote for each submission date (dynamic comparator), plus two terms measuring exposure to future primary and secondary prices. These rates remain optimal even when both exposure terms vanish and lower bounds show that the exposure terms are necessary. Our policy naturally extends to simultaneous bids and asks, where we show that static and dynamic regret are both characterized (up to constant factors) by the exposures terms (where the primary price exposure $V _ { T } ^ { \mathrm { t w o } }$ now accounts for both bids and asks) and a term of order $\sqrt { V _ { T } ^ { \mathrm { t w o } } \log T }$ . Our policy is simple, efficient and fully parameterless. We extend this setting by demonstrating how natural assumptions on the price processes lead to improved guarantees without changing the policy.

Keywords: market making, online learning, dynamic regret, persistent orders, adversarial prices

## 1. Introduction

A market maker quotes prices at which they are willing to buy and sell an asset. After a trade, they can offset their position in another (secondary) market. The difference between the quoted price and the price of this offsetting trade determines the maker’s profit. We revisit the simple sequential model of Cesa-Bianchi et al. (2025), where the maker submits a bid and ask pair to an incoming trader in each round. Unlike that setting, where an unaccepted quote does not remain available to later trades, we study what happens when: (1) orders remain active until they trade, (2) orders cannot be canceled. In our model the maker continues submitting orders even though the previous ones may still be outstanding. An ask therefore determines both a sale price and an execution date: a higher ask may earn more if it trades, but it may trade later. Hence, even though every executed position is immediately offset, outstanding orders remain exposed to future prices, and many orders can become exposed to the same later change in the hedging cost. Our model has two exogenous price sequences (one for the primary and one for the secondary market), both revealed after each decision. An outstanding ask executes the first time the primary price becomes at least as large as its quote, and the maker buys back the asset at that round’s secondary price. The execution delay of outstanding asks is monotonic in their quote, as a lower quote cannot execute later than a higher one. Finally, our setting also includes bids, which are defined symmetrically to asks.

Our results. We study two notions of regret: static regret, where we compare the maker with the best fixed quote submitted on every round, and dynamic regret, where the comparator chooses the best quote separately on every round and knowing the entire price path. In both cases, the regret after T rounds is controlled by two measures of price exposure plus a horizon-dependent term. The first measure is defined on the primary prices and counts the width of the intervals of quotes that wait and eventually execute. The second measure is defined on the secondary prices and records the possible change between the last observed hedging price and a future execution price. These measures can be small even when some orders wait a long time. The horizon-dependent term, instead, is Θ(log T) for static regret and $\Theta \big ( \sqrt { T } \big )$ for dynamic regret. These bounds are obtained by the same policy using two different learning rates that depend neither on the horizon nor on the price exposures. We also prove lower bounds showing that this difference between static and dynamic regret is necessary, even when the primary price never increases and the secondary price is constant. Separate constructions also establish the need for both exposure terms. Our policy is simple, efficient, and naturally extends to bid-ask pairs.

When the learner can post both bids and asks, the horizon-dependent term vanishes and is now replaced by a primary exposure term on both sides which controls both upward and downward movements. While this does not mean that the two-sided problem is easier than the one-sided counterparts, we characterize both static and dynamic regret by the same rate, meaning that the adversary does not gain by adaptively choosing their quotes. We naturally extend the strategy for the ask-only case and show that the resulting policy is optimal for both types of regret simultaneously.

Further, in Appendix A we investigate the impact of additional assumptions on the price processes and their impact on regret in the ask-only case. In particular, we study what happens when both processes orbit around fixed prices. Then we study the case in which the secondary price is equal to the primary, but with a one-round lag (Lo and MacKinlay, 1990). In both cases, the policy naturally ensures improved guarantees without knowledge of the assumptions.

## 2. Related work

Adversarial formulations provide an alternative foundation for financial problems traditionally studied under stochastic price models. A seminal work in the same vein is by Abernethy et al. (2013), who study option hedging as a sequential game against an adversary whose price changes satisfy a cumulative quadratic volatility budget and a bound on individual jumps. For convex, Lipschitz, eventually linear payoffs, they show that the minimax option price converges to the Black–Scholes price under suitable vanishing-jump conditions. Their result demonstrates that a classical pricing benchmark can emerge without assuming a stochastic law for price evolution, motivating the study of financial decisions through guarantees expressed in terms of price-path properties. We pursue this perspective for market making with persistent orders.

The model perhaps closest to ours is by Cesa-Bianchi et al. (2025). In both settings, a maker quotes a bid and an ask, and offsets each accepted trade in a secondary market. In their setting, the trader’s private valuation determines acceptance and plays the role of our primary price, while the market price determines the offsetting cost and plays the role of our secondary price. They assume that an accepted quote trades and is immediately hedged, while an unaccepted quote expires. We study the opposite case in which an unfilled order remains available to later trades, cannot be canceled, and may never execute. Both models hedge immediately after execution.

The other difference between the two settings is the feedback. There, the maker observes the market price and whether the trader buys, sells, or does neither, but never observes the valuation. Their algorithms learn profitable quotes from this partial feedback by combining methods for firstprice auctions and dynamic pricing, with an additional step that keeps the bid below the ask. Here both prices are revealed after every decision, even when nothing trades. This also corresponds to a form of partial observability, as the maker can compute the counterfactual reward of past quotes that would have executed at the current primary price. While the main difficulty in the setting of Cesa-Bianchi et al. (2025) arises from the partial feedback, here the difficulty is tied to the exposure to the unknown future prices. Maran and Restelli (2026) then study an intermediate feedback mode in which the trader’s valuation is revealed only when no trade occurs.

Abernethy and Kale (2013) on the other hand study adversarial market making with observed prices, competing with a class of spread-based strategies. Their learner must account for the inventory accumulated by each strategy when switching between them. Our immediate hedge removes this inventory state, while persistence creates commitments through outstanding orders. Classical models instead explain spreads through adverse selection (Glosten and Milgrom, 1985) or inventory risk (Avellaneda and Stoikov, 2008). Here future execution prices are the source of exposure despite immediate hedging.

Our work is related to the extensive literature on online learning with delayed feedback, in particular when delays are allowed to depend on the action, van der Hoeven and Cesa-Bianchi (2022) study this case in both the experts and bandit settings. More generally, Joulani et al. (2013) give reductions from delayed to standard online learning, Quanrud and Khashabi (2015) analyze gradient and perturbed-leader methods with adversarial delays, and Zimmert and Seldin (2020) obtain optimal regret bounds for adversarial bandits with arbitrary delays. These works show how delayed observations affect learning, often through bounds involving cumulative delay or outstanding feedback. Vernade et al. (2017) study stochastic bandits motivated by online advertising, where an ad may lead to a purchase realized only after a random, possibly infinite, delay. In our model, the execution time depends on both the posted quote and the primary price path, and determines which secondary price enters the reward, Thus, delays affect the payoff and the time feedback is available.

The idea of a delayed reward connects to the notion of losses with memory (e.g., switching costs), where the instantaneous loss depends not only on the current action, but also on the actions in a window on the past. Arora et al. (2012); Cesa-Bianchi et al. (2013) show that a memory buffer of size $o ( T ^ { 1 / 3 } )$ is necessary and sufficient for sublinear regret, even with full feedback, when no other restrictions are imposed on the loss functions. In contrast, our instantaneous payoff is determined by a specific function of the entire past history parameterized by the adversarial choice of the primary and secondary price paths. As a result, our regret can be sublinear, depending on how fast the resulting price exposures grow with time.

Finally, our notion of primary market exposure is related to the notion of upcrossing studied by Koolen and Vovk (2014). However, rather than bounding the investor’s regret, their goal is to derive an exact characterization of how much wealth an investor can guarantee from the occurrence of price upcrossings, without knowing when or where those upcrossings will occur.

## 3. Setting

Fix a horizon $T \geq 2$ . An adversary chooses primary prices $p _ { 1 } , \dotsc , p _ { T } \in [ 0 , 1 ]$ and secondary prices $m _ { 1 } , \ldots , m _ { T } \in [ 0 , 1 ]$ . If the learner is randomized, then the adversary may be nonoblivious, i.e., adapt to the learner’s past actions. For simplicity, we initially restrict our analysis to ask quotes only, and later extend it to ask-bid pairs. On round t, the maker submits one unit at a price $A _ { t } \in [ 0 , 1 ]$ using past observations and possibly internal randomness. The pair $( p _ { t } , m _ { t } )$ is then revealed. Every outstanding ask $A _ { s } ,$ , for $s \leq t ,$ executes if $A _ { s } \ \leq \ p _ { t }$ . The maker then sells one unit at $A _ { s }$ and immediately buys one unit in the secondary market at $m _ { t }$ . Each order executes at most once and orders cannot be canceled.

Both the primary and secondary price are revealed at the end of the round even when no trading happens, therefore the maker observes more than bandit feedback, although the final reward of a submitted order can remain unknown, as an order still outstanding after the time horizon receives zero reward. Bids, and the constraint relating them to asks, are introduced in Section 7.

For an ask a submitted at t, its execution time is $\tau _ { t } ( a ) : = \operatorname* { m i n } \{ s \in \{ t , \dots , T \} : p _ { s } \geq a \}$ where the minimum of the empty set is ∞. When the ask executes, its execution cost is $c _ { t } ( a ) : = m _ { \tau _ { t } ( a ) } .$ while its reward is

$$
r _ { t } ( a ) : = { \left\{ \begin{array} { l l } { a - c _ { t } ( a ) , } & { \tau _ { t } ( a ) \leq T , } \\ { 0 , } & { \tau _ { t } ( a ) = \infty . } \end{array} \right. }\tag{1}
$$

The subscript records the submission date, not the execution date; in particular, several asks may be traded on the same round. The boundary ask $a = 1$ always has nonnegative reward because either it does not trade or it earns $1 - c _ { t } ( 1 ) \geq 0$

Let $p _ { t } ^ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { t \pm s \leq T } p _ { s }$ be the highest remaining primary price, using this notation, the reward (1) can be equivalently defined by $r _ { t } ( a ) : = ( a - c _ { t } ( a ) ) \mathbf { 1 } \{ a \leq p _ { t } ^ { \operatorname* { m a x } } \}$ . An ask submitted at t executes by T if $a \leq p _ { t } ^ { \mathrm { m a x } }$ , note that the sequence $( p _ { t } ^ { \operatorname* { m a x } } ) _ { t }$ is nonincreasing. Importantly, if $a \leq a ^ { \prime }$ then $\tau _ { t } ( a ) \leq \tau _ { t } ( a ^ { \prime } )$ , meaning that a lower ask cannot wait longer than a higher one submitted on the same round.

Benchmarks. We define regret with respect to two different comparators. Thefixed benchmark submits the same ask on every round, while the dynamic benchmark may choose a different ask for each submission date, and both obey the same execution and horizon rules as the maker. Denote the optimal comparators and the corresponding regrets as

$$
\mathrm { O P T } _ { T } : = \operatorname* { m a x } _ { a \in [ 0 , 1 ] } \sum _ { t = 1 } ^ { T } r _ { t } ( a ) \ : , \qquad \mathrm { O P T } _ { T } ^ { \mathrm { d y n } } : = \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { a \in [ 0 , 1 ] } r _ { t } ( a ) \ : ,\tag{2}
$$

$$
R _ { T } : = \mathbb { E } \left[ \mathrm { O P T } _ { T } - \sum _ { t = 1 } ^ { T } r _ { t } ( A _ { t } ) \right] , \qquad R _ { T } ^ { \mathrm { d y n } } : = \mathbb { E } \left[ \mathrm { O P T } _ { T } ^ { \mathrm { d y n } } - \sum _ { t = 1 } ^ { T } r _ { t } ( A _ { t } ) \right] .\tag{3}
$$

Expectations are over the (possible) maker’s randomness. The safe boundary ask $a = 1$ ensures that both benchmarks are nonnegative, and $R _ { T } \leq R _ { T } ^ { \mathrm { d y n } }$ . The maxima in (2) exist. To see this, round any ask up to the smallest point of $\{ p _ { 1 } , \dotsc , p _ { T } , 1 \}$ that is at least that ask. For every submission date, this preserves execution, while maximizing the sale price. Thus a finite set contains an optimizer for either benchmark.

Market exposure. Define the primary and secondary cumulative exposures as

$$
V _ { T } : = \sum _ { t = 1 } ^ { T - 1 } ( p _ { t } ^ { \operatorname* { m a x } } - p _ { t } ) , \qquad w _ { t } : = \operatorname* { m a x } _ { t < s \leq T } | m _ { s } - m _ { t } | , \qquad W _ { T } : = \sum _ { t = 1 } ^ { T - 1 } w _ { t } .\tag{4}
$$

The asks submitted at t that execute later live in the interval $( p _ { t } , p _ { t } ^ { \operatorname* { m a x } } ]$ of length $p _ { t } ^ { \operatorname* { m a x } } - p _ { t }$ , thus $V _ { T }$ measures the total width of delayed execution opportunities, rather than the number of rounds spent waiting. The quantity $w _ { t }$ bounds the change from $m _ { t }$ , which is the last observed secondary price when choosing $A _ { t + 1 }$ , to any possible later execution cost. Both exposures are properties of the price paths and are unknown to the maker. Intuitively, low exposure terms mean more predictable price sequences, which facilitate trading. For example, a nonincreasing primary path and a constant secondary path have $V _ { T } = W _ { T } = 0 .$ . A single secondary-price change of size δ at round $k ,$ followed by a constant price, gives $W _ { T } = ( k - 1 ) \delta$ . The same change therefore has a larger exposure when it occurs later. More generally,

$$
W _ { T } \leq \sum _ { k = 2 } ^ { T } ( k - 1 ) | m _ { k } - m _ { k - 1 } |\tag{5}
$$

by applying the triangle inequality to each future movement. In the following, we use $H _ { t } : =$ $\textstyle \sum _ { s = 1 } ^ { t } 1 / s = \Theta ( \log t )$ for the harmonic sum and $( x ) _ { + } : = \operatorname* { m a x } \{ x , 0 \}$ for the positive part.

## 4. Static regret

We now provide a policy to compete against the best fixed quote achieving $\mathrm { O P T } _ { T }$ . Initially the policy quotes $A _ { 1 } = 1$ . Then, for any $t \geq 2$ and for any sequence $\left( \lambda _ { t } \right) _ { t \geq 2 }$ of numbers $\lambda _ { t } \in [ 0 , 1 ]$ , the policy sets

$$
A _ { t } = m _ { t - 1 } + ( 1 - \lambda _ { t } ) ( p _ { t - 1 } - m _ { t - 1 } ) _ { + } .\tag{6}
$$

The policy uses the last observed primary and secondary prices as proxies for quote and cost. If these proxies suggest a positive profit $( p _ { t - 1 } > m _ { t - 1 } )$ , then the policy quotes a price $A _ { t }$ in between the two prices, otherwise the policy sets $A _ { t } = m _ { t - 1 }$ . A larger $\lambda _ { t }$ sacrifices more margin when an order fills, but makes it harder for a small threshold decrease to prevent execution. For static regret, where we compete against the best fixed quote, we use $\lambda _ { t } = 1 / t .$ , so that $A _ { t }$ converges to max $\{ p _ { t - 1 } , m _ { t - 1 } \}$ as t grows.

Note that this policy requires constant time and memory to choose each new quote and does not need the horizon or either exposure as input.

Theorem 1. For every pair of price paths, policy (6) with $\lambda _ { t } = 1 / t$ satisfies

$$
R _ { T } \leq H _ { T } + V _ { T } + 2 W _ { T } .
$$

In particular, $V _ { T } + W _ { T } = o ( T )$ is sufficient for sublinear regret. When the primary price never increases and the secondary price is constant, the bound is $H _ { T } = \Theta ( \log T )$ , even if the primary price falls by a constant amount.

We prove the theorem using the last observed secondary price $m _ { t - 1 }$ as a proxy for the future execution costs $c _ { t } ( \cdot )$ . Define the surrogate rewards

$$
r _ { 1 } ^ { \prime } ( a ) : = ( a - 1 ) \mathbf { 1 } \{ a \leq p _ { 1 } ^ { \operatorname* { m a x } } \} , \quad \quad r _ { t } ^ { \prime } ( a ) : = ( a - m _ { t - 1 } ) \mathbf { 1 } \{ a \leq p _ { t } ^ { \operatorname* { m a x } } \} \quad ( t \geq 2 ) .\tag{7}
$$

Note that the condition determining execution remains unchanged. The fixed surrogate benchmark is $\begin{array} { r } { \mathrm { O P T } _ { T } ^ { \prime } : = \operatorname* { m a x } _ { a } \sum _ { t } r _ { t } ^ { \prime } ( a ) } \end{array}$ . Let $\begin{array} { r } { C _ { t } : = 1 + \sum _ { s = 1 } ^ { t - 1 } m _ { s } } \end{array}$ be the cumulative reference cost. Because $p _ { t } ^ { \mathrm { m a x } }$ is nonincreasing, a fixed ask a with $p _ { 1 } ^ { \mathrm { m a x } } \geq a$ executes precisely in a prefix [t] of the $T$ rounds for some $t \geq 1$ . The surrogate profit obtained by playing a every round is ta $- C _ { t } \leq t p _ { t } ^ { \mathrm { m a x } } - C _ { t }$ Hence we upper bound the surrogate optimum by considering the most profitable prefix

$$
\mathrm { O P T } _ { T } ^ { \prime } \leq \operatorname* { m a x } \left\{ 0 , \underset { 1 \leq t \leq T } { \operatorname* { m a x } } ( t p _ { t } ^ { \operatorname* { m a x } } - C _ { t } ) \right\} .\tag{8}
$$

The regret on the surrogate rewards is best understood using a clairvoyant policy: this is the policy (6) run on the surrogate rewards with $p _ { t - 1 } ^ { \mathrm { m a x } }$ instead of $p _ { t - 1 }$ . The policy sets $A _ { 1 } ^ { \circ } = 1$ and, for all $t \geq 2$ and any sequence $\lambda _ { t } \in [ 0 , 1 ]$

$$
A _ { t } ^ { \circ } = m _ { t - 1 } + ( 1 - \lambda _ { t } ) ( p _ { t - 1 } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) _ { + } , \qquad Q _ { t } ^ { \circ } : = \sum _ { s = 1 } ^ { t } r _ { s } ^ { \prime } ( A _ { s } ^ { \circ } ) .\tag{9}
$$

From (7) one can see that surrogate rewards $r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } )$ are always nonnegative.

Let $R _ { T } ( A )$ be the regret of policy $( 6 ) , R _ { T } ^ { \prime }$ the surrogate regret, and $R _ { T } ^ { \prime } ( A ^ { \circ } )$ the surrogate regret of the clairvoyant policy (9). Theorem 1 is an immediate consequence of the following results:

$$
R _ { T } ( A ) \leq R _ { T } ^ { \prime } ( A ) + 1 + 2 W _ { T } \ ,\tag{Lemma 2}
$$

$$
R _ { T } ^ { \prime } ( A ) \leq R _ { T } ^ { \prime } ( A ^ { \circ } ) + V _ { T } \ ,\tag{Lemma 3}
$$

$$
R _ { T } ^ { \prime } ( A ^ { \circ } ) \leq H _ { T } - 1 .\tag{Lemma 4}
$$

Lemma 2. For any choice of $\left( \lambda _ { t } \right) _ { t \geq 2 } ,$ policy (6) satisfies $R _ { T } ( A ) \leq R _ { T } ^ { \prime } ( A ) + 1 + 2 W _ { T }$

Proof. For every ask and $t \geq 2$

$$
| r _ { t } ( a ) - r _ { t } ^ { \prime } ( a ) | = | m _ { t - 1 } - m _ { \tau _ { t } ( a ) } | \mathbf { 1 } \{ a \leq p _ { t } ^ { \mathrm { m a x } } \} \leq w _ { t - 1 } .\tag{10}
$$

$\mathrm { A t } \ t = 1 , r _ { 1 } ( a ) \leq r _ { 1 } ^ { \prime } ( a ) + 1$ , while the learner has $r _ { 1 } ( 1 ) \geq r _ { 1 } ^ { \prime } ( 1 ) = 0$ . Thus

$$
\mathrm { O P T } _ { T } - \sum _ { t = 1 } ^ { T } r _ { t } ( A _ { t } ) \leq \mathrm { O P T } _ { T } ^ { \prime } - \sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ) + 1 + 2 W _ { T } \ ,
$$

concluding the proof.

Next, we prove $R _ { T } ^ { \prime } ( A ) \leq R _ { T } ^ { \prime } ( A ^ { \circ } ) + V _ { T }$ via the following lemma.

Lemma 3. For any choice of $\left( \lambda _ { t } \right) _ { t \geq 2 } ,$ , policies (6) and (9) satisfy

$$
\sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ) \geq \sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) - V _ { T } .
$$

Proof. By comparing (6) and (9) one can see that for all $t \geq 2 .$ , the actual quote $A _ { t }$ lies between $m _ { t - 1 }$ and $A _ { t } ^ { \circ }$ . Lowering an ask within this interval preserves a nonnegative surrogate margin and

can only make execution more likely. If both quotes $A _ { t } , A _ { t } ^ { \circ }$ execute, the loss is their price difference $A _ { t } ^ { \circ } - A _ { t } ; { \operatorname { i f } } A _ { t } ^ { \circ }$ does not execute, then $r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) = 0 .$ . Therefore

$$
r _ { t } ^ { \prime } ( A _ { t } ) \geq r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) - ( A _ { t } ^ { \circ } - A _ { t } ) \geq r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) - ( 1 - \lambda _ { t } ) ( p _ { t - 1 } ^ { \operatorname* { m a x } } - p _ { t - 1 } ) .
$$

Summing over $t = 1 , \dots , T$ gives

$$
\sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ) - \sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) \geq - \sum _ { t = 1 } ^ { T - 1 } ( 1 - \lambda _ { t + 1 } ) ( p _ { t } ^ { \operatorname* { m a x } } - p _ { t } ) \geq - \sum _ { t = 1 } ^ { T - 1 } ( p _ { t } ^ { \operatorname* { m a x } } - p _ { t } ) = - V _ { T } ,
$$

concluding the proof.

Finally, we prove $R _ { T } ^ { \prime } ( A ^ { \circ } ) \leq H _ { T } - 1$ . Recall that the quantity $t p _ { t } ^ { \operatorname* { m a x } } - C _ { t }$ in (8) bounds the surrogate profit of a fixed ask that executes for the first t submission dates.

Lemma 4. For every $t \in [ T ]$ , if the auxiliary policy (9) is run with $\lambda _ { t } = 1 / t$ , then it satisfies $Q _ { t } ^ { \circ } \geq t p _ { t } ^ { \operatorname* { m a x } } - C _ { t } - H _ { t } + 1$ . Consequently, $R _ { T } ^ { \prime } ( A ^ { \circ } ) \leq H _ { T } - 1$

The proof is given in Appendix B.

The primary price exposure $V _ { T }$ admits a natural interpretation in terms of upcrossings. A price path upcrosses an interval $[ a , b ]$ if it visits a price at most a before subsequently reaching a price at least b. Hence, $V _ { T }$ is the total mass of all upcrossings $\left[ p _ { t } , p _ { t } ^ { \mathrm { m a x } } \right]$ of the primary price path. Koolen and Vovk (2014) use upcrossings to characterize exactly which wealth guarantees can be achieved by an online trading strategy. While their guarantees only depend on the existence and magnitude of price upcrossings, $V _ { T }$ also captures their multiplicity across outstanding orders. In particular, a single late price increase may generate exposure proportional to $T$ by executing many previously submitted orders.

## 5. Dynamic regret

In this section we extend the previous analysis to dynamic regret (3). Surprisingly, the optimal policy (6) for static regret is also optimal for dynamic regret provided the tuning $\lambda _ { t } = 1 / t$ is changed to $\lambda _ { t } = 1 / \sqrt { t - 1 }$ . Indeed, the choice $\lambda _ { t } = 1 / t$ in (6) can lead to linear dynamic regret, even when both exposures vanish (Appendix C gives an example). Next, we state and prove our main result for this section.

Theorem 5. For every pair ofprice paths, policy (6) with $\lambda _ { t } = 1 / \sqrt { t - 1 }$ satisfies

$$
R _ { T } ^ { \mathrm { { d y n } } } \leq 1 + 3 \sqrt { T - 1 } + V _ { T } + 2 W _ { T } .
$$

Proof. Similarly to the proof of Theorem 1, we use surrogate rewards (7) and the clairvoyant strategy (9). We start by computing the optimal dynamic strategy for the surrogate reward. For any $t \geq 2$ , if $p _ { t } ^ { \operatorname* { m a x } } < m _ { t - 1 }$ , the safe boundary ask 1 attains zero reward. Otherwise the ask $p _ { t } ^ { \mathrm { m a x } }$ attains $\left( p _ { t } ^ { \operatorname* { m a x } } - m _ { t - 1 } \right)$ . Hence, the surrogate reward of the optimal dynamic strategy at each $t \geq 2$ is $( p _ { t } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) _ { + }$

Now replace $p _ { t - 1 }$ in (6) by $p _ { t - 1 } ^ { \mathrm { m a x } }$ to obtain the clairvoyant quote $A _ { t } ^ { \circ }$ . By Lemma 3,

$$
\sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ) \geq \sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) - V _ { T } .\tag{11}
$$

Hence, the additional surrogate dynamic regret from using the actual quotes $A _ { t }$ is at most $V _ { T }$ . Fix $t \geq 2 .$ , and abbreviate $x = ( p _ { t - 1 } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) _ { + }$ and $y = ( p _ { t } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) _ { + }$ . We proceed by case analysis.

1. If $A _ { t } ^ { \circ }$ executes, Then the surrogate reward is $( 1 - \lambda _ { t } ) x$ . Since $y \le x \le 1$ , the instantaneous dynamic regret is $y - ( 1 - \lambda _ { t } ) x \leq x - ( 1 - \lambda _ { t } ) x \leq \lambda _ { t }$

2. If $A _ { t } ^ { \circ }$ does not execute, when $y = 0$ , the instantaneous regret is zero, otherwise, $y > 0$ and

$$
( 1 - \lambda _ { t } ) x = ( 1 - \lambda _ { t } ) ( p _ { t - 1 } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) _ { + } > p _ { t } ^ { \operatorname* { m a x } } - m _ { t - 1 } = y > 0
$$

$\mathbf { s o } ,$ in this case, $p _ { t - 1 } ^ { \operatorname* { m a x } } - p _ { t } ^ { \operatorname* { m a x } } = x - y > \lambda _ { t } x \geq \lambda _ { t } y$ and the instantaneous regret is therefore at most $y < ( p _ { t - 1 } ^ { \operatorname* { m a x } } - p _ { t } ^ { \operatorname* { m a x } } ) / \lambda _ { t }$

Combining the two results, we have that

$$
\operatorname* { m a x } _ { a \in [ 0 , 1 ] } r _ { t } ^ { \prime } ( a ) - r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) \leq \lambda _ { t } + ( p _ { t - 1 } ^ { \operatorname* { m a x } } - p _ { t } ^ { \operatorname* { m a x } } ) / \lambda _ { t }\tag{12}
$$

Therefore,

$$
\begin{array} { r l } {  { R _ { T } ^ { \mathrm { d y n } } \le 1 + \sum _ { t = 1 } ^ { T } \displaystyle \operatorname* { m a x } _ { a \in [ 0 , 1 ] } r _ { t } ^ { \prime } ( a ) - \sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ) + 2 W _ { T } } } \\ & { \le 1 + \displaystyle \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { a \in [ 0 , 1 ] } r _ { t } ^ { \prime } ( a ) - \sum _ { t = 1 } ^ { T } r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) + V _ { T } + 2 W _ { T } } \\ & { \le 1 + \displaystyle \sum _ { t = 2 } ^ { T } ( \lambda _ { t } + \frac { p _ { t - 1 } ^ { \mathrm { m a x } } - p _ { t } ^ { \mathrm { m a x } } } { \lambda _ { t } } ) + V _ { T } + 2 W _ { T } } \\ & { < 1 + 2 \sqrt { T - 1 } + \sqrt { T - 1 } + V _ { T } + 2 W _ { T } } \end{array}\tag{proof of Lemma 2}
$$

(using (11))

(using (12))

where we used $\begin{array} { r } { \sum _ { t = 2 } ^ { T } \lambda _ { t } \leq 2 \sqrt { T - 1 } } \end{array}$ and

$$
\begin{array} { r l } {  { \sum _ { t = 2 } ^ { T } ( p _ { t - 1 } ^ { \operatorname* { m a x } } - p _ { t } ^ { \operatorname* { m a x } } ) \sqrt { t - 1 } = p _ { 1 } ^ { \operatorname* { m a x } } - p _ { T } ^ { \operatorname* { m a x } } \sqrt { T - 1 } + \sum _ { t = 2 } ^ { T - 1 } p _ { t } ^ { \operatorname* { m a x } } ( \sqrt { t } - \sqrt { t - 1 } ) } } \\ & { \leq p _ { 1 } ^ { \operatorname* { m a x } } - p _ { T } ^ { \operatorname* { m a x } } \sqrt { T - 1 } + p _ { 1 } ^ { \operatorname* { m a x } } \sqrt { T - 1 } - p _ { 1 } ^ { \operatorname* { m a x } } } \\ & { \leq \sqrt { T - 1 } . } \end{array}
$$

This concludes the proof.

## 6. Lower bounds

We now show why changing the comparator changes the baseline rate from an harmonic to a squareroot term, and why both exposure terms are needed. Every lower bound applies to randomized policies and can be realized by a deterministic oblivious price path.

To start, we show that the rates provided for static and dynamic regret are tight in $T ,$ , showing that the dynamic problem is significantly harder.

Theorem 6. For every policy and for every $T \ \geq \ 2 ,$ , there is a pair of price paths such that $\begin{array} { r } { R _ { T } \geq \frac { 3 } { 1 6 } ( H _ { T } - 1 ) } \end{array}$ , and also $R _ { T } ^ { \mathrm { d y n } } \geq \frac { 3 } { 1 6 } \sqrt { T - 1 }$ for a possibly different pair ofpaths. In both cases, the price paths are such that $V _ { T } = W _ { T } = 0$

Proof. Set $m _ { t } = 0$ for all t. The instance uses a sequence of values $\delta _ { t } \in ( 0 , 1 )$ which are specified below, these values define a random primary price process as

$$
p _ { 1 } = 1 , \qquad p _ { t } = \left\{ { \begin{array} { l l } { ( 1 - \delta _ { t } ) p _ { t - 1 } } & { \mathrm { w . p . ~ } \delta _ { t } } \\ { p _ { t - 1 } } & { \mathrm { w . p . ~ } 1 - \delta _ { t } } \end{array} } \right. \mathrm { ~ f o r ~ } t \geq 2 .
$$

Sample this sequence before play independently for each round. Every path is nonincreasing, so missed asks never execute later and $V _ { T } = W _ { T } = 0$

Given the full past with $p _ { t - 1 } = x$ , in expectation any ask earns at most $( 1 - \delta _ { t } ) x$ , because lower quotes earn no more even if they trade, quotes in $( ( 1 - \delta _ { t } ) x , x ]$ trade with probability $1 - \delta _ { t }$ , and higher quotes never trade. The same holds once we randomize the quote, write $\mu _ { t } : = \mathbb { E } [ p _ { t } ]$ and set $\mu _ { 0 } = \mu _ { 1 } = 1$ and $\delta _ { 1 } = 0$ . Then $\mu _ { t } = ( 1 - \delta _ { t } ^ { 2 } ) \mu _ { t - 1 }$ , and the learner’s expected total reward is at most $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( 1 - \delta _ { t } ) \mu _ { t - 1 } } \end{array}$ . We know consider the two cases.

• For dynamic regret, take $\delta _ { t } = \delta : = 1 / ( 2 \sqrt { T - 1 } )$ for $t \geq 2$ . Because the secondary price is zero, the comparator quotes $p _ { t }$ and earns it on every round, so its expected reward is $\mu _ { t }$ , while the learner earns at most $( 1 - \delta ) \mu _ { t - 1 }$ . The expected regret of round t is therefore

$$
\mu _ { t } - ( 1 - \delta ) \mu _ { t - 1 } = ( 1 - \delta ^ { 2 } ) \mu _ { t - 1 } - ( 1 - \delta ) \mu _ { t - 1 } = \delta ( 1 - \delta ) \mu _ { t - 1 } .
$$

Since $1 - \delta \ge 1 / 2$ and $\mu _ { t - 1 } = ( 1 - \delta ^ { 2 } ) ^ { t - 2 } \geq 1 - ( t - 2 ) \delta ^ { 2 } \geq 3 / 4$ , summing over $t \geq 2$ yields $\mathbb { E } [ R _ { T } ^ { \mathrm { d y n } } ] \geq 3 \sqrt { T - 1 } / 1 6$

• For fixed regret, take $\delta _ { t } ~ = ~ 1 / ( 2 t )$ for $t \ > \ 2$ . Now $\textstyle \mu _ { t } \ = \ \prod _ { s = 2 } ^ { t } ( 1 - 1 / ( 4 s ^ { 2 } ) ) \ \geq \ 1 \ -$ $\textstyle { \frac { 1 } { 4 } } \sum _ { s = 2 } ^ { \infty } s ^ { - 2 } \geq 3 / 4$ , using $\begin{array} { r } { \sum _ { s = 2 } ^ { \infty } s ^ { - 2 } \le \int _ { 1 } ^ { \infty } x ^ { - 2 } d x = 1 } \end{array}$ . The fixed quote $p _ { T }$ earns $T p _ { T }$ through immediate execution on every round, so $\mathrm { O P T } _ { T } \geq T p _ { T }$ . The identity

$$
\sum _ { t = 1 } ^ { T } \mu _ { t - 1 } - T \mu _ { T } = \sum _ { t = 1 } ^ { T } t ( \mu _ { t - 1 } - \mu _ { t } ) = \sum _ { t = 1 } ^ { T } t \delta _ { t } ^ { 2 } \mu _ { t - 1 }
$$

therefore gives

$$
\mathbb { E } [ R _ { T } ] \geq T \mu _ { T } - \sum _ { t = 1 } ^ { T } ( 1 - \delta _ { t } ) \mu _ { t - 1 } = \sum _ { t = 1 } ^ { T } \mu _ { t - 1 } ( \delta _ { t } - t \delta _ { t } ^ { 2 } ) = \frac { 1 } { 4 } \sum _ { t = 2 } ^ { T } \frac { \mu _ { t - 1 } } { t } \geq \frac { 3 } { 1 6 } ( H _ { T } - 1 ) .
$$

This concludes the proof.

Together with Theorems 1 and 5, these bounds give rates $\Theta ( \log T )$ and $\Theta ( { \sqrt { T } } )$ on the class $V _ { T } =$ $W _ { T } = 0$ , for the two different comparators.

The next construction shows that the dependency on the primary exposure is tight, it works by postponing every profitable execution until the last round (maximum exposure). It is a posted-price problem with no chance to learn the terminal valuation before choosing any of the prices.

Theorem 7. For every policy, $T \geq 2$ , and $v \in ( 0 , T - 1 ]$ , there are deterministic price paths with $V _ { T } \leq v , W _ { T } = 0 ,$ , and $R _ { T } \ge v / e$

Take zero primary prices until the final round and zero secondary prices. Write $\beta = v / ( T - 1 )$ and let the terminal price X lie in $[ \beta / e , \beta ]$ . The construction has $\mathbb { P } ( X \ge a ) = \beta / ( e a )$ on this interval, so every ask there has expected reward aP $( X \geq a ) = \beta / e$ which is fixed for all quotes. Asks outside this interval earn no more. Meanwhile, $\mathbb { E } [ X ] = 2 \beta / e$ , and the comparator quotes X on every round. Thus the learner earns at most half the comparator’s expected reward per order. See Appendix D.2 for all the details.

Because secondary exposure can be large even when primary exposure is arbitrarily small, we next present a separate lower bound showing that, if the learner does not know whether the secondary price will rise or fall, they are forced to accumulate secondary exposure.

Theorem 8. For every policy, $T \geq 2 _ { \mathrm { { i } } }$ , and $w \in [ 0 , ( T - 1 ) / 4 ]$ , there are deterministic price paths with $W _ { T } = w , V _ { T } = w / 4 T$ and $R _ { T } \geq 3 w / 8 .$ Additionally, for w $> 0 ,$ , the primary exposure can be made arbitrarily smaller while retaining $W _ { T } = w$ and the regret bound.

Initially the prices are $3 / 4$ and $1 / 2$ . On the final round the primary price rises slightly, while the secondary price either rises or falls by the same amount $w / ( T - 1 )$ . The fixed comparator delays trading when costs fall and trades early when they rise, while the maker must quote before knowing what is going to happen. Appendix D.3 gives the proof. Taking w proportional to $T$ yields linear regret even as $V _ { T }$ tends to zero.

The last two theorems also lower-bound dynamic regret, since $R _ { T } ^ { \mathrm { { d y n } } } \geq R _ { T }$

Next we show that the previous three lower bounds can be easily combined to show that the dependency of policy (6) on time and on both primary and secondary exposure terms is optimal.

Theorem 9. For every policy, $T \geq 2 , v \in ( 0 , T - 1 ]$ , and $w \in [ 0 , ( T - 1 ) / 4 ]$ , there is a deterministic price path with $V _ { T } \le v$ and $W _ { T } \leq$ w such that

$$
R _ { T } \geq \operatorname* { m a x } \left\{ \frac { 3 } { 1 6 } ( H _ { T } - 1 ) , \frac { v } { e } , \frac { 3 w } { 8 } \right\} = \Omega ( \log T + v + w ) .
$$

Likewise, there is such a path with

$$
R _ { T } ^ { \mathrm { { d y n } } } \geq \operatorname* { m a x } \left\{ \frac { 3 } { 1 6 } \sqrt { T - 1 } , \frac { v } { e } , \frac { 3 w } { 8 } \right\} = \Omega ( \sqrt { T } + v + w ) .
$$

Proof. Apply Theorems 6, 7, and 8. Each construction can be chosen within the budgets $V _ { T } \le v$ and $W _ { T } \le w$ , so the worst-case regret over this class is at least the maximum of the three lower bounds. Finally, the maximum of three nonnegative quantities is at least one third of their sum.

## 7. Simultaneous bids and asks

We now focus on the case in which the maker submits both a bid $B _ { t }$ and an ask $A _ { t }$ , with $0 \leq B _ { t } \leq$ $A _ { t } \le 1$ . A bid b submitted at t executes at $\tau _ { t } ^ { \mathrm { b i d } } ( b ) : = \operatorname* { m i n } \{ s \in \{ t , \dots , T \} : p _ { s } \leq b \}$ , with an empty minimum interpreted $\mathbf { a s } \infty .$ . Its reward $r _ { t } ^ { \mathrm { b i d } } ( b )$ is $m _ { \tau _ { t } ^ { \mathrm { b i d } } ( b ) } - b$ if executed and zero otherwise. We rename $r _ { t } ( a )$ to $r _ { t } ^ { \mathrm { a s k } } ( a )$ . The bid 0 and the ask 1 both have nonnegative rewards.

The orders created by the traders are independent and their execution is tied to the primary price. Because the outstanding quotes persist with time, it might be the case that bids and asks end up briefly overlapping, as the constraint $B _ { t } \leq A _ { t }$ applies only to new pairs, ignoring previously posted quotes. However, at the end of any given round, all outstanding bids lie below the primary price and all outstanding asks lie above the primary price, thus a single change in the primary price triggers either a bid or an ask, not both. This makes the coupling of the two problems simple, allowing us to run independently one policy for the asks and another one for the bids.

Define the fixed and dynamic two-sided benchmarks by

$$
\mathrm { O P T } _ { T } ^ { \mathrm { t w o } } : = \operatorname* { m a x } _ { 0 \leq b \leq a \leq 1 } \sum _ { t = 1 } ^ { T } \bigl ( r _ { t } ^ { \mathrm { b i d } } ( b ) + r _ { t } ^ { \mathrm { a s k } } ( a ) \bigr ) \ , \quad \mathrm { O P T } _ { T } ^ { \mathrm { d y n , t w o } } : = \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { 0 \leq b \leq a \leq 1 } \bigl ( r _ { t } ^ { \mathrm { b i d } } ( b ) + r _ { t } ^ { \mathrm { a s k } } ( a ) \bigr ) \ .
$$

Subtracting the learner’s expected total reward defines $R _ { T } ^ { \mathrm { t w o } }$ and $R _ { T } ^ { \mathrm { { d y n , t w o } } }$ , respectively. Maxima exist by rounding asks up and bids down to $\{ 0 , 1 , p _ { 1 } , \dotsc , p _ { T } \}$ , preserving feasibility and execution dates. Let $p _ { t } ^ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { t \le s \le T } p _ { s }$ , and set

$$
V _ { T } ^ { \mathrm { a s k } } : = V _ { T } , V _ { T } ^ { \mathrm { b i d } } : = \sum _ { t = 1 } ^ { T - 1 } ( p _ { t } - p _ { t } ^ { \mathrm { m i n } } ) , \quad \mathrm { a n d } V _ { T } ^ { \mathrm { t w o } } : = V _ { T } ^ { \mathrm { a s k } } + V _ { T } ^ { \mathrm { b i d } } = \sum _ { t = 1 } ^ { T - 1 } ( p _ { t } ^ { \mathrm { m a x } } - p _ { t } ^ { \mathrm { m i n } } ) .
$$

Recall that in the ask-only case primary exposure vanishes on decreasing price paths. The two-sided exposure $V _ { T } ^ { \mathrm { t w o } }$ instead controls price movements in both directions.

Define the two-sided strategy for any sequence $\lambda _ { t } \in [ 0 , 1 ]$ as $B _ { 1 } = 0 , A _ { 1 } = 1$ , and, for $t \geq 2 .$

$$
B _ { t } = m _ { t - 1 } - ( 1 - \lambda _ { t } ) ( m _ { t - 1 } - p _ { t - 1 } ) _ { + } , \qquad A _ { t } = m _ { t - 1 } + ( 1 - \lambda _ { t } ) ( p _ { t - 1 } - m _ { t - 1 } ) _ { + } .\tag{13}
$$

The transformation $( p , m , b ) \mapsto ( 1 - p , 1 - m , 1 - b )$ turns a bid into an ask with the same execution time and reward. Its primary exposure is $V _ { T } ^ { \mathrm { b i d } }$ , and its secondary exposure remains $W _ { T }$ . Unlike Cesa-Bianchi et al. (2025), we do not need to swap the quotes coming from the two subroutines, because these strategies are perfectly mirrored and $B _ { t } \leq A _ { t }$ is automatically enforced.

Define the observable quantity

$$
Z _ { t } : = \sum _ { s = 1 } ^ { t - 1 } \left( \operatorname* { m a x } _ { s \leq u \leq t } p _ { u } - \operatorname* { m i n } _ { s \leq u \leq t } p _ { u } \right) \ , \qquad Z _ { 1 } : = 0 .\tag{14}
$$

The sequence of $Z _ { t }$ is nondecreasing, and $Z _ { T } = V _ { T } ^ { \mathrm { t w o } }$ . We know show how we can achieve optimal static and dynamic regret by tuning $\lambda _ { t }$ with this quantity. All the proofs are deferred to Appendix E.

## Theorem 10. For every pair of price paths, policy (13) with

$$
\lambda _ { t } = { \frac { 1 } { t } } \operatorname* { m i n } \left\{ 1 , { \frac { \sqrt { 1 + ( H _ { T } - 1 ) Z _ { t - 1 } } } { H _ { T } - 1 } } \right\} \qquad ( t \geq 2 )\tag{15}
$$

satisfies

$$
R _ { T } ^ { \mathrm { t w o } } \leq R _ { T } ^ { \mathrm { d y n , t w o } } = O \bigg ( 1 + \sqrt { V _ { T } ^ { \mathrm { t w o } } H _ { T } } + V _ { T } ^ { \mathrm { t w o } } + W _ { T } \bigg ) .
$$

If the horizon T is unknown, the choice

$$
\lambda _ { t } = { \frac { 1 } { t } } \operatorname* { m i n } \left\{ 1 , { \frac { \sqrt { 1 + H _ { t - 1 } Z _ { t - 1 } } } { H _ { t - 1 } } } \right\} \qquad ( t \geq 2 )\tag{16}
$$

satisfies the same bound with an additional $O ( \log \log ( T + 1 ) )$ term.

The proof applies the one-sided analysis twice and adds the results. Adding the two sides replaces $p _ { t } ^ { \operatorname* { m a x } } \ ( \mathrm { r e p s . } p _ { t } ^ { \operatorname* { m i n } } )$ by the remaining range $D _ { t } = p _ { t } ^ { \operatorname* { m a x } } - p _ { t } ^ { \operatorname* { m i n } }$ , whose total over the horizon is $V _ { T } ^ { \mathrm { t w o } }$ because the asks are charged the decrease of $p _ { t } ^ { \mathrm { m a x } }$ and the bids the increase of $p _ { t } ^ { \operatorname* { m i n } }$ . Each round then costs the learner an amount proportional to $\lambda _ { t }$ for the traded orders plus $( D _ { t - 1 } - D _ { t } ) / \lambda _ { t }$ for the orders that a moving threshold leaves behind, and balancing the two would require $V _ { T } ^ { \mathrm { t w o } }$ in advance. The right tuning depends on how far the price will still move, which is revealed only at the end. We tune the policy using $Z _ { t }$ , which is observable and keeps track of price movements. The adversary cannot inflict a loss using price movements without enlarging $Z _ { t } .$ , so the strategy becomes more cautious if price oscillates.

Theorem 11. For every policy, $T \geq 2 , v \in ( 0 , T - 1 ] ,$ , and $w \in [ 0 , T - 1 ]$ , there is a deterministic price path with $V _ { T } ^ { \mathrm { t w o } } \leq v$ and $W _ { T } \leq w$ such that

$$
R _ { T } ^ { \mathrm { t w o } } \geq \operatorname* { m a x } \left\{ \frac { 1 } { 4 } , \frac { 1 } { 8 } \sqrt { v ( H _ { T } - 1 ) } , \frac { v } { e } , \frac { 3 w } { 3 2 } \right\} = \Omega \Bigl ( 1 + \sqrt { v \log T } + v + w \Bigr ) \ .
$$

The same lower bound holdsfor $R _ { T } ^ { \mathrm { { d y n , t w o } } }$

Each of the four terms comes from its own construction. The constant term and the two exposure terms are inherited from the one-sided lower bounds. The remaining term runs the construction of Theorem 6, which hides the final execution threshold behind random price decreases, on a budget.

The two-sided bounds are tight because primary exposure accounts for both upward and downward price movements. In particular, the nonincreasing paths in Theorem 6 have zero ask exposure, but contribute to $V _ { T } ^ { \mathrm { b i d } }$ here. The lower bound requires $v > 0$ because its secondary-exposure construction uses a positive, arbitrarily small primary-price movement. The horizon-free tuning has an additional log log $T$ term; the matching rate is attained by the horizon-dependent, exposure-adaptive policy.

To understand why static and dynamic regret have the same rate, consider that, on the hard nonincreasing instances of Theorem 6, the dynamic benchmark beats the fixed one by $\begin{array} { r } { \sum _ { t } ( p _ { t } - p _ { T } ) } \end{array}$ and since $p _ { t } ^ { \mathrm { m i n } } \leq p _ { T }$

$$
\sum _ { t = 1 } ^ { T - 1 } ( p _ { t } - p _ { T } ) \leq \sum _ { t = 1 } ^ { T - 1 } ( p _ { t } - p _ { t } ^ { \mathrm { m i n } } ) = V _ { T } ^ { \mathrm { b i d } } \leq V _ { T } ^ { \mathrm { t w o } } .
$$

So the extra power of the dynamic comparator is bounded by a term already present in the bound. In the one-sided case that same gap was $\Theta ( { \sqrt { T } } )$ , because $V _ { T } ^ { \mathrm { a s k } }$ does not capture downward shifts.

## 8. Discussion and open problems

This work shows that market making with persistent orders is possible, even in the presence of adversarial market prices. Our analysis relies on the impossibility of canceling orders to reduce exposure to future costs. Allowing cancellations is therefore the next most natural step to make our setting more realistic. Other practical aspects that our model ignores are inventory and liquidity constraints. The extent to which violations of these constraints are allowed defines the problem as online learning with constraints (where bounds on cumulative violations are sought) or as a general state-based decision process with adversarial rewards.

## AI Disclosure

GPT-6.1 Sol assisted with drafting, reorganization, developing and checking proofs, reference verification, and compilation.

## Acknowledgments

NCB and LF acknowledge support by the EU Horizon CL4-2022-HUMAN-02 research and innovation action under grant agreement 101120237, project ELIAS (European Lighthouse of AI for Sustainability). TC gratefully acknowledges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC) through grant RGPIN-2023-03688 (Discovery Grants Program).

## References

Jacob Abernethy and Satyen Kale. Adaptive market making via online learning. In Advances in Neural Information Processing Systems (NeurIPS), 2013.

Jacob Abernethy, Peter Bartlett, Rafael Frongillo, and Andre Wibisono. How to hedge an option against an adversary: Black-Scholes pricing is minimax optimal. In Advances in Neural Information Processing Systems (NeurIPS), 2013.

Raman Arora, Ofer Dekel, and Ambuj Tewari. Online bandit learning against an adaptive adversary: from regret to policy regret. In International Conference on Machine Learning (ICML), 2012.

Marco Avellaneda and Sasha Stoikov. High-frequency trading in a limit order book. Quantitative Finance, 2008.

Nicolò Cesa-Bianchi, Ofer Dekel, and Ohad Shamir. Online learning with switching costs and other adaptive adversaries. In Advances in Neural Information Processing Systems (NeurIPS), 2013.

Nicolò Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Luigi Foscari, and Vinayak Pathak. Market making without regret. In Conference on Learning Theory (COLT), 2025.

Lawrence R. Glosten and Paul R. Milgrom. Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. Journal ofFinancial Economics, 1985.

Pooria Joulani, András György, and Csaba Szepesvári. Online learning under delayed feedback. In International Conference on Machine Learning (ICML), 2013.

Wouter M. Koolen and Vladimir Vovk. Buy low, sell high. Theoretical Computer Science, 2014.

Andrew W. Lo and A. Craig MacKinlay. An econometric analysis of nonsynchronous trading. Journal ofEconometrics, 1990.

Davide Maran and Marcello Restelli. Online market making and the value of observing the order book. In Conference on Learning Theory (COLT), 2026.

Kent Quanrud and Daniel Khashabi. Online learning with adversarial delays. In Advances in Neural Information Processing Systems (NeurIPS), 2015.

Dirk van der Hoeven and Nicolò Cesa-Bianchi. Nonstochastic bandits and experts with armdependent delays. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.

Claire Vernade, Olivier Cappé, and Vianney Perchet. Stochastic bandit models for delayed conversions. In Conference on Uncertainty in Artificial Intelligence (UAI), 2017.

Julian Zimmert and Yevgeny Seldin. An optimal algorithm for adversarial bandits with arbitrary delays. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

## Appendix A. Extension: structured price paths

In this extension we study how the performance of our strategies changes when we add assumptions on the price processes. Throughout, the policies remain unchanged and receive no additional information, so these improvements occur without any tuning and changes.

## A.1. Converging prices

We start by bounding the range of oscillations of the two prices, in particular we assume that there is a pair of prices $p _ { \infty }$ and $m _ { \infty }$ such that $| p _ { t } - p _ { \infty } | \leq a _ { t }$ and $| m _ { t } - m _ { \infty } | \leq b _ { t }$ for all t, where $( a _ { t } ) _ { t }$ and $( b _ { t } ) _ { t }$ are nonincreasing nonnegative sequences and neither the limits nor the error bounds need to be known to the maker.

Intuitively, if the prices are converging, the exposure terms can be controlled by the rate of convergence, but we still pay the baseline term due to the fact that the two markets differ from each other. It then disappears if we assume that $m _ { \infty } = p _ { \infty }$

Proposition 12. For any pair $p _ { \infty } , m _ { \infty } \in [ 0 , 1 ]$ and any pair of nonincreasing nonnegative sequences $( a _ { t } ) _ { t }$ and $( b _ { t } ) _ { \mathrm { : } }$ such that $| p _ { t } - p _ { \infty } | \leq a _ { t }$ and $| m _ { t } - m _ { \infty } | \leq b _ { t }$ for all t. Policy (6) with $\lambda _ { t } = 1 / t$ and $\lambda _ { t } = 1 / \sqrt { t - 1 }$ , satisfies respectively

$$
R _ { T } \leq H _ { T } + 2 \sum _ { t = 1 } ^ { T - 1 } a _ { t } + 4 \sum _ { t = 1 } ^ { T - 1 } b _ { t } , \qquad R _ { T } ^ { \mathrm { d y n } } \leq 1 + 3 \sqrt { T - 1 } + 2 \sum _ { t = 1 } ^ { T - 1 } a _ { t } + 4 \sum _ { t = 1 } ^ { T - 1 } b _ { t } .
$$

$I f p _ { \infty } = m _ { \infty } .$ , either policy also satisfies

$$
R _ { T } ^ { \mathrm { { d y n } } } \leq \sum _ { t = 1 } ^ { T } ( a _ { t } + b _ { t } ) + 2 \sum _ { t = 1 } ^ { T - 1 } b _ { t } .\tag{17}
$$

Proof. For every $s \geq t .$ , we have $p _ { s } - p _ { t } \le 2 a _ { t }$ and $| m _ { s } - m _ { t } | \leq 2 b _ { t }$ . Therefore,

$$
V _ { T } \leq 2 \sum _ { t = 1 } ^ { T - 1 } a _ { t } , \qquad W _ { T } \leq 2 \sum _ { t = 1 } ^ { T - 1 } b _ { t } ,
$$

and the first two claims follow from Theorems 1 and 5. When the limits coincide, an ask submitted at t and executed at $s \geq t$ earns at most $p _ { s } - m _ { s } \le a _ { t } + b _ { t }$ . This also bounds the dynamic comparator’s reward for that submission date. Both policies have nonnegative reward on the first round and quote $A _ { t } \geq m _ { t - 1 }$ thereafter. An order submitted at $t \geq 2$ consequently earns at least $- 2 b _ { t - 1 }$ , whether it executes or not. Summing proves the last claim. ■

Convergence in the primary market controls $V _ { T }$ , while convergence in the secondary market controls $W _ { T } .$ , so the two exposure bounds also apply separately. If both infinite price sequences converge, their tail errors vanish, so both regrets are $o ( T )$ , but convergence alone gives no uniform finite-horizon rate, however, since any finite hard instance can be extended with constant tails.

The baseline terms remain necessary for the unchanged policies even on constant distinct paths, so when $a _ { t } = b _ { t } = 0$ for all t and $p _ { \infty } \neq m _ { \infty }$ . Taking $p _ { t } = 1$ and $m _ { t } = 1 - g$ , with $g \in ( 0 , 1 ]$ , gives respective regrets $g ( H _ { T } - 1 )$ and $g \sqrt { T - 1 }$

When $p _ { \infty } = m _ { \infty }$ , the additional dependence on the secondary error from (17) is necessary even for arbitrary policies. Take $p _ { t } = 1 / 2$ and $m _ { t } = 1 / 2 \pm b _ { t }$ , with independent equally likely signs sampled before play, where $b _ { t } \leq 1 / 2$ decreases to zero. Any ask has conditional expected reward at most zero, while the dynamic comparator earns $b _ { t }$ when the cost is low and zero otherwise. Hence

$$
\mathbb { E } [ R _ { T } ^ { \mathrm { d y n } } ] \geq \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } b _ { t } .
$$

For each horizon, some deterministic realization attains this bound, and both price sequences converge to $1 / 2$

## A.2. Lagged secondary prices

We now consider the case in which the secondary market follows the primary market with a one-round delay (Lo and MacKinlay, 1990). In particular, suppose that $m _ { t + 1 } = p _ { t }$ for every $t < T$ , while $m _ { 1 }$ is arbitrary. This assumption does not restrict the evolution of primary prices, but relates the execution cost of an order to the primary price observed in the preceding round.

This phenomenon can arise when price discovery occurs primarily in one market, while the other adjusts more slowly, for example due to differences in liquidity, trading frequency or information transmission. This creates temporary price discrepancies that the maker can exploit when outstanding orders execute, even though the two markets track the same underlying asset.

Under this assumption, an ask a executing at round $s \geq 2$ earns

$$
a - m _ { s } = a - p _ { s - 1 } \leq p _ { s } - p _ { s - 1 } .
$$

Thus, positive profits can only arise from upward price movements. To capture this, define

$$
\delta _ { 1 } = ( p _ { 1 } - m _ { 1 } ) _ { + } , \qquad \delta _ { t } = ( p _ { t } - p _ { t - 1 } ) _ { + } \quad ( t \geq 2 ) ,
$$

and let

$$
J _ { T } : = \sum _ { t = 1 } ^ { T } \operatorname* { m a x } _ { t \leq s \leq T } \delta _ { s } .
$$

The quantity $J _ { T }$ measures the cumulative opportunity created by future upward price jumps. As we show next, it also controls the regret of our strategies, without requiring any change to the policy.

Proposition 13. Suppose that $m _ { t + 1 } = p _ { t }$ for all $t < T$ . For any choice of $\lambda _ { t } \in [ 0 , 1 ]$ , policy (6) satisfies

$$
R _ { T } \leq R _ { T } ^ { \mathrm { d y n } } \leq J _ { T } + \sum _ { t = 2 } ^ { T } \lambda _ { t } \delta _ { t - 1 } \leq 2 J _ { T } .
$$

Proof. Fix a submission date t and an ask a that executes at $s = \tau _ { t } ( a )$ . Since $a \leq p _ { s }$ , the lag assumption gives

$$
r _ { t } ( a ) = a - m _ { s } \leq \delta _ { s } \leq \operatorname* { m a x } _ { t \leq k \leq T } \delta _ { k } .
$$

The same bound holds if the ask never executes, since its reward is zero. Summing over submission dates yields

$$
\mathrm { O P T } _ { T } ^ { \mathrm { d y n } } \leq J _ { T } .
$$

We now lower bound the reward of policy $( 6 )$ . The first ask $A _ { 1 } = 1$ has nonnegative reward. For $t \geq 2 .$ , suppose first that $A _ { t }$ executes immediately. Since $m _ { t } = p _ { t - 1 }$ , the policy definition gives

$$
r _ { t } ( A _ { t } ) = A _ { t } - p _ { t - 1 } \geq - \lambda _ { t } ( p _ { t - 1 } - m _ { t - 1 } ) _ { + } = - \lambda _ { t } \delta _ { t - 1 } .
$$

If instead the ask executes at some $s > t ,$ it must have remained unexecuted at round $s - 1$ , so $A _ { t } > p _ { s - 1 } = m _ { s }$ . Consequently, its reward is positive. An ask that never executes has zero reward. Therefore, in every case,

$$
r _ { t } ( A _ { t } ) \geq - \lambda _ { t } \delta _ { t - 1 } .
$$

Summing over rounds, we obtain

$$
\sum _ { t = 1 } ^ { T } r _ { t } ( A _ { t } ) \geq - \sum _ { t = 2 } ^ { T } \lambda _ { t } \delta _ { t - 1 } .
$$

Combining this with the bound on the dynamic comparator proves the first inequality. Finally, since $\lambda _ { t } \leq 1$

$$
\sum _ { t = 2 } ^ { T } \lambda _ { t } \delta _ { t - 1 } \leq \sum _ { t = 1 } ^ { T - 1 } \delta _ { t } \leq J _ { T } ,
$$

which concludes the proof.

The new bound can be substantially smaller than the general exposure bound. For example, consider the steadily increasing price paths

$$
p _ { t } = \frac { t } { T } , \qquad m _ { t } = \frac { t - 1 } { T } .
$$

These satisfy the lag assumption and give

$$
J _ { T } = 1 , \qquad V _ { T } = W _ { T } = \frac { T - 1 } { 2 } .
$$

Our general guarantees therefore scale linearly with $T ,$ , whereas the proposition gives constant regret for both comparators. More generally, if $\delta _ { t } \leq c / t$ for every t, then $J _ { T } \le c H _ { T }$ , and both regrets are ${ \cal O } ( \log T )$

The lag assumption alone, however, does not guarantee sublinear regret. Indeed, the primaryexposure lower-bound construction of Section 6 already satisfies $m _ { t + 1 } = p _ { t } ;$ primary prices remain zero until the final round, while secondary prices are identically zero. In that construction, a single upward price jump creates $J _ { T } = \Theta ( T )$ , and every policy can incur linear regret. Hence, controlling the magnitude and timing of upward price movements is still necessary.

## Appendix B. Proof of the prefix comparison

Proof. of Lemma 4 Recall that $p _ { t } ^ { \operatorname* { m a x } } \ = \ \operatorname* { m a x } _ { t \pm s \leq T } p _ { s }$ is the highest remaining primary price, $\begin{array} { r } { C _ { t } = 1 + \sum _ { s = 1 } ^ { t - 1 } m _ { s } } \end{array}$ , and $\textstyle H _ { t } = \sum _ { s = 1 } ^ { t } 1 / s$ . The quantity $Q _ { t } ^ { \circ }$ sums the eventual surrogate rewards of orders submitted on the first t rounds; those orders need not have executed by round t. The expression $t p _ { t } ^ { \operatorname* { m a x } } - C _ { t }$ bounds the surrogate profit of a fixed quote whose orders execute for precisely the first t submission dates, as shown in (8).

We prove the first inequality by induction. $\mathbf { A } \mathbf { t } \ t = 1$ , the auxiliary quote is 1, its surrogate reward is zero, and $C _ { 1 } = H _ { 1 } = 1$ . Thus the required inequality is $0 \geq p _ { 1 } ^ { \mathrm { m a x } } - 1$ , which follows from $p _ { 1 } ^ { \mathrm { m a x } } \leq 1$

Fix $t \geq 2$ and assume the inequality holds at $t - 1$ . We distinguish three cases according to the new order’s surrogate reward.

• A positive-margin order executes. Suppose $p _ { t - 1 } ^ { \operatorname* { m a x } } > m _ { t - 1 }$ and $A _ { t } ^ { \circ } \le p _ { t } ^ { \operatorname* { m a x } }$ . Since $\lambda _ { t } = 1 / t$

$$
r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } ) = ( 1 - \lambda _ { t } ) ( p _ { t - 1 } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) = \frac { t - 1 } { t } ( p _ { t - 1 } ^ { \operatorname* { m a x } } - m _ { t - 1 } ) .
$$

Add this reward to the induction hypothesis and use $C _ { t } = C _ { t - 1 } + m _ { t - 1 }$ and $H _ { t } = H _ { t - 1 } + 1 / t .$ Subtracting the desired lower bound gives

$$
Q _ { t } ^ { \circ } - ( t p _ { t } ^ { \mathrm { m a x } } - C _ { t } - H _ { t } + 1 ) \geq \left( t - { \frac { 1 } { t } } \right) ( p _ { t - 1 } ^ { \mathrm { m a x } } - p _ { t } ^ { \mathrm { m a x } } ) + { \frac { 1 + m _ { t - 1 } - p _ { t } ^ { \mathrm { m a x } } } { t } } \geq 0 .
$$

Both terms are nonnegative: the remaining primary maximum cannot increase, and all prices lie in [0, 1].

• The reference margin is nonpositive. If $p _ { t - 1 } ^ { \operatorname* { m a x } } \leq m _ { t - 1 }$ , the quote is $m _ { t - 1 }$ and its surrogate reward is zero. Also,

$$
t p _ { t } ^ { \operatorname* { m a x } } \leq t p _ { t - 1 } ^ { \operatorname* { m a x } } \leq ( t - 1 ) p _ { t - 1 } ^ { \operatorname* { m a x } } + m _ { t - 1 } .
$$

• A positive-margin order does not execute. I $\mathrm { : } p _ { t - 1 } ^ { \operatorname* { m a x } } > m _ { t - 1 }$ and $A _ { t } ^ { \circ } > p _ { t } ^ { \operatorname* { m a x } }$ , its reward is again zero. The policy definition gives

$$
t p _ { t } ^ { \operatorname* { m a x } } < t A _ { t } ^ { \circ } = ( t - 1 ) p _ { t - 1 } ^ { \operatorname* { m a x } } + m _ { t - 1 } .
$$

In each zero-reward case, subtracting $C _ { t } = C _ { t - 1 } + m _ { t - 1 }$ shows that

$$
t p _ { t } ^ { \operatorname* { m a x } } - C _ { t } \leq ( t - 1 ) p _ { t - 1 } ^ { \operatorname* { m a x } } - C _ { t - 1 } .
$$

Thus the quantity we need to match has not increased, whereas $Q _ { t } ^ { \circ } = Q _ { t - 1 } ^ { \circ }$ and $H _ { t } \geq H _ { t - 1 }$ . The induction hypothesis therefore gives the required inequality at t.

To compare with the best fixed quote, we must cover every possible prefix length. All auxiliary surrogate rewards are nonnegative, so, for every $k \leq T$

$$
Q _ { T } ^ { \circ } \geq Q _ { k } ^ { \circ } \geq k p _ { k } ^ { \operatorname* { m a x } } - C _ { k } - H _ { k } + 1 \geq k p _ { k } ^ { \operatorname* { m a x } } - C _ { k } - H _ { T } + 1 .
$$

Together with $Q _ { T } ^ { \circ } \geq 0$ , this implies

$$
\operatorname* { m a x } \left\{ 0 , \operatorname* { m a x } _ { 1 \leq k \leq T } ( k p _ { k } ^ { \operatorname* { m a x } } - C _ { k } ) \right\} \leq Q _ { T } ^ { \circ } + H _ { T } - 1 .
$$

The benchmark bound (8) now yields $R _ { T } ^ { \prime } ( A ^ { \circ } ) = \mathrm { O P T } _ { T } ^ { \prime } - Q _ { T } ^ { \circ } \leq H _ { T } - 1$

## Appendix C. The static-regret tuning can incur linear dynamic regret

The static and dynamic regret bounds use the same policy (6), with different choices of $\lambda _ { t }$ . We show that the static choice $\lambda _ { t } = 1 / t$ can have dynamic regret at least $T / 1 2 .$ , even when $V _ { T } = W _ { T } = 0$ The construction keeps prices constant for the first half of the horizon, then makes them fall slightly more than the amount subtracted by the static policy from its quote.

Let $T = 2 n$ with $n \geq 2 ,$ , set $m _ { t } = 0$ on every round, and choose

$$
p _ { t } = 1 \quad ( t \leq n ) , \qquad p _ { t } = \left( 1 - { \frac { 2 } { t } } \right) p _ { t - 1 } \quad ( t > n ) .
$$

The primary prices are positive and nonincreasing, so $V _ { T } = W _ { T } = 0$ . For every $t > n$ , the static choice gives

$$
A _ { t } = \left( 1 - \frac { 1 } { t } \right) p _ { t - 1 } > \left( 1 - \frac { 2 } { t } \right) p _ { t - 1 } = p _ { t } .
$$

This order does not execute on its submission round. Since primary prices never increase, it cannot execute later either. However, the available sale price remains bounded away from zero:

$$
p _ { t } = \prod _ { s = n + 1 } ^ { t } { \frac { s - 2 } { s } } = { \frac { n ( n - 1 ) } { t ( t - 1 ) } } \geq { \frac { 1 } { 6 } } \qquad ( n < t \leq 2 n ) .
$$

The dynamic comparator quotes $p _ { t }$ and earns $p _ { t }$ immediately on each of these n rounds. Regret on every other round is nonnegative, so $R _ { T } ^ { \mathrm { d y n } } \geq n / 6 = T / 1 2$

## Appendix D. Details for the lower bounds

## D.1. The last primary price need not be the best fixed quote

The static-regret part of Theorem 6 uses the reward of the fixed quote $p _ { T }$ as a lower bound on $\mathrm { O P T } _ { T }$ That quote need not be optimal. To see this, consider any nonincreasing primary path and a constant secondary price $c < p _ { T }$ . An order either executes immediately or never executes. Hence a fixed quote a trades on a prefix of submission dates. If that prefix has length $k \geq 1$ , its total reward is

$$
k ( a - c ) \leq k ( p _ { k } - c ) .
$$

Conversely, quoting $p _ { k }$ trades on at least k rounds, each with positive margin. A quote that never trades earns zero, so

$$
\mathrm { O P T } _ { T } = \operatorname* { m a x } _ { 1 \leq k \leq T } k ( p _ { k } - c ) .
$$

For example, with $p = ( 1 , 1 / 4 )$ and $m = ( 0 , 0 )$ , quote 1 earns 1, while quote $p _ { T } = 1 / 4$ earns only $1 / 2$

Averaging over price sequences. In the next two proofs, we fix an arbitrary policy and choose the entire price sequence randomly, independently of the policy’s internal randomness. We bound the average of the policy’s expected regret over these sequences. At least one fixed sequence must attain that average, which proves the desired deterministic lower bound.

## D.2. Primary exposure

Proof. of Theorem 7 The construction hides the only profitable primary price until the final round, after every quote has been chosen. Let $\beta = v / ( T - 1 ) \in ( 0 , 1 ]$ , set all secondary prices to zero, and take $p _ { t } = 0$ for $t < T$ and $p _ { T } = X$ . Choose $X \in [ \beta / e , \beta ]$ so that

$$
\mathbb { P } ( X \ge a ) = \left\{ \begin{array} { l l } { 1 , } & { 0 \le a \le \beta / e , } \\ { \beta / ( e a ) , } & { \beta / e < a \le \beta , } \\ { 0 , } & { a > \beta . } \end{array} \right.
$$

This distribution assigns probability $1 / e$ to $X = \beta ,$ , so the displayed probability includes execution at the endpoint. Every positive ask a executes only at $T ,$ , and does so exactly when $X \geq a ;$ a zero ask earns zero. The distribution is chosen to make the expected reward of every ask at most $\beta / e \colon$

$$
a \mathbb { P } ( X \geq a ) \leq \beta / e .
$$

The prices observed before the last round provide no information about $X .$ . Thus this bound also applies to each quote chosen by a randomized policy, including its final quote. The learner’s expected total reward is at most $T \beta / e$

The fixed comparator knows X and quotes it on every round, earning $_ { T X }$ . Moreover,

$$
\mathbb { E } [ X ] = \int _ { 0 } ^ { \beta } \mathbb { P } ( X \geq x ) d x = { \frac { \beta } { e } } + \int _ { \beta / e } ^ { \beta } { \frac { \beta } { e x } } d x = { \frac { 2 \beta } { e } } .
$$

The average regret is therefore at least $T \beta / e = T v / ( e ( T - 1 ) ) \geq v / e$ . Every realization satisfies $V _ { T } = ( T - 1 ) X \leq v$ and $W _ { T } = 0$ . Selecting a value of X with at least the average expected regret proves the claim.

## D.3. Secondary exposure

Proof. of Theorem 8 If $w = 0$ , take $p _ { t } = m _ { t } = 0$ on every round. Suppose w $> 0 .$ , and choose

$$
\delta = \frac { w } { T - 1 } , \qquad 0 < h \leq \frac { \delta } { 4 T } .
$$

Before play begins, choose one of two scenarios, each with probability $1 / 2$ . In both scenarios, the primary price is $3 / 4$ and the secondary price is $1 / 2$ on every round before T. On the last round, the primary price increases by $h .$ The only difference between the scenarios is the final secondary price: it falls by $\delta$ in one and rises by $\delta$ in the other. Each row of the table below specifies both market prices for one scenario.

<table><tr><td></td><td colspan="2">Rounds  $t < T$ </td><td colspan="2">Round T</td></tr><tr><td>Scenario</td><td>Primary Secondary</td><td></td><td></td><td>Primary Secondary</td></tr><tr><td>Secondary price falls Secondary price rises</td><td>3/4 3/4</td><td>1/2 1/2</td><td>3/4+ h 1/2 +δ</td><td> $\begin{array} { r l } { 3 / 4 + h } & { { } \ 1 / 2 - \delta } \end{array}$ </td></tr></table>

All prices lie in [0, 1], since $\delta \leq 1 / 4$ and $h \leq \delta / ( 4 T )$ . The maker observes the same prices in both scenarios before round T. Its final quote is also chosen before observing that round’s prices. Thus, when submitting any of its asks, the maker has no information about which scenario was chosen.

For $t < T$ , an ask $a \leq 3 / 4$ executes immediately and earns $a - 1 / 2$ . An ask $3 / 4 < a \le 3 / 4 + h$ waits until round $T ,$ , when the primary price first reaches it; such asks exist because $h > 0$ . Its hedging cost is then $1 / 2 - \delta \mathrm { o r } 1 / 2 + \delta ,$ , each with probability $1 / 2 .$ , so its expected reward is again $a - 1 / 2$ . Asks above $3 / 4 + h$ never execute. An ask submitted at $T$ also has expected reward $a - 1 / 2$ if $a \leq 3 / 4 + h$ , and zero otherwise. Therefore, each quote has expected reward at most $1 / 4 + h ,$ , and the learner’s expected total reward is at most $T ( 1 / 4 + h )$ .

When costs fall, the fixed quote $3 / 4 + h$ delays every order until $T$ and earns $T ( 1 / 4 + h + \delta )$ When costs rise, the fixed quote $3 / 4$ trades immediately on every round and earns $T / 4 - \delta ;$ only its last order pays the higher cost. These are valid choices for the fixed comparator in the respective sequences. Averaging their rewards and subtracting the learner’s upper bound gives regret at least

$$
\frac { T - 1 } { 2 } \delta - \frac { T } { 2 } h \geq \frac { w } { 2 } - \frac { \delta } { 8 } \geq \frac { 3 } { 8 } w ,
$$

since $\delta = w / ( T - 1 ) \leq w .$

In both scenarios, $w _ { t } = \delta$ for every $t < T$ , so $W _ { T } = ( T - 1 ) \delta = w$ , and the primary exposure is $V _ { T } = ( T - 1 ) h$ . At least one of the two deterministic price paths has regret at least the average and therefore satisfies

$$
W _ { T } = w , \qquad V _ { T } = ( T - 1 ) h , \qquad R _ { T } \geq \frac { 3 } { 8 } w .
$$

Taking $h = \delta / ( 4 T )$ gives $V _ { T } = w / ( 4 T )$ ), proving the first claim. By choosing $h > 0$ arbitrarily small, we obtain deterministic price paths with arbitrarily small $V _ { T }$ while retaining $W _ { T } = w$ and $R _ { T } \ge 3 w / 8$ , proving the final claim.

## Appendix E. Details of the simultaneous bids and asks proofs

Proof. of Theorem 10 The proof follows Theorem 5. Write $D _ { t } : = p _ { t } ^ { \operatorname* { m a x } } - p _ { t } ^ { \operatorname* { m i n } }$ , so that $D _ { T } = 0$ and $\begin{array} { r } { \sum _ { t = 1 } ^ { T - 1 } D _ { t } = V _ { T } ^ { \mathrm { t w o } } } \end{array}$ . The learner’s reward splits over the two sides by definition, so the proof of Lemma 2 contributes $1 + 2 W _ { T }$ on each and Lemma 3 contributes $V _ { T } ^ { \mathrm { a s k } }$ and $V _ { T } ^ { \mathrm { b i d } }$ respectively. Bounding what remains by (12) on each side and adding the two gives, for any $\lambda _ { t } \in ( 0 , 1 ]$

$$
\begin{array} { r l } & { R _ { T } ^ { \mathrm { d y n , t w o } } \leq 2 + 4 W _ { T } + V _ { T } ^ { \mathrm { a s k } } + V _ { T } ^ { \mathrm { b i d } } + \displaystyle \sum _ { t = 2 } ^ { T } \left( 2 \lambda _ { t } + \frac { ( p _ { t - 1 } ^ { \mathrm { m a x } } - p _ { t } ^ { \mathrm { m a x } } ) + ( p _ { t } ^ { \mathrm { m i n } } - p _ { t - 1 } ^ { \mathrm { m i n } } ) } { \lambda _ { t } } \right) } \\ & { \qquad \leq 2 + V _ { T } ^ { \mathrm { t w o } } + 4 W _ { T } + 2 \displaystyle \sum _ { t = 2 } ^ { T } \lambda _ { t } + \displaystyle \sum _ { t = 2 } ^ { T } \operatorname* { m i n } \left\{ 2 , \frac { D _ { t - 1 } - D _ { t } } { \lambda _ { t } } \right\} . } \end{array}\tag{18}
$$

The last term is truncated by 2 because the bid and ask side each contributes to the per-round surrogate regret max $\phantom { } _ { \ i } r _ { t } ^ { \prime } ( a ) - r _ { t } ^ { \prime } ( A _ { t } ^ { \circ } )$ of the clairvoyant policy, which is at most 1 because the optimal surrogate reward is at most 1 while the clairvoyant quote never crosses $m _ { t - 1 }$ and therefore has nonnegative surrogate reward. Summing the two sides bounds the contribution of round t by min $\{ 2 , 2 \lambda _ { t } + ( D _ { t - 1 } - D _ { t } ) / \lambda _ { t } \}$ , and since $\lambda _ { t } \geq 0$ the truncation at 2 may be carried by the second summand alone.

Suppose first that $V _ { T } ^ { \mathrm { t w o } } > 0$ is known. Take $\lambda _ { t } = \alpha / t$ , where $\alpha = \operatorname* { m i n } \{ 1 , \sqrt { V _ { T } ^ { \mathrm { t w o } } / ( H _ { T } - 1 ) } \}$ Since

$$
\sum _ { t = 2 } ^ { T } t ( D _ { t - 1 } - D _ { t } ) = D _ { 1 } + \sum _ { t = 1 } ^ { T - 1 } D _ { t } \leq 2 V _ { T } ^ { \mathrm { t w o } } ,\tag{19}
$$

(18) yields

$$
R _ { T } ^ { \mathrm { { d y n , t w o } } } \leq 2 + 2 \alpha ( H _ { T } - 1 ) + \frac { 2 V _ { T } ^ { \mathrm { { t w o } } } } { \alpha } + V _ { T } ^ { \mathrm { { t w o } } } + 4 W _ { T } ,
$$

which gives the claimed rate. If $V _ { T } ^ { \mathrm { t w o } } = 0$ , all primary prices are constant and taking $\lambda _ { t } = 0$ gives the corresponding constant baseline bound.

We now remove knowledge of $V _ { T } ^ { \mathrm { t w o } }$ . The relevant property of $Z _ { t }$ is that, for every $2 \leq s \leq T$

$$
\sum _ { t = 2 } ^ { s } t ( D _ { t - 1 } - D _ { t } ) = ( D _ { 1 } - D _ { s } ) + \sum _ { j = 1 } ^ { s - 1 } ( D _ { j } - D _ { s } ) \leq 2 Z _ { s } ,\tag{20}
$$

which mirrors (19). To see that it holds, note that, for $1 \leq j < s \leq T$

$$
D _ { j } - D _ { s } = ( p _ { j } ^ { \mathrm { m a x } } - p _ { s } ^ { \mathrm { m a x } } ) - ( p _ { j } ^ { \mathrm { m i n } } - p _ { s } ^ { \mathrm { m i n } } ) \leq \operatorname* { m a x } _ { j \leq u \leq s } p _ { u } - \operatorname* { m i n } _ { j \leq u \leq s } p _ { u } .
$$

For the tuning (15), let $\lambda _ { t } = \alpha _ { t } / t$ . Since $Z _ { t }$ is nondecreasing, $\alpha _ { t }$ is also nondecreasing, and

$$
\sum _ { t = 2 } ^ { T } \lambda _ { t } \leq ( H _ { T } - 1 ) \alpha _ { T } \leq \sqrt { 1 + ( H _ { T } - 1 ) V _ { T } ^ { \mathrm { t w o } } } .
$$

To bound the last sum in (18), group rounds by $2 ^ { k } \leq 1 + ( H _ { T } - 1 ) Z _ { t - 1 } < 2 ^ { k + 1 }$ . Within group $k ,$ $\alpha _ { t } \geq$ min $\{ 1 , 2 ^ { k / 2 } / ( H _ { T } - 1 ) \}$ }. If b is the last round in group k, (20) bounds the contribution of all earlier rounds by

$$
\frac { \sum _ { t = 2 } ^ { b - 1 } t ( D _ { t - 1 } - D _ { t } ) } { \operatorname* { m i n } \{ 1 , 2 ^ { k / 2 } / ( H _ { T } - 1 ) \} } \leq 4 \operatorname* { m a x } \left\{ 2 ^ { k / 2 } , \frac { 2 ^ { k } } { H _ { T } - 1 } \right\} \ .
$$

The final round contributes at most 2 by truncation. Summing over the geometric groups gives

$$
\sum _ { t = 2 } ^ { T } \operatorname* { m i n } \bigg \{ 2 , \frac { D _ { t - 1 } - D _ { t } } { \lambda _ { t } } \bigg \} = O \bigg ( 1 + \sqrt { ( H _ { T } - 1 ) V _ { T } ^ { \mathrm { t w o } } } + V _ { T } ^ { \mathrm { t w o } } \bigg ) \ .
$$

Substitution into (18) proves the horizon-dependent guarantee.

The previous tuning, while being implementable, requires knowledge of the time horizon.

The tuning (16) differs from (15) only in replacing the constant $H _ { T } - 1$ by $H _ { t - 1 }$ , we show that this costs at most a factor of two in $\alpha _ { t }$ . Dropping either term under the square root gives $\sqrt { 1 + H _ { t - 1 } Z _ { t - 1 } } \geq \operatorname* { m a x } \{ 1 , \sqrt { H _ { t - 1 } Z _ { t - 1 } } \}$ , and dividing by $H _ { t - 1 } \leq H _ { T }$ gives

$$
\frac { \sqrt { 1 + H _ { t - 1 } Z _ { t - 1 } } } { H _ { t - 1 } } \geq \operatorname* { m a x } \left\{ \frac { 1 } { H _ { T } } , \sqrt { \frac { Z _ { t - 1 } } { H _ { T } } } \right\} \geq \frac { 1 } { 2 } \cdot \frac { \sqrt { 1 + H _ { T } Z _ { t - 1 } } } { H _ { T } } ,
$$

where the last step uses $\sqrt { 1 + H _ { T } Z _ { t - 1 } } \leq 1 + \sqrt { H _ { T } Z _ { t - 1 } }$ . Hence $\begin{array} { r } { \alpha _ { t } \geq \frac { 1 } { 2 } \operatorname* { m i n } \lbrace 1 , \sqrt { 1 + H _ { T } Z _ { t - 1 } } / H _ { T } \rbrace } \end{array}$

Grouping the rounds by $2 ^ { k } \leq 1 + H _ { T } Z _ { t - 1 } < 2 ^ { k + 1 }$ now repeats the argument above with $H _ { T }$ in place of $H _ { T } - 1$ , the factor $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ only doubling the constant obtained for each group, so the last sum in (18) is again $O ( 1 + \sqrt { H _ { T } V _ { T } ^ { \mathrm { t w o } } } + V _ { T } ^ { \mathrm { t w o } } )$

What remains is $\textstyle \sum _ { t = 2 } ^ { T } \lambda _ { t }$ , which is where the extra term appears. Using $Z _ { t - 1 } \leq V _ { T } ^ { \mathrm { t w o } }$ and the same square-root inequality,

$$
\lambda _ { t } \leq \frac { 1 + \sqrt { H _ { t - 1 } V _ { T } ^ { \mathrm { t w o } } } } { t H _ { t - 1 } } = \frac { 1 } { t H _ { t - 1 } } + \frac { \sqrt { V _ { T } ^ { \mathrm { t w o } } } } { t \sqrt { H _ { t - 1 } } } .
$$

Since $H _ { t - 1 } \geq \log t .$ , comparing the two sums with the integrals of $1 / ( x \ln x )$ and $1 / ( x { \sqrt { \ln x } } )$ gives

$$
\sum _ { t = 2 } ^ { T } \lambda _ { t } = O \biggl ( \log H _ { T } + \sqrt { H _ { T } V _ { T } ^ { \mathrm { t w o } } } \biggr ) \ .
$$

The second term already appears in the horizon-dependent bound. This concludes the proof.

Proof. of Theorem 11 As in Section 6, we construct random oblivious price paths and select a deterministic realization with at least the average regret. For the constant term, take $m _ { t } = 0$ and $p _ { t } = X$ on every round, where $X$ is equally likely to be $1 / 2$ or 1. The fixed comparator earns $_ { T X }$ while the learner’s expected reward on the first round is at most $1 / 2 .$ , giving regret at least $1 / 4$ with both exposures zero. Theorems 7 and 8 give the $\Omega ( v )$ and $\Omega ( w )$ terms. Their constructions satisfy $p _ { t } \geq m _ { t }$ , so bids cannot earn positive reward. For the secondary construction, its positive primary increment can be chosen arbitrarily small to respect any $v > 0 ;$ ; for $w > ( T - 1 ) / 4 _ { \cdot }$ , apply Theorem 8 with budget $( T - 1 ) / 4 = \Omega ( w )$

It remains to obtain $\Omega ( \sqrt { v ( H _ { T } - 1 ) } )$ , and we may assume $1 / ( H _ { T } - 1 ) \le v \le H _ { T } - 1$ , since outside this range the constant term or the term $v / e$ is already of the required order. The construction of Theorem 6 hides the final execution threshold behind random price decreases, but here each decrease also costs primary exposure, so its size must be tuned against the budget. If the price falls by a factor $\delta$ with probability δ, the regret of that round is proportional to $\delta ,$ because the maker must quote below the current price to avoid missing execution, while its expected exposure is proportional to $\delta ^ { 2 }$ , because a decrease costs only when it occurs and its size is again $\delta .$ Regret is therefore linear and exposure quadratic in the same parameter, so shrinking it buys budget quadratically while costing regret only linearly. Taking $\delta$ proportional to $1 / t$ makes the two sums $\eta ( H _ { T } - 1 )$ and $\eta ^ { 2 } ( H _ { T } - 1 )$ ), so the budget v fixes η at the order of $\sqrt { v / ( H _ { T } - 1 ) }$ and leaves regret of order $\sqrt { v ( H _ { T } - 1 ) }$

Following Theorem 6, set $p _ { 1 } = 1 , m _ { t } = 0$ and $\begin{array} { r } { \eta : = \frac { 1 } { 4 } \sqrt { v / ( H _ { T } - 1 ) } } \end{array}$ , so that $v \leq H _ { T } - 1$ gives $\eta \leq 1 / 4$ . At each $t \geq 2 .$ , let $p _ { t } = ( 1 - \eta / t ) p _ { t - 1 }$ with probability $\eta / t$ and $p _ { t } = p _ { t - 1 }$ otherwise, independently of the previous draws, until $\lfloor v / \eta \rfloor$ decreases have occurred, after which the price stays constant. This cap is what makes the budget hold on every realization rather than only in expectation, and it is loose enough to bind rarely. A decrease at round t raises the exposure by $( t - 1 ) ( \eta / t ) p _ { t - 1 } \leq \eta$ , so

$$
V _ { T } ^ { \mathrm { t w o } } = V _ { T } ^ { \mathrm { b i d } } = \sum _ { t = 2 } ^ { T } ( t - 1 ) ( p _ { t - 1 } - p _ { t } ) \leq \left\lfloor \frac { v } { \eta } \right\rfloor \eta \leq v , \qquad W _ { T } = 0 \ . \qquad W _ { T } = 0 \ .
$$

Every path is nonincreasing, so $V _ { T } ^ { \mathrm { a s k } } = 0$ , and a bid earns −b when it executes, so the best bid is 0 and the two-sided benchmark coincides with the ask-only one.

Let $I _ { t }$ indicate that the cap has not yet bound at round t. Conditioning on the first $t - 1$ rounds, the argument of Theorem 6 bounds the expected reward of any ask by $( 1 - I _ { t } \eta / t ) p _ { t - 1 }$ , while the price itself satisfies $\mathbb { E } [ p _ { t } | p _ { t - 1 } ] = ( 1 - I _ { t } \eta ^ { 2 } / t ^ { 2 } ) p _ { t - 1 }$ , because a decrease of relative size $\eta / t$ occurs with probability $\eta / t$ . The fixed ask $p _ { T }$ earns $T p _ { T }$ , so the telescoping calculation of that theorem gives

$$
\mathbb { E } [ R _ { T } ^ { \mathrm { t w o } } ] \geq \sum _ { t = 2 } ^ { T } \mathbb { E } \bigg [ I _ { t } p _ { t - 1 } \left( \frac { \eta } { t } - t \frac { \eta ^ { 2 } } { t ^ { 2 } } \right) \bigg ] = \eta ( 1 - \eta ) \sum _ { t = 2 } ^ { T } \frac { \mathbb { E } \left[ I _ { t } p _ { t - 1 } \right] } { t } .
$$

This is where the harmonic sum appears, and it remains to bound $\mathbb { E } [ I _ { t } p _ { t - 1 } ]$ below by a constant independent of t.

Two events can make $\mathbb { E } [ I _ { t } p _ { t - 1 } ]$ small, the cap having already bound and the price having already fallen. We rule out both uniformly in t. The expected number of decreases is at most $\begin{array} { r } { \sum _ { t = 2 } ^ { T } \dot { \eta } / t = \eta ( H _ { T } - 1 ) } \end{array}$ , and $v / \eta = 4 \sqrt { v ( H _ { T } - 1 ) } \geq 4$ by the assumption $v \geq 1 / ( H _ { T } - 1 )$ , so $\lfloor v / \eta \rfloor \ge v / ( 2 \eta )$ and Markov’s inequality gives $\mathbb { P } ( I _ { t } = 0 ) \le 2 \eta ^ { 2 } ( H _ { T } - 1 ) / v = 1 / 8$ . Iterating $\mathbb { E } [ p _ { t } ] \geq ( 1 - \eta ^ { 2 } / t ^ { 2 } ) \mathbb { E } [ p _ { t - 1 } ]$ gives $\textstyle \mathbb { E } [ p _ { t - 1 } ] \geq 1 - \eta ^ { 2 } \sum _ { s > 2 } s ^ { - 2 } \geq 1 5 / 1 6$ . Since $p _ { t - 1 } \leq 1$ , the two estimates combine into $\mathbb { E } [ I _ { t } p _ { t - 1 } ] \geq 1 3 / 1 6 .$ , and $\eta \leq 1 / 4 \ \mathrm { { \bar { g i v e s } } } \ 1 - \eta \geq 3 / 4$ , so

$$
\mathbb { E } [ R _ { T } ^ { \mathrm { t w o } } ] \geq \frac { 1 } { 4 } \sqrt { \frac { v } { H _ { T } - 1 } } \cdot \frac { 3 } { 4 } \cdot \frac { 1 3 } { 1 6 } ( H _ { T } - 1 ) = \frac { 3 9 } { 2 5 6 } \sqrt { v ( H _ { T } - 1 ) } \geq \frac { 1 } { 8 } \sqrt { v ( H _ { T } - 1 ) } .
$$

Every realization respects both budgets, so some deterministic price path attains this bound. Taking the maximum of the four constructions proves the claim, since the maximum is within a constant factor of their sum, and $R _ { T } ^ { \mathrm { d y n , t w o } } \geq R _ { T } ^ { \mathrm { t w o } }$ extends it to the dynamic comparator.