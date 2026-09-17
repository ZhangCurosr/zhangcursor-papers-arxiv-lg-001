# A CONVERGENCE FRAMEWORK FOR DEEP � -LEARNING: ERROR PROPAGATION AND SHARP ACTION-GAP BOUNDS

Yury Kolomeytsev Faculty of Computational Mathematics and Cybernetics Lomonosov Moscow State University yury.kolomeytsev@gmail.com

## ABSTRACT

We establish convergence bounds for deep �-learning with horizon �. The algorithm fits a scalar value function to targets from executed transitions and selects actions using a predictive model and the learned value function. For current observed-successor targets with fresh true-kernel outcomes, the conditional mean is $\mathcal { T } ^ { \beta } V$ , which averages over the behavior policy’s actions. The Bellman optimality update is � � . We decompose the update error into six residuals: fitting, transition reuse, target construction, replay, action selection, and exploration. Under $L ^ { s }$ concentrability, their $L ^ { p }$ norms, with $p = s / ( s - 1 )$ , control expected �<sup>1</sup> policy loss. The bound has explicit weights on residuals from only the most recent � − 1 update blocks, plus an initialization term for shorter runs. We quantify the cost of a shared sampling distribution across horizon levels. For statistical error bounds proportional to �<sup>−�</sup>, we derive optimal continuous allocations and an integer allocation whose statistical objective is within a factor 2<sup>�</sup> of the constrained optimum. A margin condition with exponent � gives action error of order Λ<sup>1+�/�</sup>, where Λ combines network drift and score error; a matching one-step construction proves the exponent sharp. Bounds on the distance between frozen and optimal scores transfer an optimal-gap condition to frozen-iterate gap bounds while retaining the mass of optimal ties. Survival probabilities and coverage conditions at deployment yield bounds for policies selected with approximate scores. Separate spatial ReLU networks for each horizon level give a conditional neural regression rate, and the finite-state specialization gives a log-free expected fit rate. These results establish expected policy-loss consistency for the fixed-horizon generative-reset approximate-ERM procedure with exact action scores and provide an explicit residual-decay criterion for FIFO/interleaved SGD.

Keywords reinforcement learning · deep learning · deep � -learning · Markov decision processes · convergence analysis · error propagation · Bellman residuals · concentrability coefficients · action-gap regularity

## 1 Introduction

Deep �-learning fits a scalar state-value function to targets from executed transitions and selects actions using a predictive model and the learned value function. Its population target can differ from the Bellman optimality target used in fitted value iteration [1]. With a frozen bootstrap $\dot { V }$ and fresh true-kernel outcomes, its observed-successor target has population response ${ \mathcal { T } } ^ { \beta } V$ for the acting policy �; policy improvement is governed by ${ \mathcal { T V } } .$ . At any state where $\beta$ assigns positive probability to an action that does not maximize $Q _ { V }$ , the two operators differ, even with unlimited data and exact optimization.

We derive a finite-horizon policy-loss bound by decomposing the error relative to the Bellman optimality update into six terms: fitting, transition reuse, target construction, replay, action selection, and exploration. The action-selection term accounts for network drift and score approximation. Bounds for a particular replay scheme, optimizer, or state representation enter the policy-loss theorem through the corresponding residuals.

Model-based action selection with a scalar value function appears in CADRL and SA-CADRL [2, 3]. In the SARL pseudocode, a stored transition receives a frozen-target label when it is sampled for a minibatch, the SAMPLE/OBS convention. EB-CADRL forms the corresponding observed-successor label when the transition is stored, the STORE/ OBS convention [4, 5]. These choices determine whether the bootstrap uses the current or an older target network. Building on EB-CADRL, HMP-DRL [6] uses deep �-learning for local control in long-range navigation. It incorporates checkpoints from a graph-based global planner into the state representation and reward function. For the analysis, a full simulator or belief state is augmented with the remaining-step clock whenever it satisfies (13). Other observation vectors are treated as measurable compressions, with aliasing, closure, and score errors handled by Proposition 3.17.

## 1.1 Algorithms and analysis

We distinguish the replay-based algorithm $\mathsf { A } _ { \mathrm { r u n } } .$ the generative-reset procedure $\mathsf { A } _ { \mathrm { i i d } }$ , and the residual recursion R used to bound policy loss. The recursion applies whenever the six errors satisfy their stated envelopes. For $\mathsf { A } _ { \mathrm { i i d } }$ , we derive statistical fit bounds using fresh blocks from a single reset distribution. For $\mathsf { A } _ { \mathrm { r u n } }$ , bounds on the six residuals connec the implemented updates to the same recursion:

$$
\mathrm { A _ { r u n } \ \xrightarrow { \ s i x \ r e s i d u a l { \ b o u n d s } } { } \mathbb { R } , \qquad \mathsf { A } _ { i i d } \ \xrightarrow { \ s t a t i s t i c a l { \ f i t { \ b o u n d } } } { \mathsf { R } } . }\tag{1}
$$

For the matched-budget comparison, $\mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } }$ draws samples separately at each horizon level. Statement tags identify the setting or procedure to which a result applies; G denotes the general setting.

## Main results.

(C1) Convergence through residual propagation. The expected policy-loss bound is a weighted sum of the six residual envelopes and an initialization term. Its explicit weights vanish outside the most recent $H - 1$ update blocks, and the initialization term vanishes for $K \geq H - 1$ (Theorem 3.18).

(C2) Sharp horizon coefficients and sample allocation. We characterize the horizon dependence of the shared-law propagation coefficient and construct a family that attains the bound (Theorem 3.6 and Corollary 3.7). Direct level sampling gives a separate propagation bound and an exact sample count (Theorem 3.19). Under matched label budgets, we derive the optimal continuous allocation, its constrained water-filling solution, and an integer schedule within a factor $2 ^ { \nu }$ of the constrained statistical optimum (Theorem 3.20). The resulting horizon orders are given in Corollary 3.21.

(C3) Sharp action-gap and deployment bounds. The action-residual exponent depends on the norm used to measure error. We derive these exponents and prove one-step sharpness (Theorem 4.5 and Proposition 4.7). An optimal gap condition and bounds on score approximation yield gap bounds for the frozen iterates while accounting for optimal ties (Proposition 4.3). A separate construction shows that score error can sustain a positive policy-loss floor (Proposition 4.8). We also bound the loss of the controller selected by the implemented score (Theorem 6.1 and Corollary 6.2).

(C4) Nonasymptotic neural and tabular convergence. For the generative-reset procedure, we obtain neural regression and policy-loss rates, along with a log-free expected fit rate in the tabular case (Proposition 5.5, Theorem 6.3, and Corollary 5.7). With exact action scores, the procedure achieves expected policy-loss consistency (Corollary 6.4). We verify Bellman closure and the network-class assumptions in a non-tabular finite-rank model (Propositions 5.2 and 5.3).

Action-indexed regression retains the executed action as an input. Expected SARSA uses this structure and averages the next action [7, §§6.1, 6.6]. Remark 2.5 contrasts these targets with state-only regression, and Section 6.4 compares the corresponding fitted-value, neural-�, action-gap, and replay analyses.

## 2 Model and algorithm

Algorithm 1 specifies the replay-based procedure $\mathsf { A } _ { \mathrm { r u n } }$ . The analysis distinguishes six design choices:

(S1) a scalar value head $V _ { \theta } { : }$

(S2) a one-step score built from an implemented reward and transition model,

$$
\widehat { Q } _ { V } ( \tilde { s } , a ) : = \widehat { R } ( \tilde { s } , a ) + \gamma V \bigl ( \widehat { P } ( \tilde { s } , a ) \bigr ) , \qquad \widehat { a } ^ { \star } ( \tilde { s } ; V ) : = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \widehat { Q } _ { V } ( \tilde { s } , a ) ,\tag{2}
$$

used in place of $\operatorname* { m a x } _ { a } Q ;$

(S3) the network, online or frozen, at which that score is evaluated for behavior;

(S4) a replay law;

(S5) an exploration rule;

(S6) a target construction: WHEN the bootstrap is evaluated (at sampling time, or once at storage time) and FROM which successor (the observed one, or a model prediction).

Slots (S3) and (S6) generate the drift and target-construction residuals.

Algorithm 1 Deep � -learning template for $\mathsf { A } _ { \mathrm { r u n } }$   
1: Input: $\gamma , \{ \varepsilon _ { k } , U _ { k } , c _ { k , j } ^ { \mathrm { e n v } } , g _ { k , j } \} , B , B _ { \mathrm { m b } } , \xi _ { k } ;$ fix normalization and the two switches in (S6)   
2: Initialize clipped $V _ { \theta } ,$ FIFO buffer �, and $\hat { V }  \bar { V } _ { \theta }$ ◁ �<sub>0</sub>   
3: for target period $k = 0 , 1 , 2 , \hdots \mathrm { d } \mathbf { o }$   
4: Freeze $V _ { k }  \hat { V } ;$ set $V _ { \theta _ { k , 0 } }  \bar { V } _ { \theta }$   
5: for round $j = 0 , 1 , \ldots , \bar { U } _ { k } - 1$ do   
6: Collect $c _ { k , j } ^ { \mathrm { e n v } }$ transitions with the online $\varepsilon _ { k }$ -greedy score; store raw tuples or stored targets according to (S6), evicting   
beyond �   
7: for $g _ { k , j }$ minibatches from � do   
8: If SAMPLE, form $y _ { i } = r _ { i } + \gamma ( 1 - d _ { i } ) \hat { V } ( \mathsf { s u c c } _ { \mathrm { F R O M } } ( x _ { i } , \mathbf { a } _ { i } , x _ { i } ^ { \prime } ) )$ ; take one semi-gradient step on $\begin{array} { r } { B _ { \mathrm { m b } } ^ { - 1 } \sum _ { i } ( \bar { V } _ { \boldsymbol { \theta } } ( x _ { i } ) - y _ { i } ) ^ { 2 } } \end{array}$   
9: end for   
10: end for   
11: Copy: $\hat { V }  \bar { V } _ { \theta }$ $\triangleright V _ { k + 1 }$   
12: end for

Here $\mathsf { s u c c } _ { \mathrm { o b s } } ( x , \mathbf { a } , x ^ { \prime } ) = x ^ { \prime } , \mathsf { s u c c } _ { \mathrm { P R E D } } ( x , \mathbf { a } , x ^ { \prime } ) = \widehat { P } ( x , \mathbf { a } )$ , and a stored target uses both the target network and, for PRED, the predictor installed at collection time. The observed mask makes terminal labels equal their observed rewards. The target network is constant within a period and treated as a constant in the semi-gradient; the behavior network may drift.

Generative-reset variant. For $\mathsf { A } _ { \mathrm { i i d } }$ , fix the frozen $V _ { k } ,$ , an acting snapshot $W _ { k } \in \mathcal { V } ^ { \mathrm { c l i p } }$ , and its $\varepsilon _ { k } .$ -greedy law $\beta _ { k }$ , then draw conditionally independently

$$
S _ { i } \sim \varsigma , \quad \mathbf { a } _ { i } \sim \beta _ { k } ( \cdot \mid S _ { i } ) , \quad ( r _ { i } , \tilde { s } _ { i } ^ { \prime } , d _ { i } ) \sim \mathsf { P } ( \cdot \mid S _ { i } , \mathbf { a } _ { i } ) \mathrm { ~ f r e s h } , \quad Y _ { i } = r _ { i } + \gamma ( 1 - d _ { i } ) V _ { k } ( \tilde { s } _ { i } ^ { \prime } ) ,\tag{3}
$$

$i \leq n _ { k }$ . Each outcome is used once, and $V _ { k + 1 }$ is a measurable $\zeta _ { n _ { k } }$ -approximate ERM over $\mathcal { F } _ { n _ { k } } ^ { V , 0 }$ . Here $\rho _ { k , S } = \varsigma$ by Assumption 3.4. This defines the generative-reset procedure $\mathsf { A } _ { \mathrm { i i d } }$ in (1). It fits the scalar target generated by the executed action; action-indexed FVI uses a different response [1]. The variant $\mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } }$ samples directly from the slice laws of Theorem 3.19.

Score-access convention. A reset call in (3) returns one observed reward–successor pair. Access to the conditional expectation $\begin{array} { r } { Q _ { V } ( \tilde { s } , a ) = R ( \tilde { s } , a ) + \gamma \int V d \mathcal { P } ( \cdot \mid \tilde { s } , a ) } \end{array}$ is represented by the exact-score oracle $Q _ { V } ^ { \mathrm { o r } } : = Q _ { V }$ , distinct from the point score $\widehat { Q } _ { V }$ in (2). Every result using $Q ^ { \mathrm { o r } }$ assumes that additional access and replaces slot (S2) by $Q ^ { \mathrm { o r } }$ exact reward and successor maps provide it for a deterministic model. Otherwise a quantitative score estimator enters the residual bound through $\eta _ { \mathrm { s c } , k } ;$ Appendix B gives one Monte Carlo bound under the stated stronger query access.

We use three histories: $\mathcal { F } _ { k , j }$ for literal rounds; $\mathcal { H } _ { k }$ for the frozen $\mathsf { A } _ { \mathrm { i i d } }$ objects before its conditionally i.i.d. block; and $\mathcal { T } _ { k }$ for the abstract block. The latter is pre-block information under population accounting and $\boldsymbol { B } _ { k }$ under realized-buffer accounting; for $\mathsf { A } _ { \mathrm { i i d } } , \mathcal { T } _ { k } = \mathcal { H } _ { k }$ . Each history is generated by a random element taking values in a standard-Borel space. All design/replay laws and responses in one theorem application are measurable for that same history. Normalization is fixed and every network is clipped by (11).

Table 1: Conditioning guide. “Deterministic” means uniform over the histories in the theorem application.
<table><tr><td>Role</td><td>Principal symbols</td><td>Probability status and use</td></tr><tr><td>Histories and frozen objects</td><td> $\mathcal { F } _ { k , j } , \mathcal { H } _ { k } , \mathcal { T } _ { k } ; V _ { k } , W _ { k } , \rho _ { k , S } , G _ { k }$ </td><td>Information before a literal round, a fresh block, or an abstract block; the latter objects are  $\mathcal { T } _ { k }$  -measurable.</td></tr><tr><td>Responses and residuals</td><td> $a _ { \mathrm { a l i a s } , k } , \bar { \Delta } _ { k } ^ { \mathrm { e x p } } ;$ </td><td>The first two quantities are  $\mathcal { T } _ { k }$  -measurable. The fit envelope bounds a conditional mean; the other five residual envelopes and  $\stackrel { \cdot } { D } _ { k } ^ { \mathrm { e x p } }$  are</td></tr><tr><td></td><td> $\varepsilon _ { \mathrm { f i t } , k } , \varepsilon _ { \mathrm { k e r } , k } , \varepsilon _ { \mathrm { t g t } , k } , \varepsilon _ { \mathrm { b u f } , k } , \varepsilon _ { \mathrm { a c t } , k } ^ { ( p ) } , \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } }$ </td><td>deterministic almost-sure bounds.</td></tr><tr><td>Propagation Terminal-window allocation</td><td> ${ \cal B } _ { K } , w _ { K , k } ^ { ( H ) } , d _ { s } ( m ) , c _ { 2 } ^ { \mathrm { l e v } } ( m )$   $n _ { i } , L _ { i } , \mathsf { N } , b _ { i } , \nu , \mathsf { C } _ { \nu , K } , \Psi _ { \nu } , F _ { K }$ </td><td>Deterministic boundary, residual weights, and coverage coefficients Planned accepted-label allocations, validity thresholds, and statistical and floor constants in Theorem 3.20.</td></tr></table>

## 2.1 Model assumptions, clipping, and the � -backup

Standing assumptions. Throughout,

$$
H \in \mathbb { N } , \qquad 0 < \gamma < 1 , \qquad 0 < R _ { \operatorname* { m a x } } < \infty , \qquad V _ { \operatorname* { m a x } } ^ { ( 0 ) } : = 0 ,\tag{4}
$$

and $2 \leq | { \mathcal { A } } | < \infty$ . The physical state space $( \cal { S } , \Sigma _ { \cal { S } } )$ is nonempty and standard Borel, � has the discrete �-algebra and a fixed total order, and

$$
\widetilde { S } = ( S \times \{ 1 , \dots , H \} ) \sqcup \{ \tilde { s } _ { \mathrm { t e r m } } \}
$$

has the corresponding disjoint-union standard-Borel structure. Observation and parameter spaces used below are also standard Borel. Put

$$
\widetilde { S } ^ { \circ } : = \widetilde { S } \setminus \{ \widetilde { s } _ { \mathrm { t e r m } } \} , \qquad \mu _ { S } ^ { \circ } ( B ) : = \mu _ { S } ( B \cap \widetilde { S } ^ { \circ } ) , \qquad R ( \widetilde { s } , a ) : = \int r \mathsf { P } ( d r , d \widetilde { s } ^ { \prime } \mid \widetilde { s } , a ) .\tag{5}
$$

For the next-state marginal $\mathcal { P }$ of the joint kernel in Assumption 2.1, define, for measurable $B \subseteq { \tilde { S } } ^ { \circ }$

$$
\mathcal { P } _ { \circ } ( B \mid \tilde { s } , a ) : = \mathcal { P } ( B \mid \tilde { s } , a ) , \qquad \mathcal { P } _ { \circ } ^ { \pi } ( B \mid \tilde { s } ) : = \int _ { A } \mathcal { P } _ { \circ } ( B \mid \tilde { s } , a ) \pi ( d a \mid \tilde { s } ) .\tag{6}
$$

Thus $\mu _ { S } ^ { \circ }$ is a subprobability restriction and $\mathcal { P } _ { \mathrm { ~ o ~ } } ^ { \pi }$ is a substochastic kernel that integrates only over the nonterminal space. Values vanish at $\tilde { s } _ { \mathrm { t e r m } } .$ , so $L ^ { 1 } ( \mu _ { S } )$ and $\breve { L } ^ { 1 } ( \mu _ { S } ^ { \circ } )$ losses coincide. Function inequalities are pointwise unless a measure is named. The action gap is $\begin{array} { r } { \Delta _ { Q } ^ { ( k ) } ( \tilde { s } ) : = \operatorname* { m a x } _ { a } Q _ { V _ { k } } ( \tilde { s } , a ) - \operatorname* { m a x } _ { a \neq a _ { k } ^ { \mathrm { t g t } } ( \tilde { s } ) } Q _ { V _ { k } } ( \tilde { s } , a ) } \end{array}$ at the frozen $Q _ { V _ { k } }$ , with $a _ { k } ^ { \mathrm { { t g t } } }$ selected by the fixed Borel tie-breaking rule, so that $\Delta _ { Q } ^ { ( k ) } = 0$ exactly at a tie. For comparison with a fixed optimal-gap hypothesis, write

$$
\Delta _ { Q } ^ { \star } ( \tilde { s } ) : = \operatorname* { m a x } _ { a } Q _ { V ^ { \star } } ( \tilde { s } , a ) - \operatorname* { m a x } _ { a \neq a ^ { \star } ( \tilde { s } ; V ^ { \star } ) } Q _ { V ^ { \star } } ( \tilde { s } , a ) ,\tag{7}
$$

using the same rule; again $\Delta _ { Q } ^ { \star } = 0$ exactly at an optimal tie.

The horizon is � steps and the augmented state is $\tilde { s } = ( s , h )$ with ℎ the number of steps remaining. A policy called stationary below is stationary on this augmented state; viewed on the physical state alone, its clock dependence makes it nonstationary. Rewards obey $| r | \leq R _ { \mathrm { m a x } } ^ { \bar { } }$ . For every measurable stationary augmented-state policy �, let $V ^ { \pi }$ be the unique fixed point of $\tau ^ { \pi }$ . Bellman optimality gives $V ^ { \star } \geq V ^ { \pi }$ pointwise, so the expected discounted regret from the evaluation law is exactly

$$
\| V ^ { \star } - V ^ { \pi } \| _ { 1 , \mu _ { S } } = \int ( V ^ { \star } - V ^ { \pi } ) d \mu _ { S } .\tag{8}
$$

The level-ℎ value bound and its recursion are

$$
V _ { \operatorname* { m a x } } ^ { ( h ) } : = R _ { \operatorname* { m a x } } \frac { 1 - \gamma ^ { h } } { 1 - \gamma } , \qquad V _ { \operatorname* { m a x } } : = V _ { \operatorname* { m a x } } ^ { ( H ) } ,\tag{9}
$$

$$
R _ { \operatorname* { m a x } } + \gamma V _ { \operatorname* { m a x } } ^ { ( h - 1 ) } = R _ { \operatorname* { m a x } } \frac { ( 1 - \gamma ) + \gamma - \gamma ^ { h } } { 1 - \gamma } = V _ { \operatorname* { m a x } } ^ { ( h ) } \qquad ( 1 \leq h \leq H ) .\tag{10}
$$

Equality in (10) is the recursion used to prove that the level-wise band below is �-invariant. To agree with terminal masking, work on

$$
\begin{array} { r l } & { \qquad \mathcal { V } : = \big \{ V : \widetilde { S }  \mathbb { R } \mathrm { ~ b o u n d e d ~ a n d ~ m e a s u r a b l e : ~ } V ( \widetilde { s } _ { \mathrm { t e r m } } ) = 0 \big \} \subset B _ { b } ( \widetilde { S } ) , } \\ & { \| V \| _ { \infty } : = \underset { \widetilde { s } } { \operatorname* { s u p } } | V ( \widetilde { s } ) | . } \end{array}
$$

This is a closed Banach subspace of bounded measurable functions [8]. A fixed Borel tie rule supplies measurable greedy selectors. Every network is evaluated through

$$
\begin{array} { r } { \bar { V } _ { \theta } ( s , h ) = \mathrm { c l i p } \big ( V _ { \theta } ^ { \mathrm { r a w } } ( s , h ) , - V _ { \mathrm { m a x } } ^ { ( h ) } , V _ { \mathrm { m a x } } ^ { ( h ) } \big ) , } \end{array}\tag{11}
$$

whose image is the band $\mathcal { V } ^ { \mathrm { c l i p } } : = \{ V \in \mathcal { V } : | V ( s , h ) | \leq V _ { \operatorname* { m a x } } ^ { ( h ) } \forall ( s , h ) \}$ . The Bellman optimality operator and the induced one-step �-function are

$$
( T V ) ( \tilde { s } ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { V } ( \tilde { s } , a ) , \qquad Q _ { V } ( \tilde { s } , a ) : = R ( \tilde { s } , a ) + \gamma \int V ( \tilde { s } ^ { \prime } ) \mathcal { P } ( d \tilde { s } ^ { \prime } \mid \tilde { s } , a ) ,\tag{12}
$$

and $\mathcal { T } ^ { \pi }$ denotes the policy operator obtained by averaging $Q _ { V }$ over � instead of maximizing. The next three results establish the clipping property, the � -backup identity, and existence of the fixed point $V ^ { \star }$

Assumption 2.1 (Markov-sufficient analysis state, joint kernel, and terminal convention [ G ]). Let $\mathcal { H } _ { t } ^ { \mathrm { s y s } }$ denote the full controlled history ofthe underlying system, including latent variables when the analysis uses them; the resulting analysis state need not be the implemented input. The augmented representation is required to be controlled Markovfor the joint reward–successor law: there are a measurable map Ψ and a measurable kernel $\mathsf { P } ( d r , d \tilde { s } ^ { \prime } \mid \tilde { s } , a )$ on $\mathbb { R } \times \tilde { \mathcal { S } }$ such that $\tilde { s } _ { t } = \Psi ( \mathcal { H } _ { t } ^ { \mathrm { s y s } } )$ includes the remaining-step clock and, under every admissible control law,for every bounded measurable $f : \mathbb { R } \times \widetilde { S }  \mathbb { R }$

$$
{ \mathbb E } [ f ( r _ { t } , \tilde { s } _ { t + 1 } ) \mid { \mathcal H } _ { t } ^ { \mathrm { s y s } } , a _ { t } ] = \int f ( r , \tilde { s } ^ { \prime } ) { \mathbb P } ( d r , d \tilde { s } ^ { \prime } \mid \tilde { s } _ { t } , a _ { t } ) \quad a . s .\tag{13}
$$

This controlled-kernel identity includes deterministic interventions and does not condition on a possibly zero-probability event $\{ a _ { t } = a \}$ . Thus histories with the same represented state have the same conditional joint law under every action. This is the Markov-sufficiency requirement; appending a clock supplies the temporal coordinate, while the controlled-kernel identity supplies the required state sufficiency. Rewards satisfy $| r | \leq R _ { \operatorname* { m a x } } a . s$ . Write � for the next-state marginal. Every terminating transition has successor $\tilde { s } _ { \mathrm { t e r m } } ,$ and $\mathsf { P } ( \{ \bar { 0 } \} \dot { \times } \{ \tilde { s } _ { \mathrm { t e r m } } \} \mid \tilde { s } _ { \mathrm { t e r m } } , a ) = 1$ . Define the terminal mask as the following function of the successor:

$$
d : = \mathbf { 1 } \{ \tilde { s } ^ { \prime } = \tilde { s } _ { \mathrm { t e r m } } \} .\tag{14}
$$

The remaining-step coordinate is a genuine clock: from $( s , h )$ with $h > 1$ every nonterminal successor lies one level lower, while the episode may terminate at any level; from $h = 1$ absorption is certain,

$$
\mathsf { P } \Big ( \mathbb { R } \times \big ( ( \mathcal { S } \times \{ h - 1 \} ) \cup \{ \tilde { s } _ { \mathrm { t e r m } } \} \big ) \Big | \Big ( s , h \big ) , a \Big ) = 1 \ ( h > 1 ) , \qquad \mathsf { P } \big ( \mathbb { R } \times \{ \tilde { s } _ { \mathrm { t e r m } } \} \mid ( s , 1 ) , a \big ) = 1 .\tag{15}
$$

Hence the substochastic nonterminal kernel $\mathcal { P } _ { \circ } ^ { \pi }$ of $\ S 3 . 2$ is nilpotent, and so is every product of � of them: $\mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { \pi _ { H } } = 0 .$ for policies that need not coincide.

Lemma 2.2 (Measurable selectors, mixtures, and conditional responses [ G ]). Under the standing standard-Borel assumptions:

(i) the fixed-order maximizer of any jointly measurable real score on $\widetilde { S } ^ { \circ } \times { A }$ is measurable;

(ii) a finite state-dependent convex mixture of measurable Markov policies is a measurable Markov policy;

(iii) $\begin{array} { r } { \textsf { f M } ( d \tilde { s } , d a ) = \sum _ { i = 1 } ^ { J } \omega _ { j } \mu _ { j } ( d \tilde { s } ) \pi _ { j } ( d a \mid \tilde { s } ) } \end{array}$ and $\begin{array} { r } { \mathsf { M } _ { S } = \sum _ { j } \omega _ { j } \mu _ { j } } \end{array}$ , where $\omega _ { j } \geq 0$ and $\textstyle \sum _ { j } \omega _ { j } = 1$ (with all these measures and policies possibly depending measurably on a standard-Borel history), then jointly measurable versions $f _ { j } = \bar { d } ( \omega _ { j } \mu _ { j } ) \dot { / } d \mathsf { M } _ { S } \dot { \in } [ 0 , \bar { 1 } ]$ may be chosen. With $z = \textstyle \sum _ { j } f _ { j }$ and anyfixed measurable reference policy �<sub>0</sub>,

$$
\bar { \pi } ( d a \mid \tilde { s } ) : = \left\{ \begin{array} { l l } { { z ( \tilde { s } ) ^ { - 1 } \sum _ { j = 1 } ^ { J } f _ { j } ( \tilde { s } ) \pi _ { j } ( d a \mid \tilde { s } ) , } } & { { z ( \tilde { s } ) > 0 , } } \\ { { \pi _ { 0 } ( d a \mid \tilde { s } ) , } } & { { z ( \tilde { s } ) = 0 , } } \end{array} \right.\tag{16}
$$

is a Markov kernel and a version ofthe conditional action law ${ \cal M } ( d a \mid \tilde { s } ) ,$ (iv) every bounded measurable label generatedfrom a standard-Borel history, state, action, andfresh kernel draw admits a measurable conditional-response version $G _ { k } ( \tilde { s } ) = \mathbb { E } [ Y _ { k } \mid S = \tilde { s } , \mathcal { T } _ { k } ]$

All policies, replay disintegrations, and responses in the sequel refer to thesefixed versions.

Proof. For finite ordered �, each selector event is a finite intersection of measurable score comparisons, proving (i); kernel integration and finite sums give (ii). Since $\omega _ { j } \mu _ { j } \leq \mathsf { M } _ { S }$ , including when $\omega _ { j } = 0$ , bounded density versions exist with $\begin{array} { r } { \sum _ { j } \bar { f _ { j } } = 1 \mathsf { M } _ { S } \mathbf { - a } . \mathbf { e } . } \end{array}$ ; substitution against measurable rectangles proves (iii). The parameterized disintegration theorem on standard-Borel spaces supplies the jointly measurable conditional kernel (equivalently the displayed finite mixture weights) and its almost-sure uniqueness. Integrating the bounded label against this kernel proves (iv) [9, Proposition 7.27 and Corollary 7.27.1]. □

Lemma 2.3 (Level-wise clipping is a projection [ G ]). Let $\Pi ^ { \mathrm { c l i p } }$ be the statewise clip of (11). For $g \in \mathcal { V } ^ { \mathrm { c l i p } } , | \Pi ^ { \mathrm { c l i p } } f -$ $g | \leq | f - g |$ pointwise; hence it is the $\bar { L ^ { 2 } } ( \rho )$ metric projection, is nonexpansive in $\dot { L } ^ { 2 } ( \rho )$ and supremum norm, and does not increase covering or Bellman-approximation error. Moreover,

$$
V \in \gamma ^ { \mathrm { c l i p } } \quad \Longrightarrow \quad | Q _ { V } ( s , h , a ) | \leq V _ { \mathrm { m a x } } ^ { ( h ) } , \qquad | Y _ { i } | \leq V _ { \mathrm { m a x } } , \qquad 7 V \in \gamma ^ { \mathrm { c l i p } } .\tag{17}
$$

These level-wisefactsfollowfrom (10); a single global clip need not make the same score and label bounds invariant.

Proof: See Appendix A.

Lemma 2.4 (Bellman � -backup through the induced �-function [ G ]). Under Assumption 2.1, for every $V \in \mathcal { V } .$

(i) $i f ( r , \tilde { s } ^ { \prime } , d ) \sim \mathsf { P } ( \cdot , \cdot | \tilde { s } , a )$ with mask �, then $\mathbb { E } [ r + \gamma ( 1 - d ) V ( \tilde { s } ^ { \prime } ) \mid \tilde { s } , a ] = Q _ { V } ( \tilde { s } , a ) ;$

(ii) $( \mathcal { T } V ) ( \tilde { s } ) = \operatorname* { m a x } _ { a } Q _ { V } ( \tilde { s } , a ) ;$

(iii) $i f a _ { t } = a ^ { \star } ( \tilde { s } _ { t } ; V )$ then $\mathbb { E } [ r _ { t } + \gamma ( 1 - d _ { t } ) V ( \tilde { s } _ { t + 1 } ) \ | \ \tilde { s } _ { t } , a _ { t } ] = ( \mathcal { T } V ) ( \tilde { s } _ { t } ) ;$

(iv) if $V \in \mathcal { V } ^ { \mathrm { c l i p } }$ and $a _ { t }$ is �-greedy for $Q _ { V }$ under exploration law �, the discrepancy equals � times the dispersion of �<sub>�</sub> over �:

$$
( { \cal T } V ) ( \tilde { s } ) - \mathbb { E } \big [ r _ { t } + \gamma ( 1 - d _ { t } ) V ( \tilde { s } _ { t + 1 } ) \ \big | \ \tilde { s } _ { t } = \tilde { s } \big ] \ = \ \varepsilon \Delta ^ { \mathrm { e x p } } [ V , \nu ] ( \tilde { s } ) \ \geq \ 0 ,\tag{18}
$$

where

$$
\begin{array} { c } { \displaystyle \Delta ^ { \mathrm { e x p } } [ V , \nu ] ( \tilde { s } ) : = \operatorname* { m a x } _ { a } Q _ { V } ( \tilde { s } , a ) - \int Q _ { V } ( \tilde { s } , a ) \nu ( d a \mid \tilde { s } ) , } \\ { 0 \leq \Delta ^ { \mathrm { e x p } } [ V , \nu ] ( s , h ) \leq 2 V _ { \mathrm { m a x } } ^ { ( h ) } . } \end{array}
$$

Proof. (i) is (12) with $V ( \tilde { s } _ { \mathrm { t e r m } } ) = 0 ; ( \mathrm { i i } )$ is the definition; and (iii) follows from (i)–(ii). For (iv), under exploration law $\nu ,$ an �-greedy rule puts mass $1 - \varepsilon$ on the maximizer and � on $\nu ,$ so by (i) and (ii)

$$
\mathbb { E } [ Q _ { V } ( \tilde { s } , a ) \mid \tilde { s } ] - \operatorname* { m a x } _ { a } Q _ { V } ( \tilde { s } , a ) = \varepsilon \big ( \mathbb { E } _ { \nu } [ Q _ { V } ] - \operatorname* { m a x } _ { a } Q _ { V } \big ) = - \varepsilon \Delta ^ { \mathrm { e x p } } [ V , \nu ] ( \tilde { s } ) ,
$$

which is (18). Its sign is fixed because the maximum dominates any average. For the statewise range, the successor of a state at level ℎ lies at level ℎ−1 or is terminal, so the level recursion (10) gives $| Q _ { V } ( s , h , a ) | \leq R _ { \operatorname* { m a x } } + \gamma V _ { \operatorname* { m a x } } ^ { ( h - 1 ) } = V _ { \operatorname* { m a x } } ^ { ( h ) }$ and hence $\Delta ^ { \mathrm { e x p } } [ V , \nu ] ( s , h ) \in [ 0 , 2 V _ { \mathrm { m a x } } ^ { ( h ) } ]$ . Lemma 3.13 uses this identity. The upper endpoint is attained only where the maximum score equals $V _ { \mathrm { m a x } } ^ { ( h ) }$ and � concentrates on actions whose score is $- V _ { \mathrm { m a x } } ^ { ( h ) }$ □

Remark 2.5 (Executed-transition targets and action-indexed heads [ R ]). The operator analysis needs only (E1) $\mathbb { E } [ Y \mid \tilde { s } , a ] = Q _ { u _ { k } } ( \tilde { s } , a )$ for the executed action and (E2) a state-only regressand, whose response is $\begin{array} { r } { { \cal T } ^ { \beta } u _ { k } = } \end{array}$ $\textstyle \int Q _ { u _ { k } } ( \cdot , a ) \beta ( d a \mid \cdot )$ . It separates consistency $\| G _ { k } - \mathcal { T } ^ { \beta } u _ { k } \|$ from optimality $\| \mathcal { T } ^ { \beta } u _ { k } - \mathcal { T } u _ { k } \|$ . Thus the abstract results extend to any head satisfying (E1)–(E2).

A �-head and Expected SARSA index the regressand by the action and thereforefall outside (E2). One-step state-value TD satisfies it and evaluates $\mathcal { T } ^ { \beta }$ unless policy improvement is added. The routed scalar-output entropy bound in Section 5 includes the �-headfactor in (86) and no output-coordinatefactor $| { \mathcal { A } } | ;$ its approximation constants may still depend on the action set.

Lemma 2.6 (Existence and uniqueness of $V ^ { \star } \left[ \mathsf { G } \right] )$ . Under Assumption 2.1 the operator � is a �-contraction on � and leaves $\scriptstyle \gamma ^ { \mathrm { c l i p } }$ invariant, so it has a unique fixed point $V ^ { \star } \in \mathcal { V } ^ { \mathrm { c l i p } }$ , with $| V ^ { \star } ( s , h ) | \leq V _ { \mathrm { m a x } } ^ { ( h ) }$ at every level, and $\begin{array} { r } { V ^ { \star } ( \tilde { s } ) = \operatorname* { s u p } _ { \pi } \mathbb { E } _ { \pi } [ \sum _ { t > 0 } \gamma ^ { t } R _ { t } \mid \tilde { s } _ { 0 } = \tilde { s } ] } \end{array}$

Proof. For $V , W \in \mathcal { V }$ , finiteness of � and kernel integration give $\begin{array} { r } { | { \mathcal T } V ( \tilde { s } ) - { \mathcal T } W ( \tilde { s } ) | \leq \gamma \operatorname* { m a x } _ { a } \int | V - W | d \mathcal P \leq } \end{array}$ $\gamma \| V - W \| _ { \infty }$ . The terminal convention makes � a closed, complete subspace of $B _ { b } ( \tilde { \mathcal { S } } )$ , so Banach’s theorem gives a unique fixed point; invariance from Lemma 2.3 puts it in $\mathcal { V } ^ { \mathrm { c l i p } }$ . Backward induction over the clock bounds every history-dependent randomized policy by this recursion, and the measurable greedy selector attains it [9–11]. □

## 3 Residual decomposition and policy-loss propagation

## 3.1 Coverage and residuals

Residuals are measured under replay marginals $\rho _ { k , S }$ , while policy loss is evaluated from $\mu _ { S }$ . Concentrability bounds the density ratios between the state distributions reached from $\mu _ { S }$ and the replay laws. One way to verify Assumption 3.3 is domination by a reset law �:

$$
\mu _ { S } ^ { \circ } \mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { \pi _ { m } } \ \leq \ \bar { b } \varsigma \qquad \mathrm { f o r ~ a l l } \ 1 \leq m \leq H \ \mathrm { a n d ~ a l l ~ p o l i c y ~ s e q u e n c e s } ,\tag{19}
$$

then any replay law of the mixture form

$$
\begin{array} { r } { \rho _ { k , S } : = ( 1 - \kappa ) \sigma _ { k , S } + \kappa \varsigma , \qquad \kappa \in ( 0 , 1 ] , } \end{array}\tag{20}
$$

satisfies $\rho _ { k , S } \geq \kappa \varsigma$ and therefore $c _ { 2 , k } ( m ) \leq \bar { b } / \kappa$ . Finite coverage requires reset mass on every reachable clock slice; Proposition 3.6 quantifies the cost when one probability law covers all slices. Sampling i.i.d. from the mixture (20) requires arbitrary-state reset access or a snapshot-replay oracle; under $\mathsf { A } _ { \mathrm { i i d } }$ , Assumption 3.4 fixes $\rho _ { k , S } = \varsigma$ . Lemma 3.10 pairs an $L ^ { p } ( \rho _ { k , S } )$ residual with an $L ^ { s } ( \rho _ { k , S } )$ density, and the same conjugate Hölder pairing determines the margin exponent in Proposition 5.6.

The residuals are defined as envelopes over one block. Let $\mathcal { T } _ { k } ^ { \mathrm { { o p t } } }$ index the gradient updates in block � and let $\mathcal { T } _ { k } ^ { \mathrm { a c t } }$ index every online-network snapshot actually used to select a collection action, including the initial and any post-update snapshots that act. Put $\mathcal { T } _ { k } = \mathcal { T } _ { k } ^ { \mathrm { o p t } } \cup \mathcal { T } _ { k } ^ { \mathrm { a c t } }$ . With $\mathcal { D } : = \widetilde { S } ^ { \circ } \times A$ as the common comparison domain,

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { t \in \mathcal { I } _ { k } ^ { \mathrm { a c t } } } \| V _ { \theta _ { k , t } } - V _ { k } \| _ { \infty } \leq \delta _ { V , k } , } \\ & { \displaystyle \operatorname* { s u p } _ { V \in \mathcal { F } } \displaystyle \operatorname* { s u p } _ { ( \tilde { s } , a ) \in \mathcal { D } } \left| \widehat { Q } _ { V } ( \tilde { s } , a ) - Q _ { V } ( \tilde { s } , a ) \right| \leq \eta _ { \mathrm { s c } , k } , } \\ & { \displaystyle \| T ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } \| _ { 2 , \rho _ { k , S } } \leq \varepsilon _ { \mathrm { b u f } , k } , \qquad \bar { \Delta } _ { k } ^ { \mathrm { e x p } } \leq \bar { D } _ { k } ^ { \mathrm { e x p } } . } \end{array}\tag{21}
$$

These are network drift, score error, replay shift, and exploration dispersion. The symmetric class $\mathcal { F } \subseteq \mathcal { V } ^ { \mathrm { c l i p } }$ contains every frozen iterate, every acting snapshot, and their negatives $( \boldsymbol { \mathrm { e . g . } } \dot { \pm } \bigcup _ { n } \mathcal { F } _ { n } ^ { V } )$ . Restriction to $\mathcal { F }$ is essential: over the full clipped band, fixed kernel mass away from a point prediction can make the continuation error $\Omega ( V _ { \mathrm { m a x } } )$ even with exact rewards. The unindexed $\eta _ { \mathrm { s c } } : = \operatorname* { s u p } _ { k } \eta _ { \mathrm { s c } , k }$ denotes a deterministic uniform envelope when finite. The notation $\widehat { Q } _ { V }$ suppresses a possible block index on the learned model; a fixed implemented model has constant $\eta _ { \mathrm { s c } , k }$ . In (21) and the action section, $\widehat { Q }$ denotes the score installed in slot (S2): the point score for $\mathsf { A } _ { \mathrm { r u n } }$ and for the point-score $\mathsf { A } _ { \mathrm { i i d } }$ analysis, or the distinct $Q ^ { \mathrm { o r } }$ only in statements explicitly tagged as oracle results. Proposition 3.1 concerns the point-score case.

Proposition 3.1 (Dispersion control of the implemented score [ G ]). Suppose each nonterminal clock slice carries a metric dist whose distance map is jointly measurable, the displayed conditional distances are integrable, and every $V \in { \mathcal { F } }$ has the same nondecreasing concave modulus � on that slice, with $\omega ( 0 ) = 0 .$ . Suppose also that termination condition (T) ofLemma 3.2 holds pointwise on ${ \mathcal { D } } ,$ , and that the point model respects the clock and terminal conventions. With $R ( \tilde { s } , a ) : = \mathbb { E } [ r \mid \tilde { s } , a ]$ , the block score error satisfies

$$
\operatorname* { s u p } _ { V \in \mathcal { F } ( \tilde { s } , a ) \in \mathcal { D } } \big | \widehat { Q } _ { V } ( \tilde { s } , a ) - Q _ { V } ( \tilde { s } , a ) \big | \leq \operatorname* { s u p } _ { ( \tilde { s } , a ) \in \mathcal { D } } \big | \widehat { R } ( \tilde { s } , a ) - R ( \tilde { s } , a ) \big | + \gamma \omega \Big ( \operatorname* { s u p } _ { ( \tilde { s } , a ) \in \mathcal { D } } \mathbb { E } \big [ \mathrm { d i s t } \big ( \tilde { s } ^ { \prime } , \widehat { P } ( \tilde { s } , a ) \big ) \big | \tilde { s } , a \big ] \Big ) .\tag{22}
$$

Consequently any deterministic almost-sure majorant ofthe right-hand side is an admissible choice $o f \eta _ { \mathrm { s c } , k }$ in (21).

Proof. At terminal pairs both continuations vanish. Otherwise $\widetilde { s } ^ { \prime }$ and $\widehat { P } ( \tilde { s } , a )$ lie on the same clock slice, so for every $V \in { \mathcal { F } } .$

$$
| \widehat { Q } _ { V } - Q _ { V } | \leq | \widehat { R } - R | + \gamma \mathbb { E } \omega \Big ( \mathrm { d i s t } ( \tilde { s } ^ { \prime } , \widehat { P } ( \tilde { s } , a ) ) \Big ) \leq | \widehat { R } - R | + \gamma \omega \Big ( \mathbb { E } \mathrm { d i s t } ( \tilde { s } ^ { \prime } , \widehat { P } ( \tilde { s } , a ) ) \Big )
$$

by concavity and Jensen. Take the suprema over $V$ and �.

For deterministic dynamics Proposition 3.1 becomes su ${ \mathfrak { d } } _ { \mathcal { D } } \left| { \widehat { R } } - R \right| + \gamma \omega ( \operatorname { s u p } _ { \mathcal { D } } \left\| { \widehat { P } } - F \right\| )$ . For stochastic dynamics, dispersion enlarges only this upper bound. The common modulus is an equicontinuity assumption on ${ \mathcal F } .$ , imposed separately from finite ReLU representation.

All six residual bounds and $\bar { D } _ { k } ^ { \mathrm { e x p } }$ are deterministic almost-sure envelopes. Write $Z = ( r , \tilde { s } ^ { \prime } , d )$ for a selected record’s outcome and $M _ { k }$ for its standard-Borel label-construction metadata, so that $Y _ { k } = \ell _ { k } ( S , \mathbf { a } , M _ { k } , Z )$ for a bounded measurable label map fixed by $\mathcal { T } _ { k }$ . For stored labels, $M _ { k }$ includes the target-copy age and the network parameters and, for PRED, predictor parameters used at writing; these are accounting variables and need not all be retained in the buffer. No independence between $M _ { k }$ and $Z$ is assumed. For sample-time labels whose network and predictor are fixed by $\mathcal { T } _ { k }$ $M _ { k }$ may be constant. Let $G _ { k }$ be a chosen version of $\mathbb { E } [ Y _ { k } \ ] \ S , \mathcal { T } _ { k } ]$ . Define its ideal counterpart by retaining $( S , \mathbf { a } , M _ { k } )$ and redrawing only the outcome:

$$
\begin{array} { r l } & { \mathcal { L } ( Z ^ { \circ } \mid S , \mathbf { a } , M _ { k } , \mathcal { I } _ { k } ) = \mathsf { P } ( \cdot \mid S , \mathbf { a } ) , } \\ & { \qquad \quad G _ { k } ^ { \mathrm { i d e a l } } ( S ) : = \mathbb { E } [ \ell _ { k } ( S , \mathbf { a } , M _ { k } , Z ^ { \circ } ) \mid S , \mathcal { I } _ { k } ] . } \end{array}\tag{23}
$$

The laws $\rho _ { k , S } , \beta _ { k } ^ { \mathrm { r e p } }$ and both responses are $\mathcal { T } _ { k }$ -measurable. The fit link is controlled in conditional mean; the other five links are controlled almost surely by their displayed deterministic envelopes. Thus

$$
\mathbb { E } \big [ \| V _ { k + 1 } - G _ { k } \| _ { 2 , \rho _ { k , S } } \ \big | \ \mathcal { T } _ { k } \big ] \ \leq \ \varepsilon _ { \mathrm { f i t } , k }
$$

$$
( \hbar t ! \mathrm { ~ t h e ~ s o l v e r ~ a g a i n s t ~ i t s ~ o w n ~ r e s p o n s e } ) ,\tag{24}
$$

$$
\left. G _ { k } - G _ { k } ^ { \mathrm { i d e a l } } \right. _ { 2 , \rho _ { k , S } } \ \leq \ \varepsilon _ { \mathrm { k e r } , k }
$$

$$
( k e r n e l r e a l i z a t i o n ; \mathrm { r e u s e ~ o f ~ r e a l i z e d ~ o u t c o m e s } ) ,\tag{25}
$$

$$
\left. \mathrm { G } _ { k } ^ { \mathrm { i d e a l } } - \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } \right. _ { 2 , \rho _ { k , S } } \leq \varepsilon _ { \mathrm { t g t } , k }
$$

$$
( t a r g e t c o n s t r u c t i o n \colon \mathrm { t h e ~ s w i t c h e s ~ o f ~ s l o t ~ ( S 6 ) } ) ,\tag{26}
$$

Under population-law accounting, suppose that a selected record satisfies the following conditional kernel-retention hypothesis, including the label-construction metadata: for every bounded measurable $f ,$

$$
\mathbb { E } \big [ f ( r , \tilde { s } ^ { \prime } , d ) \mid S , \mathbf { a } , M _ { k } , \mathcal { I } _ { k } \big ] = \int f \mathrm { d } \mathbf { P } ( \cdot \mid S , \mathbf { a } ) \quad \mathrm { a . s . }\tag{27}
$$

Conditioning first on $\left( S , \mathbf { a } , M _ { k } , \mathcal { I } _ { k } \right)$ and then averaging shows that $G _ { k } = G _ { k } ^ { \mathrm { i d e a l } }$ , so $\varepsilon _ { \mathrm { k e r } , k } = 0$ . For the current observed-successor label, the tower property also gives the replayed operator,

$$
\begin{array} { r } { \mathbb { E } \big [ r + \gamma ( 1 - d ) V _ { k } ( \tilde { s } ^ { \prime } ) ~ \big | ~ S = \tilde { s } , \mathcal { T } _ { k } \big ] = ( \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } ) ( \tilde { s } ) , } \end{array}\tag{28}
$$

where $\mathcal { T } _ { k }$ is pre-block information. For sample-time labels fixed by $\mathcal { T } _ { k } .$ conditioning only on $( S , \mathbf { a } , \mathcal { T } _ { k } )$ in (27) suffices. For stored labels it need not suffice: record age or writing-time parameters can remain correlated with the selected outcome. The full condition holds for a fresh true-kernel draw made after the metadata are fixed. A sample-splitting or outcome-independent retention argument must establish this conditional law given both the chosen history and the metadata; it must be verified for adaptive FIFO replay. Under realized-buffer accounting, $\mathcal { I } _ { k } = B _ { k }$ and $\varepsilon _ { \mathrm { k e r } , k }$ measures the empirical response’s deviation from its fresh-outcome counterpart, ready for a process-specific replay bound. For $\mathsf { A } _ { \mathrm { i i d } } , \varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = 0$

Lemma 3.2 gives the target-switch accounting: (SAMPLE, OBS) has zero target residual, while STORE charges conditional target-network staleness and PRED charges current continuation-score error (plus its stated terminalmask correction). Their combination also charges any change between the writing-time and current predictors. An executed-transition target therefore pays behavior, online-action, and exploration links. A model-built target ma $\ x _ { a } \{ \widehat { R } ( \tilde { s } , a ) + \gamma V _ { k } ( \widehat { P } ( \tilde { s } , a ) ) \}$ removes those links but inserts score error into every target; neither design is uniformly preferred. The replay residual $\varepsilon _ { \mathrm { b u f } , k }$ is a same-state operator distance.

Lemma 3.2 (Target-switch envelope $[ \mathsf { R } ] )$ . Let $A \in \{ 0 , \ldots , k \}$ be a replayed record’s age in target copies and ${ \bar { \delta } } _ { k } ( A ) : = \| V _ { k } - V _ { k - A } \| \infty$ . Write $\widehat { P } _ { k }$ for the current predictorfixed by $\mathcal { T } _ { k }$ (the block index previously suppressed) and $\widehat { P } _ { \mathrm { w r i t e } } f o r$ the predictor used to construct a stored prediction label, identified by $M _ { k }$ . Its version need not be determined by �. For this cell define

$$
C _ { \mathrm { p r e d } , k } ( \tilde { s } ) : = \gamma \mathbb { E } \Big [ \Big | V _ { k } \big ( \widehat { P } _ { \mathrm { w r i t e } } ( S , { \mathbf a } ) \big ) - V _ { k } \big ( \widehat { P } _ { k } ( S , { \mathbf a } ) \big ) \Big | \ \big | \ S = \tilde { s } , \mathcal { T } _ { k } \Big ] .\tag{29}
$$

This quantity vanishes for a fixed predictor; set it to zero in the other cells. Let $\begin{array} { r } { \eta _ { k } ^ { \mathrm { c o n t } } : = \operatorname* { s u p } _ { V \in \mathcal { F } , \tilde { s } , a } \gamma | V ( \widehat { P } _ { k } ( \tilde { s } , a ) ) - } \end{array}$ $\textstyle \int V d { \mathcal { P } } ( \cdot \mid { \tilde { s } } , a )$ |. When the point score is installed in slot (S2), symmetry gives $\eta _ { k } ^ { \mathrm { c o n t } } \leq \eta _ { \mathrm { s c } , k } ;$ under other score access it remains a separate target-construction quantity. Ifthe terminal event is determined by $( \tilde { s } , a )$ , then thefour cells of slot (S6) satisfy

$$
\begin{array} { r l } & { \| G _ { k } ^ { \mathrm { i d e a l } } - \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } \| _ { 2 , \rho _ { k , S } } \leq { \mathbf 1 } \{ { \mathrm { F R O M } } = { \mathrm { P R E D } } \} \eta _ { k } ^ { \mathrm { c o n t } } } \\ & { \qquad + { \mathbf 1 } \{ { \mathrm { W H E N } } = { \mathrm { S T O R E } } \} \gamma \| \mathbb { E } [ \bar { \delta } _ { k } ( A ) \ | \ S , \mathcal { T } _ { k } ] \| _ { 2 , \rho _ { k , S } } } \\ & { \qquad + { \mathbf 1 } \{ \left( { \mathrm { W H E N } } , { \mathrm { F R O M } } \right) = \left( { \mathrm { S T O R E } } , { \mathrm { P R E D } } \right) \} \| C _ { \mathrm { p r e d } , k } \| _ { 2 , \rho _ { k , S } } . } \end{array}\tag{30}
$$

In the general termination case, the predicted-successor cells add $\| C _ { \mathrm { m a s k } , k } \| _ { 2 , \rho _ { k , S } ; \ell }$ , where

$$
C _ { \mathrm { m a s k } , k } ( { \tilde { s } } ) : = \gamma \int \mathrm { P r } ( d = 1 \mid { \tilde { s } } , a ) | V _ { k } ( \widehat { P } _ { k } ( { \tilde { s } } , a ) ) | \beta _ { k } ^ { \mathrm { r e p } } ( d a \mid { \tilde { s } } ) .
$$

Thus the replay-action average is taken before the state norm, and both staleness quantities remain conditional on $S$ and � . Any deterministic almost-sure majorant ofthe resulting right-hand side is an admissible $\varepsilon _ { \mathrm { t g t } , k }$

Proof: See Appendix A.

Finally, $\delta _ { V , k } ^ { \mathrm { p a t h } } , \delta _ { V , k } ^ { \mathrm { s n a p } } , \bar { \Delta } _ { k } ^ { \mathrm { e x p } }$ , and realized fit errors are random. The symbols $\delta _ { V , k } , \eta _ { \mathrm { s c } , k } , \eta _ { \mathrm { s c } }$ , and $\eta _ { \mathrm { s c } , K }$ are deterministic, as are $\varepsilon _ { \mathrm { f i t } , k } , \varepsilon _ { \mathrm { k e r } , k } , \varepsilon _ { \mathrm { t g t } , k } , \varepsilon _ { \mathrm { b u f } , k } , \varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) } \mathrm { f o r } 1 \leq r \leq 2 , \bar { D } _ { k } ^ { \mathrm { e x p } }$ , and $\Lambda _ { k }$ . Random quantities enter later bounds only inside expectations or through such envelopes.

## 3.2 Concentrability and finite-horizon propagation

Assumption 3.3 (Finite-horizon $L ^ { s }$ concentrability of population replay [ R ]). Fix $s \in [ 2 , \infty ]$ and let $p = s / ( s - 1 )$ , with $p = 1$ when $s = \infty$ . For every $1 \leq m \leq H$ and every sequence $\pi _ { 1 } , \ldots , \pi _ { m }$ of randomized Markov policies (measurable kernelsfrom $\widetilde { S } ^ { \circ }$ into the simplex over ${ \mathcal A } ,$ , including deterministic selectors), assume $\mu _ { S } ^ { \circ } \mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { \pi _ { m } } \ll \rho _ { k , S }$ and set

$$
d _ { s , k } ( \boldsymbol { m } ) : = \operatorname* { s u p } _ { \pi _ { 1 } , \ldots , \pi _ { m } } \left. \frac { d ( \mu _ { S } ^ { \circ } \mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \ldots \mathcal { P } _ { \circ } ^ { \pi _ { m } } ) } { d \rho _ { k , S } } \right. _ { s , \rho _ { k , S } } .\tag{31}
$$

We assume the pathwise form (C-strong): a deterministic $d _ { s } ( m ) < \infty$ with $d _ { s , k } ( m ) \leq d _ { s } ( m )$ almost surely for all �; by clock nilpotence take $d _ { s } ( H ) = 0$ . The supremum in $( 3 1 )$ is pointwise over all policy sequences: after a history is fixed it therefore includes policies selectedfrom that history, although every factor remains a Markov kernel in the current state. Define

$$
\phi _ { s , K } ^ { ( H ) } : = \sum _ { m = 1 } ^ { H - 1 } \operatorname* { m i n } \{ K , m \} \gamma ^ { m } d _ { s } ( m ) , \qquad \phi _ { s } ^ { ( H ) } : = \phi _ { s , H - 1 } ^ { ( H ) } = \sum _ { m = 1 } ^ { H - 1 } m \gamma ^ { m } d _ { s } ( m ) .\tag{32}
$$

$A t \ s = 2$ write $c _ { 2 , k } ( m ) : = d _ { 2 , k } ( m ) ^ { 2 } , c _ { 2 } ( m ) : = d _ { 2 } ( m ) ^ { 2 }$ , and $\phi _ { \mu _ { S } , \rho } ^ { ( H ) } : = \phi _ { 2 } ^ { ( H ) }$ ; at $s = \infty$ write $c _ { \infty } ( m ) : = d _ { \infty } ( m )$ Hölder pairs the density with an $L ^ { p } ( \rho _ { k , S } )$ residual. Existing $L ^ { 2 }$ envelopes for the nonaction links remain valid because $p \leq 2$ and $\rho _ { k , S }$ is a probability law; the action link may use its sharper �-specific envelope.

Assumption 3.4 (Generative reset access $\big [ \mathsf { A } _ { \mathrm { i i d } } , \mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } } \big ] \big )$ . For $\mathsf { A } _ { \mathrm { i i d } }$ , the simulator can be reset to a state drawnfrom the designer’s single reset law � on ${ \widetilde { S } } ^ { \circ }$ , after which one behavior-policy action is executed and one true-kernel transition is observed. For $\mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } }$ , it can be reset independently to each $\mathcal { T } _ { k }$ -measurable clock-slice law $\rho _ { k , S } ^ { ( h ) }$ specified in Theorem 3.19. These are distinct access assumptions. If slice laws are obtained by rejection from �, rejected draws must be added to the sample count; Theorem 3.19 counts direct slice-reset draws only.

Definition 3.5 (Top-slice evaluation and horizon-indexed families [ G ]). For afixed horizon �, top-slice evaluation means

$$
\mu _ { S } ( S \times \{ H \} ) = 1 .\tag{33}
$$

A horizon-indexed family $\{ \mathcal { T } _ { H } \} _ { H \ge 2 }$ consists ofone instance ofthe augmented model and its evaluation/design objects for each �, with discount $\gamma _ { H }$ , iteration index $K _ { H }$ , evaluation law $\mu _ { S , H }$ , replay laws $\rho _ { k , S , H } ,$ , survival masses ${ \mathrm { q } } H , m ,$ and concentrability envelopes $d _ { s , H } ( m )$ . Within a fixed member, the � subscripts are suppressed. Every asymptotic symbol in � refers to this family, and every constant declared uniform is independent of �. The family is near-unit-discount when $( \mathrm { i } - \gamma _ { H } ) H \to \varkappa \in [ 0 , \infty )$ , and is fixed-discount when $\gamma _ { H } \equiv \gamma$ for one $\gamma \in ( 0 , 1 )$ .

Theorem 3.6 (Finite-� $L ^ { s } / L ^ { p }$ clock tradeoff [ R ]). Assume (C-strong), top-slice evaluation in the sense ofDefinition 3.5, $H \geq 2 ,$ , and $K \geq 1$ . For $\mu _ { m } ^ { \pi _ { 1 : m } } : = \mu _ { S } ^ { \circ } \mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { \pi _ { m } }$ , set $\begin{array} { r l } { q _ { m } : = } & { { } \operatorname* { s u p } _ { \pi _ { 1 : m } } \mu _ { m } ^ { \pi _ { 1 : m } } ( \widetilde { S } ^ { \circ } ) , a _ { K , m } : = } \end{array}$ min $\{ K , m \} \gamma ^ { m }$ , and $\theta : = p / ( p + 1 )$ (equivalently �/(2� − 1)forfinite �, and 1/2 at $s = \infty )$ . Every shared design law satisfies

$$
\sum _ { m = 1 } ^ { H - 1 } \left( \frac { q _ { m } } { d _ { s } ( m ) } \right) ^ { p } \leq 1 , \qquad \phi _ { s , K } ^ { ( H ) } \geq \left\{ \sum _ { m = 1 } ^ { H - 1 } ( a _ { K , m } q _ { m } ) ^ { \theta } \right\} ^ { 1 / \theta } .\tag{34}
$$

Zero-survival coordinates are omitted. The second bound is an exact relaxation: on the deterministic one-state-per-level chain with $q _ { m } = 1$ , the �-specific clock masses

$$
r _ { H - m } = \frac { a _ { K , m } ^ { \theta } } { \sum _ { j = 1 } ^ { H - 1 } a _ { K , j } ^ { \theta } }\tag{35}
$$

give $d _ { s } ( m ) = r _ { H - m } ^ { - 1 / p }$ and equality in (34).

Proof. The depth-� law is supported on $L _ { m } = \mathcal { S } \times \{ H - m \}$ . If $r _ { H - m } = \rho _ { k , S } ( L _ { m } )$ and $f _ { m }$ is its density, slicesupported Hölder gives $q _ { m } \leq \lVert f _ { m } \rVert _ { s } r _ { H - m } ^ { 1 / p } \leq d _ { s } ( m ) r _ { H - m } ^ { 1 / p } \mathrm { . }$ ; summing over the disjoint slices proves the first inequality. Applying Hölder with exponents $1 / \theta$ and $p + 1$ to $( a _ { K , m } d _ { s } ( m ) ) ^ { \theta } ( q _ { m } / d _ { s } ( m ) ) ^ { \theta }$ proves the second. On the stated chain the depth law is a point mass on its unique slice, so substitution of (35) gives equality. □

Corollary 3.7 (Sharp horizon regimes for the shared-law clock coefficient [ R ]). Apply Theorem 3.6 to a top-slice horizon-indexedfamily and assume the uniform survivalfloor in $\ell _ { H \geq 2 } \operatorname* { i n f } _ { 1 \leq m < H } q _ { H , m } \geq q > 0 ,$

(a) In the near-unit-discount regime, $i f K _ { H } / H \to \tau \in ( 0 , \infty ]$ , every shared design law obeys

$$
\phi _ { s , \kappa _ { H } } ^ { ( H ) } \geq q A _ { s , \tau } ( \varkappa ) H ^ { 3 - 1 / s } ( 1 + o ( 1 ) ) , \quad A _ { s , \tau } ( \varkappa ) : = \left\{ \int _ { 0 } ^ { 1 } \operatorname* { m i n } \{ \tau , x \} ^ { \theta } e ^ { - \varkappa \theta x } d x \right\} ^ { 1 / \theta } ,\tag{36}
$$

where min $\{ \infty , x \} = x .$

(b) In the same near-unit-discount regime, $i f$ instead $1 \ \leq \ K _ { H } \ = \ o ( H )$ , every shared law obeys the order $\Omega ( K _ { H } H ^ { 2 - 1 / s } )$ .

(c) In the fixed-discount regime, the sharp shared-law relaxation is $\Theta ( 1 )$ for every sequence $K _ { H } \ge 1$ , whereas the uniform �-slice law on the one-state-per-levelfamily has order $\Theta ( H ^ { 1 - 1 / s } )$ .

The deterministic one-state-per-levelfamily has $q _ { H , m } = 1$ , and its �<sub>�</sub>-specific law (35) attains the orders in $\mathrm { ( a ) - ( c ) }$ Thus each order is best possiblefor the shared-law clock-coefficient problem ofTheorem 3.6.

Proof. For (a), divide $\sum _ { m < H } a _ { K _ { H } , m } ^ { \theta }$ by $H ^ { 1 + \theta }$ and use the Riemann sum in (36); $( 1 + \theta ) / \theta \ : = \ : 3 - 1 / s$ . For $K _ { H } = o ( H )$ , split at $m = K _ { H } \colon$ the tail is $\Theta ( K _ { H } ^ { \theta } H )$ and dominates the $O ( K _ { H } ^ { 1 + \theta } )$ initial part, proving (b). Theorem $3 . 6$ gives both lower bounds and its one-state family gives equality. For (c), $a _ { K _ { H } , m } \leq m \gamma ^ { m }$ makes $\textstyle \sum _ { m } a _ { K _ { H } , m } ^ { \theta }$ uniformly finite and bounded away from zero. The $K _ { H }$ -specific masses attain this constant order, while the uniform clock law has $d _ { s } ( m ) = H ^ { 1 / p }$ and a weight sum uniformly bounded above and away from zero, giving $H ^ { 1 / p } = H ^ { 1 - 1 / s }$ □

Corollary 3.8 (Endpoint attainment and direct-level comparison [ R ]). For the deterministic one-state-per-level family in the near-unit-discount regime with $K _ { H } \ge H - 1$ , the attaining masses in (35) are proportional to $( m \gamma _ { H } ^ { m } ) ^ { 2 / 3 }$ at $s = 2$ and to $( m \gamma _ { H } ^ { m } ) ^ { 1 / 2 }$ at $s = \infty$ . These attain the optimal shared-law propagation-coefficient orders $H ^ { 5 / 2 }$ and $H ^ { 3 }$ respectively. The uniform �-slice law on the same family has the same orders, with different constants. Under the distinct direct-level-reset access model, uniformly bounded $L ^ { 2 }$ level coefficients give propagation coefficient $O ( H ^ { 2 } )$ ).

Theorem 3.19 formalizes the distinct sampling scheme that pairs each propagated depth with its own clock-slice law. Lemma 3.9 (Comparison kernel for the absolute Bellman difference $[ \mathsf { G } ] )$ . Let $V , W \in \mathcal { V }$ and $\Delta _ { V , W } : = V - W$ . There exists a measurable substochastic Markov kernel $\mathcal { P } _ { \circ } ^ { V , W }$ on $\widetilde { S } ^ { \circ }$ such that, pointwise on ${ \widetilde { S } } ^ { \circ }$

$$
| T V - T W | ( \tilde { s } ) \leq \gamma \big ( \mathcal { P } _ { \circ } ^ { V , W } | \Delta _ { V , W } | \big ) ( \tilde { s } ) .
$$

Moreover, $\mathcal { P } _ { \circ } ^ { V , W }$ may be chosen as the kernel $\mathcal { P } _ { \circ } ^ { \pi }$ ofa measurable deterministic policy � that selects pointwise between the greedy selectors $\dot { a } ^ { \star } ( \cdot ; V )$ and $a ^ { \star } ( \cdot ; W )$

Proof. Put $a _ { V } : = a ^ { \star } ( \cdot ; V )$ and $a _ { W } : = a ^ { \star } ( \cdot ; W )$ , and write $P _ { a } : = \mathcal { P } _ { \circ } ( \cdot \mid \tilde { s } , a )$ . Optimality gives

$$
\begin{array} { l } { ( \mathcal T V - \mathcal T W ) ( \tilde { s } ) \leq \gamma \displaystyle \int | \Delta _ { V , W } | d P _ { a _ { V } ( \tilde { s } ) } , } \\ { ( \mathcal T W - \mathcal T V ) ( \tilde { s } ) \leq \gamma \displaystyle \int | \Delta _ { V , W } | d P _ { a _ { W } ( \tilde { s } ) } . } \end{array}
$$

Choose at each state whichever nonnegative integral is larger. Kernel integration and the two Borel selectors make this choice measurable. The result is a deterministic policy kernel that dominates both one-sided bounds. Thus (31) covers every product formed below, although the policies in a product need not coincide. □

Each transition between nonterminal states reduces the remaining horizon, so a product of � nonterminal transition kernels is zero. This truncates error propagation and gives the finite update window in the following lemma.

Lemma 3.10 (Finite-horizon residual propagation $[ \mathsf { R } ] )$ . Assume 2.1 and 3.3. Let $V _ { 0 } , \dots , V _ { K } \in \mathcal { V }$ satisfy $\| V _ { k } \| _ { \infty } \leq$ $V _ { \mathrm { m a x } } ,$ set $e _ { k } : = V _ { k + 1 } - \mathcal { T } V _ { k } ,$ , let $K \geq { \bar { 1 } }$ , and let $\pi _ { K }$ be greedy with respect to $V _ { K }$ . Put $p _ { h } : = \mu _ { S } ^ { \circ } ( S \times \{ h \} )$ and choose deterministic $D _ { 0 , h }$ such that

$$
\operatorname* { s u p } _ { s \in S } | V ^ { \star } ( s , h ) - V _ { 0 } ( s , h ) | \leq D _ { 0 , h } \quad a l m o s t s u r e l y .
$$

The global bound permits $D _ { 0 , h } = V _ { \mathrm { m a x } } ^ { ( h ) } + V _ { \mathrm { m a x } } ;$ for deterministic $V _ { 0 } ,$ , one may use its actual slice error. Define the deterministic initialization envelope

$$
\displaystyle B _ { K } : = 2 \sum _ { h = 1 } ^ { H } p _ { h } \sum _ { m = K + 1 } ^ { h - 1 } \gamma ^ { m } D _ { 0 , h - m } .\tag{37}
$$

With deterministic weights

$$
w _ { K , k } ^ { ( H ) } : = 2 \sum _ { \stackrel { \ell \geq 0 : } { 1 \leq \ell + \overline { { K } } - k < H } } \gamma ^ { \ell + K - k } d _ { s } ( \ell + K - k ) , \qquad 0 \leq k < K ,\tag{38}
$$

thefollowing bound holds pathwise:

$$
\| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } \leq \mathcal { B } _ { K } + \sum _ { k = \operatorname* { m a x } \{ 0 , K - H + 1 \} } ^ { K - 1 } w _ { K , k } ^ { ( H ) } \| e _ { k } \| _ { p , \rho _ { k , S } } , \qquad \sum _ { k = 0 } ^ { K - 1 } w _ { K , k } ^ { ( H ) } = 2 \phi _ { s , K } ^ { ( H ) } .\tag{39}
$$

For top-slice evaluation,

$$
B _ { K } = 2 \sum _ { m = K + 1 } ^ { H - 1 } \gamma ^ { m } D _ { 0 , H - m } .
$$

$I f V _ { 0 } \in \mathcal { V } ^ { \mathrm { c l i p } }$ almost surely, choosing $D _ { 0 , h } = 2 V _ { \operatorname* { m a x } } ^ { ( h ) }$ gives

$$
B _ { K } \leq 4 \sum _ { m = K + 1 } ^ { H - 1 } \gamma ^ { m } V _ { \mathrm { m a x } } ^ { ( H - m ) } .
$$

In all cases, $\begin{array} { r } { \mathcal { B } _ { K } = 0 f o r K \ge H - 1 } \end{array}$

Proofroadmap. Lemma 3.9 gives the one-step comparison recursion. Unrolling it and applying the nonnegative loss resolvent produces two policy-kernel branches of total depth $m = \ell + K - k .$ Clock nilpotence removes $m \geq H ;$ Hölder and Assumption 3.3 give the stated weights, while counting the min $\{ K , m \}$ admissible indices gives their sum. Appendix A supplies the pathwise occupancy construction and the deterministic initialization-envelope bound.

## 3.3 Residual definitions and composition

Assumption 3.11 (Behavior policies, the replay residual, and the online-action residual [ R ]). (Behavior.) The data of block � is collected by $\varepsilon _ { k }$ -greedy policies whose greedy branch is the one implemented in Algorithm 1: it maximizes the point-model score $\widehat { Q } _ { V } o f \left( 2 \right)$ at the acting network. The oracle consistency results use the separate true conditionalexpectation score $Q _ { V } ^ { \mathrm { o r } } .$ . The action analysis applies to either choice through its generic perturbation score $q _ { k } .$ . For $\mathsf { A } _ { \mathrm { i i d } }$ there is one such policy, the snapshot law $\beta _ { k }$ builtfrom $W _ { k }$ . For $\mathsf { A } _ { \mathrm { r u n } }$ there is one per round, and the period-level object $\bar { \pi } _ { k } ^ { \mathrm { o n } }$ is obtained by disintegration ofthe joint collection measure,

$$
\mathsf { M } _ { k } ( d \tilde { s } , d a ) = \sum _ { j } \omega _ { k , j } \mu _ { k , j } ( d \tilde { s } ) \pi _ { k , j } ^ { \mathrm { o n } } ( d a \mid \tilde { s } ) = \mathsf { M } _ { k , S } ( d \tilde { s } ) \pi _ { k } ^ { \mathrm { o n } } ( d a \mid \tilde { s } ) ,\tag{40}
$$

$\omega _ { k , j }$ being the sample share and $\mu _ { k , j }$ the state law ofround �. The weights of $\bar { \pi } _ { k } ^ { \mathrm { o n } }$ are state-dependent: a time average of the $\pi _ { k , j } ^ { \mathrm { o n } }$ is in general not the conditional action law of the collected data. Fix the density-weighted version from Lemma 2.2 $\mathsf { M } _ { k , S } – a . e .$ and its declared measurable mixture extension on the $\mathsf { M } _ { k , S }$ -null complement; use the same conventionfor $\beta _ { k } ^ { \mathrm { r e p } }$ below. ${ \cal I } f \rho _ { k , S }$ charges that null complement, the disintegration identity alone imposes no relation there. Therefore, for $\mathsf { A } _ { \mathrm { r u n } } .$ , require $\rho _ { k , S } \ll \mathsf { M } _ { k , S }$ whenever $\bar { \pi } _ { k } ^ { \mathrm { o n } }$ is compared under � ; the fixed extension then serves only measurability and cannot change any displayed residual norm. Write $\pi _ { k } ^ { \mathrm { o n } }$ for $\beta _ { k }$ under $\mathsf { A } _ { \mathrm { i i d } }$ andfor $\bar { \pi } _ { k } ^ { \mathrm { o n } }$ under $\mathsf { A } _ { \mathrm { r u n } } ,$ and $\pi _ { k } ^ { \mathrm { t g t } }$ for the � -greedy policy whose greedy branch maximizes the true $Q _ { V _ { k } }$ ; all share $\varepsilon _ { k }$ and $\xi _ { k }$

(The two residuals.) Let $\beta _ { k } ^ { \mathrm { r e p } }$ be the conditional action law ofthe distribution actually replayed. Every operator written $\mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } }$ refers to this law, and $\beta _ { k } ^ { \mathrm { r e p } } = \beta _ { k } f o r \mathsf { A } _ { \mathrm { i i d } }$ , whose outcomes arefresh and used once. For each block $k ,$ let the deterministic block-indexed envelopes $\varepsilon _ { \mathrm { b u f } , k }$ and $\varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) } , 1 \leq r \leq 2 ,$ , satisfy

$$
\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\| \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } \aftergroup \egroup \right\| _ { 2 , \rho _ { k , S } } \leq \varepsilon _ { \mathrm { b u f } , k } , } \end{array}\tag{41}
$$

$$
\left\| \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \right\| _ { r , \rho _ { k , S } } \leq \varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) } , \qquad 1 \leq r \leq 2 ,\tag{42}
$$

together with the fit bound (24). Here $\varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } = \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } ;$ the norm-indexed envelopes need not be equal. These are operator distances, which is what Lemma $3 . { \dot { I } } 3$ consumes.

Remark 3.12 (A total-variation envelope for replay shift [ R ]). Write

$$
\begin{array} { r l } & { V _ { \operatorname* { m a x } , k } ^ { ( \rho ) } : = \big \| V _ { \operatorname* { m a x } } ^ { ( h ( \cdot ) ) } \big \| _ { 2 , \rho _ { k , S } } = \Big ( \sum _ { h = 1 } ^ { H } p _ { k , h } \big ( V _ { \operatorname* { m a x } } ^ { ( h ) } \big ) ^ { 2 } \Big ) ^ { 1 / 2 } , } \\ & { \qquad V _ { \operatorname* { m a x } , k } ^ { ( \rho ) } \leq \overline { { V } } _ { \operatorname* { m a x } } ^ { ( \rho ) } \leq V _ { \operatorname* { m a x } } \quad a . s . f o r e \nu e r y \ k , } \end{array}\tag{43}
$$

where $p _ { k , h } : = \rho _ { k , S } ( S \times \{ h \} )$ and $\overline { { V } } _ { \mathrm { m a x } } ^ { ( \rho ) }$ is a deterministic uniform majorant (safely, $V _ { \mathrm { m a x } } ) ;$ hence $\bar { D } _ { k } ^ { \mathrm { e x p } } : = 2 \overline { { V } } _ { \mathrm { m a x } } ^ { ( \rho ) }$ is admissible in (21). Use the convention $d _ { \mathrm { T V } } ( P , Q ) : = \operatorname* { s u p } _ { A } | P ( A ) - Q ( A ) |$ . Since $| Q _ { V _ { k } } ( s , h , a ) | \leq V _ { \mathrm { m a x } } ^ { ( h ) }$ , the replay link itself satisfies

$$
\| | \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } | | _ { 2 , { \rho _ { k , s } } } \leq 2 \| V _ { \operatorname* { m a x } } ^ { ( h ( \cdot ) ) } d _ { \mathrm { T V } } ( \beta _ { k } ^ { \mathrm { r e p } } , \pi _ { k } ^ { \mathrm { o n } } ) \| _ { 2 , { \rho _ { k , s } } } \leq 2 V _ { \operatorname* { m a x } , k } ^ { ( \rho ) } \mathrm { e s s s u p } d _ { \mathrm { T V } } ( \beta _ { k } ^ { \mathrm { r e p } } , \pi _ { k } ^ { \mathrm { o n } } ) ,\tag{44}
$$

Any deterministic majorant is admissible $f o r \ \varepsilon _ { \mathrm { b u f } , k } ; \ 2 V _ { \mathrm { m a x } }$ is safe. A quantitative FIFO decay rate follows from stabilization, age/mixing, or buffer-scaling control.

Lemma 3.13 (Executed-transition response versus Bellman response $[ \mathsf { R } ] )$ . For $r \in [ 1 , 2 ]$ , set $\widetilde { \varepsilon } _ { k , r } : = \| V _ { k + 1 } -$ $\mathcal { T } V _ { k } \Vert _ { r , \rho _ { k , S } }$ . Under Assumptions 2.1 and 3.11,for $0 \leq k < K$

$$
\mathbb { E } [ \widetilde { \varepsilon } _ { k , r } \ | \ { \mathcal { I } } _ { k } ] \leq \varepsilon _ { \mathrm { f i t } , k } + \varepsilon _ { \mathrm { k e r } , k } + \varepsilon _ { \mathrm { t g t } , k } + \varepsilon _ { \mathrm { b u f } , k } + \varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) } + \varepsilon _ { k } { \bar { D } } _ { k } ^ { \exp } .\tag{45}
$$

Proof. Apply the triangle inequality along

$$
V _ { k + 1 } \to G _ { k } \to G _ { k } ^ { \mathrm { i d e a l } } \to \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } \to \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } \to \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \to \mathcal { T } V _ { k } .
$$

Every link except the online-action link is first bounded by its named $L ^ { 2 }$ envelope and hence by the same envelope in $L ^ { r }$ , since $r \leq 2$ and $\rho _ { k , S }$ is a probability law. The online-action link uses (42). The last link is exactly $\varepsilon _ { k } \Delta _ { k } ^ { \mathrm { e x p } }$ by (18), where

$$
\begin{array} { r l } & { \Delta _ { k } ^ { \mathrm { e x p } } ( \tilde { s } ) : = \Delta ^ { \mathrm { e x p } } [ V _ { k } , \xi _ { k } ] ( \tilde { s } ) } \\ & { \qquad = \operatorname* { m a x } _ { a } Q _ { V _ { k } } ( \tilde { s } , a ) - \displaystyle \int Q _ { V _ { k } } ( \tilde { s } , a ) \xi _ { k } ( d a \mid \tilde { s } ) , \qquad \bar { \Delta } _ { k } ^ { \mathrm { e x p } } : = { \left. \Delta _ { k } ^ { \mathrm { e x p } } \right. } _ { 2 , \rho _ { k , S } } . } \end{array}\tag{46}
$$

$| Q _ { V _ { k } } ( s , h , a ) | \leq V _ { \mathrm { m a x } } ^ { ( h ) }$ gives

$$
\bar { \Delta } _ { k } ^ { \mathrm { e x p } } \ \leq \ 2 V _ { \operatorname* { m a x } , k } ^ { ( \rho ) } \ \leq \ 2 \overline { { V } } _ { \operatorname* { m a x } } ^ { ( \rho ) } \ \leq \ 2 V _ { \operatorname* { m a x } }\tag{47}
$$

and $\| \Delta _ { k } ^ { \mathrm { e x p } } \| _ { r , \rho _ { k , S } } \le \bar { \Delta } _ { k } ^ { \mathrm { e x p } } \le \bar { D } _ { k } ^ { \mathrm { e x p } }$ by (21). Taking conditional expectations proves (45).

□

Definition 3.14 (Visible-target aliasing [ R ]). Fix one realization of $\mathcal { T } _ { k }$ . Let $O : \widetilde { S } ^ { \circ }  { \mathcal { O } }$ be the measurable representation visible to thefittedfunction, and let $\Pi _ { O , k }$ be conditional expectation given $\sigma ( O )$ under $\rho _ { k , S }$ . Set

$$
a _ { \mathrm { a l i a s } , k } : = \left\| \Pi _ { O , k } G _ { k } - G _ { k } \right\| _ { 2 , \rho _ { k , S } } .
$$

This is the exact distancefrom the block response in (24) to the closed subspace of $\sigma ( O )$ -measurable $L ^ { 2 } ( \rho _ { k , S } )$ functions. It is $\mathcal { T } _ { k }$ -measurable and is not assumed zero. When a deterministic envelope is needed, write $a _ { \mathrm { a l i a s } , k } \le \varepsilon _ { \mathrm { a l i a s } , k }$ almost surely.

Lemma 3.15 (Aliasing is a component of the fit residual $[ \mathsf { R } ] )$ . Suppose the regression class consists of�(�)-measurable functions and the solver satisfies the regression bound relative to the best visible target, $\mathbb { E } [ \| V _ { k + 1 } - \dot { \Pi } _ { O , k } G _ { k } \| _ { 2 , \rho _ { k , S } } \ \vert$ $\mathcal { T } _ { k } ] \leq \varepsilon _ { \mathrm { f i t } , k } ^ { \mathrm { v i s } }$ . Then $\| V - G _ { k } \| _ { 2 , \rho _ { k , S } } \geq a _ { \mathrm { a l i a s } , k } f o r$ every visible $V ,$ and

$$
\mathbb { E } [ \| V _ { k + 1 } - G _ { k } \| _ { 2 , \rho _ { k , S } } \mid \mathcal { T } _ { k } ] \leq \varepsilon _ { \mathrm { f i t } , k } ^ { \mathrm { v i s } } + \varepsilon _ { \mathrm { a l i a s } , k } .
$$

Thus $\varepsilon _ { \mathrm { f i t } , k } : = \varepsilon _ { \mathrm { f i t } , k } ^ { \mathrm { v i s } } + \varepsilon _ { \mathrm { a l i a s } , k } ;$ aliasing is not charged again.

Proof. $\Pi _ { O , k }$ is the orthogonal projection onto the visible subspace, so $\| V - G _ { k } \| _ { 2 , \rho _ { k , S } } ^ { 2 } = \| V - \Pi _ { O , k } G _ { k } \| _ { 2 , \rho _ { k , S } } ^ { 2 } + a _ { \mathrm { a l i a s } , k } ^ { 2 } ,$ and the upper bound follows by the triangle inequality. □

Remark 3.16 (What removes the floor: target sufficiency of the representation [ R ]). For the fixed history, $a _ { \mathrm { a l i a s } , k } = 0$ exactly when $G _ { k } = g _ { k } \circ O \rho _ { k , S }$ -almost surelyfor some measurable $g _ { k } .$ : the observation determines the block target mean. Hidden reward, continuation, or clock variables can therefore leave a fitfloor; augmenting the representation or passing to a beliefstate can remove it.

Proposition 3.17 (Markov-sufficient and compressed-observation routes [ G, R ]). There are two compatible routesfrom an implemented input to the abstract state.

(i) Ifthe implemented input $O _ { t } ,$ , including its clock, satisfies (13) with $\tilde { s } _ { t } = O _ { t }$ , then it may be used as the analysis state. In particular,for a (SAMPLE, OBS) target, an �-measurable acting policy, and an �-measurablefrozen value, the population response is �-measurable and $a _ { \mathrm { a l i a s } , k } = 0$

(ii) $I f O = \omega ( \tilde { s } )$ is a measurable compression of a state satisfying Assumption 2.1, then every observation-only value and policy is still a measurable value and Markov policy on ${ \widetilde { s } } .$ . The residual decomposition therefore applies on the Markov analysis state with $\varepsilon _ { \mathrm { f i t } , k } = \varepsilon _ { \mathrm { f i t } , k } ^ { \mathrm { v i s } } + \varepsilon _ { \mathrm { a l i a s } , k }$ as in Lemma 3.15. An observation-only reward/transition score is lifted in the same way, and its discrepancyfrom $Q _ { V }$ is charged by $\eta _ { \mathrm { s c } , k } ,$ closure and concentrability are likewise checked on ${ \widetilde { s } } .$

A Markov-sufficient representation gives the direct convergence route, while a compressed observation gives a quantitative performance route whose representation and scorefloors remain visible in the policy-loss bound.

Proof. Part (i) is Assumption 2.1 on the represented state space. Conditional expectation of the observed-successor label through its joint kernel is then a measurable function of $\bar { O } _ { t } .$ , so its projection onto $\sigma ( O )$ is itself. For (ii), composition with � lifts every implemented value, score, and policy to a measurable object on ${ \widetilde { s } } ;$ Lemma 3.15 supplies the fit link, and the remaining links are exactly those in Assumption 3.11. □

## 3.4 Finite-horizon policy-loss bound

Theorem 3.18 (Finite-� policy-loss bound under six residuals $[ \mathsf { R } ] )$ . Assume Assumptions 2.1, 3.3 (C-strong), and $3 . I I ,$ with $K \ge 1 , V _ { 0 } , \ldots , V _ { K } \stackrel { \cdot } { \in } \mathcal { V } ^ { \mathrm { c l i p } }$ , and $V ^ { \star }$ from Lemma 2.6. Let $\pi _ { K }$ be $t r u e  – Q _ { V _ { K } }$ greedy after � target copies; it uses the true kernel and is not directly deployable $( \ S 6 . I ) .$ . Set

$$
e _ { k , p } ^ { \mathrm { B e l l } } : = \varepsilon _ { \mathrm { f i t } , k } + \varepsilon _ { \mathrm { k e r } , k } + \varepsilon _ { \mathrm { t g t } , k } + \varepsilon _ { \mathrm { b u f } , k } + \varepsilon _ { \mathrm { a c t } , k } ^ { ( p ) } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } , \qquad \overline { { e } } _ { K , H , p } ^ { \mathrm { B e l l } } : = \operatorname* { m a x } _ { \substack { \operatorname* { m a x } \{ 0 , K - H + 1 \} \leq k < K } } e _ { k , p } ^ { \mathrm { B e l l } } .\tag{48}
$$

Then

$$
\mathbb { E } [ \| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } ] \le \mathcal { B } _ { K } + \sum _ { k = 0 } ^ { K - 1 } w _ { K , k } ^ { ( H ) } e _ { k , p } ^ { \mathrm { B e l l } } \le \mathcal { B } _ { K } + 2 \phi _ { s , K } ^ { ( H ) } \overline { { e } } _ { K , H , p } ^ { \mathrm { B e l l } } .\tag{49}
$$

Proof. Lemma 3.13 at $r = p$ and the tower property give $\mathbb { E } \widetilde { \varepsilon } _ { k , p } = \mathbb { E } [ \mathbb { E } [ \widetilde { \varepsilon } _ { k , p } \mid \mathcal { T } _ { k } ] ] \leq e _ { k , p } ^ { \mathrm { B e l l } }$ . The six summands of (48) are deterministic. For the exploration link the summand is the majorant $\bar { D } _ { k } ^ { \mathrm { e x p } }$ of (21), not the random dispersion $\bar { \Delta } _ { k } ^ { \mathrm { e x p } }$ Apply Lemma 3.10 to $e _ { k } = V _ { k + 1 } - \mathcal { T } V _ { k }$ and take expectations termwise, since the weights are deterministic. The second inequality uses the active-window maximum $\bar { e } _ { K , H , p } ^ { \mathrm { B e l l } }$ and the weight-sum identity. Because the weights are deterministic, this step does not interchange E and max. □

Theorem 3.19 (Level-indexed residual propagation and sampling $[ \mathsf { R } , \mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } } ] )$ . Assume Assumption $2 . l , H \geq 2 ,$ , and $K \geq 1 ,$ ; let $\mu _ { S } ^ { \circ }$ be carried by $\{ h = H \}$ , and put $L _ { h } : = \mathcal { S } \times \{ h \}$ . For every block � and $1 \leq h < H ,$ , let $\rho _ { k , S } ^ { ( h ) }$ be a $\mathcal { T } _ { k }$ -measurable probability law carried by $L _ { h } .$ . For $1 \leq m < H$ , assume the level-indexed coefficient

$$
c _ { 2 , k } ^ { \mathrm { l e v } } ( m ) : = \operatorname* { s u p } _ { \pi _ { 1 : m } } \left\| \frac { d ( \mu _ { S } ^ { \circ } \mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { \pi _ { m } } ) } { d \rho _ { k , S } ^ { ( H - m ) } } \right\| _ { 2 , \rho _ { k , S } ^ { ( H - m ) } } ^ { 2 } \leq c _ { 2 } ^ { \mathrm { l e v } } ( m ) < \infty\tag{50}
$$

almost surely, with a deterministic upper envelope uniform in � and in the policy sequence. Define the six residual links levelwise by replacing each $L ^ { 2 } ( \rho _ { k , S } )$ norm in (24)–(26) and Assumption $3 . I I$ by $L ^ { 2 } ( \rho _ { k , S } ^ { ( h ) } )$ , retaining the same conditional-mean conventionforfit and deterministic almost-sure conventionfor the other links, and write

$$
e _ { k , h } ^ { \mathrm { B e l l } } : = \varepsilon _ { \mathrm { f i t } , k , h } + \varepsilon _ { \mathrm { k e r } , k , h } + \varepsilon _ { \mathrm { t g t } , k , h } + \varepsilon _ { \mathrm { b u f } , k , h } + \varepsilon _ { \mathrm { a c t } , k , h } + \varepsilon _ { k } { \bar { D } } _ { k , h } ^ { \mathrm { e x p } } .\tag{51}
$$

Then every clipped abstract recursion and its true-score greedy policy satisfy

$$
\mathbb { E } \Vert V ^ { \star } - V ^ { \pi _ { K } } \Vert _ { 1 , \mu _ { S } } \leq \mathcal { B } _ { K } + 2 \sum _ { k = 0 } ^ { K - 1 } \sum _ { m = K - k } ^ { H - 1 } \gamma ^ { m } \sqrt { c _ { 2 } ^ { \mathrm { l e v } } ( m ) } e _ { k , H - m } ^ { \mathrm { B e l l } }
$$

$$
\leq \mathcal { B } _ { K } + 2 \phi _ { \mathrm { l e v } } ^ { ( H ) } \underset { 1 \leq h < H } { \operatorname* { m a x } } e _ { k , h } ^ { \mathrm { B e l l } } , \qquad \phi _ { \mathrm { l e v } } ^ { ( H ) } : = \sum _ { m = 1 } ^ { H - 1 } m \gamma ^ { m } \sqrt { c _ { 2 } ^ { \mathrm { l e v } } ( m ) } .\tag{52}
$$

Empty inner sums are zero. In particular, $i f c _ { 2 } ^ { \mathrm { l e v } } ( m ) \leq \bar { c } ,$ then $\phi _ { \mathrm { l e v } } ^ { ( H ) } \le \sqrt { \bar { c } } H ( H - 1 ) / 2 = O ( H ^ { 2 } \sqrt { \bar { c } } ) .$

Under the $\mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } }$ clause ofAssumption 3.4, a realization in a class closed under slice assembly draws $n _ { k , h }$ conditionally i.i.d. states from $\rho _ { k , S } ^ { ( h ) }$ , executes the level-ℎ behavior law, observesfresh true-kernel outcomes, andfits $V _ { k + 1 } | _ { L _ { h } }$ with a separate head or regressor. Thus only the fit slot at level ℎ is learnedfrom those $n _ { k , h }$ labels; the otherfive slots in (51) are assumed or separately bounded under the same $\rho _ { k , S } ^ { ( h ) }$ . The exact sample count is

$$
N _ { 0 : K - 1 } ^ { \mathrm { l e v } } : = \sum _ { k = 0 } ^ { K - 1 } \sum _ { h = 1 } ^ { H - 1 } n _ { k , h } ; \qquad n _ { k , h } \equiv n \implies N _ { 0 : K - 1 } ^ { \mathrm { l e v } } = K ( H - 1 ) n .\tag{53}
$$

This counts direct slice-reset outcomes. Rejection samplingfrom a single reset law instead has an additional clock-massdependent cost. An optional top-slicefit adds $\scriptstyle \sum _ { k < K } n _ { k , H }$ samples while leaving the propagation bound unchanged. A shared-parameter multitask implementation can use the same propagation formula once its joint fit residual is controlled.

Proof. In the proof of Lemma 3.10, every depth-� occupancy lies on $L _ { H - m }$ ; Cauchy–Schwarz with (50) bounds its residual integral by $\sqrt { c _ { 2 } ^ { \mathrm { l e v } } ( m ) } \lVert e _ { k } \rVert _ { 2 , \rho _ { k , S } ^ { ( H - m ) } }$ . The two resolvent branches and the exact count min $\{ K , m \}$ give (52); summing disjoint direct-reset blocks gives (53). □

## 3.5 Matched-budget propagation

A shared-reset run uses one sampling distribution across horizon levels, whereas a direct-level run samples each level separately. We compare their policy-loss bounds under a common label budget for the updates that receive nonzero propagation weights. Coordinate � has a statistical term $b _ { i } n _ { i } ^ { - \nu }$ ; the exponent � covers parametric, tabular, and nonparametric rates.

Theorem 3.20 (Matched-budget propagation and optimal planned allocation $\left[ \mathsf { A } _ { \mathrm { i i d } } , \mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } } \right] )$ . Fix a terminal copy � and $\nu \in ( 0 , 1 ]$ , and apply Theorem 3.18 at $s = 2$ to a shared-reset run and Theorem $3 . I 9$ to a direct-level-reset run on the same model, with the same evaluation law and initialization. Denote their terminal true-score greedy policies by $\pi _ { K } ^ { \mathrm { s h } }$ and $\pi _ { K } ^ { \mathrm { { l e v } } }$ , respectively, and define the active index sets

$$
\begin{array} { r l } & { \mathcal { K } _ { K } : = \big \{ k : \operatorname* { m a x } \{ 0 , K - H + 1 \} \leq k < K \big \} , } \\ & { \mathcal { T } _ { K } : = \big \{ ( k , h ) : 0 \leq k < K , ~ 1 \leq h < H , ~ K - k \leq H - h \big \} , } \end{array}\tag{54}
$$

and, $f o r \left( k , h \right) \in \mathcal { T } _ { K }$ , put

$$
A _ { k , h } ^ { \mathrm { l e v } } : = 2 \gamma ^ { H - h } \sqrt { c _ { 2 } ^ { \mathrm { l e v } } ( H - h ) } .\tag{55}
$$

Suppose the Bellman-residual envelopes separate into deterministic floors and statistical terms,

$$
e _ { k , 2 } ^ { \mathrm { B e l l } } \leq r _ { k } ^ { \mathrm { s h } } + b _ { k } ^ { \mathrm { s h } } n _ { k } ^ { - \nu } , \qquad e _ { k , h } ^ { \mathrm { B e l l } } \leq r _ { k , h } ^ { \mathrm { l e v } } + b _ { k , h } ^ { \mathrm { l e v } } n _ { k , h } ^ { - \nu } ,\tag{56}
$$

on their respective active sets. The floors may contain any label-independent component of the six links. The coefficients $b _ { i }$ arefixed before the allocation is chosen and, in particular, may not absorb afactor such as $( \log n _ { i } ) ^ { q }$ that varies with the coordinate allocation. Omit coordinates whose statistical objective coefficient $c _ { i }$ is zero and define

$$
\mathsf { C } _ { \nu , K } ^ { \mathrm { s h } } : = \left\{ \sum _ { k \in \mathcal { K } _ { K } } \left( w _ { K , k } ^ { ( H ) } b _ { k } ^ { \mathrm { s h } } \right) ^ { 1 / ( 1 + \nu ) } \right\} ^ { 1 + \nu } ,
$$

$$
F _ { K } ^ { \mathrm { s h } } : = \sum _ { k \in \mathcal { K } _ { K } } w _ { K , k } ^ { ( H ) } r _ { k } ^ { \mathrm { s h } } ,\tag{57}
$$

$$
\mathsf { C } _ { \nu , K } ^ { \mathrm { l e v } } : = \left. \sum _ { ( k , h ) \in \mathbb { Z } _ { K } } \left( A _ { k , h } ^ { \mathrm { l e v } } b _ { k , h } ^ { \mathrm { l e v } } \right) ^ { 1 / ( 1 + \nu ) } \right. ^ { 1 + \nu } ,
$$

$$
F _ { K } ^ { \mathrm { l e v } } : = \sum _ { ( k , h ) \in \mathbb { Z } _ { K } } A _ { k , h } ^ { \mathrm { l e v } } r _ { k , h } ^ { \mathrm { l e v } } .\tag{58}
$$

Under continuous terminal active-window budgets $\textstyle \sum _ { k \in \mathcal { K } _ { K } } n _ { k } = \mathsf { N } _ { \mathrm { s h } }$ and $\sum _ { ( k , h ) \in \mathcal { T } _ { K } } n _ { k , h } = \mathsf { N } _ { \mathrm { l e v } }$ , the unique optimal allocations on positive-weight coordinates are

$$
n _ { i } = \mathsf { N } \frac { c _ { i } ^ { 1 / ( 1 + \nu ) } } { \sum _ { j } c _ { j } ^ { 1 / ( 1 + \nu ) } } , \qquad \operatorname* { m i n } _ { \substack { n _ { i } > 0 : \sum _ { i } n _ { i } = \mathsf { N } } } \sum _ { i } c _ { i } n _ { i } ^ { - \nu } = \frac { \left( \sum _ { i } c _ { i } ^ { 1 / ( 1 + \nu ) } \right) ^ { 1 + \nu } } { \mathsf { N } ^ { \nu } } ,\tag{59}
$$

with $c _ { i } = w _ { K , k } ^ { ( H ) } b _ { k } ^ { \mathrm { s h } }$ for shared reset and $c _ { i } = A _ { k , h } ^ { \mathrm { l e v } } b _ { k , h } ^ { \mathrm { l e v } }$ for direct level reset. Both terminal-window budgets count observed true-kernel labels and charge only the displayed active coordinates. Labels consumed outside these sets, including earlier zero-weight blocks when $K > H - 1$ , must be added separately when reporting total run cost. The quantity $\mathsf { N } _ { \mathrm { l e v } }$ counts direct slice-reset outcomes, with any rejection overhead added before comparing physical simulator calls. The resulting terminal-policy bounds are

$$
\begin{array} { r } { \mathbb { E } \| V ^ { \star } - V ^ { \pi _ { K } ^ { \mathrm { s h } } } \| _ { 1 , \mu _ { S } } \leq { \mathcal { B } } _ { K } + F _ { K } ^ { \mathrm { s h } } + { \mathsf { C } } _ { \nu , K } ^ { \mathrm { s h } } \mathsf { N } _ { \mathrm { s h } } ^ { - \nu } , } \end{array}\tag{60}
$$

$$
\begin{array} { r } { \mathbb { E } \| V ^ { \star } - V ^ { \pi _ { K } ^ { \mathrm { l e v } } } \| _ { 1 , \mu _ { S } } \leq \mathcal { B } _ { K } + F _ { K } ^ { \mathrm { l e v } } + \mathsf { C } _ { \nu , K } ^ { \mathrm { l e v } } \mathsf { N } _ { \mathrm { l e v } } ^ { - \nu } . } \end{array}\tag{61}
$$

If the rate on coordinate � is valid only for $n _ { i } \geq L _ { i } ,$ , let $L _ { i } \in \mathbb { N }$ with $\begin{array} { r } { L _ { i } \geq 1 , \mathsf { N } \geq \sum _ { i } L _ { i } , } \end{array}$ and define

$$
\Psi _ { \nu } ( c , L , \mathsf { N } ) : = \operatorname* { m i n } _ { \substack { n _ { i } \geq L _ { i } : \sum _ { i } n _ { i } = \mathsf { N } } } \sum _ { i } c _ { i } n _ { i } ^ { - \nu } .\tag{62}
$$

For $\Nu > \textstyle \sum _ { i } L _ { i }$ , its unique continuous minimizer is

$$
n _ { i } ^ { L } = \operatorname* { m a x } \left\{ L _ { i } , \left( \frac { \nu c _ { i } } { \lambda } \right) ^ { 1 / ( 1 + \nu ) } \right\} , \qquad \sum _ { i } n _ { i } ^ { L } = \mathsf { N } ,\tag{63}
$$

where the budget equation determines a unique $\begin{array} { r } { \lambda > 0 ; i f { \sf N } = \sum _ { i } L _ { i } } \end{array}$ , the uniquefeasible allocation is $n _ { i } ^ { L } = L _ { i }$ . When lower bounds apply, replace the last statistical term in $( 6 0 ) o r \ : ( 6 1 )$ by the corresponding value $\Psi _ { \nu } .$ . Formula (59) and the closedforms (60)– (61) remain valid exactly when every unconstrained proportional allocation satisfies its lower bound.

There is also an implementable integer schedule. Starting from $\lfloor n _ { i } ^ { L } \rfloor$ , distribute the remaining $\begin{array} { r } { \mathsf { N } - \sum _ { i } \lfloor n _ { i } ^ { L } \rfloor } \end{array}$ labels one at a time to a coordinate with largest current marginal decrease

$$
\Delta _ { i } ( n _ { i } ) : = c _ { i } \{ n _ { i } ^ { - \nu } - ( n _ { i } + 1 ) ^ { - \nu } \} .\tag{64}
$$

breaking ties by a fixed index order. The result is feasible, uses the full budget, and has statistical objective at most $2 ^ { \nu } \Psi _ { \nu } ( c , \bar { L } , \mathsf { N } )$ . Here $| { \cal K } _ { K } | = \mathrm { m i n } \{ { K , H - 1 } \}$ and, with $J = \mathrm { m i n } \{ K , H - 1 \} , | \mathcal { Z } _ { K } | = J H - J ( J + 1 ) / 2$ before zero-weight coordinates are omitted.

## Proof: See Appendix B.

Corollary 3.21 (Matched-budget horizon geometry on the clock witness $[ \mathsf { A } _ { \mathrm { i i d } } , \mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } } ] ) .$ . Consider the one-state-per-level horizon-indexedfamily with unit survival, $\mathsf { \bar { \boldsymbol { K } } } _ { H } \ge \mathsf { \bar { \boldsymbol { H } } } - 1$ , and $s = 2 .$ . Usefresh (SAMPLE, OBS) outcomes, the declared exact-score oracle, $\begin{array} { r } { \dot { W } _ { k } = V _ { k } , } \end{array}$ , greedy collection with $\varepsilon _ { k } = 0 , { \cal O } = { \cal S } ,$ and a fitted procedure with the displayed statistical envelope. Assume the unconstrained planned allocations satisfy every validity threshold $\boldsymbol { L } _ { i } ;$ otherwise the exact comparison is the constrained value $\Psi _ { \nu }$ from (62). Under these choices, the kernel, target, replay, action, exploration, aliasing, and drift links vanish, the top-slice boundary is zero, and the fit link is the only nonzero term. Suppose its statistical constants satisfy $b _ { k } ^ { \mathrm { s h } } \asymp b _ { H } ^ { \mathrm { s h } }$ and $b _ { k , h } ^ { \mathrm { l e v } } \asymp b _ { H } ^ { \mathrm { l e v } }$ uniformly on the active sets. For direct level reset take $c _ { 2 } ^ { \mathrm { l e v } } ( m ) = 1$ . For shared reset compare the uniform clock law with the coefficient-optimal law (35). Then:

(a) In the near-unit-discount regime,

$$
\mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { s h } , \mathrm { u n i f } } \asymp \mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { s h } , \mathrm { o p t } } \asymp b _ { H } ^ { \mathrm { s h } } H ^ { \nu + 5 / 2 } , \qquad \mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { l e v } } \asymp b _ { H } ^ { \mathrm { l e v } } H ^ { 2 \nu + 2 } .\tag{65}
$$

For the root-� case $\nu = 1 / 2$ , both clock geometries therefore contribute $H ^ { 3 } / \sqrt { \mathsf { N } }$ before their regression constants are inserted.

(b) In the fixed-discount regime,

$$
\mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { s h } , \mathrm { u n i f } } \asymp b _ { H } ^ { \mathrm { s h } } H ^ { 1 / 2 } , \qquad \mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { s h } , \mathrm { o p t } } \asymp b _ { H } ^ { \mathrm { s h } } , \qquad \mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { l e v } } \asymp b _ { H } ^ { \mathrm { l e v } } .\tag{66}
$$

All constants are uniform in �. Direct slice access and an optimized shared clock law have the samefixed-discount budget order, while direct access achieves it without tuning cross-slice clock masses. In the one-state-per-level tabular specialization, thefit bound (98) gives $b _ { H } ^ { \mathrm { s h } } \asymp V _ { \operatorname* { m a x } } \sqrt { H }$ and $b _ { H } ^ { \mathrm { l e v } } \asymp V _ { \mathrm { m a x } }$ . At $\nu = 1 / 2 ,$ , the near-unit statistical terms are therefore respectively

$$
\Theta \left( \frac { V _ { \mathrm { m a x } } H ^ { 7 / 2 } } { \sqrt { \mathsf { N } _ { \mathrm { s h } } } } \right) \quad a n d \quad \Theta \left( \frac { V _ { \mathrm { m a x } } H ^ { 3 } } { \sqrt { \mathsf { N } _ { \mathrm { l e v } } } } \right) ,\tag{67}
$$

<sub>for</sub> <sub>the</sub> <sub>displayed</sub> <sub>bounds.</sub> <sub>This</sub> <sub>is</sub> <sub>a</sub> <sub>factor-</sub>√<sub>�</sub> <sub>statistical</sub> <sub>advantage</sub> <sub>for</sub> <sub>direct</sub> <sub>slice</sub> <sub>reset</sub> <sub>under</sub> <sub>equal</sub> <sub>sufficiently</sub> <sub>large</sub> terminal-window label budgets.

## Proof: See Appendix B.

Remark 3.22 (Routed neural specialization $\big [ \mathsf { A } _ { \mathrm { i i d } } , \mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } } \big ] \big )$ . For a fixed horizon, the neural rate of Proposition 5.5 can be compared under a common terminal-window budget while retaining its actual validity thresholds $L _ { i }$ and allocation-dependent logarithms. One may either upper-bound each $( \log n _ { i } ) ^ { q }$ by the allocation-independent $( \log \mathsf { N } ) ^ { q }$ before applying Theorem 3.20, or optimize the logarithmic objective directly. The regression constant may depend on � through $\bar { V } _ { \mathrm { m a x } }$ and the routed approximation and entropy constants. Corollary 3.21 therefore establishes the factor- � comparison for its displayed tabular constants; an analogous growing-horizon neural comparison requires uniform-in-� approximation, entropy, and reward-normalization bounds.

## 4 Online-action residual

The behavior policy may score actions with a network that differs from the frozen iterate $V _ { k }$ . Using the optimization and acting-snapshot index sets defined before (21), measure this displacement by

$$
\delta _ { V , k } ^ { \mathrm { p a t h } } : = \operatorname* { s u p } _ { t \in \mathcal { T } _ { k } ^ { \mathrm { a c t } } } \| V _ { \theta _ { k , t } } - V _ { k } \| _ { \infty } , \delta _ { V , k } ^ { \mathrm { p a t h } } \leq \delta _ { V , k } \quad \mathrm { a . s . \ f o r ~ \mathsf { A } _ { \mathrm { r u n } } , }\tag{68}
$$

$$
\begin{array} { r } { \delta _ { V , k } ^ { \mathrm { s n a p } } : = \| W _ { k } - V _ { k } \| _ { \infty } , \qquad \delta _ { V , k } ^ { \mathrm { s n a p } } \leq \delta _ { V , k } \quad \mathrm { a . s . ~ f o r ~ A _ { i i d } . } } \end{array}
$$

The deterministic envelope $\delta _ { V , k }$ bounds the appropriate displacement for each analysis object. Together with the score error $\eta _ { \mathrm { s c } , k }$ , it gives the perturbation scale

$$
\Lambda _ { k } : = \gamma \delta _ { V , k } + \eta _ { \mathrm { s c } , k } , \qquad \eta _ { \mathrm { s c } , k } \mathrm { a s } \mathrm { b o u n d e d } \mathrm { i n } ( 2 1 ) .\tag{69}
$$

Network drift and score error therefore enter on the same scale. A score perturbation of size $\Lambda _ { k }$ can change the greedy action only at states whose action gap is at most $2 \Lambda _ { k }$ . The margin condition controls how much replay probability lies in this set. We use a Mammen–Tsybakov-type condition on the frozen iterates’ action gaps and relate it to classical optimal-gap conditions [12–15].

## 4.1 Margin condition

Definition 4.1 (Frozen-iterate action-gap margin: local and global forms [ R ]). The replay laws satisfy the local frozen-iterate $( C _ { \mathrm { m a r g } } , \alpha )$ margin condition at scale $[ u _ { 0 } , \bar { u } ]$ ifthere are deterministic constants $C _ { \mathrm { m a r g } } < \infty a n d \alpha \geq 0$ such that, almost surely andfor every �,

$$
\begin{array} { r } { \rho _ { k , S } \big \{ \tilde { s } : \Delta _ { Q } ^ { ( k ) } ( \tilde { s } ) \leq u \big \} \leq C _ { \mathrm { m a r g } } u ^ { \alpha } \qquad f o r a l l u \in [ u _ { 0 } , \bar { u } ] , } \end{array}\tag{70}
$$

and the global frozen-iterate condition if the same holds for all $u > 0$ . This is a condition on the random pair $\left( Q _ { V _ { k } } , \rho _ { k , S } \right)$ , uniform over iterations with deterministic constants; it is distinctfrom a margin stated onlyfor thefixed optimal score $Q _ { V ^ { \star } }$ . The always-validfallback is $( C _ { \mathrm { m a r g } } , \alpha ) = ( 1 , 0 )$ . Each result invokes the condition only at its displayed scale $( 2 \Lambda _ { k }$ or $2 \eta _ { \mathrm { s c } , K } ) ;$ a vanishing-scale claim requires a common interval (0, �¯].

Remark 4.2 (Ties and a positive frozen-iterate exponent [ R ]). $H ( 7 0 )$ with $\alpha > 0$ holds down to zero, then $\rho _ { k , S } \{ \Delta _ { Q } ^ { ( k ) } =$ $0 \} = 0 ;$ on afinitefull-support law thisforbids ties at everyfrozen iterate. A condition imposed at one scale has no such condition. By contrast, afixed-�<sup>⋆</sup> positive-gap condition can exclude the zero-gap event and retain a separate optimal-tie mass, as follows.

Proposition 4.3 (Transfer from a fixed optimal gap to frozen iterates $[ \mathsf { G } , \mathsf { R } ] )$ . For each �, suppose a deterministic $\delta _ { k } ^ { \star } \geq 0$ satisfies

$$
\| Q _ { V _ { k } } - Q _ { V ^ { \star } } \| _ { \infty } \leq \delta _ { k } ^ { \star } \qquad a l m o s t s u r e l y .\tag{71}
$$

Suppose also that deterministic $\tau _ { \star } \in [ 0 , 1 ] , C _ { \star } < \infty ,$ , and $\alpha _ { \star } \geq 0$ satisfy, almost surely and uniformly in $k ,$

$$
\rho _ { k , S } \{ \Delta _ { Q } ^ { \star } = 0 \} \le \tau _ { \star } , \qquad \rho _ { k , S } \{ 0 < \Delta _ { Q } ^ { \star } \le v \} \le C _ { \star } v ^ { \alpha _ { \star } }\tag{72}
$$

at every scale � in a declared interval. Whenever $u + 2 \delta _ { k } ^ { \star }$ belongs to that interval,

$$
\rho _ { k , S } \{ \Delta _ { Q } ^ { ( k ) } \leq u \} \leq \tau _ { \star } + C _ { \star } ( u + 2 \delta _ { k } ^ { \star } ) ^ { \alpha _ { \star } } .\tag{73}
$$

In particular, $i f \tau _ { \star } = 0$ , the optimal-gap condition is available on the required enlarged scales, and $\delta _ { k } ^ { \star } \le c u _ { 0 }$ uniformly for some $c \geq 0 _ { ; }$ , then the localfrozen-iterate condition on $[ u _ { 0 } , \bar { u } ]$ holds with exponent $\alpha _ { \star }$ and constant $C _ { \star } ( 1 + 2 c ) ^ { \alpha _ { \star } }$ At the action scale $u = 2 \Lambda _ { k }$ , the weaker tube $\delta _ { k } ^ { \star } \le c \Lambda _ { k }$ already preserves the exponent because the right-hand side of (73) becomes ${ \cal C } _ { \star } [ 2 ( 1 + c ) \Lambda _ { k } ] ^ { \alpha , }$ when $\tau _ { \star } = 0 .$ . Finally,

$$
\delta _ { k } ^ { \star } = \gamma b _ { k } \quad i s \nu a l i d w h e n e \nu e r \quad \| V _ { k } - V ^ { \star } \| _ { \infty } \leq b _ { k } ,\tag{74}
$$

so the transfer can be verifiedfrom a deterministic value-iterate tube.

Proof. Write $q = Q _ { V _ { k } }$ and $q ^ { \star } = Q _ { V ^ { \star } } . \mathrm { I f } \Delta _ { O } ^ { \star } ( \tilde { s } ) \leq 2 \delta _ { k } ^ { \star }$ , then it already lies below $u + 2 \delta _ { k } ^ { \star }$ . If instead $\Delta _ { O } ^ { \star } ( \tilde { s } ) > 2 \delta _ { k } ^ { \star }$ the optimal maximizer is unique and remains the maximizer of �, while its separation from every competitor is at least $\Delta _ { Q } ^ { \star } \big ( \bar { \tilde { s } } \big ) - 2 \delta _ { k } ^ { \star }$ . Hence

$$
\{ \Delta _ { Q } ^ { ( k ) } \leq u \} \subseteq \{ \Delta _ { Q } ^ { \star } \leq u + 2 \delta _ { k } ^ { \star } \} .
$$

Splitting the latter event into zero and positive gaps gives (73). If $\delta _ { k } ^ { \star } \le c u _ { 0 } \le c u$ , then $u + 2 \delta _ { k } ^ { \star } \le ( 1 + 2 c ) u$ , proving the local statement. The action-scale claim is the same calculation with $u = 2 \Lambda _ { k }$ . Finally, $\| Q _ { V } \stackrel { \sim } { - } Q _ { W } \| _ { \infty } \le \gamma \| \dot { V } - \bar { W } \| _ { \infty }$ proves (74). □

Lemma 4.4 (Exact gap identity and pointwise control [ R ]). In the notation of §4.1:

(i) $\| Q _ { V _ { \theta _ { k } } } - Q _ { V _ { k } } \| _ { \infty } \leq \gamma \delta _ { V , k }$ , and hence $\| \widehat { Q } _ { V _ { \theta _ { k } } } - Q _ { V _ { k } } \| _ { \infty } \leq \Lambda _ { k }$ with $\Lambda _ { k }$ as in (69);

(ii) $i f \pi _ { k } ^ { \mathrm { o n } } = ( 1 - \varepsilon _ { k } ) \delta _ { a _ { k } ^ { \mathrm { o n } } } + \varepsilon _ { k } \xi _ { k }$ is a single �<sub>�</sub>-greedy policy, then

$$
\left( \mathcal T ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } - \mathcal T ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \right) ( \tilde { s } ) = ( 1 - \varepsilon _ { k } ) \bigl ( Q _ { V _ { k } } ( \tilde { s } , a _ { k } ^ { \mathrm { o n } } ) - Q _ { V _ { k } } ( \tilde { s } , a _ { k } ^ { \mathrm { t g t } } ) \bigr ) .
$$

Its absolute value is $( 1 - \varepsilon _ { k } ) \mathrm { s u b o p t } _ { k } ( \tilde { s } )$ ; here subop $\bar { \mathbf { \zeta } } _ { k } : = Q _ { V _ { k } } ( \cdot , a _ { k } ^ { \mathrm { t g t } } ) - Q _ { V _ { k } } ( \cdot , a _ { k } ^ { \mathrm { o n } } )$ ;

(iii) (generic perturbation) let $q : \widetilde { S } ^ { \circ } \times \mathcal { A }  \mathbb { R }$ be any score with $\| q - Q _ { V _ { k } } \| _ { \infty } \leq \Lambda$ for some $\Lambda \geq 0 ,$ , and let $a ^ { q } ( \tilde { s } ) \in \arg \operatorname* { m a x } _ { a } q ( \tilde { s } , a )$ . Then the induced suboptimality subop $\mathrm { t } _ { k } ^ { q } : = { Q _ { V _ { k } } } ( \cdot , a _ { k } ^ { \mathrm { t g t } } ) - { Q _ { V _ { k } } } ( \cdot , a ^ { q } )$ satisfies

$$
0 \leq \mathrm { s u b o p t } _ { k } ^ { q } ( \tilde { s } ) \leq 2 \Lambda \qquad a n d \qquad \{ \mathrm { s u b o p t } _ { k } ^ { q } > 0 \} \subseteq \{ \Delta _ { Q } ^ { ( k ) } \leq 2 \Lambda \} .
$$

Applied to the implemented score $q = \widehat { Q } _ { V _ { \theta _ { k } } }$ with $\Lambda = \Lambda _ { k } ,$ this gives $0 \leq \mathrm { s u b o p t } _ { k } \leq 2 \Lambda _ { k }$ and $\{ \mathrm { s u b o p t } _ { k } > 0 \} \subseteq$ ${ \{ \Delta _ { Q } ^ { ( k ) } \leq 2 \Lambda _ { k } \} }$ ; in the idealization $\eta _ { \mathrm { s c } , k } = 0 \ i$ t gives the same statements with $\Lambda _ { k } = \gamma \delta _ { V , k }$

Proof. $\begin{array} { r } { Q _ { V } - Q _ { W } = \gamma \int ( V - W ) d \mathcal { P } } \end{array}$ proves (i). In (ii), both policies share $\varepsilon _ { k } \xi _ { k } ,$ , so exploration cancels exactly; its separate difference from � remains the $\varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } }$ link. For (iii), apply the perturbation bound twice: $Q _ { V _ { k } } ( \tilde { s } , a ^ { q } ) \ge$ $q ( \tilde { s } , a ^ { q } ) - \Lambda \ge q ( \tilde { s } , a _ { k } ^ { \mathrm { t g t } } ) - \Lambda \ge Q _ { V _ { k } } ( \tilde { s } , a _ { k } ^ { \mathrm { t g t } } ) - 2 \Lambda$ , so subop $^ q _ { k } \le 2 \Lambda$ . If $\Delta _ { Q } ^ { ( k ) } ( \tilde { s } ) > 2 \Lambda$ then every $a \neq a _ { k } ^ { \mathrm { t g t } }$ has $q ( \tilde { s } , a ) \leq Q _ { V _ { k } } ( \tilde { s } , a ) + \Lambda < \operatorname* { m a x } _ { a } Q _ { V _ { k } } ( \tilde { s } , a ) - \Lambda \leq q ( \tilde { s } , a _ { k } ^ { \mathrm { t g t } } ) , \mathrm { s o } a ^ { q } = a _ { k } ^ { \mathrm { t g t } }$ □

For every $V \in \mathcal { V } ^ { \mathrm { c l i p } }$ and every Markov policy �, the elementary pointwise inequality $\mathcal T ^ { \pi } V \leq \mathcal T V$ follows by averaging action scores below their maximum.

## 4.2 Gap identity and upper bound

Theorem 4.5 (Norm-indexed action residual for the abstract recursion [ R ]). In the notation of ${ \mathfrak { s } } 4 . l ,$ let � be measurable with $\| q _ { k } - Q _ { V _ { k } } \| _ { \infty } \leq \Lambda _ { k } ,$ let $a ^ { q _ { k } }$ maximize $q _ { k }$ using thefixed tie rule, and put $\pi _ { k } ^ { q } = ( 1 - \stackrel { \cdot } { \varepsilon } _ { k } ) \delta _ { a ^ { q _ { k } } } + \varepsilon _ { k } \xi _ { k }$ . For every $r \in [ 1 , 2 ] ,$

$$
\big \| \mathcal T ^ { \pi _ { k } ^ { q } } V _ { k } - \mathcal T ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \big \| _ { r , \rho _ { k , S } } \leq ( 1 - \varepsilon _ { k } ) 2 \Lambda _ { k } .\tag{75}
$$

Under the global frozen-iterate margin (70), or its local form when $2 \Lambda _ { k } \in [ u _ { 0 } , \bar { u } ]$

$$
\begin{array} { r } { \big \| \mathcal { T } ^ { \pi _ { k } ^ { q } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \big \| _ { r , \rho _ { k , S } } \leq ( 1 - \varepsilon _ { k } ) C _ { \operatorname* { m a r g } } ^ { 1 / r } ( 2 \Lambda _ { k } ) ^ { 1 + \alpha / r } . } \end{array}\tag{76}
$$

Alternatively, under Proposition $4 . 3$ at $u = 2 \Lambda _ { k }$

$$
\big \| \mathcal { T } ^ { \pi _ { k } ^ { q } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \big \| _ { r , \rho _ { k } , s } \leq ( 1 - \varepsilon _ { k } ) 2 \Lambda _ { k } \Big \{ \tau _ { \star } + C _ { \star } ( 2 \Lambda _ { k } + 2 \delta _ { k } ^ { \star } ) ^ { \alpha _ { \star } } \Big \} ^ { 1 / r } .\tag{77}
$$

Thus a zero optimal-tie mass and $\delta _ { k } ^ { \star } \le c \Lambda _ { k }$ yield the same $\Lambda _ { k } ^ { 1 + \alpha _ { \star } / r }$ exponent directlyfrom $a f i x e d – Q ^ { \star }$ margin, with its explicit constant. Consequently one may choose

$$
\varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) } \leq ( 1 - \varepsilon _ { k } ) \operatorname* { m i n } \{ 2 \Lambda _ { k } , C _ { \mathrm { m a r g } } ^ { 1 / r } ( 2 \Lambda _ { k } ) ^ { 1 + \alpha / r } \} .
$$

When the reference-gap route is used, one may instead choose

$$
\varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) } \leq ( 1 - \varepsilon _ { k } ) \operatorname* { m i n } \biggl \{ 2 \Lambda _ { k } , 2 \Lambda _ { k } \Bigl [ \tau _ { \star } + C _ { \star } ( 2 \Lambda _ { k } + 2 \delta _ { k } ^ { \star } ) ^ { \alpha _ { \star } } \Bigr ] ^ { 1 / r } \biggr \} .
$$

Taking $r = p = s / ( s - 1 )$ supplies the propagation envelope $\varepsilon _ { \mathrm { a c t } , k } ^ { ( p ) }$ with exponent $1 + \alpha ( 1 - 1 / s )$ ; taking $r = 2$ supplies the envelope $\varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) }$ used in the regression bound, with exponent $1 + \alpha / 2$ . The two coincide at $s = 2$

Proof. Lemma 4.4(ii)–(iii) makes the integrand $( 1 \mathrm { ~ - ~ } \varepsilon _ { k } ) \mathrm { s u b o p t } _ { k } ^ { q _ { k } }$ , bounded by $( 1 - \varepsilon _ { k } ) 2 \Lambda _ { k }$ and supported, when nonzero, on $\{ \Delta _ { Q } ^ { ( k ) } ~ \le ~ 2 \Lambda _ { k } \}$ . This proves (75); under the frozen-iterate margin its �th power is at most $( 1 - \varepsilon _ { k } ) ^ { r } C _ { \mathrm { m a r g } } ( 2 \Lambda _ { k } ) ^ { r + \dot { \alpha } }$ , proving (76). Replacing the support probability by (73) proves (77). □

Corollary 4.6 (Abstract-recursion mixtures and drift control $\left[ \mathsf { A } _ { \mathrm { r u n } } , \mathsf { A } _ { \mathrm { i i d } } \right] )$ . The unconditional bound (75) remains validfor any state-dependent convex mixture ofpolicies sharing $\left( \varepsilon _ { k } , \xi _ { k } \right)$ whose greedy scores are within $\Lambda _ { k } o f Q _ { V _ { k } }$ Under the global frozen-iterate margin, or under its local form with $2 \Lambda _ { k } \in [ u _ { 0 } , \bar { u } ]$ , the margin-improved bound (76) also remains valid: convexity preserves both the pointwise $2 \Lambda _ { k }$ bound and its common gap-supported set. The same argument preserves the fixed-�<sup>⋆</sup> transfer bound (77). Thus Theorem 4.5 applies directly to the snapshot law $\beta _ { k }$ of $\mathsf { A } _ { \mathrm { i i d } }$ with $q _ { k } = \widehat { Q } _ { W _ { k } }$ , while this mixture statement applies to $\bar { \pi } _ { k } ^ { \mathrm { o n } }$ of ${ \sf A } _ { \mathrm { r u n } }$ with the corresponding online scores; their distancefrom $Q _ { V _ { k } }$ is at most $\Lambda _ { k } = \gamma \delta _ { V , k } + \eta _ { \mathrm { s c } , k } b y ( 6 9 )$ . Moreover, $i f V _ { \theta _ { k , 0 } } = V _ { k } , \theta \mapsto$ � is �-Lipschitz in supremum norm, and $\lVert \theta _ { t + 1 } - \theta _ { t } \rVert \leq \lambda _ { t } G$ , then $\begin{array} { r } { \delta _ { V , k } ^ { \mathrm { p a t h } } \leq L G \sum _ { t \in \mathcal { T } _ { k } ^ { \mathrm { o p t } } } \lambda _ { t } } \end{array}$ . For a general initialization, add $\| V _ { \theta _ { k , 0 } } - V _ { k } \| _ { \infty }$ . Here � is an explicit regularity assumption on the parameterization.

Combining Lemma 3.10 and Theorem 4.5 gives the continuous coverage–margin tradeoff: the propagated action power is $1 + \alpha ( 1 - 1 / s )$ and the exact finite-� clock multiplier is $2 \phi _ { s , K } ^ { ( H ) }$ . This single result contains the $L ^ { 2 }$ and $L ^ { \infty }$ statements as endpoint cases. The fixed- $\cdot Q ^ { \star }$ route has the same power with $\alpha = \alpha .$ <sub>⋆</sub> whenever $\tau _ { \star } = 0$ and $\delta _ { k } ^ { \star } = { \cal O } ( \Lambda _ { k } )$ on the active window.

## 4.3 One-step sharpness and lower bound

Proposition 4.7 attains the action theorem’s one-step exponent. Proposition 4.8 produces a nonzero limiting loss from a reward-score perturbation while all other residuals vanish; the harmful behavior is generated by the implemented score.

Proposition 4.7 (Componentwise sharpness of the abstract one-step residual [ R ]). For every $\alpha > 0 , p \in [ 1 , 2 ]$ $\gamma \in ( 0 , 1 )$ , and sufficiently small drift $\delta > 0$ , there exist a deterministic time-augmented MDP with an exact one-step model (so that $\eta _ { \mathrm { s c } , k } = 0$ and $\Lambda _ { k } = \gamma \delta )$ , a frozen $V _ { k } ,$ , an online $V _ { \theta _ { k } }$ with $\delta _ { V , k } = \delta _ { \mathrm { { \scriptsize ~ \cdot ~ } } }$ , and a replay marginal $\rho _ { k , S }$ that satisfies (70) with equalityfor $u \in ( 0 , \gamma c _ { \Delta } ]$ , such that, when $\varepsilon _ { k } = 0 ;$

$$
\big \| \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } - \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \big \| _ { p , \rho _ { k , S } } = \bigg ( \frac { \alpha } { \alpha + p } \bigg ) ^ { 1 / p } C _ { \mathrm { m a r g } } ^ { 1 / p } ( 2 \gamma \delta ) ^ { 1 + \alpha / p } .
$$

Thus the exponent in the one-step bound (76) cannot be improved. This establishes componentwise sharpness: the clock and action constructions independently attain their respective exponents, while theirjoint product defines a separate end-to-end lower-bound problem.

Proof. At a decision state $\sigma _ { t } , t \in [ 0 , 1 ]$ , let actions $a _ { \pm }$ lead deterministically to one-step states of values $v _ { 0 } + c _ { \Delta } t$ and $v _ { 0 } .$ . Choose $v _ { 0 } , c _ { \Delta } , \delta > 0$ with $v _ { 0 } + c _ { \Delta } + \delta \le R _ { \mathrm { m a x } }$ , and shift the online successor values by $\mp \delta .$ . Then the true gap is $\gamma c _ { \Delta } t ,$ , but the online action switches exactly on $t < t _ { \delta } : = 2 \delta / c _ { \Delta }$ . Give � density $\alpha t ^ { \alpha - 1 }$ ; hence $C _ { \mathrm { m a r g } } = ( \gamma c _ { \Delta } ) ^ { - \hat { \alpha } }$ Direct integration gives

$$
\big \| { \mathcal T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k } - { \mathcal T } ^ { \pi _ { k } ^ { \mathrm { t g t } } } V _ { k } \big \| _ { p , \rho _ { k , s } } ^ { p } = \int _ { 0 } ^ { t _ { \delta } } ( \gamma c _ { \Delta } t ) ^ { p } \alpha t ^ { \alpha - 1 } d t = \frac { \alpha } { \alpha + p } ( \gamma c _ { \Delta } ) ^ { - \alpha } ( 2 \gamma \delta ) ^ { \alpha + p } ;
$$

taking �th roots proves the formula.

Proposition 4.8 (Abstract-recursion loss induced by reward-score error $[ \mathsf { R } ] )$ ). $F i x \gamma \in ( 0 , 1 )$ and normalize $R _ { \mathrm { m a x } } = 1 .$ For every $\eta \in ( 0 , \gamma / 2 )$ and every $\epsilon > 0$ there exist a deterministic time-augmented MDP with $H = 3$ and $| { \mathcal { A } } | = 2 $ an evaluation law $\mu _ { S ; }$ , a replay marginal $\rho _ { k , S } \equiv \rho _ { S }$ offull support, and an implemented one-step score $\hat { Q }$ of the form (2) with $\| \widehat { Q } _ { V _ { k } } - Q _ { V _ { k } } \| _ { \infty } = \eta .$ for every $k ,$ such that the population recursion initialized at $V _ { 0 } = 0$ and driven by the greedy branch of $\hat { Q }$ has zero drift, $\delta _ { V , k } = 0 $ , and zero $\begin{array} { r } { \mathscr { f } \mathscr { t } , } \end{array}$ kernel, target, replay, and exploration residuals, $\varepsilon _ { \mathrm { f i t } , k } = \varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = \varepsilon _ { \mathrm { b u f } , k } = \varepsilon _ { k } = 0 .$ . The score-induced online-action residual is the sole nonzero link, and yet

$$
\operatorname* { l i m } _ { K \to \infty } \left\| V ^ { \star } - V ^ { \pi _ { K } } \right\| _ { 1 , \mu _ { S } } \ge 2 \gamma \eta - \epsilon .\tag{78}
$$

Thus the construction realizes the $\alpha = 0$ score floorfor $0 < \eta < \gamma / 2 .$ . After the finite clock transient, its harmful behavior is generated endogenously by an implemented greedy score with $\eta _ { \mathrm { s c } , k } = \eta$ in every block.

Construction sketch. Use the deterministic $H = 3$ chain detailed in Appendix A, with terminal rewards $1 , 1 - u , v ,$ and perturb only the two modeled rewards at $y _ { 1 } \mathrm { ~ b y ~ } { - } \eta , { + } \eta .$ . The score error is exactly $\eta$ and switches the choice at $y _ { 1 }$ when $u < 2 \eta / \gamma$ . Taking $u = 2 \eta / \gamma - \vartheta$ and $v ^ { \prime } \in ( \dot { 1 } - u , 1 - u + \epsilon / ( 2 \gamma ^ { 2 } ) )$ makes the learned and optimal root actions different after the clock transient, with

$$
\| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } = \gamma ^ { 2 } ( 1 - v ) > 2 \gamma \eta - \epsilon
$$

for sufficiently small �. Exact full-support population updates, matched collection and replay, frozen targets, and zero exploration make all other residuals vanish. □

## 5 Neural and tabular regression rates

For $\mathsf { A } _ { \mathrm { i i d } }$ , we fit one sparse-ReLU spatial network at each remaining-horizon level. A fixed router selects the corresponding output. This construction allows smoothness assumptions on the spatial coordinates without imposing smoothness across the discrete clock levels. Assumption 5.1 collects the approximation, closure, entropy, and nesting conditions. The sparse-ReLU exponent and corrected depth condition come from [16, 17]; Propositions 5.2 and 5.3 prove Bellman closure and compatibility of the network class with these conditions. The bounded-loss oracle inequality follows the covering approach of Györfi et al. [18]. The mean regression target can differ from the Bellman optimality update because it averages the behavior policy. We retain this difference as ${ D } _ { k } ^ { ( 2 ) }$ [19–22]. Any class with the same approximation and entropy properties can replace the network class. The neural theorem fixes �; growing-horizon comparisons require Remark 3.22.

Assumption 5.1 (Externally clipped clock-routed sparse ReLU class and Hölder closure $\lceil \mathsf { A } _ { \mathrm { i i d } } \rceil )$ . (Slice representation.) For each $h \in [ H ] : = \{ 1 , \therefore , \dot { H } \}$ , fix a bi-measurable spatial embedding $\iota _ { h } : \mathcal { S } \times \{ h \} \to \mathsf { \bar { \rho } } [ 0 , 1 ]$ ]<sup>�</sup>emb onto its image, through which the slice-ℎ rewards and kernelfactor. The deterministic router observes the clock exactly, and the Hölder condition applies to the spatial coordinates within each slice. Any other discrete task coordinates � are either routed in the same way or included in $\iota _ { h }$ . Smoothness, margin, and tube hypotheses are uniform over the resultingfinitely many slices. The direct neural convergence route uses this routed input as the Markov-sufficient analysis state. Ifthe network receives only a compression �, the same architecture can be analyzed through Proposition ${ \dot { 3 } } . I 7 ,$ with visible-target aliasing included in thefit residual and closure checked on the Markov state.

(Per-head and routed classes.) For a block ofsize $\iota \geq 2 H ,$ , set

$$
m _ { n } : = \lfloor n / H \rfloor \geq 2 .\tag{79}
$$

Let $K g \ge 1$ be a common upper bound on the coordinatewise Hölder radii and domain endpoints used to define the finitely many target classes $\mathcal { G } _ { 0 , h }$ below, and fix the raw-network envelope

$$
B _ { \mathrm { n e t } } \geq \operatorname* { m a x } \{ 1 , V _ { \mathrm { m a x } } , K _ { \mathcal { G } } \} .\tag{80}
$$

For each ℎ, let $\mathcal { N } _ { m } ^ { ( h ) }$ be the raw scalar sparse-ReLU class on $[ 0 , 1 ] ^ { d _ { \mathrm { e m b } } }$ with weight and bias radius one, raw output bounded by $B _ { \mathrm { n e t } } ,$ , exactly $L _ { m }$ hidden layers, width profile $\{ d _ { j , m } \}$ , and sparsity at most $s _ { m }$ . The budgets are nondecreasing in �, dominate the constructive lower envelopes, and obey

$$
c _ { \mathrm { l o } } \log m \leq L _ { m } \leq c _ { \mathrm { h i } } ( \log m ) ^ { \xi ^ { \star } } ,
$$

$$
m ^ { \alpha ^ { \star } } \stackrel { < } { \sim } \operatorname* { m i n } _ { 1 \leq j \leq L _ { m } } d _ { j , m } \leq \operatorname* { m a x } _ { 1 \leq j \leq L _ { m } } d _ { j , m } \lesssim m ^ { \xi ^ { \star } } ,\tag{81}
$$

$$
s _ { m } \asymp m ^ { \alpha ^ { \star } } ( \log m ) ^ { \xi ^ { \star } } , \qquad \xi ^ { \star } \geq 1 .
$$

Define the nested head class $\begin{array} { r } { \mathcal { R } _ { m } ^ { ( h ) } : = \bigcup _ { r = 2 } ^ { m } \mathcal { N } _ { r } ^ { ( h ) } } \end{array}$ . The raw clock-routed class is the finite product

$$
\mathcal { R } _ { n } ^ { \mathrm { c l k } } : = \Big \{ f : f ( s , h ) = f _ { h } \big ( \iota _ { h } ( s , h ) \big ) , \quad f _ { h } \in \mathcal { R } _ { m _ { n } } ^ { ( h ) } , \ h \in [ H ] , \quad f \big ( \tilde { s } _ { \mathrm { t e r m } } \big ) = 0 \Big \} .\tag{82}
$$

Thus the router is fixed and non-trainable, and only the selected scalar head is evaluated. A routed predictor has at most $H s _ { m _ { n } }$ trainable nonzero parameters across its heads, maximum head depth $L _ { m _ { n } }$ , and no trainable gating parameter. This is still one global function class and one ERM under the mixed design law $\varsigma ; \ i t$ is not the directslice sampling object $\mathsf { A } _ { \mathrm { i i d } } ^ { \mathrm { l e v } }$ . Let $\mathcal { F } _ { n } ^ { V ^ { * } } : = \Pi ^ { \mathrm { c l i p } } \mathcal { R } _ { n } ^ { \mathrm { c l k } }$ , where the level-wise projection is fixed external post-processing and contributes no trainable parameter. In particular, the ERM class is band-valued even though its raw networks use the larger envelope (80). Each finite-architecture bounded-parameter class is separable in supremum norm; choose countable dense subsets, take theirfinite unions and �-fold product, and then clip to obtain afixed countable supremum-norm-dense subclass $\mathcal { F } _ { n } ^ { V , 0 } \subseteq \breve { \mathcal { F } } _ { n } ^ { V }$

(Target and admissibility.) Let $\mathcal { G } _ { 0 , h }$ be the slice-ℎ compositional Hölder class of [23, Assumption 4.2], intersected with the band $\| g _ { h } \| _ { \infty } \leq V _ { \operatorname* { m a x } } ^ { ( h ) }$ , with indices and radii uniformly bounded in ℎ, and define

$$
\mathcal { G } _ { 0 } ^ { \mathrm { c l k } } : = \left\{ g \in \mathcal { V } ^ { \mathrm { c l i p } } : f o r e \nu e r y h t h e r e i s \bar { g } _ { h } \in \mathcal { G } _ { 0 , h } w i t h g ( s , h ) = \bar { g } _ { h } ( \iota _ { h } ( s , h ) ) \right\} .\tag{83}
$$

Writing dis $\begin{array} { r } { { \mathrm { t } } _ { \infty } ^ { \operatorname* { s u p } } ( \mathcal { C } , \mathcal { G } ) : = \operatorname* { s u p } _ { g \in \mathcal { G } } \operatorname* { i n f } _ { V \in \mathcal { C } } \| V - g \| _ { \infty } , } \end{array}$ assume that there is a target-uniform $C _ { \mathrm { a p p x } } < \infty$ such that

$$
\operatorname* { m a x } _ { h \in [ H ] } \mathrm { d i s t } _ { \infty } ^ { \operatorname* { s u p } } \bigl ( \Pi _ { h } ^ { \mathrm { c l i p } } \mathcal { R } _ { m } ^ { ( h ) } , \mathcal { G } _ { 0 , h } \bigr ) \leq C _ { \mathrm { a p p x } } m ^ { ( \alpha ^ { \star } - 1 ) / 2 } , \qquad m \geq 2 ,\tag{84}
$$

where $\Pi _ { h } ^ { \mathrm { c l i p } }$ clips $t o \left[ - V _ { \operatorname* { m a x } } ^ { ( h ) } , V _ { \operatorname* { m a x } } ^ { ( h ) } \right]$ on slice ℎ, and $\alpha ^ { \star } = \mathrm { m a x } _ { j } t _ { j } / ( 2 \beta _ { j } ^ { \star } + t _ { j } ) \in ( 0 , 1 )$ . Consequently, selecting one approximant independently in each of the finitely many heads gives

$$
\mathrm { d i s t } _ { \infty } ^ { \mathrm { s u p } } ( \mathcal { F } _ { n } ^ { V } , \mathcal { G } _ { 0 } ^ { \mathrm { c l k } } ) \leq C _ { \mathrm { a p p x } } m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } .\tag{85}
$$

(Closure, entropy, nesting, and uniformity.) For every $n \geq 2 H$ and $V \in \mathcal { F } _ { n } ^ { V }$ , assume $\mathcal { T V } \in \mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$ . Put $d _ { 0 , m } : = d _ { \mathrm { e m b } } ,$ $d _ { L _ { m } + 1 , m } : = 1$ , and $\begin{array} { r } { D _ { m } : = \prod _ { j = 0 } ^ { L _ { m } + 1 } ( d _ { j , m } + 1 ) } \end{array}$ . The head covering bound and the product construction give, for $0 < \delta \leq 1 _ {  }$

$$
\log N _ { \delta } ( \mathcal { F } _ { n } ^ { V } ) \leq H \left\{ \log m _ { n } + ( s _ { m _ { n } } + 1 ) \log \left( \frac { 2 ( L _ { m _ { n } } + 1 ) D _ { m _ { n } } ^ { 2 } } { \delta } \right) \right\} .\tag{86}
$$

The head classes are nested in �; hence $m _ { n } \leq m _ { n ^ { \prime } }$ implies $\mathcal { F } _ { n } ^ { V } \subseteq \mathcal { F } _ { n ^ { \prime } } ^ { V }$ . The Hölder radii, compositional indices, approximation and entropy constants are uniform in $h , n , k ,$ , and in the realized $V _ { k } \in \mathcal { F } _ { n } ^ { V }$ . Constants in the results may depend on thefixed horizon �.

(Corrected depth check.) Let $i = 0 , \ldots , q$ index a head’s composition layers, with parameters $\left( { \beta _ { i } , t _ { i } , \beta _ { i } ^ { \star } } \right)$ . Set

$$
C _ { \mathrm { S H } } : = \sum _ { i = 0 } ^ { q } \frac { \beta _ { i } + t _ { i } } { 2 \beta _ { i } ^ { \star } + t _ { i } } \log _ { 2 } ( 4 t _ { i } \vee 4 \beta _ { i } ) , \qquad \phi _ { m } : = \operatorname* { m a x } _ { i } m ^ { - 2 \beta _ { i } ^ { \star } / ( 2 \beta _ { i } ^ { \star } + t _ { i } ) } .
$$

Then $m \phi _ { m } = m ^ { \alpha ^ { \star } }$ . Consequently $c _ { \mathrm { l o } } \geq C _ { \mathrm { S H } } / \log 2$ verifies the corrected lower depth requirement �<sub>SH</sub> log<sub>2</sub> $m \leq L _ { m }$ while the polylogarithmic upper bound in (81) is eventually $L _ { m } \lesssim m \phi _ { m } .$ The same display gives $m \phi _ { m } \lesssim \operatorname* { m i n } _ { j } d _ { j }$ <sub>,�</sub>, and its sparsity budget contains the constructive scale $s _ { m } ^ { \prime } \asymp m \phi _ { m }$ log $m \le s _ { m }$ . Together with (80), these are the output-envelope, corrected depth, width, and sparsity compatibility checks. Equation (84), Bellman closure, and uniform target-class radii remain explicit hypotheses in the general compositional setting and are verifiedfor thefinite-rank model below.

Proposition 5.2 (Finite-rank smoothing implies slice-wise Bellman closure and coverage $[ \mathsf { A } _ { \mathrm { i i d } } ] )$ . Let each clock slice be $\lceil 0 , 1 \rceil ^ { d }$ with reference law $\nu _ { h } ,$ , let $\mu _ { S } = \nu _ { H } \otimes \delta _ { H } $ , and use the reset law $\begin{array} { r } { \varsigma = H ^ { - 1 } \sum _ { h = 1 } ^ { H } \nu _ { h } \otimes \delta _ { h } } \end{array}$ . Suppose that,for $h > 1$ , the action-� transition has density

$$
\begin{array} { c l } { { } } & { { p _ { h , a } ( y \mid x ) = 1 + \displaystyle \sum _ { j = 1 } ^ { r } \phi _ { h , a , j } ( x ) \psi _ { h , a , j } ( y ) \quad r e l a t i \nu e t o \nu _ { h - 1 } , } } \\ { { } } & { { } } \\ { { \displaystyle \int \psi _ { h , a , j } d \nu _ { h - 1 } = 0 , \qquad 0 \leq p _ { h , a } \leq M . } } \end{array}
$$

and that termination occurs after level one. If, uniformly in $( h , a , j ) , R _ { h , a }$ and $\phi _ { h , a , j }$ lie in afixed �-Hölder ball on $[ 0 , 1 ] ^ { d } , 0 < \beta \leq 1$ , and $\| \psi _ { h , a , j } \| _ { 1 }$ is bounded, then $\{ T V : V \in \mathcal { V } ^ { \mathrm { c l i p } } \}$ lies in a fixed �-Hölder ball on every clock slice, with a radius uniform in ℎ and �. Moreover,for every policy sequence and $1 \leq m < H$

$$
c _ { 2 } ( m ) \leq H M , \qquad c _ { \infty } ( m ) \leq H M .
$$

Consequently, $i f \mathcal { G } _ { 0 , h }$ is this common-radius Hölder ball intersected with the level-ℎ band, then $\{ { \mathcal { T } } V : V \in { \mathcal { V } } ^ { \mathrm { c l i p } } \} \subset$ $\mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$ , and the closure and uniform-radius clauses ofAssumption 5.1 hold. This proposition makes no neural-network approximation, nesting, or entropy claim.

Proof. For $V \in \mathcal { V } ^ { \mathrm { c l i p } }$

$$
( \mathcal P _ { h , a } V ) ( x ) = \int V d \nu _ { h - 1 } + \sum _ { j = 1 } ^ { r } \phi _ { h , a , j } ( x ) \int \psi _ { h , a , j } V d \nu _ { h - 1 } .
$$

The coefficients are bounded uniformly by $V _ { \operatorname* { m a x } } \Vert \psi _ { h , a , j } \Vert _ { 1 }$ ; hence $R _ { h , a } + \gamma \mathcal { P } _ { h , a } V$ has a uniform �-Hölder radius. A finite maximum preserves that radius when $\beta \leq 1$ , proving closure; the finite number of slices makes the radius uniform in ℎ. If a slice law has density � relative to $\nu _ { h } .$ , its successor density $g ^ { \prime }$ is bounded by $\begin{array} { r } { M \int g d \nu _ { h } = M } \end{array}$ . Relative to � it is therefore at most ��, so $c _ { \infty } ( m ) \leq H M$ and $\begin{array} { r } { c _ { 2 } ( m ) \leq H \int g _ { m } ^ { 2 } d \nu _ { H - m } \leq H M } \end{array}$ , where $g _ { m } \leq M$ is the propagated spatial density at depth �. □

Proposition 5.3 (Clock-routed ReLU admissibility for the finite-rank example $\big [ \mathsf { A } _ { \mathrm { i i d } } \big ] \big )$ . Under Proposition 5.2, take $d _ { \mathrm { e m b } } = d ,$ let $\iota _ { h }$ be the identity on the spatial coordinate, and let every $\mathcal { G } _ { 0 , h }$ be the corresponding band-intersected, common-radius �-Hölder ball. Let $\dot { K _ { \mathcal { G } } }$ bound that radius and choose $B _ { \mathrm { n e t } } \ \geq \ \operatorname* { m a x } \{ \mathrm { i } , V _ { \mathrm { m a x } } , K _ { \mathcal { G } } \}$ . There exist nondecreasing fixed-architecture budgets and nested per-head raw sparse-ReLUfamilies $\{ \mathcal { R } _ { m } ^ { ( h ) } \} _ { m \geq 2 }$ satisfying Assumption 5.1 with

$$
q = 0 , \qquad t = d , \qquad \alpha ^ { \star } = \frac { d } { 2 \beta + d } ,
$$

including the global approximation (85), product entropy $( 8 6 ) ,$ , countable dense subclasses, nesting, the output-envelope condition, the corrected depth condition, and the width and sparsity requirements. Hence the finite-rank model, together with the externally clipped clock-routed architecture, verifies all clauses ofAssumption 5.1.

Proof. For this $q = 0$ class, $\beta ^ { \star } = \beta , \phi _ { m } = m ^ { - 2 \beta / ( 2 \beta + d ) }$ , and $m \phi _ { m } = m ^ { d / ( 2 \beta + d ) } = m ^ { \alpha ^ { \star } }$ . Choose the exact depth, hidden widths, and sparsity budgets nondecreasingly so that, for all sufficiently large �,

$$
\begin{array} { c } { { \displaystyle \frac { \beta + d } { 2 \beta + d } \log _ { 2 } ( 4 d \vee 4 \beta ) \log _ { 2 } m \le L _ { m } \lesssim m \phi _ { m } , } } \\ { { m \phi _ { m } \lesssim \operatorname* { m i n } d _ { j , m } , \hfill } } \\ { { m \phi _ { m } \log m \lesssim s _ { m } . } } \end{array}\tag{87}
$$

while retaining the upper envelopes in (81). Enlarge the finitely many small-� budgets if necessary. These inequalities are mutually compatible because $\alpha ^ { \star } \in ( 0 , 1 )$ and $\xi ^ { \star } \geq 1$

The approximation part of Schmidt–Hieber’s construction, with condition (ii’) of the correction, now applies with sample-size index � and output envelope $ { { B _ { \mathrm { n e t } } } } \ [ 1 6 , \ 1 7 ]$ . Uniformly over the fixed Hölder ball it produces a unitparameter raw network $\widetilde { f } _ { h , m }$ , bounded by $\boldsymbol { B } _ { \mathrm { n e t } }$ , with

$$
\| \widetilde { f } _ { h , m } - g _ { h } \| _ { \infty } \leq C m ^ { - \beta / ( 2 \beta + d ) } = C m ^ { ( \alpha ^ { \star } - 1 ) / 2 } .
$$

The constructive network embeds into the declared fixed architecture: pad hidden layers by inactive neurons, and insert any additional identity layers before the constructive network. Inputs lie in $[ 0 , 1 ] ^ { d }$ , so unit-weight, zero-bias identity layers survive ReLU unchanged; their $O ( d L _ { m } )$ additional nonzero parameters are absorbed by $s _ { m }$ . This is the standard width/depth embedding used in the cited construction. Thus $\widetilde { f } _ { h , m } \in \mathcal { N } _ { m } ^ { ( h ) }$ after padding. For the finitely many small indices, the zero network belongs to the class and the constant � can be enlarged, so the bound holds for every $m \geq 2$

Because $g _ { h }$ lies in the level-ℎ band, Lemma 2.3 gives

$$
\| \Pi _ { h } ^ { \mathrm { c l i p } } \widetilde { f } _ { h , m } - g _ { h } \| _ { \infty } \leq \| \widetilde { f } _ { h , m } - g _ { h } \| _ { \infty } .
$$

This proves (84). Take the nested hul $\begin{array} { r } { \mathcal { R } _ { m } ^ { ( h ) } = \bigcup _ { r = 2 } ^ { m } \mathcal { N } _ { r } ^ { ( h ) } } \end{array}$ ; the finite number of slices makes the approximation constant uniform in ℎ. For $g \in \mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$ , choose the � clipped approximants independently and route by the observed clock. The global supremum error is the maximum of the head errors, proving (85); exact routing therefore preserves the spatial approximation rate without introducing a clock-gate approximation term.

For entropy, Lemma 6.4 of [23] bounds each base raw class $\mathcal { N } _ { r } ^ { ( h ) }$ ; imposing the additional raw-output envelope only takes a subclass. A union bound over $2 \leq r \leq$ � contributes log � to the log covering number. Take a �-net for each resulting raw head class and then apply $\Pi _ { h } ^ { \mathrm { c l i p } }$ . The Cartesian product of the � clipped nets is a �-net for the routed class because its metric is the maximum of the slice metrics. Summing the log covering numbers gives (86); clipping is nonexpansive by Lemma 2.3. Finite-parameter raw classes are separable, and finite unions, clipping, and the finite routed product preserve a countable dense subclass.

The nested hulls and deterministic router give nesting of the global clipped classes. Equation (87) verifies the corrected depth condition and the width/sparsity embedding, while (80) verifies the output-envelope condition. Closure and its uniform radius come from Proposition 5.2, completing every clause of Assumption 5.1. □

Lemma 5.4 (Oracle inequality for squared loss with bounded targets [ G ]). Let $B > 0$ and let $\mathcal { C } \subset \{ V : \widetilde { \mathcal { S } }  [ - B , B ] \}$ be deterministic and admit afixed countable supremum-norm-dense subclass $\mathcal { C } _ { 0 } .$ . Let $g : \widetilde { S }  [ - B , B ]$ be measurable, and let $( S _ { i } , Y _ { i } ) _ { i \leq n }$ be i.i.d. with $S _ { i } \sim \rho _ { k , S } , | Y _ { i } | \le B$ and $\mathbb { E } [ Y _ { i } \mid S _ { i } ] = g ( S _ { i } )$ . Let $\widehat { V } _ { k + 1 }$ be any measurable � -approximate empirical risk minimizer over $\mathcal { C } _ { 0 } \left( o r \right.$ over � itself when such a measurable minimizer is supplied): $\begin{array} { r } { n ^ { - 1 } \sum _ { i \leq n } ( \widehat { V } _ { k + 1 } ( \bar { S _ { i } } ) - Y _ { i } ) ^ { 2 } \leq \operatorname* { i n f } _ { V \in \mathcal { C } _ { 0 } } n ^ { - 1 } \sum _ { i < n } ( V ( S _ { i } ) - Y _ { i } ) ^ { 2 } + \zeta _ { n } } \end{array}$ with $\zeta _ { n } \geq 0 ,$ exact ERM being $\zeta _ { n } = 0 .$ . Then, for every $\delta \in ( 0 , 2 B ]$

$$
\mathbb { E } \Vert \widehat { V } _ { k + 1 } - g \Vert _ { 2 , \rho _ { k } , s } ^ { 2 } \leq 2 \big [ \normalfont { \mathrm { d i s t } } _ { 2 , \rho _ { k } , s } ( \mathcal { C } , g ) \big ] ^ { 2 } + \frac { C _ { 1 } B ^ { 2 } } { n } \big ( 1 + \log N _ { \delta } ( \mathcal { C } ) \big ) + C _ { 1 } B \delta + 2 \zeta _ { n } ,\tag{88}
$$

with an absolute $C _ { 1 } ;$ the proof is valid for every $C _ { 1 } \geq 1 4 0$ and makes no use ofits value. Here dist ${ \bf \rho } _ { ; 2 , \rho _ { k , S } } ( { \mathcal C } , g )$ is the $L ^ { 2 } ( \rho _ { k , S } )$ distancefrom the class to a single target and $N _ { \delta } ( { \mathcal { C } } )$ the supremum-norm covering number, assumedfinite. The same inequality holds conditionally on a sigma-algebra relative to which �, �, and $\rho _ { k , S }$ are fixed, provided the sample is conditionally i.i.d. and $\mathbb { E } [ Y _ { i } \mid S _ { i } , { \boldsymbol { \mathcal { K } } } ] { \stackrel { } { = } } g ( S _ { i } ) ;$ it is used below with $\kappa = \mathcal { H } _ { k } f o r \mathsf { A } _ { \mathrm { i i d } }$ . The approximation term is in $L ^ { 2 } ( \rho _ { k , S } )$ and only the covering radius is in supremum norm; since $\rho _ { k , S }$ is a probability measure the statement implies, and is stronger than, its supremum-norm form. If, additionally, $g \in { \mathcal { C } }$ and $\widehat { V } _ { k + 1 }$ is a supplied measurable exact ERM over ${ \mathcal { C } } ,$ , then for every $\tau \in ( 0 , 1 )$ , conditionally on the samefixed information when present, with probability at least $1 - \tau$

$$
\| \widehat { V } _ { k + 1 } - g \| _ { 2 , \rho _ { k , S } } ^ { 2 } \leq \frac { C _ { 1 } B ^ { 2 } } { n } \big ( 1 + \log N _ { \delta } ( \mathcal { C } ) + \log ( 1 / \tau ) \big ) + C _ { 1 } B \delta .\tag{89}
$$

Proof: See Appendix A.

Proposition 5.5 (Generative-reset clock-routed scalar-� regression $\left[ \mathsf { A } _ { \mathrm { i i d } } \right] )$ . Under Assumption 5.1, fix $n \geq 2 H$ condition on $\mathcal { H } _ { k }$ , and let $V _ { k } \in \mathcal { F } _ { n } ^ { V }$ . Let $( S _ { i } , \mathbf { a } _ { i } , r _ { i } , \tilde { s } _ { i } ^ { \prime } , d _ { i } ) _ { i \leq n }$ be the i.i.d. sample of $\mathsf { A } _ { \mathrm { i i d } } \left( 3 \right)$ , with $S _ { i } \sim \rho _ { k , S } \equiv \varsigma$ $\mathbf { a } _ { i } \sim \beta _ { k } ( \cdot \mid S _ { i } )$ , fresh true-kernel outcomes and labels $Y _ { i } = r _ { i } + \gamma ( 1 - d _ { i } ) V _ { k } ( \tilde { s } _ { i } ^ { \prime } )$ , and let $\widehat { V } _ { k + 1 }$ be any measurable �<sub>�</sub>-approximate ERM over the fixed countable dense subclass $\mathcal { F } _ { n } ^ { V , 0 }$ . By construction, $\mathbb { E } [ Y _ { i } \mid S _ { i } , \mathcal { H } _ { k } ] = G _ { k } ( S _ { i } )$ . Define the ${ \dot { L } } ^ { \dot { 2 } }$ policy-image defect

$$
D _ { k } ^ { ( 2 ) } : = \| G _ { k } - \mathcal { T } V _ { k } \| _ { 2 , \rho _ { k , S } } ,\tag{90}
$$

a quantity and not a hypothesis. Then there is a constant $C _ { \mathrm { V r e g } , H } ,$ uniform in �, $n ,$ and the realized $V _ { k } \in \mathcal { F } _ { n } ^ { V }$ under the uniformity clause ofAssumption 5.1. It may depend on $\bar { H , \gamma , R _ { \mathrm { m a x } } }$ through $V _ { \mathrm { m a x } } ,$ , on $| { \cal A } |$ through the assumed target class, and on $C _ { 1 } , C _ { \mathrm { a p p x } } ,$ , the architecture and covering constants, the fixed router and slice embeddings, and the compositional indices and Hölder radii, but not on �, $k ,$ or the realized iterate. Then

$$
\begin{array} { r } { \mathbb { E } \big [ \| \widehat { V } _ { k + 1 } - G _ { k } \| _ { 2 , \rho _ { k , \mathcal { S } } } \big | \mathcal { H } _ { k } \big ] \leq C _ { \mathrm { V r e g } , H } ( \log n ) ^ { ( 1 + 2 \xi ^ { \star } ) / 2 } m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } + \sqrt { 2 \zeta _ { n } } + \sqrt { 2 } D _ { k } ^ { ( 2 ) } . } \end{array}\tag{91}
$$

Greedy collection is the case $D _ { k } ^ { ( 2 ) } = 0 \colon i f { \mathbf { a } } _ { i } = a ^ { \star } ( S _ { i } ; V _ { k } )$ then $G _ { k } = \tau V _ { k }$ by Lemma 2.4(iii) and

$$
\mathbb { E } \Big [ \| \widehat { V } _ { k + 1 } - { \mathcal { T } } V _ { k } \| _ { 2 , \rho _ { k , S } } \Big | \mathcal { H } _ { k } \Big ] \leq C _ { \mathrm { V r e g } , H } ( \log n ) ^ { ( 1 + 2 \xi ^ { \star } ) / 2 } m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } + \sqrt { 2 \zeta _ { n } } .\tag{92}
$$

The required closure is precisely the Bellman condition $\mathcal { T V } \in \mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$ with uniform compositional Hölder radii from Assumption 5.1. The policy-image discrepancy of $G _ { k }$ remains in the bound: choosing the comparator in $L ^ { 2 } ( \rho _ { k , S } )$ produces exactly ${ D } _ { k } ^ { ( 2 ) }$ , which Proposition 5.6 bounds by the named residuals.

Proof: See Appendix A.

Proposition 5.6 (Bounding the regression defect in $L ^ { 2 } [ \mathsf { R } ] )$ ). For every block $k ,$ the defect in (90) satisfies

$$
D _ { k } ^ { ( 2 ) } \leq \varepsilon _ { \mathrm { k e r } , k } + \varepsilon _ { \mathrm { t g t } , k } + \varepsilon _ { \mathrm { b u f } , k } + \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } }\tag{93}
$$

$$
\leq \varepsilon _ { \mathrm { k e r } , k } + \varepsilon _ { \mathrm { t g t } , k } + \varepsilon _ { \mathrm { b u f } , k } + ( 1 - \varepsilon _ { k } ) 2 \Lambda _ { k } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } .\tag{94}
$$

Ifthe globalfrozen-iterate margin holds, or its localform holds with $2 \Lambda _ { k } \in [ u _ { 0 } , \bar { u } ]$ , then the sharper bound

$$
D _ { k } ^ { ( 2 ) } \ \leq \ \varepsilon _ { \mathrm { k e r } , k } + \varepsilon _ { \mathrm { t g t } , k } + \varepsilon _ { \mathrm { b u f } , k } + ( 1 - \varepsilon _ { k } ) \operatorname* { m i n } \{ 2 \Lambda _ { k } , \sqrt { C _ { \mathrm { m a r g } } } ( 2 \Lambda _ { k } ) ^ { 1 + \alpha / 2 } \} + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } ,\tag{95}
$$

also holds. Under Proposition $4 . 3 ,$ the same display holds with its action term replaced by

$$
\left( 1 - \varepsilon _ { k } \right) \operatorname* { m i n } \biggr \{ 2 \Lambda _ { k } , 2 \Lambda _ { k } \left[ \tau _ { \star } + C _ { \star } ( 2 \Lambda _ { k } + 2 \delta _ { k } ^ { \star } ) ^ { \alpha _ { \star } } \right] ^ { 1 / 2 } \biggr \} .
$$

The norm choice is essential. Consider thefavorable no-mismatch case $\varepsilon _ { k } = 0 , \beta _ { k } ^ { \mathrm { r e p } } = \pi _ { k } ^ { \mathrm { o n } }$ , and $\varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = 0 _ { : }$ so $G _ { k } = \mathcal { T } ^ { \pi _ { k } ^ { \mathrm { o n } } } V _ { k }$ . Suppose the selected action switches at $\tilde { s } _ { 0 }$ between � and $a ^ { \prime } ,$ , the twofunctions $Q _ { V _ { k } } ( \cdot , a ) , Q _ { V _ { k } } ( \cdot , a ^ { \prime } )$ are continuous there, and every neighborhood of $\tilde { s } _ { 0 }$ has positive $\rho _ { k }$ -mass in both selection regions. $I f \ J _ { k } \ : =$ $| Q _ { V _ { k } } ( \tilde { s } _ { 0 } , a ) - Q _ { V _ { k } } ( \tilde { s } _ { 0 } , a ^ { \prime } ) | > 0 ,$ , write

$$
\mathrm { d i s t } _ { \infty , \rho } ( \mathcal C , g ) : = \operatorname* { i n f } _ { f \in \mathcal C } \operatorname* { s s s u p } _ { \rho } | f - g | .
$$

Then every class � whose members are continuous at $\tilde { s } _ { 0 }$ satisfies

$$
\begin{array} { r } { \mathrm { d i s t } _ { \infty , \rho _ { k , S } } ( \mathcal { C } , G _ { k } ) \geq \frac { 1 } { 2 } J _ { k } . } \end{array}\tag{96}
$$

The construction of Proposition $4 . 7$ realizes $J _ { k } = 2 \Lambda _ { k } ;$ hence a continuous supremum-norm comparator remains linear in $\Lambda _ { k }$ for every margin exponent.

Proof. $G _ { k } - \mathcal { T } V _ { k }$ is the tail of the telescope of Lemma 3.13 from $G _ { k }$ to $\mathcal { T } V _ { k }$ , so the triangle inequality in the common space $L ^ { 2 } ( \rho _ { k , S } )$ along those links gives (93). Bound (75) at $r = 2 .$ , with Corollary 4.6 for a period-level mixture, then gives (94). Under the stated margin hypotheses, Bound (76) at $r = 2$ gives the margin term; taking the smaller of the two valid action bounds proves $( 9 5 )$ . The same argument using (77) proves the reference-gap version. For $( 9 6 ) , G _ { k }$ has one-sided essential limits $Q _ { V _ { k } } ( \tilde { s } _ { 0 } , a )$ and $Q _ { V _ { k } } ( \tilde { s } _ { 0 } , a ^ { \prime } )$ along the two selection regions. A function continuous at $\tilde { s } _ { 0 }$ has one limit and therefore cannot lie within less than half their separation of both; the positive-mass condition turns this into an essential-supremum bound. At the switching point in Proposition 4.7, that separation is $2 \gamma \delta = 2 \Lambda _ { k }$ □

Corollary 5.7 (Tabular generative-reset bound with known envelopes $[ \mathsf { A } _ { \mathrm { i i d } } ] )$ . Assume $H \ \geq \ 2 .$ . Let $\widetilde { S } ^ { \circ }$ be finite, $N : = | \widetilde { S } ^ { \circ } |$ , and replace Assumption 5.1 by the fixed level-wise clipped tabular class $\mathcal { F } ^ { \mathrm { t a b } } : = \mathcal { V } ^ { \mathrm { c l i p } }$ . This class is exactly Bellman closed and admits a measurable exact coordinatewise ERM: the label average at a visited state and zero at an unvisited state. Condition on the pre-block history and write $p _ { k , s } : = \rho _ { k , S } \{ s \} , g _ { k , s } : = \mathbb { E } [ Y \mid S = s , \mathcal { H } _ { k } ]$ $\sigma _ { k , s } ^ { 2 } : = \operatorname { V a r } ( Y \mid S = s , { \mathcal { H } } _ { k } )$ , and $N _ { k , s } f o r$ the number of visits to � in the � -sample block. Then, with zero-mass coordinates understood to contribute zero, its exact conditional squared risk is

$$
\mathbb { E } \Big [ \| V _ { k + 1 } - g _ { k } \| _ { 2 , \rho _ { k , S } } ^ { 2 } \ | \ \mathcal { H } _ { k } \Big ] = \sum _ { s \in \tilde { S } ^ { \circ } } p _ { k , s } \left[ \sigma _ { k , s } ^ { 2 } \mathbb { E } \Bigg \{ \frac { \mathbf { 1 } \{ N _ { k , s } > 0 \} } { N _ { k , s } } \Bigg | \ \mathcal { H } _ { k } \right\} + g _ { k , s } ^ { 2 } ( 1 - p _ { k , s } ) ^ { n _ { k } } \Bigg ] .\tag{97}
$$

Since $| Y | \leq V _ { \operatorname* { m a x } } ,$ Jensen and the binomial count calculation in the proof give the log-free deterministic fit envelope

$$
\mathrm { s t a t } _ { k } ^ { \mathrm { t a b } } : = V _ { \operatorname* { m a x } } \sqrt { \frac { N \{ 2 + ( n _ { k } / ( n _ { k } + 1 ) ) ^ { n _ { k } } \} } { n _ { k } + 1 } } \le V _ { \operatorname* { m a x } } \sqrt { \frac { 5 N } { 2 ( n _ { k } + 1 ) } } ,\tag{98}
$$

$$
n _ { k } \ge 2 : \qquad \mathrm { s t a t } _ { k } ^ { \mathrm { t a b } } \le V _ { \mathrm { m a x } } \sqrt { \frac { 2 2 N } { 9 ( n _ { k } + 1 ) } } .
$$

Supposefurther that afixedfull-support reset law � is also the state design law $\rho _ { k , S } ;$ the fresh collection and replay action laws coincide; slot (S6) uses (SAMPLE, OBS); $O = S ;$ and $W _ { k } = { \bar { V } } _ { k }$ . Then

$$
\varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = \varepsilon _ { \mathrm { b u f } , k } = \varepsilon _ { \mathrm { a l i a s } , k } = \delta _ { V , k } = 0 , \qquad \Lambda _ { k } = \eta _ { \mathrm { s c } , k } ,\tag{99}
$$

and, for every $K \geq H - 1$ -9

$$
\mathbb { E } \| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } \leq 2 \phi _ { s } ^ { ( H ) } \operatorname* { m a x } _ { K - H < k < K } \bigl [ \mathrm { s t a t } _ { k } ^ { \mathrm { t a b } } + A _ { \eta , k } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } \bigr ] ,\tag{100}
$$

where $A _ { \eta , k }$ is the chosen envelope $\varepsilon _ { \mathrm { a c t } , k } ^ { ( p ) } \colon A _ { \eta , k } : = ( 1 - \varepsilon _ { k } ) 2 \eta _ { \mathrm { s c } , k }$ without a margin, while under the frozen-iterate margin at scale $2 \eta _ { \mathrm { s c } , k }$ it may be replaced by $\left( 1 - \varepsilon _ { k } \right)$ min $\{ 2 \eta _ { \mathrm { s c } , k } , C _ { \mathrm { m a r g } } ^ { 1 / p } ( 2 \eta _ { \mathrm { s c } , k } ) ^ { 1 + \alpha / p } \}$ . Thus every residual slot is zero or explicitly bounded; $i f \eta _ { \mathrm { s c } , k } \to 0 , n _ { k } \to \infty ,$ , and $\varepsilon _ { k } \to 0 ,$ the right side tends to zero.

There is also a finite-confidence form whose tabular regression term has no additional minimum-state-mass factor; coverage in the policy bound still enters through $\phi _ { s } ^ { ( H ) }$ . For $\delta \in ( 0 , 1 )$ and fixed $K \geq H - 1$ , put

$$
\mathrm { s t a t } _ { k , K } ^ { \mathrm { t a b , h p } } ( \delta ) : = V _ { \operatorname* { m a x } } \sqrt { \frac { C _ { 1 } \{ 2 + N \log ( 2 n _ { k } + 1 ) + \log ( ( H - 1 ) / \delta ) \} } { n _ { k } } } .\tag{101}
$$

Then, with probability at least $1 - \delta$ over the adaptivefresh blocks,

$$
\| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } \leq 2 \phi _ { s } ^ { ( H ) } \operatorname* { m a x } _ { K - H < k < K } \left[ \mathrm { s t a t } _ { k , K } ^ { \mathrm { t a b , h p } } ( \delta ) + A _ { \eta , k } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } \right] .\tag{102}
$$

Proof: See Appendix A.

## 6 Deployment and consistency

The main bound evaluates a true-� greedy policy, whereas the deployed controller uses the implemented final score.   
Theorem 6.1 bounds the resulting difference, and Corollary 6.6 gives consistency when the residuals decay.

## 6.1 Implemented-score controller

Theorem 6.1 (Survival-aware transfer to the implemented-score controller [ R ]). Assume the hypotheses of Theorem 3.18, let $\widehat { \pi } _ { K } = \widehat { a } ^ { \star } ( \cdot ; V _ { K } )$ be the deployed policyfrom (2), stationary on the clock-augmented state (and generally nonstationary on the physical state alone), and let $\eta _ { \mathrm { s c } , K }$ be a deterministic upper bound on the score error at the final frozen network, $\| \widehat { Q } _ { V _ { K } } - Q _ { V _ { K } } \| _ { \infty } \leq \eta _ { \mathrm { s c } , K }$ almost surely. Suppose also that deterministic $\bar { q } _ { \ell } \in [ 0 , 1 ]$ satisfy, almost surely,

$$
\mu _ { S } ^ { \circ } ( \mathcal { P } _ { \circ } ^ { \widehat { \pi } _ { K } } ) ^ { \ell } \mathbf { 1 } \leq \bar { q } _ { \ell } , \qquad 0 \leq \ell < H ;\tag{103}
$$

the universal choice is $\bar { q } _ { \ell } = 1$ . Then

$$
\mathbb { E } \Big [ \| V ^ { \star } - V ^ { \widehat { \pi } \kappa } \| _ { 1 , \mu s } \Big ] \leq \mathcal { B } _ { K } + \sum _ { k = 0 } ^ { K - 1 } w _ { K , k } ^ { ( H ) } e _ { k , p } ^ { \mathrm { B e l l } } + 2 \eta _ { \mathrm { s c } , K } \sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } \bar { q } _ { \ell } ,\tag{104}
$$

that is, decomposition (49) holds for the deployed controller with a single survival-weighted score term. It is never worse than $2 \eta _ { \mathrm { s c } , K }$ min $\{ H , ( 1 - \gamma ) ^ { - 1 } \}$ , and no margin assumption is used.

Proof. Let $d _ { K } : = \mathcal { T } V _ { K } - \mathcal { T } ^ { \widehat { \pi } _ { K } } V _ { K }$ . The fixed Borel rule makes $\widehat { \pi } _ { K }$ measurable, deterministic and stationary, and Lemma 4.4(iii) gives pointwise

$$
0 \leq d _ { K } ( \tilde { s } ) = Q _ { V _ { K } } ( \tilde { s } , a ^ { \star } ( \tilde { s } ; V _ { K } ) ) - Q _ { V _ { K } } ( \tilde { s } , \widehat { \pi } _ { K } ( \tilde { s } ) ) \leq 2 \eta _ { \mathrm { s c } , K } .
$$

Repeating Lemma 3.10 changes only its nonnegative Singh–Yee loss-resolvent step [24], adding after integration

$$
\sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } \int d _ { K } d \{ \mu _ { S } ^ { \circ } ( \mathcal P _ { \circ } ^ { \widehat { \pi } _ { K } } ) ^ { \ell } \} \leq 2 \eta _ { \mathrm { s c } , K } \sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } \bar { q } _ { \ell } .
$$

Clock nilpotence gives the finite sum. All other residual branches and the coverage step are unchanged because $\widehat { \pi } _ { K }$ is a measurable deterministic policy, proving (104). □

Corollary 6.2 (Final-law coverage and margin for deployment $[ \mathsf { R } ] )$ . In the setting ofTheorem $6 . I ,$ set Γ $\overset { \triangledown } { \boldsymbol { \kappa } } = 0$ when $\eta _ { \mathrm { s c } , K } = 0$ , in which case no gap condition is needed. When $\eta _ { \mathrm { s c } , K } > 0 ,$ , suppose the final true gap either satisfies the globalfrozen-iterate margin (70) under $\rho _ { K , S }$ almost surely (or its localform at $2 \eta _ { \mathrm { s c } , K } ) ,$ , or satisfies the $f u x e d – Q ^ { \star }$ transfer conditions ofProposition 4.3 at $k = K$ and $u = 2 \eta _ { \mathrm { s c } , K }$ . Set $\Gamma _ { K }$ by the corresponding case:

$$
\Gamma _ { K } : = \left\{ \begin{array} { l l } { C _ { \mathrm { m a r g } } ^ { 1 / p } ( 2 \eta _ { \mathrm { s c } , K } ) ^ { 1 + \alpha / p } , } & { u n d e r t h e f r o z e n - i t e r a t e m a r g i n , } \\ { 2 \eta _ { \mathrm { s c } , K } \left\{ \tau _ { \star } + C _ { \star } ( 2 \eta _ { \mathrm { s c } , K } + 2 \delta _ { K } ^ { \star } ) ^ { \alpha _ { \star } } \right\} ^ { 1 / p } , } & { u n d e r t h e f i x e d - Q ^ { \star } \ t r a n s f e r . } \end{array} \right.\tag{105}
$$

Assume also that,for the same $s \in [ 2 , \infty ]$ and $p = s / ( s - 1 )$ as in Assumption 3.3, deterministic envelopes $d _ { s } ^ { \mathrm { f i n } } ( \ell ) < \infty$ satisfy

$$
\operatorname* { s u p } _ { \pi _ { 1 : \ell } } \| \frac { d \{ \mu _ { S } ^ { \circ } \mathcal { P } _ { \circ } ^ { \pi _ { 1 } } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { \pi _ { \ell } } \} } { d \rho _ { K , S } } \| _ { s , \rho _ { K , S } } \leq d _ { s } ^ { \mathrm { f i n } } ( \ell ) , \qquad 0 \leq \ell < H ,\tag{106}
$$

almost surely, with the empty product at $\ell = 0 .$ Then

$$
\mathbb { E } \Big [ \| V ^ { \star } - V ^ { \widehat { \pi } _ { K } } \| _ { 1 , \mu s } \Big ] \leq \mathcal { B } _ { K } + \sum _ { k = 0 } ^ { K - 1 } w _ { K , k } ^ { ( H ) } e _ { k , p } ^ { \mathrm { B e l l } } + \Gamma _ { K } \sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } d _ { s } ^ { \mathrm { f i n } } ( \ell ) .\tag{107}
$$

If both (103) and (106) hold, the final term can be sharpened depthwise to

$$
\sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } \operatorname* { m i n } \{ 2 \eta _ { \mathrm { s c } , K } \bar { q } \ell , \Gamma _ { K } d _ { s } ^ { \mathrm { f i n } } ( \ell ) \} .\tag{108}
$$

Proof. Condition on the final history and put $d _ { K } : = \mathcal { T } V _ { K } - \mathcal { T } ^ { \widehat { \pi } _ { K } } V _ { K }$ . The score comparison gives $0 \leq d _ { K } \leq 2 \eta _ { \mathrm { s c } , K }$ and $\{ d _ { K } > 0 \} \subseteq \{ \Delta _ { Q } ^ { ( K ) } \leq 2 \eta _ { \mathrm { s c } , K } \}$ . Hence $\| d _ { K } \| _ { p , \rho _ { K , S } } \leq \Gamma _ { K } .$ : use (70) on the support event for the first route and (73) for the second. The exact extra resolvent contribution is $\begin{array} { r l } { ~ } & { { } \sum _ { \ell < H } \gamma ^ { \ell } \int d _ { K } d \{ \mu _ { S } ^ { \circ } ( \mathcal { P } _ { \circ } ^ { \widehat { \pi } _ { K } } ) ^ { \ell } \} } \end{array}$ . Hölder and (106) prove (107). At each depth the same integral is also at most $2 \eta _ { \mathrm { s c } , K } \bar { q } \ell ;$ taking the smaller bound before summing gives (108). There is no additional factor two: the resolvent inserts the single defect $d _ { K }$ □

## 6.2 Consistency of the abstract recursion

Theorem 6.3 (Nonasymptotic generative-reset policy-loss bound under uniform closure $\left[ \mathsf { A } _ { \mathrm { i i d } } \right] )$ . Assume the hypotheses of Theorem 3.18 and Assumption 5.1, specializing Assumption 3.3 to $s = 2$ (and hence $p = 2 )$ , with $H \geq 2 .$ In every block, additionally use the fresh-block protocol (3): conditional on $\mathcal { H } _ { k } ,$ , draw $n _ { k } ~ \ge ~ 2 H$ i.i.d. true-kernel outcomes with the fixed state law $\rho _ { k , S } \equiv \varsigma$ and action law $\beta _ { k } ,$ rebuild $Y _ { i } = r _ { i } + \gamma ( 1 - d _ { i } ) V _ { k } ( \tilde { s } _ { i } ^ { \prime } )$ , and take a measurable �<sub>�</sub> -approximate ERM $V _ { k + 1 } \in \mathcal { F } _ { n _ { k } } ^ { V , 0 }$ with $V _ { k } \in \mathcal { F } _ { n _ { k } } ^ { V }$ . These labels are conditionally unbiased for $G _ { k }$ and $\varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = 0$ . Write $e _ { k , 2 } ^ { \mathrm { B e l l } } : = e _ { k , p } ^ { \mathrm { B e l l } } | _ { p = 2 }$ and sta $\begin{array} { r } { \mathrm { t } _ { k } : = C _ { \mathrm { V r e g } , H } ( \log n _ { k } ) ^ { \frac { 1 + 2 \xi ^ { \star } } { 2 } } m _ { n _ { k } } ^ { \frac { \alpha ^ { \star } - 1 } { 2 } } + \sqrt { 2 \zeta _ { n _ { k } } } , } \end{array}$ , where $m _ { n _ { k } } = \lfloor n _ { k } / H \rfloor$ . Then the fit envelope in (24) may be chosen from Propositions 5.5 and ${ 5 . 6 } ,$ so for every � the Bellman-residual envelope may be chosen to satisfy

$$
e _ { k , 2 } ^ { \mathrm { B e l l } } \ \leq \ \mathrm { s t a t } _ { k } + ( 1 + \sqrt { 2 } ) \big ( \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } + \varepsilon _ { \mathrm { b u f } , k } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } \big ) ,\tag{109}
$$

and hence, for every finite $K \geq 1$

$$
\mathbb { E } \| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } ~ \le ~ \mathcal { B } _ { K } + 2 \phi _ { 2 , K _ { \operatorname* { m a x } \{ 0 , K - H + 1 \} } \le k < K } ^ { ( H ) } e _ { k , 2 , \mathrm { r h s } } ^ { \mathrm { B e l l } } ,\tag{110}
$$

$e _ { k , 2 , \mathrm { r h s } } ^ { \mathrm { B e l l } }$ denoting the right-hand side of (109). Under the global frozen-iterate $( C _ { \mathrm { m a r g } } , \alpha )$ margin, or its local form with $2 \Lambda _ { k } \in [ u _ { 0 } , \bar { u } ] , \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) }$ may be replaced by $\left( 1 - \varepsilon _ { k } \right)$ min $\{ 2 \Lambda _ { k } , \sqrt { C _ { \mathrm { m a r g } } } ( 2 \Lambda _ { k } ) ^ { 1 + \alpha / 2 } \}$ , so the exponent reaches the propagated residual bound at the cost of the constant $1 + { \sqrt { 2 } } .$ The $\it { f i x e d - Q ^ { \star } }$ route uses the $r = 2$ envelope in (77) and preserves the same propagated exponent when $\tau _ { \star } = 0$ and $\delta _ { k } ^ { \star } = { \cal O } ( \Lambda _ { k } )$ on the active window.

Proof: See Appendix A.

Corollary 6.4 (Expected policy-loss consistency with an exact-score oracle $\big [ \mathsf { A } _ { \mathrm { i i d } } \big ] \big )$ . Assume Theorem $6 . 3 ,$ fixed �, and $\phi _ { \mu _ { S } , \rho } ^ { ( H ) } < \infty$ . Initialize $V _ { 0 } \in \mathcal { F } _ { n _ { 0 } } ^ { V }$ ; use a nondecreasing integer schedule $n _ { k } \ge 2 H$ with $n _ { k } \to \infty$ and $\zeta _ { n _ { k } } \to 0 ;$ set $W _ { k } = V _ { k } ,$ ; use the additional exact expectation oracle $Q _ { V } ^ { \mathrm { o r } } = Q _ { V }$ on $\mathcal { F }$ with the common tie rule; draw fresh (SAMPLE, OBS) blocks; and let $\varepsilon _ { k } \to 0$ . Then,for every $K \ge \dot { H } - 1$

$$
\mathbb { E } \| V ^ { \star } - V ^ { \pi \kappa } \| _ { 1 , \mu _ { S } } \leq 2 \phi _ { \mu _ { S } , \rho } ^ { ( H ) } \operatorname* { m a x } _ { K - H < k < K } \left[ \mathrm { s t a t } _ { k } + 2 ( 1 + \sqrt { 2 } ) V _ { \operatorname* { m a x } } \varepsilon _ { k } \right] \longrightarrow 0 .\tag{111}
$$

If the same oracle is used at deployment, its final greedy policy equals $\pi _ { K } ,$ , so the same conclusion holds for that deployed oracle policy

Proof. By nesting, $V _ { k + 1 } \in \mathcal { F } _ { n _ { k } } ^ { V , 0 } \subseteq \mathcal { F } _ { n _ { k } } ^ { V } \subseteq \mathcal { F } _ { n _ { k + 1 } } ^ { V }$ , so the initialization closes the iterate membership required by Theorem 6.3. Fresh observed blocks give $\varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = \varepsilon _ { \mathrm { b u f } , k } = 0$ , while synchronization and the exact score give $\Lambda _ { k } = \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } = 0 ;$ take the safe deterministic exploration envelope $\bar { D } _ { k } ^ { \mathrm { e x p } } = 2 V _ { \mathrm { m a x } }$ . Substitution in (110) proves the displayed bound, whose right-hand side vanishes because $\alpha ^ { \star } < 1$ , fixed � and $n _ { k } \to \infty$ imply $m _ { n _ { k } } = \lfloor n _ { k } / H \rfloor \to \infty .$ $\zeta _ { n _ { k } }  0$ , and $\varepsilon _ { k } \to 0 ;$ oracle final scoring and the common tie rule identify its deployed policy with $\pi _ { K }$ □

Corollary 6.5 (Known deterministic-model consistency $\left[ \mathsf { A } _ { \mathrm { i i d } } \right] )$ . Assume all hypotheses of Corollary 6.4 exceptfor its separate exact-score oracle. Suppose that the joint kernel is deterministic, $\bar { ( } r , \tilde { s } ^ { \prime } ) = ( R ( \tilde { s } , a ) , F ( \tilde { s } , a ) )$ , and that the controller has the exact maps �, �. Then the point score $\widehat { Q } _ { V } ( \tilde { s } , a ) = R ( \tilde { s } , a ) + \gamma V ( F ( \tilde { s } , a ) )$ ) equals $Q _ { V } ^ { \mathrm { o r } } = Q _ { V }$ without a separate expectation oracle, so the same policy-loss consistency conclusion holds

Proof. The deterministic kernel makes the Bellman integral equal evaluation at $F ( \tilde { s } , a )$ ; hence $\eta _ { \mathrm { s c } , k } = \eta _ { \mathrm { s c } , K } = 0$ , and Corollary 6.4 applies verbatim. □

Corollary 6.6 (Deployed-policy convergence and upper neighborhood $[ \mathsf { R } , \mathsf { A } _ { \mathrm { i i d } } ] )$ . Let $\widehat { \pi } _ { K }$ and $\eta _ { \mathrm { s c } , K }$ be as in Theorem 6.1, and fix �. (a) $\bar { [ \mathsf { R } ] } \bar { e } _ { k , p } ^ { \mathrm { B e l l } } \to \bar { 0 }$ implies $\mathbb { E } \| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } \to 0 ;$ if also $\eta _ { \mathrm { s c } , K }  0 ,$ then $\mathbb { E } \| V ^ { \star } - V ^ { \widehat { \pi } _ { K } } \| _ { 1 , \mu _ { S } } \to 0 .$ (b) $\left[ \mathsf { A } _ { \mathrm { i i d } } \right]$ Assume Theorem 6.3 and either the global frozen-iterate margin or a local frozen-iterate margin whose interval contains $2 \Lambda _ { k }$ for every sufficiently large �. Suppose $n _ { k } \to \infty , \zeta _ { n _ { k } } \to 0 , \varepsilon _ { \mathrm { b u f } , k } , \varepsilon _ { k } , \delta _ { V , k } \to 0 ,$ , while eventually $\eta _ { \mathrm { s c } , k } \leq \eta _ { \mathrm { s c } }$ and $\eta _ { \mathrm { s c } , K } \leq \eta _ { \mathrm { s c } }$ . Then

$$
\operatorname* { l i m s u p } _ { \kappa \to \infty } \mathbb { E } [ \| V ^ { \star } - V ^ { \pi \kappa } \| _ { 1 , \mu s } ] \leq R _ { \mathrm { s c } } , \qquad R _ { \mathrm { s c } } : = 2 ( 1 + \sqrt { 2 } ) \phi _ { \mu s , \rho } ^ { ( H ) } \sqrt { C _ { \mathrm { m a r g } } } ( 2 \eta _ { \mathrm { s c } } ) ^ { 1 + \alpha / 2 } ,\tag{112}
$$

$$
\operatorname* { l i m } _ { K \to \infty } \mathbb { E } \Big [ \| V ^ { \star } - V ^ { \widehat { \pi } _ { K } } \| _ { 1 , \mu _ { S } } \Big ] \leq R _ { \mathrm { s c } } + 2 \eta _ { \mathrm { s c } } \sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } \bar { q } _ { \ell } .\tag{113}
$$

Without a margin the same two bounds hold with $R _ { \mathrm { s c } } : = 4 ( 1 + \sqrt { 2 } ) \phi _ { \mu _ { S } , \rho } ^ { ( H ) } \eta _ { \mathrm { s c } } .$ . In particular, $i f \eta _ { \mathrm { s c } , k }  0 a n d \eta _ { \mathrm { s c } , K }  0$ then both the true-score and deployed-policy losses converge to zero.

Proof. In Theorem 3.18, the boundary vanishes for $K \geq H - 1$ and only the last $H - 1$ Bellman residuals have nonzero weights, proving the first claim in (a); Theorem 6.1 proves the second. For (b), Proposition 5.6 and (76) at $r = 2$ give the true-score radius $R _ { \mathrm { s c } }$ after all statistical and nonaction residuals vanish; the survival term of Theorem 6.1 gives the deployed radius. Without a margin, use (75) at $r = 2$ in the same calculation. If $\eta _ { \mathrm { s c } , k }  0$ , that linear bound and the remaining hypotheses give $e _ { k , 2 } ^ { \mathrm { B e l l } } \to 0 ;$ with $\eta _ { \mathrm { s c } , K }  0$ , part (a) then yields the last assertion. □

## 6.3 Convergence conditions for implementations

To apply the policy-loss bounds to $\mathsf { A } _ { \mathrm { r u n } }$ , we bound its six residuals and the final score error, as indicated in (1). For observation-based navigation, Proposition 3.17 permits either a Markov-sufficient simulator or belief state with an exposed clock, or a measurable compression whose visible-target aliasing and score errors are controlled by the corresponding residuals.

Sufficient conditions for FIFO/interleaved SGD. For fixed �, the same policy-loss theory applies to a concrete FIFO/interleaved-SGD implementation when: (i) its state satisfies the standard-Borel controlled Markov condition or its compression residuals are controlled; (ii) finite concentrability constants hold uniformly over its induced replay laws; and (iii)

$$
\operatorname* { m a x } _ { K - H < k < K } \biggl ( \varepsilon _ { \mathrm { f i t } , k } + \varepsilon _ { \mathrm { k e r } , k } + \varepsilon _ { \mathrm { t g t } , k } + \varepsilon _ { \mathrm { b u f } , k } + \varepsilon _ { \mathrm { a c t } , k } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } \biggr ) \longrightarrow 0 , \qquad \eta _ { \mathrm { s c } , K } \longrightarrow 0 .\tag{114}
$$

The unconditional linear bound gives action-residual decay when $\delta _ { V , k }  0$ and $\eta _ { \mathrm { s c } , k }  0$ , while the margin condition upgrades this decay to the sharp exponent. For stored prediction labels, $\varepsilon _ { \mathrm { t g t } , k }$ must also cover the predictor staleness in (29); current score accuracy and target-network staleness alone do not control it. Setting $\varepsilon _ { \mathrm { k e r } , k } = 0$ for stored labels requires a justification such as the metadata-conditioned law (27). Bounds for a particular replay scheme, optimizer, score estimator, or state representation enter the propagation and deployment theorems through the corresponding residuals. Their decay gives convergence through (114).

## 6.4 Related work

Fitted value iteration. Munos and Szepesvári analyze fitted value iteration by estimating every action’s continuation and then maximizing [1]. The population response in our state-only reset construction instead averages the executed action and carries a finite clock. It also differs from the action-indexed response used in fitted � regression [25, 26] Once this operator distinction is isolated, the propagation analysis follows the $L ^ { p }$ approximate-dynamic- programming tradition [27–29]. As in that literature, stability depends on the sampling and update structure [30, 31].

SARSA and Expected SARSA. SARSA provides the closest target semantics. A current observed-successor statevalue target under fresh sampling averages the acting law, whereas Expected SARSA updates $Q ( s , a )$ and averages the next action [7]. Convergence results for SARSA with function approximation require smooth improvement, ergodicity, or linear approximation [32–34]. Here the acting law’s same-state departure from its frozen-target greedy comparator remains a separate residual.

Neural � and TD analyses. Neural fitted �-iteration separates statistical and iterative error [23]. Its action-indexed response and the scalar executed-action response studied here define distinct population operators. Neural TD treats fixed-policy evaluation [35], while neural �-learning and DQN use different heads and sampling or update hypotheses [36, 37]. Our analysis identifies replay, scalar-target, and policy-drift residuals as the additional quantities for a deep-� implementation analysis. DQN, Double � bias correction, and residual-gradient TD address related algorithmic objects [31, 38, 39].

Action gaps. Action-gap regularity can convert uniform value or score approximation into superlinear greedy-policy bounds [14, 15]. The classical fixed-optimal positive-gap law controls $\{ 0 < \Delta _ { Q } ^ { \star } \leq u \}$ and can retain mass on optimal ties. The condition used here controls $\{ \Delta _ { Q } ^ { ( k ) } \leq u \}$ , including ties, uniformly over the random frozen iterates and replay laws. Proposition 4.3 connects the two: a fixed-�<sup>⋆</sup> margin and an iterate score tube yield a shifted frozen-gap bound. When the tie mass is zero and the tube is on the action-error scale, the exponent is preserved. The resulting Tsybakov-type bound [13] has one-step exponent $1 + \alpha / p$ under conjugate $L ^ { s } / L ^ { \hat { p } }$ coverage; Proposition 4.7 attains each intermediate exponent.

Replay and dependent data. Antos et al. require a stationary exponentially mixing path for their dependent fitted-policy analysis [40]. Replay-buffer, �-learning, and TD analyses impose different processes [41–43]. Their process-specific techniques provide tools for bounding the named FIFO and SGD residuals in (1). In our notation, $\varepsilon _ { \mathrm { b u f } , k }$ measures a same-state operator distance, while concentrability controls the relation between replay and evaluation occupancy laws [44].

## 7 Conclusion

We established a finite-horizon convergence framework that converts the six deep-�-learning residuals into expected policy loss. For fixed $H ,$ , only the last $H - 1$ update blocks carry residual weight, together with an initialization term for shorter runs. For $\dot { K _ { H } } \ge H - 1$ , the near-unit one-state-per-level witness attains the optimal shared-law propagation-coefficient order $\Theta ( H ^ { 5 / 2 } )$ at $s = 2 \cdot$ ; bounded direct-level coefficients give $O ( H ^ { 2 } )$ . A frozen-gap margin changes the action term to exponent $1 + \alpha ( 1 - 1 / s )$ , and the one-step construction attains that exponent. The fixed- $\cdot Q ^ { \star }$ transfer keeps the tie mass visible. Survival and final-law arguments extend the bound to the policy selected by the implemented score.

The matched-budget result gives the unconstrained continuous optimum, the constrained water-filling solution, and an integer schedule within a factor $2 ^ { \nu }$ of the latter. On the near-unit one-state-per-level witness, root-� rates give a clock term $H ^ { 3 } / \sqrt { \mathsf { N } }$ before regression constants; under equal, sufficiently large terminal-window label budgets, direct tabular reset improves the statistical bound by a factor of order $\sqrt { H }$ compared with a shared �-state fit. Bellman–Hölder closure and the clock-routed ReLU class give the fixed-� neural rate, while the tabular case gives a log-free expected fit rate. These results prove expected policy-loss consistency for the exact-score generative-reset approximate-ERM procedure at fixed $\bar { H . }$ For FIFO/interleaved SGD, (114) gives an explicit residual-decay criterion linking replay, optimization, score-estimation, and representation rates to full policy-loss convergence.

## References

[1] Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 9:815–857, 2008.

[2] Yu Fan Chen, Miao Liu, Michael Everett, and Jonathan P. How. Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning. In Proceedings ofthe IEEE International Conference on Robotics and Automation, pages 285–292, 2017.

[3] Yu Fan Chen, Michael Everett, Miao Liu, and Jonathan P. How. Socially aware motion planning with deep reinforcement learning. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems, 2017. URL https://arxiv.org/abs/1703.08862.

[4] Changan Chen, Yuejiang Liu, Sven Kreiss, and Alexandre Alahi. Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning. In Proceedings ofthe IEEE International Conference on Robotics and Automation, pages 6015–6022, 2019.

[5] Yury Kolomeytsev and Dmitry Golembiovsky. Robot navigation with entity-based collision avoidance using deep reinforcement learning. arXiv preprint arXiv:2408.14183, 2024. URL https://arxiv.org/abs/2408.14183.

[6] Yury Kolomeytsev and Dmitry Golembiovsky. Hybrid motion planning with deep reinforcement learning for mobile robot navigation. arXiv preprint arXiv:2512.24651, 2025. URL https://arxiv.org/abs/2512.24651.

[7] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2 edition, 2018.

[8] Erwin Kreyszig. Introductory Functional Analysis with Applications. Wiley, 1989.

[9] Dimitri P. Bertsekas and Steven E. Shreve. Stochastic Optimal Control: The Discrete-Time Case. Academic Press, 1978.

[10] Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming. Athena Scientific, 1996.

[11] Onésimo Hernández-Lerma and Jean B. Lasserre. Discrete-Time Markov Control Processes: Basic Optimality Criteria. Springer, 1996.

[12] Enno Mammen and Alexandre B. Tsybakov. Smooth discrimination analysis. The Annals ofStatistics, 27(6): 1808–1829, 1999. doi: 10.1214/aos/1017939240.

[13] Alexandre B. Tsybakov. Optimal aggregation of classifiers in statistical learning. The Annals ofStatistics, 32(1): 135–166, 2004. doi: 10.1214/aos/1079120131.

[14] Amir-massoud Farahmand. Action-gap phenomenon in reinforcement learning. In Advances in Neural Information Processing Systems, volume 24, pages 172–180, 2011.

[15] Marc G. Bellemare, Georg Ostrovski, Arthur Guez, Philip S. Thomas, and Rémi Munos. Increasing the action gap: New operators for reinforcement learning. In Proceedings ofthe Thirtieth AAAI Conference on Artificial Intelligence, pages 1476–1483, 2016.

[16] Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. The Annals ofStatistics, 48(4):1875–1897, 2020. doi: 10.1214/19-AOS1875.

[17] Johannes Schmidt-Hieber and Don Vu. Correction to “nonparametric regression using deep neural networks with relu activation function”. The Annals ofStatistics, 52(1):413–414, 2024. doi: 10.1214/24-AOS2351.

[18] László Györfi, Michael Kohler, Adam Krzyzak, and Harro Walk. ˙ A Distribution-Free Theory of Nonparametric Regression. Springer Series in Statistics. Springer, 2002.

[19] Ruosong Wang, Dean P. Foster, and Sham M. Kakade. What are the statistical limits of offline RL with linear function approximation? In International Conference on Learning Representations, 2021. URL https: //arxiv.org/abs/2010.11895.

[20] Andrea Zanette. Exponential lower bounds for batch reinforcement learning: Batch RL can be exponentially harder than online RL. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 12287–12297. PMLR, 2021.

[21] Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 1042–1051. PMLR, 2019. URL https://proceedings.mlr.press/v97/chen19e.html.

[22] Tengyang Xie and Nan Jiang. Batch value-function approximation with only realizability. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11404–11413. PMLR, 2021.

[23] Jianqing Fan, Zhaoran Wang, Yuchen Xie, and Zhuoran Yang. A theoretical analysis of deep Q-learning, 2020. URL https://arxiv.org/abs/1901.00137v3. arXiv:1901.00137v3.

[24] Satinder P. Singh and Richard C. Yee. An upper bound on the loss from approximate optimal-value functions. Machine Learning, 16(3):227–233, 1994.

[25] Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research, 6:503–556, 2005. URL https://jmlr.org/papers/v6/ernst05a.html.

[26] Martin Riedmiller. Neural fitted q iteration: First experiences with a data efficient neural reinforcement learning method. In Proceedings ofthe European Conference on Machine Learning, pages 317–328, 2005. doi: 10.1007/ 11564096\_32.

[27] Amir-massoud Farahmand, Rémi Munos, and Csaba Szepesvári. Error propagation for approximate policy and value iteration. In Advances in Neural Information Processing Systems, volume 23, pages 568–576, 2010.

[28] Rémi Munos. Performance bounds in �<sub>�</sub>-norm for approximate value iteration. SIAM Journal on Control and Optimization, 46(2):541–561, 2007. doi: 10.1137/040614384.

[29] Bruno Scherrer, Mohammad Ghavamzadeh, Victor Gabillon, Boris Lesner, and Matthieu Geist. Approximate modified policy iteration and its application to the game of Tetris. Journal of Machine Learning Research, 16: 1629–1676, 2015. URL https://jmlr.org/papers/v16/scherrer15a.html.

[30] John N. Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control, 42(5):674–690, 1997. doi: 10.1109/9.580874.

[31] Leemon Baird. Residual algorithms: Reinforcement learning with function approximation. In Proceedings of the Twelfth International Conference on Machine Learning, pages 30–37. Morgan Kaufmann, 1995.

[32] Theodore J. Perkins and Doina Precup. A convergent form of approximate policy iteration. In Advances in Neural Information Processing Systems, volume 15, pages 1627–1634, 2002.

[33] Francisco S. Melo, Sean P. Meyn, and M. Isabel Ribeiro. An analysis of reinforcement learning with function approximation. In Proceedings ofthe 25th International Conference on Machine Learning, pages 664–671. ACM, 2008. doi: 10.1145/1390156.1390240.

[34] Shaofeng Zou, Tengyu Xu, and Yingbin Liang. Finite-sample analysis for SARSA with linear function approximation. In Advances in Neural Information Processing Systems, volume 32, 2019.

[35] Qi Cai, Zhuoran Yang, Jason D. Lee, and Zhaoran Wang. Neural temporal-difference learning converges to global optima. In Advances in Neural Information Processing Systems, volume 32, 2019.

[36] Pan Xu and Quanquan Gu. A finite-time analysis of q-learning with neural network function approximation. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 10555–10565. PMLR, 2020. URL https://proceedings.mlr.press/ v119/xu20c.html.

[37] Shuai Zhang, Hongkang Li, Meng Wang, Miao Liu, Pin-Yu Chen, Songtao Lu, Sijia Liu, Keerthiram Murugesan, and Subhajit Chaudhury. On the convergence and sample complexity analysis of deep q-networks with �-greedy exploration. In Advances in Neural Information Processing Systems, volume 36, 2023.

[38] Volodymyr Mnih et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015. doi: 10.1038/nature14236.

[39] Hado van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double Q-learning. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, pages 2094–2100, 2016. URL https: //arxiv.org/abs/1509.06461.

[40] András Antos, Csaba Szepesvári, and Rémi Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning, 71(1):89–129, 2008.

[41] Shirli Di Castro Shashua, Shie Mannor, and Dotan Di Castro. Analysis of stochastic processes through replay buffers. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 5039–5060. PMLR, 2022.

[42] Liran Szlak and Ohad Shamir. Convergence results for q-learning with experience replay, 2021. URL https: //arxiv.org/abs/2112.04213.

[43] Han-Dong Lim and Donghwan Lee. Finite-time analysis of temporal difference learning with experience replay. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=A5ulGfDBON.

[44] Sham M. Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the Nineteenth International Conference on Machine Learning, pages 267–274, 2002.

## A Deferred proofs

## Structural and target calculations.

Proof of Lemma 2.3. At level ℎ the map projects R onto

$$
[ - V _ { \operatorname* { m a x } } ^ { ( h ) } , V _ { \operatorname* { m a x } } ^ { ( h ) } ] .
$$

It is therefore 1-Lipschitz and idempotent. If � lies in the interval, the scalar projection inequality gives $| \mathrm { c l i p } _ { h } ( x ) - g | \leq$ |� − �| pointwise; integration proves the $L ^ { 2 }$ metric-projection assertion, up to the usual a.e. identification. Applying the scalar Lipschitz inequality statewise proves the supremum- and $L ^ { 2 } .$ -nonexpansiveness claims. The score bound follows from (10) and

$$
| Q _ { V } ( s , h , a ) | \leq R _ { \operatorname* { m a x } } + \gamma V _ { \operatorname* { m a x } } ^ { ( h - 1 ) } = V _ { \operatorname* { m a x } } ^ { ( h ) } .
$$

The same recursion bounds every observed-transition label and, after maximization over actions, places $\mathcal { T V }$ in the level-wise band. Nonexpansiveness maps every supremum-norm �-net of a raw class to a �-net of its clipped image, proving the covering-number assertion. Finally, because every Bellman target $\mathcal { T V }$ lies in the band, the pointwise projection inequality proves the Bellman-approximation assertion. □

The statewise version of the disintegration (Assumption 3.11). A disintegration is determined only $\mathsf { M } _ { k , S } – \mathsf { a } . \mathsf { e }$ ., whereas $\varepsilon _ { \mathrm { b u f } , k }$ and $\varepsilon _ { \mathrm { a c t } , k } ^ { ( r ) }$ are norms under $\rho _ { k , S }$ , so a version must be fixed on all of $\widetilde { S } ^ { \circ }$ . Each round’s policy is $\varepsilon _ { k } { \mathrm { - g r e e d y } }$ for a score defined at every state, so

$$
\begin{array} { r l r } & { } & { f _ { k , j } : = \displaystyle \frac { d \left( \omega _ { k , j } \mu _ { k , j } \right) } { d \mathsf { M } _ { k , S } } , \qquad \displaystyle w _ { k , j } ( \tilde { s } ) : = \frac { f _ { k , j } \left( \tilde { s } \right) } { \sum _ { i } f _ { k , i } \left( \tilde { s } \right) } , } \\ & { } & { \bar { \pi } _ { k } ^ { \mathrm { o n } } ( \cdot  { | \begin{array} { l } { \tilde { s } } \end{array} \rangle } : = \displaystyle \sum _ { j } w _ { k , j } \left( \tilde { s } \right) \pi _ { k , j } ^ { \mathrm { o n } } ( \cdot  { | \begin{array} { l } { \tilde { s } } \end{array} \rangle } , } \end{array}\tag{115}
$$

using the bounded density versions of Lemma 2.2 and $w _ { k , j } ( \tilde { s } ) : = \omega _ { k , j }$ where the denominator vanishes. This is a statewise Markov kernel agreeing with the disintegration $\mathsf { M } _ { k , S } – \mathsf { a } . \mathsf { e } .$ ; the same convention fixes $\beta _ { k } ^ { \mathrm { r e p } }$ . Without it, a residual norm under a law not dominated by $\mathsf { M } _ { k , S }$ would depend on values on an $\mathsf { M } _ { k , S } \mathrm { - n u l l }$ set, which the disintegration does not determine; this is why Assumption 3.11 requires $\rho _ { k , S } \ll \mathsf { M } _ { k , S }$ for the $\mathsf { A } _ { \mathrm { r u n } }$ comparison. Corollary 4.6 is stated for exactly this state-dependent mixture.

Lemma A.1 (Detailed target table for Lemma 3.2 [ R ]). Let $A \in \{ 0 , \ldots , k \}$ be the record’s age, the number oftarget copies between the writing ofa replayed record and the current block, and ${ \bar { \delta } } _ { k } ( A ) : = \| V _ { k } - { \bar { V _ { k - A } } } \| _ { \infty } ; A$ is afunction ofthe record and hence in general state-dependent. Use the current and writing-time predictors $\widehat { P } _ { k }$ and $\widehat { P } _ { \mathrm { w r i t e } }$ and the

metadata $M _ { k }$ ofLemma 3.2. Thefour cells ofslot (S6) are

<table><tr><td>WHEN</td><td>FROM</td><td>Y</td></tr><tr><td>SAMPLE</td><td>OBS</td><td> $\overline { { r + \gamma ( 1 - d ) V _ { k } ( \tilde { s } ^ { \prime } ) } }$ </td></tr><tr><td>SAMPLE</td><td>PRED</td><td> $r + \gamma ( 1 - d ) V _ { k } ( \widehat { P } _ { k } ( \widetilde { s } , a ) )$ </td></tr><tr><td>STORE</td><td>OBS</td><td> $r + \gamma ( 1 - d ) V _ { k - A } ( \tilde { s } ^ { \prime } )$ </td></tr><tr><td>STORE</td><td>PRED</td><td> $r + \gamma ( 1 - d ) V _ { k - A } ( \widehat { P } _ { \mathrm { w r i t e } } ( \tilde { s } , a ) )$ </td></tr></table>

(116)

in each ofwhich the reward � and the mask � are the observed ones; only the bootstrapping network and evaluation successor change. The ideal response uses the same row with $Z ^ { \circ }$ in place of $^ { \cdot } Z ,$ retaining the writing-time parameters and age as in (23). Define the continuation component ofthe score error by

$$
\eta _ { k } ^ { \mathrm { c o n t } } : = \operatorname* { s u p } _ { V \in \mathcal { F } } \operatorname* { s u p } _ { \widetilde { s } , a } \gamma \left| V ( \widehat { P } _ { k } ( \widetilde { s } , a ) ) - \int V d \mathcal { P } ( \cdot \mid \widetilde { s } , a ) \right| .
$$

Then $\eta _ { k } ^ { \mathrm { c o n t } } \leq \eta _ { \mathrm { s c } , k } ,$ , without afactor 2, whenever the point score is the score bounded in (21). Under a different slot (S2) score, retain $\eta _ { k } ^ { \mathrm { c o n t } }$ separately. For the two PRED cells, and only those, assume in addition that the terminal event is decided by the state and action,

(T)

$$
\mathbb { P } \big [ d = 1 \big | \tilde { s } , a \big ] \in \{ 0 , 1 \} ,\tag{117}
$$

for $\rho _ { k , S } ( d { \tilde { s } } ) \beta _ { k } ^ { \mathrm { r e p } } ( d a \mid { \tilde { s } } )$ -almost every $( \tilde { s } , a )$ , which requires thefull conditioning state andjoint dynamics to determine the terminal event. Condition (T) is therefore an explicit assumption on this joint structure. Then the actual targetconstruction distance obeys

$$
\begin{array} { r l } & { \big \| { G } _ { k } ^ { \scriptscriptstyle \mathrm { i d e a l } } - \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } \big \| _ { 2 , \rho _ { k , s } } \leq { \mathbf 1 } \big \{ \mathrm { F R O M } = \mathrm { P R E D } \big \} \eta _ { k } ^ { \scriptscriptstyle \mathrm { c o n t } } } \\ & { \qquad + { \mathbf 1 } \big \{ \mathrm { W H E N } = \mathrm { S T O R E } \big \} \gamma \big \| \mathbb { E } \big [ \bar { \delta } _ { k } ( A ) \ | \ { S } , \mathcal { T } _ { k } ] \big \| _ { 2 , \rho _ { k , s } } } \\ & { \qquad + { \mathbf 1 } \big \{ \big ( \mathrm { W H E N } , \mathrm { F R O M } \big ) = \big ( \mathrm { S T O R E } , \mathrm { P R E D } \big ) \big \} \| C _ { \mathrm { p r e d } , k } \| _ { 2 , \rho _ { k , s } } . } \end{array}\tag{118}
$$

so (26) holds with $\varepsilon _ { \mathrm { t g t } , k }$ equal to any deterministic almost-sure majorant of the right-hand side. Without (T), set $p _ { T } ( \tilde { s } , a ) : = \mathbb { P } [ d = 1 | \tilde { s } , a ]$ and

$$
C _ { \mathrm { m a s k } , k } ( { \tilde { s } } ) : = \gamma \int p _ { T } ( { \tilde { s } } , a ) { \big | } V _ { k } { \big ( } { \widehat P } _ { k } ( { \tilde { s } } , a ) { \big ) } { \big | } \beta _ { k } ^ { \mathrm { r e p } } ( d a \mid { \tilde { s } } ) ,
$$

$$
\| C _ { \mathrm { m a s k } , k } \| _ { 2 , \rho _ { k , S } } \leq \gamma V _ { \operatorname* { m a x } } \left\| \int p _ { T } ( \cdot , a ) \beta _ { k } ^ { \mathrm { r e p } } ( d a \mid \cdot ) \right\| _ { 2 , \rho _ { k , S } } .\tag{119}
$$

The right side of (118) then acquires $\mathbf { 1 } \big \{ \mathrm { F R O M } = \mathrm { P R E D } \big \} \| C _ { \mathrm { m a s k } , k } \| _ { 2 , \rho _ { k , S } ; }$ , which $\eta _ { k } ^ { \mathrm { c o n t } }$ does not cover. Thus the replay action average is taken before the $L ^ { 2 } ( \rho _ { k , S } )$ norm. The conditional expectation in either staleness term may not be replaced by an unconditional one, since the age and writing-time predictor can depend on the replayed state. For the target-network term, conditional Jensen gives the $\mathcal { T } _ { k }$ -measurable majorant $\gamma ( \mathbb { E } [ \bar { \delta } _ { k } ( A ) ^ { 2 } \mid \mathcal { T } _ { k } ] ) ^ { 1 / 2 } ,$ ; the conditional essential supremum of $\dot { \gamma } \bar { \delta } _ { k } ( A )$ is another. A deterministic envelope must dominate either choice almost surely.

ProofofLemma A.1. When the point score is installed, $\eta _ { k } ^ { \mathrm { c o n t } } \leq \eta _ { \mathrm { s c } , k }$ without the factor 2 that the triangle inequality would give: since $\widehat { Q } _ { V } - Q _ { V } = ( \widehat { R } - R ) + \gamma ( V \circ \widehat { P } _ { k } - \int V d \mathcal { P } )$ and $\mathcal { F }$ is symmetric under $V \mapsto - V$ by construction, evaluating at $V$ and at −� and subtracting cancels the reward term, leaving $2 \gamma | V \circ \widehat { P } _ { k } - \textstyle \int V d \mathcal { P } | \le 2 \eta _ { \mathrm { s c } , k }$ by (21).

Fix a cell of (116) and condition on $( S , \mathcal { T } _ { k } )$ . Under (SAMPLE, OBS) the ideal fresh-redraw label’s conditional mean under the true kernel is $( \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k } ) ( \tilde { s } )$ by (23) and the terminal convention, so $G _ { k } ^ { \mathrm { i d e a l } } = \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k }$ and the left-hand side vanishes.

In the ideal redraw, changing FROM to PRED leaves $r ^ { \circ }$ and $d ^ { \circ }$ untouched. Both objects compared are conditional means, so the comparison is made after conditioning, and the redrawn mask is random: the PRED label has conditional mean $R ( \tilde { s } , a ) + \gamma ( 1 - p _ { T } ( \tilde { s } , a ) ) V _ { k } ( \widehat { P } _ { k } ( \tilde { s } , a ) )$ and the OBS label $\begin{array} { r } { R ( \tilde { s } , a ) + \gamma \int V _ { k } d \mathcal { P } } \end{array}$ , the terminal convention having absorbed the mask on that side. Adding and subtracting $\gamma V _ { k } \big ( \widehat { P } _ { k } \big ( \tilde { s } , a \big ) \big )$ ) bounds their difference by $\eta _ { k } ^ { \mathrm { c o n t } } + \gamma p _ { T } ( \widetilde s , a ) | V _ { k } ( \widehat { P } _ { k } ( \widetilde s , a ) ) |$ whose average under $\beta _ { k } ^ { \mathrm { r e p } } ( \breve { d } a \mid \tilde { s } )$ is bounded by $\eta _ { k } ^ { \mathrm { c o n t } } + \dot { C } _ { \mathrm { m a s k } , k } ( \tilde { s } )$ . Taking the state norm gives (119). Under (T) a direct cellwise comparison removes this correction: where $p _ { T } ( \tilde { s } , a ) = 0$ by inspection, and where it is 1 the fresh successor is $\tilde { s } _ { \mathrm { t e r m } }$ almost surely, so $\begin{array} { r } { \int V _ { k } d \mathcal { P } = 0 } \end{array}$ while $( 1 - d ^ { \circ } )$ annihilates the predicted continuation. Only the continuation component of the score error is charged, the reward being observed and not modelled.

For STORE $/ \mathrm { o B }$ , condition first on $\left( S , \mathbf { a } , M _ { k } , \mathcal { I } _ { k } \right)$ and use the same fresh outcome for the stored and sample-time labels. Their pointwise difference is at most $\gamma \bar { \delta } _ { k } ( A )$ , since $| 1 - d ^ { \circ } | \leq 1$ . For STORE/PRED, there is also a change of predictor:

$$
\begin{array} { r l } & { V _ { k - A } ( \widehat { P } _ { \mathrm { w r i t e } } ( S , \mathbf { a } ) ) - V _ { k } ( \widehat { P } _ { k } ( S , \mathbf { a } ) ) } \\ & { \quad = [ V _ { k - A } - V _ { k } ] ( \widehat { P } _ { \mathrm { w r i t e } } ( S , \mathbf { a } ) ) } \\ & { \quad \quad + V _ { k } ( \widehat { P } _ { \mathrm { w r i t e } } ( S , \mathbf { a } ) ) - V _ { k } ( \widehat { P } _ { k } ( S , \mathbf { a } ) ) . } \end{array}
$$

Multiplication by $\gamma ( 1 - d ^ { \circ } )$ , the triangle inequality, and $| 1 - d ^ { \circ } | \leq 1$ bound the conditional mean of the absolute label difference by $\gamma \mathbb { E } [ \bar { \delta } _ { k } ( A ) \mid S , \mathcal { T } _ { k } ] \bar { + } C _ { \mathrm { p r e d } , k } ( S )$ . Here the action and metadata are averaged conditional on $( S , \mathcal { T } _ { k } )$ Taking the state norm gives the second and third terms of (118). The stated target-network majorants follow from conditional Jensen and the essential supremum bound. Combining with the sample-time comparison proves the result, including the general terminal-mask correction. □

Why both stored-label corrections are needed. Two finite-state examples isolate the issues. Take $H = 2 , R _ { \mathrm { m a x } } = 1$ a level-two state �, zero reward there, and identical actions; all level-one states terminate at the next step. First, let the true successor of � be �, let the writing-time predictor return �, and let the current predictor be exact. Set $V _ { k - A } = V _ { k } = V$ with $V ( u ) = 1$ and $V ( v ) = - 1$ . Then a stored prediction label is $\gamma$ while $( \bar { T } ^ { \bar { \beta } _ { k } ^ { \mathrm { r e p } } } V _ { k } ) ( x ) = - \gamma$ . Both current continuation error and target-network staleness vanish, but $C _ { \mathrm { p r e d } , k } ( x ) = 2 \gamma$ , exactly the missing distance.

Second, take $k \geq 1$ , let the true successor law at � be uniform on $z _ { + }$ and $z _ { - } ,$ let $V _ { k } = 0$ , and set $V _ { k - 1 } ( z _ { + } ) = 1$ $V _ { k - 1 } ( z _ { - } ) = - 1$ . With fixed $\mathcal { T } _ { k } .$ , suppose a selected stored observation record has $( A , \tilde { s } ^ { \prime } ) = ( 1 , z _ { + } )$ or $( 0 , z _ { - } )$ , each with probability $1 / 2$ . Its outcome marginal given $( S , \mathbf { a } , \mathcal { T } _ { k } )$ is the true kernel, but $G _ { k } ( x ) = \gamma / 2$ . Retaining � and independently redrawing the successor gives $G _ { k } ^ { \mathrm { i d e a l } } ( x ) = 0 .$ . Thus conditioning without metadata cannot justify a zero kernel residual. For $\mathsf { A } _ { \mathrm { i i d } }$ , the label parameters are fixed before each fresh draw and the switches are SAMPLE/OBS, so both residuals remain zero.

Proof of Lemma 3.10. The decomposition follows the standard fitted-value-iteration argument [1, Lemmas 3–4];   
specific here are the truncation of Step 3 and the weight sum of Step 4.

Step 1 (recursion). Let $z _ { k } : = V ^ { \star } - V _ { k } , \ s \mathbf { o } \ z _ { k + 1 } = ( { \mathcal { T } } V ^ { \star } - { \mathcal { T } } V _ { k } ) - e _ { k }$ . Lemma 3.9 at $\left( V , W \right) = \left( V ^ { \star } , V _ { k } \right)$ gives a substochastic $\mathcal { P } _ { \circ } ^ { ( k ) } \mathrel { \mathop : } = \mathcal { P } _ { \circ } ^ { V ^ { \star } , V _ { k } }$ with $\vert T V ^ { \star } - \mathcal { T } V _ { k } \vert \leq \gamma \mathcal { P } _ { \circ } ^ { ( k ) } \vert z _ { k } \vert$ , so $| z _ { k + 1 } | \leq \gamma \mathcal { P } _ { \circ } ^ { ( k ) } | z _ { k } | + | e _ { k } |$ pointwise, and unrolling from � to 0,

$$
| z _ { K } | \leq \gamma ^ { K } \mathcal { P } _ { \circ } ^ { ( K - 1 ) } . . . \mathcal { P } _ { \circ } ^ { ( 0 ) } | z _ { 0 } | + \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { K - 1 - k } \mathcal { P } _ { \circ } ^ { ( K - 1 ) } . . . \mathcal { P } _ { \circ } ^ { ( k + 1 ) } | e _ { k } | .\tag{120}
$$

Step 2 (loss resolvent). With $\pi _ { K }$ greedy for $V _ { K }$ and $\pi ^ { \star }$ for $V ^ { \star } \colon$ from $V ^ { \star } = \mathcal { T } ^ { \pi ^ { \star } } V ^ { \star } , V ^ { \pi _ { K } } = \mathcal { T } ^ { \pi _ { K } } V ^ { \pi _ { K } }$ , the greedy inequality $\mathcal { T } ^ { \pi ^ { \star } } V _ { K } \le \mathcal { T } V _ { K } = \mathcal { T } ^ { \pi _ { K } } V _ { K }$ and the splitting $V _ { K } - V ^ { \pi \kappa } = ( V _ { K } - V ^ { \star } ) + ( V ^ { \star } - V ^ { \pi \kappa } ) , ( I - \gamma \mathcal P _ { \circ } ^ { \pi \kappa } ) ( V ^ { \star } - V ^ { \pi \kappa } )$ $V ^ { \pi _ { K } } ) \leq \gamma ( \mathcal { P } _ { \circ } ^ { \pi ^ { \star } } - \mathcal { P } _ { \circ } ^ { \pi _ { K } } ) z _ { K }$ . The resolvent $\begin{array} { r } { ( I - \gamma \mathcal P _ { \circ } ^ { \pi _ { K } } ) ^ { - 1 } = \sum _ { \ell > 0 } \gamma ^ { \ell } ( \mathcal P _ { \circ } ^ { \pi _ { K } } ) ^ { \ell } } \end{array}$ is nonnegative; applying it and $\pm z _ { K } \le$ $\left| z _ { K } \right|$ yields the standard loss-resolvent bound in substochastic form, $\begin{array} { r } { V ^ { \star } - V ^ { \pi \kappa } \leq \sum _ { \ell \geq 0 } \gamma ^ { \ell + 1 } ( \mathcal { P } _ { \circ } ^ { \pi \kappa } ) ^ { \ell } ( \mathcal { P } _ { \circ } ^ { \pi ^ { \star } } + \mathcal { P } _ { \circ } ^ { \pi \kappa } ) | z _ { K } | } \end{array}$ Step 3 (truncation). Substituting (120) gives, for each $| e _ { k } |$ , two families of kernel products of length $m = \ell + ( K - k )$ and weight $\gamma ^ { m }$ . By the clock decrement (15) every $\mathcal { P } _ { \circ } ^ { \pi }$ strictly decreases $h ,$ , so products of length $m \geq H$ vanish (the policies need not coincide), and the weight range is � $< H$ . The families multiplying |�<sub>0</sub>| have length at least $K + 1$ and vanish once $K \geq H - 1$

Step 4 (the exact realized occupancies). For $m = \ell + K - k < H$ , define the two subprobability measures

$$
\begin{array} { r l } & { \nu _ { K , k , \ell } ^ { 1 } : = \mu _ { S } ^ { \circ } ( \mathcal { P } _ { \circ } ^ { \pi _ { K } } ) ^ { \ell } \mathcal { P } _ { \circ } ^ { \pi ^ { \star } } \mathcal { P } _ { \circ } ^ { ( K - 1 ) } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { ( k + 1 ) } , } \\ & { \nu _ { K , k , \ell } ^ { 2 } : = \mu _ { S } ^ { \circ } ( \mathcal { P } _ { \circ } ^ { \pi _ { K } } ) ^ { \ell + 1 } \mathcal { P } _ { \circ } ^ { ( K - 1 ) } \cdot \cdot \cdot \mathcal { P } _ { \circ } ^ { ( k + 1 ) } , } \end{array}
$$

where the final product is the identity for $k = K - 1$ . For conjugate $s \in [ 2 , \infty ]$ and $p = s / ( s - 1 )$ , suppose only these realized measures are absolutely continuous with respect to $\rho _ { k , S }$ , and put $d _ { K , k , \ell , s } ^ { b } : = \| d \nu _ { K , k , \ell } ^ { b } / d \rho _ { k , S } \| _ { s , \rho _ { k , S } }$ . Hölder’s inequality applied separately to the two branches gives the exact pathwise coefficient form

$$
\| V ^ { \star } - V ^ { \pi \kappa } \| _ { 1 , \mu _ { S } } \leq B _ { K } + \sum _ { k < K } \sum _ { \ell \geq 0 ; \atop \ell + K - k < H } \gamma ^ { \ell + K - k } \big ( d _ { K , k , \ell , s } ^ { 1 } + d _ { K , k , \ell , s } ^ { 2 } \big ) \| e _ { k } \| _ { p , \rho _ { k , S } } .\tag{121}
$$

Thus the proof consumes coverage only for the two displayed families. Their coefficients remain multiplied by the residual norms; no independence or product-of-expectations step is valid in general.

Step 5 (deterministic envelope, boundary, and count). Every constructed kernel is a measurable policy kernel (Lemma 3.9). Assumption 3.3 gives $d _ { K , k , \ell , s } ^ { 1 } , d _ { K , k , \ell , s } ^ { 2 } \ \leq \ d _ { s } ( \bar { m } )$ , which reduces (121) to the residual sum in (39). For the initialization term, start on clock slice ℎ. A branch of total length � vanishes for $m \geq h$ and otherwise ends on slice $h - m$ , where $| z _ { 0 } | \le D _ { 0 , h - m }$ almost surely. Summing the two branches against the initial masses $p _ { h }$ bounds their contribution by the deterministic $\boldsymbol { B } _ { K }$ in (37). Level-wise clipping permits the choice $D _ { 0 , j } = 2 V _ { \operatorname* { m a x } } ^ { ( j ) }$ ; the boundary vanishes for $K \ge \dot { H } - 1$ regardless of that choice.

Finally, the coefficient of $\| e _ { k } \| _ { p , \rho _ { k , S } } \mathrm { i s } w _ { K , k } ^ { ( H ) }$ . With $m = \ell + K - k$ , a fixed $1 \leq m < H$ admits exactly min $\{ K , m \}$ indices �. Fubini–Tonelli therefore gives $\begin{array} { r } { \sum _ { k < K } w _ { K , k } ^ { ( H ) } = 2 \sum _ { m < H } \operatorname* { m i n } \{ K , m \} \gamma ^ { m } d _ { s } ( m ) = 2 \phi _ { s , K } ^ { ( H ) } } \end{array}$ . The weights are deterministic, so expectations pass term by term. □

ProofofProposition 4.8. MDP and laws. All unspecified rewards are zero. Use $O = S$ , the full clipped tabular class, and exact fresh population (SAMPLE, OBS) updates at every nonterminal state. Nonterminal states are � at $h = 3 , y _ { 1 } , y _ { 2 }$ at $h = 2$ , and $z _ { 1 } , z _ { 2 }$ , � at $h = 1$ . Transitions are deterministic and rewards are nonzero only at $h = 1 \colon s : u _ { 1 }  y _ { 1 }$ $s : u _ { 2 }  y _ { 2 } ; y _ { 1 } : b _ { 1 }  z _ { 1 } , y _ { 1 } : b _ { 2 }  z _ { 2 } ;$ and both actions at $y _ { 2 }$ lead to �. At $z _ { 1 } , z _ { 2 } , w ,$ , every action enters the zero-reward absorbing terminal state after receiving, respectively, $R ( z _ { 1 } ) = 1 , R ( z _ { 2 } ) = 1 - u .$ , and $R ( w ) = v$ , with $u , v \in ( 0 , 1 )$ fixed below. Take $\mu _ { S } = \delta _ { s }$ and let $\rho _ { S }$ be uniform on these six nonterminal states; thus every nonterminal state has replay probability $1 / 6$ and $\rho _ { S }$ has full support. Initialize the population recursion at $V _ { 0 } = 0$

Frozen values and the true gap. $\mathrm { A t } \ h \ = \ 1$ the target carries no bootstrap, so exact population regression gives $V _ { k } ( z _ { 1 } ) = 1 , V _ { k } ( z _ { 2 } ) = 1 - \bar { u } , \bar { V } _ { k } ( w ) = v \mathrm { f o r } k \geq 1 ;$ hence $Q _ { V _ { k } } ( y _ { 1 } , b _ { 1 } ) = \gamma$ and $Q _ { V _ { k } } ( \bar { y } _ { 1 } , \bar { b } _ { 2 } ) = \gamma ( \bar { 1 ^ { - } } u )$ , so $b _ { 1 }$ is target-greedy and $\Delta _ { Q } ^ { ( k ) } ( y _ { 1 } ) = \gamma u .$

A pure reward-model error of size �. Let $\widehat { P }$ be exact and ${ \widehat { R } } = R$ except $\widehat { R } ( y _ { 1 } , b _ { 1 } ) = - \eta , \widehat { R } ( y _ { 1 } , b _ { 2 } ) = + \eta .$ . Then $\| \widehat { Q } _ { V _ { k } } - Q _ { V _ { k } } \| _ { \infty } = \| \widehat { R } - R \| _ { \infty } = \eta$ for every �. Because the error is placed in �̂︀ alone, it does not interact with the bootstrap. The implemented branch prefers $b _ { 2 }$ at $y _ { 1 }$ exactly when $\gamma ( 1 - u ) + \eta > \gamma - \eta ,$ , i.e. $u < 2 \eta / \gamma$ . Take $0 < \vartheta < \mathrm { m i n } \{ \eta / \gamma , \epsilon \hat { / } ( 2 \gamma ^ { 2 } ) \}$ and set $u : = 2 \eta / \gamma - \vartheta$ , which lies in $( 0 , 1 )$ because $\eta < \gamma / 2$ . The acting network is the frozen target, so $\delta _ { V , k } = 0$ and $\Lambda _ { k } = \eta _ { \mathrm { s c } , k } = \eta \colon$ the harmful behavior is produced by the named score error, not stipulated.

Limit and loss. For every $k \geq 1$ , the recursion gives $V _ { k + 1 } ( y _ { 1 } ) = \gamma ( 1 - u )$ and $V _ { k + 1 } ( y _ { 2 } ) = \gamma v ;$ the clock makes all relevant values exact after $H = 3$ sweeps. Choose $v \in \left( 1 - u , \operatorname* { m i n } \{ 1 , 1 - u + \epsilon / ( 2 \gamma ^ { 2 } ) \} \right)$ ; then $Q _ { V _ { K } } ( s , u _ { 1 } ) =$ $\gamma ^ { 2 } ( 1 - u ) < \gamma ^ { 2 } v = Q _ { V _ { K } } ( s , u _ { 2 } )$ , so $\tau _ { K } ( s ) = u _ { 2 }$ strictly. Meanwhile $V ^ { \star } ( y _ { 1 } ) = \gamma$ through $b _ { 1 } , V ^ { \star } ( y _ { 2 } ) \overset { \sim } { = } \gamma v$ and $\dot { V } ^ { \star } \dot { ( } s ) = \dot { \gamma } ^ { 2 }$ max $( 1 , v ) = \gamma ^ { 2 }$ since $v < 1$ , while $V ^ { \pi \kappa } ( s ) \bar { = } \gamma V ^ { \star } ( y _ { 2 } ) = \gamma ^ { 2 } v$ ; hence $\| V ^ { \star } - \bar { V } ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } = \mathring { \gamma } ^ { 2 } ( 1 - v ) >$ $\gamma ^ { 2 } u - \epsilon / 2 = 2 \gamma \eta - \gamma ^ { 2 } \vartheta - \epsilon / 2 > 2 \gamma \eta - \epsilon$ , which is (78).

Other residuals. At each block the same current policy collects the fresh data and is its replay law, so $\varepsilon _ { \mathrm { b u f } , k } = 0 ;$ the population update is exact in the full tabular class, so $\varepsilon _ { \mathrm { f i t } , k } = 0 ;$ fresh true-kernel outcomes with the default (SAMPLE, OBS) target give $G _ { k } = G _ { k } ^ { \mathrm { i d e a l } } = \mathcal { T } ^ { \beta _ { k } ^ { \mathrm { r e p } } } V _ { k }$ , hence $\varepsilon _ { \mathrm { k e r } , k } = \varepsilon _ { \mathrm { t g t } , k } = 0 ;$ ; collection is greedy, so $\varepsilon _ { k } = 0$ . Full support of $\rho _ { S }$ gives finite $c _ { 2 } ( m )$ on this finite MDP. □

## Statistical and consequence calculations.

ProofofLemma 5.4. Write $P$ and $P _ { n }$ for the population and the empirical measure and, for $V \in { \mathcal { C } }$ , let $\ell _ { V } : =$ $( V - \dot { Y } ) ^ { 2 } - ( g - Y ) ^ { 2 }$ be the excess loss. Expanding and using $\mathbb { E } [ Y \mid { \dot { S } } ] = g ( S )$ , the cross term vanishes, so

$$
P \ell _ { V } = \| V - g \| _ { 2 , \rho _ { k , S } } ^ { 2 } = : { \mathcal { E } } ( V ) \ \geq 0 .
$$

Step 1 (three elementary bounds). Since $\ell _ { V } = ( V - g ) ( V + g - 2 Y )$ with $| V - g | \leq 2 B$ and $| V + g - 2 Y | \leq 4 B$ and since $\ell _ { V }$ is a difference of two quantities lying in $[ 0 , 4 B ^ { 2 } ] .$

$$
| \ell _ { V } | \le 4 B ^ { 2 } , \qquad \mathbb { E } [ \ell _ { V } ^ { 2 } ] \le 1 6 B ^ { 2 } { \mathcal { E } } ( V ) , \qquad | \ell _ { V } - \ell _ { V ^ { \prime } } | \le 4 B \| V - V ^ { \prime } \| _ { \infty } ,\tag{122}
$$

the last because $\ell _ { V } - \ell _ { V ^ { \prime } } = ( V - V ^ { \prime } ) ( V + V ^ { \prime } - 2 Y )$

Step 2 (reduction to a finite net). Let $\{ V _ { 1 } , \hdots , V _ { N } \} , N = N _ { \delta } ( { \mathcal { C } } )$ , be a �-net of � in supremum norm and, for $V \in { \mathcal { C } } .$ , let $j ( V )$ index a net point with $\| V - V _ { j ( V ) } \| _ { \infty } \leq \delta$ . By the third bound in (122), the empirical-process difference is at most

8��. The reverse triangle inequality for $\sqrt { \mathcal E ( V ) } = \| V - g \| _ { 2 , \rho _ { k , S } }$ gives, after separating the cases $\sqrt { \mathcal { E } ( V _ { j ( V ) } ) } \geq \delta _ { ! }$ $- \mathcal { E } ( V ) / 2 \leq - \mathcal { E } ( V _ { j ( V ) } ) / 2 + 2 B \delta$ . Hence, pointwise,

$$
( P - P _ { n } ) \ell _ { V } - \textstyle { \frac { 1 } { 2 } } \mathcal { E } ( V ) \le \operatorname* { m a x } _ { j \le N } \Bigl [ ( P - P _ { n } ) \ell _ { V _ { j } } - \textstyle { \frac { 1 } { 2 } } \mathcal { E } ( V _ { j } ) \Bigr ] + 1 0 B \delta .\tag{123}
$$

The right-hand side is a maximum of finitely many measurable functions.

Step 3 (Bernstein with a shifted threshold). The standard localization device for bounded squared-loss regression [18, Ch. 11]. Fix � and put $\mathcal { E } _ { j } : = \mathcal { E } ( V _ { j } )$ . By (122), each centered summand is bounded above by $8 B ^ { 2 }$ and has variance at most $1 6 B ^ { 2 } { \mathcal { E } } _ { j }$ . Bernstein’s inequality at threshold $t + \mathcal { E } _ { j } / 2$ , followed by a union bound, yields $\mathbb { P } ( \operatorname* { m a x } _ { j } [ ( P - P _ { n } ) \ell _ { V _ { j } } -$ $\mathcal { E } _ { j } / 2 ] > t ) \le N \exp \{ - n t / ( 7 0 B ^ { 2 } ) \}$ }. Integrating gives

$$
\mathbb { E } \Big [ \operatorname* { m a x } _ { j \leq N } \big ( ( P - P _ { n } ) \ell _ { V _ { j } } - \frac { 1 } { 2 } \mathcal { E } _ { j } \big ) \Big ] _ { + } \leq \frac { 7 0 B ^ { 2 } } { n } \big ( 1 + \log N \big ) .
$$

Step 4 (assembly). Enumerate the fixed countable class $\mathcal { C } _ { 0 }$ and let $V ^ { \star }$ be the first element satisfying $\Vert V ^ { \star } - g \Vert _ { 2 , \rho _ { k , S } } \leq$ dis $_ { ; 2 , \rho _ { k , S } } ( { \mathcal { C } } , g ) + \tau$ for a slack $\tau > 0 .$ . Supremum-norm density ensures existence and the first-index rule is measurable. The $L ^ { 2 }$ choice is essential because $\mathcal { E } ( V ) = \| V - g \| _ { 2 , \rho _ { k , S } } ^ { 2 }$ . Approximate empirical optimality gives $P _ { n } \ell _ { \widehat { V } _ { k + 1 } } \leq$ $P _ { n } \ell _ { V ^ { \star } } + \zeta _ { n } , \mathrm { { s o } }$

$$
\begin{array} { r } { \frac { 1 } { 2 } \mathcal { E } ( \widehat { V } _ { k + 1 } ) \leq \Big [ ( P - P _ { n } ) \ell _ { \widehat { V } _ { k + 1 } } - \frac { 1 } { 2 } \mathcal { E } ( \widehat { V } _ { k + 1 } ) \Big ] + ( P _ { n } - P ) \ell _ { V ^ { \star } } + \mathcal { E } ( V ^ { \star } ) + \zeta _ { n } . } \end{array}
$$

Taking expectations, applying (123) and Step 3, and using $\mathbb { E } ( P _ { n } - P ) \ell _ { V ^ { \star } } = 0$ (conditionally on � in the conditional form), yields

$$
\frac { 1 } { 2 } \mathbb { E } \mathcal { E } ( \widehat { V } _ { k + 1 } ) \ \leq \ \frac { 7 0 B ^ { 2 } } { n } \big ( 1 + \log N _ { \delta } ( \mathcal { C } ) \big ) + 1 0 B \delta + \big [ \mathrm { d i s t } _ { 2 , \rho _ { k , S } } ( \mathcal { C } , g ) + \tau \big ] ^ { 2 } + \zeta _ { n } .
$$

Multiplying by 2 and letting $\tau \downarrow 0$ gives (88) for $C _ { 1 } \geq 1 4 0$ . Only boundedness and the conditional mean of $Y$ were used, so bounded label noise is covered.

For $( 8 9 )$ , use a fresh symbol $u \in ( 0 , 1 )$ for the confidence level during this proof. Since $g \in { \mathcal { C } }$ and the supplied ERM is exact over ${ \mathcal { C } } ,$ comparison with � gives $P _ { n } \ell _ { \widehat { V } _ { k + 1 } } \leq P _ { n } \ell _ { g } = 0$ . Hence

$$
\begin{array} { r } { \frac 1 2 \mathcal { E } ( \widehat { V } _ { k + 1 } ) \leq ( P - P _ { n } ) \ell _ { \widehat { V } _ { k + 1 } } - \frac 1 2 \mathcal { E } ( \widehat { V } _ { k + 1 } ) . } \end{array}
$$

By (123) and the tail bound of Step 3, with conditional probability at least $1 - u$ the right-hand side is at most $7 \dot { 0 } B ^ { 2 } \left\{ \begin{array} { r l r } \end{array} \right.$ {log $N _ { \delta } ( \mathcal { C } ) + \log ( 1 / u ) \} / n + \mathrm { \bar { 1 0 } } B \delta$ . Multiplication by $^ { 2 , }$ enlargement to $C _ { 1 } \geq 1 \bar { 4 0 }$ , and renaming � as � prove (89). This argument does not divide by any state probability. □

ProofofProposition 5.5. Throughout $g : = G _ { k }$ <sub>�</sub>, so $\| g \| _ { \infty } \le V _ { \mathrm { m a x } }$ by (10), $| Y _ { i } | \le V _ { \operatorname* { m a x } }$ and $\mathbb { E } [ Y _ { i } \mid S _ { i } ] = g ( S _ { i } )$ Lemma 5.4 applies with $B = V _ { \mathrm { m a x } } ^ { \bar { } }$ and ${ \mathcal { C } } = { \mathcal { F } } _ { n } ^ { V }$ , being stated for an arbitrary bounded measurable � and not requiring $g \in \mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$

Step 1 (approximation). The routed approximation bound (85) already applies to the externally clipped class:

$$
\mathrm { d i s t } _ { \infty } ^ { \mathrm { s u p } } ( \mathcal { F } _ { n } ^ { V } , \mathcal { G } _ { 0 } ^ { \mathrm { c l k } } ) \leq C _ { \mathrm { a p p x } } m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } .
$$

Its proof uses Lemma 2.3 head by head, exploiting that every target in $\mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$ lies in the level-wise band.

Step 2 (covering). If $V _ { \mathrm { m a x } } = 0$ the claim is trivial. Otherwise set $\delta _ { n } = ( 1 \land V _ { \operatorname* { m a x } } ) / n$ . The product entropy bound (86) gives

$$
\begin{array} { r l r } {  { \log N _ { \delta _ { n } } ( \mathcal { F } _ { n } ^ { V } ) \leq H \{ \log m _ { n } + ( s _ { m _ { n } } + 1 ) \log ( \frac { 2 ( L _ { m _ { n } } + 1 ) D _ { m _ { n } } ^ { 2 } } { \delta _ { n } } ) \} } } \\ & { } & { \lesssim H m _ { n } ^ { \alpha ^ { \star } } ( \log n ) ^ { 1 + 2 \xi ^ { \star } } . } \end{array}\tag{124}
$$

Step 3 (combination, comparator in $L ^ { 2 } ( \rho _ { k , S } ) )$ . Since $\mathcal { T } V _ { k } \in \mathcal { G } _ { 0 } ^ { \mathrm { c l k } }$ and $\| \cdot \| _ { 2 , \rho _ { k , S } } \leq \| \cdot \| _ { \infty }$ , the triangle inequality in $L ^ { 2 } ( \rho _ { k , S } )$ gives

$$
\mathrm { d i s t } _ { 2 , \rho _ { k , S } } ( \mathcal { F } _ { n } ^ { V } , g ) \leq \mathrm { d i s t } _ { \infty } ^ { \operatorname* { s u p } } ( \mathcal { F } _ { n } ^ { V } , \mathcal { G } _ { 0 } ^ { \mathrm { c l k } } ) + \left\| T V _ { k } - G _ { k } \right\| _ { 2 , \rho _ { k , S } } \leq C _ { \mathrm { a p p x } } m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } + D _ { k } ^ { ( 2 ) } .\tag{125}
$$

Step 4 (take the square root before splitting). Apply Jensen, then subadditivity of ${ \sqrt { \cdot } } ,$ to the four summands of (88) at the radius $\delta _ { n } = ( \bar { 1 } \wedge V _ { \operatorname* { m a x } } ) / n \colon$

$$
\begin{array} { r l } & { \mathbb { E } \big [ \| \widehat { V } _ { k + 1 } - g \| _ { 2 , \rho _ { k , S } } \bigm | \mathcal { H } _ { k } \bigm ] \leq \sqrt { 2 } \mathrm { d i s t } _ { 2 , \rho _ { k , S } } ( \mathcal { F } _ { n } ^ { V } , g ) } \\ & { \qquad + \left( \frac { C _ { 1 } V _ { \operatorname* { m a x } } ^ { 2 } } { n } \big ( 1 + \log N _ { \delta _ { n } } ( \mathcal { F } _ { n } ^ { V } ) \big ) \right) ^ { 1 / 2 } } \\ & { \qquad + \left( C _ { 1 } V _ { \operatorname* { m a x } } \delta _ { n } \right) ^ { 1 / 2 } + \sqrt { 2 \zeta _ { n } } . } \end{array}
$$

By (125) the first summand is at most $\sqrt { 2 } C _ { \mathrm { a p p x } } m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } + \sqrt { 2 } D _ { k } ^ { ( 2 ) }$ . Since $n \geq H m _ { n }$ , equation (124) yields

$$
\left( \frac { H m _ { n } ^ { \alpha ^ { \star } } } { n } \right) ^ { 1 / 2 } \leq m _ { n } ^ { ( \alpha ^ { \star } - 1 ) / 2 } .
$$

The remaining $n ^ { - 1 / 2 }$ terms are no larger than a constant times this rate. Collecting constants gives (91). For fixed $H , m _ { n } \asymp n \bar { / } H ,$ so this has the same exponent in � as the one-head rate; the displayed $m _ { n }$ records the router’s �-head complexity cost. Step 5 (greedy collection). If $\mathbf { a } _ { i } = a ^ { \star } ( S _ { i } ; V _ { k } )$ then $G _ { k } = \tau V _ { k }$ by Lemma 2.4(iii), so $D _ { k } ^ { ( 2 ) } = 0$ and (91) reduces to (92). Noisy targets are covered because Lemma 5.4 requires only $\mathbb { E } [ Y _ { i } \mid S _ { i } ] = g ( S _ { i } )$ with $| Y _ { i } | \le V _ { \operatorname* { m a x } }$ 口

ProofofCorollary 5.7. Lemma 2.3 gives exact closure. Empirical means at visited states (zero at unvisited states) give a measurable exact ERM in the level-wise band because every label at level ℎ is bounded by $R _ { \operatorname* { m a x } } + \gamma V _ { \operatorname* { m a x } } ^ { ( h - 1 ) } = V _ { \operatorname* { m a x } } ^ { ( h ) }$ Condition on $\mathcal { H } _ { k }$ . Given $N _ { k , s } = r > 0$ , the coordinatewise sample mean is unbiased with squared error $\sigma _ { k , s } ^ { 2 } / r ;$ when $r = 0 ,$ , its squared error is $g _ { k , s } ^ { 2 }$ . Multiplying by $p _ { k , s }$ and summing proves the exact identity (97); coordinates with $p _ { k , s } = 0$ vanish.

For $r \geq 1 , 1 / r \leq 2 / ( r + 1 ) . \mathrm { I f } M \sim \mathrm { B i n } ( n , p )$ , then

$$
\mathbb E \frac { 1 } { M + 1 } = \int _ { 0 } ^ { 1 } ( 1 - p + p x ) ^ { n } d x = \frac { 1 - ( 1 - p ) ^ { n + 1 } } { ( n + 1 ) p } ,
$$

so $p \mathbb { E } [ \mathbf { 1 } \{ M > 0 \} / M ] \leq 2 / ( n + 1 )$ . Also $\operatorname { \geqslant P r } ( M = 0 ) = p ( 1 - p ) ^ { n } \leq ( n / ( n + 1 ) ) ^ { n } / ( n + 1 )$ , by maximizing over �. Using $\sigma _ { k , s } ^ { 2 } , g _ { k , s } ^ { 2 } \le V _ { \mathrm { m a x } } ^ { 2 } ,$ summing over the � states and applying Jensen proves the first bound in (98). Finally $( 1 + 1 / n ) ^ { n } \geq 2 ,$ and for � $. \geq 2$ its first three binomial terms are at least $9 / 4 ,$ , giving the constants $5 / 2$ and $2 2 / 9$ . These bounds are deterministic after conditioning, so they also hold unconditionally.

The additional design choices respectively remove kernel, target, replay, aliasing, and drift links; full support makes every finite-state density coefficient finite. Since $G _ { k } \in \mathcal { F } ^ { \mathrm { t a b } }$ , no policy-image approximation term is needed, so the composition chain charges the action and exploration links only once. Substitution of (98) and Theorem 4.5 at $r = p$ into Theorem 3.18, with $K \geq H - 1$ , gives (100).

For fixed �, name the �th confidence event

$$
\begin{array} { r } { \mathcal { E } _ { k , K } ( \delta ) : = \left\{ \| V _ { k + 1 } - G _ { k } \| _ { 2 , \rho _ { k , S } } \leq \mathrm { s t a t } _ { k , K } ^ { \mathrm { t a b , h p } } ( \delta ) \right\} . } \end{array}
$$

Apply (89) conditionally on $\mathcal { H } _ { k }$ with confidence $1 - \delta / ( H - 1 )$ and covering radius $V _ { \mathrm { m a x } } / n _ { k }$ . It gives $\mathbb { P } ( \mathcal { E } _ { k , K } ( \delta ) ^ { c } \mid$ $\mathcal { H } _ { k } ) \le \delta / ( H - 1 )$ directly in population $L ^ { 2 }$ , including the zero values assigned to unvisited states, so no minimumstate-mass factor occurs. The tower property and a union bound over the at most $H - 1$ relevant blocks make all these events simultaneous with probability at least $1 - \delta .$ , despite adaptive histories. On their intersection, the pathwise triangle chain of Lemma 3.13 and Lemma 3.10 gives (102); no cross-block independence is used. □

ProofofTheorem 6.3. Substitution of Proposition 5.6 into Proposition 5.5 gives

$$
\varepsilon _ { \mathrm { f i t } , k } \ \leq \ \mathrm { s t a t } _ { k } + \sqrt { 2 } \big ( \varepsilon _ { \mathrm { b u f } , k } + \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } + \varepsilon _ { k } \bar { D } _ { k } ^ { \mathrm { e x p } } \big ) ,
$$

the omitted kernel and target links being zero. Since $p = 2$ here, $\varepsilon _ { \mathrm { a c t } , k } ^ { ( p ) } = \varepsilon _ { \mathrm { a c t } , k } ^ { ( 2 ) } ;$ adding the remaining terms of $e _ { k , 2 } ^ { \mathrm { B e l l } }$ proves (109). Equation (110) follows from (49), since $w _ { K , k } ^ { ( H ) } = 0$ for $k \le K - H$ and $\begin{array} { r } { \sum _ { k } w _ { K , k } ^ { ( H ) } = 2 \phi _ { 2 , K } ^ { ( H ) } } \end{array}$ □

## B Further refinements

Proposition B.1 (One-sided propagation). Under Theorem 3.18, suppose additionally that, almost surely,

$$
V _ { 0 } \le V ^ { \star } , \qquad e _ { k } : = V _ { k + 1 } - T V _ { k } \le 0 \quad p o i n t w i s e o n \widetilde { S } ^ { \circ } , f o r e \nu e r y \ k < K .\tag{126}
$$

Then $V _ { k } \leq V ^ { \star }$ for every $k \leq K$ , and

$$
\mathbb { E } \| V ^ { \star } - V ^ { \pi \kappa } \| _ { 1 , \mu _ { S } } \leq \frac { \mathcal { B } _ { K } } { 2 } + \frac { 1 } { 2 } \sum _ { k = 0 } ^ { K - 1 } w _ { K , k } ^ { ( H ) } e _ { k , p } ^ { \mathrm { B e l l } } \leq \frac { \mathcal { B } _ { K } } { 2 } + \phi _ { s , K } ^ { ( H ) } \bar { e } _ { K , H , p } ^ { \mathrm { B e l l } } .\tag{127}
$$

The sign premise is pointwise; an $L ^ { p }$ or replay-almost-everywhere sign condition is insufficient.

Proof. For $z _ { k } : = V ^ { \star } - V _ { k }$ , monotonicity and (126) give $z _ { k } \ge 0$ inductively and $z _ { k + 1 } \leq \gamma \mathcal { P } _ { \circ } ^ { \pi ^ { \star } } z _ { k } + | e _ { k } | ,$ , so only the $\pi ^ { \star }$ branch propagates. At the loss-resolvent step, $\gamma ( \mathcal { P } _ { \circ } ^ { \pi ^ { \star } } - \mathcal { P } _ { \circ } ^ { \pi _ { K } } ) z _ { K } \leq \gamma \mathcal { P } _ { \circ } ^ { \pi ^ { \star } } z _ { K }$ . Thus the pathwise proof of Lemma 3.10 has half the initialization boundary and half every residual coefficient; the conditional envelopes of Theorem 3.18 give (127). □

Corollary B.2 $( L ^ { 2 }$ final-score transfer). In the setting ofTheorem 6.1, define

$$
E _ { K } ( \widetilde s ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } | \widehat { Q } _ { V _ { K } } ( \widetilde s , a ) - Q _ { V _ { K } } ( \widetilde s , a ) | , \qquad \eta _ { K , 2 } : = \| E _ { K } \| _ { 2 , \rho _ { K , S } } .
$$

Assume the final-law density bound (106) at $s = 2 ,$ , and let the deterministic $\bar { \eta } _ { K , 2 }$ satisfy $\eta _ { K , 2 } \leq \bar { \eta } _ { K , 2 }$ almost surely. Then

$$
\mathbb { E } \| V ^ { \star } - V ^ { \widehat { \pi } _ { K } } \| _ { 1 , \mu _ { S } } \leq \mathcal { B } _ { K } + \sum _ { k = 0 } ^ { K - 1 } w _ { K , k } ^ { ( H ) } e _ { k , p } ^ { \mathrm { B e l l } } + 2 \bar { \eta } _ { K , 2 } \sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } d _ { 2 } ^ { \mathrm { f i n } } ( \ell ) .\tag{128}
$$

$I f \eta _ { K , 2 }$ is integrable, the last term may instead retain $2 \mathbb { E } [ \eta _ { K , 2 } ]$ times the same deterministic sum.

Proof. Score comparison gives $0 \leq \mathcal { T } V _ { K } - \mathcal { T } ^ { \widehat { \pi } _ { K } } V _ { K } \leq 2 E _ { K }$ , whose resolvent contribution after integration is $2 \textstyle \sum _ { \ell < H } \gamma ^ { \ell } \int E _ { K } \dot { d } \{ \mu _ { S } ^ { \circ } ( \mathcal { P } _ { \circ } ^ { \widetilde { \pi } _ { K } } ) ^ { \ell } \}$ . Cauchy–Schwarz and (106) bound it pathwise by 2�<sub>�,2</sub> $\begin{array} { r } { \sum _ { \ell < H } \gamma ^ { \ell } d _ { 2 } ^ { \mathrm { f i n } } ( \ell ) } \end{array}$ . Use either its deterministic majorant or its expectation to conclude. □

## OA.1: History-space moment coverage

For the two realized occupancy branches in the proof of Lemma 3.10, define

$$
A _ { K , k } : = \sum _ { \ell \geq 0 \cdot \ell + K - k < H } \gamma ^ { \ell + K - k } \big ( d _ { K , k , \ell , s } ^ { 1 } + d _ { K , k , \ell , s } ^ { 2 } \big ) , \qquad R _ { k } : = \| e _ { k } \| _ { p , \rho _ { k , s } } .\tag{129}
$$

The exact pathwise proof gives $\begin{array} { r } { \| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } \leq \mathcal { B } _ { K } + \sum _ { k < K } A _ { K , k } R _ { k } } \end{array}$ . Consequently, for conjugate history-space exponents $u , v \in [ 1 , \infty ]$

$$
\mathbb { E } \| V ^ { \star } - V ^ { \pi \kappa } \| _ { 1 , \mu _ { S } } \leq \mathcal { B } _ { K } + \sum _ { k < K } \| A _ { K , k } \| _ { L ^ { u } ( \Omega ) } \| R _ { k } \| _ { L ^ { v } ( \Omega ) } .\tag{130}
$$

This is Hölder’s inequality on the history probability space. Because $A _ { K , k }$ generally depends on future iterates through $\pi _ { K }$ and the comparison kernels, factorizing $\mathbb { E } [ A _ { K , k } { \tilde { R _ { k } } } ]$ is invalid; the deterministic worst-policy theorem is the $u = \infty$ case.

Realized coverage is strictly weaker: in an $H = 2$ deterministic MDP, let both constructed branches select a covered bottom state �, set the design law to $\delta _ { x } ,$ and add an unused action leading to �. All realized measures are covered, whereas the unused policy produces $\delta _ { y } \ll \delta _ { x }$ ; hence the all-policy supremum assumption fails.

## OA.2: Geometric verification of the margin

Proposition B.3 (Tube and transverse-growth criterion for one gap law). Let Σ be a measurable switching set. Suppose, for constants $c , r , T , \kappa , u _ { 0 } > 0 ,$

$$
\Delta _ { Q } ( x ) \ge \operatorname* { m i n } \{ c \mathrm { ~ d i s t } ( x , \Sigma ) ^ { r } , u _ { 0 } \} , \quad \rho \{ \mathrm { d i s t } ( x , \Sigma ) \le \epsilon \} \le T \epsilon ^ { \kappa }
$$

whenever $0 < \epsilon \leq ( u _ { 0 } / c ) ^ { 1 / r }$ . Then the global action-gap margin holds with

$$
\alpha = \kappa / r , \qquad C _ { \mathrm { m a r g } } = \mathrm { m a x } \{ T c ^ { - \kappa / r } , u _ { 0 } ^ { - \kappa / r } \} .\tag{131}
$$

Proof. For $0 < u < u _ { 0 }$ , the event $\{ \Delta _ { Q } \leq u \}$ lies in the tube of radius $( u / c ) ^ { 1 / r }$ and has probability at most $T c ^ { - \kappa / r } u ^ { \kappa / r }$ For $u \geq u _ { 0 }$ , use $1 \leq u _ { 0 } ^ { - \kappa / r } u ^ { \kappa / r }$ . The tube condition also makes the zero-gap switching set null. □

For Definition 4.1, require this criterion almost surely for each $( Q _ { V _ { k } } , \rho _ { k , S } )$ ; alternatively verify it uniformly for $\left( Q _ { V ^ { \star } } , \rho _ { k , S } \right)$ and apply Proposition 4.3.

## OA.3: Matched-budget allocation and horizon calculations

ProofofTheorem 3.20. Substitute (56) into the first inequalities of Theorems 3.18 and 3.19. Inactive coordinates have zero propagation weight, while the nonstatistical terms give $F _ { K } ^ { \mathrm { s h } }$ and $F _ { K } ^ { \mathrm { l e v } }$ . It remains to minimize $\sum _ { i } c _ { i } n _ { i } ^ { - \nu }$ under a common terminal-window budget.

For positive $c _ { i }$ , the objective is strictly convex on the positive orthant. The Lagrange equations

$$
- \nu c _ { i } n _ { i } ^ { - \nu - 1 } + \lambda = 0
$$

give $n _ { i } \propto c _ { i } ^ { 1 / ( 1 + \nu ) }$ . Normalizing by the budget and substituting back proves (59), hence (60)–(61).

With lower bounds, strict convexity still gives a unique minimizer. The KKT conditions say that an interior coordinate satisfies $n _ { i } = ( \nu c _ { i } / \lambda ) ^ { 1 / ( 1 + \nu ) }$ , whereas a coordinate whose unconstrained value is at most $L _ { i }$ is fixed at $L _ { i } .$ . This is exactly (63). For $\begin{array} { r } { \mathsf { N } > \sum _ { i } L _ { i } } \end{array}$ , the sum of its right-hand side is continuous and strictly decreasing in � over the range relevant to the budget, from infinity to $\textstyle \sum _ { i } L _ { i } ;$ hence the required � is unique. Substitution defines (62) and proves the stated constrained bounds. The unconstrained closed form applies precisely when none of its coordinates violates a lower bound.

For integer $L _ { i } \geq 1$ and integer N, each $\lfloor n _ { i } ^ { L } \rfloor \ge L _ { i }$ , and the number of undistributed labels is the nonnegative integer $\begin{array} { r } { \mathsf { N } - \sum _ { i } \lfloor n _ { i } ^ { L } \rfloor } \end{array}$ . Allocating each one according to (64) preserves feasibility and uses the entire budget. Moreover, $\lfloor n _ { i } ^ { L } \rfloor \ge n _ { i } ^ { L } / 2$ because $n _ { i } ^ { L } \geq 1$ , while adding labels can only decrease the objective. Therefore the final integer allocation obeys

$$
\sum _ { i } c _ { i } n _ { i } ^ { - \nu } \leq 2 ^ { \nu } \sum _ { i } c _ { i } ( n _ { i } ^ { L } ) ^ { - \nu } = 2 ^ { \nu } \Psi _ { \nu } ( c , L , \mathsf { N } ) .
$$

Finally, setting $\begin{array} { r } { j = K { - } k \operatorname { g i v e s } | \mathcal { K } _ { K } | = J \operatorname { a n d } | \mathcal { Z } _ { K } | = \sum _ { j = 1 } ^ { J } ( H { - } j ) = J H { - } J ( J + 1 ) / 2 \operatorname { f o r } J = \operatorname* { m i n } \{ K , H { - } 1 \} } \end{array}$ .

Proof of Corollary 3.21. Put $q : = 1 / ( 1 + \nu )$ and index the active shared-reset blocks by $j = K _ { H } - k \in \{ 1 , \dots , H - 1 \}$ On the one-state-per-level chain, the uniform clock law has $d _ { 2 , H } ( m ) = \sqrt { H }$ , and therefore

$$
w _ { H , j } ^ { \mathrm { u n i f } } = 2 \sqrt { H } \sum _ { m = j } ^ { H - 1 } \gamma _ { H } ^ { m } .\tag{132}
$$

In the near-unit regime, �<sup>�</sup> is bounded above and below by positive constants uniformly for $m < H$ . Thus $w _ { H , j } ^ { \mathrm { u n i f } } =$ $\Theta ( H ^ { 3 / 2 } )$ for $j \le H / 2$ and is ${ \cal O } ( H ^ { 3 / 2 } )$ everywhere. Consequently

$$
\left\{ \sum _ { j = 1 } ^ { H - 1 } ( b _ { H } ^ { \mathrm { s h } } w _ { H , j } ^ { \mathrm { u n i f } } ) ^ { q } \right\} ^ { 1 / q } = \Theta \bigl ( b _ { H } ^ { \mathrm { s h } } H ^ { \nu + 5 / 2 } \bigr ) .
$$

For the coefficient-optimal shared law at $s = 2$ and $K _ { H } \ge H - 1$ , write

$$
S _ { H } : = \sum _ { m = 1 } ^ { H - 1 } ( m \gamma _ { H } ^ { m } ) ^ { 2 / 3 } , \qquad r _ { H - m } = \frac { ( m \gamma _ { H } ^ { m } ) ^ { 2 / 3 } } { S _ { H } } .
$$

Then $d _ { 2 , H } ( m ) = S _ { H } ^ { 1 / 2 } ( m \gamma _ { H } ^ { m } ) ^ { - 1 / 3 }$ and

$$
w _ { H , j } ^ { \mathrm { o p t } } = 2 S _ { H } ^ { 1 / 2 } \sum _ { m = j } ^ { H - 1 } m ^ { - 1 / 3 } \gamma _ { H } ^ { 2 m / 3 } .\tag{133}
$$

Near unit, $S _ { H } = \Theta ( H ^ { 5 / 3 } )$ ; the last sum is $\Theta ( H ^ { 2 / 3 } )$ for $j \ \leq \ H / 2$ and $O ( H ^ { 2 / 3 } )$ everywhere. Hence the same calculation gives $\mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { s h , o p t } } = \Theta ( b _ { H } ^ { \mathrm { s h } } H ^ { \nu + 5 / 2 } )$ .

For direct reset, a fixed depth � occurs in exactly � active pairs and $A _ { k , H - m } ^ { \mathrm { l e v } } = 2 \gamma _ { H } ^ { m }$ . Therefore, in the near-unit regime,

$$
\mathsf { C } _ { \nu , K _ { H } } ^ { \mathrm { l e v } } \asymp b _ { H } ^ { \mathrm { l e v } } \left( \sum _ { m = 1 } ^ { H - 1 } m \right) ^ { 1 / q } = \Theta \bigl ( b _ { H } ^ { \mathrm { l e v } } H ^ { 2 \nu + 2 } \bigr ) ,
$$

proving (65).

Under fixed discount, (132) is $\Theta ( \sqrt { H } \gamma ^ { j } )$ , so its �th-power sum gives $\Theta ( b _ { H } ^ { \mathrm { s h } } \sqrt { H } )$ . In $( 1 3 3 ) , S _ { H }$ is bounded above and below and the weights decay geometrically; their �th-power sum is finite and bounded away from zero. Likewise $\begin{array} { r } { \sum _ { m \geq 1 } m \big ( 2 \gamma ^ { m } b _ { H } ^ { \mathrm { l e v } } \big ) ^ { q } } \end{array}$ is finite and positive. This proves (66). Finally, the tabular fit envelope (98) has statistical constant Θ $( V _ { \mathrm { m a x } } \sqrt { H } )$ for a shared �-state clock law and $\Theta ( V _ { \mathrm { m a x } } )$ on each singleton slice. Substitution at $\nu = 1 / 2$ proves (67). □

## OA.4: Tie-sensitive equality at the score floor

In Proposition 4.8, take $u = 2 \eta / \gamma$ and $v = 1 - u , 0 < \eta < \gamma / 2$ . If the harmful actions win the resulting ties at $y _ { 1 }$ and the root, exact population updates give

$$
\begin{array} { r } { V ^ { \star } ( s ) - V ^ { \pi _ { K } } ( s ) = \gamma ^ { 2 } ( 1 - v ) = \gamma ^ { 2 } u = 2 \gamma \eta . } \end{array}\tag{134}
$$

All other residuals vanish; without coordinated tie-breaking, use the main $2 \gamma \eta - \epsilon$ bound.

## OA.5: Polynomial-schedule consistency rate

Let $a = ( 1 - \alpha ^ { \star } ) / 2$ and $b = ( 1 + 2 \xi ^ { \star } ) / 2$ . In the setting of Corollary 6.4, suppose $n _ { k } \asymp k ^ { r } , \zeta _ { n _ { k } } = O ( n _ { k } ^ { - t } )$ , and $\varepsilon _ { k } = { O } \left( { k ^ { - s _ { 0 } } } \right)$ , with $r , t , s _ { 0 } > 0$ . For fixed � and $K \geq 2 H$

$$
\begin{array} { r } { \mathbb { E } \| V ^ { \star } - V ^ { \pi _ { K } } \| _ { 1 , \mu _ { S } } = O \Big ( \phi _ { 2 } ^ { ( H ) } \left[ ( \log K ) ^ { b } K ^ { - r a } + K ^ { - r t / 2 } + K ^ { - s _ { 0 } } \right] \Big ) . } \end{array}\tag{135}
$$

For $K \geq 2 H$ , the boundary vanishes and active $k \asymp K ; m _ { n _ { k } } \asymp n _ { k }$ at fixed �, so (111) applies. The statistical term governs i $\ ' t \geq 2 a$ and $s _ { 0 } \geq r a$

## OA.6: Monte Carlo score bound

Assume arbitrary-query access, and let $A : = | { \cal { A } } |$ . For each action, let $Z _ { a }$ be a centered �-sample average of conditionally independent variables in $[ - V _ { \mathrm { m a x } } , V _ { \mathrm { m a x } } ] ;$ ; dependence across actions is allowed. With deterministic reward error $\eta _ { R } .$ , the score error $e ^ { M }$ satisfies

$$
\left( \mathbb { E } [ e ^ { M } ( s ) ^ { 2 } ] \right) ^ { 1 / 2 } \leq \eta _ { R } + \frac { \gamma V _ { \operatorname* { m a x } } } { \sqrt { M } } \operatorname* { m i n } \Bigl \{ \sqrt { A } , \sqrt { 2 \{ \log ( 2 A ) + 1 \} } \Bigr \} .\tag{136}
$$

Hoeffding and a union bound give Pr(max $| Z _ { a } | \geq t ) \leq$ min $\left\{ 1 , 2 A \exp [ - M t ^ { 2 } / ( 2 V _ { \operatorname* { m a x } } ^ { 2 } ) ] \right\}$ . Integrating at $t _ { 0 } ^ { 2 } =$ $2 V _ { \mathrm { m a x } } ^ { 2 } \log ( 2 A ) / M$ yields E ma $\mathrm { \bar { x } } _ { a } | Z _ { a } | ^ { 2 } \leq 2 \dot { V } _ { \mathrm { m a x } } ^ { 2 } \mathrm { \bar { \{ } l o g ( 2 A ) + 1 \} } / M .$ ; also E max $\begin{array} { r } { { \bf \Phi } _ { \boldsymbol { \imath } } | Z _ { a } | ^ { 2 } \leq \sum _ { a } \mathbb { E } Z _ { a } ^ { \frac { - } { 2 } } \leq \tilde { A } V _ { \mathrm { m a x } } ^ { 2 } / M } \end{array}$ Minkowski and $e ^ { M } \leq \eta _ { R } + \gamma \operatorname* { m a x } _ { a } \left| Z _ { a } \right|$ prove (136). The reward error is added once, not per action.

## OA.7: Moment-localized score perturbations

Let $\begin{array} { r } { E ( x ) : = \operatorname* { m a x } _ { a } | q ( x , a ) - Q _ { V } ( x , a ) } \end{array}$ | and let $D ( x )$ be the true regret of the �-greedy action. Assume $\| E \| _ { r , \rho } \leq \delta _ { r }$ with $p < r < \infty$ . For every $t > 0$ at which the margin is valid at scale 2�,

$$
\| D \| _ { p , \rho } ^ { p } \leq C _ { \mathrm { m a r g } } ( 2 t ) ^ { p + \alpha } + 2 ^ { p } \delta _ { r } ^ { r } t ^ { p - r } .\tag{137}
$$

Indeed, on $\{ E \leq t \} , D \leq 2 t$ and $D > 0$ implies $\Delta _ { Q } \leq 2 t ;$ on $\{ E > t \} , D \leq 2 E$ and $\mathbb { E } [ E ^ { p } \mathbf { 1 } \{ E > t \} ] \leq \delta _ { r } ^ { r } t ^ { p - r }$

For $C _ { \mathrm { m a r g } } , \delta _ { r } > 0$ , optimizing the right-hand side gives

$$
\begin{array} { r } { t _ { * } = \left[ \frac { ( r - p ) \delta _ { r } ^ { r } } { 2 ^ { \alpha } ( p + \alpha ) C _ { \mathrm { m a r g } } } \right] ^ { 1 / ( r + \alpha ) } , } \\ { \| D \| _ { p , \rho } \lesssim _ { p , r , \alpha } C _ { \mathrm { m a r g } } ^ { ( r - p ) / ( p ( r + \alpha ) ) } \delta _ { r } ^ { r ( p + \alpha ) / ( p ( r + \alpha ) ) } . } \end{array}\tag{138}
$$

This optimized bound requires the margin at scale $2 t _ { * } ;$ otherwise minimize (137) over admissible thresholds and compare with $\| D \| _ { p , \rho } \leq \bar { 2 \delta } _ { r } . \mathrm { I f } \delta _ { r } = 0 .$ , then $D = 0$ almost everywhere. $\mathbf { A } \mathbf { t } \boldsymbol { r } = \boldsymbol { p }$ the split gives no improved power; the formal $r  \infty$ limit recovers the uniform-error exponents only with $L ^ { \infty }$ control.