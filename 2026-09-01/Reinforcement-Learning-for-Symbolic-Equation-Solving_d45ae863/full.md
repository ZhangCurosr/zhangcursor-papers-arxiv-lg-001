# Reinforcement Learning for Symbolic Equation Solving

Kevin P. O’Keefe

Starling Research Institute

kevin.p.okeefe@gmail.com

## Abstract

We present a reinforcement-learning agent that solves symbolic equations step by step both nonlinear closed equations (radicals, exponentials, trigonometric) and, for the first time, a controlled class of restricted-open families requiring a change of variables (CoV) such as completing the square. We cast algebra as an MDP with a dynamic action space and a tree-structured policy (TreeMLP). The main policy learns the full solution procedure from reward alone, with no supervised solution traces; the CoV substitution itself comes from a supervised generator interchangeable with a CAS call (≤ 0.02 end-task efect). Success means one root verified by substitution and mapped back by exact inverse composition. On closed equations the agent matches the prior best on CommonCore (0.93 greedy vs. ConPoLe’s 0.925) and handles nonlinear families (radical, log, trig) under a single policy. On four hand-designed restricted-open families — quadratic, cubic, quartic, and exponential, each with a known closed-form substitution — it reaches 0.79 beam / 0.67 greedy (5-seed means on the 3M-step headline cohort of Section 4.2.2); the greedy mean alone exceeds the strongest non-learned search (A<sup>∗</sup>, 0.64), with beam adding a further margin. We test the CoV trigger directly against hand-coded rules: on the three single-CoV families (quadratic, cubic, quartic) a one-line detector matches or beats the policy — nothing is learned there — and learned timing has content only on the exponential family, the one family requiring a nested CoV, where the natural rule solves none of the held-out equations while the policy solves 75% from reward alone. The honest claim is recovery of a hand-engineered trigger without access to the detector, not superiority over it. At 10× scale a sharp seed-level bimodality emerges; a UCB learning-progress curriculum shows a non-significant positive trend toward mitigating it. We do not claim general open-equation solving: every openequation result here is confined to these four controlled families.

## 1 Introduction

Most machine-learning approaches to symbolic mathematics map a problem directly to its answer Lample & Charton (2020): a model emits a solution in one shot. Many downstream goals, however, need the procedure: a sequence of explicit, replayable transformations that reaches the solution and exposes the trace and the policy that produced it. Diferential-equation reductions, theorem-prover tactics, and didactic stepby-step solvers all consume such traces. We show that reinforcement learning can acquire this procedural capability for algebraic equation solving: building on RL for nonlinear closed equations (our earlier work; O’Keefe, 2025) (radical, exponential, trigonometric), our agent additionally solves, to our knowledge for the first time, the harder restricted-open setting, where the solution requires a change of variables (CoV) drawn from hand-designed families with known closed forms. The main policy learns from reward alone, with no supervised solution traces; the one supervised component is the CoV substitution generator, which is interchangeable with a CAS call. Throughout, solving means producing one root verified by substitution in the active equation and mapped back by exact inverse composition.

Scope of the claims. Every open-equation claim in this paper is scoped to a controlled restricted-open benchmark: four hand-designed families (quadratic, cubic, quartic, exponential), each restricted to forms admitting a known closed-form depression substitution, with generic instances that admit no such substitution excluded by construction (Section 4.2). We do not claim general open-equation solving, and we do not claim that the entire pipeline is reward-driven: the claim of reward-only learning attaches to the main RL policy, not to $\pi _ { \mathrm { { c o v } } } .$ , which is supervised (and replaceable by a CAS at ≤0.02 end-task cost). Success is one substitution-verified root, not the complete solution set.

We formulate symbolic algebra as an MDP whose actions pair an algebraic operation with a sub-expression of the current equation, the legal-action set re-derived from the state each step. The policy/value network is TreeMLP: node IDs are embedded, K rounds of strictly-local sum-aggregation message passing update node states, and a masked pool yields a fixed-size graph feature for the PPO heads. Three light inductive biases (curriculum, success-replay bufer, constant-relabel macroaction) are layered on top; on CommonCore this matches the prior best (ConPoLe; Poesia et al. 2021).

We expose CoV as a single macroaction in the same dynamic action space; the main policy learns the surrounding procedure and selects the CoV trigger from reward. We test that trigger directly against handcoded rules rather than asserting it (Section 4.2.4): on the three single-CoV families a one-line detector matches or beats the policy — nothing is learned there. The trigger has content only on the exponential family, which requires a nested CoV, where the natural rule solves 0/75 and the policy solves 56/75 from reward alone. The honest claim is recovery of a hand-engineered trigger without access to the detector, not superiority over it. At 10× scale a sharp seed-level bimodality emerges; a UCB learning-progress curriculum shows a non-significant positive trend toward mitigating it.

Contributions. (a) The first RL agent whose main policy is reward-driven for restricted-open (CoV) symbolic equation solving on a controlled four-family benchmark, with the trigger recovered from reward alone and composable with closed-equation steps under a single policy. The claim is scoped to recovery, and the trigger ablation of Section 4.2.4 localizes where it has content: on the three single-CoV families (quadratic, cubic, quartic) a one-line hand-coded rule matches or beats the policy, so nothing is learned there; learned timing matters only on the exponential family, the sole family requiring a nested CoV, where the policy recovers the trigger without being given the detector. (b) TreeMLP and supporting machinery: a tree-structured policy/value backbone together with relabel-constants and success replay — the configuration ppo-treerc-buf, which matches the prior best on closed CommonCore equations (0.93 greedy vs. ConPoLe’s 0.925; Table 1). (c) A seed-level failure mode and a candidate mitigation: a sharp bimodality at 10× scale in which a run either learns the task or collapses into a policy trap, with a UCB learning-progress curriculum as a non-significant positive mitigation trend.

## 2 Related work

RL for symbolic equation solving. Poesia et al. (2021) introduce ConPoLe for linear equations under low-level axioms (92.5% on CommonCore, our benchmark). Dabelow & Ueda (2024) train an RL agent on a stack-calculator interface for linear manipulation. Most closely related is our own earlier work, O’Keefe (2025), which extends RL to nonlinear closed equations (radicals, exponentials, trigonometric) with PPO over a graph-structured action space and curiosity-driven exploration. Our work difers in scope (we add the open/CoV setting, which requires introducing structure absent from the original equation), exploration mechanism (curiosity bonuses fail to match a no-curiosity baseline on top of TreeMLP; we use success replay and constant relabeling instead), and method (TreeMLP backbone, value-guided beam, bimodality analysis). The open/CoV capability is what we claim as first; nonlinear-closed RL is due to O’Keefe (2025).

Neural symbolic math. Lample & Charton (2020) train a seq2seq Transformer on ${ \sim } 1 0 ^ { 8 }$ supervised pairs to produce one-shot solutions for integration and ODEs. We learn from a per-step reward with no supervised solution traces; the only supervised component is $\pi _ { \mathrm { { c o v } } } .$ , trained on ${ \sim } 1 0 ^ { 3 }$ known substitutions.

Symbolic regression and program synthesis. DSO Petersen et al. (2021) and DreamCoder Ellis et al. (2021) generate formulae; we manipulate a given equation by legal transformations. Li et al. (2022) learn high-level actions for equation solving via expert iteration; our CoV macroaction is similar in spirit but is a single concrete substitution rather than an induced abstraction library.

Graph encoders for symbolic state. GCN Kipf & Welling (2017) and GraphSAGE Hamilton et al. (2017) assume large undirected neighbourhoods; symbolic expression trees are sparse rooted DAGs with a small ordered vocabulary. TreeMLP drops degree normalisation and learned edge transforms in favour of a strictly local sum-aggregation update aligned with tree semantics (Section 3.1).

Formal mathematical reasoning. Peano Poesia & Goodman (2023) and AlphaZero-style theorem provers target proof-style validity rather than computational equation solving; these settings are complementary.

Curriculum learning and capability collapse. Matiisen et al. (2019) introduce learning-progress (LP) curricula; our UCB-LP variant adds a UCB1 bonus on per-class sampling weights Auer et al. (2002), structurally similar to ACTOR-CURATOR Gu et al. (2026). The seed-level bimodality we observe is the symbolic-RL instance of bimodal capability acquisition documented in Zhao et al. (2025); Suau et al. (2024); Dong et al. (2025).

## 3 Problem Formulation

Our Markov Decision Process (MDP) formulation is

States: equations like $a x + b = 0$ or $c x + d = - x / b$ . We vectorize these using preorder traversal over the equation’s expression tree (Appendix A.13, Figure 7) and pad to a maximum length $L = 5 0$ . Operations and symbols are mapped to integers, e.g. {add:1, sub:2, mul: $3 , \ \ldots , \ x : 5 , \ a : 6 , \ b : 7 , \ \ldots \ b$ ; both lhs and rhs are encoded and concatenated. For example, $x + a  [ { \mathrm { a d d } } , x , a ]  [ 1 , 5 , 6 , { \mathrm { P A D } } , \ldots ]$

Actions: represented as (operation, term) pairs, such as (sub, b) or (div, a). Formally,

$$
O = \{ \mathrm { a d d } , \ \mathrm { s u b } , \ \mathrm { m u l } , \ \mathrm { d i v } \}
$$

$$
\cup \{ \mathrm { s q u a r e , ~ s q r t , ~ e x p , ~ l o g , ~ s i n , ~ c o s , ~ a s i n , ~ a c o s } \} ,\tag{1}
$$

$$
T = { \mathrm { S u b E x p r } } ( { \mathrm { l h s } } ) \cup { \mathrm { S u b E x p r } } ( { \mathrm { r h s } } ) ,\tag{2}
$$

$$
A = ( O \times T )
$$

$$
\cup \{ ( \mathrm { e x p a n d , N o n e } ) , ( \mathrm { c o l l e c t } , x ) , ( \mathrm { m u l } , - 1 ) \} .\tag{3}
$$

For the terms, we choose the list of sub-expressions in the lhs and rhs expression trees. Example of subexpressions are

$$
a x + b = 0 \Rightarrow \{ a , x , a x , b \}\tag{4}
$$

$$
\begin{array} { r } { \frac { a x + b } { c x + d } + e = 0 \implies \{ a , x , a x , b , c , d , c x , c x + d , a x + b \} } \end{array}\tag{5}
$$

This term set is expressive enough to solve rational equations and all other equation types we consider in this paper. It is also dynamic: the list of sub-expressions is derived from the equation/state and thus has variable length. We also include (mul, −1) and

expand:

collect x:

$$
\begin{array} { l } { { c x + d + e ( a x + b )  a e x + b e + c x + d } } \\ { { a x + b x  ( a + b ) x } } \end{array}
$$

These allow the agent to perform key algebraic steps (Appendix A). Finally, we index A serially and cap it at $| A | = 5 0$ (the largest set any dataset equation produces is ∼ 40 after dynamic expansion; varying $| A | \in \{ 4 0 , 7 0 , 1 0 0 \}$ produced no material change in single-seed checks). Illegal actions (e.g. division by zero) are masked at each step (Appendix A.3).

Rewards. Define the complexity C of an equation as the total number of nodes and edges in the expression tree.<sup>1</sup> For example, $C ( a x + b ) = 5 + 4 = 9$ and $C ( a x ^ { 2 } + b x + c ) = 1 0 + 9 = 1 9$ . The reward is then

$$
R ( \mathrm { a c t i o n } ) = C ( \mathrm { e q u a t i o n } ) - C ( \mathrm { e q u a t i o n ~ a f t e r ~ a c t i o n } )\tag{6}
$$

i.e., reward actions that simplify the equation.

State Transition Function. We use SymPy Meurer et al. (2017) to apply operations to a tracked $( l h s , r h s )$ pair. For example, from $( a x + b , 0 )$ , (sub, b) gives $( a x , - b )$ and then $( \mathrm { d i v } , a )$ gives $( x , - b / a )$ . The episode terminates when $l h s = x$ and substituting the rhs into the current tracked equation simplifies to 0; after

CoV or constant relabeling, the reported answer is mapped back by exact composition of the stored inverse transformations.

Solving criterion, domain assumptions, and trace validity. We now state precisely what “solved” means, since the paper’s claims should be read against this benchmark-specific criterion rather than against complete symbolic solving.

One terminally verified root, not the solution set. An episode counts as solved when three conditions hold simultaneously: (i) the tracked lhs is exactly the solve variable $x ;$ (ii) the rhs contains no occurrence of $x ;$ and (iii) substituting the candidate rhs for x into the current tracked equation symbolically reduces to 0 under a fixed hierarchy of SymPy canonicalizers (expand, powdenest, ratsimp, simplify, applied independently with operation-count guards). In an episode with no CoV, the tracked equation is the original equation (up to any exactly inverted constant relabeling). With CoV active, the check is performed in the active, post-substitution coordinate system and the verified root is then mapped to the original variable by exact composition of the stored inverse substitutions; we do not run a second independent simplification check against the original pre-CoV expression. The target is thus one satisfying root under this terminal criterion: multi-root equations count as solved when any one verified root is produced (e.g. sin $x = 0$ at $x = 0 )$ , and we make no claim about recovering complete solution sets or about equivalence-preserving derivations.

Domain and branches. All coeficients are symbolic and treated as real; sqrt, log, and the inverse trigonometric operations use $\operatorname { S y m P y } _ { \mathrm { { s } } }$ principal branch. An action whose principal-branch result drops the tracked root (e.g. asin applied when the solution lies on another branch) simply produces a state from which the terminal check cannot pass – the episode fails rather than emitting a wrong answer.

Extraneous and lost roots. Individual actions need not be equivalence-preserving: square can introduce extraneous roots, and division can remove roots. Extraneous candidates that fail the active-equation terminal check are rejected. Lost roots are handled in two parts: divisions by factors that vanish at a root of the equation are masked from the action set (Appendix A.3), and because success requires only one root, losing a diferent root does not compromise the criterion.

What is verified, and when. Intermediate steps are legal algebraic operations applied to both sides via exact symbolic $( \mathrm { S y m P y } )$ transitions, but are not individually checked for equivalence; the terminal criterion is enforced by the active-equation substitution check above. The trace is therefore replayable and inspectable – every step is an explicit symbolic operation – but it is not a proof object certifying every intermediate state equivalent to the original equation. Verification failures of any kind – including SymPy exceptions or canonicalizer op-count limits being exceeded – are conservatively treated as not solved.

Closed vs. open equations. The MDP formulation above only handles equations that are closed, in the sense that solving them requires manipulating terms already present in the equation/sub-expression list. By contrast, solving open equations requires adding new, out-of-equation terms or clever substitutions. A classic example is the quadratic equation $a x ^ { 2 } + b x + c = 0$ which is solved by completing the square – adding $( b / 2 a ) ^ { 2 }$ to each side. We call this reasoning out-of-term-set, since the term $( b / 2 a ) ^ { 2 }$ is not in the term set we have defined; “generative” here means out-of-term-set, not open-ended expression synthesis. Equations that require such out-of-term-set actions are beyond the scope of the closed-action formulation just described, and were not studied in prior RL work either Poesia et al. (2021); Dabelow & Ueda (2024). We extend the formulation to the open setting in Section 4.2 by exposing the change-of-variables as a learned macroaction inside the same dynamic action space.

## 3.1 TreeMLP: a tree-structured policy network

We introduce TreeMLP, a lightweight graph-based policy/value network purpose-built for symbolic expression trees. Nodes are tokens (operators, symbols, or terminals) and edges are parent→child relations. Each node ID is embedded and linearly projected to a hidden size, then $K$ rounds of message passing update the node states with a strictly-local sum aggregator over incoming edges (no degree normalisation): at round $t ,$

$$
h _ { i } ^ { ( t + 1 ) } = \mathrm { M L P } ( \mathrm { [ e m b } _ { i } \parallel \sum _ { j  i } h _ { j } ^ { ( t ) } ] ) ,\tag{7}
$$

where the MLP is two ReLU layers and emb is the static node embedding. Padding-aware masks zero out invalid nodes and edges at every stage. After K iterations a fixed-size graph feature is produced by masked mean (or max) pooling over valid nodes and fed to the PPO policy and value heads.

These choices – direct node-ID embedding, sum aggregation without degree normalisation, no learned edge transforms, masked pooling – align TreeMLP with the rooted, sparse structure of expression trees and avoid two encoder pathologies on this domain: GCN’s Kipf & Welling (2017) degree normalisation dilutes a single parent’s message when an operator has many children (e.g. add with 5 summands), and GraphSAGE’s Hamilton et al. (2017) learned edge transforms overfit the small algebraic vocabulary. TreeMLP outperforms both at equal hidden size on the closed-equation small dataset (Appendix A.10; Figure 4) and trains stably with PPO.

The full architecture diagram, message-passing pseudocode, and ablations against vanilla MLP / GCN / GraphSAGE baselines are in Appendix A.10.

## 3.2 Inductive biases

On top of TreeMLP we use three additional inductive biases.

Curriculum learning. The agent solves a single equation chosen randomly from a larger set each episode. Equations are selected via inverse sampling: the probability of choosing an equation is inversely proportional to the number of times it has been solved before, so the agent spends more time on equations it has not yet mastered.

Success-replay bufer. Solution traces in our environment are exact: once a sequence of $\left( \mathrm { o p , t e r m } \right)$ actions solves an instance $( \mathrm { e . g . , } a x + b = 0 )$ , replaying the same actions deterministically reproduces the solution. We exploit this with a light-weight success-replay path: solution traces $\left( { { s _ { t } } , { a _ { t } } } \right)$ are stored to a memory bufer and periodically mixed into the on-policy rollouts.

Relabel-constants macroaction. To improve parameter sharing across structurally identical equations that difer only by symbol names, the agent has a relabel-constants action that applies a consistent partial renaming $( \mathrm { e . g . , ~ } \{ d \mapsto a , e \mapsto b , \dots \} )$ to the current state, mapping arbitrary coeficient symbols onto a canonical alphabet $\{ a , b , c \}$ . The environment stores the substitution stack and, upon solving, unwinds it so the answer is expressed in the user’s original symbols. This reduces input vocabulary entropy, prevents overfitting to symbol IDs, and makes action terms like “subtract $b ^ { \prime \prime }$ reusable across many instances.

Curiosity bonuses (negative result). Curiosity bonuses (icm, rnd, ngu) fail to match the no-curiosity baseline on top of TreeMLP; details and a plausible explanation are in Appendix A.13.

## 4 Results

## 4.1 Closed equations

Algorithms and training. We use Stable-Baselines3’s Rafin et al. (2021) PPO Schulman et al. (2017) (A2C/DQN were worse in initial experiments); hyperparameters are in Appendix A. Each run is $1 0 ^ { 7 }$ steps with evaluation every $5 \times 1 0 ^ { 5 }$ , over three random seeds.

Datasets. We construct two recursive-template datasets by starting from x and applying random (operation, term) actions, with terms restricted to $\{ a , b , c \}$ (to favour long over wide equations). We discard equations SymPy cannot solve and count multi-root cases solved when any one root is found (e.g. sin $x = 0$ at $x = 0 )$ . small contains all depth-<4 equations (3874, subsampled $1 0 ^ { 3 } / \mathrm { { i 0 ^ { 2 } } }$ train/test); large contains all depth-<5 ones (15625, subsampled $1 0 ^ { 4 } / 1 0 ^ { 3 } )$ ). We additionally use the CommonCore poesia benchmark.

Results. Table 1 reports end-of-training greedy test accuracy for the inductive-bias ablation. On the poesia CommonCore benchmark, ppo-tree-rc-buf reaches 0.93 greedy test accuracy on the full benchmark, matching ConPoLe’s 0.925 Poesia et al. (2021) under matched greedy decoding. Restricting to nondegenerate templates (excluding edge cases the closed-action set cannot reach) raises accuracy further – to 0.955 for ppo-tree-rc and $\geq 0 . 9 8$ on the solvable subset (Appendix A.5). Each bias contributes: TreeMLP alone lifts poesia from 0.17 (vanilla MLP) to 0.80; relabel-constants adds a further jump on small; and the success-replay bufer is critical for scaling – ppo-tree-rc-buf degrades least with depth $( 0 . 8 5  0 . 5 0 $ small→large, vs $0 . 8 5  0 . 1 0$ for ppo-tree-rc).

Table 1: Greedy test accuracy on the three closed-equation benchmarks for the inductive-bias ablation. ConPoLe / ConPoLe-local numbers from Poesia et al. (2021). small/large are our depth-< 4 / depth-< 5 internal benchmarks; poesia is the CommonCore equations split of Poesia et al. (2021).
<table><tr><td>Algorithm</td><td>small</td><td>large</td><td>poesia</td></tr><tr><td>ConPoLe-local ConPoLe</td><td> $\mathrm { n / a }$   $\mathrm { n / a }$ </td><td> $\mathrm { n / a }$   $\mathrm { n / a }$ </td><td>0.765 0.925</td></tr><tr><td>ppo</td><td>0.04</td><td>0.02</td><td>0.17</td></tr><tr><td>ppo-tree</td><td>0.25</td><td>0.10</td><td>0.80</td></tr><tr><td>ppo-tree-rc</td><td>0.85</td><td>0.10</td><td>0.85</td></tr><tr><td>ppo-tree-buf</td><td>0.55</td><td>0.40</td><td>0.85</td></tr><tr><td>ppo-tree-rc-buf</td><td>0.85</td><td>0.50</td><td>0.93</td></tr></table>

Learned representations. t-SNE van der Maaten & Hinton (2008) projections of the TreeMLP pooled embeddings cluster cleanly by equation family (linear, radical, trig, log occupy disjoint regions), indicating the encoder captures structural identity rather than surface tokens (Appendix A.13, Figure 8).

## 4.2 Open equations

We now consider the harder class of open equations: those that cannot be solved by rearranging the subexpressions already present in the equation, but require introducing a new term via a change of variables (CoV) – a substitution $x = \phi ( y )$ that re-expresses the equation in a new variable y. We consider four template families, each paired with its solving substitution:

$$
a x ^ { 2 } + b x + c = 0 , ~ x = y - b / ( 2 a )\tag{8}
$$

$$
a x ^ { 3 } + b x ^ { 2 } + c x + d = 0 , x = y - b / ( 3 a )\tag{9}
$$

$$
a x ^ { 4 } + b x ^ { 3 } + c x ^ { 2 } + d x + e = 0 , ~ x = y - b / ( 4 a )\tag{10}
$$

$$
\textstyle a e ^ { k x } + b e ^ { - k x } + c = 0 , \ x = { \frac { 1 } { k } } \log y\tag{11}
$$

Each substitution depresses (or, for the exponential, rationalizes) the equation, after which standard closedaction steps finish the solution. The polynomial families are restricted to fully-depressable special forms: e.g. the quartic dataset constrains $( c , d , e )$ to functions of $( a , b )$ so that $y = x + b / ( 4 a )$ reduces the equation to $A y ^ { 4 } + B = 0 .$ , solved by two applications of ${ \sqrt { \cdot } } ;$ generic quartics, which admit no closed-form depression by a fixed CoV, are excluded. The exponential case yields a Laurent polynomial in y that resolves to a quadratic after one further CoV (exponential block of Table 11). We expose the CoV as a single macroaction in the same dynamic action space: when the policy selects it, the substitution $\phi$ is generated by a learned policy $\pi _ { \mathrm { c o v } }$ (Section 4.2.6) and applied to the equation. The main policy selects the trigger and composes it with ordinary steps, while $\pi _ { \mathrm { c o v } }$ supplies the substitution; Section 4.2.4 tests which trigger decisions are genuinely non-trivial. This keeps the open problem inside the same MDP, with one policy trained on a mixed closed/open dataset (Section 4.2.1).

Benchmark scope. We call this the Restricted-Open setting: arbitrary same-degree polynomials with nonspecial coeficients (e.g. a generic $ a x ^ { 4 } + b x ^ { 3 } + c x ^ { 2 } + d x + e )$ are out of scope, as they admit no closed-form depression by a fixed CoV. The goal is to study whether RL can learn when and how to invoke a CoV macroaction in a controlled regime where one exists; extending to fully open families is future work.

## 4.2.1 Method

We train a single TreeMLP-PPO policy on mixed datasets of closed and open equations: open\_small (916 train / 91 test, stratified across the four CoV families and the abel\_level1/2/3 closed classes) and a 10×- larger open\_large. (abel\_level1/2/3 are three closed-equation dificulty tiers of increasing tree depth and coeficient structure, abel\_level3 the hardest; throughout, coverage denotes the fraction of training-set equations the agent solves, as distinct from held-out test accuracy.) The policy carries over the closedequation inductive biases – curriculum learning, constant relabeling, and a success-replay bufer (Section 4.1) – with four additions specific to the open setting, which we state up front and ablate below.

![](images/a7cc584efa8e18be089538fec70842f960596e8ed63571a67a99bc888ec6de26.jpg)  
Figure 1: Open-equation learning curves (coverage, greedy and beam-width-5 test accuracy) for open\_small (top) and open\_large (bottom). Red: full method stack; grey dashed: baseline; blue: UCB-LP. Solid lines are seed means, bands min/max; seed counts per cohort are given in the text. Per-seed distributions are in Figure 3.

Action set additions. The open setting adds (i) a cube-root primitive (needed because the depressedcubic CoV collapses an equation to $a y ^ { 3 } + k = 0$ which $\sqrt { \cdot }$ alone cannot finish) and (ii) the CoV macroaction itself, with a per-episode cap of 3 applications to permit nested substitutions (an exponential equation first becomes a rational form, then a quadratic).

Action-diversity penalty. A small reward penalty $( \alpha = 0 . 1 )$ is subtracted whenever the current action’s integer index matches the previous step’s – exact (operation, term) identity, not operation alone. The penalty targets short-cycle loops: 93% of unsolved abel\_level3 failures on the larger benchmark (Section 4.2.5) are 1-cycles (81%, caught by the penalty) or a→b→a 2-cycles (12%, slipping past a one-step penalty; we return to this in Section 6).

Freshness-managed success replay. The bufer discards its oldest half every 20 rollouts. The design goal is to keep the behaviour-cloning target on the current frontier rather than on stale successes; however, the seed-matched ablation in Table 12 finds that its incremental efect changes sign across seeds, so we make no empirical improvement claim for this component.

Value-guided beam search. At evaluation we score beam paths by

$$
\mathrm { s c o r e ( p a t h ) } = \sum _ { t } \log \pi _ { \boldsymbol \theta } ( a _ { t } \mid s _ { t } ) + \lambda V _ { \boldsymbol \theta } ( s _ { T } ) - \boldsymbol \beta C ( s _ { T } ) ,\tag{12}
$$

combining the trained policy log-prob with a value-head look-ahead $( V _ { \theta } )$ and an equation-complexity proxy (C). We use width 5 throughout $( \lambda = \beta = 0$ recovers plain beam). The headline open\_small beam numbers and the seed-matched ablation (Table 12) are decoded at $\lambda = 0 ;$ ; the per-class tables (Tables 14, 15) at $( \lambda , \beta ) \ = \ ( 1 , 0 )$ . The two difer little — on our 3M checkpoints, switching λ moves $t e s t _ { \mathrm { b e a m } }$ by at most

0.044 (mean 0.023) — but we label each table rather than assert equivalence; varying widths {3, 5, 7, 9} and $\lambda \in \{ 0 , 1 \}$ produced no material change at convergence (observed spread $\leq 0 . 0 2 5$ , no table). Beam entries reaching the same canonical equation are deduplicated.

## 4.2.2 Reporting conventions and the two checkpoint cohorts

Unless stated otherwise, checkpoints are selected by validation beam on a disjoint 30% validation slice and reported on the remaining 70% test slice (Section A.14); $t e s t _ { \mathrm { b e a m } }$ is value-guided beam width 5 (Eq. 12); greedy is top-1 decoding of the same checkpoint.

Two greedy accuracies for the full-stack open\_small agent appear in this paper — 0.67 and 0.73 — and they are not in tension: they come from two diferent training cohorts, run at diferent budgets and scored on diferent denominators. Because several reviewers read them as one number, we name the cohorts once here (Table 2) and tag every subsequent figure with the cohort it belongs to. Cohort H (headline) is the 3M-step, five-seed, validation-selected cohort behind every headline number $( t e s t _ { \mathrm { b e a m } } = 0 . 7 9$ , greedy 0.67) and behind Table 3. Cohort W (width/trigger) is a separate eight-seed cohort trained to 5M steps and scored on the full 91-equation test file rather than the 64-equation slice; its five converged seeds give greedy 0.73, and it is the cohort behind Figure 2 and Tables 5–6. Both cohorts use the same validation-based checkpoint selection, so the 0.06 diference is jointly attributable to the longer training budget, the diferent seed draw, and the diferent evaluation denominator; we make no claim about which of the three dominates, and we never pool or compare numbers across the two cohorts. The seed-matched ablation of Table 12 is a third, deliberately separate protocol (three shared seeds, fixed budget, no checkpoint selection) and is likewise never pooled with either cohort.

<table><tr><td></td><td>Cohort H (headline)</td><td>Cohort W (width/trigger)</td></tr><tr><td>training budget</td><td>3M steps</td><td>5M steps</td></tr><tr><td>seeds</td><td>5</td><td>8 (5 converged, 3 stalled)</td></tr><tr><td>checkpoint selection</td><td>validation-best (30/70)</td><td>validation-best (30/70)</td></tr><tr><td>evaluation set</td><td>64-equation test slice (70%)</td><td>full 91-equation test file</td></tr><tr><td>greedy (top-1)</td><td>0.67 (0.40–0.80)</td><td>0.73 (0.55–0.82)</td></tr><tr><td>beam, width 5</td><td>0.79 (0.47–0.93)</td><td>0.84</td></tr><tr><td>reported in</td><td>Sec. 4.2.3, Table 3</td><td>Fig. 2, Tables 5–6</td></tr></table>

Table 2: The two open\_small checkpoint cohorts. Both train the identical full-stack configuration on the identical benchmark; they difer in training budget, seed draw, checkpoint selection, and evaluation denominator. This is the reconciliation of the greedy 0.67 and 0.73 figures: they are the same quantity measured on two diferent cohorts, not two measurements of one cohort. Cohort W’s stalled seeds are reported separately (Appendix A.13) and never pooled with its converged seeds.

## 4.2.3 Results

A plain ppo-tree-rc-buf-cov agent on open\_small reaches coverage = 1.0 on the training set but only $t e s t _ { \mathrm { b e a m } } = 0 . 3 1$ on test (open\_small, n = 1): it memorizes a single CoV recipe and fails to generalize across the four families. Adding the four open-equation ingredients above and selecting the best checkpoint by validation accuracy (Section A.14), we obtain a 5-seed mean $t e s t _ { \mathrm { b e a m } } = 0 . 7 9$ (open\_small, n= 5, range 0.47– 0.93; cohort H throughout this paragraph, Table 2). Its seed-matched baseline is $0 . 3 5 4 \pm 0 . 0 9 0 \ ( n { = } 3 ;$ the n = 1 figure of 0.31 is superseded). The corresponding 5-seed greedy (top-1) mean is 0.67 (range 0.40–0.80)

— the greedy anchor of cohort H, not to be confused with cohort W’s 0.73 — so value-guided beam lifts the greedy policy by ≈0.12 rather than supplying the bulk of the accuracy (the three-way greedy/beam/search comparison is in Table 3 and the beam-width ablation in Figure 2). One of the five seeds early-stopped at 1.95M of 3M steps before fully learning the CoV classes $( t e s t _ { \mathrm { b e a m } } = 0 . 4 7 )$ , and is included in the 5-seed mean as reported. Figure 1 (top row) shows the learning curves with mean and min-to-max band across the five seeds.

Ablation (seed-matched). All four training configurations on the same three seeds, same 3M-step budget, decoder fixed, no checkpoint selection (Table 12). The CoV machinery (cube-root + nested CoV) carries essentially the whole efect: +0.594, positive on every seed. The action-diversity penalty is harmful (−0.245, negative on every seed). The freshness-managed bufer is unresolvable (sign changes across seeds). Removing the penalty $( \alpha = 0 )$ raises the full-stack control from 0.490 to 0.891 and gives end-to-end lifts of $+ 0 . 5 2 1 / + 0 . 5 3 7$ , both positive on every seed, with $0 / 9$ stalls. We retain 0.79 as the headline because all downstream analyses used the $\alpha { = } 0 .$ 1 cohort. The open\_large cohorts (Section 4.2.5) were run at $\alpha { = } 0 .$ , so the bimodality there is not an artefact of this penalty.

Per-class accuracy. On open\_small the full stack solves all four CoV classes at 88–100%, lifting the exponential class from 0% to 88%; abel\_level3 remains weakest (0.53, full stack, 4-seed mean), its failures dominated by short action cycles (breakdown in Table 14). On open\_large the pattern is sharper: across escape seeds the CoV classes saturate (∼ 0.98), abel\_level3 stays weak (escape-seed mean 0.39, range [0.07, 0.55]; Table 15).

Solution trace: nested change-of-variables. The exponential class is the only one of the four CoV families that requires two CoV steps; the agent discovers this nested recipe on its own (full greedy trace in the exponential block of Table 11, Appendix A.8). The policy first applies CoV $( x \to \log x / k )$ to turn the exponential equation into a rational one, expands to a quadratic, then invokes CoV again $( x  x - B / ( 2 A ) )$ to depress the quadratic. The same agent solves quadratic, cubic, and quartic classes with a single-CoV recipe.

Policy vs. search. Table 3 compares three decoders of the same policy against two non-learned searches, all on cohort H. Greedy top-1 alone reaches 0.67 – above BFS (0.26) and $\mathrm { A ^ { * } ~ ( 0 . 3 8 – 0 . 6 4 }$ , same $1 0 ^ { 5 }$ -expansion budget). Value-guided beam adds ≈0.12 to reach 0.79 at ≤250 SymPy transitions per equation. The accuracy is policy-dominated: beam contributes a modest margin, not the bulk of the result. No learned external baseline exists for the restricted-open setting; ConPoLe and Dabelow & Ueda (2024) do not transfer (see Related Work).

<table><tr><td>decoder / search</td><td>policy?</td><td>search?</td><td>open_small acc.</td></tr><tr><td>Greedy (top-1 policy rollout)</td><td>√</td><td></td><td>0.67 (0.40–0.80)</td></tr><tr><td>Value-guided beam (width 5)</td><td>√</td><td>√</td><td>0.79 (0.47–0.93)</td></tr><tr><td>BFS  $( N { = } 1 0 ^ { 5 }$  expansions)</td><td></td><td>√</td><td>0.26</td></tr><tr><td> $\mathrm { A } ^ { * }$  (complexity heuristic,  $1 0 ^ { 5 } )$ </td><td></td><td>√</td><td>0.38-0.64</td></tr><tr><td> $\operatorname { S y m P y }$  oracle</td><td></td><td></td><td>1.00</td></tr></table>

Table 3: Policy vs. search on open\_small (cohort H: 3M steps, 5 validation-selected seeds, 70% test slice — Table $2 ;$ ranges are min–max over seeds where applicable). The learned policy’s greedy top-1 rollout (0.67) alone exceeds the strongest non-learned search over the same action space $( \mathrm { A } ^ { * }$ up to 0.64, BFS 0.26); value-guided beam adds a further ≈0.12 to 0.79. The accuracy is thus policy-dominated: search contributes a modest margin, not the bulk of the result. The SymPy oracle upper-bounds the benchmark.

Decomposing the open-equation pipeline. We re-evaluate the converged open\_large agent (seed109, open\_large, $n { = } 1 ; t e s t _ { \mathrm { b e a m } } = 0 . 8 1$ on the full 1,000-equation held-out file) in three configurations (Table 4): (A) the full learned pipeline; (B) the CoV macroaction disabled (the agent must solve using only closed-action steps); and (C) an oracle CoV (SymPy’s depression substitution, applied on every episode in which the agent invoked the CoV macroaction — 818 of the 1,000 held-out episodes; note this exceeds the 800 CoV-family equations, because the agent also invokes CoV productively on some closed equations, cf. Section 4.2.4). All denominators are the full 1,000-equation held-out file. Configuration B does not show CoV is a trainfrom-scratch necessity: the trained policy simply does not degrade gracefully when CoV is masked at test time, a regime it was never trained for. Configuration C isolates substitution generation: swapping $\pi _ { \mathrm { c o v } }$ for $\operatorname { S y m P y } _ { \mathrm { s } }$ exact depression moves accuracy by ${ \leq } 0 . 0 2$ across escape seeds (the decomposition seed seed109: $0 . 8 1  0 . 7 9 9 ;$ three further escape seeds, $A \to C \colon 0 . 7 9 \to 0 . 8 0 , 0 . 9 0 \to 0 . 9 0 , 0 . 9 1 \to 0 . 9 0 )$ , so the open-equation accuracy comes from the CoV-trigger and closed-action policy, not $\pi _ { \mathrm { c o v } } \mathrm { ^ { * } s }$ decoder.

![](images/eeae524c2163fc06f1287bae038451cd24d1c6c9a5efac0b3db1a60597f8136c.jpg)  
decode width (value-guided beam; 1 = greedy top-1)  
Figure 2: Beam-width ablation on open\_small (cohort W, Table 2). Value-guided-beam test accuracy vs. decode width (width 1 = greedy top-1) across eight full-stack open\_small checkpoints; solid grey lines are the five converged seeds, dotted the three stalled seeds (greedy ≈0, consistent with the seed-level bimodality of Section 4.2.5), red is the mean over the converged seeds with min–max band. On the converged seeds accuracy rises from greedy (0.73) through width 5 (0.84) and then saturates, supporting width 5 as the reported operating point; a stalled policy solves ≈0 at every width, so beam-width is only meaningful once the policy has learned solution paths. (This is cohort W’s greedy anchor of 0.73, not the headline greedy 0.67 of cohort H: these checkpoints were trained to 5M rather than 3M steps, are a diferent seed draw, and are scored on the full 91-equation file rather than the 70% test slice, so they sit above cohort H in absolute level. The width-dependence shape and the high greedy anchor are the point; the levels are not comparable across cohorts.)

Table 4: Pipeline decomposition on open\_large (seed109; beam width $5 ;$ denominator = 1,000 held-out equations). Config B masks the CoV macroaction at test time; config C substitutes SymPy’s exact depression for $\pi _ { \mathrm { c o v } }$ on the 818 episodes in which the agent invoked CoV.
<table><tr><td>config</td><td>what changes</td><td>beam acc</td></tr><tr><td>A</td><td>full learned pipeline</td><td>0.81</td></tr><tr><td>B</td><td>CoV macroaction masked at test</td><td>0.004 (4/1000)</td></tr><tr><td>C</td><td>oracle  $( \operatorname { S y m P y } )$  CoV vs. πcov</td><td>0.799 (799/1000)</td></tr></table>

## 4.2.4 Is the CoV trigger learned, or would a rule do?

A central question is whether the policy has recovered a non-trivial decision about when to invoke a CoV. We therefore test the trigger directly against hand-coded alternatives. We hold the trained policy fixed and mask the CoV action from it, handing the trigger decision to an external rule; the policy still selects every non-CoV action. Four regimes, all greedy-decoded on the same checkpoints:

• never — CoV masked for the whole episode.

• always@0 — fire CoV unconditionally as the first action, then hand over to the policy. Isolates detection from timing.

• rule — fire CoV if the depression detector returns a non-identity substitution on the current form and the CoV budget is unspent. This is the natural rule: fire whenever a CoV is available.

• rule+expand — as rule, but fire only once the tracked lhs is already expanded (expand(ℓ) = ℓ); when a CoV is detectable on an unexpanded form, the rule spends one forced expand step first. This regime is therefore strictly stronger than a trigger: it overrides the policy on one non-CoV action as well. We report it because it is the rule that works, and we flag the extra power it is given rather than presenting it as trigger-only.

All regimes use the same substitution generator as the learned agent, and (with the single exception just noted) delegate every non-CoV action to the trained policy, so the comparison isolates the trigger.

Table 5: CoV trigger ablation. Greedy end-task accuracy on open\_small (5 converged seeds; mean with min–max) and open\_large (seed109). The policy is fixed and identical across rows; what changes is who decides when to fire CoV (and, in the last row only, one forced expand step — see Section 4.2.4). Masking CoV drives the open classes to zero: for this trained policy the macroaction is load-bearing, though as with config B of Table 4 this does not establish that CoV is necessary to learn the task, since the policy was never trained for the masked regime. Firing CoV blindly at t = 0 recovers most, but not all, of the open accuracy, so timing matters. The naive rule does not beat blind firing on open equations — it fires at t = 0 on every one of them (Table 6 explains why). Denominators: the full open\_small test file (n=91: 60 CoVrequiring + 31 closed) and the full 1,000-equation open\_large file; every row shares the same denominator, so the comparison is internal and like-for-like. The cohort is cohort W (Table 2) — the eight open\_small checkpoints of Figure 2 (5M steps, evaluated on the same 91-equation file; the 5 converged seeds, with the 3 stalled seeds reported separately in Appendix A.13 and never pooled) — so the learned row reproduces that figure’s greedy anchor (0.73) exactly, which is our correctness gate on the harness. These are greedy numbers on cohort W and are not comparable to the headline numbers of Section 4.2.3, which use a diferent decoder and cohort H.

<table><tr><td rowspan="2">CoV trigger</td><td colspan="4">open_small (5 seeds)</td><td rowspan="2">open_large</td></tr><tr><td>all</td><td></td><td>open</td><td>closed</td></tr><tr><td>never</td><td></td><td>0.08 [0.05, 0.12]</td><td>0.00</td><td>0.25</td><td>0.004</td></tr><tr><td>always@0</td><td></td><td>0.57 [0.54, 0.60]</td><td>0.73</td><td>0.25</td><td>0.602</td></tr><tr><td>rule (detector only)</td><td></td><td>0.62 [0.54, 0.68]</td><td>0.73</td><td>0.41</td><td>0.603</td></tr><tr><td>learned (ours)</td><td></td><td>0.73 [0.55, 0.82]</td><td>0.90</td><td>0.40</td><td>0.771</td></tr><tr><td>rule + expand</td><td></td><td>0.75 [0.62, 0.82]</td><td>0.92</td><td>0.41</td><td>0.787</td></tr></table>

What the aggregate hides. In aggregate the learned trigger sits between the naive rule (0.62) and the expand-preconditioned rule (0.75), and the latter edges it out. The per-family breakdown (Table 6) shows this aggregate is misleading in both directions, and locates the entire trigger question in one family.

Table 6: Per-family trigger ablation, open (CoV-requiring) equations only, open\_small, cohort W’s 5 converged seeds (75 = 15 equations × 5 seeds per cell). On the three polynomial families the trigger is trivial — the CoV is detectable at t = 0 and firing immediately is optimal — and every rule matches or beats the policy (the policy is 2 behind on cubic and 5 behind on quartic, where it occasionally declines to fire). The exponential family is the only one requiring a nested CoV, and it is the only one where the trigger decision has any content: the naive rule scores zero.
<table><tr><td>family</td><td>never</td><td>always@0</td><td>rule</td><td>learned</td><td>rule + expand</td></tr><tr><td>quadratic</td><td>0/75</td><td>70/75</td><td>70/75</td><td>70/75</td><td>70/75</td></tr><tr><td>cubic</td><td>0/75</td><td>75/75</td><td>75/75</td><td>73/75</td><td>75/75</td></tr><tr><td>quartic</td><td>0/75</td><td>75/75</td><td>75/75</td><td>70/75</td><td>75/75</td></tr><tr><td>exponential</td><td>0/75</td><td>0/75</td><td>0/75</td><td>56/75</td><td>57/75</td></tr></table>

The trigger question is the exponential family. On quadratic, cubic, and quartic the CoV is detectable in the initial state and firing it immediately is optimal; rule therefore fires at t = 0 on every open equation (its first-CoV step is 0 in 60/60 cases), which is why it collapses onto always@0 on the open classes. There is nothing to learn here, and we say so: for single-CoV families a one-line detector is as good as the policy.

The exponential family is diferent. It is the only family requiring two CoVs $( x \to \log x / k$ turns the equation into a rational one, which must then be expanded to a quadratic and depressed; Section 4.2.3). The naive rule solves 0 of 75. The failure is not detection but timing: at the second CoV the tracked lhs is the unexpanded product $x \left( - 3 a / x - 5 c x + 5 e \right)$ . The detector reports a quadratic (it matches on expand $. ( \ell - r )$ which is $- 3 a - 5 c x ^ { 2 } + 5 e x .$ , and returns $x \mapsto x + e / ( 2 c ) )$ ) but the substitution is applied to the raw form, so $- 3 a / x$ becomes $- 3 a / ( x + e / ( 2 c ) )$ ), which never cancels; the episode ends in a rational form the closed actions cannot finish. Expanding first instead yields the clean depressed quadratic $- 3 a - 5 c x ^ { 2 } + 5 e ^ { 2 } / ( 4 c )$ The learned policy does exactly that — expands, then fires — solving $5 6 / 7 5$ from reward alone. Adding that precondition to the rule (rule+expand) recovers $5 7 / 7 5$ and closes the gap.

What we conclude. The learned trigger is matched — narrowly exceeded, by ≈0.02 — by a hand-coded rule, and only by a rule carrying a precondition we obtained by inspecting the learned policy’s traces. We therefore state the contribution as recovery rather than superiority: the policy recovers, from reward alone, a trigger as good as a hand-engineered detector, without being given the detector, and it is the only method here that solves the nested-CoV family without being told how. Where a closed-form depression detector exists per family, a rule sufices and should be used; the learned trigger’s value is that it requires no such detector, which is what makes the formulation portable to settings (e.g. diferential-equation reductions) where no per-family detector is available. We regard this as the honest scope of the “learns when” claim.

Trigger accuracy. As a binary CoV-required classifier over the full $n { = } 9 1$ file (5 converged seeds): precision 0.90, recall 0.97, F<sub>1</sub> 0.93. Recall is near-ceiling; the 0.23 false-positive rate on closed equations is partly benign — 71% of those firings occur on episodes the policy goes on to solve (e.g. squaring a closed equation to produce a solvable quadratic).

## 4.2.5 Scaling to open\_large: a bimodal failure mode and a candidate mitigation

The bimodality. On the 10×-larger open\_large benchmark (10,000 train / 1,000 held-out equations, same composition, split $3 0 / 7 0$ into validation/test), the agent exhibits a sharp seed-level bimodality. Across all cohorts (Table 7; full sweep in Table 16), results partition cleanly into two regimes with no seed landing in (0.10, 0.27): an escape regime $( t e s t _ { \mathrm { b e a m } } \geq 0 . 2$ , observed range [0.27, 0.91]) and a stall regime $( t e s t _ { \mathrm { b e a m } } \leq 0 . 1 0 )$ . The same code, data, and hyperparameters produce both regimes; only the seed difers. This resembles bimodal-across-seeds capability acquisition in LM pretraining Zhao et al. (2025) and the policy confounding Suau et al. (2024) / capability boundary collapse Dong et al. (2025) failure modes.

Mechanism: per-class trap dynamics. A post-hoc analysis of in-training solve events classifies each successful step by the source class of the equation it solved. Escape seeds distribute their solves roughly uniformly across the four CoV classes (20–25% each) and abel\_level3 (∼ 10%). Stall seeds concentrate $\geq 8 5 \%$ of their solves on abel\_level3, the easiest class during training, and approach zero on the CoV classes. Stalled seeds did learn – they routinely solve training abel\_level3 instances – but fall into short action cycles on held-out instances. Escape correlates with class diversity in training events; stall corresponds to abel\_level3 lock-in, via the short-cycle failure pattern quantified in Section 4.2.1 (93% of unsolved cases are 1- or 2-cycles).

Across three probes we find no evidence for the three most natural alternative explanations: capacity loss / dormant neurons Sokar et al. (2023) (0% dormant on both stuck and escape seeds at the standard ReDo threshold), data-distribution skew (an abel\_level3-skewed training set transfers as 0.04 to the standard split), and encoder representation collapse (t-SNE clusters cleanly on stalled seeds). Details in Appendix A.13. These probes are necessary-not-suficient (a clean encoder does not rule out a head-level issue), but the evidence is most consistent with a policy-attractor phenomenon.

Intervention sweep. We evaluated seven intervention cohorts targeting the failure mode; Table 7 reports the headline cohorts (baseline, vanilla LP curriculum, UCB-LP) with the full 9-row sweep in Appendix A.13. The most efective is a UCB-style learning-progress curriculum (UCB-LP), which replaces vanilla LP’s Matiisen et al. (2019) heuristic exploration floor with a UCB1 Auer et al. (2002) bonus on per-class sampling weights: $w _ { c } \propto \hat { p } _ { c } + \beta \sqrt { \log T / n _ { c } } \ ( \hat { p } _ { c } = \mathrm { E M A }$ learning progress on class $c , n _ { c }$ its visit count, $\beta = 2 . 0 )$ , structurally similar to ACTOR-CURATOR Gu et al. (2026). Across $n = 8$ training runs it achieves $6 / 8$ escape with mean beam = 0.83 over escape seeds (range 0.69–0.91); the best UCB-LP seed reaches 0.911 (+0.10 over the previous single-seed best of 0.81). The escape-rate change from $3 / 5$ (baseline) to $6 / 8$ (UCB-LP) is not statistically significant (Fisher exact p ≈ 1.0; Wilson CIs overlap); a powered claim would need $\geq 2 0$ seeds per cohort. UCB-LP should therefore be read as a candidate mitigation direction, not a demonstrated fix. The two non-escape seeds crashed on a SymPy/mpmath OverflowError while already stalled; we count them as stalls $( 6 / 8 )$ , giving $6 / 6$ if excluded. UCB-LP was selected for expansion to $n = 8$ because it won the initial 3-seed screen, so cross-cohort comparisons are exploratory. All numbers are val-best per seed on a 30/70 split (Appendix A.14).

Table 7: Intervention sweep on open\_large. We report test beam at the validation-selected checkpoint for each seed (checkpoint chosen by val beam on the 30% split; test beam on the disjoint 70% split). Escape $= b e a m \ge 0 . 2$ . Mean is reported both over all seeds (mixing escape and stall) and over escape seeds only, since the bimodal regime makes the all-seeds mean a noisy summary. Crash convention: the two UCB-LP seeds that crashed on a SymPy/mpmath overflow were already stalled pre-crash; we count them as stalls $( 6 / 8 )$ , and parenthetically report the figure excluding them $( 6 / 6 )$ . The Fisher/Wilson statistics on these counts are in the text. Headline cohorts only; full sweep in Appendix A.13. LP follows Matiisen et al. (2019); oracle = SymPy on the same test set.
<table><tr><td>Cohort</td><td>n</td><td>Escape rate</td><td>Mean (all)</td><td>Mean (escape)</td><td>[min, max] escape</td></tr><tr><td>Baseline (full stack)</td><td>5</td><td> $3 / 5$ </td><td>0.30</td><td>0.47</td><td>[0.27,0.83]</td></tr><tr><td>LP curriculum (vanilla)</td><td>3</td><td>2/3</td><td>0.40</td><td>0.58</td><td></td></tr><tr><td>UCB-LP curriculum</td><td>8</td><td>6/8 (6/6)</td><td>0.63</td><td>0.83</td><td>[0.69, 0.91]</td></tr><tr><td>Oracle (SymPy)</td><td></td><td></td><td>1.00</td><td>1.00</td><td></td></tr></table>

![](images/ad8ce611ee97f7d022e09c3eb249be69cdf9089b3deda70840371e4c82bfac48.jpg)  
Figure 3: Per-seed val-best test beam on open\_large. Each dot is one training run; horizontal bar is the cohort mean. The escape mode (beam $\geq 0 . 2 ;$ escapes cluster near 0.7–0.9) is more populated and reaches higher values under UCB-LP, while the stall mode at ∼0.02 is preserved (UCB-LP: 2/8 stalls, both from crashed seeds whose pre-crash trajectories were already stalled).

## 4.2.6 Learning the change of variables

$\pi _ { \mathrm { c o v } }$ maps an equation to its depression substitution, trained by supervised learning on the known closedform targets (grammar and worked example in Appendix A.7). It achieves 100% on 192 held-out test equations across eight seeds under greedy decoding, generalizing to renamed coeficients within the four training families; generalization to new substitution structures or template families outside the training grammar is untested and is future work. As shown in Section 4.2.3 (config C), it is interchangeable with a CAS depression call and is not load-bearing for end-task accuracy; the RL problem isolates when and how the policy invokes CoV, not the substitution generator.

## 5 Conclusion

We presented the first reinforcement-learning agent to solve a controlled class of restricted-open (changeof-variables) symbolic equations step-by-step with the main policy trained from reward alone (the CoV substitution generator is supervised but interchangeable with a CAS call), extending RL equation solving from the nonlinear closed case (O’Keefe, 2025) to the restricted-open setting – recovering a CoV-conditioned procedure, including trigger timing where it is non-trivial, and composing CoV with ordinary algebraic steps under a single policy. The trigger ablation (Section 4.2.4) narrows the claim to recovery: on the three single-CoV families (quadratic, cubic, quartic) a one-line rule matches or beats the policy and nothing is learned; learned timing has content only for the nested-CoV exponential family, where the natural rule solves $0 / 7 5$ and the policy solves $5 6 / 7 5$ from reward alone — matched by a rule only once handed a precondition read of the policy’s traces. Recovery needs only a reward; the rule needs a known closed-form depression per family, which is the whole dificulty in the ODE reductions this formulation targets. Under the termina criterion of Section 3, it reaches 0.79 beam-decoded test accuracy (greedy top-1 mean 0.67, itself above all non-learned search baselines, with beam adding a further margin; both on cohort H of Table 2) on the four restricted-open families – quadratic, cubic, quartic, and exponential, each restricted to forms with a known closed-form depression – beyond the closed-equation setting of prior $\mathrm { R L } ,$ and matches the prior best on closed CommonCore equations, while exposing a replayable, terminally checked trace that one-shot neural solvers and a bare CAS do not provide. The principal open problems are the gap between decode-time and trainingtime search (a natural fit for expert iteration) and the seed-level bimodality at 10× scale; no matched $\alpha = 0$ small-scale control cell stalls, while the learning-progress curriculum at 10× shows a non-significant positive trend (a candidate mitigation, not a demonstrated fix). We see the formulation as a step toward RL for the procedural reasoning that diferential-equation reductions and prover tactics require.

## 6 Limitations

Several limitations double as concrete next steps.

Our open\_small headline cohort carries a reward term that suppresses the matched controls. Under the seed-matched control the penalty costs −0.245 (negative on every seed); removing it $( \alpha = 0 )$ raises the full-stack control from 0.490 to 0.891 with $0 / 9$ stalls, and end-to-end lifts $\mathrm { o f + 0 . 5 2 1 / + 0 . 5 3 7 }$ positive on every seed (Table 12).

We have not promoted $\alpha = 0$ to the headline because all downstream analyses used $\alpha { = } 0 . 1$ checkpoints; a decode-only re-run of the trigger control at $\alpha = 0$ leaves the “recovery, not superiority” finding unchanged $( 5 8 / 7 5 \ \mathrm { v s } 5 6 / 7 5 $ on the exponential family). We decline to call 0.79 a lower bound: the seed-matched protocol gives $0 . 4 9 0 \pm 0 . 3 7 7$ with $\mathrm { ~ a ~ } 1 / 3$ stall rate, so 0.79 may be a favourable seed draw; retraining at $\alpha = 0$ under the headline protocol is the concrete next step.

The ablation resolves only large efects. Parallel training is not deterministic at a fixed seed, so seedmatching removes the seed-selection confound but not a residual run-to-run variance (0.079 mean, 0.253 worst case) that no seed count averages away without per-cell replicates. Table 12 can therefore resolve the large sign-stable directions but neither the freshness-bufer increment nor the penalised full-stack-versusbaseline contrast, both of which flip sign. A 3/3 sign agreement carries a floor two-sided p of 0.25: these are corroborated directions, not powered tests (Henderson et al., 2018; Agarwal et al., 2021), and we use them to revise no headline number.

The underlying equations are CAS-solvable. Every equation we report is solved in milliseconds by SymPy; our contribution is the RL formulation and the learned step-by-step procedure, not the underlying solvability. The framework’s value is in downstream settings – diferential-equation solving, theorem-prover tactics, didactic solvers – where the trace matters; on bare equation-solving a CAS comparison is unfavourable and would be misleading.

Residual variance under UCB-LP. UCB-LP shows a non-significant positive trend $( 6 / 8$ escape, Fisher $p \approx 1 . 0 )$ ; two seeds still stall. Tighter curricula (e.g. Thompson-sampling class gating Wu et al. (2026)) and a hardened numerical pipeline are the concrete next steps; the $2 / 8$ stall rate and 0.09 gap to oracle are the principal open problems on open\_large.

Structural ceilings and the weak class. The cubic class was unsolvable until we added a cube-root action, and analogous gaps may exist outside our four CoV templates (e.g. quintics, with no radical solution): the action set bounds the solvable class in a way learning cannot overcome. Separately, abel\_level3 stays weakest (0.53 full-stack on open\_small; escape-seed mean 0.39, range [0.07, 0.55], on open\_large); its dominant failure is the policy-looping pattern of Section 4.2.5, pointing to reward shaping rather than an action-set gap.

Search at evaluation, not training. Value-guided beam adds a further margin over the greedy policy at decode time (0.67 → 0.79 on cohort H; 0.73 → 0.84 on cohort W), yet search is applied only at evaluation – the policy trains for greedy rollouts. Folding decode-time search back into training (expert iteration / AlphaZero-style self-improvement Silver et al. (2018)) to close even this residual gap is the natural and most promising direction beyond this paper.

## Broader Impact Statement

This work applies RL to step-by-step symbolic equation solving. The equations are already solvable in milliseconds by computer algebra systems, so the method confers no new equation-solving capability and poses no direct misuse risk; its benefits are indirect, in the explicit, replayable traces it produces for tutoring and for procedural reasoning in ODE solving and prover tactics.

## Reproducibility Statement

The MDP formulation, state/action encoding, reward, and the solving criterion (one terminally checked root; domain and branch conventions; active-equation substitution followed by exact inverse composition) are specified in Section 3; the TreeMLP architecture, pseudocode, and a minimal implementation are in Appendix A.10. PPO and method hyperparameters are tabulated in Appendix A (Tables 8 and 9), and dataset generation is described in Section 4.1 (closed) and Sections 4.2–4.2.6 (open $/ \pi _ { \mathrm { c o v } } )$ , with the rationalequation construction in Appendix A. The validation-based checkpoint-selection protocol and the 30/70 split (seed 42) are given in Appendix A.14; per-seed counts and the crash-counting convention are stated alongside each cohort. We release the environment and benchmark at https://github.com/Khev/abel-rl-env: the MDP implementation, the dynamic action space and legal-action mask of Appendix A.3, the changeof-variables macroaction and its nesting cap, the curriculum / success-replay / constant-relabel machinery and the UCB learning-progress curriculum, the observation encoders, and — the component on which every reported number depends — the terminal verification check that implements the solving criterion of Section 3. The release also contains the exact train/test splits and the generation scripts for small, poesia, open\_small, open\_large, cov\_large, and the abel\_level1/2/3 closed tiers, together with a runnable example that instantiates both a closed and a restricted-open task, exercises the CoV macroaction, and exhibits verified solves with their replayable traces.

We do not release the agent: the TreeMLP policy/value network, the GCN/GraphSAGE baselines, the PPO training loop, the launch scripts, or the trained checkpoints. The consequence should be stated plainly rather than left implicit. This release is suficient to re-derive what the paper counts as a solution, to audit the verification criterion directly, and to reuse or re-score the restricted-open benchmark; combined with the architecture pseudocode and minimal implementation of Appendix A.10 and the hyperparameters of Appendix A (Tables 8 and 9), it is suficient to reimplement the method. It is not suficient to reproduce the reported seeds exactly. Independently of the release, for a subset of the earlier open\_large cohorts (Tables 7 and 16) we retain the seeds and recorded metrics but not an executable launch script, so their hyperparameters are attested by run records rather than by a script; the seed-matched ablation of Table 12 and the α =0 cells were run from scripts retained verbatim.

## References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2):235–256, 2002.

Lennart Dabelow and Masahito Ueda. Symbolic equation solving via reinforcement learning. Neurocomputing, 613:128732, 2024.

Yihong Dong, Xue Jiang, Yongheng Tao, et al. RL-PLUS: Countering capability boundary collapse of LLMs in reinforcement learning with hybrid-policy optimization. arXiv preprint arXiv:2508.00222, 2025.

Kevin Ellis, Catherine Wong, Maxwell Nye, Mathias Sablé-Meyer, Luc Morales, Luke Hewitt, Luke Cary, Armando Solar-Lezama, and Joshua B. Tenenbaum. DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. In ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 2021.

Zhengyao Gu, Jonathan Light, Raul Astudillo, Ziyu Ye, Langzhou He, Henry Peng Zou, Wei Cheng, Santiago Paternain, Philip S. Yu, and Yisong Yue. Actor-curator: Co-adaptive curriculum learning via policyimprovement bandits for RL post-training. arXiv preprint arXiv:2602.20532, 2026.

William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In AAAI Conference on Artificial Intelligence, 2018.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR), 2017.

Guillaume Lample and François Charton. Deep learning for symbolic mathematics. In International Conference on Learning Representations (ICLR), 2020.

Zhening Li, Gabriel Poesia, Omar Costilla-Reyes, Noah Goodman, and Armando Solar-Lezama. LEMMA: Bootstrapping high-level mathematical reasoning with learned symbolic abstractions. In NeurIPS Workshop on MATH-AI, 2022.

Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher–student curriculum learning. In IEEE Transactions on Neural Networks and Learning Systems, 2019.

Aaron Meurer, Christopher P. Smith, Mateusz Paprocki, Ondřej Čertík, Sergey B. Kirpichev, Matthew Rocklin, Amit Kumar, Sergiu Ivanov, Jason K. Moore, Sartaj Singh, Thilina Rathnayake, et al. SymPy: symbolic computing in Python. PeerJ Computer Science, 3:e103, 2017.

Kevin P. O’Keefe. Curiosity-driven reinforcement learning for symbolic equation solving. NeurIPS 2025 MATH-AI Workshop; arXiv:2510.17022, 2025.

Brenden K. Petersen, Mikel Landajuela, T. Nathan Mundhenk, Claudio P. Santiago, Soo K. Kim, and Joanne T. Kim. Deep symbolic regression: Recovering mathematical expressions from data via riskseeking policy gradients. In International Conference on Learning Representations (ICLR), 2021.

Gabriel Poesia and Noah D. Goodman. Peano: Learning formal mathematical reasoning. In Philosophical Transactions of the Royal Society A, volume 381, 2023.

Gabriel Poesia, WenXin Dong, and Noah Goodman. Contrastive reinforcement learning of symbolic reasoning domains. In Advances in Neural Information Processing Systems, volume 34, pp. 17729–17741, 2021.

Antonin Rafin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419):1140–1144, 2018.

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, and Utku Evci. The dormant neuron phenomenon in deep reinforcement learning. In International Conference on Machine Learning (ICML), 2023.

Miguel Suau, Matthijs T. J. Spaan, and Frans A. Oliehoek. Bad habits: Policy confounding and out-oftrajectory generalization in RL. Reinforcement Learning Journal, 4:1711–1732, 2024.

Laurens van der Maaten and Geofrey Hinton. Visualizing data using t-SNE. Journal of Machine Learning Research, 9:2579–2605, 2008.

Yuning Wu, Ke Wang, Devin Chen, and Kai Wei. Hindsight-anchored policy optimization: Turning failure into feedback in sparse reward settings. arXiv preprint arXiv:2603.11321, 2026.

Rosie Zhao, Tian Qin, David Alvarez-Melis, Sham Kakade, and Naomi Saphra. Random scaling of emergent capabilities. arXiv preprint arXiv:2502.17356, 2025.

## A Supplementary material

## A.1 Macroactions

Here we justify why the three macroactions (expand, None), (collect, x), (mul, −1) discussed in the main text are required to solve the equations we consider, not mere conveniences for the RL agent. Consider the action set without these macroactions:

$$
O = \{ \mathrm { a d d } , \ \mathrm { s u b } , \ \mathrm { m u l } , \ \mathrm { d i v } \}
$$

$$
\cup \{ \mathrm { s q u a r e , ~ s q r t , ~ e x p , ~ l o g , ~ s i n , ~ c o s , ~ a s i n , ~ a c o s } \} ,\tag{13}
$$

$$
T = { \mathrm { S u b E x p r } } ( { \mathrm { l h s } } ) \cup { \mathrm { S u b E x p r } } ( { \mathrm { r h s } } ) ,\tag{14}
$$

$$
A = ( O \times T ) .\tag{15}
$$

Now consider the equation $d x + c ( a x + b ) = e $ . The agent must distribute c into the bracketed term $a x + b ,$ which the action set above cannot do: expand is required. Once expanded to $d x + c a x + c b = e$ , one needs to factor the x common to the first two terms—this is what (collect, x) provides. Finally, consider $- x = b ;$ since −1 is not in our term set, (mul, −1) is needed to isolate x. (Alternatively, −1 could be added to the term set, but we chose to keep the term set purely symbolic.)

## A.2 Dataset generation: rational equations

To supplement the recursive datasets, we constructed rational equations of the form

$$
{ \frac { a x + b } { c x + a } } + b = 0 ,
$$

where $a , b ,$ c are (non-zero) symbolic coeficients. We excluded degenerate cases such as denominators that vanish identically or cancel trivially with numerators.

Examples of retained equations include:

$$
{ \frac { x + b } { c x + a } } = 0 \quad \Rightarrow \quad x = - b ,
$$

$$
{ \frac { a x + b } { c x + a } } + b = 0 \quad \Rightarrow \quad x = { \frac { - b - a b } { a + c b } } ,
$$

$$
{ \frac { a x + b } { c x + a } } - c = 0 \quad \Rightarrow \quad x = { \frac { - b + a c } { a - c ^ { 2 } } } .
$$

These rational forms were merged into both the small and large datasets, with their proportion limited to preserve diversity across other functional forms (logarithmic, trigonometric, exponential, etc.).

## A.3 Illegal actions

The main issue was to avoid illegal division by zero. This is not as simple as precluding (div, 0) from the action set. We also had to prohibit hidden division by zero—for example, dividing by $( x + a ) { \mathrm { ~ o r ~ } } ( x + b )$ when the equation has the form $( x + a ) ( x + b ) = 0$ , since one of the factors will vanish at the solution. To handle this, we check whether the current equation factors as $P ( x ) Q ( x ) \cdot \cdot \cdot = 0$ with $P , Q , \dots$ . polynomial in x, and remove $( \mathrm { d i v } , P ) , ( \mathrm { d i v } , Q ) , . .$ . from the action set for that state.

## A.4 Hyperparameters and compute

We used Stable Baselines 3’s default hyperparameters for PPO (Table 8), together with the architecture and method constants stated in the body and consolidated in Table 9.

Compute budget. All experiments run on a single 24-core CPU machine (no GPU). The PPO device setting is left at its Stable-Baselines3 default of ’auto’ (Table 8), but because the machine has no GPU this resolves to CPU for every run reported here. Each open\_large training is $5 \times 1 0 ^ { 6 }$ steps; the open\_small headline and seed-matched controls use $3 \times 1 0 ^ { 6 }$ , while the beam-width/trigger cohort uses $5 \times 1 0 ^ { 6 }$ . With $n _ { \mathrm { e n v s } } = 1$ , an open\_large seed takes 20–30 wall-hours (one CPU core per training; helpers for in-training eval and post-hoc beam-curve eval add up to four more cores transiently). The full $n = 8$ UCB-LP cohort plus n = 5 baseline cohort plus the LP-only and intervention-sweep cohorts amount to roughly 1,500–2,000 core-hours of training, dominated by the $5 \times 1 0 ^ { 6 } .$ -step open\_large runs. Post-hoc beam-curve evaluations for Figure 1 add an additional ∼ 50 core-hours. Closed-equation experiments $( 1 0 ^ { 7 }$ steps each) account for another ∼ 300 core-hours across all cohorts. Total: ∼ 2,500 core-hours, no GPU.

Algorithm Hyperparameters   
PPO learning\_rate = 0.0003, n\_steps = 2048, batch\_size = 64, n\_epochs = 10,   
gamma $\begin{array} { r l } { = } & { { } 0 . 9 9 . } \end{array}$ , gae\_lambda = 0.95, clip\_range $= \ 0 . 2 ,$ ent\_ $_ { c o e f } ~ = ~ 0 . 0 ,$   
vf\_coef = 0.5, max\_grad\_norm $\qquad = \quad 0 . 5 ,$ normalize\_advantage = True,   
device =<sup>′</sup> auto<sup>′</sup>   
PPO-tree learning\_rate = 0.0007, n\_steps = 5, gamma = 0.99  
Table 8: Default PPO hyperparameters from Stable Baselines 3. The $\mathtt { d e v i c e } ^ { \mathtt { - } } \mathtt { a u t o } ^ { \mathtt { - } }$ default resolves to CPU on our no-GPU machine.

<table><tr><td>Constant</td><td>Value</td></tr><tr><td>TreeMLP message-passing rounds K</td><td>3</td></tr><tr><td>TreeMLP hidden size H</td><td>128</td></tr><tr><td>TreeMLP embedding dim</td><td>64</td></tr><tr><td>TreeMLP pooling</td><td>mean or max (Sec. 3.1)</td></tr><tr><td>max state length L</td><td>50</td></tr><tr><td>action-set cap |A|</td><td>50</td></tr><tr><td>action-diversity penalty α</td><td>0.1 (open_smal1 headline); 0 in open_1arge headline†</td></tr><tr><td>value-beam  $( \lambda , \beta )$ </td><td>(1, 0) (per-class tables; λ=0 for headline &amp; ablation)</td></tr><tr><td>beam width buffer-refresh period</td><td>5</td></tr><tr><td>CoV per-episode cap</td><td>every 20 rollouts (discard oldest half) 3</td></tr><tr><td>UCB-LP exploration coeff. β</td><td></td></tr><tr><td></td><td>2.0</td></tr></table>

Table 9: Architecture and method constants, consolidated from the body and the training configuration. <sup>†</sup>Correction (this revision): the action-diversity penalty is not a global constant, as earlier versions implied. It was set to $\alpha { = } 0 . 1$ for the open\_small headline and related penalised cohorts. The open\_large baseline, LP, UCB-LP and the intervention sweep’s non-anti-loop rows, as well as the closed CommonCore runs, used the default α =0; separate anti-loop sweep rows deliberately used $\alpha \in \{ 0 . 3 , 1 . 0 \}$ (Table 16). This matters for two claims: (i) the seed-level bimodality of Section 4.2.5 is present in the $\alpha = 0$ headline cohorts, so it is not an artefact of our reward shaping; (ii) the harm in the matched open\_small control (Table 12) applies to the cohorts that used $\alpha { = } 0 . 1$ , including our headline.

## A.5 Closed-equation analysis

This appendix expands the closed-equation results of Section 4.1: a per-structural-type accuracy breakdown, the dominant failure modes, and how performance changes from the small to the large dataset.

Per-type accuracy. The “small” dataset contains a heterogeneous mix of equation types. Bucketing the test set by structural class and re-evaluating the trained ppo-tree-rc-buf agent reveals a clear pattern (Table 10): linear and radical equations are nearly fully solved $\left( \ge \ 0 . 8 8 \right)$ , polynomial equations of degree 2–4 sit in the 0.75–0.82 range, while transcendental forms—particularly trigonometric and logarithmic—are the hardest. The trigonometric bucket (113 equations, 72% accuracy) accounts for most of the remaining failures. Note that the polynomial buckets contain only “depressed” forms $( \mathrm { e . g . ~ } a x ^ { 2 } + c , x ^ { 4 } + c )$ that arise naturally from recursive action application; true open polynomials with a linear cross-term $( a x ^ { 2 } + b x + c )$ are not in this dataset and are instead studied in Section 4.2.

Table 10: Greedy test accuracy of ppo-tree-rc-buf on the small dataset, broken down by equation class. $\mathrm { \ddot { ~ } p o l y - d e g } k ^ { \mathrm { \prime \prime } }$ includes only the depressed forms reachable by the closed-equation action set $( \mathrm { e . g . ~ } a x ^ { 2 } + c , $ $x ^ { 4 } + c )$ ; equations with a linear cross-term are out of this dataset.
<table><tr><td>Type</td><td>n</td><td>solved</td><td>acc</td></tr><tr><td>radical  $( \mathrm { e . g . ~ } \sqrt { x } )$ </td><td>47</td><td>43</td><td>0.915</td></tr><tr><td>linear (poly-deg1)</td><td>110</td><td>97</td><td>0.882</td></tr><tr><td>quadratic (poly-deg2)</td><td>34</td><td>28</td><td>0.824</td></tr><tr><td>log</td><td>77</td><td>58</td><td>0.753</td></tr><tr><td>poly-deg4</td><td>4</td><td>3</td><td>0.750</td></tr><tr><td>trig</td><td>113</td><td>81</td><td>0.717</td></tr><tr><td>overall</td><td>387</td><td>312</td><td>0.806</td></tr></table>

Failure modes. Manual inspection of the failed cases (75 of 387) shows three recurring patterns: (i) trig equations that require an arccos/arcsin step the agent never selects, (ii) log equations whose simplification creates a transcendental that breaks the closed-action set, and (iii) deep polynomials where the agent exceeds the per-episode action budget despite being on a productive path.

Scaling to the large dataset. The same per-type evaluation on the “large” dataset (1000 test equations of depth $< 5 )$ reveals a uniform drop in every category: linear $0 . 8 8  0 . 6 0$ , quadratic $0 . 8 2  0 . 4 0$ , log $0 . 7 5  0 . 2 7$ , trig $0 . 7 2  0 . 3 6$ , radical $0 . 9 2  0 . 5 2$ . The drop is largest for deep polynomials (poly-deg4: $0 . 7 5  0 . 0 8 )$ . Going from depth $< 4$ to depth $< 5$ approximately quadruples the test-set size and adds substantially deeper nesting, so the harder cases are genuinely harder, not just more numerous.

Generalization to CommonCore. On the CommonCore (Poesia) benchmark, ppo-tree-rc-buf reaches 0.93 greedy test accuracy on the full benchmark (Table 1), matching ConPoLe’s 0.925. The non-bufer ppotree-rc variant scores 0.85 on the full benchmark; restricting the denominator to non-degenerate templates – excluding edge-case equations that lack a free x in the polynomial decomposition, which the closed-action set cannot solve – raises its greedy accuracy to 0.955 (linear 0.98, small rational subset 1.00), and $\geq 0 . 9 8$ on the solvable subset. The 0.85 and 0.955 figures for ppo-tree-rc thus difer only in the denominator (full benchmark vs. non-degenerate subset); both use greedy decoding.

## A.6 Open-equation embeddings (converged vs. stalled seeds)

To diagnose whether the bimodal failure on open\_large (Section 4.2.5) is a representation problem or a policy problem, we extract the pooled TreeMLP graph embedding for each test equation and project it to two dimensions with PCA and t-SNE. Both the converged seed (seed109000; $t e s t _ { \mathrm { b e a m } } = 0 . 8 1 )$ and a stalled seed (seed108000; test<sub>beam</sub> = 0.03) produce clean class clusters (quadratic / cubic / quartic / exponential each form a tight clump in t-SNE space; abel\_level3 is more difuse). The encoder thus learns the equationfamily structure in both cases; the failure is in the policy/value head’s ability to act on it, not in the input representation.

## A.7 Change-of-variables grammar $( \pi _ { \mathrm { c o v } } )$

$\pi _ { \mathrm { c o v } }$ decodes the substitution as a prefix sequence of grammar productions; production arities make decoding self-delimiting (no explicit length field or brackets needed). For example, $x - b / ( 2 a )$ is emitted as the token sequence sub x div copy mul int2 copy: sub takes two children (x and the quotient), div takes the copied symbol (b) and the product $2 a .$ mul takes int2 and a second copied symbol (a), and each copy names a coeficient symbol of the current equation. Because the model chooses each production – whether to subtract, what to divide by, which coeficients to use – it selects the substitution’s structure rather than filling a fixed template.

## A.8 Open-equation solution traces

Below are full greedy solution traces for one held-out test equation from each source class of open\_large, produced by the converged seed109 agent. The four change-of-variables classes (quadratic, cubic, quartic, exponential) share the same recipe up to constants (cov → relabel → mul(−1) → mul x → collect $x $ div → relabel); only the exponential class requires a second CoV at step 4. The depth column counts active CoV substitutions.

<table><tr><td>step</td><td>action</td><td>depth</td><td>current lhs (rhs = 0)</td></tr><tr><td colspan="4">quadratic  $\dot { \cdot } - a x ^ { 2 } + 2 c x + 5 d = 0$ </td></tr><tr><td>1</td><td>COV</td><td>1</td><td> $a x ^ { 2 } + 5 d - c ^ { 2 } / a$ </td></tr><tr><td>2</td><td>RELABEL</td><td>1</td><td> $a + b x ^ { 2 }$ </td></tr><tr><td>3</td><td> $\mathrm { \ m u l { ( - 1 ) } }$ </td><td>1</td><td> $- a - b x ^ { 2 }$ </td></tr><tr><td>4</td><td>mul x</td><td>1</td><td> $x ( - a - b x ^ { 2 } )$ </td></tr><tr><td>5</td><td>collect x</td><td>1</td><td> $x ( - a - b x ^ { 2 } )$ </td></tr><tr><td>6</td><td> $\mathrm { d i v } ( - a - b x ^ { 2 } )$ </td><td>1</td><td>x</td></tr><tr><td>7</td><td>RELABEL</td><td>1</td><td> $x \left[ = - c / a \right]$ </td></tr><tr><td colspan="4">cubic and quartic: identical 7-step recipe (only the degree of x differs); see text.</td></tr><tr><td colspan="4">exponential (nested CoV) − 2b ex + 2d −  $6 f e ^ { - x } = 0$ </td></tr><tr><td>1</td><td>COV</td><td>1</td><td> $2 b / x + 2 d - 6 f x$ </td></tr><tr><td>2</td><td>mul x</td><td>1</td><td> $x ( 2 b / x + 2 d - 6 f x )$ </td></tr><tr><td>3</td><td>EXPAND</td><td>1</td><td> $2 b + 2 d x - 6 f x ^ { 2 }$ </td></tr><tr><td>4</td><td>coV (nested)</td><td>2</td><td> $2 b + d ^ { 2 } / ( 6 f ) ^ { \top } - 6 f x ^ { 2 }$ </td></tr><tr><td>5</td><td>RELABEL</td><td>2</td><td> $a + b x ^ { 2 }$ </td></tr><tr><td>6</td><td>mul(−1)</td><td>2</td><td> $- a - b x ^ { 2 }$ </td></tr><tr><td>7</td><td>mul x</td><td>2</td><td> $x ( - a - b x ^ { 2 } )$ </td></tr><tr><td>8</td><td>collect x</td><td>2</td><td> $x ( - a - b x ^ { 2 } )$ </td></tr><tr><td>9</td><td> $\mathrm { d i v } ( - a - b x ^ { 2 } )$ </td><td>2</td><td>x</td></tr><tr><td>10</td><td>RELABEL</td><td>2</td><td> $x \left[ = - \log ( d / f ) + \log 6 \right]$ </td></tr><tr><td colspan="4">abel_level3 (closed; no CoV  ${ \mathrm { n e e d e d } } ) - a b ^ { 2 } x ^ { 2 } = 0$ </td></tr><tr><td>1</td><td>RELABEL</td><td>0</td><td> $a x ^ { 2 }$ </td></tr><tr><td>2</td><td>div a</td><td>0</td><td> $x ^ { 2 }$ </td></tr><tr><td>3</td><td>div x</td><td>0</td><td> $x \ [ = 0 ]$ </td></tr></table>

Table 11: Per-class greedy solution traces from the converged open\_large agent (seed109, full method stack). The four change-of-variables classes share the same 7-step recipe up to constants; the exponentia class is the only one that recruits a nested CoV (step 4) after the inner equation becomes a quadratic. The closed-form abel\_level3 example needs no CoV at all and solves in 3 steps.

## A.9 Closed-equation solution traces

Below are the solution traces for each equation in the fixed closed-equation environment of Section 4.1. Notice the solution trace is not always optimal. For instance, $a x + b = 0$ could be solved with (sub, b), (div, a), but instead the agent selects (sub, ax) and then (mul, −1). “Truediv” is the SymPy notation for divide.

1. Equation: a + x = 0   
Step 1: a + x = 0 | subtract, a   
Solved: x = -a   
2. Equation: ax = 0   
Step 1: a\*x = 0 | div, a   
Solved: x = 0   
3. Equation: ax + b = 0

```haskell
Step 1: ax + b = 0 | subtract, ax
Step 2: -ax = b | multiply, -1
Step 3: ax = -b | div, a
Solved: x = -b/a
4. Equation: a/x + b = 0
Step 1: a/x + b = 0 | subtract, b
Step 2: a/x = -b | truediv, 1/x
Step 3: -b*x = a | truediv, b
Step 4: -x = a/b | multiply, -1
Solved: x = -a/b
5. Equation: c(ax + b) + d = 0
Step 1: c*(ax + b) + d = 0 | expand, None
Step 2: acx + bc + d = 0 | subtract, acx
Step 3: -acx = bc + d | multiply, -1
Step 4: acx = -bc - d | truediv, c
Step 5: ax = (-bc - d)/c | truediv, a
Solved: x = (-bc - d)/(ac)
6. Equation: c + d/(ax + b) = 0
Step 1: c + d/(ax + b) = 0 | subtract, c
Step 2: d/(ax + b) = -c | multiply, (ax + b)
Step 3: d = -c (ax + b) | expand, None
Step 4: d = -c ax - c b | multiply, -1
Step 5: -d = c ax + c b | subtract, c b
Step 6: -d - c b = c ax | truediv, c
Step 7: (-d - c b)/c = ax | truediv, a
Step 8: ((-d - c b)/c)/a = x | truediv, 1/a
Solved: x = (-d - c b)/(c a)
7. Equation: cx + d + e(ax + b) = 0
Step 1: cx + d + e(ax + b) = 0 | expand, None
Step 2: aex + be + cx + d = 0 | collect, x
Step 3: be + d + x*(ae + c) = 0 | subtract, x(ae + c)
Step 4: -x(ae + c) = be + d | truediv, ae + c
Step 5: -x = (be + d)/(ae + c) | multiply, -1
Solved: x = -(be + d)/(a*e + c)
8. Equation: e + (ax + b)/(cx + d) = 0
Step 1: e + (ax + b)/(cx + d) = 0 | truediv, 1/(cx + d)
Step 2: (e + (ax + b)/(cx + d))(cx + d) = 0 | expand, None
Step 3: ax + b + cex + de = 0 | collect, x
Step 4: b + de + x*(a + ce) = 0 | subtract, x(a + ce)
Step 5: -x(a + ce) = b + de | expand, None
Step 6: -ax - cex = b + de | collect, x
Step 7: x*(-a - ce) = b + de | truediv, -a - ce
Solved: x = (b + de)/(-a - c*e)
```

Algorithm 1: TreeMLP message passing and pooling   
Require: node IDs X, edges (src, dst), masks m<sub>node</sub>, m<sub>edge</sub>, steps K   
1: emb ← Proj(Embed(X)) ▷ shape [B, N, H]   
2: h ← emb   
3: for k = 1 to K do   
4: E ← {(i → j) | m<sub>edge</sub>(i → j) = 1}   
5: agg[j] ← P<sub>(i→j)∈E</sub> h[i] ▷ scatter-sum   
6: h ← MLP [emb, agg]   
7: h ← h ⊙ m<sub>node</sub> ▷ mask padded nodes   
8: end for   
9: return pool(h, m<sub>node</sub>) ▷ mean or max

## A.10 TreeMLP Details

Architecture. TreeMLP embeds node IDs, projects to a hidden size, and performs K message-passing stages over the (directed) expression tree. At each stage it sums incoming neighbor states, concatenates the result with the fixed input embedding, and applies a two-layer MLP with ReLU. A masked global pool (mean or max) produces the graph embedding exposed to the policy/value heads.

Listing 1: TreeMLP forward pass (minimal).   
1 def forward (self , obs : dict ) -> torch . Tensor :   
2 node\_ids = self . \_to\_device (obs [" node\_features "], self . device )   
3 edge\_index = self . \_to\_device (obs [" edge\_index "], self . device )   
4 node\_mask = self . \_to\_device (obs [" node\_mask "], self . device )   
5 edge\_mask = self . \_to\_device ( obs [ " edge\_mask " ] , self . device )   
6   
7 node\_idx = self . \_id\_to\_idx ( node\_ids ) if node\_ids . dtype != torch . long else   
node\_ids   
8 B, N = node\_idx . shape   
9 \_ , \_ , E = edge\_index . shape   
10   
11 emb = self . embed ( node\_idx ) # [B ,N , embed\_dim ]   
12 emb = self . embed\_proj (emb ) # [B,N,H]   
13 h = emb . clone ()   
14   
15 src = edge\_index [:, 0, :]. long () # [B , E ]   
16 dst = edge\_index [:, 1, :]. long () # [B,E]   
17 batch\_idx = torch . arange (B, device = self . device ). unsqueeze (1). expand (B, E)   
18   
19 for \_ in range ( self .K):   
20 src\_flat = src . reshape ( -1)   
21 dst\_flat = dst . reshape ( -1)   
22 b\_flat = batch\_idx . reshape ( -1)   
23 m\_flat = edge\_mask . reshape ( -1) . bool ()   
24   
25 src\_flat , dst\_flat , b\_flat = src\_flat [ m\_flat ], dst\_flat [ m\_flat ], b\_flat   
[ m\_flat ]   
26 src\_idx = b\_flat \* N + src\_flat   
27 dst\_idx = b\_flat \* N + dst\_flat   
28   
29 h\_flat = h. reshape (B \* N, self . hidden\_dim )   
30 messages = h\_flat [ src\_idx ] # [M , H ]   
31   
32 agg\_flat = torch . zeros\_like ( h\_flat ) # [ B \*N , H ]   
33 agg\_flat . index\_add\_ (0 , dst\_idx , messages ) # sum incoming   
34 agg = agg\_flat . reshape (B, N, self . hidden\_dim ) # [B,N,H]

![](images/77f31d7ba708da47b7c583ae64f8ec683e686cdfb21bfabb4632e894fbbaf847.jpg)  
Figure 4: TreeMLP outperforms GCN and GraphSAGE on the small datasets. Mean of three random seeds with shading to min/max.

35   
36 h = self . node\_mlp ( torch . cat ([ emb , agg ] , dim = -1) ) # [B ,N , H ]   
37 h = h \* node\_mask . unsqueeze ( -1) . float ()   
38   
39 if self . pooling == " mean " :   
40 denom = node\_mask .sum (dim =1, keepdim = True ). clamp (min =1). float ()   
41 g = h.sum (dim =1) / denom   
42 else :   
43 if self . \_minus\_inf is None or self . \_dtype\_cache != h . dtype :   
44 self . \_minus\_inf = torch . finfo (h. dtype ).min   
45 self . \_dtype\_cache = h . dtype   
46 g = h . masked\_fill (\~ node\_mask . unsqueeze ( -1) , self . \_minus\_inf ) .max( dim =1)   
. values   
47   
48 return self . proj (g)

Implementation notes. We use fixed-size padding with boolean masks for nodes/edges, scatter-add for neighbor aggregation, and pool with mask awareness. All tensors are moved to CUDA when available; no MPS is used.

## A.11 Ablations

We summarize the ablation findings; the corresponding learning curves are in Figures 5 and 4.

• Replacing dense complexity-delta rewards with sparse outcome-only rewards (and disabling the inverse-frequency curriculum) substantially degrades performance. The main drawback of sparse rewards is far longer run times: we had to introduce a 1-second-per-step wall-clock timeout to keep training tractable.

• TreeMLP outperforms a vanilla GCN and GraphSAGE on the small dataset in single-seed runs (Figure 4); the tree-aware aggregation appears helpful here, though this comparison is suggestive, not conclusive.

• Curiosity bonuses (ICM, RND, NGU) do not improve TreeMLP learning on abel\_level3 (Figure 6); the body discusses this negative result in Section 3.2. The bonus runs are single-seed and so are suggestive, not conclusive.

## A.12 Deferred main-text figures and tables

This section collects figures and tables deferred from the main text. Each is referenced inline from the appropriate main-text passage.

![](images/21387bea57dd948dc121f1b366ed5f209fe021b053291952802005a1e19f0d65.jpg)  
Figure 5: Inductive-bias ablations on the small closed-equation dataset. Each curve removes one component (dense complexity-delta reward + inverse-frequency curriculum, relabel-constants, or the success-replay bufer) from the full ppo-tree-rc-buf stack; y-axis is greedy test accuracy over training steps (mean of 3 seeds, min/max shading). Conclusion: removing the dense reward/curriculum or the replay bufer substantially degrades learning, so each bias is load-bearing on this dataset.

Curiosity bonuses do not improve TreeMLP learning on abel\_level3  
![](images/2db8b9647371cdc081adb31ee8e002ea84bdb61d98556621ee5fd88a68dd5355.jpg)  
Figure 6: Curiosity bonuses (ICM, NGU, RND) do not improve TreeMLP learning on abel\_level3 (singleseed bonus runs vs. an n=4-seed no-curiosity baseline; y-axis test metrics over training steps). No bonus matches the baseline; rnd fails to learn. These single-seed results are suggestive, not conclusive.

Cumulative open-equation ablation (referenced from §4.2.3).

## A.13 CoV trigger ablation: the stalled seeds

The trigger ablation of Section 4.2.4 is run on all eight open\_small checkpoints of Figure 2, but only the five converged seeds are pooled in Table 5. The three stalled seeds (147000, 148000, 627000) are reported here and never pooled, for the reason the main text gives: a stalled policy solves ≈0 regardless of who owns the trigger, so these seeds carry no information about trigger quality and would only dilute the comparison.

![](images/4a5dced0d3bd78866c4e733e1b943d227e5664c37081e5334c352a9afe877a93.jpg)  
Figure 7: Expression trees for $a x + b$ and $a x ^ { 2 } + b x + c .$ The agent’s state vector is a preorder traversal of these trees, padded to length $L = 5 0$

Expression trees (referenced from Sec. 4 state encoding).

Curiosity-bonus negative result (referenced from §4.1). See Figure 6 and the ablation bullet above;   
the negative result and its plausible explanation are discussed in Section 3.2.

TreeMLP embeddings cluster by family (referenced from §4.1).

Per-class accuracy on open\_small (referenced from Section 4.2.3).

Per-class accuracy on open\_large across escape seeds (referenced from Section 4.2.3).

Bimodality per-seed dots (referenced from Section 4.2.5).

Full 9-row intervention sweep (referenced from Section 4.2.5).

Three competing hypotheses probed (referenced from Section 4.2.5). We find no evidence for the three most natural alternative explanations for the bimodal failure; these probes are necessary-not-suficient:

• Capacity loss / dormant neurons Sokar et al. (2023): at the standard ReDo activation threshold (0.1× per-layer max), both stuck and escape seeds have 0% dormant neurons; no evidence of a capacity-loss phenomenon.

• Data-distribution skew: an alternative training distribution skewed toward abel\_level3 (the “abelboost” set) reaches beam = 0.93 on its matched test split but only 0.04 on the standard test set – the policy overfits to whatever distribution it sees; no evidence that a training-data fix would resolve the bimodality.

• Encoder representation collapse: t-SNE projections of the TreeMLP encoder show clean per-class clusters on stalled seeds too (Appendix A.6); no evidence of representation collapse.

The evidence is most consistent with a policy-attractor phenomenon rather than a capacity, data, or representation failure, though these probes do not exhaustively rule out other explanations.

![](images/2c72e25ec7a0379eae9bfdf044c3594424fc683a0d5a66a3dd3d53b6220583d6.jpg)  
Figure 8: t-SNE projection of TreeMLP pooled embeddings for 300 small-dataset equations, colored by structural type. Clusters correspond to equation family rather than to specific coeficient assignments.

## A.14 Validation-based checkpoint selection protocol

(Referenced from Section 4.2.3.) For every reported standalone-eval beam value, we use a $3 0 / 7 0$ validation/test split of the test set with seed 42, evaluate the saved checkpoints on the validation split, select the checkpoint with the highest val beam, and report its test-split beam.

open\_large bimodality: UCB-LP reverses the failure mode (6/8 escape, mean beam over escape seeds = 0.83); other interventions only shift it  
![](images/b53a41c7d8b74253e18494f38dd88fb8f0ec275ce045de90c837f41b4754753d.jpg)

![](images/b83a5382d89b6642685cbcbc0bcd5ce86d15fcf91c91b4483cd0b498802e5fb3.jpg)  
Figure 9: Per-seed beam test accuracy and cohort escape rates from the initial 3-seed screen on open\_large. Each dot is a single seed at its validation-best checkpoint; bars show fraction of seeds with beam $\geq 0 . 2$ (escape threshold). UCB-LP was the only cohort with $3 / 3$ escape in this screen, which motivated expanding it to $n = 8$ (Table $^ { 7 , }$ where it reaches $6 / 8 )$ . Numbers in Table $7$ supersede this screen for the three headline cohorts.

<table><tr><td rowspan="2">configuration</td><td colspan="3"> $t e s t _ { \mathrm { b e a m } }$  per seed</td><td rowspan="2"> $\mathrm { m e a n } \pm \mathrm { s d }$ </td></tr><tr><td>507000</td><td>508000</td><td>509000</td></tr><tr><td>C1 baseline (PPO-TREE-RC-BUF-COV)</td><td>0.250</td><td>0.406</td><td>0.406</td><td> $0 . 3 5 4 \pm 0 . 0 9 0$ </td></tr><tr><td> $\mathrm { C 2 \ + a c t i o n – d i v e r s i t y \ ( a n t i – l o o p ) \ p e n a l t y }$ </td><td>0.109</td><td>0.109</td><td>0.109</td><td> $0 . 1 0 9 \pm 0 . 0 0 0$ </td></tr><tr><td> $\mathrm { C 4 } \ + \mathrm { c u b e - r o o t } \ \& \ \mathrm { n e s t e d } \ \mathrm { C o V } \ ( \mathrm { f i a t \ b u f f e r } )$ </td><td>0.484</td><td>0.766</td><td>0.859</td><td> $\mathbf { 0 . 7 0 3 \pm 0 . 1 9 5 }$ </td></tr><tr><td>C5 + freshness-managed buffer (full stack)</td><td>0.844</td><td>0.094</td><td>0.531</td><td> $0 . 4 9 0 \pm 0 . 3 7 7$ </td></tr><tr><td colspan="5">Paired within-seed increments (per-seed values; mean ± sd over the three shared seeds):</td></tr><tr><td> $\mathrm { C 1 } \to \mathrm { C 2 }$  action-diversity penalty</td><td colspan="3"> $- 0 . 1 4 1 , \ - 0 . 2 9 7 , \ - 0 . 2 9 7$ </td><td> $\mathbf { - 0 . 2 4 5 \pm 0 . 0 9 0 }$ </td></tr><tr><td> $\mathrm { C 2 } \to \mathrm { C 4 }$  cube-root &amp; nested CoV</td><td colspan="3"> $+ 0 . 3 7 5 , \ + 0 . 6 5 6 , \ + 0 . 7 5 0$ </td><td> $\mathbf { + 0 . 5 9 4 \pm 0 . 1 9 5 }$ </td></tr><tr><td> $\mathrm { C 4 } \to \mathrm { C 5 }$  freshness-managed buffer</td><td colspan="3"> $+ 0 . 3 5 9 , \ - 0 . 6 7 2 , \ - 0 . 3 2 8$ </td><td> $- 0 . 2 1 4 \pm 0 . 5 2 5$ </td></tr><tr><td colspan="5">End-to-end (baseline → each endpoint; the contrast the ladder&#x27;s adjacent rungs do not show):</td></tr><tr><td>C1 → C4 (no freshness buffer)</td><td colspan="3"> $+ 0 . 2 3 4 , \ + 0 . 3 5 9 , \ + 0 . 4 5 3$ </td><td> $\mathbf { + 0 . 3 4 9 \pm 0 . 1 1 0 }$ </td></tr><tr><td>C1 → C5 (full stack)</td><td colspan="3"> $+ 0 . 5 9 4 , \ - 0 . 3 1 2 , \ + 0 . 1 2 5$ </td><td> $+ 0 . 1 3 5 \pm 0 . 4 5 3$ </td></tr><tr><td colspan="5">Removing the penalty (α =0, same seeds and budget; new in this revision): 0.859</td></tr><tr><td> $\mathrm { C 6 ~ } = \mathrm { C 4 }$  at  $\alpha { = } 0$   $\mathrm { C 7 ^ { \ddagger } } \ = \mathrm { C 5 \ a t \ } \alpha { = } 0 \ ( \mathrm { f u l l \ s t a c k } )$ </td><td colspan="3">0.859 0.906 0.875</td><td> $\mathbf { 0 . 8 7 5 \pm 0 . 0 2 7 }$   $\mathbf { 0 . 8 9 1 \pm 0 . 0 2 7 }$ </td></tr><tr><td></td><td colspan="3">0.875</td><td></td></tr><tr><td>C4 → C6 (drop α)</td><td colspan="3"> $+ 0 . 3 7 5 , \ + 0 . 1 4 1 , \ + 0 . 0 0 0$ </td><td> $+ 0 . 1 7 2 \pm 0 . 1 8 9$ </td></tr><tr><td>C5 → C7 (drop α; full stack)</td><td colspan="3"> $+ 0 . 0 3 1 , \ + 0 . 7 8 1 , \ + 0 . 3 9 1$ </td><td> $\mathbf { + 0 . 4 0 1 \ ( 3 / 3 ) }$ </td></tr><tr><td>C1 → C6 (clean end-to-end, no freshness)</td><td colspan="3"> $+ 0 . 6 0 9 , \ + 0 . 5 0 0 , \ + 0 . 4 5 3$ </td><td> $\mathbf { + 0 . 5 2 1 \pm 0 . 0 8 0 }$ </td></tr><tr><td> $\mathrm { C 1 } \to \mathrm { C 7 }$  (clean end-to-end, full stack)</td><td colspan="3"> $+ 0 . 6 2 5 , \ + 0 . 4 6 9 , \ + 0 . 5 1 6$ </td><td> $\mathbf { + 0 . 5 3 7 \pm 0 . 0 8 0 }$ </td></tr></table>

Table 12: Seed-matched open-equation ablation on open\_small $( t e s t _ { \mathrm { b e a m } } ,$ beam width 5, λ = 0, the 64-equation held-out slice; every cell trained 3M steps). New in this revision at a reviewer’s request, replacing v5’s single-seed cumulative table, whose rows came from diferent seeds, reported a training peak (0.341) rather than a final value (0.275) for the action-diversity row, and difered in max\_steps — so none of its increments could be read causally. Here all configurations share the same three seeds, the same budget and the same decoder, with no early stopping and no checkpoint selection, so no configuration can win by being frozen at a luckier step. $( \mathrm { v 5 ^ { \circ } s ~ } ^ { 6 6 } +$ value-guided beam” row was a decode-time change, not a training ingredient; we hold the decoder fixed instead of making it a rung.) Findings. The change-of-variables machinery carries the result (+0.594, positive on every seed; it bundles the cube-root primitive with nested CoV, which this design cannot separate). The auxiliary RL engineering does not: the action-diversity penalty is harmful (−0.245, negative on every seed, against v5’s reported +0.03), and the freshness-managed bufer is unresolvable (its increment changes sign across seeds). End-to-end, the lift over baseline is carried by the simpler C4 (+0.349, positive on every seed); the penalised full stack’s +0.135 flips sign and is not resolvable at $n = 3$ . What this licenses. Only large efects. Re-running an identical (configuration, seed) cell reproduces $t e s t _ { \mathrm { b e a m } }$ to within 0.079 on average (0.253 worst case). That estimate is itself limited: it comes from $n = 6$ replicate cells on C1/C2 — the two lowest-variance configurations — at 1M steps on the full 91-equation file, so it is extrapolated across budget and denominator to this table and is a floor, not a bound; a diference of two cells carries ${ \sim } \sqrt { 2 }$ that noise. We therefore claim only increments whose sign is stable across all three seeds, never point estimates; $3 / 3$ sign agreement carries a floor two-sided p of 0.25, so these are corroborated directions, not powered tests. This is an ablation of credit among ingredients and is not a re-measurement of the headline, which rests on a larger 5-seed cohort at a validation-selected checkpoint (Table 3). One cell (C5/seed509000) is a re-run: its original trial died in a SymPy NaN-comparison fault after 96 s and was retrained from scratch under the identical training configuration at the same seed (509000), though alone on the machine rather than alongside its cohort. <sup>‡</sup>C7/seed507000’s original trial died at ∼2.8M steps to the SymPy/mpmath overflow on large rational coeficients (also seen for two open\_large seeds; Section 6) and was retrained from scratch at the same seed to complete the row. At $\alpha = 0$ , both C6 and C7 exceed the reported headline within this fixed-step control, variance collapses, and none of the six $\mathrm { C 6 / C 7 }$ cells stalls (including C1, 0/9 α = 0 cells stall). We do not promote either control mean to the headline because the downstream open\_small analyses and validation-selected headline use a diferent, α=0.1 cohort; nor do we call 0.79 a lower bound (Section 6).

Table 13: Trigger ablation on the three stalled open\_small seeds (greedy, $n = 9 1 )$ . Every trigger regime — including never — lands within one equation $( 1 / 9 1 = 0 . 0 1 1 )$ of every other on each seed. The trigger is irrelevant when the policy has not learned the closed-action steps needed to finish an episode after a CoV, which is precisely why these seeds are excluded from Table 5 rather than averaged into it.
<table><tr><td>seed</td><td>never</td><td> $\mathrm { \ a l w a y s @ 0 }$ </td><td>rule</td><td>learned</td><td>rule + expand</td></tr><tr><td>147000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>148000</td><td>0.011</td><td>0.011</td><td>0.011</td><td>0.011</td><td>0.011</td></tr><tr><td>627000</td><td>0.110</td><td>0.110</td><td>0.110</td><td>0.099</td><td>0.110</td></tr></table>

Table 14: Per-class test accuracy on open\_small (value-guided beam). Baseline is the single-seed ppotree-rc-buf-cov reference $( n = 1 )$ ; full-stack is the 4-seed mean of seeds completing the full budget (the 5th early-stopped seed is excluded here, whereas the headline 0.79 in Section 4.2.3 is the 5-seed mean that includes it – the two are therefore not directly comparable). The n-weighted overall of these 4-seed per-class values is $\approx 0 . 8 5$ on the full 91 equations; the headline 0.79 is on the 64-equation test slice.
<table><tr><td>class (n)</td><td>baseline</td><td>full stack</td></tr><tr><td>quadratic (15)</td><td>0.00</td><td>0.97</td></tr><tr><td>cubic (15)</td><td>0.00</td><td>1.00</td></tr><tr><td>quartic (15)</td><td>0.00</td><td>1.00</td></tr><tr><td>exponential (15)</td><td>0.00</td><td>0.88</td></tr><tr><td>abel_level1 (1)</td><td>1.00</td><td>1.00</td></tr><tr><td>abel_level2 (10)</td><td>0.50</td><td>0.75</td></tr><tr><td>abel_level3 (20)</td><td>0.25</td><td>0.53</td></tr></table>

Table 15: Per-class open\_large test accuracy (value-guided beam, 200 eqns/class) across the 3 escape seeds (overall $\in [ 0 . 7 9 , 0 . 9 1 ] )$ ; the 6 stall seeds are ≈ 0 on every class. The four CoV classes saturate while abel\_level3 remains weak and high-variance – the open\_small pattern, sharpened at scale.
<table><tr><td>class  $( n = 2 0 0 )$ </td><td>escape-seed mean [min, max]</td></tr><tr><td>quadratic</td><td>0.98 [0.98, 0.99]</td></tr><tr><td>cubic</td><td>0.99 [0.98, 1.00]</td></tr><tr><td>quartic</td><td>0.99 [0.98, 1.00]</td></tr><tr><td>exponential</td><td>0.98 [0.96, 1.00]</td></tr><tr><td>abel_level3</td><td>0.39 [0.07, 0.55]</td></tr></table>

Table 16: Initial 3-seed screen of the full intervention sweep on open\_large (mean validation-best beam over 3 seeds; escape = beam ≥ 0.2). All cohorts were first screened at 3 seeds to select interventions; the three headline cohorts were subsequently expanded to $n = 5$ (baseline), 3 (vanilla LP), and 8 (UCB-LP) seeds, with final numbers in Table 7, which supersede the corresponding rows here. The non-headline rows (anti-loop, dataset rebalance, LP+anti-loop, $\mathrm { L P + + } )$ were screened only and not expanded.
<table><tr><td>Cohort</td><td>Mean beam</td><td>Escape rate</td></tr><tr><td>Baseline (full stack)*</td><td>0.34</td><td>2/3</td></tr><tr><td>Anti-loop (α =0.3)</td><td>0.05</td><td>0/3</td></tr><tr><td>Anti-loop (α =1.0)</td><td>0.11</td><td> $1 / 3 ^ { \dagger }$ </td></tr><tr><td>Dataset rebalance (abelboost)</td><td>0.04</td><td>0/3</td></tr><tr><td>LP curriculum (vanilla)* Matiisen et al. (2019)</td><td>0.40</td><td>2/3</td></tr><tr><td>LP + anti-loop (α = 0.3)</td><td>0.07</td><td>0/3</td></tr><tr><td> $\mathrm { L P } { + } { + }$  (optim. init + EMA + boot)</td><td>0.10</td><td> $0 / 3$ </td></tr><tr><td>UCB-LP curriculum</td><td>0.81</td><td>3/3</td></tr><tr><td>Oracle (SymPy)</td><td>1.00</td><td></td></tr></table>

<sup>∗</sup> headline cohort; expanded beyond this 3-seed screen, see Table 7 for the superseding numbers. <sup>†</sup> one of three seeds crashed mid-run due to a SymPy/mpmath OverflowError on a large rational; we report rate over surviving seeds.