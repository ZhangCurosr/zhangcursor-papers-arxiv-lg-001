# Dynamics of Hidden Learning Structure

Qinyou Wang

## Abstract

A learner’s present predictions do not fully determine how it will learn next. We study the dynamics of hidden learning structure through two related operations. Fiber Regulation Theory (FRT) describes protected repair of existing function and the learning responses left available after repair. Fiber Acquisition Theory (FAT) describes the formation and renewal of trainable structure, distinguishing neutral installation, access to a learning signal, and additional functional directions. The two operations also constrain one another. A repair obligation can force a change in a learning channel, while acquired features can make new protected repairs feasible or reduce the minimum coeficient cost of existing ones. We characterize joint admissibility and response reachability, and give a finite supervised-learning construction in which the two operations together produce a favorable response absent from either operation alone. We implement protected head repair and fixed-capacity LoRA renewal in Qwen2.5-32B with AdamW, under replay and output distillation. Across two training trajectories difering in seed and within-task example order, the coordinated policy achieves higher teacher-forced Exact than either separate policy in all four comparisons. It also has the lowest macro NLL under both learning rules on the second trajectory. The implementations realize structure-conditioned self-regulation within specified information and action constraints: accessible state readouts and arrived evidence guide repairs and structural renewals, and ordinary learning continues from the modified state.

## 1 Introduction

A learner’s current function does not completely specify how it will learn next. Parameter realization, optimizer memory, trainable representations, and retained experience can afect its response to further training. Observationrelative fibers describe agreement in selected current behavior while controlled future-learning responses distinguish states within that agreement class [1]. The central question of this paper is how such structure can be acted upon: how existing function can be repaired, how new learning directions can be formed, and how the two operations reshape each other’s possibilities.

We formulate two general theories. Fiber Regulation Theory (FRT) characterizes state changes that achieve a functional repair while preserving a declared observation. Fiber Acquisition Theory (FAT) characterizes the formation, installation, and use of trainable structure from available evidence, including renewal at fixed capacity. Their shared subject is the evolving learning state; their observations, actions, and immediate objectives may difer.

FRT starts from the finite-frame head repair and intrinsic–extrinsic interaction studied by Wang [2]. That construction preserves current logits while repairing selected historical margins; its later efect depends on the continuation. Building on that finite repair family, we characterize the learning responses available after the repair obligation has been met. The new response geometry identifies directions along which later learning can change without undoing the prescribed functional repair.

FAT addresses a diferent distinction. Installing a new feature with zero output coeficient can leave the current function unchanged while changing the gradient and output response of the next update. We give separate criteria for neutral installation, access to a learning signal, and additional functional directions. A renewal that removes an existing component requires its own finite acceptance conditions. This distinction makes acquisition a property of the subsequent learning dynamics, not merely of parameter count.

When regulation and acquisition act together, their admissible actions and responses can change. Separate acceptance does not generally imply joint acceptance. We characterize a repair obligation that forces a change in a learning channel, derive the protected repair directions available after compensating the current efects of new features, and describe joint response reachability under a norm budget. A finite cross-entropy example establishes strict positive complementarity through an actual supervised gradient update.

At each decision, the policy uses accessible state readouts and arrived evidence to select admissible repairs or renewals, or to make no extra write. Ordinary learning then proceeds from the resulting state.

Protected updates and preservation–editing constraints have precedents in orthogonal gradient descent, constrained model editing, and null-space editing [3–5]. Function-preserving expansion, selective plasticity renewal, low-rank budget allocation, and gradient-aligned initialization also have established formulations [6–9]. Recent low-rank analyses address composite-update interference and parameterization-dependent optimization [10, 11]. The contribution here is the mathematical relation between the two dynamical operations: residual learning freedom after repair, compensation-adjusted repair directions supplied by acquisition, and the constraints governing their joint execution and response.

![](images/28f76bba7a04e4521c2780682bb335921f91ed5a1af5aee80bd926fa83e60c69.jpg)  
Figure 1. Regulation and acquisition within one evolving learner. The complete state $s _ { t }$ is distinct from the available information $i _ { t } .$ Regulation repairs existing function; acquisition forms or renews trainable structure. Using $\textstyle { \mathcal { T } } _ { t } ,$ , a policy selects an admissible pair $\xi _ { t } = ( f _ { t } , a _ { t } ) \in \mathcal { H } ( s _ { t } )$ through available certificates or a valid robust rule. Identity components allow either intervention, both, or no extra write. The state-transition marginal $\bar { K } _ { \xi _ { t } }$ uses the declared execution order. Ordinary learning consumes the resulting state and supplies readouts to later decisions. The FRT–FAT arrow denotes possible cross-efects in admissibility, response and repair cost.

After the preliminaries in Section 2, Sections 3–5 develop the three theoretical components. Section 6 derives their transformer/LLM specializations. Section 7 examines Qwen2.5-32B under replay and output distillation across two training trajectories. The coordinated policy achieves higher Exact than either isolated mechanism in all four comparisons. These comparisons show that the specified repairs and renewals can be executed in a trained language model and can afect its measured outcomes after continued learning.

## 2 Mathematical preliminaries

We introduce only the objects shared by regulation and acquisition. Let $s \in S$ be a complete learning state. It contains the parameters and all retained quantities needed for subsequent execution: optimizer state, structural configuration, relevant memory or teacher state, stream position, random state, and any policy or resource state that afects future operations. Diferent configurations may have diferent parameter dimensions; formally, S may be a disjoint union $\operatorname { I I } _ { \rho } S _ { \rho }$ . An observation is not assumed to retain this whole state.

Definition 2.1 (Observation-relative fiber). For a declared observation map $B : S  y$ , the fiber through � is

$$
{ \mathcal { F } } _ { B } ( s ) = \{ \tilde { s } : B ( \tilde { s } ) = B ( s ) \} .\tag{2.1}
$$

For a stochastic observation, equality refers to its declared conditional law, not to two coincident sample values.   
On a fixed diferentiable chart, ker $D B _ { s }$ is the infinitesimally invisible subspace.

The set in (2.1) needs no manifold assumption. An infinitesimally invisible direction in ker $D B _ { s }$ need not yield an exactly preserving finite displacement. Conversely, a function-neutral structural extension can be compared through � even when it changes configuration. The term hidden learning structure refers here to distinctions erased by the selected observation but relevant to permitted future learning responses; it does not denote a separately autonomous subsystem [1, 2].

For probabilistic statements, the state and outcome spaces are standard Borel spaces and all kernels and policies are measurable. At an intervention time, $\textstyle { \mathcal { T } } _ { t }$ denotes the sigma-field generated by information legally available before the decision, including any authorized randomization already drawn. Intrinsic sensing means a specified readout of this information and state interface, not free access to the complete state or to future outcomes. A lawful operation � has a declared domain and a normalized joint outcome–successor kernel $K _ { a } ( s ; d y , d s ^ { \prime } )$ on that domain. Its state-transition marginal is $\bar { K } _ { a } ( s , d s ^ { \prime } ) = K _ { a } ( s ; y _ { a } , d s ^ { \prime } )$ . Any returned outcome or cost needed for later decisions is retained in $s ^ { \prime }$ . Deterministic writes are partial maps. An unavailable operation has no probability row; an allowed rejection, deferment, or termination is an actual outcome. A no-intervention branch skips the additional write, not the ordinary learning that follows. Costs and retained diagnostic efects belong to the comparison when relevant.

A continuation � fixes the subsequent learning policy, horizon convention, output semantics, and exogenous randomization. Write $P _ { H } ^ { \pi } ( s , \cdot )$ for the resulting endpoint or path law. The response fingerprint is the family of such laws over a declared probe class, not an automatically available online vector. For an integrable common readout �, define

$$
V _ { H } ^ { \pi } ( s , a ; J ) = \int J d \big [ ( \bar { K } _ { a } P _ { H } ^ { \pi } ) ( s ) - ( \bar { K } _ { 0 } P _ { H } ^ { \pi } ) ( s ) \big ] .\tag{2.2}
$$

Here $\bar { K } _ { 0 }$ is the state-transition kernel of the stated reference branch at the same phase cut. The notation includes any prescribed installation and consumption before the continuation. Comparisons across structural dimensions use a common readout or output law, not subtraction of diferently typed parameter vectors.

For matched deterministic branches with the identity as reference, (2.2) reduces to $J ( \Phi _ { H } ^ { \pi , \omega } ( a ( s ) ) ) { - } J ( \Phi _ { H } ^ { \pi , \omega } ( s ) )$ The same � means the same declared external realization, not merely equal seed integers with incompatible random-number consumption. A shared feedback policy may issue diferent later actions after its states diverge.

Throughout the paper, local admissibility, continuation-conditioned value, and complete-policy performance are diferent claims. A current certificate establishes the first. Equation (2.2) defines the second. Comparing two live policies from a common initial law addresses the third. An Exact score is a discrete readout; smooth calculations below apply to stated losses or response maps, never to an unqualified derivative of Exact.

Linear-algebra conventions. All main finite-dimensional statements use Euclidean inner products, Frobenius matrix norms, and the Moore–Penrose inverse $M ^ { \dagger }$ . The operator norm is $\| M \| _ { \mathrm { o p } }$ . Projectors are orthogonal. Any restriction to an implemented actuator must be included in the action coordinates before applying a theorem. In particular, an ideal dense action space is not automatically an implementation’s available action set.

## 3 Fiber Regulation Theory

## 3.1 Regulation as a constrained change of learning state

A regulation problem specifies what is protected, what needs repair, and which actions can be executed. Its target is a functional obligation on an existing learner; it need not preserve every parameter or every future response.

Definition 3.1 (Regulation specification). At a fixed decision state $s ,$ let $B _ { s }$ be the declared protection observation, $R _ { s } : S  y _ { R }$ the repair-target readout, $\mathcal { O } _ { s } \subseteq \mathcal { Y } _ { R }$ its acceptable values, $\mathcal { A } _ { F } ( s )$ an executable action interface, and $c _ { F } ( s , f )$ a cost or size constraint. A successful regulation action satisfies

$$
f \in \mathcal { A } _ { F } ( s ) , \qquad B _ { s } ( f ( s ) ) = B _ { s } ( s ) , \qquad R _ { s } ( f ( s ) ) \in O _ { s } , \qquad c _ { F } ( s , f ) \leq b _ { F } .\tag{3.1}
$$

The proposal and admission decision are $\textstyle { \mathcal { I } } _ { t }$ -measurable. A registered no-write branch remains available when no proposal is accepted, without being counted as a successful repair.

Finite-precision implementations replace exact constraints by declared tolerances checked on the stored action. The subscript � fixes the consultation’s frames and meanings while its alternatives are compared. It does not permit changing the observation after seeing which action succeeds. FRT is the study of this regulated evolution and its learning consequences, not a requirement to intervene at every update.

## 3.2 Exact finite repair geometry

An important class is afine in the action even when the underlying predictor is nonlinear. Let $\boldsymbol { X } \in \mathbb { R } ^ { d \times q }$ be a permitted action coordinate, $C \in \mathbb { R } ^ { m \times d }$ the protected response matrix, $H \in \mathbb { R } ^ { p \times d }$ the obligation matrix, and

$D \in \mathbb { R } ^ { p \times q }$ the desired change. The finite repair equations are

$$
C X = 0 , \qquad H X = D .\tag{3.2}
$$

This notation includes the independently writable head slice of Wang [2], but does not require a language model. Theorem 3.2 (Finite protected repair and remaining freedom). Set $P _ { C } = I - C ^ { \dagger } C a n d N = H P _ { C }$ . Equations (3.2) are feasible if and only if

$$
N N ^ { \dagger } D = D .\tag{3.3}
$$

Whenfeasible, the unique minimum-Frobenius-norm solution is $X _ { * } = N ^ { \dagger } D$ . Every solution is

$$
X = X _ { * } + Z , \qquad C Z = H Z = 0 , \qquad \| X \| _ { F } ^ { 2 } = \| X _ { * } \| _ { F } ^ { 2 } + \| Z \| _ { F } ^ { 2 } .\tag{3.4}
$$

Consequently a norm budget � permits a repair exactly when (3.3) holds and $\| X _ { * } \| _ { F } \leq \tau$

Proof. The first constraint means $X = P _ { C } X$ . Hence $H X = N X$ , whose solvability is (3.3). The minimum-norm solution of $N X = D$ is $N ^ { \dagger } D$ . Since Ran $N ^ { \top } \subseteq \operatorname { R a n } P _ { C }$ , this solution also satisfies $C X _ { * } = 0$ . The diference of any two solutions lies in the common kernel of � and �. Every column of $X _ { * }$ lies in Ran $N ^ { \top }$ , orthogonal to the columns of such a diference. This proves the norm identity and uniqueness. □

The complete afine solution family and norm budget are inherited from Wang [2, Corollary 3.2]; they are recalled to define the residual space used in Theorem 3.3. It does not assert that the physical actuator can execute every � in (3.4). When it can, the same protected repair can have multiple learning realizations. When it cannot, its narrower action interface must be used.

## 3.3 Learning responses remaining after the same repair

The repair equations specify present functional efects. They do not generally fix the learning response. To make the distinction quantitative, work in a finite-dimensional Hilbert action space U. Let $z \subseteq { \mathcal { U } }$ be the linear space of variations that preserve both the protection and the completed repair, and let $\Pi _ { Z }$ be its projector. Let $L : \mathcal { U } \to \mathcal { V }$ describe a declared linear learning-response channel at the fixed parent. It can be an exact finite channel in an afine slice or a first-order channel on a qualified diferentiable branch; these are diferent interpretations.

Theorem 3.3 (Repair-compatible response geometry). Suppose the residual norm budget in (3.4) is $\rho \geq 0 .$ . The additional reachable responses are

$$
\mathcal { E } _ { \mathcal { Z } } ( \rho ) = \{ L Z : Z \in \mathcal { Z } , \| Z \| \leq \rho \} = \{ L \Pi _ { \mathcal { Z } } u : \| u \| \leq \rho \} .\tag{3.5}
$$

Their squared directional support is determined by

$$
G _ { \mathcal { Z } } = L \Pi _ { \mathcal { Z } } L ^ { * } , \qquad \operatorname* { s u p } _ { \nu \in \mathcal { E } _ { \mathcal { Z } } ( \rho ) } \langle w , \nu \rangle = \rho \sqrt { \langle w , G _ { \mathcal { Z } } w \rangle } .\tag{3.6}
$$

For $\rho > 0 ,$ , the channel is identically zero exactly when $L \Pi _ { Z } = 0 .$ . If added protection reduces ${ \mathcal { Z } } t o { \mathcal { Z } } ^ { \prime } \subseteq { \mathcal { Z } }$ while � isfixed, then

$$
0 \preceq G _ { \mathcal { Z } ^ { \prime } } \preceq G _ { \mathcal { Z } } .\tag{3.7}
$$

Proof. Projection does not increase norm and every $Z \in { \mathcal { Z } }$ is its own projection, giving the set identity. The support maximization is $\begin{array} { r } { \operatorname* { s u p } _ { \| u \| \leq \rho } \left. \Pi _ { \mathcal { Z } } L ^ { * } w , u \right. = \rho \left\| \Pi _ { \mathcal { Z } } L ^ { * } w \right\| } \end{array}$ . Squaring the norm gives (3.6). The zero-channel statement follows by scaling any nonzero image direction into the ball. Nested orthogonal projections satisfy $\Pi _ { Z ^ { \prime } } \preceq \Pi _ { Z }$ ; congruence by � proves (3.7). □

A large observation kernel can therefore have zero image in the chosen learning-response channel. Conversely, the same repair may leave a nonzero family of such responses. The monotonicity statement is a fixed-parent comparison: after a real repair changes the parent and hence �, it does not order the new and old Gramians automatically. This prevents geometric freedom from being mistaken for a scalar measure of overall learning quality.

## 3.4 Output protection need not protect the backward learning channel

Let a diferentiable representation $h _ { \theta } ( x ) \in \mathbb { R } ^ { d }$ be followed by an independently writable head $W \in \mathbb { R } ^ { q \times d }$ with logits $z = W h _ { \theta } ( x )$ . At a fixed example, write $J _ { \theta } = D _ { \theta } h _ { \theta } ( x )$ and $r = \nabla _ { z } \ell ( z , y )$ . A head action is $W \mapsto W + X ^ { \top }$ . Its generation is held fixed when computing the gradient consumed by subsequent learning; this is not diferentiation through a meta-generator.

Proposition 3.4 (Protected-output backward response). $H X ^ { \top } h _ { \theta } ( x ) = 0$ , then the logits, loss, and � at that example are unchanged, but

$$
\Delta _ { F } \nabla _ { \theta } \ell = J _ { \theta } ^ { \top } X r .\tag{3.8}
$$

For a fixed collection of examples and fixed sample weights, the corresponding expression is the weighted sum of (3.8), provided $X ^ { \top } h _ { \theta } ( x _ { i } ) = 0 .$ for every contributing example �. The identity is finite and exact in $X$

Proof. Before the write, the gradient is $J _ { \theta } ^ { \top } W ^ { \top } r ;$ after it, it is $J _ { \theta } ^ { \top } ( W ^ { \top } + X ) r$ . The equality of logits makes the same � valid in both expressions. Subtraction gives the result. □

This is the protected-backpropagation identity in Appendix A.4 of Wang [2]: present functional neutrality and future learning neutrality are diferent constraints. It supplies a concrete � for Theorem 3.3. Its exactness does not bypass numerical execution checks, nor does it imply that an arbitrary optimizer converts the changed gradient into a beneficial parameter step.

## 3.5 Regulation and external learning

For one fixed regulation proposal $f ,$ , two continuations $\pi _ { 0 } , \pi _ { 1 }$ , and common readout �, write $Y _ { i j } = \mathbb { E } [ J _ { i j } ]$ for the expected readout with � absent/present (�) and continuation $\pi _ { j }$ ; in deterministic matched branches, the expectation reduces to the recorded readout. Then

$$
I _ { F , \pi } = Y _ { 1 1 } - Y _ { 0 1 } - Y _ { 1 0 } + Y _ { 0 0 } = V _ { H } ^ { \pi _ { 1 } } ( s , f ; J ) - V _ { H } ^ { \pi _ { 0 } } ( s , f ; J ) .\tag{3.9}
$$

This is the intrinsic–extrinsic coupling identified by the matched design of Wang [2]. It asks whether continuation changes the marginal value of the same internal action. The equivalent reading, that the action changes the continuation’s incremental value, is the same contrast, not a reverse-time intervention. A complete regulation policy can subsequently use a state signal to choose later experience allocation, giving an additional, genuinely temporal feedback loop.

Thus FRT contains both a geometry of admissible repair and a dynamics of how those repairs are consumed by continued learning. The former determines what the intervention can guarantee immediately; the latter determines its continuation-specific role.

## 4 Fiber Acquisition Theory

## 4.1 Formation, installation, and consumption

Regulation starts from a functional obligation. Acquisition starts from available experience and the possibility of changing the learner’s trainable realization. The two problems share a state and a response language, but not necessarily the same observation, admissible action family, or target.

Definition 4.1 (Acquisition operation). Let $i _ { t }$ be the permitted information record generating $\textstyle { \mathcal { T } } _ { t }$ , and let � be a fresh authorized random variable, independent of unobserved future exogenous information conditional on $\textstyle { \mathcal { T } } _ { t }$ . An acquisition specification first declares a family of trainable components and an installation interface that creates, replaces, or changes their trainable realization. The index � refers to that interface. Identity and HOLD are control branches; an ordinary consumer update is not classified as acquisition merely by relabeling it as installation. An acquisition operation consists of a formation rule $\nu = \Gamma ( i _ { t } , u )$ measurable with respect to ${ \mathcal { T } } _ { t } \vee \sigma ( u )$ , an admission predicate using that same information, a lawful installation $A _ { \nu }$ , and a specified learning rule (the consumer) $U _ { c }$ State-dependent formation reads only the components included in $i _ { t } ; c$ specifies the update mechanism. The resulting state change is assessed through

$$
s \longmapsto A _ { \nu } ( s ) \longmapsto U _ { c } ( A _ { \nu } ( s ) ) ,\tag{4.1}
$$

including its persistent structure, optimizer state, returned outcomes, and costs. A rejected installation follows its registered no-installation branch. Continued ordinary learning is not disabled.

The generator may use a current structural readout, already encountered evidence, and finite paid diagnostics. The definition does not require a forecast of an unobserved task’s value. Acquisition is established operationally by installation and use; favorable task value is a further property. In particular, a new tensor that no future permitted computation can consume is not evidence of acquired learning capability.

There are two important interfaces. An extension preserves the old realization and adds new degrees of freedom. A renewal changes the realization within a fixed capacity, possibly removing an old component. An extension may be function-neutral at installation; a renewal may require a finite damage budget. Neither interface is defined by its parameter count alone.

## 4.2 Function-neutral birth and accessible learning

Consider a family of predictors with an installed feature map $\phi _ { \nu } ( x ) \in \mathbb { R } ^ { k }$ and new coeficients $A \in \mathbb { R } ^ { q \times k }$

$$
f _ { \nu , A } ( x ) = f _ { 0 } ( x ) + A \phi _ { \nu } ( x ) , \qquad A \big | _ { \mathrm { i n s t a l l } } = 0 .\tag{4.2}
$$

For a fixed batch of � examples, put their feature rows in $\Phi _ { \nu } \in \mathbb { R } ^ { n \times k }$ , their output-gradient rows in $D \in \mathbb { R } ^ { n \times q }$ , and absorb the declared sample weights into �. The features are held fixed for the following single-step statement. The loss has the output-composed form ${ \cal L } ( A ) = \ell ( Z _ { 0 } + \Phi _ { \nu } A ^ { \top } )$ , with $D = \nabla _ { Z } \ell ( Z _ { 0 } )$ ; it has no additional direct parameter penalty.

Theorem 4.2 (Neutral installation, learning-active structure). All choices of � in (4.2) compute the same current function at installation. Their coeficient gradients are nevertheless

$$
\left. \nabla _ { A } L \right| _ { A = 0 } = D ^ { \top } \Phi _ { \nu } .\tag{4.3}
$$

If the consumer performs one SGD step on � alone with rate $\eta > 0 ,$ , keeping $f _ { 0 }$ and $\Phi _ { \nu }$ fixed, the exact change of the batch output matrix is

$$
\Delta Z _ { \nu } = - \eta \Phi _ { \nu } \Phi _ { \nu } ^ { \top } D .\tag{4.4}
$$

For two installed structures �, �, the resulting one-step batch-output matrices difer exactly when

$$
\big ( \boldsymbol { \Phi } _ { \nu } \boldsymbol { \Phi } _ { \nu } ^ { \intercal } - \boldsymbol { \Phi } _ { w } \boldsymbol { \Phi } _ { w } ^ { \intercal } \big ) D \neq 0 .\tag{4.5}
$$

Proof. The added output is zero at $A = 0$ for every input. Diferentiating $\Phi _ { \nu } A ^ { \top }$ gives $D ^ { \top } \Phi _ { \nu }$ . The stipulated update is $A ^ { + } = - \eta D ^ { \top } \Phi _ { \nu }$ , so the new batch output is $\Phi _ { \nu } ( A ^ { + } ) ^ { \top } = - \eta \Phi _ { \nu } \Phi _ { \nu } ^ { \top } D$ . Subtracting two such outputs proves the last assertion. □

This theorem identifies a concrete hidden learning distinction: an unchanged function can have a changed accessible learning operator. The finite-feature response $\Phi _ { \nu } \Phi _ { \nu } ^ { \intercal }$ is a kernel gradient operator of the familiar finite-feature type [12]. Function-preserving expansion itself is established prior work [6, 13]. The relevant acquisition question is what response the new structure makes possible under its actual consumer. Equation (4.4) is not an exact formula for simultaneously training all parameters or for AdamW; those consumers have diferent response maps. Equation (4.3) remains the corresponding current coeficient-gradient identity.

## 4.3 Protected access to a current learning signal

A useful specialization forms a scalar feature $a ^ { \top } x$ from a currently available input frame $\ b { X } \in \mathbb { R } ^ { n \times d }$ , with a newborn output vector $b \in \mathbb { R } ^ { q }$ initialized to zero. Let � project onto a declared protected or retained input span, and let $\ b { D } \in \mathbb { R } ^ { n \times q }$ be the current output-loss gradient. The new module contributes $b ( a ^ { \top } x )$ ; the loss depends on this module through its output, as in Theorem 4.2.

Proposition 4.3 (Protected current-signal direction). For unit $a \in$ ker �, the newborn gradient is $\nabla _ { b } L = D ^ { \top } X a$ Let

$$
G = D ^ { \top } X ( I - P ) .\tag{4.6}
$$

$H G \neq 0 .$ , a leading right singular vector of� with nonzero singular value lies in ker � and attains

$$
\operatorname* { m a x } _ { \| a \| = 1 , P a = 0 } \left\| D ^ { \top } X a \right\| = \left\| G \right\| _ { \mathrm { o p } } .\tag{4.7}
$$

$H G = 0 ,$ , no direction in this protected complement has a nonzero newborn gradientfor the statedframe and loss.

Proof. The chain rule gives the gradient. For $P a = 0 , D ^ { \top } X a = G a$ . The top singular value is the largest norm over unit inputs. Since $G ^ { \top } G = ( I - P ) G ^ { \top } G ( I - P )$ , an eigenvector with positive eigenvalue lies in ker $P$ and attains the restricted maximum. The zero case is immediate. □

The singular-vector construction is a standard variational fact, related to gradient-aligned low-rank initialization [9]. The theorem’s content for FAT is the explicit intersection of current protection and current learning access. It neither forecasts the optimal future direction nor makes the instantaneous gradient norm a utility function.

The same formula clarifies an internal interaction between structural constraints and arriving evidence. In a comparison where the captured signal $H _ { e } = D _ { e } ^ { \top } X _ { e }$ and projector $P _ { i }$ are independently specified on a common domain, let $G ( P _ { i } , e ) = H _ { e } ( I - P _ { i } )$ . Its mixed diference is exactly

$$
\begin{array} { c } { { G ( P _ { 1 } , e _ { 1 } ) - G ( P _ { 0 } , e _ { 1 } ) - G ( P _ { 1 } , e _ { 0 } ) + G ( P _ { 0 } , e _ { 0 } ) } } \\ { { { } } } \\ { { { } = - ( H _ { e _ { 1 } } - H _ { e _ { 0 } } ) ( P _ { 1 } - P _ { 0 } ) . } } \end{array}\tag{4.8}
$$

This gives a conditional operator-level nonadditivity criterion. In a live learner, evidence, features, projectors, and optimizer memory can all depend on the parent. Equation (4.8) therefore cannot substitute for a properly crossed empirical comparison, and nonzero interaction does not imply positive value.

## 4.4 Functional novelty is distinct from parameter novelty

At a fixed diferentiable state and a fixed common output frame, let $J _ { o }$ map the old trainable coordinates to output variations and $J _ { n }$ map newly installed coordinates. Both are finite matrices in this statement. The relative-image construction uses the same residual-image principle as the observation-relative response geometry of Wang [1], here applied to old and newly writable coordinates.

Proposition 4.4 (Additional tangent response dimension). The number of linearly independent output directions added by the new coordinates, relative to the old tangent image, is

$$
\begin{array} { r } { \nu = \mathrm { r a n k } \big [ J _ { o } J _ { n } \big ] - \mathrm { r a n k } J _ { o } = \mathrm { r a n k } \big [ \big ( I - J _ { o } J _ { o } ^ { \dagger } \big ) J _ { n } \big ] . } \end{array}\tag{4.9}
$$

It is zero exactly when Ran $J _ { n } \subseteq \mathbb { R } { \mathrm { a n } } J _ { o } .$

Proof. Resolve Ran $J _ { n }$ into its component in Ran $J _ { o }$ and its orthogonal complement. The latter is the image of $( I - J _ { o } J _ { o } ^ { \dagger } ) J _ { n }$ , and its dimension is exactly the increment in the dimension of the summed image. □

Three distinctions follow. A nonzero new parameter may have zero current functional derivative. A nonzero derivative may lie in the old tangent image. Finally, even zero additional tangent dimension need not imply identical optimization dynamics: coordinate realization and optimizer state can change how an existing function space is traversed [11]. Thus (4.9) is a finite-frame, first-order novelty criterion, not the dimension of all future capabilities or a requirement for every useful renewal.

## 4.5 Finite protection beyond zero output

Neutral installation by itself controls only the installation instant. One can additionally bound the efect of a prescribed nonzero use of the new structure on a current representation frame. Let $H \in \mathbb { R } ^ { n \times d } , n \geq 1$ , be that frame, $\Phi \in \mathbb { R } ^ { n \times k }$ the installed features, and $U \in \mathbb { R } ^ { d \times k }$ a prospective output coeficient with $\| U \| _ { F } \le \rho$ . Put $E = \Phi U ^ { \top }$ and $H ^ { + } = H + E$

Proposition 4.5 (Finite geometric budget). The row-Gram change is exactly

$$
\begin{array} { r } { H ^ { + } ( H ^ { + } ) ^ { \top } - H H ^ { \top } = H U \Phi ^ { \top } + \Phi U ^ { \top } H ^ { \top } + \Phi U ^ { \top } U \Phi ^ { \top } . } \end{array}\tag{4.10}
$$

Writing $e = \rho \| \Phi \| _ { \mathrm { o p } }$ gives

$$
\begin{array} { r } { \left\| H ^ { + } ( H ^ { + } ) ^ { \top } - H H ^ { \top } \right\| _ { F } \leq 2 \left\| H \right\| _ { \mathrm { o p } } e + e ^ { 2 } . } \end{array}\tag{4.11}
$$

For $C _ { H } = H ^ { \top } H / n$ and $S _ { \tau } ( C ) = C ( C + \tau I ) ^ { - 1 } , \tau > 0 ,$

$$
\| S _ { \tau } ( C _ { H ^ { + } } ) - S _ { \tau } ( C _ { H } ) \| _ { \mathrm { o p } } \leq \frac { 2 \| H \| _ { \mathrm { o p } } e + e ^ { 2 } } { n \tau } .\tag{4.12}
$$

Proof. Expansion gives (4.10). Since $\| E \| _ { F } \leq e ,$ , each cross term has Frobenius norm at most $\| H \| _ { \mathrm { o p } } e$ and $\| E E ^ { \top } \| _ { F } \leq e ^ { 2 }$ . The analogous expansion of $( H + E ) ^ { \top } ( H + E )$ bounds its operator norm by the same numerator. Finally $S _ { \tau } ( C ) = I - \tau ( C + \tau I ) ^ { - 1 }$ . The resolvent identity and $\left\| ( C + \tau I ) ^ { - 1 } \right\| _ { \mathrm { o p } } \leq 1 / \tau$ for positive semidefinite � yield the stated bound. □

These bounds concern the declared frame, insertion point, and coeficient budget. If existing residual components change the true base frame from � to $H + E _ { o }$ , the bound must be recomputed or transported with a bound on $E _ { o } ;$ certifying � does not certify the new base automatically. This finite qualification is compatible with later free learning, but it is not a perpetual protection theorem. The final fixed-capacity LoRA case study below uses a diferent, directly evaluated admission interface; it does not claim to implement every spectral certificate.

## 4.6 The actual learning consumer matters

A candidate may alter a gradient, an installed representation, optimizer memory, or several of these together. A gradient summary describes the whole learning efect only under an additional suficiency condition.

Theorem 4.6 (Local consumer suficiency). On a fixed diferentiable action chart $a \in \mathbb { R } ^ { k }$ at $a \ = \ 0$ , let $L : \mathbb { R } ^ { k } \to \mathbb { R } ^ { p }$ be the derivative of a selected gradient readout and $F _ { c } : \mathbb { R } ^ { k }  \mathbb { R } ^ { m }$ the derivative ofa common coordinate representation of the complete successor under consumer �. There exists a linear $R _ { c }$ such that $\begin{array} { r } { F _ { c } = R _ { c } L } \end{array}$ if and only if

$$
\ker L \subseteq \ker F _ { c } .\tag{4.13}
$$

For a downstream diferentiable readout with derivative $P _ { H }$ , the weaker factorization $P _ { H } F _ { c } = R _ { c , H } L$ holds exactly when ker $L \subseteq \ker ( P _ { H } F _ { c } )$

Proof. A factorization makes every zero direction of � a zero direction of the factored map. Conversely, define $R _ { c } ( L a ) = F _ { c } a$ on Ran �. The kernel inclusion makes this independent of the representative $^ { a ; }$ extend linearly to the rest of the finite-dimensional space. The downstream statement applies the same argument to $P _ { H } F _ { c }$ □

For example, $L ( a _ { 1 } , a _ { 2 } ) = a _ { 1 }$ and $F _ { c } ( a _ { 1 } , a _ { 2 } ) = ( a _ { 1 } , a _ { 2 } )$ violate (4.13). The same gradient summary can conceal a persistent structural change. Even a valid derivative factorization need not be a finite one: $L ( a ) = 0$ and a successor coordinate $a ^ { 2 }$ have zero derivatives at zero but diferent finite successors.

The corresponding exact finite statement is simple and useful. If two actions induce the same complete successor law, including retained state and costs, then every subsequent common lawful policy induces the same output law. Equality of one gradient or one endpoint score is weaker. Appendix A.3 gives a concrete optimizer-coordinate example. FAT therefore treats the consumer as part of the dynamical object rather than assuming that the same raw direction has the same meaning under SGD, AdamW, replay, and distillation.

## 4.7 Renewal under bounded capacity

For a factorized module $\begin{array} { r } { M = \sum _ { j = 1 } ^ { r } b _ { j } a _ { j } ^ { \top } } \end{array}$ , replacing one physical slot by $( \tilde { a } _ { j } , 0 )$ gives

$$
\begin{array} { r } { M ^ { + } = M - b _ { j } a _ { j } ^ { \top } . } \end{array}\tag{4.14}
$$

The newborn component is output-zero; removal of the old component need not be. Full replacement must therefore satisfy its own finite admissibility checks. Two diferent accepted newborn directions can nevertheless give the same $M ^ { + }$ while producing diferent future updates through (4.3). The updated optimizer slices are part of the renewed state.

Continual backpropagation also renews selected units by resetting incoming weights and initially zeroing outgoing weights [7]. Low-rank factor resetting and optimizer-state adjustments appear in ReLoRA, which merges accumulated low-rank updates into a base weight before reinitialization [14]. The present interface instead removes one physical slot and admits the new direction only after finite protection checks.

This distinguishes the mathematics offorming a learning-active realization from a blanket requirement for exact current output preservation. FRT and FAT share the fiber viewpoint, but FAT need not inherit every FRT protection equation. Their compatibility is a separate joint problem.

## 5 Coupled regulation–acquisition dynamics

## 5.1 Cross-response, not merely co-location

FRT and FAT are two action interfaces on one evolving learner. They are not assumed to partition that learner into disjoint physical subsystems. Let $T _ { F } ^ { f }$ and $T _ { A } ^ { a }$ denote two lawful finite writes at a common parent, and let $O _ { F } , O _ { A }$ be declared regulation and acquisition readouts. For numerical components, define

$$
\Delta _ { j } O _ { i } ( s ; a _ { j } ) = O _ { i } ( T _ { j } ^ { a _ { j } } s ) - O _ { i } ( s ) , \qquad i , j \in \{ F , A \} .\tag{5.1}
$$

The collection

$$
\begin{array} { r } { \mathcal { R } ( s ) = \left( \begin{array} { l l } { \Delta _ { F } O _ { F } } & { \Delta _ { A } O _ { F } } \\ { \Delta _ { F } O _ { A } } & { \Delta _ { A } O _ { A } } \end{array} \right) } \end{array}\tag{5.2}
$$

is a typed response table, not necessarily a numerical $2 \times 2$ matrix on which multiplication is meaningful. A candidate set, a rank label, or an accept/reject outcome is compared as a set or label, not by an invented numerica diference.

The of-diagonal entries express directional cross-efects. A nonzero acquisition-gradient response after a regulation action is one such efect; a changed repair residual or feasible set after acquisition is another. Merely sharing parameters does not establish either. Nor do internal FRT–continuation and FAT–evidence interactions logically force nonzero cross-efects between FRT and FAT. Coordination is the policy that uses these efects; coupling is the underlying state-dependent response relation.

## 5.2 Joint admissibility and sequential execution

Fix candidate labels, actual decoders, state version, and execution order. Let $\mathcal { H } ( s )$ be the relation of candidate pairs whose joint execution satisfies the stated current contract.

Proposition 5.1 (Nonrectangular joint admissibility). Two independent qualification sets represent all and only jointly admissible pairs ifand only if

$$
{ \mathcal { H } } ( s ) = \operatorname { p r o j } _ { F } { \mathcal { H } } ( s ) \times \operatorname { p r o j } _ { A } { \mathcal { H } } ( s ) .\tag{5.3}
$$

For partial deterministic commit maps �, � with domains $D _ { F } , D _ { A }$

$$
\dim ( F \circ A ) = D _ { A } \cap A ^ { - 1 } ( D _ { F } ) , \qquad \operatorname { d o m } ( A \circ F ) = D _ { F } \cap F ^ { - 1 } ( D _ { A } ) .\tag{5.4}
$$

Thus separate qualification at the parent does not establish either composition.

Proof. Any nonempty product relation is the product of its projections; conversely that product is represented by its two projections. The empty case is also represented by empty qualification sets. Equation (5.4) is precisely the condition that the first map exists and its result belongs to the second map’s domain. □

Nonrectangularity can express conflict or compensation. For instance, the pairs $( 0 , 0 )$ and (1, 1) alone form a nonrectangular set; knowing that each label appears in some valid pair does not authorize (0, 1). A repair can also make an acquisition admissible that was inadmissible by itself. Consequently a joint solver or a finite ordered recertification must use the relation itself, not assume that two isolated certificates compose.

There is an equally important information condition. For a finite candidate universe, an observation $O ( s )$ determines the exact relation precisely when

$$
O ( s ) = O ( { \tilde { s } } ) \implies \mathcal { H } ( s ) = \mathcal { H } ( { \tilde { s } } ) .\tag{5.5}
$$

A robust policy using only � must instead select from the intersection over all compatible states. The proof and the distinction between lawful HOLD and nontrivial progress are given in Appendix A.1. No full-state information is granted by naming a joint relation.

## 5.3 Exact finite cross-efects on function and certificates

In an afine output slice, let � contain feature rows, � the output head, � a head increment in transposed coordinates, and � an acquisition-induced change of those feature rows. Fixed head and feature interventions have the combined output diference

$$
( C + E ) ( W ^ { \top } + X ) - C W ^ { \top } = C X + E W ^ { \top } + E X .\tag{5.6}
$$

Even if $C X = 0$ and $E W ^ { \top } = 0$ separately, the mixed term �� can remain. For example, $C = ( 1 , 0 ) , W ^ { \top } = ( 1 , 0 ) ^ { \top }$ $ { \boldsymbol { X } } = ( 0 , 1 ) ^ { \top }$ , and $E = ( 0 , 1 )$ give two individually zero output changes and a joint change of one. This finite algebraic interaction does not require noncommuting state edits; the edits of independent coordinates in this example commute.

The same calculation transports a regulation certificate. Suppose its residuals are $r _ { C } = C X$ and $r _ { H } = H X - D$ If an acquisition changes the frames and obligation to $C ^ { \prime } = C + E _ { C } , H ^ { \prime } = H + E _ { H } , D ^ { \prime } = D + \Delta D$ , while the old write � is held fixed, then

$$
r _ { C } ^ { \prime } = r _ { C } + E _ { C } X , \qquad r _ { H } ^ { \prime } = r _ { H } + E _ { H } X - \Delta D .\tag{5.7}
$$

Equation (5.7) is exact, not a first-order approximation. It explains why a certificate is state-version-specific. It does not predict the residual of a newly recomputed canonical write, and it does not identify all future forgetting with these finite-frame residuals.

## 5.4 When repair must alter a learning channel

Cross-efects need not be accidental. A repair obligation can be incompatible with preserving an acquisition channel.

Theorem 5.2 (Obligation–learning-channel conflict). At afixed representation state, let $J \in \mathbb { R } ^ { d \times k }$ be the Jacobian of a chosen acquisition coordinate. Suppose the historical frame satisfies

$$
H = L C + B J ^ { \top }\tag{5.8}
$$

for some matrices �, �. Every write satisfying $C X = 0$ and �� = � then satisfies

$$
D = B J ^ { \top } X .\tag{5.9}
$$

Ifthe repair isfeasible and $D \neq 0 ,$ , then $\left\| B \right\| _ { \mathrm { o p } } > 0 ,$ , no such write has $J ^ { \top } { \boldsymbol { X } } = 0 ,$ , and $\| J ^ { \top } X \| _ { F } \geq \| D \| _ { F } / \| B \| _ { \mathrm { o p } } .$ For the all-label statement, let $J = D _ { \theta } h _ { \theta } ( x )$ at an example � represented in the protected frame �, and use cross-entropy $\ell _ { y } ( z ) = - \log ( \operatorname { s o f t m a x } z ) _ { y }$ for a �-class head. With a centered write $X \mathbf { 1 } = 0 $ , preserving this example’s coordinate gradientfor every possible label is equivalent to $J ^ { \top } { \boldsymbol { X } } = 0$ . Thus nonzero � under (5.8) forbids that stronger gradient-preservation demand.

Proof. Multiplying (5.8) by � gives (5.9); the norm inequality is submultiplicativity. Let $M = J ^ { \top } X$ . By Proposition 3.4, the gradient change for label � is $M ( p - e _ { \mathrm { y } } )$ . If this vanishes for every �, all columns of � equal $M p$ . Since $M \mathbf { 1 } = J ^ { \top } X \mathbf { 1 } = 0$ , their common value is zero. The converse is immediate. □

The all-label statement is stronger than a claim about the particular observed label: a nonzero matrix can annihilate that label’s residual. Nevertheless the theorem gives a structural obstruction to treating coordination as “repair without ever afecting acquisition.” Whether the forced channel change helps the actual task remains a separate response and value question.

## 5.5 Acquisition can enlarge protected repair capacity

The reverse direction has a constructive characterization. Suppose a lawful acquisition makes � new features available to a subsequent regulation action. Their current protected-frame responses are $\boldsymbol { B } \in \mathbb { R } ^ { m \times k }$ and their obligation-frame responses are $\Psi \in \mathbb { R } ^ { p \times k }$ . Old and new coeficients $\boldsymbol { X } \in \mathbb { R } ^ { d \times q }$ and $Y \in \mathbb { R } ^ { k \times q }$ must satisfy

$$
C X + B Y = 0 , \qquad H X + \Psi Y = D .\tag{5.10}
$$

We assume both coeficient blocks are actually writable, share the stated output semantics, and have squared cost $\| X \| _ { F } ^ { 2 } + \| Y \| _ { F } ^ { 2 }$ . A deployed canonical bank need not have this enlarged action interface.

Theorem 5.3 (Compensation-adjusted acquisition of repair directions). No full-row-rank assumption on � is required. Define

$$
\begin{array} { c c } { { P _ { C } = I - C ^ { \dagger } C , } } & { { R _ { C } = I - C C ^ { \dagger } , } } \\ { { } } & { { } } \\ { { N = H P _ { C } , } } & { { \mathrm { R a n } Q = \ker ( R _ { C } B ) , } } \\ { { } } & { { } } \\ { { Q ^ { \top } Q = I , } } & { { L = ( \Psi - H C ^ { \dagger } B ) Q , } } \\ { { } } & { { } } \\ { { W = I + Q ^ { \top } B ^ { \top } ( C ^ { \dagger } ) ^ { \top } C ^ { \dagger } B Q . } } & { { } } \end{array}\tag{5.11}
$$

The columns of � form any orthonormal basis of that kernel; a zero-dimensional basis is allowed. Then:

(i) The achievable obligation space of (5.10), column by column, is ${ \mathrm { R a n } } [ N L ]$ . Its increase over old protected repair is

$$
\nu _ { \mathrm { r e p a i r } } = \mathrm { r a n k } [ N L ] - \mathrm { r a n k } N \geq 0 .\tag{5.12}
$$

(ii) Let $G _ { + } = N N ^ { \top } + L W ^ { - 1 } L ^ { \top }$ . A target is feasible exactly when $G _ { + } G _ { + } ^ { \dagger } D = D$ . For such a feasible target, the minimum squared coeficient cost is

$$
E _ { + } ( D ) = \mathrm { t r } ( D ^ { \top } G _ { + } ^ { \dagger } D ) .\tag{5.13}
$$

Every old-feasible target remainsfeasible and has $E _ { + } ( D ) \leq E _ { 0 } ( D )$ , where $E _ { 0 }$ is its old minimum cost.

(iii) If � has full row rank, put $G _ { 0 } = N N ^ { \top }$ and $A _ { D } = L ^ { \top } G _ { 0 } ^ { - 1 } D$ . Then

$$
\begin{array} { r } { E _ { + } ( D ) = \mathrm { t r } \big [ D ^ { \top } ( G _ { 0 } + L W ^ { - 1 } L ^ { \top } ) ^ { - 1 } D \big ] , } \end{array}\tag{5.14}
$$

$$
E _ { 0 } ( D ) - E _ { + } ( D ) = \operatorname { t r } \left[ A _ { D } ^ { \top } ( W + L ^ { \top } G _ { 0 } ^ { - 1 } L ) ^ { - 1 } A _ { D } \right] .\tag{5.15}
$$

The improvement in this cost is strict ifand only $i f A _ { D } \ne 0$

Proof. The current constraint first requires $R _ { C } B Y = 0$ , hence $Y = Q V $ . All its solutions are

$$
X = - C ^ { \dagger } B Q V + Z , \qquad C Z = 0 .\tag{5.16}
$$

The obligation becomes $H Z + L V = D$ . The two terms in (5.16) are orthogonal, and � is an isometry, so

$$
\| X \| _ { F } ^ { 2 } + \| Y \| _ { F } ^ { 2 } = \| Z \| _ { F } ^ { 2 } + \operatorname { t r } ( V ^ { \top } W V ) .\tag{5.17}
$$

Whitening � by $W ^ { 1 / 2 }$ reduces the problem to a minimum-norm linear equation with response operator $[ N L W ^ { - 1 / 2 } ]$ Unrestricted old coordinates can be projected by $P _ { C }$ without changing their response under � and without increasing cost. Therefore its image is ${ \mathrm { R a n } } [ N L ]$ , its output Gram matrix is $G _ { + }$ , and the minimum-norm solution gives (5.13). Setting $Y = 0$ embeds every old-feasible solution, proving nonincrease without assuming invertibility. When � has full row rank, the Woodbury identity gives (5.15). The middle matrix there is positive definite, so the trace is zero exactly when $A _ { D } = 0$ □

The same minimum-norm problem is an instance of the inherited protected repair geometry (Theorem 3.2) with augmented matrices $\bar { C } = [ C B ]$ and $\bar { H } = [ H \Psi ]$ . Consequently, the response Gram matrix also has the exact form

$$
\begin{array} { c } { G _ { + } = N N ^ { \top } + L W ^ { - 1 } L ^ { \top } } \\ { = \bar { H } ( I - \bar { C } ^ { \dagger } \bar { C } ) \bar { H } ^ { \top } . } \end{array}\tag{5.18}
$$

To see the equality directly, let � have orthonormal columns spanning ker �. The columns of

$$
E = \left( { \begin{array} { c c } { U } & { - C ^ { \dagger } B Q } \\ { 0 } & { Q } \end{array} } \right)
$$

span ker $\bar { C } ,$ , with $E ^ { \top } E = \mathrm { d i a g } ( I , W )$ and $\bar { H } E = [ H U L ]$ . Thus � diag(�, $W ^ { - 1 } ) E ^ { \top } = I - \bar { C } ^ { \dagger } \bar { C } .$ , which gives (5.18). The underlying least-norm projection is standard. The decomposition into compatible acquisition directions $Q .$ the response after protection compensation �, and the cost metric � makes explicit how the acquired structure changes repair reachability and expense.

The theorem identifies the new repair capacity created by acquisition: only �, the acquired response remaining after compensation on the protected frame, contributes new admissible repair directions. If $B = 0 ,$ , all new coeficients are compatible with current protection; choosing $Q = I$ gives � = � and $L = \Psi$ . If instead $\Psi = H { \cal C } ^ { \dagger }$ � on the allowable new directions, a nonzero raw historical response can contribute no additional protected repair direction.

Acquisition can therefore benefit regulation in two mathematically distinct ways. It can make a previously infeasible repair target attainable by enlarging Ran $\textit { N L }$ ; or it can strictly lower the coeficient cost of a target that was already attainable, even when $\nu _ { \mathrm { r e p a i r } } = 0$ . Both efects are exact for the writable interface and metric in (5.10); an implementation with fewer candidates must be evaluated on its actual action set.

![](images/7b6221d5d844430955f795f3b3979010f278dc548de5cabf14ccc62b55d29cf0.jpg)

![](images/c7d68e1378418671bd274f9e54c61625a95c9bb9a5b6495525b0b1fb00ad9fb5.jpg)  
Figure 2. Acquisition-adjusted repair geometry. Two independent exact afine examples of Theorem 5.3 are shown in obligation-response coordinates. The solid old set and dashed enlarged set contain responses attainable with squared coeficient cost at most one. In both examples, � = 1, $B = Q = 1$ and $W = 2 .$ (a) Set $C = ( 1 , 0 ) , H = { \binom { 1 } { 1 0 } }$ and $\Psi = ( 1 , 2 ) ^ { \intercal }$ With $\nu = 1 / 2$ , the raw acquired response $S = \Psi \nu$ receives the compensation response $- H C ^ { \dagger } B \nu .$ , yielding $L \nu = D$ . The target $D = ( 0 , 1 / 2 ) ^ { \top }$ lies outside the old reachable space. (b) Set $\begin{array} { r } { C = ( 1 , 0 , 0 ) , H = \left( \begin{array} { l l } { 1 } & { 1 } & { 0 } \\ { 1 } & { 0 } & { 1 } \end{array} \right) } \end{array}$ and $\Psi = ( 2 , 1 ) ^ { \top }$ . Both interfaces span the response plane, yet the minimum cost of $D = ( 1 , 0 ) ^ { 7 }$ falls from 1 to $2 / 3$ . The arrows in (a) express vector addition, not a learning trajectory.

## 5.6 Jointly reachable current responses and infeasibility

A second characterization asks whether repair, protection, and a desired current learning response can be satisfied within one budget. Include the actual linear actuator image in coordinates $z \in \mathbb { R } ^ { n }$ . Let $M z = d$ collect the fixed equality constraints and let $K z$ be the response of interest.

Theorem 5.4 (Joint response ellipsoid and dual obstruction). Suppose $M z = d$ is feasible. Set $z _ { 0 } = M ^ { \dagger } d ,$ $P = I - M ^ { \dagger } M$ , and let $\tau \geq \| z _ { 0 } \|$ with $\rho ^ { 2 } = \tau ^ { 2 } - \| z _ { 0 } \| ^ { 2 }$ . The responses obtainable subject to $\left\| z \right\| \leq \tau$ are

$$
K z _ { 0 } + K P \{ u : \| u \| \leq \rho \} .\tag{5.19}
$$

A target � is attainable precisely when

$$
q - K z _ { 0 } \in \mathbf { R a n } ( K P ) , \qquad \left\| ( K P ) ^ { \dagger } ( q - K z _ { 0 } ) \right\| \le \rho .\tag{5.20}
$$

For $b = K z _ { 0 } - q$ and $T = K P$ , its minimum residual has the exact dualform

$$
\operatorname* { m i n } _ { \| u \| \leq \rho } \| b + T u \| = \operatorname* { m a x } _ { \| \boldsymbol { \lambda } \| \leq 1 } \{ \langle \boldsymbol { \lambda } , b \rangle - \rho \left\| \boldsymbol { T } ^ { \top } \boldsymbol { \lambda } \right\| \} .\tag{5.21}
$$

A strictly positive value certifies current infeasibility at the stated budget.

Proof. Every solution is $z _ { 0 } + P u$ with $z _ { 0 } \perp$ Ran �; Pythagoras gives (5.19). The least-norm preimage of a response diference is $( K P ) ^ { \dag } ( q - K z _ { 0 } )$ , giving (5.20). For any unit-ball � and admissible �, $\Vert b + T u \Vert \geq \left. \lambda , b + T u \right. \geq$ $\langle \lambda , b \rangle - \rho \| T ^ { \top } \lambda \|$ . For equality, choose a minimizing $u _ { * }$ . If its residual is zero, the lower bound zero is attained by $\lambda = 0$ . Otherwise take $\lambda _ { * } = ( b + T u _ { * } ) / \Vert b + T u _ { * } \Vert$ . Convex first-order optimality on the ball implies that $u _ { * }$ minimizes $\langle T ^ { \top } \lambda _ { * } , u \rangle$ on that ball, whose value is $- \rho \| T ^ { \top } \lambda _ { * } \|$ . Substitution attains equality. □

The theorem is a finite-dimensional constrained-response result using standard convex geometry [15]. It is particularly useful for distinguishing “there is hidden freedom” from “the specified repair and acquisition response fit inside the same available action budget.” A canonical solver that emits only a few candidates must still be tested on those candidates; the full ellipsoid is then a bound or design space, not its automatic menu.

## 5.7 Safe reachable directions and finite continuation value

Joint reachability describes which responses can be realized. Their downstream value is a separate, continuationdependent question. A local geometric criterion connects the two. Let $\mathcal { K }$ be a closed convex cone of legally realizable infinitesimal action directions, already incorporating current linear protection and a specified first-order retention condition. At a fixed parent and fixed continuation, assume a diferentiable quality $j ( a )$ at zero, with $p _ { j } = \nabla j ( 0 )$ , and let $\Pi _ { \mathcal { K } }$ denote metric projection. Standard cone geometry gives

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { a \in \mathcal { K } , \| a \| \leq 1 } \left. p _ { j } , a \right. = \left\| \Pi _ { \mathcal { K } } p _ { j } \right\| , } \\ & { \displaystyle \exists a \in \mathcal { K } : \left. p _ { j } , a \right. > 0 \iff \Pi _ { \mathcal { K } } p _ { j } \neq 0 . } \end{array}\tag{5.22}
$$

If the ray ��, $0 \leq t \leq r ,$ is actually legal and $\nabla j$ is �-Lipschitz along that ray, then for unit $a \in \mathcal K$

$$
\begin{array} { r } { j ( t a ) - j ( 0 ) \geq t \left. p _ { j } , a \right. - \frac { 1 } { 2 } M t ^ { 2 } . } \end{array}\tag{5.23}
$$

For a computed estimate $\widehat { p } _ { j }$ with $\left\| { \widehat { p } } _ { j } - p _ { j } \right\| \leq e$ , suppose � bounds the contrast discrepancy:

$$
| \Delta j _ { \mathrm { n a t i v e } } ( t a ) - [ j ( t a ) - j ( 0 ) ] | \leq \delta .
$$

Then a finite-native lower bound is

$$
\begin{array} { r } { \Delta j _ { \mathrm { n a t i v e } } ( t a ) \geq t \left. \widehat { p } _ { j } , a \right. - e t - \frac { 1 } { 2 } M t ^ { 2 } - \delta . } \end{array}\tag{5.24}
$$

If instead a pointwise error bound $| j _ { \mathrm { n a t i v e } } ( b ) - j ( b ) | \leq \delta _ { 0 }$ is known at both $b = 0$ and $b = t a$ , one must use $\delta = 2 \delta _ { 0 }$ in (5.24). The equalities follow by orthogonal cone projection and Cauchy–Schwarz; the inequalities follow by integrating the derivative and bounding the gradient error. Positive target value and allowable retention damage require their separate bounds. A nonempty protected action space does not make $\Pi _ { \mathcal { K P j } }$ nonzero, and an unmeasured future covector does not constitute a certified online rule. The criterion characterizes favorable reachable directions; the finite bound specifies when that local direction also certifies a favorable realized continuation. It separates the geometry of legal intervention from its future value.

## 5.8 A finite learning example of positive complementarity

A supervised gradient update can make the two operations complementary even when each alone leaves the target response unchanged. The following construction establishes such a case with specified features, loss, and optimizer step.

Theorem 5.5 (Complementarity through an actual gradient update). Let � ∈ R be a trainable representation coordinate, initially zero, and consider two-class logits $W \phi _ { \psi } ( x )$ with a headfrozen during a single supervised update of �. On a support example $s ,$ a historical example $h ,$ and two query examples $x _ { \sigma }$ with $\sigma \in \{ - 1 , 1 \}$ define the diferentiablefeatures

$$
\begin{array} { c } { { \phi _ { \psi } ( s ) = ( 0 , \psi , 1 ) ^ { \top } , } } \\ { { { } } } \\ { { \phi _ { \psi } ( h ) = ( 1 , 0 , 1 ) ^ { \top } , } } \\ { { { } } } \\ { { \phi _ { \psi } ( x _ { \sigma } ) = ( \sigma \psi , 0 , 1 ) ^ { \top } . } } \end{array}\tag{5.25}
$$

The initial head is zero. Regulation retains the historical repair

$$
W _ { F } = \left( \begin{array} { l l l } { { d } } & { { 0 } } & { { 0 } } \\ { { - d } } & { { 0 } } & { { 0 } } \end{array} \right) , \qquad d > 0 ,\tag{5.26}
$$

and acquisition installs, for the support update only, the temporary component

$$
W _ { A } ( q ) = \left( 0 \quad b q \quad 0 \right) , \qquad b > 0 , \quad q \in \{ - 1 , 1 \} .\tag{5.27}
$$

The acquired cue � comesfrom already available information. It agrees with an unobserved target sign $t \in \{ - 1 , 1 \}$ with probability $p ,$ but thefuture query labels are not available to the action rule. The support and historical labels are class one;for query $x _ { \sigma }$ the correct label is class one exactly when $\sigma t = 1$ . Perform a single unclipped SGD step of size $\eta > 0$ on support cross-entropy, then remove $W _ { A }$ while retaining $W _ { F }$ and the updated $\psi .$ . Write $\kappa = 2 d \eta b$

At installation the support logits are identical in allfour combinations, and the historical example obtains the same repaired margin 2� whenever regulation is applied. After acquisition removal, the no-action, regulation-only, and acquisition-only policies each give zero query logits and query NLL log 2. The joint policy has expected query-NLL improvement

$$
\mathcal { G } ( \boldsymbol { p } , \kappa ) = \frac { ( 2 \boldsymbol { p } - 1 ) \kappa } { 2 } - \log \cosh \left( \frac { \kappa } { 2 } \right) ,\tag{5.28}
$$

$$
\mathcal { G } ( \boldsymbol { p } , \kappa ) > 0 \Longleftrightarrow p > \frac { 1 } { 2 } + \frac { \log \cosh ( \kappa / 2 ) } { \kappa } .
$$

Under a fixed first-class tie rule, the isolated policies have balanced query accuracy $1 / 2 ,$ , while the joint policy has expected query accuracy �. In particular, $p = 1$ yields strict positive joint NLL improvement over each isolated policy while retaining the specified historical repair exactly.

Proof. The support logits are zero at $\psi = 0 ,$ , and the regulation head multiplies the zero support coordinate. With the acquisition component, they become $( b q \psi , - b q \psi )$ , so the genuine support-loss derivative at zero $\mathrm { i s } - b q$ The SGD update is therefore $\psi ^ { + } = \eta b q$ when acquisition is present and zero otherwise. After the temporary head is removed, the joint query logits equal $( \sigma d \eta b q , - \sigma d \eta b q )$ ; every other branch yields zero query logits. The correct-class logit margin on either query equals ���, and its cross-entropy is log $( 1 + e ^ { - \kappa q t } )$ . Using $\operatorname* { P r } ( q = t ) = p$ and log $( 1 + e ^ { - z } ) = \log 2 + \log$ cosh $( z / 2 ) - z / 2$ gives (5.28). The two query labels are opposite, hence a zero-logit tie predicts one of two correctly, while joint predictions agree with both precisely when $q = t$ . Finally, the historical feature is independent of $\psi$ and annihilates $W _ { A }$ , preserving the regulation margin throughout. □

The example establishes a concrete mechanism: acquisition opens the support-gradient channel, while regulation makes the resulting representation change visible at the queries. Neither action alone yields this response. For a first bias-corrected AdamW step from zero moments with inactive clipping, the same construction replaces �� by $\eta b / ( b + \epsilon )$ (under the stated denominator convention), so the mechanism is not specific to plain SGD. Positive NLL gain depends on the cue reliability and response magnitude as specified in (5.28); an equally informed predictor that uses the cue directly may reproduce the query decisions. The construction is an exact learning example, rather than a Qwen measurement or an all-method optimality claim.

## 5.9 Propagation through continued learning

The finite interventions above change the state on which subsequent learning acts. On a common convex chart, suppose $U _ { t }$ is $C ^ { 1 }$ on an open neighborhood of the entire comparison segment between $s _ { t } ^ { 0 }$ and $s _ { t } ^ { 1 }$ ; assume the compared readout � is likewise $C ^ { 1 }$ on its relevant segments. Let both states follow this same deterministic update. With $\delta _ { t } = s _ { t } ^ { 1 } - s _ { t } ^ { 0 }$

$$
\delta _ { t + 1 } = \overline { { J } } _ { t } \delta _ { t } , \qquad \overline { { J } } _ { t } = \int _ { 0 } ^ { 1 } D U _ { t } ( s _ { t } ^ { 0 } + u \delta _ { t } ) d u .\tag{5.29}
$$

The line-segment fundamental theorem of calculus then gives the displayed identity, without assuming a constant Jacobian. Consequently $\delta _ { H } = \overline { { J } } _ { H - 1 } \cdot \cdot \cdot \overline { { J } } _ { 0 } \delta _ { 0 }$ . A readout diference has the analogous path integral through ��. In a piecewise-smooth or discrete system, (5.29) applies only on qualified branches; actual kernel composition remains the general definition.

When a policy recomputes actions after the state changes, it also changes later inputs to the update. For example, a regulation write can change the gradient read by a later FAT generator, and a renewed representation can change a later repair obligation. This is a forward-time feedback path. It is not represented by multiplying the four heterogeneous blocks of (5.2). Exact execution retains the generator, its information, the optimizer, and the action sequence in the complete state.

## 5.10 From local efects to policy value

Coupling can help, hinder, or leave a particular readout unchanged. For a valid four-policy comparison from the same parent, write $I = J _ { 1 1 } - J _ { 1 0 } - J _ { 0 1 } + J _ { 0 0 } , a = J _ { 1 0 } - J _ { 0 0 }$ , and $b = J _ { 0 1 } - J _ { 0 0 }$ . Then

$$
J _ { 1 1 } > \operatorname* { m a x } \{ J _ { 1 0 } , J _ { 0 1 } \} \quad \Longleftrightarrow \quad I > - \operatorname* { m i n } \{ a , b \} .\tag{5.30}
$$

Thus joint advantage over both separate policies does not require positive superadditivity. For instance, $( J _ { 0 0 } , J _ { 1 0 } , J _ { 0 1 } , J _ { 1 1 } ) = ( 0 , 3 , 4 , 5 )$ gives joint advantage but $I = - 2$ . Moreover a table comparing diferently coordinated policies is not automatically a valid factorial experiment.

A rigorous connection to performance is available, but it requires actual continuation value rather than a convenient current proxy. Let � be a lawful reference policy on the full compared state space, with terminal utility $J _ { T }$ and finite horizon �. Here $K _ { t } ^ { a }$ and $K _ { t } ^ { \beta }$ denote the successor marginals of the relevant kernels. Its values satisfy $V _ { T } ^ { \beta } = J _ { T }$ and $V _ { t } ^ { \beta } = K _ { t } ^ { \beta } V _ { t + 1 } ^ { \beta }$ . Define

$$
A _ { t } ^ { \beta } ( s , a ) = ( K _ { t } ^ { a } - K _ { t } ^ { \beta } ) V _ { t + 1 } ^ { \beta } ( s ) .\tag{5.31}
$$

All relevant costs are included in the state/utility, and actions have common time and stopping semantics.

Proposition 5.6 (Conditional value of coordinated intervention). For any lawful policy � with the same initial law and integrable values,

$$
J ( \alpha ) - J ( \beta ) = \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } _ { \alpha } [ A _ { t } ^ { \beta } ( s _ { t } , a _ { t } ) ] .\tag{5.32}
$$

If information available before each decision provides a valid uniform lower bound on the true advantage of every selected extra action at every reached state, and at each decision the policy either chooses an action with nonnegative certified advantage orfollows the reference decision rule, then $J ( \alpha ) \ge J ( \beta )$ . Strict positivity ofa least one expected summand gives strict improvement.

Proof. Conditional expectation turns each summand into $\mathbb { E } _ { \alpha } [ V _ { t + 1 } ^ { \beta } ( s _ { t + 1 } ) - V _ { t } ^ { \beta } ( s _ { t } ) ]$ . Summation telescopes to $\mathbb { E } _ { \alpha } J _ { T } ( s _ { T } ) - \mathbb { E } V _ { 0 } ^ { \beta } ( s _ { 0 } )$ . The latter expectation equals $J ( \beta )$ under the common initial law. The sign conclusion follows term by term. □

This is an application of a standard finite-horizon policy-diference argument, not an additional algorithm or a claim that the required lower bound is cheap or known [2]. In particular, a current margin, gradient norm, or finite CE improvement is not automatically such a bound. A probabilistic calibration event supports only its stated conditional guarantee unless failure losses are also controlled. The experiments below test useful coordinated specializations without claiming that they implement this suficient condition for universal policy improvement.

## 6 Transformer and language-model specialization

We now realize protection, acquisition, and their cross-responses in a causal Transformer with writable head and low-rank adaptation interfaces. The implementation connects the preceding general results to actual gradient updates, structural writes, and two diferent learning rules.

## 6.1 A common complete learner and writable interfaces

Let $h _ { \theta } ( x ) \in \mathbb R ^ { d }$ be the last hidden representation at a supervised token position and let $z ( x ) = W h _ { \theta } ( x ) \in \mathbb { R } ^ { V }$ The base transformer is frozen in the case study, while LoRA coordinates within � are trainable [13, 16]. A committed compact repair bank contributes to the efective head �. The complete state also includes the LoRA

factors, optimizer moments and clocks, memory, teacher when present, bank, coordination state, and causal stream position.

A captured current feature frame � and historical frame � give exactly the repair equations $C X = 0 , H X = D$ for a head increment $X = \Delta W ^ { \top }$ . With margin threshold $\mu ,$ the historical row � has deficit

$$
m _ { i } = z _ { i , y _ { i } } - \operatorname* { m a x } _ { \nu \neq y _ { i } } z _ { i , \nu } ,
$$

$$
d _ { i } = \left\{ \begin{array} { l l } { 0 , } & { m _ { i } \geq \mu , } \\ { \mu + \zeta - m _ { i } , } & { m _ { i } < \mu . } \end{array} \right.\tag{6.1}
$$

Let $S _ { i } = e _ { \mathrm { v } _ { i } } ^ { \top } - \mathbf { 1 } ^ { \top } / V$ and $D = \mathrm { d i a g } ( d ) S$ . Adding this centered row target raises every correct-versus-competitor margin by $d _ { i } , \mathbf { A }$ passing exact target therefore repairs the stated historical deficit while leaving the current frame unchanged.

The physical compact bank does not expose an arbitrary dense $d \times V$ write. It stores a factor � and the fixed label directions �, with

$$
\begin{array} { l } { { N = H P _ { C } , } } \\ { { \ } } \\ { { T = \mathrm { d i a g } ( d ) ( N N ^ { \top } ) ^ { - 1 } N , } } \\ { { \ } } \\ { { X = T ^ { \top } S . } } \end{array}\tag{6.2}
$$

on its qualified full-row-rank domain. The candidate is solved in FP64, stored in FP32, and checked after storage.   
Failed rank or numerical checks are not replaced by an unregistered pseudoinverse or ridge repair.

If � is an orthonormal basis for Ran $S ^ { \top }$ , every representable dense write has $X = Z Q ^ { \top }$ and $\| X \| _ { F } = \| Z \| _ { F }$ A response equation $A X = Y$ requires $Y ( I - Q Q ^ { \top } ) = 0$ and $A Z = Y Q$ . This is an exact restriction to the bank image. The canonical candidate in (6.2) is narrower still. Thus Theorems 3.2 and 5.4 apply only after the chosen source interface is imposed; an ideal null-space variation is not automatically executable.

## 6.2 LoRA renewal as an acquisition interface

For one target projection, write

$$
M = B _ { A } A _ { A } = \sum _ { j = 1 } ^ { r } b _ { j } a _ { j } ^ { \top } .\tag{6.3}
$$

The subscript distinguishes these factors from the observation map �. The efective projection is $M _ { 0 } + \gamma M$ , with $\gamma = \alpha / r = 2$ in the experiments. At a task-arrival event, $X _ { t }$ contains module-input rows from already arrived examples, $D _ { t }$ their module-output loss gradients, and $O _ { t }$ resident historical inputs. For physical slot $j ,$ let $K _ { t , j }$ contain the seven retained input-factor rows and let $P _ { t , j }$ project onto rowspan $[ O _ { t } ; K _ { t , j } ]$

Replay renewal. The replay realization forms its initial unit input direction from

$$
\begin{array} { r } { G _ { t , j } = D _ { t } ^ { \top } X _ { t } ( I - P _ { t , j } ) . } \end{array}\tag{6.4}
$$

A top right singular vector gives the direction in Proposition 4.3; the factor � does not change this maximizing direction. The implementation scales this direction by the median norm of the retained input factors. It replaces $a _ { j }$ , sets $b _ { j } = 0$ , and checks the resulting finite current and historical changes. The projected signal is used to propose a candidate; only these finite checks determine acceptance. Diferent accepted newborn directions after the same deletion can therefore share the same immediate efective matrix, as in (4.14).

Distillation renewal. The LwF realization also accounts for the retained output span. If $Q _ { t , j }$ has orthonormal columns spanning the seven retained � columns, its initial input-direction proposal uses

$$
\begin{array} { r } { G _ { t , j } ^ { D } = \gamma ( I - \mathcal { Q } _ { t , j } \mathcal { Q } _ { t , j } ^ { \top } ) D _ { C , t } ^ { \top } X _ { t } ( I - P _ { t , j } ) , } \end{array}\tag{6.5}
$$

where $D _ { C , t }$ is the CE component of the current module-output gradient. The proposed direction is further projected outside the historical inputs and all parent input-factor rows; a zero residual is rejected. The remaining unit direction is scaled by the replaced parent row’s norm. This final direction need not still maximize (6.5) under the enlarged projection.

The candidate is first staged with $b _ { j } = 0$ . CE and weighted-KL gradients are then recomputed at that staged state, using cached teacher outputs. A finite, generally nonzero output column is constructed along a CE-descending direction orthogonal to the staged KL gradient within the retained-output complement. Acceptance requires a strict parent-relative CE decrease, the specified finite KL-growth bound, and the historical-damage checks. Thus zeroing $b _ { j }$ is an intermediate step, not the final LwF write. Appendix B gives the amplitude and selection rules.

In both realizations, only the selected factor row/column and their moment slices are renewed; the scalar optimizer step and the seven other slots are retained. Neither complete replacement is assumed function-neutral relative to the parent. Temporary expansion followed by compression would be a diferent capacity interface. A general reparameterization of $A _ { A } , B _ { A }$ preserves neither physical-slot identity nor this intervention’s optimizer-state changes.

## 6.3 Optimizer-native response

For AdamW, denote the clipped gradient by �, the step counter by �, and the moments by $m , \nu .$ . The ideal scalar-coordinate update is [17, 18]

$$
\begin{array} { c } { { m ^ { + } = \beta _ { 1 } m + ( 1 - \beta _ { 1 } ) g , } } \\ { { \ } } \\ { { \nu ^ { + } = \beta _ { 2 } \nu + ( 1 - \beta _ { 2 } ) g ^ { 2 } . } } \\ { { \theta ^ { + } = ( 1 - \eta \lambda _ { w } ) \theta } } \\ { { \displaystyle \qquad - \eta \frac { m ^ { + } / ( 1 - \beta _ { 1 } ^ { n + 1 } ) } { \sqrt { \nu ^ { + } / ( 1 - \beta _ { 2 } ^ { n + 1 } ) } + \epsilon } . } } \end{array}\tag{6.6}
$$

(6.7)

For two branches with the same moment hyperparameters and counter, their exact moment diferences satisfy

$$
\delta m ^ { + } = \beta _ { 1 } \delta m + ( 1 - \beta _ { 1 } ) \delta g ,\tag{6.8}
$$

$$
\delta \nu ^ { + } = \beta _ { 2 } \delta \nu + ( 1 - \beta _ { 2 } ) ( 2 g ^ { 0 } \delta g + ( \delta g ) ^ { 2 } ) .\tag{6.9}
$$

The parameter response follows by subtracting (6.7) with each branch’s own moments. Equal raw gradients are insuficient when clipping, moments, or clocks difer. The finite head-gradient identity remains an exact real-arithmetic statement at the declared current cut, while the actual optimizer consumes clipped, numerically realized gradients. This distinction is the consumer boundary of Theorem 4.6.

## 6.4 Replay and output distillation are diferent consumers

With available replay [19], the ordinary loss in the first realization is

$$
\begin{array} { r } { L _ { R } = \frac { 4 } { 5 } \overline { { \mathrm { C E } } } _ { \mathrm { c u r r e n t } } + \frac { 1 } { 5 } \overline { { \mathrm { C E } } } _ { \mathrm { r e p l a y } } ; } \end{array}\tag{6.10}
$$

without an old-task pool it is current CE alone. Regulation-linked allocation can later replace an eligible replay item using a signal published by an accepted repair. The head write and experience allocation are distinct operational interfaces on the same learner.

In the second realization there is no ordinary replay training. A frozen teacher provides a token distribution $q _ { T }$ and

$$
\begin{array} { r } { L _ { D } ( z , y ; q _ { T } ) = \mathrm { C E } ( z , y ) + \lambda T ^ { 2 } \mathrm { K L } ( q _ { T } \| p _ { T } ) , \phantom { x x x x x x x x x x x x x x x x x x x } } \\ { p _ { T } = \mathrm { s o f t m a x } ( z / T ) , \phantom { x x x x x x x x x x } p = \mathrm { s o f t m a x } ( z ) . } \end{array}\tag{6.11}
$$

This is a token-level output-distillation specialization inspired by learning without forgetting [20], not a claim to reproduce every part of its original protocol. The archived runs use $T = 2$ and $\lambda = 1 \cdot$ : the multiplier of the raw temperature-two KL is therefore four.

Proposition 6.1 (Exact distillation response). For afixed teacher, the logit gradient and Hessian are

$$
r _ { D } = p - e _ { y } + \lambda T ( p _ { T } - q _ { T } ) ,\tag{6.12}
$$

$$
H _ { D } = \mathrm { d i a g } ( p ) - p p ^ { \top } + \lambda \left[ \mathrm { d i a g } ( p _ { T } ) - p _ { T } p _ { T } ^ { \top } \right] .\tag{6.13}
$$

For any finite logit change �, the exact change in loss is

$$
\begin{array} { l } { \displaystyle \Delta L _ { D } = \log \sum _ { \nu } p _ { \nu } e ^ { \delta _ { \nu } } - \delta _ { y } } \\ { \displaystyle \qquad + \lambda T ^ { 2 } \left[ \log \sum _ { \nu } ( p _ { T } ) _ { \nu } e ^ { \delta _ { \nu } / T } - q _ { T } ^ { \top } \delta / T \right] . } \end{array}\tag{6.14}
$$

A protected head write has acquisition-coordinate gradient change $J ^ { \top } X r _ { D }$ . At a fixed student, replacing $q _ { T }$ by $q _ { T } ^ { \prime }$ changes its parameter gradient by

$$
\Delta \nabla _ { \boldsymbol { \theta } } L _ { D } = - \lambda T J _ { z } ^ { \top } ( q _ { T } ^ { \prime } - q _ { T } ) ,\tag{6.15}
$$

where $J _ { z } = D _ { \theta } z .$

Proof. Use $\nabla _ { z }$ log $\begin{array} { r } { \sum _ { \nu } e ^ { z _ { \nu } / T } = p _ { T } / T } \end{array}$ . Diferentiating once more gives the temperature-cancelled covariance in (6.13). For finite �, subtract the two log-normalizers and the two linear target terms, giving (6.14). The protected head formula follows from Proposition 3.4; the teacher formula follows by subtracting (6.12) and multiplying by $J _ { z } ^ { \top }$ □

At a teacher-matched token, $q _ { T } = p _ { T }$ , the current KL gradient is zero but a finite nonconstant logit change has positive KL cost. A rule based only on that zero first-order gradient therefore misses a genuine finite admission cost. This is why the LwF acquisition specialization checks actual finite CE and KL rather than simply copying a replay-gradient heuristic. Teacher refresh is a separate temporal operation: a committed bank can be inherited by the next teacher, but its efect on a new input is measurable only after that input actually arrives.

## 6.5 The instantiated cross-coupling

Three interfaces connect the theory to the executable joint learner. A head repair changes acquisition’s current gradient through $J ^ { \top } X r ,$ even when its protected logits are unchanged. A structural renewal changes captured features and historical deficits, so the next regulation certificate must use its current state, as in (5.7). A coordinated action rule can use these arrived functional and acquisition readouts to choose among its genuinely available candidates.

In the replay joint policy, the selected FAT rule is combined with regulation-linked replay allocation. In the distillation joint policy, the rule first prefers eligible acquisition candidates that introduce no new historical margin deficits, then candidates that do not increase the existing total deficit, and then the largest already measured finite current CE gain; prescribed fallback and one-event rank cooldown remain part of the rule. The regulation decision is also benefit-conditioned among the canonical write and no-write. These are finite online preferences, not certified lower bounds on future policy value.

Theorem 5.3 describes a possible enlargedjoint repair interface, whereas these experiments use a compact canonical bank and finite structural candidates. They do not claim to solve its unrestricted old-plus-new coeficient problem. The theoretical result remains meaningful as a characterization of when acquisition can enlarge repair capacity; its stronger solver-level instantiation is distinct from the cross-gradient and certificate pathways tested here.

## 7 Empirical validation of the specializations

We examine two consequences of the theory: regulation and acquisition can change each other’s current responses, and coordinated policies can improve learning outcomes beyond either separate policy. The experiments instantiate the interfaces in Section $^ { 6 , }$ rather than the unrestricted joint solver of Theorem 5.3.

Table 1. Full-policy outcomes on the same 128-item MASSIVE-derived panel. Exact is teacher-forced label-plus-EOS correctness. Bold marks the highest Exact and lowest macro NLL within each learning rule and trajectory. These are complete-policy comparisons, not fixed-action factorial experiments.
<table><tr><td></td><td></td><td colspan="2">Original trajectory</td><td colspan="2">Changed trajectory</td></tr><tr><td>Learning rule</td><td>Policy</td><td>Exact</td><td>Macro NLL</td><td>Exact</td><td>Macro NLL</td></tr><tr><td>Replay</td><td>Baseline</td><td>102/128</td><td>0.279321</td><td>110/128</td><td>0.249136</td></tr><tr><td></td><td>FRT</td><td>113/128</td><td>0.200758</td><td>112/128</td><td>0.268962</td></tr><tr><td></td><td>FAT</td><td>114/128</td><td>0.216516</td><td>110/128</td><td>0.263532</td></tr><tr><td></td><td>Joint FRT-FAT</td><td>115/128</td><td>0.222998</td><td>113/128</td><td>0.240527</td></tr><tr><td>LwF</td><td>Baseline</td><td>85/128</td><td>0.491530</td><td>75/128</td><td>0.547257</td></tr><tr><td></td><td>FRT</td><td>93/128</td><td>0.404870</td><td>92/128</td><td>0.450175</td></tr><tr><td></td><td>FAT</td><td>87/128</td><td>0.455100</td><td>77/128</td><td>0.539343</td></tr><tr><td></td><td>Joint FRT–FAT</td><td>94/128</td><td>0.440946</td><td>94/128</td><td>0.421768</td></tr></table>

## 7.1 Setting and evaluation

The experiments use Qwen2.5-32B with rank-eight LoRA and AdamW on a four-task stream derived from MASSIVE 1.1 en-US [16, 21]. There are 16 intents, with 64 training and eight evaluation examples per intent: 1,024 training examples and a disjoint 128-item evaluation panel. Each task contains four intents and receives 64 ordinary updates, giving 256 updates per run. The panel was repeatedly examined during development.

We examine two training trajectories. The first uses the original random seed and example sequence. The second uses a new seed and a diferent permutation of examples within each task; the four task groups retain the same order. Both trajectories use the same 1,024 training examples and the same 128 evaluation items. Within each trajectory and learning rule, the four policies begin from the same experimental configuration and share the ordinary-update budget, although their states can diverge after intervention. The second trajectory tests a changed training history, not generalization to new data or the separate efects of seed and order.

Exact counts an example only when all answer-label tokens and EOS are correctly predicted under teacher forcing. It is not free-generation exact match. Macro NLL averages supervised-token mean loss within each example, then within and across intents. The balanced panel makes it equal to the mean of the 128 example losses, not to a pooled token loss. The original pilot designated NLL as primary and Exact as secondary; the Exact-first presentation here is retrospective.

Online decisions use arrived training examples and resident historical information, not evaluation scores or unseen future-task inputs. The ordinary-update budget is matched within each learning rule. Auxiliary work is not: in both trajectories, baseline, FRT, FAT, and joint LwF use 771, 771, 783, and 803 teacher forwards, respectively. Candidate probes, sensing, and repair also incur costs. Appendix B records these implementation and information-access details.

## 7.2 Independent and coordinated policies

Table 1 reports the full-policy endpoints. The coordinated policy exceeds both separate structural policies in Exact in every learning-rule–setting comparison. Its net gains over the stronger separate policy are one example for replay under each trajectory, one for distillation under the original trajectory, and two for distillation under the changed trajectory. It also has the lowest macro NLL under both learning rules in the changed trajectory.

The separate mechanisms exhibit diferent metric trade-ofs. In the changed replay trajectory, FRT raises Exact but worsens NLL, while FAT matches baseline Exact with higher NLL. Under the original trajectory, both join policies lead in Exact but not in NLL; in the changed trajectory, they lead on both readouts.

## 7.3 The changed-trajectory test

Under the changed replay trajectory, FRT and joint each commit three repairs; FAT and joint each commit three structural renewals. Under changed distillation, FRT and joint each commit five repairs, while FAT and joint each commit three renewals. The LwF joint policy selects diferent later physical slots from FAT alone. This confirms

a diferent action path without attributing the terminal gain to a single operation.

For the changed distillation trajectory, three evaluation items are correct only under the joint policy and one only under FRT, giving the net gain of two. Joint minus FRT Exact across the four tasks is (−1, 0, +1, +2). The descriptive four-policy Exact contrast is

$$
I _ { \mathrm { L w F , c h a n g e d } } = 9 4 - 9 2 - 7 7 + 7 5 = 0 ,\tag{7.1}
$$

while the macro-NLL contrast is −0.020493. Changed-trajectory replay gives +1 on Exact and −0.042830 on NLL (computed from the unrounded arm-level NLL values; diferencing the displayed six-decimal values gives −0.042831). Joint advantage does not require positive superadditivity, as (5.30) shows. These full-policy contrasts also do not replace fixed-parent, fixed-action causal comparisons.

The reported endpoints follow the fixed evaluation rule. Retained predictions, terminal summaries and action records support the numerical comparisons; Appendix C.5 states the verification scope.

## 7.4 Current cross-responses and counterexamples

The current-response checks complement the endpoint comparisons. Protected FRT writes change measured acquisition-gradient responses in the replay realization. Under distillation, a separate diagnostic policy supplies five matched-action finite probes of the certified acquisition opportunity before and after regulation. Their changes include positive, zero and negative values; Appendix C.3 defines the measured CE envelope, normalization, and comparison direction. Acquisition paths also change later functional frames, repair eligibility, and allocation. The direct same-parent evidence is stronger for the FRT-to-FAT channel; reverse efects are represented by the exac residual identity and the observed later paths.

Several controls limit the policy interpretation. A standard-replay FRT combined with a strong FAT rule need not beat FAT alone. Removing the joint LwF rule’s one-event cooldown leaves its Exact vector unchanged but worsens NLL. These results distinguish admissibility from continuation value and show why coordination requires more than executing every available action. Appendix C retains the corresponding controls and their scientific interpretation.

The language-model runs demonstrate the execution of protected repairs and structural renewals, with higher teacher-forced Exact for the coordinated policy than for either separate policy in all four comparisons. Because the trajectories share a previously examined panel and the policies difer in auxiliary work, these results do not establish wider generalization or compute-matched superiority.

## 8 Discussion and conclusions

## 8.1 Repair, acquisition, and learning response

FRT and FAT distinguish present function from the learning permitted by its realization. A repair can meet its target while leaving a family of learning responses available (Theorem 3.3). A neutral installation can leave the current function unchanged while altering the next learning response (Theorem 4.2). These results identify diferent ways of acting on hidden structure without requiring regulation and acquisition to preserve the same observation.

The coupling theory concerns how the two action families constrain each other. A repair obligation can force a gradient-channel change (Theorem 5.2). New features contribute to protected repair only through the directions that remain after compensation (Theorem 5.3). Their combined responses must fit the joint constraints and budget (Theorem 5.4). The supervised construction in Theorem 5.5 demonstrates that this interaction can produce a favorable learning response absent from either action alone. The coupling is a property of the shared learning-state dynamics, not simply the coexistence of two training rules.

## 8.2 Relation to earlier work

The finite head repair and continuation-conditioned interaction were established in Wang [2]. The present theory extends those results to repair-compatible learning responses, acquired repair capacity, and jointly constrained dynamical efects. Standard pseudoinverse, singular-value, and convex-duality methods provide the mathematical instruments; the contribution is the coupled regulation–acquisition structure obtained from them. The experiments assess realizations of this structure, rather than a benchmark-wide ranking of every protected-update or low-rank method.

At the level of complete successor laws, a method that reproduces the same state-dependent outcomes and costs from the same available information has the same continuation value. The distinction between regulation and acquisition therefore rests on their specified action interfaces and efects on learning, not on labels alone.

## 8.3 Limits and further questions

The afine repair and compensation results are exact on their stated action interfaces. Nonlinear protected lifting requires additional regularity and writable directions, as shown in Appendix A.4. A finite response dimension need not define a suficient autonomous learning state. Nor does a finite current-frame certificate protect all future inputs, optimizer memory, or later tasks. Establishing useful low-dimensional closed dynamics and finite nonlinear repair domains remains a separate problem.

The suficient conditions for positive continuation value also require more than current admissibility. Their future-value covectors or lower bounds may be unavailable or costly to estimate. The experiments instead evaluate concrete policies based on arrived information. Their joint Exact advantage is observed in two Qwen2.5-32B trajectories difering in seed and within-task example order on one reused panel; the task-group order remains fixed. New evaluation items, separate seed and order interventions, other model families, and matched end-to-end budgets would test wider empirical scope.

## 8.4 Conclusion

Regulation and acquisition act on the same learning state but answer diferent questions: how a specified function can be repaired without violating current protection, and how trainable structure can be formed or renewed for subsequent updates. This paper characterizes how those answers change when the operations are combined. It identifies learning-response freedom remaining after protected repair, conditions under which repair must alter a learning channel, and the compensation-adjusted directions through which acquisition changes protected repai reachability and minimum coeficient cost. Joint-response conditions and a supervised cross-entropy construction show that the interaction can be beneficial even when neither intervention alone changes the target outcome.

In Qwen2.5-32B, protected head repair and fixed-capacity LoRA renewal are executable under replay and output distillation. The coordinated policies achieve higher teacher-forced Exact than either separate policy in all four trajectory–learning-rule comparisons. In these policies, available state readouts and arrived evidence guide the selection of admissible repairs and renewals; ordinary learning then resumes from the modified state. Hidden learning structure is therefore both a source of variation in future learning responses and, within specified information and action constraints, a target of intervention.

Data and computational materials. The experiments use a pinned Qwen2.5-32B model and MASSIVE-derived data. The two trajectories reuse the same training and evaluation items. Compact execution records and the original complete-state checkpoints were retained for ofline checks, but are not included in the manuscript source package. Appendix B states the execution protocol and the scope of numerical verification.

## References

[1] Qinyou Wang. Fiber fingerprints of hidden learning-state dynamics, 2026. arXiv:2608.15976, version 1.

[2] Qinyou Wang. Intrinsic–extrinsic coupling in learning dynamics, 2026. arXiv:2609.30185, version 1.

[3] Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for continual learning. In Proceedings ofthe Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings ofMachine Learning Research, pages 3762–3773, 2020. URL https://proceedings.mlr.press/v108/farajtabar20a.html.

[4] Akshat Gupta, Dev Sajnani, and Gopala Anumanchipalli. A unified framework for model editing. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15403–15418, 2024. doi: 10.18653/v1/2024.findings-emnlp.903.

[5] Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, and Tat-Seng Chua. AlphaEdit: Null-space constrained knowledge editing for language models. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/29c8c615b3187ee995029284702d3f43-Abstract-Conference.html.

[6] Tianqi Chen, Ian Goodfellow, and Jonathon Shlens. Net2Net: Accelerating learning via knowledge transfer. In Internationa Conference on Learning Representations, 2016. arXiv:1511.05641.

[7] Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, and Richard S. Sutton. Loss of plasticity in deep continual learning. Nature, 632:768–774, 2024. doi: 10.1038/s41586-024-07711-7.

[8] Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. AdaLoRA: Adaptive budget allocation for parameter-eficient fine-tuning. In International Conference on Learning Representations, 2023. arXiv:2303.10512.

[9] Shaowen Wang, Linxi Yu, and Jian Li. LoRA-GA: Low-rank adaptation with gradient approximation. In Advances in Neural Information Processing Systems, volume 37, pages 54905–54931, 2024. doi: 10.52202/079017-1741.

[10] Cheng Chen, Pengpeng Zeng, Yuyu Guo, Lianli Gao, Heng Tao Shen, and Jingkuan Song. JANUS-LoRA: A balanced low-rank adaptation for continual learning. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 18026–18040, 2026. URL https://proceedings.mlr.press/v306/chen26fz.html.

[11] Valérie Castin, Kimia Nadjahi, Pierre Ablin, and Gabriel Peyré. Balanced LoRA: Removing parameter invariance to accelerate convergence. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 12024–12045, 2026. URL https://proceedings.mlr.press/v306/castin26a.html.

[12] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://proceedings.neurips.cc/paper/2018/hash 5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html.

[13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. arXiv:2106.09685.

[14] Vladislav Lialin, Sherin Muckatira, Namrata Shivagunde, and Anna Rumshisky. ReLoRA: High-rank training through low-rank updates. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper 2024/hash/d6fdc308a3b0f017f16059e5888f3346-Abstract-Conference.html.

[15] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004. URL https://web.stanford. edu/\~boyd/cvxbook/.

[16] Qwen et al. Qwen2.5 technical report, 2025. arXiv:2412.15115, version 2.

[17] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. arXiv:1412.6980.

[18] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. arXiv:1711.05101.

[19] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P. Lillicrap, and Greg Wayne. Experience replay for continual learning. In Advances in Neural Information Processing Systems, volume 32, 2019. arXiv:1811.11682.

[20] Zhizhong Li and Derek Hoiem. Learning without forgetting. In Computer Vision – ECCV 2016, volume 9908 of Lecture Notes in Computer Science, pages 614–629. Springer, 2016. doi: 10.1007/978-3-319-46493-0\_37.

[21] Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4277–4302. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.235.

## A Additional mathematical details and boundary examples

## A.1 What an observation must retain for a joint decision

Fix a finite candidate universe $\mathcal { P }$ whose labels have a common decoder, and let $\mathcal { H } ( s ) \subseteq \mathcal { P }$ be the exact current admissibility relation. There exists a map � such that ${ \mathcal { H } } = k \circ O$ if and only if (5.5) holds. Necessity follows because a map has one value for each observation. For suficiency, assign to an observation the relation of any state in its fiber; the hypothesis makes that assignment independent of the representative.

If exact decoding fails, a selector that sees only $o = O ( s )$ and must be valid for every compatible state has to choose in

$$
{ \mathcal { M } } _ { \mathrm { r o b u s t } } ( o ) = \bigcap _ { s : O ( s ) = o } { \mathcal { H } } ( s ) .\tag{A.1}
$$

This intersection is both necessary and suficient for such a pointwise robust choice. In a finite observation domain, choosing one available label poses no measurable-selection issue; an infinite stochastic decision problem requires separate measurability assumptions. If HOLD lies in every relation, robust feasibility may be trivial, whereas robust progress still requires a non-HOLD element. This is an information condition, not a performance guarantee.

For a linear hidden coordinate �, observed as ��, a desired response $L x$ is exactly recoverable by a linear decoder precisely when ker $M \subseteq \ker L$ . The proof is the factorization argument in Theorem 4.6. If new scalar linear measurements are allowed, the smallest number needed is

$$
k _ { \mathrm { m i n } } = \mathrm { r a n k } \left[ \begin{array} { c } { M } \\ { L } \end{array} \right] - \mathrm { r a n k } \ : M .\tag{A.2}
$$

Indeed, each measurement can add at most one row-space direction, and a basis of the missing row directions attains this count. This count concerns arbitrary available linear measurements; a restricted physical sensor may not be able to obtain those rows at all.

## A.2 Source-restricted and weighted action coordinates

The Euclidean statements in the main text assume that the coordinates really parameterize the allowed linear action space. If physical actions have the form $x = E z$ with injective � and physical squared size $x ^ { \top } H x , H \succ 0 ,$ then the coordinate metric is $W = E ^ { \top } H E \succ 0$ . Whitening $\tilde { z } = W ^ { 1 / 2 } z$ converts the constraints to $M E W ^ { - 1 / 2 } \tilde { z } = d$ and the response to $K E W ^ { - 1 / 2 } \tilde { z }$ . Theorems 3.2 and 5.4 then apply to this actual image and metric. Using the identity metric without this transformation would change the minimum-cost problem.

A finite canonical candidate set is diferent from a full linear image. An ellipsoid containing all representable responses can supply an impossibility result when a target lies outside it, but membership does not prove that the canonical rule emits a target-reaching candidate. Likewise, two separately budgeted writes need not obey a combined budget:

$$
\| X _ { F } + X _ { A } \| _ { F } ^ { 2 } = \| X _ { F } \| _ { F } ^ { 2 } + 2 \left. X _ { F } , X _ { A } \right. + \| X _ { A } \| _ { F } ^ { 2 } .\tag{A.3}
$$

The cross term cannot be removed merely by labeling the actions regulation and acquisition.

## A.3 A consumer-coordinate example

Equal feature kernels do not ensure identical updates under coordinatewise adaptive optimization. Consider one scalar feature $q$ represented either by one coeficient or by two coeficients with duplicated features $( q / \sqrt { 2 } , q / \sqrt { 2 } )$ At zero coeficients and zero moments, both realizations have the same function and feature kernel. Let $g \neq 0$ be the gradient of the single coeficient, and suppose both Adam realizations have the same pre-update counter �. Define

$$
a _ { n } = \frac { 1 - \beta _ { 1 } } { 1 - \beta _ { 1 } ^ { n + 1 } } , \qquad b _ { n } = \sqrt { \frac { 1 - \beta _ { 2 } } { 1 - \beta _ { 2 } ^ { n + 1 } } } .\tag{A.4}
$$

The one-coordinate update is $- \eta a _ { n } g / ( b _ { n } | g | + \epsilon )$ . Each duplicated-coordinate gradient is $g / \sqrt { 2 }$ , so summing their functional contributions gives the ratio

$$
{ \frac { \mathrm { d } \mathrm { u p l i c a t e d ~ f u n c t i o n a l ~ c h a n g e } } { \mathrm { s i n g l e ~ f u n c t i o n a l ~ c h a n g e } } } = { \frac { b _ { n } | g | + \epsilon } { b _ { n } | g | / { \sqrt { 2 } } + \epsilon } } \longrightarrow \sqrt { 2 } \quad ( \epsilon  0 ) .\tag{A.5}
$$

Plain SGD with matched scalar learning rate instead gives equal functional changes in these two realizations. The example assumes zero moments and an explicit common counter; it does not rotate arbitrary nonzero diagonal Adam second moments as if they were a full covariance. Its purpose is to separate feature-space novelty from optimizer-conditioned learning response, not to claim that duplication improves quality. Parameterization dependence is also an active object of low-rank optimization research [11].

## A.4 A local nonlinear bridge, not a global finite guarantee

The compensation theorem is exact for afine action responses. A standard implicit-function argument identifies the additional condition needed to retain its local meaning in a nonlinear interface.

Proposition A.1 (Local protected lifting). Let $F ( u , \nu ) \in \mathbb { R } ^ { m }$ be a $C ^ { 2 }$ protected observation and $G ( u , \nu ) \in \mathbb { R } ^ { p }$ a $C ^ { 1 }$ obligation readout near (0, 0), where $u \in \mathbb { R } ^ { d }$ are old writable coordinates and $\boldsymbol \nu \in \mathbb { R } ^ { k }$ are newly available coordinates. Put $C = D _ { u } F ( 0 , 0 ) , B = D _ { \nu } F ( 0 , 0 ) , H = D _ { u } G ( 0 , 0 )$ , and $\Psi = D _ { \nu } { \cal G } ( 0 , 0 )$ . If� hasfull row rank, choose an orthonormal basis $N _ { C }$ ofker �. There is a local $C ^ { 2 }$ parameterization

$$
u ( \xi , \nu ) = N _ { C } \xi + C ^ { \dagger } \eta ( \xi , \nu ) , \qquad F ( u ( \xi , \nu ) , \nu ) = F ( 0 , 0 ) ,\tag{A.6}
$$

with $\eta ( 0 , 0 ) = 0 , D _ { \xi } \eta ( 0 , 0 ) = 0$ , and $D _ { \nu } \eta ( 0 , 0 ) = - B .$ . The derivative of its obligation response is

$$
D _ { ( \xi , \nu ) } G \big ( u ( \xi , \nu ) , \nu \big ) \Big | _ { 0 } = \big [ H N _ { C } \quad \Psi - H C ^ { \dagger } B \big ] .\tag{A.7}
$$

Proof. Apply the implicit function theorem to $\widetilde { F } ( \eta , \xi , \nu ) = F ( N _ { C } \xi + C ^ { \dagger } \eta , \nu ) - F ( 0 , 0 )$ . Its derivative in � is $C C ^ { \dagger } = I _ { m }$ . Diferentiating the resulting identity gives the stated derivatives of �, since $C N _ { C } = 0$ . The chain rule then yields (A.7). □

The compensation-adjusted channel reappears, but only locally and under surjectivity of the old writable protection derivative. The proposition does not prove that a source-bound discrete actuator follows this local parameterization, that an arbitrary finite target lies in its range, or that its path stays inside a fixed resource budget. A singular derivative requires a diferent constant-rank or higher-order analysis; it cannot be rescued by merely substituting a pseudoinverse into a nonlinear update.

## A.5 Intervention value need not be inferable from current protection

Take two future environments $\omega \in \{ - 1 , 1 \}$ that agree on all currently available information, and a feasible action $a \in [ - 1 , 1 ]$ with continuation value $V _ { \omega } ( a ) = \omega a - a ^ { 2 }$ . Any nonzero deterministic action has negative value in at least one environment. For a randomized current action, averaging the two environmental expected values gives $- \mathbb { E } [ a ^ { 2 } ] \leq 0$ . Therefore no rule using only the shared past can guarantee strictly positive value in both worlds without an additional assumption connecting that past to the future.

This elementary counterexample does not invalidate useful structure-aware policies in a specified learning environment. It identifies the extra role of the task family, consumer, and continuation conditions. Theorems about admissible directions and efective repair capacity do not need such foresight assumptions; unconditional policy-improvement claims do.

## B Execution and evaluation specification

## B.1 Model, data, and ordinary training

Table 2. Execution settings of the tested model and learning rules.
<table><tr><td>Object</td><td>Specification</td></tr><tr><td>Model</td><td>Qwen2.5-32B base; 64 layers; hidden size 5120; vocabulary 152064; BF16 backbone</td></tr><tr><td>Revision</td><td>1818d35814b8319459f4bd55ed1ac8709630f003</td></tr><tr><td>Adapter</td><td>FP32 LoRA, rank 8, alpha 16, dropout 0, both factors trainable; q/v/o/down projections in all 64 layers; 30,408,704 trainable adapter parameters</td></tr><tr><td>Renewal location</td><td>Zero-based layer 32, self-attention o_proj; eight physical factor slots of width 5120</td></tr><tr><td>Optimizer</td><td>AdamW, learning rate  $1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , { \epsilon } = 1 0 ^ { - 8 }$  , weight decay 0, clipping norm 1, constant scheduler; foreach/fused disabled</td></tr><tr><td>Recorded software</td><td>PyTorch 2.10.0, CUDA wheel 12.8, Transformers 4.57.6, PEFT 0.18.1, Accelerate 1.12.0, NumPy 2.3.5; math SDPA, nonreentrant checkpointing,</td></tr><tr><td>Training trajectories</td><td>TF32 disabled Original seed 33093001; changed seed 33102020 and within-task example permutation; four tasks, 16 intents, 1,024 training and 128 development examples; maximum length 128 tokens</td></tr><tr><td>Updates</td><td>Four current examples per ordinary update; 64 updates per task, 256 total; gradients accumulated over single-example microbatches</td></tr><tr><td>Replay</td><td>After task 0, 16 resident old examples per update; current/replay loss weights  $4 / 5$  and 1/5; 3,072 historical presentations total</td></tr><tr><td>Resident memory</td><td>Per-intent keyed bottom-k reservoir, k = 8, capacity 128; only arrived current examples admitted</td></tr><tr><td>FRT</td><td>Consultation after every 32 ordinary updates; current protection from last four current examples; historical representatives from strictly older intents;</td></tr><tr><td>FRT constants</td><td>at most 32 token rows per frame Margin threshold 1; buffer  $2 ^ { - 1 0 }$  ; current and historical residual tolerances  $2 ^ { - 1 2 } ;$  current CE tolerance  $2 ^ { - 1 1 }$  ; increment-norm cap 1; relative rank</td></tr><tr><td>FAT events</td><td>tolerance  $1 0 ^ { - 1 0 }$  After updates 65, 129, and 193; the first ordinary update of the new task has already occurred</td></tr><tr><td>Distillation</td><td>No ordinary historical replay; T = 2, λ = 1, loss CE+4KL(q2lp2); teacher captured after boundary maintenance at updates 64, 128, 192</td></tr><tr><td>Evaluation</td><td>Teacher-forced answer-plus-EOS Exact and macro-intent NLL; fixed 128- item development panel; evaluation is not an online policy input</td></tr></table>

The task groups, in order, are  
alarm\_query, alarm\_set, audio\_volume\_mute, audio\_volume\_up;  
calendar\_query, calendar\_remove, calendar\_set, cooking\_recipe;  
datetime\_query, email\_query, email\_querycontact, email\_sendemail;  
general\_joke, general\_quirky, iot\_cleaning, iot\_coffee.

The learner seed initializes Python, NumPy and both CPU/CUDA PyTorch generators before model loading and adapter creation. The pinned PEFT default initialization is retained (Kaiming-uniform input factors and zero output factors); with bias=none and use\_rslora=False. The adapter tensors are then stored in

FP32. The tested device is a single NVIDIA RTX PRO 6000 Blackwell; deterministic algorithms and cuDNN determinism are enabled, benchmarking and reduced-precision BF16 matmul reduction are disabled, and CUBLAS\_WORKSPACE\_CONFIG=:4096:8. These details specify numerical execution rather than an additional intervention.

Each intent has 64 training examples and eight development examples, selected from MASSIVE 1.1 en-US. The exact prompt (including newlines) is

Identify the intent of the following utterance.

Utterance: {utterance}

Intent:

followed by one newline. The target string is the intent identifier with underscores replaced by spaces, followed by the tokenizer EOS. For example, email\_querycontact maps to email querycontact; the mapping is a literal string substitution, not a semantic relabeling. Answer-token positions and token IDs are produced by the pinned tokenizer and verified against the original input binding.

The runs used fixed sample bindings and training tapes for the 1,024 training and 128 evaluation records, including source UIDs, split assignments, utterances, labels, the four current UIDs at each ordinary update, and native replay draws. Fixed memory-priority and replay-priority orders determined the pseudorandom choices in each trajectory. These run-specific records were retained for ofline checks and are not part of the manuscript source. The frame and allocation rules are described below. Only arrived examples may be consulted online. The four task-group identities and their order remain fixed. In the second trajectory only the within-task example permutation, memory sampling, and seed-dependent random draws change; no new evaluation examples are introduced.

For development item � with $L _ { u }$ supervised positions,

$$
\operatorname { E x a c t } ( u ) = \mathbf { 1 } \{ \arg \operatorname* { m a x } _ { \nu } z _ { u , k , \nu } = y _ { u , k } { \mathrm { ~ f o r ~ e v e r y ~ } } k \} ,\tag{B.1}
$$

$$
\ell _ { u } = - L _ { u } ^ { - 1 } \sum _ { k = 1 } ^ { L _ { u } } \log p ( y _ { u , k } \mid x _ { u } , y _ { u , < k } ) ,\tag{B.2}
$$

$$
\mathrm { \mathop { M a c r o N L L } } = { \frac { 1 } { 1 6 } } \sum _ { g = 1 } ^ { 1 6 } { \frac { 1 } { 8 } } \sum _ { u : \mathrm { i n t e n t } ( u ) = g } \ell _ { u } .\tag{B.3}
$$

EOS enters both learning loss and Exact, but is excluded from the selected FRT repair frames. Preservation of those frames therefore cannot be equated with preservation of every Exact decision. The initial experimental comparison designated NLL as primary and Exact as secondary; the present Exact-first display is a retrospective presentation choice.

## B.2 Ordering, state, and transactions

Before an ordinary update, native replay is drawn from the resident pool of strictly older tasks, then the current arm may redirect one item. The update completes all weighted backwards, clips once, updates optimizer and scheduler, and clears gradients. Its four current examples are then admitted to memory, before maintenance; replay examples are not readmitted. A scheduled FRT consultation then stages the canonical repair agains fixed current and historical frames. The stored candidate is checked and either committed or rolled back. A replay signal is published only after its associated repair passes; it can afect subsequent replay positions, not the completed update. At a teacher boundary the resulting student, including its committed bank, is copied into the frozen teacher.

At an acquisition event, candidates are compared from a retained common parent. Temporary factor changes, gradients, optimizer state and RNG are restored before another candidate is evaluated. Only the selected physical action persists. Recorded rejection is not counted as a committed birth. The no-write branch may still advance bookkeeping or incur diagnostic cost according to its declared contract.

Numerical regulation checks. The bank allocates 256 rows and eight commit slots; the run additionally checks the schedule-implied bound of 192 rows and six accepted commits. The compact head is accumulated: $W _ { \mathrm { e f f } } ^ { + } = W _ { \mathrm { e f f } } ^ { \mathrm { p a r e n t } } + X ^ { \top }$ , not overwritten. The current and historical residuals are maximum absolute entries, not Frobenius residuals. Separately for � and $N \ : = \ : H P _ { C }$ , the prescribed full-row-rank screen requires $\sigma _ { \operatorname* { m i n } } > 1 0 ^ { - 1 0 } \sigma _ { \operatorname* { m a x } } > 0$ and enough columns. A QR/Gram solve in FP64 produces the FP32 stored factors; there is no ridge, pseudoinverse fallback, or cap rescaling. Outward-rounded arithmetic certifies the stored afine residuals and increment norm, and a stacked-frame inverse-residual test certifies the ideal map. Native acceptance also requires identical current argmaxes, historical margins at least one, finite logits, and current FP32 maximum absolute per-row and absolute mean CE changes at most $2 ^ { - 1 1 }$ . A second append from the restored bank must reproduce features and logits bitwise. Missing old tasks or zero deficits cause no write; unqualified rank, norm, or finite residuals reject the candidate. Failed restoration or a failed implementation check stops execution rather than authorizing a diferent action.

The FRT implementation’s backward-equivalence certificate compares a compact written model with its reference implementation at the same written state. It does not state that the learner’s pre-write and post-write gradients agree. That latter equality would generally contradict (3.8). The certificate probes each distinct stored-label contrast $e _ { y } - e _ { o } ,$ the common nonlabel component $e _ { o }$ , and zero, where � is the smallest vocabulary index outside the stored labels. Both the new increment and cumulative bank are checked on the first two historical feature rows. For each probe, its input-gradient error must be at most $1 0 ^ { - 4 } \| g _ { \mathrm { r e f } } \| _ { 2 } + \gamma _ { k } \| a \| _ { 2 }$ , where � sums the absolute FP64 reference terms componentwise, $\gamma _ { k } = k u / ( 1 - k u ) , u = 2 ^ { - 2 4 }$ and $k = 2 r + 1 6$ for � stored rows. Zero reference gradients are permitted. This is a finite, implementation-specific consistency check, not a scientific selector for helpful gradients.

## B.3 Resident representatives and token frames

Each intent retains the first eight arrived training UIDs in its fixed memory-priority order. At a consultation, intents from strictly older tasks are visited in the task/intent order listed above; the first two resident UIDs in lexicographic order represent each intent. This is not selection by loss, margin, or numerical source-example ID. A missing required pair is a contract failure. No old-task representative is used in task zero.

For FRT, the current examples are the four just trained, in batch order. The historical examples are the representatives just defined. In each frame separately, visit the first non-EOS answer prediction position of every example, then the second of every example, and so on. Keep the first occurrence of each pair (complete causal input-token prefix through that prediction position, target token ID); stop at 32 rows. This round-robin and deduplication rule, not a global sort of all token positions, defines � and �. The resulting frame rows were recorded for both trajectories.

At FAT events after updates 65, 129, and 193, use the same four current examples and the same representative rule, but take only the final non-EOS answer prediction position of each example. There are four current gate rows and respectively 8, 16, and 24 historical gate rows; there is no additional round-robin deduplication in these FAT frames. Their layer-input activations form the historical protection rows $O _ { t }$ . The location is the module with sufix layers.32.self\_attn.o\_proj; its adapter matrices are $8 \times 5 1 2 0$ and $5 1 2 0 \times 8$ in FP32. All frames are recaptured at the reached parent, including its efective repair bank.

Proposal signals are consumer-specific. Replay computes the current example’s whole answer-plus-EOS mean CE, diferentiates it with respect to the selected module’s sequence output, and retains the derivative and input activation only at that example’s final non-EOS answer prediction position. The four rows are concatenated, with no additional factor $1 / 4$ in this proposal readout. Thus the derivative includes downstream causal efects of that position; it is not a single-token CE derivative or the ordinary mixed replay gradient.

Distillation instead retains all sequence positions of each current example, including prompt positions. It diferentiates the same answer-plus-EOS mean CE and the weighted 4KL separately, and divides each example’s module derivatives by four before concatenation. Prompt-position derivatives can be nonzero through causal attention. The frozen teacher is evaluated before student hooks are installed. Historical protection and finite gates still use the single-position rows above. Proposal-gradient calls do not advance the optimizer or accumulate parameter gradients, and restore training mode and random-generator states.

## B.4 Regulation-linked replay

Let � be the last accepted repair’s update number. For each historical intent represented in that repair frame, publish a Boolean $r _ { g } \colon$ it is true exactly when at least one of that intent’s pre-repair row margins was below one. The signal is active at updates $s < t \leq s + 3 2$ . A new accepted repair replaces the signal and resets its counter; rejection leaves the previous signal unchanged, subject to expiration.

At update �, let $N _ { t }$ be the ordered native 16-example replay batch and $M _ { t }$ the pre-admission resident old-task pool. A donor position is eligible exactly when its intent was represented in the signal, has $r _ { g } = 0 \mathrm { . }$ , and has a same-source-task recipient in $M _ { t } \setminus N _ { t }$ whose intent has $r _ { g } = 1$ . Choose the eligible position first in the fixed donor-priority order, then the eligible recipient first in that source task’s recipient-priority order. Each order is keyed by trajectory, publication update and current update. The exact priority orders were recorded in ofline ordinal tables. Ties in the original priorities are resolved by position for donors and lexicographic UID for recipients.

Replace at most one position. With no active signal, no legal exchange, no native replay, or 32 replacements already consumed by that signal, leave the native batch unchanged. “Unused” means absent from $N _ { t }$ , not absent from all earlier batches in the window. Baseline and acquisition-only replay always keep $N _ { t } ;$ ; regulation and joint use their own reached-state signals. This preserves 16 distinct replay UIDs, the source task at every position, and the total loss weight $1 / 5$ . The whole-run count depends on the accepted signal windows. The resulting replay-allocation paths for the changed trajectory were retained for ofline checks.

## B.5 Acquisition rules for the two consumers

At an acquisition event at ordinary update $t \in \{ 6 5 , 1 2 9 , 1 9 3 \}$ , both consumers enumerate at most four candidate physical slots from np.random.default\_rng(680016+t).permutation(8)[:4] (NumPy 2.3.5); the eight slot indices are zero-based. This candidate-order seed is distinct from each trajectory’s learner seed. Every trial restores its candidate-specific temporary state unless that trial is selected. For the three events this gives the ordered slot lists (5, 3, 4, 6), (7, 3, 5, 2), and (2, 5, 7, 0), respectively. Replay-only acquisition and joint replay use the identical renewal generator and first-passing selection rule; their reached parameters and replay histories can difer.

Replay direction and admission. Visit the slots in the stated order. For slot $j ,$ form a row-space basis of $[ O _ { t } ; K _ { t , j } ]$ by retaining singular values greater than max $( 1 0 ^ { - 8 } \sigma _ { 1 } , 1 0 ^ { - 1 4 } )$ . Put $R = X _ { t } ( I - P _ { t , j } )$ and use its thin SVD $R = U \Sigma V ^ { \top }$ . A leading right singular vector of $D _ { t } ^ { \top } U \Sigma$ , multiplied by $V ^ { \top }$ , gives the unit input direction in (6.4). Choose its sign so that its first maximum-absolute-value coordinate is nonnegative, then scale by the median of the seven retained input-row norms. Store the row in FP32 and set the selected output column to zero. Replay does not use the symmetric CE-amplitude probes of the distillation rule.

A proposal is unavailable if the largest singular value of � is at most $1 0 ^ { - 1 0 }$ , that of $D _ { t } ^ { \top } U \Sigma$ is at most $1 0 ^ { - 1 2 }$ , or the median input-row norm is at most $1 0 ^ { - 1 2 }$ . After FP32 storage, require current-response RMS at least $1 0 ^ { - 6 }$ and $\lVert D _ { t } ^ { \top } X _ { t } a _ { j } ^ { \mathrm { n e w } } \rVert \geq 1 0 ^ { - 1 2 }$ . The maximum absolute responses on $O _ { t }$ and the retained rows must each be at most

$$
1 0 ^ { - 5 } \operatorname* { m a x } \Bigl \{ 1 , \ : r _ { A } \operatorname* { m a x } \Bigl ( \operatorname* { m a x } _ { i } \| O _ { t , i } \| , \operatorname* { m a x } _ { i } \| K _ { t , j , i } \| \Bigr ) \Bigr \} ,
$$

where $r _ { A }$ is the median norm. Nonfinite inputs or an invalid/missing required frame do not trigger an alternative estimator.

For the merged four-current-plus-historical token frame, let $z ^ { 0 } , z ^ { 1 }$ be the parent and trial logits, $m _ { i } ^ { b } \ =$ $\begin{array} { r } { z _ { i , y _ { i } } ^ { b } - \operatorname* { m a x } _ { \nu \neq y _ { i } } z _ { i , \nu } ^ { b } , c _ { i } ^ { b } = \mathrm { C E } ( z _ { i } ^ { b } , \bar { y } _ { i } ) } \end{array}$ , and $h _ { i } ^ { b } = \mathbf { 1 } \{ \arg \operatorname* { m a x } _ { \nu } z _ { i , \nu } ^ { b } = y _ { i } \}$ . The finite damage gate is

$$
\begin{array} { r l r } & { } & { \displaystyle \operatorname* { m a x } _ { i , \nu } | z _ { i , \nu } ^ { 1 } - z _ { i , \nu } ^ { 0 } | \leq 1 , \qquad \displaystyle \operatorname* { m a x } _ { i } ( m _ { i } ^ { 0 } - m _ { i } ^ { 1 } ) _ { + } \leq 1 , } \\ & { } & { \displaystyle \operatorname* { m a x } _ { i } ( c _ { i } ^ { 1 } - c _ { i } ^ { 0 } ) \leq 0 . 2 5 , \quad \displaystyle \sum _ { i } h _ { i } ^ { 0 } ( 1 - h _ { i } ^ { 1 } ) \leq 1 . } \end{array}\tag{B.4}
$$

All logits must be finite. These are maxima and a count over the combined token rows, not over complete examples or separately allowed budgets for current and historical data. The finite gate computes margins, token CE and logit diferences in CPU FP32 from the stored logits. Argmax ties select the first vocabulary index. No strict CE decrease is required by replay renewal.

Accept the first available proposal passing (B.4); otherwise restore the parent and try the next slot, ending in HOLD if all fail. On commit, reset the selected row/column slices of the first and second moments (and the maximum second moment if present). Keep the optimizer’s scalar age and all seven other slots unchanged. Rank and parameter count remain fixed.

The distillation rule is a distinct, finite-capacity renewal with a current CE/KL admission test. After the first ordinary update of each new task, it captures four arrived training examples and the existing teacher. For the same four keyed candidate slots, it forms a projected current CE direction outside historical input and retained-factor spans; the slot’s original input-factor norm sets the new input-factor norm. It stages the proposed input row with the selected output column zero. Let $g _ { C } , g _ { K }$ be the actual staged-state gradients of the four-example mean CE and weighted teacher KL with respect to that output column. Project both outside the other seven output columns to obtain $u _ { C } , u _ { K }$ . The normalized CE-descent, first-order KL-neutral direction is

$$
w = - \frac { u _ { C } - ( { \langle u _ { C } , u _ { K } \rangle } / { \| u _ { K } \| ^ { 2 } } ) u _ { K } } { \| u _ { C } - ( { \langle u _ { C } , u _ { K } \rangle } / { \| u _ { K } \| ^ { 2 } } ) u _ { K } \| } ,\tag{B.5}
$$

when $\| u _ { K } \| ^ { 2 } > 1 0 ^ { - 2 4 }$ ; otherwise set the numerator to $u _ { C }$ . Reject if its norm is at most $1 0 ^ { - 1 2 }$ , or if the resulting CE directional derivative is nonnegative.

Finite output-column amplitude. Let $r _ { B } = \| b _ { i } ^ { \mathrm { o l d } } \| , \nu _ { j }$ be the current output-factor AdamW second-moment column, $\beta _ { 2 }$ its decay, � the optimizer counter, and �, � the optimizer’s current learning rate and stabilizer. Let $P _ { \perp }$ project outside the other output-factor columns. The initial symmetric probe radius is

$$
\begin{array} { r l } & { \widetilde { g } _ { C } = P _ { \perp } \left[ g _ { C } \oslash \left( \sqrt { \nu _ { j } / ( 1 - \beta _ { 2 } ^ { n } ) } + \epsilon \right) \right] , } \\ & { h _ { A } = \eta \left| \left. \widetilde { g } _ { C } , w \right. \right| , \qquad h = \sqrt { r _ { B } \operatorname* { m i n } ( r _ { B } , h _ { A } ) } . } \end{array}\tag{B.6}
$$

The method measures mean CE at both +ℎ� and −ℎ� around the staged zero-output state. If both coincide numerically with the zero-column CE, it updates $h  \sqrt { h r _ { B } }$ , for at most three symmetric probe pairs or until $h \geq r _ { B }$ . If all probes are unresolved, the candidate is rejected. The counter must be positive, the stored moments finite and nonnegative, and both $r _ { B }$ and $h _ { A }$ positive; otherwise no candidate amplitude is available. From the final pair let

$$
\begin{array} { r l } & { \ell = - \langle g _ { C } , w \rangle > 0 , \qquad k _ { h } = \frac { \mathrm { C E } ( + h w ) + \mathrm { C E } ( - h w ) - 2 \mathrm { C E } ( 0 ) } { h ^ { 2 } } , } \\ & { \alpha _ { B } = \frac { \ell } { \operatorname* { m a x } ( k _ { h } , 0 ) + \ell / h } , \qquad b _ { j } ^ { \mathrm { n e w } } = \mathrm { F P } 3 2 ( \alpha _ { B } w ) . } \end{array}\tag{B.7}
$$

Write $B _ { C } = - \langle g _ { C } , b _ { i } ^ { \mathrm { n e w } } \rangle > 0$ . The same arrived four-example frame and cached teacher logits define the parent-relative full finite loss changes. All required row/column bases use the singular-value threshold $\operatorname* { m a x } ( 1 0 ^ { - 8 } \sigma _ { 1 } , 1 0 ^ { - 1 4 } )$ ; a nonfinite or $\mathrm { a t { - } m o s t { - } } 1 0 ^ { - 1 2 }$ projected direction is unavailable. Singular-vector/eigenvector ties follow the specified FP64 linear-algebra backend; after normalization the first maximum-absolute coordinate fixes the sign. A candidate qualifies only if the repeated CE/LwF readouts agree exactly (FP32 per-example losses averaged in FP64, with no comparison tolerance), ΔCE $< 0 , \Delta ( \mathrm { 4 K L } ) \le B _ { C }$ , and the fixed finite historical/current damage gate passes (at most one previously correct evaluated row may turn incorrect, maximum absolute-logit change 1, margin decrease 1, and per-row CE increase 0.25). This finite test, not first-order orthogonality alone, decides acceptance.

Selecting a structural renewal. For acquisition alone, sort the qualifying physical slots by descending projected CE singular value, then ascending finite ΔCE, then the keyed proposal order. Exclude the slot most recently successfully renewed; select the first remaining candidate or HOLD. An intervening HOLD does not clear this one-event cooldown. For the joint rule, begin with the same qualified and noncooled candidates and prefer, in order: candidates that create no new historical FRT margin deficit; among them candidates whose total historica

FRT deficit does not increase; among that subset the largest observed finite current CE reduction, with the preceding ranking as tie-break. At any stage with an empty preferred subset, use the preceding stage’s choice; the final fallback is the original acquisition choice, otherwise HOLD. These are current-state preferences, not future-value estimates.

Selecting a regulation write under distillation. The joint FRT consultation compares the native certified repair with HOLD. Let $B _ { H } = { \overline { { \mathrm { C E } } } } _ { H } ( s ) - { \overline { { \mathrm { C E } } } } _ { H } ( f ( s ) )$ be its actual mean token-row historical CE gain on the qualified FRT frame (FP32 bank correction followed by FP64 CE reduction), and let $\Delta _ { C }$ and $\Delta _ { K }$ be the diferences of the four-example mean CE and raw temperature-two KL (repair minus HOLD) under the same teacher. The gate computes $\Delta _ { D } = \Delta _ { C } + 4 \Delta _ { K }$ from these separately reduced scalars; it does not subtract two already-rounded combined-loss scalars. Commit the native repair precisely when $B _ { H } \geq 0 , \Delta _ { C } \leq B _ { H }$ , and $\Delta _ { D } \leq B _ { H } ;$ otherwise HOLD. This is a declared 1:1 current-loss trade-of, not a forecast of terminal Exact. All compared candidates are evaluated from the same restored parent; only an accepted physical state change persists.

## B.6 Resources and information access

The distillation learner has no ordinary replay training, but FRT and FAT can still read resident historical representatives for current sensing and qualification. It is therefore not a memory-free full-system comparison. Teacher operations, finite candidate probes, frame acquisition, repair solves and storage are additional work. The last-stage joint ranking reuses already measured candidate readouts; this does not make the candidate generation or the whole policy free. No matched end-to-end FLOP or wall-clock superiority claim is made.

## B.7 Changed training trajectory and computational accounting

The second trajectory uses random seed 33102020, rather than the original 33093001, and a diferent permutation of the training examples within each of the four tasks. The ordered task groups remain unchanged. Both runs use the same 1,024 training examples and 128 previously inspected evaluation examples, with 64 ordinary updates per task. The reordering and seed jointly change the reached learning state; this design does not identify their separate efects. Each policy is evaluated using the same teacher-forced endpoint definition within its trajectory.

Table 3. Committed structural operations in the changed training trajectory. Repair consultations may decline to write. Replay redirections count individual examples substituted during subsequent batches.
<table><tr><td>Learning rule</td><td>Policy</td><td>FRT commits</td><td>FAT commits</td><td>Replay redirections</td></tr><tr><td>Replay</td><td>Baseline</td><td>0</td><td>0</td><td>0</td></tr><tr><td></td><td>Regulation</td><td>3</td><td>0</td><td>95</td></tr><tr><td></td><td>Acquisition</td><td>0</td><td>3</td><td>0</td></tr><tr><td></td><td>Joint</td><td>3</td><td>3</td><td>96</td></tr><tr><td>Distillation</td><td>Baseline</td><td>0</td><td>0</td><td>0</td></tr><tr><td></td><td>Regulation</td><td>5</td><td>0</td><td>0</td></tr><tr><td></td><td>Acquisition</td><td>0</td><td>3</td><td>0</td></tr><tr><td></td><td>Joint</td><td>5</td><td>3</td><td>0</td></tr></table>

In replay, each accepted repair can publish a 32-update allocation signal; at most one item is redirected per subsequent update. The recorded regulation-only run has 32, 32, and 31 replacements in its three active windows, and the joint run has 32 in each. These allocations were recomputed from the retained controller records. Compact records allow replay-policy and numerical endpoint reconstruction; this does not constitute independent full-tensor re-execution of each model update.

Here one Teacher forward means one teacher-model evaluation of one utterance example (including its supervised answer-token positions). The common count $7 7 1 = 3 \times 6 4 \times 4 + 3$ comprises 768 ordinary teacher calls during tasks 1–3 and one snapshot-consistency check at each of three task boundaries. Acquisition adds four such calls at each of three structure events (12), and the joint regulation-selection diagnostic adds four calls at each of five qualified repair consultations (20); thus the acquisition and joint totals are 783 and 803.

Table 4. Teacher forward calls in distillation. Both training trajectories have the same count for each policy. Extra calls are relative to the corresponding baseline.
<table><tr><td>Policy</td><td>Teacher forwards per trajectory</td><td>Extra calls</td></tr><tr><td>Baseline</td><td>771</td><td>0</td></tr><tr><td>Regulation</td><td>771</td><td>0</td></tr><tr><td>Acquisition</td><td>783</td><td>12</td></tr><tr><td>Joint</td><td>803</td><td>32</td></tr></table>

Teacher forward count does not exhaust extra FLOPs, memory use, or wall-clock costs. Candidate diagnostics, finite CE/KL checks, regulation solves, and state inspection also consume resources. The policies match ordinary-update counts but are not claimed to match full computational expenditure.

## C Additional empirical controls and scope

## C.1 Comparability of the main policies

The replay baseline in Table 1 contains neither structural repair nor acquisition. The regulation-only policy uses both a qualified repair and regulation-linked replay allocation, whereas the acquisition-only policy retains ordinary replay allocation. The successful coordinated replay policy also uses regulation-linked allocation. Thus the policy table compares complete realizations; it is not a fixed-parent factorial isolation of structural repair from replay-content allocation. The distillation policies likewise use separate, fully executed acquisition and joint selection rules.

## C.2 Coordination alternatives

The following original-seed controls change a stated component of an otherwise fixed policy. In replay, keep the direct renewal, canonical repairs and training tape, but disable all regulation-linked redirections. This ordinary-allocation joint policy reaches 112/128 Exact, versus 114/128 for acquisition alone. The successful join row in Table 1 instead includes the allocation rule in Appendix B.4. This control distinguishes those complete coordination rules; it does not hold their later learning states fixed.

For distillation, retain the same renewal candidates, finite gates and joint preferences but remove the exclusion of the most recently renewed slot. This no-cooldown variant preserves all 128 binary Exact decisions and the count 94/128, but changes macro NLL from 0.440946 to 0.465207. Thus an unchanged Exact vector need not mean an unchanged predictive distribution. These are development controls on the same reused panel, not additional independent training settings.

## C.3 Finite cross-responses and learning realization

The five probes below come from a separate diagnostic policy on the original seed and training tape, not from the joint-policy row of Table 1. Its ordinary learner, teacher boundaries, frames and FAT-only renewal rule are those of Appendix B. It retains the one-event slot cooldown. Its FRT rule first requires the native repair certificate and the current-loss trade-of in Appendix B, then adds the opportunity test described below. This recurrence identifies the reached parents; equal step numbers alone do not.

A fixed physical proposal from the pre-repair state. At a qualified FRT time �, freeze the complete pre-write state, including its bank, factors, moments, stream, random state, prior-slot record and teacher. Use the same current four examples and historical representatives as the acquisition frame in Appendix B.3. Teacher snapshots are at steps 64, 128, 128, 192 and 192 for the five rows, respectively. Enumerate the first four slots from the same keyed permutation as for renewal, excluding the previously accepted FAT slot. Capture the full-token parent gradients $D _ { C } , D _ { K }$ , input rows �, and historical rows � before appending the FRT bank.

For each available slot, use (6.5) and the subsequent projection outside � and all eight parent input rows to obtain a unit direction �. Put $r _ { A } = \| a _ { j } \| , a _ { j } ^ { \prime } = \mathrm { F P } 3 2 ( r _ { A } \nu )$ , and form the parent-state output gradients

$$
g _ { C } = \gamma r _ { A } D _ { C } ^ { \top } X \nu , \qquad g _ { K } = \gamma r _ { A } D _ { K } ^ { \top } X \nu .\tag{C.1}
$$

Project outside the other seven output columns and use (B.5) to form �. The moment radius ℎ is exactly (B.6), with these parent gradients and moments. Set $b _ { i } ^ { \prime } = \mathrm { F P } 3 2 ( h w )$ and $B _ { j } = - \langle g _ { C } , b _ { i } ^ { \prime } \rangle$ . Reject an unavailable, nonfinite or nonpositive-radius proposal, or $B _ { j } \leq \breve { 0 }$ . Unlike the scheduled LwF renewal, this diagnostic neither recomputes the gradient at a staged zero column nor performs the symmetric amplitude probes. Among the remaining proposals, retain only the largest $B _ { j }$ , resolving ties by enumeration order; the set is empty if none remains.

Freeze that pair $( a _ { j } ^ { \prime } , b _ { j } ^ { \prime } )$ and its parent budget $B _ { j }$ . Evaluate the identical physical write both before and after the bank-only FRT action. In each branch require repeatable finite losses, parent-relative $\Delta \mathrm { C E } < 0 , \Delta ( 4 \mathrm { K L } ) \leq B _ { j }$ and the combined token-row gate (B.4). No proposal is regenerated after FRT. All trial factors and transients are restored; no shadow candidate, optimizer step, or moment reset persists. Within each pair only the regulation bank difers, while teacher and all arrived examples are held fixed.

For the four currently arrived examples $C _ { t } .$ , let $L _ { C } ( s )$ be the mean of their per-example mean answer-plus-EOS token cross-entropies (nats per example). With the same fixed physical proposal set $\mathcal { P } _ { t }$ and its state-dependent certified subset $\mathcal { P } _ { t } ^ { \mathrm { c e r t } } ( s )$ , define

$$
\begin{array} { r l } & { \boldsymbol { Q } _ { \boldsymbol { C } } ( s ; \mathcal { P } _ { t } ) = L _ { \boldsymbol { C } } ( s ) - \operatorname* { m i n } \bigl ( \{ L _ { \boldsymbol { C } } ( s ) \} \cup \{ L _ { \boldsymbol { C } } ( A _ { a } ( s ) ) : a \in \mathcal { P } _ { t } ^ { \mathrm { c e r t } } ( s ) \} \bigr ) , } \\ & { \quad \quad \Delta _ { \boldsymbol { F } } \boldsymbol { Q } _ { \boldsymbol { C } } = \boldsymbol { Q } _ { \boldsymbol { C } } ( f ( s ) ; \mathcal { P } _ { t } ) - \boldsymbol { Q } _ { \boldsymbol { C } } ( s ; \mathcal { P } _ { t } ) . } \end{array}\tag{C.2}
$$

The baseline HOLD is always included. Hence $Q _ { C } = 0$ when the candidate fails the native finite qualification; the certified subsets may difer before and after FRT even though the physical proposals do not. Positive $\Delta _ { F } Q _ { C }$ means regulation makes an immediate certified acquisition opportunity larger; negative means smaller.

Table 5. Matched-physical-proposal finite acquisition opportunity (nats per current example). All entries use arrived training samples and the same teacher within each pair.
<table><tr><td>Ordinary update</td><td>Before FRT</td><td>After FRT</td><td> $\Delta _ { F } Q _ { C }$ </td></tr><tr><td>96</td><td>0.016320877</td><td>0.016660973</td><td>+0.000340097</td></tr><tr><td>160</td><td>0.009440793</td><td>0.010511544</td><td>+0.001070751</td></tr><tr><td>192</td><td>0</td><td>0</td><td>0</td></tr><tr><td>224</td><td>0.014273338</td><td>0.014275268</td><td>+0.000001930</td></tr><tr><td>256</td><td>0.004792862</td><td>0.004691567</td><td>-0.000101294</td></tr></table>

At update 192 the shortlisted proposal is uncertified in both branches; the displayed zeros do not show absence of a raw gradient. The tiny positive diference at update 224 is a finite recorded change, not by itself a stable efect. These paired finite checks establish response dependence at the observed states, without identifying an unmeasured future policy value. Regulation also changes current acquisition-gradient responses in the replay realization; later decision rules can use such readouts to select diferent physical slots.

How these parent states are reached. After the probe, let $\Delta _ { D } ^ { \mathrm { p r o b e } }$ be the diference of the recorded four-example mean LwF loss scalars, after minus before FRT. Retain the native write only when $B _ { H } \geq 0 , \Delta _ { C } - \Delta _ { F } Q _ { C } \leq B _ { H }$ and $\Delta _ { D } ^ { \mathrm { p r o b e } } \leq B _ { H }$ , in addition to the preceding native and current-loss gates; otherwise restore HOLD. The combined-loss subtraction here and the separate CE/KL recombination in Appendix B are distinct floating-point readouts of the same real-arithmetic contrast. Ordinary training then continues. On this diagnostic trajectory, FAT commits at updates $6 5 ,$ , 129, 193 to slots 6, 7, 0, while accepted FRT writes occur at 96, 160, 192, 224, 256. The shortlisted physical slots at those five probes are 5, 3, 4, 4, 3. Retained diagnostic-anchor records contain the current/historical UIDs, teacher ages, prior-slot state and unrounded scalar readouts. The diagnostic uses 823 teacher-example forwards: the 803-call schedule plus four additional calls at each of the five probe events. These costs are separate from Table 4.

## C.4 Descriptive four-policy contrasts

For a common scalar endpoint readout, define the arithmetic comparison

$$
I = Y _ { \mathrm { j o i n t } } - Y _ { \mathrm { r e g u l a t i o n } } - Y _ { \mathrm { a c q u i s i t i o n } } + Y _ { \mathrm { b a s e l i n e } } .\tag{C.3}
$$

A joint policy can outperform both separate policies even when this descriptive Exact contrast is zero or negative.   
For NLL, lower values are preferable. The complete-policy contrasts are summarized below.

Table 6. Readout-specific complete-policy contrasts. The policies may follow diferent later states and actions, so these values are not fixed-action causal factorial efects.
<table><tr><td>Trajectory</td><td>Learning rule</td><td>Exact contrast (count)</td><td>Macro-NLL contrast</td></tr><tr><td>Original</td><td>Replay</td><td>-10</td><td>+0.085045</td></tr><tr><td>Original</td><td>Distillation</td><td>-1</td><td>+0.072506</td></tr><tr><td>Changed</td><td>Replay</td><td>+1</td><td>-0.042830</td></tr><tr><td>Changed</td><td>Distillation</td><td>0</td><td>-0.020493</td></tr></table>

For the original setting, the NLL contrasts use the six-decimal endpoints in Table 1. For the changed setting, they use the retained unrounded values; diferencing the displayed endpoints can change the last reported digit. A mixed contrast identifies a fixed-action interaction only with matched parents and treatments, not merely from four terminal policy rows.

## C.5 Verification scope

The reported endpoints are supported by retained predictions, terminal summaries and compact action records. For the second trajectory, all eight endpoint scores were recomputed from per-item records; those records and the executed replay allocations were retained for ofline verification but are not supplied with the manuscript source. Structural-factor snapshots and allocation records were also checked against the specified rules. The original-setting table additionally uses retained terminal summaries where complete per-item records were not available; its displayed precision is unchanged. These are source- and record-consistency checks; they are not additional independent model repetitions, full checkpoint tensor audits, or proofs of statistical generalization. Both trajectories reuse the same evaluation items, which were inspected during development. A new-data comparison, isolated changes of seed and within-task example order, and fully matched computational budgets would be distinct empirical questions. The original experimental comparison prioritized NLL; foregrounding Exact in this manuscript does not retroactively alter that protocol.