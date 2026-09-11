# HOW WRONG CAN A GOOD PREDICTOR BE? DIVERGING UPDATES WITH VANISHING PREDICTIVE KL

A PREPRINT

Qifu Wen<sup>1,2,\*</sup> Shuaijun Liu<sup>3,\*</sup> Zihan Zhou<sup>1</sup> Xi Zeng<sup>1</sup> Ningxin Su<sup>3</sup>

<sup>1</sup>Boston University, Boston, MA, USA <sup>2</sup>Shanghai Jiao Tong University, Shanghai, China <sup>3</sup>The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, China <sup>\*</sup>Co-first authors. qfwen@bu.edu

September 11, 2026

## ABSTRACT

Accurate posterior prediction need not require accurate approximation of Bayesian updates. We prove that an unbounded gap between the update maps can coexist with vanishing predictive KL for every fixed finite $K \geq \bar { 2 }$ in a stationary symmetric Gaussian HMM. Exact Bayesian mixing and an explicit deterministic radial filter act on the same K − 1 belief coordinates. As $q \to 0 ^ { + }$ , their separation in centered logits in the worst case grows at least linearly in the natural confidence scale $L _ { K } ( q )$ , while their categorical $D _ { \mathrm { K L } }$ (exact∥radial) vanishes at the same explicit witness. Along stationary HMM trajectories, the expected terminal KL between filtered posteriors also converges to zero at $\dot { H } ( q ) = \lceil - \log ( q ) / c \rceil + 1$ . Typical blocks without switches drive both filters into a common confidence cone, where softmax curvature suppresses their disagreement; a single Gaussian maximal event controls adaptive noise. A sweep with equally spaced Gaussians over $\bar { K \in \{ 2 , 4 , 8 \} }$ illustrates the opposing trends, and binary controls at long horizons compare saturating and nonsaturating recurrences. The result isolates two missing links between internal update gaps and predictive cost: the contribution of separating states to expected loss and decoder sensitivity. Thus even an unbounded internal update gap does not by itself certify predictive failure. The construction is fixed in K and does not provide a universal criterion for when compression is harmless or characterize when internal gaps must incur task loss.

## 1 Introduction

Transformers retain a growing context, whereas state space and recurrent alternatives compress history into a deterministic state of fixed size. Recurrence expressivity and state tracking are therefore active design constraints [36, 34], with formal comparisons also separating simplified selective SSMs from linear attention [10]. But these architectural results do not answer the question at the task level: when does a strict computational mismatch force worse prediction? In a finite HMM, exact Bayes itself occupies only $K - 1$ belief coordinates; the issue is whether a different update geometry with a fixed state must pay predictive loss.

Pointwise representability, learnability, and task cost weighted by the distribution are distinct. The predictive cost of a pointwise separation depends on the loss the decoder can see under the evaluation path law, including the loss magnitude on rare paths. Impossibility theorems at the architecture level establish representability gaps in specified models [56, 54], whereas learnability can fail even for representable functions [27]. Neither distinction alone determines this task cost. We study this boundary in a sequential model with finitely many states where both can be calculated.

Our result has two levels. At the level of the update maps, for every fixed finite $K \geq 2$ an explicit moving witness makes the quotient distance in centered logits between exact Bayesian mixing and a radial filter grow as $\Omega ( L _ { K } \breve { ( \boldsymbol { q } ) } )$ , while their categorical $D _ { \mathrm { K L } } ( \mathrm { e x a c t \| r a d i a l } )$ at that same witness tends to zero. At the level of the path law, with pairwise distinct scalar Gaussian means and slow symmetric switching, the filters consume the same stationary HMM observations and their expected terminal KL between filtered posteriors at $H ( q ) = \lceil - \log ( q ) / c \rceil + 1$ also tends to zero. Thus even an unbounded internal separation need not induce a nonvanishing performance gap.

Every fixed finite �  2; � → 0<sup>+</sup>  
![](images/ce30b416d139a6fc61f8f731bbba6138f6293b7ed55a71411c0cd2a26cfd6ebf.jpg)

![](images/eacb0d06da04bb09fe8e04eadc44e392d869f3742c25246ffcfc66b66be5a958.jpg)  
Fix � and emissions before $q  0 ^ { + } ;$ ; simplexes show three states.  
Figure 1: Two distinct fixed-K separations. (A) At a common moving input, exact and radial mixing have diverging separation in centered logits but vanishing categorical KL. (B) On shared stationary observations from the Gaussian HMM, their separate recurrences have vanishing expected terminal posterior KL at $H ( q )$ . Both statements fix K and emission parameters before $q \to 0 ^ { + }$ ; the schematic does not assert that (A) implies (B).

The result at the level of the maps is decoder geometry: both updates become increasingly confident in the same state, so softmax erases a growing logit discrepancy. The theorem under the path law adds a distributional argument. On a block with no latent switch, which occurs with high probability, one maximal event controls all adaptive Gaussian noise weights, radial saturation keeps the approximation in a confidence cone with exact Bayes, and a uniform KL envelope absorbs the remaining paths.

A sweep with equally spaced Gaussians over $K \in \{ 2 , 4 , 8 \}$ illustrates the two opposing trends at finite scale: centered states separate while predictive KL shrinks. The binary case is a microscope rather than the theorem’s scope: it visualizes the geometry and separates saturation from tanh specifically. The training experiments do not establish transfer to learned models.

We prove that, for fixed K, internal separation can diverge while predictive KL vanishes at the same input. We also prove convergence of expected predictive KL under the stationary law as the horizon grows in the limit of rare switching. The proof explains how decoder sensitivity and the contribution of separating states to expected loss control predictive error. Controlled numerical experiments illustrate the two results; simpler recurrences also achieve small KL at the tested short horizons (Table 7). We do not claim uniformity in growing K, an impossibility result for a class of architectures, or learnability of the radial rule.

Figure 1 distinguishes the statements at the common input and under the path law.

## 2 Related Work

Expressivity, learning, and task cost. Efficient sequence models make recurrence expressivity an explicit design variable. Recent work strengthens selective state space updates, analyzes their limits on tracking the state, and compares simplified selective layers with linear attention [36, 34, 10, 56, 1]. Results on the worst case and on formal languages identify computations that specified architectures cannot represent [43, 54], while complementary results show that expressible functions can still be hard to learn [27]. These results determine representation or optimization, not the predictive cost weighted by the distribution of a mismatch. Predictive cost additionally depends on decoder sensitivity and the contribution of the separating states weighted by loss under the evaluation path law. With unbounded $\mathrm { K L }$ vanishing event probability alone does not control this contribution. An expressivity analysis for a specific task for GNNs makes a related diagnosis [35]; our contribution is an explicit sequential counterexample with a proved loss under the path law.

Harmless information loss and internal divergence. Compression that is aware of the task already shows that discarded information can be harmless relative to a downstream predictor family or distortion [16, 15, 26]. We do not optimize a representation or bit rate: the state dimension and prescribed update family are fixed, while their numerica maps depend on q and their distance in centered logits diverges. The generic mathematical shape is not new in isolation. On separable data, gradient descent can send linear predictor norms to infinity while logistic loss tends to zero and the normalized direction converges [62]. Our conjunction is different: two explicit filtering maps diverge at the same centered input, their categorical KL vanishes there, and a second theorem proves predictive convergence under a stationary sequential law.

Approximate Bayesian filtering and finite memory. RNNs can approximate Bayesian filtering statistics on fixed finite horizons and, under additional conditions, uniformly in time [5]. Neural Bayesian Filtering instead learns belief embeddings of fixed length and proves particle consistency over a finite horizon under its stated assumptions [61]. Prediction with short memory gives a different benchmark: an order-ℓ Markov predictor attains average KL at most $I ( \mathcal { M } ) / \ell$ and average $\ell _ { 1 }$ error at most $\sqrt { I ( \mathcal { M } ) / \ell } \left[ 5 7 \right]$ . For an n-state HMM, $I ( { \mathcal { M } } ) \leq \log n$ , independently of mixing time. This benchmark concerns distributions of the next observation, whereas our theorem measures categorical filtered posteriors. Our predictor maintains $K - 1$ belief coordinates. Streaming projection instead retains a truncated mixture over latent paths with a fixed budget through a deterministic recurrence [17]. The distinction is the maintained object and approximation operation, not recurrence itself; our result couples diverging centered map distance with vanishing categorical KL.

Filter stability and robustness. Classical filter stability asks whether filters with different initial conditions merge while running the correct model [46, 9, 41]. Robustness theory instead perturbs model kernels and controls policy costs [31, 33] or error in the filter kernel and policy performance [14] under the respective assumptions. Control policies with a finite window can also become nearly optimal as the window grows [32]. Our recurrence is neither a differently initialized exact filter nor a sequence of kernels converging to the true one; its pointwise update mismatch grows on the stated witness.

Detection delay and the binary controls. The affine control at the long endpoint is adjacent to quickest change detection. Page introduced the cumulative sum scheme [48]; Lorden introduced a criterion on conditional delay in the worst case and proved asymptotic optimality, Moustakides established exact CUSUM optimality under it, and Pollak studied nearly minimax detection under a different criterion on conditional delay [40, 44, 52]. Pollak’s irreversible model with a single change is adjacent to but not identical to our HMM that switches repeatedly. These works calibrate the binary diagnostic but do not imply the fixed-K radial theorem.

Why state space models are the motivating case. The design pattern of a deterministic state spans structured SSMs, linear RNNs, gated linear attention, and recurrent competitors [24, 25, 60, 47, 67, 68, 49, 3, 12]. Selective SSM theory and studies of state tracking show how transition structure, composition, precision, or normalization alter representational capability [45, 59, 7, 64, 58, 38]. They motivate the logical question studied here. We do not prove that any named architecture implements, or fails to implement, either filter in our theorem.

## 3 Problem Setup: Gaussian HMM with a Fixed Finite State and Radial Filter

Definition 1 (Stationary K-state Gaussian HMM and coupled filters). Fix an integer $K \geq 2 ,$ pairwise distinct means $\mu _ { 0 } , \ldots , \mu _ { K - 1 } \in \mathbb { R } ,$ , and $\sigma > 0$ , with $0 < q < ( K - 1 ) / K$ so that $L _ { K } ( q ) > 0$ and $\alpha _ { K } ( q ) > 0$ . Let $( S _ { t } ) _ { t \geq 0 }$ be stationary and uniform on $\{ 0 , \ldots , K - 1 \}$ , with

$$
\operatorname* { P r } ( S _ { t + 1 } = i \mid S _ { t } = i ) = 1 - q , \qquad \operatorname* { P r } ( S _ { t + 1 } = j \mid S _ { t } = i ) = { \frac { q } { K - 1 } } \quad ( j \neq i ) .
$$

Conditional on $S _ { t } = s ,$ , let $X _ { t + 1 } \sim \mathcal { N } ( \mu _ { s } , \sigma ^ { 2 } )$ . Write $\begin{array} { r } { \mathcal { C } z = z - K ^ { - 1 } ( \mathbf { 1 } ^ { \top } z ) \mathbf { 1 } } \end{array}$ for centering and

$$
g ( x ) _ { s } = \frac { \mu _ { s } x } { \sigma ^ { 2 } } - \frac { \mu _ { s } ^ { 2 } } { 2 \sigma ^ { 2 } }
$$

for the Gaussian score, understood modulo common shifts. For centered logits z, define exact transition mixing

$$
\Phi _ { q } ( z ) = \mathcal { C } \log ( P _ { q } ^ { \top } \operatorname { s o f t m a x } ( z ) ) ,
$$

where the logarithm is componentwise. Let

$$
r ( z ) = \sqrt { 2 } \lVert \mathcal { C } z \rVert _ { 2 } , \qquad \alpha _ { K } ( q ) = 1 - \frac { K q } { K - 1 } , \qquad L _ { K } ( q ) = \log \frac { ( K - 1 ) ( 1 - q ) } { q } ,
$$

and define the radial map

$$
R _ { q } ( z ) = \frac { L _ { K } ( q ) \operatorname { t a n h } ( \alpha _ { K } ( q ) r ( z ) / L _ { K } ( q ) ) } { r ( z ) } \mathcal { C } z ,
$$

with continuous value $R _ { q } ( 0 ) = 0$ and scale $\alpha _ { K } ( q )$ at the origin. Starting from $z _ { 0 } ^ { E } = z _ { 0 } ^ { R } = 0 ,$ , the coupledfilters use the same observations:

$$
z _ { t + 1 } ^ { E } = { \mathcal C } \big ( \Phi _ { q } ( z _ { t } ^ { E } ) + g ( X _ { t + 1 } ) \big ) , \qquad z _ { t + 1 } ^ { R } = { \mathcal C } \big ( R _ { q } ( z _ { t } ^ { R } ) + g ( X _ { t + 1 } ) \big ) .
$$

Their terminal categorical outputs are $p _ { t } ^ { E } = \mathrm { s o f t m a x } ( z _ { t } ^ { E } )$ and $p _ { t } ^ { R } = \mathrm { s o f t m a x } ( z _ { t } ^ { R } )$ . For $t \geq 1 , p _ { t } ^ { E }$ is the filtered posterior of $S _ { t - 1 }$ given $X _ { 1 : t } ; \mathbf { \widehat { p } } _ { t } ^ { R }$ approximates that posterior. These outputs precede any additional transition or emission channel.

Define

$$
d _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i \neq j } \frac { ( \mu _ { i } - \mu _ { j } ) ^ { 2 } } { 2 \sigma ^ { 2 } } .
$$

Fix $0 < c < d _ { \operatorname* { m i n } }$ and set $H ( q ) = \lceil ( - \log q ) / c \rceil + 1$ . Constants below may depend on the fixed K, the full mean vector, σ, and c. Appendix B and Table 4 translate these centered coordinates into a geometric example with three states.

## 4 Main Result and Proof Mechanism

Definition 2 (Centered separation between the update maps). For centered $z \in \mathbb { R } ^ { K }$ , define the quotient distance

$$
\mathcal { D } _ { q } ( z ) = \operatorname* { m a x } _ { i , j } \left| [ ( \Phi _ { q } ( z ) ) _ { i } - ( \Phi _ { q } ( z ) ) _ { j } ] - [ ( R _ { q } ( z ) ) _ { i } - ( R _ { q } ( z ) ) _ { j } ] \right| .
$$

It ignores common logit shifts, which do not change a categorical prediction.

Theorem 3 (Diverging updates with vanishing predictive KL). Fix any finite $K \geq 2 ,$ , let $L = L _ { K } ( q )$ , and set

$$
z _ { q } = L ( 1 / 4 , - 1 / 4 , 0 , \ldots , 0 ) , \qquad c _ { \star } = \frac { 1 } { 2 } - \operatorname { t a n h } \frac { 1 } { 2 } > 0 .
$$

Then

$$
\frac { [ ( \Phi _ { q } ( z _ { q } ) ) _ { 0 } - ( \Phi _ { q } ( z _ { q } ) ) _ { 1 } ] - [ ( R _ { q } ( z _ { q } ) ) _ { 0 } - ( R _ { q } ( z _ { q } ) ) _ { 1 } ] } { L } \longrightarrow c _ { \star } \qquad ( q  0 ^ { + } ) .
$$

Consequently, for all sufficiently small $q ,$ with the threshold allowed to depend on fixed K,

$$
\operatorname* { s u p } _ { z : \mathcal { C } z = z } \mathcal { D } _ { q } ( z ) \geq \mathcal { D } _ { q } ( z _ { q } ) \geq \frac { c _ { \star } } { 2 } L _ { K } ( q ) \longrightarrow \infty .
$$

Nevertheless, at the same witness,

$$
D _ { \mathrm { K L } } ( \mathrm { s o f t m a x } \Phi _ { q } ( z _ { q } ) \| \mathrm { s o f t m a x } R _ { q } ( z _ { q } ) ) \longrightarrow 0 .
$$

Theorem 4 (Predictive convergence with a fixed finite state). Setting. For every fixed finite $K \geq 2 ,$ , consider Definition 1 with pairwise distinct means, $\sigma > 0 _ { : }$ , and $0 < c < d _ { \operatorname* { m i n } }$ . The twofilters share the stationary HMM observation path and are evaluated at $H ( q ) = \lceil ( - \log q ) / c \rceil + 1$

Conclusion.

$$
{ \mathbb E } \Big [ D _ { \mathrm { K L } } \Big ( p _ { H ( q ) } ^ { E } \Big | \Big | p _ { H ( q ) } ^ { R } \Big ) \Big ] \longrightarrow 0 \qquad a s q \to 0 ^ { + } .
$$

The expectation is under the stationary HMM path law.

Logical separation. Theorem 3 is a statement about the maps in the worst case, with a matched decoder statement: the internal distance diverges at an explicit moving witness, yet the two decoded predictions agree there. Theorem 4 is a distributional statement: predictive agreement also holds in expectation along trajectories generated by the stationary HMM. The first rules out the explanation that the update maps merely become uniformly close; the second shows that the phenomenon is not confined to one logit vector chosen by hand.

Table 1: Summary of results. The first three rows are mathematical results. The fourth is an illustration at finite scale, and the last row records exclusions rather than negative results.
<table><tr><td>Object</td><td>Quantifier</td><td>Status</td></tr><tr><td>Centered map distance</td><td>every fixed finite  $K \geq 2$ </td><td>proved  $\Omega ( L _ { K } )$ </td></tr><tr><td>Categorical KL at the same input</td><td>every fixed finite  $K \geq 2$ </td><td>proved → 0</td></tr><tr><td>Stationary terminal KL</td><td>every fixed finite  $K \geq 2$ </td><td>proved  $\mathbb { E } \mathrm { K L }  0$ </td></tr><tr><td>Equally spaced Gaussian illustration</td><td> $K \in \{ 2 , 4 , 8 \}$ </td><td>measured; not used in the proof</td></tr><tr><td>Growing K or architecture class</td><td> $K = K ( q )$  ; named models</td><td>not claimed</td></tr></table>

Mechanism. At the explicit witness, both maps place the same coordinate ahead by a margin proportional to $L _ { K } ( q )$ so categorical softmax curvature suppresses their growing internal discrepancy. Along HMM paths, with probability tending to one the logarithmic window contains no latent switch. Conditional on a fixed state, the centered score decomposes into a deterministic direction and one scalar Gaussian noise direction. Because every radial mixing coefficient lies in (0, 1), Abel summation controls the full adaptive noise process by one maximum of a Gaussian partial sum. For $K \geq 3 ,$ , a radial barrier keeps the state at order $L _ { K } ( q ) ^ { 2 / 3 }$ and a matching lower bound drives every margin of the true state over a wrong state to the same order after an initial transient. Exact Bayes has an even larger margin. The separate binary argument supplies a common logarithmic margin. The curvature bound in the common confidence cone makes the KL vanish on this event, while a $2 L _ { K } ( q )$ envelope absorbs switch paths and failures of the maximal event. Appendix D explains these proof steps and the dependence of the constants on K.

Scope. Both results fix K before taking $q \to 0 ^ { + }$ . The quantitative witness moves outward with $L _ { K } ( q ) ;$ ; it does not imply a gap bounded away from zero on every fixed bounded input. Neither theorem is uniform in growing K or nearly colliding means, and neither is an impossibility theorem for an entire SSM architecture class or a claim about what training by gradient descent will learn. Table 1 separates proved statements, evidence at finite scale, and exclusions.

## 5 Evidence at Finite Scale for Fixed K

Theorem 4 is asymptotic and does not specify a universal finite-q onset. We therefore use a deliberately simple numerical illustration that matches its Gaussian assumptions. For $K \in \{ 2 , 4 , 8 \}$ , the state means are equally spaced, centered, and rescaled to minimum pairwise distance one. We evaluate $L \ ' \in \{ 8 , \bar { 1 6 } , 3 2 , 6 4 , 1 2 8 \}$ with 4,096 paired stationary paths per cell. Exact Bayes and the radial filter consume the same observations. The two displayed quantities are the centered distance between terminal states and the categorical $D _ { \mathrm { K L } } ( \mathrm { e x a c t \| r a d i a l } )$ after the observation.

Figure 2 shows increasing sampled state distance and decreasing categorical KL across all three displayed values of K: internal distance grows with the scale of rare switching while predictive KL falls. The curves are not used to establish the theorem or to claim a uniform onset in K. Appendix F gives the estimands, coupling, uncertainty calculation, and exact summary statistics.

## 6 Binary Specialization and Geometry in the Finite Regime

For $K = 2 ,$ , translating the two means by their midpoint reduces the model to $( - \mu , + \mu )$ without changing likelihood ratios. Centered logits then collapse to one scalar h. Exact mixing becomes $F _ { q } ( h )$ and the radial witness becomes the tanh clamp $G _ { q } ( h )$ below. Theorem 3 already supplies an explicit, diverging separation at the level of the maps for every fixed K. This binary slice instead serves as a visualization in the finite regime and isolates why an affine map cannot represent exact mixing; the stationary fixed-K theorem does not depend on the affine obstruction proved here.

Proposition 5 (Geometry of exact mixing). For every $\begin{array} { r } { 0 < q < \frac { 1 } { 2 } } \end{array}$ the exact mixing map $F _ { q }$ of Definition 1 satisfies:

(i) Saturation: $| F _ { q } ( h ) | < L ( q )$ for every finite h.

(ii) Contraction: $F _ { q }$ contracts logit perturbations with globalfactor at most $1 - 2 q ,$ , attained at the midpoint.

(iii) Not affine: with $x = \log 2$ and

$$
\Delta _ { q } : = \left| F _ { q } ( 0 ) - 2 F _ { q } ( x ) + F _ { q } ( 2 x ) \right| > 0 ,
$$

every affine map A satisfies max $\begin{array} { r } { \dot { { \boldsymbol { \cdot } } } _ { z \in \{ 0 , x , 2 x \} } | F _ { q } ( z ) - A ( z ) | \geq \Delta _ { q } / 4 . } \end{array}$

![](images/89b6a3c64647c8b9de038a3f478578b11b447de2d225b32fc2eabc72a89e107f.jpg)

![](images/bf60b0dd4f991e407952ef98f0172ec8fb12674e8f742e3913bf4de608bb79e1.jpg)  
Equal-Gaussian K=2,4,8; 4096 paths per cell; 95% bootstrap intervals. H = ⌈L/. 225⌉ + 1. Finite illustration, not a trajectory-divergence proof.

![](images/0125742197017c2171c9301ed8cc4299f41889ac9103409d82cf4cce39e4a5b8.jpg)

![](images/b16619e2eb97cd3f4bef59527ee5bea3c0060e3e37a7b586f889c9a29d7a27b6.jpg)  
10 seeds · SE ≤ 3.1%  
Figure 2: Separation at finite scale and decoder saturation. Left pair: centered distance between terminal states grows while categorical $D _ { \mathrm { K L } }$ (exact∥radial) falls for Gaussian HMMs with $K = 2 , 4 , 8$ (4,096 paired paths per cell; 95% bootstrap intervals). Right pair: KL of the binary controls over ten seeds at $H ( q )$ and $\lceil 8 / q \rceil$ (relative $\mathrm { S E } \leq 3 . 1 \% $ ), followed by an analytic illustration with a logistic decoder, not sampled trajectories. Nearly equal endpoints are displaced for legibility. Table 7 and Appendix F give values, clipping conventions, and estimation details. Measurements illustrate, rather than prove, the asymptotic theorem.

## Proof. See Appendix E.

Remark 6 (The obstruction does not give a lower bound). Part (iii) ofProposition 5 is adverse evidencefor a naive pointwise argument. Direct expansion gives $\begin{array} { r } { \Delta _ { q } = \frac { 3 } { 4 } q + O ( q ^ { 2 } ) \ : a s \ : q  0 ^ { \mp } } \end{array}$ . Thus the obstruction is strictfor every $q > 0$ but its quantitative margin is not bounded awayfrom zero along thefamily with rare switching, so it cannot yield a nonvanishing asymptotic predictive lower bound.

Moreover, predictive loss is evaluated after the logistic decoder under the HMM path law, not in the logit norm in the worst case. The approximate map $G _ { q }$ preserves the exact saturation scale and midpoint slope, and the shared envelope $| F _ { q } ( h ) | , | G _ { q } ( h ) | \overset { \because } { < } L ( q )$ gives global control without requiring uniform closeness.

The exact, radial, and affine transition geometry is shown in Figure 3 in the appendix.

## 7 Binary Controls and Learned Transfer

The experiment with learned models asks whether recurrences faithful to the architecture recover the proved mechanism across six switch probabilities. When trained end to end, the selective SSM arm closes at least half the reference gap at four of six q values, below the prespecified criterion of five out of six; these results do not establish transfer to learned models. Under distillation, every nonreference comparison arm meets the threshold of half closure at all six values.

Closure is defined as $\mathrm { ( K L _ { S 6 } - K L _ { a r m } ) / ( K L _ { S 6 } - K L _ { t a n h } ) }$ , so the constrained scalar S6 arm sits at 0, the analytic tanh recurrence sits at 1, and values above 1 mean the learned arm beats the proved mechanism. Here every architecture loss is $D _ { \mathrm { K I } }$ (exact Bayes∥model), evaluated after converting logits to float32 probabilities and clipping both arguments to $[ \varepsilon _ { 3 2 } , 1 - \varepsilon _ { 3 2 } ]$ , where $\varepsilon _ { 3 2 } =$ torch.finfo(torch.float32).eps $\approx 1 . 1 \dot { 9 } 2 0 9 2 9 \times 1 0 ^ { - 7 }$ . This differs from the metric for the scalar controls in Table 7, which uses $1 0 ^ { - 1 5 }$ clipping. Table 2 gives the breakdown at the horizon matched to the theorem $H ( q ) = \lceil - \ln q \rceil + 1$

The learned arm passes at the four smaller tested q values, but this empirical split does not identify the theorem’s asymptotic onset. $\mathrm { ~ A t ~ } q = 2 ^ { - 3 }$ the reference gap $\mathrm { K L } _ { \mathrm { S 6 } } - \mathrm { K L } _ { \mathrm { t a n h } }$ is only $\mathbf { \dot { 1 } . 5 \times 1 0 ^ { - 3 } }$ , so closure is scale sensitive and seed dispersion is widest at the two largest q. Even so, the two points below half closure lie 5.4 and 2.8 standard errors below the threshold, whereas the four passing points lie at least 26 standard errors above it using the unrounded results across seeds. The outcome of four out of six is therefore not explained by marginal uncertainty. Under distillation, the closure computed from the ratio of mean losses ranges from 1.16 to 3.97 and every nonreference comparison arm passes at all six q values. Table 2 instead reports the mean of closure ratios for each seed.

Generalization at long horizons separates the learned architectures. The secondary endpoint on long paths evaluates the same trained models on sequences of length $\lceil 8 / q \rceil$ , up to 2048 steps, against a training horizon matched to the theorem that is at most eight. Here the ranking inverts. The analytic tanh recurrence is horizon stable, its predictive KL staying between $1 . 7 \times 1 0 ^ { - 4 }$ and $6 . 4 \times 1 0 ^ { - 3 }$ at the two tested switch probabilities, while the scalar S6 arm degrades to $1 . 9 \dot { \times } 1 \dot { 0 } ^ { - 1 }$ . The GRU inherits that stability, passing four of six q when trained end to end and all six under distillation. The official Mamba blocks do not: when trained end to end, two stacked blocks pass only two of six, and a single block fails at every q, with a worst per-q mean closure at long horizons of −32.65 at $q = 2 \AA ^ { - 3 }$ when trained end to end.

Table 2: Closure per q when trained end to end, mean ± standard error over ten evaluation seeds. Closure places the constrained scalar S6 arm at 0 and the analytic tanh recurrence at 1, so the arm faithful to the architecture meets the criterion of half closure only at $q \in \{ 2 ^ { - 5 } , 2 ^ { - 6 } , 2 ^ { - 7 } , 2 ^ { - 8 } \}$ , the four empirically passing points. The asymptotic theorem supplies no finite-q cutoff.
<table><tr><td>q</td><td>Mamba ×2</td><td>GRU</td><td>Finite window</td></tr><tr><td colspan="4">Below empirical half closure</td></tr><tr><td> $2 ^ { - 3 }$ </td><td> $- 1 . 5 9 \pm 0 . 3 9$ </td><td> $0 . 6 1 \pm 0 . 0 6$ </td><td> $- 2 . 7 5 \pm 0 . 6 5$ </td></tr><tr><td> $2 ^ { - 4 }$ </td><td> $- 0 . 0 9 \pm 0 . 2 1$ </td><td> $1 . 0 0 \pm 0 . 0 3$ </td><td> $- 1 . 1 7 \pm 0 . 4 7$ </td></tr><tr><td colspan="4">Above empirical half closure</td></tr><tr><td> $2 ^ { - 5 }$ </td><td> $1 . 0 4 \pm 0 . 0 2$ </td><td> $1 . 1 1 \pm 0 . 0 3$ </td><td> $0 . 0 4 \pm 0 . 2 0$ </td></tr><tr><td> $2 ^ { - 6 }$ </td><td> $1 . 1 4 \pm 0 . 0 1$ </td><td> $1 . 1 8 \pm 0 . 0 1$ </td><td> $0 . 6 8 \pm 0 . 0 6$ </td></tr><tr><td> $2 ^ { - 7 }$ </td><td> $1 . 1 3 \pm 0 . 0 2$ </td><td> $1 . 1 4 \pm 0 . 0 2$ </td><td> $0 . 7 4 \pm 0 . 0 7$ </td></tr><tr><td> $2 ^ { - 8 }$ </td><td> $1 . 1 6 \pm 0 . 0 1$ </td><td> $1 . 1 9 \pm 0 . 0 1$ </td><td> $0 . 8 7 \pm 0 . 0 3$ </td></tr></table>

Under distillation, both Mamba depths pass zero of six at this long endpoint. We therefore do not claim that a canonical selective SSM acquires the mechanism. Matching the exact filter at the theorem’s horizon does not ensure stability at longer horizons.

Scalar controls test a simpler explanation. A naive recurrence could also achieve low loss at the horizon matched to the theorem, making the exhibited tanh geometry unnecessary. We therefore measured three controls against exact Bayes with the same generator $( \mu = \sigma = 1$ , 200,000 paths per point at $H ( q )$ , 20,000 at $\lceil 8 / q \rceil$ , split across ten fixed seeds). They are a plain affine recurrence $h  ( 1 - 2 q ) h + \ell ( x )$ with no saturation, the class considered by the pointwise lower bound; a hard clip of that map at $\pm L ( q ) ;$ ; and the identity accumulator. The third panel of Figure 2 shows the endpoint pattern; Appendix F, Table 7, reports the exact cells.

The controls support three conclusions. First, both saturating recurrences are horizon stable at the two tested switch probabilities, while the affine and identity controls degrade at the long endpoint. Their relative ordering is not uniform: the hard clip has lower predictive KL at $\dot { \boldsymbol { q } } = 2 ^ { - 8 }$ , whereas the tanh clamp has lower predictive KL at $\bar { \boldsymbol { q } } = 2 ^ { - 3 }$ . These two points do not identify a crossover threshold. The theorem should therefore be read as proving one saturating instance, not the optimal one. Second, the endpoint matched to the theorem does not establish a separation among scalar recurrence classes; the binary empirical separation among these scalar controls occurs at the long endpoint and is measured, not proved. At that endpoint the affine map reaches 5.53 nats at $q = 2 \AA ^ { - 8 }$ , while both saturating controls remain stable. Third, the closure scale of Table 2 anchors 1 to one saturating witness among several; the readings from the learned arm remain meaningful as calibrated distances to an analytic reference with a guarantee at the stated logarithmic horizon. Its stability at longer horizons is measured here, not established by that theorem.

Hyperparameters were selected by mean validation loss over three tuning seeds before confirmation, with ties broken deterministically and incomplete settings excluded. Tuning and confirmation seeds are disjoint, all final evaluation seeds are included, and the supplement gives the procedure for model selection and implementation details.

## 8 Limitations

Both theorems fix finite K before taking $q \to 0 ^ { + }$ . Their constants can deteriorate with K and with nearly colliding means; no rate uniform in $K = K ( q )$ is proved. The quantitative witness moves with $L _ { K } ( q )$ , so it gives neither a gap on every bounded input nor a lower bound on the state dimension. The stationary theorem further assumes pairwise distinct scalar Gaussian means, symmetric switching, stationary initialization, and the stated logarithmic horizon.

The sweep with equally spaced Gaussians is an illustration at finite scale, not evidence for a uniform onset rate. The comparison concerns two explicit maps, not every SSM, and does not show that training learns the radial rule. Binary controls are diagnostics in the finite regime: hard clipping can beat tanh, and the training experiments do not establish transfer to learned models. The result is a counterexample, not a universal criterion for pruning, quantization, distillation, low rank adaptation, or other compression. The converse question, when an internal mismatch must incur nonvanishing task loss, remains open.

## 9 Conclusion

For every fixed finite latent state space, exact Bayesian mixing and our radial filter can diverge linearly in centered logits at an explicit witness while their decoded categorical KL vanishes. Predictive agreement also holds in stationary expectation in the symmetric scalar Gaussian HMM, so it is neither explained by uniform internal approximation nor confined to one input chosen by hand. A controlled sweep with equally spaced Gaussians over $\stackrel { \triangledown } { \vec { K } } \in \{ 2 , 4 , 8 \}$ illustrates both trends at finite scale, while binary controls compare behavior at short and long horizons. At the common input witness, softmax suppresses a growing map discrepancy; on typical HMM paths, a shared confidence cone suppresses predictive error. The result therefore identifies two missing links from internal separation to predictive loss: its contribution weighted by loss under the evaluation path law, and the decoder’s sensitivity there. It does not show that this failure of gap transfer is prevalent in learned models or provide a converse criterion; uniform growing-K rates and extensions to learned sequence models remain open.

## Reproducibility statement

The assumptions and complete paper proofs for the stated results are given in the appendices. Appendix F specifies the empirical estimands, aggregation, model configurations, hyperparameters, seeds, and clipping conventions. The anonymous supplementary package contains the simulation code and numerical results used for the reported experiments.

## References

[1] Eric Alsmann, Lowejatan Noori, and Martin Lange. On the expressiveness of state space models via temporal logics. In International Conference on Learning Representations, 2026.

[2] Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, and Christopher Ré. Simple linear attention language models balance the recall-throughput tradeoff. arXiv preprint arXiv:2402.18668, 2024.

[3] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long short-term memory. In Advances in Neural Information Processing Systems, 2024. arXiv:2405.04517.

[4] Philipp Becker, Niklas Freymuth, and Gerhard Neumann. KalMamba: Towards efficient probabilistic state space models for RL under uncertainty. arXiv preprint arXiv:2406.15131, 2024.

[5] Adrian N. Bishop and Edwin V. Bonilla. Recurrent neural networks and universal approximation of bayesian filters. In Proceedings ofthe 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings ofMachine Learning Research, pp. 6956–6967. PMLR, 2023. URL https://proceedings.mlr. press/v206/bishop23a.html.

[6] David Blackwell. The entropy of functions of finite-state markov chains. In Transactions of the First Prague Conference on Information Theory, Statistical Decision Functions, Random Processes, pp. 13–20. Publishing House of the Czechoslovak Academy of Sciences, 1957.

[7] Marco Bondaschi, Nived Rajaraman, Xiuying Wei, Kannan Ramchandran, Razvan Pascanu, Caglar Gulcehre, Michael Gastpar, and Ashok Vardhan Makkuva. From markov to laplace: How mamba in-context learns markov chains, 2025. arXiv:2502.10178.

[8] Loïc Cabannes, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Matthijs Douze, Maria Lomeli, Ilze Amanda Auzina, Justin Carpentier, Gabriel Synnaeve, and Hervé Jégou. Sparse delta memory: Scaling the state of linear RNNs through sparsity. arXiv preprint arXiv:2607.07386, 2026.

[9] Pavel Chigansky, Robert Liptser, and Ramon van Handel. Intrinsic methods in filter stability. In The Oxford Handbook ofNonlinear Filtering, pp. 319–351. Oxford University Press, 2011.

[10] Edo Cohen-Karlik, Itamar Zimerman, Liane Galanti, Ido Andrew Atad, Amir Globerson, and Lior Wolf. On the expressivity of selective state-space layers: A multivariate polynomial approach. In Proceedings ofthe 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings ofMachine Learning Research, pp. 136–144. PMLR, 2026. URL https://proceedings.mlr.press/v300/cohen-karlik26a. html.

[11] Adrien Corenflos, James Thornton, George Deligiannidis, and Arnaud Doucet. Differentiable particle filtering via entropy-regularized optimal transport. In International Conference on Machine Learning, 2021. arXiv:2102.07850.

[12] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning, 2024. arXiv:2405.21060.

[13] Grégoire Delétang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, and Pedro A. Ortega. Neural networks and the chomsky hierarchy. In International Conference on Learning Representations, 2023. arXiv:2207.02098.

[14] Yunus Emre Demirci, Ali Devran Kara, and Serdar Yüksel. Sensitivity of filter kernels and robustness bounds to transition and measurement kernel perturbations in partially observable stochastic control. arXiv preprint arXiv:2508.10658, 2025.

[15] Yann Dubois, Douwe Kiela, David J. Schwab, and Ramakrishna Vedantam. Learning optimal representations with the decodable information bottleneck. In Advances in Neural Information Processing Systems, 2020. arXiv:2009.12789.

[16] Yann Dubois, Benjamin Bloem-Reddy, Karen Ullrich, and Chris J. Maddison. Lossy compression for lossless prediction. In Advances in Neural Information Processing Systems, 2021. arXiv:2106.10800.

[17] Gerardo Duran-Martin. A predictive view on streaming hidden markov models, 2026. arXiv:2604.09208.

[18] Meir Feder and Yury Polyanskiy. Sequential prediction under log-loss and misspecification. arXiv preprint arXiv:2102.00050, 2021.

[19] Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, and Christopher Ré. Hungry hungry hippos: Towards language modeling with state space models. In International Conference on Learning Representations, 2023. arXiv:2212.14052.

[20] Aydin Ghojogh, M. Hadi Sepanj, and Benyamin Ghojogh. On the relation of state space models and hidden markov models. arXiv preprint arXiv:2601.13357, 2026.

[21] Tingnan Gong, Junghwan Lee, Xiuyuan Cheng, and Yao Xie. Neural network-based CUSUM for online changepoint detection. arXiv preprint arXiv:2210.17312, 2022.

[22] Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, and Massimiliano Pontil. Unlocking state-tracking in linear RNNs through negative eigenvalues. arXiv preprint arXiv:2411.12537, 2024.

[23] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024. arXiv:2312.00752.

[24] Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. HiPPO: Recurrent memory with optima polynomial projections. In Advances in Neural Information Processing Systems, 2020. arXiv:2008.07669.

[25] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022. arXiv:2111.00396.

[26] Hassan Hafez-Kolahi, Behrad Moniri, Shohreh Kasaei, and Mahdieh Soleymani Baghshah. Rate-distortion analysis of minimum excess risk in bayesian learning. In International Conference on Machine Learning, 2021 arXiv:2105.04180.

[27] Michael Hahn and Mark Rofin. Why are sensitive functions hard for transformers? In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics, 2024. arXiv:2402.09963.

[28] Ramin Hasani, Mathias Lechner, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, and Daniela Rus. Liquid structural state-space models. In International Conference on Learning Representations, 2023. arXiv:2209.12951.

[29] Vishesh Jain, Frederic Koehler, Jingbo Liu, and Elchanan Mossel. Accuracy-memory tradeoffs and phase transitions in belief propagation. arXiv preprint arXiv:1905.10031, 2019.

[30] Sham M. Kakade, Akshay Krishnamurthy, Gaurav Mahajan, and Cyril Zhang. Learning hidden markov models using conditional samples. arXiv preprint arXiv:2302.14753, 2023.

[31] Ali Devran Kara and Serdar Yüksel. Robustness to incorrect system models in stochastic control. SIAM Journal on Control and Optimization, 2020. arXiv:1803.06046.

[32] Ali Devran Kara and Serdar Yüksel. Near optimality of finite memory feedback policies in partially observed markov decision processes. Journal ofMachine Learning Research, 23(11):1–46, 2022.

[33] Ali Devran Kara, Maxim Raginsky, and Serdar Yüksel. Robustness to incorrect models and data-driven learning in average-cost optimal stochastic control. Automatica, 2022. arXiv:2003.05769.

[34] Arjun Karuvally, Franz Nowak, T. Anderson Keller, Carmen Amo Alonso, Terrence J. Sejnowski, and Hava T. Siegelmann. Bridging expressivity and scalability with adaptive unitary SSMs. arXiv preprint arXiv:2507.05238, 2025.

[35] Niklas Kemper, Tom Wollschläger, and Stephan Günnemann. What expressivity theory misses: Message passing complexity for GNNs. In Advances in Neural Information Processing Systems, 2025. arXiv:2509.01254.

[36] Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and Albert Gu. Mamba-3: Improved sequence modeling using state space principles. In International Conference on Learning Representations, 2026. arXiv:2603.15569.

[37] François Le Gland and Nadia Oudjane. Stability and uniform approximation of nonlinear filters using the hilbert metric and application to particle filters. The Annals ofApplied Probability, 14(1), 2004. doi: 10.1214/aoap/ 1075828050.

[38] Jiaoda Li and Ryan Cotterell. Characterizing the expressivity of fixed-precision transformer language models. arXiv preprint arXiv:2505.23623, 2025.

[39] Marten Lienen, Abdullah Saydemir, and Stephan Günnemann. UnHiPPO: Uncertainty-aware initialization for state space models. arXiv preprint arXiv:2506.05065, 2025.

[40] G. Lorden. Procedures for reacting to a change in distribution. The Annals ofMathematical Statistics, 42(6), 1971. doi: 10.1214/aoms/1177693055.

[41] Curtis McDonald and Serdar Yüksel. Exponential filter stability via dobrushin’s coefficient. Electronic Communications in Probability, 2020. arXiv:1910.08463.

[42] Arian Mehrfard, Bharanidhar Duraisamy, Stefan Haag, Florian Geiss, and Mirko Mählisch. Adaptive learned state estimation based on KalmanNet. arXiv preprint arXiv:2604.02441, 2026.

[43] William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in state-space models. In International Conference on Machine Learning, 2024.

[44] George V. Moustakides. Optimal stopping times for detecting changes in distributions. The Annals ofStatistics, 14(4), 1986. doi: 10.1214/aos/1176350164.

[45] Nicola Muca Cirone, Antonio Orvieto, Benjamin Walker, Cristopher Salvi, and Terry Lyons. Theoretical foundations of deep selective state-space models. In Advances in Neural Information Processing Systems, 2024.

[46] Daniel Ocone and Etienne Pardoux. Asymptotic stability of the optimal filter with respect to its initial condition. SIAM Journal on Control and Optimization, 34(1):226–243, 1996. doi: 10.1137/S0363012993256617.

[47] Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. Resurrecting recurrent neural networks for long sequences. In International Conference on Machine Learning, 2023. arXiv:2303.06349.

[48] E. S. Page. Continuous inspection schemes. Biometrika, 41(1-2):100–115, 1954. doi: 10.1093/biomet/41.1-2.100.

[49] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Xingjian Du, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemysław Kazienko, Jan Kocon, Jiaming Kong, Bartłomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri,´ Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanisław Wo´zniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. RWKV: Reinventing RNNs for the transformer era. arXiv preprint arXiv:2305.13048, 2023.

[50] Liangzu Peng, Aditya Chattopadhyay, Luca Zancato, Elvis Nunez, Wei Xia, and Stefano Soatto. Gated KalmaNet: A fading memory layer through test-time ridge regression. arXiv preprint arXiv:2511.21016, 2025.

[51] Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. Hyena hierarchy: Towards larger convolutional language models. In International Conference on Machine Learning, 2023. arXiv:2302.10866.

[52] Moshe Pollak. Optimal detection of a change in distribution. The Annals of Statistics, 13(1), 1985. doi: 10.1214/aos/1176346587.

[53] Lawrence R. Rabiner. A tutorial on hidden markov models and selected applications in speech recognition. Proceedings ofthe IEEE, 77(2):257–286, 1989.

[54] Yash Sarrof, Yana Veitsman, and Michael Hahn. The expressive capacity of state space models: A formal language perspective. In Advances in Neural Information Processing Systems, 2024.

[55] Vaisakh Shaj, Cameron Barker, Aidan Scannell, Andras Szecsenyi, Elliot J. Crowley, and Amos Storkey. Kalman linear attention: Parallel bayesian filtering for efficient language modelling and state tracking. In International Conference on Machine Learning, 2026.

[56] Mehran Shakerinava, Behnoush Khavari, Siamak Ravanbakhsh, and Sarath Chandar. The expressive limits of diagonal ssms for state-tracking. In International Conference on Learning Representations, 2026.

[57] Vatsal Sharan, Sham Kakade, Percy Liang, and Gregory Valiant. Prediction with a short memory, 2018. arXiv:1612.02526.

[58] Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. DeltaProduct: Improving state-tracking in linear RNNs via householder products. arXiv preprint arXiv:2502.10297, 2025.

[59] Julien Siems, Riccardo Grazzi, Korbinian Pöppel, Kirill Kalinin, Hitesh Ballani, and Babak Rahmani. Learning state-tracking from code using linear rnns. In International Conference on Learning Representations, 2026.

[60] Jimmy T.H. Smith, Andrew Warrington, and Scott W. Linderman. Simplified state space layers for sequence modeling. In International Conference on Learning Representations, 2023. arXiv:2208.04933.

[61] Christopher Solinas, Radovan Haluška, David Sychrovský, Finbarr Timbers, Nolan Bard, Michael Buro, Martin Schmid, Nathan R. Sturtevant, and Michael Bowling. Neural bayesian filtering, 2025. arXiv:2510.03614.

[62] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, and Nathan Srebro. The implicit bias of gradient descent on separable data. In International Conference on Learning Representations, 2018. URL https://openreview. net/forum?id=r1q7n9gAb.

[63] Alex Tang, M. Emrullah Ildiz, Batin Kurt, Samet Oymak, and Necmiye Ozay. On the generalization properties of selective state-space models for filtering tasks for unknown systems. arXiv preprint arXiv:2604.23818, 2026.

[64] Aleksandar Terzic, Nicolas Menet, Michael Hersche, Thomas Hofmann, and Abbas Rahimi. Structured sparse´ transition matrices to enable state tracking in state-space models. In Advances in Neural Information Processing Systems, 2025. arXiv:2509.22284.

[65] Benjamin Walker and Terry Lyons. Chess-world-model: A 10M-game benchmark for exact state tracking from chess move sequences. arXiv preprint arXiv:2605.30100, 2026.

[66] Geoffrey Wolfer and Aryeh Kontorovich. Estimating the mixing time of ergodic markov chains. arXiv preprint arXiv:1902.01224, 2019.

[67] Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In International Conference on Machine Learning, 2024. arXiv:2312.06635.

[68] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In Advances in Neural Information Processing Systems, 2024. arXiv:2406.06484.

[69] Di Zhang and Jiaqi Xing. Bayesian optimality of in-context learning with selective state spaces, 2026. arXiv:2602.17744.

## A Open Affine Control Lower Bound

This appendix concerns a different question from Theorem 4: whether the nonsaturating binary affine control has a nonvanishing lower bound at the much longer endpoint $\lceil 8 / q \rceil$ . The section on learned models establishes that separation only by measurement; the final analytic step for the joint event below remains open. The mechanism is sluggishness: after a latent switch, the nonsaturating affine recurrence carries an unclamped state near its AR(1) fixed point $d / ( 2 q )$ with $d = 2 \mu ^ { 2 } / \sigma ^ { 2 }$ , and needs $\Theta ( 1 / { q } )$ steps to change sign, while exact Bayes recovers on a shorter timescale for building confidence. Simulation localizes the cost accordingly: the fraction on the wrong side relative to the truth is between 0.18 and 0.20 across every q tested. The subset on which Exact Bayes is confident is reported separately below. Disagreement with the exact filter was not measured. Table 3 separates the proved ingredients from the remaining step for the terminal joint event. For the concentration calculation, the centered oriented evidence has variance proxy $v = 2 d ,$ , and at horizon t we choose the deterministic prefix budget $B _ { t } = t d .$

Table 3: The lower bound per step, step by step. Every deterministic and probabilistic ingredient is proved; the banded final row is the one step still open, and the separation at long horizons is not claimed without it.
<table><tr><td>Step</td><td>Statement</td><td>Status</td></tr><tr><td colspan="3">Deterministic core</td></tr><tr><td>Cost of a step on the wrong side</td><td>a confident exact predictive against a model on the wrong side costs  $\geq 3 / 8 0$  nats</td><td>proved</td></tr><tr><td>Affine state</td><td> $\begin{array} { r } { h _ { t } = a ^ { t } h _ { 0 } + \sum _ { i < t } a ^ { t - 1 - i } e _ { i } } \end{array}$ </td><td>proved</td></tr><tr><td>Summation by parts</td><td> $| S _ { j } | \le B _ { t }$  gives weighted sum  $\geq - 2 B _ { t }$ </td><td>proved</td></tr><tr><td>Combining the bounds</td><td>no crossing by step t, and that step costs  $\geq 3 / 8 0$ </td><td>conditional on the margin</td></tr><tr><td>Probabilistic tail</td><td></td><td></td></tr><tr><td>Gaussian sub-Gaussianity Prefix tail on a block</td><td>a centered real Gaussian is sub-Gaussian with constant its variance on a block with fixed latent signs, some prefix exceeds  $B _ { t }$  with measure at</td><td>proved here proved, discharges</td></tr><tr><td>Terminal joint event</td><td> $2 ( t + 1 ) e ^ { - B _ { t } ^ { 2 } / ( 2 t v ) }$  most uniform probability that the terminal step is on the wrong side while</td><td>the tail</td></tr><tr><td></td><td>exact Bayes is confident</td><td>open</td></tr></table>

With $v = 2 d$ and $B _ { t } = t d ,$ , the charge from the prefix tail is $2 ( t { + } 1 ) e ^ { - t d / 4 }$ . Instantiating the affine recurrence at $a = 1 - 2 q$ and $h _ { 0 } = d / ( 2 q )$ ) reduces the condition for no crossing $2 B _ { t } < a ^ { t } h _ { 0 }$ to

$$
4 q t < ( 1 - 2 q ) ^ { t } .
$$

Writing $y = q t$ and taking $q  0$ gives the boundary equation $4 y = e ^ { - 2 y }$ , whose unique positive root is $y _ { \star } =$ 0.175866 . . .. Hence $t ^ { * } q \to y _ { \star }$ , and stationarity gives limiting mass of the run age $1 - e ^ { - y _ { \star } } = 0 . 1 6 1 2 7 0 \ldots$ Both constants are analytic, independent of q, and carry no sampling error. The latter is close to the measured fraction on the wrong side relative to the truth, 0.18 to 0.20 across $q = 2 ^ { \overset { . } { - } 5 } { \mathrm { ~ t o ~ } } 2 ^ { - 1 2 }$ under the convention after mixing of Table 2; the bound from summation by parts loses a factor of two, so the analytic constant need not match the observed fraction.

What the open step actually requires. The estimand at the long endpoint is the predictive KL at the final step, so no integration along a trajectory is needed: it suffices to lower bound, uniformly in q, the probability that the terminal step is a step on the wrong side, since the cost per step above then gives $\mathbb { E } [ \mathrm { K L } _ { T } ] \stackrel { \cdot } { \geq } ( \dot { 3 } / 8 0 ) ^ { \bullet } \mathbb { P } ( W _ { \mathrm { j o i n t } } )$ . Stationarity supplies that bridge exactly. Writing $A _ { T }$ for the age of the current latent run, memorylessness gives $\mathbb { P } ( A _ { T } \geq k ) = ( 1 - q ) ^ { k }$ hence $\mathbb { P } ( \breve { A } _ { T } \leq t ^ { * } ) \to 1 - e ^ { - 0 . 1 7 6 }$ , which is the 0.161 already reported. What remains open is therefore a uniform bound on the joint event that the run is young enough for the affine state not to have crossed and old enough for the exact filter to be confident, not an integration.

Why a stepwise union bound is insufficient. Charging the worst case at every step forces the tolerance per step to grow like $\sqrt { \log ( 1 / q ) }$ , shrinking the guaranteed horizon before crossing to $\Theta ( 1 / ( q \sqrt { \log ( 1 / q ) } ) )$ and making the resulting probability of the switch window tend to zero. This decay conflicts with the measured nearly constant fraction on the wrong side. Controlling the partial sum instead preserves a constant probability of the switch window.

## B Geometry of Centered Logits and an Example with Three States

This section gives a geometric reading of Definition 1. It is not an additional assumption and is not used as a substitute for the proof. Its purpose is to make clear what the state dimension, centering operation, and radial nonlinearity mean when $\bar { K } > 2$

## B.1 What K counts

The integer K counts latent modes, not observations, layers, or time steps. A machine with normal, overheated, and failed modes has $K = 3$ even if it is monitored for a million steps. At time t, a filter assigns a probability vector

$$
p _ { t } = ( p _ { t } ( 0 ) , \ldots , p _ { t } ( K - 1 ) ) , \qquad p _ { t } ( s ) \geq 0 , \quad \sum _ { s } p _ { t } ( s ) = 1 .
$$

The vector has K entries but only $K - 1$ degrees of freedom because its entries sum to one. Logits make the same redundancy explicit: for every scalar b,

$$
\operatorname { s o f t m a x } ( z + b \mathbf { 1 } ) = \operatorname { s o f t m a x } ( z ) .
$$

Thus z and $z + b \mathbf { 1 }$ represent the same belief. Centering chooses one representative from each equivalence class,

$$
\mathcal { C } z = z - \frac { \mathbf { 1 } ^ { \top } z } { K } \mathbf { 1 } , \qquad \mathbf { 1 } ^ { \top } \mathcal { C } z = 0 .
$$

The meaningful filter state therefore lies in the $( K - 1 )$ -dimensional hyperplane orthogonal to 1. Pairwise log odds $z _ { i } - z _ { j }$ are coordinates on this quotient space. This is why every mismatch in Definition 2 is stated through pairwise differences rather than raw logits.

It is useful to keep two scale parameters separate. The state count K fixes the dimension of the belief geometry. The quantity

$$
L _ { K } ( q ) = \log \frac { ( K - 1 ) ( 1 - q ) } { q }
$$

is a confidence scale determined by the switch probability. For fixed $K$ , rare switching means $q \to 0 ^ { + }$ and hence $L _ { K } ( q ) \to \infty$ . The theorem takes this limit after fixing K; it does not let the number of modes grow with rarity.

## B.2 Direction records preference; radius records confidence

Within the centered hyperplane, regard z as an arrow. Its direction records the pattern of relative preferences among states. Its length records how strongly the filter prefers that pattern. Our convention $r ( z ) = \sqrt { 2 } \| \mathcal { C } z \| _ { 2 }$ makes this length agree with the absolute log odds in the binary specialization.

For a concrete $K = 3$ belief, take

$$
z = ( 2 , 0 , - 2 ) , \qquad \mathrm { s o f t m a x } ( z ) \approx ( 0 . 8 6 7 , 0 . 1 1 7 , 0 . 0 1 6 ) .
$$

State 0 leads state 1, which leads state 2. If a radial operation halves the arrow, it produces $( 1 , 0 , - 1 )$ and probabilities approximately (0.665, 0.245, 0.090). The ranking and all ratios between pairwise gaps are unchanged, but the confidence is lower. This example illustrates the geometry only; the actual radial factor is chosen by q and by the current radius.

More precisely, the radial transition can be written

$$
R _ { q } ( z ) = \beta _ { q } ( z ) \mathcal { C } z , \qquad \beta _ { q } ( z ) = \frac { L _ { K } ( q ) \operatorname { t a n h } ( \alpha _ { K } ( q ) r ( z ) / L _ { K } ( q ) ) } { r ( z ) } .
$$

It preserves the direction of every nonzero centered state during the transition mixing step and changes only its radius. It is not an orthogonal projection: there is no fixed subspace onto which the state is dropped, and the multiplier depends nonlinearly on the current radius. Near the origin, $\begin{array} { r } { R _ { q } \dot { ( } z ) \approx \alpha _ { K } ( q ) \mathcal { C } z ; } \end{array}$ far from the origin, its radius saturates below $L _ { K } ( q )$ . The next observation then adds the centered score $ { \mathcal { C } } g ( x )$ , which can change both direction and length. A trajectory can therefore turn many times even though each isolated radial mixing step preserves direction.

Table 4: Geometric dictionary for the fixed-K result. Observation scores are added after the transition map and may rotate the state.
<table><tr><td>Object</td><td>Operational meaning</td></tr><tr><td>K</td><td>Number of possible hidden modes; the centered belief state has dimension  $\hat { K } - 1$ </td></tr><tr><td>q and  $L _ { K } ( q )$ </td><td>Switch rarity and its associated confidence scale;  $L _ { K } ( q )$  grows as switches become rarer.</td></tr><tr><td> $\mathcal { C } z$ </td><td>Logit state after removing the common offset that does not affect prediction.</td></tr><tr><td>r(z)</td><td>Overall confidence magnitude, not a state count and not a time horizon.</td></tr><tr><td> $R _ { q }$ </td><td>Nonlinear confidence cap that preserves centered direction dur- ing transition mixing; it is not a linear projection.</td></tr></table>

## B.3 What Exact Bayes does differently

Exact transition mixing acts on probabilities before returning to centered logits. It adjusts the coordinates according to the full categorical mixture $P _ { q } ^ { \neg } p ,$ , not through one shared radial multiplier. Except in special directions, it need not preserve the centered arrow’s direction. The radial filter deliberately discards this geometry specific to each coordinate and retains only a common confidence contraction followed by the same observation score.

The two main theorems test different consequences of that difference. Theorem 3 selects an explicit moving logit $z _ { q }$ and proves that the exact and radial maps separate by order $L _ { K } ( q )$ in centered coordinates. This rules out the explanation that the two update maps merely converge to one another. Theorem 4 then couples both filters to the same stationary observation path and proves that their decoded predictive distributions nevertheless converge in expected KL at the logarithmic horizon.

The apparent paradox is resolved by the softmax decoder. A centered displacement is expensive near a decision boundary, where probabilities are sensitive to logits. It can be cheap deep inside a shared confidence cone, where both filters put almost all mass on the same state. In the example with three states, moving from $( 1 0 , 0 , - 1 0 )$ to $( 6 , 0 , - 6 )$ is a large internal change, yet both decoded vectors are overwhelmingly concentrated on state 0. The proof formalizes precisely this regime inside the common cone rather than asserting that every large logit difference is harmless.

## B.4 A dictionary for one step

The following dictionary separates objects that are easy to conflate.

## C Proof Roadmap for Fixed K

This section is a reading guide to Appendix E. The quantitative theorem at the level of the maps and the stationary theorem under the path law share an endpoint set by softmax curvature but answer different questions. The first follows from an explicit centered witness and direct asymptotics. The second has a probabilistic spine and two recovery arguments specific to the dimension.

Dependency order. Theorem 3 first compares the exact and radial gaps for the chosen pair at $z _ { q } ,$ , then invokes the curvature lemma for the common cone. For Theorem 4 and $K \geq 3 .$ , Lemma 8 and the Gaussian maximal event feed the radial barriers and Lemma 13; separately, the calculation over path mixtures yields Lemma 16. These two margins meet only in Lemma 19, with the deterministic envelope controlling the complement. The binary branch replaces the radial barrier step with common scalar recovery and then rejoins the same curvature and envelope argument.

## C.1 Proof outline

Both filters start in the same centered belief coordinates and consume the same observations. The proof under the path law first conditions on a constant latent block. It then shows that exact Bayes and the radial recurrence identify the true state with diverging margins. A deterministic lemma on the categorical KL converts that common confidence cone into an exponentially small loss on the good event, while a $2 L _ { K } ( q )$ envelope controls every remaining path.

![](images/8ed562eb60f0d071e796aaab6060393c0858045946ef012c89e5bdbe6566d222.jpg)

![](images/50b0f8a4f3b41140421cb65c5fc45ad4cc13b7152764a1230b44c510f44925fa.jpg)  
Fixed q = 1/8; full 1601-point grid evaluated.

![](images/dc450c9abebad2002b8485dacab33a83fde6321a62be186596d868b34f2589d8.jpg)  
Chord midpoint error: Δq/2 = . 02613. Any affine map: max error $\ge \Delta _ { q } / 4 = . 0 1 3 0 6 .$

Figure 3: Binary transition geometry at $q = 1 / 8 . \mathrm { A } \mathrm { : }$ signed residual of exact minus tanh. B: the maps at true scale share saturation levels and midpoint slope. C: values at 0, log 2, 2 log 2 certify that the map is not affine; their second difference $\Delta _ { q }$ forces every affine approximation to incur maximum error at least $\Delta _ { q } / 4 .$ This is a fixed-q, $K = 2$ diagnostic.

## C.2 The $K \geq 3$ branch

On a block with no switch in state s, the centered Gaussian score is $a _ { s } + w Z _ { t }$ . Because radial mixing uses one adaptive scalar coefficient, the approximate state has the exact representation $z _ { t } ^ { R } = U _ { t } a _ { s } + V _ { t } w$ . Abel summation bounds every V by twice one maximum of a Gaussian partial sum, so a single maximal inequality controls the whole horizon without a temporal union bound. Projection onto $w ^ { \perp }$ supplies a deterministic signal direction; upper and lower radial barriers then give margins for the true state of order $L _ { K } ( \boldsymbol { q } ) ^ { 2 / 3 }$ . Recovery of the exact filter is established separately by comparing the constant path in the true state with constant wrong paths and the aggregate Bayes factor of all alternatives that contain a switch.

## C.3 The binary branch

For $K = 2$ , the signal and noise directions are collinear, so the projection step is unavailable. Midpoint translation reduces arbitrary distinct means to the symmetric model. An event on the prefix evidence forces both scalar recurrences to hit a common margin, and a geometric sum over possible hit locations controls suffix persistence. Figure 3 visualizes the transition maps used by this branch. The argument unions suffix lengths, not transition errors at one step, and its failure probability remains negligible after multiplication by the KL envelope.

## C.4 Conditioning argument

There are three changes of probability law. First, the event that no switch occurs costs at most $H ( q ) q$ . Second, conditional on that event, Gaussian concentration applies to independent emissions from the fixed latent state. Third, the exact posterior still sums over every latent path; candidate paths that contain a switch are included and controlled through their aggregate Bayes factor. The final expectation is recovered by averaging the finitely many conditional bounds over the stationary uniform starting state.

## D Proof Anatomy, Constants, and Quantifiers

This section explains the main steps in the proof of Theorem 4 and the dependence of its constants on the model parameters. Full proofs are given in Appendix E.

## D.1 Quantifier order

Fix K, the distinct means $\mu _ { 0 } , \ldots , \mu _ { K - 1 }$ , the variance $\sigma ^ { 2 }$ , and a horizon constant $c < d _ { \operatorname* { m i n } }$ , where

$$
d _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i \neq j } \frac { ( \mu _ { i } - \mu _ { j } ) ^ { 2 } } { 2 \sigma ^ { 2 } } > 0 .
$$

Only then send $q \to 0 ^ { + }$ . Constants denoted $C _ { K } , c _ { s } , \kappa ,$ , or a small-q threshold may depend on these fixed parameters. This permits a finite union over states and competitors, but it does not provide a bound uniform in K or in a family whose means approach one another. The theorem has the logical form

$$
\forall K < \infty \forall ( \mu _ { 0 } , \ldots , \mu _ { K - 1 } ) { \mathrm { ~ d i s t i n c t } } , \qquad \operatorname* { l i m } _ { q \to 0 ^ { + } } \mathbb { E } D _ { \mathrm { K L } } ( p _ { H } ^ { E } \| p _ { H } ^ { R } ) = 0 ,
$$

not a joint limit over K and $q .$

## D.2 The probabilistic argument for $K \geq 3$

Condition on a block with no switch whose true state is s. The centered Gaussian score has the exact form with two vectors

$$
\begin{array} { r } { \mathcal { C } g ( X _ { t + 1 } ) = a _ { s } + w Z _ { t + 1 } , } \end{array}
$$

so the radial recurrence stays in the random plane spanned by $a _ { s }$ and w:

$$
z _ { t } ^ { R } = U _ { t } a _ { s } + V _ { t } w , \qquad U _ { t + 1 } = \beta _ { t } U _ { t } + 1 , \quad V _ { t + 1 } = \beta _ { t } V _ { t } + Z _ { t + 1 } .
$$

This does not reduce the filter to two states. It reduces the analysis of one conditioned block to one accumulated signal coefficient $U _ { t }$ and one accumulated noise coefficient $V _ { t }$

The multiplier $\beta _ { t }$ depends on the past, so the noise is not a deterministic weighted Gaussian sum. The key observation is order, not independence: for fixed terminal t, the products of positive multipliers attached to later observations are nondecreasing. Abel summation therefore gives the pathwise inequality

$$
\left| V _ { t } \right| \leq 2 \operatorname* { m a x } _ { u \leq H } \left| \sum _ { k = 1 } ^ { u } Z _ { k } \right|
$$

simultaneously for all t. One maximal bound from an exponential martingale controls this maximum. The proof does not apply a separate tail bound at every time and then union bound over the horizon.

On the resulting event, $| V _ { t } | = O ( { \sqrt { L \log L } } )$ . The tanh expansion supplies an upper barrier $U _ { t } = O ( L ^ { 2 / 3 } )$ and a matching lower bound after an initial transient of order $L ^ { 2 / 3 }$ . The exponent $2 / 3$ is the balance at which the unit signal increment competes with the cubic correction in tanh $x = x - \Theta ( x ^ { \hat { 3 } } )$ . It is sufficient for recovery because

$$
z _ { t , s } ^ { R } - z _ { t , j } ^ { R } = d _ { s j } U _ { t } + \eta _ { s j } V _ { t } , \qquad d _ { s j } = \frac { ( \mu _ { s } - \mu _ { j } ) ^ { 2 } } { 2 \sigma ^ { 2 } } > 0 , \quad \eta _ { s j } = \frac { \mu _ { s } - \mu _ { j } } { \sigma } ,
$$

and $L ^ { 2 / 3 }$ dominates ${ \sqrt { L \log L } }$

Exact Bayes is handled separately. The constant-s latent path is the likelihood reference. Constant wrong paths lose exponentially in $L ,$ , while the aggregate Bayes factor of all switched paths has expectation of order $H q .$ . This is a calculation over path mixtures; it does not assume independence between posterior coordinates. Because $H = \Theta ( L )$ and $q = \Theta ( e ^ { - L } )$ for fixed $K ,$ , mass of the switched paths vanishes.

## D.3 Bounding predictive KL

The radial and exact arguments meet only after both filters have entered a common confidence cone. On the good event, every margin of the true state over a wrong state is at least $c _ { * } L ^ { 2 / 3 }$ . Along the line segment between the two terminal logits, softmax curvature is then exponentially small. Combining this curvature with the deterministic terminal range bound yields

$$
D _ { \mathrm { K L } } ( p _ { H } ^ { E } \Vert p _ { H } ^ { R } ) \leq C _ { K } L ^ { 2 } e ^ { - c _ { * } L ^ { 2 / 3 } } .
$$

The bound on this event alone is not enough because KL is unbounded in arbitrary logits. Lemma 18 supplies a pathwise 2L envelope, which is multiplied by the vanishing probability of the complement. Combining the bound on this event with the uniform bound on its complement turns recovery with high probability into convergence of expected KL.

## D.4 Why the binary branch is separate

For $K \geq 3 ,$ , distinct scalar means imply that $a _ { s }$ cannot be collinear with the common noise direction $w \mathrm { : }$ otherwise a nonzero quadratic would agree with an affine function at at least three distinct points. This makes the radial radius control available. With two points, that polynomial contradiction disappears. The binary proof instead translates arbitrary means to the symmetric pair $( - \delta , + \bar { \delta } )$ and controls both scalar recurrences through a common recovery event. It then rejoins the same KL curvature and envelope lemmas. The separate branch is an issue of proof geometry, not a restriction to symmetric models with two states.

## D.5 Rates proved and rates not claimed

The witness at the level of the maps and stationary path result have different rates and should not be merged. At the explicit witness, centered map separation is asymptotically $c _ { \star } L .$ , and decoder KL is bounded by a polynomial in L times $e ^ { - L / 5 }$ . Along typical blocks with no switch in the stationary proof, the radial margin for the true state is only required to grow as $L ^ { \overline { { 2 } } / \overline { { 3 } } }$ , which still makes the decoder cost on the good event vanish. Neither statement identifies an optimal exponent, a sharp finite-q onset, or constants uniform over state geometries. The curves with equally spaced Gaussians in Appendix F are therefore illustrations rather than rate estimates.

## E Proofs of the Main Results

This appendix proves Theorems 3 and 4 for every fixed finite $K \geq 2 .$ . The quantitative theorem at the level of the maps uses an explicit centered witness and a direct asymptotic calculation. The stationary theorem uses a representation in two scalars of the radial filter, one Gaussian maximal event, and an analysis over path mixtures of exact Bayes for $K \geq 3 ;$ the binary case uses the same KL bounds after midpoint translation and a recovery argument in one dimension.

## E.1 Preliminaries

Write $T = - \log q , \alpha = \alpha _ { K } ( q )$ , and $L = L _ { K } ( q )$ . For fixed K,

$$
L = T + \log ( K - 1 ) + \log ( 1 - q ) = T + O ( 1 ) .
$$

Consequently, for all sufficiently small $q , L > 1 , \alpha \geq 1 / 2$ , and $H \leq C _ { H } L$ . We will also use

$$
q = \frac { K - 1 } { e ^ { L } + K - 1 } \leq ( K - 1 ) e ^ { - L } , \qquad H q \longrightarrow 0 , \qquad ( 1 - q ) ^ { - ( H - 1 ) } \longrightarrow 1 .\tag{10}
$$

Fix a state s and let

$$
{ \mathcal { N } } _ { s } = \{ S _ { 0 } = \cdot \cdot \cdot = S _ { H - 1 } = s \} .
$$

Conditional on $\mathcal N _ { s } , X _ { t + 1 } = \mu _ { s } + \sigma Z _ { t + 1 } \mathrm { f o r } t = 0 , \dots , H - 1$ , where the $Z _ { t }$ are independent standard Gaussians. With

$$
a _ { s } = \mathcal { C } \left( \frac { \mu _ { s } \pmb { \mu } } { \sigma ^ { 2 } } - \frac { \pmb { \mu } ^ { \odot 2 } } { 2 \sigma ^ { 2 } } \right) , \qquad w = \mathcal { C } \left( \frac { \pmb { \mu } } { \sigma } \right) ,\tag{11}
$$

the centered score is $g ( X _ { t + 1 } ) = a _ { s } + w Z _ { t + 1 }$

Lemma 7 (Radial dynamics in two scalars). On $\mathcal { N } _ { s } ,$ adapted scalar processes $U _ { t } , V _ { t }$ satisfy

$$
\begin{array} { r c l } { { } } & { { } } & { { z _ { t } ^ { R } = U _ { t } a _ { s } + V _ { t } w , \qquad U _ { 0 } = V _ { 0 } = 0 , } } \\ { { } } & { { } } & { { U _ { t + 1 } = \beta _ { t } U _ { t } + 1 , \qquad V _ { t + 1 } = \beta _ { t } V _ { t } + Z _ { t + 1 } , } } \end{array}\tag{12}
$$

where

$$
\beta _ { t } = \frac { L \operatorname { t a n h } ( \alpha r ( z _ { t } ^ { R } ) / L ) } { r ( z _ { t } ^ { R } ) }
$$

with continuous value α at the origin. Moreover, $0 < \beta _ { t } \le \alpha < 1$ and $U _ { t } \geq 0 .$

Proof. The radial transition multiplies every centered coordinate by the same scalar $\beta _ { t }$ . Substitution of (11) proves (12) by induction. Positivity is immediate, while tanh $x \leq x \mathrm { { g i v e s } } \beta _ { t } \leq \alpha$ □

## E.2 One maximal event controls the adaptive noise

Let $\begin{array} { r } { S _ { u } = \sum _ { k = 1 } ^ { u } Z _ { k } } \end{array}$

Lemma 8 (Adaptive Abel bound). For every realized path and every $t \leq H$

$$
\vert V _ { t } \vert \leq 2 \operatorname* { m a x } _ { u \leq H } \vert S _ { u } \vert .\tag{13}
$$

Proof. Expanding (12) gives

$$
V _ { t } = \sum _ { k = 1 } ^ { t } \gamma _ { k , t } Z _ { k } , \qquad \gamma _ { k , t } = \prod _ { \ell = k } ^ { t - 1 } \beta _ { \ell } , \qquad \gamma _ { t , t } = 1 .
$$

Although the weights are adaptive, they obey $0 \leq \gamma _ { 1 , t } \leq \cdot \cdot \cdot \leq \gamma _ { t , t } = 1$ pathwise. Abel summation therefore gives

$$
V _ { t } = S _ { t } - \sum _ { k = 1 } ^ { t - 1 } ( \gamma _ { k + 1 , t } - \gamma _ { k , t } ) S _ { k } .
$$

The coefficients in the second term are nonnegative and sum to at most one, proving (13). This pathwise bound also applies to adaptive weights. □

Lemma 9 (Gaussian maximal event). Set

$$
x _ { L } = \sqrt { 2 H \log ( 2 L ^ { 3 } ) } , \qquad { \mathcal { E } } _ { L } = \left\{ \operatorname* { m a x } _ { u \leq H } | S _ { u } | \leq x _ { L } \right\} .
$$

Then $\operatorname* { P r } ( \mathcal { E } _ { L } ^ { c } ) \le L ^ { - 3 }$ . On $\mathcal { E } _ { L } ,$ ,

$$
\operatorname* { m a x } _ { t \leq H } | V _ { t } | \leq M _ { L } : = 2 x _ { L } = O ( { \sqrt { L \log L } } ) .\tag{14}
$$

Proof. For every $\lambda > 0 , \exp ( \lambda S _ { u } - u \lambda ^ { 2 } / 2 )$ is a nonnegative martingale. Doob’s maximal inequality, optimized at $\lambda = x / H$ , yields

$$
\operatorname* { P r } \left( \operatorname* { m a x } _ { u \leq H } S _ { u } > x \right) \leq e ^ { - x ^ { 2 } / ( 2 H ) } .
$$

Apply the same argument $\mathrm { t o } - S _ { u }$ and union only the two signs. Substituting $x = x _ { L }$ and applying Lemma 8 proves the claim simultaneously over the full horizon. □

## E.3 Radial recovery when $K \geq 3$

Assume $K \geq 3 ,$ and let $P$ be Euclidean projection onto $w ^ { \perp }$

Lemma 10 (Noncollinear signal). For every state s, $P a _ { s } \neq 0 .$

Proof. Distinct means imply $w \ne 0$ . If $P a _ { s } = 0 { . }$ , centeredness would give $a _ { s } = \lambda w$ . Hence a constant b would satisfy

$$
{ \frac { \mu _ { s } \mu _ { i } } { \sigma ^ { 2 } } } - { \frac { \mu _ { i } ^ { 2 } } { 2 \sigma ^ { 2 } } } = { \frac { \lambda \mu _ { i } } { \sigma } } + b
$$

for every i. After clearing denominators, a quadratic with nonzero quadratic coefficient would vanish at the $K \geq 3$ distinct values $\mu _ { i } .$ , a contradiction.

Set $D _ { s } = \sqrt { 2 } \| P a _ { s } \| _ { 2 } > 0 , A _ { s } = r ( a _ { s } )$ , and $W = r ( w )$ . From Lemma 7,

$$
D _ { s } U _ { t } \leq r ( z _ { t } ^ { R } ) \leq A _ { s } U _ { t } + W | V _ { t } | .\tag{15}
$$

Lemma 11 (Upper radial barrier). For each $s ,$ constants $C _ { U } , C _ { r } , q _ { s } > 0$ exist such that, on $\mathcal { E } _ { L }$ and for $q < q _ { s }$

$$
U _ { t } \le C _ { U } L ^ { 2 / 3 } , \qquad r ( z _ { t } ^ { R } ) \le C _ { r } L ^ { 2 / 3 }\tag{16}
$$

for all $t \leq H .$

Proof. The function $h ( x ) = \operatorname { t a n h } ( x ) / x$ , continuously extended at zero, decreases on $[ 0 , \infty )$ . Using the lower bound in (15),

$$
\beta _ { t } U _ { t } = \alpha h ( \alpha r ( z _ { t } ^ { R } ) / L ) U _ { t } \le \frac { L } { D _ { s } } \operatorname { t a n h } ( \alpha D _ { s } U _ { t } / L ) .
$$

Let $f _ { L } ( u ) = 1 + ( L / D _ { s } )$ tanh $\left( \alpha D _ { s } u / L \right)$ . Choose $C _ { U }$ so that $D _ { s } ^ { 2 } C _ { U } ^ { 3 } > 6 4$ , and put $R = C _ { U } L ^ { 2 / 3 }$ . For large $L ,$ $\alpha D _ { s } R / L \leq 1$ . Since tanh $x \leq \dot { x } - x ^ { 3 } / 8$ on [0, 1],

$$
f _ { L } ( R ) - R \leq 1 - ( 1 - \alpha ) R - \frac { \alpha ^ { 3 } D _ { s } ^ { 2 } C _ { U } ^ { 3 } } { 8 } \leq 1 - \frac { D _ { s } ^ { 2 } C _ { U } ^ { 3 } } { 6 4 } < 0 .
$$

The map $f _ { L }$ is increasing, $U _ { 0 } \leq R ,$ , and $U _ { t + 1 } \leq f _ { L } ( U _ { t } )$ , so induction on the first crossing yields $U _ { t } \leq R$ . The upper bound in (15), together with (14) and $M _ { L } = o ( L ^ { 2 / 3 } )$ , gives the radius bound. □

Lemma 12 (Lower memory after the initial transient). There are constants $c _ { U } , C _ { 0 } , q _ { s } ^ { \prime } > 0$ such that, on $\mathcal { E } _ { L }$ and for $q < q _ { s } ^ { \prime }$

$$
U _ { t } \geq c _ { U } L ^ { 2 / 3 }\tag{17}
$$

whenever $C _ { 0 } L ^ { 2 / 3 } \leq t \leq H .$

Proof. The global inequality tanh x $\geq x - x ^ { 3 } / 3$ and Lemma 11 imply

$$
\beta _ { t } \geq \alpha - \frac { \alpha ^ { 3 } r ( z _ { t } ^ { R } ) ^ { 2 } } { 3 L ^ { 2 } } \geq 1 - \varepsilon _ { L } , \qquad \varepsilon _ { L } = \frac { K q } { K - 1 } + \frac { C _ { r } ^ { 2 } } { 3 L ^ { 2 / 3 } } .\tag{18}
$$

For small $q , 0 < \varepsilon _ { L } < 1$ and $b _ { 1 } L ^ { - 2 / 3 } \leq \varepsilon _ { L } \leq b _ { 2 } L ^ { - 2 / 3 }$ for positive constants $b _ { 1 } , b _ { 2 }$ . Iterating (12) gives

$$
U _ { t } \geq \frac { 1 - ( 1 - \varepsilon _ { L } ) ^ { t } } { \varepsilon _ { L } } .
$$

For $t \ge \lceil ( \log 2 ) / \varepsilon _ { L } \rceil$ , the numerator is at least $1 / 2 ,$ , which proves (17). This transient is $O ( L ^ { 2 / 3 } )$ and is smaller than $H = \Theta ( \dot { L } )$ □

Lemma 13 (Simultaneous radial recovery). For every s, constants $c _ { s } , q _ { s } ^ { \prime \prime } > 0$ exist such that, on $\mathcal { E } _ { L }$ and $f o r q < q _ { s } ^ { \prime \prime }$

$$
z _ { t , s } ^ { R } - z _ { t , j } ^ { R } \geq c _ { s } L ^ { 2 / 3 }\tag{19}
$$

for every $j \neq s$ and $C _ { 0 } L ^ { 2 / 3 } \leq t \leq H$

Proof. Direct calculation gives

$$
( a _ { s } ) _ { s } - ( a _ { s } ) _ { j } = d _ { s j } : = \frac { ( \mu _ { s } - \mu _ { j } ) ^ { 2 } } { 2 \sigma ^ { 2 } } > 0 , \qquad w _ { s } - w _ { j } = \eta _ { s j } : = \frac { \mu _ { s } - \mu _ { j } } { \sigma } .
$$

Thus $z _ { t , s } ^ { R } - z _ { t , j } ^ { R } = d _ { s j } U _ { t } + \eta _ { s j } V _ { t }$ . The first term is bounded below by a positive multiple of $L ^ { 2 / 3 }$ , uniformly over the finitely many competitors, whereas the second is $O ( \sqrt { L \log L } ) = o ( L ^ { 2 / 3 } )$ by (14). □

## E.4 Recovery of the exact filter by path mixtures

Continue to condition on $\mathcal { N } _ { s }$ . The exact posterior after H observations is the normalized sum of prior times likelihood over latent paths. Use the constant path $( s , \ldots , s )$ as reference; its prior mass is

$$
\pi _ { * } = \frac { 1 } { K } ( 1 - q ) ^ { H - 1 } .\tag{20}
$$

Lemma 14 (Constant wrong paths). There are $\kappa , q _ { s } > 0$ such that, for $q < q _ { s }$ , with conditional probability at least $1 - ( K - 1 ) e ^ { - \kappa L }$ , every constant path $j \neq s$ has likelihood ratio relative to the reference at most $\bar { e } ^ { - L / 2 }$

Proof. For $j \neq s ,$ , let

$$
Y _ { t } ^ { s , j } = \log \frac { \varphi _ { \mu _ { s } , \sigma } ( X _ { t } ) } { \varphi _ { \mu _ { j } , \sigma } ( X _ { t } ) } .
$$

Under $\mathcal { N } _ { s }$ , these variables are independent Gaussians with mean $d _ { s j }$ and variance $2 d _ { s j }$ . Since $H d _ { s j } \geq ( d _ { \operatorname* { m i n } } / c ) T$ with $d _ { \operatorname* { m i n } } / c > 1$ , while $L = T + O ( 1 )$ , a bound on the Gaussian lower tail gives

$$
\operatorname* { P r } \left( \sum _ { t = 1 } ^ { H } Y _ { t } ^ { s , j } < L / 2 \right) \leq e ^ { - \kappa _ { s j } L }
$$

for small q. A finite union over $j \neq s$ proves the claim.

Lemma 15 (Bayes factor of a switched path). Let $\mathcal { P } _ { \mathrm { s w } }$ be the latent paths containing at least one switch, let $\pi ( p )$ denote their prior masses, and let $\mathrm { L R } _ { p }$ be likelihood relative to the constant-s path. For

$$
\begin{array} { c } { { \displaystyle B _ { \mathrm { s w } } = \frac { \sum _ { p \in \mathcal { P } _ { \mathrm { s w } } } \pi ( p ) \mathrm { L R } _ { p } } { \pi _ { * } } , } } \\ { { \displaystyle \mathbb { E } [ B _ { \mathrm { s w } } \mid \mathcal { N } _ { s } ] \leq \frac { K H q } { ( 1 - q ) ^ { H - 1 } } , \qquad \operatorname* { P r } ( B _ { \mathrm { s w } } > e ^ { - L / 2 } \mid \mathcal { N } _ { s } ) \leq C L e ^ { - L / 2 } . } } \end{array}\tag{21}
$$

Proof. Under the reference observation law, every likelihood ratio has expectation one. The numerator in the first expectation is therefore the prior mass of paths that contain a switch, at most $H q .$ . Division by (20) gives the first bound. Markov’s inequality at threshold $e ^ { - L / 2 }$ , followed by (10), gives the second. □

Lemma 16 (Exact posterior margin). Conditional on $\mathcal { N } _ { s }$ , outside an event of probability at most

$$
( K - 1 ) e ^ { - \kappa L } + C L e ^ { - L / 2 } ,\tag{22}
$$

the exact terminal logits satisfy

$$
z _ { H , s } ^ { E } - z _ { H , j } ^ { E } \geq L / 2 - \log K\tag{23}
$$

for every $j \neq s .$

Proof. On the intersection of the events in Lemmas 14 and 15, total nonreference weight divided by reference weight is at most $K e ^ { - L / 2 }$ . Hence $p _ { H } ^ { E } ( s ) / p _ { H } ^ { E } ( j ) \geq e ^ { L / 2 } / K$ for every $j \neq s ,$ , and taking logarithms gives (23). □

## E.5 Categorical KL bounds

For vectors u, v, put $\Delta = u - v { \mathrm { ~ a n d ~ r a n g e } } ( \Delta ) = \operatorname* { m a x } _ { i } \Delta _ { i } - \operatorname* { m i n } _ { i } \Delta _ { i }$

Lemma 17 (Range envelope). For every $u , v \in \mathbb { R } ^ { K }$

$$
D _ { \mathrm { K L } } ( \operatorname { s o f t m a x } u \| \operatorname { s o f t m a x } v ) \leq \operatorname { r a n g e } ( u - v ) .\tag{24}
$$

Proof. Let $A ( z ) = \log \textstyle \sum _ { i } e ^ { z _ { i } }$ . The increment $A ( u ) - A ( v )$ lies between min<sub>i</sub> $\Delta _ { i }$ and max<sub>i</sub> $\Delta _ { i }$ . Thus every log probability ratio is at most range(∆), and averaging under softmax(u) proves (24). □

Lemma 18 (Terminal envelope). For every observation path,

$$
D _ { \mathrm { K L } } ( \mathrm { s o f t m a x } z _ { H } ^ { E } \| \mathrm { s o f t m a x } z _ { H } ^ { R } ) \leq 2 L .\tag{25}
$$

Proof. Let $m ^ { E } = \Phi _ { q } ( z _ { H - 1 } ^ { E } )$ and $m ^ { R } = R _ { q } ( z _ { H - 1 } ^ { R } )$ be the logits before the observation at the terminal step. The final shared score cancels, so $z _ { H } ^ { E } - z _ { H } ^ { R } = m ^ { E } - \bar { m } ^ { R }$ . Exact mixing places every probability coordinate in $[ q / ( K - 1 ) , 1 - q ]$ hence every exact pairwise score before the observation difference has magnitude at most L. The radial image has pairwise diameter strictly below L because its radial norm is below L. Therefore range $( m ^ { E } - m ^ { R } ) \leq 2 L$ , and (24) applies. □

Lemma 19 (Curvature on a common cone). Suppose a state s and m $\geq 0$ satisfy

$$
u _ { s } - u _ { j } \geq m , \qquad v _ { s } - v _ { j } \geq m
$$

for every $j \neq s ,$ and range $( u - v ) \leq R .$ . Then

$$
D _ { \mathrm { K L } } ( \operatorname { s o f t m a x } u \| \operatorname { s o f t m a x } v ) \leq K ( K - 1 ) R ^ { 2 } e ^ { - m } .\tag{26}
$$

Proof. Every point $u _ { \tau } = ( 1 - \tau ) u + \tau v$ has the same margin lower bound. I $p _ { \tau } = \mathrm { s o f t m a x } ( u _ { \tau } )$ , then $1 - p _ { \tau } ( s ) \leq$ $( K - 1 ) e ^ { - m }$ . The Hessian of log-sum-exp is dia $\mathrm { g } ( p _ { \tau } ) - p _ { \tau } p _ { \tau } ^ { \top }$ , so its operator norm is at most $2 ( K - 1 ) e ^ { - m }$ . The Hessian annihilates common shifts, and range $( u - v ) \leq R$ implies $\Vert \mathcal { C } ( u - v ) \Vert _ { 2 } ^ { 2 } \leq K R ^ { 2 }$ . The integral Taylor formula for the corresponding Bregman divergence now gives (26). □

## E.6 Binary specialization

When $K = 2 ,$ let

$$
m _ { 0 } = \frac { \mu _ { 0 } + \mu _ { 1 } } { 2 } , \qquad \delta = \frac { | \mu _ { 1 } - \mu _ { 0 } | } { 2 } .
$$

After relabeling if needed and translating observations by m<sub>0</sub>, the emission means are $- \delta , + \delta$ . This preserves the likelihood ratio, the latent law, and both recurrences for the pairwise logits. Their evidence drift and variance are

$$
d = { \frac { 2 \delta ^ { 2 } } { \sigma ^ { 2 } } } = { \frac { ( \mu _ { 1 } - \mu _ { 0 } ) ^ { 2 } } { 2 \sigma ^ { 2 } } } = d _ { \mathrm { m i n } } , \qquad \mathrm { V a r } ( Y _ { t } ) = 2 d .\tag{27}
$$

In pairwise log odds the two transition maps are

$$
F _ { q } ( h ) = 2 \operatorname { a r t a n h } ( ( 1 - 2 q ) \operatorname { t a n h } ( h / 2 ) ) , \qquad G _ { q } ( h ) = L \operatorname { t a n h } ( ( 1 - 2 q ) h / L ) .\tag{28}
$$

Lemma 20 (Binary common recovery). Let $r = \lceil T / c \rceil , H = r + 1 , a = 3 \log T ,$ , and $A = B = 5 a$ . There are events $B _ { q }$ such that

$$
\mathrm { P r } ( B _ { q } ) L ^ { 2 } \longrightarrow 0 ,\tag{29}
$$

and on $B _ { q } ^ { c }$ the exact and radial posterior log odds at time $H ,$ , oriented toward the block’s initial state, are both at least a.

Proof. For $x \geq 0$ , define the common budget for one step for the confidence loss

$$
\rho _ { q } ( x ) = { \mathrm { m a x } } \left\{ \log ( 1 - q + q e ^ { x } ) - \log ( 1 - q ) , 2 q x + { \frac { x ^ { 2 } } { L } } \right\} .\tag{30}
$$

The two entries bound the losses of $F _ { q }$ and $G _ { q } ,$ respectively, on a positive interval. Conditional on a block with no switch and its initial sign, the signed evidence variables $Y _ { 0 } , \ldots , Y _ { r }$ are independent $\mathcal { N } ( d , 2 d )$ . Put $N = r + 1$ and

$$
Q = L + A + r \rho _ { q } ( A ) , \qquad \Sigma _ { \mathrm { p r e } } = \exp \left[ - \frac { ( N d - Q ) ^ { 2 } } { 4 N d } \right] .\tag{31}
$$

$\textstyle \operatorname { I f } \sum _ { i = 0 } ^ { r } Y _ { i } > Q .$ , both recurrences must hit A: otherwise the lower bounds for one step $F _ { q } ( x ) \geq x - \rho _ { q } ( A )$ and $G _ { q } ( x ) \ge x - \rho _ { q } ( A )$ sum from the uniform reset to a terminal value strictly above A, a contradiction. After a hit, monotonicity and saturation give, for either map $M _ { q }$

$$
M _ { q } ( x ) \geq \operatorname* { m i n } \{ x , B \} - \rho _ { q } ( B ) .\tag{32}
$$

Thus the hit persists to margin a whenever every nonempty suffix obeys

$$
\sum _ { i = j } ^ { r } \bigl ( Y _ { i } - \rho _ { q } ( B ) \bigr ) > - ( A - a ) .\tag{33}
$$

For a suffix of length ℓ, the bound on the Gaussian lower tail is at most

$$
\exp \left[ - \frac { \{ A - a + \ell [ d - \rho _ { q } ( B ) ] \} ^ { 2 } } { 4 \ell d } \right] .
$$

Summing over possible hit locations, and using $\rho _ { q } ( 5 a ) \to 0$ , we have eventually $d - \rho _ { q } ( B ) \geq d / 2$ . With $A - a = 4 a$

$$
\frac { ( 4 a + \ell d / 2 ) ^ { 2 } } { 4 \ell d } \geq a + \frac { \ell d } { 1 6 } .
$$

The entire suffix sum is therefore at most $C _ { d } e ^ { - a } = C _ { d } T ^ { - 3 }$

For the prefix, choose $d _ { 0 } = ( c + d ) / 2$ . Eventually $Q \leq N d _ { 0 }$ , and

$$
L ^ { 2 } \Sigma _ { \mathrm { p r e } } \leq T ^ { 2 } \exp \left[ - \frac { r ( d - c ) ^ { 2 } } { 1 6 d } \right] \longrightarrow 0 .
$$

Removing the conditioning on no switch separately from the prefix and suffix bounds costs at most $2 N q ,$ and $2 N q L ^ { 2 } = O ( T ^ { 3 } e ^ { - T } )  0$ . Combining these three terms proves (29) and the deterministic hit and persist argument proves the common margin. □

## E.7 Completion of the main theorem

First let $K \geq 3$ . Conditional on $S _ { 0 } = s .$ , intersect the actual event of no switch $\mathcal { N } _ { s }$ , the maximal event $\mathcal { E } _ { L }$ , and the two events for exact recovery from Lemmas 14 and 15. On this good event, Lemmas 13 and 16 give a common terminal margin for the true state at least $c _ { * } L ^ { 2 / 3 }$ for some $c _ { * } > 0$ . Lemmas 18 and 19 then yield

$$
D _ { \mathrm { K L } } ( p _ { H } ^ { E } \Vert p _ { H } ^ { R } ) \leq C _ { K } L ^ { 2 } e ^ { - c _ { * } L ^ { 2 / 3 } } .\tag{34}
$$

The complement has conditional probability at most

$$
H q + L ^ { - 3 } + ( K - 1 ) e ^ { - \kappa L } + C L e ^ { - L / 2 } .\tag{35}
$$

This unions event types only; temporal control is already contained in the single maximal event. Applying the deterministic 2L envelope (25) on the complement shows that the conditional expected KL tends to zero. The stationary starting state is uniform, so averaging the finitely many conditional bounds over s proves Theorem 4 for $K \geq 3$

For $K = 2 ,$ Lemma 20 and (26), with $R = 2 L .$ , give on $B _ { q } ^ { c }$

$$
D _ { \mathrm { K L } } ( p _ { H } ^ { E } \Vert p _ { H } ^ { R } ) \leq C L ^ { 2 } e ^ { - 3 \log T } \longrightarrow 0 .
$$

On $B _ { q } , ( 2 5 )$ and (29) give $\mathbb { E } [ D _ { \mathrm { K L } } ; \mathcal { B } _ { q } ] \le 2 L \operatorname* { P r } ( \mathcal { B } _ { q } ) \to 0$ . This proves the binary case for arbitrary distinct means and completes the proof of Theorem 4.

## E.8 Proof of diverging updates with vanishing predictive KL

ProofofTheorem 3. Fix finite $K \geq 2$ and abbreviate

$$
\alpha = 1 - { \frac { { \cal K } q } { { \cal K } - 1 } } , \qquad { \cal L } = { \cal L } _ { \cal K } ( q ) , \qquad b = { \frac { q } { { \cal K } - 1 } } .
$$

Adding a common constant to the input leaves softmax(z), and therefore log $P _ { q } ^ { \top }$ softmax(z)), unchanged; the radial output is also unchanged because its definition centers the input. Pairwise output differences are therefore gauge invariant, and the displayed witness $z _ { q } = L ( 1 / 4 , - 1 / 4 , 0 , \dots , \bar { 0 } )$ is centered.

Put

$$
Z _ { q } = e ^ { L / 4 } + e ^ { - L / 4 } + K - 2 .
$$

The definition of L gives $b = ( 1 - q ) e ^ { - L }$ . For $i \in \{ 0 , 1 \}$ , let

$$
\eta _ { i } = \frac { b Z _ { q } } { \alpha e ^ { ( z _ { q } ) _ { i } } } .
$$

Since $Z _ { q } \le K e ^ { L / 4 }$

$$
0 \leq \eta _ { 0 } \leq \frac { K ( 1 - q ) } { \alpha } e ^ { - L } , \qquad 0 \leq \eta _ { 1 } \leq \frac { K ( 1 - q ) } { \alpha } e ^ { - L / 2 } .\tag{36}
$$

For fixed K, both vanish as $q \to 0 ^ { + }$ . Common output centering cancels in the difference for the chosen pair, so

$$
\begin{array} { l } { { ( \Phi _ { q } ( z _ { q } ) ) _ { 0 } - ( \Phi _ { q } ( z _ { q } ) ) _ { 1 } = \displaystyle \frac { L } { 2 } + \log ( 1 + \eta _ { 0 } ) - \log ( 1 + \eta _ { 1 } ) } } \\ { { \displaystyle \qquad = \displaystyle \frac { L } { 2 } + o ( 1 ) , } } \end{array}\tag{37}
$$

where the absolute remainder is at most $\eta _ { 0 } + \eta _ { 1 }$

Because $r ( z _ { q } ) = L / 2$ , the radial gap for the chosen pair is exactly

$$
( R _ { q } ( z _ { q } ) ) _ { 0 } - ( R _ { q } ( z _ { q } ) ) _ { 1 } = L \operatorname { t a n h } ( \alpha / 2 ) .\tag{38}
$$

Subtracting (38) from (37), dividing by $L _ { ☉ }$ and using $\alpha  1$ yields

$$
\frac { [ ( \Phi _ { q } ( z _ { q } ) ) _ { 0 } - ( \Phi _ { q } ( z _ { q } ) ) _ { 1 } ] - [ ( R _ { q } ( z _ { q } ) ) _ { 0 } - ( R _ { q } ( z _ { q } ) ) _ { 1 } ] } { L } \longrightarrow \frac { 1 } { 2 } - \operatorname { t a n h } \frac { 1 } { 2 } = c _ { \star } > 0 .\tag{39}
$$

The positivity follows from tanh $x <$ < x for $x > 0$ . The eventual lower bound and divergence of the centered supremum follow directly from (39).

It remains to compare the decoded distributions at the same witness. Coordinate 0 is the unique maximizer of both outputs. For $K \geq 3$ , every zero coordinate has

$$
\eta _ { \mathrm { z e r o } } = \frac { b Z _ { q } } { \alpha } \leq \frac { K ( 1 - q ) } { \alpha } e ^ { - 3 L / 4 } \longrightarrow 0 .\tag{40}
$$

Hence the exact margin of the top over zero is $L / 4 + o ( 1 )$ , while (37) gives the exact margin of the top over coordinate 1 as $L / 2 + o ( 1 )$ . When $K = 2$ there are no zero coordinates, and (37) is the only margin over a wrong state. Every exact margin of the top over a wrong state is therefore eventually at least $L / 5$

Under the radial map, every margin of the top over a wrong state is at least $( L / 2 )$ tanh $\iota ( \alpha / 2 ) \geq L / 5$ eventually. The exact transition probabilities lie strictly between b and $1 - q ,$ , so its pairwise logit diameter is below L; the radial pairwise diameter is also below L. Thus the range of $\Phi _ { q } ( z _ { q } ) - R _ { q } ( z _ { q } )$ is below 2L. Applying Lemma 19 with $m = L / 5$ and $R = 2 L$ gives

$$
D _ { \mathrm { K L } } ( \mathrm { s o f t m a x } ~ \Phi _ { q } ( z _ { q } ) \parallel \mathrm { s o f t m a x } ~ R _ { q } ( z _ { q } ) ) \leq 4 K ( K - 1 ) L ^ { 2 } e ^ { - L / 5 } \longrightarrow 0 .\tag{41}
$$

This proves all claims.

## E.9 Proof of the binary transition geometry

Proof of Proposition 5. Writing $u = e ^ { h }$ and $D _ { q } ( u ) = ( ( 1 - q ) u + q ) ( q u + 1 - q )$ gives

$$
F _ { q } ^ { \prime } ( h ) = \frac { ( 1 - 2 q ) u } { D _ { q } ( u ) } , \qquad F _ { q } ^ { \prime \prime } ( h ) = \frac { ( 1 - 2 q ) q ( 1 - q ) u ( 1 - u ^ { 2 } ) } { D _ { q } ( u ) ^ { 2 } } .
$$

The first formula, equivalently $F _ { q } ^ { \prime } ( h ) = ( 1 - 2 q ) p ( 1 - p ) / [ p ^ { \prime } ( 1 - p ^ { \prime } ) ]$ for $p = \sigma ( h )$ and $p ^ { \prime } = q + ( 1 - 2 q ) p$ , is at most $1 - 2 q$ , with equality only at $h = 0 ;$ the range $p ^ { \prime } \in ( q , 1 - q )$ proves saturation. The second formula is strictly negative for $h > 0$ , so strict concavity on $[ 0 , 2 x ]$ and $F _ { q } ( 0 ) = 0$ give $\Delta _ { q } = 2 F _ { q } ( x ) - F _ { q } ( 2 x ) > 0$ . For residuals $e _ { z } = F _ { q } ( z ) - A ( z )$ , the affine second difference vanishes and hence

$$
\Delta _ { q } = | e _ { 0 } - 2 e _ { x } + e _ { 2 x } | \leq 4 \operatorname* { m a x } _ { z } | e _ { z } | ,
$$

which proves the quantitative obstruction.

Table 5: Equally spaced Gaussian illustration at fixed scale. Internal slopes regress centered separation from exact to radial on $L ;$ slopes of the log KL regress log $D _ { \mathrm { K I } }$ (exact∥radial) on $L .$ These values illustrate the theorem’s two directions for three fixed state counts; they are not used in its proof.
<table><tr><td></td><td>K Internal slope [95% CI]</td><td>Final gap/L [95% CI]</td><td>Slope of log KL [95% CI]</td></tr><tr><td>2 [0.5788, 0.5800]</td><td>0.5794</td><td>0.5419 [0.5414,0.5423]</td><td>-0.1574 [−0.1661, -0.1454]</td></tr><tr><td>4 [0.7487,0.7499]</td><td>0.7493</td><td>0.7315 [0.7311,0.7319]</td><td>-0.0265 [-0.0292,-0.0241]</td></tr><tr><td>8 [0.8681, 0.8696]</td><td>0.8688</td><td>0.8728 [0.8724, 0.8733]</td><td>-0.0085 [-0.0090, -0.0080]</td></tr></table>

## F Empirical Estimands and Aggregation

This appendix defines the illustration with equally spaced Gaussians at fixed $K .$ , experiments with learned models, and estimands for the scalar controls. These experiments illustrate the results and are not used in the proof of Theorem 4. The displayed subset with equally spaced Gaussians was selected after the broader prespecified grid had been evaluated, to illustrate the two directions rather than test a uniform onset at finite scale. On the original Gaussian grid, four of five configurations pass the joint directional criterion; the $K = 8$ skewed configuration has an unresolved slope of the log KL over the prespecified tail. The supplement includes results for the full original grid and identifies the additional experiments conducted afterward; the displayed subset does not replace the original outcome.

## F.1 Illustration with equally spaced Gaussians at fixed K

## F.1.1 Design, coupling, and estimands

For $K \in \{ 2 , 4 , 8 \}$ , the scalar Gaussian means begin as the arithmetic progression $0 , 1 , \ldots , K - 1$ , are centered, and are rescaled so that the minimum pairwise distance is one. The observation variance is one, hence

$$
d _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { i \neq j } { \frac { ( \mu _ { i } - \mu _ { j } ) ^ { 2 } } { 2 } } = { \frac { 1 } { 2 } } , \qquad c = 0 . 4 5 d _ { \operatorname* { m i n } } = 0 . 2 2 5 .
$$

At each $L \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ we set

$$
q = \frac { K - 1 } { e ^ { L } + K - 1 } , \qquad H = \left\lceil \frac { L } { 0 . 2 2 5 } \right\rceil + 1 .
$$

This implemented horizon differs from the theorem’s $\left\lceil - \log ( q ) / c \right\rceil + 1$ by $O _ { K } ( 1 )$ steps, since $L = - \log q + \log ( K -$ $1 ) + \log ( 1 - q ) ;$ ; it shares the logarithmic scale rather than the exact schedule. Four seed blocks contribute 1,024 stationary trajectories per cell, for 4,096 paired paths. Exact Bayes and the tanh radial filter consume the same observations. Every displayed path and filter state is finite.

The internal quantity is $G _ { H } = \| \mathcal { C } ( z _ { H } ^ { E } - z _ { H } ^ { R } ) \| _ { 2 }$ and the task quantity is $D _ { \mathrm { K L } } ( p _ { H } ^ { E } \Vert p _ { H } ^ { R } )$ . The normalized gap $G _ { H } / L$ distinguishes growth proportional to the confidence scale from a merely positive absolute gap. Internal slopes regress mean $G _ { H }$ on $L ;$ predictive slopes regress the logarithm of mean predictive KL on L. Both use the predeclared tail window $L \in \{ 3 2 , 6 4 , 1 2 8 \}$ .

Uncertainty is computed by resampling complete paths. Each of 2,000 bootstrap replicates independently resamples path indices within each scale cell, then fits the statistic across those resampled cell means. Filters remain paired on each path, but independently simulated scales are not paired. Table 5 reports the three displayed configurations.

## F.1.2 Controls and implementation details

Hard clipping and the affine update are comparisons rather than assumptions of the theorem. Hard clipping may outperform tanh at finite scales because the theorem claims existence and asymptotic convergence, not optimality of the smooth map. The affine control has lower mean KL than tanh in every displayed tail cell $\bar { L } \geq 3 2$ . All 4,096 sampled paths in each such cell contain no switch even by 2H, so these curves do not test rare costs in the switch tail or identify saturation as necessary. The binary $\lceil 8 / q \rceil$ controls address a separate recovery question at the longer endpoint.

The supplementary code includes the data generator, simulation parameters, seed construction, measurements per trajectory, and the 2,000-replicate bootstrap procedure used to produce the curves and table. Numerical checks verify the KL direction and that all simulated trajectories are complete and finite. Table values are rounded to four decimal places; the accompanying data provide full precision.

## F.2 Experiments with learned models

The experiment covers six switch probabilities $q \in \{ 2 ^ { - 3 } , \ldots , 2 ^ { - 8 } \}$ , two training stages (G1 distilled and G2 trained end to end), five learned arms and an additional analytic tanh reference, three tuning seeds, and ten disjoint confirmation seeds. For each stage and model, the selected hyperparameter setting minimizes mean validation loss over all three tuning seeds, with ties broken deterministically; a setting missing any tuning seed is ineligible. This selection rule was fixed before confirmation. All 180 tuning runs and 600 final evaluation runs completed successfully. Every confirmation seed is included in the reported aggregates. The experimental protocol also specified test excess NLL on the next observation and calibration error, but neither was measured. Table 6 specifies the model architectures and selected optimizer settings; the official Mamba implementation is mamba-ssm==2.3.2.post1.

The closure statistic in Table 2 is computed within a common stage, q, seed, and endpoint:

$$
C _ { a } = \frac { D _ { \mathrm { S 6 } } - D _ { a } } { D _ { \mathrm { S 6 } } - D _ { \mathrm { t a n h } } } .
$$

Thus S6 is 0 and the analytic tanh arm is 1 before averaging. The table reports the mean and standard error of these ten closure values, one for each seed, not a standard error obtained by treating paths within a seed as independent replicates.

## F.3 Predictive KL: orientation and clipping

The experiments with learned models and the scalar controls also use different probability clipping settings. Table 2 uses $D _ { \mathrm { K L } }$ (exact Bayes∥model) after float32 sigmoid conversion and clipping at $\begin{array} { r l } { \varepsilon _ { 3 2 } } & { { } = } \end{array}$ torch.finfo(torch.float32). $\mathsf { s p s } \approx \mathrm { i . 1 9 2 0 9 2 9 \times 1 0 ^ { - 7 } }$ . The scalar controls below use float64 probabilities clipped at $1 0 ^ { - 1 5 }$ . These settings compute KL after different clipping operations, so the reported values are not directly interchangeable.

The fixed-K theorems, Table 7, and Figure 2 all use the direction $D _ { \mathrm { K I } }$ (exact∥approximation), although the experiments use their separately disclosed probability clipping settings. An additional diagnostic reverses the direction and reports $D _ { \mathrm { K L } } ( \operatorname { t a n h } \parallel \mathrm { e x a c t } )$ . Because KL is asymmetric, those values in the reverse orientation are not compared numerically with the theorem or the comparison from exact to control.

That auxiliary diagnostic evaluates the tanh recurrence at $H ( q )$ without probability clipping. It computes Bernoulli KL directly in logit space, uses the same ten seeds and $2 0 { , } 0 0 0$ paths per seed, and is reported separately from the comparison from exact to control.

$$
\frac { q } { D _ { \mathrm { K L } } ( \operatorname { t a n h } \parallel \operatorname { e x a c t } ) } \left| \frac { 2 ^ { - 3 } } { 1 . 7 4 \times 1 0 ^ { - 4 } } \right. 8 . 5 3 \times 1 0 ^ { - 4 } \left. 2 . 2 7 \times 1 0 ^ { - 3 } \right. 4 . 2 3 \times 1 0 ^ { - 3 } \left. 6 . 0 0 \times 1 0 ^ { - 3 } \right. 7 . 6 3 \times 1 0 ^ { - 3 }
$$

The largest relative standard error is $0 . 4 0 \%$ . The loss in the reverse orientation increases across this finite grid. It is a diagnostic in the finite regime, not an estimate of either proved limit from exact to radial or its asymptotic rate.

If e is the exact predictive logit, m is a model predictive logit, $p = \sigma ( e )$ , and $r = \sigma ( m )$ , the corresponding untruncated empirical estimand is

$$
D _ { \mathrm { t r u e } } ( e \| m ) = D _ { \mathrm { K L } } ( \mathrm { B e r } ( p ) \| \mathrm { B e r } ( r ) ) = p ( e - m ) + \operatorname { s p } ( m ) - \operatorname { s p } ( e ) , \qquad \operatorname { s p } ( z ) = \log ( 1 + e ^ { z } ) .
$$

The last expression is a stable evaluation in logit space of the ordinary Bernoulli KL; implementations may evaluate sp by logaddexp without clipping either probability.

The experiment with scalar controls evaluates a distinct, clipped quantity. For $c _ { \varepsilon } ( u ) = \operatorname* { m i n } \{ 1 - \varepsilon , \operatorname* { m a x } \{ \varepsilon , u \} \}$ with $\varepsilon = 1 0 ^ { - 1 5 }$ , it records

$$
D _ { \mathrm { c l i p } , \varepsilon } ( e \| m ) = D _ { \mathrm { K L } } ( \mathrm { B e r } ( c _ { \varepsilon } ( \sigma ( e ) ) ) \| \mathrm { B e r } ( c _ { \varepsilon } ( \sigma ( m ) ) ) ) .
$$

Accordingly, every scalar value currently printed in Table 7 and plotted in the third panel of Figure 2 is clipped predictive KL with orientation exact Bayes ∥ control. Clipping is nearly inactive for the two saturating controls but can substantially cap a nonsaturating control in an extreme tail. The clipped and untruncated estimands must therefore be named separately. The reported control results use clipped KL, not the untruncated quantity.

## F.4 Scalar controls and endpoints

All four scalar recurrences consume the same simulated observation paths as the exact filter. After T observations, let $h _ { T }$ be the exact posterior logit and ${ \widetilde { h } } _ { T } ^ { a }$ the state of control a. The final logits compared by the scalar evaluator are

$$
e _ { T } = F _ { q } ( h _ { T } ) , \qquad m _ { T } ^ { a } = M _ { q } ^ { a } ( \widetilde { h } _ { T } ^ { a } ) , \qquad a \in \{ \mathrm { t a n h , c l i p , a f f n e , i d e n t i t y } \} .
$$

Table 6: Model architectures and training settings. All arms use AdamW, batch size 256, at most 10,000 updates, validation every 250 updates, and patience 2,000; η and λ denote learning rate and weight decay. G1 distills the exact predictive distribution, whereas G2 minimizes negative log likelihood of the next observation. Parameter counts are matched within 5%. The arm with a finite window uses $\begin{array} { r } { H ( \check { q } ) = \lceil - \log q \rceil + 1 } \end{array}$ ; the Sharan existence bound log $2 / H ( q )$ ranges from 0.173 to 0.099 over this grid and is not a guarantee for the trained MLP.
<table><tr><td>Arm</td><td>Architecture</td><td>Parameters</td><td> $\mathbf { G 1 } / \mathbf { G 2 } \left( \eta , \lambda \right)$ </td></tr><tr><td>Mamba  $\times 2$ </td><td>two blocks;  $d _ { \mathrm { m o d e l } } = 3 2 , d _ { \mathrm { s t a t e } } = 1 6 .$  convolution 4, expansion 2</td><td>20,001</td><td> $\left( 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \right) / \left( 1 0 ^ { - 4 } , 0 \right)$ </td></tr><tr><td>Mamba ×1</td><td>one block;  $d _ { \mathrm { m o d e l } } = 4 8 ,$  otherwise as above</td><td>19,873</td><td> $\left( 1 0 ^ { - 3 } , 0 \right) / \left( 3 \times 1 0 ^ { - 4 } , 0 \right)$ </td></tr><tr><td>GRU</td><td>one layer, hidden width 80</td><td>20,001</td><td> $\left( 1 0 ^ { - 3 } , 0 \right) / \left( 1 0 ^ { - 3 } , 0 \right)$ </td></tr><tr><td>Scalar S6</td><td>one scalar state; two layers of width 139 in the SiLU coefficient network</td><td>20,018</td><td> $\left( 3 \times 1 0 ^ { - 4 } , 0 \right) /$   $( \mathrm { 3 } \times 1 0 ^ { - 4 } , 0 )$ </td></tr><tr><td>Finite window</td><td>last  $H ( q )$  observations plus validity mask; widths 136, 136, 135, 134, 134, 133 for  $q = { \dot { 2 } } ^ { - 3 } , \dots , 2 ^ { - 8 }$ </td><td>19,951–19,993</td><td> $\left( 1 0 ^ { - 3 } , 0 \right) / \left( 3 \times 1 0 ^ { - 4 } , 0 \right)$ </td></tr></table>

Table 7: Controls separate saturation from geometry. Expected KL at the final step, KL(exact Bayes∥control), at the endpoint matched to the theorem $H ( q ) \stackrel { - } { = } \lceil - \log { q } \rceil + \stackrel { - } { 1 }$ and the long endpoint $\bar { \lceil } 8 / q \rceil$ . At the long endpoint, the saturating losses remain below $6 . 4 \times 1 0 ^ { - 3 }$ , while the tested affine and identity losses reach 5.53 and 14.54 nats, respectively (bold).
<table><tr><td></td><td colspan="2">Saturating</td><td colspan="2">Nonsaturating</td></tr><tr><td>Endpoint</td><td>tanh</td><td>clip</td><td>affine</td><td>identity</td></tr><tr><td colspan="5"> $q = 2 \AA ^ { - 8 }$ </td></tr><tr><td> $H ( q )$ </td><td> $6 . 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 \times 1 0 ^ { - 3 }$ </td><td> $7 . 9 \times 1 0 ^ { - 2 }$ </td><td> $8 . 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td> $\lceil 8 / q \rceil$ </td><td> $6 . 3 6 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 7 \times 1 0 ^ { - 3 }$ </td><td>5.53</td><td>14.54</td></tr><tr><td colspan="5"> $q = 2 \AA ^ { - 3 }$ </td></tr><tr><td> $H ( q )$ </td><td> $1 . 7 2 \times 1 0 ^ { - 4 }$ </td><td> $3 . 2 0 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 4 \times 1 0 ^ { - 1 }$ </td><td> $9 . 1 5 \times 1 0 ^ { - 1 }$ </td></tr><tr><td> $\lceil 8 / q \rceil$ </td><td> $1 . 7 3 \times 1 0 ^ { - 4 }$ </td><td> $3 . 2 0 \times 1 0 ^ { - 3 }$ </td><td> $4 . 0 6 \times 1 0 ^ { - 1 }$ </td><td>10.28</td></tr></table>

The two endpoints are the horizon matched to the theorem $\begin{array} { r } { H ( q ) = \lceil - \log q \rceil + 1 } \end{array}$ and the long endpoint $\lceil 8 / q \rceil$ . The experiment in Table 7 and Figure 2 uses $q = 2 \AA ^ { - 3 }$ and $q = 2 \AA ^ { - 8 }$ , ten public seeds $1 0 8 0 , \ldots , 1 0 8 9$ , 20,000 paths per seed at $H ( q )$ , and 2,000 paths per seed at the long endpoint. Thus each short cell contains 200,000 paths and each long cell contains 20,000 paths.

For a seed s with $N _ { s }$ paths, the stored replicate is

$$
\widehat { D } _ { s } ^ { a } = \frac { 1 } { N _ { s } } \sum _ { i = 1 } ^ { N _ { s } } D _ { \mathrm { c l i p } , 1 0 ^ { - 1 5 } } ( e _ { T , i } \Vert m _ { T , i } ^ { a } ) .
$$

With $S = 1 0$ , the manuscript mean and uncertainty are

$$
\overline { { { D } } } ^ { a } = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \widehat { D } _ { s } ^ { a } , \qquad \mathrm { S E } ( \overline { { { D } } } ^ { a } ) = \sqrt { \frac { 1 } { S ( S - 1 ) } \sum _ { s = 1 } ^ { S } ( \widehat { D } _ { s } ^ { a } - \overline { { { D } } } ^ { a } ) ^ { 2 } } .
$$

The random stream for one q/endpoint/seed cell is derived from the displayed seed, the exponent of $q ,$ and the horizon; all four controls reuse that cell. The manuscript aggregates seed means rather than treating paths within a seed as independent replicates. The largest relative standard error among the sixteen plotted quantities is at most 3.1%.

The right pair in Figure 2 separates measurement from illustration. The third panel reports the mean losses of the scalar controls. The fourth panel is not a sampled trajectory: it evaluates the logistic decoder on a fixed grid, with schematic guide logits $v = 3$ and $u = 5 . 2 5$

## F.5 Errors relative to the latent state

Let $Z _ { T + 1 } \in \{ 0 , 1 \}$ be the latent state after the terminal transition, $s _ { Z } = 2 Z _ { T + 1 } - 1$ , e the exact predictive logit, and $m _ { T } ^ { \mathrm { a f f } }$ the affine predictive logit. The event that the affine prediction disagrees with the latent state is

$$
W _ { \mathrm { t r u t h } } = \mathbf { 1 } \{ s _ { Z } m _ { T } ^ { \mathrm { a f f } } \leq 0 \} .
$$

The associated confidence event used by the formal diagnostic for the lower bound is

$$
W _ { \mathrm { j o i n t } } = W _ { \mathrm { t r u t h } } { \bf 1 } \{ \sigma ( s _ { Z } e _ { T } ) \geq 1 5 / 1 6 \} .
$$

Both indicators are defined relative to the true latent state. Disagreement with the exact filter is a different quantity and was not measured in this experiment.

The experiment uses $q = 2 ^ { - 5 } , 2 ^ { - 8 } , 2 ^ { - 1 0 } , 2 ^ { - 1 2 }$ , the same ten public seeds, and 1,000 paths per seed. Means across seeds ± standard errors for $W _ { \mathrm { t r u t h } }$ are respectively $0 . 1 9 8 2 \pm 0 . 0 0 6 \dot { 4 } , 0 . 1 8 2 1 \pm 0 . 0 0 4 2 , 0 . 1 \dot { 8 } 2 9 \pm \dot { 0 } . 0 0 3 1 , \mathrm { a n d } 0 . 1 8 6 8 \pm 0 . 0 0 3 6 .$ The corresponding $W _ { \mathrm { j o i n t } }$ values, qualified by confidence, are $0 . 0 7 9 4 \pm 0 . 0 0 2 7 , 0 . 1 6 4 9 \pm 0 . 0 0 4 6 , 0 . 1 7 8 1 \pm 0 . 0 0 2 8 .$ and $0 . 1 8 5 6 \pm 0 . 0 0 3 6$ . The reported range describes errors relative to the latent state, rather than disagreement with the exact filter.

## F.6 Scope of the empirical conclusions

The experiments support two deliberately limited conclusions. First, at the long endpoint the saturating controls remain stable while the two nonsaturating controls degrade under the disclosed clipped metric. Second, matching the short theorem horizon does not by itself establish stability at long horizons for a learned architecture. Neither observation proves the open lower bound for the terminal joint event qualified by confidence, identifies tanh as the unique stable recurrence, or establishes a positive result on transfer between architectures.

## G Extended Background

This appendix places the result in the literature on recurrent models, approximate filtering, and sequential prediction.

## G.1 The family with a deterministic state

The recurrence studied here is deliberately the smallest object that exhibits the property in question: a state of fixed size updated deterministically from the previous state and the current observation. For recurrent implementations, this property permits inference in linear time and memory independent of sequence length. Configurations with long convolutions or attention hybrids need not share that memory bound.

The lineage begins with the question of how a state of fixed size should summarize a growing history at all. Optimal polynomial projection gives one answer with an explicit approximation guarantee [24], and structured state spaces turn that answer into a trainable layer [25], later simplified [60] and given variants in continuous time [28]. Initialization of these models remains an active question in its own right [39]. A parallel line showed that carefully parameterized linear recurrences recover much of the benefit without the state space machinery [47], and designs with long convolutions and hybrid designs traded recurrence for structured mixing [51, 19]. Gated linear attention and its delta rule successors reached the same destination from the attention side [67, 68], with sparsity now used to enlarge the state without paying for it densely [8], while recurrent architectures competitive at language scale arrived independently [49, 3]. The selective mechanism that makes the transition depend on the input [23] and the duality that exposes such models as a restricted form of attention [12] complete the picture.

These models motivate asking what predictive cost follows from a prescribed update geometry. Shared motivation does not extend our existence theorem to every member of this family: the theorem concerns the explicit radial filter, not its implementation by a named neural architecture.

## G.2 What is already known about the gap, and what is not

The representational side of the question is comparatively well mapped. Formal placements bound what specified architectures can express [54], while empirical studies on formal languages probe generalization outside the training distribution [13]. Arguments from circuit complexity place models of fixed depth inside classes that cannot perform certain sequential computations [43], and analyses with limited precision show how much of the apparent capacity survives finite arithmetic [38]. On the constructive side, changing the spectrum or the sparsity pattern of the transition recovers ability to track the state that diagonal models lack [22, 64, 58], and the frontier between recall and throughput has been characterized directly [2]. Benchmarks now target exact state tracking as a primary object [65].

What this literature does not settle is the step our paper takes. A proved inability to represent an update is not by itself a proved cost under the data distribution, because the distribution need not visit the region where the representation fails. The closest existing work asks about generalization of selective state space models on filtering tasks [63], which shares our setting but not our question: we fix the recurrence and ask what its representational shortfall costs in expected predictive divergence.

## G.3 Filtering, stability, and their relation to this result

Hidden Markov filtering [53] and entropy questions for functions of chains with finitely many states [6] have a long history. Exact filtering of a K-state HMM admits a deterministic continuous belief state with K − 1 coordinates. Finite dimension is not finite cardinality or bounded precision; our restriction concerns update geometry, not the existence of a sufficient state of finite dimension. The stability literature asks a superficially similar question to ours and a materially different one: whether a filter initialized incorrectly but running the correct kernels forgets its error [46], surveyed by Chigansky et al. [9], with quantitative forms via contraction coefficients [41]. Robustness results perturb the kernels instead, controlling policy costs [31, 33] or error in the filter kernel and policy performance [14]. Policies with a finite window can become nearly optimal as window length grows [32]. Our state dimension is fixed, but $R _ { q }$ changes numerically with q, and its map discrepancy in the worst case grows rather than vanishing as a perturbation parameter. How fast a correct filter forgets is itself quantified [37], and estimating the mixing time is a developed subject [66].

Learned and structured filters form a third strand. Probabilistic state space models have been fused with the selective architecture [4], particle filters made differentiable [11], belief embeddings learned with consistency guarantees [61], and Updates in the style of the Kalman filter recast as attention or as regression at test time [55, 50], including adaptive variants [42]. Learning from ordinary HMM observation sequences can be cryptographically hard, whereas query access to conditional probabilities permits efficient learning; guarantees from conditional samples also depend on a fidelity parameter [30]. Reconstruction from finite messages on trees exhibits phase transitions between memory and accuracy [29], and a recent synthesis compares state space models and HMMs [20].

## G.4 Prediction under a wrong model

Our theorem measures the categorical KL between filtered posteriors after the observation, not the KL between densities of the next observation, and binary empirical endpoints additionally apply transition mixing. These losses score decoded distributions rather than internal parameters. Prediction with short memory supplies a conceptual benchmark on the next observation [57]; sequential prediction under log loss with a misspecified model class is studied directly [18]; and streaming projection onto a mixture over paths with a fixed budget is another deterministic recurrence with a different maintained object and truncation [17]. Read as detection delay, our controls at long horizons belong to quickest change detection, from the original cumulative sum scheme [48] through its optimality theory [40, 44, 52] to learned detectors [21]. Behavior in context on Markov sources is a further contact point [7, 69].

## G.5 Comparison by mathematical object

Nearby literatures can sound interchangeable when summarized as “approximating a sequence model,” but they differ in what may change with the accuracy target and in where error is measured. Table 8 records the distinction needed here.

Table 8: Nearby questions separated by object and limit. The comparison is conceptual; it does not claim that the cited settings are special cases of ours.
<table><tr><td>Line of work</td><td>Object being compared</td><td>Accuracy resource or limit Relation to this paper</td><td></td></tr><tr><td>[43, 54, 10]</td><td>class</td><td>Architecture expressivity Function or language rep- Depth, state structure, or in- Establishes representational separations, but resented by an architecture put length varies with the the- generally does not integrate their task cost orem</td><td>under a data law. Prediction with short Predictor with full history Window length grows as the Supplies a positive approximation baseline;</td></tr><tr><td>memory [57]</td><td>versus a finite observation target error shrinks window</td><td></td><td>our state dimension and prescribed family remain fixed, but the numerical map depends on q.</td></tr><tr><td>ness [46, 41, 37]</td><td>Filter stability and robust- Two filters with different ini- Influence of the initial con- Controls belief error when the source of mis-</td><td>perturbation is taken small</td><td>tial beliefs or nearby kernels dition is forgotten or kernel match disappears; our two transition geome- tries do not converge internally.</td></tr><tr><td>Compression [16, 15, 26]</td><td>that Original representation ver- Compression is optimized Shares the lesson that discarded information tion</td><td></td><td>is aware of the task sus a compressed representa- relative to a downstream task can be irrelevant to the task, but not the re- current Bayesian construction under the path law.</td></tr><tr><td>This paper</td><td>with a fixed state</td><td>Θ(log(1/q))</td><td>Exact categorical Bayes mix- Fixed finite K; switch rar- Proves diverging centered map separation ing versus one radial update ity tends to zero at H(q) = and vanishing decoded KL, then proves sta- tionary expected predictive closure.</td></tr></table>

The comparison also limits the architectural interpretation: a selective SSM, a gated recurrent network, and the radial filter all keep a fixed state, but the theorem concerns the explicit radial map, not a class containing every such recurrence. Exact Bayes and the radial witness are mathematical filters, so their separation predates training or parameterization.

The result sits between architecture expressivity and statistical decision theory: an internal separation can fail to induce predictive loss under a specified sequential law, leaving the converse criterion and learned realization open.