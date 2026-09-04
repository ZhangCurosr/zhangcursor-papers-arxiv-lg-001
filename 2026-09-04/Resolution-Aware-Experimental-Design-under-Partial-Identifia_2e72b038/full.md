# Resolution-Aware Experimental Design under Partial Identifiability

Sofianos-Panagiotis Fotias National Technical University of Athens

## Abstract

Experimental design is commonly framed as choosing the experiment expected to provide the most information. Under partial identifiability however, persistent nuisance uncertainty can make the same observation carry diferent structural meanings. We introduce Resolution-Aware Experimental Design (RAED), which selects an experiment by the smallest expected nonempty structural candidate set achievable subject to false-exclusion control. We prove an exact cross-nuisance aliasing separation: an experiment can be preferred by structural and fulllatent information gain, average classification, and nuisance-marginalized informativeness while having arbitrarily poorer valid structural resolution. RAED nevertheless preserves the expected ordering under a genuine composite Blackwell comparison. To make this criterion operational, we develop a learned score-based implementation with finite-sample nuisance-average and positivetail calibration, and characterize a rare-tail sample-complexity obstruction. Under constrained sensing, two subsurface-flow benchmarks exhibit genuine RAED–expected-information-gain (EIG) experiment-selection disagreements, with the clearest and largest held-out resolution diferences in WCA. In a fluvial benchmark, tail protection changes the selected physical experiment and replaces hard-region false exclusions primarily with explicit ambiguity. In a mechanistic methane-oxidation benchmark, a prospectively specified 5% false-exclusion tolerance also yields a nontrivial finite-sample population guarantee for tail-sensitive nuisance risk, with 95% joint confidence across all three structural families.

## 1 Introduction

Experimental design asks which intervention, measurement, or control should be chosen before future data are observed. In Bayesian experimental design, a standard principle is to prefer experiments expected to change posterior beliefs about an unknown model or parameter as much as possible. Expected information gain formalizes this idea by measuring the expected divergence between prior and posterior beliefs, equivalently the mutual information between the unknown quantity and the future observation [1, 2]. This is natural when the scientific objective is posterior concentration. But in many model-discrimination problems, observations are also shaped by persistent nuisance variables whose values can substantially alter the response. In such settings, a more direct objective is to determine which structural explanations can be excluded while controlling the risk of excluding the truth. Suppose the structural target is

$$
M \in \{ 1 , \ldots , K \} ,
$$

and experiment e produces an observation Y . Rather than force a single structural label, the terminal decision may return a nonempty candidate set

$$
\Gamma _ { e } ( Y ) \subseteq \{ 1 , \dots , K \} .
$$

A singleton corresponds to complete structural resolution, whereas a larger set records ambiguity that cannot be safely removed. Because the nuisance state $\theta$ is not observed at decision time, the same nuisance-blind rule $\Gamma _ { e }$ must map observations to structural conclusions across all nuisance states covered by the validity requirement. The design objective is therefore to choose the experiment that supports the smallest valid candidate set.

Resolution-Aware Experimental Design (RAED) formalizes this idea. For each structural family $k ,$ the observation law depends on a persistent nuisance state $\theta \in \Theta _ { k }$ . Given a false-exclusion tolerance α, RAED controls the probability of excluding the true structural family and evaluates experiment e through the expected candidate-set size

$$
J _ { e } = \mathbb { E } | \Gamma _ { e } ( Y ) | .\tag{1}
$$

Validity can be required at every admissible nuisance state, on average across nuisance states, or over the worst δ-fraction of the nuisance distribution, where $\delta \in ( 0 , 1 ]$ specifies the fraction being protected. Smaller values of $\delta$ concentrate the validity requirement on increasingly adverse nuisance regimes, while $\delta = 1$ recovers nuisance-average validity. Together, the false-exclusion tolerance α and nuisance-tail level $\delta$ determine the trade-of between structural resolution and the strength of validity protection.

The distinction matters because nuisance averaging can conceal incompatible structural meanings of the same observation. An experiment may be highly informative about M after nuisance averaging, or even identify M perfectly at each fixed nuisance state, while no single nuisance-blind terminal rule can safely translate its observations into a precise structural conclusion. We call this phenomenon cross-nuisance structural aliasing. It is an experiment-level form of partial identifiability: the issue is not simply overlap between averaged predictive distributions, but whether a single structural decision rule can remain valid across the nuisance states that the experiment must tolerate.

Table 1 gives a concrete four-family example. Let $M \in \{ 0 , 1 , 2 , 3 \}$ denote the structural family and let the nuisance state be $Z \in \{ 0 , 1 , 2 , 3 \}$ . Consider an experiment $e _ { I }$ whose observation is

$$
Y = M \oplus Z { \pmod { 4 } } .
$$

At each fixed $Z ,$ the observation uniquely determines M. However, once $Z$ is unknown this identification breaks down. The same observed value can correspond to diferent structural families under diferent nuisance states.

<table><tr><td> $Z \backslash M$ </td><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>0</td><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>1</td><td>1</td><td>2</td><td>3</td><td>0</td></tr><tr><td>2</td><td>2</td><td>3</td><td>0</td><td>1</td></tr><tr><td>3</td><td>3</td><td>0</td><td>1</td><td>2</td></tr></table>

Entries are observations $Y = M \oplus Z$ (mod 4); bold entries mark the same observation $Y = 0$

Table 1: Cross-nuisance structural aliasing at $K = 4$ . Each row is a permutation of the structural label, so $e _ { I }$ identifies M perfectly at every fixed Z. Across nuisance states, the same observation $Y = 0$ is compatible with every M. A single nuisance-blind terminal rule therefore cannot assign that observation a unique structural meaning. Theorem 4.1 compares this informationfavoured experiment with a nuisance-independent experiment $e _ { R }$ that achieves singleton RAED resolution.

This perspective leads to three main contributions.

1. A resolution-aware design objective. RAED places a randomized nonempty structural candidate set at the end of the experiment and selects the experiment with the smallest expected candidate-set size subject to a declared false-exclusion tolerance. Its validity criterion can range from protection at every nuisance state to tail-sensitive and nuisance-average guarantees.

2. A theory of when information and structural resolution disagree. We identify crossnuisance structural aliasing as a mechanism by which an experiment can appear highly informative while still failing to support a precise nuisance-blind structural decision. An exact cyclic construction shows that this disagreement can grow with the number of structural families. We also show that RAED preserves the expected ordering under genuine composite informativeness and that the separation persists beyond exact aliasing.

3. A learned implementation with finite-sample calibration. We develop score-based decision rules with independent calibration for nuisance-average and tail-sensitive validity, turning the population objective into a practical experiment-selection procedure. Across two subsurface-flow benchmarks, RAED and expected information gain select diferent experiments and produce materially diferent held-out structural resolution. In a fluvial benchmark, tail protection changes the selected well test and replaces hard-region false exclusions primarily with explicit ambiguity. A mechanistic methane-oxidation study further demonstrates nontrivial finite-sample population certification under a prespecified false-exclusion tolerance

The remainder of the paper relates RAED to neighboring approaches, develops the formal decision problem and aliasing theory, introduces the learned calibration procedure, and evaluates the resulting experiment choices and finite-sample guarantees.

## 2 Related Work

Bayesian experimental design and model discrimination. Bayesian experimental design commonly selects e by maximizing expected information gain about a latent quantity, including structural model indices and continuous parameters [1, 2]. Model-discrimination design instead uses criteria tailored to separating rival models, including T-optimality, robust discrimination, and classification-based Bayesian design [3, 4, 5, 6, 7]. Expected model-elimination criteria similarly reward experiments that are expected to reject more competing models [8]. RAED difers in the terminal decision it optimizes: after observing the experiment, it seeks the smallest nonempty structural candidate set that can be reported while controlling false exclusion across persistent nuisance uncertainty.

Set-valued classification and confidence sets. Minimum expected set size under coverage is a classical statistical objective. Least-ambiguous set-valued classification minimizes expected prediction-set size subject to classwise coverage constraints [9]; related work studies confidence sets and classwise prediction sets with controlled ambiguity [10, 11, 12]. At the nuisance-average endpoint, fixed-experiment RAED reduces exactly to this problem when the validity and eficiency laws coincide and the same randomized decision class is used. RAED then places this termina set-valued decision inside an outer experiment-selection problem and extends the validity requirement to persistent nuisance uncertainty.

Robust and ambiguity-aware design. Robust design methods protect experiment utilities against nuisance uncertainty, model misspecification, or distributional ambiguity. Robust expected information gain introduces ambiguity sets around the information objective [13], while maximin Bayesian design frames experiment choice as a game against an adversarial nature [14]. RAED instead applies robustness to the nuisance-specific false-exclusion loss

$$
L _ { k , e } ( \theta ) = \mathbb { P } _ { k , \theta } ^ { e } \{ k \notin \Gamma _ { e } ( Y ) \} ,
$$

and optimizes structural resolution subject to the resulting validity requirement. Positive values of δ interpolate between increasingly adverse nuisance tails and nuisance-average protection, while $\delta = 0$ is treated separately as the pointwise-validity endpoint.

Comparison of statistical experiments. Blackwell’s ordering formalizes when one experiment can be simulated from another through a common garbling, while Le Cam–Torgersen deficiency theory quantifies approximate comparisons [15, 16, 17]. These notions provide a natural benchmark for RAED. If the same nuisance-independent garbling maps one experiment to another for every $( k , \theta )$ , then the full nuisance-specific decision problem is preserved and RAED respects the resulting informativeness ordering. The separation studied here arises under weaker notions of informativeness, such as comparisons made only after nuisance averaging or through mappings whose interpretation changes with the nuisance state.

Compound channels, confusability, and list decoding. Persistent nuisance uncertainty also creates conceptual links to compound channels, symmetrizability, confusability, and list decoding [18, 19, 20, 21]. These literatures study communication rates, adversarial channel states, zero-error confusability, and decoding lists. RAED addresses a diferent problem: one-step scientific experiment selection with controlled false exclusion and expected candidate-set size as the resolution objective. The common theme is that the same observation may support incompatible interpretations across nuisance states unless a single decoder can resolve them consistently.

Risk control and finite-sample calibration. The calibration layer builds on risk-controlling prediction, Learn-then-Test, optimized certainty equivalent (OCE) risk, and concentration bounds for bounded losses [22, 23, 24, 25]. In RAED, these tools provide finite-sample guarantees for the nuisance risk of learned candidate-set rules without changing the underlying population validity criterion. The parameters α and δ specify the scientific false-exclusion requirement, while ζ is the allowed probability that the finite-sample certificate itself fails.

Taken together, these connections place RAED at the intersection of experimental design and set-valued decision making under nuisance uncertainty. Its distinguishing feature is the outer design objective: experiments are compared by the smallest expected nonempty structural candidate set they can support subject to a prescribed false-exclusion requirement. The next section formalizes this decision problem and shows how, for a fixed experiment at the nuisance-average endpoint, it reduces to classical set-valued classification under the appropriate conditions.

## 3 Resolution-Aware Experimental Design

## 3.1 Structural families, nuisance states, and experiments

$$
K \geq 2 , \qquad 0 \leq \alpha < 1 , \qquad 0 \leq \delta \leq 1 .\tag{2}
$$

The structural target is

$$
M \in \{ 1 , \ldots , K \} .
$$

Conditional on $M = k ,$ , the data also depend on a nuisance state $\theta \in \Theta _ { k }$ . The nuisance state is persistent in the sense that, for a realized experiment, the same value of θ governs the entire observation Y rather than being redrawn across its components.

An experiment $e \in { \mathcal { E } }$ induces, for every structural family k and nuisance state $\theta \in \Theta _ { k }$ , an observation law

$$
P _ { k , \theta } ^ { e }
$$

on an observation space $\mathcal { V } _ { e }$ . We assume these spaces are standard Borel. The experiment and its observation protocol are chosen before Y is observed.

Two nuisance distributions may enter the formulation, and they serve diferent purposes. The measure $\mu _ { k }$ specifies how nuisance states within family k are weighted when imposing nuisanceaverage or nuisance-tail validity. The measure $\nu _ { k }$ specifies how nuisance states are weighted when evaluating the expected size of the resulting candidate set. These measures may coincide, but they need not.

Let $\pi _ { k } \geq 0$ , with $\scriptstyle \sum _ { k = 1 } ^ { K } \pi _ { k } = 1$ , denote the structural weights used in the resolution objective. Define

$$
P _ { k , \nu } ^ { e } ( A ) = \int _ { \Theta _ { k } } { P } _ { k , \theta } ^ { e } ( A ) \nu _ { k } ( \mathrm { d } \theta ) , \qquad Q _ { e } = \sum _ { k = 1 } ^ { K } \pi _ { k } P _ { k , \nu } ^ { e } .\tag{3}
$$

Thus $\Theta _ { k }$ specifies which nuisance states are admissible, $\mu _ { k }$ determines how validity is assessed across them, $\nu _ { k }$ determines how resolution is averaged across them, and $\pi _ { k }$ determines how structural families are weighted in the resolution objective.

## 3.2 Randomized nonempty candidate-set rules

After observing $Y = y$ , the terminal decision is a nonempty structural candidate set

$$
\Gamma _ { e } ( y ) \subseteq \{ 1 , \dots , K \} , \qquad \Gamma _ { e } ( y ) \neq \emptyset .
$$

RAED allows this decision rule to be randomized. Formally, for each observation $y ,$ a Markov kernel

$$
\Pi _ { e } ( S \mid y )
$$

assigns probabilities to the nonempty subsets $S \subseteq \{ 1 , \dots , K \}$ . Deterministic candidate-set rules are included as the special case in which $\Pi _ { e } ( \cdot \mid y )$ places probability one on a single set.

For each structural family k, define the inclusion probability

$$
g _ { k , e } ( y ) : = \operatorname* { P r } _ { \Pi _ { e } } \{ k \in \Gamma _ { e } ( y ) \} = \sum _ { S \ni k } \Pi _ { e } ( S \mid y ) .\tag{4}
$$

Thus $g _ { k , e } ( y )$ is the probability, over the rule’s internal randomization, that family k is retained after observing y. Because the reported candidate set is always nonempty,

$$
0 \leq g _ { k , e } ( y ) \leq 1 , \qquad \sum _ { k = 1 } ^ { K } g _ { k , e } ( y ) \geq 1 .\tag{5}
$$

For the expected-cardinality objective and the familywise false-exclusion losses considered below, these inclusion probabilities contain all information needed for optimization. Appendix B shows that every feasible collection of such marginals can be realized by a randomized rule over nonempty candidate sets.

For structural family k and nuisance state $\theta ,$ define the retention probability

$$
C _ { k , e } ( \theta ; g ) : = \mathbb { E } _ { Y \sim P _ { k , \theta } ^ { e } } \left[ g _ { k , e } ( Y ) \right] ,\tag{6}
$$

and the corresponding false-exclusion loss

$$
L _ { k , e } ( \theta ; g ) : = 1 - C _ { k , e } ( \theta ; g ) .\tag{7}
$$

Here $C _ { k , e } ( \theta ; g )$ is the probability that the candidate set retains the true structural family when the data are generated under $( k , \theta )$ , while $L _ { k , e } ( \theta ; g )$ is the probability that the true family is excluded.

## 3.3 Validity requirements

RAED controls false exclusion separately within each structural family. At the pointwise endpoint, validity requires

$$
\operatorname* { s u p } _ { \theta \in \Theta _ { k } } L _ { k , e } ( \theta ; g ) \le \alpha , \qquad k = 1 , \ldots , K .\tag{8}
$$

We use $\delta = 0$ to denote this pointwise-validity endpoint. No nuisance probability measure is required in this case.

For $0 < \delta \leq 1$ , validity is defined through the upper-tail risk of the nuisance-specific false-exclusion loss under $\mu _ { k } { \mathrm { : } }$

$$
\rho _ { \delta , \mu _ { k } } ( L _ { k , e } ) : = \operatorname* { i n f } _ { \eta \in \mathbb { R } } \left\{ \eta + \frac { 1 } { \delta } \int _ { \Theta _ { k } } \bigl ( L _ { k , e } ( \theta ; g ) - \eta \bigr ) _ { + } \mu _ { k } ( \mathrm { d } \theta ) \right\} ,\tag{9}
$$

where $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . The corresponding Tail-validity requirement is

$$
\begin{array} { r } { \rho _ { \delta , \mu _ { k } } ( L _ { k , e } ) \leq \alpha , \qquad k = 1 , \ldots , K . } \end{array}\tag{10}
$$

For $0 < \delta < 1$ , this criterion controls the average false-exclusion loss in the upper δ-tail of the nuisance distribution. Smaller values of $\delta$ therefore place greater emphasis on adverse nuisance states. $\mathrm { A t } ~ \delta = 1$ ，

$$
\begin{array} { r } { \rho _ { 1 , \mu _ { k } } ( L _ { k , e } ) = \mathbb { E } _ { \theta \sim \mu _ { k } } L _ { k , e } ( \theta ; g ) , } \end{array}
$$

so Tail validity reduces to nuisance-average validity.

For later use, let $\mathcal { G } _ { e } ( \alpha , \delta )$ denote the set of randomized nonempty candidate-set rules satisfying the corresponding familywise validity requirement: (8) when $\delta = 0$ , and (10) when $0 < \delta \leq 1$

## 3.4 Resolution objective and experiment choice

For experiment e and candidate-set rule g, define the expected candidate-set size

$$
J _ { e } ( g ) : = \mathbb { E } _ { Y \sim Q _ { e } } \left[ \mathbb { E } _ { \Pi _ { e } } ( | \Gamma _ { e } ( Y ) | \mid Y ) \right] = \mathbb { E } _ { Y \sim Q _ { e } } \left[ \sum _ { k = 1 } ^ { K } g _ { k , e } ( Y ) \right] .\tag{11}
$$

The inner expectation averages over any randomization of the terminal rule, while the outer expectation averages over the predictive observation distribution $Q _ { e }$ . Thus $J _ { e } ( g )$ is the expected number of structural families retained after experiment e.

For a fixed experiment, define its optimal valid ambiguity by

$$
V _ { e } ^ { \star } ( \alpha , \delta ) : = \operatorname* { i n f } _ { g \in { \mathcal { G } } _ { e } ( \alpha , \delta ) } J _ { e } ( g ) ,\tag{12}
$$

where $\mathcal { G } _ { e } ( \alpha , \delta )$ denotes the candidate-set rules satisfying the validity requirement defined above. Because every rule returns a nonempty set and the rule that always retains all K families is feasible,

$$
1 \leq V _ { e } ^ { \star } ( \alpha , \delta ) \leq K .
$$

For convenience, we also define the normalized resolution

$$
R _ { e } ^ { \star } ( \alpha , \delta ) : = \frac { K - V _ { e } ^ { \star } ( \alpha , \delta ) } { K - 1 } \in [ 0 , 1 ] .\tag{13}
$$

Here $R _ { e } ^ { \star } = 1$ corresponds to singleton resolution and $R _ { e } ^ { \star } = 0$ to retaining all K structural families. We use expected candidate-set size as the primary measure and normalized resolution when comparisons across problems with diferent numbers of structural families are useful.

RAED selects

$$
e ^ { \star } ( \alpha , \delta ) \in \arg \operatorname* { m i n } _ { e \in { \mathcal E } } V _ { e } ^ { \star } ( \alpha , \delta ) .\tag{14}
$$

Thus experiments are compared by the smallest expected structural ambiguity they can achieve while satisfying the same declared validity requirement.

## 3.5 Relationship to least-ambiguous set-valued classification

At the nuisance-average endpoint, the fixed-experiment RAED problem has a direct connection to classical set-valued classification.

Proposition 3.1 (Fixed-experiment nuisance-average reduction). Fix an experiment e and set $\delta = 1$ . Suppose that $\mu _ { k } = \nu _ { k }$ for every structural family, and define

$$
P _ { k } : = P _ { k , \mu } ^ { e } , \qquad Q _ { e } : = \sum _ { k = 1 } ^ { K } \pi _ { k } P _ { k } .
$$

If mandatory nonemptiness is removed and the same randomized Markov-kernel decision class is used, then fixed-experiment RAED is exactly the randomized classwise least-ambiguous set-valued classification problem

$$
\operatorname* { m i n } _ { \Gamma } \mathbb { E } _ { Q _ { e } } | \Gamma ( Y ) | \qquad s u b j e c t \ t o \qquad P _ { k } \{ k \in \Gamma ( Y ) \} \geq 1 - \alpha , \quad k = 1 , \ldots , K .\tag{15}
$$

If an optimum of this randomized classical problem can be chosen nonempty $Q _ { e }$ -almost surely, then the same optimum value is attainable by RAED. Relative to formulations restricted to deterministic set-valued rules, RAED additionally permits randomization and requires the reported candidate set to be nonempty.

Proof. At $\delta = 1$ , the nuisance-tail validity criterion reduces to nuisance-average false-exclusion control:

$$
\begin{array} { r } { \mathbb { E } _ { \theta \sim \mu _ { k } } L _ { k , e } ( \theta ; g ) \le \alpha . } \end{array}
$$

Using the definition of $L _ { k , e }$ and Fubini’s theorem,

$$
\begin{array} { r } { \mathbb { E } _ { \theta \sim \mu _ { k } } L _ { k , e } ( \theta ; g ) = 1 - \mathbb { E } _ { \theta \sim \mu _ { k } } \mathbb { E } _ { Y \sim P _ { k , \theta } ^ { e } } g _ { k , e } ( Y ) } \end{array}\tag{16}
$$

$$
= 1 - \mathbb { E } _ { Y \sim P _ { k } } g _ { k , e } ( Y ) .\tag{17}
$$

Hence the RAED validity constraint is equivalent to

$$
\begin{array} { r } { \mathbb { E } _ { Y \sim P _ { k } } g _ { k , e } ( Y ) \ge 1 - \alpha , } \end{array}
$$

which is precisely the classwise retention constraint

$$
P _ { k } \{ k \in \Gamma ( Y ) \} \geq 1 - \alpha
$$

for the randomized candidate-set rule.

Likewise, for every observation $y .$

$$
\mathbb { E } _ { \Pi _ { e } } [ | \Gamma _ { e } ( y ) | ] = \sum _ { k = 1 } ^ { K } g _ { k , e } ( y )
$$

so the RAED objective is

$$
J _ { e } ( g ) = \mathbb { E } _ { Y \sim Q _ { e } } \left[ \sum _ { k = 1 } ^ { K } g _ { k , e } ( Y ) \right] = \mathbb { E } _ { Q _ { e } } | \Gamma _ { e } ( Y ) | .
$$

Thus the objective and classwise coverage constraints coincide exactly with those of randomized least-ambiguous set-valued classification [9]. The only additional RAED restriction is mandatory nonemptiness. If a classical optimum is nonempty $Q _ { e } \mathrm { - a l m o s t }$ surely, it can therefore be represented as a RAED rule with the same objective value. □

## 3.6 Scientific validity and statistical certification

RAED is defined by two scientific parameters: the false-exclusion tolerance $\alpha$ and the nuisance-tail level δ. Together, they specify the validity requirement that a candidate-set rule must satisfy.

When this requirement is verified from finite data, a separate confidence parameter $\zeta \in ( 0 , 1 )$ is introduced. It controls the allowed probability that the statistical certification procedure fails to establish the desired validity guarantee. Thus α and $\delta$ specify the scientific target, whereas $\zeta$ specifies the confidence with which that target is certified from finite samples.

This distinction becomes important in Section 5. Calibration is a statistical procedure for certifying a learned rule against the RAED validity requirement; it does not change the validity requirement itself.

## 4 Information and Structural Resolution under Nuisance Uncertainty

Information about the structural label does not necessarily translate into a precise structural conclusion when nuisance uncertainty persists after the experiment. The central obstruction is that the same observation can acquire diferent structural meanings under diferent nuisance states.

## 4.1 Cross-nuisance structural aliasing

Cross-nuisance structural aliasing arises when an experiment distinguishes the structural families at each fixed nuisance state, but the mapping from observations to structural labels changes with that state. Since the terminal rule observes Y but not the nuisance state, it must use a single mapping across all nuisance states covered by the validity requirement. The following construction makes this separation exact.

To construct the nuisance-independent reference experiment used below, we use the $K \mathrm { - a r y }$ symmetric channel. For

$$
q \in \left[ 0 , \frac { K - 1 } { K } \right] ,
$$

let ${ \mathsf { C } } _ { K } ( { \boldsymbol { q } } )$ denote the channel

$$
{ \mathsf C } _ { K } ( q ) _ { k j } = \left\{ \displaystyle { \frac { 1 - q } { K - 1 } } , \quad j \neq k . \right.
$$

Under a uniform structural prior, its mutual information is

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { T } _ { K } ( q ) = \log K - h ( q ) - q \log ( K - 1 ) , } \end{array} } \end{array}\tag{18}
$$

where

$$
h ( q ) = - q \log q - ( 1 - q ) \log ( 1 - q )
$$

is the binary entropy function. The function $\mathcal { T } _ { K } ( \boldsymbol { q } )$ is strictly decreasing for

$$
0 \leq q < \frac { K - 1 } { K } .
$$

Fix $K \geq 2$ and parameters

$$
0 < \varepsilon < \beta \leq \alpha < 1 - \frac { 1 } { K } .
$$

Let

$$
\mathbb { Z } _ { K } = \{ 0 , \dots , K - 1 \} , \qquad M \sim \operatorname { U n i f o r m } ( \mathbb { Z } _ { K } ) ,
$$

and let the persistent nuisance state $Z \in \mathbb { Z } _ { K }$ be independent of $M .$ . The validity and eficiency nuisance distributions are taken to coincide, with

$$
\mu ( Z = 0 ) = \nu ( Z = 0 ) = 1 - \varepsilon , \qquad \mu ( Z = z ) = \nu ( Z = z ) = \frac { \varepsilon } { K - 1 } , \quad z \neq 0 .
$$

We compare two experiments. Under the aliased experiment $e _ { I }$

$$
Y = M \oplus Z { \pmod { K } } ,
$$

where $\bigoplus$ denotes addition modulo K. Thus the structural mapping is perfectly distinguishable at each fixed nuisance state, but changes with Z. Under the nuisance-independent reference experiment $e _ { R } ,$

$$
P ( Y = j ~ \vert ~ M = k , Z = z , e _ { R } ) = \mathsf C _ { K } ( \beta ) _ { k j } ,
$$

so the observation law depends on M but not on the nuisance state.

Theorem 4.1 (Cross-nuisance aliasing separation and exact Tail frontier). For the experiments $e _ { I }$ and $e _ { R }$ defined above, the following statements hold.

(i) For every fixed nuisance state $z ,$ experiment $e _ { I }$ identifies the structural label perfectly:

$$
I ( M ; Y \mid Z = z , e _ { I } ) = \log K .
$$

At each fixed $z ,$ its structural channel also Blackwell-dominates the channel of $e _ { R }$

(ii) After nuisance averaging, the structural channels of $e _ { I }$ and $e _ { R }$ are respectively

$$
{ \mathsf { C } } _ { K } ( \varepsilon ) \qquad a n d \qquad { \mathsf { C } } _ { K } ( \beta ) .
$$

Since $\varepsilon < \beta$ , the former strictly Blackwell-dominates the latter. Consequently,

$$
I ( M ; Y \mid e _ { I } ) = \mathcal { Z } _ { K } ( \varepsilon ) > \mathcal { Z } _ { K } ( \beta ) = I ( M ; Y \mid e _ { R } ) ,
$$

and both structural EIG and average Bayes classification strictly prefer $e _ { I }$ to $e _ { R }$ .

(iii) Full-latent EIG gives the same ordering:

$$
I ( ( M , Z ) ; Y \mid e _ { I } ) = \log K > { \cal Z } _ { K } ( \beta ) = I ( ( M , Z ) ; Y \mid e _ { R } ) .
$$

(iv) At the pointwise-validity endpoint,

$$
V _ { e _ { I } } ^ { \star } ( \alpha , 0 ) = K ( 1 - \alpha ) , \qquad V _ { e _ { R } } ^ { \star } ( \alpha , 0 ) = 1 .\tag{19}
$$

(v) For every $0 < \delta \leq 1$ , the exact Tail-RAED value of the aliased experiment is

$$
V _ { e _ { I } } ^ { \star } ( \alpha , \delta ) = \mathrm { m a x } \bigg \{ 1 , \mathrm { m i n } \bigg [ K ( 1 - \alpha ) , K - ( K - 1 ) \frac { \alpha \delta } { \varepsilon } \bigg ] \bigg \} ,\tag{20}
$$

while

$$
V _ { e _ { R } } ^ { \star } ( \alpha , \delta ) = 1 .
$$

Equivalently, define

$$
\delta _ { 0 } = \frac { K \varepsilon } { K - 1 } , \qquad \delta _ { 1 } = \frac { \varepsilon } { \alpha } .
$$

Then

$$
V _ { e _ { I } } ^ { \star } ( \alpha , \delta ) = \left\{ \begin{array} { l l } { K ( 1 - \alpha ) , } & { 0 < \delta \le \delta _ { 0 } , } \\ { K - ( K - 1 ) \frac { \alpha \delta } { \varepsilon } , } & { \delta _ { 0 } < \delta < \delta _ { 1 } , } \\ { 1 , } & { \delta \ge \delta _ { 1 } . } \end{array} \right.\tag{21}
$$

In particular,

$$
V _ { e _ { I } } ^ { \star } ( \alpha , \delta ) > 1 \quad \iff \quad \varepsilon > \alpha \delta .\tag{22}
$$

Thus the information-based criteria above strictly prefer $e _ { I }$ , while RAED strictly prefers the nuisance-independent experiment $e _ { R }$ whenever $\varepsilon > \alpha \delta$ , as well as at the pointwise endpoint.

Proof sketch of Theorem 4.1. Under $e _ { I } , Y = M \oplus Z$ . For every fixed $z ,$ this is a bijection in M, while nuisance averaging produces the structural channel $\mathsf C _ { K } ( \varepsilon )$ . Under $e _ { R } .$ , the structural channel is the nuisance-independent ${ \mathsf { C } } _ { K } ( \beta )$ . Since $\varepsilon < \beta , \mathsf { C } _ { K } ( \beta )$ is a garbling of $\mathsf C _ { K } ( \varepsilon )$ , giving the information and classification orderings in parts $( \mathrm { i } ) { - } ( \mathrm { i i i } )$

For pointwise validity under $e _ { I } .$ , every pair $( k , y )$ is generated with probability one by the nuisance state $z = y \ominus k$ . Any valid rule must therefore satisfy $g _ { k } ( y ) \geq 1 - \alpha$ for every $k , y .$ , which gives $J \geq K ( 1 - \alpha )$ . The bound is attainable by a randomized nonempty rule.

For positive-tail validity, cyclic symmetrization reduces the optimization to two exclusion levels: $\ell _ { 0 }$ on the aligned nuisance state $Z = 0$ , of mass $1 - \varepsilon .$ , and $\ell _ { A }$ on the alias states $Z \neq 0$ , of total mass ε. An exchange argument allows an optimum with $\ell _ { A } \geq \ell _ { 0 }$ , for which

$$
\rho _ { \delta } ( L ) = \left\{ \begin{array} { l l } { \ell _ { A } , } & { \delta \le \varepsilon , } \\ { \{ \varepsilon \ell _ { A } + ( \delta - \varepsilon ) \ell _ { 0 } \} / \delta , } & { \delta > \varepsilon . } \end{array} \right.
$$

Minimizing expected candidate-set size is then a two-variable linear program. Its optimizer changes at $\delta _ { 0 } = K \varepsilon / ( K - 1 )$ and reaches singleton resolution at $\delta _ { 1 } = \varepsilon / \alpha _ { : }$ , yielding (21). The complete proof, including attainability by randomized nonempty rules, is given in Section C.2.

![](images/99009c360bcc8ba00c3385e6ccf10ea94eb14c0683db5718a655d39e8213ac92.jpg)  
Figure 1: Exact Tail frontier in the cyclic-aliasing construction. For $K = 4 , \alpha = 0 . 0 5$ and $\varepsilon = 0 . 0 1$ , the aliased experiment $e _ { I }$ remains at $V _ { e _ { I } } ^ { \star } = 3 . 8$ for the most stringent nuisance tails and then improves with increasing $\delta ,$ reaching singleton resolution at $\delta _ { 1 } = \varepsilon / \alpha = 0 . 2$ . The nuisance-independent experiment $e _ { R }$ achieves $V _ { e _ { R } } ^ { \star } = 1$ throughout.

Corollary 4.2 (Failure of maximin conditional mutual information). In the construction of Theorem $4 . 1 ,$

$$
\operatorname* { i n f } _ { z \in \mathbb { Z } _ { K } } I ( M ; Y \mid Z = z , e _ { I } ) = \log K > \mathbb { Z } _ { K } ( \beta ) = \operatorname* { i n f } _ { z \in \mathbb { Z } _ { K } } I ( M ; Y \mid Z = z , e _ { R } ) .\tag{23}
$$

Hence the maximin criterion max<sub>e</sub> inf $_ { \zeta } I ( M ; Y \mid Z = z , e )$ strictly prefers $e _ { I { \mathrm { : } } }$ , even though RAED strictly prefers $e _ { R }$ whenever $\varepsilon >$ αδ and at the pointwise endpoint. Replacing nuisance-averaged mutual information by its worst-nuisance conditional value therefore does not resolve the aliasing: every nuisance-specific decoder is perfect under $e _ { I }$ , but those decoders are mutually incompatible when Z is unobserved. This statement concerns this particular maximin conditional-MI criterion, not every possible robust information utility.

Proof. For $e _ { I }$ , conditioning on $Z = z$ gives $Y = M \oplus z , \mathrm { ~ a ~ }$ bijection in M, so $I ( M ; Y \mid Z = z , e _ { I } ) =$ log K for every z. For $e _ { R } .$ , the channel is nuisance-independent and equals $\complement _ { K } ( \beta )$ , giving $I ( M ; Y \mid Z = z , e _ { R } ) = \mathcal { I } _ { K } ( \beta ) <$ log K because $\beta > 0$ . The RAED ordering follows from Theorem 4.1.

The separation can also grow with the number of structural families. For every fixed $\bar { \delta } < 1$ δ , the construction can be chosen so that the information-based criteria still prefer $e _ { I }$ while

$$
\frac { V _ { e _ { I } } ^ { \star } ( \alpha , \bar { \delta } ) } { V _ { e _ { R } } ^ { \star } ( \alpha , \bar { \delta } ) } \geq K c _ { \alpha , \bar { \delta } } , \qquad c _ { \alpha , \bar { \delta } } = \operatorname* { m i n } \biggr \{ 1 - \alpha , \frac { 1 - \bar { \delta } } { 1 + \bar { \delta } } \biggr \} > 0 .\tag{24}
$$

Thus the resolution gap can grow linearly with K. The formal statement and proof are given in Corollary C.3.

Beyond exact cyclic aliasing. Exact label permutations are not required for the same mechanism to arise. Suppose there is an eficiency-relevant response law such that, for every structural family, a nuisance region of $\mu _ { k } .$ -mass at least m has response laws within average total variation τ of that common law. The general near-alias result in Theorem C.4 then gives the uniform bound

$$
V _ { e } ^ { \star } ( \alpha , \delta ) \geq \operatorname* { m a x } \Bigl \{ 1 , \ K \bigl [ 1 - \tau - \alpha \kappa _ { \delta } ( m ) \bigr ] _ { + } \Bigr \} , \qquad \kappa _ { \delta } ( m ) = \left\{ 1 , \qquad m \geq \delta , \right.\tag{25}
$$

The bound is nontrivial whenever $K [ 1 - \tau - \alpha \kappa _ { \delta } ( m ) ] _ { + } > 1$ . In the small-mass regime $m < \delta ,$ the controlling term becomes $\alpha \delta / m .$ , showing explicitly that increasingly rare near-alias regions can be tolerated as $\delta$ grows. The exact cyclic construction sharpens this general mechanism to the threshold $\varepsilon > \alpha \delta$

## 4.2 Composite informativeness that RAED respects

The separation in Theorem 4.1 does not imply that RAED reverses genuine Blackwell informativeness. There, the information advantage of $e _ { I }$ appears only after nuisance averaging or after interpreting the observation with a nuisance-specific decoder. A stronger comparison is available when one experiment can simulate another through the same post-processing rule for every structural family and nuisance state.

Proposition 4.3 (Composite Blackwell monotonicity). Let $e _ { 1 }$ and $e _ { 2 }$ share the same structural weights, nuisance sets, and validity and eficiency measures. Suppose there exists a Markov kernel $G ,$ independent of both k and θ, such that

$$
P _ { k , \theta } ^ { e _ { 2 } } = P _ { k , \theta } ^ { e _ { 1 } } G f o r \ e v e r y \ k , \theta .
$$

Then, for every $( \alpha , \delta )$ ，

$$
V _ { e _ { 1 } } ^ { \star } ( \alpha , \delta ) \leq V _ { e _ { 2 } } ^ { \star } ( \alpha , \delta ) .\tag{26}
$$

The condition means that an observation from $e _ { 1 }$ can be post-processed through one common kernel G to reproduce the observation law of $e _ { 2 }$ under every composite state $( k , \theta )$ . Hence any valid candidate-set rule for $e _ { 2 }$ can also be implemented after observing $e _ { 1 } { \mathrm { : } }$ first apply $G _ { i }$ , then apply the $e _ { 2 }$ rule. This preserves the nuisance-specific false-exclusion losses and the expected candidate-set size. Therefore every resolution level achievable with $e _ { 2 }$ is also achievable with $e _ { 1 }$ , and $e _ { 1 }$ cannot have a larger optimal RAED value.

Appendix C.1 extends this result to approximate common garbling. If a single kernel simulates $e _ { 2 }$ from $e _ { 1 }$ within uniform total-variation error ε over all $( k , \theta )$ , then each nuisance-specific falseexclusion loss changes by at most ε and expected candidate-set size by at most $( K - 1 ) \varepsilon$ . The appendix also gives an exact-α repair and the corresponding deficiency bounds. Thus the separation in Section 4.1 is not a general conflict between information and structural resolution; it arises when the claimed informativeness does not hold for the full nuisance-indexed experiment through one common comparison kernel.

## 5 Learned RAED and Finite-Sample Validity

The RAED formulation in Section 3 optimizes over all measurable randomized nonempty candidateset rules. In simulator-based problems this unrestricted optimum is generally unavailable, so we work with a learned family of score-based rules. Scores are trained first, experiments are ranked

on an independent selection sample, and the selected rule is calibrated on separate data. This separation allows finite-sample validity to be certified without assuming that the learned score family attains the oracle RAED frontier.

## 5.1 Score-based candidate sets

At the nuisance-average endpoint, when $P _ { k , \mu } ^ { e } \ll Q _ { e }$ , the optimal classwise retention rule is governed by the density ratio

$$
r _ { k , e } ( y ) = { \frac { \mathrm { d } P _ { k , \mu } ^ { e } } { \mathrm { d } Q _ { e } } } ( y ) , \qquad P _ { k , \mu } ^ { e } ( A ) = \int P _ { k , \theta } ^ { e } ( A ) \mu _ { k } ( \mathrm { d } \theta ) .\tag{27}
$$

Large values of $r _ { k , e } ( y )$ indicate observations that are relatively more characteristic of structural family k than of the overall predictive mixture $Q _ { e }$ . This motivates learning family-specific scores that rank observations by evidence for retaining each family. Under positive-tail validity, the corresponding oracle characterization involves a least-favourable reweighting of the nuisance distribution; the full results are given in Sections D.1 and E.

For each experiment $e ,$ the learned procedure produces scores

$$
s _ { k , e } : \mathcal { V } _ { e } \to \mathbb { R } , \qquad k = 1 , \dots , K .
$$

Let

$$
b _ { e } ( y ) = \arg \operatorname* { m a x } _ { j } s _ { j , e } ( y ) ,
$$

with a fixed deterministic rule for resolving ties. Given family-specific thresholds $t _ { k , e } ,$ define

$$
\Gamma _ { e , t } ( y ) = \{ b _ { e } ( y ) \} \cup \{ k : s _ { k , e } ( y ) \geq t _ { k , e } \} .\tag{28}
$$

The first term guarantees that the reported candidate set is never empty, while the thresholds determine which additional structural families are retained.

## 5.2 Nested finite-sample calibration

The thresholds in (28) control the trade-of between structural resolution and false exclusion. More selective thresholds produce smaller candidate sets but increase the risk of excluding the true family. We therefore choose the deployed thresholds using an independent calibration sample.

For each familywise validity claim $j ,$ organize the candidate-set rules along a one-dimensional ordered path $\{ \Gamma _ { j , \lambda } : \lambda \in \Lambda _ { j } \}$ , where larger values of λ produce larger candidate sets:

$$
\lambda _ { 1 } \leq \lambda _ { 2 } \quad \Longrightarrow \quad \Gamma _ { j , \lambda _ { 1 } } ( y ) \subseteq \Gamma _ { j , \lambda _ { 2 } } ( y ) .
$$

We allow either a continuous interval $\Lambda _ { j } = [ \underline { { \lambda } } _ { j } , \overline { { \lambda } } _ { j } ]$ or a finite ordered grid $\Lambda _ { j } = \{ \lambda _ { j , 1 } < \dots < \lambda _ { j , m _ { j } } \}$ In either case, the largest value is a safe endpoint at which the family associated with claim $j$ is always retained. Let $L _ { j , \lambda } ( \theta ) \in [ 0 , 1 ]$ denote its false-exclusion loss at nuisance state $\theta .$

To state the finite-sample guarantee, we condition on everything fixed before the final calibration sample, including score fitting, experiment selection, and construction of the nested path. We denote this pre-calibration information by $\mathcal { F } _ { 0 }$

Along the nested path, let $R _ { j } ( \lambda )$ denote the true false-exclusion risk for claim $j .$ . Because larger candidate sets retain more families, $R _ { j } ( \lambda )$ is nonincreasing in $\lambda .$ . For an interval-valued path we additionally assume that $R _ { j }$ is continuous. At the safe endpoint,

$$
R _ { j } ( { \overline { { \lambda } } } _ { j } ) = 0 .
$$

The calibration sample is used to place a one-sided upper confidence bound (UCB) on the unknown risk $R _ { j } ( \lambda )$ . Let $\gamma _ { j } \in ( 0 , 1 )$ denote the allowed failure probability for claim $j .$ . For each fixed $\lambda \in \Lambda _ { j }$ , the calibration procedure produces an upper bound $U _ { j } ( \lambda ; \gamma _ { j } )$ satisfying

$$
\begin{array} { r } { \operatorname* { P r } \{ R _ { j } ( \lambda ) \le U _ { j } ( \lambda ; \gamma _ { j } ) \vert \mathcal { F } _ { 0 } \} \ge 1 - \gamma _ { j } . } \end{array}\tag{29}
$$

Thus, with probability at least $1 - \gamma _ { j }$ , the true false-exclusion risk at that fixed rule lies below the reported bound. In the empirical implementation, these bounds are obtained from a bounded-mean Bernoulli–KL concentration inequality; the result below only requires that the stated one-sided coverage guarantee holds.

The goal is to retain as much structural resolution as possible while certifying risk below $\alpha .$ . We therefore enlarge the candidate set only until the upper confidence bound falls below α and remains below α for every more conservative rule. Define

$$
\widehat { \lambda } _ { j } : = \operatorname* { i n f } \left\{ \lambda \in \Lambda _ { j } : U _ { j } ( \lambda ^ { \prime } ; \gamma _ { j } ) < \alpha \mathrm { ~ f o r ~ e v e r y ~ } \lambda ^ { \prime } \geq \lambda \right\} ,\tag{30}
$$

using the safe endpoint $\overline { { \lambda } } _ { j }$ if no earlier value satisfies this condition.

Theorem 5.1 (Finite-sample calibration of a nested RAED path). Under the conditions above, and assuming $\widehat { \lambda } _ { j }$ is a measurable function of the pre-calibration information and the calibration sample, the selected rule satisfies

$$
\mathrm { P r } \Big \{ R _ { j } ( \widehat { \lambda } _ { j } ) \leq \alpha \Big | \mathcal { F } _ { 0 } \Big \} \geq 1 - \gamma _ { j } .\tag{31}
$$

The nested structure is what allows the threshold to be chosen from the calibration data without applying a separate multiplicity correction to every candidate value of $\lambda .$ For a continuous path, invalid selection forces the upper confidence bound to fail at the population crossing point $R _ { j } ( \lambda ) = \alpha$ For a finite threshold grid, the corresponding witness is the last invalid threshold on that grid. In either case, the argument requires only one population-determined boundary point. The full proof is given in Section D.2.

## 5.3 Positive-tail calibration and independent certification

For $0 < \delta < 1$ , the risk controlled in Section 5.2 is the upper-δ Tail risk of the nuisance-specific false-exclusion losses. For a familywise claim $j$ associated with structural family $k ,$ this is

$$
R _ { j } ( \lambda ) = \rho _ { \delta , \mu _ { k } } ( L _ { j , \lambda } ) .
$$

Unlike nuisance-average risk, this quantity is not itself an ordinary mean. We therefore use the variational representation introduced in (9) to reduce its calibration to that of a bounded mean.

For each $\lambda ,$ choose an auxiliary value $\eta _ { j } ( \lambda ) \in [ 0 , 1 ]$ before the final calibration sample is opened. Then

$$
\rho _ { \delta , \mu _ { k } } ( L _ { j , \lambda } ) \leq \eta _ { j } ( \lambda ) + \frac { 1 } { \delta } \mathbb { E } \left[ \left( L _ { j , \lambda } ( \theta ) - \eta _ { j } ( \lambda ) \right) _ { + } \right] .\tag{32}
$$

Equality is obtained when $\eta _ { j } ( \boldsymbol { \lambda } )$ minimizes the variational expression, but any fixed choice gives a valid upper bound. In the implementation, the complete map $\lambda \mapsto \eta _ { j } ( \lambda )$ is constructed from the independent selection sample and fixed before final calibration.

The right-hand side of (32) can be calibrated using the same bounded-mean concentration machinery as in Section 5.2. In particular, a one-sided upper confidence bound for the mean of

$$
\delta \eta _ { j } ( \lambda ) + \left( L _ { j , \lambda } ( \theta ) - \eta _ { j } ( \lambda ) \right) _ { + }
$$

gives, after division by $\delta ,$ a one-sided upper confidence bound for the Tail risk. Applying Theorem 5.1 along the same nested threshold path therefore gives

$$
\mathrm { P r } \Big \{ \rho _ { \delta , \mu _ { k } } \big ( L _ { j , \widehat { \lambda } _ { j } } ^ { } \big ) \leq \alpha \Big | \mathcal { F } _ { 0 } \Big \} \geq 1 - \gamma _ { j } .\tag{33}
$$

The full argument is given in Section D.3. $\mathrm { A t } ~ \delta = 1$ , Tail risk reduces to nuisance-average risk and is covered directly by Theorem 5.1.

Repeated observation-noise draws at the same nuisance state improve the estimate of that state’s false-exclusion loss, but they do not provide additional independent nuisance states. The efective sample size for population Tail certification is therefore the number of independently sampled nuisance states. Appendix D.4 shows that estimating each nuisance-specific loss by inner Monte Carlo remains conservative for the Tail certificate.

Because experiment selection is completed before the final calibration sample is opened, certification only needs to cover the familywise validity claims of the selected experiment. At a fixed δ, assigning

$$
\gamma _ { k } = \frac \zeta K , \qquad k = 1 , \dots , K ,
$$

gives joint confidence at least $1 - \zeta$ across the K structural families. Independence of the selection and calibration roles avoids an additional multiplicity penalty for the original experiment library, while the nested construction of Theorem 5.1 avoids one over the threshold path. The general selection-aware result is given in Theorem D.4.

The resulting workflow is therefore to train the scores and select the experiment first, fix all choices that precede calibration, calibrate the selected rule on independent nuisance states, and only then use the independent evaluation data. These finite-sample guarantees apply for $0 < \delta \leq 1$ . At $\delta = 0$ , validity is required at every admissible nuisance state rather than over a nuisance population, so the iid calibration argument used here no longer applies.

## 5.4 The statistical price of small nuisance tails

Nested calibration avoids paying separately for every threshold considered, but it cannot compensate for nuisance states that are too rare to appear in the calibration sample. A simple construction gives a necessary sample-size scale. Consider a population in which the false-exclusion loss is zero almost everywhere but equals one on a nuisance region of probability

$$
p = \delta ( \alpha + \varepsilon ) , \qquad \varepsilon > 0 .
$$

Its upper-δ Tail risk is $\alpha + \varepsilon ,$ and is therefore just above the allowed level α. Yet n independent nuisance draws miss the adverse region entirely with probability $( 1 - p ) ^ { n }$ . Any distribution-free procedure that would certify after observing no losses must therefore make this event suficiently unlikely.

Proposition 5.2 (Rare-tail nuisance-sampling obstruction). Fix $0 < \alpha < 1 , 0 < \delta \leq 1$ , and an allowed false-certification probability $0 < \beta < 1$ . Any deterministic distribution-free procedure that certifies Tail risk at most α after an all-zero calibration sample must satisfy, for every $\varepsilon > 0$ with α $+ \varepsilon \leq 1$ ，

$$
\left( 1 - \delta ( \alpha + \varepsilon ) \right) ^ { n } \leq \beta .\tag{34}
$$

Consequently, when αδ is small,

$$
n = \Omega \left( \frac { \log ( 1 / \beta ) } { \alpha \delta } \right) .\tag{35}
$$

Table 2: Overview of the empirical benchmarks.
<table><tr><td>Benchmark</td><td>Structural target</td><td>Experimental design</td><td>Main role</td></tr><tr><td>Watt</td><td>Fault-model ontologies  $( K = 3 , 6 , 6 , 1 2 )$ </td><td>Active well tests; full or restricted pressure observation</td><td>Ontology and constrained-sensing studies</td></tr><tr><td>WCA</td><td>Three depositional architectures</td><td>Directional interference tests with restricted monitoring</td><td>Constrained experiment selection</td></tr><tr><td>GWAE- Fluvial</td><td>One versus two spanning channels</td><td>Nine active well tests</td><td>Tail versus nuisance-average design</td></tr><tr><td>Methane oxidation</td><td>Three kinetic mechanisms</td><td>81 reactor operating conditions</td><td>Positive-tail certification</td></tr></table>

The complete two-point argument is given in Appendix D.6. The result shows why stringent Tail certification can require many independent nuisance states even when the observed calibration losses are small. This limitation motivates using Watt and WCA as computation-limited held-out experiment-selection comparisons, while the separately powered GWAE-Fluvial and methane studies support fixed-positive-δ population certification.

## 6 Benchmark Construction and Experimental Protocol

We evaluate RAED on three reservoir-engineering benchmarks—Watt, WCA, and GWAE-Fluvial— and on a methane-oxidation mechanism-discrimination benchmark. The reservoir studies use active well tests to distinguish competing structural descriptions under persistent geological uncertainty, while methane provides a cheaper mechanistic setting in which finite-sample guarantees for positivetail false-exclusion risk can be powered directly. Table 2 summarizes the structural target, nuisance variables, physical design space, and empirical role of each benchmark. We first describe how the four benchmark problems were constructed and then give the common learned-RAED and comparator protocol.

## 6.1 Watt: distinguishing fault architectures with active well tests

The Watt field model of Arnold et al. [26] is a synthetic reservoir benchmark built around several alternative geological descriptions of the same field. The published benchmark combines three fault models (FM1–FM3), three cutof alternatives, three grid representations, and three top-structure alternatives, giving $3 ^ { 4 } = 8 1$ base scenarios. For the RAED study we retain the three fault-model alternatives while fixing the remaining hierarchy at CO3, G3, and TS2. This gives a common reservoir framework in which the principal structural diference is the interpretation of the fault system.

FM1–FM3 contain the same major named faults, but difer in their traces, spatial extent, segmentation, and therefore the compartment connections they impose on the reservoir. Each fault model is additionally combined with the two supplied property representations, OBJECT and $\mathrm { P I X E L } ,$ and the two relative-permeability families, RP0 and RP1. Seven transmissibility multipliers further vary the strength of the existing major faults. The resulting ensemble therefore contains several sources of uncertainty around each fault interpretation rather than three deterministic reservoir models.

From this common physical ensemble, we define four alternative model ontologies. For the $K = 3$ case, $M \in \{ \mathrm { F M 1 , F M 2 , F M 3 } \}$ , so that only the fault model forms part of the structural identity. We also consider two six-model descriptions, $\mathrm { F M } { \times } \mathrm { P R O P }$ and $\mathrm { F M } \times \mathrm { R P } ,$ , and the twelve-model description $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ . Here PROP distinguishes the two supplied property representations, OBJECT and PIXEL, which provide alternative spatial porosity and permeability fields for the same geological scenario, while RP distinguishes the two supplied relative-permeability families, RP0 and RP1. When PROP or RP is not included in the structural identity, it remains part of the within-family nuisance state. In every case, seven continuous fault-transmissibility multipliers also remain nuisance variables. The four ontologies therefore allow the same physical ensemble to be examined at progressively finer distinctions between structural identity and within-model uncertainty.

For each reservoir realization we simulate ten well tests. The experiment library follows established pressure-transient and well-testing practice, combining injector and producer multirate tests, pulse/fallof experiments, periodic rate perturbations, and a producer drawdown/build-up test [27, 28, 29]. Each test perturbs one designated well while the remaining wells continue under their common background controls. The response records the actuator bottom-hole pressure and realized rate together with pressure at three remote wells. These tests are intended to probe pressure communication through the reservoir and thereby expose diferences in fault-controlled connectivity.

The resulting ensemble contains 600 reservoir realizations, each simulated under the ten well tests, giving 6,000 full-physics OPM Flow responses. The simulations use the reconstructed $1 1 2 \times 3 0 \times 4 0$ Watt model and a common equilibrium initial state. Details of the source-model reconstruction, initialization, and ensemble generation are given in Section G.1.

The Watt ensemble is used in two complementary ways. First, using the full well response, we analyse all four model ontologies, with $K = 3 , 6 , 6$ , and 12. This isolates the efect of model definition: the underlying reservoir realizations and physical experiments are unchanged, while progressively finer distinctions in fault model, property representation, and relative permeability are promoted from nuisance to components of the structural identity.

We then consider a separate constrained-sensing study using the K = 3 fault-family ontology. Here the structural question is fixed and the available observation is varied instead. The response is truncated at $T \in \{ 7 , 1 4 , 3 0 , 9 0 \}$ days, and pressure is retained from 0–3 remote wells. Actuator pressure and realized rate are always observed. Each combination of observation horizon and number of remote wells therefore defines a common sensing level within which the ten well tests and admissible monitor locations are compared.

The first study examines how achievable resolution changes as the structural distinction becomes finer. The second keeps the structural target fixed and examines how experiment choice changes when observation time and remote pressure measurements are restricted.

## 6.2 WCA: depositional-architecture discrimination by pressure interference

The WCA benchmark is based on the multiple-geological-interpretation study of Park et al. [30]. The source study considers three alternative training images, TI1–TI3, representing diferent fluvial depositional architectures. TI1 is characterized by relatively thin, low-sinuosity channels, TI2 by thicker low-sinuosity channels, and TI3 by a more sinuous channel system with associated levee architecture. Their overall sand/shale proportions are similar, so the main distinction between the three interpretations lies in the geometry and connectivity of the permeable bodies rather than in gross facies volume.

For each training image we retain ten conditional facies realizations that honor the source hard data. Each realization is a diferent spatial arrangement of the channel, levee, and shale facies that is compatible with the same parent training image. These realizations are not treated as separate geological hypotheses. Instead, $M \in \{ \mathrm { T I 1 } , \mathrm { T I 2 } , \mathrm { T I 3 } \}$ defines the structural identity. Within each family, the nuisance state consists of the particular conditional facies realization together with its porosity and permeability fields. For a fixed facies geometry, these properties are varied using source-derived distributions and spatially correlated random fields, so reservoirs belonging to the same structural family can difer substantially in both local rock quality and flow response.

Constructing the dynamic benchmark required placing these geological realizations in a common reservoir model. The categorical facies fields were mapped to a $5 3 \times 4 0 \times 1 1 6$ simulation grid chosen to preserve the main geometric and directional connectivity characteristics of the source models. Porosity and permeability were then assigned within the facies using the source-derived property distributions and spatially correlated random fields described above. A common physical scale, fluid model, initial state, and oil–water contact were used across all realizations, together with a fixed $3 \times 3$ array of nine wells. Wells are completed only in active reservoir cells above the oil–water contact. Full details of the geological reconstruction, petrophysical mapping, and dynamic model are given in Section G.2.

The WCA experiment library follows the same well-testing principles as Watt and comprises ten active interference tests on the nine-well layout. Six tests probe reservoir communication along reciprocal east–west, north–south, and diagonal injector–producer directions. The remaining four use multirate excitation, alternating injection, a balanced multiwell interference pattern, and a central pulse/recovery test. Together, the experiments probe diferent flow paths and timescales through the competing depositional architectures. Each training image contributes 1000 reservoir realizations, giving 3,000 realizations in total. Simulating every realization under the ten well tests produces 30,000 full-physics OPM Flow responses.

As in Watt, the complete responses are used both under rich observation and under a separate constrained-sensing analysis. In the constrained case, the response is truncated at $T \in \{ 7 , 1 4 , 3 0 , 6 0 \}$ days, and pressure is retained from 0–3 remote wells. Actuator pressure and realized rate are always observed. The structural target remains fixed at the three parent training images, so this analysis isolates the efect of reduced observation time and spatial coverage on experiment choice and achievable resolution.

## 6.3 GWAE-Fluvial: one- versus two-channel discrimination

The GWAE-Fluvial study builds on the single- and double-channel reservoir structures described by Shishaev, Demyanov, and Arnold [31] and the associated geological prior information reported in that work. We use the reported channel dimensions and structural ranges to construct a parametric fluvial ensemble on the supplied reservoir model, with one- and two-channel families defined directly in terms of their geological geometry and petrophysical variability.

The structural target is M = 1 for a reservoir containing one spanning channel and $M = 2$ for a reservoir containing two separated channels. Within each family, the nuisance variables describe the particular channel geometry and rock properties. Channels are represented as finite sinusoidal ribbons spanning the reservoir, with nuisance variation in width, thickness, wavelength, phase, lateral and vertical position, and petrophysical quality. In the two-channel family, the two channels share the same broad geometric form but may difer in width, thickness, vertical position, and lateral separation. Exact parameter ranges, geometric constraints, and the porosity/permeability construction are given in Section G.3.

The geological realizations are simulated on the supplied reservoir grid using a common OPM Flow model and well configuration, with west and east injector banks separated by three central producers. Nine well tests are considered, including simultaneous and single-bank injection, crossreservoir and central-pair tests, staged-rate excitation, pulse/fallof, and alternating-bank injection. Responses are recorded at days 1, 3, 5, 7, and 10 and consist of injector rates and pressures together with producer oil and water production rates.

The ensemble contains 24,000 geological realizations, each simulated under the nine well tests, giving 216,000 full-physics reservoir responses. This population is used to compare nuisance-average RAED at $\delta = 1$ with Tail-RAED at $\delta = 0 . 0 5$ , both at $\alpha = 0 . 0 5$ . The earlier nuisance-average analysis in Section 7.3 uses a separate 10,000-realization population, whereas the Tail-versus-average comparison is carried out on the independent 24,000-realization ensemble described here. This allows us to examine whether protecting against a small adverse region of geological uncertainty changes the preferred well test and the resolution obtained after calibration.

## 6.4 Methane oxidation: a powered certification benchmark

The methane-oxidation study follows the complete-oxidation discrimination problem of Pankajakshan et al. [32]. The structural target consists of three rival kinetic mechanisms, $M \in \{ \mathrm { P L } , \mathrm { L H } , \mathrm { M V K } \}$ , corresponding to power-law, Langmuir–Hinshelwood, and Mars–van Krevelen rate models. Within each mechanism, persistent nuisance parameters follow the fitted multivariate Gaussian approximation reported in the source study.

The physical design space is a full $3 ^ { 4 }$ factorial over temperature, total feed flow, inlet oxygen-tomethane ratio, and inlet methane fraction, giving 81 candidate reactor conditions. Each experiment observes the outlet mole fractions of methane, oxygen, and carbon dioxide under the source-derived measurement-noise model.

Methane is used primarily to test finite-sample positive-tail calibration. Its steady reactor mode is inexpensive enough to generate much larger independent nuisance samples than the reservoir benchmarks. We therefore use 10,000 calibration nuisance states per structural family, together with a separate 5,000-state evaluation set, so that nontrivial positive-tail population guarantees can be assessed directly.

The score-training, experiment-selection, calibration, and evaluation nuisance populations are kept disjoint. At each fixed positive tail level $\delta > 0$ , the familywise calibration confidence is allocated equally across the three mechanisms, giving 95% joint confidence across PL, LH, and MVK. Exact nuisance distributions, reactor equations, design conditions, and calibration details are given in Section H.

## 6.5 Common learned-RAED protocol

All four benchmarks follow the learned-RAED workflow of Section 5. Scores are fitted first, experiments are ranked on an independent nuisance sample, the selected experiment is then calibrated on a separate sample, and final performance is measured on independent evaluation states.

Watt, WCA, and GWAE-Fluvial use one-versus-rest logistic scores on a 256-component radialbasis-function (RBF) Nyström representation. Methane uses the same construction with 96 Nyström components. Cross-validation is grouped by nuisance state, so repeated observation-noise realizations of the same physical state remain together.

The nuisance states used for score fitting, experiment selection, calibration, and evaluation are disjoint in every benchmark. Repeated noise draws at a fixed nuisance state are used only to estimate its response distribution and conditional false-exclusion loss, not as additional independent nuisance samples.

After experiment selection, only the selected experiment is calibrated. Watt and WCA use this protocol for held-out empirical comparison. Methane and GWAE-Fluvial use larger calibration populations to support finite-sample positive-tail guarantees. Exact sample sizes, score settings, noise replication, and calibration details are given in Appendix F.

## 6.6 Comparator criteria and downstream evaluation

RAED is compared with model-index expected information gain, full-latent expected information gain, Bayes forced classification, expected model elimination, and minimum ordered-pair predictive discrimination. All comparator criteria are evaluated on the same independent experiment-selection population used for RAED ranking.

The comparison is made at the level of experiment choice. Once a criterion selects an experiment, the selected experiment is evaluated using the same learned candidate-set construction, calibration procedure, and independent evaluation population as the RAED-selected experiment. Diferences in the reported structural resolution can therefore be attributed to the selected experiment rather than to a diferent decision rule.

The EIG criteria are estimated by Monte Carlo and can be numerically indistinguishable near their optimum. We therefore distinguish a diferent numerical maximizer from a clear selection disagreement: RAED and EIG are counted as selecting diferently only when the RAED-selected experiment lies outside the predefined Monte Carlo uncertainty set around the EIG optimum. Exact estimators, sampling counts, and tie rules are given in Appendix F.

## 7 Results

## 7.1 Structural granularity and achievable resolution in Watt

![](images/264a2e7ef05c08f028402452c0015bc664ced1eed379f50e8d4ce938e2a65ef7.jpg)  
Figure 2: Achievable structural resolution across Watt model ontologies. Held-out mean candidate-set size J for the four structural targets. The same INJ5 pulse/fallof test is selected in every case at $( \alpha , \delta ) = ( 0 . 0 5 , 1 )$ .

The Watt ensemble provides a direct test of how the definition of the structural target afects achievable resolution, because the same reservoir realizations, well tests, and observations can be evaluated under all four model ontologies. $\mathrm { A t } \left( \alpha , \delta \right) = \left( 0 . 0 5 , 1 \right)$ , using the full well response, learned

RAED selected the same INJ5 pulse/fallof test for the K = 3 FM, K = 6 FM×PROP, $K = 6$ FM×RP, and $K = 1 2 { \mathrm { ~ F M } } { \times } { \mathrm { P R O P } } { \times } { \mathrm { R P } }$ targets. The corresponding held-out mean candidate-set sizes are J = 1.421, 1.438, 1.525, and 1.599. Thus progressively finer distinctions increase the absolute number of surviving explanations only modestly. Because K changes across the four ontologies, J and the normalized resolution R describe complementary aspects of the result: J gives the expected number of structural explanations that remain, whereas R measures resolution relative to the size of the candidate model set. We emphasize J here because it retains the direct candidate-set interpretation of the RAED objective. Along both refinement paths, normalized resolution also increases. Both quantities are reported in Table 13. The corresponding worst-family held-out nuisance-average false-exclusion rates are 0.056, 0.067, 0.057, and 0.061, respectively. These values are close to, but slightly above, the nominal $\alpha = 0 . 0 5$ level. The complete resolution and exclusion results are reported in Table 13.

When PROP and RP are treated as nuisance in the $K = 3$ fault-family problem, RAED must distinguish FM1–FM3 in the presence of the response variability induced by those additional modeling choices. The resulting $J \approx 1 . 4 2$ therefore represents strong empirical fault-family resolution under substantial within-family variation. When PROP and RP are instead promoted into the structural identity, the candidate model set grows from three to twelve, yet the held-out mean candidate-set size increases only to about 1.60. The full well response therefore retains substantial discriminatory power even for the finer structural distinctions, although the aggregate value of J alone does not identify which particular model pairs account for the remaining ambiguity.

The fact that the same INJ5 pulse/fallof test is selected across all four ontologies is also informative. Its value is not confined to the coarsest fault-family problem: the same physical experiment remains preferred as property representation and relative-permeability family are progressively incorporated into the structural target. The modest increase in J shows that these additional distinctions do introduce further ambiguity, but much less than might be expected from the fourfold increase in the number of candidate models.

These Watt ontology results serve a diferent purpose from the constrained-sensing comparisons studied later in this section. Here the physical reservoir ensemble, well tests, and observations are unchanged; only the definition of structural identity is varied. The comparison therefore isolates how achievable resolution changes when attributes that were previously treated as nuisance are instead required to be distinguished explicitly. The reported values are held-out empirical summaries of the nontrivial learned rules rather than finite-sample population certificates. The available Watt calibration sample does not certify these nontrivial rules at the declared 5% level, so the ontology study provides an empirical resolution analysis.

## 7.2 Constrained sensing separates RAED and information gain

The constrained Watt and WCA studies keep the structural target fixed while progressively reducing the available observation time and number of remote pressure measurements. We focus on the operating point $( \alpha , \delta ) = ( 0 . 0 5 , 0 . 0 5 )$ . Within each sensing level, RAED and model-index EIG rank the same set of admissible well tests and monitor configurations using the same independent selection data. The two criteria therefore difer only in how they select the experiment. Model-index EIG does not itself produce a candidate set: after each criterion selects an experiment, that experiment is evaluated through the same learned-RAED downstream procedure, including score fitting, threshold calibration, and independent held-out evaluation.

In WCA, RAED and model-index EIG are cleanly separated in seven of the sixteen horizon– monitor combinations after accounting for the finite-sample Monte Carlo error in the EIG estimates. In five of these seven cells, the RAED-selected experiment yields a smaller held-out candidate set, with reductions in J between 0.104 and 0.385. One cell is essentially tied, with a diference of −0.004, and one yields a smaller candidate set for the EIG-selected experiment by 0.118. Across all sixteen sensing levels, the experiments selected by RAED give a mean held-out candidate-set size of $J = 2 . 5 2 0$ , compared with $J = 2 . 6 5 8$ for those selected by model-index EIG.

These diferences in candidate-set size are accompanied by substantial variation in held-out Tail false-exclusion risk. In several WCA cells, the smaller candidate sets obtained after the RAEDselected experiment also have higher worst-family upper-5%-Tail exclusion than those obtained after the EIG-selected experiment. For example, at $( T , B ) = ( 7 , 2 )$ days and remote wells, the candidate-set sizes are 2.378 and 2.763 for the RAED- and EIG-selected experiments, respectively, while the corresponding worst-family held-out Tail risks are 0.365 and 0.111. The reduction in ambiguity should therefore not be interpreted independently of the associated exclusion risk.

Watt shows the separation more weakly. RAED and model-index EIG are cleanly separated in three of the sixteen sensing levels, and in all three cases the RAED-selected experiment yields a smaller held-out candidate set, with reductions in J between 0.017 and 0.054. Averaged over all sixteen sensing levels, mean held-out J is 2.092 for the RAED selections and 2.111 for the model-index-EIG selections. In two of the three clean disagreements, the smaller candidate set is accompanied by a lower worst-family held-out Tail risk; in the third, at $( T , B ) = ( 9 0 , 0 )$ , the RAED-selected experiment has the smaller J but the larger Tail risk. Complete resolution and exclusion results for both benchmarks are reported in Tables 14 and 15.

![](images/b5e126ecb99bc7094d8c7ba02255f19a798060ac85811af006b7ca3152c10938.jpg)

![](images/55b87a1d5528ae4871da4cd6f226819363a3aaaa5d1fb2981ac752966bddb87a.jpg)  
Figure 3: Clean RAED–model-index-EIG disagreements under restricted sensing. Only cells in which the RAED-selected design lies outside the EIG Monte Carlo indistinguishability set are colored and annotated. Positive $\Delta J = J _ { \mathrm { E I G } } - J _ { \mathrm { R A E D } }$ indicates a smaller held-out candidate set for the RAED-selected experiment. The common color scale makes the stronger WCA separation visible; exact values for all 16 cells per benchmark are reported in Tables 14 and 15.

The Watt and WCA results show that information gain and structural resolution can rank the same set of experiments diferently when sensing is restricted. In many cells the two criteria still agree, or are numerically indistinguishable, and WCA also contains a clean case in which the EIG-selected experiment gives the smaller downstream candidate set. The main point is therefore not that RAED dominates EIG, but that the choice of design criterion can change both the selected experiment and the resulting balance between ambiguity and false exclusion.

The learned rules are calibrated on independent data and evaluated on held-out states. They are not accompanied by finite-sample population-validity certificates, so the Tail risks reported here should be interpreted as empirical held-out performance.

The additional comparators give a similar picture in terms of candidate-set size. In WCA, the experiments selected by full-latent EIG, Bayes forced classification, expected elimination, and minimum ordered-pair predictive discrimination all give larger mean held-out J than the RAED selections across the sixteen sensing levels, with ordered-pair discrimination the closest alternative. Several comparator rankings contain numerically indistinguishable optima, so the averages use the representative design defined for each criterion; the full summary is given in Table 16.

## 7.3 Nuisance-average validity can conceal a geological hard region

The GWAE-Fluvial benchmark provides a concrete example of a limitation of nuisance-average validity: a small family-average false-exclusion rate need not imply uniformly low risk across the geological nuisance population. In the first GWAE-Fluvial analysis, RAED was applied with $( \alpha , \delta ) = ( 0 . 0 5 , 1 )$ and selected the symmetric-bank experiment E01. Selected-only calibration on 1000 nuisance states per family gave risk upper bounds of 0.036541 for M1 and 0.049826 for M2, providing 95% joint confidence across the two familywise population claims. On the independent evaluation population, the corresponding familywise false-exclusion rates were 0.0141 for M1 and 0.0286 for M2. The rule also remained highly resolving, with J = 1.1267. Because K = 2, approximately 87.3% of observations were resolved to a singleton and 12.7% returned the ambiguous set {M1, M2}.

The family averages, however, do not show how the exclusion risk is distributed across geological realizations. For each nuisance state θ, we fixed the deterministic reservoir response and estimated its conditional false-exclusion probability over repeated draws from the assumed observation model. In both structural families, the median and 90th percentile of this statewise risk were zero, indicating that most geological realizations were almost never falsely excluded. The upper tail was very diferent: the 99th-percentile conditional risk was approximately 0.86 for M1 and 0.98 for M2, while the mean risk within the worst 1% of nuisance states was approximately 0.998 and 0.991, respectively.

![](images/6caa5637eeffcec3b233df776c45813eab347eb7328c27c25437fe5381d92258.jpg)  
Figure 4: Statewise risk concentration in the first GWAE-Fluvial campaign. Sorted conditional false-exclusion risks from 2000 untouched nuisance states per family show a long zero-risk region and an extreme upper tail. The 99th percentiles are 0.8626 for M1 and 0.9746 for M2; within the shaded worst 1%, mean risks are 0.9982 and 0.9907. These are descriptive evaluations of the independently calibrated nuisance-average rule.

The nuisance-average guarantee is therefore compatible with very low exclusion risk over most of the geological population and near-systematic failure on a small subset of hard realizations. Nothing in the $\delta = 1$ criterion prevents this concentration, even though the family-average exclusion target is satisfied. The first campaign was designed as a nuisance-average analysis rather than a Tail-RAED sensitivity study; the concentration of risk only became apparent in the subsequent statewise evaluation.

This observation motivated a second independent GWAE-Fluvial study using a new nuisance population, in which nuisance-average RAED was compared directly with Tail-RAED at $\delta =$ 0.05. Using an independent population separates the discovery of the hard-tail behavior from the subsequent test of whether protecting that tail changes the selected physical experiment and the resulting balance between resolution and false exclusion.

## 7.4 Tail protection changes the selected GWAE experiment

The second GWAE-Fluvial study compares nuisance-average RAED, $\delta = 1$ , with Tail-RAED at $\delta = 0 . 0 5$ on the same newly generated geological population. The two criteria select diferent physical experiments: nuisance-average RAED selects the symmetric-bank test E01, whereas Tail-RAED narrowly selects the staged-rate test E07. The change nevertheless shows that protecting the adverse nuisance tail can afect experiment selection itself, rather than only the thresholds calibrated after an experiment has been chosen.

Table 3: Deployed GWAE-Fluvial operating points on the common independent evaluation population. Worst-5% entries are the empirical upper-tail conditional-risk means computed separately for each method.
<table><tr><td>Method</td><td>Selected well test</td><td>J</td><td>M1 worst-5%</td><td>M2 worst-5%</td></tr><tr><td>Tail,  $\overline { { \delta = . 0 5 } }$ </td><td>E07 staged-rate pair</td><td>1.788292</td><td>0.013781</td><td>0.002977</td></tr><tr><td> $\mathrm { A v e r a g e } , \delta = 1$ </td><td>E01 symmetric banks</td><td>1.047271</td><td>0.433758</td><td>0.875844</td></tr></table>

For both selected experiments, calibration produced nontrivial candidate-set rules with familywise risk upper bounds below α = 0.05. On the common independent evaluation population, however, the two rules occupy very diferent resolution–risk operating points. Nuisance-average RAED remains highly resolving, with J = 1.047, but its empirical upper-tail conditional-risk means are 0.434 for M1 and 0.876 for M2. Tail-RAED increases the expected candidate-set size to J = 1.789, while its corresponding upper-tail risks fall to 0.014 and 0.003 respectively. The tail summaries in Table 3 are computed separately for each rule and therefore describe each rule’s own hardest 5% of nuisance states.

For K = 2, the increase in J has a direct interpretation: Tail-RAED returns the ambiguous set {M1, M2} much more often. To compare the two rules on the same dificult geological states, Figure 5 defines the hard region using the worst 5% of nuisance states under the nuisance-average rule. Within this common region, Tail-RAED largely replaces false singleton exclusions with explicit ambiguity.

The GWAE comparison therefore shows the cost of tail protection directly. Nuisance-average RAED achieves much smaller candidate sets, but a small subset of geological realizations remains exposed to high false-exclusion risk. Tail-RAED sacrifices part of that resolution in order to reduce exclusion in these dificult states, returning larger candidate sets when the observations do not support a confident structural decision.

![](images/53777453fa315ffd1c847054852aa36b11f76a18812d3d3e5b0f456c76c5b0b0.jpg)  
Figure 5: Behavior in the nuisance-average rule’s worst 5% geological region. Frequencies are computed over observation-noise draws within the 250 highest-risk nuisance states per family under the nuisance-average rule. Tail-RAED converts most behavior in this region to explicit ambiguity, whereas nuisance-average RAED produces false singleton exclusions on 43.4% of M1 draws and 87.6% of M2 draws.

## 7.5 Finite-sample Tail certification in methane oxidation

Methane oxidation provides a case in which the positive-tail validity requirement can be certified with the available nuisance sample. Across the declared design space, learned RAED and model-index EIG select the same reactor condition, so this benchmark is not used to demonstrate disagreement in experiment selection. Instead, it tests whether Tail-RAED can retain nontrivial structural resolution while satisfying the finite-sample validity requirement.

At the reference operating point α = 0.05 and δ = 0.05, calibration uses 10,000 independent nuisance states per mechanism. The resulting rule satisfies the familywise Tail-risk bounds with 95% joint confidence across PL, LH, and MVK. The three calibration upper bounds are 0.048, 0.038, and 0.048, all below the prescribed false-exclusion level.

The certified rule remains informative, with held-out mean candidate-set size J = 1.750. On the independent evaluation population, the worst-family empirical 5% Tail risk is 0.025. The finite-sample certificate therefore does not force the rule to retain all three mechanisms: positive-tail validity is certified while a nontrivial level of structural resolution is preserved.

![](images/35a27bd8af7e2ee4a1049d17ccb343039c568915832523dd32a33283e499750b.jpg)  
Figure 6: Deployed methane-oxidation frontier. Each filled point is a separately certified fixed-positive-δ operating point with 95% joint confidence across PL, LH, and MVK; the guarantees are not simultaneous across the full grid. The open δ = 0 point is empirical only. The line is a guide across the ordered declared levels, not a smooth theoretical frontier.

The same pattern holds across the other predeclared positive Tail levels. As δ decreases, stronger protection of adverse nuisance states generally requires larger candidate sets, whereas larger δ permits greater resolution. The complete frontier is reported in Appendix I.

## 8 Discussion

The empirical studies illustrate several diferent consequences of designing experiments around structural elimination rather than information gain alone. The Watt ontology study shows that the structural question itself matters: when property representation and relative-permeability family are promoted from nuisance variables to components of model identity, the number of candidate models increases substantially while the empirical candidate-set size grows only modestly. The restricted Watt and WCA studies show a diferent efect. Once observation time and spatial coverage are limited, RAED and information-based criteria can prefer diferent physical experiments even though they act on the same admissible design set. GWAE then shows that experiment choice can also depend on the form of the validity requirement itself, while methane provides a case in which positive-tail validity can be certified without collapsing to the trivial retain-all rule.

These results also make clear that resolution cannot be interpreted separately from false exclusion. In the Watt ontology study, the nontrivial learned rules remain highly resolving across all four structural targets, but their held-out exclusion rates lie slightly above the nominal level and the available calibration sample does not support a nontrivial population certificate. The restrictedsensing results make the same point more strongly. A RAED-selected experiment can yield a smaller downstream candidate set than an EIG-selected experiment while also producing a larger held-out Tail risk. In other cells the two quantities improve together. The relevant empirical outcome is therefore the pair consisting of remaining ambiguity and false-exclusion behavior, not candidate-set size alone.

GWAE provides the clearest example of why the nuisance distribution matters. Under nuisanceaverage validity, the first study satisfies its familywise population requirement and achieves high resolution, yet the statewise evaluation reveals a strongly concentrated failure mode: most geological realizations have essentially zero conditional false-exclusion risk, while a small subset is excluded almost systematically. This behavior is entirely compatible with $\delta = 1$ , because the constraint acts on the nuisance average rather than on the upper tail of the statewise risk distribution. The result gives a concrete geological interpretation to the distinction formalized by the Tail-RAED objective and to the rare-tail argument in Proposition 5.2.

The second GWAE study asks what changes when that adverse tail is protected explicitly. With $\delta = 0 . 0 5$ , Tail-RAED narrowly selects a diferent physical experiment and produces a very diferent operating point. The nuisance-average rule achieves much smaller candidate sets, whereas Tail-RAED returns the ambiguous set far more often and sharply reduces false exclusion in the dificult geological states. This is the intended role of set-valued decisions in RAED: when the observations do not support a reliable exclusion, the method is allowed to retain competing explanations instead of forcing a singleton decision. The cost is substantial, however, and smaller $\delta$ should not be interpreted as automatically preferable. The choice of $( \alpha , \delta )$ encodes how much false exclusion is acceptable and how much of the nuisance population should receive explicit protection.

The change in selected GWAE experiment should also be interpreted with some caution. E07 only narrowly outranks E01 under the Tail criterion, so the ranking itself is sensitive. The stronger result is not the identity of the winning experiment but what happens after calibration and independent evaluation: the nuisance-average and Tail rules occupy very diferent resolution–risk operating points, and Tail-RAED substantially changes the behavior of the rule on the geological states that were most problematic under nuisance-average validity. More generally, the Watt and WCA δ sweeps show that the preferred restricted-sensing design can itself depend on the requested degree of nuisance-tail protection.

The four empirical benchmarks carry diferent statistical weight. Watt and WCA provide independent held-out evaluations of learned rules, but their available nuisance populations are too small to support the stronger positive-tail population certificates used elsewhere in the paper. Their Tail-risk quantities are therefore empirical generalization diagnostics. GWAE and methane use much larger independent calibration populations, allowing fixed-operating-point familywise risk upper bounds to be certified. In GWAE, this supports both the nuisance-average and $\delta = 0 . 0 5$ Tail analyses; in methane, nontrivial certificates are obtained across the declared positive-δ operating points. In these cases the formal validity statements come from the calibration upper bounds, while held-out Tail means, statewise quantiles, and hard-region analyses describe how the already calibrated rules behave in practice.

Validity is always relative to the scientific problem that has been declared. The admissible nuisance sets $\Theta _ { k }$ determine which within-family states must be considered, $\mu _ { k }$ determines how positive-tail protection is distributed over those states, and $\nu _ { k }$ determines how resolution is averaged. The GWAE-Fluvial study, for example, uses an explicitly constructed single- and double-channel geological population with controlled geometric and petrophysical variability. Its conclusions therefore apply to that declared geological law. The same is true of the observation models, experiment libraries, and sensing restrictions used throughout the benchmarks: changing any of these changes the decision problem being solved.

Finally, the empirical studies use a restricted learned score family rather than the population oracle. Calibration controls false exclusion for the resulting learned candidate-set rule, while the oracle theory describes the ideal target that this restricted family is attempting to approximate. The present finite-sample population guarantees cover $0 < \delta \leq 1 $ ; the empirical $\delta = 0$ points should not be confused with population-uniform pointwise guarantees. The theory is also one-step: one experiment is selected, one response is observed, and one candidate set is returned. A natural extension is to allow the score model, experiment, and exclusion thresholds to be learned more jointly, or to move to sequential designs in which later experiments are chosen specifically to break the aliases that remain after earlier observations.

## 9 Conclusion

Resolution-Aware Experimental Design values an experiment by the structural explanations it can eliminate while controlling false exclusion under persistent nuisance uncertainty. Cross-nuisance aliasing shows that this objective can difer sharply from information gain, while the commongarbling results preserve the expected ordering when one experiment is genuinely less informative than another. The learned formulation makes the same objective usable with restricted score models under nuisance-average and positive-tail validity.

The empirical studies expose diferent aspects of this problem. In Watt, changing the definition of structural identity alters the attainable resolution while the preferred experiment remains remarkably stable across the declared tail levels. Under constrained sensing, Watt and especially WCA show that RAED and information-based criteria can select diferent physical experiments, with corresponding changes in both candidate-set size and held-out false-exclusion behavior. GWAE-Fluvial shows why nuisance-average validity can be insuficient when risk is concentrated on a small set of geologica realizations: protecting the upper nuisance tail changes the selected experiment and replaces many hard-region false exclusions with explicit ambiguity, at a substantial cost in average resolution. Methane oxidation provides the complementary case in which positive-tail validity can be certified nontrivially with a suficiently large nuisance-calibration population.

Taken together, the results show that experiment design under partial identifiability is not only a question of acquiring information. It also requires deciding which structural exclusions must remain reliable across nuisance uncertainty and how much unresolved ambiguity is acceptable in exchange. RAED makes that tradeof explicit through the candidate set and the operating point $( \alpha , \delta )$

## A Notation and Technical Preliminaries

For reference, the principal objects used throughout the paper are collected below.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $M \in \{ 1 , \ldots , K \}$ </td><td>structural family of scientific interest</td></tr><tr><td> $\theta \in \Theta _ { k }$ </td><td>admissible persistent nuisance state within family k</td></tr><tr><td> $e \in { \mathcal { E } }$ </td><td>complete precommitted experiment/intervention and observation protocol</td></tr><tr><td> $P _ { k , \theta } ^ { e }$ </td><td>predictive observation law under  $( k , \theta , e )$ </td></tr><tr><td> $\mu _ { k }$ </td><td>validity-mass measure used by Tail-RAED</td></tr><tr><td> $\nu _ { k }$ </td><td>nuisance efficiency measure used in the objective</td></tr><tr><td> $\pi _ { k }$ </td><td>structural efficiency weight, with  $\textstyle \sum _ { k } \pi _ { k } = 1$ </td></tr><tr><td> $\Gamma _ { e } ( Y )$ </td><td>randomized nonempty structural candidate set returned after observing</td></tr><tr><td> $g _ { k , e } ( y )$ </td><td>marginal probability that family k is included in  $\Gamma _ { e } ( y )$ </td></tr><tr><td> $L _ { k , e } ( \theta )$ </td><td>nuisance-specific false-exclusion probability  $\operatorname* { P r } \{ k \notin \Gamma _ { e } ( Y ) \mid k , \theta , e \}$ </td></tr><tr><td> $\alpha$ </td><td>scientific false-exclusion tolerance</td></tr><tr><td> $\delta$ </td><td>protected nuisance-tail mass; 0 is pointwise validity and 1 nuisance-average validity</td></tr><tr><td> $\zeta$ </td><td>finite-sample certificate failure probability</td></tr><tr><td> $J _ { e } ( g )$ </td><td>expected candidate-set size  $\mathbb { E } | \Gamma _ { e } ( Y ) |$  under the efficiency law</td></tr><tr><td> $V _ { e } ^ { \star } ( \alpha , \delta )$ </td><td>optimal population RAED value for experiment e at the declared validity level</td></tr><tr><td> $R _ { e } ^ { \star } ( \alpha , \delta )$ </td><td>optimal normalized RAED resolution  $( K - V _ { e } ^ { \star } ( \alpha , \delta ) ) / ( K - 1 )$ </td></tr></table>

For $0 < \delta \leq 1$ , Tail risk is written through the upper-tail/CVaR functional

$$
\rho _ { \delta , \mu _ { k } } ( L ) = \operatorname* { i n f } _ { \eta \in \mathbb { R } } \left\{ \eta + \frac { 1 } { \delta } \int ( L ( \theta ) - \eta ) _ { + } \mu _ { k } ( \mathrm { d } \theta ) \right\} ,
$$

with the equivalent OCE scaling used in Section 5. At $\delta \ : = \ : 0$ , validity is defined directly by $\operatorname* { s u p } _ { \theta \in \Theta _ { k } } L _ { k , e } ( \theta ) \leq c$ rather than by taking a singular limit of the Tail formula.

## B Exact Inclusion-Marginal Representation

The expected-size objective and all classwise validity constraints depend only on first-order inclusion marginals. This gives an exact compact representation of the current RAED rule problem.

Lemma B.1 (Measurable whole-set lifting). Let $g : \mathcal { V } \to \mathcal { P } _ { K }$ be measurable, where

$$
\mathcal { P } _ { K } = \left\{ u \in [ 0 , 1 ] ^ { K } : \sum _ { k = 1 } ^ { K } u _ { k } \geq 1 \right\} .
$$

Then there exists a measurable Markov kernel $\Pi ( { \cdot } \mid y )$ supported on the nonempty subsets of $\{ 1 , \ldots , K \}$ whose inclusion marginals equal $g ( y )$ for every y.

Proof. Enumerate the $2 ^ { K } - 1$ nonempty subsets as $S _ { 1 } , \ldots , S _ { m }$ and let $A \in \{ 0 , 1 \} ^ { K \times m }$ contain their incidence vectors as columns. By the polytope argument below, for every $u \in \mathcal P _ { K }$ the compact set

$$
\mathcal W ( u ) = \{ p \in \mathbb { R } _ { + } ^ { m } : { \mathbf { 1 } } ^ { \top } p = 1 , ~ A p = u \}
$$

is nonempty. Its graph is closed in the finite-dimensional product $\mathcal { P } _ { K } \times \mathbb { R } ^ { m }$ , so a standard measurableselection theorem supplies a Borel measurable selector $p ^ { \star } ( u ) \in \mathcal { W } ( u )$ . Set $\Pi ( S _ { j } \mid y ) = p _ { i } ^ { \star } ( g ( y ) )$ The resulting kernel is measurable, supported on nonempty subsets, and satisfies $\operatorname* { P r } _ { \Pi ( \cdot | y ) } \{ k \^ { } \in \Gamma \} =$ $g _ { k } ( y )$ □

Proposition B.2 (Exact inclusion-marginal representation). Fix an experiment e and a finite measurable observation partition $\{ B _ { e , r } \} _ { r = 1 } ^ { R _ { e } }$ . Let

$$
g _ { k , r } = \operatorname* { P r } \{ k \in \Gamma _ { e } ( Y ) \mid Y \in B _ { e , r } \} .
$$

A vector $g . , r$ is realizable by a randomized nonempty whole-set rule if and only if

$$
0 \leq g _ { k , r } \leq 1 , \qquad \sum _ { k = 1 } ^ { K } g _ { k , r } \geq 1 .\tag{36}
$$

Moreover, for the RAED validity regimes used in this paper–pointwise, strict Tail, and nuisanceaverage–the expected-cardinality objective and all familywise exclusion constraints depend on the rule only through these marginals. Consequently the whole-subset and marginal formulations have identical feasible loss fields and identical optimal values.

Proof. Let $x _ { r , S }$ be the conditional probability of returning nonempty subset S in response cell r. Any whole-set rule induces

$$
g _ { k , r } = \sum _ { S \ni k } x _ { r , S } ,
$$

so $0 \leq g _ { k , r } \leq 1$ . Since every realized subset is nonempty,

$$
\sum _ { k } g _ { k , r } = \sum _ { S \neq \emptyset } | S | x _ { r , S } \geq 1 .
$$

The same identity shows that expected set size in the cell is exactly $\textstyle \sum _ { k } g _ { k , r }$ . If

$$
p _ { k , \theta , r } ^ { e } : = { \cal P } _ { k , \theta } ^ { e } ( { \cal B } _ { e , r } ) ,
$$

then family-k retention under nuisance state θ is

$$
C _ { k , e } ( \theta ; g ) = \sum _ { r = 1 } ^ { R _ { e } } p _ { k , \theta , r } ^ { e } g _ { k , r } ,
$$

so every nuisance-specific exclusion loss is a function only of the corresponding marginals. Pointwise constraints, Tail/CVaR functionals of this loss field, and nuisance averages therefore all depend only on g.

Conversely, consider the polytope

$$
\mathcal { P } _ { K } = \left\{ g \in [ 0 , 1 ] ^ { K } : \sum _ { k } g _ { k } \ge 1 \right\} .
$$

Every extreme point of $\mathcal { P } _ { K }$ is a nonzero binary vector. Indeed, two fractional coordinates admit an opposite perturbation preserving feasibility; one fractional coordinate with slack in the sum constraint can be perturbed alone; and if the sum constraint is tight, a single fractional coordinate is impossible because the remaining binary coordinates have integer sum. Hence $\mathcal { P } _ { K }$ is the convex hull of the incidence vectors of the nonempty subsets of $\{ 1 , \ldots , K \}$ . Every feasible marginal vector is therefore a convex combination of nonempty subset incidences and can be realized by a randomized whole-set rule, independently in each response cell. This proves exact equivalence. □

Remark B.3 (Whole-set reconstruction). The marginal representation fixes expected cardinality and all classwise retention probabilities but generally does not identify a unique distribution over returned subsets. Coupling-sensitive summaries such as singleton frequency, the probability of a particular two-family set, or the variance of |Γ| therefore require an explicitly declared whole-set reconstruction. By Carathéodory’s theorem, at most $K + 1$ nonempty subsets are needed in a per-cell reconstruction.

## C Proofs for RAED Formulation, Aliasing, and Experiment Comparison

## C.1 Common and approximate garbling

The main text uses only the exact common-garbling monotonicity statement. The quantitative transport and deficiency results are recorded here.

Theorem C.1 (Common and approximate garbling of the RAED frontier). Fix the structural weights, admissible nuisance sets, eficiency measures, validity measures when applicable, and a scientific operating point $( \alpha , \delta )$ . Let $e _ { 1 } , e _ { 2 }$ be two experiments. Suppose a Markov kernel G, independent of both k and θ, satisfies

$$
\operatorname* { s u p } _ { 1 \leq k \leq K } \operatorname* { s u p } _ { \theta \in \Theta _ { k } } \mathrm { T V } \Big ( P _ { k , \theta } ^ { e _ { 2 } } , P _ { k , \theta } ^ { e _ { 1 } } G \Big ) \leq \varepsilon , \qquad 0 \leq \varepsilon \leq 1 .\tag{37}
$$

Define

$$
d _ { k } ( \theta ) = \mathrm { T V } \left( P _ { k , \theta } ^ { e _ { 2 } } , P _ { k , \theta } ^ { e _ { 1 } } G \right) , \qquad q _ { G } = \mathrm { T V } ( Q _ { e _ { 2 } } , Q _ { e _ { 1 } } G ) \leq \varepsilon ,
$$

and transport an $e _ { 2 }$ inclusion rule $g ^ { ( 2 ) }$ to $e _ { 1 }$ by

$$
( T _ { G } g ^ { ( 2 ) } ) _ { k } ( y _ { 1 } ) = \int g _ { k } ^ { ( 2 ) } ( y _ { 2 } ) G ( y _ { 1 } , \mathrm { d } y _ { 2 } ) .
$$

Then the transported rule is a randomized nonempty rule and, for every $k , \theta$

$$
\begin{array} { r } { \left| L _ { k , e _ { 1 } } ( \theta ; T _ { G } g ^ { ( 2 ) } ) - L _ { k , e _ { 2 } } ( \theta ; g ^ { ( 2 ) } ) \right| \le d _ { k } ( \theta ) \le \varepsilon , } \end{array}\tag{38}
$$

$$
\left| J _ { e _ { 1 } } ( T _ { G } g ^ { ( 2 ) } ) - J _ { e _ { 2 } } ( g ^ { ( 2 ) } ) \right| \le ( K - 1 ) q _ { G } \le ( K - 1 ) \varepsilon .\tag{39}
$$

Consequently, $i f \varepsilon \le 1 - \alpha _ { \mathrm { { \varepsilon } } }$ , then for every $\delta \in [ 0 , 1 ]$ ，

$$
V _ { e _ { 1 } } ^ { \star } ( \alpha + \varepsilon , \delta ) \leq V _ { e _ { 2 } } ^ { \star } ( \alpha , \delta ) + ( K - 1 ) q _ { G } .\tag{40}
$$

For an exact α comparison at arbitrary $\varepsilon ,$ set

$$
r _ { \varepsilon } = \operatorname * { m i n } \{ 1 , \alpha + \varepsilon \} , \quad \quad \lambda _ { \varepsilon } = \left( 1 - \frac { \alpha } { r _ { \varepsilon } } \right) _ { + } ,
$$

with $\lambda _ { \varepsilon } = 0$ when $r _ { \varepsilon } = 0$ . Mixing the transported rule with the full candidate set with probability $\lambda _ { \varepsilon }$ restores the original validity level and gives

$$
V _ { e _ { 1 } } ^ { \star } ( \alpha , \delta ) \leq ( 1 - \lambda _ { \varepsilon } ) \left[ V _ { e _ { 2 } } ^ { \star } ( \alpha , \delta ) + ( K - 1 ) q _ { G } \right] + \lambda _ { \varepsilon } K .\tag{41}
$$

In particular, when $\varepsilon = 0$

$$
V _ { e _ { 1 } } ^ { \star } ( \alpha , \delta ) \leq V _ { e _ { 2 } } ^ { \star } ( \alpha , \delta ) \qquad f o r \ e v e r y \ \delta \in [ 0 , 1 ] .
$$

The proof is a direct transport argument. Markov-kernel composition preserves the nonempty whole-set decision, bounded expectations move by at most total variation, and conditional candidateset size lies in [1, K], giving the factor $K - 1$ in (39). The final mixture uses the full set as a zero-exclusion safe rule. A complete proof is given in Section C.1.

The same comparison can be expressed through one-sided deficiency on the composite parameter space

$$
\Omega = \bigcup _ { k = 1 } ^ { K } \{ k \} \times \Theta _ { k } , \qquad \operatorname* { d e f } ( e _ { 1 } \sim e _ { 2 } ) = \operatorname* { i n f } _ { G } \operatorname* { s u p } _ { ( k , \theta ) \in \Omega } \mathrm { T V } \left( P _ { k , \theta } ^ { e _ { 2 } } , P _ { k , \theta } ^ { e _ { 1 } } G \right) .
$$

Corollary C.2 (Deficiency continuity of the resolution–validity frontier). Let $d _ { 1 2 } = \operatorname* { d e f } ( e _ { 1 } \rtimes e _ { 2 } )$ The bounds of Theorem C.1 hold for every $\varepsilon > d _ { 1 2 }$ , and at $\varepsilon = d _ { 1 2 }$ when the infimum is attained. If $\alpha > 0$ , the exact-repair comparison (41) also holds at a nonattained $d _ { 1 2 }$ by passage to the limit. For $\varepsilon > 0$ , if both directed deficiencies are at most ε, then for every $\delta \in [ 0 , 1 ]$ the symmetric bounds below hold. At the zero-error endpoint $\varepsilon = 0$ , they also hold when $\alpha > 0 ; i f \alpha = 0$ , they require attainment of both zero deficiencies (or another condition that supplies the same zero-error continuity). Under these conditions,

$$
| V _ { e _ { 1 } } ^ { \star } ( \alpha , \delta ) - V _ { e _ { 2 } } ^ { \star } ( \alpha , \delta ) | \le ( K - 1 ) [ ( 1 - \lambda _ { \varepsilon } ) \varepsilon + \lambda _ { \varepsilon } ] ,\tag{42}
$$

$$
\begin{array} { r } { | R _ { e _ { 1 } } ^ { \star } ( \alpha , \delta ) - R _ { e _ { 2 } } ^ { \star } ( \alpha , \delta ) | \le ( 1 - \lambda _ { \varepsilon } ) \varepsilon + \lambda _ { \varepsilon } . } \end{array}\tag{43}
$$

Proof of Theorem C.1. Let $\Pi _ { e _ { 2 } }$ be a randomized nonempty whole-set rule with inclusion marginals $g ^ { ( 2 ) }$ . Under $e _ { 1 }$ , observe $Y _ { 1 }$ , generate $\tilde { Y } _ { 2 } \sim G ( Y _ { 1 } , \cdot )$ , and then draw the returned set from $\Pi _ { e _ { 2 } } ( \cdot \vert \ : \widetilde { Y } _ { 2 } )$ This Markov-kernel composition remains supported on nonempty subsets and has inclusion marginals $T _ { G } g ^ { ( 2 ) }$

For each $( k , \theta ) , g _ { k } ^ { ( 2 ) }$ is bounded in [0, 1], so the bounded-expectation total-variation inequality gives

$$
\left| \mathbb { E } _ { P _ { k , \theta } ^ { e _ { 1 } } G } g _ { k } ^ { ( 2 ) } - \mathbb { E } _ { P _ { k , \theta } ^ { e _ { 2 } } } g _ { k } ^ { ( 2 ) } \right| \leq d _ { k } ( \theta ) .
$$

Since exclusion loss is one minus retention probability, this proves (38).

For the eficiency objective define

$$
f ( y _ { 2 } ) = \sum _ { k = 1 } ^ { K } g _ { k } ^ { ( 2 ) } ( y _ { 2 } ) .
$$

Nonemptiness and marginal feasibility imply $1 \leq f \leq K$ . The transported eficiency expectation is under $Q _ { e _ { 1 } G }$ , whereas the original expectation is under $Q _ { e _ { 2 } }$ . Applying the oscillation form of the total-variation inequality therefore yields

$$
\left| \mathbb { E } _ { Q _ { e _ { 1 } } G } f - \mathbb { E } _ { Q _ { e _ { 2 } } } f \right| \le ( K - 1 ) q _ { G } ,
$$

which is (39). Convexity of total variation under the common structural and nuisance mixing used to form $Q _ { e }$ gives $q _ { G } \leq \varepsilon .$

If the $e _ { 2 }$ rule is pointwise valid at level $\alpha ,$ (38) makes its transport pointwise valid at level $\alpha + \varepsilon .$ For $0 < \delta \leq 1$ , the risk-envelope representation implies

$$
\rho _ { \delta , \mu _ { k } } ( L _ { k , e _ { 1 } } ) \leq \rho _ { \delta , \mu _ { k } } ( L _ { k , e _ { 2 } } ) + \operatorname* { s u p } _ { \theta } | L _ { k , e _ { 1 } } ( \theta ) - L _ { k , e _ { 2 } } ( \theta ) | \leq \alpha + \varepsilon .
$$

Thus the same transfer holds for strict Tail validity and for the nuisance-average endpoint. Taking an infimizing sequence of $e _ { 2 }$ rules gives (40).

For exact repair, every transported exclusion risk is bounded by $r _ { \varepsilon } = \operatorname* { m i n } \{ 1 , \alpha + \varepsilon \}$ . Let $g ^ { \mathrm { a l l } } \equiv ( 1 , \ldots , 1 )$ and define

$$
\bar { g } = ( 1 - \lambda _ { \varepsilon } ) T _ { G } g ^ { ( 2 ) } + \lambda _ { \varepsilon } g ^ { \mathrm { a l l } } .
$$

All nuisance-specific exclusion losses are multiplied by $1 - \lambda _ { \varepsilon }$ . Pointwise supremum, mean risk, and the Tail functional are positively homogeneous, and

$$
\begin{array} { r } { ( 1 - \lambda _ { \varepsilon } ) r _ { \varepsilon } \leq \alpha . } \end{array}
$$

Hence $\bar { g }$ is exactly α-valid. Linearity of expected set size gives

$$
J _ { e _ { 1 } } ( \bar { g } ) = ( 1 - \lambda _ { \varepsilon } ) J _ { e _ { 1 } } ( T _ { G } g ^ { ( 2 ) } ) + \lambda _ { \varepsilon } K .
$$

Combining this identity with (39) and taking an infimizing sequence proves (41). The exact common-garbling claim is the case $\varepsilon = 0$ □

Proof of Corollary C.2. The first statement follows from the definition of the infimum: for every $\eta > d _ { 1 2 }$ there exists a common kernel with uniform error at most η. If the infimum is attained, use the attaining kernel. If $d _ { 1 2 } = 1$ , every kernel has total-variation error at most one.

Suppose $\alpha > 0$ and the infimum is not attained. Choose errors $\eta _ { n } \downarrow d _ { 1 2 }$ and apply the exact-repair bound of Theorem C.1. Because

$$
r _ { t } = \operatorname * { m i n } \{ 1 , \alpha + t \} , \quad \quad \lambda _ { t } = \left( 1 - \frac { \alpha } { r _ { t } } \right) _ { + } ,
$$

are continuous for $t \in [ 0 , 1 ]$ when $\alpha > 0$ , passage to the limit gives the same exact-repair comparison at $d _ { 1 2 }$

For the symmetric bound, first note that for $\alpha > 0$ the function

$$
B ( t ) : = ( 1 - \lambda _ { t } ) t + \lambda _ { t } = \left\{ \begin{array} { l l } { \displaystyle { \frac { ( 1 + \alpha ) t } { \alpha + t } } , } & { 0 \leq t \leq 1 - \alpha , } \\ { \alpha t + 1 - \alpha , } & { 1 - \alpha \leq t \leq 1 , } \end{array} \right.
$$

is continuous and nondecreasing. When $\alpha = 0 , B ( t ) = 1$ for every $t > 0$ while the declared exact endpoint has $B ( 0 ) = 0 ;$ ; hence no nonattained zero-error continuity is asserted. For $\varepsilon > 0$ approximation by errors below any value slightly above the directed deficiency and monotonicity of the positive-error bound sufice; at $\varepsilon = 0$ with $\alpha = 0$ , use the assumed attaining zero-error kernels. Apply the exact-repair comparison in the direction $e _ { 1 } \sim e _ { 2 }$ and use $V _ { e _ { 2 } } ^ { \star } \geq 1$ :

$$
\begin{array} { r l } & { \textstyle V _ { e _ { 1 } } ^ { \star } - V _ { e _ { 2 } } ^ { \star } \leq ( 1 - \lambda _ { \varepsilon } ) ( K - 1 ) \varepsilon + \lambda _ { \varepsilon } ( K - V _ { e _ { 2 } } ^ { \star } ) } \\ & { \textstyle \qquad \leq ( K - 1 ) \big [ ( 1 - \lambda _ { \varepsilon } ) \varepsilon + \lambda _ { \varepsilon } \big ] . } \end{array}
$$

The same argument with the experiments reversed bounds the opposite diference, proving (42). Finally,

$$
R _ { e } ^ { \star } = \frac { K - V _ { e } ^ { \star } } { K - 1 } ,
$$

so division by $K - 1$ gives (43).

## C.2 Exact cyclic aliasing separation

Proof of Theorem $4 . 1 .$ . Identify structural and nuisance alphabets with the cyclic group $\mathbb { Z } _ { K }$ . Under $e _ { I }$ , define

$$
Y = M \oplus Z .\tag{44}
$$

For every fixed $Z = z$ , the map $M \mapsto M \oplus z$ is a bijection, so $M = Y \ominus z$ is recovered exactly and $I ( M ; Y \mid Z = z , e _ { I } )$ = log K. A deterministic bijective channel can be garbled into the $e _ { R }$ channel defined below.

After nuisance averaging,

$$
P ( Y = j \mid M = k , e _ { I } ) = { \left\{ 1 - \varepsilon , \qquad j = k , \right. } 
$$

so the structural channel is $\mathsf C _ { K } ( \varepsilon )$ . Under $e _ { R } .$ , let the observation law be nuisance-independent:

$$
P ( Y = j ~ \vert ~ M = k , Z = z , e _ { R } ) = \left\{ 1 - \beta , \begin{array} { l l } { { j = k , } } & { { } } \\ { { \beta / ( K - 1 ) , } } & { { j \ne k . } } \end{array} \right.
$$

Its marginal channel is $\mathsf C _ { K } ( \beta )$ . Define

$$
\chi ( q ) = 1 - \frac { K } { K - 1 } q .
$$

Composition of two $K \mathrm { - a r y }$ symmetric channels multiplies their nontrivial eigenvalues $\chi ( \cdot )$ . Since $0 < \chi ( \beta ) < \chi ( \varepsilon )$ , choose

$$
\eta = \frac { K - 1 } { K } \left( 1 - \frac { \chi ( \beta ) } { \chi ( \varepsilon ) } \right) \in \left( 0 , \frac { K - 1 } { K } \right) .
$$

Then $\mathsf C _ { K } ( \beta ) = \mathsf C _ { K } ( \varepsilon ) \mathsf C _ { K } ( \eta )$ , proving nuisance-marginal Blackwell dominance. Strict model-index EIG preference follows from the monotonicity of (18), and the average Bayes error is $\varepsilon < \beta$

Under $e _ { I } , Y$ is a deterministic function of $( M , Z )$ . Since M is uniform and independent of $Z _ { i }$ $Y = M \oplus Z$ is uniform on $\mathbb { Z } _ { K }$ , hence

$$
I ( ( M , Z ) ; Y \mid e _ { I } ) = H ( Y ) = \log K .
$$

Under $e _ { R } , Y$ is conditionally independent of $Z$ given $M ,$ so

$$
I ( ( M , Z ) ; Y \mid e _ { R } ) = I ( M ; Y \mid e _ { R } ) = Z _ { K } ( \beta ) < \log K .
$$

For pointwise validity under $e _ { I } .$ , fix any $k , y \in \mathbb { Z } _ { K }$ . The nuisance state $z = y \ominus k$ is admissible and makes $Y = y$ with probability one when $M = k$ . Thus every pointwise-valid randomized rule must satisfy

$$
g _ { k } ( y ) \geq 1 - \alpha \qquad \forall k , y .
$$

Summing over k yields

$$
\sum _ { k = 1 } ^ { K } g _ { k } ( y ) \geq K ( 1 - \alpha ) ,
$$

so $J \geq K ( 1 - \alpha )$ under any eficiency distribution on $Y .$ . To attain the bound, let $t = K ( 1 - \alpha )$ $m = \lfloor t \rfloor$ , and $r = t - m$ . Independently of $Y$ , return a uniformly random m-subset with probability $1 - r$ and a uniformly random $( m + 1 )$ -subset with probability $^ { r } \cdot$ Each label is included with probability

$$
{ \frac { ( 1 - r ) m + r ( m + 1 ) } { K } } = 1 - \alpha ,
$$

and expected set size is t. Hence $V _ { e _ { I } } ^ { \star } ( \alpha , 0 ) = K ( 1 - \alpha )$ . Under $e _ { R } ,$ , the singleton rule $\Gamma ( Y ) = \{ Y \}$ has exclusion risk $\beta \leq \alpha$ at every nuisance state and achieves $J = 1$ , proving (19).

It remains to solve the strict Tail problem under $e _ { I }$ . Represent a marginal rule by the matrix $g = ( g _ { k } ( y ) ) _ { k , y \in \mathbb { Z } _ { K } }$ . For $u \in \mathbb { Z } _ { K }$ , define the simultaneous translation

$$
g _ { k } ^ { ( u ) } ( y ) = g _ { k \oplus u } ( y \oplus u ) .
$$

The feasible marginal set is convex, and the structural prior, nuisance law, objective, nonemptiness constraint, and each family Tail-risk constraint are invariant under this action. Averaging over u therefore preserves expected size and nonemptiness and, by convexity of $\rho _ { \delta }$ , cannot increase Tail risk. The averaged matrix has the form $g _ { k } ( k \oplus z ) = h _ { z }$ . Permuting the $K - 1$ nonzero ofsets and averaging again yields an optimal rule with a common exclusion probability on all alias states. Write

$$
\ell _ { 0 } = 1 - g _ { k } ( k ) , \qquad \ell _ { A } = 1 - g _ { k } ( k \oplus z ) , \quad z \neq 0 .
$$

Thus $\ell _ { 0 }$ is the false-exclusion level on the aligned nuisance state and $\ell _ { A }$ is the common false-exclusion level on the alias states. The expected candidate-set size is

$$
K - [ ( K - 1 ) \ell _ { A } + \ell _ { 0 } ] ,
$$

and mandatory nonemptiness imposes $( K - 1 ) \ell _ { A } + \ell _ { 0 } \leq K - 1$

An optimum may be taken with $\ell _ { A } \geq \ell _ { 0 }$ . Indeed, if $\ell _ { 0 } > \ell _ { A }$ , decrease $\ell _ { 0 }$ by t and increase each alias exclusion by $t / ( K - 1 )$ until the two values meet. This preserves $( K - 1 ) \ell _ { A } + \ell _ { 0 }$ . While $\ell _ { 0 } \geq \ell _ { A }$ , the aligned atom is the upper loss. If $\delta \leq 1 - \varepsilon$ , the upper-tail value decreases at unit rate. If $\delta > 1 - \varepsilon$ , the derivative of the upper-tail numerator is at most

$$
- ( 1 - \varepsilon ) + \frac { \varepsilon } { K - 1 } < 0 ,
$$

because $\varepsilon < ( K - 1 ) / K$ . Thus the exchange weakly decreases Tail risk without changing the objective.

For $\ell _ { A } \geq \ell _ { 0 }$ , the loss distribution has value $\ell _ { A }$ on total nuisance mass ε and value $\ell _ { 0 }$ on mass $1 - \varepsilon .$ , so

$$
\rho _ { \delta } ( L ) = \left\{ \begin{array} { l l } { \ell _ { A } , } & { \delta \le \varepsilon , } \\ { \frac { \varepsilon \ell _ { A } + ( \delta - \varepsilon ) \ell _ { 0 } } { \delta } , } & { \delta > \varepsilon . } \end{array} \right.
$$

When $\delta \leq \varepsilon$ , validity requires $\ell _ { A } \leq \alpha$ and hence $\ell _ { 0 } \leq \alpha$ , so total exclusion is at most $K \alpha$ and the pointwise value $K ( 1 - \alpha )$ remains optimal.

For $\delta > \varepsilon ,$ minimizing candidate-set size is equivalent to maximizing

$$
( K - 1 ) \ell _ { A } + \ell _ { 0 }
$$

subject to

$$
\varepsilon \ell _ { A } + ( \delta - \varepsilon ) \ell _ { 0 } \leq \alpha \delta , \qquad 0 \leq \ell _ { 0 } \leq \ell _ { A } \leq 1 ,
$$

and the nonemptiness cap. Ignoring the order constraint, the objective gained per unit Tail-risk budget is $( K - 1 ) / \varepsilon$ for $\ell _ { A }$ and $1 / ( \delta - \varepsilon )$ for $\ell _ { 0 }$ . These are equal at

$$
\delta _ { 0 } = \frac { K \varepsilon } { K - 1 } .
$$

For $\delta < \delta _ { 0 }$ , the order constraint forces the balanced boundary $\ell _ { A } = \ell _ { 0 } = \alpha$ , giving total exclusion $K \alpha$ . For $\delta \geq \delta _ { 0 }$ , exclusion is optimally allocated to the alias states, so $\ell _ { 0 } = 0$ and

$$
\ell _ { A } = \operatorname* { m i n } \biggl ( 1 , \frac { \alpha \delta } { \varepsilon } \biggr ) .
$$

The nonemptiness cap produces the floor at one, giving (20) and (21). Singleton resolution is reached exactly when $\alpha \delta / \varepsilon \geq 1$ , which proves (22). Under $e _ { R } .$ the singleton rule has constant exclusion risk $\beta \leq \alpha$ , so $V _ { e _ { R } } ^ { \star } ( \alpha , \delta ) = 1$ for all δ. □

## C.3 Scaling consequence

Corollary C.3 (Linear-in-K separation for every strict Tail level). Fix $\alpha \in ( 0 , 1 )$ and any $\bar { \delta } \in [ 0 , 1 )$ For all suficiently large K satisfying $\alpha < 1 - 1 / K$ , there exists a two-experiment problem in which nuisance-marginalized structural information, full-latent information, average Bayes classification, and nuisance-marginal Blackwell informativeness all strictly rank $e _ { I }$ above $e _ { R }$ , while

$$
\frac { V _ { e _ { I } } ^ { \star } ( \alpha , \bar { \delta } ) } { \operatorname* { m i n } _ { e \in \{ e _ { I } , e _ { R } \} } V _ { e } ^ { \star } ( \alpha , \bar { \delta } ) } \geq K c _ { \alpha , \bar { \delta } } , \qquad c _ { \alpha , \bar { \delta } } = \operatorname* { m i n } \biggl \{ 1 - \alpha , \frac { 1 - \bar { \delta } } { 1 + \bar { \delta } } \biggr \} > 0 .\tag{45}
$$

Proof of Corollary C.3. If $\delta = 0$ , apply the pointwise part of Theorem 4.1 with, for example, $\varepsilon = \alpha / 2$ and $\beta = 3 \alpha / 4$ . For all suficiently large $K _ { i }$ , the theorem’s parameter restriction holds and

$$
V _ { e _ { I } } ^ { \star } ( \alpha , 0 ) = K ( 1 - \alpha ) , \qquad V _ { e _ { R } } ^ { \star } ( \alpha , 0 ) = 1 ,
$$

which is (45) because $c _ { \alpha , 0 } = 1 - \alpha$

Now take $0 < \delta < 1$ and choose

$$
\varepsilon = \frac { \alpha ( 1 + \bar { \delta } ) } { 2 } , \qquad \beta = \frac { \alpha + \varepsilon } { 2 } .
$$

Then

$$
0 < \alpha \bar { \delta } < \varepsilon < \beta < \alpha .
$$

For suficiently large $K ,$ , all hypotheses of Theorem 4.1 hold, all information/classification comparisons in the corollary strictly rank $e _ { I }$ above $\textstyle e _ { R } ,$ , and $V _ { e _ { R } } ^ { \star } = 1$ . The exact Tail formula gives

$$
V _ { e _ { I } } ^ { \star } ( \alpha , \bar { \delta } ) = \operatorname* { m a x } \biggl \{ 1 , \operatorname* { m i n } \biggl [ K ( 1 - \alpha ) , K - ( K - 1 ) \frac { 2 \bar { \delta } } { 1 + \bar { \delta } } \biggr ] \biggr \} .
$$

Moreover,

$$
K - ( K - 1 ) \frac { 2 \bar { \delta } } { 1 + \bar { \delta } } \geq K \frac { 1 - \bar { \delta } } { 1 + \bar { \delta } } .
$$

Therefore

$$
V _ { e _ { I } } ^ { \star } ( \alpha , \bar { \delta } ) \geq K \operatorname* { m i n } \biggl \{ 1 - \alpha , \frac { 1 - \bar { \delta } } { 1 + \bar { \delta } } \biggr \} ,
$$

which proves the result.

## C.4 Difuse near-aliases

Theorem C.4 (General family-specific near-alias bound). Fix e and $0 < \delta \leq 1$ , and write

$$
Q _ { e } = \sum _ { r = 1 } ^ { R } \lambda _ { r } Q _ { r } , \qquad \lambda _ { r } \geq 0 , \qquad \sum _ { r } \lambda _ { r } = 1 .
$$

For every family k and component r, suppose there is a measurable $A _ { k , r } \subseteq \Theta _ { k }$ with

$$
m _ { k , r } = \mu _ { k } ( A _ { k , r } ) > 0 , \qquad \tau _ { k , r } = \frac { 1 } { m _ { k , r } } \int _ { A _ { k , r } } \mathrm { T V } ( P _ { k , \theta } ^ { e } , Q _ { r } ) \mu _ { k } ( \mathrm { d } \theta ) .
$$

Then every Tail-RAED feasible rule satisfies

$$
\mathbb { E } _ { Q _ { r } } | \Gamma _ { e } ( Y ) | \geq \operatorname* { m a x } \Biggl \{ 1 , \sum _ { k = 1 } ^ { K } \left[ 1 - \tau _ { k , r } - \alpha \kappa _ { \delta } ( m _ { k , r } ) \right] _ { + } \Biggr \}\tag{46}
$$

for every r, and hence

$$
V _ { e } ^ { \star } ( \alpha , \delta ) \geq \sum _ { r = 1 } ^ { R } \lambda _ { r } \operatorname* { m a x } \left\{ 1 , \sum _ { k = 1 } ^ { K } \left[ 1 - \tau _ { k , r } - \alpha \kappa _ { \delta } ( m _ { k , r } ) \right] _ { + } \right\} .\tag{47}
$$

Proof. Fix $k , r$ and abbreviate $A = A _ { k , r } , m = m _ { k , r }$ , and $L ( \theta ) = L _ { k , e } ( \theta )$ . Tail validity gives $\rho _ { \delta , \mu _ { k } } ( L ) \leq \alpha$ If $m \geq \delta$ , the weight $w = \mathbf { 1 } _ { A } / m$ is feasible in the Tail risk envelope, because $1 / m \leq 1 / \delta$ . Hence

$$
\frac { 1 } { m } \int _ { A } L ( \theta ) \mu _ { k } ( \mathrm { d } \theta ) \leq \alpha .
$$

If $m < \delta .$ choose

$$
w ( \theta ) = \frac { 1 } { \delta } \mathbf { 1 } _ { A } ( \theta ) + \frac { \delta - m } { \delta ( 1 - m ) } \mathbf { 1 } _ { A ^ { c } } ( \theta ) .
$$

This is a feasible envelope weight. Since $L \geq 0$

$$
\alpha \ge \rho _ { \delta , \mu _ { k } } ( L ) \ge \frac { 1 } { \delta } \int _ { A } L ( \theta ) \mu _ { k } ( \mathrm { d } \theta ) ,
$$

so

$$
\frac { 1 } { m } \int _ { A } L ( \theta ) \mu _ { k } ( \mathrm { d } \theta ) \leq \frac { \alpha \delta } { m } .
$$

Both cases are summarized by

$$
\frac { 1 } { m } \int _ { A } L ( \theta ) \mu _ { k } ( \mathrm { d } \theta ) \leq \alpha \kappa _ { \delta } ( m ) .\tag{48}
$$

Let $g _ { k } ( y ) = \operatorname* { P r } \{ k \in \Gamma _ { e } ( Y ) ~ | ~ Y = y \} \in [ 0 , 1 ]$ . Since $1 - L ( \theta ) = \mathbb { E } _ { P _ { k , \theta } ^ { e } } g _ { k } ( Y )$ , the boundedexpectation total-variation inequality gives

$$
\mathbb { E } _ { Q _ { r } } g _ { k } ( Y ) \ge \mathbb { E } _ { P _ { k , \theta } ^ { e } } g _ { k } ( Y ) - \mathrm { T V } ( P _ { k , \theta } ^ { e } , Q _ { r } ) .
$$

Average over A and use (48) to obtain

$$
\mathbb { E } _ { Q _ { r } } g _ { k } ( Y ) \geq 1 - \alpha \kappa _ { \delta } ( m _ { k , r } ) - \tau _ { k , r } .
$$

Since $g _ { k } \geq 0$ , the right-hand side may be replaced by its positive part. Summing over k gives

$$
\mathbb { E } _ { Q _ { r } } | \Gamma _ { e } ( Y ) | = \sum _ { k } \mathbb { E } _ { Q _ { r } } g _ { k } ( Y ) \geq \sum _ { k } \left[ 1 - \tau _ { k , r } - \alpha \kappa _ { \delta } ( m _ { k , r } ) \right] _ { + } .
$$

Mandatory nonemptiness supplies the independent lower bound one, proving (46). Averaging over the mixture weights $\lambda _ { r }$ proves (47). The uniform specialization reported in (25) follows because $\kappa _ { \delta } ( m )$ is nonincreasing in m and each $\tau _ { k , r } \leq \tau$ □

## D Learned RAED and Calibration Proofs

## D.1 Population score geometry supporting the learned family

Fix e and let $P _ { k , \mu } ^ { e }$ and $Q _ { e }$ be as defined in (27) and (3). Assume $P _ { k , \mu } ^ { e } \ll Q _ { \epsilon }$ and write $r _ { k , e } =$ $\mathrm { d } P _ { k , \mu } ^ { e } / \mathrm { d } Q _ { e }$

Proposition D.1 (Classwise nuisance-average density-ratio rule). For the classwise relaxation

$$
\operatorname* { m i n } _ { 0 \leq g _ { k } \leq 1 } ~ \mathbb { E } _ { Q _ { e } } g _ { k } ( Y ) \quad s u b j e c t ~ t o \quad \mathbb { E } _ { P _ { k , \mu } ^ { e } } g _ { k } ( Y ) \geq 1 - \alpha ,\tag{49}
$$

there is a threshold $t _ { k } \in [ 0 , \infty ]$ and boundary randomization $\gamma _ { k } ( y ) \in [ 0 , 1 ]$ supported on $\{ r _ { k , e } = t _ { k } \}$ such that an optimum is

$$
g _ { k } ^ { \star } ( y ) = { \bf 1 } \{ r _ { k , e } ( y ) > t _ { k } \} + \gamma _ { k } ( y ) { \bf 1 } \{ r _ { k , e } ( y ) = t _ { k } \} .\tag{50}
$$

Proof. The objective is the $Q _ { e }$ -mass of the retained region and the constraint requires at least $1 - \alpha$ mass under $P _ { k , \mu } ^ { e }$ . Introducing a nonnegative multiplier λ gives the pointwise Lagrangian integrand

$$
[ 1 - \lambda r _ { k , e } ( y ) ] g _ { k } ( y )
$$

under $Q _ { e }$ . It is minimized by $g _ { k } = 1$ where $\lambda r _ { k , e } > 1$ , by $g _ { k } = 0$ where $\lambda r _ { k , e } < 1$ , and by arbitrary randomization on equality. The multiplier and boundary randomization are chosen so that the coverage constraint is met at equality unless it is slack at the unconstrained optimum. This is the randomized Neyman–Pearson construction. □

For strict Tail validity, let

$$
\mathcal { W } _ { k , \delta } : = \left\{ w \in L ^ { \infty } ( \mu _ { k } ) : 0 \leq w \leq 1 / \delta , \ \int w \mathrm { d } \mu _ { k } = 1 \right\} ,\tag{51}
$$

and define the reweighted predictive law

$$
P _ { k , w } ^ { e } ( A ) : = \int w ( \theta ) P _ { k , \theta } ^ { e } ( A ) \mu _ { k } ( \mathrm { d } \theta ) .\tag{52}
$$

Proposition D.2 (Tail validity as robust family retention). For every measurable marginal rule and every $0 < \delta \leq 1$ ，

$$
\begin{array} { r } { \rho _ { \delta , \mu _ { k } } \bigl ( L _ { k , e } ( \cdot ; g ) \bigr ) \leq \alpha \quad \Longleftrightarrow \quad \mathbb { E } _ { P _ { k , w } ^ { e } } g _ { k } ( Y ) \geq 1 - \alpha \quad \forall w \in \mathscr { W } _ { k , \delta } . } \end{array}\tag{53}
$$

Proof. For any $w \in \mathcal { W } _ { k , \delta }$

$$
\int w ( \theta ) L _ { k , e } ( \theta ; g ) \mu _ { k } ( \mathrm { d } \theta ) = 1 - \int g _ { k } ( y ) P _ { k , w } ^ { e } ( \mathrm { d } y ) .
$$

Taking the supremum over w on the left is equivalent to taking the infimum of reweighted retention on the right. The risk-envelope definition (9) gives the result. □

These two propositions justify the score interpretation used in Section 5.1. Under the additional compactness, domination, $L ^ { 1 } ( Q _ { e } )$ -continuity, $0 < \alpha < 1$ , and strong-duality conditions of Theorem E.1, the robust problem attains least-favourable $w _ { k } ^ { \star }$ and nonnegative dual multipliers $\lambda _ { k } ^ { \star }$ yielding the pointwise coupled oracle

$$
g ^ { \star } ( y ) \in \underset { u \in [ 0 , 1 ] ^ { K } , \sum _ { k } u _ { k } \geq 1 } { \arg \operatorname* { m i n } } \sum _ { k = 1 } ^ { K } \big [ 1 - \lambda _ { k } ^ { \star } r _ { k , w _ { k } ^ { \star } } ( y ) \big ] u _ { k } ,\tag{54}
$$

where $r _ { k , w } = \mathrm { d } P _ { k , w } ^ { e } / \mathrm { d } Q _ { e }$ . The full functional-analytic attainment theorem is retained with its complete conditions in Appendix E; it is supporting oracle geometry rather than a main-paper result.

## D.2 Proof of nested nuisance-average calibration

Proof of the mean clause of Theorem 5.1. Condition on $\mathcal { F } _ { 0 }$ . The trained scores, architecture, hyperparameters, nested rule family, threshold domain, and population risk curve are then fixed, while the final calibration sample retains its declared law.

If every threshold is population-valid, the conclusion is immediate. If the calibrated passing set is empty, the procedure deploys the safe endpoint and is again valid. It remains to consider the event on which an invalid threshold is selected.

First suppose that $\Lambda _ { j }$ is a continuous interval. Define

$$
\lambda _ { j } ^ { \star } : = \operatorname* { i n f } \{ \lambda \in \Lambda _ { j } : R _ { j } ( \lambda ) \leq \alpha \} .
$$

The safe endpoint makes this set nonempty. Whenever invalid thresholds exist, continuity and monotonicity give

$$
R _ { j } ( \lambda _ { j } ^ { \star } ) = \alpha .
$$

If $\widehat { \lambda } _ { j }$ were invalid, monotonicity would imply $\widehat { \lambda } _ { j } < \lambda _ { j } ^ { \star }$ . Let $\mathcal { P } _ { j }$ denote the passing set inside the infimum in (30). Since $\widehat { \lambda } _ { j } = \operatorname* { i n f } \mathcal { P } _ { j } < \lambda _ { j } ^ { \star }$ , there exists $\lambda \in { \mathcal { P } } _ { j }$ with $\lambda < \lambda _ { j } ^ { \star }$ . The all-larger requirement defining $\mathcal { P } _ { j }$ therefore forces

$$
U _ { j } ( \lambda _ { j } ^ { \star } ; \gamma _ { j } ) < \alpha = R _ { j } ( \lambda _ { j } ^ { \star } ) .
$$

Thus invalid selection implies failure of the pointwise UCB at the single $\mathcal { F } _ { 0 } .$ -measurable population crossing point $\lambda _ { j } ^ { \star }$

Now suppose instead that

$$
\Lambda _ { j } = \{ \lambda _ { j , 1 } < \dots < \lambda _ { j , m _ { j } } \}
$$

is a finite ordered threshold grid. If any grid point is invalid, define the last invalid threshold

$$
\lambda _ { j } ^ { - } : = \operatorname* { m a x } \{ \lambda \in \Lambda _ { j } : R _ { j } ( \lambda ) > \alpha \} .
$$

This quantity is fixed conditional on $\mathcal { F } _ { 0 }$ . Because the passing set is finite, a nonempty passing set has a smallest element and $\widehat { \lambda } _ { j } \in \mathcal { P } _ { j }$ . If $\hat { \lambda _ { j } }$ were invalid, then monotonicity gives

$$
\widehat { \lambda } _ { j } \leq \lambda _ { j } ^ { - } .
$$

Since membership of $\widehat { \lambda } _ { j }$ in $\mathcal { P } _ { j }$ requires the UCB to be below $\alpha$ at every grid point at least as conservative as $\widehat { \lambda } _ { j }$ , it follows in particular that

$$
U _ { j } ( \lambda _ { j } ^ { - } ; \gamma _ { j } ) < \alpha < R _ { j } ( \lambda _ { j } ^ { - } ) .
$$

Hence invalid selection again implies failure of the pointwise UCB at one fixed, population-determined threshold.

In either case,

$$
\{ R _ { j } ( \widehat { \lambda } _ { j } ) > \alpha \}
$$

is contained in the failure event of a single pointwise UCB whose conditional failure probability is at most $\gamma _ { j }$ . Therefore

$$
\mathrm { P r } \Big \{ R _ { j } ( \widehat { \lambda } _ { j } ) \leq \alpha \Big | \mathcal { F } _ { 0 } \Big \} \geq 1 - \gamma _ { j } ,
$$

which proves (31). For a finite or countable claim set, a union bound over $j$ with $\Sigma _ { j } \gamma _ { j } \leq \zeta$ gives the simultaneous statement. No union bound over the thresholds along an individual nested path is required. □

## D.3 Positive-tail specialization of nested calibration

Proof of the positive-tail specialization. Define

$$
H _ { j , \lambda } ( \theta ) : = \delta \eta _ { j } ( \lambda ) + \left( L _ { j , \lambda } ( \theta ) - \eta _ { j } ( \lambda ) \right) _ { + } .
$$

Condition again on $\mathcal { F } _ { 0 }$ . Because the entire map $\lambda \mapsto \eta _ { j } ( \lambda )$ is fixed and measurable before the calibration sample is used, $H _ { j , \lambda }$ is a fixed bounded loss for every $\mathcal { F } _ { 0 } .$ -measurable threshold. The Rockafellar–Uryasev/OCE representation gives, for every fixed λ,

$$
\begin{array} { r l } {  { \rho _ { \delta , \mu _ { k } } ( L _ { j , \lambda } ) = \operatorname* { i n f } _ { \eta \in \mathbb { R } } \bigg \{ \eta + \frac { 1 } { \delta } \mathbb { E } ( L _ { j , \lambda } - \eta ) _ { + } \bigg \} } } \\ & { \leq \eta _ { j } ( \lambda ) + \frac { 1 } { \delta } \mathbb { E } ( L _ { j , \lambda } - \eta _ { j } ( \lambda ) ) _ { + } } \\ & { = \frac { \mathbb { E } H _ { j , \lambda } } { \delta } . } \end{array}
$$

Let $U _ { j } ^ { H } ( \lambda ; \gamma _ { j } )$ denote a pointwise upper confidence bound for $\mathbb { E } H _ { j , \lambda }$ . Because $0 \leq L \leq 1 , 0 \leq \eta \leq 1$ and $0 < \delta < 1$ , the transformed loss $H _ { j , \lambda }$ lies in [0, 1]. Hence $U _ { j } ^ { H } / \delta$ is a pointwise upper confidence bound for the true Tail risk at every $\mathcal { F } _ { 0 }$ -measurable fixed threshold. Candidate-set enlargement decreases $L _ { j , \lambda }$ pointwise and therefore decreases its Tail risk. By the assumed continuity of that risk path, the proof of the mean clause applies verbatim with $R _ { j } ( \lambda )$ replaced by $\rho _ { \delta , \mu _ { k } } ( L _ { j , \lambda } )$ and $U _ { j }$ by $U _ { j } ^ { H } / \delta$ . This proves (33); the simultaneous statement follows from the same countable union bound. □

## D.4 Nested Monte Carlo for nuisance-state loss

Corollary D.3 (Jensen-conservative nested Monte Carlo). Fix $0 < \delta < 1$ , λ, and the independently fixed $\eta _ { j } ( \boldsymbol { \lambda } )$ . Suppose that conditional on nuisance state θ,

$$
\widehat { L } _ { j , \lambda } ( \theta ) = \frac { 1 } { R } \sum _ { r = 1 } ^ { R } I _ { r , j , \lambda } ( \theta ) , \qquad \mathbb { E } \Big [ \widehat { L } _ { j , \lambda } ( \theta ) \mid \theta \Big ] = L _ { j , \lambda } ( \theta ) ,\tag{55}
$$

where the $I _ { r } \in [ 0 , 1 ]$ are generated from the declared observation-noise law. Define

$$
\widehat { H } _ { j , \lambda } ( \theta ) : = \delta \eta _ { j } ( \lambda ) + \left( \widehat { L } _ { j , \lambda } ( \theta ) - \eta _ { j } ( \lambda ) \right) _ { + } .\tag{56}
$$

Then

$$
\mathbb { E } \left[ \widehat { H } _ { j , \lambda } ( \theta ) \mid \theta \right] \geq H _ { j , \lambda } ( \theta ) .\tag{57}
$$

Consequently a valid upper confidence bound on the outer mean of independent nuisance-state observations $\widehat { H } _ { i }$ is conservative for EH, and therefore for the Tail risk through (32). The nuisance states, rather than the R inner noise replicates, remain the outer calibration sample size.

Proof of Corollary D.3. For fixed $\eta \in [ 0 , 1 ]$ , the function

$$
h _ { \eta } ( z ) : = \delta \eta + ( z - \eta ) _ { + }
$$

is convex in z. Conditional on θ, Jensen’s inequality and unbiasedness of (55) give

$$
\mathbb { E } [ h _ { \eta } ( \widehat { L } ) \mid \theta ] \ge h _ { \eta } ( \mathbb { E } [ \widehat { L } \mid \theta ] ) = h _ { \eta } ( L ( \theta ) ) .
$$

This is (57). Integrating over $\theta \sim \mu _ { k }$ yields

$$
\mathbb { E } _ { \theta , \mathrm { n o i s e } } \widehat { H } _ { j , \lambda } ( \theta ) \geq \mathbb { E } _ { \theta } H _ { j , \lambda } ( \theta ) .
$$

Therefore any valid upper confidence bound U for the mean of the outer bounded observations $\widehat { H } _ { i }$ also satisfies, on the same event,

$$
\mathbb { E } H _ { j , \lambda } \leq \mathbb { E } \widehat { H } _ { j , \lambda } \leq U .
$$

Combining this with (32) gives

$$
\rho _ { \delta , \mu _ { k } } ( L _ { j , \lambda } ) \leq U / \delta .
$$

Independence is required across the outer nuisance-state observations used by the chosen boundedmean UCB. Multiple inner draws contribute only to one $\widehat { H } _ { i }$ and therefore do not increase the outer nuisance sample size. □

## D.5 Selection-aware independent certification

Theorem D.4 (Certification after an independent shortlist). Let $\mathcal { F } _ { 0 }$ contain all randomness used to train rules, choose unprotected hyperparameters, and construct a nonempty random shortlist ${ \mathcal { S } } \subseteq { \mathcal { E } }$ Let $\mathcal { F } _ { \mathrm { c a l } }$ be generated by a final calibration sample independent of $\mathcal { F } _ { 0 }$ , and let $\mathcal { F } _ { \mathrm { r a n k } }$ contain any additional randomness used to choose among certified rules. For each realized S, let $\mathcal { T } ( S )$ be a finite or countable, measurably enumerated set of claims that must hold simultaneously. Suppose each claim $j \in \mathcal { T } ( S )$ has a conditional failure event $B _ { j }$ satisfying

$$
\operatorname* { P r } ( B _ { j } \mid { \mathcal { F } } _ { 0 } ) \leq \gamma _ { j } ( S ) \quad a . s . , \qquad \sum _ { j \in { \mathcal { I } } ( S ) } \gamma _ { j } ( S ) \leq \zeta .\tag{58}
$$

Then, with probability at least $1 - \zeta _ { i }$ , every claim in the realized shortlist is valid simultaneously. Hence any measurable deployed experiment ${ \widehat { e } } \in S$ whose final rule is chosen from those certified rules is population-valid, even if the final choice depends on the calibration and ranking data.

Proof of Theorem $D . 4 .$ . Condition on $\mathcal { F } _ { 0 }$ . The realized shortlist $s ,$ the fixed rule families, the finite or countable measurable enumeration of $\mathcal { T } ( S )$ , and the failure allocations are now fixed. Let

$$
\ A _ { \mathcal { S } } : = \bigcap _ { j \in \mathcal { T } ( \mathcal { S } ) } B _ { j } ^ { c } .
$$

The conditional union bound gives

$$
\operatorname* { P r } ( \mathcal { A } _ { S } ^ { c } \mid \mathcal { F } _ { 0 } ) \leq \sum _ { j \in \mathcal { I } ( S ) } \operatorname* { P r } ( B _ { j } \mid \mathcal { F } _ { 0 } ) \leq \sum _ { j \in \mathcal { I } ( S ) } \gamma _ { j } ( S ) \leq \zeta .
$$

Thus $\operatorname* { P r } ( \mathcal { A } _ { \mathcal { S } } \mid \mathcal { F } _ { 0 } ) \ge 1 - \zeta$ almost surely. On $\mathcal { A } _ { \mathcal { S } }$ every rule from which the later selector can choose is population-valid, so any measurable selection among those rules remains valid even if the choice uses $\mathcal { F } _ { \mathrm { c a l } }$ or $\mathcal { F } _ { \mathrm { r a n k } }$ . Finally the tower property gives

$$
\operatorname* { P r } ( \mathcal { A } _ { \mathcal { S } } ) = \mathbb { E } [ \operatorname* { P r } ( \mathcal { A } _ { \mathcal { S } } \mid \mathcal { F } _ { 0 } ) ] \ge 1 - \zeta .
$$

For selected-only calibration, $\boldsymbol { \mathcal { S } }$ is a singleton already fixed by $\mathcal { F } _ { 0 }$ , so the claim set contains only the deployed experiment’s still-open family/tail claims. This is a validity statement conditional on the independent selection stage, not a guarantee that the selected experiment is population-optimal. Tail-risk monotonicity gives any larger-δ claims that are logically implied by a certified smaller-δ statement without another union-bound term. □

## D.6 Proof of the rare-tail obstruction

Proof of Proposition 5.2. Compare the null distribution $P _ { 0 } ( Z = 0 ) = 1$ with

$$
P _ { 1 } ( Z = 1 ) = p , \qquad P _ { 1 } ( Z = 0 ) = 1 - p , \qquad p = \delta ( \alpha + \varepsilon ) .
$$

Because $p \leq \delta$ , the worst-δ Tail risk of $P _ { 1 }$ is

$$
\rho _ { \delta } ( P _ { 1 } ) = \frac { p } { \delta } = \alpha + \varepsilon > \alpha .
$$

Under $P _ { 1 }$ , however, an all-zero calibration sample occurs with probability $( 1 - p ) ^ { n }$ . Any deterministic procedure that certifies the all-zero sample therefore falsely certifies this just-invalid alternative with probability at least $( 1 - p ) ^ { n }$ . Requiring that probability to be no larger than $\beta$ proves (34). Taking logarithms gives

$$
n \geq { \frac { \log ( 1 / \beta ) } { - \log ( 1 - \delta ( \alpha + \varepsilon ) ) } } ,
$$

and $- \log ( 1 - x ) \sim x$ as x ↓ 0 yields (35) as $\varepsilon \downarrow 0$ in the small-αδ regime.

## E Additional Theoretical Results

## E.1 Attained least-favourable Tail-RAED score

The main text uses only the score interpretation of the robust Tail constraint. For completeness, the population attainment result that supports that interpretation is recorded here.

Theorem E.1 (Attained least-favourable Tail-RAED score under dominated compact predictive families). Fix $0 < \delta \leq 1$ and experiment e. Assume for each family k that: $( i ) \Theta _ { k }$ is a compact metric space and $\mu _ { k }$ is a Borel probability measure; $( i i ) \ : \mathcal { V } _ { e }$ is standard Borel; $( i i i ) \ P _ { k , \theta } ^ { e } \ll Q _ { e }$ for $\mu _ { k } - a l m o s t$ every $\theta ; \mathit { \Omega } ( i v )$ , writing

$$
r _ { k , \theta } = \frac { \mathrm { d } P _ { k , \theta } ^ { e } } { \mathrm { d } Q _ { e } } ,
$$

the map $\theta \mapsto r _ { k , \theta }$ is Borel measurable and continuous into $L ^ { 1 } ( Q _ { e } )$ ; and $( v ) 0 < \alpha < 1$ . Let

$$
\mathcal { W } _ { k , \delta } = \left\{ w \in L ^ { \infty } ( \mu _ { k } ) : 0 \leq w \leq 1 / \delta , \ \int w \mathrm { d } \mu _ { k } = 1 \right\} ,\tag{59}
$$

and define

$$
r _ { k , w } ( y ) = \int w ( \theta ) r _ { k , \theta } ( y ) \mu _ { k } ( \mathrm { d } \theta ) .\tag{60}
$$

Then the full nonempty population Tail-RAED problem has an attained primal optimum, zero Lagrange-duality gap, attained nonnegative family multipliers $\lambda _ { k } ^ { \star }$ , and attained least-favourable weights $w _ { k } ^ { \star } \in \mathcal { W } _ { k , \delta }$ . An oracle optimum may be chosen so that, for $Q _ { e }$ -almost every y,

$$
\boxed { g ^ { \star } ( y ) \in \underset { u \in [ 0 , 1 ] ^ { K } , \sum _ { k } u _ { k } \geq 1 } { \arg \operatorname* { m i n } } \sum _ { k = 1 } ^ { K } \left[ 1 - \lambda _ { k } ^ { \star } r _ { k , w _ { k } ^ { \star } } ( y ) \right] u _ { k } . }\tag{61}
$$

Thus every class with $\lambda _ { k } ^ { \star } r _ { k , w _ { k } ^ { \star } } ( y ) > 1$ is retained; if no coeficient is negative, total inclusion mass one is assigned among maximizers of $\lambda _ { k } ^ { \star } r _ { k , w _ { k } ^ { \star } } ( y )$ ; equality sets permit randomization. Whenever ${ \lambda } _ { k } ^ { \star } > 0$ , w<sup>⋆</sup> may be chosen as a nuisance reweighting that is least favourable for the optimal family-k retention constraint.

Proof. Equip

$$
\mathcal { G } = \left\{ g \in L ^ { \infty } ( Q _ { e } ) ^ { K } : 0 \leq g _ { k } \leq 1 , \ \sum _ { k } g _ { k } \geq 1 \right\}
$$

with the product weak-\* topology. It is convex, weak-\* closed and bounded, hence compact by Banach–Alaoglu. The objective

$$
J _ { e } ( g ) = \sum _ { k } \int g _ { k } ~ \mathrm { d } Q _ { e }
$$

is weak-\* continuous. Each $\mathcal { W } _ { k , \delta }$ is convex and weak-\* compact in $L ^ { \infty } ( \mu _ { k } )$ because positivity, the upper bound, and the normalization constraint are weak-\* closed.

For fixed $g _ { k }$ , write $h _ { k , g } ( \theta ) = \mathbb { E } _ { P _ { k , \theta } ^ { e } } g _ { k }$ . The stated $L ^ { 1 } ( Q _ { e } )$ continuity of $r _ { k , \theta }$ implies continuity of $\theta \mapsto h _ { k , g } ( \theta )$ , while boundedness gives measurability. Consequently

$$
( w , g _ { k } ) \longmapsto \int w ( \theta ) h _ { k , g } ( \theta ) \mu _ { k } ( \mathrm { d } \theta ) = \mathbb { E } _ { Q _ { e } } [ r _ { k , w } ( Y ) g _ { k } ( Y ) ]
$$

is separately continuous in the relevant weak-\* topologies. The robust retention functional

$$
C _ { k } ( g _ { k } ) : = \operatorname* { i n f } _ { w \in \mathcal { W } _ { k , \delta } } \mathbb { E } _ { Q _ { e } } [ r _ { k , w } g _ { k } ]
$$

is therefore concave and upper semicontinuous. The sets $\{ g : C _ { k } ( g _ { k } ) \geq 1 - \alpha \}$ are closed, so their intersection with compact $\mathcal { G }$ is a nonempty compact feasible set; the full-set rule is feasible. Hence the primal optimum is attained.

Each constraint function $( 1 - \alpha ) - C _ { k } ( g _ { k } )$ is convex and 1-Lipschitz in the sup norm. The full-set rule is strictly feasible because $C _ { k } ( 1 ) = 1 > 1 - \alpha$ . Standard Slater duality for the finite family of continuous convex inequalities therefore gives zero Lagrange-duality gap and nonnegative optimal multipliers. Multiplier attainment may be restricted to a compact set: the dual value at $\lambda = 0$ is at least one since every nonempty rule has $J _ { e } ( g ) \ge 1$ , whereas evaluating the Lagrangian at the full-set rule gives $\begin{array} { r } { K - \alpha \sum _ { k } \lambda _ { k } } \end{array}$ . Thus no maximizing multiplier has $\textstyle \sum _ { k } \lambda _ { k } > ( K - 1 ) / \alpha$ , and upper semicontinuity gives attainment on the resulting compact multiplier set.

For fixed λ, the robust constraints yield the saddle expression

$$
\sum _ { k } \lambda _ { k } ( 1 - \alpha ) + \operatorname* { i n f } _ { g \in { \mathcal { G } } } \operatorname* { s u p } _ { w _ { 1 } \in { \mathcal { W } } _ { 1 , \delta } , \ldots , w _ { K } \in { \mathcal { W } } _ { K , \delta } } \left\{ J _ { e } ( g ) - \sum _ { k } \lambda _ { k } \mathbb { E } _ { Q _ { e } } [ r _ { k , w _ { k } } g _ { k } ] \right\} .
$$

The rule and reweighting sets are convex compact and the bracketed functional is bilinear and separately continuous, so Sion’s minimax theorem permits interchange of infimum and supremum [33]. The resulting upper-semicontinuous function on the compact product reweighting set attains its supremum at $( w _ { 1 } ^ { \star } , \ldots , w _ { K } ^ { \star } )$ . With $( \lambda ^ { \star } , w ^ { \star } )$ fixed, the Lagrangian is the integral of the pointwise linear expression in (61). Pointwise minimization over the nonempty marginal polytope gives exactly the stated rule, with measurable tie-breaking or randomization on equality sets. Complementary slackness identifies active least-favourable reweightings [34]. □

This characterization explains the score target used in Section 5.1: strict-tail eficiency is governed by a least-favourable reweighted predictive mixture, while independent calibration can certify any prospectively fixed score family against the desired Tail risk.

## F Statistical and Computational Protocol

## F.1 Learned score systems

Watt and WCA use one-versus-rest logistic scores on a 256-component RBF Nyström map. The common hyperparameter grid is $C \in \{ 0 . 1 , 1 , 1 0 \}$ and $\gamma _ { \mathrm { R B F } } / \gamma _ { \mathrm { R B F , 0 } } \in \{ 0 . 5 , 1 , 2 \}$ . Five-fold stratified group cross-validation groups on the outer nuisance-state identity. Training uses exact $5 0 / 5 0$ positive/negative balancing, with negatives sampled uniformly across rival families and nuisance states. The RBF bandwidth reference γ<sub>RBF,0</sub> is computed from a deterministic pairwise-distance sample capped at 768 rows. Hyperparameter ties are resolved by mean log loss, then by C, then by bandwidth multiplier.

GWAE-Fluvial uses the same 256-component RBF-Nyström architecture for its $K = 2$ target. Methane oxidation uses 96 Nyström features with fixed $C = 1$ , a fixed median-bandwidth multiplier, and a grouped three-fold cross validation. In every benchmark, all observation-noise replicas belonging to one outer nuisance state remain in the same statistical fold.

## F.2 Comparator estimators and tie handling

For Watt and WCA, the final implementation evaluates every comparator on the independent experiment-selection pool. In whitened observation coordinates, let $z _ { k i }$ be selection-pool nuisance state i in family k, let ${ m } _ { k i , e }$ be its stored response under e, and define

$$
q _ { k i , e } ( y ) = \phi _ { d } ( y ; m _ { k i , e } , I _ { d } ) , \qquad p _ { k , e } ( y ) = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } q _ { k i , e } ( y ) , \qquad p _ { e } ( y ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } p _ { k , e } ( y ) .
$$

The supports are balanced and the comparator uses the same uniform structural weighting as the paper experiments. Define the finite source-component weight

$$
\omega _ { k i } : = \frac { 1 } { K n _ { k } } .
$$

For each source state the implementation draws $Y _ { k i r } \sim q _ { k i , * }$ <sub>e</sub> using a criterion-specific random seed fixed before comparison. With $R = 3 2$ , model-index and full-latent EIG are respectively

$$
\widehat { I } _ { M } ( e ) = \sum _ { k , i } \omega _ { k i } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \left\{ \log p _ { k , e } ( Y _ { k i r } ) - \log p _ { e } ( Y _ { k i r } ) \right\} ,\tag{62}
$$

$$
\widehat { I } _ { M , \Theta } ( e ) = \sum _ { k , i } \omega _ { k i } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \left\{ \log q _ { k i , e } ( Y _ { k i r } ) - \log p _ { e } ( Y _ { k i r } ) \right\} .\tag{63}
$$

Thus the full-latent calculation treats each finite selection-pool nuisance state as a latent mixture component. The forced-classification criterion minimizes

$$
\widehat { L } _ { \mathrm { B a y e s } } ( e ) = \sum _ { k , i } \omega _ { k i } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \mathbf { 1 } \left\{ \underset { \ell } { \arg \operatorname* { m a x } } \ p _ { \ell , e } ( Y _ { k i r } ) \neq k \right\} .
$$

For ordered families $k \neq \ell ,$ the predictive-discrimination estimate is

$$
\widehat { D } _ { k \to \ell } ( e ) = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \left\{ \log p _ { k , e } ( Y _ { k i r } ) - \log p _ { \ell , e } ( Y _ { k i r } ) \right\} ,
$$

and minimum ordered-pair predictive discrimination maximizes min $\kappa { \neq } \ell \widehat { D } _ { k  \ell } ( e )$ . Expected elimination uses $R _ { \mathrm { e l i m } } = 1 6$ fresh draws per selection state. With $c _ { d } = \chi _ { d , 0 . 9 5 } ^ { 2 }$ , family ℓ is rejected for $y$ when

$$
A _ { \ell , e } ( y ) = \mathbf { 1 } \left\{ \operatorname* { m i n } _ { j } \| y - m _ { \ell j , e } \| _ { 2 } ^ { 2 } > c _ { d } \right\} ,
$$

and the expected-elimination score is

$$
\widehat { N } _ { \mathrm { e l i m } } ( e ) = \sum _ { k , i } \omega _ { k i } \frac { 1 } { R _ { \mathrm { e l i m } } } \sum _ { r = 1 } ^ { R _ { \mathrm { e l i m } } } \left\{ \sum _ { \ell = 1 } ^ { K } A _ { \ell , e } ( Y _ { k i r } ) - A _ { k , e } ( Y _ { k i r } ) \right\} .
$$

The subtraction prevents exclusion of the true family from being counted as elimination of a rival.

The selection support has 40 states per family for Watt and 100 per family for WCA. The EIG standard error is obtained from the per-source sample variances and finite-mixture weights. If $e _ { \widehat { I } } ^ { \star }$ is the literal numerical optimum of an information estimate ${ \widehat { I } } ,$ the prospectively defined Monte Carlo indistinguishability set is

$$
\mathcal { O } _ { \mathrm { E I G } } ^ { \mathrm { M C } } = \left\{ e : | \widehat { I } ( e ) - \widehat { I } ( e _ { \widehat { I } } ^ { \star } ) | \leq \operatorname* { m a x } \left( 1 0 ^ { - 1 2 } , 2 \sqrt { \widehat { \mathrm { S E } } ( e ) ^ { 2 } + \widehat { \mathrm { S E } } ( e _ { \widehat { I } } ^ { \star } ) ^ { 2 } } \right) \right\} .
$$

For Bayes error, expected elimination, and minimum ordered-pair discrimination the corresponding absolute-score tolerance is $1 0 ^ { - 1 2 }$ . A scalar downstream $J ( e _ { \mathrm { E I G } } )$ is evaluated at the numerical optimum, with experiment identifier used only to break an exact score tie.

Methane uses the same comparator definitions on 256 selection nuisance states per family, with eight predictive draws per state. The second GWAE-Fluvial campaign compares $\delta = 0 . 0 5$ Tail-RAED with an independently ranked $\delta = 1$ nuisance-average RAED design on the same new nuisance population; the external comparator suite is not used in this comparison.

## F.3 Threshold calibration and Tail evaluation

To keep the implementation notation compact, the claim index $j$ is suppressed in $\eta ( \lambda )$ , and both $j$ and the current threshold λ are suppressed in $\widehat { L } _ { i }$ and $\widehat { H } _ { i }$ . For each fixed score system, family thresholds are nested from the all-family safe endpoint toward increasingly selective rules. For $0 < \delta < 1$ , the complete auxiliary OCE map $\eta ( \lambda )$ is constructed only from the independent selection role and fixed before final calibration. At each threshold candidate with selection-role nuisance losses $\ell _ { 1 } , \ldots , \ell _ { n }$ , the implementation minimizes

$$
\eta + \frac { 1 } { \delta n } \sum _ { i = 1 } ^ { n } ( \ell _ { i } - \eta ) _ { + }
$$

over $\{ 0 , 1 , \ell _ { 1 } , \dots , \ell _ { n } \}$ ; ties are resolved by the smallest $\eta ,$ and the exact safe endpoint is assigned $\eta = 0$

For methane and the GWAE positive-tail calibration, each independent nuisance state contributes the transformed bounded loss

$$
\widehat { H } _ { i } = \delta \eta + ( \widehat { L } _ { i } - \eta ) _ { + } .
$$

With $\begin{array} { r } { \bar { H } = n ^ { - 1 } \sum _ { i } \widehat { H } _ { i } } \end{array}$ , the bounded-mean Bernoulli–KL upper confidence bound is

$$
U _ { \mathrm { K L } } ( \bar { H } , n , \gamma ) = \operatorname* { s u p } \left\{ q \in [ \bar { H } , 1 ] : \mathrm { k l } ( \bar { H } \| q ) \leq \frac { \log ( 1 / \gamma ) } { n } \right\} ,\tag{64}
$$

where kl is binary relative entropy. The Jensen result in Corollary D.3 makes calibration based on inner Monte Carlo conservative for the underlying nuisance-specific Tail loss. Methane uses $\gamma = 0 . 0 5 / 3$ ; the GWAE Tail protocol uses $\gamma = 0 . 0 5 / 2$

Watt and WCA use the same nested threshold construction for their held-out empirical Tail analyses, but their late-role nuisance counts are not interpreted as supporting a small-tail population certificate.

## F.4 Selection and held-out firewall

Score fitting and experiment ranking occur before the final calibration pool is accessed. Once an experiment is selected, its score system and nested threshold path are fixed. Calibration then chooses the deployed threshold on an independent nuisance pool, and the untouched evaluation pool is used only after these decisions are complete. This is the selected-only architecture of Theorem D.4.

The principal role sizes are:

<table><tr><td></td><td></td><td></td><td></td><td>Score training Ranking/selection Final calibration Untouched evaluation</td></tr><tr><td>Watt states/family (draws/state)</td><td>80 (12)</td><td>40 (256)</td><td>40 (256)</td><td>40 (512)</td></tr><tr><td>WCA states/family (draws/state)</td><td>600 (12)</td><td>100 (256)</td><td>100 (256)</td><td>200 (512)</td></tr><tr><td>GWAE Tail-average campaign states/family (draws/state)</td><td>1000 (12)</td><td>1000 (256)</td><td>5000 (256)</td><td>5000 (512)</td></tr><tr><td>Methane states/family (draws/state)</td><td>128 (3)</td><td>256 (64)</td><td>10000 (64)</td><td>5000 (64)</td></tr></table>

The GWAE row refers to the independent second campaign used for the Tail-versus-average comparison. The earlier nuisance-average study in Section 7.3 uses separate populations, including 1000 calibration and 2000 evaluation nuisance states per family. The selection-draw entries for Watt, WCA, and GWAE refer to the learned RAED conditional-loss calculation. Comparator-specific Monte Carlo counts are stated above. Pools are disjoint across the four columns; common random numbers are confined to a role.

## G Reservoir Benchmark Construction and Reproducibility

## G.1 Watt

The Watt benchmark is derived from the hierarchical reservoir-characterization case study of Arnold et al. [26]. The source hierarchy combines three fault models (FM1–FM3), three cutof alternatives, three grid representations, and three top-structure alternatives. The RAED benchmark fixes CO3, G3, and TS2 and retains FM1–FM3 together with the two supplied property representations, OBJECT and PIXEL, and the two relative-permeability groups, $\mathrm { R P _ { 0 } }$ and $\mathrm { R P _ { 1 } }$ . The three fault models contain the same named major faults but difer in their traces, extents, face counts, and resulting compartment connections.

Model ontologies and nuisance. The resulting ensemble supports four definitions of structural identity: FM (K = 3), FM×PROP (K = 6), FM×RP $( K = 6 )$ , and FM×PROP×RP $( K = 1 2 )$ . PROP distinguishes the OBJECT and PIXEL porosity/permeability representations, while RP distinguishes the two relative-permeability groups. Variables not included in a given structural identity remain part of the within-family nuisance state. In all four ontologies, seven continuous multipliers vary the transmissibility of the existing named faults,

$$
T _ { 2 } , T _ { 3 } , T _ { 5 } , T _ { 6 } , T _ { 7 } , T _ { 8 } , T _ { 9 } \sim U [ 0 , 1 ] .
$$

These multipliers change fault transmissibility without modifying the underlying fault geometry. Fifty shared seven-dimensional Latin-hypercube coordinates are crossed with the three fault models, two property representations, and two relative-permeability groups, yielding

$$
5 0 \times 3 \times 2 \times 2 = 6 0 0
$$

reservoir realizations. The same physical ensemble is therefore reused across the four model ontologies;   
only the definition of which attributes belong to M and which remain in θ changes.

Grid, properties, and initialization. For G3, the source deck and property arrays define a $1 1 2 \times 3 0 \times 4 0$ model, whereas the geometry files include an additional 41st layer. We therefore retain the common 40-layer domain, giving 134,400 cells. The omitted layer contains only ten active cells, represents approximately $2 . 7 \times 1 0 ^ { - 5 }$ of the active geometric volume, and contains no wells or retained fault content. The OBJECT permeability field, which is absent from the standard scenario directory, was recovered from the oficial Watt builder archive.

SATNUM is used as the saturation-function region index, with zero-valued entries assigned to the first valid region. Each relative-permeability group retains its three source SWOF tables. Reservoir realizations are initialized directly from EQUIL using a datum depth of 5200 ft, a datum pressure of 2800 psia, an oil–water contact at 5326 ft, and the source constant dissolved-gas relation. Fault segments extending beyond the retained 40-layer grid are truncated at the model boundary.

Wells and physical controls. The common well system contains INJ1–INJ6, B, WELL1, WELL4, WELL6, and WELL9. Ten active well tests are defined in Table 4. All tests begin from the same equilibrium state and perturb one designated actuator while the remaining wells continue under their common background controls.

Table 4: Watt active well-test library.
<table><tr><td>Test</td><td>Actuator</td><td>Baseline control</td><td>Perturbation</td><td>Remote pressure wells</td></tr><tr><td>INJ1 multirate</td><td>INJ1</td><td>RATE 12500  $\overline { { { \mathrm { m } } ^ { 3 } / { \mathrm { d } } } }$ </td><td>alternating ±15% in four 7-d WELL1, WELL4, WELL9 stages</td><td></td></tr><tr><td>INJ4 multirate</td><td>INJ4</td><td>RATE 15000  $\mathrm { m ^ { 3 } / d }$ </td><td>alternating ±15% in four 7-d WELL1, WELL4, WELL6</td><td></td></tr><tr><td>INJ4 pulse/falloff</td><td>INJ4</td><td>RATE 15000  $\mathrm { m ^ { 3 } / d }$ </td><td>stages +20% for 10 d; shut 5 d; baseline WELL1, WELL4, WELL6</td><td></td></tr><tr><td>INJ5 pulse/falloff</td><td>INJ5</td><td>RATE 15000  $\mathrm { m ^ { 3 } / d }$ </td><td>thereafter +20% for 10 d; shut 5 d; baseline WELL1, WELL4, WELL9</td><td></td></tr><tr><td>INJ6 periodic</td><td>INJ6</td><td>RATE  $1 5 0 0 0 ~ \mathrm { m ^ { 3 } / d }$ </td><td>thereafter alternating ±15% in four 7-d WELL6, WELL4, WELL9</td><td></td></tr><tr><td>B periodic</td><td>B</td><td>RATE 10000  $\mathrm { m ^ { 3 } / d }$ </td><td>stages alternating ±15% in four 7-d WELL4, WELL6, WELL9</td><td></td></tr><tr><td>WELL1 multirate</td><td>WELL1 LRAT 8000</td><td> $\mathrm { m ^ { 3 } / d }$ </td><td>stages alternating ±15% in four 7-d WELL4, WELL6, INJ1</td><td></td></tr><tr><td>WELL4 multirate</td><td>WELL4</td><td> $\mathrm { L R A T ~ 8 0 0 0 ~ m ^ { 3 } / d }$ </td><td>stages alternating ±15% in four 7-d WELL1, WELL6, B</td><td></td></tr><tr><td>WELL6 multirate</td><td>WELL6</td><td> $\mathrm { L R A T ~ 8 0 0 0 ~ m ^ { 3 } / d }$ </td><td>stages alternating ±15% in four 7-d WELL4, WELL9, INJ6</td><td></td></tr><tr><td>WELL9 drawdown/build-up WELL9</td><td></td><td> $\mathrm { L R A T ~ 6 5 0 0 ~ m ^ { 3 } / d }$ </td><td>stages +15% for 7 d; shut 7 d; baseline WELL4, WELL1, INJ5 thereafter</td><td></td></tr></table>

Response and observation noise. Each well test produces actuator BHP, realized actuator rate, and BHP at three remote wells at days

$$
1 , 2 , 3 , 5 , 7 , 1 0 , 1 4 , 2 1 , 3 0 , 4 5 , 6 0 , 7 5 , 9 0 ,
$$

giving a 65-component response vector. The constrained-observation studies are obtained from these complete responses by truncating the time horizon and retaining only the selected remote pressure traces.

Observation noise is modeled as independent Gaussian perturbations. For pressure, the standard deviation is the larger of 0.5% of the median absolute response and 5 psia. For rate, it is the larger of 1% of the median absolute response and 50 STB/day. The full ensemble comprises 600 reservoir realizations under ten well tests, giving 6000 OPM Flow responses to day 90.

## G.2 WCA

The WCA benchmark is based on the multiple-geological-interpretation study of Park et al. [30]. The source study considers three parent training images, TI1–TI3, representing alternative fluvial depositional architectures. We take the parent training image as the structural identity, $M \in$ {TI1, TI2, TI3}. Each parent defines a family of conditional geological realizations, which are treated as within-family nuisance rather than as separate structural hypotheses.

Conditional geometry and petrophysics. Each training image contributes ten conditional facies realizations that honor the source hard data. These categorical models are mapped to a $5 3 \times 4 0 \times 1 1 6$ simulation grid, preserving the full native vertical resolution and the main facies geometry and directional connectivity of the source realizations. The mapping uses overlap-volume categorical remapping. The selected grid was chosen from a set of reduced candidates based on preservation of the facies masks and anisotropy.

For each facies realization, porosity and permeability are varied through spatially correlated random fields mapped to source-derived property distributions. The permeability field is coupled to the porosity field through

$$
Z _ { K } = 0 . 8 5 Z _ { \phi } + \sqrt { 1 - 0 . 8 5 ^ { 2 } } \epsilon ,
$$

after which both fields are mapped through their corresponding empirical distributions. Horizontal permeability is isotropic, $K _ { x } = K _ { y } ,$ with vertical permeability set to $K _ { z } = 0 . 1 K _ { x }$ . Each of the ten conditional geometries is combined with 100 matched petrophysical realizations, giving 1000 nuisance states per parent training image.

Physical model and wells. The simulation domain spans 1609.344 m by 804.672 m and 243.84 m vertically, with the top of the model at 1000 m depth. Facies code 0 is treated as inactive, while codes 1–3 form the active reservoir. All realizations share the same oil–water fluid model and an oil–water contact at 1058 m. Nine wells are placed on a common $3 \times 3$ lattice; their grid coordinates (I, J) are

<table><tr><td></td><td>West</td><td>Centre</td><td>East</td></tr><tr><td>North</td><td>W01 (11,8)</td><td>W02 (27,8)</td><td>W03 (43,8)</td></tr><tr><td>Middle</td><td>W04 (11, 20)</td><td>W05 (27,20)</td><td>W06 (43, 20)</td></tr><tr><td>South</td><td>W07 (11, 32)</td><td>W08 (27,32)</td><td>W09 (43, 32)</td></tr></table>

Wells are completed only in active cells above the oil–water contact, with every well retaining at least one active completion in every reservoir realization.

Physical controls. The ten well tests are listed in Table 5. The library includes reciprocal injector– producer pairs along the principal directions, diagonal interference tests, multirate excitation, alternating injection, a balanced multiwell pattern, and a central pulse/recovery test. Unless stated otherwise, the pair tests use an injector rate of 1000 m<sup>3</sup>/day and a producer controlled at 90 bar BHP.

Table 5: WCA active well-test library.
<table><tr><td>ID</td><td>Test</td><td>Well-test schedule</td></tr><tr><td>E01</td><td>West-east</td><td>W04 injects at 1000  $\mathrm { \overline { { m ^ { 3 } / d } } } ;$  W06 produces at 90 bar BHP</td></tr><tr><td>E02</td><td>East-west</td><td>W06 injects at 1000  $\mathrm { m ^ { 3 } / d ; }$  W04 produces at 90 bar BHP</td></tr><tr><td>E03</td><td>North-south</td><td>W02 injects at 1000  $\mathrm { m ^ { 3 } / d ; }$  W08 produces at 90 bar BHP</td></tr><tr><td>E04</td><td>South-north</td><td>W08 injects at 1000  $\mathrm { m ^ { 3 } / d ; }$  W02 produces at 90 bar BHP</td></tr><tr><td>E05</td><td>NW-SE diagonal</td><td>W01 injects at 1000  $\mathrm { m ^ { 3 } / d } ;$  W09 produces at 90 bar BHP</td></tr><tr><td>E06</td><td>SW-NE diagonal</td><td>W07 injects at 1000  $\mathrm { m ^ { 3 } / d ; }$  W03 produces at 90 bar BHP</td></tr><tr><td>E07</td><td>West-east multirate</td><td>W04 injects at  $5 0 0 / 1 0 0 0 / 5 0 0 ~ \mathrm { m ^ { 3 } / d }$  over three 20-d stages; W06 remains at 90 bar BHP</td></tr><tr><td>E08</td><td>Alternating interference</td><td>W04 and W06 alternate as 1000  $\mathrm { m ^ { 3 } / d }$  injectors every 10 d; W05 remains at 90 bar BHP</td></tr><tr><td>E09 E10</td><td>Multiwell interference</td><td>W01/W07 inject 500  $\mathrm { m ^ { 3 } / d }$  each; W03/W09 produce 500  $\mathrm { m ^ { 3 } / d }$  each</td></tr><tr><td></td><td>Central pulse/recovery</td><td>W05 injects 1000  $\mathrm { m ^ { 3 } / d }$  for 20 d against four 250  $\mathrm { m ^ { 3 } / d }$  corner producers, followed by near- shut-in monitoring</td></tr></table>

Observation and simulation ensemble. Each well test is represented by actuator BHP, realized actuator rate, and BHP at three remote wells, sampled at days

$$
1 , 2 , 3 , 5 , 7 , 1 0 , 1 4 , 2 1 , 3 0 , 4 0 , 5 0 , 6 0 ,
$$

giving a 60-component response vector. The constrained-sensing analysis truncates these responses at 7, 14, 30, or 60 days and retains pressure from 0–3 remote wells. Observation noise is modeled as independent Gaussian perturbations with standard deviations of 0.25 bar for pressure and 20 m<sup>3</sup>/day for realized rate.

The ensemble contains 3000 reservoir realizations, each simulated under the ten well tests, giving 30,000 OPM Flow responses to day 60. Diferent observation horizons and monitor selections are obtained directly from these complete responses without rerunning the flow simulations.

## G.3 GWAE-Fluvial

The GWAE-Fluvial benchmark builds on the single- and double-channel scenarios of Shishaev, Demyanov, and Arnold [31] and the associated geological prior information reported in that work. We use the supplied reservoir grid and forward-model assets together with the reported channe dimensions and structural ranges to construct a prospective parametric fluvial ensemble. Geological realizations are generated directly from interpretable channel-geometry and petrophysical parameters.

Structural prior and geometry. The K = 2 target distinguishes a reservoir containing one connected spanning channel from one containing two separated spanning channels. In local centerline coordinates, each channel follows

$$
v ( u ) = v _ { 0 } + \frac { A } { 2 } \sin \left( \frac { 2 \pi u } { \lambda } + \phi \right) ,
$$

where A denotes the total peak-to-trough excursion. The geometric parameter ranges are
<table><tr><td>Parameter</td><td>M1</td><td>M2</td></tr><tr><td>Width</td><td>300–500 m</td><td>300–500 m per channel</td></tr><tr><td>Thickness</td><td>10-20 m</td><td>10–20 m per channel</td></tr><tr><td>Wavelength</td><td>1000-2000 m</td><td>500-1000 m</td></tr><tr><td>Amplitude</td><td>0-300 m</td><td>500–900 m</td></tr><tr><td>Orientation</td><td> $9 0 ^ { \circ }$ </td><td> $1 2 0 ^ { \circ }$ </td></tr></table>

For M2, the two companion channels share wavelength, amplitude, orientation, and phase, while their width, thickness, vertical placement, and lateral position may difer. Their relative placement is constrained so that the two channel bodies remain spatially distinct.

Each channel body is required to span at least 80% of the longitudinal domain, have an elongation of at least 1.5, form a single connected component, contain no closed hole in plan view, and occupy at least 1% of the reservoir bulk volume. For M2, the two channels must remain separate, with no plan-view overlap and at least 300 m separation between occupied cell centers. The resulting ensemble is therefore the parametric proposal restricted to geometrically feasible oneand two-channel realizations.

Petrophysics and forward model. The supplied reservoir grid contains 16 ×12×10 = 1920 cells over approximately 2400 m by 1800 m, with depths of about 2416–2478 m. Within active channel cells, porosity and permeability are varied together through a rock-quality variable $q \sim U [ 0 , 1 ]$ mapped to source-derived quantiles of the supplied property fields. The same petrophysical law is used for M1 and M2, while cells outside the channel bodies are inactive.

The geological realizations are simulated with OPM Flow 2026.04 using the supplied grid, fluid model, and well configuration. Horizontal permeability is isotropic, $K _ { y } = K _ { x } ,$ with $K _ { z } = 0 . 1 5 K _ { x }$ All realizations start from the same equilibrium state. The well layout consists of west and east injector banks separated by three central producers, retaining the supplied well locations and completions.

Physical experiment library. All nine well tests last ten days and are observed at the same five report times. The experiment library is listed in Table 6.

Table 6: GWAE-Fluvial active well-test library. BHP values are absolute pressure; RATE values are are surface-volume rates.  
ID Test Well-test schedule   
E01 Symmetric banks I1–I6 at 260 bar; P1–P3 at 220 bar for 10 d   
E02 West bank I1–I3 at 260 bar; P1–P3 at 220 bar for 10 d   
E03 East bank I4–I6 at 260 bar; P1–P3 at 220 bar for 10 d   
E04 Cross I1–I6 I1 and I6 at 260 bar; P1 and P3 at 220 bar   
E05 Cross I3–I4 I3 and I4 at 260 bar; P1 and P3 at 220 bar   
E06 Midline pair I2 and I5 at 260 bar; P2 at 220 bar   
E07 Staged-rate pair I2/I5 at RATE 6 for days 0–5 and RATE 12 for days 5–10, with 270-bar limit; P1–P3 at 220 bar   
E08 Pulse/fallof I2/I5 at 270 bar for 3 d, then shut; P1–P3 remain at 220 bar   
E09 Alternating banks I1–I3 at 260 bar for days 0–5, followed by I4–I6 at 260 bar for days 5–10; P1–P3 at 220 bar

Response, noise, and statistical roles. Each well test produces ten signals at days 1, 3, 5, 7, and 10, giving a 50-component response vector. The recorded quantities are total injection rate for the west and east banks, BHP at I2 and I5, and oil- and water-production rates at P1–P3. Independent Gaussian observation noise with standard deviation 0.5 in the corresponding pressure or rate units is added to each component, and responses are whitened by these fixed standard deviations.

The ensemble contains 12,000 nuisance states per structural family, each simulated under all nine well tests, giving 216,000 OPM Flow responses. Within each family, 1000 states are used for score training, 1000 for independent experiment ranking, 5000 for selected-only calibration, and 5000 for untouched evaluation. The same newly generated population is used to compare Tail-RAED at $\delta \ : = \ : 0 . 0 5$ with nuisance-average RAED at δ = 1. The two criteria are ranked independently, calibrated on their respective selected experiments, and evaluated on the common untouched population. Results are reported in Sections 7.4 and I.4.

## H Methane-Oxidation Benchmark Construction and Certification Protocol

The methane benchmark follows the complete-oxidation model-discrimination study of Pankajakshan et al. [32]. The three competing mechanisms are power law (PL), Langmuir–Hinshelwood (LH), and Mars–van Krevelen (MVK). Within each mechanism, the kinetic parameters are sampled from the multivariate Gaussian approximation defined by the fitted parameter vector and covariance matrix reported in the source study, without truncation.

Mechanistic models. The three candidate rate laws are

$$
r _ { \mathrm { P L } } = k _ { 1 } P y _ { \mathrm { C H } _ { 4 } } ,
$$

$$
r _ { \mathrm { L H } } = \frac { k _ { r } K _ { \mathrm { C H _ { 4 } } } ( P y _ { \mathrm { C H _ { 4 } } } ) \sqrt { K _ { \mathrm { O _ { 2 } } } P y _ { \mathrm { O _ { 2 } } } } } { \left( 1 + K _ { \mathrm { C H _ { 4 } } } P y _ { \mathrm { C H _ { 4 } } } + \sqrt { K _ { \mathrm { O _ { 2 } } } P y _ { \mathrm { O _ { 2 } } } } \right) ^ { 2 } } ,
$$

and

$$
r _ { \mathrm { M V K } } = \frac { k _ { 1 } k _ { 2 } P ^ { 2 } y _ { \mathrm { C H _ { 4 } } } y _ { \mathrm { O _ { 2 } } } } { k _ { 1 } P y _ { \mathrm { O _ { 2 } } } + 2 k _ { 2 } P y _ { \mathrm { C H _ { 4 } } } + ( k _ { 1 } k _ { 2 } / k _ { 3 } ) P ^ { 2 } y _ { \mathrm { C H _ { 4 } } } y _ { \mathrm { O _ { 2 } } } } .
$$

The forward model is a steady isothermal plug-flow reactor for $\mathrm { C H } _ { 4 } + 2 \mathrm { O } _ { 2 } \to \mathrm { C O } _ { 2 } + 2 \mathrm { H } _ { 2 } \mathrm { O }$ , using the pressure-drop, Arrhenius, and adsorption relations of the source model.

Design and observation model. The candidate design space is a full $3 ^ { 4 }$ factorial over four reactor operating variables:

<table><tr><td>Variable</td><td>Levels</td></tr><tr><td>Temperature</td><td>250, 300,  $\overline { { 3 5 0 ^ { \circ } C } }$ </td></tr><tr><td>Normal feed flow</td><td>20, 25, 30 Nml/min</td></tr><tr><td> $\mathrm { O _ { 2 } / C H _ { 4 } }$  ratio</td><td>2, 3, 4</td></tr><tr><td>Inlet methane fraction</td><td>0.005, 0.015, 0.025</td></tr></table>

This gives 81 candidate reactor conditions. The observation is the outlet mole-fraction vector $\left( y _ { \mathrm { C H _ { 4 } } } , y _ { \mathrm { O _ { 2 } } } , y _ { \mathrm { C O _ { 2 } } } \right)$ with independent Gaussian noise,

$$
\Sigma _ { y } = \mathrm { d i a g } ( 0 . 0 0 0 4 3 ^ { 2 } , 0 . 0 0 2 0 2 ^ { 2 } , 0 . 0 0 0 5 1 ^ { 2 } ) .
$$

Statistical roles and certification. For each structural family, 128 nuisance states are used for score training, 256 for independent experiment selection, 10,000 for selected-only calibration, and 5000 for untouched evaluation. The corresponding numbers of observation-noise draws per nuisance state are 3, 64, 64, and 64. Nuisance states and noise draws are independent across the four statistical roles.

For $0 < \delta < 1$ , the selection sample is also used to fix the auxiliary map $\eta ( \lambda )$ . For each threshold, η is chosen from the observed conditional-loss values together with 0 and 1 to minimize

$$
\eta + \frac { 1 } { \delta } \operatorname* { m e a n } ( L - \eta ) _ { + } .
$$

Final calibration then uses the transformed loss

$$
H = \delta \eta + ( \widehat { L } - \eta ) _ { + }
$$

together with the Bernoulli–KL upper bound in (64). The familywise failure probability is set to

$$
\gamma _ { k } = \frac { 0 . 0 5 } { 3 } ,
$$

giving 95% joint confidence across PL, LH, and MVK at each fixed positive δ.

The declared operating points are

$$
\delta \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 1 \} .
$$

The $\delta = 0$ analysis is empirical. The guarantees at positive $\delta$ apply separately at each fixed operating point rather than simultaneously across the full δ grid.

## I Additional Empirical Results

## I.1 Watt ontology summary

The Watt ontology analysis was evaluated at $\alpha = 0 . 0 5$ over the full declared grid

$$
\delta \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 1 \} .
$$

The same INJ5 pulse/fallof experiment is selected for all four structural ontologies at every value of δ. The tables below report the held-out mean candidate-set size J, normalized resolution $R = ( K - J ) / ( K - 1 )$ , and worst-family held-out exclusion risk $\widehat { \rho } _ { \delta , \mathrm { m a x } }$

For $\delta = 0 , \widehat { \rho } _ { \delta , \operatorname* { m a x } }$ is the largest conditional false-exclusion rate over the held-out nuisance states. For $0 < \delta < 1$ , it is the largest familywise empirical upper-δ Tail false-exclusion risk, and for $\delta = 1$ it is the largest familywise nuisance-average false-exclusion rate. These are held-out empirical summaries of the nontrivial learned rules.

Table 7: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 0 )$
<table><tr><td>Structural target</td><td>K</td><td>J</td><td>R</td><td> $\widehat { \rho } _ { 0 , \mathrm { { m a x } } }$ </td></tr><tr><td>FM</td><td>3</td><td>1.80783</td><td>0.59609</td><td>0.28125</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P }$ </td><td>6</td><td>1.75653</td><td>0.84869</td><td>0.16797</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.84614</td><td>0.83077</td><td>0.09766</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.82743</td><td>0.92478</td><td>0.09375</td></tr></table>

Table 8: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 0 . 0 1 )$
<table><tr><td>Structural target</td><td>K</td><td>J</td><td>R</td><td>ρ0.01,max</td></tr><tr><td>FM</td><td>3</td><td>1.80783</td><td>0.59609</td><td>0.28125</td></tr><tr><td>FM×PROP</td><td>6</td><td>1.75653</td><td>0.84869</td><td>0.16797</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.84614</td><td>0.83077</td><td>0.09766</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.82743</td><td>0.92478</td><td>0.09375</td></tr></table>

Table 9: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 0 . 0 2 )$
<table><tr><td>Structural target</td><td> $K$ </td><td> $J$ </td><td> $R$ </td><td> $\widehat { \rho } _ { 0 . 0 2 , \mathrm { { m a x } } }$ </td></tr><tr><td>FM</td><td>3</td><td>1.80783</td><td>0.59609</td><td>0.28125</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P }$ </td><td>6</td><td>1.75653</td><td>0.84869</td><td>0.16797</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.84614</td><td>0.83077</td><td>0.09766</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.82743</td><td>0.92478</td><td>0.09375</td></tr></table>

Table 10: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 0 . 0 5 )$
<table><tr><td>Structural target</td><td> $K$ </td><td>J</td><td>R</td><td> $\widehat { \rho } _ { 0 . 0 5 , \mathrm { { m a x } } }$ </td></tr><tr><td>FM</td><td>3</td><td>1.76025</td><td>0.61987</td><td>0.17969</td></tr><tr><td>FM×PROP</td><td>6</td><td>1.75653</td><td>0.84869</td><td>0.16797</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.84614</td><td>0.83077</td><td>0.09766</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.82743</td><td>0.92478</td><td>0.09375</td></tr></table>

Table 11: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 0 . 1 0 )$
<table><tr><td>Structural target</td><td> $K$ </td><td>J</td><td>R</td><td> $\widehat { \rho } _ { 0 . 1 0 , \mathrm { { m a x } } }$ </td></tr><tr><td>FM</td><td>3</td><td>1.73477</td><td>0.63262</td><td>0.11230</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P }$ </td><td>6</td><td>1.72793</td><td>0.85441</td><td>0.13672</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.82606</td><td>0.83479</td><td>0.08789</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.82743</td><td>0.92478</td><td>0.09375</td></tr></table>

Table 12: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 0 . 2 0 )$ .
<table><tr><td>Structural target</td><td>K</td><td>J</td><td>R</td><td> $\widehat { \rho } _ { 0 . 2 0 , \mathrm { { m a x } } }$ </td></tr><tr><td>FM</td><td>3</td><td>1.68870</td><td>0.65565</td><td>0.08032</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P }$ </td><td>6</td><td>1.67109</td><td>0.86578</td><td>0.09473</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.77010</td><td>0.84598</td><td>0.07568</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.77288</td><td>0.92974</td><td>0.07520</td></tr></table>

Table 13: Held-out Watt ontology results at $( \alpha , \delta ) = ( 0 . 0 5 , 1 )$
<table><tr><td>Structural target</td><td>K</td><td> $J$ </td><td>R</td><td> $\widehat { \rho } _ { 1 , \mathrm { { m a x } } }$ </td></tr><tr><td>FM</td><td>3</td><td>1.42052</td><td>0.78974</td><td>0.05586</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P }$ </td><td>6</td><td>1.43800</td><td>0.91240</td><td>0.06709</td></tr><tr><td> $\mathrm { F M } \times \mathrm { R P }$ </td><td>6</td><td>1.52469</td><td>0.89506</td><td>0.05723</td></tr><tr><td> $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ </td><td>12</td><td>1.59907</td><td>0.94554</td><td>0.06055</td></tr></table>

For every positive $\delta ,$ the available Watt calibration sample was insuficient to certify a nontrivial population-valid rule, so these tables describe the held-out empirical frontier rather than the final safe-endpoint deployment. The $\delta = 0$ results are empirical pointwise diagnostics. Across the full empirical frontier, INJ5 pulse/fallof remains the selected experiment and the same ontology ordering in normalized resolution is retained: $\mathrm { F M } { \times } \mathrm { P R O P } { \times } \mathrm { R P }$ has the highest $R ,$ followed by $\mathrm { F M } { \times } \mathrm { P R O P } ,$ $\mathrm { F M } \times \mathrm { R P } .$ , and FM. Tightening the nuisance-validity requirement reduces or leaves unchanged the empirical resolution, but the finer ontologies retain high normalized resolution throughout the declared δ range.

## I.2 Complete restricted-observation results at $\delta = 0 . 0 5$

Tables 14 and 15 report all 16 observation-horizon and remote-monitor combinations used in the constrained-sensing studies at $\alpha = \delta = 0 . 0 5$ . For each sensing level, $| \mathcal { O } _ { \mathrm { E I G } } ^ { \mathrm { M C } } |$ denotes the size of the Monte Carlo uncertainty set around the model-index-EIG optimum. A clean disagreement is recorded when the RAED-selected experiment lies outside this set. The reported EIG result corresponds to the deterministic representative defined in Appendix F.

For each sensing level, the RAED- and EIG-selected experiments are evaluated using the same score-fitting, threshold-calibration, and independent held-out evaluation procedure. We report

$$
\Delta J = J _ { \mathrm { E I G } } - J _ { \mathrm { R A E D } } ,
$$

so that positive values indicate a smaller held-out candidate set for the RAED-selected experiment. We also report the worst-family held-out upper-5%-Tail false-exclusion risks, denoted $\widehat { \rho } _ { \mathrm { R } }$ and $\widehat { \rho } _ { \mathrm { E } }$ for the RAED- and EIG-selected experiments, respectively. These are empirical held-out risks rather than population-validity certificates.

Table 14: Complete Watt restricted-observation results at $\alpha = \delta = 0 . 0 5$ . Positive $\Delta J$ indicates a smaller held-out candidate set for the RAED-selected experiment. The final two columns report the worst-family held-out upper-5%-Tail false-exclusion risk for the RAED- and EIG-selected experiments.
<table><tr><td>T</td><td>Remote wells</td><td> $| \mathcal { O } _ { \mathrm { E I G } } ^ { \mathrm { M C } } |$ </td><td>Clean</td><td> $J _ { \mathrm { R A E D } }$ </td><td> $J _ { \mathrm { E I G } }$ </td><td> $\overline { { \Delta J } }$ </td><td> $\widehat { \rho _ { \mathrm { R } } }$ </td><td> $\hat { \rho _ { \mathrm { E } } }$ </td></tr><tr><td>7</td><td>0</td><td>1</td><td></td><td>2.34551</td><td>2.34551</td><td>0.00000</td><td>0.05957</td><td>0.05957</td></tr><tr><td>7</td><td>1</td><td>2</td><td>一</td><td>2.26751</td><td>2.24946</td><td>-0.01805</td><td>0.05176</td><td>0.04785</td></tr><tr><td>7</td><td>2</td><td>1</td><td>一</td><td>2.22135</td><td>2.22135</td><td>0.00000</td><td>0.04590</td><td>0.04590</td></tr><tr><td>7</td><td>3</td><td>2</td><td>一</td><td>2.26528</td><td>2.28561</td><td>0.02033</td><td>0.05371</td><td>0.06055</td></tr><tr><td>14</td><td>0</td><td>1</td><td>一</td><td>2.35125</td><td>2.35125</td><td>0.00000</td><td>0.05957</td><td>0.05957</td></tr><tr><td>14</td><td>1</td><td>1</td><td>一</td><td>2.16966</td><td>2.16966</td><td>0.00000</td><td>0.04395</td><td>0.04395</td></tr><tr><td>14</td><td>2</td><td>3</td><td>一</td><td>2.02651</td><td>2.02651</td><td>0.00000</td><td>0.05176</td><td>0.05176</td></tr><tr><td>14</td><td>3</td><td>2</td><td>一</td><td>2.13758</td><td>2.13758</td><td>0.00000</td><td>0.05469</td><td>0.05469</td></tr><tr><td>30</td><td>0</td><td>1</td><td>yes</td><td>2.31328</td><td>2.36720</td><td>0.05392</td><td>0.05664</td><td>0.06445</td></tr><tr><td>30</td><td>1</td><td>1</td><td>yes</td><td>2.02734</td><td>2.04469</td><td>0.01735</td><td>0.06934</td><td>0.08105</td></tr><tr><td>30</td><td>2</td><td>2</td><td>一</td><td>1.90103</td><td>1.90103</td><td>0.00000</td><td>0.05762</td><td>0.05762</td></tr><tr><td>30</td><td>3</td><td>1</td><td>一</td><td>1.99173</td><td>1.99173</td><td>0.00000</td><td>0.08301</td><td>0.08301</td></tr><tr><td>90</td><td>0</td><td>2</td><td>yes</td><td>2.21790</td><td>2.26066</td><td>0.04276</td><td>0.17383</td><td>0.08398</td></tr><tr><td>90</td><td>1</td><td>3</td><td>一</td><td>1.78530</td><td>1.97607</td><td>0.19077</td><td>0.09180</td><td>0.12402</td></tr><tr><td>90</td><td>2</td><td>1</td><td>一</td><td>1.69167</td><td>1.69167</td><td>0.00000</td><td>0.15820</td><td>0.15820</td></tr><tr><td>90</td><td>3</td><td>1</td><td>一</td><td>1.75649</td><td>1.75649</td><td>0.00000</td><td>0.08594</td><td>0.08594</td></tr></table>

Table 15: Complete WCA restricted-observation results at $\alpha = \delta = 0 . 0 5$ . Positive $\Delta J$ indicates a smaller held-out candidate set for the RAED-selected experiment. The final two columns report the worst-family held-out upper-5%-Tail false-exclusion risk for the RAED- and EIG-selected experiments.
<table><tr><td>T</td><td>Remote wells</td><td> $| \overline { { \mathcal { O } _ { \mathrm { E I G } } ^ { \mathrm { M C } } } } |$ </td><td>Clean</td><td> $J _ { \mathrm { R A E D } }$ </td><td> $J _ { \mathrm { E I G } }$ </td><td> $\overline { { \Delta J } }$ </td><td> $\widehat { \rho _ { \mathrm { R } } }$ </td><td> $\widehat { \rho _ { \mathrm { E } } }$ </td></tr><tr><td>7</td><td>0</td><td>2</td><td>yes</td><td>2.81695</td><td>2.81291</td><td>-0.00405</td><td>0.28770</td><td>0.16934</td></tr><tr><td>7</td><td>1</td><td>4</td><td>yes</td><td>2.56748</td><td>2.88200</td><td>0.31452</td><td>0.20586</td><td>0.04941</td></tr><tr><td>7</td><td>2</td><td>5</td><td>yes</td><td>2.37828</td><td>2.76330</td><td>0.38503</td><td>0.36484</td><td>0.11133</td></tr><tr><td>7</td><td>3</td><td>3</td><td>yes</td><td>2.36770</td><td>2.47161</td><td>0.10391</td><td>0.28301</td><td>0.21719</td></tr><tr><td>14</td><td>0</td><td>5</td><td>yes</td><td>2.79314</td><td>2.67491</td><td>-0.11823</td><td>0.20176</td><td>0.23594</td></tr><tr><td>14</td><td>1</td><td>16</td><td>yes</td><td>2.60173</td><td>2.74852</td><td>0.14680</td><td>0.21016</td><td>0.20898</td></tr><tr><td>14</td><td>2</td><td>16</td><td>yes</td><td>2.30893</td><td>2.60945</td><td>0.30052</td><td>0.48906</td><td>0.21758</td></tr><tr><td>14</td><td>3</td><td>6</td><td>一</td><td>2.34177</td><td>2.61700</td><td>0.27522</td><td>0.25176</td><td>0.16523</td></tr><tr><td>30</td><td>0</td><td>5</td><td>一</td><td>2.77794</td><td>2.74348</td><td>-0.03446</td><td>0.36934</td><td>0.54375</td></tr><tr><td>30</td><td>1</td><td>19</td><td>一</td><td>2.57873</td><td>2.59372</td><td>0.01499</td><td>0.38984</td><td>0.14863</td></tr><tr><td>30</td><td>2</td><td>21</td><td>一</td><td>2.52367</td><td>2.59551</td><td>0.07184</td><td>0.24688</td><td>0.16719</td></tr><tr><td>30</td><td>3</td><td>9</td><td>一</td><td>2.26447</td><td>2.56853</td><td>0.30406</td><td>0.31504</td><td>0.16367</td></tr><tr><td>60</td><td>0</td><td>7</td><td>一</td><td>2.77431</td><td>2.83377</td><td>0.05946</td><td>0.53281</td><td>0.13105</td></tr><tr><td>60</td><td>1</td><td>26</td><td>一</td><td>2.54613</td><td>2.71471</td><td>0.16858</td><td>0.43008</td><td>0.11172</td></tr><tr><td>60</td><td>2</td><td>26</td><td>一</td><td>2.35621</td><td>2.55004</td><td>0.19383</td><td>0.17773</td><td>0.24863</td></tr><tr><td>60</td><td>3</td><td>9</td><td>1</td><td>2.32793</td><td>2.34423</td><td>0.01630</td><td>0.20723</td><td>0.27480</td></tr></table>

The held-out Tail risks show that diferences in candidate-set size should not be interpreted independently of false exclusion. In Watt, the two clean disagreements at $( T , B ) = ( 3 0 , 0 )$ and (30, 1) combine smaller candidate sets under RAED with lower worst-family held-out Tail risk, whereas the clean disagreement at (90, 0) combines a smaller candidate set with higher Tail risk. WCA exhibits a wider range of resolution–risk tradeofs: several RAED-selected experiments produce substantially smaller candidate sets but also higher held-out Tail exclusion than the corresponding EIG-selected experiment. The restricted-sensing comparison therefore reports both quantities rather than interpreting $\Delta J$ alone as a measure of overall superiority.

None of the restricted-sensing rules in these tables carries a finite-sample population-validity certificate. The thresholds are independently empirically calibrated and the reported risks are measured on the held-out evaluation population.

Table 16: Mean held-out candidate-set size across the 16 restricted-sensing cells at $\alpha = \delta = 0 . 0 5$ Each comparator entry evaluates the deterministic representative of each criterion under the common learned-RAED downstream procedure.
<table><tr><td>Selection criterion</td><td>Watt mean J</td><td>WCA mean J</td></tr><tr><td>RAED</td><td>2.09184</td><td>2.52034</td></tr><tr><td>Model-index EIG</td><td>2.11103</td><td>2.65773</td></tr><tr><td>Full-latent EIG</td><td>2.32000</td><td>2.70372</td></tr><tr><td>Bayes forced classification</td><td>2.10709</td><td>2.69110</td></tr><tr><td>Expected elimination</td><td>2.13803</td><td>2.68784</td></tr><tr><td>Minimum ordered-pair discrimination</td><td>2.29097</td><td>2.58491</td></tr></table>

We do not report paired 95% confidence intervals for $\Delta J$ because the evaluation outputs contain aggregate candidate-set means rather than the nuisance-state-level paired contributions required for such an interval. Recovering those pairs would require rerunning the relevant model-fitting and evaluation procedure. Repeated observation-noise draws within a nuisance state are not independent nuisance samples and therefore cannot substitute for the missing outer-state pairs.

## I.3 Dependence of the selected restricted design on δ

Changing the Tail level δ can afect not only the calibrated candidate-set rule but also the physical experiment selected by RAED. Across the seven declared operating points $\delta \in \{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 1 \}$ the selected design changes in three of the sixteen Watt sensing levels and ten of the sixteen WCA sensing levels. The afected horizon–monitor combinations are listed below.

Benchmark Sensing levels with more than one selected design across δ   
Watt (7, 1), (7, 3), (90, 0)   
WCA (7, 1), (7, 2), (7, 3), (14, 1), (14, 2), (14, 3), (30, 1), (30, 2), (60, 1), (60, 2)

Each pair gives the observation horizon in days and the number of retained remote pressure wells. The stronger dependence in WCA indicates that the preferred sensing strategy is more sensitive there to the amount of nuisance-tail protection imposed by the design criterion. A change in selected design may reflect a diferent physical well test, a diferent monitor subset, or both.

## I.4 GWAE-Fluvial Tail-RAED comparison

The GWAE-Fluvial study compares Tail-RAED at δ = 0.05 with nuisance-average RAED at δ = 1 on the same independently generated geological population. Both criteria use the same score-training and experiment-ranking populations. Each selected experiment is then calibrated separately and evaluated on the same independent evaluation population.

Independent experiment ranking. Table 17 reports the ranking of all nine well tests. The displayed J values are computed on the independent ranking population before final selected-only calibration and therefore difer from the deployed evaluation values reported later in Table 18. Tail-RAED selects E07, the staged-rate pair, by a narrow margin over E01, while nuisance-average RAED selects E01, the symmetric-bank test.

Table 17: GWAE-Fluvial experiment ranking for Tail-RAED at $\delta = 0 . 0 5$ and nuisance-average RAED at δ = 1.
<table><tr><td>ID</td><td>Test</td><td>Tail rank</td><td>Tail J Average rank</td><td>Average J</td></tr><tr><td>E01</td><td>Symmetric banks</td><td>2 1.557934</td><td>1</td><td>1.019932</td></tr><tr><td>E02</td><td>West bank</td><td>5</td><td>1.631516</td><td>6 1.187480</td></tr><tr><td>E03</td><td>East bank</td><td>3 1.612088</td><td>5</td><td>1.152689</td></tr><tr><td>E04</td><td>Cross I1–I6</td><td>8 1.803926</td><td>7</td><td>1.242840</td></tr><tr><td>E05</td><td>Cross I3–I4</td><td>7 1.751137</td><td>8</td><td>1.255127</td></tr><tr><td>E06</td><td>Midline pair</td><td>9 1.969910</td><td>9</td><td>1.381551</td></tr><tr><td>E07</td><td>Staged-rate pair</td><td>1 1.557348</td><td>4</td><td>1.048691</td></tr><tr><td>E08</td><td>Pulse/falloff</td><td>6 1.634051</td><td>3</td><td>1.047578</td></tr><tr><td>E09</td><td>Alternating banks</td><td>4 1.625998</td><td>2</td><td>1.025992</td></tr></table>

Selected-experiment calibration. Calibration uses 5000 independent nuisance states per structural family for the experiment selected by each criterion. With familywise failure probability $\gamma _ { k } = 0 . 0 2 5$ , the resulting fixed-operating-point statement has 95% joint confidence across M1 and M2. For both selected experiments, calibration produces a nontrivial candidate-set rule:
<table><tr><td>Method</td><td>Selected experiment</td><td>M1 UCB</td><td>M2 UCB</td></tr><tr><td>Tail-RAED, δ = 0.05</td><td>E07 staged-rate pair</td><td>0.048330</td><td>0.049103</td></tr><tr><td>Nuisance-average, δ = 1</td><td>E01 symmetric banks</td><td>0.025999</td><td>0.049341</td></tr></table>

For Tail-RAED, the corresponding empirical calibration Tail losses are 0.015922 for M1 and 0.016391 for M2; the certification statement uses the upper confidence bounds reported above.

Common evaluation population. Table 18 reports the principal diagnostics on the common 5000-state-per-family evaluation population. Conditional false-exclusion risk is estimated separately for each nuisance state using repeated observation-noise draws, and the worst-5% statistic averages the 250 highest-risk states within each family.

Table 18: GWAE-Fluvial evaluation results on the common untouched nuisance population. Worst-5% is the empirical mean conditional false-exclusion risk among the 5% highest-risk nuisance states in each family.
<table><tr><td>Family</td><td>Method</td><td>Mean risk</td><td>Worst-5%</td><td>P99</td><td>Correct singleton</td><td>Ambiguous</td></tr><tr><td>M1</td><td>Tail</td><td>0.000689</td><td>0.013781</td><td>0.000000</td><td>0.093231</td><td>0.906080</td></tr><tr><td>M1</td><td>Average</td><td>0.022152</td><td>0.433758</td><td>0.994160</td><td>0.907827</td><td>0.070021</td></tr><tr><td>M2</td><td>Tail</td><td>0.000149</td><td>0.002977</td><td>0.000000</td><td>0.329347</td><td>0.670504</td></tr><tr><td>M2</td><td>Average</td><td>0.046046</td><td>0.875844</td><td>1.000000</td><td>0.929433</td><td>0.024521</td></tr></table>

With equal weighting of the two structural families, Tail-RAED gives $J ~ = ~ 1 . 7 8 8 2 9 2$ and $R = 0 . 2 1 1 7 0 8$ , compared with $J = 1 . 0 4 7 2 7 1$ and $R = 0 . 9 5 2 7 2 9$ for nuisance-average RAED. Because $K = 2$ , the corresponding overall ambiguity frequencies are 0.788292 and 0.047271. The pooled empirical false-exclusion risks are 0.000419 and 0.034099, respectively. Tail protection therefore increases expected candidate-set size by $\Delta J = 0 . 7 4 1 0 2 1$ while sharply reducing false exclusion in the adverse geological tail.

Hard-region behavior. As a separate statewise diagnostic, we first call a nuisance state dificult for a rule when its conditional false-exclusion risk satisfies $r _ { k } ( \theta ) > 0 . 0 5$ . Under this definition, the two rules divide the evaluation population as follows:
<table><tr><td></td><td></td><td>Family Easy for both Average hard, Tail protected Tail hard, Average protected Hard for both</td></tr><tr><td>M1</td><td>4825</td><td></td><td>5</td></tr><tr><td>M2</td><td>4686</td><td>170 310</td><td></td></tr></table>

Thus almost every state that is dificult for Tail-RAED is also dificult for the nuisance-average rule, whereas a substantially larger set of states is dificult only for the nuisance-average rule.

Within the nuisance-average rule’s own worst 5% of geological states, the diference is especially pronounced. For M1, Tail-RAED returns ambiguity on 98.62% of observation draws and falsely excludes the true family on 1.38%, compared with 39.28% ambiguity and 43.38% false exclusion for nuisance-average RAED. For M2, Tail-RAED returns ambiguity on 99.71% of draws and falsely excludes on 0.29%, compared with 11.75% ambiguity and 87.58% false exclusion for nuisance-average RAED. This is the behavior summarized in Figure 5.

## I.5 Methane frontier

The same reactor condition, MOX048, is selected at every declared value of δ. The $\delta = 0$ point is an empirical diagnostic only. For each fixed positive δ, calibration gives a familywise positive-tail population guarantee with 95% joint confidence across PL, LH, and MVK. These guarantees apply separately at each operating point and are not simultaneous across the full δ grid.

<table><tr><td>δ</td><td>J</td><td>R</td><td>Worst-family held-out  $\overline { { \widehat { \rho } _ { \delta } } }$ </td></tr><tr><td>0.00</td><td>1.86014</td><td>0.56993</td><td>0.06250</td></tr><tr><td>0.01</td><td>2.20015</td><td>0.39993</td><td>0.00281</td></tr><tr><td>0.02</td><td>1.90511</td><td>0.54744</td><td>0.01375</td></tr><tr><td>0.05</td><td>1.75046</td><td>0.62477</td><td>0.02475</td></tr><tr><td>0.10</td><td>1.72207</td><td>0.63897</td><td>0.02872</td></tr><tr><td>0.20</td><td>1.68874</td><td>0.65563</td><td>0.03405</td></tr><tr><td>1.00</td><td>1.56981</td><td>0.71510</td><td>0.04343</td></tr></table>

Here $R = ( K - J ) / ( K - 1 )$ is the normalized resolution of the resulting learned rule. At the reference operating point $\delta = 0 . 0 5$ , the familywise calibration upper bounds are 0.047649, 0.037982, and 0.047738 for PL, LH, and MVK, respectively. Requiring the finite-sample certificate increases the held-out mean candidate-set size by 0.061716 relative to the corresponding empirically calibrated rule.

## References

[1] Xun Huan, Jayanth Jagalur, and Youssef Marzouk. Optimal experimental design: Formulations and computations. Acta Numerica, 33:715–840, 2024.

[2] Tom Rainforth, Adam Foster, Desi R. Ivanova, and Freddie Bickford Smith. Modern bayesian experimental design. Statistical Science, 39(1):100–114, 2024.

[3] A. C. Atkinson and V. V. Fedorov. The design of experiments for discriminating between two rival models. Biometrika, 62(1):57–70, 1975.

[4] A. C. Atkinson and V. V. Fedorov. Optimal design: Experiments for discriminating between several models. Biometrika, 62(2):289–303, 1975.

[5] Holger Dette and Stefanie Titof. Optimal discrimination designs. The Annals of Statistics, 37(4):2056–2082, 2009.

[6] Holger Dette, Viatcheslav B. Melas, and Petr Shpilev. Robust T-optimal discriminating designs. The Annals of Statistics, 41(4):1693–1715, 2013.

[7] Markus Hainy, David J. Price, Olivier Restif, and Christopher Drovandi. Optimal bayesian design for model discrimination via classification. Statistics and Computing, 32(2):25, 2022.

[8] André Luís Alberton, Marcio Schwaab, Marcos Wandir Nery Lobão, and José Carlos Pinto. Design of experiments for discrimination of rival models based on the expected number of eliminated models. Chemical Engineering Science, 75:120–131, 2012.

[9] Mauricio Sadinle, Jing Lei, and Larry Wasserman. Least ambiguous set-valued classifiers with bounded error levels. Journal of the American Statistical Association, 114(525):223–234, 2019.

[10] Chad M. Schafer and Philip B. Stark. Constructing confidence regions of optimal expected size. Journal of the American Statistical Association, 104(487):1080–1089, 2009.

[11] Christophe Denis and Mohamed Hebiri. Confidence sets with expected sizes for multiclass classification. Journal of Machine Learning Research, 18(102):1–28, 2017.

[12] Yuanjie Shi, Subhankar Ghosh, Taha Belkhouja, Janardhan Rao Doppa, and Yan Yan. Conformal prediction for class-wise coverage via augmented label rank calibration. In Advances in Neural Information Processing Systems, volume 37, 2024.

[13] Jinwoo Go and Tobin Isaac. Robust expected information gain for optimal bayesian experimental design using ambiguity sets. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pages 728–737. PMLR, 2022.

[14] Hany Abdulsamad, Sahel Iqbal, Christian A. Naesseth, Takuo Matsubara, and Adrien Corenflos. Maximin robust bayesian experimental design. arXiv preprint arXiv:2603.14094, 2026.

[15] David Blackwell. Equivalent comparisons of experiments. The Annals of Mathematical Statistics, 24(2):265–272, 1953.

[16] Lucien Le Cam. Suficiency and approximate suficiency. The Annals of Mathematical Statistics, 35(4):1419–1455, 1964.

[17] Erik Torgersen. Comparison of Statistical Experiments. Cambridge University Press, Cambridge, 1991.

[18] David Blackwell, Leo Breiman, and A. J. Thomasian. The capacity of a class of channels. The Annals of Mathematical Statistics, 30(4):1229–1241, 1959.

[19] Imre Csiszár and Prakash Narayan. The capacity of the arbitrarily varying channel revisited: Positivity, constraints. IEEE Transactions on Information Theory, 34(2):181–193, 1988.

[20] Claude E. Shannon. The zero error capacity of a noisy channel. IRE Transactions on Information Theory, 2(3):8–19, 1956.

[21] Peter Elias. List decoding for noisy channels. In IRE WESCON Convention Record, volume 2, pages 94–104, 1957.

[22] Stephen Bates, Anastasios N. Angelopoulos, Lihua Lei, Jitendra Malik, and Michael I. Jordan. Distribution-free, risk-controlling prediction sets. Journal of the ACM, 68(6):1–34, 2021.

[23] Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. Learn then test: Calibrating predictive algorithms to achieve risk control. The Annals of Applied Statistics, 19(2):1641–1662, 2025.

[24] Jiayi Huang, Amirmohammad Farzaneh, and Osvaldo Simeone. Optimized certainty equivalent risk-controlling prediction sets. arXiv preprint arXiv:2602.13660, 2026.

[25] R. Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. Journal of Risk, 2(3):21–42, 2000.

[26] D. Arnold, V. Demyanov, D. Tatum, M. A. Christie, T. S. Rojas, S. Geiger, and P. W. M. Corbett. Hierarchical benchmark case study for history matching, uncertainty quantification and reservoir characterisation. Computers & Geosciences, 50:4–15, 2013.

[27] Dominique Bourdet. Well Test Analysis: The Use of Advanced Interpretation Models, volume 3 of Handbook of Petroleum Exploration and Production. Elsevier, 2002.

[28] Peter A. Fokker, Eloisa Salina Borello, Francesca Verga, and Dario Viberti. Harmonic pulse testing for well performance monitoring. Journal of Petroleum Science and Engineering, 162:446–459, 2018.

[29] Eloisa Salina Borello, Peter A. Fokker, Dario Viberti, Francesca Verga, Hannes Hofmann, Peter Meier, Ki-Bok Min, K. Yoon, and Günter Zimmermann. Harmonic pulse testing for well monitoring: Application to a fractured geothermal reservoir. Water Resources Research, 55(6):4727–4744, 2019.

[30] H. Park, C. Scheidt, D. Fenwick, A. Boucher, and J. Caers. History matching and uncertainty quantification of facies models with multiple geological interpretations. Computational Geosciences, 17(4):609–621, 2013.

[31] Gleb Shishaev, Vasily Demyanov, and Daniel Arnold. History matching under uncertainty of geological scenarios with implicit geological realism control with generative deep learning and graph convolutions. Computers & Geosciences, 214:106186, 2026.

[32] Arun Pankajakshan, Solomon Gajere Bawa, Asterios Gavriilidis, and Federico Galvanin. Autonomous kinetic model identification using optimal experimental design and retrospective data analysis: Methane complete oxidation as a case study. Reaction Chemistry & Engineering, 8(12):3000–3017, 2023.

[33] Maurice Sion. On general minimax theorems. Pacific Journal of Mathematics, 8(1):171–176, 1958.

[34] Birgit Rudlof and Ioannis Karatzas. Testing composite hypotheses via convex duality. Bernoulli, 16(4):1224–1239, 2010.