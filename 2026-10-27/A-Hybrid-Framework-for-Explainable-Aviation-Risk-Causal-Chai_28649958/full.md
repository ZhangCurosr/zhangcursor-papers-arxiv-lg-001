# A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis

Yutian Pang, Yifang Tian, Jiacheng Xie, Xinyu Zhao, Jueming Hu, John-Paul Clarke, and Hans-Arno Jacobsen, Fellow, IEEE

Abstract—Aviation authorities hold decades of accident reports whose narratives record how investigators reconstructed each accident. Computational work on these archives has mostly sorted reports into fixed taxonomies. In this work, we propose a hybrid framework in which a text-based corpus identifies the leading factors of a failure mode. Physics-based models then quantify how likely that failure is at a stated operating point. Ordered causal chains extracted from 56,202 American accident reports with a local large language model aggregate into a causation knowledge graph. On an evaluation sample of 2,000 reports, the chains recover 90.8% of investigator-flagged causes. Each physical factor whose drivers the record carries receives an occurrence model calibrated to the accident population. A leaky noisy-OR fitted to the corpus combines the factors into a failure mode probability that supports intervention. For engine power loss, the carburetor icing model separates 1,205 icing accidents from other accidents using temperature and dewpoint alone, reaching 0.61 in area under the receiver operating characteristic curve against a density ratio ceiling of 0.64. Routine weather hours give no valid control, because flying concentrates in fair weather. On later years the physics adds 0.03 to that area for occurrence and nothing for severity, so its value lies in interpretability. On four named accidents, abduction gives a probability of necessity of 0.29 to 0.50 for the heat omission, under monotonicity. The Fast Causal Inference algorithm does not license the largest backdoor adjustment on the learned networks.

Index Terms—Aviation safety, hybrid risk model, causal inference, interventional analysis, physics-based modeling.

## I. INTRODUCTION

Aviation authorities have collected a large archive of accident investigations. Each record pairs a free-text narrative with structured codes, namely an ordered sequence of occurrences and a set of findings that investigators flag as causes or contributing factors. The narratives contain causal chains such as induction icing that leads to engine power loss, then to a forced landing, and finally to terrain impact. Recovered at scale, those chains would give safety analysts a network of canonical safety factors in which each directed edge states that one factor caused another. Such a network supports the question that matters for prevention [1], namely how much the risk of a severe outcome would have fallen had a given contributing factor been absent.

Computational work on these archives has grown quickly, yet most of it stops short of that question in two respects. The first concerns what is extracted. A large body of work sorts narratives into fixed taxonomies with domain-adapted or prompted language models [2–4]. A document label records neither the order of events nor the factor the investigator held responsible. Finer studies still classify onto code sets [5] or work on a few thousand reports [6]. The second concerns what is done with the extracted structure. Knowledge graphs built with large language models [7] and Bayesian networks fitted to coded data [8] quantify risk as a frequency over the reports. That frequency inherits the selection bias of an archive that records only reportable events. It is one scalar per factor with no mechanism behind it. Engineering practice models those mechanisms through fault trees [9], gust load formulas [10] and flight simulation [11]. On its own, however, it cannot say which failure modes dominate a fleet or how often each arises. Hybrid causal risk models join the two, but experts build their structure and they quantify one accident at a time. No aviation study learns that structure from a complete report archive and attaches mechanistic models to its factors.

The main contributions of this paper are the following.

• We recover causal structure from a complete corpus of 56,202 NTSB and FAA reports rather than from a curated sample. Extraction returns one ordered causal chain per narrative under a closed factor vocabulary, so each chain is acyclic by construction. Chains from different accidents then aggregate into the edge counts that a statistical causal analysis needs (Section III-B).

• We attach a physics occurrence model to every factor whose drivers are physical. Where the record carries those drivers, the model is calibrated so that physics supplies the conditional shape of the risk while the corpus supplies its absolute level (Sections III-C and III-E). Such a factor carries a risk that responds to its physical drivers, here temperature and dewpoint for carburetor icing, in place of one frequency for every flight.

• We give a rule that combines contributing factor risks into a failure mode probability conditioned on the operating point. A single do-operator semantics covers both backdoor adjustment on the learned graph and intervention on a physical driver. A counterfactual computed by abduction applies the same models to named accidents (Sections III-F and III-G).

• We test the framework on engine power loss (Section IV) and develop carburetor icing end to end as its worked physics factor.

• We evaluate every layer of the framework and state what the archives cannot support (Section V). The physics models fit no parameter to the corpus. The combination and the calibration links are fitted, and every fitted arm is scored out of fold or on later years. Several results are negative, and we report them as findings.

## II. RELATED WORK

Text mining of safety narratives began with domain-adapted language models used as classifiers, including SafeAeroBERT and Aviation-BERT [2, 12]. It continued with large language models for human factors coding [4, 13] and fine-level event extraction [5, 6]. Knowledge graphs assembled from extracted content map interdependencies and do not support interventional queries [7, 14]. The Bayesian network of Zhang and Mahadevan [8] is learned from manually coded reports with priors from departure statistics, and it conditions on evidence without intervening on the graph.

Causal inference on accident data is concentrated on the lower two rungs of Pearl’s ladder [1, 15, 16]. Counterfactual work that is formally sound runs on models built by experts or on a handful of cases [17–19]. Language models exploit variable names and mention order and remain unreliable on whole causal graphs [20, 21], which motivates letting them supply local ordering and typed factors while the structure and probabilities come from aggregation [22, 23].

Physics-based risk modeling quantifies failure from mechanism, through fault and event trees [9], gust loads and continuous turbulence [10, 24, 25], open flight dynamics and performance models [11, 26, 27] and operational charts [28]. Hybrid automata give these mechanisms a formal home [29, 30]. Hybrid causal risk models already combine mechanisms with accident data. Examples are hybrid causal logic [31], the causal model for air transport safety and its analysis of flight TK 1951 [32–34], dynamic probabilistic risk assessment [35] and physics-informed Bayesian networks in adjacent domains [36, 37]. They take their structure from experts and their frequencies from exposure-normalized counts, and they quantify one accident at a time.

## A. Research Gap

Corpus-driven work leaves each factor as one unconditional frequency with no mechanism behind it, so an intervention on a factor can only be a statistical adjustment. Sound counterfactual work and the hybrid causal models run on structure built by experts and quantify one accident at a time. To our knowledge, no aviation study extracts causal chains over a complete corpus, attaches physics occurrence models to the factors whose drivers are physical, and checks each reconstructed chain against the physics of the individual case.

## III. METHODOLOGY

## A. Overview and Division of Labor

The framework has two layers joined by a fusion rule, and Fig. 1 traces the flow between them. The data layer (Section III-B) turns raw reports into a causation graph and reads off, for each failure mode, its leading factors, its downstream consequences and its severity. The physics layer (Section III-C) attaches an occurrence model to any failure mode with a physical mechanism, mapping continuous physical drivers to the probability that the failure occurs. The fusion step (Section III-E) combines the physics occurrence probability with the consequence terms from the data, and calibrates the absolute level of the physical model against the corpus. Section III-D states the object that joins the two layers. The causal chain is a path over the modes of a hybrid automaton. Its transitions carry the corpus conditionals, and its guards are per-case physics margins that check physical admissibility before any probability is attached.

Not every node receives a physical model. A node qualifies when its data-learned leading factors correspond to continuous physical variables. Human and organizational factors such as inappropriate decision or procedure not followed have no such correspondence, so they remain data only. The partition of nodes into physics-modeled and data-only nodes therefore follows from the structure of the problem rather than from an analyst’s choice.

## B. Data Layer of Causal Chains and Leading Factors

a) Corpus: The corpus draws on three public archives, the NTSB aviation databases [38] with 43,604 reports, the FAA accident and incident data system [39] with 12,585 and the NTSB published investigation reports [40] with 13. It holds 56,202 reports whose narratives run between 200 and 2,000 words. Every source is free to obtain, and the download code is released with this work.

Every record carries structured supervision from its source database. A cause statement accompanies 84% of the records and a coded event sequence 47%, and 70%, all in the NTSB portion, carry findings that flag each factor as a cause or a contributing factor. Occurrence codes map to the occurrence categories of the CAST and ICAO Common Taxonomy Team (CICTT) [41], and injury and damage fields resolve severity. This supervision grounds the extraction prompt and serves as the reference standard of Section V-A.

b) Chains: Qwen3.6-35B-A3B [42, 43], a locally served mixture-of-experts model with 35 billion total and 3 billion active parameters running on vLLM [44], converts each narrative into a temporally ordered chain of factor nodes. The nodes come from a closed vocabulary of 57 values, namely

![](images/055bc03d170234cde6a55289e26e01c3e52b02f78d471d999616a5dc95f3911b.jpg)  
Fig. 1. The hybrid framework. The data layer converts 56,202 report narratives into ordered causal chains and aggregates them into a causation graph. The graph gives the leading factors $\mathcal { L } ( F )$ , the consequence conditionals $P ( Y \mid F )$ , the corpus base rates ${ \widehat { P } } ( F )$ and the per-category networks $G _ { c } .$ . The physics layer maps the leading factors that have physical drivers to continuous variables θ and evaluates an occurrence model $\pi _ { F }$ at the stated operating point. The admissibility monitor rejects any chain whose transitions the guard margins $\gamma _ { e }$ rule out at the recorded state. Calibration ties the level of the physical model to the corpus marginal, and a leaky noisy-OR fitted to the corpus combines the contributing factors of a failure mode. Interventional queries act through backdoor adjustment on $G _ { c }$ or through an intervention $d o ( m )$ that transforms the physical drivers.

35 event factors, 12 condition factors, 9 outcome types and an unknown placeholder. Every node may cite only earlier nodes as its causes, so each chain is a directed acyclic graph by construction. Formally, report r yields $\mathcal { C } _ { r } = ( v _ { r , 1 } , \ldots , v _ { r , n _ { r } } )$ with $v _ { r , k } \ \in \ \mathcal { V } , \ | \mathcal { V } | \ = \ 5 7 ,$ and a parent map $\mathrm { p a } ( k ) \subseteq$ $\{ 1 , \ldots , k - 1 \}$ . The run produced 55,940 valid chains with no schema violations, and 262 records that returned no valid or an empty chain enter no count below. We write $\mathcal { F } \subset \mathcal { V }$ for the 47 non-outcome factors and Y for the 9 outcomes.

c) Schema: The categories of the extraction schema come from established taxonomies. The condition, event and outcome classes follow the separation of latent conditions from active failures [45, 46], the event groups follow the layers of HFACS [47], and the encounters, outcomes and flight phases follow CICTT [41, 48]. Each node carries its position, class, factor type, a trigger phrase, a flight phase and an attribution role that mirrors the NTSB cause and factor flags [38]. Causality lives in a list of at most four parents per node. Each parent entry names an earlier node, grades the link as direct or contributing and quotes the supporting narrative span, so every edge of the aggregated graph traces back to checkable text. Outcome nodes may cause only other outcomes, a factor type repeated within a chain counts once in the aggregation, and constrained decoding against the JSON grammar enforces the schema with one repair turn on violation.

d) Aggregation: Chains aggregate at the factor level into a directed graph. For factors $i , j \in { \mathcal { F } }$ , let $s _ { i }$ be the number of reports whose chain contains factor i. The edge support $s _ { i j }$ counts the reports that contain at least one causal link from a node of type i to a node of type j. Each edge then carries

$$
P ( j \mid i ) = { \frac { s _ { i j } } { s _ { i } } } , \qquad \operatorname { l i f t } ( i , j ) = { \frac { P ( j \mid i ) } { P ( j ) } } = { \frac { s _ { i j } / s _ { i } } { s _ { j } / N } } ,\tag{1}
$$

where N is the number of reports. Aggregating 55,940 chains gives a compact graph of 47 factors and 434 edges with support of at least 30. The edge tables of the failure study read their statistics from the unfiltered edge list. Each report also contributes a binary presence vector $\mathbf { x } _ { r } \in \{ 0 , 1 \} ^ { 4 7 }$ and a resolved severity $Y _ { r } ,$ , where structured injury and damage fields override the model’s own estimate. These vectors form the table used in the statistical analyses below.

e) Leading Factors: The leading factors of a failure mode $F \in { \mathcal { F } }$ are the antecedents that the corpus attributes to it,

$$
\mathcal { L } ( F ) = \big \{ i \in \mathcal { F } : s _ { i F } \geq \rho s _ { F } \big \} ,\tag{2}
$$

the direct parents whose edge into F explains at least a fraction $\rho$ of the reports that carry F, with $\rho = 0 . 0 2$ held fixed across failure modes. No lift threshold enters, because lift within an accident population is subject to the selection effect described next. The failure study reports lift as a diagnostic only. The set $\mathcal { L } ( F )$ is the answer of the data layer to the question of what leads to F. We identify physical drivers within this set, and we pose interventions on its members.

f) Selection on the Accident: The corpus observes a flight only when it ended in a reportable event. Conditioning on that event makes the causes compete for the same accident. The presence of one sufficient cause then lowers the chance that another is recorded in the same report, which is a collider effect of the selection [49]. Its signature is a lift below one on an edge that is causal in the flight population. Procedure not followed enters engine failure at lift 0.24, inadequate maintenance at 0.39 and structural failure at 0.42 (Table II). The fitted combination of Section III-F gives structural failure no direct weight and keeps strengths of 0.11 and 0.35 for the other two, so a low lift alone does not decide the strength. Within-accident contrasts therefore measure competition among causes as much as causation, and the causal sufficiency that the PC algorithm assumes fails by construction. Section V-G runs FCI, which allows latent confounding and selection, alongside PC on the same data.

g) Structure per Category: For a directed analysis that accounts for confounding, we group accidents by occurrence category, read from the NTSB occurrence codes (Section V-G), and learn a causal network $G _ { c }$ per category from $\left\{ { \bf x } _ { r } \right\}$

TABLE I

The PC algorithm [50, 51] provides the skeleton under $\chi ^ { 2 }$ independence tests. Ordering priors from a language model, in the manner of Vashishtha et al. [22] and Zhang et al. [23], and empirical chain precedence then orient the edges, and bootstrap resampling filters them. The network $G _ { c }$ supplies the adjustment sets used in Section III-G.

## C. Physics Layer of Failure Occurrence Models

Let F be a failure mode with a physical mechanism and continuous physical drivers $\theta ~ \in ~ \Theta _ { F }$ . The physics layer supplies an occurrence model

$$
\pi _ { F } ( \theta ) = \mathrm { P r } ( F \mathrm { o c c u r s } | \theta ) = g _ { F } ( \theta ; \phi _ { F } ) ,\tag{3}
$$

where $g _ { F }$ follows from the governing physics and $\phi _ { F }$ collects the model parameters. The drivers θ are the physical realization of the data-learned leading factors $\mathcal { L } ( F )$ . A qualitative antecedent such as moist air becomes the recorded dewpoint, and low power becomes a power setting.

In general, only a subset O of the components of θ is recorded for a given accident. The latent remainder ξ is marginalized over a prior conditioned on flight phase,

$$
\pi _ { F } ( O ) = \int _ { \Xi } \mathbf { 1 } \{ F \mid O , \xi \} p ( \xi \mid O ) d \xi .\tag{4}
$$

No factor in this paper requires that step. Carburetor icing reads temperature and dewpoint from the record and is evaluated at cruise power and at descent power separately.

The corpus enters Eq. (3) in three ways. The choice of $g _ { F }$ follows from the mechanism that the corpus identifies. The prior $p ( \xi \mid O )$ takes its shape from the observed distribution of power settings, phases and aircraft classes in accidents of that mode. The parameters $\phi _ { F }$ are anchored to the corpus by the calibration described next.

## D. The Chain Automaton and Admissibility Monitoring

A causal chain is a walk through discrete failure states, and each occurrence model of Section III-C decides from the continuous state of a flight whether a step was physically available. We take the hybrid automaton [29, 30], keep its modes and guards and drop the flows, which a narrative rarely supports. The modes are the factors and outcomes of Section III-B, and the transitions are the edges of the causation graph. A guarded transition e carries a guard margin $\gamma _ { e } ( \theta )$ , a signed quantity with physical units whose sign is oriented so that e is enabled exactly when $\gamma _ { e } ( \theta ) \geq 0 .$ Table I lists the guards of the failure study. Each is a quantity its occurrence model already computes, so the guards add no new physics. A transition is checkable for a case when it carries a guard and the record carries the drivers of that guard. The admissibility monitor rejects a chain when a checkable transition has $\gamma _ { e } ( O ) < 0$ at the recorded observables O. The chain then requires a step the physics rules out, so either the extraction is wrong or a mode is missing. No reweighting of a probabilistic blend reproduces such a rejection [52]. The verdict is invariant under any strictly increasing transformation of $\gamma _ { e } ,$ so it needs no calibration. Two corpus statistics follow. The guard coverage is the share of transitions that are checkable, and the violation rate is the share of checkable transitions that the monitor rejects. A corrupted guard control, with sign-inverted margins or the weather of an arbitrary accident, separates extraction error from model inadequacy. For a failure onset the guard condition is the occurrence event itself. Carburetor icing maps its margin through the monotone link that Eq. (5) calibrates, so rankings across operating points come from the guard alone. The monitor checks single transitions at recorded states and computes no reachability over continuous trajectories.

GUARD MARGINS BEHIND THE PHYSICS-MODELED TRANSITIONS OF THEFAILURE STUDY. SIGNS ARE ORIENTED SO THE TRANSITION IS ENABLEDWHEN THE MARGIN IS NONNEGATIVE. THE TIME t IS THE ELAPSED ORPLANNED FLIGHT TIME $\mathbf { A N D } ~ t _ { \mathrm { e n d } }$ THE FUEL ENDURANCE. THECONDENSABLE WATER w IS EVALUATED AT TEMPERATURE T , DEWPOINT$T _ { d }$ AND POWER SETTING ϖ, WITH THRESHOLD $w _ { \mathrm { t h r } } ,$ AND ITS MARGIN ISTHAT DIFFERENCE EXPRESSED AS ABSOLUTE HUMIDITY AT THETHROTTLE PLATE TEMPERATURE.
<table><tr><td>Transition into</td><td>Guard margin  $\gamma _ { e } ( \theta )$ </td><td>Units</td></tr><tr><td>Fuel exhaustion</td><td> $t - t _ { \mathrm { e n d } }$ </td><td>hours</td></tr><tr><td>Carburetor icing</td><td> $w ( T , T _ { d } , \varpi ) - w _ { \mathrm { t h r } }$ </td><td>grams per cubic meter</td></tr></table>

## E. Fusion and Tiered Calibration

The monitor decides which transitions are available for a case, and the fusion layer attaches probabilities to the available ones. Each node draws its occurrence probability from the best available source, organized as a tier. Tier 1 is a physics model, Eq. (3). Tier 2 is published failure rate data. Tier 3 is the corpus marginal ${ \widehat { P } } ( F )$ . Tier 4 is expert or model elicitation. Tier 3 forms the backbone, because it is defined for every factor in the corpus, while Tier 1 sharpens the nodes that have a mechanism.

Where a physics estimate and a corpus estimate both exist, they occupy different probability spaces. The corpus frequency is conditioned on a reportable accident having occurred, whereas $\pi _ { F }$ is a per-exposure probability with no population behind it. We therefore treat physics as the source of the conditional shape and the corpus as the source of the absolute level. Calibration fixes the level so that the physical model reproduces the corpus marginal once averaged over the observed operating point distribution D of the accident population,

$$
{ \mathbb { E } _ { O \sim \mathcal { D } } \big [ \pi _ { F } ( O ; \phi _ { F } ) \big ] \ = \ \widehat { P } ( F ) . }\tag{5}
$$

This step keeps the relative sensitivity that makes interventions informative and puts the physics term on the population the corpus observes. Section $\mathrm { V } { - } \mathrm { C }$ executes it for carburetor icing as a monotone link on the physics score. The fuel term cannot be calibrated, since fuel on board and flight time are not coded fields. Every quantity then has a stated reference class. The corpus base rates, the edge conditionals, the fitted combination of Section III-F and the calibrated physics terms refer to a reportable accident, the guard verdicts to no population, and the uncalibrated chart probability of Eq. (14) and the fuel term to an exposure, so the failure study reports the effect of the last two as ratios.

Consider a physics-modeled failure F and a downstream event or outcome Y that is reachable from F in the graph. The corpus supplies the consequence conditional $P ( Y \mid F )$ , either as a single edge from Eq. (1) or as a product along a path. The fused risk that F produces Y at operating point O is then

$$
R _ { F } ( Y \mid O ) = \underbrace { \pi _ { F } ( O ) } _ { \mathrm { p h y s i c s } } \cdot \underbrace { P ( Y \mid F ) } _ { \mathrm { d a t a } } .\tag{6}
$$

Here the physics term states whether the failure will occur under the stated conditions, while the corpus supplies how often it goes on to produce Y. Equation (6) quantifies a chain of depth two, one guarded onset followed by one consequence transition. The failure study quantifies chains of this depth, which keeps every reported number attributable to one named mechanism and one corpus conditional. Uncertainty propagates by carrying tier-specific distributions on both terms and sampling from them, which yields a confidence interval on $R _ { F }$

## F. Combining Contributing Factors

Equation (6) quantifies a single failure. A higher-level failure event Y , such as engine failure, is reached through several contributing factors. The data layer identifies them as its causal parents $\operatorname { p a } ( Y ) = \{ X _ { 1 } , . . . , X _ { k } \}$ , the leading factor set of Eq. (2). Each contributing factor carries an occurrence probability $\pi _ { i }$ , taken from its calibrated physics model where one exists and otherwise from the corpus base rate ${ \widehat { P } } ( X _ { i } )$ ). It also carries a strength $p _ { i }$ , the probability that the factor alone produces Y. We treat the contributing factors as independent causes of the top event, as in a noisy-OR or bow-tie model, with a leak ℓ for the causes the set omits. The failure mode probability then becomes

$$
P ( Y \mid O ) = 1 - ( 1 - \ell ) \prod _ { i = 1 } ^ { k } \bigl ( 1 - \pi _ { i } ( O ) p _ { i } \bigr ) ,\tag{7}
$$

and the contribution of factor i is the drop that follows when the factor is removed,

$$
\Delta _ { i } = P ( Y \mid O ) - P \big ( Y \mid O , \pi _ { i } { = } 0 \big ) .\tag{8}
$$

We fit the leak and the strengths by maximum likelihood on the binary presence table of Section III-B, with one row per report. The strength $p _ { i }$ therefore carries the noisy-OR meaning of a cause acting alone. It differs from the marginal conditional $q _ { i } = P ( Y \mid X _ { i } )$ of Eq. (1), which counts the work of the other parents as well. The leak absorbs the causes that the leading factor set does not list, and the mean of the fitted probabilities over the records reproduces the prevalence of $Y$ Evaluated at the corpus base rates, Eq. (7) gives instead the population value, which differs slightly because the parents are not independent in the presence table. A parent whose within-accident lift is below one often receives a strength near zero, for one of two reasons. Either its effect runs through a parent already in the set, or its association with $Y$ reflects the selection on the accident described in Section III-B. Equation (8) corresponds to an intervention that removes $X _ { i }$ at the level of the failure mode. The physics-modeled factors supply $\pi _ { i } ( O )$ as a function of the operating point, so $P ( \boldsymbol { Y } \mid O )$ depends on the scenario. The data-only factors enter at their corpus base rates. Equation (7) is the one-level combination, with parent activations $\pi _ { i } ( O )$ and strengths $p _ { i }$ entering unmonitored where no consequence guard applies. It gives an interpretable structural model.

## G. Interventional and Counterfactual Analysis

The framework answers interventional queries through the do-operator in two forms. The form depends on whether the target of the intervention is a categorical factor or a physical driver. The framework also answers one counterfactual query by abduction on the guarded mechanisms.

a) Graph-Level Intervention: Consider a categorical factor $X \in { \mathcal { L } } ( F )$ that has no physical model, and a binary severe outcome variable Y in category c. Comparing accidents with and without X is confounded, because X occurs together with its own causes. We therefore apply backdoor adjustment on the learned network and stratify on the parents of $X$ , written $Z = \operatorname { p a } _ { G _ { c } } ( X )$

$$
P ( Y \mid d o ( X { = } x ) ) = \sum _ { z } P ( Y \mid X { = } x , Z { = } z ) P ( Z { = } z ) ,\tag{9}
$$

with Laplace-smoothed stratum estimates and effect $\tau _ { X } ~ =$ $P ( Y \mid d o ( X { = } 1 ) ) - P ( Y \mid d o ( X { = } 0 ) )$

b) Physics-Level Intervention: For a failure mode that carries a physics model, an intervention acts on a physical driver, or on a mitigation m that maps to a driver transform $\mathcal { T } _ { m } : \theta \mapsto \theta ^ { \prime }$ . We then reevaluate the occurrence model at the intervened value,

$$
\begin{array} { r l } & { R _ { F } ( Y \mid d o ( m ) , O ) = \pi _ { F } ( \mathcal { T } _ { m } ( \theta ) ) \cdot P ( Y \mid F ) , } \\ & { ~ \Delta R = R _ { F } ( Y \mid O ) - R _ { F } ( Y \mid d o ( m ) , O ) . } \end{array}\tag{10}
$$

The intervention propagates through a mechanistic model, so ∆R is attributable to a named physical cause and monotone in the driver, and the monitor reevaluates $\gamma _ { e } ( \mathcal { T } _ { m } ( \theta ) )$ as well. Equations (9) and (10) estimate the same reduction in severe outcome probability, statistically where only categorical evidence exists and mechanistically where a physical model is available.

c) Counterfactual on a Named Accident: Both forms are interventional, the second rung of Pearl’s ladder. A counterfactual, the third rung, asks what would have happened in a particular accident had one of its causes been different. It needs abduction, action and prediction on a structural model with explicit noise. The guarded mechanisms of Section III-D supply such a model. For carburetor icing the structural equation is $\mathrm { I C E } = { \bf 1 } \{ U \le g ( w ) \}$ }, with w the condensable water at the recorded temperature and dewpoint, g the calibrated link of Eq. (5) and U uniform on [0, 1]. Abduction restricts U to the values consistent with what the record shows. The action $d o ( \mathrm { h e a t } )$ warms the intake so that the condensable water becomes $w ^ { \prime }$ , and prediction pushes the same $U$ through the modified equation. Under this monotone coupling the probability that icing would not have occurred had heat been applied, given that it did occur, is

$$
\mathrm { P N } _ { \mathrm { i c e } } = 1 - { \frac { g ( w ^ { \prime } ) } { g ( w ) } } ,\tag{11}
$$

the probability of necessity of the heat omission [53]. The engine failure node is the leaky noisy-OR of $\operatorname { E q . } \ ( 7 ) .$ , so the same three steps give the probability of necessity for the power loss. The other fitted parents enter at their base rates and stand in for the causes the record does not name. Section IV-C5 reports the quantity on four named accidents together with the bounds that hold without the monotonicity assumption.

## IV. FAILURE STUDY ON ENGINE POWER LOSS

The failure study applies the machinery of Section III to engine failure, the most frequent failure node of the causation graph with 16,406 records. It develops carburetor icing, the dominant weather-driven parent of engine failure, end to end.

## A. Contributing Factors

Table II lists the nine leading factors of $\operatorname { E q } . \ ( 2 )$ , the parents with edge support of at least 329. Engine failure suits the hybrid treatment well, because its dominant contributing factors are fuel state and induction mechanisms that carry the strongest causal edges in the graph. Fuel exhaustion or starvation leads with support 5,051, $P ( \mathrm { E F } \mid \mathrm { f u e l } ) = 0 . 9 3$ and lift 3.18, which is the strongest single edge in the causation graph. Fuel contamination follows at lift 2.57 and carburetor icing at lift 2.27. The set covers most chains. Of the 16,406 records with an engine failure node, 77% give that node a member of the set as an explicit parent, 6% give it only parents outside the set and 17% give it no parent. The leak of Section IV-D therefore stands mostly for power losses that the narrative states without a cause. Table II feeds the combination rule. Structural failure is a parent because, in about nine of ten records behind this edge, it names a fractured powerplant part, such as a propeller blade, cylinder head or connecting rod. Section V-G adjusts its effect on a severe outcome within the airframe failure category, which supplies no adjustment set to the combination of Section IV-D.

## B. Physics of Fuel Exhaustion

The dominant factor has transparent physics, but the combination uses its model only for one ratio. Exhaustion occurs when the flight outlasts the endurance $E = m _ { f } / \dot { m }$ of the usable fuel mass $m _ { f }$ at fuel flow m˙ . With reserve $r =$ $E - t _ { \mathrm { f l i g h t } }$ and a Gaussian margin of scale $\sigma = c t _ { \mathrm { f l i g h t } }$ for the uncertainty in fuel flow and routing, the exhaustion probability is $\pi _ { \mathrm { f u e l } } ( r ) = \Phi ( - r / \sigma )$ . With $c = 0 . 2 5$ on a 3 h flight it is $0 . 2 5$ at a reserve of 0.5 h and 0.004 at 2 h. Fuel on board and flight time are not coded fields, so Eq. (5) cannot calibrate the term, and the combination takes this factor at its corpus base rate. The remaining factors use the carburetor icing physics of Section IV-C1, component reliability surrogates for the fuel system and contamination, and corpus rates for defect, maintenance and procedure. Standard performance models [26, 27] supply the glide and forced landing consequences downstream of the failure.

## C. Carburetor Icing as the Worked Physics Factor

We now develop one contributing factor end to end. Carburetor icing is the dominant weather-driven parent of engine failure, and its edge into engine failure carries support 971 in the graph. It suits the role of worked physics factor for three reasons. Its physical drivers are recorded for nearly every report, and the failure is a published function of those drivers. The power setting, the one driver the record lacks, can be bracketed by evaluating the model at cruise and at descent power. A stall, by contrast, depends on airspeed and load factor that no record keeps.

1) Physics of Induction Icing: In a float carburetor, the static pressure drop across the venturi and the vaporization of fuel can lower the temperature substantially where the inducted air and fuel mix before combustion. If the mixture cools below its dewpoint and below $0 ~ ^ { \circ } \mathbf { C } ,$ , moisture condenses and freezes on the throttle plate, which restricts the induction airflow and reduces power. Let $T$ and $T _ { d }$ be the ambient temperature and dewpoint in degrees Celsius. Saturation vapor pressure follows the Magnus form [54],

$$
e _ { s } ( T ) = 6 . 1 1 2 \exp \left( { \frac { 1 7 . 6 2 T } { 2 4 3 . 1 2 + T } } \right) \mathrm { \ h P a } , \qquad \mathrm { R H } = { \frac { e _ { s } ( T _ { d } ) } { e _ { s } ( T ) } } \cdot
$$

The mixture cools to a throttle plate temperature $T _ { \mathrm { t h r } } \ =$ $T - \Delta T ( \varpi )$ . The cooling ∆T depends on the power setting ϖ and is largest at low power, because a nearly closed throttle produces the greatest pressure drop. The carburetor icing probability chart of FAA Special Airworthiness Information Bulletin CE-09-35 [28] carries separate zones for serious icing at descent power for this reason. The chart attaches no percentage of rated power to its settings, and these settings vary with the aircraft type and the pilot. The model therefore represents each setting by its cooling alone, $2 5 ~ ^ { \circ } \mathrm { C }$ at cruise power and $3 2 ~ ^ { \circ } \mathrm { C }$ at the reduced power of a descent or glide. The moisture that condenses in the subzero region, and therefore freezes, is proportional to the vapor pressure difference

$$
\begin{array} { r } { w ( T , T _ { d } , \varpi ) = \operatorname* { m a x } \big ( 0 , e _ { s } ( \operatorname* { m i n } ( T _ { d } , 0 ) ) - e _ { s } ( T _ { \mathrm { t h r } } ) \big ) \big ( \frac { } { } \Pi \big ) } \\ { \cdot { \bf 1 } \{ T _ { \mathrm { t h r } } < 0 \} , } \end{array}\tag{13}
$$

and the occurrence probability is a saturating exposure function of that quantity,

$$
\pi _ { \mathrm { i c e } } ( T , T _ { d } , \varpi ) = 1 - \exp \left( - \kappa w ( T , T _ { d } , \varpi ) \right) .\tag{14}
$$

The quantity w carries pressure units, so κ is an exposure constant per hectopascal that absorbs the induction mass flow and the exposure time. The guard margin of Table I converts the same difference to an absolute humidity at the throttle plate temperature, 216.68 $( w - w _ { \mathrm { t h r } } ) / ( T _ { \mathrm { t h r } } + 2 7 3 . 1 5 )$ in grams per cubic meter. Here $w _ { \mathrm { t h r } }$ is the onset of the light zone of the chart. The two parameters $\phi = \{ \Delta T ( \varpi ) , \kappa \}$ are the free constants of the physical model. We set them so that Eq. (14) reproduces the named zones of that chart. This gives $\kappa = 0 . 6 0$ per hectopascal and $w _ { \mathrm { t h r } } = 0 . 1 7 6$ hPa, together with the two cooling values above. The chart, and not the corpus, is therefore the empirical content behind these two constants. The calibration of Eq. (5) is a separate step that

TABLE II  
LEADING FACTORS OF ENGINE FAILURE UNDER THE RULE OF EQ. (2), WITH THEIR EDGE STATISTICS, CORPUS BASE RATE, ASSIGNED OCCURRENCE MODEL, FITTED STRENGTH p AND CONTRIBUTION ∆ TO P(EF). THE EDGE SUPPORT COUNTS THE REPORTS WITH A CAUSAL LINK FROM THE FACTOR INTO ENGINE FAILURE. INTERVALS ARE 95% BOOTSTRAP PERCENTILES OVER 500 REFITS OF THE COMBINATION.
<table><tr><td>Factor</td><td>Edge support</td><td>qi</td><td>Lift</td><td>Base rate</td><td>Model</td><td> $p _ { i }$ </td><td> $\Delta _ { i }$ </td><td>95% CI</td></tr><tr><td>Fuel exhaustion or starvation</td><td>5051</td><td>0.93</td><td>3.18</td><td>0.097</td><td>physics (endurance)†</td><td>0.96</td><td>0.071</td><td>(0.070, 0.073)</td></tr><tr><td>Latent mechanical defect</td><td>2331</td><td>0.28</td><td>0.95</td><td>0.149</td><td>data</td><td>0.19</td><td>0.021</td><td>(0.019, 0.022)</td></tr><tr><td>Procedure not followed</td><td>1137</td><td>0.07</td><td>0.24</td><td>0.293</td><td>data</td><td>0.11</td><td>0.023</td><td>(0.021, 0.025)</td></tr><tr><td>Other system failure</td><td>1035</td><td>0.28</td><td>0.97</td><td>0.065</td><td>data</td><td>0.20</td><td>0.009</td><td>(0.008, 0.010)</td></tr><tr><td>Carburetor or induction icing</td><td>971</td><td>0.67</td><td>2.27</td><td>0.026</td><td>physics (icing)</td><td>0.93</td><td>0.017</td><td>(0.016, 0.018)</td></tr><tr><td>Fuel system anomaly</td><td>954</td><td>0.57</td><td>1.94</td><td>0.030</td><td>surrogate</td><td>0.63</td><td>0.013</td><td>(0.012, 0.014)</td></tr><tr><td>Fuel contamination</td><td>826</td><td>0.75</td><td>2.57</td><td>0.020</td><td>surrogate</td><td>0.92</td><td>0.013</td><td>(0.012, 0.014)</td></tr><tr><td>Inadequate maintenance</td><td>649</td><td>0.12</td><td>0.39</td><td>0.101</td><td>data</td><td>0.35</td><td>0.025</td><td>(0.024, 0.027)</td></tr><tr><td>Structural failure</td><td>333</td><td>0.12</td><td>0.42</td><td>0.049</td><td>data</td><td>0.00</td><td>0.000</td><td>(0.000, 0.001)</td></tr></table>

<sup>†</sup>The endurance model is not calibrated, since fuel on board and flight time are not coded fields. The factor enters the combination at its corpus base rate.

Section V-C executes on the accident population. The resulting risk surface (Fig. 2) recovers the less obvious features of the chart, including serious icing at warm ambient temperatures when humidity is high.

2) Data Support: The physical drivers are recorded for 79.9% of the 93,360 NTSB event records once missing readings coded as zero and impossible values are discarded. Carburetor icing is a well supported cause of engine failure, with an edge of support 971. Carburetor icing accidents carry P(serious or fatal) = 0.23, below the corpus baseline of 0.26, because induction icing events usually end in survivable forced landings.

To test the physics against the data, we joined the model to the 39,770 extracted NTSB accidents with usable weather and asked whether the 1,205 that the narratives attribute to carburetor icing fall in higher-risk weather. They do. At descent power the physics assigns them a mean occurrence probability of 0.76 against 0.63 for the other accidents, and 88.9% of them fall in the moderate or serious zone against 75.0%. As a score, $\pi _ { \mathrm { i c e } }$ separates the two groups with an area under the ROC curve of 0.61 (Fig. 2), close to the density ratio ceiling of 0.64 that Section V-D estimates. Conducive weather is a necessary condition, and whether ice brings the aircraft down depends on carburetor heat use and engine susceptibility, whose realized rate the corpus edge P(engine failure | icing) = 0.67 supplies.

3) Causal Chain to Icing: The causal edges of the carburetor icing node show two features. The consequence is pinned down sharply, since the edge to engine failure carries support 971 with P(engine failure | icing) = 0.67 and lift 2.27. The antecedents are sparse and weak. Only procedure not followed, which in these chains records carburetor heat applied late or not at all, meets the rule of Eq. (2). In causal terms carburetor icing behaves as a root condition. Its true antecedents are the continuous physical drivers $\theta \ = \ ( T , T _ { d } , \varpi )$ , which are not nodes in the categorical vocabulary. The occurrence model $\pi _ { \mathrm { i c e } }$ supplies the missing antecedent quantification, so carburetor icing enters the engine failure combination of Section IV-D as a calibrated physics parent.

4) Intervention Results: Consider a representative carburetor icing scenario at $T = 1 3 \ ^ { \circ } \mathrm { C }$ and $T _ { d } = 1 2 ~ ^ { \circ } \mathrm { C }$ at descent power, warm and humid air that poses no airframe icing hazard. The chart physics gives $\pi _ { \mathrm { i c e } } = 0 . 9 4$ , in the serious zone of the chart. The calibrated link of Section V-C maps that operating point to 0.038 on the corpus population, against a base rate of 0.026. The fused risk of a power loss induced by carburetor ice is then the product of 0.038 and the corpus edge 0.67, or $R _ { \mathrm { i c e } } = 0 . 0 2 5$ . Two physics-level interventions follow from Eq. (10). Carburetor heat is modeled as restoring $3 0 ~ ^ { \circ } \mathrm { C }$ to the intake, so that $T _ { \mathrm { t h r } }$ rises above freezing. It brings the modeled condensable water to zero and takes the calibrated term to 0.008, the floor of the link, so $R _ { \mathrm { i c e } }$ falls by a factor of 4.8. The model applies heat before ice forms, and late or insufficient heat leaves part of the risk in place, as the coldest case of Section IV-C5 shows. A drier atmosphere, with $T _ { d }$ falling from $1 2 \mathrm { t o } - 5 \mathrm { ~ } ^ { \circ } \mathrm { C } ,$ , attenuates the risk without removing it, taking the calibrated term to 0.028 and $R _ { \mathrm { i c e } }$ down by 27%. Each reduction traces to a named driver, and the carburetor heat case is directly actionable in flight.

![](images/8b695acee26858bdb5c5996d167128c44a0c8d014ec590dc4b85663d566b4567.jpg)  
Fig. 2. Physics occurrence surface $\pi _ { \mathrm { i c e } } ( T , T _ { d } )$ at descent power, from Eqs. (12) to (14). Dashed and solid contours mark the moderate value of 0.40 and the serious value of 0.80 from the published induction icing chart. The overlaid points are the 1,205 NTSB accidents that the narratives attribute to carburetor icing. The blue contours enclose 50 and 80% of the 38,565 other weather-recorded accidents, so the figure shows separation rather than concentration alone. The area under the ROC curve between the two groups is 0.61.

5) Probability of Necessity on Named Accidents: Table III applies Eq. (11) to four accidents whose chains carry carburetor icing, a delayed or omitted application of carburetor heat, and a power loss. In each case the NTSB probable cause names carburetor icing or the late use of carburetor heat, and the investigators judged the weather conducive to it. Given that carburetor icing occurred, the probability that it would not have occurred under carburetor heat is 0.71 in the three warmer cases and 0.41 in the coldest. In the warmer cases heat removes the condensable water, and every operating point without it takes one value of the link, the floor of 0.008, which is the carburetor icing share of the 6,535 weather-recorded accidents with no condensable water. Their recorded weather also falls on one step of the link, whose top is nearly flat across the moderate and serious zones. The three cases therefore receive the same value. In 20141208X13517, at $- 1 2 . 2 \ ^ { \circ } \mathrm { C } ,$ the warmed throttle plate stays below freezing and the chart term falls only from 0.73 to 0.14. The counterfactual therefore separates the cases by whether heat clears the ice, while the link cannot separate their weather. For the power loss, with carburetor icing observed and the other fitted parents of engine failure at their base rates, the probability of necessity is 0.50 in the warmer cases and 0.29 in the coldest. Taking only the weather and the power loss as evidence lowers it to between 0.02 and 0.04, since fuel, mechanical and procedural causes compete with carburetor icing. The bounds that hold without monotonicity run from that lower value to one, so the informative content of these numbers rests entirely on the monotone coupling of Eq. (11). The computation is the third rung of the ladder on a named case, and it inherits the reference class of the calibrated link and the fitted combination.

TABLE III  
PROBABILITY OF NECESSITY OF THE CARBURETOR HEAT OMISSION ON FOUR NAMED NTSB ACCIDENTS, FROM EQ. (11). THE LINK VALUES $g ( w )$ AND $g ( w ^ { \prime } )$ ARE THE CALIBRATED CARBURETOR ICING TERMS ON THE CORPUS DENOMINATOR, AT THE RECORDED WEATHER AND WITH THE INTAKE WARMED BY $3 0 ~ ^ { \circ } \mathbf { C } .$ . THE LAST TWO COLUMNS GIVE THE PROBABILITY OF NECESSITY FOR THE POWER LOSS WITH CARBURETOR ICING OBSERVED, AND THE BOUNDS THAT HOLD WITHOUT MONOTONICITY.
<table><tr><td>Record</td><td> $T \ ( ^ { \circ } \mathrm { C } )$ </td><td> $T _ { d } ~ ( ^ { \circ } \mathrm { C } )$ </td><td>g(w)</td><td> $g ( w ^ { \prime } )$ </td><td> $\mathrm { P N } _ { \mathrm { i c e } }$ </td><td> $\mathrm { P N } _ { \mathrm { E F } }$ </td><td>Bounds</td></tr><tr><td>20001212X19636</td><td>22.2</td><td>18.9</td><td>0.028</td><td>0.008</td><td>0.71</td><td>0.50</td><td>(0.04, 1)</td></tr><tr><td>20141208X13517</td><td>-12.2</td><td>-12.8</td><td>0.025</td><td>0.015</td><td>0.41</td><td>0.29</td><td>(0.02, 1)</td></tr><tr><td>20020225X00252</td><td>2.8</td><td>-6.1</td><td>0.028</td><td>0.008</td><td>0.71</td><td>0.50</td><td>(0.04, 1)</td></tr><tr><td>20200427X62554</td><td>13.9</td><td>-2.8</td><td>0.028</td><td>0.008</td><td>0.71</td><td>0.50</td><td>(0.04, 1)</td></tr></table>

## D. Combining the Factors

The fitted combination gives a leak of ℓ = 0.084. The mean of the fitted probabilities over the records is 0.292, against an empirical prevalence of 0.293. Evaluated at the base rates, Eq. (7) gives the population value $P ( \mathrm { E F } ) ~ = ~ 0 . 3 0 0 .$ , with a 95% interval of (0.296, 0.304). It sits slightly above the prevalence because the product form ignores the dependence among the parents, and it is the reference for the scenarios below. Figure 3a shows the decomposition. Fuel exhaustion dominates at 0.071 and carries a physics model, and inadequate maintenance, procedure not followed and latent defect follow at 0.021 to 0.025. Conditioning on an operating point makes the risk specific to the scenario (Fig. 3b). Consider weather in the serious zone of the carburetor icing chart, at

$T = 1 3 ~ ^ { \circ } \mathrm { C }$ and $T _ { d } = 1 2 ~ ^ { \circ } \mathrm { C }$ at descent power, where the calibrated carburetor icing term is 0.038. This weather raises the risk to 0.307, a factor of 1.03, carburetor heat lowers it to 0.288 and the drier air to 0.301. The effect is small on the accident population because an accident in such weather carries the carburetor icing factor 1.45 times as often as an average accident. The chart probability of 0.94, read as a level, would have implied 36 times. The fuel term stays out of these numbers, since it is a per-exposure probability and its strength of 0.96 was fitted on accidents. On its own scale, a reserve of 2 h on a 3 h flight gives $\pi _ { \mathrm { f u e l } } = 0 . 0 0 4$ against 0.25 for a reserve of 0.5 h, and we report the intervention only as that ratio. Interventions on different parents also combine in Eq. (7) as independent causes by construction, so the framework does not test composition beyond that property.

## V. EVALUATION AND ROBUSTNESS

## A. Fidelity of the Extracted Chains

We score the extraction against the NTSB coded fields on an evaluation sample of 2,000 reports. We draw the sample from the NTSB records that carry cause and factor flags, the only per-factor reference the archives provide. A judge model matches each cause-flagged finding to the extracted nodes that state the same factor, allowing paraphrase. The judge is the same Qwen3.6-35B-A3B that performs the extraction, run with constrained JSON output and no reasoning trace. No fixed crosswalk from the vocabulary to the NTSB codes exists or is used, and the judge prompt and every match are released with the code. A domain-literate annotator, blind to the judge, repeated the matching on 50 of these reports. Annotator and judge agree on 85 of the 88 findings (96.6%), and in all three disagreements the annotator accepted a match that the judge had rejected. Agreement over the 580 finding and node pairs is 75%, because the annotator also accepted nodes closely related to a finding. On this sample the judge is stricter than the human reader. Recall of cause-flagged factors is 90.8% from the narrative alone and 93.1% when the coded findings are supplied in context. We cite the narrative-only figure for the method, since the grounded configuration sees the answer it is scored against. Of the 3,481 scored findings, 3,450 come from records before 2008, which we group by subject keywords. Recall is 0.96, 0.95, 0.96 and 0.93 for the 604 aircraft, 284 environment, 183 organizational and 1,581 personnel findings and 0.79 for the 798 the keywords do not classify, 3,133 of 3,450 in all. The reference therefore describes accidents before 2008, while the corpus runs to 2026 and drifts over that span (Section V-F). The 90.8% holds for the older records and is assumed for the later ones. The FAA records are extracted but not scored this way, so the measured fidelity speaks for the NTSB portion and is assumed rather than shown for the rest.

Two features of the gold standard bound these numbers. First, the reference is a set of cause flags, whereas a chain also carries intermediates and outcomes, so we score precision by role. Of the nodes marked as primary causes 62% match a cause flag, of those marked contributing 12%, or 41% over both roles, and 63% of event nodes match a coded occurrence on the 1,893 records that carry one. The gap follows the

![](images/e69bd89c4d4c295199cf48b74e1f80deafde8ca8062c201e3bb6d2f0381ca176.jpg)

![](images/8ebaca98fe56753b5afe82399fa1326cfec70d6bbf36cc0b1988d8e31dffae98.jpg)  
Fig. 3. Engine failure risk assembled from the contributing factor models. (a) Per-factor contribution to P(EF) by model tier, covering physics, surrogate and data-only factors. Fuel exhaustion is the dominant contributor. Its endurance model is not calibrated, so the factor enters at its corpus base rate (hatched) (b) The combined P(EF) conditioned on the operating point, with every term on the accident population. The vertical axis is broken between 0 and 0.28. Weather in the serious zone of the carburetor icing chart raises the population value of 0.300 to 0.307. The two interventions act on that scenario, carburetor heat lowering the risk to 0.288 and drier air to 0.301. Error bars are 95% bootstrap intervals over 500 refits of the corpus terms, which refit the leak and the strengths on resamples of the records. They cover the corpus terms only.

NTSB flags. Before 2008 contributing items carry a factor flag, which the reference omits. Against those 2,385 entries 39% of contributing and 13% of primary nodes match. Against the cause and factor flags together the figures are 47% and 69%, with the judge rerun in FP8 agreeing on 97.9% of the cause flags. The all-node figure of 29.8% mostly measures chain completeness against a sparse reference.

Second, the coded event sequence follows a coding convention within an occurrence, with the proximate subject first, and is chronological between occurrences. Agreement with it is $\tau = 0 . 3 1$ over all ordered pairs and 0.82 over cross-occurrence pairs, with 83% of reports at or above 0.75.

Two further checks support these readings. On 1,500 Transportation Safety Board of Canada summaries, whose coded sequence is chronological throughout, the all-pairs agreement is $\tau = 0 . 6 6$ . On 157 full investigation reports, three capability tiers of a second model family cover 94.4% to 78.7% of the investigators’ causal elements, while the share of supported nodes stays near 93%. Two of these documents are in the corpus, and 104 of their 151 accidents appear in it as database narratives, so the check covers a second document type about largely the same accidents rather than held out ones.

The extraction is also stable under repetition. Three further extractions of the 2,000 evaluation reports at the production temperature of 0.2 give recall between 0.906 and 0.911. The factor sets of a record agree between runs with a mean Jaccard index of 0.80, and the first primary cause node agrees in 85% of records. A temperature of 0.7 keeps recall at 0.907 and lowers the agreement to 0.72. The temperature therefore governs which secondary factors a chain names and leaves the cause-flagged content in place.

## B. Physical Admissibility of the Extracted Chains

We ran the monitor of Section III-D over the 55,940 extracted chains, reading each case’s temperature and dewpoint from the coded record. The coverage is small. Of 334,611 transitions, 1,222 (0.37%) are carburetor icing onsets with recorded weather, which the guard can check. The fuel guard checks nothing, because fuel on board and flight time are not coded fields, and most transitions concern human, organizational and procedural factors that no physics guard covers.

The carburetor icing guard rejects 2.3% of the 1,222 transitions it checks. The rate rises to 27.7% with inverted margin signs and to 4.0% with driver tuples permuted across the guarded work list. It rises to 15.5% when each chain is paired with the weather of an arbitrary weather-recorded accident. Against that null, the 28 rejections under the true pairing have a binomial tail below $1 0 ^ { - 5 \mathrm { { 0 } } }$ . The verdicts therefore depend on the pairing of chain and weather, and the 2.3% the monitor rejects are candidates for extraction error.

## C. Calibration of the Physics Layer

We executed $\operatorname { E q . }$ (5) for every physics factor of the failure study whose drivers the coded record carries. For carburetor icing the link is an isotonic regression of the extracted carburetor icing label on the condensable water w of Eq. (13). Every operating point without condensable water therefore takes one value, the floor of the link. The link is fitted on the 39,770 NTSB accidents with usable weather and scored out-of-fold over five folds. It reproduces their prevalence of 0.030 by construction and reaches an out-of-fold ROC area of 0.609. The combination of Eq. (7) uses the corpus denominator of 55,940 records, on which the carburetor icing base rate is 0.026. The link therefore enters it as a relative risk times the corpus base rate, so the worked operating point of Section IV-C4 receives $0 . 0 4 4 \times 0 . 0 2 6 / 0 . 0 3 0 = 0 . 0 3 8$ The Brier score falls from 0.50 for the chart probability to 0.0292, against 0.0294 for the base rate. The link is thus barely sharper than the base rate. Its value lies in the ordering and in correcting a level that the chart, a per-exposure zone label, overstates by a factor of 20 to 100. The fuel term is not calibrated, for want of coded fuel and flight time fields, and the failure study labels it as such.

## D. How Much Separation the Recorded Weather Can Support

An ROC area of 0.61 may be weak, or it may be close to the limit of two weather fields. To decide, we fitted supervised models to the same label from the same inputs and read off their cross-validated performance.

Scored out-of-fold over five folds on the 1,205 carburetor icing accidents and 38,565 others, a logistic regression on temperature and dewpoint reaches 0.642 (0.628, 0.656), and adding the dewpoint spread, relative humidity and quadratic terms leaves it at 0.642. Gradient boosting on the same two inputs reaches 0.623, so no nonlinear structure remains to recover. Because the input is two dimensional, the Bayesoptimal separation can also be estimated directly. It is the ROC area of the out-of-fold density ratio of the two classes from kernel density estimates. This gives 0.640 with a 95% interval of (0.626, 0.655) at a $3 ~ ^ { \circ } \mathrm { C }$ bandwidth and 0.638 at $2 ~ ^ { \circ } \mathbf { C } .$ Against that reference the physics model attains 0.612 with a DeLong interval of (0.597, 0.628) while fitting no parameter to the label, which is 80% of the separation available above chance. The gap of 0.03 to the fitted model is statistically real, with a paired DeLong test giving $z ~ = ~ 4 . 8$ , and it is small. The recorded weather cannot distinguish these accidents much more sharply than the physics already does. Most of the residual error belongs to state the database never recorded, chiefly carburetor heat use. One caveat applies to every model. The label is the investigator’s attribution, and investigators consult the same published chart, so part of the separation may measure investigator practice. Section V-E scores the model against routine weather observations at the same airports, which carry no investigator label.

## E. Separation Against Routine Weather

The evaluations above contrast accidents with accidents. To score the model against weather in which nothing happened, we matched 1,033 carburetor icing accidents and 2,190 other accidents to the nearest NOAA Integrated Surface Database station, a median of 20 km away. We drew the routine hours of the accident’s calendar month there, either all of them or those within three hours of the accident’s clock time. Against routine hours the chart probability does not separate the carburetor icing accidents, with an ROC area of 0.43 (0.41, 0.45) on any hour and 0.47 (0.46, 0.49) on matched hours, and it separates the other accidents even less, at 0.40. Accidents of every kind cluster in drier and warmer weather than the month at large, and the chart places 68% of routine hours in the serious zone. Routine hours are therefore not the exposure of the fleet, since flying concentrates in fair weather.

The ratio of the two accident groups removes that effect. The share of carburetor icing accidents in a chart zone relative to the share of routine hours, divided by the same ratio for the other accidents, is 0.54 (0.40, 0.69) in the nil zone, 0.86 (0.63, 1.11) and 0.97 (0.86, 1.09) in the light and moderate zones and 1.20 (1.15, 1.27) in the serious zone, with intervals from 2,000 resamples of the accidents. The shift toward the weather the physics marks is modest, and only the two outer zones differ from one. On the same subset the accident-against-accident ROC area is 0.618 (0.598, 0.638). The other accidents stand in for the exposure of the fleet, which is a case-control design with accident controls. A per-flight-hour level would need activity data by weather, which the general aviation activity survey supplies only by year [55].

## F. Comparison Against Either Layer on Later Years

We compare the hybrid with either layer alone on an occurrence task rather than on severity. The physics term carries almost no severity signal. Adding the carburetor icing and wind terms to a logistic model of the extracted factors moves the ROC area for a serious or fatal outcome only from 0.818 to 0.820. The carburetor icing term bears on whether a power loss occurs, and the paper makes no claim that the physics layer predicts how the forced landing ends.

The task predicts, from information available before the flight, whether an NTSB accident chain carries the carburetor icing factor, alone or together with engine failure. Records before 2015 fit every fitted arm, and the 8,610 records from 2015 onward, with 348 and 335 positives, score them. The arms are (1) the chart physics with no fit, (2) the physics calibrated on the training years, (3) a corpus-only logistic model on month, hour, light, basic weather condition and state, (4) that model with raw temperature and dewpoint, (5) the hybrid, which adds the calibrated physics term to the covariates, and (6) and (7) the hybrid with that term sign inverted or permuted. The hybrid reaches 0.586 on both targets (0.5860 and 0.5856), above the corpus-only arm at 0.553 (paired DeLong $z = 4 . 1 )$ and the calibrated physics at 0.578. The corrupted hybrids fall back to 0.553, so the gain comes from the physics. The raw feature model reaches 0.597, which a paired test does not separate from the hybrid $( z ~ = ~ 1 . 4 .$ $p = 0 . 1 7 )$ , and its lead depends on how recent its training years are. Trained before 2000 it scores 0.555 against 0.554 for the hybrid, while the calibrated physics holds between 0.574 and 0.579 over every window. The time shift costs the raw model 0.057 in ROC area and the calibrated physics 0.035. Brier scores sit at the base rate for every arm, because the icing prevalence was 2.8% in the training years and 4.0% in the test years and no arm anticipates that drift. On accuracy the hybrid does no better than a fit on the raw weather. Only the calibrated physics is stable across time, since the hybrid inherits the drift of its corpus covariates.

## G. Graph-Level Interventions

Section III-G places two kinds of intervention under a single do-operator semantics, and the failure study exercises the physical kind. For the graph-level kind, we take each learned network $\mathit { G _ { c } } ,$ read the parents of each factor as the adjustment set $Z ,$ and evaluate Eq. (9) with a severe outcome as Y . We compare the result with the crude contrast $P ( Y \mid X { = } 1 ) { - } P ( Y \mid X { = } 0 )$ that an analysis without a learned structure would report. The candidate set holds the 39 factors with at least 30 records in the loss of control category and the 27 in the airframe failure category. Of these, 15 and 11 carry a nonempty adjustment set.

Each record enters the category of its first recognized NTSB occurrence code, and neither category matches one CICTT category exactly. Code 250, loss of control in flight, decides 94% of the loss of control category, which also takes abrupt maneuver, uncontrolled altitude deviation and roll over. Code 130, airframe, component or system failure, decides 97% of the airframe failure category, which also takes propeller failure and decompression. Its nearest CICTT category is SCF-NP, the failure of a system other than the powerplant, although CICTT counts propeller failure as SCF-PP. Loss of engine power has its own code and falls in neither category, so neither network supplies an adjustment set to the combination of Section IV-D. No category isolates it, since the legacy grouping merges code 350 with three other occurrence codes. Table IV therefore uses the two categories that a single code dominates and reports the graph-level intervention apart from the failure study.

Table IV gives both columns for the factors with a nonempty adjustment set whose adjustment moves the estimate by at least 0.05. The code repository [56] lists all 26 factors with a nonempty adjustment set, together with the full networks. In the loss of control category, spatial disorientation appears to raise the probability of a severe outcome by 0.396 before adjustment. After conditioning on low visibility and darkness it raises it by 0.264, which removes a third of the apparent effect. The learned parents are the conditions that produce disorientation in the first place, so the crude figure credits disorientation with harm that the weather and the light already caused. In the airframe failure category, structural failure falls from 0.188 to 0.088 once landing gear anomaly and structural overload are adjusted for. Across the 15 and 11 factors with a nonempty adjustment set, the mean absolute shift is 0.038 and 0.028 respectively, with a maximum of 0.133.

Whether the adjustment moves the estimates toward the truth depends on the causal sufficiency that PC assumes and that Section III-B argues fails. We therefore ran FCI [50], which allows latent confounders and selection, on the same presence data with the same tests and without the ordering priors. Its partial ancestral graphs are dominated by bidirected edges, 47 of 71 adjacencies in the loss of control category and 27 of 45 in the airframe failure category. Of the 29 adjustment relations behind the 15 loss of control factors, FCI admits only 5 as possible parents, and 9 of the 11 airframe failure factors are confounded on its reading. FCI leaves the two parents of spatial disorientation oriented away from it, so their direction rests on the ordering priors and the chain precedence. It marks the relation between landing gear anomaly and structural failure as bidirected, so adjusting on it is not licensed. Recomputed on the relations FCI admits, the structural failure effect is 0.160 rather than 0.088 and the disorientation effect stays at its crude 0.396. The E-values [57] on the adjusted effects give the confounder strength that would explain each effect away. They are 2.4 for spatial disorientation and 2.3 for structural failure in the airframe failure category, and 1.5 and 1.8 at the interval limits nearest the null. Control surface anomaly, whose crude contrast of −0.0001 hides an adjusted −0.070, also excludes zero, at 2.1 (1.4). The other two intervals, structural failure in the loss of control category and icing encounter, cross zero with E-values of 1.0 at the limit, which no confounding is needed to explain, so the analysis supports only these three rows. The graphlevel interventions are estimates on a selected population whose direction rests on the ordering priors, while the physical interventions take their direction from the mechanism.

## VI. DISCUSSION AND LIMITATIONS

The main result of this paper is an account of what accident archives can support once a physical model stands behind a factor, and several of the answers are negative. Recorded temperature and dewpoint separate carburetor icing accidents from other accidents only up to a density ratio ceiling of 0.64, and the physics reaches 0.61 without fitting the label. Routine weather hours cannot serve as the control, because flying concentrates in fair weather. On later years the physics adds 0.03 in ROC area to the prediction of occurrence and nothing to severity, and a fit on the raw weather matches the hybrid in accuracy. FCI does not license the adjustment that moves the spatial disorientation effect. The physics layer therefore contributes an ordering that holds across time, a stated reference class for every number and interventions that name a physical cause. Its case rests on interpretability, and the paper claims no gain in predictive accuracy. The calibrated numbers refer to a reportable accident, so the reporting bias of the archive is made explicit and confined to the level. The guard verdicts, the rankings across operating points and the ratios between scenarios do not depend on it.

The results share one pattern, in which each layer is informative where the other is silent. The corpus pins down consequences, as the carburetor icing node shows with a sharp edge into engine failure and sparse, weak antecedents. The same division of labor underlies hybrid causal logic and the causal model for air transport safety [31, 33]. Here the structure comes from report text instead of expert elicitation, so it can be rebuilt as the archive grows.

Four limitations bound these claims. First, the corpus contains only flights that ended in a reportable event, so every calibrated probability and every data-side intervention is conditional on an occurrence, and the selection induces the collider effect of Section III-B. The carburetor icing evaluation contrasts accidents with accidents under an investigator label that draws on the same published chart, so part of the measured separation may be investigator practice. A per-flight risk needs activity data by weather, with flight hours from the general aviation activity survey as the denominator [55].

TABLE IV  
GRAPH-LEVEL INTERVENTIONS FOR THE OCCURRENCE CATEGORIES OF LOSS OF CONTROL IN FLIGHT AND AIRFRAME FAILURE. EACH ROW GIVES THE CHANGE IN THE PROBABILITY OF A SERIOUS OR FATAL OUTCOME WHEN THE FACTOR IS MADE PRESENT RATHER THAN ABSENT. ROWS ARE THE FACTORS WHOSE ADJUSTMENT MOVES THE ESTIMATE BY AT LEAST 0.05. THE CRUDE COLUMN IS THE UNADJUSTED CONTRAST. THE ADJUSTED COLUMN APPLIES EQ. (9) OVER THE PARENTS OF THE FACTOR IN THE LEARNED NETWORK, WITH 95% BOOTSTRAP PERCENTILES OVER 400 RESAMPLES. THE FCI COLUMN STATES HOW FCI MARKS EACH RELATION OF THE ADJUSTMENT SET IN ORDER. THE MARKS ARE A POSSIBLE PARENT (POSS), A BIDIRECTED EDGE (BI) OR AN EDGE ORIENTED AWAY FROM THE FACTOR (REV), AND NO RELATION IN THESE ROWS IS A DEFINITE PARENT. THE E-VALUE IS THE CONFOUNDER STRENGTH THAT WOULD EXPLAIN THE ADJUSTED EFFECT AWAY, WITH THE VALUE AT THE INTERVAL LIMIT NEAREST THE NULL IN PARENTHESES.
<table><tr><td>Factor</td><td>n</td><td>Crude</td><td>Adjusted (95% CI)</td><td>Adjustment set</td><td>FCI</td><td>E-value</td></tr><tr><td colspan="7">Loss of control in flight, 3,852 records, baseline  $P ( Y ) = 0 . 5 1 8$ </td></tr><tr><td>Spatial disorientation</td><td>111</td><td>0.396</td><td>0.264 (0.058, 0.415)</td><td>low visibility, dark night</td><td>rev, rev</td><td>2.4 (1.5)</td></tr><tr><td>Structural failure</td><td>88</td><td>0.004</td><td>-0.088 (−0.220, 0.043)</td><td>stall, latent defect, overload</td><td>bi, bi, bi</td><td>1.7 (1.0)</td></tr><tr><td>Icing encounter</td><td>45</td><td>0.165</td><td>0.096 (−0.060, 0.227)</td><td>low visibility</td><td>bi</td><td>1.7 (1.0)</td></tr><tr><td colspan="7">Airframe failure, 1,842 records, baseline  $P ( Y ) = 0 . 2 1 3$ </td></tr><tr><td>Structural failure</td><td>454</td><td>0.188</td><td>0.088 (0.045, 0.126)</td><td>gear anomaly, overload</td><td>bi, poss</td><td>2.3 (1.8)</td></tr><tr><td>Control surface anomaly</td><td>387</td><td>-0.0001</td><td>-0.070 (−0.116, -0.017)</td><td>other system failure, gear anomaly</td><td>bi, bi</td><td>2.1 (1.4)</td></tr></table>

Second, the fitted leaky noisy-OR carries the causes its set omits in a leak of 8%, and where an upstream cause acts through a parent already in the set, the per-factor decomposition attributes to that mediator what the upstream cause produced. A chain-level activation model would attribute it upstream, at the cost of the one-level decomposition.

Third, the physical models carry idealizations of their own. The carburetor icing constants come from a published chart, and the level is set by the calibration of Section V-C. The calibrated link is nearly flat across the moderate and serious zones, so it separates little among conducive weathers, and the fuel term is uncalibrated.

Fourth, the same model family performs the extraction, supplies the discovery priors and judges the extraction. The check on 157 full investigation reports varies the extractor, which limits the practical reach of that circularity without removing it. Most of its accidents are also in the corpus, 104 of 151, so it tests a second document type rather than held-out accidents. A blind check by one annotator on 50 reports agrees with the judge on 96.6% of findings and finds no leniency.

## VII. CONCLUSION

This paper presented a hybrid framework in which largescale text mining identifies the factors that lead to a failure mode and physical models quantify how likely each failure is. The two parts assemble into a single explainable failure mode risk. One do-operator treatment covers both graph-level backdoor adjustment and mechanistic physical intervention.

We developed engine power loss end to end, with carburetor icing as the worked physics factor, and computed a counterfactual on named accidents through the same models.

The carburetor icing model sits near the separation limit of the recorded weather. On later years the hybrid gains a small, physics-driven margin over the corpus and matches a fit on the raw weather. Only the physics term keeps its ordering across time, since the hybrid inherits the drift of its corpus covariates. Within the limits stated in Section VI, the approach applies to any failure mode whose contributing factors correspond to physical drivers, and it lifts such a mode from one frequency to a risk at the operating point with its reference class stated.

## ACKNOWLEDGMENT

The authors acknowledge the University of Toronto for its support in carrying out this research. Large language models are a component of the method and an object of study. A locally hosted Qwen3.6-35B-A3B [43] performs the chain extraction, acts as the judge and supplies the ordering priors. It is served by vLLM 0.19.1 with constrained JSON output and no reasoning trace, at temperature 0.2 for extraction and 0 for judging. Claude Opus 5, Claude Sonnet 5 and Claude Haiku 4.5 (identifiers claude-opus-5, claude-sonnet-5 and claude-haiku-4-5, accessed on 9 and 10 August 2026) act as alternative extractors in Section V-A. The code, prompts, extracted chains, networks, judge outputs and the scripts behind every table and figure are available at https://github.com/ YutianPangASU/AviationSafetyReportKnowledgeGraph [56], and the NTSB, FAA, TSB and BEA source records are public.

## REFERENCES

[1] J. Pearl, Causality: Models, Reasoning, and Inference, 2nd ed. Cambridge, UK: Cambridge University Press, 2009, doi: 10.1017/CBO9780511803161.

[2] S. R. Andrade and H. S. Walsh, “SafeAeroBERT: Towards a safety-informed aerospace-specific language model,” in AIAA AVIATION 2023 Forum, no. AIAA 2023-3437, 2023, doi: 10.2514/6.2023-3437.

[3] X. Jing, A. Chennakesavan, C. Chandra, M. V. Bendarkar, M. Kirby, and D. N. Mavris, “BERT for aviation text classification,” in AIAA AVIATION 2023 Forum, no. AIAA 2023-3438, 2023, doi: 10.2514/6.2023-3438.

[4] Q. Liu, F. Li, K. K. H. Ng, J. Han, and S. Feng, “Accident investigation via LLMs reasoning: HFACS-guided chainof-thoughts enhance general aviation safety,” Expert Systems with Applications, vol. 269, p. 126422, 2025, doi: 10.1016/j.eswa.2025.126422.

[5] X. Zhao, H. Yan, and Y. Liu, “Hierarchical multilabel classification for fine-level event extraction from aviation accident reports,” INFORMS Journal on Data Science, vol. 4, no. 1, pp. 51–66, 2025, doi: 10.1287/ijds.2022. 0032.

[6] V. Siddeshwar, A. Azim, S. Alwidian, and M. Makrehchi, “Towards enhancing aviation safety through advanced incident analysis using large language models,” in 2024 34th International Conference on Collaborative Advances in Software and COmputiNg (CASCON), 2024, pp. 1–7, doi: 10.1109/CASCON62161.2024.10837802.

[7] L. Chen, J. Xu, T. Wu, and J. Liu, “Information extraction of aviation accident causation knowledge graph: An LLM-based approach,” Electronics, vol. 13, no. 19, p. 3936, 2024, doi: 10.3390/electronics13193936.

[8] X. Zhang and S. Mahadevan, “Bayesian network modeling of accident investigation reports for aviation safety assessment,” Reliability Engineering & System Safety, vol. 209, p. 107371, 2021, doi: 10.1016/j.ress.2020. 107371.

[9] SAE International, “Guidelines for conducting the safety assessment process on civil aircraft, systems, and equipment,” SAE Standard ARP4761A, Recommended Practice, revised December 2023, issued December 1996, 2023, doi: 10.4271/ARP4761A.

[10] K. G. Pratt and W. G. Walker, “A revised gust-load formula and a re-evaluation of V-G data taken on civil transport airplanes from 1933 to 1950,” National Advisory Committee for Aeronautics, Tech. Rep. 1206, 1954.

[11] J. Berndt, “JSBSim: An open source flight dynamics model in C++,” in AIAA Modeling and Simulation Technologies Conference and Exhibit, no. AIAA 2004-4923, 2004, doi: 10.2514/6.2004-4923.

[12] C. Chandra, X. Jing, M. V. Bendarkar, K. Sawant, L. Elias, M. Kirby, and D. N. Mavris, “Aviation-BERT: A preliminary aviation-specific natural language model,” in AIAA AVIATION 2023 Forum, no. AIAA 2023-3436, 2023, doi: 10.2514/6.2023-3436.

[13] Y. Yan, B. Li, and G. Lodewijks, “UAV accident forensics via HFACS-LLM reasoning: Low-altitude safety insights,” Drones, vol. 9, no. 10, p. 704, 2025, doi: 10.3390/drones9100704.

[14] S. Farzadnia, R. Merkert, and M. J. Beck, “Analysing aviation safety reports for uncovering current risks and policy gaps using topic modelling and knowledge graphs,” Safety Science, vol. 200, p. 107230, 2026, doi: 10.1016/ j.ssci.2026.107230.

[15] R. Gansch, L. Putze, T. Koopmann, J. Reich, and C. Neurohr, “Causal Bayesian networks for data-driven safety analysis of complex systems,” in Model-Based Safety and Assessment: 9th International Symposium, IMBSA 2025, Athens, Greece, September 24–26, 2025, Proceedings, ser. Lecture Notes in Computer Science, P. Katsaros, Ed. Cham: Springer Nature Switzerland, 2026, vol. 15755, pp. 222–237, doi: 10.1007/978-3-032-05073-1\_15.

[16] V. Karwa, A. B. Slavkovic, and E. T. Donnell, “Causal´ inference in transportation safety studies: Comparison of potential outcomes and causal diagrams,” The Annals of Applied Statistics, vol. 5, no. 2B, pp. 1428–1455, 2011, doi: 10.1214/10-AOAS440.

[17] A. Ruiz-Tagle, E. Lopez-Droguett, and K. M. Groth, “A novel probabilistic approach to counterfactual reasoning in system safety,” Reliability Engineering & System

Safety, vol. 228, p. 108785, 2022, doi: 10.1016/j.ress. 2022.108785.

[18] E. Zibaei and R. Borth, “Building causal models for finding actual causes of unmanned aerial vehicle failures,” Frontiers in Robotics and AI, vol. 11, p. 1123762, 2024, doi: 10.3389/frobt.2024.1123762.

[19] J. M. Scanlon, K. D. Kusano, T. Daniel, C. Alderson, A. Ogle, and T. Victor, “Waymo simulated driving behavior in reconstructed fatal crashes within an autonomous vehicle operating domain,” Accident Analysis & Prevention, vol. 163, p. 106454, 2021, doi: 10.1016/j.aap.2021. 106454.

[20] E. Kiciman, R. Ness, A. Sharma, and C. Tan, “Causal reasoning and large language models: Opening a new frontier for causality,” Transactions on Machine Learning Research, Aug. 2024.

[21] Z. Jin, Y. Chen, F. Leeb, L. Gresele, O. Kamal, Z. Lyu, K. Blin, F. Gonzalez Adauto, M. Kleiman-Weiner, M. Sachan, and B. Schölkopf, “CLadder: Assessing causal reasoning in language models,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 31 038–31 065, doi: 10. 52202/075280-1353.

[22] A. Vashishtha, A. G. Reddy, A. Kumar, S. Bachu, V. N. Balasubramanian, and A. Sharma, “Causal order: The key to leveraging imperfect experts in causal inference,” in The Thirteenth International Conference on Learning Representations, 2025.

[23] X. Zhang, S. Liu, Y. Wang, and Q. Mei, “Leveraging LLM-generated structural prior for causal inference with concurrent causes,” in Causality and Large Models @NeurIPS 2024, 2024.

[24] U.S. Department of Defense, “Flying qualities of piloted airplanes,” Military Specification, Tech. Rep. MIL-F-8785C, 1980.

[25] F. M. Hoblit, Gust Loads on Aircraft: Concepts and Applications, ser. AIAA Education Series. Washington, DC: American Institute of Aeronautics and Astronautics, 1988, doi: 10.2514/4.861888.

[26] J. Sun, J. M. Hoekstra, and J. Ellerbroek, “OpenAP: An open-source aircraft performance model for air transportation studies and simulations,” Aerospace, vol. 7, no. 8, p. 104, 2020, doi: 10.3390/aerospace7080104.

[27] A. Nuic, “User manual for the base of aircraft data (BADA) revision 3.8,” EUROCONTROL Experimental Centre, Tech. Rep. EEC Technical/Scientific Report No. 2010-003, Apr. 2010.

[28] Federal Aviation Administration, “Carburetor icing prevention,” FAA Special Airworthiness Information Bulletin, Tech. Rep. CE-09-35, 2009.

[29] C. J. Tomlin, I. Mitchell, A. M. Bayen, and M. Oishi, “Computational techniques for the verification of hybrid systems,” Proceedings of the IEEE, vol. 91, no. 7, pp. 986–1001, 2003, doi: 10.1109/JPROC.2003.814621.

[30] X. Yang, O. A. Beg, M. Kenigsberg, and T. T. Johnson, “A framework for identification and validation of affine hybrid automata from input-output traces,” ACM Transactions on Cyber-Physical Systems, vol. 6, no. 2,

pp. 1–24, 2022, doi: 10.1145/3470455.

[31] K. Groth, C. Wang, and A. Mosleh, “Hybrid causal methodology and software platform for probabilistic risk assessment and safety monitoring of socio-technical systems,” Reliability Engineering & System Safety, vol. 95, no. 12, pp. 1276–1285, 2010, doi: 10.1016/j.ress.2010. 06.005.

[32] B. J. M. Ale, L. J. Bellamy, R. M. Cooke, L. H. J. Goossens, A. R. Hale, A. L. C. Roelen, and E. Smith, “Towards a causal model for air transport safety—an ongoing research project,” Safety Science, vol. 44, no. 8, pp. 657–673, 2006, doi: 10.1016/j.ssci.2006.02.002.

[33] B. J. M. Ale, L. J. Bellamy, R. van der Boom, J. Cooper, R. M. Cooke, L. H. J. Goossens, A. R. Hale, D. Kurowicka, O. Morales, A. L. C. Roelen, and J. Spouge, “Further development of a causal model for air transport safety (CATS): Building the mathematical heart,” Reliability Engineering & System Safety, vol. 94, no. 9, pp. 1433–1441, 2009, doi: 10.1016/j.ress.2009.02.024.

[34] B. J. M. Ale, L. J. Bellamy, J. Cooper, D. Ababei, D. Kurowicka, O. Morales, and J. Spouge, “Analysis of the crash of TK 1951 using CATS,” Reliability Engineering & System Safety, vol. 95, no. 5, pp. 469–477, 2010, doi: 10.1016/j.ress.2009.11.014.

[35] T. Aldemir, “A survey of dynamic methodologies for probabilistic safety assessment of nuclear power plants,” Annals of Nuclear Energy, vol. 52, pp. 113–124, 2013, doi: 10.1016/j.anucene.2012.08.001.

[36] H. Meng, X. An, and J. Xing, “A data-driven Bayesian network model integrating physical knowledge for prioritization of risk influencing factors,” Process Safety and Environmental Protection, vol. 160, pp. 434–449, 2022, doi: 10.1016/j.psep.2022.02.010.

[37] J. Xing, J. Qian, R. Peng, and E. Zio, “Physics-informed data-driven Bayesian network for the risk analysis of hydrogen refueling stations,” International Journal of Hydrogen Energy, vol. 110, pp. 371–385, 2025, doi: 10.1016/j.ijhydene.2025.02.110.

[38] National Transportation Safety Board, “NTSB downloadable aviation datasets,” Public data release, Microsoft Access format, 2026, accessed 2026-07-27.

[39] Federal Aviation Administration, “Accident/incident data system (AIDS),” Aviation Safety Information Analysis and Sharing (ASIAS), 2026, accessed 2026-07-27.

[40] National Transportation Safety Board, “Investigation reports,” Published investigation reports (AAR, AAB, SIR, AIR, ASR), 2026, accessed 2026-07-27.

[41] CAST/ICAO Common Taxonomy Team, “Aviation occurrence categories: Definitions and usage notes,” Commercial Aviation Safety Team / International Civil Aviation Organization, Tech. Rep. v4.7, Dec. 2017.

[42] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[43] Qwen Team, “Qwen3.6-35B-A3B model card,” Hugging Face model repository, https://huggingface.co/Qwen/ Qwen3.6-35B-A3B, revision 53c4317, 2026, released 15 April 2026, downloaded 23 April 2026.

[44] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with PagedAttention,” in Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), 2023, pp. 611– 626, doi: 10.1145/3600006.3613165.

[45] J. Reason, Human Error. Cambridge: Cambridge University Press, 1990, doi: 10.1017/CBO9781139062367.

[46] J. Rasmussen, “Risk management in a dynamic society: a modelling problem,” Safety Science, vol. 27, no. 2–3, pp. 183–213, 1997, doi: 10.1016/S0925-7535(97)00052-0.

[47] S. A. Shappell and D. A. Wiegmann, “The human factors analysis and classification system—HFACS,” Federal Aviation Administration, Office of Aviation Medicine, Tech. Rep. DOT/FAA/AM-00/7, 2000.

[48] CAST/ICAO Common Taxonomy Team, “Phase of flight: Definitions and usage notes,” Commercial Aviation Safety Team / International Civil Aviation Organization, Tech. Rep. v1.3, 2013.

[49] M. A. Hernán, S. Hernández-Díaz, and J. M. Robins, “A structural approach to selection bias,” Epidemiology, vol. 15, no. 5, pp. 615–625, 2004, doi: 10.1097/01.ede. 0000135174.63482.43.

[50] P. Spirtes, C. Glymour, and R. Scheines, Causation, Prediction, and Search, 2nd ed. Cambridge, MA: MIT Press, 2000, doi: 10.7551/mitpress/1754.001.0001.

[51] Y. Zheng, B. Huang, W. Chen, J. Ramsey, M. Gong, R. Cai, S. Shimizu, P. Spirtes, and K. Zhang, “Causallearn: Causal discovery in Python,” Journal of Machine Learning Research, vol. 25, no. 60, pp. 1–8, 2024.

[52] M. Kim, T. Thonet, J. Rozen, H. Lee, K. Jung, and M. Dymetman, “Guaranteed generation from large language models,” in The Thirteenth International Conference on Learning Representations, 2025.

[53] J. Tian and J. Pearl, “Probabilities of causation: Bounds and identification,” Annals of Mathematics and Artificial Intelligence, vol. 28, no. 1–4, pp. 287–313, 2000, doi: 10.1023/A:1018912507879.

[54] O. A. Alduchov and R. E. Eskridge, “Improved Magnus form approximation of saturation vapor pressure,” Journal ofApplied Meteorology, vol. 35, no. 4, pp. 601– 609, 1996, doi: 10.1175/1520-0450(1996)035<0601: IMFAOS>2.0.CO;2.

[55] Federal Aviation Administration, “General aviation and Part 135 activity surveys,” FAA Aviation Data and Statistics, 2026, accessed 2026-09-30.

[56] Y. Pang, Y. Tian, J. Xie, X. Zhao, J. Hu, J.-P. Clarke, and H.-A. Jacobsen, “AviationSafetyReportKnowledgeGraph: Code, extracted chains, learned networks and judge outputs,” Zenodo, doi: 10.5281/zenodo.23478971, release v0.2, commit 198b5ed, mirror https://github.com/ YutianPangASU/AviationSafetyReportKnowledgeGraph, 2026.

[57] T. J. VanderWeele and P. Ding, “Sensitivity analysis in observational research: Introducing the E-value,” Annals of Internal Medicine, vol. 167, no. 4, pp. 268–274, 2017, doi: 10.7326/M16-2607.